#!/usr/bin/env python3
"""Build a wider yaw alias / frame drift support manifest.

This expands the original episode-level acceptance manifest in two ways:

* keep the stable-alias calibration positives from the sequence reports
* add episode summaries synthesized from larger yaw/frame diagnostic row files

The output is still offline-only.  It is meant to feed a two-stage pipeline:
first classify stable alias vs frame drift, then regress dyaw only on the
stable-alias subset and abstain on drift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classify_c2c_v2_yaw_alias_vs_drift import _classify
from scripts.build_c2c_v2_yaw_alias_drift_manifest import build_alias_drift_manifest


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _expand_inputs(paths: list[Path], *, patterns: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            for pattern in patterns:
                out.extend(sorted(path.rglob(pattern)))
        elif path.is_file():
            out.append(path)
    dedup: list[Path] = []
    seen: set[str] = set()
    for path in out:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            dedup.append(path)
    return dedup


def _wrap_yaw_to_symmetry(yaw: float, period: float = float(np.pi / 2.0)) -> float:
    if not (np.isfinite(yaw) and np.isfinite(period) and period > 0.0):
        return float("nan")
    half = 0.5 * float(period)
    return float(((float(yaw) + half) % float(period)) - half)


def _symmetry_aware_yaw(raw_yaw: float) -> float:
    if not np.isfinite(raw_yaw):
        return float("nan")
    return float(-_wrap_yaw_to_symmetry(raw_yaw))


def _step_diff(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(((float(a) - float(b) + math.pi) % (2.0 * math.pi)) - math.pi)


def _group_by_episode(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("episode_idx", -1))].append(row)
    return grouped


def _episode_summary_from_diagnostic_rows(rows: list[dict[str, Any]], *, source_path: Path) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: int(r.get("step_idx", r.get("step", -1))))
    proxy = np.asarray([_safe_float(r.get("proxy_yaw", r.get("image_axis_yaw", 0.0)), float("nan")) for r in ordered], dtype=np.float64)
    priv = np.asarray([_safe_float(r.get("privileged_yaw", float("nan")), float("nan")) for r in ordered], dtype=np.float64)
    mask = np.isfinite(proxy) & np.isfinite(priv)
    if not np.any(mask):
        proxy = np.zeros((0,), dtype=np.float64)
        priv = np.zeros((0,), dtype=np.float64)
    else:
        proxy = proxy[mask]
        priv = priv[mask]
    symm = np.asarray([_symmetry_aware_yaw(float(v)) for v in proxy], dtype=np.float64)
    bias = float(np.mean(symm - priv)) if priv.size else 0.0
    bias_corrected = symm - bias
    raw_mae = float(np.mean(np.abs(proxy - priv))) if priv.size else 0.0
    symm_mae = float(np.mean(np.abs(symm - priv))) if priv.size else 0.0
    bias_corrected_mae = float(np.mean(np.abs(bias_corrected - priv))) if priv.size else 0.0
    jump_points = 0
    for i in range(1, int(proxy.size)):
        if abs(_step_diff(float(proxy[i]), float(proxy[i - 1]))) >= 0.40:
            jump_points += 1
    report = {
        "episode_idx": int(ordered[0].get("episode_idx", -1)) if ordered else -1,
        "failure_bucket": str(ordered[0].get("failure_bucket", "")) if ordered else "",
        "primary_blocker": str(ordered[0].get("yaw_observability_primary_blocker", "")) if ordered else "",
        "num_rows": int(len(ordered)),
        "raw_proxy_mae": float(raw_mae),
        "symmetry_aware_mae": float(symm_mae),
        "bias_corrected_mae": float(bias_corrected_mae),
        "num_jump_points": int(jump_points),
        "gif_path": None,
        "jump_sheet_path": None,
        "report_path": str(source_path.resolve()),
        "source_kind": "diagnostic_rows",
    }
    cls = _classify(report)
    if cls["label"] == "stable_alias":
        acceptance_role = "calibration_positive"
    elif cls["label"] == "frame_drift":
        acceptance_role = "frame_drift_hard_case"
    else:
        acceptance_role = "mixed_or_unclear"
    return {
        **report,
        **cls,
        "acceptance_role": acceptance_role,
        "alias_label": str(cls["label"]),
        "rows": int(len(ordered)),
        "selected_step_idxs": [int(r.get("step_idx", r.get("step", -1))) for r in ordered],
        "source_row_count": int(len(ordered)),
        "source_relabel_jsonl": _common_relabel_path(ordered),
        "source_paths": [str(source_path.resolve())],
    }


def _common_relabel_path(rows: list[dict[str, Any]]) -> str:
    candidates = {str(row.get("source_relabel_jsonl", "")) for row in rows if str(row.get("source_relabel_jsonl", ""))}
    if len(candidates) == 1:
        return next(iter(candidates))
    return ""


def _sequence_report_rows(paths: list[Path]) -> list[dict[str, Any]]:
    sequence_rows: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["report_path"] = str(path.resolve())
        cls = _classify(report)
        if cls["label"] == "stable_alias":
            acceptance_role = "calibration_positive"
        elif cls["label"] == "frame_drift":
            acceptance_role = "frame_drift_hard_case"
        else:
            acceptance_role = "mixed_or_unclear"
        sequence_rows.append(
            {
                "episode_idx": int(report.get("episode_idx", -1)),
                "failure_bucket": str(report.get("failure_bucket", "")),
                "primary_blocker": str(report.get("primary_blocker", "")),
                "num_rows": int(report.get("num_rows", 0)),
                **cls,
                "acceptance_role": acceptance_role,
                "alias_label": str(cls["label"]),
                "gif_path": report.get("gif_path"),
                "jump_sheet_path": report.get("jump_sheet_path"),
                "report_path": str(path.resolve()),
                "source_kind": "sequence_report",
                "selected_step_idxs": [int(v) for v in report.get("selected_step_idxs", []) if isinstance(v, (int, float, np.integer, np.floating))],
                "source_row_count": int(report.get("num_rows", 0)),
                "source_relabel_jsonl": str(report.get("source_relabel_jsonl", "")),
                "source_paths": [str(path.resolve())],
            }
        )
    return sequence_rows


def _merge_support_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    role_priority = {"calibration_positive": 3, "frame_drift_hard_case": 2, "mixed_or_unclear": 1}
    label_priority = {"stable_alias": 3, "frame_drift": 2, "mixed_or_unclear": 1}
    selected_by_episode: dict[int, dict[str, Any]] = {}
    sources_by_episode: dict[int, set[str]] = defaultdict(set)

    for row in rows:
        ep = int(row.get("episode_idx", -1))
        sources_by_episode[ep].update(row.get("source_paths", []))
        role = str(row.get("acceptance_role", row.get("acceptance_role", "mixed_or_unclear")))
        label = str(row.get("alias_label", row.get("label", "mixed_or_unclear")))
        priority = (
            int(label_priority.get(label, 0)),
            int(role_priority.get(role, 0)),
            int(row.get("rows", row.get("num_rows", 0))),
            1 if str(row.get("source_kind", "")) == "sequence_report" else 0,
        )
        existing = selected_by_episode.get(ep)
        if existing is None:
            selected_by_episode[ep] = dict(row, support_priority=list(priority))
            continue
        existing_priority = tuple(existing.get("support_priority", [0, 0, 0, 0]))
        if priority > existing_priority:
            selected_by_episode[ep] = dict(row, support_priority=list(priority))

    merged: list[dict[str, Any]] = []
    for ep, row in sorted(selected_by_episode.items()):
        row = dict(row)
        row["source_paths"] = sorted(sources_by_episode.get(ep, set(row.get("source_paths", []))))
        row["source_path"] = row["source_paths"][0] if row["source_paths"] else str(row.get("report_path", ""))
        row["support_priority"] = list(row.get("support_priority", []))
        row["selection_reason"] = str(row.get("reason", row.get("acceptance_reason", "")))
        merged.append(row)
    return merged


def build_support_manifest(
    *,
    sequence_reports: list[Path] | None = None,
    diagnostic_jsonl: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence_reports = sequence_reports or []
    diagnostic_jsonl = diagnostic_jsonl or []

    seq_paths = _expand_inputs(sequence_reports, patterns=("yaw_frame_sequence_report.json",))
    diag_paths = _expand_inputs(diagnostic_jsonl, patterns=("yaw_frame_diagnostic_rows.jsonl",))

    sequence_rows = _sequence_report_rows(seq_paths)
    diagnostic_rows: list[dict[str, Any]] = []
    for path in diag_paths:
        rows = _read_jsonl(path)
        for ep, subset in sorted(_group_by_episode(rows).items()):
            if not subset:
                continue
            diagnostic_rows.append(_episode_summary_from_diagnostic_rows(subset, source_path=path))

    all_rows = sequence_rows + diagnostic_rows
    merged = _merge_support_rows(all_rows)

    output_rows: list[dict[str, Any]] = []
    for row in merged:
        role = str(row.get("acceptance_role", "mixed_or_unclear"))
        label = str(row.get("alias_label", "mixed_or_unclear"))
        output_rows.append(
            {
                "schema_version": "yaw_alias_drift_support_manifest_v1",
                "episode_idx": int(row.get("episode_idx", -1)),
                "failure_bucket": str(row.get("failure_bucket", "")),
                "primary_blocker": str(row.get("primary_blocker", "")),
                "rows": int(row.get("rows", row.get("num_rows", 0))),
                "num_rows": int(row.get("rows", row.get("num_rows", 0))),
                "selected_step_idxs": [int(v) for v in row.get("selected_step_idxs", [])],
                "selected_step_count": int(len(row.get("selected_step_idxs", []))),
                "acceptance_role": role,
                "alias_label": label,
                "support_role": role,
                "support_label": label,
                "acceptance_reason": str(row.get("acceptance_reason", row.get("reason", ""))),
                "classification_source": str(row.get("source_kind", "")),
                "source_kind": str(row.get("source_kind", "")),
                "source_path": str(row.get("source_path", row.get("report_path", ""))),
                "report_path": str(row.get("report_path", "")),
                "source_paths": list(row.get("source_paths", [])),
                "source_relabel_jsonl": str(row.get("source_relabel_jsonl", "")),
                "source_row_count": int(row.get("source_row_count", row.get("rows", row.get("num_rows", 0)))),
                "raw_mae": float(row.get("raw_proxy_mae", row.get("raw_mae", 0.0))),
                "symmetry_aware_mae": float(row.get("symmetry_aware_mae", 0.0)),
                "bias_corrected_mae": float(row.get("bias_corrected_mae", 0.0)),
                "jump_points": int(row.get("num_jump_points", row.get("jump_points", 0))),
                "support_priority": list(row.get("support_priority", [])),
            }
        )

    output_rows.sort(key=lambda r: (str(r["acceptance_role"]), int(r["episode_idx"])))
    summary = {
        "schema_version": "yaw_alias_drift_support_manifest_v1",
        "sequence_report_count": int(len(seq_paths)),
        "diagnostic_jsonl_count": int(len(diag_paths)),
        "num_rows": int(len(output_rows)),
        "by_acceptance_role": dict(Counter(str(r["acceptance_role"]) for r in output_rows)),
        "by_alias_label": dict(Counter(str(r["alias_label"]) for r in output_rows)),
        "by_source_kind": dict(Counter(str(r["source_kind"]) for r in output_rows)),
        "by_episode": {f"ep{int(row['episode_idx']):03d}": 1 for row in output_rows},
        "report_paths": [str(path.resolve()) for path in seq_paths],
        "diagnostic_paths": [str(path.resolve()) for path in diag_paths],
    }
    return output_rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a wider yaw alias / frame drift support manifest.")
    ap.add_argument("--sequence_reports", type=Path, nargs="*", default=[])
    ap.add_argument("--diagnostic_jsonl", type=Path, nargs="*", default=[])
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/yaw_alias_drift_support_manifest.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/yaw_alias_drift_support_manifest.summary.json"),
    )
    args = ap.parse_args()

    sequence_reports = list(args.sequence_reports)
    diagnostic_jsonl = list(args.diagnostic_jsonl)
    if not sequence_reports:
        sequence_reports = [Path("runtime_artifacts/coarse2contact_v2/reports/yaw_frame_alignment_diagnostic")]
    if not diagnostic_jsonl:
        diagnostic_jsonl = [Path("runtime_artifacts/coarse2contact_v2/reports")]

    rows, summary = build_support_manifest(sequence_reports=sequence_reports, diagnostic_jsonl=diagnostic_jsonl)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary["output_jsonl"] = str(args.output_jsonl.resolve())
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output_jsonl)
    print(args.summary_json)


if __name__ == "__main__":
    main()
