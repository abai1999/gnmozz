#!/usr/bin/env python3
"""Build a row-level yaw alias / frame drift support manifest.

The episode-level support manifest is useful, but it can still leave the
positive side too sparse when only a couple of sequences are fully classified
as stable alias.  This helper widens the support surface by working at the
diagnostic row / window level:

* rows whose symmetry-aware alias is tight enough become calibration positives
* rows whose proxy still drifts after alias resolution become hard cases

The output is still offline-only and keeps the same manifest shape expected by
the two-stage baseline.  The only difference is that one episode can now
contribute many small support rows instead of collapsing into a single
episode-level slice.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


POSITIVE_DIAGNOSIS_LABELS = {"symmetry_alias_candidate", "sign_flip_candidate"}
NEGATIVE_DIAGNOSIS_LABELS = {
    "frame_definition_drift_candidate",
    "occlusion_blocks_frame_axis",
    "weak_frame_observability",
    "no_proxy_signal",
    "symmetry_wrapping_mismatch",
}


def _alias_drift_decision(*, acceptance_role: str, alias_label: str) -> str:
    if acceptance_role == "calibration_positive" or alias_label == "stable_alias":
        return "stable_alias_control"
    if acceptance_role == "frame_drift_hard_case" or alias_label == "frame_drift":
        return "frame_drift_abstain"
    return "unknown"


def _text_or_default(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    text = str(value).strip()
    if not text or text == "None":
        return str(default)
    return text


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
                row = json.loads(line)
                row.setdefault("source_relabel_jsonl", str(path))
                rows.append(row)
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


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def _row_skill_name(row: Mapping[str, Any]) -> str:
    return str(row.get("skill_name", row.get("skill_type", "")))


def _skill_matches(row: Mapping[str, Any], skill_type: str) -> bool:
    if not skill_type:
        return True
    row_skill = _row_skill_name(row)
    return bool(
        row_skill == skill_type
        or row_skill.startswith(skill_type)
        or skill_type.startswith(row_skill)
    )


def _classify_row(row: Mapping[str, Any], *, stable_alias_max_abs_error: float) -> tuple[str, str, str]:
    diagnosis = str(row.get("diagnosis_label", ""))
    best_alias_abs_error = _safe_float(row.get("best_symmetry_alias_abs_error"), float("inf"))
    proxy_abs_error = _safe_float(row.get("proxy_privileged_abs_error"), float("inf"))
    proxy_improvement = best_alias_abs_error + 1.0e-12 < proxy_abs_error

    if diagnosis in POSITIVE_DIAGNOSIS_LABELS and proxy_improvement and best_alias_abs_error <= float(stable_alias_max_abs_error):
        return (
            "calibration_positive",
            "stable_alias",
            f"{diagnosis}; best_alias_abs_error={best_alias_abs_error:.6f}; proxy_abs_error={proxy_abs_error:.6f}",
        )
    if diagnosis in NEGATIVE_DIAGNOSIS_LABELS or best_alias_abs_error > float(stable_alias_max_abs_error) or not proxy_improvement:
        return (
            "frame_drift_hard_case",
            "frame_drift",
            f"{diagnosis or 'unclassified'}; best_alias_abs_error={best_alias_abs_error:.6f}; proxy_abs_error={proxy_abs_error:.6f}",
        )
    return (
        "mixed_or_unclear",
        "mixed_or_unclear",
        f"{diagnosis or 'unclassified'}; best_alias_abs_error={best_alias_abs_error:.6f}; proxy_abs_error={proxy_abs_error:.6f}",
    )


def _support_row(
    row: Mapping[str, Any],
    *,
    stable_alias_max_abs_error: float,
    source_path: Path,
) -> dict[str, Any]:
    acceptance_role, alias_label, reason = _classify_row(row, stable_alias_max_abs_error=stable_alias_max_abs_error)
    step_idx = _safe_int(row.get("step_idx", row.get("step", -1)))
    selected_step_idxs = [int(step_idx)] if step_idx >= 0 else []
    return {
        "schema_version": "yaw_alias_drift_support_manifest_rows_v1",
        "episode_idx": _safe_int(row.get("episode_idx", -1)),
        "step_idx": step_idx,
        "failure_bucket": str(row.get("failure_bucket", "")),
        "skill_name": _row_skill_name(row),
        "skill_type": str(row.get("skill_type", "precision_grasp")) if str(row.get("skill_type", "")) else "precision_grasp",
        "primary_blocker": str(row.get("yaw_observability_primary_blocker", "")),
        "rows": 1,
        "num_rows": 1,
        "selected_step_idxs": selected_step_idxs,
        "selected_step_count": int(len(selected_step_idxs)),
        "acceptance_role": acceptance_role,
        "alias_label": alias_label,
        "alias_drift_decision": _text_or_default(
            row.get("alias_drift_decision", None),
            _alias_drift_decision(acceptance_role=acceptance_role, alias_label=alias_label),
        ),
        "support_role": acceptance_role,
        "support_label": alias_label,
        "acceptance_reason": reason,
        "classification_source": "diagnostic_row",
        "source_kind": "diagnostic_row",
        "source_path": str(source_path.resolve()),
        "report_path": str(source_path.resolve()),
        "source_paths": [str(source_path.resolve())],
        "source_relabel_jsonl": str(row.get("source_relabel_jsonl", source_path)),
        "source_row_count": 1,
        "diagnosis_label": str(row.get("diagnosis_label", "")),
        "best_symmetry_alias_k": _safe_int(row.get("best_symmetry_alias_k", 0)),
        "best_symmetry_alias_yaw": _safe_float(row.get("best_symmetry_alias_yaw"), float("nan")),
        "best_symmetry_alias_abs_error": _safe_float(row.get("best_symmetry_alias_abs_error"), float("nan")),
        "proxy_privileged_abs_error": _safe_float(row.get("proxy_privileged_abs_error"), float("nan")),
        "proxy_privileged_sign_match": row.get("proxy_privileged_sign_match", None),
        "proxy_yaw": _safe_float(row.get("proxy_yaw"), float("nan")),
        "privileged_yaw": _safe_float(row.get("privileged_yaw"), float("nan")),
        "raw_pose_dyaw": _safe_float(row.get("raw_pose_dyaw"), float("nan")),
        "symmetry_period": _safe_float(row.get("symmetry_period"), float("nan")),
        "visual_observability_class": str(row.get("visual_observability_class", "")),
        "yaw_observability_class": str(row.get("yaw_observability_class", "")),
        "frame_confidence": _safe_float(row.get("frame_confidence"), float("nan")),
        "frame_observability": _safe_float(row.get("frame_observability"), float("nan")),
        "frame_axis_strength": _safe_float(row.get("frame_axis_strength"), float("nan")),
        "wrist_occluded": bool(row.get("wrist_occluded", row.get("yaw_observability_wrist_occluded", False))),
        "xy_error": _safe_float(row.get("xy_error"), float("nan")),
        "near_basin_shell": bool(row.get("near_basin_shell", False)),
        "micro_entry_ready": bool(row.get("micro_entry_ready", False)),
        "support_priority": [
            1 if acceptance_role == "frame_drift_hard_case" else 0,
            1 if alias_label == "stable_alias" else 0,
            1 if str(row.get("failure_bucket", "")) in {"large_xy_large_yaw", "large_xy_small_yaw", "small_xy_large_yaw"} else 0,
            1 if str(row.get("diagnosis_label", "")) in POSITIVE_DIAGNOSIS_LABELS else 0,
            1 if str(row.get("diagnosis_label", "")) in NEGATIVE_DIAGNOSIS_LABELS else 0,
            step_idx,
        ],
        "selection_reason": reason,
    }


def build_row_support_manifest(
    rows: list[dict[str, Any]],
    *,
    stable_alias_max_abs_error: float = 0.02,
    stage_name: str = "RING_GRASP_ALIGN",
    skill_type: str = "precision_grasp",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if stage_name and str(row.get("stage_name", "")) != stage_name:
            continue
        if skill_type and not _skill_matches(row, skill_type):
            continue
        filtered.append(dict(row))

    selected = [_support_row(row, stable_alias_max_abs_error=float(stable_alias_max_abs_error), source_path=Path(str(row.get("source_relabel_jsonl", "")) or ".")) for row in filtered]

    selected.sort(
        key=lambda r: (
            str(r["acceptance_role"]),
            int(r["episode_idx"]),
            int(r["step_idx"]),
            str(r["failure_bucket"]),
        )
    )

    summary = {
        "schema_version": "yaw_alias_drift_support_manifest_rows_v1",
        "input_rows": int(len(rows)),
        "filtered_rows": int(len(filtered)),
        "selected_rows": int(len(selected)),
        "stable_alias_max_abs_error": float(stable_alias_max_abs_error),
        "by_acceptance_role": _count_by(selected, "acceptance_role"),
        "by_alias_label": _count_by(selected, "alias_label"),
        "by_alias_drift_decision": _count_by(selected, "alias_drift_decision"),
        "by_diagnosis_label": _count_by(selected, "diagnosis_label"),
        "by_failure_bucket": _count_by(selected, "failure_bucket"),
        "by_visual_observability": _count_by(selected, "visual_observability_class"),
        "by_yaw_observability": _count_by(selected, "yaw_observability_class"),
        "by_episode": {f"ep{int(ep):03d}": int(count) for ep, count in sorted(Counter(int(row["episode_idx"]) for row in selected).items())},
        "positive_rows": int(sum(1 for row in selected if str(row.get("acceptance_role", "")) == "calibration_positive")),
        "hard_case_rows": int(sum(1 for row in selected if str(row.get("acceptance_role", "")) == "frame_drift_hard_case")),
    }
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a row-level yaw alias / frame drift support manifest.")
    ap.add_argument("--diagnostic_jsonl", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/yaw_alias_drift_support_manifest_rows.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/yaw_alias_drift_support_manifest_rows.summary.json"),
    )
    ap.add_argument("--stable_alias_max_abs_error", type=float, default=0.02)
    args = ap.parse_args()

    diag_paths = _expand_inputs(list(args.diagnostic_jsonl), patterns=("yaw_frame_diagnostic_rows.jsonl",))
    rows: list[dict[str, Any]] = []
    for path in diag_paths:
        rows.extend(_read_jsonl(path))

    selected, summary = build_row_support_manifest(
        rows,
        stable_alias_max_abs_error=float(args.stable_alias_max_abs_error),
    )
    summary["diagnostic_paths"] = [str(path.resolve()) for path in diag_paths]
    summary["output_jsonl"] = str(args.output_jsonl.resolve())

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output_jsonl)
    print(args.summary_json)


if __name__ == "__main__":
    main()
