#!/usr/bin/env python3
"""Audit yaw-observable coverage for v46 held-out task-frame validation.

This script is intentionally label-side only: it reads existing manifests and
uses offline labels to answer whether a proposed validation pool contains
enough yaw-observable, non-ambiguous near-contact rows.  It does not create
runtime inputs and it does not change close/handoff behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.task_frame_v46_alignment import task_frame_v46_labels_from_row  # noqa: E402
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import source_eval_root_key  # noqa: E402
from scripts.train_c2c_v2_task_frame_v46_alignment import _normalize_row_metadata  # noqa: E402


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _episode(row: Mapping[str, Any]) -> int:
    return int(row.get("episode_idx", row.get("episode", -1)) or -1)


def _step(row: Mapping[str, Any]) -> int:
    return int(row.get("step_idx", row.get("step", -1)) or -1)


def _yaw_class(row: Mapping[str, Any], labels: Mapping[str, Any]) -> str:
    value = str(row.get("yaw_observability_class", "") or "").strip().lower()
    if value:
        return value
    if bool(labels.get("yaw_observable", False)):
        return "observable"
    if bool(labels.get("yaw_ambiguous", False)):
        return "ambiguous"
    return "unobservable"


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        source_eval_root_key(row),
        _episode(row),
        _step(row),
        str(row.get("task_frame_v46_command_sweep_candidate_name", row.get("candidate_name", "")) or ""),
    )


def audit_coverage(
    manifests: list[Path],
    *,
    output_json: Path,
    output_jsonl: Path | None = None,
    near_xy_radius: float = 0.060,
    near_z_radius: float = 0.040,
    max_abs_yaw: float = 0.350,
    min_yaw_control_rows: int = 128,
    min_yaw_control_roots: int = 16,
    max_rows: int = 0,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    by_manifest: dict[str, Counter[str]] = {}
    by_episode: Counter[str] = Counter()
    by_root: Counter[str] = Counter()
    by_yaw_class: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()

    for manifest in manifests:
        manifest_counter: Counter[str] = Counter()
        for raw in _read_jsonl(manifest):
            counters["input_rows"] += 1
            manifest_counter["input_rows"] += 1
            row = _normalize_row_metadata(raw)
            labels = task_frame_v46_labels_from_row(row)
            if labels is None:
                counters["dropped_missing_labels"] += 1
                manifest_counter["dropped_missing_labels"] += 1
                continue
            counters["label_rows"] += 1
            manifest_counter["label_rows"] += 1
            dx = _safe_float(labels.get("dx"))
            dy = _safe_float(labels.get("dy"))
            dz = _safe_float(labels.get("dz"))
            dyaw = _safe_float(labels.get("dyaw"))
            xy_norm = float(np.hypot(dx, dy))
            yaw_class = _yaw_class(row, labels)
            by_yaw_class[yaw_class] += 1
            if bool(labels.get("yaw_observable", False)):
                counters["yaw_observable_rows"] += 1
                manifest_counter["yaw_observable_rows"] += 1
            if bool(labels.get("yaw_ambiguous", False)):
                counters["yaw_ambiguous_rows"] += 1
                manifest_counter["yaw_ambiguous_rows"] += 1
            yaw_control = bool(labels.get("yaw_observable", False)) and not bool(labels.get("yaw_ambiguous", False))
            if yaw_control:
                counters["yaw_control_rows"] += 1
                manifest_counter["yaw_control_rows"] += 1
            near = xy_norm <= float(near_xy_radius) and abs(dz) <= float(near_z_radius) and abs(dyaw) <= float(max_abs_yaw)
            if near:
                counters["near_contact_rows"] += 1
                manifest_counter["near_contact_rows"] += 1
            if yaw_control and not near:
                counters["yaw_control_not_near_rows"] += 1
                manifest_counter["yaw_control_not_near_rows"] += 1
                if xy_norm > float(near_xy_radius):
                    counters["yaw_control_not_near_xy"] += 1
                    manifest_counter["yaw_control_not_near_xy"] += 1
                if abs(dz) > float(near_z_radius):
                    counters["yaw_control_not_near_z"] += 1
                    manifest_counter["yaw_control_not_near_z"] += 1
                if abs(dyaw) > float(max_abs_yaw):
                    counters["yaw_control_not_near_yaw"] += 1
                    manifest_counter["yaw_control_not_near_yaw"] += 1
            if near and not yaw_control:
                counters["near_not_yaw_control_rows"] += 1
                manifest_counter["near_not_yaw_control_rows"] += 1
                if not bool(labels.get("yaw_observable", False)):
                    counters["near_not_yaw_control_unobservable"] += 1
                    manifest_counter["near_not_yaw_control_unobservable"] += 1
                if bool(labels.get("yaw_ambiguous", False)):
                    counters["near_not_yaw_control_ambiguous"] += 1
                    manifest_counter["near_not_yaw_control_ambiguous"] += 1
            keep = bool(yaw_control and near)
            if keep:
                root = source_eval_root_key(row)
                key = _candidate_key(row)
                if key in seen:
                    counters["duplicate_selected_rows"] += 1
                    manifest_counter["duplicate_selected_rows"] += 1
                    continue
                seen.add(key)
                enriched = dict(row)
                enriched.update(
                    {
                        "source_eval_root": root,
                        "episode_idx": _episode(row),
                        "step_idx": _step(row),
                        "task_frame_v46_yaw_holdout_candidate": True,
                        "task_frame_v46_yaw_holdout_reason": "yaw_control_near_contact",
                        "task_frame_v46_label_dx": dx,
                        "task_frame_v46_label_dy": dy,
                        "task_frame_v46_label_dz": dz,
                        "task_frame_v46_label_dyaw": dyaw,
                        "task_frame_v46_label_xy_norm": xy_norm,
                        "task_frame_v46_label_yaw_observable": bool(labels.get("yaw_observable", False)),
                        "task_frame_v46_label_yaw_ambiguous": bool(labels.get("yaw_ambiguous", False)),
                        "task_frame_v46_yaw_observability_class": yaw_class,
                        "task_frame_v46_source_manifest": str(manifest),
                    }
                )
                selected.append(enriched)
                counters["selected_yaw_control_near_contact_rows"] += 1
                manifest_counter["selected_yaw_control_near_contact_rows"] += 1
                by_episode[f"ep{_episode(row):03d}"] += 1
                by_root[root] += 1
        by_manifest[str(manifest)] = manifest_counter

    selected = sorted(
        selected,
        key=lambda row: (
            str(row.get("source_eval_root", "")),
            int(row.get("episode_idx", -1)),
            int(row.get("step_idx", -1)),
            str(row.get("task_frame_v46_command_sweep_candidate_name", "")),
        ),
    )
    if max_rows > 0:
        selected = selected[: int(max_rows)]

    selected_roots = {str(row.get("source_eval_root", "")) for row in selected}
    selected_episodes = {int(row.get("episode_idx", -1)) for row in selected}
    violations: list[str] = []
    if len(selected) < int(min_yaw_control_rows):
        violations.append("insufficient_yaw_control_rows")
    if len(selected_roots) < int(min_yaw_control_roots):
        violations.append("insufficient_yaw_control_roots")
    summary = {
        "schema_version": "c2c_v2_task_frame_yaw_holdout_coverage_v1",
        "manifests": [str(path) for path in manifests],
        "counters": dict(counters),
        "by_manifest": {key: dict(value) for key, value in by_manifest.items()},
        "by_yaw_observability_class": dict(by_yaw_class),
        "selected_rows": len(selected),
        "selected_source_eval_roots": len(selected_roots),
        "selected_episodes": sorted(v for v in selected_episodes if v >= 0),
        "by_selected_episode": dict(by_episode),
        "by_selected_source_eval_root_top20": dict(by_root.most_common(20)),
        "near_xy_radius": float(near_xy_radius),
        "near_z_radius": float(near_z_radius),
        "max_abs_yaw": float(max_abs_yaw),
        "min_yaw_control_rows": int(min_yaw_control_rows),
        "min_yaw_control_roots": int(min_yaw_control_roots),
        "status": "pass" if not violations else "insufficient_coverage",
        "violations": violations,
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_audit": True,
        "privileged_label_boundary": "offline_coverage_audit_only",
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary["output_jsonl"] = str(output_jsonl)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit yaw-observable coverage for v46 held-out validation.")
    parser.add_argument("--manifest", nargs="+", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--near_xy_radius", type=float, default=0.060)
    parser.add_argument("--near_z_radius", type=float, default=0.040)
    parser.add_argument("--max_abs_yaw", type=float, default=0.350)
    parser.add_argument("--min_yaw_control_rows", type=int, default=128)
    parser.add_argument("--min_yaw_control_roots", type=int, default=16)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_coverage(
        list(args.manifest),
        output_json=args.output_json,
        output_jsonl=args.output_jsonl,
        near_xy_radius=float(args.near_xy_radius),
        near_z_radius=float(args.near_z_radius),
        max_abs_yaw=float(args.max_abs_yaw),
        min_yaw_control_rows=int(args.min_yaw_control_rows),
        min_yaw_control_roots=int(args.min_yaw_control_roots),
        max_rows=int(args.max_rows),
    )
    print(json.dumps({k: summary[k] for k in ("status", "selected_rows", "selected_source_eval_roots", "violations")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
