#!/usr/bin/env python3
"""Build a v46 task-frame alignment manifest from relabeled runtime artifacts.

The manifest is a training/evaluation input list. It keeps privileged task-frame
residuals only as offline labels, while requiring runtime-visible RGBD
observation pointers for every retained row.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.task_frame_v46_alignment import task_frame_v46_labels_from_row  # noqa: E402
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import source_eval_root_key  # noqa: E402
from scripts.train_c2c_v2_task_frame_v46_alignment import (  # noqa: E402
    _label_within_support,
    _normalize_row_metadata,
    _row_has_privileged_runtime,
    _runtime_npz_path_from_row,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_jsonl(inputs: list[Path], *, include_failure_tail_rows: bool) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        if item.is_file():
            out.append(item)
            continue
        if not item.is_dir():
            continue
        out.extend(sorted(item.rglob("frame_residual_v2.jsonl")))
        if include_failure_tail_rows:
            out.extend(sorted(item.rglob("grasp_failure_tail_intervention_rows.jsonl")))
    return sorted(set(out))


def _hash_rank(row: Mapping[str, Any], *, seed: int) -> int:
    key = (
        f"{source_eval_root_key(row)}|{int(row.get('episode_idx', -1))}|"
        f"{int(row.get('step_idx', row.get('step', -1)))}|{seed}"
    )
    value = 2166136261
    for ch in key:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _finite_label_reason(
    row: Mapping[str, Any],
    *,
    max_abs_xy_label: float,
    max_abs_z_label: float,
    max_abs_yaw_label: float,
) -> tuple[dict[str, Any] | None, str]:
    labels = task_frame_v46_labels_from_row(row)
    if labels is None:
        return None, "missing_or_nonfinite_label"
    if not _label_within_support(
        labels,
        max_abs_xy_label=max_abs_xy_label,
        max_abs_z_label=max_abs_z_label,
        max_abs_yaw_label=max_abs_yaw_label,
    ):
        return labels, "outside_local_support"
    return labels, "ok"


def build_manifest(
    input_paths: list[Path],
    *,
    output_jsonl: Path,
    summary_json: Path,
    include_failure_tail_rows: bool = False,
    max_rows_per_source: int = 0,
    max_rows_per_source_yaw_class: int = 0,
    max_abs_xy_label: float = 0.080,
    max_abs_z_label: float = 0.080,
    max_abs_yaw_label: float = 0.350,
    seed: int = 7,
) -> dict[str, Any]:
    files = _discover_jsonl(input_paths, include_failure_tail_rows=include_failure_tail_rows)
    counters: Counter[str] = Counter()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_row_counts: Counter[str] = Counter()
    for path in files:
        rows = _read_jsonl(path)
        input_row_counts[str(path)] = len(rows)
        for raw in rows:
            counters["input_rows"] += 1
            row = _normalize_row_metadata(raw)
            row.setdefault("v46_manifest_source_jsonl", str(path))
            if _row_has_privileged_runtime(row):
                counters["dropped_privileged_runtime"] += 1
                continue
            runtime_npz = _runtime_npz_path_from_row(row)
            if runtime_npz is None:
                counters["dropped_missing_runtime_obs"] += 1
                continue
            labels, reason = _finite_label_reason(
                row,
                max_abs_xy_label=max_abs_xy_label,
                max_abs_z_label=max_abs_z_label,
                max_abs_yaw_label=max_abs_yaw_label,
            )
            if reason != "ok":
                counters[f"dropped_{reason}"] += 1
                continue
            root = source_eval_root_key(row)
            if not root:
                counters["dropped_missing_source_root"] += 1
                continue
            row["runtime_obs_path"] = str(runtime_npz)
            row["source_eval_root"] = str(root)
            row["uses_privileged_runtime"] = False
            row["v46_offline_label_keys"] = ["dx", "dy", "dz", "dyaw"]
            row["v46_label_norm_xy"] = float(np.hypot(float(labels["dx"]), float(labels["dy"]))) if labels is not None else float("nan")
            by_source[root].append(row)
            counters["candidate_rows"] += 1

    retained: list[dict[str, Any]] = []
    for root, rows in sorted(by_source.items()):
        rows = sorted(rows, key=lambda row: (_hash_rank(row, seed=seed), int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1)))))
        class_limit = int(max_rows_per_source_yaw_class)
        if class_limit > 0:
            by_yaw_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                yaw_label = row.get("yaw_label", {})
                yaw_label = yaw_label if isinstance(yaw_label, Mapping) else {}
                yaw_class = str(row.get("yaw_observability_class", yaw_label.get("yaw_observability_class", "unknown")) or "unknown")
                by_yaw_class[yaw_class].append(row)
            rows = []
            for yaw_class in ("observable", "ambiguous", "unobservable", "unknown"):
                rows.extend(by_yaw_class.get(yaw_class, [])[:class_limit])
            for yaw_class, class_rows in sorted(by_yaw_class.items()):
                if yaw_class not in {"observable", "ambiguous", "unobservable", "unknown"}:
                    rows.extend(class_rows[:class_limit])
        limit = int(max_rows_per_source)
        if limit > 0:
            rows = rows[:limit]
        rows = sorted(rows, key=lambda row: (source_eval_root_key(row), int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1)))))
        retained.extend(rows)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    source_counts = Counter(source_eval_root_key(row) for row in retained)
    bucket_counts = Counter(str(row.get("failure_bucket", row.get("bucket", "unknown")) or "unknown") for row in retained)
    obs_counts = Counter(str(row.get("visual_observability_class", row.get("observability_bucket", "unknown")) or "unknown") for row in retained)
    episodes = sorted({int(row.get("episode_idx", -1)) for row in retained if int(row.get("episode_idx", -1)) >= 0})
    summary = {
        "schema_version": "c2c_v2_task_frame_v46_manifest_summary_v1",
        "output_jsonl": str(output_jsonl),
        "input_paths": [str(path) for path in input_paths],
        "discovered_jsonl_files": len(files),
        "input_row_counts": dict(input_row_counts),
        "counters": dict(counters),
        "retained_rows": len(retained),
        "source_eval_roots": len(source_counts),
        "source_eval_root_counts": dict(source_counts),
        "episode_count": len(episodes),
        "episodes": episodes,
        "failure_bucket_counts": dict(bucket_counts),
        "observability_counts": dict(obs_counts),
        "max_rows_per_source": int(max_rows_per_source),
        "max_rows_per_source_yaw_class": int(max_rows_per_source_yaw_class),
        "max_abs_xy_label": float(max_abs_xy_label),
        "max_abs_z_label": float(max_abs_z_label),
        "max_abs_yaw_label": float(max_abs_yaw_label),
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_labels_only",
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a C2C v2 v46 task-frame alignment manifest.")
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Input frame_residual_v2 jsonl files or artifact directories.")
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--include_failure_tail_rows", action="store_true", default=False)
    parser.add_argument("--max_rows_per_source", type=int, default=0)
    parser.add_argument("--max_rows_per_source_yaw_class", type=int, default=0)
    parser.add_argument("--max_abs_xy_label", type=float, default=0.080)
    parser.add_argument("--max_abs_z_label", type=float, default=0.080)
    parser.add_argument("--max_abs_yaw_label", type=float, default=0.350)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        list(args.input),
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        include_failure_tail_rows=bool(args.include_failure_tail_rows),
        max_rows_per_source=int(args.max_rows_per_source),
        max_rows_per_source_yaw_class=int(args.max_rows_per_source_yaw_class),
        max_abs_xy_label=float(args.max_abs_xy_label),
        max_abs_z_label=float(args.max_abs_z_label),
        max_abs_yaw_label=float(args.max_abs_yaw_label),
        seed=int(args.seed),
    )
    print(json.dumps({k: summary[k] for k in ("retained_rows", "source_eval_roots", "episode_count", "counters")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
