#!/usr/bin/env python3
"""Build near-field/progress supervision rows from runtime traces.

Unlike applied-transition manifests, this keeps rows that have a pre-step
offline residual label even when no C2C action was applied. The labels are
offline-only; runtime inputs are referenced through runtime_observations.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_trace_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_file() and item.name.endswith(".jsonl"):
            files.append(item)
        elif item.is_dir():
            files.extend(sorted(item.rglob("*_gripper_trace.jsonl")))
    return sorted(set(files))


def _episode_from_trace(path: Path, row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("episode_idx"))
    except Exception:
        pass
    match = re.search(r"ep(\d+)_gripper_trace", path.name)
    return int(match.group(1)) if match else -1


def _source_root_from_trace(path: Path) -> Path:
    if path.parent.name == "gripper_traces":
        return path.parent.parent
    for parent in path.parents:
        if (parent / "runtime_observations").is_dir():
            return parent
    return path.parent


def _runtime_obs_path(source_root: Path, episode_idx: int) -> Path | None:
    obs_dir = source_root / "runtime_observations"
    candidates = [obs_dir / f"ep{episode_idx:03d}_runtime_obs.npz", obs_dir / f"ep{episode_idx}_runtime_obs.npz"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(obs_dir.glob(f"*{episode_idx:03d}*runtime_obs*.npz"))
    return matches[0] if matches else None


def _vec(value: Any, length: int) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < int(length) or not np.all(np.isfinite(arr[: int(length)])):
        return None
    return arr[: int(length)].astype(np.float32)


def _build_row(
    raw: Mapping[str, Any],
    *,
    trace_path: Path,
    source_root: Path,
    runtime_obs: Path,
    episode_idx: int,
    near_field_xy_radius: float,
    near_field_z_radius: float,
) -> dict[str, Any] | None:
    pre = _vec(raw.get("grasp_probe_pre_true_error_t"), 4)
    if pre is None:
        return None
    step_idx = int(raw.get("step_idx", raw.get("step", 0)) or 0)
    xy = float(np.linalg.norm(pre[:2]))
    z_abs = float(abs(float(pre[2])))
    near = bool(xy <= float(near_field_xy_radius) and z_abs <= float(near_field_z_radius))
    far_z = bool(z_abs > max(float(near_field_z_radius) * 2.0, 0.08))
    return {
        "schema_version": "c2c_v2_task_frame_nearfield_manifest_v1",
        "source_eval_root": str(source_root),
        "source_eval_root_kind": "runtime_nearfield_progress",
        "session_id": str(source_root.name),
        "sequence_id": f"{source_root}::ep{episode_idx:03d}",
        "episode_idx": int(episode_idx),
        "step_idx": int(step_idx),
        "stage": str(raw.get("stage", raw.get("runtime_stage", "unknown"))),
        "trace_path": str(trace_path),
        "runtime_obs_path": str(runtime_obs),
        "obs_pointer": {"trace_path": str(trace_path), "runtime_obs_path": str(runtime_obs)},
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_pre_residual_nearfield_label_only",
        "runtime_input_schema": "wrist_rgbd_depth_validity_proprio_planner_prior_history",
        "offline_labels": {
            "dx": float(pre[0]),
            "dy": float(pre[1]),
            "dz": float(pre[2]),
            "dyaw": float(pre[3]),
            "xy_observable": bool(raw.get("task_frame_v46_xy_observable", True)),
            "z_observable": bool(raw.get("task_frame_v46_z_observable", True)),
            "yaw_observable": bool(raw.get("task_frame_v46_yaw_observable", False)),
            "yaw_ambiguous": bool(raw.get("task_frame_v46_yaw_ambiguous", True)),
        },
        "failure_bucket": "nearfield_positive" if near else "nearfield_far_z_negative" if far_z else "nearfield_boundary_negative",
        "observability_bucket": str(raw.get("grasp_probe_visibility_bucket", raw.get("observability_bucket", "unknown")) or "unknown"),
        "yaw_observability_class": "observable" if bool(raw.get("task_frame_v46_yaw_observable", False)) else "ambiguous_or_unobservable",
        "nearfield_label": bool(near),
        "nearfield_xy_norm_label": float(xy),
        "nearfield_z_abs_label": float(z_abs),
        "task_frame_v46_activation_ready_observed": bool(raw.get("task_frame_v46_activation_ready", False)),
        "task_frame_v46_near_field_confidence_observed": float(raw.get("task_frame_v46_near_field_confidence", 0.0) or 0.0),
    }


def build_manifest(
    input_paths: list[Path],
    *,
    output_jsonl: Path,
    summary_json: Path,
    near_field_xy_radius: float = 0.060,
    near_field_z_radius: float = 0.040,
) -> dict[str, Any]:
    trace_files = _discover_trace_files(input_paths)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for trace_path in trace_files:
        source_root = _source_root_from_trace(trace_path)
        for raw in _read_jsonl(trace_path):
            counters["input_rows"] += 1
            episode_idx = _episode_from_trace(trace_path, raw)
            if episode_idx < 0:
                counters["dropped_missing_episode"] += 1
                continue
            runtime_obs = _runtime_obs_path(source_root, episode_idx)
            if runtime_obs is None:
                counters["dropped_missing_runtime_obs"] += 1
                continue
            row = _build_row(
                raw,
                trace_path=trace_path,
                source_root=source_root,
                runtime_obs=runtime_obs,
                episode_idx=episode_idx,
                near_field_xy_radius=near_field_xy_radius,
                near_field_z_radius=near_field_z_radius,
            )
            if row is None:
                counters["dropped_missing_pre_residual"] += 1
                continue
            rows.append(row)
            counters["candidate_rows"] += 1
    rows.sort(key=lambda row: (str(row["source_eval_root"]), int(row["episode_idx"]), int(row["step_idx"])))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_counts = Counter(str(row["source_eval_root"]) for row in rows)
    episode_counts = Counter(f"ep{int(row['episode_idx']):03d}" for row in rows)
    bucket_counts = Counter(str(row["failure_bucket"]) for row in rows)
    summary = {
        "schema_version": "c2c_v2_task_frame_nearfield_manifest_summary_v1",
        "output_jsonl": str(output_jsonl),
        "input_paths": [str(path) for path in input_paths],
        "trace_files": [str(path) for path in trace_files],
        "retained_rows": len(rows),
        "source_eval_roots": len(source_counts),
        "source_eval_root_counts": dict(source_counts),
        "episode_counts": dict(episode_counts),
        "bucket_counts": dict(bucket_counts),
        "near_field_xy_radius": float(near_field_xy_radius),
        "near_field_z_radius": float(near_field_z_radius),
        "counters": dict(counters),
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_pre_residual_nearfield_label_only",
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v57/v58 near-field/progress manifest from runtime traces.")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--near_field_xy_radius", type=float, default=0.060)
    parser.add_argument("--near_field_z_radius", type=float, default=0.040)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        list(args.input),
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        near_field_xy_radius=float(args.near_field_xy_radius),
        near_field_z_radius=float(args.near_field_z_radius),
    )
    print(json.dumps({k: summary[k] for k in ("retained_rows", "source_eval_roots", "episode_counts", "bucket_counts")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
