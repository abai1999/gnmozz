#!/usr/bin/env python3
"""Build applied-transition rows for v46/v52/v53 task-frame control learning.

The source is runtime smoke gripper traces plus runtime observations. True
pre/post residuals are kept only as offline labels; runtime inputs remain
non-privileged and are referenced through the saved observation NPZ.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=32)
def _read_jsonl_cached(path_str: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_str)
    if not path.exists():
        return tuple()
    return tuple(_read_jsonl(path))


def _discover_trace_files(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        if item.is_file() and item.name.endswith(".jsonl"):
            out.append(item)
        elif item.is_dir():
            out.extend(sorted(item.rglob("*_gripper_trace.jsonl")))
    return sorted(set(out))


def _episode_from_trace(path: Path, row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("episode_idx"))
    except Exception:
        pass
    match = re.search(r"ep(\d+)_gripper_trace", path.name)
    if match:
        return int(match.group(1))
    return -1


def _source_root_from_trace(path: Path) -> Path:
    if path.parent.name == "gripper_traces":
        return path.parent.parent
    for parent in path.parents:
        if (parent / "runtime_observations").is_dir():
            return parent
    return path.parent


def _runtime_obs_path(source_root: Path, episode_idx: int) -> Path | None:
    candidates = [
        source_root / "runtime_observations" / f"ep{episode_idx:03d}_runtime_obs.npz",
        source_root / "runtime_observations" / f"ep{episode_idx}_runtime_obs.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted((source_root / "runtime_observations").glob(f"*{episode_idx:03d}*runtime_obs*.npz"))
    return matches[0] if matches else None


def _as_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _command_sweep_spec_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    spec_path_value = str(row.get("task_frame_v46_command_sweep_spec_path", "") or "").strip()
    row_index = _as_int(row.get("task_frame_v46_command_sweep_row_index", -1), -1)
    if not spec_path_value or row_index < 0:
        return None
    spec_rows = _read_jsonl_cached(spec_path_value)
    if row_index >= len(spec_rows):
        return None
    spec_row = dict(spec_rows[row_index])
    if not spec_row:
        return None
    return spec_row


def _vec(value: Any, length: int) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < int(length) or not np.all(np.isfinite(arr[: int(length)])):
        return None
    return arr[: int(length)].astype(np.float32)


def _xy_norm(vec4: np.ndarray) -> float:
    return float(np.linalg.norm(vec4[:2]))


def _hash_rank(key: str, *, seed: int) -> int:
    value = 2166136261
    for ch in f"{key}|{seed}":
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _build_row(
    raw: Mapping[str, Any],
    *,
    trace_path: Path,
    source_root: Path,
    runtime_obs: Path,
    episode_idx: int,
) -> dict[str, Any] | None:
    pre = _vec(raw.get("grasp_probe_pre_true_error_t"), 4)
    post = _vec(raw.get("grasp_probe_post_true_error_t"), 4)
    command = _vec(raw.get("grasp_probe_local_command_local_6d"), 6)
    applied_v46 = _vec(raw.get("task_frame_v46_applied_local_6d"), 6)
    if pre is None or post is None or command is None:
        return None
    if applied_v46 is None:
        applied_v46 = np.zeros((6,), dtype=np.float32)
    step_idx = int(raw.get("step_idx", raw.get("step", 0)) or 0)
    delta = post - pre
    v46_applied = bool(raw.get("task_frame_v46_applied", False))
    spec_row = _command_sweep_spec_row(raw)
    source_eval_root = str(spec_row.get("source_eval_root", source_root)) if spec_row else str(source_root)
    source_trace_path = str(spec_row.get("trace_path", trace_path)) if spec_row else str(trace_path)
    source_runtime_obs_path = str(spec_row.get("runtime_obs_path", runtime_obs)) if spec_row else str(runtime_obs)
    source_sequence_id = str(spec_row.get("sequence_id", f"{source_eval_root}::ep{episode_idx:03d}")) if spec_row else f"{source_eval_root}::ep{episode_idx:03d}"
    xy_pre = _xy_norm(pre)
    xy_post = _xy_norm(post)
    z_pre = abs(float(pre[2]))
    z_post = abs(float(post[2]))
    yaw_pre = abs(float(pre[3]))
    yaw_post = abs(float(post[3]))
    return {
        "schema_version": "c2c_v2_task_frame_applied_transition_manifest_v1",
        "source_eval_root": source_eval_root,
        "source_eval_root_kind": str(spec_row.get("source_eval_root_kind", "runtime_smoke_applied_transition")) if spec_row else "runtime_smoke_applied_transition",
        "session_id": str(spec_row.get("session_id", Path(source_eval_root).name)) if spec_row else str(source_root.name),
        "sequence_id": source_sequence_id,
        "episode_idx": int(episode_idx),
        "step_idx": int(step_idx),
        "stage": str(raw.get("stage", raw.get("runtime_stage", raw.get("grasp_probe_horizon_stage_sequence", ["unknown"])[0] if isinstance(raw.get("grasp_probe_horizon_stage_sequence"), list) and raw.get("grasp_probe_horizon_stage_sequence") else "unknown"))),
        "trace_path": str(trace_path),
        "runtime_obs_path": source_runtime_obs_path,
        "obs_pointer": {"trace_path": source_trace_path, "runtime_obs_path": source_runtime_obs_path},
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_pre_post_transition_labels_only",
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
        "next_privileged_dx": float(post[0]),
        "next_privileged_dy": float(post[1]),
        "next_privileged_dz": float(post[2]),
        "next_privileged_dyaw": float(post[3]),
        "offline_transition_delta": [float(v) for v in delta.tolist()],
        "applied_control_command_local_6d": [float(v) for v in command.tolist()],
        "applied_control_command_xy": [float(v) for v in command[:2].tolist()],
        "task_frame_v46_applied_local_6d_offline": [float(v) for v in applied_v46.tolist()],
        "task_frame_v46_applied": bool(v46_applied),
        "task_frame_v46_xy_control_source": str(raw.get("task_frame_v46_xy_control_source", "")),
        "task_frame_v46_risk_reason": str(raw.get("task_frame_v46_risk_reason", "")),
        "task_frame_v46_command_sweep_executed": bool(raw.get("task_frame_v46_command_sweep_executed", False)),
        "task_frame_v46_command_sweep_row_index": _as_int(raw.get("task_frame_v46_command_sweep_row_index", -1), -1),
        "task_frame_v46_command_sweep_candidate_name": str(raw.get("task_frame_v46_command_sweep_candidate_name", "")),
        "failure_bucket": str(raw.get("failure_bucket", "runtime_applied_transition") or "runtime_applied_transition"),
        "observability_bucket": str(raw.get("grasp_probe_visibility_bucket", raw.get("observability_bucket", "unknown")) or "unknown"),
        "yaw_observability_class": "observable" if bool(raw.get("task_frame_v46_yaw_observable", False)) else "ambiguous_or_unobservable",
        "xy_contracted_observed": bool(xy_post < xy_pre),
        "z_contracted_observed": bool(z_post < z_pre),
        "yaw_contracted_observed": bool(yaw_post < yaw_pre),
        "combined_contracted_observed": bool((xy_post + z_post + yaw_post) < (xy_pre + z_pre + yaw_pre)),
        "xy_worsen_observed": bool(xy_post > xy_pre),
        "x_worsen_observed": bool(abs(float(post[0])) > abs(float(pre[0]))),
        "y_worsen_observed": bool(abs(float(post[1])) > abs(float(pre[1]))),
        "close_leak": bool(raw.get("planner_gripper_close_allowed", False) and not raw.get("planner_gripper_handoff_allowed", False)),
        "planner_gripper_close_requested": bool(raw.get("planner_gripper_close_requested", False)),
        "planner_gripper_handoff_allowed": bool(raw.get("planner_gripper_handoff_allowed", False)),
        "alignment_ready_for_handoff": bool(raw.get("alignment_ready_for_handoff", raw.get("planner_gripper_strict_handoff_ready", False))),
    }


def build_manifest(
    input_paths: list[Path],
    *,
    output_jsonl: Path,
    summary_json: Path,
    require_v46_applied: bool = True,
    require_command_sweep_executed: bool = False,
    max_rows_per_source: int = 0,
    seed: int = 7,
) -> dict[str, Any]:
    trace_files = _discover_trace_files(input_paths)
    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for trace_path in trace_files:
        source_root = _source_root_from_trace(trace_path)
        trace_rows = _read_jsonl(trace_path)
        counters["input_rows"] += len(trace_rows)
        for raw in trace_rows:
            if require_command_sweep_executed and not bool(raw.get("task_frame_v46_command_sweep_executed", False)):
                counters["dropped_not_command_sweep_executed"] += 1
                continue
            if require_v46_applied and not bool(raw.get("task_frame_v46_applied", False)):
                counters["dropped_not_v46_applied"] += 1
                continue
            episode_idx = _episode_from_trace(trace_path, raw)
            if episode_idx < 0:
                counters["dropped_missing_episode"] += 1
                continue
            spec_row = _command_sweep_spec_row(raw)
            if spec_row is not None:
                spec_source_root_value = str(spec_row.get("source_eval_root", "") or "").strip()
                if spec_source_root_value:
                    source_root = Path(spec_source_root_value)
            runtime_obs = None
            if spec_row is not None:
                spec_runtime_obs_value = str(spec_row.get("runtime_obs_path", "") or "").strip()
                if spec_runtime_obs_value:
                    candidate = Path(spec_runtime_obs_value)
                    if candidate.exists():
                        runtime_obs = candidate
            if runtime_obs is None:
                runtime_obs = _runtime_obs_path(source_root, episode_idx)
            if runtime_obs is None:
                counters["dropped_missing_runtime_obs"] += 1
                continue
            row = _build_row(raw, trace_path=trace_path, source_root=source_root, runtime_obs=runtime_obs, episode_idx=episode_idx)
            if row is None:
                counters["dropped_missing_pre_post_or_command"] += 1
                continue
            rows.append(row)
            counters["candidate_rows"] += 1
    if max_rows_per_source > 0:
        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_source.setdefault(str(row["source_eval_root"]), []).append(row)
        kept: list[dict[str, Any]] = []
        for source, source_rows in sorted(by_source.items()):
            source_rows = sorted(
                source_rows,
                key=lambda row: (
                    _hash_rank(f"{source}|{row['episode_idx']}|{row['step_idx']}", seed=seed),
                    int(row["episode_idx"]),
                    int(row["step_idx"]),
                ),
            )
            kept.extend(source_rows[: int(max_rows_per_source)])
        rows = kept
    rows.sort(key=lambda row: (str(row["source_eval_root"]), int(row["episode_idx"]), int(row["step_idx"])))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_counts = Counter(str(row["source_eval_root"]) for row in rows)
    episode_counts = Counter(f"ep{int(row['episode_idx']):03d}" for row in rows)
    xy_contract = [bool(row["xy_contracted_observed"]) for row in rows]
    z_contract = [bool(row["z_contracted_observed"]) for row in rows]
    yaw_contract = [bool(row["yaw_contracted_observed"]) for row in rows]
    summary = {
        "schema_version": "c2c_v2_task_frame_applied_transition_manifest_summary_v1",
        "output_jsonl": str(output_jsonl),
        "input_paths": [str(path) for path in input_paths],
        "trace_files": [str(path) for path in trace_files],
        "require_v46_applied": bool(require_v46_applied),
        "require_command_sweep_executed": bool(require_command_sweep_executed),
        "counters": dict(counters),
        "retained_rows": len(rows),
        "source_eval_roots": len(source_counts),
        "source_eval_root_counts": dict(source_counts),
        "episode_counts": dict(episode_counts),
        "observed_xy_contraction": float(np.mean(xy_contract)) if xy_contract else 0.0,
        "observed_z_contraction": float(np.mean(z_contract)) if z_contract else 0.0,
        "observed_yaw_contraction": float(np.mean(yaw_contract)) if yaw_contract else 0.0,
        "close_leak_rows": int(sum(bool(row.get("close_leak", False)) for row in rows)),
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_pre_post_transition_labels_only",
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v46/v53 applied-transition manifest from runtime smoke traces.")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--include_non_applied", action="store_true", default=False)
    parser.add_argument("--require_command_sweep_executed", action="store_true", default=False)
    parser.add_argument("--max_rows_per_source", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        list(args.input),
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        require_v46_applied=not bool(args.include_non_applied),
        require_command_sweep_executed=bool(args.require_command_sweep_executed),
        max_rows_per_source=int(args.max_rows_per_source),
        seed=int(args.seed),
    )
    print(json.dumps({k: summary[k] for k in ("retained_rows", "source_eval_roots", "episode_counts", "close_leak_rows")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
