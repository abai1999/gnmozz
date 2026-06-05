#!/usr/bin/env python3
"""Build a spatial-temporal XY dataset from runtime observations and trace rows."""

from __future__ import annotations

import argparse
import json
import zipfile
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.recovery_audit import load_trace_rows, planner_bias_xyyaw, roi_box_from_action_prior, trace_episode_index
from prismatic.robot.residual_transforms import world_delta_to_local


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _row_bucket(row: dict[str, Any]) -> str:
    return str(row.get("failure_bucket", row.get("failure_morphology_bucket", "")) or "unknown")


def _row_observability_bucket(row: dict[str, Any]) -> str:
    for key in ("visual_observability_class", "grasp_probe_visibility_bucket", "visibility", "runtime_visibility_bucket"):
        value = str(row.get(key, "") or "")
        if value:
            return value
    if _safe_bool(row.get("wrist_is_occluded"), False):
        return "occluded"
    if _safe_bool(row.get("wrist_is_low_visibility"), False):
        return "low_observability"
    return "unknown"


def _read_target_pose(npz: np.lib.npyio.NpzFile) -> np.ndarray | None:
    for key in ("episode_target_pose_7d", "target_pose_7d", "privileged_target_pose_7d"):
        if key in npz.files:
            arr = np.asarray(npz[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                arr = arr[0]
            arr = arr.reshape(-1)
            if arr.size >= 7 and np.all(np.isfinite(arr[:7])):
                return arr[:7].astype(np.float32)
    return None


def _load_selected_episodes(trace_dir: Path, eval_report: dict[str, Any] | None, all_episodes: bool) -> list[int]:
    if all_episodes:
        return sorted({trace_episode_index(p) for p in trace_dir.glob("ep*_gripper_trace.jsonl") if trace_episode_index(p) >= 0})
    if eval_report and "stage_stats" in eval_report:
        selected = []
        for stat in eval_report.get("stage_stats", []):
            ep = int(stat.get("episode_index", -1))
            if ep < 0:
                continue
            if not bool(stat.get("success", False)) or int(stat.get("invalid_action_count", 0)) > 0:
                selected.append(ep)
        if selected:
            return sorted(set(selected))
    return sorted({trace_episode_index(p) for p in trace_dir.glob("ep*_gripper_trace.jsonl") if trace_episode_index(p) >= 0})


def _select_tail_rows(trace_rows: list[dict[str, Any]], *, tail_steps: int) -> list[dict[str, Any]]:
    if int(tail_steps) <= 0:
        return list(trace_rows)
    active_steps = [
        int(row.get("step", -1))
        for row in trace_rows
        if _safe_bool(row.get("grasp_probe_active"), False) and int(row.get("step", -1)) >= 0
    ]
    if active_steps:
        lo = max(0, min(active_steps) - int(tail_steps))
        hi = max(active_steps) + int(tail_steps)
    else:
        finite_steps = [int(row.get("step", -1)) for row in trace_rows if int(row.get("step", -1)) >= 0]
        if not finite_steps:
            return []
        hi = max(finite_steps)
        lo = max(0, hi - int(tail_steps) + 1)
    return [row for row in trace_rows if lo <= int(row.get("step", -1)) <= hi]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=Path, action="append", required=True, help="Evaluation root containing gripper_traces/ and runtime_observations/")
    ap.add_argument("--output_jsonl", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets_xy_spatial_temporal_v42/xy_spatial_temporal_dataset_v1.jsonl"))
    ap.add_argument("--output_summary", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets_xy_spatial_temporal_v42/xy_spatial_temporal_dataset_v1_summary.json"))
    ap.add_argument("--crop_size", type=int, default=96)
    ap.add_argument("--resize_to", type=int, default=96)
    ap.add_argument("--tail_steps", type=int, default=16)
    ap.add_argument("--all_episodes", action="store_true", default=False)
    args = ap.parse_args()

    records: list[dict[str, Any]] = []
    summary = {
        "schema_version": "c2c_v2_xy_spatial_temporal_dataset_v1",
        "eval_roots": [str(p) for p in args.eval_root],
        "output_jsonl": str(args.output_jsonl),
        "output_summary": str(args.output_summary),
        "crop_size": int(args.crop_size),
        "resize_to": int(args.resize_to),
        "tail_steps": int(args.tail_steps),
        "rows": 0,
        "episodes": 0,
        "active_rows": 0,
        "label_available_rows": 0,
        "failure_bucket_counts": {},
        "observability_bucket_counts": {},
        "source_eval_root_counts": {},
        "skipped_episodes": 0,
        "skipped_episode_reasons": {},
    }
    failure_bucket_counts: Counter[str] = Counter()
    observability_bucket_counts: Counter[str] = Counter()
    source_eval_root_counts: Counter[str] = Counter()
    skipped_episode_reasons: Counter[str] = Counter()
    selected_episode_count = 0

    for eval_root in args.eval_root:
        eval_root = eval_root.resolve()
        trace_dir = eval_root / "gripper_traces"
        obs_dir = eval_root / "runtime_observations"
        if not trace_dir.exists() or not obs_dir.exists():
            raise FileNotFoundError(f"Missing gripper_traces or runtime_observations under {eval_root}")
        eval_results_path = eval_root / "eval_results.json"
        eval_report = json.loads(eval_results_path.read_text(encoding="utf-8")) if eval_results_path.exists() else {}
        selected_eps = _load_selected_episodes(trace_dir, eval_report, bool(args.all_episodes))
        selected_episode_count += len(selected_eps)
        for ep_idx in selected_eps:
            trace_path = trace_dir / f"ep{ep_idx:03d}_gripper_trace.jsonl"
            obs_path = obs_dir / f"ep{ep_idx:03d}_runtime_obs.npz"
            if not trace_path.exists() or not obs_path.exists():
                continue
            trace_rows = _select_tail_rows(
                sorted(load_trace_rows(trace_path), key=lambda r: int(r.get("step", -1))),
                tail_steps=int(args.tail_steps),
            )
            try:
                obs_npz_ctx = np.load(obs_path, allow_pickle=False)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                skipped_episode_reasons[type(exc).__name__] += 1
                continue
            with obs_npz_ctx as obs_npz:
                target_pose = _read_target_pose(obs_npz)
                gripper_pose_arr = np.asarray(obs_npz["gripper_pose"], dtype=np.float32)
                planner_action_world_arr = np.asarray(obs_npz["planner_action_world_6d"], dtype=np.float32)
                proprio_arr = np.asarray(obs_npz["proprio"], dtype=np.float32)
                wrist_rgb_arr = np.asarray(obs_npz["wrist_rgb"], dtype=np.uint8)
                wrist_depth_arr = np.asarray(obs_npz["wrist_depth"], dtype=np.float32)
                active_rows = sum(1 for row in trace_rows if _safe_bool(row.get("grasp_probe_active"), False))
                for row in trace_rows:
                    step_idx = int(row.get("step", -1))
                    if step_idx < 0 or step_idx >= int(gripper_pose_arr.shape[0]):
                        continue
                    gripper_pose = np.asarray(gripper_pose_arr[step_idx], dtype=np.float32)
                    planner_prior_world = np.asarray(planner_action_world_arr[step_idx], dtype=np.float32)
                    proprio = np.asarray(proprio_arr[step_idx], dtype=np.float32)
                    wrist_rgb = np.asarray(wrist_rgb_arr[step_idx], dtype=np.uint8)
                    wrist_depth = np.asarray(wrist_depth_arr[step_idx], dtype=np.float32)
                    planner_prior_local = world_delta_to_local(planner_prior_world[:6], gripper_pose[3:7]).astype(np.float32)
                    local_error = np.asarray(row.get("grasp_probe_pre_true_error_t", []), dtype=np.float32).reshape(-1)
                    post_error = np.asarray(row.get("grasp_probe_post_true_error_t", []), dtype=np.float32).reshape(-1)
                    horizon_error = np.asarray(row.get("grasp_probe_horizon_final_true_error_t", []), dtype=np.float32).reshape(-1)
                    xy_error = float(np.linalg.norm(local_error[:2])) if local_error.size >= 2 and np.all(np.isfinite(local_error[:2])) else float("nan")
                    label_available = bool(local_error.size >= 2 and np.all(np.isfinite(local_error[:2])))
                    planner_xy, planner_yaw_abs, planner_dyaw, planner_score = planner_bias_xyyaw(planner_prior_local[:6])
                    record = {
                        "schema_version": "c2c_v2_xy_spatial_temporal_row_v1",
                        "task_name": str(row.get("task_name", "")),
                        "episode_idx": int(ep_idx),
                        "step_idx": int(step_idx),
                        "source_eval_root": str(eval_root),
                        "sequence_id": f"{eval_root}::ep{int(ep_idx):03d}",
                        "trace_path": str(trace_path),
                        "runtime_obs_path": str(obs_path),
                        "npz_path": str(obs_path),
                        "source_trace_path": str(row.get("source_trace_path", trace_path)),
                        "source_episode_target_pose_available": bool(target_pose is not None),
                        "bucket": _row_bucket(row),
                        "observability_bucket": _row_observability_bucket(row),
                        "failure_bucket": _row_bucket(row),
                        "alias_drift_decision": str(row.get("alias_drift_decision", row.get("yaw_alias_drift_decision", "unknown")) or "unknown"),
                        "grasp_probe_active": bool(row.get("grasp_probe_active", False)),
                        "label_available": bool(label_available),
                        "uses_privileged_label": True,
                        "uses_privileged_runtime": False,
                        "roi_box": list(map(int, roi_box_from_action_prior(wrist_rgb.shape, planner_prior_world[:6], gripper_pose, int(args.crop_size)))),
                        "planner_prior_world_6d": planner_prior_world[:6].astype(float).tolist(),
                        "planner_prior_local_6d": planner_prior_local[:6].astype(float).tolist(),
                        "planner_prior_xy_norm": float(planner_xy),
                        "planner_prior_dyaw_abs": float(planner_yaw_abs),
                        "planner_prior_score": float(planner_score),
                        "proprio": proprio.astype(float).tolist(),
                        "gripper_pose": gripper_pose.astype(float).tolist(),
                        "wrist_valid_depth_ratio": _safe_float(row.get("wrist_valid_depth_ratio"), float(np.mean(np.isfinite(wrist_depth) & (wrist_depth > 0.0)))),
                        "wrist_depth_near_fraction": _safe_float(row.get("wrist_depth_near_fraction"), float(np.mean(wrist_depth <= 0.20))),
                        "wrist_is_occluded": _safe_bool(row.get("wrist_is_occluded"), False),
                        "wrist_is_low_visibility": _safe_bool(row.get("wrist_is_low_visibility"), False),
                        "c2c_gate_localizer_visible": _safe_bool(row.get("c2c_gate_localizer_visible"), False),
                        "c2c_gate_depth_nearfield": _safe_bool(row.get("c2c_gate_depth_nearfield"), False),
                        "c2c_gate_target_xy_error": _safe_float(row.get("c2c_gate_target_xy_error"), float("nan")),
                        "local_geometry_error": _jsonable(row.get("local_geometry_error", {})),
                        "estimated_basin_error": _jsonable(row.get("estimated_basin_error", {})),
                        "runtime_xy_estimator": _jsonable(row.get("runtime_xy_estimator", {})),
                        "xy_direction_confidence": _safe_float(row.get("xy_direction_confidence"), 0.0),
                        "xy_sign_stability": _safe_float(row.get("xy_sign_stability"), 0.0),
                        "xy_step_scale": _safe_float(row.get("xy_step_scale"), 1.0),
                        "xy_risk_reason": str(row.get("xy_risk_reason", "")),
                        "xy_stall_reason": str(row.get("xy_stall_reason", "")),
                        "label_dx": float(local_error[0]) if label_available else float("nan"),
                        "label_dy": float(local_error[1]) if label_available else float("nan"),
                        "label_dz": float(local_error[2]) if label_available and local_error.size >= 3 else float("nan"),
                        "label_dyaw": float(local_error[5]) if label_available and local_error.size >= 6 else float("nan"),
                        "label_norm_xy": float(xy_error),
                        "label_pre_true_error_t": _jsonable(row.get("grasp_probe_pre_true_error_t", [])),
                        "label_post_true_error_t": _jsonable(post_error.tolist() if post_error.size else []),
                        "label_horizon_true_error_t": _jsonable(horizon_error.tolist() if horizon_error.size else []),
                        "label_source": str(row.get("label_source", "")),
                        "step_reward": _safe_float(row.get("reward"), float("nan")),
                        "step_terminate": _safe_bool(row.get("terminate"), False),
                    }
                    records.append(record)
                    failure_bucket_counts[record["failure_bucket"]] += 1
                    observability_bucket_counts[record["observability_bucket"]] += 1
                    source_eval_root_counts[record["source_eval_root"]] += 1

    records.sort(key=lambda r: (int(r["episode_idx"]), int(r["step_idx"])))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary.update(
        {
            "rows": int(len(records)),
            "episodes": int(selected_episode_count),
            "active_rows": int(sum(1 for r in records if bool(r.get("grasp_probe_active", False)))),
            "label_available_rows": int(sum(1 for r in records if bool(r.get("label_available", False)))),
            "failure_bucket_counts": dict(sorted(failure_bucket_counts.items())),
            "observability_bucket_counts": dict(sorted(observability_bucket_counts.items())),
            "source_eval_root_counts": dict(sorted(source_eval_root_counts.items())),
            "skipped_episodes": int(sum(skipped_episode_reasons.values())),
            "skipped_episode_reasons": dict(sorted(skipped_episode_reasons.items())),
        }
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
