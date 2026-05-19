#!/usr/bin/env python3
"""Legacy builder for alignment diffusion rows.

This script was used during the support-row era.  The new mainline should use
the raw near-contact collector path and emit the final schema directly.

We keep the runtime contract non-privileged by using only the observed rollout
state, planner action chunk, and learned target pose already present in the
support rows.  The target residual trajectory is a short closed-loop local servo
rollout toward the learned motion target pose.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from build_pose_candidate_dataset import apply_local_offset_to_pose, pose_delta_local_between


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _concat_input_npzs(paths: list[Path]) -> dict[str, np.ndarray]:
    merged: dict[str, list[np.ndarray]] = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            merged.setdefault(key, []).append(np.asarray(data[key]))
    return {key: np.concatenate(values, axis=0) for key, values in merged.items()}


def _as_float(arr, default=0.0) -> float:
    try:
        return float(np.asarray(arr).reshape(()))
    except Exception:
        return float(default)


def _row_or_default(raw: dict[str, np.ndarray], key: str, idx: int, default):
    if key not in raw:
        return default
    arr = raw[key]
    try:
        return arr[idx]
    except Exception:
        return default


def _ensure_rgb(arr: np.ndarray | None, n: int) -> np.ndarray | None:
    if arr is None:
        return None
    out = np.asarray(arr)
    if out.ndim == 3:
        out = out[None, ...]
    if out.ndim == 4 and out.shape[-1] == 3:
        out = np.transpose(out, (0, 3, 1, 2))
    if float(out.max()) > 1.5:
        out = out.astype(np.float32) / 255.0
    return out.astype(np.float32)


def _bucket_from_delta(delta: np.ndarray) -> str:
    delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    cur_xy = float(np.linalg.norm(delta[:2]))
    cur_z = float(abs(delta[2]))
    cur_yaw = float(abs(delta[5]))
    if cur_xy < 0.020 and cur_z < 0.050 and cur_yaw < 0.16:
        return "micro_contact_refine"
    if cur_xy < 0.060 and cur_z < 0.120 and cur_yaw < 0.30:
        return "near_alignment"
    if cur_xy < 0.16 and cur_z < 0.30:
        return "mid_approach_assist"
    return "far_coarse_approach"


def _make_servo_traj(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    horizon: int,
    k_xy: float,
    k_z: float,
    k_yaw: float,
    max_pos: float,
    max_yaw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cur_pose = np.asarray(current_pose, dtype=np.float32).reshape(7)
    target_pose = np.asarray(target_pose, dtype=np.float32).reshape(7)
    traj = np.zeros((horizon, 4), dtype=np.float32)
    traj6 = np.zeros((horizon, 6), dtype=np.float32)
    posts = np.zeros((horizon, 3), dtype=np.float32)
    for h in range(horizon):
        delta = pose_delta_local_between(cur_pose, target_pose).astype(np.float32)
        step = np.zeros(6, dtype=np.float32)
        step[:2] = delta[:2] * float(k_xy)
        step[2] = delta[2] * float(k_z)
        step[5] = delta[5] * float(k_yaw)
        pos_norm = float(np.linalg.norm(step[:3]))
        if pos_norm > max_pos and pos_norm > 1e-8:
            step[:3] *= float(max_pos / pos_norm)
        yaw_abs = float(abs(step[5]))
        if yaw_abs > max_yaw and yaw_abs > 1e-8:
            step[5] *= float(max_yaw / yaw_abs)
        traj[h] = np.asarray([step[0], step[1], step[2], step[5]], dtype=np.float32)
        traj6[h] = step
        next_pose = apply_local_offset_to_pose(cur_pose, step)
        post = pose_delta_local_between(next_pose, target_pose).astype(np.float32)
        posts[h] = np.asarray([float(np.linalg.norm(post[:2])), float(abs(post[2])), float(abs(post[5]))], dtype=np.float32)
        cur_pose = next_pose
    return traj, traj6, posts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--stage_buckets", type=str, default="near_alignment,micro_contact_refine,mid_approach_assist")
    ap.add_argument("--near_xy_threshold", type=float, default=0.06)
    ap.add_argument("--near_z_threshold", type=float, default=0.12)
    ap.add_argument("--near_yaw_threshold", type=float, default=0.30)
    ap.add_argument("--micro_xy_threshold", type=float, default=0.020)
    ap.add_argument("--micro_z_threshold", type=float, default=0.050)
    ap.add_argument("--micro_yaw_threshold", type=float, default=0.16)
    ap.add_argument("--k_xy", type=float, default=0.18)
    ap.add_argument("--k_z", type=float, default=0.12)
    ap.add_argument("--k_yaw", type=float, default=0.10)
    ap.add_argument("--max_pos_step", type=float, default=0.0015)
    ap.add_argument("--max_yaw_step", type=float, default=0.0060)
    ap.add_argument("--min_depth_proximity", type=float, default=-1.0)
    ap.add_argument("--min_rows", type=int, default=64)
    args = ap.parse_args()

    input_paths = [Path(p) for p in args.input_npz]
    raw = _concat_input_npzs(input_paths)
    if "current_pose_7d" not in raw or "motion_target_pose_7d" not in raw:
        raise SystemExit("input npz must contain current_pose_7d and motion_target_pose_7d")

    stage_filter = {s.strip() for s in str(args.stage_buckets).split(",") if s.strip()}
    n = int(raw["current_pose_7d"].shape[0])

    front_rgb = _ensure_rgb(raw["front_rgb"], n) if "front_rgb" in raw else None
    wrist_rgb = _ensure_rgb(raw["wrist_rgb"], n) if "wrist_rgb" in raw else None
    force_history_src = raw["force_history"] if "force_history" in raw else raw.get("ft_hist", None)
    if force_history_src is None:
        force_history_src = np.zeros((n, 32, 6), dtype=np.float32)
    force_history_src = np.asarray(force_history_src)

    rows = []
    for i in range(n):
        current_pose = np.asarray(raw["current_pose_7d"][i], dtype=np.float32).reshape(7)
        target_pose = np.asarray(raw["motion_target_pose_7d"][i], dtype=np.float32).reshape(7)
        delta = pose_delta_local_between(current_pose, target_pose).astype(np.float32)
        bucket = _bucket_from_delta(delta)
        if bucket not in stage_filter:
            continue
        depth_prox = _as_float(raw.get("depth_proximity", np.asarray(-1.0, dtype=np.float32))[i], -1.0) if "depth_proximity" in raw else -1.0
        if args.min_depth_proximity >= 0.0 and depth_prox >= 0.0 and depth_prox > args.min_depth_proximity:
            continue
        traj4d, traj6d, posts = _make_servo_traj(
            current_pose,
            target_pose,
            horizon=int(args.horizon),
            k_xy=float(args.k_xy),
            k_z=float(args.k_z),
            k_yaw=float(args.k_yaw),
            max_pos=float(args.max_pos_step),
            max_yaw=float(args.max_yaw_step),
        )
        current_xy = float(np.linalg.norm(delta[:2]))
        current_z = float(abs(delta[2]))
        current_yaw = float(abs(delta[5]))
        first_post = posts[0]
        progress_label = np.asarray(
            [float(first_post[0] < current_xy), float(first_post[1] < current_z), float(first_post[2] < current_yaw)],
            dtype=np.float32,
        )
        risk_label = np.asarray(
            float(
                _as_float(_row_or_default(raw, "invalid_after_trigger", i, 0.0), 0.0) > 0.5
                or _as_float(_row_or_default(raw, "teacher_invalid", i, 0.0), 0.0) > 0.5
                or _as_float(_row_or_default(raw, "overshoot_proxy", i, 0.0), 0.0) > 0.5
            ),
            dtype=np.float32,
        ).reshape(1)
        stop_label = np.asarray(
            float(current_xy < 0.006 and current_z < 0.010 and current_yaw < 0.04),
            dtype=np.float32,
        ).reshape(1)
        planner_action_src = _row_or_default(raw, "planner_base_action_local", i, None)
        if planner_action_src is None:
            planner_action_src = _row_or_default(raw, "base_action", i, np.zeros((6,), dtype=np.float32))
        planner_action = np.asarray(planner_action_src, dtype=np.float32).reshape(-1)[:6].copy()
        episode_index = int(_as_float(_row_or_default(raw, "episode_index", i, -1), -1))
        if "step_idx" in raw:
            source_step_index = int(_as_float(_row_or_default(raw, "step_idx", i, -1), -1))
        else:
            source_step_index = int(_as_float(_row_or_default(raw, "step_index", i, -1), -1))
        row = {
            "wrist_depth": np.asarray(raw["wrist_depth"][i], dtype=np.float32),
            "force_history": np.asarray(force_history_src[i], dtype=np.float32),
            "proprio": np.asarray(raw["proprio"][i], dtype=np.float32),
            "planner_action_local": planner_action.astype(np.float32),
            "gripper_context": np.asarray(raw.get("gripper_context", np.zeros((4,), dtype=np.float32))[i], dtype=np.float32),
            "residual_trajectory_4d": traj4d.astype(np.float32),
            "residual_trajectory_6d": traj6d.astype(np.float32),
            "progress_label": progress_label,
            "risk_label": risk_label,
            "stop_label": stop_label,
            "bucket_id": np.asarray({"near_alignment": 0, "micro_contact_refine": 1, "mid_approach_assist": 2, "far_coarse_approach": 3}.get(bucket, 3), dtype=np.int64),
            "stage_bucket": np.asarray(bucket),
            "current_to_target_delta_local": delta[:6].astype(np.float32),
            "runtime_current_to_target_delta_local": delta[:6].astype(np.float32),
            "source_episode_index": np.asarray(episode_index, dtype=np.int64),
            "source_step_index": np.asarray(source_step_index, dtype=np.int64),
            "current_xy_error": np.asarray(current_xy, dtype=np.float32),
            "current_z_error": np.asarray(current_z, dtype=np.float32),
            "current_yaw_error": np.asarray(current_yaw, dtype=np.float32),
            "target_post_xy_error": np.asarray(first_post[0], dtype=np.float32),
            "target_post_z_error": np.asarray(first_post[1], dtype=np.float32),
            "target_post_yaw_error": np.asarray(first_post[2], dtype=np.float32),
            "target_improves_xy": np.asarray(float(first_post[0] < current_xy), dtype=np.float32),
            "target_improves_z": np.asarray(float(first_post[1] < current_z), dtype=np.float32),
            "target_improves_yaw": np.asarray(float(first_post[2] < current_yaw), dtype=np.float32),
            "sample_weight": np.asarray(1.0, dtype=np.float32),
        }
        if front_rgb is not None:
            row["front_rgb"] = np.asarray(front_rgb[i], dtype=np.float32)
        if wrist_rgb is not None:
            row["wrist_rgb"] = np.asarray(wrist_rgb[i], dtype=np.float32)
        rows.append(row)

    if len(rows) < int(args.min_rows):
        raise SystemExit(f"too few diffusion rows: {len(rows)} < {args.min_rows}")

    keys = sorted({k for row in rows for k in row.keys()})
    out: dict[str, np.ndarray] = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row]
        out[key] = np.asarray(vals)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    report = {
        "input_npz": [str(p) for p in input_paths],
        "output_npz": str(args.output_npz),
        "rows": int(len(rows)),
        "bucket_histogram": dict(Counter(out["stage_bucket"].astype(str).tolist())),
        "current_xy_error": _stats(out["current_xy_error"]),
        "current_z_error": _stats(out["current_z_error"]),
        "current_yaw_error": _stats(out["current_yaw_error"]),
        "target_post_xy_error": _stats(out["target_post_xy_error"]),
        "target_post_z_error": _stats(out["target_post_z_error"]),
        "target_post_yaw_error": _stats(out["target_post_yaw_error"]),
        "progress_xy_rate": float(np.asarray(out["progress_label"][:, 0], dtype=np.float32).mean()),
        "progress_z_rate": float(np.asarray(out["progress_label"][:, 1], dtype=np.float32).mean()),
        "progress_yaw_rate": float(np.asarray(out["progress_label"][:, 2], dtype=np.float32).mean()),
        "risk_rate": float(np.asarray(out["risk_label"], dtype=np.float32).mean()),
        "stop_rate": float(np.asarray(out["stop_label"], dtype=np.float32).mean()),
        "horizon": int(args.horizon),
        "servo_gain": {"k_xy": float(args.k_xy), "k_z": float(args.k_z), "k_yaw": float(args.k_yaw)},
        "servo_cap": {"max_pos_step": float(args.max_pos_step), "max_yaw_step": float(args.max_yaw_step)},
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
