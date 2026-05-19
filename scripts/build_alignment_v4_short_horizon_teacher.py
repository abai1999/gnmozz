#!/usr/bin/env python3
"""Build a contract-matched V4 short-horizon teacher dataset.

This is a conservative first implementation: it normalizes compatible V3 or
residual-policy rollouts into a V4 training schema and keeps the contract
explicit in the saved npz/report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: np.asarray(raw[k]) for k in raw.files}


def _first_present(data: dict[str, np.ndarray], keys: tuple[str, ...], fallback=None):
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return fallback


def _row_delta(data: dict[str, np.ndarray], idx: int, keys: tuple[str, ...]) -> np.ndarray:
    arr = _first_present(data, keys)
    if arr is None:
        return np.zeros((6,), dtype=np.float32)
    return np.asarray(arr[idx], dtype=np.float32).reshape(-1)[:6].copy()


def _row_scalar(data: dict[str, np.ndarray], idx: int, keys: tuple[str, ...], default: float = 0.0) -> float:
    arr = _first_present(data, keys)
    if arr is None:
        return float(default)
    try:
        return float(np.asarray(arr[idx]).reshape(()))
    except Exception:
        return float(default)


def _row_vec(data: dict[str, np.ndarray], idx: int, keys: tuple[str, ...], fallback_shape: tuple[int, ...]) -> np.ndarray:
    arr = _first_present(data, keys)
    if arr is None:
        return np.zeros(fallback_shape, dtype=np.float32)
    out = np.asarray(arr[idx], dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--output_name", type=str, default="alignment_v4_short_horizon_teacher.npz")
    ap.add_argument("--stage_bucket_filter", type=str, default="near_alignment,micro_contact_refine")
    ap.add_argument("--noop_pos_epsilon", type=float, default=1e-4)
    ap.add_argument("--noop_yaw_epsilon", type=float, default=1e-4)
    args = ap.parse_args()

    data = _load_npz(args.input_npz)
    n = int(_first_present(data, ("current_to_target_delta_local", "runtime_current_to_target_delta_local", "episode_index")).shape[0])
    stage_buckets = np.asarray(data.get("stage_bucket", np.array(["unknown"] * n)), dtype=object)
    allow_buckets = {b.strip() for b in str(args.stage_bucket_filter).split(",") if b.strip()}
    mask = np.array([str(b) in allow_buckets for b in stage_buckets], dtype=bool) if stage_buckets.size == n else np.ones((n,), dtype=bool)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError("no rows survived stage_bucket_filter")

    current = np.stack([_row_delta(data, int(i), ("current_to_target_delta_local", "runtime_current_to_target_delta_local", "teacher_current_to_target_delta_local")) for i in idx], axis=0)
    teacher = np.stack([_row_delta(data, int(i), ("teacher_current_to_target_delta_local", "privileged_current_to_target_delta_local", "target_residual_source_delta_local")) for i in idx], axis=0)
    runtime = np.stack([_row_delta(data, int(i), ("runtime_current_to_target_delta_local", "current_to_target_delta_local", "teacher_current_to_target_delta_local")) for i in idx], axis=0)
    residual = current - teacher
    residual_4d = np.stack([residual[:, 0], residual[:, 1], residual[:, 2], residual[:, 5]], axis=-1).astype(np.float32)
    residual_6d = np.zeros((idx.size, 6), dtype=np.float32)
    residual_6d[:, :3] = residual[:, :3]
    residual_6d[:, 5] = residual[:, 5]

    if "target_post_xy_error" in data:
        target_post_xy = np.asarray(data["target_post_xy_error"], dtype=np.float32)[idx]
    elif "alignment_v4_target_post_xy_error" in data:
        target_post_xy = np.asarray(data["alignment_v4_target_post_xy_error"], dtype=np.float32)[idx]
    else:
        target_post_xy = np.abs(current[:, 0])

    if "target_post_z_error" in data:
        target_post_z = np.asarray(data["target_post_z_error"], dtype=np.float32)[idx]
    elif "alignment_v4_target_post_z_error" in data:
        target_post_z = np.asarray(data["alignment_v4_target_post_z_error"], dtype=np.float32)[idx]
    else:
        target_post_z = np.abs(current[:, 2])

    if "target_post_yaw_error" in data:
        target_post_yaw = np.asarray(data["target_post_yaw_error"], dtype=np.float32)[idx]
    elif "alignment_v4_target_post_yaw_error" in data:
        target_post_yaw = np.asarray(data["alignment_v4_target_post_yaw_error"], dtype=np.float32)[idx]
    else:
        target_post_yaw = np.abs(current[:, 5])

    current_xy = np.linalg.norm(current[:, :2], axis=1)
    current_z = np.abs(current[:, 2])
    current_yaw = np.abs(current[:, 5])
    target_improves_xy = (target_post_xy < current_xy).astype(np.float32)
    target_improves_z = (target_post_z < current_z).astype(np.float32)
    target_improves_yaw = (target_post_yaw < current_yaw).astype(np.float32)

    invalid_risk_proxy = np.asarray(_first_present(data, ("invalid_risk_proxy", "teacher_invalid_risk_proxy"), np.zeros((n,), dtype=np.float32)), dtype=np.float32)[idx]
    overshoot_proxy = np.asarray(_first_present(data, ("overshoot_proxy", "teacher_overshoot_proxy"), np.zeros((n,), dtype=np.float32)), dtype=np.float32)[idx]
    sample_weight = np.asarray(_first_present(data, ("sample_weight",), np.ones((n,), dtype=np.float32)), dtype=np.float32)[idx]
    sample_weight = sample_weight * (1.0 + 2.0 * (target_improves_xy * target_improves_z * target_improves_yaw) + 0.5 * np.asarray(_first_present(data, ("alignment_v4_focus_mask", "focus_mask"), np.zeros((n,), dtype=np.float32)), dtype=np.float32)[idx])

    policy_mode = np.zeros((idx.size,), dtype=np.int64)
    noop = (np.linalg.norm(residual_6d[:, :3], axis=1) <= float(args.noop_pos_epsilon)) & (np.abs(residual_6d[:, 5]) <= float(args.noop_yaw_epsilon))
    default_closeability = ((target_improves_xy > 0.5) | (target_improves_z > 0.5) | (target_improves_yaw > 0.5)).astype(np.float32)
    closeability = np.asarray(_first_present(data, ("alignment_v4_closeability_label", "closeability_label"), default_closeability), dtype=np.float32)[idx]
    policy_mode[(~noop)] = 1
    policy_mode[(noop) & (closeability > 0.5)] = 2

    out: dict[str, np.ndarray] = {}
    for key in ("front_rgb", "wrist_rgb", "wrist_depth", "proprio", "gripper_context", "has_object_in_hand", "substage_id", "contact_state", "stage_target_mode", "depth_proximity", "planner_close_intent", "planner_close_intent_strength", "episode_index", "step_index"):
        if key in data:
            out[key] = np.asarray(data[key])[idx]
    out["stage_bucket"] = stage_buckets[idx].astype("U64")
    out["current_to_target_delta_local"] = current.astype(np.float32)
    out["runtime_current_to_target_delta_local"] = runtime.astype(np.float32)
    out["teacher_current_to_target_delta_local"] = teacher.astype(np.float32)
    out["target_residual_local_4d"] = residual_4d
    out["target_residual_local_6d"] = residual_6d
    out["target_post_xy_error"] = np.asarray(target_post_xy, dtype=np.float32)
    out["target_post_z_error"] = np.asarray(target_post_z, dtype=np.float32)
    out["target_post_yaw_error"] = np.asarray(target_post_yaw, dtype=np.float32)
    out["current_xy_error"] = current_xy.astype(np.float32)
    out["current_z_error"] = current_z.astype(np.float32)
    out["current_yaw_error"] = current_yaw.astype(np.float32)
    out["target_improves_xy"] = target_improves_xy
    out["target_improves_z"] = target_improves_z
    out["target_improves_yaw"] = target_improves_yaw
    out["invalid_risk_proxy"] = invalid_risk_proxy.astype(np.float32)
    out["overshoot_proxy"] = overshoot_proxy.astype(np.float32)
    out["policy_mode_label"] = policy_mode
    out["sample_weight"] = sample_weight.astype(np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / args.output_name
    np.savez_compressed(output_npz, **out)

    residual_pos_norm = np.linalg.norm(residual_6d[:, :3], axis=1)
    report = {
        "input_npz": str(args.input_npz),
        "output_npz": str(output_npz),
        "rows": int(idx.size),
        "input_rows": int(n),
        "stage_bucket_filter": sorted(allow_buckets),
        "stage_bucket_hist": {str(k): int(v) for k, v in zip(*np.unique(stage_buckets[idx], return_counts=True))},
        "target_improves_xy_rate": float(np.mean(target_improves_xy)),
        "target_improves_z_rate": float(np.mean(target_improves_z)),
        "target_improves_yaw_rate": float(np.mean(target_improves_yaw)),
        "noop_rate": float(np.mean(noop)),
        "residual_pos_norm_mean": float(np.mean(residual_pos_norm)),
        "residual_yaw_abs_mean": float(np.mean(np.abs(residual_6d[:, 5]))),
    }
    (output_dir / "alignment_v4_short_horizon_teacher_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
