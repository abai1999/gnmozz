"""
build_mixed_posefield_dataset.py

Mix baseline full-distribution pose-field data with near-ready specialist data.
The output keeps the same row schema expected by train_pose_field_scorer.py and
adds `source_domain` so training can preserve coarse behavior on baseline rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {
    "front_rgb": lambda n: np.zeros((n, 128, 128, 3), dtype=np.uint8),
    "wrist_rgb": lambda n: np.zeros((n, 128, 128, 3), dtype=np.uint8),
    "wrist_depth": lambda n: np.zeros((n, 1, 96, 96), dtype=np.float32),
    "proprio": lambda n: np.zeros((n, 15), dtype=np.float32),
    "base_action": lambda n: np.zeros((n, 6), dtype=np.float32),
    "gripper_context": lambda n: np.zeros((n, 3), dtype=np.float32),
    "planner_close_intent": lambda n: np.zeros((n,), dtype=np.float32),
    "step_idx": lambda n: np.zeros((n,), dtype=np.int64),
    "phase_id": lambda n: np.zeros((n,), dtype=np.int64),
    "episode_index": lambda n: np.zeros((n,), dtype=np.int64),
    "phase_age": lambda n: np.zeros((n,), dtype=np.float32),
    "steps_since_last_replan": lambda n: np.zeros((n,), dtype=np.float32),
    "candidate_actions_local": lambda n: np.zeros((n, 41, 6), dtype=np.float32),
    "candidate_group_index": lambda n: np.zeros((n, 41), dtype=np.int64),
    "candidate_mask": lambda n: np.ones((n, 41), dtype=np.float32),
    "candidate_improvement": lambda n: np.zeros((n, 41), dtype=np.float32),
    "candidate_oracle_score": lambda n: np.zeros((n, 41), dtype=np.float32),
    "candidate_next_basin_distance": lambda n: np.zeros((n, 41), dtype=np.float32),
    "candidate_tier": lambda n: np.zeros((n, 41), dtype=np.int64),
    "candidate_basin_positive": lambda n: np.zeros((n, 41), dtype=np.float32),
    "current_delta_basin_target": lambda n: np.zeros((n, 6), dtype=np.float32),
    "proxy_current_delta_basin_target": lambda n: np.zeros((n, 6), dtype=np.float32),
    "teacher_current_delta_basin_target": lambda n: np.zeros((n, 6), dtype=np.float32),
    "runtime_handoff_metric_xy_error": lambda n: np.full((n,), np.nan, dtype=np.float32),
    "runtime_handoff_metric_abs_z_error": lambda n: np.full((n,), np.nan, dtype=np.float32),
    "runtime_handoff_metric_yaw_error": lambda n: np.full((n,), np.nan, dtype=np.float32),
    "runtime_handoff_metric_valid": lambda n: np.zeros((n,), dtype=np.float32),
    "runtime_handoff_ready": lambda n: np.zeros((n,), dtype=np.float32),
    "current_basin_distance": lambda n: np.zeros((n,), dtype=np.float32),
    "current_dx_sign": lambda n: np.zeros((n,), dtype=np.int64),
    "current_dy_sign": lambda n: np.zeros((n,), dtype=np.int64),
    "current_dyaw_sign": lambda n: np.zeros((n,), dtype=np.int64),
    "basin_distance_bin": lambda n: np.zeros((n,), dtype=np.int64),
    "best_candidate_index": lambda n: np.zeros((n,), dtype=np.int64),
    "best_group_index": lambda n: np.zeros((n,), dtype=np.int64),
    "ready_to_close_target": lambda n: np.zeros((n,), dtype=np.float32),
    "sample_weight": lambda n: np.ones((n,), dtype=np.float32),
    "yaw_hard_negative": lambda n: np.zeros((n,), dtype=np.float32),
    "yaw_hard_positive": lambda n: np.zeros((n,), dtype=np.float32),
    "xy_focus": lambda n: np.zeros((n,), dtype=np.float32),
}


def load_npz(path: str) -> dict:
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def ensure_fields(data: dict, n: int, ref_shapes: dict) -> dict:
    out = {}
    for key, filler in REQUIRED_FIELDS.items():
        if key in data:
            out[key] = data[key]
        else:
            default = filler(n)
            if key == "proxy_current_delta_basin_target" and "current_delta_basin_target" in data:
                default = np.asarray(data["current_delta_basin_target"], dtype=np.float32)
            if key == "teacher_current_delta_basin_target":
                if "target_delta_teacher" in data:
                    default = np.asarray(data["target_delta_teacher"], dtype=np.float32)
                elif "current_delta_basin_target" in data:
                    default = np.asarray(data["current_delta_basin_target"], dtype=np.float32)
            if key in ref_shapes:
                try:
                    default = np.broadcast_to(default, (n,) + tuple(ref_shapes[key][1:])).copy()
                except Exception:
                    pass
            out[key] = default
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_npz", required=True)
    ap.add_argument("--near_ready_npz", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--baseline_sample_weight", type=float, default=1.0)
    ap.add_argument("--near_ready_sample_weight", type=float, default=4.0)
    ap.add_argument("--near_ready_xy_focus_boost", type=float, default=1.0)
    ap.add_argument("--near_ready_yaw_hard_negative_boost", type=float, default=1.0)
    ap.add_argument("--near_ready_yaw_hard_positive_boost", type=float, default=1.0)
    args = ap.parse_args()

    baseline = load_npz(args.baseline_npz)
    near_ready = load_npz(args.near_ready_npz)
    nb = int(baseline["wrist_depth"].shape[0])
    nn = int(near_ready["wrist_depth"].shape[0])
    ref_shapes = {k: np.asarray(v).shape for k, v in baseline.items() if isinstance(v, np.ndarray)}

    baseline = ensure_fields(baseline, nb, ref_shapes)
    near_ready = ensure_fields(near_ready, nn, ref_shapes)

    baseline["sample_weight"] = baseline["sample_weight"].astype(np.float32) * float(args.baseline_sample_weight)
    near_ready["sample_weight"] = near_ready["sample_weight"].astype(np.float32) * float(args.near_ready_sample_weight)
    if "xy_focus" in near_ready:
        near_ready["sample_weight"] *= np.where(
            near_ready["xy_focus"].astype(np.float32) > 0.5,
            float(args.near_ready_xy_focus_boost),
            1.0,
        ).astype(np.float32)
    if "yaw_hard_negative" in near_ready:
        near_ready["sample_weight"] *= np.where(
            near_ready["yaw_hard_negative"].astype(np.float32) > 0.5,
            float(args.near_ready_yaw_hard_negative_boost),
            1.0,
        ).astype(np.float32)
    if "yaw_hard_positive" in near_ready:
        near_ready["sample_weight"] *= np.where(
            near_ready["yaw_hard_positive"].astype(np.float32) > 0.5,
            float(args.near_ready_yaw_hard_positive_boost),
            1.0,
        ).astype(np.float32)
    baseline["source_domain"] = np.zeros((nb,), dtype=np.int64)
    near_ready["source_domain"] = np.ones((nn,), dtype=np.int64)

    out = {}
    for key in list(REQUIRED_FIELDS.keys()) + ["source_domain"]:
        out[key] = np.concatenate([baseline[key], near_ready[key]], axis=0)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    meta = {
        "baseline_npz": args.baseline_npz,
        "near_ready_npz": args.near_ready_npz,
        "num_rows": int(nb + nn),
        "baseline_rows": nb,
        "near_ready_rows": nn,
        "baseline_sample_weight": float(args.baseline_sample_weight),
        "near_ready_sample_weight": float(args.near_ready_sample_weight),
        "near_ready_xy_focus_boost": float(args.near_ready_xy_focus_boost),
        "near_ready_yaw_hard_negative_boost": float(args.near_ready_yaw_hard_negative_boost),
        "near_ready_yaw_hard_positive_boost": float(args.near_ready_yaw_hard_positive_boost),
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
