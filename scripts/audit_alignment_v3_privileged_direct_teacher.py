#!/usr/bin/env python3
"""A/B audit for the privileged direct-control teacher vs baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_alignment_v3_privileged_direct_teacher import _bounded_servo_pseudo, _stats
from build_pose_candidate_dataset import apply_local_offset_to_pose, pose_delta_local_between


def _eval_residuals(current_pose: np.ndarray, target_pose: np.ndarray, residuals: np.ndarray) -> dict[str, np.ndarray]:
    n = int(current_pose.shape[0])
    out = {
        "residual_6d": np.zeros((n, 6), dtype=np.float32),
        "post_xy": np.zeros((n,), dtype=np.float32),
        "post_z": np.zeros((n,), dtype=np.float32),
        "post_yaw": np.zeros((n,), dtype=np.float32),
        "improves_xy": np.zeros((n,), dtype=np.float32),
        "improves_z": np.zeros((n,), dtype=np.float32),
        "improves_yaw": np.zeros((n,), dtype=np.float32),
        "all_improves": np.zeros((n,), dtype=np.float32),
        "overshoot_any": np.zeros((n,), dtype=np.float32),
        "pos_norm": np.zeros((n,), dtype=np.float32),
        "yaw_abs": np.zeros((n,), dtype=np.float32),
    }
    for i in range(n):
        cur = pose_delta_local_between(current_pose[i], target_pose[i]).astype(np.float32)
        nxt = apply_local_offset_to_pose(current_pose[i], residuals[i])
        post = pose_delta_local_between(nxt, target_pose[i]).astype(np.float32)
        cur_xy = float(np.linalg.norm(cur[:2]))
        cur_z = float(abs(cur[2]))
        cur_yaw = float(abs(cur[5]))
        post_xy = float(np.linalg.norm(post[:2]))
        post_z = float(abs(post[2]))
        post_yaw = float(abs(post[5]))
        out["residual_6d"][i] = residuals[i]
        out["post_xy"][i] = post_xy
        out["post_z"][i] = post_z
        out["post_yaw"][i] = post_yaw
        out["improves_xy"][i] = float(post_xy < cur_xy)
        out["improves_z"][i] = float(post_z < cur_z)
        out["improves_yaw"][i] = float(post_yaw < cur_yaw)
        out["all_improves"][i] = out["improves_xy"][i] * out["improves_z"][i] * out["improves_yaw"][i]
        out["overshoot_any"][i] = float(post_xy > cur_xy + 1e-8 or post_z > cur_z + 1e-8 or post_yaw > cur_yaw + 1e-8)
        out["pos_norm"][i] = float(np.linalg.norm(residuals[i, :3]))
        out["yaw_abs"][i] = float(abs(residuals[i, 5]))
    return out


def _row_summary(cur: np.ndarray, post: np.ndarray) -> dict[str, float]:
    cur_xy = np.linalg.norm(cur[:, :2], axis=-1)
    cur_z = np.abs(cur[:, 2])
    cur_yaw = np.abs(cur[:, 5])
    post_xy = np.linalg.norm(post[:, :2], axis=-1)
    post_z = np.abs(post[:, 2])
    post_yaw = np.abs(post[:, 5])
    return {
        "current_xy_mean": float(cur_xy.mean()),
        "current_z_mean": float(cur_z.mean()),
        "current_yaw_mean": float(cur_yaw.mean()),
        "post_xy_mean": float(post_xy.mean()),
        "post_z_mean": float(post_z.mean()),
        "post_yaw_mean": float(post_yaw.mean()),
        "xy_improved_rate": float((post_xy < cur_xy).mean()),
        "z_improved_rate": float((post_z < cur_z).mean()),
        "yaw_improved_rate": float((post_yaw < cur_yaw).mean()),
        "all_improved_rate": float(((post_xy < cur_xy) & (post_z < cur_z) & (post_yaw < cur_yaw)).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_npz", type=Path, required=True)
    ap.add_argument("--teacher_npz", type=Path, required=True)
    ap.add_argument("--output_report", type=Path, required=True)
    args = ap.parse_args()

    src = np.load(args.source_npz, allow_pickle=True)
    tea = np.load(args.teacher_npz, allow_pickle=True)

    src_idx = np.asarray(tea["teacher_source_row_index"], dtype=np.int64)
    current_pose = np.asarray(src["current_pose_7d"], dtype=np.float32)[src_idx]
    target_pose = np.asarray(src["motion_target_pose_7d"], dtype=np.float32)[src_idx]
    cur_delta = np.stack([pose_delta_local_between(c, t) for c, t in zip(current_pose, target_pose)], axis=0).astype(np.float32)

    teacher_res = np.asarray(tea["teacher_residual_local_6d"], dtype=np.float32)
    teacher_eval = _eval_residuals(current_pose, target_pose, teacher_res)

    best_idx = np.asarray(src["best_candidate_index"], dtype=np.int64)[src_idx]
    candidates = np.asarray(src["candidate_actions_local"], dtype=np.float32)[src_idx]
    best_residual = candidates[np.arange(src_idx.size), best_idx]
    best_eval = _eval_residuals(current_pose, target_pose, best_residual)

    servo_residuals = np.zeros_like(teacher_res, dtype=np.float32)
    for i in range(src_idx.size):
        servo_residuals[i], _ = _bounded_servo_pseudo(
            cur_delta[i],
            k_xy=0.08,
            k_z=0.06,
            k_yaw=0.04,
            max_pos=0.0010,
            max_yaw=0.0040,
        )
    servo_eval = _eval_residuals(current_pose, target_pose, servo_residuals)

    noop_residuals = np.zeros_like(teacher_res, dtype=np.float32)
    noop_eval = _eval_residuals(current_pose, target_pose, noop_residuals)

    teacher_bucket = np.asarray(tea["stage_bucket"], dtype=str)
    bucket_keys, bucket_vals = np.unique(teacher_bucket, return_counts=True)
    report = {
        "audit": "alignment_v3_privileged_direct_teacher_ab",
        "source_npz": str(args.source_npz),
        "teacher_npz": str(args.teacher_npz),
        "selected_rows": int(src_idx.size),
        "selected_bucket_histogram": {str(k): int(v) for k, v in zip(bucket_keys.tolist(), bucket_vals.tolist())},
        "teacher": {
            **_row_summary(cur_delta, teacher_res),
            "residual_pos_norm_mean": float(np.linalg.norm(teacher_res[:, :3], axis=-1).mean()),
            "residual_yaw_abs_mean": float(np.abs(teacher_res[:, 5]).mean()),
            "noop_selected_rate": float(np.asarray(tea["teacher_noop_selected"], dtype=np.float32).mean()),
            "overshoot_any_rate": float(np.asarray(tea["teacher_overshoot_any"], dtype=np.float32).mean()),
            "invalid_rate": float(np.asarray(tea["teacher_invalid"], dtype=np.float32).mean()) if "teacher_invalid" in tea.files else 0.0,
            "workspace_violation_rate": float(np.asarray(tea["teacher_workspace_violation"], dtype=np.float32).mean()) if "teacher_workspace_violation" in tea.files else 0.0,
        },
        "best_stage_candidate": _row_summary(cur_delta, best_residual),
        "servo_pseudo": _row_summary(cur_delta, servo_residuals),
        "noop": _row_summary(cur_delta, noop_residuals),
    }

    # Better-than-baseline comparisons.
    teacher_post = np.stack([teacher_eval["post_xy"], teacher_eval["post_z"], teacher_eval["post_yaw"]], axis=-1)
    best_post = np.stack([best_eval["post_xy"], best_eval["post_z"], best_eval["post_yaw"]], axis=-1)
    servo_post = np.stack([servo_eval["post_xy"], servo_eval["post_z"], servo_eval["post_yaw"]], axis=-1)
    noop_post = np.stack([noop_eval["post_xy"], noop_eval["post_z"], noop_eval["post_yaw"]], axis=-1)

    report["comparisons"] = {
        "teacher_beats_beststage_xy_rate": float((teacher_eval["post_xy"] < best_eval["post_xy"]).mean()),
        "teacher_beats_beststage_z_rate": float((teacher_eval["post_z"] < best_eval["post_z"]).mean()),
        "teacher_beats_beststage_yaw_rate": float((teacher_eval["post_yaw"] < best_eval["post_yaw"]).mean()),
        "teacher_beats_beststage_all_rate": float((teacher_eval["all_improves"] > best_eval["all_improves"]).mean()),
        "teacher_beats_servo_xy_rate": float((teacher_eval["post_xy"] < servo_eval["post_xy"]).mean()),
        "teacher_beats_servo_z_rate": float((teacher_eval["post_z"] < servo_eval["post_z"]).mean()),
        "teacher_beats_servo_yaw_rate": float((teacher_eval["post_yaw"] < servo_eval["post_yaw"]).mean()),
        "teacher_beats_noop_xy_rate": float((teacher_eval["post_xy"] < noop_eval["post_xy"]).mean()),
        "teacher_beats_noop_z_rate": float((teacher_eval["post_z"] < noop_eval["post_z"]).mean()),
        "teacher_beats_noop_yaw_rate": float((teacher_eval["post_yaw"] < noop_eval["post_yaw"]).mean()),
        "teacher_vs_beststage_post_delta_mean": {
            "xy": float((teacher_eval["post_xy"] - best_eval["post_xy"]).mean()),
            "z": float((teacher_eval["post_z"] - best_eval["post_z"]).mean()),
            "yaw": float((teacher_eval["post_yaw"] - best_eval["post_yaw"]).mean()),
        },
        "teacher_vs_servo_post_delta_mean": {
            "xy": float((teacher_eval["post_xy"] - servo_eval["post_xy"]).mean()),
            "z": float((teacher_eval["post_z"] - servo_eval["post_z"]).mean()),
            "yaw": float((teacher_eval["post_yaw"] - servo_eval["post_yaw"]).mean()),
        },
        "teacher_vs_noop_post_delta_mean": {
            "xy": float((teacher_eval["post_xy"] - noop_eval["post_xy"]).mean()),
            "z": float((teacher_eval["post_z"] - noop_eval["post_z"]).mean()),
            "yaw": float((teacher_eval["post_yaw"] - noop_eval["post_yaw"]).mean()),
        },
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
