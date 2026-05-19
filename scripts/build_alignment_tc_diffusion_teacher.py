#!/usr/bin/env python3
"""Build target-conditioned diffusion teacher data from privileged rollout rows.

This builder uses privileged target geometry only to create labels.  Runtime
inputs copied into the output remain non-privileged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prismatic.robot.stage_target_provider import apply_local_offset_to_pose, pose_delta_local_between


BUCKET_IDS = {"near_alignment": 0, "micro_contact_refine": 1, "near_contact_refine": 0, "micro_insert": 1}


def _float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x]


def _first(data, keys, default=None):
    for key in keys:
        if key in data:
            return data[key]
    if default is not None:
        return default
    raise KeyError(f"missing any of keys: {keys}")


def _ensure_rgb_chw(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 4 and out.shape[-1] == 3:
        out = np.transpose(out, (0, 3, 1, 2))
    if out.dtype == np.uint8:
        out = out.astype(np.float32) / 255.0
    return out.astype(np.float32)


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _clip_step(step: np.ndarray, max_pos: float, max_yaw: float) -> np.ndarray:
    out = np.asarray(step, dtype=np.float32).reshape(6).copy()
    pos_norm = float(np.linalg.norm(out[:3]))
    if pos_norm > float(max_pos) and pos_norm > 1e-8:
        out[:3] *= float(max_pos) / pos_norm
    if abs(float(out[5])) > float(max_yaw):
        out[5] = np.sign(out[5]) * float(max_yaw)
    out[3:5] = 0.0
    return out.astype(np.float32)


def _make_candidate_bank(delta: np.ndarray, planner_action: np.ndarray, args) -> tuple[np.ndarray, list[str]]:
    delta = np.asarray(delta, dtype=np.float32).reshape(6)
    planner = np.asarray(planner_action, dtype=np.float32).reshape(-1)[:6]
    candidates = []
    names = []

    def add(name: str, step6):
        candidates.append(_clip_step(np.asarray(step6, dtype=np.float32), args.max_pos_step, args.max_yaw_step))
        names.append(name)

    add("noop", np.zeros(6, dtype=np.float32))
    add("planner_residual", planner)
    servo = np.zeros(6, dtype=np.float32)
    servo[:2] = delta[:2] * float(args.servo_k_xy)
    servo[2] = delta[2] * float(args.servo_k_z)
    servo[5] = delta[5] * float(args.servo_k_yaw)
    add("analytic_servo", servo)
    for mag in args.xy_probe_values:
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                s = np.zeros(6, dtype=np.float32)
                s[axis] = sign * float(mag)
                add(f"xy_axis_{axis}_{sign:+.0f}_{mag:g}", s)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            s = np.zeros(6, dtype=np.float32)
            s[0] = sx * float(mag) / np.sqrt(2.0)
            s[1] = sy * float(mag) / np.sqrt(2.0)
            add(f"xy_diag_{sx:+d}_{sy:+d}_{mag:g}", s)
    for mag in args.z_probe_values:
        for sign in (-1.0, 1.0):
            s = np.zeros(6, dtype=np.float32)
            s[2] = sign * float(mag)
            add(f"z_{sign:+.0f}_{mag:g}", s)
    for mag in args.yaw_probe_values:
        for sign in (-1.0, 1.0):
            s = np.zeros(6, dtype=np.float32)
            s[5] = sign * float(mag)
            add(f"yaw_{sign:+.0f}_{mag:g}", s)
    for theta in np.linspace(0.0, 2.0 * np.pi, int(args.spiral_points), endpoint=False):
        s = np.zeros(6, dtype=np.float32)
        s[0] = np.cos(theta) * float(args.spiral_xy)
        s[1] = np.sin(theta) * float(args.spiral_xy)
        s[2] = -abs(float(args.spiral_z))
        add(f"spiral_{theta:.2f}", s)
    return np.stack(candidates, axis=0).astype(np.float32), names


def _rollout_score(current_pose: np.ndarray, target_pose: np.ndarray, traj: np.ndarray, args) -> tuple[float, np.ndarray, np.ndarray]:
    pose = np.asarray(current_pose, dtype=np.float32).reshape(7)
    posts = []
    for step in np.asarray(traj, dtype=np.float32):
        pose = apply_local_offset_to_pose(pose, step)
        delta = pose_delta_local_between(pose, target_pose)
        posts.append([np.linalg.norm(delta[:2]), abs(delta[2]), abs(delta[5])])
    posts = np.asarray(posts, dtype=np.float32)
    final = posts[-1]
    action_norm = float(np.linalg.norm(traj[:, :3], axis=1).mean() + 0.25 * np.abs(traj[:, 5]).mean())
    overshoot = float(np.max(posts[:, 0]) > args.overshoot_xy or np.max(posts[:, 1]) > args.overshoot_z)
    score = (
        args.w_xy * float(final[0])
        + args.w_z * float(final[1])
        + args.w_yaw * float(final[2])
        + args.w_action * action_norm
        + args.w_overshoot * overshoot
    )
    return float(score), final.astype(np.float32), posts.astype(np.float32)


def _heatmap_from_delta(delta: np.ndarray, size: int, xy_range: float, sigma: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    cx = (float(delta[0]) / max(float(xy_range), 1e-6)) * 0.5 + 0.5
    cy = (float(delta[1]) / max(float(xy_range), 1e-6)) * 0.5 + 0.5
    cx = float(np.clip(cx, 0.0, 1.0)) * (size - 1)
    cy = float(np.clip(cy, 0.0, 1.0)) * (size - 1)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * float(sigma) ** 2))
    return heat.astype(np.float32)[None, ...]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max_pos_step", type=float, default=0.0015)
    ap.add_argument("--max_yaw_step", type=float, default=0.006)
    ap.add_argument("--stage_buckets", type=str, default="near_alignment,micro_contact_refine,near_contact_refine,micro_insert")
    ap.add_argument("--servo_k_xy", type=float, default=0.16)
    ap.add_argument("--servo_k_z", type=float, default=0.12)
    ap.add_argument("--servo_k_yaw", type=float, default=0.10)
    ap.add_argument("--xy_probe_values", type=_float_list, default=[0.00075, 0.00125, 0.0015])
    ap.add_argument("--z_probe_values", type=_float_list, default=[0.0005, 0.0010])
    ap.add_argument("--yaw_probe_values", type=_float_list, default=[0.003, 0.006])
    ap.add_argument("--spiral_points", type=int, default=8)
    ap.add_argument("--spiral_xy", type=float, default=0.0008)
    ap.add_argument("--spiral_z", type=float, default=0.0004)
    ap.add_argument("--heatmap_size", type=int, default=16)
    ap.add_argument("--heatmap_xy_range", type=float, default=0.04)
    ap.add_argument("--heatmap_sigma", type=float, default=1.5)
    ap.add_argument("--confidence_margin_good", type=float, default=0.004)
    ap.add_argument("--confidence_margin_floor", type=float, default=0.00025)
    ap.add_argument("--w_xy", type=float, default=1.0)
    ap.add_argument("--w_z", type=float, default=0.8)
    ap.add_argument("--w_yaw", type=float, default=0.7)
    ap.add_argument("--w_action", type=float, default=0.08)
    ap.add_argument("--w_overshoot", type=float, default=0.5)
    ap.add_argument("--overshoot_xy", type=float, default=0.05)
    ap.add_argument("--overshoot_z", type=float, default=0.10)
    ap.add_argument(
        "--prefer_collected_teacher_trajectory",
        action="store_true",
        default=True,
        help="Use privileged-teacher rollout actions as the direct trajectory label when present.",
    )
    ap.add_argument(
        "--no_prefer_collected_teacher_trajectory",
        dest="prefer_collected_teacher_trajectory",
        action="store_false",
    )
    ap.add_argument("--min_rows", type=int, default=32)
    args = ap.parse_args()

    raw = {}
    for path in [Path(p) for p in args.input_npz]:
        data = np.load(path, allow_pickle=False)
        for key in data.files:
            raw.setdefault(key, []).append(np.asarray(data[key]))
    data = {k: np.concatenate(v, axis=0) for k, v in raw.items()}
    current_pose = np.asarray(_first(data, ("current_pose_7d", "gripper_pose_7d")), dtype=np.float32)
    target_pose = np.asarray(
        _first(data, ("privileged_motion_target_pose_7d", "privileged_target_pose_7d", "motion_target_pose_7d")),
        dtype=np.float32,
    )
    n = int(current_pose.shape[0])
    stage_bucket = np.asarray(data.get("stage_bucket", np.asarray(["near_alignment"] * n)))
    allowed = {s.strip() for s in str(args.stage_buckets).split(",") if s.strip()}
    planner = np.asarray(_first(data, ("planner_action_local", "planner_base_action_local_raw", "base_action"), np.zeros((n, 6), dtype=np.float32)), dtype=np.float32)
    if planner.shape[-1] > 6:
        planner = planner[:, :6]
    collected_teacher_traj4 = data.get("teacher_residual_trajectory_4d", None)
    if collected_teacher_traj4 is not None:
        collected_teacher_traj4 = np.asarray(collected_teacher_traj4, dtype=np.float32)
    collected_teacher_verified = np.asarray(data.get("teacher_grasp_verified", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1) > 0.5
    verified_horizon = np.asarray(
        data.get("teacher_verify_lift_streak", np.full((n,), -1, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    verified_window = np.isfinite(verified_horizon) & (verified_horizon >= 0.0) & (verified_horizon <= 15.0)

    rows = []
    score_margins = []
    collected_teacher_used = 0
    bucket_counts = Counter()
    for i in range(n):
        bucket = str(stage_bucket[i])
        if bucket not in allowed:
            continue
        if "invalid_action" in data and float(np.asarray(data["invalid_action"][i]).reshape(-1)[0]) > 0.5:
            continue
        if "workspace_violation" in data and float(np.asarray(data["workspace_violation"][i]).reshape(-1)[0]) > 1e-6:
            continue
        if not np.isfinite(current_pose[i]).all() or not np.isfinite(target_pose[i]).all():
            continue
        delta = pose_delta_local_between(current_pose[i], target_pose[i]).astype(np.float32)
        if not np.isfinite(delta).all():
            continue
        candidates, names = _make_candidate_bank(delta, planner[i], args)
        trajs = np.repeat(candidates[:, None, :], int(args.horizon), axis=1)
        scores, finals = [], []
        for traj in trajs:
            score, final, _ = _rollout_score(current_pose[i], target_pose[i], traj, args)
            scores.append(score)
            finals.append(final)
        scores = np.asarray(scores, dtype=np.float32)
        finals = np.asarray(finals, dtype=np.float32)
        best = int(np.argmin(scores))
        order = np.argsort(scores)
        margin = float(scores[order[1]] - scores[order[0]]) if scores.size > 1 else 0.0
        score_margins.append(margin)
        cur_err = np.asarray([np.linalg.norm(delta[:2]), abs(delta[2]), abs(delta[5])], dtype=np.float32)
        best_final = finals[best]
        progress = (best_final < cur_err).astype(np.float32)
        risk = np.asarray(float(not np.all(np.isfinite(best_final))), dtype=np.float32).reshape(1)
        stop = np.asarray(float(scores[best] >= scores[0] - 1e-6), dtype=np.float32).reshape(1)
        best_residual_trajectory_4d = np.stack(
            [trajs[best, :, 0], trajs[best, :, 1], trajs[best, :, 2], trajs[best, :, 5]],
            axis=-1,
        ).astype(np.float32)
        if (
            bool(args.prefer_collected_teacher_trajectory)
            and collected_teacher_traj4 is not None
            and i < int(collected_teacher_traj4.shape[0])
            and bool(collected_teacher_verified[i])
        ):
            collected = np.asarray(collected_teacher_traj4[i], dtype=np.float32)
            if collected.ndim == 2 and collected.shape[-1] >= 4 and np.all(np.isfinite(collected[:, :4])):
                collected = collected[:, :4]
                if collected.shape[0] < int(args.horizon):
                    pad = np.repeat(collected[-1:, :], int(args.horizon) - collected.shape[0], axis=0)
                    collected = np.concatenate([collected, pad], axis=0)
                best_residual_trajectory_4d = collected[: int(args.horizon)].astype(np.float32)
                collected_teacher_used += 1
        low_visibility = float(data.get("is_low_visibility", np.zeros((n,), dtype=np.float32))[i]) > 0.5
        occluded = float(data.get("is_occluded", np.zeros((n,), dtype=np.float32))[i]) > 0.5
        confidence = float(
            np.clip(
                (margin - float(args.confidence_margin_floor))
                / max(float(args.confidence_margin_good) - float(args.confidence_margin_floor), 1e-6),
                0.0,
                1.0,
            )
        )
        if low_visibility or occluded:
            confidence *= 0.35
        row = {
            "wrist_depth": np.asarray(data["wrist_depth"][i], dtype=np.float32),
            "force_history": np.asarray(_first(data, ("force_history", "ft_hist"), np.zeros((n, 32, 6), dtype=np.float32))[i], dtype=np.float32),
            "proprio": np.asarray(data["proprio"][i], dtype=np.float32),
            "planner_action_local": np.asarray(planner[i], dtype=np.float32),
            "gripper_context": np.asarray(data.get("gripper_context", np.zeros((n, 4), dtype=np.float32))[i], dtype=np.float32),
            "teacher_target_delta_local_6d": delta.astype(np.float32),
            "contact_heatmap_label": _heatmap_from_delta(delta, args.heatmap_size, args.heatmap_xy_range, args.heatmap_sigma),
            "target_confidence_label": np.asarray(confidence, dtype=np.float32),
            "target_visible_label": np.asarray(
                float(
                    np.isfinite(delta).all()
                    and not low_visibility
                    and not occluded
                ),
                dtype=np.float32,
            ),
            "best_residual_trajectory_4d": best_residual_trajectory_4d.astype(np.float32),
            "candidate_residual_trajectories_4d": np.stack([trajs[:, :, 0], trajs[:, :, 1], trajs[:, :, 2], trajs[:, :, 5]], axis=-1).astype(np.float32),
            "candidate_post_xy": finals[:, 0].astype(np.float32),
            "candidate_post_z": finals[:, 1].astype(np.float32),
            "candidate_post_yaw": finals[:, 2].astype(np.float32),
            "candidate_risk": np.zeros((trajs.shape[0],), dtype=np.float32),
            "candidate_valid_mask": np.ones((trajs.shape[0],), dtype=np.float32),
            "best_candidate_index": np.asarray(best, dtype=np.int64),
            "candidate_score": scores.astype(np.float32),
            "progress_label": progress,
            "risk_label": risk,
            "stop_label": stop,
            "no_op_label": stop.copy(),
            "sample_weight": np.asarray(1.5 if "micro" in bucket else 1.0, dtype=np.float32),
            "stage_bucket": np.asarray(bucket),
            "bucket_id": np.asarray(BUCKET_IDS.get(bucket, 0), dtype=np.int64),
            "teacher_beats_noop": np.asarray(float(scores[best] < scores[0] - 1e-6), dtype=np.float32),
            "teacher_score_margin": np.asarray(margin, dtype=np.float32),
            "candidate_count": np.asarray(trajs.shape[0], dtype=np.int64),
            "best_candidate_name": np.asarray(names[best]),
            "teacher_verified_row": np.asarray(float(bool(collected_teacher_verified[i])), dtype=np.float32),
            "verified_positive_window": np.asarray(float(bool(verified_window[i])), dtype=np.float32),
        }
        if bool(collected_teacher_verified[i]) and bool(verified_window[i]):
            row["sample_weight"] = np.asarray(
                3.0 if "micro" in bucket else 2.0,
                dtype=np.float32,
            )
        elif bool(collected_teacher_verified[i]):
            row["sample_weight"] = np.asarray(
                2.0 if "micro" in bucket else 1.5,
                dtype=np.float32,
            )
        for key in (
            "teacher_gripper_finger_pose_7d",
            "teacher_object_in_finger_region",
            "teacher_finger_object_lateral_error",
            "teacher_finger_object_height_overlap",
            "teacher_finger_object_yaw_error",
            "teacher_grasp_contact_ready",
            "teacher_grasp_ready",
            "teacher_grasp_readiness_score",
            "teacher_grasp_readiness_reason",
            "teacher_attached_after_close",
            "teacher_grasped_object_count",
            "teacher_grasped_target_handle_match",
            "teacher_object_to_gripper_delta_local_6d",
            "teacher_demo_basin_distance",
            "teacher_close_basin_source",
            "teacher_close_failure_reason",
            "expert_sequence_name",
            "expert_sequence_score",
            "expert_sequence_verified",
            "yaw_imitation_enabled",
            "teacher_close_contact_ready",
            "teacher_close_ready_all",
            "teacher_close_gate_reason",
            "teacher_close_action",
            "teacher_grasp_verified",
            "teacher_verify_lift_streak",
            "teacher_close_yaw_raw",
            "teacher_close_yaw_folded",
            "teacher_yaw_control_sign",
        ):
            if key in data:
                row[key] = np.asarray(data[key][i])
        if "alignment_phase" in data:
            row["alignment_phase"] = np.asarray(data["alignment_phase"][i])
        if collected_teacher_traj4 is not None and i < int(collected_teacher_traj4.shape[0]):
            collected_teacher = np.asarray(collected_teacher_traj4[i], dtype=np.float32)
            if collected_teacher.ndim == 2 and collected_teacher.shape[-1] >= 4 and np.all(np.isfinite(collected_teacher[:, :4])):
                row["teacher_residual_trajectory_4d"] = collected_teacher[:, :4].astype(np.float32)
        if "front_rgb" in data:
            row["front_rgb"] = _ensure_rgb_chw(data["front_rgb"][i : i + 1])[0]
        if "wrist_rgb" in data:
            row["wrist_rgb"] = _ensure_rgb_chw(data["wrist_rgb"][i : i + 1])[0]
        rows.append(row)
        bucket_counts[bucket] += 1

    if len(rows) < int(args.min_rows):
        raise SystemExit(f"too few target-conditioned teacher rows: {len(rows)} < {args.min_rows}")
    keys = sorted({k for row in rows for k in row})
    out = {k: np.asarray([row[k] for row in rows if k in row]) for k in keys}
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)
    report = {
        "output_npz": str(args.output_npz),
        "rows": int(len(rows)),
        "bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
        "teacher_beats_noop_rate": float(np.asarray(out["teacher_beats_noop"], dtype=np.float32).mean()),
        "collected_teacher_trajectory_used_rate": float(collected_teacher_used / max(len(rows), 1)),
        "score_margin": _stats(np.asarray(score_margins, dtype=np.float32)),
        "candidate_count": int(out["candidate_count"][0]) if len(rows) else 0,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
