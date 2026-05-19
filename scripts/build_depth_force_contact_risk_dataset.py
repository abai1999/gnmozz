#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_depth_force_contact_force_signals import load_trace_invalid_map, labels_near_event, pose_motion_deltas, rowwise_delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--trace_dir", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--contact_force_threshold", type=float, default=0.05)
    ap.add_argument("--high_force_threshold", type=float, default=0.5)
    ap.add_argument("--force_spike_threshold", type=float, default=0.05)
    ap.add_argument("--near_depth_threshold", type=float, default=0.08)
    ap.add_argument("--motion_stall_pos_threshold", type=float, default=0.0015)
    ap.add_argument("--motion_stall_rot_threshold_deg", type=float, default=2.0)
    ap.add_argument("--action_range_trans_threshold", type=float, default=0.008)
    ap.add_argument("--action_range_rot_threshold", type=float, default=0.010)
    ap.add_argument("--invalid_near_window", type=int, default=8)
    args = ap.parse_args()

    data = np.load(args.input_npz, allow_pickle=False)
    episodes = np.asarray(data["episode_index"], dtype=np.int64)
    steps = np.asarray(data.get("step_index", data.get("rollout_step")), dtype=np.int64)
    force_hist = np.asarray(data.get("force_history", data.get("ft_hist")), dtype=np.float32)
    if force_hist.ndim == 2:
        force_hist = force_hist[:, None, :]
    raw_force = np.asarray(data.get("gripper_touch_forces", force_hist[:, -1, :]), dtype=np.float32)
    force_xyz = raw_force[:, :3]
    torque_xyz = raw_force[:, 3:6] if raw_force.shape[1] >= 6 else np.zeros_like(force_xyz)
    force_norm = np.linalg.norm(force_xyz, axis=1).astype(np.float32)
    torque_norm = np.linalg.norm(torque_xyz, axis=1).astype(np.float32)
    force_delta_vec = rowwise_delta(raw_force[:, :6], episodes)
    force_delta_norm = np.linalg.norm(force_delta_vec[:, :3], axis=1).astype(np.float32)
    torque_delta_norm = np.linalg.norm(force_delta_vec[:, 3:6], axis=1).astype(np.float32)
    depth_proximity = np.asarray(data.get("depth_proximity", np.full_like(force_norm, np.nan)), dtype=np.float32)
    near_depth = np.isfinite(depth_proximity) & (depth_proximity < float(args.near_depth_threshold))
    motion_delta_pos, motion_delta_rot_deg = pose_motion_deltas(np.asarray(data.get("current_pose_7d", np.zeros((force_norm.shape[0], 7), dtype=np.float32))), episodes)
    contact_label = (force_norm > float(args.contact_force_threshold)) | (force_delta_norm > float(args.force_spike_threshold))
    force_spike_label = force_delta_norm > float(args.force_spike_threshold)
    high_force_label = force_norm > float(args.high_force_threshold)
    motion_stall_label = contact_label & (motion_delta_pos < float(args.motion_stall_pos_threshold)) & (
        motion_delta_rot_deg < float(args.motion_stall_rot_threshold_deg)
    )
    jam_label = high_force_label | motion_stall_label
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    invalid_map = load_trace_invalid_map(trace_dir)
    invalid_now = np.asarray([invalid_map.get((int(ep), int(step)), False) for ep, step in zip(episodes, steps)], dtype=bool)
    invalid_near = labels_near_event(episodes, steps, invalid_map, int(args.invalid_near_window))
    planner_base = np.asarray(
        data.get("planner_base_action_local", data.get("planner_base_action_local_raw", np.zeros((force_norm.shape[0], 6), dtype=np.float32))),
        dtype=np.float32,
    )
    planner_trans_norm = np.linalg.norm(planner_base[:, :3], axis=1).astype(np.float32)
    planner_rot_norm = np.linalg.norm(planner_base[:, 3:6], axis=1).astype(np.float32)
    action_range_invalid = invalid_now & (
        (planner_trans_norm >= float(args.action_range_trans_threshold))
        | (planner_rot_norm >= float(args.action_range_rot_threshold))
    )
    kinematic_invalid = invalid_now & ~contact_label
    safe_motion = ~(contact_label | invalid_now | motion_stall_label)

    risk_label = np.zeros((force_norm.shape[0],), dtype=np.int64)
    risk_label[near_depth] = 1
    risk_label[contact_label] = 2
    risk_label[jam_label | invalid_near] = 3

    out = {
        "front_rgb": np.asarray(data["front_rgb"], dtype=np.uint8),
        "wrist_rgb": np.asarray(data.get("wrist_rgb", data["front_rgb"]), dtype=np.uint8),
        "wrist_depth": np.asarray(data["wrist_depth"], dtype=np.float32),
        "proprio": np.asarray(data["proprio"], dtype=np.float32),
        "planner_base_action_local": planner_base[:, :6],
        "executed_action_local": np.asarray(data.get("executed_action_local", data.get("base_action")), dtype=np.float32)[:, :6],
        "force_history": force_hist.astype(np.float32),
        "gripper_touch_forces": raw_force.astype(np.float32),
        "force_norm": force_norm,
        "torque_norm": torque_norm,
        "force_delta_norm": force_delta_norm,
        "torque_delta_norm": torque_delta_norm,
        "depth_proximity": depth_proximity.astype(np.float32),
        "motion_delta_pos": motion_delta_pos.astype(np.float32),
        "motion_delta_rot_deg": motion_delta_rot_deg.astype(np.float32),
        "gripper_state": np.asarray(data.get("gripper_state", data.get("rollout_gripper_open")), dtype=np.float32),
        "stage_token": np.asarray(data.get("stage_token", data.get("substage_id", np.zeros_like(episodes))), dtype=np.int64),
        "episode_index": episodes,
        "step_index": steps,
        "contact_label": contact_label.astype(np.float32),
        "force_spike_label": force_spike_label.astype(np.float32),
        "high_force_label": high_force_label.astype(np.float32),
        "jam_label": jam_label.astype(np.float32),
        "motion_stall_label": motion_stall_label.astype(np.float32),
        "near_depth_label": near_depth.astype(np.float32),
        "invalid_action_label": invalid_now.astype(np.float32),
        "invalid_action_nearby_label": invalid_near.astype(np.float32),
        "kinematic_invalid_label": kinematic_invalid.astype(np.float32),
        "action_range_invalid_label": action_range_invalid.astype(np.float32),
        "safe_motion_label": safe_motion.astype(np.float32),
        "contact_risk": risk_label,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "depth_force_contact_risk_dataset.npz"
    np.savez_compressed(out_path, **out)

    per_episode = []
    for ep in np.unique(episodes):
        idx = episodes == ep
        per_episode.append(
            {
                "episode": int(ep),
                "rows": int(np.sum(idx)),
                "contact_label_count": int(np.sum(contact_label[idx])),
                "force_spike_count": int(np.sum(force_spike_label[idx])),
                "high_force_count": int(np.sum(high_force_label[idx])),
                "motion_stall_count": int(np.sum(motion_stall_label[idx])),
                "jam_count": int(np.sum(jam_label[idx])),
                "invalid_action_count": int(np.sum(invalid_now[idx])),
                "invalid_action_nearby_count": int(np.sum(invalid_near[idx])),
                "kinematic_invalid_count": int(np.sum(kinematic_invalid[idx])),
                "action_range_invalid_count": int(np.sum(action_range_invalid[idx])),
                "risk_counts": {str(i): int(np.sum(risk_label[idx] == i)) for i in range(4)},
                "force_norm_max": float(np.max(force_norm[idx])),
                "force_delta_norm_max": float(np.max(force_delta_norm[idx])),
                "motion_delta_pos_mean": float(np.mean(motion_delta_pos[idx])),
                "motion_delta_rot_deg_mean": float(np.mean(motion_delta_rot_deg[idx])),
            }
        )
    report = {
        "input_npz": str(args.input_npz),
        "output_npz": str(out_path),
        "rows": int(force_norm.shape[0]),
        "episodes": int(np.unique(episodes).size),
        "contact_positive_rate": float(np.mean(contact_label)),
        "force_spike_positive_rate": float(np.mean(force_spike_label)),
        "high_force_positive_rate": float(np.mean(high_force_label)),
        "motion_stall_positive_rate": float(np.mean(motion_stall_label)),
        "jam_positive_rate": float(np.mean(jam_label)),
        "invalid_action_positive_rate": float(np.mean(invalid_now)),
        "invalid_action_nearby_positive_rate": float(np.mean(invalid_near)),
        "kinematic_invalid_positive_rate": float(np.mean(kinematic_invalid)),
        "action_range_invalid_positive_rate": float(np.mean(action_range_invalid)),
        "risk_counts": {str(i): int(np.sum(risk_label == i)) for i in range(4)},
        "force_norm_mean": float(np.mean(force_norm)),
        "force_norm_p95": float(np.percentile(force_norm, 95)),
        "force_delta_norm_p95": float(np.percentile(force_delta_norm, 95)),
        "motion_delta_pos_mean": float(np.mean(motion_delta_pos)),
        "motion_delta_rot_deg_mean": float(np.mean(motion_delta_rot_deg)),
        "per_episode": per_episode,
    }
    (out_dir / "depth_force_contact_risk_dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
