#!/usr/bin/env python3
"""Collect target-conditioned alignment data from a privileged closed-loop expert.

This collector deliberately avoids planner-natural near/micro trajectories.  It
samples privileged near/contact initial states around the commit target, moves
the arm there, then lets a small contact-aware expert run a closed-loop local
alignment policy.  Runtime observations are recorded, but privileged geometry is
used only for labels and audit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("VLA_PLATFORM", "RLBENCH")

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from prismatic.robot.residual_safety import ResidualSafety
from prismatic.robot.residual_transforms import local_delta_to_world
from prismatic.vla.constants import FORCE_DIM

from evaluate_rlbench import (
    ImageSequenceClip,
    TASK_MAP,
    _alignment_diffusion_action4d_from_local,
    _lazy_import_tasks,
    apply_executed_local_delta_to_pose,
    apply_yaw_symmetry_to_delta,
    build_phase1_teacher_targets,
    compute_wrist_visibility_stats,
    delta_to_absolute,
    load_phase1_grasp_spec,
    pose_delta_local_between,
    process_obs,
    resolve_live_target_handle,
    safe_live_target_pose_7d,
    write_rows_npz,
)
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


def _rate(mask) -> float:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    return float(arr.mean()) if arr.size else 0.0


def _stats(values) -> dict:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _bucket_from_delta(delta6: np.ndarray) -> str:
    delta = np.asarray(delta6, dtype=np.float32).reshape(6)
    xy = float(np.linalg.norm(delta[:2]))
    z = float(abs(delta[2]))
    yaw = float(abs(delta[5]))
    if xy <= 0.010 and z <= 0.030 and yaw <= 0.18:
        return "micro_contact_refine"
    if xy <= 0.030 and z <= 0.070 and yaw <= 0.35:
        return "near_contact_refine"
    return "broad_near"


def _sample_initial_error(rng: np.random.Generator, bucket: str) -> np.ndarray:
    if bucket == "micro_contact_refine":
        xy_r = rng.uniform(0.0015, 0.010)
        z = rng.uniform(-0.030, -0.006)
        yaw = rng.uniform(-0.18, 0.18)
    elif bucket == "contact_stall":
        xy_r = rng.uniform(0.002, 0.018)
        z = rng.uniform(-0.018, 0.004)
        yaw = rng.uniform(-0.28, 0.28)
    else:
        xy_r = rng.uniform(0.010, 0.040)
        z = rng.uniform(-0.070, -0.015)
        yaw = rng.uniform(-0.35, 0.35)
    theta = rng.uniform(-np.pi, np.pi)
    out = np.zeros(6, dtype=np.float32)
    out[0] = float(xy_r * np.cos(theta))
    out[1] = float(xy_r * np.sin(theta))
    out[2] = float(z)
    out[5] = float(yaw)
    return out


def _pose_from_target_delta(target_pose_7d: np.ndarray, delta_local_6d: np.ndarray) -> np.ndarray:
    target = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
    delta = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    r_target = Rotation.from_quat(target[3:7])
    r_delta = Rotation.from_rotvec(delta[3:6])
    r_current = r_delta.inv() * r_target
    current = target.copy()
    current[3:7] = r_current.as_quat().astype(np.float32)
    current[:3] = target[:3] - r_current.apply(delta[:3]).astype(np.float32)
    return current.astype(np.float32)


def _absolute_action_from_pose(pose_7d: np.ndarray, gripper_open: float = 1.0) -> np.ndarray:
    pose = np.asarray(pose_7d, dtype=np.float32).reshape(7)
    return np.concatenate([pose, [1.0 if gripper_open > 0.5 else 0.0]]).astype(np.float32)


def _move_to_pose(task, obs, target_pose_7d: np.ndarray, *, steps: int, safety: ResidualSafety, teleport_ik: bool = True):
    start_pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(7)
    target_pose = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
    if teleport_ik:
        try:
            robot = getattr(task, "_robot", None)
            arm = getattr(robot, "arm", None)
            if arm is None:
                task_obj = getattr(task, "_task", None)
                robot = getattr(task_obj, "robot", None) or getattr(task_obj, "_robot", None)
                arm = getattr(robot, "arm", None)
            if arm is not None:
                configs = arm.solve_ik_via_sampling(
                    target_pose[:3],
                    quaternion=target_pose[3:7],
                    ignore_collisions=True,
                    trials=800,
                    max_configs=1,
                    max_time_ms=80,
                )
                if len(configs) > 0:
                    arm.set_joint_positions(configs[0], disable_dynamics=True)
                    scene = getattr(task, "_scene", None)
                    if scene is not None and hasattr(scene, "get_observation"):
                        return scene.get_observation(), 0
                    return obs, 0
        except Exception:
            pass
    rots = Rotation.from_quat(np.stack([start_pose[3:7], target_pose[3:7]], axis=0))
    slerp = Slerp([0.0, 1.0], rots)
    cur_obs = obs
    invalid = 0
    for i in range(1, max(int(steps), 1) + 1):
        t = float(i) / float(max(int(steps), 1))
        pose = np.zeros(7, dtype=np.float32)
        pose[:3] = (1.0 - t) * start_pose[:3] + t * target_pose[:3]
        pose[:3] = safety.clamp_workspace(pose[:3])
        pose[3:7] = slerp([t]).as_quat()[0].astype(np.float32)
        try:
            cur_obs, reward, terminate = task.step(_absolute_action_from_pose(pose, 1.0))
        except InvalidActionError:
            invalid += 1
            continue
    return cur_obs, invalid


def _scene_observation(task):
    scene = getattr(task, "_scene", None)
    if scene is not None and hasattr(scene, "get_observation"):
        return scene.get_observation()
    return None


def _task_arm(task):
    robot = getattr(task, "_robot", None)
    arm = getattr(robot, "arm", None)
    if arm is not None:
        return arm
    scene = getattr(task, "_scene", None)
    robot = getattr(scene, "robot", None) if scene is not None else None
    return getattr(robot, "arm", None)


def _task_gripper(task):
    robot = getattr(task, "_robot", None)
    gripper = getattr(robot, "gripper", None)
    if gripper is not None:
        return gripper
    scene = getattr(task, "_scene", None)
    robot = getattr(scene, "robot", None) if scene is not None else None
    return getattr(robot, "gripper", None)


def _restore_demo_obs(task, demo_obs):
    arm = _task_arm(task)
    if arm is None or getattr(demo_obs, "joint_positions", None) is None:
        return None
    arm.set_joint_positions(np.asarray(demo_obs.joint_positions, dtype=np.float32), disable_dynamics=True)
    gripper = _task_gripper(task)
    grip_joints = getattr(demo_obs, "gripper_joint_positions", None)
    if gripper is not None and grip_joints is not None and hasattr(gripper, "set_joint_positions"):
        try:
            gripper.set_joint_positions(np.asarray(grip_joints, dtype=np.float32), disable_dynamics=True)
        except Exception:
            pass
    return _scene_observation(task)


def _set_ee_pose_with_ik(task, pose_7d: np.ndarray, *, trials: int = 500, max_time_ms: int = 80):
    arm = _task_arm(task)
    if arm is None:
        return None, False
    pose = np.asarray(pose_7d, dtype=np.float32).reshape(7)
    try:
        configs = arm.solve_ik_via_sampling(
            pose[:3],
            quaternion=pose[3:7],
            ignore_collisions=True,
            trials=int(trials),
            max_configs=1,
            max_time_ms=int(max_time_ms),
        )
        if len(configs) <= 0:
            return _scene_observation(task), False
        arm.set_joint_positions(configs[0], disable_dynamics=True)
        return _scene_observation(task), True
    except Exception:
        return _scene_observation(task), False


def _sample_demo_perturb(rng: np.random.Generator, bucket: str, args) -> np.ndarray:
    out = np.zeros(6, dtype=np.float32)
    if bucket == "micro_contact_refine":
        xy_std = float(args.perturb_micro_xy_std)
        z_std = float(args.perturb_micro_z_std)
        yaw_std = float(args.perturb_micro_yaw_std)
    else:
        xy_std = float(args.perturb_near_xy_std)
        z_std = float(args.perturb_near_z_std)
        yaw_std = float(args.perturb_near_yaw_std)
    out[0] = float(rng.normal(0.0, xy_std))
    out[1] = float(rng.normal(0.0, xy_std))
    out[2] = float(rng.normal(0.0, z_std))
    out[5] = float(rng.normal(0.0, yaw_std))
    pos_norm = float(np.linalg.norm(out[:3]))
    max_pos = float(args.perturb_max_pos)
    if pos_norm > max_pos:
        out[:3] *= max_pos / max(pos_norm, 1e-8)
    out[5] = float(np.clip(out[5], -float(args.perturb_max_yaw), float(args.perturb_max_yaw)))
    return out.astype(np.float32)


def _force_norm(raw_force) -> float:
    if raw_force is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(raw_force, dtype=np.float32)[:3]))


def _object_handle(obj) -> int | None:
    if obj is None:
        return None
    try:
        return int(obj.get_handle())
    except Exception:
        return None


def _grasped_target_status(task, target_handle) -> tuple[bool, int]:
    gripper = _task_gripper(task)
    if gripper is None:
        return False, 0
    try:
        grasped = list(gripper.get_grasped_objects())
    except Exception:
        return False, 0
    target_id = _object_handle(target_handle)
    if target_id is None:
        return False, len(grasped)
    for obj in grasped:
        if _object_handle(obj) == target_id:
            return True, len(grasped)
    return False, len(grasped)


def _target_pose_from_handle(target_handle) -> np.ndarray:
    if target_handle is None:
        return np.full((7,), np.nan, dtype=np.float32)
    try:
        return np.asarray(target_handle.get_pose(), dtype=np.float32).reshape(7)
    except Exception:
        return np.full((7,), np.nan, dtype=np.float32)


def _expert_action(delta6: np.ndarray, force_norm: float, step: int, args) -> tuple[np.ndarray, str]:
    delta = apply_yaw_symmetry_to_delta(np.asarray(delta6, dtype=np.float32).reshape(6), np.pi / 2.0)
    xy = float(np.linalg.norm(delta[:2]))
    z = float(abs(delta[2]))
    yaw = float(abs(delta[5]))
    action = np.zeros(6, dtype=np.float32)

    if force_norm >= float(args.jam_force_threshold):
        action[2] = float(args.unjam_lift_step)
        if xy > 1e-6:
            action[:2] = -delta[:2] / max(xy, 1e-6) * float(args.unjam_lateral_step)
        action[5] = -np.sign(delta[5]) * min(float(args.max_yaw_step), float(args.unjam_yaw_step))
        return action.astype(np.float32), "UNJAM"

    if xy > float(args.align_xy_threshold) or yaw > float(args.align_yaw_threshold):
        action[:2] = float(args.k_xy_align) * delta[:2]
        action[2] = np.clip(float(args.k_z_hold) * delta[2], -float(args.align_z_step), float(args.align_z_step))
        action[5] = float(args.expert_yaw_sign) * float(args.k_yaw_align) * delta[5]
        state = "ALIGN_ABOVE"
    elif z > float(args.commit_z_threshold) and force_norm < float(args.light_contact_force):
        action[:2] = float(args.k_xy_descend) * delta[:2]
        action[2] = float(args.k_z_descend) * delta[2]
        action[5] = float(args.expert_yaw_sign) * float(args.k_yaw_descend) * delta[5]
        state = "DESCEND_LIGHT"
    elif xy > float(args.success_xy_threshold) or yaw > float(args.success_yaw_threshold):
        angle = 0.9 * float(step)
        spiral = float(args.spiral_step) * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        action[:2] = float(args.k_xy_contact) * delta[:2] + spiral
        action[2] = np.clip(float(args.k_z_contact) * delta[2], -float(args.contact_z_step), float(args.contact_z_step))
        action[5] = float(args.expert_yaw_sign) * float(args.k_yaw_contact) * delta[5]
        state = "CONTACT_SEARCH"
    else:
        action[:2] = float(args.k_xy_commit) * delta[:2]
        action[2] = float(args.k_z_commit) * delta[2]
        action[5] = float(args.expert_yaw_sign) * float(args.k_yaw_commit) * delta[5]
        state = "COMMIT"

    pos_norm = float(np.linalg.norm(action[:3]))
    max_pos = float(args.max_pos_step)
    if pos_norm > max_pos:
        action[:3] *= max_pos / max(pos_norm, 1e-8)
    action[5] = float(np.clip(action[5], -float(args.max_yaw_step), float(args.max_yaw_step)))
    return action.astype(np.float32), state


def _grasp_recovery_close_ready(delta6: np.ndarray, args) -> bool:
    delta = apply_yaw_symmetry_to_delta(np.asarray(delta6, dtype=np.float32).reshape(6), np.pi / 2.0)
    return bool(
        np.linalg.norm(delta[:2]) <= float(args.grasp_recovery_close_xy_threshold)
        and abs(float(delta[2])) <= float(args.grasp_recovery_close_z_threshold)
        and abs(float(delta[5])) <= float(args.grasp_recovery_close_yaw_threshold)
    )


def _row_from_obs(
    *,
    obs,
    force_buffer,
    raw_force,
    force_hist,
    proprio,
    depth_tensor_96,
    depth_proximity,
    target_pose,
    object_pose,
    delta,
    action_local,
    post_delta,
    state,
    bucket,
    episode_idx,
    step_idx,
    init_bucket,
    phase,
    target_phase: str | None = None,
    invalid,
    workspace_violation,
    reward,
    success,
    args,
) -> dict:
    wrist_depth_np = (
        depth_tensor_96.detach().cpu().numpy().astype(np.float32)
        if depth_tensor_96 is not None
        else np.zeros((1, 96, 96), dtype=np.float32)
    )
    valid_ratio, near_fraction, is_occluded, is_low_visibility = compute_wrist_visibility_stats(wrist_depth_np)
    force_norm = _force_norm(raw_force)
    post = apply_yaw_symmetry_to_delta(np.asarray(post_delta, dtype=np.float32).reshape(6), np.pi / 2.0)
    cur = apply_yaw_symmetry_to_delta(np.asarray(delta, dtype=np.float32).reshape(6), np.pi / 2.0)
    improves_xy = float(np.linalg.norm(post[:2]) < np.linalg.norm(cur[:2]))
    improves_z = float(abs(post[2]) < abs(cur[2]))
    improves_yaw = float(abs(post[5]) < abs(cur[5]))
    phase_name = str(target_phase if target_phase is not None else phase)
    insert_target_pose = (
        np.asarray(target_pose, dtype=np.float32)
        if phase_name == "insert_commit"
        else np.full((7,), np.nan, dtype=np.float32)
    )
    return {
        "episode_index": np.asarray(int(episode_idx), dtype=np.int64),
        "step_index": np.asarray(int(step_idx), dtype=np.int64),
        "task_name": np.asarray(str(args.task_name)),
        "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
        "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
        "wrist_depth": wrist_depth_np.astype(np.float32),
        "force_history": np.asarray(
            force_hist.detach().cpu().numpy().astype(np.float32)
            if force_hist is not None
            else np.zeros((32, FORCE_DIM), dtype=np.float32),
            dtype=np.float32,
        ),
        "proprio": np.asarray(proprio, dtype=np.float32),
        "planner_action_local": np.zeros((7,), dtype=np.float32),
        "gripper_context": np.asarray([float(obs.gripper_open), 1.0, 0.0, 0.0], dtype=np.float32),
        "stage_bucket": np.asarray(str(bucket)),
        "alignment_phase": np.asarray(str(phase)),
        "target_phase": np.asarray(str(phase_name)),
        "expert_state": np.asarray(str(state)),
        "initial_bucket": np.asarray(str(init_bucket)),
        "depth_proximity": np.asarray(float(depth_proximity) if depth_proximity is not None else np.nan, dtype=np.float32),
        "wrist_depth_median": np.asarray(float(np.nanmedian(wrist_depth_np)), dtype=np.float32),
        "wrist_valid_depth_ratio": np.asarray(float(valid_ratio), dtype=np.float32),
        "wrist_depth_near_fraction": np.asarray(float(near_fraction), dtype=np.float32),
        "is_occluded": np.asarray(float(is_occluded), dtype=np.float32),
        "is_low_visibility": np.asarray(float(is_low_visibility), dtype=np.float32),
        "force_norm": np.asarray(float(force_norm), dtype=np.float32),
        "force_spike": np.asarray(float(force_norm >= float(args.force_spike_threshold)), dtype=np.float32),
        "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
        "privileged_current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
        "privileged_motion_target_pose_7d": np.asarray(target_pose, dtype=np.float32),
        "privileged_basin_center_pose_7d": np.asarray(target_pose, dtype=np.float32),
        "privileged_grasp_commit_target_pose_7d": np.asarray(target_pose, dtype=np.float32),
        "privileged_insert_target_pose_7d": np.asarray(insert_target_pose, dtype=np.float32),
        "privileged_object_anchor_pose_7d": np.asarray(object_pose, dtype=np.float32),
        "privileged_current_to_target_delta_local": np.asarray(cur, dtype=np.float32),
        "teacher_target_delta_local_6d": np.asarray(cur, dtype=np.float32),
        "teacher_target_phase": np.asarray(str(phase_name)),
        "teacher_insert_target_pose_7d": np.asarray(insert_target_pose, dtype=np.float32),
        "expert_action_local_6d": np.asarray(action_local, dtype=np.float32),
        "teacher_residual_action_6d": np.asarray(action_local, dtype=np.float32),
        "teacher_residual_action_4d": _alignment_diffusion_action4d_from_local(action_local).astype(np.float32),
        "teacher_post_delta_local_6d": np.asarray(post, dtype=np.float32),
        "post_xy": np.asarray(float(np.linalg.norm(post[:2])), dtype=np.float32),
        "post_z": np.asarray(float(abs(post[2])), dtype=np.float32),
        "post_yaw": np.asarray(float(abs(post[5])), dtype=np.float32),
        "teacher_improves_xy": np.asarray(improves_xy, dtype=np.float32),
        "teacher_improves_z": np.asarray(improves_z, dtype=np.float32),
        "teacher_improves_yaw": np.asarray(improves_yaw, dtype=np.float32),
        "teacher_improves_two_axis": np.asarray(float((improves_xy + improves_z + improves_yaw) >= 2), dtype=np.float32),
        "teacher_all_improves": np.asarray(float((improves_xy + improves_z + improves_yaw) >= 3), dtype=np.float32),
        "risk_label": np.asarray(float(invalid or workspace_violation > 0.0 or force_norm >= float(args.force_spike_threshold)), dtype=np.float32),
        "action_imitation_weight": np.asarray(
            float((not invalid) and workspace_violation <= 0.0 and force_norm < float(args.force_spike_threshold)),
            dtype=np.float32,
        ),
        "stop_label": np.asarray(float(success), dtype=np.float32),
        "success_label": np.asarray(float(success or reward > 0), dtype=np.float32),
        "invalid_action": np.asarray(float(invalid), dtype=np.float32),
        "workspace_violation": np.asarray(float(workspace_violation), dtype=np.float32),
        "reward": np.asarray(float(reward), dtype=np.float32),
    }


def _summarize(rows: list[dict], episode_summaries: list[dict]) -> dict:
    if rows:
        bucket = np.asarray([str(r["stage_bucket"].item()) for r in rows])
        phase = np.asarray([str(r.get("alignment_phase", np.asarray("unknown")).item()) for r in rows])
        state = np.asarray([str(r["expert_state"].item()) for r in rows])
        xy = np.asarray([float(r["teacher_improves_xy"]) for r in rows]) > 0.5
        z = np.asarray([float(r["teacher_improves_z"]) for r in rows]) > 0.5
        yaw = np.asarray([float(r["teacher_improves_yaw"]) for r in rows]) > 0.5
        action = np.stack([np.asarray(r["teacher_residual_action_4d"], dtype=np.float32) for r in rows], axis=0)
        pos_norm = np.linalg.norm(action[:, :3], axis=1)
        invalid = np.asarray([float(r["invalid_action"]) for r in rows]) > 0.5
        workspace = np.asarray([float(r["workspace_violation"]) for r in rows])
        imitation_weight = np.asarray([float(r.get("action_imitation_weight", np.asarray(1.0))) for r in rows])
    else:
        bucket = phase = state = np.asarray([])
        xy = z = yaw = invalid = np.asarray([], dtype=bool)
        pos_norm = workspace = np.asarray([], dtype=np.float32)
        imitation_weight = np.asarray([], dtype=np.float32)
    two = (xy.astype(int) + z.astype(int) + yaw.astype(int)) >= 2
    all_imp = xy & z & yaw
    out = {
        "rows": int(len(rows)),
        "episodes": int(len(episode_summaries)),
        "episode_success_rate": _rate([e.get("success", False) for e in episode_summaries]),
        "episode_basin_success_rate": _rate([e.get("basin_success", False) for e in episode_summaries]),
        "episode_invalid_rate": _rate([e.get("invalid_count", 0) > 0 for e in episode_summaries]),
        "stage_bucket_counts": dict(Counter(bucket.tolist())),
        "alignment_phase_counts": dict(Counter(phase.tolist())),
        "expert_state_counts": dict(Counter(state.tolist())),
        "invalid_action_rate": _rate(invalid),
        "action_imitation_row_rate": _rate(imitation_weight > 0.5),
        "invalid_action_imitation_rate": _rate(np.logical_and(invalid, imitation_weight > 0.5)),
        "workspace_violation_rate": _rate(workspace > 0.0),
        "teacher_improves_xy_rate": _rate(xy),
        "teacher_improves_z_rate": _rate(z),
        "teacher_improves_yaw_rate": _rate(yaw),
        "teacher_two_axis_improve_rate": _rate(two),
        "teacher_all_improves_rate": _rate(all_imp),
        "teacher_pos_action_norm_stats_m": _stats(pos_norm),
        "episodes_detail": episode_summaries,
    }
    for b in ("near_contact_refine", "micro_contact_refine", "broad_near"):
        mask = bucket == b
        if not mask.any():
            continue
        out[f"{b}_rows"] = int(mask.sum())
        out[f"{b}_teacher_xy_improve_rate"] = _rate(xy[mask])
        out[f"{b}_teacher_yaw_improve_rate"] = _rate(yaw[mask])
        out[f"{b}_teacher_two_axis_improve_rate"] = _rate(two[mask])
        out[f"{b}_teacher_action_norm_stats_m"] = _stats(pos_norm[mask])
    for p in ("grasp_commit", "insert_commit", "commit_target", "unknown"):
        mask = phase == p
        if not mask.any():
            continue
        out[f"{p}_rows"] = int(mask.sum())
        out[f"{p}_teacher_xy_improve_rate"] = _rate(xy[mask])
        out[f"{p}_teacher_z_improve_rate"] = _rate(z[mask])
        out[f"{p}_teacher_yaw_improve_rate"] = _rate(yaw[mask])
        out[f"{p}_teacher_two_axis_improve_rate"] = _rate(two[mask])
        out[f"{p}_teacher_all_improves_rate"] = _rate(all_imp[mask])
        out[f"{p}_action_imitation_row_rate"] = _rate(imitation_weight[mask] > 0.5)
        out[f"{p}_teacher_action_norm_stats_m"] = _stats(pos_norm[mask])
    return out


def _demo_target_pose_from_obs(demo) -> np.ndarray:
    # The last gripper pose in a successful demonstration is the most reliable
    # direct-control target for a plug-in low-level controller.  We keep the
    # task success-centre separately as contact geometry supervision.
    return np.asarray(demo[-1].gripper_pose, dtype=np.float32).reshape(7)


def _demo_grasp_commit_index(demo, *, threshold: float = 0.5) -> int | None:
    """Return the first demo index at/after the open-to-closed gripper transition."""
    if len(demo) < 2:
        return None
    gripper = np.asarray([float(getattr(obs, "gripper_open", 1.0)) for obs in demo], dtype=np.float32)
    for i in range(len(gripper) - 1):
        if gripper[i] > float(threshold) and gripper[i + 1] <= float(threshold):
            return int(i + 1)
    closed = np.flatnonzero(gripper <= float(threshold))
    if closed.size:
        return int(closed[0])
    return None


def _demo_target_contexts(demo, args, *, success_centre: np.ndarray | None = None, close_idx: int | None = None) -> list[dict]:
    contexts: list[dict] = []
    stage = str(args.demo_target_stage)
    if stage in ("grasp_commit", "both"):
        grasp_close_idx = _demo_grasp_commit_index(demo, threshold=float(args.gripper_close_threshold))
        if grasp_close_idx is not None:
            start = max(0, int(grasp_close_idx) - int(args.grasp_preclose_window))
            end = min(max(0, len(demo) - 1), int(grasp_close_idx) + int(args.grasp_postclose_window))
            contexts.append(
                {
                    "phase": "grasp_commit",
                    "target_pose": np.asarray(demo[int(grasp_close_idx)].gripper_pose, dtype=np.float32).reshape(7),
                    "candidate_range": range(start, max(start, end)),
                    "target_index": int(grasp_close_idx),
                }
            )
    if stage in ("insert_commit", "both"):
        insert_target = None
        if success_centre is not None:
            candidate = np.asarray(success_centre, dtype=np.float32).reshape(7)
            if np.all(np.isfinite(candidate)) and np.linalg.norm(candidate[3:7]) > 1e-6:
                insert_target = candidate
        if insert_target is None:
            insert_target = _demo_target_pose_from_obs(demo)
        insert_start = int(close_idx) if close_idx is not None else 0
        insert_post_window = int(getattr(args, "grasp_postclose_window", 2))
        insert_end = min(len(demo) - 1, max(insert_start + 1, insert_start + max(insert_post_window, 1)))
        contexts.append(
            {
                "phase": "insert_commit",
                "target_pose": insert_target,
                "candidate_range": range(insert_start, max(insert_start, insert_end)),
                "target_index": int(insert_start),
            }
        )
    return contexts


def _success_centre_from_demo_obs(obs) -> np.ndarray:
    low = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).reshape(-1)
    if low.size >= 7:
        best = None
        for idx in range(low.size // 7):
            chunk = low[idx * 7 : (idx + 1) * 7]
            if np.all(np.isfinite(chunk)) and np.linalg.norm(chunk[3:7]) > 1e-6:
                best = chunk.astype(np.float32)
        if best is not None:
            return best
        if np.all(np.isfinite(low[:7])) and np.linalg.norm(low[3:7]) > 1e-6:
            return low[:7].astype(np.float32)
    return np.full((7,), np.nan, dtype=np.float32)


def collect_from_demos(args) -> dict:
    _lazy_import_tasks()
    if args.task_name not in TASK_MAP:
        raise ValueError(f"Unknown task {args.task_name!r}; available={sorted(TASK_MAP)}")

    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    if hasattr(obs_config, "gripper_touch_forces"):
        obs_config.gripper_touch_forces = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(TASK_MAP[args.task_name])
    rows: list[dict] = []
    episode_summaries: list[dict] = []
    norm_stats = {}
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    try:
        demos = task.get_demos(
            int(args.num_episodes),
            live_demos=True,
            random_selection=False,
            from_episode_number=int(args.demo_from_episode),
            max_attempts=max(10, int(args.num_episodes) * 3),
        )
        for ep, demo in enumerate(demos):
            force_buffer = deque(maxlen=256)
            success_centre = _success_centre_from_demo_obs(demo[0])
            contexts = _demo_target_contexts(demo, args)
            ep_start = len(rows)
            frames = []
            for context in contexts:
                phase = str(context["phase"])
                target_ee = np.asarray(context["target_pose"], dtype=np.float32).reshape(7)
                for i in context["candidate_range"]:
                    if i < 0 or i >= len(demo) - 1:
                        continue
                    obs = demo[i]
                    next_obs = demo[i + 1]
                    if args.record_video:
                        frames.append(obs.front_rgb.copy())
                    _, _, proprio, _, force_hist, depth_96, raw_force = process_obs(
                        obs,
                        norm_stats,
                        force_buffer,
                        use_depth=True,
                        use_force=True,
                        depth_max=float(args.depth_max),
                    )
                    depth_proximity = np.nan
                    if depth_96 is not None:
                        depth_arr = depth_96.detach().cpu().numpy().astype(np.float32)
                        valid = depth_arr[np.isfinite(depth_arr)]
                        if valid.size:
                            depth_proximity = float(np.percentile(valid, 5.0))
                    delta_to_final = apply_yaw_symmetry_to_delta(
                        pose_delta_local_between(obs.gripper_pose, target_ee),
                        np.pi / 2.0,
                    )
                    bucket = _bucket_from_delta(delta_to_final)
                    if bucket == "broad_near" and not bool(args.keep_demo_broad_rows):
                        continue
                    action_local = pose_delta_local_between(obs.gripper_pose, next_obs.gripper_pose)
                    action_local = apply_yaw_symmetry_to_delta(action_local, np.pi / 2.0)
                    post_delta = apply_yaw_symmetry_to_delta(
                        pose_delta_local_between(next_obs.gripper_pose, target_ee),
                        np.pi / 2.0,
                    )
                    state = (
                        f"DEMO_{phase}_MICRO"
                        if bucket == "micro_contact_refine"
                        else f"DEMO_{phase}_NEAR"
                        if bucket == "near_contact_refine"
                        else f"DEMO_{phase}_BROAD"
                    )
                    rows.append(
                        _row_from_obs(
                            obs=obs,
                            force_buffer=force_buffer,
                            raw_force=raw_force,
                            force_hist=force_hist,
                            proprio=proprio,
                            depth_tensor_96=depth_96,
                            depth_proximity=depth_proximity,
                            target_pose=target_ee,
                            object_pose=success_centre,
                            delta=delta_to_final,
                            action_local=action_local,
                            post_delta=post_delta,
                            state=state,
                            bucket=bucket,
                            episode_idx=ep + int(args.demo_from_episode),
                            step_idx=i,
                            init_bucket=f"demo_success_{phase}",
                            phase=phase,
                            target_phase=phase,
                            invalid=False,
                            workspace_violation=0.0,
                            reward=1.0 if i == len(demo) - 2 else 0.0,
                            success=i == len(demo) - 2,
                            args=args,
                        )
                    )
            if args.record_video and len(frames) > 1:
                clip = ImageSequenceClip(frames, fps=int(args.video_fps))
                clip.write_videofile(str(video_dir / f"demo_ep{ep:03d}.mp4"), codec="libx264", audio=False, logger=None)
            episode_summaries.append(
                {
                    "episode_index": int(ep + int(args.demo_from_episode)),
                    "initial_bucket": "demo_success",
                    "rows": int(len(rows) - ep_start),
                    "success": True,
                    "basin_success": True,
                    "invalid_count": 0,
                    "demo_length": int(len(demo)),
                }
            )
            print(f"demo {ep:03d}: rows={len(rows) - ep_start} len={len(demo)}", flush=True)
    finally:
        env.shutdown()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_npz = write_rows_npz(rows, Path(args.output_npz))
    report = _summarize(rows, episode_summaries)
    report["source"] = "rlbench_live_demo_success"
    report["output_npz"] = str(raw_npz) if raw_npz is not None else None
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _run_expert_from_current_obs(
    *,
    task,
    obs,
    target_ee,
    success_centre,
    episode_index: int,
    init_bucket: str,
    phase: str,
    args,
    record_frames: bool,
) -> tuple[list[dict], dict, list[np.ndarray]]:
    norm_stats = {}
    force_buffer = deque(maxlen=256)
    rows: list[dict] = []
    frames: list[np.ndarray] = []
    invalid_count = 0
    success = False
    basin_success = False
    final_delta = np.full((6,), np.nan, dtype=np.float32)
    current_obs = obs
    max_steps = int(
        getattr(args, "insert_teacher_steps", args.perturb_expert_steps)
        if str(phase) == "insert_commit"
        else args.perturb_expert_steps
    )
    keep_gripper_closed = bool(
        getattr(args, "insert_teacher_keep_closed", True)
        if str(phase) == "insert_commit"
        else False
    )
    for step in range(max_steps):
        if record_frames:
            frames.append(current_obs.front_rgb.copy())
        _, _, proprio, _, force_hist, depth_96, raw_force = process_obs(
            current_obs,
            norm_stats,
            force_buffer,
            use_depth=True,
            use_force=True,
            depth_max=float(args.depth_max),
        )
        depth_proximity = np.nan
        if depth_96 is not None:
            depth_arr = depth_96.detach().cpu().numpy().astype(np.float32)
            valid_depth = depth_arr[np.isfinite(depth_arr)]
            if valid_depth.size:
                depth_proximity = float(np.percentile(valid_depth, 5.0))
        delta = apply_yaw_symmetry_to_delta(
            pose_delta_local_between(current_obs.gripper_pose, target_ee),
            np.pi / 2.0,
        )
        bucket = _bucket_from_delta(delta)
        action_local, state = _expert_action(delta, _force_norm(raw_force), step, args)
        world_delta = local_delta_to_world(action_local, np.asarray(current_obs.gripper_pose[3:7], dtype=np.float32))
        delta_action = np.zeros(7, dtype=np.float32)
        delta_action[:6] = world_delta[:6]
        delta_action[6] = 0.0 if keep_gripper_closed else 1.0
        abs_action = delta_to_absolute(delta_action, current_obs.gripper_pose)
        invalid = False
        workspace_violation = 0.0
        reward = 0.0
        try:
            next_obs, reward, terminate = task.step(abs_action)
        except InvalidActionError:
            invalid = True
            invalid_count += 1
            recovery = np.asarray(current_obs.gripper_pose, dtype=np.float32).copy()
            recovery[2] += float(args.unjam_lift_step)
            try:
                next_obs, reward, terminate = task.step(_absolute_action_from_pose(recovery, 0.0 if keep_gripper_closed else 1.0))
            except InvalidActionError:
                next_obs, reward, terminate = current_obs, 0.0, False
        post_delta = apply_yaw_symmetry_to_delta(
            pose_delta_local_between(next_obs.gripper_pose, target_ee),
            np.pi / 2.0,
        )
        final_delta = post_delta
        xy = float(np.linalg.norm(post_delta[:2]))
        z = float(abs(post_delta[2]))
        yaw = float(abs(post_delta[5]))
        basin_success = bool(
            xy <= float(args.success_xy_threshold)
            and z <= float(args.success_z_threshold)
            and yaw <= float(args.success_yaw_threshold)
        )
        success = bool(basin_success or (reward > 0 and not bool(args.ignore_reward_success)))
        rows.append(
            _row_from_obs(
                obs=current_obs,
                force_buffer=force_buffer,
                raw_force=raw_force,
                force_hist=force_hist,
                proprio=proprio,
                depth_tensor_96=depth_96,
                depth_proximity=depth_proximity,
                target_pose=target_ee,
                object_pose=success_centre,
                delta=delta,
                action_local=action_local,
                post_delta=post_delta,
                state=state,
                bucket=bucket,
                episode_idx=episode_index,
                step_idx=step,
                init_bucket=init_bucket,
                phase=phase,
                target_phase=phase,
                invalid=invalid,
                workspace_violation=workspace_violation,
                reward=reward,
                success=success,
                args=args,
            )
        )
        current_obs = next_obs
        if success and not bool(args.run_full_horizon_on_success):
            break
    summary = {
        "episode_index": int(episode_index),
        "initial_bucket": str(init_bucket),
        "rows": int(len(rows)),
        "success": bool(success),
        "basin_success": bool(basin_success),
        "invalid_count": int(invalid_count),
        "final_xy": float(np.linalg.norm(final_delta[:2])) if np.all(np.isfinite(final_delta[:2])) else None,
        "final_z": float(abs(final_delta[2])) if np.isfinite(final_delta[2]) else None,
        "final_yaw": float(abs(final_delta[5])) if np.isfinite(final_delta[5]) else None,
    }
    rollout_imitation_weight = 1.0 if bool(summary["success"]) and int(invalid_count) == 0 else 0.0
    for row in rows:
        current_weight = float(row.get("action_imitation_weight", np.asarray(1.0)))
        row["action_imitation_weight"] = np.asarray(
            float(current_weight * rollout_imitation_weight),
            dtype=np.float32,
        )
    return rows, summary, frames


def _run_demo_grasp_recovery_from_current_obs(
    *,
    task,
    obs,
    target_handle,
    target_ee,
    episode_index: int,
    init_bucket: str,
    phase: str,
    args,
    record_frames: bool,
) -> tuple[list[dict], dict, list[np.ndarray]]:
    """Recover a perturbed successful-demo preclose state back into attach basin.

    This expert is intentionally privileged: the target is the successful demo
    grasp frame, and success is defined by PyRep/RLBench attachment plus lift
    verification.  Rows from failed rollouts are kept only as diagnostic/risk
    negatives; positive imitation weight is assigned after verified attach.
    """

    norm_stats = {}
    force_buffer = deque(maxlen=256)
    rows: list[dict] = []
    frames: list[np.ndarray] = []
    invalid_count = 0
    current_obs = obs
    target_ee = np.asarray(target_ee, dtype=np.float32).reshape(7)
    target_object_pose = _target_pose_from_handle(target_handle)
    close_attempt_step = -1
    verified_step = -1
    attached_after_close = False
    verified_lift = False
    grasped_count_at_close = 0
    close_failure_reason = "not_closed"
    final_delta = np.full((6,), np.nan, dtype=np.float32)

    for step in range(int(args.perturb_expert_steps)):
        if record_frames:
            frames.append(current_obs.front_rgb.copy())
        _, _, proprio, _, force_hist, depth_96, raw_force = process_obs(
            current_obs,
            norm_stats,
            force_buffer,
            use_depth=True,
            use_force=True,
            depth_max=float(args.depth_max),
        )
        depth_proximity = np.nan
        if depth_96 is not None:
            depth_arr = depth_96.detach().cpu().numpy().astype(np.float32)
            valid_depth = depth_arr[np.isfinite(depth_arr)]
            if valid_depth.size:
                depth_proximity = float(np.percentile(valid_depth, 5.0))
        delta = apply_yaw_symmetry_to_delta(
            pose_delta_local_between(current_obs.gripper_pose, target_ee),
            np.pi / 2.0,
        )
        bucket = _bucket_from_delta(delta)
        close_ready = _grasp_recovery_close_ready(delta, args)
        if close_ready:
            close_attempt_step = int(step)
            break
        action_local, state = _expert_action(delta, _force_norm(raw_force), step, args)
        world_delta = local_delta_to_world(action_local, np.asarray(current_obs.gripper_pose[3:7], dtype=np.float32))
        delta_action = np.zeros(7, dtype=np.float32)
        delta_action[:6] = world_delta[:6]
        delta_action[6] = 1.0
        abs_action = delta_to_absolute(delta_action, current_obs.gripper_pose)
        invalid = False
        reward = 0.0
        try:
            next_obs, reward, terminate = task.step(abs_action)
        except InvalidActionError:
            invalid = True
            invalid_count += 1
            recovery = np.asarray(current_obs.gripper_pose, dtype=np.float32).copy()
            recovery[2] += float(args.unjam_lift_step)
            try:
                next_obs, reward, terminate = task.step(_absolute_action_from_pose(recovery, 1.0))
            except InvalidActionError:
                next_obs, reward, terminate = current_obs, 0.0, False
        post_delta = apply_yaw_symmetry_to_delta(
            pose_delta_local_between(next_obs.gripper_pose, target_ee),
            np.pi / 2.0,
        )
        final_delta = post_delta
        row = _row_from_obs(
            obs=current_obs,
            force_buffer=force_buffer,
            raw_force=raw_force,
            force_hist=force_hist,
            proprio=proprio,
            depth_tensor_96=depth_96,
            depth_proximity=depth_proximity,
            target_pose=target_ee,
            object_pose=target_object_pose,
            delta=delta,
            action_local=action_local,
            post_delta=post_delta,
            state=state,
            bucket=bucket,
            episode_idx=episode_index,
            step_idx=step,
            init_bucket=init_bucket,
            phase=phase,
            target_phase=phase,
            invalid=invalid,
            workspace_violation=0.0,
            reward=reward,
            success=False,
            args=args,
        )
        row.update(
            {
                "teacher_close_action": np.asarray("continue_align"),
                "teacher_close_ready": np.asarray(float(close_ready), dtype=np.float32),
                "teacher_close_attempt_step": np.asarray(-1, dtype=np.int64),
                "teacher_grasp_verified_step": np.asarray(-1, dtype=np.int64),
                "teacher_attached_after_close": np.asarray(0.0, dtype=np.float32),
                "teacher_grasped_object_count": np.asarray(0, dtype=np.int64),
                "teacher_grasped_target_handle_match": np.asarray(0.0, dtype=np.float32),
                "teacher_grasp_verified": np.asarray(0.0, dtype=np.float32),
                "expert_sequence_verified": np.asarray(0.0, dtype=np.float32),
                "teacher_close_failure_reason": np.asarray("pending"),
            }
        )
        rows.append(row)
        current_obs = next_obs

    if close_attempt_step >= 0:
        close_pose = np.asarray(current_obs.gripper_pose, dtype=np.float32).copy()
        for close_i in range(int(args.grasp_recovery_close_steps)):
            if record_frames:
                frames.append(current_obs.front_rgb.copy())
            try:
                current_obs, reward, terminate = task.step(_absolute_action_from_pose(close_pose, 0.0))
            except InvalidActionError:
                invalid_count += 1
                break
            attached_after_close, grasped_count_at_close = _grasped_target_status(task, target_handle)
            if attached_after_close and close_i + 1 >= int(args.grasp_recovery_min_close_steps):
                break
        close_failure_reason = "attached" if attached_after_close else "not_attached"

    if attached_after_close:
        lift_streak = 0
        start_object_pose = _target_pose_from_handle(target_handle)
        start_obj_z = float(start_object_pose[2]) if np.isfinite(start_object_pose[2]) else np.nan
        for lift_i in range(int(args.grasp_recovery_lift_steps)):
            if record_frames:
                frames.append(current_obs.front_rgb.copy())
            lift_pose = np.asarray(current_obs.gripper_pose, dtype=np.float32).copy()
            lift_pose[2] += float(args.grasp_recovery_lift_step)
            try:
                current_obs, reward, terminate = task.step(_absolute_action_from_pose(lift_pose, 0.0))
            except InvalidActionError:
                invalid_count += 1
                break
            still_attached, grasped_count_at_close = _grasped_target_status(task, target_handle)
            obj_pose = _target_pose_from_handle(target_handle)
            obj_lift = float(obj_pose[2] - start_obj_z) if np.isfinite(obj_pose[2]) and np.isfinite(start_obj_z) else 0.0
            if still_attached and obj_lift >= float(args.grasp_recovery_verify_lift_threshold):
                lift_streak += 1
            else:
                lift_streak = 0
            if lift_streak >= int(args.grasp_recovery_verify_consecutive_steps):
                verified_lift = True
                verified_step = int(close_attempt_step + lift_i + 1)
                close_failure_reason = "verified"
                break
        if not verified_lift and close_failure_reason == "attached":
            close_failure_reason = "lift_not_verified"

    rollout_verified = bool(attached_after_close and verified_lift)
    for row in rows:
        row["teacher_close_attempt_step"] = np.asarray(int(close_attempt_step), dtype=np.int64)
        row["teacher_grasp_verified_step"] = np.asarray(int(verified_step), dtype=np.int64)
        row["teacher_attached_after_close"] = np.asarray(float(attached_after_close), dtype=np.float32)
        row["teacher_grasped_object_count"] = np.asarray(int(grasped_count_at_close), dtype=np.int64)
        row["teacher_grasped_target_handle_match"] = np.asarray(float(attached_after_close), dtype=np.float32)
        row["teacher_grasp_verified"] = np.asarray(float(rollout_verified), dtype=np.float32)
        row["expert_sequence_verified"] = np.asarray(float(rollout_verified), dtype=np.float32)
        row["teacher_close_failure_reason"] = np.asarray(str(close_failure_reason))
        row["success_label"] = np.asarray(float(rollout_verified), dtype=np.float32)
        row["stop_label"] = np.asarray(float(rollout_verified), dtype=np.float32)
        row["action_imitation_weight"] = np.asarray(
            float(rollout_verified) * float(row.get("action_imitation_weight", np.asarray(1.0))),
            dtype=np.float32,
        )

    summary = {
        "episode_index": int(episode_index),
        "initial_bucket": str(init_bucket),
        "rows": int(len(rows)),
        "success": bool(rollout_verified),
        "basin_success": bool(rollout_verified),
        "attached_after_close": bool(attached_after_close),
        "verified_lift": bool(verified_lift),
        "invalid_count": int(invalid_count),
        "close_attempt_step": int(close_attempt_step),
        "verified_step": int(verified_step),
        "grasped_object_count": int(grasped_count_at_close),
        "close_failure_reason": str(close_failure_reason),
        "final_xy": float(np.linalg.norm(final_delta[:2])) if np.all(np.isfinite(final_delta[:2])) else None,
        "final_z": float(abs(final_delta[2])) if np.isfinite(final_delta[2]) else None,
        "final_yaw": float(abs(final_delta[5])) if np.isfinite(final_delta[5]) else None,
    }
    return rows, summary, frames


def collect_from_demo_perturb(args) -> dict:
    _lazy_import_tasks()
    if args.task_name not in TASK_MAP:
        raise ValueError(f"Unknown task {args.task_name!r}; available={sorted(TASK_MAP)}")

    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    if hasattr(obs_config, "gripper_touch_forces"):
        obs_config.gripper_touch_forces = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(TASK_MAP[args.task_name])
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict] = []
    episode_summaries: list[dict] = []
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    rollout_id = 0
    videos_written = 0
    try:
        for ep in range(int(args.num_episodes)):
            demos = task.get_demos(
                1,
                live_demos=True,
                random_selection=False,
                from_episode_number=int(args.demo_from_episode) + int(ep),
                max_attempts=max(10, int(args.demo_max_attempts)),
            )
            if not demos:
                episode_summaries.append(
                    {
                        "episode_index": int(ep + int(args.demo_from_episode)),
                        "initial_bucket": "demo_perturb_missing_demo",
                        "rows": 0,
                        "success": False,
                        "basin_success": False,
                        "invalid_count": 0,
                        "reason": "missing_demo",
                    }
                )
                continue
            demo = demos[0]
            close_idx_for_insert = _demo_grasp_commit_index(demo, threshold=float(args.gripper_close_threshold))
            success_centre = None
            if close_idx_for_insert is not None and 0 <= int(close_idx_for_insert) < len(demo):
                success_centre = _success_centre_from_demo_obs(demo[int(close_idx_for_insert)])
            if success_centre is None or not np.all(np.isfinite(success_centre)) or np.linalg.norm(success_centre[3:7]) <= 1e-6:
                success_centre = _success_centre_from_demo_obs(demo[min(max(int(close_idx_for_insert or 0), 0), len(demo) - 1)])
            if success_centre is None or not np.all(np.isfinite(success_centre)) or np.linalg.norm(success_centre[3:7]) <= 1e-6:
                success_centre = _success_centre_from_demo_obs(demo[-1])
            contexts = _demo_target_contexts(
                demo,
                args,
                success_centre=success_centre,
                close_idx=close_idx_for_insert,
            )
            demo_rollouts = 0
            context_candidate_counts = {}
            for context in contexts:
                phase = str(context["phase"])
                target_ee = np.asarray(context["target_pose"], dtype=np.float32).reshape(7)
                candidate = []
                for i in context["candidate_range"]:
                    if i < 0 or i >= len(demo) - 1:
                        continue
                    d = apply_yaw_symmetry_to_delta(pose_delta_local_between(demo[i].gripper_pose, target_ee), np.pi / 2.0)
                    b = _bucket_from_delta(d)
                    if b in ("near_contact_refine", "micro_contact_refine"):
                        candidate.append((i, b))
                if int(args.perturb_frames_per_demo) > 0 and len(candidate) > int(args.perturb_frames_per_demo):
                    idx = np.linspace(0, len(candidate) - 1, int(args.perturb_frames_per_demo)).round().astype(int)
                    candidate = [candidate[int(j)] for j in idx]
                context_candidate_counts[phase] = len(candidate)
                for frame_idx, base_bucket in candidate:
                    for copy_idx in range(int(args.perturb_copies_per_frame)):
                        base_obs = demo[frame_idx]
                        _restore_demo_obs(task, base_obs)
                        perturb = _sample_demo_perturb(rng, base_bucket, args)
                        perturbed_pose = apply_executed_local_delta_to_pose(base_obs.gripper_pose, perturb)
                        start_obs, ok = _set_ee_pose_with_ik(task, perturbed_pose, trials=int(args.perturb_ik_trials), max_time_ms=int(args.perturb_ik_max_time_ms))
                        if start_obs is None or not ok:
                            continue
                        start_delta = apply_yaw_symmetry_to_delta(
                            pose_delta_local_between(start_obs.gripper_pose, target_ee),
                            np.pi / 2.0,
                        )
                        start_bucket = _bucket_from_delta(start_delta)
                        if start_bucket == "broad_near" and not bool(args.keep_perturb_broad_rows):
                            continue
                        record_frames = bool(args.record_video and videos_written < int(args.video_episodes))
                        sub_rows, summary, frames = _run_expert_from_current_obs(
                            task=task,
                            obs=start_obs,
                            target_ee=target_ee,
                            success_centre=success_centre,
                            episode_index=rollout_id,
                            init_bucket=f"demo_perturb_{phase}_{start_bucket}",
                            phase=phase,
                            args=args,
                            record_frames=record_frames,
                        )
                        summary.update(
                            {
                                "alignment_phase": phase,
                                "demo_index": int(ep + int(args.demo_from_episode)),
                                "demo_frame_index": int(frame_idx),
                                "copy_index": int(copy_idx),
                                "base_bucket": str(base_bucket),
                                "start_bucket": str(start_bucket),
                                "perturb_ik_ok": bool(ok),
                                "start_xy": float(np.linalg.norm(start_delta[:2])),
                                "start_z": float(abs(start_delta[2])),
                                "start_yaw": float(abs(start_delta[5])),
                            }
                        )
                        rows.extend(sub_rows)
                        episode_summaries.append(summary)
                        if record_frames and len(frames) > 1:
                            clip = ImageSequenceClip(frames, fps=int(args.video_fps))
                            clip.write_videofile(
                                str(video_dir / f"perturb_{phase}_rollout{rollout_id:05d}.mp4"),
                                codec="libx264",
                                audio=False,
                                logger=None,
                            )
                            videos_written += 1
                        rollout_id += 1
                        demo_rollouts += 1
                        if 0 < int(args.max_perturb_rollouts) <= rollout_id:
                            break
                    if 0 < int(args.max_perturb_rollouts) <= rollout_id:
                        break
                if 0 < int(args.max_perturb_rollouts) <= rollout_id:
                    break
            print(
                f"demo {ep:03d}: candidates={context_candidate_counts} perturb_rollouts={demo_rollouts} "
                f"total_rollouts={rollout_id}",
                flush=True,
            )
            if 0 < int(args.max_perturb_rollouts) <= rollout_id:
                break
    finally:
        env.shutdown()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_npz = write_rows_npz(rows, Path(args.output_npz))
    report = _summarize(rows, episode_summaries)
    report["source"] = "rlbench_live_demo_state_perturb_expert"
    report["output_npz"] = str(raw_npz) if raw_npz is not None else None
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def collect_from_demo_grasp_recovery(args) -> dict:
    _lazy_import_tasks()
    if args.task_name not in TASK_MAP:
        raise ValueError(f"Unknown task {args.task_name!r}; available={sorted(TASK_MAP)}")

    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    if hasattr(obs_config, "gripper_touch_forces"):
        obs_config.gripper_touch_forces = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(TASK_MAP[args.task_name])
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict] = []
    episode_summaries: list[dict] = []
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    rollout_id = 0
    videos_written = 0
    try:
        for ep in range(int(args.num_episodes)):
            demos = task.get_demos(
                1,
                live_demos=True,
                random_selection=False,
                from_episode_number=int(args.demo_from_episode) + int(ep),
                max_attempts=max(10, int(args.demo_max_attempts)),
            )
            if not demos:
                episode_summaries.append(
                    {
                        "episode_index": int(ep + int(args.demo_from_episode)),
                        "initial_bucket": "demo_grasp_recovery_missing_demo",
                        "rows": 0,
                        "success": False,
                        "basin_success": False,
                        "invalid_count": 0,
                        "reason": "missing_demo",
                    }
                )
                continue
            demo = demos[0]
            close_idx = _demo_grasp_commit_index(demo, threshold=float(args.gripper_close_threshold))
            if close_idx is None:
                episode_summaries.append(
                    {
                        "episode_index": int(ep + int(args.demo_from_episode)),
                        "initial_bucket": "demo_grasp_recovery_missing_close",
                        "rows": 0,
                        "success": False,
                        "basin_success": False,
                        "invalid_count": 0,
                        "reason": "missing_close_transition",
                    }
                )
                continue
            target_ee = np.asarray(demo[int(close_idx)].gripper_pose, dtype=np.float32).reshape(7)
            start = max(0, int(close_idx) - int(args.grasp_preclose_window))
            end = max(start, int(close_idx) - int(args.grasp_recovery_min_preclose_gap))
            candidates: list[tuple[int, str]] = []
            for i in range(start, end):
                d = apply_yaw_symmetry_to_delta(pose_delta_local_between(demo[i].gripper_pose, target_ee), np.pi / 2.0)
                b = _bucket_from_delta(d)
                if b in ("near_contact_refine", "micro_contact_refine"):
                    candidates.append((i, b))
            if int(args.perturb_frames_per_demo) > 0 and len(candidates) > int(args.perturb_frames_per_demo):
                idx = np.linspace(0, len(candidates) - 1, int(args.perturb_frames_per_demo)).round().astype(int)
                candidates = [candidates[int(j)] for j in idx]

            demo_rollouts = 0
            for frame_idx, base_bucket in candidates:
                for copy_idx in range(int(args.perturb_copies_per_frame)):
                    # Reset the scene to this successful demo's initial state before
                    # restoring the preclose arm state.  The square ring is static
                    # before the first close, so this gives the correct object pose
                    # while avoiding the end-of-demo lifted object state.
                    task.reset_to_demo(demo)
                    target_handle = resolve_live_target_handle(task)
                    base_obs = demo[frame_idx]
                    restored_obs = _restore_demo_obs(task, base_obs)
                    if restored_obs is None:
                        continue
                    perturb = _sample_demo_perturb(rng, base_bucket, args)
                    perturbed_pose = apply_executed_local_delta_to_pose(base_obs.gripper_pose, perturb)
                    start_obs, ok = _set_ee_pose_with_ik(
                        task,
                        perturbed_pose,
                        trials=int(args.perturb_ik_trials),
                        max_time_ms=int(args.perturb_ik_max_time_ms),
                    )
                    if start_obs is None or not ok:
                        continue
                    start_delta = apply_yaw_symmetry_to_delta(
                        pose_delta_local_between(start_obs.gripper_pose, target_ee),
                        np.pi / 2.0,
                    )
                    start_bucket = _bucket_from_delta(start_delta)
                    if start_bucket == "broad_near" and not bool(args.keep_perturb_broad_rows):
                        continue
                    record_frames = bool(args.record_video and videos_written < int(args.video_episodes))
                    sub_rows, summary, frames = _run_demo_grasp_recovery_from_current_obs(
                        task=task,
                        obs=start_obs,
                        target_handle=target_handle,
                        target_ee=target_ee,
                        episode_index=rollout_id,
                        init_bucket=f"demo_grasp_recovery_{start_bucket}",
                        phase="grasp_commit",
                        args=args,
                        record_frames=record_frames,
                    )
                    summary.update(
                        {
                            "alignment_phase": "grasp_commit",
                            "demo_index": int(ep + int(args.demo_from_episode)),
                            "demo_close_index": int(close_idx),
                            "demo_frame_index": int(frame_idx),
                            "copy_index": int(copy_idx),
                            "base_bucket": str(base_bucket),
                            "start_bucket": str(start_bucket),
                            "perturb_ik_ok": bool(ok),
                            "start_xy": float(np.linalg.norm(start_delta[:2])),
                            "start_z": float(abs(start_delta[2])),
                            "start_yaw": float(abs(start_delta[5])),
                        }
                    )
                    rows.extend(sub_rows)
                    episode_summaries.append(summary)
                    if record_frames and len(frames) > 1:
                        clip = ImageSequenceClip(frames, fps=int(args.video_fps))
                        suffix = "verified" if bool(summary.get("success", False)) else "fail"
                        clip.write_videofile(
                            str(video_dir / f"grasp_recovery_rollout{rollout_id:05d}_{suffix}.mp4"),
                            codec="libx264",
                            audio=False,
                            logger=None,
                        )
                        videos_written += 1
                    rollout_id += 1
                    demo_rollouts += 1
                    if 0 < int(args.max_perturb_rollouts) <= rollout_id:
                        break
                if 0 < int(args.max_perturb_rollouts) <= rollout_id:
                    break
            print(
                f"demo {ep:03d}: candidates={len(candidates)} grasp_recovery_rollouts={demo_rollouts} "
                f"total_rollouts={rollout_id}",
                flush=True,
            )
            if 0 < int(args.max_perturb_rollouts) <= rollout_id:
                break
    finally:
        env.shutdown()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_npz = write_rows_npz(rows, Path(args.output_npz))
    report = _summarize(rows, episode_summaries)
    report["source"] = "rlbench_success_demo_preclose_perturb_grasp_recovery"
    report["output_npz"] = str(raw_npz) if raw_npz is not None else None
    report["attached_after_close_rate"] = _rate([e.get("attached_after_close", False) for e in episode_summaries])
    report["verified_lift_rate"] = _rate([e.get("verified_lift", False) for e in episode_summaries])
    report["positive_imitation_row_rate"] = _rate(
        [float(r.get("action_imitation_weight", np.asarray(0.0))) > 0.5 for r in rows]
    )
    report["close_failure_reason_hist"] = dict(
        Counter([str(e.get("close_failure_reason", "unknown")) for e in episode_summaries])
    )
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def collect(args) -> dict:
    _lazy_import_tasks()
    if args.task_name not in TASK_MAP:
        raise ValueError(f"Unknown task {args.task_name!r}; available={sorted(TASK_MAP)}")

    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    if hasattr(obs_config, "gripper_touch_forces"):
        obs_config.gripper_touch_forces = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(TASK_MAP[args.task_name])
    safety = ResidualSafety(
        max_delta_pos=float(args.max_pos_step),
        max_delta_rot=float(args.max_yaw_step),
    )
    rows: list[dict] = []
    episode_summaries: list[dict] = []
    rng = np.random.default_rng(int(args.seed))
    norm_stats = {}
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    try:
        for ep in range(int(args.num_episodes)):
            random.seed(int(args.seed) + ep)
            np.random.seed(int(args.seed) + ep)
            _, obs = task.reset()
            live_target = resolve_live_target_handle(task)
            object_pose = safe_live_target_pose_7d(live_target)
            if object_pose is None:
                episode_summaries.append({"episode_index": ep, "success": False, "reason": "missing_object_pose"})
                continue
            pregrasp, commit = build_phase1_teacher_targets(object_pose, load_phase1_grasp_spec(args.task_name))
            init_bucket = rng.choice(
                ["near_contact_refine", "micro_contact_refine", "contact_stall"],
                p=[float(args.near_reset_prob), float(args.micro_reset_prob), float(args.stall_reset_prob)],
            )
            init_delta = _sample_initial_error(rng, str(init_bucket))
            init_pose = _pose_from_target_delta(commit, init_delta)
            obs, reset_invalid = _move_to_pose(
                task,
                obs,
                init_pose,
                steps=int(args.reset_move_steps),
                safety=safety,
                teleport_ik=bool(args.privileged_teleport_ik_reset),
            )
            reset_delta = apply_yaw_symmetry_to_delta(pose_delta_local_between(obs.gripper_pose, commit), np.pi / 2.0)
            reset_bucket = _bucket_from_delta(reset_delta)
            if reset_bucket == "broad_near" and bool(args.skip_failed_near_reset):
                episode_summaries.append(
                    {
                        "episode_index": ep,
                        "initial_bucket": str(init_bucket),
                        "rows": 0,
                        "success": False,
                        "basin_success": False,
                        "invalid_count": int(reset_invalid),
                        "reason": "failed_near_reset",
                        "reset_xy": float(np.linalg.norm(reset_delta[:2])),
                        "reset_z": float(abs(reset_delta[2])),
                        "reset_yaw": float(abs(reset_delta[5])),
                    }
                )
                print(
                    f"episode {ep:03d}: skipped failed reset "
                    f"xy={np.linalg.norm(reset_delta[:2]):.4f} z={abs(reset_delta[2]):.4f} yaw={abs(reset_delta[5]):.4f}",
                    flush=True,
                )
                continue

            force_buffer = deque(maxlen=256)
            frames = []
            ep_rows_start = len(rows)
            invalid_count = int(reset_invalid)
            success = False
            basin_success = False
            final_delta = np.full((6,), np.nan, dtype=np.float32)

            for step in range(int(args.expert_steps)):
                if args.record_video:
                    frames.append(obs.front_rgb.copy())
                _, _, proprio, _, force_hist, depth_96, raw_force = process_obs(
                    obs,
                    norm_stats,
                    force_buffer,
                    use_depth=True,
                    use_force=True,
                    depth_max=float(args.depth_max),
                )
                depth_proximity = np.nan
                if depth_96 is not None:
                    depth_arr = depth_96.detach().cpu().numpy().astype(np.float32)
                    valid_depth = depth_arr[np.isfinite(depth_arr)]
                    if valid_depth.size:
                        depth_proximity = float(np.percentile(valid_depth, 5.0))
                delta = pose_delta_local_between(obs.gripper_pose, commit)
                delta = apply_yaw_symmetry_to_delta(delta, np.pi / 2.0)
                bucket = _bucket_from_delta(delta)
                action_local, state = _expert_action(delta, _force_norm(raw_force), step, args)
                world_delta = local_delta_to_world(action_local, np.asarray(obs.gripper_pose[3:7], dtype=np.float32))
                delta_action = np.zeros(7, dtype=np.float32)
                delta_action[:6] = world_delta[:6]
                delta_action[6] = 1.0
                abs_action = delta_to_absolute(delta_action, obs.gripper_pose)
                current_workspace = safety.workspace_violation(np.asarray(obs.gripper_pose[:3], dtype=np.float32))
                next_workspace = safety.workspace_violation(abs_action[:3])
                workspace_violation = max(0.0, float(next_workspace - current_workspace))
                invalid = False
                reward = 0.0
                try:
                    next_obs, reward, terminate = task.step(abs_action)
                except InvalidActionError:
                    invalid = True
                    invalid_count += 1
                    # Back off and keep collecting the failure context as risk data.
                    recovery = np.asarray(obs.gripper_pose, dtype=np.float32).copy()
                    recovery[2] += float(args.unjam_lift_step)
                    recovery[:3] = safety.clamp_workspace(recovery[:3])
                    try:
                        next_obs, reward, terminate = task.step(_absolute_action_from_pose(recovery, 1.0))
                    except InvalidActionError:
                        next_obs, reward, terminate = obs, 0.0, False
                post_delta = pose_delta_local_between(next_obs.gripper_pose, commit)
                post_delta = apply_yaw_symmetry_to_delta(post_delta, np.pi / 2.0)
                final_delta = post_delta
                xy = float(np.linalg.norm(post_delta[:2]))
                z = float(abs(post_delta[2]))
                yaw = float(abs(post_delta[5]))
                basin_success = bool(
                    xy <= float(args.success_xy_threshold)
                    and z <= float(args.success_z_threshold)
                    and yaw <= float(args.success_yaw_threshold)
                )
                success = bool(reward > 0 or basin_success)
                rows.append(
                    _row_from_obs(
                        obs=obs,
                        force_buffer=force_buffer,
                        raw_force=raw_force,
                        force_hist=force_hist,
                        proprio=proprio,
                        depth_tensor_96=depth_96,
                        depth_proximity=depth_proximity,
                        target_pose=commit,
                        object_pose=object_pose,
                        delta=delta,
                        action_local=action_local,
                        post_delta=post_delta,
                        state=state,
                        bucket=bucket,
                        episode_idx=ep,
                        step_idx=step,
                        init_bucket=str(init_bucket),
                        phase="commit_target",
                        invalid=invalid,
                        workspace_violation=workspace_violation,
                        reward=reward,
                        success=success,
                        args=args,
                    )
                )
                obs = next_obs
                if success and not bool(args.run_full_horizon_on_success):
                    break

            if args.record_video and len(frames) > 1:
                clip = ImageSequenceClip(frames, fps=int(args.video_fps))
                suffix = "success" if success else "fail"
                clip.write_videofile(str(video_dir / f"ep{ep:03d}_{suffix}.mp4"), codec="libx264", audio=False, logger=None)

            episode_summaries.append(
                {
                    "episode_index": int(ep),
                    "initial_bucket": str(init_bucket),
                    "rows": int(len(rows) - ep_rows_start),
                    "success": bool(success),
                    "basin_success": bool(basin_success),
                    "invalid_count": int(invalid_count),
                    "final_xy": float(np.linalg.norm(final_delta[:2])) if np.all(np.isfinite(final_delta[:2])) else None,
                    "final_z": float(abs(final_delta[2])) if np.isfinite(final_delta[2]) else None,
                    "final_yaw": float(abs(final_delta[5])) if np.isfinite(final_delta[5]) else None,
                }
            )
            print(
                f"episode {ep:03d}: success={success} basin={basin_success} "
                f"rows={len(rows) - ep_rows_start} invalid={invalid_count}",
                flush=True,
            )
    finally:
        env.shutdown()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_npz = write_rows_npz(rows, Path(args.output_npz))
    report = _summarize(rows, episode_summaries)
    report["output_npz"] = str(raw_npz) if raw_npz is not None else None
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="rollout",
        choices=["rollout", "demo", "demo_perturb", "demo_grasp_recovery"],
    )
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--expert_steps", type=int, default=80)
    parser.add_argument("--reset_move_steps", type=int, default=12)
    parser.add_argument("--privileged_teleport_ik_reset", action="store_true", default=True)
    parser.add_argument("--no_privileged_teleport_ik_reset", dest="privileged_teleport_ik_reset", action="store_false")
    parser.add_argument("--skip_failed_near_reset", action="store_true", default=True)
    parser.add_argument("--keep_failed_near_reset", dest="skip_failed_near_reset", action="store_false")
    parser.add_argument("--seed", type=int, default=5110)
    parser.add_argument("--output_dir", type=str, default="runtime_artifacts/alignment_tc_diffusion/privileged_expert_rollout_20260511a")
    parser.add_argument("--output_npz", type=str, default="runtime_artifacts/alignment_tc_diffusion/privileged_expert_rollout_20260511a/alignment_tc_privileged_expert_raw_20260511a.npz")
    parser.add_argument("--report_json", type=str, default="runtime_artifacts/alignment_tc_diffusion/privileged_expert_rollout_20260511a/report_20260511a.json")
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--demo_from_episode", type=int, default=0)
    parser.add_argument("--demo_max_attempts", type=int, default=10)
    parser.add_argument(
        "--demo_target_stage",
        type=str,
        default="insert_commit",
        choices=["grasp_commit", "insert_commit", "both"],
        help="Which successful-demo commit target to perturb around.",
    )
    parser.add_argument("--gripper_close_threshold", type=float, default=0.5)
    parser.add_argument("--grasp_preclose_window", type=int, default=80)
    parser.add_argument("--grasp_postclose_window", type=int, default=2)
    parser.add_argument(
        "--grasp_recovery_min_preclose_gap",
        type=int,
        default=2,
        help="Exclude the final N frames before the demo close transition when sampling perturb starts.",
    )
    parser.add_argument("--keep_demo_broad_rows", action="store_true", default=False)
    parser.add_argument("--perturb_frames_per_demo", type=int, default=12)
    parser.add_argument("--perturb_copies_per_frame", type=int, default=4)
    parser.add_argument("--perturb_expert_steps", type=int, default=40)
    parser.add_argument("--max_perturb_rollouts", type=int, default=-1)
    parser.add_argument("--video_episodes", type=int, default=3)
    parser.add_argument("--keep_perturb_broad_rows", action="store_true", default=False)
    parser.add_argument("--perturb_micro_xy_std", type=float, default=0.0025)
    parser.add_argument("--perturb_micro_z_std", type=float, default=0.0020)
    parser.add_argument("--perturb_micro_yaw_std", type=float, default=0.035)
    parser.add_argument("--perturb_near_xy_std", type=float, default=0.0045)
    parser.add_argument("--perturb_near_z_std", type=float, default=0.0035)
    parser.add_argument("--perturb_near_yaw_std", type=float, default=0.060)
    parser.add_argument("--perturb_max_pos", type=float, default=0.008)
    parser.add_argument("--perturb_max_yaw", type=float, default=0.12)
    parser.add_argument("--perturb_ik_trials", type=int, default=500)
    parser.add_argument("--perturb_ik_max_time_ms", type=int, default=80)
    parser.add_argument("--run_full_horizon_on_success", action="store_true", default=False)
    parser.add_argument("--grasp_recovery_close_xy_threshold", type=float, default=0.0032)
    parser.add_argument("--grasp_recovery_close_z_threshold", type=float, default=0.0035)
    parser.add_argument("--grasp_recovery_close_yaw_threshold", type=float, default=0.025)
    parser.add_argument("--grasp_recovery_close_steps", type=int, default=18)
    parser.add_argument("--grasp_recovery_min_close_steps", type=int, default=4)
    parser.add_argument("--grasp_recovery_lift_steps", type=int, default=14)
    parser.add_argument("--grasp_recovery_lift_step", type=float, default=0.0030)
    parser.add_argument("--grasp_recovery_verify_lift_threshold", type=float, default=0.010)
    parser.add_argument("--grasp_recovery_verify_consecutive_steps", type=int, default=2)
    parser.add_argument(
        "--ignore_reward_success",
        action="store_true",
        default=True,
        help="Treat task reward as an audit field only; perturb rollouts should stop on local basin success, not stale demo success.",
    )
    parser.add_argument("--use_reward_success", dest="ignore_reward_success", action="store_false")
    parser.add_argument("--depth_max", type=float, default=1.0)
    parser.add_argument("--near_reset_prob", type=float, default=0.45)
    parser.add_argument("--micro_reset_prob", type=float, default=0.35)
    parser.add_argument("--stall_reset_prob", type=float, default=0.20)
    parser.add_argument("--max_pos_step", type=float, default=0.0030)
    parser.add_argument("--max_yaw_step", type=float, default=0.014)
    parser.add_argument("--expert_yaw_sign", type=float, default=-1.0)
    parser.add_argument("--k_xy_align", type=float, default=0.45)
    parser.add_argument("--k_z_hold", type=float, default=0.04)
    parser.add_argument("--k_yaw_align", type=float, default=0.34)
    parser.add_argument("--k_xy_descend", type=float, default=0.22)
    parser.add_argument("--k_z_descend", type=float, default=0.28)
    parser.add_argument("--k_yaw_descend", type=float, default=0.18)
    parser.add_argument("--k_xy_contact", type=float, default=0.35)
    parser.add_argument("--k_z_contact", type=float, default=0.05)
    parser.add_argument("--k_yaw_contact", type=float, default=0.30)
    parser.add_argument("--k_xy_commit", type=float, default=0.12)
    parser.add_argument("--k_z_commit", type=float, default=0.22)
    parser.add_argument("--k_yaw_commit", type=float, default=0.10)
    parser.add_argument("--align_xy_threshold", type=float, default=0.008)
    parser.add_argument("--align_yaw_threshold", type=float, default=0.08)
    parser.add_argument("--commit_z_threshold", type=float, default=0.004)
    parser.add_argument("--success_xy_threshold", type=float, default=0.0045)
    parser.add_argument("--success_z_threshold", type=float, default=0.004)
    parser.add_argument("--success_yaw_threshold", type=float, default=0.055)
    parser.add_argument("--light_contact_force", type=float, default=0.45)
    parser.add_argument("--force_spike_threshold", type=float, default=2.5)
    parser.add_argument("--jam_force_threshold", type=float, default=3.5)
    parser.add_argument("--align_z_step", type=float, default=0.0008)
    parser.add_argument("--contact_z_step", type=float, default=0.0006)
    parser.add_argument("--spiral_step", type=float, default=0.0006)
    parser.add_argument("--unjam_lift_step", type=float, default=0.003)
    parser.add_argument("--unjam_lateral_step", type=float, default=0.0012)
    parser.add_argument("--unjam_yaw_step", type=float, default=0.010)
    args = parser.parse_args()
    probs = np.asarray([args.near_reset_prob, args.micro_reset_prob, args.stall_reset_prob], dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-9)
    args.near_reset_prob, args.micro_reset_prob, args.stall_reset_prob = [float(x) for x in probs]
    if args.source == "demo":
        collect_from_demos(args)
    elif args.source == "demo_perturb":
        collect_from_demo_perturb(args)
    elif args.source == "demo_grasp_recovery":
        collect_from_demo_grasp_recovery(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
