#!/usr/bin/env python3
"""Collect planner-state expert recovery data.

Frozen planner runs normally from successful RLBench demo initial states. Once
the planner enters a near/broad-near/micro band, we snapshot the current
observation and let a privileged grasp recovery expert take over to drive the
scene into the verified attach basin. Only verified recovery trajectories are
kept as positive teacher data.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("VLA_PLATFORM", "RLBENCH")

import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from prismatic.vla.constants import FORCE_DIM, NUM_ACTIONS_CHUNK
from prismatic.robot.residual_transforms import world_delta_to_local

from scripts.collect_alignment_tc_privileged_expert_rollout import (
    _absolute_action_from_pose,
    _bucket_from_delta,
    _demo_grasp_commit_index,
    _grasped_target_status,
    _lazy_import_tasks,
    _restore_demo_obs,
    _run_demo_grasp_recovery_from_current_obs,
    _grasp_recovery_close_ready,
    _force_norm,
    _row_from_obs,
    _sample_demo_perturb,
    _summarize,
    _target_pose_from_handle,
    apply_yaw_symmetry_to_delta,
    pose_delta_local_between,
    resolve_live_target_handle,
    write_rows_npz,
    _expert_action,
)
from scripts.collect_residual_data import load_planner
import evaluate_rlbench as ev
from prismatic.robot.stage_target_provider import align_square_edge_pair_target_pose
from prismatic.robot.residual_transforms import local_delta_to_world
from scripts.evaluate_rlbench import (
    build_phase1_teacher_targets,
    delta_to_absolute,
    load_phase1_grasp_spec,
    predict_actions,
    process_obs,
)


def _rate(values) -> float:
    arr = np.asarray(values, dtype=bool).reshape(-1)
    return float(arr.mean()) if arr.size else 0.0


def _normalize_rows_for_npz(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    exemplars: dict[str, np.ndarray] = {}
    for row in rows:
        for key, value in row.items():
            if key not in exemplars:
                exemplars[key] = np.asarray(value)
    normalized: list[dict] = []
    for row in rows:
        fixed = dict(row)
        for key, example in exemplars.items():
            if key in fixed:
                continue
            if example.shape == ():
                if example.dtype.kind in {"U", "S", "O"}:
                    fixed[key] = np.asarray("")
                elif example.dtype.kind == "b":
                    fixed[key] = np.asarray(False, dtype=np.bool_)
                elif example.dtype.kind in {"i", "u"}:
                    fixed[key] = np.asarray(0, dtype=example.dtype)
                else:
                    fixed[key] = np.asarray(0.0, dtype=example.dtype if example.dtype.kind in {"f", "i", "u"} else np.float32)
            else:
                fixed[key] = np.zeros_like(example)
        normalized.append(fixed)
    return normalized


def _make_env(task_name: str, dataset_root: Path):
    ev._lazy_import_tasks()
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
    env = Environment(action_mode, dataset_root=str(dataset_root), obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(ev.TASK_MAP[task_name])
    return env, task


def _planner_takeover_bucket(delta6: np.ndarray) -> str:
    delta = np.asarray(delta6, dtype=np.float32).reshape(6)
    xy = float(np.linalg.norm(delta[:2]))
    z = float(abs(delta[2]))
    yaw = float(abs(delta[5]))
    if xy <= 0.010 and z <= 0.030 and yaw <= 0.18:
        return "micro_contact_refine"
    if xy <= 0.030 and z <= 0.070 and yaw <= 0.35:
        return "near_contact_refine"
    if xy <= 0.060 and z <= 0.120 and yaw <= 0.60:
        return "broad_near"
    return "far"


def _planner_takeover_ready(
    *,
    delta6: np.ndarray,
    step_idx: int,
    min_steps_before_takeover: int,
    allow_broad_near: bool,
    takeover_xy_threshold: float,
    takeover_abs_z_threshold: float,
    takeover_yaw_threshold: float,
    takeover_yaw_guard_threshold: float,
) -> tuple[bool, str]:
    bucket = _planner_takeover_bucket(delta6)
    delta = np.asarray(delta6, dtype=np.float32).reshape(6)
    xy = float(np.linalg.norm(delta[:2]))
    z = float(abs(delta[2]))
    yaw = float(abs(delta[5]))
    if step_idx < int(min_steps_before_takeover):
        return False, "planner_warmup"
    if z <= float(takeover_abs_z_threshold):
        return True, "z_takeover_window"
    if bucket in {"near_contact_refine", "micro_contact_refine"}:
        return True, bucket
    if allow_broad_near and bucket == "broad_near":
        return True, bucket
    return False, bucket


def _motion_phase_from_state(state: str, delta6: np.ndarray) -> str:
    state = str(state)
    delta = np.asarray(delta6, dtype=np.float32).reshape(6)
    xy = float(np.linalg.norm(delta[:2]))
    z = float(abs(delta[2]))
    yaw = float(abs(delta[5]))
    if state == "UNJAM":
        return "unjam"
    if state == "ALIGN_ABOVE":
        if yaw > 0.04:
            return "yaw_correct"
        return "align_xy_yaw"
    if state == "DESCEND_LIGHT":
        return "descend_z"
    if state == "CONTACT_SEARCH":
        return "enter_finger_region"
    if state == "COMMIT":
        if xy <= 0.0045 and z <= 0.0040 and yaw <= 0.055:
            return "close_ready"
        return "grasp_commit"
    return "align_xy_yaw"


def _motion_phase_from_grasp_contract(
    *,
    state: str,
    delta6: np.ndarray,
    grasp_contract: dict | None,
) -> str:
    base_phase = _motion_phase_from_state(state, delta6)
    if not grasp_contract:
        return base_phase
    xy = float(np.linalg.norm(np.asarray(delta6, dtype=np.float32).reshape(6)[:2]))
    z = float(abs(np.asarray(delta6, dtype=np.float32).reshape(6)[2]))
    yaw = float(abs(np.asarray(delta6, dtype=np.float32).reshape(6)[5]))
    contact_ready = bool(grasp_contract.get("contact_ready", False))
    depth_ready = bool(grasp_contract.get("depth_ready", False))
    object_in_finger_region = bool(grasp_contract.get("object_in_finger_region", False))
    grasp_ready = bool(grasp_contract.get("grasp_ready", False))
    readiness_score = float(grasp_contract.get("grasp_readiness_score", 0.0))
    finger_xy_threshold = float(grasp_contract.get("finger_xy_threshold", 0.01))
    finger_abs_z_threshold = float(grasp_contract.get("finger_abs_z_threshold", 0.03))
    finger_yaw_threshold = float(grasp_contract.get("finger_yaw_threshold", 0.18))
    enter_region = bool(
        (contact_ready or depth_ready or readiness_score >= 0.30)
        and not grasp_ready
        and not object_in_finger_region
    )
    enter_region = enter_region or bool(
        not grasp_ready
        and not object_in_finger_region
        and xy <= max(float(finger_xy_threshold) * 1.5, 0.012)
        and z <= max(float(finger_abs_z_threshold) * 1.25, 0.045)
        and yaw <= max(float(finger_yaw_threshold) * 1.5, 0.08)
    )
    if grasp_ready:
        return "close_ready"
    if object_in_finger_region:
        return "descend_z"
    if enter_region:
        return "enter_finger_region"
    if contact_ready or depth_ready:
        return "enter_finger_region"
    return base_phase


def _takeover_delta_score(delta6: np.ndarray) -> float:
    delta = np.asarray(delta6, dtype=np.float32).reshape(6)
    xy = float(np.linalg.norm(delta[:2]))
    z = float(abs(delta[2]))
    yaw = float(abs(delta[5]))
    return float(xy / 0.060 + z / 0.120 + yaw / 0.600)


def _run_takeover_from_obs(
    *,
    task,
    obs,
    target_handle,
    target_ee,
    episode_index: int,
    takeover_step: int,
    takeover_bucket: str,
    takeover_reason: str,
    takeover_origin: str,
    takeover_weight_scale: float,
    takeover_delta6: np.ndarray | None,
    demo_close_index: int,
    args,
    rows: list[dict],
    episode_summaries: list[dict],
    takeover_trace: list[dict],
    video_dir: Path,
) -> tuple[dict, list[np.ndarray]]:
    motion_rows, motion_summary, motion_frames, motion_obs = _run_motion_corridor_from_obs(
        task=task,
        obs=obs,
        target_handle=target_handle,
        target_ee=target_ee,
        episode_index=int(episode_index),
        init_bucket=f"planner_state_{takeover_bucket}",
        phase="planner_state_takeover",
        args=args,
        record_frames=bool(args.record_video and len(takeover_trace) < int(args.video_episodes)),
    )
    takeover_rows, takeover_summary, takeover_frames = _run_demo_grasp_recovery_from_current_obs(
        task=task,
        obs=motion_obs,
        target_handle=target_handle,
        target_ee=target_ee,
        episode_index=int(episode_index),
        init_bucket=f"planner_state_{takeover_bucket}",
        phase="grasp_commit",
        args=args,
        record_frames=bool(args.record_video and len(takeover_trace) < int(args.video_episodes)),
    )
    combined_frames = list(motion_frames)
    combined_frames.extend(takeover_frames)
    rows.extend(motion_rows)
    rows.extend(takeover_rows)
    for row in motion_rows:
        row["takeover_origin"] = np.asarray("motion_corridor")
        row["takeover_weight_scale"] = np.asarray(0.0, dtype=np.float32)
    for row in takeover_rows:
        row["takeover_origin"] = np.asarray(str(takeover_origin))
        row["takeover_weight_scale"] = np.asarray(float(takeover_weight_scale), dtype=np.float32)
    takeover_summary.update(
        {
            "episode_index": int(episode_index),
            "takeover_step": int(takeover_step),
            "takeover_bucket": str(takeover_bucket),
            "takeover_origin": str(takeover_origin),
            "takeover_weight_scale": float(takeover_weight_scale),
            "planner_steps_before_takeover": int(takeover_step),
            "planner_takeover_reason": str(takeover_reason),
            "planner_takeover_mode": str(takeover_bucket),
            "planner_takeover_delta_xy": float(np.linalg.norm(np.asarray(takeover_delta6, dtype=np.float32).reshape(6)[:2]))
            if takeover_delta6 is not None
            else np.nan,
            "planner_takeover_delta_z": float(abs(np.asarray(takeover_delta6, dtype=np.float32).reshape(6)[2]))
            if takeover_delta6 is not None
            else np.nan,
            "planner_takeover_delta_yaw": float(abs(np.asarray(takeover_delta6, dtype=np.float32).reshape(6)[5]))
            if takeover_delta6 is not None
            else np.nan,
            "demo_close_index": int(demo_close_index),
            "motion_corridor_rows": int(len(motion_rows)),
            "motion_corridor_success": bool(motion_summary.get("success", False)),
            "motion_corridor_final_xy": float(motion_summary.get("final_xy", np.nan)),
            "motion_corridor_final_z": float(motion_summary.get("final_z", np.nan)),
            "motion_corridor_final_yaw": float(motion_summary.get("final_yaw", np.nan)),
            "motion_corridor_phase_counts": dict(motion_summary.get("motion_phase_counts", {})),
            "positive_rows": int(
                sum(float(r.get("action_imitation_weight", np.asarray(0.0))) > 0.5 for r in takeover_rows)
            ),
        }
    )
    if float(takeover_weight_scale) != 1.0:
        for row in takeover_rows:
            current_weight = float(row.get("action_imitation_weight", np.asarray(1.0)))
            row["action_imitation_weight"] = np.asarray(float(current_weight * float(takeover_weight_scale)), dtype=np.float32)
    episode_summaries.append(takeover_summary)
    takeover_trace.append(takeover_summary)
    try:
        trace_path = Path(args.takeover_trace_jsonl)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(takeover_summary, sort_keys=True) + "\n")
    except Exception:
        pass
    if args.record_video and combined_frames:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(combined_frames, fps=int(args.video_fps))
        clip.write_videofile(
            str(
                video_dir
                / f"episode_{int(episode_index):03d}_{str(takeover_origin)}_takeover_{int(takeover_step):04d}.mp4"
            ),
            codec="libx264",
            audio=False,
            logger=None,
        )
    return takeover_summary, combined_frames


def _run_motion_corridor_from_obs(
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
) -> tuple[list[dict], dict, list[np.ndarray], object]:
    """Run the 0419a-style motion corridor before verified close/verify.

    This phase is intentionally permissive: it can enter from near/micro and
    uses a stronger step scale / candidate bank via the same expert action
    heuristic as the privileged teacher collector. Rows from this phase are
    diagnostic only and never get positive imitation weight.
    """

    norm_stats = {}
    force_buffer = deque(maxlen=256)
    rows: list[dict] = []
    frames: list[np.ndarray] = []
    current_obs = obs
    motion_phase_counts: Counter[str] = Counter()
    final_delta = np.full((6,), np.nan, dtype=np.float32)

    for step in range(int(getattr(args, "planner_takeover_motion_steps", 24))):
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
        live_object_pose = _target_pose_from_handle(target_handle) if target_handle is not None else None
        delta = apply_yaw_symmetry_to_delta(
            pose_delta_local_between(current_obs.gripper_pose, target_ee),
            np.pi / 2.0,
        )
        bucket = _bucket_from_delta(delta)
        close_ready = _grasp_recovery_close_ready(delta, args)
        if close_ready:
            break

        force_norm = _force_norm(raw_force)
        object_delta = (
            apply_yaw_symmetry_to_delta(
                pose_delta_local_between(current_obs.gripper_pose, live_object_pose),
                np.pi / 2.0,
            )
            if live_object_pose is not None
            else delta
        )
        low_visibility = False
        is_occluded = False
        if depth_96 is not None:
            _, _, is_occluded, low_visibility = ev.compute_wrist_visibility_stats(
                depth_96.detach().cpu().numpy().astype(np.float32)
            )
        grasp_contract = ev.compute_tc_grasp_expert_contract(
            current_pose_7d=np.asarray(current_obs.gripper_pose, dtype=np.float32),
            object_pose_7d=np.asarray(live_object_pose, dtype=np.float32) if live_object_pose is not None else None,
            gripper_open=float(current_obs.gripper_open),
            depth_proximity=depth_proximity,
            stage_contact_state=None,
            refiner_contact_state=None,
            close_xy_threshold=float(args.grasp_recovery_close_xy_threshold),
            close_abs_z_threshold=float(args.grasp_recovery_close_z_threshold),
            close_yaw_threshold=float(args.grasp_recovery_close_yaw_threshold),
            close_contact_depth_threshold=float(getattr(args, "teacher_close_contact_depth_threshold", 0.020)),
            low_visibility=bool(low_visibility),
            occluded=bool(is_occluded),
            grasp_ready_threshold=float(getattr(args, "teacher_grasp_ready_threshold", 0.55)),
            desired_object_delta_local_6d=None,
        )
        motion_phase = _motion_phase_from_grasp_contract(state="UNKNOWN", delta6=delta, grasp_contract=grasp_contract)
        xy_to_close = float(np.linalg.norm(delta[:2]))
        z_to_close = float(abs(delta[2]))
        yaw_to_close = float(abs(delta[5]))
        force_descend_ready = bool(
            motion_phase == "enter_finger_region"
            and z_to_close > float(args.commit_z_threshold)
            and (
                step >= int(args.motion_corridor_force_descend_after_steps)
                or (
                    xy_to_close <= float(args.motion_corridor_descend_xy_threshold)
                    and yaw_to_close <= float(args.motion_corridor_descend_yaw_threshold)
                )
            )
        )
        if force_descend_ready:
            motion_phase = "descend_z"
        control_delta = object_delta if motion_phase in {"enter_finger_region", "descend_z"} else delta
        if motion_phase == "descend_z":
            control_delta = delta
        action_local, state = _expert_action(control_delta, force_norm, step, args)
        if motion_phase == "enter_finger_region":
            state = "CONTACT_SEARCH"
        elif motion_phase == "descend_z":
            state = "DESCEND_LIGHT"
        elif motion_phase == "close_ready":
            state = "COMMIT"
        motion_phase_counts[motion_phase] += 1

        world_delta = local_delta_to_world(
            action_local,
            np.asarray(current_obs.gripper_pose[3:7], dtype=np.float32),
        )
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
            object_pose=live_object_pose if live_object_pose is not None else _target_pose_from_handle(resolve_live_target_handle(task)),
            delta=delta,
            action_local=action_local,
            post_delta=post_delta,
            state=state,
            bucket=bucket,
            episode_idx=episode_index,
            step_idx=step,
            init_bucket=init_bucket,
            phase=phase,
            invalid=invalid,
            workspace_violation=0.0,
            reward=reward,
            success=False,
            args=args,
        )
        row.update(
            {
                "takeover_active": np.asarray(1.0, dtype=np.float32),
                "takeover_ready": np.asarray(1.0, dtype=np.float32),
                "takeover_reason": np.asarray("motion_corridor"),
                "teacher_motion_phase": np.asarray(motion_phase),
                "motion_corridor_force_descend_ready": np.asarray(float(force_descend_ready), dtype=np.float32),
                "teacher_close_action": np.asarray("continue_align"),
                "teacher_close_ready": np.asarray(0.0, dtype=np.float32),
                "teacher_close_ready_all": np.asarray(0.0, dtype=np.float32),
                "teacher_close_gate_reason": np.asarray("motion_corridor"),
                "teacher_attached_after_close": np.asarray(0.0, dtype=np.float32),
                "teacher_grasped_object_count": np.asarray(0, dtype=np.int64),
                "teacher_grasp_verified": np.asarray(0.0, dtype=np.float32),
                "verified_lift": np.asarray(0.0, dtype=np.float32),
                "teacher_gripper_finger_pose_7d": np.asarray(current_obs.gripper_pose, dtype=np.float32),
                "teacher_object_to_gripper_delta_local_6d": np.asarray(object_delta, dtype=np.float32),
                "teacher_object_in_finger_region": np.asarray(float(grasp_contract["object_in_finger_region"]), dtype=np.float32),
                "teacher_grasp_contact_ready": np.asarray(float(grasp_contract["contact_ready"]), dtype=np.float32),
                "teacher_grasp_ready": np.asarray(float(grasp_contract["grasp_ready"]), dtype=np.float32),
                "teacher_grasp_readiness_score": np.asarray(float(grasp_contract["grasp_readiness_score"]), dtype=np.float32),
                "teacher_finger_object_lateral_error": np.asarray(float(grasp_contract["finger_object_lateral_error"]), dtype=np.float32),
                "teacher_finger_object_height_overlap": np.asarray(float(grasp_contract["finger_object_height_overlap"]), dtype=np.float32),
                "teacher_finger_object_yaw_error": np.asarray(float(grasp_contract["finger_object_yaw_error"]), dtype=np.float32),
                "action_imitation_weight": np.asarray(0.0, dtype=np.float32),
            }
        )
        rows.append(row)
        current_obs = next_obs
        if _grasp_recovery_close_ready(post_delta, args):
            break

    summary = {
        "episode_index": int(episode_index),
        "initial_bucket": str(init_bucket),
        "rows": int(len(rows)),
        "success": False,
        "basin_success": False,
        "invalid_count": 0,
        "final_xy": float(np.linalg.norm(final_delta[:2])) if np.all(np.isfinite(final_delta[:2])) else None,
        "final_z": float(abs(final_delta[2])) if np.isfinite(final_delta[2]) else None,
        "final_yaw": float(abs(final_delta[5])) if np.isfinite(final_delta[5]) else None,
        "motion_phase_counts": dict(motion_phase_counts),
    }
    return rows, summary, frames, current_obs


def collect(args) -> dict:
    ev._lazy_import_tasks()
    if args.task_name not in ev.TASK_MAP:
        raise ValueError(f"Unknown task {args.task_name!r}; available={sorted(ev.TASK_MAP)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    print("[planner_state_expert_recovery] Loading frozen planner...")
    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        args.checkpoint_dir,
        args.vlm_path,
        args.config_path,
        args.planner_use_depth,
        args.planner_use_force,
    )

    env, task = _make_env(args.task_name, Path(args.data_root))
    rows: list[dict] = []
    episode_summaries: list[dict] = []
    takeover_trace = []
    rng = np.random.default_rng(int(args.seed))
    shard_frames = []
    total_takeovers = 0

    selected_episode_numbers = None
    if getattr(args, "episode_indices", None):
        selected_episode_numbers = [
            int(part.strip())
            for part in str(args.episode_indices).split(",")
            if part.strip() != ""
        ]
    try:
        loop_episode_numbers = (
            selected_episode_numbers
            if selected_episode_numbers is not None and len(selected_episode_numbers) > 0
            else [int(args.demo_from_episode) + int(ep) for ep in range(int(args.num_episodes))]
        )
        for ep, episode_number in enumerate(loop_episode_numbers):
            demos = task.get_demos(
                1,
                live_demos=True,
                random_selection=False,
                from_episode_number=int(episode_number),
                max_attempts=max(10, int(args.demo_max_attempts)),
            )
            if not demos:
                episode_summaries.append(
                    {
                        "episode_index": int(episode_number),
                        "rows": 0,
                        "success": False,
                        "reason": "missing_demo",
                        "takeover_step": -1,
                    }
                )
                continue

            demo = demos[0]
            close_idx = _demo_grasp_commit_index(demo, threshold=float(args.gripper_close_threshold))
            if close_idx is None:
                episode_summaries.append(
                    {
                        "episode_index": int(episode_number),
                        "rows": 0,
                        "success": False,
                        "reason": "missing_close_transition",
                        "takeover_step": -1,
                    }
                )
                continue
            target_handle = resolve_live_target_handle(task)
            target_ee = np.asarray(demo[int(close_idx)].gripper_pose, dtype=np.float32).reshape(7)
            edgepair_grasp_spec = load_phase1_grasp_spec(args.task_name) if bool(args.record_edgepair_labels) else None
            edgepair_target_mode = str(getattr(args, "edgepair_label_target_mode", "commit"))

            descs, obs = task.reset_to_demo(demo)
            instruction = descs[0] if isinstance(descs, list) else str(descs)
            force_buffer = deque(maxlen=256)
            action_queue: list[np.ndarray] = []
            frames: list[np.ndarray] = []
            planner_rows_start = len(rows)
            takeover_started = False
            takeover_summary = None
            takeover_bucket = "none"
            takeover_step = -1
            takeover_reason = "not_triggered"
            local_step = 0
            max_steps = len(demo) if args.max_rollout_steps <= 0 else min(len(demo), int(args.max_rollout_steps))
            reset_taken_over = False
            episode_summary_written = False
            last_planner_bucket = "far"
            best_fallback_obs = None
            best_fallback_step = -1
            best_fallback_bucket = "none"
            best_fallback_delta = None
            best_fallback_score = np.inf
            last_planner_delta = np.full((6,), np.nan, dtype=np.float32)

            while local_step < max_steps:
                front_pil, wrist_pil, proprio, depth_tensor_224, force_hist, depth_tensor_96, raw_force = process_obs(
                    obs,
                    norm_stats,
                    force_buffer,
                    use_depth=True,
                    use_force=True,
                    depth_max=1.0,
                )
                if args.record_video:
                    frames.append(obs.front_rgb.copy())

                if len(action_queue) == 0:
                    actions = predict_actions(
                        vla,
                        processor,
                        action_head,
                        proprio_projector,
                        front_pil,
                        wrist_pil,
                        proprio,
                        depth_tensor_224 if args.planner_use_depth else None,
                        force_hist if args.planner_use_force else None,
                        instruction,
                        unnorm_key="rlbench",
                    )
                    action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), NUM_ACTIONS_CHUNK))]
                if not action_queue:
                    break

                base_action = action_queue.pop(0)
                current_quat = np.asarray(obs.gripper_pose[3:7], dtype=np.float32)
                base_action_local = world_delta_to_local(base_action[:6], current_quat)
                planner_action_local = np.concatenate(
                    [np.asarray(base_action_local[:6], dtype=np.float32), np.asarray([base_action[6]], dtype=np.float32)],
                    axis=0,
                )
                planner_delta_to_demo_close = apply_yaw_symmetry_to_delta(
                    pose_delta_local_between(obs.gripper_pose, target_ee),
                    np.pi / 2.0,
                )
                last_planner_delta = np.asarray(planner_delta_to_demo_close, dtype=np.float32).copy()
                planner_bucket = _planner_takeover_bucket(planner_delta_to_demo_close)
                last_planner_bucket = str(planner_bucket)
                ready, takeover_reason = _planner_takeover_ready(
                    delta6=planner_delta_to_demo_close,
                    step_idx=local_step,
                    min_steps_before_takeover=int(args.planner_min_steps_before_takeover),
                    allow_broad_near=bool(args.allow_broad_near_takeover),
                    takeover_xy_threshold=float(args.planner_takeover_xy_threshold),
                    takeover_abs_z_threshold=float(args.planner_takeover_abs_z_threshold),
                    takeover_yaw_threshold=float(args.planner_takeover_yaw_threshold),
                    takeover_yaw_guard_threshold=float(args.planner_takeover_yaw_guard_threshold),
                )
                if (
                    bool(args.fallback_to_best_broad_near)
                    and local_step >= int(args.best_broad_near_min_step)
                    and planner_bucket in {"broad_near", "near_contact_refine", "micro_contact_refine"}
                ):
                    score = _takeover_delta_score(planner_delta_to_demo_close)
                    if score < float(best_fallback_score):
                        best_fallback_obs = obs
                        best_fallback_step = int(local_step)
                        best_fallback_bucket = str(planner_bucket)
                        best_fallback_delta = np.asarray(planner_delta_to_demo_close, dtype=np.float32).copy()
                        best_fallback_score = float(score)

                row = {
                    "episode_index": np.asarray(int(episode_number), dtype=np.int64),
                    "step_index": np.asarray(int(local_step), dtype=np.int64),
                    "task_name": np.asarray(str(args.task_name)),
                    "stage_bucket": np.asarray(str(planner_bucket)),
                    "planner_phase": np.asarray("planner_state"),
                    "alignment_phase": np.asarray("planner_state"),
                    "expert_state": np.asarray("PLANNER_ONLY"),
                    "takeover_active": np.asarray(float(False), dtype=np.float32),
                    "takeover_ready": np.asarray(float(ready), dtype=np.float32),
                    "takeover_reason": np.asarray(str(takeover_reason)),
                    "takeover_origin": np.asarray("planner_only"),
                    "takeover_weight_scale": np.asarray(0.0, dtype=np.float32),
                    "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
                    "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
                    "wrist_depth": depth_tensor_96.detach().cpu().numpy().astype(np.float32)
                    if depth_tensor_96 is not None
                    else np.zeros((1, 96, 96), dtype=np.float32),
                    "force_history": np.asarray(
                        force_hist.detach().cpu().numpy().astype(np.float32)
                        if force_hist is not None
                        else np.zeros((32, FORCE_DIM), dtype=np.float32),
                        dtype=np.float32,
                    ),
                    "proprio": np.asarray(proprio, dtype=np.float32),
                    "planner_action_local": np.asarray(planner_action_local, dtype=np.float32),
                    "gripper_context": np.asarray([float(obs.gripper_open), 1.0, 0.0, 0.0], dtype=np.float32),
                    "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                    "privileged_motion_target_pose_7d": np.asarray(target_ee, dtype=np.float32),
                    "privileged_object_anchor_pose_7d": np.asarray(target_handle.get_pose(), dtype=np.float32)
                    if target_handle is not None
                    else np.full((7,), np.nan, dtype=np.float32),
                    "privileged_current_to_target_delta_local": np.asarray(
                        planner_delta_to_demo_close,
                        dtype=np.float32,
                    ),
                    "teacher_target_delta_local_6d": np.asarray(planner_delta_to_demo_close, dtype=np.float32),
                    "teacher_residual_action_6d": np.zeros((6,), dtype=np.float32),
                    "teacher_residual_action_4d": np.zeros((4,), dtype=np.float32),
                    "teacher_post_delta_local_6d": np.asarray(planner_delta_to_demo_close, dtype=np.float32),
                    "teacher_improves_xy": np.asarray(0.0, dtype=np.float32),
                    "teacher_improves_z": np.asarray(0.0, dtype=np.float32),
                    "teacher_improves_yaw": np.asarray(0.0, dtype=np.float32),
                    "teacher_close_ready": np.asarray(float(ready), dtype=np.float32),
                    "teacher_attached_after_close": np.asarray(0.0, dtype=np.float32),
                    "teacher_grasped_object_count": np.asarray(0, dtype=np.int64),
                    "teacher_grasp_verified": np.asarray(0.0, dtype=np.float32),
                    "verified_lift": np.asarray(0.0, dtype=np.float32),
                    "close_failure_reason": np.asarray("planner_only"),
                    "action_imitation_weight": np.asarray(0.0, dtype=np.float32),
                    "reward": np.asarray(0.0, dtype=np.float32),
                    "success_label": np.asarray(0.0, dtype=np.float32),
                    "stop_label": np.asarray(0.0, dtype=np.float32),
                    "invalid_action": np.asarray(0.0, dtype=np.float32),
                    "workspace_violation": np.asarray(0.0, dtype=np.float32),
                }
                live_object_pose = _target_pose_from_handle(target_handle)
                planner_object_delta_local_6d = apply_yaw_symmetry_to_delta(
                    pose_delta_local_between(obs.gripper_pose, live_object_pose),
                    np.pi / 2.0,
                )
                low_visibility = False
                is_occluded = False
                depth_arr_96 = None
                if depth_tensor_96 is not None:
                    depth_arr_96 = depth_tensor_96.detach().cpu().numpy().astype(np.float32)
                    _, _, is_occluded, low_visibility = ev.compute_wrist_visibility_stats(depth_arr_96)
                planner_grasp_contract = ev.compute_tc_grasp_expert_contract(
                    current_pose_7d=np.asarray(obs.gripper_pose, dtype=np.float32),
                    object_pose_7d=np.asarray(live_object_pose, dtype=np.float32),
                    gripper_open=float(obs.gripper_open),
                    depth_proximity=float(np.percentile(depth_arr_96[np.isfinite(depth_arr_96)], 5.0))
                    if depth_arr_96 is not None and np.isfinite(depth_arr_96).any()
                    else None,
                    stage_contact_state=None,
                    refiner_contact_state=None,
                    close_xy_threshold=float(args.grasp_recovery_close_xy_threshold),
                    close_abs_z_threshold=float(args.grasp_recovery_close_z_threshold),
                    close_yaw_threshold=float(args.grasp_recovery_close_yaw_threshold),
                    close_contact_depth_threshold=float(getattr(args, "teacher_close_contact_depth_threshold", 0.020)),
                    low_visibility=bool(low_visibility),
                    occluded=bool(is_occluded),
                    grasp_ready_threshold=float(getattr(args, "teacher_grasp_ready_threshold", 0.55)),
                    desired_object_delta_local_6d=None,
                )
                row.update(
                    {
                        "teacher_gripper_finger_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                        "teacher_object_to_gripper_delta_local_6d": np.asarray(planner_object_delta_local_6d, dtype=np.float32),
                        "teacher_object_in_finger_region": np.asarray(
                            float(planner_grasp_contract["object_in_finger_region"]),
                            dtype=np.float32,
                        ),
                        "teacher_grasp_contact_ready": np.asarray(float(planner_grasp_contract["contact_ready"]), dtype=np.float32),
                        "teacher_grasp_ready": np.asarray(float(planner_grasp_contract["grasp_ready"]), dtype=np.float32),
                        "teacher_grasp_readiness_score": np.asarray(
                            float(planner_grasp_contract["grasp_readiness_score"]),
                            dtype=np.float32,
                        ),
                        "teacher_finger_object_lateral_error": np.asarray(
                            float(planner_grasp_contract["finger_object_lateral_error"]),
                            dtype=np.float32,
                        ),
                        "teacher_finger_object_height_overlap": np.asarray(
                            float(planner_grasp_contract["finger_object_height_overlap"]),
                            dtype=np.float32,
                        ),
                        "teacher_finger_object_yaw_error": np.asarray(
                            float(planner_grasp_contract["finger_object_yaw_error"]),
                            dtype=np.float32,
                        ),
                        "teacher_close_contact_ready": np.asarray(
                            float(planner_grasp_contract["contact_ready"]),
                            dtype=np.float32,
                        ),
                        "teacher_close_contact_ready_by_depth": np.asarray(
                            float(planner_grasp_contract["depth_ready"]),
                            dtype=np.float32,
                        ),
                        "teacher_close_contact_ready_by_stage": np.asarray(
                            float(planner_grasp_contract["stage_ready"]),
                            dtype=np.float32,
                        ),
                        "teacher_close_contact_ready_by_geometry": np.asarray(
                            float(planner_grasp_contract["object_in_finger_region"]),
                            dtype=np.float32,
                        ),
                    }
                )
                if edgepair_grasp_spec is not None:
                    try:
                        object_pose = _target_pose_from_handle(target_handle)
                        _, grasp_commit_target = build_phase1_teacher_targets(object_pose, edgepair_grasp_spec)
                        edgepair_target, edge_idx, edge_family, edge_yaw_err = align_square_edge_pair_target_pose(
                            np.asarray(obs.gripper_pose, dtype=np.float32),
                            np.asarray(grasp_commit_target, dtype=np.float32),
                            float(edgepair_grasp_spec.yaw_symmetry_period),
                        )
                        label_target = np.asarray(
                            apply_yaw_symmetry_to_delta(
                                pose_delta_local_between(obs.gripper_pose, edgepair_target),
                                float(edgepair_grasp_spec.yaw_symmetry_period),
                            ),
                            dtype=np.float32,
                        )
                        target_source = (
                            f"label_only_phase1_grasp_commit__edge_pair_q{int(edge_idx)}_f{int(edge_family)}"
                            if int(edge_idx) >= 0 and int(edge_family) >= 0
                            else "label_only_phase1_grasp_commit__none"
                        )
                        row.update(
                            {
                                "teacher_grasp_commit_target_pose_7d": np.asarray(grasp_commit_target, dtype=np.float32),
                                "teacher_edgepair_label_source": np.asarray(str(target_source)),
                                "teacher_grasp_commit_edge_pair_index": np.asarray(int(edge_idx), dtype=np.int64),
                                "teacher_grasp_commit_edge_pair_family": np.asarray(int(edge_family), dtype=np.int64),
                                "teacher_grasp_commit_edge_pair_yaw_error": np.asarray(float(edge_yaw_err), dtype=np.float32),
                                "teacher_target_delta_local_6d": np.asarray(label_target, dtype=np.float32),
                            }
                        )
                    except Exception:
                        row.update(
                            {
                                "teacher_grasp_commit_target_pose_7d": np.asarray(np.full((7,), np.nan, dtype=np.float32)),
                                "teacher_edgepair_label_source": np.asarray("label_exception"),
                            }
                        )
                        row["teacher_grasp_commit_edge_pair_index"] = np.asarray(-1, dtype=np.int64)
                        row["teacher_grasp_commit_edge_pair_family"] = np.asarray(-1, dtype=np.int64)
                        row["teacher_grasp_commit_edge_pair_yaw_error"] = np.asarray(np.nan, dtype=np.float32)
                rows.append(row)

                if ready:
                    takeover_started = True
                    takeover_step = int(local_step)
                    takeover_bucket = str(planner_bucket)
                    takeover_summary, takeover_frames = _run_takeover_from_obs(
                        task=task,
                        obs=obs,
                        target_handle=target_handle,
                        target_ee=target_ee,
                        episode_index=int(episode_number),
                        takeover_step=int(takeover_step),
                        takeover_bucket=str(takeover_bucket),
                        takeover_reason=str(takeover_reason),
                        takeover_origin="natural_takeover",
                        takeover_weight_scale=1.0,
                        takeover_delta6=planner_delta_to_demo_close,
                        demo_close_index=int(close_idx),
                        args=args,
                        rows=rows,
                        episode_summaries=episode_summaries,
                        takeover_trace=takeover_trace,
                        video_dir=video_dir,
                    )
                    total_takeovers += 1
                    reset_taken_over = True
                    episode_summary_written = True
                    break

                abs_action = delta_to_absolute(np.asarray(base_action, dtype=np.float32), obs.gripper_pose)
                try:
                    obs, reward, terminate = task.step(abs_action)
                except Exception as e:
                    if (
                        bool(args.fallback_to_best_broad_near)
                        and best_fallback_obs is not None
                        and best_fallback_bucket == "broad_near"
                    ):
                        restored = _restore_demo_obs(task, best_fallback_obs)
                        fallback_obs = restored if restored is not None else best_fallback_obs
                        takeover_started = True
                        takeover_step = int(best_fallback_step)
                        takeover_bucket = str(best_fallback_bucket)
                        takeover_reason = f"best_{best_fallback_bucket}_fallback_after_{type(e).__name__}"
                        takeover_summary, takeover_frames = _run_takeover_from_obs(
                            task=task,
                            obs=fallback_obs,
                            target_handle=target_handle,
                            target_ee=target_ee,
                            episode_index=int(episode_number),
                            takeover_step=int(takeover_step),
                            takeover_bucket=str(takeover_bucket),
                            takeover_reason=str(takeover_reason),
                            takeover_origin="forced_probe",
                            takeover_weight_scale=0.35,
                            takeover_delta6=best_fallback_delta,
                            demo_close_index=int(close_idx),
                            args=args,
                            rows=rows,
                            episode_summaries=episode_summaries,
                            takeover_trace=takeover_trace,
                            video_dir=video_dir,
                        )
                        total_takeovers += 1
                        reset_taken_over = True
                        episode_summary_written = True
                        break
                    episode_summaries.append(
                        {
                            "episode_index": int(episode_number),
                            "rows": int(len(rows) - planner_rows_start),
                            "success": False,
                            "reason": type(e).__name__,
                            "takeover_step": -1,
                            "planner_steps_before_takeover": int(local_step),
                        }
                    )
                    episode_summary_written = True
                    break

                local_step += 1
                if reward > 0 or terminate:
                    break

            if (
                (not takeover_started)
                and bool(args.force_alignment_probe_if_no_takeover)
            ):
                fallback_obs = obs
                fallback_bucket = str(last_planner_bucket)
                fallback_step = int(local_step)
                fallback_reason = "no_takeover_end_fallback"
                if bool(args.fallback_to_best_broad_near) and best_fallback_obs is not None:
                    restored = _restore_demo_obs(task, best_fallback_obs)
                    fallback_obs = restored if restored is not None else best_fallback_obs
                    fallback_bucket = str(best_fallback_bucket)
                    fallback_step = int(best_fallback_step)
                    fallback_reason = f"best_{best_fallback_bucket}_fallback"
                takeover_started = True
                takeover_step = int(fallback_step)
                takeover_bucket = str(fallback_bucket)
                takeover_reason = str(fallback_reason)
                takeover_summary, takeover_frames = _run_takeover_from_obs(
                    task=task,
                    obs=fallback_obs,
                    target_handle=target_handle,
                    target_ee=target_ee,
                    episode_index=int(episode_number),
                    takeover_step=int(takeover_step),
                    takeover_bucket=str(takeover_bucket),
                    takeover_reason=str(takeover_reason),
                    takeover_origin="forced_probe",
                    takeover_weight_scale=0.35,
                    takeover_delta6=best_fallback_delta if best_fallback_delta is not None else last_planner_delta,
                    demo_close_index=int(close_idx),
                    args=args,
                    rows=rows,
                    episode_summaries=episode_summaries,
                    takeover_trace=takeover_trace,
                    video_dir=video_dir,
                )
                total_takeovers += 1
                reset_taken_over = True
                episode_summary_written = True
                takeover_summary["forced_alignment_probe"] = True

            if takeover_started and takeover_summary is not None:
                print(
                    f"episode {episode_number:03d}: takeover_step={takeover_step} bucket={takeover_bucket} "
                    f"origin={takeover_summary.get('takeover_origin', 'unknown')} "
                    f"reason={takeover_summary.get('planner_takeover_reason', 'unknown')} "
                    f"delta_xy={float(takeover_summary.get('planner_takeover_delta_xy', np.nan)):.4f} "
                    f"delta_z={float(takeover_summary.get('planner_takeover_delta_z', np.nan)):.4f} "
                    f"delta_yaw={float(takeover_summary.get('planner_takeover_delta_yaw', np.nan)):.4f} "
                    f"verified={takeover_summary.get('success', False)} rows={len(rows) - planner_rows_start}",
                    flush=True,
                )
            else:
                if not episode_summary_written:
                    episode_summaries.append(
                        {
                            "episode_index": int(episode_number),
                            "rows": int(len(rows) - planner_rows_start),
                            "success": False,
                            "reason": "no_takeover",
                            "takeover_step": -1,
                        }
                    )
                print(f"episode {episode_number:03d}: no takeover rows={len(rows) - planner_rows_start}", flush=True)
            if args.record_video and frames and not reset_taken_over:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

                clip = ImageSequenceClip(frames, fps=int(args.video_fps))
                clip.write_videofile(
                    str(video_dir / f"episode_{episode_number:03d}_planner_only.mp4"),
                    codec="libx264",
                    audio=False,
                    logger=None,
                )
    finally:
        env.shutdown()

    rows = _normalize_rows_for_npz(rows)
    raw_npz = write_rows_npz(rows, Path(args.output_npz))
    report = _summarize(rows, episode_summaries)
    report["source"] = "planner_state_expert_recovery"
    report["output_npz"] = str(raw_npz) if raw_npz is not None else None
    report["takeover_count"] = int(total_takeovers)
    report["takeover_origin_counts"] = dict(Counter(str(e.get("takeover_origin", "natural_takeover")) for e in episode_summaries if e.get("takeover_step", -1) >= 0))
    report["takeover_bucket_counts"] = dict(Counter(str(e.get("takeover_bucket", "none")) for e in episode_summaries if e.get("takeover_step", -1) >= 0))
    report["takeover_reason_counts"] = dict(Counter(str(e.get("planner_takeover_reason", e.get("reason", "unknown"))) for e in episode_summaries if e.get("takeover_step", -1) >= 0))
    report["natural_takeover_count"] = int(sum(str(e.get("takeover_origin", "natural_takeover")) == "natural_takeover" for e in episode_summaries if e.get("takeover_step", -1) >= 0))
    report["forced_alignment_probe_count"] = int(sum(str(e.get("takeover_origin", "natural_takeover")) == "forced_probe" for e in episode_summaries if e.get("takeover_step", -1) >= 0))
    report["natural_takeover_success_rate"] = _rate(
        [bool(e.get("success", False)) for e in episode_summaries if str(e.get("takeover_origin", "natural_takeover")) == "natural_takeover"]
    )
    report["forced_alignment_probe_success_rate"] = _rate(
        [bool(e.get("success", False)) for e in episode_summaries if str(e.get("takeover_origin", "natural_takeover")) == "forced_probe"]
    )
    report["natural_takeover_positive_rows"] = int(
        sum(
            int(e.get("positive_rows", 0))
            for e in episode_summaries
            if str(e.get("takeover_origin", "natural_takeover")) == "natural_takeover"
        )
    )
    report["forced_alignment_probe_positive_rows"] = int(
        sum(
            int(e.get("positive_rows", 0))
            for e in episode_summaries
            if str(e.get("takeover_origin", "natural_takeover")) == "forced_probe"
        )
    )
    if bool(getattr(args, "record_edgepair_labels", False)):
        edgepair_nonneg = [
            int(
                int(np.asarray(row.get("teacher_grasp_commit_edge_pair_index", -1)).reshape(())) >= 0
                and int(np.asarray(row.get("teacher_grasp_commit_edge_pair_family", -1)).reshape(())) >= 0
            )
            for row in rows
        ]
        report["edgepair_label_nonnull_rows"] = int(sum(edgepair_nonneg))
        report["edgepair_label_nonnull_rate"] = _rate(edgepair_nonneg)
        report["edgepair_label_source_counts"] = dict(
            Counter(str(np.asarray(row.get("teacher_edgepair_label_source", "none")).item()) for row in rows)
        )
    report["planner_steps_before_takeover_mean"] = float(
        np.mean([e.get("planner_steps_before_takeover", np.nan) for e in episode_summaries if "planner_steps_before_takeover" in e])
        if any("planner_steps_before_takeover" in e for e in episode_summaries)
        else np.nan
    )
    report["natural_takeover_from_near_or_micro_rate"] = _rate(
        [
            str(e.get("takeover_bucket", "none")) in {"near_contact_refine", "micro_contact_refine"}
            for e in episode_summaries
            if str(e.get("takeover_origin", "natural_takeover")) == "natural_takeover"
        ]
    )
    report["takeover_from_near_or_micro_rate"] = _rate(
        [str(e.get("takeover_bucket", "none")) in {"near_contact_refine", "micro_contact_refine"} for e in episode_summaries]
    )
    report["motion_corridor_rows"] = int(sum(int(e.get("motion_corridor_rows", 0)) for e in episode_summaries))
    report["motion_corridor_success_rate"] = _rate([e.get("motion_corridor_success", False) for e in episode_summaries])
    report["forced_alignment_probe_trace_count"] = int(sum(bool(e.get("forced_alignment_probe", False)) for e in takeover_trace))
    report["motion_corridor_phase_counts"] = dict(
        Counter(
            phase
            for e in episode_summaries
            for phase, count in dict(e.get("motion_corridor_phase_counts", {})).items()
            for _ in range(int(count))
        )
    )
    report["verified_lift_rate"] = _rate([e.get("verified_lift", False) for e in episode_summaries])
    report["attached_after_close_rate"] = _rate([e.get("attached_after_close", False) for e in episode_summaries])
    report["false_success_count"] = int(
        sum(bool(e.get("success", False)) and not bool(e.get("verified_lift", False)) for e in episode_summaries)
    )
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    Path(args.takeover_trace_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.takeover_trace_jsonl, "w", encoding="utf-8") as f:
        for item in takeover_trace:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, default="data/rlbench_data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--report_json", type=str, required=True)
    parser.add_argument("--takeover_trace_jsonl", type=str, required=True)
    parser.add_argument("--vlm_path", type=str, default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", type=str, default="pretrained_models/configs/config.json")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--demo_from_episode", type=int, default=0)
    parser.add_argument(
        "--episode_indices",
        type=str,
        default=None,
        help="Optional comma-separated list of exact demo episode indices to collect instead of a contiguous range.",
    )
    parser.add_argument("--demo_max_attempts", type=int, default=10)
    parser.add_argument("--max_rollout_steps", type=int, default=-1)
    parser.add_argument("--video_episodes", type=int, default=10)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=5110)
    parser.add_argument("--planner_use_depth", action="store_true", default=False)
    parser.add_argument("--planner_use_force", action="store_true", default=False)
    parser.add_argument("--planner_min_steps_before_takeover", type=int, default=8)
    parser.add_argument("--planner_takeover_xy_threshold", type=float, default=0.045)
    parser.add_argument("--planner_takeover_abs_z_threshold", type=float, default=0.090)
    parser.add_argument("--planner_takeover_yaw_threshold", type=float, default=0.450)
    parser.add_argument("--planner_takeover_yaw_guard_threshold", type=float, default=1.200)
    parser.add_argument("--force_alignment_probe_if_no_takeover", action="store_true", default=True)
    parser.add_argument("--no_force_alignment_probe_if_no_takeover", dest="force_alignment_probe_if_no_takeover", action="store_false")
    parser.add_argument("--allow_broad_near_takeover", action="store_true", default=False)
    parser.add_argument("--no_allow_broad_near_takeover", dest="allow_broad_near_takeover", action="store_false")
    parser.add_argument("--fallback_to_best_broad_near", action="store_true", default=False)
    parser.add_argument("--no_fallback_to_best_broad_near", dest="fallback_to_best_broad_near", action="store_false")
    parser.add_argument("--best_broad_near_min_step", type=int, default=20)
    parser.add_argument("--planner_takeover_motion_steps", type=int, default=80)
    parser.add_argument("--motion_corridor_force_descend_after_steps", type=int, default=12)
    parser.add_argument("--motion_corridor_descend_xy_threshold", type=float, default=0.010)
    parser.add_argument("--motion_corridor_descend_yaw_threshold", type=float, default=0.100)
    parser.add_argument("--gripper_close_threshold", type=float, default=0.5)
    parser.add_argument("--grasp_recovery_close_xy_threshold", type=float, default=0.0032)
    parser.add_argument("--grasp_recovery_close_z_threshold", type=float, default=0.0035)
    parser.add_argument("--grasp_recovery_close_yaw_threshold", type=float, default=0.025)
    parser.add_argument("--grasp_recovery_close_steps", type=int, default=18)
    parser.add_argument("--grasp_recovery_min_close_steps", type=int, default=4)
    parser.add_argument("--grasp_recovery_lift_steps", type=int, default=14)
    parser.add_argument("--grasp_recovery_lift_step", type=float, default=0.0030)
    parser.add_argument("--grasp_recovery_verify_lift_threshold", type=float, default=0.010)
    parser.add_argument("--grasp_recovery_verify_consecutive_steps", type=int, default=2)
    parser.add_argument("--teacher_grasp_ready_threshold", type=float, default=0.55)
    parser.add_argument("--teacher_close_contact_depth_threshold", type=float, default=0.020)
    parser.add_argument("--record_edgepair_labels", action="store_true", default=False)
    parser.add_argument("--edgepair_label_target_mode", type=str, default="commit", choices=["commit", "pregrasp"])
    parser.add_argument("--depth_max", type=float, default=1.0)
    parser.add_argument("--perturb_expert_steps", type=int, default=48)
    parser.add_argument("--force_spike_threshold", type=float, default=2.5)
    parser.add_argument("--jam_force_threshold", type=float, default=3.5)
    parser.add_argument("--light_contact_force", type=float, default=0.45)
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
    parser.add_argument("--align_z_step", type=float, default=0.0008)
    parser.add_argument("--contact_z_step", type=float, default=0.0006)
    parser.add_argument("--spiral_step", type=float, default=0.0006)
    parser.add_argument("--unjam_lift_step", type=float, default=0.003)
    parser.add_argument("--unjam_lateral_step", type=float, default=0.0012)
    parser.add_argument("--unjam_yaw_step", type=float, default=0.010)
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
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
