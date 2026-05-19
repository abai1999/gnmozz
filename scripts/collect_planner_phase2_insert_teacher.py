#!/usr/bin/env python3
"""Collect true planner-phase2 insert teacher data.

This collector is deliberately different from demo perturb collection:

1. Reset to an RLBench demo initial state.
2. Let the frozen planner run the first phase until the grasp teacher can take over.
3. Use the privileged phase-1 teacher to grasp/verify the ring.
4. Let the frozen planner continue into the insertion phase.
5. Preserve a planner-only phase-2 MP4, then repeat the same setup and let an
   insert teacher take over before insertion.

The positive insert rows are therefore anchored in planner phase-2 states, not
already-inserted successful-demo frames.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("VLA_PLATFORM", "RLBENCH")

# ── CoppeliaSim / PyRep environment setup (must happen before rlbench import) ─
_COPPELIASIM_ROOT = os.environ.setdefault("COPPELIASIM_ROOT", os.path.expanduser("~/CoppeliaSim"))
_base_ld = f"{_COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
_existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _existing_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{_existing_ld}:{_base_ld}"
else:
    os.environ["LD_LIBRARY_PATH"] = _base_ld
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _COPPELIASIM_ROOT)
os.environ.setdefault("QT_PLUGIN_PATH", _COPPELIASIM_ROOT)

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from scripts.collect_alignment_tc_privileged_expert_rollout import (
    _bucket_from_delta,
    _demo_grasp_commit_index,
    _run_expert_from_current_obs,
    _scene_observation,
    _summarize,
    _target_pose_from_handle,
    apply_yaw_symmetry_to_delta,
    pose_delta_local_between,
    resolve_live_target_handle,
    write_rows_npz,
)
from scripts.collect_planner_state_expert_recovery import (
    _make_env,
    _planner_takeover_ready,
    _run_takeover_from_obs,
    _normalize_rows_for_npz,
)
from scripts.collect_residual_data import load_planner
from scripts.evaluate_rlbench import delta_to_absolute, predict_actions, process_obs
from prismatic.robot.residual_transforms import world_delta_to_local
from prismatic.vla.constants import FORCE_DIM, NUM_ACTIONS_CHUNK
from rlbench.backend.exceptions import InvalidActionError


def _valid_pose7(pose) -> bool:
    if pose is None:
        return False
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    return bool(arr.size >= 7 and np.all(np.isfinite(arr[:7])) and np.linalg.norm(arr[3:7]) > 1e-6)


def _insert_target_from_demo(demo) -> np.ndarray:
    # EE target for insertion.  The final demo gripper pose is an action target,
    # not a rollout start state, so using it here is OK.
    final_pose = np.asarray(demo[-1].gripper_pose, dtype=np.float32).reshape(7)
    if _valid_pose7(final_pose):
        return final_pose
    for obs in reversed(demo):
        pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(7)
        if _valid_pose7(pose):
            return pose
    raise ValueError("No valid insert target pose found in demo")


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if len(frames) <= 1:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ImageSequenceClip(frames, fps=int(fps)).write_videofile(str(path), codec="libx264", audio=False, logger=None)


def _decode_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _phase2_supervision_bucket(row: dict, is_verified_tail: bool) -> str:
    stage_bucket = _decode_str(row.get("stage_bucket", ""))
    if is_verified_tail:
        return "insert_commit_verified"
    if stage_bucket == "micro_contact_refine":
        return "insert_precommit_micro"
    if stage_bucket == "near_contact_refine":
        return "insert_near_align"
    if stage_bucket == "broad_near":
        return "insert_broad_near_aux"
    return "insert_negative"


def _reweight_phase2_teacher_rows(rows: list[dict], success: bool, args) -> list[dict]:
    """Keep broad-near takeover for success, but focus exported supervision on near/micro/verified windows."""
    if not rows:
        return rows
    total = len(rows)
    tail = max(1, int(args.phase2_positive_tail_steps))
    start = max(0, total - tail)
    for idx, row in enumerate(rows):
        row["student_supervision_bucket"] = np.asarray("insert_negative")
        # Failed insert rows stay available for risk/stop/confidence only.
        if not success:
            row["action_imitation_weight"] = np.asarray(0.0, dtype=np.float32)
            continue
        if idx < start:
            row["action_imitation_weight"] = np.asarray(0.0, dtype=np.float32)
            continue
        is_last = idx == total - 1 and bool(float(row.get("success_label", 0.0)) > 0.5)
        bucket = _phase2_supervision_bucket(row, is_last)
        row["student_supervision_bucket"] = np.asarray(bucket)
        if bucket == "insert_commit_verified":
            weight = float(args.phase2_positive_weight_verified)
        elif bucket == "insert_precommit_micro":
            weight = float(args.phase2_positive_weight_micro)
        elif bucket == "insert_near_align":
            weight = float(args.phase2_positive_weight_near)
        elif bucket == "insert_broad_near_aux":
            weight = float(args.phase2_positive_weight_broad_near_aux)
        else:
            weight = 0.0
        if not bool(row.get("teacher_improves_two_axis", False)):
            weight *= float(args.phase2_positive_single_axis_scale)
        row["action_imitation_weight"] = np.asarray(weight, dtype=np.float32)
    return rows


def _planner_action_queue_step(
    *,
    vla,
    processor,
    action_head,
    proprio_projector,
    instruction: str,
    obs,
    norm_stats,
    force_buffer,
    action_queue: list[np.ndarray],
    use_depth: bool,
    use_force: bool,
):
    front_pil, wrist_pil, proprio, depth_tensor_224, force_hist, depth_tensor_96, raw_force = process_obs(
        obs,
        norm_stats,
        force_buffer,
        use_depth=bool(use_depth),
        use_force=bool(use_force),
        depth_max=1.0,
    )
    if len(action_queue) == 0:
        actions = predict_actions(
            vla,
            processor,
            action_head,
            proprio_projector,
            front_pil,
            wrist_pil,
            proprio,
            depth_tensor_224 if use_depth else None,
            force_hist if use_force else None,
            instruction,
            unnorm_key="rlbench",
        )
        action_queue.extend(
            [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), NUM_ACTIONS_CHUNK))]
        )
    if not action_queue:
        return None, proprio, force_hist, depth_tensor_96, raw_force
    return action_queue.pop(0), proprio, force_hist, depth_tensor_96, raw_force




def _prepare_phase1_with_teacher(
    *,
    task,
    demo,
    ep_index: int,
    vla,
    processor,
    action_head,
    proprio_projector,
    norm_stats,
    args,
    video_dir: Path,
    branch_name: str,
) -> tuple[object | None, dict, list[np.ndarray]]:
    """Phase-1 bridge that intentionally matches 20260514n semantics."""
    close_idx = _demo_grasp_commit_index(demo, threshold=float(args.gripper_close_threshold))
    if close_idx is None:
        return None, {"success": False, "reason": "missing_demo_close"}, []
    target_handle = resolve_live_target_handle(task)
    target_ee = np.asarray(demo[int(close_idx)].gripper_pose, dtype=np.float32).reshape(7)
    descs, obs = task.reset_to_demo(demo)
    instruction = descs[0] if isinstance(descs, list) else str(descs)
    force_buffer = deque(maxlen=256)
    action_queue: list[np.ndarray] = []
    rows: list[dict] = []
    episode_summaries: list[dict] = []
    takeover_trace: list[dict] = []
    frames: list[np.ndarray] = []

    for step in range(int(args.phase1_planner_max_steps)):
        if args.record_video:
            frames.append(obs.front_rgb.copy())
        base_action, *_ = _planner_action_queue_step(
            vla=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            instruction=instruction,
            obs=obs,
            norm_stats=norm_stats,
            force_buffer=force_buffer,
            action_queue=action_queue,
            use_depth=args.planner_use_depth,
            use_force=args.planner_use_force,
        )
        if base_action is None:
            break
        delta_to_close = apply_yaw_symmetry_to_delta(
            pose_delta_local_between(obs.gripper_pose, target_ee),
            np.pi / 2.0,
        )
        ready, reason = _planner_takeover_ready(
            delta6=delta_to_close,
            step_idx=step,
            min_steps_before_takeover=int(args.planner_min_steps_before_takeover),
            allow_broad_near=True,
            takeover_xy_threshold=float(args.planner_takeover_xy_threshold),
            takeover_abs_z_threshold=float(args.planner_takeover_abs_z_threshold),
            takeover_yaw_threshold=float(args.planner_takeover_yaw_threshold),
            takeover_yaw_guard_threshold=float(args.planner_takeover_yaw_guard_threshold),
        )
        if (
            not ready
            and int(args.phase1_force_teacher_after_steps) >= 0
            and step >= int(args.phase1_force_teacher_after_steps)
        ):
            ready = True
            reason = "phase1_forced_teacher_after_steps"
        if ready:
            summary, teacher_frames = _run_takeover_from_obs(
                task=task,
                obs=obs,
                target_handle=target_handle,
                target_ee=target_ee,
                episode_index=int(ep_index),
                takeover_step=int(step),
                takeover_bucket=str(_bucket_from_delta(delta_to_close)),
                takeover_reason=str(reason),
                takeover_origin=f"{branch_name}_phase1_teacher",
                takeover_weight_scale=0.0,
                takeover_delta6=delta_to_close,
                demo_close_index=int(close_idx),
                args=args,
                rows=rows,
                episode_summaries=episode_summaries,
                takeover_trace=takeover_trace,
                video_dir=video_dir,
            )
            frames.extend(teacher_frames)
            obs_after = _scene_observation(task)
            _write_video(frames, video_dir / f"episode_{ep_index:03d}_{branch_name}_phase1_teacher.mp4", args.video_fps)
            summary["phase1_teacher_success"] = bool(summary.get("success", False))
            return (obs_after, summary, frames) if bool(summary.get("success", False)) else (None, summary, frames)
        try:
            obs, reward, terminate = task.step(delta_to_absolute(base_action, obs.gripper_pose))
        except InvalidActionError:
            return None, {"success": False, "reason": "phase1_planner_invalid", "step": int(step)}, frames
    return None, {"success": False, "reason": "phase1_no_takeover"}, frames


def _run_phase2_planner(
    *,
    task,
    obs,
    demo,
    insert_target,
    ep_index: int,
    vla,
    processor,
    action_head,
    proprio_projector,
    norm_stats,
    args,
    video_dir: Path,
) -> tuple[dict, list[np.ndarray]]:
    descs = ["put the ring on the red spoke"]
    instruction = descs[0]
    force_buffer = deque(maxlen=256)
    action_queue: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    invalid = 0
    best_delta = None
    for step in range(int(args.phase2_planner_steps)):
        if args.record_video:
            frames.append(obs.front_rgb.copy())
        delta = apply_yaw_symmetry_to_delta(pose_delta_local_between(obs.gripper_pose, insert_target), np.pi / 2.0)
        best_delta = delta
        base_action, *_ = _planner_action_queue_step(
            vla=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            instruction=instruction,
            obs=obs,
            norm_stats=norm_stats,
            force_buffer=force_buffer,
            action_queue=action_queue,
            use_depth=args.planner_use_depth,
            use_force=args.planner_use_force,
        )
        if base_action is None:
            break
        try:
            obs, reward, terminate = task.step(delta_to_absolute(base_action, obs.gripper_pose))
            if reward > 0 or terminate:
                break
        except InvalidActionError:
            invalid += 1
            break
    if args.record_video:
        _write_video(frames, video_dir / f"episode_{ep_index:03d}_phase2_planner_only.mp4", args.video_fps)
    return {
        "phase2_planner_steps": int(len(frames)),
        "phase2_planner_invalid": int(invalid),
        "phase2_planner_final_xy": float(np.linalg.norm(best_delta[:2])) if best_delta is not None else None,
        "phase2_planner_final_z": float(abs(best_delta[2])) if best_delta is not None else None,
        "phase2_planner_final_yaw": float(abs(best_delta[5])) if best_delta is not None else None,
    }, frames


def _run_phase2_teacher(
    *,
    task,
    obs,
    insert_target,
    ep_index: int,
    vla,
    processor,
    action_head,
    proprio_projector,
    norm_stats,
    args,
    video_dir: Path,
) -> tuple[list[dict], dict, list[np.ndarray]]:
    instruction = "put the ring on the red spoke"
    force_buffer = deque(maxlen=256)
    action_queue: list[np.ndarray] = []
    planner_frames: list[np.ndarray] = []
    target_handle = resolve_live_target_handle(task)
    takeover_obs = None
    takeover_step = -1
    takeover_delta = None
    for step in range(int(args.phase2_planner_steps)):
        if args.record_video:
            planner_frames.append(obs.front_rgb.copy())
        delta = apply_yaw_symmetry_to_delta(pose_delta_local_between(obs.gripper_pose, insert_target), np.pi / 2.0)
        xy, z, yaw = float(np.linalg.norm(delta[:2])), float(abs(delta[2])), float(abs(delta[5]))
        if (
            step >= int(args.phase2_min_planner_steps)
            and z <= float(args.phase2_takeover_abs_z_threshold)
            and xy <= float(args.phase2_takeover_xy_threshold)
            and yaw <= float(args.phase2_takeover_yaw_threshold)
        ):
            takeover_obs = obs
            takeover_step = int(step)
            takeover_delta = delta
            break
        base_action, *_ = _planner_action_queue_step(
            vla=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            instruction=instruction,
            obs=obs,
            norm_stats=norm_stats,
            force_buffer=force_buffer,
            action_queue=action_queue,
            use_depth=args.planner_use_depth,
            use_force=args.planner_use_force,
        )
        if base_action is None:
            break
        try:
            obs, reward, terminate = task.step(delta_to_absolute(base_action, obs.gripper_pose))
            if reward > 0 or terminate:
                break
        except InvalidActionError:
            break
    if takeover_obs is None:
        takeover_obs = obs
        takeover_step = len(planner_frames)
        takeover_delta = apply_yaw_symmetry_to_delta(pose_delta_local_between(obs.gripper_pose, insert_target), np.pi / 2.0)

    rows, summary, teacher_frames = _run_expert_from_current_obs(
        task=task,
        obs=takeover_obs,
        target_ee=insert_target,
        success_centre=_target_pose_from_handle(target_handle) if target_handle is not None else insert_target,
        episode_index=ep_index,
        init_bucket=f"planner_phase2_{_bucket_from_delta(takeover_delta)}",
        phase="insert_commit",
        args=args,
        record_frames=bool(args.record_video),
    )
    for row in rows:
        row["takeover_origin"] = np.asarray("planner_phase2_insert_teacher")
        row["phase2_planner_steps_before_takeover"] = np.asarray(int(takeover_step), dtype=np.int64)
    combined = list(planner_frames) + list(teacher_frames)
    if args.record_video:
        _write_video(combined, video_dir / f"episode_{ep_index:03d}_phase2_insert_teacher.mp4", args.video_fps)
    summary.update(
        {
            "phase2_takeover_step": int(takeover_step),
            "phase2_takeover_bucket": str(_bucket_from_delta(takeover_delta)),
            "phase2_takeover_xy": float(np.linalg.norm(takeover_delta[:2])),
            "phase2_takeover_z": float(abs(takeover_delta[2])),
            "phase2_takeover_yaw": float(abs(takeover_delta[5])),
        }
    )
    return rows, summary, combined


def collect(args) -> dict:
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        args.checkpoint_dir,
        args.vlm_path,
        args.config_path,
        args.planner_use_depth,
        args.planner_use_force,
    )
    env, task = _make_env(args.task_name, Path(args.data_root))
    rows: list[dict] = []
    summaries: list[dict] = []
    try:
        for ep in range(int(args.num_episodes)):
            demos = task.get_demos(
                1,
                live_demos=True,
                random_selection=False,
                from_episode_number=int(args.demo_from_episode) + ep,
                max_attempts=max(10, int(args.demo_max_attempts)),
            )
            if not demos:
                summaries.append({"episode_index": ep, "success": False, "reason": "missing_demo"})
                continue
            demo = demos[0]
            insert_target = _insert_target_from_demo(demo)

            obs_planner, phase1_planner_summary, phase1_planner_frames = _prepare_phase1_with_teacher(
                task=task,
                demo=demo,
                ep_index=ep,
                vla=vla,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                norm_stats=norm_stats,
                args=args,
                video_dir=video_dir,
                branch_name="planner_compare",
            )
            planner_summary = {}
            planner_phase2_frames: list[np.ndarray] = []
            if obs_planner is not None and phase1_planner_summary.get("success", False):
                planner_summary, planner_phase2_frames = _run_phase2_planner(
                    task=task,
                    obs=obs_planner,
                    demo=demo,
                    insert_target=insert_target,
                    ep_index=ep,
                    vla=vla,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    norm_stats=norm_stats,
                    args=args,
                    video_dir=video_dir,
                )
                if args.record_video:
                    _write_video(
                        list(phase1_planner_frames) + list(planner_phase2_frames),
                        video_dir / f"episode_{ep:03d}_planner_full_task.mp4",
                        args.video_fps,
                    )

            obs_teacher, phase1_teacher_summary, phase1_teacher_frames = _prepare_phase1_with_teacher(
                task=task,
                demo=demo,
                ep_index=ep,
                vla=vla,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                norm_stats=norm_stats,
                args=args,
                video_dir=video_dir,
                branch_name="teacher_compare",
            )
            teacher_rows = []
            teacher_summary = {}
            teacher_phase2_frames: list[np.ndarray] = []
            if obs_teacher is not None and phase1_teacher_summary.get("success", False):
                teacher_rows, teacher_summary, teacher_phase2_frames = _run_phase2_teacher(
                    task=task,
                    obs=obs_teacher,
                    insert_target=insert_target,
                    ep_index=ep,
                    vla=vla,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    norm_stats=norm_stats,
                    args=args,
                    video_dir=video_dir,
                )
                teacher_rows = _reweight_phase2_teacher_rows(
                    teacher_rows,
                    bool(teacher_summary.get("success", False)),
                    args,
                )
                rows.extend(teacher_rows)
                if args.record_video:
                    _write_video(
                        list(phase1_teacher_frames) + list(teacher_phase2_frames),
                        video_dir / f"episode_{ep:03d}_teacher_full_task.mp4",
                        args.video_fps,
                    )
            episode_summary = {
                "episode_index": int(ep),
                "phase1_planner_branch_success": bool(phase1_planner_summary.get("success", False)),
                "phase1_teacher_branch_success": bool(phase1_teacher_summary.get("success", False)),
                "phase1_planner_branch_reason": str(phase1_planner_summary.get("reason", "")),
                "phase1_teacher_branch_reason": str(phase1_teacher_summary.get("reason", "")),
                "phase2_teacher_success": bool(teacher_summary.get("success", False)),
                "phase2_teacher_rows": int(len(teacher_rows)),
                **{f"planner_{k}": v for k, v in planner_summary.items()},
                **{f"teacher_{k}": v for k, v in teacher_summary.items()},
            }
            summaries.append(episode_summary)
            print(json.dumps(episode_summary, sort_keys=True), flush=True)
    finally:
        env.shutdown()

    rows = _normalize_rows_for_npz(rows)
    raw_npz = write_rows_npz(rows, Path(args.output_npz))
    report = _summarize(rows, summaries)
    report["source"] = "planner_phase2_insert_teacher"
    report["output_npz"] = str(raw_npz) if raw_npz is not None else None
    report["phase2_teacher_success_rate"] = float(np.mean([s.get("phase2_teacher_success", False) for s in summaries])) if summaries else 0.0
    report["phase2_takeover_bucket_counts"] = dict(Counter(str(s.get("teacher_phase2_takeover_bucket", "none")) for s in summaries))
    report["student_supervision_bucket_counts"] = dict(
        Counter(
            _decode_str(r.get("student_supervision_bucket", "insert_negative"))
            for r in rows
            if float(r.get("action_imitation_weight", 0.0)) > 0.0
        )
    )
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
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
    parser.add_argument("--vlm_path", type=str, default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", type=str, default="pretrained_models/configs/config.json")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--demo_from_episode", type=int, default=0)
    parser.add_argument("--demo_max_attempts", type=int, default=10)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--planner_use_depth", action="store_true", default=False)
    parser.add_argument("--planner_use_force", action="store_true", default=False)
    parser.add_argument("--phase1_planner_max_steps", type=int, default=260)
    parser.add_argument("--phase1_force_teacher_after_steps", type=int, default=80)
    parser.add_argument("--phase1_hard_grasp_fallback", action="store_true", default=True)
    parser.add_argument("--phase1_hard_pregrasp_lift", type=float, default=0.055)
    parser.add_argument("--phase1_hard_move_steps", type=int, default=18)
    parser.add_argument("--phase1_hard_close_steps", type=int, default=8)
    parser.add_argument("--phase1_hard_lift_steps", type=int, default=10)
    parser.add_argument("--phase1_hard_lift_height", type=float, default=0.060)
    parser.add_argument("--phase1_demo_replay_fallback", action="store_true", default=True)
    parser.add_argument("--phase1_demo_replay_postclose_steps", type=int, default=12)
    parser.add_argument("--phase2_planner_steps", type=int, default=220)
    parser.add_argument("--phase2_min_planner_steps", type=int, default=8)
    parser.add_argument("--phase2_takeover_xy_threshold", type=float, default=0.060)
    parser.add_argument("--phase2_takeover_abs_z_threshold", type=float, default=0.090)
    parser.add_argument("--phase2_takeover_yaw_threshold", type=float, default=0.60)
    parser.add_argument("--insert_teacher_steps", type=int, default=160)
    parser.add_argument("--insert_teacher_keep_closed", action="store_true", default=True)
    parser.add_argument("--planner_min_steps_before_takeover", type=int, default=8)
    parser.add_argument("--planner_takeover_xy_threshold", type=float, default=0.045)
    parser.add_argument("--planner_takeover_abs_z_threshold", type=float, default=0.090)
    parser.add_argument("--planner_takeover_yaw_threshold", type=float, default=0.450)
    parser.add_argument("--planner_takeover_yaw_guard_threshold", type=float, default=1.200)
    parser.add_argument("--takeover_trace_jsonl", type=str, default="/tmp/planner_phase2_insert_takeover_trace.jsonl")
    parser.add_argument("--video_episodes", type=int, default=6)
    parser.add_argument("--depth_max", type=float, default=1.0)
    parser.add_argument("--gripper_close_threshold", type=float, default=0.5)
    parser.add_argument("--perturb_expert_steps", type=int, default=96)
    parser.add_argument("--run_full_horizon_on_success", action="store_true", default=False)
    parser.add_argument("--ignore_reward_success", action="store_true", default=True)
    parser.add_argument("--force_spike_threshold", type=float, default=3.0)
    parser.add_argument("--success_xy_threshold", type=float, default=0.004)
    parser.add_argument("--success_z_threshold", type=float, default=0.006)
    parser.add_argument("--success_yaw_threshold", type=float, default=0.04)
    parser.add_argument("--expert_k_xy", type=float, default=0.35)
    parser.add_argument("--expert_k_z", type=float, default=0.28)
    parser.add_argument("--expert_k_yaw", type=float, default=0.18)
    parser.add_argument("--expert_max_pos_step", type=float, default=0.003)
    parser.add_argument("--expert_max_yaw_step", type=float, default=0.010)
    parser.add_argument("--jam_force_threshold", type=float, default=3.0)
    parser.add_argument("--contact_force_threshold", type=float, default=0.8)
    parser.add_argument("--light_contact_force", type=float, default=0.45)
    parser.add_argument("--unjam_lift_step", type=float, default=0.006)
    parser.add_argument("--unjam_lateral_step", type=float, default=0.0015)
    parser.add_argument("--unjam_yaw_step", type=float, default=0.006)
    parser.add_argument("--align_xy_threshold", type=float, default=0.006)
    parser.add_argument("--align_yaw_threshold", type=float, default=0.04)
    parser.add_argument("--align_z_step", type=float, default=0.0015)
    parser.add_argument("--k_xy_align", type=float, default=0.35)
    parser.add_argument("--k_z_hold", type=float, default=0.08)
    parser.add_argument("--k_yaw_align", type=float, default=0.18)
    parser.add_argument("--k_xy_descend", type=float, default=0.18)
    parser.add_argument("--k_z_descend", type=float, default=0.30)
    parser.add_argument("--k_yaw_descend", type=float, default=0.12)
    parser.add_argument("--k_xy_contact", type=float, default=0.16)
    parser.add_argument("--k_z_contact", type=float, default=0.12)
    parser.add_argument("--k_yaw_contact", type=float, default=0.10)
    parser.add_argument("--k_xy_commit", type=float, default=0.10)
    parser.add_argument("--k_z_commit", type=float, default=0.12)
    parser.add_argument("--k_yaw_commit", type=float, default=0.08)
    parser.add_argument("--spiral_step", type=float, default=0.0004)
    parser.add_argument("--contact_z_step", type=float, default=0.0010)
    parser.add_argument("--expert_yaw_sign", type=float, default=-1.0)
    parser.add_argument("--max_pos_step", type=float, default=0.003)
    parser.add_argument("--max_yaw_step", type=float, default=0.010)
    parser.add_argument("--commit_xy_threshold", type=float, default=0.006)
    parser.add_argument("--commit_z_threshold", type=float, default=0.010)
    parser.add_argument("--commit_yaw_threshold", type=float, default=0.06)
    parser.add_argument("--grasp_recovery_close_xy_threshold", type=float, default=0.0032)
    parser.add_argument("--grasp_recovery_close_z_threshold", type=float, default=0.0035)
    parser.add_argument("--grasp_recovery_close_yaw_threshold", type=float, default=0.025)
    parser.add_argument("--grasp_recovery_close_steps", type=int, default=18)
    parser.add_argument("--grasp_recovery_min_close_steps", type=int, default=2)
    parser.add_argument("--grasp_recovery_lift_steps", type=int, default=12)
    parser.add_argument("--grasp_recovery_lift_step", type=float, default=0.006)
    parser.add_argument("--grasp_recovery_verify_lift_threshold", type=float, default=0.006)
    parser.add_argument("--grasp_recovery_verify_consecutive_steps", type=int, default=2)
    parser.add_argument("--planner_takeover_motion_steps", type=int, default=80)
    parser.add_argument("--motion_corridor_force_descend_after_steps", type=int, default=4)
    parser.add_argument("--motion_corridor_descend_xy_threshold", type=float, default=0.010)
    parser.add_argument("--motion_corridor_descend_yaw_threshold", type=float, default=0.08)
    parser.add_argument("--teacher_close_contact_depth_threshold", type=float, default=0.020)
    parser.add_argument("--teacher_grasp_ready_threshold", type=float, default=0.55)
    parser.add_argument("--phase2_positive_tail_steps", type=int, default=24)
    parser.add_argument("--phase2_positive_weight_broad_near_aux", type=float, default=0.15)
    parser.add_argument("--phase2_positive_weight_near", type=float, default=1.0)
    parser.add_argument("--phase2_positive_weight_micro", type=float, default=1.35)
    parser.add_argument("--phase2_positive_weight_verified", type=float, default=1.75)
    parser.add_argument("--phase2_positive_single_axis_scale", type=float, default=0.5)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
