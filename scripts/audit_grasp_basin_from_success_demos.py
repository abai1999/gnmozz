#!/usr/bin/env python3
"""Audit a grasp basin profile from successful RLBench demos.

This script replays successful demos, extracts the grasp-commit window around
the first close transition, and turns that window into a calibrated basin
profile. The profile is intended to hard-gate close-ready teacher behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
os.environ.setdefault("COPPELIASIM_ROOT", os.path.expanduser("~/CoppeliaSim"))
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.environ["COPPELIASIM_ROOT"])
os.environ.setdefault("QT_PLUGIN_PATH", os.environ["COPPELIASIM_ROOT"])
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.insert_onto_square_peg import InsertOntoSquarePeg


TASK_MAP = {
    "insert_onto_square_peg": InsertOntoSquarePeg,
}


def _lazy_import_tasks():
    return None


def _rate(mask) -> float:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    return float(arr.mean()) if arr.size else float("nan")


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
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _safe_task_low_dim_pose_7d(obs) -> np.ndarray:
    low = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).reshape(-1)
    if low.size >= 7 and np.all(np.isfinite(low[:7])) and np.linalg.norm(low[3:7]) > 1e-6:
        return low[:7].astype(np.float32)
    return np.full((7,), np.nan, dtype=np.float32)


def _resolve_live_target_handle(task):
    task_obj = getattr(task, "_task", task)
    if hasattr(task_obj, "_square_ring"):
        return task_obj._square_ring
    graspables = getattr(task_obj, "_graspable_objects", None)
    if graspables:
        return graspables[0]
    return None


def _task_low_dim_pose_index_for_target(obs, target_pose_7d: np.ndarray) -> int | None:
    low = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).reshape(-1)
    target = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
    if low.size < 7 or low.size % 7 != 0:
        return None
    best_idx = None
    best_dist = float("inf")
    for idx in range(low.size // 7):
        chunk = low[idx * 7 : (idx + 1) * 7]
        if not np.all(np.isfinite(chunk)) or np.linalg.norm(chunk[3:7]) <= 1e-6:
            continue
        dist = float(np.linalg.norm(chunk[:3] - target[:3]))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _task_low_dim_pose_at_index(obs, pose_index: int | None) -> np.ndarray:
    low = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).reshape(-1)
    if pose_index is None or low.size < (pose_index + 1) * 7:
        return _safe_task_low_dim_pose_7d(obs)
    chunk = low[pose_index * 7 : (pose_index + 1) * 7]
    if np.all(np.isfinite(chunk)) and np.linalg.norm(chunk[3:7]) > 1e-6:
        return chunk.astype(np.float32)
    return _safe_task_low_dim_pose_7d(obs)


def pose_delta_local_between(current_pose_7d, target_pose_7d):
    current_pose_7d = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
    target_pose_7d = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
    delta_pos_world = target_pose_7d[:3] - current_pose_7d[:3]
    r_cur = Rotation.from_quat(current_pose_7d[3:7])
    r_tgt = Rotation.from_quat(target_pose_7d[3:7])
    delta_rot = (r_tgt * r_cur.inv()).as_rotvec().astype(np.float32)
    delta_pos_local = r_cur.inv().apply(delta_pos_world.astype(np.float32)).astype(np.float32)
    return np.concatenate([delta_pos_local, delta_rot], axis=0).astype(np.float32)


def _demo_grasp_commit_index(demo, *, threshold: float = 0.5) -> int | None:
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


def _format_delta(delta_local_6d: np.ndarray) -> dict[str, float]:
    delta = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    return {
        "xy": float(np.linalg.norm(delta[:2])),
        "abs_z": float(abs(delta[2])),
        "yaw": float(abs(delta[5])),
    }


def _as_row(
    *,
    demo_index: int,
    frame_index: int,
    close_index: int,
    phase: str,
    demo_success: bool,
    obs,
    object_pose_7d: np.ndarray,
    close_object_pose_7d: np.ndarray,
    close_gripper_pose_7d: np.ndarray,
    depth_proximity: float,
    force_norm: float,
    attached_after_close: bool,
    verified_lift: bool,
    object_in_finger_region: bool,
    close_xy_threshold: float,
    close_abs_z_threshold: float,
    close_yaw_threshold: float,
    close_contact_depth_threshold: float,
) -> dict:
    gripper_pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(7)
    delta_local = pose_delta_local_between(gripper_pose, object_pose_7d)
    delta_local = delta_local.astype(np.float32)
    geom = _format_delta(delta_local)
    grasp_ready_score = 0.0
    if np.all(np.isfinite(object_pose_7d[:7])):
        xy_score = float(np.clip(1.0 - geom["xy"] / max(float(close_xy_threshold), 1e-6), 0.0, 1.0))
        z_score = float(np.clip(1.0 - geom["abs_z"] / max(float(close_abs_z_threshold), 1e-6), 0.0, 1.0))
        yaw_score = float(np.clip(1.0 - geom["yaw"] / max(float(close_yaw_threshold), 1e-6), 0.0, 1.0))
        depth_score = float(np.clip(1.0 - float(depth_proximity) / max(float(close_contact_depth_threshold), 1e-6), 0.0, 1.0))
        grasp_ready_score = float(np.clip(0.45 * xy_score + 0.30 * z_score + 0.15 * yaw_score + 0.10 * depth_score, 0.0, 1.0))
    return {
        "demo_index": np.asarray(int(demo_index), dtype=np.int64),
        "frame_index": np.asarray(int(frame_index), dtype=np.int64),
        "close_index": np.asarray(int(close_index), dtype=np.int64),
        "window_offset": np.asarray(int(frame_index - close_index), dtype=np.int64),
        "phase": np.asarray(str(phase)),
        "demo_success": np.asarray(float(bool(demo_success)), dtype=np.float32),
        "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
        "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
        "wrist_depth": np.asarray(obs.wrist_depth, dtype=np.float32),
        "gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
        "gripper_pose_7d": gripper_pose.astype(np.float32),
        "object_pose_7d": object_pose_7d.astype(np.float32),
        "object_to_gripper_delta_local_6d": delta_local.astype(np.float32),
        "object_delta_xy": np.asarray(float(geom["xy"]), dtype=np.float32),
        "object_delta_abs_z": np.asarray(float(geom["abs_z"]), dtype=np.float32),
        "object_delta_yaw": np.asarray(float(geom["yaw"]), dtype=np.float32),
        "depth_proximity": np.asarray(float(depth_proximity), dtype=np.float32),
        "force_norm": np.asarray(float(force_norm), dtype=np.float32),
        "attached_after_close": np.asarray(float(bool(attached_after_close)), dtype=np.float32),
        "verified_lift": np.asarray(float(bool(verified_lift)), dtype=np.float32),
        "object_in_finger_region": np.asarray(float(bool(object_in_finger_region)), dtype=np.float32),
        "gripper_aperture_ready": np.asarray(float(float(obs.gripper_open) >= 0.5), dtype=np.float32),
        "close_xy_threshold": np.asarray(float(close_xy_threshold), dtype=np.float32),
        "close_abs_z_threshold": np.asarray(float(close_abs_z_threshold), dtype=np.float32),
        "close_yaw_threshold": np.asarray(float(close_yaw_threshold), dtype=np.float32),
        "close_contact_depth_threshold": np.asarray(float(close_contact_depth_threshold), dtype=np.float32),
        "grasp_ready_score": np.asarray(float(grasp_ready_score), dtype=np.float32),
    }


def _rows_to_npz(rows: list[dict], output_npz: Path) -> Path:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    out = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        first = values[0]
        if isinstance(first, (str, np.str_)):
            out[key] = np.asarray(values, dtype=object)
        else:
            out[key] = np.asarray(values)
    np.savez_compressed(output_npz, **out)
    return output_npz


def audit(args) -> dict:
    _lazy_import_tasks()
    if args.task_name not in TASK_MAP:
        raise ValueError(f"Unknown task {args.task_name!r}; available={sorted(TASK_MAP)}")

    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.task_low_dim_state = True
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
    preclose_deltas = []
    close_deltas = []
    postclose_deltas = []
    preclose_depths = []
    close_depths = []
    preclose_success_flags = []
    close_success_flags = []
    postclose_success_flags = []
    video_dir = Path(args.output_dir) / "videos"
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    try:
        demos = task.get_demos(
            int(args.num_demos),
            live_demos=True,
            random_selection=False,
            from_episode_number=int(args.demo_from_episode),
            max_attempts=max(10, int(args.num_demos) * 3),
        )
        for demo_index, demo in enumerate(demos):
            task.reset_to_demo(demo)
            live_target_handle = _resolve_live_target_handle(task)
            target_pose = None
            if live_target_handle is not None:
                try:
                    target_pose = np.asarray(live_target_handle.get_pose(), dtype=np.float32).reshape(7)
                except Exception:
                    target_pose = None
            pose_index = None
            if target_pose is not None:
                pose_index = _task_low_dim_pose_index_for_target(demo[0], target_pose)
            close_index = _demo_grasp_commit_index(demo, threshold=float(args.gripper_close_threshold))
            if close_index is None:
                episode_summaries.append(
                    {
                        "demo_index": int(demo_index + int(args.demo_from_episode)),
                        "rows": 0,
                        "success": False,
                        "reason": "missing_close_transition",
                    }
                )
                continue
            demo_success = True
            close_object_pose = _task_low_dim_pose_at_index(demo[close_index], pose_index)
            if not np.all(np.isfinite(close_object_pose[:7])) or np.linalg.norm(close_object_pose[3:7]) <= 1e-6:
                episode_summaries.append(
                    {
                        "demo_index": int(demo_index + int(args.demo_from_episode)),
                        "rows": 0,
                        "success": False,
                        "reason": "missing_object_pose",
                    }
                )
                continue
            close_gripper_pose = np.asarray(demo[close_index].gripper_pose, dtype=np.float32).reshape(7)
            pre_start = max(0, int(close_index) - int(args.close_pre_window))
            post_end = min(len(demo) - 1, int(close_index) + int(args.close_post_window))
            frames = []
            demo_row_start = len(rows)

            for frame_index in range(pre_start, post_end + 1):
                obs = demo[frame_index]
                if args.record_video:
                    frames.append(obs.front_rgb.copy())
                object_pose = _task_low_dim_pose_at_index(obs, pose_index)
                if not np.all(np.isfinite(object_pose[:7])) or np.linalg.norm(object_pose[3:7]) <= 1e-6:
                    continue
                delta_local = pose_delta_local_between(np.asarray(obs.gripper_pose, dtype=np.float32), object_pose)
                delta_local = delta_local.astype(np.float32)
                geom = _format_delta(delta_local)
                depth = np.asarray(obs.wrist_depth, dtype=np.float32)
                valid = depth[np.isfinite(depth)]
                depth_proximity = float(np.percentile(valid, 5.0)) if valid.size else float("nan")
                if np.asarray(getattr(obs, "gripper_touch_forces", np.zeros(6)), dtype=np.float32).size:
                    force_norm = float(np.linalg.norm(np.asarray(obs.gripper_touch_forces, dtype=np.float32)[:3]))
                else:
                    force_norm = 0.0
                phase = "preclose" if frame_index < close_index else "close" if frame_index == close_index else "postclose"
                attached_after_close = bool(frame_index >= close_index)
                verified_lift = bool(
                    demo_success
                    and frame_index > close_index
                    and np.isfinite(close_object_pose[2])
                    and np.isfinite(object_pose[2])
                    and (float(object_pose[2]) - float(close_object_pose[2]) >= float(args.verify_lift_threshold))
                )
                rows.append(
                    _as_row(
                        demo_index=int(demo_index + int(args.demo_from_episode)),
                        frame_index=int(frame_index),
                        close_index=int(close_index),
                        phase=phase,
                        demo_success=demo_success,
                        obs=obs,
                        object_pose_7d=object_pose,
                        close_object_pose_7d=close_object_pose,
                        close_gripper_pose_7d=close_gripper_pose,
                        depth_proximity=depth_proximity,
                        force_norm=force_norm,
                        attached_after_close=attached_after_close,
                        verified_lift=verified_lift,
                        object_in_finger_region=False,
                        close_xy_threshold=float(args.close_xy_threshold),
                        close_abs_z_threshold=float(args.close_abs_z_threshold),
                        close_yaw_threshold=float(args.close_yaw_threshold),
                        close_contact_depth_threshold=float(args.close_contact_depth_threshold),
                    )
                )
                if phase == "preclose":
                    preclose_deltas.append(delta_local)
                    preclose_depths.append(depth_proximity)
                    preclose_success_flags.append(True)
                elif phase == "close":
                    close_deltas.append(delta_local)
                    close_depths.append(depth_proximity)
                    close_success_flags.append(True)
                else:
                    postclose_deltas.append(delta_local)
                    postclose_success_flags.append(bool(verified_lift))

            if args.record_video and len(frames) > 1:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

                clip = ImageSequenceClip(frames, fps=int(args.video_fps))
                clip.write_videofile(
                    str(video_dir / f"demo_{demo_index:03d}.mp4"),
                    codec="libx264",
                    audio=False,
                    logger=None,
                )

            episode_summaries.append(
                {
                    "demo_index": int(demo_index + int(args.demo_from_episode)),
                    "rows": int(len(rows) - demo_row_start),
                    "success": True,
                    "close_index": int(close_index),
                    "preclose_rows": int(max(close_index - pre_start, 0)),
                    "close_rows": int(1),
                    "postclose_rows": int(max(post_end - close_index, 0)),
                }
            )
            print(
                f"demo {demo_index:03d}: close={close_index} rows={len(rows) - demo_row_start}",
                flush=True,
            )
    finally:
        env.shutdown()

    if not rows:
        raise RuntimeError("No successful demo rows collected.")

    preclose_xy = np.array([np.linalg.norm(r["object_to_gripper_delta_local_6d"][:2]) for r in rows if str(r["phase"].item()) == "preclose"], dtype=np.float32)
    preclose_abs_z = np.array([abs(float(r["object_to_gripper_delta_local_6d"][2])) for r in rows if str(r["phase"].item()) == "preclose"], dtype=np.float32)
    preclose_yaw = np.array([abs(float(r["object_to_gripper_delta_local_6d"][5])) for r in rows if str(r["phase"].item()) == "preclose"], dtype=np.float32)
    preclose_depth = np.array([float(r["depth_proximity"]) for r in rows if str(r["phase"].item()) == "preclose"], dtype=np.float32)
    close_xy = np.array([np.linalg.norm(r["object_to_gripper_delta_local_6d"][:2]) for r in rows if str(r["phase"].item()) == "close"], dtype=np.float32)
    close_abs_z = np.array([abs(float(r["object_to_gripper_delta_local_6d"][2])) for r in rows if str(r["phase"].item()) == "close"], dtype=np.float32)
    close_yaw = np.array([abs(float(r["object_to_gripper_delta_local_6d"][5])) for r in rows if str(r["phase"].item()) == "close"], dtype=np.float32)

    close_xy_threshold = float(max(np.percentile(preclose_xy, 90) if preclose_xy.size else args.close_xy_threshold, args.min_close_xy_threshold))
    close_abs_z_threshold = float(max(np.percentile(preclose_abs_z, 90) if preclose_abs_z.size else args.close_abs_z_threshold, args.min_close_abs_z_threshold))
    close_yaw_threshold = float(max(np.percentile(preclose_yaw, 90) if preclose_yaw.size else args.close_yaw_threshold, args.min_close_yaw_threshold))
    close_contact_depth_threshold = float(max(np.percentile(preclose_depth[np.isfinite(preclose_depth)], 90) if np.isfinite(preclose_depth).any() else args.close_contact_depth_threshold, args.min_close_contact_depth_threshold))

    # The demo-derived z basin is often too permissive if we use the raw p90
    # directly. Keep a conservative ceiling so the profile still reflects a
    # true low-close contract for runtime teacher gating.
    close_abs_z_threshold_raw = float(close_abs_z_threshold)
    close_abs_z_threshold = float(min(close_abs_z_threshold, 0.012))

    for row in rows:
        phase = str(row["phase"].item())
        delta = np.asarray(row["object_to_gripper_delta_local_6d"], dtype=np.float32).reshape(6)
        xy = float(np.linalg.norm(delta[:2]))
        abs_z = float(abs(delta[2]))
        yaw = float(abs(delta[5]))
        row["close_xy_threshold"] = np.asarray(float(close_xy_threshold), dtype=np.float32)
        row["close_abs_z_threshold"] = np.asarray(float(close_abs_z_threshold), dtype=np.float32)
        row["close_yaw_threshold"] = np.asarray(float(close_yaw_threshold), dtype=np.float32)
        row["close_contact_depth_threshold"] = np.asarray(float(close_contact_depth_threshold), dtype=np.float32)
        row["object_delta_xy"] = np.asarray(float(xy), dtype=np.float32)
        row["object_delta_abs_z"] = np.asarray(float(abs_z), dtype=np.float32)
        row["object_delta_yaw"] = np.asarray(float(yaw), dtype=np.float32)
        row["object_in_finger_region"] = np.asarray(
            float(
                phase == "preclose"
                and xy <= close_xy_threshold
                and abs_z <= close_abs_z_threshold
                and yaw <= close_yaw_threshold
            ),
            dtype=np.float32,
        )
        row["gripper_aperture_ready"] = np.asarray(float(phase == "preclose" and float(row["gripper_open"]) >= 0.5), dtype=np.float32)
        row["grasp_ready_score"] = np.asarray(
            float(
                np.clip(
                    0.45 * np.clip(1.0 - xy / max(close_xy_threshold, 1e-6), 0.0, 1.0)
                    + 0.30 * np.clip(1.0 - abs_z / max(close_abs_z_threshold, 1e-6), 0.0, 1.0)
                    + 0.15 * np.clip(1.0 - yaw / max(close_yaw_threshold, 1e-6), 0.0, 1.0)
                    + 0.10 * np.clip(1.0 - float(row["depth_proximity"]) / max(close_contact_depth_threshold, 1e-6), 0.0, 1.0),
                    0.0,
                    1.0,
                )
            ),
            dtype=np.float32,
        )

    profile = {
        "profile_source": "success_demo_profile",
        "task_name": str(args.task_name),
        "demo_count": int(len(episode_summaries)),
        "rows": int(len(rows)),
        "close_window_pre": int(args.close_pre_window),
        "close_window_post": int(args.close_post_window),
        "close_xy_threshold": float(close_xy_threshold),
        "close_abs_z_threshold_raw": float(close_abs_z_threshold_raw),
        "close_abs_z_threshold": float(close_abs_z_threshold),
        "close_yaw_threshold": float(close_yaw_threshold),
        "grasp_xy_threshold": float(close_xy_threshold),
        "grasp_abs_z_threshold_raw": float(close_abs_z_threshold_raw),
        "grasp_abs_z_threshold": float(close_abs_z_threshold),
        "grasp_yaw_threshold": float(close_yaw_threshold),
        "close_contact_depth_threshold": float(close_contact_depth_threshold),
        "grasp_ready_threshold": float(args.grasp_ready_threshold),
        "min_close_xy_threshold": float(args.min_close_xy_threshold),
        "min_close_abs_z_threshold": float(args.min_close_abs_z_threshold),
        "min_close_yaw_threshold": float(args.min_close_yaw_threshold),
        "min_close_contact_depth_threshold": float(args.min_close_contact_depth_threshold),
        "preclose_object_in_finger_region_rate": _rate(np.asarray([float(r["object_in_finger_region"]) for r in rows if str(r["phase"].item()) == "preclose"], dtype=np.float32) > 0.5),
        "attached_after_close_rate": _rate(np.asarray([float(r["attached_after_close"]) for r in rows if str(r["phase"].item()) in {"close", "postclose"}], dtype=np.float32) > 0.5),
        "verified_lift_rate": _rate(np.asarray([float(r["verified_lift"]) for r in rows if str(r["phase"].item()) == "postclose"], dtype=np.float32) > 0.5),
        "preclose_xy_stats": _stats(preclose_xy),
        "preclose_abs_z_stats": _stats(preclose_abs_z),
        "preclose_yaw_stats": _stats(preclose_yaw),
        "preclose_depth_proximity_stats": _stats(preclose_depth),
        "close_xy_stats": _stats(close_xy),
        "close_abs_z_stats": _stats(close_abs_z),
        "close_yaw_stats": _stats(close_yaw),
        "close_row_count": int(close_xy.size),
        "preclose_row_count": int(preclose_xy.size),
        "postclose_row_count": int(sum(1 for r in rows if str(r["phase"].item()) == "postclose")),
    }
    for row in rows:
        row["close_xy_threshold"] = np.asarray(float(close_xy_threshold), dtype=np.float32)
        row["close_abs_z_threshold"] = np.asarray(float(close_abs_z_threshold), dtype=np.float32)
        row["close_yaw_threshold"] = np.asarray(float(close_yaw_threshold), dtype=np.float32)
        row["close_contact_depth_threshold"] = np.asarray(float(close_contact_depth_threshold), dtype=np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _rows_to_npz(rows, Path(args.output_npz))
    Path(args.output_profile_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_profile_json).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    print(json.dumps(profile, indent=2, sort_keys=True))
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--demo_from_episode", type=int, default=0)
    parser.add_argument("--gripper_close_threshold", type=float, default=0.5)
    parser.add_argument("--close_pre_window", type=int, default=16)
    parser.add_argument("--close_post_window", type=int, default=16)
    parser.add_argument("--verify_lift_threshold", type=float, default=0.012)
    parser.add_argument("--grasp_ready_threshold", type=float, default=0.55)
    parser.add_argument("--close_xy_threshold", type=float, default=0.006)
    parser.add_argument("--close_abs_z_threshold", type=float, default=0.005)
    parser.add_argument("--close_yaw_threshold", type=float, default=0.12)
    parser.add_argument("--close_contact_depth_threshold", type=float, default=0.022)
    parser.add_argument("--min_close_xy_threshold", type=float, default=0.004)
    parser.add_argument("--min_close_abs_z_threshold", type=float, default=0.006)
    parser.add_argument("--min_close_yaw_threshold", type=float, default=0.04)
    parser.add_argument("--min_close_contact_depth_threshold", type=float, default=0.012)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--output_profile_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/alignment_tc_diffusion/grasp_basin_audit"))
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_fps", type=int, default=10)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
