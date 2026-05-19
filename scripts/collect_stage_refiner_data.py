"""
collect_stage_refiner_data.py

Collect generic stage-aware refiner supervision from planner-state or subgoal-reset rollouts.
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
_COPPELIASIM_ROOT = os.path.expanduser("~/CoppeliaSim")
_CONDA_LIBSTDCXX = os.path.expanduser("~/my_conda_envs/vla-adapter/lib/libstdc++.so.6")
os.environ.setdefault("COPPELIASIM_ROOT", _COPPELIASIM_ROOT)
_base_ld = f"{_COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
_existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
if not _existing_ld:
    os.environ["LD_LIBRARY_PATH"] = _base_ld
elif _COPPELIASIM_ROOT not in _existing_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{_base_ld}:{_existing_ld}"
os.environ.setdefault("LD_PRELOAD", _CONDA_LIBSTDCXX)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _COPPELIASIM_ROOT)
os.environ.setdefault("QT_PLUGIN_PATH", _COPPELIASIM_ROOT)
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
_HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/mnt/ssd/guoning/hf-cache")
os.environ.setdefault("HF_HOME", _HF_CACHE_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.exceptions import InvalidActionError

from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from prismatic.robot.stage_aware_refiner import StageAwareRefiner
from prismatic.robot.stage_manager import StageManager, StagePhase
from prismatic.vla.constants import FORCE_DIM, NUM_ACTIONS_CHUNK
from scripts.collect_residual_data import load_planner
from scripts.evaluate_rlbench import (
    TASK_MAP,
    _lazy_import_tasks,
    delta_to_absolute,
    predict_actions,
    process_obs,
    safe_recovery_absolute,
)

INTERACTION_ROLE_CODES = {
    "pre_grasp": 0,
    "pre_place": 1,
    "pre_insert": 2,
}


def make_env(task_name, dataset_root):
    _lazy_import_tasks()
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
    task = env.get_task(TASK_MAP[task_name])
    return env, task


def clip_delta(delta, clip_pos, clip_rot):
    out = delta.astype(np.float32).copy()
    pos_norm = np.linalg.norm(out[:3])
    if pos_norm > clip_pos:
        out[:3] = out[:3] * (clip_pos / max(pos_norm, 1e-8))
    rot_norm = np.linalg.norm(out[3:6])
    if rot_norm > clip_rot:
        out[3:6] = out[3:6] * (clip_rot / max(rot_norm, 1e-8))
    return out


def phase_to_contact_mask(phase_id: int) -> int:
    if phase_id <= int(StagePhase.TRANSIT):
        return 0
    if phase_id == int(StagePhase.ALIGN):
        return 1
    return 2


def stage_role_from_phase(phase_id: int) -> int:
    return 0 if phase_id in (int(StagePhase.TRANSIT), int(StagePhase.ALIGN)) else 1


def interaction_role_to_code(role_name: str) -> int:
    return int(INTERACTION_ROLE_CODES.get(role_name, 0))


def is_near_expert_gripper_close(npz_data, frame_idx: int, window: int) -> bool:
    """Return True for frames immediately before/around an expert close event."""
    if "gripper_open" not in npz_data:
        return False
    go = np.asarray(npz_data["gripper_open"]).reshape(-1)
    if go.size == 0:
        return False
    start = max(0, int(frame_idx) - 1)
    end = min(go.shape[0] - 1, int(frame_idx) + max(int(window), 0))
    if end <= start:
        return False
    segment_now = go[start : end + 1]
    segment_next = go[start + 1 : end + 1]
    segment_prev = go[start:end]
    close_state_soon = bool(np.any(segment_now < 0.5))
    close_transition_soon = bool(np.any((segment_prev >= 0.5) & (segment_next < 0.5))) if segment_next.size else False
    return close_state_soon or close_transition_soon


def find_expert_close_transitions(npz_data, open_threshold: float = 0.5) -> np.ndarray:
    """Return indices t where the expert gripper transitions open at t to closed at t+1."""
    if "gripper_open" not in npz_data:
        return np.asarray([], dtype=np.int64)
    go = np.asarray(npz_data["gripper_open"], dtype=np.float32).reshape(-1)
    if go.size < 2:
        return np.asarray([], dtype=np.int64)
    return np.where((go[:-1] >= open_threshold) & (go[1:] < open_threshold))[0].astype(np.int64)


def frames_to_next_expert_close(npz_data, frame_idx: int, open_threshold: float = 0.5) -> int:
    """Return steps until the next expert close transition, or -1 if none remains."""
    transitions = find_expert_close_transitions(npz_data, open_threshold=open_threshold)
    if transitions.size == 0:
        return -1
    future = transitions[transitions >= int(frame_idx)]
    if future.size == 0:
        return -1
    return int(future[0] - int(frame_idx))


def gripper_state_target_from_expert(
    npz_data,
    frame_idx: int,
    ready_window: int,
    open_threshold: float = 0.5,
) -> int:
    """Return 0=open, 1=close-now, 2=hold-closed from expert gripper state."""
    if "gripper_open" not in npz_data:
        return -1
    go = np.asarray(npz_data["gripper_open"], dtype=np.float32).reshape(-1)
    if go.size == 0:
        return -1
    t = min(max(int(frame_idx), 0), go.shape[0] - 1)
    if float(go[t]) < open_threshold:
        return 2
    frames = frames_to_next_expert_close(npz_data, t, open_threshold=open_threshold)
    if 0 <= frames <= int(ready_window):
        return 1
    return 0


def planner_close_intent_from_context(gripper_context: np.ndarray, threshold: float) -> tuple[bool, float]:
    min_raw = float(np.asarray(gripper_context, dtype=np.float32)[1])
    intent = bool(min_raw <= threshold)
    strength = float(np.clip((threshold - min_raw) / max(threshold, 1e-6), 0.0, 1.0))
    return intent, strength


def rotation_distance_rad(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    r_a = Rotation.from_quat(np.asarray(quat_a, dtype=np.float32))
    r_b = Rotation.from_quat(np.asarray(quat_b, dtype=np.float32))
    return float(np.linalg.norm((r_b * r_a.inv()).as_rotvec()))


def pose_delta_local_between(current_pose_7d: np.ndarray, target_pose_7d: np.ndarray) -> np.ndarray:
    current_pose_7d = np.asarray(current_pose_7d, dtype=np.float32)
    target_pose_7d = np.asarray(target_pose_7d, dtype=np.float32)
    delta_pos_world = target_pose_7d[:3] - current_pose_7d[:3]
    r_cur = Rotation.from_quat(current_pose_7d[3:7])
    r_tgt = Rotation.from_quat(target_pose_7d[3:7])
    delta_rot = (r_tgt * r_cur.inv()).as_rotvec().astype(np.float32)
    current_quat = current_pose_7d[3:7]
    delta_pos_local = world_delta_to_local(
        np.concatenate([delta_pos_world.astype(np.float32), np.zeros(3, dtype=np.float32)], axis=0),
        current_quat,
    )[:3]
    return np.concatenate([delta_pos_local.astype(np.float32), delta_rot.astype(np.float32)], axis=0)


def apply_local_offset_to_pose(pose_7d: np.ndarray, delta_local_6d: np.ndarray) -> np.ndarray:
    pose_7d = np.asarray(pose_7d, dtype=np.float32).copy()
    delta_local_6d = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    r_cur = Rotation.from_quat(pose_7d[3:7])
    pose_7d[:3] = pose_7d[:3] + r_cur.apply(delta_local_6d[:3]).astype(np.float32)
    r_delta = Rotation.from_rotvec(delta_local_6d[3:6].astype(np.float32))
    pose_7d[3:7] = (r_delta * r_cur).as_quat().astype(np.float32)
    return pose_7d


def parse_float_list_arg(raw: str) -> list[float]:
    if raw is None:
        return []
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def build_local_perturb_offsets(
    xy_values: list[float],
    z_values: list[float],
    yaw_values: list[float],
    include_diagonals: bool = True,
) -> list[np.ndarray]:
    offsets: list[np.ndarray] = []
    seen = set()

    def _add(offset):
        arr = np.asarray(offset, dtype=np.float32).reshape(6)
        key = tuple(np.round(arr, 6).tolist())
        if key in seen or np.linalg.norm(arr) < 1e-8:
            return
        seen.add(key)
        offsets.append(arr)

    for mag in xy_values:
        m = abs(float(mag))
        if m <= 0:
            continue
        _add([m, 0.0, 0.0, 0.0, 0.0, 0.0])
        _add([-m, 0.0, 0.0, 0.0, 0.0, 0.0])
        _add([0.0, m, 0.0, 0.0, 0.0, 0.0])
        _add([0.0, -m, 0.0, 0.0, 0.0, 0.0])
        if include_diagonals:
            _add([m, m, 0.0, 0.0, 0.0, 0.0])
            _add([m, -m, 0.0, 0.0, 0.0, 0.0])
            _add([-m, m, 0.0, 0.0, 0.0, 0.0])
            _add([-m, -m, 0.0, 0.0, 0.0, 0.0])

    for mag in z_values:
        m = abs(float(mag))
        if m <= 0:
            continue
        _add([0.0, 0.0, m, 0.0, 0.0, 0.0])
        _add([0.0, 0.0, -m, 0.0, 0.0, 0.0])

    for mag in yaw_values:
        m = abs(float(mag))
        if m <= 0:
            continue
        _add([0.0, 0.0, 0.0, 0.0, 0.0, m])
        _add([0.0, 0.0, 0.0, 0.0, 0.0, -m])

    return offsets


def find_preclose_anchor_index(npz_data, frame_idx: int, open_threshold: float = 0.5) -> tuple[int, int]:
    transitions = find_expert_close_transitions(npz_data, open_threshold=open_threshold)
    if transitions.size == 0:
        return -1, -1
    future = transitions[transitions >= int(frame_idx)]
    if future.size == 0:
        return -1, -1
    close_idx = int(future[0])
    anchor_idx = max(int(frame_idx), close_idx)
    go = np.asarray(npz_data["gripper_open"], dtype=np.float32).reshape(-1)
    while anchor_idx > int(frame_idx) and anchor_idx < go.shape[0] and float(go[anchor_idx]) < open_threshold:
        anchor_idx -= 1
    anchor_idx = max(int(frame_idx), anchor_idx)
    return anchor_idx, close_idx


def build_reference_preclose_segment(npz_data, start_idx: int, open_threshold: float = 0.5) -> tuple[np.ndarray, int, int]:
    go = np.asarray(npz_data["gripper_open"], dtype=np.float32).reshape(-1)
    transitions = find_expert_close_transitions(npz_data, open_threshold=open_threshold)
    future = transitions[transitions >= int(start_idx)]
    if future.size == 0:
        return np.asarray([], dtype=np.int64), -1, -1
    close_idx = int(future[0])
    candidate = np.arange(max(int(start_idx), 0), close_idx + 1, dtype=np.int64)
    candidate = candidate[go[candidate] >= open_threshold]
    anchor_idx = int(candidate[-1]) if candidate.size > 0 else close_idx
    return candidate, close_idx, anchor_idx


def compute_basin_center_pose(
    npz_data,
    candidate_indices: np.ndarray,
    center_k: int = 3,
    mode: str = "success_region_proxy",
    close_idx: int | None = None,
    success_region_window: int = 8,
    success_region_exclude_last: int = 1,
) -> np.ndarray | None:
    if candidate_indices.size == 0:
        return None
    gp = np.asarray(npz_data["gripper_pose"], dtype=np.float32)
    if mode == "lastk_mean":
        sel = candidate_indices[-max(int(center_k), 1) :]
        poses = gp[sel]
        center_pos = poses[:, :3].mean(axis=0).astype(np.float32)
        center_rot = Rotation.from_quat(poses[:, 3:7]).mean().as_quat().astype(np.float32)
        return np.concatenate([center_pos, center_rot], axis=0).astype(np.float32)

    if close_idx is None:
        close_idx = int(candidate_indices[-1])
    open_region = candidate_indices[(close_idx - candidate_indices >= int(success_region_exclude_last))]
    open_region = open_region[(close_idx - open_region) <= int(success_region_window)]
    if open_region.size < max(3, int(center_k)):
        fallback = candidate_indices[:-int(success_region_exclude_last)] if int(success_region_exclude_last) > 0 else candidate_indices
        if fallback.size == 0:
            fallback = candidate_indices
        open_region = fallback[-max(int(success_region_window), int(center_k), 3) :]

    poses = gp[open_region]
    pos = poses[:, :3]
    rots = Rotation.from_quat(poses[:, 3:7])
    n = poses.shape[0]
    pair_cost = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dpos = pos[i] - pos[j]
            drot = (rots[j] * rots[i].inv()).as_rotvec().astype(np.float32)
            cost = (
                np.linalg.norm(dpos[:2]) / 0.008
                + abs(float(dpos[2])) / 0.01
                + abs(float(drot[2])) / 0.05
            )
            pair_cost[i, j] = cost
            pair_cost[j, i] = cost
    medoid_idx = int(np.argmin(pair_cost.sum(axis=1)))
    return poses[medoid_idx].astype(np.float32)


def match_reference_index(
    npz_data,
    current_pose_7d: np.ndarray,
    candidate_indices: np.ndarray,
    pos_weight: float = 1.0,
    rot_weight: float = 0.05,
) -> int:
    if candidate_indices.size == 0:
        return -1
    gp = np.asarray(npz_data["gripper_pose"], dtype=np.float32)
    cur = np.asarray(current_pose_7d, dtype=np.float32)
    best_idx = -1
    best_score = float("inf")
    for idx in candidate_indices:
        ref = gp[int(idx)]
        pos_err = float(np.linalg.norm(ref[:3] - cur[:3]))
        rot_err = rotation_distance_rad(cur[3:7], ref[3:7])
        score = pos_weight * pos_err + rot_weight * rot_err
        if score < best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def compute_basin_metrics(
    delta_basin_target: np.ndarray,
    r_xy: float,
    r_z: float,
    r_yaw: float,
) -> tuple[float, float, float, float]:
    delta_arr = np.asarray(delta_basin_target, dtype=np.float32).reshape(-1)
    e_xy = float(np.linalg.norm(delta_arr[:2])) if delta_arr.size >= 2 else 0.0
    e_z = float(abs(delta_arr[2])) if delta_arr.size >= 3 else 0.0
    e_yaw = float(abs(delta_arr[5])) if delta_arr.size >= 6 else 0.0
    basin_distance = max(
        e_xy / max(float(r_xy), 1e-6),
        e_z / max(float(r_z), 1e-6),
        e_yaw / max(float(r_yaw), 1e-6),
    )
    return basin_distance, e_xy, e_z, e_yaw


def compute_rollout_trigger_outcomes(
    rollout_history,
    start_idx: int,
    hold_horizon: int,
    open_threshold: float,
    lift_threshold: float,
) -> tuple[float, float, float, float, float]:
    if start_idx < 0 or start_idx >= len(rollout_history):
        return 0.0, 0.0, 0.0, 0.0, 1.0
    end_idx = min(len(rollout_history), start_idx + max(int(hold_horizon), 1))
    segment = rollout_history[start_idx:end_idx]
    invalid_after_trigger = float(any(step.get("invalid", False) for step in segment))
    next_open = np.asarray([float(step.get("next_gripper_open", 1.0)) for step in segment], dtype=np.float32)
    next_z = np.asarray([float(step.get("next_z", 0.0)) for step in segment], dtype=np.float32)
    closed_indices = np.where(next_open < float(open_threshold))[0]
    reopen_after_trigger = 0.0
    grasp_lift_proxy = 0.0
    if closed_indices.size > 0:
        first_closed = int(closed_indices[0])
        later_open = next_open[first_closed + 1 :] >= float(open_threshold)
        reopen_after_trigger = float(np.any(later_open)) if later_open.size > 0 else 0.0
        base_z = float(next_z[first_closed])
        future_z = next_z[first_closed:]
        grasp_lift_proxy = float(np.max(future_z - base_z) > float(lift_threshold)) if future_z.size > 0 else 0.0
    post_trigger_stability = float(
        (invalid_after_trigger < 0.5)
        and (reopen_after_trigger < 0.5)
        and (grasp_lift_proxy > 0.5)
    )
    no_progress_after_trigger = float(
        (invalid_after_trigger < 0.5)
        and (reopen_after_trigger < 0.5)
        and (grasp_lift_proxy < 0.5)
    )
    return post_trigger_stability, grasp_lift_proxy, reopen_after_trigger, no_progress_after_trigger, invalid_after_trigger


def classify_negative_reason(
    basin_positive: float,
    basin_distance: float,
    post_trigger_stability_proxy: float,
    reopen_after_trigger: float,
    no_progress_after_trigger: float,
    invalid_after_trigger: float,
) -> int:
    if basin_positive > 0.5:
        return -1
    if invalid_after_trigger > 0.5:
        return 4
    if reopen_after_trigger > 0.5:
        return 2
    if no_progress_after_trigger > 0.5:
        return 3
    if basin_distance <= 1.0 and post_trigger_stability_proxy <= 0.5:
        return 1
    return 0


def make_gripper_context(base_action, future_actions=None) -> np.ndarray:
    """Small planner-intent feature: current, min-lookahead, mean-lookahead gripper raw."""
    values = [float(np.asarray(base_action, dtype=np.float32)[6])]
    if future_actions is not None:
        values.extend(float(np.asarray(a, dtype=np.float32)[6]) for a in list(future_actions))
    arr = np.asarray(values, dtype=np.float32)
    return np.asarray([arr[0], float(np.min(arr)), float(np.mean(arr))], dtype=np.float32)


def choose_pregrasp_reset_index(npz_data, window: int, open_threshold: float = 0.5) -> int:
    """Reset before the first expert close transition to target near-grasp alignment."""
    transitions = find_expert_close_transitions(npz_data, open_threshold=open_threshold)
    if transitions.size == 0:
        return 0
    return max(0, int(transitions[0]) - max(int(window), 0))


def summarize_int_values(values):
    if len(values) == 0:
        return {}
    arr = np.asarray(values, dtype=np.int64)
    return {int(v): int((arr == v).sum()) for v in np.unique(arr)}


def summarize_float_stats(values):
    if len(values) == 0:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def flush_shard(output_dir, shard_idx, buffers):
    shard_path = output_dir / f"residual_shard_{shard_idx:04d}.npz"
    phase_summary = summarize_int_values(buffers["phase_id"])
    stage_summary = summarize_int_values(buffers["stage_role"])
    failure_summary = summarize_int_values(buffers["failure_mode"])
    basin_summary = summarize_int_values(buffers["readiness_label"])
    hold_summary = summarize_int_values(buffers["hold_label"])
    gripper_summary = summarize_int_values(buffers["gripper_state_target"])
    planner_close_summary = summarize_int_values(buffers["planner_close_intent"])
    negative_summary = summarize_int_values([v for v in buffers["negative_reason"] if int(v) >= 0])
    frames = np.asarray(buffers["frames_to_reference_trigger"], dtype=np.float32)
    valid_frames = frames[frames >= 0]
    deltas = np.asarray(buffers["delta_align_target"], dtype=np.float32)
    delta_abs = np.abs(deltas) if deltas.size else np.zeros((0, 6), dtype=np.float32)
    basin_dist = np.asarray(buffers["basin_distance"], dtype=np.float32)
    frame_summary = {
        "min": int(valid_frames.min()) if valid_frames.size else -1,
        "p50": float(np.percentile(valid_frames, 50)) if valid_frames.size else -1.0,
        "max": int(valid_frames.max()) if valid_frames.size else -1,
    }
    delta_summary = {
        "xy_p50": float(np.percentile(np.linalg.norm(deltas[:, :2], axis=-1), 50)) if deltas.size else 0.0,
        "xy_p95": float(np.percentile(np.linalg.norm(deltas[:, :2], axis=-1), 95)) if deltas.size else 0.0,
        "yaw_p50": float(np.percentile(delta_abs[:, 5], 50)) if delta_abs.size else 0.0,
        "yaw_p95": float(np.percentile(delta_abs[:, 5], 95)) if delta_abs.size else 0.0,
    }
    np.savez_compressed(
        shard_path,
        wrist_depth=np.asarray(buffers["wrist_depth"], dtype=np.float32),
        ft_hist=np.asarray(buffers["ft_hist"], dtype=np.float32),
        proprio=np.asarray(buffers["proprio"], dtype=np.float32),
        current_pose_7d=np.asarray(buffers["current_pose_7d"], dtype=np.float32),
        basin_center_pose_7d=np.asarray(buffers["basin_center_pose_7d"], dtype=np.float32),
        reference_anchor_pose_7d=np.asarray(buffers["reference_anchor_pose_7d"], dtype=np.float32),
        base_action=np.asarray(buffers["base_action"], dtype=np.float32),
        gripper_context=np.asarray(buffers["gripper_context"], dtype=np.float32),
        interaction_role=np.asarray(buffers["interaction_role"], dtype=np.int64),
        step_idx=np.asarray(buffers["step_idx"], dtype=np.int64),
        delta_target=np.asarray(buffers["delta_target"], dtype=np.float32),
        delta_align_target=np.asarray(buffers["delta_align_target"], dtype=np.float32),
        delta_basin_target=np.asarray(buffers["delta_align_target"], dtype=np.float32),
        contact_mask=np.asarray(buffers["contact_mask"], dtype=np.int64),
        phase_label=np.asarray(buffers["phase_label"], dtype=np.int64),
        phase_id=np.asarray(buffers["phase_id"], dtype=np.int64),
        phase_age=np.asarray(buffers["phase_age"], dtype=np.float32),
        steps_since_last_replan=np.asarray(buffers["steps_since_last_replan"], dtype=np.float32),
        stage_role=np.asarray(buffers["stage_role"], dtype=np.int64),
        failure_mode=np.asarray(buffers["failure_mode"], dtype=np.int64),
        transition_flag=np.asarray(buffers["transition_flag"], dtype=np.int64),
        subgoal_progress=np.asarray(buffers["subgoal_progress"], dtype=np.float32),
        frames_to_expert_close=np.asarray(buffers["frames_to_reference_trigger"], dtype=np.int64),
        frames_to_reference_trigger=np.asarray(buffers["frames_to_reference_trigger"], dtype=np.int64),
        rollout_gripper_open=np.asarray(buffers["rollout_gripper_open"], dtype=np.float32),
        depth_proximity=np.asarray(buffers["depth_proximity"], dtype=np.float32),
        planner_close_intent=np.asarray(buffers["planner_close_intent"], dtype=np.float32),
        planner_close_intent_strength=np.asarray(buffers["planner_close_intent_strength"], dtype=np.float32),
        geometry_conditioned_pose_support=np.asarray(buffers["geometry_conditioned_pose_support"], dtype=np.int64),
        planner_conditioned_support=np.asarray(buffers["planner_conditioned_support"], dtype=np.int64),
        background_align_support=np.asarray(buffers["background_align_support"], dtype=np.int64),
        readiness_label=np.asarray(buffers["readiness_label"], dtype=np.float32),
        basin_positive=np.asarray(buffers["readiness_label"], dtype=np.float32),
        hold_label=np.asarray(buffers["hold_label"], dtype=np.float32),
        basin_distance=np.asarray(buffers["basin_distance"], dtype=np.float32),
        negative_reason=np.asarray(buffers["negative_reason"], dtype=np.int64),
        post_close_stability_proxy=np.asarray(buffers["post_close_stability_proxy"], dtype=np.float32),
        grasp_lift_proxy=np.asarray(buffers["grasp_lift_proxy"], dtype=np.float32),
        reopen_within_horizon=np.asarray(buffers["reopen_within_horizon"], dtype=np.float32),
        reopen_after_trigger=np.asarray(buffers["reopen_within_horizon"], dtype=np.float32),
        no_progress_after_trigger=np.asarray(buffers["no_progress_after_trigger"], dtype=np.float32),
        invalid_after_trigger=np.asarray(buffers["invalid_after_trigger"], dtype=np.float32),
        ready_to_close=np.asarray(buffers["readiness_label"], dtype=np.float32),
        gripper_state_target=np.asarray(buffers["gripper_state_target"], dtype=np.int64),
        planner_close_too_early=np.asarray(buffers["planner_close_too_early"], dtype=np.float32),
        expert_hold_after_close=np.asarray(buffers["expert_hold_after_close"], dtype=np.float32),
        is_augmented=np.asarray(buffers["is_augmented"], dtype=np.int64),
        augment_offset_local=np.asarray(buffers["augment_offset_local"], dtype=np.float32),
    )
    print(f"  [stage shard {shard_idx}] saved {len(buffers['delta_target'])} samples -> {shard_path}")
    print(f"    phase_counts={phase_summary} stage_role_counts={stage_summary} failure_counts={failure_summary}")
    print(
        f"    planner_close_intent={planner_close_summary} basin_positive={basin_summary} hold={hold_summary} "
        f"gripper_state_counts={gripper_summary} negative_reason={negative_summary}"
    )
    print(
        f"    frames_to_reference_trigger={frame_summary} delta_stats={delta_summary} "
        f"basin_distance={summarize_float_stats(basin_dist.tolist())}"
    )
    total = max(len(buffers["phase_id"]), 1)
    dominant = max(phase_summary.values(), default=0) / total
    if dominant > 0.9:
        print(f"    WARNING: shard {shard_idx} is phase-skewed (dominant_fraction={dominant:.3f})")
    negative_total = max(sum(negative_summary.values()), 1)
    dominant_negative = max(negative_summary.values(), default=0) / negative_total
    if dominant_negative > 0.85:
        print(
            f"    WARNING: shard {shard_idx} has unhealthy negative_reason dominance "
            f"(dominant_fraction={dominant_negative:.3f})"
        )


def reconstruct_expert_action_7d(npz_data, frame_idx: int) -> np.ndarray:
    """Return a 7D delta action regardless of whether stored action_targets are 7D or legacy 10D."""
    at = npz_data["action_targets"]
    if at.shape[1] == 7:
        return at[frame_idx].astype(np.float32)

    gp = npz_data["gripper_pose"]
    go = npz_data["gripper_open"]
    t = min(frame_idx, at.shape[0] - 1)
    if t < at.shape[0] - 1:
        delta_pos = gp[t + 1, :3] - gp[t, :3]
        r0 = Rotation.from_quat(gp[t, 3:7])
        r1 = Rotation.from_quat(gp[t + 1, 3:7])
        delta_rv = (r1 * r0.inv()).as_rotvec()
        gripper = go[t + 1, 0]
    else:
        delta_pos = np.zeros(3, dtype=np.float32)
        delta_rv = np.zeros(3, dtype=np.float32)
        gripper = go[t, 0]
    return np.concatenate([delta_pos, delta_rv, [gripper]]).astype(np.float32)


def compute_demo_stage_indices(ep_path, npz_data):
    manager = StageManager()
    stage_indices = {phase: None for phase in StagePhase}
    T = npz_data["action_targets"].shape[0]
    force_arr = npz_data.get("gripper_touch_forces", np.zeros((T, FORCE_DIM), dtype=np.float32))
    proprio_arr = npz_data["proprio"]
    for t in range(T):
        depth_path = ep_path / "wrist_depth" / f"{t}.png"
        depth_prox = None
        if depth_path.exists():
            from PIL import Image

            depth = np.array(Image.open(depth_path), dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            depth = np.clip(depth / 255.0, 0.0, 1.0)
            valid = depth[np.isfinite(depth)]
            if valid.size > 0:
                depth_prox = float(np.percentile(valid, 5.0))
        manager.update(
            force_reading=force_arr[t],
            gripper_pose=proprio_arr[t][7:14] if proprio_arr.shape[1] >= 14 else None,
            gripper_open=float(proprio_arr[t][-1]) if proprio_arr.shape[1] >= 1 else None,
            depth_proximity=depth_prox,
            base_action=reconstruct_expert_action_7d(npz_data, t),
        )
        if stage_indices[manager.phase] is None:
            stage_indices[manager.phase] = t
    return stage_indices


def maybe_reset_to_subgoal(task, demo_obs, ep_path, npz_data, target_stage: StagePhase, mode: str):
    descs, obs = task.reset_to_demo(demo_obs)
    if mode == "none":
        return descs, obs, 0

    stage_indices = compute_demo_stage_indices(ep_path, npz_data)
    target_idx = stage_indices.get(target_stage, None)
    T = npz_data["action_targets"].shape[0]
    if mode == "oracle_subgoal":
        if target_idx is None:
            progress_ratio = 0.35 if target_stage == StagePhase.ALIGN else 0.7
            target_idx = int(progress_ratio * max(T - 1, 1))
        else:
            target_idx = max(0, int(target_idx) - (2 if target_stage == StagePhase.ALIGN else 1))
    if target_idx is None:
        return descs, obs, 0

    replay_until = max(0, min(int(target_idx), T - 1))
    for i in range(replay_until):
        expert_action = reconstruct_expert_action_7d(npz_data, i)
        abs_action = delta_to_absolute(expert_action, obs.gripper_pose)
        obs, _, terminate = task.step(abs_action)
        if terminate:
            break
    return descs, obs, replay_until


def build_refiner(args):
    if args.rollout_mode == "planner_only":
        return None
    from scripts.evaluate_rlbench import _load_optional_controller

    alignment_controller = _load_optional_controller(args.alignment_ckpt) if args.alignment_ckpt else None
    contact_controller = _load_optional_controller(args.contact_ckpt) if args.contact_ckpt else None
    return StageAwareRefiner(
        mode=args.rollout_mode,
        alignment_controller=alignment_controller,
        contact_controller=contact_controller,
        max_residual_pos=args.max_residual_pos,
        max_residual_rot=args.max_residual_rot,
        learned_residual_scale=args.learned_residual_scale,
        require_close_intent_for_alignment=args.require_close_intent_for_alignment,
        max_alignment_corrections_per_window=args.max_alignment_corrections_per_window,
        alignment_depth_threshold=args.alignment_depth_threshold,
    )


def main():
    parser = argparse.ArgumentParser(description="Collect stage-aware refiner data")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, default="data/rlbench_data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vlm_path", type=str, default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", type=str, default="pretrained_models/configs/config.json")
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--no_depth", dest="use_depth", action="store_false")
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--no_force", dest="use_force", action="store_false")
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.set_defaults(planner_use_depth=None, planner_use_force=None)
    parser.add_argument("--rollout_mode", type=str, default="planner_only", choices=["planner_only", "safety_only", "alignment", "contact", "full"])
    parser.add_argument("--alignment_ckpt", type=str, default=None)
    parser.add_argument("--contact_ckpt", type=str, default=None)
    parser.add_argument("--subgoal_reset_mode", type=str, default="none", choices=["none", "demo_subgoal", "oracle_subgoal"])
    parser.add_argument("--subgoal_stage", type=str, default="ALIGN", choices=["ALIGN", "INTERACT"])
    parser.add_argument("--max_residual_pos", type=float, default=0.01)
    parser.add_argument("--max_residual_rot", type=float, default=0.03)
    parser.add_argument("--learned_residual_scale", type=float, default=1.0)
    parser.add_argument("--require_close_intent_for_alignment", action="store_true", default=False)
    parser.add_argument("--no_require_close_intent_for_alignment", dest="require_close_intent_for_alignment", action="store_false")
    parser.add_argument("--max_alignment_corrections_per_window", type=int, default=20)
    parser.add_argument("--alignment_depth_threshold", type=float, default=0.08)
    parser.add_argument("--delta_clip_pos", type=float, default=0.01)
    parser.add_argument("--delta_clip_rot", type=float, default=0.05)
    parser.add_argument("--shard_size", type=int, default=2000)
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--episode_offset", type=int, default=0)
    parser.add_argument("--max_rollout_steps", type=int, default=-1)
    parser.add_argument("--align_focus", action="store_true", default=False, help="Only save ALIGN / near expert gripper-close samples for alignment-refiner training.")
    parser.add_argument("--align_window", type=int, default=8, help="Frames before/around expert gripper close to keep when --align_focus is enabled.")
    parser.add_argument("--pregrasp_focus", action="store_true", default=False, help="Only save near-target, still-open samples shortly before expert gripper close.")
    parser.add_argument("--pregrasp_window", type=int, default=16, help="Number of expert frames before close to treat as pre-grasp alignment supervision.")
    parser.add_argument("--pregrasp_reset", action="store_true", default=False, help="Replay each demo to just before its first expert close before starting rollout.")
    parser.add_argument("--near_depth_threshold", type=float, default=0.18, help="Depth-proximity threshold for near-target pre-grasp samples.")
    parser.add_argument("--open_gripper_threshold", type=float, default=0.5, help="Rollout/expert gripper open threshold.")
    parser.add_argument("--ready_close_window", type=int, default=2, help="Legacy alias kept for compatibility; basin trigger no longer uses expert-close timeline directly.")
    parser.add_argument("--planner_close_threshold", type=float, default=0.5, help="Planner raw gripper <= this is treated as close intent.")
    parser.add_argument("--planner_close_lookahead", type=int, default=4, help="Number of future planner gripper commands used for close-intent support.")
    parser.add_argument("--hold_horizon", type=int, default=16, help="Future horizon used to compute post-close stability proxies.")
    parser.add_argument("--lift_threshold", type=float, default=0.02, help="EE z-lift used by the generic grasp_lift_proxy.")
    parser.add_argument("--support_align_only", action="store_true", default=True, help="Only keep ALIGN states for planner-conditioned readiness/alignment training.")
    parser.add_argument("--allow_non_align_support", dest="support_align_only", action="store_false")
    parser.add_argument("--non_intent_keep_window", type=int, default=12, help="Keep some ALIGN+open+no-close-intent samples when they are still close to the matched reference close event.")
    parser.add_argument("--match_rot_weight", type=float, default=0.05, help="Rotation weight when matching current rollout pose to the reference preclose segment.")
    parser.add_argument("--ready_align_xy_threshold", type=float, default=0.005, help="Legacy compatibility alias for basin xy radius.")
    parser.add_argument("--ready_align_yaw_threshold", type=float, default=0.04, help="Legacy compatibility alias for basin yaw radius.")
    parser.add_argument("--interaction_role", type=str, default="pre_grasp", choices=["pre_grasp", "pre_place", "pre_insert"])
    parser.add_argument("--dataset_view", type=str, default="combined", choices=["combined", "basin_trigger_view", "basin_pose_view"])
    parser.add_argument("--basin_center_k", type=int, default=3)
    parser.add_argument("--basin_center_mode", type=str, default="success_region_proxy", choices=["lastk_mean", "success_region_proxy"])
    parser.add_argument("--success_region_window", type=int, default=8)
    parser.add_argument("--success_region_exclude_last", type=int, default=1)
    parser.add_argument("--basin_radius_xy", type=float, default=0.005)
    parser.add_argument("--basin_radius_z", type=float, default=0.01)
    parser.add_argument("--basin_radius_yaw", type=float, default=0.04)
    parser.add_argument(
        "--pose_support_basin_distance_threshold",
        type=float,
        default=3.5,
        help="Legacy compatibility knob. basin_pose_view support is now gated by decoupled z/xy(/yaw) thresholds instead of this scalar basin distance.",
    )
    parser.add_argument(
        "--pose_support_xy_threshold",
        type=float,
        default=0.02,
        help="Geometry-conditioned support threshold for basin_pose_view; only keep samples whose local xy error is within this range.",
    )
    parser.add_argument(
        "--pose_support_abs_z_threshold",
        type=float,
        default=0.02,
        help="Geometry-conditioned support threshold for basin_pose_view; only keep samples whose local z error is within this range.",
    )
    parser.add_argument(
        "--pose_support_yaw_threshold",
        type=float,
        default=-1.0,
        help="Optional geometry-conditioned yaw threshold for basin_pose_view. Negative disables the yaw check.",
    )
    parser.add_argument(
        "--pose_support_depth_threshold",
        type=float,
        default=-1.0,
        help="Optional depth proximity threshold for basin_pose_view. Negative disables the depth check.",
    )
    parser.add_argument("--enable_local_perturb_aug", action="store_true", default=False)
    parser.add_argument("--augment_xy_values", type=str, default="0.005,0.01")
    parser.add_argument("--augment_z_values", type=str, default="")
    parser.add_argument("--augment_yaw_values", type=str, default="0.03,0.05")
    parser.add_argument("--augment_include_diagonals", action="store_true", default=True)
    parser.add_argument("--no_augment_include_diagonals", dest="augment_include_diagonals", action="store_false")
    parser.add_argument("--augment_pose_origin", type=str, default="current_pose", choices=["current_pose", "basin_center"])
    args = parser.parse_args()

    if args.planner_use_depth is None:
        args.planner_use_depth = False
    if args.planner_use_force is None:
        args.planner_use_force = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        args.checkpoint_dir,
        args.vlm_path,
        args.config_path,
        args.planner_use_depth,
        args.planner_use_force,
    )
    refiner = build_refiner(args)

    episodes_dir = Path(args.data_root) / args.task_name / "train" / "episodes"
    ep_dirs = sorted([d for d in os.listdir(episodes_dir) if d.startswith("episode")], key=lambda x: int(x.replace("episode", "")))
    if args.episode_offset > 0:
        ep_dirs = ep_dirs[int(args.episode_offset):]
    if args.max_episodes > 0:
        ep_dirs = ep_dirs[:args.max_episodes]

    env, task = make_env(args.task_name, Path(args.data_root))
    target_stage = StagePhase[args.subgoal_stage]

    buffers = {k: [] for k in [
        "wrist_depth", "ft_hist", "proprio", "current_pose_7d", "basin_center_pose_7d", "reference_anchor_pose_7d",
        "base_action", "step_idx", "delta_target",
        "delta_align_target",
        "contact_mask", "phase_label", "phase_id", "phase_age", "steps_since_last_replan", "stage_role",
        "failure_mode", "transition_flag", "subgoal_progress",
        "frames_to_reference_trigger", "rollout_gripper_open", "depth_proximity",
        "gripper_context", "planner_close_intent", "planner_close_intent_strength",
        "geometry_conditioned_pose_support", "planner_conditioned_support", "background_align_support",
        "interaction_role", "readiness_label", "hold_label", "basin_distance", "negative_reason",
        "post_close_stability_proxy", "grasp_lift_proxy", "reopen_within_horizon", "no_progress_after_trigger", "invalid_after_trigger",
        "ready_to_close", "gripper_state_target",
        "planner_close_too_early", "expert_hold_after_close", "is_augmented", "augment_offset_local",
    ]}
    shard_idx = 0
    total_samples = 0
    seen_steps = 0
    kept_align_focus = 0
    kept_close_window = 0
    basin_positive_counts = {0: 0, 1: 0}
    hold_counts = {0: 0, 1: 0}
    gripper_state_counts = {-1: 0, 0: 0, 1: 0, 2: 0}
    planner_close_intent_counts = {0: 0, 1: 0}
    negative_reason_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    phase_counts = {int(phase): 0 for phase in StagePhase}
    seen_phase_counts = {int(phase): 0 for phase in StagePhase}
    failure_counts = {}
    post_close_values = []
    reopen_values = []
    lift_values = []
    delta_xy_values = []
    delta_yaw_values = []
    frames_to_trigger_values = []
    basin_distance_values = []
    base_support_sample_count = 0
    augmented_sample_count = 0
    support_source_counts = {"geometry": 0, "planner_close": 0, "background": 0}
    geometry_support_close_intent_counts = {0: 0, 1: 0}
    kept_planner_close_too_early_count = 0
    geometry_planner_close_too_early_count = 0
    perturb_offsets = build_local_perturb_offsets(
        parse_float_list_arg(args.augment_xy_values),
        parse_float_list_arg(args.augment_z_values),
        parse_float_list_arg(args.augment_yaw_values),
        include_diagonals=bool(args.augment_include_diagonals),
    )

    try:
        for ep_name in ep_dirs:
            ep_path = episodes_dir / ep_name
            npz_path = ep_path / "model_inputs.npz"
            if not npz_path.exists():
                continue
            npz_data = dict(np.load(npz_path))
            T = npz_data["action_targets"].shape[0]
            low_dim_obs_path = ep_path / "low_dim_obs.pkl"
            if not low_dim_obs_path.exists():
                continue
            with open(low_dim_obs_path, "rb") as f:
                demo_obs = pickle.load(f)
            descs, obs, rollout_offset = maybe_reset_to_subgoal(
                task, demo_obs, ep_path, npz_data, target_stage, args.subgoal_reset_mode
            )
            if args.pregrasp_reset:
                descs, obs = task.reset_to_demo(demo_obs)
                rollout_offset = choose_pregrasp_reset_index(
                    npz_data,
                    window=args.pregrasp_window,
                    open_threshold=args.open_gripper_threshold,
                )
                for i in range(rollout_offset):
                    expert_action = reconstruct_expert_action_7d(npz_data, i)
                    abs_action = delta_to_absolute(expert_action, obs.gripper_pose)
                    obs, _, terminate = task.step(abs_action)
                    if terminate:
                        break
            instruction = descs[0] if isinstance(descs, list) else str(descs)
            force_buffer = deque(maxlen=256)
            action_queue = []
            chunk_step = 0
            if refiner is not None:
                refiner.reset()
            rollout_manager = StageManager()
            max_steps = T if args.max_rollout_steps <= 0 else min(T, args.max_rollout_steps)
            reference_candidates, reference_close_idx, reference_anchor_idx = build_reference_preclose_segment(
                npz_data,
                start_idx=rollout_offset,
                open_threshold=args.open_gripper_threshold,
            )
            basin_center_pose = compute_basin_center_pose(
                npz_data,
                reference_candidates,
                center_k=args.basin_center_k,
                mode=args.basin_center_mode,
                close_idx=reference_close_idx,
                success_region_window=args.success_region_window,
                success_region_exclude_last=args.success_region_exclude_last,
            )
            interaction_role_code = interaction_role_to_code(args.interaction_role)
            episode_history = []
            episode_samples = []

            for rollout_step in range(rollout_offset, max_steps):
                front_pil, wrist_pil, proprio, depth_tensor_224, force_hist, depth_tensor_96, raw_force = process_obs(
                    obs,
                    norm_stats,
                    force_buffer,
                    use_depth=args.use_depth,
                    use_force=args.use_force,
                    depth_max=1.0,
                )

                if len(action_queue) == 0:
                    actions = predict_actions(
                        vla, processor, action_head, proprio_projector,
                        front_pil, wrist_pil, proprio,
                        depth_tensor_224 if args.planner_use_depth else None,
                        force_hist if args.planner_use_force else None,
                        instruction,
                        unnorm_key="rlbench",
                    )
                    max_chunk = refiner.get_chunk_size() if refiner is not None else len(actions)
                    action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), max_chunk))]
                    chunk_step = 0

                base_action = action_queue.pop(0)
                future_gripper_actions = action_queue[: args.planner_close_lookahead]
                expert_action = reconstruct_expert_action_7d(npz_data, rollout_step)
                current_quat = obs.gripper_pose[3:7]
                base_action_local = world_delta_to_local(base_action[:6], current_quat)
                delta_target = clip_delta(
                    world_delta_to_local(expert_action[:6] - base_action[:6], current_quat),
                    args.delta_clip_pos,
                    args.delta_clip_rot,
                )

                exec_action = base_action.copy()
                if refiner is not None:
                    exec_action = refiner.step(
                        a_base_7d=base_action,
                        step_idx=chunk_step,
                        force_reading=raw_force,
                        gripper_z=float(obs.gripper_pose[2]),
                        wrist_depth=depth_tensor_96,
                        ft_hist=force_hist,
                        proprio=proprio,
                        gripper_pose=obs.gripper_pose,
                        gripper_open=float(obs.gripper_open),
                    )
                    manager = refiner.manager
                else:
                    rollout_manager.update(
                        force_reading=raw_force,
                        gripper_pose=obs.gripper_pose,
                        gripper_open=float(obs.gripper_open),
                        depth_proximity=StageAwareRefiner.compute_depth_proximity(depth_tensor_96),
                        base_action=base_action,
                    )
                    manager = rollout_manager

                phase_id = int(manager.phase)
                seen_steps += 1
                seen_phase_counts[phase_id] = seen_phase_counts.get(phase_id, 0) + 1
                failure_mode = int(manager.failure_mode)
                depth_prox = StageAwareRefiner.compute_depth_proximity(depth_tensor_96)
                gripper_context = make_gripper_context(base_action, future_gripper_actions)
                planner_close_intent, planner_close_intent_strength = planner_close_intent_from_context(
                    gripper_context,
                    threshold=args.planner_close_threshold,
                )
                matched_ref_idx = match_reference_index(
                    npz_data,
                    obs.gripper_pose,
                    reference_candidates,
                    rot_weight=args.match_rot_weight,
                )
                frames_to_reference_trigger = (
                    int(reference_anchor_idx - matched_ref_idx)
                    if matched_ref_idx >= 0 and reference_anchor_idx >= 0
                    else -1
                )
                if basin_center_pose is not None:
                    raw_delta_align_target = pose_delta_local_between(obs.gripper_pose, basin_center_pose)
                    delta_align_target = clip_delta(
                        raw_delta_align_target,
                        args.delta_clip_pos,
                        args.delta_clip_rot,
                    )
                else:
                    raw_delta_align_target = np.zeros(6, dtype=np.float32)
                    delta_align_target = np.zeros(6, dtype=np.float32)
                basin_distance, delta_xy, _, delta_yaw = compute_basin_metrics(
                    raw_delta_align_target,
                    r_xy=args.basin_radius_xy,
                    r_z=args.basin_radius_z,
                    r_yaw=args.basin_radius_yaw,
                )
                rollout_still_open = float(obs.gripper_open) >= args.open_gripper_threshold
                basin_positive = float(
                    rollout_still_open
                    and phase_id == int(StagePhase.ALIGN)
                    and basin_distance <= 1.0
                )
                support_align = (phase_id == int(StagePhase.ALIGN)) if args.support_align_only else True
                planner_conditioned_support = bool(support_align and rollout_still_open and planner_close_intent)
                geometry_conditioned_pose_support = bool(
                    support_align
                    and rollout_still_open
                    and basin_center_pose is not None
                    and abs(float(raw_delta_align_target[2])) <= float(args.pose_support_abs_z_threshold)
                    and float(delta_xy) <= float(args.pose_support_xy_threshold)
                    and (
                        float(args.pose_support_yaw_threshold) < 0.0
                        or float(abs(delta_yaw)) <= float(args.pose_support_yaw_threshold)
                    )
                    and (
                        float(args.pose_support_depth_threshold) < 0.0
                        or (
                            depth_prox is not None
                            and np.isfinite(depth_prox)
                            and float(depth_prox) <= float(args.pose_support_depth_threshold)
                        )
                    )
                )
                background_align_support = bool(
                    support_align
                    and rollout_still_open
                    and (not planner_close_intent)
                    and 0 <= frames_to_reference_trigger <= int(args.non_intent_keep_window)
                )
                if args.dataset_view == "basin_pose_view":
                    keep_sample = geometry_conditioned_pose_support
                elif args.dataset_view == "basin_trigger_view":
                    keep_sample = planner_conditioned_support or background_align_support
                else:
                    keep_sample = geometry_conditioned_pose_support or planner_conditioned_support or background_align_support
                sample_history_idx = len(episode_history)
                if keep_sample:
                    episode_samples.append(
                        {
                            "history_idx": sample_history_idx,
                            "wrist_depth": depth_tensor_96.numpy().astype(np.float32)
                            if depth_tensor_96 is not None
                            else np.zeros((1, 96, 96), dtype=np.float32),
                            "ft_hist": force_hist.numpy().astype(np.float32)
                            if force_hist is not None
                            else np.zeros((32, FORCE_DIM), dtype=np.float32),
                            "proprio": proprio.astype(np.float32),
                            "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                            "basin_center_pose_7d": (
                                np.asarray(basin_center_pose, dtype=np.float32)
                                if basin_center_pose is not None
                                else np.zeros(7, dtype=np.float32)
                            ),
                            "reference_anchor_pose_7d": (
                                np.asarray(npz_data["gripper_pose"][reference_anchor_idx], dtype=np.float32)
                                if reference_anchor_idx >= 0
                                else np.zeros(7, dtype=np.float32)
                            ),
                            "base_action": base_action_local.astype(np.float32),
                            "gripper_context": gripper_context.astype(np.float32),
                            "interaction_role": interaction_role_code,
                            "step_idx": chunk_step,
                            "delta_target": delta_target.astype(np.float32),
                            "delta_align_target": delta_align_target.astype(np.float32),
                            "contact_mask": phase_to_contact_mask(phase_id),
                            "phase_label": phase_id,
                            "phase_id": phase_id,
                            "phase_age": float(manager.phase_age),
                            "steps_since_last_replan": float(manager.steps_since_last_replan),
                            "stage_role": stage_role_from_phase(phase_id),
                            "failure_mode": failure_mode,
                            "transition_flag": int(manager.last_transitioned),
                            "subgoal_progress": float(manager.get_subgoal_progress()),
                            "frames_to_reference_trigger": int(frames_to_reference_trigger),
                            "rollout_gripper_open": float(obs.gripper_open),
                            "depth_proximity": float(depth_prox) if depth_prox is not None and np.isfinite(depth_prox) else -1.0,
                            "planner_close_intent": float(planner_close_intent),
                            "planner_close_intent_strength": float(planner_close_intent_strength),
                            "geometry_conditioned_pose_support": int(geometry_conditioned_pose_support),
                            "planner_conditioned_support": int(planner_conditioned_support),
                            "background_align_support": int(background_align_support),
                            "basin_distance": float(basin_distance),
                            "basin_positive": float(basin_positive),
                            "delta_xy": float(delta_xy),
                            "delta_yaw": float(delta_yaw),
                            "base_action_world": np.asarray(base_action[:6], dtype=np.float32),
                            "delta_target_world": np.asarray(expert_action[:6] - base_action[:6], dtype=np.float32),
                            "is_augmented": 0,
                            "augment_offset_local": np.zeros(6, dtype=np.float32),
                        }
                    )

                history_entry = {
                    "invalid": False,
                    "next_gripper_open": float(obs.gripper_open),
                    "next_z": float(obs.gripper_pose[2]),
                    "reward": 0.0,
                    "terminate": False,
                }
                abs_action = delta_to_absolute(exec_action, obs.gripper_pose)
                if refiner is not None:
                    abs_action[:3] = refiner.safety.clamp_workspace(abs_action[:3])
                try:
                    obs, reward, terminate = task.step(abs_action)
                    history_entry.update(
                        {
                            "next_gripper_open": float(obs.gripper_open),
                            "next_z": float(obs.gripper_pose[2]),
                            "reward": float(reward),
                            "terminate": bool(terminate),
                        }
                    )
                except InvalidActionError as exc:
                    print(f"  [stage_collect invalid] {ep_name} step={rollout_step}: {exc}")
                    action_queue.clear()
                    chunk_step = 0
                    history_entry["invalid"] = True
                    if refiner is not None and hasattr(refiner, "on_invalid_action"):
                        _ = refiner.on_invalid_action(exec_action, raw_force)
                        recovery_abs = safe_recovery_absolute(
                            obs.gripper_pose,
                            obs.gripper_open,
                            lift=0.008,
                            safety=refiner.safety,
                        )
                        try:
                            obs, reward, terminate = task.step(recovery_abs)
                            history_entry.update(
                                {
                                    "next_gripper_open": float(obs.gripper_open),
                                    "next_z": float(obs.gripper_pose[2]),
                                    "reward": float(reward),
                                    "terminate": bool(terminate),
                                }
                            )
                        except InvalidActionError:
                            episode_history.append(history_entry)
                            continue
                    else:
                        episode_history.append(history_entry)
                        continue
                episode_history.append(history_entry)
                chunk_step += 1
                if refiner is not None and refiner.should_replan():
                    if len(action_queue) > 0:
                        refiner.note_replan()
                    action_queue.clear()
                if reward > 0 or terminate:
                    break

            def append_record(
                record: dict,
                readiness_label: float,
                hold_label: float,
                negative_reason: int,
                post_trigger_stability_proxy: float,
                grasp_lift_proxy: float,
                reopen_after_trigger: float,
                no_progress_after_trigger: float,
                invalid_after_trigger: float,
                ready_to_close: float,
                gripper_state_target: int,
                planner_close_too_early: float,
                expert_hold_after_close: float,
            ) -> None:
                nonlocal total_samples, kept_align_focus, base_support_sample_count, augmented_sample_count
                nonlocal support_source_counts, geometry_support_close_intent_counts
                nonlocal kept_planner_close_too_early_count, geometry_planner_close_too_early_count
                phase_id_local = int(record["phase_id"])
                failure_mode_local = int(record["failure_mode"])
                phase_counts[phase_id_local] += 1
                failure_counts[failure_mode_local] = failure_counts.get(failure_mode_local, 0) + 1
                kept_align_focus += int(phase_id_local == int(StagePhase.ALIGN))
                basin_positive_counts[int(readiness_label)] = basin_positive_counts.get(int(readiness_label), 0) + 1
                hold_counts[int(hold_label)] = hold_counts.get(int(hold_label), 0) + 1
                gripper_state_counts[int(gripper_state_target)] = gripper_state_counts.get(int(gripper_state_target), 0) + 1
                planner_close_intent_counts[int(record["planner_close_intent"] > 0.5)] = planner_close_intent_counts.get(int(record["planner_close_intent"] > 0.5), 0) + 1
                negative_reason_counts[int(negative_reason)] = negative_reason_counts.get(int(negative_reason), 0) + 1
                post_close_values.append(float(post_trigger_stability_proxy))
                reopen_values.append(float(reopen_after_trigger))
                lift_values.append(float(grasp_lift_proxy))
                delta_xy_values.append(float(record["delta_xy"]))
                delta_yaw_values.append(float(record["delta_yaw"]))
                basin_distance_values.append(float(record["basin_distance"]))
                if int(record["frames_to_reference_trigger"]) >= 0:
                    frames_to_trigger_values.append(float(record["frames_to_reference_trigger"]))

                if int(record.get("is_augmented", 0)) > 0:
                    augmented_sample_count += 1
                else:
                    base_support_sample_count += 1

                buffers["wrist_depth"].append(record["wrist_depth"])
                buffers["ft_hist"].append(record["ft_hist"])
                buffers["proprio"].append(record["proprio"])
                buffers["current_pose_7d"].append(np.asarray(record["current_pose_7d"], dtype=np.float32))
                buffers["basin_center_pose_7d"].append(np.asarray(record["basin_center_pose_7d"], dtype=np.float32))
                buffers["reference_anchor_pose_7d"].append(np.asarray(record["reference_anchor_pose_7d"], dtype=np.float32))
                buffers["base_action"].append(record["base_action"])
                buffers["gripper_context"].append(record["gripper_context"])
                buffers["interaction_role"].append(int(record["interaction_role"]))
                buffers["step_idx"].append(int(record["step_idx"]))
                buffers["delta_target"].append(record["delta_target"])
                buffers["delta_align_target"].append(record["delta_align_target"])
                buffers["contact_mask"].append(int(record["contact_mask"]))
                buffers["phase_label"].append(int(record["phase_label"]))
                buffers["phase_id"].append(int(record["phase_id"]))
                buffers["phase_age"].append(float(record["phase_age"]))
                buffers["steps_since_last_replan"].append(float(record["steps_since_last_replan"]))
                buffers["stage_role"].append(int(record["stage_role"]))
                buffers["failure_mode"].append(int(record["failure_mode"]))
                buffers["transition_flag"].append(int(record["transition_flag"]))
                buffers["subgoal_progress"].append(float(record["subgoal_progress"]))
                buffers["frames_to_reference_trigger"].append(int(record["frames_to_reference_trigger"]))
                buffers["rollout_gripper_open"].append(float(record["rollout_gripper_open"]))
                buffers["depth_proximity"].append(float(record["depth_proximity"]))
                buffers["planner_close_intent"].append(float(record["planner_close_intent"]))
                buffers["planner_close_intent_strength"].append(float(record["planner_close_intent_strength"]))
                buffers["geometry_conditioned_pose_support"].append(int(record.get("geometry_conditioned_pose_support", 0)))
                buffers["planner_conditioned_support"].append(int(record.get("planner_conditioned_support", 0)))
                buffers["background_align_support"].append(int(record.get("background_align_support", 0)))
                buffers["readiness_label"].append(float(readiness_label))
                buffers["hold_label"].append(float(hold_label))
                buffers["basin_distance"].append(float(record["basin_distance"]))
                buffers["negative_reason"].append(int(negative_reason))
                buffers["post_close_stability_proxy"].append(float(post_trigger_stability_proxy))
                buffers["grasp_lift_proxy"].append(float(grasp_lift_proxy))
                buffers["reopen_within_horizon"].append(float(reopen_after_trigger))
                buffers["no_progress_after_trigger"].append(float(no_progress_after_trigger))
                buffers["invalid_after_trigger"].append(float(invalid_after_trigger))
                buffers["ready_to_close"].append(float(ready_to_close))
                buffers["gripper_state_target"].append(int(gripper_state_target))
                buffers["planner_close_too_early"].append(float(planner_close_too_early))
                buffers["expert_hold_after_close"].append(float(expert_hold_after_close))
                buffers["is_augmented"].append(int(record.get("is_augmented", 0)))
                buffers["augment_offset_local"].append(np.asarray(record.get("augment_offset_local", np.zeros(6, dtype=np.float32)), dtype=np.float32))
                if int(record.get("geometry_conditioned_pose_support", 0)) > 0:
                    support_source_counts["geometry"] += 1
                    geometry_support_close_intent_counts[int(float(record["planner_close_intent"]) > 0.5)] += 1
                    if float(planner_close_too_early) > 0.5:
                        geometry_planner_close_too_early_count += 1
                elif int(record.get("planner_conditioned_support", 0)) > 0:
                    support_source_counts["planner_close"] += 1
                elif int(record.get("background_align_support", 0)) > 0:
                    support_source_counts["background"] += 1
                if float(planner_close_too_early) > 0.5:
                    kept_planner_close_too_early_count += 1
                total_samples += 1

            for sample in episode_samples:
                post_trigger_stability_proxy, grasp_lift_proxy, reopen_after_trigger, no_progress_after_trigger, invalid_after_trigger = compute_rollout_trigger_outcomes(
                    episode_history,
                    start_idx=int(sample["history_idx"]),
                    hold_horizon=args.hold_horizon,
                    open_threshold=args.open_gripper_threshold,
                    lift_threshold=args.lift_threshold,
                )
                basin_positive = float(sample["basin_positive"])
                negative_reason = classify_negative_reason(
                    basin_positive=basin_positive,
                    basin_distance=float(sample["basin_distance"]),
                    post_trigger_stability_proxy=post_trigger_stability_proxy,
                    reopen_after_trigger=reopen_after_trigger,
                    no_progress_after_trigger=no_progress_after_trigger,
                    invalid_after_trigger=invalid_after_trigger,
                )
                hold_label = 0.0
                readiness_label = basin_positive
                gripper_state_target = 1 if basin_positive > 0.5 else 0
                ready_to_close = basin_positive
                expert_hold_after_close = 0.0
                planner_close_too_early = float(sample["planner_close_intent"] > 0.5 and basin_positive < 0.5)
                append_record(
                    record=sample,
                    readiness_label=readiness_label,
                    hold_label=hold_label,
                    negative_reason=negative_reason,
                    post_trigger_stability_proxy=post_trigger_stability_proxy,
                    grasp_lift_proxy=grasp_lift_proxy,
                    reopen_after_trigger=reopen_after_trigger,
                    no_progress_after_trigger=no_progress_after_trigger,
                    invalid_after_trigger=invalid_after_trigger,
                    ready_to_close=ready_to_close,
                    gripper_state_target=gripper_state_target,
                    planner_close_too_early=planner_close_too_early,
                    expert_hold_after_close=expert_hold_after_close,
                )

                if (
                    args.enable_local_perturb_aug
                    and args.dataset_view == "basin_pose_view"
                    and bool(sample.get("geometry_conditioned_pose_support", 0) > 0)
                    and perturb_offsets
                    and basin_center_pose is not None
                ):
                    for offset_local in perturb_offsets:
                        pose_origin = basin_center_pose if args.augment_pose_origin == "basin_center" else sample["current_pose_7d"]
                        synth_pose = apply_local_offset_to_pose(pose_origin, offset_local)
                        proprio_aug = np.asarray(sample["proprio"], dtype=np.float32).copy()
                        if proprio_aug.shape[0] >= 14:
                            proprio_aug[7:14] = synth_pose
                        base_action_aug = world_delta_to_local(sample["base_action_world"], synth_pose[3:7]).astype(np.float32)
                        delta_target_aug = clip_delta(
                            world_delta_to_local(sample["delta_target_world"], synth_pose[3:7]),
                            args.delta_clip_pos,
                            args.delta_clip_rot,
                        )
                        delta_align_target_aug = clip_delta(
                            pose_delta_local_between(synth_pose, basin_center_pose),
                            args.delta_clip_pos,
                            args.delta_clip_rot,
                        )
                        basin_distance_aug, delta_xy_aug, _, delta_yaw_aug = compute_basin_metrics(
                            delta_align_target_aug,
                            r_xy=args.basin_radius_xy,
                            r_z=args.basin_radius_z,
                            r_yaw=args.basin_radius_yaw,
                        )
                        matched_ref_idx_aug = match_reference_index(
                            npz_data,
                            synth_pose,
                            reference_candidates,
                            rot_weight=args.match_rot_weight,
                        )
                        frames_to_reference_trigger_aug = (
                            int(reference_anchor_idx - matched_ref_idx_aug)
                            if matched_ref_idx_aug >= 0 and reference_anchor_idx >= 0
                            else -1
                        )
                        basin_positive_aug = float(
                            phase_id == int(StagePhase.ALIGN)
                            and float(sample["rollout_gripper_open"]) >= args.open_gripper_threshold
                            and basin_distance_aug <= 1.0
                        )
                        negative_reason_aug = -1 if basin_positive_aug > 0.5 else 0
                        aug_record = dict(sample)
                        aug_record.update(
                            {
                                "proprio": proprio_aug.astype(np.float32),
                                "base_action": base_action_aug.astype(np.float32),
                                "delta_target": delta_target_aug.astype(np.float32),
                                "delta_align_target": delta_align_target_aug.astype(np.float32),
                                "frames_to_reference_trigger": int(frames_to_reference_trigger_aug),
                                "basin_distance": float(basin_distance_aug),
                                "basin_positive": float(basin_positive_aug),
                                "delta_xy": float(delta_xy_aug),
                                "delta_yaw": float(delta_yaw_aug),
                                "is_augmented": 1,
                                "augment_offset_local": np.asarray(offset_local, dtype=np.float32),
                                "geometry_conditioned_pose_support": 1,
                                "planner_conditioned_support": 0,
                                "background_align_support": 0,
                            }
                        )
                        append_record(
                            record=aug_record,
                            readiness_label=basin_positive_aug,
                            hold_label=0.0,
                            negative_reason=negative_reason_aug,
                            post_trigger_stability_proxy=1.0 if basin_positive_aug > 0.5 else 0.0,
                            grasp_lift_proxy=1.0 if basin_positive_aug > 0.5 else 0.0,
                            reopen_after_trigger=0.0,
                            no_progress_after_trigger=0.0,
                            invalid_after_trigger=0.0,
                            ready_to_close=basin_positive_aug,
                            gripper_state_target=1 if basin_positive_aug > 0.5 else 0,
                            planner_close_too_early=float(sample["planner_close_intent"] > 0.5 and basin_positive_aug < 0.5),
                            expert_hold_after_close=0.0,
                        )

                if len(buffers["delta_target"]) >= args.shard_size:
                    flush_shard(output_dir, shard_idx, buffers)
                    shard_idx += 1
                    for key in buffers:
                        buffers[key].clear()

            print(f"[stage_collect] {ep_name}: total_samples={total_samples}")
    finally:
        env.shutdown()

    if len(buffers["delta_target"]) > 0:
        flush_shard(output_dir, shard_idx, buffers)

    negative_only = {k: int(v) for k, v in negative_reason_counts.items() if int(k) >= 0}
    negative_total = max(sum(negative_only.values()), 1)
    dominant_negative_fraction = max(negative_only.values(), default=0) / negative_total
    meta = {
        "checkpoint_dir": args.checkpoint_dir,
        "task_name": args.task_name,
        "rollout_mode": args.rollout_mode,
        "subgoal_reset_mode": args.subgoal_reset_mode,
        "subgoal_stage": args.subgoal_stage,
        "total_samples": total_samples,
        "seen_steps": seen_steps,
        "align_focus": args.align_focus,
        "align_window": args.align_window,
        "kept_align_focus": kept_align_focus,
        "kept_close_window": kept_close_window,
        "pregrasp_focus": args.pregrasp_focus,
        "pregrasp_window": args.pregrasp_window,
        "pregrasp_reset": args.pregrasp_reset,
        "near_depth_threshold": args.near_depth_threshold,
        "open_gripper_threshold": args.open_gripper_threshold,
        "ready_close_window": args.ready_close_window,
        "planner_close_threshold": args.planner_close_threshold,
        "planner_close_lookahead": args.planner_close_lookahead,
        "hold_horizon": args.hold_horizon,
        "lift_threshold": args.lift_threshold,
        "support_align_only": args.support_align_only,
        "dataset_view": args.dataset_view,
        "basin_center_k": args.basin_center_k,
        "basin_center_mode": args.basin_center_mode,
        "success_region_window": args.success_region_window,
        "success_region_exclude_last": args.success_region_exclude_last,
        "non_intent_keep_window": args.non_intent_keep_window,
        "match_rot_weight": args.match_rot_weight,
        "interaction_role": args.interaction_role,
        "basin_radius_xy": args.basin_radius_xy,
        "basin_radius_z": args.basin_radius_z,
        "basin_radius_yaw": args.basin_radius_yaw,
        "pose_support_basin_distance_threshold": args.pose_support_basin_distance_threshold,
        "pose_support_xy_threshold": args.pose_support_xy_threshold,
        "pose_support_abs_z_threshold": args.pose_support_abs_z_threshold,
        "pose_support_yaw_threshold": args.pose_support_yaw_threshold,
        "pose_support_depth_threshold": args.pose_support_depth_threshold,
        "enable_local_perturb_aug": args.enable_local_perturb_aug,
        "augment_xy_values": parse_float_list_arg(args.augment_xy_values),
        "augment_z_values": parse_float_list_arg(args.augment_z_values),
        "augment_yaw_values": parse_float_list_arg(args.augment_yaw_values),
        "augment_include_diagonals": bool(args.augment_include_diagonals),
        "augment_pose_origin": args.augment_pose_origin,
        "base_support_sample_count": int(base_support_sample_count),
        "augmented_sample_count": int(augmented_sample_count),
        "support_source_counts": support_source_counts,
        "geometry_support_close_intent_counts": geometry_support_close_intent_counts,
        "kept_planner_close_too_early_count": int(kept_planner_close_too_early_count),
        "geometry_planner_close_too_early_count": int(geometry_planner_close_too_early_count),
        "geometry_planner_close_too_early_fraction": float(
            geometry_planner_close_too_early_count / max(support_source_counts["geometry"], 1)
        ),
        "planner_close_intent_counts": planner_close_intent_counts,
        "readiness_counts": basin_positive_counts,
        "basin_positive_counts": basin_positive_counts,
        "hold_counts": hold_counts,
        "gripper_state_counts": gripper_state_counts,
        "negative_reason_counts": negative_reason_counts,
        "phase_counts": phase_counts,
        "seen_phase_counts": seen_phase_counts,
        "failure_counts": failure_counts,
        "post_close_stability_proxy_stats": summarize_float_stats(post_close_values),
        "reopen_within_horizon_counts": summarize_int_values(np.rint(reopen_values).astype(np.int64)) if len(reopen_values) > 0 else {},
        "grasp_lift_proxy_stats": summarize_float_stats(lift_values),
        "delta_xy_stats": summarize_float_stats(delta_xy_values),
        "delta_yaw_stats": summarize_float_stats(delta_yaw_values),
        "frames_to_expert_close_stats": summarize_float_stats(frames_to_trigger_values),
        "frames_to_reference_trigger_stats": summarize_float_stats(frames_to_trigger_values),
        "basin_distance_stats": summarize_float_stats(basin_distance_values),
        "actual_basin_distance_stats": summarize_float_stats(basin_distance_values),
        "dominant_negative_reason_fraction": float(dominant_negative_fraction),
        "negative_reason_health_ok": bool(dominant_negative_fraction <= 0.85),
        "label_frame": "ee_local",
        "interaction_role_codes": INTERACTION_ROLE_CODES,
    }
    with open(output_dir / "residual_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
