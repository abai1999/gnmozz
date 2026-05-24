#!/usr/bin/env python
"""Standalone RLBench evaluator for Coarse2Contact v2."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

_HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/mnt/ssd/guoning/hf-cache")
os.environ.setdefault("HF_HOME", _HF_CACHE_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_HF_CACHE_ROOT, "transformers"))
os.environ.setdefault("TORCH_HOME", os.path.join(_HF_CACHE_ROOT, "torch"))
os.environ.setdefault("TIMM_HOME", os.path.join(_HF_CACHE_ROOT, "timm"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    from moviepy import ImageSequenceClip
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from scipy.spatial.transform import Rotation

from scripts.evaluate_c2c_rlbench import (
    _jsonable_value,
    _lazy_import_tasks,
    compute_wrist_visibility_stats,
    delta_to_absolute,
    load_checkpoint,
    maybe_apply_workspace_filter,
    predict_actions,
    process_obs,
)
from scripts.evaluate_rlbench import resolve_live_target_handle, safe_live_target_pose_7d
from prismatic.robot.stage_target_provider import apply_yaw_symmetry_to_delta, build_phase1_teacher_targets, load_phase1_grasp_spec, pose_delta_local_between
from prismatic.robot.coarse2contact_v2.grasp_probe_shell import grasp_probe_inactive_reason, grasp_probe_shell_fields
from prismatic.robot.coarse2contact_v2.recovery_audit import in_close_ready_basin, in_near_grasp_basin, recovery_overshoot_flag

from prismatic.robot.coarse2contact_v2 import BasinRecoveryConfig, PrecisionSkillSupervisor, load_precision_task_spec, load_basin_state_calibration_report
from prismatic.robot.coarse2contact_v2.learned_force import LearnedForceClassifierAdapter
from prismatic.robot.coarse2contact_v2.learned_localizer import LearnedDepthLocalizerAdapter
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from prismatic.vla.constants import FORCE_HISTORY_LEN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _select_task_spec(task_name: str):
    return load_precision_task_spec(task_name)


def _maybe_attach_basin_state_calibration(task_spec, calibration_report: str | None):
    if task_spec is None:
        return None
    if calibration_report:
        calibration = load_basin_state_calibration_report(calibration_report)
        if calibration is not None:
            task_spec.runtime_flags["basin_state_calibration"] = calibration.to_dict()
            task_spec.runtime_flags["basin_state_calibration_report"] = str(calibration_report)
            return task_spec
    return task_spec


def _mode_to_flags(mode: str) -> tuple[bool, bool]:
    if mode == "planner_only":
        return False, False
    return mode in {"c2c_stage_shadow", "grasp_depth_apply", "spoke_depth_apply", "force_recovery", "full_owner_by_stage", "basin_recovery_shadow", "basin_recovery_only"}, mode in {"c2c_stage_shadow", "basin_recovery_shadow"}


def _compose_video_frame(front_rgb: np.ndarray, wrist_rgb: np.ndarray | None, *, layout: str = "front") -> np.ndarray:
    front = np.asarray(front_rgb, dtype=np.uint8)
    if layout == "front" or wrist_rgb is None:
        return front
    wrist = np.asarray(wrist_rgb, dtype=np.uint8)
    if wrist.ndim != 3 or wrist.shape[-1] != 3:
        return front
    if wrist.shape[0] != front.shape[0]:
        scale = float(front.shape[0]) / max(float(wrist.shape[0]), 1.0)
        new_w = max(1, int(round(wrist.shape[1] * scale)))
        wrist = np.asarray(Image.fromarray(wrist).resize((new_w, front.shape[0]), resample=Image.NEAREST), dtype=np.uint8)
    if layout == "wrist":
        return wrist
    sep = np.full((front.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([front, sep, wrist], axis=1)


def _stack_runtime_obs_rows(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    out: dict[str, np.ndarray] = {}
    keys = sorted(rows[0].keys())
    for key in keys:
        values = [row[key] for row in rows]
        first = values[0]
        if isinstance(first, np.ndarray):
            out[key] = np.stack(values, axis=0)
        else:
            out[key] = np.asarray(values)
    return out


def _safe_pose_from_handle(handle) -> np.ndarray | None:
    if handle is None:
        return None
    try:
        pose = np.asarray(handle.get_pose(), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if pose.size < 7 or not np.all(np.isfinite(pose[:7])):
        return None
    return pose[:7].astype(np.float32)


def _resolve_live_frame_handle(task, candidate_names: tuple[str, ...]) -> object | None:
    task_obj = getattr(task, "_task", task)
    for name in candidate_names:
        if hasattr(task_obj, name):
            return getattr(task_obj, name)
    return None


def _episode_privileged_frame_pack(task, obs) -> dict[str, np.ndarray | None]:
    nan_pose = np.full((7,), np.nan, dtype=np.float32)
    gripper_pose = np.asarray(getattr(obs, "gripper_pose", np.full((7,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1)
    gripper_pose = gripper_pose[:7] if gripper_pose.size >= 7 else nan_pose.copy()
    ring_handle = resolve_live_target_handle(task)
    ring_pose = _safe_pose_from_handle(ring_handle)
    spoke_handle = _resolve_live_frame_handle(task, ("_success_centre", "_target", "_target_frame", "_spoke_axis", "_spoke_axis_frame"))
    spoke_pose = _safe_pose_from_handle(spoke_handle)
    task_low_dim_pose = None
    try:
        task_low_dim_pose = np.asarray(task.get_low_dim_state(), dtype=np.float32).reshape(-1)
        if task_low_dim_pose.size >= 7 and np.all(np.isfinite(task_low_dim_pose[:7])):
            task_low_dim_pose = task_low_dim_pose[:7].astype(np.float32)
        else:
            task_low_dim_pose = None
    except Exception:
        task_low_dim_pose = None
    if spoke_pose is None and task_low_dim_pose is not None:
        spoke_pose = task_low_dim_pose.copy()
    return {
        "episode_gripper_pose_7d": gripper_pose.astype(np.float32),
        "episode_ring_pose_7d": ring_pose if ring_pose is not None else nan_pose.copy(),
        "episode_spoke_pose_7d": spoke_pose if spoke_pose is not None else nan_pose.copy(),
        "episode_target_pose_7d": ring_pose if ring_pose is not None else nan_pose.copy(),
        "episode_task_low_dim_pose_7d": task_low_dim_pose if task_low_dim_pose is not None else nan_pose.copy(),
    }


def _grasp_teacher_error_from_pack(
    frame_pack: dict[str, np.ndarray | None] | None,
    grasp_spec,
) -> np.ndarray | None:
    if frame_pack is None or grasp_spec is None:
        return None
    gripper_pose = np.asarray(frame_pack.get("episode_gripper_pose_7d", np.full((7,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1)[:7]
    ring_pose = np.asarray(frame_pack.get("episode_ring_pose_7d", np.full((7,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1)[:7]
    if not np.all(np.isfinite(gripper_pose[:7])) or not np.all(np.isfinite(ring_pose[:7])):
        return None
    try:
        _, grasp_commit = build_phase1_teacher_targets(ring_pose, grasp_spec)
        if not np.all(np.isfinite(grasp_commit[:7])):
            return None
        delta = pose_delta_local_between(gripper_pose, grasp_commit)
        return apply_yaw_symmetry_to_delta(delta, float(getattr(grasp_spec, "yaw_symmetry_period", -1.0))).astype(np.float32)
    except Exception:
        return None


def _bounded_xy_oracle_probe_step(error_local_6d: np.ndarray, *, xy_gain: float, max_xy_step: float) -> np.ndarray:
    correction = np.zeros(6, dtype=np.float32)
    err = np.asarray(error_local_6d, dtype=np.float32).reshape(-1)
    if err.size < 2 or not np.all(np.isfinite(err[:2])):
        return correction
    correction[0] = float(xy_gain * err[0])
    correction[1] = float(xy_gain * err[1])
    norm = float(np.linalg.norm(correction[:2]))
    if norm > float(max_xy_step) > 0.0:
        correction[:2] = correction[:2] * (float(max_xy_step) / max(norm, 1.0e-9))
    return correction.astype(np.float32)


def _absolute_to_world_delta(abs_action: np.ndarray, current_gripper_pose: np.ndarray) -> np.ndarray:
    action = np.asarray(abs_action, dtype=np.float32).reshape(-1)
    pose = np.asarray(current_gripper_pose, dtype=np.float32).reshape(-1)
    delta = np.zeros(6, dtype=np.float32)
    if action.size < 7 or pose.size < 7 or not np.all(np.isfinite(action[:7])) or not np.all(np.isfinite(pose[:7])):
        return delta
    delta[:3] = action[:3] - pose[:3]
    try:
        r_new = Rotation.from_quat(action[3:7])
        r_cur = Rotation.from_quat(pose[3:7])
        delta[3:6] = (r_new * r_cur.inv()).as_rotvec().astype(np.float32)
    except Exception:
        delta[3:6] = 0.0
    return delta.astype(np.float32)


def _grasp_probe_metric_fields(
    pre_probe: np.ndarray,
    post_probe: np.ndarray,
    *,
    visibility_bucket: str,
    near_grasp_xy_threshold: float,
    near_grasp_yaw_threshold: float,
    close_ready_xy_threshold: float,
    close_ready_yaw_threshold: float,
) -> dict[str, object]:
    pre = np.asarray(pre_probe, dtype=np.float32).reshape(-1)[:4]
    post = np.asarray(post_probe, dtype=np.float32).reshape(-1)[:4]
    visible = str(visibility_bucket) != "prior_only"
    return {
        "grasp_probe_pre_xy_error": float(np.hypot(float(pre[0]), float(pre[1]))),
        "grasp_probe_post_xy_error": float(np.hypot(float(post[0]), float(post[1]))),
        "grasp_probe_pre_error_norm": float(np.linalg.norm(np.asarray([pre[0], pre[1], 0.04 * pre[3]], dtype=np.float32))),
        "grasp_probe_post_error_norm": float(np.linalg.norm(np.asarray([post[0], post[1], 0.04 * post[3]], dtype=np.float32))),
        "grasp_probe_xy_contracted": bool(float(np.hypot(float(post[0]), float(post[1]))) < float(np.hypot(float(pre[0]), float(pre[1]))) - 1.0e-9),
        "grasp_probe_overshoot": bool(recovery_overshoot_flag(pre[:3], post[:3])),
        "grasp_probe_micro_entry_ready_before": bool(
            visible
            and in_near_grasp_basin(
                float(pre[0]),
                float(pre[1]),
                float(pre[3]),
                xy_threshold=float(near_grasp_xy_threshold),
                yaw_threshold=float(near_grasp_yaw_threshold),
            )
        ),
        "grasp_probe_micro_entry_ready_after": bool(
            visible
            and in_near_grasp_basin(
                float(post[0]),
                float(post[1]),
                float(post[3]),
                xy_threshold=float(near_grasp_xy_threshold),
                yaw_threshold=float(near_grasp_yaw_threshold),
            )
        ),
        "grasp_probe_near_grasp_before": bool(
            in_near_grasp_basin(
                float(pre[0]),
                float(pre[1]),
                float(pre[3]),
                xy_threshold=float(near_grasp_xy_threshold),
                yaw_threshold=float(near_grasp_yaw_threshold),
            )
        ),
        "grasp_probe_near_grasp_after": bool(
            in_near_grasp_basin(
                float(post[0]),
                float(post[1]),
                float(post[3]),
                xy_threshold=float(near_grasp_xy_threshold),
                yaw_threshold=float(near_grasp_yaw_threshold),
            )
        ),
        "grasp_probe_close_ready_before": bool(
            in_close_ready_basin(
                float(pre[0]),
                float(pre[1]),
                float(pre[3]),
                xy_threshold=float(close_ready_xy_threshold),
                yaw_threshold=float(close_ready_yaw_threshold),
            )
        ),
        "grasp_probe_close_ready_after": bool(
            in_close_ready_basin(
                float(post[0]),
                float(post[1]),
                float(post[3]),
                xy_threshold=float(close_ready_xy_threshold),
                yaw_threshold=float(close_ready_yaw_threshold),
            )
        ),
    }


def _nan_grasp_probe_metric_fields() -> dict[str, object]:
    return {
        "grasp_probe_pre_xy_error": float("nan"),
        "grasp_probe_post_xy_error": float("nan"),
        "grasp_probe_pre_error_norm": float("nan"),
        "grasp_probe_post_error_norm": float("nan"),
        "grasp_probe_xy_contracted": False,
        "grasp_probe_overshoot": False,
        "grasp_probe_micro_entry_ready_before": False,
        "grasp_probe_micro_entry_ready_after": False,
        "grasp_probe_near_grasp_before": False,
        "grasp_probe_near_grasp_after": False,
        "grasp_probe_close_ready_before": False,
        "grasp_probe_close_ready_after": False,
    }


def _prefix_grasp_probe_fields(fields: dict[str, object], prefix: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in fields.items():
        if key.startswith("grasp_probe_"):
            out[f"{prefix}_{key[len('grasp_probe_'):]}"] = value
        else:
            out[f"{prefix}_{key}"] = value
    return out


def build_c2c_v2_supervisor(args, task_spec):
    if args.mode == "planner_only" and task_spec is None:
        return None
    grasp_localizer = None
    spoke_localizer = None
    force_classifier = None
    if args.depth_localizer_backend == "learned":
        if not args.depth_localizer_ckpt:
            raise ValueError("--depth_localizer_ckpt is required when --depth_localizer_backend=learned")
        if args.allow_learned_depth_apply:
            print(
                "[c2c-v2] Current learned depth checkpoints are diagnostic-only; "
                "--allow_learned_depth_apply is ignored until the RingFrameLocalizer + GraspSkillHead path is trained.",
                flush=True,
            )
            args.allow_learned_depth_apply = False
        learned = LearnedDepthLocalizerAdapter.from_checkpoint(args.depth_localizer_ckpt, device=DEVICE)
        grasp_localizer = learned
        spoke_localizer = learned
    if args.force_classifier_backend == "learned":
        if not args.force_classifier_ckpt:
            raise ValueError("--force_classifier_ckpt is required when --force_classifier_backend=learned")
        force_classifier = LearnedForceClassifierAdapter.from_checkpoint(args.force_classifier_ckpt, device=torch.device("cpu"))
    basin_recovery_config = BasinRecoveryConfig(
        variant_name=str(args.basin_pullback_variant),
        yaw_control_enabled=bool(args.basin_pullback_variant != "xy_only"),
        visual_gain=float(args.basin_visual_gain),
        max_pullback_xy_step=float(args.basin_max_pullback_xy_step),
        max_recovery_steps=int(args.basin_max_recovery_steps),
    )
    return PrecisionSkillSupervisor(
        task_spec,
        mode=args.mode,
        shadow_only=bool(
            args.shadow_only
            or args.mode in {"c2c_stage_shadow", "basin_recovery_shadow"}
            or (args.depth_localizer_backend == "learned" and not args.allow_learned_depth_apply)
        ),
        grasp_localizer=grasp_localizer,
        spoke_localizer=spoke_localizer,
        force_classifier=force_classifier,
        basin_recovery_config=basin_recovery_config,
        max_xy_step=float(args.c2c_max_xy_step),
        max_yaw_step=float(args.c2c_max_yaw_step),
        max_dz_step=float(args.c2c_max_dz_step),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Coarse2Contact v2 RLBench evaluator")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument(
        "--mode",
        type=str,
        default="planner_only",
        choices=[
            "planner_only",
            "c2c_stage_shadow",
            "grasp_depth_apply",
            "spoke_depth_apply",
            "force_recovery",
            "full_owner_by_stage",
            "basin_recovery_shadow",
            "basin_recovery_only",
        ],
    )
    parser.add_argument("--depth_localizer_backend", type=str, default="heuristic", choices=["heuristic", "learned"])
    parser.add_argument("--depth_localizer_ckpt", type=str, default="")
    parser.add_argument("--allow_learned_depth_apply", action="store_true", default=False)
    parser.add_argument("--force_classifier_backend", type=str, default="rule", choices=["rule", "learned"])
    parser.add_argument("--force_classifier_ckpt", type=str, default="")
    parser.add_argument("--basin_pullback_variant", type=str, default="xy_plus_yaw", choices=["xy_only", "xy_plus_yaw"])
    parser.add_argument("--basin_visual_gain", type=float, default=0.35)
    parser.add_argument("--basin_max_pullback_xy_step", type=float, default=0.0030)
    parser.add_argument("--basin_max_recovery_steps", type=int, default=24)
    parser.add_argument("--c2c_grasp_probe_policy", type=str, default="off", choices=["off", "replay_oracle_xy"])
    parser.add_argument("--c2c_grasp_probe_xy_gain", type=float, default=0.35)
    parser.add_argument("--c2c_grasp_probe_max_xy_step", type=float, default=0.0030)
    parser.add_argument("--c2c_grasp_probe_horizon", type=int, default=1)
    parser.add_argument("--c2c_grasp_probe_flush_planner_queue", action="store_true", default=False)
    parser.add_argument("--c2c_grasp_probe_window_mode", type=str, default="stage", choices=["stage", "forced_shell"])
    parser.add_argument("--c2c_grasp_probe_shell_filter", type=str, default="off", choices=["off", "near_yaw_feasible", "coarse_yaw_feasible"])
    parser.add_argument("--near_grasp_xy_threshold", type=float, default=0.015)
    parser.add_argument("--near_grasp_yaw_threshold", type=float, default=0.08)
    parser.add_argument("--close_ready_xy_threshold", type=float, default=0.005)
    parser.add_argument("--close_ready_yaw_threshold", type=float, default=0.03)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--episode_indices", type=str, default="5,8,19")
    parser.add_argument("--max_steps", type=int, default=320)
    parser.add_argument("--eval_seed", type=int, default=3407)
    parser.add_argument("--depth_max", type=float, default=1.0)
    parser.add_argument("--output_root", type=str, default="runtime_artifacts/coarse2contact_v2/eval_3ep")
    parser.add_argument("--name_suffix", type=str, default="coarse2contact_v2_eval_3ep")
    parser.add_argument("--shadow_only", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="record_video", action="store_false")
    parser.add_argument("--video_layout", type=str, default="front", choices=["front", "wrist", "front_wrist"])
    parser.add_argument("--write_episode_videos", action="store_true", default=True)
    parser.add_argument("--no_episode_videos", dest="write_episode_videos", action="store_false")
    parser.add_argument("--record_gripper_trace", action="store_true", default=True)
    parser.add_argument("--no_gripper_trace", dest="record_gripper_trace", action="store_false")
    parser.add_argument("--dump_runtime_obs", action="store_true", default=False, help="Save per-step runtime observation bundles for offline failure mining.")
    parser.add_argument(
        "--dump_runtime_obs_all_episodes",
        action="store_true",
        default=False,
        help="If set with --dump_runtime_obs, save every episode instead of failures only.",
    )
    parser.add_argument(
        "--capture_failure_target_pose",
        action="store_true",
        default=False,
        help="Save privileged target pose into the runtime observation dump for offline label generation only.",
    )
    parser.add_argument("--no_best_gif", action="store_true", default=True)
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.set_defaults(planner_use_depth=False, planner_use_force=False)
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--workspace_clamp_mode", type=str, default="diagnostic", choices=["diagnostic", "hard", "tolerance"])
    parser.add_argument("--workspace_clamp_tolerance", type=float, default=0.0)
    parser.add_argument("--c2c_max_xy_step", type=float, default=0.0010)
    parser.add_argument("--c2c_max_yaw_step", type=float, default=0.035)
    parser.add_argument("--c2c_max_dz_step", type=float, default=0.0010)
    parser.add_argument("--runtime_obs_root", type=str, default="")
    parser.add_argument(
        "--basin_state_calibration_report",
        type=str,
        default="runtime_artifacts/coarse2contact_v2/reports/basin_state_calibration/basin_state_calibration.json",
        help="Optional basin-state calibration report JSON to tighten runtime apply/recovery axis policies.",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> float:
    _lazy_import_tasks()
    from rlbench.tasks import InsertOntoSquarePeg, PlugChargerInPowerSupply, PushButton, StackBlocks

    task_map = {
        "insert_onto_square_peg": InsertOntoSquarePeg,
        "stack_blocks": StackBlocks,
        "plug_charger_in_power_supply": PlugChargerInPowerSupply,
        "push_button": PushButton,
    }
    if args.task_name not in task_map:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {sorted(task_map)}")

    task_spec = _select_task_spec(args.task_name)
    task_spec = _maybe_attach_basin_state_calibration(task_spec, args.basin_state_calibration_report or None)
    grasp_spec = load_phase1_grasp_spec(args.task_name)
    if task_spec is None:
        print(f"[c2c-v2] No task spec found for {args.task_name}; falling back to planner-only runtime.", flush=True)
        args.mode = "planner_only"
        args.shadow_only = True
    if args.c2c_grasp_probe_policy != "off":
        if not args.dump_runtime_obs or not args.capture_failure_target_pose:
            raise ValueError("--c2c_grasp_probe_policy requires --dump_runtime_obs and --capture_failure_target_pose")
        if not args.shadow_only and args.mode not in {"c2c_stage_shadow", "basin_recovery_shadow"}:
            raise ValueError("--c2c_grasp_probe_policy is intended for shadow-only intervention data collection")

    vla, processor, action_head, proprio_projector, norm_stats = load_checkpoint(
        args.checkpoint_dir,
        use_depth=args.planner_use_depth,
        use_force=args.planner_use_force,
    )
    c2c = build_c2c_v2_supervisor(args, task_spec)

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

    env = Environment(
        MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()),
        obs_config=obs_config,
        headless=True,
    )
    env.launch()
    task = env.get_task(task_map[args.task_name])

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    trace_dir = output_dir / "gripper_traces"
    gate_frame_dir = output_dir / "gate_frames"
    probe_frame_dir = output_dir / "probe_frames"
    runtime_obs_dir = Path(args.runtime_obs_root) if args.runtime_obs_root else (output_dir / "runtime_observations")
    if args.record_video and args.write_episode_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
    if args.record_gripper_trace:
        trace_dir.mkdir(parents=True, exist_ok=True)
        gate_frame_dir.mkdir(parents=True, exist_ok=True)
        probe_frame_dir.mkdir(parents=True, exist_ok=True)
    if args.dump_runtime_obs:
        runtime_obs_dir.mkdir(parents=True, exist_ok=True)

    episode_indices = [int(x.strip()) for x in str(args.episode_indices).split(",") if x.strip()] if args.episode_indices else list(range(args.num_episodes))
    results = {
        "mode": args.mode,
        "task_name": args.task_name,
        "episode_indices": episode_indices,
        "successes": [],
        "episode_lengths": [],
        "invalid_action_counts": [],
        "stage_stats": [],
        "video_paths": [],
        "gripper_trace_paths": [],
        "runtime_obs_paths": [],
        "uses_privileged_target": False,
        "uses_rlbench_mask_runtime": False,
        "uses_privileged_label_for_eval": bool(args.dump_runtime_obs and args.capture_failure_target_pose),
        "c2c_grasp_probe_policy": str(args.c2c_grasp_probe_policy),
        "c2c_grasp_probe_xy_gain": float(args.c2c_grasp_probe_xy_gain),
        "c2c_grasp_probe_max_xy_step": float(args.c2c_grasp_probe_max_xy_step),
        "c2c_grasp_probe_horizon": int(args.c2c_grasp_probe_horizon),
        "c2c_grasp_probe_flush_planner_queue": bool(args.c2c_grasp_probe_flush_planner_queue),
        "c2c_grasp_probe_window_mode": str(args.c2c_grasp_probe_window_mode),
        "c2c_grasp_probe_shell_filter": str(args.c2c_grasp_probe_shell_filter),
        "depth_error_trend": [],
    }

    for loop_idx, ep_idx in enumerate(episode_indices):
        print(f"\n[c2c-v2] Episode {loop_idx + 1}/{len(episode_indices)} (ep={ep_idx})", flush=True)
        if args.eval_seed is not None:
            seed = int(args.eval_seed) + int(ep_idx)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        descriptions, obs = task.reset()
        instruction = descriptions[0] if descriptions else f"complete {args.task_name.replace('_', ' ')}"
        force_buffer: deque[np.ndarray] = deque(maxlen=FORCE_HISTORY_LEN)
        action_queue: list[np.ndarray] = []
        frames = []
        gripper_trace = []
        invalid_action_count = 0
        workspace_violation_count = 0
        workspace_violation_max = 0.0
        success = False
        reward = 0.0
        terminate = False
        runtime_obs_rows: list[dict[str, np.ndarray]] = []
        gate_frame_saved = False
        probe_frame_saved = False
        episode_target_pose_7d = None
        if c2c is not None:
            c2c.reset()
        if args.dump_runtime_obs and args.capture_failure_target_pose:
            try:
                episode_target_pose_7d = safe_live_target_pose_7d(resolve_live_target_handle(task))
            except Exception:
                episode_target_pose_7d = None

        for step_idx in range(args.max_steps):
            if args.record_video:
                frames.append(_compose_video_frame(obs.front_rgb.copy(), obs.wrist_rgb.copy() if obs.wrist_rgb is not None else None, layout=args.video_layout))

            front_pil, wrist_pil, proprio, depth_tensor, force_hist, raw_force = process_obs(
                obs,
                norm_stats,
                force_buffer,
                use_depth=args.use_depth,
                use_force=args.use_force,
                depth_max=args.depth_max,
            )
            wrist_valid_depth_ratio, wrist_depth_near_fraction, wrist_is_occluded, wrist_is_low_visibility = compute_wrist_visibility_stats(obs.wrist_depth)

            if not action_queue:
                actions = predict_actions(
                    vla,
                    processor,
                    action_head,
                    proprio_projector,
                    front_pil,
                    wrist_pil,
                    proprio,
                    depth_tensor if args.planner_use_depth else None,
                    force_hist if args.planner_use_force else None,
                    instruction,
                )
                action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(len(actions))]

            delta_action = action_queue.pop(0)
            base_delta_action = delta_action.copy()
            planner_chunk_local = world_delta_to_local(np.asarray(base_delta_action[:6], dtype=np.float32), np.asarray(obs.gripper_pose[3:7], dtype=np.float32)).astype(np.float32)
            privileged_frame_pack = _episode_privileged_frame_pack(task, obs) if args.dump_runtime_obs and args.capture_failure_target_pose else None
            probe_true_error_before = _grasp_teacher_error_from_pack(privileged_frame_pack, grasp_spec) if args.c2c_grasp_probe_policy != "off" else None
            trace_entry = {
                "episode_idx": int(ep_idx),
                "episode_loop_idx": int(loop_idx),
                "step": int(step_idx),
                "planner_action_world_6d": _jsonable_value(np.asarray(base_delta_action[:6], dtype=np.float32)),
                "planner_action_world_8d": _jsonable_value(np.asarray(base_delta_action, dtype=np.float32)),
                "planner_chunk_world_6d": _jsonable_value(np.asarray(base_delta_action[:6], dtype=np.float32)),
                "planner_chunk_local_6d": _jsonable_value(planner_chunk_local),
                "base_gripper_raw": float(base_delta_action[6]) if len(base_delta_action) > 6 else 0.0,
                "obs_gripper_open": float(obs.gripper_open),
                "wrist_valid_depth_ratio": float(wrist_valid_depth_ratio),
                "wrist_depth_near_fraction": float(wrist_depth_near_fraction),
                "wrist_is_occluded": bool(wrist_is_occluded),
                "wrist_is_low_visibility": bool(wrist_is_low_visibility),
                "invalid_action": False,
                "invalid_action_flag": False,
                "c2c_v2_stage": "planner_only",
                "planner_reaches_precontact": False,
                "planner_reaches_preinsert": False,
                "uses_privileged_target": False,
                "uses_rlbench_mask_runtime": False,
                "retry_id": 0,
                "depth_conf": None,
                "depth_obs_quality": None,
                "phase_owner": "planner",
                "phase_reason": "planner",
                "raw_wrench": _jsonable_value(np.asarray(raw_force if raw_force is not None else np.zeros(6, dtype=np.float32), dtype=np.float32)),
                "filtered_wrench": _jsonable_value(np.zeros(6, dtype=np.float32)),
                "mp4_path": None,
                "uses_privileged_runtime": False,
                "uses_privileged_label_for_eval": bool(args.dump_runtime_obs and args.capture_failure_target_pose),
            }
            if privileged_frame_pack is not None:
                trace_entry.update({k: _jsonable_value(v) for k, v in privileged_frame_pack.items()})

            if c2c is not None:
                delta_action = c2c.step(
                    delta_action,
                    observation=obs,
                    robot_state={
                        "invalid_action_flag": False,
                        "wrist_valid_depth_ratio": float(wrist_valid_depth_ratio),
                        "wrist_depth_near_fraction": float(wrist_depth_near_fraction),
                        "wrist_is_occluded": bool(wrist_is_occluded),
                        "wrist_is_low_visibility": bool(wrist_is_low_visibility),
                    },
                    task_spec=task_spec,
                    current_instruction=instruction,
                )
                trace_entry.update(_jsonable_value(c2c.get_last_trace()))
                if args.record_video and not gate_frame_saved and bool(c2c.get_last_trace().get("c2c_gate_active", False)) and frames:
                    gate_path = gate_frame_dir / f"ep{ep_idx:03d}_gate_start_step{step_idx:03d}.png"
                    Image.fromarray(np.asarray(frames[-1], dtype=np.uint8)).save(gate_path)
                    trace_entry["c2c_gate_frame_path"] = str(gate_path)
                    gate_frame_saved = True
                probe_visibility_bucket = str(
                    trace_entry.get("basin_recovery_visual_evidence_class", trace_entry.get("visual_observability_class", "prior_only"))
                )
                probe_stage = str(trace_entry.get("c2c_v2_stage", ""))
                probe_shell_fields = _grasp_probe_shell_fields(
                    probe_true_error_before,
                    near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
                    near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
                    max_xy_step=float(args.c2c_grasp_probe_max_xy_step),
                    horizon_steps=int(max(1, int(args.c2c_grasp_probe_horizon))),
                )
                probe_has_error = bool(probe_true_error_before is not None)
                probe_finite_xy = bool(
                    probe_true_error_before is not None
                    and np.all(np.isfinite(np.asarray(probe_true_error_before, dtype=np.float32).reshape(-1)[:2]))
                )
                probe_stage_ok = bool(
                    probe_stage == "RING_GRASP_ALIGN"
                    or args.c2c_grasp_probe_window_mode == "forced_shell"
                )
                probe_shell_ok = bool(
                    args.c2c_grasp_probe_shell_filter == "off"
                    or (
                        args.c2c_grasp_probe_shell_filter == "near_yaw_feasible"
                        and bool(probe_shell_fields.get("grasp_probe_near_basin_shell", False))
                    )
                    or (
                        args.c2c_grasp_probe_shell_filter == "coarse_yaw_feasible"
                        and (
                            bool(probe_shell_fields.get("grasp_probe_near_basin_shell", False))
                            or bool(probe_shell_fields.get("grasp_probe_coarse_pullback_candidate", False))
                        )
                    )
                )
                probe_eligible = bool(
                    args.c2c_grasp_probe_policy == "replay_oracle_xy"
                    and probe_stage_ok
                    and probe_visibility_bucket != "prior_only"
                    and probe_has_error
                    and probe_finite_xy
                    and probe_shell_ok
                )
                trace_entry["grasp_probe_policy"] = str(args.c2c_grasp_probe_policy)
                trace_entry["grasp_probe_visibility_bucket"] = probe_visibility_bucket
                trace_entry["grasp_probe_active"] = bool(probe_eligible)
                trace_entry["grasp_probe_window_mode"] = str(args.c2c_grasp_probe_window_mode)
                trace_entry["grasp_probe_shell_filter"] = str(args.c2c_grasp_probe_shell_filter)
                trace_entry["grasp_probe_stage_ok"] = bool(probe_stage_ok)
                trace_entry["grasp_probe_stage_source"] = "runtime_stage" if probe_stage == "RING_GRASP_ALIGN" else ("forced_shell" if probe_stage_ok else "not_grasp_align")
                trace_entry["grasp_probe_reason"] = "replay_oracle_xy" if probe_eligible else _grasp_probe_inactive_reason(
                    policy=str(args.c2c_grasp_probe_policy),
                    stage_ok=bool(probe_stage_ok),
                    visibility_bucket=probe_visibility_bucket,
                    has_error=bool(probe_has_error),
                    finite_xy=bool(probe_finite_xy),
                    shell_filter=str(args.c2c_grasp_probe_shell_filter),
                    shell_fields=probe_shell_fields,
                )
                trace_entry["grasp_probe_requested_horizon"] = int(max(1, int(args.c2c_grasp_probe_horizon)))
                trace_entry["grasp_probe_horizon_steps_executed"] = 0
                trace_entry["grasp_probe_close_locked"] = bool(probe_eligible)
                trace_entry["grasp_probe_flush_planner_queue_requested"] = bool(args.c2c_grasp_probe_flush_planner_queue)
                trace_entry["grasp_probe_queue_len_before"] = int(len(action_queue))
                trace_entry["grasp_probe_queue_len_after"] = int(len(action_queue))
                trace_entry["grasp_probe_queue_flushed"] = False
                trace_entry["grasp_probe_pre_true_error_t"] = _jsonable_value(
                    np.asarray(probe_true_error_before, dtype=np.float32).reshape(-1)[:4] if probe_true_error_before is not None else np.full((4,), np.nan, dtype=np.float32)
                )
                trace_entry.update(probe_shell_fields)
                if probe_eligible:
                    probe_correction_local = _bounded_xy_oracle_probe_step(
                        np.asarray(probe_true_error_before, dtype=np.float32),
                        xy_gain=float(args.c2c_grasp_probe_xy_gain),
                        max_xy_step=float(args.c2c_grasp_probe_max_xy_step),
                    )
                    current_local_command = world_delta_to_local(np.asarray(delta_action[:6], dtype=np.float32), np.asarray(obs.gripper_pose[3:7], dtype=np.float32)).astype(np.float32)
                    probe_local_command = current_local_command.copy()
                    probe_local_command[0] += float(probe_correction_local[0])
                    probe_local_command[1] += float(probe_correction_local[1])
                    probe_world_delta = local_delta_to_world(probe_local_command, np.asarray(obs.gripper_pose[3:7], dtype=np.float32)).astype(np.float32)
                    delta_action = delta_action.copy()
                    delta_action[:6] = probe_world_delta[:6]
                    delta_action[6] = 1.0
                    trace_entry["grasp_probe_applied_xy_step_local_6d"] = _jsonable_value(probe_correction_local)
                    trace_entry["grasp_probe_local_command_local_6d"] = _jsonable_value(probe_local_command)
                    trace_entry["grasp_probe_control_gate_axes"] = list(c2c.get_last_trace().get("basin_control_gate_axes", []))
                    trace_entry["grasp_probe_pullback_ready_axes"] = list(c2c.get_last_trace().get("basin_pullback_ready_axes", []))
                    if args.record_video and not probe_frame_saved and frames:
                        probe_path = probe_frame_dir / f"ep{ep_idx:03d}_probe_start_step{step_idx:03d}.png"
                        Image.fromarray(np.asarray(frames[-1], dtype=np.uint8)).save(probe_path)
                        trace_entry["grasp_probe_frame_path"] = str(probe_path)
                        probe_frame_saved = True
                else:
                    inactive_local_command = world_delta_to_local(np.asarray(delta_action[:6], dtype=np.float32), np.asarray(obs.gripper_pose[3:7], dtype=np.float32)).astype(np.float32)
                    trace_entry["grasp_probe_applied_xy_step_local_6d"] = _jsonable_value(np.zeros(6, dtype=np.float32))
                    trace_entry["grasp_probe_local_command_local_6d"] = _jsonable_value(inactive_local_command)
                    trace_entry["grasp_probe_control_gate_axes"] = list(c2c.get_last_trace().get("basin_control_gate_axes", []))
                    trace_entry["grasp_probe_pullback_ready_axes"] = list(c2c.get_last_trace().get("basin_pullback_ready_axes", []))

            abs_action = delta_to_absolute(delta_action, obs.gripper_pose)
            safety = c2c.safety if c2c is not None else None
            abs_action, workspace_violation = maybe_apply_workspace_filter(
                abs_action,
                safety,
                mode=args.workspace_clamp_mode,
                tolerance=args.workspace_clamp_tolerance,
            )
            if workspace_violation > 0.0:
                workspace_violation_count += 1
                workspace_violation_max = max(workspace_violation_max, float(workspace_violation))
            post_clip_world_delta = _absolute_to_world_delta(abs_action, obs.gripper_pose)
            trace_entry["pre_clip_action_world_6d"] = _jsonable_value(np.asarray(delta_action[:6], dtype=np.float32))
            trace_entry["post_clip_action_world_6d"] = _jsonable_value(post_clip_world_delta)
            trace_entry["pre_clip_action_absolute_6d"] = _jsonable_value(np.asarray(abs_action[:6], dtype=np.float32))
            trace_entry["commanded_action_world_8d"] = _jsonable_value(abs_action.astype(np.float32))

            executed_action = abs_action
            recovery_applied = False
            try:
                obs, reward, terminate = task.step(abs_action)
            except InvalidActionError as exc:
                trace_entry["invalid_action"] = True
                trace_entry["invalid_action_flag"] = True
                trace_entry["invalid_error"] = type(exc).__name__
                invalid_action_count += 1
                action_queue.clear()
                if c2c is not None and obs.gripper_pose is not None:
                    recovery_abs = c2c.build_invalid_action_recovery_absolute(
                        obs.gripper_pose,
                        obs.gripper_open,
                        force_reading=raw_force,
                        proprio=proprio,
                    )
                    recovery_abs, workspace_violation = maybe_apply_workspace_filter(
                        recovery_abs,
                        c2c.safety,
                        mode=args.workspace_clamp_mode,
                        tolerance=args.workspace_clamp_tolerance,
                    )
                    executed_action = recovery_abs
                    trace_entry["coarse2contact_invalid_action_recovery"] = True
                    trace_entry["coarse2contact_invalid_action_recovery_phase"] = str(c2c.get_last_trace().get("force_skill_state", "IDLE"))
                    trace_entry["coarse2contact_invalid_action_recovery_primitive"] = str(c2c.get_last_trace().get("local_correction_owner", "none"))
                    trace_entry["coarse2contact_invalid_action_recovery_reason"] = str(c2c.get_last_trace().get("phase_reason", "invalid_action"))
                    trace_entry["retry_id"] = int(c2c.get_last_trace().get("recovery_cycle_id", 0))
                    trace_entry["post_clip_action_absolute_6d"] = _jsonable_value(np.asarray(recovery_abs[:6], dtype=np.float32))
                    try:
                        obs, reward, terminate = task.step(recovery_abs)
                        recovery_applied = True
                    except InvalidActionError:
                        gripper_trace.append(trace_entry)
                        if args.dump_runtime_obs:
                            obs_row = {
                                "step": np.asarray(int(step_idx), dtype=np.int32),
                                "episode_idx": np.asarray(int(ep_idx), dtype=np.int32),
                                "episode_loop_idx": np.asarray(int(loop_idx), dtype=np.int32),
                                "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
                                "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
                                "wrist_depth": np.asarray(
                                    obs.wrist_depth if obs.wrist_depth is not None else np.zeros((96, 96), dtype=np.float32),
                                    dtype=np.float32,
                                ),
                                "gripper_pose": np.asarray(obs.gripper_pose, dtype=np.float32),
                                "gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
                                "proprio": np.asarray(proprio, dtype=np.float32),
                                "raw_wrench": np.asarray(raw_force if raw_force is not None else np.zeros(6, dtype=np.float32), dtype=np.float32),
                                "filtered_wrench": np.asarray(trace_entry.get("filtered_wrench", np.zeros(6, dtype=np.float32)), dtype=np.float32),
                                "planner_action_world_6d": np.asarray(base_delta_action[:6], dtype=np.float32),
                                "pre_clip_action_world_6d": np.asarray(trace_entry.get("pre_clip_action_world_6d", delta_action[:6]), dtype=np.float32),
                                "post_clip_action_world_6d": np.asarray(trace_entry.get("post_clip_action_world_6d", post_clip_world_delta), dtype=np.float32),
                                "pre_clip_action_absolute_6d": np.asarray(abs_action[:6], dtype=np.float32),
                                "post_clip_action_absolute_6d": np.asarray(recovery_abs[:6], dtype=np.float32),
                                "executed_action_world_6d": np.asarray(executed_action[:6], dtype=np.float32),
                                "invalid_action": np.asarray(float(trace_entry["invalid_action"]), dtype=np.float32),
                                "reward": np.asarray(float(reward), dtype=np.float32),
                                "terminate": np.asarray(float(terminate), dtype=np.float32),
                                "uses_privileged_runtime": np.asarray(0.0, dtype=np.float32),
                                "uses_privileged_label_for_eval": np.asarray(float(args.dump_runtime_obs and args.capture_failure_target_pose), dtype=np.float32),
                            }
                            if episode_target_pose_7d is not None:
                                obs_row["episode_target_pose_7d"] = np.asarray(episode_target_pose_7d, dtype=np.float32)
                            if privileged_frame_pack is not None:
                                for key, value in privileged_frame_pack.items():
                                    obs_row[key] = np.asarray(value, dtype=np.float32)
                            runtime_obs_rows.append(obs_row)
                        continue
                else:
                    gripper_trace.append(trace_entry)
                    if args.dump_runtime_obs:
                        obs_row = {
                            "step": np.asarray(int(step_idx), dtype=np.int32),
                            "episode_idx": np.asarray(int(ep_idx), dtype=np.int32),
                            "episode_loop_idx": np.asarray(int(loop_idx), dtype=np.int32),
                            "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
                            "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
                            "wrist_depth": np.asarray(
                                obs.wrist_depth if obs.wrist_depth is not None else np.zeros((96, 96), dtype=np.float32),
                                dtype=np.float32,
                            ),
                            "gripper_pose": np.asarray(obs.gripper_pose, dtype=np.float32),
                            "gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
                            "proprio": np.asarray(proprio, dtype=np.float32),
                            "raw_wrench": np.asarray(raw_force if raw_force is not None else np.zeros(6, dtype=np.float32), dtype=np.float32),
                            "filtered_wrench": np.asarray(trace_entry.get("filtered_wrench", np.zeros(6, dtype=np.float32)), dtype=np.float32),
                            "planner_action_world_6d": np.asarray(base_delta_action[:6], dtype=np.float32),
                            "pre_clip_action_world_6d": np.asarray(trace_entry.get("pre_clip_action_world_6d", delta_action[:6]), dtype=np.float32),
                            "post_clip_action_world_6d": np.asarray(trace_entry.get("post_clip_action_world_6d", post_clip_world_delta), dtype=np.float32),
                            "pre_clip_action_absolute_6d": np.asarray(abs_action[:6], dtype=np.float32),
                            "post_clip_action_absolute_6d": np.asarray(abs_action[:6], dtype=np.float32),
                            "executed_action_world_6d": np.asarray(executed_action[:6], dtype=np.float32),
                            "invalid_action": np.asarray(float(trace_entry["invalid_action"]), dtype=np.float32),
                            "reward": np.asarray(float(reward), dtype=np.float32),
                            "terminate": np.asarray(float(terminate), dtype=np.float32),
                            "uses_privileged_runtime": np.asarray(0.0, dtype=np.float32),
                            "uses_privileged_label_for_eval": np.asarray(float(args.dump_runtime_obs and args.capture_failure_target_pose), dtype=np.float32),
                        }
                        if episode_target_pose_7d is not None:
                            obs_row["episode_target_pose_7d"] = np.asarray(episode_target_pose_7d, dtype=np.float32)
                        if privileged_frame_pack is not None:
                            for key, value in privileged_frame_pack.items():
                                obs_row[key] = np.asarray(value, dtype=np.float32)
                        runtime_obs_rows.append(obs_row)
                    continue

            trace_entry["reward"] = float(reward)
            trace_entry["terminate"] = bool(terminate)
            if (
                bool(trace_entry.get("grasp_probe_active", False))
                and int(trace_entry.get("grasp_probe_requested_horizon", 1) or 1) > 1
                and bool(args.c2c_grasp_probe_flush_planner_queue)
            ):
                before_flush = int(len(action_queue))
                action_queue.clear()
                trace_entry["grasp_probe_queue_len_before"] = int(before_flush)
                trace_entry["grasp_probe_queue_len_after"] = int(len(action_queue))
                trace_entry["grasp_probe_queue_flushed"] = bool(before_flush > 0)
            if c2c is not None:
                last_trace = c2c.get_last_trace()
                trace_entry["depth_conf"] = float(last_trace.get("localizer_confidence", 0.0))
                local_geometry = last_trace.get("local_geometry_error", {})
                if isinstance(local_geometry, dict):
                    active_skill_type = str(last_trace.get("c2c_v2_skill_type", "none"))
                    key = "grasp" if active_skill_type == "precision_grasp" else "spoke"
                    trace_entry["depth_obs_quality"] = float(local_geometry.get(key, {}).get("observability", 0.0))
                else:
                    trace_entry["depth_obs_quality"] = 0.0
                trace_entry["phase_owner"] = str(last_trace.get("phase_owner", "planner"))
                trace_entry["phase_reason"] = str(last_trace.get("phase_reason", "planner"))
                trace_entry["retry_id"] = int(last_trace.get("recovery_cycle_id", 0))
                trace_entry["invalid_action_flag"] = bool(last_trace.get("invalid_action_flag", trace_entry["invalid_action"]))
                trace_entry["planner_reaches_precontact"] = bool(last_trace.get("planner_reaches_precontact", trace_entry["planner_reaches_precontact"]))
                trace_entry["planner_reaches_preinsert"] = bool(last_trace.get("planner_reaches_preinsert", trace_entry["planner_reaches_preinsert"]))
                trace_entry["pre_clip_action_world_6d"] = _jsonable_value(np.asarray(last_trace.get("pre_clip_action_world_6d", delta_action[:6]), dtype=np.float32))
                trace_entry["post_clip_action_world_6d"] = _jsonable_value(np.asarray(last_trace.get("post_clip_action_world_6d", post_clip_world_delta), dtype=np.float32))
                trace_entry["pre_clip_action_absolute_6d"] = _jsonable_value(np.asarray(abs_action[:6], dtype=np.float32))
                trace_entry["post_clip_action_absolute_6d"] = _jsonable_value(np.asarray(abs_action[:6], dtype=np.float32))
                trace_entry["executed_action_world_6d"] = _jsonable_value(np.asarray(executed_action[:6], dtype=np.float32))
                trace_entry["executed_action_world_8d"] = _jsonable_value(np.asarray(executed_action, dtype=np.float32))
                trace_entry["planner_action_world"] = _jsonable_value(np.asarray(base_delta_action[:6], dtype=np.float32))
                trace_entry["raw_wrench"] = _jsonable_value(np.asarray(last_trace.get("raw_wrench", raw_force if raw_force is not None else np.zeros(6, dtype=np.float32)), dtype=np.float32))
                trace_entry["filtered_wrench"] = _jsonable_value(np.asarray(last_trace.get("filtered_wrench", np.zeros(6, dtype=np.float32)), dtype=np.float32))
                trace_entry["c2c_v2_stage"] = str(last_trace.get("c2c_v2_stage", "planner_only"))
                trace_entry["c2c_v2_skill_type"] = str(last_trace.get("c2c_v2_skill_type", "none"))
                trace_entry["c2c_v2_owner"] = str(last_trace.get("c2c_v2_owner", "planner"))
                trace_entry["c2c_v2_controlled_dofs"] = list(last_trace.get("c2c_v2_controlled_dofs", []))
                trace_entry["c2c_v2_target_entity"] = str(last_trace.get("c2c_v2_target_entity", "none"))
                trace_entry["c2c_v2_reference_entity"] = str(last_trace.get("c2c_v2_reference_entity", "none"))
                trace_entry["local_geometry_error"] = _jsonable_value(last_trace.get("local_geometry_error", {}))
                trace_entry["estimated_basin_error"] = _jsonable_value(last_trace.get("estimated_basin_error", {}))
                trace_entry["localizer_abstained"] = bool(last_trace.get("localizer_abstained", True))
                trace_entry["force_skill_state"] = str(last_trace.get("force_skill_state", "planner"))
                trace_entry["recovery_cycle_id"] = int(last_trace.get("recovery_cycle_id", 0))
                trace_entry["basin_pullback_variant"] = str(last_trace.get("basin_pullback_variant", args.basin_pullback_variant))
                trace_entry["basin_visual_gain"] = float(last_trace.get("basin_visual_gain", args.basin_visual_gain))
                trace_entry["basin_max_pullback_xy_step"] = float(last_trace.get("basin_max_pullback_xy_step", args.basin_max_pullback_xy_step))
                trace_entry["basin_max_recovery_steps"] = int(last_trace.get("basin_max_recovery_steps", args.basin_max_recovery_steps))
                trace_entry["basin_axis_validity"] = _jsonable_value(last_trace.get("basin_axis_validity", {}))
                trace_entry["basin_axis_confidence"] = _jsonable_value(last_trace.get("basin_axis_confidence", {}))
                trace_entry["basin_axis_policy"] = _jsonable_value(last_trace.get("basin_axis_policy", {}))
                trace_entry["basin_axis_source"] = str(last_trace.get("basin_axis_source", "none"))
                trace_entry["basin_frame_consistency"] = float(last_trace.get("basin_frame_consistency", 0.0))
                trace_entry["basin_close_ready"] = bool(last_trace.get("basin_close_ready", False))
                trace_entry["uses_privileged_target"] = False
                trace_entry["uses_rlbench_mask_runtime"] = False
                trace_entry["grasp_gripper_override"] = last_trace.get("grasp_gripper_override", None)
                trace_entry["slide_gripper_override"] = last_trace.get("slide_gripper_override", None)
                trace_entry["c2c_stage_age"] = int(last_trace.get("c2c_stage_age", 0))
                trace_entry["uses_privileged_runtime"] = False
                trace_entry["uses_privileged_label_for_eval"] = bool(args.dump_runtime_obs and args.capture_failure_target_pose)
            else:
                trace_entry["pre_clip_action_absolute_6d"] = _jsonable_value(np.asarray(abs_action[:6], dtype=np.float32))
                trace_entry["post_clip_action_world_6d"] = _jsonable_value(post_clip_world_delta)
                trace_entry["post_clip_action_absolute_6d"] = _jsonable_value(np.asarray(abs_action[:6], dtype=np.float32))
                trace_entry["executed_action_world_6d"] = _jsonable_value(np.asarray(executed_action[:6], dtype=np.float32))
                trace_entry["planner_action_world"] = _jsonable_value(np.asarray(base_delta_action[:6], dtype=np.float32))
                trace_entry["uses_privileged_runtime"] = False
                trace_entry["uses_privileged_label_for_eval"] = bool(args.dump_runtime_obs and args.capture_failure_target_pose)
            trace_entry["executed_action_world_8d"] = _jsonable_value(np.asarray(executed_action, dtype=np.float32))
            trace_entry["invalid_action_recovery_executed"] = bool(recovery_applied)
            trace_entry["final_action_world_6d"] = _jsonable_value(np.asarray(delta_action[:6], dtype=np.float32))
            if args.c2c_grasp_probe_policy != "off" and bool(trace_entry.get("grasp_probe_active", False)) and args.dump_runtime_obs and args.capture_failure_target_pose:
                probe_after_pack = _episode_privileged_frame_pack(task, obs)
                probe_true_error_after = _grasp_teacher_error_from_pack(probe_after_pack, grasp_spec)
                trace_entry["grasp_probe_horizon_records"] = []
                trace_entry["grasp_probe_horizon_stage_sequence"] = []
                trace_entry["grasp_probe_horizon_owner_sequence"] = []
                if probe_true_error_after is not None and probe_true_error_before is not None:
                    pre_probe = np.asarray(probe_true_error_before, dtype=np.float32).reshape(-1)[:4]
                    post_probe = np.asarray(probe_true_error_after, dtype=np.float32).reshape(-1)[:4]
                    visibility_bucket = str(trace_entry.get("grasp_probe_visibility_bucket", "prior_only"))
                    horizon_stage = str(trace_entry.get("c2c_v2_stage", "unknown"))
                    horizon_owner = str(trace_entry.get("phase_owner", trace_entry.get("c2c_v2_owner", "unknown")))
                    metric_fields = _grasp_probe_metric_fields(
                        pre_probe,
                        post_probe,
                        visibility_bucket=visibility_bucket,
                        near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
                        near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
                        close_ready_xy_threshold=float(args.close_ready_xy_threshold),
                        close_ready_yaw_threshold=float(args.close_ready_yaw_threshold),
                    )
                    trace_entry["grasp_probe_post_true_error_t"] = _jsonable_value(post_probe)
                    trace_entry.update(metric_fields)
                    trace_entry["grasp_probe_horizon_steps_executed"] = 1
                    trace_entry["grasp_probe_horizon_stage_sequence"].append(horizon_stage)
                    trace_entry["grasp_probe_horizon_owner_sequence"].append(horizon_owner)
                    trace_entry["grasp_probe_horizon_records"].append(
                        {
                            "horizon_step": 1,
                            "stage": horizon_stage,
                            "owner": horizon_owner,
                            "pre_true_error_t": _jsonable_value(pre_probe),
                            "post_true_error_t": _jsonable_value(post_probe),
                            "applied_xy_step_local_6d": trace_entry.get("grasp_probe_applied_xy_step_local_6d", _jsonable_value(np.zeros(6, dtype=np.float32))),
                            "xy_contracted": bool(metric_fields["grasp_probe_xy_contracted"]),
                            "near_grasp_after": bool(metric_fields["grasp_probe_near_grasp_after"]),
                            "micro_entry_ready_after": bool(metric_fields["grasp_probe_micro_entry_ready_after"]),
                            "overshoot": bool(metric_fields["grasp_probe_overshoot"]),
                        }
                    )

                    final_probe = post_probe.copy()
                    requested_horizon = int(max(1, int(args.c2c_grasp_probe_horizon)))
                    while (
                        int(trace_entry["grasp_probe_horizon_steps_executed"]) < requested_horizon
                        and not bool(terminate)
                    ):
                        latest_pack = _episode_privileged_frame_pack(task, obs)
                        latest_error = _grasp_teacher_error_from_pack(latest_pack, grasp_spec)
                        if latest_error is None:
                            break
                        step_pre = np.asarray(latest_error, dtype=np.float32).reshape(-1)[:4]
                        if not np.all(np.isfinite(step_pre[:2])):
                            break
                        step_correction_local = _bounded_xy_oracle_probe_step(
                            step_pre,
                            xy_gain=float(args.c2c_grasp_probe_xy_gain),
                            max_xy_step=float(args.c2c_grasp_probe_max_xy_step),
                        )
                        step_local_command = np.zeros(6, dtype=np.float32)
                        step_local_command[0] = float(step_correction_local[0])
                        step_local_command[1] = float(step_correction_local[1])
                        step_world_delta = local_delta_to_world(step_local_command, np.asarray(obs.gripper_pose[3:7], dtype=np.float32)).astype(np.float32)
                        extra_delta = np.zeros_like(delta_action, dtype=np.float32)
                        extra_delta[:6] = step_world_delta[:6]
                        if extra_delta.shape[0] > 6:
                            extra_delta[6] = 1.0
                        extra_abs = delta_to_absolute(extra_delta, obs.gripper_pose)
                        extra_abs, extra_workspace_violation = maybe_apply_workspace_filter(
                            extra_abs,
                            safety,
                            mode=args.workspace_clamp_mode,
                            tolerance=args.workspace_clamp_tolerance,
                        )
                        if extra_workspace_violation > 0.0:
                            workspace_violation_count += 1
                            workspace_violation_max = max(workspace_violation_max, float(extra_workspace_violation))
                        try:
                            obs, reward, terminate = task.step(extra_abs)
                            executed_action = extra_abs
                        except InvalidActionError as exc:
                            trace_entry["grasp_probe_horizon_invalid_action"] = type(exc).__name__
                            trace_entry["invalid_action"] = True
                            trace_entry["invalid_action_flag"] = True
                            invalid_action_count += 1
                            break
                        if args.record_video:
                            frames.append(_compose_video_frame(obs.front_rgb.copy(), obs.wrist_rgb.copy() if obs.wrist_rgb is not None else None, layout=args.video_layout))
                        step_after_pack = _episode_privileged_frame_pack(task, obs)
                        step_after_error = _grasp_teacher_error_from_pack(step_after_pack, grasp_spec)
                        if step_after_error is None:
                            break
                        step_post = np.asarray(step_after_error, dtype=np.float32).reshape(-1)[:4]
                        step_metrics = _grasp_probe_metric_fields(
                            step_pre,
                            step_post,
                            visibility_bucket=visibility_bucket,
                            near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
                            near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
                            close_ready_xy_threshold=float(args.close_ready_xy_threshold),
                            close_ready_yaw_threshold=float(args.close_ready_yaw_threshold),
                        )
                        trace_entry["grasp_probe_horizon_steps_executed"] = int(trace_entry["grasp_probe_horizon_steps_executed"]) + 1
                        trace_entry["grasp_probe_horizon_stage_sequence"].append(horizon_stage)
                        trace_entry["grasp_probe_horizon_owner_sequence"].append(horizon_owner)
                        trace_entry["grasp_probe_horizon_records"].append(
                            {
                                "horizon_step": int(trace_entry["grasp_probe_horizon_steps_executed"]),
                                "stage": horizon_stage,
                                "owner": horizon_owner,
                                "pre_true_error_t": _jsonable_value(step_pre),
                                "post_true_error_t": _jsonable_value(step_post),
                                "applied_xy_step_local_6d": _jsonable_value(step_correction_local),
                                "xy_contracted": bool(step_metrics["grasp_probe_xy_contracted"]),
                                "near_grasp_after": bool(step_metrics["grasp_probe_near_grasp_after"]),
                                "micro_entry_ready_after": bool(step_metrics["grasp_probe_micro_entry_ready_after"]),
                                "overshoot": bool(step_metrics["grasp_probe_overshoot"]),
                            }
                        )
                        final_probe = step_post.copy()

                    trace_entry["grasp_probe_horizon_final_true_error_t"] = _jsonable_value(final_probe)
                    trace_entry["grasp_probe_queue_len_after"] = int(len(action_queue))
                    trace_entry.update(
                        _prefix_grasp_probe_fields(
                            _grasp_probe_metric_fields(
                                pre_probe,
                                final_probe,
                                visibility_bucket=visibility_bucket,
                                near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
                                near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
                                close_ready_xy_threshold=float(args.close_ready_xy_threshold),
                                close_ready_yaw_threshold=float(args.close_ready_yaw_threshold),
                            ),
                            "grasp_probe_horizon",
                        )
                    )
                else:
                    nan_error = np.full((4,), np.nan, dtype=np.float32)
                    trace_entry["grasp_probe_post_true_error_t"] = _jsonable_value(nan_error)
                    trace_entry["grasp_probe_horizon_final_true_error_t"] = _jsonable_value(nan_error)
                    trace_entry.update(_nan_grasp_probe_metric_fields())
                    trace_entry.update(_prefix_grasp_probe_fields(_nan_grasp_probe_metric_fields(), "grasp_probe_horizon"))
            if privileged_frame_pack is not None:
                trace_entry.update({k: _jsonable_value(v) for k, v in privileged_frame_pack.items()})
            gripper_trace.append(trace_entry)
            if args.dump_runtime_obs:
                obs_row = {
                    "step": np.asarray(int(step_idx), dtype=np.int32),
                    "episode_idx": np.asarray(int(ep_idx), dtype=np.int32),
                    "episode_loop_idx": np.asarray(int(loop_idx), dtype=np.int32),
                    "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
                    "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
                    "wrist_depth": np.asarray(
                        obs.wrist_depth if obs.wrist_depth is not None else np.zeros((96, 96), dtype=np.float32),
                        dtype=np.float32,
                    ),
                    "gripper_pose": np.asarray(obs.gripper_pose, dtype=np.float32),
                    "gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
                    "proprio": np.asarray(proprio, dtype=np.float32),
                    "raw_wrench": np.asarray(raw_force if raw_force is not None else np.zeros(6, dtype=np.float32), dtype=np.float32),
                    "filtered_wrench": np.asarray(trace_entry.get("filtered_wrench", np.zeros(6, dtype=np.float32)), dtype=np.float32),
                    "planner_action_world_6d": np.asarray(base_delta_action[:6], dtype=np.float32),
                    "pre_clip_action_world_6d": np.asarray(trace_entry.get("pre_clip_action_world_6d", delta_action[:6]), dtype=np.float32),
                    "post_clip_action_world_6d": np.asarray(trace_entry.get("post_clip_action_world_6d", delta_action[:6]), dtype=np.float32),
                    "pre_clip_action_absolute_6d": np.asarray(abs_action[:6], dtype=np.float32),
                    "post_clip_action_absolute_6d": np.asarray(trace_entry.get("post_clip_action_absolute_6d", abs_action[:6]), dtype=np.float32),
                    "executed_action_world_6d": np.asarray(executed_action[:6], dtype=np.float32),
                    "invalid_action": np.asarray(float(trace_entry["invalid_action"]), dtype=np.float32),
                    "reward": np.asarray(float(reward), dtype=np.float32),
                    "terminate": np.asarray(float(terminate), dtype=np.float32),
                    "uses_privileged_runtime": np.asarray(0.0, dtype=np.float32),
                    "uses_privileged_label_for_eval": np.asarray(float(args.dump_runtime_obs and args.capture_failure_target_pose), dtype=np.float32),
                }
                if episode_target_pose_7d is not None:
                    obs_row["episode_target_pose_7d"] = np.asarray(episode_target_pose_7d, dtype=np.float32)
                if privileged_frame_pack is not None:
                    for key, value in privileged_frame_pack.items():
                        obs_row[key] = np.asarray(value, dtype=np.float32)
                runtime_obs_rows.append(obs_row)

            if reward > 0.0:
                success = True
            if terminate:
                break

        episode_mp4_path = None
        if args.record_video and args.write_episode_videos and len(frames) > 1:
            status = "succ" if success else "fail"
            episode_mp4_path = str(video_dir / f"ep{ep_idx:03d}_{status}.mp4")
            clip = ImageSequenceClip(frames, fps=20)
            clip.write_videofile(episode_mp4_path, fps=20, codec="libx264", bitrate="3000k", logger=None)
            results["video_paths"].append(episode_mp4_path)
            for item in gripper_trace:
                item["mp4_path"] = episode_mp4_path

        if args.record_gripper_trace:
            trace_path = trace_dir / f"ep{ep_idx:03d}_gripper_trace.jsonl"
            with open(trace_path, "w") as handle:
                for item in gripper_trace:
                    handle.write(json.dumps(item, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            results["gripper_trace_paths"].append(str(trace_path))

        runtime_obs_path = None
        if args.dump_runtime_obs and (args.dump_runtime_obs_all_episodes or not success):
            runtime_obs_path = runtime_obs_dir / f"ep{ep_idx:03d}_runtime_obs.npz"
            runtime_npz = _stack_runtime_obs_rows(runtime_obs_rows)
            if runtime_npz:
                np.savez_compressed(runtime_obs_path, **runtime_npz)
                results["runtime_obs_paths"].append(str(runtime_obs_path))

        if c2c is not None:
            stage_counter = {
                "c2c_v2_stage": str(c2c.get_last_trace().get("c2c_v2_stage", "planner_only")),
                "c2c_v2_owner": str(c2c.get_last_trace().get("c2c_v2_owner", "planner")),
                "c2c_v2_skill_type": str(c2c.get_last_trace().get("c2c_v2_skill_type", "none")),
            }
        else:
            stage_counter = {"c2c_v2_stage": "planner_only", "c2c_v2_owner": "planner", "c2c_v2_skill_type": "none"}
        episode_precontact = any(bool(item.get("planner_reaches_precontact", False)) for item in gripper_trace)
        episode_preinsert = any(bool(item.get("planner_reaches_preinsert", False)) for item in gripper_trace)
        depth_error_norms = []
        for item in gripper_trace:
            lg = item.get("local_geometry_error", {})
            active_key = "grasp" if item.get("c2c_v2_skill_type") == "precision_grasp" else "spoke"
            geom = lg.get(active_key, {}) if isinstance(lg, dict) else {}
            if geom:
                dx = float(geom.get("dx", 0.0) or 0.0)
                dy = float(geom.get("dy", 0.0) or 0.0)
                dz = float(geom.get("dz", 0.0) or 0.0)
                dyaw = float(geom.get("dyaw", 0.0) or 0.0)
                depth_error_norms.append(float(np.sqrt(dx * dx + dy * dy + dz * dz + 0.25 * dyaw * dyaw)))
        if depth_error_norms:
            trend = {
                "episode_index": int(ep_idx),
                "start_error": float(depth_error_norms[0]),
                "end_error": float(depth_error_norms[-1]),
                "min_error": float(min(depth_error_norms)),
                "mean_error": float(np.mean(depth_error_norms)),
                "nonincreasing_rate": float(np.mean([1.0 if depth_error_norms[i] <= depth_error_norms[i - 1] + 1e-6 else 0.0 for i in range(1, len(depth_error_norms))])) if len(depth_error_norms) > 1 else 1.0,
            }
        else:
            trend = {
                "episode_index": int(ep_idx),
                "start_error": 0.0,
                "end_error": 0.0,
                "min_error": 0.0,
                "mean_error": 0.0,
                "nonincreasing_rate": 0.0,
            }

        stage_stat = {
            "episode_index": int(ep_idx),
            "success": bool(success),
            "reward": float(reward),
            "episode_length": int(len(gripper_trace)),
            "invalid_action_count": int(invalid_action_count),
            "workspace_violation_count": int(workspace_violation_count),
            "workspace_violation_max": float(workspace_violation_max),
            "coarse2contact_v2_stage": stage_counter["c2c_v2_stage"],
            "coarse2contact_v2_owner": stage_counter["c2c_v2_owner"],
            "coarse2contact_v2_skill_type": stage_counter["c2c_v2_skill_type"],
            "uses_privileged_target": False,
            "uses_rlbench_mask_runtime": False,
            "uses_privileged_label_for_eval": bool(args.dump_runtime_obs and args.capture_failure_target_pose),
            "mp4_path": episode_mp4_path,
            "runtime_obs_path": str(runtime_obs_path) if runtime_obs_path is not None else None,
            "c2c_stage_shadow": bool(args.mode == "c2c_stage_shadow" or args.shadow_only),
            "planner_reaches_precontact": bool(episode_precontact),
            "planner_reaches_preinsert": bool(episode_preinsert),
            "depth_error_trend": trend,
            "c2c_gate_frame_path": next((str(item.get("c2c_gate_frame_path")) for item in gripper_trace if item.get("c2c_gate_frame_path")), None),
            "basin_pullback_variant": str(args.basin_pullback_variant),
            "basin_visual_gain": float(args.basin_visual_gain),
            "basin_max_pullback_xy_step": float(args.basin_max_pullback_xy_step),
            "basin_max_recovery_steps": int(args.basin_max_recovery_steps),
        }
        results["successes"].append(bool(success))
        results["episode_lengths"].append(int(len(gripper_trace)))
        results["invalid_action_counts"].append(int(invalid_action_count))
        results["stage_stats"].append(stage_stat)
        results["depth_error_trend"].append(trend)

    results["success_rate"] = float(np.mean(results["successes"])) if results["successes"] else 0.0
    results["avg_episode_length"] = float(np.mean(results["episode_lengths"])) if results["episode_lengths"] else 0.0
    results["planner_reaches_precontact_count"] = int(sum(int(bool(s.get("planner_reaches_precontact", False))) for s in results["stage_stats"]))
    results["planner_reaches_preinsert_count"] = int(sum(int(bool(s.get("planner_reaches_preinsert", False))) for s in results["stage_stats"]))
    results["coarse2contact_invalid_action_count"] = int(sum(int(s.get("invalid_action_count", 0)) for s in results["stage_stats"]))
    if results["depth_error_trend"]:
        results["depth_error_nonincreasing_rate"] = float(np.mean([float(t["nonincreasing_rate"]) for t in results["depth_error_trend"]]))
        results["depth_error_start_mean"] = float(np.mean([float(t["start_error"]) for t in results["depth_error_trend"]]))
        results["depth_error_end_mean"] = float(np.mean([float(t["end_error"]) for t in results["depth_error_trend"]]))

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as handle:
        json.dump(_jsonable_value(results), handle, indent=2)
    print(f"\n[c2c-v2] Saved results to {results_path}", flush=True)
    env.shutdown()
    return float(results["success_rate"])


def main() -> int:
    args = parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
