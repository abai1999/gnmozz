#!/usr/bin/env python
"""Standalone RLBench evaluator for Coarse2Contact v2."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import replace
from collections import deque
from pathlib import Path
from typing import Any, Mapping

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
from prismatic.robot.coarse2contact_v2.grasp_probe_execution import (
    candidate_within_xy_activation_window,
    candidate_xy_correction_ready,
    grasp_probe_close_arbiter_decision,
    grasp_probe_close_ready_with_z,
    planner_gripper_authority_decision,
    precision_takeover_activation_status,
    smooth_grasp_probe_xy_step,
)
from prismatic.robot.coarse2contact_v2.alignment_takeover import (
    AlignmentTakeoverConfig,
    AlignmentTakeoverSession,
    TaskFrameResidualEstimate,
    evaluate_alignment_readiness,
)
from prismatic.robot.coarse2contact_v2.task_frame_readiness import (
    TaskFrameReadinessNet,
    TaskFrameYawReadinessEstimate,
    TaskFrameZReadinessEstimate,
    load_task_frame_readiness_checkpoint,
    task_frame_readiness_feature_vector,
)
from prismatic.robot.coarse2contact_v2.task_frame_v45_candidate import (
    TaskFrameV45CandidateNet,
    TaskFrameV45CandidateEstimate,
    TaskFrameV45MicroServoDecision,
    load_task_frame_v45_candidate_checkpoint,
    task_frame_v45_candidate_feature_vector,
    task_frame_v45_micro_servo_step,
)
from prismatic.robot.coarse2contact_v2.recovery_audit import in_close_ready_basin, in_near_grasp_basin, recovery_overshoot_flag
from prismatic.robot.coarse2contact_v2.runtime_xy_residual import (
    RUNTIME_XY_SOFT_ACTIVATION_RADIUS,
    RuntimeXYAffineCalibration,
    calibrated_runtime_xy_residual_from_trace,
)

from prismatic.robot.coarse2contact_v2 import BasinRecoveryConfig, PrecisionSkillSupervisor, load_precision_task_spec, load_basin_state_calibration_report
from prismatic.robot.coarse2contact_v2.learned_force import LearnedForceClassifierAdapter
from prismatic.robot.coarse2contact_v2.learned_localizer import LearnedDepthLocalizerAdapter
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from prismatic.vla.constants import FORCE_HISTORY_LEN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _select_task_spec(task_name: str):
    return load_precision_task_spec(task_name)


def _maybe_attach_basin_state_calibration(
    task_spec,
    calibration_report: str | None,
    *,
    enable_runtime_xy_pullback_calibration: bool = False,
):
    if task_spec is None:
        return None
    if calibration_report:
        calibration = load_basin_state_calibration_report(calibration_report)
        if calibration is not None:
            if bool(enable_runtime_xy_pullback_calibration):
                calibration = calibration.with_xy_pullback_trusted()
                task_spec.runtime_flags["runtime_xy_pullback_calibration"] = True
            task_spec.runtime_flags["basin_state_calibration"] = calibration.to_dict()
            task_spec.runtime_flags["basin_state_calibration_report"] = str(calibration_report)
            return task_spec
    return task_spec


def _mode_to_flags(mode: str) -> tuple[bool, bool]:
    if mode == "planner_only":
        return False, False
    return mode in {"c2c_stage_shadow", "grasp_depth_apply", "spoke_depth_apply", "force_recovery", "full_owner_by_stage", "basin_recovery_shadow", "basin_recovery_only"}, mode in {"c2c_stage_shadow", "basin_recovery_shadow"}


def _candidate_alias_drift_decision(row: Mapping[str, Any]) -> str:
    decision = row.get("alias_drift_decision", None)
    if decision is None or str(decision).strip() in {"", "None"}:
        decision = row.get("yaw_alias_drift_decision", None)
    decision = str(decision if decision is not None else "unknown").strip()
    if decision in {"stable_alias_control", "frame_drift_abstain", "unknown"}:
        return decision
    label = str(row.get("alias_label", "")).strip()
    role = str(row.get("acceptance_role", "")).strip()
    if role == "calibration_positive" or label == "stable_alias":
        return "stable_alias_control"
    if role == "frame_drift_hard_case" or label == "frame_drift":
        return "frame_drift_abstain"
    return "unknown"


def _runtime_probe_visibility_bucket(trace_entry: Mapping[str, Any]) -> str:
    explicit = trace_entry.get("basin_recovery_visual_evidence_class", trace_entry.get("visual_observability_class", None))
    if explicit is not None and str(explicit) and str(explicit) != "prior_only":
        return str(explicit)
    est = trace_entry.get("estimated_basin_error", {})
    if isinstance(est, Mapping):
        est_valid = bool(est.get("estimated_basin_error_valid", est.get("valid", False)))
        x_valid = bool(est.get("estimated_basin_error_x_valid", est.get("x_valid", False)))
        y_valid = bool(est.get("estimated_basin_error_y_valid", est.get("y_valid", False)))
        reason = str(est.get("estimated_basin_error_reason", est.get("reason", "")))
        if est_valid and (x_valid or y_valid) and reason not in {"prior_only", "prior_only_reacquire", "reacquire_needed"}:
            return "runtime_estimated"
    local_geometry = trace_entry.get("local_geometry_error", {})
    grasp = local_geometry.get("grasp", {}) if isinstance(local_geometry, Mapping) else {}
    if isinstance(grasp, Mapping) and bool(grasp.get("valid", False)):
        return "visual_observable_proxy"
    return str(explicit or "prior_only")


def _offline_eval_only_pack(privileged_frame_pack: Mapping[str, Any] | None) -> dict[str, Any]:
    if privileged_frame_pack is None:
        return {}
    return {
        "schema_version": "c2c_v2_offline_eval_only_v1",
        "uses_privileged_label_for_eval": True,
        "privileged_frame_pack": {str(k): _jsonable_value(v) for k, v in privileged_frame_pack.items()},
    }


def _attach_offline_eval_only(trace_entry: dict[str, Any], privileged_frame_pack: Mapping[str, Any] | None) -> None:
    if privileged_frame_pack is None:
        return
    trace_entry["offline_eval_only"] = _offline_eval_only_pack(privileged_frame_pack)


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


def _bounded_xy_estimator_probe_step(dx: float, dy: float, *, xy_gain: float, max_xy_step: float) -> np.ndarray:
    correction = np.zeros(6, dtype=np.float32)
    vec = np.asarray([float(dx), float(dy)], dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        return correction
    correction[:2] = float(xy_gain) * vec
    norm = float(np.linalg.norm(correction[:2]))
    if norm > float(max_xy_step) > 0.0:
        correction[:2] = correction[:2] * (float(max_xy_step) / max(norm, 1.0e-9))
    return correction.astype(np.float32)


def _compact_grasp_error(error_local: np.ndarray | None) -> np.ndarray:
    """Return [dx, dy, dz, dyaw] from either compact 4D or local 6D residuals."""

    if error_local is None:
        return np.full((4,), np.nan, dtype=np.float32)
    raw = np.asarray(error_local, dtype=np.float32).reshape(-1)
    out = np.full((4,), np.nan, dtype=np.float32)
    if raw.size > 0:
        out[0] = raw[0]
    if raw.size > 1:
        out[1] = raw[1]
    if raw.size > 2:
        out[2] = raw[2]
    yaw_idx = 5 if raw.size >= 6 else 3
    if raw.size > yaw_idx:
        out[3] = raw[yaw_idx]
    return out


def _estimated_basin_trace_value(trace_row: Mapping[str, object], name: str, default: float = 0.0) -> float:
    est = trace_row.get("estimated_basin_error", {})
    if isinstance(est, Mapping):
        value = est.get(f"estimated_basin_error_{name}", est.get(name, default))
    else:
        value = default
    try:
        return float(value)
    except Exception:
        return float(default)


def _estimated_basin_trace_bool(trace_row: Mapping[str, object], name: str, default: bool = False) -> bool:
    est = trace_row.get("estimated_basin_error", {})
    if isinstance(est, Mapping):
        value = est.get(f"estimated_basin_error_{name}", est.get(name, default))
    else:
        value = default
    if isinstance(value, bool):
        return bool(value)
    try:
        return bool(float(value) > 0.5)
    except Exception:
        return bool(default)


def _task_frame_residual_from_runtime_trace(
    trace_row: Mapping[str, object],
    runtime_xy_estimate,
) -> TaskFrameResidualEstimate:
    active_skill = str(trace_row.get("c2c_v2_skill_type", "precision_grasp"))
    stage_name = str(trace_row.get("c2c_v2_stage", ""))
    if active_skill == "precision_align":
        reference_frame = "ring_aperture_frame"
        target_frame = "target_spoke_axis_frame"
    else:
        reference_frame = "gripper_jaw_frame"
        target_frame = "ring_grasp_frame"
    z_valid = _estimated_basin_trace_bool(trace_row, "z_valid", False)
    yaw_valid = _estimated_basin_trace_bool(trace_row, "yaw_valid", False)
    return TaskFrameResidualEstimate(
        skill_id=active_skill,
        stage_name=stage_name,
        reference_frame=reference_frame,
        target_frame=target_frame,
        active_dofs=("x", "y", "z", "yaw"),
        dx=float(runtime_xy_estimate.dx) if bool(runtime_xy_estimate.entry_ready) else _estimated_basin_trace_value(trace_row, "dx", 0.0),
        dy=float(runtime_xy_estimate.dy) if bool(runtime_xy_estimate.entry_ready) else _estimated_basin_trace_value(trace_row, "dy", 0.0),
        dz=_estimated_basin_trace_value(trace_row, "dz", 0.0),
        dyaw=_estimated_basin_trace_value(trace_row, "dyaw", 0.0),
        axis_validity={
            "x": bool(runtime_xy_estimate.entry_ready),
            "y": bool(runtime_xy_estimate.entry_ready),
            "z": bool(z_valid),
            "yaw": bool(yaw_valid),
        },
        axis_confidence={
            "x": float(runtime_xy_estimate.confidence),
            "y": float(runtime_xy_estimate.confidence),
            "z": _estimated_basin_trace_value(trace_row, "z_confidence", 0.0),
            "yaw": _estimated_basin_trace_value(trace_row, "yaw_confidence", 0.0),
        },
        observability=float(runtime_xy_estimate.observability),
        frame_consistency=float(runtime_xy_estimate.frame_consistency),
        abstain_reason="" if bool(runtime_xy_estimate.entry_ready) else str(runtime_xy_estimate.reason),
        z_semantics="task_approach_axis_residual",
        yaw_semantics="task_frame_yaw_residual",
        source=str(runtime_xy_estimate.source),
        uses_privileged_runtime=False,
    )


def _task_frame_readiness_prediction(
    trace_row: Mapping[str, Any],
    *,
    model: TaskFrameReadinessNet | None,
    head: str,
) -> TaskFrameZReadinessEstimate | TaskFrameYawReadinessEstimate | None:
    if model is None:
        return None
    if str(getattr(model, "head_type", "")) != str(head):
        raise ValueError(f"task frame readiness checkpoint head mismatch: expected {head}, got {getattr(model, 'head_type', '')}")
    features = task_frame_readiness_feature_vector(trace_row)
    return model.predict_numpy(features)


def _apply_task_frame_readiness_to_residual(
    residual: TaskFrameResidualEstimate,
    *,
    z_readiness: TaskFrameZReadinessEstimate | None,
    yaw_readiness: TaskFrameYawReadinessEstimate | None,
) -> TaskFrameResidualEstimate:
    axis_validity = dict(residual.axis_validity)
    axis_confidence = dict(residual.axis_confidence)
    abstain_reason = str(residual.abstain_reason)
    if z_readiness is not None:
        axis_validity["z"] = bool(z_readiness.z_ready)
        axis_confidence["z"] = float(z_readiness.z_confidence)
        if not bool(z_readiness.z_ready):
            abstain_reason = str(z_readiness.z_abstain_reason or abstain_reason)
    if yaw_readiness is not None:
        axis_validity["yaw"] = bool(yaw_readiness.yaw_ready)
        axis_confidence["yaw"] = float(yaw_readiness.yaw_confidence)
        if not bool(yaw_readiness.yaw_ready):
            abstain_reason = str(yaw_readiness.yaw_abstain_reason or abstain_reason)
    return replace(
        residual,
        axis_validity=axis_validity,
        axis_confidence=axis_confidence,
        abstain_reason=abstain_reason,
    )


def _load_grasp_probe_candidate_rows(path_text: str | None) -> tuple[set[tuple[int, int]], dict[tuple[int, int], dict[str, object]]]:
    if not path_text:
        return set(), {}
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Missing --c2c_grasp_probe_candidate_jsonl: {path}")
    keys: set[tuple[int, int]] = set()
    rows: dict[tuple[int, int], dict[str, object]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1))))
            keys.add(key)
            rows[key] = row
    return keys, rows


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
    close_ready_z_threshold: float,
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
            grasp_probe_close_ready_with_z(
                float(pre[0]),
                float(pre[1]),
                float(pre[2]),
                float(pre[3]),
                xy_threshold=float(close_ready_xy_threshold),
                yaw_threshold=float(close_ready_yaw_threshold),
                z_threshold=float(close_ready_z_threshold),
            )
        ),
        "grasp_probe_close_ready_after": bool(
            grasp_probe_close_ready_with_z(
                float(post[0]),
                float(post[1]),
                float(post[2]),
                float(post[3]),
                xy_threshold=float(close_ready_xy_threshold),
                yaw_threshold=float(close_ready_yaw_threshold),
                z_threshold=float(close_ready_z_threshold),
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
    parser.add_argument("--c2c_grasp_probe_policy", type=str, default="off", choices=["off", "replay_oracle_xy", "runtime_estimator_xy"])
    parser.add_argument(
        "--runtime_xy_calibration_json",
        type=str,
        default="",
        help="Optional non-privileged XY affine calibration used by --c2c_grasp_probe_policy runtime_estimator_xy.",
    )
    parser.add_argument(
        "--task_frame_z_readiness_ckpt",
        type=str,
        default="",
        help="Optional non-privileged Z readiness checkpoint used to gate strict handoff.",
    )
    parser.add_argument(
        "--task_frame_yaw_readiness_ckpt",
        type=str,
        default="",
        help="Optional non-privileged yaw readiness checkpoint used to gate strict handoff.",
    )
    parser.add_argument(
        "--task_frame_v45_ckpt",
        type=str,
        default="",
        help="Optional non-privileged task-frame dz/dyaw checkpoint used for guarded micro-servo.",
    )
    parser.add_argument(
        "--enable_v45_task_frame_micro_servo",
        action="store_true",
        default=False,
        help="Enable the guarded task-frame Z/Yaw micro-servo for smoke/replay only.",
    )
    parser.add_argument("--v45_task_frame_z_gain", type=float, default=0.35)
    parser.add_argument("--v45_task_frame_yaw_gain", type=float, default=0.25)
    parser.add_argument("--v45_task_frame_max_z_step", type=float, default=0.0030)
    parser.add_argument("--v45_task_frame_max_yaw_step", type=float, default=0.020)
    parser.add_argument("--v45_task_frame_min_z_confidence", type=float, default=0.45)
    parser.add_argument("--v45_task_frame_min_yaw_confidence", type=float, default=0.45)
    parser.add_argument("--v45_task_frame_min_step_scale", type=float, default=0.05)
    parser.add_argument("--v45_task_frame_force_safe_threshold", type=float, default=0.18)
    parser.add_argument(
        "--c2c_grasp_probe_smoke_type",
        type=str,
        default="diagnostic_privileged_probe",
        choices=["diagnostic_privileged_probe", "runtime_style_c2c"],
        help=(
            "MP4/eval evidence type. diagnostic_privileged_probe may use privileged oracle residuals for eval-only "
            "intervention probes; runtime_style_c2c enforces runtime-stage windows and must not be reported as "
            "closed-loop non-privileged success while replay_oracle_xy is active."
        ),
    )
    parser.add_argument("--c2c_grasp_probe_xy_gain", type=float, default=0.35)
    parser.add_argument("--c2c_grasp_probe_max_xy_step", type=float, default=0.0030)
    parser.add_argument("--c2c_grasp_probe_horizon", type=int, default=1)
    parser.add_argument("--c2c_grasp_probe_flush_planner_queue", action="store_true", default=False)
    parser.add_argument("--c2c_grasp_probe_window_mode", type=str, default="stage", choices=["stage", "forced_shell"])
    parser.add_argument(
        "--c2c_grasp_probe_candidate_jsonl",
        type=str,
        default="",
        help="Optional grasp failure-tail candidate JSONL. When set, replay_oracle_xy only activates on listed episode/step rows.",
    )
    parser.add_argument(
        "--c2c_grasp_probe_shell_filter",
        type=str,
        default="off",
        choices=[
            "off",
            "near_yaw_feasible",
            "tight_near_yaw_feasible",
            "coarse_yaw_feasible",
            "frontier_pullback_feasible",
            "small_xy_large_yaw_frontier_feasible",
        ],
    )
    parser.add_argument("--near_grasp_xy_threshold", type=float, default=0.015)
    parser.add_argument("--near_grasp_yaw_threshold", type=float, default=0.08)
    parser.add_argument("--close_ready_xy_threshold", type=float, default=0.005)
    parser.add_argument("--close_ready_yaw_threshold", type=float, default=0.03)
    parser.add_argument("--close_ready_z_threshold", type=float, default=0.020)
    parser.add_argument("--c2c_grasp_probe_outer_pullback_xy_threshold", type=float, default=0.120)
    parser.add_argument("--c2c_grasp_probe_frontier_pullback_xy_threshold", type=float, default=0.180)
    parser.add_argument("--c2c_grasp_probe_small_xy_large_yaw_xy_threshold", type=float, default=0.060)
    parser.add_argument(
        "--c2c_grasp_probe_max_candidate_xy_error",
        type=float,
        default=-1.0,
        help="Maximum candidate xy_error allowed for C2C probe activation. Negative means use frontier threshold.",
    )
    parser.add_argument(
        "--c2c_grasp_probe_min_stage_age",
        type=int,
        default=12,
        help="Minimum runtime stage age before a manifest row can activate runtime-style precision takeover.",
    )
    parser.add_argument(
        "--c2c_grasp_probe_runtime_takeover_tiers",
        type=str,
        default="outer_pullback_candidate,near_basin_shell,micro_entry_ready,yaw_entry_blocked",
        help=(
            "Comma-separated takeover tiers allowed to activate runtime-style precision takeover. "
            "close_ready is intentionally excluded here and remains a diagnostic/offline bucket."
        ),
    )
    parser.add_argument(
        "--c2c_grasp_probe_require_queue_empty",
        action="store_true",
        default=False,
        help="Require the planner action queue to be empty before activating runtime-style precision takeover.",
    )
    parser.add_argument("--c2c_grasp_probe_relax_small_xy_large_yaw_candidate", action="store_true", default=False)
    parser.add_argument("--c2c_grasp_probe_sticky_steps", type=int, default=3)
    parser.add_argument("--c2c_grasp_probe_correction_ema_alpha", type=float, default=0.65)
    parser.add_argument("--c2c_grasp_probe_micro_deadband", type=float, default=0.005)
    parser.add_argument("--c2c_grasp_probe_micro_hysteresis_alpha", type=float, default=0.45)
    parser.add_argument("--c2c_grasp_probe_sticky_decay", type=float, default=0.85)
    parser.add_argument(
        "--c2c_grasp_probe_gripper_mode",
        type=str,
        default="lock_open",
        choices=["lock_open", "planner_after_near", "eval_close_after_near"],
        help="Eval-only probe gripper behavior. Default preserves XY-only open-gripper probes.",
    )
    parser.add_argument("--c2c_grasp_probe_close_handoff_steps", type=int, default=1)
    parser.add_argument(
        "--c2c_grasp_probe_close_validation_closed_steps",
        type=int,
        default=2,
        help="Consecutive observed closed-gripper frames required to mark an eval close handoff as physically closed.",
    )
    parser.add_argument(
        "--c2c_grasp_probe_latch_close_after_handoff",
        action="store_true",
        default=False,
        help="Eval-only smoke option: keep gripper closed after a close handoff triggers.",
    )
    parser.add_argument(
        "--c2c_grasp_probe_block_planner_close_until_ready",
        action="store_true",
        default=True,
        help=(
            "Smoke/eval close arbiter, enabled by default: in C2C handoff windows, keep the gripper open "
            "when the planner asks to close before strict alignment_ready_for_handoff is true."
        ),
    )
    parser.add_argument(
        "--disable_c2c_grasp_probe_block_planner_close_until_ready",
        dest="c2c_grasp_probe_block_planner_close_until_ready",
        action="store_false",
        help="Debug escape hatch: allow planner close without the C2C probe close arbiter.",
    )
    parser.add_argument(
        "--c2c_grasp_probe_close_arbiter_guard_steps",
        type=int,
        default=24,
        help="Steps after the last active XY takeover during which planner close remains guarded.",
    )
    parser.add_argument(
        "--disable_c2c_grasp_probe_alignment_lifecycle",
        action="store_true",
        default=False,
        help="Debug escape hatch: fall back to sticky-window probe activation instead of session lifecycle.",
    )
    parser.add_argument("--c2c_grasp_probe_alignment_max_steps", type=int, default=24)
    parser.add_argument("--c2c_grasp_probe_alignment_max_retries", type=int, default=2)
    parser.add_argument("--c2c_grasp_probe_alignment_reacquire_steps", type=int, default=8)
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
    parser.add_argument(
        "--disable_runtime_xy_pullback_calibration",
        action="store_true",
        default=False,
        help="Do not override report-loaded x/y policies for XY-only bounded pullback.",
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
    task_spec = _maybe_attach_basin_state_calibration(
        task_spec,
        args.basin_state_calibration_report or None,
        enable_runtime_xy_pullback_calibration=not bool(args.disable_runtime_xy_pullback_calibration),
    )
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
        if str(args.c2c_grasp_probe_smoke_type) == "runtime_style_c2c" and str(args.c2c_grasp_probe_window_mode) == "forced_shell":
            raise ValueError("runtime_style_c2c smoke cannot use --c2c_grasp_probe_window_mode forced_shell")
    grasp_probe_candidate_keys, grasp_probe_candidate_rows = _load_grasp_probe_candidate_rows(
        getattr(args, "c2c_grasp_probe_candidate_jsonl", "")
    )
    runtime_xy_calibration = RuntimeXYAffineCalibration.load(getattr(args, "runtime_xy_calibration_json", "") or None)
    runtime_xy_calibration_window_size = max(1, int(getattr(runtime_xy_calibration, "window_size", 1) if runtime_xy_calibration is not None else 1))
    task_frame_z_readiness_model = None
    task_frame_yaw_readiness_model = None
    task_frame_v45_model = None
    if str(getattr(args, "task_frame_z_readiness_ckpt", "") or ""):
        task_frame_z_readiness_model, task_frame_z_readiness_meta = load_task_frame_readiness_checkpoint(
            getattr(args, "task_frame_z_readiness_ckpt"),
            map_location="cpu",
        )
    else:
        task_frame_z_readiness_meta = {}
    if str(getattr(args, "task_frame_yaw_readiness_ckpt", "") or ""):
        task_frame_yaw_readiness_model, task_frame_yaw_readiness_meta = load_task_frame_readiness_checkpoint(
            getattr(args, "task_frame_yaw_readiness_ckpt"),
            map_location="cpu",
        )
    else:
        task_frame_yaw_readiness_meta = {}
    if str(getattr(args, "task_frame_v45_ckpt", "") or ""):
        task_frame_v45_model, task_frame_v45_meta = load_task_frame_v45_candidate_checkpoint(
            getattr(args, "task_frame_v45_ckpt"),
            map_location="cpu",
        )
    else:
        task_frame_v45_meta = {}

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
        "c2c_grasp_probe_smoke_type": str(args.c2c_grasp_probe_smoke_type),
        "c2c_grasp_probe_evidence_boundary": (
            "runtime_style_estimator_xy_uses_privileged_labels_for_eval_only"
            if str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy"
            else "runtime_style_eval_probe_uses_privileged_oracle_residual"
            if str(args.c2c_grasp_probe_smoke_type) == "runtime_style_c2c" and str(args.c2c_grasp_probe_policy) != "off"
            else "diagnostic_privileged_probe" if str(args.c2c_grasp_probe_policy) != "off" else "planner_or_runtime_without_probe"
        ),
        "c2c_grasp_probe_xy_gain": float(args.c2c_grasp_probe_xy_gain),
        "c2c_grasp_probe_max_xy_step": float(args.c2c_grasp_probe_max_xy_step),
        "c2c_grasp_probe_horizon": int(args.c2c_grasp_probe_horizon),
        "c2c_grasp_probe_flush_planner_queue": bool(args.c2c_grasp_probe_flush_planner_queue),
        "c2c_grasp_probe_window_mode": str(args.c2c_grasp_probe_window_mode),
        "c2c_grasp_probe_shell_filter": str(args.c2c_grasp_probe_shell_filter),
        "c2c_grasp_probe_frontier_pullback_xy_threshold": float(args.c2c_grasp_probe_frontier_pullback_xy_threshold),
        "close_ready_z_threshold": float(args.close_ready_z_threshold),
        "c2c_grasp_probe_max_candidate_xy_error": (
            float(args.c2c_grasp_probe_max_candidate_xy_error)
            if float(args.c2c_grasp_probe_max_candidate_xy_error) >= 0.0
            else float(args.c2c_grasp_probe_frontier_pullback_xy_threshold)
        ),
        "c2c_grasp_probe_min_stage_age": int(args.c2c_grasp_probe_min_stage_age),
        "c2c_grasp_probe_runtime_takeover_tiers": str(args.c2c_grasp_probe_runtime_takeover_tiers),
        "c2c_grasp_probe_require_queue_empty": bool(args.c2c_grasp_probe_require_queue_empty),
        "c2c_grasp_probe_small_xy_large_yaw_xy_threshold": float(args.c2c_grasp_probe_small_xy_large_yaw_xy_threshold),
        "c2c_grasp_probe_relax_small_xy_large_yaw_candidate": bool(args.c2c_grasp_probe_relax_small_xy_large_yaw_candidate),
        "c2c_grasp_probe_sticky_steps": int(args.c2c_grasp_probe_sticky_steps),
        "c2c_grasp_probe_correction_ema_alpha": float(args.c2c_grasp_probe_correction_ema_alpha),
        "c2c_grasp_probe_micro_deadband": float(args.c2c_grasp_probe_micro_deadband),
        "c2c_grasp_probe_micro_hysteresis_alpha": float(args.c2c_grasp_probe_micro_hysteresis_alpha),
        "c2c_grasp_probe_sticky_decay": float(args.c2c_grasp_probe_sticky_decay),
        "c2c_grasp_probe_gripper_mode": str(args.c2c_grasp_probe_gripper_mode),
        "c2c_grasp_probe_close_handoff_steps": int(args.c2c_grasp_probe_close_handoff_steps),
        "c2c_grasp_probe_close_validation_closed_steps": int(args.c2c_grasp_probe_close_validation_closed_steps),
        "c2c_grasp_probe_latch_close_after_handoff": bool(args.c2c_grasp_probe_latch_close_after_handoff),
        "c2c_grasp_probe_alignment_lifecycle_enabled": bool(not args.disable_c2c_grasp_probe_alignment_lifecycle),
        "c2c_grasp_probe_alignment_max_steps": int(args.c2c_grasp_probe_alignment_max_steps),
        "c2c_grasp_probe_alignment_max_retries": int(args.c2c_grasp_probe_alignment_max_retries),
        "c2c_grasp_probe_alignment_reacquire_steps": int(args.c2c_grasp_probe_alignment_reacquire_steps),
        "c2c_grasp_probe_candidate_jsonl": str(getattr(args, "c2c_grasp_probe_candidate_jsonl", "")),
        "c2c_grasp_probe_candidate_count": int(len(grasp_probe_candidate_keys)),
        "runtime_xy_calibration_json": str(getattr(args, "runtime_xy_calibration_json", "")),
        "runtime_xy_calibration_loaded": bool(runtime_xy_calibration is not None),
        "runtime_xy_calibration_window_size": int(runtime_xy_calibration_window_size),
        "runtime_xy_pullback_calibration": bool(not args.disable_runtime_xy_pullback_calibration),
        "task_frame_z_readiness_ckpt": str(getattr(args, "task_frame_z_readiness_ckpt", "")),
        "task_frame_z_readiness_loaded": bool(task_frame_z_readiness_model is not None),
        "task_frame_z_readiness_meta": _jsonable_value(task_frame_z_readiness_meta),
        "task_frame_yaw_readiness_ckpt": str(getattr(args, "task_frame_yaw_readiness_ckpt", "")),
        "task_frame_yaw_readiness_loaded": bool(task_frame_yaw_readiness_model is not None),
        "task_frame_yaw_readiness_meta": _jsonable_value(task_frame_yaw_readiness_meta),
        "task_frame_v45_ckpt": str(getattr(args, "task_frame_v45_ckpt", "")),
        "task_frame_v45_loaded": bool(task_frame_v45_model is not None),
        "task_frame_v45_meta": _jsonable_value(task_frame_v45_meta),
        "enable_v45_task_frame_micro_servo": bool(args.enable_v45_task_frame_micro_servo),
        "v45_task_frame_z_gain": float(args.v45_task_frame_z_gain),
        "v45_task_frame_yaw_gain": float(args.v45_task_frame_yaw_gain),
        "v45_task_frame_max_z_step": float(args.v45_task_frame_max_z_step),
        "v45_task_frame_max_yaw_step": float(args.v45_task_frame_max_yaw_step),
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
        probe_last_active_step_idx: int | None = None
        probe_last_active_correction_local = np.zeros(6, dtype=np.float32)
        probe_close_handoff_latched = False
        probe_observed_closed_streak = 0
        probe_sticky_steps_remaining = 0
        probe_close_arbiter_guard_steps_remaining = 0
        runtime_xy_history: deque[dict[str, object]] = deque(maxlen=max(0, int(runtime_xy_calibration_window_size) - 1))
        alignment_session_counter = 0
        alignment_session = AlignmentTakeoverSession()
        alignment_config = AlignmentTakeoverConfig(
            max_control_steps=int(args.c2c_grasp_probe_alignment_max_steps),
            max_retries=int(args.c2c_grasp_probe_alignment_max_retries),
            reacquire_steps=int(args.c2c_grasp_probe_alignment_reacquire_steps),
            xy_threshold=float(args.close_ready_xy_threshold),
            z_threshold=float(args.close_ready_z_threshold),
            yaw_threshold=float(args.close_ready_yaw_threshold),
            min_observability=5.0e-4,
            min_frame_consistency=0.20,
            z_required=True,
            yaw_required=True,
        )
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
            runtime_robot_state = {
                "invalid_action_flag": False,
                "wrist_valid_depth_ratio": float(wrist_valid_depth_ratio),
                "wrist_depth_near_fraction": float(wrist_depth_near_fraction),
                "wrist_is_occluded": bool(wrist_is_occluded),
                "wrist_is_low_visibility": bool(wrist_is_low_visibility),
                "proprio": np.asarray(proprio, dtype=np.float32),
                "planner_delta_7d": np.asarray(base_delta_action[:6], dtype=np.float32),
            }
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
                "c2c_grasp_probe_smoke_type": str(args.c2c_grasp_probe_smoke_type),
                "c2c_grasp_probe_evidence_boundary": (
                    "runtime_style_estimator_xy_uses_privileged_labels_for_eval_only"
                    if str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy"
                    else "runtime_style_eval_probe_uses_privileged_oracle_residual"
                    if str(args.c2c_grasp_probe_smoke_type) == "runtime_style_c2c" and str(args.c2c_grasp_probe_policy) != "off"
                    else "diagnostic_privileged_probe" if str(args.c2c_grasp_probe_policy) != "off" else "planner_or_runtime_without_probe"
                ),
                "runtime_xy_pullback_calibration": bool(not args.disable_runtime_xy_pullback_calibration),
            }
            _attach_offline_eval_only(trace_entry, privileged_frame_pack)

            if c2c is not None:
                delta_action = c2c.step(
                    delta_action,
                    observation=obs,
                    robot_state=runtime_robot_state,
                    task_spec=task_spec,
                    current_instruction=instruction,
                )
                trace_entry.update(_jsonable_value(c2c.get_last_trace()))
                if args.record_video and not gate_frame_saved and bool(c2c.get_last_trace().get("c2c_gate_active", False)) and frames:
                    gate_path = gate_frame_dir / f"ep{ep_idx:03d}_gate_start_step{step_idx:03d}.png"
                    Image.fromarray(np.asarray(frames[-1], dtype=np.uint8)).save(gate_path)
                    trace_entry["c2c_gate_frame_path"] = str(gate_path)
                    gate_frame_saved = True
                probe_visibility_bucket = _runtime_probe_visibility_bucket(trace_entry)
                trace_entry["visual_observability_class"] = str(probe_visibility_bucket)
                runtime_xy_estimate = calibrated_runtime_xy_residual_from_trace(
                    trace_entry,
                    runtime_xy_calibration,
                    history_rows=list(reversed(runtime_xy_history)),
                    observation=obs.as_dict() if hasattr(obs, "as_dict") else {
                        "wrist_rgb": getattr(obs, "wrist_rgb", None),
                        "wrist_depth": getattr(obs, "wrist_depth", None),
                        "front_rgb": getattr(obs, "front_rgb", None),
                        "front_depth": getattr(obs, "front_depth", None),
                        "gripper_pose": getattr(obs, "gripper_pose", None),
                    },
                    robot_state=runtime_robot_state,
                )
                trace_entry["runtime_xy_estimator"] = runtime_xy_estimate.to_dict()
                trace_entry["runtime_xy_estimator_calibration_loaded"] = bool(runtime_xy_calibration is not None)
                trace_entry["runtime_xy_estimator_history_window_size"] = int(runtime_xy_calibration_window_size)
                trace_entry["runtime_xy_estimator_history_rows"] = int(len(runtime_xy_history))
                trace_entry["xy_direction_confidence"] = float(getattr(runtime_xy_estimate, "xy_direction_confidence", 0.0))
                trace_entry["xy_sign_stability"] = float(getattr(runtime_xy_estimate, "xy_sign_stability", 0.0))
                trace_entry["xy_step_scale"] = float(getattr(runtime_xy_estimate, "xy_step_scale", 1.0))
                trace_entry["xy_risk_reason"] = str(getattr(runtime_xy_estimate, "xy_risk_reason", ""))
                trace_entry["xy_stall_reason"] = str(getattr(runtime_xy_estimate, "xy_stall_reason", ""))
                probe_stage = str(trace_entry.get("c2c_v2_stage", ""))
                probe_shell_fields = grasp_probe_shell_fields(
                    probe_true_error_before,
                    near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
                    near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
                    max_xy_step=float(args.c2c_grasp_probe_max_xy_step),
                    horizon_steps=int(max(1, int(args.c2c_grasp_probe_horizon))),
                    outer_pullback_xy_threshold=float(args.c2c_grasp_probe_outer_pullback_xy_threshold),
                    frontier_pullback_xy_threshold=float(args.c2c_grasp_probe_frontier_pullback_xy_threshold),
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
                probe_candidate_required = bool(grasp_probe_candidate_keys)
                probe_candidate_key = (int(ep_idx), int(step_idx))
                probe_candidate_row = grasp_probe_candidate_rows.get(probe_candidate_key, {})
                probe_candidate_match = bool((not probe_candidate_required) or probe_candidate_key in grasp_probe_candidate_keys)
                probe_candidate_axes = probe_candidate_row.get("recommended_intervention_axes", [])
                if not isinstance(probe_candidate_axes, (list, tuple)):
                    probe_candidate_axes = []
                probe_alias_drift_decision = _candidate_alias_drift_decision(probe_candidate_row)
                probe_max_candidate_xy_error = (
                    float(args.c2c_grasp_probe_max_candidate_xy_error)
                    if float(args.c2c_grasp_probe_max_candidate_xy_error) >= 0.0
                    else float(args.c2c_grasp_probe_frontier_pullback_xy_threshold)
                )
                probe_candidate_xy_ready = candidate_xy_correction_ready(
                    probe_candidate_row,
                    max_xy_error=float(probe_max_candidate_xy_error),
                )
                probe_candidate_within_xy_window = candidate_within_xy_activation_window(
                    probe_candidate_row,
                    max_xy_error=float(probe_max_candidate_xy_error),
                )
                probe_candidate_actionable = bool(
                    (not probe_candidate_required)
                    or probe_candidate_xy_ready
                    or (
                        probe_candidate_within_xy_window
                        and not str(probe_candidate_row.get("abstain_reason", ""))
                        and "x" in {str(axis) for axis in probe_candidate_axes}
                        and "y" in {str(axis) for axis in probe_candidate_axes}
                    )
                )
                probe_current_xy_error = float(probe_shell_fields.get("grasp_probe_pre_xy_error", float("nan")))
                if not np.isfinite(probe_current_xy_error):
                    probe_current_xy_error = float(probe_shell_fields.get("grasp_probe_horizon_final_xy_error", float("nan")))
                probe_runtime_soft_xy_ready = bool(
                    str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy"
                    and bool(runtime_xy_estimate.entry_ready)
                    and probe_stage_ok
                    and probe_visibility_bucket != "prior_only"
                    and probe_has_error
                    and probe_finite_xy
                    and np.isfinite(probe_current_xy_error)
                    and float(probe_current_xy_error) <= float(RUNTIME_XY_SOFT_ACTIVATION_RADIUS) + 1.0e-9
                )
                probe_runtime_allowed_tiers = {
                    str(part).strip()
                    for part in str(args.c2c_grasp_probe_runtime_takeover_tiers).split(",")
                    if str(part).strip()
                }
                probe_precision_ready, probe_precision_block_reason = precision_takeover_activation_status(
                    probe_candidate_row,
                    stage_age=int(trace_entry.get("c2c_stage_age", 0) or 0),
                    queue_len=int(len(action_queue)),
                    max_xy_error=float(probe_max_candidate_xy_error),
                    min_stage_age=int(args.c2c_grasp_probe_min_stage_age),
                    allowed_tiers=probe_runtime_allowed_tiers,
                    require_queue_empty=bool(args.c2c_grasp_probe_require_queue_empty),
                )
                small_xy_large_yaw_candidate_relaxed = bool(
                    args.c2c_grasp_probe_relax_small_xy_large_yaw_candidate
                    and args.c2c_grasp_probe_shell_filter == "small_xy_large_yaw_frontier_feasible"
                    and str(probe_candidate_row.get("failure_bucket", "")) == "small_xy_large_yaw"
                    and probe_candidate_required
                    and probe_candidate_match
                    and "x" in {str(axis) for axis in probe_candidate_axes}
                    and "y" in {str(axis) for axis in probe_candidate_axes}
                )
                probe_shell_ok = bool(
                    args.c2c_grasp_probe_shell_filter == "off"
                    or (
                        args.c2c_grasp_probe_shell_filter == "near_yaw_feasible"
                        and bool(probe_shell_fields.get("grasp_probe_near_basin_shell", False))
                    )
                    or (
                        args.c2c_grasp_probe_shell_filter == "tight_near_yaw_feasible"
                        and bool(probe_shell_fields.get("grasp_probe_tight_near_basin_shell", False))
                    )
                    or (
                        args.c2c_grasp_probe_shell_filter == "coarse_yaw_feasible"
                        and (
                            bool(probe_shell_fields.get("grasp_probe_near_basin_shell", False))
                            or bool(probe_shell_fields.get("grasp_probe_coarse_pullback_candidate", False))
                        )
                    )
                    or (
                        args.c2c_grasp_probe_shell_filter == "frontier_pullback_feasible"
                        and (
                            bool(probe_shell_fields.get("grasp_probe_outer_pullback_candidate", False))
                            or bool(probe_shell_fields.get("grasp_probe_coarse_pullback_candidate", False))
                            or bool(probe_shell_fields.get("grasp_probe_near_basin_shell", False))
                            or bool(probe_shell_fields.get("grasp_probe_frontier_pullback_candidate", False))
                        )
                    )
                    or (
                        args.c2c_grasp_probe_shell_filter == "small_xy_large_yaw_frontier_feasible"
                        and str(probe_candidate_row.get("failure_bucket", "")) == "small_xy_large_yaw"
                        and float(probe_shell_fields.get("grasp_probe_pre_xy_error", float("inf"))) <= float(args.c2c_grasp_probe_small_xy_large_yaw_xy_threshold)
                    )
                )
                probe_eligible = bool(
                    args.c2c_grasp_probe_policy in {"replay_oracle_xy", "runtime_estimator_xy"}
                    and probe_stage_ok
                    and probe_visibility_bucket != "prior_only"
                    and probe_has_error
                    and probe_finite_xy
                    and (probe_candidate_match or probe_runtime_soft_xy_ready)
                    and (probe_precision_ready or probe_runtime_soft_xy_ready)
                    and (probe_shell_ok or small_xy_large_yaw_candidate_relaxed or probe_runtime_soft_xy_ready)
                    and (probe_candidate_actionable or small_xy_large_yaw_candidate_relaxed or probe_runtime_soft_xy_ready)
                    and (
                        str(args.c2c_grasp_probe_policy) != "runtime_estimator_xy"
                        or bool(runtime_xy_estimate.entry_ready or probe_runtime_soft_xy_ready)
                    )
                )
                trace_entry["grasp_probe_policy"] = str(args.c2c_grasp_probe_policy)
                trace_entry["grasp_probe_smoke_type"] = str(args.c2c_grasp_probe_smoke_type)
                trace_entry["grasp_probe_evidence_boundary"] = (
                    "runtime_style_estimator_xy_uses_privileged_labels_for_eval_only"
                    if str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy"
                    else "runtime_style_eval_probe_uses_privileged_oracle_residual"
                    if str(args.c2c_grasp_probe_smoke_type) == "runtime_style_c2c" and str(args.c2c_grasp_probe_policy) != "off"
                    else "diagnostic_privileged_probe"
                )
                trace_entry["grasp_probe_visibility_bucket"] = probe_visibility_bucket
                trace_entry["grasp_probe_active"] = bool(probe_eligible)
                trace_entry["grasp_probe_candidate_required"] = bool(probe_candidate_required)
                trace_entry["grasp_probe_candidate_match"] = bool(probe_candidate_match)
                trace_entry["grasp_probe_candidate_actionable"] = bool(probe_candidate_actionable)
                trace_entry["grasp_probe_runtime_soft_xy_ready"] = bool(probe_runtime_soft_xy_ready)
                trace_entry["grasp_probe_soft_xy_activation_radius"] = float(RUNTIME_XY_SOFT_ACTIVATION_RADIUS)
                trace_entry["grasp_probe_xy_correction_ready"] = bool(probe_candidate_xy_ready)
                trace_entry["grasp_probe_candidate_within_xy_activation_window"] = bool(probe_candidate_within_xy_window)
                trace_entry["grasp_probe_precision_activation_ready"] = bool(probe_precision_ready)
                trace_entry["grasp_probe_precision_activation_block_reason"] = str(probe_precision_block_reason)
                trace_entry["grasp_probe_min_stage_age"] = int(args.c2c_grasp_probe_min_stage_age)
                trace_entry["grasp_probe_runtime_takeover_tiers"] = sorted(probe_runtime_allowed_tiers)
                trace_entry["grasp_probe_require_queue_empty"] = bool(args.c2c_grasp_probe_require_queue_empty)
                try:
                    trace_entry["grasp_probe_candidate_xy_error"] = float(probe_candidate_row.get("xy_error", float("nan")))
                except Exception:
                    trace_entry["grasp_probe_candidate_xy_error"] = float("nan")
                trace_entry["grasp_probe_max_candidate_xy_error"] = float(probe_max_candidate_xy_error)
                trace_entry["grasp_probe_candidate_actionable_relaxed_small_xy_large_yaw"] = bool(small_xy_large_yaw_candidate_relaxed)
                trace_entry["alias_drift_decision"] = str(probe_alias_drift_decision)
                trace_entry["yaw_alias_drift_decision"] = str(probe_alias_drift_decision)
                trace_entry["grasp_probe_candidate_jsonl"] = str(getattr(args, "c2c_grasp_probe_candidate_jsonl", ""))
                trace_entry["grasp_probe_failure_tail_sample_role"] = str(probe_candidate_row.get("sample_role", ""))
                trace_entry["grasp_probe_failure_tail_takeover_tier"] = str(probe_candidate_row.get("takeover_tier", ""))
                trace_entry["grasp_probe_failure_tail_abstain_reason"] = str(probe_candidate_row.get("abstain_reason", ""))
                trace_entry["grasp_probe_failure_tail_planner_natural_outcome"] = str(probe_candidate_row.get("planner_natural_outcome", ""))
                trace_entry["failure_bucket"] = str(probe_candidate_row.get("failure_bucket", trace_entry.get("failure_bucket", "")))
                trace_entry["grasp_probe_window_mode"] = str(args.c2c_grasp_probe_window_mode)
                trace_entry["grasp_probe_shell_filter"] = str(args.c2c_grasp_probe_shell_filter)
                trace_entry["grasp_probe_stage_ok"] = bool(probe_stage_ok)
                trace_entry["grasp_probe_stage_source"] = "runtime_stage" if probe_stage == "RING_GRASP_ALIGN" else ("forced_shell" if probe_stage_ok else "not_grasp_align")
                if probe_eligible:
                    trace_entry["grasp_probe_reason"] = (
                        "runtime_estimator_xy_soft_gate"
                        if probe_runtime_soft_xy_ready
                        else str(args.c2c_grasp_probe_policy)
                    )
                elif probe_candidate_required and not probe_candidate_match:
                    trace_entry["grasp_probe_reason"] = "not_failure_tail_candidate"
                elif str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy" and not bool(runtime_xy_estimate.entry_ready):
                    trace_entry["grasp_probe_reason"] = str(runtime_xy_estimate.reason)
                elif probe_runtime_soft_xy_ready:
                    trace_entry["grasp_probe_reason"] = "runtime_estimator_xy_soft_gate"
                elif probe_candidate_required and not probe_precision_ready:
                    trace_entry["grasp_probe_reason"] = str(probe_precision_block_reason)
                elif small_xy_large_yaw_candidate_relaxed:
                    trace_entry["grasp_probe_reason"] = "small_xy_large_yaw_frontier_relaxed"
                elif probe_candidate_required and not probe_candidate_actionable:
                    trace_entry["grasp_probe_reason"] = "failure_tail_candidate_abstain"
                else:
                    trace_entry["grasp_probe_reason"] = grasp_probe_inactive_reason(
                        policy=str(args.c2c_grasp_probe_policy),
                        stage_ok=bool(probe_stage_ok),
                        visibility_bucket=probe_visibility_bucket,
                        has_error=bool(probe_has_error),
                        finite_xy=bool(probe_finite_xy),
                        shell_filter=str(args.c2c_grasp_probe_shell_filter),
                        shell_fields=probe_shell_fields,
                    )
                task_frame_residual = _task_frame_residual_from_runtime_trace(trace_entry, runtime_xy_estimate)
                z_readiness = _task_frame_readiness_prediction(
                    trace_entry,
                    model=task_frame_z_readiness_model,
                    head="z",
                )
                yaw_readiness = _task_frame_readiness_prediction(
                    trace_entry,
                    model=task_frame_yaw_readiness_model,
                    head="yaw",
                )
                if isinstance(z_readiness, TaskFrameZReadinessEstimate):
                    trace_entry.update(_jsonable_value(z_readiness.to_dict()))
                    trace_entry["task_frame_z_ready"] = bool(z_readiness.z_ready)
                    trace_entry["task_frame_z_readiness_loaded"] = True
                else:
                    trace_entry.setdefault("z_readiness_source", "heuristic_alignment_trace")
                    trace_entry.setdefault("z_readiness_loaded", False)
                    trace_entry["task_frame_z_readiness_loaded"] = False
                if isinstance(yaw_readiness, TaskFrameYawReadinessEstimate):
                    trace_entry.update(_jsonable_value(yaw_readiness.to_dict()))
                    trace_entry["task_frame_yaw_ready"] = bool(yaw_readiness.yaw_ready)
                    trace_entry["task_frame_yaw_readiness_loaded"] = True
                else:
                    trace_entry.setdefault("yaw_readiness_source", "heuristic_alignment_trace")
                    trace_entry.setdefault("yaw_readiness_loaded", False)
                    trace_entry["task_frame_yaw_readiness_loaded"] = False
                task_frame_residual = _apply_task_frame_readiness_to_residual(
                    task_frame_residual,
                    z_readiness=z_readiness if isinstance(z_readiness, TaskFrameZReadinessEstimate) else None,
                    yaw_readiness=yaw_readiness if isinstance(yaw_readiness, TaskFrameYawReadinessEstimate) else None,
                )
                alignment_readiness = evaluate_alignment_readiness(task_frame_residual, alignment_config)
                task_frame_v45_estimate = None
                task_frame_v45_decision = None
                task_frame_v45_local_step = np.zeros(6, dtype=np.float32)
                if bool(args.enable_v45_task_frame_micro_servo) and task_frame_v45_model is not None:
                    raw_force_norm = float(np.linalg.norm(np.asarray(raw_force if raw_force is not None else np.zeros(6, dtype=np.float32), dtype=np.float32)))
                    trace_entry["task_frame_v45_force_norm"] = float(raw_force_norm)
                    task_frame_v45_estimate, task_frame_v45_decision, task_frame_v45_local_step = task_frame_v45_micro_servo_step(
                        trace_entry,
                        model=task_frame_v45_model,
                        history_rows=list(runtime_xy_history),
                        xy_ready=bool(alignment_readiness.xy_ready),
                        z_readiness=z_readiness if isinstance(z_readiness, TaskFrameZReadinessEstimate) else None,
                        yaw_readiness=yaw_readiness if isinstance(yaw_readiness, TaskFrameYawReadinessEstimate) else None,
                        force_safe=bool(raw_force_norm < float(args.v45_task_frame_force_safe_threshold)),
                        z_gain=float(args.v45_task_frame_z_gain),
                        yaw_gain=float(args.v45_task_frame_yaw_gain),
                        max_z_step=float(args.v45_task_frame_max_z_step),
                        max_yaw_step=float(args.v45_task_frame_max_yaw_step),
                        min_z_confidence=float(args.v45_task_frame_min_z_confidence),
                        min_yaw_confidence=float(args.v45_task_frame_min_yaw_confidence),
                        min_step_scale=float(args.v45_task_frame_min_step_scale),
                    )
                    trace_entry.update(_jsonable_value(task_frame_v45_estimate.to_dict()))
                    trace_entry.update(_jsonable_value(task_frame_v45_decision.to_dict()))
                else:
                    trace_entry["task_frame_v45_applied"] = False
                    trace_entry["task_frame_v45_block_reason"] = "disabled"
                    trace_entry["task_frame_v45_z_block_reason"] = "disabled"
                    trace_entry["task_frame_v45_yaw_block_reason"] = "disabled"
                    trace_entry["task_frame_v45_step_scale"] = 0.0
                trace_entry["task_frame_v45_enabled"] = bool(args.enable_v45_task_frame_micro_servo and task_frame_v45_model is not None)
                lifecycle_enabled = bool(args.c2c_grasp_probe_policy != "off" and not args.disable_c2c_grasp_probe_alignment_lifecycle)
                if lifecycle_enabled and probe_eligible and not alignment_session.active:
                    alignment_session_counter += 1
                    alignment_session = alignment_session.begin(alignment_session_counter)
                alignment_visual_ready = bool(
                    probe_stage_ok
                    and probe_visibility_bucket != "prior_only"
                    and (
                        bool(runtime_xy_estimate.visual_evidence_valid)
                        or (probe_has_error and probe_finite_xy)
                    )
                )
                if lifecycle_enabled:
                    alignment_session = alignment_session.update(
                        eligible_now=bool(probe_eligible),
                        visual_ready=bool(alignment_visual_ready),
                        readiness=alignment_readiness,
                        config=alignment_config,
                    )
                lifecycle_takeover_active = bool(lifecycle_enabled and alignment_session.active)
                sticky_takeover_active = bool(
                    probe_sticky_steps_remaining > 0
                    and probe_candidate_match
                    and probe_stage_ok
                    and probe_visibility_bucket != "prior_only"
                    and probe_has_error
                    and probe_finite_xy
                    and probe_precision_ready
                    and (probe_shell_ok or small_xy_large_yaw_candidate_relaxed)
                    and (probe_candidate_actionable or small_xy_large_yaw_candidate_relaxed)
                    and (
                        str(args.c2c_grasp_probe_policy) != "runtime_estimator_xy"
                        or float(np.linalg.norm(probe_last_active_correction_local[:2])) > 0.0
                    )
                )
                probe_takeover_active = bool(
                    lifecycle_takeover_active
                    or probe_eligible
                    or (sticky_takeover_active and not lifecycle_enabled)
                )
                probe_control_step_ready = bool(
                    probe_eligible
                    or (
                        lifecycle_takeover_active
                        and probe_stage_ok
                        and probe_visibility_bucket != "prior_only"
                        and probe_has_error
                        and probe_finite_xy
                        and (
                            str(args.c2c_grasp_probe_policy) != "runtime_estimator_xy"
                            or bool(runtime_xy_estimate.entry_ready or probe_runtime_soft_xy_ready)
                        )
                    )
                )
                trace_entry["grasp_probe_active"] = bool(probe_takeover_active)
                trace_entry["grasp_probe_control_step_ready"] = bool(probe_control_step_ready)
                trace_entry["task_frame_residual_estimate"] = _jsonable_value(task_frame_residual.to_dict())
                trace_entry.update(_jsonable_value(alignment_readiness.to_dict()))
                trace_entry.update(_jsonable_value(alignment_session.to_trace()))
                trace_entry["alignment_lifecycle_enabled"] = bool(lifecycle_enabled)
                trace_entry["planner_gripper_handoff_allowed"] = bool(
                    alignment_readiness.alignment_ready_for_handoff
                    or alignment_session.alignment_ready_for_handoff
                )
                trace_entry["grasp_probe_sticky_takeover_active"] = bool(probe_sticky_steps_remaining > 0)
                trace_entry["grasp_probe_sticky_steps_remaining"] = int(probe_sticky_steps_remaining)
                if probe_takeover_active:
                    probe_close_arbiter_guard_steps_remaining = max(
                        int(args.c2c_grasp_probe_close_arbiter_guard_steps),
                        0,
                    )
                trace_entry["grasp_probe_close_arbiter_guard_steps_remaining"] = int(
                    probe_close_arbiter_guard_steps_remaining
                )
                trace_entry["grasp_probe_requested_horizon"] = int(
                    1 if str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy" else max(1, int(args.c2c_grasp_probe_horizon))
                )
                trace_entry["grasp_probe_horizon_steps_executed"] = 0
                trace_entry["grasp_probe_gripper_mode"] = str(args.c2c_grasp_probe_gripper_mode)
                trace_entry["grasp_probe_close_locked"] = bool(
                    probe_takeover_active and str(args.c2c_grasp_probe_gripper_mode) == "lock_open"
                )
                trace_entry["grasp_probe_planner_gripper_handoff"] = False
                trace_entry["grasp_probe_eval_close_handoff"] = False
                trace_entry["grasp_probe_eval_close_handoff_steps_executed"] = 0
                trace_entry["grasp_probe_close_handoff_latched"] = bool(probe_close_handoff_latched)
                trace_entry["grasp_probe_close_commanded"] = False
                trace_entry["grasp_probe_observed_closed_after"] = False
                trace_entry["grasp_probe_observed_closed_streak"] = int(probe_observed_closed_streak)
                trace_entry["grasp_probe_close_handoff_observed_closed"] = False
                trace_entry["grasp_probe_close_handoff_success_reward"] = False
                trace_entry["grasp_probe_close_handoff_validated"] = False
                trace_entry["grasp_probe_flush_planner_queue_requested"] = bool(args.c2c_grasp_probe_flush_planner_queue)
                trace_entry["grasp_probe_queue_len_before"] = int(len(action_queue))
                trace_entry["grasp_probe_queue_len_after"] = int(len(action_queue))
                trace_entry["grasp_probe_queue_flushed"] = False
                trace_entry["grasp_probe_pre_true_error_t"] = _jsonable_value(_compact_grasp_error(probe_true_error_before))
                trace_entry.update(probe_shell_fields)
                probe_raw_correction_local = np.zeros(6, dtype=np.float32)
                probe_applied_correction_local = np.zeros(6, dtype=np.float32)
                probe_residual_norm_xy = float(
                    np.linalg.norm(np.asarray(probe_true_error_before, dtype=np.float32).reshape(-1)[:2])
                ) if probe_has_error and probe_finite_xy else float("nan")
                probe_runtime_estimator_residual_norm_xy = float(
                    np.hypot(float(runtime_xy_estimate.dx), float(runtime_xy_estimate.dy))
                ) if bool(runtime_xy_estimate.entry_ready) else float("nan")
                probe_micro_deadband_active = bool(
                    np.isfinite(probe_runtime_estimator_residual_norm_xy if str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy" else probe_residual_norm_xy)
                    and float(probe_runtime_estimator_residual_norm_xy if str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy" else probe_residual_norm_xy) < float(args.c2c_grasp_probe_micro_deadband)
                )
                probe_micro_hysteresis_alpha_used = float(args.c2c_grasp_probe_correction_ema_alpha)
                if probe_control_step_ready:
                    if str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy":
                        if bool(runtime_xy_estimate.entry_ready):
                            probe_raw_correction_local = _bounded_xy_estimator_probe_step(
                                float(runtime_xy_estimate.dx),
                                float(runtime_xy_estimate.dy),
                                xy_gain=float(args.c2c_grasp_probe_xy_gain),
                                max_xy_step=float(args.c2c_grasp_probe_max_xy_step),
                            )
                        else:
                            probe_raw_correction_local = np.zeros(6, dtype=np.float32)
                            trace_entry["grasp_probe_reason"] = str(runtime_xy_estimate.reason)
                    else:
                        probe_raw_correction_local = _bounded_xy_oracle_probe_step(
                            np.asarray(probe_true_error_before, dtype=np.float32),
                            xy_gain=float(args.c2c_grasp_probe_xy_gain),
                            max_xy_step=float(args.c2c_grasp_probe_max_xy_step),
                        )
                    if probe_last_active_step_idx is not None and step_idx == int(probe_last_active_step_idx) + 1:
                        if probe_micro_deadband_active and probe_last_active_correction_local is not None:
                            probe_micro_hysteresis_alpha_used = float(
                                min(float(args.c2c_grasp_probe_correction_ema_alpha), float(args.c2c_grasp_probe_micro_hysteresis_alpha))
                            )
                        probe_applied_correction_local = smooth_grasp_probe_xy_step(
                            probe_raw_correction_local,
                            probe_last_active_correction_local,
                            alpha=float(probe_micro_hysteresis_alpha_used),
                            max_xy_step=float(args.c2c_grasp_probe_max_xy_step),
                            residual_norm=float(probe_runtime_estimator_residual_norm_xy if str(args.c2c_grasp_probe_policy) == "runtime_estimator_xy" else probe_residual_norm_xy),
                            micro_deadband=float(args.c2c_grasp_probe_micro_deadband),
                            micro_hysteresis_alpha=float(args.c2c_grasp_probe_micro_hysteresis_alpha),
                        )
                    else:
                        probe_applied_correction_local = probe_raw_correction_local.copy()
                    probe_last_active_step_idx = int(step_idx)
                    probe_last_active_correction_local = probe_applied_correction_local.copy()
                    probe_sticky_steps_remaining = max(int(args.c2c_grasp_probe_sticky_steps) - 1, 0)
                elif probe_takeover_active and probe_sticky_steps_remaining > 0:
                    probe_raw_correction_local = probe_last_active_correction_local.copy()
                    probe_applied_correction_local = (float(args.c2c_grasp_probe_sticky_decay) * probe_last_active_correction_local).astype(np.float32)
                    probe_last_active_correction_local = probe_applied_correction_local.copy()
                    probe_last_active_step_idx = int(step_idx)
                    probe_sticky_steps_remaining = max(probe_sticky_steps_remaining - 1, 0)
                if probe_takeover_active:
                    probe_pre_near_or_micro = bool(
                        in_near_grasp_basin(
                            float(probe_true_error_before[0]),
                            float(probe_true_error_before[1]),
                            float(probe_true_error_before[5] if np.asarray(probe_true_error_before).reshape(-1).size >= 6 else probe_true_error_before[3]),
                            xy_threshold=float(args.near_grasp_xy_threshold),
                            yaw_threshold=float(args.near_grasp_yaw_threshold),
                        )
                    ) if probe_has_error and np.asarray(probe_true_error_before).reshape(-1).size >= 4 else False
                    probe_pre_close_ready = bool(
                        alignment_session.alignment_ready_for_handoff
                        or alignment_readiness.alignment_ready_for_handoff
                    )
                    current_local_command = world_delta_to_local(np.asarray(delta_action[:6], dtype=np.float32), np.asarray(obs.gripper_pose[3:7], dtype=np.float32)).astype(np.float32)
                    probe_local_command = current_local_command.copy()
                    probe_local_command[0] += float(probe_applied_correction_local[0])
                    probe_local_command[1] += float(probe_applied_correction_local[1])
                    if bool(task_frame_v45_decision is not None and task_frame_v45_decision.applied):
                        probe_local_command[2] += float(task_frame_v45_local_step[2])
                        probe_local_command[5] += float(task_frame_v45_local_step[5])
                        trace_entry["task_frame_v45_applied_local_6d"] = _jsonable_value(task_frame_v45_local_step)
                        trace_entry["task_frame_v45_applied"] = True
                    else:
                        trace_entry.setdefault("task_frame_v45_applied_local_6d", _jsonable_value(np.zeros(6, dtype=np.float32)))
                        trace_entry["task_frame_v45_applied"] = False
                    probe_world_delta = local_delta_to_world(probe_local_command, np.asarray(obs.gripper_pose[3:7], dtype=np.float32)).astype(np.float32)
                    delta_action = delta_action.copy()
                    delta_action[:6] = probe_world_delta[:6]
                    if bool(probe_close_handoff_latched):
                        delta_action[6] = 0.0
                        trace_entry["grasp_probe_close_locked"] = False
                        trace_entry["grasp_probe_close_handoff_latched"] = True
                    elif str(args.c2c_grasp_probe_gripper_mode) == "planner_after_near" and probe_pre_close_ready:
                        delta_action[6] = float(base_delta_action[6]) if len(base_delta_action) > 6 else 1.0
                        trace_entry["grasp_probe_planner_gripper_handoff"] = True
                        trace_entry["grasp_probe_close_locked"] = False
                    else:
                        delta_action[6] = 1.0
                    trace_entry["grasp_probe_raw_xy_step_local_6d"] = _jsonable_value(probe_raw_correction_local)
                    trace_entry["grasp_probe_smoothed_xy_step_local_6d"] = _jsonable_value(probe_applied_correction_local)
                    trace_entry["grasp_probe_applied_xy_step_local_6d"] = _jsonable_value(probe_applied_correction_local)
                    trace_entry["grasp_probe_micro_deadband_active"] = bool(probe_micro_deadband_active)
                    trace_entry["grasp_probe_pre_near_or_micro_for_diagnostic"] = bool(probe_pre_near_or_micro)
                    trace_entry["grasp_probe_pre_close_ready_for_handoff"] = bool(probe_pre_close_ready)
                    trace_entry["alignment_ready_for_handoff"] = bool(probe_pre_close_ready)
                    trace_entry["alignment_handoff_block_reason"] = str(alignment_readiness.block_reason)
                    trace_entry["grasp_probe_close_ready_z_threshold"] = float(args.close_ready_z_threshold)
                    trace_entry["grasp_probe_micro_deadband_threshold"] = float(args.c2c_grasp_probe_micro_deadband)
                    trace_entry["grasp_probe_micro_hysteresis_alpha_used"] = float(probe_micro_hysteresis_alpha_used)
                    trace_entry["grasp_probe_residual_norm_xy"] = float(probe_residual_norm_xy)
                    trace_entry["grasp_probe_runtime_estimator_residual_norm_xy"] = float(probe_runtime_estimator_residual_norm_xy)
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
                    trace_entry["grasp_probe_raw_xy_step_local_6d"] = _jsonable_value(np.zeros(6, dtype=np.float32))
                    trace_entry["grasp_probe_smoothed_xy_step_local_6d"] = _jsonable_value(np.zeros(6, dtype=np.float32))
                    trace_entry["grasp_probe_applied_xy_step_local_6d"] = _jsonable_value(np.zeros(6, dtype=np.float32))
                    trace_entry["grasp_probe_micro_deadband_active"] = False
                    trace_entry["grasp_probe_micro_deadband_threshold"] = float(args.c2c_grasp_probe_micro_deadband)
                    trace_entry["grasp_probe_micro_hysteresis_alpha_used"] = float(args.c2c_grasp_probe_correction_ema_alpha)
                    trace_entry["grasp_probe_residual_norm_xy"] = float(probe_residual_norm_xy)
                    trace_entry["grasp_probe_runtime_estimator_residual_norm_xy"] = float(probe_runtime_estimator_residual_norm_xy)
                    trace_entry["grasp_probe_local_command_local_6d"] = _jsonable_value(inactive_local_command)
                    trace_entry["grasp_probe_control_gate_axes"] = list(c2c.get_last_trace().get("basin_control_gate_axes", []))
                    trace_entry["grasp_probe_pullback_ready_axes"] = list(c2c.get_last_trace().get("basin_pullback_ready_axes", []))

            if bool(probe_close_handoff_latched):
                delta_action = delta_action.copy()
                if delta_action.shape[0] > 6:
                    delta_action[6] = 0.0
                trace_entry["grasp_probe_close_handoff_latched"] = True
                trace_entry["grasp_probe_close_locked"] = False

            if args.c2c_grasp_probe_policy != "off":
                compact_pre_for_close = (
                    _compact_grasp_error(probe_true_error_before)
                    if probe_true_error_before is not None
                    else np.full((4,), np.nan, dtype=np.float32)
                )
                close_arbiter_xy_norm = float(np.hypot(float(compact_pre_for_close[0]), float(compact_pre_for_close[1])))
                close_arbiter_z_abs = float(abs(float(compact_pre_for_close[2])))
                close_arbiter_yaw_abs = float(abs(float(compact_pre_for_close[3])))
                close_arbiter_xy_ready = bool(
                    np.isfinite(close_arbiter_xy_norm)
                    and close_arbiter_xy_norm <= float(args.close_ready_xy_threshold)
                )
                close_arbiter_z_ready = bool(
                    np.isfinite(close_arbiter_z_abs)
                    and close_arbiter_z_abs <= float(args.close_ready_z_threshold)
                )
                close_arbiter_yaw_ready = bool(
                    np.isfinite(close_arbiter_yaw_abs)
                    and close_arbiter_yaw_abs <= float(args.close_ready_yaw_threshold)
                )
                runtime_close_ready = bool(
                    trace_entry.get("planner_gripper_handoff_allowed", False)
                    or trace_entry.get("alignment_ready_for_handoff", False)
                )
                close_arbiter_has_eval_label = bool(
                    args.dump_runtime_obs
                    and args.capture_failure_target_pose
                    and np.all(np.isfinite(compact_pre_for_close[:4]))
                )
                close_arbiter_ready = bool(runtime_close_ready)
                if bool(probe_close_handoff_latched):
                    close_arbiter_ready = True
                planner_gripper_value = float(base_delta_action[6]) if len(base_delta_action) > 6 else 1.0
                if bool(probe_close_handoff_latched):
                    planner_gripper_value = 0.0
                authority = planner_gripper_authority_decision(
                    planner_gripper_value=planner_gripper_value,
                    planner_close_threshold=0.5,
                    alignment_ready_for_handoff=bool(close_arbiter_ready),
                    stage_name=str(trace_entry.get("c2c_v2_stage", "")),
                    enabled=bool(args.c2c_grasp_probe_block_planner_close_until_ready),
                    guard_active=bool(probe_close_arbiter_guard_steps_remaining > 0),
                    active=bool(trace_entry.get("grasp_probe_active", False)),
                    candidate_match=bool(trace_entry.get("grasp_probe_candidate_match", False)),
                    gripper_mode=str(args.c2c_grasp_probe_gripper_mode),
                    c2c_open_safety_requested=bool(
                        bool(trace_entry.get("safe_abstain_open", False))
                        or bool(trace_entry.get("failed_retryable", False))
                        or bool(trace_entry.get("failed_terminal", False))
                        or str(args.c2c_grasp_probe_gripper_mode) == "lock_open"
                    ),
                    c2c_close_recommendation=bool(probe_close_handoff_latched),
                    handoff_already_latched=bool(probe_close_handoff_latched),
                )
                if delta_action.shape[0] > 6:
                    delta_action = delta_action.copy()
                    delta_action[6] = float(authority["gripper_open"])
                trace_entry["grasp_probe_planner_close_requested"] = bool(authority["planner_gripper_close_requested"])
                trace_entry["planner_gripper_close_requested"] = bool(authority["planner_gripper_close_requested"])
                trace_entry["planner_gripper_close_allowed"] = bool(authority["planner_gripper_close_allowed"])
                trace_entry["planner_gripper_close_blocked"] = bool(authority["planner_gripper_close_blocked"])
                trace_entry["planner_gripper_handoff_allowed"] = bool(authority["planner_gripper_handoff_allowed"])
                trace_entry["planner_gripper_strict_handoff_ready"] = bool(authority["planner_gripper_strict_handoff_ready"])
                trace_entry["planner_gripper_handoff_latched"] = bool(authority["planner_gripper_handoff_latched"])
                trace_entry["c2c_gripper_open_safety_requested"] = bool(authority["c2c_gripper_open_safety_requested"])
                trace_entry["c2c_gripper_close_recommendation"] = bool(authority["c2c_gripper_close_recommendation"])
                trace_entry["c2c_gripper_close_recommendation_ignored"] = bool(authority["c2c_gripper_close_recommendation_ignored"])
                trace_entry["gripper_authority_source"] = str(authority["gripper_authority_source"])
                trace_entry["gripper_authority_reason"] = str(authority["reason"])
                trace_entry["gripper_authority_decision"] = str(authority["decision"])
                trace_entry["grasp_probe_close_arbiter_enabled"] = bool(args.c2c_grasp_probe_block_planner_close_until_ready)
                trace_entry["grasp_probe_close_arbiter_source"] = (
                    "alignment_takeover_lifecycle"
                    if bool(trace_entry.get("alignment_lifecycle_enabled", False))
                    else "runtime_gate"
                )
                trace_entry["grasp_probe_close_arbiter_ready"] = bool(close_arbiter_ready)
                trace_entry["grasp_probe_close_arbiter_xy_ready"] = bool(close_arbiter_xy_ready)
                trace_entry["grasp_probe_close_arbiter_z_ready"] = bool(close_arbiter_z_ready)
                trace_entry["grasp_probe_close_arbiter_yaw_ready"] = bool(close_arbiter_yaw_ready)
                trace_entry["grasp_probe_close_arbiter_xy_norm"] = float(close_arbiter_xy_norm)
                trace_entry["grasp_probe_close_arbiter_z_abs"] = float(close_arbiter_z_abs)
                trace_entry["grasp_probe_close_arbiter_yaw_abs"] = float(close_arbiter_yaw_abs)
                trace_entry["grasp_probe_close_blocked_by_c2c"] = bool(authority["planner_gripper_close_blocked"])
                trace_entry["grasp_probe_close_block_reason"] = str(authority["reason"])
                trace_entry["alignment_gripper_handoff_block_reason"] = str(
                    "ready" if close_arbiter_ready else trace_entry.get("alignment_handoff_block_reason", authority["reason"])
                )
                trace_entry["grasp_probe_close_arbiter_protected_window"] = bool(authority["protected_window"])
                if not bool(trace_entry.get("grasp_probe_active", False)) and probe_close_arbiter_guard_steps_remaining > 0:
                    probe_close_arbiter_guard_steps_remaining = max(probe_close_arbiter_guard_steps_remaining - 1, 0)

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
                if not bool(trace_entry.get("grasp_probe_active", False)):
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
                trace_entry["basin_close_ready_diagnostic_legacy"] = bool(last_trace.get("basin_close_ready_diagnostic_legacy", last_trace.get("basin_close_ready", False)))
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
                    pre_probe = _compact_grasp_error(probe_true_error_before)
                    post_probe = _compact_grasp_error(probe_true_error_after)
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
                        close_ready_z_threshold=float(args.close_ready_z_threshold),
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
                    requested_horizon = int(max(1, int(trace_entry.get("grasp_probe_requested_horizon", args.c2c_grasp_probe_horizon) or 1)))
                    while (
                        int(trace_entry["grasp_probe_horizon_steps_executed"]) < requested_horizon
                        and not bool(terminate)
                    ):
                        latest_pack = _episode_privileged_frame_pack(task, obs)
                        latest_error = _grasp_teacher_error_from_pack(latest_pack, grasp_spec)
                        if latest_error is None:
                            break
                        step_pre = _compact_grasp_error(latest_error)
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
                        step_post = _compact_grasp_error(step_after_error)
                        step_metrics = _grasp_probe_metric_fields(
                            step_pre,
                            step_post,
                            visibility_bucket=visibility_bucket,
                            near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
                            near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
                            close_ready_xy_threshold=float(args.close_ready_xy_threshold),
                            close_ready_yaw_threshold=float(args.close_ready_yaw_threshold),
                            close_ready_z_threshold=float(args.close_ready_z_threshold),
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
                    horizon_metric_fields = _prefix_grasp_probe_fields(
                        _grasp_probe_metric_fields(
                            pre_probe,
                            final_probe,
                            visibility_bucket=visibility_bucket,
                            near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
                        near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
                        close_ready_xy_threshold=float(args.close_ready_xy_threshold),
                        close_ready_yaw_threshold=float(args.close_ready_yaw_threshold),
                        close_ready_z_threshold=float(args.close_ready_z_threshold),
                        ),
                        "grasp_probe_horizon",
                    )
                    trace_entry.update(horizon_metric_fields)
                    strict_handoff_ready_for_eval_close = bool(
                        alignment_session.alignment_ready_for_handoff
                        or alignment_readiness.alignment_ready_for_handoff
                        or trace_entry.get("alignment_ready_for_handoff", False)
                    )
                    close_handoff_ready = bool(
                        horizon_metric_fields.get("grasp_probe_horizon_close_ready_after", False)
                        and strict_handoff_ready_for_eval_close
                    )
                    trace_entry["grasp_probe_close_handoff_ready"] = bool(close_handoff_ready)
                    trace_entry["grasp_probe_close_handoff_strict_ready"] = bool(strict_handoff_ready_for_eval_close)
                    if (
                        str(args.c2c_grasp_probe_gripper_mode) == "eval_close_after_near"
                        and close_handoff_ready
                        and int(args.c2c_grasp_probe_close_handoff_steps) > 0
                        and not bool(terminate)
                    ):
                        for _close_step in range(int(args.c2c_grasp_probe_close_handoff_steps)):
                            close_delta = np.zeros_like(delta_action, dtype=np.float32)
                            if close_delta.shape[0] > 6:
                                close_delta[6] = 0.0
                            close_abs = delta_to_absolute(close_delta, obs.gripper_pose)
                            close_abs, close_workspace_violation = maybe_apply_workspace_filter(
                                close_abs,
                                safety,
                                mode=args.workspace_clamp_mode,
                                tolerance=args.workspace_clamp_tolerance,
                            )
                            if close_workspace_violation > 0.0:
                                workspace_violation_count += 1
                                workspace_violation_max = max(workspace_violation_max, float(close_workspace_violation))
                            try:
                                obs, reward, terminate = task.step(close_abs)
                                executed_action = close_abs
                            except InvalidActionError as exc:
                                trace_entry["grasp_probe_close_handoff_invalid_action"] = type(exc).__name__
                                trace_entry["invalid_action"] = True
                                trace_entry["invalid_action_flag"] = True
                                invalid_action_count += 1
                                break
                            trace_entry["grasp_probe_eval_close_handoff"] = True
                            trace_entry["grasp_probe_close_locked"] = False
                            if bool(args.c2c_grasp_probe_latch_close_after_handoff):
                                probe_close_handoff_latched = True
                                trace_entry["grasp_probe_close_handoff_latched"] = True
                            trace_entry["grasp_probe_eval_close_handoff_steps_executed"] = int(
                                trace_entry["grasp_probe_eval_close_handoff_steps_executed"]
                            ) + 1
                            if args.record_video:
                                frames.append(_compose_video_frame(obs.front_rgb.copy(), obs.wrist_rgb.copy() if obs.wrist_rgb is not None else None, layout=args.video_layout))
                            if bool(terminate):
                                break
                else:
                    nan_error = np.full((4,), np.nan, dtype=np.float32)
                    trace_entry["grasp_probe_post_true_error_t"] = _jsonable_value(nan_error)
                    trace_entry["grasp_probe_horizon_final_true_error_t"] = _jsonable_value(nan_error)
                    trace_entry.update(_nan_grasp_probe_metric_fields())
                    trace_entry.update(_prefix_grasp_probe_fields(_nan_grasp_probe_metric_fields(), "grasp_probe_horizon"))
            executed_arr = np.asarray(executed_action, dtype=np.float32).reshape(-1)
            executed_gripper = float(executed_arr[-1]) if executed_arr.size > 6 else 1.0
            close_commanded = bool(
                bool(trace_entry.get("grasp_probe_eval_close_handoff", False))
                or bool(trace_entry.get("grasp_probe_close_handoff_latched", False))
                or (executed_arr.size > 6 and executed_gripper <= 0.0)
            )
            trace_entry["grasp_probe_executed_gripper_value"] = float(executed_gripper)
            observed_closed = bool(float(getattr(obs, "gripper_open", 1.0)) < 0.5)
            if observed_closed:
                probe_observed_closed_streak += 1
            else:
                probe_observed_closed_streak = 0
            close_observed_closed = bool(
                close_commanded
                and probe_observed_closed_streak >= int(max(1, int(args.c2c_grasp_probe_close_validation_closed_steps)))
            )
            close_success_reward = bool(close_commanded and float(reward) > 0.0)
            trace_entry["grasp_probe_close_commanded"] = bool(close_commanded)
            trace_entry["grasp_probe_observed_closed_after"] = bool(observed_closed)
            trace_entry["grasp_probe_observed_closed_streak"] = int(probe_observed_closed_streak)
            trace_entry["grasp_probe_close_handoff_observed_closed"] = bool(close_observed_closed)
            trace_entry["grasp_probe_close_handoff_success_reward"] = bool(close_success_reward)
            trace_entry["grasp_probe_close_handoff_validated"] = bool(close_observed_closed and close_success_reward)
            _attach_offline_eval_only(trace_entry, privileged_frame_pack)
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
            runtime_xy_history.append(dict(trace_entry))

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
