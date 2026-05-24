#!/usr/bin/env python3
"""Offline privileged basin relabel for C2C v2 frame contracts.

This script reads saved runtime observations and traces, then reconstructs
frame-to-frame basin residual labels offline only.  It never feeds privileged
poses back into the runtime controller.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.basin_recovery import classify_visual_evidence_for_basin
from prismatic.robot.coarse2contact_v2.recovery_augmentation import failure_morphology_bucket
from prismatic.robot.coarse2contact_v2.recovery_audit import in_close_ready_basin, in_near_grasp_basin
from prismatic.robot.stage_target_provider import apply_yaw_symmetry_to_delta, build_phase1_teacher_targets, load_phase1_grasp_spec, pose_delta_local_between
from prismatic.robot.residual_transforms import world_delta_to_local

NEAR_GRASP_XY_THRESHOLD = 0.015
NEAR_GRASP_YAW_THRESHOLD = 0.08
CLOSE_READY_XY_THRESHOLD = 0.005
CLOSE_READY_YAW_THRESHOLD = 0.03
CLOSE_READY_Z_THRESHOLD = 0.010
DEFAULT_PULLBACK_HORIZON = 3
DEFAULT_MAX_XY_STEP = 0.003
COARSE_PULLBACK_XY_THRESHOLD = 0.060


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("step", -1)))
    return rows


def _safe_pose(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size < 7 or not np.all(np.isfinite(arr[:7])):
        return None
    return arr[:7].astype(np.float32)


def _pose_or_nan(arr: np.ndarray | None) -> np.ndarray:
    pose = _safe_pose(arr)
    if pose is None:
        return np.full((7,), np.nan, dtype=np.float32)
    return pose


def _episode_pose(npz: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray | None:
    for key in keys:
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > 0:
            arr = arr[0]
        arr = arr.reshape(-1)
        if arr.size >= 7 and np.all(np.isfinite(arr[:7])):
            return arr[:7].astype(np.float32)
    return None


def _episode_arrays(npz: np.lib.npyio.NpzFile, *keys: str, fallback_shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    for key in keys:
        if key in npz.files:
            return np.asarray(npz[key], dtype=np.float32)
    if fallback_shape is None:
        return None
    return np.full(fallback_shape, np.nan, dtype=np.float32)


def _axis_value(trace_row: Mapping[str, Any], axis: str) -> float:
    geom = (trace_row.get("local_geometry_error") or {}).get("grasp") or {}
    est = trace_row.get("estimated_basin_error") or {}
    if axis == "x":
        return float(est.get("dx", geom.get("dx", 0.0)) or 0.0)
    if axis == "y":
        return float(est.get("dy", geom.get("dy", 0.0)) or 0.0)
    if axis == "z":
        return float(est.get("dz", geom.get("dz", 0.0)) or 0.0)
    if axis == "yaw":
        return float(est.get("dyaw", geom.get("dyaw", 0.0)) or 0.0)
    return 0.0


def _visual_record(trace_row: Mapping[str, Any]) -> dict[str, Any]:
    geom = (trace_row.get("local_geometry_error") or {}).get("grasp") or {}
    est = trace_row.get("estimated_basin_error") or {}
    conf = max(float(geom.get("confidence", 0.0) or 0.0), float(est.get("estimated_basin_error_confidence", est.get("confidence", 0.0)) or 0.0))
    obs = max(float(geom.get("observability", 0.0) or 0.0), float(est.get("frame_consistency", 0.0) or 0.0))
    axis = max(0.0, 1.0 - float(geom.get("fit_residual", 0.0) or 0.0))
    return {
        "frame_confidence": conf,
        "frame_observability": obs,
        "frame_axis_strength": axis,
        "wide_ring_visible": bool((not bool(trace_row.get("wrist_is_occluded", False))) and obs > 0.0),
    }


def _yaw_observability_class(trace_row: Mapping[str, Any], visual_record: Mapping[str, Any]) -> str:
    """Classify yaw observability from non-privileged runtime evidence only."""
    if bool(trace_row.get("wrist_is_occluded", False)):
        return "unobservable"
    conf = float(visual_record.get("frame_confidence", 0.0) or 0.0)
    obs = float(visual_record.get("frame_observability", 0.0) or 0.0)
    axis = float(visual_record.get("frame_axis_strength", 0.0) or 0.0)
    if conf >= 0.50 and obs >= 0.10 and axis >= 0.80:
        return "observable"
    if conf >= 0.05 or obs >= 0.005 or bool(visual_record.get("wide_ring_visible", False)):
        return "ambiguous"
    return "unobservable"


def _finite_residual(row: Mapping[str, Any]) -> bool:
    return all(np.isfinite(float(row.get(key, float("nan")))) for key in ("privileged_dx", "privileged_dy", "privileged_dz", "privileged_dyaw"))


def _near_basin_shell_from_error(row: Mapping[str, Any]) -> bool:
    xy = float(row.get("xy_error", float("nan")))
    yaw = float(row.get("yaw_abs", float("nan")))
    return bool(
        np.isfinite(xy)
        and np.isfinite(yaw)
        and xy <= NEAR_GRASP_XY_THRESHOLD + DEFAULT_MAX_XY_STEP * DEFAULT_PULLBACK_HORIZON + 1.0e-9
        and yaw <= NEAR_GRASP_YAW_THRESHOLD + 1.0e-9
    )


def _xy_contracted(row: Mapping[str, Any]) -> bool:
    xy = float(row.get("xy_error", float("nan")))
    next_xy = float(row.get("next_xy_error", float("nan")))
    return bool(np.isfinite(xy) and np.isfinite(next_xy) and next_xy < xy - 1.0e-9)


def _overshoot(row: Mapping[str, Any]) -> bool:
    for key in ("dx", "dy", "dyaw"):
        now = float(row.get("true_basin_error_t", {}).get(key, float("nan")))
        nxt = float(row.get("true_basin_error_t_plus_1", {}).get(key, float("nan")))
        if np.isfinite(now) and np.isfinite(nxt) and np.sign(now) != np.sign(nxt) and abs(nxt) >= abs(now):
            return True
    return False


def _takeover_tier(row: Mapping[str, Any]) -> str:
    if not _finite_residual(row):
        return "invalid"
    if str(row.get("visual_observability_class", "prior_only")) == "prior_only":
        return "abstain_prior_only"
    if bool(row.get("close_ready_ready", False)):
        return "close_ready"
    if bool(row.get("micro_entry_ready", False)):
        return "micro_entry_ready"
    if bool(row.get("near_basin_shell", False)):
        return "near_basin_shell"
    xy = float(row.get("xy_error", float("nan")))
    if np.isfinite(xy) and xy <= COARSE_PULLBACK_XY_THRESHOLD and _xy_contracted(row):
        return "coarse_pullback_candidate"
    return "outside_takeover"


def _label_invalid_reason(row: Mapping[str, Any], *, precision_row: bool) -> str:
    if not precision_row:
        return "not_precision_stage"
    if _finite_residual(row):
        return ""
    missing = [
        key
        for key in ("privileged_dx", "privileged_dy", "privileged_dz", "privileged_dyaw")
        if not np.isfinite(float(row.get(key, float("nan"))))
    ]
    return "missing_" + "+".join(missing)


def _normalized_estimated_basin_error(trace_row: Mapping[str, Any]) -> dict[str, Any]:
    est = trace_row.get("estimated_basin_error") or {}
    if not isinstance(est, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in est.items():
        if isinstance(key, str) and key.startswith("estimated_basin_error_"):
            out[key.removeprefix("estimated_basin_error_")] = value
        else:
            out[key] = value
    axis_validity = out.get("axis_validity", {})
    axis_confidence = out.get("axis_confidence", {})
    if isinstance(axis_validity, Mapping):
        out.setdefault("x_valid", bool(axis_validity.get("x", False)))
        out.setdefault("y_valid", bool(axis_validity.get("y", False)))
        out.setdefault("z_valid", bool(axis_validity.get("z", False)))
        out.setdefault("yaw_valid", bool(axis_validity.get("yaw", False)))
    if isinstance(axis_confidence, Mapping):
        out.setdefault("x_confidence", float(axis_confidence.get("x", 0.0)))
        out.setdefault("y_confidence", float(axis_confidence.get("y", 0.0)))
        out.setdefault("z_confidence", float(axis_confidence.get("z", 0.0)))
        out.setdefault("yaw_confidence", float(axis_confidence.get("yaw", 0.0)))
    return out


def _selected_proxy_geometry(trace_row: Mapping[str, Any]) -> dict[str, Any]:
    geom = trace_row.get("local_geometry_error") or {}
    if not isinstance(geom, Mapping):
        return {}
    skill_type = str(trace_row.get("c2c_v2_skill_type", trace_row.get("skill_type", "")))
    selected = geom.get("grasp") if skill_type == "precision_grasp" else geom.get("spoke") if skill_type == "precision_align" else None
    if isinstance(selected, Mapping):
        return dict(selected)
    return {}


def _frame_label_fields(
    *,
    task_name: str,
    trace_row: Mapping[str, Any],
    gripper_pose: np.ndarray,
    ring_pose: np.ndarray,
    spoke_pose: np.ndarray,
) -> dict[str, Any]:
    spec = load_phase1_grasp_spec(task_name)
    nan_pose = np.full((7,), np.nan, dtype=np.float32)
    grasp_pregrasp, grasp_commit = build_phase1_teacher_targets(ring_pose, spec) if np.all(np.isfinite(ring_pose[:7])) else (np.full((7,), np.nan, dtype=np.float32), np.full((7,), np.nan, dtype=np.float32))
    grasp_residual = (
        apply_yaw_symmetry_to_delta(pose_delta_local_between(gripper_pose, grasp_commit), float(spec.yaw_symmetry_period))
        if np.all(np.isfinite(gripper_pose[:7])) and np.all(np.isfinite(grasp_commit[:7]))
        else np.full((6,), np.nan, dtype=np.float32)
    )
    align_residual = (
        apply_yaw_symmetry_to_delta(pose_delta_local_between(ring_pose, spoke_pose), float(spec.yaw_symmetry_period))
        if np.all(np.isfinite(ring_pose[:7])) and np.all(np.isfinite(spoke_pose[:7]))
        else np.full((6,), np.nan, dtype=np.float32)
    )

    skill_type = str(trace_row.get("c2c_v2_skill_type", trace_row.get("skill_type", "")))
    stage_name = str(trace_row.get("c2c_v2_stage", trace_row.get("stage_name", "")))
    if skill_type == "precision_grasp":
        skill_name = "grasp_contact_ring" if "CONTACT" in stage_name else "precision_grasp_ring"
    elif skill_type == "precision_align":
        skill_name = "precision_align_ring_to_spoke"
    else:
        skill_name = str(trace_row.get("skill_name", skill_type or ""))
    precision_row = skill_type in {"precision_grasp", "precision_align"}
    reference_frame = "gripper_jaw_frame" if skill_type == "precision_grasp" else ("held_ring_aperture_frame" if skill_type == "precision_align" else "")
    target_frame = "ring_grasp_frame" if skill_type == "precision_grasp" else ("target_spoke_axis_frame" if skill_type == "precision_align" else "")
    label_pose = grasp_commit if skill_type == "precision_grasp" else (spoke_pose if skill_type == "precision_align" else nan_pose)
    raw_residual = grasp_residual if skill_type == "precision_grasp" else (align_residual if skill_type == "precision_align" else np.full((6,), np.nan, dtype=np.float32))
    if raw_residual.size < 6:
        raw_residual = np.pad(raw_residual, (0, max(0, 6 - raw_residual.size)))[:6]

    privileged_xy = float(np.linalg.norm(raw_residual[:2]))
    privileged_yaw_abs = float(abs(raw_residual[5])) if raw_residual.size >= 6 else float("nan")
    privileged_z = float(raw_residual[2]) if raw_residual.size >= 3 else float("nan")
    z_semantics = "descend_progress_to_grasp_frame" if skill_type == "precision_grasp" else "axis_alignment_depth"
    visual_record = _visual_record(trace_row)
    visual_obs = classify_visual_evidence_for_basin(visual_record)
    yaw_obs_class = "unobservable" if visual_obs.value == "prior_only" else _yaw_observability_class(trace_row, visual_record)
    try:
        skill_spec = spec.get_skill(skill_name)
    except Exception:
        skill_spec = None
    requires_yaw_observability = bool(getattr(skill_spec, "requires_yaw_observability", False)) if precision_row else False
    yaw_observable = bool(precision_row and yaw_obs_class == "observable")
    reacquire_needed = bool(precision_row and visual_obs == visual_obs.__class__.PRIOR_ONLY)
    pullback_allowed = bool(precision_row and not reacquire_needed)
    axis_gate_policy = {
        "x": "trusted_control" if pullback_allowed else "abstain",
        "y": "trusted_control" if pullback_allowed else "abstain",
        "z": "diagnostic_only" if pullback_allowed else "abstain",
        "yaw": "trusted_control" if yaw_observable else "abstain",
    }
    near_basin_shell = bool(
        precision_row
        and not reacquire_needed
        and np.isfinite(privileged_xy)
        and np.isfinite(privileged_yaw_abs)
        and privileged_xy <= NEAR_GRASP_XY_THRESHOLD + DEFAULT_MAX_XY_STEP * DEFAULT_PULLBACK_HORIZON + 1.0e-9
        and privileged_yaw_abs <= NEAR_GRASP_YAW_THRESHOLD + 1.0e-9
    )
    micro_entry_ready = bool(
        pullback_allowed
        and in_near_grasp_basin(float(raw_residual[0]), float(raw_residual[1]), float(raw_residual[5]), xy_threshold=NEAR_GRASP_XY_THRESHOLD, yaw_threshold=NEAR_GRASP_YAW_THRESHOLD)
        and (not requires_yaw_observability or yaw_observable)
    )
    close_ready_ready = bool(
        pullback_allowed
        and in_close_ready_basin(float(raw_residual[0]), float(raw_residual[1]), float(raw_residual[5]), xy_threshold=CLOSE_READY_XY_THRESHOLD, yaw_threshold=CLOSE_READY_YAW_THRESHOLD)
        and abs(float(privileged_z)) <= CLOSE_READY_Z_THRESHOLD
        and (not requires_yaw_observability or yaw_observable)
    )
    micro_entry_block_reason_parts = []
    if precision_row and reacquire_needed:
        micro_entry_block_reason_parts.append("prior_only")
    if precision_row and not (float(np.hypot(float(raw_residual[0]), float(raw_residual[1]))) <= NEAR_GRASP_XY_THRESHOLD):
        micro_entry_block_reason_parts.append("xy")
    if precision_row and requires_yaw_observability and not yaw_observable:
        micro_entry_block_reason_parts.append("yaw")
    micro_entry_block_reason = "+".join(micro_entry_block_reason_parts) if micro_entry_block_reason_parts else "ready"
    close_ready_block_reason_parts = []
    if precision_row and reacquire_needed:
        close_ready_block_reason_parts.append("prior_only")
    if precision_row and not (float(np.hypot(float(raw_residual[0]), float(raw_residual[1]))) <= CLOSE_READY_XY_THRESHOLD):
        close_ready_block_reason_parts.append("xy")
    if precision_row and abs(float(privileged_z)) > CLOSE_READY_Z_THRESHOLD:
        close_ready_block_reason_parts.append("z")
    if precision_row and requires_yaw_observability and not yaw_observable:
        close_ready_block_reason_parts.append("yaw")
    if not precision_row:
        micro_entry_block_reason_parts = ["not_precision"]
        close_ready_block_reason_parts = ["not_precision"]
        micro_entry_block_reason = "not_precision"
    close_ready_block_reason = "+".join(close_ready_block_reason_parts) if close_ready_block_reason_parts else "ready"

    estimated = trace_row.get("estimated_basin_error", {}) or {}
    proxy = trace_row.get("local_geometry_error", {}) or {}
    local_residual = np.asarray(trace_row.get("local_residual_vs_planner_local_6d", [0.0] * 6), dtype=np.float32).reshape(-1)
    local_residual = np.pad(local_residual, (0, max(0, 6 - local_residual.size)))[:6]
    planner_world = np.asarray(trace_row.get("planner_action_world", trace_row.get("planner_action_world_6d", [0.0] * 6)), dtype=np.float32).reshape(-1)
    planner_world = np.pad(planner_world, (0, max(0, 6 - planner_world.size)))[:6]
    planner_local = world_delta_to_local(planner_world, gripper_pose[3:7]).astype(np.float32)
    bucket_source = {
        "recovery_target_dx": float(raw_residual[0]),
        "recovery_target_dy": float(raw_residual[1]),
        "recovery_target_dyaw": float(raw_residual[5]) if raw_residual.size >= 6 else 0.0,
        "planner_bias_score": float(trace_row.get("planner_bias_score", 0.0) or 0.0),
    }
    failure_bucket = failure_morphology_bucket(bucket_source)

    label_valid = bool(precision_row and np.all(np.isfinite(raw_residual[[0, 1, 2, 5]])))
    label_invalid_reason = "" if label_valid else ("not_precision_stage" if not precision_row else "missing_privileged_residual")
    record = {
        "schema_version": "frame_residual_v2",
        "task_name": task_name,
        "frame_contract": {
            "target_frame": target_frame,
            "reference_frame": reference_frame,
            "error_frame": str(getattr(skill_spec, "error_frame", "reference_local")),
            "yaw_mode": str(getattr(skill_spec, "yaw_mode", "proxy_axis")),
            "z_semantics": z_semantics if precision_row else "none",
            "requires_yaw_observability": bool(requires_yaw_observability),
        },
        "obs_t": {
            "episode_idx": int(trace_row.get("episode_idx", trace_row.get("episode_index", -1))),
            "step_idx": int(trace_row.get("step", trace_row.get("step_idx", -1))),
            "visual_observability_class": visual_obs.value,
            "frame_confidence": float(visual_record["frame_confidence"]),
            "frame_observability": float(visual_record["frame_observability"]),
            "frame_axis_strength": float(visual_record["frame_axis_strength"]),
            "yaw_observability_class": str(yaw_obs_class),
            "source_phase_owner": str(trace_row.get("phase_owner", trace_row.get("c2c_v2_owner", ""))),
            "source_basin_recovery_mode": str(trace_row.get("basin_recovery_mode", "")),
            "source_localizer_abstained": bool(trace_row.get("localizer_abstained", False)),
            "uses_privileged_target": False,
            "uses_privileged_runtime": False,
            "uses_rlbench_mask_runtime": False,
        },
        "planner_prior": {
            "world_delta_6d": planner_world.tolist(),
            "local_delta_6d": planner_local.tolist(),
        },
        "stage_name": stage_name,
        "skill_name": skill_name,
        "skill_type": skill_type,
        "failure_bucket": failure_bucket,
        "target_frame": target_frame,
        "reference_frame": reference_frame,
        "reference_frame_pose_7d": _pose_or_nan(gripper_pose if skill_type == "precision_grasp" else ring_pose).tolist(),
        "target_frame_pose_7d": _pose_or_nan(label_pose).tolist(),
        "ring_frame_pose_7d": _pose_or_nan(ring_pose).tolist(),
        "spoke_frame_pose_7d": _pose_or_nan(spoke_pose).tolist(),
        "episode_idx": int(trace_row.get("episode_idx", trace_row.get("episode_index", -1))),
        "step_idx": int(trace_row.get("step", trace_row.get("step_idx", -1))),
        "privileged_dx": float(raw_residual[0]) if raw_residual.size >= 1 else float("nan"),
        "privileged_dy": float(raw_residual[1]) if raw_residual.size >= 2 else float("nan"),
        "privileged_dz": privileged_z,
        "privileged_dyaw": float(raw_residual[5]) if raw_residual.size >= 6 else float("nan"),
        "descend_progress_to_grasp_frame": privileged_z if skill_type == "precision_grasp" else float("nan"),
        "axis_alignment_depth": privileged_z if skill_type != "precision_grasp" else float("nan"),
        "xy_error": privileged_xy,
        "yaw_abs": privileged_yaw_abs,
        "yaw_observable": bool(yaw_observable),
        "yaw_observability_class": str(yaw_obs_class),
        "reacquire_needed": bool(reacquire_needed),
        "pullback_allowed": bool(pullback_allowed),
        "near_basin_shell": bool(near_basin_shell),
        "coarse_pullback_candidate": False,
        "takeover_tier": "pending_next_error",
        "label_valid": bool(label_valid),
        "label_invalid_reason": str(label_invalid_reason),
        "micro_entry_ready": bool(micro_entry_ready),
        "micro_entry_block_reason": str(micro_entry_block_reason),
        "close_ready_ready": bool(close_ready_ready),
        "close_ready_block_reason": str(close_ready_block_reason),
        "axis_gate_policy": axis_gate_policy,
        "true_basin_error_t": {
            "dx": float(raw_residual[0]) if raw_residual.size >= 1 else float("nan"),
            "dy": float(raw_residual[1]) if raw_residual.size >= 2 else float("nan"),
            "dz": privileged_z,
            "dyaw": float(raw_residual[5]) if raw_residual.size >= 6 else float("nan"),
        },
        "action_t": {
            "local_correction_local_6d": np.asarray(trace_row.get("local_correction_local_6d", local_residual.tolist()), dtype=np.float32).reshape(-1)[:6].tolist(),
            "planner_local_delta_6d": planner_local.tolist(),
        },
        "near_grasp_basin": bool(in_near_grasp_basin(float(raw_residual[0]), float(raw_residual[1]), float(raw_residual[5]), xy_threshold=NEAR_GRASP_XY_THRESHOLD, yaw_threshold=NEAR_GRASP_YAW_THRESHOLD)),
        "close_ready_basin": bool(in_close_ready_basin(float(raw_residual[0]), float(raw_residual[1]), float(raw_residual[5]), xy_threshold=CLOSE_READY_XY_THRESHOLD, yaw_threshold=CLOSE_READY_YAW_THRESHOLD)),
        "near_insert_basin": False,
        "true_basin_error_t_plus_1": {
            "dx": float("nan"),
            "dy": float("nan"),
            "dz": float("nan"),
            "dyaw": float("nan"),
        },
        "next_privileged_dx": float("nan"),
        "next_privileged_dy": float("nan"),
        "next_privileged_dz": float("nan"),
        "next_privileged_dyaw": float("nan"),
        "next_xy_error": float("nan"),
        "next_yaw_abs": float("nan"),
        "proxy_local_geometry_error": _selected_proxy_geometry(trace_row),
        "proxy_local_geometry_error_raw": proxy,
        "estimated_basin_error": _normalized_estimated_basin_error(trace_row),
        "estimated_basin_error_raw": estimated,
        "planner_local_delta_6d": planner_local.tolist(),
        "local_residual_vs_planner_local_6d": local_residual.tolist(),
        "local_correction_local_6d": np.asarray(trace_row.get("local_correction_local_6d", local_residual.tolist()), dtype=np.float32).reshape(-1)[:6].tolist(),
        "visual_observability_class": visual_obs.value,
        "z_semantics": z_semantics,
        "label_source": "privileged_pose_offline",
        "uses_privileged_label": True,
        "uses_privileged_runtime": False,
        "uses_privileged_target": False,
        "uses_rlbench_mask_runtime": False,
        "source_trace_path": str(trace_row.get("source_trace_path", "")),
        "source_runtime_obs_path": str(trace_row.get("runtime_obs_path", "")),
        "source_phase_owner": str(trace_row.get("phase_owner", trace_row.get("c2c_v2_owner", ""))),
        "source_basin_recovery_mode": str(trace_row.get("basin_recovery_mode", "")),
        "source_c2c_stage": str(trace_row.get("c2c_v2_stage", "")),
        "source_localizer_abstained": bool(trace_row.get("localizer_abstained", False)),
        "source_frame_confidence": float(visual_record["frame_confidence"]),
        "source_frame_observability": float(visual_record["frame_observability"]),
        "source_frame_axis_strength": float(visual_record["frame_axis_strength"]),
    }
    record["takeover_tier"] = _takeover_tier(record)
    record["coarse_pullback_candidate"] = bool(record["takeover_tier"] == "coarse_pullback_candidate")
    record["contraction"] = False
    record["xy_contracted"] = False
    record["overshoot"] = False
    return record


def _plot_episode(ep_tag: str, rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    steps = [int(r["step_idx"]) for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for ax, axis_name in zip(axes, ["x", "y", "z", "yaw"]):
        priv = [float(r[f"privileged_d{axis_name}" if axis_name != "yaw" else "privileged_dyaw"]) if axis_name != "yaw" else float(r["privileged_dyaw"]) for r in rows]
        proxy = []
        est = []
        plan = []
        for r in rows:
            proxy_geom = r.get("proxy_local_geometry_error", {}) or {}
            est_geom = r.get("estimated_basin_error", {}) or {}
            proxy.append(float(proxy_geom.get("dx" if axis_name == "x" else "dy" if axis_name == "y" else "dz" if axis_name == "z" else "dyaw", 0.0) or 0.0))
            est.append(float(est_geom.get("dx" if axis_name == "x" else "dy" if axis_name == "y" else "dz" if axis_name == "z" else "dyaw", 0.0) or 0.0))
            plan.append(float(np.asarray(r.get("planner_local_delta_6d", [0.0] * 6), dtype=np.float32).reshape(-1)[0 if axis_name == "x" else 1 if axis_name == "y" else 2 if axis_name == "z" else 5]))
        ax.plot(steps, priv, label="privileged", linewidth=1.8)
        ax.plot(steps, proxy, label="proxy", linewidth=1.2)
        ax.plot(steps, est, label="estimated", linewidth=1.2)
        ax.plot(steps, plan, label="planner_local", linewidth=1.0, linestyle="--")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(axis_name)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.suptitle(f"{ep_tag} privileged basin relabel")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ep_tag}_privileged_basin_relabel.png", dpi=160)
    plt.close(fig)


def evaluate_root(eval_root: Path, task_name: str, output_dir: Path) -> dict[str, Any]:
    results_path = eval_root / "eval_results.json"
    trace_dir = eval_root / "gripper_traces"
    obs_dir = eval_root / "runtime_observations"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing eval_results.json under {eval_root}")
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing gripper_traces under {eval_root}")
    if not obs_dir.exists():
        raise FileNotFoundError(f"Missing runtime_observations under {eval_root}; rerun with --dump_runtime_obs")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = _read_json(results_path)
    stage_stats = list(results.get("stage_stats", []))
    all_rows: list[dict[str, Any]] = []
    per_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()

    for trace_path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep_tag = trace_path.stem.split("_")[0]
        ep_idx = int(ep_tag.replace("ep", ""))
        obs_path = obs_dir / f"{ep_tag}_runtime_obs.npz"
        if not obs_path.exists():
            continue
        trace_rows = _trace_rows(trace_path)
        if not trace_rows:
            continue
        with np.load(obs_path, allow_pickle=True) as obs_npz:
            ring_pose = _episode_pose(obs_npz, "episode_ring_pose_7d", "episode_target_pose_7d")
            spoke_pose = _episode_pose(obs_npz, "episode_spoke_pose_7d", "episode_task_low_dim_pose_7d")
            gripper_poses = _episode_arrays(obs_npz, "episode_gripper_pose_7d", "gripper_pose", fallback_shape=(len(trace_rows), 7))
            if ring_pose is None:
                ring_pose = np.full((7,), np.nan, dtype=np.float32)
            if spoke_pose is None:
                spoke_pose = np.full((7,), np.nan, dtype=np.float32)
            rows: list[dict[str, Any]] = []
            for row in trace_rows:
                step_idx = int(row.get("step", -1))
                if step_idx < 0 or gripper_poses is None or step_idx >= int(np.asarray(gripper_poses).shape[0]):
                    continue
                gripper_pose = np.asarray(gripper_poses[step_idx], dtype=np.float32).reshape(-1)[:7]
                if not np.all(np.isfinite(gripper_pose[:7])):
                    continue
                record = _frame_label_fields(
                    task_name=task_name,
                    trace_row=row,
                    gripper_pose=gripper_pose,
                    ring_pose=ring_pose,
                    spoke_pose=spoke_pose,
                )
                record["episode_idx"] = int(ep_idx)
                record["step_idx"] = int(step_idx)
                record["source_trace_path"] = str(trace_path)
                record["source_runtime_obs_path"] = str(obs_path)
                rows.append(record)
            for idx, record in enumerate(rows):
                next_record = rows[idx + 1] if idx + 1 < len(rows) else None
                if next_record is not None:
                    record["next_privileged_dx"] = float(next_record["privileged_dx"])
                    record["next_privileged_dy"] = float(next_record["privileged_dy"])
                    record["next_privileged_dz"] = float(next_record["privileged_dz"])
                    record["next_privileged_dyaw"] = float(next_record["privileged_dyaw"])
                    record["next_xy_error"] = float(next_record["xy_error"])
                    record["next_yaw_abs"] = float(next_record["yaw_abs"])
                    record["true_basin_error_t_plus_1"] = {
                        "dx": float(next_record["privileged_dx"]),
                        "dy": float(next_record["privileged_dy"]),
                        "dz": float(next_record["privileged_dz"]),
                        "dyaw": float(next_record["privileged_dyaw"]),
                    }
                record["contraction"] = bool(_xy_contracted(record))
                record["xy_contracted"] = bool(_xy_contracted(record))
                record["overshoot"] = bool(_overshoot(record))
                record["near_basin_shell"] = bool(_near_basin_shell_from_error(record))
                record["takeover_tier"] = _takeover_tier(record)
                record["coarse_pullback_candidate"] = bool(record["takeover_tier"] == "coarse_pullback_candidate")
                record["label_invalid_reason"] = _label_invalid_reason(record, precision_row=bool(record.get("skill_type") in {"precision_grasp", "precision_align"}))
                record["label_valid"] = bool(record["label_invalid_reason"] == "")
                all_rows.append(record)
                key = (str(record["stage_name"]), str(record["skill_name"]), str(record["visual_observability_class"]))
                per_group[key].append(record)
            _plot_episode(ep_tag, rows, plot_dir)

    if not all_rows:
        raise RuntimeError(f"No relabeled rows found under {eval_root}")

    out_jsonl = output_dir / "basin_residual_labels.jsonl"
    out_v2_jsonl = output_dir / "frame_residual_v2.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with open(out_v2_jsonl, "w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _corr(a: list[float], b: list[float]) -> float:
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        mask = np.isfinite(aa) & np.isfinite(bb)
        if np.count_nonzero(mask) < 2:
            return 0.0
        aa = aa[mask]
        bb = bb[mask]
        if np.std(aa) <= 1e-9 or np.std(bb) <= 1e-9:
            return 0.0
        return float(np.corrcoef(aa, bb)[0, 1])

    axis_summary: dict[str, dict[str, Any]] = {}
    for axis in ["x", "y", "z", "yaw"]:
        proxy_vals: list[float] = []
        est_vals: list[float] = []
        priv_vals: list[float] = []
        action_vals: list[float] = []
        next_priv_vals: list[float] = []
        sign_match = []
        action_sign_match = []
        contraction = []
        overshoot = []
        mono_prefix = []
        for row in all_rows:
            proxy = row.get("proxy_local_geometry_error", {}) or {}
            est = row.get("estimated_basin_error", {}) or {}
            axis_key = "dyaw" if axis == "yaw" else axis
            proxy_v = float(proxy.get(axis_key, 0.0) or 0.0)
            est_v = float(est.get(axis_key, proxy_v) or proxy_v)
            priv_v = float(row[f"privileged_dyaw" if axis == "yaw" else f"privileged_d{axis}"])
            action_v = float(np.asarray(row.get("local_residual_vs_planner_local_6d", [0.0] * 6), dtype=np.float32).reshape(-1)[0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 5])
            next_v = float(row[f"next_privileged_dyaw" if axis == "yaw" else f"next_privileged_d{axis}"]) if np.isfinite(float(row[f"next_privileged_dyaw" if axis == "yaw" else f"next_privileged_d{axis}"])) else float("nan")
            proxy_vals.append(proxy_v)
            est_vals.append(est_v)
            priv_vals.append(priv_v)
            action_vals.append(action_v)
            next_priv_vals.append(next_v)
            if np.isfinite(proxy_v) and np.isfinite(priv_v) and abs(priv_v) > 1e-6 and abs(proxy_v) > 1e-6:
                sign_match.append(1.0 if np.sign(proxy_v) == np.sign(priv_v) else 0.0)
            if np.isfinite(action_v) and np.isfinite(priv_v) and abs(priv_v) > 1e-6 and abs(action_v) > 1e-6:
                action_sign_match.append(1.0 if np.sign(action_v) == np.sign(priv_v) else 0.0)
            if np.isfinite(next_v) and np.isfinite(priv_v):
                contraction.append(1.0 if abs(next_v) <= abs(priv_v) + 1e-9 else 0.0)
            if np.isfinite(next_v) and np.isfinite(priv_v):
                overshoot.append(1.0 if np.sign(next_v) != np.sign(priv_v) and abs(next_v) >= abs(priv_v) else 0.0)
        axis_summary[axis] = {
            "num_samples": int(len(priv_vals)),
            "sign_match_rate": float(np.mean(sign_match)) if sign_match else 0.0,
            "action_sign_match_rate": float(np.mean(action_sign_match)) if action_sign_match else 0.0,
            "contraction_rate": float(np.mean(contraction)) if contraction else 0.0,
            "overshoot_rate": float(np.mean(overshoot)) if overshoot else 0.0,
            "proxy_priv_corr": _corr(proxy_vals, priv_vals),
            "estimated_priv_corr": _corr(est_vals, priv_vals),
            "action_priv_corr": _corr(action_vals, priv_vals),
            "privileged_next_corr": _corr(priv_vals, next_priv_vals),
            "recommended_policy": "trusted_control"
            if (np.mean(sign_match) if sign_match else 0.0) >= 0.70
            and (np.mean(contraction) if contraction else 0.0) >= 0.70
            and abs(_corr(proxy_vals, priv_vals)) >= 0.25
            else ("diagnostic_only" if (np.mean(sign_match) if sign_match else 0.0) >= 0.40 else "abstain"),
        }

    group_summary: list[dict[str, Any]] = []
    for (stage_name, skill_name, vis_cls), rows in sorted(per_group.items()):
        group_summary.append(
            {
                "stage_name": stage_name,
                "skill_name": skill_name,
                "visual_observability_class": vis_cls,
                "failure_bucket_counts": dict(Counter(r["failure_bucket"] for r in rows)),
                "num_rows": len(rows),
                "near_grasp_rate": float(np.mean([bool(r["near_grasp_basin"]) for r in rows])),
                "close_ready_rate": float(np.mean([bool(r["close_ready_basin"]) for r in rows])),
                "near_basin_shell_rate": float(np.mean([bool(r["near_basin_shell"]) for r in rows])),
                "coarse_pullback_candidate_rate": float(np.mean([bool(r["coarse_pullback_candidate"]) for r in rows])),
                "takeover_tier_counts": dict(Counter(str(r["takeover_tier"]) for r in rows)),
                "yaw_observability_counts": dict(Counter(str(r["yaw_observability_class"]) for r in rows)),
                "mean_xy_error": float(np.mean([float(r["xy_error"]) for r in rows])),
                "mean_yaw_abs": float(np.mean([float(r["yaw_abs"]) for r in rows])),
            }
        )

    counts["episodes"] = len({int(r["episode_idx"]) for r in all_rows})
    counts["rows"] = len(all_rows)
    counts["near_grasp_hits"] = int(sum(1 for r in all_rows if bool(r["near_grasp_basin"])))
    counts["close_ready_hits"] = int(sum(1 for r in all_rows if bool(r["close_ready_basin"])))
    counts["near_basin_shell_rows"] = int(sum(1 for r in all_rows if bool(r["near_basin_shell"])))
    counts["coarse_pullback_candidate_rows"] = int(sum(1 for r in all_rows if bool(r["coarse_pullback_candidate"])))
    counts["micro_entry_ready_rows"] = int(sum(1 for r in all_rows if bool(r["micro_entry_ready"])))
    counts["yaw_observable_rows"] = int(sum(1 for r in all_rows if str(r["yaw_observability_class"]) == "observable"))
    counts["label_valid_rows"] = int(sum(1 for r in all_rows if bool(r["label_valid"])))

    manifest = {
        "schema_version": "frame_residual_v2",
        "task_name": task_name,
        "rows": len(all_rows),
        "jsonl": str(out_v2_jsonl),
        "legacy_jsonl": str(out_jsonl),
        "inputs": {
            "eval_root": str(eval_root),
            "trace_dir": str(trace_dir),
            "runtime_observations_dir": str(obs_dir),
        },
        "label_columns": [
            "frame_contract",
            "obs_t",
            "planner_prior",
            "true_basin_error_t",
            "true_basin_error_t_plus_1",
            "yaw_observability_class",
            "takeover_tier",
            "contraction",
            "overshoot",
        ],
        "runtime_invariants": {
            "uses_privileged_target": False,
            "uses_privileged_runtime": False,
            "uses_privileged_label": True,
            "uses_rlbench_mask_runtime": False,
        },
    }
    manifest_path = output_dir / "frame_residual_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "task_name": task_name,
        "eval_root": str(eval_root),
        "output_dir": str(output_dir),
        "rows": len(all_rows),
        "counts": dict(counts),
        "frame_residual_v2_jsonl": str(out_v2_jsonl),
        "frame_residual_v2_manifest": str(manifest_path),
        "axis_summary": axis_summary,
        "group_summary": group_summary,
        "runtime_invariants": {
            "uses_privileged_target": False,
            "uses_privileged_runtime": False,
            "uses_privileged_label": True,
            "uses_rlbench_mask_runtime": False,
        },
    }
    out_json = output_dir / "frame_contract_relabel.json"
    out_md = output_dir / "frame_contract_relabel.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [
        f"# Frame Contract Relabel Report",
        "",
        f"- task: `{task_name}`",
        f"- eval_root: `{eval_root}`",
        f"- rows: `{len(all_rows)}`",
        "",
        "## Axis Summary",
    ]
    for axis, stats in axis_summary.items():
        md_lines.append(
            f"- `{axis}`: policy={stats['recommended_policy']}, sign_match={stats['sign_match_rate']:.3f}, "
            f"contraction={stats['contraction_rate']:.3f}, proxy_corr={stats['proxy_priv_corr']:.3f}, "
            f"action_corr={stats['action_priv_corr']:.3f}"
        )
    md_lines.append("")
    md_lines.append("## Group Summary")
    for item in group_summary:
        md_lines.append(
            f"- {item['stage_name']} / {item['skill_name']} / {item['visual_observability_class']}: "
            f"rows={item['num_rows']}, near_grasp={item['near_grasp_rate']:.3f}, "
            f"near_shell={item['near_basin_shell_rate']:.3f}, coarse={item['coarse_pullback_candidate_rate']:.3f}, "
            f"close_ready={item['close_ready_rate']:.3f}"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(out_json)
    print(out_md)
    print(out_jsonl)
    print(out_v2_jsonl)
    print(manifest_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=Path, required=True)
    ap.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/frame_contract_relabel"),
    )
    args = ap.parse_args()
    evaluate_root(args.eval_root.resolve(), args.task_name, args.output_dir.resolve())


if __name__ == "__main__":
    main()
