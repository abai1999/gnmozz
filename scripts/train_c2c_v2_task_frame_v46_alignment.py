#!/usr/bin/env python3
"""Train the v46 unified spatial-temporal task-frame alignment candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.runtime_xy_residual import (  # noqa: E402
    DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    _load_spatial_temporal_rgbd,
    runtime_xy_spatial_temporal_context_feature_vector_from_trace,
    runtime_xy_spatial_temporal_feature_names,
)
from prismatic.robot.coarse2contact_v2.task_frame_readiness import TASK_FRAME_READINESS_FEATURE_NAMES  # noqa: E402
from prismatic.robot.coarse2contact_v2.task_frame_v46_alignment import (  # noqa: E402
    TASK_FRAME_V46_RISK_CLASSES,
    TaskFrameV46AlignmentNet,
    save_task_frame_v46_alignment_checkpoint,
    task_frame_v46_labels_from_row,
    task_frame_v46_scalar_feature_vector,
)
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import (  # noqa: E402
    source_eval_root_key,
    split_records_by_source_root,
)


TASK_FRAME_V46_RESIDUAL_SCALES = torch.tensor([0.040, 0.040, 0.030, 0.250], dtype=torch.float32)
PRIVILEGED_RUNTIME_KEYS = {
    "privileged_frame_pack",
    "teacher_target_pose",
    "target_pose",
    "success_pose",
    "rlbench_object_handle",
    "rlbench_mask",
    "gt_mask",
    "grasp_probe_pre_true_error",
    "grasp_probe_pre_true_error_t",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _trace_row_for_step(row: Mapping[str, Any], *, cache: dict[Path, list[dict[str, Any]]]) -> Mapping[str, Any]:
    path_value = row.get("trace_path", row.get("source_trace_path", ""))
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    if path not in cache:
        cache[path] = _read_jsonl(path)
    rows = cache[path]
    if not rows:
        return {}
    idx = int(np.clip(int(row.get("step_idx", row.get("step", 0)) or 0), 0, len(rows) - 1))
    return rows[idx]


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in paths:
        files = sorted(item.glob("*.jsonl")) if item.is_dir() else [item]
        for path in files:
            rows.extend(_read_jsonl(path))
    rows.sort(key=lambda row: (source_eval_root_key(row), int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1)))))
    return rows


def _normalize_row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    obs_pointer = out.get("obs_pointer", {})
    obs_pointer = obs_pointer if isinstance(obs_pointer, Mapping) else {}
    if "step_idx" not in out and "step" in out:
        out["step_idx"] = out.get("step")
    if "runtime_obs_path" not in out and obs_pointer.get("runtime_obs_path"):
        out["runtime_obs_path"] = obs_pointer.get("runtime_obs_path")
    if "trace_path" not in out and obs_pointer.get("trace_path"):
        out["trace_path"] = obs_pointer.get("trace_path")
    if "trace_path" not in out and out.get("source_trace_path"):
        out["trace_path"] = out.get("source_trace_path")
    root = str(out.get("source_eval_root", "") or "")
    if not root:
        path = _runtime_npz_path_from_row(out)
        if path is not None:
            root = str(path.parent.parent if path.parent.name == "runtime_observations" else path.parent)
            out["source_eval_root"] = root
    if "sequence_id" not in out:
        out["sequence_id"] = f"{root or 'unknown'}::ep{int(out.get('episode_idx', -1)):03d}"
    return out


def _row_has_privileged_runtime(row: Mapping[str, Any]) -> bool:
    runtime = row.get("runtime_features", {})
    runtime = runtime if isinstance(runtime, Mapping) else {}
    for key in PRIVILEGED_RUNTIME_KEYS:
        if key in row or key in runtime:
            return True
    return bool(row.get("uses_privileged_runtime", False))


def _array_at_step(npz: Mapping[str, Any], keys: tuple[str, ...], step_idx: int) -> np.ndarray | None:
    for key in keys:
        if key not in npz:
            continue
        arr = np.asarray(npz[key])
        if arr.ndim >= 4:
            idx = int(np.clip(step_idx, 0, arr.shape[0] - 1))
            return np.asarray(arr[idx])
        if arr.ndim == 3 and arr.shape[-1] not in {1, 3, 4}:
            idx = int(np.clip(step_idx, 0, arr.shape[0] - 1))
            return np.asarray(arr[idx])
        if arr.ndim == 2 and arr.shape[1] <= 64 and arr.shape[0] > 1:
            idx = int(np.clip(step_idx, 0, arr.shape[0] - 1))
            return np.asarray(arr[idx])
        if arr.ndim >= 2:
            return arr
    return None


def _runtime_npz_path_from_row(row: Mapping[str, Any]) -> Path | None:
    obs_pointer = row.get("obs_pointer", {})
    obs_pointer = obs_pointer if isinstance(obs_pointer, Mapping) else {}
    path_value = row.get(
        "npz_path",
        row.get(
            "runtime_obs_path",
            row.get(
                "runtime_observation_npz",
                row.get("source_runtime_obs_path", obs_pointer.get("runtime_obs_path", "")),
            ),
        ),
    )
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    return path


def _load_runtime_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {str(key): np.asarray(npz[key]) for key in npz.files}


def _load_runtime_npz_row(
    row: Mapping[str, Any],
    *,
    cache: dict[Path, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None] | None:
    path = _runtime_npz_path_from_row(row)
    if path is None:
        return None
    step_idx = int(row.get("step_idx", row.get("step", 0)) or 0)
    payload: Mapping[str, np.ndarray]
    if cache is not None:
        if path not in cache:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[path] = _load_runtime_npz_payload(path)
        payload = cache[path]
    else:
        payload = _load_runtime_npz_payload(path)
    wrist_rgb = _array_at_step(payload, ("wrist_rgb", "wrist_image", "wrist_rgb_uint8"), step_idx)
    wrist_depth = _array_at_step(payload, ("wrist_depth", "wrist_depth_float", "wrist_depth_m"), step_idx)
    front_rgb = _array_at_step(payload, ("front_rgb", "front_image", "front_rgb_uint8"), step_idx)
    front_depth = _array_at_step(payload, ("front_depth", "front_depth_float", "front_depth_m"), step_idx)
    proprio = _array_at_step(payload, ("proprio", "robot_proprio"), step_idx)
    planner = _array_at_step(payload, ("planner_action_world_6d", "planner_local_delta_6d", "pre_clip_action_world_6d"), step_idx)
    obs: dict[str, Any] = {}
    if wrist_rgb is not None:
        obs["wrist_rgb"] = wrist_rgb
    if wrist_depth is not None:
        obs["wrist_depth"] = wrist_depth
    if front_rgb is not None:
        obs["front_rgb"] = front_rgb
    if front_depth is not None:
        obs["front_depth"] = front_depth
    if not (("wrist_rgb" in obs or "front_rgb" in obs) and ("wrist_depth" in obs or "front_depth" in obs)):
        return None
    return obs, proprio, planner


def _coerce_vector(value: Any, length: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.asarray([], dtype=np.float32)
    if arr.size < int(length):
        arr = np.concatenate([arr.astype(np.float32), np.zeros((int(length) - arr.size,), dtype=np.float32)], axis=0)
    return arr[: int(length)].astype(np.float32)


def _risk_target(row: Mapping[str, Any], labels: Mapping[str, Any]) -> int:
    if bool(row.get("wrist_is_occluded", False)):
        name = "occlusion"
    elif bool(row.get("wrist_is_low_visibility", False)):
        name = "low_visibility"
    elif bool(labels.get("yaw_ambiguous", False)):
        name = "direction_conflict"
    elif not bool(labels.get("xy_observable", True)) or not bool(labels.get("z_observable", True)) or not bool(labels.get("yaw_observable", True)):
        name = "insufficient_support"
    else:
        name = "normal"
    return int({name: idx for idx, name in enumerate(TASK_FRAME_V46_RISK_CLASSES)}.get(name, 0))


def _sample_weight(row: Mapping[str, Any]) -> float:
    weight = 1.0
    bucket = str(row.get("failure_bucket", row.get("bucket", "")) or "")
    obs_bucket = str(row.get("observability_bucket", "") or "")
    yaw_label = row.get("yaw_label", {})
    yaw_label = yaw_label if isinstance(yaw_label, Mapping) else {}
    yaw_class = str(row.get("yaw_observability_class", yaw_label.get("yaw_observability_class", "")) or "").strip().lower()
    if bucket in {"large_xy_large_yaw", "small_xy_large_yaw", "worsen_tail"}:
        weight *= 1.8
    if obs_bucket in {"occluded", "low_observability", "low_visibility", "partial_observable", "partial_observation"}:
        weight *= 1.7
    if bool(row.get("yaw_control_observable", row.get("yaw_observable", yaw_label.get("yaw_control_observable", False)))) or yaw_class == "observable":
        weight *= 10.0
    elif yaw_class == "ambiguous":
        weight *= 2.0
    if bool(row.get("grasp_probe_active", False)) or bool(row.get("c2c_active", False)):
        weight *= 1.3
    if bool(row.get("task_frame_v46_applied", False)) or str(row.get("schema_version", "")).startswith("c2c_v2_task_frame_applied_transition"):
        weight *= 12.0
    if bool(row.get("c2c_worsen", row.get("worsen", False))):
        weight *= 2.0
    if bool(row.get("xy_worsen_observed", False)) or bool(row.get("y_worsen_observed", False)):
        weight *= 6.0
    labels = row.get("offline_labels", {})
    labels = labels if isinstance(labels, Mapping) else {}
    dz_value = labels.get("dz", row.get("privileged_dz", None))
    dx_value = labels.get("dx", row.get("privileged_dx", None))
    dy_value = labels.get("dy", row.get("privileged_dy", None))
    try:
        z_abs = abs(float(dz_value))
        xy_norm = float(np.hypot(float(dx_value), float(dy_value)))
    except Exception:
        z_abs = 0.0
        xy_norm = 0.0
    if z_abs > 0.08:
        weight *= 8.0
    if z_abs > 0.04 and xy_norm <= 0.06:
        weight *= 4.0
    if bool(row.get("nearfield_label", False)):
        weight *= 8.0
    if str(row.get("failure_bucket", "")) in {"nearfield_far_z_negative", "nearfield_boundary_negative"}:
        weight *= 4.0
    if "false_activation" in str(row.get("source_eval_root", "")) or "v56_transition_guarded" in str(row.get("source_eval_root", "")):
        weight *= 4.0
    return float(weight)


def _planner_prior_from_row(row: Mapping[str, Any], npz_planner: np.ndarray | None, planner_prior_dim: int) -> np.ndarray:
    planner_prior = row.get("planner_prior", {})
    planner_prior = planner_prior if isinstance(planner_prior, Mapping) else {}
    value = row.get("planner_local_delta_6d", row.get("planner_prior_delta", planner_prior.get("local_delta_6d", None)))
    if value is None:
        value = npz_planner
    return _coerce_vector(value, planner_prior_dim)


def _source_root_from_row(row: Mapping[str, Any]) -> str:
    root = source_eval_root_key(row)
    if root:
        return str(root)
    path = _runtime_npz_path_from_row(row)
    if path is None:
        return ""
    return str(path.parent.parent if path.parent.name == "runtime_observations" else path.parent)


def _label_within_support(labels: Mapping[str, Any], *, max_abs_xy_label: float, max_abs_z_label: float, max_abs_yaw_label: float) -> bool:
    xy = float(np.hypot(float(labels["dx"]), float(labels["dy"])))
    return bool(
        np.isfinite(xy)
        and xy <= float(max_abs_xy_label)
        and abs(float(labels["dz"])) <= float(max_abs_z_label)
        and abs(float(labels["dyaw"])) <= float(max_abs_yaw_label)
    )


def _next_residual_from_row(row: Mapping[str, Any]) -> tuple[list[float], float]:
    values: list[float] = []
    for key in ("dx", "dy", "dz", "dyaw"):
        value = float("nan")
        for candidate in (row.get(f"next_privileged_{key}", None), row.get(f"privileged_next_{key}", None)):
            try:
                value = float(candidate)
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                break
        values.append(float(value))
    has_next = bool(np.all(np.isfinite(np.asarray(values, dtype=np.float32))))
    if not has_next:
        values = [0.0, 0.0, 0.0, 0.0]
    return values, float(has_next)


def _command_xy_from_row(row: Mapping[str, Any], trace_row: Mapping[str, Any]) -> tuple[list[float], float]:
    value = row.get(
        "applied_control_command_xy",
        row.get(
            "applied_control_command_local_6d",
            row.get(
                "grasp_probe_local_command_local_6d",
                trace_row.get(
                    "grasp_probe_local_command_local_6d",
                    row.get("local_command_local_6d", row.get("planner_local_delta_6d", None)),
                ),
            ),
        ),
    )
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.asarray([], dtype=np.float32)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return [0.0, 0.0], 0.0
    return [float(arr[0]), float(arr[1])], float(np.linalg.norm(arr[:2]) > 1.0e-7)


def _command_6d_from_row(row: Mapping[str, Any], trace_row: Mapping[str, Any]) -> tuple[list[float], float]:
    value = row.get(
        "applied_control_command_local_6d",
        row.get(
            "grasp_probe_local_command_local_6d",
            trace_row.get("grasp_probe_local_command_local_6d", row.get("local_command_local_6d", row.get("planner_local_delta_6d", None))),
        ),
    )
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.asarray([], dtype=np.float32)
    if arr.size < 6:
        arr = np.concatenate([arr.astype(np.float32), np.zeros((6 - arr.size,), dtype=np.float32)], axis=0)
    arr = arr[:6].astype(np.float32)
    if not np.all(np.isfinite(arr)):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0
    if "applied_control_command_xy" in row:
        xy = _coerce_vector(row.get("applied_control_command_xy"), 2)
        arr[:2] = xy[:2]
    return [float(v) for v in arr.tolist()], float(np.linalg.norm(arr) > 1.0e-7)


def _yaw_hypothesis_target(dyaw: float, *, yaw_hypotheses: int = 4, max_abs_yaw: float = 0.350) -> int:
    if not np.isfinite(float(dyaw)):
        return 0
    bins = int(max(1, yaw_hypotheses))
    clipped = float(np.clip(float(dyaw), -float(max_abs_yaw), float(max_abs_yaw)))
    normalized = (clipped + float(max_abs_yaw)) / max(1.0e-6, 2.0 * float(max_abs_yaw))
    return int(np.clip(np.floor(normalized * bins), 0, bins - 1))


def _build_arrays(
    rows: list[dict[str, Any]],
    *,
    image_crop_size: int,
    image_resize_size: int,
    history_window_size: int,
    proprio_dim: int,
    planner_prior_dim: int,
    max_abs_xy_label: float,
    max_abs_z_label: float,
    max_abs_yaw_label: float,
    near_field_xy_radius: float = 0.060,
    near_field_z_radius: float = 0.040,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    images: list[np.ndarray] = []
    scalar: list[np.ndarray] = []
    history: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    planner: list[np.ndarray] = []
    labels_out: list[list[float]] = []
    next_labels_out: list[list[float]] = []
    has_next_out: list[float] = []
    command_xy_out: list[list[float]] = []
    has_command_xy_out: list[float] = []
    command_6d_out: list[list[float]] = []
    has_command_6d_out: list[float] = []
    command_support_out: list[float] = []
    observability: list[list[float]] = []
    confidence_target: list[list[float]] = []
    yaw_ambiguous: list[float] = []
    yaw_hypothesis: list[int] = []
    risk: list[int] = []
    near_field: list[float] = []
    weights: list[float] = []
    kept: list[dict[str, Any]] = []
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    npz_cache: dict[Path, dict[str, np.ndarray]] = {}
    trace_cache: dict[Path, list[dict[str, Any]]] = {}
    for row in rows:
        seq = str(row.get("sequence_id", f"{source_eval_root_key(row)}::ep{int(row.get('episode_idx', -1)):03d}"))
        by_sequence.setdefault(seq, []).append(row)
    history_names = runtime_xy_spatial_temporal_feature_names(tuple(DEFAULT_RUNTIME_XY_FEATURE_NAMES), history_window_size)
    for seq_rows in by_sequence.values():
        seq_rows.sort(key=lambda row: int(row.get("step_idx", row.get("step", 0)) or 0))
        for idx, row in enumerate(seq_rows):
            if _row_has_privileged_runtime(row):
                continue
            labels = task_frame_v46_labels_from_row(row)
            if labels is None:
                continue
            if not _label_within_support(
                labels,
                max_abs_xy_label=max_abs_xy_label,
                max_abs_z_label=max_abs_z_label,
                max_abs_yaw_label=max_abs_yaw_label,
            ):
                continue
            runtime = _load_runtime_npz_row(row, cache=npz_cache)
            if runtime is None:
                continue
            obs, npz_proprio, npz_planner = runtime
            robot_state = {
                "proprio": _coerce_vector(row.get("proprio", row.get("robot_proprio", npz_proprio)), proprio_dim),
                "planner_delta_7d": _planner_prior_from_row(row, npz_planner, planner_prior_dim),
            }
            rgbd = _load_spatial_temporal_rgbd(obs, robot_state, crop_size=image_crop_size, resize_size=image_resize_size)
            if rgbd is None:
                continue
            past = list(reversed(seq_rows[max(0, idx - history_window_size + 1) : idx]))
            hist = runtime_xy_spatial_temporal_context_feature_vector_from_trace(
                row,
                history_rows=past,
                base_feature_names=tuple(DEFAULT_RUNTIME_XY_FEATURE_NAMES),
                window_size=history_window_size,
            )
            if hist.size != len(history_names):
                continue
            images.append(rgbd.numpy().astype(np.float32))
            scalar.append(task_frame_v46_scalar_feature_vector(row).astype(np.float32))
            history.append(hist.astype(np.float32))
            proprio.append(robot_state["proprio"])
            planner.append(robot_state["planner_delta_7d"])
            labels_out.append([float(labels["dx"]), float(labels["dy"]), float(labels["dz"]), float(labels["dyaw"])])
            next_residual, has_next = _next_residual_from_row(row)
            next_labels_out.append(next_residual)
            has_next_out.append(float(has_next))
            trace_row = _trace_row_for_step(row, cache=trace_cache)
            command_xy, has_command_xy = _command_xy_from_row(row, trace_row)
            command_6d, has_command_6d = _command_6d_from_row(row, trace_row)
            command_xy_out.append(command_xy)
            has_command_xy_out.append(float(has_command_xy))
            command_6d_out.append(command_6d)
            has_command_6d_out.append(float(has_command_6d))
            pre_score = float(np.linalg.norm(np.asarray(labels_out[-1][:3], dtype=np.float32)) + abs(float(labels_out[-1][3])))
            next_score = float(np.linalg.norm(np.asarray(next_residual[:3], dtype=np.float32)) + abs(float(next_residual[3]))) if has_next > 0.5 else pre_score
            command_support_out.append(float(has_next > 0.5 and has_command_6d > 0.5 and next_score < pre_score))
            observability.append([float(labels["xy_observable"]), float(labels["z_observable"]), float(labels["yaw_observable"])])
            confidence_target.append([
                float(labels["xy_observable"]),
                float(labels["z_observable"]),
                float(bool(labels["yaw_observable"]) and not bool(labels["yaw_ambiguous"])),
            ])
            yaw_ambiguous.append(float(labels["yaw_ambiguous"]))
            yaw_hypothesis.append(_yaw_hypothesis_target(float(labels["dyaw"]), yaw_hypotheses=4, max_abs_yaw=max_abs_yaw_label))
            risk.append(_risk_target(row, labels))
            xy_norm = float(np.hypot(float(labels["dx"]), float(labels["dy"])))
            near_field.append(float(xy_norm <= float(near_field_xy_radius) and abs(float(labels["dz"])) <= float(near_field_z_radius)))
            weights.append(_sample_weight(row))
            kept_row = dict(row)
            kept_row.setdefault("source_eval_root", _source_root_from_row(row))
            kept_row.setdefault("runtime_obs_path", str(_runtime_npz_path_from_row(row) or ""))
            kept.append(kept_row)
    if not kept:
        raise RuntimeError("no v46 rows with non-privileged runtime input, RGBD observation, and offline labels")
    arrays = {
        "image": np.stack(images).astype(np.float32),
        "scalar": np.stack(scalar).astype(np.float32),
        "history": np.stack(history).astype(np.float32),
        "proprio": np.stack(proprio).astype(np.float32),
        "planner": np.stack(planner).astype(np.float32),
        "residual": np.asarray(labels_out, dtype=np.float32),
        "next_residual": np.asarray(next_labels_out, dtype=np.float32),
        "has_next_residual": np.asarray(has_next_out, dtype=np.float32),
        "command_xy": np.asarray(command_xy_out, dtype=np.float32),
        "has_command_xy": np.asarray(has_command_xy_out, dtype=np.float32),
        "command_6d": np.asarray(command_6d_out, dtype=np.float32),
        "has_command_6d": np.asarray(has_command_6d_out, dtype=np.float32),
        "command_support": np.asarray(command_support_out, dtype=np.float32),
        "observability": np.asarray(observability, dtype=np.float32),
        "confidence": np.asarray(confidence_target, dtype=np.float32),
        "yaw_ambiguous": np.asarray(yaw_ambiguous, dtype=np.float32),
        "yaw_hypothesis": np.asarray(yaw_hypothesis, dtype=np.int64),
        "risk": np.asarray(risk, dtype=np.int64),
        "near_field": np.asarray(near_field, dtype=np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
    }
    return arrays, kept


def _controller_bounded_step(
    pred: torch.Tensor,
    out: Mapping[str, torch.Tensor],
    *,
    observable: torch.Tensor | None = None,
    yaw_ambig: torch.Tensor | None = None,
    use_predicted_gate: bool,
) -> torch.Tensor:
    step_scale = out["axis_step_scale"]
    if use_predicted_gate:
        pred_observable = out["axis_observability"]
        xy_allowed = (pred_observable[:, 0] >= 0.5) & (out["axis_confidence"][:, 0] >= 0.20) & (step_scale[:, 0] >= 0.05)
        z_allowed = (pred_observable[:, 1] >= 0.5) & (out["axis_confidence"][:, 1] >= 0.20) & (step_scale[:, 1] >= 0.05)
        yaw_allowed = (
            (pred_observable[:, 2] >= 0.5)
            & (out["axis_confidence"][:, 2] >= 0.45)
            & (out["yaw_ambiguous"] < 0.5)
            & (step_scale[:, 2] >= 0.05)
        )
    else:
        if observable is None:
            raise ValueError("observable is required when use_predicted_gate=False")
        xy_allowed = observable[:, 0] >= 0.5
        z_allowed = observable[:, 1] >= 0.5
        yaw_allowed = observable[:, 2] >= 0.5
        if yaw_ambig is not None:
            yaw_allowed = yaw_allowed & (yaw_ambig < 0.5)
    zeros = torch.zeros_like(pred[:, 0])
    return torch.stack(
        [
            torch.where(xy_allowed, torch.clamp(0.35 * pred[:, 0] * step_scale[:, 0], -0.003, 0.003), zeros),
            torch.where(xy_allowed, torch.clamp(0.35 * pred[:, 1] * step_scale[:, 0], -0.003, 0.003), zeros),
            torch.where(z_allowed, torch.clamp(0.35 * pred[:, 2] * step_scale[:, 1], -0.003, 0.003), zeros),
            torch.where(yaw_allowed, torch.clamp(0.25 * pred[:, 3] * step_scale[:, 2], -0.020, 0.020), zeros),
        ],
        dim=-1,
    )


def _effect_aware_xy_step(
    residual_xy: torch.Tensor,
    planner_xy: torch.Tensor,
    xy_effect: torch.Tensor,
    *,
    max_xy_step: float = 0.003,
) -> torch.Tensor:
    eye = torch.eye(2, dtype=xy_effect.dtype, device=xy_effect.device).unsqueeze(0)
    damped = xy_effect + 1.0e-3 * eye
    desired_full = -torch.linalg.pinv(damped) @ residual_xy.unsqueeze(-1)
    correction = desired_full.squeeze(-1) - planner_xy
    return torch.clamp(correction, -float(max_xy_step), float(max_xy_step))


def _metrics(model: TaskFrameV46AlignmentNet, arrays: dict[str, np.ndarray], *, device: str) -> dict[str, Any]:
    model.eval()
    if arrays["image"].shape[0] == 0:
        return {"rows": 0}
    with torch.no_grad():
        out = model(
            torch.as_tensor(arrays["image"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["scalar"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["history"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["proprio"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["planner"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays.get("command_6d", np.zeros((arrays["residual"].shape[0], 6), dtype=np.float32)), dtype=torch.float32, device=device),
        )
    target = torch.as_tensor(arrays["residual"], dtype=torch.float32, device=device)
    next_target = torch.as_tensor(arrays.get("next_residual", np.zeros_like(arrays["residual"])), dtype=torch.float32, device=device)
    has_next = torch.as_tensor(
        arrays.get("has_next_residual", np.zeros((arrays["residual"].shape[0],), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    planner_xy = torch.as_tensor(arrays["planner"][:, :2], dtype=torch.float32, device=device)
    pred = torch.stack([out["dx"], out["dy"], out["dz"], out["dyaw"]], dim=-1)
    step = _controller_bounded_step(pred, out, use_predicted_gate=True)
    effect_step_xy = _effect_aware_xy_step(target[:, :2], planner_xy, out["xy_control_effect"])
    pred_effect_step_xy = _effect_aware_xy_step(pred[:, :2], planner_xy, out["xy_control_effect"])
    effect_delta_xy = torch.bmm(out["xy_control_effect"], (planner_xy + effect_step_xy).unsqueeze(-1)).squeeze(-1)
    pred_effect_delta_xy = torch.bmm(out["xy_control_effect"], (planner_xy + pred_effect_step_xy).unsqueeze(-1)).squeeze(-1)
    effect_post_xy = target[:, :2] + effect_delta_xy
    pred_effect_post_xy = target[:, :2] + pred_effect_delta_xy
    post = target - step
    pre_norm = torch.linalg.norm(target[:, :3], dim=-1) + torch.abs(target[:, 3])
    post_norm = torch.linalg.norm(post[:, :3], dim=-1) + torch.abs(post[:, 3])
    pre_xy = torch.linalg.norm(target[:, :2], dim=-1)
    post_xy = torch.linalg.norm(post[:, :2], dim=-1)
    pre_z = torch.abs(target[:, 2])
    post_z = torch.abs(post[:, 2])
    pre_yaw = torch.abs(target[:, 3])
    post_yaw = torch.abs(post[:, 3])
    next_xy = torch.linalg.norm(next_target[:, :2], dim=-1)
    next_z = torch.abs(next_target[:, 2])
    next_yaw = torch.abs(next_target[:, 3])
    next_mask = has_next > 0.5
    command_xy = torch.as_tensor(arrays.get("command_xy", np.zeros((arrays["residual"].shape[0], 2), dtype=np.float32)), dtype=torch.float32, device=device)
    has_command_xy = torch.as_tensor(arrays.get("has_command_xy", np.zeros((arrays["residual"].shape[0],), dtype=np.float32)), dtype=torch.float32, device=device)
    has_command_6d = torch.as_tensor(arrays.get("has_command_6d", np.zeros((arrays["residual"].shape[0],), dtype=np.float32)), dtype=torch.float32, device=device)
    effect_mask = (has_next > 0.5) & (has_command_xy > 0.5)
    command_mask = (has_next > 0.5) & (has_command_6d > 0.5)
    observed_delta_xy = next_target[:, :2] - target[:, :2]
    predicted_delta_xy = torch.bmm(out["xy_control_effect"], command_xy.unsqueeze(-1)).squeeze(-1)
    observed_delta = next_target - target
    predicted_command_delta = out["command_delta"]
    command_post = target + predicted_command_delta
    command_post_xy = torch.linalg.norm(command_post[:, :2], dim=-1)
    command_post_z = torch.abs(command_post[:, 2])
    command_post_yaw = torch.abs(command_post[:, 3])
    command_post_norm = torch.linalg.norm(command_post[:, :3], dim=-1) + torch.abs(command_post[:, 3])
    command_support_target = torch.as_tensor(arrays.get("command_support", np.zeros((arrays["residual"].shape[0],), dtype=np.float32)), dtype=torch.float32, device=device)
    command_support_pred = out["command_support"] >= 0.5
    command_support_true = command_support_target >= 0.5
    pre_xy_norm = torch.linalg.norm(target[:, :2], dim=-1)
    effect_post_xy_norm = torch.linalg.norm(effect_post_xy, dim=-1)
    pred_effect_post_xy_norm = torch.linalg.norm(pred_effect_post_xy, dim=-1)
    x_pre_abs = torch.abs(target[:, 0])
    y_pre_abs = torch.abs(target[:, 1])
    x_next_abs = torch.abs(next_target[:, 0])
    y_next_abs = torch.abs(next_target[:, 1])
    x_effect_post_abs = torch.abs(effect_post_xy[:, 0])
    y_effect_post_abs = torch.abs(effect_post_xy[:, 1])
    x_pred_effect_post_abs = torch.abs(pred_effect_post_xy[:, 0])
    y_pred_effect_post_abs = torch.abs(pred_effect_post_xy[:, 1])
    near_field_target = torch.as_tensor(arrays.get("near_field", np.zeros((arrays["residual"].shape[0],), dtype=np.float32)), dtype=torch.float32, device=device)
    near_field_pred = out["near_field_confidence"]
    near_field_binary = near_field_pred >= 0.5
    near_field_target_binary = near_field_target >= 0.5
    near_field_positive = near_field_target_binary
    observable_target = torch.as_tensor(arrays["observability"], dtype=torch.float32, device=device)
    observable_pred = out["axis_observability"] >= 0.5
    yaw_ambiguous_target = torch.as_tensor(arrays["yaw_ambiguous"], dtype=torch.float32, device=device) >= 0.5
    yaw_ambiguous_pred = out["yaw_ambiguous"] >= 0.5
    yaw_control_target = (observable_target[:, 2] >= 0.5) & (~yaw_ambiguous_target)
    yaw_control_pred = observable_pred[:, 2] & (~yaw_ambiguous_pred)
    return {
        "rows": int(target.shape[0]),
        "mae_xy": float(torch.mean(torch.abs(pred[:, :2] - target[:, :2])).item()),
        "mae_z": float(torch.mean(torch.abs(pred[:, 2] - target[:, 2])).item()),
        "mae_yaw": float(torch.mean(torch.abs(pred[:, 3] - target[:, 3])).item()),
        "xy_sign_match": float((torch.sign(pred[:, :2]) == torch.sign(target[:, :2])).float().mean().item()),
        "z_sign_match": float((torch.sign(pred[:, 2]) == torch.sign(target[:, 2])).float().mean().item()),
        "yaw_sign_match": float((torch.sign(pred[:, 3]) == torch.sign(target[:, 3])).float().mean().item()),
        "yaw_hypothesis_accuracy": float(
            (
                torch.argmax(out["yaw_hypothesis_logits"], dim=-1)
                == torch.as_tensor(arrays["yaw_hypothesis"], device=device)
            ).float().mean().item()
        ),
        "bounded_step_contraction": float((post_norm < pre_norm).float().mean().item()),
        "bounded_step_worsen": float((post_norm > pre_norm).float().mean().item()),
        "xy_bounded_step_contraction": float((post_xy < pre_xy).float().mean().item()),
        "xy_bounded_step_worsen": float((post_xy > pre_xy).float().mean().item()),
        "z_bounded_step_contraction": float((post_z < pre_z).float().mean().item()),
        "z_bounded_step_worsen": float((post_z > pre_z).float().mean().item()),
        "yaw_bounded_step_contraction": float((post_yaw < pre_yaw).float().mean().item()),
        "yaw_bounded_step_worsen": float((post_yaw > pre_yaw).float().mean().item()),
        "transition_rows": int(next_mask.float().sum().item()),
        "transition_xy_contraction": float((next_xy[next_mask] < pre_xy[next_mask]).float().mean().item()) if bool(next_mask.any()) else 0.0,
        "transition_z_contraction": float((next_z[next_mask] < pre_z[next_mask]).float().mean().item()) if bool(next_mask.any()) else 0.0,
        "transition_yaw_contraction": float((next_yaw[next_mask] < pre_yaw[next_mask]).float().mean().item()) if bool(next_mask.any()) else 0.0,
        "effect_rows": int(effect_mask.float().sum().item()),
        "command_transition_rows": int(command_mask.float().sum().item()),
        "command_delta_mae": float(torch.mean(torch.abs(predicted_command_delta[command_mask] - observed_delta[command_mask])).item()) if bool(command_mask.any()) else 0.0,
        "command_x_delta_mae": float(torch.mean(torch.abs(predicted_command_delta[command_mask, 0] - observed_delta[command_mask, 0])).item()) if bool(command_mask.any()) else 0.0,
        "command_y_delta_mae": float(torch.mean(torch.abs(predicted_command_delta[command_mask, 1] - observed_delta[command_mask, 1])).item()) if bool(command_mask.any()) else 0.0,
        "command_z_delta_mae": float(torch.mean(torch.abs(predicted_command_delta[command_mask, 2] - observed_delta[command_mask, 2])).item()) if bool(command_mask.any()) else 0.0,
        "command_yaw_delta_mae": float(torch.mean(torch.abs(predicted_command_delta[command_mask, 3] - observed_delta[command_mask, 3])).item()) if bool(command_mask.any()) else 0.0,
        "command_xy_predicted_contraction": float((command_post_xy[command_mask] < pre_xy[command_mask]).float().mean().item()) if bool(command_mask.any()) else 0.0,
        "command_xy_predicted_worsen": float((command_post_xy[command_mask] > pre_xy[command_mask]).float().mean().item()) if bool(command_mask.any()) else 0.0,
        "command_z_predicted_contraction": float((command_post_z[command_mask] < pre_z[command_mask]).float().mean().item()) if bool(command_mask.any()) else 0.0,
        "command_yaw_predicted_contraction": float((command_post_yaw[command_mask] < pre_yaw[command_mask]).float().mean().item()) if bool(command_mask.any()) else 0.0,
        "command_combined_predicted_contraction": float((command_post_norm[command_mask] < pre_norm[command_mask]).float().mean().item()) if bool(command_mask.any()) else 0.0,
        "command_support_accuracy": float((command_support_pred[command_mask] == command_support_true[command_mask]).float().mean().item()) if bool(command_mask.any()) else 0.0,
        "command_support_predicted_positive_rate": float(command_support_pred[command_mask].float().mean().item()) if bool(command_mask.any()) else 0.0,
        "command_support_true_positive_rate": float(command_support_true[command_mask].float().mean().item()) if bool(command_mask.any()) else 0.0,
        "xy_effect_delta_mae": float(torch.mean(torch.abs(predicted_delta_xy[effect_mask] - observed_delta_xy[effect_mask])).item()) if bool(effect_mask.any()) else 0.0,
        "x_effect_delta_mae": float(torch.mean(torch.abs(predicted_delta_xy[effect_mask, 0] - observed_delta_xy[effect_mask, 0])).item()) if bool(effect_mask.any()) else 0.0,
        "y_effect_delta_mae": float(torch.mean(torch.abs(predicted_delta_xy[effect_mask, 1] - observed_delta_xy[effect_mask, 1])).item()) if bool(effect_mask.any()) else 0.0,
        "x_observed_transition_contraction": float((x_next_abs[effect_mask] < x_pre_abs[effect_mask]).float().mean().item()) if bool(effect_mask.any()) else 0.0,
        "x_observed_transition_worsen": float((x_next_abs[effect_mask] > x_pre_abs[effect_mask]).float().mean().item()) if bool(effect_mask.any()) else 0.0,
        "y_observed_transition_contraction": float((y_next_abs[effect_mask] < y_pre_abs[effect_mask]).float().mean().item()) if bool(effect_mask.any()) else 0.0,
        "y_observed_transition_worsen": float((y_next_abs[effect_mask] > y_pre_abs[effect_mask]).float().mean().item()) if bool(effect_mask.any()) else 0.0,
        "xy_effect_aware_contraction": float((effect_post_xy_norm < pre_xy_norm).float().mean().item()),
        "xy_effect_aware_worsen": float((effect_post_xy_norm > pre_xy_norm).float().mean().item()),
        "x_effect_aware_contraction": float((x_effect_post_abs < x_pre_abs).float().mean().item()),
        "x_effect_aware_worsen": float((x_effect_post_abs > x_pre_abs).float().mean().item()),
        "y_effect_aware_contraction": float((y_effect_post_abs < y_pre_abs).float().mean().item()),
        "y_effect_aware_worsen": float((y_effect_post_abs > y_pre_abs).float().mean().item()),
        "xy_predicted_effect_aware_contraction": float((pred_effect_post_xy_norm < pre_xy_norm).float().mean().item()),
        "xy_predicted_effect_aware_worsen": float((pred_effect_post_xy_norm > pre_xy_norm).float().mean().item()),
        "x_predicted_effect_aware_contraction": float((x_pred_effect_post_abs < x_pre_abs).float().mean().item()),
        "x_predicted_effect_aware_worsen": float((x_pred_effect_post_abs > x_pre_abs).float().mean().item()),
        "y_predicted_effect_aware_contraction": float((y_pred_effect_post_abs < y_pre_abs).float().mean().item()),
        "y_predicted_effect_aware_worsen": float((y_pred_effect_post_abs > y_pre_abs).float().mean().item()),
        "near_field_accuracy": float((near_field_binary == near_field_target_binary).float().mean().item()),
        "near_field_positive_rate": float(near_field_target.float().mean().item()),
        "near_field_predicted_positive_rate": float(near_field_binary.float().mean().item()),
        "near_field_recall": float((near_field_binary[near_field_positive]).float().mean().item()) if bool(near_field_positive.any()) else 0.0,
        "risk_accuracy": float((torch.argmax(out["risk_logits"], dim=-1) == torch.as_tensor(arrays["risk"], device=device)).float().mean().item()),
        "xy_observable_target_rate": float((observable_target[:, 0] >= 0.5).float().mean().item()),
        "xy_observable_predicted_rate": float(observable_pred[:, 0].float().mean().item()),
        "z_observable_target_rate": float((observable_target[:, 1] >= 0.5).float().mean().item()),
        "z_observable_predicted_rate": float(observable_pred[:, 1].float().mean().item()),
        "yaw_observable_target_rate": float((observable_target[:, 2] >= 0.5).float().mean().item()),
        "yaw_observable_predicted_rate": float(observable_pred[:, 2].float().mean().item()),
        "yaw_ambiguous_target_rate": float(yaw_ambiguous_target.float().mean().item()),
        "yaw_ambiguous_predicted_rate": float(yaw_ambiguous_pred.float().mean().item()),
        "yaw_control_target_rate": float(yaw_control_target.float().mean().item()),
        "yaw_control_predicted_rate": float(yaw_control_pred.float().mean().item()),
    }


def _normalized_residual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scales = TASK_FRAME_V46_RESIDUAL_SCALES.to(device=pred.device, dtype=pred.dtype).reshape(1, -1)
    return F.smooth_l1_loss(pred / scales, target / scales)


def _direction_sign_loss(pred: torch.Tensor, target: torch.Tensor, observable: torch.Tensor | None = None, yaw_ambig: torch.Tensor | None = None) -> torch.Tensor:
    scales = TASK_FRAME_V46_RESIDUAL_SCALES.to(device=pred.device, dtype=pred.dtype).reshape(1, -1)
    target_norm = target / scales
    pred_norm = pred / scales
    min_abs = torch.as_tensor([0.0015, 0.0015, 0.0015, 0.010], dtype=pred.dtype, device=pred.device).reshape(1, -1)
    mask = (torch.abs(target) >= min_abs).float()
    if observable is not None:
        mask[:, 0] *= observable[:, 0]
        mask[:, 1] *= observable[:, 0]
        mask[:, 2] *= observable[:, 1]
        mask[:, 3] *= observable[:, 2]
    if yaw_ambig is not None:
        mask[:, 3] *= (yaw_ambig < 0.5).float()
    if float(mask.sum().detach().cpu().item()) <= 0.0:
        return pred.new_tensor(0.0)
    signed_margin = pred_norm * torch.sign(target_norm)
    return (F.softplus(-signed_margin) * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def _yaw_hypothesis_loss(logits: torch.Tensor, target: torch.Tensor, observable: torch.Tensor, yaw_ambig: torch.Tensor) -> torch.Tensor:
    per_row = F.cross_entropy(logits, target, reduction="none")
    yaw_observable = observable[:, 2].float()
    weights = 0.10 + 0.90 * yaw_observable + 0.30 * yaw_ambig.float()
    return (per_row * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def _weighted_bce(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    if weight is None:
        return loss.mean()
    weight = weight.to(device=loss.device, dtype=loss.dtype)
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    return (loss * weight).sum() / torch.clamp(weight.expand_as(loss).sum(), min=1.0)


def _axis_observability_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    yaw_ambig: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    yaw_control_positive_weight: float = 8.0,
    yaw_observable_negative_weight: float = 1.5,
) -> torch.Tensor:
    weights = torch.ones_like(target)
    yaw_control = (target[:, 2] >= 0.5) & (yaw_ambig < 0.5)
    yaw_noncontrol = ~yaw_control
    weights[:, 2] = torch.where(
        yaw_control,
        torch.full_like(weights[:, 2], float(yaw_control_positive_weight)),
        torch.full_like(weights[:, 2], float(yaw_observable_negative_weight)),
    )
    sample_weight = sample_weight.to(device=weights.device, dtype=weights.dtype)
    weights = weights * torch.clamp(sample_weight.reshape(-1, 1), min=0.05)
    return _weighted_bce(pred, target, weights)


def _yaw_ambiguity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    observable: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    observable_clear_negative_weight: float = 8.0,
    ambiguous_positive_weight: float = 2.0,
) -> torch.Tensor:
    yaw_clear_control = (observable[:, 2] >= 0.5) & (target < 0.5)
    weights = torch.where(
        yaw_clear_control,
        torch.full_like(target, float(observable_clear_negative_weight)),
        torch.where(target >= 0.5, torch.full_like(target, float(ambiguous_positive_weight)), torch.ones_like(target)),
    )
    weights = weights * torch.clamp(sample_weight.to(device=weights.device, dtype=weights.dtype), min=0.05)
    return _weighted_bce(pred, target, weights)


def train(
    dataset_jsonl: list[Path],
    *,
    output_checkpoint: Path,
    output_json: Path,
    split_mode: str = "root",
    val_fraction: float = 0.2,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1.0e-3,
    seed: int = 7,
    image_hidden_dim: int = 128,
    fusion_hidden_dim: int = 128,
    use_spatial_moments: bool = False,
    history_window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    max_abs_xy_label: float = 0.080,
    max_abs_z_label: float = 0.080,
    max_abs_yaw_label: float = 0.350,
    near_field_xy_radius: float = 0.060,
    near_field_z_radius: float = 0.040,
    yaw_control_positive_weight: float = 8.0,
    yaw_ambiguity_clear_negative_weight: float = 8.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    rows = [_normalize_row_metadata(row) for row in _load_rows(dataset_jsonl)]
    split = split_records_by_source_root(rows, split_mode=split_mode, val_fraction=val_fraction, test_fraction=0.0, seed=seed)
    if not split.val_records:
        raise RuntimeError(
            "v46 validation split is empty. Add more source roots for root-held-out training, "
            "or use --split_mode episode only for smoke/dry-run pipeline validation."
        )
    train_arrays, train_kept = _build_arrays(
        list(split.train_records),
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
        history_window_size=history_window_size,
        proprio_dim=15,
        planner_prior_dim=6,
        max_abs_xy_label=max_abs_xy_label,
        max_abs_z_label=max_abs_z_label,
        max_abs_yaw_label=max_abs_yaw_label,
        near_field_xy_radius=near_field_xy_radius,
        near_field_z_radius=near_field_z_radius,
    )
    val_arrays, val_kept = _build_arrays(
        list(split.val_records),
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
        history_window_size=history_window_size,
        proprio_dim=15,
        planner_prior_dim=6,
        max_abs_xy_label=max_abs_xy_label,
        max_abs_z_label=max_abs_z_label,
        max_abs_yaw_label=max_abs_yaw_label,
        near_field_xy_radius=near_field_xy_radius,
        near_field_z_radius=near_field_z_radius,
    )
    model = TaskFrameV46AlignmentNet(
        image_hidden_dim=image_hidden_dim,
        scalar_feature_dim=len(TASK_FRAME_READINESS_FEATURE_NAMES),
        history_feature_dim=max(1, train_arrays["history"].shape[1] // max(1, history_window_size)),
        history_window_size=history_window_size,
        fusion_hidden_dim=fusion_hidden_dim,
        risk_classes=TASK_FRAME_V46_RISK_CLASSES,
        max_abs_xy=max_abs_xy_label,
        max_abs_z=max_abs_z_label,
        max_abs_yaw=max_abs_yaw_label,
        use_spatial_moments=use_spatial_moments,
    ).to(device)
    dataset = TensorDataset(
        torch.as_tensor(train_arrays["image"], dtype=torch.float32),
        torch.as_tensor(train_arrays["scalar"], dtype=torch.float32),
        torch.as_tensor(train_arrays["history"], dtype=torch.float32),
        torch.as_tensor(train_arrays["proprio"], dtype=torch.float32),
        torch.as_tensor(train_arrays["planner"], dtype=torch.float32),
        torch.as_tensor(train_arrays["residual"], dtype=torch.float32),
        torch.as_tensor(train_arrays["next_residual"], dtype=torch.float32),
        torch.as_tensor(train_arrays["has_next_residual"], dtype=torch.float32),
        torch.as_tensor(train_arrays["command_xy"], dtype=torch.float32),
        torch.as_tensor(train_arrays["has_command_xy"], dtype=torch.float32),
        torch.as_tensor(train_arrays["command_6d"], dtype=torch.float32),
        torch.as_tensor(train_arrays["has_command_6d"], dtype=torch.float32),
        torch.as_tensor(train_arrays["command_support"], dtype=torch.float32),
        torch.as_tensor(train_arrays["observability"], dtype=torch.float32),
        torch.as_tensor(train_arrays["confidence"], dtype=torch.float32),
        torch.as_tensor(train_arrays["yaw_ambiguous"], dtype=torch.float32),
        torch.as_tensor(train_arrays["yaw_hypothesis"], dtype=torch.long),
        torch.as_tensor(train_arrays["risk"], dtype=torch.long),
        torch.as_tensor(train_arrays["near_field"], dtype=torch.float32),
        torch.as_tensor(train_arrays["weights"], dtype=torch.float32),
    )
    sampler = WeightedRandomSampler(torch.as_tensor(train_arrays["weights"], dtype=torch.float32), num_samples=len(train_arrays["weights"]), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    for _ in range(int(epochs)):
        model.train()
        for image, scalar, history, proprio, planner, residual, next_residual, has_next_residual, command_xy, has_command_xy, command_6d, has_command_6d, command_support, observable, confidence, yaw_ambig, yaw_hyp, risk, near_field, sample_weight in loader:
            image = image.to(device)
            scalar = scalar.to(device)
            history = history.to(device)
            proprio = proprio.to(device)
            planner = planner.to(device)
            residual = residual.to(device)
            next_residual = next_residual.to(device)
            has_next_residual = has_next_residual.to(device)
            command_xy = command_xy.to(device)
            has_command_xy = has_command_xy.to(device)
            command_6d = command_6d.to(device)
            has_command_6d = has_command_6d.to(device)
            command_support = command_support.to(device)
            observable = observable.to(device)
            confidence = confidence.to(device)
            yaw_ambig = yaw_ambig.to(device)
            yaw_hyp = yaw_hyp.to(device)
            risk = risk.to(device)
            near_field = near_field.to(device)
            sample_weight = sample_weight.to(device)
            out = model(image, scalar, history, proprio, planner, command_6d)
            pred = torch.stack([out["dx"], out["dy"], out["dz"], out["dyaw"]], dim=-1)
            step = _controller_bounded_step(pred, out, observable=observable, yaw_ambig=yaw_ambig, use_predicted_gate=False)
            post = residual - step
            pre_norm = torch.linalg.norm(residual[:, :3], dim=-1) + torch.abs(residual[:, 3])
            post_norm = torch.linalg.norm(post[:, :3], dim=-1) + torch.abs(post[:, 3])
            contraction_loss = F.relu(post_norm - 0.90 * pre_norm).mean()
            overshoot_loss = F.relu(torch.abs(step) - torch.abs(residual)).mean() + F.relu(-(step * residual)).mean()
            transition_mask = has_next_residual > 0.5
            if bool(transition_mask.any()):
                expected_delta = post - residual
                observed_delta = next_residual - residual
                transition_direction_loss = F.relu(-(expected_delta[transition_mask] * observed_delta[transition_mask])).mean()
                transition_post_loss = F.smooth_l1_loss(post[transition_mask], next_residual[transition_mask])
            else:
                transition_direction_loss = torch.zeros((), dtype=residual.dtype, device=device)
                transition_post_loss = torch.zeros((), dtype=residual.dtype, device=device)
            effect_mask = (has_next_residual > 0.5) & (has_command_xy > 0.5)
            command_mask = (has_next_residual > 0.5) & (has_command_6d > 0.5)
            if bool(command_mask.any()):
                observed_delta = next_residual - residual
                predicted_delta = out["command_delta"]
                command_logvar = out["command_logvar"]
                scales = TASK_FRAME_V46_RESIDUAL_SCALES.to(device=residual.device, dtype=residual.dtype).reshape(1, -1)
                delta_error = (predicted_delta - observed_delta) / scales
                command_nll = 0.5 * (torch.exp(-command_logvar[command_mask]) * delta_error[command_mask] ** 2 + command_logvar[command_mask]).mean()
                command_delta_loss = F.smooth_l1_loss(predicted_delta[command_mask] / scales, observed_delta[command_mask] / scales)
                command_sign_loss = F.relu(-(predicted_delta[command_mask] * observed_delta[command_mask]) / scales).mean()
                command_y_sign_loss = F.relu(-(predicted_delta[command_mask, 1] * observed_delta[command_mask, 1]) / scales[0, 1]).mean()
                predicted_post = residual + predicted_delta
                command_post_loss = F.smooth_l1_loss(predicted_post[command_mask] / scales, next_residual[command_mask] / scales)
                command_support_loss = F.binary_cross_entropy(out["command_support"][command_mask], command_support[command_mask])
            else:
                command_nll = torch.zeros((), dtype=residual.dtype, device=device)
                command_delta_loss = torch.zeros((), dtype=residual.dtype, device=device)
                command_sign_loss = torch.zeros((), dtype=residual.dtype, device=device)
                command_y_sign_loss = torch.zeros((), dtype=residual.dtype, device=device)
                command_post_loss = torch.zeros((), dtype=residual.dtype, device=device)
                command_support_loss = torch.zeros((), dtype=residual.dtype, device=device)
            if bool(effect_mask.any()):
                observed_delta_xy = next_residual[:, :2] - residual[:, :2]
                predicted_delta_xy = torch.bmm(out["xy_control_effect"], command_xy.unsqueeze(-1)).squeeze(-1)
                effect_delta_loss = F.smooth_l1_loss(predicted_delta_xy[effect_mask], observed_delta_xy[effect_mask])
                effect_y_delta_loss = F.smooth_l1_loss(predicted_delta_xy[effect_mask, 1], observed_delta_xy[effect_mask, 1])
                effect_sign_loss = F.relu(-(predicted_delta_xy[effect_mask] * observed_delta_xy[effect_mask])).mean()
                effect_y_sign_loss = F.relu(-(predicted_delta_xy[effect_mask, 1] * observed_delta_xy[effect_mask, 1])).mean()
                actual_predicted_post_xy = residual[:, :2] + predicted_delta_xy
                actual_command_contraction_loss = F.relu(
                    torch.linalg.norm(actual_predicted_post_xy[effect_mask], dim=-1)
                    - 1.02 * torch.linalg.norm(next_residual[effect_mask, :2], dim=-1)
                ).mean()
                effect_step_xy = _effect_aware_xy_step(residual[:, :2], planner[:, :2], out["xy_control_effect"])
                effect_delta_after = torch.bmm(out["xy_control_effect"], (planner[:, :2] + effect_step_xy).unsqueeze(-1)).squeeze(-1)
                effect_post_xy = residual[:, :2] + effect_delta_after
                effect_pre_xy_norm = torch.linalg.norm(residual[:, :2], dim=-1)
                effect_post_xy_norm = torch.linalg.norm(effect_post_xy, dim=-1)
                effect_contraction_loss = F.relu(effect_post_xy_norm[effect_mask] - 0.92 * effect_pre_xy_norm[effect_mask]).mean()
                effect_y_contraction_loss = F.relu(torch.abs(effect_post_xy[effect_mask, 1]) - 0.92 * torch.abs(residual[effect_mask, 1])).mean()
            else:
                effect_delta_loss = torch.zeros((), dtype=residual.dtype, device=device)
                effect_y_delta_loss = torch.zeros((), dtype=residual.dtype, device=device)
                effect_sign_loss = torch.zeros((), dtype=residual.dtype, device=device)
                effect_y_sign_loss = torch.zeros((), dtype=residual.dtype, device=device)
                actual_command_contraction_loss = torch.zeros((), dtype=residual.dtype, device=device)
                effect_contraction_loss = torch.zeros((), dtype=residual.dtype, device=device)
                effect_y_contraction_loss = torch.zeros((), dtype=residual.dtype, device=device)
            loss = (
                1.2 * _normalized_residual_loss(pred, residual)
                + 1.8 * _direction_sign_loss(pred, residual, observable=observable, yaw_ambig=yaw_ambig)
                + 0.9 * _weighted_bce(out["axis_confidence"], confidence, sample_weight)
                + 1.2
                * _axis_observability_loss(
                    out["axis_observability"],
                    observable,
                    yaw_ambig,
                    sample_weight,
                    yaw_control_positive_weight=float(yaw_control_positive_weight),
                )
                + 1.0
                * _yaw_ambiguity_loss(
                    out["yaw_ambiguous"],
                    yaw_ambig,
                    observable,
                    sample_weight,
                    observable_clear_negative_weight=float(yaw_ambiguity_clear_negative_weight),
                )
                + 0.8 * _yaw_hypothesis_loss(out["yaw_hypothesis_logits"], yaw_hyp, observable, yaw_ambig)
                + 0.5 * F.cross_entropy(out["risk_logits"], risk)
                + 1.0 * F.binary_cross_entropy(out["near_field_confidence"], near_field)
                + 1.2 * contraction_loss
                + 0.8 * overshoot_loss
                + 0.8 * transition_direction_loss
                + 0.4 * transition_post_loss
                + 1.2 * effect_delta_loss
                + 1.0 * effect_y_delta_loss
                + 0.8 * effect_sign_loss
                + 0.8 * effect_y_sign_loss
                + 0.6 * actual_command_contraction_loss
                + 0.8 * effect_contraction_loss
                + 0.8 * effect_y_contraction_loss
                + 1.4 * command_nll
                + 1.0 * command_delta_loss
                + 0.9 * command_sign_loss
                + 1.0 * command_y_sign_loss
                + 0.6 * command_post_loss
                + 0.6 * command_support_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    train_metrics = _metrics(model, train_arrays, device=device)
    val_metrics = _metrics(model, val_arrays, device=device)
    metadata = {
        "schema_version": "c2c_v2_task_frame_v46_alignment_report_v1",
        "model": "v46_unified_task_frame_alignment_candidate",
        "dataset_jsonl": [str(p) for p in dataset_jsonl],
        "split_mode": str(split.split_mode),
        "train_rows": int(train_arrays["image"].shape[0]),
        "val_rows": int(val_arrays["image"].shape[0]),
        "train_source_eval_roots": sorted({source_eval_root_key(row) for row in train_kept}),
        "val_source_eval_roots": sorted({source_eval_root_key(row) for row in val_kept}),
        "max_abs_xy_label": float(max_abs_xy_label),
        "max_abs_z_label": float(max_abs_z_label),
        "max_abs_yaw_label": float(max_abs_yaw_label),
        "near_field_xy_radius": float(near_field_xy_radius),
        "near_field_z_radius": float(near_field_z_radius),
        "yaw_control_positive_weight": float(yaw_control_positive_weight),
        "yaw_ambiguity_clear_negative_weight": float(yaw_ambiguity_clear_negative_weight),
        "use_spatial_moments": bool(use_spatial_moments),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_labels_only",
        "upgrade_gate": "pending_random_holdout_closed_loop_validation",
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_task_frame_v46_alignment_checkpoint(
        output_checkpoint,
        model,
        metadata=metadata,
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
    )
    output_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    ap.add_argument("--output_checkpoint", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--split_mode", choices=("root", "episode", "auto"), default="root")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--image_hidden_dim", type=int, default=128)
    ap.add_argument("--fusion_hidden_dim", type=int, default=128)
    ap.add_argument("--use_spatial_moments", action="store_true", default=False)
    ap.add_argument("--history_window_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW)
    ap.add_argument("--image_crop_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE)
    ap.add_argument("--image_resize_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE)
    ap.add_argument("--max_abs_xy_label", type=float, default=0.080)
    ap.add_argument("--max_abs_z_label", type=float, default=0.080)
    ap.add_argument("--max_abs_yaw_label", type=float, default=0.350)
    ap.add_argument("--near_field_xy_radius", type=float, default=0.060)
    ap.add_argument("--near_field_z_radius", type=float, default=0.040)
    ap.add_argument("--yaw_control_positive_weight", type=float, default=8.0)
    ap.add_argument("--yaw_ambiguity_clear_negative_weight", type=float, default=8.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train(
        list(args.dataset_jsonl),
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        split_mode=str(args.split_mode),
        val_fraction=float(args.val_fraction),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        image_hidden_dim=int(args.image_hidden_dim),
        fusion_hidden_dim=int(args.fusion_hidden_dim),
        use_spatial_moments=bool(args.use_spatial_moments),
        history_window_size=int(args.history_window_size),
        image_crop_size=int(args.image_crop_size),
        image_resize_size=int(args.image_resize_size),
        max_abs_xy_label=float(args.max_abs_xy_label),
        max_abs_z_label=float(args.max_abs_z_label),
        max_abs_yaw_label=float(args.max_abs_yaw_label),
        near_field_xy_radius=float(args.near_field_xy_radius),
        near_field_z_radius=float(args.near_field_z_radius),
        yaw_control_positive_weight=float(args.yaw_control_positive_weight),
        yaw_ambiguity_clear_negative_weight=float(args.yaw_ambiguity_clear_negative_weight),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
