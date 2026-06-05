"""Non-privileged XY residual evidence for runtime precision takeover.

This module is intentionally conservative.  It does not make yaw or close
control decisions, and it does not consume privileged target pose.  Its job is
to separate visual evidence, calibrated proxy validity, and the eventual
contract-aligned XY residual estimate that runtime C2C can use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basin_state import EstimatedBasinError
from .localizers import LocalGeometryError
from .learned_localizer import _ImageEncoder, _roi_center_from_prior

RUNTIME_XY_NORMALIZATION_STD_FLOOR = 1.0e-3
RUNTIME_XY_NORMALIZED_FEATURE_CLIP = 10.0
RUNTIME_XY_MAX_RESIDUAL_NORM = 0.18
RUNTIME_XY_DIRECTION_MIN_COSINE = 0.25
RUNTIME_XY_LOW_VIS_DIRECTION_MIN_COSINE = 0.50
RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW = 6
RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE = 96
RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE = 96
# Keep the spatial-temporal runtime a little conservative so it generalizes
# better on sentinel slices without opening close/handoff authority.
RUNTIME_XY_SPATIOTEMPORAL_STEP_SCALE_SAFETY_MARGIN = 0.90
RUNTIME_XY_SPATIAL_TEMPORAL_RISK_CLASSES: tuple[str, ...] = (
    "normal",
    "low_visibility",
    "direction_conflict",
    "insufficient_support",
)
RUNTIME_XY_SPATIOTEMPORAL_RISK_CLASSES = RUNTIME_XY_SPATIAL_TEMPORAL_RISK_CLASSES


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _coerce_vector(value: Any, *, length: int, default: float = 0.0) -> np.ndarray:
    arr: np.ndarray
    if value is None:
        arr = np.asarray([], dtype=np.float32)
    else:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.asarray([], dtype=np.float32)
    if arr.size < int(length):
        pad = np.full((int(length) - int(arr.size),), float(default), dtype=np.float32)
        arr = np.concatenate([arr.astype(np.float32), pad], axis=0)
    return arr[: int(length)].astype(np.float32)


def _clip_xy_norm(xy: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float32).reshape(-1)
    out = np.zeros((2,), dtype=np.float32)
    if arr.size >= 2:
        out[:] = arr[:2]
    if not np.all(np.isfinite(out)):
        return np.zeros((2,), dtype=np.float32)
    norm = float(np.linalg.norm(out))
    if norm > float(max_norm) > 0.0:
        out = (out * (float(max_norm) / max(norm, 1.0e-9))).astype(np.float32)
    return out.astype(np.float32)


def _xy_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)[:2]
    bb = np.asarray(b, dtype=np.float32).reshape(-1)[:2]
    if aa.size < 2 or bb.size < 2 or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return 0.0
    an = float(np.linalg.norm(aa))
    bn = float(np.linalg.norm(bb))
    if an <= 1.0e-9 or bn <= 1.0e-9:
        return 0.0
    return float(np.dot(aa, bb) / max(an * bn, 1.0e-9))


def _xy_sign_match(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)[:2]
    bb = np.asarray(b, dtype=np.float32).reshape(-1)[:2]
    if aa.size < 2 or bb.size < 2 or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return 0.0
    return float(np.mean(np.sign(aa) == np.sign(bb)))


def _roi_crop_box_from_observation(
    observation: Mapping[str, Any],
    robot_state: Mapping[str, Any] | None,
    *,
    crop_size: int,
) -> tuple[int, int, int, int] | None:
    rgb = None
    for key in ("wrist_rgb", "front_rgb"):
        value = observation.get(key) if isinstance(observation, Mapping) else None
        if value is not None:
            rgb = np.asarray(value)
            break
    if rgb is None or rgb.ndim < 2:
        return None
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h <= 0 or w <= 0:
        return None
    cx, cy = _roi_center_from_prior(observation, robot_state, w, h)
    crop = int(max(8, min(crop_size, min(w, h))))
    half = crop // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x0 = max(0, min(x0, w - crop))
    y0 = max(0, min(y0, h - crop))
    return int(x0), int(y0), int(x0 + crop), int(y0 + crop)


def _load_spatial_temporal_rgbd(
    observation: Mapping[str, Any],
    robot_state: Mapping[str, Any] | None,
    *,
    crop_size: int,
    resize_size: int,
) -> torch.Tensor | None:
    rgb = None
    depth = None
    for key in ("wrist_rgb", "front_rgb"):
        value = observation.get(key) if isinstance(observation, Mapping) else None
        if value is not None:
            rgb = np.asarray(value, dtype=np.float32)
            break
    for key in ("wrist_depth", "front_depth"):
        value = observation.get(key) if isinstance(observation, Mapping) else None
        if value is not None:
            depth = np.asarray(value, dtype=np.float32)
            break
    if rgb is None or depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[..., 0]
    if float(np.nanmax(depth)) > 1.5:
        depth = depth / 255.0
    depth = np.clip(depth, 0.0, 1.0)
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    valid = np.isfinite(depth).astype(np.float32)
    rgbd = np.concatenate([rgb, depth[..., None], valid[..., None]], axis=-1).astype(np.float32)
    crop_box = _roi_crop_box_from_observation(observation, robot_state, crop_size=crop_size)
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        rgbd = rgbd[y0:y1, x0:x1]
    if resize_size > 0 and (rgbd.shape[0] != resize_size or rgbd.shape[1] != resize_size):
        rgb_img = Image.fromarray(np.clip(rgbd[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
        depth_img = Image.fromarray(np.clip(rgbd[..., 3], 0.0, 1.0).astype(np.float32), mode="F")
        valid_img = Image.fromarray(np.clip(rgbd[..., 4], 0.0, 1.0).astype(np.float32), mode="F")
        rgb_img = rgb_img.resize((resize_size, resize_size), resample=Image.BILINEAR)
        depth_img = depth_img.resize((resize_size, resize_size), resample=Image.BILINEAR)
        valid_img = valid_img.resize((resize_size, resize_size), resample=Image.BILINEAR)
        rgb_arr = np.asarray(rgb_img, dtype=np.float32) / 255.0
        depth_arr = np.asarray(depth_img, dtype=np.float32)
        valid_arr = np.asarray(valid_img, dtype=np.float32)
        rgbd = np.concatenate([rgb_arr, depth_arr[..., None], valid_arr[..., None]], axis=-1).astype(np.float32)
    h, w = rgbd.shape[:2]
    xs = np.linspace(-1.0, 1.0, num=w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    rgbd = np.concatenate([rgbd, grid_x[..., None], grid_y[..., None]], axis=-1)
    return torch.from_numpy(np.transpose(rgbd, (2, 0, 1))).float()


def _nested(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, Mapping):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, Mapping) else {}


def _flat_estimated_basin_from_trace(data: Mapping[str, Any] | None) -> EstimatedBasinError | None:
    if not isinstance(data, Mapping) or not data:
        return None
    if "estimated_basin_error_valid" not in data:
        return EstimatedBasinError.from_dict(data)
    return EstimatedBasinError(
        valid=_as_bool(data.get("estimated_basin_error_valid"), False),
        confidence=_as_float(data.get("estimated_basin_error_confidence"), 0.0),
        dx=_as_float(data.get("estimated_basin_error_dx"), 0.0),
        dy=_as_float(data.get("estimated_basin_error_dy"), 0.0),
        dz=_as_float(data.get("estimated_basin_error_dz"), 0.0),
        dyaw=_as_float(data.get("estimated_basin_error_dyaw"), 0.0),
        x_valid=_as_bool(data.get("estimated_basin_error_x_valid"), False),
        y_valid=_as_bool(data.get("estimated_basin_error_y_valid"), False),
        z_valid=_as_bool(data.get("estimated_basin_error_z_valid"), False),
        yaw_valid=_as_bool(data.get("estimated_basin_error_yaw_valid"), False),
        x_confidence=_as_float(data.get("estimated_basin_error_x_confidence"), 0.0),
        y_confidence=_as_float(data.get("estimated_basin_error_y_confidence"), 0.0),
        z_confidence=_as_float(data.get("estimated_basin_error_z_confidence"), 0.0),
        yaw_confidence=_as_float(data.get("estimated_basin_error_yaw_confidence"), 0.0),
        frame_consistency=_as_float(data.get("estimated_basin_error_frame_consistency"), 0.0),
        source=str(data.get("estimated_basin_error_source", "trace")),
        reason=str(data.get("estimated_basin_error_reason", "")),
        target_entity=str(data.get("estimated_basin_error_target_entity", "")),
        reference_entity=str(data.get("estimated_basin_error_reference_entity", "")),
        stage_name=str(data.get("estimated_basin_error_stage_name", "")),
        proxy_dx=_as_float(data.get("estimated_basin_error_proxy_dx"), 0.0),
        proxy_dy=_as_float(data.get("estimated_basin_error_proxy_dy"), 0.0),
        proxy_dz=_as_float(data.get("estimated_basin_error_proxy_dz"), 0.0),
        proxy_dyaw=_as_float(data.get("estimated_basin_error_proxy_dyaw"), 0.0),
        proxy_image_axis_yaw=_as_float(data.get("estimated_basin_error_proxy_image_axis_yaw"), 0.0),
    )


@dataclass(frozen=True)
class GraspFrameResidualEstimate:
    """Runtime-only grasp XY estimate with explicit validity semantics."""

    valid: bool
    dx: float
    dy: float
    confidence: float
    visual_evidence_valid: bool
    calibrated_proxy_valid: bool
    contract_aligned: bool
    entry_ready: bool
    reason: str
    source: str = "runtime_xy_residual_baseline"
    observability: float = 0.0
    frame_consistency: float = 0.0
    xy_direction_confidence: float = 0.0
    xy_sign_stability: float = 0.0
    xy_step_scale: float = 1.0
    xy_risk_reason: str = ""
    xy_stall_reason: str = ""
    uses_privileged_runtime: bool = False
    yaw_control_allowed: bool = False
    close_control_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "grasp_runtime_xy_residual_v1",
            "valid": bool(self.valid),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "confidence": float(self.confidence),
            "visual_evidence_valid": bool(self.visual_evidence_valid),
            "calibrated_proxy_valid": bool(self.calibrated_proxy_valid),
            "contract_aligned": bool(self.contract_aligned),
            "entry_ready": bool(self.entry_ready),
            "reason": str(self.reason),
            "source": str(self.source),
            "observability": float(self.observability),
            "frame_consistency": float(self.frame_consistency),
            "xy_direction_confidence": float(self.xy_direction_confidence),
            "xy_sign_stability": float(self.xy_sign_stability),
            "xy_step_scale": float(self.xy_step_scale),
            "xy_risk_reason": str(self.xy_risk_reason),
            "xy_stall_reason": str(self.xy_stall_reason),
            "uses_privileged_runtime": bool(self.uses_privileged_runtime),
            "yaw_control_allowed": bool(self.yaw_control_allowed),
            "close_control_allowed": bool(self.close_control_allowed),
        }


@dataclass(frozen=True)
class RuntimeXYAffineCalibration:
    """Lightweight non-privileged proxy-to-XY affine calibration."""

    feature_names: tuple[str, ...]
    weights: np.ndarray
    bias: np.ndarray
    source: str = "runtime_xy_affine_calibration"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeXYAffineCalibration | RuntimeXYMLPCalibration":
        schema = str(data.get("schema_version", ""))
        model_type = str(data.get("model_type", ""))
        if schema in {
            "c2c_v2_runtime_xy_mlp_calibration_v1",
            "c2c_v2_runtime_xy_context_mlp_calibration_v1",
            "c2c_v2_runtime_xy_spatial_temporal_calibration_v1",
        } or model_type in {"mlp", "temporal_mlp", "spatial_temporal"}:
            if schema == "c2c_v2_runtime_xy_spatial_temporal_calibration_v1" or model_type == "spatial_temporal":
                return RuntimeXYSpatialTemporalCalibration.from_dict(data)
            return RuntimeXYMLPCalibration.from_dict(data)
        feature_names = tuple(str(x) for x in data.get("feature_names", ()))
        weights = np.asarray(data.get("weights", []), dtype=np.float32)
        bias = np.asarray(data.get("bias", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2]
        if weights.shape != (2, len(feature_names)):
            raise ValueError(f"invalid runtime XY calibration weights shape {weights.shape}; expected {(2, len(feature_names))}")
        if bias.size != 2:
            raise ValueError("runtime XY calibration bias must have length 2")
        return cls(feature_names=feature_names, weights=weights, bias=bias, source=str(data.get("source", "runtime_xy_affine_calibration")))

    @classmethod
    def load(cls, path: str | Path | None) -> "RuntimeXYAffineCalibration | RuntimeXYMLPCalibration | None":
        if path is None or not str(path):
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"missing runtime XY calibration: {p}")
        if p.suffix.lower() in {".pt", ".pth"}:
            state = torch.load(p, map_location="cpu")
            if not isinstance(state, Mapping):
                raise ValueError(f"runtime XY spatial-temporal checkpoint must be a mapping: {p}")
            schema = str(state.get("schema_version", ""))
            model_type = str(state.get("model_type", ""))
            if schema == "c2c_v2_runtime_xy_spatial_temporal_checkpoint_v1" or model_type == "spatial_temporal":
                return RuntimeXYSpatialTemporalCalibration.from_dict(state)
            raise ValueError(f"unrecognized runtime XY torch checkpoint schema: {schema or model_type or p}")
        with open(p, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "c2c_v2_runtime_xy_affine_calibration_v1",
            "feature_names": list(self.feature_names),
            "weights": self.weights.astype(float).tolist(),
            "bias": self.bias.astype(float).tolist(),
            "source": str(self.source),
        }

    def predict_from_trace(self, row: Mapping[str, Any], history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None) -> tuple[np.ndarray, np.ndarray]:
        features = runtime_xy_feature_vector_from_trace(row, self.feature_names)
        pred = self.weights @ features + self.bias
        return pred.astype(np.float32), features.astype(np.float32)


@dataclass(frozen=True)
class RuntimeXYMLPCalibration:
    """Small JSON-serializable MLP for non-privileged XY residual calibration."""

    feature_names: tuple[str, ...]
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    window_size: int = 1
    base_feature_names: tuple[str, ...] = ()
    source: str = "runtime_xy_mlp_calibration"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeXYMLPCalibration":
        feature_names = tuple(str(x) for x in data.get("feature_names", ()))
        window_size = max(1, int(data.get("window_size", 1)))
        base_feature_names = tuple(str(x) for x in data.get("base_feature_names", ()))
        if window_size > 1:
            if not base_feature_names:
                base_feature_names = _infer_runtime_xy_base_feature_names(feature_names, window_size)
            expected_feature_names = runtime_xy_context_feature_names(base_feature_names, window_size)
            if not feature_names:
                feature_names = expected_feature_names
            elif tuple(feature_names) != tuple(expected_feature_names):
                raise ValueError("runtime XY temporal MLP feature_names do not match the configured window size")
        elif not base_feature_names:
            base_feature_names = feature_names
        layers_payload = data.get("layers", [])
        layers: list[tuple[np.ndarray, np.ndarray]] = []
        in_dim = len(feature_names)
        for i, layer in enumerate(layers_payload):
            if not isinstance(layer, Mapping):
                raise ValueError("runtime XY MLP layer must be a mapping")
            weights = np.asarray(layer.get("weights", []), dtype=np.float32)
            bias = np.asarray(layer.get("bias", []), dtype=np.float32).reshape(-1)
            if weights.ndim != 2:
                raise ValueError("runtime XY MLP layer weights must be 2D")
            if weights.shape[1] != in_dim:
                raise ValueError(f"invalid runtime XY MLP layer {i} input dim {weights.shape[1]}; expected {in_dim}")
            if bias.shape != (weights.shape[0],):
                raise ValueError(f"invalid runtime XY MLP layer {i} bias shape {bias.shape}; expected {(weights.shape[0],)}")
            layers.append((weights, bias))
            in_dim = weights.shape[0]
        if not layers or layers[-1][0].shape[0] != 2:
            raise ValueError("runtime XY MLP must end with 2 outputs")
        mean = np.asarray(data.get("feature_mean", [0.0] * len(feature_names)), dtype=np.float32).reshape(-1)
        std = np.asarray(data.get("feature_std", [1.0] * len(feature_names)), dtype=np.float32).reshape(-1)
        if mean.shape != (len(feature_names),) or std.shape != (len(feature_names),):
            raise ValueError("runtime XY MLP normalization shape must match feature_names")
        std_floor = RUNTIME_XY_NORMALIZATION_STD_FLOOR if int(window_size) > 1 else 1.0e-6
        std_replacement = RUNTIME_XY_NORMALIZATION_STD_FLOOR if int(window_size) > 1 else 1.0
        std = np.where(np.abs(std) < std_floor, std_replacement, std).astype(np.float32)
        return cls(
            feature_names=feature_names,
            layers=tuple(layers),
            feature_mean=mean.astype(np.float32),
            feature_std=std.astype(np.float32),
            window_size=int(window_size),
            base_feature_names=base_feature_names if window_size > 1 else feature_names,
            source=str(data.get("source", "runtime_xy_mlp_calibration")),
        )

    def to_dict(self) -> dict[str, Any]:
        schema_version = "c2c_v2_runtime_xy_context_mlp_calibration_v1" if int(self.window_size) > 1 else "c2c_v2_runtime_xy_mlp_calibration_v1"
        payload = {
            "schema_version": schema_version,
            "model_type": "temporal_mlp" if int(self.window_size) > 1 else "mlp",
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.astype(float).tolist(),
            "feature_std": self.feature_std.astype(float).tolist(),
            "layers": [
                {
                    "weights": weights.astype(float).tolist(),
                    "bias": bias.astype(float).tolist(),
                }
                for weights, bias in self.layers
            ],
            "source": str(self.source),
            "window_size": int(self.window_size),
        }
        if int(self.window_size) > 1:
            payload["base_feature_names"] = list(self.base_feature_names or self.feature_names)
        return payload

    def predict_from_trace(
        self,
        row: Mapping[str, Any],
        history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if int(self.window_size) > 1:
            base_feature_names = self.base_feature_names or _infer_runtime_xy_base_feature_names(tuple(self.feature_names), int(self.window_size))
            features = runtime_xy_context_feature_vector_from_trace(
                row,
                history_rows=history_rows,
                base_feature_names=base_feature_names,
                window_size=int(self.window_size),
            )
        else:
            features = runtime_xy_feature_vector_from_trace(row, self.feature_names)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x = ((features.astype(np.float64) - self.feature_mean.astype(np.float64)) / self.feature_std.astype(np.float64)).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=RUNTIME_XY_NORMALIZED_FEATURE_CLIP, neginf=-RUNTIME_XY_NORMALIZED_FEATURE_CLIP)
        x = np.clip(x, -RUNTIME_XY_NORMALIZED_FEATURE_CLIP, RUNTIME_XY_NORMALIZED_FEATURE_CLIP).astype(np.float32)
        for i, (weights, bias) in enumerate(self.layers):
            x = weights @ x + bias
            x = np.nan_to_num(x, nan=0.0, posinf=RUNTIME_XY_MAX_RESIDUAL_NORM, neginf=-RUNTIME_XY_MAX_RESIDUAL_NORM)
            if i < len(self.layers) - 1:
                x = np.clip(np.maximum(x, 0.0), 0.0, RUNTIME_XY_NORMALIZED_FEATURE_CLIP).astype(np.float32)
        pred = _clip_xy_norm(x[:2], RUNTIME_XY_MAX_RESIDUAL_NORM)
        return pred.astype(np.float32), features.astype(np.float32)

    def context_ready_from_trace(
        self,
        row: Mapping[str, Any],
        history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    ) -> tuple[bool, str, int]:
        if int(self.window_size) <= 1:
            base = estimate_runtime_xy_residual_from_trace(row)
            return bool(base.entry_ready), str(base.reason), int(1 if base.entry_ready else 0)
        ready, reason, support_rows = _runtime_xy_context_support_ready(row, history_rows, window_size=int(self.window_size))
        return bool(ready), str(reason), int(support_rows)


DEFAULT_RUNTIME_XY_FEATURE_NAMES: tuple[str, ...] = (
    "local_dx",
    "local_dy",
    "estimated_proxy_dx",
    "estimated_proxy_dy",
    "local_confidence",
    "local_observability",
    "local_fit_residual",
    "local_inlier_ratio",
    "frame_consistency",
)

RUNTIME_XY_CONTEXT_SCALAR_FEATURE_NAMES: tuple[str, ...] = (
    "c2c_stage_age",
    "wrist_valid_depth_ratio",
    "wrist_depth_near_fraction",
    "wrist_is_occluded",
    "wrist_is_low_visibility",
    "c2c_gate_localizer_visible",
    "c2c_gate_depth_nearfield",
    "c2c_gate_target_xy_error",
    "grasp_probe_pre_xy_error",
    "grasp_probe_residual_norm_xy",
    "grasp_probe_runtime_estimator_residual_norm_xy",
    "grasp_probe_applied_xy_step_x",
    "grasp_probe_applied_xy_step_y",
    "grasp_probe_local_command_x",
    "grasp_probe_local_command_y",
)


def runtime_xy_context_feature_names(
    base_feature_names: tuple[str, ...] = DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    window_size: int = 1,
) -> tuple[str, ...]:
    window = max(1, int(window_size))
    names: list[str] = []
    for lag in range(window):
        prefix = f"lag{lag}"
        names.extend(f"{prefix}_{name}" for name in base_feature_names)
        names.extend(f"{prefix}_{name}" for name in RUNTIME_XY_CONTEXT_SCALAR_FEATURE_NAMES)
        names.append(f"{prefix}_history_valid")
    return tuple(names)


def _infer_runtime_xy_base_feature_names(feature_names: tuple[str, ...], window_size: int) -> tuple[str, ...]:
    if int(window_size) <= 1:
        return tuple(feature_names)
    prefix = "lag0_"
    base: list[str] = []
    for name in feature_names:
        if not str(name).startswith(prefix):
            break
        suffix = str(name)[len(prefix) :]
        if suffix in RUNTIME_XY_CONTEXT_SCALAR_FEATURE_NAMES or suffix == "history_valid":
            break
        base.append(suffix)
    return tuple(base) if base else tuple(DEFAULT_RUNTIME_XY_FEATURE_NAMES)


def _xy_step_xy(row: Mapping[str, Any], key: str) -> np.ndarray:
    value = row.get(key, None)
    if value is None:
        return np.zeros((2,), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return np.zeros((2,), dtype=np.float32)
    return arr[:2].astype(np.float32)


def _runtime_xy_context_scalar_vector_from_trace(row: Mapping[str, Any]) -> np.ndarray:
    c2c_gate_visible = 1.0 if _as_bool(row.get("c2c_gate_localizer_visible"), False) else 0.0
    c2c_gate_depth_nearfield = 1.0 if _as_bool(row.get("c2c_gate_depth_nearfield"), False) else 0.0
    values = {
        "c2c_stage_age": _as_float(row.get("c2c_stage_age"), 0.0),
        "wrist_valid_depth_ratio": _as_float(row.get("wrist_valid_depth_ratio"), 0.0),
        "wrist_depth_near_fraction": _as_float(row.get("wrist_depth_near_fraction"), 0.0),
        "wrist_is_occluded": 1.0 if _as_bool(row.get("wrist_is_occluded"), False) else 0.0,
        "wrist_is_low_visibility": 1.0 if _as_bool(row.get("wrist_is_low_visibility"), False) else 0.0,
        "c2c_gate_localizer_visible": c2c_gate_visible,
        "c2c_gate_depth_nearfield": c2c_gate_depth_nearfield,
        "c2c_gate_target_xy_error": _as_float(row.get("c2c_gate_target_xy_error"), 0.0),
        "grasp_probe_pre_xy_error": _as_float(row.get("grasp_probe_pre_xy_error"), 0.0),
        "grasp_probe_residual_norm_xy": _as_float(row.get("grasp_probe_residual_norm_xy"), 0.0),
        "grasp_probe_runtime_estimator_residual_norm_xy": _as_float(row.get("grasp_probe_runtime_estimator_residual_norm_xy"), 0.0),
        "grasp_probe_applied_xy_step_x": float(_xy_step_xy(row, "grasp_probe_applied_xy_step_local_6d")[0]),
        "grasp_probe_applied_xy_step_y": float(_xy_step_xy(row, "grasp_probe_applied_xy_step_local_6d")[1]),
        "grasp_probe_local_command_x": float(_xy_step_xy(row, "grasp_probe_local_command_local_6d")[0]),
        "grasp_probe_local_command_y": float(_xy_step_xy(row, "grasp_probe_local_command_local_6d")[1]),
    }
    return np.asarray([values.get(str(name), 0.0) for name in RUNTIME_XY_CONTEXT_SCALAR_FEATURE_NAMES], dtype=np.float32)


def runtime_xy_context_feature_vector_from_trace(
    row: Mapping[str, Any],
    history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    *,
    base_feature_names: tuple[str, ...] = DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    window_size: int = 1,
) -> np.ndarray:
    window = max(1, int(window_size))
    history = list(history_rows or ())
    rows = [row] + history[: max(0, window - 1)]
    features: list[np.ndarray] = []
    for lag in range(window):
        current = rows[lag] if lag < len(rows) else None
        if current is None:
            features.append(np.zeros((len(base_feature_names) + len(RUNTIME_XY_CONTEXT_SCALAR_FEATURE_NAMES) + 1,), dtype=np.float32))
            continue
        base = runtime_xy_feature_vector_from_trace(current, base_feature_names)
        scalar = _runtime_xy_context_scalar_vector_from_trace(current)
        history_valid = np.asarray([1.0], dtype=np.float32)
        if lag >= len(rows):
            history_valid[:] = 0.0
        features.append(np.concatenate([base, scalar, history_valid], axis=0).astype(np.float32))
    return np.concatenate(features, axis=0).astype(np.float32)


RUNTIME_XY_SPATIOTEMPORAL_SCALAR_FEATURE_NAMES: tuple[str, ...] = (
    "local_dx",
    "local_dy",
    "estimated_proxy_dx",
    "estimated_proxy_dy",
    "local_confidence",
    "local_observability",
    "local_fit_residual",
    "local_inlier_ratio",
    "frame_consistency",
    "wrist_valid_depth_ratio",
    "wrist_depth_near_fraction",
    "wrist_is_occluded",
    "wrist_is_low_visibility",
    "c2c_gate_localizer_visible",
    "c2c_gate_depth_nearfield",
    "c2c_gate_target_xy_error",
    "grasp_probe_pre_xy_error",
    "grasp_probe_residual_norm_xy",
    "grasp_probe_runtime_estimator_residual_norm_xy",
    "grasp_probe_applied_xy_step_x",
    "grasp_probe_applied_xy_step_y",
    "grasp_probe_local_command_x",
    "grasp_probe_local_command_y",
    "planner_prior_local_x",
    "planner_prior_local_y",
    "planner_prior_local_norm",
    "runtime_xy_estimator_dx",
    "runtime_xy_estimator_dy",
    "runtime_xy_estimator_confidence",
    "runtime_xy_estimator_entry_ready",
    "xy_direction_confidence",
    "xy_sign_stability",
    "xy_step_scale",
    "xy_visible_confidence",
    "history_support_rows",
    "history_recent_support_rows",
    "history_support_age",
)


def runtime_xy_spatial_temporal_feature_names(
    base_feature_names: tuple[str, ...] = DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
) -> tuple[str, ...]:
    window = max(1, int(window_size))
    names: list[str] = []
    for lag in range(window):
        prefix = f"lag{lag}"
        names.extend(f"{prefix}_{name}" for name in base_feature_names)
        names.extend(f"{prefix}_{name}" for name in RUNTIME_XY_SPATIOTEMPORAL_SCALAR_FEATURE_NAMES)
        names.append(f"{prefix}_history_valid")
    return tuple(names)


def runtime_xy_spatial_temporal_context_feature_names(
    base_feature_names: tuple[str, ...] = DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
) -> tuple[str, ...]:
    """Compatibility alias for callers that expect a context-prefixed name."""

    return runtime_xy_spatial_temporal_feature_names(base_feature_names, window_size)


def _planner_prior_local_2d(row: Mapping[str, Any]) -> np.ndarray:
    for key in ("grasp_probe_local_command_local_6d", "planner_chunk_local_6d", "planner_prior_delta_local_6d", "planner_prior_delta"):
        value = row.get(key, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            return arr[:2].astype(np.float32)
    return np.zeros((2,), dtype=np.float32)


def _runtime_xy_spatial_temporal_scalar_vector_from_trace(row: Mapping[str, Any]) -> np.ndarray:
    base = _runtime_xy_context_scalar_vector_from_trace(row)
    grasp = _nested(row, "local_geometry_error", "grasp")
    est = row.get("estimated_basin_error", {})
    if not isinstance(est, Mapping):
        est = {}
    runtime_est = row.get("runtime_xy_estimator", {})
    if not isinstance(runtime_est, Mapping):
        runtime_est = {}
    history_support_rows = float(row.get("runtime_xy_estimator_history_rows", row.get("xy_history_support_rows", 0)) or 0.0)
    history_recent_support_rows = float(row.get("runtime_xy_estimator_recent_support_rows", row.get("xy_history_recent_support_rows", 0)) or 0.0)
    history_support_age = float(row.get("runtime_xy_estimator_support_age", row.get("xy_history_support_age", 0)) or 0.0)
    values = {
        "local_dx": _as_float(grasp.get("dx"), 0.0),
        "local_dy": _as_float(grasp.get("dy"), 0.0),
        "estimated_proxy_dx": _as_float(est.get("estimated_basin_error_proxy_dx", est.get("proxy_dx")), 0.0),
        "estimated_proxy_dy": _as_float(est.get("estimated_basin_error_proxy_dy", est.get("proxy_dy")), 0.0),
        "local_confidence": _as_float(grasp.get("confidence"), 0.0),
        "local_observability": _as_float(grasp.get("observability"), 0.0),
        "local_fit_residual": _as_float(grasp.get("fit_residual"), 0.0),
        "local_inlier_ratio": _as_float(grasp.get("inlier_ratio"), 0.0),
        "frame_consistency": _as_float(est.get("estimated_basin_error_frame_consistency", est.get("frame_consistency")), 0.0),
        "wrist_valid_depth_ratio": _as_float(row.get("wrist_valid_depth_ratio"), 0.0),
        "wrist_depth_near_fraction": _as_float(row.get("wrist_depth_near_fraction"), 0.0),
        "wrist_is_occluded": 1.0 if _as_bool(row.get("wrist_is_occluded"), False) else 0.0,
        "wrist_is_low_visibility": 1.0 if _as_bool(row.get("wrist_is_low_visibility"), False) else 0.0,
        "c2c_gate_localizer_visible": 1.0 if _as_bool(row.get("c2c_gate_localizer_visible"), False) else 0.0,
        "c2c_gate_depth_nearfield": 1.0 if _as_bool(row.get("c2c_gate_depth_nearfield"), False) else 0.0,
        "c2c_gate_target_xy_error": _as_float(row.get("c2c_gate_target_xy_error"), 0.0),
        "grasp_probe_pre_xy_error": _as_float(row.get("grasp_probe_pre_xy_error"), 0.0),
        "grasp_probe_residual_norm_xy": _as_float(row.get("grasp_probe_residual_norm_xy"), 0.0),
        "grasp_probe_runtime_estimator_residual_norm_xy": _as_float(row.get("grasp_probe_runtime_estimator_residual_norm_xy"), 0.0),
        "grasp_probe_applied_xy_step_x": float(_xy_step_xy(row, "grasp_probe_applied_xy_step_local_6d")[0]),
        "grasp_probe_applied_xy_step_y": float(_xy_step_xy(row, "grasp_probe_applied_xy_step_local_6d")[1]),
        "grasp_probe_local_command_x": float(_xy_step_xy(row, "grasp_probe_local_command_local_6d")[0]),
        "grasp_probe_local_command_y": float(_xy_step_xy(row, "grasp_probe_local_command_local_6d")[1]),
        "planner_prior_local_x": float(_planner_prior_local_2d(row)[0]),
        "planner_prior_local_y": float(_planner_prior_local_2d(row)[1]),
        "planner_prior_local_norm": float(np.linalg.norm(_planner_prior_local_2d(row))),
        "runtime_xy_estimator_dx": _as_float(runtime_est.get("dx"), 0.0),
        "runtime_xy_estimator_dy": _as_float(runtime_est.get("dy"), 0.0),
        "runtime_xy_estimator_confidence": _as_float(runtime_est.get("confidence"), 0.0),
        "runtime_xy_estimator_entry_ready": 1.0 if _as_bool(runtime_est.get("entry_ready"), False) else 0.0,
        "xy_direction_confidence": _as_float(row.get("xy_direction_confidence"), 0.0),
        "xy_sign_stability": _as_float(row.get("xy_sign_stability"), 0.0),
        "xy_step_scale": _as_float(row.get("xy_step_scale"), 1.0),
        "xy_visible_confidence": _as_float(row.get("xy_visible_confidence", runtime_est.get("confidence", 0.0)), 0.0),
        "history_support_rows": history_support_rows,
        "history_recent_support_rows": history_recent_support_rows,
        "history_support_age": history_support_age,
    }
    return np.asarray([values.get(str(name), 0.0) for name in RUNTIME_XY_SPATIOTEMPORAL_SCALAR_FEATURE_NAMES], dtype=np.float32)


def runtime_xy_spatial_temporal_context_feature_vector_from_trace(
    row: Mapping[str, Any],
    history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    *,
    base_feature_names: tuple[str, ...] = DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
) -> np.ndarray:
    window = max(1, int(window_size))
    history = list(history_rows or ())
    rows = [row] + history[: max(0, window - 1)]
    features: list[np.ndarray] = []
    for lag in range(window):
        current = rows[lag] if lag < len(rows) else None
        if current is None:
            features.append(np.zeros((len(base_feature_names) + len(RUNTIME_XY_SPATIOTEMPORAL_SCALAR_FEATURE_NAMES) + 1,), dtype=np.float32))
            continue
        base = runtime_xy_feature_vector_from_trace(current, base_feature_names)
        scalar = _runtime_xy_spatial_temporal_scalar_vector_from_trace(current)
        history_valid = np.asarray([1.0], dtype=np.float32)
        if lag >= len(rows):
            history_valid[:] = 0.0
        features.append(np.concatenate([base, scalar, history_valid], axis=0).astype(np.float32))
    return np.concatenate(features, axis=0).astype(np.float32)


def _runtime_xy_signal_available(row: Mapping[str, Any]) -> bool:
    runtime_estimate = row.get("runtime_xy_estimator", {})
    if isinstance(runtime_estimate, Mapping):
        if _as_bool(runtime_estimate.get("entry_ready"), False) or _as_bool(runtime_estimate.get("valid"), False):
            return True
    grasp = _nested(row, "local_geometry_error", "grasp")
    if grasp:
        conf = _as_float(grasp.get("confidence"), 0.0)
        obs = _as_float(grasp.get("observability"), 0.0)
        fit = _as_float(grasp.get("fit_residual"), 0.0)
        inlier = _as_float(grasp.get("inlier_ratio"), 0.0)
        if (_as_bool(grasp.get("valid"), False) or conf >= 0.10 or obs >= 5.0e-4 or inlier >= 0.20 or fit > 0.0) and (conf >= 0.05 or obs >= 5.0e-4 or inlier >= 0.20):
            return True
    est = row.get("estimated_basin_error", {})
    if isinstance(est, Mapping):
        valid = _as_bool(est.get("estimated_basin_error_valid", est.get("valid", False)), False)
        x_valid = _as_bool(est.get("estimated_basin_error_x_valid", est.get("x_valid", False)), False)
        y_valid = _as_bool(est.get("estimated_basin_error_y_valid", est.get("y_valid", False)), False)
        frame_consistency = _as_float(est.get("estimated_basin_error_frame_consistency", est.get("frame_consistency", 0.0)), 0.0)
        if valid and x_valid and y_valid and frame_consistency >= 0.20:
            return True
    return False


def _runtime_xy_low_visibility(row: Mapping[str, Any], base: GraspFrameResidualEstimate | None = None) -> bool:
    if base is not None and not bool(base.visual_evidence_valid):
        return True
    if _as_bool(row.get("wrist_is_occluded"), False) or _as_bool(row.get("wrist_is_low_visibility"), False):
        return True
    vis = str(row.get("visual_observability_class", "")).strip().lower()
    return vis in {"occluded", "low_visibility", "low_observability", "partial_observable", "partial_observation"}


def _runtime_xy_context_support_metrics(
    row: Mapping[str, Any],
    history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
    *,
    window_size: int,
) -> dict[str, Any]:
    window = max(1, int(window_size))
    rows = [row] + list(history_rows or ())[: max(0, window - 1)]
    support_indices = [idx for idx, item in enumerate(rows) if _runtime_xy_signal_available(item)]
    support_rows = [item for item in rows if _runtime_xy_signal_available(item)]
    support_vectors: list[np.ndarray] = []
    for item in support_rows:
        vec = runtime_xy_feature_vector_from_trace(item, DEFAULT_RUNTIME_XY_FEATURE_NAMES)
        xy = np.asarray(vec[:2], dtype=np.float32).reshape(-1)
        if xy.size >= 2 and np.all(np.isfinite(xy[:2])):
            norm = float(np.linalg.norm(xy[:2]))
            if norm > 1.0e-9:
                support_vectors.append(xy[:2].astype(np.float32))
    support_count = int(len(support_rows))
    recent_support_rows = int(sum(1 for idx in support_indices if idx <= min(2, max(0, window - 1))))
    support_age = int(min(support_indices) if support_indices else window)
    direction_confidence = 0.0
    sign_stability = 0.0
    support_mean_xy = np.zeros((2,), dtype=np.float32)
    if support_vectors:
        stacked = np.stack(support_vectors).astype(np.float32)
        mean_vec = np.mean(stacked, axis=0).astype(np.float32)
        support_mean_xy = _clip_xy_norm(mean_vec, RUNTIME_XY_MAX_RESIDUAL_NORM)
        mean_norm = float(np.linalg.norm(mean_vec))
        if mean_norm > 1.0e-9:
            cosines: list[float] = []
            sign_matches: list[float] = []
            mean_sign = np.sign(mean_vec)
            for vec in stacked:
                norm = float(np.linalg.norm(vec))
                if norm > 1.0e-9:
                    cosines.append(float(np.dot(vec, mean_vec) / max(norm * mean_norm, 1.0e-9)))
                sign_matches.append(float(np.mean(np.sign(vec) == mean_sign)))
            direction_confidence = float(np.clip(max(0.0, float(np.mean(cosines)) if cosines else 0.0), 0.0, 1.0))
            sign_stability = float(np.clip(float(np.mean(sign_matches)) if sign_matches else 0.0, 0.0, 1.0))
    return {
        "support_rows": support_count,
        "recent_support_rows": recent_support_rows,
        "support_age": support_age,
        "direction_confidence": direction_confidence,
        "sign_stability": sign_stability,
        "support_mean_xy": support_mean_xy,
        "low_visibility": bool(_runtime_xy_low_visibility(row)),
    }


def _runtime_xy_context_support_ready(
    row: Mapping[str, Any],
    history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
    *,
    window_size: int,
) -> tuple[bool, str, int]:
    metrics = _runtime_xy_context_support_metrics(row, history_rows, window_size=window_size)
    support_rows = int(metrics["support_rows"])
    recent_support_rows = int(metrics.get("recent_support_rows", 0))
    support_age = int(metrics.get("support_age", window_size))
    if bool(metrics["low_visibility"]):
        if (
            support_rows >= 2
            and recent_support_rows >= 2
            and support_age <= min(2, max(0, int(window_size) - 1))
            and float(metrics["direction_confidence"]) >= 0.60
            and float(metrics["sign_stability"]) >= 0.80
        ):
            return True, "history_supported_low_visibility", support_rows
        if support_rows >= 2:
            return False, "direction_stability_low", support_rows
        if support_rows >= 1:
            return False, "support_insufficient_for_low_visibility", support_rows
        return False, "insufficient_context_support", support_rows
    if support_rows >= 1:
        return True, "ready", support_rows
    return False, "insufficient_context_support", support_rows


def _runtime_xy_context_policy(
    row: Mapping[str, Any],
    history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
    *,
    window_size: int,
) -> dict[str, Any]:
    base = estimate_runtime_xy_residual_from_trace(row)
    metrics = _runtime_xy_context_support_metrics(row, history_rows, window_size=window_size)
    support_rows = int(metrics["support_rows"])
    recent_support_rows = int(metrics.get("recent_support_rows", 0))
    support_age = int(metrics.get("support_age", window_size))
    direction_confidence = float(metrics["direction_confidence"])
    sign_stability = float(metrics["sign_stability"])
    low_visibility = bool(metrics["low_visibility"])
    ready = bool(base.entry_ready)
    reason = str(base.reason)
    step_scale = 1.0
    risk_reason = ""
    stall_reason = ""
    if low_visibility:
        risk_reason = "low_visibility_temporal_support"
        if (
            support_rows >= 2
            and recent_support_rows >= 2
            and support_age <= min(2, max(0, int(window_size) - 1))
            and direction_confidence >= 0.60
            and sign_stability >= 0.80
        ):
            ready = True
            reason = "history_supported_low_visibility"
            step_scale = 0.35
        else:
            ready = False
            if support_rows >= 2:
                reason = "direction_stability_low"
                step_scale = 0.15
            elif support_rows >= 1:
                reason = "support_insufficient_for_low_visibility"
                step_scale = 0.08
            else:
                reason = "insufficient_context_support"
                step_scale = 0.0
            stall_reason = reason
    return {
        "base": base,
        "ready": bool(ready),
        "reason": str(reason),
        "support_rows": support_rows,
        "direction_confidence": direction_confidence,
        "sign_stability": sign_stability,
        "step_scale": float(step_scale),
        "risk_reason": str(risk_reason),
        "stall_reason": str(stall_reason),
        "low_visibility": low_visibility,
    }


def runtime_xy_feature_vector_from_trace(row: Mapping[str, Any], feature_names: tuple[str, ...] = DEFAULT_RUNTIME_XY_FEATURE_NAMES) -> np.ndarray:
    grasp = _nested(row, "local_geometry_error", "grasp")
    est = row.get("estimated_basin_error", {})
    if not isinstance(est, Mapping):
        est = {}
    values = {
        "local_dx": _as_float(grasp.get("dx"), 0.0),
        "local_dy": _as_float(grasp.get("dy"), 0.0),
        "estimated_dx": _as_float(est.get("estimated_basin_error_dx", est.get("dx")), 0.0),
        "estimated_dy": _as_float(est.get("estimated_basin_error_dy", est.get("dy")), 0.0),
        "estimated_proxy_dx": _as_float(est.get("estimated_basin_error_proxy_dx", est.get("proxy_dx", grasp.get("dx"))), 0.0),
        "estimated_proxy_dy": _as_float(est.get("estimated_basin_error_proxy_dy", est.get("proxy_dy", grasp.get("dy"))), 0.0),
        "local_confidence": _as_float(grasp.get("confidence"), 0.0),
        "local_observability": _as_float(grasp.get("observability"), 0.0),
        "local_fit_residual": _as_float(grasp.get("fit_residual"), 0.0),
        "local_inlier_ratio": _as_float(grasp.get("inlier_ratio"), 0.0),
        "frame_consistency": _as_float(est.get("estimated_basin_error_frame_consistency", est.get("frame_consistency")), 0.0),
    }
    return np.asarray([values.get(str(name), 0.0) for name in feature_names], dtype=np.float32)


def calibrated_runtime_xy_residual_from_trace(
    row: Mapping[str, Any],
    calibration: RuntimeXYAffineCalibration | RuntimeXYMLPCalibration | "RuntimeXYSpatialTemporalCalibration" | None,
    history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    observation: Mapping[str, Any] | None = None,
    robot_state: Mapping[str, Any] | None = None,
) -> GraspFrameResidualEstimate:
    base = estimate_runtime_xy_residual_from_trace(row)
    if calibration is None:
        return base
    if isinstance(calibration, RuntimeXYSpatialTemporalCalibration):
        pred, _features, aux = calibration.predict_from_trace(
            row,
            observation=observation,
            robot_state=robot_state,
            history_rows=history_rows,
        )
        estimate = aux.get("estimate")
        if isinstance(estimate, GraspFrameResidualEstimate):
            return estimate
        return GraspFrameResidualEstimate(
            valid=True,
            dx=float(pred[0]),
            dy=float(pred[1]),
            confidence=float(base.confidence),
            visual_evidence_valid=bool(base.visual_evidence_valid),
            calibrated_proxy_valid=bool(base.calibrated_proxy_valid),
            contract_aligned=bool(base.contract_aligned),
            entry_ready=bool(base.entry_ready),
            reason=str(base.reason),
            source=str(calibration.source),
            observability=float(base.observability),
            frame_consistency=float(base.frame_consistency),
            uses_privileged_runtime=False,
            yaw_control_allowed=False,
            close_control_allowed=False,
        )
    if isinstance(calibration, RuntimeXYMLPCalibration) and int(getattr(calibration, "window_size", 1)) > 1:
        policy = _runtime_xy_context_policy(row, history_rows, window_size=int(calibration.window_size))
        temporal_ready = bool(policy["ready"])
        temporal_reason = str(policy["reason"])
        support_rows = int(policy["support_rows"])
        direction_confidence = float(policy["direction_confidence"])
        sign_stability = float(policy["sign_stability"])
        step_scale = float(policy["step_scale"])
        risk_reason = str(policy["risk_reason"])
        stall_reason = str(policy["stall_reason"])
        if bool(policy["low_visibility"]) and not temporal_ready:
            pred, _features = calibration.predict_from_trace(row, history_rows=history_rows)
            pred = _clip_xy_norm(pred, RUNTIME_XY_MAX_RESIDUAL_NORM)
            return GraspFrameResidualEstimate(
                valid=False,
                dx=float(pred[0]),
                dy=float(pred[1]),
                confidence=float(max(base.confidence, direction_confidence)),
                visual_evidence_valid=bool(base.visual_evidence_valid),
                calibrated_proxy_valid=bool(base.calibrated_proxy_valid),
                contract_aligned=False,
                entry_ready=False,
                reason=str(temporal_reason),
                source=str(calibration.source),
                observability=float(base.observability),
                frame_consistency=float(base.frame_consistency),
                xy_direction_confidence=float(direction_confidence),
                xy_sign_stability=float(sign_stability),
                xy_step_scale=float(step_scale),
                xy_risk_reason=str(risk_reason),
                xy_stall_reason=str(stall_reason or temporal_reason),
                uses_privileged_runtime=False,
                yaw_control_allowed=False,
                close_control_allowed=False,
            )
        if not temporal_ready and not bool(base.entry_ready):
            return GraspFrameResidualEstimate(
                valid=False,
                dx=float(base.dx),
                dy=float(base.dy),
                confidence=float(base.confidence),
                visual_evidence_valid=bool(base.visual_evidence_valid),
                calibrated_proxy_valid=bool(base.calibrated_proxy_valid),
                contract_aligned=False,
                entry_ready=False,
                reason=str(temporal_reason),
                source=str(calibration.source),
                observability=float(base.observability),
                frame_consistency=float(base.frame_consistency),
                xy_direction_confidence=float(direction_confidence),
                xy_sign_stability=float(sign_stability),
                xy_step_scale=float(step_scale),
                xy_risk_reason=str(risk_reason),
                xy_stall_reason=str(stall_reason or temporal_reason),
                uses_privileged_runtime=False,
                yaw_control_allowed=False,
                close_control_allowed=False,
            )
        pred, _features = calibration.predict_from_trace(row, history_rows=history_rows)
        pred = _clip_xy_norm(pred, RUNTIME_XY_MAX_RESIDUAL_NORM)
        metrics = _runtime_xy_context_support_metrics(row, history_rows, window_size=int(calibration.window_size))
        support_xy = _clip_xy_norm(np.asarray(metrics.get("support_mean_xy", np.zeros((2,), dtype=np.float32)), dtype=np.float32), RUNTIME_XY_MAX_RESIDUAL_NORM)
        support_norm = float(np.linalg.norm(support_xy))
        pred_norm = float(np.linalg.norm(pred))
        if support_norm > 1.0e-9 and pred_norm > 1.0e-9:
            min_cos = RUNTIME_XY_LOW_VIS_DIRECTION_MIN_COSINE if bool(policy["low_visibility"]) else RUNTIME_XY_DIRECTION_MIN_COSINE
            pred_support_cos = _xy_cosine(pred, support_xy)
            pred_support_sign = _xy_sign_match(pred, support_xy)
            if pred_support_cos < float(min_cos) or pred_support_sign < 0.5:
                risk_reason = "+".join([x for x in (risk_reason, "mlp_direction_conflict") if x])
                stall_reason = "mlp_direction_conflict"
                step_scale = min(float(step_scale), 0.35 if bool(policy["low_visibility"]) else 0.60)
        if temporal_ready:
            pred = (pred.astype(np.float32) * float(step_scale)).astype(np.float32)
            pred = _clip_xy_norm(pred, RUNTIME_XY_MAX_RESIDUAL_NORM)
        return GraspFrameResidualEstimate(
            valid=bool(temporal_ready or base.entry_ready),
            dx=float(pred[0]),
            dy=float(pred[1]),
            confidence=float(max(base.confidence, direction_confidence)),
            visual_evidence_valid=bool(base.visual_evidence_valid),
            calibrated_proxy_valid=bool(base.calibrated_proxy_valid),
            contract_aligned=bool(temporal_ready or base.entry_ready),
            entry_ready=bool(temporal_ready or base.entry_ready),
            reason=str(temporal_reason if temporal_ready else base.reason),
            source=str(calibration.source),
            observability=float(base.observability),
            frame_consistency=float(base.frame_consistency),
            xy_direction_confidence=float(direction_confidence),
            xy_sign_stability=float(sign_stability),
            xy_step_scale=float(step_scale),
            xy_risk_reason=str(risk_reason),
            xy_stall_reason=str(stall_reason),
            uses_privileged_runtime=False,
            yaw_control_allowed=False,
            close_control_allowed=False,
        )
    if not base.entry_ready:
        return base
    pred, _features = calibration.predict_from_trace(row, history_rows=history_rows)
    pred = _clip_xy_norm(pred, RUNTIME_XY_MAX_RESIDUAL_NORM)
    base_xy = _clip_xy_norm(np.asarray([base.dx, base.dy], dtype=np.float32), RUNTIME_XY_MAX_RESIDUAL_NORM)
    if float(np.linalg.norm(base_xy)) > 1.0e-9 and float(np.linalg.norm(pred)) > 1.0e-9:
        if _xy_cosine(pred, base_xy) < RUNTIME_XY_DIRECTION_MIN_COSINE or _xy_sign_match(pred, base_xy) < 0.5:
            pred = base_xy
    return GraspFrameResidualEstimate(
        valid=bool(base.valid),
        dx=float(pred[0]),
        dy=float(pred[1]),
        confidence=float(base.confidence),
        visual_evidence_valid=bool(base.visual_evidence_valid),
        calibrated_proxy_valid=bool(base.calibrated_proxy_valid),
        contract_aligned=bool(base.contract_aligned),
        entry_ready=bool(base.entry_ready),
        reason=str(base.reason),
        source=str(calibration.source),
        observability=float(base.observability),
        frame_consistency=float(base.frame_consistency),
        xy_direction_confidence=float(base.xy_direction_confidence),
        xy_sign_stability=float(base.xy_sign_stability),
        xy_step_scale=1.0,
        xy_risk_reason=str(base.xy_risk_reason),
        xy_stall_reason=str(base.xy_stall_reason),
        uses_privileged_runtime=False,
        yaw_control_allowed=False,
        close_control_allowed=False,
    )


def estimate_runtime_xy_residual(
    local_error: LocalGeometryError | None,
    estimated_basin_error: EstimatedBasinError | None,
    *,
    min_visual_confidence: float = 0.20,
    min_observability: float = 5.0e-4,
    min_frame_consistency: float = 0.20,
    max_xy_error: float = 0.18,
) -> GraspFrameResidualEstimate:
    """Build the current conservative non-privileged XY residual estimate."""

    if local_error is None:
        return GraspFrameResidualEstimate(False, 0.0, 0.0, 0.0, False, False, False, False, "missing_local_geometry")
    visual_ok = bool(
        local_error.valid
        and local_error.confidence >= float(min_visual_confidence)
        and local_error.observability >= float(min_observability)
        and np.isfinite(float(local_error.dx))
        and np.isfinite(float(local_error.dy))
    )
    if estimated_basin_error is None:
        return GraspFrameResidualEstimate(
            False,
            float(local_error.dx),
            float(local_error.dy),
            float(local_error.confidence),
            visual_ok,
            False,
            False,
            False,
            "missing_estimated_basin_error" if visual_ok else str(local_error.reason),
            observability=float(local_error.observability),
        )
    xy_norm = float(np.hypot(float(estimated_basin_error.dx), float(estimated_basin_error.dy)))
    proxy_ok = bool(
        estimated_basin_error.valid
        and estimated_basin_error.x_valid
        and estimated_basin_error.y_valid
        and estimated_basin_error.frame_consistency >= float(min_frame_consistency)
        and xy_norm <= float(max_xy_error) + 1.0e-9
    )
    if not visual_ok:
        reason = str(local_error.reason or "visual_evidence_invalid")
    elif not estimated_basin_error.valid:
        reason = str(estimated_basin_error.reason or "estimated_basin_invalid")
    elif not (estimated_basin_error.x_valid and estimated_basin_error.y_valid):
        reason = "no_xy_trusted_axis"
    elif estimated_basin_error.frame_consistency < float(min_frame_consistency):
        reason = "low_frame_consistency"
    elif xy_norm > float(max_xy_error) + 1.0e-9:
        reason = "outside_xy_activation_window"
    else:
        reason = "ready"
    ready = bool(visual_ok and proxy_ok)
    return GraspFrameResidualEstimate(
        valid=ready,
        dx=float(estimated_basin_error.dx),
        dy=float(estimated_basin_error.dy),
        confidence=float(min(estimated_basin_error.x_confidence, estimated_basin_error.y_confidence)),
        visual_evidence_valid=visual_ok,
        calibrated_proxy_valid=proxy_ok,
        contract_aligned=ready,
        entry_ready=ready,
        reason=reason,
        observability=float(local_error.observability),
        frame_consistency=float(estimated_basin_error.frame_consistency),
    )


def estimate_runtime_xy_residual_from_trace(row: Mapping[str, Any]) -> GraspFrameResidualEstimate:
    """Convenience adapter for evaluator trace rows."""

    grasp = _nested(row, "local_geometry_error", "grasp")
    local_error = None
    if grasp:
        local_error = LocalGeometryError(
            valid=_as_bool(grasp.get("valid"), False),
            confidence=_as_float(grasp.get("confidence"), 0.0),
            dx=_as_float(grasp.get("dx"), 0.0),
            dy=_as_float(grasp.get("dy"), 0.0),
            dz=_as_float(grasp.get("dz"), 0.0),
            dyaw=_as_float(grasp.get("dyaw"), 0.0),
            observability=_as_float(grasp.get("observability"), 0.0),
            fit_residual=_as_float(grasp.get("fit_residual"), 0.0),
            inlier_ratio=_as_float(grasp.get("inlier_ratio"), 0.0),
            reason=str(grasp.get("reason", "")),
            target_entity=str(grasp.get("target_entity", "")),
            reference_entity=str(grasp.get("reference_entity", "")),
            stage_name=str(grasp.get("stage_name", "")),
            yaw_valid=_as_bool(grasp.get("yaw_valid"), False),
            yaw_reason=str(grasp.get("yaw_reason", "")),
            image_axis_yaw=_as_float(grasp.get("image_axis_yaw"), 0.0),
        )
    est = _flat_estimated_basin_from_trace(row.get("estimated_basin_error"))
    return estimate_runtime_xy_residual(local_error, est)


def _spatial_temporal_support_metrics(
    row: Mapping[str, Any],
    history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
    *,
    window_size: int,
) -> dict[str, Any]:
    window = max(1, int(window_size))
    rows = [row] + list(history_rows or ())[: max(0, window - 1)]
    support_rows = [item for item in rows if _runtime_xy_signal_available(item)]
    support_vectors: list[np.ndarray] = []
    for item in support_rows:
        vec = runtime_xy_feature_vector_from_trace(item, DEFAULT_RUNTIME_XY_FEATURE_NAMES)
        xy = np.asarray(vec[:2], dtype=np.float32).reshape(-1)
        if xy.size >= 2 and np.all(np.isfinite(xy[:2])):
            support_vectors.append(xy[:2].astype(np.float32))
    support_count = int(len(support_rows))
    recent_support_rows = int(sum(1 for idx, item in enumerate(rows[: min(3, len(rows))]) if _runtime_xy_signal_available(item)))
    support_age = int(
        min([idx for idx, item in enumerate(rows) if _runtime_xy_signal_available(item)] or [window])
    )
    direction_confidence = 0.0
    sign_stability = 0.0
    support_mean_xy = np.zeros((2,), dtype=np.float32)
    if support_vectors:
        stacked = np.stack(support_vectors).astype(np.float32)
        mean_vec = np.mean(stacked, axis=0).astype(np.float32)
        support_mean_xy = _clip_xy_norm(mean_vec, RUNTIME_XY_MAX_RESIDUAL_NORM)
        mean_norm = float(np.linalg.norm(mean_vec))
        if mean_norm > 1.0e-9:
            cosines: list[float] = []
            sign_matches: list[float] = []
            mean_sign = np.sign(mean_vec)
            for vec in stacked:
                norm = float(np.linalg.norm(vec))
                if norm > 1.0e-9:
                    cosines.append(float(np.dot(vec, mean_vec) / max(norm * mean_norm, 1.0e-9)))
                sign_matches.append(float(np.mean(np.sign(vec) == mean_sign)))
            direction_confidence = float(np.clip(max(0.0, float(np.mean(cosines)) if cosines else 0.0), 0.0, 1.0))
            sign_stability = float(np.clip(float(np.mean(sign_matches)) if sign_matches else 0.0, 0.0, 1.0))
    return {
        "support_rows": support_count,
        "recent_support_rows": recent_support_rows,
        "support_age": support_age,
        "direction_confidence": direction_confidence,
        "sign_stability": sign_stability,
        "support_mean_xy": support_mean_xy,
        "low_visibility": bool(_runtime_xy_low_visibility(row)),
    }


def _spatial_temporal_policy(
    row: Mapping[str, Any],
    history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
    *,
    window_size: int,
) -> dict[str, Any]:
    base = estimate_runtime_xy_residual_from_trace(row)
    metrics = _spatial_temporal_support_metrics(row, history_rows, window_size=window_size)
    support_rows = int(metrics["support_rows"])
    recent_support_rows = int(metrics["recent_support_rows"])
    support_age = int(metrics["support_age"])
    direction_confidence = float(metrics["direction_confidence"])
    sign_stability = float(metrics["sign_stability"])
    low_visibility = bool(metrics["low_visibility"])
    ready = bool(base.entry_ready)
    step_scale = 1.0
    risk_reason = ""
    stall_reason = ""
    reason = str(base.reason)
    if low_visibility:
        risk_reason = "low_visibility"
        if support_rows >= 2 and recent_support_rows >= 2 and direction_confidence >= 0.60 and sign_stability >= 0.80:
            ready = True
            reason = "history_supported_low_visibility"
            step_scale = 0.35
        elif support_rows >= 2:
            ready = False
            reason = "direction_stability_low"
            step_scale = 0.15
            stall_reason = "direction_stability_low"
        elif support_rows >= 1:
            ready = False
            reason = "support_insufficient_for_low_visibility"
            step_scale = 0.08
            stall_reason = "support_insufficient_for_low_visibility"
        else:
            ready = False
            reason = "low_visibility_no_support"
            step_scale = 0.0
            stall_reason = "low_visibility_no_support"
    if support_age > int(max(1, window_size)) - 1:
        risk_reason = "+".join(x for x in (risk_reason, "stale_support") if x)
        if ready:
            step_scale = min(step_scale, 0.25)
    return {
        "base": base,
        "ready": bool(ready),
        "reason": str(reason),
        "support_rows": support_rows,
        "recent_support_rows": recent_support_rows,
        "support_age": support_age,
        "direction_confidence": direction_confidence,
        "sign_stability": sign_stability,
        "step_scale": float(step_scale),
        "risk_reason": str(risk_reason),
        "stall_reason": str(stall_reason),
        "low_visibility": low_visibility,
        "support_mean_xy": np.asarray(metrics["support_mean_xy"], dtype=np.float32),
    }


class XYSpatialTemporalHeadNet(nn.Module):
    """Spatial-temporal non-privileged XY estimator."""

    def __init__(
        self,
        *,
        image_in_channels: int = 7,
        image_hidden_dim: int = 128,
        history_feature_dim: int = 37,
        history_window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
        proprio_dim: int = 15,
        planner_prior_dim: int = 6,
        fusion_hidden_dim: int = 96,
        risk_classes: tuple[str, ...] = RUNTIME_XY_SPATIAL_TEMPORAL_RISK_CLASSES,
    ) -> None:
        super().__init__()
        self.image_in_channels = int(image_in_channels)
        self.image_hidden_dim = int(image_hidden_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.history_window_size = int(history_window_size)
        self.proprio_dim = int(proprio_dim)
        self.planner_prior_dim = int(planner_prior_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.risk_classes = tuple(risk_classes)
        self.image_encoder = _ImageEncoder(in_channels=self.image_in_channels, hidden_dim=self.image_hidden_dim)
        self.history_mlp = nn.Sequential(
            nn.Linear(max(1, self.history_feature_dim * self.history_window_size), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.proprio_prior_mlp = nn.Sequential(
            nn.Linear(max(1, self.proprio_dim + self.planner_prior_dim), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.image_hidden_dim + 64 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.xy_head = nn.Linear(self.fusion_hidden_dim, 2)
        self.direction_conf_head = nn.Linear(self.fusion_hidden_dim, 1)
        self.step_scale_head = nn.Linear(self.fusion_hidden_dim, 1)
        self.visible_conf_head = nn.Linear(self.fusion_hidden_dim, 1)
        self.risk_head = nn.Linear(self.fusion_hidden_dim, max(1, len(self.risk_classes)))

    def forward(
        self,
        image_rgbd: torch.Tensor,
        history_features: torch.Tensor,
        proprio: torch.Tensor,
        planner_prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_feat = self.image_encoder(image_rgbd, pooled=True)
        history = history_features.float().reshape(history_features.shape[0], -1)
        if history.shape[-1] < self.history_feature_dim * self.history_window_size:
            history = F.pad(history, (0, self.history_feature_dim * self.history_window_size - history.shape[-1]))
        history_feat = self.history_mlp(history[..., : self.history_feature_dim * self.history_window_size])
        prior = torch.cat([proprio.float(), planner_prior.float()], dim=-1)
        if prior.shape[-1] < self.proprio_dim + self.planner_prior_dim:
            prior = F.pad(prior, (0, self.proprio_dim + self.planner_prior_dim - prior.shape[-1]))
        prior_feat = self.proprio_prior_mlp(prior[..., : self.proprio_dim + self.planner_prior_dim])
        fused = self.fusion(torch.cat([image_feat, history_feat, prior_feat], dim=-1))
        xy = torch.tanh(self.xy_head(fused)) * float(RUNTIME_XY_MAX_RESIDUAL_NORM)
        return {
            "dx": xy[:, 0],
            "dy": xy[:, 1],
            "xy_direction_confidence": torch.sigmoid(self.direction_conf_head(fused)[:, 0]),
            "xy_step_scale": torch.sigmoid(self.step_scale_head(fused)[:, 0]),
            "xy_visible_confidence": torch.sigmoid(self.visible_conf_head(fused)[:, 0]),
            "risk_logits": self.risk_head(fused),
        }


@dataclass(frozen=True)
class RuntimeXYSpatialTemporalCalibration:
    feature_names: tuple[str, ...]
    history_feature_names: tuple[str, ...]
    history_window_size: int
    image_in_channels: int
    image_hidden_dim: int
    image_crop_size: int
    image_resize_size: int
    proprio_dim: int
    planner_prior_dim: int
    risk_classes: tuple[str, ...]
    model_state_dict: dict[str, torch.Tensor]
    source: str = "runtime_xy_spatial_temporal_calibration"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeXYSpatialTemporalCalibration":
        config = dict(data.get("config", {}))
        feature_names = tuple(str(x) for x in config.get("feature_names", data.get("feature_names", ())))
        history_feature_names = tuple(str(x) for x in config.get("history_feature_names", data.get("history_feature_names", ())))
        history_window_size = max(1, int(config.get("history_window_size", data.get("history_window_size", RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW))))
        image_in_channels = int(config.get("image_in_channels", data.get("image_in_channels", 7)))
        image_hidden_dim = int(config.get("image_hidden_dim", data.get("image_hidden_dim", 128)))
        image_crop_size = int(config.get("image_crop_size", data.get("image_crop_size", RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE)))
        image_resize_size = int(config.get("image_resize_size", data.get("image_resize_size", RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE)))
        proprio_dim = int(config.get("proprio_dim", data.get("proprio_dim", 15)))
        planner_prior_dim = int(config.get("planner_prior_dim", data.get("planner_prior_dim", 6)))
        risk_classes = tuple(str(x) for x in config.get("risk_classes", data.get("risk_classes", RUNTIME_XY_SPATIAL_TEMPORAL_RISK_CLASSES)))
        model = XYSpatialTemporalHeadNet(
            image_in_channels=image_in_channels,
            image_hidden_dim=image_hidden_dim,
            history_feature_dim=max(1, len(history_feature_names) // history_window_size if history_feature_names else len(feature_names) // history_window_size),
            history_window_size=history_window_size,
            proprio_dim=proprio_dim,
            planner_prior_dim=planner_prior_dim,
            risk_classes=risk_classes,
        )
        state_dict = {str(k): torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in dict(data.get("model_state_dict", {})).items()}
        model.load_state_dict(state_dict, strict=False)
        return cls(
            feature_names=feature_names,
            history_feature_names=history_feature_names,
            history_window_size=history_window_size,
            image_in_channels=image_in_channels,
            image_hidden_dim=image_hidden_dim,
            image_crop_size=image_crop_size,
            image_resize_size=image_resize_size,
            proprio_dim=proprio_dim,
            planner_prior_dim=planner_prior_dim,
            risk_classes=risk_classes,
            model_state_dict=state_dict,
            source=str(data.get("source", "runtime_xy_spatial_temporal_calibration")),
        ).with_model(model)

    @classmethod
    def load(cls, path: str | Path | None) -> "RuntimeXYSpatialTemporalCalibration | None":
        if path is None or not str(path):
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"missing runtime XY spatial-temporal calibration: {p}")
        data = torch.load(p, map_location="cpu")
        if not isinstance(data, Mapping):
            raise ValueError(f"runtime XY spatial-temporal checkpoint must be a mapping: {p}")
        return cls.from_dict(data)

    def with_model(self, model: XYSpatialTemporalHeadNet) -> "RuntimeXYSpatialTemporalCalibration":
        object.__setattr__(self, "_model", model)
        return self

    @property
    def model(self) -> XYSpatialTemporalHeadNet:
        model = getattr(self, "_model", None)
        if model is None:
            model = XYSpatialTemporalHeadNet(
                image_in_channels=self.image_in_channels,
                image_hidden_dim=self.image_hidden_dim,
                history_feature_dim=max(1, len(self.history_feature_names) // self.history_window_size),
                history_window_size=self.history_window_size,
                proprio_dim=self.proprio_dim,
                planner_prior_dim=self.planner_prior_dim,
                risk_classes=self.risk_classes,
            )
            model.load_state_dict(self.model_state_dict, strict=False)
            object.__setattr__(self, "_model", model)
        return model

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "c2c_v2_runtime_xy_spatial_temporal_checkpoint_v1",
            "model_type": "spatial_temporal",
            "config": {
                "feature_names": list(self.feature_names),
                "history_feature_names": list(self.history_feature_names),
                "history_window_size": int(self.history_window_size),
                "image_in_channels": int(self.image_in_channels),
                "image_hidden_dim": int(self.image_hidden_dim),
                "image_crop_size": int(self.image_crop_size),
                "image_resize_size": int(self.image_resize_size),
                "proprio_dim": int(self.proprio_dim),
                "planner_prior_dim": int(self.planner_prior_dim),
                "risk_classes": list(self.risk_classes),
            },
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model_state_dict.items()},
            "source": str(self.source),
        }

    def predict_from_trace(
        self,
        row: Mapping[str, Any],
        *,
        observation: Mapping[str, Any] | None = None,
        robot_state: Mapping[str, Any] | None = None,
        history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        base = estimate_runtime_xy_residual_from_trace(row)
        if observation is None:
            return np.asarray([base.dx, base.dy], dtype=np.float32), np.zeros((1,), dtype=np.float32), {"risk_reason": "missing_observation", "step_scale": 0.0}
        rgbd = _load_spatial_temporal_rgbd(
            observation,
            robot_state,
            crop_size=self.image_crop_size,
            resize_size=self.image_resize_size,
        )
        if rgbd is None:
            return np.asarray([base.dx, base.dy], dtype=np.float32), np.zeros((1,), dtype=np.float32), {"risk_reason": "missing_rgbd", "step_scale": 0.0}
        history = list(history_rows or ())
        if int(self.history_window_size) > 1:
            history = history[: max(0, self.history_window_size - 1)]
        history_features = runtime_xy_spatial_temporal_context_feature_vector_from_trace(
            row,
            history_rows=history,
            base_feature_names=DEFAULT_RUNTIME_XY_FEATURE_NAMES,
            window_size=int(self.history_window_size),
        )
        if history_features.size == 0:
            history_features = np.zeros((self.history_window_size * len(RUNTIME_XY_SPATIOTEMPORAL_SCALAR_FEATURE_NAMES),), dtype=np.float32)
        proprio = _coerce_vector((robot_state or {}).get("proprio", None), length=self.proprio_dim)
        planner_prior = _coerce_vector((robot_state or {}).get("planner_delta_7d", None), length=self.planner_prior_dim)
        with torch.no_grad():
            out = self.model(
                rgbd.unsqueeze(0),
                torch.from_numpy(history_features.reshape(1, -1)).float(),
                torch.from_numpy(proprio.reshape(1, -1)).float(),
                torch.from_numpy(planner_prior.reshape(1, -1)).float(),
            )
        pred = np.asarray([float(out["dx"][0]), float(out["dy"][0])], dtype=np.float32)
        step_scale = float(out["xy_step_scale"][0])
        direction_confidence = float(out["xy_direction_confidence"][0])
        visible_confidence = float(out["xy_visible_confidence"][0])
        risk_logits = np.asarray(out["risk_logits"][0].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        risk_idx = int(np.argmax(risk_logits)) if risk_logits.size else 0
        risk_reason = RUNTIME_XY_SPATIAL_TEMPORAL_RISK_CLASSES[min(max(risk_idx, 0), len(RUNTIME_XY_SPATIAL_TEMPORAL_RISK_CLASSES) - 1)]
        metrics = _spatial_temporal_support_metrics(row, history_rows, window_size=int(self.history_window_size))
        support_rows = int(metrics["support_rows"])
        recent_support_rows = int(metrics["recent_support_rows"])
        support_age = int(metrics["support_age"])
        support_xy = np.asarray(metrics["support_mean_xy"], dtype=np.float32)
        low_visibility = bool(metrics["low_visibility"])
        policy_step_scale = 1.0
        ready = bool(base.entry_ready)
        reason = str(base.reason)
        stall_reason = ""
        if support_rows >= 2 and float(metrics["direction_confidence"]) >= 0.60 and float(metrics["sign_stability"]) >= 0.80:
            if _xy_cosine(pred, support_xy) < (RUNTIME_XY_LOW_VIS_DIRECTION_MIN_COSINE if low_visibility else RUNTIME_XY_DIRECTION_MIN_COSINE) or _xy_sign_match(pred, support_xy) < 0.5:
                risk_reason = "+".join([x for x in (risk_reason, "direction_conflict") if x])
                stall_reason = "direction_conflict"
                policy_step_scale = min(policy_step_scale, 0.35 if low_visibility else 0.60)
        if low_visibility:
            if support_rows >= 2 and recent_support_rows >= 2 and float(metrics["direction_confidence"]) >= 0.60 and float(metrics["sign_stability"]) >= 0.80:
                ready = True
                policy_step_scale = 0.35
                reason = "history_supported_low_visibility"
            elif support_rows >= 2:
                ready = False
                policy_step_scale = 0.15
                reason = "direction_stability_low"
                stall_reason = "direction_stability_low"
            elif support_rows >= 1:
                ready = False
                policy_step_scale = 0.08
                reason = "support_insufficient_for_low_visibility"
                stall_reason = "support_insufficient_for_low_visibility"
            else:
                ready = False
                policy_step_scale = 0.0
                reason = "low_visibility_no_support"
                stall_reason = "low_visibility_no_support"
        if support_age >= int(max(1, self.history_window_size)) - 1 and ready:
            policy_step_scale = min(policy_step_scale, 0.25)
            risk_reason = "+".join([x for x in (risk_reason, "stale_support") if x])
        pred = _clip_xy_norm(pred, RUNTIME_XY_MAX_RESIDUAL_NORM)
        runtime_step_scale = float(
            np.clip(
                float(step_scale) * float(policy_step_scale) * float(RUNTIME_XY_SPATIOTEMPORAL_STEP_SCALE_SAFETY_MARGIN),
                0.0,
                1.0,
            )
        )
        pred = (pred * runtime_step_scale).astype(np.float32)
        pred = _clip_xy_norm(pred, RUNTIME_XY_MAX_RESIDUAL_NORM)
        estimate = GraspFrameResidualEstimate(
            valid=bool(ready),
            dx=float(pred[0]),
            dy=float(pred[1]),
            confidence=float(max(base.confidence, direction_confidence, visible_confidence)),
            visual_evidence_valid=bool(base.visual_evidence_valid or visible_confidence >= 0.25),
            calibrated_proxy_valid=bool(base.calibrated_proxy_valid),
            contract_aligned=bool(ready),
            entry_ready=bool(ready),
            reason=str(reason),
            source=str(self.source),
            observability=float(base.observability),
            frame_consistency=float(base.frame_consistency),
            xy_direction_confidence=float(direction_confidence),
            xy_sign_stability=float(float(metrics["sign_stability"])),
            xy_step_scale=float(runtime_step_scale),
            xy_risk_reason=str(risk_reason),
            xy_stall_reason=str(stall_reason),
            uses_privileged_runtime=False,
            yaw_control_allowed=False,
            close_control_allowed=False,
        )
        return pred.astype(np.float32), history_features.astype(np.float32), {"estimate": estimate}
