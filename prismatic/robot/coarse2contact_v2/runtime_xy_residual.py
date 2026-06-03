"""Non-privileged XY residual evidence for runtime precision takeover.

This module is intentionally conservative.  It does not make yaw or close
control decisions, and it does not consume privileged target pose.  Its job is
to separate visual evidence, calibrated proxy validity, and the eventual
contract-aligned XY residual estimate that runtime C2C can use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .basin_state import EstimatedBasinError
from .localizers import LocalGeometryError


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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
        if schema == "c2c_v2_runtime_xy_mlp_calibration_v1" or str(data.get("model_type", "")) == "mlp":
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

    def predict_from_trace(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
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
    source: str = "runtime_xy_mlp_calibration"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeXYMLPCalibration":
        feature_names = tuple(str(x) for x in data.get("feature_names", ()))
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
        std = np.where(np.abs(std) < 1.0e-6, 1.0, std).astype(np.float32)
        return cls(
            feature_names=feature_names,
            layers=tuple(layers),
            feature_mean=mean.astype(np.float32),
            feature_std=std.astype(np.float32),
            source=str(data.get("source", "runtime_xy_mlp_calibration")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "c2c_v2_runtime_xy_mlp_calibration_v1",
            "model_type": "mlp",
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
        }

    def predict_from_trace(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        features = runtime_xy_feature_vector_from_trace(row, self.feature_names)
        x = ((features.astype(np.float32) - self.feature_mean) / self.feature_std).astype(np.float32)
        for i, (weights, bias) in enumerate(self.layers):
            x = weights @ x + bias
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return x[:2].astype(np.float32), features.astype(np.float32)


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
    calibration: RuntimeXYAffineCalibration | RuntimeXYMLPCalibration | None,
) -> GraspFrameResidualEstimate:
    base = estimate_runtime_xy_residual_from_trace(row)
    if calibration is None or not base.entry_ready:
        return base
    pred, _features = calibration.predict_from_trace(row)
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
