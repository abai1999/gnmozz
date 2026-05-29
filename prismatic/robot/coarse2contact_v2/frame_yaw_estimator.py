"""Frame-to-frame yaw residual estimator for C2C v2.

This module is intentionally label-driven: targets come from offline
`frame_residual_v2` privileged relabels, while features are restricted to
runtime-available planner, proxy, and observability signals.  The heuristic
image/PCA axis may appear as a diagnostic input, but it is never treated as the
label or as a calibrated jaw-local residual by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn


VISUAL_CLASSES: tuple[str, ...] = ("prior_only", "partial_observable", "visual_observable")
STAGE_CLASSES: tuple[str, ...] = ("RING_GRASP_ALIGN", "RING_SPOKE_ALIGN", "SLIDE_ON_SPOKE")
SKILL_CLASSES: tuple[str, ...] = ("precision_grasp", "precision_align", "precision_slide")

FRAME_YAW_FEATURE_NAMES: tuple[str, ...] = (
    "planner_local_dx",
    "planner_local_dy",
    "planner_local_dz",
    "planner_local_droll",
    "planner_local_dpitch",
    "planner_local_dyaw",
    "proxy_dx",
    "proxy_dy",
    "proxy_dz",
    "proxy_residual_dyaw",
    "proxy_image_axis_yaw",
    "proxy_confidence",
    "proxy_observability",
    "proxy_fit_residual",
    "proxy_inlier_ratio",
    "proxy_valid",
    "proxy_yaw_valid",
    "estimated_dx",
    "estimated_dy",
    "estimated_dz",
    "estimated_dyaw",
    "estimated_x_valid",
    "estimated_y_valid",
    "estimated_z_valid",
    "estimated_yaw_valid",
    "estimated_confidence",
    "estimated_yaw_confidence",
    "frame_confidence",
    "frame_observability",
    "frame_axis_strength",
    "wide_ring_visible",
    "wrist_occluded",
    "visual_prior_only",
    "visual_partial_observable",
    "visual_visual_observable",
    "stage_ring_grasp_align",
    "stage_ring_spoke_align",
    "stage_slide_on_spoke",
    "skill_precision_grasp",
    "skill_precision_align",
    "skill_precision_slide",
    "requires_yaw_observability",
)


@dataclass(frozen=True)
class FrameYawEstimate:
    dyaw: float
    yaw_observable: bool
    yaw_observable_probability: float
    confidence: float
    source: str = "frame_yaw_estimator"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    return value if isinstance(value, Mapping) else {}


def _vec6(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.float32)
    arr = np.pad(arr, (0, max(0, 6 - arr.size)))[:6]
    arr[~np.isfinite(arr)] = 0.0
    return arr.astype(np.float32)


def _one_hot(value: str, classes: tuple[str, ...]) -> list[float]:
    return [1.0 if str(value) == name else 0.0 for name in classes]


def _obs_value(row: Mapping[str, Any], key: str, default: Any = 0.0) -> Any:
    obs = _mapping(row, "obs_t")
    if key in obs:
        return obs.get(key)
    return row.get(key, default)


def _proxy_geometry(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row, "proxy_local_geometry_error")


def _estimated_geometry(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row, "estimated_basin_error")


def _true_residual(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row, "true_basin_error_t")


def resolve_yaw_observable_threshold(metadata: Mapping[str, Any] | None, *, default: float = 0.5) -> float:
    """Resolve the calibrated yaw-observability working point from checkpoint metadata.

    The preferred source is an explicit top-level working-point field.  For
    backwards compatibility, we also fall back to nested train/val threshold
    sweep results written by the training script.
    """

    candidates: list[Any] = []
    if isinstance(metadata, Mapping):
        for key in (
            "calibrated_yaw_observable_threshold",
            "yaw_observable_threshold_working_point",
            "yaw_observable_threshold",
            "threshold_best_threshold",
        ):
            if key in metadata:
                candidates.append(metadata.get(key))
        for container_key in ("val", "validation", "train"):
            container = metadata.get(container_key)
            if isinstance(container, Mapping):
                for key in (
                    "calibrated_yaw_observable_threshold",
                    "yaw_observable_threshold_working_point",
                    "yaw_observable_threshold",
                    "threshold_best_threshold",
                ):
                    if key in container:
                        candidates.append(container.get(key))
    for candidate in candidates:
        try:
            threshold = float(candidate)
        except Exception:
            continue
        if np.isfinite(threshold):
            return float(threshold)
    return float(default)


def frame_yaw_feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    """Build runtime-available features for frame yaw estimation.

    Do not add privileged residual fields here.  Unit tests rely on this
    function being invariant to changes in `privileged_dyaw`.
    """

    planner = _mapping(row, "planner_prior")
    planner_local = _vec6(planner.get("local_delta_6d", row.get("planner_local_delta_6d", [0.0] * 6)))
    proxy = _proxy_geometry(row)
    estimated = _estimated_geometry(row)
    visual_class = str(_obs_value(row, "visual_observability_class", row.get("visual_observability_class", "")))
    stage_name = str(row.get("stage_name", ""))
    skill_type = str(row.get("skill_type", ""))
    frame_contract = _mapping(row, "frame_contract")

    values: list[float] = []
    values.extend(float(x) for x in planner_local)
    values.extend(
        [
            _safe_float(proxy.get("dx", 0.0)),
            _safe_float(proxy.get("dy", 0.0)),
            _safe_float(proxy.get("dz", 0.0)),
            _safe_float(proxy.get("dyaw", 0.0)),
            _safe_float(proxy.get("image_axis_yaw", 0.0)),
            _safe_float(proxy.get("confidence", 0.0)),
            _safe_float(proxy.get("observability", 0.0)),
            _safe_float(proxy.get("fit_residual", 0.0)),
            _safe_float(proxy.get("inlier_ratio", 0.0)),
            1.0 if _safe_bool(proxy.get("valid", False)) else 0.0,
            1.0 if _safe_bool(proxy.get("yaw_valid", False)) else 0.0,
            _safe_float(estimated.get("dx", 0.0)),
            _safe_float(estimated.get("dy", 0.0)),
            _safe_float(estimated.get("dz", 0.0)),
            _safe_float(estimated.get("dyaw", 0.0)),
            1.0 if _safe_bool(estimated.get("x_valid", _mapping(estimated, "axis_validity").get("x", False))) else 0.0,
            1.0 if _safe_bool(estimated.get("y_valid", _mapping(estimated, "axis_validity").get("y", False))) else 0.0,
            1.0 if _safe_bool(estimated.get("z_valid", _mapping(estimated, "axis_validity").get("z", False))) else 0.0,
            1.0 if _safe_bool(estimated.get("yaw_valid", _mapping(estimated, "axis_validity").get("yaw", False))) else 0.0,
            _safe_float(estimated.get("confidence", 0.0)),
            _safe_float(estimated.get("yaw_confidence", _mapping(estimated, "axis_confidence").get("yaw", 0.0))),
            _safe_float(_obs_value(row, "frame_confidence", row.get("source_frame_confidence", 0.0))),
            _safe_float(_obs_value(row, "frame_observability", row.get("source_frame_observability", 0.0))),
            _safe_float(_obs_value(row, "frame_axis_strength", row.get("source_frame_axis_strength", 0.0))),
            1.0 if _safe_bool(_obs_value(row, "wide_ring_visible", row.get("wide_ring_visible", False))) else 0.0,
            1.0 if _safe_bool(row.get("yaw_observability_wrist_occluded", row.get("wrist_is_occluded", False))) else 0.0,
        ]
    )
    values.extend(_one_hot(visual_class, VISUAL_CLASSES))
    values.extend(_one_hot(stage_name, STAGE_CLASSES))
    values.extend(_one_hot(skill_type, SKILL_CLASSES))
    values.append(1.0 if _safe_bool(frame_contract.get("requires_yaw_observability", row.get("requires_yaw_observability", False))) else 0.0)
    out = np.asarray(values, dtype=np.float32)
    if out.shape[0] != len(FRAME_YAW_FEATURE_NAMES):
        raise RuntimeError(f"frame yaw feature length mismatch: {out.shape[0]} != {len(FRAME_YAW_FEATURE_NAMES)}")
    out[~np.isfinite(out)] = 0.0
    return out


def frame_yaw_label_from_row(row: Mapping[str, Any]) -> tuple[float, float, float]:
    residual = _true_residual(row)
    dyaw = _safe_float(residual.get("dyaw", row.get("privileged_dyaw", float("nan"))), float("nan"))
    yaw_observable = 1.0 if _safe_bool(row.get("yaw_control_observable", row.get("yaw_observable", False))) else 0.0
    valid = 1.0 if np.isfinite(dyaw) and bool(row.get("label_valid", True)) else 0.0
    return float(dyaw), float(yaw_observable), float(valid)


class FrameYawEstimatorNet(nn.Module):
    """MLP that predicts jaw-local frame residual yaw and observability."""

    def __init__(self, *, feature_dim: int = len(FRAME_YAW_FEATURE_NAMES), hidden_dim: int = 96, max_abs_yaw: float = float(np.pi / 2.0)) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.max_abs_yaw = float(max_abs_yaw)
        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.dyaw_head = nn.Linear(hidden_dim, 1)
        self.yaw_observable_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(features.float())
        dyaw = torch.tanh(self.dyaw_head(h)[:, 0]) * self.max_abs_yaw
        yaw_logit = self.yaw_observable_head(h)[:, 0]
        conf_logit = self.confidence_head(h)[:, 0]
        return {
            "dyaw": dyaw,
            "yaw_observable_logit": yaw_logit,
            "yaw_observable_probability": torch.sigmoid(yaw_logit),
            "confidence_logit": conf_logit,
            "confidence": torch.sigmoid(conf_logit),
        }

    def predict_numpy(self, features: np.ndarray, *, yaw_observable_threshold: float = 0.5) -> FrameYawEstimate:
        self.eval()
        feat = torch.as_tensor(np.asarray(features, dtype=np.float32).reshape(1, -1))
        with torch.no_grad():
            out = self.forward(feat)
        prob = float(out["yaw_observable_probability"][0].item())
        return FrameYawEstimate(
            dyaw=float(out["dyaw"][0].item()),
            yaw_observable=bool(prob >= float(yaw_observable_threshold)),
            yaw_observable_probability=prob,
            confidence=float(out["confidence"][0].item()),
        )


def save_frame_yaw_checkpoint(path: str | Path, model: FrameYawEstimatorNet, *, metadata: Mapping[str, Any] | None = None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_names": list(FRAME_YAW_FEATURE_NAMES),
        "feature_dim": int(model.feature_dim),
        "max_abs_yaw": float(model.max_abs_yaw),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, str(path))


def load_frame_yaw_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[FrameYawEstimatorNet, dict[str, Any]]:
    payload = torch.load(str(path), map_location=map_location)
    model = FrameYawEstimatorNet(
        feature_dim=int(payload.get("feature_dim", len(FRAME_YAW_FEATURE_NAMES))),
        max_abs_yaw=float(payload.get("max_abs_yaw", np.pi / 2.0)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, dict(payload.get("metadata", {}))
