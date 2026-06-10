"""Unified spatial-temporal task-frame alignment candidate for C2C v2.

The v46 candidate is a non-privileged state estimator/controller scaffold. It
predicts task-frame residuals and per-axis confidence/observability for XY, Z,
and Yaw in parallel. It intentionally does not own gripper close authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .learned_localizer import _ImageEncoder
from .runtime_xy_residual import (
    DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    _load_spatial_temporal_rgbd,
    runtime_xy_spatial_temporal_context_feature_vector_from_trace,
    runtime_xy_spatial_temporal_feature_names,
)
from .task_frame_readiness import TASK_FRAME_READINESS_FEATURE_NAMES, task_frame_readiness_feature_vector


TASK_FRAME_V46_RISK_CLASSES: tuple[str, ...] = (
    "normal",
    "low_visibility",
    "direction_conflict",
    "insufficient_support",
    "occlusion",
    "force_guard",
    "alias_drift",
)

TASK_FRAME_V46_DEFAULT_MAX_XY_STEP = 0.0030
TASK_FRAME_V46_DEFAULT_MAX_Z_STEP = 0.0030
TASK_FRAME_V46_DEFAULT_MAX_YAW_STEP = 0.020
TASK_FRAME_V46_DEFAULT_XY_GAIN = 0.35
TASK_FRAME_V46_DEFAULT_Z_GAIN = 0.35
TASK_FRAME_V46_DEFAULT_YAW_GAIN = 0.25
TASK_FRAME_V46_DEFAULT_CONFIDENCE_THRESHOLD = 0.45
TASK_FRAME_V46_WEAK_CONFIDENCE_THRESHOLD = 0.20
TASK_FRAME_V46_DEFAULT_MIN_STEP_SCALE = 0.05
TASK_FRAME_V46_LOW_VIS_MIN_STEP_SCALE = 0.0015
TASK_FRAME_V46_YAW_HISTORY_SUPPORT = 2
TASK_FRAME_V46_XY_RISK_STEP_SCALE: dict[str, float] = {
    "low_visibility": 0.25,
    "direction_conflict": 0.35,
    "insufficient_support": 0.35,
    "occlusion": 0.25,
    "alias_drift": 0.35,
}
TASK_FRAME_V46_Z_RISK_STEP_SCALE: dict[str, float] = {
    "low_visibility": 0.50,
    "direction_conflict": 0.70,
    "insufficient_support": 0.70,
    "occlusion": 0.50,
    "alias_drift": 0.70,
}

TASK_FRAME_V46_SPATIAL_MOMENT_DIM = 26

TASK_FRAME_V46_YAW_SELECTOR_BASE_FEATURE_NAMES: tuple[str, ...] = (
    "v46_yaw_observable_score",
    "v46_yaw_confidence",
    "v46_yaw_ambiguous_score",
    "v46_yaw_step_scale",
    "v46_pred_dyaw",
    "v46_abs_pred_dyaw",
    "v46_yaw_hypothesis_gap",
    "v46_near_field_confidence",
    "v46_xy_observable_score",
    "v46_z_observable_score",
    "v46_xy_confidence",
    "v46_z_confidence",
    "v46_xy_step_scale",
    "v46_z_step_scale",
)


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
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    return bool(default)


def _coerce_vector(value: Any, *, length: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        arr = np.asarray([], dtype=np.float32)
    else:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.asarray([], dtype=np.float32)
    if arr.size < int(length):
        arr = np.concatenate(
            [arr.astype(np.float32), np.full((int(length) - int(arr.size),), float(default), dtype=np.float32)],
            axis=0,
        )
    return arr[: int(length)].astype(np.float32)


def _safe_sign(value: float, *, eps: float = 1.0e-6) -> int:
    if not np.isfinite(float(value)) or abs(float(value)) <= float(eps):
        return 0
    return 1 if float(value) > 0.0 else -1


def _history_direction_stable(
    history_rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None,
    *,
    field: str,
    current_value: float,
    required_support: int = TASK_FRAME_V46_YAW_HISTORY_SUPPORT,
) -> bool:
    recent: list[float] = []
    for row in reversed(list(history_rows or ())):
        value = row.get(field, row.get(f"task_frame_v46_{field}"))
        value_f = _as_float(value, float("nan"))
        if not np.isfinite(value_f) or _safe_sign(value_f) == 0:
            continue
        recent.append(value_f)
        if len(recent) >= int(required_support):
            break
    if len(recent) < int(required_support):
        return False
    sign = _safe_sign(current_value)
    return bool(sign != 0 and all(_safe_sign(value) == sign for value in recent))


def task_frame_v46_scalar_feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    return task_frame_readiness_feature_vector(row)


def _weighted_moments(image_rgbd: torch.Tensor, weight: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
    eps = torch.as_tensor(1.0e-6, dtype=image_rgbd.dtype, device=image_rgbd.device)
    denom = torch.clamp(weight.sum(dim=(-2, -1)), min=float(eps.item()))
    mx = (weight * grid_x).sum(dim=(-2, -1)) / denom
    my = (weight * grid_y).sum(dim=(-2, -1)) / denom
    vx = (weight * (grid_x - mx[:, None, None]) ** 2).sum(dim=(-2, -1)) / denom
    vy = (weight * (grid_y - my[:, None, None]) ** 2).sum(dim=(-2, -1)) / denom
    cov = (weight * (grid_x - mx[:, None, None]) * (grid_y - my[:, None, None])).sum(dim=(-2, -1)) / denom
    mass = weight.mean(dim=(-2, -1))
    return torch.stack([mx, my, vx, vy, cov, mass], dim=-1)


def task_frame_v46_spatial_moment_features(image_rgbd: torch.Tensor) -> torch.Tensor:
    """Non-privileged spatial geometry moments from RGBD/depth-valid input."""

    x = image_rgbd.float()
    if x.ndim != 4:
        raise ValueError(f"expected BCHW image tensor, got shape {tuple(x.shape)}")
    b, c, h, w = x.shape
    depth = x[:, 3] if c > 3 else torch.zeros((b, h, w), dtype=x.dtype, device=x.device)
    valid = x[:, 4] if c > 4 else torch.ones((b, h, w), dtype=x.dtype, device=x.device)
    if c > 6:
        grid_x = x[:, 5]
        grid_y = x[:, 6]
    else:
        xs = torch.linspace(-1.0, 1.0, steps=w, dtype=x.dtype, device=x.device)
        ys = torch.linspace(-1.0, 1.0, steps=h, dtype=x.dtype, device=x.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)
    valid = torch.clamp(valid, 0.0, 1.0)
    depth = torch.clamp(depth, 0.0, 1.0)
    rgb = x[:, :3] if c >= 3 else torch.zeros((b, 3, h, w), dtype=x.dtype, device=x.device)
    valid_denom = torch.clamp(valid.sum(dim=(-2, -1)), min=1.0e-6)
    depth_mean = (valid * depth).sum(dim=(-2, -1)) / valid_denom
    depth_var = (valid * (depth - depth_mean[:, None, None]) ** 2).sum(dim=(-2, -1)) / valid_denom
    rgb_mean = (rgb * valid[:, None]).sum(dim=(-2, -1)) / valid_denom[:, None]
    near_weight = valid * torch.clamp(depth_mean[:, None, None] - depth, min=0.0)
    far_weight = valid * torch.clamp(depth - depth_mean[:, None, None], min=0.0)
    valid_moments = _weighted_moments(x, valid, grid_x, grid_y)
    near_moments = _weighted_moments(x, near_weight, grid_x, grid_y)
    far_moments = _weighted_moments(x, far_weight, grid_x, grid_y)
    extras = torch.stack(
        [
            valid.mean(dim=(-2, -1)),
            depth_mean,
            torch.sqrt(torch.clamp(depth_var, min=0.0)),
            near_weight.mean(dim=(-2, -1)),
            far_weight.mean(dim=(-2, -1)),
        ],
        dim=-1,
    )
    return torch.cat([valid_moments, near_moments, far_moments, extras, rgb_mean], dim=-1)


@dataclass(frozen=True)
class TaskFrameV46AlignmentEstimate:
    dx: float
    dy: float
    dz: float
    dyaw: float
    xy_confidence: float
    z_confidence: float
    yaw_confidence: float
    xy_observable: bool
    z_observable: bool
    yaw_observable: bool
    yaw_ambiguous: bool
    yaw_unobservable: bool
    xy_step_scale: float
    z_step_scale: float
    yaw_step_scale: float
    yaw_hypothesis_index: int
    yaw_hypothesis_gap: float
    xy_control_effect: tuple[float, float, float, float]
    risk_reason: str
    near_field_confidence: float = 0.0
    near_field_head_available: bool = True
    xy_observable_score: float = 0.0
    z_observable_score: float = 0.0
    yaw_observable_score: float = 0.0
    yaw_ambiguous_score: float = 0.0
    source: str = "v46_unified_task_frame_alignment_candidate"
    uses_privileged_runtime: bool = False
    close_control_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_frame_v46_dx": float(self.dx),
            "task_frame_v46_dy": float(self.dy),
            "task_frame_v46_dz": float(self.dz),
            "task_frame_v46_dyaw": float(self.dyaw),
            "task_frame_v46_xy_confidence": float(self.xy_confidence),
            "task_frame_v46_z_confidence": float(self.z_confidence),
            "task_frame_v46_yaw_confidence": float(self.yaw_confidence),
            "task_frame_v46_xy_observable": bool(self.xy_observable),
            "task_frame_v46_z_observable": bool(self.z_observable),
            "task_frame_v46_yaw_observable": bool(self.yaw_observable),
            "task_frame_v46_yaw_ambiguous": bool(self.yaw_ambiguous),
            "task_frame_v46_yaw_unobservable": bool(self.yaw_unobservable),
            "task_frame_v46_xy_step_scale": float(self.xy_step_scale),
            "task_frame_v46_z_step_scale": float(self.z_step_scale),
            "task_frame_v46_yaw_step_scale": float(self.yaw_step_scale),
            "task_frame_v46_yaw_hypothesis_index": int(self.yaw_hypothesis_index),
            "task_frame_v46_yaw_hypothesis_gap": float(self.yaw_hypothesis_gap),
            "task_frame_v46_xy_control_effect": [float(v) for v in self.xy_control_effect],
            "task_frame_v46_near_field_confidence": float(self.near_field_confidence),
            "task_frame_v46_near_field_head_available": bool(self.near_field_head_available),
            "task_frame_v46_xy_observable_score": float(self.xy_observable_score),
            "task_frame_v46_z_observable_score": float(self.z_observable_score),
            "task_frame_v46_yaw_observable_score": float(self.yaw_observable_score),
            "task_frame_v46_yaw_ambiguous_score": float(self.yaw_ambiguous_score),
            "task_frame_v46_risk_reason": str(self.risk_reason),
            "task_frame_v46_source": str(self.source),
            "task_frame_v46_uses_privileged_runtime": bool(self.uses_privileged_runtime),
            "task_frame_v46_close_control_allowed": bool(self.close_control_allowed),
        }


@dataclass(frozen=True)
class TaskFrameV46ControlDecision:
    applied: bool
    xy_allowed: bool
    z_allowed: bool
    yaw_allowed: bool
    xy_weak: bool
    z_weak: bool
    yaw_history_stable: bool
    force_safe: bool
    xy_block_reason: str
    z_block_reason: str
    yaw_block_reason: str
    block_reason: str
    dx_step: float
    dy_step: float
    dz_step: float
    dyaw_step: float
    xy_risk_step_scale: float
    z_risk_step_scale: float
    source: str = "v46_unified_task_frame_alignment_candidate"
    uses_privileged_runtime: bool = False
    close_control_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_frame_v46_applied": bool(self.applied),
            "task_frame_v46_xy_allowed": bool(self.xy_allowed),
            "task_frame_v46_z_allowed": bool(self.z_allowed),
            "task_frame_v46_yaw_allowed": bool(self.yaw_allowed),
            "task_frame_v46_xy_weak": bool(self.xy_weak),
            "task_frame_v46_z_weak": bool(self.z_weak),
            "task_frame_v46_yaw_history_stable": bool(self.yaw_history_stable),
            "task_frame_v46_force_safe": bool(self.force_safe),
            "task_frame_v46_xy_block_reason": str(self.xy_block_reason),
            "task_frame_v46_z_block_reason": str(self.z_block_reason),
            "task_frame_v46_yaw_block_reason": str(self.yaw_block_reason),
            "task_frame_v46_block_reason": str(self.block_reason),
            "task_frame_v46_dx_step": float(self.dx_step),
            "task_frame_v46_dy_step": float(self.dy_step),
            "task_frame_v46_dz_step": float(self.dz_step),
            "task_frame_v46_dyaw_step": float(self.dyaw_step),
            "task_frame_v46_xy_risk_step_scale": float(self.xy_risk_step_scale),
            "task_frame_v46_z_risk_step_scale": float(self.z_risk_step_scale),
            "task_frame_v46_source": str(self.source),
            "task_frame_v46_uses_privileged_runtime": bool(self.uses_privileged_runtime),
            "task_frame_v46_close_control_allowed": bool(self.close_control_allowed),
        }


@dataclass(frozen=True)
class TaskFrameV46YawControlSelectorDecision:
    allowed: bool
    score: float
    threshold: float
    block_reason: str
    source: str = "v46_yaw_control_permission_selector"
    uses_privileged_runtime: bool = False
    close_control_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_frame_v46_yaw_selector_loaded": True,
            "task_frame_v46_yaw_selector_allowed": bool(self.allowed),
            "task_frame_v46_yaw_selector_score": float(self.score),
            "task_frame_v46_yaw_selector_threshold": float(self.threshold),
            "task_frame_v46_yaw_selector_block_reason": str(self.block_reason),
            "task_frame_v46_yaw_selector_source": str(self.source),
            "task_frame_v46_yaw_selector_uses_privileged_runtime": bool(self.uses_privileged_runtime),
            "task_frame_v46_yaw_selector_close_control_allowed": bool(self.close_control_allowed),
        }


@dataclass(frozen=True)
class TaskFrameV46CommandSearchResult:
    selected_step_local_6d: tuple[float, float, float, float, float, float]
    selected_command_local_6d: tuple[float, float, float, float, float, float]
    selected_delta: tuple[float, float, float, float]
    selected_logvar: tuple[float, float, float, float]
    selected_support: float
    selected_score: float
    no_correction_score: float
    pre_score: float
    xy_contracts: bool
    z_contracts: bool
    yaw_contracts: bool
    applied: bool
    valid: bool
    reason: str
    candidate_count: int
    uses_privileged_runtime: bool = False
    close_control_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        selected_step = np.asarray(self.selected_step_local_6d, dtype=np.float32)
        return {
            "task_frame_v46_command_search_valid": bool(self.valid),
            "task_frame_v46_command_search_reason": str(self.reason),
            "task_frame_v46_command_search_applied": bool(self.applied),
            "task_frame_v46_command_search_candidate_count": int(self.candidate_count),
            "task_frame_v46_command_search_selected_step_local_6d": [float(v) for v in selected_step.tolist()],
            "task_frame_v46_command_search_selected_command_local_6d": [float(v) for v in self.selected_command_local_6d],
            "task_frame_v46_command_transition_delta": [float(v) for v in self.selected_delta],
            "task_frame_v46_command_transition_logvar": [float(v) for v in self.selected_logvar],
            "task_frame_v46_command_transition_support": float(self.selected_support),
            "task_frame_v46_command_transition_score": float(self.selected_score),
            "task_frame_v46_command_transition_no_correction_score": float(self.no_correction_score),
            "task_frame_v46_command_transition_pre_score": float(self.pre_score),
            "task_frame_v46_command_transition_xy_contracts": bool(self.xy_contracts),
            "task_frame_v46_command_transition_z_contracts": bool(self.z_contracts),
            "task_frame_v46_command_transition_yaw_contracts": bool(self.yaw_contracts),
            "task_frame_v46_transition_guard_suppressed_xy": bool(abs(float(selected_step[0])) <= 1.0e-9 and abs(float(selected_step[1])) <= 1.0e-9),
            "task_frame_v46_transition_guard_suppressed_z": bool(abs(float(selected_step[2])) <= 1.0e-9),
            "task_frame_v46_transition_guard_suppressed_yaw": bool(abs(float(selected_step[5])) <= 1.0e-9),
            "task_frame_v46_uses_privileged_runtime": bool(self.uses_privileged_runtime),
            "task_frame_v46_close_control_allowed": bool(self.close_control_allowed),
        }


class TaskFrameV46AlignmentNet(nn.Module):
    """Spatial-temporal estimator for parallel XY/Z/Yaw task-frame state."""

    def __init__(
        self,
        *,
        image_in_channels: int = 7,
        image_hidden_dim: int = 128,
        scalar_feature_dim: int = len(TASK_FRAME_READINESS_FEATURE_NAMES),
        history_feature_dim: int | None = None,
        history_window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
        proprio_dim: int = 15,
        planner_prior_dim: int = 6,
        command_feature_dim: int = 6,
        fusion_hidden_dim: int = 128,
        yaw_hypotheses: int = 4,
        risk_classes: tuple[str, ...] = TASK_FRAME_V46_RISK_CLASSES,
        max_abs_xy: float = 0.040,
        max_abs_z: float = 0.030,
        max_abs_yaw: float = 0.250,
        use_spatial_moments: bool = False,
    ) -> None:
        super().__init__()
        self.image_in_channels = int(image_in_channels)
        self.image_hidden_dim = int(image_hidden_dim)
        self.scalar_feature_dim = int(scalar_feature_dim)
        self.history_window_size = int(history_window_size)
        default_history_names = runtime_xy_spatial_temporal_feature_names(tuple(DEFAULT_RUNTIME_XY_FEATURE_NAMES), self.history_window_size)
        self.history_feature_dim = int(history_feature_dim if history_feature_dim is not None else max(1, len(default_history_names) // max(1, self.history_window_size)))
        self.proprio_dim = int(proprio_dim)
        self.planner_prior_dim = int(planner_prior_dim)
        self.command_feature_dim = int(max(1, command_feature_dim))
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.yaw_hypotheses = int(max(1, yaw_hypotheses))
        self.risk_classes = tuple(risk_classes)
        self.max_abs_xy = float(max_abs_xy)
        self.max_abs_z = float(max_abs_z)
        self.max_abs_yaw = float(max_abs_yaw)
        self.use_spatial_moments = bool(use_spatial_moments)
        self.image_encoder = _ImageEncoder(in_channels=self.image_in_channels, hidden_dim=self.image_hidden_dim)
        self.spatial_moment_mlp = nn.Sequential(
            nn.Linear(TASK_FRAME_V46_SPATIAL_MOMENT_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.scalar_mlp = nn.Sequential(nn.Linear(max(1, self.scalar_feature_dim), 64), nn.ReLU(inplace=True), nn.Linear(64, 48), nn.ReLU(inplace=True))
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
        spatial_dim = 32 if self.use_spatial_moments else 0
        self.fusion = nn.Sequential(nn.Linear(self.image_hidden_dim + spatial_dim + 48 + 64 + 32, 192), nn.ReLU(inplace=True), nn.Linear(192, self.fusion_hidden_dim), nn.ReLU(inplace=True))
        self.residual_head = nn.Linear(self.fusion_hidden_dim, 4)
        self.confidence_head = nn.Linear(self.fusion_hidden_dim, 3)
        self.observability_head = nn.Linear(self.fusion_hidden_dim, 3)
        self.yaw_ambiguous_head = nn.Linear(self.fusion_hidden_dim, 1)
        self.step_scale_head = nn.Linear(self.fusion_hidden_dim, 3)
        self.yaw_hypothesis_head = nn.Linear(self.fusion_hidden_dim, self.yaw_hypotheses)
        self.risk_head = nn.Linear(self.fusion_hidden_dim, max(1, len(self.risk_classes)))
        self.xy_control_effect_head = nn.Linear(self.fusion_hidden_dim, 4)
        self.near_field_head = nn.Linear(self.fusion_hidden_dim, 1)
        self.command_transition_mlp = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim + self.command_feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.command_delta_head = nn.Linear(64, 4)
        self.command_logvar_head = nn.Linear(64, 4)
        self.command_support_head = nn.Linear(64, 1)
        self.command_outcome_head = nn.Linear(64, 9)
        self.near_field_head_available = True

    def forward(
        self,
        image_rgbd: torch.Tensor,
        scalar_features: torch.Tensor,
        history_features: torch.Tensor,
        proprio: torch.Tensor,
        planner_prior: torch.Tensor,
        command_6d: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image = image_rgbd.float()
        image_feat = self.image_encoder(image, pooled=True)
        if self.use_spatial_moments:
            spatial_feat = self.spatial_moment_mlp(task_frame_v46_spatial_moment_features(image))
        else:
            spatial_feat = None
        scalar = scalar_features.float()
        if scalar.ndim == 1:
            scalar = scalar.unsqueeze(0)
        if scalar.shape[-1] < self.scalar_feature_dim:
            scalar = F.pad(scalar, (0, self.scalar_feature_dim - scalar.shape[-1]))
        scalar_feat = self.scalar_mlp(scalar[..., : self.scalar_feature_dim])
        history = history_features.float().reshape(history_features.shape[0], -1)
        history_dim = self.history_feature_dim * self.history_window_size
        if history.shape[-1] < history_dim:
            history = F.pad(history, (0, history_dim - history.shape[-1]))
        history_feat = self.history_mlp(history[..., :history_dim])
        prior = torch.cat([proprio.float(), planner_prior.float()], dim=-1)
        prior_dim = self.proprio_dim + self.planner_prior_dim
        if prior.shape[-1] < prior_dim:
            prior = F.pad(prior, (0, prior_dim - prior.shape[-1]))
        prior_feat = self.proprio_prior_mlp(prior[..., :prior_dim])
        fusion_parts = [image_feat]
        if spatial_feat is not None:
            fusion_parts.append(spatial_feat)
        fusion_parts.extend([scalar_feat, history_feat, prior_feat])
        fused = self.fusion(torch.cat(fusion_parts, dim=-1))
        if command_6d is None:
            command = torch.zeros((fused.shape[0], self.command_feature_dim), dtype=fused.dtype, device=fused.device)
        else:
            command = command_6d.float()
            if command.ndim == 1:
                command = command.unsqueeze(0)
            if command.shape[-1] < self.command_feature_dim:
                command = F.pad(command, (0, self.command_feature_dim - command.shape[-1]))
            command = command[..., : self.command_feature_dim]
        command_feat = self.command_transition_mlp(torch.cat([fused, command], dim=-1))
        raw = torch.tanh(self.residual_head(fused))
        return {
            "dx": raw[:, 0] * self.max_abs_xy,
            "dy": raw[:, 1] * self.max_abs_xy,
            "dz": raw[:, 2] * self.max_abs_z,
            "dyaw": raw[:, 3] * self.max_abs_yaw,
            "axis_confidence": torch.sigmoid(self.confidence_head(fused)),
            "axis_observability": torch.sigmoid(self.observability_head(fused)),
            "yaw_ambiguous": torch.sigmoid(self.yaw_ambiguous_head(fused)[:, 0]),
            "axis_step_scale": torch.sigmoid(self.step_scale_head(fused)),
            "yaw_hypothesis_logits": self.yaw_hypothesis_head(fused),
            "risk_logits": self.risk_head(fused),
            "xy_control_effect": 2.5 * torch.tanh(self.xy_control_effect_head(fused)).reshape(-1, 2, 2),
            "near_field_confidence": torch.sigmoid(self.near_field_head(fused)[:, 0]),
            "command_delta": self.command_delta_head(command_feat),
            "command_logvar": torch.clamp(self.command_logvar_head(command_feat), min=-8.0, max=4.0),
            "command_support": torch.sigmoid(self.command_support_head(command_feat)[:, 0]),
            "command_outcome_logits": self.command_outcome_head(command_feat),
        }

    def predict_numpy(
        self,
        image_rgbd: torch.Tensor,
        scalar_features: np.ndarray,
        history_features: np.ndarray,
        proprio: np.ndarray,
        planner_prior: np.ndarray,
        command_6d: np.ndarray | None = None,
        *,
        source: str = "v46_unified_task_frame_alignment_candidate",
    ) -> TaskFrameV46AlignmentEstimate:
        self.eval()
        with torch.no_grad():
            out = self.forward(
                image_rgbd.unsqueeze(0) if image_rgbd.ndim == 3 else image_rgbd,
                torch.as_tensor(np.asarray(scalar_features, dtype=np.float32).reshape(1, -1)),
                torch.as_tensor(np.asarray(history_features, dtype=np.float32).reshape(1, -1)),
                torch.as_tensor(np.asarray(proprio, dtype=np.float32).reshape(1, -1)),
                torch.as_tensor(np.asarray(planner_prior, dtype=np.float32).reshape(1, -1)),
                torch.as_tensor(np.asarray(command_6d if command_6d is not None else np.zeros((6,), dtype=np.float32), dtype=np.float32).reshape(1, -1)),
            )
        axis_conf = out["axis_confidence"][0].detach().cpu().numpy().astype(np.float32)
        axis_obs = out["axis_observability"][0].detach().cpu().numpy().astype(np.float32)
        step_scale = out["axis_step_scale"][0].detach().cpu().numpy().astype(np.float32)
        yaw_probs = torch.softmax(out["yaw_hypothesis_logits"][0], dim=-1).detach().cpu().numpy().astype(np.float32)
        order = np.argsort(-yaw_probs)
        best = int(order[0]) if order.size else 0
        gap = float(yaw_probs[order[0]] - yaw_probs[order[1]]) if order.size >= 2 else 1.0
        risk_logits = out["risk_logits"][0].detach().cpu().numpy().astype(np.float32)
        xy_control_effect = out["xy_control_effect"][0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        near_field_confidence = float(out["near_field_confidence"][0].item())
        risk_idx = int(np.argmax(risk_logits)) if risk_logits.size else 0
        risk_reason = self.risk_classes[min(max(risk_idx, 0), len(self.risk_classes) - 1)]
        yaw_ambiguous_prob = float(out["yaw_ambiguous"][0].item())
        yaw_observable = bool(axis_obs[2] >= 0.5)
        return TaskFrameV46AlignmentEstimate(
            dx=float(out["dx"][0].item()),
            dy=float(out["dy"][0].item()),
            dz=float(out["dz"][0].item()),
            dyaw=float(out["dyaw"][0].item()),
            xy_confidence=float(axis_conf[0]),
            z_confidence=float(axis_conf[1]),
            yaw_confidence=float(axis_conf[2]),
            xy_observable=bool(axis_obs[0] >= 0.5),
            z_observable=bool(axis_obs[1] >= 0.5),
            yaw_observable=yaw_observable,
            yaw_ambiguous=bool(yaw_ambiguous_prob >= 0.5 or gap < 0.15),
            yaw_unobservable=bool(not yaw_observable),
            xy_step_scale=float(np.clip(step_scale[0], 0.0, 1.0)),
            z_step_scale=float(np.clip(step_scale[1], 0.0, 1.0)),
            yaw_step_scale=float(np.clip(step_scale[2], 0.0, 1.0)),
            yaw_hypothesis_index=best,
            yaw_hypothesis_gap=gap,
            xy_control_effect=tuple(float(v) for v in xy_control_effect[:4]),
            risk_reason=str(risk_reason),
            near_field_confidence=float(np.clip(near_field_confidence, 0.0, 1.0)),
            near_field_head_available=bool(getattr(self, "near_field_head_available", True)),
            xy_observable_score=float(np.clip(axis_obs[0], 0.0, 1.0)),
            z_observable_score=float(np.clip(axis_obs[1], 0.0, 1.0)),
            yaw_observable_score=float(np.clip(axis_obs[2], 0.0, 1.0)),
            yaw_ambiguous_score=float(np.clip(yaw_ambiguous_prob, 0.0, 1.0)),
            source=str(source),
        )


class TaskFrameV46YawControlSelectorNet(nn.Module):
    """Small permission-only yaw selector used to suppress unsafe yaw servo."""

    def __init__(self, feature_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).reshape(-1)


def task_frame_v46_yaw_selector_feature_names(
    *,
    include_scalar_features: bool = True,
    include_spatial_moment_features: bool = True,
) -> tuple[str, ...]:
    names = list(TASK_FRAME_V46_YAW_SELECTOR_BASE_FEATURE_NAMES)
    if bool(include_scalar_features):
        names.extend(f"scalar::{name}" for name in TASK_FRAME_READINESS_FEATURE_NAMES)
    if bool(include_spatial_moment_features):
        names.extend(f"spatial_moment_{idx:02d}" for idx in range(TASK_FRAME_V46_SPATIAL_MOMENT_DIM))
    return tuple(names)


def _task_frame_v46_yaw_selector_base_features(estimate: TaskFrameV46AlignmentEstimate) -> np.ndarray:
    return np.asarray(
        [
            float(estimate.yaw_observable_score),
            float(estimate.yaw_confidence),
            float(estimate.yaw_ambiguous_score),
            float(estimate.yaw_step_scale),
            float(estimate.dyaw),
            abs(float(estimate.dyaw)),
            float(estimate.yaw_hypothesis_gap),
            float(estimate.near_field_confidence),
            float(estimate.xy_observable_score),
            float(estimate.z_observable_score),
            float(estimate.xy_confidence),
            float(estimate.z_confidence),
            float(estimate.xy_step_scale),
            float(estimate.z_step_scale),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class TaskFrameV46YawControlSelectorCalibration:
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float
    hidden_dim: int
    include_scalar_features: bool
    include_spatial_moment_features: bool
    model_state_dict: dict[str, torch.Tensor]
    source: str = "v46_yaw_control_permission_selector"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskFrameV46YawControlSelectorCalibration":
        metadata = dict(payload.get("metadata", {}))
        mean = np.asarray(payload.get("feature_mean", []), dtype=np.float32).reshape(-1)
        std = np.asarray(payload.get("feature_std", []), dtype=np.float32).reshape(-1)
        if mean.size == 0 or std.size == 0 or mean.size != std.size:
            raise ValueError("yaw selector checkpoint must contain matching feature_mean/feature_std")
        include_scalar = bool(metadata.get("include_scalar_features", True))
        include_spatial = bool(metadata.get("include_spatial_moment_features", False))
        feature_names = tuple(str(x) for x in metadata.get("feature_names", payload.get("feature_names", ())))
        if len(feature_names) != int(mean.size):
            feature_names = task_frame_v46_yaw_selector_feature_names(
                include_scalar_features=include_scalar,
                include_spatial_moment_features=include_spatial,
            )
        if len(feature_names) != int(mean.size):
            feature_names = tuple(f"feature_{idx:03d}" for idx in range(int(mean.size)))
        state_dict = {str(k): torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in dict(payload.get("model_state_dict", {})).items()}
        hidden_dim = int(payload.get("hidden_dim", metadata.get("hidden_dim", 24)))
        model = TaskFrameV46YawControlSelectorNet(feature_dim=int(mean.size), hidden_dim=hidden_dim)
        model.load_state_dict(state_dict, strict=True)
        return cls(
            feature_names=tuple(feature_names),
            feature_mean=mean.astype(np.float32),
            feature_std=np.where(std.astype(np.float32) < 1.0e-6, 1.0, std.astype(np.float32)).astype(np.float32),
            threshold=float(payload.get("selected_threshold", metadata.get("selected_threshold", 0.5))),
            hidden_dim=hidden_dim,
            include_scalar_features=include_scalar,
            include_spatial_moment_features=include_spatial,
            model_state_dict=state_dict,
            source=str(payload.get("model_type", "v46_yaw_control_permission_selector")),
        ).with_model(model)

    @classmethod
    def load(cls, path: str | Path | None) -> "TaskFrameV46YawControlSelectorCalibration | None":
        if path is None or not str(path):
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"missing task-frame v46 yaw selector checkpoint: {p}")
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"task-frame v46 yaw selector checkpoint must be a mapping: {p}")
        return cls.from_dict(payload)

    def with_model(self, model: TaskFrameV46YawControlSelectorNet) -> "TaskFrameV46YawControlSelectorCalibration":
        object.__setattr__(self, "_model", model)
        return self

    @property
    def model(self) -> TaskFrameV46YawControlSelectorNet:
        model = getattr(self, "_model", None)
        if model is None:
            model = TaskFrameV46YawControlSelectorNet(feature_dim=int(self.feature_mean.size), hidden_dim=int(self.hidden_dim))
            model.load_state_dict(self.model_state_dict, strict=True)
            object.__setattr__(self, "_model", model)
        return model

    def feature_vector_from_trace(
        self,
        row: Mapping[str, Any],
        estimate: TaskFrameV46AlignmentEstimate,
        *,
        observation: Mapping[str, Any] | None = None,
        robot_state: Mapping[str, Any] | None = None,
        image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
        image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    ) -> tuple[np.ndarray | None, str]:
        parts = [_task_frame_v46_yaw_selector_base_features(estimate)]
        if bool(self.include_scalar_features):
            parts.append(task_frame_v46_scalar_feature_vector(row))
        if bool(self.include_spatial_moment_features):
            if observation is None:
                return None, "missing_observation"
            rgbd = _load_spatial_temporal_rgbd(
                observation,
                robot_state,
                crop_size=int(image_crop_size),
                resize_size=int(image_resize_size),
            )
            if rgbd is None:
                return None, "missing_rgbd"
            with torch.no_grad():
                spatial = task_frame_v46_spatial_moment_features(rgbd.unsqueeze(0) if rgbd.ndim == 3 else rgbd)
            parts.append(spatial.detach().cpu().numpy().reshape(-1).astype(np.float32))
        features = np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts], axis=0).astype(np.float32)
        if int(features.size) < int(self.feature_mean.size):
            features = np.concatenate(
                [features, np.zeros((int(self.feature_mean.size) - int(features.size),), dtype=np.float32)],
                axis=0,
            )
        return features[: int(self.feature_mean.size)].astype(np.float32), "ok"

    def predict_from_trace(
        self,
        row: Mapping[str, Any],
        estimate: TaskFrameV46AlignmentEstimate,
        *,
        observation: Mapping[str, Any] | None = None,
        robot_state: Mapping[str, Any] | None = None,
        image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
        image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    ) -> TaskFrameV46YawControlSelectorDecision:
        features, reason = self.feature_vector_from_trace(
            row,
            estimate,
            observation=observation,
            robot_state=robot_state,
            image_crop_size=image_crop_size,
            image_resize_size=image_resize_size,
        )
        if features is None:
            return TaskFrameV46YawControlSelectorDecision(
                allowed=False,
                score=0.0,
                threshold=float(self.threshold),
                block_reason=str(reason),
                source=str(self.source),
            )
        normalized = (features - self.feature_mean) / self.feature_std
        self.model.eval()
        with torch.no_grad():
            score = float(torch.sigmoid(self.model(torch.as_tensor(normalized, dtype=torch.float32).reshape(1, -1)))[0].item())
        allowed = bool(np.isfinite(score) and score >= float(self.threshold))
        return TaskFrameV46YawControlSelectorDecision(
            allowed=allowed,
            score=score if np.isfinite(score) else 0.0,
            threshold=float(self.threshold),
            block_reason="ready" if allowed else "selector_below_threshold",
            source=str(self.source),
        )


@dataclass(frozen=True)
class TaskFrameV46AlignmentCalibration:
    scalar_feature_names: tuple[str, ...]
    history_feature_names: tuple[str, ...]
    history_window_size: int
    image_in_channels: int
    image_hidden_dim: int
    image_crop_size: int
    image_resize_size: int
    proprio_dim: int
    planner_prior_dim: int
    command_feature_dim: int
    fusion_hidden_dim: int
    yaw_hypotheses: int
    risk_classes: tuple[str, ...]
    max_abs_xy: float
    max_abs_z: float
    max_abs_yaw: float
    use_spatial_moments: bool
    model_state_dict: dict[str, torch.Tensor]
    source: str = "v46_unified_task_frame_alignment_candidate"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskFrameV46AlignmentCalibration":
        config = dict(payload.get("config", {}))
        scalar_feature_names = tuple(str(x) for x in config.get("scalar_feature_names", payload.get("scalar_feature_names", TASK_FRAME_READINESS_FEATURE_NAMES)))
        history_window_size = int(config.get("history_window_size", payload.get("history_window_size", RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW)))
        history_feature_names = tuple(
            str(x)
            for x in config.get(
                "history_feature_names",
                payload.get("history_feature_names", runtime_xy_spatial_temporal_feature_names(tuple(DEFAULT_RUNTIME_XY_FEATURE_NAMES), history_window_size)),
            )
        )
        image_in_channels = int(config.get("image_in_channels", payload.get("image_in_channels", 7)))
        image_hidden_dim = int(config.get("image_hidden_dim", payload.get("image_hidden_dim", 128)))
        image_crop_size = int(config.get("image_crop_size", payload.get("image_crop_size", RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE)))
        image_resize_size = int(config.get("image_resize_size", payload.get("image_resize_size", RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE)))
        proprio_dim = int(config.get("proprio_dim", payload.get("proprio_dim", 15)))
        planner_prior_dim = int(config.get("planner_prior_dim", payload.get("planner_prior_dim", 6)))
        state_dict = {str(k): torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in dict(payload.get("model_state_dict", {})).items()}
        has_near_field_head = bool("near_field_head.weight" in state_dict and "near_field_head.bias" in state_dict)
        if "fusion_hidden_dim" in config:
            fusion_hidden_dim = int(config["fusion_hidden_dim"])
        elif "fusion.2.bias" in state_dict:
            fusion_hidden_dim = int(state_dict["fusion.2.bias"].reshape(-1).shape[0])
        else:
            fusion_hidden_dim = int(payload.get("fusion_hidden_dim", 128))
        command_feature_dim = int(config.get("command_feature_dim", payload.get("command_feature_dim", 6)))
        yaw_hypotheses = int(config.get("yaw_hypotheses", payload.get("yaw_hypotheses", 4)))
        risk_classes = tuple(str(x) for x in config.get("risk_classes", payload.get("risk_classes", TASK_FRAME_V46_RISK_CLASSES)))
        max_abs_xy = float(config.get("max_abs_xy", payload.get("max_abs_xy", 0.040)))
        max_abs_z = float(config.get("max_abs_z", payload.get("max_abs_z", 0.030)))
        max_abs_yaw = float(config.get("max_abs_yaw", payload.get("max_abs_yaw", 0.250)))
        use_spatial_moments = bool(config.get("use_spatial_moments", payload.get("use_spatial_moments", False)))
        model = TaskFrameV46AlignmentNet(
            image_in_channels=image_in_channels,
            image_hidden_dim=image_hidden_dim,
            scalar_feature_dim=len(scalar_feature_names),
            history_feature_dim=max(1, len(history_feature_names) // max(1, history_window_size)),
            history_window_size=history_window_size,
            proprio_dim=proprio_dim,
            planner_prior_dim=planner_prior_dim,
            command_feature_dim=command_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            yaw_hypotheses=yaw_hypotheses,
            risk_classes=risk_classes,
            max_abs_xy=max_abs_xy,
            max_abs_z=max_abs_z,
            max_abs_yaw=max_abs_yaw,
            use_spatial_moments=use_spatial_moments,
        )
        model.load_state_dict(state_dict, strict=False)
        if not has_near_field_head:
            with torch.no_grad():
                model.near_field_head.weight.zero_()
                model.near_field_head.bias.fill_(-20.0)
            model.near_field_head_available = False
        return cls(
            scalar_feature_names=scalar_feature_names,
            history_feature_names=history_feature_names,
            history_window_size=history_window_size,
            image_in_channels=image_in_channels,
            image_hidden_dim=image_hidden_dim,
            image_crop_size=image_crop_size,
            image_resize_size=image_resize_size,
            proprio_dim=proprio_dim,
            planner_prior_dim=planner_prior_dim,
            command_feature_dim=command_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            yaw_hypotheses=yaw_hypotheses,
            risk_classes=risk_classes,
            max_abs_xy=max_abs_xy,
            max_abs_z=max_abs_z,
            max_abs_yaw=max_abs_yaw,
            use_spatial_moments=use_spatial_moments,
            model_state_dict=state_dict,
            source=str(payload.get("source", "v46_unified_task_frame_alignment_candidate")),
        ).with_model(model)

    @classmethod
    def load(cls, path: str | Path | None) -> "TaskFrameV46AlignmentCalibration | None":
        if path is None or not str(path):
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"missing task-frame v46 checkpoint: {p}")
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"task-frame v46 checkpoint must be a mapping: {p}")
        return cls.from_dict(payload)

    def with_model(self, model: TaskFrameV46AlignmentNet) -> "TaskFrameV46AlignmentCalibration":
        object.__setattr__(self, "_model", model)
        return self

    @property
    def model(self) -> TaskFrameV46AlignmentNet:
        model = getattr(self, "_model", None)
        if model is None:
            model = TaskFrameV46AlignmentNet(
                image_in_channels=self.image_in_channels,
                image_hidden_dim=self.image_hidden_dim,
                scalar_feature_dim=len(self.scalar_feature_names),
                history_feature_dim=max(1, len(self.history_feature_names) // max(1, self.history_window_size)),
                history_window_size=self.history_window_size,
                proprio_dim=self.proprio_dim,
                planner_prior_dim=self.planner_prior_dim,
                command_feature_dim=self.command_feature_dim,
                fusion_hidden_dim=self.fusion_hidden_dim,
                yaw_hypotheses=self.yaw_hypotheses,
                risk_classes=self.risk_classes,
                max_abs_xy=self.max_abs_xy,
                max_abs_z=self.max_abs_z,
                max_abs_yaw=self.max_abs_yaw,
                use_spatial_moments=self.use_spatial_moments,
            )
            model.load_state_dict(self.model_state_dict, strict=False)
            object.__setattr__(self, "_model", model)
        return model

    def to_checkpoint_dict(self, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": "c2c_v2_task_frame_v46_alignment_checkpoint_v1",
            "model_type": "v46_unified_task_frame_alignment",
            "config": {
                "scalar_feature_names": list(self.scalar_feature_names),
                "history_feature_names": list(self.history_feature_names),
                "history_window_size": int(self.history_window_size),
                "image_in_channels": int(self.image_in_channels),
                "image_hidden_dim": int(self.image_hidden_dim),
                "image_crop_size": int(self.image_crop_size),
                "image_resize_size": int(self.image_resize_size),
                "proprio_dim": int(self.proprio_dim),
                "planner_prior_dim": int(self.planner_prior_dim),
                "command_feature_dim": int(self.command_feature_dim),
                "fusion_hidden_dim": int(self.fusion_hidden_dim),
                "yaw_hypotheses": int(self.yaw_hypotheses),
                "risk_classes": list(self.risk_classes),
                "max_abs_xy": float(self.max_abs_xy),
                "max_abs_z": float(self.max_abs_z),
                "max_abs_yaw": float(self.max_abs_yaw),
                "use_spatial_moments": bool(self.use_spatial_moments),
                "has_near_field_head": bool("near_field_head.weight" in self.model_state_dict),
            },
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model_state_dict.items()},
            "source": str(self.source),
            "metadata": dict(metadata or {}),
        }

    def predict_from_trace(
        self,
        row: Mapping[str, Any],
        *,
        observation: Mapping[str, Any] | None = None,
        robot_state: Mapping[str, Any] | None = None,
        history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
    ) -> TaskFrameV46AlignmentEstimate:
        if observation is None:
            return invalid_task_frame_v46_estimate("missing_observation", source=self.source)
        rgbd = _load_spatial_temporal_rgbd(observation, robot_state, crop_size=int(self.image_crop_size), resize_size=int(self.image_resize_size))
        if rgbd is None:
            return invalid_task_frame_v46_estimate("missing_rgbd", source=self.source)
        history = runtime_xy_spatial_temporal_context_feature_vector_from_trace(
            row,
            history_rows=list(history_rows or ())[: max(0, int(self.history_window_size) - 1)],
            base_feature_names=tuple(DEFAULT_RUNTIME_XY_FEATURE_NAMES),
            window_size=int(self.history_window_size),
        )
        robot_state = robot_state or {}
        proprio = _coerce_vector(robot_state.get("proprio", row.get("proprio", None)), length=int(self.proprio_dim))
        planner_prior = _coerce_vector(
            robot_state.get("planner_delta_7d", row.get("planner_local_delta_6d", row.get("planner_prior_delta", None))),
            length=int(self.planner_prior_dim),
        )
        return self.model.predict_numpy(
            rgbd,
            task_frame_v46_scalar_feature_vector(row),
            history,
            proprio,
            planner_prior,
            source=self.source,
        )

    def predict_command_transition_from_trace(
        self,
        row: Mapping[str, Any],
        *,
        observation: Mapping[str, Any] | None = None,
        robot_state: Mapping[str, Any] | None = None,
        history_rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
        command_6d: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Predict the task-frame residual delta for a proposed local 6D command.

        This is a runtime-visible transition estimate. It consumes the same
        non-privileged RGBD/proprio/history inputs as the residual estimator and
        a candidate local command. It does not grant close authority.
        """

        if observation is None:
            return {"valid": False, "reason": "missing_observation"}
        rgbd = _load_spatial_temporal_rgbd(observation, robot_state, crop_size=int(self.image_crop_size), resize_size=int(self.image_resize_size))
        if rgbd is None:
            return {"valid": False, "reason": "missing_rgbd"}
        history = runtime_xy_spatial_temporal_context_feature_vector_from_trace(
            row,
            history_rows=list(history_rows or ())[: max(0, int(self.history_window_size) - 1)],
            base_feature_names=tuple(DEFAULT_RUNTIME_XY_FEATURE_NAMES),
            window_size=int(self.history_window_size),
        )
        robot_state = robot_state or {}
        proprio = _coerce_vector(robot_state.get("proprio", row.get("proprio", None)), length=int(self.proprio_dim))
        planner_prior = _coerce_vector(
            robot_state.get("planner_delta_7d", row.get("planner_local_delta_6d", row.get("planner_prior_delta", None))),
            length=int(self.planner_prior_dim),
        )
        command = _coerce_vector(command_6d, length=6)
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward(
                rgbd.unsqueeze(0) if rgbd.ndim == 3 else rgbd,
                torch.as_tensor(task_frame_v46_scalar_feature_vector(row), dtype=torch.float32).reshape(1, -1),
                torch.as_tensor(history, dtype=torch.float32).reshape(1, -1),
                torch.as_tensor(proprio, dtype=torch.float32).reshape(1, -1),
                torch.as_tensor(planner_prior, dtype=torch.float32).reshape(1, -1),
                torch.as_tensor(command, dtype=torch.float32).reshape(1, -1),
            )
        delta = out["command_delta"][0].detach().cpu().numpy().astype(np.float32)
        logvar = out["command_logvar"][0].detach().cpu().numpy().astype(np.float32)
        support = float(out["command_support"][0].item())
        return {
            "valid": bool(np.all(np.isfinite(delta)) and np.all(np.isfinite(logvar)) and np.isfinite(support)),
            "reason": "ok",
            "delta": [float(v) for v in delta[:4].tolist()],
            "logvar": [float(v) for v in logvar[:4].tolist()],
            "support": float(np.clip(support, 0.0, 1.0)),
            "uses_privileged_runtime": False,
            "close_control_allowed": False,
            "source": str(self.source),
        }


def invalid_task_frame_v46_estimate(reason: str, *, source: str = "v46_unified_task_frame_alignment_candidate") -> TaskFrameV46AlignmentEstimate:
    return TaskFrameV46AlignmentEstimate(
        dx=0.0,
        dy=0.0,
        dz=0.0,
        dyaw=0.0,
        xy_confidence=0.0,
        z_confidence=0.0,
        yaw_confidence=0.0,
        xy_observable=False,
        z_observable=False,
        yaw_observable=False,
        yaw_ambiguous=False,
        yaw_unobservable=True,
        xy_step_scale=0.0,
        z_step_scale=0.0,
        yaw_step_scale=0.0,
        yaw_hypothesis_index=0,
        yaw_hypothesis_gap=0.0,
        xy_control_effect=(0.0, 0.0, 0.0, 0.0),
        risk_reason=str(reason),
        near_field_confidence=0.0,
        near_field_head_available=False,
        source=str(source),
    )


def save_task_frame_v46_alignment_checkpoint(
    path: str | Path,
    model: TaskFrameV46AlignmentNet,
    *,
    metadata: Mapping[str, Any] | None = None,
    image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
) -> None:
    history_names = runtime_xy_spatial_temporal_feature_names(tuple(DEFAULT_RUNTIME_XY_FEATURE_NAMES), int(model.history_window_size))
    calibration = TaskFrameV46AlignmentCalibration(
        scalar_feature_names=tuple(TASK_FRAME_READINESS_FEATURE_NAMES),
        history_feature_names=tuple(history_names),
        history_window_size=int(model.history_window_size),
        image_in_channels=int(model.image_in_channels),
        image_hidden_dim=int(model.image_hidden_dim),
        image_crop_size=int(image_crop_size),
        image_resize_size=int(image_resize_size),
        proprio_dim=int(model.proprio_dim),
        planner_prior_dim=int(model.planner_prior_dim),
        command_feature_dim=int(model.command_feature_dim),
        fusion_hidden_dim=int(model.fusion_hidden_dim),
        yaw_hypotheses=int(model.yaw_hypotheses),
        risk_classes=tuple(model.risk_classes),
        max_abs_xy=float(model.max_abs_xy),
        max_abs_z=float(model.max_abs_z),
        max_abs_yaw=float(model.max_abs_yaw),
        use_spatial_moments=bool(model.use_spatial_moments),
        model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
    )
    torch.save(calibration.to_checkpoint_dict(metadata=metadata), str(path))


def load_task_frame_v46_alignment_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TaskFrameV46AlignmentCalibration, dict[str, Any]]:
    payload = torch.load(str(path), map_location=map_location)
    calibration = TaskFrameV46AlignmentCalibration.from_dict(payload)
    return calibration, dict(payload.get("metadata", {}))


def load_task_frame_v46_yaw_control_selector_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TaskFrameV46YawControlSelectorCalibration, dict[str, Any]]:
    payload = torch.load(str(path), map_location=map_location)
    calibration = TaskFrameV46YawControlSelectorCalibration.from_dict(payload)
    return calibration, dict(payload.get("metadata", {}))


def task_frame_v46_micro_servo_step(
    estimate: TaskFrameV46AlignmentEstimate,
    *,
    history_rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None = None,
    force_safe: bool = True,
    xy_gain: float = TASK_FRAME_V46_DEFAULT_XY_GAIN,
    z_gain: float = TASK_FRAME_V46_DEFAULT_Z_GAIN,
    yaw_gain: float = TASK_FRAME_V46_DEFAULT_YAW_GAIN,
    max_xy_step: float = TASK_FRAME_V46_DEFAULT_MAX_XY_STEP,
    max_z_step: float = TASK_FRAME_V46_DEFAULT_MAX_Z_STEP,
    max_yaw_step: float = TASK_FRAME_V46_DEFAULT_MAX_YAW_STEP,
    confidence_threshold: float = TASK_FRAME_V46_DEFAULT_CONFIDENCE_THRESHOLD,
    weak_confidence_threshold: float = TASK_FRAME_V46_WEAK_CONFIDENCE_THRESHOLD,
    min_step_scale: float = TASK_FRAME_V46_DEFAULT_MIN_STEP_SCALE,
    low_visibility: bool = False,
    yaw_selector_decision: TaskFrameV46YawControlSelectorDecision | None = None,
) -> tuple[TaskFrameV46ControlDecision, np.ndarray]:
    effective_min_step_scale = float(min(min_step_scale, TASK_FRAME_V46_LOW_VIS_MIN_STEP_SCALE if low_visibility else min_step_scale))
    xy_conf_ok = float(estimate.xy_confidence) >= float(confidence_threshold)
    z_conf_ok = float(estimate.z_confidence) >= float(confidence_threshold)
    yaw_conf_ok = float(estimate.yaw_confidence) >= float(confidence_threshold)
    xy_weak = bool(float(estimate.xy_confidence) >= float(weak_confidence_threshold) and not xy_conf_ok)
    z_weak = bool(float(estimate.z_confidence) >= float(weak_confidence_threshold) and not z_conf_ok)
    xy_scale_ok = bool(float(estimate.xy_step_scale) >= effective_min_step_scale)
    z_scale_ok = bool(float(estimate.z_step_scale) >= effective_min_step_scale)
    yaw_scale_ok = bool(float(estimate.yaw_step_scale) >= effective_min_step_scale)
    yaw_history_stable = _history_direction_stable(history_rows, field="dyaw", current_value=float(estimate.dyaw))
    xy_allowed = bool(force_safe and estimate.xy_observable and xy_scale_ok and (xy_conf_ok or xy_weak))
    z_allowed = bool(force_safe and estimate.z_observable and z_scale_ok and (z_conf_ok or z_weak))
    yaw_allowed = bool(
        force_safe
        and estimate.yaw_observable
        and not estimate.yaw_ambiguous
        and not estimate.yaw_unobservable
        and yaw_conf_ok
        and yaw_scale_ok
        and yaw_history_stable
    )
    yaw_selector_loaded = yaw_selector_decision is not None
    yaw_selector_allowed = bool(yaw_selector_decision.allowed) if yaw_selector_decision is not None else True
    if yaw_selector_loaded and not yaw_selector_allowed:
        yaw_allowed = False
    risk_reason = str(estimate.risk_reason)
    xy_risk_scale = float(TASK_FRAME_V46_XY_RISK_STEP_SCALE.get(risk_reason, 1.0))
    z_risk_scale = float(TASK_FRAME_V46_Z_RISK_STEP_SCALE.get(risk_reason, 1.0))
    xy_scale = float(estimate.xy_step_scale) * (0.35 if xy_weak else 1.0) * xy_risk_scale
    z_scale = float(estimate.z_step_scale) * (0.35 if z_weak else 1.0) * z_risk_scale
    dx_step = float(np.clip(float(xy_gain) * float(estimate.dx) * xy_scale, -float(max_xy_step), float(max_xy_step))) if xy_allowed else 0.0
    dy_step = float(np.clip(float(xy_gain) * float(estimate.dy) * xy_scale, -float(max_xy_step), float(max_xy_step))) if xy_allowed else 0.0
    dz_step = float(np.clip(float(z_gain) * float(estimate.dz) * z_scale, -float(max_z_step), float(max_z_step))) if z_allowed else 0.0
    dyaw_step = float(np.clip(float(yaw_gain) * float(estimate.dyaw) * float(estimate.yaw_step_scale), -float(max_yaw_step), float(max_yaw_step))) if yaw_allowed else 0.0
    xy_block = "ready" if xy_allowed else ("force_guard" if not force_safe else "xy_not_observable" if not estimate.xy_observable else "xy_step_scale_too_small" if not xy_scale_ok else "xy_low_confidence")
    z_block = "ready" if z_allowed else ("force_guard" if not force_safe else "z_not_observable" if not estimate.z_observable else "z_step_scale_too_small" if not z_scale_ok else "z_low_confidence")
    yaw_block = "ready" if yaw_allowed else (
        "force_guard" if not force_safe else
        "yaw_not_observable" if (not estimate.yaw_observable or estimate.yaw_unobservable) else
        "yaw_ambiguous" if estimate.yaw_ambiguous else
        "yaw_step_scale_too_small" if not yaw_scale_ok else
        "yaw_low_confidence" if not yaw_conf_ok else
        str(yaw_selector_decision.block_reason) if yaw_selector_loaded and not yaw_selector_allowed else
        "yaw_history_not_stable"
    )
    applied = bool(xy_allowed or z_allowed or yaw_allowed)
    block_reason = "ready" if applied else (xy_block if xy_block != "ready" else z_block if z_block != "ready" else yaw_block)
    local_step = np.zeros((6,), dtype=np.float32)
    local_step[0] = dx_step
    local_step[1] = dy_step
    local_step[2] = dz_step
    local_step[5] = dyaw_step
    return (
        TaskFrameV46ControlDecision(
            applied=applied,
            xy_allowed=xy_allowed,
            z_allowed=z_allowed,
            yaw_allowed=yaw_allowed,
            xy_weak=xy_weak,
            z_weak=z_weak,
            yaw_history_stable=yaw_history_stable,
            force_safe=bool(force_safe),
            xy_block_reason=str(xy_block),
            z_block_reason=str(z_block),
            yaw_block_reason=str(yaw_block),
            block_reason=str(block_reason),
            dx_step=dx_step,
            dy_step=dy_step,
            dz_step=dz_step,
            dyaw_step=dyaw_step,
            xy_risk_step_scale=xy_risk_scale,
            z_risk_step_scale=z_risk_scale,
        ),
        local_step,
    )


def task_frame_v46_effect_aware_xy_correction(
    estimate: TaskFrameV46AlignmentEstimate,
    *,
    current_local_xy: np.ndarray | list[float] | tuple[float, float],
    max_xy_step: float = TASK_FRAME_V46_DEFAULT_MAX_XY_STEP,
    min_confidence: float = TASK_FRAME_V46_WEAK_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """Choose a bounded XY correction using the learned local control effect.

    The effect matrix maps local XY command to expected task-frame residual
    delta. The returned value is a correction to add on top of the current
    local command. It does not grant close authority. The correction still
    obeys the same evidence/risk soft gate as the direct v46 micro-servo path,
    so an effect head cannot turn low evidence or direction-conflict frames
    into full-step blind XY control.
    """

    effect = np.asarray(estimate.xy_control_effect, dtype=np.float32).reshape(2, 2)
    current = _coerce_vector(current_local_xy, length=2)
    residual = np.asarray([float(estimate.dx), float(estimate.dy)], dtype=np.float32)
    confidence = float(np.clip(float(estimate.xy_confidence), 0.0, 1.0))
    step_scale = float(np.clip(float(estimate.xy_step_scale), 0.0, 1.0))
    risk_scale = float(TASK_FRAME_V46_XY_RISK_STEP_SCALE.get(str(estimate.risk_reason), 1.0))
    effective_max_xy_step = float(max_xy_step) * step_scale * risk_scale
    if not bool(estimate.xy_observable) or confidence < float(min_confidence):
        return np.zeros((2,), dtype=np.float32)
    if not np.isfinite(effective_max_xy_step) or effective_max_xy_step <= 0.0:
        return np.zeros((2,), dtype=np.float32)
    if not np.all(np.isfinite(effect)) or not np.all(np.isfinite(current)) or not np.all(np.isfinite(residual)):
        return np.zeros((2,), dtype=np.float32)
    if float(np.linalg.norm(effect, ord="fro")) <= 1.0e-6:
        return np.zeros((2,), dtype=np.float32)
    damped = effect + 1.0e-3 * np.eye(2, dtype=np.float32)
    try:
        desired_full = -np.linalg.pinv(damped).dot(residual)
    except Exception:
        return np.zeros((2,), dtype=np.float32)
    correction = desired_full.astype(np.float32) - current.astype(np.float32)
    return np.clip(correction, -effective_max_xy_step, effective_max_xy_step).astype(np.float32)


def _score_task_frame_v46_post_residual(
    residual: np.ndarray,
    *,
    xy_weight: float = 1.0,
    z_weight: float = 1.0,
    yaw_weight: float = 0.25,
    uncertainty: np.ndarray | None = None,
    uncertainty_weight: float = 0.05,
) -> float:
    r = _coerce_vector(residual, length=4)
    if not np.all(np.isfinite(r)):
        return float("inf")
    score = (
        float(xy_weight) * float(np.linalg.norm(r[:2]))
        + float(z_weight) * abs(float(r[2]))
        + float(yaw_weight) * abs(float(r[3]))
    )
    if uncertainty is not None:
        u = _coerce_vector(uncertainty, length=4)
        if np.all(np.isfinite(u)):
            score += float(uncertainty_weight) * float(np.mean(np.sqrt(np.exp(np.clip(u, -20.0, 20.0)))))
    return float(score)


def task_frame_v46_transition_command_search(
    estimate: TaskFrameV46AlignmentEstimate,
    *,
    base_command_local_6d: np.ndarray | list[float] | tuple[float, ...],
    proposed_step_local_6d: np.ndarray | list[float] | tuple[float, ...],
    transition_predictor: Callable[[np.ndarray], Mapping[str, Any]],
    max_xy_step: float = TASK_FRAME_V46_DEFAULT_MAX_XY_STEP,
    max_z_step: float = TASK_FRAME_V46_DEFAULT_MAX_Z_STEP,
    max_yaw_step: float = TASK_FRAME_V46_DEFAULT_MAX_YAW_STEP,
    support_threshold: float = 0.20,
    improvement_margin: float = 1.0e-5,
) -> TaskFrameV46CommandSearchResult:
    """Select a bounded local correction with the command-transition head.

    The selector is intentionally local and conservative: it evaluates a small
    set of bounded additive commands and returns a non-zero correction only when
    the predicted post-command residual is better than the no-correction command
    with enough transition support. It never emits gripper close authority.
    """

    base = _coerce_vector(base_command_local_6d, length=6)
    proposed = _coerce_vector(proposed_step_local_6d, length=6)
    proposed[0] = float(np.clip(float(proposed[0]), -float(max_xy_step), float(max_xy_step)))
    proposed[1] = float(np.clip(float(proposed[1]), -float(max_xy_step), float(max_xy_step)))
    proposed[2] = float(np.clip(float(proposed[2]), -float(max_z_step), float(max_z_step)))
    proposed[5] = float(np.clip(float(proposed[5]), -float(max_yaw_step), float(max_yaw_step)))
    residual = np.asarray([float(estimate.dx), float(estimate.dy), float(estimate.dz), float(estimate.dyaw)], dtype=np.float32)
    pre_score = _score_task_frame_v46_post_residual(residual)

    def _empty(reason: str) -> TaskFrameV46CommandSearchResult:
        zero = np.zeros((6,), dtype=np.float32)
        return TaskFrameV46CommandSearchResult(
            selected_step_local_6d=tuple(float(v) for v in zero.tolist()),
            selected_command_local_6d=tuple(float(v) for v in base.tolist()),
            selected_delta=(0.0, 0.0, 0.0, 0.0),
            selected_logvar=(0.0, 0.0, 0.0, 0.0),
            selected_support=0.0,
            selected_score=float(pre_score),
            no_correction_score=float(pre_score),
            pre_score=float(pre_score),
            xy_contracts=False,
            z_contracts=False,
            yaw_contracts=False,
            applied=False,
            valid=False,
            reason=str(reason),
            candidate_count=0,
        )

    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(proposed)) or not np.all(np.isfinite(residual)):
        return _empty("invalid_input")

    candidates: list[np.ndarray] = []

    def _add_candidate(step: np.ndarray) -> None:
        s = _coerce_vector(step, length=6)
        s[0] = float(np.clip(float(s[0]), -float(max_xy_step), float(max_xy_step)))
        s[1] = float(np.clip(float(s[1]), -float(max_xy_step), float(max_xy_step)))
        s[2] = float(np.clip(float(s[2]), -float(max_z_step), float(max_z_step)))
        s[3] = 0.0
        s[4] = 0.0
        s[5] = float(np.clip(float(s[5]), -float(max_yaw_step), float(max_yaw_step)))
        if not any(np.allclose(s, existing, atol=1.0e-9, rtol=0.0) for existing in candidates):
            candidates.append(s.astype(np.float32))

    _add_candidate(np.zeros((6,), dtype=np.float32))
    _add_candidate(proposed)
    _add_candidate(0.5 * proposed)
    for axis in (0, 1, 2, 5):
        if abs(float(proposed[axis])) > 1.0e-9:
            s = np.zeros((6,), dtype=np.float32)
            s[axis] = float(proposed[axis])
            _add_candidate(s)
            s = np.zeros((6,), dtype=np.float32)
            s[axis] = -float(proposed[axis])
            _add_candidate(s)
    if bool(estimate.xy_observable):
        xy_mag = float(max(abs(float(proposed[0])), abs(float(proposed[1])), min(float(max_xy_step), 0.0015)))
        for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)):
            s = np.zeros((6,), dtype=np.float32)
            s[axis] = float(sign * xy_mag)
            _add_candidate(s)
    if bool(estimate.z_observable):
        z_mag = float(max(abs(float(proposed[2])), min(float(max_z_step), 0.0015)))
        for sign in (1.0, -1.0):
            s = np.zeros((6,), dtype=np.float32)
            s[2] = float(sign * z_mag)
            _add_candidate(s)
    if bool(estimate.yaw_observable) and not bool(estimate.yaw_ambiguous) and not bool(estimate.yaw_unobservable):
        yaw_mag = float(max(abs(float(proposed[5])), min(float(max_yaw_step), 0.008)))
        for sign in (1.0, -1.0):
            s = np.zeros((6,), dtype=np.float32)
            s[5] = float(sign * yaw_mag)
            _add_candidate(s)

    best: dict[str, Any] | None = None
    no_correction_score = float("inf")
    no_correction_transition: dict[str, Any] | None = None
    for step in candidates:
        command = (base + step).astype(np.float32)
        try:
            transition = dict(transition_predictor(command))
        except Exception as exc:
            transition = {"valid": False, "reason": type(exc).__name__}
        delta = _coerce_vector(transition.get("delta", None), length=4)
        logvar = _coerce_vector(transition.get("logvar", None), length=4)
        support = float(np.clip(_as_float(transition.get("support", 0.0), 0.0), 0.0, 1.0))
        valid = bool(transition.get("valid", False)) and support >= float(support_threshold)
        post = residual + delta
        score = _score_task_frame_v46_post_residual(post, uncertainty=logvar)
        is_zero = bool(np.linalg.norm(step) <= 1.0e-9)
        if is_zero:
            no_correction_score = float(score if valid else pre_score)
            no_correction_transition = {"delta": delta, "logvar": logvar, "support": support, "valid": valid, "command": command, "step": step}
        if not valid or not np.isfinite(score):
            continue
        post = residual + delta
        pre_xy = float(np.linalg.norm(residual[:2]))
        post_xy = float(np.linalg.norm(post[:2]))
        xy_contracts = bool(np.isfinite(pre_xy) and np.isfinite(post_xy) and post_xy < pre_xy - 1.0e-7)
        z_contracts = bool(np.isfinite(post[2]) and abs(float(post[2])) < abs(float(residual[2])) - 1.0e-7)
        yaw_contracts = bool(np.isfinite(post[3]) and abs(float(post[3])) < abs(float(residual[3])) - 1.0e-7)
        if not is_zero:
            moves_xy = bool(float(np.linalg.norm(step[:2])) > 1.0e-9)
            moves_z = bool(abs(float(step[2])) > 1.0e-9)
            moves_yaw = bool(abs(float(step[5])) > 1.0e-9)
            if (moves_xy and not xy_contracts) or (moves_z and not z_contracts) or (moves_yaw and not yaw_contracts):
                continue
        record = {
            "step": step,
            "command": command,
            "delta": delta,
            "logvar": logvar,
            "support": support,
            "score": float(score),
        }
        if best is None or float(record["score"]) < float(best["score"]):
            best = record

    if best is None:
        return _empty("no_valid_transition_candidate")
    baseline_score = float(no_correction_score if np.isfinite(no_correction_score) else pre_score)
    applied = bool(np.linalg.norm(best["step"]) > 1.0e-9 and float(best["score"]) < baseline_score - float(improvement_margin))
    selected = best if applied else (
        no_correction_transition
        if no_correction_transition is not None
        else {"step": np.zeros((6,), dtype=np.float32), "command": base, "delta": np.zeros((4,), dtype=np.float32), "logvar": np.zeros((4,), dtype=np.float32), "support": 0.0, "score": baseline_score}
    )
    selected_step = _coerce_vector(selected["step"], length=6)
    selected_command = _coerce_vector(selected["command"], length=6)
    selected_delta = _coerce_vector(selected["delta"], length=4)
    selected_logvar = _coerce_vector(selected["logvar"], length=4)
    predicted_post = residual + selected_delta
    pre_xy = float(np.linalg.norm(residual[:2]))
    post_xy = float(np.linalg.norm(predicted_post[:2]))
    xy_contracts = bool(np.isfinite(pre_xy) and np.isfinite(post_xy) and post_xy < pre_xy - 1.0e-7)
    z_contracts = bool(np.isfinite(predicted_post[2]) and abs(float(predicted_post[2])) < abs(float(residual[2])) - 1.0e-7)
    yaw_contracts = bool(np.isfinite(predicted_post[3]) and abs(float(predicted_post[3])) < abs(float(residual[3])) - 1.0e-7)
    return TaskFrameV46CommandSearchResult(
        selected_step_local_6d=tuple(float(v) for v in selected_step.tolist()),
        selected_command_local_6d=tuple(float(v) for v in selected_command.tolist()),
        selected_delta=tuple(float(v) for v in selected_delta.tolist()),
        selected_logvar=tuple(float(v) for v in selected_logvar.tolist()),
        selected_support=float(selected.get("support", 0.0)),
        selected_score=float(selected.get("score", baseline_score)),
        no_correction_score=float(baseline_score),
        pre_score=float(pre_score),
        xy_contracts=xy_contracts,
        z_contracts=z_contracts,
        yaw_contracts=yaw_contracts,
        applied=bool(applied),
        valid=True,
        reason="selected_transition_candidate" if applied else "no_candidate_improves_no_correction",
        candidate_count=len(candidates),
    )


def task_frame_v46_labels_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    labels = row.get("offline_labels", {})
    labels = labels if isinstance(labels, Mapping) else {}
    yaw_label = row.get("yaw_label", {})
    yaw_label = yaw_label if isinstance(yaw_label, Mapping) else {}
    true_residual = row.get("true_residual", {})
    true_residual = true_residual if isinstance(true_residual, Mapping) else {}
    dx = _as_float(labels.get("dx", row.get("privileged_dx", true_residual.get("dx", float("nan")))), float("nan"))
    dy = _as_float(labels.get("dy", row.get("privileged_dy", true_residual.get("dy", float("nan")))), float("nan"))
    dz = _as_float(labels.get("dz", row.get("privileged_dz", true_residual.get("dz", float("nan")))), float("nan"))
    dyaw = _as_float(labels.get("dyaw", row.get("privileged_dyaw", true_residual.get("dyaw", float("nan")))), float("nan"))
    if not np.all(np.isfinite([dx, dy, dz, dyaw])):
        return None
    yaw_class = str(row.get("yaw_observability_class", yaw_label.get("yaw_observability_class", "")) or "").strip().lower()
    yaw_ambiguous = _as_bool(
        labels.get("yaw_ambiguous", labels.get("yaw_ambiguous_label", yaw_class in {"ambiguous", "unobservable"})),
        yaw_class in {"ambiguous", "unobservable"},
    )
    yaw_observable = _as_bool(
        labels.get(
            "yaw_observable",
            labels.get(
                "yaw_observable_label",
                row.get("yaw_control_observable", row.get("yaw_observable", yaw_label.get("yaw_control_observable", yaw_class == "observable"))),
            ),
        ),
        yaw_class == "observable",
    )
    if yaw_class == "unobservable":
        yaw_observable = False
    if yaw_class == "ambiguous":
        yaw_ambiguous = True
    z_observable = _as_bool(labels.get("z_observable", labels.get("z_observable_label", True)), True)
    xy_observable = _as_bool(labels.get("xy_observable", labels.get("xy_observable_label", True)), True)
    return {
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "dyaw": float(dyaw),
        "xy_observable": bool(xy_observable),
        "z_observable": bool(z_observable),
        "yaw_observable": bool(yaw_observable),
        "yaw_ambiguous": bool(yaw_ambiguous),
        "uses_privileged_runtime": False,
        "privileged_label_offline_only": True,
    }
