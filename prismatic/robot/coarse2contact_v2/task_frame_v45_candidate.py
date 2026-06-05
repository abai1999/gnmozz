"""Joint task-frame Z/Yaw residual candidate for C2C v2.

The v42 XY baseline stays frozen. This module only predicts bounded task-frame
Z/Yaw residuals, readiness confidences, and a small step-scale / risk label
that can be used for a guarded micro-servo or for replay-only trace audit.
It does not own close authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .task_frame_readiness import (
    TASK_FRAME_READINESS_FEATURE_NAMES,
    TaskFrameYawReadinessEstimate,
    TaskFrameZReadinessEstimate,
    task_frame_readiness_feature_vector,
)

TASK_FRAME_V45_RISK_CLASSES: tuple[str, ...] = (
    "normal",
    "low_visibility",
    "direction_conflict",
    "insufficient_support",
    "force_guard",
)

TASK_FRAME_V45_DEFAULT_Z_GAIN = 0.35
TASK_FRAME_V45_DEFAULT_YAW_GAIN = 0.25
TASK_FRAME_V45_DEFAULT_MAX_Z_STEP = 0.0030
TASK_FRAME_V45_DEFAULT_MAX_YAW_STEP = 0.020
TASK_FRAME_V45_MIN_HISTORY_SUPPORT = 2
TASK_FRAME_V45_DEFAULT_MIN_STEP_SCALE = 0.05
TASK_FRAME_V45_LOW_VIS_MIN_STEP_SCALE = 0.0015


def task_frame_v45_candidate_feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    return task_frame_readiness_feature_vector(row)


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


def _offline_labels(row: Mapping[str, Any]) -> Mapping[str, Any]:
    labels = row.get("offline_labels", {})
    return labels if isinstance(labels, Mapping) else {}


def _risk_reason_from_row(row: Mapping[str, Any]) -> str:
    runtime = row.get("runtime_features", {})
    runtime = runtime if isinstance(runtime, Mapping) else {}
    labels = _offline_labels(row)
    if _as_bool(row.get("wrist_is_occluded"), False) or _as_bool(row.get("wrist_is_low_visibility"), False):
        return "low_visibility"
    force_norm = _as_float(runtime.get("force_norm", row.get("grasp_contact_rule_force_norm", 0.0)), 0.0)
    if _as_bool(runtime.get("contact_confirmed", row.get("grasp_contact_rule_contact_confirmed", False)), False) or force_norm > 0.18:
        return "force_guard"
    yaw_ambiguous = _as_bool(labels.get("yaw_ambiguous", labels.get("yaw_ambiguous_label", False)), False)
    yaw_ready = _as_bool(labels.get("yaw_ready", labels.get("yaw_ready_label", False)), False)
    z_ready = _as_bool(labels.get("z_ready", labels.get("z_ready_label", False)), False)
    if yaw_ambiguous:
        return "direction_conflict"
    if not z_ready or not yaw_ready:
        return "insufficient_support"
    return "normal"


def _step_scale_target_from_row(row: Mapping[str, Any]) -> float:
    labels = _offline_labels(row)
    runtime = row.get("runtime_features", {})
    runtime = runtime if isinstance(runtime, Mapping) else {}
    dz = abs(_as_float(labels.get("dz", row.get("privileged_dz", 0.0)), 0.0))
    dyaw = abs(_as_float(labels.get("dyaw", row.get("privileged_dyaw", 0.0)), 0.0))
    z_ready = _as_bool(labels.get("z_ready", labels.get("z_ready_label", False)), False)
    yaw_ready = _as_bool(labels.get("yaw_ready", labels.get("yaw_ready_label", False)), False)
    low_visibility = _as_bool(row.get("wrist_is_low_visibility"), False) or _as_bool(row.get("wrist_is_occluded"), False)
    contact = _as_bool(runtime.get("contact_confirmed", row.get("grasp_contact_rule_contact_confirmed", False)), False)
    force_norm = _as_float(runtime.get("force_norm", row.get("grasp_contact_rule_force_norm", 0.0)), 0.0)
    z_scale = float(np.clip(1.0 - dz / 0.020, 0.05, 1.0))
    yaw_scale = float(np.clip(1.0 - dyaw / 0.140, 0.05, 1.0))
    target = float(min(z_scale, yaw_scale))
    if not z_ready:
        target *= 0.85
    if not yaw_ready:
        target *= 0.75
    if low_visibility:
        target *= 0.80
    if contact or force_norm > 0.18:
        target *= 0.75
    return float(np.clip(target, 0.05, 1.0))


def _confidence_target_from_axis_error(
    *,
    axis_error: float,
    axis_threshold: float,
    ready: bool,
    observable: bool,
    low_visibility: bool,
) -> float:
    if not np.isfinite(float(axis_error)) or float(axis_threshold) <= 0.0:
        return 0.0
    base = float(np.clip(1.0 - abs(float(axis_error)) / max(float(axis_threshold) * 2.0, 1.0e-6), 0.05, 1.0))
    if not bool(ready):
        base *= 0.80
    if not bool(observable):
        base *= 0.70
    if bool(low_visibility):
        base *= 0.80
    return float(np.clip(base, 0.05, 1.0))


def _safe_sign(value: float, *, eps: float = 1.0e-6) -> int:
    if not np.isfinite(float(value)):
        return 0
    if abs(float(value)) <= float(eps):
        return 0
    return 1 if float(value) > 0.0 else -1


def _history_direction_stable(
    history_rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None,
    *,
    field: str,
    current_value: float,
    required_support: int = TASK_FRAME_V45_MIN_HISTORY_SUPPORT,
) -> bool:
    if history_rows is None:
        return False
    recent: list[float] = []
    for row in reversed(list(history_rows)):
        value = row.get(field)
        try:
            value_f = float(value)
        except Exception:
            continue
        if not np.isfinite(value_f):
            continue
        if _safe_sign(value_f) == 0:
            continue
        recent.append(value_f)
        if len(recent) >= int(required_support):
            break
    if len(recent) < int(required_support):
        return False
    current_sign = _safe_sign(current_value)
    if current_sign == 0:
        return False
    if any(_safe_sign(value) != current_sign for value in recent):
        return False
    return True


def _row_low_visibility(row: Mapping[str, Any]) -> bool:
    return bool(
        _as_bool(row.get("wrist_is_occluded"), False)
        or _as_bool(row.get("wrist_is_low_visibility"), False)
        or str(row.get("visual_observability_class", "")).strip().lower() in {
            "occluded",
            "low_visibility",
            "low_observability",
            "partial_observable",
            "partial_observation",
        }
    )


@dataclass(frozen=True)
class TaskFrameV45MicroServoDecision:
    applied: bool
    xy_ready: bool
    z_observable: bool
    yaw_observable: bool
    yaw_ambiguous: bool
    yaw_history_stable: bool
    force_safe: bool
    z_allowed: bool
    yaw_allowed: bool
    z_block_reason: str
    yaw_block_reason: str
    block_reason: str
    dz_step: float
    dyaw_step: float
    step_scale: float
    source: str = "task_frame_v45_candidate"
    uses_privileged_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_frame_v45_applied": bool(self.applied),
            "task_frame_v45_xy_ready": bool(self.xy_ready),
            "task_frame_v45_z_observable": bool(self.z_observable),
            "task_frame_v45_yaw_observable": bool(self.yaw_observable),
            "task_frame_v45_yaw_ambiguous": bool(self.yaw_ambiguous),
            "task_frame_v45_yaw_history_stable": bool(self.yaw_history_stable),
            "task_frame_v45_force_safe": bool(self.force_safe),
            "task_frame_v45_z_allowed": bool(self.z_allowed),
            "task_frame_v45_yaw_allowed": bool(self.yaw_allowed),
            "task_frame_v45_z_block_reason": str(self.z_block_reason),
            "task_frame_v45_yaw_block_reason": str(self.yaw_block_reason),
            "task_frame_v45_block_reason": str(self.block_reason),
            "task_frame_v45_dz_step": float(self.dz_step),
            "task_frame_v45_dyaw_step": float(self.dyaw_step),
            "task_frame_v45_step_scale": float(self.step_scale),
            "task_frame_v45_source": str(self.source),
            "uses_privileged_runtime": bool(self.uses_privileged_runtime),
        }


def task_frame_v45_micro_servo_step(
    row: Mapping[str, Any],
    *,
    model: TaskFrameV45CandidateNet,
    history_rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None = None,
    xy_ready: bool,
    z_readiness: TaskFrameZReadinessEstimate | None = None,
    yaw_readiness: TaskFrameYawReadinessEstimate | None = None,
    force_safe: bool = True,
    z_gain: float = TASK_FRAME_V45_DEFAULT_Z_GAIN,
    yaw_gain: float = TASK_FRAME_V45_DEFAULT_YAW_GAIN,
    max_z_step: float = TASK_FRAME_V45_DEFAULT_MAX_Z_STEP,
    max_yaw_step: float = TASK_FRAME_V45_DEFAULT_MAX_YAW_STEP,
    min_z_confidence: float = 0.45,
    min_yaw_confidence: float = 0.45,
    min_step_scale: float = TASK_FRAME_V45_DEFAULT_MIN_STEP_SCALE,
) -> tuple[TaskFrameV45CandidateEstimate, TaskFrameV45MicroServoDecision, np.ndarray]:
    features = task_frame_v45_candidate_feature_vector(row)
    estimate = model.predict_numpy(features)
    z_observable = bool(z_readiness.z_observable) if z_readiness is not None else bool(estimate.z_ready)
    yaw_observable = bool(yaw_readiness.yaw_observable) if yaw_readiness is not None else bool(estimate.yaw_ready)
    yaw_ambiguous = bool(yaw_readiness.yaw_ambiguous) if yaw_readiness is not None else bool(estimate.yaw_ambiguous)
    z_confidence_ok = bool(float(estimate.z_confidence) >= float(min_z_confidence))
    yaw_confidence_ok = bool(float(estimate.yaw_confidence) >= float(min_yaw_confidence))
    step_scale = float(np.clip(float(estimate.step_scale), 0.0, 1.0))
    low_visibility = bool(_row_low_visibility(row))
    effective_min_step_scale = float(min(min_step_scale, TASK_FRAME_V45_LOW_VIS_MIN_STEP_SCALE if low_visibility else min_step_scale))
    step_scale_ok = bool(step_scale >= float(effective_min_step_scale))
    yaw_history_stable = _history_direction_stable(
        history_rows,
        field="task_frame_v45_dyaw",
        current_value=float(estimate.dyaw),
        required_support=TASK_FRAME_V45_MIN_HISTORY_SUPPORT,
    )
    z_allowed = bool(z_observable and force_safe and z_confidence_ok and step_scale_ok)
    yaw_allowed = bool(yaw_observable and not yaw_ambiguous and force_safe and yaw_confidence_ok and step_scale_ok and yaw_history_stable)
    z_block_reason = "ready" if z_allowed else (
        "z_not_observable" if not z_observable else
        "force_guard" if not force_safe else
        "z_low_confidence" if not z_confidence_ok else
        "z_step_scale_too_small" if not step_scale_ok else
        "z_blocked"
    )
    yaw_block_reason = "ready" if yaw_allowed else (
        "yaw_not_observable" if not yaw_observable else
        "yaw_ambiguous" if yaw_ambiguous else
        "force_guard" if not force_safe else
        "yaw_low_confidence" if not yaw_confidence_ok else
        "yaw_history_not_stable" if not yaw_history_stable else
        "yaw_step_scale_too_small" if not step_scale_ok else
        "yaw_blocked"
    )
    applied = bool(z_allowed or yaw_allowed)
    dz_step = float(np.clip(float(z_gain) * float(estimate.dz) * step_scale, -float(max_z_step), float(max_z_step))) if z_allowed else 0.0
    dyaw_step = float(np.clip(float(yaw_gain) * float(estimate.dyaw) * step_scale, -float(max_yaw_step), float(max_yaw_step))) if yaw_allowed else 0.0
    block_reason = "ready" if applied else (
        z_block_reason if z_block_reason != "ready" else yaw_block_reason
    )
    local_step = np.zeros(6, dtype=np.float32)
    local_step[2] = float(dz_step)
    local_step[5] = float(dyaw_step)
    decision = TaskFrameV45MicroServoDecision(
        applied=applied,
        xy_ready=bool(xy_ready),
        z_observable=bool(z_observable),
        yaw_observable=bool(yaw_observable),
        yaw_ambiguous=bool(yaw_ambiguous),
        yaw_history_stable=bool(yaw_history_stable),
        force_safe=bool(force_safe),
        z_allowed=bool(z_allowed),
        yaw_allowed=bool(yaw_allowed),
        z_block_reason=str(z_block_reason),
        yaw_block_reason=str(yaw_block_reason),
        block_reason=str(block_reason),
        dz_step=float(dz_step),
        dyaw_step=float(dyaw_step),
        step_scale=float(step_scale),
    )
    return estimate, decision, local_step


def task_frame_v45_candidate_labels_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    labels = _offline_labels(row)
    if not labels:
        return None
    dz = _as_float(labels.get("dz", row.get("privileged_dz", float("nan"))), float("nan"))
    dyaw = _as_float(labels.get("dyaw", row.get("privileged_dyaw", float("nan"))), float("nan"))
    if not np.isfinite(dz) or not np.isfinite(dyaw):
        return None
    z_ready = _as_bool(labels.get("z_ready", labels.get("z_ready_label", False)), False)
    yaw_ready = _as_bool(labels.get("yaw_ready", labels.get("yaw_ready_label", False)), False)
    yaw_ambiguous = _as_bool(labels.get("yaw_ambiguous", labels.get("yaw_ambiguous_label", False)), False)
    z_observable = _as_bool(labels.get("z_observable", labels.get("z_observable_label", False)), False)
    yaw_observable = _as_bool(labels.get("yaw_observable", labels.get("yaw_observable_label", False)), False)
    low_visibility = _as_bool(row.get("wrist_is_low_visibility"), False) or _as_bool(row.get("wrist_is_occluded"), False)
    risk_reason = _risk_reason_from_row(row)
    step_scale = _step_scale_target_from_row(row)
    z_confidence = _confidence_target_from_axis_error(
        axis_error=dz,
        axis_threshold=0.020,
        ready=z_ready,
        observable=z_observable,
        low_visibility=low_visibility,
    )
    yaw_confidence = _confidence_target_from_axis_error(
        axis_error=dyaw,
        axis_threshold=0.140,
        ready=yaw_ready,
        observable=yaw_observable,
        low_visibility=low_visibility,
    )
    return {
        "dz": float(dz),
        "dyaw": float(dyaw),
        "z_ready": float(bool(z_ready)),
        "z_confidence": float(z_confidence),
        "yaw_ready": float(bool(yaw_ready)),
        "yaw_confidence": float(yaw_confidence),
        "yaw_ambiguous": float(bool(yaw_ambiguous)),
        "step_scale": float(step_scale),
        "risk_reason": str(risk_reason),
    }


@dataclass(frozen=True)
class TaskFrameV45CandidateEstimate:
    dz: float
    dyaw: float
    z_ready: bool
    z_confidence: float
    yaw_ready: bool
    yaw_confidence: float
    yaw_ambiguous: bool
    step_scale: float
    risk_reason: str
    source: str = "task_frame_v45_candidate"
    uses_privileged_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dz": float(self.dz),
            "dyaw": float(self.dyaw),
            "z_ready": bool(self.z_ready),
            "z_confidence": float(self.z_confidence),
            "yaw_ready": bool(self.yaw_ready),
            "yaw_confidence": float(self.yaw_confidence),
            "yaw_ambiguous": bool(self.yaw_ambiguous),
            "step_scale": float(self.step_scale),
            "risk_reason": str(self.risk_reason),
            "v45_source": str(self.source),
            "uses_privileged_runtime": bool(self.uses_privileged_runtime),
        }


class TaskFrameV45CandidateNet(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = len(TASK_FRAME_READINESS_FEATURE_NAMES),
        hidden_dim: int = 96,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        max_abs_dz: float = 0.020,
        max_abs_dyaw: float = 0.140,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_abs_dz = float(max_abs_dz)
        self.max_abs_dyaw = float(max_abs_dyaw)
        mean = np.asarray(feature_mean, dtype=np.float32).reshape(-1) if feature_mean is not None else np.zeros((self.feature_dim,), dtype=np.float32)
        std = np.asarray(feature_std, dtype=np.float32).reshape(-1) if feature_std is not None else np.ones((self.feature_dim,), dtype=np.float32)
        if mean.size != self.feature_dim:
            mean = np.zeros((self.feature_dim,), dtype=np.float32)
        if std.size != self.feature_dim:
            std = np.ones((self.feature_dim,), dtype=np.float32)
        std = np.where(np.isfinite(std) & (np.abs(std) > 1.0e-6), std, 1.0).astype(np.float32)
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(std, dtype=torch.float32))
        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.dz_head = nn.Linear(hidden_dim, 1)
        self.dyaw_head = nn.Linear(hidden_dim, 1)
        self.z_ready_head = nn.Linear(hidden_dim, 1)
        self.z_confidence_head = nn.Linear(hidden_dim, 1)
        self.yaw_ready_head = nn.Linear(hidden_dim, 1)
        self.yaw_confidence_head = nn.Linear(hidden_dim, 1)
        self.yaw_ambiguous_head = nn.Linear(hidden_dim, 1)
        self.step_scale_head = nn.Linear(hidden_dim, 1)
        self.risk_reason_head = nn.Linear(hidden_dim, len(TASK_FRAME_V45_RISK_CLASSES))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        features = features.float()
        if self.feature_mean.numel() == self.feature_dim and self.feature_std.numel() == self.feature_dim:
            features = (features - self.feature_mean) / self.feature_std
        h = self.trunk(features)
        dz = torch.tanh(self.dz_head(h)[:, 0]) * self.max_abs_dz
        dyaw = torch.tanh(self.dyaw_head(h)[:, 0]) * self.max_abs_dyaw
        z_ready_logit = self.z_ready_head(h)[:, 0]
        z_conf_logit = self.z_confidence_head(h)[:, 0]
        yaw_ready_logit = self.yaw_ready_head(h)[:, 0]
        yaw_conf_logit = self.yaw_confidence_head(h)[:, 0]
        yaw_ambiguous_logit = self.yaw_ambiguous_head(h)[:, 0]
        step_scale = torch.sigmoid(self.step_scale_head(h)[:, 0])
        risk_logits = self.risk_reason_head(h)
        return {
            "dz": dz,
            "dyaw": dyaw,
            "z_ready_logit": z_ready_logit,
            "z_ready_probability": torch.sigmoid(z_ready_logit),
            "z_confidence_logit": z_conf_logit,
            "z_confidence_probability": torch.sigmoid(z_conf_logit),
            "yaw_ready_logit": yaw_ready_logit,
            "yaw_ready_probability": torch.sigmoid(yaw_ready_logit),
            "yaw_confidence_logit": yaw_conf_logit,
            "yaw_confidence_probability": torch.sigmoid(yaw_conf_logit),
            "yaw_ambiguous_logit": yaw_ambiguous_logit,
            "yaw_ambiguous_probability": torch.sigmoid(yaw_ambiguous_logit),
            "step_scale": step_scale,
            "risk_reason_logits": risk_logits,
            "risk_reason_probability": torch.softmax(risk_logits, dim=-1),
        }

    def predict_numpy(
        self,
        features: np.ndarray,
        *,
        ready_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        ambiguous_threshold: float = 0.5,
    ) -> TaskFrameV45CandidateEstimate:
        self.eval()
        feat = torch.as_tensor(np.asarray(features, dtype=np.float32).reshape(1, -1))
        with torch.no_grad():
            out = self.forward(feat)
        dz = float(out["dz"][0].item())
        dyaw = float(out["dyaw"][0].item())
        z_ready_prob = float(out["z_ready_probability"][0].item())
        z_conf = float(out["z_confidence_probability"][0].item())
        yaw_ready_prob = float(out["yaw_ready_probability"][0].item())
        yaw_conf = float(out["yaw_confidence_probability"][0].item())
        yaw_ambiguous_prob = float(out["yaw_ambiguous_probability"][0].item())
        step_scale = float(np.clip(float(out["step_scale"][0].item()), 0.0, 1.0))
        risk_idx = int(torch.argmax(out["risk_reason_probability"][0]).item())
        risk_reason = TASK_FRAME_V45_RISK_CLASSES[risk_idx]
        return TaskFrameV45CandidateEstimate(
            dz=dz,
            dyaw=dyaw,
            z_ready=bool(z_ready_prob >= float(ready_threshold)),
            z_confidence=float(z_conf),
            yaw_ready=bool(yaw_ready_prob >= float(ready_threshold)),
            yaw_confidence=float(yaw_conf),
            yaw_ambiguous=bool(yaw_ambiguous_prob >= float(ambiguous_threshold)),
            step_scale=float(step_scale),
            risk_reason=str(risk_reason),
        )


def save_task_frame_v45_candidate_checkpoint(
    path: str | Path,
    model: TaskFrameV45CandidateNet,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_names": list(TASK_FRAME_READINESS_FEATURE_NAMES),
        "feature_dim": int(model.feature_dim),
        "hidden_dim": int(model.hidden_dim),
        "max_abs_dz": float(model.max_abs_dz),
        "max_abs_dyaw": float(model.max_abs_dyaw),
        "feature_mean": model.feature_mean.detach().cpu().numpy().tolist() if hasattr(model, "feature_mean") else None,
        "feature_std": model.feature_std.detach().cpu().numpy().tolist() if hasattr(model, "feature_std") else None,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, str(path))


def load_task_frame_v45_candidate_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TaskFrameV45CandidateNet, dict[str, Any]]:
    payload = torch.load(str(path), map_location=map_location)
    model = TaskFrameV45CandidateNet(
        feature_dim=int(payload.get("feature_dim", len(TASK_FRAME_READINESS_FEATURE_NAMES))),
        hidden_dim=int(payload.get("hidden_dim", 96)),
        feature_mean=np.asarray(payload.get("feature_mean"), dtype=np.float32) if payload.get("feature_mean") is not None else None,
        feature_std=np.asarray(payload.get("feature_std"), dtype=np.float32) if payload.get("feature_std") is not None else None,
        max_abs_dz=float(payload.get("max_abs_dz", 0.020)),
        max_abs_dyaw=float(payload.get("max_abs_dyaw", 0.140)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, dict(payload.get("metadata", {}))
