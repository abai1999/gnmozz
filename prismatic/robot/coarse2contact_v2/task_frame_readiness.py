"""Non-privileged task-frame Z/Yaw readiness estimators for C2C v2.

The XY baseline stays frozen.  This module only models whether the task-frame
approach axis and yaw evidence are trustworthy enough to contribute to
alignment handoff.  It does not own close authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

TASK_FRAME_READINESS_FEATURE_NAMES: tuple[str, ...] = (
    "alignment_xy_ready",
    "xy_error_proxy",
    "runtime_xy_dx",
    "runtime_xy_dy",
    "runtime_xy_confidence",
    "runtime_xy_entry_ready",
    "runtime_xy_step_scale",
    "local_dx",
    "local_dy",
    "local_dz_proxy",
    "local_image_axis_yaw",
    "local_confidence",
    "local_observability",
    "local_fit_residual",
    "local_inlier_ratio",
    "estimated_proxy_dx",
    "estimated_proxy_dy",
    "estimated_proxy_dz",
    "estimated_proxy_dyaw",
    "frame_consistency",
    "wrist_valid_depth_ratio",
    "wrist_depth_near_fraction",
    "wrist_is_occluded",
    "wrist_is_low_visibility",
    "force_norm",
    "contact_confirmed",
    "planner_local_dx",
    "planner_local_dy",
    "planner_local_dz",
    "planner_local_dyaw",
    "planner_local_norm",
    "stage_precision_grasp",
    "stage_precision_align",
    "skill_precision_grasp",
    "skill_precision_align",
    "alias_stable_control",
    "alias_frame_drift",
    "alias_unknown",
    "visual_prior_only",
    "visual_partial_observable",
    "visual_visual_observable",
    "c2c_stage_age",
    "close_arbiter_guard_steps_remaining",
    "sticky_steps_remaining",
    "requires_yaw_observability",
)


def _safe_float(value: Any, default: float = 0.0, max_abs: float | None = None) -> float:
    try:
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        if max_abs is not None and np.isfinite(max_abs) and abs(out) > float(max_abs):
            return float(default)
        return out
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    return value if isinstance(value, Mapping) else {}


def _row_runtime_features(row: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime_features = row.get("runtime_features", {})
    return runtime_features if isinstance(runtime_features, Mapping) else {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _planner_local_norm(row: Mapping[str, Any]) -> float:
    runtime_features = _row_runtime_features(row)
    return float(
        np.linalg.norm(
            np.asarray(
                [
                    _safe_float(runtime_features.get("planner_local_dx"), 0.0),
                    _safe_float(runtime_features.get("planner_local_dy"), 0.0),
                    _safe_float(runtime_features.get("planner_local_dz"), 0.0),
                ],
                dtype=np.float32,
            )
        )
    )


def task_frame_readiness_feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    """Build a runtime-visible feature vector for Z/Yaw readiness heads."""

    runtime = _row_runtime_features(row)
    runtime_xy = _mapping(row, "runtime_xy_estimator")
    local_geometry = _mapping(row, "local_geometry_error")
    grasp = _mapping(local_geometry, "grasp")
    est = _mapping(row, "estimated_basin_error")
    planner = _mapping(row, "planner_prior")
    obs = _mapping(row, "obs_t")
    frame_contract = _mapping(row, "frame_contract")

    visual_class = str(obs.get("visual_observability_class", row.get("visual_observability_class", "")))
    alias_decision = str(row.get("alias_drift_decision", row.get("yaw_alias_drift_decision", "unknown")) or "unknown")
    stage_name = str(row.get("stage_name", row.get("c2c_v2_stage", "")))
    skill_type = str(row.get("skill_type", row.get("c2c_v2_skill_type", "")))
    c2c_stage_age = _safe_float(row.get("c2c_stage_age"), 0.0)

    values = {
        "alignment_xy_ready": 1.0 if _safe_bool(row.get("alignment_xy_ready", runtime_xy.get("entry_ready", runtime.get("runtime_xy_entry_ready", False))), False) else 0.0,
        "xy_error_proxy": _safe_float(
            row.get("grasp_probe_pre_xy_error", runtime.get("runtime_xy_estimator_residual_norm_xy", row.get("xy_error", 0.0))),
            0.0,
            max_abs=5.0,
        ),
        "runtime_xy_dx": _safe_float(runtime.get("runtime_xy_dx", runtime_xy.get("dx", 0.0)), 0.0, max_abs=5.0),
        "runtime_xy_dy": _safe_float(runtime.get("runtime_xy_dy", runtime_xy.get("dy", 0.0)), 0.0, max_abs=5.0),
        "runtime_xy_confidence": _safe_float(
            runtime.get("runtime_xy_confidence", runtime_xy.get("confidence", runtime.get("runtime_xy_estimator_confidence", 0.0))),
            0.0,
        ),
        "runtime_xy_entry_ready": 1.0 if _safe_bool(runtime.get("runtime_xy_entry_ready", runtime_xy.get("entry_ready", False)), False) else 0.0,
        "runtime_xy_step_scale": _safe_float(runtime.get("runtime_xy_step_scale", row.get("xy_step_scale", runtime_xy.get("xy_step_scale", 1.0))), 1.0),
        "local_dx": _safe_float(_first_value(grasp.get("dx"), runtime.get("local_dx")), 0.0, max_abs=5.0),
        "local_dy": _safe_float(_first_value(grasp.get("dy"), runtime.get("local_dy")), 0.0, max_abs=5.0),
        "local_dz_proxy": _safe_float(_first_value(grasp.get("dz"), runtime.get("local_dz_proxy")), 0.0, max_abs=5.0),
        "local_image_axis_yaw": _safe_float(_first_value(grasp.get("image_axis_yaw"), runtime.get("image_axis_yaw"), runtime.get("local_image_axis_yaw")), 0.0, max_abs=5.0),
        "local_confidence": _safe_float(_first_value(grasp.get("confidence"), runtime.get("local_confidence")), 0.0),
        "local_observability": _safe_float(_first_value(grasp.get("observability"), runtime.get("local_observability")), 0.0),
        "local_fit_residual": _safe_float(_first_value(grasp.get("fit_residual"), runtime.get("local_fit_residual")), 0.0),
        "local_inlier_ratio": _safe_float(_first_value(grasp.get("inlier_ratio"), runtime.get("local_inlier_ratio")), 0.0),
        "estimated_proxy_dx": _safe_float(est.get("estimated_basin_error_proxy_dx", est.get("proxy_dx")), 0.0, max_abs=5.0),
        "estimated_proxy_dy": _safe_float(est.get("estimated_basin_error_proxy_dy", est.get("proxy_dy")), 0.0, max_abs=5.0),
        "estimated_proxy_dz": _safe_float(est.get("estimated_basin_error_proxy_dz", est.get("proxy_dz")), 0.0, max_abs=5.0),
        "estimated_proxy_dyaw": _safe_float(est.get("estimated_basin_error_proxy_dyaw", est.get("proxy_dyaw")), 0.0, max_abs=5.0),
        "frame_consistency": _safe_float(est.get("estimated_basin_error_frame_consistency", est.get("frame_consistency")), 0.0),
        "wrist_valid_depth_ratio": _safe_float(_first_value(row.get("wrist_valid_depth_ratio"), runtime.get("wrist_valid_depth_ratio")), 0.0),
        "wrist_depth_near_fraction": _safe_float(_first_value(row.get("wrist_depth_near_fraction"), runtime.get("wrist_depth_near_fraction")), 0.0),
        "wrist_is_occluded": 1.0 if _safe_bool(_first_value(row.get("wrist_is_occluded"), runtime.get("wrist_is_occluded")), False) else 0.0,
        "wrist_is_low_visibility": 1.0 if _safe_bool(_first_value(row.get("wrist_is_low_visibility"), runtime.get("wrist_is_low_visibility")), False) else 0.0,
        "force_norm": _safe_float(runtime.get("force_norm", row.get("grasp_contact_rule_force_norm", 0.0)), 0.0),
        "contact_confirmed": 1.0 if _safe_bool(runtime.get("contact_confirmed", row.get("grasp_contact_rule_contact_confirmed", False)), False) else 0.0,
        "planner_local_dx": _safe_float(planner.get("local_delta_6d", row.get("planner_local_delta_6d", [0.0] * 6))[0] if isinstance(planner.get("local_delta_6d", None), (list, tuple, np.ndarray)) and len(planner.get("local_delta_6d", [])) >= 1 else _safe_float(runtime.get("planner_local_dx"), 0.0), 0.0, max_abs=5.0),
        "planner_local_dy": _safe_float(planner.get("local_delta_6d", row.get("planner_local_delta_6d", [0.0] * 6))[1] if isinstance(planner.get("local_delta_6d", None), (list, tuple, np.ndarray)) and len(planner.get("local_delta_6d", [])) >= 2 else _safe_float(runtime.get("planner_local_dy"), 0.0), 0.0, max_abs=5.0),
        "planner_local_dz": _safe_float(planner.get("local_delta_6d", row.get("planner_local_delta_6d", [0.0] * 6))[2] if isinstance(planner.get("local_delta_6d", None), (list, tuple, np.ndarray)) and len(planner.get("local_delta_6d", [])) >= 3 else _safe_float(runtime.get("planner_local_dz"), 0.0), 0.0, max_abs=5.0),
        "planner_local_dyaw": _safe_float(planner.get("local_delta_6d", row.get("planner_local_delta_6d", [0.0] * 6))[5] if isinstance(planner.get("local_delta_6d", None), (list, tuple, np.ndarray)) and len(planner.get("local_delta_6d", [])) >= 6 else _safe_float(runtime.get("planner_local_dyaw"), 0.0), 0.0, max_abs=5.0),
        "planner_local_norm": _planner_local_norm(row),
        "stage_precision_grasp": 1.0 if stage_name == "RING_GRASP_ALIGN" else 0.0,
        "stage_precision_align": 1.0 if stage_name == "RING_SPOKE_ALIGN" else 0.0,
        "skill_precision_grasp": 1.0 if skill_type == "precision_grasp" else 0.0,
        "skill_precision_align": 1.0 if skill_type == "precision_align" else 0.0,
        "alias_stable_control": 1.0 if alias_decision == "stable_alias_control" else 0.0,
        "alias_frame_drift": 1.0 if alias_decision == "frame_drift_abstain" else 0.0,
        "alias_unknown": 1.0 if alias_decision == "unknown" else 0.0,
        "visual_prior_only": 1.0 if visual_class == "prior_only" else 0.0,
        "visual_partial_observable": 1.0 if visual_class == "partial_observable" else 0.0,
        "visual_visual_observable": 1.0 if visual_class == "visual_observable" else 0.0,
        "c2c_stage_age": c2c_stage_age,
        "close_arbiter_guard_steps_remaining": _safe_float(row.get("grasp_probe_close_arbiter_guard_steps_remaining"), 0.0),
        "sticky_steps_remaining": _safe_float(row.get("grasp_probe_sticky_steps_remaining"), 0.0),
        "requires_yaw_observability": 1.0 if _safe_bool(frame_contract.get("requires_yaw_observability", row.get("requires_yaw_observability", False)), False) else 0.0,
    }
    out = np.asarray([values.get(str(name), 0.0) for name in TASK_FRAME_READINESS_FEATURE_NAMES], dtype=np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _safe_prob(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return float(np.clip(out, 0.0, 1.0))


def task_frame_z_label_from_row(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    labels = _mapping(row, "offline_labels")
    dz = _safe_float(labels.get("dz", row.get("privileged_dz", float("nan"))), float("nan"))
    z_observable = _safe_prob(labels.get("z_observable", labels.get("z_observable_label", 0.0)))
    z_near = _safe_prob(labels.get("z_near_alignment", labels.get("z_near_alignment_label", 0.0)))
    z_contact = _safe_prob(labels.get("z_contact_or_depth_ready", labels.get("z_contact_or_depth_ready_label", 0.0)))
    # Z readiness is intentionally independent from XY readiness.  The strict
    # planner handoff gate is formed later from XY + Z + Yaw + observability.
    z_ready = float(z_observable > 0.5 and z_near > 0.5 and z_contact > 0.5)
    valid = 1.0 if np.isfinite(dz) else 0.0
    return float(z_ready), float(z_observable), float(z_near), float(z_contact), float(valid)


def task_frame_yaw_label_from_row(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    labels = _mapping(row, "offline_labels")
    dyaw = _safe_float(labels.get("dyaw", row.get("privileged_dyaw", float("nan"))), float("nan"))
    yaw_observable = _safe_prob(labels.get("yaw_observable", labels.get("yaw_observable_label", 0.0)))
    yaw_ambiguous = _safe_prob(labels.get("yaw_ambiguous", labels.get("yaw_ambiguous_label", 0.0)))
    yaw_unobservable = _safe_prob(labels.get("yaw_unobservable", labels.get("yaw_unobservable_label", 0.0)))
    yaw_ready = float(yaw_observable > 0.5 and yaw_ambiguous < 0.5 and yaw_unobservable < 0.5)
    valid = 1.0 if np.isfinite(dyaw) else 0.0
    return float(yaw_ready), float(yaw_observable), float(yaw_ambiguous), float(yaw_unobservable), float(valid)


@dataclass(frozen=True)
class TaskFrameZReadinessEstimate:
    z_ready: bool
    z_observable: bool
    z_near_alignment: bool
    z_contact_or_depth_ready: bool
    z_confidence: float
    z_abstain_reason: str
    source: str = "task_frame_z_readiness"
    uses_privileged_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "z_ready": bool(self.z_ready),
            "z_observable": bool(self.z_observable),
            "z_near_alignment": bool(self.z_near_alignment),
            "z_contact_or_depth_ready": bool(self.z_contact_or_depth_ready),
            "z_confidence": float(self.z_confidence),
            "z_abstain_reason": str(self.z_abstain_reason),
            "z_readiness_source": str(self.source),
            "uses_privileged_runtime": bool(self.uses_privileged_runtime),
        }


@dataclass(frozen=True)
class TaskFrameYawReadinessEstimate:
    yaw_ready: bool
    yaw_observable: bool
    yaw_ambiguous: bool
    yaw_unobservable: bool
    yaw_confidence: float
    yaw_abstain_reason: str
    source: str = "task_frame_yaw_readiness"
    uses_privileged_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "yaw_ready": bool(self.yaw_ready),
            "yaw_observable": bool(self.yaw_observable),
            "yaw_ambiguous": bool(self.yaw_ambiguous),
            "yaw_unobservable": bool(self.yaw_unobservable),
            "yaw_confidence": float(self.yaw_confidence),
            "yaw_abstain_reason": str(self.yaw_abstain_reason),
            "yaw_readiness_source": str(self.source),
            "uses_privileged_runtime": bool(self.uses_privileged_runtime),
        }


class TaskFrameReadinessNet(nn.Module):
    """Small MLP that predicts task-frame readiness states from runtime-visible features."""

    def __init__(
        self,
        *,
        head_type: str,
        feature_dim: int = len(TASK_FRAME_READINESS_FEATURE_NAMES),
        hidden_dim: int = 96,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if head_type not in {"z", "yaw"}:
            raise ValueError(f"unsupported head_type: {head_type}")
        self.head_type = str(head_type)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
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
        if self.head_type == "z":
            self.z_ready_head = nn.Linear(hidden_dim, 1)
            self.z_observable_head = nn.Linear(hidden_dim, 1)
            self.z_near_head = nn.Linear(hidden_dim, 1)
            self.z_contact_ready_head = nn.Linear(hidden_dim, 1)
            self.z_confidence_head = nn.Linear(hidden_dim, 1)
        else:
            self.yaw_ready_head = nn.Linear(hidden_dim, 1)
            self.yaw_observable_head = nn.Linear(hidden_dim, 1)
            self.yaw_ambiguous_head = nn.Linear(hidden_dim, 1)
            self.yaw_confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        features = features.float()
        if self.feature_mean.numel() == self.feature_dim and self.feature_std.numel() == self.feature_dim:
            features = (features - self.feature_mean) / self.feature_std
        h = self.trunk(features)
        if self.head_type == "z":
            ready_logit = self.z_ready_head(h)[:, 0]
            observable_logit = self.z_observable_head(h)[:, 0]
            near_logit = self.z_near_head(h)[:, 0]
            contact_logit = self.z_contact_ready_head(h)[:, 0]
            conf_logit = self.z_confidence_head(h)[:, 0]
            return {
                "z_ready_logit": ready_logit,
                "z_ready_probability": torch.sigmoid(ready_logit),
                "z_observable_logit": observable_logit,
                "z_observable_probability": torch.sigmoid(observable_logit),
                "z_near_alignment_logit": near_logit,
                "z_near_alignment_probability": torch.sigmoid(near_logit),
                "z_contact_or_depth_ready_logit": contact_logit,
                "z_contact_or_depth_ready_probability": torch.sigmoid(contact_logit),
                "z_confidence_logit": conf_logit,
                "z_confidence_probability": torch.sigmoid(conf_logit),
            }
        ready_logit = self.yaw_ready_head(h)[:, 0]
        observable_logit = self.yaw_observable_head(h)[:, 0]
        ambiguous_logit = self.yaw_ambiguous_head(h)[:, 0]
        conf_logit = self.yaw_confidence_head(h)[:, 0]
        return {
            "yaw_ready_logit": ready_logit,
            "yaw_ready_probability": torch.sigmoid(ready_logit),
            "yaw_observable_logit": observable_logit,
            "yaw_observable_probability": torch.sigmoid(observable_logit),
            "yaw_ambiguous_logit": ambiguous_logit,
            "yaw_ambiguous_probability": torch.sigmoid(ambiguous_logit),
            "yaw_unobservable_probability": torch.clamp(1.0 - torch.sigmoid(observable_logit), 0.0, 1.0),
            "yaw_confidence_logit": conf_logit,
            "yaw_confidence_probability": torch.sigmoid(conf_logit),
        }

    def predict_numpy(
        self,
        features: np.ndarray,
        *,
        ready_threshold: float = 0.5,
        observable_threshold: float = 0.5,
        ambiguous_threshold: float = 0.5,
    ) -> TaskFrameZReadinessEstimate | TaskFrameYawReadinessEstimate:
        self.eval()
        feat = torch.as_tensor(np.asarray(features, dtype=np.float32).reshape(1, -1))
        with torch.no_grad():
            out = self.forward(feat)
        if self.head_type == "z":
            ready_prob = float(out["z_ready_probability"][0].item())
            observable_prob = float(out["z_observable_probability"][0].item())
            near_prob = float(out["z_near_alignment_probability"][0].item())
            contact_prob = float(out["z_contact_or_depth_ready_probability"][0].item())
            confidence = float(out["z_confidence_probability"][0].item())
            ready = bool(ready_prob >= float(ready_threshold))
            reason = "ready" if ready else (
                "z_not_observable" if observable_prob < float(observable_threshold) else "z_not_near_alignment" if near_prob < float(ready_threshold) else "z_not_contact_ready" if contact_prob < float(ready_threshold) else "z_low_confidence"
            )
            return TaskFrameZReadinessEstimate(
                z_ready=ready,
                z_observable=bool(observable_prob >= float(observable_threshold)),
                z_near_alignment=bool(near_prob >= float(ready_threshold)),
                z_contact_or_depth_ready=bool(contact_prob >= float(ready_threshold)),
                z_confidence=confidence,
                z_abstain_reason=str(reason),
            )
        ready_prob = float(out["yaw_ready_probability"][0].item())
        observable_prob = float(out["yaw_observable_probability"][0].item())
        ambiguous_prob = float(out["yaw_ambiguous_probability"][0].item())
        confidence = float(out["yaw_confidence_probability"][0].item())
        ready = bool(ready_prob >= float(ready_threshold))
        yaw_observable = bool(observable_prob >= float(observable_threshold))
        yaw_ambiguous = bool(ambiguous_prob >= float(ambiguous_threshold))
        yaw_unobservable = bool(not yaw_observable and not yaw_ambiguous)
        reason = "ready" if ready else (
            "yaw_unobservable" if yaw_unobservable else "yaw_ambiguous" if yaw_ambiguous else "yaw_not_ready"
        )
        return TaskFrameYawReadinessEstimate(
            yaw_ready=ready,
            yaw_observable=yaw_observable,
            yaw_ambiguous=yaw_ambiguous,
            yaw_unobservable=yaw_unobservable,
            yaw_confidence=confidence,
            yaw_abstain_reason=str(reason),
        )


def save_task_frame_readiness_checkpoint(
    path: str | Path,
    model: TaskFrameReadinessNet,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_names": list(TASK_FRAME_READINESS_FEATURE_NAMES),
        "feature_dim": int(model.feature_dim),
        "hidden_dim": int(model.hidden_dim),
        "head_type": str(model.head_type),
        "feature_mean": model.feature_mean.detach().cpu().numpy().tolist() if hasattr(model, "feature_mean") else None,
        "feature_std": model.feature_std.detach().cpu().numpy().tolist() if hasattr(model, "feature_std") else None,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, str(path))


def load_task_frame_readiness_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TaskFrameReadinessNet, dict[str, Any]]:
    payload = torch.load(str(path), map_location=map_location)
    model = TaskFrameReadinessNet(
        head_type=str(payload.get("head_type", "z")),
        feature_dim=int(payload.get("feature_dim", len(TASK_FRAME_READINESS_FEATURE_NAMES))),
        hidden_dim=int(payload.get("hidden_dim", 96)),
        feature_mean=np.asarray(payload.get("feature_mean"), dtype=np.float32) if payload.get("feature_mean") is not None else None,
        feature_std=np.asarray(payload.get("feature_std"), dtype=np.float32) if payload.get("feature_std") is not None else None,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, dict(payload.get("metadata", {}))
