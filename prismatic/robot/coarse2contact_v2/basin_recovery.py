"""Basin-oriented recovery skills for Coarse2Contact v2.

This module deliberately separates recovery *state and mode selection* from
single-step residual prediction.  Learned heads may provide proposals, but the
supervisor owns observability gating, bounded visual pullback, and basin
verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .recovery_audit import (
    apply_closed_loop_recovery_step,
    in_close_ready_basin,
    in_near_grasp_basin,
    recovery_error_norm,
    recovery_overshoot_flag,
)
from .basin_state import EstimatedBasinError


class BasinRecoveryMode(str, Enum):
    IDLE = "IDLE"
    REACQUIRE_VIEW = "REACQUIRE_VIEW"
    VISUAL_PULLBACK = "VISUAL_PULLBACK"
    MICRO_SERVO_TO_BASIN = "MICRO_SERVO_TO_BASIN"
    VERIFY_BASIN = "VERIFY_BASIN"
    ABSTAIN_FAIL = "ABSTAIN_FAIL"


class VisualEvidenceClass(str, Enum):
    VISUAL_OBSERVABLE = "visual_observable"
    PARTIAL_OBSERVABLE = "partial_observable"
    PRIOR_ONLY = "prior_only"


class BasinLabel(str, Enum):
    OUTSIDE = "outside"
    NEAR_GRASP = "near_grasp"
    CLOSE_READY = "close_ready"
    NEAR_INSERT = "near_insert"


@dataclass(frozen=True)
class BasinRecoveryConfig:
    variant_name: str = "xy_plus_yaw"
    yaw_control_enabled: bool = True
    near_grasp_xy_threshold: float = 0.015
    near_grasp_yaw_threshold: float = 0.08
    close_ready_xy_threshold: float = 0.005
    close_ready_z_threshold: float = 0.010
    close_ready_yaw_threshold: float = 0.03
    visual_conf_threshold: float = 0.01
    visual_observability_threshold: float = 0.002
    visual_axis_strength_threshold: float = 1.0e-5
    partial_conf_threshold: float = 1.0e-3
    partial_observability_threshold: float = 5.0e-4
    partial_axis_strength_threshold: float = 1.0e-6
    visual_servo_min_confidence: float = 0.01
    learned_proposal_min_confidence: float = 0.65
    visual_gain: float = 0.35
    micro_gain: float = 0.20
    max_pullback_xy_step: float = 0.0030
    max_pullback_z_step: float = 0.0012
    max_pullback_yaw_step: float = 0.050
    max_micro_xy_step: float = 0.0006
    max_micro_z_step: float = 0.0006
    max_micro_yaw_step: float = 0.015
    micro_entry_xy_threshold: float = 0.0060
    micro_entry_z_threshold: float = 0.0120
    micro_entry_required_frames: int = 2
    micro_yaw_activation_threshold: float = 0.18
    reacquire_lift_step: float = 0.0040
    reacquire_lateral_step: float = 0.0015
    reacquire_yaw_step: float = 0.015
    verify_required_frames: int = 3
    max_recovery_steps: int = 24
    eval_line_search_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)


@dataclass
class BasinRecoveryDecision:
    mode: BasinRecoveryMode
    visual_evidence_class: VisualEvidenceClass
    basin_label: BasinLabel
    correction_xyyaw: np.ndarray
    local_action_6d: np.ndarray
    confidence: float
    reason: str
    correction_dz: float = 0.0
    stable_basin_count: int = 0
    recovery_step: int = 0
    line_search_scale: float = 1.0
    variant_name: str = ""
    micro_entry_ready: bool = False
    micro_yaw_active: bool = False
    micro_entry_xy_error: float = float("nan")
    micro_entry_dyaw: float = float("nan")
    used_learned_proposal: bool = False
    used_visual_geometry: bool = False
    dry_run_scaled_for_eval: bool = False
    uses_privileged_target: bool = False
    uses_privileged_label_for_eval: bool = False

    def to_trace(self, *, prefix: str = "basin_recovery") -> dict[str, Any]:
        corr = np.asarray(self.correction_xyyaw, dtype=np.float32).reshape(3)
        action = np.asarray(self.local_action_6d, dtype=np.float32).reshape(6)
        return {
            f"{prefix}_mode": self.mode.value,
            f"{prefix}_variant": self.variant_name,
            f"{prefix}_visual_evidence_class": self.visual_evidence_class.value,
            f"{prefix}_basin_label": self.basin_label.value,
            f"{prefix}_correction_dx": float(corr[0]),
            f"{prefix}_correction_dy": float(corr[1]),
            f"{prefix}_correction_dyaw": float(corr[2]),
            f"{prefix}_correction_dz": float(self.correction_dz),
            f"{prefix}_local_action_6d": [float(x) for x in action.tolist()],
            f"{prefix}_confidence": float(self.confidence),
            f"{prefix}_reason": self.reason,
            f"{prefix}_stable_basin_count": int(self.stable_basin_count),
            f"{prefix}_recovery_step": int(self.recovery_step),
            f"{prefix}_line_search_scale": float(self.line_search_scale),
            f"{prefix}_micro_entry_ready": bool(self.micro_entry_ready),
            f"{prefix}_micro_yaw_active": bool(self.micro_yaw_active),
            f"{prefix}_micro_entry_xy_error": float(self.micro_entry_xy_error),
            f"{prefix}_micro_entry_dyaw": float(self.micro_entry_dyaw),
            f"{prefix}_used_learned_proposal": bool(self.used_learned_proposal),
            f"{prefix}_used_visual_geometry": bool(self.used_visual_geometry),
            f"{prefix}_dry_run_scaled_for_eval": bool(self.dry_run_scaled_for_eval),
            "uses_privileged_target": bool(self.uses_privileged_target),
            "uses_privileged_label_for_eval": bool(self.uses_privileged_label_for_eval),
        }


def _float(record: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key, default))
    except Exception:
        return float(default)


def _wide_ring_error_from_record(record: Mapping[str, Any]) -> np.ndarray | None:
    visible = bool(record.get("wide_ring_visible", False))
    if not visible:
        return None
    dx = _float(record, "wide_ring_dx", float("nan"))
    dy = _float(record, "wide_ring_dy", float("nan"))
    if not np.isfinite(dx) or not np.isfinite(dy):
        return None
    return np.asarray([dx, dy, 0.0], dtype=np.float32)


def classify_basin_label(
    error_state: Iterable[float],
    *,
    config: BasinRecoveryConfig | None = None,
) -> BasinLabel:
    cfg = config or BasinRecoveryConfig()
    err = np.asarray(list(error_state), dtype=np.float32).reshape(-1)
    err = np.pad(err, (0, max(0, 3 - err.size)))[:3]
    if in_close_ready_basin(
        float(err[0]),
        float(err[1]),
        float(err[2]),
        xy_threshold=cfg.close_ready_xy_threshold,
        yaw_threshold=cfg.close_ready_yaw_threshold,
    ):
        return BasinLabel.CLOSE_READY
    if in_near_grasp_basin(
        float(err[0]),
        float(err[1]),
        float(err[2]),
        xy_threshold=cfg.near_grasp_xy_threshold,
        yaw_threshold=cfg.near_grasp_yaw_threshold,
    ):
        return BasinLabel.NEAR_GRASP
    return BasinLabel.OUTSIDE


def classify_visual_evidence_for_basin(
    record: Mapping[str, Any],
    *,
    config: BasinRecoveryConfig | None = None,
) -> VisualEvidenceClass:
    cfg = config or BasinRecoveryConfig()
    conf = _float(record, "frame_confidence")
    obs = _float(record, "frame_observability")
    axis = _float(record, "frame_axis_strength")
    if conf >= cfg.visual_conf_threshold and obs >= cfg.visual_observability_threshold and axis >= cfg.visual_axis_strength_threshold:
        return VisualEvidenceClass.VISUAL_OBSERVABLE
    if bool(record.get("wide_ring_visible", False)) and obs >= cfg.partial_observability_threshold:
        return VisualEvidenceClass.PARTIAL_OBSERVABLE
    if conf >= cfg.partial_conf_threshold and obs >= cfg.partial_observability_threshold and axis >= cfg.partial_axis_strength_threshold:
        return VisualEvidenceClass.PARTIAL_OBSERVABLE
    return VisualEvidenceClass.PRIOR_ONLY


def visual_error_from_record(record: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            _float(record, "trace_error_dx"),
            _float(record, "trace_error_dy"),
            _float(record, "trace_error_dyaw"),
        ],
        dtype=np.float32,
    )


def target_error_from_record(record: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            _float(record, "recovery_target_dx", _float(record, "trace_error_dx")),
            _float(record, "recovery_target_dy", _float(record, "trace_error_dy")),
            _float(record, "recovery_target_dyaw", _float(record, "trace_error_dyaw")),
        ],
        dtype=np.float32,
    )


def planner_prior_from_record(record: Mapping[str, Any]) -> np.ndarray:
    vals = list(record.get("planner_prior_delta", []) or [])
    return np.asarray((vals + [0.0] * 6)[:6], dtype=np.float32)


def _clamp_xyyaw(correction: Iterable[float], *, max_xy: float, max_yaw: float) -> np.ndarray:
    corr = np.asarray(list(correction), dtype=np.float32).reshape(-1)
    corr = np.pad(corr, (0, max(0, 3 - corr.size)))[:3]
    xy = corr[:2]
    norm = float(np.linalg.norm(xy))
    if norm > float(max_xy) > 0.0:
        xy = xy * (float(max_xy) / max(norm, 1.0e-9))
    corr[:2] = xy
    corr[2] = float(np.clip(corr[2], -float(max_yaw), float(max_yaw)))
    return corr.astype(np.float32)


def _clamp_basin_correction(
    correction: Iterable[float],
    *,
    max_xy: float,
    max_z: float,
    max_yaw: float,
) -> np.ndarray:
    corr = np.asarray(list(correction), dtype=np.float32).reshape(-1)
    corr = np.pad(corr, (0, max(0, 4 - corr.size)))[:4]
    xy = corr[:2]
    norm = float(np.linalg.norm(xy))
    if norm > float(max_xy) > 0.0:
        xy = xy * (float(max_xy) / max(norm, 1.0e-9))
    corr[:2] = xy
    corr[2] = float(np.clip(corr[2], -float(max_z), float(max_z)))
    corr[3] = float(np.clip(corr[3], -float(max_yaw), float(max_yaw)))
    return corr.astype(np.float32)


@dataclass
class GraspOnlyBasinPullbackPolicy:
    """Minimal grasp-only pullback policy for privileged replay/audit.

    The policy is intentionally conservative: it only acts on x/y by default,
    abstains or reacquires under prior-only visibility, and keeps z as a
    diagnostic channel unless explicitly enabled.
    """

    config: BasinRecoveryConfig = field(default_factory=BasinRecoveryConfig)
    xy_gain: float = 0.35
    yaw_gain: float = 0.0
    z_gain: float = 0.0
    yaw_observable_min_abs: float = 0.01
    allow_z: bool = False
    allow_yaw: bool = False

    @staticmethod
    def _as_estimated(estimated_basin_error: EstimatedBasinError | Mapping[str, Any] | None) -> EstimatedBasinError | None:
        if estimated_basin_error is None:
            return None
        if isinstance(estimated_basin_error, EstimatedBasinError):
            return estimated_basin_error
        try:
            return EstimatedBasinError.from_dict(estimated_basin_error)
        except Exception:
            return None

    def step(
        self,
        *,
        estimated_basin_error: EstimatedBasinError | Mapping[str, Any] | None,
        visual_evidence_class: VisualEvidenceClass | str = VisualEvidenceClass.PRIOR_ONLY,
        stage_name: str = "",
    ) -> BasinRecoveryDecision:
        est = self._as_estimated(estimated_basin_error)
        visual_class = visual_evidence_class.value if isinstance(visual_evidence_class, VisualEvidenceClass) else str(visual_evidence_class)
        if est is None or visual_class == VisualEvidenceClass.PRIOR_ONLY.value or not est.valid:
            local_action = np.zeros(6, dtype=np.float32)
            local_action[2] = -float(self.config.reacquire_lift_step)
            return BasinRecoveryDecision(
                mode=BasinRecoveryMode.REACQUIRE_VIEW,
                visual_evidence_class=VisualEvidenceClass.PRIOR_ONLY if visual_class == VisualEvidenceClass.PRIOR_ONLY.value else VisualEvidenceClass.PARTIAL_OBSERVABLE,
                basin_label=BasinLabel.OUTSIDE,
                correction_xyyaw=np.zeros(3, dtype=np.float32),
                local_action_6d=local_action,
                confidence=0.0,
                reason="prior_only_reacquire" if visual_class == VisualEvidenceClass.PRIOR_ONLY.value else "replay_abstain",
                stable_basin_count=0,
                recovery_step=0,
                variant_name=self.config.variant_name,
                micro_entry_ready=False,
                micro_yaw_active=False,
                micro_entry_xy_error=float("nan"),
                micro_entry_dyaw=float("nan"),
                used_learned_proposal=False,
                used_visual_geometry=False,
                dry_run_scaled_for_eval=False,
                uses_privileged_target=False,
                uses_privileged_label_for_eval=True,
            )

        correction = np.zeros(6, dtype=np.float32)
        if est.x_valid:
            correction[0] = float(self.xy_gain * est.dx)
        if est.y_valid:
            correction[1] = float(self.xy_gain * est.dy)
        if self.allow_z and est.z_valid:
            correction[2] = float(self.z_gain * est.dz)
        if self.allow_yaw and est.yaw_valid and abs(float(est.dyaw)) >= float(self.yaw_observable_min_abs):
            correction[5] = float(self.yaw_gain * est.dyaw)

        max_xy = self.config.max_micro_xy_step if est.close_ready(
            xy_threshold=self.config.close_ready_xy_threshold,
            z_threshold=self.config.close_ready_z_threshold,
            yaw_threshold=self.config.close_ready_yaw_threshold,
            yaw_required=False,
        ) else self.config.max_pullback_xy_step
        max_z = self.config.max_micro_z_step if self.allow_z else 0.0
        max_yaw = self.config.max_micro_yaw_step if self.allow_yaw else 0.0
        correction = _clamp_basin_correction(
            np.asarray([correction[0], correction[1], correction[2], correction[5]], dtype=np.float32),
            max_xy=max_xy,
            max_z=max_z,
            max_yaw=max_yaw,
        )
        micro_ready = bool(
            est.close_ready(
                xy_threshold=self.config.close_ready_xy_threshold,
                z_threshold=self.config.close_ready_z_threshold,
                yaw_threshold=self.config.close_ready_yaw_threshold,
                yaw_required=False,
            )
        )
        mode = BasinRecoveryMode.MICRO_SERVO_TO_BASIN if micro_ready else BasinRecoveryMode.VISUAL_PULLBACK
        if self.allow_yaw and abs(float(correction[3])) > 1.0e-9:
            mode = BasinRecoveryMode.MICRO_SERVO_TO_BASIN
        if visual_class in VisualEvidenceClass._value2member_map_:
            visual_enum = VisualEvidenceClass(visual_class)
        else:
            visual_enum = VisualEvidenceClass.VISUAL_OBSERVABLE
        local_action = np.zeros(6, dtype=np.float32)
        local_action[0] = float(correction[0])
        local_action[1] = float(correction[1])
        local_action[2] = float(correction[2])
        local_action[5] = float(correction[3])
        return BasinRecoveryDecision(
            mode=mode,
            visual_evidence_class=visual_enum,
            basin_label=BasinLabel.CLOSE_READY if mode == BasinRecoveryMode.MICRO_SERVO_TO_BASIN else BasinLabel.NEAR_GRASP,
            correction_xyyaw=np.asarray([correction[0], correction[1], correction[3]], dtype=np.float32),
            local_action_6d=local_action,
            confidence=float(np.clip(est.confidence, 0.0, 1.0)),
            reason="grasp_only_replay_pullback",
            stable_basin_count=0,
            recovery_step=0,
            variant_name=self.config.variant_name,
            micro_entry_ready=bool(micro_ready),
            micro_yaw_active=bool(self.allow_yaw and abs(float(correction[3])) > 1.0e-9),
            micro_entry_xy_error=float(np.hypot(float(est.dx), float(est.dy))),
            micro_entry_dyaw=float(est.dyaw),
            used_learned_proposal=False,
            used_visual_geometry=True,
            dry_run_scaled_for_eval=True,
            uses_privileged_target=False,
            uses_privileged_label_for_eval=True,
        )


class BasinRecoverySupervisor:
    """Mode-aware recovery controller for grasp-basin pullback.

    The controller is intentionally conservative.  In weak-visual-evidence
    states it refuses to invent a target correction and only emits a
    reacquire-view primitive.  In visual states it uses explicit local geometry
    if present, with learned residuals treated as proposals rather than owners.
    """

    def __init__(self, config: BasinRecoveryConfig | None = None) -> None:
        self.config = config or BasinRecoveryConfig()
        self.reset()

    def reset(self) -> None:
        self.mode = BasinRecoveryMode.IDLE
        self.recovery_step = 0
        self.stable_basin_count = 0
        self.stable_xy_count = 0
        self.cycle_id = 0

    def _reacquire_action(self) -> np.ndarray:
        cfg = self.config
        phase = self.recovery_step % 6
        action = np.zeros(6, dtype=np.float32)
        # The tool z-axis points roughly downward in this task family, so
        # retreating from the workspace means moving along negative local z.
        action[2] = -cfg.reacquire_lift_step
        if phase == 1:
            action[0] = cfg.reacquire_lateral_step
        elif phase == 2:
            action[0] = -cfg.reacquire_lateral_step
        elif phase == 3:
            action[1] = cfg.reacquire_lateral_step
        elif phase == 4:
            action[1] = -cfg.reacquire_lateral_step
        elif phase == 5:
            action[5] = cfg.reacquire_yaw_step
        return action

    def _proposal_from_inputs(
        self,
        *,
        estimated_basin_error: EstimatedBasinError | None,
        visual_error_state: Iterable[float] | None,
        model_prediction: Mapping[str, float] | None,
        basin_label: BasinLabel,
    ) -> tuple[np.ndarray, float, bool, bool, str]:
        cfg = self.config
        used_visual = False
        used_model = False
        reason = "no_reliable_proposal"
        confidence = 0.0

        if estimated_basin_error is not None:
            est = estimated_basin_error
            proposal = np.zeros(4, dtype=np.float32)
            if est.x_valid:
                proposal[0] = float(est.dx)
            if est.y_valid:
                proposal[1] = float(est.dy)
            if est.z_valid:
                proposal[2] = float(est.dz)
            if est.yaw_valid and cfg.yaw_control_enabled:
                proposal[3] = float(est.dyaw)
            if est.valid and np.any(np.asarray([est.x_valid, est.y_valid, est.z_valid, est.yaw_valid], dtype=bool)):
                confidence = float(np.clip(est.confidence, 0.0, 1.0))
                used_visual = True
                reason = "estimated_basin_geometry_pullback"
                return proposal.astype(np.float32), confidence, used_visual, used_model, reason
            return np.zeros(4, dtype=np.float32), 0.0, False, False, "estimated_basin_abstain"

        if visual_error_state is not None:
            visual = np.asarray(list(visual_error_state), dtype=np.float32).reshape(-1)
            if visual.size >= 4:
                visual4 = np.asarray(visual[:4], dtype=np.float32)
            else:
                visual = np.pad(visual, (0, max(0, 3 - visual.size)))[:3]
                visual4 = np.asarray([visual[0], visual[1], 0.0, visual[2]], dtype=np.float32)
            if np.all(np.isfinite(visual4)):
                gain = cfg.micro_gain if basin_label in {BasinLabel.NEAR_GRASP, BasinLabel.CLOSE_READY} else cfg.visual_gain
                confidence = max(confidence, 1.0)
                used_visual = True
                reason = "visual_geometry_pullback"
                proposal = (gain * visual4).astype(np.float32)
                proposal[3] = proposal[3] if cfg.yaw_control_enabled else 0.0
                return proposal, confidence, used_visual, used_model, reason

        if model_prediction is not None:
            conf = float(model_prediction.get("confidence", 0.0))
            if conf >= cfg.learned_proposal_min_confidence:
                pred = np.asarray(
                    [
                        float(model_prediction.get("dx", 0.0)),
                        float(model_prediction.get("dy", 0.0)),
                        float(model_prediction.get("dz", 0.0)),
                        float(model_prediction.get("dyaw", 0.0)),
                    ],
                    dtype=np.float32,
                )
                used_model = True
                confidence = max(confidence, conf)
                reason = "learned_pullback_proposal"
                return pred, confidence, used_visual, used_model, reason

        return np.zeros(3, dtype=np.float32), confidence, used_visual, used_model, reason

    def _eval_line_search(
        self,
        *,
        pre_error_state: Iterable[float],
        planner_prior_state: Iterable[float],
        proposal: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        cfg = self.config
        pre = np.asarray(list(pre_error_state), dtype=np.float32).reshape(-1)
        pre = np.pad(pre, (0, max(0, 3 - pre.size)))[:3]
        prior = np.asarray(list(planner_prior_state), dtype=np.float32).reshape(-1)
        prior = np.pad(prior, (0, max(0, 6 - prior.size)))[:6]
        best_scale = 0.0
        best_corr = np.zeros(3, dtype=np.float32)
        best_norm = recovery_error_norm(float(pre[0]), float(pre[1]), float(pre[2]))
        for scale in cfg.eval_line_search_scales:
            corr = (float(scale) * np.asarray(proposal, dtype=np.float32).reshape(-1)[:3]).astype(np.float32)
            post, _ = apply_closed_loop_recovery_step(pre, prior, corr)
            post_norm = recovery_error_norm(float(post[0]), float(post[1]), float(post[2]))
            if recovery_overshoot_flag(pre, post) and post_norm >= best_norm:
                continue
            if post_norm <= best_norm + 1.0e-9:
                best_norm = float(post_norm)
                best_scale = float(scale)
                best_corr = corr
        return best_corr.astype(np.float32), best_scale

    def step(
        self,
        *,
        record: Mapping[str, Any] | None = None,
        planner_prior_state: Iterable[float] | None = None,
        estimated_basin_error: EstimatedBasinError | Mapping[str, Any] | None = None,
        visual_error_state: Iterable[float] | None = None,
        model_prediction: Mapping[str, float] | None = None,
        target_error_state_for_eval: Iterable[float] | None = None,
        allow_eval_line_search: bool = False,
    ) -> BasinRecoveryDecision:
        cfg = self.config
        rec = record or {}
        evidence = classify_visual_evidence_for_basin(rec, config=cfg)
        est = None
        if estimated_basin_error is not None:
            if isinstance(estimated_basin_error, EstimatedBasinError):
                est = estimated_basin_error
            else:
                try:
                    est = EstimatedBasinError.from_dict(estimated_basin_error)
                except Exception:
                    est = None
        if est is None and visual_error_state is not None:
            visual = np.asarray(list(visual_error_state), dtype=np.float32).reshape(-1)
            if visual.size >= 4:
                dx_v, dy_v, dz_v, dyaw_v = [float(v) for v in visual[:4]]
            else:
                visual = np.pad(visual, (0, max(0, 3 - visual.size)))[:3]
                dx_v, dy_v, dyaw_v = [float(v) for v in visual[:3]]
                dz_v = 0.0
            est = EstimatedBasinError(
                valid=bool(np.all(np.isfinite([dx_v, dy_v, dz_v, dyaw_v]))),
                confidence=float(evidence != VisualEvidenceClass.PRIOR_ONLY),
                dx=dx_v,
                dy=dy_v,
                dz=dz_v,
                dyaw=dyaw_v,
                x_valid=bool(np.isfinite(dx_v)),
                y_valid=bool(np.isfinite(dy_v)),
                z_valid=bool(np.isfinite(dz_v)),
                yaw_valid=bool(cfg.yaw_control_enabled and np.isfinite(dyaw_v) and abs(float(dyaw_v)) <= cfg.micro_yaw_activation_threshold),
                x_confidence=float(max(0.0, evidence != VisualEvidenceClass.PRIOR_ONLY)),
                y_confidence=float(max(0.0, evidence != VisualEvidenceClass.PRIOR_ONLY)),
                z_confidence=float(max(0.0, evidence != VisualEvidenceClass.PRIOR_ONLY)),
                yaw_confidence=float(max(0.0, evidence != VisualEvidenceClass.PRIOR_ONLY) * 0.25),
                frame_consistency=float(max(0.0, evidence != VisualEvidenceClass.PRIOR_ONLY)),
                source="proxy_fallback",
                reason="legacy_proxy_fallback",
                proxy_dx=dx_v,
                proxy_dy=dy_v,
                proxy_dz=dz_v,
                proxy_dyaw=dyaw_v,
            )
        eval_error = target_error_state_for_eval if target_error_state_for_eval is not None else (
            est.to_vector() if est is not None else visual_error_state
        )
        basin_input: Iterable[float]
        if eval_error is None:
            basin_input = [999.0, 999.0, 999.0]
        else:
            eval_arr_for_basin = np.asarray(list(eval_error), dtype=np.float32).reshape(-1)
            eval_arr_for_basin = np.pad(eval_arr_for_basin, (0, max(0, 4 - eval_arr_for_basin.size)))[:4]
            basin_input = [float(eval_arr_for_basin[0]), float(eval_arr_for_basin[1]), float(eval_arr_for_basin[3])]
        basin = classify_basin_label(basin_input, config=cfg)
        xy_error = float("inf")
        z_error = float("inf")
        dyaw_error = float("nan")
        if eval_error is not None:
            eval_arr = np.asarray(list(eval_error), dtype=np.float32).reshape(-1)
            eval_arr = np.pad(eval_arr, (0, max(0, 4 - eval_arr.size)))[:4]
            if np.all(np.isfinite(eval_arr[:2])):
                xy_error = float(np.hypot(float(eval_arr[0]), float(eval_arr[1])))
            if est is not None and np.isfinite(est.dz):
                z_error = float(abs(est.dz))
            if np.isfinite(eval_arr[3]):
                dyaw_error = float(eval_arr[3])
        if planner_prior_state is None:
            prior_values = [0.0] * 6
        else:
            prior_values = list(planner_prior_state)
        prior = np.asarray(prior_values, dtype=np.float32)
        prior = np.pad(prior, (0, max(0, 6 - prior.size)))[:6]

        if self.recovery_step >= cfg.max_recovery_steps:
            self.mode = BasinRecoveryMode.ABSTAIN_FAIL
            decision = BasinRecoveryDecision(
                mode=self.mode,
                visual_evidence_class=evidence,
                basin_label=basin,
                correction_xyyaw=np.zeros(3, dtype=np.float32),
                correction_dz=0.0,
                local_action_6d=np.zeros(6, dtype=np.float32),
                confidence=0.0,
                reason="recovery_budget_exhausted",
                stable_basin_count=self.stable_basin_count,
                recovery_step=self.recovery_step,
                variant_name=cfg.variant_name,
            )
            self.recovery_step += 1
            return decision

        wide_error = _wide_ring_error_from_record(rec)

        if evidence == VisualEvidenceClass.PRIOR_ONLY:
            self.mode = BasinRecoveryMode.REACQUIRE_VIEW
            self.stable_basin_count = 0
            decision = BasinRecoveryDecision(
                mode=self.mode,
                visual_evidence_class=evidence,
                basin_label=basin,
                correction_xyyaw=np.zeros(3, dtype=np.float32),
                correction_dz=0.0,
                local_action_6d=self._reacquire_action(),
                confidence=0.0,
                reason="target_not_observable_reacquire_view",
                stable_basin_count=self.stable_basin_count,
                recovery_step=self.recovery_step,
                variant_name=cfg.variant_name,
            )
            self.recovery_step += 1
            return decision

        if basin in {BasinLabel.NEAR_GRASP, BasinLabel.CLOSE_READY}:
            self.stable_basin_count += 1
            self.stable_xy_count += 1
            self.mode = BasinRecoveryMode.VERIFY_BASIN if self.stable_basin_count >= cfg.verify_required_frames else BasinRecoveryMode.MICRO_SERVO_TO_BASIN
        else:
            self.stable_basin_count = 0
            micro_entry_ready = bool(
                evidence == VisualEvidenceClass.VISUAL_OBSERVABLE
                and est is not None
                and est.valid
                and np.isfinite(xy_error)
                and xy_error <= float(cfg.micro_entry_xy_threshold)
                and (not np.isfinite(est.dz) or abs(float(est.dz)) <= float(cfg.micro_entry_z_threshold))
            )
            if micro_entry_ready:
                self.stable_xy_count += 1
            else:
                self.stable_xy_count = 0
            if self.stable_xy_count >= max(int(cfg.micro_entry_required_frames), 1):
                self.mode = BasinRecoveryMode.MICRO_SERVO_TO_BASIN
            else:
                self.mode = BasinRecoveryMode.VISUAL_PULLBACK if evidence == VisualEvidenceClass.VISUAL_OBSERVABLE else BasinRecoveryMode.REACQUIRE_VIEW

        if self.mode == BasinRecoveryMode.REACQUIRE_VIEW:
            local_action = self._reacquire_action()
            if wide_error is not None:
                center_corr = _clamp_xyyaw(-0.35 * wide_error, max_xy=0.75 * cfg.max_pullback_xy_step, max_yaw=0.0)
                local_action[0] += float(center_corr[0])
                local_action[1] += float(center_corr[1])
            decision = BasinRecoveryDecision(
                mode=self.mode,
                visual_evidence_class=evidence,
                basin_label=basin,
                correction_xyyaw=np.zeros(3, dtype=np.float32),
                correction_dz=float(local_action[2]),
                local_action_6d=local_action,
                confidence=0.0,
                reason="partial_observability_reacquire_before_control" if wide_error is None else "partial_observability_center_and_reacquire",
                stable_basin_count=self.stable_basin_count,
                recovery_step=self.recovery_step,
                variant_name=cfg.variant_name,
                micro_entry_ready=False,
                micro_yaw_active=False,
                micro_entry_xy_error=float(xy_error),
                micro_entry_dyaw=float(dyaw_error),
            )
            self.recovery_step += 1
            return decision

        raw, conf, used_visual, used_model, reason = self._proposal_from_inputs(
            estimated_basin_error=est,
            visual_error_state=visual_error_state,
            model_prediction=model_prediction,
            basin_label=basin,
        )
        if self.mode == BasinRecoveryMode.MICRO_SERVO_TO_BASIN:
            proposal = _clamp_basin_correction(
                raw,
                max_xy=cfg.max_micro_xy_step,
                max_z=cfg.max_micro_z_step,
                max_yaw=cfg.max_micro_yaw_step,
            )
        elif self.mode == BasinRecoveryMode.VERIFY_BASIN:
            proposal = _clamp_basin_correction(
                raw,
                max_xy=0.5 * cfg.max_micro_xy_step,
                max_z=0.5 * cfg.max_micro_z_step,
                max_yaw=0.5 * cfg.max_micro_yaw_step,
            )
        else:
            proposal = _clamp_basin_correction(
                raw,
                max_xy=cfg.max_pullback_xy_step,
                max_z=cfg.max_pullback_z_step,
                max_yaw=cfg.max_pullback_yaw_step,
            )
        yaw_active = bool(
            self.mode == BasinRecoveryMode.MICRO_SERVO_TO_BASIN
            and cfg.yaw_control_enabled
            and est is not None
            and est.yaw_valid
            and est.yaw_confidence >= 0.5 * cfg.learned_proposal_min_confidence
            and np.isfinite(dyaw_error)
            and abs(float(dyaw_error)) <= float(cfg.micro_yaw_activation_threshold)
        )
        if not cfg.yaw_control_enabled or not yaw_active:
            proposal[3] = 0.0
        z_active = bool(est is not None and est.z_valid and np.isfinite(z_error))
        if not z_active:
            proposal[2] = 0.0

        line_scale = 1.0
        dry_run = False
        if allow_eval_line_search and target_error_state_for_eval is not None:
            best_corr, line_scale = self._eval_line_search(
                pre_error_state=target_error_state_for_eval,
                planner_prior_state=prior,
                proposal=np.asarray([proposal[0], proposal[1], proposal[3]], dtype=np.float32),
            )
            proposal = np.asarray([best_corr[0], best_corr[1], proposal[2], best_corr[2]], dtype=np.float32)
            dry_run = True
            if line_scale == 0.0:
                reason = f"{reason}+eval_line_search_abstain"

        local_action = np.zeros(6, dtype=np.float32)
        local_action[0] = proposal[0]
        local_action[1] = proposal[1]
        local_action[2] = proposal[2]
        local_action[5] = proposal[3]
        decision = BasinRecoveryDecision(
            mode=self.mode,
            visual_evidence_class=evidence,
            basin_label=basin,
            correction_xyyaw=np.asarray([proposal[0], proposal[1], proposal[3]], dtype=np.float32),
            correction_dz=float(proposal[2]),
            local_action_6d=local_action,
            confidence=conf,
            reason=reason,
            stable_basin_count=self.stable_basin_count,
            recovery_step=self.recovery_step,
            line_search_scale=line_scale,
            variant_name=cfg.variant_name,
            micro_entry_ready=bool(self.mode == BasinRecoveryMode.MICRO_SERVO_TO_BASIN or self.mode == BasinRecoveryMode.VERIFY_BASIN),
            micro_yaw_active=bool(yaw_active),
            micro_entry_xy_error=float(xy_error),
            micro_entry_dyaw=float(dyaw_error),
            used_learned_proposal=used_model,
            used_visual_geometry=used_visual,
            dry_run_scaled_for_eval=dry_run,
            uses_privileged_target=False,
            uses_privileged_label_for_eval=bool(allow_eval_line_search and target_error_state_for_eval is not None),
        )
        self.recovery_step += 1
        return decision


class BasinStateEstimatorNet(nn.Module):
    """Lightweight classifier for recovery mode evidence and basin labels."""

    def __init__(self, *, feature_dim: int = 32, hidden_dim: int = 96, evidence_classes: int = 3, basin_classes: int = 3) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.evidence_head = nn.Linear(hidden_dim, evidence_classes)
        self.basin_head = nn.Linear(hidden_dim, basin_classes)
        self.yaw_observable_head = nn.Linear(hidden_dim, 1)
        self.reacquire_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(features.float())
        return {
            "visual_evidence_logits": self.evidence_head(h),
            "basin_logits": self.basin_head(h),
            "yaw_observable_logit": self.yaw_observable_head(h)[:, 0],
            "reacquire_needed_logit": self.reacquire_head(h)[:, 0],
        }


class BasinPullbackPolicyNet(nn.Module):
    """Bounded proposal network trained with short-horizon basin losses."""

    def __init__(
        self,
        *,
        feature_dim: int = 32,
        hidden_dim: int = 128,
        max_xy_step: float = 0.003,
        max_yaw_step: float = 0.05,
        max_z_step: float = 0.002,
    ) -> None:
        super().__init__()
        self.max_xy_step = float(max_xy_step)
        self.max_yaw_step = float(max_yaw_step)
        self.max_z_step = float(max_z_step)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.xy_head = nn.Linear(hidden_dim, 2)
        self.z_head = nn.Linear(hidden_dim, 1)
        self.yaw_head = nn.Linear(hidden_dim, 1)
        self.conf_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(features.float())
        xy_raw = self.xy_head(h)
        xy = F.normalize(xy_raw, dim=-1, eps=1.0e-6) * torch.tanh(torch.linalg.norm(xy_raw, dim=-1, keepdim=True)) * self.max_xy_step
        dz = torch.tanh(self.z_head(h)[:, 0]) * self.max_z_step
        dyaw = torch.tanh(self.yaw_head(h)[:, 0]) * self.max_yaw_step
        conf = self.conf_head(h)[:, 0]
        return {
            "dx": xy[:, 0],
            "dy": xy[:, 1],
            "dz": dz,
            "dyaw": dyaw,
            "confidence_logit": conf,
            "confidence": torch.sigmoid(conf),
        }


def basin_recovery_feature_vector(record: Mapping[str, Any]) -> list[float]:
    """Compact non-image feature vector for first-pass basin state learning."""

    prior = planner_prior_from_record(record)
    visual = visual_error_from_record(record)
    gripper = list(record.get("gripper_pose", []) or [])[:7]
    gripper = (gripper + [0.0] * 7)[:7]
    return [
        float(record.get("frame_confidence", 0.0)),
        float(record.get("frame_observability", 0.0)),
        float(record.get("frame_axis_strength", 0.0)),
        float(record.get("frame_completeness", 0.0)),
        float(record.get("frame_border_touch", 1.0)),
        float(record.get("trace_error_confidence", 0.0)),
        float(visual[0]),
        float(visual[1]),
        float(visual[2]),
        float(prior[0]),
        float(prior[1]),
        float(prior[2]),
        float(prior[5]),
        float(record.get("planner_bias_score", 0.0)),
        float(record.get("trajectory_step", 0.0)),
        float(record.get("trajectory_len", 1.0)),
        *[float(x) for x in gripper],
    ]
