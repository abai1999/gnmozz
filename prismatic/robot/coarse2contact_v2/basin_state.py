"""Estimated basin state for Coarse2Contact v2.

The runtime localizer still emits proxy geometry.  This module wraps that proxy
into an explicit estimated basin state so the supervisor can reason about which
axes are actually trusted for control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .localizers import LocalGeometryError
from .specs import PrecisionSkillSpec, PrecisionTaskSpec


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


@dataclass(frozen=True)
class BasinAxisCalibration:
    """Axis-level proxy-to-basin calibration policy."""

    valid: bool = False
    policy: str = "abstain"
    sign: float = 1.0
    scale: float = 1.0
    confidence: float = 0.0
    source: str = "proxy"
    reason: str = "uninitialized"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, default_valid: bool = False) -> "BasinAxisCalibration":
        data = data or {}
        policy = str(data.get("policy", "trusted_control" if default_valid else "abstain"))
        return cls(
            valid=bool(_as_bool(data.get("valid", default_valid), default_valid) and policy == "trusted_control"),
            policy=policy,
            sign=float(np.sign(_as_float(data.get("sign", 1.0), 1.0) or 1.0)),
            scale=float(max(_as_float(data.get("scale", 1.0), 1.0), 0.0)),
            confidence=float(np.clip(_as_float(data.get("confidence", 0.0), 0.0), 0.0, 1.0)),
            source=str(data.get("source", "proxy")),
            reason=str(data.get("reason", "uninitialized")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "policy": str(self.policy),
            "sign": float(self.sign),
            "scale": float(self.scale),
            "confidence": float(self.confidence),
            "source": str(self.source),
            "reason": str(self.reason),
        }


@dataclass(frozen=True)
class BasinStateCalibration:
    """Calibration policy for turning proxy geometry into estimated basin error."""

    x: BasinAxisCalibration = field(default_factory=lambda: BasinAxisCalibration(valid=True, sign=1.0, scale=1.0, confidence=0.45, source="default", reason="default_xy"))
    y: BasinAxisCalibration = field(default_factory=lambda: BasinAxisCalibration(valid=True, sign=1.0, scale=1.0, confidence=0.35, source="default", reason="default_xy"))
    z: BasinAxisCalibration = field(default_factory=lambda: BasinAxisCalibration(valid=True, sign=1.0, scale=1.0, confidence=0.40, source="default", reason="default_z"))
    yaw: BasinAxisCalibration = field(default_factory=lambda: BasinAxisCalibration(valid=False, sign=1.0, scale=1.0, confidence=0.0, source="default", reason="default_yaw_abstain"))
    xy_confidence_floor: float = 0.25
    z_confidence_floor: float = 0.25
    yaw_confidence_floor: float = 0.30
    frame_consistency_floor: float = 0.20

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "BasinStateCalibration":
        data = data or {}
        return cls(
            x=BasinAxisCalibration.from_dict(data.get("x"), default_valid=True),
            y=BasinAxisCalibration.from_dict(data.get("y"), default_valid=True),
            z=BasinAxisCalibration.from_dict(data.get("z"), default_valid=True),
            yaw=BasinAxisCalibration.from_dict(data.get("yaw"), default_valid=False),
            xy_confidence_floor=float(_as_float(data.get("xy_confidence_floor"), 0.25)),
            z_confidence_floor=float(_as_float(data.get("z_confidence_floor"), 0.25)),
            yaw_confidence_floor=float(_as_float(data.get("yaw_confidence_floor"), 0.30)),
            frame_consistency_floor=float(_as_float(data.get("frame_consistency_floor"), 0.20)),
        )

    @classmethod
    def from_report(cls, report: Mapping[str, Any] | None) -> "BasinStateCalibration":
        report = report or {}
        axis_summary = dict(report.get("axis_summary", {}) or {})
        recommendation = dict(report.get("recommendation", {}) or {})

        def _axis(name: str) -> BasinAxisCalibration:
            s = dict(axis_summary.get(name, {}) or {})
            policy = str(recommendation.get(name, s.get("recommended_policy", "abstain")))
            return BasinAxisCalibration(
                valid=bool(policy == "trusted_control"),
                policy=policy,
                sign=1.0,
                scale=1.0,
                confidence=float(s.get("proxy_priv_corr", 0.0) if np.isfinite(_as_float(s.get("proxy_priv_corr"), 0.0)) else 0.0),
                source=str(report.get("variant_name", "report")),
                reason=str(s.get("reason", policy)),
            )

        return cls(
            x=_axis("x"),
            y=_axis("y"),
            z=_axis("z"),
            yaw=_axis("yaw"),
            xy_confidence_floor=0.25,
            z_confidence_floor=0.25,
            yaw_confidence_floor=0.30,
            frame_consistency_floor=0.20,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "z": self.z.to_dict(),
            "yaw": self.yaw.to_dict(),
            "xy_confidence_floor": float(self.xy_confidence_floor),
            "z_confidence_floor": float(self.z_confidence_floor),
            "yaw_confidence_floor": float(self.yaw_confidence_floor),
            "frame_consistency_floor": float(self.frame_consistency_floor),
        }


def load_basin_state_calibration_report(path: str | Path | None) -> BasinStateCalibration | None:
    if path is None:
        return None
    report_path = Path(path)
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return BasinStateCalibration.from_report(data)


@dataclass(frozen=True)
class EstimatedBasinError:
    """Estimated jaw-local basin error used by control and close gating."""

    valid: bool
    confidence: float
    dx: float
    dy: float
    dz: float
    dyaw: float
    x_valid: bool
    y_valid: bool
    z_valid: bool
    yaw_valid: bool
    x_confidence: float
    y_confidence: float
    z_confidence: float
    yaw_confidence: float
    frame_consistency: float
    source: str
    reason: str
    target_entity: str = ""
    reference_entity: str = ""
    stage_name: str = ""
    proxy_dx: float = 0.0
    proxy_dy: float = 0.0
    proxy_dz: float = 0.0
    proxy_dyaw: float = 0.0
    proxy_image_axis_yaw: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "EstimatedBasinError":
        data = data or {}
        axis_validity = dict(data.get("axis_validity", {}) or {})
        axis_confidence = dict(data.get("axis_confidence", {}) or {})
        return cls(
            valid=_as_bool(data.get("valid", False), False),
            confidence=_as_float(data.get("confidence", 0.0), 0.0),
            dx=_as_float(data.get("dx", 0.0), 0.0),
            dy=_as_float(data.get("dy", 0.0), 0.0),
            dz=_as_float(data.get("dz", 0.0), 0.0),
            dyaw=_as_float(data.get("dyaw", 0.0), 0.0),
            x_valid=_as_bool(data.get("x_valid", axis_validity.get("x", False)), False),
            y_valid=_as_bool(data.get("y_valid", axis_validity.get("y", False)), False),
            z_valid=_as_bool(data.get("z_valid", axis_validity.get("z", False)), False),
            yaw_valid=_as_bool(data.get("yaw_valid", axis_validity.get("yaw", False)), False),
            x_confidence=_as_float(data.get("x_confidence", axis_confidence.get("x", 0.0)), 0.0),
            y_confidence=_as_float(data.get("y_confidence", axis_confidence.get("y", 0.0)), 0.0),
            z_confidence=_as_float(data.get("z_confidence", axis_confidence.get("z", 0.0)), 0.0),
            yaw_confidence=_as_float(data.get("yaw_confidence", axis_confidence.get("yaw", 0.0)), 0.0),
            frame_consistency=_as_float(data.get("frame_consistency", 0.0), 0.0),
            source=str(data.get("source", "diagnostic_proxy")),
            reason=str(data.get("reason", "uninitialized")),
            target_entity=str(data.get("target_entity", "")),
            reference_entity=str(data.get("reference_entity", "")),
            stage_name=str(data.get("stage_name", "")),
            proxy_dx=_as_float(data.get("proxy_dx", 0.0), 0.0),
            proxy_dy=_as_float(data.get("proxy_dy", 0.0), 0.0),
            proxy_dz=_as_float(data.get("proxy_dz", 0.0), 0.0),
            proxy_dyaw=_as_float(data.get("proxy_dyaw", 0.0), 0.0),
            proxy_image_axis_yaw=_as_float(data.get("proxy_image_axis_yaw", 0.0), 0.0),
        )

    def to_vector(self) -> np.ndarray:
        return np.asarray([float(self.dx), float(self.dy), float(self.dz), float(self.dyaw)], dtype=np.float32)

    @property
    def xy_valid(self) -> bool:
        return bool(self.x_valid and self.y_valid)

    @property
    def axis_validity(self) -> dict[str, bool]:
        return {"x": bool(self.x_valid), "y": bool(self.y_valid), "z": bool(self.z_valid), "yaw": bool(self.yaw_valid)}

    @property
    def axis_confidence(self) -> dict[str, float]:
        return {
            "x": float(self.x_confidence),
            "y": float(self.y_confidence),
            "z": float(self.z_confidence),
            "yaw": float(self.yaw_confidence),
        }

    @property
    def has_trusted_control_axis(self) -> bool:
        return bool(self.x_valid or self.y_valid or self.z_valid or self.yaw_valid)

    @property
    def trusted_control_axes(self) -> tuple[str, ...]:
        axes: list[str] = []
        if self.x_valid:
            axes.append("x")
        if self.y_valid:
            axes.append("y")
        if self.z_valid:
            axes.append("z")
        if self.yaw_valid:
            axes.append("yaw")
        return tuple(axes)

    @property
    def pullback_ready_axes(self) -> tuple[str, ...]:
        """Axes that may participate in bounded pullback even if entry is not yet ready.

        This is intentionally weaker than `close_ready()`: it separates control
        permission from basin-entry permission.
        """

        axes: list[str] = []
        if self.x_valid:
            axes.append("x")
        if self.y_valid:
            axes.append("y")
        if self.z_valid:
            axes.append("z")
        if self.yaw_valid:
            axes.append("yaw")
        return tuple(axes)

    def close_ready(
        self,
        *,
        xy_threshold: float,
        z_threshold: float,
        yaw_threshold: float,
        yaw_required: bool | None = None,
        min_frame_consistency: float = 0.0,
    ) -> bool:
        xy = float(np.hypot(float(self.dx), float(self.dy)))
        z_ok = bool(abs(float(self.dz)) <= float(z_threshold))
        xy_ok = bool(xy <= float(xy_threshold) and self.xy_valid)
        if yaw_required is None:
            yaw_required = bool(self.yaw_valid)
        yaw_ok = bool((not yaw_required) or (self.yaw_valid and abs(float(self.dyaw)) <= float(yaw_threshold)))
        frame_ok = bool(float(self.frame_consistency) >= float(min_frame_consistency))
        return bool(self.valid and xy_ok and z_ok and yaw_ok and frame_ok)

    def to_trace(
        self,
        *,
        prefix: str = "estimated_basin_error",
        xy_threshold: float = 0.005,
        z_threshold: float = 0.010,
        yaw_threshold: float = 0.03,
        yaw_required: bool | None = None,
        min_frame_consistency: float = 0.0,
    ) -> dict[str, Any]:
        return {
            f"{prefix}_valid": bool(self.valid),
            f"{prefix}_confidence": float(self.confidence),
            f"{prefix}_dx": float(self.dx),
            f"{prefix}_dy": float(self.dy),
            f"{prefix}_dz": float(self.dz),
            f"{prefix}_dyaw": float(self.dyaw),
            f"{prefix}_x_valid": bool(self.x_valid),
            f"{prefix}_y_valid": bool(self.y_valid),
            f"{prefix}_z_valid": bool(self.z_valid),
            f"{prefix}_yaw_valid": bool(self.yaw_valid),
            f"{prefix}_x_confidence": float(self.x_confidence),
            f"{prefix}_y_confidence": float(self.y_confidence),
            f"{prefix}_z_confidence": float(self.z_confidence),
            f"{prefix}_yaw_confidence": float(self.yaw_confidence),
            f"{prefix}_frame_consistency": float(self.frame_consistency),
            f"{prefix}_source": str(self.source),
            f"{prefix}_reason": str(self.reason),
            f"{prefix}_target_entity": str(self.target_entity),
            f"{prefix}_reference_entity": str(self.reference_entity),
            f"{prefix}_stage_name": str(self.stage_name),
            f"{prefix}_proxy_dx": float(self.proxy_dx),
            f"{prefix}_proxy_dy": float(self.proxy_dy),
            f"{prefix}_proxy_dz": float(self.proxy_dz),
            f"{prefix}_proxy_dyaw": float(self.proxy_dyaw),
            f"{prefix}_proxy_image_axis_yaw": float(self.proxy_image_axis_yaw),
            f"{prefix}_axis_validity": self.axis_validity,
            f"{prefix}_axis_confidence": self.axis_confidence,
            f"{prefix}_pullback_ready_axes": list(self.pullback_ready_axes),
            f"{prefix}_close_ready": bool(
                self.close_ready(
                    xy_threshold=xy_threshold,
                    z_threshold=z_threshold,
                    yaw_threshold=yaw_threshold,
                    yaw_required=yaw_required,
                    min_frame_consistency=min_frame_consistency,
                )
            ),
        }


@dataclass(frozen=True)
class FrameRelabelBasinEstimator:
    """Replay-friendly estimator backed by privileged relabel rows.

    The estimator does not infer geometry from RGBD.  It consumes the offline
    relabel schema and emits a conservative basin estimate so replay/audit can
    verify whether a grasp-only pullback policy would contract the privileged
    error.
    """

    yaw_observable_min_abs: float = 0.01
    close_ready_z_threshold: float = 0.010

    def _row_error(self, row: Mapping[str, Any]) -> tuple[float, float, float, float]:
        if isinstance(row.get("true_basin_error_t"), Mapping):
            err = dict(row.get("true_basin_error_t") or {})
        else:
            err = row
        dx = _as_float(err.get("dx", err.get("privileged_dx", 0.0)), 0.0)
        dy = _as_float(err.get("dy", err.get("privileged_dy", 0.0)), 0.0)
        dz = _as_float(err.get("dz", err.get("privileged_dz", 0.0)), 0.0)
        dyaw = _as_float(err.get("dyaw", err.get("privileged_dyaw", 0.0)), 0.0)
        return dx, dy, dz, dyaw

    def estimate(
        self,
        relabel_row: Mapping[str, Any],
        *,
        stage_name: str = "",
    ) -> EstimatedBasinError:
        obs = dict(relabel_row.get("obs_t") or {})
        visual_class = str(obs.get("visual_observability_class", relabel_row.get("visual_observability_class", "prior_only")))
        frame_conf = float(obs.get("frame_confidence", relabel_row.get("source_frame_confidence", 0.0)) or 0.0)
        frame_obs = float(obs.get("frame_observability", relabel_row.get("source_frame_observability", 0.0)) or 0.0)
        frame_axis = float(obs.get("frame_axis_strength", relabel_row.get("source_frame_axis_strength", 0.0)) or 0.0)
        yaw_observable = bool(relabel_row.get("yaw_observable", False))
        reacquire_needed = bool(relabel_row.get("reacquire_needed", visual_class == "prior_only"))
        dx, dy, dz, dyaw = self._row_error(relabel_row)

        pullback_allowed = bool(not reacquire_needed)
        x_valid = bool(pullback_allowed and np.isfinite(dx))
        y_valid = bool(pullback_allowed and np.isfinite(dy))
        z_valid = bool(pullback_allowed and np.isfinite(dz))
        yaw_valid = bool(pullback_allowed and yaw_observable and np.isfinite(dyaw) and abs(float(dyaw)) >= float(self.yaw_observable_min_abs))

        confidence = float(np.clip(max(frame_conf, frame_obs, frame_axis), 0.0, 1.0))
        if visual_class == "prior_only":
            confidence *= 0.0
            x_valid = y_valid = z_valid = yaw_valid = False

        if not any((x_valid, y_valid, z_valid, yaw_valid)):
            reason = "reacquire_needed" if visual_class == "prior_only" else "abstain_low_axis_validity"
        elif yaw_valid:
            reason = "replay_yaw_observable"
        elif x_valid or y_valid:
            reason = "replay_xy_pullback"
        else:
            reason = "replay_z_diagnostic"

        proxy = dict(relabel_row.get("proxy_local_geometry_error") or {})
        est = dict(relabel_row.get("estimated_basin_error") or {})
        return EstimatedBasinError(
            valid=bool(any((x_valid, y_valid, z_valid, yaw_valid))),
            confidence=confidence,
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
            dyaw=float(dyaw),
            x_valid=bool(x_valid),
            y_valid=bool(y_valid),
            z_valid=bool(z_valid),
            yaw_valid=bool(yaw_valid),
            x_confidence=float(confidence if x_valid else 0.0),
            y_confidence=float(confidence if y_valid else 0.0),
            z_confidence=float(confidence if z_valid else 0.0),
            yaw_confidence=float(confidence if yaw_valid else 0.0),
            frame_consistency=float(np.clip(max(frame_obs, frame_axis), 0.0, 1.0)),
            source="privileged_relabel",
            reason=reason,
            target_entity=str((relabel_row.get("frame_contract") or {}).get("target_frame", relabel_row.get("target_frame", ""))),
            reference_entity=str((relabel_row.get("frame_contract") or {}).get("reference_frame", relabel_row.get("reference_frame", ""))),
            stage_name=str(stage_name or relabel_row.get("stage_name", "")),
            proxy_dx=float(proxy.get("dx", est.get("dx", 0.0)) or 0.0),
            proxy_dy=float(proxy.get("dy", est.get("dy", 0.0)) or 0.0),
            proxy_dz=float(proxy.get("dz", est.get("dz", 0.0)) or 0.0),
            proxy_dyaw=float(proxy.get("dyaw", est.get("dyaw", 0.0)) or 0.0),
        )


@dataclass(frozen=True)
class ReplayBasinResult:
    estimated_basin_error: EstimatedBasinError
    correction_local_6d: np.ndarray
    post_error_t: np.ndarray
    post_error_t_plus_1: np.ndarray | None
    one_step_contraction: bool
    overshoot: bool
    monotonic_prefix: bool
    micro_entry_ready: bool
    micro_entry_block_reason: str
    close_ready_ready: bool
    close_ready_block_reason: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_basin_error": self.estimated_basin_error.to_trace(),
            "correction_local_6d": [float(x) for x in np.asarray(self.correction_local_6d, dtype=np.float32).reshape(-1)[:6]],
            "post_error_t": [float(x) for x in np.asarray(self.post_error_t, dtype=np.float32).reshape(-1)[:3]],
            "post_error_t_plus_1": None if self.post_error_t_plus_1 is None else [float(x) for x in np.asarray(self.post_error_t_plus_1, dtype=np.float32).reshape(-1)[:3]],
            "one_step_contraction": bool(self.one_step_contraction),
            "overshoot": bool(self.overshoot),
            "monotonic_prefix": bool(self.monotonic_prefix),
            "micro_entry_ready": bool(self.micro_entry_ready),
            "micro_entry_block_reason": str(self.micro_entry_block_reason),
            "close_ready_ready": bool(self.close_ready_ready),
            "close_ready_block_reason": str(self.close_ready_block_reason),
            "reason": str(self.reason),
        }


@dataclass(frozen=True)
class ReplayBasinEstimator:
    """Replay the privileged relabel into a minimal grasp-only pullback step."""

    estimator: FrameRelabelBasinEstimator = field(default_factory=FrameRelabelBasinEstimator)
    xy_gain: float = 0.35
    yaw_gain: float = 0.0
    z_gain: float = 0.0
    max_xy_step: float = 0.003
    max_z_step: float = 0.0012
    max_yaw_step: float = 0.0
    yaw_enabled: bool = False
    z_enabled: bool = False

    @staticmethod
    def _clamp_xyyaw(correction: np.ndarray, *, max_xy: float, max_z: float, max_yaw: float) -> np.ndarray:
        corr = np.asarray(correction, dtype=np.float32).reshape(-1)
        corr = np.pad(corr, (0, max(0, 6 - corr.size)))[:6]
        xy = corr[:2]
        norm = float(np.linalg.norm(xy))
        if norm > float(max_xy) > 0.0:
            xy = xy * (float(max_xy) / max(norm, 1.0e-9))
        corr[:2] = xy
        corr[2] = float(np.clip(corr[2], -float(max_z), float(max_z)))
        corr[5] = float(np.clip(corr[5], -float(max_yaw), float(max_yaw)))
        return corr.astype(np.float32)

    def propose(self, relabel_row: Mapping[str, Any], *, stage_name: str = "") -> dict[str, Any]:
        est = self.estimator.estimate(relabel_row, stage_name=stage_name)
        visual_class = str((relabel_row.get("obs_t") or {}).get("visual_observability_class", relabel_row.get("visual_observability_class", "prior_only")))
        if visual_class == "prior_only" or not est.valid:
            corr = np.zeros(6, dtype=np.float32)
            return {
                "estimated_basin_error": est,
                "correction_local_6d": corr,
                "mode": "REACQUIRE_VIEW",
                "reason": "prior_only_reacquire" if visual_class == "prior_only" else "replay_abstain",
            }

        corr = np.zeros(6, dtype=np.float32)
        if est.x_valid:
            corr[0] = float(self.xy_gain * est.dx)
        if est.y_valid:
            corr[1] = float(self.xy_gain * est.dy)
        if self.z_enabled and est.z_valid:
            corr[2] = float(self.z_gain * est.dz)
        if self.yaw_enabled and est.yaw_valid:
            corr[5] = float(self.yaw_gain * est.dyaw)
        corr = self._clamp_xyyaw(corr, max_xy=self.max_xy_step, max_z=self.max_z_step, max_yaw=self.max_yaw_step if self.yaw_enabled else 0.0)
        mode = "VISUAL_PULLBACK"
        if self.yaw_enabled and abs(float(corr[5])) > 1.0e-9:
            mode = "MICRO_SERVO_TO_BASIN"
        return {
            "estimated_basin_error": est,
            "correction_local_6d": corr,
            "mode": mode,
            "reason": "privileged_relabel_replay",
        }

    def replay(self, relabel_row: Mapping[str, Any], *, stage_name: str = "") -> ReplayBasinResult:
        from .recovery_audit import apply_closed_loop_recovery_step, monotonic_decay_prefix, recovery_error_norm, recovery_overshoot_flag

        proposal = self.propose(relabel_row, stage_name=stage_name)
        est = proposal["estimated_basin_error"]
        correction = np.asarray(proposal["correction_local_6d"], dtype=np.float32).reshape(-1)[:6]
        true_error = np.asarray(
            [
                _as_float((relabel_row.get("true_basin_error_t") or relabel_row).get("dx", relabel_row.get("privileged_dx", 0.0))),
                _as_float((relabel_row.get("true_basin_error_t") or relabel_row).get("dy", relabel_row.get("privileged_dy", 0.0))),
                _as_float((relabel_row.get("true_basin_error_t") or relabel_row).get("dyaw", relabel_row.get("privileged_dyaw", 0.0))),
            ],
            dtype=np.float32,
        )
        planner_prior = np.asarray((list((relabel_row.get("planner_prior") or {}).get("local_delta_6d", relabel_row.get("planner_local_delta_6d", []))) + [0.0] * 6)[:6], dtype=np.float32)
        post_error, _ = apply_closed_loop_recovery_step(true_error, planner_prior, correction[:3])
        next_error = relabel_row.get("true_basin_error_t_plus_1") or {}
        next_vec = np.asarray(
            [
                _as_float(next_error.get("dx", relabel_row.get("next_privileged_dx", float("nan"))), float("nan")),
                _as_float(next_error.get("dy", relabel_row.get("next_privileged_dy", float("nan"))), float("nan")),
                _as_float(next_error.get("dyaw", relabel_row.get("next_privileged_dyaw", float("nan"))), float("nan")),
            ],
            dtype=np.float32,
        )
        pre_norm = recovery_error_norm(float(true_error[0]), float(true_error[1]), float(true_error[2]))
        post_norm = recovery_error_norm(float(post_error[0]), float(post_error[1]), float(post_error[2]))
        next_norm = recovery_error_norm(float(next_vec[0]), float(next_vec[1]), float(next_vec[2])) if np.all(np.isfinite(next_vec)) else float("nan")
        return ReplayBasinResult(
            estimated_basin_error=est,
            correction_local_6d=correction.astype(np.float32),
            post_error_t=post_error.astype(np.float32),
            post_error_t_plus_1=next_vec.astype(np.float32) if np.all(np.isfinite(next_vec)) else None,
            one_step_contraction=bool(post_norm <= pre_norm + 1.0e-9),
            overshoot=bool(recovery_overshoot_flag(true_error, post_error)),
            monotonic_prefix=bool(monotonic_decay_prefix([pre_norm, post_norm, next_norm])) if np.isfinite(next_norm) else bool(pre_norm >= post_norm),
            micro_entry_ready=bool(est.close_ready(xy_threshold=0.015, z_threshold=self.estimator.close_ready_z_threshold, yaw_threshold=0.08, yaw_required=False)),
            micro_entry_block_reason=str(proposal.get("reason", "")),
            close_ready_ready=bool(est.close_ready(xy_threshold=0.005, z_threshold=self.estimator.close_ready_z_threshold, yaw_threshold=0.03, yaw_required=False)),
            close_ready_block_reason=str(proposal.get("reason", "")),
            reason=str(proposal.get("reason", "")),
        )

class BasinStateEstimator(Protocol):
    def estimate(
        self,
        local_error: LocalGeometryError,
        robot_state: Mapping[str, Any],
        task_spec: PrecisionTaskSpec,
        skill_spec: PrecisionSkillSpec,
        *,
        stage_name: str = "",
    ) -> EstimatedBasinError: ...


class CalibratedGraspBasinEstimator:
    """Rule-based proxy-to-basin estimator with per-axis calibration."""

    def __init__(
        self,
        calibration: BasinStateCalibration | Mapping[str, Any] | None = None,
    ) -> None:
        if calibration is None:
            calibration = BasinStateCalibration()
        elif not isinstance(calibration, BasinStateCalibration):
            calibration = BasinStateCalibration.from_dict(calibration)
        self.calibration = calibration

    def _axis_from_proxy(
        self,
        *,
        axis_name: str,
        proxy_value: float,
        axis_cal: BasinAxisCalibration,
        base_confidence: float,
        obs: float,
        fit_residual: float,
        frame_consistency: float,
        controlled_dofs: set[str],
    ) -> tuple[float, bool, float, str]:
        if not np.isfinite(proxy_value):
            return 0.0, False, 0.0, "non_finite_proxy"
        conf = float(np.clip(axis_cal.confidence + base_confidence + 0.5 * obs - 0.15 * fit_residual, 0.0, 1.0))
        if axis_name not in controlled_dofs:
            valid = False
            reason = "not_controlled_by_skill"
        else:
            valid = bool(axis_cal.valid and conf >= 0.05 and frame_consistency >= self.calibration.frame_consistency_floor)
            reason = "calibrated" if valid else "diagnostic_only"
        corrected = float(axis_cal.sign * axis_cal.scale * proxy_value)
        return corrected, valid, conf, reason

    def estimate(
        self,
        local_error: LocalGeometryError,
        robot_state: Mapping[str, Any],
        task_spec: PrecisionTaskSpec,
        skill_spec: PrecisionSkillSpec,
        *,
        stage_name: str = "",
    ) -> EstimatedBasinError:
        localizer_yaw_valid = bool(getattr(local_error, "yaw_valid", True))
        proxy_image_axis_yaw = float(getattr(local_error, "image_axis_yaw", 0.0) or 0.0)
        proxy = np.asarray(
            [
                float(local_error.dx),
                float(local_error.dy),
                float(local_error.dz),
                float(local_error.dyaw) if localizer_yaw_valid else 0.0,
            ],
            dtype=np.float32,
        )
        obs = float(np.clip(local_error.observability, 0.0, 1.0))
        fit_residual = float(max(local_error.fit_residual, 0.0))
        frame_consistency = float(np.clip(1.0 - fit_residual + 0.5 * obs, 0.0, 1.0))
        conf_base = float(np.clip(local_error.confidence, 0.0, 1.0))
        controlled_dofs = set(str(x) for x in skill_spec.controlled_dofs)

        x, x_valid, x_conf, x_reason = self._axis_from_proxy(
            axis_name="x",
            proxy_value=float(proxy[0]),
            axis_cal=self.calibration.x,
            base_confidence=conf_base,
            obs=obs,
            fit_residual=fit_residual,
            frame_consistency=frame_consistency,
            controlled_dofs=controlled_dofs,
        )
        y, y_valid, y_conf, y_reason = self._axis_from_proxy(
            axis_name="y",
            proxy_value=float(proxy[1]),
            axis_cal=self.calibration.y,
            base_confidence=conf_base,
            obs=obs,
            fit_residual=fit_residual,
            frame_consistency=frame_consistency,
            controlled_dofs=controlled_dofs,
        )
        z, z_valid, z_conf, z_reason = self._axis_from_proxy(
            axis_name="z",
            proxy_value=float(proxy[2]),
            axis_cal=self.calibration.z,
            base_confidence=conf_base,
            obs=obs,
            fit_residual=fit_residual,
            frame_consistency=frame_consistency,
            controlled_dofs=controlled_dofs,
        )
        if localizer_yaw_valid:
            yaw, yaw_valid, yaw_conf, yaw_reason = self._axis_from_proxy(
                axis_name="yaw",
                proxy_value=float(proxy[3]),
                axis_cal=self.calibration.yaw,
                base_confidence=conf_base,
                obs=obs,
                fit_residual=fit_residual,
                frame_consistency=frame_consistency,
                controlled_dofs=controlled_dofs,
            )
        else:
            yaw = 0.0
            yaw_valid = False
            yaw_conf = 0.0
            yaw_reason = str(getattr(local_error, "yaw_reason", "") or "localizer_yaw_not_residual")

        # If the observation itself is weak, keep the estimate conservative.
        wrist_valid_depth_ratio = float(robot_state.get("wrist_valid_depth_ratio", 0.0) or 0.0)
        wrist_depth_near_fraction = float(robot_state.get("wrist_depth_near_fraction", 0.0) or 0.0)
        wrist_is_occluded = bool(robot_state.get("wrist_is_occluded", False))
        wrist_is_low_visibility = bool(robot_state.get("wrist_is_low_visibility", False))
        if wrist_is_low_visibility:
            x_conf *= 0.6
            y_conf *= 0.6
            z_conf *= 0.7
            yaw_conf *= 0.5
            frame_consistency *= 0.7
        if wrist_is_occluded:
            x_conf *= 0.35
            y_conf *= 0.35
            z_conf *= 0.45
            yaw_conf *= 0.25
            frame_consistency *= 0.5

        x_valid = bool(x_valid and x_conf >= self.calibration.xy_confidence_floor)
        y_valid = bool(y_valid and y_conf >= self.calibration.xy_confidence_floor)
        z_valid = bool(z_valid and z_conf >= self.calibration.z_confidence_floor)
        yaw_valid = bool(yaw_valid and yaw_conf >= self.calibration.yaw_confidence_floor)

        # Use depth structure as an extra sanity check for the z channel.
        if z_valid and wrist_valid_depth_ratio <= 0.0 and wrist_depth_near_fraction <= 0.0:
            z_valid = False
            z_reason = "low_depth_support"

        valid = bool(x_valid or y_valid or z_valid or yaw_valid)
        confidence = float(np.clip(max(x_conf, y_conf, z_conf, yaw_conf), 0.0, 1.0))
        if not valid:
            reason = "abstain_low_axis_validity"
        elif yaw_valid:
            reason = "calibrated_yaw_supported"
        elif z_valid and (x_valid or y_valid):
            reason = "calibrated_xy_z_supported"
        elif x_valid or y_valid:
            reason = "calibrated_xy_supported"
        else:
            reason = "calibrated_z_supported"

        return EstimatedBasinError(
            valid=valid,
            confidence=confidence,
            dx=float(x),
            dy=float(y),
            dz=float(z),
            dyaw=float(yaw),
            x_valid=bool(x_valid),
            y_valid=bool(y_valid),
            z_valid=bool(z_valid),
            yaw_valid=bool(yaw_valid),
            x_confidence=float(x_conf),
            y_confidence=float(y_conf),
            z_confidence=float(z_conf),
            yaw_confidence=float(yaw_conf),
            frame_consistency=float(np.clip(frame_consistency, 0.0, 1.0)),
            source="calibrated_proxy" if valid else "diagnostic_proxy",
            reason=reason,
            target_entity=skill_spec.target_entity,
            reference_entity=skill_spec.reference_entity,
            stage_name=stage_name,
            proxy_dx=float(proxy[0]),
            proxy_dy=float(proxy[1]),
            proxy_dz=float(proxy[2]),
            proxy_dyaw=float(proxy[3]),
            proxy_image_axis_yaw=proxy_image_axis_yaw,
        )
