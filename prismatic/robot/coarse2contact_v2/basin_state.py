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
            f"{prefix}_axis_validity": self.axis_validity,
            f"{prefix}_axis_confidence": self.axis_confidence,
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
        proxy = np.asarray(
            [float(local_error.dx), float(local_error.dy), float(local_error.dz), float(local_error.dyaw)],
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
        )
