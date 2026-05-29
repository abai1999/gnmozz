"""Formal C2C v2 residual, observability, and takeover-tier contract.

This module is deliberately small and runtime-neutral.  It centralizes the
semantic decisions that were previously spread across relabel/audit/probe code:

* Residual Estimator contract: frame-to-frame error as dx/dy/dz/dyaw.
* Observability Gate contract: which non-privileged evidence makes yaw usable.
* Takeover Tier contract: coarse pullback, near shell, micro entry, close-ready,
  or abstain/outside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


NEAR_GRASP_XY_THRESHOLD = 0.015
NEAR_GRASP_YAW_THRESHOLD = 0.08
CLOSE_READY_XY_THRESHOLD = 0.005
CLOSE_READY_YAW_THRESHOLD = 0.03
CLOSE_READY_Z_THRESHOLD = 0.010
DEFAULT_PULLBACK_HORIZON = 3
DEFAULT_MAX_XY_STEP = 0.003
COARSE_PULLBACK_XY_THRESHOLD = 0.060
OUTER_PULLBACK_XY_THRESHOLD = 0.120

YAW_OBSERVABLE = "observable"
YAW_AMBIGUOUS = "ambiguous"
YAW_UNOBSERVABLE = "unobservable"

TIER_COARSE_PULLBACK = "coarse_pullback_candidate"
TIER_OUTER_PULLBACK = "outer_pullback_candidate"
TIER_NEAR_BASIN = "near_basin_shell"
TIER_MICRO_ENTRY = "micro_entry_ready"
TIER_CLOSE_READY = "close_ready"
TIER_ABSTAIN_PRIOR = "abstain_prior_only"
TIER_OUTSIDE = "outside_takeover"
TIER_INVALID = "invalid"


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _near_grasp(dx: float, dy: float, dyaw: float, *, xy_threshold: float, yaw_threshold: float) -> bool:
    return bool(
        np.isfinite(dx)
        and np.isfinite(dy)
        and np.isfinite(dyaw)
        and float(np.hypot(dx, dy)) <= float(xy_threshold)
        and abs(float(dyaw)) <= float(yaw_threshold)
    )


@dataclass(frozen=True)
class FrameResidual:
    """Frame-to-frame residual expressed in the reference frame."""

    dx: float
    dy: float
    dz: float
    dyaw: float
    reference_frame: str = ""
    target_frame: str = ""
    z_semantics: str = ""
    source: str = "unknown"

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, source: str = "mapping") -> "FrameResidual":
        err = row.get("true_basin_error_t") if isinstance(row.get("true_basin_error_t"), Mapping) else row
        err = err if isinstance(err, Mapping) else {}
        return cls(
            dx=_as_float(err.get("dx", row.get("privileged_dx", float("nan")))),
            dy=_as_float(err.get("dy", row.get("privileged_dy", float("nan")))),
            dz=_as_float(err.get("dz", row.get("privileged_dz", float("nan")))),
            dyaw=_as_float(err.get("dyaw", row.get("privileged_dyaw", float("nan")))),
            reference_frame=str(row.get("reference_frame", (row.get("frame_contract") or {}).get("reference_frame", ""))),
            target_frame=str(row.get("target_frame", (row.get("frame_contract") or {}).get("target_frame", ""))),
            z_semantics=str(row.get("z_semantics", (row.get("frame_contract") or {}).get("z_semantics", ""))),
            source=source,
        )

    @property
    def finite(self) -> bool:
        return bool(all(_finite(v) for v in (self.dx, self.dy, self.dz, self.dyaw)))

    @property
    def xy_error(self) -> float:
        return float(np.hypot(float(self.dx), float(self.dy))) if _finite(self.dx) and _finite(self.dy) else float("nan")

    @property
    def yaw_abs(self) -> float:
        return float(abs(float(self.dyaw))) if _finite(self.dyaw) else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dx": float(self.dx),
            "dy": float(self.dy),
            "dz": float(self.dz),
            "dyaw": float(self.dyaw),
            "xy_error": float(self.xy_error),
            "yaw_abs": float(self.yaw_abs),
            "reference_frame": str(self.reference_frame),
            "target_frame": str(self.target_frame),
            "z_semantics": str(self.z_semantics),
            "source": str(self.source),
            "finite": bool(self.finite),
        }


@dataclass(frozen=True)
class ObservabilityDecision:
    """Non-privileged observability gate for frame/yaw use."""

    visual_observability_class: str
    yaw_observability_class: str
    yaw_observable: bool
    reacquire_needed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_observability_class": str(self.visual_observability_class),
            "yaw_observability_class": str(self.yaw_observability_class),
            "yaw_observable": bool(self.yaw_observable),
            "reacquire_needed": bool(self.reacquire_needed),
            "reason": str(self.reason),
        }


@dataclass(frozen=True)
class TakeoverThresholds:
    near_xy: float = NEAR_GRASP_XY_THRESHOLD
    near_yaw: float = NEAR_GRASP_YAW_THRESHOLD
    close_xy: float = CLOSE_READY_XY_THRESHOLD
    close_yaw: float = CLOSE_READY_YAW_THRESHOLD
    close_z: float = CLOSE_READY_Z_THRESHOLD
    max_xy_step: float = DEFAULT_MAX_XY_STEP
    pullback_horizon: int = DEFAULT_PULLBACK_HORIZON
    coarse_xy: float = COARSE_PULLBACK_XY_THRESHOLD
    outer_xy: float = OUTER_PULLBACK_XY_THRESHOLD


@dataclass(frozen=True)
class TakeoverTierDecision:
    """Tiered takeover decision derived from residual and observability."""

    takeover_tier: str
    pullback_allowed: bool
    yaw_entry_feasible: bool
    yaw_control_observable: bool
    near_basin_shell: bool
    coarse_pullback_candidate: bool
    outer_pullback_candidate: bool
    micro_entry_ready: bool
    close_ready_ready: bool
    yaw_entry_block_reason: str
    yaw_control_block_reason: str
    micro_entry_block_reason: str
    close_ready_block_reason: str
    axis_gate_policy: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "takeover_tier": str(self.takeover_tier),
            "pullback_allowed": bool(self.pullback_allowed),
            "yaw_entry_feasible": bool(self.yaw_entry_feasible),
            "yaw_control_observable": bool(self.yaw_control_observable),
            "near_basin_shell": bool(self.near_basin_shell),
            "coarse_pullback_candidate": bool(self.coarse_pullback_candidate),
            "outer_pullback_candidate": bool(self.outer_pullback_candidate),
            "micro_entry_ready": bool(self.micro_entry_ready),
            "close_ready_ready": bool(self.close_ready_ready),
            "yaw_entry_block_reason": str(self.yaw_entry_block_reason),
            "yaw_control_block_reason": str(self.yaw_control_block_reason),
            "micro_entry_block_reason": str(self.micro_entry_block_reason),
            "close_ready_block_reason": str(self.close_ready_block_reason),
            "axis_gate_policy": dict(self.axis_gate_policy),
        }


def classify_yaw_observability(
    trace_row: Mapping[str, Any],
    visual_record: Mapping[str, Any],
    *,
    visual_observability_class: str,
) -> ObservabilityDecision:
    """Classify yaw use from non-privileged visual evidence only."""

    visual_class = str(visual_observability_class)
    if visual_class == "prior_only":
        return ObservabilityDecision(
            visual_observability_class=visual_class,
            yaw_observability_class=YAW_UNOBSERVABLE,
            yaw_observable=False,
            reacquire_needed=True,
            reason="prior_only",
        )
    if bool(trace_row.get("wrist_is_occluded", False)):
        return ObservabilityDecision(
            visual_observability_class=visual_class,
            yaw_observability_class=YAW_UNOBSERVABLE,
            yaw_observable=False,
            reacquire_needed=False,
            reason="wrist_occluded",
        )

    conf = _as_float(visual_record.get("frame_confidence", 0.0), 0.0)
    obs = _as_float(visual_record.get("frame_observability", 0.0), 0.0)
    axis = _as_float(visual_record.get("frame_axis_strength", 0.0), 0.0)
    wide_visible = bool(visual_record.get("wide_ring_visible", False))
    if conf >= 0.50 and obs >= 0.10 and axis >= 0.80:
        cls = YAW_OBSERVABLE
        reason = "frame_axis_consistent"
    elif conf >= 0.05 or obs >= 0.005 or wide_visible:
        cls = YAW_AMBIGUOUS
        reason = "partial_frame_evidence"
    else:
        cls = YAW_UNOBSERVABLE
        reason = "low_frame_evidence"
    return ObservabilityDecision(
        visual_observability_class=visual_class,
        yaw_observability_class=cls,
        yaw_observable=bool(cls == YAW_OBSERVABLE),
        reacquire_needed=False,
        reason=reason,
    )


def explain_yaw_observability(
    trace_row: Mapping[str, Any],
    visual_record: Mapping[str, Any],
    *,
    visual_observability_class: str,
) -> dict[str, Any]:
    """Explain which non-privileged evidence gates yaw observability.

    This mirrors :func:`classify_yaw_observability` but exposes the threshold
    failures so offline relabel/audit can tell whether rows are blocked because
    of weak frame observability, low confidence, weak axis evidence, or a true
    prior-only / occlusion fallback.
    """

    visual_class = str(visual_observability_class)
    conf = _as_float(visual_record.get("frame_confidence", 0.0), 0.0)
    obs = _as_float(visual_record.get("frame_observability", 0.0), 0.0)
    axis = _as_float(visual_record.get("frame_axis_strength", 0.0), 0.0)
    wide_visible = bool(visual_record.get("wide_ring_visible", False))
    wrist_occluded = bool(trace_row.get("wrist_is_occluded", False))

    gate_passes = {
        "frame_confidence": bool(conf >= 0.50),
        "frame_observability": bool(obs >= 0.10),
        "frame_axis_strength": bool(axis >= 0.80),
        "wide_ring_visible": bool(wide_visible),
        "wrist_not_occluded": bool(not wrist_occluded),
    }

    blockers: list[str] = []
    primary_blocker = "observable"
    reason = "frame_axis_consistent"

    if visual_class == "prior_only":
        blockers = ["prior_only"]
        primary_blocker = "prior_only"
        reason = "prior_only"
    elif wrist_occluded:
        blockers = ["wrist_occluded"]
        primary_blocker = "wrist_occluded"
        reason = "wrist_occluded"
    else:
        if not gate_passes["frame_observability"]:
            blockers.append("frame_observability_lt_010")
        if not gate_passes["frame_confidence"]:
            blockers.append("frame_confidence_lt_050")
        if not gate_passes["frame_axis_strength"]:
            blockers.append("frame_axis_strength_lt_080")

        if blockers:
            primary_blocker = blockers[0]
            if wide_visible or gate_passes["frame_confidence"] or gate_passes["frame_observability"]:
                reason = "partial_frame_evidence"
            else:
                reason = "low_frame_evidence"

    return {
        "visual_observability_class": visual_class,
        "frame_confidence": conf,
        "frame_observability": obs,
        "frame_axis_strength": axis,
        "wide_ring_visible": wide_visible,
        "wrist_is_occluded": wrist_occluded,
        "gate_passes": gate_passes,
        "blockers": blockers,
        "blocker_combo": "+".join(blockers) if blockers else "",
        "primary_blocker": primary_blocker,
        "reason": reason,
    }


def decide_takeover_tier(
    residual: FrameResidual,
    observability: ObservabilityDecision,
    *,
    precision_row: bool,
    requires_yaw_observability: bool,
    xy_contracted: bool = False,
    thresholds: TakeoverThresholds | None = None,
) -> TakeoverTierDecision:
    """Derive the formal takeover tier from residual and observability."""

    t = thresholds or TakeoverThresholds()
    pullback_allowed = bool(precision_row and not observability.reacquire_needed)
    finite = bool(precision_row and residual.finite)
    yaw_entry_feasible = bool(finite and residual.yaw_abs <= float(t.near_yaw) + 1.0e-9)
    close_yaw_entry_feasible = bool(finite and residual.yaw_abs <= float(t.close_yaw) + 1.0e-9)
    yaw_control_observable = bool(precision_row and observability.yaw_observable)

    near_shell = bool(
        finite
        and pullback_allowed
        and yaw_entry_feasible
        and residual.xy_error <= float(t.near_xy) + float(t.max_xy_step) * float(max(1, int(t.pullback_horizon))) + 1.0e-9
    )
    micro_ready = bool(
        finite
        and pullback_allowed
        and _near_grasp(residual.dx, residual.dy, residual.dyaw, xy_threshold=t.near_xy, yaw_threshold=t.near_yaw)
        and yaw_entry_feasible
    )
    close_ready = bool(
        finite
        and pullback_allowed
        and _near_grasp(residual.dx, residual.dy, residual.dyaw, xy_threshold=t.close_xy, yaw_threshold=t.close_yaw)
        and abs(float(residual.dz)) <= float(t.close_z)
        and close_yaw_entry_feasible
    )
    coarse_candidate = bool(
        finite
        and pullback_allowed
        and not near_shell
        and residual.xy_error <= float(t.coarse_xy)
        and yaw_entry_feasible
        and bool(xy_contracted)
    )
    outer_candidate = bool(
        finite
        and pullback_allowed
        and not near_shell
        and not coarse_candidate
        and residual.xy_error <= float(t.outer_xy)
        and bool(xy_contracted)
    )

    micro_blocks: list[str] = []
    close_blocks: list[str] = []
    yaw_entry_block_reason = "ready"
    yaw_control_block_reason = "ready" if yaw_control_observable else str(observability.reason or observability.yaw_observability_class)
    if not precision_row:
        micro_blocks.append("not_precision")
        close_blocks.append("not_precision")
        yaw_entry_block_reason = "not_precision"
        yaw_control_block_reason = "not_precision"
    elif not residual.finite:
        micro_blocks.append("invalid_residual")
        close_blocks.append("invalid_residual")
        yaw_entry_block_reason = "invalid_residual"
        yaw_control_block_reason = "invalid_residual"
    elif observability.reacquire_needed:
        micro_blocks.append("prior_only")
        close_blocks.append("prior_only")
        yaw_entry_block_reason = "prior_only"
        yaw_control_block_reason = "prior_only"
    else:
        if residual.xy_error > float(t.near_xy):
            micro_blocks.append("xy")
        if residual.xy_error > float(t.close_xy):
            close_blocks.append("xy")
        if abs(float(residual.dz)) > float(t.close_z):
            close_blocks.append("z")
        if not yaw_entry_feasible:
            micro_blocks.append("yaw_entry")
            close_blocks.append("yaw_entry")
            yaw_entry_block_reason = "yaw_abs_gt_near_threshold"
        elif not close_yaw_entry_feasible:
            close_blocks.append("yaw_entry_close")

    if not precision_row:
        tier = TIER_INVALID
    elif observability.reacquire_needed:
        tier = TIER_ABSTAIN_PRIOR
    elif not residual.finite:
        tier = TIER_INVALID
    elif close_ready:
        tier = TIER_CLOSE_READY
    elif micro_ready:
        tier = TIER_MICRO_ENTRY
    elif near_shell:
        tier = TIER_NEAR_BASIN
    elif coarse_candidate:
        tier = TIER_COARSE_PULLBACK
    elif outer_candidate:
        tier = TIER_OUTER_PULLBACK
    else:
        tier = TIER_OUTSIDE

    axis_policy = {
        "x": "trusted_control" if pullback_allowed and residual.finite else "abstain",
        "y": "trusted_control" if pullback_allowed and residual.finite else "abstain",
        "z": "diagnostic_only" if pullback_allowed and residual.finite else "abstain",
        "yaw": "trusted_control" if pullback_allowed and yaw_control_observable and residual.finite else "abstain",
    }

    return TakeoverTierDecision(
        takeover_tier=tier,
        pullback_allowed=bool(pullback_allowed),
        yaw_entry_feasible=bool(yaw_entry_feasible),
        yaw_control_observable=bool(yaw_control_observable),
        near_basin_shell=bool(near_shell),
        coarse_pullback_candidate=bool(coarse_candidate),
        outer_pullback_candidate=bool(outer_candidate),
        micro_entry_ready=bool(micro_ready),
        close_ready_ready=bool(close_ready),
        yaw_entry_block_reason=str(yaw_entry_block_reason),
        yaw_control_block_reason=str(yaw_control_block_reason),
        micro_entry_block_reason="+".join(micro_blocks) if micro_blocks else "ready",
        close_ready_block_reason="+".join(close_blocks) if close_blocks else "ready",
        axis_gate_policy=axis_policy,
    )
