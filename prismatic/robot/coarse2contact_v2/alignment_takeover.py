"""Task-frame alignment takeover lifecycle for C2C v2.

The lifecycle separates precision alignment from gripper closing: C2C may allow
planner gripper handoff only after the task-frame alignment predicate is ready.
If alignment cannot be established, C2C exits through an explicit safe/failed
terminal state while keeping close handoff blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np


STATE_IDLE = "IDLE"
STATE_TAKEOVER_ACTIVE = "TAKEOVER_ACTIVE"
STATE_ALIGNING_XY = "ALIGNING_XY"
STATE_ALIGNING_Z = "ALIGNING_Z"
STATE_ALIGNING_YAW = "ALIGNING_YAW"
STATE_ALIGNMENT_READY_FOR_HANDOFF = "ALIGNMENT_READY_FOR_HANDOFF"
STATE_HANDOFF_TO_PLANNER_GRIPPER = "HANDOFF_TO_PLANNER_GRIPPER"
STATE_REACQUIRE_VIEW = "REACQUIRE_VIEW"
STATE_SAFE_ABSTAIN_OPEN = "SAFE_ABSTAIN_OPEN"
STATE_FAILED_RETRYABLE = "FAILED_RETRYABLE"
STATE_FAILED_TERMINAL = "FAILED_TERMINAL"

TERMINAL_STATES = {
    STATE_ALIGNMENT_READY_FOR_HANDOFF,
    STATE_HANDOFF_TO_PLANNER_GRIPPER,
    STATE_SAFE_ABSTAIN_OPEN,
    STATE_FAILED_RETRYABLE,
    STATE_FAILED_TERMINAL,
}


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
    if isinstance(value, (float, int, np.floating, np.integer)):
        return bool(float(value) > 0.5)
    return bool(value)


@dataclass(frozen=True)
class TaskFrameResidualEstimate:
    """Non-privileged task-frame residual contract shared by grasp/align skills."""

    skill_id: str
    stage_name: str
    reference_frame: str
    target_frame: str
    active_dofs: tuple[str, ...]
    dx: float
    dy: float
    dz: float
    dyaw: float
    axis_validity: dict[str, bool]
    axis_confidence: dict[str, float]
    observability: float
    frame_consistency: float
    abstain_reason: str = ""
    z_semantics: str = "task_approach_axis_residual"
    yaw_semantics: str = "task_frame_yaw_residual"
    source: str = "runtime_task_frame_estimate"
    uses_privileged_runtime: bool = False

    @property
    def xy_norm(self) -> float:
        return float(np.hypot(float(self.dx), float(self.dy)))

    def axis_ready(self, axis: str, *, min_confidence: float = 0.0) -> bool:
        return bool(
            self.axis_validity.get(str(axis), False)
            and float(self.axis_confidence.get(str(axis), 0.0)) >= float(min_confidence)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "c2c_v2_task_frame_residual_estimate_v1",
            "skill_id": str(self.skill_id),
            "stage_name": str(self.stage_name),
            "reference_frame": str(self.reference_frame),
            "target_frame": str(self.target_frame),
            "active_dofs": list(self.active_dofs),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "dz": float(self.dz),
            "dyaw": float(self.dyaw),
            "xy_norm": float(self.xy_norm),
            "axis_validity": {str(k): bool(v) for k, v in self.axis_validity.items()},
            "axis_confidence": {str(k): float(v) for k, v in self.axis_confidence.items()},
            "observability": float(self.observability),
            "frame_consistency": float(self.frame_consistency),
            "abstain_reason": str(self.abstain_reason),
            "z_semantics": str(self.z_semantics),
            "yaw_semantics": str(self.yaw_semantics),
            "source": str(self.source),
            "uses_privileged_runtime": bool(self.uses_privileged_runtime),
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "TaskFrameResidualEstimate":
        active = row.get("active_dofs", ())
        if isinstance(active, str):
            active_dofs = tuple(part.strip() for part in active.split(",") if part.strip())
        else:
            active_dofs = tuple(str(part) for part in active)
        axis_validity = row.get("axis_validity", {})
        axis_confidence = row.get("axis_confidence", {})
        return cls(
            skill_id=str(row.get("skill_id", "")),
            stage_name=str(row.get("stage_name", "")),
            reference_frame=str(row.get("reference_frame", "")),
            target_frame=str(row.get("target_frame", "")),
            active_dofs=active_dofs,
            dx=_as_float(row.get("dx"), 0.0),
            dy=_as_float(row.get("dy"), 0.0),
            dz=_as_float(row.get("dz"), 0.0),
            dyaw=_as_float(row.get("dyaw"), 0.0),
            axis_validity={str(k): _as_bool(v, False) for k, v in dict(axis_validity).items()},
            axis_confidence={str(k): _as_float(v, 0.0) for k, v in dict(axis_confidence).items()},
            observability=_as_float(row.get("observability"), 0.0),
            frame_consistency=_as_float(row.get("frame_consistency"), 0.0),
            abstain_reason=str(row.get("abstain_reason", "")),
            z_semantics=str(row.get("z_semantics", "task_approach_axis_residual")),
            yaw_semantics=str(row.get("yaw_semantics", "task_frame_yaw_residual")),
            source=str(row.get("source", "runtime_task_frame_estimate")),
            uses_privileged_runtime=_as_bool(row.get("uses_privileged_runtime"), False),
        )


@dataclass(frozen=True)
class AlignmentReadiness:
    alignment_ready_for_handoff: bool
    block_reason: str
    xy_ready: bool
    z_ready: bool
    yaw_ready: bool
    observability_ready: bool
    frame_consistency_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignment_ready_for_handoff": bool(self.alignment_ready_for_handoff),
            "alignment_handoff_block_reason": str(self.block_reason),
            "alignment_xy_ready": bool(self.xy_ready),
            "alignment_z_ready": bool(self.z_ready),
            "alignment_yaw_ready": bool(self.yaw_ready),
            "alignment_observability_ready": bool(self.observability_ready),
            "alignment_frame_consistency_ready": bool(self.frame_consistency_ready),
        }


@dataclass(frozen=True)
class AlignmentTakeoverConfig:
    max_control_steps: int = 24
    max_retries: int = 2
    reacquire_steps: int = 8
    xy_threshold: float = 0.005
    z_threshold: float = 0.020
    yaw_threshold: float = 0.030
    min_observability: float = 5.0e-4
    min_frame_consistency: float = 0.20
    z_required: bool = True
    yaw_required: bool = True


@dataclass(frozen=True)
class AlignmentTakeoverSession:
    session_id: int = 0
    state: str = STATE_IDLE
    terminal_state: str = ""
    exit_reason: str = ""
    retry_count: int = 0
    budget_used: int = 0
    reacquire_used: int = 0
    alignment_ready_for_handoff: bool = False
    safe_abstain_open: bool = False
    failed_retryable: bool = False
    failed_terminal: bool = False
    final_axis_readiness: dict[str, bool] | None = None

    @property
    def active(self) -> bool:
        return bool(self.state not in {STATE_IDLE, *TERMINAL_STATES})

    @property
    def terminal(self) -> bool:
        return bool(self.state in TERMINAL_STATES or self.terminal_state)

    def begin(self, session_id: int) -> "AlignmentTakeoverSession":
        return replace(
            self,
            session_id=int(session_id),
            state=STATE_TAKEOVER_ACTIVE,
            terminal_state="",
            exit_reason="",
            retry_count=0,
            budget_used=0,
            reacquire_used=0,
            alignment_ready_for_handoff=False,
            safe_abstain_open=False,
            failed_retryable=False,
            failed_terminal=False,
            final_axis_readiness=None,
        )

    def update(
        self,
        *,
        eligible_now: bool,
        visual_ready: bool,
        readiness: AlignmentReadiness,
        config: AlignmentTakeoverConfig,
    ) -> "AlignmentTakeoverSession":
        if not self.active:
            return self
        budget = int(self.budget_used) + 1
        reacquire = int(self.reacquire_used)
        retry = int(self.retry_count)
        axis = {
            "xy": bool(readiness.xy_ready),
            "z": bool(readiness.z_ready),
            "yaw": bool(readiness.yaw_ready),
            "observability": bool(readiness.observability_ready),
            "frame_consistency": bool(readiness.frame_consistency_ready),
        }
        if bool(readiness.alignment_ready_for_handoff):
            return replace(
                self,
                state=STATE_ALIGNMENT_READY_FOR_HANDOFF,
                terminal_state=STATE_ALIGNMENT_READY_FOR_HANDOFF,
                exit_reason="alignment_ready_for_handoff",
                budget_used=budget,
                alignment_ready_for_handoff=True,
                final_axis_readiness=axis,
            )
        if not bool(visual_ready):
            reacquire += 1
            if reacquire > int(config.reacquire_steps):
                return replace(
                    self,
                    state=STATE_SAFE_ABSTAIN_OPEN,
                    terminal_state=STATE_SAFE_ABSTAIN_OPEN,
                    exit_reason="reacquire_needed",
                    budget_used=budget,
                    reacquire_used=reacquire,
                    safe_abstain_open=True,
                    final_axis_readiness=axis,
                )
            return replace(
                self,
                state=STATE_REACQUIRE_VIEW,
                budget_used=budget,
                reacquire_used=reacquire,
                final_axis_readiness=axis,
            )
        if budget >= int(config.max_control_steps):
            retry += 1
            if retry < int(config.max_retries):
                return replace(
                    self,
                    state=STATE_FAILED_RETRYABLE,
                    terminal_state=STATE_FAILED_RETRYABLE,
                    exit_reason="budget_exhausted_retryable",
                    retry_count=retry,
                    budget_used=budget,
                    failed_retryable=True,
                    final_axis_readiness=axis,
                )
            return replace(
                self,
                state=STATE_FAILED_TERMINAL,
                terminal_state=STATE_FAILED_TERMINAL,
                exit_reason="budget_exhausted",
                retry_count=retry,
                budget_used=budget,
                failed_terminal=True,
                final_axis_readiness=axis,
            )
        if bool(eligible_now):
            if not readiness.xy_ready:
                state = STATE_ALIGNING_XY
            elif not readiness.z_ready:
                state = STATE_ALIGNING_Z
            elif not readiness.yaw_ready:
                state = STATE_ALIGNING_YAW
            else:
                state = STATE_TAKEOVER_ACTIVE
        else:
            state = STATE_TAKEOVER_ACTIVE
        return replace(
            self,
            state=state,
            budget_used=budget,
            reacquire_used=0,
            final_axis_readiness=axis,
        )

    def to_trace(self) -> dict[str, Any]:
        return {
            "takeover_session_id": int(self.session_id),
            "takeover_lifecycle_state": str(self.state),
            "terminal_state": str(self.terminal_state),
            "exit_reason": str(self.exit_reason),
            "alignment_ready_for_handoff": bool(self.alignment_ready_for_handoff),
            "safe_abstain_open": bool(self.safe_abstain_open),
            "failed_retryable": bool(self.failed_retryable),
            "failed_terminal": bool(self.failed_terminal),
            "retry_count": int(self.retry_count),
            "budget_used": int(self.budget_used),
            "reacquire_used": int(self.reacquire_used),
            "final_axis_readiness": dict(self.final_axis_readiness or {}),
        }


def evaluate_alignment_readiness(
    residual: TaskFrameResidualEstimate,
    config: AlignmentTakeoverConfig,
) -> AlignmentReadiness:
    xy_ready = bool(
        residual.axis_ready("x")
        and residual.axis_ready("y")
        and np.isfinite(residual.xy_norm)
        and residual.xy_norm <= float(config.xy_threshold)
    )
    z_ready = bool(
        (not bool(config.z_required))
        or (
            residual.axis_ready("z")
            and np.isfinite(float(residual.dz))
            and abs(float(residual.dz)) <= float(config.z_threshold)
        )
    )
    yaw_ready = bool(
        (not bool(config.yaw_required))
        or (
            residual.axis_ready("yaw")
            and np.isfinite(float(residual.dyaw))
            and abs(float(residual.dyaw)) <= float(config.yaw_threshold)
        )
    )
    obs_ready = bool(float(residual.observability) >= float(config.min_observability))
    frame_ready = bool(float(residual.frame_consistency) >= float(config.min_frame_consistency))
    blocks: list[str] = []
    if not xy_ready:
        blocks.append("xy_not_ready")
    if not z_ready:
        blocks.append("z_not_ready")
    if not yaw_ready:
        blocks.append("yaw_not_ready")
    if not obs_ready:
        blocks.append("low_observability")
    if not frame_ready:
        blocks.append("frame_inconsistent")
    ready = bool(xy_ready and z_ready and yaw_ready and obs_ready and frame_ready and not residual.abstain_reason)
    if residual.abstain_reason:
        blocks.append(str(residual.abstain_reason))
    return AlignmentReadiness(
        alignment_ready_for_handoff=ready,
        block_reason="+".join(blocks) if blocks else "ready",
        xy_ready=xy_ready,
        z_ready=z_ready,
        yaw_ready=yaw_ready,
        observability_ready=obs_ready,
        frame_consistency_ready=frame_ready,
    )
