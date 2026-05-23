"""Skill-aware force/contact controllers for Coarse2Contact v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from prismatic.robot.residual_safety import ResidualSafety

from .localizers import LocalGeometryError
from .specs import PrecisionSkillSpec


class RecoveryPhase(str, Enum):
    IDLE = "IDLE"
    BACKOFF = "BACKOFF"
    UNLOAD = "UNLOAD"
    MICRO_SEARCH = "MICRO_SEARCH"
    REAPPROACH = "REAPPROACH"
    FAILED = "FAILED"


@dataclass
class SkillControlResult:
    delta_local: np.ndarray
    state_name: str
    primitive: str
    reason: str
    gripper_override: Optional[float] = None
    contact_confirmed: bool = False
    stable: bool = False
    active: bool = True
    recovery_cycle_id: int = 0


def _force_vector(force_reading: Any) -> np.ndarray:
    if force_reading is None:
        return np.zeros(6, dtype=np.float32)
    arr = np.asarray(force_reading, dtype=np.float32).reshape(-1)[:6]
    if arr.size < 6:
        arr = np.pad(arr, (0, 6 - arr.size), constant_values=0.0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class RecoveryFSM:
    def __init__(
        self,
        *,
        safety: ResidualSafety,
        backoff_m: float = 0.003,
        lateral_m: float = 0.0015,
        unload_m: float = 0.0010,
        yaw_rad: float = 0.035,
        max_cycles: int = 4,
    ) -> None:
        self.safety = safety
        self.backoff_m = float(backoff_m)
        self.lateral_m = float(lateral_m)
        self.unload_m = float(unload_m)
        self.yaw_rad = float(yaw_rad)
        self.max_cycles = int(max_cycles)
        self.reset()

    def reset(self) -> None:
        self.phase = RecoveryPhase.IDLE
        self.phase_age = 0
        self.reason = "idle"
        self.cycle_id = 0
        self.last_primitive = "none"
        self.invalid_count = 0
        self.jam_count = 0

    def _enter(self, phase: RecoveryPhase, reason: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_age = 0
        self.reason = str(reason)

    def trigger(self, *, force_reading: Any, invalid_action: bool = False, jam: bool = False) -> bool:
        force_vec = _force_vector(force_reading)
        force_norm = float(np.linalg.norm(force_vec[:3]))
        torque_norm = float(np.linalg.norm(force_vec[3:6]))
        force_stop = bool(self.safety.check_force_stop(force_vec))

        if invalid_action:
            self.invalid_count += 1
            self.cycle_id = min(self.cycle_id + 1, self.max_cycles)
            self._enter(RecoveryPhase.BACKOFF, "invalid_action")
            return True
        if jam or force_stop or torque_norm > self.safety.torque_threshold or force_norm > self.safety.backoff_force_threshold:
            self.jam_count += 1
            self.cycle_id = min(self.cycle_id + 1, self.max_cycles)
            self._enter(RecoveryPhase.BACKOFF, "jam_or_force_stop")
            return True
        if self.phase != RecoveryPhase.IDLE:
            return True
        return False

    def step(self, *, force_reading: Any, local_base: np.ndarray, invalid_action: bool = False, jam: bool = False) -> SkillControlResult:
        self.trigger(force_reading=force_reading, invalid_action=invalid_action, jam=jam)
        force_vec = _force_vector(force_reading)
        force_norm = float(np.linalg.norm(force_vec[:3]))
        base = np.asarray(local_base, dtype=np.float32).reshape(-1)[:6]

        if self.phase == RecoveryPhase.IDLE:
            self.last_primitive = "none"
            return SkillControlResult(np.zeros(6, dtype=np.float32), self.phase.value, "none", "idle", active=False, recovery_cycle_id=self.cycle_id)

        if self.phase == RecoveryPhase.BACKOFF:
            delta = np.zeros(6, dtype=np.float32)
            delta[2] = self.backoff_m
            delta[0] = self.lateral_m if (self.cycle_id % 2 == 1) else -self.lateral_m
            self.last_primitive = "backoff"
            if self.phase_age >= 1 and force_norm <= self.safety.backoff_force_threshold:
                self._enter(RecoveryPhase.UNLOAD, "backoff_complete")
            else:
                self.phase_age += 1
            return SkillControlResult(delta, self.phase.value, self.last_primitive, self.reason, active=True, recovery_cycle_id=self.cycle_id)

        if self.phase == RecoveryPhase.UNLOAD:
            delta = np.zeros(6, dtype=np.float32)
            delta[2] = self.unload_m
            self.last_primitive = "unload"
            if self.phase_age >= 1 and force_norm <= self.safety.backoff_force_threshold * 0.5:
                self._enter(RecoveryPhase.MICRO_SEARCH, "unload_complete")
            else:
                self.phase_age += 1
            return SkillControlResult(delta, self.phase.value, self.last_primitive, self.reason, active=True, recovery_cycle_id=self.cycle_id)

        if self.phase == RecoveryPhase.MICRO_SEARCH:
            delta = np.zeros(6, dtype=np.float32)
            delta[0] = self.lateral_m if (self.cycle_id % 2 == 0) else -self.lateral_m
            delta[1] = -0.5 * delta[0]
            delta[5] = self.yaw_rad if (self.cycle_id % 2 == 0) else -self.yaw_rad
            self.last_primitive = "micro_search"
            if self.phase_age >= 2:
                self._enter(RecoveryPhase.REAPPROACH, "micro_search_complete")
            else:
                self.phase_age += 1
            return SkillControlResult(delta, self.phase.value, self.last_primitive, self.reason, active=True, recovery_cycle_id=self.cycle_id)

        if self.phase == RecoveryPhase.REAPPROACH:
            delta = np.zeros(6, dtype=np.float32)
            delta[2] = -0.5 * self.backoff_m
            self.last_primitive = "reapproach"
            if self.phase_age >= 1:
                self._enter(RecoveryPhase.IDLE, "recovery_complete")
            else:
                self.phase_age += 1
            return SkillControlResult(delta, self.phase.value, self.last_primitive, self.reason, active=True, recovery_cycle_id=self.cycle_id)

        self.phase = RecoveryPhase.FAILED
        self.last_primitive = "failed"
        return SkillControlResult(np.zeros(6, dtype=np.float32), self.phase.value, "failed", "failed", active=False, recovery_cycle_id=self.cycle_id)


class GraspContactController:
    def __init__(
        self,
        *,
        safety: ResidualSafety,
        contact_threshold: float = 0.18,
        stable_force_threshold: float = 0.12,
        yaw_relax_threshold: float = 0.20,
        xy_relax_threshold: float = 0.015,
        max_xy_step: float = 0.0010,
        max_yaw_step: float = 0.035,
    ) -> None:
        self.safety = safety
        self.contact_threshold = float(contact_threshold)
        self.stable_force_threshold = float(stable_force_threshold)
        self.yaw_relax_threshold = float(yaw_relax_threshold)
        self.xy_relax_threshold = float(xy_relax_threshold)
        self.max_xy_step = float(max_xy_step)
        self.max_yaw_step = float(max_yaw_step)
        self.recovery = RecoveryFSM(safety=safety, backoff_m=0.0025, lateral_m=0.0012, unload_m=0.0010, yaw_rad=max_yaw_step)
        self.reset()

    def reset(self) -> None:
        self.state = "SEARCH"
        self.state_age = 0
        self.contact_confirmed = False
        self.stable_grasp = False
        self.recovery.reset()

    def step(
        self,
        *,
        error: LocalGeometryError,
        force_reading: Any,
        local_base: np.ndarray,
        gripper_open: float,
        invalid_action: bool = False,
    ) -> SkillControlResult:
        force_vec = _force_vector(force_reading)
        force_norm = float(np.linalg.norm(force_vec[:3]))
        if self.recovery.trigger(force_reading=force_vec, invalid_action=invalid_action, jam=force_norm > self.safety.force_stop_threshold):
            self.state = "REGRASP"
            rec = self.recovery.step(force_reading=force_vec, local_base=local_base, invalid_action=invalid_action, jam=force_norm > self.safety.force_stop_threshold)
            return SkillControlResult(rec.delta_local, rec.state_name, rec.primitive, rec.reason, gripper_override=0.0, contact_confirmed=False, stable=False, active=True, recovery_cycle_id=rec.recovery_cycle_id)

        delta = np.zeros(6, dtype=np.float32)
        if error.valid and error.confidence >= self.xy_relax_threshold:
            delta[0] = float(np.clip(-error.dx, -self.max_xy_step, self.max_xy_step))
            delta[1] = float(np.clip(-error.dy, -self.max_xy_step, self.max_xy_step))
            delta[5] = float(np.clip(-0.7 * error.dyaw, -self.max_yaw_step, self.max_yaw_step))
            self.state = "ALIGN"
        else:
            self.state = "SEARCH"

        if error.valid and error.confidence >= self.contact_threshold and force_norm <= self.contact_threshold:
            gripper_override = 0.0
            self.contact_confirmed = True
            self.state = "CLOSE"
        elif force_norm > self.contact_threshold:
            self.contact_confirmed = True
            self.state = "CONTACT"
            gripper_override = 0.0
        else:
            gripper_override = None if gripper_open < 0.5 else 1.0

        self.stable_grasp = bool(self.contact_confirmed and force_norm <= self.stable_force_threshold and gripper_open < 0.5)
        if self.stable_grasp:
            self.state = "VERIFY"

        return SkillControlResult(
            delta_local=delta,
            state_name=self.state,
            primitive="grasp_adjust" if np.linalg.norm(delta[:3]) > 0 else "hold",
            reason="ok" if error.valid else error.reason,
            gripper_override=gripper_override,
            contact_confirmed=self.contact_confirmed,
            stable=self.stable_grasp,
            active=True,
            recovery_cycle_id=self.recovery.cycle_id,
        )


class GuardedSlideController:
    def __init__(
        self,
        *,
        safety: ResidualSafety,
        contact_threshold: float = 0.18,
        jam_threshold: float = 0.55,
        torque_threshold: float = 0.12,
        max_xy_step: float = 0.0010,
        max_yaw_step: float = 0.035,
        max_z_step: float = 0.0015,
    ) -> None:
        self.safety = safety
        self.contact_threshold = float(contact_threshold)
        self.jam_threshold = float(jam_threshold)
        self.torque_threshold = float(torque_threshold)
        self.max_xy_step = float(max_xy_step)
        self.max_yaw_step = float(max_yaw_step)
        self.max_z_step = float(max_z_step)
        self.recovery = RecoveryFSM(safety=safety, backoff_m=0.0030, lateral_m=0.0015, unload_m=0.0010, yaw_rad=max_yaw_step)
        self.reset()

    def reset(self) -> None:
        self.state = "FREE"
        self.state_age = 0
        self.jam_detected = False
        self.first_touch = False
        self.recovery.reset()

    def step(
        self,
        *,
        error: LocalGeometryError,
        force_reading: Any,
        local_base: np.ndarray,
        invalid_action: bool = False,
    ) -> SkillControlResult:
        force_vec = _force_vector(force_reading)
        force_norm = float(np.linalg.norm(force_vec[:3]))
        lateral = float(np.linalg.norm(force_vec[:2]))
        torque = float(np.linalg.norm(force_vec[3:5]))
        jam = bool(
            force_norm > self.jam_threshold
            or torque > self.torque_threshold
            or self.safety.check_force_stop(force_vec)
            or invalid_action
        )
        if jam:
            self.jam_detected = True
            self.state = "RECOVER"
            rec = self.recovery.step(force_reading=force_vec, local_base=local_base, invalid_action=invalid_action, jam=True)
            return SkillControlResult(rec.delta_local, rec.state_name, rec.primitive, rec.reason, gripper_override=0.0, contact_confirmed=False, stable=False, active=True, recovery_cycle_id=rec.recovery_cycle_id)

        delta = np.zeros(6, dtype=np.float32)
        if error.valid and error.confidence >= 0.25:
            delta[0] = float(np.clip(-error.dx, -self.max_xy_step, self.max_xy_step))
            delta[1] = float(np.clip(-error.dy, -self.max_xy_step, self.max_xy_step))
            delta[5] = float(np.clip(-0.5 * error.dyaw, -self.max_yaw_step, self.max_yaw_step))
        if error.valid and error.confidence >= 0.15 and (force_norm <= self.contact_threshold or np.isnan(force_norm)):
            delta[2] = float(np.clip(-0.5 * error.dz, -self.max_z_step, self.max_z_step))

        if force_norm > self.contact_threshold:
            self.first_touch = True
            self.state = "SLIDE"
        elif self.first_touch:
            self.state = "FIRST_TOUCH"
        else:
            self.state = "FREE"

        primitive = "slide_guard" if np.linalg.norm(delta[:3]) > 0 else "hold"
        reason = "ok" if error.valid else error.reason
        return SkillControlResult(
            delta_local=delta,
            state_name=self.state,
            primitive=primitive,
            reason=reason,
            gripper_override=None,
            contact_confirmed=self.first_touch,
            stable=force_norm <= self.contact_threshold,
            active=True,
            recovery_cycle_id=self.recovery.cycle_id,
        )
