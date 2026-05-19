"""
contact_trigger.py

Detects contact phase (free-space / pre-contact / contact) from force and depth observations
with hysteresis and minimum dwell time.
"""

from collections import deque
from enum import IntEnum
from typing import Optional

import numpy as np


class ContactPhase(IntEnum):
    FREE_SPACE = 0
    PRE_CONTACT = 1
    CONTACT = 2


class ContactTrigger:
    """Rule-based contact phase detector using force magnitude and wrist depth proximity."""

    def __init__(
        self,
        force_contact_threshold: float = 0.5,
        force_jam_threshold: float = 3.0,
        depth_proximity_threshold: float = 0.15,
        z_pre_contact: float = 0.90,
        force_contact_exit_threshold: Optional[float] = None,
        depth_proximity_exit_threshold: Optional[float] = None,
        z_pre_contact_exit: Optional[float] = None,
        pre_contact_chunk_k: int = 2,
        contact_chunk_k: int = 1,
        free_chunk_k: int = 8,
        jam_window: int = 10,
        jam_count_threshold: int = 5,
        min_phase_steps: int = 2,
    ):
        self.force_contact_threshold = force_contact_threshold
        self.force_jam_threshold = force_jam_threshold
        self.depth_proximity_threshold = depth_proximity_threshold
        self.z_pre_contact = z_pre_contact
        self.force_contact_exit_threshold = (
            force_contact_exit_threshold
            if force_contact_exit_threshold is not None
            else force_contact_threshold * 0.65
        )
        self.depth_proximity_exit_threshold = (
            depth_proximity_exit_threshold
            if depth_proximity_exit_threshold is not None
            else depth_proximity_threshold * 1.25
        )
        self.z_pre_contact_exit = (
            z_pre_contact_exit if z_pre_contact_exit is not None else z_pre_contact + 0.02
        )
        self.pre_contact_chunk_k = pre_contact_chunk_k
        self.contact_chunk_k = contact_chunk_k
        self.free_chunk_k = free_chunk_k
        self.jam_window = jam_window
        self.jam_count_threshold = jam_count_threshold
        self.min_phase_steps = min_phase_steps

        self._force_history = deque(maxlen=jam_window)
        self._phase = ContactPhase.FREE_SPACE
        self._jam_detected = False
        self._phase_steps = 0

    @property
    def phase(self) -> ContactPhase:
        return self._phase

    @property
    def jam_detected(self) -> bool:
        return self._jam_detected

    @property
    def phase_age(self) -> int:
        return self._phase_steps

    def update(
        self,
        force_reading: Optional[np.ndarray] = None,
        gripper_z: Optional[float] = None,
        depth_proximity: Optional[float] = None,
    ) -> ContactPhase:
        force_mag = 0.0
        if force_reading is not None:
            force_mag = float(np.linalg.norm(force_reading))
            self._force_history.append(force_mag)

        self._jam_detected = False
        if len(self._force_history) >= self.jam_window:
            high_count = sum(1 for f in self._force_history if f > self.force_jam_threshold)
            self._jam_detected = high_count >= self.jam_count_threshold

        depth_near_enter = (
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and depth_proximity < self.depth_proximity_threshold
        )
        depth_near_exit = (
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and depth_proximity < self.depth_proximity_exit_threshold
        )
        z_near_enter = gripper_z is not None and gripper_z < self.z_pre_contact
        z_near_exit = gripper_z is not None and gripper_z < self.z_pre_contact_exit

        next_phase = self._phase
        if self._phase == ContactPhase.CONTACT:
            if force_mag > self.force_contact_exit_threshold:
                next_phase = ContactPhase.CONTACT
            elif depth_near_enter or z_near_enter:
                next_phase = ContactPhase.PRE_CONTACT
            else:
                next_phase = ContactPhase.FREE_SPACE
        elif self._phase == ContactPhase.PRE_CONTACT:
            if force_mag > self.force_contact_threshold:
                next_phase = ContactPhase.CONTACT
            elif depth_near_exit or z_near_exit:
                next_phase = ContactPhase.PRE_CONTACT
            else:
                next_phase = ContactPhase.FREE_SPACE
        else:
            if force_mag > self.force_contact_threshold:
                next_phase = ContactPhase.CONTACT
            elif depth_near_enter or z_near_enter:
                next_phase = ContactPhase.PRE_CONTACT
            else:
                next_phase = ContactPhase.FREE_SPACE

        if next_phase != self._phase and self._phase_steps < self.min_phase_steps:
            if next_phase < self._phase:
                next_phase = self._phase

        if next_phase == self._phase:
            self._phase_steps += 1
        else:
            self._phase = next_phase
            self._phase_steps = 1

        return self._phase

    def get_chunk_size(self) -> int:
        if self._phase == ContactPhase.CONTACT:
            return self.contact_chunk_k
        if self._phase == ContactPhase.PRE_CONTACT:
            return self.pre_contact_chunk_k
        return self.free_chunk_k

    def should_replan(self) -> bool:
        return self._jam_detected or self._phase == ContactPhase.CONTACT

    def reset(self):
        self._force_history.clear()
        self._phase = ContactPhase.FREE_SPACE
        self._jam_detected = False
        self._phase_steps = 0
