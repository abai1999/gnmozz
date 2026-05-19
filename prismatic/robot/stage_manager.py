"""
stage_manager.py

Task-agnostic rule-based interaction stage manager for frozen chunk planners.
"""

from collections import deque
from enum import IntEnum
from typing import Optional

import numpy as np


class StagePhase(IntEnum):
    TRANSIT = 0
    ALIGN = 1
    INTERACT = 2
    RECOVER = 3


class StageSubgoal(IntEnum):
    TRANSIT = 0
    ALIGN_PREGRASP = 1
    ALIGN_HELD = 2
    INTERACT_GUARDED = 3
    RECOVER = 4


class ContactState(IntEnum):
    FREE_SPACE = 0
    NEAR_CONTACT = 1
    IN_CONTACT = 2
    OVERLOAD = 3


class StageTargetMode(IntEnum):
    NONE = 0
    PREGRASP_OBJECT = 1
    HELD_RECEPTACLE = 2
    CONTACT_GUARDED = 3
    RECOVERY_CANONICAL = 4


class FailureMode(IntEnum):
    NONE = 0
    NO_PROGRESS = 1
    FORCE_OVERLOAD = 2
    TORQUE_OVERLOAD = 3
    INVALID_ACTION = 4


class StageManager:
    """Generic phase manager for multi-stage manipulation tasks."""

    def __init__(
        self,
        depth_align_enter: float = 0.18,
        depth_align_exit: Optional[float] = None,
        force_interact_enter: float = 0.5,
        force_interact_exit: Optional[float] = None,
        force_recover_threshold: float = 3.0,
        torque_recover_threshold: float = 1.0,
        align_action_norm_threshold: float = 0.04,
        movement_progress_epsilon: float = 0.001,
        no_progress_window: int = 6,
        min_phase_steps: int = 2,
        transit_chunk_k: int = 8,
        align_chunk_k: int = 2,
        interact_chunk_k: int = 1,
        recover_chunk_k: int = 1,
    ):
        self.depth_align_enter = depth_align_enter
        self.depth_align_exit = depth_align_exit if depth_align_exit is not None else depth_align_enter * 1.3
        self.force_interact_enter = force_interact_enter
        self.force_interact_exit = (
            force_interact_exit if force_interact_exit is not None else force_interact_enter * 0.6
        )
        self.force_recover_threshold = force_recover_threshold
        self.torque_recover_threshold = torque_recover_threshold
        self.align_action_norm_threshold = align_action_norm_threshold
        self.movement_progress_epsilon = movement_progress_epsilon
        self.no_progress_window = no_progress_window
        self.min_phase_steps = min_phase_steps
        self.transit_chunk_k = transit_chunk_k
        self.align_chunk_k = align_chunk_k
        self.interact_chunk_k = interact_chunk_k
        self.recover_chunk_k = recover_chunk_k

        self._force_history = deque(maxlen=max(8, no_progress_window))
        self._phase = StagePhase.TRANSIT
        self._phase_steps = 0
        self._no_progress_steps = 0
        self._steps_since_last_replan = 0
        self._transition_count = 0
        self._max_phase_reached = StagePhase.TRANSIT
        self._last_ee_pos = None
        self._last_gripper_open = None
        self._failure_mode = FailureMode.NONE
        self._last_transitioned = False
        self._contact_state = ContactState.FREE_SPACE
        self._has_object_in_hand = False
        self._held_object_steps = 0
        self._substage = StageSubgoal.TRANSIT
        self._stage_target_mode = StageTargetMode.NONE

    @property
    def phase(self) -> StagePhase:
        return self._phase

    @property
    def phase_age(self) -> int:
        return self._phase_steps

    @property
    def no_progress_steps(self) -> int:
        return self._no_progress_steps

    @property
    def steps_since_last_replan(self) -> int:
        return self._steps_since_last_replan

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def max_phase_reached(self) -> int:
        return int(self._max_phase_reached)

    @property
    def failure_mode(self) -> FailureMode:
        return self._failure_mode

    @property
    def last_transitioned(self) -> bool:
        return self._last_transitioned

    @property
    def contact_state(self) -> ContactState:
        return self._contact_state

    @property
    def has_object_in_hand(self) -> bool:
        return self._has_object_in_hand

    @property
    def substage(self) -> StageSubgoal:
        return self._substage

    @property
    def stage_target_mode(self) -> StageTargetMode:
        return self._stage_target_mode

    def get_stage_role(self) -> int:
        return 0 if self._phase in (StagePhase.TRANSIT, StagePhase.ALIGN) else 1

    def get_subgoal_progress(self) -> float:
        stability = min(1.0, float(self._phase_steps) / float(max(self.min_phase_steps, 1)))
        phase_extent = max(int(self._max_phase_reached), int(self._phase))
        return (float(phase_extent) + 0.5 * stability) / (float(int(StagePhase.RECOVER)) + 0.5)

    def get_failure_mode_name(self) -> str:
        return self._failure_mode.name

    def update(
        self,
        force_reading: Optional[np.ndarray] = None,
        gripper_pose: Optional[np.ndarray] = None,
        gripper_open: Optional[float] = None,
        depth_proximity: Optional[float] = None,
        base_action: Optional[np.ndarray] = None,
        just_replanned: bool = False,
    ) -> StagePhase:
        force_xyz = np.zeros(3, dtype=np.float32)
        torque_xyz = np.zeros(3, dtype=np.float32)
        if force_reading is not None:
            force_arr = np.asarray(force_reading, dtype=np.float32)
            force_xyz = force_arr[:3]
            torque_xyz = force_arr[3:6] if force_arr.shape[0] >= 6 else torque_xyz
        force_mag = float(np.linalg.norm(force_xyz))
        force_lateral = float(np.linalg.norm(force_xyz[:2]))
        torque_mag = float(np.linalg.norm(torque_xyz))
        self._force_history.append(force_mag)

        action_norm = None
        if base_action is not None:
            action_norm = float(np.linalg.norm(np.asarray(base_action, dtype=np.float32)[:3]))

        moved = 0.0
        if gripper_pose is not None:
            ee_pos = np.asarray(gripper_pose[:3], dtype=np.float32)
            if self._last_ee_pos is not None:
                moved = float(np.linalg.norm(ee_pos - self._last_ee_pos))
            self._last_ee_pos = ee_pos

        if action_norm is not None and action_norm > self.align_action_norm_threshold * 0.25 and moved < self.movement_progress_epsilon:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0

        depth_align_enter = (
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and depth_proximity < self.depth_align_enter
        )
        depth_align_exit = (
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and depth_proximity < self.depth_align_exit
        )

        gripping = gripper_open is not None and float(gripper_open) < 0.5
        grip_changed = (
            self._last_gripper_open is not None
            and gripper_open is not None
            and abs(float(self._last_gripper_open) - float(gripper_open)) > 0.4
        )
        self._last_gripper_open = gripper_open

        stalled = self._no_progress_steps >= self.no_progress_window
        recover_signal = (
            force_mag > self.force_recover_threshold
            or torque_mag > self.torque_recover_threshold
            or (self._phase in (StagePhase.ALIGN, StagePhase.INTERACT) and stalled)
        )
        if torque_mag > self.torque_recover_threshold:
            self._failure_mode = FailureMode.TORQUE_OVERLOAD
        elif force_mag > self.force_recover_threshold:
            self._failure_mode = FailureMode.FORCE_OVERLOAD
        elif self._phase in (StagePhase.ALIGN, StagePhase.INTERACT) and stalled:
            self._failure_mode = FailureMode.NO_PROGRESS
        else:
            self._failure_mode = FailureMode.NONE

        if recover_signal:
            self._contact_state = ContactState.OVERLOAD
        elif force_mag > self.force_interact_enter:
            self._contact_state = ContactState.IN_CONTACT
        elif depth_align_enter or (gripping and depth_align_exit):
            self._contact_state = ContactState.NEAR_CONTACT
        else:
            self._contact_state = ContactState.FREE_SPACE

        held_object_signal = bool(
            gripping and (
                force_mag > self.force_interact_enter * 0.25
                or self._phase in (StagePhase.INTERACT, StagePhase.RECOVER)
                or (depth_align_enter and self._phase in (StagePhase.ALIGN, StagePhase.INTERACT))
            )
        )
        if held_object_signal:
            self._held_object_steps += 1
        elif not gripping:
            self._held_object_steps = 0
        else:
            self._held_object_steps = max(self._held_object_steps - 1, 0)
        self._has_object_in_hand = bool(self._held_object_steps >= 2)
        # A gripper command change alone is not evidence of successful interaction:
        # the planner can close early while still missing the object. Keep it as
        # context, but require contact-like force or sustained near-field gripping
        # before promoting to INTERACT.
        interact_signal = force_mag > self.force_interact_enter or (gripping and depth_align_enter)
        align_signal = depth_align_enter or grip_changed or (
            action_norm is not None and action_norm < self.align_action_norm_threshold
        )

        next_phase = self._phase
        if self._phase == StagePhase.RECOVER:
            if recover_signal:
                next_phase = StagePhase.RECOVER
            elif force_mag > self.force_interact_exit:
                next_phase = StagePhase.INTERACT
            elif depth_align_exit:
                next_phase = StagePhase.ALIGN
            else:
                next_phase = StagePhase.TRANSIT
        elif self._phase == StagePhase.INTERACT:
            if recover_signal:
                next_phase = StagePhase.RECOVER
            elif force_mag > self.force_interact_exit or (gripping and depth_align_exit):
                next_phase = StagePhase.INTERACT
            elif depth_align_exit:
                next_phase = StagePhase.ALIGN
            else:
                next_phase = StagePhase.TRANSIT
        elif self._phase == StagePhase.ALIGN:
            if recover_signal:
                next_phase = StagePhase.RECOVER
            elif interact_signal:
                next_phase = StagePhase.INTERACT
            elif align_signal or depth_align_exit:
                next_phase = StagePhase.ALIGN
            else:
                next_phase = StagePhase.TRANSIT
        else:
            if recover_signal:
                next_phase = StagePhase.RECOVER
            elif interact_signal:
                next_phase = StagePhase.INTERACT
            elif align_signal:
                next_phase = StagePhase.ALIGN
            else:
                next_phase = StagePhase.TRANSIT

        if next_phase != self._phase and self._phase_steps < self.min_phase_steps:
            next_phase = self._phase

        if next_phase == self._phase:
            self._phase_steps += 1
            self._last_transitioned = False
        else:
            self._phase = next_phase
            self._phase_steps = 1
            self._transition_count += 1
            self._last_transitioned = True

        self._max_phase_reached = max(self._max_phase_reached, self._phase)
        if just_replanned:
            self._steps_since_last_replan = 0
        else:
            self._steps_since_last_replan += 1
        if self._phase == StagePhase.RECOVER:
            self._substage = StageSubgoal.RECOVER
            self._stage_target_mode = StageTargetMode.RECOVERY_CANONICAL
        elif self._phase == StagePhase.INTERACT:
            self._substage = StageSubgoal.INTERACT_GUARDED
            self._stage_target_mode = StageTargetMode.CONTACT_GUARDED
        elif self._phase == StagePhase.ALIGN and self._has_object_in_hand:
            self._substage = StageSubgoal.ALIGN_HELD
            self._stage_target_mode = StageTargetMode.HELD_RECEPTACLE
        elif self._phase == StagePhase.ALIGN:
            self._substage = StageSubgoal.ALIGN_PREGRASP
            self._stage_target_mode = StageTargetMode.PREGRASP_OBJECT
        else:
            self._substage = StageSubgoal.TRANSIT
            self._stage_target_mode = StageTargetMode.NONE
        return self._phase

    def should_replan(self) -> bool:
        return self._phase == StagePhase.RECOVER

    def note_invalid_action(self) -> StagePhase:
        """Mark an execution failure as a recover-state hard negative."""
        self._failure_mode = FailureMode.INVALID_ACTION
        if self._phase != StagePhase.RECOVER:
            self._phase = StagePhase.RECOVER
            self._phase_steps = 1
            self._transition_count += 1
            self._last_transitioned = True
        else:
            self._phase_steps += 1
            self._last_transitioned = False
        self._max_phase_reached = max(self._max_phase_reached, self._phase)
        self._steps_since_last_replan += 1
        return self._phase

    def note_replan(self):
        self._steps_since_last_replan = 0

    def get_chunk_size(self) -> int:
        if self._phase == StagePhase.RECOVER:
            return self.recover_chunk_k
        if self._phase == StagePhase.INTERACT:
            return self.interact_chunk_k
        if self._phase == StagePhase.ALIGN:
            return self.align_chunk_k
        return self.transit_chunk_k

    def reset(self):
        self._force_history.clear()
        self._phase = StagePhase.TRANSIT
        self._phase_steps = 0
        self._no_progress_steps = 0
        self._steps_since_last_replan = 0
        self._transition_count = 0
        self._max_phase_reached = StagePhase.TRANSIT
        self._last_ee_pos = None
        self._last_gripper_open = None
        self._failure_mode = FailureMode.NONE
        self._last_transitioned = False
        self._contact_state = ContactState.FREE_SPACE
        self._has_object_in_hand = False
        self._held_object_steps = 0
        self._substage = StageSubgoal.TRANSIT
        self._stage_target_mode = StageTargetMode.NONE
