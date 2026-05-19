"""
gripper_supervisor.py

Diagnostic gripper commit/latch layer for chunk planners.

This module only edits action[6]. It never changes xyz/rotation, chunk length,
replanning, or workspace handling, so it can be ablated independently from the
near-field residual controllers.
"""

from typing import Optional, Sequence

import numpy as np


class GripperSupervisor:
    """Diagnostic-only gripper smoother; not part of the default NFCR policy."""

    def __init__(
        self,
        close_threshold: float = 0.5,
        open_threshold: float = 0.8,
        near_depth_threshold: float = 0.08,
        gripper_open_threshold: float = 0.5,
        close_lookahead: int = 4,
        min_close_votes: int = 2,
        min_hold_steps: int = 40,
        release_open_votes: int = 3,
        allow_release: bool = True,
    ):
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.near_depth_threshold = float(near_depth_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.close_lookahead = int(close_lookahead)
        self.min_close_votes = int(min_close_votes)
        self.min_hold_steps = int(min_hold_steps)
        self.release_open_votes = int(release_open_votes)
        self.allow_release = bool(allow_release)
        self.reset()

    def reset(self):
        self.state = "free"
        self.hold_steps = 0
        self.close_commit_count = 0
        self.blocked_early_close_count = 0
        self.hold_override_count = 0
        self.release_count = 0
        self.raw_close_before_near_count = 0
        self.raw_reopen_during_hold_count = 0
        self.command_flip_count = 0
        self._last_raw_is_close = None
        self._last_trace = {}

    def _plan_values(self, raw_gripper: float, future_gripper_actions: Optional[Sequence[float]]):
        values = [float(raw_gripper)]
        if future_gripper_actions is not None:
            values.extend(float(x) for x in list(future_gripper_actions)[: self.close_lookahead])
        return values

    def step(
        self,
        action_7d: np.ndarray,
        depth_proximity: Optional[float],
        gripper_open: Optional[float],
        future_gripper_actions: Optional[Sequence[float]] = None,
        phase_id: Optional[int] = None,
    ) -> np.ndarray:
        """Return a copy of action_7d with only the gripper command supervised."""
        out = np.asarray(action_7d, dtype=np.float32).copy()
        raw = float(out[6])
        plan = self._plan_values(raw, future_gripper_actions)
        near_target = (
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and float(depth_proximity) < self.near_depth_threshold
        )
        gripper_is_open = gripper_open is None or float(gripper_open) >= self.gripper_open_threshold
        close_votes = sum(v <= self.close_threshold for v in plan)
        open_votes = sum(v >= self.open_threshold for v in plan)
        raw_is_close = raw <= self.close_threshold
        raw_is_open = raw >= self.open_threshold

        if self._last_raw_is_close is not None and raw_is_close != self._last_raw_is_close:
            self.command_flip_count += 1
        self._last_raw_is_close = raw_is_close

        if raw_is_close and not near_target:
            self.raw_close_before_near_count += 1

        event = "pass"
        if self.state == "holding":
            self.hold_steps += 1
            if raw_is_open:
                self.raw_reopen_during_hold_count += 1
            can_release = (
                self.allow_release
                and near_target
                and self.hold_steps >= self.min_hold_steps
                and open_votes >= self.release_open_votes
            )
            if can_release:
                self.state = "released"
                self.release_count += 1
                out[6] = 1.0
                event = "release"
            else:
                out[6] = 0.0
                if raw_is_open:
                    self.hold_override_count += 1
                    event = "hold_override"
                else:
                    event = "hold"
        elif raw_is_close:
            close_allowed = near_target and gripper_is_open and close_votes >= self.min_close_votes
            if close_allowed:
                self.state = "holding"
                self.hold_steps = 0
                self.close_commit_count += 1
                out[6] = 0.0
                event = "commit_close"
            else:
                out[6] = 1.0
                self.blocked_early_close_count += 1
                event = "block_close"
        else:
            out[6] = raw
            if self.state == "released" and gripper_is_open:
                self.state = "free"
                self.hold_steps = 0

        self._last_trace = {
            "gripper_state": self.state,
            "gripper_event": event,
            "raw_gripper": raw,
            "exec_gripper": float(out[6]),
            "near_target": bool(near_target),
            "depth_proximity": None if depth_proximity is None else float(depth_proximity),
            "close_votes": int(close_votes),
            "open_votes": int(open_votes),
            "hold_steps": int(self.hold_steps),
            "phase_id": None if phase_id is None else int(phase_id),
        }
        return out

    def get_last_trace(self) -> dict:
        return dict(self._last_trace)

    def get_stats(self) -> dict:
        return {
            "gripper_supervisor_state": self.state,
            "gripper_close_commit_count": self.close_commit_count,
            "gripper_blocked_early_close_count": self.blocked_early_close_count,
            "gripper_hold_override_count": self.hold_override_count,
            "gripper_release_count": self.release_count,
            "gripper_raw_close_before_near_count": self.raw_close_before_near_count,
            "gripper_raw_reopen_during_hold_count": self.raw_reopen_during_hold_count,
            "gripper_command_flip_count": self.command_flip_count,
            "gripper_hold_steps": self.hold_steps,
            "gripper_close_threshold": self.close_threshold,
            "gripper_open_threshold": self.open_threshold,
            "gripper_near_depth_threshold": self.near_depth_threshold,
            "gripper_close_lookahead": self.close_lookahead,
            "gripper_min_close_votes": self.min_close_votes,
            "gripper_min_hold_steps": self.min_hold_steps,
            "gripper_release_open_votes": self.release_open_votes,
            "gripper_allow_release": self.allow_release,
        }
