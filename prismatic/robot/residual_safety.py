"""
residual_safety.py

Rule-based safety layer for residual corrections.
"""

from typing import Optional

import numpy as np


class ResidualSafety:
    """Safety filters applied after residual correction to the base action."""

    def __init__(
        self,
        max_residual_pos: float = 0.005,
        max_residual_rot: float = 0.03,
        force_stop_threshold: float = 5.0,
        backoff_distance: float = 0.002,
        backoff_force_threshold: float = 3.0,
        workspace_min: Optional[list] = None,
        workspace_max: Optional[list] = None,
        max_delta_pos: float = 0.025,
        max_delta_rot: float = 0.10,
        lateral_force_threshold: float = 2.0,
        vertical_force_threshold: float = 2.5,
        torque_threshold: float = 1.5,
        z_relief_distance: float = 0.0015,
        yaw_relief_rot: float = 0.02,
    ):
        self.max_residual_pos = max_residual_pos
        self.max_residual_rot = max_residual_rot
        self.force_stop_threshold = force_stop_threshold
        self.backoff_distance = backoff_distance
        self.backoff_force_threshold = backoff_force_threshold
        self.lateral_force_threshold = lateral_force_threshold
        self.vertical_force_threshold = vertical_force_threshold
        self.torque_threshold = torque_threshold
        self.z_relief_distance = z_relief_distance
        self.yaw_relief_rot = yaw_relief_rot
        self.workspace_min = np.array(workspace_min or [-0.10, -0.35, 0.77], dtype=np.float32)
        self.workspace_max = np.array(workspace_max or [0.50, 0.37, 1.49], dtype=np.float32)
        self.max_delta_pos = max_delta_pos
        self.max_delta_rot = max_delta_rot

        self.force_stop_count = 0
        self.backoff_count = 0
        self.clip_count = 0
        self.workspace_clamp_count = 0

    def clip_residual(self, delta_pose_6d: np.ndarray) -> np.ndarray:
        out = delta_pose_6d.copy()
        pos_norm = np.linalg.norm(out[:3])
        if pos_norm > self.max_residual_pos:
            out[:3] = out[:3] * (self.max_residual_pos / max(pos_norm, 1e-8))
            self.clip_count += 1
        rot_norm = np.linalg.norm(out[3:6])
        if rot_norm > self.max_residual_rot:
            out[3:6] = out[3:6] * (self.max_residual_rot / max(rot_norm, 1e-8))
            self.clip_count += 1
        return out

    def apply_residual(self, base_action_7d: np.ndarray, delta_pose_6d: np.ndarray) -> np.ndarray:
        delta_pose_6d = self.clip_residual(delta_pose_6d)
        out = base_action_7d.copy()
        out[:6] = out[:6] + delta_pose_6d
        return out

    def check_force_stop(self, force_reading: Optional[np.ndarray]) -> bool:
        if force_reading is None:
            return False
        if np.linalg.norm(force_reading) > self.force_stop_threshold:
            self.force_stop_count += 1
            return True
        return False

    def compute_reflex_override(self, force_reading: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if force_reading is None:
            return None

        adjust = np.zeros(6, dtype=np.float32)
        triggered = False

        fxyz = np.asarray(force_reading[:3], dtype=np.float32)
        fxy = fxyz[:2]
        tau = np.asarray(force_reading[3:6], dtype=np.float32)

        lateral_mag = np.linalg.norm(fxy)
        if lateral_mag > self.lateral_force_threshold:
            adjust[:2] += (-fxy / max(lateral_mag, 1e-8)) * self.backoff_distance
            triggered = True

        if abs(float(fxyz[2])) > self.vertical_force_threshold:
            adjust[2] += self.z_relief_distance
            triggered = True

        torque_mag = np.linalg.norm(tau)
        if torque_mag > self.torque_threshold:
            yaw_sign = np.sign(tau[2]) if abs(float(tau[2])) > 1e-6 else 0.0
            if yaw_sign != 0.0:
                adjust[5] += -yaw_sign * self.yaw_relief_rot
                triggered = True

        if np.linalg.norm(fxyz) > self.backoff_force_threshold and not triggered:
            direction = -fxyz / max(np.linalg.norm(fxyz), 1e-8)
            adjust[:3] += direction * self.backoff_distance
            triggered = True

        if triggered:
            self.backoff_count += 1
            return adjust
        return None

    def compute_backoff(self, force_reading: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return self.compute_reflex_override(force_reading)

    def compute_invalid_action_recovery(
        self,
        base_action_7d: Optional[np.ndarray] = None,
        force_reading: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a conservative one-step recovery action after IK/action failure."""
        out = np.zeros(7, dtype=np.float32)
        if base_action_7d is not None:
            out[6] = float(np.asarray(base_action_7d, dtype=np.float32)[6])

        reflex_adjust = self.compute_reflex_override(force_reading)
        if reflex_adjust is not None:
            out[:6] = reflex_adjust
        else:
            # If the failed command was likely unreachable rather than force-induced,
            # hold pose for one control step and force a planner replan.
            out[:6] = 0.0
        return self.clip_final_action(out)

    def clip_final_action(self, delta_7d: np.ndarray) -> np.ndarray:
        out = delta_7d.copy()
        out[:3] = np.clip(out[:3], -self.max_delta_pos, self.max_delta_pos)
        out[3:6] = np.clip(out[3:6], -self.max_delta_rot, self.max_delta_rot)
        return out

    def clamp_workspace(self, target_xyz: np.ndarray) -> np.ndarray:
        clamped = np.clip(target_xyz, self.workspace_min, self.workspace_max)
        if not np.allclose(clamped, target_xyz):
            self.workspace_clamp_count += 1
        return clamped

    def workspace_violation(self, target_xyz: np.ndarray) -> float:
        """Return L2 distance from the workspace box; 0 means inside."""
        target_xyz = np.asarray(target_xyz, dtype=np.float32)
        below = np.maximum(self.workspace_min - target_xyz, 0.0)
        above = np.maximum(target_xyz - self.workspace_max, 0.0)
        return float(np.linalg.norm(below + above))

    def reset_counters(self):
        self.force_stop_count = 0
        self.backoff_count = 0
        self.clip_count = 0
        self.workspace_clamp_count = 0

    def get_stats(self) -> dict:
        return {
            "force_stop_count": self.force_stop_count,
            "backoff_count": self.backoff_count,
            "clip_count": self.clip_count,
            "workspace_clamp_count": self.workspace_clamp_count,
        }
