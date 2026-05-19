"""
residual_transforms.py

Helpers for converting 6D pose deltas between world frame and the current
end-effector local frame. Translation and rotvec components are both rotated
by the current EE orientation for a small-angle local residual approximation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def world_delta_to_local(delta_pose_6d: np.ndarray, current_quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert a world-frame 6D delta into the current EE local frame."""
    delta = _as_float32(delta_pose_6d).copy()
    quat = _as_float32(current_quat_xyzw)
    r_world_ee = Rotation.from_quat(quat)
    delta[:3] = r_world_ee.inv().apply(delta[:3]).astype(np.float32)
    delta[3:6] = r_world_ee.inv().apply(delta[3:6]).astype(np.float32)
    return delta


def local_delta_to_world(delta_pose_6d: np.ndarray, current_quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an EE-local 6D delta into the world frame."""
    delta = _as_float32(delta_pose_6d).copy()
    quat = _as_float32(current_quat_xyzw)
    r_world_ee = Rotation.from_quat(quat)
    delta[:3] = r_world_ee.apply(delta[:3]).astype(np.float32)
    delta[3:6] = r_world_ee.apply(delta[3:6]).astype(np.float32)
    return delta

