"""Small helpers for grasp probe execution.

These are intentionally light-weight so unit tests can import them without
pulling in the full evaluation stack.
"""

from __future__ import annotations

import numpy as np


XY_CORRECTION_READY_TIERS = {
    "coarse_pullback_candidate",
    "outer_pullback_candidate",
    "frontier_pullback_candidate",
    "near_basin_shell",
    "micro_entry_ready",
    "yaw_entry_blocked",
}


def smooth_grasp_probe_xy_step(
    current_local_6d: np.ndarray,
    previous_local_6d: np.ndarray | None,
    *,
    alpha: float,
    max_xy_step: float,
) -> np.ndarray:
    current = np.asarray(current_local_6d, dtype=np.float32).reshape(-1)
    current = np.pad(current, (0, max(0, 6 - current.size)))[:6].astype(np.float32)
    if previous_local_6d is not None:
        previous = np.asarray(previous_local_6d, dtype=np.float32).reshape(-1)
        previous = np.pad(previous, (0, max(0, 6 - previous.size)))[:6].astype(np.float32)
        blended = current.copy()
        blended[:2] = float(alpha) * current[:2] + (1.0 - float(alpha)) * previous[:2]
        current = blended
    norm = float(np.linalg.norm(current[:2]))
    if norm > float(max_xy_step) > 0.0:
        current[:2] = current[:2] * (float(max_xy_step) / max(norm, 1.0e-9))
    return current.astype(np.float32)


def candidate_xy_correction_ready(candidate_row: dict[str, object] | None) -> bool:
    if not candidate_row:
        return False
    if bool(candidate_row.get("label_valid", True)) is False:
        return False
    if str(candidate_row.get("sample_role", "")) == "success_window_control":
        return False
    if str(candidate_row.get("takeover_tier", "")) == "abstain_prior_only":
        return False
    if str(candidate_row.get("takeover_tier", "")) == "too_far":
        return False
    if bool(candidate_row.get("xy_correction_ready", False)):
        return True
    if str(candidate_row.get("takeover_tier", "")) in XY_CORRECTION_READY_TIERS:
        return True
    if bool(candidate_row.get("grasp_probe_candidate_actionable_relaxed_small_xy_large_yaw", False)):
        return True
    return False
