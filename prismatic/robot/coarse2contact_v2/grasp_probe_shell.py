from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def grasp_probe_shell_fields(
    probe_error: np.ndarray | None,
    *,
    near_grasp_xy_threshold: float,
    near_grasp_yaw_threshold: float,
    max_xy_step: float,
    horizon_steps: int,
    coarse_pullback_xy_threshold: float = 0.060,
    outer_pullback_xy_threshold: float = 0.120,
    frontier_pullback_xy_threshold: float = 0.180,
) -> dict[str, object]:
    if probe_error is None:
        raw = np.full((6,), np.nan, dtype=np.float32)
    else:
        raw = np.asarray(probe_error, dtype=np.float32).reshape(-1)
    dx = float(raw[0]) if raw.size > 0 and np.isfinite(raw[0]) else float("nan")
    dy = float(raw[1]) if raw.size > 1 and np.isfinite(raw[1]) else float("nan")
    yaw_idx = 5 if raw.size >= 6 else 3
    dyaw = float(raw[yaw_idx]) if raw.size > yaw_idx and np.isfinite(raw[yaw_idx]) else float("nan")
    pre_xy = float(np.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else float("nan")
    pre_yaw_abs = float(abs(dyaw)) if np.isfinite(dyaw) else float("nan")
    one_step_xy_feasible = bool(np.isfinite(pre_xy) and pre_xy <= float(near_grasp_xy_threshold) + float(max_xy_step) + 1.0e-9)
    horizon_xy_feasible = bool(
        np.isfinite(pre_xy)
        and pre_xy <= float(near_grasp_xy_threshold) + float(max_xy_step) * float(max(1, int(horizon_steps))) + 1.0e-9
    )
    yaw_feasible = bool(np.isfinite(pre_yaw_abs) and pre_yaw_abs <= float(near_grasp_yaw_threshold) + 1.0e-9)
    tight_near_shell = bool(one_step_xy_feasible and yaw_feasible)
    near_shell = bool(horizon_xy_feasible and yaw_feasible)
    coarse_pullback_candidate = bool(
        np.isfinite(pre_xy)
        and np.isfinite(pre_yaw_abs)
        and pre_xy > float(near_grasp_xy_threshold) + float(max_xy_step) * float(max(1, int(horizon_steps))) + 1.0e-9
        and pre_xy <= float(coarse_pullback_xy_threshold)
        and yaw_feasible
    )
    outer_pullback_candidate = bool(
        np.isfinite(pre_xy)
        and pre_xy > float(near_grasp_xy_threshold) + float(max_xy_step) * float(max(1, int(horizon_steps))) + 1.0e-9
        and pre_xy <= float(outer_pullback_xy_threshold)
    )
    frontier_pullback_candidate = bool(
        np.isfinite(pre_xy)
        and pre_xy > float(outer_pullback_xy_threshold)
        and pre_xy <= float(frontier_pullback_xy_threshold)
    )
    return {
        "grasp_probe_pre_xy_error": pre_xy,
        "grasp_probe_pre_abs_yaw": pre_yaw_abs,
        "grasp_probe_one_step_xy_feasible": one_step_xy_feasible,
        "grasp_probe_horizon_xy_feasible": horizon_xy_feasible,
        "grasp_probe_yaw_feasible": yaw_feasible,
        "grasp_probe_tight_near_basin_shell": tight_near_shell,
        "grasp_probe_near_basin_shell": near_shell,
        "grasp_probe_coarse_pullback_candidate": coarse_pullback_candidate,
        "grasp_probe_outer_pullback_candidate": outer_pullback_candidate,
        "grasp_probe_frontier_pullback_candidate": frontier_pullback_candidate,
    }


def grasp_probe_inactive_reason(
    *,
    policy: str,
    stage_ok: bool,
    visibility_bucket: str,
    has_error: bool,
    finite_xy: bool,
    shell_filter: str,
    shell_fields: Mapping[str, Any],
) -> str:
    if policy == "off":
        return "off"
    if not has_error or not finite_xy:
        return "missing_privileged_error"
    if str(visibility_bucket) == "prior_only":
        return "prior_only_abstain"
    if not stage_ok:
        return "stage_not_grasp_align"
    if shell_filter == "near_yaw_feasible":
        if not bool(shell_fields.get("grasp_probe_horizon_xy_feasible", False)):
            return "shell_xy_outside_horizon"
        if not bool(shell_fields.get("grasp_probe_yaw_feasible", False)):
            return "shell_yaw_blocked"
    if shell_filter == "tight_near_yaw_feasible":
        if not bool(shell_fields.get("grasp_probe_one_step_xy_feasible", False)):
            return "shell_xy_outside_one_step"
        if not bool(shell_fields.get("grasp_probe_yaw_feasible", False)):
            return "shell_yaw_blocked"
    if shell_filter == "coarse_yaw_feasible":
        if not bool(shell_fields.get("grasp_probe_yaw_feasible", False)):
            return "shell_yaw_blocked"
        if not bool(shell_fields.get("grasp_probe_horizon_xy_feasible", False)) and not bool(shell_fields.get("grasp_probe_coarse_pullback_candidate", False)):
            return "shell_outside_coarse_window"
    if shell_filter == "frontier_pullback_feasible":
        if not (
            bool(shell_fields.get("grasp_probe_outer_pullback_candidate", False))
            or bool(shell_fields.get("grasp_probe_coarse_pullback_candidate", False))
            or bool(shell_fields.get("grasp_probe_near_basin_shell", False))
            or bool(shell_fields.get("grasp_probe_frontier_pullback_candidate", False))
        ):
            return "shell_outside_frontier_window"
    if shell_filter == "small_xy_large_yaw_frontier_feasible":
        if float(shell_fields.get("grasp_probe_pre_xy_error", float("inf"))) > 0.060 + 1.0e-9:
            return "shell_outside_small_xy_large_yaw_frontier"
    return "inactive"
