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

DEFAULT_PRECISION_TAKEOVER_TIERS = {
    "outer_pullback_candidate",
    "near_basin_shell",
    "micro_entry_ready",
    "close_ready",
    "yaw_entry_blocked",
}


def smooth_grasp_probe_xy_step(
    current_local_6d: np.ndarray,
    previous_local_6d: np.ndarray | None,
    *,
    alpha: float,
    max_xy_step: float,
    residual_norm: float | None = None,
    micro_deadband: float = 0.005,
    micro_hysteresis_alpha: float = 0.45,
) -> np.ndarray:
    current = np.asarray(current_local_6d, dtype=np.float32).reshape(-1)
    current = np.pad(current, (0, max(0, 6 - current.size)))[:6].astype(np.float32)
    alpha_used = float(alpha)
    if (
        previous_local_6d is not None
        and residual_norm is not None
        and np.isfinite(float(residual_norm))
        and float(residual_norm) < float(micro_deadband)
    ):
        alpha_used = float(min(float(alpha_used), float(micro_hysteresis_alpha)))
    if previous_local_6d is not None:
        previous = np.asarray(previous_local_6d, dtype=np.float32).reshape(-1)
        previous = np.pad(previous, (0, max(0, 6 - previous.size)))[:6].astype(np.float32)
        blended = current.copy()
        blended[:2] = alpha_used * current[:2] + (1.0 - alpha_used) * previous[:2]
        current = blended
    norm = float(np.linalg.norm(current[:2]))
    if norm > float(max_xy_step) > 0.0:
        current[:2] = current[:2] * (float(max_xy_step) / max(norm, 1.0e-9))
    return current.astype(np.float32)


def grasp_probe_close_ready_with_z(
    dx: float,
    dy: float,
    dz: float,
    dyaw: float,
    *,
    xy_threshold: float,
    yaw_threshold: float,
    z_threshold: float,
) -> bool:
    return bool(
        np.isfinite(float(dx))
        and np.isfinite(float(dy))
        and np.isfinite(float(dz))
        and np.isfinite(float(dyaw))
        and float(np.hypot(float(dx), float(dy))) <= float(xy_threshold)
        and abs(float(dyaw)) <= float(yaw_threshold)
        and abs(float(dz)) <= float(z_threshold)
    )


def grasp_probe_close_arbiter_decision(
    *,
    planner_gripper_value: float,
    planner_close_threshold: float = 0.5,
    close_ready: bool,
    stage_name: str,
    enabled: bool,
    guard_active: bool,
    active: bool,
    candidate_match: bool,
    gripper_mode: str,
) -> dict[str, object]:
    """Decide whether a planner close command is allowed during grasp-align smoke.

    This arbiter is intentionally about gripper authority only. It does not make
    yaw/z control available; it prevents planner close from bypassing a stricter
    close-ready decision after XY pullback has taken over or identified a
    failure-tail candidate.
    """

    planner_close_requested = bool(
        np.isfinite(float(planner_gripper_value))
        and float(planner_gripper_value) < float(planner_close_threshold)
    )
    protected_window = bool(
        str(stage_name) == "RING_GRASP_ALIGN"
        and str(gripper_mode) in {"planner_after_near", "eval_close_after_near"}
        and (bool(active) or bool(guard_active) or bool(candidate_match))
    )
    ready = bool(close_ready)
    blocked = bool(enabled and planner_close_requested and protected_window and not ready)
    if not bool(enabled):
        reason = "disabled"
    elif not planner_close_requested:
        reason = "planner_open"
    elif str(stage_name) != "RING_GRASP_ALIGN":
        reason = "not_grasp_align"
    elif str(gripper_mode) not in {"planner_after_near", "eval_close_after_near"}:
        reason = "gripper_mode_not_arbited"
    elif not (bool(active) or bool(guard_active) or bool(candidate_match)):
        reason = "outside_guard_window"
    elif ready:
        reason = "ready"
    else:
        reason = "not_close_ready"
    return {
        "planner_close_requested": bool(planner_close_requested),
        "protected_window": bool(protected_window),
        "close_ready": bool(ready),
        "blocked": bool(blocked),
        "reason": str(reason),
    }


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def candidate_xy_correction_ready(
    candidate_row: dict[str, object] | None,
    *,
    max_xy_error: float | None = None,
) -> bool:
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
    if max_xy_error is not None:
        xy_error = _safe_float(candidate_row.get("xy_error", float("nan")))
        if not np.isfinite(xy_error) or xy_error > float(max_xy_error) + 1.0e-9:
            return False
    if bool(candidate_row.get("xy_correction_ready", False)):
        return True
    if str(candidate_row.get("takeover_tier", "")) in XY_CORRECTION_READY_TIERS:
        return True
    if bool(candidate_row.get("grasp_probe_candidate_actionable_relaxed_small_xy_large_yaw", False)):
        return True
    return False


def candidate_within_xy_activation_window(
    candidate_row: dict[str, object] | None,
    *,
    max_xy_error: float | None,
) -> bool:
    """Return whether a manifest row is inside the runtime probe XY window."""
    if not candidate_row:
        return False
    if max_xy_error is None:
        return True
    xy_error = _safe_float(candidate_row.get("xy_error", float("nan")))
    return bool(np.isfinite(xy_error) and xy_error <= float(max_xy_error) + 1.0e-9)


def precision_takeover_activation_status(
    candidate_row: dict[str, object] | None,
    *,
    stage_age: int,
    queue_len: int,
    max_xy_error: float | None,
    min_stage_age: int = 12,
    allowed_tiers: set[str] | None = None,
    require_queue_empty: bool = False,
) -> tuple[bool, str]:
    """Conservative runtime-style activation guard for high-precision probes.

    Candidate manifests can be intentionally broad for audit support. Runtime
    takeover evidence is narrower: the planner must have owned the stage for a
    short dwell, the row must be inside the XY activation window, and frontier /
    coarse support tiers are excluded unless explicitly allowed by the caller.
    """
    if not candidate_row:
        return False, "missing_candidate"
    if int(stage_age) < int(max(0, min_stage_age)):
        return False, "stage_too_young"
    if bool(require_queue_empty) and int(queue_len) > 0:
        return False, "planner_queue_not_empty"
    if not candidate_within_xy_activation_window(candidate_row, max_xy_error=max_xy_error):
        return False, "outside_xy_activation_window"
    tier = str(candidate_row.get("takeover_tier", ""))
    allowed = DEFAULT_PRECISION_TAKEOVER_TIERS if allowed_tiers is None else {str(item) for item in allowed_tiers}
    if tier not in allowed:
        return False, "takeover_tier_not_runtime_precision"
    return True, "ready"
