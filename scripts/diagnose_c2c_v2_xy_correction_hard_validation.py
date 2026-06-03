#!/usr/bin/env python3
"""Hard validation for C2C v2 XY correction rows.

For every active grasp probe row, compare:
- privileged residual direction,
- non-privileged localizer proxy dx/dy direction,
- actual applied XY step,
- post residual contraction,
- observed end-effector displacement from runtime observations when available.

The output is meant to decide whether the next fix belongs in action
application/queue/smoothing, non-privileged estimator calibration, step sizing,
or frame/sign convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.residual_transforms import world_delta_to_local


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_from_path(path: Path) -> int:
    for token in path.stem.split("_"):
        if token.startswith("ep") and token[2:].isdigit():
            return int(token[2:])
    return -1


def load_trace_rows(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep = _episode_from_path(path)
        for row in _read_jsonl(path):
            item = dict(row)
            item.setdefault("episode_idx", ep)
            item.setdefault("source_trace_path", str(path))
            rows.append(item)
    rows.sort(key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step", r.get("step_idx", -1)))))
    return rows


def _safe_vec(value: Any, *, length: int, default: float = float("nan")) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), constant_values=default)
    return arr[:length].astype(np.float32)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _xy_norm(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return float("nan")
    return float(np.linalg.norm(arr[:2]))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)[:2]
    bb = np.asarray(b, dtype=np.float32).reshape(-1)[:2]
    if aa.size < 2 or bb.size < 2 or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return float("nan")
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1.0e-12:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def _axis_sign_match(a: np.ndarray, b: np.ndarray, *, eps: float = 1.0e-6) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)[:2]
    bb = np.asarray(b, dtype=np.float32).reshape(-1)[:2]
    matches = []
    for av, bv in zip(aa, bb):
        if not np.isfinite(float(av)) or not np.isfinite(float(bv)):
            continue
        if abs(float(av)) <= float(eps) or abs(float(bv)) <= float(eps):
            continue
        matches.append(np.sign(float(av)) == np.sign(float(bv)))
    if not matches:
        return float("nan")
    return float(np.mean(matches))


def _nested(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, Mapping):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, Mapping) else {}


def _load_runtime_pose_cache(runtime_obs_dir: Path | None) -> dict[int, np.ndarray]:
    if runtime_obs_dir is None or not runtime_obs_dir.exists():
        return {}
    out: dict[int, np.ndarray] = {}
    for path in sorted(runtime_obs_dir.glob("ep*_runtime_obs.npz")):
        ep = _episode_from_path(path)
        try:
            npz = np.load(path, allow_pickle=False)
            if "gripper_pose" in npz:
                out[ep] = np.asarray(npz["gripper_pose"], dtype=np.float32)
        except Exception:
            continue
    return out


def _observed_ee_delta_local(row: Mapping[str, Any], pose_cache: Mapping[int, np.ndarray]) -> tuple[np.ndarray, str]:
    ep = int(row.get("episode_idx", -1))
    step = int(row.get("step", row.get("step_idx", -1)))
    poses = pose_cache.get(ep)
    if poses is not None and step >= 0 and step + 1 < len(poses):
        before = np.asarray(poses[step], dtype=np.float32).reshape(-1)
        after = np.asarray(poses[step + 1], dtype=np.float32).reshape(-1)
        if before.size >= 7 and after.size >= 3 and np.all(np.isfinite(before[:7])) and np.all(np.isfinite(after[:3])):
            world = np.zeros(6, dtype=np.float32)
            world[:3] = after[:3] - before[:3]
            return world_delta_to_local(world, before[3:7]).astype(np.float32), "runtime_obs_gripper_pose_delta"
    world_delta = _safe_vec(row.get("final_action_world_6d", row.get("post_clip_action_world_6d")), length=6)
    # This fallback is less faithful to the video because it is the commanded
    # delta, not the observed pose delta.
    return world_delta.astype(np.float32), "commanded_final_action_world_fallback"


def diagnose_active_row(row: Mapping[str, Any], pose_cache: Mapping[int, np.ndarray]) -> dict[str, Any]:
    pre = _safe_vec(row.get("grasp_probe_pre_true_error_t"), length=4)
    post = _safe_vec(row.get("grasp_probe_post_true_error_t"), length=4)
    horizon = _safe_vec(row.get("grasp_probe_horizon_final_true_error_t"), length=4)
    proxy = _safe_vec(
        [
            _safe_float(_nested(row, "local_geometry_error", "grasp").get("dx")),
            _safe_float(_nested(row, "local_geometry_error", "grasp").get("dy")),
        ],
        length=2,
    )
    est_proxy = _safe_vec(
        [
            _safe_float(_nested(row, "estimated_basin_error").get("estimated_basin_error_proxy_dx")),
            _safe_float(_nested(row, "estimated_basin_error").get("estimated_basin_error_proxy_dy")),
        ],
        length=2,
    )
    runtime_est = _nested(row, "runtime_xy_estimator")
    runtime_est_xy = _safe_vec([_safe_float(runtime_est.get("dx")), _safe_float(runtime_est.get("dy"))], length=2)
    applied = _safe_vec(row.get("grasp_probe_applied_xy_step_local_6d"), length=6)
    raw = _safe_vec(row.get("grasp_probe_raw_xy_step_local_6d"), length=6)
    observed_delta, observed_source = _observed_ee_delta_local(row, pose_cache)
    planner_local = _safe_vec(row.get("planner_chunk_local_6d"), length=6)
    final_local = _safe_vec(row.get("grasp_probe_local_command_local_6d"), length=6)

    pre_xy = _xy_norm(pre[:2])
    post_xy = _xy_norm(post[:2])
    horizon_xy = _xy_norm(horizon[:2])
    applied_norm = _xy_norm(applied[:2])
    proxy_norm = _xy_norm(proxy[:2])
    observed_norm = _xy_norm(observed_delta[:2])
    one_step_delta = post_xy - pre_xy if np.isfinite(pre_xy) and np.isfinite(post_xy) else float("nan")
    horizon_delta = horizon_xy - pre_xy if np.isfinite(pre_xy) and np.isfinite(horizon_xy) else float("nan")
    oracle_step_cosine = _cosine(applied[:2], pre[:2])
    proxy_residual_cosine = _cosine(proxy[:2], pre[:2])
    est_proxy_residual_cosine = _cosine(est_proxy[:2], pre[:2])
    runtime_est_residual_cosine = _cosine(runtime_est_xy[:2], pre[:2])
    observed_applied_cosine = _cosine(observed_delta[:2], applied[:2])
    final_minus_planner = final_local[:2] - planner_local[:2]
    observed_final_command_cosine = _cosine(observed_delta[:2], final_local[:2])
    applied_command_cosine = _cosine(final_minus_planner, applied[:2])
    step_to_residual_ratio = float(applied_norm / max(pre_xy, 1.0e-9)) if np.isfinite(applied_norm) and np.isfinite(pre_xy) else float("nan")

    flags: list[str] = []
    if np.isfinite(oracle_step_cosine) and oracle_step_cosine < -0.2:
        flags.append("oracle_direction_opposes_residual")
    if np.isfinite(proxy_residual_cosine) and proxy_residual_cosine < 0.2:
        flags.append("proxy_residual_misaligned")
    if runtime_est and np.isfinite(runtime_est_residual_cosine) and runtime_est_residual_cosine < 0.2:
        flags.append("runtime_estimator_residual_misaligned")
    if np.isfinite(observed_applied_cosine) and observed_applied_cosine < 0.2:
        flags.append("observed_motion_not_following_applied_step")
    if np.isfinite(applied_command_cosine) and applied_command_cosine < 0.8:
        flags.append("command_delta_not_matching_applied_step")
    if np.isfinite(step_to_residual_ratio) and step_to_residual_ratio < 0.08:
        flags.append("step_too_small")
    if np.isfinite(horizon_delta) and horizon_delta >= -1.0e-9:
        flags.append("oracle_horizon_not_contracting")
    if bool(row.get("grasp_probe_horizon_overshoot", False)):
        flags.append("overshoot")

    return {
        "episode_idx": int(row.get("episode_idx", -1)),
        "step": int(row.get("step", row.get("step_idx", -1))),
        "takeover_tier": str(row.get("grasp_probe_failure_tail_takeover_tier", "")),
        "visibility": str(row.get("grasp_probe_visibility_bucket", "")),
        "privileged_residual_xy": [float(pre[0]), float(pre[1])],
        "localizer_proxy_xy": [float(proxy[0]), float(proxy[1])],
        "estimated_proxy_xy": [float(est_proxy[0]), float(est_proxy[1])],
        "runtime_estimator_xy": [float(runtime_est_xy[0]), float(runtime_est_xy[1])],
        "runtime_estimator_source": str(runtime_est.get("source", "")) if runtime_est else "",
        "runtime_estimator_entry_ready": bool(runtime_est.get("entry_ready", False)) if runtime_est else False,
        "applied_xy_step_local": [float(applied[0]), float(applied[1])],
        "raw_xy_step_local": [float(raw[0]), float(raw[1])],
        "observed_ee_delta_local_xy": [float(observed_delta[0]), float(observed_delta[1])],
        "observed_ee_delta_source": observed_source,
        "planner_local_xy": [float(planner_local[0]), float(planner_local[1])],
        "final_local_xy": [float(final_local[0]), float(final_local[1])],
        "final_minus_planner_local_xy": [float(final_minus_planner[0]), float(final_minus_planner[1])],
        "pre_xy_error": float(pre_xy),
        "post_xy_error": float(post_xy),
        "horizon_post_xy_error": float(horizon_xy),
        "one_step_xy_delta": float(one_step_delta),
        "horizon_xy_delta": float(horizon_delta),
        "oracle_step_cosine_to_residual": float(oracle_step_cosine),
        "proxy_cosine_to_privileged_residual": float(proxy_residual_cosine),
        "estimated_proxy_cosine_to_privileged_residual": float(est_proxy_residual_cosine),
        "runtime_estimator_cosine_to_privileged_residual": float(runtime_est_residual_cosine),
        "xy_estimator_cosine_to_privileged": float(runtime_est_residual_cosine),
        "xy_estimator_axis_sign_match": float(_axis_sign_match(runtime_est_xy[:2], pre[:2])),
        "xy_residual_norm_before": float(pre_xy),
        "xy_residual_norm_after": float(horizon_xy if np.isfinite(horizon_xy) else post_xy),
        "observed_ee_delta_cosine_to_applied_step": float(observed_applied_cosine),
        "observed_ee_delta_cosine_to_final_command": float(observed_final_command_cosine),
        "command_delta_cosine_to_applied_step": float(applied_command_cosine),
        "applied_step_to_residual_ratio": float(step_to_residual_ratio),
        "horizon_xy_contracted": bool(horizon_delta < -1.0e-9) if np.isfinite(horizon_delta) else False,
        "near_grasp_after": bool(row.get("grasp_probe_horizon_near_grasp_after", False)),
        "overshoot": bool(row.get("grasp_probe_horizon_overshoot", False)),
        "takeover_session_id": int(row.get("takeover_session_id", 0) or 0),
        "takeover_lifecycle_state": str(row.get("takeover_lifecycle_state", "")),
        "terminal_state": str(row.get("terminal_state", "")),
        "alignment_ready_for_handoff": bool(row.get("alignment_ready_for_handoff", False)),
        "safe_abstain_open": bool(row.get("safe_abstain_open", False)),
        "failed_retryable": bool(row.get("failed_retryable", False)),
        "failed_terminal": bool(row.get("failed_terminal", False)),
        "alignment_handoff_block_reason": str(row.get("alignment_handoff_block_reason", "")),
        "planner_gripper_handoff_allowed": bool(row.get("planner_gripper_handoff_allowed", False)),
        "planner_gripper_close_requested": bool(row.get("planner_gripper_close_requested", False)),
        "planner_gripper_close_blocked": bool(row.get("planner_gripper_close_blocked", False)),
        "alignment_xy_ready": bool(row.get("alignment_xy_ready", False)),
        "alignment_z_ready": bool(row.get("alignment_z_ready", False)),
        "alignment_yaw_ready": bool(row.get("alignment_yaw_ready", False)),
        "alignment_observability_ready": bool(row.get("alignment_observability_ready", False)),
        "alignment_frame_consistency_ready": bool(row.get("alignment_frame_consistency_ready", False)),
        "queue_len_before": int(row.get("grasp_probe_queue_len_before", 0) or 0),
        "queue_len_after": int(row.get("grasp_probe_queue_len_after", 0) or 0),
        "flags": flags,
    }


def _rate(rows: list[Mapping[str, Any]], pred) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(pred(row)) for row in rows]))


def _session_key(row: Mapping[str, Any]) -> tuple[int, int] | None:
    session_id = int(row.get("takeover_session_id", 0) or 0)
    if session_id <= 0:
        return None
    return (int(row.get("episode_idx", -1)), session_id)


def summarize(rows: list[dict[str, Any]], pose_cache: Mapping[int, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    active_rows = [r for r in rows if bool(r.get("grasp_probe_active", False))]
    diag_rows = [diagnose_active_row(r, pose_cache) for r in active_rows]
    flag_counts = Counter(flag for row in diag_rows for flag in row["flags"])
    terminal_rows = [
        r for r in rows
        if str(r.get("terminal_state", "")) or bool(r.get("alignment_ready_for_handoff", False)) or bool(r.get("safe_abstain_open", False)) or bool(r.get("failed_terminal", False))
    ]
    session_ids = {key for r in rows if (key := _session_key(r)) is not None}
    terminal_session_ids = {key for r in terminal_rows if (key := _session_key(r)) is not None}
    handoff_allowed_rows = [r for r in rows if bool(r.get("planner_gripper_handoff_allowed", False))]
    planner_close_requested_rows = [r for r in rows if bool(r.get("planner_gripper_close_requested", False))]
    planner_close_blocked_rows = [r for r in rows if bool(r.get("planner_gripper_close_blocked", False))]
    handoff_block_reasons = Counter(
        str(r.get("alignment_handoff_block_reason", ""))
        for r in rows
        if str(r.get("alignment_handoff_block_reason", ""))
    )
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_flag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diag_rows:
        by_episode[int(row["episode_idx"])].append(row)
        for flag in row["flags"]:
            by_flag[str(flag)].append(row)

    oracle_contracts = _rate(diag_rows, lambda r: r["horizon_xy_contracted"])
    proxy_aligned = _rate(
        diag_rows,
        lambda r: np.isfinite(float(r["proxy_cosine_to_privileged_residual"])) and float(r["proxy_cosine_to_privileged_residual"]) > 0.5,
    )
    observed_follows = _rate(
        diag_rows,
        lambda r: np.isfinite(float(r["observed_ee_delta_cosine_to_applied_step"])) and float(r["observed_ee_delta_cosine_to_applied_step"]) > 0.5,
    )
    observed_follows_final_command = _rate(
        diag_rows,
        lambda r: np.isfinite(float(r["observed_ee_delta_cosine_to_final_command"])) and float(r["observed_ee_delta_cosine_to_final_command"]) > 0.5,
    )
    applied_aligned = _rate(
        diag_rows,
        lambda r: np.isfinite(float(r["oracle_step_cosine_to_residual"])) and float(r["oracle_step_cosine_to_residual"]) > 0.5,
    )
    estimator_aligned_rows = [r for r in diag_rows if str(r.get("runtime_estimator_source", ""))]
    estimator_aligned = _rate(
        estimator_aligned_rows,
        lambda r: np.isfinite(float(r["runtime_estimator_cosine_to_privileged_residual"])) and float(r["runtime_estimator_cosine_to_privileged_residual"]) > 0.5,
    )
    small_step = _rate(diag_rows, lambda r: "step_too_small" in r["flags"])
    near_after = _rate(diag_rows, lambda r: r["near_grasp_after"])
    overshoot = _rate(diag_rows, lambda r: r["overshoot"])
    xy_ready = _rate(diag_rows, lambda r: r["alignment_xy_ready"])
    z_ready = _rate(diag_rows, lambda r: r["alignment_z_ready"])
    yaw_ready = _rate(diag_rows, lambda r: r["alignment_yaw_ready"])
    handoff_ready = _rate(rows, lambda r: r.get("alignment_ready_for_handoff", False))
    safe_abstain = _rate(rows, lambda r: r.get("safe_abstain_open", False))
    failed_terminal = _rate(rows, lambda r: r.get("failed_terminal", False))
    close_block_rate = (
        float(len(planner_close_blocked_rows) / max(1, len(planner_close_requested_rows)))
        if planner_close_requested_rows
        else 0.0
    )

    has_runtime_estimator = bool(estimator_aligned_rows)
    trusted_direction_rate = float(estimator_aligned if has_runtime_estimator else proxy_aligned)
    if oracle_contracts < 0.6 or (observed_follows < 0.5 and observed_follows_final_command < 0.5):
        decision = "fix_action_application_queue_smoothing_handoff"
    elif has_runtime_estimator and estimator_aligned < 0.6:
        decision = "train_or_calibrate_non_privileged_xy_estimator"
    elif not has_runtime_estimator and proxy_aligned < 0.6:
        decision = "train_or_calibrate_non_privileged_xy_estimator"
    elif applied_aligned < 0.6:
        decision = "fix_frame_sign_convention"
    elif small_step > 0.35 or (trusted_direction_rate >= 0.8 and near_after < 0.6):
        decision = "increase_step_horizon_or_sticky"
    elif (
        (not has_runtime_estimator and flag_counts.get("proxy_residual_misaligned", 0))
        or (applied_aligned < 0.8 and flag_counts.get("oracle_direction_opposes_residual", 0))
    ):
        decision = "fix_frame_sign_convention"
    else:
        decision = "xy_oracle_probe_contracts_continue_runtime_estimator_work"

    report = {
        "schema_version": "c2c_v2_xy_correction_hard_validation_v1",
        "active_rows": int(len(diag_rows)),
        "oracle_horizon_contraction_rate": float(oracle_contracts),
        "near_grasp_after_rate": float(near_after),
        "overshoot_rate": float(overshoot),
        "proxy_aligned_with_privileged_residual_rate": float(proxy_aligned),
        "runtime_estimator_aligned_with_privileged_residual_rate": float(estimator_aligned),
        "applied_step_aligned_with_privileged_residual_rate": float(applied_aligned),
        "observed_ee_delta_follows_applied_step_rate": float(observed_follows),
        "observed_ee_delta_follows_final_command_rate": float(observed_follows_final_command),
        "step_too_small_rate": float(small_step),
        "takeover_sessions": int(len(session_ids)),
        "terminal_takeover_sessions": int(len(terminal_session_ids)),
        "alignment_success_rate": float(
            len({key for r in terminal_rows if bool(r.get("alignment_ready_for_handoff", False)) and (key := _session_key(r)) is not None})
            / max(1, len(session_ids))
        ),
        "safe_abstain_rate": float(safe_abstain),
        "failed_terminal_rate": float(failed_terminal),
        "alignment_ready_for_handoff_rate": float(handoff_ready),
        "handoff_allowed_rows": int(len(handoff_allowed_rows)),
        "planner_close_requested_rows": int(len(planner_close_requested_rows)),
        "planner_close_blocked_rows": int(len(planner_close_blocked_rows)),
        "planner_close_blocked_rate": float(close_block_rate),
        "alignment_xy_ready_rate": float(xy_ready),
        "alignment_z_ready_rate": float(z_ready),
        "alignment_yaw_ready_rate": float(yaw_ready),
        "handoff_block_reason_counts": dict(handoff_block_reasons),
        "flag_counts": dict(flag_counts),
        "decision": decision,
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_analysis": True,
        "by_episode": [],
        "by_flag": [],
        "representative_rows": [],
    }
    for ep, ep_rows in sorted(by_episode.items()):
        report["by_episode"].append(
            {
                "episode_idx": int(ep),
                "active_rows": int(len(ep_rows)),
                "oracle_horizon_contraction_rate": _rate(ep_rows, lambda r: r["horizon_xy_contracted"]),
                "near_grasp_after_rate": _rate(ep_rows, lambda r: r["near_grasp_after"]),
                "overshoot_rate": _rate(ep_rows, lambda r: r["overshoot"]),
                "proxy_aligned_with_privileged_residual_rate": _rate(
                    ep_rows,
                    lambda r: np.isfinite(float(r["proxy_cosine_to_privileged_residual"])) and float(r["proxy_cosine_to_privileged_residual"]) > 0.5,
                ),
                "runtime_estimator_aligned_with_privileged_residual_rate": _rate(
                    [r for r in ep_rows if str(r.get("runtime_estimator_source", ""))],
                    lambda r: np.isfinite(float(r["runtime_estimator_cosine_to_privileged_residual"])) and float(r["runtime_estimator_cosine_to_privileged_residual"]) > 0.5,
                ),
                "applied_step_aligned_with_privileged_residual_rate": _rate(
                    ep_rows,
                    lambda r: np.isfinite(float(r["oracle_step_cosine_to_residual"])) and float(r["oracle_step_cosine_to_residual"]) > 0.5,
                ),
                "observed_ee_delta_follows_applied_step_rate": _rate(
                    ep_rows,
                    lambda r: np.isfinite(float(r["observed_ee_delta_cosine_to_applied_step"])) and float(r["observed_ee_delta_cosine_to_applied_step"]) > 0.5,
                ),
                "observed_ee_delta_follows_final_command_rate": _rate(
                    ep_rows,
                    lambda r: np.isfinite(float(r["observed_ee_delta_cosine_to_final_command"])) and float(r["observed_ee_delta_cosine_to_final_command"]) > 0.5,
                ),
                "step_too_small_rate": _rate(ep_rows, lambda r: "step_too_small" in r["flags"]),
                "alignment_ready_for_handoff_rate": _rate(ep_rows, lambda r: r["alignment_ready_for_handoff"]),
                "alignment_z_ready_rate": _rate(ep_rows, lambda r: r["alignment_z_ready"]),
                "alignment_yaw_ready_rate": _rate(ep_rows, lambda r: r["alignment_yaw_ready"]),
                "planner_close_blocked_rows": int(sum(1 for r in ep_rows if r["planner_gripper_close_blocked"])),
            }
        )
    for flag, flag_rows in sorted(by_flag.items()):
        report["by_flag"].append(
            {
                "flag": flag,
                "rows": int(len(flag_rows)),
                "oracle_horizon_contraction_rate": _rate(flag_rows, lambda r: r["horizon_xy_contracted"]),
                "near_grasp_after_rate": _rate(flag_rows, lambda r: r["near_grasp_after"]),
                "observed_ee_delta_follows_applied_step_rate": _rate(
                    flag_rows,
                    lambda r: np.isfinite(float(r["observed_ee_delta_cosine_to_applied_step"])) and float(r["observed_ee_delta_cosine_to_applied_step"]) > 0.5,
                ),
            }
        )
    for row in diag_rows[:]:
        if row["flags"] and len(report["representative_rows"]) < 8:
            report["representative_rows"].append(
                {
                    "episode_idx": row["episode_idx"],
                    "step": row["step"],
                    "flags": row["flags"],
                    "privileged_residual_xy": row["privileged_residual_xy"],
                    "localizer_proxy_xy": row["localizer_proxy_xy"],
                    "runtime_estimator_xy": row["runtime_estimator_xy"],
                    "applied_xy_step_local": row["applied_xy_step_local"],
                    "observed_ee_delta_local_xy": row["observed_ee_delta_local_xy"],
                    "horizon_xy_delta": row["horizon_xy_delta"],
                    "proxy_cosine_to_privileged_residual": row["proxy_cosine_to_privileged_residual"],
                    "runtime_estimator_cosine_to_privileged_residual": row["runtime_estimator_cosine_to_privileged_residual"],
                    "observed_ee_delta_cosine_to_applied_step": row["observed_ee_delta_cosine_to_applied_step"],
                    "observed_ee_delta_cosine_to_final_command": row["observed_ee_delta_cosine_to_final_command"],
                    "applied_step_to_residual_ratio": row["applied_step_to_residual_ratio"],
                }
            )
    return report, diag_rows


def write_report(report: Mapping[str, Any], rows: list[Mapping[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "xy_correction_hard_validation.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(output_dir / "xy_correction_hard_validation_rows.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    lines = [
        "# XY Correction Hard Validation",
        "",
        f"- active_rows: `{report['active_rows']}`",
        f"- oracle_horizon_contraction_rate: `{report['oracle_horizon_contraction_rate']:.3f}`",
        f"- near_grasp_after_rate: `{report['near_grasp_after_rate']:.3f}`",
        f"- overshoot_rate: `{report['overshoot_rate']:.3f}`",
        f"- proxy_aligned_with_privileged_residual_rate: `{report['proxy_aligned_with_privileged_residual_rate']:.3f}`",
        f"- runtime_estimator_aligned_with_privileged_residual_rate: `{report['runtime_estimator_aligned_with_privileged_residual_rate']:.3f}`",
        f"- applied_step_aligned_with_privileged_residual_rate: `{report['applied_step_aligned_with_privileged_residual_rate']:.3f}`",
        f"- observed_ee_delta_follows_applied_step_rate: `{report['observed_ee_delta_follows_applied_step_rate']:.3f}`",
        f"- observed_ee_delta_follows_final_command_rate: `{report['observed_ee_delta_follows_final_command_rate']:.3f}`",
        f"- step_too_small_rate: `{report['step_too_small_rate']:.3f}`",
        f"- takeover_sessions: `{report['takeover_sessions']}`",
        f"- terminal_takeover_sessions: `{report['terminal_takeover_sessions']}`",
        f"- alignment_success_rate: `{report['alignment_success_rate']:.3f}`",
        f"- safe_abstain_rate: `{report['safe_abstain_rate']:.3f}`",
        f"- failed_terminal_rate: `{report['failed_terminal_rate']:.3f}`",
        f"- alignment_ready_for_handoff_rate: `{report['alignment_ready_for_handoff_rate']:.3f}`",
        f"- alignment_z_ready_rate: `{report['alignment_z_ready_rate']:.3f}`",
        f"- alignment_yaw_ready_rate: `{report['alignment_yaw_ready_rate']:.3f}`",
        f"- handoff_allowed_rows: `{report['handoff_allowed_rows']}`",
        f"- planner_close_requested_rows: `{report['planner_close_requested_rows']}`",
        f"- planner_close_blocked_rows: `{report['planner_close_blocked_rows']}`",
        f"- planner_close_blocked_rate: `{report['planner_close_blocked_rate']:.3f}`",
        f"- decision: `{report['decision']}`",
        f"- flag_counts: `{report['flag_counts']}`",
        f"- handoff_block_reason_counts: `{report['handoff_block_reason_counts']}`",
        "",
        "## By Episode",
    ]
    for item in report["by_episode"]:
        lines.append(
            f"- ep`{item['episode_idx']:03d}` active=`{item['active_rows']}` "
            f"contract=`{item['oracle_horizon_contraction_rate']:.3f}` near=`{item['near_grasp_after_rate']:.3f}` "
            f"proxy_align=`{item['proxy_aligned_with_privileged_residual_rate']:.3f}` "
            f"est_align=`{item['runtime_estimator_aligned_with_privileged_residual_rate']:.3f}` "
            f"step_align=`{item['applied_step_aligned_with_privileged_residual_rate']:.3f}` "
            f"observed_follow=`{item['observed_ee_delta_follows_applied_step_rate']:.3f}` "
            f"final_follow=`{item['observed_ee_delta_follows_final_command_rate']:.3f}` "
            f"small_step=`{item['step_too_small_rate']:.3f}` "
            f"handoff=`{item['alignment_ready_for_handoff_rate']:.3f}` "
            f"z_ready=`{item['alignment_z_ready_rate']:.3f}` "
            f"yaw_ready=`{item['alignment_yaw_ready_rate']:.3f}` "
            f"close_blocked=`{item['planner_close_blocked_rows']}`"
        )
    lines.extend(["", "## By Flag"])
    for item in report["by_flag"]:
        lines.append(
            f"- `{item['flag']}` rows=`{item['rows']}` "
            f"contract=`{item['oracle_horizon_contraction_rate']:.3f}` "
            f"near=`{item['near_grasp_after_rate']:.3f}` "
            f"observed_follow=`{item['observed_ee_delta_follows_applied_step_rate']:.3f}`"
        )
    lines.extend(["", "## Representative Rows"])
    for row in report["representative_rows"]:
        lines.append(
            f"- ep`{row['episode_idx']:03d}` step=`{row['step']}` flags=`{row['flags']}` "
            f"priv=`{row['privileged_residual_xy']}` proxy=`{row['localizer_proxy_xy']}` "
            f"est=`{row['runtime_estimator_xy']}` "
            f"step=`{row['applied_xy_step_local']}` ee_delta=`{row['observed_ee_delta_local_xy']}` "
            f"horizon_delta=`{row['horizon_xy_delta']:.6f}` "
            f"proxy_cos=`{row['proxy_cosine_to_privileged_residual']:.3f}` "
            f"est_cos=`{row['runtime_estimator_cosine_to_privileged_residual']:.3f}` "
            f"ee_step_cos=`{row['observed_ee_delta_cosine_to_applied_step']:.3f}` "
            f"ee_final_cos=`{row['observed_ee_delta_cosine_to_final_command']:.3f}` "
            f"step_ratio=`{row['applied_step_to_residual_ratio']:.3f}`"
        )
    (output_dir / "xy_correction_hard_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--runtime_obs_dir", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    rows = load_trace_rows(args.trace_dir)
    pose_cache = _load_runtime_pose_cache(args.runtime_obs_dir)
    report, diag_rows = summarize(rows, pose_cache)
    write_report(report, diag_rows, args.output_dir)


if __name__ == "__main__":
    main()
