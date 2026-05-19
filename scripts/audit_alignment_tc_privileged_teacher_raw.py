#!/usr/bin/env python3
"""Audit vNext privileged-teacher raw near/micro collection files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _as_float_array(data: dict, key: str, default=np.nan) -> np.ndarray:
    if key not in data:
        return np.asarray([default], dtype=np.float32)
    arr = np.asarray(data[key])
    if arr.dtype.kind in {"U", "S", "O"}:
        return np.asarray([default], dtype=np.float32)
    return arr.astype(np.float32)


def _as_str_list(data: dict, key: str) -> list[str]:
    if key not in data:
        return []
    arr = np.asarray(data[key])
    return [str(x.item() if hasattr(x, "item") else x) for x in arr.reshape(-1)]


def _fit_rows(arr: np.ndarray, rows: int, fill_value=np.nan) -> np.ndarray:
    arr = np.asarray(arr)
    if rows <= 0:
        return arr.reshape(0, *arr.shape[1:]) if arr.ndim > 0 else np.asarray([], dtype=arr.dtype if hasattr(arr, "dtype") else np.float32)
    if arr.ndim == 0:
        return np.full((rows,), arr.item(), dtype=arr.dtype if hasattr(arr, "dtype") else np.float32)
    if arr.shape[0] == rows:
        return arr
    if arr.size == 1:
        return np.full((rows,) + tuple(arr.shape[1:]), arr.reshape(()).item(), dtype=arr.dtype)
    return np.full((rows,) + tuple(arr.shape[1:]), fill_value, dtype=arr.dtype if arr.dtype.kind != "O" else np.float32)


def _maybe_float(value):
    try:
        if value is None:
            return float("nan")
        return float(np.asarray(value).reshape(()))
    except Exception:
        return float("nan")


def _rate(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    return float(mask.mean()) if mask.size else float("nan")


def _stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def audit(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files}
    rows = int(next(iter(data.values())).shape[0]) if data else 0
    delta = _as_float_array(data, "privileged_current_to_target_delta_local")
    if delta.ndim == 1:
        delta = delta.reshape(rows, -1) if rows else delta.reshape(0, -1)
    finite_delta = delta.shape[1] >= 6 and np.all(np.isfinite(delta[:, :6]), axis=1) if rows else np.zeros(0, dtype=bool)
    xy = np.linalg.norm(delta[:, :2], axis=1) if rows and delta.shape[1] >= 2 else np.asarray([])
    abs_z = np.abs(delta[:, 2]) if rows and delta.shape[1] >= 3 else np.asarray([])
    yaw = np.abs(delta[:, 5]) if rows and delta.shape[1] >= 6 else np.asarray([])
    stage_buckets = _as_str_list(data, "stage_bucket")
    target_buckets = _as_str_list(data, "target_delta_gate_bucket")
    sources = _as_str_list(data, "privileged_target_provider_source")
    teacher_sources = _as_str_list(data, "teacher_action_source")
    close_actions = _as_str_list(data, "teacher_close_action")
    close_reasons = _as_str_list(data, "teacher_close_gate_reason")
    close_failures = _as_str_list(data, "close_failure_reason")
    stop_reasons = _as_str_list(data, "stop_reason")
    motion_phases = _as_str_list(data, "teacher_motion_phase")
    tc_motion_phases = _as_str_list(data, "teacher_tc_motion_phase")
    close_ready_all = _as_float_array(data, "teacher_close_ready_all", 0.0).reshape(-1)
    close_contact_depth = _as_float_array(data, "teacher_close_contact_ready_by_depth", 0.0).reshape(-1)
    close_contact_stage = _as_float_array(data, "teacher_close_contact_ready_by_stage", 0.0).reshape(-1)
    close_contact_geometry = _as_float_array(data, "teacher_close_contact_ready_by_geometry", 0.0).reshape(-1)
    close_contact_visibility = _as_float_array(data, "teacher_close_contact_visibility_ok", 0.0).reshape(-1)
    close_contact_confidence = _as_float_array(data, "teacher_close_contact_confidence", np.nan).reshape(-1)
    close_yaw_raw = _as_float_array(data, "teacher_close_yaw_raw", np.nan).reshape(-1)
    close_yaw_folded = _as_float_array(data, "teacher_close_yaw_folded", np.nan).reshape(-1)
    yaw_control_sign = _as_float_array(data, "teacher_yaw_control_sign", np.nan).reshape(-1)
    grasp_ready = _as_float_array(data, "teacher_grasp_ready", 0.0).reshape(-1)
    grasp_ready_score = _as_float_array(data, "teacher_grasp_readiness_score", np.nan).reshape(-1)
    grasp_contact_ready = _as_float_array(data, "teacher_grasp_contact_ready", 0.0).reshape(-1)
    object_in_finger_region = _as_float_array(data, "teacher_object_in_finger_region", 0.0).reshape(-1)
    expert_sequence_name = _as_str_list(data, "expert_sequence_name")
    expert_sequence_verified = _as_float_array(data, "expert_sequence_verified", 0.0).reshape(-1)
    expert_sequence_score = _as_float_array(data, "expert_sequence_score", np.nan).reshape(-1)
    yaw_imitation_enabled = _as_float_array(data, "yaw_imitation_enabled", 0.0).reshape(-1)
    verify_object_pose_available = _as_float_array(data, "teacher_verify_object_pose_available", 0.0).reshape(-1)
    verify_object_pose_source = _as_str_list(data, "teacher_verify_object_pose_source")
    verify_object_is_grasped = _as_float_array(data, "teacher_verify_object_is_grasped", 0.0).reshape(-1)
    verify_follow_distance = _as_float_array(data, "teacher_verify_follow_distance", np.nan).reshape(-1)
    verify_fail_reason = _as_str_list(data, "teacher_verify_fail_reason")
    attached_after_close = _fit_rows(_as_float_array(data, "teacher_attached_after_close", 0.0).reshape(-1), rows, 0.0).reshape(-1)
    grasped_object_count = _fit_rows(_as_float_array(data, "teacher_grasped_object_count", 0.0).reshape(-1), rows, 0.0).reshape(-1)
    grasped_target_match = _fit_rows(_as_float_array(data, "teacher_grasped_target_handle_match", 0.0).reshape(-1), rows, 0.0).reshape(-1)
    object_to_gripper_delta = _as_float_array(data, "teacher_object_to_gripper_delta_local_6d", np.nan)
    if object_to_gripper_delta.ndim == 1:
        if object_to_gripper_delta.size == rows * 6 and rows:
            object_to_gripper_delta = object_to_gripper_delta.reshape(rows, 6)
        else:
            object_to_gripper_delta = np.full((rows, 6), np.nan, dtype=np.float32)
    demo_basin_distance = _fit_rows(_as_float_array(data, "teacher_demo_basin_distance", np.nan).reshape(-1), rows, np.nan).reshape(-1)
    close_basin_source = _as_str_list(data, "teacher_close_basin_source")
    close_failure_reason = _as_str_list(data, "teacher_close_failure_reason")
    takeover_origin = _as_str_list(data, "takeover_origin")
    teacher_action = _as_float_array(data, "teacher_residual_action_4d", 0.0)
    if teacher_action.ndim == 1 and rows:
        teacher_action = teacher_action.reshape(rows, -1)
    teacher_norm = np.linalg.norm(teacher_action[:, :3], axis=1) if rows and teacher_action.shape[1] >= 3 else np.asarray([])
    teacher_yaw = np.abs(teacher_action[:, 3]) if rows and teacher_action.shape[1] >= 4 else np.asarray([])
    close_now_mask = np.asarray([a == "close_now" for a in close_actions], dtype=bool) if close_actions else np.zeros((rows,), dtype=bool)
    close_related_mask = np.asarray(
        [a in {"close_now", "verify_lift", "verified_success"} for a in close_actions],
        dtype=bool,
    ) if close_actions else np.zeros((rows,), dtype=bool)
    verified_mask = _as_float_array(data, "teacher_grasp_verified", 0.0).reshape(-1) > 0.5
    success_label_mask = _as_float_array(data, "close_success_label", 0.0).reshape(-1) > 0.5
    episode_index = _as_float_array(data, "episode_index", np.arange(rows, dtype=np.float32)).reshape(-1).astype(np.int64) if rows else np.zeros((0,), dtype=np.int64)
    verified_episode_count = 0
    success_episode_count = 0
    false_success_count = 0
    if rows:
        for ep in np.unique(episode_index):
            mask = episode_index == int(ep)
            ep_verified = bool(np.any(verified_mask[mask]))
            ep_success = bool(np.any(success_label_mask[mask]))
            verified_episode_count += int(ep_verified)
            success_episode_count += int(ep_success)
            if ep_success and not ep_verified:
                false_success_count += 1
    close_now_with_contact_ready_rate = float("nan")
    if close_now_mask.any():
        close_now_with_contact_ready_rate = _rate(_as_float_array(data, "teacher_close_contact_ready", 0.0).reshape(-1)[close_now_mask] > 0.5)
    close_now_grasp_ready_rate = float("nan")
    if close_now_mask.any():
        close_now_grasp_ready_rate = _rate(grasp_ready[close_now_mask] > 0.5)
    grasp_readiness_at_close = grasp_ready_score[close_now_mask] if close_now_mask.any() else np.asarray([])
    pre16_mask = _as_float_array(data, "time_to_verified_grasp", -1.0).reshape(-1)
    pre16_mask = np.logical_and(pre16_mask >= 0, pre16_mask <= 15)
    report = {
        "path": str(path),
        "rows": rows,
        "finite_privileged_delta_rate": _rate(finite_delta),
        "stage_bucket_counts": dict(Counter(stage_buckets)),
        "target_delta_gate_bucket_counts": dict(Counter(target_buckets)),
        "privileged_source_counts": dict(Counter(sources)),
        "teacher_action_source_counts": dict(Counter(teacher_sources)),
        "pregrasp_source_rate": _rate(np.asarray(["pregrasp" in s for s in sources], dtype=bool)) if sources else 0.0,
        "teacher_collect_active_rate": _rate(_as_float_array(data, "teacher_collect_active", 0.0).reshape(-1) > 0.5),
        "invalid_action_rate": _rate(_as_float_array(data, "invalid_action", 0.0).reshape(-1) > 0.5),
        "workspace_violation_rate": _rate(_as_float_array(data, "workspace_violation", 0.0).reshape(-1) > 0.0),
        "force_spike_rate": _rate(_as_float_array(data, "force_spike", 0.0).reshape(-1) > 0.5),
        "success_label_rate": _rate(_as_float_array(data, "success_label", 0.0).reshape(-1) > 0.5),
        "close_success_label_rate": _rate(_as_float_array(data, "close_success_label", 0.0).reshape(-1) > 0.5),
        "teacher_close_ready_rate": _rate(_as_float_array(data, "teacher_close_ready", 0.0).reshape(-1) > 0.5),
        "teacher_close_ready_all_rate": _rate(close_ready_all > 0.5),
        "teacher_grasp_verified_rate": _rate(_as_float_array(data, "teacher_grasp_verified", 0.0).reshape(-1) > 0.5),
        "close_attempt_rate": _rate(np.asarray([a == "close_now" for a in close_actions], dtype=bool)) if close_actions else 0.0,
        "verify_lift_rate": _rate(np.asarray([a == "verify_lift" for a in close_actions], dtype=bool)) if close_actions else 0.0,
        "retry_rate": _rate(np.asarray([a == "retry" for a in close_actions], dtype=bool)) if close_actions else 0.0,
        "teacher_close_action_counts": dict(Counter(close_actions)),
        "teacher_close_gate_reason_counts": dict(Counter(close_reasons)),
        "close_failure_reason_counts": dict(Counter(close_failures)),
        "stop_reason_counts": dict(Counter(stop_reasons)),
        "takeover_origin_counts": dict(Counter(takeover_origin)),
        "teacher_motion_phase_counts": dict(Counter(motion_phases)),
        "teacher_tc_motion_phase_counts": dict(Counter(tc_motion_phases)),
        "teacher_improves_xy_rate": _rate(_as_float_array(data, "teacher_improves_xy", 0.0).reshape(-1) > 0.5),
        "teacher_improves_z_rate": _rate(_as_float_array(data, "teacher_improves_z", 0.0).reshape(-1) > 0.5),
        "teacher_improves_yaw_rate": _rate(_as_float_array(data, "teacher_improves_yaw", 0.0).reshape(-1) > 0.5),
        "teacher_improves_two_axis_rate": _rate(_as_float_array(data, "teacher_improves_two_axis", 0.0).reshape(-1) > 0.5),
        "teacher_all_improves_rate": _rate(_as_float_array(data, "teacher_all_improves", 0.0).reshape(-1) > 0.5),
        "teacher_close_contact_ready_rate": _rate(_as_float_array(data, "teacher_close_contact_ready", 0.0).reshape(-1) > 0.5),
        "teacher_close_contact_ready_by_depth_rate": _rate(close_contact_depth > 0.5),
        "teacher_close_contact_ready_by_stage_rate": _rate(close_contact_stage > 0.5),
        "teacher_close_contact_ready_by_geometry_rate": _rate(close_contact_geometry > 0.5),
        "teacher_close_contact_visibility_ok_rate": _rate(close_contact_visibility > 0.5),
        "teacher_close_contact_confidence_stats": _stats(close_contact_confidence),
        "teacher_grasp_ready_rate": _rate(grasp_ready > 0.5),
        "teacher_grasp_contact_ready_rate": _rate(grasp_contact_ready > 0.5),
        "teacher_object_in_finger_region_rate": _rate(object_in_finger_region > 0.5),
        "teacher_grasp_readiness_score_stats": _stats(grasp_ready_score),
        "teacher_grasp_readiness_score_at_close_stats": _stats(grasp_readiness_at_close),
        "teacher_close_yaw_raw_stats": _stats(np.abs(close_yaw_raw)),
        "teacher_close_yaw_folded_stats": _stats(np.abs(close_yaw_folded)),
        "teacher_yaw_control_sign_stats": _stats(yaw_control_sign),
        "teacher_yaw_imitation_enabled_rate": _rate(yaw_imitation_enabled > 0.5),
        "teacher_verify_lift_streak_stats": _stats(_as_float_array(data, "teacher_verify_lift_streak", 0.0).reshape(-1)),
        "teacher_verify_object_pose_available_rate": _rate(verify_object_pose_available > 0.5),
        "teacher_verify_object_is_grasped_rate": _rate(verify_object_is_grasped > 0.5),
        "teacher_verify_object_pose_source_counts": dict(Counter(verify_object_pose_source)),
        "teacher_verify_follow_distance_stats": _stats(verify_follow_distance),
        "teacher_verify_fail_reason_counts": dict(Counter(verify_fail_reason)),
        "teacher_attached_after_close_rate": _rate(attached_after_close[close_now_mask] > 0.5) if close_now_mask.any() and attached_after_close.shape[0] == rows else _rate(attached_after_close > 0.5),
        "teacher_grasped_object_count_stats": _stats(grasped_object_count[close_related_mask]) if close_related_mask.any() and grasped_object_count.shape[0] == rows else _stats(grasped_object_count),
        "teacher_grasped_target_handle_match_rate": _rate(grasped_target_match[close_now_mask] > 0.5) if close_now_mask.any() and grasped_target_match.shape[0] == rows else _rate(grasped_target_match > 0.5),
        "teacher_object_to_gripper_xy_stats": _stats(np.linalg.norm(object_to_gripper_delta[close_related_mask, :2], axis=1)) if rows and close_related_mask.any() and object_to_gripper_delta.shape[1] >= 2 else {"count": 0},
        "teacher_object_to_gripper_abs_z_stats": _stats(np.abs(object_to_gripper_delta[close_related_mask, 2])) if rows and close_related_mask.any() and object_to_gripper_delta.shape[1] >= 3 else {"count": 0},
        "teacher_object_to_gripper_yaw_stats": _stats(np.abs(object_to_gripper_delta[close_related_mask, 5])) if rows and close_related_mask.any() and object_to_gripper_delta.shape[1] >= 6 else {"count": 0},
        "teacher_demo_basin_distance_stats": _stats(demo_basin_distance[close_related_mask]) if close_related_mask.any() else _stats(demo_basin_distance),
        "teacher_close_basin_source_counts": dict(Counter(close_basin_source)),
        "teacher_close_failure_reason_counts": dict(Counter(close_failure_reason)) if close_failure_reason else dict(Counter(verify_fail_reason)),
        "expert_sequence_counts": dict(Counter(expert_sequence_name)),
        "expert_sequence_score_stats": _stats(expert_sequence_score),
        "verified_success_rate": float(verified_episode_count / max(len(np.unique(episode_index)) if rows else 1, 1)),
        "success_episode_rate": float(success_episode_count / max(len(np.unique(episode_index)) if rows else 1, 1)),
        "false_success_count": int(false_success_count),
        "target_xy_stats": _stats(xy),
        "target_abs_z_stats": _stats(abs_z),
        "target_yaw_stats": _stats(yaw),
        "teacher_pos_action_norm_stats": _stats(teacher_norm),
        "teacher_yaw_action_abs_stats": _stats(teacher_yaw),
        "close_now_with_contact_ready_rate": float(close_now_with_contact_ready_rate),
        "close_now_with_grasp_ready_rate": float(close_now_grasp_ready_rate),
        "yaw_sign_improve_rate_by_bucket": {},
        "verified_success_pre16_xy_improve_rate": _rate(_as_float_array(data, "teacher_improves_xy", 0.0).reshape(-1)[pre16_mask] > 0.5) if pre16_mask.any() else float("nan"),
        "verified_success_pre16_z_improve_rate": _rate(_as_float_array(data, "teacher_improves_z", 0.0).reshape(-1)[pre16_mask] > 0.5) if pre16_mask.any() else float("nan"),
        "verified_success_pre16_yaw_improve_rate": _rate(_as_float_array(data, "teacher_improves_yaw", 0.0).reshape(-1)[pre16_mask] > 0.5) if pre16_mask.any() else float("nan"),
    }
    for bucket in ("near_contact_refine", "micro_contact_refine"):
        mask = np.asarray([b == bucket for b in stage_buckets], dtype=bool)
        if not mask.any():
            continue
        report[f"{bucket}_rows"] = int(mask.sum())
        report[f"{bucket}_teacher_two_axis_improve_rate"] = _rate(
            _as_float_array(data, "teacher_improves_two_axis", 0.0).reshape(-1)[mask] > 0.5
        )
        report[f"{bucket}_teacher_xy_improve_rate"] = _rate(
            _as_float_array(data, "teacher_improves_xy", 0.0).reshape(-1)[mask] > 0.5
        )
        report[f"{bucket}_teacher_yaw_improve_rate"] = _rate(
            _as_float_array(data, "teacher_improves_yaw", 0.0).reshape(-1)[mask] > 0.5
        )
        report[f"{bucket}_teacher_close_ready_all_rate"] = _rate(close_ready_all[mask] > 0.5)
        report[f"{bucket}_teacher_close_contact_ready_rate"] = _rate(
            _as_float_array(data, "teacher_close_contact_ready", 0.0).reshape(-1)[mask] > 0.5
        )
        if takeover_origin:
            report[f"{bucket}_takeover_origin_counts"] = dict(
                Counter(
                    [takeover_origin[i] for i, ok in enumerate(mask) if ok and i < len(takeover_origin)]
                )
            )
        report[f"{bucket}_yaw_sign_improve_rate"] = _rate(
            _as_float_array(data, "teacher_improves_yaw", 0.0).reshape(-1)[mask] > 0.5
        )
        if close_actions:
            report[f"{bucket}_close_attempt_rate"] = _rate(
                np.asarray([a == "close_now" for a in close_actions], dtype=bool)[mask]
            )
            report[f"{bucket}_verified_grasp_rate"] = _rate(
                _as_float_array(data, "teacher_grasp_verified", 0.0).reshape(-1)[mask] > 0.5
            )
        report["yaw_sign_improve_rate_by_bucket"][bucket] = report[f"{bucket}_yaw_sign_improve_rate"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--grasp_basin_profile_json", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.npz)
    if args.grasp_basin_profile_json is not None and args.grasp_basin_profile_json.exists():
        profile = json.loads(args.grasp_basin_profile_json.read_text())
        teacher_close_xy = _maybe_float(report.get("teacher_object_to_gripper_xy_stats", {}).get("p90"))
        teacher_close_z = _maybe_float(report.get("teacher_object_to_gripper_abs_z_stats", {}).get("p90"))
        teacher_close_yaw = _maybe_float(report.get("teacher_object_to_gripper_yaw_stats", {}).get("p90"))
        if isinstance(profile, dict):
            denom_xy = _maybe_float(profile.get("close_xy_threshold", np.nan))
            denom_z = _maybe_float(profile.get("close_abs_z_threshold", np.nan))
            denom_yaw = _maybe_float(profile.get("close_yaw_threshold", np.nan))
            if np.isfinite(denom_xy) and np.isfinite(teacher_close_xy):
                report["teacher_vs_demo_xy_p90_ratio"] = float(teacher_close_xy / max(denom_xy, 1e-6))
            if np.isfinite(denom_z) and np.isfinite(teacher_close_z):
                report["teacher_vs_demo_z_p90_ratio"] = float(teacher_close_z / max(denom_z, 1e-6))
            if np.isfinite(denom_yaw) and np.isfinite(teacher_close_yaw):
                report["teacher_vs_demo_yaw_p90_ratio"] = float(teacher_close_yaw / max(denom_yaw, 1e-6))
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
