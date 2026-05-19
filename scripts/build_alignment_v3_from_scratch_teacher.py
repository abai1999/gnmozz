#!/usr/bin/env python3
"""Build a from-scratch privileged direct-control teacher for alignment_v3.

This builder consumes raw privileged rollout traces, filters near/micro rows,
and constructs a direct local residual teacher by auditing a dense residual
search space against the privileged target pose.

The teacher is trusted-but-audited, not assumed absolute truth. If the audit
does not pass, the script exits non-zero and no dataset should be built.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from build_pose_candidate_dataset import apply_local_offset_to_pose, pose_delta_local_between


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _parse_floats(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty float list: {text!r}")
    return np.asarray(vals, dtype=np.float32)


def _finite_pose(pose: np.ndarray | None) -> bool:
    if pose is None:
        return False
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    return arr.size == 7 and np.all(np.isfinite(arr))


def _row_int(row: dict[str, np.ndarray], key: str, default: int = -1) -> int:
    if key not in row:
        return int(default)
    try:
        return int(np.asarray(row[key]).reshape(()))
    except Exception:
        return int(default)


def _row_float(row: dict[str, np.ndarray], key: str, default: float = 0.0) -> float:
    if key not in row:
        return float(default)
    try:
        return float(np.asarray(row[key]).reshape(()))
    except Exception:
        return float(default)


def _row_vec(row: dict[str, np.ndarray], key: str, min_size: int = 1) -> np.ndarray | None:
    if key not in row:
        return None
    try:
        arr = np.asarray(row[key], dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < int(min_size):
        return None
    return arr


def _optional_binary_row(
    row: dict[str, np.ndarray],
    *keys: str,
    default: float = 0.0,
) -> float:
    for key in keys:
        if key not in row:
            continue
        try:
            arr = np.asarray(row[key], dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if arr.size:
            return float(arr[0])
    return float(default)


def _target_source_rank(reason: str) -> int | None:
    reason = str(reason or "")
    for part in reason.split(","):
        part = part.strip()
        if part.startswith("rank="):
            try:
                return int(part.split("=", 1)[1])
            except Exception:
                return None
    return None


def _target_candidate_order(
    stage_bucket: str,
    stage_target_mode: int,
    contact_state: int,
) -> list[tuple[str, str]]:
    stage_bucket = str(stage_bucket)
    stage_target_mode = int(stage_target_mode)
    contact_state = int(contact_state)
    if stage_bucket == "micro_contact_refine":
        return [
            ("privileged_object_anchor_pose_7d", "object_anchor"),
            ("privileged_grasp_commit_target_pose_7d", "grasp_commit_target"),
            ("privileged_motion_target_pose_7d", "motion_target"),
            ("privileged_pregrasp_target_pose_7d", "pregrasp_target"),
            ("privileged_basin_center_pose_7d", "basin_center"),
        ]
    if stage_bucket == "near_alignment":
        if stage_target_mode == 3 or contact_state >= 2:
            return [
                ("privileged_object_anchor_pose_7d", "object_anchor"),
                ("privileged_grasp_commit_target_pose_7d", "grasp_commit_target"),
                ("privileged_motion_target_pose_7d", "motion_target"),
                ("privileged_pregrasp_target_pose_7d", "pregrasp_target"),
                ("privileged_basin_center_pose_7d", "basin_center"),
            ]
        return [
            ("privileged_motion_target_pose_7d", "motion_target"),
            ("privileged_pregrasp_target_pose_7d", "pregrasp_target"),
            ("privileged_grasp_commit_target_pose_7d", "grasp_commit_target"),
            ("privileged_object_anchor_pose_7d", "object_anchor"),
            ("privileged_basin_center_pose_7d", "basin_center"),
        ]
    if stage_target_mode == 3 or contact_state >= 2:
        return [
            ("privileged_object_anchor_pose_7d", "object_anchor"),
            ("privileged_grasp_commit_target_pose_7d", "grasp_commit_target"),
            ("privileged_motion_target_pose_7d", "motion_target"),
            ("privileged_pregrasp_target_pose_7d", "pregrasp_target"),
            ("privileged_basin_center_pose_7d", "basin_center"),
        ]
    if stage_target_mode == 2:
        return [
            ("privileged_object_anchor_pose_7d", "object_anchor"),
            ("privileged_grasp_commit_target_pose_7d", "grasp_commit_target"),
            ("privileged_motion_target_pose_7d", "motion_target"),
            ("privileged_pregrasp_target_pose_7d", "pregrasp_target"),
            ("privileged_basin_center_pose_7d", "basin_center"),
        ]
    if stage_target_mode == 1 or contact_state >= 1:
        return [
            ("privileged_pregrasp_target_pose_7d", "pregrasp_target"),
            ("privileged_motion_target_pose_7d", "motion_target"),
            ("privileged_grasp_commit_target_pose_7d", "grasp_commit_target"),
            ("privileged_object_anchor_pose_7d", "object_anchor"),
            ("privileged_basin_center_pose_7d", "basin_center"),
        ]
    return [
        ("privileged_motion_target_pose_7d", "motion_target"),
        ("privileged_pregrasp_target_pose_7d", "pregrasp_target"),
        ("privileged_grasp_commit_target_pose_7d", "grasp_commit_target"),
        ("privileged_object_anchor_pose_7d", "object_anchor"),
        ("privileged_basin_center_pose_7d", "basin_center"),
    ]


def _choose_target_pose(
    row: dict[str, np.ndarray],
    current_pose: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str, str]:
    stage_target_mode = _row_int(row, "stage_target_mode", 0)
    contact_state = _row_int(row, "contact_state", 0)
    contact_like = bool(stage_target_mode == 3)
    bucket_hint = "unknown"
    motion_pose = row.get("privileged_motion_target_pose_7d", row.get("motion_target_pose_7d"))
    if current_pose is not None and _finite_pose(motion_pose):
        cur_delta_motion = pose_delta_local_between(np.asarray(current_pose, dtype=np.float32).reshape(7), np.asarray(motion_pose, dtype=np.float32).reshape(7)).astype(np.float32)
        cur_xy = float(np.linalg.norm(cur_delta_motion[:2]))
        cur_z = float(abs(cur_delta_motion[2]))
        cur_yaw = float(abs(cur_delta_motion[5]))
        if contact_like and cur_xy <= 0.020 and cur_z <= 0.040 and cur_yaw <= 0.150:
            bucket_hint = "micro_contact_refine"
        elif cur_xy <= 0.020 and cur_z <= 0.040 and cur_yaw <= 0.350:
            bucket_hint = "near_alignment"
        else:
            bucket_hint = "coarse"

    stage_bucket = bucket_hint if bucket_hint != "unknown" else ("micro_contact_refine" if contact_like else "near_alignment")
    for rank, (key, label) in enumerate(_target_candidate_order(stage_bucket, stage_target_mode, contact_state)):
        if key in row and _finite_pose(row[key]):
            reason = f"stage_bucket={stage_bucket},stage_target_mode={stage_target_mode},contact_state={contact_state},rank={rank}"
            return np.asarray(row[key], dtype=np.float32).reshape(7), label, reason
    return None, "missing", f"stage_bucket={stage_bucket},stage_target_mode={stage_target_mode},contact_state={contact_state}"


def _generate_teacher_grid(
    dx_values: np.ndarray,
    dy_values: np.ndarray,
    dz_values: np.ndarray,
    dyaw_values: np.ndarray,
) -> np.ndarray:
    grid = np.array(list(product(dx_values, dy_values, dz_values, dyaw_values)), dtype=np.float32)
    residuals = np.zeros((grid.shape[0] + 1, 6), dtype=np.float32)
    residuals[1:, 0] = grid[:, 0]
    residuals[1:, 1] = grid[:, 1]
    residuals[1:, 2] = grid[:, 2]
    residuals[1:, 5] = grid[:, 3]
    return residuals


def _evaluate_residuals(current_pose: np.ndarray, target_pose: np.ndarray, residuals: np.ndarray) -> dict[str, np.ndarray]:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    residuals = np.asarray(residuals, dtype=np.float32).reshape(-1, 6)

    n = int(residuals.shape[0])
    out = {
        "residual_6d": residuals.astype(np.float32),
        "residual_4d": np.zeros((n, 4), dtype=np.float32),
        "post_xy": np.zeros((n,), dtype=np.float32),
        "post_z": np.zeros((n,), dtype=np.float32),
        "post_yaw": np.zeros((n,), dtype=np.float32),
        "improves_xy": np.zeros((n,), dtype=np.float32),
        "improves_z": np.zeros((n,), dtype=np.float32),
        "improves_yaw": np.zeros((n,), dtype=np.float32),
        "all_improves": np.zeros((n,), dtype=np.float32),
        "overshoot_any": np.zeros((n,), dtype=np.float32),
        "pos_norm": np.zeros((n,), dtype=np.float32),
        "yaw_abs": np.zeros((n,), dtype=np.float32),
        "action_pos_norm": np.zeros((n,), dtype=np.float32),
        "action_yaw_abs": np.zeros((n,), dtype=np.float32),
    }

    if current_pose.ndim == 1 and target_pose.ndim == 1:
        current_pose = current_pose.reshape(7)
        target_pose = target_pose.reshape(7)
        cur = pose_delta_local_between(current_pose, target_pose).astype(np.float32)
        cur_xy = float(np.linalg.norm(cur[:2]))
        cur_z = float(abs(cur[2]))
        cur_yaw = float(abs(cur[5]))
        for i in range(n):
            nxt = apply_local_offset_to_pose(current_pose, residuals[i])
            post = pose_delta_local_between(nxt, target_pose).astype(np.float32)
            post_xy = float(np.linalg.norm(post[:2]))
            post_z = float(abs(post[2]))
            post_yaw = float(abs(post[5]))
            out["residual_4d"][i] = np.asarray([residuals[i, 0], residuals[i, 1], residuals[i, 2], residuals[i, 5]], dtype=np.float32)
            out["post_xy"][i] = post_xy
            out["post_z"][i] = post_z
            out["post_yaw"][i] = post_yaw
            out["improves_xy"][i] = float(post_xy < cur_xy)
            out["improves_z"][i] = float(post_z < cur_z)
            out["improves_yaw"][i] = float(post_yaw < cur_yaw)
            out["all_improves"][i] = out["improves_xy"][i] * out["improves_z"][i] * out["improves_yaw"][i]
            out["overshoot_any"][i] = float(post_xy > cur_xy + 1e-8 or post_z > cur_z + 1e-8 or post_yaw > cur_yaw + 1e-8)
            out["pos_norm"][i] = float(np.linalg.norm(residuals[i, :3]))
            out["yaw_abs"][i] = float(abs(residuals[i, 5]))
            out["action_pos_norm"][i] = out["pos_norm"][i]
            out["action_yaw_abs"][i] = out["yaw_abs"][i]
        return out

    current_pose = current_pose.reshape(-1, 7)
    target_pose = target_pose.reshape(-1, 7)
    if current_pose.shape[0] != n or target_pose.shape[0] != n:
        raise ValueError(
            f"batch shape mismatch: current_pose={current_pose.shape}, target_pose={target_pose.shape}, residuals={residuals.shape}"
        )

    for i in range(n):
        cur = pose_delta_local_between(current_pose[i], target_pose[i]).astype(np.float32)
        cur_xy = float(np.linalg.norm(cur[:2]))
        cur_z = float(abs(cur[2]))
        cur_yaw = float(abs(cur[5]))
        nxt = apply_local_offset_to_pose(current_pose[i], residuals[i])
        post = pose_delta_local_between(nxt, target_pose[i]).astype(np.float32)
        post_xy = float(np.linalg.norm(post[:2]))
        post_z = float(abs(post[2]))
        post_yaw = float(abs(post[5]))
        out["residual_4d"][i] = np.asarray([residuals[i, 0], residuals[i, 1], residuals[i, 2], residuals[i, 5]], dtype=np.float32)
        out["post_xy"][i] = post_xy
        out["post_z"][i] = post_z
        out["post_yaw"][i] = post_yaw
        out["improves_xy"][i] = float(post_xy < cur_xy)
        out["improves_z"][i] = float(post_z < cur_z)
        out["improves_yaw"][i] = float(post_yaw < cur_yaw)
        out["all_improves"][i] = out["improves_xy"][i] * out["improves_z"][i] * out["improves_yaw"][i]
        out["overshoot_any"][i] = float(post_xy > cur_xy + 1e-8 or post_z > cur_z + 1e-8 or post_yaw > cur_yaw + 1e-8)
        out["pos_norm"][i] = float(np.linalg.norm(residuals[i, :3]))
        out["yaw_abs"][i] = float(abs(residuals[i, 5]))
        out["action_pos_norm"][i] = out["pos_norm"][i]
        out["action_yaw_abs"][i] = out["yaw_abs"][i]
    return out


def _bounded_servo_pseudo(
    delta_local: np.ndarray,
    *,
    k_xy: float,
    k_z: float,
    k_yaw: float,
    max_pos: float,
    max_yaw: float,
) -> np.ndarray:
    delta_local = np.asarray(delta_local, dtype=np.float32).reshape(6)
    residual = np.zeros((6,), dtype=np.float32)
    residual[:2] = delta_local[:2] * float(k_xy)
    residual[2] = delta_local[2] * float(k_z)
    residual[5] = delta_local[5] * float(k_yaw)

    pos_norm = float(np.linalg.norm(residual[:3]))
    if pos_norm > max_pos and pos_norm > 1e-8:
        residual[:3] *= float(max_pos / pos_norm)
    yaw_abs = float(abs(residual[5]))
    if yaw_abs > max_yaw and yaw_abs > 1e-8:
        residual[5] *= float(max_yaw / yaw_abs)
    return residual


def _score_tuple(metrics: dict[str, np.ndarray], idx: int, stage_bucket: str) -> tuple[float, ...]:
    if stage_bucket == "micro_contact_refine":
        joint_fail = 0.0 if (metrics["improves_xy"][idx] > 0.5 and metrics["improves_yaw"][idx] > 0.5) else (
            1.0 if (metrics["improves_xy"][idx] > 0.5 or metrics["improves_yaw"][idx] > 0.5) else 2.0
        )
        primary = float(metrics["post_xy"][idx] * 1.25 + metrics["post_z"][idx] * 0.35 + metrics["post_yaw"][idx] * 1.75)
        return (
            joint_fail,
            primary,
            float(metrics["post_xy"][idx]),
            float(metrics["post_yaw"][idx]),
            float(metrics["post_z"][idx]),
            float(metrics["action_pos_norm"][idx]),
            float(metrics["action_yaw_abs"][idx]),
            float(metrics["overshoot_any"][idx]),
            float(idx == 0),
        )
    primary = float(metrics["post_xy"][idx] * 1.0 + metrics["post_z"][idx] * 0.75 + metrics["post_yaw"][idx] * 1.15)
    return (
        primary,
        float(metrics["post_xy"][idx]),
        float(metrics["post_z"][idx]),
        float(metrics["post_yaw"][idx]),
        float(metrics["action_pos_norm"][idx]),
        float(metrics["action_yaw_abs"][idx]),
        float(metrics["overshoot_any"][idx]),
        float(idx == 0),
    )


def _clip_residual(residual: np.ndarray, max_pos: float, max_yaw: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float32).reshape(6).copy()
    pos_norm = float(np.linalg.norm(residual[:3]))
    if pos_norm > float(max_pos) and pos_norm > 1e-8:
        residual[:3] *= float(max_pos / pos_norm)
    yaw_abs = float(abs(residual[5]))
    if yaw_abs > float(max_yaw) and yaw_abs > 1e-8:
        residual[5] *= float(max_yaw / yaw_abs)
    return residual.astype(np.float32)


def _unique_residual_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    seen: set[tuple[float, ...]] = set()
    uniq: list[np.ndarray] = []
    for row in rows:
        key = tuple(np.round(row, 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row.astype(np.float32))
    return np.asarray(uniq, dtype=np.float32) if uniq else np.zeros((0, 6), dtype=np.float32)


def _best_index_from_score_rows(score_rows: np.ndarray) -> int:
    score_rows = np.asarray(score_rows, dtype=np.float32)
    if score_rows.ndim != 2 or score_rows.shape[0] == 0:
        return 0
    return int(np.lexsort(tuple(score_rows[:, ::-1].T))[0])


def _ordered_indices_from_score_rows(score_rows: np.ndarray) -> np.ndarray:
    score_rows = np.asarray(score_rows, dtype=np.float32)
    if score_rows.ndim != 2 or score_rows.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(np.lexsort(tuple(score_rows[:, ::-1].T)), dtype=np.int64)


def _select_teacher(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    coarse_residuals: np.ndarray,
    refine_offsets: np.ndarray,
    stage_bucket: str,
    *,
    coarse_topk: int,
    max_pos: float,
    max_yaw: float,
    short_horizon_topk: int,
    very_close_abstain_xy_threshold: float,
    very_close_abstain_z_threshold: float,
    very_close_abstain_yaw_threshold: float,
) -> tuple[int, dict[str, np.ndarray], dict[str, float]]:
    coarse_metrics = _evaluate_residuals(current_pose, target_pose, coarse_residuals)
    cur = pose_delta_local_between(current_pose, target_pose).astype(np.float32)
    cur_xy = float(np.linalg.norm(cur[:2]))
    cur_z = float(abs(cur[2]))
    cur_yaw = float(abs(cur[5]))

    coarse_scores = np.array([_score_tuple(coarse_metrics, i, stage_bucket) for i in range(coarse_residuals.shape[0])], dtype=np.float32)
    coarse_order = _ordered_indices_from_score_rows(coarse_scores)
    coarse_candidates = coarse_order[: max(int(coarse_topk), 1)] if coarse_order.size else np.asarray([0], dtype=np.int64)

    candidate_rows = [coarse_residuals]
    for c1 in coarse_candidates.tolist():
        base = coarse_residuals[int(c1)]
        refined = np.asarray(
            [_clip_residual(base + off, max_pos=max_pos, max_yaw=max_yaw) for off in refine_offsets],
            dtype=np.float32,
        )
        candidate_rows.append(refined)
    residuals = _unique_residual_rows(np.concatenate(candidate_rows, axis=0))
    metrics = _evaluate_residuals(current_pose, target_pose, residuals)

    scores = np.array([_score_tuple(metrics, i, stage_bucket) for i in range(residuals.shape[0])], dtype=np.float32)
    one_step_order = _ordered_indices_from_score_rows(scores)
    best_one_step_idx = int(one_step_order[0]) if one_step_order.size else 0
    noop_idx = 0
    noop_reason = "noop_due_to_best_candidate_is_noop"
    two_step_improve_count_lt2 = False
    two_step_horizon_ge_noop = False
    two_step_horizon_score_margin = 0.0
    best_improve_count = 0
    best_improve_missing_xy = False
    best_improve_missing_z = False
    best_improve_missing_yaw = False
    very_close_abstain_xy = False
    very_close_abstain_z = False
    very_close_abstain_yaw = False
    very_close_abstain_margin_xy = 0.0
    very_close_abstain_margin_z = 0.0
    very_close_abstain_margin_yaw = 0.0

    horizon_candidates = one_step_order[: max(int(short_horizon_topk), 1)] if one_step_order.size else np.asarray([0], dtype=np.int64)
    best_horizon_idx = best_one_step_idx
    best_horizon_score = float("inf")
    for c1 in horizon_candidates.tolist():
        pose1 = apply_local_offset_to_pose(current_pose, residuals[int(c1)])
        second_metrics = _evaluate_residuals(pose1, target_pose, residuals)
        second_scores = np.array([_score_tuple(second_metrics, j, stage_bucket) for j in range(residuals.shape[0])], dtype=np.float32)
        c2 = _best_index_from_score_rows(second_scores)
        two_score = float(_score_tuple(second_metrics, c2, stage_bucket)[0])
        if two_score < best_horizon_score:
            best_horizon_score = two_score
            best_horizon_idx = int(c1)
    best_idx = int(best_horizon_idx)
    noop_two_step_score = float(_score_tuple(metrics, best_one_step_idx, stage_bucket)[0])
    two_step_horizon_score_margin = float(noop_two_step_score - best_horizon_score)

    best_improve_xy = bool(metrics["improves_xy"][best_idx] > 0.5)
    best_improve_z = bool(metrics["improves_z"][best_idx] > 0.5)
    best_improve_yaw = bool(metrics["improves_yaw"][best_idx] > 0.5)
    best_improve_count = int(best_improve_xy + best_improve_z + best_improve_yaw)
    best_improve_missing_xy = not best_improve_xy
    best_improve_missing_z = not best_improve_z
    best_improve_missing_yaw = not best_improve_yaw
    noop_score = _score_tuple(metrics, noop_idx, stage_bucket)
    best_score = _score_tuple(metrics, best_idx, stage_bucket)
    two_step_improve_count_lt2 = bool(best_improve_count < 2)
    two_step_horizon_ge_noop = bool(best_horizon_score >= noop_two_step_score)
    if best_idx != 0 and (two_step_improve_count_lt2 and two_step_horizon_ge_noop):
        best_idx = 0
        noop_reason = "noop_due_to_two_step_fallback"
    very_close_abstain_xy = bool(cur_xy <= float(very_close_abstain_xy_threshold))
    very_close_abstain_z = bool(cur_z <= float(very_close_abstain_z_threshold))
    very_close_abstain_yaw = bool(cur_yaw <= float(very_close_abstain_yaw_threshold))
    very_close_abstain_margin_xy = float(cur_xy - float(very_close_abstain_xy_threshold))
    very_close_abstain_margin_z = float(cur_z - float(very_close_abstain_z_threshold))
    very_close_abstain_margin_yaw = float(cur_yaw - float(very_close_abstain_yaw_threshold))
    very_close_abstain = very_close_abstain_xy and very_close_abstain_z and very_close_abstain_yaw
    if best_idx != 0 and very_close_abstain:
        best_idx = 0
        noop_reason = "noop_due_to_very_close_abstain"
    if best_idx != 0:
        noop_reason = "selected_nonnoop"

    selected = {
        "teacher_residual_local_4d": metrics["residual_4d"][best_idx].astype(np.float32),
        "teacher_residual_local_6d": metrics["residual_6d"][best_idx].astype(np.float32),
        "teacher_post_xy_error": np.asarray(metrics["post_xy"][best_idx], dtype=np.float32),
        "teacher_post_z_error": np.asarray(metrics["post_z"][best_idx], dtype=np.float32),
        "teacher_post_yaw_error": np.asarray(metrics["post_yaw"][best_idx], dtype=np.float32),
        "teacher_improves_xy": np.asarray(metrics["improves_xy"][best_idx], dtype=np.float32),
        "teacher_improves_z": np.asarray(metrics["improves_z"][best_idx], dtype=np.float32),
        "teacher_improves_yaw": np.asarray(metrics["improves_yaw"][best_idx], dtype=np.float32),
        "teacher_all_improves": np.asarray(metrics["all_improves"][best_idx], dtype=np.float32),
        "teacher_overshoot_xy": np.asarray(float(metrics["overshoot_any"][best_idx] > 0.0), dtype=np.float32),
        "teacher_overshoot_z": np.asarray(float(metrics["overshoot_any"][best_idx] > 0.0), dtype=np.float32),
        "teacher_overshoot_yaw": np.asarray(float(metrics["overshoot_any"][best_idx] > 0.0), dtype=np.float32),
        "teacher_overshoot_any": np.asarray(metrics["overshoot_any"][best_idx], dtype=np.float32),
        "teacher_noop_selected": np.asarray(float(best_idx == 0), dtype=np.float32),
        "teacher_action_pos_norm": np.asarray(metrics["pos_norm"][best_idx], dtype=np.float32),
        "teacher_action_yaw_abs": np.asarray(metrics["yaw_abs"][best_idx], dtype=np.float32),
        "teacher_best_candidate_index": np.asarray(best_idx, dtype=np.int64),
        "teacher_one_step_best_candidate_index": np.asarray(int(best_one_step_idx), dtype=np.int64),
        "teacher_short_horizon_best_candidate_index": np.asarray(best_idx, dtype=np.int64),
        "teacher_noop_reason": np.asarray(noop_reason, dtype=object),
        "teacher_two_step_improve_count_lt2": np.asarray(float(two_step_improve_count_lt2), dtype=np.float32),
        "teacher_two_step_horizon_ge_noop": np.asarray(float(two_step_horizon_ge_noop), dtype=np.float32),
        "teacher_two_step_fallback_triggered": np.asarray(float(noop_reason == "noop_due_to_two_step_fallback"), dtype=np.float32),
        "teacher_two_step_horizon_score_margin": np.asarray(two_step_horizon_score_margin, dtype=np.float32),
        "teacher_best_improve_count": np.asarray(best_improve_count, dtype=np.int64),
        "teacher_best_improve_missing_xy": np.asarray(float(best_improve_missing_xy), dtype=np.float32),
        "teacher_best_improve_missing_z": np.asarray(float(best_improve_missing_z), dtype=np.float32),
        "teacher_best_improve_missing_yaw": np.asarray(float(best_improve_missing_yaw), dtype=np.float32),
        "teacher_very_close_abstain_xy": np.asarray(float(very_close_abstain_xy), dtype=np.float32),
        "teacher_very_close_abstain_z": np.asarray(float(very_close_abstain_z), dtype=np.float32),
        "teacher_very_close_abstain_yaw": np.asarray(float(very_close_abstain_yaw), dtype=np.float32),
        "teacher_very_close_abstain_margin_xy": np.asarray(very_close_abstain_margin_xy, dtype=np.float32),
        "teacher_very_close_abstain_margin_z": np.asarray(very_close_abstain_margin_z, dtype=np.float32),
        "teacher_very_close_abstain_margin_yaw": np.asarray(very_close_abstain_margin_yaw, dtype=np.float32),
        "teacher_very_close_abstain_triggered": np.asarray(float(noop_reason == "noop_due_to_very_close_abstain"), dtype=np.float32),
        "teacher_very_close_abstain_xy_threshold": np.asarray(float(very_close_abstain_xy_threshold), dtype=np.float32),
        "teacher_very_close_abstain_z_threshold": np.asarray(float(very_close_abstain_z_threshold), dtype=np.float32),
        "teacher_very_close_abstain_yaw_threshold": np.asarray(float(very_close_abstain_yaw_threshold), dtype=np.float32),
        "teacher_objective_stage": np.asarray(0 if stage_bucket == "near_alignment" else 1, dtype=np.int64),
        "teacher_objective_primary": np.asarray(
            metrics["post_z"][best_idx]
            if stage_bucket == "near_alignment"
            else (metrics["post_xy"][best_idx] + metrics["post_z"][best_idx] + 1.5 * metrics["post_yaw"][best_idx]),
            dtype=np.float32,
        ),
        "teacher_objective_secondary": np.asarray(
            metrics["post_xy"][best_idx] if stage_bucket == "near_alignment" else metrics["overshoot_any"][best_idx],
            dtype=np.float32,
        ),
        "teacher_objective_tertiary": np.asarray(
            metrics["post_yaw"][best_idx] if stage_bucket == "near_alignment" else metrics["pos_norm"][best_idx],
            dtype=np.float32,
        ),
        "teacher_objective_quaternary": np.asarray(
            metrics["pos_norm"][best_idx] if stage_bucket == "near_alignment" else metrics["yaw_abs"][best_idx],
            dtype=np.float32,
        ),
        "teacher_objective_quinary": np.asarray(
            metrics["overshoot_any"][best_idx] if stage_bucket == "near_alignment" else float(best_idx != 0),
            dtype=np.float32,
        ),
    }

    audit = {
        "current_xy": cur_xy,
        "current_z": cur_z,
        "current_yaw": cur_yaw,
        "best_idx_before_noop_fallback": int(best_idx),
        "best_score": [float(x) for x in best_score],
        "noop_score": [float(x) for x in noop_score],
        "best_improve_count": best_improve_count,
        "best_improve_missing_xy": bool(best_improve_missing_xy),
        "best_improve_missing_z": bool(best_improve_missing_z),
        "best_improve_missing_yaw": bool(best_improve_missing_yaw),
        "very_close_abstain_xy": bool(very_close_abstain_xy),
        "very_close_abstain_z": bool(very_close_abstain_z),
        "very_close_abstain_yaw": bool(very_close_abstain_yaw),
        "very_close_abstain_margin_xy": float(very_close_abstain_margin_xy),
        "very_close_abstain_margin_z": float(very_close_abstain_margin_z),
        "very_close_abstain_margin_yaw": float(very_close_abstain_margin_yaw),
        "very_close_abstain_triggered": bool(noop_reason == "noop_due_to_very_close_abstain"),
        "very_close_abstain_xy_threshold": float(very_close_abstain_xy_threshold),
        "very_close_abstain_z_threshold": float(very_close_abstain_z_threshold),
        "very_close_abstain_yaw_threshold": float(very_close_abstain_yaw_threshold),
        "two_step_improve_count_lt2": bool(two_step_improve_count_lt2),
        "two_step_horizon_ge_noop": bool(two_step_horizon_ge_noop),
        "two_step_horizon_score_margin": float(two_step_horizon_score_margin),
        "two_step_fallback_triggered": bool(noop_reason == "noop_due_to_two_step_fallback"),
        "teacher_selected_noop": bool(best_idx == 0),
        "teacher_noop_reason": str(noop_reason),
        "coarse_candidate_count": int(coarse_candidates.size),
        "final_candidate_count": int(residuals.shape[0]),
        "noop_two_step_score": float(noop_two_step_score),
        "best_two_step_score": float(best_horizon_score),
    }
    return best_idx, selected, audit


def _fallback_has_object_in_hand(row: dict[str, np.ndarray]) -> float:
    if "has_object_in_hand" in row:
        try:
            arr = np.asarray(row["has_object_in_hand"], dtype=np.float32).reshape(-1)
            if arr.size:
                return float(arr[0])
        except Exception:
            pass
    return max(
        _optional_binary_row(row, "teacher_attached_after_close", "teacher_grasp_verified", default=0.0),
        _optional_binary_row(row, "verified_lift", default=0.0),
    )


def _build_has_object_in_hand_array(src_data: dict[str, np.ndarray]) -> np.ndarray:
    if "has_object_in_hand" in src_data:
        return np.asarray(src_data["has_object_in_hand"], dtype=np.float32).reshape(-1)
    n = int(np.asarray(src_data["current_pose_7d"]).shape[0])
    fallback_keys = ("teacher_attached_after_close", "teacher_grasp_verified", "verified_lift")
    arrays: list[np.ndarray] = []
    for key in fallback_keys:
        if key in src_data:
            arrays.append(np.asarray(src_data[key], dtype=np.float32).reshape(-1))
    if arrays:
        out = np.zeros((n,), dtype=np.float32)
        for arr in arrays:
            if arr.shape[0] == n:
                out = np.maximum(out, arr.astype(np.float32))
        return out
    return np.zeros((n,), dtype=np.float32)


def _short_horizon_audit(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    residuals: np.ndarray,
    stage_buckets: np.ndarray,
    sample_size: int,
) -> dict[str, float]:
    current_pose = np.asarray(current_pose, dtype=np.float32).reshape(-1, 7)
    target_pose = np.asarray(target_pose, dtype=np.float32).reshape(-1, 7)
    stage_buckets = np.asarray(stage_buckets, dtype=str).reshape(-1)
    n = min(int(sample_size), current_pose.shape[0], target_pose.shape[0])
    if n <= 0:
        return {
            "sample_size": 0,
            "conflict_rate": 0.0,
            "mean_one_step_score": 0.0,
            "mean_two_step_score": 0.0,
        }
    idxs = np.linspace(0, current_pose.shape[0] - 1, num=n, dtype=np.int64)
    conflict = 0
    one_scores: list[float] = []
    two_scores: list[float] = []
    conflict_by_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for idx in idxs.tolist():
        cur_pose = np.asarray(current_pose[idx], dtype=np.float32).reshape(7)
        tgt_pose = np.asarray(target_pose[idx], dtype=np.float32).reshape(7)
        stage_bucket = str(stage_buckets[idx]) if idx < stage_buckets.shape[0] else "near_alignment"
        first_metrics = _evaluate_residuals(cur_pose, tgt_pose, residuals)
        first_scores = np.array([_score_tuple(first_metrics, i, stage_bucket) for i in range(residuals.shape[0])], dtype=np.float32)
        first_idx = _best_index_from_score_rows(first_scores)
        one_scores.append(float(_score_tuple(first_metrics, first_idx, stage_bucket)[0]))

        best_two_step_score = np.inf
        best_two_step_idx = 0
        for c1 in range(residuals.shape[0]):
            pose1 = apply_local_offset_to_pose(cur_pose, residuals[c1])
            second_metrics = _evaluate_residuals(pose1, tgt_pose, residuals)
            second_scores = np.array([_score_tuple(second_metrics, j, stage_bucket) for j in range(residuals.shape[0])], dtype=np.float32)
            c2 = _best_index_from_score_rows(second_scores)
            two_score = float(_score_tuple(second_metrics, c2, stage_bucket)[0])
            if two_score < best_two_step_score:
                best_two_step_score = two_score
                best_two_step_idx = c1
        two_scores.append(float(best_two_step_score))
        if best_two_step_idx != first_idx:
            conflict += 1
            conflict_by_bucket[stage_bucket][0] += 1
        conflict_by_bucket[stage_bucket][1] += 1
    return {
        "sample_size": float(n),
        "conflict_rate": float(conflict / max(n, 1)),
        "mean_one_step_score": float(np.mean(one_scores)) if one_scores else 0.0,
        "mean_two_step_score": float(np.mean(two_scores)) if two_scores else 0.0,
        "mean_two_minus_one_score": float((np.mean(two_scores) - np.mean(one_scores))) if one_scores else 0.0,
        "conflict_rate_by_bucket": {
            bucket: float(vals[0] / max(vals[1], 1))
            for bucket, vals in sorted(conflict_by_bucket.items(), key=lambda kv: kv[0])
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_npz", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, required=True)
    ap.add_argument("--dx_values", type=str, default="-0.010,-0.005,0.0,0.005,0.010")
    ap.add_argument("--dy_values", type=str, default="-0.010,-0.005,0.0,0.005,0.010")
    ap.add_argument("--dz_values", type=str, default="-0.016,-0.008,0.0,0.008,0.016")
    ap.add_argument("--dyaw_values", type=str, default="-0.030,-0.015,0.0,0.015,0.030")
    ap.add_argument("--refine_dx_values", type=str, default="-0.004,0.0,0.004")
    ap.add_argument("--refine_dy_values", type=str, default="-0.004,0.0,0.004")
    ap.add_argument("--refine_dz_values", type=str, default="-0.006,0.0,0.006")
    ap.add_argument("--refine_dyaw_values", type=str, default="-0.010,0.0,0.010")
    ap.add_argument("--micro_refine_dx_values", type=str, default="-0.006,-0.003,0.0,0.003,0.006")
    ap.add_argument("--micro_refine_dy_values", type=str, default="-0.006,-0.003,0.0,0.003,0.006")
    ap.add_argument("--micro_refine_dz_values", type=str, default="-0.004,0.0,0.004")
    ap.add_argument("--micro_refine_dyaw_values", type=str, default="-0.015,-0.0075,0.0,0.0075,0.015")
    ap.add_argument("--coarse_topk", type=int, default=4)
    ap.add_argument("--keep_source_xy_norm", type=float, default=0.050)
    ap.add_argument("--keep_source_z_abs", type=float, default=0.100)
    ap.add_argument("--keep_source_yaw_abs", type=float, default=0.350)
    ap.add_argument("--micro_source_xy_norm", type=float, default=0.020)
    ap.add_argument("--micro_source_z_abs", type=float, default=0.040)
    ap.add_argument("--micro_source_yaw_abs", type=float, default=0.150)
    ap.add_argument("--very_close_abstain_xy", type=float, default=0.010)
    ap.add_argument("--very_close_abstain_z", type=float, default=0.020)
    ap.add_argument("--very_close_abstain_yaw", type=float, default=0.150)
    ap.add_argument("--servo_k_xy", type=float, default=0.08)
    ap.add_argument("--servo_k_z", type=float, default=0.06)
    ap.add_argument("--servo_k_yaw", type=float, default=0.04)
    ap.add_argument("--servo_max_pos", type=float, default=0.0010)
    ap.add_argument("--servo_max_yaw", type=float, default=0.0080)
    ap.add_argument("--short_horizon_topk", type=int, default=4)
    ap.add_argument("--short_horizon_audit_rows", type=int, default=4)
    ap.add_argument("--min_rows", type=int, default=64)
    ap.add_argument("--min_micro_rows", type=int, default=16)
    ap.add_argument("--enforce_audit_pass", action="store_true", default=True)
    ap.add_argument("--no_enforce_audit_pass", dest="enforce_audit_pass", action="store_false")
    args = ap.parse_args()

    src = np.load(args.source_npz, allow_pickle=True)
    src_data = {k: np.asarray(src[k]) for k in src.files}
    required = [
        "current_pose_7d",
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "gripper_context",
    ]
    for key in required:
        if key not in src_data:
            raise SystemExit(f"source npz missing required field: {key}")

    current_pose_all = np.asarray(src_data["current_pose_7d"], dtype=np.float32)
    has_object_in_hand_all = _build_has_object_in_hand_array(src_data)
    target_pose_all = []
    target_source_all = []
    target_source_reason_all = []
    keep_mask = np.zeros((current_pose_all.shape[0],), dtype=bool)
    stage_bucket_all = np.asarray([""] * current_pose_all.shape[0], dtype=object)
    dropped_missing_target = 0
    for i in range(current_pose_all.shape[0]):
        row = {k: src_data[k][i] for k in src_data}
        target_pose, target_source, target_reason = _choose_target_pose(row, current_pose=current_pose_all[i])
        if target_pose is None:
            target_pose_all.append(np.full((7,), np.nan, dtype=np.float32))
            target_source_all.append(target_source)
            target_source_reason_all.append(target_reason)
            dropped_missing_target += 1
            continue
        cur_delta = pose_delta_local_between(current_pose_all[i], target_pose).astype(np.float32)
        cur_xy = float(np.linalg.norm(cur_delta[:2]))
        cur_z = float(abs(cur_delta[2]))
        cur_yaw = float(abs(cur_delta[5]))
        if (
            cur_xy <= float(args.keep_source_xy_norm)
            and cur_z <= float(args.keep_source_z_abs)
            and cur_yaw <= float(args.keep_source_yaw_abs)
        ):
            keep_mask[i] = True
            stage_bucket_all[i] = (
                "micro_contact_refine"
                if (
                    cur_xy <= float(args.micro_source_xy_norm)
                    and cur_z <= float(args.micro_source_z_abs)
                    and cur_yaw <= float(args.micro_source_yaw_abs)
                )
                else "near_alignment"
        )
        target_pose_all.append(target_pose)
        target_source_all.append(target_source)
        target_source_reason_all.append(target_reason)

    selected = np.where(keep_mask)[0]
    if selected.size == 0:
        raise SystemExit("No rows passed privileged geometry filtering.")

    selected_micro = stage_bucket_all[selected] == "micro_contact_refine"
    selected_stage_bucket = np.where(selected_micro, "micro_contact_refine", "near_alignment")

    coarse_bank = _generate_teacher_grid(
        _parse_floats(args.dx_values),
        _parse_floats(args.dy_values),
        _parse_floats(args.dz_values),
        _parse_floats(args.dyaw_values),
    )
    refine_offsets = _generate_teacher_grid(
        _parse_floats(args.refine_dx_values),
        _parse_floats(args.refine_dy_values),
        _parse_floats(args.refine_dz_values),
        _parse_floats(args.refine_dyaw_values),
    )
    micro_refine_offsets = _generate_teacher_grid(
        _parse_floats(args.micro_refine_dx_values),
        _parse_floats(args.micro_refine_dy_values),
        _parse_floats(args.micro_refine_dz_values),
        _parse_floats(args.micro_refine_dyaw_values),
    )

    current_pose = current_pose_all[selected]
    target_pose = np.stack([target_pose_all[i] for i in selected.tolist()], axis=0).astype(np.float32)
    current_delta = np.stack([pose_delta_local_between(c, t) for c, t in zip(current_pose, target_pose)], axis=0).astype(np.float32)
    teacher_rows: list[dict[str, np.ndarray | float | int | str]] = []
    teacher_indices = []
    audit_bits = []
    for i, src_i in enumerate(selected.tolist()):
        best_idx, selected_row, audit = _select_teacher(
            current_pose[i],
            target_pose[i],
            coarse_bank,
            micro_refine_offsets if str(selected_stage_bucket[i]) == "micro_contact_refine" else refine_offsets,
            str(selected_stage_bucket[i]),
            coarse_topk=int(args.coarse_topk),
            max_pos=float(args.servo_max_pos) * 20.0,
            max_yaw=float(args.servo_max_yaw) * 4.0,
            short_horizon_topk=int(args.short_horizon_topk),
            very_close_abstain_xy_threshold=float(args.very_close_abstain_xy),
            very_close_abstain_z_threshold=float(args.very_close_abstain_z),
            very_close_abstain_yaw_threshold=float(args.very_close_abstain_yaw),
        )
        target_source = str(target_source_all[src_i])
        target_source_reason = str(target_source_reason_all[src_i])
        target_rank = _target_source_rank(target_source_reason)
        noop_reason = str(selected_row.get("teacher_noop_reason", "selected_nonnoop"))
        if bool(selected_row.get("teacher_selected_noop", False)) and (
            target_source == "missing" or (target_rank is not None and target_rank > 0)
        ):
            noop_reason = "noop_due_to_target_missing_or_ranked_away"
        selected_row["teacher_noop_reason"] = np.asarray(noop_reason, dtype=object)
        selected_row["teacher_noop_reason_detail"] = np.asarray(
            f"selection={noop_reason};target_source={target_source};target_source_reason={target_source_reason}",
            dtype=object,
        )
        teacher_indices.append(int(src_i))
        audit_bits.append(audit)
        teacher_rows.append(
            {
                **selected_row,
                "teacher_source_row_index": np.asarray(src_i, dtype=np.int64),
                "row_index": np.asarray(src_i, dtype=np.int64),
                "episode_index": np.asarray(int(src_data["episode_index"][src_i]) if "episode_index" in src_data else -1, dtype=np.int64),
                "step_index": np.asarray(int(src_data["step_index"][src_i]) if "step_index" in src_data else -1, dtype=np.int64),
                "stage_bucket": np.asarray(selected_stage_bucket[i]),
                "teacher_target_source": np.asarray(target_source, dtype=object),
                "teacher_target_source_reason": np.asarray(target_source_reason, dtype=object),
                "current_pose_7d": np.asarray(current_pose[i], dtype=np.float32),
                "teacher_target_pose_7d": np.asarray(target_pose[i], dtype=np.float32),
                "motion_target_pose_7d": np.asarray(target_pose[i], dtype=np.float32),
                "teacher_current_to_target_delta_local": np.asarray(current_delta[i], dtype=np.float32),
                "current_to_target_delta_local": np.asarray(current_delta[i], dtype=np.float32),
                "teacher_runtime_delta_contract": np.asarray(target_source_all[src_i], dtype=object),
            }
        )

    def _stack(key: str, dtype=None):
        arr = np.stack([np.asarray(r[key]) for r in teacher_rows], axis=0)
        return arr.astype(dtype) if dtype is not None else arr

    out: dict[str, np.ndarray] = {}
    pass_keys = (
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "gripper_context",
        "substage_id",
        "contact_state",
        "stage_target_mode",
        "depth_proximity",
        "planner_base_action_local",
        "planner_base_action_local_raw",
        "planner_base_action_7d_raw",
        "base_action",
        "force_history",
        "force_history_raw",
        "force_history_normalized",
        "ft_hist",
        "gripper_touch_forces",
        "force_norm",
        "torque_norm",
        "wrist_depth_median",
        "wrist_valid_depth_ratio",
        "wrist_depth_near_fraction",
        "is_occluded",
        "is_low_visibility",
        "phase_id",
        "phase_age",
        "steps_since_last_replan",
        "current_xy_error",
        "current_z_error",
        "current_yaw_error",
        "privileged_current_pose_7d",
        "privileged_motion_target_pose_7d",
        "privileged_basin_center_pose_7d",
        "privileged_pregrasp_target_pose_7d",
        "privileged_grasp_commit_target_pose_7d",
        "privileged_object_anchor_pose_7d",
        "privileged_current_delta_basin_target",
        "privileged_target_provider_source",
        "privileged_target_provider_uses_privileged",
    )
    for key in pass_keys:
        if key in src_data:
            out[key] = np.asarray(src_data[key])[selected]

    if "has_object_in_hand" not in out:
        out["has_object_in_hand"] = has_object_in_hand_all[selected].astype(np.float32)

    if "planner_base_action_local" in out:
        out["planner_action_local"] = np.asarray(out["planner_base_action_local"], dtype=np.float32)
    elif "planner_base_action_local_raw" in out:
        out["planner_action_local"] = np.asarray(out["planner_base_action_local_raw"], dtype=np.float32)
    elif "base_action" in out:
        out["planner_action_local"] = np.asarray(out["base_action"], dtype=np.float32)
    else:
        out["planner_action_local"] = np.zeros((selected.size, 6), dtype=np.float32)

    if "force_history" not in out:
        out["force_history"] = np.zeros((selected.size, 32, 6), dtype=np.float32)

    out["teacher_source_row_index"] = np.asarray(teacher_indices, dtype=np.int64)
    out["row_index"] = out["teacher_source_row_index"].astype(np.int64)
    out["stage_bucket"] = np.asarray([r["stage_bucket"] for r in teacher_rows])
    out["teacher_target_source"] = np.asarray([r["teacher_target_source"] for r in teacher_rows])
    out["teacher_target_source_reason"] = np.asarray([r["teacher_target_source_reason"] for r in teacher_rows])
    out["teacher_runtime_delta_contract"] = np.asarray([r["teacher_runtime_delta_contract"] for r in teacher_rows])
    out["teacher_target_pose_7d"] = _stack("teacher_target_pose_7d", np.float32)
    out["motion_target_pose_7d"] = _stack("motion_target_pose_7d", np.float32)
    out["current_pose_7d"] = _stack("current_pose_7d", np.float32)
    out["teacher_current_to_target_delta_local"] = _stack("teacher_current_to_target_delta_local", np.float32)
    out["current_to_target_delta_local"] = out["teacher_current_to_target_delta_local"].astype(np.float32)

    out["teacher_residual_local_4d"] = _stack("teacher_residual_local_4d", np.float32)
    out["teacher_residual_local_6d"] = _stack("teacher_residual_local_6d", np.float32)
    out["teacher_post_xy_error"] = _stack("teacher_post_xy_error", np.float32)
    out["teacher_post_z_error"] = _stack("teacher_post_z_error", np.float32)
    out["teacher_post_yaw_error"] = _stack("teacher_post_yaw_error", np.float32)
    out["teacher_improves_xy"] = _stack("teacher_improves_xy", np.float32)
    out["teacher_improves_z"] = _stack("teacher_improves_z", np.float32)
    out["teacher_improves_yaw"] = _stack("teacher_improves_yaw", np.float32)
    out["teacher_all_improves"] = _stack("teacher_all_improves", np.float32)
    out["teacher_overshoot_xy"] = _stack("teacher_overshoot_xy", np.float32)
    out["teacher_overshoot_z"] = _stack("teacher_overshoot_z", np.float32)
    out["teacher_overshoot_yaw"] = _stack("teacher_overshoot_yaw", np.float32)
    out["teacher_overshoot_any"] = _stack("teacher_overshoot_any", np.float32)
    out["teacher_noop_selected"] = _stack("teacher_noop_selected", np.float32)
    out["teacher_action_pos_norm"] = _stack("teacher_action_pos_norm", np.float32)
    out["teacher_action_yaw_abs"] = _stack("teacher_action_yaw_abs", np.float32)
    out["teacher_best_candidate_index"] = _stack("teacher_best_candidate_index", np.int64)
    out["teacher_one_step_best_candidate_index"] = _stack("teacher_one_step_best_candidate_index", np.int64)
    out["teacher_short_horizon_best_candidate_index"] = _stack("teacher_short_horizon_best_candidate_index", np.int64)
    out["teacher_noop_reason"] = _stack("teacher_noop_reason")
    out["teacher_noop_reason_detail"] = _stack("teacher_noop_reason_detail")
    out["teacher_two_step_improve_count_lt2"] = _stack("teacher_two_step_improve_count_lt2", np.float32)
    out["teacher_two_step_horizon_ge_noop"] = _stack("teacher_two_step_horizon_ge_noop", np.float32)
    out["teacher_two_step_fallback_triggered"] = _stack("teacher_two_step_fallback_triggered", np.float32)
    out["teacher_two_step_horizon_score_margin"] = _stack("teacher_two_step_horizon_score_margin", np.float32)
    out["teacher_best_improve_count"] = _stack("teacher_best_improve_count", np.int64)
    out["teacher_best_improve_missing_xy"] = _stack("teacher_best_improve_missing_xy", np.float32)
    out["teacher_best_improve_missing_z"] = _stack("teacher_best_improve_missing_z", np.float32)
    out["teacher_best_improve_missing_yaw"] = _stack("teacher_best_improve_missing_yaw", np.float32)
    out["teacher_very_close_abstain_xy"] = _stack("teacher_very_close_abstain_xy", np.float32)
    out["teacher_very_close_abstain_z"] = _stack("teacher_very_close_abstain_z", np.float32)
    out["teacher_very_close_abstain_yaw"] = _stack("teacher_very_close_abstain_yaw", np.float32)
    out["teacher_very_close_abstain_margin_xy"] = _stack("teacher_very_close_abstain_margin_xy", np.float32)
    out["teacher_very_close_abstain_margin_z"] = _stack("teacher_very_close_abstain_margin_z", np.float32)
    out["teacher_very_close_abstain_margin_yaw"] = _stack("teacher_very_close_abstain_margin_yaw", np.float32)
    out["teacher_very_close_abstain_triggered"] = _stack("teacher_very_close_abstain_triggered", np.float32)
    out["teacher_very_close_abstain_xy_threshold"] = _stack("teacher_very_close_abstain_xy_threshold", np.float32)
    out["teacher_very_close_abstain_z_threshold"] = _stack("teacher_very_close_abstain_z_threshold", np.float32)
    out["teacher_very_close_abstain_yaw_threshold"] = _stack("teacher_very_close_abstain_yaw_threshold", np.float32)
    out["teacher_objective_stage"] = _stack("teacher_objective_stage", np.int64)
    out["teacher_objective_primary"] = _stack("teacher_objective_primary", np.float32)
    out["teacher_objective_secondary"] = _stack("teacher_objective_secondary", np.float32)
    out["teacher_objective_tertiary"] = _stack("teacher_objective_tertiary", np.float32)
    out["teacher_objective_quaternary"] = _stack("teacher_objective_quaternary", np.float32)
    out["teacher_objective_quinary"] = _stack("teacher_objective_quinary", np.float32)
    out["target_residual_local_4d"] = out["teacher_residual_local_4d"].astype(np.float32)
    out["target_residual_local_6d"] = out["teacher_residual_local_6d"].astype(np.float32)
    out["target_post_xy_error"] = out["teacher_post_xy_error"].astype(np.float32)
    out["target_post_z_error"] = out["teacher_post_z_error"].astype(np.float32)
    out["target_post_yaw_error"] = out["teacher_post_yaw_error"].astype(np.float32)
    out["target_improves_xy"] = out["teacher_improves_xy"].astype(np.float32)
    out["target_improves_z"] = out["teacher_improves_z"].astype(np.float32)
    out["target_improves_yaw"] = out["teacher_improves_yaw"].astype(np.float32)
    out["overshoot_proxy"] = out["teacher_overshoot_any"].astype(np.float32)
    out["teacher_source"] = np.asarray(["from_scratch_privileged_direct_teacher_v1"] * selected.size)
    out["runtime_target_delta_source"] = np.asarray(["raw_learned_predictor_from_scratch"] * selected.size)
    out["runtime_target_delta_context_mode"] = np.asarray(["raw_rollout_context"] * selected.size)

    # Preserve privileged labels for audit.
    out["privileged_target_pose_7d"] = out["motion_target_pose_7d"].astype(np.float32)

    pos_norm = np.linalg.norm(out["teacher_residual_local_6d"][:, :3], axis=-1)
    yaw_abs = np.abs(out["teacher_residual_local_6d"][:, 5])
    pos_hi = float(np.percentile(pos_norm, 90))
    yaw_hi = float(np.percentile(yaw_abs, 90))
    out["teacher_workspace_violation"] = ((pos_norm > pos_hi) | (yaw_abs > yaw_hi)).astype(np.float32)
    out["teacher_invalid"] = ((out["teacher_overshoot_any"] > 0.5) | (out["teacher_workspace_violation"] > 0.5)).astype(np.float32)
    out["invalid_risk_proxy"] = out["teacher_invalid"].astype(np.float32)

    # Short-horizon audit on a subset.
    short_horizon = _short_horizon_audit(
        current_pose=current_pose,
        target_pose=target_pose,
        residuals=coarse_bank,
        stage_buckets=selected_stage_bucket,
        sample_size=int(args.short_horizon_audit_rows),
    )

    teacher_vs_noop_xy = float((out["teacher_post_xy_error"] < np.linalg.norm(current_delta[:, :2], axis=-1)).mean())
    teacher_vs_noop_z = float((out["teacher_post_z_error"] < np.abs(current_delta[:, 2])).mean())
    teacher_vs_noop_yaw = float((out["teacher_post_yaw_error"] < np.abs(current_delta[:, 5])).mean())
    teacher_vs_servo_res = np.stack(
        [
            _bounded_servo_pseudo(
                current_delta[i],
                k_xy=float(args.servo_k_xy),
                k_z=float(args.servo_k_z),
                k_yaw=float(args.servo_k_yaw),
                max_pos=float(args.servo_max_pos),
                max_yaw=float(args.servo_max_yaw),
            )
            for i in range(current_delta.shape[0])
        ],
        axis=0,
    )
    servo_eval = _evaluate_residuals(current_pose, target_pose, teacher_vs_servo_res)
    noop_eval = _evaluate_residuals(current_pose, target_pose, np.zeros_like(teacher_vs_servo_res))
    teacher_eval = _evaluate_residuals(current_pose, target_pose, out["teacher_residual_local_6d"])

    audit_passed = bool(
        selected.size >= int(args.min_rows)
        and int((stage_bucket_all[selected] == "micro_contact_refine").sum()) >= int(args.min_micro_rows)
        and 0.03 <= float(np.mean(out["teacher_noop_selected"])) <= 0.80
        and float(np.mean(out["teacher_post_xy_error"])) < float(np.mean(np.linalg.norm(current_delta[:, :2], axis=-1)))
        and float(np.mean(out["teacher_post_z_error"])) < float(np.mean(np.abs(current_delta[:, 2])))
        and float(np.mean(out["teacher_post_yaw_error"])) < float(np.mean(np.abs(current_delta[:, 5])))
        and float(np.mean(out["teacher_improves_xy"])) > 0.5
        and float(np.mean(out["teacher_improves_z"])) > 0.5
        and float(np.mean(out["teacher_improves_yaw"])) > 0.5
        and float(np.mean(out["teacher_all_improves"])) > 0.1
        and float(np.mean(out["teacher_overshoot_any"])) < 0.60
        and float(np.mean(out["teacher_workspace_violation"])) < 0.35
        and float(np.mean(out["teacher_invalid"])) < 0.60
        and float(np.mean(out["teacher_post_xy_error"])) <= float(np.mean(servo_eval["post_xy"]))
        and float(np.mean(out["teacher_post_z_error"])) <= float(np.mean(servo_eval["post_z"]))
        and float(np.mean(out["teacher_post_yaw_error"])) <= float(np.mean(servo_eval["post_yaw"]))
    )

    out["teacher_audit_passed"] = np.asarray(float(audit_passed), dtype=np.float32)
    out["teacher_audit_fail_reason"] = np.asarray(
        "passed" if audit_passed else f"audit_failed_thresholds:short_horizon_conflict={float(short_horizon['conflict_rate']):.3f}",
        dtype=object,
    )

    teacher_best_improve_count_arr = np.asarray(out["teacher_best_improve_count"], dtype=np.int64)
    teacher_best_missing_xy_arr = np.asarray(out["teacher_best_improve_missing_xy"], dtype=np.float32)
    teacher_best_missing_z_arr = np.asarray(out["teacher_best_improve_missing_z"], dtype=np.float32)
    teacher_best_missing_yaw_arr = np.asarray(out["teacher_best_improve_missing_yaw"], dtype=np.float32)
    teacher_vca_xy_arr = np.asarray(out["teacher_very_close_abstain_xy"], dtype=np.float32)
    teacher_vca_z_arr = np.asarray(out["teacher_very_close_abstain_z"], dtype=np.float32)
    teacher_vca_yaw_arr = np.asarray(out["teacher_very_close_abstain_yaw"], dtype=np.float32)
    teacher_vca_margin_xy_arr = np.asarray(out["teacher_very_close_abstain_margin_xy"], dtype=np.float32)
    teacher_vca_margin_z_arr = np.asarray(out["teacher_very_close_abstain_margin_z"], dtype=np.float32)
    teacher_vca_margin_yaw_arr = np.asarray(out["teacher_very_close_abstain_margin_yaw"], dtype=np.float32)
    teacher_vca_triggered_arr = np.asarray(out["teacher_very_close_abstain_triggered"], dtype=np.float32)
    teacher_vca_threshold_xy = float(np.asarray(out["teacher_very_close_abstain_xy_threshold"], dtype=np.float32).reshape(-1)[0]) if "teacher_very_close_abstain_xy_threshold" in out else float(args.very_close_abstain_xy)
    teacher_vca_threshold_z = float(np.asarray(out["teacher_very_close_abstain_z_threshold"], dtype=np.float32).reshape(-1)[0]) if "teacher_very_close_abstain_z_threshold" in out else float(args.very_close_abstain_z)
    teacher_vca_threshold_yaw = float(np.asarray(out["teacher_very_close_abstain_yaw_threshold"], dtype=np.float32).reshape(-1)[0]) if "teacher_very_close_abstain_yaw_threshold" in out else float(args.very_close_abstain_yaw)
    teacher_two_step_margin_arr = np.asarray(out["teacher_two_step_horizon_score_margin"], dtype=np.float32)
    lt2_mask = teacher_best_improve_count_arr < 2
    missing_patterns_all = [
        "+".join(
            axis
            for axis, flag in (("xy", bool(xy)), ("z", bool(z)), ("yaw", bool(yaw)))
            if flag
        )
        or "none"
        for xy, z, yaw in zip(teacher_best_missing_xy_arr, teacher_best_missing_z_arr, teacher_best_missing_yaw_arr)
    ]
    missing_patterns_lt2 = [
        "+".join(
            axis
            for axis, flag in (("xy", bool(xy)), ("z", bool(z)), ("yaw", bool(yaw)))
            if flag
        )
        or "none"
        for xy, z, yaw, keep in zip(
            teacher_best_missing_xy_arr,
            teacher_best_missing_z_arr,
            teacher_best_missing_yaw_arr,
            lt2_mask,
        )
        if bool(keep)
    ]

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    report = {
        "audit": "alignment_v3_from_scratch_teacher_build",
        "source_npz": str(args.source_npz),
        "output_npz": str(args.output_npz),
        "rows_total": int(current_pose_all.shape[0]),
        "rows_selected": int(selected.size),
        "rows_missing_target": int(dropped_missing_target),
        "selection": {
            "keep_source_xy_norm": float(args.keep_source_xy_norm),
            "keep_source_z_abs": float(args.keep_source_z_abs),
            "keep_source_yaw_abs": float(args.keep_source_yaw_abs),
            "micro_source_xy_norm": float(args.micro_source_xy_norm),
            "micro_source_z_abs": float(args.micro_source_z_abs),
            "micro_source_yaw_abs": float(args.micro_source_yaw_abs),
            "selected_bucket_histogram": dict(Counter(selected_stage_bucket.tolist())),
            "target_source_histogram": dict(Counter(np.asarray([target_source_all[i] for i in selected.tolist()], dtype=str).tolist())),
            "target_source_reason_histogram": dict(Counter(np.asarray([target_source_reason_all[i] for i in selected.tolist()], dtype=str).tolist())),
            "noop_reason_histogram": dict(Counter(np.asarray(out["teacher_noop_reason"], dtype=str).tolist())),
            "noop_reason_histogram_by_bucket": {
                bucket: dict(Counter(np.asarray(out["teacher_noop_reason"], dtype=str)[np.asarray(out["stage_bucket"], dtype=str) == bucket].tolist()))
                for bucket in sorted(set(np.asarray(out["stage_bucket"], dtype=str).tolist()))
            },
            "two_step_improve_count_lt2_rate": float(np.mean(out["teacher_two_step_improve_count_lt2"])),
            "two_step_horizon_ge_noop_rate": float(np.mean(out["teacher_two_step_horizon_ge_noop"])),
            "two_step_fallback_triggered_rate": float(np.mean(out["teacher_two_step_fallback_triggered"])),
            "two_step_horizon_score_margin": _stats(teacher_two_step_margin_arr),
            "two_step_horizon_score_margin_by_bucket": {
                bucket: _stats(teacher_two_step_margin_arr[np.asarray(out["stage_bucket"], dtype=str) == bucket])
                for bucket in sorted(set(np.asarray(out["stage_bucket"], dtype=str).tolist()))
            },
            "best_improve_missing_axis_histogram": dict(Counter(missing_patterns_all)),
            "best_improve_missing_axis_histogram_given_lt2": dict(Counter(missing_patterns_lt2)),
            "best_improve_missing_axis_rate_given_lt2": {
                "xy": float(np.mean(teacher_best_missing_xy_arr[lt2_mask])) if np.any(lt2_mask) else 0.0,
                "z": float(np.mean(teacher_best_missing_z_arr[lt2_mask])) if np.any(lt2_mask) else 0.0,
                "yaw": float(np.mean(teacher_best_missing_yaw_arr[lt2_mask])) if np.any(lt2_mask) else 0.0,
            },
            "very_close_abstain_rate": float(np.mean(teacher_vca_triggered_arr)),
            "very_close_abstain_axis_rate": {
                "xy": float(np.mean(teacher_vca_xy_arr)),
                "z": float(np.mean(teacher_vca_z_arr)),
                "yaw": float(np.mean(teacher_vca_yaw_arr)),
            },
            "very_close_abstain_thresholds": {
                "xy": teacher_vca_threshold_xy,
                "z": teacher_vca_threshold_z,
                "yaw": teacher_vca_threshold_yaw,
            },
            "very_close_abstain_axis_rate_by_bucket": {
                bucket: {
                    "xy": float(np.mean(teacher_vca_xy_arr[np.asarray(out["stage_bucket"], dtype=str) == bucket])),
                    "z": float(np.mean(teacher_vca_z_arr[np.asarray(out["stage_bucket"], dtype=str) == bucket])),
                    "yaw": float(np.mean(teacher_vca_yaw_arr[np.asarray(out["stage_bucket"], dtype=str) == bucket])),
                    "triggered": float(np.mean(teacher_vca_triggered_arr[np.asarray(out["stage_bucket"], dtype=str) == bucket])),
                    "margin_xy": _stats(teacher_vca_margin_xy_arr[np.asarray(out["stage_bucket"], dtype=str) == bucket]),
                    "margin_z": _stats(teacher_vca_margin_z_arr[np.asarray(out["stage_bucket"], dtype=str) == bucket]),
                    "margin_yaw": _stats(teacher_vca_margin_yaw_arr[np.asarray(out["stage_bucket"], dtype=str) == bucket]),
                }
                for bucket in sorted(set(np.asarray(out["stage_bucket"], dtype=str).tolist()))
            },
            "two_step_rates_by_bucket": {
                bucket: {
                    "improve_count_lt2": float(np.mean(np.asarray(out["teacher_two_step_improve_count_lt2"], dtype=np.float32)[np.asarray(out["stage_bucket"], dtype=str) == bucket])),
                    "horizon_ge_noop": float(np.mean(np.asarray(out["teacher_two_step_horizon_ge_noop"], dtype=np.float32)[np.asarray(out["stage_bucket"], dtype=str) == bucket])),
                    "fallback_triggered": float(np.mean(np.asarray(out["teacher_two_step_fallback_triggered"], dtype=np.float32)[np.asarray(out["stage_bucket"], dtype=str) == bucket])),
                }
                for bucket in sorted(set(np.asarray(out["stage_bucket"], dtype=str).tolist()))
            },
            "stage_target_mode_histogram": dict(Counter(np.asarray(src_data["stage_target_mode"])[selected].tolist())) if "stage_target_mode" in src_data else {},
            "contact_state_histogram": dict(Counter(np.asarray(src_data["contact_state"])[selected].tolist())) if "contact_state" in src_data else {},
            "current_xy": _stats(current_delta[:, :2].reshape(-1)),
            "current_z": _stats(np.abs(current_delta[:, 2])),
            "current_yaw": _stats(np.abs(current_delta[:, 5])),
        },
        "teacher": {
            "post_xy": _stats(out["teacher_post_xy_error"]),
            "post_z": _stats(out["teacher_post_z_error"]),
            "post_yaw": _stats(out["teacher_post_yaw_error"]),
            "improves_xy_rate": float(np.mean(out["teacher_improves_xy"])),
            "improves_z_rate": float(np.mean(out["teacher_improves_z"])),
            "improves_yaw_rate": float(np.mean(out["teacher_improves_yaw"])),
            "all_improves_rate": float(np.mean(out["teacher_all_improves"])),
            "noop_selected_rate": float(np.mean(out["teacher_noop_selected"])),
            "overshoot_any_rate": float(np.mean(out["teacher_overshoot_any"])),
            "invalid_rate": float(np.mean(out["teacher_invalid"])),
            "workspace_violation_rate": float(np.mean(out["teacher_workspace_violation"])),
            "action_pos_norm_mean": float(pos_norm.mean()) if pos_norm.size else 0.0,
            "action_yaw_abs_mean": float(yaw_abs.mean()) if yaw_abs.size else 0.0,
            "candidate_bank_coarse_size": int(coarse_bank.shape[0]),
            "candidate_bank_refine_size": int(refine_offsets.shape[0]),
            "candidate_bank_micro_refine_size": int(micro_refine_offsets.shape[0]),
            "noop_reason_histogram": dict(Counter(np.asarray(out["teacher_noop_reason"], dtype=str).tolist())),
            "noop_reason_histogram_by_bucket": {
                bucket: dict(Counter(np.asarray(out["teacher_noop_reason"], dtype=str)[np.asarray(out["stage_bucket"], dtype=str) == bucket].tolist()))
                for bucket in sorted(set(np.asarray(out["stage_bucket"], dtype=str).tolist()))
            },
            "two_step_improve_count_lt2_rate": float(np.mean(out["teacher_two_step_improve_count_lt2"])),
            "two_step_horizon_ge_noop_rate": float(np.mean(out["teacher_two_step_horizon_ge_noop"])),
            "two_step_fallback_triggered_rate": float(np.mean(out["teacher_two_step_fallback_triggered"])),
            "saturation_rate_dx": float((np.abs(out["teacher_residual_local_6d"][:, 0]) >= np.max(np.abs(_parse_floats(args.dx_values))) - 1e-8).mean()),
            "saturation_rate_dy": float((np.abs(out["teacher_residual_local_6d"][:, 1]) >= np.max(np.abs(_parse_floats(args.dy_values))) - 1e-8).mean()),
            "saturation_rate_dz": float((np.abs(out["teacher_residual_local_6d"][:, 2]) >= np.max(np.abs(_parse_floats(args.dz_values))) - 1e-8).mean()),
            "saturation_rate_dyaw": float((np.abs(out["teacher_residual_local_6d"][:, 5]) >= np.max(np.abs(_parse_floats(args.dyaw_values))) - 1e-8).mean()),
            "beats_noop_xy_rate": teacher_vs_noop_xy,
            "beats_noop_z_rate": teacher_vs_noop_z,
            "beats_noop_yaw_rate": teacher_vs_noop_yaw,
            "beats_servo_xy_rate": float((teacher_eval["post_xy"] < servo_eval["post_xy"]).mean()),
            "beats_servo_z_rate": float((teacher_eval["post_z"] < servo_eval["post_z"]).mean()),
            "beats_servo_yaw_rate": float((teacher_eval["post_yaw"] < servo_eval["post_yaw"]).mean()),
            "beats_noop_mean_delta": {
                "xy": float((teacher_eval["post_xy"] - noop_eval["post_xy"]).mean()),
                "z": float((teacher_eval["post_z"] - noop_eval["post_z"]).mean()),
                "yaw": float((teacher_eval["post_yaw"] - noop_eval["post_yaw"]).mean()),
            },
            "beats_servo_mean_delta": {
                "xy": float((teacher_eval["post_xy"] - servo_eval["post_xy"]).mean()),
                "z": float((teacher_eval["post_z"] - servo_eval["post_z"]).mean()),
                "yaw": float((teacher_eval["post_yaw"] - servo_eval["post_yaw"]).mean()),
            },
        },
        "short_horizon_audit": short_horizon,
        "audit_passed": audit_passed,
        "note": (
            "From-scratch privileged direct-control teacher built from raw privileged rollout; "
            "must pass audit before dataset construction."
        ),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.enforce_audit_pass and not audit_passed:
        raise SystemExit("teacher audit did not pass; refusing to continue")


if __name__ == "__main__":
    main()
