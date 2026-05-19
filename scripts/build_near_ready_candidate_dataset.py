"""
build_near_ready_candidate_dataset.py

Legacy teacher-side builder for near-ready candidate supervision.

This dataset is intentionally teacher-first: it uses teacher-truth handoff
geometry to decide which rows are near-ready and which candidates are best.
That makes it useful for teacher-side candidate supervision, but unsafe as a
runtime self-eval dataset unless a downstream script explicitly redefines the
labels from runtime/student namespaces.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from build_pose_candidate_dataset import (
    apply_local_offset_to_pose,
    basin_distance_bin,
    improvement_tiers,
    pose_delta_local_between,
    sign_bucket,
)


def _choose_target_pose(raw: dict[str, np.ndarray], idx: int) -> np.ndarray | None:
    for key in (
        "grasp_commit_target_pose_7d",
        "handoff_target_pose_7d",
        "motion_target_pose_7d",
        "pregrasp_target_pose_7d",
        "basin_center_pose_7d",
    ):
        arr = raw.get(key, None)
        if arr is None:
            continue
        row = np.asarray(arr[idx])
        if row.shape == (7,) and np.all(np.isfinite(row)):
            return row.astype(np.float32)
    return None


def _metric_from_row(raw: dict[str, np.ndarray], idx: int, teacher_key: str, fallback_key: str) -> float:
    val = np.nan
    if teacher_key in raw:
        val = float(raw[teacher_key][idx])
    if (not np.isfinite(val)) and fallback_key in raw:
        val = float(raw[fallback_key][idx])
    return val


def _threshold_from_row(raw: dict[str, np.ndarray], idx: int, teacher_key: str, fallback_key: str, default: float) -> float:
    val = np.nan
    if teacher_key in raw:
        val = float(raw[teacher_key][idx])
    if (not np.isfinite(val)) and fallback_key in raw:
        val = float(raw[fallback_key][idx])
    if not np.isfinite(val) or val <= 0.0:
        val = float(default)
    return float(val)


def _safe_float(val, default: float) -> float:
    try:
        out = float(val)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _runtime_metric_from_row(raw: dict[str, np.ndarray], idx: int, key: str) -> float:
    runtime_key = f"runtime_handoff_metric_{key}"
    legacy_key = f"handoff_metric_{key}"
    if runtime_key in raw:
        return _safe_float(raw[runtime_key][idx], np.nan)
    if legacy_key in raw:
        return _safe_float(raw[legacy_key][idx], np.nan)
    return float("nan")


def _near_ready_cost(
    xy: float,
    z: float,
    yaw: float,
    *,
    rel_xy: float,
    rel_z: float,
    rel_yaw: float,
    w_xy: float,
    w_yaw: float,
    w_z_guard: float,
) -> float:
    xy_term = w_xy * (xy / max(rel_xy, 1e-6)) ** 2
    yaw_term = w_yaw * (yaw / max(rel_yaw, 1e-6)) ** 2
    z_term = w_z_guard * max(z / max(rel_z, 1e-6) - 1.0, 0.0) ** 2
    return float(xy_term + yaw_term + z_term)


def _load_row_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: np.asarray(data[k]) for k in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=str, action="append", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--open_threshold", type=float, default=0.5)
    parser.add_argument("--release_xy_default", type=float, default=0.007)
    parser.add_argument("--release_abs_z_default", type=float, default=0.0035)
    parser.add_argument("--release_yaw_default", type=float, default=0.12434)
    parser.add_argument("--near_xy_max_mult", type=float, default=4.0)
    parser.add_argument("--near_abs_z_max_mult", type=float, default=2.0)
    parser.add_argument("--near_yaw_max_mult", type=float, default=4.0)
    parser.set_defaults(exclude_already_ready=True)
    parser.add_argument("--exclude_already_ready", dest="exclude_already_ready", action="store_true")
    parser.add_argument("--keep_already_ready", dest="exclude_already_ready", action="store_false")
    parser.add_argument("--keep_ready_fraction", type=float, default=0.0)
    parser.add_argument("--w_xy", type=float, default=1.0)
    parser.add_argument("--w_yaw", type=float, default=0.75)
    parser.add_argument("--w_z_guard", type=float, default=0.35)
    parser.add_argument("--ready_bonus", type=float, default=0.50)
    parser.add_argument("--near_ready_sample_weight", type=float, default=2.0)
    parser.add_argument("--hard_negative_sample_weight", type=float, default=2.5)
    parser.add_argument("--hard_positive_sample_weight", type=float, default=1.5)
    args = parser.parse_args()

    rows = []
    candidate_actions = None
    candidate_group_index = None

    for raw_path in args.input_npz:
        raw = _load_row_npz(raw_path)
        row_count = int(raw["current_pose_7d"].shape[0])
        keep_stride = None
        keep_fraction = float(args.keep_ready_fraction)
        if keep_fraction > 0.0:
            keep_stride = max(1, int(round(1.0 / max(keep_fraction, 1e-6))))
        for idx in range(row_count):
            if "phase_id" in raw and int(raw["phase_id"][idx]) != 1:
                continue
            if "rollout_gripper_open" in raw and float(raw["rollout_gripper_open"][idx]) < float(args.open_threshold):
                continue
            target_pose = _choose_target_pose(raw, idx)
            if target_pose is None:
                continue
            current_pose = np.asarray(raw["current_pose_7d"][idx], dtype=np.float32)
            cur_xy = _metric_from_row(raw, idx, "teacher_truth_handoff_metric_xy_error", "handoff_metric_xy_error")
            cur_z = _metric_from_row(raw, idx, "teacher_truth_handoff_metric_abs_z_error", "handoff_metric_abs_z_error")
            cur_yaw = _metric_from_row(raw, idx, "teacher_truth_handoff_metric_yaw_error", "handoff_metric_yaw_error")
            if not (np.isfinite(cur_xy) and np.isfinite(cur_z) and np.isfinite(cur_yaw)):
                cur_delta = pose_delta_local_between(current_pose, target_pose)
                cur_xy = float(np.linalg.norm(cur_delta[:2]))
                cur_z = float(abs(cur_delta[2]))
                cur_yaw = float(abs(cur_delta[5]))
            rel_xy = _threshold_from_row(
                raw, idx,
                "teacher_truth_handoff_release_threshold_xy_error",
                "handoff_release_threshold_xy_error",
                args.release_xy_default,
            )
            rel_z = _threshold_from_row(
                raw, idx,
                "teacher_truth_handoff_release_threshold_abs_z_error",
                "handoff_release_threshold_abs_z_error",
                args.release_abs_z_default,
            )
            rel_yaw = _threshold_from_row(
                raw, idx,
                "teacher_truth_handoff_release_threshold_yaw_error",
                "handoff_release_threshold_yaw_error",
                args.release_yaw_default,
            )
            near_ready = bool(
                cur_xy <= rel_xy * float(args.near_xy_max_mult)
                and cur_z <= rel_z * float(args.near_abs_z_max_mult)
                and cur_yaw <= rel_yaw * float(args.near_yaw_max_mult)
            )
            already_ready = bool(cur_xy <= rel_xy and cur_z <= rel_z and cur_yaw <= rel_yaw)
            if not near_ready:
                continue
            if bool(args.exclude_already_ready) and already_ready:
                keep_ready = bool(
                    keep_stride is not None
                    and (idx % keep_stride == 0)
                )
                if not keep_ready:
                    continue
            teacher_ready = bool(
                "teacher_truth_handoff_ready" in raw
                and float(raw["teacher_truth_handoff_ready"][idx]) > 0.5
            )
            if teacher_ready:
                already_ready = True
            if bool(args.exclude_already_ready) and already_ready and keep_stride is None and not teacher_ready:
                continue

            row_cands = np.asarray(raw["candidate_actions_local"][idx], dtype=np.float32)
            row_groups = np.asarray(raw["candidate_group_index"][idx], dtype=np.int64)
            if candidate_actions is None:
                candidate_actions = row_cands
                candidate_group_index = row_groups
            row_mask = np.asarray(raw["candidate_mask"][idx], dtype=np.float32) if "candidate_mask" in raw else np.ones((row_cands.shape[0],), dtype=np.float32)
            rows.append(
                {
                    "front_rgb": np.asarray(raw["front_rgb"][idx], dtype=np.uint8) if "front_rgb" in raw else np.zeros((128, 128, 3), dtype=np.uint8),
                    "wrist_rgb": np.asarray(raw["wrist_rgb"][idx], dtype=np.uint8) if "wrist_rgb" in raw else np.zeros((128, 128, 3), dtype=np.uint8),
                    "wrist_depth": np.asarray(raw["wrist_depth"][idx], dtype=np.float32),
                    "proprio": np.asarray(raw["proprio"][idx], dtype=np.float32),
                    "base_action": np.asarray(raw["base_action"][idx], dtype=np.float32),
                    "gripper_context": np.asarray(raw["gripper_context"][idx], dtype=np.float32),
                    "step_idx": int(raw["step_idx"][idx]) if "step_idx" in raw else 0,
                    "phase_id": int(raw["phase_id"][idx]) if "phase_id" in raw else 1,
                    "substage_id": int(raw["substage_id"][idx]) if "substage_id" in raw else int(raw["phase_id"][idx]) if "phase_id" in raw else 1,
                    "episode_index": int(raw["episode_index"][idx]) if "episode_index" in raw else 0,
                    "phase_age": float(raw["phase_age"][idx]) if "phase_age" in raw else 0.0,
                    "steps_since_last_replan": float(raw["steps_since_last_replan"][idx]) if "steps_since_last_replan" in raw else 0.0,
                    "current_pose": current_pose,
                    "target_pose": target_pose,
                    "candidate_mask": row_mask,
                    "cur_xy": float(cur_xy),
                    "cur_z": float(cur_z),
                    "cur_yaw": float(cur_yaw),
                    "rel_xy": float(rel_xy),
                    "rel_z": float(rel_z),
                    "rel_yaw": float(rel_yaw),
                    "teacher_truth_handoff_ready": _safe_float(raw["teacher_truth_handoff_ready"][idx], 0.0) if "teacher_truth_handoff_ready" in raw else 0.0,
                    "teacher_truth_xy": _safe_float(raw["teacher_truth_handoff_metric_xy_error"][idx], float(cur_xy)) if "teacher_truth_handoff_metric_xy_error" in raw else float(cur_xy),
                    "teacher_truth_yaw": _safe_float(raw["teacher_truth_handoff_metric_yaw_error"][idx], float(cur_yaw)) if "teacher_truth_handoff_metric_yaw_error" in raw else float(cur_yaw),
                    "student_runtime_xy": _runtime_metric_from_row(raw, idx, "xy_error"),
                    "student_runtime_yaw": _runtime_metric_from_row(raw, idx, "yaw_error"),
                    "student_runtime_valid": _safe_float(
                        raw["runtime_handoff_metric_valid"][idx],
                        0.0,
                    )
                    if "runtime_handoff_metric_valid" in raw
                    else float(np.isfinite(_runtime_metric_from_row(raw, idx, "xy_error"))),
                    "proxy_current_delta": np.asarray(
                        raw["proxy_current_delta_basin_target"][idx], dtype=np.float32
                    )
                    if "proxy_current_delta_basin_target" in raw
                    else (
                        np.asarray(raw["current_delta_basin_target"][idx], dtype=np.float32)
                        if "current_delta_basin_target" in raw
                        else pose_delta_local_between(current_pose, target_pose).astype(np.float32)
                    ),
                    "teacher_current_delta": np.asarray(
                        raw["teacher_current_delta_basin_target"][idx], dtype=np.float32
                    )
                    if "teacher_current_delta_basin_target" in raw
                    else (
                        np.asarray(raw["target_delta_teacher"][idx], dtype=np.float32)
                        if "target_delta_teacher" in raw
                        else pose_delta_local_between(current_pose, target_pose).astype(np.float32)
                    ),
                }
            )

    if not rows or candidate_actions is None or candidate_group_index is None:
        raise RuntimeError("No near-ready candidate distillation rows found.")

    num_states = len(rows)
    num_cands = int(candidate_actions.shape[0])
    out = {
        "front_rgb": np.zeros((num_states, 128, 128, 3), dtype=np.uint8),
        "wrist_rgb": np.zeros((num_states, 128, 128, 3), dtype=np.uint8),
        "wrist_depth": np.zeros((num_states, 1, 96, 96), dtype=np.float32),
        "proprio": np.zeros((num_states, 15), dtype=np.float32),
        "base_action": np.zeros((num_states, 6), dtype=np.float32),
        "gripper_context": np.zeros((num_states, 3), dtype=np.float32),
        "planner_close_intent": np.zeros((num_states,), dtype=np.float32),
        "step_idx": np.zeros((num_states,), dtype=np.int64),
        "phase_id": np.ones((num_states,), dtype=np.int64),
        "substage_id": np.ones((num_states,), dtype=np.int64),
        "episode_index": np.zeros((num_states,), dtype=np.int64),
        "phase_age": np.zeros((num_states,), dtype=np.float32),
        "steps_since_last_replan": np.zeros((num_states,), dtype=np.float32),
        "candidate_actions_local": np.repeat(candidate_actions[None, :, :], num_states, axis=0).astype(np.float32),
        "candidate_group_index": np.repeat(candidate_group_index[None, :], num_states, axis=0).astype(np.int64),
        "candidate_mask": np.zeros((num_states, num_cands), dtype=np.float32),
        "candidate_improvement": np.zeros((num_states, num_cands), dtype=np.float32),
        "candidate_oracle_score": np.full((num_states, num_cands), -1e9, dtype=np.float32),
        "candidate_next_basin_distance": np.zeros((num_states, num_cands), dtype=np.float32),
        "candidate_tier": np.zeros((num_states, num_cands), dtype=np.int64),
        "candidate_basin_positive": np.zeros((num_states, num_cands), dtype=np.float32),
        "current_delta_basin_target": np.zeros((num_states, 6), dtype=np.float32),
        "proxy_current_delta_basin_target": np.zeros((num_states, 6), dtype=np.float32),
        "teacher_current_delta_basin_target": np.zeros((num_states, 6), dtype=np.float32),
        "teacher_truth_handoff_metric_xy_error": np.zeros((num_states,), dtype=np.float32),
        "teacher_truth_handoff_metric_abs_z_error": np.zeros((num_states,), dtype=np.float32),
        "teacher_truth_handoff_metric_yaw_error": np.zeros((num_states,), dtype=np.float32),
        "teacher_metrics_norm": np.zeros((num_states, 3), dtype=np.float32),
        "current_basin_distance": np.zeros((num_states,), dtype=np.float32),
        "current_dx_sign": np.zeros((num_states,), dtype=np.int64),
        "current_dy_sign": np.zeros((num_states,), dtype=np.int64),
        "current_dyaw_sign": np.zeros((num_states,), dtype=np.int64),
        "basin_distance_bin": np.zeros((num_states,), dtype=np.int64),
        "best_candidate_index": np.zeros((num_states,), dtype=np.int64),
        "best_group_index": np.zeros((num_states,), dtype=np.int64),
        "ready_to_close_target": np.zeros((num_states,), dtype=np.float32),
        "sample_weight": np.ones((num_states,), dtype=np.float32),
        "yaw_hard_negative": np.zeros((num_states,), dtype=np.float32),
        "yaw_hard_positive": np.zeros((num_states,), dtype=np.float32),
        "xy_focus": np.zeros((num_states,), dtype=np.float32),
        "near_ready_xy_z_band": np.ones((num_states,), dtype=np.float32),
        "teacher_truth_handoff_ready": np.zeros((num_states,), dtype=np.float32),
        "teacher_minus_student_xy_gap": np.zeros((num_states,), dtype=np.float32),
        "teacher_minus_student_yaw_gap": np.zeros((num_states,), dtype=np.float32),
        "runtime_handoff_metric_valid": np.zeros((num_states,), dtype=np.float32),
        "runtime_handoff_ready": np.zeros((num_states,), dtype=np.float32),
        "runtime_handoff_ready_pred": np.zeros((num_states,), dtype=np.float32),
        "runtime_handoff_ready_applied": np.zeros((num_states,), dtype=np.float32),
        "dominant_axis_bucket": np.zeros((num_states,), dtype=np.int64),
        "ready_support": np.zeros((num_states,), dtype=np.float32),
        "near_xy_hard": np.zeros((num_states,), dtype=np.float32),
        "near_yaw_hard": np.zeros((num_states,), dtype=np.float32),
        "near_coupled": np.zeros((num_states,), dtype=np.float32),
        "dataset_semantics": np.asarray("legacy_teacher_side", dtype="<U32"),
        "label_semantics": np.asarray("teacher_first_near_ready", dtype="<U64"),
    }

    for row, row_data in enumerate(rows):
        current_pose = row_data["current_pose"]
        target_pose = row_data["target_pose"]
        cur_xy = row_data["cur_xy"]
        cur_z = row_data["cur_z"]
        cur_yaw = row_data["cur_yaw"]
        rel_xy = row_data["rel_xy"]
        rel_z = row_data["rel_z"]
        rel_yaw = row_data["rel_yaw"]
        proxy_current_delta = np.asarray(row_data["proxy_current_delta"], dtype=np.float32).reshape(6)
        teacher_current_delta = np.asarray(row_data["teacher_current_delta"], dtype=np.float32).reshape(6)
        current_delta = proxy_current_delta
        out["front_rgb"][row] = row_data["front_rgb"]
        out["wrist_rgb"][row] = row_data["wrist_rgb"]
        depth = np.asarray(row_data["wrist_depth"], dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None, ...]
        out["wrist_depth"][row] = depth
        out["proprio"][row] = row_data["proprio"]
        out["base_action"][row] = row_data["base_action"]
        out["gripper_context"][row] = row_data["gripper_context"]
        out["planner_close_intent"][row] = float(row_data["gripper_context"][1] <= 0.5) if row_data["gripper_context"].shape[0] >= 2 else 0.0
        out["step_idx"][row] = row_data["step_idx"]
        out["phase_id"][row] = row_data["phase_id"]
        out["substage_id"][row] = row_data["substage_id"]
        out["episode_index"][row] = row_data["episode_index"]
        out["phase_age"][row] = row_data["phase_age"]
        out["steps_since_last_replan"][row] = row_data["steps_since_last_replan"]
        out["candidate_mask"][row] = row_data["candidate_mask"]
        out["current_delta_basin_target"][row] = current_delta
        out["proxy_current_delta_basin_target"][row] = proxy_current_delta
        out["teacher_current_delta_basin_target"][row] = teacher_current_delta
        out["teacher_truth_handoff_metric_xy_error"][row] = float(cur_xy)
        out["teacher_truth_handoff_metric_abs_z_error"][row] = float(cur_z)
        out["teacher_truth_handoff_metric_yaw_error"][row] = float(cur_yaw)
        out["teacher_metrics_norm"][row] = np.asarray(
            [
                float(cur_xy / max(rel_xy, 1e-6)),
                float(cur_z / max(rel_z, 1e-6)),
                float(cur_yaw / max(rel_yaw, 1e-6)),
            ],
            dtype=np.float32,
        )

        current_cost = _near_ready_cost(
            cur_xy, cur_z, cur_yaw,
            rel_xy=rel_xy, rel_z=rel_z, rel_yaw=rel_yaw,
            w_xy=float(args.w_xy), w_yaw=float(args.w_yaw), w_z_guard=float(args.w_z_guard),
        )
        out["current_basin_distance"][row] = float(current_cost)
        out["current_dx_sign"][row] = sign_bucket(float(current_delta[0]), 1e-4)
        out["current_dy_sign"][row] = sign_bucket(float(current_delta[1]), 1e-4)
        out["current_dyaw_sign"][row] = sign_bucket(float(current_delta[5]), 1e-3)
        out["basin_distance_bin"][row] = basin_distance_bin(float(current_cost))

        ready_target = float(cur_xy <= rel_xy and cur_z <= rel_z and cur_yaw <= rel_yaw)
        teacher_ready_target = float(row_data["teacher_truth_handoff_ready"] > 0.5 or ready_target > 0.5)
        out["ready_to_close_target"][row] = ready_target
        out["teacher_truth_handoff_ready"][row] = teacher_ready_target
        out["ready_support"][row] = teacher_ready_target
        runtime_metric_valid = bool(row_data["student_runtime_valid"] > 0.5 and np.isfinite(row_data["student_runtime_xy"]) and np.isfinite(row_data["student_runtime_yaw"]))
        teacher_minus_student_xy_gap = float(max(row_data["teacher_truth_xy"] - row_data["student_runtime_xy"], 0.0)) if runtime_metric_valid else 0.0
        teacher_minus_student_yaw_gap = float(max(row_data["teacher_truth_yaw"] - row_data["student_runtime_yaw"], 0.0)) if runtime_metric_valid else 0.0
        out["teacher_minus_student_xy_gap"][row] = teacher_minus_student_xy_gap
        out["teacher_minus_student_yaw_gap"][row] = teacher_minus_student_yaw_gap
        out["runtime_handoff_metric_valid"][row] = float(runtime_metric_valid)
        out["runtime_handoff_ready"][row] = float(
            runtime_metric_valid
            and row_data["student_runtime_xy"] <= rel_xy
            and cur_z <= rel_z
            and row_data["student_runtime_yaw"] <= rel_yaw
        )
        out["runtime_handoff_ready_pred"][row] = out["runtime_handoff_ready"][row]
        out["runtime_handoff_ready_applied"][row] = 0.0
        xy_norm = float(cur_xy / max(rel_xy, 1e-6))
        yaw_norm = float(cur_yaw / max(rel_yaw, 1e-6))
        if xy_norm > yaw_norm * 1.1:
            dominant_axis = 1
            out["xy_focus"][row] = 1.0
        elif yaw_norm > xy_norm * 1.1:
            dominant_axis = 2
        else:
            dominant_axis = 3
            out["xy_focus"][row] = 1.0
        out["dominant_axis_bucket"][row] = dominant_axis
        near_xy_hard = bool(cur_z <= rel_z * 1.2 and cur_xy > rel_xy and cur_yaw <= rel_yaw * 1.2)
        near_yaw_hard = bool(cur_z <= rel_z * 1.2 and cur_yaw > rel_yaw and cur_xy <= rel_xy * 1.2)
        near_coupled = bool(cur_z <= rel_z * 1.2 and cur_xy > rel_xy and cur_yaw > rel_yaw)
        out["near_xy_hard"][row] = float(near_xy_hard)
        out["near_yaw_hard"][row] = float(near_yaw_hard)
        out["near_coupled"][row] = float(near_coupled)
        hard_negative = bool(cur_z <= rel_z * 1.2 and (cur_xy > rel_xy or cur_yaw > rel_yaw))
        if hard_negative:
            out["sample_weight"][row] *= float(args.hard_negative_sample_weight)
            out["yaw_hard_negative"][row] = float((cur_yaw > rel_yaw) or (teacher_minus_student_yaw_gap > 0.0))
        if teacher_ready_target > 0.5:
            out["sample_weight"][row] *= float(args.hard_positive_sample_weight)
            out["yaw_hard_positive"][row] = float(cur_yaw <= rel_yaw)
        else:
            out["sample_weight"][row] *= float(args.near_ready_sample_weight)

        best_idx = 0
        best_score = -1e18
        for j in range(num_cands):
            if out["candidate_mask"][row, j] <= 0.5:
                continue
            cand = candidate_actions[j]
            next_pose = apply_local_offset_to_pose(current_pose, cand)
            next_delta = pose_delta_local_between(next_pose, target_pose)
            next_xy = float(np.linalg.norm(next_delta[:2]))
            next_z = float(abs(next_delta[2]))
            next_yaw = float(abs(next_delta[5]))
            next_cost = _near_ready_cost(
                next_xy, next_z, next_yaw,
                rel_xy=rel_xy, rel_z=rel_z, rel_yaw=rel_yaw,
                w_xy=float(args.w_xy), w_yaw=float(args.w_yaw), w_z_guard=float(args.w_z_guard),
            )
            improvement = float(current_cost - next_cost)
            ready_bonus = float(args.ready_bonus) if (next_xy <= rel_xy and next_z <= rel_z and next_yaw <= rel_yaw) else 0.0
            score = float(improvement + ready_bonus)
            out["candidate_next_basin_distance"][row, j] = float(next_cost)
            out["candidate_improvement"][row, j] = improvement
            out["candidate_oracle_score"][row, j] = score
            out["candidate_basin_positive"][row, j] = float(next_xy <= rel_xy and next_z <= rel_z and next_yaw <= rel_yaw)
            if score > best_score:
                best_score = score
                best_idx = j
        out["best_candidate_index"][row] = int(best_idx)
        out["best_group_index"][row] = int(candidate_group_index[best_idx])
        out["candidate_tier"][row] = improvement_tiers(
            out["candidate_oracle_score"][row],
            out["candidate_basin_positive"][row],
        ).astype(np.int64)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)

    meta = {
        "input_npz": [str(x) for x in args.input_npz],
        "num_rows": int(num_states),
        "num_candidates": int(num_cands),
        "ready_positive_count": int(np.sum(out["ready_to_close_target"] > 0.5)),
        "teacher_ready_positive_count": int(np.sum(out["teacher_truth_handoff_ready"] > 0.5)),
        "hard_negative_count": int(np.sum(out["yaw_hard_negative"] > 0.5)),
        "best_candidate_hist": {
            int(k): int(v) for k, v in zip(*np.unique(out["best_candidate_index"], return_counts=True))
        },
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
