#!/usr/bin/env python3
"""Build a privileged direct-control teacher dataset for alignment_v3.

The builder uses privileged pose targets from support states and searches a
dense local residual grid directly against the motion target pose. It does not
reuse the frozen K=8 best-stage candidate bank as the teacher source.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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


def _solve_teacher_for_row(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    current_delta: np.ndarray,
    candidate_bank: np.ndarray,
    stage_bucket: str,
) -> dict[str, np.ndarray | float | int | str]:
    cur_pose = np.asarray(current_pose, dtype=np.float32).reshape(7)
    tgt_pose = np.asarray(target_pose, dtype=np.float32).reshape(7)
    cur_delta = np.asarray(current_delta, dtype=np.float32).reshape(6)
    cands = np.asarray(candidate_bank, dtype=np.float32).reshape(-1, 6)

    cur_xy = float(np.linalg.norm(cur_delta[:2]))
    cur_z = float(abs(cur_delta[2]))
    cur_yaw = float(abs(cur_delta[5]))

    cur_rot = Rotation.from_quat(cur_pose[3:7])
    tgt_rot = Rotation.from_quat(tgt_pose[3:7])

    cand_rot = Rotation.from_rotvec(cands[:, 3:6].astype(np.float32))
    next_rot = cand_rot * cur_rot
    next_pos = cur_pose[:3][None, :] + cur_rot.apply(cands[:, :3]).astype(np.float32)
    delta_pos_world = tgt_pose[:3][None, :] - next_pos
    post_pos_local = next_rot.inv().apply(delta_pos_world).astype(np.float32)
    post_rot = (tgt_rot * next_rot.inv()).as_rotvec().astype(np.float32)
    post = np.concatenate([post_pos_local, post_rot], axis=-1)

    post_xy = np.linalg.norm(post[:, :2], axis=-1)
    post_z = np.abs(post[:, 2])
    post_yaw = np.abs(post[:, 5])
    action_pos_norm = np.linalg.norm(cands[:, :3], axis=-1)
    action_yaw_abs = np.abs(cands[:, 5])
    overshoot_any = (post_xy > cur_xy + 1e-8) | (post_z > cur_z + 1e-8) | (post_yaw > cur_yaw + 1e-8)
    nonzero = ((action_pos_norm > 1e-8) | (action_yaw_abs > 1e-8)).astype(np.int32)
    overshoot_i = overshoot_any.astype(np.int32)
    safe = np.zeros_like(nonzero, dtype=np.int32)

    if stage_bucket == "micro_contact_refine":
        primary = (post_xy + post_z + 1.5 * post_yaw).astype(np.float32)
        keys = (nonzero, action_yaw_abs, action_pos_norm, overshoot_i, primary, safe)
    else:
        # Near-contact: z first, then xy, yaw, action norm, overshoot.
        keys = (nonzero, overshoot_i, action_pos_norm, post_yaw, post_xy, post_z, safe)

    # np.lexsort uses the last key as the primary key.
    best_idx = int(np.lexsort(keys)[0])
    best_residual = cands[best_idx]
    best_post = post[best_idx]

    teacher_residual_6d = best_residual.astype(np.float32)
    teacher_residual_4d = np.asarray(
        [teacher_residual_6d[0], teacher_residual_6d[1], teacher_residual_6d[2], teacher_residual_6d[5]],
        dtype=np.float32,
    )
    teacher_post_xy = float(np.linalg.norm(best_post[:2]))
    teacher_post_z = float(abs(best_post[2]))
    teacher_post_yaw = float(abs(best_post[5]))
    teacher_improves_xy = float(teacher_post_xy < cur_xy)
    teacher_improves_z = float(teacher_post_z < cur_z)
    teacher_improves_yaw = float(teacher_post_yaw < cur_yaw)
    teacher_all_improves = float(teacher_improves_xy * teacher_improves_z * teacher_improves_yaw)
    teacher_overshoot_xy = float(teacher_post_xy > cur_xy + 1e-8)
    teacher_overshoot_z = float(teacher_post_z > cur_z + 1e-8)
    teacher_overshoot_yaw = float(teacher_post_yaw > cur_yaw + 1e-8)
    teacher_overshoot_any = float(teacher_overshoot_xy or teacher_overshoot_z or teacher_overshoot_yaw)
    teacher_action_pos_norm = float(np.linalg.norm(best_residual[:3]))
    teacher_action_yaw_abs = float(abs(best_residual[5]))

    return {
        "teacher_residual_local_4d": teacher_residual_4d,
        "teacher_residual_local_6d": teacher_residual_6d,
        "teacher_post_xy_error": np.asarray(teacher_post_xy, dtype=np.float32),
        "teacher_post_z_error": np.asarray(teacher_post_z, dtype=np.float32),
        "teacher_post_yaw_error": np.asarray(teacher_post_yaw, dtype=np.float32),
        "teacher_improves_xy": np.asarray(teacher_improves_xy, dtype=np.float32),
        "teacher_improves_z": np.asarray(teacher_improves_z, dtype=np.float32),
        "teacher_improves_yaw": np.asarray(teacher_improves_yaw, dtype=np.float32),
        "teacher_all_improves": np.asarray(teacher_all_improves, dtype=np.float32),
        "teacher_overshoot_xy": np.asarray(teacher_overshoot_xy, dtype=np.float32),
        "teacher_overshoot_z": np.asarray(teacher_overshoot_z, dtype=np.float32),
        "teacher_overshoot_yaw": np.asarray(teacher_overshoot_yaw, dtype=np.float32),
        "teacher_overshoot_any": np.asarray(teacher_overshoot_any, dtype=np.float32),
        "teacher_noop_selected": np.asarray(float(best_idx == 0), dtype=np.float32),
        "teacher_action_pos_norm": np.asarray(teacher_action_pos_norm, dtype=np.float32),
        "teacher_action_yaw_abs": np.asarray(teacher_action_yaw_abs, dtype=np.float32),
        "teacher_best_candidate_index": np.asarray(best_idx, dtype=np.int64),
        "teacher_objective_stage": np.asarray(0 if stage_bucket == "near_alignment" else 1, dtype=np.int64),
        "teacher_objective_primary": np.asarray(
            post_z[best_idx] if stage_bucket == "near_alignment" else (post_xy + post_z + 1.5 * post_yaw)[best_idx],
            dtype=np.float32,
        ),
        "teacher_objective_secondary": np.asarray(
            post_xy[best_idx] if stage_bucket == "near_alignment" else overshoot_i[best_idx],
            dtype=np.float32,
        ),
        "teacher_objective_tertiary": np.asarray(
            post_yaw[best_idx] if stage_bucket == "near_alignment" else action_pos_norm[best_idx],
            dtype=np.float32,
        ),
        "teacher_objective_quaternary": np.asarray(
            action_pos_norm[best_idx] if stage_bucket == "near_alignment" else action_yaw_abs[best_idx],
            dtype=np.float32,
        ),
        "teacher_objective_quinary": np.asarray(
            overshoot_i[best_idx] if stage_bucket == "near_alignment" else nonzero[best_idx],
            dtype=np.float32,
        ),
        "current_xy_error": np.asarray(cur_xy, dtype=np.float32),
        "current_z_error": np.asarray(cur_z, dtype=np.float32),
        "current_yaw_error": np.asarray(cur_yaw, dtype=np.float32),
        "current_to_target_delta_local": cur_delta.astype(np.float32),
        "current_pose_7d": cur_pose.astype(np.float32),
        "motion_target_pose_7d": tgt_pose.astype(np.float32),
    }


def _bounded_servo_pseudo(
    delta_local: np.ndarray,
    *,
    k_xy: float,
    k_z: float,
    k_yaw: float,
    max_pos: float,
    max_yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
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

    post = delta_local - residual
    return residual, post


def _evaluate_policy_rows(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, np.ndarray]:
    n = int(current_pose.shape[0])
    out = {
        "residual_4d": np.zeros((n, 4), dtype=np.float32),
        "residual_6d": np.zeros((n, 6), dtype=np.float32),
        "post_xy": np.zeros((n,), dtype=np.float32),
        "post_z": np.zeros((n,), dtype=np.float32),
        "post_yaw": np.zeros((n,), dtype=np.float32),
        "improves_xy": np.zeros((n,), dtype=np.float32),
        "improves_z": np.zeros((n,), dtype=np.float32),
        "improves_yaw": np.zeros((n,), dtype=np.float32),
        "all_improves": np.zeros((n,), dtype=np.float32),
        "overshoot_any": np.zeros((n,), dtype=np.float32),
        "action_pos_norm": np.zeros((n,), dtype=np.float32),
        "action_yaw_abs": np.zeros((n,), dtype=np.float32),
    }
    for i in range(n):
        nxt = apply_local_offset_to_pose(current_pose[i], residuals[i])
        post = pose_delta_local_between(nxt, target_pose[i]).astype(np.float32)
        cur = pose_delta_local_between(current_pose[i], target_pose[i]).astype(np.float32)
        post_xy = float(np.linalg.norm(post[:2]))
        post_z = float(abs(post[2]))
        post_yaw = float(abs(post[5]))
        cur_xy = float(np.linalg.norm(cur[:2]))
        cur_z = float(abs(cur[2]))
        cur_yaw = float(abs(cur[5]))
        out["residual_4d"][i] = np.asarray([residuals[i, 0], residuals[i, 1], residuals[i, 2], residuals[i, 5]], dtype=np.float32)
        out["residual_6d"][i] = residuals[i].astype(np.float32)
        out["post_xy"][i] = post_xy
        out["post_z"][i] = post_z
        out["post_yaw"][i] = post_yaw
        out["improves_xy"][i] = float(post_xy < cur_xy)
        out["improves_z"][i] = float(post_z < cur_z)
        out["improves_yaw"][i] = float(post_yaw < cur_yaw)
        out["all_improves"][i] = out["improves_xy"][i] * out["improves_z"][i] * out["improves_yaw"][i]
        out["overshoot_any"][i] = float(post_xy > cur_xy + 1e-8 or post_z > cur_z + 1e-8 or post_yaw > cur_yaw + 1e-8)
        out["action_pos_norm"][i] = float(np.linalg.norm(residuals[i, :3]))
        out["action_yaw_abs"][i] = float(abs(residuals[i, 5]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_npz", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, required=True)
    ap.add_argument("--dx_values", type=str, default="-0.004,-0.002,0.0,0.002,0.004")
    ap.add_argument("--dy_values", type=str, default="-0.004,-0.002,0.0,0.002,0.004")
    ap.add_argument("--dz_values", type=str, default="-0.010,-0.005,0.0,0.005,0.010")
    ap.add_argument("--dyaw_values", type=str, default="-0.006,-0.003,0.0,0.003,0.006")
    ap.add_argument("--keep_source_xy_norm", type=float, default=0.010)
    ap.add_argument("--keep_source_z_abs", type=float, default=0.004)
    ap.add_argument("--keep_source_yaw_abs", type=float, default=0.015)
    ap.add_argument("--micro_source_xy_norm", type=float, default=0.005)
    ap.add_argument("--micro_source_z_abs", type=float, default=0.004)
    ap.add_argument("--micro_source_yaw_abs", type=float, default=0.015)
    ap.add_argument("--servo_k_xy", type=float, default=0.08)
    ap.add_argument("--servo_k_z", type=float, default=0.06)
    ap.add_argument("--servo_k_yaw", type=float, default=0.04)
    ap.add_argument("--servo_max_pos", type=float, default=0.0010)
    ap.add_argument("--servo_max_yaw", type=float, default=0.0040)
    args = ap.parse_args()

    src = np.load(args.source_npz, allow_pickle=True)
    required = ["current_pose_7d", "motion_target_pose_7d", "residual_label_local"]
    for k in required:
        if k not in src.files:
            raise SystemExit(f"source npz missing required field: {k}")

    src_residual = np.asarray(src["residual_label_local"], dtype=np.float32)
    src_res_xy = np.linalg.norm(src_residual[:, :2], axis=-1)
    src_res_z = np.abs(src_residual[:, 2])
    src_res_yaw = np.abs(src_residual[:, 5])
    source_keep = (
        (src_res_xy <= float(args.keep_source_xy_norm))
        & (src_res_z <= float(args.keep_source_z_abs))
        & (src_res_yaw <= float(args.keep_source_yaw_abs))
    )
    if not np.any(source_keep):
        raise SystemExit("No rows passed source near/micro filter.")
    source_micro = (
        source_keep
        & (src_res_xy <= float(args.micro_source_xy_norm))
        & (src_res_z <= float(args.micro_source_z_abs))
        & (src_res_yaw <= float(args.micro_source_yaw_abs))
    )

    indices = np.where(source_keep)[0]
    selected_stage_bucket = np.where(source_micro[indices], "micro_contact_refine", "near_alignment")

    current_pose = np.asarray(src["current_pose_7d"], dtype=np.float32)[indices]
    target_pose = np.asarray(src["motion_target_pose_7d"], dtype=np.float32)[indices]
    source_current_delta = np.stack([pose_delta_local_between(c, t) for c, t in zip(current_pose, target_pose)], axis=0).astype(np.float32)
    servo_residuals = np.zeros((indices.size, 6), dtype=np.float32)
    for i, delta in enumerate(source_current_delta):
        servo_residuals[i], _ = _bounded_servo_pseudo(
            delta,
            k_xy=float(args.servo_k_xy),
            k_z=float(args.servo_k_z),
            k_yaw=float(args.servo_k_yaw),
            max_pos=float(args.servo_max_pos),
            max_yaw=float(args.servo_max_yaw),
        )

    servo_eval = _evaluate_policy_rows(current_pose, target_pose, servo_residuals)

    teacher_rows: list[dict[str, np.ndarray | float | int | str]] = []
    for i, src_i in enumerate(indices.tolist()):
        teacher_rows.append(
            {
                "teacher_residual_local_4d": np.asarray(
                    [servo_residuals[i, 0], servo_residuals[i, 1], servo_residuals[i, 2], servo_residuals[i, 5]],
                    dtype=np.float32,
                ),
                "teacher_residual_local_6d": servo_residuals[i].astype(np.float32),
                "teacher_post_xy_error": np.asarray(servo_eval["post_xy"][i], dtype=np.float32),
                "teacher_post_z_error": np.asarray(servo_eval["post_z"][i], dtype=np.float32),
                "teacher_post_yaw_error": np.asarray(servo_eval["post_yaw"][i], dtype=np.float32),
                "teacher_improves_xy": np.asarray(servo_eval["improves_xy"][i], dtype=np.float32),
                "teacher_improves_z": np.asarray(servo_eval["improves_z"][i], dtype=np.float32),
                "teacher_improves_yaw": np.asarray(servo_eval["improves_yaw"][i], dtype=np.float32),
                "teacher_all_improves": np.asarray(servo_eval["all_improves"][i], dtype=np.float32),
                "teacher_overshoot_xy": np.asarray(float(servo_eval["overshoot_any"][i] > 0.0), dtype=np.float32),
                "teacher_overshoot_z": np.asarray(float(servo_eval["overshoot_any"][i] > 0.0), dtype=np.float32),
                "teacher_overshoot_yaw": np.asarray(float(servo_eval["overshoot_any"][i] > 0.0), dtype=np.float32),
                "teacher_overshoot_any": np.asarray(servo_eval["overshoot_any"][i], dtype=np.float32),
                "teacher_noop_selected": np.asarray(float(np.linalg.norm(servo_residuals[i, :3]) <= 1e-8 and abs(servo_residuals[i, 5]) <= 1e-8), dtype=np.float32),
                "teacher_action_pos_norm": np.asarray(servo_eval["action_pos_norm"][i], dtype=np.float32),
                "teacher_action_yaw_abs": np.asarray(servo_eval["action_yaw_abs"][i], dtype=np.float32),
                "teacher_best_candidate_index": np.asarray(-1, dtype=np.int64),
                "teacher_objective_stage": np.asarray(0 if selected_stage_bucket[i] == "near_alignment" else 1, dtype=np.int64),
                "teacher_objective_primary": np.asarray(servo_eval["post_z"][i] if selected_stage_bucket[i] == "near_alignment" else (servo_eval["post_xy"][i] + servo_eval["post_z"][i] + 1.5 * servo_eval["post_yaw"][i]), dtype=np.float32),
                "teacher_objective_secondary": np.asarray(servo_eval["post_xy"][i] if selected_stage_bucket[i] == "near_alignment" else servo_eval["overshoot_any"][i], dtype=np.float32),
                "teacher_objective_tertiary": np.asarray(servo_eval["post_yaw"][i] if selected_stage_bucket[i] == "near_alignment" else servo_eval["action_pos_norm"][i], dtype=np.float32),
                "teacher_objective_quaternary": np.asarray(servo_eval["action_pos_norm"][i] if selected_stage_bucket[i] == "near_alignment" else servo_eval["action_yaw_abs"][i], dtype=np.float32),
                "teacher_objective_quinary": np.asarray(servo_eval["overshoot_any"][i] if selected_stage_bucket[i] == "near_alignment" else float(np.linalg.norm(servo_residuals[i, :3]) > 1e-8 or abs(servo_residuals[i, 5]) > 1e-8), dtype=np.float32),
                "source_row_index": np.asarray(src_i, dtype=np.int64),
                "stage_bucket": np.asarray(selected_stage_bucket[i]),
                "source_stage_bucket": np.asarray("micro_source" if selected_stage_bucket[i] == "micro_contact_refine" else "near_source"),
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
        "candidate_actions_local",
        "candidate_mask",
        "candidate_kind",
        "candidate_tier",
        "candidate_oracle_score",
        "candidate_improvement",
        "candidate_next_basin_distance",
        "candidate_basin_positive",
        "best_candidate_index",
        "oracle_candidate_index",
        "runtime_selected_candidate_index",
        "pred_candidate_index",
        "topk_candidate_index",
        "topk_candidate_prob",
        "candidate_scores",
        "candidate_probs",
        "invalid_after_trigger",
        "current_pose_7d",
        "motion_target_pose_7d",
        "basin_center_pose_7d",
        "pregrasp_target_pose_7d",
        "grasp_commit_target_pose_7d",
        "target_delta_teacher",
        "teacher_current_delta_basin_target",
        "proxy_current_delta_basin_target",
        "current_delta_basin_target",
        "motion_target_delta_local",
        "planner_base_action_local_raw",
        "executed_action_local",
        "oracle_action_local",
        "base_action",
        "episode_index",
    )
    for key in pass_keys:
        if key in src.files:
            out[key] = np.asarray(src[key])[indices]

    if "planner_base_action_local_raw" in out:
        out["planner_base_action_local"] = np.asarray(out["planner_base_action_local_raw"], dtype=np.float32)
    elif "base_action" in out:
        out["planner_base_action_local"] = np.asarray(out["base_action"], dtype=np.float32)
    elif "executed_action_local" in out:
        out["planner_base_action_local"] = np.asarray(out["executed_action_local"], dtype=np.float32)
    else:
        out["planner_base_action_local"] = np.zeros((indices.size, 6), dtype=np.float32)

    # force_history is absent in the support shard; provide a zero-filled
    # placeholder so the current trainer can still consume this dataset.
    out["force_history"] = np.zeros((indices.size, 32, 6), dtype=np.float32)

    out["current_to_target_delta_local"] = source_current_delta.astype(np.float32)
    out["current_xy_error"] = np.linalg.norm(source_current_delta[:, :2], axis=-1).astype(np.float32)
    out["current_z_error"] = np.abs(source_current_delta[:, 2]).astype(np.float32)
    out["current_yaw_error"] = np.abs(source_current_delta[:, 5]).astype(np.float32)

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
    out["teacher_objective_stage"] = _stack("teacher_objective_stage", np.int64)
    out["teacher_objective_primary"] = _stack("teacher_objective_primary", np.float32)
    out["teacher_objective_secondary"] = _stack("teacher_objective_secondary", np.float32)
    out["teacher_objective_tertiary"] = _stack("teacher_objective_tertiary", np.float32)
    out["teacher_objective_quaternary"] = _stack("teacher_objective_quaternary", np.float32)
    out["teacher_objective_quinary"] = _stack("teacher_objective_quinary", np.float32)
    out["teacher_source_row_index"] = _stack("source_row_index", np.int64)
    out["row_index"] = out["teacher_source_row_index"].astype(np.int64)
    out["teacher_source"] = np.asarray(["privileged_servo_pseudo_v1"] * indices.size)
    out["stage_bucket"] = np.asarray([r["stage_bucket"] for r in teacher_rows])
    out["source_stage_bucket"] = np.asarray([r["source_stage_bucket"] for r in teacher_rows])

    # Compatibility aliases for the current trainer.
    out["target_residual_local_4d"] = out["teacher_residual_local_4d"].astype(np.float32)
    out["target_residual_local_6d"] = out["teacher_residual_local_6d"].astype(np.float32)
    out["target_post_xy_error"] = out["teacher_post_xy_error"].astype(np.float32)
    out["target_post_z_error"] = out["teacher_post_z_error"].astype(np.float32)
    out["target_post_yaw_error"] = out["teacher_post_yaw_error"].astype(np.float32)
    out["target_improves_xy"] = out["teacher_improves_xy"].astype(np.float32)
    out["target_improves_z"] = out["teacher_improves_z"].astype(np.float32)
    out["target_improves_yaw"] = out["teacher_improves_yaw"].astype(np.float32)
    out["overshoot_proxy"] = out["teacher_overshoot_any"].astype(np.float32)

    # Risk proxy: use a 90th-percentile threshold on selected teacher action size
    # so the proxy is non-degenerate.
    pos_norm = np.linalg.norm(out["teacher_residual_local_6d"][:, :3], axis=-1)
    yaw_abs = np.abs(out["teacher_residual_local_6d"][:, 5])
    pos_hi = float(np.percentile(pos_norm, 90))
    yaw_hi = float(np.percentile(yaw_abs, 90))
    out["teacher_workspace_violation"] = ((pos_norm > pos_hi) | (yaw_abs > yaw_hi)).astype(np.float32)
    out["teacher_invalid"] = ((out["teacher_overshoot_any"] > 0.5) | (out["teacher_workspace_violation"] > 0.5)).astype(np.float32)
    out["invalid_risk_proxy"] = out["teacher_invalid"].astype(np.float32)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    report = {
        "audit": "alignment_v3_privileged_direct_teacher_build",
        "source_npz": str(args.source_npz),
        "output_npz": str(args.output_npz),
        "rows_total": int(src["current_pose_7d"].shape[0]),
        "rows_selected": int(indices.size),
        "selection": {
            "keep_source_xy_norm": float(args.keep_source_xy_norm),
            "keep_source_z_abs": float(args.keep_source_z_abs),
            "keep_source_yaw_abs": float(args.keep_source_yaw_abs),
            "micro_source_xy_norm": float(args.micro_source_xy_norm),
            "micro_source_z_abs": float(args.micro_source_z_abs),
            "micro_source_yaw_abs": float(args.micro_source_yaw_abs),
            "selected_bucket_histogram": dict(Counter(out["stage_bucket"].tolist())),
            "source_residual_xy_norm_selected": _stats(src_res_xy[source_keep]),
            "source_residual_z_abs_selected": _stats(src_res_z[source_keep]),
            "source_residual_yaw_abs_selected": _stats(src_res_yaw[source_keep]),
            "source_residual_xy_norm_all": _stats(src_res_xy),
            "source_residual_z_abs_all": _stats(src_res_z),
            "source_residual_yaw_abs_all": _stats(src_res_yaw),
        },
        "teacher": {
            "residual_pos_norm": _stats(pos_norm),
            "residual_yaw_abs": _stats(yaw_abs),
            "post_xy_error": _stats(out["teacher_post_xy_error"]),
            "post_z_error": _stats(out["teacher_post_z_error"]),
            "post_yaw_error": _stats(out["teacher_post_yaw_error"]),
            "improves_xy_rate": float(np.asarray(out["teacher_improves_xy"], dtype=np.float32).mean()),
            "improves_z_rate": float(np.asarray(out["teacher_improves_z"], dtype=np.float32).mean()),
            "improves_yaw_rate": float(np.asarray(out["teacher_improves_yaw"], dtype=np.float32).mean()),
            "all_improves_rate": float(np.asarray(out["teacher_all_improves"], dtype=np.float32).mean()),
            "noop_selected_rate": float(np.asarray(out["teacher_noop_selected"], dtype=np.float32).mean()),
            "overshoot_any_rate": float(np.asarray(out["teacher_overshoot_any"], dtype=np.float32).mean()),
            "invalid_rate": float(np.asarray(out["teacher_invalid"], dtype=np.float32).mean()),
            "workspace_violation_rate": float(np.asarray(out["teacher_workspace_violation"], dtype=np.float32).mean()),
            "action_pos_norm_mean": float(pos_norm.mean()) if pos_norm.size else 0.0,
            "action_yaw_abs_mean": float(yaw_abs.mean()) if yaw_abs.size else 0.0,
            "saturation_rate_dx": float((np.abs(out["teacher_residual_local_6d"][:, 0]) >= np.max(np.abs(_parse_floats(args.dx_values))) - 1e-8).mean()),
            "saturation_rate_dy": float((np.abs(out["teacher_residual_local_6d"][:, 1]) >= np.max(np.abs(_parse_floats(args.dy_values))) - 1e-8).mean()),
            "saturation_rate_dz": float((np.abs(out["teacher_residual_local_6d"][:, 2]) >= np.max(np.abs(_parse_floats(args.dz_values))) - 1e-8).mean()),
            "saturation_rate_dyaw": float((np.abs(out["teacher_residual_local_6d"][:, 5]) >= np.max(np.abs(_parse_floats(args.dyaw_values))) - 1e-8).mean()),
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
