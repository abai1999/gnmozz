#!/usr/bin/env python3
"""Build a fully support-rebuilt contract-matched v3 dataset.

This builder does not rely on the previous v3 teacher shard as the source of
truth. Instead, it starts from the original support shard and reconstructs:
  - the near/micro stage selection
  - the privileged direct-control teacher residual via dense local search
  - the runtime learned target-delta predictor contract using the real support
    auxiliary context

The goal is to test whether v3 improves when both the teacher and the runtime
input contract are rebuilt from the original support states.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from build_pose_candidate_dataset import apply_local_offset_to_pose, pose_delta_local_between
from prismatic.models.target_delta_predictor import TargetDeltaPredictor


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


def _is_legacy_output_head(state_dict: dict[str, torch.Tensor]) -> bool:
    return "delta_head.weight" not in state_dict and "head.4.weight" in state_dict


def _parse_floats(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty float list: {text!r}")
    return np.asarray(vals, dtype=np.float32)


def _generate_teacher_grid(dx_values: np.ndarray, dy_values: np.ndarray, dz_values: np.ndarray, dyaw_values: np.ndarray) -> np.ndarray:
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
        keys = (nonzero, overshoot_i, action_pos_norm, post_yaw, post_xy, post_z, safe)

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


def _predict_raw_delta(
    model: TargetDeltaPredictor,
    *,
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    wrist_depth: np.ndarray,
    proprio: np.ndarray,
    gripper_context: np.ndarray,
    has_object_in_hand: np.ndarray,
    substage_id: np.ndarray,
    contact_state: np.ndarray,
    stage_target_mode: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n = int(front_rgb.shape[0])
    preds: list[np.ndarray] = []

    def _prep_rgb(arr: np.ndarray) -> torch.Tensor:
        ten = torch.from_numpy(np.asarray(arr)).to(device=device, dtype=torch.float32)
        if ten.ndim == 3 and ten.shape[-1] == 3:
            ten = ten.permute(2, 0, 1)
        elif ten.ndim == 4 and ten.shape[-1] == 3:
            ten = ten.permute(0, 3, 1, 2)
        if ten.ndim == 3:
            ten = ten.unsqueeze(0)
        if float(ten.max().item()) > 1.5:
            ten = ten / 255.0
        return ten.contiguous()

    def _prep_depth(arr: np.ndarray) -> torch.Tensor:
        ten = torch.from_numpy(np.asarray(arr)).to(device=device, dtype=torch.float32)
        if ten.ndim == 2:
            ten = ten.unsqueeze(0).unsqueeze(0)
        elif ten.ndim == 3:
            if ten.shape[0] == 1:
                ten = ten.unsqueeze(0)
            elif ten.shape[-1] == 1:
                ten = ten.permute(2, 0, 1).unsqueeze(0)
            elif ten.shape[-1] != 96:
                ten = ten.unsqueeze(1)
        elif ten.ndim == 4 and ten.shape[-1] == 1:
            ten = ten.permute(0, 3, 1, 2)
        ten = torch.clamp(ten, 0.0, 1.0)
        return ten.contiguous()

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_out = model(
                front_rgb=_prep_rgb(front_rgb[start:end]),
                wrist_rgb=_prep_rgb(wrist_rgb[start:end]),
                wrist_depth=_prep_depth(wrist_depth[start:end]),
                proprio=torch.from_numpy(proprio[start:end]).to(device=device, dtype=torch.float32),
                gripper_context=torch.from_numpy(gripper_context[start:end]).to(device=device, dtype=torch.float32),
                has_object_in_hand=torch.from_numpy(has_object_in_hand[start:end]).to(device=device, dtype=torch.float32),
                substage_id=torch.from_numpy(substage_id[start:end]).to(device=device, dtype=torch.long),
                contact_state=torch.from_numpy(contact_state[start:end]).to(device=device, dtype=torch.long),
                stage_target_mode=torch.from_numpy(stage_target_mode[start:end]).to(device=device, dtype=torch.long),
                return_aux=False,
            )
            pred = batch_out.detach().float().cpu().numpy()
            if pred.ndim == 1:
                pred = pred[None, :]
            preds.append(pred.astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 6), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", type=Path, required=True)
    ap.add_argument("--predictor_ckpt", type=Path, default=Path("runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt"))
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
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    support = np.load(args.support_npz, allow_pickle=True)
    required_support = [
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "gripper_context",
        "has_object_in_hand",
        "substage_id",
        "contact_state",
        "stage_target_mode",
        "current_pose_7d",
        "motion_target_pose_7d",
        "planner_base_action_local_raw",
        "residual_label_local",
    ]
    for key in required_support:
        if key not in support.files:
            raise SystemExit(f"support npz missing required field: {key}")

    current_pose = np.asarray(support["current_pose_7d"], dtype=np.float32)
    target_pose = np.asarray(support["motion_target_pose_7d"], dtype=np.float32)
    current_delta = np.stack([pose_delta_local_between(c, t) for c, t in zip(current_pose, target_pose)], axis=0).astype(np.float32)
    residual_label = np.asarray(support["residual_label_local"], dtype=np.float32)
    cur_xy = np.linalg.norm(residual_label[:, :2], axis=-1)
    cur_z = np.abs(residual_label[:, 2])
    cur_yaw = np.abs(residual_label[:, 5])

    source_keep = (
        (cur_xy <= float(args.keep_source_xy_norm))
        & (cur_z <= float(args.keep_source_z_abs))
        & (cur_yaw <= float(args.keep_source_yaw_abs))
    )
    if not np.any(source_keep):
        raise SystemExit("No rows passed source near/micro filter.")
    source_micro = (
        source_keep
        & (cur_xy <= float(args.micro_source_xy_norm))
        & (cur_z <= float(args.micro_source_z_abs))
        & (cur_yaw <= float(args.micro_source_yaw_abs))
    )

    indices = np.where(source_keep)[0]
    selected_stage_bucket = np.where(source_micro[indices], "micro_contact_refine", "near_alignment")

    dx_values = _parse_floats(args.dx_values)
    dy_values = _parse_floats(args.dy_values)
    dz_values = _parse_floats(args.dz_values)
    dyaw_values = _parse_floats(args.dyaw_values)
    candidate_bank = _generate_teacher_grid(dx_values, dy_values, dz_values, dyaw_values)

    teacher_rows: list[dict[str, np.ndarray | float | int | str]] = []
    for i, src_i in enumerate(indices.tolist()):
        teacher_rows.append(
            {
                **_solve_teacher_for_row(
                    current_pose=current_pose[src_i],
                    target_pose=target_pose[src_i],
                    current_delta=np.asarray(support["target_delta_teacher"], dtype=np.float32)[src_i],
                    candidate_bank=candidate_bank,
                    stage_bucket=str(selected_stage_bucket[i]),
                ),
                "source_row_index": np.asarray(src_i, dtype=np.int64),
                "stage_bucket": np.asarray(selected_stage_bucket[i]),
                "source_stage_bucket": np.asarray("micro_source" if selected_stage_bucket[i] == "micro_contact_refine" else "near_source"),
            }
        )

    def _stack(key: str, dtype=None):
        arr = np.stack([np.asarray(r[key]) for r in teacher_rows], axis=0)
        return arr.astype(dtype) if dtype is not None else arr

    ckpt = torch.load(args.predictor_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    predictor = TargetDeltaPredictor(legacy_output_head=_is_legacy_output_head(state_dict))
    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = predictor.to(device)

    support_rows = indices
    support_gripper_context = np.asarray(support["gripper_context"], dtype=np.float32)[support_rows]
    support_has_object = np.asarray(support["has_object_in_hand"], dtype=np.float32)[support_rows]
    support_substage = np.asarray(support["substage_id"], dtype=np.int64)[support_rows]
    support_contact = np.asarray(support["contact_state"], dtype=np.int64)[support_rows]
    support_stage_target_mode = np.asarray(support["stage_target_mode"], dtype=np.int64)[support_rows]
    support_depth_proximity = np.asarray(support["depth_proximity"], dtype=np.float32)[support_rows]
    support_planner_close_intent = (
        np.asarray(support["planner_close_intent"], dtype=np.float32)[support_rows]
        if "planner_close_intent" in support.files
        else np.full((support_rows.shape[0],), np.nan, dtype=np.float32)
    )

    raw_delta = _predict_raw_delta(
        predictor,
        front_rgb=np.asarray(support["front_rgb"], dtype=np.uint8)[support_rows],
        wrist_rgb=np.asarray(support["wrist_rgb"], dtype=np.uint8)[support_rows],
        wrist_depth=np.asarray(support["wrist_depth"], dtype=np.float32)[support_rows],
        proprio=np.asarray(support["proprio"], dtype=np.float32)[support_rows],
        gripper_context=support_gripper_context,
        has_object_in_hand=support_has_object,
        substage_id=support_substage,
        contact_state=support_contact,
        stage_target_mode=support_stage_target_mode,
        batch_size=int(args.batch_size),
        device=device,
    )
    zero_delta = _predict_raw_delta(
        predictor,
        front_rgb=np.asarray(support["front_rgb"], dtype=np.uint8)[support_rows],
        wrist_rgb=np.asarray(support["wrist_rgb"], dtype=np.uint8)[support_rows],
        wrist_depth=np.asarray(support["wrist_depth"], dtype=np.float32)[support_rows],
        proprio=np.asarray(support["proprio"], dtype=np.float32)[support_rows],
        gripper_context=np.zeros_like(support_gripper_context, dtype=np.float32),
        has_object_in_hand=np.zeros_like(support_has_object, dtype=np.float32),
        substage_id=np.zeros_like(support_substage, dtype=np.int64),
        contact_state=np.zeros_like(support_contact, dtype=np.int64),
        stage_target_mode=np.zeros_like(support_stage_target_mode, dtype=np.int64),
        batch_size=int(args.batch_size),
        device=device,
    )

    if raw_delta.shape[0] != int(indices.size):
        raise SystemExit(f"predictor row mismatch: raw_delta={raw_delta.shape[0]} vs selected={indices.size}")

    teacher_current_delta = np.asarray(support["target_delta_teacher"], dtype=np.float32)[indices]
    teacher_residual_4d = _stack("teacher_residual_local_4d", np.float32)
    teacher_residual_6d = _stack("teacher_residual_local_6d", np.float32)
    teacher_post_xy = _stack("teacher_post_xy_error", np.float32)
    teacher_post_z = _stack("teacher_post_z_error", np.float32)
    teacher_post_yaw = _stack("teacher_post_yaw_error", np.float32)

    out: dict[str, np.ndarray] = {}
    for key in [
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "gripper_context",
        "has_object_in_hand",
        "substage_id",
        "contact_state",
        "stage_target_mode",
        "depth_proximity",
        "planner_base_action_local_raw",
        "executed_action_local",
        "episode_index",
    ]:
        if key in support.files:
            out[key] = np.asarray(support[key])[indices]

    out["planner_base_action_local"] = np.asarray(out.get("planner_base_action_local_raw", out["executed_action_local"]), dtype=np.float32)
    out["force_history"] = np.zeros((indices.size, 32, 6), dtype=np.float32)
    out["current_pose_7d"] = current_pose[indices].astype(np.float32)
    out["motion_target_pose_7d"] = target_pose[indices].astype(np.float32)
    out["basin_center_pose_7d"] = np.asarray(support["basin_center_pose_7d"], dtype=np.float32)[indices]
    out["pregrasp_target_pose_7d"] = np.asarray(support["pregrasp_target_pose_7d"], dtype=np.float32)[indices]
    out["grasp_commit_target_pose_7d"] = np.asarray(support["grasp_commit_target_pose_7d"], dtype=np.float32)[indices]
    out["target_delta_teacher"] = np.asarray(support["target_delta_teacher"], dtype=np.float32)[indices]
    out["current_delta_basin_target"] = np.asarray(support["current_delta_basin_target"], dtype=np.float32)[indices]
    out["proxy_current_delta_basin_target"] = np.asarray(support["proxy_current_delta_basin_target"], dtype=np.float32)[indices]
    out["teacher_current_delta_basin_target"] = np.asarray(support["current_delta_basin_target"], dtype=np.float32)[indices]

    out["current_to_target_delta_local"] = raw_delta.astype(np.float32)
    out["runtime_current_to_target_delta_local"] = raw_delta.astype(np.float32)
    out["teacher_current_to_target_delta_local"] = teacher_current_delta.astype(np.float32)
    out["raw_learned_predictor_delta_local"] = raw_delta.astype(np.float32)
    out["raw_learned_predictor_delta_zero_context_local"] = zero_delta.astype(np.float32)
    out["runtime_target_delta_source"] = np.asarray(["learned_target_predictor"] * indices.size)
    out["runtime_target_delta_context_mode"] = np.asarray(["support_context_recovered"] * indices.size)

    out["teacher_residual_local_4d"] = teacher_residual_4d.astype(np.float32)
    out["teacher_residual_local_6d"] = teacher_residual_6d.astype(np.float32)
    out["target_residual_local_4d"] = teacher_residual_4d.astype(np.float32)
    out["target_residual_local_6d"] = teacher_residual_6d.astype(np.float32)
    out["teacher_post_xy_error"] = teacher_post_xy.astype(np.float32)
    out["teacher_post_z_error"] = teacher_post_z.astype(np.float32)
    out["teacher_post_yaw_error"] = teacher_post_yaw.astype(np.float32)
    out["target_post_xy_error"] = teacher_post_xy.astype(np.float32)
    out["target_post_z_error"] = teacher_post_z.astype(np.float32)
    out["target_post_yaw_error"] = teacher_post_yaw.astype(np.float32)
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
    out["teacher_workspace_violation"] = ((np.linalg.norm(teacher_residual_6d[:, :3], axis=-1) > 0.01) | (np.abs(teacher_residual_6d[:, 5]) > 0.004)).astype(np.float32)
    out["teacher_invalid"] = ((out["teacher_overshoot_any"] > 0.5) | (out["teacher_workspace_violation"] > 0.5)).astype(np.float32)
    out["invalid_risk_proxy"] = out["teacher_invalid"].astype(np.float32)
    out["overshoot_proxy"] = out["teacher_overshoot_any"].astype(np.float32)
    out["target_improves_xy"] = out["teacher_improves_xy"].astype(np.float32)
    out["target_improves_z"] = out["teacher_improves_z"].astype(np.float32)
    out["target_improves_yaw"] = out["teacher_improves_yaw"].astype(np.float32)
    out["teacher_source"] = np.asarray(["privileged_direct_rollout_v2_support_rebuilt"] * indices.size)
    out["teacher_source_row_index"] = indices.astype(np.int64)
    out["row_index"] = indices.astype(np.int64)
    out["stage_bucket"] = np.asarray([r["stage_bucket"] for r in teacher_rows])
    out["source_stage_bucket"] = np.asarray([r["source_stage_bucket"] for r in teacher_rows])

    # Preserve the recovered runtime context for audit and future analysis.
    out["planner_close_intent"] = support_planner_close_intent.astype(np.float32)

    # Useful provenance.
    out["support_source_npz"] = np.asarray(str(args.support_npz))
    out["predictor_ckpt"] = np.asarray(str(args.predictor_ckpt))

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    raw_pos_norm = np.linalg.norm(raw_delta[:, :3], axis=-1)
    raw_yaw_abs = np.abs(raw_delta[:, 5])
    zero_pos_norm = np.linalg.norm(zero_delta[:, :3], axis=-1)
    zero_yaw_abs = np.abs(zero_delta[:, 5])
    teacher_pos_norm = np.linalg.norm(teacher_residual_4d[:, :3], axis=-1)
    teacher_yaw_abs = np.abs(teacher_residual_4d[:, 3])

    report = {
        "audit": "alignment_v3_support_rebuilt_contract_build",
        "support_npz": str(args.support_npz),
        "predictor_ckpt": str(args.predictor_ckpt),
        "output_npz": str(args.output_npz),
        "rows_total": int(current_pose.shape[0]),
        "rows_selected": int(indices.size),
        "selection": {
            "keep_source_xy_norm": float(args.keep_source_xy_norm),
            "keep_source_z_abs": float(args.keep_source_z_abs),
            "keep_source_yaw_abs": float(args.keep_source_yaw_abs),
            "micro_source_xy_norm": float(args.micro_source_xy_norm),
            "micro_source_z_abs": float(args.micro_source_z_abs),
            "micro_source_yaw_abs": float(args.micro_source_yaw_abs),
            "selected_bucket_histogram": dict(Counter(out["stage_bucket"].tolist())),
            "source_residual_xy_norm_selected": _stats(cur_xy[source_keep]),
            "source_residual_z_abs_selected": _stats(cur_z[source_keep]),
            "source_residual_yaw_abs_selected": _stats(cur_yaw[source_keep]),
            "source_residual_xy_norm_all": _stats(cur_xy),
            "source_residual_z_abs_all": _stats(cur_z),
            "source_residual_yaw_abs_all": _stats(cur_yaw),
        },
        "predictor_load": {
            "legacy_output_head": bool(_is_legacy_output_head(state_dict)),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
        "runtime_context": {
            "gripper_context_norm": _stats(np.linalg.norm(support_gripper_context, axis=-1)),
            "has_object_rate": float(np.mean(support_has_object)),
            "substage_hist": {str(k): int(v) for k, v in Counter(support_substage.tolist()).items()},
            "contact_state_hist": {str(k): int(v) for k, v in Counter(support_contact.tolist()).items()},
            "stage_target_mode_hist": {str(k): int(v) for k, v in Counter(support_stage_target_mode.tolist()).items()},
            "depth_proximity": _stats(support_depth_proximity),
            "planner_close_intent_rate": float(np.nanmean(support_planner_close_intent)),
        },
        "input_contract": {
            "raw_learned_predictor_pos_norm": _stats(raw_pos_norm),
            "raw_learned_predictor_yaw_abs": _stats(raw_yaw_abs),
            "zero_context_pos_norm": _stats(zero_pos_norm),
            "zero_context_yaw_abs": _stats(zero_yaw_abs),
            "teacher_current_pos_norm": _stats(np.linalg.norm(teacher_current_delta[:, :3], axis=-1)),
            "teacher_current_yaw_abs": _stats(np.abs(teacher_current_delta[:, 5])),
            "teacher_residual_pos_norm": _stats(teacher_pos_norm),
            "teacher_residual_yaw_abs": _stats(teacher_yaw_abs),
            "mae_raw_vs_teacher_current_xyzrpy": [float(x) for x in np.mean(np.abs(raw_delta - teacher_current_delta), axis=0).tolist()],
            "mae_zero_vs_teacher_current_xyzrpy": [float(x) for x in np.mean(np.abs(zero_delta - teacher_current_delta), axis=0).tolist()],
            "mae_raw_vs_teacher_residual_xyzrpy": [float(x) for x in np.mean(np.abs(raw_delta - teacher_residual_6d), axis=0).tolist()],
            "mae_zero_vs_teacher_residual_xyzrpy": [float(x) for x in np.mean(np.abs(zero_delta - teacher_residual_6d), axis=0).tolist()],
        },
        "teacher": {
            "post_xy": _stats(teacher_post_xy),
            "post_z": _stats(teacher_post_z),
            "post_yaw": _stats(teacher_post_yaw),
            "improves_xy_rate": float(np.asarray(out["teacher_improves_xy"], dtype=np.float32).mean()),
            "improves_z_rate": float(np.asarray(out["teacher_improves_z"], dtype=np.float32).mean()),
            "improves_yaw_rate": float(np.asarray(out["teacher_improves_yaw"], dtype=np.float32).mean()),
            "all_improves_rate": float(np.asarray(out["teacher_all_improves"], dtype=np.float32).mean()),
            "noop_selected_rate": float(np.asarray(out["teacher_noop_selected"], dtype=np.float32).mean()),
            "overshoot_any_rate": float(np.asarray(out["teacher_overshoot_any"], dtype=np.float32).mean()),
            "invalid_rate": float(np.asarray(out["teacher_invalid"], dtype=np.float32).mean()),
            "workspace_violation_rate": float(np.asarray(out["teacher_workspace_violation"], dtype=np.float32).mean()),
            "action_pos_norm_mean": float(teacher_pos_norm.mean()) if teacher_pos_norm.size else 0.0,
            "action_yaw_abs_mean": float(teacher_yaw_abs.mean()) if teacher_yaw_abs.size else 0.0,
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
