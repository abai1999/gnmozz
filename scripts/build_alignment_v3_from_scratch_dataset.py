#!/usr/bin/env python3
"""Build a contract-matched v3 dataset from scratch.

This dataset uses:
- runtime input contract: raw learned target predictor delta with real context
- supervision contract: privileged direct-control teacher residual

The teacher must already have passed audit; this script refuses to build a
training dataset from an unvetted teacher.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

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


def _build_has_object_in_hand_array(src: dict[str, np.ndarray], n: int) -> np.ndarray:
    if "has_object_in_hand" in src:
        return np.asarray(src["has_object_in_hand"], dtype=np.float32).reshape(-1)
    arrays: list[np.ndarray] = []
    for key in ("teacher_attached_after_close", "teacher_grasp_verified", "verified_lift"):
        if key in src:
            arrays.append(np.asarray(src[key], dtype=np.float32).reshape(-1))
    out = np.zeros((n,), dtype=np.float32)
    for arr in arrays:
        if arr.shape[0] == n:
            out = np.maximum(out, arr.astype(np.float32))
    return out


def _build_optional_int_context_array(src: dict[str, np.ndarray], key: str, n: int) -> np.ndarray:
    if key in src:
        return np.asarray(src[key], dtype=np.int64).reshape(-1)
    out = np.zeros((n,), dtype=np.int64)
    if key == "substage_id":
        phase = None
        if "teacher_motion_phase" in src:
            phase = np.asarray(src["teacher_motion_phase"], dtype=str).reshape(-1)
        elif "alignment_phase" in src:
            phase = np.asarray(src["alignment_phase"], dtype=str).reshape(-1)
        if phase is not None and phase.shape[0] == n:
            mapping = {
                "": 0,
                "planner_state": 0,
                "planner_state_takeover": 1,
                "align_xy_yaw": 2,
                "enter_finger_region": 3,
                "descend_z": 4,
                "close": 5,
                "settle": 6,
                "lift_verify": 7,
            }
            out = np.asarray([mapping.get(str(x), 0) for x in phase.tolist()], dtype=np.int64)
        return out
    if key == "contact_state":
        close_ready = None
        if "teacher_close_contact_ready" in src:
            close_ready = np.asarray(src["teacher_close_contact_ready"], dtype=np.float32).reshape(-1)
        elif "teacher_grasp_contact_ready" in src:
            close_ready = np.asarray(src["teacher_grasp_contact_ready"], dtype=np.float32).reshape(-1)
        elif "teacher_close_ready" in src:
            close_ready = np.asarray(src["teacher_close_ready"], dtype=np.float32).reshape(-1)
        if close_ready is not None and close_ready.shape[0] == n:
            out = np.where(close_ready > 0.5, 2, 0).astype(np.int64)
        return out
    if key == "stage_target_mode":
        bucket = None
        if "stage_bucket" in src:
            bucket = np.asarray(src["stage_bucket"], dtype=str).reshape(-1)
        elif "alignment_phase" in src:
            bucket = np.asarray(src["alignment_phase"], dtype=str).reshape(-1)
        if bucket is not None and bucket.shape[0] == n:
            mapping = {
                "far": 0,
                "coarse": 0,
                "broad_near": 1,
                "near": 2,
                "near_alignment": 2,
                "near_contact_refine": 3,
                "micro_contact_refine": 3,
                "grasp_commit": 3,
                "planner_state": 0,
                "planner_state_takeover": 2,
            }
            out = np.asarray([mapping.get(str(x), 0) for x in bucket.tolist()], dtype=np.int64)
        return out
    return out


def _build_optional_binary_array(src: dict[str, np.ndarray], keys: tuple[str, ...], n: int) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for key in keys:
        if key in src:
            arrays.append(np.asarray(src[key], dtype=np.float32).reshape(-1))
    out = np.zeros((n,), dtype=np.float32)
    for arr in arrays:
        if arr.shape[0] == n:
            out = np.maximum(out, arr.astype(np.float32))
    return out


def _make_contact_heatmap(delta_local: np.ndarray, size: int = 16) -> np.ndarray:
    delta = np.asarray(delta_local, dtype=np.float32).reshape(-1, 6)
    hm = np.zeros((delta.shape[0], size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    sigma = max(1.5, size / 6.0)
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    for i, row in enumerate(delta):
        # Map local xy residual into a coarse heatmap offset.  The proxy is
        # intentionally simple and bounded so it remains stable under the
        # current dataset contract.
        off_x = float(np.clip(row[0] / 0.02, -1.5, 1.5) * (size / 6.0))
        off_y = float(np.clip(row[1] / 0.02, -1.5, 1.5) * (size / 6.0))
        cx = center + off_x
        cy = center + off_y
        gauss = np.exp(-(((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2.0 * sigma**2)))
        hm[i] = gauss.astype(np.float32)
    return hm


def _prepare_rgb(arr: np.ndarray, device: torch.device) -> torch.Tensor:
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


def _prepare_depth(arr: np.ndarray, device: torch.device) -> torch.Tensor:
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
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            grip = np.asarray(gripper_context[start:end], dtype=np.float32)
            if grip.ndim == 2 and grip.shape[1] > 3:
                grip = grip[:, :3]
            elif grip.ndim == 1 and grip.shape[0] > 3:
                grip = grip[:3]
            out = model(
                front_rgb=_prepare_rgb(front_rgb[start:end], device),
                wrist_rgb=_prepare_rgb(wrist_rgb[start:end], device),
                wrist_depth=_prepare_depth(wrist_depth[start:end], device),
                proprio=torch.as_tensor(proprio[start:end], device=device, dtype=torch.float32),
                gripper_context=torch.as_tensor(grip, device=device, dtype=torch.float32),
                has_object_in_hand=torch.as_tensor(has_object_in_hand[start:end], device=device, dtype=torch.float32),
                substage_id=torch.as_tensor(substage_id[start:end], device=device, dtype=torch.long),
                contact_state=torch.as_tensor(contact_state[start:end], device=device, dtype=torch.long),
                stage_target_mode=torch.as_tensor(stage_target_mode[start:end], device=device, dtype=torch.long),
                return_aux=False,
            )
            if isinstance(out, dict):
                out = out.get("target_delta", out)
            pred = np.asarray(out.detach().float().cpu().numpy(), dtype=np.float32)
            if pred.ndim == 1:
                pred = pred[None, :]
            preds.append(pred)
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 6), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_npz", type=Path, required=True, help="Raw privileged rollout support states npz")
    ap.add_argument("--teacher_npz", type=Path, required=True, help="Audited teacher npz")
    ap.add_argument(
        "--predictor_ckpt",
        type=Path,
        default=Path("runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt"),
    )
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--require_teacher_audit_pass", action="store_true", default=True)
    ap.add_argument("--no_require_teacher_audit_pass", dest="require_teacher_audit_pass", action="store_false")
    args = ap.parse_args()

    src = np.load(args.source_npz, allow_pickle=True)
    src_data = {k: np.asarray(src[k]) for k in src.files}
    tea = np.load(args.teacher_npz, allow_pickle=True)

    if args.require_teacher_audit_pass:
        if "teacher_audit_passed" not in tea.files or float(np.asarray(tea["teacher_audit_passed"]).reshape(())) < 0.5:
            raise SystemExit("teacher audit did not pass; refusing to build dataset")

    required = [
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "force_history",
        "gripper_context",
        "current_pose_7d",
    ]
    for key in required:
        if key not in src_data:
            raise SystemExit(f"source npz missing required field: {key}")

    if "teacher_source_row_index" not in tea.files:
        raise SystemExit("teacher npz missing teacher_source_row_index")

    src_idx = np.asarray(tea["teacher_source_row_index"], dtype=np.int64)
    if src_idx.size == 0:
        raise SystemExit("teacher npz contains no selected rows")

    has_object_in_hand_all = _build_has_object_in_hand_array(src_data, int(np.asarray(src_data["current_pose_7d"]).shape[0]))
    substage_id_all = _build_optional_int_context_array(src_data, "substage_id", int(np.asarray(src_data["current_pose_7d"]).shape[0]))
    contact_state_all = _build_optional_int_context_array(src_data, "contact_state", int(np.asarray(src_data["current_pose_7d"]).shape[0]))
    stage_target_mode_all = _build_optional_int_context_array(src_data, "stage_target_mode", int(np.asarray(src_data["current_pose_7d"]).shape[0]))
    attached_after_close_all = _build_optional_binary_array(
        src_data,
        ("teacher_attached_after_close", "teacher_grasp_verified", "verified_lift"),
        int(np.asarray(src_data["current_pose_7d"]).shape[0]),
    )
    close_ready_all = _build_optional_binary_array(
        src_data,
        ("teacher_close_contact_ready", "teacher_grasp_contact_ready", "teacher_close_ready"),
        int(np.asarray(src_data["current_pose_7d"]).shape[0]),
    )

    ckpt = torch.load(args.predictor_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    predictor = TargetDeltaPredictor(legacy_output_head=_is_legacy_output_head(state_dict))
    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = predictor.to(device)

    raw_delta = _predict_raw_delta(
        predictor,
        front_rgb=np.asarray(src["front_rgb"], dtype=np.uint8)[src_idx],
        wrist_rgb=np.asarray(src["wrist_rgb"], dtype=np.uint8)[src_idx],
        wrist_depth=np.asarray(src["wrist_depth"], dtype=np.float32)[src_idx],
        proprio=np.asarray(src["proprio"], dtype=np.float32)[src_idx],
        gripper_context=np.asarray(src["gripper_context"], dtype=np.float32)[src_idx],
        has_object_in_hand=has_object_in_hand_all[src_idx],
        substage_id=substage_id_all[src_idx],
        contact_state=contact_state_all[src_idx],
        stage_target_mode=stage_target_mode_all[src_idx],
        batch_size=int(args.batch_size),
        device=device,
    )
    if raw_delta.shape[0] != src_idx.shape[0]:
        raise SystemExit(f"predictor row mismatch: raw_delta={raw_delta.shape[0]} vs rows={src_idx.shape[0]}")

    teacher_current_delta = np.asarray(tea["teacher_current_to_target_delta_local"], dtype=np.float32)
    teacher_residual_4d = np.asarray(tea["teacher_residual_local_4d"], dtype=np.float32)
    teacher_residual_6d = np.asarray(tea["teacher_residual_local_6d"], dtype=np.float32)
    teacher_post_xy = np.asarray(tea["teacher_post_xy_error"], dtype=np.float32)
    teacher_post_z = np.asarray(tea["teacher_post_z_error"], dtype=np.float32)
    teacher_post_yaw = np.asarray(tea["teacher_post_yaw_error"], dtype=np.float32)
    teacher_buckets = np.asarray(tea["stage_bucket"], dtype=str)

    out: dict[str, np.ndarray] = {}
    pass_keys = (
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "force_history",
        "force_history_raw",
        "force_history_normalized",
        "ft_hist",
        "gripper_touch_forces",
        "gripper_context",
        "substage_id",
        "contact_state",
        "stage_target_mode",
        "depth_proximity",
        "planner_action_local",
        "planner_base_action_local",
        "planner_base_action_local_raw",
        "planner_base_action_7d_raw",
        "base_action",
        "current_pose_7d",
        "motion_target_pose_7d",
        "privileged_current_pose_7d",
        "privileged_motion_target_pose_7d",
        "privileged_basin_center_pose_7d",
        "privileged_pregrasp_target_pose_7d",
        "privileged_grasp_commit_target_pose_7d",
        "privileged_object_anchor_pose_7d",
        "privileged_current_delta_basin_target",
        "privileged_target_provider_source",
        "privileged_target_provider_uses_privileged",
        "episode_index",
        "step_index",
        "row_index",
        "stage_bucket",
        "teacher_target_source",
        "teacher_target_source_reason",
        "teacher_runtime_delta_contract",
        "teacher_source_row_index",
        "teacher_noop_reason",
        "teacher_noop_reason_detail",
        "teacher_two_step_improve_count_lt2",
        "teacher_two_step_horizon_ge_noop",
        "teacher_two_step_fallback_triggered",
        "teacher_current_to_target_delta_local",
        "teacher_residual_local_4d",
        "teacher_residual_local_6d",
        "teacher_post_xy_error",
        "teacher_post_z_error",
        "teacher_post_yaw_error",
        "teacher_improves_xy",
        "teacher_improves_z",
        "teacher_improves_yaw",
        "teacher_all_improves",
        "teacher_overshoot_xy",
        "teacher_overshoot_z",
        "teacher_overshoot_yaw",
        "teacher_overshoot_any",
        "teacher_noop_selected",
        "teacher_action_pos_norm",
        "teacher_action_yaw_abs",
        "teacher_best_candidate_index",
        "teacher_objective_stage",
        "teacher_objective_primary",
        "teacher_objective_secondary",
        "teacher_objective_tertiary",
        "teacher_objective_quaternary",
        "teacher_objective_quinary",
        "teacher_workspace_violation",
        "teacher_invalid",
        "invalid_risk_proxy",
        "overshoot_proxy",
        "teacher_source",
        "teacher_audit_passed",
        "teacher_audit_fail_reason",
        "teacher_target_pose_7d",
    )
    for key in pass_keys:
        if key in src.files:
            out[key] = np.asarray(src[key])[src_idx]

    if "has_object_in_hand" not in out:
        out["has_object_in_hand"] = has_object_in_hand_all[src_idx].astype(np.float32)
    if "substage_id" not in out:
        out["substage_id"] = substage_id_all[src_idx].astype(np.int64)
    if "contact_state" not in out:
        out["contact_state"] = contact_state_all[src_idx].astype(np.int64)
    if "stage_target_mode" not in out:
        out["stage_target_mode"] = stage_target_mode_all[src_idx].astype(np.int64)
    for key in (
        "teacher_target_source",
        "teacher_target_source_reason",
        "teacher_runtime_delta_contract",
        "teacher_source",
        "teacher_audit_fail_reason",
    ):
        if key in tea.files:
            out[key] = np.asarray(tea[key])

    if "planner_action_local" not in out:
        if "planner_base_action_local" in out:
            out["planner_action_local"] = np.asarray(out["planner_base_action_local"], dtype=np.float32)
        elif "planner_base_action_local_raw" in out:
            out["planner_action_local"] = np.asarray(out["planner_base_action_local_raw"], dtype=np.float32)
        elif "base_action" in out:
            out["planner_action_local"] = np.asarray(out["base_action"], dtype=np.float32)
        else:
            out["planner_action_local"] = np.zeros((src_idx.size, 6), dtype=np.float32)

    if "force_history" not in out:
        out["force_history"] = np.zeros((src_idx.size, 32, 6), dtype=np.float32)

    out["teacher_current_to_target_delta_local"] = teacher_current_delta.astype(np.float32)
    out["teacher_target_delta_local_6d"] = teacher_current_delta.astype(np.float32)
    out["raw_learned_predictor_delta_local"] = raw_delta.astype(np.float32)
    out["current_to_target_delta_local"] = raw_delta.astype(np.float32)
    out["runtime_current_to_target_delta_local"] = raw_delta.astype(np.float32)
    out["runtime_target_delta_source"] = np.asarray(["raw_learned_predictor_from_scratch"] * src_idx.size)
    out["runtime_target_delta_context_mode"] = np.asarray(["raw_rollout_context"] * src_idx.size)
    out["target_residual_local_4d"] = teacher_residual_4d.astype(np.float32)
    out["target_residual_local_6d"] = teacher_residual_6d.astype(np.float32)
    out["best_residual_trajectory_4d"] = np.repeat(teacher_residual_4d[:, None, :], repeats=8, axis=1).astype(np.float32)
    out["teacher_residual_trajectory_4d"] = out["best_residual_trajectory_4d"].astype(np.float32)
    out["target_post_xy_error"] = teacher_post_xy.astype(np.float32)
    out["target_post_z_error"] = teacher_post_z.astype(np.float32)
    out["target_post_yaw_error"] = teacher_post_yaw.astype(np.float32)
    out["target_improves_xy"] = np.asarray(tea["teacher_improves_xy"], dtype=np.float32)
    out["target_improves_z"] = np.asarray(tea["teacher_improves_z"], dtype=np.float32)
    out["target_improves_yaw"] = np.asarray(tea["teacher_improves_yaw"], dtype=np.float32)
    out["overshoot_proxy"] = np.asarray(tea["teacher_overshoot_any"], dtype=np.float32)
    out["invalid_risk_proxy"] = np.asarray(tea["teacher_invalid"], dtype=np.float32)
    out["contact_heatmap_label"] = _make_contact_heatmap(teacher_current_delta, size=16).astype(np.float32)
    out["target_confidence_label"] = np.asarray(
        np.where(
            attached_after_close_all[src_idx] > 0.5,
            1.0,
            np.where(close_ready_all[src_idx] > 0.5, 0.65, 0.25),
        ),
        dtype=np.float32,
    )
    out["progress_label"] = np.stack(
        [
            np.asarray(tea["teacher_improves_xy"], dtype=np.float32),
            np.asarray(tea["teacher_improves_z"], dtype=np.float32),
            np.asarray(tea["teacher_improves_yaw"], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    out["risk_label"] = np.maximum(
        np.asarray(tea["teacher_invalid"], dtype=np.float32),
        np.asarray(tea["teacher_workspace_violation"], dtype=np.float32),
    ).astype(np.float32)[:, None]
    out["stop_label"] = np.asarray(tea["teacher_noop_selected"], dtype=np.float32)[:, None]
    out["sample_weight"] = np.asarray(
        np.where(
            np.asarray(tea["teacher_all_improves"], dtype=np.float32) > 0.5,
            1.5,
            np.where(
                np.asarray(tea["teacher_improves_xy"], dtype=np.float32) + np.asarray(tea["teacher_improves_z"], dtype=np.float32) > 0.5,
                1.0,
                0.75,
            ),
        ),
        dtype=np.float32,
    )

    # Keep privileged teacher fields around for audit / debugging.
    out["teacher_target_pose_7d"] = np.asarray(tea["motion_target_pose_7d"], dtype=np.float32)
    out["teacher_source"] = np.asarray(tea["teacher_source"], dtype=object)
    out["teacher_audit_passed"] = np.asarray(tea["teacher_audit_passed"], dtype=np.float32)
    out["teacher_audit_fail_reason"] = np.asarray(tea["teacher_audit_fail_reason"], dtype=object)
    if "teacher_noop_reason" in tea.files:
        out["teacher_noop_reason"] = np.asarray(tea["teacher_noop_reason"], dtype=object)
    if "teacher_noop_reason_detail" in tea.files:
        out["teacher_noop_reason_detail"] = np.asarray(tea["teacher_noop_reason_detail"], dtype=object)
    for key in (
        "teacher_two_step_improve_count_lt2",
        "teacher_two_step_horizon_ge_noop",
        "teacher_two_step_fallback_triggered",
        "teacher_two_step_horizon_score_margin",
        "teacher_best_improve_count",
        "teacher_best_improve_missing_xy",
        "teacher_best_improve_missing_z",
        "teacher_best_improve_missing_yaw",
        "teacher_very_close_abstain_xy",
        "teacher_very_close_abstain_z",
        "teacher_very_close_abstain_yaw",
        "teacher_very_close_abstain_margin_xy",
        "teacher_very_close_abstain_margin_z",
        "teacher_very_close_abstain_margin_yaw",
        "teacher_very_close_abstain_triggered",
        "teacher_very_close_abstain_xy_threshold",
        "teacher_very_close_abstain_z_threshold",
        "teacher_very_close_abstain_yaw_threshold",
    ):
        if key in tea.files:
            if key == "teacher_best_improve_count":
                out[key] = np.asarray(tea[key], dtype=np.int64)
            else:
                out[key] = np.asarray(tea[key], dtype=np.float32)

    raw_pos_norm = np.linalg.norm(raw_delta[:, :3], axis=-1)
    raw_yaw_abs = np.abs(raw_delta[:, 5])
    teacher_pos_norm = np.linalg.norm(teacher_residual_6d[:, :3], axis=-1)
    teacher_yaw_abs = np.abs(teacher_residual_6d[:, 5])
    input_mae_vs_teacher_current = np.mean(np.abs(raw_delta - teacher_current_delta), axis=0)
    input_mae_vs_teacher_residual = np.mean(np.abs(raw_delta - teacher_residual_6d), axis=0)

    out["current_xy_error"] = np.linalg.norm(teacher_current_delta[:, :2], axis=-1).astype(np.float32)
    out["current_z_error"] = np.abs(teacher_current_delta[:, 2]).astype(np.float32)
    out["current_yaw_error"] = np.abs(teacher_current_delta[:, 5]).astype(np.float32)

    # Preserve runtime contract metadata.
    out["teacher_source_row_index"] = src_idx.astype(np.int64)
    out["row_index"] = src_idx.astype(np.int64)
    out["stage_bucket"] = teacher_buckets

    out["target_residual_local_4d"] = teacher_residual_4d.astype(np.float32)
    out["target_residual_local_6d"] = teacher_residual_6d.astype(np.float32)
    out["target_post_xy_error"] = teacher_post_xy.astype(np.float32)
    out["target_post_z_error"] = teacher_post_z.astype(np.float32)
    out["target_post_yaw_error"] = teacher_post_yaw.astype(np.float32)
    out["target_improves_xy"] = np.asarray(tea["teacher_improves_xy"], dtype=np.float32)
    out["target_improves_z"] = np.asarray(tea["teacher_improves_z"], dtype=np.float32)
    out["target_improves_yaw"] = np.asarray(tea["teacher_improves_yaw"], dtype=np.float32)
    out["teacher_current_to_target_delta_local"] = teacher_current_delta.astype(np.float32)
    out["current_to_target_delta_local"] = raw_delta.astype(np.float32)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    report = {
        "audit": "alignment_v3_from_scratch_dataset_build",
        "source_npz": str(args.source_npz),
        "teacher_npz": str(args.teacher_npz),
        "predictor_ckpt": str(args.predictor_ckpt),
        "output_npz": str(args.output_npz),
        "rows": int(src_idx.size),
        "teacher_audit_passed": bool(float(np.asarray(tea["teacher_audit_passed"]).reshape(())) >= 0.5),
        "bucket_histogram": dict(Counter(teacher_buckets.tolist())),
        "predictor_load": {
            "legacy_output_head": bool(_is_legacy_output_head(state_dict)),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
        "input_contract": {
            "raw_learned_predictor_pos_norm": _stats(raw_pos_norm),
            "raw_learned_predictor_yaw_abs": _stats(raw_yaw_abs),
            "teacher_current_pos_norm": _stats(np.linalg.norm(teacher_current_delta[:, :3], axis=-1)),
            "teacher_current_yaw_abs": _stats(np.abs(teacher_current_delta[:, 5])),
            "teacher_residual_pos_norm": _stats(teacher_pos_norm),
            "teacher_residual_yaw_abs": _stats(teacher_yaw_abs),
            "mae_raw_vs_teacher_current_xyzrpy": [float(x) for x in input_mae_vs_teacher_current.tolist()],
            "mae_raw_vs_teacher_residual_xyzrpy": [float(x) for x in input_mae_vs_teacher_residual.tolist()],
        },
        "teacher": {
            "post_xy": _stats(teacher_post_xy),
            "post_z": _stats(teacher_post_z),
            "post_yaw": _stats(teacher_post_yaw),
            "target_improve_xy_rate": float(np.asarray(tea["teacher_improves_xy"], dtype=np.float32).mean()),
            "target_improve_z_rate": float(np.asarray(tea["teacher_improves_z"], dtype=np.float32).mean()),
            "target_improve_yaw_rate": float(np.asarray(tea["teacher_improves_yaw"], dtype=np.float32).mean()),
            "target_all_improves_rate": float(np.asarray(tea["teacher_all_improves"], dtype=np.float32).mean()),
            "invalid_rate": float(np.asarray(tea["teacher_invalid"], dtype=np.float32).mean()),
            "workspace_violation_rate": float(np.asarray(tea["teacher_workspace_violation"], dtype=np.float32).mean()),
            "noop_reason_histogram": dict(Counter(np.asarray(tea["teacher_noop_reason"], dtype=str).tolist())) if "teacher_noop_reason" in tea.files else {},
            "noop_reason_histogram_by_bucket": {
                bucket: dict(Counter(np.asarray(tea["teacher_noop_reason"], dtype=str)[teacher_buckets == bucket].tolist()))
                for bucket in sorted(set(teacher_buckets.tolist()))
            } if "teacher_noop_reason" in tea.files else {},
            "two_step_improve_count_lt2_rate": float(np.asarray(tea["teacher_two_step_improve_count_lt2"], dtype=np.float32).mean()) if "teacher_two_step_improve_count_lt2" in tea.files else 0.0,
            "two_step_horizon_ge_noop_rate": float(np.asarray(tea["teacher_two_step_horizon_ge_noop"], dtype=np.float32).mean()) if "teacher_two_step_horizon_ge_noop" in tea.files else 0.0,
            "two_step_fallback_triggered_rate": float(np.asarray(tea["teacher_two_step_fallback_triggered"], dtype=np.float32).mean()) if "teacher_two_step_fallback_triggered" in tea.files else 0.0,
            "two_step_horizon_score_margin": _stats(np.asarray(tea["teacher_two_step_horizon_score_margin"], dtype=np.float32)) if "teacher_two_step_horizon_score_margin" in tea.files else {},
            "best_improve_count_histogram": dict(Counter(np.asarray(tea["teacher_best_improve_count"], dtype=np.int64).tolist())) if "teacher_best_improve_count" in tea.files else {},
            "best_improve_missing_axis_histogram": {
                axis: float(np.mean(np.asarray(tea[f"teacher_best_improve_missing_{axis}"], dtype=np.float32)))
                for axis in ("xy", "z", "yaw")
                if f"teacher_best_improve_missing_{axis}" in tea.files
            },
            "very_close_abstain_rate": float(np.asarray(tea["teacher_very_close_abstain_triggered"], dtype=np.float32).mean()) if "teacher_very_close_abstain_triggered" in tea.files else 0.0,
            "very_close_abstain_axis_rate": {
                axis: float(np.asarray(tea[f"teacher_very_close_abstain_{axis}"], dtype=np.float32).mean())
                for axis in ("xy", "z", "yaw")
                if f"teacher_very_close_abstain_{axis}" in tea.files
            },
            "very_close_abstain_axis_margin": {
                axis: _stats(np.asarray(tea[f"teacher_very_close_abstain_margin_{axis}"], dtype=np.float32))
                for axis in ("xy", "z", "yaw")
                if f"teacher_very_close_abstain_margin_{axis}" in tea.files
            },
            "very_close_abstain_thresholds": {
                axis: float(np.asarray(tea[f"teacher_very_close_abstain_{axis}_threshold"], dtype=np.float32).reshape(-1)[0])
                for axis in ("xy", "z", "yaw")
                if f"teacher_very_close_abstain_{axis}_threshold" in tea.files
            },
        },
        "runtime_contract": {
            "source_histogram": dict(Counter(np.asarray(out["runtime_target_delta_source"], dtype=str).tolist())),
            "context_mode_histogram": dict(Counter(np.asarray(out["runtime_target_delta_context_mode"], dtype=str).tolist())),
        },
        "note": (
            "current_to_target_delta_local overwritten with raw learned predictor output; "
            "teacher residual preserved separately for supervision/audit"
        ),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
