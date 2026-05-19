#!/usr/bin/env python3
"""Build a contract-matched v3 dataset with recovered predictor context.

This version reuses the privileged direct-control teacher residual labels from
the v3 teacher shard, but reconstructs the runtime predictor input contract from
the original support shard so TargetDeltaPredictor sees its auxiliary context
instead of the previous all-zero fallback.
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
            pred = model(
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
            pred = pred.detach().float().cpu().numpy()
            if pred.ndim == 1:
                pred = pred[None, :]
            preds.append(pred.astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 6), dtype=np.float32)


def _safe_row_lookup(data: np.lib.npyio.NpzFile, key: str, rows: np.ndarray, *, default=None):
    if key not in data.files:
        if default is None:
            raise KeyError(key)
        arr = np.asarray(default)
        if arr.ndim == 0:
            arr = np.full((rows.shape[0],), arr, dtype=np.asarray(default).dtype if np.ndim(default) else np.float32)
        return arr
    arr = np.asarray(data[key])
    return arr[rows]


def _optional_support_field(
    data: np.lib.npyio.NpzFile,
    key: str,
    rows: np.ndarray,
    *,
    fill_value,
    dtype,
):
    if key in data.files:
        return np.asarray(data[key], dtype=dtype)[rows]
    return np.full((rows.shape[0],), fill_value, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_teacher_npz",
        type=Path,
        required=True,
        help="Existing privileged-direct teacher dataset (near/micro only).",
    )
    parser.add_argument(
        "--support_npz",
        type=Path,
        required=True,
        help="Original support shard with runtime-like auxiliary context fields.",
    )
    parser.add_argument(
        "--predictor_ckpt",
        type=Path,
        default=Path("runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt"),
    )
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    teacher = np.load(args.source_teacher_npz, allow_pickle=True)
    support = np.load(args.support_npz, allow_pickle=True)

    required_teacher = [
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "force_history",
        "current_to_target_delta_local",
        "teacher_residual_local_4d",
        "teacher_residual_local_6d",
        "teacher_post_xy_error",
        "teacher_post_z_error",
        "teacher_post_yaw_error",
        "stage_bucket",
        "teacher_source_row_index",
    ]
    for key in required_teacher:
        if key not in teacher.files:
            raise SystemExit(f"teacher npz missing required field: {key}")

    if "gripper_context" not in support.files:
        raise SystemExit("support npz missing required field: gripper_context")
    if "has_object_in_hand" not in support.files:
        raise SystemExit("support npz missing required field: has_object_in_hand")
    if "substage_id" not in support.files:
        raise SystemExit("support npz missing required field: substage_id")
    if "contact_state" not in support.files:
        raise SystemExit("support npz missing required field: contact_state")
    if "stage_target_mode" not in support.files:
        raise SystemExit("support npz missing required field: stage_target_mode")

    teacher_rows = np.asarray(teacher["teacher_source_row_index"], dtype=np.int64)
    if teacher_rows.size != int(teacher["front_rgb"].shape[0]):
        raise SystemExit("teacher row index count mismatch")

    support_rows = teacher_rows
    support_gripper_context = np.asarray(support["gripper_context"], dtype=np.float32)[support_rows]
    support_has_object = np.asarray(support["has_object_in_hand"], dtype=np.float32)[support_rows]
    support_substage = np.asarray(support["substage_id"], dtype=np.int64)[support_rows]
    support_contact = np.asarray(support["contact_state"], dtype=np.int64)[support_rows]
    support_stage_target_mode = np.asarray(support["stage_target_mode"], dtype=np.int64)[support_rows]

    # Optional support-side fields for auditing / runtime-contract understanding.
    support_depth_proximity = _optional_support_field(
        support, "depth_proximity", support_rows, fill_value=np.nan, dtype=np.float32
    )
    support_planner_close_intent = _optional_support_field(
        support, "planner_close_intent", support_rows, fill_value=np.nan, dtype=np.float32
    )
    support_planner_close_intent_strength = _optional_support_field(
        support, "planner_close_intent_strength", support_rows, fill_value=np.nan, dtype=np.float32
    )

    ckpt = torch.load(args.predictor_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    predictor = TargetDeltaPredictor(legacy_output_head=_is_legacy_output_head(state_dict))
    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = predictor.to(device)

    raw_delta = _predict_raw_delta(
        predictor,
        front_rgb=np.asarray(teacher["front_rgb"], dtype=np.uint8),
        wrist_rgb=np.asarray(teacher["wrist_rgb"], dtype=np.uint8),
        wrist_depth=np.asarray(teacher["wrist_depth"], dtype=np.float32),
        proprio=np.asarray(teacher["proprio"], dtype=np.float32),
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
        front_rgb=np.asarray(teacher["front_rgb"], dtype=np.uint8),
        wrist_rgb=np.asarray(teacher["wrist_rgb"], dtype=np.uint8),
        wrist_depth=np.asarray(teacher["wrist_depth"], dtype=np.float32),
        proprio=np.asarray(teacher["proprio"], dtype=np.float32),
        gripper_context=np.zeros_like(support_gripper_context, dtype=np.float32),
        has_object_in_hand=np.zeros_like(support_has_object, dtype=np.float32),
        substage_id=np.zeros_like(support_substage, dtype=np.int64),
        contact_state=np.zeros_like(support_contact, dtype=np.int64),
        stage_target_mode=np.zeros_like(support_stage_target_mode, dtype=np.int64),
        batch_size=int(args.batch_size),
        device=device,
    )

    if raw_delta.shape[0] != int(teacher["current_to_target_delta_local"].shape[0]):
        raise SystemExit(
            f"predictor row mismatch: raw_delta={raw_delta.shape[0]} vs teacher={teacher['current_to_target_delta_local'].shape[0]}"
        )

    teacher_current_delta = np.asarray(teacher["current_to_target_delta_local"], dtype=np.float32)
    teacher_residual_4d = np.asarray(teacher["teacher_residual_local_4d"], dtype=np.float32)
    teacher_residual_6d = np.asarray(teacher["teacher_residual_local_6d"], dtype=np.float32)
    teacher_post_xy = np.asarray(teacher["teacher_post_xy_error"], dtype=np.float32)
    teacher_post_z = np.asarray(teacher["teacher_post_z_error"], dtype=np.float32)
    teacher_post_yaw = np.asarray(teacher["teacher_post_yaw_error"], dtype=np.float32)

    out = {k: np.asarray(teacher[k]) for k in teacher.files}
    # Preserve teacher current delta separately and overwrite the runtime-facing
    # input contract with the context-recovered predictor output.
    out["teacher_current_to_target_delta_local"] = teacher_current_delta.astype(np.float32)
    out["raw_learned_predictor_delta_local"] = raw_delta.astype(np.float32)
    out["raw_learned_predictor_delta_zero_context_local"] = zero_delta.astype(np.float32)
    out["current_to_target_delta_local"] = raw_delta.astype(np.float32)
    out["runtime_current_to_target_delta_local"] = raw_delta.astype(np.float32)
    out["runtime_target_delta_source"] = np.asarray(["learned_target_predictor"] * raw_delta.shape[0])
    out["runtime_target_delta_context_mode"] = np.asarray(
        ["support_context_recovered"] * raw_delta.shape[0]
    )
    out["teacher_residual_local_4d"] = teacher_residual_4d.astype(np.float32)
    out["teacher_residual_local_6d"] = teacher_residual_6d.astype(np.float32)
    out["target_residual_local_4d"] = teacher_residual_4d.astype(np.float32)
    out["target_residual_local_6d"] = teacher_residual_6d.astype(np.float32)
    out["target_post_xy_error"] = teacher_post_xy.astype(np.float32)
    out["target_post_z_error"] = teacher_post_z.astype(np.float32)
    out["target_post_yaw_error"] = teacher_post_yaw.astype(np.float32)

    # Reattach the context fields so the dataset now carries the actual runtime
    # auxiliary contract instead of default zeros.
    out["gripper_context"] = support_gripper_context.astype(np.float32)
    out["has_object_in_hand"] = support_has_object.astype(np.float32)
    out["substage_id"] = support_substage.astype(np.int64)
    out["contact_state"] = support_contact.astype(np.int64)
    out["stage_target_mode"] = support_stage_target_mode.astype(np.int64)
    out["depth_proximity"] = support_depth_proximity.astype(np.float32)
    out["planner_close_intent"] = support_planner_close_intent.astype(np.float32)
    out["planner_close_intent_strength"] = support_planner_close_intent_strength.astype(np.float32)

    out["contract_matched_source_teacher_npz"] = np.asarray(str(args.source_teacher_npz))
    out["contract_matched_support_npz"] = np.asarray(str(args.support_npz))
    out["contract_matched_predictor_ckpt"] = np.asarray(str(args.predictor_ckpt))

    raw_pos_norm = np.linalg.norm(raw_delta[:, :3], axis=-1)
    raw_yaw_abs = np.abs(raw_delta[:, 5])
    zero_pos_norm = np.linalg.norm(zero_delta[:, :3], axis=-1)
    zero_yaw_abs = np.abs(zero_delta[:, 5])
    teacher_pos_norm = np.linalg.norm(teacher_residual_4d[:, :3], axis=-1)
    teacher_yaw_abs = np.abs(teacher_residual_4d[:, 3])
    mae_raw_vs_teacher_current = np.mean(np.abs(raw_delta - teacher_current_delta), axis=0)
    mae_zero_vs_teacher_current = np.mean(np.abs(zero_delta - teacher_current_delta), axis=0)
    mae_raw_vs_teacher_residual = np.mean(np.abs(raw_delta - teacher_residual_6d), axis=0)
    mae_zero_vs_teacher_residual = np.mean(np.abs(zero_delta - teacher_residual_6d), axis=0)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    report = {
        "audit": "alignment_v3_contract_matched_dataset_context_recovered_build",
        "source_teacher_npz": str(args.source_teacher_npz),
        "support_npz": str(args.support_npz),
        "predictor_ckpt": str(args.predictor_ckpt),
        "output_npz": str(args.output_npz),
        "rows": int(raw_delta.shape[0]),
        "teacher_bucket_hist": dict(Counter(np.asarray(teacher["stage_bucket"], dtype=str).tolist())),
        "predictor_load": {
            "legacy_output_head": bool(_is_legacy_output_head(state_dict)),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
        "runtime_context": {
            "gripper_context": _stats(np.linalg.norm(support_gripper_context, axis=-1)),
            "has_object_rate": float(np.mean(support_has_object)),
            "substage_hist": {str(k): int(v) for k, v in Counter(support_substage.tolist()).items()},
            "contact_state_hist": {str(k): int(v) for k, v in Counter(support_contact.tolist()).items()},
            "stage_target_mode_hist": {str(k): int(v) for k, v in Counter(support_stage_target_mode.tolist()).items()},
            "depth_proximity": _stats(support_depth_proximity),
            "planner_close_intent_rate": float(np.nanmean(support_planner_close_intent)),
            "planner_close_intent_strength": _stats(support_planner_close_intent_strength),
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
            "mae_raw_vs_teacher_current_xyzrpy": [float(x) for x in mae_raw_vs_teacher_current.tolist()],
            "mae_zero_vs_teacher_current_xyzrpy": [float(x) for x in mae_zero_vs_teacher_current.tolist()],
            "mae_raw_vs_teacher_residual_xyzrpy": [float(x) for x in mae_raw_vs_teacher_residual.tolist()],
            "mae_zero_vs_teacher_residual_xyzrpy": [float(x) for x in mae_zero_vs_teacher_residual.tolist()],
        },
        "teacher": {
            "post_xy": _stats(teacher_post_xy),
            "post_z": _stats(teacher_post_z),
            "post_yaw": _stats(teacher_post_yaw),
            "target_improve_xy_rate": float(np.mean(np.asarray(teacher["teacher_improves_xy"], dtype=np.float32))),
            "target_improve_z_rate": float(np.mean(np.asarray(teacher["teacher_improves_z"], dtype=np.float32))),
            "target_improve_yaw_rate": float(np.mean(np.asarray(teacher["teacher_improves_yaw"], dtype=np.float32))),
            "target_all_improves_rate": float(np.mean(np.asarray(teacher["teacher_all_improves"], dtype=np.float32))),
            "invalid_rate": float(np.mean(np.asarray(teacher["teacher_invalid"], dtype=np.float32)))
            if "teacher_invalid" in teacher.files
            else None,
        },
        "note": (
            "current_to_target_delta_local overwritten with raw learned predictor output using recovered support context; "
            "teacher residual preserved separately for supervision/audit; zero-context predictor also logged for contrast"
        ),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
