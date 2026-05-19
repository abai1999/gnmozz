#!/usr/bin/env python3
"""Build a contract-matched v3 dataset.

This dataset keeps the privileged direct-control teacher residual labels from
the existing near/micro v3 teacher shard, but replaces the student input
contract with the runtime-default raw learned target-delta predictor output.

The goal is to align training and runtime around the same signed target-relative
semantic contract while preserving the stronger privileged supervision signal.
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
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n = int(front_rgb.shape[0])
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            pred = model.predict(
                front_rgb=front_rgb[start:end],
                wrist_rgb=wrist_rgb[start:end],
                wrist_depth=wrist_depth[start:end],
                proprio=proprio[start:end],
            )
            pred = np.asarray(pred, dtype=np.float32)
            if pred.ndim == 1:
                pred = pred[None, :]
            preds.append(pred.astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 6), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_npz", type=Path, required=True, help="Existing privileged direct teacher dataset")
    parser.add_argument(
        "--predictor_ckpt",
        type=Path,
        default=Path("runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt"),
    )
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    src = np.load(args.source_npz, allow_pickle=True)
    required = [
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "force_history",
        "current_to_target_delta_local",
        "teacher_residual_local_4d",
        "target_residual_local_4d",
        "target_residual_local_6d",
        "teacher_post_xy_error",
        "teacher_post_z_error",
        "teacher_post_yaw_error",
        "stage_bucket",
    ]
    for key in required:
        if key not in src.files:
            raise SystemExit(f"source npz missing required field: {key}")

    ckpt = torch.load(args.predictor_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    predictor = TargetDeltaPredictor(legacy_output_head=_is_legacy_output_head(state_dict))
    missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = predictor.to(device)

    raw_delta = _predict_raw_delta(
        predictor,
        front_rgb=np.asarray(src["front_rgb"], dtype=np.uint8),
        wrist_rgb=np.asarray(src["wrist_rgb"], dtype=np.uint8),
        wrist_depth=np.asarray(src["wrist_depth"], dtype=np.float32),
        proprio=np.asarray(src["proprio"], dtype=np.float32),
        batch_size=int(args.batch_size),
        device=device,
    )
    if raw_delta.shape[0] != int(src["current_to_target_delta_local"].shape[0]):
        raise SystemExit(
            f"predictor row mismatch: raw_delta={raw_delta.shape[0]} vs source={src['current_to_target_delta_local'].shape[0]}"
        )

    teacher_current_delta = np.asarray(src["current_to_target_delta_local"], dtype=np.float32)
    teacher_residual_4d = np.asarray(src["teacher_residual_local_4d"], dtype=np.float32)
    teacher_residual_6d = np.asarray(src["teacher_residual_local_6d"], dtype=np.float32)
    teacher_post_xy = np.asarray(src["teacher_post_xy_error"], dtype=np.float32)
    teacher_post_z = np.asarray(src["teacher_post_z_error"], dtype=np.float32)
    teacher_post_yaw = np.asarray(src["teacher_post_yaw_error"], dtype=np.float32)

    out = {k: np.asarray(src[k]) for k in src.files}
    out["teacher_current_to_target_delta_local"] = teacher_current_delta.astype(np.float32)
    out["raw_learned_predictor_delta_local"] = raw_delta.astype(np.float32)
    out["current_to_target_delta_local"] = raw_delta.astype(np.float32)
    out["runtime_current_to_target_delta_local"] = raw_delta.astype(np.float32)
    out["runtime_target_delta_source"] = np.asarray(["learned_target_predictor"] * raw_delta.shape[0])
    out["runtime_target_delta_context_mode"] = np.asarray(["default_zero_context"] * raw_delta.shape[0])
    out["teacher_residual_local_4d"] = teacher_residual_4d.astype(np.float32)
    out["teacher_residual_local_6d"] = teacher_residual_6d.astype(np.float32)
    out["target_residual_local_4d"] = teacher_residual_4d.astype(np.float32)
    out["target_residual_local_6d"] = teacher_residual_6d.astype(np.float32)
    out["target_post_xy_error"] = teacher_post_xy.astype(np.float32)
    out["target_post_z_error"] = teacher_post_z.astype(np.float32)
    out["target_post_yaw_error"] = teacher_post_yaw.astype(np.float32)

    # Keep the teacher fields around for audit; training will ignore them.
    out["contract_matched_source_teacher_npz"] = np.asarray(str(args.source_npz))
    out["contract_matched_predictor_ckpt"] = np.asarray(str(args.predictor_ckpt))

    raw_pos_norm = np.linalg.norm(raw_delta[:, :3], axis=-1)
    raw_yaw_abs = np.abs(raw_delta[:, 5])
    teacher_pos_norm = np.linalg.norm(teacher_residual_4d[:, :3], axis=-1)
    teacher_yaw_abs = np.abs(teacher_residual_4d[:, 3])
    input_mae_vs_teacher_current = np.mean(np.abs(raw_delta - teacher_current_delta), axis=0)
    input_mae_vs_teacher_residual = np.mean(np.abs(raw_delta - teacher_residual_6d), axis=0)

    # Preserve all other arrays unchanged.
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    report = {
        "audit": "alignment_v3_contract_matched_dataset_build",
        "source_npz": str(args.source_npz),
        "predictor_ckpt": str(args.predictor_ckpt),
        "output_npz": str(args.output_npz),
        "rows": int(raw_delta.shape[0]),
        "source_bucket_hist": dict(Counter(np.asarray(src["stage_bucket"], dtype=str).tolist())),
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
            "target_improve_xy_rate": float(np.mean(np.asarray(src["target_improves_xy"], dtype=np.float32))),
            "target_improve_z_rate": float(np.mean(np.asarray(src["target_improves_z"], dtype=np.float32))),
            "target_improve_yaw_rate": float(np.mean(np.asarray(src["target_improves_yaw"], dtype=np.float32))),
            "target_all_improves_rate": float(np.mean(np.asarray(src["teacher_all_improves"], dtype=np.float32))),
            "invalid_rate": float(np.mean(np.asarray(src["teacher_invalid"], dtype=np.float32))) if "teacher_invalid" in src.files else None,
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
