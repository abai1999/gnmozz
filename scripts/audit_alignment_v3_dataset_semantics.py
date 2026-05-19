#!/usr/bin/env python3
"""Audit v3 direct-local dataset semantics against the source v2 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_v2_npz", type=Path, required=True)
    parser.add_argument("--v3_npz", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    args = parser.parse_args()

    src = np.load(args.source_v2_npz, allow_pickle=True)
    v3 = np.load(args.v3_npz, allow_pickle=True)

    mask = np.isin(src["stage_bucket"], ["near_alignment", "micro_contact_refine"])
    indices = np.where(mask)[0]
    row_ids = np.arange(indices.size, dtype=np.int64)
    best_idx = np.asarray(src["best_stage_action_index"], dtype=np.int64)[indices]

    best_residual_6d = np.asarray(src["proposal_actions_local"], dtype=np.float32)[indices][row_ids, best_idx]
    best_post_xy = np.asarray(src["post_xy_error"], dtype=np.float32)[indices][row_ids, best_idx]
    best_post_z = np.asarray(src["post_z_error"], dtype=np.float32)[indices][row_ids, best_idx]
    best_post_yaw = np.asarray(src["post_yaw_error"], dtype=np.float32)[indices][row_ids, best_idx]
    current_xy = np.asarray(src["current_xy_error"], dtype=np.float32)[indices]
    current_z = np.asarray(src["current_z_error"], dtype=np.float32)[indices]
    current_yaw = np.asarray(src["current_yaw_error"], dtype=np.float32)[indices]
    best_overshoot_any = (
        (np.asarray(src["overshoot_xy"], dtype=np.float32)[indices][row_ids, best_idx] > 0)
        | (np.asarray(src["overshoot_z"], dtype=np.float32)[indices][row_ids, best_idx] > 0)
        | (np.asarray(src["overshoot_yaw"], dtype=np.float32)[indices][row_ids, best_idx] > 0)
    ).astype(np.float32)

    v3_residual_6d = np.asarray(v3["target_residual_local_6d"], dtype=np.float32)
    v3_post_xy = np.asarray(v3["target_post_xy_error"], dtype=np.float32)
    v3_post_z = np.asarray(v3["target_post_z_error"], dtype=np.float32)
    v3_post_yaw = np.asarray(v3["target_post_yaw_error"], dtype=np.float32)
    v3_overshoot = np.asarray(v3["overshoot_proxy"], dtype=np.float32)
    v3_invalid = np.asarray(v3["invalid_risk_proxy"], dtype=np.float32)

    best_pos_norm = np.linalg.norm(best_residual_6d[:, :3], axis=-1)
    v3_pos_norm = np.linalg.norm(v3_residual_6d[:, :3], axis=-1)
    best_yaw_abs = np.abs(best_residual_6d[:, 5])
    v3_yaw_abs = np.abs(v3_residual_6d[:, 5])
    xyz_cos = np.sum(best_residual_6d[:, :3] * v3_residual_6d[:, :3], axis=-1) / (
        np.linalg.norm(best_residual_6d[:, :3], axis=-1) * np.linalg.norm(v3_residual_6d[:, :3], axis=-1) + 1e-8
    )

    report = {
        "audit": "alignment_v3_dataset_semantics",
        "source_v2_npz": str(args.source_v2_npz),
        "v3_npz": str(args.v3_npz),
        "rows": int(indices.size),
        "current_errors": {
            "xy": _stats(current_xy),
            "z": _stats(current_z),
            "yaw": _stats(current_yaw),
        },
        "best_stage_candidate": {
            "pos_norm": _stats(best_pos_norm),
            "yaw_abs": _stats(best_yaw_abs),
            "post_xy": _stats(best_post_xy),
            "post_z": _stats(best_post_z),
            "post_yaw": _stats(best_post_yaw),
            "improve_xy_rate": float((best_post_xy < current_xy).mean()),
            "improve_z_rate": float((best_post_z < current_z).mean()),
            "improve_yaw_rate": float((best_post_yaw < current_yaw).mean()),
            "overshoot_any_rate": float(best_overshoot_any.mean()),
        },
        "v3_targets": {
            "pos_norm": _stats(v3_pos_norm),
            "yaw_abs": _stats(v3_yaw_abs),
            "post_xy": _stats(v3_post_xy),
            "post_z": _stats(v3_post_z),
            "post_yaw": _stats(v3_post_yaw),
            "improve_xy_rate": float((v3_post_xy < current_xy).mean()),
            "improve_z_rate": float((v3_post_z < current_z).mean()),
            "improve_yaw_rate": float((v3_post_yaw < current_yaw).mean()),
            "overshoot_proxy_rate": float(v3_overshoot.mean()),
            "invalid_risk_proxy_rate": float(v3_invalid.mean()),
        },
        "alignment_between_v3_and_best_candidate": {
            "xyz_cosine": _stats(xyz_cos),
            "pos_norm_ratio_mean": float(v3_pos_norm.mean() / max(best_pos_norm.mean(), 1e-8)),
            "yaw_abs_ratio_mean": float(v3_yaw_abs.mean() / max(best_yaw_abs.mean(), 1e-8)),
        },
    }

    report["semantic_judgement"] = {
        "target_residual_consistent_with_real_near_micro": bool(
            report["alignment_between_v3_and_best_candidate"]["pos_norm_ratio_mean"] > 0.6
            and report["alignment_between_v3_and_best_candidate"]["pos_norm_ratio_mean"] < 1.4
            and report["alignment_between_v3_and_best_candidate"]["yaw_abs_ratio_mean"] > 0.6
            and report["alignment_between_v3_and_best_candidate"]["yaw_abs_ratio_mean"] < 1.6
        ),
        "risk_proxy_non_degenerate": bool(0.02 <= float(v3_invalid.mean()) <= 0.98),
        "overshoot_proxy_non_degenerate": bool(0.02 <= float(v3_overshoot.mean()) <= 0.98),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
