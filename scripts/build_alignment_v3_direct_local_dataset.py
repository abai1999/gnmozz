#!/usr/bin/env python3
"""Build near/micro-only dataset for alignment_v3_direct_local_controller.

This script intentionally excludes far/coarse rows. It supports two target
construction modes:

1. ``best_stage_candidate`` (preferred):
   Reuse the best near/micro proposal target already present in the v2 dataset.
   This keeps labels grounded in the frozen K=8 candidate bank and preserves
   non-trivial improve/overshoot behavior.

2. ``servo_pseudo``:
   Build a bounded proportional target from ``current_to_target_delta_local``.
   This is useful as a geometry-chain baseline, but it is too synthetic for the
   main direct-controller training path.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

NEAR_MICRO_BUCKETS = ("near_alignment", "micro_contact_refine")


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


def _bounded_direct_target(
    delta_local: np.ndarray,
    *,
    k_xy: float,
    k_z: float,
    k_yaw: float,
    max_pos: float,
    max_yaw: float,
):
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
    residual_4d = np.asarray([residual[0], residual[1], residual[2], residual[5]], dtype=np.float32)
    return residual_4d, residual, post


def _build_best_stage_candidate_targets(src, indices: np.ndarray) -> dict[str, np.ndarray]:
    proposal_actions = np.asarray(src["proposal_actions_local"], dtype=np.float32)[indices]
    post_xy = np.asarray(src["post_xy_error"], dtype=np.float32)[indices]
    post_z = np.asarray(src["post_z_error"], dtype=np.float32)[indices]
    post_yaw = np.asarray(src["post_yaw_error"], dtype=np.float32)[indices]
    overshoot_xy = np.asarray(src["overshoot_xy"], dtype=np.float32)[indices]
    overshoot_z = np.asarray(src["overshoot_z"], dtype=np.float32)[indices]
    overshoot_yaw = np.asarray(src["overshoot_yaw"], dtype=np.float32)[indices]
    geometry_improvement = np.asarray(src["geometry_improvement"], dtype=np.float32)[indices]
    candidate_valid_mask = np.asarray(src["candidate_valid_mask"], dtype=np.float32)[indices]
    current_xy = np.asarray(src["current_xy_error"], dtype=np.float32)[indices]
    current_z = np.asarray(src["current_z_error"], dtype=np.float32)[indices]
    current_yaw = np.asarray(src["current_yaw_error"], dtype=np.float32)[indices]
    target_idx = np.asarray(src["best_stage_action_index"], dtype=np.int64)[indices]

    row_ids = np.arange(indices.size, dtype=np.int64)
    target_residual_6d = proposal_actions[row_ids, target_idx]
    target_residual_4d = np.stack(
        [
            target_residual_6d[:, 0],
            target_residual_6d[:, 1],
            target_residual_6d[:, 2],
            target_residual_6d[:, 5],
        ],
        axis=-1,
    ).astype(np.float32)
    target_post_xy = post_xy[row_ids, target_idx].astype(np.float32)
    target_post_z = post_z[row_ids, target_idx].astype(np.float32)
    target_post_yaw = post_yaw[row_ids, target_idx].astype(np.float32)
    target_improves_xy = (target_post_xy < current_xy).astype(np.float32)
    target_improves_z = (target_post_z < current_z).astype(np.float32)
    target_improves_yaw = (target_post_yaw < current_yaw).astype(np.float32)
    overshoot_proxy = (
        (overshoot_xy[row_ids, target_idx] > 0)
        | (overshoot_z[row_ids, target_idx] > 0)
        | (overshoot_yaw[row_ids, target_idx] > 0)
    ).astype(np.float32)

    pos_norm = np.linalg.norm(target_residual_6d[:, :3], axis=-1)
    yaw_abs = np.abs(target_residual_6d[:, 5])
    large_residual = (pos_norm > np.percentile(pos_norm, 90)) | (yaw_abs > np.percentile(yaw_abs, 90))
    invalid_risk_proxy = (
        (candidate_valid_mask[row_ids, target_idx] < 0.5)
        | (geometry_improvement[row_ids, target_idx] <= 0.0)
        | large_residual
        | (overshoot_proxy > 0)
    ).astype(np.float32)

    return {
        "target_residual_local_4d": target_residual_4d,
        "target_residual_local_6d": target_residual_6d.astype(np.float32),
        "target_post_xy_error": target_post_xy,
        "target_post_z_error": target_post_z,
        "target_post_yaw_error": target_post_yaw,
        "target_improves_xy": target_improves_xy,
        "target_improves_z": target_improves_z,
        "target_improves_yaw": target_improves_yaw,
        "overshoot_proxy": overshoot_proxy,
        "invalid_risk_proxy": invalid_risk_proxy,
    }


def _build_servo_pseudo_targets(
    cur_delta: np.ndarray,
    current_xy: np.ndarray,
    current_z: np.ndarray,
    current_yaw: np.ndarray,
    *,
    k_xy: float,
    k_z: float,
    k_yaw: float,
    max_pos: float,
    max_yaw: float,
) -> dict[str, np.ndarray]:
    n = cur_delta.shape[0]
    target_residual_4d = np.zeros((n, 4), dtype=np.float32)
    target_residual_6d = np.zeros((n, 6), dtype=np.float32)
    target_post_xy = np.zeros((n,), dtype=np.float32)
    target_post_z = np.zeros((n,), dtype=np.float32)
    target_post_yaw = np.zeros((n,), dtype=np.float32)
    target_improves_xy = np.zeros((n,), dtype=np.float32)
    target_improves_z = np.zeros((n,), dtype=np.float32)
    target_improves_yaw = np.zeros((n,), dtype=np.float32)
    overshoot_proxy = np.zeros((n,), dtype=np.float32)
    invalid_risk_proxy = np.zeros((n,), dtype=np.float32)

    for row, delta in enumerate(cur_delta):
        residual_4d, residual_6d, post = _bounded_direct_target(
            delta,
            k_xy=k_xy,
            k_z=k_z,
            k_yaw=k_yaw,
            max_pos=max_pos,
            max_yaw=max_yaw,
        )
        post_xy = float(np.linalg.norm(post[:2]))
        post_z = float(abs(post[2]))
        post_yaw = float(abs(post[5]))
        target_residual_4d[row] = residual_4d
        target_residual_6d[row] = residual_6d
        target_post_xy[row] = post_xy
        target_post_z[row] = post_z
        target_post_yaw[row] = post_yaw
        target_improves_xy[row] = float(post_xy < float(current_xy[row]))
        target_improves_z[row] = float(post_z < float(current_z[row]))
        target_improves_yaw[row] = float(post_yaw < float(current_yaw[row]))
        overshoot_proxy[row] = float(post_xy > float(current_xy[row]) or post_z > float(current_z[row]))
        invalid_risk_proxy[row] = float(np.linalg.norm(residual_6d[:3]) > 0.0020 or abs(residual_6d[5]) > 0.0080)

    return {
        "target_residual_local_4d": target_residual_4d,
        "target_residual_local_6d": target_residual_6d,
        "target_post_xy_error": target_post_xy,
        "target_post_z_error": target_post_z,
        "target_post_yaw_error": target_post_yaw,
        "target_improves_xy": target_improves_xy,
        "target_improves_z": target_improves_z,
        "target_improves_yaw": target_improves_yaw,
        "overshoot_proxy": overshoot_proxy,
        "invalid_risk_proxy": invalid_risk_proxy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_npz", type=Path, required=True)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--k_xy", type=float, default=0.16)
    parser.add_argument("--k_z", type=float, default=0.12)
    parser.add_argument("--k_yaw", type=float, default=0.08)
    parser.add_argument("--max_pos", type=float, default=0.0025)
    parser.add_argument("--max_yaw", type=float, default=0.0100)
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=("best_stage_candidate", "servo_pseudo"),
        default="best_stage_candidate",
    )
    args = parser.parse_args()

    src = np.load(args.source_npz, allow_pickle=True)
    buckets = np.asarray(src["stage_bucket"])
    mask = np.array([b in NEAR_MICRO_BUCKETS for b in buckets], dtype=bool)
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise SystemExit("No near/micro rows found in source dataset.")

    cur_delta = np.asarray(src["current_to_target_delta_local"], dtype=np.float32)[indices, :6]
    current_xy = np.asarray(src["current_xy_error"], dtype=np.float32)[indices]
    current_z = np.asarray(src["current_z_error"], dtype=np.float32)[indices]
    current_yaw = np.asarray(src["current_yaw_error"], dtype=np.float32)[indices]

    if args.target_mode == "best_stage_candidate":
        target_dict = _build_best_stage_candidate_targets(src, indices)
    else:
        target_dict = _build_servo_pseudo_targets(
            cur_delta,
            current_xy,
            current_z,
            current_yaw,
            k_xy=args.k_xy,
            k_z=args.k_z,
            k_yaw=args.k_yaw,
            max_pos=args.max_pos,
            max_yaw=args.max_yaw,
        )

    out = {}
    passthrough = (
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "force_history",
        "proprio",
        "planner_base_action_local",
        "current_to_target_delta_local",
        "current_xy_error",
        "current_z_error",
        "current_yaw_error",
        "episode_index",
        "step_index",
        "row_index",
        "stage_bucket",
    )
    for key in passthrough:
        if key in src.files:
            out[key] = np.asarray(src[key])[indices]
    out.update(target_dict)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    report = {
        "audit": "alignment_v3_direct_local_dataset_build",
        "source_npz": str(args.source_npz),
        "output_npz": str(args.output_npz),
        "rows_total": int(buckets.shape[0]),
        "rows_selected": int(indices.size),
        "selected_bucket_histogram": dict(Counter(out["stage_bucket"].tolist())),
        "target_mode": str(args.target_mode),
        "k_xy": float(args.k_xy),
        "k_z": float(args.k_z),
        "k_yaw": float(args.k_yaw),
        "max_pos": float(args.max_pos),
        "max_yaw": float(args.max_yaw),
        "current_xy_error": _stats(current_xy),
        "current_z_error": _stats(current_z),
        "current_yaw_error": _stats(current_yaw),
        "target_post_xy_error": _stats(out["target_post_xy_error"]),
        "target_post_z_error": _stats(out["target_post_z_error"]),
        "target_post_yaw_error": _stats(out["target_post_yaw_error"]),
        "target_residual_pos_norm": _stats(np.linalg.norm(out["target_residual_local_6d"][:, :3], axis=-1)),
        "target_residual_yaw_abs": _stats(np.abs(out["target_residual_local_6d"][:, 5])),
        "improves_xy_rate": float(np.asarray(out["target_improves_xy"], dtype=np.float32).mean()),
        "improves_z_rate": float(np.asarray(out["target_improves_z"], dtype=np.float32).mean()),
        "improves_yaw_rate": float(np.asarray(out["target_improves_yaw"], dtype=np.float32).mean()),
        "overshoot_proxy_rate": float(np.asarray(out["overshoot_proxy"], dtype=np.float32).mean()),
        "invalid_risk_proxy_rate": float(np.asarray(out["invalid_risk_proxy"], dtype=np.float32).mean()),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
