"""
Build a near-ready xy+yaw dataset from support-state rows that include
teacher-truth diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _finite(a):
    arr = np.asarray(a)
    return np.all(np.isfinite(arr))


def _safe_threshold(arr, fallback: float) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    bad = ~np.isfinite(out) | (out <= 0.0)
    out[bad] = float(fallback)
    return out


def _runtime_xyyaw_from_rows(data) -> tuple[np.ndarray, np.ndarray]:
    runtime_xy = np.asarray(data["handoff_metric_xy_error"], dtype=np.float32) if "handoff_metric_xy_error" in data.files else None
    runtime_yaw = np.asarray(data["handoff_metric_yaw_error"], dtype=np.float32) if "handoff_metric_yaw_error" in data.files else None
    if runtime_xy is None or runtime_yaw is None or (not np.isfinite(runtime_xy).any()) or (not np.isfinite(runtime_yaw).any()):
        current_delta = np.asarray(data["current_delta_basin_target"], dtype=np.float32)
        runtime_xy = np.linalg.norm(current_delta[:, :2], axis=1).astype(np.float32)
        runtime_yaw = np.abs(current_delta[:, 5]).astype(np.float32)
    return runtime_xy, runtime_yaw


def _teacher_positive_fields(data, fallback_release_xy: float, fallback_release_z: float, fallback_release_yaw: float):
    if "handoff_ready_target" in data.files:
        ready = np.asarray(data["handoff_ready_target"], dtype=np.float32)
        teacher_xy = np.asarray(data["handoff_metric_xy_error"], dtype=np.float32)
        teacher_z = np.asarray(data["handoff_metric_abs_z_error"], dtype=np.float32)
        teacher_yaw = np.asarray(data["handoff_metric_yaw_error"], dtype=np.float32)
        rel_xy = _safe_threshold(
            np.asarray(data["handoff_threshold_xy_error"], dtype=np.float32),
            fallback_release_xy,
        )
        rel_z = _safe_threshold(
            np.asarray(data["handoff_threshold_abs_z_error"], dtype=np.float32),
            fallback_release_z,
        )
        rel_yaw = _safe_threshold(
            np.asarray(data["handoff_threshold_yaw_error"], dtype=np.float32),
            fallback_release_yaw,
        )
        return ready, teacher_xy, teacher_z, teacher_yaw, rel_xy, rel_z, rel_yaw
    ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32)
    teacher_xy = np.asarray(data["teacher_truth_handoff_metric_xy_error"], dtype=np.float32)
    teacher_z = np.asarray(data["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float32)
    teacher_yaw = np.asarray(data["teacher_truth_handoff_metric_yaw_error"], dtype=np.float32)
    rel_xy = _safe_threshold(
        np.asarray(data["teacher_truth_handoff_release_threshold_xy_error"], dtype=np.float32),
        fallback_release_xy,
    )
    rel_z = _safe_threshold(
        np.asarray(data["teacher_truth_handoff_release_threshold_abs_z_error"], dtype=np.float32),
        fallback_release_z,
    )
    rel_yaw = _safe_threshold(
        np.asarray(data["teacher_truth_handoff_release_threshold_yaw_error"], dtype=np.float32),
        fallback_release_yaw,
    )
    return ready, teacher_xy, teacher_z, teacher_yaw, rel_xy, rel_z, rel_yaw


def _append_rows(
    rows: list[dict[str, np.ndarray]],
    data,
    keep: np.ndarray,
    *,
    teacher_xy: np.ndarray,
    teacher_yaw: np.ndarray,
    teacher_z: np.ndarray,
    runtime_xy: np.ndarray,
    runtime_yaw: np.ndarray,
    rel_xy: np.ndarray,
    rel_yaw: np.ndarray,
    rel_z: np.ndarray,
    ready: np.ndarray,
    hard_negative: np.ndarray,
    hard_positive: np.ndarray,
    sample_weight: np.ndarray,
    source_tag: str,
) -> None:
    teacher_xy_norm = (teacher_xy[keep] / np.maximum(rel_xy[keep], 1e-6)).astype(np.float32)
    teacher_yaw_norm = (teacher_yaw[keep] / np.maximum(rel_yaw[keep], 1e-6)).astype(np.float32)
    runtime_xy_norm = (runtime_xy[keep] / np.maximum(rel_xy[keep], 1e-6)).astype(np.float32)
    runtime_yaw_norm = (runtime_yaw[keep] / np.maximum(rel_yaw[keep], 1e-6)).astype(np.float32)
    rows.append(
        {
            "front_rgb": np.asarray(data["front_rgb"][keep], dtype=np.uint8),
            "wrist_rgb": np.asarray(data["wrist_rgb"][keep], dtype=np.uint8),
            "wrist_depth": np.asarray(data["wrist_depth"][keep], dtype=np.float32),
            "proprio": np.asarray(data["proprio"][keep], dtype=np.float32),
            "gripper_context": np.asarray(data["gripper_context"][keep], dtype=np.float32),
            "substage_id": np.asarray(data["substage_id"][keep], dtype=np.int64),
            "contact_state": np.asarray(data["contact_state"][keep], dtype=np.int64),
            "stage_target_mode": np.asarray(data["stage_target_mode"][keep], dtype=np.int64),
            "teacher_truth_xy_error": np.asarray(teacher_xy[keep], dtype=np.float32),
            "teacher_truth_yaw_error": np.asarray(teacher_yaw[keep], dtype=np.float32),
            "teacher_truth_xyyaw_norm": np.stack([teacher_xy_norm, teacher_yaw_norm], axis=-1).astype(np.float32),
            "runtime_xyyaw_norm": np.stack([runtime_xy_norm, runtime_yaw_norm], axis=-1).astype(np.float32),
            "teacher_truth_ready_target": np.asarray(ready[keep], dtype=np.float32),
            "hard_negative_target": np.asarray(hard_negative[keep], dtype=np.float32),
            "hard_positive_target": np.asarray(hard_positive[keep], dtype=np.float32),
            "teacher_truth_release_xy_threshold": np.asarray(rel_xy[keep], dtype=np.float32),
            "teacher_truth_release_yaw_threshold": np.asarray(rel_yaw[keep], dtype=np.float32),
            "teacher_truth_release_abs_z_threshold": np.asarray(rel_z[keep], dtype=np.float32),
            "teacher_truth_abs_z_error": np.asarray(teacher_z[keep], dtype=np.float32),
            "sample_weight": np.asarray(sample_weight[keep], dtype=np.float32),
            "source_index": np.asarray(keep, dtype=np.int64),
            "source_tag": np.asarray([source_tag] * int(keep.size)),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support_npz", type=str, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--meta_json", type=str, default=None)
    parser.add_argument("--positive_support_npz", type=str, nargs="*", default=None)
    parser.add_argument("--substage_id", type=int, default=1)
    parser.add_argument("--z_near_mult", type=float, default=2.0)
    parser.add_argument("--xy_window_mult", type=float, default=4.0)
    parser.add_argument("--yaw_window_mult", type=float, default=4.0)
    parser.add_argument("--runtime_near_xy_mult", type=float, default=1.5)
    parser.add_argument("--runtime_near_yaw_mult", type=float, default=1.5)
    parser.add_argument("--positive_window_before", type=int, default=6)
    parser.add_argument("--positive_window_after", type=int, default=0)
    parser.add_argument("--positive_ready_mult_xy", type=float, default=1.15)
    parser.add_argument("--positive_ready_mult_yaw", type=float, default=1.15)
    parser.add_argument("--positive_ready_mult_z", type=float, default=1.5)
    parser.add_argument("--positive_sample_boost", type=float, default=6.0)
    parser.add_argument("--fallback_release_xy", type=float, default=0.0085)
    parser.add_argument("--fallback_release_z", type=float, default=0.0035)
    parser.add_argument("--fallback_release_yaw", type=float, default=0.1243404)
    args = parser.parse_args()

    rows = []
    neg_keep_count = 0
    neg_hard_negative_count = 0
    pos_keep_count = 0
    pos_ready_count = 0

    with np.load(args.support_npz, allow_pickle=False) as data:
        n = int(data["front_rgb"].shape[0])
        keep = []
        hard_negative = []
        hard_positive = []
        teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32)
        teacher_xy = np.asarray(data["teacher_truth_basin_xy"], dtype=np.float32)
        teacher_z = np.asarray(data["teacher_truth_basin_z"], dtype=np.float32)
        teacher_yaw = np.asarray(data["teacher_truth_basin_yaw"], dtype=np.float32)
        runtime_xy, runtime_yaw = _runtime_xyyaw_from_rows(data)
        rel_xy = _safe_threshold(
            np.asarray(data["teacher_truth_handoff_release_threshold_xy_error"], dtype=np.float32),
            float(args.fallback_release_xy),
        )
        rel_z = _safe_threshold(
            np.asarray(data["teacher_truth_handoff_release_threshold_abs_z_error"], dtype=np.float32),
            float(args.fallback_release_z),
        )
        rel_yaw = _safe_threshold(
            np.asarray(data["teacher_truth_handoff_release_threshold_yaw_error"], dtype=np.float32),
            float(args.fallback_release_yaw),
        )
        if "teacher_truth_handoff_metric_xy_error" in data.files:
            teacher_xy_metric = np.asarray(data["teacher_truth_handoff_metric_xy_error"], dtype=np.float32)
            teacher_z_metric = np.asarray(data["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float32)
            teacher_yaw_metric = np.asarray(data["teacher_truth_handoff_metric_yaw_error"], dtype=np.float32)
            if np.isfinite(teacher_xy_metric).any():
                teacher_xy = np.where(np.isfinite(teacher_xy_metric), teacher_xy_metric, teacher_xy)
            if np.isfinite(teacher_z_metric).any():
                teacher_z = np.where(np.isfinite(teacher_z_metric), teacher_z_metric, teacher_z)
            if np.isfinite(teacher_yaw_metric).any():
                teacher_yaw = np.where(np.isfinite(teacher_yaw_metric), teacher_yaw_metric, teacher_yaw)
        substage = np.asarray(data["substage_id"], dtype=np.int64)
        has_object = np.asarray(data["has_object_in_hand"], dtype=np.float32)
        gripper_open = np.asarray(data["rollout_gripper_open"], dtype=np.float32)

        for i in range(n):
            if int(substage[i]) != int(args.substage_id):
                continue
            if float(has_object[i]) > 0.5:
                continue
            if float(gripper_open[i]) < 0.5:
                continue
            vals = [teacher_xy[i], teacher_z[i], teacher_yaw[i], runtime_xy[i], runtime_yaw[i], rel_xy[i], rel_z[i], rel_yaw[i]]
            if not _finite(vals):
                continue
            if float(rel_xy[i]) <= 0.0 or float(rel_z[i]) <= 0.0:
                continue
            yaw_thr = float(rel_yaw[i]) if float(rel_yaw[i]) > 0.0 else 0.12
            teacher_z_near = float(teacher_z[i]) <= float(rel_z[i]) * float(args.z_near_mult)
            teacher_xy_window = float(teacher_xy[i]) <= float(rel_xy[i]) * float(args.xy_window_mult)
            teacher_yaw_window = float(teacher_yaw[i]) <= yaw_thr * float(args.yaw_window_mult)
            runtime_xy_near = float(runtime_xy[i]) <= float(rel_xy[i]) * float(args.runtime_near_xy_mult)
            runtime_yaw_near = float(runtime_yaw[i]) <= yaw_thr * float(args.runtime_near_yaw_mult)

            keep_row = bool(
                teacher_z_near
                and (
                    float(teacher_ready[i]) > 0.5
                    or teacher_xy_window
                    or teacher_yaw_window
                    or runtime_xy_near
                    or runtime_yaw_near
                )
            )
            if not keep_row:
                continue
            keep.append(i)
            hard_negative.append(
                float(runtime_xy_near and runtime_yaw_near and float(teacher_ready[i]) <= 0.5)
            )
            hard_positive.append(float(float(teacher_ready[i]) > 0.5))

        keep = np.asarray(keep, dtype=np.int64)
        if keep.size == 0:
            raise ValueError("No near-ready rows selected; check truth-diag support NPZ.")
        hard_negative_arr = np.zeros((n,), dtype=np.float32)
        hard_positive_arr = np.zeros((n,), dtype=np.float32)
        hard_negative_arr[keep] = np.asarray(hard_negative, dtype=np.float32)
        hard_positive_arr[keep] = np.asarray(hard_positive, dtype=np.float32)
        sample_weight = np.ones((n,), dtype=np.float32)
        sample_weight += 3.0 * hard_negative_arr
        sample_weight += 3.0 * hard_positive_arr
        _append_rows(
            rows,
            data,
            keep,
            teacher_xy=teacher_xy,
            teacher_yaw=teacher_yaw,
            teacher_z=teacher_z,
            runtime_xy=runtime_xy,
            runtime_yaw=runtime_yaw,
            rel_xy=rel_xy,
            rel_yaw=rel_yaw,
            rel_z=rel_z,
            ready=teacher_ready,
            hard_negative=hard_negative_arr,
            hard_positive=hard_positive_arr,
            sample_weight=sample_weight,
            source_tag="truthdiag_negative",
        )
        neg_keep_count = int(keep.size)
        neg_hard_negative_count = int(np.sum(hard_negative_arr > 0.5))

    for positive_npz in (args.positive_support_npz or []):
        with np.load(positive_npz, allow_pickle=False) as data:
            n = int(data["front_rgb"].shape[0])
            substage = np.asarray(data["substage_id"], dtype=np.int64)
            has_object = np.asarray(data["has_object_in_hand"], dtype=np.float32)
            gripper_open = np.asarray(data["rollout_gripper_open"], dtype=np.float32)
            teacher_ready, teacher_xy, teacher_z, teacher_yaw, rel_xy, rel_z, rel_yaw = _teacher_positive_fields(
                data,
                float(args.fallback_release_xy),
                float(args.fallback_release_z),
                float(args.fallback_release_yaw),
            )
            runtime_xy, runtime_yaw = _runtime_xyyaw_from_rows(data)
            episode_index = np.asarray(data["episode_index"], dtype=np.int64)
            rollout_step = np.asarray(data["rollout_step"], dtype=np.int64)
            positive_anchor_idx = np.where(teacher_ready > 0.5)[0]
            positive_margin = (
                (teacher_xy <= rel_xy * float(args.positive_ready_mult_xy))
                & (teacher_yaw <= rel_yaw * float(args.positive_ready_mult_yaw))
                & (teacher_z <= rel_z * float(args.positive_ready_mult_z))
            )
            selected = set()
            for anchor in positive_anchor_idx.tolist():
                ep = int(episode_index[anchor])
                step = int(rollout_step[anchor])
                lo = step - int(args.positive_window_before)
                hi = step + int(args.positive_window_after)
                mask = (
                    (episode_index == ep)
                    & (rollout_step >= lo)
                    & (rollout_step <= hi)
                    & (substage == int(args.substage_id))
                    & (has_object <= 0.5)
                    & (gripper_open >= 0.5)
                )
                for idx in np.where(mask)[0].tolist():
                    selected.add(int(idx))
            soft_anchor_idx = np.where(positive_margin)[0]
            for anchor in soft_anchor_idx.tolist():
                ep = int(episode_index[anchor])
                step = int(rollout_step[anchor])
                lo = step - int(args.positive_window_before)
                hi = step + int(args.positive_window_after)
                mask = (
                    (episode_index == ep)
                    & (rollout_step >= lo)
                    & (rollout_step <= hi)
                    & (substage == int(args.substage_id))
                    & (has_object <= 0.5)
                    & (gripper_open >= 0.5)
                )
                for idx in np.where(mask)[0].tolist():
                    selected.add(int(idx))
            if not selected:
                continue
            keep = np.asarray(sorted(selected), dtype=np.int64)
            hard_positive = (
                (teacher_ready > 0.5) | positive_margin
            ).astype(np.float32)
            hard_negative = np.zeros((n,), dtype=np.float32)
            sample_weight = np.ones((n,), dtype=np.float32)
            sample_weight += float(args.positive_sample_boost) * hard_positive
            _append_rows(
                rows,
                data,
                keep,
                teacher_xy=teacher_xy,
                teacher_yaw=teacher_yaw,
                teacher_z=teacher_z,
                runtime_xy=runtime_xy,
                runtime_yaw=runtime_yaw,
                rel_xy=rel_xy,
                rel_yaw=rel_yaw,
                rel_z=rel_z,
                ready=hard_positive,
                hard_negative=hard_negative,
                hard_positive=hard_positive,
                sample_weight=sample_weight,
                source_tag=f"teacher_positive::{Path(positive_npz).parent.name}",
            )
            pos_keep_count += int(keep.size)
            pos_ready_count += int(np.sum(hard_positive[keep] > 0.5))

    out = {}
    for key in rows[0].keys():
        out[key] = np.concatenate([row[key] for row in rows], axis=0)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    meta = {
        "support_npz": str(args.support_npz),
        "positive_support_npz": [str(p) for p in (args.positive_support_npz or [])],
        "num_rows": int(out["front_rgb"].shape[0]),
        "num_rows_negative_source": int(neg_keep_count),
        "num_rows_positive_source": int(pos_keep_count),
        "num_hard_negative": int(np.sum(out["hard_negative_target"] > 0.5)),
        "num_hard_positive": int(np.sum(out["hard_positive_target"] > 0.5)),
        "num_teacher_ready_positive": int(np.sum(out["teacher_truth_ready_target"] > 0.5)),
        "num_teacher_ready_positive_from_teacher_support": int(pos_ready_count),
        "substage_id": int(args.substage_id),
        "z_near_mult": float(args.z_near_mult),
        "xy_window_mult": float(args.xy_window_mult),
        "yaw_window_mult": float(args.yaw_window_mult),
        "runtime_near_xy_mult": float(args.runtime_near_xy_mult),
        "runtime_near_yaw_mult": float(args.runtime_near_yaw_mult),
        "positive_window_before": int(args.positive_window_before),
        "positive_window_after": int(args.positive_window_after),
        "positive_ready_mult_xy": float(args.positive_ready_mult_xy),
        "positive_ready_mult_yaw": float(args.positive_ready_mult_yaw),
        "positive_ready_mult_z": float(args.positive_ready_mult_z),
        "positive_sample_boost": float(args.positive_sample_boost),
    }
    meta_path = Path(args.meta_json) if args.meta_json else output_npz.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
