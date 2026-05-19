#!/usr/bin/env python3
"""Build a non-privileged learned-provider distillation dataset.

This script turns a clean verified teacher / student-vNext support set into a
runtime-safe training set for the deployment-time learned target provider.

The resulting NPZ intentionally excludes any privileged target-pose fields.
Only non-privileged inputs remain:
  - RGB / depth / proprio
  - gripper context
  - coarse stage/state context

Teacher geometry is reduced to label-only supervision:
  - target-delta labels
  - handoff readiness labels
  - explicit ready / near / support buckets

The student runtime will never read these teacher labels directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_float(val, default: float = 0.0) -> float:
    try:
        out = float(val)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _safe_threshold(arr: np.ndarray | None, fallback: float) -> np.ndarray:
    if arr is None:
        return np.asarray(fallback, dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32).copy()
    bad = ~np.isfinite(out) | (out <= 0.0)
    out[bad] = float(fallback)
    return out


def _to_hwc_uint8(arr: np.ndarray | None) -> np.ndarray:
    if arr is None:
        return np.zeros((96, 96, 3), dtype=np.uint8)
    ten = np.asarray(arr)
    if ten.ndim == 3 and ten.shape[0] == 3 and ten.shape[-1] != 3:
        ten = np.transpose(ten, (1, 2, 0))
    elif ten.ndim == 4 and ten.shape[0] == 1 and ten.shape[1] == 3:
        ten = np.transpose(ten[0], (1, 2, 0))
    elif ten.ndim == 4 and ten.shape[-1] == 3:
        ten = ten[0]
    ten = ten.astype(np.float32)
    if ten.max(initial=0.0) <= 1.5:
        ten = ten * 255.0
    ten = np.clip(np.rint(ten), 0, 255).astype(np.uint8)
    return ten


def _to_depth_hw(arr: np.ndarray | None) -> np.ndarray:
    if arr is None:
        return np.zeros((96, 96), dtype=np.float32)
    ten = np.asarray(arr, dtype=np.float32)
    if ten.ndim == 3 and ten.shape[0] == 1:
        ten = ten[0]
    elif ten.ndim == 3 and ten.shape[-1] == 1:
        ten = ten[..., 0]
    elif ten.ndim == 4 and ten.shape[0] == 1:
        ten = ten[0]
        if ten.ndim == 3 and ten.shape[0] == 1:
            ten = ten[0]
    return ten.astype(np.float32)


def _normalize_gripper_context(arr: np.ndarray | None, depth_proxy: float) -> np.ndarray:
    if arr is None:
        return np.asarray([1.0, 1.0, depth_proxy], dtype=np.float32)
    ten = np.asarray(arr, dtype=np.float32).reshape(-1)
    if ten.size == 0:
        return np.asarray([1.0, 1.0, depth_proxy], dtype=np.float32)
    if ten.size == 1:
        ten = np.asarray([ten[0], ten[0], depth_proxy], dtype=np.float32)
    elif ten.size == 2:
        ten = np.asarray([ten[0], ten[1], depth_proxy], dtype=np.float32)
    else:
        ten = np.asarray([ten[0], ten[1], depth_proxy], dtype=np.float32)
    return ten.astype(np.float32)


def _teacher_band_label(
    teacher_xy: np.ndarray,
    teacher_z: np.ndarray,
    teacher_yaw: np.ndarray,
    rel_xy: np.ndarray,
    rel_z: np.ndarray,
    rel_yaw: np.ndarray,
    teacher_ready: np.ndarray,
) -> np.ndarray:
    out = np.zeros((teacher_xy.shape[0],), dtype=np.int64)
    release_ready = (
        (teacher_ready > 0.5)
        | ((teacher_xy <= rel_xy) & (teacher_z <= rel_z) & (teacher_yaw <= rel_yaw))
    )
    very_near = (
        (teacher_xy <= 1.5 * rel_xy)
        & (teacher_z <= 1.5 * rel_z)
        & (teacher_yaw <= 1.5 * rel_yaw)
    )
    out[very_near] = 1
    out[release_ready] = 2
    return out


def _derived_contact_state(stage_bucket_id: np.ndarray, teacher_band_label: np.ndarray) -> np.ndarray:
    # Keep the runtime contract coarse and non-privileged:
    # 0=support-ish, 1=very near, 2=release-ready.
    out = np.asarray(teacher_band_label, dtype=np.int64).copy()
    out = np.clip(out, 0, 2)
    if stage_bucket_id.size == out.size:
        out = np.where(np.isfinite(stage_bucket_id), np.clip(stage_bucket_id.astype(np.int64), 0, 7), out)
    return out.astype(np.int64)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--report_json", required=True)
    ap.add_argument("--source_name", action="append", default=[])
    ap.add_argument("--source_weight_mult", action="append", type=float, default=[])
    ap.add_argument("--fallback_release_xy", type=float, default=0.0085)
    ap.add_argument("--fallback_release_z", type=float, default=0.0035)
    ap.add_argument("--fallback_release_yaw", type=float, default=0.1243404)
    ap.add_argument("--min_band_label", type=int, default=0)
    ap.add_argument("--keep_verified_positive_only", action="store_true", default=False)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.source_name and len(args.source_name) != len(args.input_npz):
        raise SystemExit("--source_name must match --input_npz count")
    if args.source_weight_mult and len(args.source_weight_mult) != len(args.input_npz):
        raise SystemExit("--source_weight_mult must match --input_npz count")

    source_names = list(args.source_name or [Path(p).stem for p in args.input_npz])
    source_weight_mults = list(args.source_weight_mult or [1.0] * len(args.input_npz))

    rows: list[dict[str, np.ndarray]] = []
    source_summary: dict[str, dict[str, int]] = {}
    band_hist = {"support": 0, "very_near": 0, "release_ready": 0}
    bucket_hist = {
        "support": 0,
        "near_xy_hard": 0,
        "near_yaw_hard": 0,
        "near_coupled": 0,
        "very_near": 0,
        "release_ready": 0,
    }

    for support_path, source_name, source_weight_mult in zip(args.input_npz, source_names, source_weight_mults):
        raw = np.load(support_path, allow_pickle=False)
        data = {k: np.asarray(raw[k]) for k in raw.files}
        n = int(data["teacher_target_delta_local_6d"].shape[0] if "teacher_target_delta_local_6d" in data else data["front_rgb"].shape[0])

        target_delta = np.asarray(data.get("teacher_target_delta_local_6d"), dtype=np.float32)
        if target_delta.ndim != 2 or target_delta.shape[1] < 6:
            raise SystemExit(f"{support_path}: missing teacher_target_delta_local_6d[:,6]")
        target_delta = target_delta[:, :6]

        teacher_xy = np.linalg.norm(target_delta[:, :2], axis=1).astype(np.float32)
        teacher_z = np.abs(target_delta[:, 2]).astype(np.float32)
        teacher_yaw = np.abs(target_delta[:, 5]).astype(np.float32)
        rel_xy = _safe_threshold(data.get("teacher_truth_handoff_release_threshold_xy_error"), args.fallback_release_xy)
        rel_z = _safe_threshold(data.get("teacher_truth_handoff_release_threshold_abs_z_error"), args.fallback_release_z)
        rel_yaw = _safe_threshold(data.get("teacher_truth_handoff_release_threshold_yaw_error"), args.fallback_release_yaw)
        if rel_xy.ndim == 0:
            rel_xy = np.full((n,), float(rel_xy), dtype=np.float32)
        if rel_z.ndim == 0:
            rel_z = np.full((n,), float(rel_z), dtype=np.float32)
        if rel_yaw.ndim == 0:
            rel_yaw = np.full((n,), float(rel_yaw), dtype=np.float32)

        if "verified_positive" in data:
            verified_positive = np.asarray(data["verified_positive"], dtype=np.float32) > 0.5
        else:
            verified_positive = np.zeros((n,), dtype=bool)
        if "phase_id" in data:
            phase_id = np.asarray(data["phase_id"], dtype=np.int64)
        else:
            phase_id = np.zeros((n,), dtype=np.int64)
        if "stage_bucket_id" in data:
            stage_bucket_id = np.asarray(data["stage_bucket_id"], dtype=np.int64)
        else:
            stage_bucket_id = np.zeros((n,), dtype=np.int64)

        teacher_ready = (teacher_xy <= rel_xy) & (teacher_z <= rel_z) & (teacher_yaw <= rel_yaw)
        teacher_ready = teacher_ready | verified_positive
        band_label = _teacher_band_label(teacher_xy, teacher_z, teacher_yaw, rel_xy, rel_z, rel_yaw, teacher_ready.astype(np.float32))

        keep = (band_label >= int(args.min_band_label)) | verified_positive
        if bool(args.keep_verified_positive_only):
            keep = verified_positive
        keep_idx = np.flatnonzero(keep)
        if keep_idx.size == 0:
            continue

        # Runtime-safe inputs.
        front_rgb = np.stack([_to_hwc_uint8(data["front_rgb"][i]) for i in keep_idx], axis=0)
        wrist_rgb = np.stack([_to_hwc_uint8(data["wrist_rgb"][i]) for i in keep_idx], axis=0)
        wrist_depth = np.stack([_to_depth_hw(data["wrist_depth"][i]) for i in keep_idx], axis=0)
        proprio = np.asarray(data["proprio"][keep_idx], dtype=np.float32)
        gripper_context = np.stack(
            [
                _normalize_gripper_context(
                    data.get("gripper_context", None)[i] if "gripper_context" in data else None,
                    float(np.percentile(_to_depth_hw(data["wrist_depth"][i])[np.isfinite(_to_depth_hw(data["wrist_depth"][i]))], 5.0))
                    if np.isfinite(_to_depth_hw(data["wrist_depth"][i])).any()
                    else 0.0,
                )
                for i in keep_idx
            ],
            axis=0,
        )

        # Runtime-safe state labels.
        has_object_in_hand = (phase_id[keep_idx] > 0).astype(np.float32)
        substage_id = np.clip(phase_id[keep_idx], 0, 7).astype(np.int64)
        contact_state = _derived_contact_state(stage_bucket_id[keep_idx], band_label[keep_idx])
        stage_target_mode = np.clip(phase_id[keep_idx], 0, 7).astype(np.int64)

        teacher_truth_ready = teacher_ready[keep_idx].astype(np.float32)
        teacher_band = band_label[keep_idx].astype(np.int64)
        ready_support = (teacher_band >= 1).astype(np.float32)
        near_xy_hard = (
            (teacher_z[keep_idx] <= 1.2 * rel_z[keep_idx])
            & (teacher_xy[keep_idx] > rel_xy[keep_idx])
            & (teacher_yaw[keep_idx] <= 1.2 * rel_yaw[keep_idx])
        ).astype(np.float32)
        near_yaw_hard = (
            (teacher_z[keep_idx] <= 1.2 * rel_z[keep_idx])
            & (teacher_yaw[keep_idx] > rel_yaw[keep_idx])
            & (teacher_xy[keep_idx] <= 1.2 * rel_xy[keep_idx])
        ).astype(np.float32)
        near_coupled = (
            (teacher_z[keep_idx] <= 1.2 * rel_z[keep_idx])
            & (teacher_xy[keep_idx] > rel_xy[keep_idx])
            & (teacher_yaw[keep_idx] > rel_yaw[keep_idx])
        ).astype(np.float32)
        disagreement = ((teacher_band == 1) & (~teacher_truth_ready.astype(bool))).astype(np.float32)

        handoff_metric_xy = teacher_xy[keep_idx].astype(np.float32)
        handoff_metric_z = teacher_z[keep_idx].astype(np.float32)
        handoff_metric_yaw = teacher_yaw[keep_idx].astype(np.float32)

        bucket_label = np.zeros((keep_idx.size,), dtype=np.int64)
        bucket_label = np.where(teacher_band >= 2, 5, bucket_label)
        bucket_label = np.where((bucket_label == 0) & (teacher_band == 1), 4, bucket_label)
        bucket_label = np.where((bucket_label == 0) & (near_coupled > 0.5), 3, bucket_label)
        bucket_label = np.where((bucket_label == 0) & (near_yaw_hard > 0.5), 2, bucket_label)
        bucket_label = np.where((bucket_label == 0) & (near_xy_hard > 0.5), 1, bucket_label)
        bucket_label = np.where((bucket_label == 0) & (teacher_band == 0), 0, bucket_label)

        is_ready = teacher_truth_ready > 0.5
        pos_count = max(int(np.sum(is_ready)), 1)
        neg_count = max(int(np.sum(~is_ready)), 1)
        neg_balance = float(pos_count) / float(neg_count)
        neg_balance = float(np.clip(neg_balance, 1.0, 4.0))

        category_weight = np.ones((keep_idx.size,), dtype=np.float32)
        category_weight = np.where(bucket_label == 0, 1.0, category_weight)
        category_weight = np.where(bucket_label == 1, 2.0, category_weight)
        category_weight = np.where(bucket_label == 2, 2.0, category_weight)
        category_weight = np.where(bucket_label == 3, 2.5, category_weight)
        category_weight = np.where(bucket_label == 4, 1.5, category_weight)
        category_weight = np.where(bucket_label == 5, 1.0, category_weight)

        ready_balance_weight = np.where(is_ready, 1.0, neg_balance).astype(np.float32)
        sample_weight = ready_balance_weight * category_weight.astype(np.float32) * float(source_weight_mult)

        out_rows = []
        for j, idx in enumerate(keep_idx.tolist()):
            out_rows.append(
                {
                    "front_rgb": front_rgb[j],
                    "wrist_rgb": wrist_rgb[j],
                    "wrist_depth": wrist_depth[j],
                    "proprio": proprio[j],
                    "gripper_context": gripper_context[j],
                    "has_object_in_hand": np.asarray(has_object_in_hand[j], dtype=np.float32),
                    "substage_id": np.asarray(substage_id[j], dtype=np.int64),
                    "contact_state": np.asarray(contact_state[j], dtype=np.int64),
                    "stage_target_mode": np.asarray(stage_target_mode[j], dtype=np.int64),
                    "target_delta_teacher": target_delta[idx].astype(np.float32),
                    "teacher_target_delta_local_6d": target_delta[idx].astype(np.float32),
                    "handoff_metric_xy_error": np.asarray(handoff_metric_xy[j], dtype=np.float32),
                    "handoff_metric_abs_z_error": np.asarray(handoff_metric_z[j], dtype=np.float32),
                    "handoff_metric_yaw_error": np.asarray(handoff_metric_yaw[j], dtype=np.float32),
                    "handoff_ready_target": np.asarray(teacher_truth_ready[j], dtype=np.float32),
                    "teacher_truth_handoff_ready": np.asarray(teacher_truth_ready[j], dtype=np.float32),
                    "teacher_truth_handoff_metric_xy_error": np.asarray(handoff_metric_xy[j], dtype=np.float32),
                    "teacher_truth_handoff_metric_abs_z_error": np.asarray(handoff_metric_z[j], dtype=np.float32),
                    "teacher_truth_handoff_metric_yaw_error": np.asarray(handoff_metric_yaw[j], dtype=np.float32),
                    "teacher_truth_handoff_release_threshold_xy_error": np.asarray(rel_xy[idx], dtype=np.float32),
                    "teacher_truth_handoff_release_threshold_abs_z_error": np.asarray(rel_z[idx], dtype=np.float32),
                    "teacher_truth_handoff_release_threshold_yaw_error": np.asarray(rel_yaw[idx], dtype=np.float32),
                    "teacher_band_label": np.asarray(teacher_band[j], dtype=np.int64),
                    "handoff_bucket_label": np.asarray(bucket_label[j], dtype=np.int64),
                    "ready_support": np.asarray(ready_support[j], dtype=np.float32),
                    "near_xy_hard": np.asarray(near_xy_hard[j], dtype=np.float32),
                    "near_yaw_hard": np.asarray(near_yaw_hard[j], dtype=np.float32),
                    "near_coupled": np.asarray(near_coupled[j], dtype=np.float32),
                    "disagreement": np.asarray(disagreement[j], dtype=np.float32),
                    "sample_weight": np.asarray(sample_weight[j], dtype=np.float32),
                    "phase_id": np.asarray(phase_id[idx], dtype=np.int64),
                    "stage_bucket_id": np.asarray(stage_bucket_id[idx], dtype=np.int64),
                    "verified_positive": np.asarray(float(verified_positive[idx]), dtype=np.float32),
                    "source_name": np.asarray(source_name),
                    "alignment_phase": np.asarray(data["alignment_phase"][idx]) if "alignment_phase" in data else np.asarray("unknown"),
                    "stage_bucket": np.asarray(data["stage_bucket"][idx]) if "stage_bucket" in data else np.asarray("unknown"),
                    "teacher_close_ready": np.asarray(teacher_truth_ready[j], dtype=np.float32),
                    "teacher_close_ready_all": np.asarray(teacher_truth_ready[j], dtype=np.float32),
                    "close_ready_bridge_mask": np.asarray(float(teacher_band[j] >= 1), dtype=np.float32),
                    "close_ready_exact_mask": np.asarray(float(teacher_band[j] >= 2), dtype=np.float32),
                    "teacher_close_ready_score": np.asarray(float(1.0 / (1.0 + max(teacher_xy[j] / max(rel_xy[idx], 1e-6), handoff_metric_z[j] / max(rel_z[idx], 1e-6), handoff_metric_yaw[j] / max(rel_yaw[idx], 1e-6)))), dtype=np.float32),
                    "teacher_residual_action_4d": np.asarray(data["teacher_residual_action_4d"][idx], dtype=np.float32)[:4] if "teacher_residual_action_4d" in data else np.zeros((4,), dtype=np.float32),
                    "teacher_residual_trajectory_4d": np.asarray(data["teacher_residual_trajectory_4d"][idx], dtype=np.float32)[:,:4] if "teacher_residual_trajectory_4d" in data else np.repeat(np.zeros((1,4), dtype=np.float32), 8, axis=0),
                    "teacher_progress_label": np.asarray(data["teacher_progress_label"][idx], dtype=np.float32) if "teacher_progress_label" in data else np.zeros((3,), dtype=np.float32),
                    "teacher_risk_label": np.asarray(data["teacher_risk_label"][idx], dtype=np.float32) if "teacher_risk_label" in data else np.asarray(float(teacher_band[j] == 0), dtype=np.float32),
                    "teacher_stop_label": np.asarray(data["teacher_stop_label"][idx], dtype=np.float32) if "teacher_stop_label" in data else np.asarray(float(teacher_band[j] == 0), dtype=np.float32),
                    "teacher_confidence_label": np.asarray(data["teacher_confidence_label"][idx], dtype=np.float32) if "teacher_confidence_label" in data else np.asarray(float(teacher_band[j]), dtype=np.float32),
                }
            )

        rows.extend(out_rows)
        key = str(source_name)
        source_summary[key] = {
            "input_rows": int(n),
            "kept_rows": int(keep_idx.size),
            "teacher_ready_pos": int(np.sum(teacher_truth_ready > 0.5)),
            "band1_rows": int(np.sum(teacher_band >= 1)),
            "band2_rows": int(np.sum(teacher_band >= 2)),
        }
        band_hist["support"] += int(np.sum(teacher_band == 0))
        band_hist["very_near"] += int(np.sum(teacher_band == 1))
        band_hist["release_ready"] += int(np.sum(teacher_band >= 2))
        bucket_hist["support"] += int(np.sum(bucket_label == 0))
        bucket_hist["near_xy_hard"] += int(np.sum(bucket_label == 1))
        bucket_hist["near_yaw_hard"] += int(np.sum(bucket_label == 2))
        bucket_hist["near_coupled"] += int(np.sum(bucket_label == 3))
        bucket_hist["very_near"] += int(np.sum(bucket_label == 4))
        bucket_hist["release_ready"] += int(np.sum(bucket_label == 5))

    if not rows:
        raise SystemExit("no rows selected for learned-provider distillation")

    keys = sorted({k for row in rows for k in row.keys()})
    out = {k: np.asarray([row[k] for row in rows]) for k in keys}
    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    report = {
        "output_npz": str(out_path),
        "rows": int(len(rows)),
        "band_histogram": band_hist,
        "bucket_histogram": bucket_hist,
        "source_summary": source_summary,
        "fields": sorted(list(out.keys())),
        "notes": [
            "This dataset is non-privileged at runtime: no teacher_truth_* values are fed to the deployed student.",
            "Teacher geometry is distilled only into labels used for training target/handoff predictors.",
        ],
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
