"""
Build a phase-1 student handoff-state distillation dataset.

This dataset is runtime-safe: model inputs come only from non-privileged
runtime-observable fields, while teacher geometry / readiness are label-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_threshold(arr: np.ndarray, fallback: float) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    bad = ~np.isfinite(out) | (out <= 0.0)
    out[bad] = float(fallback)
    return out


def _teacher_band_label(
    teacher_xy: np.ndarray,
    teacher_z: np.ndarray,
    teacher_yaw: np.ndarray,
    rel_xy: np.ndarray,
    rel_z: np.ndarray,
    rel_yaw: np.ndarray,
    teacher_ready: np.ndarray,
) -> np.ndarray:
    # 0=support, 1=very_near, 2=release_ready
    out = np.zeros((teacher_xy.shape[0],), dtype=np.int64)
    release_ready = (
        (teacher_ready > 0.5)
        | (
            (teacher_xy <= rel_xy)
            & (teacher_z <= rel_z)
            & (teacher_yaw <= rel_yaw)
        )
    )
    very_near = (
        (teacher_xy <= 1.5 * rel_xy)
        & (teacher_z <= 1.5 * rel_z)
        & (teacher_yaw <= 1.5 * rel_yaw)
    )
    out[very_near] = 1
    out[release_ready] = 2
    return out


def _resolve_support_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"support npz not found: {path}")
    return path


def _resolve_pool_args(args):
    support_paths = [
        _resolve_support_path(p)
        for p in (args.support_npz or [])
    ]
    if not support_paths:
        raise RuntimeError("At least one --support_npz must be provided.")

    if args.source_name and len(args.source_name) != len(support_paths):
        raise RuntimeError("--source_name must match --support_npz count.")
    if args.source_weight_mult and len(args.source_weight_mult) != len(support_paths):
        raise RuntimeError("--source_weight_mult must match --support_npz count.")

    source_names = list(args.source_name or [p.parent.name for p in support_paths])
    source_weights = list(args.source_weight_mult or [1.0] * len(support_paths))
    return list(zip(support_paths, source_names, source_weights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", action="append", default=[])
    ap.add_argument("--source_name", action="append", default=[])
    ap.add_argument("--source_weight_mult", action="append", type=float, default=[])
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", default=None)
    ap.add_argument("--phase_id", type=int, default=1)
    ap.add_argument("--open_threshold", type=float, default=0.5)
    ap.add_argument("--fallback_release_xy", type=float, default=0.0085)
    ap.add_argument("--fallback_release_z", type=float, default=0.0035)
    ap.add_argument("--fallback_release_yaw", type=float, default=0.1243404)
    ap.add_argument("--sample_weight_xy_hard", type=float, default=3.0)
    ap.add_argument("--sample_weight_yaw_hard", type=float, default=2.0)
    ap.add_argument("--sample_weight_ready", type=float, default=2.0)
    args = ap.parse_args()

    source_specs = _resolve_pool_args(args)
    merged = {}
    source_rows = []
    source_summary = []

    for source_id, (support_path, source_name, source_weight_mult) in enumerate(source_specs):
        raw = np.load(support_path, allow_pickle=False)
        data = {k: np.asarray(raw[k]) for k in raw.files}

        phase_id = np.asarray(data.get("phase_id", np.ones((data["front_rgb"].shape[0],), dtype=np.int64)), dtype=np.int64)
        gripper_open = np.asarray(
            data.get("rollout_gripper_open", np.ones((data["front_rgb"].shape[0],), dtype=np.float32)),
            dtype=np.float32,
        )
        has_object = np.asarray(
            data.get("has_object_in_hand", np.zeros((data["front_rgb"].shape[0],), dtype=np.float32)),
            dtype=np.float32,
        )
        teacher_xy = np.asarray(data["teacher_truth_handoff_metric_xy_error"], dtype=np.float32)
        teacher_z = np.asarray(data["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float32)
        teacher_yaw = np.asarray(data["teacher_truth_handoff_metric_yaw_error"], dtype=np.float32)
        teacher_ready = np.asarray(
            data.get("teacher_truth_handoff_ready", np.zeros((teacher_xy.shape[0],), dtype=np.float32)),
            dtype=np.float32,
        )
        rel_xy = _safe_threshold(
            np.asarray(data.get("teacher_truth_handoff_release_threshold_xy_error", np.full_like(teacher_xy, np.nan)), dtype=np.float32),
            float(args.fallback_release_xy),
        )
        rel_z = _safe_threshold(
            np.asarray(data.get("teacher_truth_handoff_release_threshold_abs_z_error", np.full_like(teacher_z, np.nan)), dtype=np.float32),
            float(args.fallback_release_z),
        )
        rel_yaw = _safe_threshold(
            np.asarray(data.get("teacher_truth_handoff_release_threshold_yaw_error", np.full_like(teacher_yaw, np.nan)), dtype=np.float32),
            float(args.fallback_release_yaw),
        )
        runtime_valid = np.asarray(
            data.get("runtime_handoff_metric_valid", np.zeros((teacher_xy.shape[0],), dtype=np.float32)),
            dtype=np.float32,
        )
        runtime_ready = np.asarray(
            data.get(
                "runtime_handoff_ready_pred",
                data.get("runtime_handoff_ready", np.zeros((teacher_xy.shape[0],), dtype=np.float32)),
            ),
            dtype=np.float32,
        )
        runtime_xy = np.asarray(
            data.get("runtime_handoff_metric_xy_error", np.full_like(teacher_xy, np.nan)),
            dtype=np.float32,
        )
        runtime_z = np.asarray(
            data.get("runtime_handoff_metric_abs_z_error", np.full_like(teacher_z, np.nan)),
            dtype=np.float32,
        )
        runtime_yaw = np.asarray(
            data.get("runtime_handoff_metric_yaw_error", np.full_like(teacher_yaw, np.nan)),
            dtype=np.float32,
        )

        valid_teacher = np.isfinite(teacher_xy) & np.isfinite(teacher_z) & np.isfinite(teacher_yaw)
        keep = (
            (phase_id == int(args.phase_id))
            & (gripper_open >= float(args.open_threshold))
            & (has_object <= 0.5)
            & valid_teacher
        )
        keep_idx = np.flatnonzero(keep)
        if keep_idx.size == 0:
            continue

        teacher_xy_norm = (teacher_xy[keep] / np.maximum(rel_xy[keep], 1e-6)).astype(np.float32)
        teacher_z_norm = (teacher_z[keep] / np.maximum(rel_z[keep], 1e-6)).astype(np.float32)
        teacher_yaw_norm = (teacher_yaw[keep] / np.maximum(rel_yaw[keep], 1e-6)).astype(np.float32)
        band_label = _teacher_band_label(
            teacher_xy[keep],
            teacher_z[keep],
            teacher_yaw[keep],
            rel_xy[keep],
            rel_z[keep],
            rel_yaw[keep],
            teacher_ready[keep],
        )

        near_xy_hard = (
            (teacher_z[keep] <= 1.2 * rel_z[keep])
            & (teacher_xy[keep] > rel_xy[keep])
            & (teacher_yaw[keep] <= 1.2 * rel_yaw[keep])
        ).astype(np.float32)
        broad_xy_recovery = (
            (teacher_z[keep] <= 1.5 * rel_z[keep])
            & (teacher_xy[keep] > rel_xy[keep])
            & (teacher_yaw[keep] <= 1.5 * rel_yaw[keep])
        ).astype(np.float32)
        near_yaw_hard = (
            (teacher_z[keep] <= 1.2 * rel_z[keep])
            & (teacher_yaw[keep] > rel_yaw[keep])
            & (teacher_xy[keep] <= 1.2 * rel_xy[keep])
        ).astype(np.float32)
        near_coupled = (
            (teacher_z[keep] <= 1.2 * rel_z[keep])
            & (teacher_xy[keep] > rel_xy[keep])
            & (teacher_yaw[keep] > rel_yaw[keep])
        ).astype(np.float32)
        ready_support = (teacher_ready[keep] > 0.5).astype(np.float32)

        sample_weight = np.ones((keep_idx.size,), dtype=np.float32)
        sample_weight += near_xy_hard * float(args.sample_weight_xy_hard - 1.0)
        sample_weight += near_yaw_hard * float(args.sample_weight_yaw_hard - 1.0)
        sample_weight += ready_support * float(args.sample_weight_ready - 1.0)
        sample_weight *= float(source_weight_mult)

        disagreement = np.zeros((keep_idx.size,), dtype=np.float32)
        finite_runtime = np.isfinite(runtime_xy[keep]) & np.isfinite(runtime_z[keep]) & np.isfinite(runtime_yaw[keep]) & (runtime_valid[keep] > 0.5)
        if np.any(finite_runtime):
            runtime_release = (
                (runtime_xy[keep] <= rel_xy[keep])
                & (runtime_z[keep] <= rel_z[keep])
                & (runtime_yaw[keep] <= rel_yaw[keep])
            )
            disagreement = (runtime_release.astype(np.float32) != (teacher_ready[keep] > 0.5).astype(np.float32)).astype(np.float32)

        source_out = {
            "front_rgb": np.asarray(data["front_rgb"][keep], dtype=np.uint8),
            "wrist_rgb": np.asarray(data["wrist_rgb"][keep], dtype=np.uint8),
            "wrist_depth": np.asarray(data["wrist_depth"][keep], dtype=np.float32),
            "proprio": np.asarray(data["proprio"][keep], dtype=np.float32),
            "gripper_context": np.asarray(data["gripper_context"][keep], dtype=np.float32),
            "proxy_current_delta_basin_target": np.asarray(
                data.get("proxy_current_delta_basin_target", data["current_delta_basin_target"])[keep],
                dtype=np.float32,
            ),
            "current_dx_sign": np.asarray(data["current_dx_sign"][keep], dtype=np.int64),
            "current_dy_sign": np.asarray(data["current_dy_sign"][keep], dtype=np.int64),
            "current_dyaw_sign": np.asarray(data["current_dyaw_sign"][keep], dtype=np.int64),
            "basin_distance_bin": np.asarray(data["basin_distance_bin"][keep], dtype=np.int64),
            "substage_id": np.asarray(data["substage_id"][keep], dtype=np.int64),
            "contact_state": np.asarray(data["contact_state"][keep], dtype=np.int64),
            "stage_target_mode": np.asarray(data["stage_target_mode"][keep], dtype=np.int64),
            "episode_index": np.asarray(data["episode_index"][keep], dtype=np.int64),
            "teacher_xy_norm": teacher_xy_norm,
            "teacher_abs_z_norm": teacher_z_norm,
            "teacher_yaw_norm": teacher_yaw_norm,
            "teacher_metrics_norm": np.stack([teacher_xy_norm, teacher_z_norm, teacher_yaw_norm], axis=-1).astype(np.float32),
            "teacher_band_label": band_label.astype(np.int64),
            "teacher_truth_handoff_ready": np.asarray((teacher_ready[keep] > 0.5).astype(np.float32), dtype=np.float32),
            "teacher_truth_release_threshold_xy_error": np.asarray(rel_xy[keep], dtype=np.float32),
            "teacher_truth_release_threshold_abs_z_error": np.asarray(rel_z[keep], dtype=np.float32),
            "teacher_truth_release_threshold_yaw_error": np.asarray(rel_yaw[keep], dtype=np.float32),
            "runtime_handoff_metric_xy_error": np.asarray(runtime_xy[keep], dtype=np.float32),
            "runtime_handoff_metric_abs_z_error": np.asarray(runtime_z[keep], dtype=np.float32),
            "runtime_handoff_metric_yaw_error": np.asarray(runtime_yaw[keep], dtype=np.float32),
            "runtime_handoff_metric_valid": np.asarray(runtime_valid[keep], dtype=np.float32),
            "runtime_handoff_ready": np.asarray(runtime_ready[keep], dtype=np.float32),
            "runtime_handoff_ready_pred": np.asarray(runtime_ready[keep], dtype=np.float32),
            "near_xy_hard": near_xy_hard.astype(np.float32),
            "broad_xy_recovery": broad_xy_recovery.astype(np.float32),
            "near_yaw_hard": near_yaw_hard.astype(np.float32),
            "near_coupled": near_coupled.astype(np.float32),
            "ready_support": ready_support.astype(np.float32),
            "disagreement": disagreement.astype(np.float32),
            "sample_weight": sample_weight.astype(np.float32),
            "source_id": np.full((keep_idx.size,), source_id, dtype=np.int64),
            "source_name": np.full((keep_idx.size,), source_name, dtype="U128"),
        }
        for key, value in source_out.items():
            merged.setdefault(key, []).append(value)

        source_rows.append(int(keep_idx.size))
        source_summary.append(
            {
                "source_id": int(source_id),
                "source_name": source_name,
                "support_npz": str(support_path),
                "source_weight_mult": float(source_weight_mult),
                "rows": int(keep_idx.size),
                "episodes": int(np.unique(source_out["episode_index"]).size),
                "near_xy_hard_pos": int(np.sum(source_out["near_xy_hard"] > 0.5)),
                "broad_xy_recovery_pos": int(np.sum(source_out["broad_xy_recovery"] > 0.5)),
                "ready_support_pos": int(np.sum(source_out["ready_support"] > 0.5)),
            }
        )

    if not merged:
        raise RuntimeError("No valid phase-1 open rows found for handoff-state dataset.")

    out = {}
    for key, parts in merged.items():
        out[key] = np.concatenate(parts, axis=0)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    meta = {
        "support_npz": [str(p) for p, _, _ in source_specs],
        "num_rows": int(out["teacher_truth_handoff_ready"].shape[0]),
        "teacher_ready_pos": int(np.sum(out["teacher_truth_handoff_ready"] > 0.5)),
        "runtime_valid_pos": int(np.sum(out["runtime_handoff_metric_valid"] > 0.5)),
        "disagreement_pos": int(np.sum(out["disagreement"] > 0.5)),
        "near_xy_hard_pos": int(np.sum(out["near_xy_hard"] > 0.5)),
        "broad_xy_recovery_pos": int(np.sum(out["broad_xy_recovery"] > 0.5)),
        "near_yaw_hard_pos": int(np.sum(out["near_yaw_hard"] > 0.5)),
        "near_coupled_pos": int(np.sum(out["near_coupled"] > 0.5)),
        "ready_support_pos": int(np.sum(out["ready_support"] > 0.5)),
        "sources": source_summary,
    }
    meta_path = Path(args.meta_json) if args.meta_json else output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
