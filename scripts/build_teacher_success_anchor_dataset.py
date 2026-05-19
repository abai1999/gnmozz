"""
Build a teacher-success anchor dataset for Stage-0 warm start.

This script is intentionally conservative:
- If an input support set already contains explicit teacher-ready labels, use them.
- If an older teacher-success support set lacks explicit ready labels (e.g. formal30),
  derive a small "success-end" proxy window near the minimum teacher distance.

The output schema matches the handoff-state v2 trainer input schema.
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


def _quat_to_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
    x = quat_xyzw[..., 0]
    y = quat_xyzw[..., 1]
    z = quat_xyzw[..., 2]
    w = quat_xyzw[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


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


def _compute_teacher_metrics(data: dict[str, np.ndarray], fallback_release_xy: float, fallback_release_z: float, fallback_release_yaw: float):
    if "teacher_truth_handoff_metric_xy_error" in data:
        teacher_xy = np.asarray(data["teacher_truth_handoff_metric_xy_error"], dtype=np.float32)
        teacher_z = np.asarray(data["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float32)
        teacher_yaw = np.asarray(data["teacher_truth_handoff_metric_yaw_error"], dtype=np.float32)
        teacher_ready = np.asarray(data.get("teacher_truth_handoff_ready", np.zeros_like(teacher_xy)), dtype=np.float32)
        rel_xy = _safe_threshold(np.asarray(data.get("teacher_truth_handoff_release_threshold_xy_error", np.full_like(teacher_xy, np.nan)), dtype=np.float32), fallback_release_xy)
        rel_z = _safe_threshold(np.asarray(data.get("teacher_truth_handoff_release_threshold_abs_z_error", np.full_like(teacher_z, np.nan)), dtype=np.float32), fallback_release_z)
        rel_yaw = _safe_threshold(np.asarray(data.get("teacher_truth_handoff_release_threshold_yaw_error", np.full_like(teacher_yaw, np.nan)), dtype=np.float32), fallback_release_yaw)
        return teacher_xy, teacher_z, teacher_yaw, teacher_ready, rel_xy, rel_z, rel_yaw, False

    delta = None
    if "target_delta_teacher" in data:
        delta = np.asarray(data["target_delta_teacher"], dtype=np.float32)
    elif "teacher_current_delta_basin_target" in data:
        delta = np.asarray(data["teacher_current_delta_basin_target"], dtype=np.float32)
    elif "current_delta_basin_target" in data:
        delta = np.asarray(data["current_delta_basin_target"], dtype=np.float32)
    elif "reference_anchor_pose_7d" in data and "current_pose_7d" in data:
        cur_pose = np.asarray(data["current_pose_7d"], dtype=np.float32)
        ref_pose = np.asarray(data["reference_anchor_pose_7d"], dtype=np.float32)
        delta = np.zeros((cur_pose.shape[0], 6), dtype=np.float32)
        delta[:, :3] = cur_pose[:, :3] - ref_pose[:, :3]
        delta[:, 3:6] = cur_pose[:, 3:6] - ref_pose[:, 3:6]
    if delta is None:
        raise RuntimeError(
            "support npz missing teacher handoff metrics and also lacks target_delta_teacher/current_delta_basin_target/current_pose_7d fallback"
        )

    teacher_xy = np.linalg.norm(delta[:, :2], axis=1).astype(np.float32)
    teacher_z = np.abs(delta[:, 2]).astype(np.float32)

    if "current_pose_7d" in data and "reference_anchor_pose_7d" in data:
        cur_q = np.asarray(data["current_pose_7d"][:, 3:7], dtype=np.float32)
        ref_q = np.asarray(data["reference_anchor_pose_7d"][:, 3:7], dtype=np.float32)
        teacher_yaw = np.abs(_wrap_to_pi(_quat_to_yaw(cur_q) - _quat_to_yaw(ref_q))).astype(np.float32)
    else:
        teacher_yaw = np.linalg.norm(delta[:, 3:6], axis=1).astype(np.float32)

    rel_xy = np.full_like(teacher_xy, float(fallback_release_xy), dtype=np.float32)
    rel_z = np.full_like(teacher_z, float(fallback_release_z), dtype=np.float32)
    rel_yaw = np.full_like(teacher_yaw, float(fallback_release_yaw), dtype=np.float32)
    if "teacher_truth_handoff_ready" in data:
        teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32)
    elif "ready_to_close_target" in data:
        teacher_ready = np.asarray(data["ready_to_close_target"], dtype=np.float32)
    else:
        teacher_ready = np.zeros_like(teacher_xy, dtype=np.float32)
    return teacher_xy, teacher_z, teacher_yaw, teacher_ready, rel_xy, rel_z, rel_yaw, True


def _mark_proxy_ready_per_episode(
    keep: np.ndarray,
    episode_index: np.ndarray,
    rollout_step: np.ndarray,
    teacher_xy: np.ndarray,
    teacher_z: np.ndarray,
    teacher_yaw: np.ndarray,
    rel_xy: np.ndarray,
    rel_z: np.ndarray,
    rel_yaw: np.ndarray,
    ready_window_radius: int,
    very_near_window_radius: int,
):
    proxy_ready = np.zeros_like(teacher_xy, dtype=np.float32)
    proxy_very_near = np.zeros_like(teacher_xy, dtype=np.float32)
    for ep in np.unique(episode_index[keep]):
        m = keep & (episode_index == ep)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        score = (
            (teacher_xy[idx] / np.maximum(rel_xy[idx], 1e-6))
            + 0.5 * (teacher_z[idx] / np.maximum(rel_z[idx], 1e-6))
            + 0.5 * (teacher_yaw[idx] / np.maximum(rel_yaw[idx], 1e-6))
        )
        best_local = int(np.argmin(score))
        best_idx = int(idx[best_local])
        best_step = int(rollout_step[best_idx])
        steps = rollout_step[idx]
        proxy_ready[idx[np.abs(steps - best_step) <= int(ready_window_radius)]] = 1.0
        proxy_very_near[idx[np.abs(steps - best_step) <= int(very_near_window_radius)]] = 1.0
    return proxy_ready, proxy_very_near


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
    ap.add_argument("--ready_window_radius", type=int, default=2)
    ap.add_argument("--very_near_window_radius", type=int, default=6)
    ap.add_argument("--sample_weight_ready", type=float, default=2.5)
    ap.add_argument("--sample_weight_very_near", type=float, default=1.75)
    args = ap.parse_args()

    if not args.support_npz:
        raise RuntimeError("At least one --support_npz must be provided.")
    if args.source_name and len(args.source_name) != len(args.support_npz):
        raise RuntimeError("--source_name must match --support_npz count")
    if args.source_weight_mult and len(args.source_weight_mult) != len(args.support_npz):
        raise RuntimeError("--source_weight_mult must match --support_npz count")

    source_names = list(args.source_name or [Path(p).parent.name for p in args.support_npz])
    source_weights = list(args.source_weight_mult or [1.0] * len(args.support_npz))

    merged: dict[str, list[np.ndarray]] = {}
    source_summary = []

    for source_id, (support_npz, source_name, source_weight_mult) in enumerate(zip(args.support_npz, source_names, source_weights)):
        raw = np.load(support_npz, allow_pickle=False)
        data = {k: np.asarray(raw[k]) for k in raw.files}

        teacher_xy, teacher_z, teacher_yaw, teacher_ready, rel_xy, rel_z, rel_yaw, used_proxy_ready = _compute_teacher_metrics(
            data,
            float(args.fallback_release_xy),
            float(args.fallback_release_z),
            float(args.fallback_release_yaw),
        )

        phase_id = np.asarray(data.get("phase_id", np.ones((teacher_xy.shape[0],), dtype=np.int64)), dtype=np.int64)
        gripper_open = np.asarray(data.get("rollout_gripper_open", np.ones((teacher_xy.shape[0],), dtype=np.float32)), dtype=np.float32)
        has_object = np.asarray(data.get("has_object_in_hand", np.zeros((teacher_xy.shape[0],), dtype=np.float32)), dtype=np.float32)
        episode_index = np.asarray(
            data.get("episode_index", data.get("support_source_index", np.arange(teacher_xy.shape[0]))),
            dtype=np.int64,
        )
        rollout_step = np.asarray(data.get("rollout_step", data.get("step_idx", np.arange(teacher_xy.shape[0]))), dtype=np.int64)

        keep = (
            (phase_id == int(args.phase_id))
            & (gripper_open >= float(args.open_threshold))
            & (has_object <= 0.5)
            & np.isfinite(teacher_xy)
            & np.isfinite(teacher_z)
            & np.isfinite(teacher_yaw)
        )

        proxy_ready = np.zeros_like(teacher_ready, dtype=np.float32)
        proxy_very_near = np.zeros_like(teacher_ready, dtype=np.float32)
        if used_proxy_ready:
            proxy_ready, proxy_very_near = _mark_proxy_ready_per_episode(
                keep,
                episode_index,
                rollout_step,
                teacher_xy,
                teacher_z,
                teacher_yaw,
                rel_xy,
                rel_z,
                rel_yaw,
                int(args.ready_window_radius),
                int(args.very_near_window_radius),
            )
            teacher_ready = np.maximum(teacher_ready, proxy_ready).astype(np.float32)

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
        ready_support = (teacher_ready[keep] > 0.5).astype(np.float32)
        very_near = (
            (teacher_xy[keep] <= 1.5 * rel_xy[keep])
            & (teacher_z[keep] <= 1.5 * rel_z[keep])
            & (teacher_yaw[keep] <= 1.5 * rel_yaw[keep])
        ).astype(np.float32)
        very_near = np.maximum(very_near, proxy_very_near[keep]).astype(np.float32)
        broad_xy_recovery = (
            (teacher_z[keep] <= 1.5 * rel_z[keep])
            & (teacher_xy[keep] > rel_xy[keep])
            & (teacher_yaw[keep] <= 1.5 * rel_yaw[keep])
        ).astype(np.float32)
        near_xy_hard = (
            (teacher_z[keep] <= 1.2 * rel_z[keep])
            & (teacher_xy[keep] > rel_xy[keep])
            & (teacher_yaw[keep] <= 1.2 * rel_yaw[keep])
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

        sample_weight = np.ones((keep_idx.size,), dtype=np.float32)
        sample_weight += ready_support * float(args.sample_weight_ready - 1.0)
        sample_weight += very_near * float(args.sample_weight_very_near - 1.0)
        sample_weight *= float(source_weight_mult)

        proxy_delta_src = data.get(
            "proxy_current_delta_basin_target",
            data.get("current_delta_basin_target", data.get("teacher_current_delta_basin_target", data.get("target_delta_teacher"))),
        )
        if proxy_delta_src is None:
            raise RuntimeError("support npz missing proxy/current teacher delta fields needed for anchor dataset")
        proxy_delta = np.asarray(proxy_delta_src[keep], dtype=np.float32)
        source_out = {
            "front_rgb": np.asarray(data["front_rgb"][keep], dtype=np.uint8),
            "wrist_rgb": np.asarray(data["wrist_rgb"][keep], dtype=np.uint8),
            "wrist_depth": np.asarray(data["wrist_depth"][keep], dtype=np.float32),
            "proprio": np.asarray(data["proprio"][keep], dtype=np.float32),
            "gripper_context": np.asarray(data["gripper_context"][keep], dtype=np.float32),
            "proxy_current_delta_basin_target": proxy_delta,
            "current_dx_sign": np.asarray(data["current_dx_sign"][keep], dtype=np.int64),
            "current_dy_sign": np.asarray(data["current_dy_sign"][keep], dtype=np.int64),
            "current_dyaw_sign": np.asarray(data["current_dyaw_sign"][keep], dtype=np.int64),
            "basin_distance_bin": np.asarray(data["basin_distance_bin"][keep], dtype=np.int64),
            "substage_id": np.asarray(data.get("substage_id", np.zeros_like(episode_index))[keep], dtype=np.int64),
            "contact_state": np.asarray(data.get("contact_state", np.zeros_like(episode_index))[keep], dtype=np.int64),
            "stage_target_mode": np.asarray(data.get("stage_target_mode", np.zeros_like(episode_index))[keep], dtype=np.int64),
            "episode_index": np.asarray(episode_index[keep], dtype=np.int64),
            "teacher_xy_norm": teacher_xy_norm,
            "teacher_abs_z_norm": teacher_z_norm,
            "teacher_yaw_norm": teacher_yaw_norm,
            "teacher_metrics_norm": np.stack([teacher_xy_norm, teacher_z_norm, teacher_yaw_norm], axis=-1).astype(np.float32),
            "teacher_band_label": band_label.astype(np.int64),
            "teacher_truth_handoff_ready": ready_support.astype(np.float32),
            "teacher_truth_release_threshold_xy_error": np.asarray(rel_xy[keep], dtype=np.float32),
            "teacher_truth_release_threshold_abs_z_error": np.asarray(rel_z[keep], dtype=np.float32),
            "teacher_truth_release_threshold_yaw_error": np.asarray(rel_yaw[keep], dtype=np.float32),
            "runtime_handoff_metric_xy_error": np.full((keep_idx.size,), np.nan, dtype=np.float32),
            "runtime_handoff_metric_abs_z_error": np.full((keep_idx.size,), np.nan, dtype=np.float32),
            "runtime_handoff_metric_yaw_error": np.full((keep_idx.size,), np.nan, dtype=np.float32),
            "runtime_handoff_metric_valid": np.zeros((keep_idx.size,), dtype=np.float32),
            "runtime_handoff_ready": np.zeros((keep_idx.size,), dtype=np.float32),
            "runtime_handoff_ready_pred": np.zeros((keep_idx.size,), dtype=np.float32),
            "near_xy_hard": near_xy_hard.astype(np.float32),
            "broad_xy_recovery": broad_xy_recovery.astype(np.float32),
            "near_yaw_hard": near_yaw_hard.astype(np.float32),
            "near_coupled": near_coupled.astype(np.float32),
            "ready_support": ready_support.astype(np.float32),
            "disagreement": np.zeros((keep_idx.size,), dtype=np.float32),
            "sample_weight": sample_weight.astype(np.float32),
            "source_id": np.full((keep_idx.size,), source_id, dtype=np.int64),
            "source_name": np.full((keep_idx.size,), source_name, dtype="U128"),
        }

        for key, value in source_out.items():
            merged.setdefault(key, []).append(value)

        source_summary.append(
            {
                "source_id": int(source_id),
                "source_name": source_name,
                "support_npz": str(Path(support_npz).resolve()),
                "source_weight_mult": float(source_weight_mult),
                "used_proxy_ready": bool(used_proxy_ready),
                "rows": int(keep_idx.size),
                "episodes": int(np.unique(source_out["episode_index"]).size),
                "ready_support_pos": int(np.sum(source_out["ready_support"] > 0.5)),
                "very_near_rows": int(np.sum(very_near > 0.5)),
                "broad_xy_recovery_pos": int(np.sum(source_out["broad_xy_recovery"] > 0.5)),
            }
        )

    if not merged:
        raise RuntimeError("No valid rows selected for teacher-success anchor dataset.")

    out = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    meta = {
        "num_rows": int(out["front_rgb"].shape[0]),
        "teacher_ready_pos": int(np.sum(out["teacher_truth_handoff_ready"] > 0.5)),
        "very_near_rows": int(np.sum((out["teacher_band_label"] > 0))),
        "sources": source_summary,
    }
    meta_path = Path(args.meta_json) if args.meta_json else output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
