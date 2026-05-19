#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from prismatic.robot.stage_target_provider import wrap_yaw_to_symmetry


INPUT_KEYS = [
    "front_rgb",
    "wrist_rgb",
    "wrist_depth",
    "proprio",
    "gripper_context",
    "proxy_current_delta_basin_target",
    "current_dx_sign",
    "current_dy_sign",
    "current_dyaw_sign",
    "basin_distance_bin",
    "substage_id",
    "contact_state",
    "stage_target_mode",
]


def _concat(paths: list[Path]) -> dict[str, np.ndarray]:
    chunks: list[dict[str, np.ndarray]] = []
    for path in paths:
        raw = np.load(path, allow_pickle=False)
        chunks.append({k: np.asarray(raw[k]) for k in raw.files})
    keys = sorted(set().union(*(c.keys() for c in chunks)))
    out: dict[str, np.ndarray] = {}
    for key in keys:
        exemplar = next((c[key] for c in chunks if key in c), None)
        if exemplar is None:
            continue
        arrs = []
        for c in chunks:
            n = int(next(iter(c.values())).shape[0])
            if key in c and tuple(np.asarray(c[key]).shape[1:]) == tuple(exemplar.shape[1:]):
                arrs.append(np.asarray(c[key]))
            else:
                shape = (n,) + tuple(exemplar.shape[1:])
                if exemplar.dtype.kind in ("U", "S", "O"):
                    arrs.append(np.full(shape, "", dtype=exemplar.dtype))
                else:
                    arrs.append(np.zeros(shape, dtype=exemplar.dtype))
        out[key] = np.concatenate(arrs, axis=0)
    return out


def _safe_arr(data: dict[str, np.ndarray], key: str, fallback: np.ndarray) -> np.ndarray:
    return np.asarray(data.get(key, fallback))


def _safe_threshold(values: np.ndarray, fallback: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    bad = ~np.isfinite(out) | (out <= 0.0)
    out[bad] = float(fallback)
    return out


def _has_v3_alignment_schema(data: dict[str, np.ndarray]) -> bool:
    return "alignment_v3_teacher_delta_basin_target" in data


def _weighted_cost(delta: np.ndarray, xy_thr: np.ndarray, z_thr: np.ndarray, yaw_thr: np.ndarray, yaw_period: float) -> np.ndarray:
    xy = np.linalg.norm(delta[:, :2], axis=1) / np.maximum(xy_thr, 1e-6)
    z = np.abs(delta[:, 2]) / np.maximum(z_thr, 1e-6)
    yaw_raw = np.asarray(delta[:, 5], dtype=np.float32)
    if yaw_period > 0.0:
        yaw_raw = np.asarray([wrap_yaw_to_symmetry(float(v), yaw_period) for v in yaw_raw], dtype=np.float32)
    yaw = np.abs(yaw_raw) / np.maximum(yaw_thr, 1e-6)
    return 0.45 * xy + 0.30 * z + 0.25 * yaw


def _clip_residual(residual6: np.ndarray, max_xyz: float, max_yaw: float, yaw_period: float) -> np.ndarray:
    out = np.asarray(residual6, dtype=np.float32).copy()
    pos_norm = np.linalg.norm(out[:, :3], axis=1)
    scale = np.minimum(1.0, float(max_xyz) / np.maximum(pos_norm, 1e-8))
    out[:, :3] *= scale[:, None]
    if yaw_period > 0.0:
        out[:, 5] = np.asarray([wrap_yaw_to_symmetry(float(v), yaw_period) for v in out[:, 5]], dtype=np.float32)
    out[:, 5] = np.clip(out[:, 5], -float(max_yaw), float(max_yaw))
    out[:, 3:5] = 0.0
    return out


def _make_residual_target(
    data: dict[str, np.ndarray],
    current_delta: np.ndarray,
    max_xyz: float,
    max_yaw: float,
    yaw_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(current_delta.shape[0])
    residual = np.zeros((n, 6), dtype=np.float32)
    source_rank = np.zeros((n,), dtype=np.int64)
    for rank, key in enumerate(("residual_label_local", "oracle_action_local", "teacher_current_delta_basin_target", "target_delta_teacher"), start=1):
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 6:
            continue
        candidate = arr[:, :6].copy()
        # Absolute teacher deltas must be converted into a local residual action:
        # after_delta = current_delta - residual_delta ~= teacher_delta.
        if key in ("teacher_current_delta_basin_target", "target_delta_teacher"):
            candidate = current_delta - candidate
        valid = (
            np.all(np.isfinite(candidate[:, :6]), axis=1)
            & (np.linalg.norm(candidate[:, [0, 1, 2, 5]], axis=1) > 1e-8)
            & (source_rank == 0)
        )
        residual[valid] = candidate[valid, :6]
        source_rank[valid] = rank
    if _has_v3_alignment_schema(data):
        teacher_delta = np.asarray(data["alignment_v3_teacher_delta_basin_target"], dtype=np.float32)
        candidate = current_delta - teacher_delta[:, :6]
        valid = (
            np.all(np.isfinite(candidate[:, :6]), axis=1)
            & np.all(np.isfinite(teacher_delta[:, :6]), axis=1)
            & (np.linalg.norm(candidate[:, [0, 1, 2, 5]], axis=1) > 1e-8)
            & (source_rank == 0)
        )
        residual[valid] = candidate[valid, :6]
        source_rank[valid] = 5
    residual = _clip_residual(residual, max_xyz=max_xyz, max_yaw=max_yaw, yaw_period=yaw_period)
    valid = source_rank > 0
    return residual, valid.astype(np.float32), source_rank.astype(np.int64)


def _temporal_summary(data: dict[str, np.ndarray], horizon: int) -> np.ndarray:
    n = int(data["proxy_current_delta_basin_target"].shape[0])
    eps = _safe_arr(data, "episode_index", np.arange(n, dtype=np.int64)).astype(np.int64)
    step = _safe_arr(data, "step_index", np.arange(n, dtype=np.int64)).astype(np.int64)
    delta = np.asarray(data["proxy_current_delta_basin_target"], dtype=np.float32)
    action = np.asarray(data.get("executed_action_local", np.zeros((n, 6), dtype=np.float32)), dtype=np.float32)
    out = np.zeros((n, 32), dtype=np.float32)
    order = np.lexsort((step, eps))
    prev_by_ep: dict[int, list[int]] = {}
    for idx in order.tolist():
        ep = int(eps[idx])
        hist = prev_by_ep.setdefault(ep, [])
        if hist:
            prev = hist[-1]
            out[idx, 0:6] = delta[idx] - delta[prev]
            out[idx, 6:12] = action[prev]
        if len(hist) >= horizon:
            prev = hist[-horizon]
            out[idx, 12:18] = delta[idx] - delta[prev]
            out[idx, 18:24] = np.mean(action[hist[-horizon:]], axis=0)
        out[idx, 24] = min(float(len(hist)), float(horizon)) / max(float(horizon), 1.0)
        hist.append(idx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_residual_xyz", type=float, default=0.006)
    ap.add_argument("--max_residual_yaw", type=float, default=0.03)
    ap.add_argument("--yaw_symmetry_period", type=float, default=1.5707963267948966)
    ap.add_argument("--improvement_margin", type=float, default=0.05)
    ap.add_argument("--temporal_summary_horizon", type=int, default=3)
    ap.add_argument("--allowed_phase_id", action="append", default=["1"])
    ap.add_argument("--require_focus_mask", action="store_true", default=True)
    ap.add_argument("--no_require_focus_mask", dest="require_focus_mask", action="store_false")
    ap.add_argument("--require_closeability_positive", action="store_true", default=True)
    ap.add_argument("--no_require_closeability_positive", dest="require_closeability_positive", action="store_false")
    ap.add_argument("--max_teacher_xy_norm", type=float, default=6.0)
    ap.add_argument("--max_teacher_z_norm", type=float, default=12.0)
    ap.add_argument("--max_teacher_yaw_norm", type=float, default=2.5)
    ap.add_argument("--max_unclipped_residual_xyz_norm", type=float, default=0.03)
    ap.add_argument("--max_unclipped_residual_yaw_abs", type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _concat([Path(p) for p in args.input_npz])
    n = int(data["proxy_current_delta_basin_target"].shape[0])
    zeros = np.zeros((n,), dtype=np.float32)
    delta = np.asarray(data["proxy_current_delta_basin_target"], dtype=np.float32)
    residual6, residual_valid, residual_source_rank = _make_residual_target(
        data,
        current_delta=delta,
        max_xyz=float(args.max_residual_xyz),
        max_yaw=float(args.max_residual_yaw),
        yaw_period=float(args.yaw_symmetry_period),
    )
    raw_residual6 = np.zeros_like(delta, dtype=np.float32)
    if _has_v3_alignment_schema(data):
        raw_residual6[:, :6] = delta[:, :6] - np.asarray(data["alignment_v3_teacher_delta_basin_target"], dtype=np.float32)[:, :6]
    else:
        raw_residual6[:, :6] = residual6[:, :6]

    if _has_v3_alignment_schema(data):
        base_mask = _safe_arr(data, "alignment_v3_corrective_mask", np.ones((n,), dtype=np.float32)).astype(np.float32) > 0.5
        focus_mask = _safe_arr(data, "alignment_v3_corrective_focus_mask", zeros).astype(np.float32) > 0.5
        closeability_raw = _safe_arr(data, "alignment_v3_closeability_label", zeros).astype(np.float32)
        teacher_metrics_norm = np.asarray(data.get("teacher_metrics_norm", np.zeros((n, 3), dtype=np.float32)), dtype=np.float32)
        ready = _safe_arr(data, "teacher_truth_handoff_ready", zeros).astype(np.float32)
        xy_thr = np.full((n,), 0.0085, dtype=np.float32)
        z_thr = np.full((n,), 0.0035, dtype=np.float32)
        yaw_thr = np.full((n,), 0.1243404, dtype=np.float32)
        strict_mask = np.ones((n,), dtype=np.bool_)
        if bool(args.require_focus_mask):
            strict_mask &= focus_mask
        if bool(args.require_closeability_positive):
            strict_mask &= closeability_raw > 0.5
        strict_mask &= np.isfinite(teacher_metrics_norm[:, 0]) & (teacher_metrics_norm[:, 0] <= float(args.max_teacher_xy_norm))
        strict_mask &= np.isfinite(teacher_metrics_norm[:, 1]) & (teacher_metrics_norm[:, 1] <= float(args.max_teacher_z_norm))
        strict_mask &= np.isfinite(teacher_metrics_norm[:, 2]) & (teacher_metrics_norm[:, 2] <= float(args.max_teacher_yaw_norm))
        raw_xyz_norm = np.linalg.norm(raw_residual6[:, :3], axis=1)
        raw_yaw_abs = np.abs(
            np.asarray(
                [wrap_yaw_to_symmetry(float(v), float(args.yaw_symmetry_period)) for v in raw_residual6[:, 5].tolist()],
                dtype=np.float32,
            )
        )
        strict_mask &= np.isfinite(raw_xyz_norm) & (raw_xyz_norm <= float(args.max_unclipped_residual_xyz_norm))
        strict_mask &= np.isfinite(raw_yaw_abs) & (raw_yaw_abs <= float(args.max_unclipped_residual_yaw_abs))
        base_mask &= strict_mask
    else:
        phase = _safe_arr(data, "phase_id", np.zeros((n,), dtype=np.int64)).astype(np.int64)
        allowed = {int(x) for item in args.allowed_phase_id for x in str(item).split(",") if x.strip()}
        role = _safe_arr(data, "handoff_target_role", np.full((n,), "", dtype="U32")).astype(str)
        provider = _safe_arr(data, "target_provider_source", np.full((n,), "", dtype="U96")).astype(str)
        gripper_open = _safe_arr(data, "rollout_gripper_open", np.ones((n,), dtype=np.float32)).astype(np.float32)
        metric_valid = _safe_arr(data, "runtime_handoff_metric_valid", np.ones((n,), dtype=np.float32)).astype(np.float32)
        close_like = (
            np.isin(role, ["pregrasp_close", "close", "commit_close"])
            | (np.char.find(provider, "canonical_close_orientation_contract") >= 0)
            | (np.char.find(provider, "teacher_motion") >= 0)
        )
        base_mask = np.isin(phase, sorted(allowed)) & close_like & (gripper_open >= 0.5) & (metric_valid >= 0.0)
        focus_mask = base_mask.copy()
        xy_thr = _safe_threshold(_safe_arr(data, "teacher_truth_handoff_release_threshold_xy_error", np.full((n,), 0.0085)), 0.0085)
        z_thr = _safe_threshold(_safe_arr(data, "teacher_truth_handoff_release_threshold_abs_z_error", np.full((n,), 0.0035)), 0.0035)
        yaw_thr = _safe_threshold(_safe_arr(data, "teacher_truth_handoff_release_threshold_yaw_error", np.full((n,), 0.1243404)), 0.1243404)
        ready = _safe_arr(data, "teacher_truth_handoff_ready", zeros).astype(np.float32)
        closeability_raw = np.zeros((n,), dtype=np.float32)
        teacher_metrics_norm = np.stack(
            [
                _safe_arr(data, "teacher_truth_handoff_metric_xy_error", np.full((n,), np.nan)) / xy_thr,
                _safe_arr(data, "teacher_truth_handoff_metric_abs_z_error", np.full((n,), np.nan)) / z_thr,
                _safe_arr(data, "teacher_truth_handoff_metric_yaw_error", np.full((n,), np.nan)) / yaw_thr,
            ],
            axis=1,
        ).astype(np.float32)

    cost_before = _weighted_cost(delta, xy_thr, z_thr, yaw_thr, float(args.yaw_symmetry_period))
    delta_after = delta.copy()
    delta_after[:, :3] -= residual6[:, :3]
    delta_after[:, 5] -= residual6[:, 5]
    cost_after = _weighted_cost(delta_after, xy_thr, z_thr, yaw_thr, float(args.yaw_symmetry_period))
    improvement = cost_before - cost_after
    improvement_label = (improvement >= float(args.improvement_margin)).astype(np.float32)

    keep = base_mask & (residual_valid > 0.5) & np.isfinite(cost_before) & np.isfinite(cost_after)
    idx = np.where(keep)[0]
    if idx.size == 0:
        raise RuntimeError("no v4 residual rows survived filtering")

    out: dict[str, np.ndarray] = {}
    for key in INPUT_KEYS:
        if key in data:
            out[key] = np.asarray(data[key])[idx]
    if "proxy_current_delta_basin_target" not in out:
        out["proxy_current_delta_basin_target"] = delta[idx]
    out["temporal_action_summary"] = _temporal_summary(data, int(args.temporal_summary_horizon))[idx]
    out["episode_index"] = _safe_arr(data, "episode_index", np.arange(n, dtype=np.int64)).astype(np.int64)[idx]
    out["step_index"] = _safe_arr(data, "step_index", np.arange(n, dtype=np.int64)).astype(np.int64)[idx]
    out["alignment_v4_residual_target"] = residual6[idx][:, [0, 1, 2, 5]].astype(np.float32)
    out["alignment_v4_residual_mask"] = np.ones((idx.size,), dtype=np.float32)
    out["alignment_v4_cost_before"] = cost_before[idx].astype(np.float32)
    out["alignment_v4_cost_after_target"] = cost_after[idx].astype(np.float32)
    out["alignment_v4_improvement"] = improvement[idx].astype(np.float32)
    out["alignment_v4_improvement_label"] = improvement_label[idx].astype(np.float32)
    closeable = closeability_raw if _has_v3_alignment_schema(data) else ((cost_before <= 1.0) | (ready > 0.5)).astype(np.float32)
    if _has_v3_alignment_schema(data):
        confidence_gate = base_mask.astype(np.float32) * (closeable > 0.5).astype(np.float32)
        confidence_target = np.clip(improvement, 0.0, 1.0) * confidence_gate
    else:
        confidence_target = np.clip(improvement, 0.0, 1.0)
    out["alignment_v4_residual_confidence_target"] = confidence_target[idx].astype(np.float32)
    out["alignment_v4_closeability_label"] = closeable[idx].astype(np.float32)
    out["alignment_v4_progress_label"] = _safe_arr(data, "alignment_v2_progress_label", zeros).astype(np.float32)[idx]
    out["alignment_v4_progress_mask"] = _safe_arr(data, "alignment_v2_progress_mask", zeros).astype(np.float32)[idx]
    out["alignment_v4_focus_mask"] = focus_mask[idx].astype(np.float32)
    out["alignment_v4_residual_source_rank"] = residual_source_rank[idx].astype(np.int64)
    out["teacher_metrics_norm"] = teacher_metrics_norm[idx].astype(np.float32)
    out["teacher_truth_handoff_ready"] = ready[idx]
    sample_weight = (1.0 + 2.0 * improvement_label[idx] + 1.0 * closeable[idx] + 1.0 * focus_mask[idx].astype(np.float32))
    if "sample_weight" in data:
        sample_weight = sample_weight * np.clip(np.asarray(data["sample_weight"], dtype=np.float32)[idx], 0.25, 8.0)
    out["sample_weight"] = sample_weight.astype(np.float32)
    out["source_name"] = _safe_arr(data, "target_provider_source", np.full((n,), "unknown", dtype="U96"))[idx].astype("U96")

    output_npz = out_dir / "alignment_v4_residual_dataset.npz"
    np.savez_compressed(output_npz, **out)
    report = {
        "source_mode": "alignment_v3_dataset" if _has_v3_alignment_schema(data) else "support_rows",
        "rows": int(idx.size),
        "input_rows": int(n),
        "episodes": int(np.unique(out["episode_index"]).size),
        "improvement_positive_rows": int(np.sum(out["alignment_v4_improvement_label"] > 0.5)),
        "closeability_positive_rows": int(np.sum(out["alignment_v4_closeability_label"] > 0.5)),
        "focus_rows": int(np.sum(out["alignment_v4_focus_mask"] > 0.5)),
        "cost_before_mean": float(np.mean(out["alignment_v4_cost_before"])),
        "target_improvement_rate": float(np.mean(out["alignment_v4_improvement"] > 0.0)),
        "residual_xyz_norm_mean": float(np.mean(np.linalg.norm(out["alignment_v4_residual_target"][:, :3], axis=1))),
        "residual_yaw_abs_mean": float(np.mean(np.abs(out["alignment_v4_residual_target"][:, 3]))),
        "raw_residual_xyz_norm_mean": float(np.mean(np.linalg.norm(raw_residual6[idx][:, :3], axis=1))),
        "raw_residual_xyz_norm_p90": float(np.percentile(np.linalg.norm(raw_residual6[idx][:, :3], axis=1), 90.0)),
        "raw_residual_yaw_abs_mean": float(np.mean(np.abs(raw_residual6[idx][:, 5]))),
        "clipped_xyz_fraction": float(
            np.mean(np.linalg.norm(raw_residual6[idx][:, :3], axis=1) > float(args.max_residual_xyz))
        ),
        "strict_filter_config": {
            "require_focus_mask": bool(args.require_focus_mask),
            "require_closeability_positive": bool(args.require_closeability_positive),
            "max_teacher_xy_norm": float(args.max_teacher_xy_norm),
            "max_teacher_z_norm": float(args.max_teacher_z_norm),
            "max_teacher_yaw_norm": float(args.max_teacher_yaw_norm),
            "max_unclipped_residual_xyz_norm": float(args.max_unclipped_residual_xyz_norm),
            "max_unclipped_residual_yaw_abs": float(args.max_unclipped_residual_yaw_abs),
        },
        "output_npz": str(output_npz),
    }
    (out_dir / "alignment_v4_dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
