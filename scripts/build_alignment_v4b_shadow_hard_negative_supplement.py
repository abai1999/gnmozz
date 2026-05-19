#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
    "episode_index",
]


def _parse_csv_ints(text: str | None) -> set[int]:
    out: set[int] = set()
    if not text:
        return out
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def _yaw_bucket(yaw: np.ndarray, small_abs: float, large_abs: float) -> np.ndarray:
    out = np.zeros_like(yaw, dtype=np.int64)
    out[(yaw >= float(small_abs)) & (yaw < float(large_abs))] = 1
    out[(yaw >= float(large_abs))] = 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    ap.add_argument("--small_yaw_abs", type=float, default=0.02)
    ap.add_argument("--large_yaw_abs", type=float, default=0.08)
    ap.add_argument("--worse_weight", type=float, default=2.0)
    ap.add_argument("--better_weight", type=float, default=0.5)
    ap.add_argument("--hard_episode_indices", type=str, default="17")
    ap.add_argument("--hard_episode_weight", type=float, default=3.0)
    ap.add_argument("--require_changed", action="store_true", default=True)
    ap.add_argument("--shadow_yaw_probe_values", type=str, default="0.06,0.12")
    args = ap.parse_args()

    data_chunks = []
    for npz_path in args.support_npz:
        raw = np.load(npz_path, allow_pickle=False)
        data_chunks.append({k: np.asarray(raw[k]) for k in raw.files})
    keys = sorted(set().union(*(c.keys() for c in data_chunks)))
    data: dict[str, np.ndarray] = {}
    for key in keys:
        exemplar = next((c[key] for c in data_chunks if key in c), None)
        if exemplar is None:
            continue
        arrs = []
        for c in data_chunks:
            n_c = int(next(iter(c.values())).shape[0])
            if key in c and tuple(np.asarray(c[key]).shape[1:]) == tuple(exemplar.shape[1:]):
                arrs.append(np.asarray(c[key]))
            else:
                shape = (n_c,) + tuple(exemplar.shape[1:])
                if exemplar.dtype.kind in ("U", "S", "O"):
                    arrs.append(np.full(shape, "", dtype=exemplar.dtype))
                else:
                    arrs.append(np.zeros(shape, dtype=exemplar.dtype))
        data[key] = np.concatenate(arrs, axis=0)
    n = int(next(iter(data.values())).shape[0])

    gate_open = np.asarray(data.get("b2_candidate_shadow_gate_open", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    changed = np.asarray(data.get("b2_candidate_shadow_changed", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    regret_delta = np.asarray(data.get("b2_candidate_shadow_regret_delta", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32)
    cost_valid = np.isfinite(regret_delta)
    keep = gate_open & cost_valid
    if bool(args.require_changed):
        keep &= changed

    if not np.any(keep):
        raise RuntimeError("no shadow hard-negative rows survived filtering")

    idx = np.where(keep)[0]
    shadow_candidate_actions = np.asarray(data.get("b2_candidate_shadow_candidate_actions_local", []), dtype=np.float32)
    shadow_candidate_scores = np.asarray(data.get("b2_candidate_shadow_candidate_oracle_score", []), dtype=np.float32)
    if shadow_candidate_actions.size > 0 and shadow_candidate_actions.shape[0] == n:
        candidate_actions = shadow_candidate_actions[idx].astype(np.float32)
        oracle_scores = shadow_candidate_scores[idx].astype(np.float32)
        candidate_mask = np.asarray(
            data.get("b2_candidate_shadow_candidate_valid_mask", np.ones(shadow_candidate_actions.shape[:2], dtype=np.float32)),
            dtype=np.float32,
        )[idx]
    else:
        candidate_actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)[idx]
        oracle_scores = np.asarray(data["candidate_oracle_score"], dtype=np.float32)[idx]
        candidate_mask = np.asarray(data.get("candidate_mask", np.ones(candidate_actions.shape[:2], dtype=np.float32)), dtype=np.float32)[idx]
        probe_values = [abs(float(v)) for v in str(args.shadow_yaw_probe_values).split(",") if str(v).strip()]
        if probe_values:
            probe_actions = []
            for mag in probe_values:
                if mag <= 0.0:
                    continue
                pos = np.zeros((6,), dtype=np.float32)
                neg = np.zeros((6,), dtype=np.float32)
                pos[5] = mag
                neg[5] = -mag
                probe_actions.extend([pos, neg])
            if probe_actions:
                probe_np = np.stack(probe_actions, axis=0).astype(np.float32)
                probe_np = np.broadcast_to(probe_np[None, :, :], (candidate_actions.shape[0], probe_np.shape[0], 6)).copy()
                candidate_actions = np.concatenate([candidate_actions, probe_np], axis=1)
                probe_scores = np.full((oracle_scores.shape[0], probe_np.shape[1]), np.nan, dtype=np.float32)
                candidate_mask = np.concatenate([candidate_mask, np.ones((candidate_mask.shape[0], probe_np.shape[1]), dtype=np.float32)], axis=1)
                oracle_scores = np.concatenate([oracle_scores, probe_scores], axis=1)
    baseline_idx = np.asarray(data.get("b2_candidate_shadow_baseline_index", data.get("runtime_selected_candidate_index")), dtype=np.int64)[idx]
    pred_idx = np.asarray(data.get("b2_candidate_shadow_pred_index", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)[idx]
    best_idx = np.asarray(data.get("b2_candidate_shadow_best_index", data.get("oracle_candidate_index", np.full((n,), -1, dtype=np.int64))), dtype=np.int64)[idx]
    worse = regret_delta[idx] < -1e-6
    better = regret_delta[idx] > 1e-6
    episode_index = np.asarray(data.get("episode_index", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)[idx]
    hard_episode_indices = _parse_csv_ints(args.hard_episode_indices)
    hard_episode = np.isin(episode_index, np.asarray(sorted(hard_episode_indices), dtype=np.int64)) if hard_episode_indices else np.zeros_like(worse)

    teacher_delta = np.asarray(
        data.get("teacher_current_delta_basin_target", data.get("proxy_current_delta_basin_target")),
        dtype=np.float32,
    )[idx]
    teacher_delta_finite = np.isfinite(teacher_delta)
    if teacher_delta.ndim >= 2:
        teacher_valid = np.all(teacher_delta_finite, axis=1)
    else:
        teacher_valid = teacher_delta_finite
    if not np.all(teacher_valid):
        candidate_actions = candidate_actions[teacher_valid]
        oracle_scores = oracle_scores[teacher_valid]
        candidate_mask = candidate_mask[teacher_valid]
        baseline_idx = baseline_idx[teacher_valid]
        pred_idx = pred_idx[teacher_valid]
        best_idx = best_idx[teacher_valid]
        worse = worse[teacher_valid]
        better = better[teacher_valid]
        idx = idx[teacher_valid]
        episode_index = episode_index[teacher_valid]
        hard_episode = hard_episode[teacher_valid]
        teacher_delta = teacher_delta[teacher_valid]
        if idx.size == 0:
            raise RuntimeError("no valid shadow hard-negative rows after dropping non-finite teacher rows")
    handoff_xy = np.asarray(data.get("runtime_handoff_release_threshold_xy_error", np.full((n,), 0.006, dtype=np.float32)), dtype=np.float32)[idx]
    handoff_z = np.asarray(data.get("runtime_handoff_release_threshold_abs_z_error", np.full((n,), 0.005, dtype=np.float32)), dtype=np.float32)[idx]
    handoff_yaw = np.asarray(data.get("runtime_handoff_release_threshold_yaw_error", np.full((n,), 0.12, dtype=np.float32)), dtype=np.float32)[idx]
    handoff_valid = np.isfinite(handoff_xy) & np.isfinite(handoff_z) & np.isfinite(handoff_yaw)

    num_cands = candidate_actions.shape[1]
    valid_indices = (
        (baseline_idx >= 0) & (baseline_idx < num_cands)
        & (pred_idx >= 0) & (pred_idx < num_cands)
        & (best_idx >= 0) & (best_idx < num_cands)
        & handoff_valid
    )
    candidate_actions = candidate_actions[valid_indices]
    oracle_scores = oracle_scores[valid_indices]
    candidate_mask = candidate_mask[valid_indices]
    baseline_idx = baseline_idx[valid_indices]
    pred_idx = pred_idx[valid_indices]
    best_idx = best_idx[valid_indices]
    worse = worse[valid_indices]
    better = better[valid_indices]
    idx = idx[valid_indices]
    hard_episode = hard_episode[valid_indices]
    teacher_delta = teacher_delta[valid_indices]
    handoff_xy = handoff_xy[valid_indices]
    handoff_z = handoff_z[valid_indices]
    handoff_yaw = handoff_yaw[valid_indices]
    if idx.size == 0:
        raise RuntimeError("no valid shadow hard-negative rows after index filtering")

    weights = np.stack(
        [
            1.0 / np.maximum(handoff_xy, 1e-4),
            1.0 / np.maximum(handoff_xy, 1e-4),
            1.0 / np.maximum(handoff_z, 1e-4),
            np.zeros_like(handoff_xy),
            np.zeros_like(handoff_xy),
            1.0 / np.maximum(handoff_yaw, 1e-4),
        ],
        axis=1,
    ).astype(np.float32)
    residual = teacher_delta[:, None, :6] - candidate_actions[:, :, :6]
    cost = np.linalg.norm(residual * weights[:, None, :], axis=2).astype(np.float32)
    oracle_scores = -cost
    best_idx = np.argmin(cost, axis=1).astype(np.int64)

    row = np.arange(idx.size, dtype=np.int64)
    pred_yaw = np.abs(candidate_actions[row, np.clip(pred_idx, 0, candidate_actions.shape[1] - 1), 5])
    baseline_yaw = np.abs(candidate_actions[row, np.clip(baseline_idx, 0, candidate_actions.shape[1] - 1), 5])
    oracle_yaw = np.abs(candidate_actions[row, np.clip(best_idx, 0, candidate_actions.shape[1] - 1), 5])
    has_yaw_candidate = np.any(np.abs(candidate_actions[:, :, 5]) > float(args.keep_yaw_abs), axis=1)
    pred_yaw_bucket = _yaw_bucket(pred_yaw, float(args.small_yaw_abs), float(args.large_yaw_abs))
    baseline_yaw_bucket = _yaw_bucket(baseline_yaw, float(args.small_yaw_abs), float(args.large_yaw_abs))
    oracle_yaw_bucket = _yaw_bucket(oracle_yaw, float(args.small_yaw_abs), float(args.large_yaw_abs))
    best_yaw_bucket = oracle_yaw_bucket.copy()

    out = {}
    for key in INPUT_KEYS:
        if key in data:
            out[key] = np.asarray(data[key])[idx]
    out["candidate_actions_local"] = candidate_actions.astype(np.float32)
    out["candidate_mask"] = candidate_mask.astype(np.float32)
    out["candidate_oracle_score"] = oracle_scores.astype(np.float32)
    out["candidate_best_index"] = best_idx.astype(np.int64)
    out["candidate_baseline_index"] = baseline_idx.astype(np.int64)
    out["candidate_bad_index"] = pred_idx.astype(np.int64)
    out["sample_weight"] = np.where(worse, float(args.worse_weight), np.where(better, float(args.better_weight), 1.0)).astype(np.float32)
    out["is_shadow_hard_negative"] = worse.astype(np.float32)
    out["pred_worse_than_baseline"] = worse.astype(np.float32)
    out["pred_better_than_baseline"] = better.astype(np.float32)
    out["pred_has_yaw"] = (pred_yaw > float(args.keep_yaw_abs)).astype(np.float32)
    out["oracle_has_yaw"] = (oracle_yaw > float(args.keep_yaw_abs)).astype(np.float32)
    out["baseline_has_yaw"] = (baseline_yaw > float(args.keep_yaw_abs)).astype(np.float32)
    out["pred_large_yaw_negative"] = (
        worse
        & (pred_yaw >= float(args.large_yaw_abs))
        & (oracle_yaw < float(args.small_yaw_abs))
    ).astype(np.float32)
    out["pred_large_yaw_positive"] = (
        better
        & (pred_yaw >= float(args.large_yaw_abs))
    ).astype(np.float32)
    out["pred_small_yaw_positive"] = (
        better
        & (pred_yaw >= float(args.small_yaw_abs))
        & (pred_yaw < float(args.large_yaw_abs))
    ).astype(np.float32)
    out["pred_yaw_bucket"] = pred_yaw_bucket.astype(np.int64)
    out["baseline_yaw_bucket"] = baseline_yaw_bucket.astype(np.int64)
    out["oracle_yaw_bucket"] = oracle_yaw_bucket.astype(np.int64)
    out["best_yaw_bucket"] = best_yaw_bucket.astype(np.int64)
    out["hard_episode_negative"] = (worse & hard_episode).astype(np.float32)
    out["candidate_teacher_norm"] = np.stack(
        [
            np.asarray(data.get("teacher_truth_handoff_metric_xy_error", np.full((n,), np.nan)), dtype=np.float32)[idx]
            / np.maximum(np.asarray(data.get("teacher_truth_handoff_release_threshold_xy_error", np.full((n,), 0.0085)), dtype=np.float32)[idx], 1e-6),
            np.asarray(data.get("teacher_truth_handoff_metric_abs_z_error", np.full((n,), np.nan)), dtype=np.float32)[idx]
            / np.maximum(np.asarray(data.get("teacher_truth_handoff_release_threshold_abs_z_error", np.full((n,), 0.0035)), dtype=np.float32)[idx], 1e-6),
            np.asarray(data.get("teacher_truth_handoff_metric_yaw_error", np.full((n,), np.nan)), dtype=np.float32)[idx]
            / np.maximum(np.asarray(data.get("teacher_truth_handoff_release_threshold_yaw_error", np.full((n,), 0.1243404)), dtype=np.float32)[idx], 1e-6),
        ],
        axis=1,
    ).astype(np.float32)
    out["candidate_scope_size"] = np.sum(out["candidate_mask"] > 0.5, axis=1).astype(np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "alignment_v4b_shadow_hard_negative_supplement.npz"
    np.savez_compressed(out_path, **out)

    report = {
        "rows": int(idx.size),
        "worse_rows": int(np.sum(worse)),
        "better_rows": int(np.sum(better)),
        "hard_episode_rows": int(np.sum(hard_episode)),
        "mean_regret_delta": float(np.nanmean(regret_delta[idx])),
        "pred_has_yaw_rate": float(np.mean(out["pred_has_yaw"])),
        "oracle_has_yaw_rate": float(np.mean(out["oracle_has_yaw"])),
        "baseline_has_yaw_rate": float(np.mean(out["baseline_has_yaw"])),
        "pred_large_yaw_negative_rate": float(np.mean(out["pred_large_yaw_negative"])),
        "pred_large_yaw_positive_rate": float(np.mean(out["pred_large_yaw_positive"])),
        "pred_small_yaw_positive_rate": float(np.mean(out["pred_small_yaw_positive"])),
        "pred_large_yaw_bucket_rate": float(np.mean(pred_yaw_bucket >= 2)),
        "pred_small_yaw_bucket_rate": float(np.mean(pred_yaw_bucket == 1)),
        "rows_with_any_yaw_candidate": float(np.mean(has_yaw_candidate)),
        "output_npz": str(out_path),
    }
    (out_dir / "alignment_v4b_shadow_hard_negative_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
