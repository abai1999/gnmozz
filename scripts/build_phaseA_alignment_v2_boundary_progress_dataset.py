#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


INPUT_KEYS = [
    "front_rgb",
    "wrist_rgb",
    "wrist_depth",
    "proprio",
    "gripper_context",
    "current_dx_sign",
    "current_dy_sign",
    "current_dyaw_sign",
    "basin_distance_bin",
    "substage_id",
    "contact_state",
    "stage_target_mode",
]


def _concat_inputs(paths: list[Path]) -> dict[str, np.ndarray]:
    chunks = []
    for path in paths:
        raw = np.load(path, allow_pickle=False)
        chunks.append({k: np.asarray(raw[k]) for k in raw.files})
    if not chunks:
        raise RuntimeError("no input support npz files provided")
    keys = list(chunks[0].keys())
    out = {}
    for key in keys:
        out[key] = np.concatenate([c[key] for c in chunks], axis=0)
    return out


def _safe_threshold(values: np.ndarray, fallback: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    bad = ~np.isfinite(out) | (out <= 0.0)
    out[bad] = float(fallback)
    return out


def _percentiles(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def _band_label(norms: np.ndarray, ready: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    max_norm = np.max(norms, axis=1)
    out = np.zeros((norms.shape[0],), dtype=np.int64)
    out[max_norm <= 3.0] = 1
    out[max_norm <= 1.5] = 2
    out[(ready > 0.5) | (counterfactual > 0.5)] = 2
    return out


def _alignment_v2_band_label_4(norms: np.ndarray, ready: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    max_norm = np.max(norms, axis=1)
    out = np.zeros((norms.shape[0],), dtype=np.int64)
    out[max_norm <= 3.0] = 1
    out[max_norm <= 1.5] = 2
    out[(ready > 0.5) | (counterfactual > 0.5)] = 3
    return out


def _bucket_for_norms(
    xy: float,
    z: float,
    yaw: float,
    ready: bool,
    strict_near: float,
    broad_near: float,
    far_min: float,
) -> str | None:
    if ready:
        return "teacher_ready_anchor"
    norms = np.asarray([xy, z, yaw], dtype=np.float32)
    near_strict = norms <= strict_near
    near_broad = norms <= broad_near
    over = norms > 1.0
    if bool(over[0] and near_strict[1] and near_strict[2]):
        return "xy_block_boundary"
    if bool(over[1] and near_strict[0] and near_strict[2]):
        return "z_block_boundary"
    if bool(over[2] and near_strict[0] and near_strict[1]):
        return "yaw_block_boundary"
    if int(np.sum(near_broad)) >= 2 and float(np.max(norms)) > 1.0:
        return "multi_axis_boundary"
    if float(np.max(norms)) >= far_min:
        return "far_negative"
    return None


def _row_priority(bucket: str, norms: np.ndarray, step: int) -> tuple[float, float]:
    max_norm = float(np.max(norms))
    if bucket == "far_negative":
        return (float(step), -max_norm)
    if bucket == "xy_block_boundary":
        return (abs(float(norms[0]) - 1.0), max_norm)
    if bucket == "z_block_boundary":
        return (abs(float(norms[1]) - 1.0), max_norm)
    if bucket == "yaw_block_boundary":
        return (abs(float(norms[2]) - 1.0), max_norm)
    return (abs(max_norm - 1.0), float(step))


def _select_rows(rows_by_ep: dict[int, list[dict]], caps: dict[str, int], seed: int) -> list[dict]:
    selected = []
    for ep in sorted(rows_by_ep):
        rows = rows_by_ep[ep]
        by_bucket: dict[str, list[dict]] = {}
        for row in rows:
            by_bucket.setdefault(row["bucket"], []).append(row)
        for bucket, bucket_rows in by_bucket.items():
            cap = int(caps.get(bucket, caps.get("default", 64)))
            ordered = sorted(bucket_rows, key=lambda r: r["priority"])
            if len(ordered) > cap and bucket == "far_negative":
                rng = random.Random(seed + ep)
                ordered = rng.sample(ordered, cap)
            else:
                ordered = ordered[:cap]
            selected.extend(ordered)
    return selected


def _source_summary(source_name: np.ndarray, episode_index: np.ndarray, ready: np.ndarray) -> dict:
    out = {}
    names = source_name.astype(str)
    for name in sorted(np.unique(names).tolist()):
        mask = names == name
        out[name] = {
            "rows": int(np.sum(mask)),
            "episodes": int(np.unique(episode_index[mask]).size),
            "teacher_ready_rows": int(np.sum(ready[mask] > 0.5)),
        }
    return out


def _subset_rows(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] if np.asarray(v).shape[:1] == mask.shape[:1] else v for k, v in data.items()}


def _temporal_action_summary(
    *,
    raw: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    keep: np.ndarray,
    episode_index: np.ndarray,
    rollout_step: np.ndarray,
    proxy: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Build runtime-safe temporal/action features from support rows.

    The summary intentionally uses only runtime-observable quantities: recent
    proxy deltas and executed/planner local actions. Teacher metrics remain
    labels only.
    """
    n = int(episode_index.shape[0])
    action = np.asarray(
        raw.get("executed_action_local", raw.get("base_action", raw.get("planner_base_action_local_raw", np.zeros((n, 6), dtype=np.float32)))),
        dtype=np.float32,
    )
    if action.ndim != 2 or action.shape[1] < 6:
        action = np.zeros((n, 6), dtype=np.float32)
    action = action[:, :6]
    proxy = np.asarray(proxy, dtype=np.float32)
    selected_pos: dict[int, list[int]] = {}
    for i, idx in enumerate(selected_indices.tolist()):
        selected_pos.setdefault(int(idx), []).append(i)
    out = np.zeros((selected_indices.shape[0], 32), dtype=np.float32)
    k = max(1, int(horizon))
    for ep in np.unique(episode_index[selected_indices]).tolist():
        ep_all = np.flatnonzero(keep & (episode_index == ep))
        if ep_all.size == 0:
            continue
        ep_all = ep_all[np.argsort(rollout_step[ep_all])]
        for j, idx_np in enumerate(ep_all.tolist()):
            idx = int(idx_np)
            positions = selected_pos.get(idx)
            if not positions:
                continue
            prev1 = int(ep_all[max(0, j - 1)])
            prevk = int(ep_all[max(0, j - k)])
            win_start = max(0, j - k)
            win = ep_all[win_start : j + 1]
            a_win = action[win]
            current_proxy = proxy[idx, :6]
            prev_action = action[prev1, :6]
            mean_action = np.mean(a_win[:, :6], axis=0)
            std_action = np.std(a_win[:, :6], axis=0)
            delta_change_1 = current_proxy - proxy[prev1, :6]
            delta_change_k = current_proxy - proxy[prevk, :6]
            xyz_norm = float(np.linalg.norm(prev_action[:3]) * np.linalg.norm(current_proxy[:3]) + 1e-6)
            xyz_reduce_dot = float(np.dot(prev_action[:3], -current_proxy[:3]) / xyz_norm)
            yaw_sign_alignment = float(np.sign(prev_action[5]) * np.sign(-current_proxy[5]))
            summary = np.concatenate(
                [
                    delta_change_1[:6],
                    delta_change_k[:6],
                    prev_action[:6],
                    mean_action[:6],
                    std_action[:6],
                    np.asarray([xyz_reduce_dot, yaw_sign_alignment], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)
            for pos in positions:
                out[pos] = summary
    return np.clip(out, -10.0, 10.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--fallback_release_xy", type=float, default=0.0085)
    ap.add_argument("--fallback_release_z", type=float, default=0.0035)
    ap.add_argument("--fallback_release_yaw", type=float, default=0.1243404)
    ap.add_argument("--strict_near_norm", type=float, default=1.35)
    ap.add_argument("--broad_near_norm", type=float, default=3.0)
    ap.add_argument("--far_negative_min_norm", type=float, default=6.0)
    ap.add_argument("--min_boundary_rows", type=int, default=32)
    ap.add_argument("--boundary_cap_per_episode", type=int, default=96)
    ap.add_argument("--multi_axis_cap_per_episode", type=int, default=128)
    ap.add_argument("--far_negative_cap_per_episode", type=int, default=96)
    ap.add_argument("--counterfactual_multiplier", type=float, default=1.0)
    ap.add_argument("--counterfactual_weight", type=float, default=0.5)
    ap.add_argument("--progress_k", type=int, default=3)
    ap.add_argument("--progress_weighted_sum_delta_margin", type=float, default=0.10)
    ap.add_argument("--progress_max_axis_delta_margin", type=float, default=0.10)
    ap.add_argument("--pair_weighted_sum_delta_margin", type=float, default=0.10)
    ap.add_argument("--pair_max_axis_delta_margin", type=float, default=0.10)
    ap.add_argument("--progress_negative_weight_mult", type=float, default=2.0)
    ap.add_argument("--boundary_negative_weight_mult", type=float, default=2.5)
    ap.add_argument("--temporal_summary_horizon", type=int, default=3)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _concat_inputs([Path(p) for p in args.input_npz])

    episode_index = np.asarray(raw["episode_index"], dtype=np.int64)
    n = int(episode_index.shape[0])
    proxy = np.asarray(raw.get("proxy_current_delta_basin_target", raw.get("current_delta_basin_target")), dtype=np.float32)
    teacher_xy = np.asarray(raw["teacher_truth_handoff_metric_xy_error"], dtype=np.float32)
    teacher_z = np.asarray(raw["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float32)
    teacher_yaw = np.asarray(raw["teacher_truth_handoff_metric_yaw_error"], dtype=np.float32)
    teacher_ready = np.asarray(raw.get("teacher_truth_handoff_ready", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    rel_xy = _safe_threshold(raw.get("teacher_truth_handoff_release_threshold_xy_error", np.full((n,), np.nan)), args.fallback_release_xy)
    rel_z = _safe_threshold(raw.get("teacher_truth_handoff_release_threshold_abs_z_error", np.full((n,), np.nan)), args.fallback_release_z)
    rel_yaw = _safe_threshold(raw.get("teacher_truth_handoff_release_threshold_yaw_error", np.full((n,), np.nan)), args.fallback_release_yaw)
    norms_all = np.stack(
        [
            teacher_xy / np.maximum(rel_xy, 1e-6),
            teacher_z / np.maximum(rel_z, 1e-6),
            teacher_yaw / np.maximum(rel_yaw, 1e-6),
        ],
        axis=-1,
    ).astype(np.float32)

    runtime_valid = np.asarray(raw.get("runtime_handoff_metric_valid", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    planner_close_intent = np.asarray(raw.get("planner_close_intent", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    target_role = np.asarray(raw.get("handoff_target_role", np.full((n,), "none", dtype="U64"))).astype(str)
    target_source = np.asarray(raw.get("target_provider_source", np.full((n,), "unknown", dtype="U128"))).astype(str)
    phase_id = np.asarray(raw.get("phase_id", np.ones((n,), dtype=np.int64)), dtype=np.int64)
    gripper_open = np.asarray(raw.get("rollout_gripper_open", np.ones((n,), dtype=np.float32)), dtype=np.float32)
    has_object = np.asarray(raw.get("has_object_in_hand", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
    rollout_step = np.asarray(raw.get("rollout_step", raw.get("step_idx", np.arange(n))), dtype=np.int64)
    close_like = (
        planner_close_intent
        | np.isin(target_role, ["pregrasp_close", "close", "commit_close"])
        | (target_source == "learned_target_predictor__canonical_close_orientation_contract")
    )
    keep = (
        runtime_valid
        & close_like
        & (phase_id == 1)
        & (gripper_open >= 0.5)
        & (has_object <= 0.5)
        & np.all(np.isfinite(norms_all), axis=1)
    )

    rows_by_ep: dict[int, list[dict]] = {}
    raw_bucket_counts: dict[str, int] = {}
    for idx in np.flatnonzero(keep).tolist():
        norms = norms_all[idx]
        bucket = _bucket_for_norms(
            float(norms[0]),
            float(norms[1]),
            float(norms[2]),
            bool(teacher_ready[idx]),
            float(args.strict_near_norm),
            float(args.broad_near_norm),
            float(args.far_negative_min_norm),
        )
        if bucket is None:
            continue
        raw_bucket_counts[bucket] = raw_bucket_counts.get(bucket, 0) + 1
        ep = int(episode_index[idx])
        rows_by_ep.setdefault(ep, []).append(
            {
                "idx": int(idx),
                "bucket": bucket,
                "priority": _row_priority(bucket, norms, int(rollout_step[idx])),
            }
        )

    caps = {
        "xy_block_boundary": int(args.boundary_cap_per_episode),
        "z_block_boundary": int(args.boundary_cap_per_episode),
        "yaw_block_boundary": int(args.boundary_cap_per_episode),
        "multi_axis_boundary": int(args.multi_axis_cap_per_episode),
        "teacher_ready_anchor": int(args.boundary_cap_per_episode),
        "far_negative": int(args.far_negative_cap_per_episode),
        "default": int(args.boundary_cap_per_episode),
    }
    selected = _select_rows(rows_by_ep, caps, int(args.seed))
    real_indices = np.asarray(sorted([r["idx"] for r in selected]), dtype=np.int64)
    idx_to_bucket = {int(r["idx"]): str(r["bucket"]) for r in selected}
    boundary_count = sum(1 for r in selected if str(r["bucket"]) != "far_negative")
    if boundary_count < int(args.min_boundary_rows):
        report = {
            "decision": "blocked_insufficient_boundary_rows",
            "boundary_rows": int(boundary_count),
            "raw_bucket_counts": raw_bucket_counts,
            "selected_rows": int(real_indices.shape[0]),
        }
        (output_dir / "alignment_v2_dataset_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2

    source_names = np.asarray([idx_to_bucket[int(i)] for i in real_indices.tolist()], dtype="U64")
    real_norms = norms_all[real_indices].astype(np.float32)
    real_ready = teacher_ready[real_indices].astype(np.float32)
    real_cf = np.zeros((real_indices.shape[0],), dtype=np.float32)
    real_weight_map = {
        "teacher_ready_anchor": 8.0,
        "xy_block_boundary": 4.0,
        "z_block_boundary": 4.0,
        "yaw_block_boundary": 4.0,
        "multi_axis_boundary": 3.0,
        "far_negative": 1.0,
    }
    real_sample_weight = np.asarray([real_weight_map.get(str(s), 1.0) for s in source_names], dtype=np.float32)

    boundary_real_mask = source_names != "far_negative"
    cf_base = real_indices[boundary_real_mask]
    cf_count = int(round(cf_base.shape[0] * float(args.counterfactual_multiplier)))
    if cf_count > 0 and cf_base.shape[0] > 0:
        rng = np.random.default_rng(int(args.seed))
        cf_indices = rng.choice(cf_base, size=cf_count, replace=cf_count > cf_base.shape[0]).astype(np.int64)
    else:
        cf_indices = np.zeros((0,), dtype=np.int64)

    all_indices = np.concatenate([real_indices, cf_indices], axis=0)
    is_counterfactual = np.concatenate([real_cf, np.ones((cf_indices.shape[0],), dtype=np.float32)], axis=0)
    source_name = np.concatenate(
        [source_names, np.full((cf_indices.shape[0],), "counterfactual_near_ready", dtype="U64")],
        axis=0,
    )
    sample_weight = np.concatenate(
        [real_sample_weight, np.full((cf_indices.shape[0],), float(args.counterfactual_weight), dtype=np.float32)],
        axis=0,
    )
    temporal_action_summary = _temporal_action_summary(
        raw=raw,
        selected_indices=all_indices,
        keep=keep,
        episode_index=episode_index,
        rollout_step=rollout_step,
        proxy=proxy,
        horizon=int(args.temporal_summary_horizon),
    )
    metrics = norms_all[all_indices].astype(np.float32)
    if cf_indices.shape[0] > 0:
        cf_start = real_indices.shape[0]
        metrics[cf_start:] = np.minimum(metrics[cf_start:], 0.98)
        metrics[cf_start:] = np.maximum(metrics[cf_start:], 0.85)

    ready = teacher_ready[all_indices].astype(np.float32)
    ready[is_counterfactual > 0.5] = 0.0
    max_norm = np.max(metrics, axis=1)
    weighted_sum = (0.45 * metrics[:, 0] + 0.30 * metrics[:, 1] + 0.25 * metrics[:, 2]).astype(np.float32)
    axis_block_label = np.argmax(metrics, axis=1).astype(np.int64)
    band3 = _band_label(metrics, ready, is_counterfactual)
    band4 = _alignment_v2_band_label_4(metrics, ready, is_counterfactual)

    # Progress is defined on real rows from the original trace. Counterfactual
    # rows intentionally do not contribute to progress supervision.
    progress_label = np.zeros((all_indices.shape[0],), dtype=np.float32)
    progress_mask = np.zeros((all_indices.shape[0],), dtype=np.float32)
    progress_ambiguous_mask = np.zeros((all_indices.shape[0],), dtype=np.float32)
    progress_delta_weighted_sum = np.zeros((all_indices.shape[0],), dtype=np.float32)
    progress_delta_max_axis = np.zeros((all_indices.shape[0],), dtype=np.float32)
    progress_label_k1 = np.zeros((all_indices.shape[0],), dtype=np.float32)
    progress_mask_k1 = np.zeros((all_indices.shape[0],), dtype=np.float32)
    pair_prev_index = np.full((all_indices.shape[0],), -1, dtype=np.int64)
    pair_label = np.zeros((all_indices.shape[0],), dtype=np.float32)
    pair_mask = np.zeros((all_indices.shape[0],), dtype=np.float32)
    pair_delta_weighted_sum = np.zeros((all_indices.shape[0],), dtype=np.float32)
    pair_delta_max_axis = np.zeros((all_indices.shape[0],), dtype=np.float32)
    selected_real_pos = {int(idx): i for i, idx in enumerate(real_indices.tolist())}
    for ep in np.unique(episode_index[real_indices]).tolist():
        ep_all = np.flatnonzero(keep & (episode_index == ep))
        if ep_all.size < 2:
            continue
        ep_all = ep_all[np.argsort(rollout_step[ep_all])]
        ep_score = 0.45 * norms_all[ep_all, 0] + 0.30 * norms_all[ep_all, 1] + 0.25 * norms_all[ep_all, 2]
        for j in range(1, ep_all.size):
            idx = int(ep_all[j])
            if idx in selected_real_pos:
                delta1 = float(ep_score[j - 1] - ep_score[j])
                if abs(delta1) >= float(args.progress_weighted_sum_delta_margin):
                    pos = selected_real_pos[idx]
                    progress_mask_k1[pos] = 1.0
                    progress_label_k1[pos] = 1.0 if delta1 > 0.0 else 0.0
        ep_max = np.max(norms_all[ep_all], axis=1)
        k = max(1, int(args.progress_k))
        for j in range(k, ep_all.size):
            idx = int(ep_all[j])
            if idx not in selected_real_pos:
                continue
            delta_sum = float(ep_score[j - k] - ep_score[j])
            delta_max = float(ep_max[j - k] - ep_max[j])
            pos = selected_real_pos[idx]
            progress_delta_weighted_sum[pos] = delta_sum
            progress_delta_max_axis[pos] = delta_max
            if (
                abs(delta_sum) < float(args.progress_weighted_sum_delta_margin)
                or abs(delta_max) < float(args.progress_max_axis_delta_margin)
                or np.sign(delta_sum) != np.sign(delta_max)
            ):
                progress_ambiguous_mask[pos] = 1.0
                continue
            progress_mask[pos] = 1.0
            progress_label[pos] = 1.0 if delta_sum > 0.0 else 0.0

        # Pairwise progress is an ordering target over selected close-like rows:
        # "which of these two observable states is closer to ready?"  It avoids
        # turning progress into a single-frame BCE threshold.
        ep_selected = [int(i) for i in ep_all.tolist() if int(i) in selected_real_pos]
        if len(ep_selected) >= 2:
            ep_selected = sorted(ep_selected, key=lambda x: int(rollout_step[x]))
            for prev_raw, cur_raw in zip(ep_selected[:-1], ep_selected[1:]):
                prev_pos = selected_real_pos[int(prev_raw)]
                cur_pos = selected_real_pos[int(cur_raw)]
                prev_sum = float(0.45 * norms_all[prev_raw, 0] + 0.30 * norms_all[prev_raw, 1] + 0.25 * norms_all[prev_raw, 2])
                cur_sum = float(0.45 * norms_all[cur_raw, 0] + 0.30 * norms_all[cur_raw, 1] + 0.25 * norms_all[cur_raw, 2])
                prev_max = float(np.max(norms_all[prev_raw]))
                cur_max = float(np.max(norms_all[cur_raw]))
                delta_sum = prev_sum - cur_sum
                delta_max = prev_max - cur_max
                pair_prev_index[cur_pos] = int(prev_pos)
                pair_delta_weighted_sum[cur_pos] = float(delta_sum)
                pair_delta_max_axis[cur_pos] = float(delta_max)
                if (
                    abs(delta_sum) < float(args.pair_weighted_sum_delta_margin)
                    or abs(delta_max) < float(args.pair_max_axis_delta_margin)
                    or np.sign(delta_sum) != np.sign(delta_max)
                ):
                    continue
                pair_mask[cur_pos] = 1.0
                pair_label[cur_pos] = 1.0 if delta_sum > 0.0 else 0.0

    out: dict[str, np.ndarray] = {}
    for key in INPUT_KEYS:
        if key in raw:
            out[key] = np.asarray(raw[key][all_indices])
    out["proxy_current_delta_basin_target"] = proxy[all_indices].astype(np.float32)
    out["temporal_action_summary"] = temporal_action_summary.astype(np.float32)
    out["episode_index"] = episode_index[all_indices].astype(np.int64)
    out["rollout_step"] = rollout_step[all_indices].astype(np.int64)
    out["teacher_metrics_norm"] = metrics.astype(np.float32)
    out["teacher_xy_norm"] = metrics[:, 0].astype(np.float32)
    out["teacher_abs_z_norm"] = metrics[:, 1].astype(np.float32)
    out["teacher_yaw_norm"] = metrics[:, 2].astype(np.float32)
    out["teacher_truth_handoff_ready"] = ready.astype(np.float32)
    out["teacher_band_label"] = band3.astype(np.int64)
    out["alignment_v2_band_label_4"] = band4.astype(np.int64)
    out["alignment_v2_axis_block_label"] = axis_block_label.astype(np.int64)
    out["alignment_v2_progress_label"] = progress_label.astype(np.float32)
    out["alignment_v2_progress_mask"] = progress_mask.astype(np.float32)
    out["alignment_v2_progress_ambiguous_mask"] = progress_ambiguous_mask.astype(np.float32)
    out["alignment_v2_progress_delta_weighted_sum"] = progress_delta_weighted_sum.astype(np.float32)
    out["alignment_v2_progress_delta_max_axis"] = progress_delta_max_axis.astype(np.float32)
    out["alignment_v2_progress_label_k1"] = progress_label_k1.astype(np.float32)
    out["alignment_v2_progress_mask_k1"] = progress_mask_k1.astype(np.float32)
    out["alignment_v2_pair_prev_index"] = pair_prev_index.astype(np.int64)
    out["alignment_v2_pair_label"] = pair_label.astype(np.float32)
    out["alignment_v2_pair_mask"] = pair_mask.astype(np.float32)
    out["alignment_v2_pair_delta_weighted_sum"] = pair_delta_weighted_sum.astype(np.float32)
    out["alignment_v2_pair_delta_max_axis"] = pair_delta_max_axis.astype(np.float32)
    out["alignment_v2_weighted_sum_norm"] = weighted_sum.astype(np.float32)
    out["alignment_v2_max_axis_norm"] = max_norm.astype(np.float32)
    out["is_counterfactual"] = is_counterfactual.astype(np.float32)
    out["source_name"] = source_name.astype("U64")
    out["sample_weight"] = sample_weight.astype(np.float32)
    out["runtime_handoff_metric_valid"] = raw.get("runtime_handoff_metric_valid", np.ones((n,), dtype=np.float32))[all_indices].astype(np.float32)
    out["runtime_handoff_ready"] = raw.get("runtime_handoff_ready_applied", raw.get("runtime_handoff_ready", np.zeros((n,), dtype=np.float32)))[all_indices].astype(np.float32)
    out["runtime_handoff_ready_pred"] = raw.get("runtime_handoff_ready_pred", np.zeros((n,), dtype=np.float32))[all_indices].astype(np.float32)
    out["planner_close_intent"] = planner_close_intent[all_indices].astype(np.float32)
    out["handoff_target_role"] = target_role[all_indices].astype("U64")
    out["target_provider_source"] = target_source[all_indices].astype("U128")
    out["near_xy_hard"] = ((source_name == "xy_block_boundary") | (source_name == "multi_axis_boundary")).astype(np.float32)
    out["near_yaw_hard"] = (source_name == "yaw_block_boundary").astype(np.float32)
    out["near_coupled"] = (source_name == "multi_axis_boundary").astype(np.float32)
    out["broad_xy_recovery"] = out["near_xy_hard"].copy()
    out["ready_support"] = ready.astype(np.float32)
    out["focus_window_mask_v2"] = (source_name != "far_negative").astype(np.float32)
    out["teacher_ready_exact_mask_v2"] = (teacher_ready[all_indices] & (is_counterfactual <= 0.5)).astype(np.float32)
    out["boundary_band_mask_v2"] = np.isin(source_name, ["xy_block_boundary", "z_block_boundary", "yaw_block_boundary", "multi_axis_boundary"]).astype(np.float32)
    out["far_negative_mask_v2"] = (source_name == "far_negative").astype(np.float32)
    out["current_profile_hard_negative_v1"] = np.zeros((all_indices.shape[0],), dtype=np.float32)
    progress_negative = (out["alignment_v2_progress_mask"] > 0.5) & (out["alignment_v2_progress_label"] <= 0.5)
    boundary_negative = progress_negative & (out["boundary_band_mask_v2"] > 0.5)
    out["sample_weight"][progress_negative] *= float(args.progress_negative_weight_mult)
    out["sample_weight"][boundary_negative] *= float(args.boundary_negative_weight_mult)

    full_npz = output_dir / "handoff_state_dataset_v2_alignment_v2_full.npz"
    stagea_npz = output_dir / "handoff_state_dataset_v2_alignment_v2_stageA.npz"
    stagea_mask = out["is_counterfactual"] <= 0.5
    np.savez_compressed(full_npz, **out)
    np.savez_compressed(stagea_npz, **_subset_rows(out, stagea_mask))

    source_summary = _source_summary(out["source_name"], out["episode_index"], out["teacher_truth_handoff_ready"])
    report = {
        "decision": "ok",
        "input_npz": [str(p) for p in args.input_npz],
        "full_npz": str(full_npz),
        "stagea_npz": str(stagea_npz),
        "rows": int(all_indices.shape[0]),
        "real_rows": int(real_indices.shape[0]),
        "counterfactual_rows": int(cf_indices.shape[0]),
        "teacher_ready_rows_real": int(np.sum(teacher_ready[real_indices])),
        "runtime_ready_rows_real": int(np.sum(raw.get("runtime_handoff_ready", np.zeros((n,), dtype=np.float32))[real_indices] > 0.5)),
        "raw_bucket_counts": raw_bucket_counts,
        "source_summary": source_summary,
        "close_intent_rows_real": int(np.sum(planner_close_intent[real_indices])),
        "norm_percentiles": {
            "xy": _percentiles(metrics[:, 0]),
            "z": _percentiles(metrics[:, 1]),
            "yaw": _percentiles(metrics[:, 2]),
            "weighted_sum": _percentiles(weighted_sum),
            "max_axis": _percentiles(max_norm),
        },
        "progress": {
            "masked_rows": int(np.sum(progress_mask)),
            "positive_rows": int(np.sum((progress_mask > 0.5) & (progress_label > 0.5))),
            "negative_rows": int(np.sum((progress_mask > 0.5) & (progress_label <= 0.5))),
            "ambiguous_rows": int(np.sum(progress_ambiguous_mask > 0.5)),
            "k1_masked_rows": int(np.sum(progress_mask_k1 > 0.5)),
            "k1_positive_rows": int(np.sum((progress_mask_k1 > 0.5) & (progress_label_k1 > 0.5))),
            "k1_negative_rows": int(np.sum((progress_mask_k1 > 0.5) & (progress_label_k1 <= 0.5))),
            "progress_k": int(args.progress_k),
            "weighted_sum_delta_margin": float(args.progress_weighted_sum_delta_margin),
            "max_axis_delta_margin": float(args.progress_max_axis_delta_margin),
            "temporal_summary_horizon": int(args.temporal_summary_horizon),
        },
        "pairwise_progress": {
            "masked_pairs": int(np.sum(pair_mask)),
            "positive_pairs": int(np.sum((pair_mask > 0.5) & (pair_label > 0.5))),
            "negative_pairs": int(np.sum((pair_mask > 0.5) & (pair_label <= 0.5))),
            "weighted_sum_delta_margin": float(args.pair_weighted_sum_delta_margin),
            "max_axis_delta_margin": float(args.pair_max_axis_delta_margin),
        },
        "temporal_action_summary": {
            "dim": 32,
            "nonzero_rows": int(np.sum(np.linalg.norm(temporal_action_summary, axis=1) > 1e-8)),
            "mean_abs": float(np.mean(np.abs(temporal_action_summary))) if temporal_action_summary.size else 0.0,
        },
        "compatibility_note": "teacher_band_label remains 3-class for StudentHandoffStateHeadV2 runtime compatibility; alignment_v2_band_label_4 is diagnostic.",
    }
    (output_dir / "alignment_v2_dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
