#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


RECOLLECTION_FOCUS_EPS = [18, 34, 45, 13]


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


def _concat_inputs(paths: list[Path]) -> dict[str, np.ndarray]:
    chunks: list[dict[str, np.ndarray]] = []
    for path in paths:
        raw = np.load(path, allow_pickle=False)
        chunks.append({k: np.asarray(raw[k]) for k in raw.files})
    if not chunks:
        raise RuntimeError("no input support npz provided")
    keys = list(chunks[0].keys())
    out: dict[str, np.ndarray] = {}
    for key in keys:
        out[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return out


def _bucket_name(
    teacher_ready: bool,
    xy_norm: float,
    z_norm: float,
    yaw_norm: float,
) -> str | None:
    if teacher_ready:
        return "runtime_close_teacher_ready_v1"
    max_norm = max(xy_norm, z_norm, yaw_norm)
    near_axes = sum(float(v <= 1.10) for v in (xy_norm, z_norm, yaw_norm))
    if max_norm <= 1.35 and near_axes >= 2:
        return "runtime_close_boundary_v1"
    if xy_norm > 1.0 and z_norm <= 1.10 and yaw_norm <= 1.10:
        return "runtime_close_xy_block_v1"
    if z_norm > 1.0 and xy_norm <= 1.10 and yaw_norm <= 1.10:
        return "runtime_close_z_block_v1"
    if max_norm >= 1.80:
        return "runtime_broad_negative_v1"
    return None


def _score_row(bucket: str, xy_norm: float, z_norm: float, yaw_norm: float, step_idx: int) -> tuple[float, float]:
    max_norm = max(xy_norm, z_norm, yaw_norm)
    if bucket == "runtime_close_teacher_ready_v1":
        return (float(step_idx), max_norm)
    if bucket == "runtime_close_boundary_v1":
        return (abs(max_norm - 1.0), float(step_idx))
    if bucket == "runtime_close_xy_block_v1":
        return (abs(xy_norm - 1.0), float(step_idx))
    if bucket == "runtime_close_z_block_v1":
        return (abs(z_norm - 1.0), float(step_idx))
    return (float(step_idx), max_norm)


def _select_rows(
    episode_rows: list[dict],
    boundary_cap: int,
    block_cap: int,
    broad_cap: int,
    seed: int,
) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in episode_rows:
        buckets.setdefault(str(row["bucket_name"]), []).append(row)
    selected: list[dict] = []
    selected.extend(buckets.get("runtime_close_teacher_ready_v1", []))
    selected.extend(sorted(buckets.get("runtime_close_boundary_v1", []), key=lambda r: r["priority"])[:boundary_cap])
    selected.extend(sorted(buckets.get("runtime_close_xy_block_v1", []), key=lambda r: r["priority"])[:block_cap])
    selected.extend(sorted(buckets.get("runtime_close_z_block_v1", []), key=lambda r: r["priority"])[:block_cap])
    broad = list(buckets.get("runtime_broad_negative_v1", []))
    if len(broad) > broad_cap:
        rng = random.Random(seed + int(episode_rows[0]["episode_index"]))
        broad = rng.sample(broad, broad_cap)
    selected.extend(broad)
    return selected


def _subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == mask.shape[0]:
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def _source_summary(data: dict[str, np.ndarray]) -> dict[str, dict]:
    names = np.asarray(data["source_name"]).astype(str)
    episodes = np.asarray(data["episode_index"], dtype=np.int64)
    ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    out: dict[str, dict] = {}
    for src in np.unique(names).tolist():
        mask = names == src
        out[str(src)] = {
            "rows": int(np.sum(mask)),
            "episodes": int(np.unique(episodes[mask]).size),
            "teacher_ready_pos": int(np.sum(ready[mask])),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--recollection_episode_csv", default="")
    ap.add_argument("--fallback_release_xy", type=float, default=0.0085)
    ap.add_argument("--fallback_release_z", type=float, default=0.0035)
    ap.add_argument("--fallback_release_yaw", type=float, default=0.1243404)
    ap.add_argument("--boundary_cap_per_episode", type=int, default=64)
    ap.add_argument("--block_cap_per_episode", type=int, default=64)
    ap.add_argument("--broad_negative_cap_per_episode", type=int, default=96)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(p) for p in args.input_npz]
    raw = _concat_inputs(input_paths)

    proxy = np.asarray(
        raw.get("proxy_current_delta_basin_target", raw.get("current_delta_basin_target")),
        dtype=np.float32,
    )
    if proxy is None:
        raise RuntimeError("support rows missing proxy/current delta field")
    teacher_xy = np.asarray(raw["teacher_truth_handoff_metric_xy_error"], dtype=np.float32)
    teacher_z = np.asarray(raw["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float32)
    teacher_yaw = np.asarray(raw["teacher_truth_handoff_metric_yaw_error"], dtype=np.float32)
    teacher_ready = np.asarray(raw["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    rel_xy = _safe_threshold(
        np.asarray(raw.get("teacher_truth_handoff_release_threshold_xy_error", np.full_like(teacher_xy, np.nan)), dtype=np.float32),
        float(args.fallback_release_xy),
    )
    rel_z = _safe_threshold(
        np.asarray(raw.get("teacher_truth_handoff_release_threshold_abs_z_error", np.full_like(teacher_z, np.nan)), dtype=np.float32),
        float(args.fallback_release_z),
    )
    rel_yaw = _safe_threshold(
        np.asarray(raw.get("teacher_truth_handoff_release_threshold_yaw_error", np.full_like(teacher_yaw, np.nan)), dtype=np.float32),
        float(args.fallback_release_yaw),
    )
    episode_index = np.asarray(raw["episode_index"], dtype=np.int64)
    phase_id = np.asarray(raw.get("phase_id", np.ones((episode_index.shape[0],), dtype=np.int64)), dtype=np.int64)
    gripper_open = np.asarray(raw.get("rollout_gripper_open", np.ones((episode_index.shape[0],), dtype=np.float32)), dtype=np.float32)
    has_object = np.asarray(raw.get("has_object_in_hand", np.zeros((episode_index.shape[0],), dtype=np.float32)), dtype=np.float32)
    runtime_valid = np.asarray(raw.get("runtime_handoff_metric_valid", np.zeros((episode_index.shape[0],), dtype=np.float32)), dtype=np.float32) > 0.5
    planner_close_intent = np.asarray(raw.get("planner_close_intent", np.zeros((episode_index.shape[0],), dtype=np.float32)), dtype=np.float32) > 0.5
    target_role = np.asarray(raw.get("handoff_target_role", np.full((episode_index.shape[0],), "none", dtype="U64"))).astype(str)
    target_source = np.asarray(raw.get("target_provider_source", np.full((episode_index.shape[0],), "unknown", dtype="U128"))).astype(str)
    rollout_step = np.asarray(raw.get("rollout_step", np.arange(episode_index.shape[0], dtype=np.int64)), dtype=np.int64)

    valid_teacher = np.isfinite(teacher_xy) & np.isfinite(teacher_z) & np.isfinite(teacher_yaw)
    close_like = (
        planner_close_intent
        | np.isin(target_role, ["pregrasp_close", "close", "commit_close"])
        | (target_source == "learned_target_predictor__canonical_close_orientation_contract")
    )
    keep = (
        (phase_id == 1)
        & (gripper_open >= 0.5)
        & (has_object <= 0.5)
        & runtime_valid
        & valid_teacher
        & close_like
    )
    keep_idx = np.flatnonzero(keep)
    if keep_idx.size == 0:
        raise RuntimeError("no runtime-ready support rows survive filtering")

    teacher_xy_norm = (teacher_xy[keep] / np.maximum(rel_xy[keep], 1e-6)).astype(np.float32)
    teacher_z_norm = (teacher_z[keep] / np.maximum(rel_z[keep], 1e-6)).astype(np.float32)
    teacher_yaw_norm = (teacher_yaw[keep] / np.maximum(rel_yaw[keep], 1e-6)).astype(np.float32)

    selected_rows_by_episode: dict[int, list[dict]] = {}
    raw_bucket_counts: dict[str, int] = {}
    for local_i, idx in enumerate(keep_idx.tolist()):
        bucket = _bucket_name(
            bool(teacher_ready[idx]),
            float(teacher_xy_norm[local_i]),
            float(teacher_z_norm[local_i]),
            float(teacher_yaw_norm[local_i]),
        )
        if bucket is None:
            continue
        raw_bucket_counts[bucket] = raw_bucket_counts.get(bucket, 0) + 1
        ep = int(episode_index[idx])
        selected_rows_by_episode.setdefault(ep, []).append(
            {
                "global_index": idx,
                "episode_index": ep,
                "bucket_name": bucket,
                "priority": _score_row(
                    bucket,
                    float(teacher_xy_norm[local_i]),
                    float(teacher_z_norm[local_i]),
                    float(teacher_yaw_norm[local_i]),
                    int(rollout_step[idx]),
                ),
            }
        )

    selected_global_indices: list[int] = []
    selected_bucket_counts: dict[str, int] = {}
    for ep in sorted(selected_rows_by_episode):
        chosen = _select_rows(
            selected_rows_by_episode[ep],
            boundary_cap=int(args.boundary_cap_per_episode),
            block_cap=int(args.block_cap_per_episode),
            broad_cap=int(args.broad_negative_cap_per_episode),
            seed=int(args.seed),
        )
        for row in chosen:
            selected_global_indices.append(int(row["global_index"]))
            selected_bucket_counts[str(row["bucket_name"])] = selected_bucket_counts.get(str(row["bucket_name"]), 0) + 1

    if not selected_global_indices:
        raise RuntimeError("no rows selected into runtime-ready dataset")

    selected_global_indices = sorted(selected_global_indices)
    sel = np.asarray(selected_global_indices, dtype=np.int64)
    out: dict[str, np.ndarray] = {
        "front_rgb": np.asarray(raw["front_rgb"][sel], dtype=np.uint8),
        "wrist_rgb": np.asarray(raw["wrist_rgb"][sel], dtype=np.uint8),
        "wrist_depth": np.asarray(raw["wrist_depth"][sel], dtype=np.float32),
        "proprio": np.asarray(raw["proprio"][sel], dtype=np.float32),
        "gripper_context": np.asarray(raw["gripper_context"][sel], dtype=np.float32),
        "proxy_current_delta_basin_target": np.asarray(proxy[sel], dtype=np.float32),
        "current_dx_sign": np.asarray(raw["current_dx_sign"][sel], dtype=np.int64),
        "current_dy_sign": np.asarray(raw["current_dy_sign"][sel], dtype=np.int64),
        "current_dyaw_sign": np.asarray(raw["current_dyaw_sign"][sel], dtype=np.int64),
        "basin_distance_bin": np.asarray(raw["basin_distance_bin"][sel], dtype=np.int64),
        "substage_id": np.asarray(raw.get("substage_id", np.zeros_like(episode_index))[sel], dtype=np.int64),
        "contact_state": np.asarray(raw.get("contact_state", np.zeros_like(episode_index))[sel], dtype=np.int64),
        "stage_target_mode": np.asarray(raw.get("stage_target_mode", np.zeros_like(episode_index))[sel], dtype=np.int64),
        "episode_index": np.asarray(episode_index[sel], dtype=np.int64),
        "teacher_xy_norm": np.asarray(teacher_xy[sel] / np.maximum(rel_xy[sel], 1e-6), dtype=np.float32),
        "teacher_abs_z_norm": np.asarray(teacher_z[sel] / np.maximum(rel_z[sel], 1e-6), dtype=np.float32),
        "teacher_yaw_norm": np.asarray(teacher_yaw[sel] / np.maximum(rel_yaw[sel], 1e-6), dtype=np.float32),
        "teacher_truth_handoff_ready": np.asarray(teacher_ready[sel].astype(np.float32), dtype=np.float32),
        "teacher_truth_release_threshold_xy_error": np.asarray(rel_xy[sel], dtype=np.float32),
        "teacher_truth_release_threshold_abs_z_error": np.asarray(rel_z[sel], dtype=np.float32),
        "teacher_truth_release_threshold_yaw_error": np.asarray(rel_yaw[sel], dtype=np.float32),
        "runtime_handoff_metric_xy_error": np.asarray(raw["runtime_handoff_metric_xy_error"][sel], dtype=np.float32),
        "runtime_handoff_metric_abs_z_error": np.asarray(raw["runtime_handoff_metric_abs_z_error"][sel], dtype=np.float32),
        "runtime_handoff_metric_yaw_error": np.asarray(raw["runtime_handoff_metric_yaw_error"][sel], dtype=np.float32),
        "runtime_handoff_metric_valid": np.asarray(raw["runtime_handoff_metric_valid"][sel], dtype=np.float32),
        "runtime_handoff_ready": np.asarray(raw.get("runtime_handoff_ready_applied", raw.get("runtime_handoff_ready", np.zeros_like(teacher_xy)))[sel], dtype=np.float32),
        "runtime_handoff_ready_pred": np.asarray(raw.get("runtime_handoff_ready_pred", np.zeros_like(teacher_xy))[sel], dtype=np.float32),
        "teacher_ready_exact_mask_v2": np.zeros((sel.shape[0],), dtype=np.float32),
        "focus_window_mask_v2": np.zeros((sel.shape[0],), dtype=np.float32),
        "boundary_band_mask_v2": np.zeros((sel.shape[0],), dtype=np.float32),
        "far_negative_mask_v2": np.zeros((sel.shape[0],), dtype=np.float32),
        "current_profile_hard_negative_v1": np.zeros((sel.shape[0],), dtype=np.float32),
        "rollout_step": np.asarray(rollout_step[sel], dtype=np.int64),
        "source_id": np.zeros((sel.shape[0],), dtype=np.int64),
        "source_name": np.full((sel.shape[0],), "", dtype="U64"),
        "planner_close_intent": np.asarray(planner_close_intent[sel].astype(np.float32), dtype=np.float32),
        "handoff_target_role": np.asarray(target_role[sel]).astype("U64"),
        "target_provider_source": np.asarray(target_source[sel]).astype("U128"),
        "runtime_handoff_pred_ready_prob": np.asarray(raw.get("runtime_handoff_pred_ready_prob", np.full_like(teacher_xy, np.nan))[sel], dtype=np.float32),
    }
    out["teacher_metrics_norm"] = np.stack(
        [out["teacher_xy_norm"], out["teacher_abs_z_norm"], out["teacher_yaw_norm"]],
        axis=-1,
    ).astype(np.float32)
    out["teacher_band_label"] = _teacher_band_label(
        teacher_xy[sel],
        teacher_z[sel],
        teacher_yaw[sel],
        rel_xy[sel],
        rel_z[sel],
        rel_yaw[sel],
        out["teacher_truth_handoff_ready"],
    )
    out["near_xy_hard"] = (
        (teacher_z[sel] <= 1.2 * rel_z[sel])
        & (teacher_xy[sel] > rel_xy[sel])
        & (teacher_yaw[sel] <= 1.2 * rel_yaw[sel])
    ).astype(np.float32)
    out["broad_xy_recovery"] = (
        (teacher_z[sel] <= 1.5 * rel_z[sel])
        & (teacher_xy[sel] > rel_xy[sel])
        & (teacher_yaw[sel] <= 1.5 * rel_yaw[sel])
    ).astype(np.float32)
    out["near_yaw_hard"] = (
        (teacher_z[sel] <= 1.2 * rel_z[sel])
        & (teacher_yaw[sel] > rel_yaw[sel])
        & (teacher_xy[sel] <= 1.2 * rel_xy[sel])
    ).astype(np.float32)
    out["near_coupled"] = (
        (teacher_z[sel] <= 1.2 * rel_z[sel])
        & (teacher_xy[sel] > rel_xy[sel])
        & (teacher_yaw[sel] > rel_yaw[sel])
    ).astype(np.float32)
    out["ready_support"] = out["teacher_truth_handoff_ready"].copy()

    source_id_map = {
        "runtime_close_teacher_ready_v1": 0,
        "runtime_close_boundary_v1": 1,
        "runtime_close_xy_block_v1": 2,
        "runtime_close_z_block_v1": 3,
        "runtime_broad_negative_v1": 4,
    }
    sample_weight_map = {
        "runtime_close_teacher_ready_v1": 10.0,
        "runtime_close_boundary_v1": 6.0,
        "runtime_close_xy_block_v1": 4.0,
        "runtime_close_z_block_v1": 4.0,
        "runtime_broad_negative_v1": 1.0,
    }
    index_to_bucket = {}
    for rows in selected_rows_by_episode.values():
        for row in rows:
            index_to_bucket[int(row["global_index"])] = str(row["bucket_name"])
    source_names = []
    sample_weight = np.ones((sel.shape[0],), dtype=np.float32)
    for i, idx in enumerate(sel.tolist()):
        src = index_to_bucket[int(idx)]
        source_names.append(src)
        out["source_id"][i] = int(source_id_map[src])
        sample_weight[i] = float(sample_weight_map[src])
        if src == "runtime_close_teacher_ready_v1":
            out["teacher_ready_exact_mask_v2"][i] = 1.0
            out["focus_window_mask_v2"][i] = 1.0
        elif src == "runtime_close_boundary_v1":
            out["focus_window_mask_v2"][i] = 1.0
            out["boundary_band_mask_v2"][i] = 1.0
        elif src == "runtime_broad_negative_v1":
            out["far_negative_mask_v2"][i] = 1.0
    out["source_name"] = np.asarray(source_names, dtype="U64")
    out["sample_weight"] = sample_weight

    full_npz = output_dir / "handoff_state_dataset_v2_runtime_ready_full.npz"
    stagea_npz = output_dir / "handoff_state_dataset_v2_runtime_ready_stageA.npz"
    stagea_mask = np.asarray(out["source_name"]) != "runtime_broad_negative_v1"
    np.savez_compressed(full_npz, **out)
    np.savez_compressed(stagea_npz, **_subset(out, stagea_mask))

    per_episode_counts: dict[int, dict[str, int]] = {}
    full_names = np.asarray(out["source_name"]).astype(str)
    full_eps = np.asarray(out["episode_index"], dtype=np.int64)
    full_ready = np.asarray(out["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    for ep in np.unique(full_eps).tolist():
        mask = full_eps == ep
        stats = {"rows": int(np.sum(mask)), "teacher_ready_rows": int(np.sum(full_ready[mask]))}
        for src in np.unique(full_names[mask]).tolist():
            stats[str(src)] = int(np.sum(full_names[mask] == src))
        per_episode_counts[int(ep)] = stats

    recollection_eps = [int(x) for x in args.recollection_episode_csv.split(",") if str(x).strip()]
    teacher_ready_rank = sorted(
        per_episode_counts.items(),
        key=lambda kv: (-kv[1].get("teacher_ready_rows", 0), -kv[1].get("runtime_close_boundary_v1", 0), kv[0]),
    )
    boundary_rank = sorted(
        per_episode_counts.items(),
        key=lambda kv: (-kv[1].get("runtime_close_boundary_v1", 0), -kv[1].get("rows", 0), kv[0]),
    )
    val_eps: list[int] = []
    for ep, stats in teacher_ready_rank:
        if stats.get("teacher_ready_rows", 0) <= 0:
            break
        if ep not in val_eps:
            val_eps.append(ep)
        if len(val_eps) >= 2:
            break
    for ep in RECOLLECTION_FOCUS_EPS:
        if ep in per_episode_counts and ep not in val_eps:
            val_eps.append(ep)
            break
    for ep, _ in boundary_rank:
        if ep not in val_eps:
            val_eps.append(ep)
            break

    meta = {
        "input_npz": [str(p) for p in input_paths],
        "full_npz": str(full_npz),
        "stagea_npz": str(stagea_npz),
        "recollection_episode_csv": args.recollection_episode_csv,
        "rows": int(sel.shape[0]),
        "stagea_rows": int(np.sum(stagea_mask)),
        "raw_bucket_counts": raw_bucket_counts,
        "selected_bucket_counts": selected_bucket_counts,
        "source_summary_full": _source_summary(out),
        "source_summary_stageA": _source_summary(_subset(out, stagea_mask)),
        "per_episode_counts": per_episode_counts,
        "recommended_val_episode_indices": val_eps,
        "recommended_val_episode_csv": ",".join(str(x) for x in val_eps),
        "weights": sample_weight_map,
        "caps": {
            "boundary_cap_per_episode": int(args.boundary_cap_per_episode),
            "block_cap_per_episode": int(args.block_cap_per_episode),
            "broad_negative_cap_per_episode": int(args.broad_negative_cap_per_episode),
        },
    }
    (output_dir / "builder_meta.json").write_text(json.dumps(meta, indent=2))
    (output_dir / "val_episode_csv.txt").write_text(meta["recommended_val_episode_csv"] + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
