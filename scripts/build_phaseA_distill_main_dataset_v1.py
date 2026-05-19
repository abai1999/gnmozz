#!/usr/bin/env python3
"""
Build Phase-A distillation main dataset v1 with explicit coverage buckets.

Goal:
- Keep runtime_like as main runtime distribution base.
- Add window-level teacher-support rows to cover:
  1) teacher_ready cross-episode
  2) very_near + xy_block
  3) yaw_needed
  4) far_negative
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _sample_rows(
    rng: np.random.Generator,
    idx: np.ndarray,
    cap: int,
) -> np.ndarray:
    if cap < 0 or idx.size <= cap:
        return np.asarray(idx, dtype=np.int64)
    return np.asarray(rng.choice(idx, size=cap, replace=False), dtype=np.int64)


def _sample_teacher_ready_balanced(
    rng: np.random.Generator,
    teacher_ready_idx: np.ndarray,
    episode_index: np.ndarray,
    cap: int,
) -> np.ndarray:
    if cap < 0 or teacher_ready_idx.size <= cap:
        return np.asarray(teacher_ready_idx, dtype=np.int64)
    eps = np.unique(episode_index[teacher_ready_idx])
    if eps.size == 0:
        return np.zeros((0,), dtype=np.int64)
    per_ep = max(1, cap // int(eps.size))
    picked = []
    remain = []
    for e in eps.tolist():
        ep_rows = teacher_ready_idx[episode_index[teacher_ready_idx] == e]
        take = min(per_ep, ep_rows.size)
        if take > 0:
            picked.append(rng.choice(ep_rows, size=take, replace=False))
        if ep_rows.size > take:
            remain.append(np.setdiff1d(ep_rows, np.concatenate(picked[-1:]) if picked else np.zeros((0,), dtype=np.int64)))
    picked_arr = np.concatenate(picked) if picked else np.zeros((0,), dtype=np.int64)
    if picked_arr.size >= cap:
        return picked_arr[:cap]
    remain_arr = np.concatenate(remain) if remain else np.zeros((0,), dtype=np.int64)
    need = cap - picked_arr.size
    if remain_arr.size > 0 and need > 0:
        add = rng.choice(remain_arr, size=min(need, remain_arr.size), replace=False)
        picked_arr = np.concatenate([picked_arr, add])
    return np.asarray(picked_arr, dtype=np.int64)


def _subset_summary(
    idx: np.ndarray,
    episode_index: np.ndarray,
    teacher_ready: np.ndarray,
    xy_block: np.ndarray,
    yaw_needed: np.ndarray,
    far_negative: np.ndarray,
) -> dict:
    if idx.size == 0:
        return {
            "rows": 0,
            "episodes": 0,
            "teacher_ready_rows": 0,
            "teacher_ready_eps": 0,
            "xy_block_rows": 0,
            "xy_block_eps": 0,
            "yaw_needed_rows": 0,
            "yaw_needed_eps": 0,
            "far_negative_rows": 0,
            "far_negative_eps": 0,
        }
    ep = episode_index[idx]
    out = {
        "rows": int(idx.size),
        "episodes": int(np.unique(ep).size),
    }
    for name, mask in [
        ("teacher_ready", teacher_ready),
        ("xy_block", xy_block),
        ("yaw_needed", yaw_needed),
        ("far_negative", far_negative),
    ]:
        m = mask[idx]
        out[f"{name}_rows"] = int(np.sum(m))
        out[f"{name}_eps"] = int(np.unique(ep[m]).size) if np.any(m) else 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True, help="Merged handoff-state dataset npz")
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--runtime_source_name", default="runtime_like")
    ap.add_argument("--max_runtime_rows", type=int, default=-1)
    ap.add_argument("--max_teacher_ready_rows", type=int, default=-1)
    ap.add_argument("--max_xy_block_rows", type=int, default=1200)
    ap.add_argument("--max_yaw_needed_rows", type=int, default=900)
    ap.add_argument("--max_far_negative_rows", type=int, default=1600)
    ap.add_argument("--very_near_max_norm", type=float, default=1.8)
    ap.add_argument("--xy_block_z_max", type=float, default=1.2)
    ap.add_argument("--xy_block_yaw_max", type=float, default=1.2)
    ap.add_argument("--yaw_needed_z_max", type=float, default=1.2)
    ap.add_argument("--yaw_needed_xy_max", type=float, default=1.3)
    ap.add_argument("--far_negative_min_norm", type=float, default=2.5)
    ap.add_argument("--min_teacher_ready_eps", type=int, default=3)
    ap.add_argument("--min_yaw_needed_eps", type=int, default=1)
    ap.add_argument("--weight_teacher_ready", type=float, default=1.8)
    ap.add_argument("--weight_xy_block", type=float, default=1.5)
    ap.add_argument("--weight_yaw_needed", type=float, default=1.6)
    ap.add_argument("--weight_far_negative", type=float, default=1.2)
    args = ap.parse_args()

    raw = np.load(args.input_npz, allow_pickle=False)
    data = {k: np.asarray(raw[k]) for k in raw.files}
    n = int(data["teacher_truth_handoff_ready"].shape[0])
    rng = _rng(args.seed)

    episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    source_name = np.asarray(data.get("source_name", np.full((n,), "unknown", dtype="U32"))).astype(str)
    teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    teacher_metrics = np.asarray(data["teacher_metrics_norm"], dtype=np.float32)
    xy = teacher_metrics[:, 0]
    z = teacher_metrics[:, 1]
    yaw = teacher_metrics[:, 2]
    max_norm = np.maximum(np.maximum(xy, z), yaw)
    teacher_band = np.asarray(data["teacher_band_label"], dtype=np.int64)

    very_near = max_norm <= float(args.very_near_max_norm)
    xy_block = very_near & (xy > 1.0) & (z <= float(args.xy_block_z_max)) & (yaw <= float(args.xy_block_yaw_max))
    yaw_needed = very_near & (yaw > 1.0) & (z <= float(args.yaw_needed_z_max)) & (xy <= float(args.yaw_needed_xy_max))
    far_negative = (teacher_ready <= 0) & (teacher_band == 0) & (
        (xy >= float(args.far_negative_min_norm))
        | (z >= float(args.far_negative_min_norm))
        | (yaw >= float(args.far_negative_min_norm))
    )
    runtime_base = source_name == str(args.runtime_source_name)

    runtime_idx = np.flatnonzero(runtime_base)
    runtime_idx = _sample_rows(rng, runtime_idx, int(args.max_runtime_rows))
    ready_idx = np.flatnonzero(teacher_ready)
    ready_idx = _sample_teacher_ready_balanced(rng, ready_idx, episode_index, int(args.max_teacher_ready_rows))
    xy_block_idx = _sample_rows(rng, np.flatnonzero(xy_block), int(args.max_xy_block_rows))
    yaw_needed_idx = _sample_rows(rng, np.flatnonzero(yaw_needed), int(args.max_yaw_needed_rows))
    far_negative_idx = _sample_rows(rng, np.flatnonzero(far_negative), int(args.max_far_negative_rows))

    keep_idx = np.unique(np.concatenate([runtime_idx, ready_idx, xy_block_idx, yaw_needed_idx, far_negative_idx]))
    if keep_idx.size == 0:
        raise RuntimeError("No rows selected for distill dataset v1.")

    selected_teacher_ready_eps = np.unique(episode_index[keep_idx][teacher_ready[keep_idx]])
    if int(selected_teacher_ready_eps.size) < int(args.min_teacher_ready_eps):
        raise RuntimeError(
            f"teacher_ready episode coverage too low: {selected_teacher_ready_eps.size} < {args.min_teacher_ready_eps}"
        )
    selected_yaw_needed_eps = np.unique(episode_index[keep_idx][yaw_needed[keep_idx]])
    if int(selected_yaw_needed_eps.size) < int(args.min_yaw_needed_eps):
        raise RuntimeError(
            f"yaw_needed episode coverage too low: {selected_yaw_needed_eps.size} < {args.min_yaw_needed_eps}"
        )

    out = {}
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[k] = arr[keep_idx]
        else:
            out[k] = arr

    # Add explicit v1 bucket flags for later diagnostics.
    out["teacher_ready_v1"] = teacher_ready[keep_idx].astype(np.float32)
    out["very_near_v1"] = very_near[keep_idx].astype(np.float32)
    out["xy_block_v1"] = xy_block[keep_idx].astype(np.float32)
    out["yaw_needed_v1"] = yaw_needed[keep_idx].astype(np.float32)
    out["far_negative_v1"] = far_negative[keep_idx].astype(np.float32)
    out["runtime_like_v1"] = runtime_base[keep_idx].astype(np.float32)

    # Reweight sample_weight by bucket intent (multiplicative, runtime-safe).
    sw = np.asarray(out.get("sample_weight", np.ones((keep_idx.size,), dtype=np.float32)), dtype=np.float32).copy()
    sw *= np.where(out["teacher_ready_v1"] > 0.5, float(args.weight_teacher_ready), 1.0).astype(np.float32)
    sw *= np.where(out["xy_block_v1"] > 0.5, float(args.weight_xy_block), 1.0).astype(np.float32)
    sw *= np.where(out["yaw_needed_v1"] > 0.5, float(args.weight_yaw_needed), 1.0).astype(np.float32)
    sw *= np.where(out["far_negative_v1"] > 0.5, float(args.weight_far_negative), 1.0).astype(np.float32)
    out["sample_weight"] = sw.astype(np.float32)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    full_summary = _subset_summary(
        np.arange(n, dtype=np.int64), episode_index, teacher_ready, xy_block, yaw_needed, far_negative
    )
    kept_summary = _subset_summary(
        keep_idx, episode_index, teacher_ready, xy_block, yaw_needed, far_negative
    )
    meta = {
        "input_npz": str(args.input_npz),
        "output_npz": str(output_npz),
        "seed": int(args.seed),
        "runtime_source_name": str(args.runtime_source_name),
        "selection_caps": {
            "max_runtime_rows": int(args.max_runtime_rows),
            "max_teacher_ready_rows": int(args.max_teacher_ready_rows),
            "max_xy_block_rows": int(args.max_xy_block_rows),
            "max_yaw_needed_rows": int(args.max_yaw_needed_rows),
            "max_far_negative_rows": int(args.max_far_negative_rows),
        },
        "thresholds": {
            "very_near_max_norm": float(args.very_near_max_norm),
            "xy_block_z_max": float(args.xy_block_z_max),
            "xy_block_yaw_max": float(args.xy_block_yaw_max),
            "yaw_needed_z_max": float(args.yaw_needed_z_max),
            "yaw_needed_xy_max": float(args.yaw_needed_xy_max),
            "far_negative_min_norm": float(args.far_negative_min_norm),
            "min_teacher_ready_eps": int(args.min_teacher_ready_eps),
            "min_yaw_needed_eps": int(args.min_yaw_needed_eps),
        },
        "weights": {
            "teacher_ready": float(args.weight_teacher_ready),
            "xy_block": float(args.weight_xy_block),
            "yaw_needed": float(args.weight_yaw_needed),
            "far_negative": float(args.weight_far_negative),
        },
        "full_summary": full_summary,
        "kept_summary": kept_summary,
        "selected_teacher_ready_eps": [int(x) for x in np.unique(out["episode_index"][out["teacher_ready_v1"] > 0.5]).tolist()],
        "selected_yaw_needed_eps": [int(x) for x in np.unique(out["episode_index"][out["yaw_needed_v1"] > 0.5]).tolist()],
    }
    meta_path = Path(args.meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
