#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_int_list(items):
    out = []
    for item in items or []:
        if item is None:
            continue
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return sorted(set(out))


def parse_source_multipliers(items):
    out = {}
    for item in items or []:
        if not item:
            continue
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"invalid source multiplier `{part}`; expected source:mult")
            name, mult = part.split(":", 1)
            out[str(name)] = float(mult)
    return out


def contiguous_window_mask(length: int, centers: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros((length,), dtype=bool)
    if centers.size == 0:
        return mask
    for c in centers.tolist():
        lo = max(0, int(c) - int(radius))
        hi = min(length, int(c) + int(radius) + 1)
        mask[lo:hi] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--focus_source", default="learned32")
    ap.add_argument("--teacher_ready_focus_episodes", action="append", default=[])
    ap.add_argument("--teacher_ready_window", type=int, default=16)
    ap.add_argument("--focus_window_mult", type=float, default=4.0)
    ap.add_argument("--teacher_ready_exact_mult", type=float, default=3.0)
    ap.add_argument("--boundary_band_mult", type=float, default=3.0)
    ap.add_argument("--boundary_xy_max", type=float, default=1.15)
    ap.add_argument("--boundary_z_max", type=float, default=1.0)
    ap.add_argument("--boundary_yaw_max", type=float, default=1.0)
    ap.add_argument("--far_negative_mult", type=float, default=1.75)
    ap.add_argument("--far_negative_min", type=float, default=2.5)
    ap.add_argument("--source_mult", action="append", default=[])
    args = ap.parse_args()

    arr = np.load(args.dataset_npz, allow_pickle=False)
    data = {k: np.asarray(arr[k]) for k in arr.files}
    n = int(data["sample_weight"].shape[0])
    source_name = np.asarray(data.get("source_name", np.full((n,), "unknown", dtype="U128"))).astype(str)
    episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    teacher_band = np.asarray(data["teacher_band_label"], dtype=np.int64)
    teacher_xy = np.asarray(data["teacher_xy_norm"], dtype=np.float32)
    teacher_z = np.asarray(data["teacher_abs_z_norm"], dtype=np.float32)
    teacher_yaw = np.asarray(data["teacher_yaw_norm"], dtype=np.float32)
    sample_weight = np.asarray(data["sample_weight"], dtype=np.float32).copy()

    focus_episodes = parse_int_list(args.teacher_ready_focus_episodes)
    source_mult = parse_source_multipliers(args.source_mult)
    focus_source_mask = source_name == str(args.focus_source)

    focus_window_mask = np.zeros((n,), dtype=bool)
    teacher_ready_exact_mask = np.zeros((n,), dtype=bool)
    boundary_band_mask = np.zeros((n,), dtype=bool)

    for ep in focus_episodes:
        ep_mask = focus_source_mask & (episode_index == int(ep))
        ep_idx = np.flatnonzero(ep_mask)
        if ep_idx.size == 0:
            continue
        ep_teacher_ready_local = np.flatnonzero(teacher_ready[ep_mask])
        if ep_teacher_ready_local.size == 0:
            continue
        local_window = contiguous_window_mask(ep_idx.size, ep_teacher_ready_local, int(args.teacher_ready_window))
        focus_window_mask[ep_idx[local_window]] = True
        teacher_ready_exact_mask[ep_idx[ep_teacher_ready_local]] = True

    # Teacher-ready neighborhood but not yet ready: learn the band->ready transition.
    boundary_band_mask = (
        focus_window_mask
        & (~teacher_ready)
        & (teacher_band > 0)
        & (teacher_xy <= float(args.boundary_xy_max))
        & (teacher_z <= float(args.boundary_z_max))
        & (teacher_yaw <= float(args.boundary_yaw_max))
    )

    # Calibration negatives: clearly not-ready rows on the learned mainline.
    far_negative_mask = (
        focus_source_mask
        & (~teacher_ready)
        & (teacher_band == 0)
        & (
            (teacher_xy >= float(args.far_negative_min))
            | (teacher_z >= float(args.far_negative_min))
            | (teacher_yaw >= float(args.far_negative_min))
        )
    )

    sample_weight[focus_window_mask] *= float(args.focus_window_mult)
    sample_weight[teacher_ready_exact_mask] *= float(args.teacher_ready_exact_mult)
    sample_weight[boundary_band_mask] *= float(args.boundary_band_mult)
    sample_weight[far_negative_mask] *= float(args.far_negative_mult)

    for src, mult in source_mult.items():
        sample_weight[source_name == src] *= float(mult)

    data["sample_weight"] = sample_weight.astype(np.float32)
    data["focus_window_mask_v2"] = focus_window_mask.astype(np.float32)
    data["teacher_ready_exact_mask_v2"] = teacher_ready_exact_mask.astype(np.float32)
    data["boundary_band_mask_v2"] = boundary_band_mask.astype(np.float32)
    data["far_negative_mask_v2"] = far_negative_mask.astype(np.float32)

    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **data)

    meta = {
        "dataset_npz": str(args.dataset_npz),
        "focus_source": str(args.focus_source),
        "teacher_ready_focus_episodes": focus_episodes,
        "teacher_ready_window": int(args.teacher_ready_window),
        "focus_window_mult": float(args.focus_window_mult),
        "teacher_ready_exact_mult": float(args.teacher_ready_exact_mult),
        "boundary_band_mult": float(args.boundary_band_mult),
        "boundary_xy_max": float(args.boundary_xy_max),
        "boundary_z_max": float(args.boundary_z_max),
        "boundary_yaw_max": float(args.boundary_yaw_max),
        "far_negative_mult": float(args.far_negative_mult),
        "far_negative_min": float(args.far_negative_min),
        "source_mult": source_mult,
        "rows": n,
        "focus_window_rows": int(np.sum(focus_window_mask)),
        "teacher_ready_exact_rows": int(np.sum(teacher_ready_exact_mask)),
        "boundary_band_rows": int(np.sum(boundary_band_mask)),
        "far_negative_rows": int(np.sum(far_negative_mask)),
        "sample_weight_mean_before": float(np.mean(np.asarray(arr["sample_weight"], dtype=np.float32))),
        "sample_weight_mean_after": float(np.mean(sample_weight)),
    }
    Path(args.meta_json).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
