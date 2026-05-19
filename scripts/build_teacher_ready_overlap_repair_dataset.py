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
    ap.add_argument("--calibration_negative_episodes", action="append", default=[])
    ap.add_argument("--teacher_ready_window", type=int, default=12)
    ap.add_argument("--teacher_ready_focus_mult", type=float, default=4.0)
    ap.add_argument("--teacher_ready_exact_mult", type=float, default=2.0)
    ap.add_argument("--calibration_negative_mult", type=float, default=0.5)
    args = ap.parse_args()

    arr = np.load(args.dataset_npz, allow_pickle=False)
    data = {k: np.asarray(arr[k]) for k in arr.files}
    n = int(data["sample_weight"].shape[0])
    source_name = np.asarray(data.get("source_name", np.full((n,), "unknown", dtype="U128"))).astype(str)
    episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    sample_weight = np.asarray(data["sample_weight"], dtype=np.float32).copy()

    focus_episodes = parse_int_list(args.teacher_ready_focus_episodes)
    calibration_episodes = parse_int_list(args.calibration_negative_episodes)

    focus_source_mask = source_name == str(args.focus_source)
    overlap_focus_mask = np.zeros((n,), dtype=bool)
    overlap_teacher_ready_exact = np.zeros((n,), dtype=bool)

    for ep in focus_episodes:
        ep_mask = focus_source_mask & (episode_index == int(ep))
        ep_idx = np.flatnonzero(ep_mask)
        if ep_idx.size == 0:
            continue
        ep_teacher_ready_local = np.flatnonzero(teacher_ready[ep_mask])
        if ep_teacher_ready_local.size == 0:
            continue
        local_window = contiguous_window_mask(ep_idx.size, ep_teacher_ready_local, int(args.teacher_ready_window))
        overlap_focus_mask[ep_idx[local_window]] = True
        overlap_teacher_ready_exact[ep_idx[ep_teacher_ready_local]] = True

    calibration_negative_mask = np.zeros((n,), dtype=bool)
    for ep in calibration_episodes:
        calibration_negative_mask |= focus_source_mask & (episode_index == int(ep))

    sample_weight[overlap_focus_mask] *= float(args.teacher_ready_focus_mult)
    sample_weight[overlap_teacher_ready_exact] *= float(args.teacher_ready_exact_mult)
    sample_weight[calibration_negative_mask] *= float(args.calibration_negative_mult)

    data["sample_weight"] = sample_weight.astype(np.float32)
    data["overlap_focus_mask"] = overlap_focus_mask.astype(np.float32)
    data["overlap_teacher_ready_exact"] = overlap_teacher_ready_exact.astype(np.float32)
    data["calibration_negative_mask"] = calibration_negative_mask.astype(np.float32)

    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **data)

    meta = {
        "dataset_npz": str(args.dataset_npz),
        "focus_source": str(args.focus_source),
        "teacher_ready_focus_episodes": focus_episodes,
        "calibration_negative_episodes": calibration_episodes,
        "teacher_ready_window": int(args.teacher_ready_window),
        "teacher_ready_focus_mult": float(args.teacher_ready_focus_mult),
        "teacher_ready_exact_mult": float(args.teacher_ready_exact_mult),
        "calibration_negative_mult": float(args.calibration_negative_mult),
        "rows": n,
        "overlap_focus_rows": int(np.sum(overlap_focus_mask)),
        "overlap_teacher_ready_exact_rows": int(np.sum(overlap_teacher_ready_exact)),
        "calibration_negative_rows": int(np.sum(calibration_negative_mask)),
        "sample_weight_mean_before": float(np.mean(np.asarray(arr["sample_weight"], dtype=np.float32))),
        "sample_weight_mean_after": float(np.mean(sample_weight)),
    }
    Path(args.meta_json).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
