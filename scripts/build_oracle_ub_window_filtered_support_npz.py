#!/usr/bin/env python3
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Filter oracle-target-upper-bound support rows into runtime-safe window-level "
            "teacher label subset: teacher_ready/release/very_near(max_norm<=X), with yaw outlier control."
        )
    )
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--phase_id", type=int, default=1)
    ap.add_argument("--open_threshold", type=float, default=0.5)
    ap.add_argument("--max_norm_very_near", type=float, default=1.8)
    ap.add_argument("--yaw_norm_max", type=float, default=2.0)
    ap.add_argument("--fallback_release_xy", type=float, default=0.0085)
    ap.add_argument("--fallback_release_z", type=float, default=0.0035)
    ap.add_argument("--fallback_release_yaw", type=float, default=0.1243404)
    ap.add_argument(
        "--drop_yaw_outlier",
        action="store_true",
        default=True,
        help="Drop rows with yaw_norm > yaw_norm_max. If disabled, rows are kept and flagged only in meta.",
    )
    ap.add_argument(
        "--no_drop_yaw_outlier",
        dest="drop_yaw_outlier",
        action="store_false",
    )
    args = ap.parse_args()

    raw = np.load(args.input_npz, allow_pickle=False)
    data = {k: np.asarray(raw[k]) for k in raw.files}

    n = int(data["teacher_truth_handoff_ready"].shape[0])
    phase_id = np.asarray(data.get("phase_id", np.ones((n,), dtype=np.int64)), dtype=np.int64)
    gripper_open = np.asarray(data.get("rollout_gripper_open", np.ones((n,), dtype=np.float32)), dtype=np.float32)

    teacher_ready = np.asarray(data.get("teacher_truth_handoff_ready", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    teacher_xy = np.asarray(data["teacher_truth_handoff_metric_xy_error"], dtype=np.float32)
    teacher_z = np.asarray(data["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float32)
    teacher_yaw = np.asarray(data["teacher_truth_handoff_metric_yaw_error"], dtype=np.float32)

    rel_xy = _safe_threshold(
        np.asarray(
            data.get(
                "teacher_truth_handoff_release_threshold_xy_error",
                np.full((n,), np.nan, dtype=np.float32),
            ),
            dtype=np.float32,
        ),
        float(args.fallback_release_xy),
    )
    rel_z = _safe_threshold(
        np.asarray(
            data.get(
                "teacher_truth_handoff_release_threshold_abs_z_error",
                np.full((n,), np.nan, dtype=np.float32),
            ),
            dtype=np.float32,
        ),
        float(args.fallback_release_z),
    )
    rel_yaw = _safe_threshold(
        np.asarray(
            data.get(
                "teacher_truth_handoff_release_threshold_yaw_error",
                np.full((n,), np.nan, dtype=np.float32),
            ),
            dtype=np.float32,
        ),
        float(args.fallback_release_yaw),
    )

    base = (phase_id == int(args.phase_id)) & (gripper_open >= float(args.open_threshold))
    finite = np.isfinite(teacher_xy) & np.isfinite(teacher_z) & np.isfinite(teacher_yaw)

    xy_norm = teacher_xy / np.maximum(rel_xy, 1e-6)
    z_norm = teacher_z / np.maximum(rel_z, 1e-6)
    yaw_norm = teacher_yaw / np.maximum(rel_yaw, 1e-6)
    max_norm = np.maximum(np.maximum(xy_norm, z_norm), yaw_norm)

    release = finite & (xy_norm <= 1.0) & (z_norm <= 1.0) & (yaw_norm <= 1.0)
    very_near = finite & (max_norm <= float(args.max_norm_very_near))
    keep_window = teacher_ready | release | very_near
    yaw_outlier = finite & (yaw_norm > float(args.yaw_norm_max))

    keep = base & finite & keep_window
    if bool(args.drop_yaw_outlier):
        keep = keep & (~yaw_outlier)

    keep_idx = np.flatnonzero(keep)
    if keep_idx.size == 0:
        raise RuntimeError("No rows left after oracle-UB window filtering.")

    out = {}
    for k, v in data.items():
        arr = np.asarray(v)
        # Slice only row-aligned arrays; keep metadata / config arrays untouched.
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[k] = np.asarray(arr[keep_idx])
        else:
            out[k] = arr

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    meta = {
        "input_npz": str(args.input_npz),
        "output_npz": str(output_npz),
        "rows_input": int(n),
        "rows_base_phase_open": int(np.sum(base)),
        "rows_teacher_ready_raw": int(np.sum(base & teacher_ready)),
        "rows_release_raw": int(np.sum(base & release)),
        "rows_very_near_raw": int(np.sum(base & very_near)),
        "rows_keep_before_yaw_filter": int(np.sum(base & finite & keep_window)),
        "rows_yaw_outlier_in_keep": int(np.sum(base & finite & keep_window & yaw_outlier)),
        "drop_yaw_outlier": bool(args.drop_yaw_outlier),
        "yaw_norm_max": float(args.yaw_norm_max),
        "max_norm_very_near": float(args.max_norm_very_near),
        "rows_output": int(keep_idx.size),
        "teacher_ready_rows_output": int(np.sum(teacher_ready[keep_idx])),
        "release_rows_output": int(np.sum(release[keep_idx])),
        "very_near_rows_output": int(np.sum(very_near[keep_idx])),
        "teacher_ready_eps_output": int(
            np.unique(np.asarray(data["episode_index"], dtype=np.int64)[keep_idx][teacher_ready[keep_idx]]).size
        )
        if np.any(teacher_ready[keep_idx])
        else 0,
        "release_eps_output": int(
            np.unique(np.asarray(data["episode_index"], dtype=np.int64)[keep_idx][release[keep_idx]]).size
        )
        if np.any(release[keep_idx])
        else 0,
        "very_near_eps_output": int(
            np.unique(np.asarray(data["episode_index"], dtype=np.int64)[keep_idx][very_near[keep_idx]]).size
        )
        if np.any(very_near[keep_idx])
        else 0,
    }
    meta_path = Path(args.meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
