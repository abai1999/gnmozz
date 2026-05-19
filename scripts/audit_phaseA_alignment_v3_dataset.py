#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_stat(arr: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    arr = np.load(args.dataset_npz, allow_pickle=False)
    data = {k: np.asarray(arr[k]) for k in arr.files}
    n = int(data["sample_weight"].shape[0])
    source = np.asarray(data.get("source_name", np.full((n,), "unknown", dtype="U64"))).astype(str)
    stage_bucket = np.asarray(data.get("alignment_v3_stage_bucket", np.full((n,), "unknown", dtype="U64"))).astype(str)
    teacher_ready = np.asarray(data.get("teacher_truth_handoff_ready", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    closeability = np.asarray(data.get("alignment_v3_closeability_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    borderline = np.asarray(data.get("alignment_v3_closeability_borderline_mask", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    corrective_mask = np.asarray(data.get("alignment_v3_corrective_mask", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    cf = np.asarray(data.get("is_counterfactual", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    progress_mask = np.asarray(data.get("alignment_v2_progress_mask", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    progress_pos = np.asarray(data.get("alignment_v2_progress_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    pair_mask = np.asarray(data.get("alignment_v2_pair_mask", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5

    report = {
        "dataset_npz": str(Path(args.dataset_npz).resolve()),
        "rows": n,
        "real_rows": int(np.sum(~cf)),
        "counterfactual_rows": int(np.sum(cf)),
        "teacher_ready_rows": int(np.sum(teacher_ready)),
        "closeability_positive_rows": int(np.sum(closeability)),
        "closeability_borderline_rows": int(np.sum(borderline)),
        "corrective_mask_rows": int(np.sum(corrective_mask)),
        "progress_mask_rows": int(np.sum(progress_mask)),
        "progress_positive_rows": int(np.sum(progress_mask & progress_pos)),
        "pair_mask_rows": int(np.sum(pair_mask)),
        "source_counts": {str(src): int(np.sum(source == src)) for src in sorted(set(source.tolist()))},
        "stage_bucket_counts": {str(stage): int(np.sum(stage_bucket == stage)) for stage in sorted(set(stage_bucket.tolist()))},
        "teacher_ready_by_stage": {str(stage): int(np.sum((stage_bucket == stage) & teacher_ready)) for stage in sorted(set(stage_bucket.tolist()))},
        "closeability_by_stage": {str(stage): int(np.sum((stage_bucket == stage) & closeability)) for stage in sorted(set(stage_bucket.tolist()))},
        "sample_weight": _safe_stat(np.asarray(data.get("sample_weight", np.ones((n,), dtype=np.float32)), dtype=np.float64)),
        "teacher_xy_norm": _safe_stat(np.asarray(data.get("teacher_xy_norm", np.zeros((n,), dtype=np.float32)), dtype=np.float64)),
        "teacher_abs_z_norm": _safe_stat(np.asarray(data.get("teacher_abs_z_norm", np.zeros((n,), dtype=np.float32)), dtype=np.float64)),
        "teacher_yaw_norm": _safe_stat(np.asarray(data.get("teacher_yaw_norm", np.zeros((n,), dtype=np.float32)), dtype=np.float64)),
        "alignment_v2_weighted_sum_norm": _safe_stat(np.asarray(data.get("alignment_v2_weighted_sum_norm", np.zeros((n,), dtype=np.float32)), dtype=np.float64)),
        "alignment_v2_max_axis_norm": _safe_stat(np.asarray(data.get("alignment_v2_max_axis_norm", np.zeros((n,), dtype=np.float32)), dtype=np.float64)),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
