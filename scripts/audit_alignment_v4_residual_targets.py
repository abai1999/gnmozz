#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    raw = np.load(args.dataset_npz, allow_pickle=False)
    residual = np.asarray(raw["alignment_v4_residual_target"], dtype=np.float32)
    improve = np.asarray(raw["alignment_v4_improvement"], dtype=np.float32)
    improve_label = np.asarray(raw["alignment_v4_improvement_label"], dtype=np.float32)
    conf = np.asarray(raw["alignment_v4_residual_confidence_target"], dtype=np.float32)
    focus = np.asarray(raw.get("alignment_v4_focus_mask", np.zeros((residual.shape[0],), dtype=np.float32)), dtype=np.float32)
    close = np.asarray(raw.get("alignment_v4_closeability_label", np.zeros((residual.shape[0],), dtype=np.float32)), dtype=np.float32)
    teacher_metrics = np.asarray(raw.get("teacher_metrics_norm", np.zeros((residual.shape[0], 3), dtype=np.float32)), dtype=np.float32)
    src_rank = np.asarray(raw.get("alignment_v4_residual_source_rank", np.zeros((residual.shape[0],), dtype=np.int64)), dtype=np.int64)

    xyz_norm = np.linalg.norm(residual[:, :3], axis=1)
    yaw_abs = np.abs(residual[:, 3])
    report = {
        "rows": int(residual.shape[0]),
        "focus_rows": int(np.sum(focus > 0.5)),
        "closeability_positive_rows": int(np.sum(close > 0.5)),
        "improvement_positive_rows": int(np.sum(improve_label > 0.5)),
        "xyz_norm": {
            "mean": float(np.mean(xyz_norm)),
            "p50": float(np.percentile(xyz_norm, 50.0)),
            "p90": float(np.percentile(xyz_norm, 90.0)),
            "near_clip_rate_90pct_bound": float(np.mean(xyz_norm >= 0.9 * 0.006)),
        },
        "yaw_abs": {
            "mean": float(np.mean(yaw_abs)),
            "p50": float(np.percentile(yaw_abs, 50.0)),
            "p90": float(np.percentile(yaw_abs, 90.0)),
            "nonzero_rate": float(np.mean(yaw_abs > 1e-5)),
        },
        "improvement": {
            "mean": float(np.mean(improve)),
            "p50": float(np.percentile(improve, 50.0)),
            "p90": float(np.percentile(improve, 90.0)),
        },
        "confidence_target": {
            "mean": float(np.mean(conf)),
            "p50": float(np.percentile(conf, 50.0)),
            "p90": float(np.percentile(conf, 90.0)),
            "positive_rate": float(np.mean(conf > 0.05)),
        },
        "teacher_metrics_norm": {
            "xy_mean": float(np.mean(teacher_metrics[:, 0])),
            "z_mean": float(np.mean(teacher_metrics[:, 1])),
            "yaw_mean": float(np.mean(teacher_metrics[:, 2])),
        },
        "residual_source_rank_counts": {
            str(int(k)): int(v) for k, v in zip(*np.unique(src_rank, return_counts=True))
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
