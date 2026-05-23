#!/usr/bin/env python3
"""Build a narrow tail-bucket recovery dataset focused on the hardest failure morphologies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.recovery_augmentation import build_bucket_tail_replay_augmented_records


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dataset", type=Path, required=True)
    ap.add_argument("--bucket_report", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets_recovery_tailbucket"))
    ap.add_argument("--focus_buckets", type=str, default="")
    ap.add_argument("--trajectory_fraction", type=float, default=0.60)
    ap.add_argument("--min_trajectories_per_bucket", type=int, default=2)
    ap.add_argument("--tail_rows", type=int, default=6)
    ap.add_argument("--min_tail_rows_per_bucket", type=int, default=10)
    ap.add_argument("--xy_threshold", type=float, default=0.06)
    ap.add_argument("--yaw_threshold", type=float, default=0.15)
    ap.add_argument("--small_xy_small_yaw_modes", type=str, default="oscillate,shear")
    ap.add_argument("--large_xy_small_yaw_modes", type=str, default="overshoot,cross_couple")
    ap.add_argument("--small_xy_small_yaw_strengths", type=str, default="1.03,1.07,1.12")
    ap.add_argument("--large_xy_small_yaw_strengths", type=str, default="1.08,1.14,1.20")
    ap.add_argument("--small_xy_small_yaw_drifts", type=str, default="0.05,0.10,0.15")
    ap.add_argument("--large_xy_small_yaw_drifts", type=str, default="0.08,0.12,0.18")
    ap.add_argument("--small_xy_small_yaw_weight", type=float, default=1.80)
    ap.add_argument("--large_xy_small_yaw_weight", type=float, default=1.40)
    args = ap.parse_args()

    base_records = _load_jsonl(args.base_dataset)
    bucket_report = json.loads(args.bucket_report.read_text(encoding="utf-8"))
    if args.focus_buckets.strip():
        focus_buckets = set(_parse_csv(args.focus_buckets))
    else:
        focus_buckets = set()
        for item in bucket_report.get("hard_buckets", [])[:2]:
            bucket_name = str(item.get("bucket", "")).strip()
            if bucket_name:
                focus_buckets.add(bucket_name)
        if not focus_buckets:
            focus_buckets = {"small_xy_small_yaw", "large_xy_small_yaw"}

    replay_modes_by_bucket = {
        "small_xy_small_yaw": _parse_csv(args.small_xy_small_yaw_modes),
        "large_xy_small_yaw": _parse_csv(args.large_xy_small_yaw_modes),
    }
    replay_strengths_by_bucket = {
        "small_xy_small_yaw": _parse_float_list(args.small_xy_small_yaw_strengths),
        "large_xy_small_yaw": _parse_float_list(args.large_xy_small_yaw_strengths),
    }
    drift_strengths_by_bucket = {
        "small_xy_small_yaw": _parse_float_list(args.small_xy_small_yaw_drifts),
        "large_xy_small_yaw": _parse_float_list(args.large_xy_small_yaw_drifts),
    }
    bucket_weight_by_bucket = {
        "small_xy_small_yaw": float(args.small_xy_small_yaw_weight),
        "large_xy_small_yaw": float(args.large_xy_small_yaw_weight),
    }

    combined, report = build_bucket_tail_replay_augmented_records(
        base_records,
        focus_buckets=sorted(focus_buckets),
        xy_threshold=float(args.xy_threshold),
        yaw_threshold=float(args.yaw_threshold),
        trajectory_fraction=float(args.trajectory_fraction),
        min_trajectories_per_bucket=int(args.min_trajectories_per_bucket),
        tail_rows=int(args.tail_rows),
        replay_strengths_by_bucket=replay_strengths_by_bucket,
        drift_strengths_by_bucket=drift_strengths_by_bucket,
        replay_modes_by_bucket=replay_modes_by_bucket,
        bucket_weight_by_bucket=bucket_weight_by_bucket,
        min_tail_rows_per_bucket=int(args.min_tail_rows_per_bucket),
    )

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "grasp_recovery_dataset_v5_tailbucket.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "base_dataset": str(args.base_dataset),
        "bucket_report": str(args.bucket_report),
        "output_dataset": str(out_path),
        "num_base_rows": len(base_records),
        "num_total_rows": len(combined),
        "num_augmented_rows": len(combined) - len(base_records),
        **report,
    }
    summary_path = out_root / "grasp_recovery_dataset_v5_tailbucket_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
