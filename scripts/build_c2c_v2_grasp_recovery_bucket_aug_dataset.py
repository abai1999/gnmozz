#!/usr/bin/env python3
"""Build a bucket-balanced recovery augmentation dataset from failure morphologies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.recovery_augmentation import (
    augment_recovery_record,
    failure_morphology_bucket,
)


BUCKET_SCALE_PRESETS: dict[str, list[float]] = {
    "large_xy_large_yaw": [1.05, 1.15, 1.25],
    "large_xy_small_yaw": [1.10, 1.25, 1.40],
    "small_xy_large_yaw": [1.00, 1.10, 1.20],
    "small_xy_small_yaw": [1.00, 1.05, 1.10],
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dataset", type=Path, required=True)
    ap.add_argument("--bucket_report", type=Path, required=True)
    ap.add_argument("--bias_template", type=Path, default=None)
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets_recovery_bucket_aug"))
    ap.add_argument("--focus_buckets", type=str, default="")
    ap.add_argument("--augment_multiplier", type=int, default=2, help="Number of augmented copies per selected row per bucket.")
    ap.add_argument("--xy_threshold", type=float, default=0.06)
    ap.add_argument("--yaw_threshold", type=float, default=0.15)
    args = ap.parse_args()

    base_records = _load_jsonl(args.base_dataset)
    bucket_report = json.loads(args.bucket_report.read_text(encoding="utf-8"))
    bias_template = json.loads(args.bias_template.read_text(encoding="utf-8")) if args.bias_template else bucket_report

    if args.focus_buckets.strip():
        focus_buckets = set(_parse_csv(args.focus_buckets))
    else:
        focus_buckets = set()
        for item in bucket_report.get("hard_buckets", [])[:2]:
            bucket_name = str(item.get("bucket", "")).strip()
            if bucket_name:
                focus_buckets.add(bucket_name)
        if not focus_buckets:
            hard_order = _parse_csv(",".join(str(x) for x in bucket_report.get("bucket_order_by_gain", [])))
            focus_buckets = set(hard_order[:2])
    if not focus_buckets:
        focus_buckets = {"large_xy_small_yaw", "small_xy_small_yaw"}

    base_rows = [dict(r) for r in base_records]
    bucketed_rows = []
    selected_indices: list[int] = []
    bucket_counts: dict[str, int] = {}
    for idx, record in enumerate(base_rows):
        bucket = failure_morphology_bucket(record, xy_threshold=float(args.xy_threshold), yaw_threshold=float(args.yaw_threshold))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if bucket not in focus_buckets:
            continue
        selected_indices.append(idx)
        scales = BUCKET_SCALE_PRESETS.get(bucket, [1.05, 1.10, 1.15])
        for copy_idx in range(max(int(args.augment_multiplier), 1)):
            scale = float(scales[copy_idx % len(scales)])
            bucketed_rows.append(
                augment_recovery_record(
                    record,
                    scale=scale,
                    template=bias_template,
                    source_index=idx,
                )
            )

    combined = base_rows + bucketed_rows

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "grasp_recovery_dataset_v4_bucket_aug.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "base_dataset": str(args.base_dataset),
        "bucket_report": str(args.bucket_report),
        "bias_template": str(args.bias_template) if args.bias_template else "",
        "output_dataset": str(out_path),
        "num_base_rows": len(base_rows),
        "num_total_rows": len(combined),
        "num_augmented_rows": len(bucketed_rows),
        "focus_buckets": sorted(focus_buckets),
        "augment_multiplier": int(args.augment_multiplier),
        "xy_threshold": float(args.xy_threshold),
        "yaw_threshold": float(args.yaw_threshold),
        "bucket_counts": bucket_counts,
        "selected_source_count": len(selected_indices),
        "selected_source_indices": selected_indices,
        "bucket_scale_presets": BUCKET_SCALE_PRESETS,
    }
    summary_path = out_root / "grasp_recovery_dataset_v4_bucket_aug_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
