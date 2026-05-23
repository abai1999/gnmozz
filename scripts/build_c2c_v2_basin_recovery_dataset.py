#!/usr/bin/env python3
"""Build basin-recovery training records from runtime failure-tail samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.basin_recovery import (  # noqa: E402
    BasinRecoveryConfig,
    basin_recovery_feature_vector,
    classify_basin_label,
    classify_visual_evidence_for_basin,
    target_error_from_record,
)
from prismatic.robot.coarse2contact_v2.datasets import read_jsonl  # noqa: E402
from prismatic.robot.coarse2contact_v2.recovery_augmentation import failure_morphology_bucket  # noqa: E402
from prismatic.robot.coarse2contact_v2.recovery_audit import recovery_error_norm  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def build_rows(records: list[dict[str, Any]], *, config: BasinRecoveryConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    evidence_counts: dict[str, int] = {}
    basin_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}

    for record in records:
        if str(record.get("view_name", "")) != "wrist":
            continue
        if not record.get("rgb_path") or not record.get("depth_path"):
            continue
        target_error = target_error_from_record(record)
        evidence = classify_visual_evidence_for_basin(record, config=config).value
        basin = classify_basin_label(target_error, config=config).value
        bucket = failure_morphology_bucket(record)
        target_norm = recovery_error_norm(float(target_error[0]), float(target_error[1]), float(target_error[2]))
        xy = float(np.linalg.norm(target_error[:2]))

        row = dict(record)
        row.update(
            {
                "dataset_type": "basin_recovery",
                "visual_evidence_class": evidence,
                "failure_bucket": bucket,
                "initial_error_dx": float(target_error[0]),
                "initial_error_dy": float(target_error[1]),
                "initial_error_dz": _float(record, "recovery_target_dz", _float(record, "trace_error_dz", 0.0)),
                "initial_error_dyaw": float(target_error[2]),
                "initial_error_norm": float(target_norm),
                "initial_error_xy": xy,
                "basin_label": basin,
                "reacquire_needed": bool(evidence != "visual_observable"),
                "pullback_allowed": bool(evidence == "visual_observable" and basin == "outside"),
                "micro_servo_allowed": bool(evidence == "visual_observable" and basin in {"near_grasp", "close_ready"}),
                "state_feature_vector": basin_recovery_feature_vector(record),
                "uses_privileged_label": True,
                "uses_privileged_runtime": False,
                "uses_privileged_target": False,
                "uses_rlbench_mask_runtime": False,
                "label_source": str(record.get("label_source", "")) + "+basin_recovery_relabel",
            }
        )
        out.append(row)
        evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
        basin_counts[basin] = basin_counts.get(basin, 0) + 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    summary = {
        "num_input_records": int(len(records)),
        "num_output_records": int(len(out)),
        "evidence_counts": evidence_counts,
        "basin_counts": basin_counts,
        "failure_bucket_counts": bucket_counts,
        "uses_privileged_label": True,
        "uses_privileged_runtime": False,
        "uses_privileged_target": False,
        "purpose": "offline_training_labels_only__runtime_must_not_read_privileged_error",
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets_runtime_failure_30ep_hardmix/grasp_recovery_runtime_failure_dataset_v1.jsonl"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets_basin_recovery/basin_recovery_dataset_v1.jsonl"),
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets_basin_recovery/basin_recovery_dataset_v1_summary.json"),
    )
    ap.add_argument("--visual_conf_threshold", type=float, default=0.01)
    ap.add_argument("--visual_observability_threshold", type=float, default=0.002)
    ap.add_argument("--visual_axis_strength_threshold", type=float, default=1.0e-5)
    args = ap.parse_args()

    config = BasinRecoveryConfig(
        visual_conf_threshold=float(args.visual_conf_threshold),
        visual_observability_threshold=float(args.visual_observability_threshold),
        visual_axis_strength_threshold=float(args.visual_axis_strength_threshold),
    )
    rows, summary = build_rows(read_jsonl(args.input), config=config)
    _write_jsonl(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    print(args.summary)


if __name__ == "__main__":
    main()
