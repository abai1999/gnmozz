#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset_report", type=Path, required=True)
    ap.add_argument("--stagea_gate", type=Path, required=True)
    ap.add_argument("--stageb_gate", type=Path, required=True)
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    dataset = load_json(args.dataset_report)
    stagea = load_json(args.stagea_gate)
    stageb = load_json(args.stageb_gate)
    candidate = stageb.get("candidate_val_metrics", {})
    baseline = stageb.get("baseline_val_metrics", {})

    boundary_report = {
        "dataset_rows": dataset.get("rows", 0),
        "real_rows": dataset.get("real_rows", 0),
        "counterfactual_rows": dataset.get("counterfactual_rows", 0),
        "teacher_ready_rows_real": dataset.get("teacher_ready_rows_real", 0),
        "source_summary": dataset.get("source_summary", {}),
        "progress_rows": dataset.get("progress", {}),
        "baseline_progress_balanced_acc": baseline.get("progress_balanced_acc"),
        "candidate_progress_balanced_acc": candidate.get("progress_balanced_acc"),
        "candidate_pair_balanced_acc": candidate.get("pair_balanced_acc"),
        "candidate_pair_calibrated_balanced_acc": candidate.get("pair_calibrated_balanced_acc"),
        "candidate_pair_calibrated_pos_recall": candidate.get("pair_calibrated_pos_recall"),
        "candidate_pair_calibrated_neg_recall": candidate.get("pair_calibrated_neg_recall"),
        "candidate_pair_calibrated_threshold": candidate.get("pair_calibrated_threshold"),
        "baseline_boundary_mae": {
            "xy": baseline.get("boundary_mae_xy_norm"),
            "z": baseline.get("boundary_mae_z_norm"),
            "yaw": baseline.get("boundary_mae_yaw_norm"),
        },
        "candidate_boundary_mae": {
            "xy": candidate.get("boundary_mae_xy_norm"),
            "z": candidate.get("boundary_mae_z_norm"),
            "yaw": candidate.get("boundary_mae_yaw_norm"),
        },
        "axis_block_acc": candidate.get("axis_block_acc"),
        "decision": stageb.get("decision"),
    }
    false_ready_report = {
        "baseline_far_negative_ready_prob_mean": baseline.get("far_negative_ready_prob_mean"),
        "candidate_far_negative_ready_prob_mean": candidate.get("far_negative_ready_prob_mean"),
        "baseline_ready_prob_mean": baseline.get("ready_prob_mean"),
        "candidate_ready_prob_mean": candidate.get("ready_prob_mean"),
        "baseline_pred_release_band_rate": baseline.get("pred_release_band_rate"),
        "candidate_pred_release_band_rate": candidate.get("pred_release_band_rate"),
        "safe_for_shadow": bool(
            (candidate.get("far_negative_ready_prob_mean", 1.0) <= 0.01)
            and (candidate.get("pred_release_band_rate", 1.0) <= 0.001)
        ),
    }
    stageb_shadow_candidate = stageb.get("decision") == "shadow_candidate"
    shadow_compare = {
        "status": "not_run",
        "reason": (
            "offline gate passed; runtime shadow is the next validation step"
            if stageb_shadow_candidate
            else "offline gate did not pass; runtime shadow should wait for progress/boundary gate improvement"
        ),
        "stagea_decision": stagea.get("decision"),
        "stageb_decision": stageb.get("decision"),
        "recommended_next_step": (
            "run shadow-only validation before any applied control"
            if stageb_shadow_candidate
            else "improve progress supervision / boundary data balance before runtime shadow"
        ),
    }

    (args.root / "boundary_progress_trace_report.json").write_text(json.dumps(boundary_report, indent=2))
    (args.root / "false_ready_safety_report.json").write_text(json.dumps(false_ready_report, indent=2))
    (args.root / "alignment_v2_shadow_compare.json").write_text(json.dumps(shadow_compare, indent=2))
    print(json.dumps({"boundary_report": boundary_report, "false_ready_report": false_ready_report, "shadow_compare": shadow_compare}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
