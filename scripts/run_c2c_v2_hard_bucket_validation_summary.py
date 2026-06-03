#!/usr/bin/env python3
"""Run summary analysis for hard-bucket validation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_c2c_v2_grasp_failure_tail_intervention import audit as audit_failure_tail
from scripts.audit_c2c_v2_small_xy_micro_stability import audit as audit_small_micro
from scripts.build_c2c_v2_grasp_timing_ablation_manifest import build_timing_ablation_manifests


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize hard-bucket validation artifacts.")
    ap.add_argument("--root", type=Path, default=Path("/home/guoning/code/VLA2/runtime_artifacts/coarse2contact_v2"))
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/guoning/code/VLA2/runtime_artifacts/coarse2contact_v2/reports/hard_bucket_validation_summary_v1"),
    )
    args = ap.parse_args()

    root = args.root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_candidates = _read_jsonl(root / "datasets" / "grasp_failure_tail_candidates_large_xy_large_yaw_focus_alias_epfallback.jsonl")
    v16_rows_paths = sorted((root / "grasp_shell_episode_sweep_hard_bucket_30k_validation_v16_alias_clean" / "large_xy_large_yaw_focus_alias_epfallback_retain_30k").glob("chunk_*/audit/grasp_probe_intervention_rows.jsonl"))
    v17_rows_paths = sorted((root / "grasp_shell_episode_sweep_hard_bucket_30k_validation_v17_large_support" / "large_xy_large_yaw_focus_alias_epfallback_retain_frontier022_30k").glob("chunk_*/audit/grasp_probe_intervention_rows.jsonl"))
    small_rows_paths = sorted((root / "grasp_shell_episode_sweep_hard_bucket_30k_validation_v16_alias_clean" / "small_xy_large_yaw_focus_alias_epfallback_flush_xg050_ms005_h5_30k").glob("chunk_*/audit/grasp_probe_intervention_rows.jsonl"))

    v16_rows = _collect(v16_rows_paths)
    v17_rows = _collect(v17_rows_paths)
    small_rows = _collect(small_rows_paths)

    manifests = build_timing_ablation_manifests(
        base_candidates,
        v16_rows,
        v17_rows,
        failure_bucket="large_xy_large_yaw",
        episodes={20, 21},
    )
    manifest_dir = output_dir / "large_timing_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for mode, rows in manifests.items():
        with open(manifest_dir / f"grasp_failure_tail_candidates_large_xy_large_yaw_{mode}.jsonl", "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    large_reports = {
        "v16_active_set_only": audit_failure_tail(manifests["v16_active_set_only"], v16_rows),
        "v17_new_active_set_only": audit_failure_tail(manifests["v17_new_active_set_only"], v17_rows),
        "combined": audit_failure_tail(manifests["combined"], v16_rows + v17_rows),
    }
    small_report = audit_small_micro(
        small_rows,
        failure_bucket="small_xy_large_yaw",
        episodes={4, 16, 18, 27, 29},
        active_only=True,
    )

    large_summary = {
        mode: {
            "candidate_rows": report["overall"]["num_rows"],
            "trace_found_rows": report["overall"]["intervention_trace_found_rows"],
            "active_rows": report["overall"]["active_failure_tail_rows"],
            "scalar_xy_contraction_rate": report["overall"]["scalar_xy_contraction_rate"],
            "vector_norm_contraction_rate": report["overall"]["vector_norm_contraction_rate"],
            "scalar_vector_xy_agreement_rate": report["overall"]["scalar_vector_xy_agreement_rate"],
            "oracle_intervention_contraction_rate": report["overall"]["oracle_intervention_contraction_rate"],
            "planner_natural_contraction_rate": report["overall"]["planner_natural_contraction_rate"],
            "intervention_vs_planner_improvement_rate": report["overall"]["intervention_vs_planner_improvement_rate"],
            "near_grasp_entry_gain": report["overall"]["near_grasp_entry_gain"],
            "overshoot_rate": report["overall"]["overshoot_rate"],
            "alias_unknown_rate": next((item.get("unknown_rate", 0.0) for item in report["active_by_alias_drift_decision"] if item.get("alias_drift_decision") == "unknown"), 0.0),
            "blocked_reason_counts": report["overall"]["blocked_reason_counts"],
        }
        for mode, report in large_reports.items()
    }
    small_summary = {
        "rows": small_report["overall"]["rows"],
        "scalar_xy_contraction_rate": small_report["overall"]["scalar_xy_contraction_rate"],
        "vector_norm_contraction_rate": small_report["overall"]["vector_norm_contraction_rate"],
        "near_entry_rate": small_report["overall"]["near_entry_rate"],
        "micro_entry_ready_after_rate": small_report["overall"]["micro_entry_ready_after_rate"],
        "overshoot_rate": small_report["overall"]["overshoot_rate"],
        "mean_final_xy_norm": small_report["overall"]["mean_final_xy_norm"],
        "p50_final_xy_norm": small_report["overall"]["p50_final_xy_norm"],
        "p90_final_xy_norm": small_report["overall"]["p90_final_xy_norm"],
        "direction_hint_counts": small_report["overall"]["direction_hint_counts"],
        "by_residual_norm_bin": small_report["by_residual_norm_bin"],
        "by_alias_drift_decision": small_report["by_alias_drift_decision"],
    }

    _write_json(output_dir / "large_timing_ablation_summary.json", large_summary)
    _write_json(output_dir / "small_xy_micro_stability_audit.json", {k: v for k, v in small_report.items() if k != "rows"})
    _write_json(output_dir / "summary.json", {"large": large_summary, "small": small_summary})

    print(json.dumps({"large": large_summary, "small": small_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
