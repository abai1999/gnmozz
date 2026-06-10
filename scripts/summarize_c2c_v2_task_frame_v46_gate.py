#!/usr/bin/env python3
"""Summarize v46 task-frame alignment promotion gates from eval reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MIN_AXIS_CONTRACTION = 0.50
MAX_COMBINED_WORSEN = 0.25
MIN_YAW_EVIDENCE_RATE = 0.05
MIN_RANKER_LOO_FOLDS = 10
MIN_RANKER_WORST_ROOT_ZERO_MARGIN = 1.0e-6


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(overall: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = float(overall.get(name, default))
        return value if value == value else float(default)
    except Exception:
        return float(default)


def _eval_gate(name: str, report: dict[str, Any]) -> dict[str, Any]:
    overall = report.get("overall", {})
    rows = int(report.get("eval_rows", overall.get("rows", 0)) or 0)
    yaw_target = _metric(overall, "yaw_control_target_rate")
    yaw_pred = _metric(overall, "yaw_control_predicted_rate")
    yaw_runtime_pred = _metric(overall, "yaw_selector_control_predicted_rate", yaw_pred)
    yaw_selector_precision = _metric(overall, "yaw_selector_precision", -1.0)
    yaw_selector_fpr = _metric(overall, "yaw_selector_false_positive_rate", -1.0)
    yaw_amb_target = _metric(overall, "yaw_ambiguous_target_rate")
    yaw_amb_pred = _metric(overall, "yaw_ambiguous_predicted_rate")
    xy_obs_target = _metric(overall, "xy_observable_target_rate")
    xy_obs_pred = _metric(overall, "xy_observable_predicted_rate")
    violations: list[str] = []
    if rows <= 0:
        violations.append("no_eval_rows")
    if xy_obs_target >= 0.5 and xy_obs_pred < 0.5:
        violations.append("xy_observable_false_negative")
    if _metric(overall, "bounded_step_contraction") < MIN_AXIS_CONTRACTION:
        violations.append("combined_contraction_below_gate")
    if _metric(overall, "xy_bounded_step_contraction") < MIN_AXIS_CONTRACTION and xy_obs_target >= 0.5:
        violations.append("xy_contraction_below_gate")
    if _metric(overall, "z_bounded_step_contraction") < MIN_AXIS_CONTRACTION:
        violations.append("z_contraction_below_gate")
    if _metric(overall, "bounded_step_worsen") > MAX_COMBINED_WORSEN:
        violations.append("combined_worsen_above_gate")
    if yaw_target <= 0.05 and yaw_runtime_pred > 0.10:
        violations.append("yaw_control_false_positive")
    if "yaw_selector_control_predicted_rate" in overall and yaw_target >= MIN_YAW_EVIDENCE_RATE:
        if yaw_selector_precision < 0.80:
            violations.append("yaw_selector_precision_below_gate")
        if yaw_selector_fpr > 0.01:
            violations.append("yaw_selector_false_positive_rate_above_gate")
    if yaw_amb_target >= 0.50 and yaw_amb_pred < 0.50:
        violations.append("yaw_ambiguity_false_negative")
    if yaw_target < MIN_YAW_EVIDENCE_RATE:
        violations.append("insufficient_yaw_control_evidence")
    if yaw_target >= MIN_YAW_EVIDENCE_RATE and _metric(overall, "yaw_bounded_step_contraction") < MIN_AXIS_CONTRACTION:
        violations.append("yaw_observable_contraction_below_gate")
    return {
        "name": name,
        "eval_rows": rows,
        "status": "fail" if violations else "offline_pass",
        "violations": violations,
        "metrics": {
            "bounded_step_contraction": _metric(overall, "bounded_step_contraction"),
            "bounded_step_worsen": _metric(overall, "bounded_step_worsen"),
            "xy_bounded_step_contraction": _metric(overall, "xy_bounded_step_contraction"),
            "z_bounded_step_contraction": _metric(overall, "z_bounded_step_contraction"),
            "yaw_bounded_step_contraction": _metric(overall, "yaw_bounded_step_contraction"),
            "xy_observable_target_rate": xy_obs_target,
            "xy_observable_predicted_rate": xy_obs_pred,
            "yaw_control_target_rate": yaw_target,
            "yaw_control_predicted_rate": yaw_pred,
            "yaw_runtime_control_predicted_rate": yaw_runtime_pred,
            "yaw_selector_precision": yaw_selector_precision,
            "yaw_selector_false_positive_rate": yaw_selector_fpr,
            "yaw_ambiguous_target_rate": yaw_amb_target,
            "yaw_ambiguous_predicted_rate": yaw_amb_pred,
            "risk_accuracy": _metric(overall, "risk_accuracy"),
            "near_field_recall": _metric(overall, "near_field_recall"),
        },
    }


def _nested_metric(payload: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return float(default)
        current = current[part]
    try:
        value = float(current)
        return value if value == value else float(default)
    except Exception:
        return float(default)


def _ranker_loo_gate(name: str, report: dict[str, Any]) -> dict[str, Any]:
    folds = int(report.get("folds", 0) or 0)
    worst = dict(report.get("worst_folds", {}) or {})
    fold_summaries = list(report.get("fold_summaries", []) or [])
    worst_combined_zero_margin = _nested_metric(
        worst,
        ("top1_minus_zero_combined_contraction", "top1_minus_zero_combined_contraction"),
        default=0.0,
    )
    worst_yaw_zero_margin = _nested_metric(
        worst,
        ("top1_minus_zero_yaw_contraction", "top1_minus_zero_yaw_contraction"),
        default=0.0,
    )
    worst_match = _nested_metric(worst, ("top1_best_score_match", "top1_best_score_match"), default=0.0)
    worst_combined = _nested_metric(worst, ("top1_combined_contraction", "top1_combined_contraction"), default=0.0)
    worse_than_zero_folds = [
        fold
        for fold in fold_summaries
        if _metric(fold, "top1_worse_than_zero_xy_rate") > 0.0
        or _metric(fold, "top1_worse_than_zero_z_rate") > 0.0
        or _metric(fold, "top1_worse_than_zero_yaw_rate") > 0.0
        or _metric(fold, "top1_worse_than_zero_combined_rate") > 0.0
    ]
    violations: list[str] = []
    if folds < MIN_RANKER_LOO_FOLDS:
        violations.append("insufficient_ranker_loo_folds")
    if worst_match < 0.80:
        violations.append("ranker_top1_match_below_gate")
    if worst_combined < MIN_AXIS_CONTRACTION:
        violations.append("ranker_combined_contraction_below_gate")
    if worst_combined_zero_margin <= MIN_RANKER_WORST_ROOT_ZERO_MARGIN:
        violations.append("ranker_worst_root_combined_not_beating_zero")
    if worst_yaw_zero_margin <= MIN_RANKER_WORST_ROOT_ZERO_MARGIN:
        violations.append("ranker_worst_root_yaw_not_beating_zero")
    if worse_than_zero_folds:
        violations.append("ranker_top1_worse_than_zero")
    return {
        "name": name,
        "folds": folds,
        "status": "fail" if violations else "offline_pass",
        "violations": violations,
        "metrics": {
            "worst_top1_best_score_match": worst_match,
            "worst_top1_combined_contraction": worst_combined,
            "worst_top1_minus_zero_combined_contraction": worst_combined_zero_margin,
            "worst_top1_minus_zero_yaw_contraction": worst_yaw_zero_margin,
            "worse_than_zero_folds": int(len(worse_than_zero_folds)),
            "min_required_folds": int(MIN_RANKER_LOO_FOLDS),
        },
    }


def _parse_named_path(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip() or Path(path).stem, Path(path)
    path = Path(text)
    return path.stem, path


def summarize(eval_json: list[str], *, output_json: Path, ranker_loo_json: list[str] | None = None) -> dict[str, Any]:
    reports = []
    for item in eval_json:
        name, path = _parse_named_path(item)
        reports.append(_eval_gate(name, _read_json(path)))
    ranker_loo_reports = []
    for item in ranker_loo_json or []:
        name, path = _parse_named_path(item)
        ranker_loo_reports.append(_ranker_loo_gate(name, _read_json(path)))
    failing = [report for report in reports if report["status"] == "fail"] + [
        report for report in ranker_loo_reports if report["status"] == "fail"
    ]
    summary = {
        "schema_version": "c2c_v2_task_frame_v46_gate_summary_v1",
        "reports": reports,
        "ranker_loo_reports": ranker_loo_reports,
        "offline_gate_status": "fail" if failing else "pass",
        "promotion_status": "fail_offline_gate" if failing else "pending_runtime_insert_success",
        "required_runtime_evidence": [
            "canonical RLBench smoke path with front+wrist MP4",
            "random held-out closed-loop residual contraction",
            "insert success improvement without close authority regression",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v46 task-frame offline promotion gates.")
    parser.add_argument("--eval_json", nargs="+", required=True, help="Eval report paths, optionally name=path.")
    parser.add_argument("--ranker_loo_json", nargs="*", default=[], help="LOO ranker reports, optionally name=path.")
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(list(args.eval_json), output_json=args.output_json, ranker_loo_json=list(args.ranker_loo_json))
    print(json.dumps({k: summary[k] for k in ("offline_gate_status", "promotion_status")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
