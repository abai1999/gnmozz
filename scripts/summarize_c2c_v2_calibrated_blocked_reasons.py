#!/usr/bin/env python3
"""Summarize remaining blocked rows after calibrated yaw observability audit.

This script keeps the semantic diagnosis separate from any queue/window
protocol study.  It only reads a calibrated frame-contract audit report and
breaks down the residual blocked set by blocker family and a small fixability
heuristic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    try:
        return int(value)
    except Exception:
        return 0


def summarize(report: Mapping[str, Any]) -> dict[str, Any]:
    overall = report.get("overall", {})
    primary = overall.get("yaw_primary_blocker_counts", {}) or {}
    blocker_combo = overall.get("yaw_blocker_combo_counts", {}) or {}
    visual_blocked = _counts(overall, "visual_observable_yaw_blocked_rows")
    visual_xy_blocked = _counts(overall, "visual_observable_xy_contracted_yaw_blocked_rows")
    blocked_total = _counts(overall, "yaw_control_blocked_rows")
    observable_total = _counts(overall, "yaw_control_observable_rows")
    near_blocked = _counts(overall, "near_basin_shell_yaw_control_blocked_rows")
    near_observable = _counts(overall, "near_basin_shell_yaw_control_observable_rows")

    fixable_frame = _counts(primary, "frame_observability_lt_010")
    fixable_frame_combo = _counts(blocker_combo, "frame_observability_lt_010+frame_confidence_lt_050")
    fixable_visual = fixable_frame + fixable_frame_combo
    hard_prior = _counts(primary, "prior_only")
    hard_wrist = _counts(primary, "wrist_occluded")

    return {
        "schema_version": "calibrated_blocked_reason_summary_v1",
        "yaw_control_observable_rows": observable_total,
        "yaw_control_blocked_rows": blocked_total,
        "near_basin_shell_yaw_control_observable_rows": near_observable,
        "near_basin_shell_yaw_control_blocked_rows": near_blocked,
        "visual_observable_yaw_blocked_rows": visual_blocked,
        "visual_observable_xy_contracted_yaw_blocked_rows": visual_xy_blocked,
        "blocked_primary_counts": dict(primary),
        "blocked_combo_counts": dict(blocker_combo),
        "likely_fixable_rows": {
            "frame_observability_limited_rows": int(fixable_visual),
            "frame_observability_limited_share_of_blocked": float(fixable_visual / blocked_total) if blocked_total else 0.0,
            "frame_observability_limited_share_of_visual_blocked": float(fixable_visual / visual_blocked) if visual_blocked else 0.0,
        },
        "likely_hard_rows": {
            "prior_only_rows": int(hard_prior),
            "wrist_occluded_rows": int(hard_wrist),
            "prior_only_share_of_blocked": float(hard_prior / blocked_total) if blocked_total else 0.0,
            "wrist_occluded_share_of_blocked": float(hard_wrist / blocked_total) if blocked_total else 0.0,
            "hard_total_share_of_blocked": float((hard_prior + hard_wrist) / blocked_total) if blocked_total else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize calibrated yaw blocked reasons from a frame-contract audit report.")
    ap.add_argument("--audit_json", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, default=None)
    ap.add_argument("--output_md", type=Path, default=None)
    args = ap.parse_args()

    report = _read_json(args.audit_json)
    summary = summarize(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        md = [
            "# Calibrated Blocked Reason Summary",
            "",
            f"- audit: `{args.audit_json}`",
            f"- yaw_control_observable_rows: `{summary['yaw_control_observable_rows']}`",
            f"- yaw_control_blocked_rows: `{summary['yaw_control_blocked_rows']}`",
            f"- near_basin_shell_yaw_control_observable_rows: `{summary['near_basin_shell_yaw_control_observable_rows']}`",
            f"- near_basin_shell_yaw_control_blocked_rows: `{summary['near_basin_shell_yaw_control_blocked_rows']}`",
            f"- visual_observable_yaw_blocked_rows: `{summary['visual_observable_yaw_blocked_rows']}`",
            f"- visual_observable_xy_contracted_yaw_blocked_rows: `{summary['visual_observable_xy_contracted_yaw_blocked_rows']}`",
            "",
            "## Likely Fixable",
            f"- frame_observability_limited_rows: `{summary['likely_fixable_rows']['frame_observability_limited_rows']}`",
            f"- frame_observability_limited_share_of_blocked: `{summary['likely_fixable_rows']['frame_observability_limited_share_of_blocked']:.3f}`",
            f"- frame_observability_limited_share_of_visual_blocked: `{summary['likely_fixable_rows']['frame_observability_limited_share_of_visual_blocked']:.3f}`",
            "",
            "## Likely Hard",
            f"- prior_only_rows: `{summary['likely_hard_rows']['prior_only_rows']}`",
            f"- wrist_occluded_rows: `{summary['likely_hard_rows']['wrist_occluded_rows']}`",
            f"- hard_total_share_of_blocked: `{summary['likely_hard_rows']['hard_total_share_of_blocked']:.3f}`",
        ]
        args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
