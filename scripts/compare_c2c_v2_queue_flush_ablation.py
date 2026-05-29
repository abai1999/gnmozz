#!/usr/bin/env python3
"""Compare queue-flush vs retain C2C grasp shell sweeps.

This comparison is intentionally separate from semantic frame/yaw audits.
It only compares the window protocol ablation artifacts already produced by
the sweep pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_contract_summary(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = obj.get("frame_contract_summary")
    return summary if isinstance(summary, Mapping) else {}


def _queue_flag(obj: Mapping[str, Any]) -> bool:
    sweep = obj.get("sweep_config") if isinstance(obj.get("sweep_config"), Mapping) else {}
    if "c2c_grasp_probe_flush_planner_queue" in sweep:
        return bool(sweep.get("c2c_grasp_probe_flush_planner_queue", False))
    return bool(obj.get("c2c_grasp_probe_flush_planner_queue", False))


def compare(flush_obj: Mapping[str, Any], retain_obj: Mapping[str, Any]) -> dict[str, Any]:
    flush = _frame_contract_summary(flush_obj)
    retain = _frame_contract_summary(retain_obj)
    def _get(summary: Mapping[str, Any], key: str, default: Any = 0) -> Any:
        return summary.get(key, default)
    return {
        "schema_version": "queue_flush_ablation_compare_v1",
        "flush": {
            "queue_flushed": _queue_flag(flush_obj),
            "close_ready_rows": _get(flush, "close_ready_rows", 0),
            "coarse_pullback_candidate_rows": _get(flush, "coarse_pullback_candidate_rows", 0),
            "micro_entry_ready_rows": _get(flush, "micro_entry_ready_rows", 0),
            "near_basin_shell_rows": _get(flush, "near_basin_shell_rows", 0),
            "yaw_blocked_rows": _get(flush, "yaw_blocked_rows", 0),
            "yaw_observable_rows": _get(flush, "yaw_observable_rows", 0),
            "contraction_rate_by_tier": dict(_get(flush, "contraction_rate_by_tier", {})),
            "contraction_lower_ci_by_tier": dict(_get(flush, "contraction_lower_ci_by_tier", {})),
            "shell_hit_bucket_counts": dict(flush_obj.get("shell_hit_bucket_counts", {})),
            "shell_hit_episode_counts": dict(flush_obj.get("shell_hit_episode_counts", {})),
        },
        "retain": {
            "queue_flushed": _queue_flag(retain_obj),
            "close_ready_rows": _get(retain, "close_ready_rows", 0),
            "coarse_pullback_candidate_rows": _get(retain, "coarse_pullback_candidate_rows", 0),
            "micro_entry_ready_rows": _get(retain, "micro_entry_ready_rows", 0),
            "near_basin_shell_rows": _get(retain, "near_basin_shell_rows", 0),
            "yaw_blocked_rows": _get(retain, "yaw_blocked_rows", 0),
            "yaw_observable_rows": _get(retain, "yaw_observable_rows", 0),
            "contraction_rate_by_tier": dict(_get(retain, "contraction_rate_by_tier", {})),
            "contraction_lower_ci_by_tier": dict(_get(retain, "contraction_lower_ci_by_tier", {})),
            "shell_hit_bucket_counts": dict(retain_obj.get("shell_hit_bucket_counts", {})),
            "shell_hit_episode_counts": dict(retain_obj.get("shell_hit_episode_counts", {})),
        },
        "delta": {
            "close_ready_rows": int(_get(flush, "close_ready_rows", 0) - _get(retain, "close_ready_rows", 0)),
            "coarse_pullback_candidate_rows": int(_get(flush, "coarse_pullback_candidate_rows", 0) - _get(retain, "coarse_pullback_candidate_rows", 0)),
            "micro_entry_ready_rows": int(_get(flush, "micro_entry_ready_rows", 0) - _get(retain, "micro_entry_ready_rows", 0)),
            "near_basin_shell_rows": int(_get(flush, "near_basin_shell_rows", 0) - _get(retain, "near_basin_shell_rows", 0)),
            "yaw_blocked_rows": int(_get(flush, "yaw_blocked_rows", 0) - _get(retain, "yaw_blocked_rows", 0)),
            "yaw_observable_rows": int(_get(flush, "yaw_observable_rows", 0) - _get(retain, "yaw_observable_rows", 0)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare queue flush vs retain sweep summaries.")
    ap.add_argument("--flush_summary", type=Path, required=True)
    ap.add_argument("--retain_summary", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, default=None)
    ap.add_argument("--output_md", type=Path, default=None)
    args = ap.parse_args()

    flush_obj = _read_json(args.flush_summary)
    retain_obj = _read_json(args.retain_summary)
    report = compare(flush_obj, retain_obj)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        md = [
            "# Queue Flush Ablation Comparison",
            "",
            f"- flush_summary: `{args.flush_summary}`",
            f"- retain_summary: `{args.retain_summary}`",
            "",
            "## Flush",
            f"- close_ready_rows: `{report['flush']['close_ready_rows']}`",
            f"- coarse_pullback_candidate_rows: `{report['flush']['coarse_pullback_candidate_rows']}`",
            f"- micro_entry_ready_rows: `{report['flush']['micro_entry_ready_rows']}`",
            f"- near_basin_shell_rows: `{report['flush']['near_basin_shell_rows']}`",
            f"- yaw_blocked_rows: `{report['flush']['yaw_blocked_rows']}`",
            "",
            "## Retain",
            f"- close_ready_rows: `{report['retain']['close_ready_rows']}`",
            f"- coarse_pullback_candidate_rows: `{report['retain']['coarse_pullback_candidate_rows']}`",
            f"- micro_entry_ready_rows: `{report['retain']['micro_entry_ready_rows']}`",
            f"- near_basin_shell_rows: `{report['retain']['near_basin_shell_rows']}`",
            f"- yaw_blocked_rows: `{report['retain']['yaw_blocked_rows']}`",
            "",
            "## Delta (flush - retain)",
            f"- close_ready_rows: `{report['delta']['close_ready_rows']}`",
            f"- coarse_pullback_candidate_rows: `{report['delta']['coarse_pullback_candidate_rows']}`",
            f"- micro_entry_ready_rows: `{report['delta']['micro_entry_ready_rows']}`",
            f"- near_basin_shell_rows: `{report['delta']['near_basin_shell_rows']}`",
            f"- yaw_blocked_rows: `{report['delta']['yaw_blocked_rows']}`",
            f"- yaw_observable_rows: `{report['delta']['yaw_observable_rows']}`",
        ]
        args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
