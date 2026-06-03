#!/usr/bin/env python3
"""Build a hard-bucket gap report for grasp failure-tail intervention rows.

The report joins a candidate JSONL with the corresponding intervention trace
rows and answers a narrow question:

    For each hard bucket, what is blocking active probe rows?

The output keeps the summary intentionally simple so it can be used as a
stable diagnostic artifact in the failure-tail support sweep.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


HARD_FAILURE_BUCKETS = {"large_xy_large_yaw", "large_xy_small_yaw", "small_xy_large_yaw", "small_xy_small_yaw"}
BLOCKED_REASON_ORDER = (
    "missing_trace",
    "not_failure_tail_candidate",
    "candidate_actionable",
    "shell_xy_outside_horizon",
    "shell_yaw_blocked",
    "shell_outside_frontier_window",
    "shell_outside_coarse_window",
    "active",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (_safe_int(row.get("episode_idx", -1)), _safe_int(row.get("step_idx", row.get("step", -1))))


def _group_rows(rows: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(dict(row))
    return grouped


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def _window_protocol_label(value: Any) -> str:
    if isinstance(value, Mapping):
        queue = "flush" if bool(value.get("queue_flushed", False)) else "retain"
        mode = str(value.get("window_mode", value.get("stage_source", "")))
        shell = str(value.get("shell_filter", ""))
        horizon = value.get("requested_horizon", value.get("horizon", ""))
        parts = [queue]
        if mode:
            parts.append(mode)
        if shell:
            parts.append(shell)
        if horizon != "":
            parts.append(f"h{horizon}")
        return "|".join(parts)
    return str(value)


def _text_or_default(value: Any, default: str = "unknown") -> str:
    if value is None:
        return str(default)
    text = str(value).strip()
    if not text or text == "None":
        return str(default)
    return text


def _candidate_match(row: Mapping[str, Any]) -> bool:
    if "grasp_probe_candidate_match" in row:
        return bool(row.get("grasp_probe_candidate_match", False))
    if "candidate_match" in row:
        return bool(row.get("candidate_match", False))
    return True


def _coverage_bucket(row: Mapping[str, Any]) -> str:
    if _candidate_match(row):
        return "candidate_match_true"
    reason = str(row.get("intervention_reason", row.get("grasp_probe_reason", "")) or "").strip()
    if reason == "not_failure_tail_candidate":
        return "not_failure_tail_candidate"
    return "candidate_match_false"


def _row_blocked_reason(row: Mapping[str, Any]) -> str:
    if bool(row.get("intervention_active", False)):
        return "active"
    if bool(row.get("xy_correction_ready", row.get("grasp_probe_xy_correction_ready", False))):
        return "xy_ready"
    reason = str(row.get("intervention_reason", row.get("grasp_probe_reason", "")))
    if reason:
        return reason
    if bool(row.get("intervention_trace_found", False)) and not bool(row.get("grasp_probe_candidate_actionable", False)):
        return "candidate_actionable"
    return "missing_trace"


def _join_candidates(
    candidate_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace_by_key = {_key(row): row for row in trace_rows}
    joined: list[dict[str, Any]] = []
    for cand in candidate_rows:
        trace = trace_by_key.get(_key(cand))
        if trace is None:
            joined.append(
                {
                    **cand,
                    "intervention_trace_found": False,
                    "intervention_active": False,
                    "intervention_reason": "missing_trace",
                    "window_protocol": _window_protocol_label(cand.get("window_protocol", cand.get("grasp_probe_window_protocol", cand.get("probe_window_protocol", "")))),
                    "alias_drift_decision": _text_or_default(cand.get("alias_drift_decision", None)),
                    "grasp_probe_candidate_actionable": False,
                    "grasp_probe_candidate_actionable_relaxed_small_xy_large_yaw": False,
                }
            )
            continue

        joined.append(
            {
                **cand,
                **{k: v for k, v in trace.items() if k not in cand or k.startswith("grasp_probe_")},
                "intervention_trace_found": True,
                "intervention_active": bool(trace.get("intervention_active", trace.get("grasp_probe_active", False))),
                "intervention_reason": str(
                    trace.get("intervention_reason", trace.get("grasp_probe_reason", "missing_trace")) or "missing_trace"
                ),
                "window_protocol": _window_protocol_label(
                    trace.get("window_protocol", trace.get("grasp_probe_window_protocol", cand.get("window_protocol", cand.get("probe_window_protocol", ""))))
                ),
                "alias_drift_decision": _text_or_default(
                    cand.get("alias_drift_decision", None),
                    _text_or_default(trace.get("alias_drift_decision", None), _text_or_default(trace.get("yaw_alias_drift_decision", None))),
                ),
                "grasp_probe_candidate_actionable": bool(
                    trace.get("grasp_probe_candidate_actionable", trace.get("candidate_actionable", False))
                ),
                "grasp_probe_candidate_actionable_relaxed_small_xy_large_yaw": bool(
                    trace.get("grasp_probe_candidate_actionable_relaxed_small_xy_large_yaw", False)
                ),
            }
        )
    joined.sort(key=lambda row: (_safe_int(row.get("episode_idx", -1)), _safe_int(row.get("step_idx", row.get("step", -1)))))
    return joined


def _classify_gap(row: Mapping[str, Any]) -> str:
    if bool(row.get("intervention_active", False)):
        return "active"
    reason = str(row.get("intervention_reason", ""))
    if reason == "missing_trace":
        return "missing_trace"
    if reason == "not_failure_tail_candidate":
        return "not_failure_tail_candidate"
    if reason in {"failure_tail_candidate_abstain", "candidate_actionable"}:
        return "candidate_actionable"
    if reason in {"shell_xy_outside_horizon", "shell_outside_frontier_window", "shell_outside_coarse_window"}:
        return "shell_xy_outside_horizon"
    if reason == "shell_yaw_blocked":
        return "shell_yaw_blocked"
    return "candidate_actionable"


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if bool(row.get("intervention_active", False))]
    found = [row for row in rows if bool(row.get("intervention_trace_found", False))]
    missing = [row for row in rows if not bool(row.get("intervention_trace_found", False))]
    candidate_match_rows = [row for row in rows if _candidate_match(row)]
    candidate_match_false_rows = [row for row in rows if not _candidate_match(row)]
    not_failure_tail_candidate = [
        row for row in rows
        if not _candidate_match(row) and _coverage_bucket(row) == "not_failure_tail_candidate"
    ]
    candidate_actionable_blocked = [
        row
        for row in rows
        if bool(row.get("intervention_trace_found", False))
        and not bool(row.get("intervention_active", False))
        and _classify_gap(row) == "candidate_actionable"
    ]
    shell_xy_outside_horizon = [row for row in rows if _classify_gap(row) == "shell_xy_outside_horizon"]
    shell_yaw_blocked = [row for row in rows if _classify_gap(row) == "shell_yaw_blocked"]
    xy_ready_rows = [row for row in rows if _row_blocked_reason(row) == "xy_ready"]
    return {
        "candidate_rows": int(len(rows)),
        "trace_found_rows": int(len(found)),
        "active_rows": int(len(active)),
        "candidate_match_rows": int(len(candidate_match_rows)),
        "candidate_match_false_rows": int(len(candidate_match_false_rows)),
        "xy_correction_ready_rows": int(len(xy_ready_rows)),
        "missing_trace_rows": int(len(missing)),
        "not_failure_tail_candidate_rows": int(len(not_failure_tail_candidate)),
        "candidate_actionable_blocked_rows": int(len(candidate_actionable_blocked)),
        "shell_xy_outside_horizon_rows": int(len(shell_xy_outside_horizon)),
        "shell_yaw_blocked_rows": int(len(shell_yaw_blocked)),
        "intervention_reason_counts": dict(Counter(str(row.get("intervention_reason", "")) for row in rows)),
        "blocked_reason_counts": dict(Counter(_classify_gap(row) for row in rows)),
        "row_blocked_reason_counts": dict(Counter(_row_blocked_reason(row) for row in rows)),
        "coverage_bucket_counts": dict(Counter(_coverage_bucket(row) for row in candidate_match_false_rows)),
        "failure_bucket_counts": _count_by(rows, "failure_bucket"),
        "takeover_tier_counts": _count_by(rows, "takeover_tier"),
        "yaw_observability_counts": _count_by(rows, "yaw_observability_class"),
        "visual_observability_counts": _count_by(rows, "visual_observability_class"),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    out: list[dict[str, Any]] = []
    for value, subset in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        out.append({key: value, **_summarize(subset)})
    return out


def _group_summary_candidate_match(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return _group_summary([row for row in rows if _candidate_match(row)], key)


def _group_summary_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _candidate_match(row):
            continue
        grouped[_coverage_bucket(row)].append(row)
    out: list[dict[str, Any]] = []
    for value, subset in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        out.append({"candidate_coverage_bucket": value, **_summarize(subset)})
    return out


def build_gap_report(candidate_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    joined = _join_candidates(candidate_rows, trace_rows)
    return {
        "schema_version": "grasp_failure_tail_hard_bucket_gap_report_v1",
        "overall": _summarize(joined),
        "by_failure_bucket": _group_summary(joined, "failure_bucket"),
        "by_episode": _group_summary(joined, "episode_idx"),
        "by_takeover_tier": _group_summary(joined, "takeover_tier"),
        "by_yaw_observability": _group_summary(joined, "yaw_observability_class"),
        "by_window_protocol": _group_summary(joined, "window_protocol"),
        "by_alias_drift_decision": _group_summary_candidate_match(joined, "alias_drift_decision"),
        "by_candidate_coverage_bucket": _group_summary_coverage(joined),
        "by_intervention_reason": _group_summary(joined, "intervention_reason"),
        "joined_rows": joined,
        "runtime_invariants": {
            "uses_privileged_runtime": False,
            "uses_privileged_target": False,
            "uses_privileged_label_for_eval": True,
            "uses_rlbench_mask_runtime": False,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a hard-bucket gap report for grasp failure-tail intervention data.")
    ap.add_argument("--candidate_jsonl", type=Path, required=True)
    ap.add_argument("--trace_rows_jsonl", type=Path, required=True)
    ap.add_argument(
        "--output_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/hard_bucket_gap_report.json"),
    )
    ap.add_argument(
        "--output_md",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/hard_bucket_gap_report.md"),
    )
    args = ap.parse_args()

    candidate_rows = _read_jsonl(args.candidate_jsonl)
    trace_rows = _read_jsonl(args.trace_rows_jsonl)
    report = build_gap_report(candidate_rows, trace_rows)
    report["source_candidate_jsonl"] = str(args.candidate_jsonl.resolve())
    report["source_trace_rows_jsonl"] = str(args.trace_rows_jsonl.resolve())

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({k: v for k, v in report.items() if k != "joined_rows"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md_lines = [
        "# Hard Bucket Gap Report",
        "",
        f"- candidate_jsonl: `{args.candidate_jsonl.resolve()}`",
        f"- trace_rows_jsonl: `{args.trace_rows_jsonl.resolve()}`",
        "",
        "## Overall",
    ]
    for key in [
        "candidate_rows",
        "trace_found_rows",
        "active_rows",
        "candidate_match_rows",
        "candidate_match_false_rows",
        "missing_trace_rows",
        "not_failure_tail_candidate_rows",
        "candidate_actionable_blocked_rows",
        "shell_xy_outside_horizon_rows",
        "shell_yaw_blocked_rows",
    ]:
        md_lines.append(f"- {key}: `{report['overall'][key]}`")
    md_lines.append("")
    md_lines.append("## By Failure Bucket")
    for item in report["by_failure_bucket"]:
        md_lines.append(
            f"- `{item['failure_bucket']}`: candidate={item['candidate_rows']}, trace={item['trace_found_rows']}, active={item['active_rows']}, "
            f"missing={item['missing_trace_rows']}, actionable={item['candidate_actionable_blocked_rows']}, "
            f"xy_blocked={item['shell_xy_outside_horizon_rows']}, yaw_blocked={item['shell_yaw_blocked_rows']}"
        )
    md_lines.append("")
    md_lines.append("## By Takeover Tier")
    for item in report["by_takeover_tier"]:
        md_lines.append(
            f"- `{item['takeover_tier']}`: candidate={item['candidate_rows']}, trace={item['trace_found_rows']}, active={item['active_rows']}, "
            f"missing={item['missing_trace_rows']}, actionable={item['candidate_actionable_blocked_rows']}, "
            f"xy_blocked={item['shell_xy_outside_horizon_rows']}, yaw_blocked={item['shell_yaw_blocked_rows']}"
        )
    md_lines.append("")
    md_lines.append("## By Yaw Observability")
    for item in report["by_yaw_observability"]:
        md_lines.append(
            f"- `{item['yaw_observability_class']}`: candidate={item['candidate_rows']}, trace={item['trace_found_rows']}, active={item['active_rows']}, "
            f"missing={item['missing_trace_rows']}, actionable={item['candidate_actionable_blocked_rows']}, "
            f"xy_blocked={item['shell_xy_outside_horizon_rows']}, yaw_blocked={item['shell_yaw_blocked_rows']}"
        )
    md_lines.append("")
    md_lines.append("## By Window Protocol")
    for item in report["by_window_protocol"]:
        md_lines.append(
            f"- `{item['window_protocol']}`: candidate={item['candidate_rows']}, trace={item['trace_found_rows']}, active={item['active_rows']}, "
            f"missing={item['missing_trace_rows']}, actionable={item['candidate_actionable_blocked_rows']}, "
            f"xy_blocked={item['shell_xy_outside_horizon_rows']}, yaw_blocked={item['shell_yaw_blocked_rows']}"
        )
    md_lines.append("")
    md_lines.append("## By Alias/Drift Decision")
    for item in report["by_alias_drift_decision"]:
        md_lines.append(
            f"- `{item['alias_drift_decision']}`: candidate={item['candidate_rows']}, trace={item['trace_found_rows']}, active={item['active_rows']}, "
            f"missing={item['missing_trace_rows']}, actionable={item['candidate_actionable_blocked_rows']}, "
            f"xy_blocked={item['shell_xy_outside_horizon_rows']}, yaw_blocked={item['shell_yaw_blocked_rows']}"
        )
    md_lines.append("")
    md_lines.append("## Candidate Coverage Bucket")
    if report.get("by_candidate_coverage_bucket"):
        for item in report["by_candidate_coverage_bucket"]:
            md_lines.append(
                f"- `{item['candidate_coverage_bucket']}`: candidate={item['candidate_rows']}, trace={item['trace_found_rows']}, active={item['active_rows']}, "
                f"missing={item['missing_trace_rows']}, actionable={item['candidate_actionable_blocked_rows']}, "
                f"xy_blocked={item['shell_xy_outside_horizon_rows']}, yaw_blocked={item['shell_yaw_blocked_rows']}"
            )
    else:
        md_lines.append("- none")
    md_lines.append("")
    md_lines.append("## By Intervention Reason")
    for item in report["by_intervention_reason"]:
        md_lines.append(
            f"- `{item['intervention_reason']}`: candidate={item['candidate_rows']}, trace={item['trace_found_rows']}, active={item['active_rows']}, "
            f"missing={item['missing_trace_rows']}, actionable={item['candidate_actionable_blocked_rows']}, "
            f"xy_blocked={item['shell_xy_outside_horizon_rows']}, yaw_blocked={item['shell_yaw_blocked_rows']}"
        )
    md_lines.append("")
    md_lines.append("## By Episode")
    for item in report["by_episode"]:
        ep = _safe_int(item["episode_idx"], -1)
        md_lines.append(
            f"- `ep{ep:03d}`: candidate={item['candidate_rows']}, trace={item['trace_found_rows']}, active={item['active_rows']}, "
            f"missing={item['missing_trace_rows']}, actionable={item['candidate_actionable_blocked_rows']}, "
            f"xy_blocked={item['shell_xy_outside_horizon_rows']}, yaw_blocked={item['shell_yaw_blocked_rows']}"
        )
    md_lines.append("")
    md_lines.append("## Blocked Reason Counts")
    md_lines.append(f"`{report['overall']['blocked_reason_counts']}`")
    md_lines.append("")
    md_lines.append("## Intervention Reason Counts")
    md_lines.append(f"`{report['overall']['intervention_reason_counts']}`")
    md_lines.append("")
    md_lines.append("## Failure Bucket Counts")
    md_lines.append(f"`{report['overall']['failure_bucket_counts']}`")
    md_lines.append("")
    md_lines.append("## Takeover Tier Counts")
    md_lines.append(f"`{report['overall']['takeover_tier_counts']}`")
    md_lines.append("")
    md_lines.append("## Yaw Observability Counts")
    md_lines.append(f"`{report['overall']['yaw_observability_counts']}`")
    md_lines.append("")
    md_lines.append("## Visual Observability Counts")
    md_lines.append(f"`{report['overall']['visual_observability_counts']}`")
    args.output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
