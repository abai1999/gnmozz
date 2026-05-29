#!/usr/bin/env python3
"""Merge balanced and hard failure-tail manifests into a support-growth slice.

This is the practical answer to "grow the support surface" without throwing
away the shell rows that already proved useful:

* keep the balanced manifest as the core
* add hard-coverage supplement rows that are not already covered
* deduplicate by episode / step / task / stage / skill

The resulting manifest is still offline-only and still uses the same candidate
schema.  It simply widens the surface so downstream audit and training can see
both the previously useful shell and the harder low-recovery tail.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


HARD_FAILURE_BUCKETS = {"large_xy_large_yaw", "large_xy_small_yaw", "small_xy_large_yaw"}
SUPPORT_TAKEOVER_TIERS = {"frontier_pullback_candidate", "outer_pullback_candidate", "coarse_pullback_candidate", "near_basin_shell", "micro_entry_ready"}


def _support_tier_rank(tier: str) -> int:
    return {
        "frontier_pullback_candidate": 4,
        "outer_pullback_candidate": 3,
        "coarse_pullback_candidate": 2,
        "near_basin_shell": 1,
        "micro_entry_ready": 1,
    }.get(tier, 0)


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


def _dedupe_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("task_name", "")),
        _safe_int(row.get("episode_idx", -1)),
        _safe_int(row.get("step_idx", row.get("step", -1))),
        str(row.get("stage_name", "")),
        str(row.get("skill_type", "")),
    )


def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        _safe_int(row.get("episode_idx", -1)),
        _safe_int(row.get("step_idx", row.get("step", -1))),
        str(row.get("failure_bucket", "")),
        str(row.get("takeover_tier", "")),
    )


def _support_priority(row: Mapping[str, Any], *, source_name: str) -> tuple[int, int, int, int, int, int]:
    failure_bucket = str(row.get("failure_bucket", ""))
    visual = str(row.get("visual_observability_class", ""))
    yaw = str(row.get("yaw_observability_class", ""))
    tier = str(row.get("takeover_tier", ""))
    natural_outcome = str(row.get("planner_natural_outcome", ""))
    return (
        1 if failure_bucket in HARD_FAILURE_BUCKETS else 0,
        _support_tier_rank(tier),
        1 if visual in {"prior_only", "partial_observable"} else 0,
        1 if yaw in {"ambiguous", "unobservable"} else 0,
        1 if natural_outcome not in {"already_success_window", "natural_enters_basin"} else 0,
        1 if source_name == "supplement" else 0,
    )


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def merge_manifests(
    core_rows: list[dict[str, Any]],
    supplement_rows: list[dict[str, Any]],
    *,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    priority_by_key: dict[tuple[Any, ...], tuple[int, int, int, int, int, int]] = {}
    source_by_key: dict[tuple[Any, ...], str] = {}
    source_counts = Counter()
    replacement_count = 0

    for source_name, rows in (("core", core_rows), ("supplement", supplement_rows)):
        for row in rows:
            key = _dedupe_key(row)
            enriched = dict(row)
            priority = _support_priority(enriched, source_name=source_name)
            if key not in selected_by_key or priority > priority_by_key[key]:
                if key in selected_by_key:
                    replacement_count += 1
                    source_counts[source_by_key[key]] -= 1
                selected_by_key[key] = enriched
                priority_by_key[key] = priority
                source_by_key[key] = source_name
                source_counts[source_name] += 1

    merged = []
    for key, row in selected_by_key.items():
        row = dict(row)
        row["manifest_source"] = source_by_key[key]
        row["support_priority"] = list(priority_by_key[key])
        merged.append(row)
    merged.sort(key=_sort_key)
    if max_rows is not None:
        merged = merged[: max(0, int(max_rows))]
    source_counts = Counter(str(row.get("manifest_source", "")) for row in merged)
    hard_support_rows = [
        row
        for row in merged
        if str(row.get("failure_bucket", "")) in HARD_FAILURE_BUCKETS
        and str(row.get("takeover_tier", "")) in SUPPORT_TAKEOVER_TIERS
    ]
    summary = {
        "schema_version": "grasp_failure_tail_support_manifest_v1",
        "core_rows": int(len(core_rows)),
        "supplement_rows": int(len(supplement_rows)),
        "selected_rows": int(len(merged)),
        "selected_core_rows": int(source_counts.get("core", 0)),
        "selected_supplement_rows": int(source_counts.get("supplement", 0)),
        "replaced_rows": int(replacement_count),
        "by_manifest_source": dict(source_counts),
        "by_episode": {f"ep{int(ep):03d}": int(count) for ep, count in sorted(Counter(_safe_int(row.get("episode_idx", -1)) for row in merged).items())},
        "by_failure_bucket": _count_by(merged, "failure_bucket"),
        "by_visual_observability": _count_by(merged, "visual_observability_class"),
        "by_yaw_observability": _count_by(merged, "yaw_observability_class"),
        "by_takeover_tier": _count_by(merged, "takeover_tier"),
        "hard_support_rows": int(len(hard_support_rows)),
        "hard_support_by_failure_bucket": _count_by(hard_support_rows, "failure_bucket"),
        "hard_support_by_visual_observability": _count_by(hard_support_rows, "visual_observability_class"),
        "hard_support_by_yaw_observability": _count_by(hard_support_rows, "yaw_observability_class"),
        "hard_support_by_takeover_tier": _count_by(hard_support_rows, "takeover_tier"),
    }
    return merged, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge balanced and hard failure-tail manifests into a support-growth slice.")
    ap.add_argument("--core_jsonl", type=Path, required=True)
    ap.add_argument("--supplement_jsonl", type=Path, action="append", required=True)
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_support.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_support.summary.json"),
    )
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    core_rows = _read_jsonl(args.core_jsonl)
    supplement_rows: list[dict[str, Any]] = []
    for supplement_jsonl in args.supplement_jsonl:
        supplement_rows.extend(_read_jsonl(supplement_jsonl))
    merged, summary = merge_manifests(core_rows, supplement_rows, max_rows=args.max_rows)
    summary["core_jsonl"] = str(args.core_jsonl.resolve())
    summary["supplement_jsonl"] = [str(path.resolve()) for path in args.supplement_jsonl]
    summary["output_jsonl"] = str(args.output_jsonl.resolve())

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output_jsonl)
    print(args.summary_json)


if __name__ == "__main__":
    main()
