#!/usr/bin/env python3
"""Build a hard-coverage grasp failure-tail manifest for C2C v2.

The balanced manifest is useful for coverage, but it can still under-sample the
rows we most need for support-surface growth: low-recovery buckets and harder
observability combinations.  This helper keeps the same offline-only candidate
schema, but biases selection toward:

* harder failure buckets
* partial/prior-only visual observability
* ambiguous/unobservable yaw observability
* conservative takeover tiers such as yaw_entry_blocked / too_far

The goal is not to relax gates.  It is to widen the training / audit support
surface so downstream estimators see more of the real failure tail.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


HARD_FAILURE_BUCKETS = {"large_xy_large_yaw", "large_xy_small_yaw", "small_xy_large_yaw"}
HARD_VISUAL_CLASSES = {"prior_only", "partial_observable"}
HARD_YAW_CLASSES = {"ambiguous", "unobservable"}
HARD_TAKEOVER_TIERS = {"abstain_prior_only", "too_far", "yaw_entry_blocked"}
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


def _coverage_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("failure_bucket", "")),
        str(row.get("visual_observability_class", "")),
        str(row.get("yaw_observability_class", "")),
        str(row.get("takeover_tier", "")),
    )


def _coverage_key_str(key: tuple[str, str, str, str]) -> str:
    return "|".join(key)


def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        _safe_int(row.get("episode_idx", -1)),
        _safe_int(row.get("step_idx", row.get("step", -1))),
        str(row.get("failure_bucket", "")),
        str(row.get("takeover_tier", "")),
    )


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def _priority_components(key: tuple[str, str, str, str]) -> dict[str, int]:
    failure_bucket, visual, yaw, tier = key
    return {
        "failure_bucket": 1 if failure_bucket in HARD_FAILURE_BUCKETS else 0,
        "visual": 1 if visual in HARD_VISUAL_CLASSES else 0,
        "yaw": 1 if yaw in HARD_YAW_CLASSES else 0,
        "support_tier": _support_tier_rank(tier),
        "tier": 1 if tier in HARD_TAKEOVER_TIERS else 0,
    }


def _coverage_priority(key: tuple[str, str, str, str]) -> int:
    parts = _priority_components(key)
    return int(parts["failure_bucket"] * 4 + parts["support_tier"] * 3 + parts["visual"] * 2 + parts["yaw"] * 2 + parts["tier"])


def _row_priority(row: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
    key = _coverage_key(row)
    parts = _priority_components(key)
    # Prefer rows that represent the hard tail, then rows that are not already
    # near-success windows.
    natural_outcome = str(row.get("planner_natural_outcome", ""))
    natural_penalty = 1 if natural_outcome not in {"natural_enters_basin", "already_success_window"} else 0
    soft_focus = 1 if str(row.get("takeover_tier", "")) in SUPPORT_TAKEOVER_TIERS else 0
    return (
        int(parts["failure_bucket"]),
        int(parts["visual"]),
        int(parts["yaw"]),
        int(parts["tier"]),
        int(natural_penalty),
        int(soft_focus),
    )


def build_hard_manifest(
    rows: list[dict[str, Any]],
    *,
    include_success_controls: bool = False,
    max_rows_per_episode: int = 96,
    easy_rows_per_coverage_key: int = 6,
    hard_rows_per_coverage_key: int = 24,
    hard_coverage_threshold: int = 4,
    max_rows_per_coverage_key: int = 48,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic hard-coverage subset of failure-tail candidate rows."""

    filtered: list[dict[str, Any]] = []
    for row in rows:
        sample_role = str(row.get("sample_role", "failure_tail_candidate"))
        if not include_success_controls and sample_role == "success_window_control":
            continue
        if str(row.get("stage_name", "")) != "RING_GRASP_ALIGN":
            continue
        if str(row.get("skill_type", "")) != "precision_grasp":
            continue
        filtered.append(dict(row))

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(filtered, key=_sort_key):
        grouped[_coverage_key(row)].append(row)

    selected: list[dict[str, Any]] = []
    per_episode_counts: Counter[int] = Counter()
    per_key_counts: Counter[tuple[str, str, str, str]] = Counter()
    dropped_by_reason: Counter[str] = Counter()

    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            -_coverage_priority(key),
            len(grouped[key]),
            key,
        ),
    )

    for key in ordered_keys:
        key_rows = sorted(grouped[key], key=lambda row: (-_row_priority(row)[0], -_row_priority(row)[1], -_row_priority(row)[2], -_row_priority(row)[3], -_row_priority(row)[4], -_row_priority(row)[5], _safe_int(row.get("episode_idx", -1)), _safe_int(row.get("step_idx", row.get("step", -1)))))
        key_priority = _coverage_priority(key)
        key_cap = hard_rows_per_coverage_key if key_priority >= int(hard_coverage_threshold) else easy_rows_per_coverage_key
        key_cap = min(int(key_cap), int(max_rows_per_coverage_key))
        key_cap = max(int(key_cap), 1)

        for key_rank, row in enumerate(key_rows):
            episode_idx = _safe_int(row.get("episode_idx", -1))
            if max_rows_per_episode > 0 and per_episode_counts[episode_idx] >= max_rows_per_episode:
                dropped_by_reason["episode_cap"] += 1
                continue
            if max_rows_per_coverage_key > 0 and per_key_counts[key] >= key_cap:
                dropped_by_reason["coverage_key_cap"] += 1
                continue

            enriched = dict(row)
            enriched["coverage_key"] = {
                "failure_bucket": key[0],
                "visual_observability_class": key[1],
                "yaw_observability_class": key[2],
                "takeover_tier": key[3],
            }
            enriched["coverage_key_str"] = _coverage_key_str(key)
            enriched["coverage_key_rank"] = int(key_rank)
            enriched["coverage_key_priority"] = int(key_priority)
            enriched["episode_rank"] = int(per_episode_counts[episode_idx])
            enriched["selection_reason"] = "hard_coverage_prioritized"
            selected.append(enriched)
            per_episode_counts[episode_idx] += 1
            per_key_counts[key] += 1

    selected.sort(key=_sort_key)
    hard_selected = [
        row
        for row in selected
        if str(row.get("failure_bucket", "")) in HARD_FAILURE_BUCKETS
        or str(row.get("visual_observability_class", "")) in HARD_VISUAL_CLASSES
        or str(row.get("yaw_observability_class", "")) in HARD_YAW_CLASSES
        or str(row.get("takeover_tier", "")) in HARD_TAKEOVER_TIERS
    ]
    easy_selected = [row for row in selected if row not in hard_selected]
    hard_support_selected = [
        row
        for row in selected
        if str(row.get("failure_bucket", "")) in HARD_FAILURE_BUCKETS
        and str(row.get("takeover_tier", "")) in SUPPORT_TAKEOVER_TIERS
    ]
    summary = {
        "schema_version": "grasp_failure_tail_hard_manifest_v1",
        "input_rows": int(len(rows)),
        "filtered_rows": int(len(filtered)),
        "selected_rows": int(len(selected)),
        "selected_hard_rows": int(len(hard_selected)),
        "selected_easy_rows": int(len(easy_selected)),
        "selected_hard_support_rows": int(len(hard_support_selected)),
        "dropped_rows": int(len(filtered) - len(selected)),
        "include_success_controls": bool(include_success_controls),
        "max_rows_per_episode": int(max_rows_per_episode),
        "easy_rows_per_coverage_key": int(easy_rows_per_coverage_key),
        "hard_rows_per_coverage_key": int(hard_rows_per_coverage_key),
        "hard_coverage_threshold": int(hard_coverage_threshold),
        "max_rows_per_coverage_key": int(max_rows_per_coverage_key),
        "dropped_by_reason": dict(dropped_by_reason),
        "by_episode": {f"ep{ep:03d}": int(count) for ep, count in sorted(per_episode_counts.items())},
        "by_failure_bucket": _count_by(selected, "failure_bucket"),
        "by_visual_observability": _count_by(selected, "visual_observability_class"),
        "by_yaw_observability": _count_by(selected, "yaw_observability_class"),
        "by_takeover_tier": _count_by(selected, "takeover_tier"),
        "by_hard_support_takeover_tier": _count_by(hard_support_selected, "takeover_tier"),
        "by_hard_support_failure_bucket": _count_by(hard_support_selected, "failure_bucket"),
        "by_hard_support_yaw_observability": _count_by(hard_support_selected, "yaw_observability_class"),
        "by_sample_role": _count_by(selected, "sample_role"),
        "by_coverage_key": {
            _coverage_key_str(key): int(count)
            for key, count in sorted(per_key_counts.items(), key=lambda item: (-item[1], item[0]))
        },
    }
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a hard-coverage failure-tail manifest from grasp failure-tail candidates.")
    ap.add_argument("--input_jsonl", type=Path, required=True)
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_hard.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_hard.summary.json"),
    )
    ap.add_argument("--include_success_controls", action="store_true", default=False)
    ap.add_argument("--max_rows_per_episode", type=int, default=96)
    ap.add_argument("--easy_rows_per_coverage_key", type=int, default=6)
    ap.add_argument("--hard_rows_per_coverage_key", type=int, default=24)
    ap.add_argument("--hard_coverage_threshold", type=int, default=4)
    ap.add_argument("--max_rows_per_coverage_key", type=int, default=48)
    args = ap.parse_args()

    rows = _read_jsonl(args.input_jsonl)
    selected, summary = build_hard_manifest(
        rows,
        include_success_controls=bool(args.include_success_controls),
        max_rows_per_episode=int(args.max_rows_per_episode),
        easy_rows_per_coverage_key=int(args.easy_rows_per_coverage_key),
        hard_rows_per_coverage_key=int(args.hard_rows_per_coverage_key),
        hard_coverage_threshold=int(args.hard_coverage_threshold),
        max_rows_per_coverage_key=int(args.max_rows_per_coverage_key),
    )
    summary["input_jsonl"] = str(args.input_jsonl.resolve())
    summary["output_jsonl"] = str(args.output_jsonl.resolve())

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output_jsonl)
    print(args.summary_json)


if __name__ == "__main__":
    main()
