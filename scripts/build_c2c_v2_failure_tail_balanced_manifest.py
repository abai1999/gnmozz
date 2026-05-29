#!/usr/bin/env python3
"""Build a coverage-balanced grasp failure-tail manifest for C2C v2.

The failure-tail candidate surface is often dominated by a few easy-to-recover
episodes or one or two bucket/observability combinations.  This helper keeps
the failure-tail semantics intact while capping the number of rows taken from
any single coverage key so downstream estimator training sees a broader slice
of the support.

Coverage is stratified by:

* failure bucket
* visual observability class
* yaw observability class
* takeover tier

The output stays offline-only and preserves the candidate schema, but adds a
few bookkeeping fields that make the slice auditable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


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


def build_balanced_manifest(
    rows: list[dict[str, Any]],
    *,
    include_success_controls: bool = False,
    max_rows_per_episode: int = 96,
    max_rows_per_coverage_key: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic balanced subset of failure-tail candidate rows."""

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

    for key in sorted(grouped, key=lambda item: (len(grouped[item]), item)):
        key_rows = grouped[key]
        for key_rank, row in enumerate(key_rows):
            episode_idx = _safe_int(row.get("episode_idx", -1))
            if max_rows_per_episode > 0 and per_episode_counts[episode_idx] >= max_rows_per_episode:
                dropped_by_reason["episode_cap"] += 1
                continue
            if max_rows_per_coverage_key > 0 and per_key_counts[key] >= max_rows_per_coverage_key:
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
            enriched["episode_rank"] = int(per_episode_counts[episode_idx])
            enriched["selection_reason"] = "balanced_failure_tail_coverage"
            selected.append(enriched)
            per_episode_counts[episode_idx] += 1
            per_key_counts[key] += 1

    selected.sort(key=_sort_key)
    summary = {
        "schema_version": "grasp_failure_tail_balanced_manifest_v1",
        "input_rows": int(len(rows)),
        "filtered_rows": int(len(filtered)),
        "selected_rows": int(len(selected)),
        "dropped_rows": int(len(filtered) - len(selected)),
        "include_success_controls": bool(include_success_controls),
        "max_rows_per_episode": int(max_rows_per_episode),
        "max_rows_per_coverage_key": int(max_rows_per_coverage_key),
        "dropped_by_reason": dict(dropped_by_reason),
        "by_episode": {f"ep{ep:03d}": int(count) for ep, count in sorted(per_episode_counts.items())},
        "by_failure_bucket": _count_by(selected, "failure_bucket"),
        "by_visual_observability": _count_by(selected, "visual_observability_class"),
        "by_yaw_observability": _count_by(selected, "yaw_observability_class"),
        "by_takeover_tier": _count_by(selected, "takeover_tier"),
        "by_sample_role": _count_by(selected, "sample_role"),
        "by_coverage_key": {
            _coverage_key_str(key): int(count)
            for key, count in sorted(per_key_counts.items(), key=lambda item: (-item[1], item[0]))
        },
    }
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a balanced failure-tail manifest from grasp failure-tail candidates.")
    ap.add_argument("--input_jsonl", type=Path, required=True)
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_balanced.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_balanced.summary.json"),
    )
    ap.add_argument("--include_success_controls", action="store_true", default=False)
    ap.add_argument("--max_rows_per_episode", type=int, default=96)
    ap.add_argument("--max_rows_per_coverage_key", type=int, default=32)
    args = ap.parse_args()

    rows = _read_jsonl(args.input_jsonl)
    selected, summary = build_balanced_manifest(
        rows,
        include_success_controls=bool(args.include_success_controls),
        max_rows_per_episode=int(args.max_rows_per_episode),
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
