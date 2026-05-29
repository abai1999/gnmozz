#!/usr/bin/env python3
"""Build a focused hard-bucket grasp failure-tail manifest.

This helper trims an existing failure-tail candidate JSONL down to a specific
bucket / tier / visibility slice so the next probe sweep can stay narrow and
auditable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def _parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        _safe_int(row.get("episode_idx", -1)),
        _safe_int(row.get("step_idx", row.get("step", -1))),
        str(row.get("failure_bucket", "")),
        str(row.get("takeover_tier", "")),
    )


def build_focus_manifest(
    rows: list[dict[str, Any]],
    *,
    failure_buckets: set[str],
    allowed_takeover_tiers: set[str] | None = None,
    require_non_prior_only: bool = True,
    require_grasp_stage: bool = True,
    require_grasp_skill: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for row_in in rows:
        row = dict(row_in)
        if require_grasp_stage and str(row.get("stage_name", "")) != "RING_GRASP_ALIGN":
            dropped["stage"] += 1
            continue
        if require_grasp_skill and str(row.get("skill_type", "")) != "precision_grasp":
            dropped["skill"] += 1
            continue
        if failure_buckets and str(row.get("failure_bucket", "")) not in failure_buckets:
            dropped["failure_bucket"] += 1
            continue
        if allowed_takeover_tiers is not None and str(row.get("takeover_tier", "")) not in allowed_takeover_tiers:
            dropped["takeover_tier"] += 1
            continue
        if require_non_prior_only and str(row.get("visual_observability_class", "")) == "prior_only":
            dropped["prior_only"] += 1
            continue

        enriched = dict(row)
        enriched["focus_failure_bucket"] = str(row.get("failure_bucket", ""))
        enriched["focus_takeover_tier"] = str(row.get("takeover_tier", ""))
        enriched["focus_visual_observability_class"] = str(row.get("visual_observability_class", ""))
        enriched["focus_yaw_observability_class"] = str(row.get("yaw_observability_class", ""))
        enriched["focus_selection_reason"] = "hard_bucket_focus"
        selected.append(enriched)

    selected.sort(key=_sort_key)
    summary = {
        "schema_version": "grasp_failure_tail_hard_bucket_focus_manifest_v1",
        "input_rows": int(len(rows)),
        "selected_rows": int(len(selected)),
        "failure_buckets": sorted(failure_buckets),
        "allowed_takeover_tiers": sorted(allowed_takeover_tiers) if allowed_takeover_tiers is not None else None,
        "require_non_prior_only": bool(require_non_prior_only),
        "require_grasp_stage": bool(require_grasp_stage),
        "require_grasp_skill": bool(require_grasp_skill),
        "dropped_by_reason": dict(dropped),
        "by_episode": {f"ep{ep:03d}": int(count) for ep, count in sorted(Counter(_safe_int(row.get("episode_idx", -1)) for row in selected).items())},
        "by_failure_bucket": _count_by(selected, "failure_bucket"),
        "by_takeover_tier": _count_by(selected, "takeover_tier"),
        "by_visual_observability": _count_by(selected, "visual_observability_class"),
        "by_yaw_observability": _count_by(selected, "yaw_observability_class"),
    }
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a focused hard-bucket failure-tail manifest.")
    ap.add_argument("--input_jsonl", type=Path, required=True)
    ap.add_argument("--failure_buckets", type=str, required=True, help="Comma-separated failure buckets to keep.")
    ap.add_argument(
        "--allowed_takeover_tiers",
        type=str,
        default="",
        help="Comma-separated takeover tiers to keep. Leave empty to keep all tiers.",
    )
    ap.add_argument("--keep_prior_only", action="store_true", default=False)
    ap.add_argument("--no_require_grasp_stage", action="store_true", default=False)
    ap.add_argument("--no_require_grasp_skill", action="store_true", default=False)
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_focus.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_focus.summary.json"),
    )
    args = ap.parse_args()

    rows = _read_jsonl(args.input_jsonl)
    failure_buckets = set(_parse_csv(args.failure_buckets))
    allowed_tiers_raw = _parse_csv(args.allowed_takeover_tiers)
    allowed_takeover_tiers = set(allowed_tiers_raw) if allowed_tiers_raw else None
    selected, summary = build_focus_manifest(
        rows,
        failure_buckets=failure_buckets,
        allowed_takeover_tiers=allowed_takeover_tiers,
        require_non_prior_only=not bool(args.keep_prior_only),
        require_grasp_stage=not bool(args.no_require_grasp_stage),
        require_grasp_skill=not bool(args.no_require_grasp_skill),
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
