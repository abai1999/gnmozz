#!/usr/bin/env python3
"""Build a hard-observability supplement for the C2C v2 failure-tail slice.

This supplement is the next step after the balanced + hard coverage manifests:
it pulls in rows that are difficult to observe but that the calibrated yaw
estimator still considers recoverable.  The goal is to widen the support
surface on the hard buckets and harder observability combinations without
changing any runtime gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rank_c2c_v2_frame_observability_limited_candidates import rank_candidates
from scripts.build_c2c_v2_failure_tail_hard_manifest import build_hard_manifest


HARD_FAILURE_BUCKETS = {"large_xy_large_yaw", "large_xy_small_yaw", "small_xy_large_yaw"}


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


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def build_hard_observability_supplement(
    rows: list[dict[str, Any]],
    *,
    checkpoint: Path,
    threshold: float | None = None,
    max_rows_per_episode: int = 96,
    easy_rows_per_coverage_key: int = 4,
    hard_rows_per_coverage_key: int = 24,
    hard_coverage_threshold: int = 4,
    max_rows_per_coverage_key: int = 48,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked_report = rank_candidates(rows, checkpoint=checkpoint, threshold=threshold)
    ranked_by_key = {_key(row): row for row in ranked_report["rows"]}

    annotated: list[dict[str, Any]] = []
    for row in rows:
        rank = ranked_by_key.get(_key(row))
        if not rank:
            continue
        if str(row.get("stage_name", "")) != "RING_GRASP_ALIGN":
            continue
        if str(row.get("skill_type", "")) != "precision_grasp":
            continue
        if not bool(rank.get("recoverable_by_estimator", False)):
            continue
        if str(row.get("visual_observability_class", "")) == "prior_only":
            continue

        enriched = dict(row)
        enriched["frame_observability_limited"] = True
        enriched["frame_observability_recoverable"] = True
        enriched["estimator_yaw_observable_probability"] = float(rank.get("estimator_yaw_observable_probability", 0.0))
        enriched["estimator_yaw_observable_threshold"] = float(rank.get("estimator_yaw_observable_threshold", 0.0))
        enriched["estimator_yaw_margin"] = float(rank.get("estimator_yaw_margin", 0.0))
        enriched["recoverable_by_estimator"] = True
        enriched["selection_reason"] = "hard_observability_recoverable_support"
        enriched["hard_observability_support"] = True
        annotated.append(enriched)

    selected, hard_summary = build_hard_manifest(
        annotated,
        include_success_controls=False,
        max_rows_per_episode=max_rows_per_episode,
        easy_rows_per_coverage_key=easy_rows_per_coverage_key,
        hard_rows_per_coverage_key=hard_rows_per_coverage_key,
        hard_coverage_threshold=hard_coverage_threshold,
        max_rows_per_coverage_key=max_rows_per_coverage_key,
    )

    hard_selected = [
        row
        for row in selected
        if str(row.get("failure_bucket", "")) in HARD_FAILURE_BUCKETS
        and bool(row.get("recoverable_by_estimator", False))
    ]
    for row in selected:
        row["selection_reason"] = "hard_observability_recoverable_support"

    summary = {
        "schema_version": "grasp_failure_tail_hard_observability_supplement_v1",
        "checkpoint": str(checkpoint.resolve()),
        "threshold": float(threshold) if threshold is not None else None,
        "input_rows": int(len(rows)),
        "ranked_rows": int(ranked_report["candidate_rows"]),
        "recoverable_rows": int(ranked_report["recoverable_rows"]),
        "recoverable_rate": float(ranked_report["recoverable_rate"]),
        "annotated_rows": int(len(annotated)),
        "selected_rows": int(len(selected)),
        "selected_hard_rows": int(len(hard_selected)),
        "selected_hard_support_rows": int(sum(str(r.get("takeover_tier", "")) in {"frontier_pullback_candidate", "outer_pullback_candidate", "coarse_pullback_candidate", "near_basin_shell", "micro_entry_ready"} for r in selected)),
        "by_episode": {f"ep{ep:03d}": int(count) for ep, count in sorted(Counter(_safe_int(row.get("episode_idx", -1)) for row in selected).items())},
        "by_failure_bucket": _count_by(selected, "failure_bucket"),
        "by_visual_observability": _count_by(selected, "visual_observability_class"),
        "by_yaw_observability": _count_by(selected, "yaw_observability_class"),
        "by_takeover_tier": _count_by(selected, "takeover_tier"),
        "by_recoverable": _count_by(selected, "recoverable_by_estimator"),
        "by_selection_reason": _count_by(selected, "selection_reason"),
        "hard_summary": hard_summary,
    }
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a hard-observability supplement manifest for C2C v2 failure-tail coverage.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/frame_yaw_estimator_observability_balanced.pt"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_hard_observability.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_hard_observability.summary.json"),
    )
    ap.add_argument("--max_rows_per_episode", type=int, default=96)
    ap.add_argument("--easy_rows_per_coverage_key", type=int, default=4)
    ap.add_argument("--hard_rows_per_coverage_key", type=int, default=24)
    ap.add_argument("--hard_coverage_threshold", type=int, default=4)
    ap.add_argument("--max_rows_per_coverage_key", type=int, default=48)
    args = ap.parse_args()

    rows = _read_jsonl(args.relabel_jsonl)
    selected, summary = build_hard_observability_supplement(
        rows,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        max_rows_per_episode=int(args.max_rows_per_episode),
        easy_rows_per_coverage_key=int(args.easy_rows_per_coverage_key),
        hard_rows_per_coverage_key=int(args.hard_rows_per_coverage_key),
        hard_coverage_threshold=int(args.hard_coverage_threshold),
        max_rows_per_coverage_key=int(args.max_rows_per_coverage_key),
    )
    summary["input_jsonl"] = str(args.relabel_jsonl.resolve())
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
