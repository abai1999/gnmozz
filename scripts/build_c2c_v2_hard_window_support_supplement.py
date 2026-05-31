#!/usr/bin/env python3
"""Build a targeted hard-window support supplement for C2C v2.

This helper is intentionally offline-only.  It is meant to widen the support
surface around a few adjacent episode windows without touching the runtime
gate.  The strict target is still hard-bucket rows that already land in
`coarse_pullback_candidate` / `near_basin_shell`.  If that slice is empty for a
window, the script can optionally promote the nearest hard-bucket
`outside_takeover` rows into an outer pullback support tier so the supplement
is not empty and the next sweep can test whether the support surface actually
grew.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HARD_FAILURE_BUCKETS = {"large_xy_large_yaw", "large_xy_small_yaw", "small_xy_large_yaw"}
STRICT_SUPPORT_TIERS = {"coarse_pullback_candidate", "near_basin_shell"}
RELAXED_SUPPORT_TIER = "outer_pullback_candidate"
FRONTIER_SUPPORT_TIER = "frontier_pullback_candidate"


def _parse_windows(text: str) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            ep = int(part)
            windows.append((ep, ep))
            continue
        left, right = part.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            raise ValueError(f"Invalid episode window: {part}")
        windows.append((start, end))
    if not windows:
        raise ValueError("No episode windows provided")
    return windows


def _in_any_window(ep_idx: int, windows: list[tuple[int, int]]) -> bool:
    return any(start <= ep_idx <= end for start, end in windows)


def _window_tag(ep_idx: int, windows: list[tuple[int, int]]) -> str:
    matches = [(abs(ep_idx - ((start + end) / 2.0)), start, end) for start, end in windows if start <= ep_idx <= end]
    if not matches:
        return ""
    _, start, end = sorted(matches, key=lambda item: (item[0], item[1], item[2]))[0]
    return f"{start}-{end}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def build_hard_window_support_supplement(
    rows: list[dict[str, Any]],
    *,
    episode_windows: list[tuple[int, int]],
    outer_xy_threshold: float = 0.12,
    frontier_xy_threshold: float = 0.18,
    max_rows_per_episode: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidate_rows = 0
    selected_by_episode: Counter[int] = Counter()
    selected_by_mode: Counter[str] = Counter()
    selected_by_window: Counter[str] = Counter()
    selected_by_tier: Counter[str] = Counter()
    selected_by_bucket: Counter[str] = Counter()
    source_tier_counts: Counter[str] = Counter()

    for row_in in rows:
        row = dict(row_in)
        ep = int(row.get("episode_idx", -1))
        if not _in_any_window(ep, episode_windows):
            continue
        if str(row.get("failure_bucket", "")) not in HARD_FAILURE_BUCKETS:
            continue
        candidate_rows += 1

        tier = str(row.get("takeover_tier", ""))
        mode = ""
        enriched = dict(row)
        if tier in STRICT_SUPPORT_TIERS:
            mode = "strict"
            enriched["selection_reason"] = "hard_window_strict_support"
        elif (
            str(row.get("failure_bucket", "")) in HARD_FAILURE_BUCKETS
            and tier in {"outside_takeover", "yaw_entry_blocked"}
            and float(row.get("xy_error", float("inf"))) <= float(frontier_xy_threshold)
        ):
            mode = "frontier"
            source_tier_counts[tier] += 1
            enriched["source_takeover_tier"] = tier
            enriched["takeover_tier"] = FRONTIER_SUPPORT_TIER
            enriched["coarse_pullback_candidate"] = False
            enriched["near_basin_shell"] = False
            enriched["outer_pullback_candidate"] = False
            enriched["abstain_reason"] = ""
            enriched["recommended_intervention_axes"] = ["x", "y"]
            enriched["support_frontier"] = "pre_takeover"
            enriched["selection_reason"] = "hard_window_frontier_support"
        elif tier == "outside_takeover" and float(row.get("xy_error", float("inf"))) <= float(outer_xy_threshold):
            mode = "relaxed"
            source_tier_counts[tier] += 1
            enriched["source_takeover_tier"] = tier
            enriched["takeover_tier"] = RELAXED_SUPPORT_TIER
            enriched["coarse_pullback_candidate"] = False
            enriched["near_basin_shell"] = False
            enriched["outer_pullback_candidate"] = True
            enriched["abstain_reason"] = ""
            enriched["recommended_intervention_axes"] = ["x", "y"]
            enriched["support_frontier"] = "pre_takeover"
            enriched["selection_reason"] = "hard_window_pre_takeover_frontier"
        else:
            continue

        if max_rows_per_episode > 0 and selected_by_episode[ep] >= max_rows_per_episode:
            continue

        enriched["support_window_tag"] = _window_tag(ep, episode_windows)
        enriched["support_window_match"] = True
        enriched["support_mode"] = mode
        if "recommended_intervention_axes" not in enriched:
            enriched["recommended_intervention_axes"] = ["x", "y"] if tier in STRICT_SUPPORT_TIERS else []
        if "abstain_reason" not in enriched:
            enriched["abstain_reason"] = str(row.get("abstain_reason", ""))
        selected.append(enriched)
        selected_by_episode[ep] += 1
        selected_by_mode[mode] += 1
        selected_by_window[str(enriched.get("support_window_tag", ""))] += 1
        selected_by_tier[str(enriched.get("takeover_tier", ""))] += 1
        selected_by_bucket[str(enriched.get("failure_bucket", ""))] += 1

    selected.sort(key=lambda row: (int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1))), str(row.get("failure_bucket", "")), str(row.get("takeover_tier", ""))))
    summary = {
        "schema_version": "grasp_failure_tail_hard_window_supplement_v1",
        "input_rows": int(len(rows)),
        "candidate_rows": int(candidate_rows),
        "selected_rows": int(len(selected)),
        "outer_xy_threshold": float(outer_xy_threshold),
        "frontier_xy_threshold": float(frontier_xy_threshold),
        "episode_windows": [f"{start}-{end}" for start, end in episode_windows],
        "max_rows_per_episode": int(max_rows_per_episode),
        "selected_by_mode": dict(sorted(selected_by_mode.items())),
        "selected_by_window": dict(sorted(selected_by_window.items())),
        "selected_by_takeover_tier": dict(sorted(selected_by_tier.items())),
        "selected_by_failure_bucket": dict(sorted(selected_by_bucket.items())),
        "selected_by_episode": {f"ep{ep:03d}": int(count) for ep, count in sorted(selected_by_episode.items())},
        "source_tier_counts": dict(sorted(source_tier_counts.items())),
        "strict_support_rows": int(selected_by_mode.get("strict", 0)),
        "frontier_support_rows": int(selected_by_mode.get("frontier", 0)),
        "relaxed_frontier_rows": int(selected_by_mode.get("relaxed", 0)),
        "by_selection_reason": _count_by(selected, "selection_reason"),
    }
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a hard-window support supplement for C2C v2 failure-tail data.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument(
        "--episode_windows",
        type=str,
        default="5-7,8-10,10-12,13-15,14-16",
        help="Comma-separated episode windows, e.g. '5-7,8-10,10-12'.",
    )
    ap.add_argument("--outer_xy_threshold", type=float, default=0.12)
    ap.add_argument("--frontier_xy_threshold", type=float, default=0.18)
    ap.add_argument("--max_rows_per_episode", type=int, default=64)
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_hard_window_supplement.jsonl"),
    )
    ap.add_argument(
        "--summary_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates_hard_window_supplement.summary.json"),
    )
    args = ap.parse_args()

    rows = _read_rows(args.relabel_jsonl)
    episode_windows = _parse_windows(args.episode_windows)
    selected, summary = build_hard_window_support_supplement(
        rows,
        episode_windows=episode_windows,
        outer_xy_threshold=float(args.outer_xy_threshold),
        frontier_xy_threshold=float(args.frontier_xy_threshold),
        max_rows_per_episode=int(args.max_rows_per_episode),
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
