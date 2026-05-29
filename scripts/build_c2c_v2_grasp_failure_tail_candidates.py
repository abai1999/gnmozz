#!/usr/bin/env python3
"""Build grasp failure-tail candidates from planner-only frame residual labels.

The output of this script is the main sampling surface for C2C v2 grasp
correction evidence.  It intentionally filters away planner success windows so
the downstream intervention probe is evaluated on planner behavior that still
needs high-precision correction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.takeover_contract import (
    CLOSE_READY_XY_THRESHOLD,
    CLOSE_READY_YAW_THRESHOLD,
    CLOSE_READY_Z_THRESHOLD,
    COARSE_PULLBACK_XY_THRESHOLD,
    NEAR_GRASP_XY_THRESHOLD,
    NEAR_GRASP_YAW_THRESHOLD,
    TIER_CLOSE_READY,
    TIER_COARSE_PULLBACK,
    TIER_MICRO_ENTRY,
    TIER_NEAR_BASIN,
    TIER_OUTER_PULLBACK,
    OUTER_PULLBACK_XY_THRESHOLD,
)


SUCCESS_TIERS = {TIER_MICRO_ENTRY, TIER_CLOSE_READY}
HARD_FAILURE_BUCKETS = {"large_xy_large_yaw", "large_xy_small_yaw", "small_xy_large_yaw"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _xy(row: Mapping[str, Any], *, next_value: bool = False) -> float:
    if next_value:
        return _safe_float(row.get("next_xy_error", float("nan")))
    return _safe_float(row.get("xy_error", float("nan")))


def _yaw(row: Mapping[str, Any], *, next_value: bool = False) -> float:
    if next_value:
        return _safe_float(row.get("next_yaw_abs", float("nan")))
    return _safe_float(row.get("yaw_abs", float("nan")))


def _yaw_entry_feasible(row: Mapping[str, Any]) -> bool:
    if "yaw_entry_feasible" in row:
        return bool(row.get("yaw_entry_feasible", False))
    yaw = _yaw(row)
    return bool(np.isfinite(yaw) and yaw <= NEAR_GRASP_YAW_THRESHOLD)


def _yaw_control_observable(row: Mapping[str, Any]) -> bool:
    if "yaw_control_observable" in row:
        return bool(row.get("yaw_control_observable", False))
    return bool(row.get("yaw_observable", False))


def _dz(row: Mapping[str, Any], *, next_value: bool = False) -> float:
    if next_value:
        nested = row.get("true_basin_error_t_plus_1") if isinstance(row.get("true_basin_error_t_plus_1"), Mapping) else {}
        return abs(_safe_float(nested.get("dz", row.get("next_privileged_dz", float("nan")))))
    return abs(_safe_float(row.get("privileged_dz", row.get("true_basin_error_t", {}).get("dz", float("nan")))))


def _near_grasp(row: Mapping[str, Any], *, next_value: bool = False) -> bool:
    xy = _xy(row, next_value=next_value)
    yaw = _yaw(row, next_value=next_value)
    return bool(np.isfinite(xy) and np.isfinite(yaw) and xy <= NEAR_GRASP_XY_THRESHOLD and yaw <= NEAR_GRASP_YAW_THRESHOLD)


def _close_ready(row: Mapping[str, Any], *, next_value: bool = False) -> bool:
    xy = _xy(row, next_value=next_value)
    yaw = _yaw(row, next_value=next_value)
    dz = _dz(row, next_value=next_value)
    return bool(
        np.isfinite(xy)
        and np.isfinite(yaw)
        and np.isfinite(dz)
        and xy <= CLOSE_READY_XY_THRESHOLD
        and yaw <= CLOSE_READY_YAW_THRESHOLD
        and dz <= CLOSE_READY_Z_THRESHOLD
    )


def _planner_natural_outcome(row: Mapping[str, Any]) -> str:
    if _near_grasp(row) or _close_ready(row) or str(row.get("takeover_tier", "")) in SUCCESS_TIERS:
        return "already_success_window"
    if _near_grasp(row, next_value=True) or _close_ready(row, next_value=True):
        return "natural_enters_basin"
    if bool(row.get("xy_contracted", row.get("contraction", False))):
        return "natural_contracts"
    if _finite(row.get("next_xy_error")) and _finite(row.get("xy_error")):
        return "natural_diverges_or_stalls"
    return "natural_outcome_unknown"


def _failure_tail_tier(row: Mapping[str, Any]) -> str:
    if not bool(row.get("label_valid", True)):
        return "invalid"
    if str(row.get("visual_observability_class", "")) == "prior_only" or bool(row.get("reacquire_needed", False)):
        return "abstain_prior_only"
    if _close_ready(row):
        return TIER_CLOSE_READY
    if _near_grasp(row):
        return TIER_MICRO_ENTRY
    if bool(row.get("near_basin_shell", False)):
        return TIER_NEAR_BASIN
    xy = _xy(row)
    if np.isfinite(xy) and xy <= COARSE_PULLBACK_XY_THRESHOLD and _yaw_entry_feasible(row):
        return TIER_COARSE_PULLBACK
    # Keep a bucket-specific outer support tier for small_xy_large_yaw even when
    # yaw is still blocked.  This widens the offline support surface without
    # claiming entry readiness, and keeps yaw as a separate audited axis.
    if (
        str(row.get("failure_bucket", "")) == "small_xy_large_yaw"
        and np.isfinite(xy)
        and xy <= OUTER_PULLBACK_XY_THRESHOLD
    ):
        return TIER_OUTER_PULLBACK
    if (
        str(row.get("failure_bucket", "")) in HARD_FAILURE_BUCKETS
        and np.isfinite(xy)
        and xy <= OUTER_PULLBACK_XY_THRESHOLD
        and bool(row.get("xy_contracted", row.get("contraction", False)))
    ):
        return TIER_OUTER_PULLBACK
    if not _yaw_entry_feasible(row):
        return "yaw_entry_blocked"
    return "too_far"


def _abstain_reason(row: Mapping[str, Any], *, failure_tail_tier: str) -> str:
    if not bool(row.get("label_valid", True)):
        return str(row.get("label_invalid_reason", "invalid_residual") or "invalid_residual")
    if failure_tail_tier == "abstain_prior_only":
        return "prior_only"
    if failure_tail_tier == "yaw_entry_blocked":
        return str(row.get("yaw_entry_block_reason", "yaw_abs_gt_near_threshold") or "yaw_abs_gt_near_threshold")
    if failure_tail_tier == "too_far":
        return "outside_coarse_pullback_window"
    return ""


def _recommended_axes(row: Mapping[str, Any], *, failure_tail_tier: str) -> list[str]:
    if failure_tail_tier in {TIER_COARSE_PULLBACK, TIER_OUTER_PULLBACK, TIER_NEAR_BASIN, TIER_MICRO_ENTRY}:
        return ["x", "y"]
    return []


def build_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    include_success_controls: bool = False,
    exclude_episode_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    excluded = exclude_episode_indices or set()
    for row_in in rows:
        row = dict(row_in)
        ep = int(row.get("episode_idx", -1))
        if ep in excluded:
            continue
        if str(row.get("stage_name", "")) != "RING_GRASP_ALIGN":
            continue
        if str(row.get("skill_type", "")) != "precision_grasp":
            continue

        natural_outcome = _planner_natural_outcome(row)
        success_window = natural_outcome in {"already_success_window", "natural_enters_basin"}
        if success_window and not include_success_controls:
            continue

        failure_tail_tier = _failure_tail_tier(row)
        sample_role = "success_window_control" if success_window else "failure_tail_candidate"
        abstain_reason = _abstain_reason(row, failure_tail_tier=failure_tail_tier)
        recommended_axes = _recommended_axes(row, failure_tail_tier=failure_tail_tier)

        candidate = {
            "schema_version": "grasp_failure_tail_candidate_v1",
            "sample_role": sample_role,
            "task_name": str(row.get("task_name", "insert_onto_square_peg")),
            "episode_idx": ep,
            "step_idx": int(row.get("step_idx", row.get("step", -1))),
            "stage_name": str(row.get("stage_name", "")),
            "skill_name": str(row.get("skill_name", "")),
            "skill_type": str(row.get("skill_type", "")),
            "obs_pointer": {
                "runtime_obs_path": str(row.get("source_runtime_obs_path", "")),
                "trace_path": str(row.get("source_trace_path", "")),
                "episode_idx": ep,
                "step_idx": int(row.get("step_idx", row.get("step", -1))),
            },
            "planner_prior": row.get("planner_prior", {}),
            "true_residual": row.get("true_basin_error_t", {}),
            "next_planner_residual": row.get("true_basin_error_t_plus_1", {}),
            "xy_error": _xy(row),
            "yaw_abs": _yaw(row),
            "next_xy_error": _xy(row, next_value=True),
            "next_yaw_abs": _yaw(row, next_value=True),
            "planner_natural_outcome": natural_outcome,
            "planner_natural_xy_contracted": bool(row.get("xy_contracted", row.get("contraction", False))),
            "planner_natural_overshoot": bool(row.get("overshoot", False)),
            "visual_observability_class": str(row.get("visual_observability_class", "")),
            "yaw_observability_class": str(row.get("yaw_observability_class", "")),
            "yaw_observable": bool(row.get("yaw_observable", False)),
            "yaw_entry_feasible": bool(_yaw_entry_feasible(row)),
            "yaw_control_observable": bool(_yaw_control_observable(row)),
            "yaw_entry_block_reason": str(row.get("yaw_entry_block_reason", "")),
            "yaw_control_block_reason": str(row.get("yaw_control_block_reason", "")),
            "source_takeover_tier": str(row.get("takeover_tier", "")),
            "takeover_tier": failure_tail_tier,
            "near_basin_shell": bool(failure_tail_tier == TIER_NEAR_BASIN),
            "coarse_pullback_candidate": bool(failure_tail_tier == TIER_COARSE_PULLBACK),
            "outer_pullback_candidate": bool(failure_tail_tier == TIER_OUTER_PULLBACK),
            "failure_bucket": str(row.get("failure_bucket", "")),
            "recommended_intervention_axes": recommended_axes,
            "abstain_reason": abstain_reason,
            "label_valid": bool(row.get("label_valid", True)),
            "label_invalid_reason": str(row.get("label_invalid_reason", "")),
            "uses_privileged_label": True,
            "uses_privileged_runtime": False,
            "uses_privileged_target": False,
            "uses_rlbench_mask_runtime": False,
            "source_relabel_schema": str(row.get("schema_version", "")),
        }
        out.append(candidate)
    out.sort(key=lambda r: (int(r["episode_idx"]), int(r["step_idx"])))
    return out


def summarize(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = {
        "by_sample_role": Counter(str(r.get("sample_role", "")) for r in candidates),
        "by_takeover_tier": Counter(str(r.get("takeover_tier", "")) for r in candidates),
        "by_yaw_observability": Counter(str(r.get("yaw_observability_class", "")) for r in candidates),
        "by_visual_observability": Counter(str(r.get("visual_observability_class", "")) for r in candidates),
        "by_failure_bucket": Counter(str(r.get("failure_bucket", "")) for r in candidates),
        "by_planner_natural_outcome": Counter(str(r.get("planner_natural_outcome", "")) for r in candidates),
    }
    by_episode: dict[int, int] = defaultdict(int)
    for row in candidates:
        by_episode[int(row.get("episode_idx", -1))] += 1
    actionable = [r for r in candidates if r.get("recommended_intervention_axes")]
    return {
        "num_rows": int(len(candidates)),
        "actionable_rows": int(len(actionable)),
        "abstain_rows": int(sum(1 for r in candidates if str(r.get("abstain_reason", "")))),
        "by_episode": {f"ep{ep:03d}": count for ep, count in sorted(by_episode.items())},
        **{key: dict(counter) for key, counter in groups.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build C2C v2 grasp failure-tail candidate rows from frame_residual_v2 labels.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets/grasp_failure_tail_candidates.jsonl"),
    )
    ap.add_argument("--summary_json", type=Path, default=None)
    ap.add_argument("--include_success_controls", action="store_true", default=False)
    ap.add_argument(
        "--exclude_episode_indices",
        type=str,
        default="6",
        help="Comma-separated episode indices excluded from main evidence by default; ep6 remains a sanity/control sample.",
    )
    args = ap.parse_args()

    excluded = {int(part.strip()) for part in str(args.exclude_episode_indices).split(",") if part.strip()}
    rows = read_jsonl(args.relabel_jsonl)
    candidates = build_candidates(rows, include_success_controls=bool(args.include_success_controls), exclude_episode_indices=excluded)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "source_relabel_jsonl": str(args.relabel_jsonl.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "include_success_controls": bool(args.include_success_controls),
        "excluded_episode_indices": sorted(excluded),
        **summarize(candidates),
    }
    summary_json = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
