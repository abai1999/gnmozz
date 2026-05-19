#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_rows(trace_dir: Path) -> list[dict]:
    files = sorted((trace_dir / "gripper_traces").glob("ep*_gripper_trace.jsonl"))
    rows: list[dict] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _mean(rows: list[dict], key: str) -> float:
    vals = []
    for r in rows:
        v = r.get(key, None)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else 0.0


def _rows_where(rows: list[dict], predicate) -> list[dict]:
    return [r for r in rows if predicate(r)]


def _episode_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    episodes = sorted({int(r["episode"]) for r in rows})
    out: dict[str, dict[str, float]] = {}
    for ep in episodes:
        bucket = [r for r in rows if int(r["episode"]) == ep]
        out[str(ep)] = {
            "rows": int(len(bucket)),
            "selected_best_safe_hit_rate": _mean(bucket, "shadow_selected_best_safe_hit"),
            "selected_pareto_hit_rate": _mean(bucket, "shadow_selected_pareto_hit"),
            "selected_geom_gain_mean": _mean(bucket, "shadow_selected_geom_gain"),
            "selected_risk_delta_mean": _mean(bucket, "shadow_selected_risk_delta"),
            "best_safe_rank_mean": _mean(bucket, "shadow_best_safe_score_rank"),
            "pareto_rank_mean": _mean(bucket, "shadow_pareto_score_rank"),
            "fallback_rate": _mean(bucket, "shadow_fallback_used"),
            "selected_from_pareto_pool_rate": _mean(bucket, "shadow_selected_from_pareto_pool"),
            "selected_by_best_safe_head_rate": _mean(bucket, "shadow_selected_by_best_safe_head"),
            "selected_by_geometry_head_rate": _mean(bucket, "shadow_selected_by_geometry_head"),
            "selected_with_yaw_bonus_rate": _mean(bucket, "shadow_selected_with_yaw_bonus"),
            "risk_tiebreak_used_rate": _mean(bucket, "shadow_risk_tiebreak_used"),
            "close_contact_rate": _mean(bucket, "shadow_close_contact"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--close_contact_depth_threshold", type=float, default=0.08)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    rows = _load_rows(trace_dir)
    if not rows:
        raise RuntimeError(f"no trace rows found under {trace_dir}")

    close_rows = _rows_where(rows, lambda r: bool(r.get("shadow_close_contact", False)))
    non_close_rows = _rows_where(rows, lambda r: not bool(r.get("shadow_close_contact", False)))

    selected_changed_rate = float(np.mean([bool(r.get("shadow_changed", False)) for r in rows]))
    applied_equals_baseline_rate = float(np.mean([int(r.get("shadow_applied_index", -1)) == int(r.get("shadow_baseline_idx", -2)) for r in rows]))
    shadow_applied_false_rate = float(np.mean([not bool(r.get("shadow_applied", True)) for r in rows]))

    report = {
        "trace_dir": str(trace_dir),
        "rows": int(len(rows)),
        "episodes": sorted({int(r["episode"]) for r in rows}),
        "shadow_name": str(rows[0].get("shadow_name", "unknown")),
        "overall": {
            "shadow_gate_open_rate": _mean(rows, "shadow_gate_open"),
            "shadow_applied_rate": _mean(rows, "shadow_applied"),
            "shadow_applied_false_rate": shadow_applied_false_rate,
            "applied_equals_baseline_rate": applied_equals_baseline_rate,
            "shadow_changed_rate": selected_changed_rate,
            "selected_from_pareto_pool_rate": _mean(rows, "shadow_selected_from_pareto_pool"),
            "fallback_rate": _mean(rows, "shadow_fallback_used"),
            "selected_by_best_safe_head_rate": _mean(rows, "shadow_selected_by_best_safe_head"),
            "selected_by_geometry_head_rate": _mean(rows, "shadow_selected_by_geometry_head"),
            "selected_with_yaw_bonus_rate": _mean(rows, "shadow_selected_with_yaw_bonus"),
            "risk_tiebreak_used_rate": _mean(rows, "shadow_risk_tiebreak_used"),
            "selected_best_safe_hit_rate": _mean(rows, "shadow_selected_best_safe_hit"),
            "selected_pareto_hit_rate": _mean(rows, "shadow_selected_pareto_hit"),
            "selected_geom_gain_mean": _mean(rows, "shadow_selected_geom_gain"),
            "selected_risk_delta_mean": _mean(rows, "shadow_selected_risk_delta"),
            "best_safe_rank_mean": _mean(rows, "shadow_best_safe_score_rank"),
            "pareto_rank_mean": _mean(rows, "shadow_pareto_score_rank"),
            "selected_yaw_match_rate": _mean(rows, "shadow_selected_yaw_match_rate"),
            "selected_correct_yaw_sign_rate": _mean(rows, "shadow_selected_correct_yaw_sign"),
            "close_contact_rate": _mean(rows, "shadow_close_contact"),
        },
        "close_contact": {
            "rows": int(len(close_rows)),
            "selected_best_safe_hit_rate": _mean(close_rows, "shadow_selected_best_safe_hit"),
            "selected_pareto_hit_rate": _mean(close_rows, "shadow_selected_pareto_hit"),
            "selected_geom_gain_mean": _mean(close_rows, "shadow_selected_geom_gain"),
            "selected_risk_delta_mean": _mean(close_rows, "shadow_selected_risk_delta"),
            "best_safe_rank_mean": _mean(close_rows, "shadow_best_safe_score_rank"),
            "pareto_rank_mean": _mean(close_rows, "shadow_pareto_score_rank"),
            "fallback_rate": _mean(close_rows, "shadow_fallback_used"),
            "selected_from_pareto_pool_rate": _mean(close_rows, "shadow_selected_from_pareto_pool"),
        },
        "non_close_contact": {
            "rows": int(len(non_close_rows)),
            "selected_best_safe_hit_rate": _mean(non_close_rows, "shadow_selected_best_safe_hit"),
            "selected_pareto_hit_rate": _mean(non_close_rows, "shadow_selected_pareto_hit"),
            "selected_geom_gain_mean": _mean(non_close_rows, "shadow_selected_geom_gain"),
            "selected_risk_delta_mean": _mean(non_close_rows, "shadow_selected_risk_delta"),
            "best_safe_rank_mean": _mean(non_close_rows, "shadow_best_safe_score_rank"),
            "pareto_rank_mean": _mean(non_close_rows, "shadow_pareto_score_rank"),
            "fallback_rate": _mean(non_close_rows, "shadow_fallback_used"),
            "selected_from_pareto_pool_rate": _mean(non_close_rows, "shadow_selected_from_pareto_pool"),
        },
        "per_episode": _episode_summary(rows),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
