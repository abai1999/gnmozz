#!/usr/bin/env python3
"""Audit alignment short-circuit sources across gate timing, semantics, scale, and yaw.

This script is intentionally read-only. It summarizes trace-level fields so we can
separate:
  1) takeover timing / gate openness
  2) target-relative source semantics
  3) residual scale before/after clipping
  4) yaw-specific behavior
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _load_traces(trace_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _stats(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _first_true_step(rows: list[dict], key: str) -> int:
    for idx, row in enumerate(rows):
        if bool(row.get(key, False)):
            step_idx = row.get("step_idx", None)
            if step_idx is None:
                step_idx = row.get("episode_step", None)
            if step_idx is None:
                step_idx = idx
            return int(step_idx)
    return -1


def _bool_rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(r.get(key, False)) for r in rows]))


def _array_like(value, fallback_dim: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if fallback_dim is not None and arr.size < fallback_dim:
        return None
    return arr


def _bucket_name(xy: float, z: float, yaw: float) -> str:
    if xy < 0.015 and z < 0.03 and yaw < 0.12:
        return "micro_contact_refine"
    if xy < 0.05 and z < 0.10 and yaw < 0.25:
        return "near_alignment"
    if xy < 0.12 and z < 0.25:
        return "mid_approach_assist"
    return "far_coarse_approach"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_traces(args.trace_dir)
    if not rows:
        raise SystemExit(f"no trace rows found under {args.trace_dir}")

    gate_rows = [r for r in rows if bool(r.get("refiner_alignment_window_active", False))]
    takeover_rows = [r for r in rows if bool(r.get("refiner_alignment_takeover_active", False))]
    nz_rows = [r for r in rows if bool(r.get("refiner_alignment_near_zone_gate_pass", False))]
    close_rows = [r for r in rows if bool(r.get("refiner_alignment_planner_close_intent", False))]

    # 1) Gate timing
    zone_hist = Counter(str(r.get("refiner_zone_state", "unknown")) for r in rows)
    blocked_hist = Counter(str(r.get("refiner_alignment_blocked_reason", "unknown")) for r in rows)
    gate_summary = {
        "rows": len(rows),
        "planner_close_intent_rate": _bool_rate(rows, "refiner_alignment_planner_close_intent"),
        "near_zone_gate_pass_rate": _bool_rate(rows, "refiner_alignment_near_zone_gate_pass"),
        "alignment_window_active_rate": _bool_rate(rows, "refiner_alignment_window_active"),
        "takeover_active_rate": _bool_rate(rows, "refiner_alignment_takeover_active"),
        "first_planner_close_intent_step": _first_true_step(rows, "refiner_alignment_planner_close_intent"),
        "first_near_zone_pass_step": _first_true_step(rows, "refiner_alignment_near_zone_gate_pass"),
        "first_alignment_window_step": _first_true_step(rows, "refiner_alignment_window_active"),
        "first_takeover_step": _first_true_step(rows, "refiner_alignment_takeover_active"),
        "zone_histogram": dict(zone_hist),
        "blocked_reason_histogram": dict(blocked_hist),
        "intersection": {
            "close_and_near": int(sum(1 for r in rows if bool(r.get("refiner_alignment_planner_close_intent")) and bool(r.get("refiner_alignment_near_zone_gate_pass")))),
            "window_and_takeover": int(sum(1 for r in rows if bool(r.get("refiner_alignment_window_active")) and bool(r.get("refiner_alignment_takeover_active")))),
        },
    }

    # 2) Delta source semantics
    src_hist = Counter(str(r.get("refiner_v2_delta_source", "none")) for r in rows)
    gate_src_hist = Counter(str(r.get("refiner_v2_delta_source", "none")) for r in gate_rows)
    pred_src_hist = Counter(str(r.get("refiner_v2_delta_source", "none")) for r in rows if int(r.get("refiner_v2_selected_candidate_index", -1) or -1) >= 0)
    v2_sel_hist = Counter()
    v2_agree = 0
    v2_pred_rows = 0
    source_xy = defaultdict(list)
    source_z = defaultdict(list)
    source_yaw = defaultdict(list)
    source_sel_xy = defaultdict(list)
    source_sel_z = defaultdict(list)
    source_sel_yaw = defaultdict(list)

    for r in rows:
        src = str(r.get("refiner_v2_delta_source", "none"))
        cur_xy = r.get("refiner_v2_cur_xy")
        cur_z = r.get("refiner_v2_cur_z")
        cur_yaw = r.get("refiner_v2_cur_yaw")
        if cur_xy is not None:
            source_xy[src].append(float(cur_xy))
        if cur_z is not None:
            source_z[src].append(float(cur_z))
        if cur_yaw is not None:
            source_yaw[src].append(float(cur_yaw))

        sel_idx = int(r.get("refiner_v2_selected_candidate_index", -1) or -1)
        if sel_idx >= 0:
            v2_pred_rows += 1
            v2_sel_hist[str(sel_idx)] += 1
            post_xy = r.get("refiner_v2_selected_post_xy")
            post_z = r.get("refiner_v2_selected_post_z")
            post_yaw = r.get("refiner_v2_selected_post_yaw")
            if post_xy is not None:
                source_sel_xy[src].append(float(post_xy))
            if post_z is not None:
                source_sel_z[src].append(float(post_z))
            if post_yaw is not None:
                source_sel_yaw[src].append(float(post_yaw))
            fi = int(r.get("refiner_last_scorer_candidate_index", -1) or -1)
            if fi >= 0 and fi == sel_idx:
                v2_agree += 1

    delta_semantics = {
        "source_histogram": dict(src_hist),
        "source_histogram_at_gate_pass": dict(gate_src_hist),
        "source_histogram_at_prediction": dict(pred_src_hist),
        "v2_candidate_histogram": dict(v2_sel_hist),
        "v2_ff_agree_rate": float(v2_agree / max(v2_pred_rows, 1)),
        "source_stats": {},
        "pred_rows": v2_pred_rows,
    }
    for src in sorted(src_hist.keys()):
        delta_semantics["source_stats"][src] = {
            "cur_xy": _stats(source_xy[src]),
            "cur_z": _stats(source_z[src]),
            "cur_yaw": _stats(source_yaw[src]),
            "selected_post_xy": _stats(source_sel_xy[src]),
            "selected_post_z": _stats(source_sel_z[src]),
            "selected_post_yaw": _stats(source_sel_yaw[src]),
        }

    # 3) Residual scale / clipping
    residual_pos = {
        "raw": _stats([float(r.get("refiner_raw_residual_pos_norm", 0.0) or 0.0) for r in rows if r.get("refiner_raw_residual_pos_norm") is not None]),
        "preclip": _stats([float(r.get("refiner_preclip_residual_pos_norm", 0.0) or 0.0) for r in rows if r.get("refiner_preclip_residual_pos_norm") is not None]),
        "clipped": _stats([float(r.get("refiner_clipped_residual_pos_norm", 0.0) or 0.0) for r in rows if r.get("refiner_clipped_residual_pos_norm") is not None]),
        "raw_yaw": _stats([float(r.get("refiner_raw_residual_yaw_abs", 0.0) or 0.0) for r in rows if r.get("refiner_raw_residual_yaw_abs") is not None]),
        "preclip_yaw": _stats([float(r.get("refiner_preclip_residual_yaw_abs", 0.0) or 0.0) for r in rows if r.get("refiner_preclip_residual_yaw_abs") is not None]),
        "clipped_yaw": _stats([float(r.get("refiner_clipped_residual_yaw_abs", 0.0) or 0.0) for r in rows if r.get("refiner_clipped_residual_yaw_abs") is not None]),
        "learned_residual_scale": _stats([float(r.get("refiner_learned_residual_scale", 0.0) or 0.0) for r in rows if r.get("refiner_learned_residual_scale") is not None]),
        "clip_hit_rate_proxy": (
            float(np.mean([
                abs(float(r.get("refiner_preclip_residual_pos_norm", 0.0) or 0.0) - float(r.get("refiner_clipped_residual_pos_norm", 0.0) or 0.0)) > 1e-8
                for r in rows
                if r.get("refiner_preclip_residual_pos_norm") is not None and r.get("refiner_clipped_residual_pos_norm") is not None
            ]))
            if any(
                r.get("refiner_preclip_residual_pos_norm") is not None and r.get("refiner_clipped_residual_pos_norm") is not None
                for r in rows
            )
            else 0.0
        ),
    }

    # 4) Yaw audit
    yaw_rows = [r for r in rows if r.get("refiner_v2_cur_yaw") is not None]
    yaw_summary = {
        "cur_yaw": _stats([float(r.get("refiner_v2_cur_yaw", 0.0) or 0.0) for r in yaw_rows]),
        "selected_post_yaw": _stats([float(r.get("refiner_v2_selected_post_yaw", 0.0) or 0.0) for r in yaw_rows if r.get("refiner_v2_selected_candidate_index", -1) is not None and int(r.get("refiner_v2_selected_candidate_index", -1) or -1) >= 0]),
        "improved_rate": float(np.mean([
            float(r.get("refiner_v2_selected_post_yaw", 1e9) or 1e9) < float(r.get("refiner_v2_cur_yaw", 1e9) or 1e9)
            for r in yaw_rows
            if r.get("refiner_v2_selected_post_yaw") is not None and r.get("refiner_v2_cur_yaw") is not None
        ])) if yaw_rows else 0.0,
        "candidate_hist": dict(v2_sel_hist),
        "far_rows": int(sum(1 for r in yaw_rows if _bucket_name(float(r.get("refiner_v2_cur_xy", 999) or 999), float(r.get("refiner_v2_cur_z", 999) or 999), float(r.get("refiner_v2_cur_yaw", 999) or 999)) == "far_coarse_approach")),
        "near_rows": int(sum(1 for r in yaw_rows if _bucket_name(float(r.get("refiner_v2_cur_xy", 999) or 999), float(r.get("refiner_v2_cur_z", 999) or 999), float(r.get("refiner_v2_cur_yaw", 999) or 999)) in ("near_alignment", "micro_contact_refine"))),
    }

    report = {
        "audit": "alignment_sixway_short_circuit",
        "trace_dir": str(args.trace_dir),
        "gate_timing": gate_summary,
        "delta_semantics": delta_semantics,
        "residual_scale": residual_pos,
        "yaw": yaw_summary,
        "verdict": {
            "planner_close_intent_sparse": gate_summary["planner_close_intent_rate"] < 0.1,
            "near_zone_gate_sparse": gate_summary["near_zone_gate_pass_rate"] < 0.1,
            "predictor_source_dominant": src_hist.get("target_delta_predictor", 0) >= max(src_hist.values()),
            "residual_clipped_often": residual_pos["clip_hit_rate_proxy"] > 0.2,
            "yaw_weaker_than_xyz": yaw_summary["improved_rate"] < 0.5,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
