#!/usr/bin/env python3
"""Audit v2 shadow contract for predictor-driven runtime deltas.

This script is read-only. It summarizes runtime trace rows for 3ep shadow runs
and checks whether v2 is actually driven by the target_delta_predictor rather
than basin or zero fallbacks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PREDICTOR_SOURCES = {
    "target_delta_predictor",
    "learned_target_predictor",
    "learned_target_predictor__canonical_close_orientation_contract",
}
DIAGNOSTIC_SOURCES = {
    "runtime_motion_target_pose",
    "canonical_basin_center_pose",
    "hardcoded_basin_center",
    "current_delta_np",
    "fallback_zero",
    "none",
}


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


def _to_array(value, *, fallback_dim: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = arr.reshape(-1)
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


def _extract_selected_trace_metrics(row: dict) -> dict | None:
    idx = row.get("refiner_v2_selected_candidate_index", -1)
    if idx is None or int(idx) < 0:
        return None

    post = _to_array(row.get("refiner_v2_post_candidate_delta"))
    geom = _to_array(row.get("refiner_v2_geometry_improvement"))
    cur_xy = row.get("refiner_v2_cur_xy")
    cur_z = row.get("refiner_v2_cur_z")
    cur_yaw = row.get("refiner_v2_cur_yaw")
    if post is None or post.size < 6:
        return None

    idx = int(idx)
    if post.size % 6 != 0:
        return None
    cand_count = post.size // 6
    if idx >= cand_count:
        return None
    post = post.reshape(cand_count, 6)
    selected = post[idx]
    selected_xy = float(np.linalg.norm(selected[:2]))
    selected_z = float(abs(selected[2]))
    selected_yaw = float(abs(selected[5]))

    sel_geom = None
    if geom is not None and geom.size > idx:
        sel_geom = float(geom[idx])

    return {
        "selected_candidate_index": idx,
        "selected_post_xy": selected_xy,
        "selected_post_z": selected_z,
        "selected_post_yaw": selected_yaw,
        "selected_post_bucket": _bucket_name(selected_xy, selected_z, selected_yaw),
        "selected_geometry_improvement": sel_geom,
        "current_xy": None if cur_xy is None else float(cur_xy),
        "current_z": None if cur_z is None else float(cur_z),
        "current_yaw": None if cur_yaw is None else float(cur_yaw),
        "selected_xy_improved": None if cur_xy is None else bool(selected_xy < float(cur_xy)),
        "selected_z_improved": None if cur_z is None else bool(selected_z < float(cur_z)),
        "selected_yaw_improved": None if cur_yaw is None else bool(selected_yaw < float(cur_yaw)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_traces(args.trace_dir)
    if not rows:
        raise SystemExit(f"no trace rows found under {args.trace_dir}")

    source_hist = Counter()
    source_gate_hist = Counter()
    source_pred_hist = Counter()
    source_delta_norms = defaultdict(list)
    source_xy = defaultdict(list)
    source_z = defaultdict(list)
    source_yaw = defaultdict(list)
    source_sel_hist = defaultdict(Counter)
    source_sel_xy = defaultdict(list)
    source_sel_z = defaultdict(list)
    source_sel_yaw = defaultdict(list)
    source_sel_geom = defaultdict(list)
    source_sel_xy_improve = defaultdict(list)
    source_sel_z_improve = defaultdict(list)
    source_sel_yaw_improve = defaultdict(list)

    all_sel_hist = Counter()
    all_gate = 0
    all_pred = 0
    all_pred_sources = Counter()
    all_gate_sources = Counter()
    all_selected_sources = Counter()
    all_selected_rows = 0

    for row in rows:
        src = str(row.get("refiner_v2_delta_source", "none"))
        gate = bool(row.get("refiner_v2_gate_pass", False))
        idx = int(row.get("refiner_v2_selected_candidate_index", -1) or -1)
        cur_xy = row.get("refiner_v2_cur_xy")
        cur_z = row.get("refiner_v2_cur_z")
        cur_yaw = row.get("refiner_v2_cur_yaw")
        delta_norm = row.get("refiner_v2_delta_norm")

        source_hist[src] += 1
        if gate:
            all_gate += 1
            source_gate_hist[src] += 1
        if idx >= 0:
            all_pred += 1
            source_pred_hist[src] += 1
            all_selected_sources[src] += 1
        if delta_norm is not None:
            source_delta_norms[src].append(float(delta_norm))
        if cur_xy is not None:
            source_xy[src].append(float(cur_xy))
        if cur_z is not None:
            source_z[src].append(float(cur_z))
        if cur_yaw is not None:
            source_yaw[src].append(float(cur_yaw))

        sel = _extract_selected_trace_metrics(row)
        if sel is not None:
            all_selected_rows += 1
            all_sel_hist[int(sel["selected_candidate_index"])] += 1
            source_sel_hist[src][int(sel["selected_candidate_index"])] += 1
            source_sel_xy[src].append(float(sel["selected_post_xy"]))
            source_sel_z[src].append(float(sel["selected_post_z"]))
            source_sel_yaw[src].append(float(sel["selected_post_yaw"]))
            if sel["selected_geometry_improvement"] is not None:
                source_sel_geom[src].append(float(sel["selected_geometry_improvement"]))
            if sel["selected_xy_improved"] is not None:
                source_sel_xy_improve[src].append(float(sel["selected_xy_improved"]))
            if sel["selected_z_improved"] is not None:
                source_sel_z_improve[src].append(float(sel["selected_z_improved"]))
            if sel["selected_yaw_improved"] is not None:
                source_sel_yaw_improve[src].append(float(sel["selected_yaw_improved"]))

    trace_summary = {
        "rows": len(rows),
        "gate_pass_rows": all_gate,
        "selected_rows": all_pred,
        "gate_pass_rate": all_gate / max(len(rows), 1),
        "selected_rate": all_pred / max(len(rows), 1),
        "candidate_histogram": dict(all_sel_hist.most_common()),
        "source_histogram": dict(source_hist),
        "source_gate_histogram": dict(source_gate_hist),
        "source_selected_histogram": dict(source_pred_hist),
        "source_group_stats": {},
        "predictor_source_rows": int(sum(source_hist[s] for s in PREDICTOR_SOURCES)),
        "predictor_gate_rows": int(sum(source_gate_hist[s] for s in PREDICTOR_SOURCES)),
        "diagnostic_source_rows": int(sum(source_hist[s] for s in DIAGNOSTIC_SOURCES)),
    }

    for src in sorted(source_hist.keys()):
        trace_summary["source_group_stats"][src] = {
            "rows": int(source_hist[src]),
            "gate_rows": int(source_gate_hist[src]),
            "selected_rows": int(source_pred_hist[src]),
            "gate_pass_rate": float(source_gate_hist[src] / max(source_hist[src], 1)),
            "selected_rate": float(source_pred_hist[src] / max(source_hist[src], 1)),
            "delta_norm": _stats(source_delta_norms[src]),
            "cur_xy": _stats(source_xy[src]),
            "cur_z": _stats(source_z[src]),
            "cur_yaw": _stats(source_yaw[src]),
            "selected_candidate_hist": dict(source_sel_hist[src].most_common()),
            "selected_post_xy": _stats(source_sel_xy[src]),
            "selected_post_z": _stats(source_sel_z[src]),
            "selected_post_yaw": _stats(source_sel_yaw[src]),
            "selected_geometry_improvement": _stats(source_sel_geom[src]),
            "selected_xy_improved_rate": float(np.mean(source_sel_xy_improve[src])) if source_sel_xy_improve[src] else 0.0,
            "selected_z_improved_rate": float(np.mean(source_sel_z_improve[src])) if source_sel_z_improve[src] else 0.0,
            "selected_yaw_improved_rate": float(np.mean(source_sel_yaw_improve[src])) if source_sel_yaw_improve[src] else 0.0,
            "selected_bucket_hist": dict(Counter(
                _bucket_name(xy, z, yaw) for xy, z, yaw in zip(
                    source_sel_xy[src], source_sel_z[src], source_sel_yaw[src]
                )
            )),
        }

    predictor_hist = Counter()
    for src in PREDICTOR_SOURCES:
        predictor_hist.update(source_sel_hist[src])

    report = {
        "audit": "v2_predictor_shadow_contract",
        "trace_dir": str(args.trace_dir),
        "trace_summary": trace_summary,
        "verdict": {
            "predictor_loaded_in_runtime": trace_summary["predictor_source_rows"] > 0,
            "predictor_used_at_gate_pass": trace_summary["predictor_gate_rows"] > 0,
            "predictor_not_template_collapsed": len(predictor_hist) > 1,
            "shadow_gate_active_without_takeover": trace_summary["gate_pass_rows"] > 0,
            "ready_for_conservative_shadow_assist": bool(
                trace_summary["predictor_gate_rows"] > 0
                and len(predictor_hist) > 1
                and trace_summary["gate_pass_rate"] > 0.0
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report["verdict"], indent=2, ensure_ascii=False))
    print(f"[audit] report -> {args.output}")


if __name__ == "__main__":
    main()
