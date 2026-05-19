#!/usr/bin/env python3
"""Summarize v4 shadow/apply traces with continuous-error diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(trace_path: Path):
    return [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _slope(values):
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    x_mean = (len(vals) - 1) / 2.0
    y_mean = sum(vals) / len(vals)
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(len(vals)))
    return num / den if den > 0 else None


def _downstep_rate(values, eps=1e-8):
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return None
    return sum(1 for prev, cur in zip(vals[:-1], vals[1:]) if cur < prev - eps) / (len(vals) - 1)


def _count_micro_like(rows):
    count = 0
    for row in rows:
        cur_xy = row.get("refiner_alignment_v4_shadow_cur_xy")
        cur_z = row.get("refiner_alignment_v4_shadow_cur_z")
        cur_yaw = row.get("refiner_alignment_v4_shadow_cur_yaw")
        if cur_xy is None or cur_z is None or cur_yaw is None:
            continue
        if float(cur_xy) < 0.015 and float(cur_z) < 0.03 and float(cur_yaw) < 0.12:
            count += 1
    return count


def summarize(trace_path: Path):
    rows = _load_rows(trace_path)
    cur_xy = [row.get("refiner_alignment_v4_shadow_cur_xy") for row in rows]
    cur_yaw = [row.get("refiner_alignment_v4_shadow_cur_yaw") for row in rows]
    post_xy = [row.get("refiner_alignment_v4_shadow_post_xy") for row in rows]
    post_yaw = [row.get("refiner_alignment_v4_shadow_post_yaw") for row in rows]
    stage_bucket = [row.get("refiner_alignment_v4_shadow_stage_bucket") for row in rows]
    out = {
        "rows": len(rows),
        "shadow_active_rate": sum(bool(r.get("refiner_alignment_v4_shadow_active")) for r in rows) / len(rows) if rows else 0.0,
        "apply_rate": sum(bool(r.get("refiner_alignment_v4_apply_applied")) for r in rows) / len(rows) if rows else 0.0,
        "shadow_xy_improved_rate": sum(bool(r.get("refiner_alignment_v4_shadow_xy_improved")) for r in rows) / len(rows) if rows else 0.0,
        "shadow_yaw_improved_rate": sum(bool(r.get("refiner_alignment_v4_shadow_yaw_improved")) for r in rows) / len(rows) if rows else 0.0,
        "shadow_all_improved_rate": sum(bool(r.get("refiner_alignment_v4_shadow_all_improved")) for r in rows) / len(rows) if rows else 0.0,
        "mean_cur_xy": _mean(cur_xy),
        "mean_cur_yaw": _mean(cur_yaw),
        "mean_post_xy": _mean(post_xy),
        "mean_post_yaw": _mean(post_yaw),
        "cur_xy_downstep_rate": _downstep_rate(cur_xy),
        "cur_yaw_downstep_rate": _downstep_rate(cur_yaw),
        "cur_xy_slope": _slope(cur_xy),
        "cur_yaw_slope": _slope(cur_yaw),
        "micro_contact_refine_count": sum(1 for b in stage_bucket if b == "micro_contact_refine"),
        "near_alignment_count": sum(1 for b in stage_bucket if b == "near_alignment"),
        "mid_approach_assist_count": sum(1 for b in stage_bucket if b == "mid_approach_assist"),
        "far_coarse_approach_count": sum(1 for b in stage_bucket if b == "far_coarse_approach"),
        "micro_like_count": _count_micro_like(rows),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()
    out = summarize(Path(args.trace))
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
