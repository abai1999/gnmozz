#!/usr/bin/env python3
"""Audit the conservative predictor-driven near/micro apply path."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _stats(values: list[float]) -> dict:
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


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(r.get(key, False)) for r in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_traces(args.trace_dir)
    if not rows:
        raise SystemExit(f"no trace rows found under {args.trace_dir}")

    apply_rows = [r for r in rows if bool(r.get("refiner_v2_predictor_micro_assist_applied", False))]
    gate_rows = [r for r in rows if bool(r.get("refiner_v2_apply_gate_pass", False))]

    report = {
        "rows": len(rows),
        "apply_rows": len(apply_rows),
        "gate_rows": len(gate_rows),
        "predictor_micro_assist_enabled_rate": _rate(rows, "refiner_v2_predictor_micro_assist_enabled"),
        "apply_rate": _rate(rows, "refiner_v2_predictor_micro_assist_applied"),
        "gate_pass_rate": _rate(rows, "refiner_v2_apply_gate_pass"),
        "planner_close_intent_rate": _rate(rows, "refiner_alignment_planner_close_intent"),
        "near_zone_gate_pass_rate": _rate(rows, "refiner_alignment_near_zone_gate_pass"),
        "alignment_takeover_rate": _rate(rows, "refiner_alignment_takeover_active"),
        "zone_histogram": dict(Counter(str(r.get("zone_state", "unknown")) for r in rows)),
        "apply_block_reason_histogram": dict(
            Counter(str(r.get("refiner_v2_apply_block_reason", "unknown")) for r in rows)
        ),
        "predictor_micro_assist_block_reason_histogram": dict(
            Counter(str(r.get("refiner_v2_predictor_micro_assist_block_reason", "unknown")) for r in rows)
        ),
        "delta_source_histogram": dict(Counter(str(r.get("refiner_v2_delta_source", "unknown")) for r in rows)),
        "selected_candidate_histogram": dict(
            Counter(int(r.get("refiner_v2_selected_candidate_index", -1) or -1) for r in rows if int(r.get("refiner_v2_selected_candidate_index", -1) or -1) >= 0)
        ),
        "assist_local_pos_norm": _stats([float(np.linalg.norm(np.asarray(r.get("refiner_v2_apply_local_delta"), dtype=np.float32)[:3])) for r in apply_rows if r.get("refiner_v2_apply_local_delta") is not None]),
        "assist_local_rot_norm": _stats([float(np.linalg.norm(np.asarray(r.get("refiner_v2_apply_local_delta"), dtype=np.float32)[3:6])) for r in apply_rows if r.get("refiner_v2_apply_local_delta") is not None]),
        "assist_world_pos_norm": _stats([float(np.linalg.norm(np.asarray(r.get("refiner_v2_apply_world_delta"), dtype=np.float32)[:3])) for r in apply_rows if r.get("refiner_v2_apply_world_delta") is not None]),
        "assist_world_rot_norm": _stats([float(np.linalg.norm(np.asarray(r.get("refiner_v2_apply_world_delta"), dtype=np.float32)[3:6])) for r in apply_rows if r.get("refiner_v2_apply_world_delta") is not None]),
        "current_v2_selected_post_xy": _stats([float(r.get("refiner_v2_selected_post_xy", 0.0) or 0.0) for r in gate_rows if r.get("refiner_v2_selected_post_xy") is not None]),
        "current_v2_selected_post_z": _stats([float(r.get("refiner_v2_selected_post_z", 0.0) or 0.0) for r in gate_rows if r.get("refiner_v2_selected_post_z") is not None]),
        "current_v2_selected_post_yaw": _stats([float(r.get("refiner_v2_selected_post_yaw", 0.0) or 0.0) for r in gate_rows if r.get("refiner_v2_selected_post_yaw") is not None]),
        "workspace_violation_count": int(sum(int(r.get("workspace_violation_count", 0) or 0) for r in rows)),
        "invalid_action_count": int(sum(int(r.get("invalid_action_count", 0) or 0) for r in rows)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
