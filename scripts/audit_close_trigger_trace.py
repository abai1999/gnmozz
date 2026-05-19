"""
audit_close_trigger_trace.py

Trace-level audit for pre-contact close timing.

It surfaces three patterns directly from a controlled-rollout JSON:
  1. close in the air / before entering the basin
  2. close followed by immediate reopen
  3. close even though the rollout could still approach the target later
"""

import argparse
import json
from pathlib import Path


def find_close_events(curves, close_threshold: float):
    events = []
    prev_raw = float(curves[0].get("exec_gripper_raw", 1.0))
    for row in curves[1:]:
        raw = float(row.get("exec_gripper_raw", 1.0))
        if prev_raw >= close_threshold and raw < close_threshold:
            events.append(int(row["step"]))
        prev_raw = raw
    return events


def first_reopen_after(curves, start_step: int, close_threshold: float, horizon: int):
    end_step = min(len(curves) - 1, int(start_step) + int(horizon))
    for step in range(int(start_step) + 1, end_step + 1):
        raw = float(curves[step].get("exec_gripper_raw", 1.0))
        if raw >= close_threshold:
            return step
    return -1


def min_future_basin(curves, start_step: int, horizon: int):
    end_step = min(len(curves) - 1, int(start_step) + int(horizon))
    best_step = int(start_step)
    best_val = float(curves[int(start_step)]["basin_distance"])
    for step in range(int(start_step), end_step + 1):
        val = float(curves[step]["basin_distance"])
        if val < best_val:
            best_val = val
            best_step = step
    return best_step, best_val


def audit_mode(
    curves,
    close_threshold: float,
    basin_threshold: float,
    air_z_threshold: float,
    air_xy_threshold: float,
    reopen_horizon: int,
    improve_horizon: int,
    improve_eps: float,
):
    close_steps = find_close_events(curves, close_threshold=close_threshold)
    events = []
    for close_step in close_steps:
        row = curves[close_step]
        reopen_step = first_reopen_after(curves, close_step, close_threshold, reopen_horizon)
        future_best_step, future_best_basin = min_future_basin(curves, close_step, improve_horizon)
        basin_now = float(row["basin_distance"])
        basin_xy = float(row["basin_xy"])
        basin_z = float(row["basin_z"])
        close_before_basin = basin_now > basin_threshold
        air_close = basin_z > air_z_threshold or basin_xy > air_xy_threshold
        planner_can_still_approach = future_best_basin < (basin_now - improve_eps)
        events.append(
            {
                "close_step": int(close_step),
                "basin_distance": basin_now,
                "basin_xy": basin_xy,
                "basin_z": basin_z,
                "planner_close_intent": bool(row.get("planner_close_intent", False)),
                "close_before_basin": bool(close_before_basin),
                "air_close": bool(air_close),
                "reopen_step": int(reopen_step),
                "reopen_within_horizon": bool(reopen_step >= 0),
                "future_best_step": int(future_best_step),
                "future_best_basin": float(future_best_basin),
                "planner_can_still_approach": bool(planner_can_still_approach),
            }
        )
    summary = {
        "num_close_events": len(events),
        "close_before_basin_count": sum(int(e["close_before_basin"]) for e in events),
        "air_close_count": sum(int(e["air_close"]) for e in events),
        "reopen_within_horizon_count": sum(int(e["reopen_within_horizon"]) for e in events),
        "planner_can_still_approach_count": sum(int(e["planner_can_still_approach"]) for e in events),
        "close_before_basin_and_reopen_count": sum(
            int(e["close_before_basin"] and e["reopen_within_horizon"]) for e in events
        ),
        "close_before_basin_and_future_improve_count": sum(
            int(e["close_before_basin"] and e["planner_can_still_approach"]) for e in events
        ),
    }
    return {"summary": summary, "events": events}


def main():
    parser = argparse.ArgumentParser(description="Audit close-before-basin and reopen patterns from rollout trace JSON.")
    parser.add_argument("--trace_json", type=str, required=True)
    parser.add_argument("--close_threshold", type=float, default=0.5)
    parser.add_argument("--basin_threshold", type=float, default=1.0)
    parser.add_argument("--air_z_threshold", type=float, default=0.01)
    parser.add_argument("--air_xy_threshold", type=float, default=0.01)
    parser.add_argument("--reopen_horizon", type=int, default=8)
    parser.add_argument("--improve_horizon", type=int, default=20)
    parser.add_argument("--improve_eps", type=float, default=0.25)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    with Path(args.trace_json).open("r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for mode_name, mode_data in data.items():
        curves = mode_data.get("curves", [])
        out[mode_name] = audit_mode(
            curves,
            close_threshold=args.close_threshold,
            basin_threshold=args.basin_threshold,
            air_z_threshold=args.air_z_threshold,
            air_xy_threshold=args.air_xy_threshold,
            reopen_horizon=args.reopen_horizon,
            improve_horizon=args.improve_horizon,
            improve_eps=args.improve_eps,
        )

    print(json.dumps(out, indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
