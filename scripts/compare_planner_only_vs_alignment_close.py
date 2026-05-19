#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def as_float(v, default=math.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def first_planner_close(rows: list[dict], threshold: float = 0.5) -> dict | None:
    for row in rows:
        grip = as_float(row.get("base_gripper_raw"))
        if math.isfinite(grip) and grip <= threshold:
            return row
    return None


def summarize_episode(ep: str, planner_rows: list[dict], align_rows: list[dict], threshold: float) -> dict:
    planner_close_frames = sum(
        1 for r in planner_rows if math.isfinite(as_float(r.get("base_gripper_raw"))) and as_float(r.get("base_gripper_raw")) <= threshold
    )
    align_close_intent_frames = sum(bool(r.get("refiner_alignment_planner_close_intent", False)) for r in align_rows)
    align_near_target_frames = sum(bool(r.get("refiner_alignment_near_target", False)) for r in align_rows)
    align_close_like_frames = sum(
        bool(r.get("refiner_alignment_near_target", False))
        and str(r.get("refiner_current_handoff_target_role", "none")) == "pregrasp_close"
        and str(r.get("refiner_target_provider_source", "")) == "learned_target_predictor__canonical_close_orientation_contract"
        for r in align_rows
    )

    first_close = first_planner_close(planner_rows, threshold)
    at_same_step = None
    if first_close is not None:
        step = min(int(first_close.get("step", 0)), len(align_rows) - 1)
        ar = align_rows[step]
        at_same_step = {
            "step": int(ar.get("step", step)),
            "near_target": bool(ar.get("refiner_alignment_near_target", False)),
            "planner_close_intent": bool(ar.get("refiner_alignment_planner_close_intent", False)),
            "base_gripper_raw": as_float(ar.get("base_gripper_raw")),
            "close_state": str(ar.get("refiner_close_state", "missing")),
            "blocked_reason": str(ar.get("refiner_close_blocked_reason", "none")),
            "target_role": str(ar.get("refiner_current_handoff_target_role", "none")),
            "provider_source": str(ar.get("refiner_target_provider_source", "none")),
        }

    return {
        "episode_id": ep,
        "planner_only": {
            "close_frames": int(planner_close_frames),
            "min_base_gripper_raw": min(as_float(r.get("base_gripper_raw")) for r in planner_rows if r.get("base_gripper_raw") is not None),
            "first_close": None if first_close is None else {
                "step": int(first_close.get("step", -1)),
                "phase_before": int(first_close.get("phase_before", -1)),
                "phase_after": int(first_close.get("phase_after", -1)),
                "depth_proximity": as_float(first_close.get("depth_proximity")),
                "base_gripper_raw": as_float(first_close.get("base_gripper_raw")),
            },
        },
        "alignment": {
            "planner_close_intent_frames": int(align_close_intent_frames),
            "near_target_frames": int(align_near_target_frames),
            "close_like_frames": int(align_close_like_frames),
            "min_base_gripper_raw": min(as_float(r.get("base_gripper_raw")) for r in align_rows if r.get("base_gripper_raw") is not None),
            "teacher_ready_frames": int(sum(bool(r.get("teacher_truth_handoff_ready", False)) for r in align_rows)),
            "runtime_ready_frames": int(sum(bool(r.get("refiner_current_handoff_ready", False)) for r in align_rows)),
        },
        "alignment_at_planner_close_step": at_same_step,
    }


def build_takeaways(episodes: list[dict]) -> list[str]:
    planner_close_eps = sum(ep["planner_only"]["close_frames"] > 0 for ep in episodes)
    suppressed = sum(
        ep["planner_only"]["close_frames"] > 0
        and ep["alignment"]["planner_close_intent_frames"] == 0
        and ep["alignment"]["close_like_frames"] > 0
        for ep in episodes
    )
    notes = []
    if suppressed > 0:
        notes.append(
            "There are multiple episodes where planner-only emits close on the raw trajectory, but after alignment-driven trajectory changes "
            "the planner no longer emits close even while the trace remains near-target / pregrasp-close."
        )
    if planner_close_eps > 0:
        notes.append(
            "This points to an interface gap: alignment can change the state distribution seen by the planner gripper head, "
            "but close-intent is still sourced from planner raw gripper output rather than recomputed from corrected geometry."
        )
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner_trace_dir", type=Path, required=True)
    ap.add_argument("--alignment_trace_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--close_threshold", type=float, default=0.5)
    args = ap.parse_args()

    planner_dir = args.planner_trace_dir / "gripper_traces" if (args.planner_trace_dir / "gripper_traces").is_dir() else args.planner_trace_dir
    align_dir = args.alignment_trace_dir / "gripper_traces" if (args.alignment_trace_dir / "gripper_traces").is_dir() else args.alignment_trace_dir

    episodes = []
    for p in sorted(planner_dir.glob("ep*_gripper_trace.jsonl")):
        ep = p.stem.replace("_gripper_trace", "")
        apath = align_dir / f"{ep}_gripper_trace.jsonl"
        if not apath.exists():
            continue
        episodes.append(
            summarize_episode(ep, load_jsonl(p), load_jsonl(apath), args.close_threshold)
        )

    report = {
        "planner_trace_dir": str(args.planner_trace_dir),
        "alignment_trace_dir": str(args.alignment_trace_dir),
        "close_threshold": float(args.close_threshold),
        "summary": {
            "episode_count": len(episodes),
            "planner_close_episode_count": int(sum(ep["planner_only"]["close_frames"] > 0 for ep in episodes)),
            "alignment_close_intent_episode_count": int(sum(ep["alignment"]["planner_close_intent_frames"] > 0 for ep in episodes)),
            "suppressed_after_alignment_episode_count": int(
                sum(
                    ep["planner_only"]["close_frames"] > 0
                    and ep["alignment"]["planner_close_intent_frames"] == 0
                    and ep["alignment"]["close_like_frames"] > 0
                    for ep in episodes
                )
            ),
        },
        "takeaways": build_takeaways(episodes),
        "episodes": episodes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
