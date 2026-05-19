#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "gripper_traces").is_dir():
        path = path / "gripper_traces"
    files = sorted(path.glob("*_gripper_trace.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    return files


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_bool(row: dict, key: str) -> bool:
    return bool(row.get(key, False))


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        out = float(row.get(key, default))
    except Exception:
        return default
    return out if math.isfinite(out) else default


def first_step(rows: list[dict], pred) -> int:
    for i, row in enumerate(rows):
        if pred(row):
            return int(row.get("step", i))
    return -1


def count(rows: list[dict], pred) -> int:
    return sum(1 for row in rows if pred(row))


def mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else math.nan


def min_finite(vals) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(min(vals)) if vals else math.nan


def episode_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_gripper_trace"):
        stem = stem[: -len("_gripper_trace")]
    return stem


def classify_episode(ep: dict) -> str:
    if ep["planner_close_frames"] <= 0:
        return "no-close-intent"
    if ep["close_veto_pass_frames"] > 0:
        return "actual-close-but-task-fail"
    if ep["z_block_frames"] >= max(ep["xy_block_frames"], ep["yaw_block_frames"], ep["shadow_blocked_frames"], 1):
        return "close-intent-but-z-block"
    if ep["xy_block_frames"] >= max(ep["z_block_frames"], ep["yaw_block_frames"], ep["shadow_blocked_frames"], 1):
        return "close-intent-but-xy-block"
    if ep["shadow_blocked_frames"] > 0:
        return "shadow-handoff-blocked"
    if ep["b2_worse_frames"] > ep["b2_better_frames"] and ep["b2_worse_frames"] > 0:
        return "B2-negative-regret"
    return "close-intent-other-block"


def summarize_episode(path: Path, rows: list[dict]) -> dict:
    close_states = collections.Counter(str(r.get("refiner_close_state", "missing")) for r in rows)
    close_actions = collections.Counter(str(r.get("refiner_close_action_decision", "missing")) for r in rows)
    blocked_reasons = collections.Counter(str(r.get("refiner_close_blocked_reason", "none")) for r in rows)
    transitions = collections.Counter()
    prev = None
    for r in rows:
        cur = (
            str(r.get("refiner_close_state", "missing")),
            str(r.get("refiner_close_action_decision", "missing")),
            str(r.get("refiner_close_blocked_reason", "none")),
        )
        if prev is not None and cur != prev:
            transitions[f"{prev[0]}|{prev[1]}|{prev[2]} -> {cur[0]}|{cur[1]}|{cur[2]}"] += 1
        prev = cur
    b2_deltas = [as_float(r, "b2_candidate_shadow_regret_delta") for r in rows]
    b2_gate = [r for r in rows if as_bool(r, "b2_candidate_shadow_gate_open")]
    ep = {
        "episode_trace": path.name,
        "episode_id": episode_id_from_path(path),
        "frames": len(rows),
        "first_planner_close_step": first_step(rows, lambda r: as_bool(r, "refiner_alignment_planner_close_intent")),
        "first_handoff_pred_ready_step": first_step(rows, lambda r: as_bool(r, "refiner_close_handoff_ready_pred") or as_bool(r, "handoff_ready_pred")),
        "first_handoff_applied_ready_step": first_step(rows, lambda r: as_bool(r, "refiner_close_handoff_ready_applied") or as_bool(r, "runtime_handoff_ready_applied") or as_bool(r, "refiner_current_handoff_ready")),
        "first_runtime_geometry_ready_step": first_step(rows, lambda r: as_bool(r, "refiner_close_runtime_geometry_ready")),
        "first_close_veto_pass_step": first_step(rows, lambda r: str(r.get("refiner_close_action_decision", "")) in {"pass_close", "latch_release_close", "bounded_auto_close", "force_close_after_b2"}),
        "first_shadow_auto_close_candidate_step": first_step(rows, lambda r: as_bool(r, "refiner_close_intent_shadow_would_auto_close")),
        "planner_close_frames": count(rows, lambda r: as_bool(r, "refiner_alignment_planner_close_intent")),
        "handoff_pred_ready_frames": count(rows, lambda r: as_bool(r, "refiner_close_handoff_ready_pred") or as_bool(r, "handoff_ready_pred")),
        "handoff_applied_ready_frames": count(rows, lambda r: as_bool(r, "refiner_close_handoff_ready_applied") or as_bool(r, "runtime_handoff_ready_applied") or as_bool(r, "refiner_current_handoff_ready")),
        "runtime_geometry_ready_frames": count(rows, lambda r: as_bool(r, "refiner_close_runtime_geometry_ready")),
        "close_veto_pass_frames": count(rows, lambda r: str(r.get("refiner_close_action_decision", "")) in {"pass_close", "latch_release_close", "bounded_auto_close", "force_close_after_b2"}),
        "close_veto_block_frames": count(rows, lambda r: as_bool(r, "refiner_current_close_veto_blocked") or str(r.get("refiner_close_action_decision", "")) == "block_close"),
        "shadow_auto_close_candidate_frames": count(rows, lambda r: as_bool(r, "refiner_close_intent_shadow_would_auto_close")),
        "z_block_frames": count(rows, lambda r: str(r.get("refiner_close_blocked_reason", "")) == "z"),
        "xy_block_frames": count(rows, lambda r: str(r.get("refiner_close_blocked_reason", "")) == "xy"),
        "yaw_block_frames": count(rows, lambda r: str(r.get("refiner_close_blocked_reason", "")) == "yaw"),
        "support_outer_block_frames": count(rows, lambda r: str(r.get("refiner_close_blocked_reason", "")) == "support_outer"),
        "shadow_blocked_frames": count(rows, lambda r: str(r.get("refiner_close_blocked_reason", "")) == "shadow_blocked"),
        "open_command_frames": count(rows, lambda r: str(r.get("refiner_close_state", "")) == "open_command"),
        "fallback_used_frames": count(rows, lambda r: as_bool(r, "refiner_close_fallback_used")),
        "b2_gate_open_frames": len(b2_gate),
        "b2_apply_frames": count(rows, lambda r: as_bool(r, "b2_candidate_bounded_applied") or int(r.get("b2_candidate_shadow_mode", -1) or -1) == 2),
        "b2_better_frames": count(rows, lambda r: as_float(r, "b2_candidate_shadow_regret_delta") > 1e-6),
        "b2_worse_frames": count(rows, lambda r: as_float(r, "b2_candidate_shadow_regret_delta") < -1e-6),
        "b2_regret_delta_mean": mean(b2_deltas),
        "close_state_counts": dict(close_states),
        "close_action_counts": dict(close_actions),
        "blocked_reason_counts": dict(blocked_reasons),
        "top_close_transitions": dict(transitions.most_common(20)),
        "xy_error_min": min_finite(as_float(r, "refiner_close_xy_error") for r in rows),
        "abs_z_error_min": min_finite(as_float(r, "refiner_close_abs_z_error") for r in rows),
        "yaw_error_min": min_finite(as_float(r, "refiner_close_yaw_error") for r in rows),
        "xy_threshold_mean": mean([as_float(r, "refiner_close_xy_threshold") for r in rows]),
        "abs_z_threshold_mean": mean([as_float(r, "refiner_close_abs_z_threshold") for r in rows]),
        "yaw_threshold_mean": mean([as_float(r, "refiner_close_yaw_threshold") for r in rows]),
    }
    ep["bucket"] = classify_episode(ep)
    return ep


def combine(episodes: list[dict]) -> dict:
    bucket_counts = collections.Counter(ep["bucket"] for ep in episodes)
    total = {
        "episode_count": len(episodes),
        "frames": sum(ep["frames"] for ep in episodes),
        "bucket_counts": dict(bucket_counts),
    }
    for key in (
        "planner_close_frames",
        "handoff_pred_ready_frames",
        "handoff_applied_ready_frames",
        "runtime_geometry_ready_frames",
        "close_veto_pass_frames",
        "close_veto_block_frames",
        "shadow_auto_close_candidate_frames",
        "z_block_frames",
        "xy_block_frames",
        "yaw_block_frames",
        "support_outer_block_frames",
        "shadow_blocked_frames",
        "open_command_frames",
        "fallback_used_frames",
        "b2_gate_open_frames",
        "b2_apply_frames",
        "b2_better_frames",
        "b2_worse_frames",
    ):
        total[key] = sum(int(ep[key]) for ep in episodes)
    total["b2_regret_delta_mean_episode_weighted"] = mean([ep["b2_regret_delta_mean"] for ep in episodes])
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    args = ap.parse_args()
    files = find_trace_files(args.trace_dir)
    episodes = [summarize_episode(path, load_jsonl(path)) for path in files]
    report = {
        "trace_dir": str(args.trace_dir),
        "summary": combine(episodes),
        "episodes": episodes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
