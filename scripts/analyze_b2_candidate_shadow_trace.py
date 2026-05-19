#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from statistics import median


def as_bool(row: dict, key: str) -> bool:
    return bool(row.get(key, False))


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return default


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def finite(values) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else math.nan


def p50(vals: list[float]) -> float:
    return float(median(vals)) if vals else math.nan


def rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


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


def summarize_bucket(rows: list[dict], key: str) -> dict:
    bucket_rows = [r for r in rows if as_bool(r, key)]
    pred = finite(as_float(r, "b2_candidate_shadow_pred_regret") for r in bucket_rows)
    base = finite(as_float(r, "b2_candidate_shadow_baseline_regret") for r in bucket_rows)
    delta = finite(as_float(r, "b2_candidate_shadow_regret_delta") for r in bucket_rows)
    return {
        "frames": len(bucket_rows),
        "pred_regret_mean": mean(pred),
        "pred_regret_p50": p50(pred),
        "baseline_regret_mean": mean(base),
        "regret_delta_mean_baseline_minus_pred": mean(delta),
    }


def summarize_episode(path: Path, rows: list[dict]) -> dict:
    gate = [r for r in rows if as_bool(r, "b2_candidate_shadow_gate_open")]
    cost_valid = [r for r in gate if math.isfinite(as_float(r, "b2_candidate_shadow_regret_delta"))]
    changed = [r for r in gate if as_bool(r, "b2_candidate_shadow_changed")]
    better = [r for r in cost_valid if as_float(r, "b2_candidate_shadow_regret_delta") > 1e-6]
    worse = [r for r in cost_valid if as_float(r, "b2_candidate_shadow_regret_delta") < -1e-6]
    tie = [r for r in cost_valid if abs(as_float(r, "b2_candidate_shadow_regret_delta")) <= 1e-6]
    deltas = finite(as_float(r, "b2_candidate_shadow_regret_delta") for r in cost_valid)
    pred_regrets = finite(as_float(r, "b2_candidate_shadow_pred_regret") for r in cost_valid)
    base_regrets = finite(as_float(r, "b2_candidate_shadow_baseline_regret") for r in cost_valid)
    conf = finite(as_float(r, "b2_candidate_shadow_mode_confidence") for r in gate)
    margin = finite(as_float(r, "b2_candidate_shadow_mode_margin") for r in gate)
    scope = finite(as_float(r, "b2_candidate_shadow_runtime_scope_size") for r in gate)
    small_scope = finite(as_float(r, "b2_candidate_shadow_small_yaw_scope_size") for r in gate)
    large_scope = finite(as_float(r, "b2_candidate_shadow_large_yaw_scope_size") for r in gate)
    probe_count = finite(as_float(r, "b2_candidate_shadow_probe_count") for r in gate)
    modes = collections.Counter(as_int(r, "b2_candidate_shadow_mode", -1) for r in gate)
    apply_mode = max([m for m in modes if m >= 0], default=-1)
    keep_count = int(modes.get(0, 0))
    apply_count = int(modes.get(apply_mode, 0)) if apply_mode > 0 else 0
    ambiguous_count = int(sum(c for m, c in modes.items() if m not in (0, apply_mode)))

    return {
        "episode_trace": path.name,
        "frames": len(rows),
        "gate_open_frames": len(gate),
        "close_neighborhood_frames": sum(as_bool(r, "b2_candidate_shadow_close_neighborhood") for r in rows),
        "nearish_runtime_frames": sum(as_bool(r, "b2_candidate_shadow_nearish_runtime") for r in rows),
        "changed_frames": len(changed),
        "keep_baseline_forced_frames": sum(as_bool(r, "b2_candidate_shadow_keep_baseline_forced") for r in gate),
        "change_rate_gate": rate(len(changed), len(gate)),
        "cost_valid_frames": len(cost_valid),
        "pred_better_than_baseline_count": len(better),
        "pred_worse_than_baseline_count": len(worse),
        "pred_tie_baseline_count": len(tie),
        "pred_better_than_baseline_rate": rate(len(better), len(cost_valid)),
        "pred_worse_than_baseline_rate": rate(len(worse), len(cost_valid)),
        "regret_delta_mean_baseline_minus_pred": mean(deltas),
        "regret_delta_p50_baseline_minus_pred": p50(deltas),
        "pred_regret_mean": mean(pred_regrets),
        "pred_regret_p50": p50(pred_regrets),
        "baseline_regret_mean": mean(base_regrets),
        "baseline_regret_p50": p50(base_regrets),
        "mode_counts": {str(int(k)): int(v) for k, v in sorted(modes.items())},
        "mode_keep_count": keep_count,
        "mode_apply_count": apply_count,
        "mode_ambiguous_count": ambiguous_count,
        "mode_keep_rate_gate": rate(keep_count, len(gate)),
        "mode_apply_rate_gate": rate(apply_count, len(gate)),
        "mode_confidence_mean": mean(conf),
        "mode_confidence_p50": p50(conf),
        "mode_margin_mean": mean(margin),
        "mode_margin_p50": p50(margin),
        "runtime_scope_size_mean": mean(scope),
        "small_yaw_scope_size_mean": mean(small_scope),
        "large_yaw_scope_size_mean": mean(large_scope),
        "probe_count_mean": mean(probe_count),
        "yaw_needed": summarize_bucket(gate, "b2_candidate_shadow_yaw_needed"),
        "yaw_keep": summarize_bucket(gate, "b2_candidate_shadow_yaw_keep"),
        "teacher_ready": summarize_bucket(gate, "b2_candidate_shadow_teacher_ready"),
        "xy_block": summarize_bucket(gate, "b2_candidate_shadow_xy_block"),
    }


def combine(episodes: list[dict]) -> dict:
    totals = {
        "episodes": len(episodes),
        "frames": sum(e["frames"] for e in episodes),
        "gate_open_frames": sum(e["gate_open_frames"] for e in episodes),
        "close_neighborhood_frames": sum(e["close_neighborhood_frames"] for e in episodes),
        "nearish_runtime_frames": sum(e["nearish_runtime_frames"] for e in episodes),
        "changed_frames": sum(e["changed_frames"] for e in episodes),
        "keep_baseline_forced_frames": sum(e["keep_baseline_forced_frames"] for e in episodes),
        "cost_valid_frames": sum(e["cost_valid_frames"] for e in episodes),
        "pred_better_than_baseline_count": sum(e["pred_better_than_baseline_count"] for e in episodes),
        "pred_worse_than_baseline_count": sum(e["pred_worse_than_baseline_count"] for e in episodes),
        "pred_tie_baseline_count": sum(e["pred_tie_baseline_count"] for e in episodes),
        "mode_keep_count": sum(e["mode_keep_count"] for e in episodes),
        "mode_apply_count": sum(e["mode_apply_count"] for e in episodes),
        "mode_ambiguous_count": sum(e["mode_ambiguous_count"] for e in episodes),
    }
    for rate_key, num_key, den_key in (
        ("change_rate_gate", "changed_frames", "gate_open_frames"),
        ("pred_better_than_baseline_rate", "pred_better_than_baseline_count", "cost_valid_frames"),
        ("pred_worse_than_baseline_rate", "pred_worse_than_baseline_count", "cost_valid_frames"),
        ("mode_keep_rate_gate", "mode_keep_count", "gate_open_frames"),
        ("mode_apply_rate_gate", "mode_apply_count", "gate_open_frames"),
        ("keep_baseline_forced_rate_gate", "keep_baseline_forced_frames", "gate_open_frames"),
    ):
        totals[rate_key] = rate(totals[num_key], totals[den_key])
    weighted_by_cost = (
        "regret_delta_mean_baseline_minus_pred",
        "pred_regret_mean",
        "baseline_regret_mean",
    )
    weighted_by_gate = (
        "mode_confidence_mean",
        "mode_margin_mean",
        "runtime_scope_size_mean",
        "small_yaw_scope_size_mean",
        "large_yaw_scope_size_mean",
        "probe_count_mean",
    )
    for key in weighted_by_cost:
        vals = [(e[key], e["cost_valid_frames"]) for e in episodes if math.isfinite(float(e[key]))]
        den = sum(w for _, w in vals)
        totals[key] = float(sum(float(v) * int(w) for v, w in vals) / den) if den else math.nan
    for key in weighted_by_gate:
        vals = [(e[key], e["gate_open_frames"]) for e in episodes if math.isfinite(float(e[key]))]
        den = sum(w for _, w in vals)
        totals[key] = float(sum(float(v) * int(w) for v, w in vals) / den) if den else math.nan
    for bucket in ("yaw_needed", "yaw_keep", "teacher_ready", "xy_block"):
        frames = sum(e[bucket]["frames"] for e in episodes)

        def bucket_weighted(metric: str) -> float:
            vals = [
                (e[bucket][metric], e[bucket]["frames"])
                for e in episodes
                if e[bucket]["frames"] > 0 and math.isfinite(float(e[bucket][metric]))
            ]
            den = sum(w for _, w in vals)
            return float(sum(float(v) * int(w) for v, w in vals) / den) if den else math.nan

        totals[bucket] = {
            "frames": frames,
            "pred_regret_mean": bucket_weighted("pred_regret_mean"),
            "baseline_regret_mean": bucket_weighted("baseline_regret_mean"),
            "regret_delta_mean_baseline_minus_pred": bucket_weighted("regret_delta_mean_baseline_minus_pred"),
        }
    return totals


def gate_decision(summary: dict) -> dict:
    blocked: list[str] = []
    if summary["gate_open_frames"] <= 0:
        blocked.append("no_b2_shadow_gate_open_frames")
    if summary["cost_valid_frames"] <= 0:
        blocked.append("no_teacher_cost_valid_shadow_frames")
    if summary["mode_keep_count"] <= 0:
        blocked.append("mode_keep_absent_in_shadow")
    if summary["mode_apply_count"] <= 0:
        blocked.append("mode_apply_absent_in_shadow")
    if summary["cost_valid_frames"] > 0 and summary["pred_worse_than_baseline_rate"] > 0.40:
        blocked.append("pred_worse_than_baseline_rate_high")
    if math.isfinite(summary["regret_delta_mean_baseline_minus_pred"]) and summary["regret_delta_mean_baseline_minus_pred"] < -0.05:
        blocked.append("mean_regret_delta_negative")
    for bucket, limit in (
        ("yaw_keep", 8.0),
        ("yaw_needed", 8.0),
        ("teacher_ready", 2.5),
        ("xy_block", 5.0),
    ):
        b = summary[bucket]
        if b["frames"] > 0 and math.isfinite(b["pred_regret_mean"]) and b["pred_regret_mean"] > limit:
            blocked.append(f"{bucket}_pred_regret_high")
    return {
        "runtime_shadow_allowed": not blocked,
        "bounded_candidate": False,
        "blocked_reasons": blocked,
        "bounded_candidate_reason": "shadow_only_first_run_requires_manual_review",
    }


def parse_focus_episodes(text: str) -> set[str]:
    return {x.strip().zfill(3) for x in text.split(",") if x.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--focus_output_json", type=Path, default=None)
    parser.add_argument("--gate_output_json", type=Path, default=None)
    parser.add_argument("--focus_episodes", type=str, default="18,34,45")
    args = parser.parse_args()

    files = find_trace_files(args.trace_dir)
    episodes = [summarize_episode(path, load_jsonl(path)) for path in files]
    summary = combine(episodes)
    decision = gate_decision(summary)
    focus_ids = parse_focus_episodes(args.focus_episodes)
    focus = [
        e
        for e in episodes
        if any(token in e["episode_trace"] for token in focus_ids)
        or any(f"episode_{int(token)}" in e["episode_trace"] for token in focus_ids)
    ]
    report = {
        "trace_dir": str(args.trace_dir),
        "num_trace_files": len(files),
        "summary": summary,
        "episodes": episodes,
        "focus_episode_diagnostics": focus,
        "gate_decision": decision,
    }
    text = json.dumps(report, indent=2, allow_nan=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    else:
        print(text)
    if args.focus_output_json:
        args.focus_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.focus_output_json.write_text(json.dumps({"focus_episode_diagnostics": focus}, indent=2, allow_nan=True) + "\n")
    if args.gate_output_json:
        args.gate_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.gate_output_json.write_text(json.dumps(decision, indent=2, allow_nan=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
