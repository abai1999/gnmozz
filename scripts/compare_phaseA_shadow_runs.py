#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def safe_get(d: dict, *keys, default=math.nan):
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    if cur is None:
        return default
    return cur


def summarize_run(path: Path) -> dict:
    readiness = load_json(path / "close_readiness_trace_report.json")
    target = load_json(path / "runtime_target_frame_audit.json")
    align = load_json(path / "phasea_ready_runtime_alignment_audit.json")
    close_chain = load_json(path / "close_chain_bucket_report.json") if (path / "close_chain_bucket_report.json").exists() else {"summary": {"bucket_counts": {}}}
    return {
        "run_dir": str(path),
        "pred_ready_prob_p99": safe_get(align, "runtime_trace_audit", "pred_ready_prob", "p99"),
        "ready_prob_peak_max": safe_get(readiness, "summary", "ready_prob_peak_max"),
        "runtime_handoff_ready_frames": safe_get(readiness, "summary", "runtime_handoff_ready_frames"),
        "teacher_runtime_handoff_ready_overlap_frames": safe_get(readiness, "summary", "teacher_runtime_handoff_ready_overlap_frames"),
        "false_close_apply_rate": safe_get(readiness, "summary", "false_close_apply_rate"),
        "close_veto_pass_frames": safe_get(close_chain, "summary", "close_veto_pass_frames", default=0.0),
        "close_veto_block_frames": safe_get(close_chain, "summary", "close_veto_block_frames", default=0.0),
        "shadow_handoff_blocked_episode_count": int(
            safe_get(close_chain, "summary", "bucket_counts", default={}).get("shadow-handoff-blocked", 0)
        ) if isinstance(safe_get(close_chain, "summary", "bucket_counts", default={}), dict) else 0,
        "runtime_minus_teacher_yaw_mean_episode_weighted": safe_get(target, "summary", "runtime_minus_teacher_yaw_mean_episode_weighted"),
        "runtime_over_threshold_xy_mean_episode_weighted": safe_get(target, "summary", "runtime_over_threshold_xy_mean_episode_weighted"),
        "runtime_over_threshold_z_mean_episode_weighted": safe_get(target, "summary", "runtime_over_threshold_z_mean_episode_weighted"),
        "runtime_over_threshold_yaw_mean_episode_weighted": safe_get(target, "summary", "runtime_over_threshold_yaw_mean_episode_weighted"),
        "privileged_runtime_frames": safe_get(target, "summary", "privileged_runtime_frames"),
    }


def delta(a: float, b: float) -> float | None:
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        return None
    return float(a - b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=Path, required=True)
    ap.add_argument("--candidate_dir", type=Path, action="append", required=True)
    ap.add_argument("--candidate_name", action="append", default=[])
    ap.add_argument("--output_json", type=Path, required=True)
    args = ap.parse_args()

    baseline = summarize_run(args.baseline_dir)
    names = list(args.candidate_name or [])
    while len(names) < len(args.candidate_dir):
        names.append(f"candidate_{len(names)+1}")

    candidates = []
    for name, path in zip(names, args.candidate_dir):
        row = summarize_run(path)
        row["name"] = name
        row["delta_vs_baseline"] = {
            "pred_ready_prob_p99": delta(row["pred_ready_prob_p99"], baseline["pred_ready_prob_p99"]),
            "ready_prob_peak_max": delta(row["ready_prob_peak_max"], baseline["ready_prob_peak_max"]),
            "runtime_handoff_ready_frames": delta(row["runtime_handoff_ready_frames"], baseline["runtime_handoff_ready_frames"]),
            "teacher_runtime_handoff_ready_overlap_frames": delta(
                row["teacher_runtime_handoff_ready_overlap_frames"],
                baseline["teacher_runtime_handoff_ready_overlap_frames"],
            ),
            "false_close_apply_rate": delta(row["false_close_apply_rate"], baseline["false_close_apply_rate"]),
            "close_veto_pass_frames": delta(row["close_veto_pass_frames"], baseline["close_veto_pass_frames"]),
            "shadow_handoff_blocked_episode_count": delta(
                row["shadow_handoff_blocked_episode_count"],
                baseline["shadow_handoff_blocked_episode_count"],
            ),
            "runtime_minus_teacher_yaw_mean_episode_weighted_abs": None
            if not math.isfinite(float(row["runtime_minus_teacher_yaw_mean_episode_weighted"]))
            or not math.isfinite(float(baseline["runtime_minus_teacher_yaw_mean_episode_weighted"]))
            else abs(float(row["runtime_minus_teacher_yaw_mean_episode_weighted"]))
            - abs(float(baseline["runtime_minus_teacher_yaw_mean_episode_weighted"])),
        }
        candidates.append(row)

    report = {
        "baseline": baseline,
        "candidates": candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
