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


def as_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def as_bool(value) -> bool:
    return bool(value)


def episode_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_gripper_trace"):
        stem = stem[: -len("_gripper_trace")]
    return stem


def _best_norm_snapshot(rows: list[dict]) -> dict:
    best = None
    for row in rows:
        aux = row.get("handoff_aux_provider") or {}
        xy = as_float(aux.get("pred_xy_norm"))
        z = as_float(aux.get("pred_abs_z_norm"))
        yaw = as_float(aux.get("pred_yaw_norm"))
        if not (math.isfinite(xy) and math.isfinite(z) and math.isfinite(yaw)):
            continue
        score = max(xy, z, yaw)
        item = {
            "step": int(row.get("step", -1)),
            "max_norm": float(score),
            "xy_norm": float(xy),
            "z_norm": float(z),
            "yaw_norm": float(yaw),
            "planner_close_intent": as_bool(row.get("refiner_alignment_planner_close_intent", False)),
            "near_target": as_bool(row.get("refiner_alignment_near_target", False)),
            "base_gripper_raw": as_float(row.get("base_gripper_raw")),
            "shadow_reason": str(row.get("refiner_close_intent_shadow_reason", "missing")),
            "closeness_score": as_float(row.get("handoff_pred_closeness_score")),
            "progress_prob": as_float(row.get("handoff_pred_progress_prob")),
        }
        if best is None or item["max_norm"] < best["max_norm"]:
            best = item
    return best or {}


def classify_episode(summary: dict) -> str:
    near_ratio = summary["near_target_ratio"]
    pregrasp_ratio = summary["pregrasp_close_ratio"]
    min_gripper = summary["base_gripper_min"]
    close_frames = summary["planner_close_frames"]
    shadow_block_frames = summary["shadow_block_frames"]
    xy_block_frames = summary["xy_block_frames"]
    if close_frames == 0 and near_ratio >= 0.5 and pregrasp_ratio >= 0.5 and min_gripper >= 0.9:
        return "planner-open-template-near-close"
    if close_frames > 0 and shadow_block_frames > 0:
        return "planner-close-jitter-but-shadow-blocked"
    if close_frames == 0 and near_ratio < 0.5:
        return "approach-not-near-enough"
    if close_frames == 0 and near_ratio >= 0.5:
        return "planner-open-no-close"
    if close_frames > 0 and xy_block_frames > 0:
        return "planner-close-but-geometry-not-ready"
    return "mixed"


def summarize_episode(path: Path, rows: list[dict], close_threshold: float) -> dict:
    n = max(len(rows), 1)
    planner_close_frames = sum(as_bool(r.get("refiner_alignment_planner_close_intent", False)) for r in rows)
    near_target_frames = sum(as_bool(r.get("refiner_alignment_near_target", False)) for r in rows)
    pregrasp_close_frames = sum(str(r.get("refiner_current_handoff_target_role", "none")) == "pregrasp_close" for r in rows)
    canonical_close_frames = sum(
        str(r.get("refiner_target_provider_source", "")) == "learned_target_predictor__canonical_close_orientation_contract"
        for r in rows
    )
    close_requirement_frames = sum(as_bool(r.get("refiner_alignment_close_requirement_satisfied", False)) for r in rows)
    near_target_for_gripper_frames = sum(as_bool(r.get("near_target_for_gripper", False)) for r in rows)
    shadow_block_frames = sum(str(r.get("refiner_close_blocked_reason", "")) == "shadow_blocked" for r in rows)
    xy_block_frames = sum(str(r.get("refiner_close_blocked_reason", "")) == "xy" for r in rows)
    z_block_frames = sum(str(r.get("refiner_close_blocked_reason", "")) == "z" for r in rows)
    yaw_block_frames = sum(str(r.get("refiner_close_blocked_reason", "")) == "yaw" for r in rows)
    auto_close_candidate_frames = sum(as_bool(r.get("refiner_close_intent_shadow_would_auto_close", False)) for r in rows)
    grip_values = [as_float(r.get("base_gripper_raw")) for r in rows if math.isfinite(as_float(r.get("base_gripper_raw")))]
    shadow_reason_counts = collections.Counter(str(r.get("refiner_close_intent_shadow_reason", "missing")) for r in rows)
    shadow_axis_counts = collections.Counter(str(r.get("refiner_close_intent_shadow_blocking_axis", "missing")) for r in rows)
    progress_vals = [as_float(r.get("handoff_pred_progress_prob")) for r in rows]
    progress_vals = [v for v in progress_vals if math.isfinite(v)]
    summary = {
        "episode_id": episode_id_from_path(path),
        "frames": len(rows),
        "planner_close_frames": int(planner_close_frames),
        "near_target_frames": int(near_target_frames),
        "pregrasp_close_frames": int(pregrasp_close_frames),
        "canonical_close_target_frames": int(canonical_close_frames),
        "close_requirement_frames": int(close_requirement_frames),
        "near_target_for_gripper_frames": int(near_target_for_gripper_frames),
        "shadow_block_frames": int(shadow_block_frames),
        "xy_block_frames": int(xy_block_frames),
        "z_block_frames": int(z_block_frames),
        "yaw_block_frames": int(yaw_block_frames),
        "auto_close_candidate_frames": int(auto_close_candidate_frames),
        "near_target_ratio": float(near_target_frames / n),
        "pregrasp_close_ratio": float(pregrasp_close_frames / n),
        "canonical_close_ratio": float(canonical_close_frames / n),
        "close_requirement_ratio": float(close_requirement_frames / n),
        "near_target_for_gripper_ratio": float(near_target_for_gripper_frames / n),
        "base_gripper_min": float(min(grip_values)) if grip_values else math.nan,
        "base_gripper_close_threshold": float(close_threshold),
        "shadow_reason_counts": dict(shadow_reason_counts),
        "shadow_axis_counts": dict(shadow_axis_counts),
        "peak_progress_prob": float(max(progress_vals)) if progress_vals else math.nan,
        "best_norm_snapshot": _best_norm_snapshot(rows),
    }
    summary["bucket"] = classify_episode(summary)
    return summary


def combine(episodes: list[dict]) -> dict:
    bucket_counts = collections.Counter(ep["bucket"] for ep in episodes)
    planner_close_eps = sum(ep["planner_close_frames"] > 0 for ep in episodes)
    near_majority_eps = sum(ep["near_target_ratio"] >= 0.5 for ep in episodes)
    pregrasp_majority_eps = sum(ep["pregrasp_close_ratio"] >= 0.5 for ep in episodes)
    absent_despite_near_eps = sum(
        ep["planner_close_frames"] == 0 and ep["near_target_ratio"] >= 0.5 and ep["pregrasp_close_ratio"] >= 0.5
        for ep in episodes
    )
    return {
        "episode_count": len(episodes),
        "bucket_counts": dict(bucket_counts),
        "episodes_with_any_planner_close_intent": int(planner_close_eps),
        "episodes_near_target_majority": int(near_majority_eps),
        "episodes_pregrasp_close_majority": int(pregrasp_majority_eps),
        "episodes_no_close_intent_despite_near_target": int(absent_despite_near_eps),
        "planner_close_frame_sum": int(sum(ep["planner_close_frames"] for ep in episodes)),
        "shadow_block_frame_sum": int(sum(ep["shadow_block_frames"] for ep in episodes)),
        "xy_block_frame_sum": int(sum(ep["xy_block_frames"] for ep in episodes)),
        "auto_close_candidate_frame_sum": int(sum(ep["auto_close_candidate_frames"] for ep in episodes)),
    }


def build_takeaways(summary: dict) -> list[str]:
    notes: list[str] = []
    if summary["episodes_no_close_intent_despite_near_target"] > 0:
        notes.append(
            "Most episodes reach close-like / pregrasp-close geometry, but planner gripper output stays open; "
            "close-intent scarcity is primarily a planner close-command issue."
        )
    if summary["episodes_with_any_planner_close_intent"] <= 2:
        notes.append(
            "Planner close-intent appears in only a small minority of heldout episodes, so alignment cannot rely on runtime close-positive rows as its main trigger."
        )
    if summary["shadow_block_frame_sum"] > 0:
        notes.append(
            "When planner does emit close-intent, the dominant downstream outcome is shadow-blocked rather than successful close pass, so those few close bursts are not useful success anchors."
        )
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--close_threshold", type=float, default=0.50)
    args = ap.parse_args()

    files = find_trace_files(args.trace_dir)
    episodes = [summarize_episode(path, load_jsonl(path), args.close_threshold) for path in files]
    report = {
        "trace_dir": str(args.trace_dir),
        "summary": combine(episodes),
        "takeaways": build_takeaways(combine(episodes)),
        "episodes": episodes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
