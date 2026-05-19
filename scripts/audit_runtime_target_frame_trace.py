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
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        out = float(row.get(key, default))
    except Exception:
        return default
    return out if math.isfinite(out) else default


def as_bool(row: dict, key: str) -> bool:
    return bool(row.get(key, False))


def finite(vals) -> list[float]:
    return [float(v) for v in vals if math.isfinite(float(v))]


def mean(vals) -> float:
    vals = finite(vals)
    return float(sum(vals) / len(vals)) if vals else math.nan


def max_abs(vals) -> float:
    vals = finite(vals)
    return float(max((abs(v) for v in vals), default=math.nan))


def runtime_components(row: dict) -> tuple[float, float, float]:
    delta = row.get("refiner_current_delta_basin_target")
    if not isinstance(delta, list) or len(delta) < 6:
        return math.nan, math.nan, math.nan
    try:
        xy = math.sqrt(float(delta[0]) ** 2 + float(delta[1]) ** 2)
        z = abs(float(delta[2]))
        yaw = abs(float(delta[5]))
    except Exception:
        return math.nan, math.nan, math.nan
    return xy, z, yaw


def frame_bucket(row: dict) -> bool:
    return bool(
        as_bool(row, "refiner_alignment_planner_close_intent")
        or as_bool(row, "b2_candidate_shadow_close_neighborhood")
        or as_bool(row, "b2_candidate_shadow_nearish_runtime")
        or as_bool(row, "refiner_close_intent_shadow_would_auto_close")
    )


def summarize_episode(path: Path, rows: list[dict]) -> dict:
    audit_rows = [r for r in rows if frame_bucket(r)]
    if not audit_rows:
        audit_rows = rows
    diffs = {"xy": [], "z": [], "yaw": []}
    runtime_vals = {"xy": [], "z": [], "yaw": []}
    teacher_vals = {"xy": [], "z": [], "yaw": []}
    threshold_ratios = {"xy": [], "z": [], "yaw": []}
    target_roles = collections.Counter()
    provider_sources = collections.Counter()
    target_sources = collections.Counter()
    privileged_runtime_frames = 0
    close_block_reasons = collections.Counter()
    for row in audit_rows:
        rxy, rz, ryaw = runtime_components(row)
        txy = as_float(row, "teacher_truth_basin_xy")
        tz = as_float(row, "teacher_truth_basin_z")
        tyaw = as_float(row, "teacher_truth_basin_yaw")
        for name, rv, tv in (("xy", rxy, txy), ("z", rz, tz), ("yaw", ryaw, tyaw)):
            runtime_vals[name].append(rv)
            teacher_vals[name].append(tv)
            if math.isfinite(rv) and math.isfinite(tv):
                diffs[name].append(rv - tv)
        for name, key in (
            ("xy", "refiner_close_xy_threshold"),
            ("z", "refiner_close_abs_z_threshold"),
            ("yaw", "refiner_close_yaw_threshold"),
        ):
            rv = {"xy": rxy, "z": rz, "yaw": ryaw}[name]
            th = as_float(row, key)
            if math.isfinite(rv) and math.isfinite(th) and th > 0:
                threshold_ratios[name].append(rv / th)
        target_roles[str(row.get("refiner_current_handoff_target_role") or row.get("handoff_target_role") or "none")] += 1
        provider_sources[str(row.get("motion_target_provider_source") or "unknown")] += 1
        target_sources[str(row.get("refiner_target_provider_source") or "unknown")] += 1
        privileged_runtime_frames += int(as_bool(row, "refiner_target_uses_privileged_runtime"))
        close_block_reasons[str(row.get("refiner_close_blocked_reason", "none"))] += 1
    summary = {
        "episode_trace": path.name,
        "frames": len(rows),
        "audit_frames": len(audit_rows),
        "target_role_counts": dict(target_roles),
        "motion_provider_source_counts": dict(provider_sources),
        "target_provider_source_counts": dict(target_sources),
        "privileged_runtime_frames": privileged_runtime_frames,
        "close_block_reason_counts": dict(close_block_reasons),
    }
    for name in ("xy", "z", "yaw"):
        summary[f"runtime_{name}_mean"] = mean(runtime_vals[name])
        summary[f"teacher_{name}_mean"] = mean(teacher_vals[name])
        summary[f"runtime_minus_teacher_{name}_mean"] = mean(diffs[name])
        summary[f"runtime_minus_teacher_{name}_max_abs"] = max_abs(diffs[name])
        summary[f"runtime_over_threshold_{name}_mean"] = mean(threshold_ratios[name])
    return summary


def combine(episodes: list[dict]) -> dict:
    role_counts = collections.Counter()
    provider_counts = collections.Counter()
    target_counts = collections.Counter()
    block_counts = collections.Counter()
    for ep in episodes:
        role_counts.update(ep["target_role_counts"])
        provider_counts.update(ep["motion_provider_source_counts"])
        target_counts.update(ep["target_provider_source_counts"])
        block_counts.update(ep["close_block_reason_counts"])
    summary = {
        "episode_count": len(episodes),
        "frames": sum(ep["frames"] for ep in episodes),
        "audit_frames": sum(ep["audit_frames"] for ep in episodes),
        "privileged_runtime_frames": sum(ep["privileged_runtime_frames"] for ep in episodes),
        "target_role_counts": dict(role_counts),
        "motion_provider_source_counts": dict(provider_counts),
        "target_provider_source_counts": dict(target_counts),
        "close_block_reason_counts": dict(block_counts),
    }
    for name in ("xy", "z", "yaw"):
        summary[f"runtime_minus_teacher_{name}_mean_episode_weighted"] = mean(
            [ep[f"runtime_minus_teacher_{name}_mean"] for ep in episodes]
        )
        summary[f"runtime_minus_teacher_{name}_max_abs"] = max(
            finite(ep[f"runtime_minus_teacher_{name}_max_abs"] for ep in episodes),
            default=math.nan,
        )
        summary[f"runtime_over_threshold_{name}_mean_episode_weighted"] = mean(
            [ep[f"runtime_over_threshold_{name}_mean"] for ep in episodes]
        )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    args = ap.parse_args()
    episodes = [summarize_episode(path, load_jsonl(path)) for path in find_trace_files(args.trace_dir)]
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
