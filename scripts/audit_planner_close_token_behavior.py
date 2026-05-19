#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np


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


def pstats(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "close_rate_le_0_5": float(np.mean(arr <= 0.5)),
        "close_rate_le_0_2": float(np.mean(arr <= 0.2)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--close_threshold", type=float, default=0.5)
    args = ap.parse_args()

    files = find_trace_files(args.trace_dir)
    by_phase = collections.defaultdict(list)
    by_role = collections.defaultdict(list)
    by_bucket = collections.defaultdict(list)
    per_episode = []

    for path in files:
        rows = load_jsonl(path)
        ep_id = episode_id_from_path(path)
        ep_close_like = []
        ep_near_target = []
        ep_planner_close = []
        for row in rows:
            grip = as_float(row.get("base_gripper_raw"))
            if not math.isfinite(grip):
                continue
            phase = str(row.get("refiner_stage_target_mode", "unknown"))
            role = str(row.get("refiner_current_handoff_target_role", "none"))
            near_target = as_bool(row.get("refiner_alignment_near_target", False))
            canonical_close = str(row.get("refiner_target_provider_source", "")) == "learned_target_predictor__canonical_close_orientation_contract"
            close_like = near_target and role == "pregrasp_close" and canonical_close
            planner_close = as_bool(row.get("refiner_alignment_planner_close_intent", False))

            if near_target:
                by_bucket["near_target"].append(grip)
                ep_near_target.append(grip)
            if close_like:
                by_bucket["close_like"].append(grip)
                by_phase[phase].append(grip)
                by_role[role].append(grip)
                ep_close_like.append(grip)
            if planner_close:
                by_bucket["planner_close_intent"].append(grip)
                ep_planner_close.append(grip)

        per_episode.append(
            {
                "episode_id": ep_id,
                "close_like_gripper": pstats(ep_close_like),
                "near_target_gripper": pstats(ep_near_target),
                "planner_close_gripper": pstats(ep_planner_close),
            }
        )

    report = {
        "trace_dir": str(args.trace_dir),
        "close_threshold": float(args.close_threshold),
        "overall": {k: pstats(v) for k, v in by_bucket.items()},
        "phase_bucket_stats": {k: pstats(v) for k, v in sorted(by_phase.items())},
        "target_role_stats": {k: pstats(v) for k, v in sorted(by_role.items())},
        "episodes": per_episode,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
