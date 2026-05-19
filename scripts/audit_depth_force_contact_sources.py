#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEYS = [
    "front_rgb",
    "wrist_rgb",
    "wrist_depth",
    "proprio",
    "ft_hist",
    "force_history",
    "gripper_touch_forces",
    "planner_base_action_local_raw",
    "executed_action_local",
    "oracle_action_local",
    "candidate_actions_local",
    "candidate_oracle_score",
    "oracle_candidate_index",
    "best_candidate_index",
    "episode_index",
    "step_index",
]


def finite_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    if arr.dtype.kind not in ("f", "i", "u", "b"):
        return 1.0
    return float(np.isfinite(arr).mean())


def yaw_bucket_counts(actions: np.ndarray) -> dict[str, int]:
    yaw_abs = np.abs(actions[..., 5]).reshape(-1)
    return {
        "no_yaw": int(np.sum(yaw_abs < 0.01)),
        "small_yaw": int(np.sum((yaw_abs >= 0.01) & (yaw_abs < 0.05))),
        "medium_yaw": int(np.sum((yaw_abs >= 0.05) & (yaw_abs < 0.09))),
        "large_yaw": int(np.sum(yaw_abs >= 0.09)),
    }


def audit_one(path: Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    report: dict = {"path": str(path), "keys": {}, "usable": {}}
    n = None
    for key in raw.files:
        arr = np.asarray(raw[key])
        if n is None and arr.ndim > 0:
            n = int(arr.shape[0])
    report["rows"] = int(n or 0)

    for key in KEYS:
        if key not in raw.files:
            report["keys"][key] = {"exists": False}
            continue
        arr = np.asarray(raw[key])
        report["keys"][key] = {
            "exists": True,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_ratio": finite_ratio(arr),
        }

    if "wrist_depth" in raw.files:
        depth = np.asarray(raw["wrist_depth"], dtype=np.float32)
        report["depth"] = {
            "finite_ratio": finite_ratio(depth),
            "near_fraction_lt_0p08": float(np.mean(depth[np.isfinite(depth)] < 0.08)) if np.isfinite(depth).any() else 0.0,
        }

    force = None
    for key in ("ft_hist", "force_history"):
        if key in raw.files:
            force = np.asarray(raw[key], dtype=np.float32)
            break
    if force is None and "gripper_touch_forces" in raw.files:
        force = np.asarray(raw["gripper_touch_forces"], dtype=np.float32)
    if force is not None:
        flat = force.reshape(force.shape[0], -1)
        report["force"] = {
            "finite_ratio": finite_ratio(force),
            "norm_mean": float(np.nanmean(np.linalg.norm(flat, axis=1))),
            "norm_p95": float(np.nanpercentile(np.linalg.norm(flat, axis=1), 95)),
        }

    if "candidate_actions_local" in raw.files:
        actions = np.asarray(raw["candidate_actions_local"], dtype=np.float32)
        report["candidate_bank"] = {
            "shape": list(actions.shape),
            "yaw_buckets": yaw_bucket_counts(actions),
            "has_scores": "candidate_oracle_score" in raw.files,
            "has_best_index": "oracle_candidate_index" in raw.files or "best_candidate_index" in raw.files,
        }
        if "candidate_oracle_score" in raw.files:
            scores = np.asarray(raw["candidate_oracle_score"], dtype=np.float32)
            report["candidate_bank"]["score_std_mean"] = float(np.nanmean(np.nanstd(scores, axis=1)))

    report["usable"]["candidate_ranking"] = bool(
        "front_rgb" in raw.files
        and "wrist_depth" in raw.files
        and "proprio" in raw.files
        and "candidate_actions_local" in raw.files
        and "candidate_oracle_score" in raw.files
    )
    report["usable"]["force_aware"] = bool("ft_hist" in raw.files or "force_history" in raw.files or "gripper_touch_forces" in raw.files)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    reports = [audit_one(Path(p)) for p in args.input_npz]
    summary = {
        "files": reports,
        "usable_candidate_ranking_files": int(sum(r["usable"]["candidate_ranking"] for r in reports)),
        "force_aware_files": int(sum(r["usable"]["force_aware"] for r in reports)),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
