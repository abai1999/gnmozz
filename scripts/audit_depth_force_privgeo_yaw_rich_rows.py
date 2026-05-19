#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_pose_candidate_dataset import pose_delta_local_between


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _angle_abs_sym(yaw: np.ndarray, period: float) -> np.ndarray:
    y = np.asarray(yaw, dtype=np.float32)
    if not np.isfinite(float(period)) or float(period) <= 0.0:
        return np.abs((y + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    p = float(period)
    return np.abs((y + 0.5 * p) % p - 0.5 * p).astype(np.float32)


def _summary(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--yaw_threshold_deg", type=float, default=8.0)
    ap.add_argument("--xy_max", type=float, default=0.05)
    ap.add_argument("--z_max", type=float, default=0.08)
    ap.add_argument("--yaw_symmetry_period", type=float, default=np.pi / 2.0)
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    support = _load_npz(Path(args.support_npz))
    cur_key = "privileged_current_pose_7d" if "privileged_current_pose_7d" in support else "current_pose_7d"
    tgt_key = "privileged_motion_target_pose_7d"
    if cur_key not in support or tgt_key not in support:
        raise SystemExit("missing current/target pose fields for yaw-rich audit")

    current = np.asarray(support[cur_key], dtype=np.float32)
    target = np.asarray(support[tgt_key], dtype=np.float32)
    eps = min(current.shape[0], target.shape[0])
    current = current[:eps]
    target = target[:eps]
    d = np.array([pose_delta_local_between(current[i], target[i]) for i in range(eps)], dtype=np.float32)
    yaw_abs = _angle_abs_sym(d[:, 5], float(args.yaw_symmetry_period))
    xy_norm = np.linalg.norm(d[:, :2], axis=-1).astype(np.float32)
    z_abs = np.abs(d[:, 2]).astype(np.float32)
    yaw_rich = (yaw_abs >= np.deg2rad(float(args.yaw_threshold_deg))) & (xy_norm <= float(args.xy_max)) & (z_abs <= float(args.z_max))

    episode = np.asarray(support.get("episode_index", np.zeros((eps,), dtype=np.int64)), dtype=np.int64)[:eps]
    report = {
        "support_npz": str(args.support_npz),
        "rows": int(eps),
        "yaw_threshold_deg": float(args.yaw_threshold_deg),
        "xy_max": float(args.xy_max),
        "z_max": float(args.z_max),
        "yaw_rich_count": int(np.sum(yaw_rich)),
        "yaw_rich_rate": float(np.mean(yaw_rich.astype(np.float32))),
        "yaw_abs": _summary(yaw_abs),
        "xy_norm": _summary(xy_norm),
        "z_abs": _summary(z_abs),
        "per_episode": {},
    }
    for ep in sorted(int(x) for x in np.unique(episode)):
        m = episode == ep
        report["per_episode"][str(ep)] = {
            "rows": int(np.sum(m)),
            "yaw_rich_count": int(np.sum(yaw_rich[m])),
            "yaw_rich_rate": float(np.mean(yaw_rich[m].astype(np.float32))) if np.any(m) else 0.0,
            "yaw_abs": _summary(yaw_abs[m]),
            "xy_norm": _summary(xy_norm[m]),
            "z_abs": _summary(z_abs[m]),
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
