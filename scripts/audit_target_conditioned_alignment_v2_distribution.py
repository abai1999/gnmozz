#!/usr/bin/env python3
"""Audit target-delta coverage for target-conditioned alignment v2.

This script is intentionally read-only.  It compares target-relative error
coverage in the local proposal training/cache data against runtime trace rows,
especially alignment takeover rows.  The goal is to decide whether existing data
is sufficient for a target-conditioned alignment model or whether new runtime
state sampling is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


XYZ_ACTION_SCALE = np.asarray([0.008, 0.008, 0.006], dtype=np.float32)
YAW_ACTION_SCALE = 0.12


def _load_npz_delta(path: Path) -> tuple[np.ndarray, str]:
    data = np.load(path, allow_pickle=True)
    for key in (
        "proposal_target_delta_local",
        "motion_target_delta_local",
        "current_delta_basin_target",
        "target_delta_teacher",
        "privileged_current_delta_basin_target",
    ):
        if key in data.files:
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[:, 0, :]
            return arr[:, :6], key
    raise KeyError(f"{path} does not contain a known target-delta field")


def _iter_trace_rows(trace_dir: Path) -> Iterable[dict]:
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_trace_file"] = str(path)
                    yield row


def _trace_delta(row: dict) -> np.ndarray | None:
    for key in (
        "refiner_current_delta_basin_target",
        "motion_target_delta_local",
        "current_delta_basin_target",
    ):
        value = row.get(key)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if arr.size >= 6 and np.all(np.isfinite(arr[:6])) and float(np.linalg.norm(arr[:6])) > 1e-8:
            return arr[:6]
    metrics = row.get("refiner_current_handoff_metrics")
    if isinstance(metrics, dict):
        xy = metrics.get("xy_error")
        z = metrics.get("z_error", metrics.get("abs_z_error"))
        yaw = metrics.get("yaw_error")
        if xy is not None and z is not None and yaw is not None:
            # Direction is not recoverable from scalar metrics; store magnitudes.
            return np.asarray([float(xy), 0.0, float(z), 0.0, 0.0, float(yaw)], dtype=np.float32)
    return None


def _collect_trace_delta(trace_dir: Path, *, takeover_only: bool = False) -> np.ndarray:
    deltas: list[np.ndarray] = []
    for row in _iter_trace_rows(trace_dir):
        if takeover_only and row.get("refiner_alignment_takeover_active") is not True:
            continue
        delta = _trace_delta(row)
        if delta is not None:
            deltas.append(delta)
    if not deltas:
        return np.zeros((0, 6), dtype=np.float32)
    return np.stack(deltas, axis=0).astype(np.float32)


def _hist(values: np.ndarray, bins: list[float]) -> dict:
    if values.size == 0:
        return {"bins": bins, "counts": [], "rates": []}
    counts, edges = np.histogram(values, bins=np.asarray(bins, dtype=np.float32))
    total = max(int(values.size), 1)
    return {
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
        "rates": [float(x) / float(total) for x in counts.tolist()],
    }


def _bucket_counts(xy: np.ndarray, z: np.ndarray, yaw: np.ndarray) -> dict:
    if xy.size == 0:
        return {}
    micro = (xy < 0.015) & (z < 0.03) & (yaw < 0.12)
    near = (~micro) & (xy < 0.05) & (z < 0.10) & (yaw < 0.25)
    mid = (~micro) & (~near) & (xy < 0.12) & (z < 0.25)
    far = ~(micro | near | mid)
    masks = {
        "micro_contact_refine": micro,
        "near_alignment": near,
        "mid_approach_assist": mid,
        "far_coarse_approach": far,
    }
    return {
        name: {"count": int(mask.sum()), "rate": float(mask.mean())}
        for name, mask in masks.items()
    }


def _summary(name: str, delta: np.ndarray) -> dict:
    delta = np.asarray(delta, dtype=np.float32)
    if delta.size == 0:
        return {"name": name, "rows": 0}
    xy = np.linalg.norm(delta[:, :2], axis=1)
    z = np.abs(delta[:, 2])
    yaw = np.abs(delta[:, 5])
    xyz_ratio = np.stack(
        [
            np.abs(delta[:, 0]) / XYZ_ACTION_SCALE[0],
            np.abs(delta[:, 1]) / XYZ_ACTION_SCALE[1],
            np.abs(delta[:, 2]) / XYZ_ACTION_SCALE[2],
        ],
        axis=1,
    )
    return {
        "name": name,
        "rows": int(delta.shape[0]),
        "xy": _stats(xy),
        "z": _stats(z),
        "yaw": _stats(yaw),
        "bucket_counts": _bucket_counts(xy, z, yaw),
        "xy_hist": _hist(xy, [0.0, 0.005, 0.015, 0.03, 0.05, 0.10, 0.20, 0.50, 1.0]),
        "z_hist": _hist(z, [0.0, 0.0035, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.0]),
        "yaw_hist": _hist(yaw, [0.0, 0.02, 0.06, 0.12, 0.25, 0.50, 1.0]),
        "action_scale_ratio": {
            "abs_dx_over_scale": _stats(xyz_ratio[:, 0]),
            "abs_dy_over_scale": _stats(xyz_ratio[:, 1]),
            "abs_dz_over_scale": _stats(xyz_ratio[:, 2]),
            "abs_dyaw_over_scale": _stats(yaw / YAW_ACTION_SCALE),
        },
    }


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_npz", type=Path, default=None)
    parser.add_argument("--cache_npz", type=Path, default=None)
    parser.add_argument("--trace_dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict = {
        "audit": "target_conditioned_alignment_v2_distribution",
        "notes": [
            "Trace rows may only expose scalar handoff metrics. In that case xy/z/yaw magnitudes are audited, but xy direction/sign is not recoverable.",
            "Dataset/cache rows use proposal_target_delta_local when present, preserving signed target-relative deltas.",
        ],
        "thresholds": {
            "micro_contact_refine": "xy<0.015 and z<0.03 and yaw<0.12",
            "near_alignment": "xy<0.05 and z<0.10 and yaw<0.25, excluding micro",
            "mid_approach_assist": "xy<0.12 and z<0.25, excluding micro/near",
            "far_coarse_approach": "everything else",
        },
        "summaries": {},
    }

    if args.dataset_npz is not None:
        delta, key = _load_npz_delta(args.dataset_npz)
        report["summaries"]["dataset"] = _summary(f"dataset:{key}", delta)
        report["dataset_npz"] = str(args.dataset_npz)

    if args.cache_npz is not None:
        delta, key = _load_npz_delta(args.cache_npz)
        report["summaries"]["cache"] = _summary(f"cache:{key}", delta)
        report["cache_npz"] = str(args.cache_npz)

    trace_reports = []
    for trace_dir in args.trace_dir:
        all_delta = _collect_trace_delta(trace_dir, takeover_only=False)
        takeover_delta = _collect_trace_delta(trace_dir, takeover_only=True)
        trace_reports.append(
            {
                "trace_dir": str(trace_dir),
                "all_rows": _summary("runtime_trace_all", all_delta),
                "takeover_rows": _summary("runtime_trace_takeover", takeover_delta),
            }
        )
    report["trace_summaries"] = trace_reports

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(args.output), "sections": list(report["summaries"].keys()), "trace_dirs": len(trace_reports)}, indent=2))


if __name__ == "__main__":
    main()
