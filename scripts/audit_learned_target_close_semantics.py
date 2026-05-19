#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def safe_float(v) -> float:
    try:
        out = float(v)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def percentile_summary(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def audit_target_dataset(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    delta = np.asarray(data["target_delta_teacher"], dtype=np.float32)
    ready = np.asarray(data.get("handoff_ready_target", np.zeros((delta.shape[0],), dtype=np.float32)), dtype=np.float32)
    release_xy = np.asarray(data.get("handoff_threshold_xy_error", np.full((delta.shape[0],), np.nan, dtype=np.float32)), dtype=np.float32)
    release_z = np.asarray(data.get("handoff_threshold_abs_z_error", np.full((delta.shape[0],), np.nan, dtype=np.float32)), dtype=np.float32)
    release_yaw = np.asarray(data.get("handoff_threshold_yaw_error", np.full((delta.shape[0],), np.nan, dtype=np.float32)), dtype=np.float32)
    substage = np.asarray(data.get("substage_id", np.zeros((delta.shape[0],), dtype=np.int64)), dtype=np.int64)
    stage_target_mode = np.asarray(data.get("stage_target_mode", np.zeros((delta.shape[0],), dtype=np.int64)), dtype=np.int64)

    yaw_abs = np.abs(delta[:, 5])
    xy = np.linalg.norm(delta[:, :2], axis=1)
    z_abs = np.abs(delta[:, 2])
    ready_mask = ready > 0.5
    release_ready_by_delta = (
        np.isfinite(release_xy)
        & np.isfinite(release_z)
        & (xy <= np.maximum(release_xy, 1e-6))
        & (z_abs <= np.maximum(release_z, 1e-6))
        & (
            (~np.isfinite(release_yaw))
            | (release_yaw < 0.0)
            | (yaw_abs <= np.maximum(release_yaw, 1e-6))
        )
    )
    return {
        "dataset": str(npz_path),
        "rows": int(delta.shape[0]),
        "substage_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(substage, return_counts=True))},
        "stage_target_mode_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(stage_target_mode, return_counts=True))},
        "ready_rate": float(np.mean(ready_mask)),
        "release_ready_by_delta_rate": float(np.mean(release_ready_by_delta)),
        "yaw_abs": percentile_summary(yaw_abs),
        "xy_error": percentile_summary(xy),
        "abs_z_error": percentile_summary(z_abs),
        "ready_rows": {
            "count": int(np.sum(ready_mask)),
            "yaw_abs": percentile_summary(yaw_abs[ready_mask]),
            "xy_error": percentile_summary(xy[ready_mask]),
            "abs_z_error": percentile_summary(z_abs[ready_mask]),
        },
        "non_ready_rows": {
            "count": int(np.sum(~ready_mask)),
            "yaw_abs": percentile_summary(yaw_abs[~ready_mask]),
            "xy_error": percentile_summary(xy[~ready_mask]),
            "abs_z_error": percentile_summary(z_abs[~ready_mask]),
        },
    }


def audit_runtime_trace(trace_dir: Path) -> dict:
    yaw_runtime = []
    yaw_teacher = []
    xy_runtime = []
    xy_teacher = []
    z_runtime = []
    z_teacher = []
    close_rows = 0
    provider_sources: dict[str, int] = {}
    target_roles: dict[str, int] = {}
    for path in find_trace_files(trace_dir):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                provider = str(row.get("target_provider_source", "unknown"))
                provider_sources[provider] = provider_sources.get(provider, 0) + 1
                role = str(row.get("handoff_target_role", row.get("refiner_current_handoff_target_role", "none")))
                target_roles[role] = target_roles.get(role, 0) + 1
                close_like = bool(
                    row.get("refiner_alignment_planner_close_intent", False)
                    or str(row.get("handoff_target_role", "none")) == "pregrasp_close"
                )
                if not close_like:
                    continue
                rt = np.asarray(row.get("refiner_current_delta_basin_target", row.get("current_delta_basin_target", [np.nan] * 6)), dtype=np.float32).reshape(-1)
                tt = np.asarray(row.get("teacher_current_delta_basin_target", [np.nan] * 6), dtype=np.float32).reshape(-1)
                if rt.size < 6 or tt.size < 6:
                    continue
                close_rows += 1
                yaw_runtime.append(abs(safe_float(rt[5])))
                yaw_teacher.append(abs(safe_float(tt[5])))
                xy_runtime.append(float(np.linalg.norm(rt[:2])) if np.all(np.isfinite(rt[:2])) else math.nan)
                xy_teacher.append(float(np.linalg.norm(tt[:2])) if np.all(np.isfinite(tt[:2])) else math.nan)
                z_runtime.append(abs(safe_float(rt[2])))
                z_teacher.append(abs(safe_float(tt[2])))
    yaw_runtime_np = np.asarray(yaw_runtime, dtype=np.float32)
    yaw_teacher_np = np.asarray(yaw_teacher, dtype=np.float32)
    xy_runtime_np = np.asarray(xy_runtime, dtype=np.float32)
    xy_teacher_np = np.asarray(xy_teacher, dtype=np.float32)
    z_runtime_np = np.asarray(z_runtime, dtype=np.float32)
    z_teacher_np = np.asarray(z_teacher, dtype=np.float32)
    return {
        "trace_dir": str(trace_dir),
        "close_like_rows": int(close_rows),
        "provider_source_counts": provider_sources,
        "handoff_target_role_counts": target_roles,
        "runtime_yaw_abs": percentile_summary(yaw_runtime_np),
        "teacher_yaw_abs": percentile_summary(yaw_teacher_np),
        "runtime_minus_teacher_yaw_mean": float(np.nanmean(yaw_runtime_np - yaw_teacher_np)) if yaw_runtime_np.size else math.nan,
        "runtime_xy": percentile_summary(xy_runtime_np),
        "teacher_xy": percentile_summary(xy_teacher_np),
        "runtime_minus_teacher_xy_mean": float(np.nanmean(xy_runtime_np - xy_teacher_np)) if xy_runtime_np.size else math.nan,
        "runtime_abs_z": percentile_summary(z_runtime_np),
        "teacher_abs_z": percentile_summary(z_teacher_np),
        "runtime_minus_teacher_abs_z_mean": float(np.nanmean(z_runtime_np - z_teacher_np)) if z_runtime_np.size else math.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dataset_npz", type=Path, required=True)
    ap.add_argument("--trace_dir", type=Path, default=None)
    ap.add_argument("--output_json", type=Path, required=True)
    args = ap.parse_args()

    report = {
        "target_dataset_audit": audit_target_dataset(args.target_dataset_npz),
        "trace_audit": None if args.trace_dir is None else audit_runtime_trace(args.trace_dir),
        "diagnosis": {
            "expected_contract": "teacher motion target is basin-compatible; close/handoff readiness uses stage-spec target",
            "current_risk": "learned target predictor is trained on motion delta labels but runtime had been using an implicit full-6d learned pose contract in close stage",
            "recommended_fix": "preserve canonical fallback orientation contract for close-stage learned motion target while keeping learned translation residual runtime-safe",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
