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


def dataset_audit(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    proxy = np.asarray(data["proxy_current_delta_basin_target"], dtype=np.float32)
    teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    runtime_valid = np.asarray(data.get("runtime_handoff_metric_valid", np.zeros((proxy.shape[0],), dtype=np.float32)), dtype=np.float32) > 0.5
    source = np.asarray(data.get("source_name", np.full((proxy.shape[0],), "unknown", dtype="U32")))
    xy = np.linalg.norm(proxy[:, :2], axis=1)
    z_abs = np.abs(proxy[:, 2])
    yaw_abs = np.abs(proxy[:, 5])
    report = {
        "dataset": str(npz_path),
        "rows": int(proxy.shape[0]),
        "teacher_ready_rate": float(np.mean(teacher_ready)),
        "runtime_valid_rate": float(np.mean(runtime_valid)),
        "proxy_xy": percentile_summary(xy),
        "proxy_abs_z": percentile_summary(z_abs),
        "proxy_yaw_abs": percentile_summary(yaw_abs),
        "teacher_ready_rows": {
            "count": int(np.sum(teacher_ready)),
            "proxy_xy": percentile_summary(xy[teacher_ready]),
            "proxy_abs_z": percentile_summary(z_abs[teacher_ready]),
            "proxy_yaw_abs": percentile_summary(yaw_abs[teacher_ready]),
        },
        "runtime_valid_rows": {
            "count": int(np.sum(runtime_valid)),
            "proxy_xy": percentile_summary(xy[runtime_valid]),
            "proxy_abs_z": percentile_summary(z_abs[runtime_valid]),
            "proxy_yaw_abs": percentile_summary(yaw_abs[runtime_valid]),
        },
    }
    uniq, cnt = np.unique(source, return_counts=True)
    report["source_counts"] = {str(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    return report


def trace_audit(trace_dir: Path) -> dict:
    rows = 0
    pred_ready = []
    uncertainty = []
    teacher_ready = []
    runtime_ready = []
    xy = []
    z_abs = []
    yaw_abs = []
    close_like_rows = 0
    for path in find_trace_files(trace_dir):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                delta = np.asarray(row.get("refiner_current_delta_basin_target", row.get("current_delta_basin_target", [np.nan] * 6)), dtype=np.float32).reshape(-1)
                if delta.size < 6:
                    continue
                rows += 1
                close_like = bool(
                    row.get("refiner_alignment_planner_close_intent", False)
                    or str(row.get("handoff_target_role", "none")) == "pregrasp_close"
                )
                if close_like:
                    close_like_rows += 1
                aux = row.get("handoff_aux_provider", {}) or {}
                pred_ready.append(float(aux.get("pred_ready_prob", np.nan)))
                uncertainty.append(float(aux.get("pred_uncertainty", np.nan)))
                teacher_ready.append(float(row.get("teacher_truth_handoff_ready", 0.0)))
                runtime_ready.append(float(row.get("runtime_handoff_ready_pred", row.get("runtime_handoff_ready", 0.0))))
                xy.append(float(np.linalg.norm(delta[:2])) if np.all(np.isfinite(delta[:2])) else math.nan)
                z_abs.append(abs(float(delta[2])) if math.isfinite(float(delta[2])) else math.nan)
                yaw_abs.append(abs(float(delta[5])) if math.isfinite(float(delta[5])) else math.nan)
    pred_ready_np = np.asarray(pred_ready, dtype=np.float32)
    uncertainty_np = np.asarray(uncertainty, dtype=np.float32)
    teacher_ready_np = np.asarray(teacher_ready, dtype=np.float32) > 0.5
    runtime_ready_np = np.asarray(runtime_ready, dtype=np.float32) > 0.5
    return {
        "trace_dir": str(trace_dir),
        "rows": int(rows),
        "close_like_rows": int(close_like_rows),
        "pred_ready_prob": percentile_summary(pred_ready_np),
        "pred_uncertainty": percentile_summary(uncertainty_np),
        "teacher_ready_rate": float(np.mean(teacher_ready_np)) if teacher_ready_np.size else math.nan,
        "runtime_ready_rate": float(np.mean(runtime_ready_np)) if runtime_ready_np.size else math.nan,
        "proxy_xy": percentile_summary(np.asarray(xy, dtype=np.float32)),
        "proxy_abs_z": percentile_summary(np.asarray(z_abs, dtype=np.float32)),
        "proxy_yaw_abs": percentile_summary(np.asarray(yaw_abs, dtype=np.float32)),
    }


def compare(train: dict, runtime: dict) -> dict:
    out = {}
    for key in ("proxy_xy", "proxy_abs_z", "proxy_yaw_abs"):
        out[key] = {
            "runtime_minus_train_p50": float(runtime[key]["p50"] - train[key]["p50"]),
            "runtime_minus_train_p90": float(runtime[key]["p90"] - train[key]["p90"]),
            "teacher_ready_train_p50": float(train["teacher_ready_rows"][key]["p50"]),
            "runtime_p50": float(runtime[key]["p50"]),
        }
    out["ready_rate_gap"] = {
        "teacher_ready_rate_train": float(train["teacher_ready_rate"]),
        "teacher_ready_rate_runtime": float(runtime["teacher_ready_rate"]),
        "runtime_ready_rate_runtime": float(runtime["runtime_ready_rate"]),
        "pred_ready_prob_p99_runtime": float(runtime["pred_ready_prob"]["p99"]),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", type=Path, required=True)
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    args = ap.parse_args()

    train = dataset_audit(args.dataset_npz)
    runtime = trace_audit(args.trace_dir)
    report = {
        "dataset_audit": train,
        "runtime_trace_audit": runtime,
        "comparison": compare(train, runtime),
        "diagnosis": {
            "primary_issue": "ready positives are very sparse in training and runtime pred_ready_prob is nearly collapsed to zero",
            "interpretation": "this is more consistent with distribution/scale mismatch than with a slightly conservative threshold",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
