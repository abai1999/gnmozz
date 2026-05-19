#!/usr/bin/env python3
"""Build a diagnostic failure split from the 20260517a edge-pair phase1 raw.

This split is intentionally non-training: it preserves the raw teacher trace
rows for failure analysis, but keeps them out of the main student dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _as_float_array(data: dict, key: str, default=np.nan) -> np.ndarray:
    if key not in data:
        return np.asarray([default], dtype=np.float32)
    arr = np.asarray(data[key])
    if arr.dtype.kind in {"U", "S", "O"}:
        return np.asarray([default], dtype=np.float32)
    return arr.astype(np.float32)


def _as_str_list(data: dict, key: str) -> list[str]:
    if key not in data:
        return []
    arr = np.asarray(data[key])
    return [str(x.item() if hasattr(x, "item") else x) for x in arr.reshape(-1)]


def _stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, required=True)
    ap.add_argument("--split_name", type=str, default="edgepair_grasp_10ep_failure_diagnostic")
    ap.add_argument("--split_role", type=str, default="failure_diagnostic")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.raw.exists():
        raise SystemExit(f"missing raw dataset: {args.raw}")

    data = {k: np.asarray(v) for k, v in np.load(args.raw, allow_pickle=False).items()}
    rows = int(data["episode_index"].shape[0]) if "episode_index" in data else 0
    episode_index = _as_float_array(data, "episode_index", np.nan).reshape(-1).astype(np.int64) if rows else np.zeros((0,), dtype=np.int64)
    success_label = _as_float_array(data, "success_label", 0.0).reshape(-1) > 0.5
    verified_mask = (
        _as_float_array(data, "teacher_grasp_verified", 0.0).reshape(-1) > 0.5
    ) | (
        _as_float_array(data, "teacher_attached_after_close", 0.0).reshape(-1) > 0.5
    ) | (
        _as_float_array(data, "verified_lift", 0.0).reshape(-1) > 0.5
    )
    close_ready = _as_float_array(data, "teacher_close_ready_all", 0.0).reshape(-1) > 0.5
    close_contact_ready = _as_float_array(data, "teacher_close_contact_ready", 0.0).reshape(-1) > 0.5
    close_attempt = _as_str_list(data, "teacher_close_action")
    close_failure = _as_str_list(data, "teacher_close_failure_reason")
    motion_phase = _as_str_list(data, "teacher_motion_phase")
    tc_motion_phase = _as_str_list(data, "teacher_tc_motion_phase")
    stage_bucket = _as_str_list(data, "stage_bucket")
    target_edge_pair_index = _as_float_array(data, "teacher_grasp_commit_edge_pair_index", -1).reshape(-1).astype(np.int64)
    target_edge_pair_family = _as_float_array(data, "teacher_grasp_commit_edge_pair_family", -1).reshape(-1).astype(np.int64)
    target_edge_pair_yaw_error = _as_float_array(data, "teacher_grasp_commit_edge_pair_yaw_error", np.nan).reshape(-1)

    out = {k: np.asarray(v) for k, v in data.items()}
    out["split_role"] = np.asarray([args.split_role] * rows)
    out["split_name"] = np.asarray([args.split_name] * rows)
    out["split_source"] = np.asarray([args.raw.stem] * rows)
    out["split_is_diagnostic"] = np.asarray([1.0] * rows, dtype=np.float32)
    out["split_verified_mask"] = verified_mask.astype(np.float32)
    out["split_success_mask"] = success_label.astype(np.float32)
    out["split_close_ready_mask"] = close_ready.astype(np.float32)
    out["split_close_contact_ready_mask"] = close_contact_ready.astype(np.float32)
    out["split_teacher_close_action"] = np.asarray(close_attempt)
    out["split_teacher_close_failure_reason"] = np.asarray(close_failure)
    out["split_teacher_motion_phase"] = np.asarray(motion_phase)
    out["split_teacher_tc_motion_phase"] = np.asarray(tc_motion_phase)
    out["split_stage_bucket"] = np.asarray(stage_bucket)
    out["split_teacher_edge_pair_index"] = target_edge_pair_index.astype(np.int64)
    out["split_teacher_edge_pair_family"] = target_edge_pair_family.astype(np.int64)
    out["split_teacher_edge_pair_yaw_error"] = target_edge_pair_yaw_error.astype(np.float32)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    report = {
        "output_npz": str(args.output_npz),
        "raw_source": str(args.raw),
        "split_name": args.split_name,
        "split_role": args.split_role,
        "rows": int(rows),
        "episode_count": int(len(np.unique(episode_index))) if rows else 0,
        "success_label_rate": float(success_label.mean()) if rows else 0.0,
        "verified_rate": float(verified_mask.mean()) if rows else 0.0,
        "close_ready_rate": float(close_ready.mean()) if rows else 0.0,
        "close_contact_ready_rate": float(close_contact_ready.mean()) if rows else 0.0,
        "teacher_motion_phase_counts": dict(Counter(motion_phase)),
        "teacher_tc_motion_phase_counts": dict(Counter(tc_motion_phase)),
        "stage_bucket_counts": dict(Counter(stage_bucket)),
        "teacher_close_action_counts": dict(Counter(close_attempt)),
        "teacher_close_failure_reason_counts": dict(Counter(close_failure)),
        "teacher_edge_pair_index_counts": dict(Counter(target_edge_pair_index.tolist())),
        "teacher_edge_pair_family_counts": dict(Counter(target_edge_pair_family.tolist())),
        "teacher_edge_pair_yaw_error_stats": _stats(target_edge_pair_yaw_error),
        "diagnostic_note": (
            "This split is failure-only / diagnostic-only. It is intentionally excluded "
            "from student positive imitation."
        ),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
