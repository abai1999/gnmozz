#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTION_FIELDS = {
    "base_action": {"kind": "action", "frame_flag": None, "priority": "baseline"},
    "planner_base_action_local_raw": {"kind": "action", "frame_flag": "planner_base_action_frame_is_local", "priority": "baseline"},
    "executed_action_local": {"kind": "action", "frame_flag": None, "priority": "executed"},
    "executed_motion_local": {"kind": "action", "frame_flag": None, "priority": "executed"},
    "oracle_action_local": {"kind": "action", "frame_flag": "oracle_action_frame_is_local", "priority": "teacher"},
    "residual_label_local": {"kind": "residual", "frame_flag": None, "priority": "teacher"},
    "residual_label_world": {"kind": "residual", "frame_flag": None, "priority": "teacher"},
    "target_delta_teacher": {"kind": "state_delta", "frame_flag": None, "priority": "teacher"},
    "teacher_current_delta_basin_target": {"kind": "state_delta", "frame_flag": None, "priority": "teacher"},
    "candidate_actions_local": {"kind": "candidate_bank", "frame_flag": None, "priority": "teacher"},
    "candidate_next_basin_distance": {"kind": "candidate_score", "frame_flag": None, "priority": "teacher"},
    "candidate_improvement": {"kind": "candidate_score", "frame_flag": None, "priority": "teacher"},
    "candidate_oracle_score": {"kind": "candidate_score", "frame_flag": None, "priority": "teacher"},
    "candidate_scores": {"kind": "candidate_score", "frame_flag": None, "priority": "runtime"},
    "candidate_probs": {"kind": "candidate_score", "frame_flag": None, "priority": "runtime"},
    "best_candidate_index": {"kind": "candidate_label", "frame_flag": None, "priority": "teacher"},
    "oracle_candidate_index": {"kind": "candidate_label", "frame_flag": None, "priority": "teacher"},
    "runtime_selected_candidate_index": {"kind": "candidate_label", "frame_flag": None, "priority": "runtime"},
    "pred_candidate_index": {"kind": "candidate_label", "frame_flag": None, "priority": "runtime"},
}


def _numeric_stats(arr: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(arr)
    out: dict[str, float | int] = {
        "finite_ratio": float(np.mean(finite)) if arr.size else 0.0,
    }
    if arr.size and np.any(finite):
        vals = arr[finite]
        out.update(
            {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p50": float(np.percentile(vals, 50.0)),
                "p90": float(np.percentile(vals, 90.0)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
    return out


def _action_stats(arr: np.ndarray) -> dict[str, float | int]:
    stats: dict[str, float | int] = {}
    if arr.ndim == 2 and arr.shape[1] >= 6:
        xyz = np.linalg.norm(arr[:, :3], axis=1)
        yaw = np.abs(arr[:, 5])
        stats["xyz_norm_mean"] = float(np.nanmean(xyz))
        stats["xyz_norm_p90"] = float(np.nanpercentile(xyz, 90.0))
        stats["yaw_abs_mean"] = float(np.nanmean(yaw))
        stats["yaw_abs_p90"] = float(np.nanpercentile(yaw, 90.0))
        stats["yaw_nonzero_ratio"] = float(np.mean(np.isfinite(yaw) & (yaw > 1e-5)))
    elif arr.ndim == 3 and arr.shape[2] >= 6:
        xyz = np.linalg.norm(arr[:, :, :3], axis=2)
        yaw = np.abs(arr[:, :, 5])
        stats["xyz_norm_mean"] = float(np.nanmean(xyz))
        stats["xyz_norm_p90"] = float(np.nanpercentile(xyz, 90.0))
        stats["yaw_abs_mean"] = float(np.nanmean(yaw))
        stats["yaw_abs_p90"] = float(np.nanpercentile(yaw, 90.0))
        stats["yaw_nonzero_ratio"] = float(np.mean(np.isfinite(yaw) & (yaw > 1e-5)))
    return stats


def audit_npz(path: Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    report: dict[str, object] = {
        "path": str(path),
        "rows": int(next(iter(raw.values())).shape[0]) if raw.files else 0,
        "fields": {},
        "conclusions": {},
    }
    fields: dict[str, object] = {}
    for name, meta in ACTION_FIELDS.items():
        if name not in raw.files:
            continue
        arr = np.asarray(raw[name])
        item: dict[str, object] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "kind": meta["kind"],
            "priority": meta["priority"],
        }
        if np.issubdtype(arr.dtype, np.number):
            item.update(_numeric_stats(arr.astype(np.float32, copy=False)))
            item.update(_action_stats(arr.astype(np.float32, copy=False)))
        if meta["frame_flag"] and meta["frame_flag"] in raw.files:
            flag = np.asarray(raw[meta["frame_flag"]], dtype=np.float32)
            item["frame_flag"] = meta["frame_flag"]
            item["frame_local_ratio"] = float(np.mean(flag > 0.5))
        fields[name] = item
    report["fields"] = fields

    conclusions = {
        "has_teacher_action_local": "oracle_action_local" in raw.files,
        "has_executed_action_local": "executed_action_local" in raw.files,
        "has_residual_label_local": "residual_label_local" in raw.files,
        "has_candidate_bank": "candidate_actions_local" in raw.files,
        "has_teacher_candidate_label": ("oracle_candidate_index" in raw.files or "best_candidate_index" in raw.files),
        "has_teacher_candidate_score": ("candidate_oracle_score" in raw.files or "candidate_improvement" in raw.files),
        "recommended_target_mode": "candidate_ranking"
        if ("candidate_actions_local" in raw.files and ("candidate_oracle_score" in raw.files or "oracle_candidate_index" in raw.files))
        else ("continuous_residual" if "oracle_action_local" in raw.files and "planner_base_action_local_raw" in raw.files else "insufficient"),
    }
    if "oracle_action_local" in raw.files and "planner_base_action_local_raw" in raw.files:
        oracle = np.asarray(raw["oracle_action_local"], dtype=np.float32)
        base = np.asarray(raw["planner_base_action_local_raw"], dtype=np.float32)
        if oracle.shape == base.shape and oracle.ndim == 2 and oracle.shape[1] >= 6:
            resid = oracle[:, :6] - base[:, :6]
            conclusions["teacher_minus_baseline_xyz_norm_mean"] = float(np.mean(np.linalg.norm(resid[:, :3], axis=1)))
            conclusions["teacher_minus_baseline_yaw_abs_mean"] = float(np.mean(np.abs(resid[:, 5])))
    report["conclusions"] = conclusions
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    reports = [audit_npz(Path(p)) for p in args.input_npz]
    merged = {
        "reports": reports,
        "recommended_next_step": "build_v4_candidate_ranking_dataset"
        if any(r["conclusions"]["recommended_target_mode"] == "candidate_ranking" for r in reports)
        else "build_v4_continuous_residual_dataset",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
