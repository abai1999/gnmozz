"""
Audit teacher/student distillation chain for phase-1 alignment.

This script summarizes three potential failure points:
1. runtime input mismatch: proxy delta vs teacher delta
2. label sparsity / optimism gap around near-ready windows
3. dataset retention: what the near-ready builder keeps vs what support rows contain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_stat(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _episode_coverage(episode_index: np.ndarray, mask: np.ndarray) -> int:
    if episode_index is None or episode_index.size == 0:
        return 0
    if not np.any(mask):
        return 0
    return int(np.unique(episode_index[mask]).size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--near_ready_npz", default=None)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_md", default=None)
    ap.add_argument("--strict_namespace", action="store_true")
    ap.add_argument("--near_xy_mult", type=float, default=3.0)
    ap.add_argument("--near_z_mult", type=float, default=1.5)
    ap.add_argument("--near_yaw_mult", type=float, default=1.5)
    args = ap.parse_args()

    support = np.load(args.support_npz, allow_pickle=False)
    support_data = {k: support[k] for k in support.files}

    if args.strict_namespace:
        required = [
            "proxy_current_delta_basin_target",
            "teacher_current_delta_basin_target",
            "runtime_handoff_metric_xy_error",
            "runtime_handoff_metric_abs_z_error",
            "runtime_handoff_metric_yaw_error",
            "runtime_handoff_metric_valid",
            "runtime_handoff_ready_pred",
            "runtime_handoff_ready_applied",
        ]
        missing = [k for k in required if k not in support_data]
        if missing:
            raise RuntimeError(f"strict namespace audit missing required fields: {missing}")

    proxy_delta = np.asarray(
        support_data["proxy_current_delta_basin_target"] if args.strict_namespace else support_data.get(
            "proxy_current_delta_basin_target", support_data.get("current_delta_basin_target")
        ),
        dtype=np.float64,
    )
    teacher_delta = np.asarray(
        support_data["teacher_current_delta_basin_target"] if args.strict_namespace else support_data.get(
            "teacher_current_delta_basin_target", support_data.get("target_delta_teacher", proxy_delta)
        ),
        dtype=np.float64,
    )
    episode_index = (
        np.asarray(support_data["episode_index"], dtype=np.int64)
        if "episode_index" in support_data
        else np.zeros((proxy_delta.shape[0],), dtype=np.int64)
    )
    phase_id = (
        np.asarray(support_data["phase_id"], dtype=np.int64)
        if "phase_id" in support_data
        else np.ones((proxy_delta.shape[0],), dtype=np.int64)
    )
    gripper_open = np.asarray(
        support_data.get("rollout_gripper_open", np.ones((proxy_delta.shape[0],), dtype=np.float32)),
        dtype=np.float64,
    )
    teacher_xy = np.asarray(support_data.get("teacher_truth_handoff_metric_xy_error"), dtype=np.float64)
    teacher_z = np.asarray(support_data.get("teacher_truth_handoff_metric_abs_z_error"), dtype=np.float64)
    teacher_yaw = np.asarray(support_data.get("teacher_truth_handoff_metric_yaw_error"), dtype=np.float64)
    runtime_xy = np.asarray(
        support_data["runtime_handoff_metric_xy_error"] if args.strict_namespace else support_data.get(
            "runtime_handoff_metric_xy_error", support_data.get("handoff_metric_xy_error", np.full_like(teacher_xy, np.nan))
        ),
        dtype=np.float64,
    )
    runtime_z = np.asarray(
        support_data["runtime_handoff_metric_abs_z_error"] if args.strict_namespace else support_data.get(
            "runtime_handoff_metric_abs_z_error", support_data.get("handoff_metric_abs_z_error", np.full_like(teacher_z, np.nan))
        ),
        dtype=np.float64,
    )
    runtime_yaw = np.asarray(
        support_data["runtime_handoff_metric_yaw_error"] if args.strict_namespace else support_data.get(
            "runtime_handoff_metric_yaw_error", support_data.get("handoff_metric_yaw_error", np.full_like(teacher_yaw, np.nan))
        ),
        dtype=np.float64,
    )
    runtime_valid = np.asarray(
        support_data["runtime_handoff_metric_valid"] if args.strict_namespace else support_data.get(
            "runtime_handoff_metric_valid", np.isfinite(runtime_xy).astype(np.float32)
        ),
        dtype=np.float64,
    ) > 0.5
    teacher_ready = np.asarray(
        support_data.get("teacher_truth_handoff_ready", np.zeros((proxy_delta.shape[0],), dtype=np.float32)),
        dtype=np.float64,
    ) > 0.5
    runtime_ready = np.asarray(
        support_data["runtime_handoff_ready_pred"] if args.strict_namespace else support_data.get(
            "runtime_handoff_ready_pred",
            support_data.get("runtime_handoff_ready", support_data.get("ready_to_close_target", np.zeros((proxy_delta.shape[0],), dtype=np.float32))),
        ),
        dtype=np.float64,
    ) > 0.5
    runtime_ready_applied = np.asarray(
        support_data["runtime_handoff_ready_applied"] if args.strict_namespace else support_data.get(
            "runtime_handoff_ready_applied", np.zeros((proxy_delta.shape[0],), dtype=np.float32)
        ),
        dtype=np.float64,
    ) > 0.5
    rel_xy = np.asarray(support_data.get("teacher_truth_handoff_release_threshold_xy_error"), dtype=np.float64)
    rel_z = np.asarray(support_data.get("teacher_truth_handoff_release_threshold_abs_z_error"), dtype=np.float64)
    rel_yaw = np.asarray(support_data.get("teacher_truth_handoff_release_threshold_yaw_error"), dtype=np.float64)

    base_mask = (phase_id == 1) & (gripper_open >= 0.5)
    valid_metric_mask = (
        base_mask
        & np.isfinite(teacher_xy)
        & np.isfinite(teacher_z)
        & np.isfinite(teacher_yaw)
        & np.isfinite(rel_xy)
        & np.isfinite(rel_z)
        & np.isfinite(rel_yaw)
    )
    near_ready_mask = (
        valid_metric_mask
        & (teacher_xy <= args.near_xy_mult * rel_xy)
        & (teacher_z <= args.near_z_mult * rel_z)
        & (teacher_yaw <= args.near_yaw_mult * rel_yaw)
    )
    very_near_xyyaw_mask = (
        valid_metric_mask
        & (teacher_z <= 1.2 * rel_z)
        & ((teacher_xy > rel_xy) | (teacher_yaw > rel_yaw))
    )
    proxy_teacher_gap = np.linalg.norm(proxy_delta - teacher_delta, axis=1)
    optimism_xy = np.maximum(teacher_xy - runtime_xy, 0.0)
    optimism_yaw = np.maximum(teacher_yaw - runtime_yaw, 0.0)
    finite_runtime_mask = runtime_valid & np.isfinite(runtime_xy) & np.isfinite(runtime_z) & np.isfinite(runtime_yaw)
    runtime_release_ready = finite_runtime_mask & (runtime_xy <= rel_xy) & (runtime_z <= rel_z) & (runtime_yaw <= rel_yaw)
    disagreement_rows = finite_runtime_mask & (runtime_release_ready != teacher_ready)

    report = {
        "support_npz": str(Path(args.support_npz).resolve()),
        "support_rows": int(proxy_delta.shape[0]),
        "support_open_phase1_rows": int(base_mask.sum()),
        "separation": {
            "all": _safe_stat(proxy_teacher_gap),
            "near_ready": _safe_stat(proxy_teacher_gap[near_ready_mask]),
            "eq_ratio": float(np.mean(np.all(np.isclose(proxy_delta, teacher_delta), axis=1))),
        },
        "teacher_runtime_alignment": {
            "teacher_ready_pos": int(teacher_ready.sum()),
            "runtime_ready_pos": int(runtime_ready.sum()),
            "runtime_ready_applied_pos": int(runtime_ready_applied.sum()),
            "runtime_metric_valid_pos": int(np.sum(finite_runtime_mask)),
            "near_ready_rows": int(near_ready_mask.sum()),
            "near_ready_episode_coverage": _episode_coverage(episode_index, near_ready_mask),
            "very_near_xyyaw_rows": int(very_near_xyyaw_mask.sum()),
            "teacher_xy_error_mm": _safe_stat(teacher_xy[near_ready_mask] * 1000.0),
            "teacher_abs_z_error_mm": _safe_stat(teacher_z[near_ready_mask] * 1000.0),
            "teacher_yaw_error_deg": _safe_stat(np.rad2deg(teacher_yaw[near_ready_mask])),
            "optimism_gap_xy_mm": _safe_stat(optimism_xy[near_ready_mask & finite_runtime_mask] * 1000.0),
            "optimism_gap_yaw_deg": _safe_stat(np.rad2deg(optimism_yaw[near_ready_mask & finite_runtime_mask])),
            "disagreement_rows": int(np.sum(disagreement_rows)),
            "false_ready_rows": int(np.sum(finite_runtime_mask & runtime_release_ready & ~teacher_ready)),
        },
    }

    if args.near_ready_npz:
        near = np.load(args.near_ready_npz, allow_pickle=False)
        near_data = {k: near[k] for k in near.files}
        near_teacher_ready = np.asarray(
            near_data.get("teacher_truth_handoff_ready", np.zeros((near_data["candidate_actions_local"].shape[0],), dtype=np.float32)),
            dtype=np.float64,
        ) > 0.5
        near_runtime_ready = np.asarray(
            near_data.get("ready_to_close_target", np.zeros((near_data["candidate_actions_local"].shape[0],), dtype=np.float32)),
            dtype=np.float64,
        ) > 0.5
        near_episode_index = (
            np.asarray(near_data["episode_index"], dtype=np.int64)
            if "episode_index" in near_data
            else np.zeros((near_data["candidate_actions_local"].shape[0],), dtype=np.int64)
        )
        report["near_ready_dataset"] = {
            "path": str(Path(args.near_ready_npz).resolve()),
            "rows": int(near_data["candidate_actions_local"].shape[0]),
            "teacher_ready_pos": int(near_teacher_ready.sum()),
            "runtime_ready_pos": int(near_runtime_ready.sum()),
            "episode_coverage": int(np.unique(near_episode_index).size),
            "xy_focus_pos": int(np.sum(np.asarray(near_data.get("xy_focus", 0)) > 0.5)),
            "near_xy_hard_pos": int(np.sum(np.asarray(near_data.get("near_xy_hard", 0)) > 0.5)),
            "near_yaw_hard_pos": int(np.sum(np.asarray(near_data.get("near_yaw_hard", 0)) > 0.5)),
            "near_coupled_pos": int(np.sum(np.asarray(near_data.get("near_coupled", 0)) > 0.5)),
            "ready_support_pos": int(np.sum(np.asarray(near_data.get("ready_support", 0)) > 0.5)),
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))

    if args.output_md:
        md = []
        md.append("# Distillation Audit")
        md.append("")
        md.append(f"- support rows: `{report['support_rows']}`")
        md.append(f"- phase1 open rows: `{report['support_open_phase1_rows']}`")
        md.append(f"- near-ready rows: `{report['teacher_runtime_alignment']['near_ready_rows']}`")
        md.append(f"- near-ready episode coverage: `{report['teacher_runtime_alignment']['near_ready_episode_coverage']}`")
        md.append(f"- teacher ready positives: `{report['teacher_runtime_alignment']['teacher_ready_pos']}`")
        md.append(f"- runtime ready positives: `{report['teacher_runtime_alignment']['runtime_ready_pos']}`")
        md.append("")
        md.append("## Separation")
        md.append("")
        md.append(f"- proxy/teacher eq ratio: `{report['separation']['eq_ratio']:.4f}`")
        md.append(f"- proxy-teacher gap mean: `{report['separation']['all']['mean']}`")
        md.append(f"- proxy-teacher gap near-ready mean: `{report['separation']['near_ready']['mean']}`")
        md.append("")
        md.append("## Near-Ready")
        md.append("")
        md.append(f"- very-near xy/yaw rows: `{report['teacher_runtime_alignment']['very_near_xyyaw_rows']}`")
        md.append(f"- optimism gap xy mean mm: `{report['teacher_runtime_alignment']['optimism_gap_xy_mm']['mean']}`")
        md.append(f"- optimism gap yaw mean deg: `{report['teacher_runtime_alignment']['optimism_gap_yaw_deg']['mean']}`")
        if "near_ready_dataset" in report:
            md.append("")
            md.append("## Retention")
            md.append("")
            md.append(f"- near-ready dataset rows: `{report['near_ready_dataset']['rows']}`")
            md.append(f"- near-ready dataset teacher-ready positives: `{report['near_ready_dataset']['teacher_ready_pos']}`")
            md.append(f"- near-ready dataset ready-support positives: `{report['near_ready_dataset']['ready_support_pos']}`")
            md.append(f"- near-ready dataset near-xy-hard: `{report['near_ready_dataset']['near_xy_hard_pos']}`")
            md.append(f"- near-ready dataset near-yaw-hard: `{report['near_ready_dataset']['near_yaw_hard_pos']}`")
        Path(args.output_md).write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
