"""
Audit close timing vs teacher-ready windows on support rows.

This answers:
- how much earlier/later planner close intent appears relative to teacher-ready
- whether ready windows are extremely narrow
- whether ready-like rows were filtered out by the open-phase constraint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_count(mask: np.ndarray) -> int:
    return int(np.sum(np.asarray(mask, dtype=bool)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    arr = np.load(args.support_npz, allow_pickle=False)
    data = {k: np.asarray(arr[k]) for k in arr.files}

    required = [
        "episode_index",
        "rollout_step",
        "planner_close_intent",
        "teacher_truth_handoff_ready",
        "teacher_truth_handoff_metric_xy_error",
        "teacher_truth_handoff_metric_abs_z_error",
        "teacher_truth_handoff_metric_yaw_error",
        "teacher_truth_handoff_release_threshold_xy_error",
        "teacher_truth_handoff_release_threshold_abs_z_error",
        "teacher_truth_handoff_release_threshold_yaw_error",
        "rollout_gripper_open",
        "phase_id",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"missing required fields: {missing}")

    episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    rollout_step = np.asarray(data["rollout_step"], dtype=np.int64)
    planner_close = np.asarray(data["planner_close_intent"], dtype=np.float32) > 0.5
    teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    gripper_open = np.asarray(data["rollout_gripper_open"], dtype=np.float32)
    phase_id = np.asarray(data["phase_id"], dtype=np.int64)

    teacher_xy = np.asarray(data["teacher_truth_handoff_metric_xy_error"], dtype=np.float64)
    teacher_z = np.asarray(data["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float64)
    teacher_yaw = np.asarray(data["teacher_truth_handoff_metric_yaw_error"], dtype=np.float64)
    rel_xy = np.asarray(data["teacher_truth_handoff_release_threshold_xy_error"], dtype=np.float64)
    rel_z = np.asarray(data["teacher_truth_handoff_release_threshold_abs_z_error"], dtype=np.float64)
    rel_yaw = np.asarray(data["teacher_truth_handoff_release_threshold_yaw_error"], dtype=np.float64)

    valid_geom = (
        np.isfinite(teacher_xy)
        & np.isfinite(teacher_z)
        & np.isfinite(teacher_yaw)
        & np.isfinite(rel_xy)
        & np.isfinite(rel_z)
        & np.isfinite(rel_yaw)
    )
    release_geom = (
        valid_geom
        & (teacher_xy <= rel_xy)
        & (teacher_z <= rel_z)
        & (teacher_yaw <= rel_yaw)
    )
    very_near = (
        valid_geom
        & (teacher_xy <= 1.5 * rel_xy)
        & (teacher_z <= 1.5 * rel_z)
        & (teacher_yaw <= 1.5 * rel_yaw)
    )

    per_episode = []
    close_minus_ready = []
    ready_window_lengths = []
    close_inside_ready_window = 0
    ready_open_filtered_rows = 0

    for ep in np.unique(episode_index):
        m = episode_index == ep
        steps = rollout_step[m]
        order = np.argsort(steps)
        steps = steps[order]
        ready_ep = teacher_ready[m][order]
        close_ep = planner_close[m][order]
        open_ep = gripper_open[m][order]
        release_ep = release_geom[m][order]
        very_near_ep = very_near[m][order]

        ready_steps = steps[ready_ep]
        close_steps = steps[close_ep]
        release_steps = steps[release_ep]
        very_near_steps = steps[very_near_ep]

        first_ready = int(ready_steps[0]) if ready_steps.size else None
        last_ready = int(ready_steps[-1]) if ready_steps.size else None
        first_close = int(close_steps[0]) if close_steps.size else None
        first_release = int(release_steps[0]) if release_steps.size else None
        first_very_near = int(very_near_steps[0]) if very_near_steps.size else None
        ready_window_len = int(ready_steps.size)
        very_near_len = int(very_near_steps.size)

        if first_ready is not None and first_close is not None:
            close_minus_ready.append(int(first_close - first_ready))
        if ready_window_len > 0:
            ready_window_lengths.append(ready_window_len)
            if first_close is not None and first_ready <= first_close <= last_ready:
                close_inside_ready_window += 1

        ready_open_filtered_rows += _safe_count(release_ep & (open_ep < 0.5))

        per_episode.append(
            {
                "episode_index": int(ep),
                "first_teacher_ready_step": first_ready,
                "last_teacher_ready_step": last_ready,
                "teacher_ready_rows": ready_window_len,
                "first_release_geom_step": first_release,
                "release_geom_rows": int(release_steps.size),
                "first_very_near_step": first_very_near,
                "very_near_rows": very_near_len,
                "first_planner_close_intent_step": first_close,
                "planner_close_rows": int(close_steps.size),
                "close_minus_first_ready": None if (first_ready is None or first_close is None) else int(first_close - first_ready),
            }
        )

    report = {
        "support_npz": str(Path(args.support_npz).resolve()),
        "rows": int(len(episode_index)),
        "phase1_rows": int(np.sum(phase_id == 1)),
        "teacher_ready_rows": int(np.sum(teacher_ready)),
        "release_geom_rows": int(np.sum(release_geom)),
        "very_near_rows": int(np.sum(very_near)),
        "planner_close_rows": int(np.sum(planner_close)),
        "episodes": int(np.unique(episode_index).size),
        "episodes_with_teacher_ready": int(sum(1 for row in per_episode if row["teacher_ready_rows"] > 0)),
        "episodes_with_planner_close": int(sum(1 for row in per_episode if row["planner_close_rows"] > 0)),
        "episodes_close_inside_ready_window": int(close_inside_ready_window),
        "close_minus_ready_step_stats": None
        if not close_minus_ready
        else {
            "count": int(len(close_minus_ready)),
            "mean": float(np.mean(close_minus_ready)),
            "median": float(np.median(close_minus_ready)),
            "min": int(np.min(close_minus_ready)),
            "max": int(np.max(close_minus_ready)),
        },
        "teacher_ready_window_len_stats": None
        if not ready_window_lengths
        else {
            "count": int(len(ready_window_lengths)),
            "mean": float(np.mean(ready_window_lengths)),
            "median": float(np.median(ready_window_lengths)),
            "min": int(np.min(ready_window_lengths)),
            "max": int(np.max(ready_window_lengths)),
            "narrow_leq_2": int(np.sum(np.asarray(ready_window_lengths) <= 2)),
        },
        "release_geom_rows_blocked_by_gripper_open_filter": int(ready_open_filtered_rows),
        "per_episode": per_episode,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
