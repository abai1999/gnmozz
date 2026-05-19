"""
Build a targeted learned recollection plan from support rows.

Focus windows:
- A: episodes containing teacher-ready frames
- B: episodes with very-near frames but xy still out-of-band
- C: episodes where planner close intent is early relative to teacher-ready
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _first_step(mask: np.ndarray, steps: np.ndarray):
    if not np.any(mask):
        return None
    return int(np.min(steps[mask]))


def _append_window(collector: list[dict], ep: int, center_step: int, radius: int, tag: str):
    collector.append(
        {
            "episode_index": int(ep),
            "window_start": int(max(0, center_step - max(radius, 0))),
            "window_end": int(center_step + max(radius, 0)),
            "center_step": int(center_step),
            "tag": str(tag),
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_windows_jsonl", required=True)
    ap.add_argument("--ready_window_radius", type=int, default=10)
    ap.add_argument("--very_near_window_radius", type=int, default=8)
    ap.add_argument("--close_early_window_radius", type=int, default=8)
    ap.add_argument("--max_episodes", type=int, default=16)
    ap.add_argument("--phase_id", type=int, default=1)
    ap.add_argument("--open_threshold", type=float, default=0.5)
    args = ap.parse_args()

    raw = np.load(args.support_npz, allow_pickle=False)
    data = {k: np.asarray(raw[k]) for k in raw.files}

    required = [
        "episode_index",
        "rollout_step",
        "phase_id",
        "rollout_gripper_open",
        "teacher_truth_handoff_ready",
        "teacher_truth_handoff_metric_xy_error",
        "teacher_truth_handoff_metric_abs_z_error",
        "teacher_truth_handoff_metric_yaw_error",
        "teacher_truth_handoff_release_threshold_xy_error",
        "teacher_truth_handoff_release_threshold_abs_z_error",
        "teacher_truth_handoff_release_threshold_yaw_error",
        "planner_close_intent",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"missing required fields: {missing}")

    ep = np.asarray(data["episode_index"], dtype=np.int64)
    step = np.asarray(data["rollout_step"], dtype=np.int64)
    phase = np.asarray(data["phase_id"], dtype=np.int64)
    gripper_open = np.asarray(data["rollout_gripper_open"], dtype=np.float32)
    teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    planner_close = np.asarray(data["planner_close_intent"], dtype=np.float32) > 0.5

    teacher_xy = np.asarray(data["teacher_truth_handoff_metric_xy_error"], dtype=np.float64)
    teacher_z = np.asarray(data["teacher_truth_handoff_metric_abs_z_error"], dtype=np.float64)
    teacher_yaw = np.asarray(data["teacher_truth_handoff_metric_yaw_error"], dtype=np.float64)
    rel_xy = np.asarray(data["teacher_truth_handoff_release_threshold_xy_error"], dtype=np.float64)
    rel_z = np.asarray(data["teacher_truth_handoff_release_threshold_abs_z_error"], dtype=np.float64)
    rel_yaw = np.asarray(data["teacher_truth_handoff_release_threshold_yaw_error"], dtype=np.float64)

    base = (phase == int(args.phase_id)) & (gripper_open >= float(args.open_threshold))
    finite = (
        np.isfinite(teacher_xy)
        & np.isfinite(teacher_z)
        & np.isfinite(teacher_yaw)
        & np.isfinite(rel_xy)
        & np.isfinite(rel_z)
        & np.isfinite(rel_yaw)
    )
    release = (
        finite
        & (teacher_xy <= rel_xy)
        & (teacher_z <= rel_z)
        & (teacher_yaw <= rel_yaw)
    )
    very_near = (
        finite
        & (teacher_xy <= 1.5 * rel_xy)
        & (teacher_z <= 1.5 * rel_z)
        & (teacher_yaw <= 1.5 * rel_yaw)
    )
    near_xy_hard = very_near & (teacher_xy > rel_xy)

    episodes = sorted(np.unique(ep).tolist())
    per_episode = []
    windows: list[dict] = []
    scored = []

    for e in episodes:
        m = base & (ep == e)
        if not np.any(m):
            continue
        steps = step[m]
        ready_m = teacher_ready[m]
        close_m = planner_close[m]
        release_m = release[m]
        very_near_m = very_near[m]
        near_xy_hard_m = near_xy_hard[m]

        first_ready = _first_step(ready_m, steps)
        first_close = _first_step(close_m, steps)
        first_release = _first_step(release_m, steps)
        close_minus_ready = None if (first_ready is None or first_close is None) else int(first_close - first_ready)
        close_early = bool(close_minus_ready is not None and close_minus_ready < 0)
        close_without_ready = bool(first_close is not None and first_ready is None)

        has_teacher_ready = bool(np.any(ready_m))
        has_near_xy_hard = bool(np.any(near_xy_hard_m))

        score = 0.0
        tags = []

        if has_teacher_ready:
            score += 8.0
            tags.append("A_teacher_ready")
            ready_steps = steps[ready_m]
            for s in ready_steps:
                _append_window(windows, e, int(s), int(args.ready_window_radius), "A_ready_window")

        if has_near_xy_hard:
            score += 4.0 + 0.01 * float(np.sum(near_xy_hard_m))
            tags.append("B_near_xy_hard")
            nh_steps = steps[near_xy_hard_m]
            for s in nh_steps:
                _append_window(windows, e, int(s), int(args.very_near_window_radius), "B_near_xy_hard")

        if close_early or close_without_ready:
            score += 3.0
            tags.append("C_close_early")
            if first_close is not None:
                _append_window(windows, e, int(first_close), int(args.close_early_window_radius), "C_close_early")

        if not tags:
            continue

        row = {
            "episode_index": int(e),
            "tags": tags,
            "score": float(score),
            "teacher_ready_rows": int(np.sum(ready_m)),
            "release_rows": int(np.sum(release_m)),
            "very_near_rows": int(np.sum(very_near_m)),
            "near_xy_hard_rows": int(np.sum(near_xy_hard_m)),
            "close_rows": int(np.sum(close_m)),
            "first_ready_step": first_ready,
            "first_release_step": first_release,
            "first_close_step": first_close,
            "close_minus_first_ready": close_minus_ready,
            "close_early": bool(close_early),
            "close_without_ready": bool(close_without_ready),
        }
        per_episode.append(row)
        scored.append((score, int(e)))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [ep_idx for _, ep_idx in scored[: max(int(args.max_episodes), 1)]]
    selected_set = set(selected)
    selected_rows = [r for r in per_episode if int(r["episode_index"]) in selected_set]
    selected_rows.sort(key=lambda r: (-float(r["score"]), int(r["episode_index"])))

    # De-duplicate windows for selected episodes.
    dedup = {}
    for w in windows:
        if int(w["episode_index"]) not in selected_set:
            continue
        key = (int(w["episode_index"]), int(w["window_start"]), int(w["window_end"]), str(w["tag"]))
        dedup[key] = w
    selected_windows = list(dedup.values())
    selected_windows.sort(key=lambda w: (int(w["episode_index"]), int(w["window_start"]), str(w["tag"])))

    report = {
        "support_npz": str(Path(args.support_npz).resolve()),
        "selected_episode_count": int(len(selected)),
        "selected_episode_indices": selected,
        "selected_episode_indices_csv": ",".join(str(x) for x in selected),
        "selected_rows": selected_rows,
        "window_count": int(len(selected_windows)),
        "notes": {
            "A_teacher_ready": "episodes containing teacher_ready rows",
            "B_near_xy_hard": "very-near rows where xy is still outside release threshold",
            "C_close_early": "planner close appears before teacher-ready or without teacher-ready",
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    out_jsonl = Path(args.output_windows_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for row in selected_windows:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
