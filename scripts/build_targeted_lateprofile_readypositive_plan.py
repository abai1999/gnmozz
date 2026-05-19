"""
Build a late-profile recollection plan that hard-anchors A episodes from
previous late-profile rollouts which actually produced teacher-ready rows.

This avoids selecting A_teacher_ready only from mainline data and then losing
all positives after switching to late-profile collection.
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


def _pick_with_quota(candidates: list[dict], selected: set[int], quota: int, out: list[int]) -> None:
    if quota <= 0:
        return
    for row in candidates:
        if len(out) >= quota:
            return
        ep = int(row["episode_index"])
        if ep in selected:
            continue
        selected.add(ep)
        out.append(ep)


def _build_rows(
    support_npz: str,
    phase_id: int,
    open_threshold: float,
    ready_window_radius: int,
    very_near_window_radius: int,
    close_early_window_radius: int,
    release_near_factor: float,
):
    raw = np.load(support_npz, allow_pickle=False)
    data = {k: np.asarray(raw[k]) for k in raw.files}

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

    base = (phase == int(phase_id)) & (gripper_open >= float(open_threshold))
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
    release_near = (
        finite
        & (teacher_z <= float(release_near_factor) * rel_z)
        & (teacher_yaw <= float(release_near_factor) * rel_yaw)
    )
    broad_xy_recovery = release_near & (teacher_xy > rel_xy)

    episodes = sorted(np.unique(ep).tolist())
    per_episode = []
    windows = []

    for e in episodes:
        m = base & (ep == e)
        if not np.any(m):
            continue
        steps = step[m]
        ready_m = teacher_ready[m]
        close_m = planner_close[m]
        release_m = release[m]
        very_near_m = very_near[m]
        broad_xy_m = broad_xy_recovery[m]

        first_ready = _first_step(ready_m, steps)
        first_close = _first_step(close_m, steps)
        first_release = _first_step(release_m, steps)
        close_minus_ready = None if (first_ready is None or first_close is None) else int(first_close - first_ready)
        close_early = bool(close_minus_ready is not None and close_minus_ready < 0)
        close_without_ready = bool(first_close is not None and first_ready is None)

        tags: list[str] = []
        if np.any(ready_m):
            tags.append("A_teacher_ready")
            for s in steps[ready_m]:
                _append_window(windows, e, int(s), int(ready_window_radius), "A_ready_window")
        if np.any(broad_xy_m):
            tags.append("B_broad_xy_recovery")
            for s in steps[broad_xy_m]:
                _append_window(windows, e, int(s), int(very_near_window_radius), "B_broad_xy_recovery")
        if close_early or close_without_ready:
            tags.append("C_close_early")
            if first_close is not None:
                _append_window(windows, e, int(first_close), int(close_early_window_radius), "C_close_early")
        if not tags:
            continue

        per_episode.append(
            {
                "episode_index": int(e),
                "tags": tags,
                "teacher_ready_rows": int(np.sum(ready_m)),
                "release_rows": int(np.sum(release_m)),
                "very_near_rows": int(np.sum(very_near_m)),
                "broad_xy_recovery_rows": int(np.sum(broad_xy_m)),
                "close_rows": int(np.sum(close_m)),
                "first_ready_step": first_ready,
                "first_release_step": first_release,
                "first_close_step": first_close,
                "close_minus_first_ready": close_minus_ready,
                "close_early": bool(close_early),
                "close_without_ready": bool(close_without_ready),
                "score_a_teacher_ready": float(np.sum(ready_m)),
                "score_b_broad_xy_recovery": float(np.sum(broad_xy_m)),
                "score_c_close_early": 1.0 + (1.0 if close_early else 0.0),
            }
        )
    return per_episode, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_support_npz", required=True, help="late-profile support npz with actual teacher-ready positives")
    ap.add_argument("--primary_support_npz", required=True, help="new late-profile support npz for broad xy / close-early mining")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_windows_jsonl", required=True)
    ap.add_argument("--ready_window_radius", type=int, default=10)
    ap.add_argument("--very_near_window_radius", type=int, default=8)
    ap.add_argument("--close_early_window_radius", type=int, default=8)
    ap.add_argument("--phase_id", type=int, default=1)
    ap.add_argument("--open_threshold", type=float, default=0.5)
    ap.add_argument("--max_episodes", type=int, default=16)
    ap.add_argument("--quota_anchor_ready", type=int, default=4)
    ap.add_argument("--quota_b_broad_xy_recovery", type=int, default=8)
    ap.add_argument("--quota_c_close_early", type=int, default=4)
    ap.add_argument("--release_near_factor", type=float, default=1.2)
    args = ap.parse_args()

    anchor_rows, anchor_windows = _build_rows(
        args.anchor_support_npz,
        args.phase_id,
        args.open_threshold,
        args.ready_window_radius,
        args.very_near_window_radius,
        args.close_early_window_radius,
        args.release_near_factor,
    )
    primary_rows, primary_windows = _build_rows(
        args.primary_support_npz,
        args.phase_id,
        args.open_threshold,
        args.ready_window_radius,
        args.very_near_window_radius,
        args.close_early_window_radius,
        args.release_near_factor,
    )

    anchor_a = [r for r in anchor_rows if "A_teacher_ready" in r["tags"]]
    anchor_a.sort(key=lambda r: (-float(r["score_a_teacher_ready"]), int(r["episode_index"])))
    primary_b = [r for r in primary_rows if "B_broad_xy_recovery" in r["tags"]]
    primary_b.sort(key=lambda r: (-float(r["score_b_broad_xy_recovery"]), int(r["episode_index"])))
    primary_c = [r for r in primary_rows if "C_close_early" in r["tags"]]
    primary_c.sort(key=lambda r: (-float(r["score_c_close_early"]), int(r["episode_index"])))

    selected_set: set[int] = set()
    selected: list[int] = []
    _pick_with_quota(anchor_a, selected_set, int(args.quota_anchor_ready), selected)
    _pick_with_quota(primary_b, selected_set, int(args.quota_b_broad_xy_recovery), selected)
    _pick_with_quota(primary_c, selected_set, int(args.quota_c_close_early), selected)

    all_rows = anchor_rows + [r for r in primary_rows if int(r["episode_index"]) not in {int(x["episode_index"]) for x in anchor_rows}]
    all_rows.sort(
        key=lambda r: (
            -float(r.get("score_a_teacher_ready", 0.0)),
            -float(r.get("score_b_broad_xy_recovery", 0.0)),
            -float(r.get("score_c_close_early", 0.0)),
            int(r["episode_index"]),
        )
    )
    for row in all_rows:
        if len(selected) >= int(args.max_episodes):
            break
        ep = int(row["episode_index"])
        if ep in selected_set:
            continue
        selected_set.add(ep)
        selected.append(ep)

    selected = selected[: int(args.max_episodes)]
    selected_rows = []
    row_by_ep = {}
    for row in anchor_rows + primary_rows:
        row_by_ep.setdefault(int(row["episode_index"]), row)
    for ep in selected:
        if ep in row_by_ep:
            selected_rows.append(row_by_ep[ep])

    windows = []
    for w in anchor_windows + primary_windows:
        if int(w["episode_index"]) in selected_set:
            windows.append(w)

    out = {
        "anchor_support_npz": str(Path(args.anchor_support_npz).resolve()),
        "primary_support_npz": str(Path(args.primary_support_npz).resolve()),
        "selected_episode_count": int(len(selected)),
        "selected_episode_indices": [int(x) for x in selected],
        "selected_episode_indices_csv": ",".join(str(int(x)) for x in selected),
        "selected_rows": selected_rows,
        "window_count": int(len(windows)),
        "quota": {
            "max_episodes": int(args.max_episodes),
            "quota_anchor_ready": int(args.quota_anchor_ready),
            "quota_b_broad_xy_recovery": int(args.quota_b_broad_xy_recovery),
            "quota_c_close_early": int(args.quota_c_close_early),
        },
        "selected_tag_coverage": {
            "episodes_with_anchor_ready": int(sum("A_teacher_ready" in r["tags"] for r in selected_rows)),
            "episodes_with_B_xy_recovery": int(sum("B_broad_xy_recovery" in r["tags"] for r in selected_rows)),
            "episodes_with_C_close_early": int(sum("C_close_early" in r["tags"] for r in selected_rows)),
        },
        "notes": {
            "selection_policy": "anchor-ready from actual late-profile positives first, then primary late-profile B/C fill",
            "anchor_ready_meaning": "episode actually produced teacher-ready rows under late-profile collection",
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2))
    windows_path = Path(args.output_windows_jsonl)
    windows_path.write_text("".join(json.dumps(w) + "\n" for w in windows))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

