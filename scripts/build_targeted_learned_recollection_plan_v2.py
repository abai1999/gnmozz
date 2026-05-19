"""
Build a targeted learned recollection plan (v2) with explicit per-bucket quotas.

Compared to v1 (score-topK), this script forces coverage for broad xy-recovery
episodes so recollection is not dominated by A_teacher_ready + C_close_early.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_windows_jsonl", required=True)
    ap.add_argument("--ready_window_radius", type=int, default=10)
    ap.add_argument("--very_near_window_radius", type=int, default=8)
    ap.add_argument("--close_early_window_radius", type=int, default=8)
    ap.add_argument("--phase_id", type=int, default=1)
    ap.add_argument("--open_threshold", type=float, default=0.5)
    ap.add_argument("--max_episodes", type=int, default=16)
    ap.add_argument("--quota_b_broad_xy_recovery", type=int, default=10)
    ap.add_argument("--quota_a_teacher_ready", type=int, default=2)
    ap.add_argument("--quota_c_close_early", type=int, default=4)
    ap.add_argument(
        "--release_near_factor",
        type=float,
        default=1.2,
        help="near-release factor for z/yaw when building broad xy recovery windows",
    )
    ap.add_argument(
        "--broad_xy_upper_factor",
        type=float,
        default=0.0,
        help="upper clip factor for xy in broad xy recovery mask; <=0 disables upper clip",
    )
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
    release_near = (
        finite
        & (teacher_z <= float(args.release_near_factor) * rel_z)
        & (teacher_yaw <= float(args.release_near_factor) * rel_yaw)
    )
    if float(args.broad_xy_upper_factor) > 0.0:
        release_near = release_near & (
            teacher_xy <= float(args.broad_xy_upper_factor) * rel_xy
        )
    broad_xy_recovery = release_near & (teacher_xy > rel_xy)

    episodes = sorted(np.unique(ep).tolist())
    per_episode: list[dict] = []
    windows: list[dict] = []

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
        broad_xy_recovery_m = broad_xy_recovery[m]

        first_ready = _first_step(ready_m, steps)
        first_close = _first_step(close_m, steps)
        first_release = _first_step(release_m, steps)
        close_minus_ready = None if (first_ready is None or first_close is None) else int(first_close - first_ready)
        close_early = bool(close_minus_ready is not None and close_minus_ready < 0)
        close_without_ready = bool(first_close is not None and first_ready is None)

        has_teacher_ready = bool(np.any(ready_m))
        has_near_xy_hard = bool(np.any(near_xy_hard_m))
        has_broad_xy_recovery = bool(np.any(broad_xy_recovery_m))
        has_c = bool(close_early or close_without_ready)

        tags: list[str] = []
        if has_teacher_ready:
            tags.append("A_teacher_ready")
            for s in steps[ready_m]:
                _append_window(windows, e, int(s), int(args.ready_window_radius), "A_ready_window")
        if has_broad_xy_recovery:
            tags.append("B_broad_xy_recovery")
            for s in steps[broad_xy_recovery_m]:
                _append_window(windows, e, int(s), int(args.very_near_window_radius), "B_broad_xy_recovery")
        elif has_near_xy_hard:
            # Fallback legacy B definition in case broad mask is unexpectedly empty.
            tags.append("B_near_xy_hard")
            for s in steps[near_xy_hard_m]:
                _append_window(windows, e, int(s), int(args.very_near_window_radius), "B_near_xy_hard")
        if has_c:
            tags.append("C_close_early")
            if first_close is not None:
                _append_window(windows, e, int(first_close), int(args.close_early_window_radius), "C_close_early")
        if not tags:
            continue

        score_b = float(np.sum(broad_xy_recovery_m))
        score_a = float(np.sum(ready_m))
        score_c = 1.0 + (1.0 if close_early else 0.0)

        per_episode.append(
            {
                "episode_index": int(e),
                "tags": tags,
                "score_b_near_xy_hard": score_b,
                "score_a_teacher_ready": score_a,
                "score_c_close_early": score_c,
                "teacher_ready_rows": int(np.sum(ready_m)),
                "release_rows": int(np.sum(release_m)),
                "very_near_rows": int(np.sum(very_near_m)),
                "near_xy_hard_rows": int(np.sum(near_xy_hard_m)),
                "broad_xy_recovery_rows": int(np.sum(broad_xy_recovery_m)),
                "close_rows": int(np.sum(close_m)),
                "first_ready_step": first_ready,
                "first_release_step": first_release,
                "first_close_step": first_close,
                "close_minus_first_ready": close_minus_ready,
                "close_early": bool(close_early),
                "close_without_ready": bool(close_without_ready),
            }
        )

    b_rows = [
        r
        for r in per_episode
        if ("B_broad_xy_recovery" in r["tags"] or "B_near_xy_hard" in r["tags"])
    ]
    a_rows = [r for r in per_episode if "A_teacher_ready" in r["tags"]]
    c_rows = [r for r in per_episode if "C_close_early" in r["tags"]]

    b_rows.sort(key=lambda r: (-float(r["score_b_near_xy_hard"]), int(r["episode_index"])))
    a_rows.sort(key=lambda r: (-float(r["score_a_teacher_ready"]), int(r["episode_index"])))
    c_rows.sort(key=lambda r: (-float(r["score_c_close_early"]), int(r["episode_index"])))

    selected_set: set[int] = set()
    selected: list[int] = []

    _pick_with_quota(b_rows, selected_set, int(args.quota_b_broad_xy_recovery), selected)
    _pick_with_quota(a_rows, selected_set, int(args.quota_a_teacher_ready), selected)
    _pick_with_quota(c_rows, selected_set, int(args.quota_c_close_early), selected)

    max_eps = max(int(args.max_episodes), 1)
    if len(selected) < max_eps:
        all_rows = sorted(
            per_episode,
            key=lambda r: (
                -float(r["score_b_near_xy_hard"]),
                -float(r["score_a_teacher_ready"]),
                -float(r["score_c_close_early"]),
                int(r["episode_index"]),
            ),
        )
        for row in all_rows:
            if len(selected) >= max_eps:
                break
            ep_idx = int(row["episode_index"])
            if ep_idx in selected_set:
                continue
            selected_set.add(ep_idx)
            selected.append(ep_idx)

    selected = selected[:max_eps]

    selected_rows = [r for r in per_episode if int(r["episode_index"]) in selected_set]
    selected_rows.sort(
        key=lambda r: (
            -float(r["score_b_near_xy_hard"]),
            -float(r["score_a_teacher_ready"]),
            -float(r["score_c_close_early"]),
            int(r["episode_index"]),
        )
    )

    # de-dup windows for selected episodes
    dedup = {}
    for w in windows:
        if int(w["episode_index"]) not in selected_set:
            continue
        key = (int(w["episode_index"]), int(w["window_start"]), int(w["window_end"]), str(w["tag"]))
        dedup[key] = w
    selected_windows = list(dedup.values())
    selected_windows.sort(key=lambda w: (int(w["episode_index"]), int(w["window_start"]), str(w["tag"])))

    selected_b = sum(
        1
        for r in selected_rows
        if ("B_broad_xy_recovery" in r["tags"] or "B_near_xy_hard" in r["tags"])
    )
    selected_a = sum(1 for r in selected_rows if "A_teacher_ready" in r["tags"])
    selected_c = sum(1 for r in selected_rows if "C_close_early" in r["tags"])

    report = {
        "support_npz": str(Path(args.support_npz).resolve()),
        "selected_episode_count": int(len(selected)),
        "selected_episode_indices": selected,
        "selected_episode_indices_csv": ",".join(str(x) for x in selected),
        "selected_rows": selected_rows,
        "window_count": int(len(selected_windows)),
        "quota": {
            "max_episodes": int(args.max_episodes),
            "quota_b_broad_xy_recovery": int(args.quota_b_broad_xy_recovery),
            "quota_a_teacher_ready": int(args.quota_a_teacher_ready),
            "quota_c_close_early": int(args.quota_c_close_early),
        },
        "selected_tag_coverage": {
            "episodes_with_B_xy_recovery": int(selected_b),
            "episodes_with_A_teacher_ready": int(selected_a),
            "episodes_with_C_close_early": int(selected_c),
        },
        "notes": {
            "selection_policy": "quota-first (B -> A -> C), then fill by B/A/C scores",
            "A_teacher_ready": "episodes containing teacher_ready rows",
            "B_broad_xy_recovery": "z/yaw in-band-or-near, xy still outside release (primary)",
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
