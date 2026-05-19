#!/usr/bin/env python3
"""Build a phase1-only grasp bridge repair dataset from verified teacher raw rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ALLOWED_BUCKETS = {"near_contact_refine", "micro_contact_refine", "close_basin_verified"}
STAGE_BUCKET_ID = {
    "near_alignment": 0,
    "near_contact_refine": 0,
    "micro_contact_refine": 1,
    "close_basin_verified": 2,
    "broad_near": 7,
    "unknown": 8,
}


def _scalar_at(data, key: str, idx: int, default: float = 0.0) -> float:
    if key not in data:
        return float(default)
    arr = np.asarray(data[key][idx], dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _rgb_chw_at(data, key: str, idx: int) -> np.ndarray:
    if key not in data:
        return np.zeros((3, 96, 96), dtype=np.float32)
    arr = np.asarray(data[key][idx])
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr


def _teacher_contact_repr(data, idx: int, target_delta: np.ndarray, verified_positive: bool) -> np.ndarray:
    contact = np.zeros((8,), dtype=np.float32)
    contact[0] = float(target_delta[0])
    contact[1] = float(target_delta[1])
    contact[2] = float(target_delta[2])
    contact[3] = float(target_delta[5])
    contact[4] = _scalar_at(data, "teacher_object_in_finger_region", idx, 0.0)
    contact[5] = max(
        _scalar_at(data, "teacher_grasp_contact_ready", idx, 0.0),
        _scalar_at(data, "teacher_close_contact_ready", idx, 0.0),
    )
    contact[6] = max(
        _scalar_at(data, "teacher_attached_after_close", idx, 0.0),
        _scalar_at(data, "teacher_grasp_verified", idx, 0.0),
        _scalar_at(data, "verified_lift", idx, 0.0),
    )
    contact[7] = float(verified_positive)
    return contact


def _yaw_direction_label(action4: np.ndarray) -> int:
    yaw = float(np.asarray(action4, dtype=np.float32).reshape(-1)[3])
    if yaw > 1e-4:
        return 2
    if yaw < -1e-4:
        return 0
    return 1


def _residual_traj_4d(data, idx: int, horizon: int) -> np.ndarray:
    if "teacher_residual_trajectory_4d" in data:
        arr = np.asarray(data["teacher_residual_trajectory_4d"][idx], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[-1] >= 4:
            arr = arr[:, :4]
            if arr.shape[0] < horizon:
                pad = np.repeat(arr[-1:, :], horizon - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            return arr[:horizon]
    action4 = np.asarray(data["teacher_residual_action_4d"][idx], dtype=np.float32).reshape(-1)[:4]
    return np.repeat(action4[None, :], horizon, axis=0).astype(np.float32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_raw", type=Path, required=True)
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n_pre", type=int, default=8)
    ap.add_argument("--n_align", type=int, default=16)
    ap.add_argument("--n_post", type=int, default=8)
    ap.add_argument("--max_negative_ratio", type=float, default=0.30)
    ap.add_argument("--min_rows", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()
    data = {k: np.asarray(v) for k, v in np.load(args.phase1_raw, allow_pickle=False).items()}
    n = int(data["episode_index"].shape[0])

    by_episode: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if str(np.asarray(data["alignment_phase"][i]).item()) != "grasp_commit":
            continue
        by_episode[int(np.asarray(data["episode_index"][i]).reshape(-1)[0])].append(i)

    positive_rows: list[dict] = []
    negative_rows: list[dict] = []
    bucket_hist = Counter()
    phase_hist = Counter()
    bridge_type_hist = Counter()
    bridge_episode_rows = Counter()
    corridor_depth = []
    corridor_force = []
    corridor_action = []

    for ep, indices in by_episode.items():
        indices = sorted(indices, key=lambda i: int(np.asarray(data["step_index"][i]).reshape(-1)[0]))
        verified_step = -1
        takeover_step = -1
        phase_success = False
        episode_attached = False
        episode_verified = False
        for i in indices:
            step = int(np.asarray(data["step_index"][i]).reshape(-1)[0])
            if takeover_step < 0 and _scalar_at(data, "takeover_active", i, 0.0) > 0.5:
                takeover_step = step
            episode_attached = episode_attached or (_scalar_at(data, "teacher_attached_after_close", i, 0.0) > 0.5)
            episode_verified = episode_verified or (
                _scalar_at(data, "verified_lift", i, 0.0) > 0.5
                or _scalar_at(data, "teacher_grasp_verified", i, 0.0) > 0.5
                or _scalar_at(data, "success_label", i, 0.0) > 0.5
            )
            verified_now = max(
                _scalar_at(data, "teacher_grasp_verified", i, 0.0),
                _scalar_at(data, "verified_lift", i, 0.0),
                _scalar_at(data, "teacher_attached_after_close", i, 0.0),
                _scalar_at(data, "success_label", i, 0.0),
            ) > 0.5
            if verified_now:
                verified_step = step if verified_step < 0 else min(verified_step, step)
                phase_success = True
        if takeover_step < 0:
            takeover_step = int(np.asarray(data["step_index"][indices[0]]).reshape(-1)[0])
        if not phase_success or verified_step < 0:
            verified_step = -1

        align_start = max(takeover_step - int(args.n_pre), 0)
        bridge_start = max(verified_step - int(args.n_align), align_start) if verified_step >= 0 else 10**9
        bridge_end = verified_step + int(args.n_post) if verified_step >= 0 else -1

        for i in indices:
            step = int(np.asarray(data["step_index"][i]).reshape(-1)[0])
            stage_bucket = str(np.asarray(data["stage_bucket"][i]).item())
            target_delta = np.asarray(data["teacher_target_delta_local_6d"][i], dtype=np.float32).reshape(-1)[:6]
            if not np.all(np.isfinite(target_delta)):
                continue
            teacher_residual_action_4d = np.asarray(data["teacher_residual_action_4d"][i], dtype=np.float32).reshape(-1)[:4]

            valid_chain = bool(
                phase_success
                and episode_attached
                and episode_verified
                and _scalar_at(data, "invalid_action", i, 0.0) <= 0.0
                and _scalar_at(data, "workspace_violation", i, 0.0) <= 0.0
            )
            in_align_window = bool(align_start <= step < bridge_start)
            in_bridge_window = bool(verified_step >= 0 and bridge_start <= step <= bridge_end)
            verified_positive = bool(phase_success and valid_chain and stage_bucket in ALLOWED_BUCKETS and (in_align_window or in_bridge_window))

            teacher_progress = np.asarray(
                [
                    _scalar_at(data, "teacher_improves_xy", i, 0.0),
                    _scalar_at(data, "teacher_improves_z", i, 0.0),
                    _scalar_at(data, "teacher_improves_yaw", i, 0.0),
                ],
                dtype=np.float32,
            )
            teacher_risk = float(
                max(
                    _scalar_at(data, "risk_label", i, 0.0),
                    _scalar_at(data, "invalid_action", i, 0.0),
                    float(_scalar_at(data, "workspace_violation", i, 0.0) > 1e-6),
                    _scalar_at(data, "force_spike", i, 0.0),
                )
            )
            teacher_stop = float(
                max(
                    _scalar_at(data, "stop_label", i, 0.0),
                    0.0 if np.linalg.norm(teacher_residual_action_4d[:3]) > 1e-6 or abs(float(teacher_residual_action_4d[3])) > 1e-4 else 1.0,
                )
            )
            yaw_enabled = float(_scalar_at(data, "yaw_imitation_enabled", i, 1.0))
            teacher_conf = 1.0 if verified_positive else (0.5 if phase_success else 0.25)
            if _scalar_at(data, "is_occluded", i, 0.0) > 0.5 or _scalar_at(data, "is_low_visibility", i, 0.0) > 0.5:
                teacher_conf *= 0.5

            sample_weight = 0.0
            bridge_type = "negative"
            if verified_positive:
                if in_bridge_window:
                    sample_weight = 4.0 if stage_bucket == "close_basin_verified" else 3.5
                    bridge_type = "bridge-positive"
                else:
                    sample_weight = 2.0 if stage_bucket == "micro_contact_refine" else 1.5
                    bridge_type = "align-positive"
            elif teacher_risk > 0.5 or teacher_stop > 0.5 or stage_bucket not in ALLOWED_BUCKETS:
                sample_weight = 0.35
                bridge_type = "risk-negative"

            row = {
                "front_rgb": _rgb_chw_at(data, "front_rgb", i),
                "wrist_rgb": _rgb_chw_at(data, "wrist_rgb", i),
                "wrist_depth": np.asarray(data["wrist_depth"][i], dtype=np.float32),
                "force_history": np.asarray(data["force_history"][i], dtype=np.float32),
                "proprio": np.asarray(data["proprio"][i], dtype=np.float32),
                "planner_action_local": np.asarray(data["planner_action_local"][i], dtype=np.float32).reshape(-1)[:6],
                "gripper_context": np.asarray(data["gripper_context"][i], dtype=np.float32).reshape(-1)[:4],
                "teacher_target_delta_local_6d": target_delta.astype(np.float32),
                "teacher_contact_repr": _teacher_contact_repr(data, i, target_delta, verified_positive),
                "teacher_residual_action_4d": teacher_residual_action_4d.astype(np.float32),
                "teacher_residual_trajectory_4d": _residual_traj_4d(data, i, int(args.horizon)),
                "teacher_progress_label": teacher_progress.astype(np.float32),
                "teacher_risk_label": np.asarray(teacher_risk, dtype=np.float32),
                "teacher_stop_label": np.asarray(teacher_stop, dtype=np.float32),
                "teacher_confidence_label": np.asarray(teacher_conf, dtype=np.float32),
                "sample_weight": np.asarray(sample_weight, dtype=np.float32),
                "phase_id": np.asarray(0, dtype=np.int64),
                "stage_bucket_id": np.asarray(STAGE_BUCKET_ID.get(stage_bucket, STAGE_BUCKET_ID["unknown"]), dtype=np.int64),
                "yaw_direction_label": np.asarray(_yaw_direction_label(teacher_residual_action_4d), dtype=np.int64),
                "yaw_imitation_enabled": np.asarray(yaw_enabled, dtype=np.float32),
                "verified_positive": np.asarray(float(verified_positive), dtype=np.float32),
                "stage_bucket": np.asarray(stage_bucket),
                "alignment_phase": np.asarray("grasp_commit"),
                "source_name": np.asarray(args.phase1_raw.stem),
            }

            if verified_positive:
                positive_rows.append(row)
                bucket_hist[stage_bucket] += 1
                phase_hist["grasp_commit"] += 1
                bridge_type_hist[bridge_type] += 1
                bridge_episode_rows[ep] += 1
                corridor_depth.append(_scalar_at(data, "depth_proximity", i, np.nan))
                corridor_force.append(_scalar_at(data, "force_norm", i, 0.0))
                corridor_action.append(float(np.linalg.norm(np.asarray(row["planner_action_local"], dtype=np.float32)[:3])))
            elif sample_weight > 0.0:
                negative_rows.append(row)

    if not positive_rows:
        raise SystemExit("no phase1 bridge positive rows found")

    max_negative = int(len(positive_rows) * float(args.max_negative_ratio))
    negative_rows = negative_rows[:max_negative]
    rows = positive_rows + negative_rows
    if len(rows) < int(args.min_rows):
        raise SystemExit(f"too few phase1 bridge rows: {len(rows)} < {args.min_rows}")

    keys = sorted({k for row in rows for k in row.keys()})
    out = {k: np.asarray([row[k] for row in rows]) for k in keys}
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    def _stats(arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
        }

    report = {
        "output_npz": str(args.output_npz),
        "rows": int(len(rows)),
        "positive_rows": int(len(positive_rows)),
        "negative_rows": int(len(negative_rows)),
        "phase_histogram": {k: int(v) for k, v in phase_hist.items()},
        "bucket_histogram": {k: int(v) for k, v in bucket_hist.items()},
        "bridge_type_histogram": {k: int(v) for k, v in bridge_type_hist.items()},
        "positive_episode_rows": {f"ep{ep}": int(v) for ep, v in bridge_episode_rows.items()},
        "runtime_corridor": {
            "grasp_commit": {
                "depth_p90": _stats(corridor_depth)["p90"],
                "force_p90": _stats(corridor_force)["p90"],
                "planner_action_pos_norm_p90": _stats(corridor_action)["p90"],
            }
        },
        "defaults": {
            "n_pre": int(args.n_pre),
            "n_align": int(args.n_align),
            "n_post": int(args.n_post),
            "max_negative_ratio": float(args.max_negative_ratio),
            "horizon": int(args.horizon),
            "allowed_buckets": sorted(ALLOWED_BUCKETS),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
