#!/usr/bin/env python3
"""Build verified-only phase1/phase2 student vNext dataset.

This builder merges planner-tail phase1 grasp teacher rows and phase2 insert
teacher rows into a single non-privileged student dataset. Positive action
imitation comes only from verified-success teacher windows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PHASE_ID = {"grasp_commit": 0, "insert_commit": 1, "commit_target": 2, "unknown": 3}
STAGE_BUCKET_ID = {
    "near_alignment": 0,
    "near_contact_refine": 0,
    "micro_contact_refine": 1,
    "close_basin_verified": 2,
    "insert_broad_near": 3,
    "insert_near_align": 4,
    "insert_precommit_micro": 5,
    "insert_commit_verified": 6,
    "broad_near": 7,
    "unknown": 8,
}


def _scalar_at(data, key: str, idx: int, default: float = 0.0) -> float:
    if key not in data:
        return float(default)
    arr = np.asarray(data[key][idx], dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _string_at(data, key: str, idx: int, default: str = "") -> str:
    if key not in data:
        return default
    return str(np.asarray(data[key][idx]).item())


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


def _teacher_contact_repr(data, idx: int, phase_name: str, target_delta: np.ndarray, phase_success: bool) -> np.ndarray:
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
    if phase_name == "insert_commit":
        contact[6] = _scalar_at(data, "success_label", idx, 0.0)
    else:
        contact[6] = max(
            _scalar_at(data, "teacher_attached_after_close", idx, 0.0),
            _scalar_at(data, "teacher_grasp_verified", idx, 0.0),
            _scalar_at(data, "verified_lift", idx, 0.0),
        )
    contact[7] = float(phase_success)
    return contact


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


def _yaw_direction_label(action4: np.ndarray) -> int:
    yaw = float(np.asarray(action4, dtype=np.float32).reshape(-1)[3])
    if yaw > 1e-4:
        return 2
    if yaw < -1e-4:
        return 0
    return 1


def _stats(arr: np.ndarray) -> dict:
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


def _threshold_at(data, idx: int, *keys: str, default: float) -> float:
    for key in keys:
        if key not in data:
            continue
        val = _scalar_at(data, key, idx, np.nan)
        if np.isfinite(val) and float(val) > 0.0:
            return float(val)
    return float(default)


def _close_ready_score(
    target_delta: np.ndarray,
    *,
    xy_threshold: float,
    z_threshold: float,
    yaw_threshold: float,
    exact_ready: bool,
) -> float:
    if exact_ready:
        return 1.0
    norm_xy = float(np.linalg.norm(target_delta[:2]) / max(float(xy_threshold), 1e-6))
    norm_z = float(abs(float(target_delta[2])) / max(float(z_threshold), 1e-6))
    norm_yaw = float(abs(float(target_delta[5])) / max(float(yaw_threshold), 1e-6))
    return float(1.0 / (1.0 + max(norm_xy, norm_z, norm_yaw)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_raw", type=Path, nargs="+", required=True)
    ap.add_argument("--phase2_raw", type=Path, nargs="*", default=[])
    ap.add_argument("--output_npz", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--verified_window_steps", type=int, default=16)
    ap.add_argument("--pre_context_steps", type=int, default=8)
    ap.add_argument("--post_context_steps", type=int, default=8)
    ap.add_argument("--close_ready_pre_window_steps", type=int, default=8)
    ap.add_argument("--close_ready_post_window_steps", type=int, default=4)
    ap.add_argument("--close_ready_bridge_weight", type=float, default=2.5)
    ap.add_argument("--close_ready_exact_weight", type=float, default=4.0)
    ap.add_argument("--handoff_ready_weight_boost", type=float, default=2.0)
    ap.add_argument("--include_close_bridge_rows", action="store_true", default=True)
    ap.add_argument("--no_include_close_bridge_rows", dest="include_close_bridge_rows", action="store_false")
    ap.add_argument("--max_negative_ratio", type=float, default=0.30)
    ap.add_argument("--min_rows", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()
    sources = list(args.phase1_raw) + list(args.phase2_raw)
    datasets: list[tuple[str, dict[str, np.ndarray]]] = []
    for path in sources:
        if not path.exists():
            raise SystemExit(f"missing raw dataset: {path}")
        datasets.append((path.stem, {k: np.asarray(v) for k, v in np.load(path, allow_pickle=False).items()}))

    positive_rows: list[dict] = []
    negative_rows: list[dict] = []
    bucket_hist = Counter()
    phase_hist = Counter()
    positive_episode_rows = Counter()
    phase_success_counts = Counter()
    close_bridge_episode_rows = Counter()
    close_bridge_step_counts = Counter()
    corridor_depth = defaultdict(list)
    corridor_force = defaultdict(list)
    corridor_action = defaultdict(list)

    for source_name, data in datasets:
        n = int(data["episode_index"].shape[0])
        by_episode: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            by_episode[int(np.asarray(data["episode_index"][i]).reshape(-1)[0])].append(i)

        successful_episodes: dict[int, bool] = {}
        success_step_by_episode: dict[int, int] = {}
        close_ready_step_by_episode: dict[int, int] = {}
        for ep, indices in by_episode.items():
            success_mask = np.asarray([_scalar_at(data, "success_label", i, 0.0) > 0.5 for i in indices], dtype=bool)
            successful_episodes[ep] = bool(success_mask.any())
            success_step_by_episode[ep] = int(max((_scalar_at(data, "step_index", i, -1) for i in indices if _scalar_at(data, "success_label", i, 0.0) > 0.5), default=-1))
            close_ready_step = -1
            for i in sorted(indices, key=lambda j: _scalar_at(data, "step_index", j, -1)):
                target_delta = np.asarray(data["teacher_target_delta_local_6d"][i], dtype=np.float32).reshape(-1)[:6]
                if not np.all(np.isfinite(target_delta)):
                    continue
                xy_thr = _threshold_at(
                    data,
                    i,
                    "teacher_close_xy_threshold",
                    "teacher_truth_handoff_release_threshold_xy_error",
                    default=0.006,
                )
                z_thr = _threshold_at(
                    data,
                    i,
                    "teacher_close_abs_z_threshold",
                    "teacher_truth_handoff_release_threshold_abs_z_error",
                    default=0.005,
                )
                yaw_thr = _threshold_at(
                    data,
                    i,
                    "teacher_close_yaw_threshold",
                    "teacher_truth_handoff_release_threshold_yaw_error",
                    default=0.12,
                )
                threshold_ready = bool(
                    np.linalg.norm(target_delta[:2]) <= xy_thr
                    and abs(float(target_delta[2])) <= z_thr
                    and abs(float(target_delta[5])) <= yaw_thr
                )
                raw_ready = bool(
                    _scalar_at(data, "teacher_close_ready_all", i, 0.0) > 0.5
                    or _scalar_at(data, "teacher_close_ready", i, 0.0) > 0.5
                    or _scalar_at(data, "teacher_truth_handoff_ready", i, 0.0) > 0.5
                )
                if threshold_ready or raw_ready:
                    close_ready_step = int(_scalar_at(data, "step_index", i, -1))
                    break
            close_ready_step_by_episode[ep] = int(close_ready_step)

        for i in range(n):
            ep = int(np.asarray(data["episode_index"][i]).reshape(-1)[0])
            step = int(np.asarray(data["step_index"][i]).reshape(-1)[0])
            phase_name = _string_at(data, "alignment_phase", i, "unknown")
            if phase_name not in {"grasp_commit", "insert_commit"}:
                continue
            stage_bucket = _string_at(data, "stage_bucket", i, "unknown")
            target_delta = np.asarray(data["teacher_target_delta_local_6d"][i], dtype=np.float32).reshape(-1)[:6]
            if not np.all(np.isfinite(target_delta)):
                continue

            phase_success = bool(successful_episodes.get(ep, False))
            verified_positive = False
            if phase_name == "grasp_commit":
                verified_positive = bool(
                    phase_success
                    and _scalar_at(data, "action_imitation_weight", i, 0.0) > 0.5
                    and max(
                        _scalar_at(data, "teacher_grasp_verified", i, 0.0),
                        _scalar_at(data, "teacher_attached_after_close", i, 0.0),
                        _scalar_at(data, "success_label", i, 0.0),
                    )
                    > 0.5
                )
            else:
                verified_positive = bool(
                    phase_success
                    and _scalar_at(data, "action_imitation_weight", i, 0.0) > 0.5
                )

            success_step = success_step_by_episode.get(ep, -1)
            in_verified_window = bool(success_step >= 0 and step >= success_step - int(args.verified_window_steps))
            close_ready_step = close_ready_step_by_episode.get(ep, -1)
            bridge_window = bool(
                bool(args.include_close_bridge_rows)
                and phase_success
                and phase_name == "grasp_commit"
                and close_ready_step >= 0
                and step >= close_ready_step - int(args.close_ready_pre_window_steps)
                and step <= close_ready_step + int(args.close_ready_post_window_steps)
                and _scalar_at(data, "rollout_gripper_open", i, 1.0) > 0.5
            )
            exact_close_ready = bool(
                (
                    np.linalg.norm(target_delta[:2]) <= _threshold_at(
                        data,
                        i,
                        "teacher_close_xy_threshold",
                        "teacher_truth_handoff_release_threshold_xy_error",
                        default=0.006,
                    )
                    and abs(float(target_delta[2]))
                    <= _threshold_at(
                        data,
                        i,
                        "teacher_close_abs_z_threshold",
                        "teacher_truth_handoff_release_threshold_abs_z_error",
                        default=0.005,
                    )
                    and abs(float(target_delta[5]))
                    <= _threshold_at(
                        data,
                        i,
                        "teacher_close_yaw_threshold",
                        "teacher_truth_handoff_release_threshold_yaw_error",
                        default=0.12,
                    )
                )
                or _scalar_at(data, "teacher_close_ready_all", i, 0.0) > 0.5
                or _scalar_at(data, "teacher_close_ready", i, 0.0) > 0.5
                or _scalar_at(data, "teacher_truth_handoff_ready", i, 0.0) > 0.5
            )
            close_ready_score = _close_ready_score(
                target_delta,
                xy_threshold=_threshold_at(
                    data,
                    i,
                    "teacher_close_xy_threshold",
                    "teacher_truth_handoff_release_threshold_xy_error",
                    default=0.006,
                ),
                z_threshold=_threshold_at(
                    data,
                    i,
                    "teacher_close_abs_z_threshold",
                    "teacher_truth_handoff_release_threshold_abs_z_error",
                    default=0.005,
                ),
                yaw_threshold=_threshold_at(
                    data,
                    i,
                    "teacher_close_yaw_threshold",
                    "teacher_truth_handoff_release_threshold_yaw_error",
                    default=0.12,
                ),
                exact_ready=exact_close_ready,
            )

            teacher_residual_action_4d = np.asarray(data["teacher_residual_action_4d"][i], dtype=np.float32).reshape(-1)[:4]
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

            sample_weight = 1.0
            if verified_positive:
                sample_weight = 4.0 if in_verified_window else 2.0
                if "micro" in stage_bucket:
                    sample_weight += 1.0
            elif bridge_window:
                sample_weight = float(args.close_ready_bridge_weight)
                if exact_close_ready:
                    sample_weight = max(sample_weight, float(args.close_ready_exact_weight))
                if _scalar_at(data, "teacher_truth_handoff_ready", i, 0.0) > 0.5:
                    sample_weight = max(sample_weight, float(args.handoff_ready_weight_boost))
            elif phase_success:
                sample_weight = 0.75
            elif teacher_risk > 0.5 or teacher_stop > 0.5:
                sample_weight = 0.35

            row = {
                "front_rgb": _rgb_chw_at(data, "front_rgb", i),
                "wrist_rgb": _rgb_chw_at(data, "wrist_rgb", i),
                "wrist_depth": np.asarray(data["wrist_depth"][i], dtype=np.float32),
                "force_history": np.asarray(data["force_history"][i], dtype=np.float32),
                "proprio": np.asarray(data["proprio"][i], dtype=np.float32),
                "planner_action_local": np.asarray(data["planner_action_local"][i], dtype=np.float32).reshape(-1)[:6],
                "gripper_context": np.asarray(data["gripper_context"][i], dtype=np.float32).reshape(-1)[:4],
                "teacher_target_delta_local_6d": target_delta.astype(np.float32),
                "teacher_contact_repr": _teacher_contact_repr(data, i, phase_name, target_delta, phase_success),
                "teacher_residual_action_4d": teacher_residual_action_4d.astype(np.float32),
                "teacher_residual_trajectory_4d": _residual_traj_4d(data, i, int(args.horizon)),
                "teacher_progress_label": teacher_progress.astype(np.float32),
                "teacher_risk_label": np.asarray(teacher_risk, dtype=np.float32),
                "teacher_stop_label": np.asarray(teacher_stop, dtype=np.float32),
                "teacher_confidence_label": np.asarray(teacher_conf, dtype=np.float32),
                "teacher_close_ready": np.asarray(
                    float(_scalar_at(data, "teacher_close_ready", i, 0.0) > 0.5 or exact_close_ready),
                    dtype=np.float32,
                ),
                "teacher_close_ready_all": np.asarray(
                    float(_scalar_at(data, "teacher_close_ready_all", i, 0.0) > 0.5 or exact_close_ready),
                    dtype=np.float32,
                ),
                "teacher_truth_handoff_ready": np.asarray(
                    float(_scalar_at(data, "teacher_truth_handoff_ready", i, 0.0) > 0.5),
                    dtype=np.float32,
                ),
                "teacher_close_ready_score": np.asarray(float(close_ready_score), dtype=np.float32),
                "close_ready_bridge_mask": np.asarray(float(bridge_window), dtype=np.float32),
                "close_ready_exact_mask": np.asarray(float(exact_close_ready), dtype=np.float32),
                "sample_weight": np.asarray(sample_weight, dtype=np.float32),
                "phase_id": np.asarray(PHASE_ID.get(phase_name, PHASE_ID["unknown"]), dtype=np.int64),
                "stage_bucket_id": np.asarray(STAGE_BUCKET_ID.get(stage_bucket, STAGE_BUCKET_ID["unknown"]), dtype=np.int64),
                "yaw_direction_label": np.asarray(_yaw_direction_label(teacher_residual_action_4d), dtype=np.int64),
                "yaw_imitation_enabled": np.asarray(yaw_enabled, dtype=np.float32),
                "verified_positive": np.asarray(float(verified_positive), dtype=np.float32),
                "teacher_grasp_commit_edge_pair_index": np.asarray(
                    int(_scalar_at(data, "teacher_grasp_commit_edge_pair_index", i, -1)),
                    dtype=np.int64,
                ),
                "teacher_grasp_commit_edge_pair_family": np.asarray(
                    int(_scalar_at(data, "teacher_grasp_commit_edge_pair_family", i, -1)),
                    dtype=np.int64,
                ),
                "teacher_grasp_commit_edge_pair_yaw_error": np.asarray(
                    float(_scalar_at(data, "teacher_grasp_commit_edge_pair_yaw_error", i, np.nan)),
                    dtype=np.float32,
                ),
                "teacher_edgepair_label_source": np.asarray(
                    _string_at(data, "teacher_edgepair_label_source", i, "none")
                ),
                "stage_bucket": np.asarray(stage_bucket),
                "alignment_phase": np.asarray(phase_name),
                "source_name": np.asarray(source_name),
            }

            if verified_positive:
                positive_rows.append(row)
                bucket_hist[stage_bucket] += 1
                phase_hist[phase_name] += 1
                positive_episode_rows[(source_name, ep)] += 1
                phase_success_counts[phase_name] += 1
                corridor_depth[phase_name].append(_scalar_at(data, "depth_proximity", i, np.nan))
                corridor_force[phase_name].append(_scalar_at(data, "force_norm", i, 0.0))
                corridor_action[phase_name].append(float(np.linalg.norm(np.asarray(row["planner_action_local"], dtype=np.float32)[:3])))
            elif bridge_window:
                positive_rows.append(row)
                close_bridge_episode_rows[(source_name, ep)] += 1
                close_bridge_step_counts[int(step - close_ready_step)] += 1
                if exact_close_ready:
                    bucket_hist[stage_bucket] += 1
                    phase_hist[phase_name] += 1
            elif teacher_risk > 0.5 or teacher_stop > 0.5:
                negative_rows.append(row)

    if not positive_rows:
        raise SystemExit("no verified positive rows found for vNext dataset build")
    max_negative = int(len(positive_rows) * float(args.max_negative_ratio))
    negative_rows = negative_rows[:max_negative]
    rows = positive_rows + negative_rows
    if len(rows) < int(args.min_rows):
        raise SystemExit(f"too few vNext rows: {len(rows)} < {args.min_rows}")

    keys = sorted({k for row in rows for k in row.keys()})
    out = {k: np.asarray([row[k] for row in rows]) for k in keys}
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    runtime_corridor = {}
    for phase_name in ("grasp_commit", "insert_commit"):
        runtime_corridor[phase_name] = {
            "depth_p90": _stats(np.asarray(corridor_depth[phase_name], dtype=np.float32))["p90"],
            "force_p90": _stats(np.asarray(corridor_force[phase_name], dtype=np.float32))["p90"],
            "planner_action_pos_norm_p90": _stats(np.asarray(corridor_action[phase_name], dtype=np.float32))["p90"],
        }

    report = {
        "output_npz": str(args.output_npz),
        "rows": int(len(rows)),
        "positive_rows": int(len(positive_rows)),
        "negative_rows": int(len(negative_rows)),
        "phase_histogram": {k: int(v) for k, v in phase_hist.items()},
        "bucket_histogram": {k: int(v) for k, v in bucket_hist.items()},
        "positive_episode_rows": {f"{k[0]}::ep{k[1]}": int(v) for k, v in positive_episode_rows.items()},
        "close_bridge_episode_rows": {f"{k[0]}::ep{k[1]}": int(v) for k, v in close_bridge_episode_rows.items()},
        "close_bridge_step_counts": {str(k): int(v) for k, v in close_bridge_step_counts.items()},
        "phase_success_counts": {k: int(v) for k, v in phase_success_counts.items()},
        "edgepair_label_nonnull_rows": int(
            sum(
                int(
                    int(np.asarray(row.get("teacher_grasp_commit_edge_pair_index", -1)).reshape(())) >= 0
                    and int(np.asarray(row.get("teacher_grasp_commit_edge_pair_family", -1)).reshape(())) >= 0
                )
                for row in rows
            )
        ),
        "edgepair_label_nonnull_rate": float(
            np.mean(
                [
                    int(
                        int(np.asarray(row.get("teacher_grasp_commit_edge_pair_index", -1)).reshape(())) >= 0
                        and int(np.asarray(row.get("teacher_grasp_commit_edge_pair_family", -1)).reshape(())) >= 0
                    )
                    for row in rows
                ]
            )
            if rows
            else 0.0
        ),
        "close_ready_exact_rows": int(sum(int(np.asarray(row.get("close_ready_exact_mask", 0.0)).reshape(())) > 0 for row in rows)),
        "close_ready_bridge_rows": int(sum(int(np.asarray(row.get("close_ready_bridge_mask", 0.0)).reshape(())) > 0 for row in rows)),
        "runtime_corridor": runtime_corridor,
        "defaults": {
            "verified_window_steps": int(args.verified_window_steps),
            "pre_context_steps": int(args.pre_context_steps),
            "post_context_steps": int(args.post_context_steps),
            "close_ready_pre_window_steps": int(args.close_ready_pre_window_steps),
            "close_ready_post_window_steps": int(args.close_ready_post_window_steps),
            "max_negative_ratio": float(args.max_negative_ratio),
            "horizon": int(args.horizon),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
