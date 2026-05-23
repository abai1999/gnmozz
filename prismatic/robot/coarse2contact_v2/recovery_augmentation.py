"""Targeted augmentation helpers for Coarse2Contact v2 recovery training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .recovery_audit import planner_bias_xyyaw


def load_bias_template(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def failure_morphology_bucket(
    record: Mapping[str, Any],
    *,
    xy_threshold: float = 0.06,
    yaw_threshold: float = 0.15,
) -> str:
    """Assign a coarse tail bucket from the recovery target geometry."""

    dx = float(record.get("recovery_target_dx", record.get("trace_error_dx", 0.0)) or 0.0)
    dy = float(record.get("recovery_target_dy", record.get("trace_error_dy", 0.0)) or 0.0)
    dyaw = float(record.get("recovery_target_dyaw", record.get("trace_error_dyaw", 0.0)) or 0.0)
    xy_norm = float(np.hypot(dx, dy))
    yaw_norm = float(abs(dyaw))
    if xy_norm >= float(xy_threshold) and yaw_norm >= float(yaw_threshold):
        return "large_xy_large_yaw"
    if xy_norm >= float(xy_threshold) and yaw_norm < float(yaw_threshold):
        return "large_xy_small_yaw"
    if xy_norm < float(xy_threshold) and yaw_norm >= float(yaw_threshold):
        return "small_xy_large_yaw"
    return "small_xy_small_yaw"


def failure_morphology_stats(
    records: Sequence[Mapping[str, Any]],
    *,
    xy_threshold: float = 0.06,
    yaw_threshold: float = 0.15,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        bucket = failure_morphology_bucket(record, xy_threshold=xy_threshold, yaw_threshold=yaw_threshold)
        buckets.setdefault(bucket, []).append(record)
    stats: dict[str, dict[str, float]] = {}
    for bucket, rows in buckets.items():
        dx = np.asarray([float(r.get("recovery_target_dx", r.get("trace_error_dx", 0.0)) or 0.0) for r in rows], dtype=np.float32)
        dy = np.asarray([float(r.get("recovery_target_dy", r.get("trace_error_dy", 0.0)) or 0.0) for r in rows], dtype=np.float32)
        dyaw = np.asarray([float(r.get("recovery_target_dyaw", r.get("trace_error_dyaw", 0.0)) or 0.0) for r in rows], dtype=np.float32)
        score = np.asarray([float(r.get("planner_bias_score", 0.0)) for r in rows], dtype=np.float32)
        stats[bucket] = {
            "count": float(len(rows)),
            "xy_norm_median": float(np.median(np.hypot(dx, dy))) if rows else 0.0,
            "yaw_abs_median": float(np.median(np.abs(dyaw))) if rows else 0.0,
            "planner_bias_score_median": float(np.median(score)) if rows else 0.0,
            "planner_bias_score_mean": float(np.mean(score)) if rows else 0.0,
        }
    return stats


def _component_stat(template: Mapping[str, Any], name: str, stat: str, default: float = 0.0) -> float:
    comp = template.get("planner_bias_components", {}) if isinstance(template, Mapping) else {}
    entry = comp.get(name, {}) if isinstance(comp, Mapping) else {}
    try:
        return float(entry.get(stat, default))
    except Exception:
        return float(default)


def select_hard_trajectories(
    report: Mapping[str, Any],
    *,
    hard_fraction: float = 0.35,
    min_trajectories: int = 6,
) -> list[str]:
    summaries = list(report.get("trajectory_summaries", []))
    if not summaries:
        return []
    large_bias_threshold = float(report.get("large_bias_threshold", 0.0))
    candidates = [s for s in summaries if float(s.get("bias_score_max", 0.0)) >= large_bias_threshold]
    if not candidates:
        candidates = list(summaries)
    candidates = sorted(candidates, key=lambda s: (float(s.get("gain_mean", 0.0)), -float(s.get("bias_score_max", 0.0)), int(s.get("num_steps", 0))))
    count = max(int(min_trajectories), int(round(len(candidates) * float(np.clip(hard_fraction, 0.0, 1.0)))))
    count = max(1, min(count, len(candidates)))
    return [str(s.get("trajectory_id", "")) for s in candidates[:count] if str(s.get("trajectory_id", ""))]


def augment_recovery_record(
    record: Mapping[str, Any],
    *,
    scale: float,
    template: Mapping[str, Any] | None = None,
    source_index: int = -1,
) -> dict[str, Any]:
    template = template or {}
    prior = np.asarray(record.get("planner_prior_delta", []), dtype=np.float32).reshape(-1)
    if prior.size < 6:
        prior = np.pad(prior, (0, 6 - prior.size))
    prior = prior[:6].astype(np.float32)
    scale = float(max(scale, 1.0))
    aug_prior = prior.copy()
    aug_prior[0] *= scale
    aug_prior[1] *= scale
    aug_prior[5] *= scale
    shift = np.zeros(6, dtype=np.float32)
    shift[0] = np.sign(prior[0] if abs(float(prior[0])) > 1e-9 else _component_stat(template, "dx", "median", 0.0)) * abs(_component_stat(template, "dx", "median", 0.0)) * (scale - 1.0)
    shift[1] = np.sign(prior[1] if abs(float(prior[1])) > 1e-9 else _component_stat(template, "dy", "median", 0.0)) * abs(_component_stat(template, "dy", "median", 0.0)) * (scale - 1.0)
    shift[5] = np.sign(prior[5] if abs(float(prior[5])) > 1e-9 else _component_stat(template, "dyaw", "median", 0.0)) * abs(_component_stat(template, "dyaw", "median", 0.0)) * (scale - 1.0)
    aug_prior = aug_prior + shift
    planner_xy, planner_yaw_abs, planner_dyaw, planner_score = planner_bias_xyyaw(aug_prior)
    base_bias_score = float(record.get("planner_bias_score", planner_score))
    threshold = float(template.get("bias_score_threshold", base_bias_score)) if isinstance(template, Mapping) else base_bias_score
    out = dict(record)
    out["planner_prior_delta"] = aug_prior.astype(np.float32).tolist()
    out["planner_action_world"] = aug_prior.astype(np.float32).tolist()
    out["planner_chunk_local_6d"] = aug_prior.astype(np.float32).tolist()
    out["planner_bias_xy"] = float(planner_xy)
    out["planner_bias_yaw"] = float(planner_yaw_abs)
    out["planner_bias_dyaw"] = float(planner_dyaw)
    out["planner_bias_score"] = float(planner_score)
    out["planner_bias_rank"] = float(planner_score / max(threshold, 1e-8))
    out["recovery_needed"] = bool(float(record.get("recovery_needed", 0.0)) > 0.5 or planner_score >= threshold)
    out["recovery_bias_source"] = "planner_tail_bias_template+targeted_aug"
    out["is_augmented"] = 1.0
    out["augment_scale"] = float(scale)
    out["augment_source_index"] = int(source_index)
    out["augment_offset_local"] = (aug_prior - prior).astype(np.float32).tolist()
    out["augment_strategy"] = "targeted_large_bias_scale"
    out["augment_from_trajectory"] = str(record.get("trajectory_id", ""))
    out["uses_privileged_target"] = False
    out["uses_rlbench_mask_runtime"] = False
    return out


def build_failure_replay_library(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_trajectories: set[str],
    tail_rows: int = 6,
    large_bias_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        traj_id = str(record.get("trajectory_id", ""))
        if traj_id not in selected_trajectories:
            continue
        grouped.setdefault(traj_id, []).append(dict(record))
    library: list[dict[str, Any]] = []
    for traj_id, traj in grouped.items():
        traj = sorted(traj, key=lambda r: int(r.get("trajectory_step", r.get("step_idx", 0))))
        if not traj:
            continue
        tail = traj[max(0, len(traj) - int(max(tail_rows, 1))):]
        prev_residual = None
        for row in tail:
            prior = np.asarray(row.get("planner_prior_delta", []), dtype=np.float32).reshape(-1)
            if prior.size < 6:
                prior = np.pad(prior, (0, 6 - prior.size))
            prior = prior[:6].astype(np.float32)
            target = np.asarray(
                [
                    float(row.get("recovery_target_dx", row.get("trace_error_dx", 0.0))),
                    float(row.get("recovery_target_dy", row.get("trace_error_dy", 0.0))),
                    float(row.get("recovery_target_dyaw", row.get("trace_error_dyaw", 0.0))),
                ],
                dtype=np.float32,
            )
            residual = prior[:3] - target
            if prev_residual is None:
                drift = np.zeros_like(residual)
            else:
                drift = residual - prev_residual
            prev_residual = residual
            _, _, _, score = planner_bias_xyyaw(prior)
            if float(score) < float(large_bias_threshold):
                continue
            library.append(
                {
                    "failure_bucket": failure_morphology_bucket(row),
                    "trajectory_id": traj_id,
                    "trajectory_step": int(row.get("trajectory_step", row.get("step_idx", 0))),
                    "trajectory_phase": str(row.get("trajectory_phase", "")),
                    "phase_name": str(row.get("phase_name", "")),
                    "planner_prior_delta": prior.tolist(),
                    "recovery_target": target.tolist(),
                    "failure_residual": residual.tolist(),
                    "failure_drift": drift.tolist(),
                    "planner_bias_score": float(score),
                    "planner_bias_rank": float(row.get("planner_bias_rank", 0.0)),
                    "frame_confidence": float(row.get("frame_confidence", 0.0)),
                    "frame_observability": float(row.get("frame_observability", 0.0)),
                    "frame_axis_strength": float(row.get("frame_axis_strength", 0.0)),
                    "frame_completeness": float(row.get("frame_completeness", 0.0)),
                }
            )
    return library


def select_focus_trajectories_by_bucket(
    records: Sequence[Mapping[str, Any]],
    *,
    focus_buckets: Sequence[str],
    xy_threshold: float = 0.06,
    yaw_threshold: float = 0.15,
    trajectory_fraction: float = 0.65,
    min_trajectories_per_bucket: int = 2,
) -> dict[str, list[str]]:
    focus = {str(bucket).strip() for bucket in focus_buckets if str(bucket).strip()}
    bucket_trajs: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for record in records:
        bucket = failure_morphology_bucket(record, xy_threshold=xy_threshold, yaw_threshold=yaw_threshold)
        if bucket not in focus:
            continue
        traj_id = str(record.get("trajectory_id", ""))
        if not traj_id:
            continue
        bucket_trajs.setdefault(bucket, {}).setdefault(traj_id, []).append(record)
    selected: dict[str, list[str]] = {}
    for bucket, traj_map in bucket_trajs.items():
        scored: list[tuple[float, int, str]] = []
        for traj_id, rows in traj_map.items():
            score = max(float(r.get("planner_bias_score", 0.0)) for r in rows) if rows else 0.0
            num_tail = len(rows)
            scored.append((score, num_tail, traj_id))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        keep = max(int(min_trajectories_per_bucket), int(round(len(scored) * float(np.clip(trajectory_fraction, 0.0, 1.0)))))
        keep = max(1, min(keep, len(scored)))
        selected[bucket] = [traj_id for _, _, traj_id in scored[:keep]]
    return selected


def build_bucket_tail_replay_augmented_records(
    records: Sequence[Mapping[str, Any]],
    *,
    focus_buckets: Sequence[str],
    xy_threshold: float = 0.06,
    yaw_threshold: float = 0.15,
    trajectory_fraction: float = 0.65,
    min_trajectories_per_bucket: int = 2,
    tail_rows: int = 6,
    replay_strengths_by_bucket: Mapping[str, Sequence[float]] | None = None,
    drift_strengths_by_bucket: Mapping[str, Sequence[float]] | None = None,
    replay_modes_by_bucket: Mapping[str, Sequence[str]] | None = None,
    bucket_weight_by_bucket: Mapping[str, float] | None = None,
    min_tail_rows_per_bucket: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = [dict(r) for r in records]
    selected_by_bucket = select_focus_trajectories_by_bucket(
        base_rows,
        focus_buckets=focus_buckets,
        xy_threshold=xy_threshold,
        yaw_threshold=yaw_threshold,
        trajectory_fraction=trajectory_fraction,
        min_trajectories_per_bucket=min_trajectories_per_bucket,
    )
    selected_trajectories = {traj_id for trajs in selected_by_bucket.values() for traj_id in trajs}
    large_bias_threshold = 0.0
    if base_rows:
        scores = np.asarray([float(r.get("planner_bias_score", 0.0)) for r in base_rows], dtype=np.float32)
        large_bias_threshold = float(np.quantile(scores, 0.7)) if scores.size else 0.0
    replay_library = build_failure_replay_library(
        base_rows,
        selected_trajectories=selected_trajectories,
        tail_rows=tail_rows,
        large_bias_threshold=large_bias_threshold,
    )
    library_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for item in replay_library:
        library_by_bucket.setdefault(str(item.get("failure_bucket", "")), []).append(item)
    focus = {str(bucket).strip() for bucket in focus_buckets if str(bucket).strip()}
    replay_strengths_by_bucket = replay_strengths_by_bucket or {}
    drift_strengths_by_bucket = drift_strengths_by_bucket or {}
    replay_modes_by_bucket = replay_modes_by_bucket or {}
    bucket_weight_by_bucket = bucket_weight_by_bucket or {}

    aug_rows: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = {}
    selected_row_counts: dict[str, int] = {}
    augmented_row_counts: dict[str, int] = {}
    selected_source_indices: list[int] = []
    by_bucket_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, record in enumerate(base_rows):
        bucket = failure_morphology_bucket(record, xy_threshold=xy_threshold, yaw_threshold=yaw_threshold)
        if bucket not in focus:
            continue
        by_bucket_rows.setdefault(bucket, []).append((idx, record))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    for bucket, rows in by_bucket_rows.items():
        rows = sorted(
            rows,
            key=lambda item: (
                float(item[1].get("planner_bias_score", 0.0)),
                int(item[1].get("trajectory_step", item[1].get("step_idx", 0))),
            ),
            reverse=True,
        )
        keep = max(int(min_tail_rows_per_bucket), int(round(len(rows) * float(np.clip(trajectory_fraction, 0.0, 1.0)))))
        keep = max(1, min(keep, len(rows)))
        selected = rows[:keep]
        selected_row_counts[bucket] = len(selected)
        selected_indices = [idx for idx, _ in selected]
        selected_source_indices.extend(selected_indices)
        modes = [str(m).strip().lower() for m in (replay_modes_by_bucket.get(bucket, ("overshoot",)))]
        if not modes:
            modes = ["overshoot"]
        strengths = [float(max(s, 1.0)) for s in replay_strengths_by_bucket.get(bucket, (1.15, 1.25))]
        drift_strengths = [float(max(s, 0.0)) for s in drift_strengths_by_bucket.get(bucket, (0.15, 0.25))]
        templates = library_by_bucket.get(bucket) or replay_library
        if not templates:
            templates = []
        for local_i, (idx, record) in enumerate(selected):
            for copy_i in range(max(len(strengths), 1)):
                strength = strengths[copy_i % len(strengths)] if strengths else 1.0
                drift_strength = drift_strengths[copy_i % len(drift_strengths)] if drift_strengths else 0.0
                mode = modes[(local_i + copy_i) % len(modes)] if modes else "overshoot"
                template = templates[(local_i + copy_i) % len(templates)] if templates else {}
                aug = augment_recovery_record_replay(
                    record,
                    replay_template=template,
                    replay_strength=strength,
                    replay_drift_strength=drift_strength,
                    replay_mode=mode,
                    source_index=idx,
                )
                aug["recovery_bucket_name"] = bucket
                aug["recovery_bucket_focus"] = 1.0
                aug["recovery_bucket_weight"] = float(bucket_weight_by_bucket.get(bucket, 1.0))
                aug["recovery_tail_source"] = "real_runtime_failure_tail"
                aug_rows.append(aug)
                augmented_row_counts[bucket] = augmented_row_counts.get(bucket, 0) + 1

    report = {
        "focus_buckets": sorted(focus),
        "selected_trajectories_by_bucket": selected_by_bucket,
        "selected_source_indices": selected_source_indices,
        "selected_source_count": len(selected_source_indices),
        "bucket_counts": bucket_counts,
        "selected_row_counts": selected_row_counts,
        "augmented_row_counts": augmented_row_counts,
        "replay_library_count": len(replay_library),
        "tail_rows": int(tail_rows),
        "trajectory_fraction": float(trajectory_fraction),
        "min_trajectories_per_bucket": int(min_trajectories_per_bucket),
        "min_tail_rows_per_bucket": int(min_tail_rows_per_bucket),
        "xy_threshold": float(xy_threshold),
        "yaw_threshold": float(yaw_threshold),
        "bucket_weight_by_bucket": {k: float(v) for k, v in bucket_weight_by_bucket.items()},
        "replay_modes_by_bucket": {k: list(v) for k, v in replay_modes_by_bucket.items()},
        "replay_strengths_by_bucket": {k: [float(x) for x in v] for k, v in replay_strengths_by_bucket.items()},
        "drift_strengths_by_bucket": {k: [float(x) for x in v] for k, v in drift_strengths_by_bucket.items()},
        "augmentation_mode": "bucket_tail_failure_replay",
    }
    return base_rows + aug_rows, report


def augment_recovery_record_replay(
    record: Mapping[str, Any],
    *,
    replay_template: Mapping[str, Any],
    replay_strength: float,
    replay_drift_strength: float,
    replay_mode: str = "overshoot",
    source_index: int = -1,
) -> dict[str, Any]:
    target = np.asarray(
        [
            float(record.get("recovery_target_dx", record.get("trace_error_dx", 0.0))),
            float(record.get("recovery_target_dy", record.get("trace_error_dy", 0.0))),
            float(record.get("recovery_target_dyaw", record.get("trace_error_dyaw", 0.0))),
        ],
        dtype=np.float32,
    )
    prior = np.asarray(record.get("planner_prior_delta", []), dtype=np.float32).reshape(-1)
    if prior.size < 6:
        prior = np.pad(prior, (0, 6 - prior.size))
    prior = prior[:6].astype(np.float32)
    residual = np.asarray(replay_template.get("failure_residual", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)[:3]
    drift = np.asarray(replay_template.get("failure_drift", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)[:3]
    replay_strength = float(max(replay_strength, 1.0))
    replay_drift_strength = float(max(replay_drift_strength, 0.0))
    replay_mode = str(replay_mode or "overshoot").lower()
    if replay_mode == "oscillate":
        replay_delta = replay_strength * residual - replay_drift_strength * drift
    elif replay_mode == "cross_couple":
        replay_delta = replay_strength * residual + replay_drift_strength * np.asarray(
            [
                drift[1],
                drift[0],
                np.sign(float(residual[0]) + float(residual[1])) * abs(float(drift[2])),
            ],
            dtype=np.float32,
        )
    elif replay_mode == "shear":
        replay_delta = replay_strength * residual + replay_drift_strength * np.asarray(
            [
                0.65 * drift[0] + 0.35 * drift[1],
                0.65 * drift[1] - 0.35 * drift[0],
                drift[2],
            ],
            dtype=np.float32,
        )
    else:
        replay_delta = replay_strength * residual + replay_drift_strength * drift
    aug_prior = prior.copy()
    aug_prior[:3] = target + replay_delta
    aug_prior[0] += 0.08 * replay_delta[1]
    aug_prior[1] += 0.08 * replay_delta[0]
    aug_prior[5] += 0.18 * np.tanh(replay_delta[2] / 0.05) * abs(float(replay_delta[2]))
    planner_xy, planner_yaw_abs, planner_dyaw, planner_score = planner_bias_xyyaw(aug_prior)
    threshold = float(replay_template.get("large_bias_threshold", record.get("planner_bias_score", planner_score)))
    out = dict(record)
    out["planner_prior_delta"] = aug_prior.astype(np.float32).tolist()
    out["planner_action_world"] = aug_prior.astype(np.float32).tolist()
    out["planner_chunk_local_6d"] = aug_prior.astype(np.float32).tolist()
    out["planner_bias_xy"] = float(planner_xy)
    out["planner_bias_yaw"] = float(planner_yaw_abs)
    out["planner_bias_dyaw"] = float(planner_dyaw)
    out["planner_bias_score"] = float(planner_score)
    out["planner_bias_rank"] = float(planner_score / max(threshold, 1e-8))
    out["recovery_needed"] = True
    out["recovery_bias_source"] = "tail_failure_replay"
    out["is_augmented"] = 1.0
    out["augment_scale"] = float(replay_strength)
    out["augment_drift_scale"] = float(replay_drift_strength)
    out["augment_source_index"] = int(source_index)
    out["augment_offset_local"] = (aug_prior - prior).astype(np.float32).tolist()
    out["augment_strategy"] = "tail_failure_replay"
    out["replay_mode"] = replay_mode
    out["augment_from_trajectory"] = str(record.get("trajectory_id", ""))
    out["replay_source_trajectory"] = str(replay_template.get("trajectory_id", ""))
    out["replay_source_step_idx"] = int(replay_template.get("trajectory_step", -1))
    out["replay_source_phase"] = str(replay_template.get("trajectory_phase", ""))
    out["replay_source_residual"] = residual.astype(np.float32).tolist()
    out["replay_source_drift"] = drift.astype(np.float32).tolist()
    out["replay_strength"] = float(replay_strength)
    out["replay_drift_strength"] = float(replay_drift_strength)
    out["uses_privileged_target"] = False
    out["uses_rlbench_mask_runtime"] = False
    return out


def build_failure_replay_augmented_records(
    records: Sequence[Mapping[str, Any]],
    *,
    shadow_report: Mapping[str, Any],
    bias_template: Mapping[str, Any] | None = None,
    hard_fraction: float = 0.35,
    min_trajectories: int = 6,
    tail_rows: int = 6,
    replay_strengths: Sequence[float] = (1.25, 1.5, 1.75),
    drift_strengths: Sequence[float] = (0.25, 0.5, 0.75),
    replay_modes: Sequence[str] = ("overshoot", "oscillate", "cross_couple"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_trajectories = set(select_hard_trajectories(shadow_report, hard_fraction=hard_fraction, min_trajectories=min_trajectories))
    base_rows = [dict(r) for r in records]
    large_bias_threshold = float(shadow_report.get("large_bias_threshold", 0.0))
    replay_library = build_failure_replay_library(
        base_rows,
        selected_trajectories=selected_trajectories,
        tail_rows=tail_rows,
        large_bias_threshold=large_bias_threshold,
    )
    if not replay_library:
        return base_rows, {
            "selected_trajectories": sorted(selected_trajectories),
            "selected_source_indices": [],
            "selected_source_count": 0,
            "replay_library_count": 0,
            "augmentation_count": 0,
            "hard_fraction": float(hard_fraction),
            "min_trajectories": int(min_trajectories),
            "tail_rows": int(tail_rows),
            "replay_strengths": [float(s) for s in replay_strengths],
            "drift_strengths": [float(s) for s in drift_strengths],
            "augmentation_mode": "tail_failure_replay",
        }
    aug_rows: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    replay_count = len(replay_library)
    replay_strengths = [float(max(s, 1.0)) for s in replay_strengths]
    drift_strengths = [float(max(s, 0.0)) for s in drift_strengths]
    replay_modes = [str(m).strip().lower() for m in replay_modes if str(m).strip()]
    if not replay_modes:
        replay_modes = ["overshoot"]
    for idx, record in enumerate(base_rows):
        traj_id = str(record.get("trajectory_id", ""))
        if traj_id not in selected_trajectories:
            continue
        if float(record.get("planner_bias_score", 0.0)) < large_bias_threshold:
            continue
        selected_indices.append(idx)
        for mode_i, mode in enumerate(replay_modes):
            for strength_i, strength in enumerate(replay_strengths):
                drift_strength = drift_strengths[min(strength_i, len(drift_strengths) - 1)] if drift_strengths else 0.0
                template = replay_library[(idx + mode_i + strength_i) % replay_count]
                aug_rows.append(
                    augment_recovery_record_replay(
                        record,
                        replay_template=template,
                        replay_strength=strength,
                        replay_drift_strength=drift_strength,
                        replay_mode=mode,
                        source_index=idx,
                    )
                )
    report = {
        "selected_trajectories": sorted(selected_trajectories),
        "selected_source_indices": selected_indices,
        "selected_source_count": len(selected_indices),
        "replay_library_count": int(replay_count),
        "augmentation_count": len(aug_rows),
        "hard_fraction": float(hard_fraction),
        "min_trajectories": int(min_trajectories),
        "tail_rows": int(tail_rows),
        "replay_strengths": [float(s) for s in replay_strengths],
        "drift_strengths": [float(s) for s in drift_strengths],
        "replay_modes": replay_modes,
        "augmentation_mode": "tail_failure_replay",
    }
    return base_rows + aug_rows, report


def build_targeted_augmented_records(
    records: Sequence[Mapping[str, Any]],
    *,
    shadow_report: Mapping[str, Any],
    bias_template: Mapping[str, Any] | None = None,
    hard_fraction: float = 0.35,
    min_trajectories: int = 6,
    scales: Sequence[float] = (1.25, 1.5, 1.75),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_trajectories = set(select_hard_trajectories(shadow_report, hard_fraction=hard_fraction, min_trajectories=min_trajectories))
    base_rows = [dict(r) for r in records]
    aug_rows: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for idx, record in enumerate(base_rows):
        if str(record.get("trajectory_id", "")) not in selected_trajectories:
            continue
        if not bool(record.get("recovery_needed", False)):
            continue
        if float(record.get("planner_bias_score", 0.0)) < float(shadow_report.get("large_bias_threshold", 0.0)):
            continue
        selected_indices.append(idx)
        for scale in scales:
            aug_rows.append(
                augment_recovery_record(
                    record,
                    scale=float(scale),
                    template=bias_template or shadow_report,
                    source_index=idx,
                )
            )
    report = {
        "selected_trajectories": sorted(selected_trajectories),
        "selected_source_indices": selected_indices,
        "selected_source_count": len(selected_indices),
        "augmentation_scales": [float(s) for s in scales],
        "augmentation_count": len(aug_rows),
        "hard_fraction": float(hard_fraction),
        "min_trajectories": int(min_trajectories),
    }
    return base_rows + aug_rows, report
