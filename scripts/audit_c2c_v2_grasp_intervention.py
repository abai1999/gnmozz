#!/usr/bin/env python3
"""Audit grasp-only intervention probes for C2C v2 basin pullback.

This script is intentionally narrow:
it consumes the probe trace produced by `evaluate_c2c_v2_rlbench.py` when
`--c2c_grasp_probe_policy replay_oracle_xy` is enabled, and reports whether
the inserted bounded xy step actually contracts the privileged grasp basin
error in live dynamics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_vec(value: Any, *, length: int, default: float = float("nan")) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), constant_values=default)
    return arr[:length].astype(np.float32)


def _wilson_lower_bound(successes: int, n: int, *, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = float(successes) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = phat + (z * z) / (2.0 * float(n))
    margin = z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * float(n))) / float(n))
    return float(max((center - margin) / denom, 0.0))


def _episode_from_trace_path(path: Path) -> int:
    stem = path.stem
    for token in stem.split("_"):
        if token.startswith("ep") and token[2:].isdigit():
            return int(token[2:])
    return -1


def _load_trace_rows(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        episode_idx = _episode_from_trace_path(trace_path)
        for row in _read_jsonl(trace_path):
            row = dict(row)
            row.setdefault("episode_idx", int(episode_idx))
            row.setdefault("source_trace_path", str(trace_path))
            rows.append(row)
    rows.sort(key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step", r.get("step_idx", -1)))))
    return rows


def _trace_row_error(row: Mapping[str, Any], prefix: str) -> np.ndarray:
    return _safe_vec(row.get(prefix), length=4, default=float("nan"))


def _trace_row_post_error(row: Mapping[str, Any], *, horizon: bool = False) -> np.ndarray:
    if horizon and row.get("grasp_probe_horizon_final_true_error_t") is not None:
        err = _trace_row_error(row, "grasp_probe_horizon_final_true_error_t")
        if np.all(np.isfinite(err[:2])):
            return err
    return _trace_row_error(row, "grasp_probe_post_true_error_t")


def _xy_norm(vec: Iterable[float]) -> float:
    arr = np.asarray(list(vec), dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return float("nan")
    return float(np.hypot(float(arr[0]), float(arr[1])))


def _axis_abs_contraction_rate(rows: list[dict[str, Any]], axis: str) -> float:
    if not rows:
        return 0.0
    idx = 0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 3
    good = []
    for row in rows:
        pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
        post = _trace_row_error(row, "grasp_probe_post_true_error_t")
        if not np.all(np.isfinite(pre[:4])) or not np.all(np.isfinite(post[:4])):
            continue
        good.append(bool(abs(float(post[idx])) <= abs(float(pre[idx])) + 1.0e-9))
    return float(np.mean(good)) if good else 0.0


def _xy_contraction_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    good = []
    for row in rows:
        pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
        post = _trace_row_error(row, "grasp_probe_post_true_error_t")
        if not np.all(np.isfinite(pre[:4])) or not np.all(np.isfinite(post[:4])):
            continue
        good.append(bool(_xy_norm(pre[:2]) > _xy_norm(post[:2]) + 1.0e-9))
    return float(np.mean(good)) if good else 0.0


def _horizon_xy_contraction_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    good = []
    for row in rows:
        pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
        post = _trace_row_post_error(row, horizon=True)
        if not np.all(np.isfinite(pre[:4])) or not np.all(np.isfinite(post[:4])):
            continue
        good.append(bool(_xy_norm(pre[:2]) > _xy_norm(post[:2]) + 1.0e-9))
    return float(np.mean(good)) if good else 0.0


def _mean_step_delta(rows: list[dict[str, Any]], axis: str) -> float:
    if not rows:
        return 0.0
    idx = 0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 3
    deltas = []
    for row in rows:
        pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
        post = _trace_row_error(row, "grasp_probe_post_true_error_t")
        if not np.all(np.isfinite(pre[:4])) or not np.all(np.isfinite(post[:4])):
            continue
        deltas.append(float(post[idx] - pre[idx]))
    return float(np.mean(deltas)) if deltas else 0.0


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(r.get(key, False)) for r in rows]))


def _probe_bool(row: Mapping[str, Any], suffix: str, *, horizon: bool = False) -> bool:
    if horizon:
        key = f"grasp_probe_horizon_{suffix}"
        if key in row:
            return bool(row.get(key, False))
    return bool(row.get(f"grasp_probe_{suffix}", False))


def _pre_xy_error(row: Mapping[str, Any]) -> float:
    pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
    return _xy_norm(pre[:2])


def _pre_abs_yaw(row: Mapping[str, Any]) -> float:
    pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
    if pre.size < 4 or not np.isfinite(pre[3]):
        return float("nan")
    return float(abs(float(pre[3])))


def _xy_feasible(row: Mapping[str, Any], *, near_grasp_xy_threshold: float, max_xy_step: float, horizon_steps: int) -> bool:
    pre_xy = _pre_xy_error(row)
    if not np.isfinite(pre_xy):
        return False
    return bool(pre_xy <= float(near_grasp_xy_threshold) + float(max_xy_step) * float(max(1, horizon_steps)) + 1.0e-9)


def _yaw_feasible(row: Mapping[str, Any], *, near_grasp_yaw_threshold: float) -> bool:
    yaw = _pre_abs_yaw(row)
    if not np.isfinite(yaw):
        return False
    return bool(yaw <= float(near_grasp_yaw_threshold) + 1.0e-9)


def _shell_summary(
    rows: list[dict[str, Any]],
    *,
    near_grasp_xy_threshold: float,
    near_grasp_yaw_threshold: float,
    max_xy_step: float,
    horizon_steps: int,
) -> dict[str, Any]:
    active = _active_probe_rows(rows)
    one_step = [
        r for r in active
        if _xy_feasible(r, near_grasp_xy_threshold=near_grasp_xy_threshold, max_xy_step=max_xy_step, horizon_steps=1)
    ]
    horizon_xy = [
        r for r in active
        if _xy_feasible(r, near_grasp_xy_threshold=near_grasp_xy_threshold, max_xy_step=max_xy_step, horizon_steps=horizon_steps)
    ]
    yaw = [r for r in active if _yaw_feasible(r, near_grasp_yaw_threshold=near_grasp_yaw_threshold)]
    horizon_xy_yaw = [
        r for r in horizon_xy
        if _yaw_feasible(r, near_grasp_yaw_threshold=near_grasp_yaw_threshold)
    ]

    def _summary(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": int(len(subset)),
            "rate_of_active": float(len(subset) / len(active)) if active else 0.0,
            "one_step_near_grasp_after_rate": float(np.mean([_probe_bool(r, "near_grasp_after") for r in subset])) if subset else 0.0,
            "horizon_near_grasp_after_rate": float(np.mean([_probe_bool(r, "near_grasp_after", horizon=True) for r in subset])) if subset else 0.0,
            "horizon_micro_entry_ready_after_rate": float(np.mean([_probe_bool(r, "micro_entry_ready_after", horizon=True) for r in subset])) if subset else 0.0,
            "horizon_xy_contraction_rate": _horizon_xy_contraction_rate(subset),
            "one_step_overshoot_rate": float(np.mean([_probe_bool(r, "overshoot") for r in subset])) if subset else 0.0,
            "horizon_overshoot_rate": float(np.mean([_probe_bool(r, "overshoot", horizon=True) for r in subset])) if subset else 0.0,
            "overshoot_rate": float(np.mean([_probe_bool(r, "overshoot", horizon=True) for r in subset])) if subset else 0.0,
        }

    return {
        "one_step_xy_feasible": _summary(one_step),
        "horizon_xy_feasible": _summary(horizon_xy),
        "yaw_feasible": _summary(yaw),
        "horizon_xy_and_yaw_feasible": _summary(horizon_xy_yaw),
    }


def _yaw_blocked_rate_within_horizon_xy_feasible(
    rows: list[dict[str, Any]],
    *,
    near_grasp_xy_threshold: float,
    near_grasp_yaw_threshold: float,
    max_xy_step: float,
    horizon_steps: int,
) -> float:
    feasible = [
        r for r in _active_probe_rows(rows)
        if _xy_feasible(r, near_grasp_xy_threshold=near_grasp_xy_threshold, max_xy_step=max_xy_step, horizon_steps=horizon_steps)
    ]
    if not feasible:
        return 0.0
    blocked = [
        not _yaw_feasible(r, near_grasp_yaw_threshold=near_grasp_yaw_threshold)
        for r in feasible
    ]
    return float(np.mean(blocked)) if blocked else 0.0


def _ring_grasp_align_dwell_steps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode = _group_rows(rows, ("episode_idx",))
    dwell = [
        sum(1 for r in subset if _row_group_value(r, "c2c_v2_stage", "") == "RING_GRASP_ALIGN")
        for _, subset in by_episode.items()
    ]
    return {
        "total": int(sum(dwell)),
        "mean_per_episode": float(np.mean(dwell)) if dwell else 0.0,
        "max_per_episode": int(max(dwell, default=0)),
    }


def _recover_preempt_rate_before_first_probe_step(rows: list[dict[str, Any]]) -> float:
    by_episode = _group_rows(rows, ("episode_idx",))
    flags: list[bool] = []
    for _, subset in by_episode.items():
        ordered = sorted(subset, key=lambda r: int(r.get("step", r.get("step_idx", -1)) or -1))
        active_indices = [idx for idx, row in enumerate(ordered) if bool(row.get("grasp_probe_active", False))]
        limit = active_indices[0] if active_indices else len(ordered)
        prefix = ordered[:limit]
        flags.append(any(_row_group_value(r, "c2c_v2_stage", "") == "RECOVER" for r in prefix))
    return float(np.mean(flags)) if flags else 0.0


def _queue_protocol_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = _active_probe_rows(rows)
    flushed = [r for r in active if bool(r.get("grasp_probe_queue_flushed", False))]
    retained = [r for r in active if not bool(r.get("grasp_probe_queue_flushed", False))]
    flushed_rate = _horizon_xy_contraction_rate(flushed)
    retained_rate = _horizon_xy_contraction_rate(retained)
    return {
        "queue_flushed_rate": float(len(flushed) / len(active)) if active else 0.0,
        "mean_queue_len_before": float(np.mean([_safe_float(r.get("grasp_probe_queue_len_before", 0.0)) for r in active])) if active else 0.0,
        "mean_queue_len_after": float(np.mean([_safe_float(r.get("grasp_probe_queue_len_after", 0.0)) for r in active])) if active else 0.0,
        "flushed_horizon_xy_contraction_rate": float(flushed_rate),
        "retained_horizon_xy_contraction_rate": float(retained_rate),
        "queue_flush_ablation_delta": float(flushed_rate - retained_rate) if flushed and retained else 0.0,
    }


def _row_group_value(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    return str(value)


def _probe_pre_xy(row: Mapping[str, Any]) -> float:
    if "grasp_probe_pre_xy_error" in row:
        return _safe_float(row.get("grasp_probe_pre_xy_error", float("nan")), float("nan"))
    return _xy_norm(_trace_row_error(row, "grasp_probe_pre_true_error_t")[:2])


def _probe_yaw_observable(row: Mapping[str, Any], *, near_grasp_yaw_threshold: float) -> bool:
    if "grasp_probe_yaw_feasible" in row:
        return bool(row.get("grasp_probe_yaw_feasible", False))
    return _yaw_feasible(row, near_grasp_yaw_threshold=near_grasp_yaw_threshold)


def _probe_horizon_xy_feasible(
    row: Mapping[str, Any],
    *,
    near_grasp_xy_threshold: float,
    max_xy_step: float,
    horizon_steps: int,
) -> bool:
    if "grasp_probe_horizon_xy_feasible" in row:
        return bool(row.get("grasp_probe_horizon_xy_feasible", False))
    return _xy_feasible(
        row,
        near_grasp_xy_threshold=near_grasp_xy_threshold,
        max_xy_step=max_xy_step,
        horizon_steps=horizon_steps,
    )


def _probe_near_basin_shell(
    row: Mapping[str, Any],
    *,
    near_grasp_xy_threshold: float,
    near_grasp_yaw_threshold: float,
    max_xy_step: float,
    horizon_steps: int,
) -> bool:
    if "grasp_probe_near_basin_shell" in row:
        return bool(row.get("grasp_probe_near_basin_shell", False))
    return bool(
        _probe_horizon_xy_feasible(
            row,
            near_grasp_xy_threshold=near_grasp_xy_threshold,
            max_xy_step=max_xy_step,
            horizon_steps=horizon_steps,
        )
        and _probe_yaw_observable(row, near_grasp_yaw_threshold=near_grasp_yaw_threshold)
    )


def _probe_coarse_pullback_candidate(
    row: Mapping[str, Any],
    *,
    near_grasp_xy_threshold: float,
    near_grasp_yaw_threshold: float = 0.08,
    max_xy_step: float,
    horizon_steps: int,
) -> bool:
    if bool(row.get("grasp_probe_coarse_pullback_candidate", False)):
        return True
    if _row_group_value(row, "grasp_probe_visibility_bucket", "prior_only") == "prior_only":
        return False
    if _probe_near_basin_shell(
        row,
        near_grasp_xy_threshold=near_grasp_xy_threshold,
        near_grasp_yaw_threshold=near_grasp_yaw_threshold,
        max_xy_step=max_xy_step,
        horizon_steps=horizon_steps,
    ):
        return False
    pre_xy = _probe_pre_xy(row)
    near_shell_xy = float(near_grasp_xy_threshold) + float(max_xy_step) * float(max(1, int(horizon_steps)))
    return bool(np.isfinite(pre_xy) and near_shell_xy < pre_xy <= 0.060)


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(_row_group_value(row, key) for key in keys)].append(row)
    return groups


def _probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("grasp_probe_policy", "off")) != "off"]


def _active_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("grasp_probe_active", False))]


def _plot_episode(ep_tag: str, rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    steps = [int(r.get("step", r.get("step_idx", -1))) for r in rows]
    pre = np.asarray([_trace_row_error(r, "grasp_probe_pre_true_error_t") for r in rows], dtype=np.float32)
    post = np.asarray([_trace_row_error(r, "grasp_probe_post_true_error_t") for r in rows], dtype=np.float32)
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    labels = ["dx", "dy", "dz", "dyaw"]
    for idx, ax in enumerate(axes[:4]):
        ax.plot(steps, pre[:, idx], label=f"pre_{labels[idx]}", linewidth=1.5)
        ax.plot(steps, post[:, idx], label=f"post_{labels[idx]}", linewidth=1.5)
        active_steps = [s for s, r in zip(steps, rows) if bool(r.get("grasp_probe_active", False))]
        active_post = [float(_trace_row_error(r, "grasp_probe_post_true_error_t")[idx]) for r in rows if bool(r.get("grasp_probe_active", False))]
        if active_steps:
            ax.scatter(active_steps, active_post, s=18, alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(labels[idx])
        ax.legend(loc="upper right")
    xy_pre = np.asarray([_xy_norm(r.get("grasp_probe_pre_true_error_t", [np.nan, np.nan])) for r in rows], dtype=np.float32)
    xy_post = np.asarray([_xy_norm(r.get("grasp_probe_post_true_error_t", [np.nan, np.nan])) for r in rows], dtype=np.float32)
    axes[4].plot(steps, xy_pre, label="pre_xy", linewidth=1.5)
    axes[4].plot(steps, xy_post, label="post_xy", linewidth=1.5)
    axes[4].set_ylabel("xy_norm")
    axes[4].set_xlabel("step")
    axes[4].legend(loc="upper right")
    fig.suptitle(f"{ep_tag} grasp probe intervention")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ep_tag}_grasp_probe_intervention.png", dpi=160)
    plt.close(fig)


def _bucket_summary(
    rows: list[dict[str, Any]],
    *,
    near_grasp_xy_threshold: float,
    near_grasp_yaw_threshold: float,
    max_xy_step: float,
    horizon_steps: int,
) -> dict[str, Any]:
    active = _active_probe_rows(rows)
    prior_only = [r for r in rows if _row_group_value(r, "grasp_probe_visibility_bucket", "prior_only") == "prior_only"]
    visual = [r for r in rows if _row_group_value(r, "grasp_probe_visibility_bucket", "") == "visual_observable"]
    partial = [r for r in rows if _row_group_value(r, "grasp_probe_visibility_bucket", "") == "partial_observable"]
    active_xy = [r for r in active if np.all(np.isfinite(_trace_row_error(r, "grasp_probe_pre_true_error_t")[:2])) and np.all(np.isfinite(_trace_row_error(r, "grasp_probe_post_true_error_t")[:2]))]
    xy_contracted = [bool(_xy_norm(_trace_row_error(r, "grasp_probe_post_true_error_t")[:2]) < _xy_norm(_trace_row_error(r, "grasp_probe_pre_true_error_t")[:2]) - 1.0e-9) for r in active_xy]
    active_count = len(active)
    contracted_count = int(np.count_nonzero(xy_contracted))
    active_xy_rate = float(np.mean(xy_contracted)) if xy_contracted else 0.0
    micro_after = float(np.mean([bool(r.get("grasp_probe_micro_entry_ready_after", False)) for r in active])) if active else 0.0
    near_after = float(np.mean([bool(r.get("grasp_probe_near_grasp_after", False)) for r in active])) if active else 0.0
    close_after = float(np.mean([bool(r.get("grasp_probe_close_ready_after", False)) for r in active])) if active else 0.0
    overshoot = float(np.mean([bool(r.get("grasp_probe_overshoot", False)) for r in active])) if active else 0.0
    horizon_micro_after = float(np.mean([_probe_bool(r, "micro_entry_ready_after", horizon=True) for r in active])) if active else 0.0
    horizon_near_after = float(np.mean([_probe_bool(r, "near_grasp_after", horizon=True) for r in active])) if active else 0.0
    horizon_close_after = float(np.mean([_probe_bool(r, "close_ready_after", horizon=True) for r in active])) if active else 0.0
    horizon_overshoot = float(np.mean([_probe_bool(r, "overshoot", horizon=True) for r in active])) if active else 0.0
    prior_abstain = float(np.mean([_row_group_value(r, "grasp_probe_reason", "") == "prior_only_abstain" for r in prior_only])) if prior_only else 0.0

    def _hist_key(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "+".join(str(x) for x in value)
        return str(value)

    return {
        "count": int(len(rows)),
        "active_count": int(active_count),
        "xy_contraction_rate": float(active_xy_rate),
        "xy_contraction_lower_ci": float(_wilson_lower_bound(contracted_count, len(active_xy))),
        "micro_entry_ready_after_rate": micro_after,
        "near_grasp_after_rate": near_after,
        "close_ready_after_rate": close_after,
        "overshoot_rate": overshoot,
        "horizon_xy_contraction_rate": _horizon_xy_contraction_rate(active),
        "horizon_micro_entry_ready_after_rate": horizon_micro_after,
        "horizon_near_grasp_after_rate": horizon_near_after,
        "horizon_close_ready_after_rate": horizon_close_after,
        "horizon_overshoot_rate": horizon_overshoot,
        "prior_only_abstain_rate": prior_abstain,
        "mean_delta_dx": _mean_step_delta(active, "x"),
        "mean_delta_dy": _mean_step_delta(active, "y"),
        "mean_delta_dz": _mean_step_delta(active, "z"),
        "mean_delta_dyaw": _mean_step_delta(active, "yaw"),
        "axis_abs_contraction_rate": {
            "x": _axis_abs_contraction_rate(active, "x"),
            "y": _axis_abs_contraction_rate(active, "y"),
            "z": _axis_abs_contraction_rate(active, "z"),
            "yaw": _axis_abs_contraction_rate(active, "yaw"),
        },
        "visual_observable_count": int(len(visual)),
        "partial_observable_count": int(len(partial)),
        "prior_only_count": int(len(prior_only)),
        "active_xy_rows": int(len(active_xy)),
        "near_basin_shell_rows": int(sum(_probe_near_basin_shell(r, near_grasp_xy_threshold=near_grasp_xy_threshold, near_grasp_yaw_threshold=near_grasp_yaw_threshold, max_xy_step=max_xy_step, horizon_steps=horizon_steps) for r in rows)),
        "coarse_pullback_candidate_rows": int(sum(_probe_coarse_pullback_candidate(r, near_grasp_xy_threshold=near_grasp_xy_threshold, near_grasp_yaw_threshold=near_grasp_yaw_threshold, max_xy_step=max_xy_step, horizon_steps=horizon_steps) for r in rows)),
        "yaw_feasible_rows": int(sum(_probe_yaw_observable(r, near_grasp_yaw_threshold=near_grasp_yaw_threshold) for r in rows)),
        "yaw_observable_rows": int(sum(_probe_yaw_observable(r, near_grasp_yaw_threshold=near_grasp_yaw_threshold) for r in rows)),
        "horizon_xy_feasible_rows": int(sum(_probe_horizon_xy_feasible(r, near_grasp_xy_threshold=near_grasp_xy_threshold, max_xy_step=max_xy_step, horizon_steps=horizon_steps) for r in rows)),
        "micro_entry_ready_rows": int(sum(_probe_bool(r, "micro_entry_ready_after", horizon=True) for r in rows)),
        "max_requested_horizon": int(max([int(r.get("grasp_probe_requested_horizon", 1) or 1) for r in active], default=1)),
        "mean_horizon_steps_executed": float(np.mean([int(r.get("grasp_probe_horizon_steps_executed", 0) or 0) for r in active])) if active else 0.0,
        "yaw_blocked_rate_within_horizon_xy_feasible": _yaw_blocked_rate_within_horizon_xy_feasible(
            rows,
            near_grasp_xy_threshold=near_grasp_xy_threshold,
            near_grasp_yaw_threshold=near_grasp_yaw_threshold,
            max_xy_step=max_xy_step,
            horizon_steps=horizon_steps,
        ),
        "recover_preempt_rate_before_first_probe_step": _recover_preempt_rate_before_first_probe_step(rows),
        "ring_grasp_align_dwell_steps": _ring_grasp_align_dwell_steps(rows),
        "queue_protocol": _queue_protocol_summary(rows),
        "feasible_shells": _shell_summary(
            rows,
            near_grasp_xy_threshold=near_grasp_xy_threshold,
            near_grasp_yaw_threshold=near_grasp_yaw_threshold,
            max_xy_step=max_xy_step,
            horizon_steps=horizon_steps,
        ),
        "active_gate_axes_hist": dict(Counter(_hist_key(r.get("grasp_probe_control_gate_axes", [])) for r in active)),
        "active_pullback_axes_hist": dict(Counter(_hist_key(r.get("grasp_probe_pullback_ready_axes", [])) for r in active)),
        "stage_source_counts": dict(Counter(_row_group_value(r, "grasp_probe_stage_source", "") for r in rows)),
        "reason_counts": dict(Counter(_row_group_value(r, "grasp_probe_reason", "") for r in rows)),
    }


def _active_xy_contracted(row: Mapping[str, Any]) -> bool:
    pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
    post = _trace_row_error(row, "grasp_probe_post_true_error_t")
    if not np.all(np.isfinite(pre[:2])) or not np.all(np.isfinite(post[:2])):
        return False
    return bool(_xy_norm(post[:2]) < _xy_norm(pre[:2]) - 1.0e-9)


def audit(
    rows: list[dict[str, Any]],
    *,
    near_grasp_xy_threshold: float = 0.015,
    near_grasp_yaw_threshold: float = 0.08,
    max_xy_step: float = 0.003,
    horizon_steps: int = 3,
) -> dict[str, Any]:
    probe_rows = _probe_rows(rows)
    active_rows = _active_probe_rows(rows)
    by_bucket: list[dict[str, Any]] = []
    for key, subset in sorted(_group_rows(probe_rows, ("failure_bucket",)).items()):
        by_bucket.append({
            "failure_bucket": key[0],
            **_bucket_summary(
                subset,
                near_grasp_xy_threshold=near_grasp_xy_threshold,
                near_grasp_yaw_threshold=near_grasp_yaw_threshold,
                max_xy_step=max_xy_step,
                horizon_steps=horizon_steps,
            ),
        })

    by_visual: list[dict[str, Any]] = []
    for key, subset in sorted(_group_rows(probe_rows, ("grasp_probe_visibility_bucket",)).items()):
        by_visual.append({
            "grasp_probe_visibility_bucket": key[0],
            **_bucket_summary(
                subset,
                near_grasp_xy_threshold=near_grasp_xy_threshold,
                near_grasp_yaw_threshold=near_grasp_yaw_threshold,
                max_xy_step=max_xy_step,
                horizon_steps=horizon_steps,
            ),
        })

    by_episode: list[dict[str, Any]] = []
    episode_groups = list(_group_rows(probe_rows, ("episode_idx",)).items())
    episode_groups.sort(key=lambda item: int(item[0][0]) if str(item[0][0]).lstrip("-").isdigit() else -1)
    for key, subset in episode_groups:
        by_episode.append({
            "episode_idx": int(key[0]) if str(key[0]).lstrip("-").isdigit() else -1,
            **_bucket_summary(
                subset,
                near_grasp_xy_threshold=near_grasp_xy_threshold,
                near_grasp_yaw_threshold=near_grasp_yaw_threshold,
                max_xy_step=max_xy_step,
                horizon_steps=horizon_steps,
            ),
        })

    by_episode_failure_bucket: list[dict[str, Any]] = []
    episode_bucket_groups = list(_group_rows(probe_rows, ("episode_idx", "failure_bucket")).items())
    episode_bucket_groups.sort(
        key=lambda item: (
            int(item[0][0]) if str(item[0][0]).lstrip("-").isdigit() else -1,
            str(item[0][1]),
        )
    )
    for key, subset in episode_bucket_groups:
        by_episode_failure_bucket.append({
            "episode_idx": int(key[0]) if str(key[0]).lstrip("-").isdigit() else -1,
            "failure_bucket": key[1],
            **_bucket_summary(
                subset,
                near_grasp_xy_threshold=near_grasp_xy_threshold,
                near_grasp_yaw_threshold=near_grasp_yaw_threshold,
                max_xy_step=max_xy_step,
                horizon_steps=horizon_steps,
            ),
        })

    overall = {
        "num_rows": int(len(rows)),
        "probe_rows": int(len(probe_rows)),
        "active_rows": int(len(active_rows)),
        "xy_contraction_rate": float(np.mean([_active_xy_contracted(r) for r in active_rows])) if active_rows else 0.0,
        "xy_contraction_lower_ci": float(_wilson_lower_bound(
            int(np.count_nonzero([_active_xy_contracted(r) for r in active_rows])),
            len(active_rows),
        )),
        "micro_entry_ready_after_rate": float(np.mean([bool(r.get("grasp_probe_micro_entry_ready_after", False)) for r in active_rows])) if active_rows else 0.0,
        "near_grasp_after_rate": float(np.mean([bool(r.get("grasp_probe_near_grasp_after", False)) for r in active_rows])) if active_rows else 0.0,
        "close_ready_after_rate": float(np.mean([bool(r.get("grasp_probe_close_ready_after", False)) for r in active_rows])) if active_rows else 0.0,
        "overshoot_rate": float(np.mean([bool(r.get("grasp_probe_overshoot", False)) for r in active_rows])) if active_rows else 0.0,
        "horizon_xy_contraction_rate": _horizon_xy_contraction_rate(active_rows),
        "horizon_micro_entry_ready_after_rate": float(np.mean([_probe_bool(r, "micro_entry_ready_after", horizon=True) for r in active_rows])) if active_rows else 0.0,
        "horizon_near_grasp_after_rate": float(np.mean([_probe_bool(r, "near_grasp_after", horizon=True) for r in active_rows])) if active_rows else 0.0,
        "horizon_close_ready_after_rate": float(np.mean([_probe_bool(r, "close_ready_after", horizon=True) for r in active_rows])) if active_rows else 0.0,
        "horizon_overshoot_rate": float(np.mean([_probe_bool(r, "overshoot", horizon=True) for r in active_rows])) if active_rows else 0.0,
        "prior_only_abstain_rate": float(np.mean([
            _row_group_value(r, "grasp_probe_reason", "") == "prior_only_abstain"
            for r in probe_rows
            if _row_group_value(r, "grasp_probe_visibility_bucket", "prior_only") == "prior_only"
        ])) if any(_row_group_value(r, "grasp_probe_visibility_bucket", "prior_only") == "prior_only" for r in probe_rows) else 0.0,
        "reacquire_rate": float(np.mean([_row_group_value(r, "grasp_probe_reason", "") == "prior_only_abstain" or _row_group_value(r, "grasp_probe_reason", "") == "inactive" for r in probe_rows])) if probe_rows else 0.0,
        "max_requested_horizon": int(max([int(r.get("grasp_probe_requested_horizon", 1) or 1) for r in active_rows], default=1)),
        "mean_horizon_steps_executed": float(np.mean([int(r.get("grasp_probe_horizon_steps_executed", 0) or 0) for r in active_rows])) if active_rows else 0.0,
        "near_basin_shell_rows": int(sum(_probe_near_basin_shell(r, near_grasp_xy_threshold=near_grasp_xy_threshold, near_grasp_yaw_threshold=near_grasp_yaw_threshold, max_xy_step=max_xy_step, horizon_steps=horizon_steps) for r in probe_rows)),
        "coarse_pullback_candidate_rows": int(sum(_probe_coarse_pullback_candidate(r, near_grasp_xy_threshold=near_grasp_xy_threshold, near_grasp_yaw_threshold=near_grasp_yaw_threshold, max_xy_step=max_xy_step, horizon_steps=horizon_steps) for r in probe_rows)),
        "yaw_feasible_rows": int(sum(_probe_yaw_observable(r, near_grasp_yaw_threshold=near_grasp_yaw_threshold) for r in probe_rows)),
        "yaw_observable_rows": int(sum(_probe_yaw_observable(r, near_grasp_yaw_threshold=near_grasp_yaw_threshold) for r in probe_rows)),
        "horizon_xy_feasible_rows": int(sum(_probe_horizon_xy_feasible(r, near_grasp_xy_threshold=near_grasp_xy_threshold, max_xy_step=max_xy_step, horizon_steps=horizon_steps) for r in probe_rows)),
        "micro_entry_ready_rows": int(sum(_probe_bool(r, "micro_entry_ready_after", horizon=True) for r in probe_rows)),
        "yaw_blocked_rate_within_horizon_xy_feasible": _yaw_blocked_rate_within_horizon_xy_feasible(
            probe_rows,
            near_grasp_xy_threshold=near_grasp_xy_threshold,
            near_grasp_yaw_threshold=near_grasp_yaw_threshold,
            max_xy_step=max_xy_step,
            horizon_steps=horizon_steps,
        ),
        "recover_preempt_rate_before_first_probe_step": _recover_preempt_rate_before_first_probe_step(probe_rows),
        "ring_grasp_align_dwell_steps": _ring_grasp_align_dwell_steps(probe_rows),
        "queue_protocol": _queue_protocol_summary(probe_rows),
        "feasible_shells": _shell_summary(
            probe_rows,
            near_grasp_xy_threshold=near_grasp_xy_threshold,
            near_grasp_yaw_threshold=near_grasp_yaw_threshold,
            max_xy_step=max_xy_step,
            horizon_steps=horizon_steps,
        ),
        "axis_abs_contraction_rate": {
            "x": _axis_abs_contraction_rate(active_rows, "x"),
            "y": _axis_abs_contraction_rate(active_rows, "y"),
            "z": _axis_abs_contraction_rate(active_rows, "z"),
            "yaw": _axis_abs_contraction_rate(active_rows, "yaw"),
        },
        "mean_delta": {
            "dx": _mean_step_delta(active_rows, "x"),
            "dy": _mean_step_delta(active_rows, "y"),
            "dz": _mean_step_delta(active_rows, "z"),
            "dyaw": _mean_step_delta(active_rows, "yaw"),
        },
        "reason_counts": dict(Counter(_row_group_value(r, "grasp_probe_reason", "") for r in probe_rows)),
        "stage_source_counts": dict(Counter(_row_group_value(r, "grasp_probe_stage_source", "") for r in probe_rows)),
    }

    return {
        "audit_config": {
            "near_grasp_xy_threshold": float(near_grasp_xy_threshold),
            "near_grasp_yaw_threshold": float(near_grasp_yaw_threshold),
            "max_xy_step": float(max_xy_step),
            "horizon_steps": int(horizon_steps),
        },
        "overall": overall,
        "by_failure_bucket": by_bucket,
        "by_visibility_bucket": by_visual,
        "by_episode": by_episode,
        "by_episode_failure_bucket": by_episode_failure_bucket,
        "runtime_invariants": {
            "uses_privileged_target": False,
            "uses_privileged_runtime": False,
            "uses_privileged_label_for_eval": True,
            "uses_rlbench_mask_runtime": False,
        },
    }


def _plot_overview(report: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["xy_contract", "micro_ready", "near_grasp", "horizon_near", "overshoot"]
    bars = [
        report["overall"]["xy_contraction_rate"],
        report["overall"]["micro_entry_ready_after_rate"],
        report["overall"]["near_grasp_after_rate"],
        report["overall"]["horizon_near_grasp_after_rate"],
        report["overall"]["overshoot_rate"],
    ]
    ax.bar(labels, bars, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("C2C grasp probe intervention summary")
    fig.tight_layout()
    fig.savefig(output_dir / "grasp_probe_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_intervention"),
    )
    ap.add_argument("--near_grasp_xy_threshold", type=float, default=0.015)
    ap.add_argument("--near_grasp_yaw_threshold", type=float, default=0.08)
    ap.add_argument("--max_xy_step", type=float, default=0.003)
    ap.add_argument("--horizon_steps", type=int, default=3)
    args = ap.parse_args()

    trace_dir = args.trace_dir.resolve()
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing trace_dir: {trace_dir}")

    rows = _load_trace_rows(trace_dir)
    if not rows:
        raise RuntimeError(f"No trace rows found in {trace_dir}")
    probe_rows = _probe_rows(rows)
    if not probe_rows:
        raise RuntimeError(
            f"No grasp probe rows found in {trace_dir}. "
            "Re-run evaluate_c2c_v2_rlbench.py with --c2c_grasp_probe_policy replay_oracle_xy."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    report = audit(
        rows,
        near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
        near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
        max_xy_step=float(args.max_xy_step),
        horizon_steps=int(max(1, int(args.horizon_steps))),
    )
    report["source_trace_dir"] = str(trace_dir)
    report["num_trace_files"] = int(len(list(trace_dir.glob("ep*_gripper_trace.jsonl"))))

    out_json = output_dir / "grasp_probe_intervention_audit.json"
    out_md = output_dir / "grasp_probe_intervention_audit.md"
    out_rows = output_dir / "grasp_probe_intervention_rows.jsonl"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _plot_overview(report, output_dir)

    with open(out_rows, "w", encoding="utf-8") as handle:
        for row in probe_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            ep_tag = f"ep{int(row.get('episode_idx', -1)):03d}"
    for key, subset in sorted(_group_rows(probe_rows, ("episode_idx",)).items()):
        ep_tag = f"ep{int(key[0]):03d}" if str(key[0]).lstrip("-").isdigit() else f"episode_{key[0]}"
        _plot_episode(ep_tag, subset, plot_dir)

    md_lines = [
        "# Grasp Intervention Audit",
        "",
        f"- source_trace_dir: `{trace_dir}`",
        f"- trace_files: `{report['num_trace_files']}`",
        f"- rows: `{report['overall']['num_rows']}`",
        f"- probe_rows: `{report['overall']['probe_rows']}`",
        f"- active_rows: `{report['overall']['active_rows']}`",
        f"- horizon_steps: `{report['audit_config']['horizon_steps']}`",
        "",
        "## Overall",
        f"- xy_contraction_rate: `{report['overall']['xy_contraction_rate']:.3f}`",
        f"- xy_contraction_lower_ci: `{report['overall']['xy_contraction_lower_ci']:.3f}`",
        f"- micro_entry_ready_after_rate: `{report['overall']['micro_entry_ready_after_rate']:.3f}`",
        f"- near_grasp_after_rate: `{report['overall']['near_grasp_after_rate']:.3f}`",
        f"- close_ready_after_rate: `{report['overall']['close_ready_after_rate']:.3f}`",
        f"- overshoot_rate: `{report['overall']['overshoot_rate']:.3f}`",
        f"- horizon_xy_contraction_rate: `{report['overall']['horizon_xy_contraction_rate']:.3f}`",
        f"- horizon_micro_entry_ready_after_rate: `{report['overall']['horizon_micro_entry_ready_after_rate']:.3f}`",
        f"- horizon_near_grasp_after_rate: `{report['overall']['horizon_near_grasp_after_rate']:.3f}`",
        f"- horizon_overshoot_rate: `{report['overall']['horizon_overshoot_rate']:.3f}`",
        f"- mean_horizon_steps_executed: `{report['overall']['mean_horizon_steps_executed']:.2f}`",
        f"- coarse_pullback_candidate_rows: `{report['overall']['coarse_pullback_candidate_rows']}`",
        f"- near_basin_shell_rows: `{report['overall']['near_basin_shell_rows']}`",
        f"- horizon_xy_feasible_rows: `{report['overall']['horizon_xy_feasible_rows']}`",
        f"- yaw_feasible_rows: `{report['overall']['yaw_feasible_rows']}`",
        f"- yaw_observable_rows: `{report['overall']['yaw_observable_rows']}`",
        f"- micro_entry_ready_rows: `{report['overall']['micro_entry_ready_rows']}`",
        f"- yaw_blocked_rate_within_horizon_xy_feasible: `{report['overall']['yaw_blocked_rate_within_horizon_xy_feasible']:.3f}`",
        f"- recover_preempt_rate_before_first_probe_step: `{report['overall']['recover_preempt_rate_before_first_probe_step']:.3f}`",
        f"- prior_only_abstain_rate: `{report['overall']['prior_only_abstain_rate']:.3f}`",
        f"- ring_grasp_align_dwell_total: `{report['overall']['ring_grasp_align_dwell_steps']['total']}`",
        f"- queue_flushed_rate: `{report['overall']['queue_protocol']['queue_flushed_rate']:.3f}`",
        f"- queue_flush_ablation_delta: `{report['overall']['queue_protocol']['queue_flush_ablation_delta']:.3f}`",
        "",
        "## Feasible Shells",
    ]
    for name, item in report["overall"]["feasible_shells"].items():
        md_lines.append(f"- `{name}`")
        md_lines.append(f"  - count: `{item['count']}`")
        md_lines.append(f"  - rate_of_active: `{item['rate_of_active']:.3f}`")
        md_lines.append(f"  - horizon_near_grasp_after_rate: `{item['horizon_near_grasp_after_rate']:.3f}`")
        md_lines.append(f"  - horizon_xy_contraction_rate: `{item['horizon_xy_contraction_rate']:.3f}`")
        md_lines.append(f"  - overshoot_rate: `{item['overshoot_rate']:.3f}`")
    md_lines.extend([
        "",
        "## Axis Contraction",
    ])
    for axis, value in report["overall"]["axis_abs_contraction_rate"].items():
        md_lines.append(f"- `{axis}`: `{float(value):.3f}`")
    md_lines.append("")
    md_lines.append("## Failure Buckets")
    for item in report["by_failure_bucket"]:
        md_lines.append(f"- `{item['failure_bucket']}`")
        md_lines.append(f"  - count: `{item['count']}`")
        md_lines.append(f"  - active_count: `{item['active_count']}`")
        md_lines.append(f"  - xy_contraction_rate: `{item['xy_contraction_rate']:.3f}`")
        md_lines.append(f"  - xy_contraction_lower_ci: `{item['xy_contraction_lower_ci']:.3f}`")
        md_lines.append(f"  - micro_entry_ready_after_rate: `{item['micro_entry_ready_after_rate']:.3f}`")
        md_lines.append(f"  - near_grasp_after_rate: `{item['near_grasp_after_rate']:.3f}`")
        md_lines.append(f"  - close_ready_after_rate: `{item['close_ready_after_rate']:.3f}`")
        md_lines.append(f"  - overshoot_rate: `{item['overshoot_rate']:.3f}`")
        md_lines.append(f"  - horizon_near_grasp_after_rate: `{item['horizon_near_grasp_after_rate']:.3f}`")
        md_lines.append(f"  - horizon_xy_contraction_rate: `{item['horizon_xy_contraction_rate']:.3f}`")
        md_lines.append(f"  - prior_only_abstain_rate: `{item['prior_only_abstain_rate']:.3f}`")
        md_lines.append(
            "  - axis_abs_contraction_rate: "
            + ", ".join(f"{axis}={float(val):.3f}" for axis, val in item["axis_abs_contraction_rate"].items())
        )
    md_lines.append("")
    md_lines.append("## Episode Buckets")
    for item in report["by_episode_failure_bucket"]:
        md_lines.append(f"- `ep{int(item['episode_idx']):03d}` / `{item['failure_bucket']}`")
        md_lines.append(f"  - count: `{item['count']}`")
        md_lines.append(f"  - active_count: `{item['active_count']}`")
        md_lines.append(f"  - coarse_pullback_candidate_rows: `{item['coarse_pullback_candidate_rows']}`")
        md_lines.append(f"  - near_basin_shell_rows: `{item['near_basin_shell_rows']}`")
        md_lines.append(f"  - yaw_feasible_rows: `{item['yaw_feasible_rows']}`")
        md_lines.append(f"  - yaw_observable_rows: `{item['yaw_observable_rows']}`")
        md_lines.append(f"  - micro_entry_ready_rows: `{item['micro_entry_ready_rows']}`")
        md_lines.append(f"  - horizon_xy_feasible_rows: `{item['horizon_xy_feasible_rows']}`")
        md_lines.append(f"  - horizon_near_grasp_after_rate: `{item['horizon_near_grasp_after_rate']:.3f}`")
        md_lines.append(f"  - queue_flush_ablation_delta: `{item['queue_protocol']['queue_flush_ablation_delta']:.3f}`")
    md_lines.append("")
    md_lines.append("## Visibility Buckets")
    for item in report["by_visibility_bucket"]:
        md_lines.append(f"- `{item['grasp_probe_visibility_bucket']}`")
        md_lines.append(f"  - xy_contraction_rate: `{item['xy_contraction_rate']:.3f}`")
        md_lines.append(f"  - micro_entry_ready_after_rate: `{item['micro_entry_ready_after_rate']:.3f}`")
        md_lines.append(f"  - near_grasp_after_rate: `{item['near_grasp_after_rate']:.3f}`")
        md_lines.append(f"  - close_ready_after_rate: `{item['close_ready_after_rate']:.3f}`")
        md_lines.append(f"  - overshoot_rate: `{item['overshoot_rate']:.3f}`")
        md_lines.append(f"  - horizon_near_grasp_after_rate: `{item['horizon_near_grasp_after_rate']:.3f}`")
        md_lines.append(f"  - horizon_xy_contraction_rate: `{item['horizon_xy_contraction_rate']:.3f}`")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(out_json)
    print(out_md)
    print(out_rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
