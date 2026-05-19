#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np


INPUT_KEYS = [
    "front_rgb",
    "wrist_rgb",
    "wrist_depth",
    "proprio",
    "gripper_context",
    "proxy_current_delta_basin_target",
    "current_dx_sign",
    "current_dy_sign",
    "current_dyaw_sign",
    "basin_distance_bin",
    "substage_id",
    "contact_state",
    "stage_target_mode",
    "episode_index",
]


def _parse_csv_ints(text: str | None) -> set[int]:
    out: set[int] = set()
    if not text:
        return out
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def _load_concat_npz(paths: list[str]) -> dict[str, np.ndarray]:
    chunks = []
    for path in paths:
        raw = np.load(path, allow_pickle=False)
        chunks.append({k: np.asarray(raw[k]) for k in raw.files})
    keys = sorted(set().union(*(c.keys() for c in chunks)))
    data: dict[str, np.ndarray] = {}
    for key in keys:
        exemplar = next((c[key] for c in chunks if key in c), None)
        if exemplar is None:
            continue
        arrs = []
        for c in chunks:
            n_c = int(next(iter(c.values())).shape[0])
            if key in c and tuple(np.asarray(c[key]).shape[1:]) == tuple(exemplar.shape[1:]):
                arrs.append(np.asarray(c[key]))
            else:
                shape = (n_c,) + tuple(exemplar.shape[1:])
                if exemplar.dtype.kind in ("U", "S", "O"):
                    arrs.append(np.full(shape, "", dtype=exemplar.dtype))
                else:
                    arrs.append(np.zeros(shape, dtype=exemplar.dtype))
        data[key] = np.concatenate(arrs, axis=0)
    return data


def _yaw_bucket(abs_yaw: float, keep_abs: float, small_abs: float, large_abs: float) -> str:
    if abs_yaw < keep_abs:
        return "no_yaw"
    if abs_yaw < small_abs:
        return "small_yaw"
    if abs_yaw < large_abs:
        return "medium_yaw"
    return "large_yaw"


def _axis_dominance(action: np.ndarray) -> str:
    xy = float(np.hypot(float(action[0]), float(action[1])))
    z = float(abs(float(action[2])))
    yaw = float(abs(float(action[5])))
    axis = int(np.argmax(np.asarray([xy, z, yaw], dtype=np.float32)))
    return ("xy", "z", "yaw")[axis]


def _classify_row(
    pred_action: np.ndarray,
    oracle_action: np.ndarray,
    baseline_action: np.ndarray,
    oracle_scores: np.ndarray,
    candidate_mask: np.ndarray,
    shadow_mode: int,
    pred_regret: float,
    baseline_regret: float,
    keep_yaw_abs: float,
    small_yaw_abs: float,
    large_yaw_abs: float,
    candidate_score_std_min: float,
    oracle_baseline_gap_min: float,
    mode_keep_margin: float,
    yaw_needed: bool,
    include_candidate_bank_missing: bool,
) -> list[str]:
    valid = candidate_mask > 0.5
    score = oracle_scores[valid]
    score = score[np.isfinite(score)]
    score_std = float(np.std(score)) if score.size else 0.0
    score_gap = float(np.max(score) - np.min(score)) if score.size else 0.0
    if score_std < float(candidate_score_std_min) or score_gap < float(oracle_baseline_gap_min):
        if include_candidate_bank_missing:
            return ["candidate_bank_missing"]
        return []

    pred_yaw = float(abs(float(pred_action[5])))
    oracle_yaw = float(abs(float(oracle_action[5])))
    base_yaw = float(abs(float(baseline_action[5])))

    pred_bucket = _yaw_bucket(pred_yaw, keep_yaw_abs, small_yaw_abs, large_yaw_abs)
    oracle_bucket = _yaw_bucket(oracle_yaw, keep_yaw_abs, small_yaw_abs, large_yaw_abs)
    base_bucket = _yaw_bucket(base_yaw, keep_yaw_abs, small_yaw_abs, large_yaw_abs)
    oracle_axis = _axis_dominance(oracle_action)
    pred_axis = _axis_dominance(pred_action)

    labels: list[str] = []
    if math.isfinite(float(pred_regret)) and math.isfinite(float(baseline_regret)):
        advantage = float(baseline_regret) - float(pred_regret)
        if advantage <= float(mode_keep_margin):
            labels.append("baseline_preserve")
    if math.isfinite(float(pred_regret)) and math.isfinite(float(baseline_regret)):
        if int(shadow_mode) == 0 and float(pred_regret) > float(baseline_regret) + 1e-6:
            labels.append("mode_keep_failure")
        elif int(shadow_mode) == 1 and float(pred_regret) > float(baseline_regret) + 1e-6:
            labels.append("mode_apply_failure")
    if yaw_needed and pred_bucket in {"no_yaw", "small_yaw"} and oracle_bucket in {"small_yaw", "medium_yaw", "large_yaw"}:
        labels.append("yaw_needed_missing")
    if yaw_needed and pred_bucket == "large_yaw" and oracle_bucket in {"no_yaw", "small_yaw"}:
        labels.extend(["large_yaw_overuse", "small_vs_large_yaw"])
        if oracle_axis == "xy":
            labels.append("xy_over_yaw")
        elif oracle_axis == "z":
            labels.append("z_over_yaw")
        labels.append("yaw_needed_overuse")
    elif pred_bucket in {"medium_yaw", "large_yaw"} and oracle_bucket in {"no_yaw", "small_yaw"}:
        labels.append("yaw_not_needed_but_selected")
        if oracle_axis == "xy":
            labels.append("xy_over_yaw")
        elif oracle_axis == "z":
            labels.append("z_over_yaw")

    if pred_bucket in {"medium_yaw", "large_yaw"} and oracle_bucket in {"medium_yaw", "large_yaw"}:
        if np.sign(float(pred_action[5])) != np.sign(float(oracle_action[5])) and abs(float(oracle_action[5])) >= float(small_yaw_abs):
            labels.append("wrong_yaw_sign")

    if oracle_bucket in {"medium_yaw", "large_yaw"} and pred_bucket in {"no_yaw", "small_yaw"}:
        labels.append("yaw_needed_but_not_selected")

    if not yaw_needed and pred_bucket in {"medium_yaw", "large_yaw"}:
        labels.append("mode_apply_overuse")

    if not labels:
        if pred_axis != oracle_axis:
            labels.append(f"{pred_axis}_vs_{oracle_axis}")
        else:
            labels.append("generic_negative")

    if base_bucket != oracle_bucket and oracle_bucket in {"medium_yaw", "large_yaw"}:
        labels.append("baseline_yaw_mismatch")

    return labels


def _primary_label(labels: list[str]) -> str:
    priority = [
        "candidate_bank_missing",
        "mode_keep_failure",
        "mode_apply_failure",
        "yaw_needed_missing",
        "wrong_yaw_sign",
        "mode_apply_overuse",
        "baseline_preserve",
        "large_yaw_overuse",
        "small_vs_large_yaw",
        "yaw_not_needed_but_selected",
        "yaw_needed_but_not_selected",
        "xy_over_yaw",
        "z_over_yaw",
        "generic_negative",
    ]
    for key in priority:
        if key in labels:
            return key
    return labels[0] if labels else "generic_negative"


def _type_weight(
    label: str,
    worse_weight: float,
    better_weight: float,
    baseline_preserve_weight: float,
    mode_keep_failure_weight: float,
    mode_apply_failure_weight: float,
    large_yaw_negative_weight: float,
    wrong_yaw_sign_weight: float,
    xy_over_yaw_weight: float,
    z_over_yaw_weight: float,
    yaw_needed_but_not_selected_weight: float,
    hard_episode: bool,
    hard_episode_weight: float,
) -> float:
    if label == "candidate_bank_missing":
        return 0.0
    if label == "baseline_preserve":
        w = float(baseline_preserve_weight)
    if label == "mode_keep_failure":
        w = float(mode_keep_failure_weight)
    elif label == "mode_apply_failure":
        w = float(mode_apply_failure_weight)
    elif label == "mode_apply_overuse":
        w = float(large_yaw_negative_weight)
    elif label == "yaw_needed_missing":
        w = float(yaw_needed_but_not_selected_weight)
    elif label == "large_yaw_overuse":
        w = float(large_yaw_negative_weight)
    elif label == "wrong_yaw_sign":
        w = float(wrong_yaw_sign_weight)
    elif label == "small_vs_large_yaw":
        w = float(large_yaw_negative_weight)
    elif label == "xy_over_yaw":
        w = float(xy_over_yaw_weight)
    elif label == "z_over_yaw":
        w = float(z_over_yaw_weight)
    elif label == "yaw_needed_but_not_selected":
        w = float(yaw_needed_but_not_selected_weight)
    else:
        w = float(worse_weight if label != "positive_preserve" else better_weight)
    if hard_episode and label != "positive_preserve":
        w *= float(max(hard_episode_weight, 1.0))
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    ap.add_argument("--small_yaw_abs", type=float, default=0.05)
    ap.add_argument("--large_yaw_abs", type=float, default=0.09)
    ap.add_argument("--mode_keep_margin", type=float, default=0.05)
    ap.add_argument("--worse_weight", type=float, default=1.0)
    ap.add_argument("--better_weight", type=float, default=0.5)
    ap.add_argument("--baseline_preserve_weight", type=float, default=0.75)
    ap.add_argument("--hard_episode_indices", type=str, default="17")
    ap.add_argument("--hard_episode_weight", type=float, default=3.0)
    ap.add_argument("--candidate_score_std_min", type=float, default=0.5)
    ap.add_argument("--oracle_baseline_gap_min", type=float, default=1.0)
    ap.add_argument("--include_candidate_bank_missing", action="store_true", default=False)
    ap.add_argument("--mode_keep_failure_weight", type=float, default=2.5)
    ap.add_argument("--mode_apply_failure_weight", type=float, default=2.0)
    ap.add_argument("--large_yaw_negative_weight", type=float, default=2.0)
    ap.add_argument("--wrong_yaw_sign_weight", type=float, default=2.0)
    ap.add_argument("--xy_over_yaw_weight", type=float, default=2.5)
    ap.add_argument("--z_over_yaw_weight", type=float, default=2.5)
    ap.add_argument("--yaw_needed_but_not_selected_weight", type=float, default=1.8)
    args = ap.parse_args()

    data = _load_concat_npz(args.support_npz)
    n = int(next(iter(data.values())).shape[0])

    gate_open = np.asarray(data.get("b2_candidate_shadow_gate_open", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    changed = np.asarray(data.get("b2_candidate_shadow_changed", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    regret_delta = np.asarray(data.get("b2_candidate_shadow_regret_delta", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32)
    keep = gate_open & changed & np.isfinite(regret_delta)

    if not np.any(keep):
        raise RuntimeError("no shadow hard-negative rows survived filtering")

    idx = np.where(keep)[0]
    candidate_actions = np.asarray(data.get("b2_candidate_shadow_candidate_actions_local", data.get("candidate_actions_local")), dtype=np.float32)[idx]
    candidate_mask = np.asarray(data.get("b2_candidate_shadow_candidate_valid_mask", data.get("candidate_mask", np.ones(candidate_actions.shape[:2], dtype=np.float32))), dtype=np.float32)[idx]
    oracle_scores = np.asarray(data.get("b2_candidate_shadow_candidate_oracle_score", data.get("candidate_oracle_score")), dtype=np.float32)[idx]
    baseline_idx = np.asarray(data.get("runtime_selected_candidate_index", data.get("b2_candidate_shadow_baseline_index", data.get("candidate_baseline_index"))), dtype=np.int64)[idx]
    shadow_mode = np.asarray(data.get("b2_candidate_shadow_mode", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)[idx]
    shadow_pred_idx = np.asarray(data.get("b2_candidate_shadow_pred_index", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)[idx]
    pred_idx = np.asarray(data.get("pred_candidate_index", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)[idx]
    best_idx = np.asarray(data.get("b2_candidate_shadow_best_index", data.get("oracle_candidate_index", np.full((n,), -1, dtype=np.int64))), dtype=np.int64)[idx]
    episode_index = np.asarray(data.get("episode_index", np.full((n,), -1, dtype=np.int64)), dtype=np.int64)[idx]

    hard_episode_indices = _parse_csv_ints(args.hard_episode_indices)
    hard_episode_mask = np.isin(episode_index, np.asarray(sorted(hard_episode_indices), dtype=np.int64)) if hard_episode_indices else np.zeros_like(episode_index, dtype=bool)

    teacher_delta = np.asarray(
        data.get("teacher_current_delta_basin_target", data.get("proxy_current_delta_basin_target")),
        dtype=np.float32,
    )[idx]
    teacher_valid = np.all(np.isfinite(teacher_delta), axis=1)
    handoff_xy = np.asarray(data.get("runtime_handoff_release_threshold_xy_error", np.full((n,), 0.006, dtype=np.float32)), dtype=np.float32)[idx]
    handoff_z = np.asarray(data.get("runtime_handoff_release_threshold_abs_z_error", np.full((n,), 0.005, dtype=np.float32)), dtype=np.float32)[idx]
    handoff_yaw = np.asarray(data.get("runtime_handoff_release_threshold_yaw_error", np.full((n,), 0.12, dtype=np.float32)), dtype=np.float32)[idx]
    handoff_valid = np.isfinite(handoff_xy) & np.isfinite(handoff_z) & np.isfinite(handoff_yaw)

    valid_indices = (
        teacher_valid
        & handoff_valid
        & (baseline_idx >= 0)
        & (baseline_idx < candidate_actions.shape[1])
        & (pred_idx >= 0)
        & (pred_idx < candidate_actions.shape[1])
        & (best_idx >= 0)
        & (best_idx < candidate_actions.shape[1])
    )

    if not np.any(valid_indices):
        raise RuntimeError("no valid rows after index and teacher filtering")

    candidate_actions = candidate_actions[valid_indices]
    candidate_mask = candidate_mask[valid_indices]
    oracle_scores = oracle_scores[valid_indices]
    baseline_idx = baseline_idx[valid_indices]
    pred_idx = pred_idx[valid_indices]
    best_idx = best_idx[valid_indices]
    episode_index = episode_index[valid_indices]
    teacher_delta = teacher_delta[valid_indices]
    handoff_xy = handoff_xy[valid_indices]
    handoff_z = handoff_z[valid_indices]
    handoff_yaw = handoff_yaw[valid_indices]
    hard_episode_mask = hard_episode_mask[valid_indices]
    shadow_mode = shadow_mode[valid_indices]
    shadow_pred_idx = shadow_pred_idx[valid_indices]
    shadow_pred_regret = np.asarray(data.get("b2_candidate_shadow_pred_regret", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32)[idx][valid_indices]
    shadow_baseline_regret = np.asarray(data.get("b2_candidate_shadow_baseline_regret", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32)[idx][valid_indices]
    regret_delta = regret_delta[idx][valid_indices]

    sliced_inputs: dict[str, np.ndarray] = {}
    for key in INPUT_KEYS:
        if key in data:
            sliced_inputs[key] = np.asarray(data[key])[idx][valid_indices]
    front_rgb_arr = sliced_inputs.get("front_rgb")
    wrist_rgb_arr = sliced_inputs.get("wrist_rgb")
    wrist_depth_arr = sliced_inputs.get("wrist_depth")
    proprio_arr = sliced_inputs.get("proprio")
    gripper_context_arr = sliced_inputs.get("gripper_context")
    proxy_delta_arr = sliced_inputs.get("proxy_current_delta_basin_target")
    current_dx_arr = sliced_inputs.get("current_dx_sign")
    current_dy_arr = sliced_inputs.get("current_dy_sign")
    current_dyaw_arr = sliced_inputs.get("current_dyaw_sign")
    basin_bin_arr = sliced_inputs.get("basin_distance_bin")
    substage_arr = sliced_inputs.get("substage_id")
    contact_state_arr = sliced_inputs.get("contact_state")
    stage_target_mode_arr = sliced_inputs.get("stage_target_mode")
    episode_arr = sliced_inputs.get("episode_index")

    weights = np.stack(
        [
            1.0 / np.maximum(handoff_xy, 1e-4),
            1.0 / np.maximum(handoff_xy, 1e-4),
            1.0 / np.maximum(handoff_z, 1e-4),
            np.zeros_like(handoff_xy),
            np.zeros_like(handoff_xy),
            1.0 / np.maximum(handoff_yaw, 1e-4),
        ],
        axis=1,
    ).astype(np.float32)
    residual = teacher_delta[:, None, :6] - candidate_actions[:, :, :6]
    cost = np.linalg.norm(residual * weights[:, None, :], axis=2).astype(np.float32)
    oracle_scores = -cost
    best_idx = np.argmin(cost, axis=1).astype(np.int64)

    rows: list[dict[str, object]] = []
    row = np.arange(candidate_actions.shape[0], dtype=np.int64)
    pred_actions = candidate_actions[row, pred_idx]
    baseline_actions = candidate_actions[row, baseline_idx]
    oracle_actions = candidate_actions[row, best_idx]
    pred_yaw = np.abs(pred_actions[:, 5])
    oracle_yaw = np.abs(oracle_actions[:, 5])
    baseline_yaw = np.abs(baseline_actions[:, 5])
    pred_yaw_bucket = np.asarray([_yaw_bucket(float(v), float(args.keep_yaw_abs), float(args.small_yaw_abs), float(args.large_yaw_abs)) for v in pred_yaw], dtype=object)
    baseline_yaw_bucket = np.asarray([_yaw_bucket(float(v), float(args.keep_yaw_abs), float(args.small_yaw_abs), float(args.large_yaw_abs)) for v in baseline_yaw], dtype=object)
    oracle_yaw_bucket = np.asarray([_yaw_bucket(float(v), float(args.keep_yaw_abs), float(args.small_yaw_abs), float(args.large_yaw_abs)) for v in oracle_yaw], dtype=object)
    hardneg_type = np.asarray(["" for _ in range(candidate_actions.shape[0])], dtype=object)
    hardneg_type_weight = np.zeros((candidate_actions.shape[0],), dtype=np.float32)
    oracle_gap = np.asarray([float(oracle_scores[i, best_idx[i]] - oracle_scores[i, baseline_idx[i]]) for i in range(candidate_actions.shape[0])], dtype=np.float32)
    teacher_xy = np.asarray([float(np.linalg.norm(teacher_delta[i, :2])) for i in range(candidate_actions.shape[0])], dtype=np.float32)
    teacher_z = np.asarray([float(abs(teacher_delta[i, 2])) for i in range(candidate_actions.shape[0])], dtype=np.float32)
    teacher_yaw = np.asarray([float(abs(teacher_delta[i, 5])) for i in range(candidate_actions.shape[0])], dtype=np.float32)
    yaw_needed = teacher_yaw > np.asarray(handoff_yaw, dtype=np.float32)
    yaw_needed &= teacher_xy <= np.maximum(np.asarray(handoff_xy, dtype=np.float32) * 2.5, 0.030)

    report_type_counts: Counter[str] = Counter()
    kept = 0
    for i in range(candidate_actions.shape[0]):
        labels = _classify_row(
            pred_action=pred_actions[i],
            oracle_action=oracle_actions[i],
            baseline_action=baseline_actions[i],
            oracle_scores=oracle_scores[i],
            candidate_mask=candidate_mask[i],
            shadow_mode=int(shadow_mode[i]),
            pred_regret=float(shadow_pred_regret[i]),
            baseline_regret=float(shadow_baseline_regret[i]),
            keep_yaw_abs=float(args.keep_yaw_abs),
            small_yaw_abs=float(args.small_yaw_abs),
            large_yaw_abs=float(args.large_yaw_abs),
            candidate_score_std_min=float(args.candidate_score_std_min),
            oracle_baseline_gap_min=float(args.oracle_baseline_gap_min),
            mode_keep_margin=float(args.mode_keep_margin),
            yaw_needed=bool(yaw_needed[i]),
            include_candidate_bank_missing=bool(args.include_candidate_bank_missing),
        )
        mode_keep_failure = bool(
            int(shadow_mode[i]) == 0
            and math.isfinite(float(shadow_pred_regret[i]))
            and math.isfinite(float(shadow_baseline_regret[i]))
            and float(shadow_pred_regret[i]) > float(shadow_baseline_regret[i]) + 1e-6
        )
        mode_apply_failure = bool(
            int(shadow_mode[i]) == 1
            and math.isfinite(float(shadow_pred_regret[i]))
            and math.isfinite(float(shadow_baseline_regret[i]))
            and float(shadow_pred_regret[i]) > float(shadow_baseline_regret[i]) + 1e-6
        )
        if mode_keep_failure:
            labels = ["mode_keep_failure"] + [lab for lab in labels if lab != "candidate_bank_missing"]
        elif mode_apply_failure:
            labels = ["mode_apply_failure"] + [lab for lab in labels if lab != "candidate_bank_missing"]
        if not labels:
            labels = ["generic_negative"] if regret_delta[i] < -1e-6 else ["positive_preserve"]
        primary = _primary_label(labels)
        if primary == "candidate_bank_missing" and not bool(args.include_candidate_bank_missing):
            continue
        sample_weight = _type_weight(
            primary,
            worse_weight=float(args.worse_weight),
            better_weight=float(args.better_weight),
            baseline_preserve_weight=float(args.baseline_preserve_weight),
            mode_keep_failure_weight=float(args.mode_keep_failure_weight),
            mode_apply_failure_weight=float(args.mode_apply_failure_weight),
            large_yaw_negative_weight=float(args.large_yaw_negative_weight),
            wrong_yaw_sign_weight=float(args.wrong_yaw_sign_weight),
            xy_over_yaw_weight=float(args.xy_over_yaw_weight),
            z_over_yaw_weight=float(args.z_over_yaw_weight),
            yaw_needed_but_not_selected_weight=float(args.yaw_needed_but_not_selected_weight),
            hard_episode=bool(hard_episode_mask[i] and (regret_delta[i] < -1e-6)),
            hard_episode_weight=float(args.hard_episode_weight),
        )
        if regret_delta[i] > 1e-6:
            sample_weight = float(args.better_weight)
            primary = "positive_preserve"
        elif regret_delta[i] < -1e-6 and sample_weight <= 0.0:
            sample_weight = float(args.worse_weight)
        mode_target_override = -1
        if primary == "mode_keep_failure":
            mode_target_override = 1
        elif primary == "mode_apply_failure":
            mode_target_override = 0
        elif primary == "baseline_preserve":
            mode_target_override = 0
        elif primary in {"yaw_needed_missing", "yaw_not_needed_but_selected", "mode_apply_overuse"}:
            mode_target_override = 0 if float(oracle_gap[i]) <= float(args.mode_keep_margin) else 1
        bad_idx = int(shadow_pred_idx[i]) if 0 <= int(shadow_pred_idx[i]) < candidate_actions.shape[1] else int(pred_idx[i])
        if bad_idx < 0 or bad_idx >= candidate_actions.shape[1]:
            bad_idx = -1
        hardneg_type[i] = primary
        hardneg_type_weight[i] = float(sample_weight)
        report_type_counts[primary] += 1
        rows.append(
            {
                "front_rgb": front_rgb_arr[i],
                "wrist_rgb": wrist_rgb_arr[i],
                "wrist_depth": wrist_depth_arr[i],
                "proprio": proprio_arr[i],
                "gripper_context": gripper_context_arr[i],
                "proxy_current_delta_basin_target": proxy_delta_arr[i],
                "current_dx_sign": current_dx_arr[i],
                "current_dy_sign": current_dy_arr[i],
                "current_dyaw_sign": current_dyaw_arr[i],
                "basin_distance_bin": basin_bin_arr[i],
                "substage_id": substage_arr[i],
                "contact_state": contact_state_arr[i],
                "stage_target_mode": stage_target_mode_arr[i],
                "episode_index": int(episode_arr[i]),
                "candidate_actions_local": candidate_actions[i].astype(np.float32),
                "candidate_mask": candidate_mask[i].astype(np.float32),
                "candidate_oracle_score": oracle_scores[i].astype(np.float32),
                "candidate_best_index": int(best_idx[i]),
                "candidate_baseline_index": int(baseline_idx[i]),
                "candidate_bad_index": int(bad_idx),
                "sample_weight": float(sample_weight),
                "mode_target_override": int(mode_target_override),
                "shadow_mode": int(shadow_mode[i]),
                "shadow_pred_index": int(shadow_pred_idx[i]),
                "is_shadow_hard_negative": float(regret_delta[i] < -1e-6),
                "pred_worse_than_baseline": float(regret_delta[i] < -1e-6),
                "pred_better_than_baseline": float(regret_delta[i] > 1e-6),
                "pred_has_yaw": float(pred_yaw[i] > float(args.keep_yaw_abs)),
                "oracle_has_yaw": float(oracle_yaw[i] > float(args.keep_yaw_abs)),
                "baseline_has_yaw": float(baseline_yaw[i] > float(args.keep_yaw_abs)),
                "pred_large_yaw_negative": float("large_yaw_overuse" in labels or "small_vs_large_yaw" in labels),
                "pred_large_yaw_positive": float((regret_delta[i] > 1e-6) and (pred_yaw[i] >= float(args.large_yaw_abs))),
                "pred_small_yaw_positive": float((regret_delta[i] > 1e-6) and (pred_yaw[i] >= float(args.small_yaw_abs)) and (pred_yaw[i] < float(args.large_yaw_abs))),
                "pred_yaw_bucket": pred_yaw_bucket[i],
                "baseline_yaw_bucket": baseline_yaw_bucket[i],
                "oracle_yaw_bucket": oracle_yaw_bucket[i],
                "best_yaw_bucket": oracle_yaw_bucket[i],
                "hard_episode_negative": float(hard_episode_mask[i] and (regret_delta[i] < -1e-6)),
                "hardneg_type": primary,
                "hardneg_type_weight": float(sample_weight),
                "hardneg_mode_keep_failure": float(primary == "mode_keep_failure"),
                "hardneg_mode_apply_failure": float(primary == "mode_apply_failure"),
                "hardneg_large_yaw_overuse": float("large_yaw_overuse" in labels),
                "hardneg_wrong_yaw_sign": float("wrong_yaw_sign" in labels),
                "hardneg_small_vs_large_yaw": float("small_vs_large_yaw" in labels),
                "hardneg_xy_over_yaw": float("xy_over_yaw" in labels),
                "hardneg_z_over_yaw": float("z_over_yaw" in labels),
                "hardneg_yaw_needed_but_not_selected": float("yaw_needed_but_not_selected" in labels),
                "hardneg_candidate_bank_missing": float("candidate_bank_missing" in labels),
                "candidate_teacher_norm": np.stack(
                    [
                        float(np.asarray(data.get("teacher_truth_handoff_metric_xy_error", np.full((n,), np.nan)), dtype=np.float32)[idx][valid_indices][i])
                        / max(float(np.asarray(data.get("teacher_truth_handoff_release_threshold_xy_error", np.full((n,), 0.0085)), dtype=np.float32)[idx][valid_indices][i]), 1e-6),
                        float(np.asarray(data.get("teacher_truth_handoff_metric_abs_z_error", np.full((n,), np.nan)), dtype=np.float32)[idx][valid_indices][i])
                        / max(float(np.asarray(data.get("teacher_truth_handoff_release_threshold_abs_z_error", np.full((n,), 0.0035)), dtype=np.float32)[idx][valid_indices][i]), 1e-6),
                        float(np.asarray(data.get("teacher_truth_handoff_metric_yaw_error", np.full((n,), np.nan)), dtype=np.float32)[idx][valid_indices][i])
                        / max(float(np.asarray(data.get("teacher_truth_handoff_release_threshold_yaw_error", np.full((n,), 0.1243404)), dtype=np.float32)[idx][valid_indices][i]), 1e-6),
                    ],
                    axis=0,
                ).astype(np.float32),
            }
        )
        kept += 1

    if not rows:
        raise RuntimeError("no rows selected for typed supplement")

    out: dict[str, np.ndarray] = {}
    for key in INPUT_KEYS:
        if key in data:
            out[key] = np.asarray(data[key])[idx][valid_indices][:kept]
    out["candidate_actions_local"] = np.stack([r["candidate_actions_local"] for r in rows], axis=0).astype(np.float32)
    out["candidate_mask"] = np.stack([r["candidate_mask"] for r in rows], axis=0).astype(np.float32)
    out["candidate_oracle_score"] = np.stack([r["candidate_oracle_score"] for r in rows], axis=0).astype(np.float32)
    out["candidate_best_index"] = np.asarray([r["candidate_best_index"] for r in rows], dtype=np.int64)
    out["candidate_baseline_index"] = np.asarray([r["candidate_baseline_index"] for r in rows], dtype=np.int64)
    out["candidate_bad_index"] = np.asarray([r["candidate_bad_index"] for r in rows], dtype=np.int64)
    out["sample_weight"] = np.asarray([r["sample_weight"] for r in rows], dtype=np.float32)
    out["mode_target_override"] = np.asarray([r["mode_target_override"] for r in rows], dtype=np.int64)
    out["shadow_mode"] = np.asarray([r["shadow_mode"] for r in rows], dtype=np.int64)
    out["shadow_pred_index"] = np.asarray([r["shadow_pred_index"] for r in rows], dtype=np.int64)
    out["is_shadow_hard_negative"] = np.asarray([r["is_shadow_hard_negative"] for r in rows], dtype=np.float32)
    out["pred_worse_than_baseline"] = np.asarray([r["pred_worse_than_baseline"] for r in rows], dtype=np.float32)
    out["pred_better_than_baseline"] = np.asarray([r["pred_better_than_baseline"] for r in rows], dtype=np.float32)
    out["pred_has_yaw"] = np.asarray([r["pred_has_yaw"] for r in rows], dtype=np.float32)
    out["oracle_has_yaw"] = np.asarray([r["oracle_has_yaw"] for r in rows], dtype=np.float32)
    out["baseline_has_yaw"] = np.asarray([r["baseline_has_yaw"] for r in rows], dtype=np.float32)
    out["pred_large_yaw_negative"] = np.asarray([r["pred_large_yaw_negative"] for r in rows], dtype=np.float32)
    out["pred_large_yaw_positive"] = np.asarray([r["pred_large_yaw_positive"] for r in rows], dtype=np.float32)
    out["pred_small_yaw_positive"] = np.asarray([r["pred_small_yaw_positive"] for r in rows], dtype=np.float32)
    out["pred_yaw_bucket"] = np.asarray([{"no_yaw": 0, "small_yaw": 1, "medium_yaw": 2, "large_yaw": 3}[str(r["pred_yaw_bucket"])] for r in rows], dtype=np.int64)
    out["baseline_yaw_bucket"] = np.asarray([{"no_yaw": 0, "small_yaw": 1, "medium_yaw": 2, "large_yaw": 3}[str(r["baseline_yaw_bucket"])] for r in rows], dtype=np.int64)
    out["oracle_yaw_bucket"] = np.asarray([{"no_yaw": 0, "small_yaw": 1, "medium_yaw": 2, "large_yaw": 3}[str(r["oracle_yaw_bucket"])] for r in rows], dtype=np.int64)
    out["best_yaw_bucket"] = np.asarray([{"no_yaw": 0, "small_yaw": 1, "medium_yaw": 2, "large_yaw": 3}[str(r["best_yaw_bucket"])] for r in rows], dtype=np.int64)
    out["hard_episode_negative"] = np.asarray([r["hard_episode_negative"] for r in rows], dtype=np.float32)
    out["hardneg_type"] = np.asarray([r["hardneg_type"] for r in rows], dtype="<U40")
    out["hardneg_type_weight"] = np.asarray([r["hardneg_type_weight"] for r in rows], dtype=np.float32)
    out["candidate_advantage"] = np.asarray([float(oracle_gap[i]) for i in range(len(rows))], dtype=np.float32)
    out["teacher_xy_norm"] = teacher_xy.astype(np.float32)
    out["teacher_z_norm"] = teacher_z.astype(np.float32)
    out["teacher_yaw_norm"] = teacher_yaw.astype(np.float32)
    out["yaw_needed"] = yaw_needed.astype(np.float32)
    out["mode_keep_margin"] = np.full((len(rows),), float(args.mode_keep_margin), dtype=np.float32)
    out["hardneg_mode_keep_failure"] = np.asarray([r["hardneg_mode_keep_failure"] for r in rows], dtype=np.float32)
    out["hardneg_mode_apply_failure"] = np.asarray([r["hardneg_mode_apply_failure"] for r in rows], dtype=np.float32)
    out["hardneg_large_yaw_overuse"] = np.asarray([r["hardneg_large_yaw_overuse"] for r in rows], dtype=np.float32)
    out["hardneg_wrong_yaw_sign"] = np.asarray([r["hardneg_wrong_yaw_sign"] for r in rows], dtype=np.float32)
    out["hardneg_small_vs_large_yaw"] = np.asarray([r["hardneg_small_vs_large_yaw"] for r in rows], dtype=np.float32)
    out["hardneg_xy_over_yaw"] = np.asarray([r["hardneg_xy_over_yaw"] for r in rows], dtype=np.float32)
    out["hardneg_z_over_yaw"] = np.asarray([r["hardneg_z_over_yaw"] for r in rows], dtype=np.float32)
    out["hardneg_yaw_needed_but_not_selected"] = np.asarray([r["hardneg_yaw_needed_but_not_selected"] for r in rows], dtype=np.float32)
    out["hardneg_candidate_bank_missing"] = np.asarray([r["hardneg_candidate_bank_missing"] for r in rows], dtype=np.float32)
    out["candidate_teacher_norm"] = np.stack([r["candidate_teacher_norm"] for r in rows], axis=0).astype(np.float32)
    out["candidate_scope_size"] = np.sum(out["candidate_mask"] > 0.5, axis=1).astype(np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "alignment_v4b_typed_hard_negative_supplement.npz"
    np.savez_compressed(out_path, **out)

    report = {
        "rows": int(len(rows)),
        "worse_rows": int(np.sum(np.asarray([r["is_shadow_hard_negative"] for r in rows], dtype=np.float32) > 0.5)),
        "better_rows": int(np.sum(np.asarray([r["pred_better_than_baseline"] for r in rows], dtype=np.float32) > 0.5)),
        "hard_episode_rows": int(np.sum(np.asarray([r["hard_episode_negative"] for r in rows], dtype=np.float32) > 0.5)),
        "mean_regret_delta": float(np.mean([float(regret_delta[i]) for i in range(len(regret_delta))])) if len(regret_delta) else float("nan"),
        "type_counts": dict(report_type_counts),
        "output_npz": str(out_path),
    }
    (out_dir / "alignment_v4b_typed_hard_negative_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
