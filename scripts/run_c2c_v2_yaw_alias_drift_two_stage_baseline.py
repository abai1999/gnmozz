#!/usr/bin/env python3
"""Run a two-stage alias-vs-drift baseline.

Stage 1:
    classify a row as stable alias or frame drift.

Stage 2:
    regress jaw-local dyaw only on rows predicted as stable alias.

Frame drift rows are expected to be rejected or abstained from.
This is intentionally tiny and offline-only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_c2c_v2_yaw_alias_drift_baseline import (
    _corr,
    _fit_ridge,
    _jump_points,
    _mae,
    _predict,
    _read_jsonl,
    _row_features,
    _sign_match,
    _stack,
    _standardize,
    _safe_float,
)

FRAME_YAW_FEATURE_NAMES = (
    "planner_local_dx",
    "planner_local_dy",
    "planner_local_dz",
    "planner_local_droll",
    "planner_local_dpitch",
    "planner_local_dyaw",
    "proxy_dx",
    "proxy_dy",
    "proxy_dz",
    "proxy_residual_dyaw",
    "proxy_image_axis_yaw",
    "proxy_confidence",
    "proxy_observability",
    "proxy_fit_residual",
    "proxy_inlier_ratio",
    "proxy_valid",
    "proxy_yaw_valid",
    "estimated_dx",
    "estimated_dy",
    "estimated_dz",
    "estimated_dyaw",
    "estimated_x_valid",
    "estimated_y_valid",
    "estimated_z_valid",
    "estimated_yaw_valid",
    "estimated_confidence",
    "estimated_yaw_confidence",
    "frame_confidence",
    "frame_observability",
    "frame_axis_strength",
    "wide_ring_visible",
    "wrist_occluded",
    "visual_prior_only",
    "visual_partial_observable",
    "visual_visual_observable",
    "stage_ring_grasp_align",
    "stage_ring_spoke_align",
    "stage_slide_on_spoke",
    "skill_precision_grasp",
    "skill_precision_align",
    "skill_precision_slide",
    "requires_yaw_observability",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _threshold_sweep(prob: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if prob.size == 0:
        return {
            "best_threshold": 0.5,
            "best_balanced_accuracy": 0.0,
            "best_accuracy": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_specificity": 0.0,
            "best_f1": 0.0,
        }
    thresholds = np.linspace(0.0, 1.0, 201)
    target_bool = target > 0.5
    best: dict[str, Any] = {
        "best_threshold": 0.5,
        "best_balanced_accuracy": 0.0,
        "best_accuracy": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_specificity": 0.0,
        "best_f1": 0.0,
    }
    for thr in thresholds:
        pred = prob >= thr
        tp = int(np.count_nonzero(pred & target_bool))
        fp = int(np.count_nonzero(pred & ~target_bool))
        fn = int(np.count_nonzero(~pred & target_bool))
        tn = int(np.count_nonzero(~pred & ~target_bool))
        recall = float(tp / max(tp + fn, 1))
        precision = float(tp / max(tp + fp, 1))
        specificity = float(tn / max(tn + fp, 1))
        balanced = float(0.5 * (recall + specificity))
        accuracy = float(np.mean(pred == target_bool))
        f1 = float(2.0 * precision * recall / max(precision + recall, 1.0e-9))
        if balanced > best["best_balanced_accuracy"] or (balanced == best["best_balanced_accuracy"] and f1 > best["best_f1"]):
            best = {
                "best_threshold": float(thr),
                "best_balanced_accuracy": balanced,
                "best_accuracy": accuracy,
                "best_precision": precision,
                "best_recall": recall,
                "best_specificity": specificity,
                "best_f1": f1,
            }
    return best


def _calibrate_threshold(
    prob: np.ndarray,
    target: np.ndarray,
    *,
    min_specificity: float = 0.95,
    anchor_threshold: float | None = None,
) -> dict[str, Any]:
    """Pick a stable classifier threshold from a separate calibration set.

    Preference order:
    1. meet the requested specificity floor;
    2. among those, maximize recall;
    3. break ties with precision, then balanced accuracy, then threshold.

    If nothing satisfies the specificity floor, fall back to the best-balanced
    threshold so the caller still gets a deterministic operating point.
    """

    if prob.size == 0:
        return {
            "selected_threshold": float(anchor_threshold if anchor_threshold is not None else 0.5),
            "selection_policy": "empty_anchor" if anchor_threshold is not None else "empty",
            "target_specificity_floor": float(min_specificity),
            "best_threshold": float(anchor_threshold if anchor_threshold is not None else 0.5),
            "best_balanced_accuracy": 0.0,
            "best_accuracy": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_specificity": 0.0,
            "best_f1": 0.0,
        }

    thresholds = np.linspace(0.0, 1.0, 201)
    target_bool = target > 0.5
    eligible: list[dict[str, Any]] = []
    fallback: dict[str, Any] | None = None
    for thr in thresholds:
        pred = prob >= thr
        tp = int(np.count_nonzero(pred & target_bool))
        fp = int(np.count_nonzero(pred & ~target_bool))
        fn = int(np.count_nonzero(~pred & target_bool))
        tn = int(np.count_nonzero(~pred & ~target_bool))
        recall = float(tp / max(tp + fn, 1))
        precision = float(tp / max(tp + fp, 1))
        specificity = float(tn / max(tn + fp, 1))
        balanced = float(0.5 * (recall + specificity))
        accuracy = float(np.mean(pred == target_bool))
        f1 = float(2.0 * precision * recall / max(precision + recall, 1.0e-9))
        item = {
            "threshold": float(thr),
            "balanced_accuracy": balanced,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        if fallback is None or (
            item["balanced_accuracy"] > fallback["balanced_accuracy"]
            or (item["balanced_accuracy"] == fallback["balanced_accuracy"] and item["f1"] > fallback["f1"])
            or (
                item["balanced_accuracy"] == fallback["balanced_accuracy"]
                and item["f1"] == fallback["f1"]
                and item["threshold"] > fallback["threshold"]
            )
        ):
            fallback = item
        if specificity >= float(min_specificity):
            eligible.append(item)

    if eligible:
        eligible.sort(
            key=lambda item: (
                -float(item["recall"]),
                -float(item["precision"]),
                -float(item["balanced_accuracy"]),
                -float(item["specificity"]),
                -float(item["threshold"]),
            )
        )
        best = eligible[0]
        policy = "specificity_floor"
    else:
        best = fallback if fallback is not None else {
            "threshold": 0.5,
            "balanced_accuracy": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
        policy = "best_balanced_fallback"

    if anchor_threshold is not None and (
        float(best["recall"]) <= 0.0
        or float(best["balanced_accuracy"]) <= 0.5 + 1.0e-12
        or float(best["specificity"]) >= float(min_specificity) and float(best["recall"]) <= 0.0
    ):
        return {
            "selected_threshold": float(anchor_threshold),
            "selection_policy": "anchor_fallback_degenerate_calibration",
            "target_specificity_floor": float(min_specificity),
            "best_threshold": float(best["threshold"]),
            "best_balanced_accuracy": float(best["balanced_accuracy"]),
            "best_accuracy": float(best["accuracy"]),
            "best_precision": float(best["precision"]),
            "best_recall": float(best["recall"]),
            "best_specificity": float(best["specificity"]),
            "best_f1": float(best["f1"]),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "tn": int(best["tn"]),
            "candidate_count": int(len(thresholds)),
            "eligible_count": int(len(eligible)),
            "anchor_threshold": float(anchor_threshold),
        }

    return {
        "selected_threshold": float(best["threshold"]),
        "selection_policy": str(policy),
        "target_specificity_floor": float(min_specificity),
        "best_threshold": float(best["threshold"]),
        "best_balanced_accuracy": float(best["balanced_accuracy"]),
        "best_accuracy": float(best["accuracy"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_specificity": float(best["specificity"]),
        "best_f1": float(best["f1"]),
        "tp": int(best["tp"]),
        "fp": int(best["fp"]),
        "fn": int(best["fn"]),
        "tn": int(best["tn"]),
        "candidate_count": int(len(thresholds)),
        "eligible_count": int(len(eligible)),
    }


def _fit_logistic_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 1.0e-2,
    steps: int = 400,
    lr: float = 0.15,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.zeros((x_aug.shape[1],), dtype=np.float64)
    if sample_weight is None:
        weight = np.ones_like(target, dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    weight = weight / max(float(np.mean(weight)), 1.0e-9)
    for _ in range(int(steps)):
        logits = x_aug @ w
        prob = _sigmoid(logits)
        err = (prob - target) * weight
        grad = (x_aug.T @ err) / max(float(x_aug.shape[0]), 1.0)
        grad[1:] += float(ridge) * w[1:]
        w -= float(lr) * grad
    return w


def _predict_logistic(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    if x.size == 0 or w.size == 0:
        return np.zeros((x.shape[0],), dtype=np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return _sigmoid(x_aug @ w)


def _support_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: (str(r.get("acceptance_role", "")), int(r.get("episode_idx", -1))))
    return rows


def _episode_split(episodes: list[int], *, holdout_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    eps = sorted(set(int(ep) for ep in episodes))
    if not eps:
        return [], []
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(eps, dtype=np.int64)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return [int(shuffled[0])], []
    holdout_count = max(1, int(round(float(holdout_ratio) * len(shuffled))))
    holdout_count = min(holdout_count, len(shuffled) - 1)
    holdout = [int(v) for v in shuffled[:holdout_count]]
    train = [int(v) for v in shuffled[holdout_count:]]
    return train, holdout


def _select_rows_from_support(rows: list[dict[str, Any]], support_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    ep = int(support_row.get("episode_idx", -1))
    stage = str(support_row.get("stage_name", "RING_GRASP_ALIGN"))
    skill = str(support_row.get("skill_type", "precision_grasp"))
    bucket = str(support_row.get("failure_bucket", ""))
    step_idxs = {int(v) for v in support_row.get("selected_step_idxs", []) if isinstance(v, (int, float, np.integer, np.floating))}
    selected = [
        row
        for row in rows
        if int(row.get("episode_idx", -1)) == ep
        and str(row.get("stage_name", "")) == stage
        and (
            str(row.get("skill_type", row.get("skill_name", ""))) == skill
            or str(row.get("skill_type", row.get("skill_name", ""))).startswith(skill)
            or skill.startswith(str(row.get("skill_type", row.get("skill_name", ""))))
        )
        and (not bucket or str(row.get("failure_bucket", "")) == bucket)
        and (not step_idxs or int(row.get("step_idx", row.get("step", -1))) in step_idxs)
    ]
    selected.sort(key=lambda r: int(r.get("step_idx", r.get("step", -1))))
    return selected


def _true_dyaw(row: Mapping[str, Any]) -> float:
    residual = row.get("true_basin_error_t") if isinstance(row.get("true_basin_error_t"), Mapping) else {}
    if "dyaw" in residual:
        return _safe_float(residual.get("dyaw"), float("nan"))
    return _safe_float(row.get("privileged_dyaw", row.get("privileged_yaw", float("nan"))), float("nan"))


def run_two_stage_baseline(
    *,
    relabel_jsonl: Path,
    support_manifest_jsonl: Path,
    output_dir: Path,
    holdout_ratio: float = 0.25,
    calibration_ratio: float = 0.25,
    seed: int = 7,
    ridge: float = 1.0e-2,
    classifier_ridge: float = 1.0e-2,
    calibration_min_specificity: float = 0.95,
) -> dict[str, Any]:
    rows = _read_jsonl(relabel_jsonl)
    support_rows = _support_rows(support_manifest_jsonl)
    grouped: dict[str, list[dict[str, Any]]] = {"calibration_positive": [], "frame_drift_hard_case": [], "mixed_or_unclear": []}
    for row in support_rows:
        grouped.setdefault(str(row.get("acceptance_role", "mixed_or_unclear")), []).append(row)

    pos_train_eps, pos_holdout_eps = _episode_split(
        [int(r["episode_idx"]) for r in grouped["calibration_positive"]],
        holdout_ratio=float(holdout_ratio),
        seed=int(seed),
    )
    neg_train_eps, neg_holdout_eps = _episode_split(
        [int(r["episode_idx"]) for r in grouped["frame_drift_hard_case"]],
        holdout_ratio=float(holdout_ratio),
        seed=int(seed) + 13,
    )

    pos_fit_eps, pos_calib_eps = _episode_split(
        [int(r["episode_idx"]) for r in grouped["calibration_positive"] if int(r["episode_idx"]) in pos_train_eps],
        holdout_ratio=float(calibration_ratio),
        seed=int(seed) + 101,
    )
    neg_fit_eps, neg_calib_eps = _episode_split(
        [int(r["episode_idx"]) for r in grouped["frame_drift_hard_case"] if int(r["episode_idx"]) in neg_train_eps],
        holdout_ratio=float(calibration_ratio),
        seed=int(seed) + 131,
    )

    train_support = [
        row
        for row in support_rows
        if (
            (str(row.get("acceptance_role", "")) == "calibration_positive" and int(row.get("episode_idx", -1)) in pos_train_eps)
            or (str(row.get("acceptance_role", "")) == "frame_drift_hard_case" and int(row.get("episode_idx", -1)) in neg_train_eps)
        )
    ]
    fit_support = [
        row
        for row in train_support
        if (
            (str(row.get("acceptance_role", "")) == "calibration_positive" and int(row.get("episode_idx", -1)) in pos_fit_eps)
            or (str(row.get("acceptance_role", "")) == "frame_drift_hard_case" and int(row.get("episode_idx", -1)) in neg_fit_eps)
        )
    ]
    calib_support = [
        row
        for row in train_support
        if (
            (str(row.get("acceptance_role", "")) == "calibration_positive" and int(row.get("episode_idx", -1)) in pos_calib_eps)
            or (str(row.get("acceptance_role", "")) == "frame_drift_hard_case" and int(row.get("episode_idx", -1)) in neg_calib_eps)
        )
    ]
    holdout_support = [
        row
        for row in support_rows
        if (
            (str(row.get("acceptance_role", "")) == "calibration_positive" and int(row.get("episode_idx", -1)) in pos_holdout_eps)
            or (str(row.get("acceptance_role", "")) == "frame_drift_hard_case" and int(row.get("episode_idx", -1)) in neg_holdout_eps)
        )
    ]

    fit_selected: list[dict[str, Any]] = []
    calib_selected: list[dict[str, Any]] = []
    holdout_selected: list[dict[str, Any]] = []
    fit_slices: list[dict[str, Any]] = []
    calib_slices: list[dict[str, Any]] = []
    holdout_slices: list[dict[str, Any]] = []
    fit_role_by_row: list[float] = []
    calib_role_by_row: list[float] = []
    holdout_role_by_row: list[float] = []

    for support_row in fit_support:
        label = 1.0 if str(support_row.get("acceptance_role", "")) == "calibration_positive" else 0.0
        selected = _select_rows_from_support(rows, support_row)
        fit_selected.extend(selected)
        fit_role_by_row.extend([label] * len(selected))
        fit_slices.append(
            {
                "episode_idx": int(support_row.get("episode_idx", -1)),
                "acceptance_role": str(support_row.get("acceptance_role", "")),
                "alias_label": str(support_row.get("alias_label", "")),
                "failure_bucket": str(support_row.get("failure_bucket", "")),
                "rows": int(len(selected)),
                "selected_step_count": int(len(support_row.get("selected_step_idxs", []))),
                "source_kind": str(support_row.get("source_kind", "")),
                "source_path": str(support_row.get("source_path", "")),
            }
        )

    for support_row in calib_support:
        label = 1.0 if str(support_row.get("acceptance_role", "")) == "calibration_positive" else 0.0
        selected = _select_rows_from_support(rows, support_row)
        calib_selected.extend(selected)
        calib_role_by_row.extend([label] * len(selected))
        calib_slices.append(
            {
                "episode_idx": int(support_row.get("episode_idx", -1)),
                "acceptance_role": str(support_row.get("acceptance_role", "")),
                "alias_label": str(support_row.get("alias_label", "")),
                "failure_bucket": str(support_row.get("failure_bucket", "")),
                "rows": int(len(selected)),
                "selected_step_count": int(len(support_row.get("selected_step_idxs", []))),
                "source_kind": str(support_row.get("source_kind", "")),
                "source_path": str(support_row.get("source_path", "")),
            }
        )

    for support_row in holdout_support:
        label = 1.0 if str(support_row.get("acceptance_role", "")) == "calibration_positive" else 0.0
        selected = _select_rows_from_support(rows, support_row)
        holdout_selected.extend(selected)
        holdout_role_by_row.extend([label] * len(selected))
        holdout_slices.append(
            {
                "episode_idx": int(support_row.get("episode_idx", -1)),
                "acceptance_role": str(support_row.get("acceptance_role", "")),
                "alias_label": str(support_row.get("alias_label", "")),
                "failure_bucket": str(support_row.get("failure_bucket", "")),
                "rows": int(len(selected)),
                "selected_step_count": int(len(support_row.get("selected_step_idxs", []))),
                "source_kind": str(support_row.get("source_kind", "")),
                "source_path": str(support_row.get("source_path", "")),
            }
        )

    def _stack_labeled_rows(
        selected_rows: list[dict[str, Any]],
        labels: list[float],
    ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], np.ndarray]:
        kept_rows: list[dict[str, Any]] = []
        features: list[np.ndarray] = []
        targets: list[float] = []
        symm_targets: list[float] = []
        meta: list[dict[str, float]] = []
        kept_labels: list[float] = []
        for row, label in zip(selected_rows, labels):
            feat_vec, info = _row_features(row)
            dyaw = _true_dyaw(row)
            if not np.isfinite(dyaw):
                continue
            kept_rows.append(row)
            features.append(feat_vec)
            targets.append(float(dyaw))
            symm_targets.append(float(info["symmetry_aware_proxy_yaw"]))
            meta.append(info)
            kept_labels.append(float(label))
        if features:
            return (
                kept_rows,
                np.stack(features).astype(np.float64),
                np.asarray(targets, dtype=np.float64),
                np.asarray(symm_targets, dtype=np.float64),
                meta,
                np.asarray(kept_labels, dtype=np.float64),
            )
        return kept_rows, np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), meta, np.zeros((0,), dtype=np.float64)

    fit_selected, x_fit_raw, y_fit_dyaw, symm_fit_raw, fit_meta, fit_role_arr = _stack_labeled_rows(fit_selected, fit_role_by_row)
    calib_selected, x_calib_raw, y_calib_dyaw, symm_calib_raw, calib_meta, calib_role_arr = _stack_labeled_rows(calib_selected, calib_role_by_row)
    holdout_selected, x_hold_raw, y_hold_dyaw, symm_hold_raw, hold_meta, hold_role_arr = _stack_labeled_rows(holdout_selected, holdout_role_by_row)
    if x_fit_raw.size == 0:
        raise RuntimeError("No training rows selected for alias/drift two-stage baseline")
    if x_fit_raw.shape[1] == 0:
        raise RuntimeError("Empty feature matrix for alias/drift two-stage baseline")

    x_train_std, x_hold_std, stats = _standardize(
        x_fit_raw,
        x_hold_raw if x_hold_raw.size else np.zeros((0, x_fit_raw.shape[1]), dtype=np.float64),
    )

    class_weight = np.ones((fit_role_arr.shape[0],), dtype=np.float64)
    pos_count = float(np.count_nonzero(fit_role_arr > 0.5))
    neg_count = float(max(fit_role_arr.size - np.count_nonzero(fit_role_arr > 0.5), 1))
    if pos_count > 0:
        class_weight[fit_role_arr > 0.5] = min(neg_count / max(pos_count, 1.0), 20.0)
    clf_w = _fit_logistic_ridge(
        x_train_std,
        fit_role_arr,
        ridge=float(classifier_ridge),
        steps=500,
        lr=0.18,
        sample_weight=class_weight,
    )
    fit_prob = _predict_logistic(x_train_std, clf_w)
    _, calib_std, _ = _standardize(
        x_fit_raw,
        x_calib_raw if x_calib_raw.size else np.zeros((0, x_fit_raw.shape[1]), dtype=np.float64),
    )
    calib_prob = _predict_logistic(calib_std, clf_w) if x_calib_raw.size else np.zeros((0,), dtype=np.float64)
    hold_prob = _predict_logistic(x_hold_std, clf_w) if x_hold_std.size else np.zeros((0,), dtype=np.float64)
    threshold_sweep_fit = _threshold_sweep(fit_prob, fit_role_arr)
    threshold_calibration = _calibrate_threshold(
        calib_prob,
        calib_role_arr,
        min_specificity=float(calibration_min_specificity),
        anchor_threshold=float(threshold_sweep_fit["best_threshold"]),
    )
    threshold = float(threshold_calibration["selected_threshold"])

    fit_pred = fit_prob >= threshold
    calib_pred = calib_prob >= threshold
    hold_pred = hold_prob >= threshold

    def _class_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        target_bool = target > 0.5
        tp = int(np.count_nonzero(pred & target_bool))
        fp = int(np.count_nonzero(pred & ~target_bool))
        fn = int(np.count_nonzero(~pred & target_bool))
        tn = int(np.count_nonzero(~pred & ~target_bool))
        recall = float(tp / max(tp + fn, 1))
        precision = float(tp / max(tp + fp, 1))
        specificity = float(tn / max(tn + fp, 1))
        balanced = float(0.5 * (recall + specificity))
        return {
            "rows": int(target_bool.size),
            "accuracy": float(np.mean(pred == target_bool)) if target_bool.size else 0.0,
            "balanced_accuracy": balanced,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": float(2.0 * precision * recall / max(precision + recall, 1.0e-9)),
            "predicted_positive_rate": float(np.mean(pred)) if pred.size else 0.0,
            "target_positive_rate": float(np.mean(target_bool)) if target_bool.size else 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    classifier_fit = _class_metrics(fit_pred, fit_role_arr)
    classifier_calibration = _class_metrics(calib_pred, calib_role_arr)
    classifier_holdout = _class_metrics(hold_pred, hold_role_arr) if hold_role_arr.size else {"rows": 0}

    positive_fit_rows = [row for row, role in zip(fit_selected, fit_role_arr) if role > 0.5]
    positive_calib_rows = [row for row, role in zip(calib_selected, calib_role_arr) if role > 0.5]
    positive_hold_rows = [row for row, role in zip(holdout_selected, hold_role_arr) if role > 0.5]
    drift_hold_rows = [row for row, role in zip(holdout_selected, hold_role_arr) if role <= 0.5]

    x_pos_train_raw, y_pos_train, symm_pos_train, pos_train_meta = _stack(positive_fit_rows)
    x_pos_calib_raw, y_pos_calib, symm_pos_calib, pos_calib_meta = _stack(positive_calib_rows)
    x_pos_hold_raw, y_pos_hold, symm_pos_hold, pos_hold_meta = _stack(positive_hold_rows)
    x_drift_hold_raw, y_drift_hold, symm_drift_hold, drift_hold_meta = _stack(drift_hold_rows)
    if x_pos_train_raw.size == 0:
        raise RuntimeError("No positive calibration rows selected for regression stage")

    x_pos_train, x_pos_hold, stats_pos = _standardize(
        x_pos_train_raw,
        x_pos_hold_raw if x_pos_hold_raw.size else np.zeros((0, x_pos_train_raw.shape[1]), dtype=np.float64),
    )
    reg_w = _fit_ridge(x_pos_train, y_pos_train, ridge=float(ridge))
    pred_pos_train = _predict(x_pos_train, reg_w)
    pred_pos_hold = _predict(x_pos_hold, reg_w) if x_pos_hold.size else np.zeros((0,), dtype=np.float64)
    pred_drift_hold = _predict(x_drift_hold_raw, reg_w) if x_drift_hold_raw.size else np.zeros((0,), dtype=np.float64)

    pos_train_report = {
        "rows": int(y_pos_train.shape[0]),
        "raw_proxy_mae": _mae(np.asarray([m["raw_proxy_yaw"] for m in pos_train_meta], dtype=np.float64), y_pos_train),
        "symmetry_aware_mae": _mae(symm_pos_train, y_pos_train),
        "symmetry_aware_bias": float(np.mean(symm_pos_train - y_pos_train)) if y_pos_train.size else 0.0,
        "symmetry_aware_bias_corrected_mae": _mae(symm_pos_train - float(np.mean(symm_pos_train - y_pos_train)), y_pos_train) if y_pos_train.size else 0.0,
        "learned_mae": _mae(pred_pos_train, y_pos_train),
        "learned_corr": _corr(pred_pos_train, y_pos_train),
        "learned_sign_match": _sign_match(pred_pos_train, y_pos_train),
        "learned_jump_count": int(len(_jump_points(pred_pos_train))),
    }
    pred_pos_calib = _predict(x_pos_calib_raw, reg_w) if x_pos_calib_raw.size else np.zeros((0,), dtype=np.float64)
    pos_calib_report = {
        "rows": int(y_pos_calib.shape[0]),
        "raw_proxy_mae": _mae(np.asarray([m["raw_proxy_yaw"] for m in pos_calib_meta], dtype=np.float64), y_pos_calib) if y_pos_calib.size else 0.0,
        "symmetry_aware_mae": _mae(symm_pos_calib, y_pos_calib) if y_pos_calib.size else 0.0,
        "symmetry_aware_bias": float(np.mean(symm_pos_train - y_pos_train)) if y_pos_train.size else 0.0,
        "symmetry_aware_bias_corrected_mae": _mae(
            symm_pos_calib - float(np.mean(symm_pos_train - y_pos_train)) if y_pos_calib.size else symm_pos_calib,
            y_pos_calib,
        ) if y_pos_calib.size else 0.0,
        "learned_mae": _mae(pred_pos_calib, y_pos_calib) if y_pos_calib.size else 0.0,
        "learned_corr": _corr(pred_pos_calib, y_pos_calib) if y_pos_calib.size else 0.0,
        "learned_sign_match": _sign_match(pred_pos_calib, y_pos_calib) if y_pos_calib.size else 0.0,
        "learned_jump_count": int(len(_jump_points(pred_pos_calib))),
    }
    pos_hold_report = {
        "rows": int(y_pos_hold.shape[0]),
        "raw_proxy_mae": _mae(np.asarray([m["raw_proxy_yaw"] for m in pos_hold_meta], dtype=np.float64), y_pos_hold) if y_pos_hold.size else 0.0,
        "symmetry_aware_mae": _mae(symm_pos_hold, y_pos_hold) if y_pos_hold.size else 0.0,
        "symmetry_aware_bias": float(np.mean(symm_pos_train - y_pos_train)) if y_pos_train.size else 0.0,
        "symmetry_aware_bias_corrected_mae": _mae(
            symm_pos_hold - float(np.mean(symm_pos_train - y_pos_train)) if y_pos_hold.size else symm_pos_hold,
            y_pos_hold,
        ) if y_pos_hold.size else 0.0,
        "learned_mae": _mae(pred_pos_hold, y_pos_hold) if y_pos_hold.size else 0.0,
        "learned_corr": _corr(pred_pos_hold, y_pos_hold) if y_pos_hold.size else 0.0,
        "learned_sign_match": _sign_match(pred_pos_hold, y_pos_hold) if y_pos_hold.size else 0.0,
        "learned_jump_count": int(len(_jump_points(pred_pos_hold))),
    }

    drift_hold_report = {
        "rows": int(y_drift_hold.shape[0]),
        "raw_proxy_mae": _mae(np.asarray([m["raw_proxy_yaw"] for m in drift_hold_meta], dtype=np.float64), y_drift_hold) if y_drift_hold.size else 0.0,
        "symmetry_aware_mae": _mae(symm_drift_hold, y_drift_hold) if y_drift_hold.size else 0.0,
        "regression_mae": _mae(pred_drift_hold, y_drift_hold) if y_drift_hold.size else 0.0,
        "regression_corr": _corr(pred_drift_hold, y_drift_hold) if y_drift_hold.size else 0.0,
        "regression_sign_match": _sign_match(pred_drift_hold, y_drift_hold) if y_drift_hold.size else 0.0,
        "regression_jump_count": int(len(_jump_points(pred_drift_hold))),
    }

    row_jsonl = output_dir / "yaw_alias_drift_two_stage_baseline_rows.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(row_jsonl, "w", encoding="utf-8") as handle:
        for split_name, selected_rows, feat_rows, target, prob, label in (
            ("fit", fit_selected, x_fit_raw, fit_role_arr, fit_prob, fit_pred),
            ("calibration", calib_selected, x_calib_raw, calib_role_arr, calib_prob, calib_pred),
            ("holdout", holdout_selected, x_hold_raw, hold_role_arr, hold_prob, hold_pred),
        ):
            for row, feat_vec, tgt, p, pred in zip(selected_rows, feat_rows, target, prob, label):
                _, meta = _row_features(row)
                handle.write(
                    json.dumps(
                        {
                            "split": split_name,
                            "episode_idx": int(row.get("episode_idx", -1)),
                            "step_idx": int(row.get("step_idx", row.get("step", -1))),
                            "failure_bucket": str(row.get("failure_bucket", "")),
                            "acceptance_role": "calibration_positive" if float(tgt) > 0.5 else "frame_drift_hard_case",
                            "raw_proxy_yaw": float(meta["raw_proxy_yaw"]),
                            "symmetry_aware_proxy_yaw": float(meta["symmetry_aware_proxy_yaw"]),
                            "true_dyaw": float(_true_dyaw(row)),
                            "predicted_stable_alias_probability": float(p),
                            "predicted_stable_alias": bool(pred),
                            "accepted_by_stage1": bool(pred),
                            "regressed_dyaw": float(_predict(np.asarray(feat_vec, dtype=np.float64).reshape(1, -1), reg_w)[0]) if bool(pred) else float("nan"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    classifier_positive_accept_rate = float(np.mean(hold_pred)) if hold_pred.size else 0.0
    drift_false_accept_rate = float(np.mean(hold_pred[hold_role_arr <= 0.5])) if np.any(hold_role_arr <= 0.5) else 0.0
    drift_abstain_rate = float(np.mean(~hold_pred[hold_role_arr <= 0.5])) if np.any(hold_role_arr <= 0.5) else 1.0
    positive_accept_rate = float(np.mean(hold_pred[hold_role_arr > 0.5])) if np.any(hold_role_arr > 0.5) else 0.0
    end_to_end_positive_mae = _mae(pred_pos_hold[hold_pred[hold_role_arr > 0.5]], y_pos_hold[hold_pred[hold_role_arr > 0.5]]) if np.any(hold_role_arr > 0.5) and np.any(hold_pred[hold_role_arr > 0.5]) else 0.0
    end_to_end_abstain_correct_rate = drift_abstain_rate

    report = {
        "schema_version": "yaw_alias_drift_two_stage_baseline_v1",
        "relabel_jsonl": str(relabel_jsonl.resolve()),
        "support_manifest_jsonl": str(support_manifest_jsonl.resolve()),
        "output_dir": str(output_dir.resolve()),
        "support_summary": {
            "fit_support_rows": int(len(fit_support)),
            "calibration_support_rows": int(len(calib_support)),
            "train_support_rows": int(len(train_support)),
            "holdout_support_rows": int(len(holdout_support)),
            "fit_positive_episodes": int(len(pos_fit_eps)),
            "calibration_positive_episodes": int(len(pos_calib_eps)),
            "fit_frame_drift_episodes": int(len(neg_fit_eps)),
            "calibration_frame_drift_episodes": int(len(neg_calib_eps)),
            "train_positive_episodes": int(len(pos_train_eps)),
            "holdout_positive_episodes": int(len(pos_holdout_eps)),
            "train_frame_drift_episodes": int(len(neg_train_eps)),
            "holdout_frame_drift_episodes": int(len(neg_holdout_eps)),
            "fit_rows": int(len(fit_selected)),
            "calibration_rows": int(len(calib_selected)),
            "train_rows": int(len(fit_selected) + len(calib_selected)),
            "holdout_rows": int(len(holdout_selected)),
            "fit_positive_rows": int(len(positive_fit_rows)),
            "calibration_positive_rows": int(len(positive_calib_rows)),
            "train_positive_rows": int(len(positive_fit_rows) + len(positive_calib_rows)),
            "holdout_positive_rows": int(len(positive_hold_rows)),
            "holdout_frame_drift_rows": int(len(drift_hold_rows)),
        },
        "classifier": {
            "threshold": float(threshold),
            "threshold_source": str(
                "calibration_specificity_floor"
                if threshold_calibration["selection_policy"] == "specificity_floor"
                else "calibration_anchor_fallback"
                if threshold_calibration["selection_policy"] == "anchor_fallback_degenerate_calibration"
                else "calibration_best_balanced_fallback"
            ),
            "threshold_sweep_fit": threshold_sweep_fit,
            "threshold_calibration": threshold_calibration,
            "fit": classifier_fit,
            "calibration": classifier_calibration,
            "holdout": classifier_holdout,
            "holdout_positive_accept_rate": float(positive_accept_rate),
            "holdout_drift_false_accept_rate": float(drift_false_accept_rate),
            "holdout_drift_abstain_rate": float(drift_abstain_rate),
        },
        "regression": {
            "train": pos_train_report,
            "calibration": pos_calib_report,
            "holdout": pos_hold_report,
            "drift_holdout": drift_hold_report,
            "feature_names": list(FRAME_YAW_FEATURE_NAMES),
            "ridge": float(ridge),
            "classifier_ridge": float(classifier_ridge),
            "standardization": {
                "feature_mean": stats[0].tolist() if stats.size else [],
                "feature_std": stats[1].tolist() if stats.size else [],
                "positive_feature_mean": stats_pos[0].tolist() if stats_pos.size else [],
                "positive_feature_std": stats_pos[1].tolist() if stats_pos.size else [],
            },
        },
        "end_to_end": {
            "positive_accept_rate": float(positive_accept_rate),
            "drift_false_accept_rate": float(drift_false_accept_rate),
            "drift_abstain_rate": float(drift_abstain_rate),
            "abstain_correct_rate": float(end_to_end_abstain_correct_rate),
            "positive_mae": float(end_to_end_positive_mae),
        },
        "fit_slices": fit_slices,
        "calibration_slices": calib_slices,
        "holdout_slices": holdout_slices,
        "rows_jsonl": str(row_jsonl.resolve()),
    }

    out_json = output_dir / "yaw_alias_drift_two_stage_baseline_report.json"
    out_md = output_dir / "yaw_alias_drift_two_stage_baseline_report.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Yaw Alias Drift Two-Stage Baseline",
        "",
        f"- support rows: `{report['support_summary']['fit_support_rows'] + report['support_summary']['calibration_support_rows'] + report['support_summary']['holdout_support_rows']}`",
        f"- fit support rows: `{report['support_summary']['fit_support_rows']}`",
        f"- calibration support rows: `{report['support_summary']['calibration_support_rows']}`",
        f"- train support rows: `{report['support_summary']['train_support_rows']}`",
        f"- holdout support rows: `{report['support_summary']['holdout_support_rows']}`",
        f"- fit positive rows: `{report['support_summary']['fit_positive_rows']}`",
        f"- calibration positive rows: `{report['support_summary']['calibration_positive_rows']}`",
        f"- train positive rows: `{report['support_summary']['train_positive_rows']}`",
        f"- holdout positive rows: `{report['support_summary']['holdout_positive_rows']}`",
        f"- holdout frame drift rows: `{report['support_summary']['holdout_frame_drift_rows']}`",
        "",
        "## Classifier",
        f"- threshold: `{report['classifier']['threshold']:.3f}`",
        f"- threshold source: `{report['classifier']['threshold_source']}`",
        f"- fit balanced accuracy: `{report['classifier']['fit']['balanced_accuracy']:.3f}`",
        f"- calibration balanced accuracy: `{report['classifier']['calibration']['balanced_accuracy']:.3f}`" if report["classifier"]["calibration"] else "- calibration balanced accuracy: `0.000`",
        f"- holdout balanced accuracy: `{report['classifier']['holdout']['balanced_accuracy']:.3f}`" if report["classifier"]["holdout"] else "- holdout balanced accuracy: `0.000`",
        f"- holdout positive accept rate: `{report['classifier']['holdout_positive_accept_rate']:.3f}`",
        f"- holdout drift false accept rate: `{report['classifier']['holdout_drift_false_accept_rate']:.3f}`",
        f"- holdout drift abstain rate: `{report['classifier']['holdout_drift_abstain_rate']:.3f}`",
        "",
        "## Regression",
        f"- positive train MAE: `{report['regression']['train']['learned_mae']:.6f}`",
        f"- positive calibration MAE: `{report['regression']['calibration']['learned_mae']:.6f}`",
        f"- positive holdout MAE: `{report['regression']['holdout']['learned_mae']:.6f}`",
        f"- drift holdout regression MAE: `{report['regression']['drift_holdout']['regression_mae']:.6f}`",
        f"- end-to-end positive MAE: `{report['end_to_end']['positive_mae']:.6f}`",
        f"- end-to-end drift abstain rate: `{report['end_to_end']['drift_abstain_rate']:.3f}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a two-stage alias-vs-drift baseline.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument("--support_manifest_jsonl", type=Path, required=True)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/yaw_alias_drift_two_stage_baseline"),
    )
    ap.add_argument("--holdout_ratio", type=float, default=0.25)
    ap.add_argument("--calibration_ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ridge", type=float, default=1.0e-2)
    ap.add_argument("--classifier_ridge", type=float, default=1.0e-2)
    ap.add_argument("--calibration_min_specificity", type=float, default=0.95)
    args = ap.parse_args()

    report = run_two_stage_baseline(
        relabel_jsonl=args.relabel_jsonl,
        support_manifest_jsonl=args.support_manifest_jsonl,
        output_dir=args.output_dir.resolve(),
        holdout_ratio=float(args.holdout_ratio),
        calibration_ratio=float(args.calibration_ratio),
        seed=int(args.seed),
        ridge=float(args.ridge),
        classifier_ridge=float(args.classifier_ridge),
        calibration_min_specificity=float(args.calibration_min_specificity),
    )
    print(json.dumps(
        {
            "classifier": report["classifier"],
            "regression": {
                "train": report["regression"]["train"],
                "holdout": report["regression"]["holdout"],
                "drift_holdout": report["regression"]["drift_holdout"],
            },
            "end_to_end": report["end_to_end"],
            "rows_jsonl": report["rows_jsonl"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
