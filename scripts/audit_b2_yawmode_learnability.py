#!/usr/bin/env python3
"""Audit whether B2 yaw-mode labels are learnable from runtime-visible features.

This script is read-only. It compares:

1. teacher-curve features that directly derive the current label semantics;
2. runtime-observable geometry/candidate-summary features available to B2.

The main question is not whether the labels are internally consistent, but
whether they generalize across episodes when the model only sees runtime-visible
signals. That is the gating question for B2 offline -> runtime shadow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    w = np.ones((y.shape[0],), dtype=np.float64)
    for cls in (0, 1):
        m = y == cls
        if np.any(m):
            w[m] = 0.5 / float(np.mean(m))
    return w


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    steps: int = 2500,
    lr: float = 0.15,
    l2: float = 1e-2,
) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if sample_weight is None:
        sample_weight = np.ones((x.shape[0],), dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    denom = max(float(np.sum(sample_weight)), 1e-8)
    w = np.zeros((x.shape[1],), dtype=np.float64)
    b = 0.0
    for _ in range(int(steps)):
        z = x @ w + b
        p = _sigmoid(z)
        diff = (p - y) * sample_weight
        grad_w = (x.T @ diff) / denom + float(l2) * w
        grad_b = float(np.sum(diff) / denom)
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w.astype(np.float64), float(b)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    apply = y_true == 1
    keep = y_true == 0
    apply_recall = float(np.mean(y_pred[apply] == 1)) if np.any(apply) else 0.0
    keep_recall = float(np.mean(y_pred[keep] == 0)) if np.any(keep) else 0.0
    return {
        "rows": int(y_true.shape[0]),
        "apply_rows": int(np.sum(apply)),
        "keep_rows": int(np.sum(keep)),
        "apply_recall": apply_recall,
        "keep_recall": keep_recall,
        "balanced_acc": 0.5 * (apply_recall + keep_recall),
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
    }


def _feature_stats(x: np.ndarray, names: list[str]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for i, name in enumerate(names):
        col = np.asarray(x[:, i], dtype=np.float64)
        col = col[np.isfinite(col)]
        if col.size == 0:
            out[name] = {"mean": None, "std": None, "p10": None, "p50": None, "p90": None}
            continue
        out[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "p10": float(np.percentile(col, 10)),
            "p50": float(np.percentile(col, 50)),
            "p90": float(np.percentile(col, 90)),
        }
    return out


def _candidate_summary(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    mask = np.asarray(data["candidate_mask"], dtype=np.float32) > 0.5
    scope = np.asarray(data.get("b2_yaw_aware_candidate_scope_v3", data["candidate_mask"]), dtype=np.float32) > 0.5
    scope &= mask
    yaw = actions[:, :, 5]
    proxy = np.asarray(data["proxy_current_delta_basin_target"], dtype=np.float32)
    base = np.asarray(data["base_action"], dtype=np.float32)
    plan = np.asarray(data.get("planner_base_action_local_raw", data["base_action"]), dtype=np.float32)
    scope_size = np.asarray(data.get("b2_yaw_aware_scope_size_v3", np.sum(scope, axis=1)), dtype=np.float32)
    scope_range = np.asarray(data.get("b2_yaw_aware_scope_yaw_range_v3", np.zeros((yaw.shape[0],), dtype=np.float32)), dtype=np.float32)

    feats = np.zeros((yaw.shape[0], 16), dtype=np.float32)
    for i in range(yaw.shape[0]):
        idx = np.flatnonzero(scope[i])
        local = yaw[i, idx] if idx.size else np.zeros((1,), dtype=np.float32)
        abs_local = np.abs(local)
        pos = float(np.mean(local > 1e-6)) if idx.size else 0.0
        neg = float(np.mean(local < -1e-6)) if idx.size else 0.0
        no = float(np.mean(abs_local <= 0.035)) if idx.size else 1.0
        small = float(np.mean((abs_local > 0.035) & (abs_local <= 0.075))) if idx.size else 0.0
        large = float(np.mean(abs_local > 0.075)) if idx.size else 0.0
        feats[i] = np.asarray(
            [
                float(proxy[i, 5]),
                float(abs(proxy[i, 5])),
                float(np.sign(proxy[i, 5])),
                float(base[i, 5]),
                float(abs(base[i, 5])),
                float(plan[i, 5]),
                float(abs(plan[i, 5])),
                float(np.min(local)),
                float(np.max(local)),
                float(np.mean(local)),
                float(np.std(local)),
                float(np.mean(abs_local)),
                pos,
                neg,
                float(scope_size[i]),
                float(scope_range[i]),
            ],
            dtype=np.float32,
        )
        feats[i, 10] = float(np.std(local)) if idx.size > 1 else 0.0
        feats[i, 11] = float(np.mean(abs_local))
        # fold bin fractions into the last four coordinates via additive cues
        feats[i, 12] = pos - neg
        feats[i, 13] = no + 0.5 * small + 0.25 * large
        feats[i, 14] = float(scope_size[i])
        feats[i, 15] = float(scope_range[i]) + 0.5 * large - 0.25 * no
    names = [
        "proxy_dyaw",
        "proxy_dyaw_abs",
        "proxy_dyaw_sign",
        "base_yaw",
        "base_yaw_abs",
        "planner_yaw",
        "planner_yaw_abs",
        "candidate_yaw_min",
        "candidate_yaw_max",
        "candidate_yaw_mean",
        "candidate_yaw_std",
        "candidate_yaw_abs_mean",
        "candidate_sign_balance",
        "candidate_no_yaw_bias",
        "scope_size",
        "scope_range_aug",
    ]
    return feats, names


def _teacher_curve_features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    scores = np.asarray(data["b2_yaw_cost_curve_scores_v11"], dtype=np.float32)
    no_score = scores[:, 0]
    small_score = np.max(scores[:, 1:3], axis=1)
    large_score = np.max(scores[:, 3:5], axis=1)
    apply_margin = large_score - np.maximum(no_score, small_score)
    feats = np.stack(
        [
            np.asarray(data["yaw_advantage_cont_v11"], dtype=np.float32),
            np.asarray(data["yaw_small_advantage_v11"], dtype=np.float32),
            np.asarray(data["yaw_large_advantage_v11"], dtype=np.float32),
            apply_margin.astype(np.float32),
        ],
        axis=1,
    )
    names = ["yaw_advantage", "yaw_small_advantage", "yaw_large_advantage", "apply_margin"]
    return feats, names


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(train_x, axis=0, keepdims=True)
    sigma = np.std(train_x, axis=0, keepdims=True)
    sigma = np.where(sigma > 1e-6, sigma, 1.0)
    return (train_x - mu) / sigma, (test_x - mu) / sigma


def _run_probe(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, object]:
    train_x = np.asarray(x[train_mask], dtype=np.float64)
    test_x = np.asarray(x[test_mask], dtype=np.float64)
    train_y = np.asarray(y[train_mask], dtype=np.int64)
    test_y = np.asarray(y[test_mask], dtype=np.int64)
    if train_x.shape[0] < 4 or test_x.shape[0] < 2:
        return {"ok": False, "reason": "insufficient_rows"}
    if np.unique(train_y).size < 2 or np.unique(test_y).size < 2:
        return {"ok": False, "reason": "missing_class"}
    train_xs, test_xs = _standardize(train_x, test_x)
    weights = _balanced_weights(train_y)
    w, b = _fit_logistic(train_xs, train_y, sample_weight=weights)
    train_prob = _sigmoid(train_xs @ w + b)
    test_prob = _sigmoid(test_xs @ w + b)
    train_pred = (train_prob >= 0.5).astype(np.int64)
    test_pred = (test_prob >= 0.5).astype(np.int64)
    return {
        "ok": True,
        "train": _metrics(train_y, train_pred),
        "test": _metrics(test_y, test_pred),
        "test_prob_mean": float(np.mean(test_prob)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--focus_episodes", default="18,34,45")
    args = ap.parse_args()

    arr = np.load(args.dataset_npz, allow_pickle=False)
    data = {k: np.asarray(arr[k]) for k in arr.files}
    ep = np.asarray(data["episode_index"], dtype=np.int64)
    label = np.asarray(data.get("yaw_mode3_label_v11", np.full((ep.shape[0],), -1, dtype=np.int64)), dtype=np.int64)
    valid = np.asarray(data.get("yaw_mode_valid_v11", label >= 0), dtype=np.float32) > 0.5
    split = np.asarray(data.get("split_v11", np.zeros((ep.shape[0],), dtype=np.int64)), dtype=np.int64)
    source_profile = np.asarray(data.get("source_profile_v12", np.full((ep.shape[0],), "unknown", dtype="U32"))).astype(str)

    binary_mask = valid & np.isin(label, [0, 2])
    binary_y = (label[binary_mask] == 2).astype(np.int64)
    binary_ep = ep[binary_mask]
    binary_split = split[binary_mask]

    runtime_x, runtime_names = _candidate_summary(data)
    teacher_x, teacher_names = _teacher_curve_features(data)
    runtime_x = runtime_x[binary_mask]
    teacher_x = teacher_x[binary_mask]

    mixed_eps = sorted(
        int(e)
        for e in np.unique(binary_ep)
        if np.any((binary_ep == e) & (binary_y == 0)) and np.any((binary_ep == e) & (binary_y == 1))
    )

    feature_sets = {
        "runtime_observable": (runtime_x, runtime_names),
        "teacher_curve": (teacher_x, teacher_names),
    }

    split_eval: dict[str, object] = {}
    for name, (x, feat_names) in feature_sets.items():
        split_eval[name] = {
            "feature_stats": _feature_stats(x, feat_names),
            "existing_split": _run_probe(x, binary_y, binary_split == 0, binary_split == 1),
        }

    loo: dict[str, dict[str, object]] = {name: {} for name in feature_sets}
    for holdout_ep in mixed_eps:
        train_mask = binary_ep != holdout_ep
        test_mask = binary_ep == holdout_ep
        for name, (x, _) in feature_sets.items():
            loo[name][str(holdout_ep)] = _run_probe(x, binary_y, train_mask, test_mask)

    focus_eps = [int(x) for x in str(args.focus_episodes).split(",") if x.strip()]
    focus_summary = {}
    for e in focus_eps:
        m = binary_ep == e
        if not np.any(m):
            continue
        profs = source_profile[binary_mask][m]
        yy = binary_y[m]
        focus_summary[str(e)] = {
            "rows": int(np.sum(m)),
            "keep_rows": int(np.sum(yy == 0)),
            "apply_rows": int(np.sum(yy == 1)),
            "profile_counts": {str(k): int(v) for k, v in zip(*np.unique(profs, return_counts=True))},
        }

    runtime_loo_bas = [
        float(v["test"]["balanced_acc"])
        for v in loo["runtime_observable"].values()
        if isinstance(v, dict) and v.get("ok")
    ]
    teacher_loo_bas = [
        float(v["test"]["balanced_acc"])
        for v in loo["teacher_curve"].values()
        if isinstance(v, dict) and v.get("ok")
    ]
    runtime_focus = {
        k: float(v["test"]["balanced_acc"])
        for k, v in loo["runtime_observable"].items()
        if isinstance(v, dict) and v.get("ok") and int(k) in focus_eps
    }
    diagnosis = {
        "binary_rows": int(binary_mask.sum()),
        "mixed_label_episodes": mixed_eps,
        "runtime_observable_mean_loo_balanced_acc": float(np.mean(runtime_loo_bas)) if runtime_loo_bas else None,
        "teacher_curve_mean_loo_balanced_acc": float(np.mean(teacher_loo_bas)) if teacher_loo_bas else None,
        "runtime_focus_episode_balanced_acc": runtime_focus,
        "runtime_observable_learnable": bool(runtime_loo_bas and np.mean(runtime_loo_bas) >= 0.65 and min(runtime_loo_bas) >= 0.55),
        "teacher_curve_learnable": bool(teacher_loo_bas and np.mean(teacher_loo_bas) >= 0.90),
    }
    if diagnosis["teacher_curve_learnable"] and not diagnosis["runtime_observable_learnable"]:
        diagnosis["summary"] = (
            "labels are internally consistent in teacher-curve space but do not generalize "
            "from runtime-observable features across episodes"
        )
        diagnosis["recommended_action"] = "reconstruct_v14_mode_semantics"
    elif diagnosis["runtime_observable_learnable"]:
        diagnosis["summary"] = "runtime-visible features appear sufficient; continue trainer/model work"
        diagnosis["recommended_action"] = "continue_v13_training"
    else:
        diagnosis["summary"] = "both teacher and runtime probes are weak; inspect label construction first"
        diagnosis["recommended_action"] = "inspect_label_construction"

    result = {
        "dataset_npz": str(args.dataset_npz),
        "feature_sets": split_eval,
        "leave_one_episode_out": loo,
        "focus_episode_label_summary": focus_summary,
        "diagnosis": diagnosis,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
