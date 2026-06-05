#!/usr/bin/env python3
"""Train a lightweight non-privileged runtime XY residual calibrator.

Inputs are evaluator trace rows.  Features come only from runtime-visible
localizer / estimated-basin proxy fields.  Labels use privileged true residuals
for offline calibration and evaluation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.runtime_xy_residual import (  # noqa: E402
    DEFAULT_RUNTIME_XY_FEATURE_NAMES,
    RuntimeXYAffineCalibration,
    RuntimeXYMLPCalibration,
    runtime_xy_context_feature_names,
    runtime_xy_context_feature_vector_from_trace,
    runtime_xy_feature_vector_from_trace,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_from_path(path: Path) -> int:
    for token in path.stem.split("_"):
        if token.startswith("ep") and token[2:].isdigit():
            return int(token[2:])
    return -1


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in paths:
        candidates = sorted(item.glob("ep*_gripper_trace.jsonl")) if item.is_dir() else [item]
        for path in candidates:
            ep = _episode_from_path(path)
            for row in _read_jsonl(path):
                r = dict(row)
                r.setdefault("episode_idx", ep)
                r.setdefault("_load_order", len(out))
                out.append(r)
    return out


def resolve_trace_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        if item.is_dir():
            files.extend(sorted(item.glob("ep*_gripper_trace.jsonl")))
        else:
            files.append(item)
    return files


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    ep = int(row.get("episode_idx", -1))
    step = row.get("step", row.get("step_idx", row.get("episode_loop_idx", row.get("_load_order", 0))))
    try:
        step_idx = int(step)
    except Exception:
        step_idx = int(row.get("_load_order", 0))
    return ep, step_idx, int(row.get("_load_order", 0))


def _group_rows_by_episode(rows: list[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episodes[int(row.get("episode_idx", -1))].append(dict(row))
    grouped: list[list[dict[str, Any]]] = []
    for ep in sorted(episodes):
        grouped.append(sorted(episodes[ep], key=_row_sort_key))
    return grouped


def _vec2(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    return arr[:2].astype(np.float32)


def build_dataset(
    rows: list[Mapping[str, Any]],
    *,
    feature_names: tuple[str, ...],
    active_only: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[list[dict[str, Any]]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    contexts: list[list[dict[str, Any]]] = []
    for episode_rows in _group_rows_by_episode(rows):
        history: list[dict[str, Any]] = []
        for row in episode_rows:
            if active_only and not bool(row.get("grasp_probe_active", False)):
                history.append(dict(row))
                continue
            y = _vec2(row.get("grasp_probe_pre_true_error_t"))
            x = runtime_xy_feature_vector_from_trace(row, feature_names)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                history.append(dict(row))
                continue
            xs.append(x)
            ys.append(y)
            kept.append(dict(row))
            contexts.append([])
            history.append(dict(row))
    if not xs:
        return np.zeros((0, len(feature_names)), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), [], []
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), kept, contexts


def build_temporal_dataset(
    rows: list[Mapping[str, Any]],
    *,
    base_feature_names: tuple[str, ...],
    window_size: int,
    active_only: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[list[dict[str, Any]]], tuple[str, ...]]:
    window = max(1, int(window_size))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    contexts: list[list[dict[str, Any]]] = []
    flattened_feature_names = runtime_xy_context_feature_names(base_feature_names, window)
    for episode_rows in _group_rows_by_episode(rows):
        history: list[dict[str, Any]] = []
        for row in episode_rows:
            if active_only and not bool(row.get("grasp_probe_active", False)):
                history.append(dict(row))
                continue
            y = _vec2(row.get("grasp_probe_pre_true_error_t"))
            history_rows = list(reversed(history[-(window - 1) :])) if window > 1 else []
            x = runtime_xy_context_feature_vector_from_trace(
                row,
                history_rows=history_rows,
                base_feature_names=base_feature_names,
                window_size=window,
            )
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                history.append(dict(row))
                continue
            xs.append(x)
            ys.append(y)
            kept.append(dict(row))
            contexts.append(history_rows)
            history.append(dict(row))
    if not xs:
        return np.zeros((0, len(flattened_feature_names)), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), [], [], flattened_feature_names
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), kept, contexts, flattened_feature_names


def _cosine_rows(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(pred[:, :2], axis=1) * np.linalg.norm(target[:, :2], axis=1)
    out = np.full((pred.shape[0],), np.nan, dtype=np.float32)
    ok = denom > 1.0e-12
    out[ok] = np.sum(pred[ok, :2] * target[ok, :2], axis=1) / denom[ok]
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    ok = np.isfinite(arr)
    if not np.any(ok):
        return 0.0
    if weights is None:
        return float(np.mean(arr[ok]))
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)
    ww = ww[ok]
    if ww.size == 0 or float(np.sum(ww)) <= 1.0e-12:
        return float(np.mean(arr[ok]))
    return float(np.sum(arr[ok] * ww) / np.sum(ww))


def _bounded_xy_control_step(pred: np.ndarray, *, xy_gain: float, max_xy_step: float) -> np.ndarray:
    correction = np.asarray(pred, dtype=np.float32).copy()
    if correction.size == 0:
        return correction
    correction = float(xy_gain) * correction
    norm = np.linalg.norm(correction[:, :2], axis=1, keepdims=True)
    if float(max_xy_step) > 0.0:
        scale = np.ones_like(norm, dtype=np.float32)
        mask = norm > float(max_xy_step)
        scale[mask] = float(max_xy_step) / np.maximum(norm[mask], 1.0e-9)
        correction[:, :2] = correction[:, :2] * scale
    return correction.astype(np.float32)


def _metrics(pred: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    if pred.size == 0:
        return {"rows": 0, "mae": 0.0, "rmse": 0.0, "cosine_mean": 0.0, "cosine_gt_05_rate": 0.0, "sign_match_rate": 0.0}
    err = pred[:, :2] - target[:, :2]
    cos = _cosine_rows(pred, target)
    sign_match = (np.sign(pred[:, :2]) == np.sign(target[:, :2])).astype(np.float32)
    abs_err = np.mean(np.abs(err), axis=1)
    sq_err = np.mean(err * err, axis=1)
    return {
        "rows": int(pred.shape[0]),
        "mae": _weighted_mean(abs_err, weights),
        "rmse": float(np.sqrt(max(0.0, _weighted_mean(sq_err, weights)))),
        "cosine_mean": _weighted_mean(cos, weights),
        "cosine_gt_05_rate": _weighted_mean((cos > 0.5).astype(np.float32), weights) if cos.size else 0.0,
        "sign_match_rate": _weighted_mean(np.mean(sign_match, axis=1), weights),
    }


def _control_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    xy_gain: float,
    max_xy_step: float,
) -> dict[str, float]:
    if pred.size == 0:
        return {"control_contraction_rate": 0.0, "control_overshoot_rate": 0.0, "control_reverse_rate": 0.0, "control_mae": 0.0}
    control = _bounded_xy_control_step(pred, xy_gain=xy_gain, max_xy_step=max_xy_step)
    pre = np.asarray(target[:, :2], dtype=np.float32)
    post = pre - control[:, :2]
    pre_norm = np.linalg.norm(pre, axis=1)
    post_norm = np.linalg.norm(post, axis=1)
    control_norm = np.linalg.norm(control[:, :2], axis=1)
    dot = np.sum(control[:, :2] * pre, axis=1)
    return {
        "control_contraction_rate": _weighted_mean((post_norm < pre_norm).astype(np.float32), weights),
        "control_worsen_rate": _weighted_mean((post_norm > pre_norm).astype(np.float32), weights),
        "control_overshoot_rate": _weighted_mean((control_norm > pre_norm).astype(np.float32), weights),
        "control_reverse_rate": _weighted_mean((dot < 0.0).astype(np.float32), weights),
        "control_mae": _weighted_mean(np.mean(np.abs(post), axis=1), weights),
    }


def split_by_episode(rows: list[Mapping[str, Any]], val_fraction: float) -> tuple[set[int], set[int]]:
    eps = sorted({int(r.get("episode_idx", -1)) for r in rows})
    if len(eps) <= 1:
        return set(eps), set(eps)
    n_val = max(1, int(round(len(eps) * float(val_fraction))))
    val = set(eps[-n_val:])
    train = set(e for e in eps if e not in val)
    return train, val


def _parse_episode_set(text: str | None) -> set[int]:
    if not text:
        return set()
    out: set[int] = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("ep"):
            item = item[2:]
        out.add(int(item))
    return out


def _bool_row(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _float_row(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _row_contracts(row: Mapping[str, Any]) -> bool:
    if "grasp_probe_horizon_xy_delta" in row:
        return bool(_float_row(row, "grasp_probe_horizon_xy_delta") < -1.0e-9)
    before = _float_row(row, "grasp_probe_horizon_pre_xy_error")
    after = _float_row(row, "grasp_probe_horizon_post_xy_error")
    if np.isfinite(before) and np.isfinite(after):
        return bool(after < before)
    return bool(
        _bool_row(row, "grasp_probe_horizon_xy_contracted", False)
        or _bool_row(row, "horizon_xy_contracted", False)
        or _bool_row(row, "scalar_xy_contracted", False)
        or _bool_row(row, "vector_norm_xy_contracted", False)
    )


def _row_bucket(row: Mapping[str, Any]) -> str:
    return str(row.get("failure_bucket", row.get("failure_morphology_bucket", "")) or "unknown")


def _row_alias(row: Mapping[str, Any]) -> str:
    return str(row.get("alias_drift_decision", row.get("yaw_alias_drift_decision", "")) or "unknown")


def _row_observability_bucket(row: Mapping[str, Any]) -> str:
    for key in ("visual_observability_class", "grasp_probe_visibility_bucket", "visibility", "runtime_visibility_bucket"):
        value = str(row.get(key, "") or "")
        if value:
            return value
    if _bool_row(row, "wrist_is_occluded", False):
        return "occluded"
    if _bool_row(row, "wrist_is_low_visibility", False):
        return "low_observability"
    return "unknown"


def _row_weights(
    rows: list[Mapping[str, Any]],
    *,
    base_weight: float,
    active_contract_weight: float,
    hard_bucket_weight: float,
    occlusion_weight: float,
    low_observability_weight: float,
) -> np.ndarray:
    weights = np.full((len(rows),), float(base_weight), dtype=np.float64)
    hard_buckets = {"large_xy_large_yaw", "small_xy_large_yaw", "large_xy_small_yaw", "small_xy_small_yaw"}
    for i, row in enumerate(rows):
        if _row_contracts(row):
            weights[i] *= float(active_contract_weight)
        if _row_bucket(row) in hard_buckets:
            weights[i] *= float(hard_bucket_weight)
        obs = _row_observability_bucket(row)
        if obs == "occluded" or "occlusion" in obs:
            weights[i] *= float(occlusion_weight)
        elif obs in {"low_observability", "low_visibility"} or "low" in obs:
            weights[i] *= float(low_observability_weight)
    return weights.astype(np.float64)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float, sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] == 0:
        raise ValueError("no rows available for runtime XY calibration")
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if sample_weight is None:
        sample_weight = np.ones((x64.shape[0],), dtype=np.float64)
    ww = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if ww.size != x64.shape[0]:
        raise ValueError("sample_weight length must match rows")
    ww = np.clip(ww, 1.0e-9, np.inf)
    mean = np.average(x64, axis=0, weights=ww).reshape(1, -1)
    var = np.average((x64 - mean) ** 2, axis=0, weights=ww).reshape(1, -1)
    std = np.sqrt(var)
    std[std < 1.0e-6] = 1.0
    xn = (x64 - mean) / std
    design = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float64)], axis=1)
    sqrt_w = np.sqrt(ww).reshape(-1, 1)
    weighted_design = design * sqrt_w
    weighted_y = y64 * sqrt_w
    reg = float(ridge) * np.eye(design.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(weighted_design.T @ weighted_design + reg, weighted_design.T @ weighted_y)
    weights_norm = beta[:-1].T
    bias_norm = beta[-1]
    weights = weights_norm / std.reshape(1, -1)
    bias = bias_norm - (weights_norm @ (mean.reshape(-1) / std.reshape(-1)))
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
        raise ValueError("runtime XY calibration produced non-finite weights")
    return weights.astype(np.float32), bias.astype(np.float32)


def _parse_ridge_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if str(part).strip()]
    return values or [1.0e-4]


def _select_ridge(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    *,
    sample_weight: np.ndarray,
    ridge_grid: list[float],
    max_abs_weight: float,
    direction_weight: float,
    sign_weight: float,
    mae_weight: float,
) -> tuple[float, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for ridge in ridge_grid:
        try:
            weights, bias = fit_ridge(x[train_idx], y[train_idx], ridge=float(ridge), sample_weight=sample_weight[train_idx])
            pred_val = x[val_idx] @ weights.T + bias.reshape(1, 2)
            metrics = _metrics(pred_val, y[val_idx], sample_weight[val_idx])
            max_weight = float(np.max(np.abs(weights))) if weights.size else 0.0
            valid = bool(
                np.all(np.isfinite(weights))
                and np.all(np.isfinite(bias))
                and np.all(np.isfinite(pred_val))
                and max_weight <= float(max_abs_weight)
            )
            score = float(
                float(direction_weight) * (1.0 - metrics["cosine_gt_05_rate"])
                + float(sign_weight) * (1.0 - metrics["sign_match_rate"])
                + float(mae_weight) * metrics["mae"]
            )
            candidates.append(
                {
                    "ridge": float(ridge),
                    "valid": bool(valid),
                    "max_abs_weight": float(max_weight),
                    "val": metrics,
                    "score_terms": {
                        "direction_loss": float(1.0 - metrics["cosine_gt_05_rate"]),
                        "sign_loss": float(1.0 - metrics["sign_match_rate"]),
                        "mae": float(metrics["mae"]),
                    },
                    "score": float(score),
                }
            )
            if valid and (best is None or score < best[0]):
                best = (score, float(ridge), weights, bias)
        except Exception as exc:
            candidates.append({"ridge": float(ridge), "valid": False, "error": type(exc).__name__})
    if best is None:
        raise RuntimeError(f"no stable runtime XY ridge candidate found: {candidates}")
    return best[1], best[2], best[3], candidates


def _parse_hidden_dims(text: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in str(text).split(",") if str(part).strip())
    return dims or (32, 16)


def _make_mlp(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, int(dim)))
        layers.append(nn.ReLU())
        prev = int(dim)
    layers.append(nn.Linear(prev, 2))
    return nn.Sequential(*layers)


def _mlp_to_calibration(
    model: nn.Sequential,
    *,
    feature_names: tuple[str, ...],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_size: int = 1,
    base_feature_names: tuple[str, ...] | None = None,
) -> RuntimeXYMLPCalibration:
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for module in model:
        if isinstance(module, nn.Linear):
            layers.append(
                (
                    module.weight.detach().cpu().numpy().astype(np.float32),
                    module.bias.detach().cpu().numpy().astype(np.float32),
                )
            )
    return RuntimeXYMLPCalibration(
        feature_names=feature_names,
        layers=tuple(layers),
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_std=np.asarray(feature_std, dtype=np.float32),
        window_size=int(window_size),
        base_feature_names=tuple(base_feature_names or feature_names),
        source="runtime_xy_mlp_direction_first_calibration",
    )


def _fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    *,
    sample_weight: np.ndarray,
    feature_names: tuple[str, ...],
    hidden_dims: tuple[int, ...],
    epochs: int,
    lr: float,
    weight_decay: float,
    direction_weight: float,
    sign_weight: float,
    mae_weight: float,
    contraction_weight: float,
    control_reverse_weight: float,
    batch_size: int,
    seed: int,
    window_size: int,
    base_feature_names: tuple[str, ...],
    contexts: list[list[dict[str, Any]]],
    xy_gain: float,
    max_xy_step: float,
) -> tuple[RuntimeXYMLPCalibration, list[dict[str, Any]]]:
    if not train_idx:
        raise ValueError("no train rows available for runtime XY MLP")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    x64 = np.asarray(x, dtype=np.float64)
    w_train = np.asarray(sample_weight[train_idx], dtype=np.float64)
    mean = np.average(x64[train_idx], axis=0, weights=w_train)
    var = np.average((x64[train_idx] - mean.reshape(1, -1)) ** 2, axis=0, weights=w_train)
    std = np.sqrt(var)
    std[std < 1.0e-6] = 1.0
    xn = ((x64 - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
    yt = np.asarray(y, dtype=np.float32)
    wt = np.asarray(sample_weight, dtype=np.float32)
    device = torch.device("cpu")
    model = _make_mlp(xn.shape[1], hidden_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_idx_arr = np.asarray(train_idx, dtype=np.int64)
    history: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], dict[str, torch.Tensor]] | None = None
    batch = max(1, int(batch_size))
    eps = 1.0e-6
    temp = 0.002

    def eval_metrics() -> tuple[float, dict[str, Any]]:
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.from_numpy(xn[val_idx]).to(device)).cpu().numpy()
        metrics = _metrics(pred_val, yt[val_idx], wt[val_idx])
        control_metrics = _control_metrics(
            pred_val,
            yt[val_idx],
            wt[val_idx],
            xy_gain=float(xy_gain),
            max_xy_step=float(max_xy_step),
        )
        metrics.update(control_metrics)
        score = float(
            float(direction_weight) * (1.0 - metrics["cosine_gt_05_rate"])
            + float(sign_weight) * (1.0 - metrics["sign_match_rate"])
            + float(mae_weight) * metrics["mae"]
            + float(contraction_weight) * (1.0 - metrics["control_contraction_rate"])
            + float(control_reverse_weight) * metrics["control_reverse_rate"]
            + 0.5 * float(control_reverse_weight) * metrics["control_overshoot_rate"]
            + 0.5 * float(contraction_weight) * metrics.get("control_worsen_rate", 0.0)
        )
        return score, metrics

    for epoch in range(max(1, int(epochs))):
        model.train()
        perm = np.random.permutation(train_idx_arr)
        for start in range(0, perm.size, batch):
            idx = perm[start : start + batch]
            xb = torch.from_numpy(xn[idx]).to(device)
            yb = torch.from_numpy(yt[idx]).to(device)
            wb = torch.from_numpy(wt[idx]).to(device)
            pred = model(xb)
            cos = F.cosine_similarity(pred, yb, dim=1, eps=eps)
            direction_loss = 1.0 - cos
            sign_loss = 1.0 - torch.tanh((pred * yb) / temp).mean(dim=1)
            mae_loss = torch.mean(torch.abs(pred - yb), dim=1)
            control_xy = float(xy_gain) * pred[:, :2]
            control_norm_preclip = torch.linalg.norm(control_xy, dim=1)
            control_scale = torch.ones_like(control_norm_preclip)
            if float(max_xy_step) > 0.0:
                control_scale = torch.where(
                    control_norm_preclip > float(max_xy_step),
                    float(max_xy_step) / torch.clamp(control_norm_preclip, min=eps),
                    torch.ones_like(control_norm_preclip),
                )
            control_xy = control_xy * control_scale[:, None]
            control_norm = torch.linalg.norm(control_xy, dim=1)
            post = yb[:, :2] - control_xy
            pre_norm = torch.linalg.norm(yb[:, :2], dim=1)
            post_norm = torch.linalg.norm(post, dim=1)
            contraction_loss = torch.relu(post_norm - pre_norm)
            worsen_loss = torch.relu(post_norm - pre_norm)
            overshoot_loss = torch.relu(control_norm - pre_norm)
            reverse_loss = torch.relu(-F.cosine_similarity(control_xy, yb[:, :2], dim=1, eps=eps))
            loss_row = (
                float(direction_weight) * direction_loss
                + float(sign_weight) * sign_loss
                + float(mae_weight) * mae_loss
                + float(contraction_weight) * contraction_loss
                + 0.5 * float(contraction_weight) * worsen_loss
                + float(control_reverse_weight) * reverse_loss
                + 0.5 * float(control_reverse_weight) * overshoot_loss
            )
            loss = torch.sum(loss_row * wb) / torch.clamp(torch.sum(wb), min=eps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        if epoch == 0 or epoch == int(epochs) - 1 or (epoch + 1) % max(1, int(epochs) // 5) == 0:
            score, metrics = eval_metrics()
            item = {"epoch": int(epoch + 1), "score": float(score), "val": metrics}
            history.append(item)
            if best is None or score < best[0]:
                best = (
                    score,
                    item,
                    {name: param.detach().cpu().clone() for name, param in model.state_dict().items()},
                )
    if best is not None:
        model.load_state_dict(best[2])
    cal = _mlp_to_calibration(
        model,
        feature_names=feature_names,
        feature_mean=mean,
        feature_std=std,
        window_size=window_size,
        base_feature_names=base_feature_names,
    )
    return cal, history


def train(args: argparse.Namespace) -> dict[str, Any]:
    feature_names = tuple(str(x).strip() for x in str(args.feature_names).split(",") if str(x).strip())
    trace_path_args = [Path(p) for p in args.trace_paths]
    rows = load_rows(trace_path_args)
    window_size = max(1, int(getattr(args, "window_size", 1)))
    active_only = not bool(getattr(args, "include_inactive_rows", False))
    contraction_weight = float(getattr(args, "contraction_weight", 0.15))
    model_type = str(getattr(args, "model_type", "affine") or "affine")
    if model_type == "temporal_mlp":
        x, y, kept, contexts, calibration_feature_names = build_temporal_dataset(
            rows,
            base_feature_names=feature_names,
            window_size=window_size,
            active_only=active_only,
        )
    else:
        x, y, kept, contexts = build_dataset(rows, feature_names=feature_names, active_only=active_only)
        calibration_feature_names = feature_names
    explicit_val_eps = _parse_episode_set(getattr(args, "val_episodes", ""))
    if explicit_val_eps:
        all_eps = {int(r.get("episode_idx", -1)) for r in kept}
        val_eps = set(ep for ep in explicit_val_eps if ep in all_eps)
        train_eps = set(ep for ep in all_eps if ep not in val_eps)
        if not val_eps:
            train_eps, val_eps = split_by_episode(kept, float(args.val_fraction))
    else:
        train_eps, val_eps = split_by_episode(kept, float(args.val_fraction))
    train_idx = [i for i, r in enumerate(kept) if int(r.get("episode_idx", -1)) in train_eps]
    val_idx = [i for i, r in enumerate(kept) if int(r.get("episode_idx", -1)) in val_eps]
    if not train_idx:
        train_idx = list(range(x.shape[0]))
    if not val_idx:
        val_idx = train_idx
    sample_weight = _row_weights(
        kept,
        base_weight=float(args.base_row_weight),
        active_contract_weight=float(args.active_contract_weight),
        hard_bucket_weight=float(args.hard_bucket_weight),
        occlusion_weight=float(args.occlusion_weight),
        low_observability_weight=float(args.low_observability_weight),
    )
    ridge_grid = _parse_ridge_grid(getattr(args, "ridge_grid", "") or str(args.ridge))
    raw_proxy = x[:, [feature_names.index("local_dx"), feature_names.index("local_dy")]] if "local_dx" in feature_names and "local_dy" in feature_names else x[:, :2]
    selected_ridge = 0.0
    ridge_candidates: list[dict[str, Any]] = []
    mlp_history: list[dict[str, Any]] = []
    if model_type == "affine":
        selected_ridge, weights, bias, ridge_candidates = _select_ridge(
            x,
            y,
            train_idx,
            val_idx,
            sample_weight=sample_weight,
            ridge_grid=ridge_grid,
            max_abs_weight=float(args.max_abs_weight),
            direction_weight=float(args.direction_weight),
            sign_weight=float(args.sign_weight),
            mae_weight=float(args.mae_weight),
        )
        cal: RuntimeXYAffineCalibration | RuntimeXYMLPCalibration = RuntimeXYAffineCalibration(
            feature_names=feature_names,
            weights=weights,
            bias=bias,
            source="runtime_xy_affine_direction_first_calibration",
        )
        pred = x @ weights.T + bias.reshape(1, 2)
        max_abs_param = float(np.max(np.abs(weights))) if weights.size else 0.0
    elif model_type in {"mlp", "temporal_mlp"}:
        calibration_feature_names = calibration_feature_names if model_type == "temporal_mlp" else feature_names
        cal, mlp_history = _fit_mlp(
            x,
            y,
            train_idx,
            val_idx,
            sample_weight=sample_weight,
            feature_names=calibration_feature_names,
            hidden_dims=_parse_hidden_dims(str(args.hidden_dims)),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            direction_weight=float(args.direction_weight),
            sign_weight=float(args.sign_weight),
            mae_weight=float(args.mae_weight),
            contraction_weight=contraction_weight,
            control_reverse_weight=float(getattr(args, "control_reverse_weight", 0.25)),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            window_size=window_size,
            base_feature_names=feature_names,
            contexts=contexts,
            xy_gain=float(getattr(args, "xy_gain", 0.35)),
            max_xy_step=float(getattr(args, "max_xy_step", 0.003)),
        )
        pred = np.stack([cal.predict_from_trace(row, history_rows=context)[0] for row, context in zip(kept, contexts)]).astype(np.float32)
        max_abs_param = float(
            max(
                [float(np.max(np.abs(weights))) for weights, _bias in cal.layers]
                + [float(np.max(np.abs(bias))) for _weights, bias in cal.layers]
            )
        )
    else:
        raise ValueError(f"unknown --model_type {model_type}; expected affine or mlp")

    by_episode: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(kept):
        by_episode[int(row.get("episode_idx", -1))].append(i)

    def grouped_metrics(group_key) -> list[dict[str, Any]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(kept):
            groups[str(group_key(row))].append(i)
        return [
            {
                "group": str(group),
                "rows": int(len(idxs)),
                "mean_row_weight": float(np.mean(sample_weight[idxs])) if idxs else 0.0,
                "raw_proxy": _metrics(raw_proxy[idxs], y[idxs], sample_weight[idxs]),
                "calibrated": _metrics(pred[idxs], y[idxs], sample_weight[idxs]),
            }
            for group, idxs in sorted(groups.items())
        ]

    resolved_trace_files = [str(p) for p in resolve_trace_files(trace_path_args)]
    contract_flags = [_row_contracts(r) for r in kept]
    row_weight_summary = {
        "min": float(np.min(sample_weight)) if sample_weight.size else 0.0,
        "max": float(np.max(sample_weight)) if sample_weight.size else 0.0,
        "mean": float(np.mean(sample_weight)) if sample_weight.size else 0.0,
        "active_contract_rows": int(sum(bool(x) for x in contract_flags)),
        "active_contract_row_rate": float(np.mean(contract_flags)) if contract_flags else 0.0,
        "failure_bucket_counts": dict(Counter(_row_bucket(r) for r in kept)),
        "alias_drift_decision_counts": dict(Counter(_row_alias(r) for r in kept)),
        "observability_bucket_counts": dict(Counter(_row_observability_bucket(r) for r in kept)),
    }
    report = {
        "schema_version": "c2c_v2_runtime_xy_calibrator_train_v1",
        "training_objective": "control_aware_direction_first_temporal" if model_type == "temporal_mlp" else "control_aware_direction_first",
        "model_type": model_type,
        "selection_metric": "direction_weight*(1-cosine_gt_05_rate)+sign_weight*(1-sign_match_rate)+mae_weight*mae+contraction_weight*(1-control_contraction_rate)+control_reverse_weight*control_reverse_rate+0.5*control_reverse_weight*control_overshoot_rate",
        "rows": int(x.shape[0]),
        "active_only": bool(active_only),
        "include_inactive_rows": bool(getattr(args, "include_inactive_rows", False)),
        "trace_paths": [str(p) for p in trace_path_args],
        "resolved_trace_files": resolved_trace_files,
        "feature_names": list(feature_names),
        "context_feature_names": list(calibration_feature_names),
        "window_size": int(window_size),
        "explicit_val_episodes": sorted(explicit_val_eps),
        "train_episodes": sorted(train_eps),
        "val_episodes": sorted(val_eps),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "ridge": float(selected_ridge),
        "ridge_grid": [float(v) for v in ridge_grid],
        "ridge_candidates": ridge_candidates,
        "mlp_config": {
            "hidden_dims": list(_parse_hidden_dims(str(args.hidden_dims))),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "window_size": int(window_size),
            "base_feature_names": list(feature_names),
        } if model_type in {"mlp", "temporal_mlp"} else {},
        "mlp_history": mlp_history,
        "selection_weights": {
            "direction_weight": float(args.direction_weight),
            "sign_weight": float(args.sign_weight),
            "mae_weight": float(args.mae_weight),
            "contraction_weight": contraction_weight,
            "control_reverse_weight": float(getattr(args, "control_reverse_weight", 0.25)),
        },
        "control_config": {
            "xy_gain": float(getattr(args, "xy_gain", 0.35)),
            "max_xy_step": float(getattr(args, "max_xy_step", 0.003)),
        },
        "row_weight_config": {
            "base_row_weight": float(args.base_row_weight),
            "active_contract_weight": float(args.active_contract_weight),
            "hard_bucket_weight": float(args.hard_bucket_weight),
            "occlusion_weight": float(args.occlusion_weight),
            "low_observability_weight": float(args.low_observability_weight),
        },
        "row_weight_summary": row_weight_summary,
        "max_abs_weight": float(max_abs_param),
        "raw_proxy_train": _metrics(raw_proxy[train_idx], y[train_idx], sample_weight[train_idx]),
        "raw_proxy_val": _metrics(raw_proxy[val_idx], y[val_idx], sample_weight[val_idx]),
        "calibrated_train": _metrics(pred[train_idx], y[train_idx], sample_weight[train_idx]),
        "calibrated_val": _metrics(pred[val_idx], y[val_idx], sample_weight[val_idx]),
        "raw_proxy_train_control": _control_metrics(raw_proxy[train_idx], y[train_idx], sample_weight[train_idx], xy_gain=float(getattr(args, "xy_gain", 0.35)), max_xy_step=float(getattr(args, "max_xy_step", 0.003))),
        "raw_proxy_val_control": _control_metrics(raw_proxy[val_idx], y[val_idx], sample_weight[val_idx], xy_gain=float(getattr(args, "xy_gain", 0.35)), max_xy_step=float(getattr(args, "max_xy_step", 0.003))),
        "calibrated_train_control": _control_metrics(pred[train_idx], y[train_idx], sample_weight[train_idx], xy_gain=float(getattr(args, "xy_gain", 0.35)), max_xy_step=float(getattr(args, "max_xy_step", 0.003))),
        "calibrated_val_control": _control_metrics(pred[val_idx], y[val_idx], sample_weight[val_idx], xy_gain=float(getattr(args, "xy_gain", 0.35)), max_xy_step=float(getattr(args, "max_xy_step", 0.003))),
        "by_episode": [
            {
                "episode_idx": int(ep),
                "rows": int(len(idxs)),
                "mean_row_weight": float(np.mean(sample_weight[idxs])) if idxs else 0.0,
                "raw_proxy": _metrics(raw_proxy[idxs], y[idxs], sample_weight[idxs]),
                "calibrated": _metrics(pred[idxs], y[idxs], sample_weight[idxs]),
            }
            for ep, idxs in sorted(by_episode.items())
        ],
        "by_failure_bucket": grouped_metrics(_row_bucket),
        "by_alias_drift_decision": grouped_metrics(_row_alias),
        "by_observability_bucket": grouped_metrics(_row_observability_bucket),
        "runtime_upgrade_gate": {
            "requires_mp4_runtime_ab": True,
            "requires_hard_bucket_runtime_ab": True,
            "decision": "pending_runtime_ab_validation",
        },
        "uses_privileged_label_for_training": True,
        "uses_privileged_runtime": False,
        "calibration": cal.to_dict(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_calibration.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_calibration, "w", encoding="utf-8") as handle:
        json.dump(cal.to_dict(), handle, indent=2)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    lines = [
        "# Runtime XY Calibrator",
        "",
        f"- rows: `{report['rows']}`",
        f"- train_rows: `{report['train_rows']}`",
        f"- val_rows: `{report['val_rows']}`",
        f"- training_objective: `{report['training_objective']}`",
        f"- model_type: `{report['model_type']}`",
        f"- window_size: `{report['window_size']}`",
        f"- control_xy_gain: `{report['control_config']['xy_gain']}`",
        f"- control_max_xy_step: `{report['control_config']['max_xy_step']}`",
        f"- raw_proxy_val_cosine_gt_05_rate: `{report['raw_proxy_val']['cosine_gt_05_rate']:.3f}`",
        f"- calibrated_val_cosine_gt_05_rate: `{report['calibrated_val']['cosine_gt_05_rate']:.3f}`",
        f"- raw_proxy_val_sign_match_rate: `{report['raw_proxy_val']['sign_match_rate']:.3f}`",
        f"- calibrated_val_sign_match_rate: `{report['calibrated_val']['sign_match_rate']:.3f}`",
        f"- raw_proxy_val_mae: `{report['raw_proxy_val']['mae']:.6f}`",
        f"- calibrated_val_mae: `{report['calibrated_val']['mae']:.6f}`",
        f"- calibrated_val_control_contraction_rate: `{report['calibrated_val_control']['control_contraction_rate']:.3f}`",
        f"- calibrated_val_control_worsen_rate: `{report['calibrated_val_control']['control_worsen_rate']:.3f}`",
        f"- calibrated_val_control_overshoot_rate: `{report['calibrated_val_control']['control_overshoot_rate']:.3f}`",
        f"- active_contract_rows: `{report['row_weight_summary']['active_contract_rows']}`",
        f"- runtime_upgrade_gate: `{report['runtime_upgrade_gate']['decision']}`",
        f"- calibration: `{args.output_calibration}`",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_paths", nargs="+", required=True, help="Trace JSONL files or directories containing ep*_gripper_trace.jsonl")
    ap.add_argument("--output_calibration", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_affine_calibration.json"))
    ap.add_argument("--output_json", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/runtime_xy_calibrator_train.json"))
    ap.add_argument("--output_md", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/runtime_xy_calibrator_train.md"))
    ap.add_argument("--feature_names", type=str, default=",".join(DEFAULT_RUNTIME_XY_FEATURE_NAMES))
    ap.add_argument("--model_type", type=str, default="affine", choices=["affine", "mlp", "temporal_mlp"])
    ap.add_argument("--ridge", type=float, default=1.0e-4)
    ap.add_argument("--ridge_grid", type=str, default="1e-4,1e-3,1e-2,1e-1,1,10,100")
    ap.add_argument("--max_abs_weight", type=float, default=1000.0)
    ap.add_argument("--val_fraction", type=float, default=0.25)
    ap.add_argument("--val_episodes", type=str, default="", help="Comma-separated episode ids, e.g. ep026,ep027, to force into validation.")
    ap.add_argument("--active_only", action="store_true", default=True)
    ap.add_argument(
        "--include_inactive_rows",
        action="store_true",
        default=False,
        help="Include inactive trace rows in the temporal training set while keeping active/contractive rows weighted more heavily.",
    )
    ap.add_argument("--direction_weight", type=float, default=1.0)
    ap.add_argument("--sign_weight", type=float, default=0.5)
    ap.add_argument("--mae_weight", type=float, default=0.05)
    ap.add_argument("--contraction_weight", type=float, default=0.15)
    ap.add_argument("--control_reverse_weight", type=float, default=0.25)
    ap.add_argument("--xy_gain", type=float, default=0.35)
    ap.add_argument("--max_xy_step", type=float, default=0.003)
    ap.add_argument("--base_row_weight", type=float, default=1.0)
    ap.add_argument("--active_contract_weight", type=float, default=3.0)
    ap.add_argument("--hard_bucket_weight", type=float, default=1.5)
    ap.add_argument("--occlusion_weight", type=float, default=1.5)
    ap.add_argument("--low_observability_weight", type=float, default=1.25)
    ap.add_argument("--hidden_dims", type=str, default="32,16")
    ap.add_argument("--window_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
