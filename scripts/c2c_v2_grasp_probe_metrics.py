#!/usr/bin/env python3
"""Shared XY probe metric helpers for C2C v2 grasp audits."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def trace_vec(row: Mapping[str, Any], key: str, *, length: int = 4) -> np.ndarray:
    value = row.get(key, None)
    if value is None:
        return np.full((length,), np.nan, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), constant_values=np.nan)
    return arr[:length]


def xy_norm(vec: Any) -> float:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return float("nan")
    return float(np.hypot(float(arr[0]), float(arr[1])))


def _first_finite_scalar(row: Mapping[str, Any], keys: list[str]) -> tuple[float, str]:
    for key in keys:
        if key in row:
            value = safe_float(row.get(key, float("nan")))
            if np.isfinite(value):
                return value, key
    return float("nan"), ""


def _first_finite_vector_xy(row: Mapping[str, Any], keys: list[str]) -> tuple[float, str]:
    for key in keys:
        if key not in row:
            continue
        vec = trace_vec(row, key)
        value = xy_norm(vec[:2])
        if np.isfinite(value):
            return value, key
    return float("nan"), ""


def grasp_probe_xy_metric_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return scalar and vector XY before/after metrics with a shared fallback order."""

    scalar_before, scalar_before_source = _first_finite_scalar(
        row,
        [
            "grasp_probe_horizon_pre_xy_error",
            "grasp_probe_pre_xy_error",
            "oracle_xy_before",
        ],
    )
    if not np.isfinite(scalar_before):
        scalar_before, scalar_before_source = _first_finite_vector_xy(
            row,
            [
                "grasp_probe_pre_true_error_t",
                "true_basin_error_t",
            ],
        )
        if scalar_before_source:
            scalar_before_source = f"norm({scalar_before_source}[:2])"

    scalar_after, scalar_after_source = _first_finite_scalar(
        row,
        [
            "grasp_probe_horizon_final_xy_error",
            "grasp_probe_horizon_post_xy_error",
            "grasp_probe_post_xy_error",
            "oracle_xy_after",
        ],
    )
    if not np.isfinite(scalar_after):
        scalar_after, scalar_after_source = _first_finite_vector_xy(
            row,
            [
                "grasp_probe_horizon_final_true_error_t",
                "grasp_probe_post_true_error_t",
                "true_basin_error_t_plus_1",
            ],
        )
        if scalar_after_source:
            scalar_after_source = f"norm({scalar_after_source}[:2])"

    vector_before = xy_norm(trace_vec(row, "grasp_probe_pre_true_error_t")[:2])
    vector_after = xy_norm(
        trace_vec(row, "grasp_probe_horizon_final_true_error_t")[:2]
        if row.get("grasp_probe_horizon_final_true_error_t") is not None
        else trace_vec(row, "grasp_probe_post_true_error_t")[:2]
    )
    if not np.isfinite(vector_after) and row.get("true_basin_error_t_plus_1") is not None:
        vector_after = xy_norm(trace_vec(row, "true_basin_error_t_plus_1")[:2])

    scalar_delta = scalar_after - scalar_before if np.isfinite(scalar_after) and np.isfinite(scalar_before) else float("nan")
    vector_delta = vector_after - vector_before if np.isfinite(vector_after) and np.isfinite(vector_before) else float("nan")
    scalar_contracted = bool(np.isfinite(scalar_delta) and scalar_delta < -1.0e-9)
    vector_contracted = bool(np.isfinite(vector_delta) and vector_delta < -1.0e-9)
    scalar_vector_agree = bool(
        np.isfinite(scalar_before)
        and np.isfinite(scalar_after)
        and np.isfinite(vector_before)
        and np.isfinite(vector_after)
        and abs(scalar_before - vector_before) <= 1.0e-6
        and abs(scalar_after - vector_after) <= 1.0e-6
    )

    return {
        "scalar_xy_before": scalar_before,
        "scalar_xy_after": scalar_after,
        "scalar_xy_delta": scalar_delta,
        "scalar_xy_contracted": scalar_contracted,
        "vector_xy_before": vector_before,
        "vector_xy_after": vector_after,
        "vector_xy_delta": vector_delta,
        "vector_norm_contracted": vector_contracted,
        "scalar_vector_xy_agree": scalar_vector_agree,
        "scalar_xy_before_source": scalar_before_source or "",
        "scalar_xy_after_source": scalar_after_source or "",
    }
