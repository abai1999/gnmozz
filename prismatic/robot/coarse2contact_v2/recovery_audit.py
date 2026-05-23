"""Helpers for recovery-style near-grasp auditing in Coarse2Contact v2."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from prismatic.robot.residual_transforms import world_delta_to_local


_TRACE_EP_RE = re.compile(r"ep(\d+)_gripper_trace\.jsonl$")


def trace_episode_index(trace_path: Path) -> int:
    match = _TRACE_EP_RE.search(trace_path.name)
    if match is None:
        return -1
    return int(match.group(1))


def load_trace_rows(trace_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(trace_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_trace_rows_by_step(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row.get("step", -1)): dict(row) for row in rows}


def _load_pickle(path: Path) -> Any:
    import pickle

    with open(path, "rb") as handle:
        return pickle.load(handle)


def episode_dir_from_task_root(task_root: Path, episode_idx: int) -> Path:
    return Path(task_root) / "train" / "episodes" / f"episode{int(episode_idx):03d}"


def load_episode_inputs(episode_dir: Path) -> tuple[dict[str, Any], int | None]:
    inputs_path = Path(episode_dir) / "model_inputs.npz"
    if not inputs_path.exists():
        return {}, None
    model_inputs = np.load(inputs_path, allow_pickle=True)
    close_idx = first_close_index(model_inputs.get("gripper_open"))
    return {k: model_inputs[k] for k in model_inputs.files}, close_idx


def first_close_index(gripper_open: Any) -> int | None:
    opened = np.asarray(gripper_open, dtype=np.float32).reshape(-1)
    if opened.size == 0:
        return None
    for idx in range(1, opened.size):
        if opened[idx - 1] > 0.5 and opened[idx] <= 0.5:
            return int(idx)
    return None


def _load_img(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def binary_mask(mask_img: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask_img)
    if arr.ndim == 2:
        return arr > 0
    return np.any(arr > 0, axis=-1)


def color_mask(rgb: np.ndarray, color_hint: str | None) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    hint = (color_hint or "").lower()
    if hint == "red":
        return (r > 90.0) & (r > 1.25 * g) & (r > 1.15 * b)
    if hint == "blue":
        return (b > 90.0) & (b > 1.15 * r) & (b > 1.10 * g)
    return np.ones(rgb.shape[:2], dtype=bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    if mask.size == 0 or not np.any(mask):
        return mask
    labeled, num = ndi.label(mask)
    if num <= 1:
        return mask
    counts = ndi.sum(mask.astype(np.int32), labeled, index=np.arange(1, num + 1))
    keep = int(np.argmax(counts) + 1)
    return labeled == keep


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.mean(ys))


def principal_axis(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 0.0
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float32))
    if not np.all(np.isfinite(vals)):
        return 0.0
    major = vecs[:, int(np.argmax(vals))]
    return float(np.arctan2(float(major[1]), float(major[0])))


def depth_summary(depth: np.ndarray | None, mask: np.ndarray) -> tuple[float, float]:
    if depth is None or mask.size == 0 or not np.any(mask):
        return float("nan"), 0.0
    values = np.asarray(depth, dtype=np.float32)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), 0.0
    p95 = float(np.percentile(values, 95))
    p05 = float(np.percentile(values, 5))
    return float(np.median(values)), float(max(p95 - p05, 0.0))


def wrap_with_period(angle: float, period: float | None) -> float:
    angle = float(angle)
    if period is None or period <= 0.0:
        return angle
    return float(((angle + 0.5 * period) % period) - 0.5 * period)


def scale_from_depth(depth_m: float, width: int, height: int) -> tuple[float, float]:
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        depth_m = 0.25
    fx = max(width * 1.15, 1.0)
    fy = max(height * 1.15, 1.0)
    return depth_m / fx, depth_m / fy


def frame_keypoints_from_crop(
    rgb: np.ndarray,
    depth: np.ndarray,
    obj_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    *,
    symmetry: float | None = None,
) -> dict[str, float]:
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    crop_depth = depth[y0:y1, x0:x1]
    crop_mask = np.asarray(obj_mask[y0:y1, x0:x1], dtype=bool)
    h, w = crop_mask.shape[:2]
    if h <= 0 or w <= 0 or not np.any(crop_mask):
        return {
            "center_u": 0.5,
            "center_v": 0.5,
            "axis_u": 0.5,
            "axis_v": 0.5,
            "axis_pos_u": 0.5,
            "axis_pos_v": 0.5,
            "axis_neg_u": 0.5,
            "axis_neg_v": 0.5,
            "axis_dir_x": 1.0,
            "axis_dir_y": 0.0,
            "frame_confidence": 0.0,
            "frame_observability": 0.0,
            "frame_axis_strength": 0.0,
            "frame_completeness": 0.0,
            "frame_border_touch": 1.0,
        }
    cx, cy = centroid(crop_mask)
    axis = principal_axis(crop_mask)
    if symmetry and symmetry > 0.0:
        axis = wrap_with_period(axis, symmetry)
    if np.cos(axis) < 0.0 or (abs(np.cos(axis)) < 1e-6 and np.sin(axis) < 0.0):
        axis = float(axis + np.pi)
    axis_len = 0.30 * float(min(w, h))
    axis_dx = float(np.cos(axis) * axis_len)
    axis_dy = float(np.sin(axis) * axis_len)
    ax_pos = float(np.clip(cx + axis_dx, 0.0, max(w - 1, 0)))
    ay_pos = float(np.clip(cy + axis_dy, 0.0, max(h - 1, 0)))
    ax_neg = float(np.clip(cx - axis_dx, 0.0, max(w - 1, 0)))
    ay_neg = float(np.clip(cy - axis_dy, 0.0, max(h - 1, 0)))
    depth_med, depth_spread = depth_summary(crop_depth, crop_mask)
    observability = float(np.count_nonzero(crop_mask) / max(crop_mask.size, 1))
    confidence = float(np.clip(0.15 + 3.0 * observability - 0.2 * depth_spread, 0.0, 1.0))
    ys, xs = np.nonzero(crop_mask)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    margin_px = max(2, int(round(0.04 * min(w, h))))
    touches = (
        x_min <= margin_px
        or y_min <= margin_px
        or x_max >= (w - 1 - margin_px)
        or y_max >= (h - 1 - margin_px)
    )
    border_touch = 1.0 if touches else 0.0
    completeness = float(np.clip(1.0 - border_touch, 0.0, 1.0))
    return {
        "center_u": float(cx / max(w - 1, 1)),
        "center_v": float(cy / max(h - 1, 1)),
        "axis_u": float(ax_pos / max(w - 1, 1)),
        "axis_v": float(ay_pos / max(h - 1, 1)),
        "axis_pos_u": float(ax_pos / max(w - 1, 1)),
        "axis_pos_v": float(ay_pos / max(h - 1, 1)),
        "axis_neg_u": float(ax_neg / max(w - 1, 1)),
        "axis_neg_v": float(ay_neg / max(h - 1, 1)),
        "axis_dir_x": float(np.cos(axis)),
        "axis_dir_y": float(np.sin(axis)),
        "frame_confidence": confidence,
        "frame_observability": observability,
        "frame_axis_strength": float(np.clip(1.0 - depth_spread, 0.0, 1.0)),
        "frame_completeness": completeness,
        "frame_border_touch": border_touch,
    }


def roi_box_from_action_prior(
    rgb_shape: tuple[int, int, int],
    action_target: np.ndarray,
    gripper_pose: np.ndarray,
    crop_size: int,
) -> tuple[int, int, int, int]:
    h, w = int(rgb_shape[0]), int(rgb_shape[1])
    crop = int(max(8, min(crop_size, min(w, h))))
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    try:
        action_target = np.asarray(action_target, dtype=np.float32).reshape(-1)[:6]
        gripper_pose = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)
        quat = gripper_pose[3:7] if gripper_pose.size >= 7 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        local_delta = world_delta_to_local(action_target, quat).astype(np.float32)
        px_per_meter = 1200.0
        cx = cx + float(np.clip(local_delta[0] * px_per_meter, -0.28 * w, 0.28 * w))
        cy = cy + float(np.clip(-local_delta[1] * px_per_meter, -0.28 * h, 0.28 * h))
    except Exception:
        pass
    half = crop // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x0 = max(0, min(x0, w - crop))
    y0 = max(0, min(y0, h - crop))
    return int(x0), int(y0), int(x0 + crop), int(y0 + crop)


def local_residual_from_action(action_target: np.ndarray, gripper_pose: np.ndarray) -> tuple[float, float, float, float]:
    action_target = np.asarray(action_target, dtype=np.float32).reshape(-1)
    gripper_pose = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)
    current_quat = gripper_pose[3:7] if gripper_pose.size >= 7 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    delta_world = action_target[:6]
    delta_local = world_delta_to_local(delta_world, current_quat).astype(np.float32)
    dx = float(delta_local[0])
    dy = float(delta_local[1])
    dz = float(delta_local[2])
    dyaw = float(delta_local[5])
    return dx, dy, dz, dyaw


def planner_bias_xyyaw(local6d: Iterable[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(list(local6d), dtype=np.float32).reshape(-1)
    arr = np.pad(arr, (0, max(0, 6 - arr.size)))[:6]
    dx = float(arr[0])
    dy = float(arr[1])
    dyaw = float(arr[5])
    xy = float(np.linalg.norm(arr[:2]))
    yaw = float(abs(dyaw))
    score = float(np.linalg.norm(np.array([dx, dy, 0.04 * dyaw], dtype=np.float32)))
    return xy, yaw, dyaw, score


def trace_grasp_error(row: dict[str, Any]) -> dict[str, float]:
    lg = (row.get("local_geometry_error") or {}).get("grasp") or {}
    return {
        "valid": float(lg.get("valid", 0.0)),
        "confidence": float(lg.get("confidence", 0.0)),
        "dx": float(lg.get("dx", 0.0)),
        "dy": float(lg.get("dy", 0.0)),
        "dz": float(lg.get("dz", 0.0)),
        "dyaw": float(lg.get("dyaw", 0.0)),
        "observability": float(lg.get("observability", 0.0)),
        "fit_residual": float(lg.get("fit_residual", 0.0)),
        "inlier_ratio": float(lg.get("inlier_ratio", 0.0)),
        "reason": str(lg.get("reason", "")),
    }


def recovery_phase_label(step_idx: int, close_idx: int, window_before_close: int) -> str:
    rel = int(close_idx) - int(step_idx)
    if rel >= window_before_close - 2:
        return "BIAS"
    if rel >= 5:
        return "REFINE"
    if rel >= 1:
        return "RECOVER"
    return "CLOSE"


def recovery_error_norm(
    dx: float,
    dy: float,
    dyaw: float,
    *,
    yaw_scale: float = 0.04,
) -> float:
    return float(np.linalg.norm(np.asarray([float(dx), float(dy), float(yaw_scale) * float(dyaw)], dtype=np.float32)))


def in_near_grasp_basin(
    dx: float,
    dy: float,
    dyaw: float,
    *,
    xy_threshold: float = 0.015,
    yaw_threshold: float = 0.08,
) -> bool:
    xy = float(np.hypot(float(dx), float(dy)))
    return bool(xy <= float(xy_threshold) and abs(float(dyaw)) <= float(yaw_threshold))


def in_close_ready_basin(
    dx: float,
    dy: float,
    dyaw: float,
    *,
    xy_threshold: float = 0.005,
    yaw_threshold: float = 0.03,
) -> bool:
    xy = float(np.hypot(float(dx), float(dy)))
    return bool(xy <= float(xy_threshold) and abs(float(dyaw)) <= float(yaw_threshold))


def classify_visual_evidence(
    record: dict[str, Any],
    *,
    conf_threshold: float = 1.0e-3,
    observability_threshold: float = 1.0e-3,
    axis_strength_threshold: float = 1.0e-6,
) -> str:
    conf = float(record.get("frame_confidence", 0.0))
    obs = float(record.get("frame_observability", 0.0))
    axis = float(record.get("frame_axis_strength", 0.0))
    if conf >= float(conf_threshold) and obs >= float(observability_threshold) and axis >= float(axis_strength_threshold):
        return "visual_observable"
    return "prior_only"


def apply_closed_loop_recovery_step(
    error_state: Iterable[float],
    planner_prior_state: Iterable[float],
    correction: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    err = np.asarray(list(error_state), dtype=np.float32).reshape(-1)
    prior = np.asarray(list(planner_prior_state), dtype=np.float32).reshape(-1)
    corr = np.asarray(list(correction), dtype=np.float32).reshape(-1)
    err = np.pad(err, (0, max(0, 3 - err.size)))[:3]
    prior = np.pad(prior, (0, max(0, 6 - prior.size)))[:6]
    corr = np.pad(corr, (0, max(0, 3 - corr.size)))[:3]

    post_err = err.copy()
    post_err[0] -= corr[0]
    post_err[1] -= corr[1]
    post_err[2] -= corr[2]

    post_prior = prior.copy()
    post_prior[0] -= corr[0]
    post_prior[1] -= corr[1]
    post_prior[5] -= corr[2]
    return post_err.astype(np.float32), post_prior.astype(np.float32)


def recovery_overshoot_flag(
    pre_error_state: Iterable[float],
    post_error_state: Iterable[float],
    *,
    yaw_scale: float = 0.04,
    eps: float = 1.0e-9,
) -> bool:
    pre = np.asarray(list(pre_error_state), dtype=np.float32).reshape(-1)
    post = np.asarray(list(post_error_state), dtype=np.float32).reshape(-1)
    pre = np.pad(pre, (0, max(0, 3 - pre.size)))[:3]
    post = np.pad(post, (0, max(0, 3 - post.size)))[:3]
    crossed = False
    for idx in range(3):
        if abs(float(pre[idx])) <= eps or abs(float(post[idx])) <= eps:
            continue
        if float(pre[idx]) * float(post[idx]) < 0.0:
            crossed = True
            break
    if not crossed:
        return False
    pre_norm = recovery_error_norm(float(pre[0]), float(pre[1]), float(pre[2]), yaw_scale=yaw_scale)
    post_norm = recovery_error_norm(float(post[0]), float(post[1]), float(post[2]), yaw_scale=yaw_scale)
    return bool(post_norm > pre_norm + eps)


def monotonic_decay_prefix(error_norms: Iterable[float], *, eps: float = 1.0e-9) -> bool:
    vals = [float(v) for v in error_norms]
    if len(vals) <= 1:
        return True
    return all(vals[idx] <= vals[idx - 1] + float(eps) for idx in range(1, len(vals)))


def choose_gated_hybrid_candidate(
    failure_bucket: str,
    *,
    v11_post_error_norm: float,
    v16_post_error_norm: float,
    hard_buckets: Iterable[str] = ("small_xy_small_yaw", "large_xy_small_yaw"),
) -> str:
    hard = {str(item).strip() for item in hard_buckets if str(item).strip()}
    if str(failure_bucket) in hard and float(v16_post_error_norm) + 1.0e-9 < float(v11_post_error_norm):
        return "v16_specialist"
    return "v11_general"
