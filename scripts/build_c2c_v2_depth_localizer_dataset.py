#!/usr/bin/env python3
"""Build a contract-aware depth localizer dataset for Coarse2Contact v2."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2 import load_precision_task_spec
from prismatic.robot.residual_transforms import world_delta_to_local


def _load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_img(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def _binary_mask(mask_img: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask_img)
    if arr.ndim == 2:
        return arr > 0
    return np.any(arr > 0, axis=-1)


def _color_mask(rgb: np.ndarray, color_hint: str | None) -> np.ndarray:
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


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    if mask.size == 0 or not np.any(mask):
        return mask
    labeled, num = ndi.label(mask)
    if num <= 1:
        return mask
    counts = ndi.sum(mask.astype(np.int32), labeled, index=np.arange(1, num + 1))
    keep = int(np.argmax(counts) + 1)
    return labeled == keep


def _principal_axis(mask: np.ndarray) -> float:
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


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.mean(ys))


def _depth_summary(depth: np.ndarray | None, mask: np.ndarray) -> tuple[float, float]:
    if depth is None or mask.size == 0 or not np.any(mask):
        return float("nan"), 0.0
    values = np.asarray(depth, dtype=np.float32)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), 0.0
    p95 = float(np.percentile(values, 95))
    p05 = float(np.percentile(values, 5))
    return float(np.median(values)), float(max(p95 - p05, 0.0))


def _scale_from_depth(depth_m: float, width: int, height: int) -> tuple[float, float]:
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        depth_m = 0.25
    fx = max(width * 1.15, 1.0)
    fy = max(height * 1.15, 1.0)
    return depth_m / fx, depth_m / fy


def _wrap_with_period(angle: float, period: float | None) -> float:
    angle = float(angle)
    if period is None or period <= 0.0:
        return angle
    return float(((angle + 0.5 * period) % period) - 0.5 * period)


def _center_crop_box(width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    crop = int(max(8, min(crop_size, min(width, height))))
    half = crop // 2
    cx = width // 2
    cy = height // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(width, x0 + crop)
    y1 = min(height, y0 + crop)
    if x1 - x0 < crop:
        x0 = max(0, x1 - crop)
    if y1 - y0 < crop:
        y0 = max(0, y1 - crop)
    return int(x0), int(y0), int(x1), int(y1)


def _roi_box_from_action_prior(rgb_shape: tuple[int, int, int], action_target: np.ndarray, gripper_pose: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
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


def _local_residual_from_action(action_target: np.ndarray, gripper_pose: np.ndarray) -> tuple[float, float, float, float]:
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


def _progress_to_close(steps_to_close: int, window: int = 12) -> float:
    if steps_to_close < 1:
        return 1.0
    span = max(int(window) - 1, 1)
    return float(np.clip(1.0 - (float(steps_to_close) - 1.0) / float(span), 0.0, 1.0))


def _frame_keypoints_from_crop(
    rgb: np.ndarray,
    depth: np.ndarray,
    obj_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    *,
    symmetry: float | None = None,
) -> dict[str, float]:
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    crop_rgb = rgb[y0:y1, x0:x1]
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
    cx, cy = _centroid(crop_mask)
    axis = _principal_axis(crop_mask)
    if symmetry and symmetry > 0.0:
        axis = float(((axis + 0.5 * symmetry) % symmetry) - 0.5 * symmetry)
    if np.cos(axis) < 0.0 or (abs(np.cos(axis)) < 1e-6 and np.sin(axis) < 0.0):
        axis = float(axis + np.pi)
    axis_len = 0.30 * float(min(w, h))
    axis_dx = float(np.cos(axis) * axis_len)
    axis_dy = float(np.sin(axis) * axis_len)
    ax_pos = float(np.clip(cx + axis_dx, 0.0, max(w - 1, 0)))
    ay_pos = float(np.clip(cy + axis_dy, 0.0, max(h - 1, 0)))
    ax_neg = float(np.clip(cx - axis_dx, 0.0, max(w - 1, 0)))
    ay_neg = float(np.clip(cy - axis_dy, 0.0, max(h - 1, 0)))
    depth_med, depth_spread = _depth_summary(crop_depth, crop_mask)
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


def _project_frame_to_crop(
    frame_label: dict[str, float],
    crop_box: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))

    def _proj_u(key: str) -> float:
        u_full = float(frame_label.get(key, 0.5))
        px = u_full * float(max(image_width - 1, 1))
        return float((px - float(x0)) / float(max(crop_w - 1, 1)))

    def _proj_v(key: str) -> float:
        v_full = float(frame_label.get(key, 0.5))
        py = v_full * float(max(image_height - 1, 1))
        return float((py - float(y0)) / float(max(crop_h - 1, 1)))

    return {
        "center_u": _proj_u("center_u"),
        "center_v": _proj_v("center_v"),
        "axis_pos_u": _proj_u("axis_pos_u"),
        "axis_pos_v": _proj_v("axis_pos_v"),
        "axis_neg_u": _proj_u("axis_neg_u"),
        "axis_neg_v": _proj_v("axis_neg_v"),
        "axis_dir_x": float(frame_label.get("axis_dir_x", 1.0)),
        "axis_dir_y": float(frame_label.get("axis_dir_y", 0.0)),
        "frame_confidence": float(frame_label.get("frame_confidence", 0.0)),
        "frame_observability": float(frame_label.get("frame_observability", 0.0)),
        "frame_axis_strength": float(frame_label.get("frame_axis_strength", 0.0)),
        "frame_completeness": float(frame_label.get("frame_completeness", 0.0)),
        "frame_border_touch": float(frame_label.get("frame_border_touch", 1.0)),
    }


def _label_from_mask(
    rgb: np.ndarray,
    depth: np.ndarray,
    obj_mask: np.ndarray,
    *,
    symmetry: float | None = None,
    center_xy: tuple[float, float] | None = None,
    yaw_reference: float | None = None,
) -> dict[str, float]:
    obj_mask = _largest_component(np.asarray(obj_mask, dtype=bool))
    if obj_mask.size == 0 or not np.any(obj_mask):
        return {
            "dx": 0.0,
            "dy": 0.0,
            "dyaw": 0.0,
            "confidence": 0.0,
            "observability": 0.0,
            "fit_residual": 1.0,
            "inlier_ratio": 0.0,
        }
    h, w = obj_mask.shape
    cx, cy = _centroid(obj_mask)
    ref_cx, ref_cy = center_xy if center_xy is not None else (0.5 * (w - 1), 0.5 * (h - 1))
    depth_med, depth_spread = _depth_summary(depth, obj_mask)
    sx, sy = _scale_from_depth(depth_med, w, h)
    dx = (cx - ref_cx) * sx
    dy = (cy - ref_cy) * sy
    yaw = _principal_axis(obj_mask)
    if symmetry and symmetry > 0.0:
        yaw = _wrap_with_period(yaw, symmetry)
    if yaw_reference is not None:
        yaw = yaw - float(yaw_reference)
        if symmetry and symmetry > 0.0:
            yaw = _wrap_with_period(yaw, symmetry)
    observability = float(np.count_nonzero(obj_mask) / max(obj_mask.size, 1))
    confidence = float(np.clip(0.15 + 3.0 * observability - 0.2 * depth_spread, 0.0, 1.0))
    return {
        "dx": float(dx),
        "dy": float(dy),
        "dyaw": float(yaw),
        "confidence": confidence,
        "observability": observability,
        "fit_residual": float(depth_spread),
        "inlier_ratio": float(np.clip(1.0 - depth_spread, 0.0, 1.0)),
    }


def _label_from_crop(
    rgb: np.ndarray,
    depth: np.ndarray,
    obj_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    *,
    symmetry: float | None = None,
) -> dict[str, float]:
    x0, y0, x1, y1 = crop_box
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_depth = depth[y0:y1, x0:x1]
    crop_mask = obj_mask[y0:y1, x0:x1]
    if crop_mask.size == 0:
        return _label_from_mask(crop_rgb, crop_depth, crop_mask, symmetry=symmetry)
    ref_cx = 0.5 * (crop_mask.shape[1] - 1)
    ref_cy = 0.5 * (crop_mask.shape[0] - 1)
    return _label_from_mask(crop_rgb, crop_depth, crop_mask, symmetry=symmetry, center_xy=(ref_cx, ref_cy))


def _select_view(ep_dir: Path, step_idx: int, color_hint: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    views = ["wrist", "front"]
    candidates = []
    for view in views:
        rgb_path = ep_dir / f"{view}_rgb" / f"{step_idx}.png"
        depth_path = ep_dir / f"{view}_depth" / f"{step_idx}.png"
        mask_path = ep_dir / f"{view}_mask" / f"{step_idx}.png"
        if not (rgb_path.exists() and depth_path.exists() and mask_path.exists()):
            continue
        rgb = _load_img(rgb_path)
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        mask = _binary_mask(_load_img(mask_path))
        color = _color_mask(rgb, color_hint)
        obj_mask = _largest_component(mask & color)
        label = _label_from_mask(rgb, depth, obj_mask, symmetry=np.pi / 2.0 if color_hint == "blue" else None)
        candidates.append((view, rgb, depth, obj_mask, label, rgb_path, depth_path, mask_path))
    if not candidates:
        return "", np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool)
    candidates.sort(key=lambda item: (item[4]["confidence"], item[4]["observability"]), reverse=True)
    view, rgb, depth, mask, label, *_ = candidates[0]
    return view, rgb, depth, mask


def _best_candidate(
    ep_dir: Path,
    step_idx: int,
    color_hint: str,
    symmetry: float | None,
    *,
    allowed_views: tuple[str, ...] = ("wrist", "front"),
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, float], Path, Path, Path]:
    candidates = []
    for view in allowed_views:
        rgb_path = ep_dir / f"{view}_rgb" / f"{step_idx}.png"
        depth_path = ep_dir / f"{view}_depth" / f"{step_idx}.png"
        mask_path = ep_dir / f"{view}_mask" / f"{step_idx}.png"
        if not (rgb_path.exists() and depth_path.exists() and mask_path.exists()):
            continue
        rgb = _load_img(rgb_path)
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        if float(depth.max()) > 1.5:
            depth = depth / 255.0
        mask = _binary_mask(_load_img(mask_path))
        color = _color_mask(rgb, color_hint)
        obj_mask = _largest_component(mask & color)
        label = _label_from_mask(rgb, depth, obj_mask, symmetry=symmetry)
        candidates.append((view, rgb, depth, obj_mask, label, rgb_path, depth_path, mask_path))
    if not candidates:
        return "", np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool), {"confidence": 0.0, "observability": 0.0, "fit_residual": 1.0, "inlier_ratio": 0.0, "dx": 0.0, "dy": 0.0, "dyaw": 0.0}, Path(), Path(), Path()
    candidates.sort(key=lambda item: (item[4]["confidence"], item[4]["observability"]), reverse=True)
    return candidates[0]


def _episode_dirs(root: Path) -> list[Path]:
    episodes_root = root / "train" / "episodes"
    if not episodes_root.exists():
        return []
    def _episode_idx(p: Path) -> int:
        try:
            return int(p.name.replace("episode", ""))
        except Exception:
            return -1
    return sorted([p for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode")], key=_episode_idx)


def _estimate_wrist_grasp_reference(episode_dirs: list[Path]) -> dict[str, float]:
    centers_u: list[float] = []
    centers_v: list[float] = []
    axis_angles: list[float] = []
    observabilities: list[float] = []
    for ep_dir in episode_dirs:
        inputs_path = ep_dir / "model_inputs.npz"
        if not inputs_path.exists():
            continue
        model_inputs = np.load(inputs_path, allow_pickle=True)
        close_idx = _first_close_index(model_inputs["gripper_open"])
        if close_idx is None:
            continue
        for dt in (1, 2):
            t = int(close_idx) - dt
            if t < 0:
                continue
            rgb_path = ep_dir / "wrist_rgb" / f"{t}.png"
            depth_path = ep_dir / "wrist_depth" / f"{t}.png"
            mask_path = ep_dir / "wrist_mask" / f"{t}.png"
            if not (rgb_path.exists() and depth_path.exists() and mask_path.exists()):
                continue
            rgb = _load_img(rgb_path)
            depth = np.asarray(Image.open(depth_path), dtype=np.float32)
            if float(depth.max()) > 1.5:
                depth = depth / 255.0
            mask = _binary_mask(_load_img(mask_path))
            color = _color_mask(rgb, "blue")
            obj_mask = _largest_component(mask & color)
            h, w = rgb.shape[:2]
            frame = _frame_keypoints_from_crop(rgb, depth, obj_mask, (0, 0, w, h), symmetry=np.pi / 2.0)
            if float(frame["frame_observability"]) < 0.03:
                continue
            centers_u.append(float(frame["center_u"]))
            centers_v.append(float(frame["center_v"]))
            axis_angle = float(np.arctan2(float(frame["axis_dir_y"]), float(frame["axis_dir_x"])))
            axis_angles.append(_wrap_with_period(axis_angle, np.pi / 2.0))
            observabilities.append(float(frame["frame_observability"]))
    if not centers_u:
        return {
            "jaw_center_u": 0.5,
            "jaw_center_v": 0.5,
            "jaw_axis_angle": 0.0,
            "num_reference_frames": 0,
            "observability_median": 0.0,
        }
    return {
        "jaw_center_u": float(np.median(np.asarray(centers_u, dtype=np.float32))),
        "jaw_center_v": float(np.median(np.asarray(centers_v, dtype=np.float32))),
        "jaw_axis_angle": float(np.median(np.asarray(axis_angles, dtype=np.float32))),
        "num_reference_frames": int(len(centers_u)),
        "observability_median": float(np.median(np.asarray(observabilities, dtype=np.float32))),
    }


def _phase_name(phase_annotation: dict, phase_id: int) -> str:
    mapping = phase_annotation.get("phase_id_to_name", {})
    return str(mapping.get(str(int(phase_id)), mapping.get(int(phase_id), f"phase_{phase_id}")))


def _first_close_index(gripper_open: np.ndarray) -> int | None:
    opened = np.asarray(gripper_open, dtype=np.float32).reshape(-1)
    if opened.size == 0:
        return None
    for idx in range(1, opened.size):
        if opened[idx - 1] > 0.5 and opened[idx] <= 0.5:
            return int(idx)
    return None


def _is_near_grasp_basin(
    *,
    view_name: str,
    steps_to_close: int,
    observability: float,
    frame_confidence: float,
    frame_completeness: float,
    geom_dx: float,
    geom_dy: float,
    geom_dyaw: float,
    skill_xy_tolerance: float,
    skill_yaw_tolerance: float,
) -> bool:
    if view_name != "wrist":
        return False
    if steps_to_close < 1 or steps_to_close > 8:
        return False
    if observability < 0.02:
        return False
    if frame_confidence < 0.20:
        return False
    xy_gate = max(float(skill_xy_tolerance) * 2.0, 0.0025)
    yaw_gate = max(float(skill_yaw_tolerance), 0.10)
    return bool(abs(geom_dx) <= xy_gate and abs(geom_dy) <= xy_gate and abs(geom_dyaw) <= yaw_gate)


def _yaw_observability_labels(
    *,
    xyyaw_mask: bool,
    full_frame_label: dict | None,
    geom_dyaw: float,
) -> tuple[bool, bool, bool, float]:
    near_zero = bool(abs(float(geom_dyaw)) < 0.01)
    if not xyyaw_mask:
        return False, False, near_zero, 0.0
    if full_frame_label is None:
        return False, True, near_zero, 0.0
    axis_strength = float(full_frame_label.get("frame_axis_strength", 0.0))
    observability = float(full_frame_label.get("frame_observability", 0.0))
    score = float(
        np.clip(
            0.65 * np.clip(axis_strength / 0.18, 0.0, 1.0)
            + 0.35 * np.clip((observability - 0.10) / 0.03, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    observable = bool(
        axis_strength >= 0.10
        and observability >= 0.11
        and not near_zero
    )
    low_observability = bool(not observable)
    return observable, low_observability, near_zero, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_root", type=Path, default=Path("data/rlbench_data/insert_onto_square_peg"))
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets"))
    ap.add_argument("--sample_stride", type=int, default=2)
    ap.add_argument("--negative_stride", type=int, default=4)
    ap.add_argument("--max_samples_per_episode", type=int, default=0)
    ap.add_argument("--max_episodes", type=int, default=0)
    args = ap.parse_args()

    root = args.task_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "depth_localizer_dataset_v2.jsonl"
    spec = load_precision_task_spec("insert_onto_square_peg")
    if spec is None:
        raise RuntimeError("Missing task spec for insert_onto_square_peg")

    grasp_skill = spec.get_skill("precision_grasp_ring")
    spoke_skill = spec.get_skill("precision_align_ring_to_spoke")
    samples = []
    ring_frame_samples = []
    grasp_skill_samples = []
    summary = {
        "task_name": "insert_onto_square_peg",
        "task_root": str(root),
        "output_path": str(out_path),
        "ring_frame_output_path": str(out_root / "ring_frame_dataset_v1.jsonl"),
        "grasp_skill_output_path": str(out_root / "grasp_skill_dataset_v1.jsonl"),
        "positive_counts": Counter(),
        "negative_counts": Counter(),
        "ring_frame_counts": Counter(),
        "grasp_skill_counts": Counter(),
        "view_counts": Counter(),
        "label_sources": Counter(),
        "episodes": [],
    }

    episode_dirs = _episode_dirs(root)
    if args.max_episodes and int(args.max_episodes) > 0:
        episode_dirs = episode_dirs[: int(args.max_episodes)]
    grasp_reference = _estimate_wrist_grasp_reference(episode_dirs)
    summary["grasp_reference"] = dict(grasp_reference)

    for ep_dir in episode_dirs:
        ep_idx = int(ep_dir.name.replace("episode", ""))
        phase_path = ep_dir / "phase_annotation.json"
        phase_ids_path = ep_dir / "phase_ids.npy"
        inputs_path = ep_dir / "model_inputs.npz"
        desc_path = ep_dir / "variation_descriptions.pkl"
        if not (phase_path.exists() and phase_ids_path.exists() and inputs_path.exists() and desc_path.exists()):
            continue
        phase_annotation = json.loads(phase_path.read_text(encoding="utf-8"))
        phase_ids = np.load(phase_ids_path)
        model_inputs = np.load(inputs_path, allow_pickle=True)
        action_targets = model_inputs["action_targets"]
        gripper_pose = model_inputs["gripper_pose"]
        gripper_open = model_inputs["gripper_open"]
        proprio = model_inputs["proprio"]
        descriptions = _load_pickle(desc_path)
        close_idx = _first_close_index(gripper_open)
        grasp_skill_window = set()
        if close_idx is not None:
            lo = max(0, int(close_idx) - 12)
            grasp_skill_window = set(range(lo, int(close_idx)))
        phase_counts = Counter()
        episode_pos = Counter()
        episode_neg = Counter()
        episode_sample_count = 0

        for t in range(0, int(len(phase_ids)), max(int(args.sample_stride), 1)):
            phase_id = int(phase_ids[t])
            phase_name = _phase_name(phase_annotation, phase_id)
            phase_counts[phase_name] += 1
            for skill_name, skill, target_phase_ids, color_hint, symmetry in [
                ("precision_grasp_ring", grasp_skill, {0, 1}, "blue", np.pi / 2.0),
                ("precision_align_ring_to_spoke", spoke_skill, {2, 3}, "red", None),
            ]:
                stage_name = "RING_GRASP_ALIGN" if skill_name == "precision_grasp_ring" else "RING_SPOKE_ALIGN"
                is_positive_stage = phase_id in target_phase_ids
                if not is_positive_stage and (t % max(int(args.negative_stride), 1) != 0):
                    continue
                allowed_views = ("wrist",) if skill_name == "precision_grasp_ring" else ("wrist", "front")
                view, rgb, depth, mask, label, rgb_path, depth_path, mask_path = _best_candidate(
                    ep_dir,
                    t,
                    color_hint,
                    symmetry,
                    allowed_views=allowed_views,
                )
                if not view:
                    continue
                label_source = "privileged_mask"
                visible = bool(label["observability"] >= 0.03)
                valid = bool(visible and is_positive_stage)
                if not valid:
                    label_source = "wrong_stage_negative" if not is_positive_stage else "low_observability_negative"
                else:
                    label_source = "privileged_grasp_geometry" if skill_name == "precision_grasp_ring" else "privileged_mask"
                roi_size_px = int(getattr(skill, "roi_size_px", 96) or 96)
                roi_resize_px = int(getattr(skill, "roi_resize_px", 128) or 128)
                heatmap_xy_range_m = float(getattr(skill, "heatmap_xy_range_m", 0.040) or 0.040)
                heatmap_size = int(getattr(skill, "heatmap_size", 16) or 16)
                heatmap_sigma_px = float(getattr(skill, "heatmap_sigma_px", 1.5) or 1.5)
                heatmap_pos_weight = float(getattr(skill, "heatmap_pos_weight", 8.0) or 8.0)
                h, w = rgb.shape[:2]
                roi_box = _roi_box_from_action_prior(rgb.shape, action_targets[t], gripper_pose[t], roi_size_px)
                crop_label = _label_from_crop(rgb, depth, mask, roi_box, symmetry=symmetry)
                frame_label = _frame_keypoints_from_crop(rgb, depth, mask, roi_box, symmetry=symmetry)
                grasp_geom_label = None
                full_frame_label = None
                full_frame_projected = None
                if skill_name == "precision_grasp_ring":
                    h_full, w_full = rgb.shape[:2]
                    full_frame_label = _frame_keypoints_from_crop(rgb, depth, mask, (0, 0, w_full, h_full), symmetry=symmetry)
                    full_frame_projected = _project_frame_to_crop(full_frame_label, roi_box, image_width=w_full, image_height=h_full)
                    grasp_geom_label = _label_from_mask(
                        rgb,
                        depth,
                        mask,
                        symmetry=symmetry,
                        center_xy=(
                            float(grasp_reference["jaw_center_u"]) * float(max(w_full - 1, 1)),
                            float(grasp_reference["jaw_center_v"]) * float(max(h_full - 1, 1)),
                        ),
                        yaw_reference=float(grasp_reference["jaw_axis_angle"]),
                    )
                label_dx = label_dy = label_dz = label_dyaw = 0.0
                if valid:
                    if grasp_geom_label is not None:
                        label_dx = float(grasp_geom_label["dx"])
                        label_dy = float(grasp_geom_label["dy"])
                        label_dyaw = float(grasp_geom_label["dyaw"])
                    else:
                        label_dx = float(crop_label["dx"])
                        label_dy = float(crop_label["dy"])
                        label_dyaw = float(crop_label["dyaw"])
                    label_dz = float(_local_residual_from_action(action_targets[t], gripper_pose[t])[2])
                sample = {
                    "task_name": "insert_onto_square_peg",
                    "episode_idx": ep_idx,
                    "step_idx": int(t),
                    "phase_id": phase_id,
                    "phase_name": phase_name,
                    "stage_name": stage_name,
                    "skill_name": skill_name,
                    "skill_type": skill.skill_type,
                    "target_entity": skill.target_entity,
                    "reference_entity": skill.reference_entity,
                    "controlled_dofs": list(skill.controlled_dofs),
                    "view_name": view,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "mask_path": str(mask_path),
                    "roi_box": list(roi_box),
                    "roi_size_px": int(roi_size_px),
                    "roi_resize_px": int(roi_resize_px),
                    "heatmap_xy_range_m": float(heatmap_xy_range_m),
                    "heatmap_size": int(heatmap_size),
                    "heatmap_sigma_px": float(heatmap_sigma_px),
                    "heatmap_pos_weight": float(heatmap_pos_weight),
                    "gripper_pose": np.asarray(gripper_pose[t], dtype=np.float32).tolist(),
                    "proprio": np.asarray(proprio[t], dtype=np.float32).tolist(),
                    "planner_prior_delta": None,
                    "action_target": np.asarray(action_targets[t], dtype=np.float32).tolist(),
                    "label_dx": float(label_dx if valid else 0.0),
                    "label_dy": float(label_dy if valid else 0.0),
                    "label_dz": float(label_dz if valid else 0.0),
                    "label_dyaw": float(label_dyaw if valid else 0.0),
                    "label_confidence": float(label["confidence"] if valid else 0.0),
                    "frame_center_u": float(frame_label["center_u"] if valid else 0.5),
                    "frame_center_v": float(frame_label["center_v"] if valid else 0.5),
                    "frame_axis_u": float(frame_label["axis_u"] if valid else 0.5),
                    "frame_axis_v": float(frame_label["axis_v"] if valid else 0.5),
                    "frame_axis_pos_u": float(frame_label["axis_pos_u"] if valid else 0.5),
                    "frame_axis_pos_v": float(frame_label["axis_pos_v"] if valid else 0.5),
                    "frame_axis_neg_u": float(frame_label["axis_neg_u"] if valid else 0.5),
                    "frame_axis_neg_v": float(frame_label["axis_neg_v"] if valid else 0.5),
                    "frame_axis_dir_x": float(frame_label["axis_dir_x"] if valid else 1.0),
                    "frame_axis_dir_y": float(frame_label["axis_dir_y"] if valid else 0.0),
                    "frame_confidence": float(frame_label["frame_confidence"] if valid else 0.0),
                    "frame_completeness": float(frame_label["frame_completeness"] if valid else 0.0),
                    "frame_border_touch": float(frame_label["frame_border_touch"] if valid else 1.0),
                    "label_source": label_source,
                    "label_observability": float(label["observability"]),
                    "label_fit_residual": float(label["fit_residual"]),
                    "label_inlier_ratio": float(label["inlier_ratio"]),
                    "metric_scale": "approx_depth_scale",
                    "uses_privileged_label": bool(valid),
                    "uses_privileged_runtime": False,
                    "sample_kind": "positive" if valid else "negative",
                    "language_targets": list(phase_annotation.get("language_targets", [])),
                }
                samples.append(sample)
                episode_sample_count += 1
                if skill_name == "precision_grasp_ring":
                    ring_frame_sample = dict(sample)
                    ring_frame_valid = bool(
                        view == "wrist"
                        and label["observability"] >= 0.03
                        and float(frame_label["frame_completeness"]) >= 0.5
                    )
                    ring_frame_sample.update(
                        {
                            "dataset_role": "ring_frame_localizer",
                            "stage_name": "RING_FRAME_OBSERVATION",
                            "skill_name": "ring_frame_localizer",
                            "label_dx": 0.0,
                            "label_dy": 0.0,
                            "label_dz": 0.0,
                            "label_dyaw": 0.0,
                            "label_confidence": float(label["confidence"] if ring_frame_valid else 0.0),
                            "frame_center_u": float(frame_label["center_u"] if ring_frame_valid else 0.5),
                            "frame_center_v": float(frame_label["center_v"] if ring_frame_valid else 0.5),
                            "frame_axis_pos_u": float(frame_label["axis_pos_u"] if ring_frame_valid else 0.5),
                            "frame_axis_pos_v": float(frame_label["axis_pos_v"] if ring_frame_valid else 0.5),
                            "frame_axis_neg_u": float(frame_label["axis_neg_u"] if ring_frame_valid else 0.5),
                            "frame_axis_neg_v": float(frame_label["axis_neg_v"] if ring_frame_valid else 0.5),
                            "frame_axis_dir_x": float(frame_label["axis_dir_x"] if ring_frame_valid else 1.0),
                            "frame_axis_dir_y": float(frame_label["axis_dir_y"] if ring_frame_valid else 0.0),
                            "frame_completeness": float(frame_label["frame_completeness"] if ring_frame_valid else 0.0),
                            "frame_border_touch": float(frame_label["frame_border_touch"] if ring_frame_valid else 1.0),
                            "label_source": "privileged_mask_ring_frame" if ring_frame_valid else "ring_not_visible_negative",
                            "sample_kind": "positive" if ring_frame_valid else "negative",
                            "uses_privileged_label": bool(ring_frame_valid),
                        }
                    )
                    ring_frame_samples.append(ring_frame_sample)
                    summary["ring_frame_counts"][ring_frame_sample["sample_kind"]] += 1

                    if t in grasp_skill_window:
                        local_dx, local_dy, local_dz, local_dyaw = _local_residual_from_action(action_targets[t], gripper_pose[t])
                        steps_to_close = int(int(close_idx) - int(t)) if close_idx is not None else -1
                        ready_visible = bool(view == "wrist" and label["observability"] >= 0.03)
                        geom_src = grasp_geom_label if grasp_geom_label is not None else label
                        geom_dx = float(geom_src["dx"])
                        geom_dy = float(geom_src["dy"])
                        geom_dyaw = float(geom_src["dyaw"])
                        descend_amount = float(max(local_dz, 0.0))
                        progress_to_close = _progress_to_close(steps_to_close, window=12)
                        near_grasp_basin = _is_near_grasp_basin(
                            view_name=view,
                            steps_to_close=steps_to_close,
                            observability=float(label["observability"]),
                            frame_confidence=float(frame_label["frame_confidence"]),
                            frame_completeness=float(frame_label["frame_completeness"]),
                            geom_dx=geom_dx,
                            geom_dy=geom_dy,
                            geom_dyaw=geom_dyaw,
                            skill_xy_tolerance=float(grasp_skill.xy_tolerance),
                            skill_yaw_tolerance=float(grasp_skill.yaw_tolerance),
                        )
                        xyyaw_mask = bool(near_grasp_basin and steps_to_close >= 1 and steps_to_close <= 8)
                        yaw_observable_mask, yaw_low_observability, yaw_near_zero, yaw_observability_score = _yaw_observability_labels(
                            xyyaw_mask=xyyaw_mask,
                            full_frame_label=full_frame_label,
                            geom_dyaw=geom_dyaw,
                        )
                        strong_yaw_mask = bool(
                            yaw_observable_mask
                            and full_frame_label is not None
                            and float(full_frame_label["frame_axis_strength"]) >= 0.12
                            and float(full_frame_label["frame_observability"]) >= 0.112
                            and abs(float(geom_dyaw)) >= 0.025
                        )
                        z_mask = bool(ready_visible and steps_to_close >= 1 and steps_to_close <= 12)
                        ready_mask = bool(ready_visible and steps_to_close >= 1 and steps_to_close <= 6)
                        ready_label = float(ready_mask and steps_to_close <= 2)
                        grasp_skill_sample = dict(ring_frame_sample)
                        grasp_skill_sample.update(
                            {
                                "dataset_role": "grasp_skill_head",
                                "stage_name": "RING_GRASP_PRE_CLOSE",
                                "skill_name": "grasp_skill_head",
                                "controlled_dofs": ["x", "y", "z", "yaw"],
                                "frame_center_u": float(frame_label["center_u"]),
                                "frame_center_v": float(frame_label["center_v"]),
                                "frame_axis_pos_u": float(frame_label["axis_pos_u"]),
                                "frame_axis_pos_v": float(frame_label["axis_pos_v"]),
                                "frame_axis_neg_u": float(frame_label["axis_neg_u"]),
                                "frame_axis_neg_v": float(frame_label["axis_neg_v"]),
                                "frame_axis_dir_x": float(frame_label["axis_dir_x"]),
                                "frame_axis_dir_y": float(frame_label["axis_dir_y"]),
                                "frame_confidence": float(frame_label["frame_confidence"]),
                                "frame_completeness": float(frame_label["frame_completeness"]),
                                "frame_border_touch": float(frame_label["frame_border_touch"]),
                                "priv_frame_center_u": float(full_frame_projected["center_u"]) if full_frame_projected is not None else 0.5,
                                "priv_frame_center_v": float(full_frame_projected["center_v"]) if full_frame_projected is not None else 0.5,
                                "priv_frame_axis_pos_u": float(full_frame_projected["axis_pos_u"]) if full_frame_projected is not None else 0.5,
                                "priv_frame_axis_pos_v": float(full_frame_projected["axis_pos_v"]) if full_frame_projected is not None else 0.5,
                                "priv_frame_axis_neg_u": float(full_frame_projected["axis_neg_u"]) if full_frame_projected is not None else 0.5,
                                "priv_frame_axis_neg_v": float(full_frame_projected["axis_neg_v"]) if full_frame_projected is not None else 0.5,
                                "priv_frame_axis_dir_x": float(full_frame_projected["axis_dir_x"]) if full_frame_projected is not None else 1.0,
                                "priv_frame_axis_dir_y": float(full_frame_projected["axis_dir_y"]) if full_frame_projected is not None else 0.0,
                                "priv_frame_confidence": float(full_frame_projected["frame_confidence"]) if full_frame_projected is not None else 0.0,
                                "priv_frame_observability": float(full_frame_projected["frame_observability"]) if full_frame_projected is not None else 0.0,
                                "priv_frame_axis_strength": float(full_frame_projected["frame_axis_strength"]) if full_frame_projected is not None else 0.0,
                                "priv_frame_completeness": float(full_frame_projected["frame_completeness"]) if full_frame_projected is not None else 0.0,
                                "label_dx": geom_dx,
                                "label_dy": geom_dy,
                                "label_dz": descend_amount,
                                "label_dyaw": geom_dyaw,
                                "label_descend_amount": descend_amount,
                                "label_progress_to_close": progress_to_close,
                                "label_confidence": 1.0,
                                "ready_to_close": ready_label,
                                "steps_to_close": steps_to_close,
                                "near_grasp_basin": bool(near_grasp_basin),
                                "xyyaw_supervision_mask": float(xyyaw_mask),
                                "yaw_observable_target": float(yaw_observable_mask),
                                "yaw_low_observability": float(yaw_low_observability),
                                "yaw_near_zero": float(yaw_near_zero),
                                "yaw_observability_score": float(yaw_observability_score),
                                "yaw_strong_supervision_mask": float(strong_yaw_mask),
                                "z_supervision_mask": float(z_mask),
                                "ready_supervision_mask": float(ready_mask),
                                "geom_label_dx": geom_dx,
                                "geom_label_dy": geom_dy,
                                "geom_label_dyaw": geom_dyaw,
                                "crop_geom_dx": float(crop_label["dx"]),
                                "crop_geom_dy": float(crop_label["dy"]),
                                "crop_geom_dyaw": float(crop_label["dyaw"]),
                                "expert_action_dx": float(local_dx),
                                "expert_action_dy": float(local_dy),
                                "expert_action_dyaw": float(local_dyaw),
                                "jaw_reference_u": float(grasp_reference["jaw_center_u"]),
                                "jaw_reference_v": float(grasp_reference["jaw_center_v"]),
                                "jaw_reference_yaw": float(grasp_reference["jaw_axis_angle"]),
                                "label_source": "privileged_grasp_geometry",
                                "sample_kind": "positive",
                                "uses_privileged_label": True,
                            }
                        )
                        grasp_skill_samples.append(grasp_skill_sample)
                        summary["grasp_skill_counts"]["positive"] += 1
                summary["view_counts"][view] += 1
                summary["label_sources"][label_source] += 1
                if valid:
                    summary["positive_counts"][skill_name] += 1
                    episode_pos[skill_name] += 1
                else:
                    summary["negative_counts"][skill_name] += 1
                    episode_neg[skill_name] += 1
                if args.max_samples_per_episode and episode_sample_count >= args.max_samples_per_episode:
                    break
            if args.max_samples_per_episode and episode_sample_count >= args.max_samples_per_episode:
                break

        summary["episodes"].append(
            {
                "episode_idx": ep_idx,
                "num_steps": int(len(phase_ids)),
                "close_idx": None if close_idx is None else int(close_idx),
                "grasp_skill_window": [min(grasp_skill_window), max(grasp_skill_window)] if grasp_skill_window else [],
                "phase_counts": dict(phase_counts),
                "positive_counts": dict(episode_pos),
                "negative_counts": dict(episode_neg),
                "descriptions": list(descriptions) if isinstance(descriptions, (list, tuple)) else [str(descriptions)],
            }
        )

    with open(out_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    ring_frame_path = out_root / "ring_frame_dataset_v1.jsonl"
    with open(ring_frame_path, "w", encoding="utf-8") as handle:
        for sample in ring_frame_samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
    grasp_skill_path = out_root / "grasp_skill_dataset_v1.jsonl"
    with open(grasp_skill_path, "w", encoding="utf-8") as handle:
        for sample in grasp_skill_samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")

    summary["positive_counts"] = dict(summary["positive_counts"])
    summary["negative_counts"] = dict(summary["negative_counts"])
    summary["ring_frame_counts"] = dict(summary["ring_frame_counts"])
    summary["grasp_skill_counts"] = dict(summary["grasp_skill_counts"])
    summary["view_counts"] = dict(summary["view_counts"])
    summary["label_sources"] = dict(summary["label_sources"])
    summary["num_samples"] = len(samples)
    summary["num_ring_frame_samples"] = len(ring_frame_samples)
    summary["num_grasp_skill_samples"] = len(grasp_skill_samples)
    summary_path = out_root / "depth_localizer_dataset_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    print(ring_frame_path)
    print(grasp_skill_path)
    print(summary_path)


if __name__ == "__main__":
    main()
