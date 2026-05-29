"""Contract-aware local geometry estimators for Coarse2Contact v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy import ndimage as ndi

from .specs import EntitySpec, PrecisionSkillSpec, PrecisionTaskSpec


def _obs_get(observation: Any, key: str, default: Any = None) -> Any:
    if observation is None:
        return default
    if isinstance(observation, Mapping):
        return observation.get(key, default)
    return getattr(observation, key, default)


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_rgb_array(image: Any) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    return arr.astype(np.uint8, copy=False)


def _as_depth_array(depth: Any) -> Optional[np.ndarray]:
    if depth is None:
        return None
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _maybe_center_crop(mask: np.ndarray, center_fraction: float = 0.68) -> np.ndarray:
    if mask.size == 0:
        return mask
    h, w = mask.shape
    if h == 0 or w == 0:
        return mask
    frac = float(np.clip(center_fraction, 0.2, 1.0))
    half_h = max(1, int(0.5 * h * frac))
    half_w = max(1, int(0.5 * w * frac))
    cy = h // 2
    cx = w // 2
    crop = np.zeros_like(mask, dtype=bool)
    crop[max(0, cy - half_h) : min(h, cy + half_h), max(0, cx - half_w) : min(w, cx + half_w)] = True
    return mask & crop


def _color_mask(rgb: np.ndarray, color_hint: Optional[str]) -> np.ndarray:
    if rgb is None:
        return np.zeros((0, 0), dtype=bool)
    rgb = np.asarray(rgb, dtype=np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    hint = (color_hint or "").lower()
    if hint in {"red", "crimson", "ruby"}:
        return (r > 90.0) & (r > 1.25 * g) & (r > 1.15 * b)
    if hint in {"blue", "cyan", "azure"}:
        return (b > 90.0) & (b > 1.15 * r) & (b > 1.10 * g)
    if hint in {"green", "emerald"}:
        return (g > 90.0) & (g > 1.15 * r) & (g > 1.10 * b)
    if hint in {"yellow", "amber"}:
        return (r > 110.0) & (g > 110.0) & (b < 170.0) & ((r + g) > 1.5 * b)
    return np.ones(rgb.shape[:2], dtype=bool)


def _largest_component(mask: np.ndarray) -> np.ndarray:
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


def _depth_summary(depth: Optional[np.ndarray], mask: np.ndarray) -> tuple[float, float]:
    if depth is None or mask.size == 0 or not np.any(mask):
        return float("nan"), 0.0
    values = depth[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), 0.0
    p95 = float(np.percentile(values, 95))
    p05 = float(np.percentile(values, 5))
    return float(np.median(values)), float(max(p95 - p05, 0.0))


def _entity_frame(entity: Optional[EntitySpec]) -> str:
    if entity is None:
        return ""
    return str(entity.observable_hints.get("frame_type", entity.primitive or entity.name))


def _entity_confidence_from_hints(entity: Optional[EntitySpec]) -> float:
    if entity is None:
        return 0.0
    score = 0.0
    if entity.color_hint:
        score += 0.15
    if entity.primitive:
        score += 0.10
    if entity.rgb_hint is not None:
        score += 0.10
    if entity.observable_hints:
        score += 0.10
    return float(min(score, 0.5))


@dataclass(frozen=True)
class LocalGeometryError:
    valid: bool
    confidence: float
    dx: float
    dy: float
    dz: float
    dyaw: float
    observability: float
    fit_residual: float
    inlier_ratio: float
    reason: str
    target_entity: str = ""
    reference_entity: str = ""
    stage_name: str = ""
    yaw_valid: bool = True
    yaw_reason: str = ""
    image_axis_yaw: float = 0.0


class _BaseLocalizer:
    def __init__(self, *, shadow_only: bool = True) -> None:
        self.shadow_only = bool(shadow_only)

    @staticmethod
    def _scale_from_depth(depth_m: float, width: int, height: int) -> tuple[float, float]:
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            depth_m = 0.25
        fx = max(width * 1.15, 1.0)
        fy = max(height * 1.15, 1.0)
        return depth_m / fx, depth_m / fy

    @staticmethod
    def _error_from_mask(mask: np.ndarray, depth: Optional[np.ndarray]) -> tuple[float, float, float, float, float]:
        if mask.size == 0 or not np.any(mask):
            return float("nan"), float("nan"), float("nan"), 0.0, 0.0
        mask = _largest_component(mask)
        cx, cy = _centroid(mask)
        angle = _principal_axis(mask)
        depth_med, depth_spread = _depth_summary(depth, mask)
        h, w = mask.shape
        sx, sy = _BaseLocalizer._scale_from_depth(depth_med, w, h)
        dx = (cx - 0.5 * (w - 1)) * sx
        dy = (cy - 0.5 * (h - 1)) * sy
        obs = float(np.count_nonzero(mask) / max(mask.size, 1))
        fit_residual = float(depth_spread)
        return dx, dy, angle, obs, fit_residual


class RingGraspLocalizer(_BaseLocalizer):
    """Estimate ring grasp alignment relative to the gripper centerline."""

    def localize(
        self,
        observation: Any,
        robot_state: Any,
        task_spec: PrecisionTaskSpec,
        skill_spec: PrecisionSkillSpec,
        *,
        stage_name: str = "",
    ) -> LocalGeometryError:
        target = task_spec.get_entity(skill_spec.target_entity)
        rgb = _first_non_none(_as_rgb_array(_obs_get(observation, "wrist_rgb")), _as_rgb_array(_obs_get(observation, "front_rgb")))
        depth = _first_non_none(_as_depth_array(_obs_get(observation, "wrist_depth")), _as_depth_array(_obs_get(observation, "front_depth")))
        if rgb is None or depth is None:
            return LocalGeometryError(
                False,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                "missing_rgbd",
                target.name,
                skill_spec.reference_entity,
                stage_name,
                yaw_valid=False,
                yaw_reason="missing_rgbd",
            )

        mask = _color_mask(rgb, target.color_hint)
        mask = _maybe_center_crop(mask, center_fraction=0.74)
        mask = ndi.binary_opening(mask, iterations=1)
        mask = ndi.binary_closing(mask, iterations=1)
        dx, dy, image_axis_yaw, observability, fit_residual = self._error_from_mask(mask, depth)
        if not np.isfinite(dx) or observability <= 0.0005:
            return LocalGeometryError(
                False,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                observability,
                fit_residual,
                0.0,
                "weak_ring_mask",
                target.name,
                skill_spec.reference_entity,
                stage_name,
                yaw_valid=False,
                yaw_reason="weak_ring_mask",
                image_axis_yaw=float(image_axis_yaw) if np.isfinite(image_axis_yaw) else 0.0,
            )

        depth_med, _ = _depth_summary(depth, mask)
        dz = float(depth_med) if np.isfinite(depth_med) else 0.0
        confidence = float(
            np.clip(
                0.10 + 3.2 * observability + _entity_confidence_from_hints(target) - 0.18 * fit_residual,
                0.0,
                1.0,
            )
        )
        return LocalGeometryError(
            valid=confidence >= skill_spec.shadow_confidence,
            confidence=confidence,
            dx=float(dx),
            dy=float(dy),
            dz=dz,
            dyaw=0.0,
            observability=observability,
            fit_residual=float(fit_residual),
            inlier_ratio=float(np.clip(1.0 - fit_residual, 0.0, 1.0)),
            reason="ok" if confidence >= skill_spec.shadow_confidence else "low_confidence",
            target_entity=target.name,
            reference_entity=skill_spec.reference_entity,
            stage_name=stage_name,
            yaw_valid=False,
            yaw_reason="image_pca_axis_not_jaw_local_residual",
            image_axis_yaw=float(image_axis_yaw),
        )


class RingSpokeAlignLocalizer(_BaseLocalizer):
    """Estimate the held ring aperture relative to the target spoke axis."""

    def localize(
        self,
        observation: Any,
        robot_state: Any,
        task_spec: PrecisionTaskSpec,
        skill_spec: PrecisionSkillSpec,
        *,
        stage_name: str = "",
    ) -> LocalGeometryError:
        target = task_spec.get_entity(skill_spec.target_entity)
        ring_entity = task_spec.get_entity(skill_spec.reference_entity) if skill_spec.reference_entity in task_spec.entities else None
        if ring_entity is None:
            for entity in task_spec.entities.values():
                if "ring" in entity.primitive.lower() or "ring" in entity.name.lower():
                    ring_entity = entity
                    break

        rgb = _first_non_none(_as_rgb_array(_obs_get(observation, "wrist_rgb")), _as_rgb_array(_obs_get(observation, "front_rgb")))
        depth = _first_non_none(_as_depth_array(_obs_get(observation, "wrist_depth")), _as_depth_array(_obs_get(observation, "front_depth")))
        if rgb is None or depth is None or ring_entity is None:
            return LocalGeometryError(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "missing_rgbd_or_ring", target.name, skill_spec.reference_entity, stage_name)

        spoke_mask = _color_mask(rgb, target.color_hint)
        ring_mask = _color_mask(rgb, ring_entity.color_hint)
        spoke_mask = _maybe_center_crop(spoke_mask, center_fraction=0.90)
        ring_mask = _maybe_center_crop(ring_mask, center_fraction=0.90)
        spoke_mask = ndi.binary_opening(spoke_mask, iterations=1)
        ring_mask = ndi.binary_opening(ring_mask, iterations=1)
        spoke_mask = _largest_component(spoke_mask)
        ring_mask = _largest_component(ring_mask)

        spoke_dx, spoke_dy, spoke_yaw, spoke_obs, spoke_resid = self._error_from_mask(spoke_mask, depth)
        ring_dx, ring_dy, ring_yaw, ring_obs, ring_resid = self._error_from_mask(ring_mask, depth)
        if not np.isfinite(spoke_dx) or not np.isfinite(ring_dx):
            return LocalGeometryError(False, 0.0, 0.0, 0.0, 0.0, 0.0, float(min(spoke_obs, ring_obs)), float(max(spoke_resid, ring_resid)), 0.0, "weak_spoke_or_ring_mask", target.name, skill_spec.reference_entity, stage_name)

        dx = float(spoke_dx - ring_dx)
        dy = float(spoke_dy - ring_dy)
        symmetry = float(
            target.observable_hints.get(
                "symmetry_period",
                ring_entity.observable_hints.get("symmetry_period", np.pi / 2.0),
            )
        )
        yaw_error = float(((spoke_yaw - ring_yaw + 0.5 * symmetry) % symmetry) - 0.5 * symmetry)
        depth_med_spoke, _ = _depth_summary(depth, spoke_mask)
        depth_med_ring, _ = _depth_summary(depth, ring_mask)
        dz = float(depth_med_spoke - depth_med_ring) if np.isfinite(depth_med_spoke) and np.isfinite(depth_med_ring) else 0.0
        observability = float(min(spoke_obs, ring_obs))
        fit_residual = float(max(spoke_resid, ring_resid))
        confidence = float(
            np.clip(
                0.08 + 2.9 * observability + _entity_confidence_from_hints(target) + _entity_confidence_from_hints(ring_entity) - 0.22 * fit_residual,
                0.0,
                1.0,
            )
        )
        return LocalGeometryError(
            valid=confidence >= skill_spec.shadow_confidence,
            confidence=confidence,
            dx=dx,
            dy=dy,
            dz=dz,
            dyaw=yaw_error,
            observability=observability,
            fit_residual=fit_residual,
            inlier_ratio=float(np.clip(1.0 - fit_residual, 0.0, 1.0)),
            reason="ok" if confidence >= skill_spec.shadow_confidence else "low_confidence",
            target_entity=target.name,
            reference_entity=skill_spec.reference_entity,
            stage_name=stage_name,
        )
