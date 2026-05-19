"""Non-privileged Coarse2Contact runtime controller.

This module keeps the VLA planner frozen and delegates near-contact control to
two explicit owners:

* a local depth geometry estimator that emits a bounded planar correction
  [dx, dy, dyaw, confidence]
* a recovery state machine that owns jam / invalid-action / backoff behavior

The implementation is intentionally conservative. It does not read any
privileged target or teacher signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from prismatic.robot.residual_safety import ResidualSafety
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local


class Coarse2ContactPhase(IntEnum):
    COARSE = 0
    VISUAL_ALIGN = 1
    PROBE_CONTACT = 2
    CONTACT_INSERT = 3
    RECOVER = 4
    DONE = 5
    FAIL = 6


class Coarse2ContactContactState(IntEnum):
    FREE = 0
    TOUCH = 1
    EDGE = 2
    PARTIAL = 3
    JAM = 4
    SEATED = 5


class RecoveryPhase(IntEnum):
    IDLE = 0
    BACKOFF = 1
    UNLOAD = 2
    SEARCH = 3
    REAPPROACH = 4
    FAILED = 5


@dataclass
class DepthLocalizerEstimate:
    valid: bool
    confidence: float
    dx: float
    dy: float
    dyaw: float
    depth_gap: float
    support_fraction: float
    centroid_u: float
    centroid_v: float
    reason: str
    correction_local: np.ndarray


@dataclass
class VisualAlignmentEstimate:
    valid: bool
    confidence: float
    xy_error: float
    z_gap: float
    yaw_error: float
    correction_local: np.ndarray
    reason: str
    mask_fraction: float
    centroid_u: float
    centroid_v: float
    dx: float
    dy: float
    dyaw: float


class WrenchFilter:
    def __init__(
        self,
        *,
        alpha: float = 0.25,
        bias_alpha: float = 0.03,
        bias_window: int = 24,
        low_force_threshold: float = 0.20,
        low_force_axes_threshold: float = 0.12,
    ) -> None:
        self.alpha = float(alpha)
        self.bias_alpha = float(bias_alpha)
        self.bias_window = int(bias_window)
        self.low_force_threshold = float(low_force_threshold)
        self.low_force_axes_threshold = float(low_force_axes_threshold)
        self.reset()

    def reset(self) -> None:
        self._filtered = np.zeros(6, dtype=np.float32)
        self._bias = np.zeros(6, dtype=np.float32)
        self._samples = 0
        self._bias_samples: list[np.ndarray] = []
        self._last_raw = np.zeros(6, dtype=np.float32)
        self._last_delta = np.zeros(6, dtype=np.float32)
        self._last_is_bias_update = False

    def update(self, force_reading: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        raw = np.zeros(6, dtype=np.float32) if force_reading is None else np.asarray(force_reading, dtype=np.float32).reshape(-1)[:6]
        if raw.size < 6:
            raw = np.pad(raw, (0, 6 - raw.size), constant_values=0.0)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._last_raw = raw.copy()
        centered = raw - self._bias
        if self._samples == 0:
            self._filtered = centered.copy()
        else:
            self._filtered = self.alpha * centered + (1.0 - self.alpha) * self._filtered
        self._samples += 1

        raw_norm = float(np.linalg.norm(raw[:3]))
        low_force = bool(raw_norm <= self.low_force_threshold and float(np.max(np.abs(raw[3:6]))) <= self.low_force_axes_threshold)
        if low_force:
            self._bias_samples.append(raw.copy())
            self._bias_samples = self._bias_samples[-self.bias_window :]
            self._bias = (1.0 - self.bias_alpha) * self._bias + self.bias_alpha * raw
            self._last_is_bias_update = True
        else:
            self._last_is_bias_update = False

        delta = centered - self._filtered
        self._last_delta = delta.copy()
        return raw.copy(), self._filtered.copy(), delta.copy(), bool(low_force)

    def get_stats(self) -> dict:
        return {
            "samples": int(self._samples),
            "bias": self._bias.astype(np.float32).tolist(),
            "filtered": self._filtered.astype(np.float32).tolist(),
            "last_raw": self._last_raw.astype(np.float32).tolist(),
            "last_delta": self._last_delta.astype(np.float32).tolist(),
            "bias_window": int(len(self._bias_samples)),
            "last_is_bias_update": bool(self._last_is_bias_update),
        }


class DepthVisualAligner:
    """Local wrist-depth geometry locator for pre-contact alignment."""

    def __init__(
        self,
        *,
        precontact_depth_threshold: float = 0.20,
        contact_depth_threshold: float = 0.035,
        xy_threshold: float = 0.0015,
        yaw_threshold: float = 0.0349,
        max_xy_step: float = 0.0005,
        max_yaw_step: float = 0.0087,
        min_mask_fraction: float = 0.002,
        roi_fraction: float = 0.68,
        center_prior_sigma: float = 0.36,
        support_depth_band: float = 0.018,
        wrist_fov_deg: float = 65.0,
        min_support_points: int = 32,
    ) -> None:
        self.precontact_depth_threshold = float(precontact_depth_threshold)
        self.contact_depth_threshold = float(contact_depth_threshold)
        self.xy_threshold = float(xy_threshold)
        self.yaw_threshold = float(yaw_threshold)
        self.max_xy_step = float(max_xy_step)
        self.max_yaw_step = float(max_yaw_step)
        self.min_mask_fraction = float(min_mask_fraction)
        self.roi_fraction = float(roi_fraction)
        self.center_prior_sigma = float(center_prior_sigma)
        self.support_depth_band = float(support_depth_band)
        self.wrist_fov_deg = float(wrist_fov_deg)
        self.min_support_points = int(min_support_points)

    @staticmethod
    def depth_proximity(wrist_depth) -> float:
        if wrist_depth is None:
            return float("nan")
        if hasattr(wrist_depth, "detach"):
            wrist_depth = wrist_depth.detach().float().cpu().numpy()
        depth = np.asarray(wrist_depth, dtype=np.float32).squeeze()
        valid = depth[np.isfinite(depth)]
        if valid.size == 0:
            return float("nan")
        return float(np.percentile(valid, 8.0))

    def localize(self, wrist_depth) -> DepthLocalizerEstimate:
        if wrist_depth is None:
            return self._empty("missing_depth")
        if hasattr(wrist_depth, "detach"):
            wrist_depth = wrist_depth.detach().float().cpu().numpy()
        depth = np.asarray(wrist_depth, dtype=np.float32).squeeze()
        if depth.ndim != 2 or depth.size == 0:
            return self._empty("bad_depth_shape")

        finite = np.logical_and(np.isfinite(depth), depth > 1.0e-6)
        if not np.any(finite):
            return self._empty("no_valid_depth")

        h, w = depth.shape
        roi_w = max(int(round(w * self.roi_fraction)), 8)
        roi_h = max(int(round(h * self.roi_fraction)), 8)
        x0 = max((w - roi_w) // 2, 0)
        y0 = max((h - roi_h) // 2, 0)
        x1 = min(x0 + roi_w, w)
        y1 = min(y0 + roi_h, h)
        roi = depth[y0:y1, x0:x1]
        roi_finite = finite[y0:y1, x0:x1]
        if not np.any(roi_finite):
            return self._empty("no_valid_roi")

        roi_valid = roi[roi_finite]
        prox = float(np.percentile(roi_valid, 8.0))
        support_hi = prox + max(self.support_depth_band, prox * 0.08)
        support_mask = np.logical_and(roi_finite, np.logical_and(roi >= prox, roi <= support_hi))
        support_fraction = float(np.mean(support_mask))
        support_points = int(np.count_nonzero(support_mask))
        if support_fraction < self.min_mask_fraction or support_points < self.min_support_points:
            return self._empty("weak_depth_support", depth_gap=prox, support_fraction=support_fraction)

        ys, xs = np.nonzero(support_mask)
        z = roi[support_mask].astype(np.float32)
        weights = (support_hi - z).astype(np.float32) + 1.0e-3
        if weights.size == 0 or float(np.sum(weights)) <= 1.0e-8:
            return self._empty("zero_weight_support", depth_gap=prox, support_fraction=support_fraction)

        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)
        center_x = float((x1 - x0 - 1) * 0.5)
        center_y = float((y1 - y0 - 1) * 0.5)
        dist2 = ((xs - center_x) / max(center_x, 1.0)) ** 2 + ((ys - center_y) / max(center_y, 1.0)) ** 2
        center_prior = np.exp(-0.5 * dist2 / max(self.center_prior_sigma**2, 1.0e-6)).astype(np.float32)
        weights = weights * center_prior
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1.0e-8:
            return self._empty("zero_centered_support", depth_gap=prox, support_fraction=support_fraction)

        cx_px = float(np.sum(xs * weights) / weight_sum)
        cy_px = float(np.sum(ys * weights) / weight_sum)
        u = float((cx_px - center_x) / max(center_x, 1.0))
        v = float((cy_px - center_y) / max(center_y, 1.0))

        fov_rad = np.deg2rad(self.wrist_fov_deg)
        fx = max((x1 - x0 - 1) * 0.5 / max(np.tan(fov_rad * 0.5), 1.0e-6), 1.0)
        fy = max((y1 - y0 - 1) * 0.5 / max(np.tan(fov_rad * 0.5), 1.0e-6), 1.0)
        x_metric = ((xs - center_x) / fx) * z
        y_metric = ((ys - center_y) / fy) * z
        points_xy = np.stack([x_metric, y_metric], axis=1)
        centroid_xy = np.sum(points_xy * weights[:, None], axis=0) / weight_sum

        dx = float(np.clip(-centroid_xy[0], -self.max_xy_step, self.max_xy_step))
        dy = float(np.clip(-centroid_xy[1], -self.max_xy_step, self.max_xy_step))

        dyaw = 0.0
        planarity = 0.0
        if xs.size >= 8:
            centered_metric = points_xy - centroid_xy[None, :]
            try:
                cov = np.cov(centered_metric.T, aweights=weights.astype(np.float64))
                vals, vecs = np.linalg.eigh(cov)
                vals = np.maximum(vals, 0.0)
                major = vecs[:, int(np.argmax(vals))]
                angle = float(np.arctan2(major[1], major[0]))
                yaw_error = float(((angle + np.pi / 4.0) % (np.pi / 2.0)) - np.pi / 4.0)
                dyaw = float(np.clip(-0.65 * yaw_error, -self.max_yaw_step, self.max_yaw_step))
                planarity = float(vals.max() / max(float(np.sum(vals)), 1.0e-6))
            except Exception:
                dyaw = 0.0

        depth_contrast = float(np.clip((float(np.percentile(roi_valid, 35.0)) - prox) / 0.05, 0.0, 1.0))
        area_score = float(np.clip(support_fraction / max(self.min_mask_fraction * 10.0, 1.0e-6), 0.0, 1.0))
        center_score = float(np.clip(1.0 - min(np.linalg.norm(centroid_xy) / max(self.max_xy_step * 4.0, 1.0e-6), 1.0), 0.0, 1.0))
        support_score = float(np.clip(support_points / max(self.min_support_points * 4.0, 1.0), 0.0, 1.0))
        confidence = float(
            np.clip(
                0.10
                + 0.30 * area_score
                + 0.20 * depth_contrast
                + 0.20 * center_score
                + 0.20 * support_score * max(planarity, 0.5),
                0.0,
                1.0,
            )
        )
        correction = np.zeros(6, dtype=np.float32)
        correction[0] = dx
        correction[1] = dy
        correction[5] = dyaw
        return DepthLocalizerEstimate(
            valid=True,
            confidence=confidence,
            dx=dx,
            dy=dy,
            dyaw=dyaw,
            depth_gap=prox,
            support_fraction=support_fraction,
            centroid_u=u,
            centroid_v=v,
            reason="ok",
            correction_local=correction,
        )

    def estimate(self, wrist_depth) -> VisualAlignmentEstimate:
        loc = self.localize(wrist_depth)
        if not loc.valid:
            return VisualAlignmentEstimate(
                valid=False,
                confidence=0.0,
                xy_error=float("nan"),
                z_gap=float(loc.depth_gap),
                yaw_error=float("nan"),
                correction_local=np.zeros(6, dtype=np.float32),
                reason=loc.reason,
                mask_fraction=float(loc.support_fraction),
                centroid_u=float(loc.centroid_u),
                centroid_v=float(loc.centroid_v),
                dx=float(loc.dx),
                dy=float(loc.dy),
                dyaw=float(loc.dyaw),
            )
        xy_error = float(np.hypot(loc.dx, loc.dy))
        yaw_error = float(abs(loc.dyaw))
        return VisualAlignmentEstimate(
            valid=True,
            confidence=float(loc.confidence),
            xy_error=xy_error,
            z_gap=float(loc.depth_gap),
            yaw_error=yaw_error,
            correction_local=loc.correction_local.copy(),
            reason=loc.reason,
            mask_fraction=float(loc.support_fraction),
            centroid_u=float(loc.centroid_u),
            centroid_v=float(loc.centroid_v),
            dx=float(loc.dx),
            dy=float(loc.dy),
            dyaw=float(loc.dyaw),
        )

    @staticmethod
    def _empty(
        reason: str,
        *,
        depth_gap: float = float("nan"),
        support_fraction: float = 0.0,
    ) -> DepthLocalizerEstimate:
        return DepthLocalizerEstimate(
            valid=False,
            confidence=0.0,
            dx=0.0,
            dy=0.0,
            dyaw=0.0,
            depth_gap=float(depth_gap),
            support_fraction=float(support_fraction),
            centroid_u=float("nan"),
            centroid_v=float("nan"),
            reason=str(reason),
            correction_local=np.zeros(6, dtype=np.float32),
        )


class ContactStateEstimator:
    def __init__(
        self,
        *,
        contact_threshold: float = 0.18,
        contact_delta_threshold: float = 0.05,
        jam_threshold: float = 0.55,
        torque_threshold: float = 0.12,
        seated_depth_threshold: float = 0.010,
        progress_window: int = 8,
        progress_threshold: float = 0.00035,
        edge_xy_threshold: float = 0.004,
        force_spike_threshold: float = 1.0,
    ) -> None:
        self.contact_threshold = float(contact_threshold)
        self.contact_delta_threshold = float(contact_delta_threshold)
        self.jam_threshold = float(jam_threshold)
        self.torque_threshold = float(torque_threshold)
        self.seated_depth_threshold = float(seated_depth_threshold)
        self.progress_window = int(progress_window)
        self.progress_threshold = float(progress_threshold)
        self.edge_xy_threshold = float(edge_xy_threshold)
        self.force_spike_threshold = float(force_spike_threshold)
        self._prev_force_norm: Optional[float] = None
        self._z_hist: list[float] = []
        self.state = Coarse2ContactContactState.FREE
        self.reason = "init"
        self.jam_count = 0
        self.force_spike_count = 0

    def reset(self) -> None:
        self._prev_force_norm = None
        self._z_hist.clear()
        self.state = Coarse2ContactContactState.FREE
        self.reason = "reset"
        self.jam_count = 0
        self.force_spike_count = 0

    def update(
        self,
        *,
        force_reading: Optional[np.ndarray],
        gripper_z: Optional[float],
        depth_gap: float,
        visual_xy_error: float,
    ) -> Coarse2ContactContactState:
        f = np.zeros(6, dtype=np.float32) if force_reading is None else np.asarray(force_reading, dtype=np.float32).reshape(-1)[:6]
        if f.size < 6:
            f = np.pad(f, (0, 6 - f.size), constant_values=0.0)
        force_norm = float(np.linalg.norm(f[:3]))
        fz = abs(float(f[2]))
        fxy = float(np.linalg.norm(f[:2]))
        torque_xy = float(np.linalg.norm(f[3:5]))
        delta_force = 0.0 if self._prev_force_norm is None else force_norm - self._prev_force_norm
        self._prev_force_norm = force_norm
        if gripper_z is not None and np.isfinite(float(gripper_z)):
            self._z_hist.append(float(gripper_z))
            self._z_hist = self._z_hist[-self.progress_window :]
        z_progress = 0.0
        if len(self._z_hist) >= 2:
            z_progress = abs(float(self._z_hist[-1] - self._z_hist[0]))

        jam = bool(
            (fz >= self.jam_threshold and z_progress < self.progress_threshold)
            or torque_xy >= self.torque_threshold
            or force_norm >= self.force_spike_threshold
        )
        contact = bool(
            fz >= self.contact_threshold
            or fxy >= self.contact_threshold
            or delta_force >= self.contact_delta_threshold
        )
        if jam:
            self.state = Coarse2ContactContactState.JAM
            self.reason = "jam_force_or_torque" if force_norm < self.force_spike_threshold else "force_spike"
            self.jam_count += 1
            if force_norm >= self.force_spike_threshold:
                self.force_spike_count += 1
        elif contact and np.isfinite(depth_gap) and depth_gap <= self.seated_depth_threshold:
            self.state = Coarse2ContactContactState.SEATED
            self.reason = "seated_depth_contact"
        elif contact and np.isfinite(visual_xy_error) and visual_xy_error > self.edge_xy_threshold:
            self.state = Coarse2ContactContactState.EDGE
            self.reason = "lateral_error_under_contact"
        elif contact and np.isfinite(depth_gap) and depth_gap > self.seated_depth_threshold * 2.0:
            self.state = Coarse2ContactContactState.PARTIAL
            self.reason = "contact_not_seated_yet"
        elif contact:
            self.state = Coarse2ContactContactState.TOUCH
            self.reason = "contact_onset"
        else:
            self.state = Coarse2ContactContactState.FREE
            self.reason = "free"
        return self.state


class RecoveryPrimitiveBank:
    def __init__(
        self,
        *,
        backoff_m: float = 0.003,
        lateral_m: float = 0.0015,
        unload_m: float = 0.0010,
        yaw_rad: float = 0.0087,
    ) -> None:
        self.backoff_m = float(backoff_m)
        self.lateral_m = float(lateral_m)
        self.unload_m = float(unload_m)
        self.yaw_rad = float(yaw_rad)
        self._toggle = 1.0

    def reset(self) -> None:
        self._toggle = 1.0

    def backoff(self) -> tuple[np.ndarray, str]:
        out = np.zeros(6, dtype=np.float32)
        out[2] = self.backoff_m
        out[0] = self._toggle * self.lateral_m
        self._toggle *= -1.0
        return out, "backoff"

    def unload(self) -> tuple[np.ndarray, str]:
        out = np.zeros(6, dtype=np.float32)
        out[2] = self.unload_m
        return out, "unload"

    def search(self) -> tuple[np.ndarray, str]:
        out = np.zeros(6, dtype=np.float32)
        out[0] = self._toggle * self.lateral_m
        out[1] = -self._toggle * 0.5 * self.lateral_m
        out[5] = self._toggle * self.yaw_rad
        self._toggle *= -1.0
        return out, "micro_search"

    def reapproach(self) -> tuple[np.ndarray, str]:
        out = np.zeros(6, dtype=np.float32)
        out[2] = -0.5 * self.backoff_m
        return out, "reapproach"

    def edge_relief(self, force_reading: Optional[np.ndarray]) -> tuple[np.ndarray, str]:
        out = np.zeros(6, dtype=np.float32)
        if force_reading is not None:
            f = np.asarray(force_reading, dtype=np.float32).reshape(-1)[:3]
            fxy = f[:2]
            mag = float(np.linalg.norm(fxy))
            if mag > 1.0e-6:
                out[:2] = (-fxy / mag * self.lateral_m).astype(np.float32)
            else:
                out[0] = self._toggle * self.lateral_m
        else:
            out[0] = self._toggle * self.lateral_m
        out[5] = self._toggle * self.yaw_rad
        self._toggle *= -1.0
        return out, "edge_relief"

    def touch_slowdown(self, base_local: np.ndarray) -> tuple[np.ndarray, str]:
        out = np.zeros(6, dtype=np.float32)
        base = np.asarray(base_local, dtype=np.float32).reshape(-1)[:6]
        out[2] = float(base[2]) * -0.35 if float(base[2]) < 0.0 else float(base[2]) * 0.35
        return out, "touch_slowdown"


class RecoveryStateMachine:
    def __init__(
        self,
        *,
        safety: ResidualSafety,
        recovery_bank: RecoveryPrimitiveBank,
        backoff_steps: int = 2,
        unload_steps: int = 2,
        search_steps: int = 3,
        reapproach_steps: int = 1,
        max_recovery_cycles: int = 4,
    ) -> None:
        self.safety = safety
        self.recovery_bank = recovery_bank
        self.backoff_steps = int(backoff_steps)
        self.unload_steps = int(unload_steps)
        self.search_steps = int(search_steps)
        self.reapproach_steps = int(reapproach_steps)
        self.max_recovery_cycles = int(max_recovery_cycles)
        self.reset()

    def reset(self) -> None:
        self.phase = RecoveryPhase.IDLE
        self.phase_age = 0
        self.reason = "idle"
        self.recovery_cycles = 0
        self.invalid_count = 0
        self.force_stop_count = 0
        self.jam_count = 0
        self.last_primitive = "none"

    def _enter(self, phase: RecoveryPhase, reason: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_age = 0
        self.reason = str(reason)

    def trigger(
        self,
        *,
        contact_state: Coarse2ContactContactState,
        force_reading: Optional[np.ndarray],
        invalid_action: bool = False,
    ) -> bool:
        force_stop = bool(self.safety.check_force_stop(force_reading))
        if force_stop:
            self.force_stop_count += 1
        if invalid_action:
            self.invalid_count += 1
            self._enter(RecoveryPhase.BACKOFF, "invalid_action")
            self.recovery_cycles += 1
            return True
        if contact_state == Coarse2ContactContactState.JAM or force_stop:
            self.jam_count += int(contact_state == Coarse2ContactContactState.JAM)
            self._enter(RecoveryPhase.BACKOFF, "jam" if contact_state == Coarse2ContactContactState.JAM else "force_stop")
            self.recovery_cycles += 1
            return True
        if self.phase == RecoveryPhase.IDLE and contact_state in (
            Coarse2ContactContactState.TOUCH,
            Coarse2ContactContactState.EDGE,
            Coarse2ContactContactState.PARTIAL,
        ):
            reason = {
                Coarse2ContactContactState.TOUCH: "touch_contact",
                Coarse2ContactContactState.EDGE: "edge_contact",
                Coarse2ContactContactState.PARTIAL: "partial_contact",
            }[contact_state]
            self._enter(RecoveryPhase.BACKOFF, reason)
            self.recovery_cycles += 1
            return True
        return self.phase != RecoveryPhase.IDLE

    def is_active(self) -> bool:
        return self.phase not in (RecoveryPhase.IDLE, RecoveryPhase.FAILED)

    def step(
        self,
        *,
        contact_state: Coarse2ContactContactState,
        force_reading: Optional[np.ndarray],
        local_base: np.ndarray,
        invalid_action: bool = False,
    ) -> tuple[np.ndarray, str, str, bool]:
        self.trigger(contact_state=contact_state, force_reading=force_reading, invalid_action=invalid_action)
        force_vec = np.zeros(6, dtype=np.float32) if force_reading is None else np.asarray(force_reading, dtype=np.float32).reshape(-1)[:6]
        if force_vec.size < 6:
            force_vec = np.pad(force_vec, (0, 6 - force_vec.size), constant_values=0.0)
        force_norm = float(np.linalg.norm(force_vec[:3]))
        base = np.asarray(local_base, dtype=np.float32).reshape(-1)[:6]

        if self.phase == RecoveryPhase.IDLE:
            self.last_primitive = "none"
            return np.zeros(6, dtype=np.float32), "none", "idle", False

        if self.phase == RecoveryPhase.BACKOFF:
            delta, primitive = self.recovery_bank.backoff()
            self.last_primitive = primitive
            if self.phase_age >= self.backoff_steps or force_norm <= self.safety.backoff_force_threshold:
                self._enter(RecoveryPhase.UNLOAD, "backoff_complete")
            else:
                self.phase_age += 1
            return delta, primitive, self.reason, True

        if self.phase == RecoveryPhase.UNLOAD:
            delta, primitive = self.recovery_bank.unload()
            self.last_primitive = primitive
            if self.phase_age >= self.unload_steps and force_norm <= self.safety.backoff_force_threshold * 0.5:
                self._enter(RecoveryPhase.SEARCH, "unload_complete")
            else:
                self.phase_age += 1
            return delta, primitive, self.reason, True

        if self.phase == RecoveryPhase.SEARCH:
            delta, primitive = self.recovery_bank.search()
            self.last_primitive = primitive
            if self.phase_age >= self.search_steps:
                self._enter(RecoveryPhase.REAPPROACH, "search_complete")
            else:
                self.phase_age += 1
            return delta, primitive, self.reason, True

        if self.phase == RecoveryPhase.REAPPROACH:
            delta, primitive = self.recovery_bank.reapproach()
            self.last_primitive = primitive
            if self.phase_age >= self.reapproach_steps:
                self._enter(RecoveryPhase.IDLE, "recovery_complete")
            else:
                self.phase_age += 1
            return delta, primitive, self.reason, True

        self.phase = RecoveryPhase.FAILED
        self.last_primitive = "failed"
        return np.zeros(6, dtype=np.float32), "failed", "failed", False


class ForceReflexController:
    def __init__(self, *, safety: ResidualSafety, recovery_bank: RecoveryPrimitiveBank) -> None:
        self.safety = safety
        self.recovery_bank = recovery_bank
        self.state_machine = RecoveryStateMachine(safety=safety, recovery_bank=recovery_bank)

    def reset(self) -> None:
        self.state_machine.reset()

    def correction(
        self,
        *,
        contact_state: Coarse2ContactContactState,
        force_reading: Optional[np.ndarray],
        local_base: np.ndarray,
        invalid_action: bool = False,
    ) -> tuple[np.ndarray, str, str, bool]:
        return self.state_machine.step(
            contact_state=contact_state,
            force_reading=force_reading,
            local_base=local_base,
            invalid_action=invalid_action,
        )

    def is_active(self) -> bool:
        return self.state_machine.is_active()

    @property
    def phase(self) -> RecoveryPhase:
        return self.state_machine.phase

    @property
    def reason(self) -> str:
        return self.state_machine.reason

    @property
    def last_primitive(self) -> str:
        return self.state_machine.last_primitive

    def get_stats(self) -> dict:
        return {
            "recovery_phase": self.state_machine.phase.name,
            "recovery_phase_id": int(self.state_machine.phase),
            "recovery_reason": self.state_machine.reason,
            "recovery_cycles": int(self.state_machine.recovery_cycles),
            "invalid_count": int(self.state_machine.invalid_count),
            "force_stop_count": int(self.state_machine.force_stop_count),
            "jam_count": int(self.state_machine.jam_count),
            "last_primitive": str(self.state_machine.last_primitive),
        }


class Coarse2ContactSupervisor:
    """Frozen-planner local correction supervisor."""

    def __init__(
        self,
        *,
        mode: str = "depth_force",
        shadow_only: bool = False,
        visual_xy_threshold: float = 0.0015,
        visual_yaw_threshold: float = 0.0349,
        visual_precontact_depth_threshold: float = 0.20,
        visual_contact_depth_threshold: float = 0.035,
        max_xy_step: float = 0.0005,
        max_yaw_step: float = 0.0087,
        force_contact_threshold: float = 0.18,
        force_delta_contact_threshold: float = 0.05,
        force_jam_threshold: float = 0.55,
        force_torque_threshold: float = 0.12,
        force_spike_threshold: float = 1.0,
        backoff_m: float = 0.003,
        lateral_m: float = 0.0015,
        chunk_size: int = 4,
    ) -> None:
        allowed = {"depth_shadow", "depth_apply", "force_reflex", "depth_force"}
        if mode not in allowed:
            raise ValueError(f"Unknown Coarse2Contact mode {mode!r}; expected one of {sorted(allowed)}")
        self.mode = str(mode)
        self.shadow_only = bool(shadow_only or mode == "depth_shadow")
        self.chunk_size = int(chunk_size)
        self.visual = DepthVisualAligner(
            precontact_depth_threshold=visual_precontact_depth_threshold,
            contact_depth_threshold=visual_contact_depth_threshold,
            xy_threshold=visual_xy_threshold,
            yaw_threshold=visual_yaw_threshold,
            max_xy_step=max_xy_step,
            max_yaw_step=max_yaw_step,
        )
        self.contact = ContactStateEstimator(
            contact_threshold=force_contact_threshold,
            contact_delta_threshold=force_delta_contact_threshold,
            jam_threshold=force_jam_threshold,
            torque_threshold=force_torque_threshold,
            seated_depth_threshold=visual_contact_depth_threshold * 0.35,
            force_spike_threshold=force_spike_threshold,
        )
        self.recovery = RecoveryPrimitiveBank(backoff_m=backoff_m, lateral_m=lateral_m, yaw_rad=max_yaw_step)
        self.wrench_filter = WrenchFilter()
        self.safety = ResidualSafety(
            max_residual_pos=max(max_xy_step, lateral_m, backoff_m),
            max_residual_rot=max_yaw_step,
            force_stop_threshold=force_spike_threshold,
            backoff_distance=backoff_m,
            backoff_force_threshold=force_jam_threshold,
            lateral_force_threshold=force_contact_threshold * 1.5,
            vertical_force_threshold=force_jam_threshold,
            torque_threshold=force_torque_threshold,
            max_delta_pos=0.025,
            max_delta_rot=0.10,
        )
        self.force_reflex = ForceReflexController(safety=self.safety, recovery_bank=self.recovery)
        self.phase = Coarse2ContactPhase.COARSE
        self.correction_count = 0
        self.visual_apply_count = 0
        self.force_apply_count = 0
        self.recovery_count = 0
        self.precontact_count = 0
        self.preinsert_count = 0
        self.invalid_action_count = 0
        self._last_trace: dict = {}

    def reset(self) -> None:
        self.contact.reset()
        self.recovery.reset()
        self.wrench_filter.reset()
        self.force_reflex.reset()
        self.safety.reset_counters()
        self.phase = Coarse2ContactPhase.COARSE
        self.correction_count = 0
        self.visual_apply_count = 0
        self.force_apply_count = 0
        self.recovery_count = 0
        self.precontact_count = 0
        self.preinsert_count = 0
        self.invalid_action_count = 0
        self._last_trace = {}

    def get_chunk_size(self) -> int:
        return max(1, self.chunk_size)

    @staticmethod
    def _extract_current_quat(proprio) -> np.ndarray:
        if proprio is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        arr = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if arr.size >= 14:
            return arr[10:14].copy()
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    @staticmethod
    def _pose_to_abs_action(current_gripper_pose: np.ndarray, delta_local: np.ndarray, gripper_open: float) -> np.ndarray:
        pose = np.asarray(current_gripper_pose, dtype=np.float32).copy().reshape(7)
        delta = np.asarray(delta_local, dtype=np.float32).copy().reshape(6)
        delta_world = local_delta_to_world(delta, pose[3:7]).astype(np.float32)
        pose[:3] = pose[:3] + delta_world[:3]
        r_cur = Rotation.from_quat(pose[3:7])
        r_delta = Rotation.from_rotvec(delta_world[3:6])
        pose[3:7] = (r_delta * r_cur).as_quat().astype(np.float32)
        gripper_cmd = 1.0 if float(gripper_open) > 0.5 else 0.0
        return np.concatenate([pose[:7], [gripper_cmd]]).astype(np.float32)

    def build_invalid_action_recovery_absolute(
        self,
        current_gripper_pose: np.ndarray,
        gripper_open: float,
        *,
        force_reading: Optional[np.ndarray] = None,
        proprio=None,
    ) -> np.ndarray:
        raw_force = np.zeros(6, dtype=np.float32) if force_reading is None else np.asarray(force_reading, dtype=np.float32).reshape(-1)[:6]
        if raw_force.size < 6:
            raw_force = np.pad(raw_force, (0, 6 - raw_force.size), constant_values=0.0)
        raw_force, filtered_force, _, _ = self.wrench_filter.update(raw_force)
        local_base = np.zeros(6, dtype=np.float32)
        delta_local, primitive, reason, active = self.force_reflex.correction(
            contact_state=self.contact.state,
            force_reading=filtered_force,
            local_base=local_base,
            invalid_action=True,
        )
        if not active and np.allclose(delta_local, 0.0):
            delta_local = np.zeros(6, dtype=np.float32)
            delta_local[2] = self.recovery.backoff_m
        abs_action = self._pose_to_abs_action(current_gripper_pose, delta_local, gripper_open)
        self._last_trace = {
            "coarse2contact_phase": self.phase.name,
            "coarse2contact_phase_id": int(self.phase),
            "coarse2contact_mode": self.mode,
            "coarse2contact_shadow_only": bool(self.shadow_only),
            "uses_privileged_target": False,
            "invalid_action": True,
            "invalid_action_count": int(self.invalid_action_count + 1),
            "invalid_action_flag": True,
            "recovery_phase": self.force_reflex.phase.name,
            "recovery_phase_id": int(self.force_reflex.phase),
            "force_reflex_reason": str(reason),
            "recovery_primitive": str(primitive),
            "recovery_active": bool(active),
            "raw_wrench_6d": raw_force.astype(np.float32).tolist(),
            "filtered_wrench_6d": filtered_force.astype(np.float32).tolist(),
            "raw_wrench": raw_force.astype(np.float32).tolist(),
            "filtered_wrench": filtered_force.astype(np.float32).tolist(),
            "depth_conf": 0.0,
            "depth_obs_quality": 0.0,
            "phase_owner": "recover",
            "phase_reason": str(reason),
            "retry_id": int(self.force_reflex.state_machine.recovery_cycles),
        }
        self.invalid_action_count += 1
        self.phase = Coarse2ContactPhase.RECOVER
        return abs_action

    def step(
        self,
        a_base_7d: np.ndarray,
        *,
        force_reading: Optional[np.ndarray] = None,
        gripper_z: Optional[float] = None,
        wrist_depth=None,
        proprio=None,
        **_: object,
    ) -> np.ndarray:
        base = np.asarray(a_base_7d, dtype=np.float32).copy()
        if base.size < 7:
            raise ValueError(f"a_base_7d must have at least 7 elements, got {base.shape}")

        raw_force, filtered_force, force_filter_delta, force_low_force = self.wrench_filter.update(force_reading)
        estimate = self.visual.estimate(wrist_depth)
        state = self.contact.update(
            force_reading=filtered_force,
            gripper_z=gripper_z,
            depth_gap=estimate.z_gap,
            visual_xy_error=estimate.xy_error,
        )
        precontact = bool(estimate.valid and np.isfinite(estimate.z_gap) and estimate.z_gap <= self.visual.precontact_depth_threshold)
        if precontact:
            self.precontact_count += 1
        visual_ready = bool(
            estimate.valid
            and estimate.confidence >= 0.35
            and abs(estimate.dx) <= self.visual.xy_threshold
            and abs(estimate.dy) <= self.visual.xy_threshold
            and abs(estimate.dyaw) <= self.visual.yaw_threshold
        )
        if visual_ready:
            self.preinsert_count += 1

        recovery_active = self.force_reflex.is_active()
        if state in (Coarse2ContactContactState.JAM, Coarse2ContactContactState.SEATED):
            recovery_active = recovery_active or state == Coarse2ContactContactState.JAM
        if recovery_active and self.force_reflex.phase == RecoveryPhase.IDLE and state != Coarse2ContactContactState.SEATED:
            recovery_active = False

        if state == Coarse2ContactContactState.SEATED:
            self.phase = Coarse2ContactPhase.DONE
        elif self.force_reflex.phase != RecoveryPhase.IDLE:
            self.phase = Coarse2ContactPhase.RECOVER
        elif state == Coarse2ContactContactState.JAM:
            self.phase = Coarse2ContactPhase.RECOVER
        elif state in (Coarse2ContactContactState.TOUCH, Coarse2ContactContactState.EDGE, Coarse2ContactContactState.PARTIAL):
            self.phase = Coarse2ContactPhase.CONTACT_INSERT
        elif visual_ready:
            self.phase = Coarse2ContactPhase.PROBE_CONTACT
        elif precontact:
            self.phase = Coarse2ContactPhase.VISUAL_ALIGN
        else:
            self.phase = Coarse2ContactPhase.COARSE

        quat = self._extract_current_quat(proprio)
        local_base = world_delta_to_local(base[:6], quat)
        local_out = local_base.copy()
        visual_delta = np.zeros(6, dtype=np.float32)
        force_delta = np.zeros(6, dtype=np.float32)
        recovery_primitive = "none"
        recovery_reason = "idle"
        correction_owner = "planner"
        applied = False
        recovery_mode_active = False

        use_visual = self.mode in ("depth_shadow", "depth_apply", "depth_force")
        use_force = self.mode in ("force_reflex", "depth_force")

        if self.phase == Coarse2ContactPhase.VISUAL_ALIGN and use_visual and estimate.valid and estimate.confidence >= 0.20:
            visual_delta[0] = estimate.dx
            visual_delta[1] = estimate.dy
            visual_delta[5] = estimate.dyaw
            correction_owner = "visual_align"
            local_out[:6] = local_base[:6] + visual_delta
        elif self.phase == Coarse2ContactPhase.PROBE_CONTACT:
            if use_visual and estimate.valid and estimate.confidence >= 0.35:
                visual_delta[0] = estimate.dx
                visual_delta[1] = estimate.dy
                visual_delta[5] = estimate.dyaw
                correction_owner = "probe_visual"
                local_out[:6] = local_base[:6] + visual_delta
            if use_force:
                force_delta, recovery_primitive, recovery_reason, recovery_mode_active = self.force_reflex.correction(
                    contact_state=state,
                    force_reading=filtered_force,
                    local_base=local_base,
                )
                if recovery_mode_active:
                    correction_owner = "force_probe"
                    local_out[:6] = local_base[:6] + force_delta
                elif state == Coarse2ContactContactState.TOUCH:
                    touch_delta, recovery_primitive = self.recovery.touch_slowdown(local_base)
                    force_delta = touch_delta
                    recovery_reason = "touch"
                    correction_owner = "touch_slowdown"
                    local_out[:6] = local_base[:6] + force_delta
        elif self.phase == Coarse2ContactPhase.CONTACT_INSERT:
            if use_force:
                force_delta, recovery_primitive, recovery_reason, recovery_mode_active = self.force_reflex.correction(
                    contact_state=state,
                    force_reading=filtered_force,
                    local_base=local_base,
                )
                if recovery_mode_active:
                    correction_owner = "recover"
                    local_out[:6] = force_delta
                elif state == Coarse2ContactContactState.TOUCH:
                    touch_delta, recovery_primitive = self.recovery.touch_slowdown(local_base)
                    force_delta = touch_delta
                    recovery_reason = "touch"
                    correction_owner = "touch_slowdown"
                    local_out[:6] = local_base[:6] + force_delta
                elif state == Coarse2ContactContactState.EDGE:
                    edge_delta, recovery_primitive = self.recovery.edge_relief(filtered_force)
                    force_delta = edge_delta
                    recovery_reason = "edge"
                    correction_owner = "edge_relief"
                    local_out[:6] = local_base[:6] + force_delta
                elif state == Coarse2ContactContactState.PARTIAL:
                    force_delta, recovery_primitive = self.recovery.unload()
                    recovery_reason = "partial"
                    correction_owner = "partial_unload"
                    local_out[:6] = local_base[:6] + force_delta
                else:
                    local_out = local_base.copy()
            elif use_visual and estimate.valid and estimate.confidence >= 0.30:
                visual_delta[0] = estimate.dx
                visual_delta[1] = estimate.dy
                visual_delta[5] = estimate.dyaw
                correction_owner = "contact_visual"
                local_out[:6] = local_base[:6] + visual_delta
        elif self.phase == Coarse2ContactPhase.RECOVER:
            force_delta, recovery_primitive, recovery_reason, recovery_mode_active = self.force_reflex.correction(
                contact_state=state,
                force_reading=filtered_force,
                local_base=local_base,
            )
            correction_owner = "recover"
            if not recovery_mode_active and not np.allclose(force_delta, 0.0):
                local_out[:6] = local_base[:6] + force_delta
            else:
                local_out[:6] = force_delta
                recovery_mode_active = True
        else:
            local_out = local_base.copy()

        if self.shadow_only:
            local_out = local_base.copy()
            visual_delta[:] = 0.0
            force_delta[:] = 0.0
            recovery_primitive = "none"
            recovery_reason = "shadow_only"
            correction_owner = "shadow"
            recovery_mode_active = False
            applied = False
        else:
            applied = not np.allclose(local_out, local_base)

        world_out = local_delta_to_world(local_out, quat).astype(np.float32)
        out = base.copy()
        out[:6] = world_out
        pre_clip_world = out[:6].copy()
        out = self.safety.clip_final_action(out)
        post_clip_world = out[:6].copy()

        if applied:
            self.correction_count += 1
        if use_visual and np.linalg.norm(visual_delta) > 0.0 and not self.shadow_only:
            self.visual_apply_count += 1
        if use_force and np.linalg.norm(force_delta) > 0.0 and not self.shadow_only:
            self.force_apply_count += 1
        if recovery_mode_active and not self.shadow_only:
            self.recovery_count += 1

        self._last_trace = {
            "coarse2contact_phase": self.phase.name,
            "coarse2contact_phase_id": int(self.phase),
            "coarse2contact_mode": self.mode,
            "coarse2contact_shadow_only": bool(self.shadow_only),
            "uses_privileged_target": False,
            "phase_owner": str(correction_owner),
            "phase_reason": str(recovery_reason if recovery_reason != "idle" else self.contact.reason),
            "planner_reaches_precontact": bool(precontact),
            "planner_reaches_preinsert": bool(visual_ready),
            "visual_ready_for_contact": bool(visual_ready),
            "depth_conf": float(estimate.confidence),
            "depth_obs_quality": float(
                np.clip(
                    0.40 * float(estimate.confidence)
                    + 0.25 * float(estimate.mask_fraction)
                    + 0.15 * float(estimate.mask_fraction)
                    + 0.20 * float(1.0 - min(abs(float(estimate.z_gap)) / max(self.visual.precontact_depth_threshold, 1.0e-6), 1.0)),
                    0.0,
                    1.0,
                )
            ),
            "visual_error_xy": float(estimate.xy_error),
            "visual_error_z": float(estimate.z_gap),
            "visual_error_yaw": float(estimate.yaw_error),
            "visual_localizer_dx": float(getattr(estimate, "dx", 0.0)),
            "visual_localizer_dy": float(getattr(estimate, "dy", 0.0)),
            "visual_localizer_dyaw": float(getattr(estimate, "dyaw", 0.0)),
            "visual_localizer_confidence": float(estimate.confidence),
            "visual_confidence": float(estimate.confidence),
            "visual_reason": str(estimate.reason),
            "visual_mask_fraction": float(estimate.mask_fraction),
            "visual_centroid_u": float(estimate.centroid_u),
            "visual_centroid_v": float(estimate.centroid_v),
            "visual_correction_local_6d": visual_delta.astype(np.float32).tolist(),
            "contact_state": state.name.lower(),
            "contact_state_id": int(state),
            "force_reflex_reason": str(recovery_reason if recovery_reason != "idle" else self.contact.reason),
            "recovery_phase": self.force_reflex.phase.name,
            "recovery_phase_id": int(self.force_reflex.phase),
            "recovery_primitive": str(recovery_primitive),
            "local_correction_owner": str(correction_owner),
            "planner_action_world": base[:6].astype(np.float32).tolist(),
            "pre_clip_action_world": pre_clip_world.astype(np.float32).tolist(),
            "post_clip_action_world": post_clip_world.astype(np.float32).tolist(),
            "executed_action_world": post_clip_world.astype(np.float32).tolist(),
            "local_correction_applied": bool(applied),
            "force_correction_local_6d": force_delta.astype(np.float32).tolist(),
            "local_correction_local_6d": local_out.astype(np.float32).tolist(),
            "planner_chunk_local_6d": local_base.astype(np.float32).tolist(),
            "final_action_local_6d": local_out.astype(np.float32).tolist(),
            "pre_clip_action_world_6d": pre_clip_world.astype(np.float32).tolist(),
            "post_clip_action_world_6d": post_clip_world.astype(np.float32).tolist(),
            "raw_wrench_6d": raw_force.astype(np.float32).tolist(),
            "filtered_wrench_6d": filtered_force.astype(np.float32).tolist(),
            "raw_wrench": raw_force.astype(np.float32).tolist(),
            "filtered_wrench": filtered_force.astype(np.float32).tolist(),
            "force_filter_delta_6d": force_filter_delta.astype(np.float32).tolist(),
            "force_filter_low_force_update": bool(force_low_force),
            "invalid_action_flag": False,
            "retry_id": int(self.force_reflex.state_machine.recovery_cycles),
            "mp4_path": None,
        }
        return out

    def get_last_trace(self) -> dict:
        return dict(self._last_trace)

    def get_stats(self) -> dict:
        stats = dict(self.safety.get_stats())
        stats.update(
            {
                "coarse2contact_mode": self.mode,
                "coarse2contact_phase": self.phase.name,
                "coarse2contact_correction_count": int(self.correction_count),
                "coarse2contact_visual_apply_count": int(self.visual_apply_count),
                "coarse2contact_force_apply_count": int(self.force_apply_count),
                "coarse2contact_recovery_count": int(self.recovery_count),
                "coarse2contact_precontact_count": int(self.precontact_count),
                "coarse2contact_preinsert_count": int(self.preinsert_count),
                "coarse2contact_jam_count": int(self.contact.jam_count),
                "coarse2contact_force_spike_count": int(self.contact.force_spike_count),
                "coarse2contact_invalid_action_count": int(self.invalid_action_count),
                "recovery_phase": self.force_reflex.phase.name,
                "recovery_reason": self.force_reflex.reason,
                "recovery_primitive": self.force_reflex.last_primitive,
                "uses_privileged_target": False,
                "raw_wrench": self.wrench_filter._last_raw.astype(np.float32).tolist(),
                "filtered_wrench": self.wrench_filter._filtered.astype(np.float32).tolist(),
                "wrench_filter_stats": self.wrench_filter.get_stats(),
            }
        )
        stats.update(self._last_trace)
        return stats


# Backward-compatible aliases for the earlier single-file scaffold.
Coarse2ContactState = Coarse2ContactContactState
Coarse2ContactController = Coarse2ContactSupervisor
