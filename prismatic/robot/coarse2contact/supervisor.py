"""Non-privileged Coarse2Contact runtime controller.

This module deliberately avoids teacher-oracle targets and legacy
alignment/student assets. It wraps frozen VLA planner actions with small,
auditable local corrections from wrist depth and force/torque signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

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


class DepthVisualAligner:
    """Conservative wrist-depth proxy for pre-contact correction."""

    def __init__(
        self,
        *,
        precontact_depth_threshold: float = 0.20,
        contact_depth_threshold: float = 0.035,
        xy_threshold: float = 0.0015,
        yaw_threshold: float = 0.0349,
        max_xy_step: float = 0.0005,
        max_z_step: float = 0.0005,
        max_yaw_step: float = 0.0087,
        pixel_to_meter: float = 0.004,
        min_mask_fraction: float = 0.002,
    ) -> None:
        self.precontact_depth_threshold = float(precontact_depth_threshold)
        self.contact_depth_threshold = float(contact_depth_threshold)
        self.xy_threshold = float(xy_threshold)
        self.yaw_threshold = float(yaw_threshold)
        self.max_xy_step = float(max_xy_step)
        self.max_z_step = float(max_z_step)
        self.max_yaw_step = float(max_yaw_step)
        self.pixel_to_meter = float(pixel_to_meter)
        self.min_mask_fraction = float(min_mask_fraction)

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
        return float(np.percentile(valid, 5.0))

    def estimate(self, wrist_depth) -> VisualAlignmentEstimate:
        if wrist_depth is None:
            return self._empty("missing_depth")
        if hasattr(wrist_depth, "detach"):
            wrist_depth = wrist_depth.detach().float().cpu().numpy()
        depth = np.asarray(wrist_depth, dtype=np.float32).squeeze()
        if depth.ndim != 2 or depth.size == 0:
            return self._empty("bad_depth_shape")
        finite = np.isfinite(depth)
        if not np.any(finite):
            return self._empty("no_valid_depth")

        prox = self.depth_proximity(depth)
        if not np.isfinite(prox):
            return self._empty("no_proximity")
        near_cut = min(float(np.percentile(depth[finite], 15.0)), prox + 0.030)
        mask = np.logical_and(finite, depth <= near_cut)
        mask_fraction = float(np.mean(mask))
        if mask_fraction < self.min_mask_fraction:
            return self._empty("weak_depth_mask", z_gap=prox, mask_fraction=mask_fraction)

        ys, xs = np.nonzero(mask)
        h, w = depth.shape
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        u = (cx - (w - 1) * 0.5) / max((w - 1) * 0.5, 1.0)
        v = (cy - (h - 1) * 0.5) / max((h - 1) * 0.5, 1.0)
        xy_error = float(np.sqrt(u * u + v * v) * self.pixel_to_meter)

        yaw_error = 0.0
        if xs.size >= 8:
            pts = np.stack([xs.astype(np.float32) - cx, ys.astype(np.float32) - cy], axis=1)
            cov = np.cov(pts.T)
            try:
                vals, vecs = np.linalg.eigh(cov)
                major = vecs[:, int(np.argmax(vals))]
                angle = float(np.arctan2(major[1], major[0]))
                yaw_error = float(((angle + np.pi / 4.0) % (np.pi / 2.0)) - np.pi / 4.0)
            except Exception:
                yaw_error = 0.0

        correction = np.zeros(6, dtype=np.float32)
        correction[0] = float(np.clip(-u * self.pixel_to_meter, -self.max_xy_step, self.max_xy_step))
        correction[1] = float(np.clip(-v * self.pixel_to_meter, -self.max_xy_step, self.max_xy_step))
        if prox > self.contact_depth_threshold:
            correction[2] = -min(self.max_z_step, max(0.0, prox - self.contact_depth_threshold))
        correction[5] = float(np.clip(-0.25 * yaw_error, -self.max_yaw_step, self.max_yaw_step))
        if xy_error > max(self.xy_threshold * 3.0, 1e-6) or prox > self.contact_depth_threshold * 2.0:
            correction[5] *= 0.35

        conf_area = min(mask_fraction / max(self.min_mask_fraction * 8.0, 1e-6), 1.0)
        conf_depth = 1.0 if prox <= self.precontact_depth_threshold else 0.25
        confidence = float(np.clip(0.15 + 0.70 * conf_area * conf_depth, 0.0, 1.0))
        return VisualAlignmentEstimate(
            valid=True,
            confidence=confidence,
            xy_error=xy_error,
            z_gap=float(prox),
            yaw_error=abs(float(yaw_error)),
            correction_local=correction,
            reason="ok",
            mask_fraction=mask_fraction,
            centroid_u=float(u),
            centroid_v=float(v),
        )

    @staticmethod
    def _empty(
        reason: str,
        *,
        z_gap: float = float("nan"),
        mask_fraction: float = 0.0,
    ) -> VisualAlignmentEstimate:
        return VisualAlignmentEstimate(
            valid=False,
            confidence=0.0,
            xy_error=float("nan"),
            z_gap=float(z_gap),
            yaw_error=float("nan"),
            correction_local=np.zeros(6, dtype=np.float32),
            reason=str(reason),
            mask_fraction=float(mask_fraction),
            centroid_u=float("nan"),
            centroid_v=float("nan"),
        )


class ContactStateEstimator:
    def __init__(
        self,
        *,
        contact_threshold: float = 1.0,
        contact_delta_threshold: float = 0.5,
        jam_threshold: float = 6.0,
        torque_threshold: float = 0.25,
        seated_depth_threshold: float = 0.012,
        progress_window: int = 8,
        progress_threshold: float = 0.0005,
    ) -> None:
        self.contact_threshold = float(contact_threshold)
        self.contact_delta_threshold = float(contact_delta_threshold)
        self.jam_threshold = float(jam_threshold)
        self.torque_threshold = float(torque_threshold)
        self.seated_depth_threshold = float(seated_depth_threshold)
        self.progress_window = int(progress_window)
        self.progress_threshold = float(progress_threshold)
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
        torque_xy = float(np.linalg.norm(f[3:5]))
        delta_force = 0.0 if self._prev_force_norm is None else force_norm - self._prev_force_norm
        self._prev_force_norm = force_norm
        if gripper_z is not None and np.isfinite(float(gripper_z)):
            self._z_hist.append(float(gripper_z))
            self._z_hist = self._z_hist[-self.progress_window :]
        z_progress = 0.0
        if len(self._z_hist) >= 2:
            z_progress = abs(float(self._z_hist[-1] - self._z_hist[0]))

        contact = bool(fz >= self.contact_threshold or delta_force >= self.contact_delta_threshold)
        jam = bool((fz >= self.jam_threshold and z_progress < self.progress_threshold) or torque_xy >= self.torque_threshold)
        if jam:
            self.state = Coarse2ContactContactState.JAM
            self.reason = "jam_force_or_torque"
            self.jam_count += 1
        elif force_norm >= self.jam_threshold * 1.75:
            self.state = Coarse2ContactContactState.JAM
            self.reason = "force_spike"
            self.force_spike_count += 1
        elif contact and np.isfinite(depth_gap) and depth_gap <= self.seated_depth_threshold:
            self.state = Coarse2ContactContactState.SEATED
            self.reason = "seated_depth_contact"
        elif contact and np.isfinite(depth_gap) and depth_gap > self.seated_depth_threshold * 2.0:
            self.state = Coarse2ContactContactState.PARTIAL
            self.reason = "contact_not_seated_yet"
        elif contact and np.isfinite(visual_xy_error) and visual_xy_error > 0.004:
            self.state = Coarse2ContactContactState.EDGE
            self.reason = "lateral_error_under_contact"
        elif contact:
            self.state = Coarse2ContactContactState.TOUCH
            self.reason = "contact_onset"
        else:
            self.state = Coarse2ContactContactState.FREE
            self.reason = "free"
        return self.state


class RecoveryPrimitiveBank:
    def __init__(self, *, backoff_m: float = 0.003, lateral_m: float = 0.0015, yaw_rad: float = 0.0262) -> None:
        self.backoff_m = float(backoff_m)
        self.lateral_m = float(lateral_m)
        self.yaw_rad = float(yaw_rad)
        self._toggle = 1.0

    def reset(self) -> None:
        self._toggle = 1.0

    def action(self, state: Coarse2ContactContactState, force_reading: Optional[np.ndarray]) -> tuple[np.ndarray, str]:
        out = np.zeros(6, dtype=np.float32)
        if state == Coarse2ContactContactState.JAM:
            out[2] = self.backoff_m
            out[0] = self._toggle * self.lateral_m
            self._toggle *= -1.0
            return out, "backoff_lateral_nudge"
        if state == Coarse2ContactContactState.EDGE:
            f = np.zeros(3, dtype=np.float32) if force_reading is None else np.asarray(force_reading, dtype=np.float32).reshape(-1)[:3]
            fxy = f[:2]
            mag = float(np.linalg.norm(fxy))
            if mag > 1e-6:
                out[:2] = (-fxy / mag * self.lateral_m).astype(np.float32)
            else:
                out[0] = self._toggle * self.lateral_m
                self._toggle *= -1.0
            out[5] = self._toggle * min(self.yaw_rad, 0.0087)
            return out, "edge_relief"
        return out, "none"


class ForceReflexController:
    def __init__(self, *, safety: ResidualSafety, recovery_bank: RecoveryPrimitiveBank) -> None:
        self.safety = safety
        self.recovery_bank = recovery_bank

    def correction(
        self,
        *,
        state: Coarse2ContactContactState,
        force_reading: Optional[np.ndarray],
        local_base: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        if self.safety.check_force_stop(force_reading):
            out = -np.asarray(local_base, dtype=np.float32).copy()
            return out, "force_stop"
        recovery, primitive = self.recovery_bank.action(state, force_reading)
        if primitive != "none":
            return recovery, primitive
        if state == Coarse2ContactContactState.TOUCH:
            out = np.zeros(6, dtype=np.float32)
            out[2] = min(float(local_base[2]), 0.0) * -0.60
            return out, "touch_slowdown"
        return np.zeros(6, dtype=np.float32), "none"


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
        max_z_step: float = 0.0005,
        max_yaw_step: float = 0.0087,
        force_contact_threshold: float = 1.0,
        force_delta_contact_threshold: float = 0.5,
        force_jam_threshold: float = 6.0,
        force_torque_threshold: float = 0.25,
        force_spike_threshold: float = 10.5,
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
            max_z_step=max_z_step,
            max_yaw_step=max_yaw_step,
        )
        self.contact = ContactStateEstimator(
            contact_threshold=force_contact_threshold,
            contact_delta_threshold=force_delta_contact_threshold,
            jam_threshold=force_jam_threshold,
            torque_threshold=force_torque_threshold,
            seated_depth_threshold=visual_contact_depth_threshold * 0.35,
        )
        self.recovery = RecoveryPrimitiveBank(backoff_m=backoff_m, lateral_m=lateral_m, yaw_rad=max_yaw_step)
        self.safety = ResidualSafety(
            max_residual_pos=max(max_xy_step, max_z_step, lateral_m, backoff_m),
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
        self._last_trace: dict = {}

    def reset(self) -> None:
        self.contact.reset()
        self.recovery.reset()
        self.safety.reset_counters()
        self.phase = Coarse2ContactPhase.COARSE
        self.correction_count = 0
        self.visual_apply_count = 0
        self.force_apply_count = 0
        self.recovery_count = 0
        self.precontact_count = 0
        self.preinsert_count = 0
        self._last_trace = {}

    def get_chunk_size(self) -> int:
        return max(1, self.chunk_size)

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

        estimate = self.visual.estimate(wrist_depth)
        state = self.contact.update(
            force_reading=force_reading,
            gripper_z=gripper_z,
            depth_gap=estimate.z_gap,
            visual_xy_error=estimate.xy_error,
        )
        precontact = bool(estimate.valid and np.isfinite(estimate.z_gap) and estimate.z_gap <= self.visual.precontact_depth_threshold)
        if precontact:
            self.precontact_count += 1
        visual_ready = bool(
            estimate.valid
            and estimate.xy_error <= self.visual.xy_threshold
            and estimate.yaw_error <= self.visual.yaw_threshold
            and estimate.z_gap <= self.visual.contact_depth_threshold
        )
        if visual_ready:
            self.preinsert_count += 1
        self._update_phase(precontact=precontact, visual_ready=visual_ready, contact_state=state)

        quat = self._extract_current_quat(proprio)
        local_base = world_delta_to_local(base[:6], quat)
        local_out = local_base.copy()
        visual_delta = np.zeros(6, dtype=np.float32)
        force_delta = np.zeros(6, dtype=np.float32)
        recovery_name = "none"
        applied = False

        use_visual = self.mode in ("depth_shadow", "depth_apply", "depth_force")
        use_force = self.mode in ("force_reflex", "depth_force")
        if use_visual and self.phase in (Coarse2ContactPhase.VISUAL_ALIGN, Coarse2ContactPhase.PROBE_CONTACT) and estimate.valid and estimate.confidence >= 0.20:
            visual_delta = estimate.correction_local.copy()
        if use_force and self.phase in (Coarse2ContactPhase.PROBE_CONTACT, Coarse2ContactPhase.CONTACT_INSERT, Coarse2ContactPhase.RECOVER):
            force_delta, recovery_name = self.force_reflex.correction(
                state=state,
                force_reading=force_reading,
                local_base=local_base,
            )

        correction = visual_delta + force_delta
        if self.shadow_only:
            correction[:] = 0.0
        elif np.linalg.norm(correction) > 0.0:
            local_out = local_out + correction
            applied = True

        world_out = local_delta_to_world(local_out, quat).astype(np.float32)
        out = base.copy()
        out[:6] = world_out
        out = self.safety.clip_final_action(out)

        if applied:
            self.correction_count += 1
        if use_visual and np.linalg.norm(visual_delta) > 0.0 and not self.shadow_only:
            self.visual_apply_count += 1
        if use_force and np.linalg.norm(force_delta) > 0.0 and not self.shadow_only:
            self.force_apply_count += 1
        if recovery_name not in ("none", "touch_slowdown") and not self.shadow_only:
            self.recovery_count += 1

        self._last_trace = {
            "coarse2contact_phase": self.phase.name,
            "coarse2contact_phase_id": int(self.phase),
            "coarse2contact_mode": self.mode,
            "coarse2contact_shadow_only": bool(self.shadow_only),
            "uses_privileged_target": False,
            "planner_reaches_precontact": bool(precontact),
            "planner_reaches_preinsert": bool(visual_ready),
            "visual_ready_for_contact": bool(visual_ready),
            "visual_error_xy": float(estimate.xy_error),
            "visual_error_z": float(estimate.z_gap),
            "visual_error_yaw": float(estimate.yaw_error),
            "visual_confidence": float(estimate.confidence),
            "visual_reason": str(estimate.reason),
            "visual_mask_fraction": float(estimate.mask_fraction),
            "visual_centroid_u": float(estimate.centroid_u),
            "visual_centroid_v": float(estimate.centroid_v),
            "visual_correction_local_6d": visual_delta.astype(np.float32).tolist(),
            "contact_state": state.name.lower(),
            "contact_state_id": int(state),
            "force_reflex_reason": str(self.contact.reason if recovery_name == "none" else recovery_name),
            "force_correction_local_6d": force_delta.astype(np.float32).tolist(),
            "recovery_primitive": str(recovery_name),
            "local_correction_applied": bool(applied),
            "local_correction_local_6d": correction.astype(np.float32).tolist(),
            "planner_chunk_local_6d": local_base.astype(np.float32).tolist(),
            "final_action_local_6d": local_out.astype(np.float32).tolist(),
            "mp4_path": None,
        }
        return out

    def _update_phase(
        self,
        *,
        precontact: bool,
        visual_ready: bool,
        contact_state: Coarse2ContactContactState,
    ) -> None:
        if contact_state == Coarse2ContactContactState.SEATED:
            self.phase = Coarse2ContactPhase.DONE
        elif contact_state == Coarse2ContactContactState.JAM:
            self.phase = Coarse2ContactPhase.FAIL if self.contact.jam_count > 3 else Coarse2ContactPhase.RECOVER
        elif contact_state in (Coarse2ContactContactState.TOUCH, Coarse2ContactContactState.EDGE, Coarse2ContactContactState.PARTIAL):
            self.phase = Coarse2ContactPhase.CONTACT_INSERT
        elif visual_ready:
            self.phase = Coarse2ContactPhase.PROBE_CONTACT
        elif precontact:
            self.phase = Coarse2ContactPhase.VISUAL_ALIGN
        else:
            self.phase = Coarse2ContactPhase.COARSE

    @staticmethod
    def _extract_current_quat(proprio) -> np.ndarray:
        if proprio is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        arr = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if arr.size >= 14:
            return arr[10:14].copy()
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

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
                "uses_privileged_target": False,
            }
        )
        stats.update(self._last_trace)
        return stats


# Backward-compatible aliases for the earlier single-file scaffold.
Coarse2ContactState = Coarse2ContactContactState
Coarse2ContactController = Coarse2ContactSupervisor
