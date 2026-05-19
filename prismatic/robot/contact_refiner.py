"""
contact_refiner.py

Orchestrates planner + residual + safety for each control step.
"""

from typing import Optional

import numpy as np

from prismatic.robot.contact_trigger import ContactPhase, ContactTrigger
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from prismatic.robot.residual_safety import ResidualSafety


class ContactRefiner:
    """Per-step closed-loop refiner wrapping rule reflex and/or learned residual."""

    def __init__(
        self,
        mode: str = "rule_reflex",
        residual_controller=None,
        force_contact_threshold: float = 0.5,
        force_jam_threshold: float = 3.0,
        depth_proximity_threshold: float = 0.15,
        z_pre_contact: float = 0.90,
        max_residual_pos: float = 0.005,
        max_residual_rot: float = 0.03,
        force_stop_threshold: float = 5.0,
        backoff_distance: float = 0.002,
        backoff_force_threshold: float = 3.0,
        workspace_min: Optional[list] = None,
        workspace_max: Optional[list] = None,
    ):
        assert mode in ("planner_only", "rule_reflex", "learned_residual", "full"), f"Unknown mode: {mode}"
        self.mode = mode
        self.residual_controller = residual_controller

        self.trigger = ContactTrigger(
            force_contact_threshold=force_contact_threshold,
            force_jam_threshold=force_jam_threshold,
            depth_proximity_threshold=depth_proximity_threshold,
            z_pre_contact=z_pre_contact,
        )

        self.safety = ResidualSafety(
            max_residual_pos=max_residual_pos,
            max_residual_rot=max_residual_rot,
            force_stop_threshold=force_stop_threshold,
            backoff_distance=backoff_distance,
            backoff_force_threshold=backoff_force_threshold,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
        )

        self.correction_count = 0
        self.replan_count = 0
        self.alpha_sum = 0.0
        self.residual_norm_sum = 0.0

    @staticmethod
    def compute_depth_proximity(wrist_depth) -> Optional[float]:
        if wrist_depth is None:
            return None
        if hasattr(wrist_depth, "detach"):
            wrist_depth = wrist_depth.detach().float().cpu().numpy()
        depth_arr = np.asarray(wrist_depth, dtype=np.float32).squeeze()
        valid = depth_arr[np.isfinite(depth_arr)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, 5.0))

    def step(
        self,
        a_base_7d: np.ndarray,
        step_idx: int,
        force_reading: Optional[np.ndarray] = None,
        gripper_z: Optional[float] = None,
        wrist_depth=None,
        ft_hist=None,
        proprio=None,
    ) -> np.ndarray:
        if self.mode == "planner_only":
            return a_base_7d.copy()

        phase = self.trigger.update(
            force_reading=force_reading,
            gripper_z=gripper_z,
            depth_proximity=self.compute_depth_proximity(wrist_depth),
        )

        if self.safety.check_force_stop(force_reading):
            a_stop = np.zeros(7, dtype=np.float32)
            a_stop[6] = a_base_7d[6]
            return a_stop

        a_exec = a_base_7d.copy()

        if self.mode in ("learned_residual", "full") and self.residual_controller is not None:
            if phase in (ContactPhase.PRE_CONTACT, ContactPhase.CONTACT):
                import torch

                device = next(self.residual_controller.parameters()).device
                dtype = next(self.residual_controller.parameters()).dtype

                wd = torch.zeros(1, 1, 96, 96, device=device, dtype=dtype)
                if wrist_depth is not None:
                    if isinstance(wrist_depth, np.ndarray):
                        wrist_depth = torch.from_numpy(wrist_depth)
                    wd = wrist_depth.unsqueeze(0).to(device=device, dtype=dtype)

                fh = torch.zeros(1, 32, 6, device=device, dtype=dtype)
                if ft_hist is not None:
                    if isinstance(ft_hist, np.ndarray):
                        ft_hist = torch.from_numpy(ft_hist)
                    fh = ft_hist.unsqueeze(0).to(device=device, dtype=dtype)

                pr = torch.zeros(1, 15, device=device, dtype=dtype)
                if proprio is not None:
                    if isinstance(proprio, np.ndarray):
                        proprio = torch.from_numpy(proprio)
                    pr = proprio.unsqueeze(0).to(device=device, dtype=dtype)

                current_quat = self._extract_current_quat(proprio)
                base_action_local = world_delta_to_local(a_base_7d[:6], current_quat)
                ba = torch.from_numpy(base_action_local.copy()).unsqueeze(0).to(device=device, dtype=dtype)
                si = torch.tensor([step_idx], device=device, dtype=torch.long)

                with torch.no_grad():
                    outputs = self.residual_controller(wd, fh, pr, ba, si, return_aux=True)

                delta_pose_local = outputs["delta_pose_gated"].squeeze(0).float().cpu().numpy()
                alpha = float(outputs["alpha"].squeeze(0).float().cpu().item())
                delta_pose_local = self.safety.clip_residual(delta_pose_local)
                delta_pose_world = local_delta_to_world(delta_pose_local, current_quat)
                a_exec[:6] = a_exec[:6] + delta_pose_world
                self.correction_count += 1
                self.alpha_sum += alpha
                self.residual_norm_sum += float(np.linalg.norm(delta_pose_world[:3]))

        if self.mode in ("rule_reflex", "full"):
            reflex_adjust = self.safety.compute_reflex_override(force_reading)
            if reflex_adjust is not None:
                a_exec[:6] = a_exec[:6] + reflex_adjust

        a_exec = self.safety.clip_final_action(a_exec)
        return a_exec

    @staticmethod
    def _extract_current_quat(proprio) -> np.ndarray:
        if proprio is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        arr = np.asarray(proprio, dtype=np.float32)
        if arr.shape[0] >= 14:
            return arr[10:14].copy()
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def should_replan(self) -> bool:
        if self.mode == "planner_only":
            return False
        return self.trigger.should_replan()

    def note_replan(self):
        self.replan_count += 1

    def on_invalid_action(
        self,
        base_action_7d: Optional[np.ndarray] = None,
        force_reading: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Enter a replan path and return a conservative recovery delta."""
        self.replan_count += 1
        return self.safety.compute_invalid_action_recovery(base_action_7d, force_reading)

    def get_chunk_size(self) -> int:
        if self.mode == "planner_only":
            return 8
        return self.trigger.get_chunk_size()

    def reset(self):
        self.trigger.reset()
        self.safety.reset_counters()
        self.correction_count = 0
        self.replan_count = 0
        self.alpha_sum = 0.0
        self.residual_norm_sum = 0.0

    def get_stats(self) -> dict:
        stats = self.safety.get_stats()
        stats["phase"] = self.trigger.phase.name
        stats["phase_age"] = self.trigger.phase_age
        stats["jam_detected"] = self.trigger.jam_detected
        stats["correction_count"] = self.correction_count
        stats["replan_count"] = self.replan_count
        stats["alpha_mean"] = self.alpha_sum / max(self.correction_count, 1)
        stats["residual_pos_norm_mean"] = self.residual_norm_sum / max(self.correction_count, 1)
        return stats
