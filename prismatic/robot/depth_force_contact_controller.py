"""
Shadow-only wrapper for the depth-force local contact policy.

This controller is deliberately separate from StageAwareRefiner. It does not
apply actions by default; it only scores local candidate actions and returns
diagnostics needed for shadow evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from prismatic.models.depth_force_contact_policy import (
    DepthForceCandidateBankConfig,
    DepthForceLocalContactPolicy,
    build_depth_force_candidate_bank,
)
from prismatic.robot.contact_trigger import ContactPhase, ContactTrigger
from prismatic.robot.residual_safety import ResidualSafety


@dataclass
class DepthForceContactDecision:
    gate_open: bool
    selected_index: int
    baseline_index: int
    switch_prob: float
    selected_action_local: np.ndarray
    candidate_values: np.ndarray
    contact_risk: int
    contact_phase: int
    depth_proximity: float
    force_stop: bool


class DepthForceContactController:
    """Independent depth/force contact policy runner."""

    def __init__(
        self,
        policy: DepthForceLocalContactPolicy,
        candidate_bank: Optional[np.ndarray] = None,
        device: str | torch.device = "cpu",
        shadow_only: bool = True,
        switch_threshold: float = 0.65,
        min_depth_valid: float = 0.0,
        force_stop_threshold: float = 5.0,
    ) -> None:
        self.policy = policy.to(device).eval()
        self.device = torch.device(device)
        self.shadow_only = bool(shadow_only)
        self.switch_threshold = float(switch_threshold)
        self.min_depth_valid = float(min_depth_valid)
        self.safety = ResidualSafety(force_stop_threshold=force_stop_threshold)
        self.contact_trigger = ContactTrigger()
        if candidate_bank is None:
            candidate_bank_t = build_depth_force_candidate_bank(DepthForceCandidateBankConfig())
            candidate_bank = candidate_bank_t.cpu().numpy()
        self.candidate_bank = np.asarray(candidate_bank, dtype=np.float32)
        if self.candidate_bank.ndim != 2 or self.candidate_bank.shape[1] != 6:
            raise ValueError(f"candidate_bank must have shape (N,6), got {self.candidate_bank.shape}")

    @staticmethod
    def compute_depth_proximity(wrist_depth) -> float:
        if wrist_depth is None:
            return float("nan")
        if hasattr(wrist_depth, "detach"):
            wrist_depth = wrist_depth.detach().float().cpu().numpy()
        depth = np.asarray(wrist_depth, dtype=np.float32).squeeze()
        valid = depth[np.isfinite(depth)]
        if valid.size == 0:
            return float("nan")
        return float(np.percentile(valid, 5.0))

    @staticmethod
    def _tensor(x, device, dtype=torch.float32):
        return torch.as_tensor(x, device=device, dtype=dtype)

    @torch.no_grad()
    def shadow_step(
        self,
        *,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        wrist_depth: np.ndarray,
        force_history: np.ndarray,
        proprio: np.ndarray,
        planner_base_action_local: np.ndarray,
        gripper_state: float = 1.0,
        stage_token: int = 0,
        candidate_bank: Optional[np.ndarray] = None,
        force_reading: Optional[np.ndarray] = None,
        gripper_z: Optional[float] = None,
    ) -> DepthForceContactDecision:
        depth_prox = self.compute_depth_proximity(wrist_depth)
        phase = self.contact_trigger.update(
            force_reading=force_reading,
            gripper_z=gripper_z,
            depth_proximity=depth_prox,
        )
        force_stop = self.safety.check_force_stop(force_reading)
        bank = self.candidate_bank if candidate_bank is None else np.asarray(candidate_bank, dtype=np.float32)
        mask = np.ones((bank.shape[0],), dtype=np.float32)

        fr = self._tensor(front_rgb.transpose(2, 0, 1)[None] / 255.0, self.device)
        wr = self._tensor(wrist_rgb.transpose(2, 0, 1)[None] / 255.0, self.device)
        wd = self._tensor(wrist_depth[None], self.device)
        if wd.ndim == 3:
            wd = wd.unsqueeze(1)
        fh = self._tensor(force_history[None], self.device)
        prop = self._tensor(proprio[None], self.device)
        base = self._tensor(planner_base_action_local[None, :6], self.device)
        cand = self._tensor(bank[None], self.device)
        cand_mask = self._tensor(mask[None], self.device)

        out = self.policy(
            front_rgb=fr,
            wrist_rgb=wr,
            wrist_depth=wd,
            force_history=fh,
            proprio=prop,
            planner_base_action_local=base,
            candidate_actions_local=cand,
            candidate_mask=cand_mask,
            stage_token=torch.tensor([stage_token], device=self.device),
            contact_phase=torch.tensor([int(phase)], device=self.device),
            depth_proximity=torch.tensor([0.0 if not np.isfinite(depth_prox) else depth_prox], device=self.device),
            gripper_state=torch.tensor([gripper_state], device=self.device),
        )
        values = out["candidate_value"][0].detach().cpu().numpy().astype(np.float32)
        selected = int(np.argmax(values))
        switch_prob = float(out["switch_prob"][0].detach().cpu())
        gate_open = (
            not force_stop
            and np.isfinite(depth_prox)
            and switch_prob >= self.switch_threshold
            and bank.shape[0] > 0
        )
        return DepthForceContactDecision(
            gate_open=bool(gate_open),
            selected_index=selected,
            baseline_index=0,
            switch_prob=switch_prob,
            selected_action_local=bank[selected].copy(),
            candidate_values=values,
            contact_risk=int(torch.argmax(out["contact_risk_logits"][0]).detach().cpu()),
            contact_phase=int(phase),
            depth_proximity=float(depth_prox),
            force_stop=bool(force_stop),
        )


def load_depth_force_contact_controller(
    ckpt_path: str,
    device: str | torch.device = "cpu",
    shadow_only: bool = True,
) -> DepthForceContactController:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = DepthForceLocalContactPolicy(**ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    bank = ckpt.get("candidate_bank", None)
    return DepthForceContactController(model, candidate_bank=bank, device=device, shadow_only=shadow_only)
