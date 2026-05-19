"""
Depth-force local contact policy.

This module is intentionally independent from the legacy alignment stack
(`StudentHandoffStateHeadV2`, pose-field scorer, B1/B2). It predicts values for
runtime-safe local candidate actions plus conservative auxiliary heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.residual_encoders import (
    BaseActionEncoder,
    DepthEncoderTiny,
    ForceEncoderTiny,
    ProprioEncoder,
    RGBEncoderTiny,
)


@dataclass(frozen=True)
class DepthForceCandidateBankConfig:
    xyz_steps_m: tuple[float, ...] = (0.001, 0.002, 0.004)
    z_steps_m: tuple[float, ...] = (0.001, 0.002)
    yaw_steps_rad: tuple[float, ...] = (
        math.radians(1.0),
        math.radians(2.0),
        math.radians(4.0),
    )
    include_mixed_xy_yaw: bool = True
    include_backoff: bool = True


def build_depth_force_candidate_bank(
    cfg: DepthForceCandidateBankConfig | None = None,
) -> torch.Tensor:
    """Return a fixed end-effector-local candidate bank, including keep-baseline."""
    cfg = cfg or DepthForceCandidateBankConfig()
    actions: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    for step in cfg.xyz_steps_m:
        actions.extend(
            [
                [step, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-step, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, step, 0.0, 0.0, 0.0, 0.0],
                [0.0, -step, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    for step in cfg.z_steps_m:
        actions.extend(
            [
                [0.0, 0.0, step, 0.0, 0.0, 0.0],
                [0.0, 0.0, -step, 0.0, 0.0, 0.0],
            ]
        )
    for step in cfg.yaw_steps_rad:
        actions.extend(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, step],
                [0.0, 0.0, 0.0, 0.0, 0.0, -step],
            ]
        )

    if cfg.include_mixed_xy_yaw:
        xy = cfg.xyz_steps_m[1] if len(cfg.xyz_steps_m) > 1 else cfg.xyz_steps_m[0]
        yaw = cfg.yaw_steps_rad[1] if len(cfg.yaw_steps_rad) > 1 else cfg.yaw_steps_rad[0]
        for sx in (-1.0, 1.0):
            for syaw in (-1.0, 1.0):
                actions.append([sx * xy, 0.0, 0.0, 0.0, 0.0, syaw * yaw])
                actions.append([0.0, sx * xy, 0.0, 0.0, 0.0, syaw * yaw])

    if cfg.include_backoff:
        actions.extend(
            [
                [0.0, 0.0, 0.002, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.004, 0.0, 0.0, 0.0],
            ]
        )

    return torch.tensor(actions, dtype=torch.float32)


class DepthForceLocalContactPolicy(nn.Module):
    """Candidate-ranking policy for local contact alignment."""

    def __init__(
        self,
        proprio_input_dim: int = 15,
        force_input_dim: int = 6,
        stage_vocab_size: int = 8,
        contact_phase_vocab_size: int = 4,
        state_dim: int = 384,
        candidate_dim: int = 96,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.front_encoder = RGBEncoderTiny(out_dim=96)
        self.wrist_encoder = RGBEncoderTiny(out_dim=96)
        self.depth_encoder = DepthEncoderTiny(out_dim=96)
        self.force_encoder = ForceEncoderTiny(force_dim=force_input_dim, out_dim=64)
        self.proprio_encoder = ProprioEncoder(proprio_dim=proprio_input_dim, out_dim=64)
        self.action_encoder = BaseActionEncoder(action_dim=6, out_dim=48)
        self.stage_embedding = nn.Embedding(stage_vocab_size, 16)
        self.contact_embedding = nn.Embedding(contact_phase_vocab_size, 8)
        self.scalar_encoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU(inplace=True), nn.Linear(16, 16))

        fused_dim = 96 + 96 + 96 + 64 + 64 + 48 + 16 + 8 + 16
        self.state_trunk = nn.Sequential(
            nn.Linear(fused_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(inplace=True),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(6, candidate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(candidate_dim, candidate_dim),
            nn.ReLU(inplace=True),
        )
        self.value_head = nn.Sequential(
            nn.Linear(state_dim + candidate_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.geometry_value_head = nn.Sequential(
            nn.Linear(state_dim + candidate_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.risk_value_head = nn.Sequential(
            nn.Linear(state_dim + candidate_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.switch_head = nn.Linear(state_dim, 1)
        self.contact_risk_head = nn.Linear(state_dim, 4)
        self.mode_head = nn.Linear(state_dim, 5)
        self.progress_head = nn.Linear(state_dim, 1)
        self.residual_aux_head = nn.Linear(state_dim, 6)

        # Start conservative: prefer keep-baseline until trained.
        nn.init.constant_(self.switch_head.bias, -2.0)
        nn.init.zeros_(self.residual_aux_head.weight)
        nn.init.zeros_(self.residual_aux_head.bias)

    @staticmethod
    def _resize_rgb(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected RGB tensor (B,3,H,W), got {tuple(x.shape)}")
        if x.shape[-2:] != (128, 128):
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _resize_depth(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[-2:] != (96, 96):
            x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        return x

    def encode_state(
        self,
        front_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_base_action_local: torch.Tensor,
        stage_token: torch.Tensor | None = None,
        contact_phase: torch.Tensor | None = None,
        depth_proximity: torch.Tensor | None = None,
        gripper_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = planner_base_action_local.shape[0]
        if stage_token is None:
            stage_token = torch.zeros(bsz, device=planner_base_action_local.device, dtype=torch.long)
        if contact_phase is None:
            contact_phase = torch.zeros(bsz, device=planner_base_action_local.device, dtype=torch.long)
        if depth_proximity is None:
            depth_proximity = torch.zeros(bsz, device=planner_base_action_local.device, dtype=planner_base_action_local.dtype)
        if gripper_state is None:
            gripper_state = torch.zeros(bsz, device=planner_base_action_local.device, dtype=planner_base_action_local.dtype)

        scalar = torch.stack(
            [
                torch.nan_to_num(depth_proximity.to(planner_base_action_local.dtype), nan=0.0, posinf=1.0),
                torch.clamp(gripper_state.to(planner_base_action_local.dtype), 0.0, 1.0),
                torch.linalg.norm(force_history[:, -1, :3].to(planner_base_action_local.dtype), dim=-1),
            ],
            dim=-1,
        )
        fused = torch.cat(
            [
                self.front_encoder(self._resize_rgb(front_rgb)),
                self.wrist_encoder(self._resize_rgb(wrist_rgb)),
                self.depth_encoder(self._resize_depth(wrist_depth)),
                self.force_encoder(force_history),
                self.proprio_encoder(proprio),
                self.action_encoder(planner_base_action_local),
                self.stage_embedding(torch.clamp(stage_token.long(), min=0, max=self.stage_embedding.num_embeddings - 1)),
                self.contact_embedding(torch.clamp(contact_phase.long(), min=0, max=self.contact_embedding.num_embeddings - 1)),
                self.scalar_encoder(scalar),
            ],
            dim=-1,
        )
        return self.state_trunk(fused)

    def forward(
        self,
        front_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_base_action_local: torch.Tensor,
        candidate_actions_local: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        stage_token: torch.Tensor | None = None,
        contact_phase: torch.Tensor | None = None,
        depth_proximity: torch.Tensor | None = None,
        gripper_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_state(
            front_rgb=front_rgb,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            force_history=force_history,
            proprio=proprio,
            planner_base_action_local=planner_base_action_local,
            stage_token=stage_token,
            contact_phase=contact_phase,
            depth_proximity=depth_proximity,
            gripper_state=gripper_state,
        )
        cand = self.candidate_encoder(candidate_actions_local)
        state_exp = state[:, None, :].expand(-1, cand.shape[1], -1)
        pair = torch.cat([state_exp, cand], dim=-1)
        value = self.value_head(pair).squeeze(-1)
        geometry_value = self.geometry_value_head(pair).squeeze(-1)
        risk_value = self.risk_value_head(pair).squeeze(-1)
        if candidate_mask is not None:
            value = value.masked_fill(candidate_mask <= 0.5, -1e9)
            geometry_value = geometry_value.masked_fill(candidate_mask <= 0.5, -1e9)
            risk_value = risk_value.masked_fill(candidate_mask <= 0.5, -1e9)
        return {
            "candidate_value": value,
            "candidate_total_value": value,
            "candidate_geometry_value": geometry_value,
            "candidate_risk_value": risk_value,
            "switch_logit": self.switch_head(state).squeeze(-1),
            "switch_prob": torch.sigmoid(self.switch_head(state).squeeze(-1)),
            "contact_risk_logits": self.contact_risk_head(state),
            "mode_logits": self.mode_head(state),
            "progress_delta": self.progress_head(state).squeeze(-1),
            "residual_aux": self.residual_aux_head(state),
            "state_latent": state,
        }
