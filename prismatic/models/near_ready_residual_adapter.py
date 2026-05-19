"""
near_ready_residual_adapter.py

Lightweight candidate-score residual for late open-phase / near-ready geometry
correction. It is designed to be added on top of a frozen baseline pose-field
scorer without changing group / step-scale / ready timing heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.residual_encoders import BaseActionEncoder


class NearReadyResidualScoreAdapter(nn.Module):
    def __init__(
        self,
        action_dim: int = 32,
        delta_dim: int = 32,
        sign_dim: int = 4,
        basin_bin_dim: int = 4,
        gripper_dim: int = 16,
        context_dim: int = 16,
        hidden_dim: int = 64,
        clip_rho: float = 0.35,
    ):
        super().__init__()
        self.clip_rho = float(clip_rho)
        self.delta_encoder = nn.Sequential(
            nn.Linear(6, delta_dim),
            nn.ReLU(inplace=True),
            nn.Linear(delta_dim, delta_dim),
        )
        self.dx_sign_embedding = nn.Embedding(3, sign_dim)
        self.dy_sign_embedding = nn.Embedding(3, sign_dim)
        self.dyaw_sign_embedding = nn.Embedding(3, sign_dim)
        self.basin_bin_embedding = nn.Embedding(4, basin_bin_dim)
        self.gripper_encoder = nn.Sequential(
            nn.Linear(3, gripper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gripper_dim, gripper_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
        )
        self.candidate_encoder = BaseActionEncoder(action_dim=6, out_dim=action_dim)
        state_dim = delta_dim + sign_dim * 3 + basin_bin_dim + gripper_dim + context_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self._zero_init_last()

    def _zero_init_last(self) -> None:
        last = self.score_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        candidate_actions: torch.Tensor,
        gripper_context: torch.Tensor,
        phase_age: torch.Tensor,
        steps_since_last_replan: torch.Tensor,
        current_delta_basin_target: torch.Tensor,
        current_dx_sign: torch.Tensor,
        current_dy_sign: torch.Tensor,
        current_dyaw_sign: torch.Tensor,
        basin_distance_bin: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, num_cands, _ = candidate_actions.shape
        context = torch.stack(
            [
                torch.clamp(phase_age.to(candidate_actions.dtype), min=0.0, max=32.0) / 32.0,
                torch.clamp(steps_since_last_replan.to(candidate_actions.dtype), min=0.0, max=32.0) / 32.0,
            ],
            dim=-1,
        )
        h_delta = self.delta_encoder(current_delta_basin_target.to(candidate_actions.dtype))
        h_dx = self.dx_sign_embedding(torch.clamp(current_dx_sign.long() + 1, min=0, max=2))
        h_dy = self.dy_sign_embedding(torch.clamp(current_dy_sign.long() + 1, min=0, max=2))
        h_dyaw = self.dyaw_sign_embedding(torch.clamp(current_dyaw_sign.long() + 1, min=0, max=2))
        h_bin = self.basin_bin_embedding(torch.clamp(basin_distance_bin.long(), min=0, max=3))
        h_grip = self.gripper_encoder(gripper_context.to(candidate_actions.dtype))
        h_ctx = self.context_encoder(context)
        state_hidden = self.state_mlp(torch.cat([h_delta, h_dx, h_dy, h_dyaw, h_bin, h_grip, h_ctx], dim=-1))
        cand_flat = candidate_actions.reshape(bsz * num_cands, 6)
        h_cand = self.candidate_encoder(cand_flat).reshape(bsz, num_cands, -1)
        state_expand = state_hidden.unsqueeze(1).expand(-1, num_cands, -1)
        residual = self.score_head(torch.cat([state_expand, h_cand], dim=-1)).squeeze(-1)
        residual = torch.clamp(residual, min=-self.clip_rho, max=self.clip_rho)
        if candidate_mask is not None:
            residual = residual.masked_fill(candidate_mask.to(dtype=torch.bool) <= 0, 0.0)
        return residual


class NearReadyGroupResidualAdapter(nn.Module):
    """
    Near-ready group-logit residual adapter.

    It does not replace the baseline group head; it only predicts a bounded
    residual delta-g that is added to frozen baseline group logits inside a
    near-ready runtime gate.
    """

    def __init__(
        self,
        num_groups: int = 37,
        delta_dim: int = 32,
        sign_dim: int = 4,
        basin_bin_dim: int = 4,
        gripper_dim: int = 16,
        context_dim: int = 16,
        hidden_dim: int = 64,
        clip_rho: float = 0.35,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.clip_rho = float(clip_rho)
        self.delta_encoder = nn.Sequential(
            nn.Linear(6, delta_dim),
            nn.ReLU(inplace=True),
            nn.Linear(delta_dim, delta_dim),
        )
        self.dx_sign_embedding = nn.Embedding(3, sign_dim)
        self.dy_sign_embedding = nn.Embedding(3, sign_dim)
        self.dyaw_sign_embedding = nn.Embedding(3, sign_dim)
        self.basin_bin_embedding = nn.Embedding(4, basin_bin_dim)
        self.gripper_encoder = nn.Sequential(
            nn.Linear(3, gripper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gripper_dim, gripper_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
        )
        state_dim = delta_dim + sign_dim * 3 + basin_bin_dim + gripper_dim + context_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.group_residual_head = nn.Linear(hidden_dim, self.num_groups)
        self._zero_init_last()

    def _zero_init_last(self) -> None:
        nn.init.zeros_(self.group_residual_head.weight)
        nn.init.zeros_(self.group_residual_head.bias)

    def forward(
        self,
        gripper_context: torch.Tensor,
        phase_age: torch.Tensor,
        steps_since_last_replan: torch.Tensor,
        current_delta_basin_target: torch.Tensor,
        current_dx_sign: torch.Tensor,
        current_dy_sign: torch.Tensor,
        current_dyaw_sign: torch.Tensor,
        basin_distance_bin: torch.Tensor,
        group_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = torch.stack(
            [
                torch.clamp(phase_age.to(current_delta_basin_target.dtype), min=0.0, max=32.0) / 32.0,
                torch.clamp(steps_since_last_replan.to(current_delta_basin_target.dtype), min=0.0, max=32.0) / 32.0,
            ],
            dim=-1,
        )
        h_delta = self.delta_encoder(current_delta_basin_target)
        h_dx = self.dx_sign_embedding(torch.clamp(current_dx_sign.long() + 1, min=0, max=2))
        h_dy = self.dy_sign_embedding(torch.clamp(current_dy_sign.long() + 1, min=0, max=2))
        h_dyaw = self.dyaw_sign_embedding(torch.clamp(current_dyaw_sign.long() + 1, min=0, max=2))
        h_bin = self.basin_bin_embedding(torch.clamp(basin_distance_bin.long(), min=0, max=3))
        h_grip = self.gripper_encoder(gripper_context.to(current_delta_basin_target.dtype))
        h_ctx = self.context_encoder(context)
        state_hidden = self.state_mlp(torch.cat([h_delta, h_dx, h_dy, h_dyaw, h_bin, h_grip, h_ctx], dim=-1))
        residual = self.group_residual_head(state_hidden)
        residual = torch.clamp(residual, min=-self.clip_rho, max=self.clip_rho)
        if group_valid_mask is not None:
            residual = residual.masked_fill(group_valid_mask.to(dtype=torch.bool) <= 0, 0.0)
        return residual
