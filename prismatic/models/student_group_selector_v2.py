from __future__ import annotations

import torch
import torch.nn as nn


class StudentGroupSelectorV2(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        delta_dim: int = 32,
        sign_dim: int = 4,
        basin_bin_dim: int = 4,
        hidden_dim: int = 128,
        num_groups: int = 37,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.delta_encoder = nn.Sequential(
            nn.Linear(6, delta_dim),
            nn.ReLU(inplace=True),
            nn.Linear(delta_dim, delta_dim),
        )
        self.dx_sign_embedding = nn.Embedding(3, sign_dim)
        self.dy_sign_embedding = nn.Embedding(3, sign_dim)
        self.dyaw_sign_embedding = nn.Embedding(3, sign_dim)
        self.basin_bin_embedding = nn.Embedding(4, basin_bin_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim + delta_dim + sign_dim * 3 + basin_bin_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_groups),
        )

    def forward(
        self,
        *,
        handoff_latent: torch.Tensor,
        proxy_current_delta_basin_target: torch.Tensor,
        current_dx_sign: torch.Tensor,
        current_dy_sign: torch.Tensor,
        current_dyaw_sign: torch.Tensor,
        basin_distance_bin: torch.Tensor,
    ) -> torch.Tensor:
        h_delta = self.delta_encoder(proxy_current_delta_basin_target.float())
        h_dx = self.dx_sign_embedding(torch.clamp(current_dx_sign.long() + 1, min=0, max=2))
        h_dy = self.dy_sign_embedding(torch.clamp(current_dy_sign.long() + 1, min=0, max=2))
        h_dyaw = self.dyaw_sign_embedding(torch.clamp(current_dyaw_sign.long() + 1, min=0, max=2))
        h_bin = self.basin_bin_embedding(torch.clamp(basin_distance_bin.long(), min=0, max=3))
        return self.head(torch.cat([handoff_latent, h_delta, h_dx, h_dy, h_dyaw, h_bin], dim=-1))
