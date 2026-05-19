"""
near_ready_xyyaw_predictor.py

Lightweight observation-conditioned head for phase-1 near-ready estimation.
This head is intentionally separate from motion alignment so it cannot perturb
the current alignment baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.residual_encoders import DepthEncoderTiny, ProprioEncoder, RGBEncoderTiny


class NearReadyXYYawPredictor(nn.Module):
    def __init__(
        self,
        rgb_dim: int = 96,
        depth_dim: int = 96,
        proprio_dim: int = 64,
        context_dim: int = 48,
        hidden_dim: int = 192,
        substage_vocab: int = 8,
        contact_vocab: int = 8,
        target_mode_vocab: int = 8,
    ):
        super().__init__()
        self.front_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.wrist_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.depth_encoder = DepthEncoderTiny(out_dim=depth_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim=15, out_dim=proprio_dim)
        self.gripper_context = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )
        self.substage_emb = nn.Embedding(substage_vocab, 16)
        self.contact_emb = nn.Embedding(contact_vocab, 8)
        self.target_mode_emb = nn.Embedding(target_mode_vocab, 8)
        self.context_mlp = nn.Sequential(
            nn.Linear(32 + 16 + 8 + 8 + 2, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(inplace=True),
        )
        fused_dim = rgb_dim * 2 + depth_dim + proprio_dim + context_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.metric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.ready_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        *,
        front_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        wrist_depth: torch.Tensor,
        proprio: torch.Tensor,
        gripper_context: torch.Tensor,
        runtime_xyyaw_norm: torch.Tensor,
        substage_id: torch.Tensor,
        contact_state: torch.Tensor,
        stage_target_mode: torch.Tensor,
    ):
        if front_rgb.shape[-2:] != (96, 96):
            front_rgb = F.interpolate(front_rgb, size=(96, 96), mode="bilinear", align_corners=False)
        if wrist_rgb.shape[-2:] != (96, 96):
            wrist_rgb = F.interpolate(wrist_rgb, size=(96, 96), mode="bilinear", align_corners=False)
        if wrist_depth.shape[-2:] != (96, 96):
            wrist_depth = F.interpolate(wrist_depth, size=(96, 96), mode="bilinear", align_corners=False)

        h_front = self.front_rgb_encoder(front_rgb)
        h_wrist = self.wrist_rgb_encoder(wrist_rgb)
        h_depth = self.depth_encoder(wrist_depth)
        h_prop = self.proprio_encoder(proprio)
        h_grip = self.gripper_context(gripper_context)
        h_ctx = self.context_mlp(
            torch.cat(
                [
                    h_grip,
                    self.substage_emb(substage_id.long()),
                    self.contact_emb(contact_state.long()),
                    self.target_mode_emb(stage_target_mode.long()),
                    runtime_xyyaw_norm.float(),
                ],
                dim=-1,
            )
        )
        hidden = self.trunk(torch.cat([h_front, h_wrist, h_depth, h_prop, h_ctx], dim=-1))
        metrics = F.softplus(self.metric_head(hidden))
        ready_logit = self.ready_head(hidden).squeeze(-1)
        return {
            "xyyaw_norm": metrics,
            "ready_logit": ready_logit,
        }
