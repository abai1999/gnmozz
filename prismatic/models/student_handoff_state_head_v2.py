"""
student_handoff_state_head_v2.py

Observation-conditioned student handoff-state head for phase-1 alignment.
This model is explicitly runtime-safe: it only consumes non-privileged inputs
and predicts the student's own near-ready geometry / readiness state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.residual_encoders import DepthEncoderTiny, ProprioEncoder, RGBEncoderTiny


class StudentHandoffStateHeadV2(nn.Module):
    def __init__(
        self,
        rgb_dim: int = 96,
        depth_dim: int = 96,
        proprio_dim: int = 64,
        context_dim: int = 64,
        hidden_dim: int = 192,
        latent_dim: int = 128,
        delta_dim: int = 32,
        sign_dim: int = 4,
        basin_bin_dim: int = 4,
        substage_vocab: int = 8,
        contact_vocab: int = 8,
        target_mode_vocab: int = 8,
        num_bands: int = 3,
        temporal_summary_dim: int = 32,
        residual_xyz_bound: float = 0.006,
        residual_yaw_bound: float = 0.03,
    ):
        super().__init__()
        self.num_bands = int(num_bands)
        self.temporal_summary_dim = int(temporal_summary_dim)
        self.residual_xyz_bound = float(residual_xyz_bound)
        self.residual_yaw_bound = float(residual_yaw_bound)
        self.front_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.wrist_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.depth_encoder = DepthEncoderTiny(out_dim=depth_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim=15, out_dim=proprio_dim)
        self.gripper_context_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(6, delta_dim),
            nn.ReLU(inplace=True),
            nn.Linear(delta_dim, delta_dim),
        )
        self.dx_sign_embedding = nn.Embedding(3, sign_dim)
        self.dy_sign_embedding = nn.Embedding(3, sign_dim)
        self.dyaw_sign_embedding = nn.Embedding(3, sign_dim)
        self.basin_bin_embedding = nn.Embedding(4, basin_bin_dim)
        self.substage_emb = nn.Embedding(substage_vocab, 16)
        self.contact_emb = nn.Embedding(contact_vocab, 8)
        self.target_mode_emb = nn.Embedding(target_mode_vocab, 8)
        self.context_mlp = nn.Sequential(
            nn.Linear(32 + delta_dim + sign_dim * 3 + basin_bin_dim + 16 + 8 + 8, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(inplace=True),
        )
        fused_dim = rgb_dim * 2 + depth_dim + proprio_dim + context_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.metric_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.band_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, self.num_bands),
        )
        self.ready_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.progress_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.axis_block_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.closeness_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.closeability_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.residual_delta_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 4),
        )
        self.residual_confidence_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.corrective_dx_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.corrective_dy_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.corrective_dz_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.corrective_dyaw_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.corrective_dyaw_coarse_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 5),
        )
        self.corrective_dyaw_residual_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.temporal_action_encoder = nn.Sequential(
            nn.Linear(self.temporal_summary_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim // 2),
            nn.ReLU(inplace=True),
        )
        temporal_head_dim = latent_dim + latent_dim // 2
        self.temporal_progress_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.temporal_axis_block_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.temporal_closeness_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.temporal_closeability_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.temporal_residual_delta_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 4),
        )
        self.temporal_residual_confidence_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )
        self.temporal_corrective_dx_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.temporal_corrective_dy_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.temporal_corrective_dz_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.temporal_corrective_dyaw_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 3),
        )
        self.temporal_corrective_dyaw_coarse_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 5),
        )
        self.temporal_corrective_dyaw_residual_head = nn.Sequential(
            nn.Linear(temporal_head_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(
        self,
        *,
        front_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        wrist_depth: torch.Tensor,
        proprio: torch.Tensor,
        gripper_context: torch.Tensor,
        proxy_current_delta_basin_target: torch.Tensor,
        current_dx_sign: torch.Tensor,
        current_dy_sign: torch.Tensor,
        current_dyaw_sign: torch.Tensor,
        basin_distance_bin: torch.Tensor,
        substage_id: torch.Tensor,
        contact_state: torch.Tensor,
        stage_target_mode: torch.Tensor,
        temporal_action_summary: torch.Tensor | None = None,
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
        h_grip = self.gripper_context_encoder(gripper_context)
        h_delta = self.delta_encoder(proxy_current_delta_basin_target.float())
        h_dx = self.dx_sign_embedding(torch.clamp(current_dx_sign.long() + 1, min=0, max=2))
        h_dy = self.dy_sign_embedding(torch.clamp(current_dy_sign.long() + 1, min=0, max=2))
        h_dyaw = self.dyaw_sign_embedding(torch.clamp(current_dyaw_sign.long() + 1, min=0, max=2))
        h_bin = self.basin_bin_embedding(torch.clamp(basin_distance_bin.long(), min=0, max=3))
        h_ctx = self.context_mlp(
            torch.cat(
                [
                    h_grip,
                    h_delta,
                    h_dx,
                    h_dy,
                    h_dyaw,
                    h_bin,
                    self.substage_emb(substage_id.long()),
                    self.contact_emb(contact_state.long()),
                    self.target_mode_emb(stage_target_mode.long()),
                ],
                dim=-1,
            )
        )
        latent = self.trunk(torch.cat([h_front, h_wrist, h_depth, h_prop, h_ctx], dim=-1))
        pred_metrics = F.softplus(self.metric_head(latent))
        band_logits = self.band_head(latent)
        ready_logit = self.ready_head(latent).squeeze(-1)
        # Softplus keeps uncertainty non-negative and easy to threshold.
        uncertainty = F.softplus(self.uncertainty_head(latent)).squeeze(-1)
        progress_logit = self.progress_head(latent).squeeze(-1)
        axis_block_logits = self.axis_block_head(latent)
        closeness_score = self.closeness_head(latent).squeeze(-1)
        closeability_logit = self.closeability_head(latent).squeeze(-1)
        residual_delta_raw = self.residual_delta_head(latent)
        residual_delta_local = torch.cat(
            [
                torch.tanh(residual_delta_raw[:, :3]) * self.residual_xyz_bound,
                torch.tanh(residual_delta_raw[:, 3:4]) * self.residual_yaw_bound,
            ],
            dim=-1,
        )
        residual_confidence_logit = self.residual_confidence_head(latent).squeeze(-1)
        corrective_dx_logits = self.corrective_dx_head(latent)
        corrective_dy_logits = self.corrective_dy_head(latent)
        corrective_dz_logits = self.corrective_dz_head(latent)
        corrective_dyaw_legacy_logits = self.corrective_dyaw_head(latent)
        corrective_dyaw_logits = self.corrective_dyaw_coarse_head(latent)
        corrective_dyaw_residual = self.corrective_dyaw_residual_head(latent).squeeze(-1)
        if temporal_action_summary is not None:
            h_temporal = self.temporal_action_encoder(temporal_action_summary.float())
            h_progress = torch.cat([latent, h_temporal], dim=-1)
            progress_logit = self.temporal_progress_head(h_progress).squeeze(-1)
            axis_block_logits = self.temporal_axis_block_head(h_progress)
            closeness_score = self.temporal_closeness_head(h_progress).squeeze(-1)
            closeability_logit = self.temporal_closeability_head(h_progress).squeeze(-1)
            residual_delta_raw = self.temporal_residual_delta_head(h_progress)
            residual_delta_local = torch.cat(
                [
                    torch.tanh(residual_delta_raw[:, :3]) * self.residual_xyz_bound,
                    torch.tanh(residual_delta_raw[:, 3:4]) * self.residual_yaw_bound,
                ],
                dim=-1,
            )
            residual_confidence_logit = self.temporal_residual_confidence_head(h_progress).squeeze(-1)
            corrective_dx_logits = self.temporal_corrective_dx_head(h_progress)
            corrective_dy_logits = self.temporal_corrective_dy_head(h_progress)
            corrective_dz_logits = self.temporal_corrective_dz_head(h_progress)
            corrective_dyaw_legacy_logits = self.temporal_corrective_dyaw_head(h_progress)
            corrective_dyaw_logits = self.temporal_corrective_dyaw_coarse_head(h_progress)
            corrective_dyaw_residual = self.temporal_corrective_dyaw_residual_head(h_progress).squeeze(-1)
        return {
            "latent": latent,
            "pred_metrics_norm": pred_metrics,
            "band_logits": band_logits,
            "ready_logit": ready_logit,
            "uncertainty": uncertainty,
            "progress_logit": progress_logit,
            "axis_block_logits": axis_block_logits,
            "closeness_score": closeness_score,
            "closeability_logit": closeability_logit,
            "residual_delta_local": residual_delta_local,
            "residual_confidence_logit": residual_confidence_logit,
            "corrective_dx_logits": corrective_dx_logits,
            "corrective_dy_logits": corrective_dy_logits,
            "corrective_dz_logits": corrective_dz_logits,
            "corrective_dyaw_logits": corrective_dyaw_logits,
            "corrective_dyaw_legacy_logits": corrective_dyaw_legacy_logits,
            "corrective_dyaw_residual": corrective_dyaw_residual,
        }
