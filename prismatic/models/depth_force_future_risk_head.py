"""
depth_force_future_risk_head.py

Small candidate-conditioned future-risk head that reuses the frozen
DepthForceLocalContactPolicy encoders. The head stays runtime-safe: it only
consumes RGB-D, force history, proprio, planner action, candidate action and an
optional geometry-score feature produced by the frozen geometry scorer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prismatic.models.depth_force_contact_policy import DepthForceLocalContactPolicy


class DepthForceCandidateFutureRiskHead(nn.Module):
    def __init__(
        self,
        backbone: DepthForceLocalContactPolicy | None = None,
        hidden_dim: int = 256,
        geometry_score_dim: int = 16,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone or DepthForceLocalContactPolicy()
        state_dim = int(getattr(self.backbone.state_trunk[-2], "out_features", 384))
        candidate_dim = int(getattr(self.backbone.candidate_encoder[-2], "out_features", 96))
        self.geometry_score_proj = nn.Sequential(
            nn.Linear(1, geometry_score_dim),
            nn.ReLU(inplace=True),
            nn.Linear(geometry_score_dim, geometry_score_dim),
            nn.ReLU(inplace=True),
        )
        self.pair_trunk = nn.Sequential(
            nn.Linear(state_dim + candidate_dim + geometry_score_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.future_total_risk_head = nn.Linear(hidden_dim, 1)
        self.future_contact_risk_head = nn.Linear(hidden_dim, 1)
        self.future_force_spike_risk_head = nn.Linear(hidden_dim, 1)
        self.future_jam_risk_head = nn.Linear(hidden_dim, 1)
        self.future_motion_stall_risk_head = nn.Linear(hidden_dim, 1)
        self.future_kinematic_invalid_risk_head = nn.Linear(hidden_dim, 1)
        self.future_action_invalid_risk_head = nn.Linear(hidden_dim, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def encode_pair(
        self,
        front_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_base_action_local: torch.Tensor,
        candidate_actions_local: torch.Tensor,
        geometry_scores: torch.Tensor | None = None,
        stage_token: torch.Tensor | None = None,
        contact_phase: torch.Tensor | None = None,
        depth_proximity: torch.Tensor | None = None,
        gripper_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.backbone.encode_state(
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
        cand = self.backbone.candidate_encoder(candidate_actions_local)
        if geometry_scores is None:
            geom_feat = torch.zeros(
                (*candidate_actions_local.shape[:2], self.geometry_score_proj[0].out_features),
                device=candidate_actions_local.device,
                dtype=candidate_actions_local.dtype,
            )
        else:
            geom_feat = self.geometry_score_proj(geometry_scores.unsqueeze(-1))
        state_exp = state[:, None, :].expand(-1, cand.shape[1], -1)
        return torch.cat([state_exp, cand, geom_feat], dim=-1)

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
        geometry_scores: torch.Tensor | None = None,
        stage_token: torch.Tensor | None = None,
        contact_phase: torch.Tensor | None = None,
        depth_proximity: torch.Tensor | None = None,
        gripper_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pair = self.encode_pair(
            front_rgb=front_rgb,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            force_history=force_history,
            proprio=proprio,
            planner_base_action_local=planner_base_action_local,
            candidate_actions_local=candidate_actions_local,
            geometry_scores=geometry_scores,
            stage_token=stage_token,
            contact_phase=contact_phase,
            depth_proximity=depth_proximity,
            gripper_state=gripper_state,
        )
        hidden = self.pair_trunk(pair)
        future_total_risk = self.future_total_risk_head(hidden).squeeze(-1)
        future_contact_risk = self.future_contact_risk_head(hidden).squeeze(-1)
        future_force_spike_risk = self.future_force_spike_risk_head(hidden).squeeze(-1)
        future_jam_risk = self.future_jam_risk_head(hidden).squeeze(-1)
        future_motion_stall_risk = self.future_motion_stall_risk_head(hidden).squeeze(-1)
        future_kinematic_invalid_risk = self.future_kinematic_invalid_risk_head(hidden).squeeze(-1)
        future_action_invalid_risk = self.future_action_invalid_risk_head(hidden).squeeze(-1)
        if candidate_mask is not None:
            mask = candidate_mask <= 0.5
            future_total_risk = future_total_risk.masked_fill(mask, 1e9)
            future_contact_risk = future_contact_risk.masked_fill(mask, 1e9)
            future_force_spike_risk = future_force_spike_risk.masked_fill(mask, 1e9)
            future_jam_risk = future_jam_risk.masked_fill(mask, 1e9)
            future_motion_stall_risk = future_motion_stall_risk.masked_fill(mask, 1e9)
            future_kinematic_invalid_risk = future_kinematic_invalid_risk.masked_fill(mask, 1e9)
            future_action_invalid_risk = future_action_invalid_risk.masked_fill(mask, 1e9)
        return {
            "future_total_risk": future_total_risk,
            "future_contact_risk": future_contact_risk,
            "future_force_spike_risk": future_force_spike_risk,
            "future_jam_risk": future_jam_risk,
            "future_motion_stall_risk": future_motion_stall_risk,
            "future_kinematic_invalid_risk": future_kinematic_invalid_risk,
            "future_action_invalid_risk": future_action_invalid_risk,
        }
