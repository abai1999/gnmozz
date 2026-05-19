"""Target-Conditioned Alignment Policy v2.

Near/micro local alignment evaluator.  Consumes target-relative error and
per-candidate post-action delta.  Outputs per-candidate scores and a
stage-aware selection.

Deliberately separate from DepthForceLocalProposalPolicy — this module
target-conditions every candidate, while the old module does not.
"""
from __future__ import annotations

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


class TargetConditionedAlignmentPolicy(nn.Module):
    """Target-conditioned candidate evaluator for near/micro alignment."""

    def __init__(
        self,
        proprio_input_dim: int = 15,
        force_input_dim: int = 6,
        state_dim: int = 256,
        hidden_dim: int = 192,
        proposal_count: int = 8,
        target_delta_dim: int = 6,
        candidate_feat_dim: int = 32,
        use_wrist_depth: bool = True,
        use_force: bool = True,
        use_front_rgb: bool = False,
        use_wrist_rgb: bool = False,
    ) -> None:
        super().__init__()
        self.proposal_count = int(proposal_count)
        self.use_wrist_depth = bool(use_wrist_depth)
        self.use_force = bool(use_force)
        self.use_front_rgb = bool(use_front_rgb)
        self.use_wrist_rgb = bool(use_wrist_rgb)

        # --- State encoders ---
        self.depth_encoder = DepthEncoderTiny(out_dim=64)
        self.force_encoder = ForceEncoderTiny(force_dim=force_input_dim, out_dim=48)
        self.proprio_encoder = ProprioEncoder(proprio_dim=proprio_input_dim, out_dim=48)
        self.base_action_encoder = BaseActionEncoder(action_dim=6, out_dim=32)
        self.target_delta_encoder = nn.Sequential(
            nn.Linear(target_delta_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 64),
        )

        state_fused = 64 + 48 + 48 + 32 + 64  # depth + force + proprio + planner_action + target_delta
        self.state_trunk = nn.Sequential(
            nn.Linear(state_fused, state_dim), nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim), nn.ReLU(inplace=True),
        )

        # --- Candidate encoder ---
        # Input: proposal[6] + post_delta[6] + improvement[4] (xy,z,yaw,geom)
        cand_input_dim = 6 + 6 + 4
        self.candidate_encoder = nn.Sequential(
            nn.Linear(cand_input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, candidate_feat_dim), nn.ReLU(inplace=True),
        )

        # --- Score head: state+candidate → score ---
        self.score_head = nn.Sequential(
            nn.Linear(state_dim + candidate_feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # --- Aux heads ---
        self.overshoot_head = nn.Sequential(
            nn.Linear(state_dim + candidate_feat_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_state(
        self,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_base_action_local: torch.Tensor,
        current_to_target_delta_local: torch.Tensor,
    ) -> torch.Tensor:
        bsz = planner_base_action_local.shape[0]
        parts = []
        if self.use_wrist_depth:
            if wrist_depth.ndim == 3:
                wrist_depth = wrist_depth.unsqueeze(1)
            if wrist_depth.shape[-2:] != (96, 96):
                wrist_depth = F.interpolate(wrist_depth, size=(96, 96), mode="bilinear", align_corners=False)
            parts.append(self.depth_encoder(wrist_depth))
        else:
            parts.append(torch.zeros((bsz, 64), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        if self.use_force:
            parts.append(self.force_encoder(force_history.to(dtype=planner_base_action_local.dtype)))
        else:
            parts.append(torch.zeros((bsz, 48), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        parts.append(self.proprio_encoder(proprio.to(dtype=planner_base_action_local.dtype)))
        parts.append(self.base_action_encoder(planner_base_action_local.to(dtype=planner_base_action_local.dtype)))
        parts.append(self.target_delta_encoder(current_to_target_delta_local.to(dtype=planner_base_action_local.dtype)))
        return self.state_trunk(torch.cat(parts, dim=-1))

    def score_candidates(
        self,
        state: torch.Tensor,
        proposal_actions: torch.Tensor,
        post_candidate_delta: torch.Tensor,
        xy_improvement: torch.Tensor | None = None,
        z_improvement: torch.Tensor | None = None,
        yaw_improvement: torch.Tensor | None = None,
        geometry_improvement: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score K candidates per row.

        Returns (scores, overshoot_logits), each (B, K).
        """
        bsz, K, _ = proposal_actions.shape
        device = state.device
        dtype = state.dtype

        if xy_improvement is None:
            xy_improvement = torch.zeros(bsz, K, device=device, dtype=dtype)
        if z_improvement is None:
            z_improvement = torch.zeros(bsz, K, device=device, dtype=dtype)
        if yaw_improvement is None:
            yaw_improvement = torch.zeros(bsz, K, device=device, dtype=dtype)
        if geometry_improvement is None:
            geometry_improvement = torch.zeros(bsz, K, device=device, dtype=dtype)

        cand_in = torch.cat(
            [
                proposal_actions,
                post_candidate_delta,
                xy_improvement.unsqueeze(-1),
                z_improvement.unsqueeze(-1),
                yaw_improvement.unsqueeze(-1),
                geometry_improvement.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, K, 6+6+4)
        cand_feat = self.candidate_encoder(cand_in.reshape(bsz * K, -1)).view(bsz, K, -1)
        state_exp = state.unsqueeze(1).expand(-1, K, -1)
        pair = torch.cat([state_exp, cand_feat], dim=-1).reshape(bsz * K, -1)
        scores = self.score_head(pair).view(bsz, K)
        overshoot = self.overshoot_head(pair).view(bsz, K)
        return scores, overshoot

    def forward(
        self,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_base_action_local: torch.Tensor,
        current_to_target_delta_local: torch.Tensor,
        proposal_actions: torch.Tensor,
        post_candidate_delta: torch.Tensor,
        xy_improvement: torch.Tensor | None = None,
        z_improvement: torch.Tensor | None = None,
        yaw_improvement: torch.Tensor | None = None,
        geometry_improvement: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_state(
            wrist_depth=wrist_depth,
            force_history=force_history,
            proprio=proprio,
            planner_base_action_local=planner_base_action_local,
            current_to_target_delta_local=current_to_target_delta_local,
        )
        scores, overshoot_logits = self.score_candidates(
            state=state,
            proposal_actions=proposal_actions,
            post_candidate_delta=post_candidate_delta,
            xy_improvement=xy_improvement,
            z_improvement=z_improvement,
            yaw_improvement=yaw_improvement,
            geometry_improvement=geometry_improvement,
        )
        return {
            "candidate_scores": scores,
            "overshoot_logits": overshoot_logits,
            "state": state,
        }
