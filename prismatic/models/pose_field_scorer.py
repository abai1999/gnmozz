"""
pose_field_scorer.py

State-conditioned local candidate action scorer for near-field alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.residual_encoders import (
    BaseActionEncoder,
    DepthEncoderTiny,
    ProprioEncoder,
    RGBEncoderTiny,
    StepEmbedding,
)


class PoseFieldScorer(nn.Module):
    """Scores a fixed set of local candidate actions for one state."""

    def __init__(
        self,
        depth_dim: int = 128,
        proprio_dim: int = 64,
        action_dim: int = 32,
        gripper_dim: int = 16,
        step_dim: int = 16,
        phase_dim: int = 8,
        context_dim: int = 16,
        delta_dim: int = 32,
        sign_dim: int = 4,
        basin_bin_dim: int = 4,
        hidden_dim: int = 128,
        num_chunk_steps: int = 8,
        phase_vocab_size: int = 4,
        num_candidate_groups: int = 11,
        fire_only_head: bool = True,
        use_depth: bool = True,
        use_base_action: bool = True,
        use_proprio: bool = True,
        use_target_context: bool = True,
        use_front_rgb: bool = False,
        use_wrist_rgb: bool = False,
        rgb_dim: int = 96,
    ):
        super().__init__()
        self.fire_only_head = bool(fire_only_head)
        self.use_depth = bool(use_depth)
        self.use_base_action = bool(use_base_action)
        self.use_proprio = bool(use_proprio)
        self.use_target_context = bool(use_target_context)
        self.use_front_rgb = bool(use_front_rgb)
        self.use_wrist_rgb = bool(use_wrist_rgb)
        self.depth_dim = int(depth_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.rgb_dim = int(rgb_dim)
        self.depth_encoder = DepthEncoderTiny(out_dim=depth_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim=15, out_dim=proprio_dim)
        self.base_action_encoder = BaseActionEncoder(action_dim=6, out_dim=action_dim)
        self.candidate_encoder = BaseActionEncoder(action_dim=6, out_dim=action_dim)
        self.front_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.wrist_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.gripper_encoder = nn.Sequential(
            nn.Linear(3, gripper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gripper_dim, gripper_dim),
        )
        self.step_embedding = StepEmbedding(num_steps=num_chunk_steps, out_dim=step_dim)
        self.phase_embedding = nn.Embedding(phase_vocab_size, phase_dim)
        self.num_candidate_groups = int(num_candidate_groups)
        self.delta_encoder = nn.Sequential(
            nn.Linear(6, delta_dim),
            nn.ReLU(inplace=True),
            nn.Linear(delta_dim, delta_dim),
        )
        self.dx_sign_embedding = nn.Embedding(3, sign_dim)
        self.dy_sign_embedding = nn.Embedding(3, sign_dim)
        self.dyaw_sign_embedding = nn.Embedding(3, sign_dim)
        self.basin_bin_embedding = nn.Embedding(4, basin_bin_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
        )

        target_context_dim = (delta_dim + sign_dim * 3 + basin_bin_dim) if self.use_target_context else 0
        rgb_in = (rgb_dim if self.use_front_rgb else 0) + (rgb_dim if self.use_wrist_rgb else 0)
        state_in = (
            depth_dim + proprio_dim + action_dim + gripper_dim + step_dim + phase_dim + context_dim
            + rgb_in
            + target_context_dim
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.group_head = nn.Linear(hidden_dim, self.num_candidate_groups)
        self.ready_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.ready_head.weight)
        nn.init.constant_(self.ready_head.bias, -2.0)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.step_scale_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self._readiness_heads_loaded = True

    def forward(
        self,
        front_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        wrist_depth: torch.Tensor,
        proprio: torch.Tensor,
        base_action: torch.Tensor,
        gripper_context: torch.Tensor,
        step_idx: torch.Tensor,
        candidate_actions: torch.Tensor,
        phase_id: torch.Tensor = None,
        phase_age: torch.Tensor = None,
        steps_since_last_replan: torch.Tensor = None,
        current_delta_basin_target: torch.Tensor = None,
        current_dx_sign: torch.Tensor = None,
        current_dy_sign: torch.Tensor = None,
        current_dyaw_sign: torch.Tensor = None,
        basin_distance_bin: torch.Tensor = None,
        candidate_mask: torch.Tensor = None,
        return_aux: bool = False,
    ):
        """
        Args:
            front_rgb: (B,3,H,W)
            wrist_rgb: (B,3,H,W)
            wrist_depth: (B,1,96,96)
            proprio: (B,15)
            base_action: (B,6)
            gripper_context: (B,3)
            step_idx: (B,)
            candidate_actions: (B,N,6)
        Returns:
            scores: (B,N)
        """
        B, N, _ = candidate_actions.shape
        if phase_id is None:
            phase_id = torch.zeros_like(step_idx)
        if phase_age is None:
            phase_age = torch.zeros_like(step_idx, dtype=base_action.dtype)
        if steps_since_last_replan is None:
            steps_since_last_replan = torch.zeros_like(step_idx, dtype=base_action.dtype)
        if current_delta_basin_target is None:
            current_delta_basin_target = torch.zeros(B, 6, device=base_action.device, dtype=base_action.dtype)
        if current_dx_sign is None:
            current_dx_sign = torch.zeros_like(step_idx)
        if current_dy_sign is None:
            current_dy_sign = torch.zeros_like(step_idx)
        if current_dyaw_sign is None:
            current_dyaw_sign = torch.zeros_like(step_idx)
        if basin_distance_bin is None:
            basin_distance_bin = torch.zeros_like(step_idx)

        if front_rgb.shape[-1] != 96 or front_rgb.shape[-2] != 96:
            front_rgb = F.interpolate(front_rgb, size=(96, 96), mode="bilinear", align_corners=False)
        if wrist_rgb.shape[-1] != 96 or wrist_rgb.shape[-2] != 96:
            wrist_rgb = F.interpolate(wrist_rgb, size=(96, 96), mode="bilinear", align_corners=False)
        if self.use_depth:
            h_d = self.depth_encoder(wrist_depth)
        else:
            h_d = torch.zeros(B, self.depth_dim, device=proprio.device, dtype=proprio.dtype)
        if self.use_front_rgb:
            h_front = self.front_rgb_encoder(front_rgb.to(base_action.dtype))
        else:
            h_front = torch.zeros(B, self.rgb_dim, device=proprio.device, dtype=proprio.dtype)
        if self.use_wrist_rgb:
            h_wrist = self.wrist_rgb_encoder(wrist_rgb.to(base_action.dtype))
        else:
            h_wrist = torch.zeros(B, self.rgb_dim, device=proprio.device, dtype=proprio.dtype)
        if self.use_proprio:
            h_p = self.proprio_encoder(proprio)
        else:
            h_p = torch.zeros(B, self.proprio_dim, device=proprio.device, dtype=proprio.dtype)
        if self.use_base_action:
            h_a = self.base_action_encoder(base_action)
        else:
            h_a = torch.zeros(B, self.action_dim, device=proprio.device, dtype=proprio.dtype)
        h_g = self.gripper_encoder(gripper_context.to(base_action.dtype))
        h_s = self.step_embedding(step_idx)
        h_phase = self.phase_embedding(torch.clamp(phase_id.long(), min=0, max=self.phase_embedding.num_embeddings - 1))
        context = torch.stack(
            [
                torch.clamp(phase_age.to(base_action.dtype), min=0.0, max=32.0) / 32.0,
                torch.clamp(steps_since_last_replan.to(base_action.dtype), min=0.0, max=32.0) / 32.0,
            ],
            dim=-1,
        )
        h_ctx = self.context_encoder(context)
        state_features = [h_d, h_p, h_a, h_g, h_s, h_phase, h_ctx]
        if self.use_front_rgb:
            state_features.insert(0, h_front)
        if self.use_wrist_rgb:
            insert_at = 1 if self.use_front_rgb else 0
            state_features.insert(insert_at, h_wrist)
        if self.use_target_context:
            h_delta = self.delta_encoder(current_delta_basin_target.to(base_action.dtype))
            h_dx = self.dx_sign_embedding(torch.clamp((current_dx_sign.long() + 1), min=0, max=2))
            h_dy = self.dy_sign_embedding(torch.clamp((current_dy_sign.long() + 1), min=0, max=2))
            h_dyaw = self.dyaw_sign_embedding(torch.clamp((current_dyaw_sign.long() + 1), min=0, max=2))
            h_basin_bin = self.basin_bin_embedding(
                torch.clamp(basin_distance_bin.long(), min=0, max=self.basin_bin_embedding.num_embeddings - 1)
            )
            state_features.extend([h_delta, h_dx, h_dy, h_dyaw, h_basin_bin])
        state_hidden = self.state_mlp(torch.cat(state_features, dim=-1))
        group_logits = self.group_head(state_hidden)
        ready_logits = self.ready_head(state_hidden).squeeze(-1)
        ready_to_close = torch.sigmoid(ready_logits)

        cand_flat = candidate_actions.reshape(B * N, 6)
        h_c = self.candidate_encoder(cand_flat).reshape(B, N, -1)
        state_expand = state_hidden.unsqueeze(1).expand(-1, N, -1)
        state_candidate = torch.cat([state_expand, h_c], dim=-1)
        scores = self.score_head(state_candidate).squeeze(-1)
        step_scale = 2.0 * torch.sigmoid(self.step_scale_head(state_candidate).squeeze(-1))
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask.to(dtype=torch.bool) <= 0, -1e9)
            step_scale = step_scale.masked_fill(candidate_mask.to(dtype=torch.bool) <= 0, 0.0)
        if return_aux:
            return {
                "candidate_scores": scores,
                "candidate_step_scale": step_scale,
                "group_logits": group_logits,
                "ready_to_close_logits": ready_logits,
                "ready_to_close": ready_to_close,
            }
        return scores
