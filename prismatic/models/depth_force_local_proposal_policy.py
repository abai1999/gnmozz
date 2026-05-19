"""
Depth-force local proposal policy.

This module is intentionally small and self-contained. It consumes only
runtime-safe inputs and emits a small set of local action proposals plus
proposal scores / mode logits. Geometry and future-risk remain separate
evaluators.
"""

from __future__ import annotations

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
class LocalProposalActionScale:
    xyz: tuple[float, float, float] = (0.008, 0.008, 0.006)
    rot: tuple[float, float, float] = (0.06, 0.06, 0.12)

    def as_tensor(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        vals = torch.tensor([*self.xyz, *self.rot], device=device, dtype=dtype or torch.float32)
        return vals


class DepthForceLocalProposalPolicy(nn.Module):
    """State-conditioned local proposal generator for contact alignment."""

    MULTI_HEAD_NAMES = ("best_safe", "pareto", "yaw_match", "risk_safe", "geometry_gain")

    def __init__(
        self,
        proprio_input_dim: int = 15,
        force_input_dim: int = 6,
        stage_vocab_size: int = 8,
        contact_phase_vocab_size: int = 4,
        state_dim: int = 384,
        hidden_dim: int = 256,
        proposal_count: int = 8,
        score_dim: int = 128,
        mode_count: int = 5,
        proposal_action_scale: LocalProposalActionScale | None = None,
        use_front_rgb: bool = False,
        use_wrist_rgb: bool = False,
        use_wrist_depth: bool = True,
        use_force: bool = True,
        use_proprio: bool = True,
        use_stage: bool = False,
        use_contact: bool = False,
        use_scalar: bool = True,
        use_candidate_depth_context: bool = True,
        use_candidate_force_context: bool = True,
    ) -> None:
        super().__init__()
        self.proposal_count = int(proposal_count)
        self.proposal_action_scale = proposal_action_scale or LocalProposalActionScale()
        self.action_scale = nn.Parameter(self.proposal_action_scale.as_tensor(), requires_grad=False)
        self.use_front_rgb = bool(use_front_rgb)
        self.use_wrist_rgb = bool(use_wrist_rgb)
        self.use_wrist_depth = bool(use_wrist_depth)
        self.use_force = bool(use_force)
        self.use_proprio = bool(use_proprio)
        self.use_stage = bool(use_stage)
        self.use_contact = bool(use_contact)
        self.use_scalar = bool(use_scalar)
        self.use_candidate_depth_context = bool(use_candidate_depth_context)
        self.use_candidate_force_context = bool(use_candidate_force_context)

        self.front_encoder = RGBEncoderTiny(out_dim=96)
        self.wrist_encoder = RGBEncoderTiny(out_dim=96)
        self.depth_encoder = DepthEncoderTiny(out_dim=96)
        self.force_encoder = ForceEncoderTiny(force_dim=force_input_dim, out_dim=64)
        self.proprio_encoder = ProprioEncoder(proprio_dim=proprio_input_dim, out_dim=64)
        self.base_action_encoder = BaseActionEncoder(action_dim=6, out_dim=48)
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
        self.front_dim = 96
        self.wrist_dim = 96
        self.depth_dim = 96
        self.force_dim = 64
        self.proprio_dim = 64
        self.stage_dim = 16
        self.contact_dim = 8
        self.scalar_dim = 16
        self.proposal_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.proposal_count * 6),
        )
        self.proposal_candidate_encoder = nn.Sequential(
            nn.Linear(6, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
        )
        self.proposal_context_encoder = nn.Sequential(
            nn.Linear(state_dim + 96 + 64 + 6 + 6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
        )
        self.proposal_score_head = nn.Sequential(
            nn.Linear(state_dim + 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Candidate-local evaluator used after the K=8 proposal generator is frozen.
        # The inputs are deliberately per-candidate: proposal action, candidate
        # delta, local depth samples shifted by the candidate xy, and force-action
        # interaction terms. This keeps depth/force from collapsing into a row-level
        # bias shared by all candidates.
        self.candidate_depth_stats_dim = 10
        self.force_action_interaction_dim = 8
        self.multi_head_context_encoder = nn.Sequential(
            nn.Linear(state_dim + 6 + 6 + self.candidate_depth_stats_dim + self.force_action_interaction_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.multi_head_score_head = nn.Linear(128, len(self.MULTI_HEAD_NAMES))
        self.mode_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mode_count),
        )
        self.aux_state_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

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

    def _candidate_depth_stats(self, wrist_depth: torch.Tensor, proposal_actions: torch.Tensor) -> torch.Tensor:
        """Return simple candidate-shifted local depth statistics.

        This is intentionally geometric and lightweight: each candidate xy offset
        shifts a small sampling grid around the image center. The stats give the
        score head candidate-specific depth cues without requiring a heavy visual
        backbone or camera calibration.
        """
        bsz, count, _ = proposal_actions.shape
        depth = self._resize_depth(wrist_depth.to(device=proposal_actions.device, dtype=proposal_actions.dtype))
        center_grid_vals = torch.linspace(-0.08, 0.08, 5, device=proposal_actions.device, dtype=proposal_actions.dtype)
        gy, gx = torch.meshgrid(center_grid_vals, center_grid_vals, indexing="ij")
        base_grid = torch.stack([gx, gy], dim=-1).view(1, 1, 5, 5, 2)

        xy_scale = torch.clamp(self.action_scale[:2].to(device=proposal_actions.device, dtype=proposal_actions.dtype), min=1e-6)
        # Convert local action xy to a conservative normalized image offset.
        offsets = torch.clamp(proposal_actions[:, :, :2] / xy_scale.view(1, 1, 2), -1.5, 1.5) * 0.18
        grid = base_grid + offsets.view(bsz, count, 1, 1, 2)
        grid = torch.clamp(grid, -1.0, 1.0).view(bsz * count, 5, 5, 2)
        depth_rep = depth[:, None].expand(-1, count, -1, -1, -1).reshape(bsz * count, 1, depth.shape[-2], depth.shape[-1])
        patch = F.grid_sample(depth_rep, grid, mode="bilinear", padding_mode="border", align_corners=False).view(bsz, count, 5, 5)

        mean = patch.mean(dim=(-1, -2))
        std = patch.std(dim=(-1, -2), unbiased=False)
        mn = patch.amin(dim=(-1, -2))
        mx = patch.amax(dim=(-1, -2))
        center = patch[:, :, 2, 2]
        left = patch[:, :, :, :2].mean(dim=(-1, -2))
        right = patch[:, :, :, 3:].mean(dim=(-1, -2))
        top = patch[:, :, :2, :].mean(dim=(-1, -2))
        bottom = patch[:, :, 3:, :].mean(dim=(-1, -2))
        grad_x = right - left
        grad_y = bottom - top
        global_mean = depth.mean(dim=(-1, -2, -3))[:, None]
        global_std = depth.std(dim=(-1, -2, -3), unbiased=False)[:, None]
        return torch.stack(
            [
                mean,
                std,
                mn,
                mx,
                center,
                center - global_mean,
                std - global_std,
                grad_x,
                grad_y,
                torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8),
            ],
            dim=-1,
        )

    def _force_action_interactions(self, force_history: torch.Tensor, proposal_actions: torch.Tensor) -> torch.Tensor:
        bsz, count, _ = proposal_actions.shape
        force = force_history.to(device=proposal_actions.device, dtype=proposal_actions.dtype)
        if force.ndim != 3:
            force = torch.zeros((bsz, 1, 6), device=proposal_actions.device, dtype=proposal_actions.dtype)
        last = force[:, -1, :6]
        prev = force[:, -2, :6] if force.shape[1] > 1 else torch.zeros_like(last)
        delta = last - prev
        f_xyz = last[:, :3]
        torque = last[:, 3:6]
        f_norm = torch.linalg.norm(f_xyz, dim=-1, keepdim=True)
        torque_norm = torch.linalg.norm(torque, dim=-1, keepdim=True)
        spike = torch.linalg.norm(delta[:, :3], dim=-1, keepdim=True)

        cand_xyz = proposal_actions[:, :, :3]
        cand_rot = proposal_actions[:, :, 3:6]
        action_norm = torch.linalg.norm(proposal_actions, dim=-1)
        force_xy = f_xyz[:, None, :2]
        torque_xy = torque[:, None, :2]
        return torch.stack(
            [
                f_norm[:, None, 0] * cand_xyz[:, :, 2],
                torch.sum(force_xy * cand_xyz[:, :, :2], dim=-1),
                torque[:, None, 2] * cand_rot[:, :, 2],
                spike[:, None, 0] * action_norm,
                torch.sum(torque_xy * cand_rot[:, :, :2], dim=-1),
                f_xyz[:, None, 2] * cand_xyz[:, :, 2],
                torque_norm[:, None, 0] * torch.abs(cand_rot[:, :, 2]),
                f_norm[:, None, 0] * torch.linalg.norm(cand_xyz[:, :, :2], dim=-1),
            ],
            dim=-1,
        )

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
        parts: list[torch.Tensor] = []
        if self.use_front_rgb:
            parts.append(self.front_encoder(self._resize_rgb(front_rgb)))
        else:
            parts.append(torch.zeros((bsz, self.front_dim), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        if self.use_wrist_rgb:
            parts.append(self.wrist_encoder(self._resize_rgb(wrist_rgb)))
        else:
            parts.append(torch.zeros((bsz, self.wrist_dim), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        if self.use_wrist_depth:
            parts.append(self.depth_encoder(self._resize_depth(wrist_depth)))
        else:
            parts.append(torch.zeros((bsz, self.depth_dim), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        if self.use_force:
            parts.append(self.force_encoder(force_history))
        else:
            parts.append(torch.zeros((bsz, self.force_dim), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        if self.use_proprio:
            parts.append(self.proprio_encoder(proprio))
        else:
            parts.append(torch.zeros((bsz, self.proprio_dim), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        parts.append(self.base_action_encoder(planner_base_action_local))
        if self.use_stage:
            parts.append(torch.clamp(stage_token.long(), min=0, max=self.stage_embedding.num_embeddings - 1).to(planner_base_action_local.device))
            parts[-1] = self.stage_embedding(parts[-1])
        else:
            parts.append(torch.zeros((bsz, self.stage_dim), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        if self.use_contact:
            parts.append(torch.clamp(contact_phase.long(), min=0, max=self.contact_embedding.num_embeddings - 1).to(planner_base_action_local.device))
            parts[-1] = self.contact_embedding(parts[-1])
        else:
            parts.append(torch.zeros((bsz, self.contact_dim), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        if self.use_scalar:
            parts.append(self.scalar_encoder(scalar))
        else:
            parts.append(torch.zeros((bsz, self.scalar_dim), device=planner_base_action_local.device, dtype=planner_base_action_local.dtype))
        fused = torch.cat(parts, dim=-1)
        return self.state_trunk(fused)

    def forward(
        self,
        front_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_base_action_local: torch.Tensor,
        proposal_actions_local: torch.Tensor | None = None,
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
        if proposal_actions_local is None:
            proposal_raw = self.proposal_head(state).view(-1, self.proposal_count, 6)
            proposal_actions = torch.tanh(proposal_raw) * self.action_scale.view(1, 1, 6)
        else:
            proposal_actions = proposal_actions_local.to(device=state.device, dtype=state.dtype)
            if proposal_actions.ndim != 3 or proposal_actions.shape[-1] != 6:
                raise ValueError(f"expected proposal_actions_local with shape (B, K, 6), got {tuple(proposal_actions.shape)}")
            if proposal_actions.shape[1] != self.proposal_count:
                raise ValueError(
                    f"proposal_actions_local proposal_count mismatch: got {proposal_actions.shape[1]}, expected {self.proposal_count}"
                )
        depth_feat = (
            self.depth_encoder(self._resize_depth(wrist_depth))
            if self.use_wrist_depth
            else torch.zeros((state.shape[0], self.depth_dim), device=state.device, dtype=state.dtype)
        )
        force_feat = (
            self.force_encoder(force_history)
            if self.use_force
            else torch.zeros((state.shape[0], self.force_dim), device=state.device, dtype=state.dtype)
        )
        cand = self.proposal_candidate_encoder(proposal_actions.reshape(-1, 6)).view(-1, self.proposal_count, 64)
        cand_delta = proposal_actions - planner_base_action_local[:, None, :].to(device=state.device, dtype=state.dtype)
        context_in = torch.cat(
            [
                state[:, None, :].expand(-1, self.proposal_count, -1),
                depth_feat[:, None, :].expand(-1, self.proposal_count, -1),
                force_feat[:, None, :].expand(-1, self.proposal_count, -1),
                proposal_actions,
                cand_delta,
            ],
            dim=-1,
        )
        context = self.proposal_context_encoder(context_in.reshape(-1, context_in.shape[-1])).view(-1, self.proposal_count, 64)
        cand = cand + context
        state_exp = state[:, None, :].expand(-1, self.proposal_count, -1)
        score = self.proposal_score_head(torch.cat([state_exp, cand], dim=-1)).squeeze(-1)
        if self.use_candidate_depth_context:
            depth_stats = self._candidate_depth_stats(wrist_depth, proposal_actions)
        else:
            depth_stats = torch.zeros(
                (state.shape[0], self.proposal_count, self.candidate_depth_stats_dim),
                device=state.device,
                dtype=state.dtype,
            )
        if self.use_candidate_force_context:
            force_interactions = self._force_action_interactions(force_history, proposal_actions)
        else:
            force_interactions = torch.zeros(
                (state.shape[0], self.proposal_count, self.force_action_interaction_dim),
                device=state.device,
                dtype=state.dtype,
            )
        multi_in = torch.cat([state_exp, proposal_actions, cand_delta, depth_stats, force_interactions], dim=-1)
        multi_latent = self.multi_head_context_encoder(multi_in.reshape(-1, multi_in.shape[-1])).view(-1, self.proposal_count, 128)
        multi_scores = self.multi_head_score_head(multi_latent)
        multi = {name: multi_scores[:, :, i] for i, name in enumerate(self.MULTI_HEAD_NAMES)}
        mode_logits = self.mode_head(state)
        aux_state = self.aux_state_head(state).squeeze(-1)
        return {
            "proposal_actions_local": proposal_actions,
            "proposal_scores": score,
            "multi_head_scores": multi_scores,
            "multi_head_score_dict": multi,
            "candidate_depth_stats": depth_stats,
            "force_action_interactions": force_interactions,
            "mode_logits": mode_logits,
            "state_value": aux_state,
            "state_latent": state,
        }
