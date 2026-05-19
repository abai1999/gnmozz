"""Direct local alignment controller v4.

V4 keeps the near/micro runtime contract explicit, but moves beyond fixed-grid
proposal ranking. It predicts a continuous bounded residual together with
short-horizon diagnostics, risk, confidence, and an interpretable policy mode.
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


class AlignmentV4DirectLocalController(nn.Module):
    """Target-conditioned near-contact direct local controller."""

    def __init__(
        self,
        proprio_input_dim: int = 15,
        force_input_dim: int = 6,
        state_dim: int = 256,
        hidden_dim: int = 192,
        depth_dim: int = 64,
        rgb_dim: int = 48,
        force_dim: int = 48,
        proprio_dim: int = 48,
        planner_dim: int = 32,
        target_dim: int = 64,
        max_pos: float = 0.015,
        max_yaw: float = 0.006,
        use_front_rgb: bool = False,
        use_wrist_depth: bool = True,
        use_force: bool = True,
        use_planner_action: bool = True,
    ) -> None:
        super().__init__()
        self._controller_type = "alignment_v4_direct_local_controller"
        self.use_front_rgb = bool(use_front_rgb)
        self.use_wrist_depth = bool(use_wrist_depth)
        self.use_force = bool(use_force)
        self.use_planner_action = bool(use_planner_action)
        self.max_pos = float(max_pos)
        self.max_yaw = float(max_yaw)

        self.front_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.depth_encoder = DepthEncoderTiny(out_dim=depth_dim)
        self.force_encoder = ForceEncoderTiny(force_dim=force_input_dim, out_dim=force_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim=proprio_input_dim, out_dim=proprio_dim)
        self.planner_encoder = BaseActionEncoder(action_dim=6, out_dim=planner_dim)
        self.target_delta_encoder = nn.Sequential(
            nn.Linear(6, target_dim),
            nn.ReLU(inplace=True),
            nn.Linear(target_dim, target_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = target_dim + proprio_dim + planner_dim + depth_dim + force_dim + (rgb_dim if self.use_front_rgb else 0)
        self.state_trunk = nn.Sequential(
            nn.Linear(fused_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(inplace=True),
        )

        self.residual_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 4))
        self.post_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 3))
        self.reduction_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 3))
        self.risk_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 1))
        self.confidence_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.policy_mode_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 3))

        for module in (
            self.residual_head[-1],
            self.post_head[-1],
            self.reduction_head[-1],
            self.risk_head[-1],
            self.confidence_head[-1],
            self.policy_mode_head[-1],
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def _encode_optional_rgb(self, front_rgb: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if front_rgb.ndim == 3:
            front_rgb = front_rgb.unsqueeze(0)
        if front_rgb.shape[-2:] != (96, 96):
            front_rgb = F.interpolate(front_rgb, size=(96, 96), mode="bilinear", align_corners=False)
        return self.front_rgb_encoder(front_rgb.to(dtype=dtype))

    def encode_state(
        self,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_action_local: torch.Tensor,
        current_to_target_delta_local: torch.Tensor,
        front_rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = current_to_target_delta_local.shape[0]
        dtype = current_to_target_delta_local.dtype
        device = current_to_target_delta_local.device

        parts = []
        if self.use_front_rgb and front_rgb is not None:
            parts.append(self._encode_optional_rgb(front_rgb, dtype=dtype))
        elif self.use_front_rgb:
            parts.append(torch.zeros((bsz, 48), device=device, dtype=dtype))

        if self.use_wrist_depth:
            if wrist_depth.ndim == 3:
                wrist_depth = wrist_depth.unsqueeze(1)
            if wrist_depth.shape[-2:] != (96, 96):
                wrist_depth = F.interpolate(wrist_depth, size=(96, 96), mode="bilinear", align_corners=False)
            parts.append(self.depth_encoder(wrist_depth.to(dtype=dtype)))
        else:
            parts.append(torch.zeros((bsz, self.depth_encoder.fc.out_features), device=device, dtype=dtype))

        if self.use_force:
            parts.append(self.force_encoder(force_history.to(dtype=dtype)))
        else:
            parts.append(torch.zeros((bsz, self.force_encoder.fc.out_features), device=device, dtype=dtype))

        parts.append(self.proprio_encoder(proprio.to(dtype=dtype)))
        if self.use_planner_action:
            parts.append(self.planner_encoder(planner_action_local.to(dtype=dtype)))
        else:
            parts.append(torch.zeros((bsz, self.planner_encoder.mlp[-1].out_features), device=device, dtype=dtype))
        parts.append(self.target_delta_encoder(current_to_target_delta_local.to(dtype=dtype)))
        return self.state_trunk(torch.cat(parts, dim=-1))

    def _bound_residual(self, residual_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_4d = residual_4d.to(dtype=torch.float32)
        pos = residual_4d[:, :3]
        yaw = residual_4d[:, 3:4]

        pos_norm = torch.linalg.norm(pos, dim=-1, keepdim=True).clamp_min(1e-8)
        pos_scale = torch.clamp(torch.full_like(pos_norm, self.max_pos) / pos_norm, max=1.0)
        pos_bounded = pos * pos_scale

        yaw_abs = yaw.abs().clamp_min(1e-8)
        yaw_scale = torch.clamp(torch.full_like(yaw_abs, self.max_yaw) / yaw_abs, max=1.0)
        yaw_bounded = yaw * yaw_scale

        residual_6d = torch.zeros((residual_4d.shape[0], 6), device=residual_4d.device, dtype=residual_4d.dtype)
        residual_6d[:, :3] = pos_bounded
        residual_6d[:, 5] = yaw_bounded.squeeze(-1)
        return torch.cat([pos_bounded, yaw_bounded], dim=-1), residual_6d

    def forward(
        self,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_action_local: torch.Tensor,
        current_to_target_delta_local: torch.Tensor,
        front_rgb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_state(
            wrist_depth=wrist_depth,
            force_history=force_history,
            proprio=proprio,
            planner_action_local=planner_action_local,
            current_to_target_delta_local=current_to_target_delta_local,
            front_rgb=front_rgb,
        )
        raw_residual_4d = self.residual_head(state)
        bounded_residual_4d, bounded_residual_6d = self._bound_residual(raw_residual_4d)
        post_error_xyz_yaw = self.post_head(state)
        delta_reduction = self.reduction_head(state)
        risk_logit = self.risk_head(state).squeeze(-1)
        confidence_logit = self.confidence_head(state).squeeze(-1)
        policy_mode_logits = self.policy_mode_head(state)
        return {
            "state": state,
            "raw_residual_4d": raw_residual_4d,
            "direct_residual_4d": bounded_residual_4d,
            "direct_residual_6d": bounded_residual_6d,
            "shadow_post_xyz_yaw": post_error_xyz_yaw,
            "shadow_delta_reduction": delta_reduction,
            "risk_logit": risk_logit,
            "confidence_logit": confidence_logit,
            "policy_mode_logits": policy_mode_logits,
        }
