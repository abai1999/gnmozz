"""Non-privileged short-horizon alignment diffusion refiner.

This module intentionally does not consume target pose, oracle delta, CAD pose,
or privileged geometry.  It proposes bounded end-effector local residual
trajectories from runtime-observable context, and exposes progress/risk heads
for a runtime safety shield to select or reject candidates.
"""

from __future__ import annotations

import math

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


class AlignmentDiffusionRefiner(nn.Module):
    """Tiny conditional trajectory generator for near-contact residual control."""

    def __init__(
        self,
        proprio_input_dim: int = 15,
        force_input_dim: int = 6,
        horizon: int = 8,
        action_dim: int = 4,
        state_dim: int = 256,
        hidden_dim: int = 192,
        depth_dim: int = 64,
        rgb_dim: int = 48,
        force_dim: int = 48,
        proprio_dim: int = 48,
        planner_dim: int = 48,
        gripper_dim: int = 16,
        max_pos_step: float = 0.0015,
        max_yaw_step: float = 0.0060,
        use_front_rgb: bool = False,
        use_wrist_rgb: bool = True,
        use_wrist_depth: bool = True,
        use_force: bool = True,
        use_planner_action: bool = True,
    ) -> None:
        super().__init__()
        self._controller_type = "alignment_diffusion_refiner"
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.max_pos_step = float(max_pos_step)
        self.max_yaw_step = float(max_yaw_step)
        self.use_front_rgb = bool(use_front_rgb)
        self.use_wrist_rgb = bool(use_wrist_rgb)
        self.use_wrist_depth = bool(use_wrist_depth)
        self.use_force = bool(use_force)
        self.use_planner_action = bool(use_planner_action)

        self.front_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.wrist_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.depth_encoder = DepthEncoderTiny(out_dim=depth_dim)
        self.force_encoder = ForceEncoderTiny(force_dim=force_input_dim, out_dim=force_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim=proprio_input_dim, out_dim=proprio_dim)
        self.planner_encoder = BaseActionEncoder(action_dim=6, out_dim=planner_dim)
        self.gripper_encoder = nn.Sequential(
            nn.Linear(4, gripper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gripper_dim, gripper_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = (
            proprio_dim
            + gripper_dim
            + (planner_dim if self.use_planner_action else 0)
            + (depth_dim if self.use_wrist_depth else 0)
            + (force_dim if self.use_force else 0)
            + (rgb_dim if self.use_front_rgb else 0)
            + (rgb_dim if self.use_wrist_rgb else 0)
        )
        self.state_trunk = nn.Sequential(
            nn.Linear(fused_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(inplace=True),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.horizon * self.action_dim),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, self.horizon * self.action_dim),
        )
        self.progress_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        for module in (self.traj_head[-1], self.progress_head[-1], self.risk_head[-1], self.stop_head[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.constant_(self.scale_head[-1].bias, -3.0)
        nn.init.zeros_(self.scale_head[-1].weight)

    def _rgb(self, x: torch.Tensor | None, bsz: int, device: torch.device, dtype: torch.dtype, encoder: nn.Module) -> torch.Tensor:
        if x is None:
            return torch.zeros((bsz, encoder.fc.out_features), device=device, dtype=dtype)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[-2:] != (96, 96):
            x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        return encoder(x.to(device=device, dtype=dtype))

    def _depth(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[-2:] != (96, 96):
            x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        return self.depth_encoder(x.to(dtype=dtype))

    def encode_state(
        self,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_action_local: torch.Tensor,
        gripper_context: torch.Tensor | None = None,
        front_rgb: torch.Tensor | None = None,
        wrist_rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = proprio.shape[0]
        dtype = proprio.dtype
        device = proprio.device
        parts = []
        if self.use_front_rgb:
            parts.append(self._rgb(front_rgb, bsz, device, dtype, self.front_rgb_encoder))
        if self.use_wrist_rgb:
            parts.append(self._rgb(wrist_rgb, bsz, device, dtype, self.wrist_rgb_encoder))
        if self.use_wrist_depth:
            parts.append(self._depth(wrist_depth.to(device=device, dtype=dtype), dtype=dtype))
        if self.use_force:
            parts.append(self.force_encoder(force_history.to(device=device, dtype=dtype)))
        parts.append(self.proprio_encoder(proprio.to(device=device, dtype=dtype)))
        if self.use_planner_action:
            parts.append(self.planner_encoder(planner_action_local.to(device=device, dtype=dtype)))
        if gripper_context is None:
            gripper_context = torch.zeros((bsz, 4), device=device, dtype=dtype)
        if gripper_context.ndim == 1:
            gripper_context = gripper_context.unsqueeze(0)
        if gripper_context.shape[-1] < 4:
            pad = torch.zeros((bsz, 4 - gripper_context.shape[-1]), device=device, dtype=dtype)
            gripper_context = torch.cat([gripper_context.to(device=device, dtype=dtype), pad], dim=-1)
        parts.append(self.gripper_encoder(gripper_context[:, :4].to(device=device, dtype=dtype)))
        return self.state_trunk(torch.cat(parts, dim=-1))

    def _bound_trajectory(self, traj_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = traj_4d[..., :3]
        yaw = traj_4d[..., 3:4]
        pos_norm = torch.linalg.norm(pos, dim=-1, keepdim=True).clamp_min(1e-8)
        pos_scale = torch.clamp(torch.full_like(pos_norm, self.max_pos_step) / pos_norm, max=1.0)
        yaw_abs = yaw.abs().clamp_min(1e-8)
        yaw_scale = torch.clamp(torch.full_like(yaw_abs, self.max_yaw_step) / yaw_abs, max=1.0)
        bounded_4d = torch.cat([pos * pos_scale, yaw * yaw_scale], dim=-1)
        bounded_6d = torch.zeros(
            (*bounded_4d.shape[:-1], 6),
            device=bounded_4d.device,
            dtype=bounded_4d.dtype,
        )
        bounded_6d[..., :3] = bounded_4d[..., :3]
        bounded_6d[..., 5] = bounded_4d[..., 3]
        return bounded_4d, bounded_6d

    def forward(
        self,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_action_local: torch.Tensor,
        gripper_context: torch.Tensor | None = None,
        front_rgb: torch.Tensor | None = None,
        wrist_rgb: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        noise_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_state(
            wrist_depth=wrist_depth,
            force_history=force_history,
            proprio=proprio,
            planner_action_local=planner_action_local,
            gripper_context=gripper_context,
            front_rgb=front_rgb,
            wrist_rgb=wrist_rgb,
        )
        mean = self.traj_head(state).view(-1, self.horizon, self.action_dim)
        log_scale = self.scale_head(state).view(-1, self.horizon, self.action_dim).clamp(-6.0, 1.0)
        if noise is None:
            raw = mean
        else:
            raw = mean + noise.to(device=mean.device, dtype=mean.dtype) * log_scale.exp() * float(noise_scale)
        traj_4d, traj_6d = self._bound_trajectory(raw)
        return {
            "state": state,
            "trajectory_mean_4d": mean,
            "trajectory_scale_4d": log_scale.exp(),
            "trajectory_4d": traj_4d,
            "trajectory_6d": traj_6d,
            "first_residual_4d": traj_4d[:, 0],
            "first_residual_6d": traj_6d[:, 0],
            "progress_logits": self.progress_head(state),
            "risk_logit": self.risk_head(state).squeeze(-1),
            "stop_logit": self.stop_head(state).squeeze(-1),
        }

    @torch.no_grad()
    def sample_candidates(
        self,
        num_samples: int,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Sample K candidate trajectories plus a deterministic mean candidate."""
        num_samples = max(int(num_samples), 1)
        proprio = kwargs["proprio"]
        bsz = proprio.shape[0]
        state = self.encode_state(**{k: v for k, v in kwargs.items() if k != "num_samples"})
        mean = self.traj_head(state).view(bsz, self.horizon, self.action_dim)
        scale = self.scale_head(state).view(bsz, self.horizon, self.action_dim).clamp(-6.0, 1.0).exp()
        if num_samples == 1:
            raw = mean[:, None]
        else:
            noise = torch.randn(
                (bsz, num_samples - 1, self.horizon, self.action_dim),
                device=mean.device,
                dtype=mean.dtype,
            )
            raw = torch.cat([mean[:, None], mean[:, None] + noise * scale[:, None]], dim=1)
        flat_4d, flat_6d = self._bound_trajectory(raw.reshape(-1, self.horizon, self.action_dim))
        traj_4d = flat_4d.view(bsz, num_samples, self.horizon, self.action_dim)
        traj_6d = flat_6d.view(bsz, num_samples, self.horizon, 6)
        progress = self.progress_head(state)
        risk = self.risk_head(state).squeeze(-1)
        stop = self.stop_head(state).squeeze(-1)
        base_score = progress.sum(dim=-1) - risk - torch.sigmoid(stop)
        smooth = torch.linalg.norm(traj_4d[..., 1:, :] - traj_4d[..., :-1, :], dim=-1).mean(dim=-1)
        action_norm = torch.linalg.norm(traj_4d[..., 0, :3], dim=-1) + traj_4d[..., 0, 3].abs()
        sample_scores = base_score[:, None] - 0.1 * smooth - 0.05 * action_norm
        best_index = torch.argmax(sample_scores, dim=-1)
        gather_idx = best_index.view(bsz, 1, 1, 1).expand(-1, 1, self.horizon, self.action_dim)
        best_4d = torch.gather(traj_4d, 1, gather_idx).squeeze(1)
        gather_idx6 = best_index.view(bsz, 1, 1, 1).expand(-1, 1, self.horizon, 6)
        best_6d = torch.gather(traj_6d, 1, gather_idx6).squeeze(1)
        diversity = torch.zeros((bsz,), device=mean.device, dtype=mean.dtype)
        if num_samples > 1:
            centered = traj_4d[..., 0, :] - traj_4d[..., 0, :].mean(dim=1, keepdim=True)
            diversity = torch.sqrt(torch.mean(centered.square(), dim=(1, 2)).clamp_min(0.0))
        return {
            "candidate_trajectory_4d": traj_4d,
            "candidate_trajectory_6d": traj_6d,
            "candidate_scores": sample_scores,
            "best_index": best_index,
            "best_trajectory_4d": best_4d,
            "best_trajectory_6d": best_6d,
            "first_residual_4d": best_4d[:, 0],
            "first_residual_6d": best_6d[:, 0],
            "progress_logits": progress,
            "risk_logit": risk,
            "stop_logit": stop,
            "candidate_diversity": diversity,
            "num_samples": torch.full((bsz,), num_samples, device=mean.device, dtype=torch.long),
        }
