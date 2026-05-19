"""Target-conditioned diffusion alignment refiner.

Training may use privileged target/contact labels, but runtime inputs remain
non-privileged.  The model first estimates a target/contact representation from
observable state and then conditions a bounded short-horizon residual generator
on that representation.
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


class AlignmentTargetEstimator(nn.Module):
    """Estimate target-relative geometry from runtime-observable context."""

    def __init__(
        self,
        proprio_input_dim: int = 15,
        force_input_dim: int = 6,
        state_dim: int = 256,
        depth_dim: int = 64,
        rgb_dim: int = 48,
        force_dim: int = 48,
        proprio_dim: int = 48,
        planner_dim: int = 48,
        gripper_dim: int = 16,
        target_repr_dim: int = 64,
        heatmap_size: int = 16,
        use_front_rgb: bool = False,
        use_wrist_rgb: bool = True,
        use_wrist_depth: bool = True,
        use_force: bool = True,
        use_planner_action: bool = True,
    ) -> None:
        super().__init__()
        self.heatmap_size = int(heatmap_size)
        self.target_repr_dim = int(target_repr_dim)
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
        self.delta_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 6))
        self.confidence_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.heatmap_head = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.heatmap_size * self.heatmap_size),
        )
        self.target_repr_head = nn.Sequential(
            nn.Linear(state_dim + 6 + 1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.target_repr_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

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

    def encode_observation(
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

    def forward(
        self,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_action_local: torch.Tensor,
        gripper_context: torch.Tensor | None = None,
        front_rgb: torch.Tensor | None = None,
        wrist_rgb: torch.Tensor | None = None,
        teacher_target_delta_local: torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
        phase_id: torch.Tensor | None = None,
        stage_bucket_id: torch.Tensor | None = None,
        **unused_kwargs,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_observation(
            wrist_depth=wrist_depth,
            force_history=force_history,
            proprio=proprio,
            planner_action_local=planner_action_local,
            gripper_context=gripper_context,
            front_rgb=front_rgb,
            wrist_rgb=wrist_rgb,
        )
        pred_delta = self.delta_head(state)
        confidence_logit = self.confidence_head(state).squeeze(-1)
        confidence = torch.sigmoid(confidence_logit)
        heatmap_logits = self.heatmap_head(state).view(-1, 1, self.heatmap_size, self.heatmap_size)

        target_delta_for_repr = pred_delta
        if teacher_target_delta_local is not None and float(teacher_force_prob) > 0.0:
            teacher = teacher_target_delta_local.to(device=pred_delta.device, dtype=pred_delta.dtype)
            if float(teacher_force_prob) >= 1.0:
                target_delta_for_repr = teacher
            else:
                target_delta_for_repr = (1.0 - float(teacher_force_prob)) * pred_delta + float(teacher_force_prob) * teacher
        repr_input = torch.cat([state, target_delta_for_repr, confidence.unsqueeze(-1)], dim=-1)
        target_repr = self.target_repr_head(repr_input)
        return {
            "obs_state": state,
            "pred_target_delta_local_6d": pred_delta,
            "pred_contact_heatmap_logits": heatmap_logits,
            "target_confidence_logit": confidence_logit,
            "target_confidence": confidence,
            "target_repr": target_repr,
        }


class TargetConditionedAlignmentDiffusionRefiner(nn.Module):
    """Bounded trajectory generator conditioned on estimated target/contact repr."""

    def __init__(
        self,
        horizon: int = 8,
        action_dim: int = 4,
        state_dim: int = 256,
        hidden_dim: int = 192,
        target_repr_dim: int = 64,
        max_pos_step: float = 0.0015,
        max_yaw_step: float = 0.0060,
        **estimator_kwargs,
    ) -> None:
        super().__init__()
        self._controller_type = "alignment_tc_diffusion_refiner"
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.max_pos_step = float(max_pos_step)
        self.max_yaw_step = float(max_yaw_step)
        self.target_estimator = AlignmentTargetEstimator(
            state_dim=state_dim,
            target_repr_dim=target_repr_dim,
            **estimator_kwargs,
        )
        fused_dim = state_dim + target_repr_dim
        self.policy_trunk = nn.Sequential(
            nn.Linear(fused_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(inplace=True),
        )
        self.traj_head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, self.horizon * self.action_dim))
        self.scale_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, self.horizon * self.action_dim))
        self.progress_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 3))
        self.risk_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 1))
        self.stop_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 1))
        for module in (self.traj_head[-1], self.progress_head[-1], self.risk_head[-1], self.stop_head[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.constant_(self.scale_head[-1].bias, -3.0)

    def _bound_trajectory(self, traj_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = traj_4d[..., :3]
        yaw = traj_4d[..., 3:4]
        pos_norm = torch.linalg.norm(pos, dim=-1, keepdim=True).clamp_min(1e-8)
        pos_scale = torch.clamp(torch.full_like(pos_norm, self.max_pos_step) / pos_norm, max=1.0)
        yaw_abs = yaw.abs().clamp_min(1e-8)
        yaw_scale = torch.clamp(torch.full_like(yaw_abs, self.max_yaw_step) / yaw_abs, max=1.0)
        bounded_4d = torch.cat([pos * pos_scale, yaw * yaw_scale], dim=-1)
        bounded_6d = torch.zeros((*bounded_4d.shape[:-1], 6), device=bounded_4d.device, dtype=bounded_4d.dtype)
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
        teacher_target_delta_local: torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
        noise: torch.Tensor | None = None,
        noise_scale: float = 1.0,
        phase_id: torch.Tensor | None = None,
        stage_bucket_id: torch.Tensor | None = None,
        **unused_kwargs,
    ) -> dict[str, torch.Tensor]:
        target = self.target_estimator(
            wrist_depth=wrist_depth,
            force_history=force_history,
            proprio=proprio,
            planner_action_local=planner_action_local,
            gripper_context=gripper_context,
            front_rgb=front_rgb,
            wrist_rgb=wrist_rgb,
            teacher_target_delta_local=teacher_target_delta_local,
            teacher_force_prob=teacher_force_prob,
        )
        state = self.policy_trunk(torch.cat([target["obs_state"], target["target_repr"]], dim=-1))
        mean = self.traj_head(state).view(-1, self.horizon, self.action_dim)
        scale = self.scale_head(state).view(-1, self.horizon, self.action_dim).clamp(-6.0, 1.0).exp()
        raw = mean if noise is None else mean + noise.to(device=mean.device, dtype=mean.dtype) * scale * float(noise_scale)
        traj_4d, traj_6d = self._bound_trajectory(raw)
        out = {
            **target,
            "state": state,
            "trajectory_mean_4d": mean,
            "trajectory_scale_4d": scale,
            "trajectory_4d": traj_4d,
            "trajectory_6d": traj_6d,
            "first_residual_4d": traj_4d[:, 0],
            "first_residual_6d": traj_6d[:, 0],
            "progress_logits": self.progress_head(state),
            "risk_logit": self.risk_head(state).squeeze(-1),
            "stop_logit": self.stop_head(state).squeeze(-1),
        }
        return out

    @torch.no_grad()
    def sample_candidates(self, num_samples: int, top_k: int | None = None, **kwargs) -> dict[str, torch.Tensor]:
        num_samples = max(int(num_samples), 1)
        target = self.target_estimator(**kwargs)
        bsz = target["obs_state"].shape[0]
        state = self.policy_trunk(torch.cat([target["obs_state"], target["target_repr"]], dim=-1))
        mean = self.traj_head(state).view(bsz, self.horizon, self.action_dim)
        scale = self.scale_head(state).view(bsz, self.horizon, self.action_dim).clamp(-6.0, 1.0).exp()
        if num_samples == 1:
            raw = mean[:, None]
        else:
            noise = torch.randn((bsz, num_samples - 1, self.horizon, self.action_dim), device=mean.device, dtype=mean.dtype)
            raw = torch.cat([mean[:, None], mean[:, None] + noise * scale[:, None]], dim=1)
        flat4, flat6 = self._bound_trajectory(raw.reshape(-1, self.horizon, self.action_dim))
        traj4 = flat4.view(bsz, num_samples, self.horizon, self.action_dim)
        traj6 = flat6.view(bsz, num_samples, self.horizon, 6)
        progress = self.progress_head(state)
        risk = self.risk_head(state).squeeze(-1)
        stop = self.stop_head(state).squeeze(-1)
        base_score = progress.sum(dim=-1) - risk - torch.sigmoid(stop)
        effort = torch.linalg.norm(traj4[..., 0, :3], dim=-1) + 0.25 * traj4[..., 0, 3].abs()
        candidate_scores = base_score[:, None] - 0.05 * effort
        best_index = torch.argmax(candidate_scores, dim=1)
        gather = best_index.view(bsz, 1, 1, 1).expand(-1, 1, self.horizon, self.action_dim)
        best4 = torch.gather(traj4, 1, gather).squeeze(1)
        gather6 = best_index.view(bsz, 1, 1, 1).expand(-1, 1, self.horizon, 6)
        best6 = torch.gather(traj6, 1, gather6).squeeze(1)
        diversity = traj4[..., 0, :].std(dim=1).mean(dim=-1) if num_samples > 1 else torch.zeros((bsz,), device=mean.device)
        return {
            **target,
            "trajectory_4d": best4,
            "trajectory_6d": best6,
            "first_residual_4d": best4[:, 0],
            "first_residual_6d": best6[:, 0],
            "candidate_trajectory_4d": traj4,
            "candidate_trajectory_6d": traj6,
            "candidate_scores": candidate_scores,
            "best_index": best_index,
            "num_samples": torch.full((bsz,), int(num_samples), device=mean.device, dtype=torch.long),
            "top_k": torch.full((bsz,), int(top_k or min(3, num_samples)), device=mean.device, dtype=torch.long),
            "candidate_diversity": diversity,
            "progress_logits": progress,
            "risk_logit": risk,
            "stop_logit": stop,
        }
