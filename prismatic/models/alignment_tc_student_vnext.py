"""Phase-aware target-conditioned alignment student vNext.

This keeps the same runtime spirit as the existing target-conditioned diffusion
refiner, but adds explicit target/contact estimation heads, phase/stage
conditioning, and a cleaner training contract for verified-only phase1/phase2
teacher data.
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


def _maybe_interp_rgb(x: torch.Tensor | None, encoder: nn.Module, bsz: int, device, dtype) -> torch.Tensor:
    if x is None:
        return torch.zeros((bsz, encoder.fc.out_features), device=device, dtype=dtype)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[-2:] != (96, 96):
        x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
    return encoder(x.to(device=device, dtype=dtype))


def _maybe_interp_depth(x: torch.Tensor | None, encoder: nn.Module, bsz: int, device, dtype) -> torch.Tensor:
    if x is None:
        return torch.zeros((bsz, encoder.fc.out_features), device=device, dtype=dtype)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.shape[-2:] != (96, 96):
        x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
    return encoder(x.to(device=device, dtype=dtype))


class AlignmentTCStudentVNext(nn.Module):
    """Verified-only phase1/phase2 target-conditioned residual controller."""

    def __init__(
        self,
        horizon: int = 8,
        action_dim: int = 4,
        proprio_input_dim: int = 15,
        force_input_dim: int = 6,
        state_dim: int = 256,
        hidden_dim: int = 192,
        depth_dim: int = 64,
        rgb_dim: int = 48,
        force_dim: int = 48,
        proprio_dim: int = 48,
        planner_dim: int = 48,
        gripper_dim: int = 16,
        phase_emb_dim: int = 8,
        stage_emb_dim: int = 8,
        phase_vocab_size: int = 4,
        stage_vocab_size: int = 10,
        target_repr_dim: int = 96,
        contact_repr_dim: int = 8,
        max_pos_step: float = 0.0015,
        max_yaw_step: float = 0.0060,
        use_front_rgb: bool = False,
        use_wrist_rgb: bool = True,
        use_wrist_depth: bool = True,
        use_force: bool = True,
        use_planner_action: bool = True,
        y_bridge_max_step: float | None = None,
    ) -> None:
        super().__init__()
        self._controller_type = "alignment_tc_student_vnext"
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.max_pos_step = float(max_pos_step)
        self.max_yaw_step = float(max_yaw_step)
        self.contact_repr_dim = int(contact_repr_dim)
        self.target_repr_dim = int(target_repr_dim)
        self.use_front_rgb = bool(use_front_rgb)
        self.use_wrist_rgb = bool(use_wrist_rgb)
        self.use_wrist_depth = bool(use_wrist_depth)
        self.use_force = bool(use_force)
        self.use_planner_action = bool(use_planner_action)
        self.y_bridge_max_step = float(y_bridge_max_step) if y_bridge_max_step is not None else float(max_pos_step) * 0.85

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
        self.phase_embedding = nn.Embedding(phase_vocab_size, phase_emb_dim)
        self.stage_embedding = nn.Embedding(stage_vocab_size, stage_emb_dim)

        fused_dim = (
            proprio_dim
            + gripper_dim
            + phase_emb_dim
            + stage_emb_dim
            + (planner_dim if self.use_planner_action else 0)
            + (depth_dim if self.use_wrist_depth else 0)
            + (force_dim if self.use_force else 0)
            + (rgb_dim if self.use_front_rgb else 0)
            + (rgb_dim if self.use_wrist_rgb else 0)
        )
        self.obs_trunk = nn.Sequential(
            nn.Linear(fused_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(inplace=True),
        )
        self.target_delta_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 6))
        self.contact_repr_head = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.contact_repr_dim),
        )
        self.confidence_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.close_ready_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.handoff_ready_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.progress_prior_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))
        self.yaw_direction_head = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))
        self.target_repr_head = nn.Sequential(
            nn.Linear(state_dim + 6 + self.contact_repr_dim + 1, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, self.target_repr_dim),
            nn.ReLU(inplace=True),
        )

        policy_in_dim = state_dim + self.target_repr_dim + phase_emb_dim + stage_emb_dim
        self.policy_trunk = nn.Sequential(
            nn.Linear(policy_in_dim, state_dim),
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
        self.y_bridge_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, self.horizon),
        )
        self.progress_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 3))
        self.risk_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 1))
        self.stop_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 1))
        self.apply_confidence_head = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 1))

        for module in (
            self.target_delta_head[-1],
            self.contact_repr_head[-1],
            self.confidence_head[-1],
            self.close_ready_head[-1],
            self.handoff_ready_head[-1],
            self.progress_prior_head[-1],
            self.yaw_direction_head[-1],
            self.traj_head[-1],
            self.progress_head[-1],
            self.risk_head[-1],
            self.stop_head[-1],
            self.apply_confidence_head[-1],
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.y_bridge_head[-1].weight)
        nn.init.zeros_(self.y_bridge_head[-1].bias)
        nn.init.constant_(self.scale_head[-1].bias, -3.0)
        nn.init.zeros_(self.scale_head[-1].weight)

    def _embed_phase_stage(
        self,
        phase_id: torch.Tensor | None,
        stage_bucket_id: torch.Tensor | None,
        bsz: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if phase_id is None:
            phase_id = torch.zeros((bsz,), dtype=torch.long, device=device)
        else:
            phase_id = phase_id.reshape(-1).to(device=device, dtype=torch.long).clamp_min(0)
        if stage_bucket_id is None:
            stage_bucket_id = torch.zeros((bsz,), dtype=torch.long, device=device)
        else:
            stage_bucket_id = stage_bucket_id.reshape(-1).to(device=device, dtype=torch.long).clamp_min(0)
        phase_emb = self.phase_embedding(phase_id.clamp_max(self.phase_embedding.num_embeddings - 1))
        stage_emb = self.stage_embedding(stage_bucket_id.clamp_max(self.stage_embedding.num_embeddings - 1))
        return phase_emb, stage_emb

    def encode_observation(
        self,
        wrist_depth: torch.Tensor,
        force_history: torch.Tensor,
        proprio: torch.Tensor,
        planner_action_local: torch.Tensor,
        gripper_context: torch.Tensor | None = None,
        front_rgb: torch.Tensor | None = None,
        wrist_rgb: torch.Tensor | None = None,
        phase_id: torch.Tensor | None = None,
        stage_bucket_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz = int(proprio.shape[0])
        dtype = proprio.dtype
        device = proprio.device
        phase_emb, stage_emb = self._embed_phase_stage(phase_id, stage_bucket_id, bsz, device)
        parts: list[torch.Tensor] = [phase_emb.to(dtype=dtype), stage_emb.to(dtype=dtype)]
        if self.use_front_rgb:
            parts.append(_maybe_interp_rgb(front_rgb, self.front_rgb_encoder, bsz, device, dtype))
        if self.use_wrist_rgb:
            parts.append(_maybe_interp_rgb(wrist_rgb, self.wrist_rgb_encoder, bsz, device, dtype))
        if self.use_wrist_depth:
            parts.append(_maybe_interp_depth(wrist_depth, self.depth_encoder, bsz, device, dtype))
        if self.use_force:
            parts.append(self.force_encoder(force_history.to(device=device, dtype=dtype)))
        parts.append(self.proprio_encoder(proprio.to(device=device, dtype=dtype)))
        if self.use_planner_action:
            parts.append(self.planner_encoder(planner_action_local.to(device=device, dtype=dtype)))
        if gripper_context is None:
            gripper_context = torch.zeros((bsz, 4), dtype=dtype, device=device)
        if gripper_context.ndim == 1:
            gripper_context = gripper_context.unsqueeze(0)
        if gripper_context.shape[-1] < 4:
            pad = torch.zeros((bsz, 4 - gripper_context.shape[-1]), dtype=dtype, device=device)
            gripper_context = torch.cat([gripper_context.to(device=device, dtype=dtype), pad], dim=-1)
        parts.append(self.gripper_encoder(gripper_context[:, :4].to(device=device, dtype=dtype)))
        obs_state = self.obs_trunk(torch.cat(parts, dim=-1))
        return {"obs_state": obs_state, "phase_emb": phase_emb, "stage_emb": stage_emb}

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
        phase_id: torch.Tensor | None = None,
        stage_bucket_id: torch.Tensor | None = None,
        teacher_target_delta_local: torch.Tensor | None = None,
        teacher_contact_repr: torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
        teacher_contact_force_prob: float = 0.0,
        noise: torch.Tensor | None = None,
        noise_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_observation(
            wrist_depth=wrist_depth,
            force_history=force_history,
            proprio=proprio,
            planner_action_local=planner_action_local,
            gripper_context=gripper_context,
            front_rgb=front_rgb,
            wrist_rgb=wrist_rgb,
            phase_id=phase_id,
            stage_bucket_id=stage_bucket_id,
        )
        obs_state = encoded["obs_state"]
        pred_target_delta = self.target_delta_head(obs_state)
        pred_contact_repr = self.contact_repr_head(obs_state)
        target_confidence_logit = self.confidence_head(obs_state).squeeze(-1)
        target_confidence = torch.sigmoid(target_confidence_logit)
        close_ready_logit = self.close_ready_head(obs_state).squeeze(-1)
        close_ready = torch.sigmoid(close_ready_logit)
        handoff_ready_logit = self.handoff_ready_head(obs_state).squeeze(-1)
        handoff_ready = torch.sigmoid(handoff_ready_logit)
        progress_prior_logits = self.progress_prior_head(obs_state)
        yaw_direction_logits = self.yaw_direction_head(obs_state)

        target_delta_for_repr = pred_target_delta
        if teacher_target_delta_local is not None and float(teacher_force_prob) > 0.0:
            teacher = teacher_target_delta_local.to(device=pred_target_delta.device, dtype=pred_target_delta.dtype)
            if float(teacher_force_prob) >= 1.0:
                target_delta_for_repr = teacher
            else:
                target_delta_for_repr = (1.0 - float(teacher_force_prob)) * pred_target_delta + float(teacher_force_prob) * teacher

        contact_repr_for_repr = pred_contact_repr
        if teacher_contact_repr is not None and float(teacher_contact_force_prob) > 0.0:
            teacher_contact = teacher_contact_repr.to(device=pred_contact_repr.device, dtype=pred_contact_repr.dtype)
            if float(teacher_contact_force_prob) >= 1.0:
                contact_repr_for_repr = teacher_contact
            else:
                contact_repr_for_repr = (1.0 - float(teacher_contact_force_prob)) * pred_contact_repr + float(teacher_contact_force_prob) * teacher_contact

        repr_input = torch.cat(
            [obs_state, target_delta_for_repr, contact_repr_for_repr, target_confidence.unsqueeze(-1)],
            dim=-1,
        )
        target_repr = self.target_repr_head(repr_input)

        state = self.policy_trunk(torch.cat([obs_state, target_repr, encoded["phase_emb"], encoded["stage_emb"]], dim=-1))
        mean = self.traj_head(state).view(-1, self.horizon, self.action_dim)
        scale = self.scale_head(state).view(-1, self.horizon, self.action_dim).clamp(-6.0, 1.0).exp()
        raw = mean if noise is None else mean + noise.to(device=mean.device, dtype=mean.dtype) * scale * float(noise_scale)
        traj_4d_main, traj_6d_main = self._bound_trajectory(raw)
        y_bridge = torch.tanh(self.y_bridge_head(state)).view(-1, self.horizon, 1) * float(self.y_bridge_max_step)
        y_bridge_4d = torch.zeros_like(traj_4d_main)
        y_bridge_4d[..., 1:2] = y_bridge
        traj_4d = traj_4d_main + y_bridge_4d
        traj_6d = torch.zeros((*traj_4d.shape[:-1], 6), device=traj_4d.device, dtype=traj_4d.dtype)
        traj_6d[..., :3] = traj_4d[..., :3]
        traj_6d[..., 5] = traj_4d[..., 3]

        apply_confidence_logit = self.apply_confidence_head(state).squeeze(-1)
        out = {
            "obs_state": obs_state,
            "pred_target_delta_local_6d": pred_target_delta,
            "pred_contact_repr": pred_contact_repr,
            "target_confidence_logit": target_confidence_logit,
            "target_confidence": target_confidence,
            "close_ready_logit": close_ready_logit,
            "close_ready": close_ready,
            "handoff_ready_logit": handoff_ready_logit,
            "handoff_ready": handoff_ready,
            "progress_prior_logits": progress_prior_logits,
            "yaw_direction_logits": yaw_direction_logits,
            "target_repr": target_repr,
            "state": state,
            "trajectory_mean_4d": mean,
            "trajectory_scale_4d": scale,
            "trajectory_main_4d": traj_4d_main,
            "trajectory_main_6d": traj_6d_main,
            "y_bridge_trajectory_4d": y_bridge_4d,
            "trajectory_4d": traj_4d,
            "trajectory_6d": traj_6d,
            "first_residual_4d": traj_4d[:, 0],
            "first_residual_6d": traj_6d[:, 0],
            "progress_logits": self.progress_head(state),
            "risk_logit": self.risk_head(state).squeeze(-1),
            "stop_logit": self.stop_head(state).squeeze(-1),
            "apply_confidence_logit": apply_confidence_logit,
            "apply_confidence": torch.sigmoid(apply_confidence_logit),
        }
        return out

    @torch.no_grad()
    def sample_candidates(self, num_samples: int, top_k: int | None = None, **kwargs) -> dict[str, torch.Tensor]:
        num_samples = max(int(num_samples), 1)
        out = self.forward(**kwargs)
        bsz = out["state"].shape[0]
        mean = out["trajectory_mean_4d"]
        scale = out["trajectory_scale_4d"]
        if num_samples == 1:
            raw = mean[:, None]
        else:
            noise = torch.randn((bsz, num_samples - 1, self.horizon, self.action_dim), device=mean.device, dtype=mean.dtype)
            raw = torch.cat([mean[:, None], mean[:, None] + noise * scale[:, None]], dim=1)
        flat4, flat6 = self._bound_trajectory(raw.reshape(-1, self.horizon, self.action_dim))
        traj4 = flat4.view(bsz, num_samples, self.horizon, self.action_dim)
        traj6 = flat6.view(bsz, num_samples, self.horizon, 6)

        progress = out["progress_logits"]
        risk = out["risk_logit"]
        stop = out["stop_logit"]
        apply_conf = out["apply_confidence"]
        base_score = progress.sum(dim=-1) - risk - torch.sigmoid(stop) + 0.25 * apply_conf
        effort = torch.linalg.norm(traj4[..., 0, :3], dim=-1) + 0.25 * traj4[..., 0, 3].abs()
        candidate_scores = base_score[:, None] - 0.05 * effort
        best_index = torch.argmax(candidate_scores, dim=1)
        gather4 = best_index.view(bsz, 1, 1, 1).expand(-1, 1, self.horizon, self.action_dim)
        gather6 = best_index.view(bsz, 1, 1, 1).expand(-1, 1, self.horizon, 6)
        best4 = torch.gather(traj4, 1, gather4).squeeze(1)
        best6 = torch.gather(traj6, 1, gather6).squeeze(1)
        diversity = traj4[..., 0, :].std(dim=1).mean(dim=-1) if num_samples > 1 else torch.zeros((bsz,), device=mean.device)
        return {
            **out,
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
        }
