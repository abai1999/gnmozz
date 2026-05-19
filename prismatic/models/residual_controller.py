"""
residual_controller.py

Route 1 MVP: encoder pooling + MLP fusion for per-step residual correction.

Inputs:
  - wrist_depth: (B, 1, 96, 96)
  - ft_hist:     (B, 32, 6)
  - proprio:     (B, 15)
  - base_action: (B, 6)     — 6D pose only, no gripper
  - gripper_context: (B, 3) — optional [current, min-lookahead, mean-lookahead]
  - step_idx:    (B,)       — chunk step index 0..7

Output:
  - delta_pose:        (B, 6)    — raw local residual
  - delta_pose_gated:  (B, 6)    — alpha-gated residual for backward compatibility
  - alpha:             (B,)      — confidence gate in [0,1]
  - ready_to_close:    (B,)      — learned near-field readiness probability
  - hold_after_close:  (B,)      — learned hold probability derived from gripper logits
  - gripper_logits:    (B, 3)    — 0=open, 1=close, 2=hold

Last layer is zero-initialized so initial residual ≈ 0.
Total parameters: ~0.5-1M.
"""

import torch
import torch.nn as nn

from prismatic.models.residual_encoders import (
    BaseActionEncoder,
    DepthEncoderTiny,
    ForceEncoderTiny,
    ProprioEncoder,
    StepEmbedding,
)


class ResidualController(nn.Module):
    """Minimal MLP-based residual controller (Route 1 MVP)."""

    def __init__(
        self,
        depth_dim: int = 128,
        force_dim: int = 64,
        proprio_dim: int = 64,
        action_dim: int = 32,
        step_dim: int = 16,
        hidden_dims: tuple = (256, 256, 128),
        output_dim: int = 6,
        proprio_input_dim: int = 15,
        force_input_dim: int = 6,
        force_history_len: int = 32,
        num_chunk_steps: int = 8,
        phase_vocab_size: int = 4,
        phase_dim: int = 8,
        context_dim: int = 16,
        gripper_context_dim: int = 8,
        gripper_input_dim: int = 3,
        gripper_state_dim: int = 3,
        pose_output_mode: str = "gated",
        pose_use_depth: bool = True,
        pose_use_force: bool = True,
        pose_use_proprio: bool = True,
        pose_use_action: bool = True,
        fire_only_head: bool = False,
        ready_use_context: bool = True,
        ready_use_gripper_context: bool = True,
    ):
        super().__init__()
        if pose_output_mode not in ("gated", "raw"):
            raise ValueError(f"Unsupported pose_output_mode: {pose_output_mode}")
        self.pose_output_mode = pose_output_mode
        self.pose_use_depth = bool(pose_use_depth)
        self.pose_use_force = bool(pose_use_force)
        self.pose_use_proprio = bool(pose_use_proprio)
        self.pose_use_action = bool(pose_use_action)
        self.fire_only_head = bool(fire_only_head)
        self.ready_use_context = bool(ready_use_context)
        self.ready_use_gripper_context = bool(ready_use_gripper_context)

        # Encoders
        self.depth_encoder = DepthEncoderTiny(out_dim=depth_dim)
        self.force_encoder = ForceEncoderTiny(
            force_dim=force_input_dim, history_len=force_history_len, out_dim=force_dim
        )
        self.proprio_encoder = ProprioEncoder(proprio_dim=proprio_input_dim, out_dim=proprio_dim)
        self.action_encoder = BaseActionEncoder(action_dim=6, out_dim=action_dim)
        self.step_embedding = StepEmbedding(num_steps=num_chunk_steps, out_dim=step_dim)
        self.phase_embedding = nn.Embedding(phase_vocab_size, phase_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
        )
        self.gripper_context_encoder = nn.Sequential(
            nn.Linear(gripper_input_dim, gripper_context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gripper_context_dim, gripper_context_dim),
        )

        # Fusion MLP. Keep this trunk shape stable so older pose-only residual
        # checkpoints remain load-compatible; gripper context is only used by
        # the new readiness/gripper heads below.
        concat_dim = depth_dim + force_dim + proprio_dim + action_dim + step_dim + phase_dim + context_dim
        layers = []
        in_dim = concat_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim

        self.fusion_mlp = nn.Sequential(*layers)

        self.output_head = nn.Linear(in_dim, output_dim)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        self.alpha_head = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, -2.0)

        readiness_in_dim = in_dim + gripper_context_dim
        self.ready_head = nn.Linear(readiness_in_dim, 1)
        nn.init.zeros_(self.ready_head.weight)
        nn.init.constant_(self.ready_head.bias, -2.0)

        if self.fire_only_head:
            self.gripper_head = None
        else:
            self.gripper_head = nn.Linear(readiness_in_dim, gripper_state_dim)
            nn.init.zeros_(self.gripper_head.weight)
            nn.init.zeros_(self.gripper_head.bias)
            # Bias toward "open" until readiness/gripper labels are trained.
            with torch.no_grad():
                self.gripper_head.bias[0] = 1.0
        self._readiness_heads_loaded = True

    def forward(
        self,
        wrist_depth: torch.Tensor,
        ft_hist: torch.Tensor,
        proprio: torch.Tensor,
        base_action: torch.Tensor,
        step_idx: torch.Tensor,
        phase_id: torch.Tensor = None,
        phase_age: torch.Tensor = None,
        steps_since_last_replan: torch.Tensor = None,
        gripper_context: torch.Tensor = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            wrist_depth: (B, 1, 96, 96)
            ft_hist:     (B, 32, 6)
            proprio:     (B, 15)
            base_action: (B, 6)
            step_idx:    (B,) long
            phase_id:    (B,) long
            phase_age:   (B,) float
            steps_since_last_replan: (B,) float
            gripper_context: (B,3) float, optional

        Returns:
            default: pose residual selected by pose_output_mode
            if return_aux=True: dict with raw/gated delta, alpha, and readiness heads
        """
        h_d = self.depth_encoder(wrist_depth)    # (B, 128)
        h_f = self.force_encoder(ft_hist)        # (B, 64)
        h_p = self.proprio_encoder(proprio)      # (B, 64)
        h_a = self.action_encoder(base_action)   # (B, 32)
        h_s = self.step_embedding(step_idx)      # (B, 16)
        if phase_id is None:
            phase_id = torch.zeros_like(step_idx)
        if phase_age is None:
            phase_age = torch.zeros_like(step_idx, dtype=base_action.dtype)
        if steps_since_last_replan is None:
            steps_since_last_replan = torch.zeros_like(step_idx, dtype=base_action.dtype)
        h_phase = self.phase_embedding(torch.clamp(phase_id.long(), min=0, max=self.phase_embedding.num_embeddings - 1))
        context = torch.stack(
            [
                torch.clamp(phase_age.to(base_action.dtype), min=0.0, max=32.0) / 32.0,
                torch.clamp(steps_since_last_replan.to(base_action.dtype), min=0.0, max=32.0) / 32.0,
            ],
            dim=-1,
        )
        h_ctx = self.context_encoder(context)
        if gripper_context is None:
            gripper_context = torch.zeros(
                base_action.shape[0],
                self.gripper_context_encoder[0].in_features,
                device=base_action.device,
                dtype=base_action.dtype,
            )
        h_g = self.gripper_context_encoder(gripper_context.to(base_action.dtype))
        ready_h_ctx = h_ctx if self.ready_use_context else torch.zeros_like(h_ctx)
        ready_h_g = h_g if self.ready_use_gripper_context else torch.zeros_like(h_g)

        # Use full features for readiness/gripper, but allow the pose head to
        # ablate specific inputs so we can test whether depth/force are
        # collapsing the local correction field.
        fused_full = torch.cat([h_d, h_f, h_p, h_a, h_s, h_phase, ready_h_ctx], dim=-1)
        pose_h_d = h_d if self.pose_use_depth else torch.zeros_like(h_d)
        pose_h_f = h_f if self.pose_use_force else torch.zeros_like(h_f)
        pose_h_p = h_p if self.pose_use_proprio else torch.zeros_like(h_p)
        pose_h_a = h_a if self.pose_use_action else torch.zeros_like(h_a)
        fused_pose = torch.cat([pose_h_d, pose_h_f, pose_h_p, pose_h_a, h_s, h_phase, h_ctx], dim=-1)

        pose_hidden = self.fusion_mlp(fused_pose)
        hidden = self.fusion_mlp(fused_full)
        delta_pose = self.output_head(pose_hidden)
        alpha = torch.sigmoid(self.alpha_head(pose_hidden)).squeeze(-1)  # (B,)
        delta_pose_gated = delta_pose * alpha.unsqueeze(-1)
        readiness_hidden = torch.cat([hidden, ready_h_g], dim=-1)
        ready_to_close = torch.sigmoid(self.ready_head(readiness_hidden)).squeeze(-1)
        gripper_logits = None
        hold_after_close = None
        if self.gripper_head is not None:
            gripper_logits = self.gripper_head(readiness_hidden)
            hold_after_close = torch.softmax(gripper_logits, dim=-1)[..., 2]

        if return_aux:
            outputs = {
                "delta_pose": delta_pose,
                "alpha": alpha,
                "delta_pose_gated": delta_pose_gated,
                "ready_to_close": ready_to_close,
            }
            if hold_after_close is not None:
                outputs["hold_after_close"] = hold_after_close
            if gripper_logits is not None:
                outputs["gripper_logits"] = gripper_logits
            return outputs
        return delta_pose if self.pose_output_mode == "raw" else delta_pose_gated
