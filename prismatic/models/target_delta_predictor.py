"""
target_delta_predictor.py

Lightweight observation-conditioned predictor for deployment-time target deltas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.residual_encoders import DepthEncoderTiny, ProprioEncoder, RGBEncoderTiny


def _normalize_gripper_context(gripper_context: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if gripper_context is None:
        return torch.zeros((1, 3), device=device, dtype=torch.float32)
    if not isinstance(gripper_context, torch.Tensor):
        gripper_context = torch.as_tensor(gripper_context, device=device, dtype=torch.float32)
    else:
        gripper_context = gripper_context.to(device=device, dtype=torch.float32)
    if gripper_context.ndim == 0:
        gripper_context = gripper_context.view(1, 1)
    if gripper_context.ndim == 1:
        gripper_context = gripper_context.unsqueeze(0)
    if gripper_context.shape[-1] < 3:
        pad = torch.zeros((*gripper_context.shape[:-1], 3 - gripper_context.shape[-1]), device=device, dtype=torch.float32)
        gripper_context = torch.cat([gripper_context, pad], dim=-1)
    elif gripper_context.shape[-1] > 3:
        gripper_context = gripper_context[..., :3]
    return gripper_context.contiguous()


class TargetDeltaPredictor(nn.Module):
    def __init__(
        self,
        rgb_dim: int = 128,
        depth_dim: int = 128,
        proprio_dim: int = 64,
        context_dim: int = 64,
        hidden_dim: int = 256,
        substage_vocab: int = 8,
        contact_vocab: int = 8,
        target_mode_vocab: int = 8,
        legacy_output_head: bool = False,
    ):
        super().__init__()
        self.legacy_output_head = bool(legacy_output_head)
        self.front_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.wrist_rgb_encoder = RGBEncoderTiny(out_dim=rgb_dim)
        self.depth_encoder = DepthEncoderTiny(out_dim=depth_dim)
        self.proprio_encoder = ProprioEncoder(proprio_dim=15, out_dim=proprio_dim)
        self.gripper_context = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )
        self.substage_emb = nn.Embedding(substage_vocab, 16)
        self.contact_emb = nn.Embedding(contact_vocab, 8)
        self.target_mode_emb = nn.Embedding(target_mode_vocab, 8)
        self.context_mlp = nn.Sequential(
            nn.Linear(1 + 32 + 16 + 8 + 8, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(inplace=True),
        )
        fused_dim = rgb_dim * 2 + depth_dim + proprio_dim + context_dim
        if self.legacy_output_head:
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 6),
            )
            self.delta_head = None
            self.handoff_metric_head = None
            self.handoff_ready_head = None
            self.supports_handoff_aux = False
        else:
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.delta_head = nn.Linear(hidden_dim, 6)
            self.handoff_metric_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 4),
            )
            self.handoff_ready_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.supports_handoff_aux = True

    def forward(
        self,
        front_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        wrist_depth: torch.Tensor,
        proprio: torch.Tensor,
        gripper_context: torch.Tensor,
        has_object_in_hand: torch.Tensor,
        substage_id: torch.Tensor,
        contact_state: torch.Tensor,
        stage_target_mode: torch.Tensor,
        return_aux: bool = False,
    ):
        if front_rgb.shape[-1] != 96 or front_rgb.shape[-2] != 96:
            front_rgb = F.interpolate(front_rgb, size=(96, 96), mode="bilinear", align_corners=False)
        if wrist_rgb.shape[-1] != 96 or wrist_rgb.shape[-2] != 96:
            wrist_rgb = F.interpolate(wrist_rgb, size=(96, 96), mode="bilinear", align_corners=False)
        if wrist_depth.shape[-1] != 96 or wrist_depth.shape[-2] != 96:
            wrist_depth = F.interpolate(wrist_depth, size=(96, 96), mode="bilinear", align_corners=False)

        h_front = self.front_rgb_encoder(front_rgb)
        h_wrist = self.wrist_rgb_encoder(wrist_rgb)
        h_depth = self.depth_encoder(wrist_depth)
        h_prop = self.proprio_encoder(proprio)
        gripper_context = _normalize_gripper_context(gripper_context, device=front_rgb.device)
        h_grip = self.gripper_context(gripper_context)
        h_ctx = self.context_mlp(
            torch.cat(
                [
                    has_object_in_hand.float().view(-1, 1),
                    h_grip,
                    self.substage_emb(substage_id.long()),
                    self.contact_emb(contact_state.long()),
                    self.target_mode_emb(stage_target_mode.long()),
                ],
                dim=-1,
            )
        )
        fused = torch.cat([h_front, h_wrist, h_depth, h_prop, h_ctx], dim=-1)
        if self.legacy_output_head:
            delta = self.head(fused)
            hidden = None
        else:
            hidden = self.head(fused)
            delta = self.delta_head(hidden)
        if not return_aux:
            return delta
        if self.legacy_output_head:
            raise RuntimeError("Legacy TargetDeltaPredictor does not support auxiliary handoff outputs.")
        handoff_metrics = self.handoff_metric_head(hidden)
        # Metrics are non-negative distances/errors.
        handoff_metrics = F.softplus(handoff_metrics)
        handoff_ready_logit = self.handoff_ready_head(hidden).squeeze(-1)
        return {
            "target_delta": delta,
            "handoff_metrics": handoff_metrics,
            "handoff_ready_logit": handoff_ready_logit,
        }

    @torch.no_grad()
    def predict(
        self,
        front_rgb,
        wrist_rgb,
        wrist_depth,
        proprio,
        has_object_in_hand=0.0,
        contact_state=0,
        substage_id=0,
        target_mode=0,
        gripper_context=None,
    ):
        device = next(self.parameters()).device

        def _as_tensor(arr, *, dtype=torch.float32):
            if isinstance(arr, torch.Tensor):
                return arr.to(device=device, dtype=dtype)
            return torch.as_tensor(arr, device=device, dtype=dtype)

        def _prep_rgb(arr):
            ten = _as_tensor(arr)
            if ten.ndim == 3 and ten.shape[-1] == 3:
                ten = ten.permute(2, 0, 1)
            elif ten.ndim == 4 and ten.shape[-1] == 3:
                ten = ten.permute(0, 3, 1, 2)
            if ten.ndim == 3:
                ten = ten.unsqueeze(0)
            if float(ten.max().item()) > 1.5:
                ten = ten / 255.0
            return ten.contiguous()

        def _prep_depth(arr):
            ten = _as_tensor(arr)
            if ten.ndim == 2:
                ten = ten.unsqueeze(0).unsqueeze(0)
            elif ten.ndim == 3:
                if ten.shape[0] == 1:
                    ten = ten.unsqueeze(0)
                elif ten.shape[-1] == 1:
                    ten = ten.permute(2, 0, 1).unsqueeze(0)
                elif ten.shape[-1] != 96:
                    ten = ten.unsqueeze(1)
            elif ten.ndim == 4 and ten.shape[-1] == 1:
                ten = ten.permute(0, 3, 1, 2)
            ten = torch.clamp(ten, 0.0, 1.0)
            return ten.contiguous()

        front_rgb = _prep_rgb(front_rgb)
        wrist_rgb = _prep_rgb(wrist_rgb)
        wrist_depth = _prep_depth(wrist_depth)
        proprio = _as_tensor(proprio)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if gripper_context is None:
            gripper_context = torch.zeros(front_rgb.shape[0], 3, device=device, dtype=torch.float32)
        else:
            gripper_context = _normalize_gripper_context(gripper_context, device=device)
        if isinstance(has_object_in_hand, torch.Tensor):
            has_object_in_hand = has_object_in_hand.to(device=device, dtype=torch.float32)
        else:
            has_object_in_hand = _as_tensor(has_object_in_hand)
        if has_object_in_hand.ndim == 0:
            has_object_in_hand = has_object_in_hand.view(1)
        if has_object_in_hand.ndim == 1:
            has_object_in_hand = has_object_in_hand.reshape(-1)
        if has_object_in_hand.numel() == 1 and front_rgb.shape[0] > 1:
            has_object_in_hand = has_object_in_hand.expand(front_rgb.shape[0]).contiguous()
        if has_object_in_hand.shape[0] != front_rgb.shape[0]:
            has_object_in_hand = has_object_in_hand.reshape(front_rgb.shape[0])
        substage_id = torch.full((front_rgb.shape[0],), int(substage_id), device=device, dtype=torch.long)
        contact_state = torch.full((front_rgb.shape[0],), int(contact_state), device=device, dtype=torch.long)
        stage_target_mode = torch.full((front_rgb.shape[0],), int(target_mode), device=device, dtype=torch.long)
        out = self(
            front_rgb=front_rgb,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            proprio=proprio,
            gripper_context=gripper_context,
            has_object_in_hand=has_object_in_hand,
            substage_id=substage_id,
            contact_state=contact_state,
            stage_target_mode=stage_target_mode,
            return_aux=False,
        )
        if isinstance(out, dict):
            out = out.get("target_delta", out)
        out = out.detach().float().cpu().numpy()
        if out.ndim == 2 and out.shape[0] == 1:
            out = out[0]
        return out
