from __future__ import annotations

import torch
import torch.nn as nn


class StudentCandidateEvaluatorV2(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 32,
        delta_dim: int = 32,
        summary_dim: int = 16,
        hidden_dim: int = 128,
        yaw_mode_classes: int = 2,
    ):
        super().__init__()
        self.delta_encoder = nn.Sequential(
            nn.Linear(6, delta_dim),
            nn.ReLU(inplace=True),
            nn.Linear(delta_dim, delta_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(6, action_dim),
            nn.ReLU(inplace=True),
            nn.Linear(action_dim, action_dim),
        )
        self.candidate_summary_dim = 12
        self.mode_feature_version = "candidate_yaw_summary_v2"
        self.mode_input_path = "hybrid"
        self.candidate_summary_encoder = nn.Sequential(
            nn.Linear(self.candidate_summary_dim, summary_dim),
            nn.ReLU(inplace=True),
            nn.Linear(summary_dim, summary_dim),
        )
        self.summary_context_head = nn.Sequential(
            nn.Linear(delta_dim + summary_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.context_head = nn.Sequential(
            nn.Linear(latent_dim + delta_dim + summary_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.yaw_mode_head = nn.Linear(hidden_dim, int(yaw_mode_classes))
        self.score_head = nn.Sequential(
            nn.Linear(latent_dim + delta_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.keep_yaw_score_head = nn.Sequential(
            nn.Linear(latent_dim + delta_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply_yaw_score_head = nn.Sequential(
            nn.Linear(latent_dim + delta_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def set_mode_input_path(self, path: str) -> None:
        if path not in {"hybrid", "summary_only"}:
            raise ValueError(f"unknown mode input path: {path}")
        self.mode_input_path = str(path)

    def forward(
        self,
        *,
        handoff_latent: torch.Tensor,
        proxy_current_delta_basin_target: torch.Tensor,
        candidate_actions_local: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        yaw_aware_candidate_scope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_mode(
            handoff_latent=handoff_latent,
            proxy_current_delta_basin_target=proxy_current_delta_basin_target,
            candidate_actions_local=candidate_actions_local,
            candidate_mask=candidate_mask,
            yaw_aware_candidate_scope=yaw_aware_candidate_scope,
        )["candidate_scores"]

    def _candidate_yaw_summary(
        self,
        candidate_actions_local: torch.Tensor,
        proxy_current_delta_basin_target: torch.Tensor,
        candidate_mask: torch.Tensor | None,
        yaw_aware_candidate_scope: torch.Tensor | None,
    ) -> torch.Tensor:
        yaw = candidate_actions_local[:, :, 5].float()
        if yaw_aware_candidate_scope is not None:
            valid = yaw_aware_candidate_scope.float() > 0.5
        elif candidate_mask is not None:
            valid = candidate_mask.float() > 0.5
        else:
            valid = torch.ones_like(yaw, dtype=torch.bool)
        valid_f = valid.float()
        denom = torch.clamp(valid_f.sum(dim=1), min=1.0)
        masked_yaw = yaw.masked_fill(~valid, 0.0)
        yaw_mean = masked_yaw.sum(dim=1) / denom
        yaw_centered = (yaw - yaw_mean.unsqueeze(1)).masked_fill(~valid, 0.0)
        yaw_std = torch.sqrt(torch.clamp((yaw_centered * yaw_centered).sum(dim=1) / denom, min=0.0))
        yaw_abs = torch.abs(yaw).masked_fill(~valid, 0.0)
        yaw_abs_mean = yaw_abs.sum(dim=1) / denom
        yaw_min = yaw.masked_fill(~valid, 1e6).amin(dim=1)
        yaw_max = yaw.masked_fill(~valid, -1e6).amax(dim=1)
        empty = denom <= 1.0e-6
        yaw_min = torch.where(empty, torch.zeros_like(yaw_min), yaw_min)
        yaw_max = torch.where(empty, torch.zeros_like(yaw_max), yaw_max)
        no_yaw_frac = ((yaw_abs <= 0.035) & valid).float().sum(dim=1) / denom
        small_yaw_frac = ((yaw_abs > 0.035) & (yaw_abs <= 0.075) & valid).float().sum(dim=1) / denom
        large_yaw_frac = ((yaw_abs > 0.075) & valid).float().sum(dim=1) / denom
        pos_yaw_frac = ((yaw > 1e-6) & valid).float().sum(dim=1) / denom
        neg_yaw_frac = ((yaw < -1e-6) & valid).float().sum(dim=1) / denom
        proxy_dyaw = proxy_current_delta_basin_target[:, 5].float()
        return torch.stack(
            [
                yaw_min,
                yaw_max,
                yaw_mean,
                yaw_std,
                yaw_abs_mean,
                no_yaw_frac,
                small_yaw_frac,
                large_yaw_frac,
                pos_yaw_frac,
                neg_yaw_frac,
                torch.abs(proxy_dyaw),
                torch.sign(proxy_dyaw),
            ],
            dim=-1,
        )

    def forward_with_mode(
        self,
        *,
        handoff_latent: torch.Tensor,
        proxy_current_delta_basin_target: torch.Tensor,
        candidate_actions_local: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        yaw_aware_candidate_scope: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, num_cands, _ = candidate_actions_local.shape
        h_delta_base = self.delta_encoder(proxy_current_delta_basin_target.float())
        h_delta = h_delta_base.unsqueeze(1).expand(-1, num_cands, -1)
        h_act = self.action_encoder(candidate_actions_local.reshape(bsz * num_cands, 6)).reshape(bsz, num_cands, -1)
        h_latent = handoff_latent.unsqueeze(1).expand(-1, num_cands, -1)
        yaw_summary = self._candidate_yaw_summary(
            candidate_actions_local,
            proxy_current_delta_basin_target,
            candidate_mask,
            yaw_aware_candidate_scope,
        )
        h_summary = self.candidate_summary_encoder(yaw_summary)
        if self.mode_input_path == "summary_only":
            context = self.summary_context_head(torch.cat([h_delta_base, h_summary], dim=-1))
        else:
            context = self.context_head(torch.cat([handoff_latent, h_delta_base, h_summary], dim=-1))
        score_features = torch.cat([h_latent, h_delta, h_act], dim=-1)
        return {
            "candidate_scores": self.score_head(score_features).squeeze(-1),
            "candidate_scores_keep": self.keep_yaw_score_head(score_features).squeeze(-1),
            "candidate_scores_apply": self.apply_yaw_score_head(score_features).squeeze(-1),
            "yaw_mode_logits": self.yaw_mode_head(context),
        }
