"""
residual_encoders.py

Lightweight encoders for the residual controller (Route 1 MVP).
Each encoder produces a fixed-dimensional embedding:
  - DepthEncoderTiny:   (B,1,96,96) -> (B,128)
  - ForceEncoderTiny:   (B,32,6)    -> (B,64)
  - ProprioEncoder:     (B,15)      -> (B,64)
  - BaseActionEncoder:  (B,6)       -> (B,32)
  - StepEmbedding:      (B,)        -> (B,16)
"""

import torch
import torch.nn as nn


class DepthEncoderTiny(nn.Module):
    """3-layer CNN with global + center pooling on 96x96 depth -> 128-d vector."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 96 -> 48
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48 -> 24
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 24 -> 12
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.center_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Linear(128 + 128, out_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, 96, 96)
        Returns:
            (B, out_dim)
        """
        fmap = self.encoder(depth)                     # (B, 128, 12, 12)
        global_feat = self.global_pool(fmap).flatten(1)
        center_feat = self.center_pool(fmap[:, :, 4:8, 4:8]).mean(dim=(-1, -2))
        x = torch.cat([global_feat, center_feat], dim=-1)
        return self.fc(x)


class RGBEncoderTiny(nn.Module):
    """3-layer CNN with global pooling on RGB -> fixed-d vector."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        fmap = self.encoder(rgb)
        return self.fc(self.pool(fmap).flatten(1))


class ForceEncoderTiny(nn.Module):
    """2-layer 1D-CNN on force history -> 64-d vector."""

    def __init__(self, force_dim: int = 6, history_len: int = 32, out_dim: int = 64):
        super().__init__()
        self.force_dim = force_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(force_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 64, 1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, ft_hist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ft_hist: (B, 32, 6) — z-score normalized force history
        Returns:
            (B, out_dim)
        """
        # Conv1d expects (B, C, L) where C=force_dim
        if ft_hist.shape[-1] == self.force_dim and ft_hist.ndim == 3:
            ft_hist = ft_hist.transpose(-1, -2)  # (B, 6, 32)
        x = self.encoder(ft_hist)     # (B, 64, 1)
        x = x.flatten(1)             # (B, 64)
        return self.fc(x)            # (B, out_dim)


class ProprioEncoder(nn.Module):
    """MLP: proprio (15D) -> 64-d."""

    def __init__(self, proprio_dim: int = 15, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio: (B, 15)
        Returns:
            (B, out_dim)
        """
        return self.mlp(proprio)


class BaseActionEncoder(nn.Module):
    """MLP: base action 6D pose -> 32-d."""

    def __init__(self, action_dim: int = 6, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_dim),
        )

    def forward(self, base_action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_action: (B, 6) — 6D pose part of planned action
        Returns:
            (B, out_dim)
        """
        return self.mlp(base_action)


class StepEmbedding(nn.Module):
    """Learnable embedding for chunk step index 0..K-1."""

    def __init__(self, num_steps: int = 8, out_dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_steps, out_dim)

    def forward(self, step_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_idx: (B,) long tensor
        Returns:
            (B, out_dim)
        """
        return self.emb(step_idx)
