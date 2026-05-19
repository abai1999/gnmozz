"""
depth_adapter.py

Lightweight CNN adapter that encodes a single-channel depth map into a sequence of tokens
compatible with the action head's cross-attention mechanism.

Input : (B, 1, 224, 224)  — wrist depth image
Output: (B, DEPTH_TOKENS, llm_dim)  — depth token embeddings
"""

import torch
import torch.nn as nn

from prismatic.vla.constants import DEPTH_TOKENS


class DepthAdapter(nn.Module):
    def __init__(self, llm_dim: int = 896, num_tokens: int = DEPTH_TOKENS) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.num_tokens = num_tokens

        # 4-layer CNN: 1 → 32 → 64 → 128 → 256
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 224 → 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 112 → 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 56 → 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 28 → 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Spatial pooling to get fixed token count
        pool_size = int(num_tokens ** 0.5)  # 16 → 4×4
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Project to LLM dimension — zero-init for stable training start
        self.proj = nn.Linear(256, llm_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Normalize output to prevent scale explosion in downstream cross-attention
        self.out_ln = nn.LayerNorm(llm_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, 224, 224) — single-channel depth image in [0, 1]
        Returns:
            (B, num_tokens, llm_dim) — depth token embeddings
        """
        x = self.encoder(depth)           # (B, 256, 14, 14)
        x = self.pool(x)                  # (B, 256, 4, 4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 16, 256)
        x = self.proj(x)                  # (B, 16, llm_dim)
        x = self.out_ln(x)                # (B, 16, llm_dim)
        return x
