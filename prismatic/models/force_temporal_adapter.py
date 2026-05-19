"""
force_temporal_adapter.py

Lightweight 1D-CNN adapter that encodes a temporal window of force readings into a sequence
of tokens compatible with the action head's cross-attention mechanism.

Input : (B, FORCE_DIM, FORCE_HISTORY_LEN)  — z-score normalized force history
Output: (B, FORCE_TOKENS, llm_dim)          — force token embeddings
"""

import torch
import torch.nn as nn

from prismatic.vla.constants import FORCE_DIM, FORCE_HISTORY_LEN, FORCE_TOKENS


class ForceTemporalAdapter(nn.Module):
    def __init__(
        self,
        force_dim: int = FORCE_DIM,
        history_len: int = FORCE_HISTORY_LEN,
        llm_dim: int = 896,
        num_tokens: int = FORCE_TOKENS,
    ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.num_tokens = num_tokens

        # 3-layer 1D-CNN over temporal dimension
        self.encoder = nn.Sequential(
            nn.Conv1d(force_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Pool temporal dimension to fixed token count
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

        # Project to LLM dimension — zero-init for stable training start
        self.proj = nn.Linear(256, llm_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Normalize output to prevent scale explosion in downstream cross-attention
        self.out_ln = nn.LayerNorm(llm_dim)

    def forward(self, force_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            force_history: (B, FORCE_DIM, FORCE_HISTORY_LEN) — z-score normalized force readings
                           OR (B, FORCE_HISTORY_LEN, FORCE_DIM) — will be transposed automatically
        Returns:
            (B, num_tokens, llm_dim) — force token embeddings
        """
        # Auto-detect layout: Conv1d expects (B, C, L) where C=force_dim
        if force_history.shape[-1] == FORCE_DIM and force_history.shape[-2] != FORCE_DIM:
            force_history = force_history.transpose(-1, -2)  # (B, FORCE_DIM, L)

        x = self.encoder(force_history)    # (B, 256, L)
        x = self.pool(x)                  # (B, 256, num_tokens)
        x = x.transpose(1, 2)             # (B, num_tokens, 256)
        x = self.proj(x)                  # (B, num_tokens, llm_dim)
        x = self.out_ln(x)                # (B, num_tokens, llm_dim)
        return x
