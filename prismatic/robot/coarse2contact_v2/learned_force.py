"""Learned force/contact trigger for Coarse2Contact v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_vocab(values: Iterable[str], *, add_unknown: bool = True) -> dict[str, int]:
    vocab: dict[str, int] = {}
    if add_unknown:
        vocab["<unk>"] = 0
    for value in values:
        key = str(value)
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab


def _lookup(vocab: Mapping[str, int], value: str) -> int:
    if value in vocab:
        return int(vocab[value])
    return int(vocab.get("<unk>", 0))


class ForceContactClassifierNet(nn.Module):
    """Sequence classifier for contact / jam / misgrasp / slip / recovery-needed."""

    def __init__(
        self,
        *,
        skill_type_vocab: Mapping[str, int],
        stage_vocab: Mapping[str, int],
        input_dim: int = 27,
        embed_dim: int = 16,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.skill_type_vocab = dict(skill_type_vocab)
        self.stage_vocab = dict(stage_vocab)
        self.skill_emb = nn.Embedding(max(len(self.skill_type_vocab), 1), embed_dim)
        self.stage_emb = nn.Embedding(max(len(self.stage_vocab), 1), embed_dim)
        self.gru = nn.GRU(input_dim + embed_dim * 2, hidden_dim, batch_first=True, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5),
        )

    @classmethod
    def from_vocab(cls, vocab: Mapping[str, Mapping[str, int]], **kwargs) -> "ForceContactClassifierNet":
        return cls(skill_type_vocab=vocab["skill_type"], stage_vocab=vocab["stage_name"], **kwargs)

    def forward(
        self,
        sequence_features: torch.Tensor,
        skill_type_id: torch.Tensor,
        stage_id: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequence_features.ndim != 3:
            raise ValueError("sequence_features must have shape [B, T, F]")
        bsz, steps, _ = sequence_features.shape
        skill_feat = self.skill_emb(skill_type_id).unsqueeze(1).expand(bsz, steps, -1)
        stage_feat = self.stage_emb(stage_id).unsqueeze(1).expand(bsz, steps, -1)
        x = torch.cat([sequence_features, skill_feat, stage_feat], dim=-1)
        out, _ = self.gru(x)
        if lengths is None:
            last = out[:, -1, :]
        else:
            idx = torch.clamp(lengths.long() - 1, min=0)
            last = out[torch.arange(out.shape[0], device=out.device), idx]
        return self.head(last)

    def predict(
        self,
        sequence_features: torch.Tensor,
        skill_type_id: torch.Tensor,
        stage_id: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.forward(sequence_features, skill_type_id, stage_id, lengths=lengths)
        probs = torch.sigmoid(logits)
        return {
            "contact": probs[..., 0],
            "jam": probs[..., 1],
            "misgrasp": probs[..., 2],
            "slip": probs[..., 3],
            "recovery_needed": probs[..., 4],
        }


@dataclass(frozen=True)
class ForceContactPrediction:
    contact: float
    jam: float
    misgrasp: float
    slip: float
    recovery_needed: float


def load_force_classifier_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[ForceContactClassifierNet, dict[str, dict[str, int]]]:
    ckpt = torch.load(Path(path), map_location=map_location)
    vocab = ckpt["vocab"]
    model = ForceContactClassifierNet.from_vocab(vocab, **ckpt.get("config", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


class LearnedForceClassifierAdapter:
    """Runtime adapter that exposes a trained force/contact classifier."""

    def __init__(self, model: ForceContactClassifierNet, vocab: Mapping[str, Mapping[str, int]], *, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.vocab = {name: dict(mapping) for name, mapping in vocab.items()}
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "LearnedForceClassifierAdapter":
        model, vocab = load_force_classifier_checkpoint(path, map_location=device)
        return cls(model, vocab, device=device)

    def predict(self, sequence_features: torch.Tensor, *, skill_type: str = "", stage_name: str = "", lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        skill_id = torch.tensor([_lookup(self.vocab["skill_type"], skill_type)], dtype=torch.long, device=self.device)
        stage_id = torch.tensor([_lookup(self.vocab["stage_name"], stage_name)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            probs = self.model.predict(sequence_features.to(self.device), skill_id, stage_id, lengths=lengths)
        return probs
