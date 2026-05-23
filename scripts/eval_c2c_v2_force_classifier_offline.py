#!/usr/bin/env python3
"""Offline evaluation for the Coarse2Contact v2 force/contact classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import ForceContactJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_force import ForceContactClassifierNet
from scripts.train_c2c_v2_force_classifier import _collate


def _evaluate(model: ForceContactClassifierNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0}
    steps = 0
    labels = ["contact", "jam", "misgrasp", "slip", "recovery_needed"]
    for label in labels:
        totals[f"{label}_acc"] = 0.0
        totals[f"{label}_pos_rate"] = 0.0
    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence_features"].to(device)
            lengths = batch["lengths"].to(device)
            skill_type_id = batch["skill_type_id"].to(device)
            stage_id = batch["stage_id"].to(device)
            y = batch["labels"].to(device)
            logits = model(seq, skill_type_id, stage_id, lengths=lengths)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            totals["loss"] += float(loss.item())
            for idx, name in enumerate(labels):
                totals[f"{name}_acc"] += float((pred[:, idx] == y[:, idx]).float().mean().item())
                totals[f"{name}_pos_rate"] += float(probs[:, idx].mean().item())
            steps += 1
    if steps == 0:
        return {}
    return {k: v / steps for k, v in totals.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    dataset = ForceContactJsonlDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: _collate(batch, vocab))
    model = ForceContactClassifierNet.from_vocab(vocab, **ckpt.get("config", {})).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = _evaluate(model, loader, torch.device(args.device))
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "metrics": metrics,
    }
    out_path = args.output_root / "force_classifier_eval.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
