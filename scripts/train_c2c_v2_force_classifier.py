#!/usr/bin/env python3
"""Train the Coarse2Contact v2 force/contact classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import ForceContactJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_force import ForceContactClassifierNet


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate(batch: list[dict], vocab: dict[str, dict[str, int]]) -> dict[str, torch.Tensor]:
    seqs = [item["sequence_features"] for item in batch]
    lengths = torch.tensor([seq.shape[0] for seq in seqs], dtype=torch.long)
    sequence_features = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    skill_type_id = torch.tensor([vocab["skill_type"].get(item["skill_type"], vocab["skill_type"].get("<unk>", 0)) for item in batch], dtype=torch.long)
    stage_id = torch.tensor([vocab["stage_name"].get(item["stage_name"], vocab["stage_name"].get("<unk>", 0)) for item in batch], dtype=torch.long)
    labels = torch.tensor(
        [[item["label_contact"], item["label_jam"], item["label_misgrasp"], item["label_slip"], item["label_recovery_needed"]] for item in batch],
        dtype=torch.float32,
    )
    return {
        "sequence_features": sequence_features,
        "lengths": lengths,
        "skill_type_id": skill_type_id,
        "stage_id": stage_id,
        "labels": labels,
    }


def _evaluate(model: ForceContactClassifierNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    steps = 0
    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence_features"].to(device)
            lengths = batch["lengths"].to(device)
            skill_type_id = batch["skill_type_id"].to(device)
            stage_id = batch["stage_id"].to(device)
            labels = batch["labels"].to(device)
            logits = model(seq, skill_type_id, stage_id, lengths=lengths)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            totals["loss"] += float(loss.item())
            for idx, name in enumerate(["contact", "jam", "misgrasp", "slip", "recovery_needed"]):
                totals[f"{name}_acc"] += float((pred[:, idx] == labels[:, idx]).float().mean().item())
                totals[f"{name}_pos_rate"] += float(probs[:, idx].mean().item())
            steps += 1
    if steps == 0:
        return {}
    return {k: v / steps for k, v in totals.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/force_classifier"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    _seed_everything(args.seed)
    dataset = ForceContactJsonlDataset(args.dataset)
    vocab = dataset.build_vocab()

    episodes = sorted({int(r["episode_idx"]) for r in dataset.records})
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * float(args.val_fraction))))
    val_episodes = set(episodes[:val_count])
    train_records = [r for r in dataset.records if int(r["episode_idx"]) not in val_episodes]
    val_records = [r for r in dataset.records if int(r["episode_idx"]) in val_episodes]
    train_ds = ForceContactJsonlDataset(args.dataset, records=train_records)
    val_ds = ForceContactJsonlDataset(args.dataset, records=val_records)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda batch: _collate(batch, vocab))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: _collate(batch, vocab))

    model = ForceContactClassifierNet.from_vocab(vocab).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    label_sums = np.zeros(5, dtype=np.float32)
    label_count = 0
    for rec in train_records:
        label_sums += np.array([rec["label_contact"], rec["label_jam"], rec["label_misgrasp"], rec["label_slip"], rec["label_recovery_needed"]], dtype=np.float32)
        label_count += 1
    pos_weight = torch.tensor(np.clip((label_count - label_sums) / np.maximum(label_sums, 1.0), 1.0, 50.0), dtype=torch.float32, device=args.device)

    best_val = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "dataset": str(args.dataset),
        "vocab": vocab,
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = defaultdict(float)
        steps = 0
        for batch in train_loader:
            seq = batch["sequence_features"].to(args.device)
            lengths = batch["lengths"].to(args.device)
            skill_type_id = batch["skill_type_id"].to(args.device)
            stage_id = batch["stage_id"].to(args.device)
            labels = batch["labels"].to(args.device)
            logits = model(seq, skill_type_id, stage_id, lengths=lengths)
            loss_per_label = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            weighted = loss_per_label * (1.0 + pos_weight * labels)
            loss = weighted.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["loss"] += float(loss.item())
            steps += 1

        train_metrics = {k: v / max(steps, 1) for k, v in running.items()}
        val_metrics = _evaluate(model, val_loader, torch.device(args.device))
        report["epochs"].append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        val_loss = val_metrics.get("loss", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = args.output_dir / "force_classifier_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "input_dim": 27,
                        "embed_dim": 16,
                        "hidden_dim": 96,
                    },
                    "metrics": val_metrics,
                },
                ckpt_path,
            )

    (args.output_dir / "force_classifier_train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output_dir / "force_classifier_best.pt")


if __name__ == "__main__":
    main()
