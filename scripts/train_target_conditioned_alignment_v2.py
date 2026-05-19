#!/usr/bin/env python3
"""Train Target-Conditioned Alignment v2 on near/micro rows.

Uses pairwise ranking loss: the oracle best_stage_action_index should rank
higher than all other valid candidates.
"""
from __future__ import annotations

import argparse, json, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prismatic.models.target_conditioned_alignment_policy import TargetConditionedAlignmentPolicy
from prismatic.vla.datasets.target_conditioned_alignment_v2_dataset import TargetConditionedAlignmentV2Dataset


def _pairwise_ranking_loss(scores: torch.Tensor, target_idx: torch.Tensor,
                           valid_mask: torch.Tensor | None = None, margin: float = 0.1) -> torch.Tensor:
    """Softplus pairwise ranking: target should outrank all others."""
    bsz, K = scores.shape
    target_score = scores.gather(1, target_idx.view(-1, 1))  # (B, 1)
    diff = scores - target_score  # (B, K); positive = worse than target
    loss = F.softplus(diff + margin)
    if valid_mask is not None:
        loss = loss * valid_mask
    # Don't penalize the target itself
    mask = torch.ones_like(loss, dtype=torch.bool)
    mask.scatter_(1, target_idx.view(-1, 1), False)
    loss = loss * mask.float()
    denom = mask.float().sum(dim=1).clamp(min=1)
    return loss.sum(dim=1) / denom


def _top1_accuracy(scores: torch.Tensor, target_idx: torch.Tensor) -> float:
    pred = scores.argmax(dim=-1)
    return float((pred == target_idx).float().mean().item())


def _top3_accuracy(scores: torch.Tensor, target_idx: torch.Tensor) -> float:
    _, top3 = scores.topk(min(3, scores.shape[1]), dim=-1)
    return float(top3.eq(target_idx.view(-1, 1)).any(dim=1).float().mean().item())


def _train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    top1s = []
    top3s = []
    for batch in loader:
        wd = batch["wrist_depth"].to(device)
        fh = batch["force_history"].to(device)
        pr = batch["proprio"].to(device)
        ba = batch["planner_base_action_local"].to(device)
        td = batch["current_to_target_delta_local"].to(device)
        pa = batch["proposal_actions"].to(device)
        pd = batch["post_candidate_delta"].to(device)
        xi = batch["xy_improvement"].to(device)
        zi = batch["z_improvement"].to(device)
        yi = batch["yaw_improvement"].to(device)
        gi = batch["geometry_improvement"].to(device)
        target = batch["best_stage_action_index"].to(device)
        valid = batch["candidate_valid_mask"].to(device)

        out = model(wd, fh, pr, ba, td, pa, pd, xi, zi, yi, gi)
        scores = out["candidate_scores"]
        loss = _pairwise_ranking_loss(scores, target, valid).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        top1s.append(_top1_accuracy(scores, target))
        top3s.append(_top3_accuracy(scores, target))

    return {
        "loss": float(np.mean(losses)),
        "top1_accuracy": float(np.mean(top1s)),
        "top3_accuracy": float(np.mean(top3s)),
    }


@torch.no_grad()
def _eval_epoch(model, loader, device):
    model.eval()
    losses = []
    top1s = []
    top3s = []
    per_bucket = defaultdict(lambda: {"loss": [], "top1": [], "n": 0})

    for batch in loader:
        wd = batch["wrist_depth"].to(device)
        fh = batch["force_history"].to(device)
        pr = batch["proprio"].to(device)
        ba = batch["planner_base_action_local"].to(device)
        td = batch["current_to_target_delta_local"].to(device)
        pa = batch["proposal_actions"].to(device)
        pd = batch["post_candidate_delta"].to(device)
        xi = batch["xy_improvement"].to(device)
        zi = batch["z_improvement"].to(device)
        yi = batch["yaw_improvement"].to(device)
        gi = batch["geometry_improvement"].to(device)
        target = batch["best_stage_action_index"].to(device)
        valid = batch["candidate_valid_mask"].to(device)
        buckets = batch["stage_bucket"]

        out = model(wd, fh, pr, ba, td, pa, pd, xi, zi, yi, gi)
        scores = out["candidate_scores"]
        loss = _pairwise_ranking_loss(scores, target, valid).mean()

        losses.append(loss.item())
        top1s.append(_top1_accuracy(scores, target))
        top3s.append(_top3_accuracy(scores, target))

        for j in range(len(buckets)):
            b = str(buckets[j])
            per_bucket[b]["loss"].append(loss.item())
            per_bucket[b]["top1"].append(float((scores[j].argmax() == target[j]).item()))
            per_bucket[b]["n"] += 1

    bucket_stats = {}
    for b, v in per_bucket.items():
        bucket_stats[b] = {
            "n": v["n"],
            "loss": round(float(np.mean(v["loss"])), 6),
            "top1_accuracy": round(float(np.mean(v["top1"])), 4),
        }

    return {
        "loss": float(np.mean(losses)),
        "top1_accuracy": float(np.mean(top1s)),
        "top3_accuracy": float(np.mean(top3s)),
        "per_bucket": bucket_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Train v2 target-conditioned alignment")
    parser.add_argument("--dataset_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--overfit_sanity", action="store_true", default=False,
                        help="Train+eval on same small subset to check model capacity")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_buckets = ["near_alignment", "micro_contact_refine"]
    print(f"[train] loading dataset from {args.dataset_npz}, filter={train_buckets}")

    full_ds = TargetConditionedAlignmentV2Dataset(
        str(args.dataset_npz), stage_bucket_filter=train_buckets
    )
    print(f"[train] near+micro rows: {len(full_ds)}")

    if args.overfit_sanity:
        n_small = min(64, len(full_ds))
        indices = torch.randperm(len(full_ds))[:n_small].tolist()
        train_ds = torch.utils.data.Subset(full_ds, indices)
        val_ds = torch.utils.data.Subset(full_ds, indices)
        print(f"[train] OVERFIT SANITY: train={len(train_ds)} val={len(val_ds)} (same subset)")
    else:
        n_val = max(1, int(len(full_ds) * 0.15))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
        print(f"[train] train={n_train} val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = TargetConditionedAlignmentPolicy(proposal_count=8).to(device)
    print(f"[train] model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []

    best_top1 = 0.0
    for epoch in range(args.epochs):
        train_stats = _train_epoch(model, train_loader, optimizer, device)
        val_stats = _eval_epoch(model, val_loader, device)

        record = {
            "epoch": epoch + 1,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(record)

        print(f"  epoch {epoch+1:3d}: train_loss={train_stats['loss']:.4f} train_top1={train_stats['top1_accuracy']:.3f} "
              f"val_loss={val_stats['loss']:.4f} val_top1={val_stats['top1_accuracy']:.3f} "
              f"val_top3={val_stats['top3_accuracy']:.3f}")

        if val_stats["top1_accuracy"] > best_top1:
            best_top1 = val_stats["top1_accuracy"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_top1": best_top1},
                output_dir / "target_conditioned_alignment_v2_best.pt",
            )

    # Final eval on all buckets (full dataset, no filter)
    print(f"\n[train] final eval on ALL buckets...")
    all_ds = TargetConditionedAlignmentV2Dataset(str(args.dataset_npz), stage_bucket_filter=None)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False)
    final_stats = _eval_epoch(model, all_loader, device)
    print(f"  all_buckets: loss={final_stats['loss']:.4f} top1={final_stats['top1_accuracy']:.3f} top3={final_stats['top3_accuracy']:.3f}")
    for b, s in final_stats["per_bucket"].items():
        print(f"    {b}: n={s['n']} loss={s['loss']:.4f} top1={s.get('top1_accuracy', s.get('top1', 0)):.3f}")

    # Save final checkpoint and history
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": args.epochs, "val_top1": best_top1},
        output_dir / "target_conditioned_alignment_v2_final.pt",
    )
    with open(output_dir / "train_history.json", "w") as f:
        json.dump({"history": history, "final_eval": final_stats}, f, indent=2, ensure_ascii=False)

    print(f"[train] done. best_val_top1={best_top1:.4f}")
    print(f"[train] outputs in {output_dir}")


if __name__ == "__main__":
    main()
