#!/usr/bin/env python3
"""Train the non-privileged alignment diffusion refiner.

This is the first training entry for the new alignment mainline.  It expects a
contract-matched self-supervised NPZ built from near/contact planner rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from prismatic.models.alignment_diffusion_refiner import AlignmentDiffusionRefiner
from prismatic.vla.datasets.alignment_diffusion_dataset import AlignmentDiffusionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlignmentDiffusionRefiner")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max_pos_step", type=float, default=0.0015)
    parser.add_argument("--max_yaw_step", type=float, default=0.0060)
    parser.add_argument("--progress_weight", type=float, default=0.5)
    parser.add_argument("--risk_weight", type=float, default=0.5)
    parser.add_argument("--stop_weight", type=float, default=0.2)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def batch_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def compute_loss(model, batch, args):
    sample_weight = batch.get("sample_weight", None)
    if sample_weight is None:
        sample_weight = torch.ones((batch["proprio"].shape[0],), device=batch["proprio"].device, dtype=torch.float32)
    else:
        sample_weight = sample_weight.to(batch["proprio"].device, dtype=torch.float32).reshape(-1)
    out = model(
        wrist_depth=batch["wrist_depth"],
        force_history=batch["force_history"],
        proprio=batch["proprio"],
        planner_action_local=batch["planner_action_local"],
        gripper_context=batch["gripper_context"],
        front_rgb=batch["front_rgb"],
        wrist_rgb=batch["wrist_rgb"],
    )
    pred = out["trajectory_4d"]
    target = batch["residual_trajectory_4d"][:, : pred.shape[1], : pred.shape[2]]
    bc_loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=(1, 2))
    bc_loss = (bc_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    progress_loss = F.binary_cross_entropy_with_logits(out["progress_logits"], batch["progress_label"], reduction="none")
    progress_loss = progress_loss.mean(dim=1)
    progress_loss = (progress_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    risk_target = batch["risk_label"].reshape(-1)
    risk_loss = F.binary_cross_entropy_with_logits(out["risk_logit"], risk_target, reduction="none")
    risk_loss = (risk_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    stop_target = batch["stop_label"].reshape(-1)
    stop_loss = F.binary_cross_entropy_with_logits(out["stop_logit"], stop_target, reduction="none")
    stop_loss = (stop_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    smooth_loss = torch.zeros((), device=pred.device)
    if pred.shape[1] > 1:
        smooth_loss = (pred[:, 1:] - pred[:, :-1]).square().mean(dim=(1, 2))
        smooth_loss = (smooth_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    total = (
        bc_loss
        + args.progress_weight * progress_loss
        + args.risk_weight * risk_loss
        + args.stop_weight * stop_loss
        + args.smooth_weight * smooth_loss
    )
    return total, {
        "loss": float(total.detach().cpu().item()),
        "bc_loss": float(bc_loss.detach().cpu().item()),
        "progress_loss": float(progress_loss.detach().cpu().item()),
        "risk_loss": float(risk_loss.detach().cpu().item()),
        "stop_loss": float(stop_loss.detach().cpu().item()),
        "smooth_loss": float(smooth_loss.detach().cpu().item()),
    }


def run_epoch(model, loader, args, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    sums = {}
    count = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            loss, metrics = compute_loss(model, batch, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        bsz = int(batch["proprio"].shape[0])
        count += bsz
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * bsz
    return {key: value / max(count, 1) for key, value in sums.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = AlignmentDiffusionDataset(args.dataset)
    val_len = max(1, int(len(dataset) * args.val_fraction))
    train_len = max(1, len(dataset) - val_len)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlignmentDiffusionRefiner(
        horizon=args.horizon,
        max_pos_step=args.max_pos_step,
        max_yaw_step=args.max_yaw_step,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, device, optimizer)
        val_metrics = run_epoch(model, val_loader, args, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "horizon": args.horizon,
                    "max_pos_step": args.max_pos_step,
                    "max_yaw_step": args.max_yaw_step,
                    "dataset": str(args.dataset),
                    "best_val_loss": best_val,
                    "controller_type": "alignment_diffusion_refiner",
                },
                out_dir / "alignment_diffusion_refiner_best.pt",
            )
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
