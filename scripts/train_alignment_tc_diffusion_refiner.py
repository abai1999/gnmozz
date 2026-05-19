#!/usr/bin/env python3
"""Train target-conditioned diffusion alignment.

Privileged labels are used only as losses.  Runtime inputs remain observable
RGB-D/force/proprio/planner context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from prismatic.models.alignment_tc_diffusion_refiner import TargetConditionedAlignmentDiffusionRefiner
from prismatic.vla.datasets.alignment_tc_diffusion_dataset import AlignmentTCDiffusionDataset


def parse_args():
    ap = argparse.ArgumentParser(description="Train target-conditioned diffusion alignment")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_fraction", type=float, default=0.10)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max_pos_step", type=float, default=0.0015)
    ap.add_argument("--max_yaw_step", type=float, default=0.0060)
    ap.add_argument("--delta_weight", type=float, default=1.0)
    ap.add_argument("--heatmap_weight", type=float, default=0.25)
    ap.add_argument("--confidence_weight", type=float, default=0.35)
    ap.add_argument("--trajectory_weight", type=float, default=1.0)
    ap.add_argument("--progress_weight", type=float, default=0.4)
    ap.add_argument("--risk_weight", type=float, default=0.5)
    ap.add_argument("--stop_weight", type=float, default=0.25)
    ap.add_argument("--smooth_weight", type=float, default=0.05)
    ap.add_argument("--delta_xy_weight", type=float, default=1.0)
    ap.add_argument("--delta_z_weight", type=float, default=1.0)
    ap.add_argument("--delta_yaw_weight", type=float, default=1.0)
    ap.add_argument("--delta_rollpitch_weight", type=float, default=0.5)
    ap.add_argument("--trajectory_xy_weight", type=float, default=1.0)
    ap.add_argument("--trajectory_z_weight", type=float, default=1.0)
    ap.add_argument("--trajectory_yaw_weight", type=float, default=1.0)
    ap.add_argument("--progress_xy_weight", type=float, default=1.0)
    ap.add_argument("--progress_z_weight", type=float, default=1.0)
    ap.add_argument("--progress_yaw_weight", type=float, default=1.0)
    ap.add_argument("--teacher_force_warmup_fraction", type=float, default=0.30)
    ap.add_argument("--teacher_force_decay_fraction", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def batch_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def _axis_weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    axis_weights: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    while axis_weights.ndim < loss.ndim:
        axis_weights = axis_weights.unsqueeze(0)
    return (loss * axis_weights).mean(dim=tuple(range(1, loss.ndim)))


def teacher_force_prob(epoch: int, epochs: int, args) -> float:
    warmup = max(1, int(float(args.teacher_force_warmup_fraction) * epochs))
    decay = max(1, int(float(args.teacher_force_decay_fraction) * epochs))
    if epoch <= warmup:
        return 1.0
    t = min(max(epoch - warmup, 0), decay)
    return float(max(0.0, 1.0 - t / decay))


def compute_loss(model, batch, args, tf_prob: float):
    weight = batch["sample_weight"].reshape(-1).float()
    delta_axis_weights = torch.tensor(
        [
            float(args.delta_xy_weight),
            float(args.delta_xy_weight),
            float(args.delta_z_weight),
            float(args.delta_rollpitch_weight),
            float(args.delta_rollpitch_weight),
            float(args.delta_yaw_weight),
        ],
        device=batch["teacher_target_delta_local_6d"].device,
        dtype=batch["teacher_target_delta_local_6d"].dtype,
    )
    traj_axis_weights = torch.tensor(
        [
            float(args.trajectory_xy_weight),
            float(args.trajectory_xy_weight),
            float(args.trajectory_z_weight),
            float(args.trajectory_yaw_weight),
        ],
        device=batch["best_residual_trajectory_4d"].device,
        dtype=batch["best_residual_trajectory_4d"].dtype,
    )
    progress_axis_weights = torch.tensor(
        [
            float(args.progress_xy_weight),
            float(args.progress_z_weight),
            float(args.progress_yaw_weight),
        ],
        device=batch["progress_label"].device,
        dtype=batch["progress_label"].dtype,
    )
    out = model(
        wrist_depth=batch["wrist_depth"],
        force_history=batch["force_history"],
        proprio=batch["proprio"],
        planner_action_local=batch["planner_action_local"],
        gripper_context=batch["gripper_context"],
        front_rgb=batch["front_rgb"],
        wrist_rgb=batch["wrist_rgb"],
        teacher_target_delta_local=batch["teacher_target_delta_local_6d"],
        teacher_force_prob=tf_prob,
    )
    pred_delta = out["pred_target_delta_local_6d"]
    delta_loss = _axis_weighted_smooth_l1(pred_delta, batch["teacher_target_delta_local_6d"], delta_axis_weights)
    delta_loss = weighted_mean(delta_loss, weight)

    heatmap_target = batch["contact_heatmap_label"]
    if heatmap_target.ndim == 3:
        heatmap_target = heatmap_target.unsqueeze(1)
    heatmap_loss = F.binary_cross_entropy_with_logits(out["pred_contact_heatmap_logits"], heatmap_target, reduction="none").mean(dim=(1, 2, 3))
    heatmap_loss = weighted_mean(heatmap_loss, weight)

    conf_loss = F.binary_cross_entropy_with_logits(out["target_confidence_logit"], batch["target_confidence_label"].reshape(-1), reduction="none")
    conf_loss = weighted_mean(conf_loss, weight)

    pred_traj = out["trajectory_4d"]
    target_traj = batch["best_residual_trajectory_4d"][:, : pred_traj.shape[1], : pred_traj.shape[2]]
    traj_loss = F.smooth_l1_loss(pred_traj, target_traj, reduction="none")
    while traj_axis_weights.ndim < traj_loss.ndim:
        traj_axis_weights = traj_axis_weights.unsqueeze(0)
    traj_loss = (traj_loss * traj_axis_weights).mean(dim=(1, 2))
    traj_loss = weighted_mean(traj_loss, weight)

    progress_loss = F.binary_cross_entropy_with_logits(out["progress_logits"], batch["progress_label"], reduction="none")
    while progress_axis_weights.ndim < progress_loss.ndim:
        progress_axis_weights = progress_axis_weights.unsqueeze(0)
    progress_loss = (progress_loss * progress_axis_weights).mean(dim=1)
    progress_loss = weighted_mean(progress_loss, weight)
    risk_loss = F.binary_cross_entropy_with_logits(out["risk_logit"], batch["risk_label"].reshape(-1), reduction="none")
    risk_loss = weighted_mean(risk_loss, weight)
    stop_loss = F.binary_cross_entropy_with_logits(out["stop_logit"], batch["stop_label"].reshape(-1), reduction="none")
    stop_loss = weighted_mean(stop_loss, weight)
    smooth_loss = torch.zeros((), device=pred_traj.device)
    if pred_traj.shape[1] > 1:
        smooth_loss = (pred_traj[:, 1:] - pred_traj[:, :-1]).square().mean(dim=(1, 2))
        smooth_loss = weighted_mean(smooth_loss, weight)

    total = (
        args.delta_weight * delta_loss
        + args.heatmap_weight * heatmap_loss
        + args.confidence_weight * conf_loss
        + args.trajectory_weight * traj_loss
        + args.progress_weight * progress_loss
        + args.risk_weight * risk_loss
        + args.stop_weight * stop_loss
        + args.smooth_weight * smooth_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "delta_loss": float(delta_loss.detach().cpu()),
        "heatmap_loss": float(heatmap_loss.detach().cpu()),
        "confidence_loss": float(conf_loss.detach().cpu()),
        "trajectory_loss": float(traj_loss.detach().cpu()),
        "progress_loss": float(progress_loss.detach().cpu()),
        "risk_loss": float(risk_loss.detach().cpu()),
        "stop_loss": float(stop_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "teacher_force_prob": float(tf_prob),
    }


def run_epoch(model, loader, args, device, epoch: int, epochs: int, optimizer=None):
    train = optimizer is not None
    model.train(train)
    sums = {}
    count = 0
    tf_prob = teacher_force_prob(epoch, epochs, args) if train else 0.0
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            loss, metrics = compute_loss(model, batch, args, tf_prob)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        bsz = int(batch["proprio"].shape[0])
        count += bsz
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * bsz
    return {k: v / max(count, 1) for k, v in sums.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = AlignmentTCDiffusionDataset(args.dataset)
    val_len = max(1, int(len(dataset) * args.val_fraction))
    train_len = max(1, len(dataset) - val_len)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetConditionedAlignmentDiffusionRefiner(
        horizon=args.horizon,
        max_pos_step=args.max_pos_step,
        max_yaw_step=args.max_yaw_step,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, device, epoch, args.epochs, optimizer)
        val_metrics = run_epoch(model, val_loader, args, device, epoch, args.epochs, None)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "controller_type": "alignment_tc_diffusion_refiner",
                    "horizon": args.horizon,
                    "max_pos_step": args.max_pos_step,
                    "max_yaw_step": args.max_yaw_step,
                    "dataset": str(args.dataset),
                    "best_val_loss": best_val,
                },
                out_dir / "alignment_tc_diffusion_refiner_best.pt",
            )
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
