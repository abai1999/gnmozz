"""
train_residual.py

Train the ResidualController (Route 1 MVP) on pre-collected residual data.

Loss:
  L = lambda_delta * L_delta + lambda_zero * L_zero + lambda_alpha_zero * L_alpha_zero
      + lambda_ready * L_ready + lambda_gripper * L_gripper + lambda_hold * L_hold
  L_delta = SmoothL1(raw_pred, target) by default for v2 alignment training
  L_zero  = ||pred||_1 on non-close-intent / far samples only
  L_alpha_zero = alpha on non-close-intent / far samples only (optional legacy regularizer)
  L_ready/L_gripper/L_hold supervise planner-conditioned readiness and hold behavior

Usage:
    python scripts/train_residual.py \
        --data_dir data/residual_data/insert_onto_square_peg \
        --output_dir outputs/residual_train/insert_v1 \
        --max_steps 50000 \
        --batch_size 64 \
        --lr 1e-3
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("VLA_PLATFORM", "RLBENCH")

from prismatic.models.residual_controller import ResidualController
from prismatic.vla.datasets.residual_rlbench_dataset import ResidualRLBenchDataset


def set_training_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine annealing schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[train_residual] Device: {device}")
    if args.seed is not None:
        set_training_seed(args.seed)
        print(f"[train_residual] Using fixed seed: {args.seed}")

    # ── Dataset ──
    dataset = ResidualRLBenchDataset(
        data_dir=args.data_dir,
        oversample_contact=args.oversample_contact,
        oversample_pre_contact=args.oversample_pre_contact,
        oversample_jam=args.oversample_jam,
        stage_role_filter=args.stage_role if args.stage_role != "all" else None,
    )
    summary = dataset.get_summary()
    print(f"[train_residual] Dataset summary: {json.dumps(summary, indent=2)}")
    if (
        summary.get("dataset_view") == "basin_pose_view"
        and args.stage_role == "align"
        and not args.keep_zero_regularizer_for_basin_pose
    ):
        if args.lambda_zero != 0.0 or args.lambda_alpha_zero != 0.0:
            print("[train_residual] basin_pose_view detected: forcing lambda_zero=lambda_alpha_zero=0.0")
        args.lambda_zero = 0.0
        args.lambda_alpha_zero = 0.0
    ready_counts = summary.get("basin_positive_counts", summary.get("readiness_counts", {}))
    hold_counts = summary.get("hold_counts", {})
    ready_pos = int(ready_counts.get("1", ready_counts.get(1, 0)))
    ready_neg = int(ready_counts.get("0", ready_counts.get(0, 0)))
    hold_pos = int(hold_counts.get("1", hold_counts.get(1, 0)))
    hold_neg = int(hold_counts.get("0", hold_counts.get(0, 0)))
    ready_pos_weight = 1.0
    hold_pos_weight = 1.0
    if ready_pos > 0 and ready_neg > 0:
        ready_pos_weight = min(args.max_ready_pos_weight, max(1.0, ready_neg / max(ready_pos, 1)))
    if hold_pos > 0 and hold_neg > 0:
        hold_pos_weight = min(args.max_hold_pos_weight, max(1.0, hold_neg / max(hold_pos, 1)))
    print(
        f"[train_residual] class weights: ready_pos_weight={ready_pos_weight:.3f} "
        f"hold_pos_weight={hold_pos_weight:.3f}"
    )

    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    def _worker_init_fn(worker_id):
        if args.seed is None:
            return
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=_worker_init_fn,
    )

    # ── Model ──
    model = ResidualController(
        pose_output_mode=args.pose_supervision,
        pose_use_depth=not args.pose_no_depth,
        pose_use_force=not args.pose_no_force,
        pose_use_proprio=not args.pose_no_proprio,
        pose_use_action=not args.pose_no_action,
        fire_only_head=args.trigger_fire_only,
        ready_use_context=not args.trigger_no_context,
        ready_use_gripper_context=not args.trigger_no_gripper_context,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train_residual] Model parameters: {total_params:,}")

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    # ── Loss ──
    if args.stage_role == "contact":
        dim_weights = torch.tensor([1.5, 1.5, 2.0, 0.25, 0.25, 2.75], device=device)
        phase_weights = torch.tensor([0.90, 1.05, 1.30, 1.55], device=device)
    else:
        dim_weights = torch.tensor([4.0, 4.0, 1.5, 0.0, 0.0, 4.5], device=device)
        phase_weights = torch.tensor([0.75, 1.35, 1.05, 1.20], device=device)
    dim_weights = dim_weights / dim_weights.sum()  # normalize

    bce = nn.BCELoss(reduction="none")
    cross_entropy = nn.CrossEntropyLoss(reduction="none")

    # ── Output ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Training loop ──
    model.train()
    data_iter = iter(dataloader)
    step = 0
    log_interval = 100
    save_interval = args.save_freq

    running_loss = 0.0
    running_delta_loss = 0.0
    running_zero_loss = 0.0
    running_alpha_zero_loss = 0.0
    running_contact_loss = 0.0
    running_free_loss = 0.0
    running_ready_loss = 0.0
    running_gripper_loss = 0.0
    running_hold_loss = 0.0
    n_contact_steps = 0
    n_free_steps = 0

    t_start = time.time()

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        wrist_depth = batch["wrist_depth"].to(device)      # (B, 1, 96, 96)
        ft_hist = batch["ft_hist"].to(device)              # (B, 32, 6)
        proprio = batch["proprio"].to(device)              # (B, 15)
        base_action = batch["base_action"].to(device)      # (B, 6)
        gripper_context = batch.get("gripper_context", None)
        if gripper_context is not None:
            gripper_context = gripper_context.to(device)   # (B, 3)
        step_idx = batch["step_idx"].to(device)            # (B,)
        delta_target = batch.get("delta_basin_target", batch.get("delta_align_target", batch["delta_target"])).to(device)    # (B, 6)
        contact_mask = batch["contact_mask"].to(device)    # (B,)
        phase_id = batch["phase_id"].to(device)
        phase_age = batch["phase_age"].to(device)
        steps_since_last_replan = batch["steps_since_last_replan"].to(device)
        planner_close_intent = batch.get("planner_close_intent", None)
        if planner_close_intent is not None:
            planner_close_intent = planner_close_intent.to(device)
        ready_target = batch.get("basin_positive", batch.get("readiness_label", batch.get("ready_to_close", None)))
        if ready_target is not None:
            ready_target = ready_target.to(device)
        hold_target = batch.get("hold_label", None)
        if hold_target is not None:
            hold_target = hold_target.to(device)
        gripper_target = batch.get("gripper_state_target", None)
        if gripper_target is not None:
            gripper_target = gripper_target.to(device)
        negative_reason = batch.get("negative_reason", None)
        if negative_reason is not None:
            negative_reason = negative_reason.to(device)

        # Forward
        outputs = model(
            wrist_depth,
            ft_hist,
            proprio,
            base_action,
            step_idx,
            phase_id=phase_id,
            phase_age=phase_age,
            steps_since_last_replan=steps_since_last_replan,
            gripper_context=gripper_context,
            return_aux=True,
        )
        raw_delta_pred = outputs["delta_pose"]
        delta_pred = outputs["delta_pose"] if args.pose_supervision == "raw" else outputs["delta_pose_gated"]
        alpha = outputs["alpha"]
        ready_pred = outputs.get("ready_to_close", None)
        hold_pred = outputs.get("hold_after_close", None)
        gripper_logits = outputs.get("gripper_logits", None)

        # L_delta: weighted SmoothL1 focused on close-intent alignment states
        per_dim_loss = F.smooth_l1_loss(
            delta_pred,
            delta_target,
            reduction="none",
            beta=args.delta_smooth_l1_beta,
        )  # (B, 6)
        weighted_loss = per_dim_loss * dim_weights.unsqueeze(0)  # (B, 6)
        sample_weights = phase_weights[torch.clamp(phase_id, 0, phase_weights.numel() - 1)]
        pose_mask = torch.ones_like(sample_weights)
        if hold_target is not None:
            pose_mask = pose_mask * (hold_target < 0.5).float()
        pose_weights = torch.ones_like(sample_weights)
        if planner_close_intent is not None:
            pose_weights = pose_weights + (planner_close_intent > 0.5).float() * args.pose_intent_weight
        if ready_target is not None:
            pose_weights = pose_weights + (ready_target > 0.5).float() * args.pose_basin_positive_weight
        if negative_reason is not None:
            pose_weights = pose_weights + (negative_reason == 0).float() * args.pose_far_from_basin_weight
            pose_weights = pose_weights + (negative_reason == 1).float() * args.pose_in_basin_unstable_weight
            pose_weights = pose_weights + (negative_reason == 3).float() * args.pose_no_progress_weight
            pose_weights = pose_weights + (negative_reason == 4).float() * args.pose_invalid_weight
        pose_target_4d = torch.stack(
            [delta_target[:, 0], delta_target[:, 1], delta_target[:, 2], delta_target[:, 5]],
            dim=-1,
        )
        pose_target_norm = torch.linalg.norm(pose_target_4d, dim=-1)
        pose_norm_gain = torch.clamp(pose_target_norm / max(args.pose_target_norm_ref, 1e-6), min=1.0)
        pose_norm_gain = torch.clamp(pose_norm_gain, max=args.pose_target_norm_max_gain)
        pose_weights = pose_weights * (1.0 + args.pose_target_norm_weight * (pose_norm_gain - 1.0))
        pose_sample_weight = sample_weights * pose_weights * pose_mask
        denom = pose_sample_weight.sum()
        if denom.item() > 0:
            L_delta = (weighted_loss.mean(dim=-1) * pose_sample_weight).sum() / denom
        else:
            L_delta = torch.tensor(0.0, device=device)

        # L_zero: L1 norm of predictions on free-space samples
        if planner_close_intent is not None:
            free_mask = (planner_close_intent < 0.5).float()
        else:
            free_mask = (contact_mask == 0).float()  # (B,)
        n_free = free_mask.sum().item()
        if n_free > 0:
            L_zero = (delta_pred.abs().mean(dim=-1) * free_mask).sum() / max(n_free, 1)
            L_alpha_zero = (alpha * free_mask).sum() / max(n_free, 1)
        else:
            L_zero = torch.tensor(0.0, device=device)
            L_alpha_zero = torch.tensor(0.0, device=device)

        if ready_target is not None and ready_pred is not None:
            ready_mask = ready_target >= 0
            if ready_mask.any():
                ready_losses = bce(ready_pred[ready_mask], ready_target[ready_mask].float())
                ready_weights = torch.ones_like(ready_losses)
                ready_weights = torch.where(
                    ready_target[ready_mask] > 0.5,
                    ready_weights * ready_pos_weight,
                    ready_weights,
                )
                L_ready = (ready_losses * ready_weights).mean()
            else:
                L_ready = torch.tensor(0.0, device=device)
        else:
            L_ready = torch.tensor(0.0, device=device)

        if (not args.trigger_fire_only) and hold_target is not None and hold_pred is not None:
            hold_mask = hold_target >= 0
            if hold_mask.any():
                hold_losses = bce(hold_pred[hold_mask], hold_target[hold_mask].float())
                hold_weights = torch.ones_like(hold_losses)
                hold_weights = torch.where(
                    hold_target[hold_mask] > 0.5,
                    hold_weights * hold_pos_weight,
                    hold_weights,
                )
                L_hold = (hold_losses * hold_weights).mean()
            else:
                L_hold = torch.tensor(0.0, device=device)
        else:
            L_hold = torch.tensor(0.0, device=device)

        if (not args.trigger_fire_only) and gripper_target is not None and gripper_logits is not None:
            gripper_mask = gripper_target >= 0
            if gripper_mask.any():
                L_gripper = cross_entropy(gripper_logits[gripper_mask], gripper_target[gripper_mask]).mean()
            else:
                L_gripper = torch.tensor(0.0, device=device)
        else:
            L_gripper = torch.tensor(0.0, device=device)

        # Total loss
        loss = (
            args.lambda_delta * L_delta
            + args.lambda_zero * L_zero
            + args.lambda_alpha_zero * L_alpha_zero
            + args.lambda_ready * L_ready
            + args.lambda_gripper * L_gripper
            + args.lambda_hold * L_hold
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── Logging ──
        running_loss += loss.item()
        running_delta_loss += L_delta.item()
        running_zero_loss += L_zero.item()
        running_alpha_zero_loss += L_alpha_zero.item()
        running_ready_loss += L_ready.item()
        running_gripper_loss += L_gripper.item()
        running_hold_loss += L_hold.item()

        # Track contact vs free loss separately
        contact_mask_bool = contact_mask > 0
        if contact_mask_bool.any():
            contact_loss_val = weighted_loss[contact_mask_bool].mean().item()
            running_contact_loss += contact_loss_val
            n_contact_steps += 1
        if (~contact_mask_bool).any():
            free_loss_val = weighted_loss[~contact_mask_bool].mean().item()
            running_free_loss += free_loss_val
            n_free_steps += 1

        step += 1

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_delta = running_delta_loss / log_interval
            avg_zero = running_zero_loss / log_interval
            avg_alpha_zero = running_alpha_zero_loss / log_interval
            avg_ready = running_ready_loss / log_interval
            avg_gripper = running_gripper_loss / log_interval
            avg_hold = running_hold_loss / log_interval
            avg_contact = running_contact_loss / max(n_contact_steps, 1)
            avg_free = running_free_loss / max(n_free_steps, 1)
            pred_mag = delta_pred.abs().mean().item()
            raw_pred_mag = raw_delta_pred.abs().mean().item()
            alpha_mean = alpha.mean().item()
            ready_mean = ready_pred.mean().item() if ready_pred is not None else 0.0
            hold_mean = hold_pred.mean().item() if hold_pred is not None else 0.0
            ready_acc = 0.0
            if ready_target is not None and ready_pred is not None:
                ready_mask = ready_target >= 0
                if ready_mask.any():
                    ready_acc = (
                        ((ready_pred[ready_mask] >= 0.5).long() == ready_target[ready_mask].long())
                        .float()
                        .mean()
                        .item()
                    )
            gripper_pred_hist = []
            gripper_confusion = []
            if gripper_logits is not None:
                gripper_pred = torch.argmax(gripper_logits.detach(), dim=-1)
                gripper_pred_hist = [
                    int((gripper_pred == cls).sum().item()) for cls in range(gripper_logits.shape[-1])
                ]
                if gripper_target is not None:
                    gripper_mask = gripper_target >= 0
                    if gripper_mask.any():
                        n_cls = gripper_logits.shape[-1]
                        confusion = torch.zeros(n_cls, n_cls, device=device, dtype=torch.long)
                        for t_cls in range(n_cls):
                            for p_cls in range(n_cls):
                                confusion[t_cls, p_cls] = (
                                    (gripper_target[gripper_mask] == t_cls)
                                    & (gripper_pred[gripper_mask] == p_cls)
                                ).sum()
                        gripper_confusion = confusion.cpu().tolist()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            its = step / elapsed

            print(
                f"[step {step:6d}/{args.max_steps}] "
                f"loss={avg_loss:.5f}  delta={avg_delta:.5f}  zero={avg_zero:.5f}  alpha0={avg_alpha_zero:.5f}  "
                f"ready={avg_ready:.5f}  grip={avg_gripper:.5f}  hold={avg_hold:.5f}  contact_L={avg_contact:.5f}  free_L={avg_free:.5f}  "
                f"|pred|={pred_mag:.6f}  |raw|={raw_pred_mag:.6f}  alpha={alpha_mean:.4f}  basin_p={ready_mean:.4f}  hold_p={hold_mean:.4f}  "
                f"ready_acc={ready_acc:.3f}  grip_hist={gripper_pred_hist}  "
                f"grip_conf={gripper_confusion}  lr={lr:.2e}  it/s={its:.1f}"
            )

            running_loss = 0.0
            running_delta_loss = 0.0
            running_zero_loss = 0.0
            running_alpha_zero_loss = 0.0
            running_ready_loss = 0.0
            running_gripper_loss = 0.0
            running_hold_loss = 0.0
            running_contact_loss = 0.0
            running_free_loss = 0.0
            n_contact_steps = 0
            n_free_steps = 0

        if step % save_interval == 0:
            ckpt_path = output_dir / f"residual_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "pose_output_mode": args.pose_supervision,
                "pose_use_depth": not args.pose_no_depth,
                "pose_use_force": not args.pose_no_force,
                "pose_use_proprio": not args.pose_no_proprio,
                "pose_use_action": not args.pose_no_action,
                "fire_only_head": args.trigger_fire_only,
                "ready_use_context": not args.trigger_no_context,
                "ready_use_gripper_context": not args.trigger_no_gripper_context,
                "residual_version": 2,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save ──
    final_path = output_dir / "residual_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "pose_output_mode": args.pose_supervision,
        "pose_use_depth": not args.pose_no_depth,
        "pose_use_force": not args.pose_no_force,
        "pose_use_proprio": not args.pose_no_proprio,
        "pose_use_action": not args.pose_no_action,
        "fire_only_head": args.trigger_fire_only,
        "ready_use_context": not args.trigger_no_context,
        "ready_use_gripper_context": not args.trigger_no_gripper_context,
        "residual_version": 2,
    }, final_path)
    print(f"\n[train_residual] Training complete. Final model: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train residual controller")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lambda_delta", type=float, default=1.0)
    parser.add_argument("--lambda_zero", type=float, default=0.1)
    parser.add_argument("--lambda_alpha_zero", type=float, default=0.05)
    parser.add_argument("--lambda_ready", type=float, default=0.2)
    parser.add_argument("--lambda_gripper", type=float, default=0.5)
    parser.add_argument("--lambda_hold", type=float, default=0.1)
    parser.add_argument("--max_ready_pos_weight", type=float, default=8.0)
    parser.add_argument("--max_hold_pos_weight", type=float, default=12.0)
    parser.add_argument("--pose_intent_weight", type=float, default=1.0)
    parser.add_argument("--pose_basin_positive_weight", type=float, default=2.0)
    parser.add_argument("--pose_far_from_basin_weight", type=float, default=3.0)
    parser.add_argument("--pose_in_basin_unstable_weight", type=float, default=2.0)
    parser.add_argument("--pose_no_progress_weight", type=float, default=2.5)
    parser.add_argument("--pose_invalid_weight", type=float, default=3.0)
    parser.add_argument("--delta_smooth_l1_beta", type=float, default=0.01)
    parser.add_argument("--pose_target_norm_weight", type=float, default=2.0)
    parser.add_argument("--pose_target_norm_ref", type=float, default=0.005)
    parser.add_argument("--pose_target_norm_max_gain", type=float, default=4.0)
    parser.add_argument("--oversample_contact", type=int, default=5)
    parser.add_argument("--oversample_pre_contact", type=int, default=3)
    parser.add_argument("--oversample_jam", type=int, default=7)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage_role", type=str, default="all", choices=["all", "align", "contact"])
    parser.add_argument("--pose_supervision", type=str, default="raw", choices=["raw", "gated"])
    parser.add_argument("--keep_zero_regularizer_for_basin_pose", action="store_true", default=False)
    parser.add_argument("--pose_no_depth", action="store_true", default=False)
    parser.add_argument("--pose_no_force", action="store_true", default=False)
    parser.add_argument("--pose_no_proprio", action="store_true", default=False)
    parser.add_argument("--pose_no_action", action="store_true", default=False)
    parser.add_argument("--trigger_fire_only", action="store_true", default=False)
    parser.add_argument("--trigger_no_context", action="store_true", default=False)
    parser.add_argument("--trigger_no_gripper_context", action="store_true", default=False)

    args = parser.parse_args()
    if args.trigger_fire_only:
        if args.lambda_gripper != 0.0 or args.lambda_hold != 0.0:
            print("[train_residual] trigger_fire_only enabled: forcing lambda_gripper=lambda_hold=0.0")
        args.lambda_gripper = 0.0
        args.lambda_hold = 0.0
    train(args)


if __name__ == "__main__":
    main()
