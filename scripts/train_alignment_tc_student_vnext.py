#!/usr/bin/env python3
"""Train verified-only phase1/phase2 alignment student vNext."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from prismatic.models.alignment_tc_student_vnext import AlignmentTCStudentVNext
from prismatic.vla.datasets.alignment_tc_student_vnext_dataset import AlignmentTCStudentVNextDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage", choices=["stage1_estimator", "stage2_teacher_forcing", "stage3_student_finetune"], required=True)
    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_fraction", type=float, default=0.10)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max_pos_step", type=float, default=0.0015)
    ap.add_argument("--max_yaw_step", type=float, default=0.0060)
    ap.add_argument("--teacher_force_warmup_fraction", type=float, default=0.25)
    ap.add_argument("--teacher_force_decay_fraction", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--phase1_target_axis_weights", type=str, default="1.0,1.75,1.0,0.25,0.25,2.0")
    ap.add_argument("--phase2_target_axis_weights", type=str, default="1.0,1.50,1.0,0.25,0.25,1.75")
    ap.add_argument("--phase1_action_axis_weights", type=str, default="1.0,1.75,1.0,2.0")
    ap.add_argument("--phase2_action_axis_weights", type=str, default="1.0,1.50,1.0,1.75")
    ap.add_argument("--phase1_yaw_dir_weight", type=float, default=1.50)
    ap.add_argument("--phase2_yaw_dir_weight", type=float, default=1.00)
    ap.add_argument("--phase1_verified_weight_boost", type=float, default=1.15)
    ap.add_argument("--enable_phase1_bridge_repair_losses", action="store_true")
    ap.add_argument("--phase1_sign_y_weight", type=float, default=0.35)
    ap.add_argument("--phase1_sign_yaw_weight", type=float, default=0.50)
    ap.add_argument("--phase1_mag_floor_y_weight", type=float, default=0.15)
    ap.add_argument("--phase1_mag_floor_yaw_weight", type=float, default=0.20)
    ap.add_argument("--phase1_sign_target_threshold_y", type=float, default=2e-4)
    ap.add_argument("--phase1_sign_target_threshold_yaw", type=float, default=1e-3)
    ap.add_argument("--phase1_mag_floor_action_threshold_y", type=float, default=1.5e-4)
    ap.add_argument("--phase1_mag_floor_action_threshold_yaw", type=float, default=1.2e-3)
    ap.add_argument("--phase1_mag_floor_fraction_y", type=float, default=0.60)
    ap.add_argument("--phase1_mag_floor_fraction_yaw", type=float, default=0.65)
    ap.add_argument("--enable_decoupled_y_bridge", action="store_true")
    ap.add_argument("--y_bridge_max_step", type=float, default=0.0010)
    ap.add_argument("--freeze_main_residual_for_y_bridge", action="store_true")
    ap.add_argument("--enable_close_ready_bridge_supervision", action="store_true")
    ap.add_argument("--phase1_close_ready_loss_weight", type=float, default=1.0)
    ap.add_argument("--phase1_handoff_ready_loss_weight", type=float, default=0.75)
    ap.add_argument("--phase1_bridge_xy_boost", type=float, default=0.50)
    ap.add_argument("--phase1_bridge_z_boost", type=float, default=0.50)
    ap.add_argument("--phase1_bridge_yaw_floor", type=float, default=0.25)
    ap.add_argument("--phase1_bridge_yaw_cap", type=float, default=1.25)
    ap.add_argument("--freeze_all_but_close_ready_handoff_heads", action="store_true")
    return ap.parse_args()


def batch_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def parse_weight_list(raw: str, expected_len: int) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} weights, got {len(vals)} from {raw!r}")
    return vals


def teacher_force_prob(epoch: int, epochs: int, args) -> float:
    if args.stage == "stage1_estimator":
        return 0.0
    if args.stage == "stage2_teacher_forcing":
        return 1.0
    warmup = max(1, int(float(args.teacher_force_warmup_fraction) * epochs))
    decay = max(1, int(float(args.teacher_force_decay_fraction) * epochs))
    if epoch <= warmup:
        return 0.75
    t = min(max(epoch - warmup, 0), decay)
    return float(max(0.0, 0.75 * (1.0 - t / decay)))


def compute_loss(model, batch, args, tf_prob: float):
    weight = batch["sample_weight"].reshape(-1).float()
    phase_id = batch["phase_id"].reshape(-1).long()
    verified_positive = batch["verified_positive"].reshape(-1).float()
    device = weight.device

    phase1_target_w = torch.tensor(args.phase1_target_axis_weights, device=device, dtype=torch.float32)
    phase2_target_w = torch.tensor(args.phase2_target_axis_weights, device=device, dtype=torch.float32)
    phase1_action_w = torch.tensor(args.phase1_action_axis_weights, device=device, dtype=torch.float32)
    phase2_action_w = torch.tensor(args.phase2_action_axis_weights, device=device, dtype=torch.float32)
    target_axis_w = torch.where((phase_id == 0).unsqueeze(1), phase1_target_w.unsqueeze(0), phase2_target_w.unsqueeze(0))
    action_axis_w = torch.where((phase_id == 0).unsqueeze(1), phase1_action_w.unsqueeze(0), phase2_action_w.unsqueeze(0))
    yaw_dir_weight = torch.where(
        phase_id == 0,
        torch.full_like(weight, float(args.phase1_yaw_dir_weight)),
        torch.full_like(weight, float(args.phase2_yaw_dir_weight)),
    )
    verified_boost = torch.where(
        phase_id == 0,
        1.0 + torch.clamp(verified_positive, 0.0, 1.0) * float(max(args.phase1_verified_weight_boost - 1.0, 0.0)),
        torch.ones_like(weight),
    )
    main_weight = weight * verified_boost
    close_ready_score = batch.get("teacher_close_ready_score")
    if close_ready_score is None:
        close_ready_score = torch.zeros_like(weight)
    else:
        close_ready_score = close_ready_score.reshape(-1).float().clamp(0.0, 1.0)
    close_ready_bridge_mask = batch.get("close_ready_bridge_mask")
    if close_ready_bridge_mask is None:
        close_ready_bridge_mask = torch.zeros_like(weight)
    else:
        close_ready_bridge_mask = close_ready_bridge_mask.reshape(-1).float().clamp(0.0, 1.0)
    close_ready_exact_mask = batch.get("close_ready_exact_mask")
    if close_ready_exact_mask is None:
        close_ready_exact_mask = torch.zeros_like(weight)
    else:
        close_ready_exact_mask = close_ready_exact_mask.reshape(-1).float().clamp(0.0, 1.0)
    handoff_ready_label = batch.get("teacher_truth_handoff_ready")
    if handoff_ready_label is None:
        handoff_ready_label = torch.zeros_like(weight)
    else:
        handoff_ready_label = handoff_ready_label.reshape(-1).float().clamp(0.0, 1.0)

    bridge_phase1_mask = ((phase_id == 0).float() * close_ready_bridge_mask).clamp(0.0, 1.0)
    if args.enable_close_ready_bridge_supervision:
        phase1_xy_boost = 1.0 + bridge_phase1_mask * float(args.phase1_bridge_xy_boost) * (1.0 - close_ready_score)
        phase1_z_boost = 1.0 + bridge_phase1_mask * float(args.phase1_bridge_z_boost) * (1.0 - close_ready_score)
        phase1_yaw_scale = torch.where(
            bridge_phase1_mask > 0.5,
            torch.clamp(
                float(args.phase1_bridge_yaw_floor)
                + (float(args.phase1_bridge_yaw_cap) - float(args.phase1_bridge_yaw_floor)) * close_ready_score,
                min=float(args.phase1_bridge_yaw_floor),
                max=float(args.phase1_bridge_yaw_cap),
            ),
            torch.ones_like(weight),
        )
    else:
        phase1_xy_boost = torch.ones_like(weight)
        phase1_z_boost = torch.ones_like(weight)
        phase1_yaw_scale = torch.ones_like(weight)
    phase1_yaw_dir_scale = torch.where(
        phase_id == 0,
        torch.clamp(0.35 + 0.65 * close_ready_score, min=0.35, max=1.0),
        torch.ones_like(weight),
    )

    out = model(
        wrist_depth=batch["wrist_depth"],
        force_history=batch["force_history"],
        proprio=batch["proprio"],
        planner_action_local=batch["planner_action_local"],
        gripper_context=batch["gripper_context"],
        front_rgb=batch["front_rgb"],
        wrist_rgb=batch["wrist_rgb"],
        phase_id=batch["phase_id"],
        stage_bucket_id=batch["stage_bucket_id"],
        teacher_target_delta_local=batch["teacher_target_delta_local_6d"],
        teacher_contact_repr=batch["teacher_contact_repr"],
        teacher_force_prob=tf_prob,
        teacher_contact_force_prob=tf_prob,
    )

    target_delta_elem = F.smooth_l1_loss(
        out["pred_target_delta_local_6d"], batch["teacher_target_delta_local_6d"], reduction="none"
    )
    target_axis_w = target_axis_w.clone()
    target_axis_w[:, 0] = target_axis_w[:, 0] * phase1_xy_boost
    target_axis_w[:, 1] = target_axis_w[:, 1] * phase1_xy_boost
    target_axis_w[:, 2] = target_axis_w[:, 2] * phase1_z_boost
    target_axis_w[:, 5] = target_axis_w[:, 5] * phase1_yaw_scale
    target_delta_loss = weighted_mean((target_delta_elem * target_axis_w).mean(dim=1), main_weight)

    contact_repr_loss = F.smooth_l1_loss(
        out["pred_contact_repr"], batch["teacher_contact_repr"], reduction="none"
    ).mean(dim=1)
    contact_repr_loss = weighted_mean(contact_repr_loss, weight)

    confidence_loss = F.binary_cross_entropy_with_logits(
        out["target_confidence_logit"], batch["teacher_confidence_label"].reshape(-1), reduction="none"
    )
    confidence_loss = weighted_mean(confidence_loss, weight)

    progress_prior_loss = F.binary_cross_entropy_with_logits(
        out["progress_prior_logits"], batch["teacher_progress_label"], reduction="none"
    ).mean(dim=1)
    progress_prior_loss = weighted_mean(progress_prior_loss, weight)

    yaw_mask = batch["yaw_imitation_enabled"].reshape(-1).float()
    yaw_weight = main_weight * torch.clamp(yaw_mask, min=0.0, max=1.0) * yaw_dir_weight * phase1_yaw_dir_scale
    yaw_dir_loss = F.cross_entropy(out["yaw_direction_logits"], batch["yaw_direction_label"], reduction="none")
    yaw_dir_loss = weighted_mean(yaw_dir_loss, yaw_weight + 1e-6)

    traj_target = batch["teacher_residual_trajectory_4d"][:, : out["trajectory_4d"].shape[1], :]
    traj_elem = F.smooth_l1_loss(out["trajectory_4d"], traj_target, reduction="none")
    action_axis_w = action_axis_w.clone()
    action_axis_w[:, 0] = action_axis_w[:, 0] * phase1_xy_boost
    action_axis_w[:, 1] = action_axis_w[:, 1] * phase1_xy_boost
    action_axis_w[:, 2] = action_axis_w[:, 2] * phase1_z_boost
    action_axis_w[:, 3] = action_axis_w[:, 3] * phase1_yaw_scale
    traj_loss = weighted_mean((traj_elem * action_axis_w.unsqueeze(1)).mean(dim=(1, 2)), main_weight)

    first_pred = out["first_residual_4d"]
    first_target = batch["teacher_residual_action_4d"]
    action_elem = F.smooth_l1_loss(first_pred, first_target, reduction="none")
    action_loss = weighted_mean((action_elem * action_axis_w).mean(dim=1), main_weight)

    close_ready_target = close_ready_score
    close_ready_logit = out["close_ready_logit"]
    close_ready_loss = F.binary_cross_entropy_with_logits(close_ready_logit, close_ready_target, reduction="none")
    close_ready_loss = weighted_mean(close_ready_loss, main_weight * (1.0 + bridge_phase1_mask + close_ready_exact_mask))

    handoff_ready_logit = out["handoff_ready_logit"]
    handoff_ready_loss = F.binary_cross_entropy_with_logits(handoff_ready_logit, handoff_ready_label, reduction="none")
    handoff_ready_loss = weighted_mean(handoff_ready_loss, main_weight * (1.0 + close_ready_exact_mask))

    progress_loss = F.binary_cross_entropy_with_logits(
        out["progress_logits"], batch["teacher_progress_label"], reduction="none"
    ).mean(dim=1)
    progress_loss = weighted_mean(progress_loss, main_weight)

    risk_loss = F.binary_cross_entropy_with_logits(out["risk_logit"], batch["teacher_risk_label"].reshape(-1), reduction="none")
    risk_loss = weighted_mean(risk_loss, weight)

    stop_loss = F.binary_cross_entropy_with_logits(out["stop_logit"], batch["teacher_stop_label"].reshape(-1), reduction="none")
    stop_loss = weighted_mean(stop_loss, weight)

    apply_conf_loss = F.binary_cross_entropy_with_logits(
        out["apply_confidence_logit"],
        batch["verified_positive"].reshape(-1),
        reduction="none",
    )
    apply_conf_loss = weighted_mean(apply_conf_loss, weight)

    smooth_loss = torch.zeros((), device=first_pred.device)
    if out["trajectory_4d"].shape[1] > 1:
        smooth_loss = (out["trajectory_4d"][:, 1:] - out["trajectory_4d"][:, :-1]).square().mean(dim=(1, 2))
        smooth_loss = weighted_mean(smooth_loss, main_weight)

    sign_y_loss = torch.zeros((), device=first_pred.device)
    sign_yaw_loss = torch.zeros((), device=first_pred.device)
    mag_floor_y_loss = torch.zeros((), device=first_pred.device)
    mag_floor_yaw_loss = torch.zeros((), device=first_pred.device)
    if args.enable_phase1_bridge_repair_losses:
        phase1_mask = (phase_id == 0).float()
        positive_mask = torch.clamp(verified_positive, 0.0, 1.0)
        y_target = batch["teacher_target_delta_local_6d"][:, 1]
        y_action = batch["teacher_residual_action_4d"][:, 1]
        y_pred = first_pred[:, 1]
        y_sign_mask = phase1_mask * positive_mask * (y_target.abs() >= float(args.phase1_sign_target_threshold_y)).float()
        if float(y_sign_mask.sum().detach().cpu()) > 0.0:
            y_sign = torch.sign(y_target).detach()
            sign_y_elem = F.softplus(-(y_sign * y_pred) / 5e-4)
            sign_y_loss = weighted_mean(sign_y_elem, main_weight * y_sign_mask)
        y_floor_mask = phase1_mask * positive_mask * (y_action.abs() >= float(args.phase1_mag_floor_action_threshold_y)).float()
        if float(y_floor_mask.sum().detach().cpu()) > 0.0:
            y_floor = y_action.abs().detach() * float(args.phase1_mag_floor_fraction_y)
            mag_floor_y_elem = torch.relu(y_floor - y_pred.abs())
            mag_floor_y_loss = weighted_mean(mag_floor_y_elem, main_weight * y_floor_mask)

        yaw_target = batch["teacher_target_delta_local_6d"][:, 5]
        yaw_action = batch["teacher_residual_action_4d"][:, 3]
        yaw_pred = first_pred[:, 3]
        yaw_dir = batch["yaw_direction_label"].reshape(-1)
        yaw_non_neutral = (yaw_dir != 1).float()
        yaw_sign_mask = phase1_mask * positive_mask * yaw_non_neutral * (yaw_target.abs() >= float(args.phase1_sign_target_threshold_yaw)).float()
        if float(yaw_sign_mask.sum().detach().cpu()) > 0.0:
            yaw_sign = torch.where(yaw_dir == 2, torch.ones_like(yaw_pred), -torch.ones_like(yaw_pred)).detach()
            sign_yaw_elem = F.softplus(-(yaw_sign * yaw_pred) / 1e-3)
            sign_yaw_loss = weighted_mean(sign_yaw_elem, main_weight * yaw_sign_mask)
        yaw_floor_mask = phase1_mask * positive_mask * yaw_non_neutral * (yaw_action.abs() >= float(args.phase1_mag_floor_action_threshold_yaw)).float()
        if float(yaw_floor_mask.sum().detach().cpu()) > 0.0:
            yaw_floor = yaw_action.abs().detach() * float(args.phase1_mag_floor_fraction_yaw)
            mag_floor_yaw_elem = torch.relu(yaw_floor - yaw_pred.abs())
            mag_floor_yaw_loss = weighted_mean(mag_floor_yaw_elem, main_weight * yaw_floor_mask)

    if args.stage == "stage1_estimator":
        total = (
            1.0 * target_delta_loss
            + 1.0 * contact_repr_loss
            + 0.25 * confidence_loss
            + 0.5 * progress_prior_loss
            + 0.75 * yaw_dir_loss
        )
    else:
        total = (
            1.0 * target_delta_loss
            + 1.0 * contact_repr_loss
            + 2.0 * action_loss
            + 1.0 * traj_loss
            + 0.5 * progress_loss
            + 0.5 * risk_loss
            + 0.5 * stop_loss
            + 0.25 * confidence_loss
            + 0.25 * apply_conf_loss
            + 0.1 * smooth_loss
            + 0.5 * progress_prior_loss
            + 0.75 * yaw_dir_loss
        )
        if args.enable_close_ready_bridge_supervision:
            total = total + float(args.phase1_close_ready_loss_weight) * close_ready_loss
            total = total + float(args.phase1_handoff_ready_loss_weight) * handoff_ready_loss
        if args.enable_phase1_bridge_repair_losses:
            total = (
                total
                + float(args.phase1_sign_y_weight) * sign_y_loss
                + float(args.phase1_sign_yaw_weight) * sign_yaw_loss
                + float(args.phase1_mag_floor_y_weight) * mag_floor_y_loss
                + float(args.phase1_mag_floor_yaw_weight) * mag_floor_yaw_loss
            )

    metrics = {
        "loss": float(total.detach().cpu()),
        "target_delta_loss": float(target_delta_loss.detach().cpu()),
        "contact_repr_loss": float(contact_repr_loss.detach().cpu()),
        "confidence_loss": float(confidence_loss.detach().cpu()),
        "progress_prior_loss": float(progress_prior_loss.detach().cpu()),
        "yaw_dir_loss": float(yaw_dir_loss.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "traj_loss": float(traj_loss.detach().cpu()),
        "progress_loss": float(progress_loss.detach().cpu()),
        "risk_loss": float(risk_loss.detach().cpu()),
        "stop_loss": float(stop_loss.detach().cpu()),
        "apply_conf_loss": float(apply_conf_loss.detach().cpu()),
        "close_ready_loss": float(close_ready_loss.detach().cpu()),
        "handoff_ready_loss": float(handoff_ready_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "sign_y_loss": float(sign_y_loss.detach().cpu()),
        "sign_yaw_loss": float(sign_yaw_loss.detach().cpu()),
        "mag_floor_y_loss": float(mag_floor_y_loss.detach().cpu()),
        "mag_floor_yaw_loss": float(mag_floor_yaw_loss.detach().cpu()),
        "teacher_force_prob": float(tf_prob),
    }
    return total, metrics


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
    args.phase1_target_axis_weights = parse_weight_list(args.phase1_target_axis_weights, 6)
    args.phase2_target_axis_weights = parse_weight_list(args.phase2_target_axis_weights, 6)
    args.phase1_action_axis_weights = parse_weight_list(args.phase1_action_axis_weights, 4)
    args.phase2_action_axis_weights = parse_weight_list(args.phase2_action_axis_weights, 4)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = AlignmentTCStudentVNextDataset(args.dataset)
    val_len = max(1, int(len(dataset) * args.val_fraction))
    train_len = max(1, len(dataset) - val_len)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlignmentTCStudentVNext(
        horizon=args.horizon,
        max_pos_step=args.max_pos_step,
        max_yaw_step=args.max_yaw_step,
        y_bridge_max_step=args.y_bridge_max_step,
    ).to(device)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(json.dumps({"init_ckpt": args.init_ckpt, "missing": missing, "unexpected": unexpected}))

    if args.stage == "stage2_teacher_forcing":
        for name, param in model.named_parameters():
            if name.startswith("target_delta_head") or name.startswith("contact_repr_head") or name.startswith("confidence_head") or name.startswith("progress_prior_head") or name.startswith("yaw_direction_head") or name.startswith("obs_trunk") or name.startswith("front_rgb_encoder") or name.startswith("wrist_rgb_encoder") or name.startswith("depth_encoder") or name.startswith("force_encoder") or name.startswith("proprio_encoder") or name.startswith("planner_encoder") or name.startswith("gripper_encoder") or name.startswith("phase_embedding") or name.startswith("stage_embedding"):
                param.requires_grad = False

    if args.enable_decoupled_y_bridge and args.freeze_main_residual_for_y_bridge:
        for name, param in model.named_parameters():
            if name.startswith("y_bridge_head"):
                param.requires_grad = True
            elif name.startswith("traj_head") or name.startswith("scale_head") or name.startswith("policy_trunk") or name.startswith("target_repr_head") or name.startswith("target_delta_head") or name.startswith("contact_repr_head") or name.startswith("yaw_direction_head") or name.startswith("obs_trunk") or name.startswith("front_rgb_encoder") or name.startswith("wrist_rgb_encoder") or name.startswith("depth_encoder") or name.startswith("force_encoder") or name.startswith("proprio_encoder") or name.startswith("planner_encoder") or name.startswith("gripper_encoder") or name.startswith("phase_embedding") or name.startswith("stage_embedding") or name.startswith("progress_head") or name.startswith("risk_head") or name.startswith("stop_head") or name.startswith("apply_confidence_head") or name.startswith("confidence_head") or name.startswith("progress_prior_head"):
                param.requires_grad = False

    if args.freeze_all_but_close_ready_handoff_heads:
        for name, param in model.named_parameters():
            if name.startswith("close_ready_head") or name.startswith("handoff_ready_head"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    history = []
    best_val = float("inf")
    best_path = out_dir / "alignment_tc_student_vnext_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, device, epoch, args.epochs, optimizer)
        val_metrics = run_epoch(model, val_loader, args, device, epoch, args.epochs, None)
        row = {"epoch": epoch, "stage": args.stage, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "controller_type": "alignment_tc_student_vnext",
                    "horizon": args.horizon,
                    "max_pos_step": args.max_pos_step,
                    "max_yaw_step": args.max_yaw_step,
                    "y_bridge_max_step": args.y_bridge_max_step,
                    "enable_decoupled_y_bridge": bool(args.enable_decoupled_y_bridge),
                    "dataset": str(args.dataset),
                    "train_stage": args.stage,
                    "best_val_loss": best_val,
                    "use_front_rgb": False,
                    "use_wrist_rgb": True,
                    "use_wrist_depth": True,
                    "use_force": True,
                    "use_planner_action": True,
                },
                best_path,
            )
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
