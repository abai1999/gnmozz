#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from prismatic.models.depth_force_contact_policy import DepthForceLocalContactPolicy
from prismatic.models.depth_force_future_risk_head import DepthForceCandidateFutureRiskHead
from prismatic.vla.datasets.depth_force_candidate_future_risk_dataset import DepthForceCandidateFutureRiskDataset


def _split_episodes(dataset: DepthForceCandidateFutureRiskDataset, val_episodes_csv: str | None) -> tuple[np.ndarray, np.ndarray]:
    eps = np.asarray(dataset.data.get("episode_index", np.zeros((dataset.length,), dtype=np.int64)), dtype=np.int64)
    uniq = np.unique(eps)
    if val_episodes_csv:
        val_eps = np.asarray([int(x) for x in val_episodes_csv.split(",") if x.strip()], dtype=np.int64)
    else:
        val_eps = uniq[-1:] if uniq.size else np.zeros((0,), dtype=np.int64)
    train_idx = np.where(~np.isin(eps, val_eps))[0]
    val_idx = np.where(np.isin(eps, val_eps))[0]
    return train_idx, val_idx


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.float().sum(), min=1.0)
    return (x * mask.float()).sum() / denom


def _pairwise_rank_loss(pred_risk: torch.Tensor, target_risk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Lower target risk should correspond to lower predicted risk."""
    score_diff = pred_risk.unsqueeze(1) - pred_risk.unsqueeze(2)
    cost_diff = target_risk.unsqueeze(1) - target_risk.unsqueeze(2)
    pair_mask = (cost_diff > 1e-6) & mask.unsqueeze(2) & mask.unsqueeze(1)
    if not torch.any(pair_mask):
        return pred_risk.sum() * 0.0
    # If target_i > target_j, then pred_i should also be greater (worse).
    loss = F.softplus(-(score_diff)) * torch.clamp(cost_diff, min=0.0)
    return loss[pair_mask].sum() / torch.clamp(torch.clamp(cost_diff, min=0.0)[pair_mask].sum(), min=1e-6)


def _rowwise_normalize(cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = torch.clamp(mask_f.sum(dim=1, keepdim=True), min=1.0)
    mean = (cost * mask_f).sum(dim=1, keepdim=True) / denom
    var = (((cost - mean) * mask_f) ** 2).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return ((cost - mean) / std).masked_fill(~mask, 0.0)


def _binary_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().cpu().to(torch.bool)
    target = target.detach().cpu().to(torch.bool)
    tp = torch.sum(pred & target).item()
    fp = torch.sum(pred & ~target).item()
    fn = torch.sum(~pred & target).item()
    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.detach().cpu().flatten().numpy()
    y = labels.detach().cpu().flatten().numpy().astype(np.int64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    pos_rank_sum = float(ranks[pos].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc)


def _brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = probs.detach().cpu().to(torch.float32).flatten()
    labels = labels.detach().cpu().to(torch.float32).flatten()
    if probs.numel() == 0:
        return 0.0
    return float(torch.mean((probs - labels) ** 2).item())


def _expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    probs = probs.detach().cpu().to(torch.float32).flatten()
    labels = labels.detach().cpu().to(torch.float32).flatten()
    if probs.numel() == 0:
        return 0.0
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = float(len(probs_np))
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < 1.0:
            sel = (probs_np >= lo) & (probs_np < hi)
        else:
            sel = (probs_np >= lo) & (probs_np <= hi)
        if not np.any(sel):
            continue
        conf = float(np.mean(probs_np[sel]))
        acc = float(np.mean(labels_np[sel]))
        ece += (float(np.sum(sel)) / total) * abs(conf - acc)
    return float(ece)


def _candidate_pair_accuracy(pred_risk: torch.Tensor, true_label: torch.Tensor, baseline_idx: torch.Tensor) -> float:
    row = torch.arange(pred_risk.shape[0], device=pred_risk.device)
    pred_delta = pred_risk - pred_risk[row, baseline_idx][:, None]
    pred_label = pred_delta > 0.0
    return float(torch.mean((pred_label == (true_label > 0.5)).float()).item())


def _baseline_geometry_risk_accuracy(
    pred_risk: torch.Tensor,
    geom_idx: torch.Tensor,
    baseline_idx: torch.Tensor,
    true_delta: torch.Tensor,
) -> float:
    row = torch.arange(pred_risk.shape[0], device=pred_risk.device)
    pred_rel = pred_risk[row, geom_idx] > pred_risk[row, baseline_idx]
    true_rel = true_delta[row, geom_idx] > 0.0
    return float(torch.mean((pred_rel == true_rel).float()).item())


def _evaluate(
    model: DepthForceCandidateFutureRiskHead,
    geom_model: DepthForceLocalContactPolicy | None,
    loader: DataLoader,
    device: torch.device,
    use_geometry_score_feature: bool,
) -> dict[str, float]:
    model.eval()
    if geom_model is not None:
        geom_model.eval()
    total_batches = 0
    agg: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        agg.setdefault(name, []).append(float(value))

    with torch.no_grad():
        for batch in loader:
            front = batch["front_rgb"].to(device=device, dtype=torch.float32)
            wrist = batch["wrist_rgb"].to(device=device, dtype=torch.float32)
            depth = batch["wrist_depth"].to(device=device, dtype=torch.float32)
            force_hist = batch["force_history"].to(device=device, dtype=torch.float32)
            proprio = batch["proprio"].to(device=device, dtype=torch.float32)
            planner = batch["planner_base_action_local"].to(device=device, dtype=torch.float32)
            candidates = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
            mask = batch["candidate_mask"].to(device=device, dtype=torch.float32) > 0.5
            stage_token = batch["stage_token"].to(device=device)
            contact_phase = batch["contact_phase"].to(device=device)
            depth_prox = batch["depth_proximity"].to(device=device, dtype=torch.float32)
            gripper_state = batch["gripper_state"].to(device=device, dtype=torch.float32)
            baseline_idx = batch["candidate_baseline_index"].to(device=device)
            geom_idx = batch["candidate_geom_index"].to(device=device)
            true_geom_cost = batch["candidate_geometry_cost"].to(device=device, dtype=torch.float32)
            true_total_risk = batch["candidate_future_risk_score"].to(device=device, dtype=torch.float32)
            true_risk_increase = batch["candidate_future_risk_label"].to(device=device, dtype=torch.float32)
            true_risk_noninc = batch["candidate_future_risk_nonincrease_label"].to(device=device, dtype=torch.float32)
            true_contact = batch["candidate_future_contact_risk"].to(device=device, dtype=torch.float32)
            true_spike = batch["candidate_future_force_spike_risk"].to(device=device, dtype=torch.float32)
            true_jam = batch["candidate_future_jam_risk"].to(device=device, dtype=torch.float32)
            true_stall = batch["candidate_future_motion_stall_risk"].to(device=device, dtype=torch.float32)
            true_kin = batch["candidate_future_kinematic_invalid_risk"].to(device=device, dtype=torch.float32)
            true_action_invalid = batch["candidate_future_action_range_invalid_risk"].to(device=device, dtype=torch.float32)

            geom_scores = None
            if use_geometry_score_feature and geom_model is not None:
                geom_out = geom_model(
                    front_rgb=front,
                    wrist_rgb=wrist,
                    wrist_depth=depth,
                    force_history=force_hist,
                    proprio=proprio,
                    planner_base_action_local=planner,
                    candidate_actions_local=candidates,
                    candidate_mask=mask.float(),
                    stage_token=stage_token,
                    contact_phase=contact_phase,
                    depth_proximity=depth_prox,
                    gripper_state=gripper_state,
                )
                geom_scores = geom_out["candidate_geometry_value"].detach()

            out = model(
                front_rgb=front,
                wrist_rgb=wrist,
                wrist_depth=depth,
                force_history=force_hist,
                proprio=proprio,
                planner_base_action_local=planner,
                candidate_actions_local=candidates,
                candidate_mask=mask.float(),
                geometry_scores=geom_scores,
                stage_token=stage_token,
                contact_phase=contact_phase,
                depth_proximity=depth_prox,
                gripper_state=gripper_state,
            )

            pred_total = out["future_total_risk"]
            pred_contact = out["future_contact_risk"]
            pred_spike = out["future_force_spike_risk"]
            pred_jam = out["future_jam_risk"]
            pred_stall = out["future_motion_stall_risk"]
            pred_kin = out["future_kinematic_invalid_risk"]
            pred_action_invalid = out["future_action_invalid_risk"]

            row = torch.arange(pred_total.shape[0], device=device)
            pred_base = pred_total[row, baseline_idx]
            pred_geom = pred_total[row, geom_idx]
            pred_total_delta = pred_total - pred_base[:, None]
            pred_risk_increase = pred_total_delta > 0.0
            pred_risk_noninc = ~pred_risk_increase
            pred_risk_prob = torch.sigmoid(pred_total_delta)
            pred_contact_lbl = pred_contact > 0.0
            pred_spike_lbl = pred_spike > 0.0
            pred_jam_lbl = pred_jam > 0.0
            pred_stall_lbl = pred_stall > 0.0
            pred_kin_lbl = pred_kin > 0.0
            pred_action_invalid_lbl = pred_action_invalid > 0.0

            target_norm = _rowwise_normalize(true_total_risk, mask)
            reg_loss = F.smooth_l1_loss(pred_total[mask], target_norm[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            pair_loss = _pairwise_rank_loss(pred_total, target_norm, mask)
            inc_loss = F.binary_cross_entropy_with_logits(pred_total_delta[mask], true_risk_increase[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            noninc_loss = F.binary_cross_entropy_with_logits(-pred_total_delta[mask], true_risk_noninc[mask]) if torch.any(mask) else pred_total.sum() * 0.0

            contact_loss = F.binary_cross_entropy_with_logits(pred_contact[mask], true_contact[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            spike_loss = F.binary_cross_entropy_with_logits(pred_spike[mask], true_spike[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            jam_loss = F.binary_cross_entropy_with_logits(pred_jam[mask], true_jam[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            stall_loss = F.binary_cross_entropy_with_logits(pred_stall[mask], true_stall[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            kin_loss = F.binary_cross_entropy_with_logits(pred_kin[mask], true_kin[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            action_invalid_loss = F.binary_cross_entropy_with_logits(pred_action_invalid[mask], true_action_invalid[mask]) if torch.any(mask) else pred_total.sum() * 0.0

            pred_inc_auc = _roc_auc(pred_total_delta[mask], true_risk_increase[mask]) if torch.any(mask) else 0.0
            pred_inc_f1 = _binary_f1(pred_risk_increase[mask], true_risk_increase[mask]) if torch.any(mask) else 0.0
            pred_noninc_f1 = _binary_f1(pred_risk_noninc[mask], true_risk_noninc[mask]) if torch.any(mask) else 0.0
            pred_inc_brier = _brier_score(pred_risk_prob[mask], true_risk_increase[mask]) if torch.any(mask) else 0.0
            pred_inc_ece = _expected_calibration_error(pred_risk_prob[mask], true_risk_increase[mask]) if torch.any(mask) else 0.0
            candidate_pair_acc = _candidate_pair_accuracy(pred_total, true_risk_increase, baseline_idx)
            baseline_geom_acc = _baseline_geometry_risk_accuracy(pred_total, geom_idx, baseline_idx, true_total_risk)
            geometry_best_vs_baseline = torch.mean((true_geom_cost[row, geom_idx] < true_geom_cost[row, baseline_idx] - 1e-6).float()).item()
            geometry_best_vs_baseline_gap = torch.mean(true_geom_cost[row, baseline_idx] - true_geom_cost[row, geom_idx]).item()

            add("reg_loss", float(reg_loss.item()))
            add("pair_loss", float(pair_loss.item()))
            add("inc_loss", float(inc_loss.item()))
            add("noninc_loss", float(noninc_loss.item()))
            add("contact_loss", float(contact_loss.item()))
            add("spike_loss", float(spike_loss.item()))
            add("jam_loss", float(jam_loss.item()))
            add("stall_loss", float(stall_loss.item()))
            add("kin_loss", float(kin_loss.item()))
            add("action_invalid_loss", float(action_invalid_loss.item()))
            add("future_risk_auc", float(pred_inc_auc))
            add("risk_increase_f1", float(pred_inc_f1))
            add("risk_nonincrease_f1", float(pred_noninc_f1))
            add("risk_increase_brier", float(pred_inc_brier))
            add("risk_increase_ece", float(pred_inc_ece))
            add("candidate_pair_accuracy", float(candidate_pair_acc))
            add("baseline_vs_geometry_risk_accuracy", float(baseline_geom_acc))
            add("geometry_best_vs_baseline_gap", float(geometry_best_vs_baseline_gap))
            add("geometry_best_vs_baseline_rate", float(geometry_best_vs_baseline))
            add("per_risk_contact_f1", _binary_f1(pred_contact_lbl[mask], true_contact[mask]) if torch.any(mask) else 0.0)
            add("per_risk_spike_f1", _binary_f1(pred_spike_lbl[mask], true_spike[mask]) if torch.any(mask) else 0.0)
            add("per_risk_jam_f1", _binary_f1(pred_jam_lbl[mask], true_jam[mask]) if torch.any(mask) else 0.0)
            add("per_risk_stall_f1", _binary_f1(pred_stall_lbl[mask], true_stall[mask]) if torch.any(mask) else 0.0)
            add("per_risk_kin_f1", _binary_f1(pred_kin_lbl[mask], true_kin[mask]) if torch.any(mask) else 0.0)
            add("per_risk_action_invalid_f1", _binary_f1(pred_action_invalid_lbl[mask], true_action_invalid[mask]) if torch.any(mask) else 0.0)
            total_batch = {
                "rows": float(pred_total.shape[0]),
                "risk_increase_rate": float(true_risk_increase.float().mean().item()),
                "risk_nonincrease_rate": float(true_risk_noninc.float().mean().item()),
                "geometry_selected_risk_increase_rate": float((true_total_risk[row, geom_idx] > true_total_risk[row, baseline_idx] + 1e-6).float().mean().item()),
            }
            add("geometry_selected_risk_increase_rate", total_batch["geometry_selected_risk_increase_rate"])
            total_batches += 1

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
    out["batches"] = int(total_batches)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--geometry_checkpoint", default="")
    ap.add_argument("--val_episodes", default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_geometry_score_feature", action="store_true", default=False)
    ap.add_argument("--freeze_backbone", action="store_true", default=False)
    ap.add_argument("--reg_weight", type=float, default=0.75)
    ap.add_argument("--pair_weight", type=float, default=0.75)
    ap.add_argument("--inc_weight", type=float, default=0.75)
    ap.add_argument("--noninc_weight", type=float, default=0.75)
    ap.add_argument("--contact_weight", type=float, default=0.50)
    ap.add_argument("--spike_weight", type=float, default=0.50)
    ap.add_argument("--jam_weight", type=float, default=0.50)
    ap.add_argument("--stall_weight", type=float, default=0.35)
    ap.add_argument("--kin_weight", type=float, default=0.35)
    ap.add_argument("--action_invalid_weight", type=float, default=0.35)
    ap.add_argument("--val_every", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = DepthForceCandidateFutureRiskDataset(args.dataset_npz)
    train_idx, val_idx = _split_episodes(dataset, args.val_episodes)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)

    geom_model = None
    backbone = None
    if args.geometry_checkpoint:
        geom_ckpt = torch.load(args.geometry_checkpoint, map_location="cpu")
        geom_model = DepthForceLocalContactPolicy().to(device)
        geom_model.load_state_dict(geom_ckpt["model_state_dict"], strict=True)
        geom_model.eval()
        backbone = DepthForceLocalContactPolicy()
        backbone.load_state_dict(geom_ckpt["model_state_dict"], strict=True)
    model = DepthForceCandidateFutureRiskHead(backbone=backbone, freeze_backbone=bool(args.freeze_backbone or args.geometry_checkpoint)).to(device)
    if args.geometry_checkpoint and not args.freeze_backbone:
        # If a geometry checkpoint is provided, the shared backbone should stay fixed
        # unless the caller explicitly opts out.
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = None
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        if geom_model is not None:
            geom_model.eval()
        train_losses: list[float] = []
        for batch in train_loader:
            front = batch["front_rgb"].to(device=device, dtype=torch.float32)
            wrist = batch["wrist_rgb"].to(device=device, dtype=torch.float32)
            depth = batch["wrist_depth"].to(device=device, dtype=torch.float32)
            force_hist = batch["force_history"].to(device=device, dtype=torch.float32)
            proprio = batch["proprio"].to(device=device, dtype=torch.float32)
            planner = batch["planner_base_action_local"].to(device=device, dtype=torch.float32)
            candidates = batch["candidate_actions_local"].to(device=device, dtype=torch.float32)
            mask = batch["candidate_mask"].to(device=device, dtype=torch.float32) > 0.5
            stage_token = batch["stage_token"].to(device=device)
            contact_phase = batch["contact_phase"].to(device=device)
            depth_prox = batch["depth_proximity"].to(device=device, dtype=torch.float32)
            gripper_state = batch["gripper_state"].to(device=device, dtype=torch.float32)
            baseline_idx = batch["candidate_baseline_index"].to(device=device)
            geom_idx = batch["candidate_geom_index"].to(device=device)
            true_total_risk = batch["candidate_future_risk_score"].to(device=device, dtype=torch.float32)
            true_risk_increase = batch["candidate_future_risk_label"].to(device=device, dtype=torch.float32)
            true_risk_noninc = batch["candidate_future_risk_nonincrease_label"].to(device=device, dtype=torch.float32)
            true_contact = batch["candidate_future_contact_risk"].to(device=device, dtype=torch.float32)
            true_spike = batch["candidate_future_force_spike_risk"].to(device=device, dtype=torch.float32)
            true_jam = batch["candidate_future_jam_risk"].to(device=device, dtype=torch.float32)
            true_stall = batch["candidate_future_motion_stall_risk"].to(device=device, dtype=torch.float32)
            true_kin = batch["candidate_future_kinematic_invalid_risk"].to(device=device, dtype=torch.float32)
            true_action_invalid = batch["candidate_future_action_range_invalid_risk"].to(device=device, dtype=torch.float32)

            geom_scores = None
            if geom_model is not None and args.use_geometry_score_feature:
                with torch.no_grad():
                    geom_out = geom_model(
                        front_rgb=front,
                        wrist_rgb=wrist,
                        wrist_depth=depth,
                        force_history=force_hist,
                        proprio=proprio,
                        planner_base_action_local=planner,
                        candidate_actions_local=candidates,
                        candidate_mask=mask.float(),
                        stage_token=stage_token,
                        contact_phase=contact_phase,
                        depth_proximity=depth_prox,
                        gripper_state=gripper_state,
                    )
                    geom_scores = geom_out["candidate_geometry_value"].detach()

            out = model(
                front_rgb=front,
                wrist_rgb=wrist,
                wrist_depth=depth,
                force_history=force_hist,
                proprio=proprio,
                planner_base_action_local=planner,
                candidate_actions_local=candidates,
                candidate_mask=mask.float(),
                geometry_scores=geom_scores,
                stage_token=stage_token,
                contact_phase=contact_phase,
                depth_proximity=depth_prox,
                gripper_state=gripper_state,
            )
            pred_total = out["future_total_risk"]
            pred_contact = out["future_contact_risk"]
            pred_spike = out["future_force_spike_risk"]
            pred_jam = out["future_jam_risk"]
            pred_stall = out["future_motion_stall_risk"]
            pred_kin = out["future_kinematic_invalid_risk"]
            pred_action_invalid = out["future_action_invalid_risk"]
            row = torch.arange(pred_total.shape[0], device=device)
            pred_base = pred_total[row, baseline_idx]
            pred_delta = pred_total - pred_base[:, None]

            target_norm = _rowwise_normalize(true_total_risk, mask)
            reg_loss = F.smooth_l1_loss(pred_total[mask], target_norm[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            pair_loss = _pairwise_rank_loss(pred_total, target_norm, mask)
            inc_loss = F.binary_cross_entropy_with_logits(pred_delta[mask], true_risk_increase[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            noninc_loss = F.binary_cross_entropy_with_logits(-pred_delta[mask], true_risk_noninc[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            contact_loss = F.binary_cross_entropy_with_logits(pred_contact[mask], true_contact[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            spike_loss = F.binary_cross_entropy_with_logits(pred_spike[mask], true_spike[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            jam_loss = F.binary_cross_entropy_with_logits(pred_jam[mask], true_jam[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            stall_loss = F.binary_cross_entropy_with_logits(pred_stall[mask], true_stall[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            kin_loss = F.binary_cross_entropy_with_logits(pred_kin[mask], true_kin[mask]) if torch.any(mask) else pred_total.sum() * 0.0
            action_invalid_loss = F.binary_cross_entropy_with_logits(pred_action_invalid[mask], true_action_invalid[mask]) if torch.any(mask) else pred_total.sum() * 0.0

            loss = (
                args.reg_weight * reg_loss
                + args.pair_weight * pair_loss
                + args.inc_weight * inc_loss
                + args.noninc_weight * noninc_loss
                + args.contact_weight * contact_loss
                + args.spike_weight * spike_loss
                + args.jam_weight * jam_loss
                + args.stall_weight * stall_loss
                + args.kin_weight * kin_loss
                + args.action_invalid_weight * action_invalid_loss
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
            optim.step()
            train_losses.append(float(loss.item()))

        if epoch % int(args.val_every) == 0 or epoch == int(args.epochs):
            val_metrics = _evaluate(
                model=model,
                geom_model=geom_model,
                loader=val_loader,
                device=device,
                use_geometry_score_feature=bool(args.use_geometry_score_feature),
            )
            score = (
                0.30 * val_metrics.get("future_risk_auc", 0.0)
                + 0.20 * val_metrics.get("risk_increase_f1", 0.0)
                + 0.20 * val_metrics.get("risk_nonincrease_f1", 0.0)
                + 0.20 * val_metrics.get("candidate_pair_accuracy", 0.0)
                + 0.10 * val_metrics.get("baseline_vs_geometry_risk_accuracy", 0.0)
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                    **val_metrics,
                    "score": float(score),
                }
            )
            if best_score is None or score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "args": vars(args),
                        "best_score": float(score),
                        "epoch": int(epoch),
                    },
                    out_dir / "depth_force_candidate_future_risk_head_best.pt",
                )
            (out_dir / "future_risk_train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps(history[-1], indent=2))

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    final_metrics = _evaluate(
        model=model,
        geom_model=geom_model,
        loader=val_loader,
        device=device,
        use_geometry_score_feature=bool(args.use_geometry_score_feature),
    )
    report = {
        "dataset_npz": args.dataset_npz,
        "geometry_checkpoint": args.geometry_checkpoint,
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "val_episodes": [int(x) for x in np.unique(dataset.data.get("episode_index", np.zeros((dataset.length,), dtype=np.int64))[val_idx])],
        "history": history,
        "final_val_metrics": final_metrics,
        "best_score": float(best_score) if best_score is not None else 0.0,
        "checkpoint_path": str(out_dir / "depth_force_candidate_future_risk_head_best.pt"),
    }
    (out_dir / "future_risk_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
