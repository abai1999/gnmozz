#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from prismatic.models.depth_force_contact_policy import DepthForceLocalContactPolicy


MODE_TO_INDEX = {
    "planner": 0,
    "align_refine": 1,
    "near_hold": 2,
    "contact_backoff": 3,
    "kinematic_hold": 4,
}
INDEX_TO_MODE = {v: k for k, v in MODE_TO_INDEX.items()}


class PrivilegedGeometryCandidateDataset(Dataset):
    def __init__(self, npz_path: str):
        raw = np.load(npz_path, allow_pickle=False)
        self.data = {k: np.asarray(raw[k]) for k in raw.files}
        self.length = int(self.data["candidate_actions_local"].shape[0])
        self.mode_labels = self._build_mode_labels()

    def _build_mode_labels(self) -> np.ndarray:
        modes = np.asarray(self.data.get("candidate_target_mode", self.data.get("target_mode"))).astype(str)
        y = np.asarray([MODE_TO_INDEX.get(str(m), MODE_TO_INDEX["planner"]) for m in modes], dtype=np.int64)
        return y

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        force_arr = self.data.get("force_history_normalized", self.data.get("force_history", self.data.get("ft_hist")))
        if force_arr is None:
            force_arr = np.zeros((self.length, 32, 6), dtype=np.float32)
        force_hist = torch.from_numpy(np.asarray(force_arr[idx], dtype=np.float32))
        if force_hist.ndim == 2 and force_hist.shape[-1] == 6:
            pass
        elif force_hist.ndim == 2 and force_hist.shape[0] == 6:
            force_hist = force_hist.transpose(0, 1)
        proprio = torch.from_numpy(self.data["proprio"][idx].astype(np.float32))
        planner = torch.from_numpy(
            np.asarray(
                self.data.get("planner_base_action_local_raw", self.data.get("planner_base_action_local"))[idx],
                dtype=np.float32,
            )
        )
        candidates = torch.from_numpy(self.data["candidate_actions_local"][idx].astype(np.float32))
        mask_default = np.ones((self.data["candidate_actions_local"].shape[1],), dtype=np.float32)
        mask = torch.from_numpy(self.data.get("candidate_mask", mask_default)[idx].astype(np.float32))
        stage_token = torch.tensor(
            int(self.data.get("stage_token", self.data.get("substage_id", np.zeros((self.length,), dtype=np.int64)))[idx]),
            dtype=torch.long,
        )
        contact_phase = torch.tensor(
            int(self.data.get("contact_state", np.zeros((self.length,), dtype=np.int64))[idx]),
            dtype=torch.long,
        )
        depth_prox = torch.tensor(
            float(self.data.get("depth_proximity", np.zeros((self.length,), dtype=np.float32))[idx]),
            dtype=torch.float32,
        )
        gripper_state = torch.tensor(
            float(self.data.get("gripper_state", np.zeros((self.length,), dtype=np.float32))[idx]),
            dtype=torch.float32,
        )
        return {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "force_history": force_hist,
            "proprio": proprio,
            "planner_base_action_local": planner,
            "candidate_actions_local": candidates,
            "candidate_mask": mask,
            "stage_token": stage_token,
            "contact_phase": contact_phase,
            "depth_proximity": depth_prox,
            "gripper_state": gripper_state,
            "candidate_baseline_index": torch.tensor(int(self.data["candidate_baseline_index"][idx]), dtype=torch.long),
            "candidate_best_index": torch.tensor(int(self.data["candidate_best_index"][idx]), dtype=torch.long),
            "candidate_best_geometry_index": torch.tensor(
                int(self.data.get("best_geometry_candidate_index", self.data["candidate_best_geometry_index"])[idx]),
                dtype=torch.long,
            ),
            "candidate_total_cost": torch.from_numpy(self.data["candidate_total_cost"][idx].astype(np.float32)),
            "candidate_geometry_cost": torch.from_numpy(
                self.data.get("candidate_privileged_geometry_cost", self.data["candidate_geometry_cost"])[idx].astype(np.float32)
            ),
            "candidate_risk_cost": torch.from_numpy(self.data["candidate_risk_cost"][idx].astype(np.float32)),
            "candidate_total_cost_norm": torch.from_numpy(
                self.data.get("candidate_total_cost_norm", self.data["candidate_total_cost"])[idx].astype(np.float32)
            ),
            "candidate_geometry_cost_norm": torch.from_numpy(
                self.data.get("candidate_geometry_cost_norm", self.data.get("candidate_privileged_geometry_cost", self.data["candidate_geometry_cost"]))[idx].astype(np.float32)
            ),
            "candidate_risk_cost_norm": torch.from_numpy(
                self.data.get("candidate_risk_cost_norm", self.data["candidate_risk_cost"])[idx].astype(np.float32)
            ),
            "candidate_best_index_norm": torch.tensor(
                int(self.data.get("candidate_best_index_norm", self.data["candidate_best_index"])[idx]), dtype=torch.long
            ),
            "candidate_best_geometry_index_norm": torch.tensor(
                int(self.data.get("candidate_best_geometry_index_norm", self.data["candidate_best_geometry_index"])[idx]),
                dtype=torch.long,
            ),
            "mode_label": torch.tensor(int(self.mode_labels[idx]), dtype=torch.long),
            "yaw_opportunity_label": torch.tensor(
                float(self.data.get("yaw_opportunity_label", np.zeros((self.length,), dtype=np.float32))[idx]),
                dtype=torch.float32,
            ),
            "yaw_augmentation_applied": torch.tensor(
                float(self.data.get("yaw_augmentation_applied", np.zeros((self.length,), dtype=np.float32))[idx]),
                dtype=torch.float32,
            ),
            "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
        }


def pairwise_rank_loss(pred_scores: torch.Tensor, target_cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Costs are lower-is-better, while model scores are higher-is-better.
    # For a pair (i, j) where cost_i > cost_j, candidate j should outrank i.
    score_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(2)
    cost_diff = target_cost.unsqueeze(2) - target_cost.unsqueeze(1)
    pair_mask = (cost_diff > 1e-6) & mask.unsqueeze(2) & mask.unsqueeze(1)
    if not torch.any(pair_mask):
        return pred_scores.sum() * 0.0
    cost_gap = torch.clamp(cost_diff, min=0.0)
    loss = F.softplus(-torch.clamp(score_diff, min=-20.0, max=20.0)) * cost_gap
    return loss[pair_mask].sum() / torch.clamp(cost_gap[pair_mask].sum(), min=1e-6)


def normalized_value_loss(pred: torch.Tensor, cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = torch.clamp(mask.float().sum(dim=1), min=1.0)
    mean = cost.masked_fill(~mask, 0.0).sum(dim=1) / count
    var = (((cost - mean.unsqueeze(1)) * mask.float()) ** 2).sum(dim=1) / count
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    target = -((cost - mean.unsqueeze(1)) / std.unsqueeze(1)).masked_fill(~mask, 0.0)
    return F.smooth_l1_loss(pred[mask], target[mask]) if torch.any(mask) else pred.sum() * 0.0


def balanced_mode_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 5) -> dict[str, float]:
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    out: dict[str, float] = {"mode_acc": float(np.mean(yt == yp)) if yt.size else 0.0}
    recalls = []
    for i in range(num_classes):
        m = yt == i
        rec = float(np.mean(yp[m] == i)) if np.any(m) else 0.0
        out[f"mode_recall_{INDEX_TO_MODE[i]}"] = rec
        recalls.append(rec)
    out["mode_balanced_acc"] = float(np.mean(recalls))
    return out


def _pick_cost(batch: dict[str, torch.Tensor], raw_key: str, norm_key: str, use_normalized: bool) -> torch.Tensor:
    if use_normalized and norm_key in batch:
        return batch[norm_key]
    return batch[raw_key]


def _masked_argmin(cost: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inf = torch.full_like(cost, 1e9)
    masked = torch.where(mask, cost, inf)
    idx = torch.argmin(masked, dim=1)
    has_any = torch.any(mask, dim=1)
    return idx, has_any


def _subset_eval(model: DepthForceLocalContactPolicy, dataset: PrivilegedGeometryCandidateDataset, indices: np.ndarray, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    if indices.size == 0:
        return {}
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
    return evaluate(model, loader, device, args)


def _group_eval_reports(model: DepthForceLocalContactPolicy, dataset: PrivilegedGeometryCandidateDataset, indices: np.ndarray, args: argparse.Namespace, device: torch.device) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    eps = np.asarray(dataset.data.get("episode_index", np.zeros((dataset.length,), dtype=np.int64)), dtype=np.int64)
    yaw_aug = np.asarray(dataset.data.get("yaw_augmentation_applied", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(dataset.data.get("yaw_opportunity_label", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    groups = {
        "all_rows": indices,
        "original_rows": indices[~yaw_aug[indices]],
        "yaw_augmented_rows": indices[yaw_aug[indices]],
        "yaw_opportunity_rows": indices[yaw_opp[indices]],
        "non_yaw_rows": indices[~yaw_opp[indices]],
    }
    for name, idx in groups.items():
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            out[name] = {"rows": 0}
            continue
        metrics = _subset_eval(model, dataset, idx, args, device)
        metrics["rows"] = int(idx.size)
        out[name] = metrics
    return out


def compute_batch_loss(model: DepthForceLocalContactPolicy, batch: dict[str, torch.Tensor], device: torch.device, args: argparse.Namespace):
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
    use_norm = bool(getattr(args, "use_normalized_costs", False))
    baseline_idx = batch["candidate_baseline_index"].to(device=device)
    best_total_idx = (
        batch["candidate_best_index_norm"].to(device=device)
        if use_norm and "candidate_best_index_norm" in batch
        else batch["candidate_best_index"].to(device=device)
    )
    best_geom_idx = (
        batch["candidate_best_geometry_index_norm"].to(device=device)
        if use_norm and "candidate_best_geometry_index_norm" in batch
        else batch["candidate_best_geometry_index"].to(device=device)
    )
    total_cost = _pick_cost(batch, "candidate_total_cost", "candidate_total_cost_norm", use_norm).to(device=device, dtype=torch.float32)
    geom_cost = _pick_cost(batch, "candidate_geometry_cost", "candidate_geometry_cost_norm", use_norm).to(device=device, dtype=torch.float32)
    risk_cost = _pick_cost(batch, "candidate_risk_cost", "candidate_risk_cost_norm", use_norm).to(device=device, dtype=torch.float32)
    mode_label = batch["mode_label"].to(device=device)
    yaw_opp = batch["yaw_opportunity_label"].to(device=device, dtype=torch.float32) > 0.5
    yaw_aug = batch["yaw_augmentation_applied"].to(device=device, dtype=torch.float32) > 0.5

    out = model(
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
    total_scores = out["candidate_total_value"]
    geom_scores = out["candidate_geometry_value"]
    risk_scores = out["candidate_risk_value"]
    row = torch.arange(total_scores.shape[0], device=device)

    mode_loss = F.cross_entropy(out["mode_logits"], mode_label)
    total_rank = pairwise_rank_loss(total_scores, total_cost, mask)
    geom_rank = pairwise_rank_loss(geom_scores, geom_cost, mask)
    risk_rank = pairwise_rank_loss(risk_scores, risk_cost, mask)
    value_loss = (
        normalized_value_loss(total_scores, total_cost, mask)
        + normalized_value_loss(geom_scores, geom_cost, mask)
        + normalized_value_loss(risk_scores, risk_cost, mask)
    ) / 3.0
    total_ce = F.cross_entropy(total_scores.masked_fill(~mask, -1e9), best_total_idx)
    geom_ce = F.cross_entropy(geom_scores.masked_fill(~mask, -1e9), best_geom_idx)

    yaw_abs = torch.abs(candidates[..., 5])
    yaw_mask = yaw_abs > float(args.keep_yaw_abs)
    no_yaw_mask = ~yaw_mask
    small_yaw_mask = yaw_mask & (yaw_abs <= float(args.small_yaw_abs))
    large_yaw_mask = yaw_abs >= float(args.large_yaw_abs)
    pos_yaw_mask = yaw_mask & (candidates[..., 5] > 0)
    neg_yaw_mask = yaw_mask & (candidates[..., 5] < 0)

    best_yaw_idx, has_yaw = _masked_argmin(geom_cost, yaw_mask & mask)
    best_no_yaw_idx, has_no_yaw = _masked_argmin(geom_cost, no_yaw_mask & mask)
    best_pos_yaw_idx, has_pos_yaw = _masked_argmin(geom_cost, pos_yaw_mask & mask)
    best_neg_yaw_idx, has_neg_yaw = _masked_argmin(geom_cost, neg_yaw_mask & mask)
    best_small_yaw_idx, has_small_yaw = _masked_argmin(geom_cost, small_yaw_mask & mask)
    best_large_yaw_idx, has_large_yaw = _masked_argmin(geom_cost, large_yaw_mask & mask)

    best_total = total_cost[row, best_total_idx]
    baseline_total = total_cost[row, baseline_idx]
    switch_target = ((baseline_total - best_total) > float(args.switch_margin)).float()
    switch_loss = F.binary_cross_entropy_with_logits(out["switch_logit"], switch_target)
    progress_loss = F.smooth_l1_loss(out["progress_delta"], baseline_total - best_total)
    residual_target = candidates[row, best_total_idx] - planner
    residual_loss = F.smooth_l1_loss(out["residual_aux"], residual_target)

    yaw_pair_rows = yaw_opp & has_yaw & has_no_yaw & mask.any(dim=1)
    if torch.any(yaw_pair_rows):
        yaw_pair_loss = F.softplus(-(geom_scores[row, best_yaw_idx] - geom_scores[row, best_no_yaw_idx]))
        yaw_pair_loss = yaw_pair_loss[yaw_pair_rows].mean()
    else:
        yaw_pair_loss = total_scores.sum() * 0.0

    sign_rows = has_pos_yaw & has_neg_yaw & mask.any(dim=1)
    if torch.any(sign_rows):
        pos_better = geom_cost[row, best_pos_yaw_idx] <= geom_cost[row, best_neg_yaw_idx]
        correct_idx = torch.where(pos_better, best_pos_yaw_idx, best_neg_yaw_idx)
        wrong_idx = torch.where(pos_better, best_neg_yaw_idx, best_pos_yaw_idx)
        sign_loss = F.softplus(-(geom_scores[row, correct_idx] - geom_scores[row, wrong_idx]))
        sign_loss = sign_loss[sign_rows].mean()
    else:
        sign_loss = total_scores.sum() * 0.0

    small_large_rows = has_small_yaw & has_large_yaw & mask.any(dim=1)
    if torch.any(small_large_rows):
        small_better = geom_cost[row, best_small_yaw_idx] <= geom_cost[row, best_large_yaw_idx]
        small_idx = torch.where(small_better, best_small_yaw_idx, best_large_yaw_idx)
        large_idx = torch.where(small_better, best_large_yaw_idx, best_small_yaw_idx)
        small_large_loss = F.softplus(-(geom_scores[row, small_idx] - geom_scores[row, large_idx]))
        small_large_loss = small_large_loss[small_large_rows].mean()
    else:
        small_large_loss = total_scores.sum() * 0.0

    loss = (
        args.mode_weight * mode_loss
        + args.total_rank_weight * total_rank
        + args.geometry_rank_weight * geom_rank
        + args.risk_rank_weight * risk_rank
        + args.value_weight * value_loss
        + args.total_ce_weight * total_ce
        + args.geometry_ce_weight * geom_ce
        + args.switch_weight * switch_loss
        + args.progress_weight * progress_loss
        + args.residual_weight * residual_loss
        + args.yaw_pair_weight * yaw_pair_loss
        + args.yaw_sign_weight * sign_loss
        + args.small_over_large_yaw_weight * small_large_loss
    )

    with torch.no_grad():
        total_pred = torch.argmax(total_scores.masked_fill(~mask, -1e9), dim=1)
        geom_pred = torch.argmax(geom_scores.masked_fill(~mask, -1e9), dim=1)
        top3_total = torch.topk(total_scores.masked_fill(~mask, -1e9), k=min(3, total_scores.shape[1]), dim=1).indices
        top5_total = torch.topk(total_scores.masked_fill(~mask, -1e9), k=min(5, total_scores.shape[1]), dim=1).indices
        top3_geom = torch.topk(geom_scores.masked_fill(~mask, -1e9), k=min(3, geom_scores.shape[1]), dim=1).indices
        pred_total_cost = total_cost[row, total_pred]
        base_total_cost = total_cost[row, baseline_idx]
        pred_geom_cost = geom_cost[row, total_pred]
        base_geom_cost = geom_cost[row, baseline_idx]
        pred_risk_cost = risk_cost[row, total_pred]
        base_risk_cost = risk_cost[row, baseline_idx]
        geom_head_cost = geom_cost[row, geom_pred]
        masked_total_cost = total_cost.masked_fill(~mask, -1e9)
        masked_geom_cost = geom_cost.masked_fill(~mask, -1e9)
        worst_total_idx = torch.argmax(masked_total_cost, dim=1)
        worst_geom_idx = torch.argmax(masked_geom_cost, dim=1)
        yaw_selected = torch.abs(candidates[row, total_pred, 5]) > float(args.keep_yaw_abs)
        mode_pred = torch.argmax(out["mode_logits"], dim=1)
        metrics = {
            "loss": float(loss.item()),
            "mode_loss": float(mode_loss.item()),
            "total_top1": float(torch.mean((total_pred == best_total_idx).float()).item()),
            "total_top3": float(torch.mean(torch.any(top3_total == best_total_idx.unsqueeze(1), dim=1).float()).item()),
            "total_top5": float(torch.mean(torch.any(top5_total == best_total_idx.unsqueeze(1), dim=1).float()).item()),
            "geometry_top1": float(torch.mean((geom_pred == best_geom_idx).float()).item()),
            "geometry_top3": float(torch.mean(torch.any(top3_geom == best_geom_idx.unsqueeze(1), dim=1).float()).item()),
            "selected_total_improves_rate": float(torch.mean((pred_total_cost < base_total_cost - 1e-6).float()).item()),
            "selected_geometry_improves_rate": float(torch.mean((pred_geom_cost < base_geom_cost - 1e-6).float()).item()),
            "geometry_head_improves_rate": float(torch.mean((geom_head_cost < base_geom_cost - 1e-6).float()).item()),
            "risk_non_increase_rate": float(torch.mean((pred_risk_cost <= base_risk_cost + 1e-6).float()).item()),
            "total_regret_delta_mean": float(torch.mean(base_total_cost - pred_total_cost).item()),
            "geometry_regret_delta_mean": float(torch.mean(base_geom_cost - pred_geom_cost).item()),
            "risk_delta_mean": float(torch.mean(base_risk_cost - pred_risk_cost).item()),
            "switch_acc": float(torch.mean(((torch.sigmoid(out["switch_logit"]) >= 0.5) == (switch_target > 0.5)).float()).item()),
            "switch_target_rate": float(torch.mean(switch_target).item()),
            "yaw_opportunity_selected_rate": float(torch.mean(yaw_selected[yaw_opp].float()).item()) if torch.any(yaw_opp) else 0.0,
            "yaw_opportunity_count": float(torch.sum(yaw_opp.float()).item()),
            "yaw_augmentation_rate": float(torch.mean(yaw_aug.float()).item()) if torch.any(yaw_aug) else 0.0,
            "yaw_augmentation_count": float(torch.sum(yaw_aug.float()).item()),
            "yaw_pair_loss": float(yaw_pair_loss.item()),
            "yaw_sign_loss": float(sign_loss.item()),
            "small_large_yaw_loss": float(small_large_loss.item()),
            "total_score_best_minus_worst": float(torch.mean(total_scores[row, best_total_idx] - total_scores[row, worst_total_idx]).item()),
            "geometry_score_best_minus_worst": float(torch.mean(geom_scores[row, best_geom_idx] - geom_scores[row, worst_geom_idx]).item()),
            "total_score_best_minus_baseline": float(torch.mean(total_scores[row, best_total_idx] - total_scores[row, baseline_idx]).item()),
            "geometry_score_best_minus_baseline": float(torch.mean(geom_scores[row, best_geom_idx] - geom_scores[row, baseline_idx]).item()),
        }
        metrics.update(balanced_mode_metrics(mode_label, mode_pred, num_classes=len(MODE_TO_INDEX)))
    return loss, metrics


@torch.no_grad()
def evaluate(model: DepthForceLocalContactPolicy, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    count = 0
    for batch in loader:
        _, metrics = compute_batch_loss(model, batch, device, args)
        bsz = int(batch["mode_label"].shape[0])
        count += bsz
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bsz
    return {k: float(v / max(count, 1)) for k, v in agg.items()}


def split_indices(dataset: PrivilegedGeometryCandidateDataset) -> list[tuple[int, np.ndarray, np.ndarray]]:
    eps = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    splits = []
    for heldout in sorted(int(x) for x in np.unique(eps)):
        train_idx = np.where(eps != heldout)[0]
        val_idx = np.where(eps == heldout)[0]
        if train_idx.size and val_idx.size:
            splits.append((heldout, train_idx, val_idx))
    return splits


def run_split(dataset: PrivilegedGeometryCandidateDataset, train_idx: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace, device: torch.device) -> dict:
    model = DepthForceLocalContactPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
    best_score = None
    best_state = None
    best_metrics = {}
    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_batch_loss(model, batch, device, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        metrics = evaluate(model, val_loader, device, args)
        score = (
            metrics.get("selected_total_improves_rate", 0.0)
            + 0.5 * metrics.get("selected_geometry_improves_rate", 0.0)
            + 0.25 * metrics.get("risk_non_increase_rate", 0.0)
            + 0.25 * metrics.get("mode_balanced_acc", 0.0)
            + max(metrics.get("total_regret_delta_mean", 0.0), 0.0)
        )
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = dict(metrics)
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    train_metrics = evaluate(model, train_loader, device, args)
    val_groups = _group_eval_reports(model, dataset, val_idx, args, device)
    train_groups = _group_eval_reports(model, dataset, train_idx, args, device)
    return {
        "heldout_episode": int(dataset.data["episode_index"][val_idx[0]]),
        "train": train_metrics,
        "val": best_metrics,
        "train_groups": train_groups,
        "val_groups": val_groups,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--switch_margin", type=float, default=0.001)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    ap.add_argument("--mode_weight", type=float, default=0.75)
    ap.add_argument("--total_rank_weight", type=float, default=1.0)
    ap.add_argument("--geometry_rank_weight", type=float, default=1.0)
    ap.add_argument("--risk_rank_weight", type=float, default=0.5)
    ap.add_argument("--value_weight", type=float, default=0.2)
    ap.add_argument("--total_ce_weight", type=float, default=0.5)
    ap.add_argument("--geometry_ce_weight", type=float, default=0.35)
    ap.add_argument("--switch_weight", type=float, default=0.4)
    ap.add_argument("--progress_weight", type=float, default=0.2)
    ap.add_argument("--residual_weight", type=float, default=0.05)
    ap.add_argument("--yaw_pair_weight", type=float, default=0.15)
    ap.add_argument("--yaw_sign_weight", type=float, default=0.15)
    ap.add_argument("--small_over_large_yaw_weight", type=float, default=0.15)
    ap.add_argument("--small_yaw_abs", type=float, default=0.05)
    ap.add_argument("--large_yaw_abs", type=float, default=0.09)
    ap.add_argument("--use_normalized_costs", action="store_true", default=False)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = PrivilegedGeometryCandidateDataset(args.dataset_npz)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    split_reports = []
    for _, train_idx, val_idx in split_indices(dataset):
        split_reports.append(run_split(dataset, train_idx, val_idx, args, device))

    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    final_model = DepthForceLocalContactPolicy().to(device)
    opt = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_metrics = {}
    best_score = None
    for _ in range(args.epochs):
        final_model.train()
        for batch in full_loader:
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_batch_loss(final_model, batch, device, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), args.grad_clip)
            opt.step()
        metrics = evaluate(final_model, full_loader, device, args)
        score = (
            metrics.get("selected_total_improves_rate", 0.0)
            + 0.5 * metrics.get("selected_geometry_improves_rate", 0.0)
            + 0.25 * metrics.get("risk_non_increase_rate", 0.0)
            + 0.25 * metrics.get("mode_balanced_acc", 0.0)
            + max(metrics.get("total_regret_delta_mean", 0.0), 0.0)
        )
        if best_score is None or score > best_score:
            best_score = score
            best_metrics = dict(metrics)
            best_state = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
    assert best_state is not None
    final_model.load_state_dict(best_state, strict=True)
    ckpt_path = out_dir / "depth_force_mode_first_geometry_risk_policy.pt"
    torch.save({"model_state_dict": final_model.state_dict(), "args": vars(args), "mode_to_index": MODE_TO_INDEX}, ckpt_path)

    report = {
        "dataset_npz": str(args.dataset_npz),
        "mode_to_index": MODE_TO_INDEX,
        "split_reports": split_reports,
        "final_train_metrics": best_metrics,
        "checkpoint_path": str(ckpt_path),
    }
    (out_dir / "mode_first_geometry_risk_policy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
