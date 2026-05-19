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


class DepthForceCandidateCostDataset(Dataset):
    def __init__(self, npz_path: str):
        raw = np.load(npz_path, allow_pickle=False)
        self.data = {k: np.asarray(raw[k]) for k in raw.files}
        self.length = int(self.data["candidate_actions_local"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        force_hist = torch.from_numpy(
            np.asarray(
                self.data.get("force_history_normalized", self.data.get("force_history", self.data.get("ft_hist")))[idx],
                dtype=np.float32,
            )
        )
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
        stage_token = torch.tensor(int(self.data.get("stage_token", self.data.get("substage_id", np.zeros((self.length,), dtype=np.int64)))[idx]), dtype=torch.long)
        contact_phase = torch.tensor(int(self.data.get("contact_state", np.zeros((self.length,), dtype=np.int64))[idx]), dtype=torch.long)
        depth_prox = torch.tensor(float(self.data.get("depth_proximity", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32)
        gripper_state = torch.tensor(float(self.data.get("gripper_state", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32)
        contact_risk = torch.tensor(int(self.data.get("contact_risk", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.long)
        baseline_idx = torch.tensor(int(self.data["candidate_baseline_index"][idx]), dtype=torch.long)
        best_idx = torch.tensor(int(self.data["candidate_best_index"][idx]), dtype=torch.long)
        total_cost = torch.from_numpy(self.data["candidate_total_cost"][idx].astype(np.float32))
        geom_cost = torch.from_numpy(self.data["candidate_geometry_cost"][idx].astype(np.float32))
        risk_cost = torch.from_numpy(self.data["candidate_risk_cost"][idx].astype(np.float32))
        safe_target = torch.from_numpy(self.data["safe_target_action_local"][idx].astype(np.float32))
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
            "contact_risk": contact_risk,
            "candidate_baseline_index": baseline_idx,
            "candidate_best_index": best_idx,
            "candidate_total_cost": total_cost,
            "candidate_geometry_cost": geom_cost,
            "candidate_risk_cost": risk_cost,
            "safe_target_action_local": safe_target,
            "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
        }


def pairwise_rank_loss(pred_scores: torch.Tensor, target_cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    score_diff = pred_scores.unsqueeze(2) - pred_scores.unsqueeze(1)
    cost_diff = target_cost.unsqueeze(2) - target_cost.unsqueeze(1)
    pair_mask = (cost_diff > 1e-6) & mask.unsqueeze(2) & mask.unsqueeze(1)
    if not torch.any(pair_mask):
        return pred_scores.sum() * 0.0
    cost_gap = torch.clamp(cost_diff, min=0.0)
    loss = F.softplus(-torch.clamp(score_diff, min=-20.0, max=20.0)) * cost_gap
    denom = torch.clamp(cost_gap[pair_mask].sum(), min=1e-6)
    return loss[pair_mask].sum() / denom


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
    contact_risk = batch["contact_risk"].to(device=device, dtype=torch.long)
    baseline_idx = batch["candidate_baseline_index"].to(device=device)
    best_idx = batch["candidate_best_index"].to(device=device)
    total_cost = batch["candidate_total_cost"].to(device=device, dtype=torch.float32)
    geom_cost = batch["candidate_geometry_cost"].to(device=device, dtype=torch.float32)
    safe_target = batch["safe_target_action_local"].to(device=device, dtype=torch.float32)

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

    scores = out["candidate_value"]
    masked_scores = scores.masked_fill(~mask, -1e9)
    row = torch.arange(scores.shape[0], device=device)

    # Value head: normalized negative total cost.
    cost_mean = total_cost.masked_fill(~mask, 0.0).sum(dim=1) / torch.clamp(mask.float().sum(dim=1), min=1.0)
    cost_std = torch.sqrt(
        torch.clamp(
            (((total_cost - cost_mean.unsqueeze(1)) * mask.float()) ** 2).sum(dim=1) / torch.clamp(mask.float().sum(dim=1), min=1.0),
            min=1e-6,
        )
    )
    value_target = -((total_cost - cost_mean.unsqueeze(1)) / cost_std.unsqueeze(1)).masked_fill(~mask, 0.0)
    value_loss = F.smooth_l1_loss(scores[mask], value_target[mask]) if torch.any(mask) else scores.sum() * 0.0
    rank_loss = pairwise_rank_loss(scores, total_cost, mask)
    best_ce_loss = F.cross_entropy(masked_scores, best_idx)

    best_cost = total_cost[row, best_idx]
    baseline_cost = total_cost[row, baseline_idx]
    improvement = baseline_cost - best_cost
    switch_target = (improvement > float(args.switch_margin)).float()
    switch_loss = F.binary_cross_entropy_with_logits(out["switch_logit"], switch_target)

    contact_loss = F.cross_entropy(out["contact_risk_logits"], contact_risk)
    progress_loss = F.smooth_l1_loss(out["progress_delta"], improvement)
    residual_target = safe_target - planner
    residual_loss = F.smooth_l1_loss(out["residual_aux"], residual_target)

    loss = (
        args.rank_weight * rank_loss
        + args.value_weight * value_loss
        + args.best_ce_weight * best_ce_loss
        + args.switch_weight * switch_loss
        + args.contact_weight * contact_loss
        + args.progress_weight * progress_loss
        + args.residual_weight * residual_loss
    )

    with torch.no_grad():
        pred_idx = torch.argmax(masked_scores, dim=1)
        topk = torch.topk(masked_scores, k=min(3, masked_scores.shape[1]), dim=1).indices
        pred_cost = total_cost[row, pred_idx]
        pred_better = pred_cost < baseline_cost - 1e-6
        pred_worse = pred_cost > baseline_cost + 1e-6
        metrics = {
            "loss": float(loss.item()),
            "top1_recall": float(torch.mean((pred_idx == best_idx).float()).item()),
            "top3_recall": float(torch.mean(torch.any(topk == best_idx.unsqueeze(1), dim=1).float()).item()),
            "selected_improves_rate": float(torch.mean(pred_better.float()).item()),
            "negative_selection_rate": float(torch.mean(pred_worse.float()).item()),
            "regret_delta_mean": float(torch.mean(pred_cost - baseline_cost).item()),
            "baseline_cost_mean": float(torch.mean(baseline_cost).item()),
            "best_cost_mean": float(torch.mean(best_cost).item()),
            "switch_acc": float(torch.mean(((torch.sigmoid(out["switch_logit"]) >= 0.5) == (switch_target > 0.5)).float()).item()),
            "switch_prob_mean": float(torch.mean(torch.sigmoid(out["switch_logit"]).detach()).item()),
            "switch_target_rate": float(torch.mean(switch_target).item()),
            "contact_ce": float(contact_loss.item()),
            "progress_loss": float(progress_loss.item()),
            "residual_loss": float(residual_loss.item()),
        }
    return loss, metrics


@torch.no_grad()
def evaluate(model: DepthForceLocalContactPolicy, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    count = 0
    for batch in loader:
        _, metrics = compute_batch_loss(model, batch, device, args)
        bsz = int(batch["candidate_best_index"].shape[0])
        count += bsz
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bsz
    if count <= 0:
        return {k: 0.0 for k in ("loss", "top1_recall", "top3_recall", "selected_improves_rate", "negative_selection_rate", "regret_delta_mean")}
    return {k: float(v / count) for k, v in agg.items()}


def split_indices(dataset: DepthForceCandidateCostDataset) -> list[tuple[int, np.ndarray, np.ndarray]]:
    eps = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = sorted(int(x) for x in np.unique(eps))
    splits = []
    for heldout in uniq:
        train_idx = np.where(eps != heldout)[0]
        val_idx = np.where(eps == heldout)[0]
        if train_idx.size and val_idx.size:
            splits.append((heldout, train_idx, val_idx))
    return splits


def run_split(dataset: DepthForceCandidateCostDataset, train_idx: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace, device: torch.device) -> dict:
    model = DepthForceLocalContactPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
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
        score = metrics["selected_improves_rate"] + 0.35 * metrics["top1_recall"] + 0.15 * metrics["top3_recall"] + 0.10 * max(metrics["regret_delta_mean"], 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = dict(metrics)
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    train_metrics = evaluate(model, train_loader, device, args)
    return {
        "heldout_episode": int(dataset.data["episode_index"][val_idx[0]]),
        "train": train_metrics,
        "val": best_metrics,
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
    ap.add_argument("--rank_weight", type=float, default=1.0)
    ap.add_argument("--value_weight", type=float, default=0.25)
    ap.add_argument("--best_ce_weight", type=float, default=0.75)
    ap.add_argument("--switch_weight", type=float, default=0.5)
    ap.add_argument("--contact_weight", type=float, default=0.5)
    ap.add_argument("--progress_weight", type=float, default=0.25)
    ap.add_argument("--residual_weight", type=float, default=0.25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = DepthForceCandidateCostDataset(args.dataset_npz)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    split_reports = []
    for heldout, train_idx, val_idx in split_indices(dataset):
        split_reports.append(run_split(dataset, train_idx, val_idx, args, device))

    # Fit a final model on the full dataset for shadow use.
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    final_model = DepthForceLocalContactPolicy().to(device)
    opt = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = None
    best_state = None
    best_metrics = {}
    for _ in range(args.epochs):
        final_model.train()
        for batch in full_loader:
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_batch_loss(final_model, batch, device, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), args.grad_clip)
            opt.step()
        metrics = evaluate(final_model, full_loader, device, args)
        score = metrics["selected_improves_rate"] + 0.35 * metrics["top1_recall"] + 0.15 * metrics["top3_recall"] + 0.10 * max(metrics["regret_delta_mean"], 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
            best_metrics = dict(metrics)
    assert best_state is not None
    final_model.load_state_dict(best_state, strict=True)
    ckpt_path = out_dir / "depth_force_local_contact_policy.pt"
    torch.save({"model_state_dict": final_model.state_dict(), "args": vars(args)}, ckpt_path)

    report = {
        "dataset_npz": str(args.dataset_npz),
        "split_reports": split_reports,
        "final_train_metrics": best_metrics,
        "checkpoint_path": str(ckpt_path),
    }
    (out_dir / "depth_force_local_contact_policy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
