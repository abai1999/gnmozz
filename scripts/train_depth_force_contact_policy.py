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


class DepthForceContactDataset(Dataset):
    def __init__(self, npz_path: str | Path) -> None:
        raw = np.load(npz_path, allow_pickle=False)
        self.data = {k: np.asarray(raw[k]) for k in raw.files}
        self.length = int(self.data["best_candidate_index"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        return {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "force_history": torch.from_numpy(self.data["force_history"][idx].astype(np.float32)),
            "proprio": torch.from_numpy(self.data["proprio"][idx].astype(np.float32)),
            "planner_base_action_local": torch.from_numpy(self.data["planner_base_action_local"][idx].astype(np.float32)),
            "candidate_actions_local": torch.from_numpy(self.data["candidate_actions_local"][idx].astype(np.float32)),
            "candidate_mask": torch.from_numpy(self.data["candidate_mask"][idx].astype(np.float32)),
            "candidate_value": torch.from_numpy(self.data["candidate_value"][idx].astype(np.float32)),
            "best_candidate_index": torch.tensor(int(self.data["best_candidate_index"][idx]), dtype=torch.long),
            "baseline_candidate_index": torch.tensor(int(self.data["baseline_candidate_index"][idx]), dtype=torch.long),
            "switch_target": torch.tensor(float(self.data["switch_target"][idx]), dtype=torch.float32),
            "contact_risk": torch.tensor(int(self.data["contact_risk"][idx]), dtype=torch.long),
            "progress_target": torch.tensor(float(self.data["progress_target"][idx]), dtype=torch.float32),
            "residual_aux": torch.from_numpy(self.data["residual_aux"][idx].astype(np.float32)),
            "depth_proximity": torch.tensor(float(self.data["depth_proximity"][idx]), dtype=torch.float32),
            "gripper_state": torch.tensor(float(self.data["gripper_state"][idx]), dtype=torch.float32),
            "stage_token": torch.tensor(int(self.data["stage_token"][idx]), dtype=torch.long),
            "episode_index": torch.tensor(int(self.data["episode_index"][idx]), dtype=torch.long),
        }


def split_by_episode(dataset: DepthForceContactDataset, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    episodes = sorted(set(int(x) for x in dataset.data["episode_index"].tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * val_fraction))) if len(episodes) > 1 else 1
    val_eps = set(episodes[:n_val])
    train_idx = [i for i, ep in enumerate(dataset.data["episode_index"].tolist()) if int(ep) not in val_eps]
    val_idx = [i for i, ep in enumerate(dataset.data["episode_index"].tolist()) if int(ep) in val_eps]
    if not train_idx:
        train_idx = val_idx
    return train_idx, val_idx


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward_model(model: DepthForceLocalContactPolicy, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        front_rgb=batch["front_rgb"],
        wrist_rgb=batch["wrist_rgb"],
        wrist_depth=batch["wrist_depth"],
        force_history=batch["force_history"],
        proprio=batch["proprio"],
        planner_base_action_local=batch["planner_base_action_local"],
        candidate_actions_local=batch["candidate_actions_local"],
        candidate_mask=batch["candidate_mask"],
        stage_token=batch["stage_token"],
        contact_phase=batch["contact_risk"].clamp(0, 3),
        depth_proximity=batch["depth_proximity"],
        gripper_state=batch["gripper_state"],
    )


def compute_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args) -> tuple[torch.Tensor, dict[str, float]]:
    values = out["candidate_value"]
    labels = batch["candidate_value"]
    mask = batch["candidate_mask"] > 0.5
    row = torch.arange(values.shape[0], device=values.device)
    best = batch["best_candidate_index"]
    baseline = batch["baseline_candidate_index"]

    ce = F.cross_entropy(values, best)
    best_pred = values[row, best]
    baseline_pred = values[row, baseline]
    rank = F.softplus(args.rank_margin - (best_pred - baseline_pred)).mean()

    valid_labels = torch.isfinite(labels) & mask
    centered_labels = labels - labels.masked_fill(~valid_labels, 0.0).sum(dim=1, keepdim=True) / valid_labels.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    centered_labels = torch.nan_to_num(centered_labels, nan=0.0).clamp(-20.0, 20.0)
    value = F.smooth_l1_loss(values[valid_labels], centered_labels[valid_labels]) if valid_labels.any() else values.sum() * 0.0

    switch = F.binary_cross_entropy_with_logits(out["switch_logit"], batch["switch_target"])
    contact = F.cross_entropy(out["contact_risk_logits"], batch["contact_risk"].clamp(0, 3))
    progress = F.smooth_l1_loss(out["progress_delta"], batch["progress_target"].clamp(-20.0, 20.0))
    residual = F.smooth_l1_loss(out["residual_aux"], batch["residual_aux"])

    loss = (
        args.best_ce_weight * ce
        + args.rank_weight * rank
        + args.value_weight * value
        + args.switch_weight * switch
        + args.contact_weight * contact
        + args.progress_weight * progress
        + args.residual_aux_weight * residual
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "rank": float(rank.detach().cpu()),
        "value": float(value.detach().cpu()),
        "switch": float(switch.detach().cpu()),
        "contact": float(contact.detach().cpu()),
        "progress": float(progress.detach().cpu()),
        "residual": float(residual.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: DepthForceLocalContactPolicy, loader: DataLoader, device: torch.device, switch_threshold: float) -> dict[str, float]:
    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    selected_improve = 0
    selected_worse = 0
    regret_sum = 0.0
    switch_correct = 0
    contact_correct = 0
    yaw_selected = 0
    oracle_yaw = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = forward_model(model, batch)
        values = out["candidate_value"]
        pred = torch.argmax(values, dim=1)
        topk = torch.topk(values, k=min(3, values.shape[1]), dim=1).indices
        best = batch["best_candidate_index"]
        baseline = batch["baseline_candidate_index"]
        labels = batch["candidate_value"]
        row = torch.arange(values.shape[0], device=device)
        pred_score = labels[row, pred]
        baseline_score = labels[row, baseline]
        best_score = labels[row, best]
        total += int(values.shape[0])
        top1 += int((pred == best).sum().item())
        top3 += int((topk == best[:, None]).any(dim=1).sum().item())
        selected_improve += int((pred_score > baseline_score).sum().item())
        selected_worse += int((pred_score < baseline_score).sum().item())
        # Positive means the selected candidate is better than the baseline
        # under the teacher/cost label.
        regret_sum += float((pred_score - baseline_score).sum().item())
        switch_pred = (out["switch_prob"] >= switch_threshold).float()
        switch_correct += int((switch_pred == batch["switch_target"]).sum().item())
        contact_correct += int((torch.argmax(out["contact_risk_logits"], dim=1) == batch["contact_risk"]).sum().item())
        actions = batch["candidate_actions_local"]
        yaw_selected += int((torch.abs(actions[row, pred, 5]) > 1e-4).sum().item())
        oracle_yaw += int((torch.abs(actions[row, best, 5]) > 1e-4).sum().item())
    denom = max(total, 1)
    return {
        "rows": int(total),
        "top1_recall": top1 / denom,
        "top3_recall": top3 / denom,
        "selected_improves_rate": selected_improve / denom,
        "negative_selection_rate": selected_worse / denom,
        "regret_delta_mean_baseline_minus_pred": regret_sum / denom,
        "switch_acc": switch_correct / denom,
        "contact_risk_acc": contact_correct / denom,
        "yaw_candidate_selection_rate": yaw_selected / denom,
        "oracle_yaw_candidate_rate": oracle_yaw / denom,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--switch_threshold", type=float, default=0.65)
    ap.add_argument("--rank_margin", type=float, default=0.5)
    ap.add_argument("--best_ce_weight", type=float, default=0.2)
    ap.add_argument("--rank_weight", type=float, default=1.0)
    ap.add_argument("--value_weight", type=float, default=0.5)
    ap.add_argument("--switch_weight", type=float, default=0.5)
    ap.add_argument("--contact_weight", type=float, default=0.3)
    ap.add_argument("--progress_weight", type=float, default=0.2)
    ap.add_argument("--residual_aux_weight", type=float, default=0.05)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DepthForceContactDataset(args.dataset_npz)
    train_idx, val_idx = split_by_episode(dataset, args.val_fraction, args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=2)

    proprio_dim = int(dataset.data["proprio"].shape[1])
    model = DepthForceLocalContactPolicy(proprio_input_dim=proprio_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1e9
    history = []
    best_path = out_dir / "depth_force_contact_policy_best.pt"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            out = forward_model(model, batch)
            loss, parts = compute_loss(out, batch, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(parts)
        val_metrics = evaluate(model, val_loader, device, args.switch_threshold)
        metric = val_metrics["selected_improves_rate"] + val_metrics["regret_delta_mean_baseline_minus_pred"] * 0.01
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean([x["loss"] for x in losses])) if losses else 0.0,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row))
        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": {"proprio_input_dim": proprio_dim},
                    "candidate_bank": dataset.data["candidate_actions_local"][0].astype(np.float32),
                    "switch_threshold": float(args.switch_threshold),
                    "history": history,
                },
                best_path,
            )

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    final_val = evaluate(model, val_loader, device, args.switch_threshold)
    train_eval = evaluate(model, DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=False), device, args.switch_threshold)
    report = {
        "dataset": str(args.dataset_npz),
        "rows": int(len(dataset)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "best_ckpt": str(best_path),
        "final_val": final_val,
        "train_eval": train_eval,
        "history": history,
    }
    (out_dir / "depth_force_contact_policy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
