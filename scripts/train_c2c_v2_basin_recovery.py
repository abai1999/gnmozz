#!/usr/bin/env python3
"""Train first-pass basin recovery state/pullback heads.

The first implementation is deliberately feature-based.  It does not replace
RGBD localizers; it trains the new recovery semantics: evidence classification,
basin labeling, reacquire gating, and bounded contraction proposals.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.basin_recovery import (  # noqa: E402
    BasinPullbackPolicyNet,
    BasinStateEstimatorNet,
    basin_recovery_feature_vector,
)
from prismatic.robot.coarse2contact_v2.datasets import read_jsonl  # noqa: E402
from prismatic.robot.coarse2contact_v2.recovery_audit import recovery_error_norm  # noqa: E402


EVIDENCE_TO_ID = {"visual_observable": 0, "partial_observable": 1, "prior_only": 2}
BASIN_TO_ID = {"outside": 0, "near_grasp": 1, "close_ready": 2}


class BasinRecoveryDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], *, feature_dim: int | None = None) -> None:
        self.records = records
        max_dim = max((len(r.get("state_feature_vector") or basin_recovery_feature_vector(r)) for r in records), default=1)
        self.feature_dim = int(feature_dim or max_dim)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        feat = list(r.get("state_feature_vector") or basin_recovery_feature_vector(r))
        feat = (feat + [0.0] * self.feature_dim)[: self.feature_dim]
        target = torch.tensor(
            [
                float(r.get("initial_error_dx", r.get("recovery_target_dx", 0.0))),
                float(r.get("initial_error_dy", r.get("recovery_target_dy", 0.0))),
                float(r.get("initial_error_dyaw", r.get("recovery_target_dyaw", 0.0))),
            ],
            dtype=torch.float32,
        )
        evidence = EVIDENCE_TO_ID.get(str(r.get("visual_evidence_class", "prior_only")), 2)
        basin = BASIN_TO_ID.get(str(r.get("basin_label", "outside")), 0)
        pullback_allowed = float(bool(r.get("pullback_allowed", False)))
        return {
            "features": torch.tensor(feat, dtype=torch.float32),
            "evidence": torch.tensor(evidence, dtype=torch.long),
            "basin": torch.tensor(basin, dtype=torch.long),
            "reacquire_needed": torch.tensor(float(bool(r.get("reacquire_needed", evidence != 0))), dtype=torch.float32),
            "pullback_allowed": torch.tensor(pullback_allowed, dtype=torch.float32),
            "target_error": target,
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]}


def _split(records: list[dict[str, Any]], *, val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(records)
    rng.shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_fraction))) if len(rows) > 1 else 0
    return rows[n_val:], rows[:n_val]


def _policy_targets(target_error: torch.Tensor, *, xy_step: float, yaw_step: float) -> torch.Tensor:
    xy = target_error[:, :2]
    xy_norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    xy_scale = torch.clamp(torch.tensor(float(xy_step), device=target_error.device) / xy_norm, max=0.35)
    xy_target = xy * xy_scale
    yaw_target = torch.clamp(target_error[:, 2:3] * 0.35, -float(yaw_step), float(yaw_step))
    return torch.cat([xy_target, yaw_target], dim=-1)


def _contraction_loss(target_error: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, *, steps: int = 4) -> torch.Tensor:
    if float(mask.sum().detach().cpu()) <= 0.0:
        return pred.sum() * 0.0
    err = target_error
    prev_norm = torch.linalg.norm(torch.stack([err[:, 0], err[:, 1], 0.04 * err[:, 2]], dim=-1), dim=-1)
    losses = []
    for _ in range(int(steps)):
        err = err - pred
        norm = torch.linalg.norm(torch.stack([err[:, 0], err[:, 1], 0.04 * err[:, 2]], dim=-1), dim=-1)
        losses.append(F.relu(norm - prev_norm + 1.0e-5) * mask)
        prev_norm = norm
    return torch.stack(losses, dim=0).sum(dim=0).sum() / mask.sum().clamp_min(1.0)


def _evaluate(state_net: BasinStateEstimatorNet, policy_net: BasinPullbackPolicyNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    state_net.eval()
    policy_net.eval()
    evidence_ok = []
    basin_ok = []
    gains = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target_error"].to(device)
            st = state_net(features)
            pol = policy_net(features)
            evidence_ok.extend((torch.argmax(st["visual_evidence_logits"], dim=-1) == batch["evidence"].to(device)).detach().cpu().numpy().tolist())
            basin_ok.extend((torch.argmax(st["basin_logits"], dim=-1) == batch["basin"].to(device)).detach().cpu().numpy().tolist())
            pred = torch.stack([pol["dx"], pol["dy"], pol["dyaw"]], dim=-1)
            post = target - pred
            pre_norm = torch.linalg.norm(torch.stack([target[:, 0], target[:, 1], 0.04 * target[:, 2]], dim=-1), dim=-1)
            post_norm = torch.linalg.norm(torch.stack([post[:, 0], post[:, 1], 0.04 * post[:, 2]], dim=-1), dim=-1)
            gains.extend((pre_norm - post_norm).detach().cpu().numpy().tolist())
    return {
        "evidence_accuracy": float(np.mean(evidence_ok)) if evidence_ok else 0.0,
        "basin_accuracy": float(np.mean(basin_ok)) if basin_ok else 0.0,
        "single_step_gain_mean": float(np.mean(gains)) if gains else 0.0,
        "single_step_gain_positive_rate": float(np.mean(np.asarray(gains) > 0.0)) if gains else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets_basin_recovery/basin_recovery_dataset_v1.jsonl"))
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/basin_recovery_v1"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--val_fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_xy_step", type=float, default=0.003)
    ap.add_argument("--max_yaw_step", type=float, default=0.05)
    args = ap.parse_args()

    rows = read_jsonl(args.dataset)
    if len(rows) < 2:
        raise RuntimeError("Need at least two basin recovery records")
    train_rows, val_rows = _split(rows, val_fraction=float(args.val_fraction), seed=int(args.seed))
    feature_dim = max(len(r.get("state_feature_vector") or basin_recovery_feature_vector(r)) for r in rows)
    train_ds = BasinRecoveryDataset(train_rows, feature_dim=feature_dim)
    val_ds = BasinRecoveryDataset(val_rows, feature_dim=feature_dim)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=_collate)

    device = torch.device(args.device)
    state_net = BasinStateEstimatorNet(feature_dim=feature_dim).to(device)
    policy_net = BasinPullbackPolicyNet(feature_dim=feature_dim, max_xy_step=float(args.max_xy_step), max_yaw_step=float(args.max_yaw_step)).to(device)
    opt = torch.optim.AdamW(list(state_net.parameters()) + list(policy_net.parameters()), lr=float(args.lr), weight_decay=1.0e-4)

    best_metric = -1.0e9
    reports = []
    for epoch in range(int(args.epochs)):
        state_net.train()
        policy_net.train()
        train_losses = []
        for batch in train_loader:
            features = batch["features"].to(device)
            evidence = batch["evidence"].to(device)
            basin = batch["basin"].to(device)
            reacquire = batch["reacquire_needed"].to(device)
            pullback = batch["pullback_allowed"].to(device)
            target_error = batch["target_error"].to(device)

            st = state_net(features)
            pol = policy_net(features)
            pred = torch.stack([pol["dx"], pol["dy"], pol["dyaw"]], dim=-1)
            pullback_target = _policy_targets(target_error, xy_step=float(args.max_xy_step), yaw_step=float(args.max_yaw_step))
            state_loss = (
                F.cross_entropy(st["visual_evidence_logits"], evidence)
                + F.cross_entropy(st["basin_logits"], basin)
                + F.binary_cross_entropy_with_logits(st["reacquire_needed_logit"], reacquire)
            )
            regression = F.smooth_l1_loss(pred, pullback_target, reduction="none").sum(dim=-1)
            regression = (regression * pullback).sum() / pullback.sum().clamp_min(1.0)
            abstain = (pred[:, :2].abs().sum(dim=-1) + 0.04 * pred[:, 2].abs()) * (1.0 - pullback)
            contraction = _contraction_loss(target_error, pred, pullback)
            loss = state_loss + regression + 0.5 * contraction + 0.25 * abstain.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))

        metrics = _evaluate(state_net, policy_net, val_loader, device)
        metrics["epoch"] = int(epoch)
        metrics["train_loss_mean"] = float(np.mean(train_losses)) if train_losses else 0.0
        reports.append(metrics)
        metric = metrics["evidence_accuracy"] + metrics["basin_accuracy"] + 5.0 * metrics["single_step_gain_mean"]
        if metric > best_metric:
            best_metric = float(metric)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_type": "basin_recovery",
                    "state_model_state_dict": state_net.state_dict(),
                    "policy_model_state_dict": policy_net.state_dict(),
                    "config": {
                        "feature_dim": int(feature_dim),
                        "max_xy_step": float(args.max_xy_step),
                        "max_yaw_step": float(args.max_yaw_step),
                    },
                    "metrics": metrics,
                    "evidence_to_id": EVIDENCE_TO_ID,
                    "basin_to_id": BASIN_TO_ID,
                    "uses_privileged_label": True,
                    "uses_privileged_runtime": False,
                },
                args.output_dir / "best.pt",
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": str(args.dataset),
        "num_train": int(len(train_ds)),
        "num_val": int(len(val_ds)),
        "feature_dim": int(feature_dim),
        "best_metric": float(best_metric),
        "epochs": reports,
        "uses_privileged_label": True,
        "uses_privileged_runtime": False,
    }
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output_dir / "best.pt")
    print(args.output_dir / "train_report.json")


if __name__ == "__main__":
    main()
