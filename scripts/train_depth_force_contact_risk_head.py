#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HEADS = (
    "contact_label",
    "force_spike_label",
    "jam_label",
    "motion_stall_label",
    "kinematic_invalid_label",
    "action_range_invalid_label",
    "near_depth_label",
)


class MultiHeadRiskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in HEADS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.trunk(x)
        return {name: head(state).squeeze(-1) for name, head in self.heads.items()}


def binary_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_prob = 1.0 / (1.0 + np.exp(-logits))
    y_pred = y_prob >= 0.5
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    tn = float(np.sum(~y_true & ~y_pred))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    tpr = recall
    tnr = tn / max(tn + fp, 1.0)
    balanced_acc = 0.5 * (tpr + tnr)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "positive_rate_pred": float(np.mean(y_pred)),
        "positive_rate_true": float(np.mean(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def make_base_features(data: dict[str, np.ndarray]) -> np.ndarray:
    n = int(np.asarray(data["episode_index"]).shape[0])
    depth_prox = np.asarray(data.get("depth_proximity", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1)
    depth_prox = np.nan_to_num(depth_prox, nan=1.0, posinf=1.0, neginf=0.0)
    gripper = np.asarray(data.get("gripper_state", data.get("rollout_gripper_open", np.ones((n,), dtype=np.float32))), dtype=np.float32).reshape(n, 1)
    stage = np.asarray(data.get("stage_token", data.get("substage_id", np.zeros((n,), dtype=np.float32))), dtype=np.float32).reshape(n, 1)
    phase = np.asarray(data.get("phase_id", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1)
    near = np.asarray(data.get("near_depth_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1)
    return np.concatenate([depth_prox, gripper, stage, phase, near], axis=1).astype(np.float32)


def build_features(data: dict[str, np.ndarray], feature_mode: str) -> np.ndarray:
    base = make_base_features(data)
    n = base.shape[0]
    depth = np.stack(
        [
            np.asarray(data.get("wrist_depth_median", np.zeros((n,), dtype=np.float32)), dtype=np.float32),
            np.asarray(data.get("wrist_valid_depth_ratio", np.zeros((n,), dtype=np.float32)), dtype=np.float32),
            np.asarray(data.get("wrist_depth_near_fraction", np.zeros((n,), dtype=np.float32)), dtype=np.float32),
            np.asarray(data.get("is_occluded", np.zeros((n,), dtype=np.float32)), dtype=np.float32),
            np.asarray(data.get("is_low_visibility", np.zeros((n,), dtype=np.float32)), dtype=np.float32),
        ],
        axis=1,
    )
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    force_hist = np.asarray(data.get("force_history", data.get("ft_hist", np.zeros((n, 32, 6), dtype=np.float32))), dtype=np.float32)
    if force_hist.ndim == 2:
        force_hist = force_hist[:, None, :]
    force_last = np.asarray(data.get("gripper_touch_forces", force_hist[:, -1, :]), dtype=np.float32)
    force_prev = force_hist[:, -2, :6] if force_hist.shape[1] >= 2 else np.zeros_like(force_last[:, :6])
    force_delta = force_last[:, :6] - force_prev[:, :6]
    force_stats = np.concatenate(
        [
            force_last[:, :6],
            force_delta[:, :6],
            np.mean(force_hist[:, :, :6], axis=1),
            np.std(force_hist[:, :, :6], axis=1),
            np.max(np.abs(force_hist[:, :, :6]), axis=1),
            np.linalg.norm(force_last[:, :3], axis=1, keepdims=True),
            np.linalg.norm(force_last[:, 3:6], axis=1, keepdims=True),
            np.linalg.norm(force_delta[:, :3], axis=1, keepdims=True),
            np.linalg.norm(force_delta[:, 3:6], axis=1, keepdims=True),
        ],
        axis=1,
    )

    proprio = np.asarray(data.get("proprio", np.zeros((n, 15), dtype=np.float32)), dtype=np.float32)
    planner = np.asarray(data.get("planner_base_action_local", data.get("planner_base_action_local_raw", np.zeros((n, 6), dtype=np.float32))), dtype=np.float32)[:, :6]
    executed = np.asarray(data.get("executed_action_local", planner), dtype=np.float32)[:, :6]
    action_stats = np.concatenate(
        [
            np.linalg.norm(planner[:, :3], axis=1, keepdims=True),
            np.linalg.norm(planner[:, 3:6], axis=1, keepdims=True),
            np.linalg.norm(executed[:, :3], axis=1, keepdims=True),
            np.linalg.norm(executed[:, 3:6], axis=1, keepdims=True),
            np.asarray(data.get("planner_close_intent", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1),
            np.asarray(data.get("abs_gripper_cmd", np.ones((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1),
        ],
        axis=1,
    )
    motion_stats = np.stack(
        [
            np.asarray(data.get("motion_delta_pos", np.zeros((n,), dtype=np.float32)), dtype=np.float32),
            np.asarray(data.get("motion_delta_rot_deg", np.zeros((n,), dtype=np.float32)), dtype=np.float32),
        ],
        axis=1,
    )
    motion_stats = np.nan_to_num(motion_stats, nan=0.0, posinf=0.0, neginf=0.0)

    pieces = [base]
    if feature_mode in ("depth_only", "depth_force", "depth_force_proprio_action"):
        pieces.append(depth)
    if feature_mode in ("force_only", "depth_force", "depth_force_proprio_action"):
        pieces.append(force_stats)
    if feature_mode == "depth_force_proprio_action":
        pieces.append(proprio)
        pieces.append(action_stats)
        pieces.append(motion_stats)
    x = np.concatenate(pieces, axis=1).astype(np.float32)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    return ((np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def train_one_split(
    x: np.ndarray,
    labels: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
) -> dict:
    device = torch.device(args.device)
    model = MultiHeadRiskMLP(x.shape[1], hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = torch.from_numpy(x[train_idx]).float()
    x_val = torch.from_numpy(x[val_idx]).float()
    y_train = {name: torch.from_numpy(labels[name][train_idx].astype(np.float32)).float() for name in HEADS}
    y_val = {name: labels[name][val_idx].astype(np.float32) for name in HEADS}

    pos_weights = {}
    for name in HEADS:
        pos = float(np.sum(labels[name][train_idx] > 0.5))
        neg = float(np.sum(labels[name][train_idx] <= 0.5))
        pos_weights[name] = torch.tensor(neg / max(pos, 1.0), device=device, dtype=torch.float32)

    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0])
        for start in range(0, perm.numel(), args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            xb = x_train[batch_idx].to(device)
            pred = model(xb)
            batch_loss = 0.0
            for name in HEADS:
                target = y_train[name][batch_idx].to(device)
                batch_loss = batch_loss + F.binary_cross_entropy_with_logits(
                    pred[name],
                    target,
                    pos_weight=pos_weights[name],
                )
            batch_loss = batch_loss / float(len(HEADS))
            opt.zero_grad(set_to_none=True)
            batch_loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_val.to(device))
    report = {"rows": int(val_idx.size)}
    for name in HEADS:
        logits_np = logits[name].cpu().numpy()
        y_true = y_val[name]
        metrics = binary_metrics(y_true, logits_np)
        report[name] = metrics
    report["mean_head_f1"] = float(np.mean([report[name]["f1"] for name in HEADS]))
    report["mean_head_balanced_acc"] = float(np.mean([report[name]["balanced_acc"] for name in HEADS]))
    return report


def run_ablation(data: dict[str, np.ndarray], feature_mode: str, args) -> dict:
    x = build_features(data, feature_mode=feature_mode)
    labels = {name: np.asarray(data[name], dtype=np.float32) for name in HEADS}
    episodes = np.asarray(data["episode_index"], dtype=np.int64)
    split_reports = []
    for ep in np.unique(episodes):
        val_idx = np.where(episodes == ep)[0]
        train_idx = np.where(episodes != ep)[0]
        report = train_one_split(x, labels, train_idx, val_idx, args)
        report["heldout_episode"] = int(ep)
        split_reports.append(report)

    summary = {
        "feature_mode": feature_mode,
        "feature_dim": int(x.shape[1]),
        "splits": split_reports,
        "mean_head_f1": float(np.mean([r["mean_head_f1"] for r in split_reports])),
        "mean_head_balanced_acc": float(np.mean([r["mean_head_balanced_acc"] for r in split_reports])),
        "head_means": {
            name: {
                "f1": float(np.mean([r[name]["f1"] for r in split_reports])),
                "balanced_acc": float(np.mean([r[name]["balanced_acc"] for r in split_reports])),
                "precision": float(np.mean([r[name]["precision"] for r in split_reports])),
                "recall": float(np.mean([r[name]["recall"] for r in split_reports])),
                "positive_rate_true": float(np.mean([r[name]["positive_rate_true"] for r in split_reports])),
                "positive_rate_pred": float(np.mean([r[name]["positive_rate_pred"] for r in split_reports])),
            }
            for name in HEADS
        },
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument(
        "--feature_modes",
        type=str,
        default="depth_only,force_only,depth_force,depth_force_proprio_action",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = {k: np.asarray(v) for k, v in np.load(args.dataset_npz, allow_pickle=False).items()}
    modes = [m.strip() for m in args.feature_modes.split(",") if m.strip()]
    reports = {}
    baseline_mode = None
    for mode in modes:
        reports[mode] = run_ablation(data, feature_mode=mode, args=args)
        if baseline_mode is None:
            baseline_mode = mode
    report = {
        "dataset_npz": str(args.dataset_npz),
        "feature_modes": modes,
        "reports": reports,
    }
    if "depth_only" in reports and "depth_force" in reports:
        report["delta_depth_force_vs_depth_only"] = {
            "mean_head_f1": float(reports["depth_force"]["mean_head_f1"] - reports["depth_only"]["mean_head_f1"]),
            "mean_head_balanced_acc": float(
                reports["depth_force"]["mean_head_balanced_acc"] - reports["depth_only"]["mean_head_balanced_acc"]
            ),
            "head_f1_delta": {
                name: float(reports["depth_force"]["head_means"][name]["f1"] - reports["depth_only"]["head_means"][name]["f1"])
                for name in HEADS
            },
        }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
