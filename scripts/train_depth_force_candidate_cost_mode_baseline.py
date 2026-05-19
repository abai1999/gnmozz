#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODE_TO_INDEX = {
    "planner": 0,
    "near_hold": 1,
    "contact_backoff": 2,
    "kinematic_hold": 3,
}


class ModeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def _safe_mean_std(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    return (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / np.maximum(std, 1e-6)


def build_features(data: dict[str, np.ndarray], feature_mode: str) -> np.ndarray:
    n = int(np.asarray(data["episode_index"]).shape[0])
    depth_prox = np.asarray(data.get("depth_proximity", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1)
    gripper = np.asarray(data.get("gripper_state", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1)
    stage = np.asarray(data.get("stage_token", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1)
    contact_phase = np.asarray(data.get("contact_phase", data.get("contact_state", np.zeros((n,), dtype=np.float32))), dtype=np.float32).reshape(n, 1)

    force_hist = np.asarray(data.get("force_history", data.get("ft_hist", np.zeros((n, 32, 6), dtype=np.float32))), dtype=np.float32)
    if force_hist.ndim == 2:
        force_hist = force_hist[:, None, :]
    if force_hist.shape[-1] != 6 and force_hist.shape[1] == 6:
        force_hist = np.transpose(force_hist, (0, 2, 1))
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

    planner = np.asarray(data.get("planner_base_action_local", data.get("planner_base_action_local_raw", np.zeros((n, 6), dtype=np.float32))), dtype=np.float32)
    if planner.ndim == 1:
        planner = np.repeat(planner[None, :], n, axis=0)
    action_stats = np.concatenate(
        [
            np.linalg.norm(planner[:, :3], axis=1, keepdims=True),
            np.linalg.norm(planner[:, 3:6], axis=1, keepdims=True),
            np.abs(planner[:, 5]).reshape(n, 1),
            np.asarray(data.get("planner_close_intent", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1),
            np.asarray(data.get("depth_proximity", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(n, 1),
        ],
        axis=1,
    )

    pieces = [depth_prox, gripper, stage, contact_phase, force_stats, action_stats]
    if feature_mode == "augmented":
        total_cost = np.asarray(data.get("candidate_total_cost", np.zeros((n, 65), dtype=np.float32)), dtype=np.float32)
        geom_cost = np.asarray(data.get("candidate_geometry_cost", np.zeros((n, 65), dtype=np.float32)), dtype=np.float32)
        risk_cost = np.asarray(data.get("candidate_risk_cost", np.zeros((n, 65), dtype=np.float32)), dtype=np.float32)
        base = np.stack(
            [
                np.mean(total_cost, axis=1),
                np.std(total_cost, axis=1),
                np.max(total_cost, axis=1),
                np.min(total_cost, axis=1),
                np.max(total_cost, axis=1) - np.min(total_cost, axis=1),
                np.mean(geom_cost, axis=1),
                np.mean(risk_cost, axis=1),
            ],
            axis=1,
        )
        pieces.append(base)

    x = np.concatenate(pieces, axis=1).astype(np.float32)
    return _safe_mean_std(x).astype(np.float32)


def balanced_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    out = {}
    recalls = []
    for c in range(num_classes):
        mask = y_true == c
        if np.any(mask):
            rec = float(np.mean(y_pred[mask] == c))
        else:
            rec = 0.0
        out[f"recall_{c}"] = rec
        recalls.append(rec)
    out["balanced_acc"] = float(np.mean(recalls)) if recalls else 0.0
    out["accuracy"] = float(np.mean(y_true == y_pred))
    return out


def load_data(path: str, feature_mode: str) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    raw = {k: np.asarray(v) for k, v in np.load(path, allow_pickle=False).items()}
    x = build_features(raw, feature_mode=feature_mode)
    mode = np.asarray(raw["candidate_target_mode"]).astype(str)
    y = np.array([MODE_TO_INDEX.get(m, -1) for m in mode], dtype=np.int64)
    keep = y >= 0
    x = x[keep]
    y = y[keep]
    episodes = np.asarray(raw["episode_index"], dtype=np.int64)[keep]
    labels = [m for m in sorted(set(mode.tolist())) if m in MODE_TO_INDEX]
    return x, y, episodes.tolist(), labels


def train_split(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    device = torch.device(args.device)
    num_classes = len(MODE_TO_INDEX)
    model = ModeMLP(x.shape[1], hidden_dim=args.hidden_dim, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = torch.from_numpy(x[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    x_val = torch.from_numpy(x[val_idx]).float().to(device)
    y_val = y[val_idx]
    class_counts = np.bincount(y[train_idx], minlength=num_classes).astype(np.float32)
    class_weight = torch.tensor(class_counts.sum() / np.maximum(class_counts, 1.0), device=device, dtype=torch.float32)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0])
        for start in range(0, perm.numel(), args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_val)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    metrics = balanced_acc(y_val, pred, num_classes=num_classes)
    metrics["per_class_counts"] = {str(i): int(np.sum(y_val == i)) for i in range(num_classes)}
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--feature_mode", type=str, default="runtime_safe", choices=["runtime_safe", "augmented"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    x, y, episodes, labels = load_data(args.dataset_npz, args.feature_mode)
    episodes = np.asarray(episodes, dtype=np.int64)
    uniq = sorted(int(x) for x in np.unique(episodes))
    splits = []
    for heldout in uniq:
        train_idx = np.where(episodes != heldout)[0]
        val_idx = np.where(episodes == heldout)[0]
        if train_idx.size and val_idx.size:
            metrics = train_split(x, y, train_idx, val_idx, args)
            metrics["heldout_episode"] = int(heldout)
            splits.append(metrics)

    report = {
        "dataset_npz": str(args.dataset_npz),
        "feature_mode": str(args.feature_mode),
        "labels": labels,
        "splits": splits,
        "mean_accuracy": float(np.mean([s["accuracy"] for s in splits])) if splits else 0.0,
        "mean_balanced_acc": float(np.mean([s["balanced_acc"] for s in splits])) if splits else 0.0,
        "per_class_recall": {
            str(i): float(np.mean([s[f"recall_{i}"] for s in splits])) if splits else 0.0
            for i in range(len(MODE_TO_INDEX))
        },
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
