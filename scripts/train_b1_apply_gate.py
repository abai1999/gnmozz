#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ApplyGateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_eps(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(v.strip()) for v in value.split(",") if v.strip()}


def metrics_for(name: str, probs: np.ndarray, labels: np.ndarray, regret_delta: np.ndarray, threshold: float) -> dict:
    apply = probs >= threshold
    selected = regret_delta[apply]
    neg_selected = selected[selected <= 0.0]
    pos_selected = selected[selected > 0.0]
    return {
        f"{name}_rows": int(labels.size),
        f"{name}_apply_count": int(np.sum(apply)),
        f"{name}_apply_rate": float(np.mean(apply)) if labels.size else 0.0,
        f"{name}_precision_positive_regret": float(np.mean(selected > 0.0)) if selected.size else 0.0,
        f"{name}_negative_apply_count": int(neg_selected.size),
        f"{name}_negative_apply_rate": float(neg_selected.size / max(selected.size, 1)),
        f"{name}_positive_kept_count": int(np.sum((regret_delta > 0.0) & (~apply))),
        f"{name}_mean_regret_delta_selected": float(np.mean(selected)) if selected.size else 0.0,
        f"{name}_mean_regret_delta_all": float(np.mean(regret_delta)) if regret_delta.size else 0.0,
        f"{name}_oracle_positive_rate": float(np.mean(labels > 0.5)) if labels.size else 0.0,
        f"{name}_selected_positive_sum": float(np.sum(pos_selected)) if pos_selected.size else 0.0,
        f"{name}_selected_negative_sum": float(np.sum(neg_selected)) if neg_selected.size else 0.0,
    }


def select_threshold(
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    train_delta: np.ndarray,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    val_delta: np.ndarray,
) -> tuple[float, dict]:
    candidates = np.linspace(0.05, 0.99, 95, dtype=np.float32)
    best_threshold = 0.5
    best_score = -1e18
    best_metrics = {}
    for threshold in candidates:
        train_m = metrics_for("train", train_probs, train_labels, train_delta, float(threshold))
        val_m = metrics_for("val", val_probs, val_labels, val_delta, float(threshold))
        # Safety first: block hard-negative validation windows. Among equally safe
        # thresholds, keep as much positive training regret as possible.
        score = (
            -10000.0 * val_m["val_negative_apply_count"]
            -100.0 * val_m["val_negative_apply_rate"]
            + train_m["train_selected_positive_sum"]
            + 10.0 * train_m["train_precision_positive_regret"]
            - 0.1 * train_m["train_apply_count"]
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {**train_m, **val_m, "threshold_score": float(score)}
    return best_threshold, best_metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--val_episodes", default="11,17")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.dataset_npz, allow_pickle=True)
    x = np.asarray(data["features"], dtype=np.float32)
    y = np.asarray(data["labels"], dtype=np.float32)
    regret_delta = np.asarray(data["regret_delta"], dtype=np.float32)
    episode_index = np.asarray(data["episode_index"], dtype=np.int64)
    feature_names = [str(v) for v in data["feature_names"].tolist()]
    val_eps = parse_eps(args.val_episodes)
    val_mask = np.isin(episode_index, sorted(val_eps))
    if not np.any(val_mask):
        unique = np.unique(episode_index)
        val_mask = episode_index == unique[-1]
        val_eps = {int(unique[-1])}
    train_mask = ~val_mask
    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ApplyGateMLP(x.shape[1], hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tx = torch.from_numpy(x_norm[train_mask]).to(device)
    ty = torch.from_numpy(y[train_mask]).to(device)
    vx = torch.from_numpy(x_norm[val_mask]).to(device)
    pos_weight = torch.tensor(
        [float(np.sum(y[train_mask] <= 0.5) / max(np.sum(y[train_mask] > 0.5), 1.0))],
        device=device,
    )

    best_score = -1e9
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(tx)
        loss = F.binary_cross_entropy_with_logits(logits, ty, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                train_probs = torch.sigmoid(model(tx)).cpu().numpy()
                val_probs = torch.sigmoid(model(vx)).cpu().numpy()
            row = {"epoch": epoch, "loss": float(loss.item())}
            row.update(metrics_for("train", train_probs, y[train_mask], regret_delta[train_mask], args.threshold))
            row.update(metrics_for("val", val_probs, y[val_mask], regret_delta[val_mask], args.threshold))
            history.append(row)
            # Prefer blocking hard validation negatives, then useful positive coverage.
            score = (
                -10000.0 * row["val_negative_apply_count"]
                -100.0 * row["val_negative_apply_rate"]
                + 2.0 * row["val_precision_positive_regret"]
                + row["val_apply_rate"]
                + 0.1 * row["val_mean_regret_delta_selected"]
            )
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(row))

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x_norm).to(device))).cpu().numpy()
    threshold, threshold_metrics = select_threshold(
        probs[train_mask],
        y[train_mask],
        regret_delta[train_mask],
        probs[val_mask],
        y[val_mask],
        regret_delta[val_mask],
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_names": feature_names,
        "threshold": float(threshold),
        "requested_threshold": float(args.threshold),
        "hidden_dim": int(args.hidden_dim),
        "dataset_npz": args.dataset_npz,
        "val_episodes": sorted(int(v) for v in val_eps),
        "model_kind": "b1_apply_gate_mlp",
    }
    torch.save(ckpt, out / "b1_apply_gate_best.pt")
    np.savez_compressed(
        out / "apply_gate_predictions.npz",
        probs=probs.astype(np.float32),
        labels=y,
        regret_delta=regret_delta,
        episode_index=episode_index,
    )
    summary = {
        "dataset_npz": args.dataset_npz,
        "output_dir": str(out),
        "val_episodes": sorted(int(v) for v in val_eps),
        "train_rows": int(np.sum(train_mask)),
        "val_rows": int(np.sum(val_mask)),
        "threshold": float(threshold),
        "requested_threshold": float(args.threshold),
        "threshold_selection": threshold_metrics,
        "final": {
            **metrics_for("all", probs, y, regret_delta, threshold),
            **metrics_for("train", probs[train_mask], y[train_mask], regret_delta[train_mask], threshold),
            **metrics_for("val", probs[val_mask], y[val_mask], regret_delta[val_mask], threshold),
        },
        "history": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
