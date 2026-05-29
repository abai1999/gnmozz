#!/usr/bin/env python3
"""Train a lightweight C2C v2 frame-yaw estimator.

The model predicts jaw-local `dyaw` from runtime-available features and learns a
separate yaw-observability logit.  Privileged yaw is used only as the offline
supervision target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.frame_yaw_estimator import (
    FRAME_YAW_FEATURE_NAMES,
    FrameYawEstimatorNet,
    save_frame_yaw_checkpoint,
)


def _split_by_episode(episodes: np.ndarray, *, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    eps = np.unique(np.asarray(episodes, dtype=np.int64))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(eps)
    if eps.size <= 1:
        idx = np.arange(len(episodes), dtype=np.int64)
        cut = max(1, int(round((1.0 - float(val_ratio)) * len(idx))))
        return idx[:cut], idx[cut:]
    val_count = max(1, int(round(float(val_ratio) * eps.size)))
    val_eps = set(int(x) for x in eps[:val_count])
    val_mask = np.asarray([int(ep) in val_eps for ep in episodes], dtype=bool)
    train_idx = np.where(~val_mask)[0].astype(np.int64)
    val_idx = np.where(val_mask)[0].astype(np.int64)
    if train_idx.size == 0:
        train_idx, val_idx = val_idx, train_idx
    return train_idx, val_idx


def _split_from_dataset(data: np.lib.npyio.NpzFile, episodes: np.ndarray, *, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    if "split" in data.files:
        split = np.asarray(data["split"]).astype(str)
        train_idx = np.where(split == "train")[0].astype(np.int64)
        val_idx = np.where(split == "val")[0].astype(np.int64)
        if train_idx.size > 0 and val_idx.size > 0:
            return train_idx, val_idx, "dataset_split"
    train_idx, val_idx = _split_by_episode(episodes, val_ratio=float(val_ratio), seed=int(seed))
    return train_idx, val_idx, "episode_random_split"


def _metrics(model: FrameYawEstimatorNet, x: torch.Tensor, y: torch.Tensor, obs: torch.Tensor) -> dict[str, Any]:
    if x.numel() == 0:
        return {"rows": 0}
    model.eval()
    with torch.no_grad():
        out = model(x)
    pred = out["dyaw"]
    prob = out["yaw_observable_probability"]
    err = torch.abs(pred - y)
    sign_mask = (torch.abs(y) > 1.0e-6) & (torch.abs(pred) > 1.0e-6)
    sign_match = ((torch.sign(pred[sign_mask]) == torch.sign(y[sign_mask])).float().mean().item() if bool(torch.any(sign_mask)) else 0.0)
    pred_obs = prob >= 0.5
    obs_bool = obs >= 0.5
    tp = int(torch.count_nonzero(pred_obs & obs_bool).item())
    fp = int(torch.count_nonzero(pred_obs & ~obs_bool).item())
    fn = int(torch.count_nonzero(~pred_obs & obs_bool).item())
    tn = int(torch.count_nonzero(~pred_obs & ~obs_bool).item())
    recall = float(tp / max(tp + fn, 1))
    precision = float(tp / max(tp + fp, 1))
    specificity = float(tn / max(tn + fp, 1))
    return {
        "rows": int(x.shape[0]),
        "dyaw_mae": float(err.mean().item()),
        "dyaw_p95_abs_error": float(torch.quantile(err, 0.95).item()) if x.shape[0] > 1 else float(err.mean().item()),
        "dyaw_sign_match_rate": float(sign_match),
        "yaw_observable_accuracy": float((pred_obs == obs_bool).float().mean().item()),
        "yaw_observable_precision": precision,
        "yaw_observable_recall": recall,
        "yaw_observable_specificity": specificity,
        "yaw_observable_balanced_accuracy": float(0.5 * (recall + specificity)),
        "yaw_observable_positive_rate": float(obs_bool.float().mean().item()),
        "predicted_yaw_observable_rate": float(pred_obs.float().mean().item()),
    }


def _threshold_sweep(prob: torch.Tensor, obs: torch.Tensor) -> dict[str, Any]:
    if prob.numel() == 0:
        return {"best_threshold": 0.5, "best_balanced_accuracy": 0.0, "best_accuracy": 0.0}
    thresholds = torch.linspace(0.0, 1.0, 201, dtype=torch.float32)
    obs_bool = obs >= 0.5
    best: dict[str, Any] = {
        "best_threshold": 0.5,
        "best_balanced_accuracy": 0.0,
        "best_accuracy": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_specificity": 0.0,
        "best_f1": 0.0,
    }
    for thr in thresholds:
        pred = prob >= thr
        tp = int(torch.count_nonzero(pred & obs_bool).item())
        fp = int(torch.count_nonzero(pred & ~obs_bool).item())
        fn = int(torch.count_nonzero(~pred & obs_bool).item())
        tn = int(torch.count_nonzero(~pred & ~obs_bool).item())
        recall = float(tp / max(tp + fn, 1))
        precision = float(tp / max(tp + fp, 1))
        specificity = float(tn / max(tn + fp, 1))
        balanced = float(0.5 * (recall + specificity))
        accuracy = float((pred == obs_bool).float().mean().item())
        f1 = float(2.0 * precision * recall / max(precision + recall, 1.0e-9))
        if balanced > best["best_balanced_accuracy"] or (balanced == best["best_balanced_accuracy"] and f1 > best["best_f1"]):
            best = {
                "best_threshold": float(thr.item()),
                "best_balanced_accuracy": balanced,
                "best_accuracy": accuracy,
                "best_precision": precision,
                "best_recall": recall,
                "best_specificity": specificity,
                "best_f1": f1,
            }
    return best


def _proxy_metrics_from_features(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    if x.numel() == 0:
        return {"rows": 0}
    proxy_image_idx = FRAME_YAW_FEATURE_NAMES.index("proxy_image_axis_yaw")
    proxy_residual_idx = FRAME_YAW_FEATURE_NAMES.index("proxy_residual_dyaw")
    proxy_image = x[:, proxy_image_idx]
    proxy_residual = x[:, proxy_residual_idx]
    image_err = torch.abs(proxy_image - y)
    residual_err = torch.abs(proxy_residual - y)
    return {
        "rows": int(x.shape[0]),
        "proxy_image_axis_yaw_mae": float(image_err.mean().item()),
        "proxy_image_axis_yaw_sign_match_rate": float(
            ((torch.sign(proxy_image[torch.abs(proxy_image) > 1.0e-6]) == torch.sign(y[torch.abs(proxy_image) > 1.0e-6])).float().mean().item())
            if bool(torch.any(torch.abs(proxy_image) > 1.0e-6))
            else 0.0
        ),
        "proxy_image_axis_yaw_corr": float(np.corrcoef(proxy_image.detach().cpu().numpy(), y.detach().cpu().numpy())[0, 1]) if x.shape[0] > 1 else 0.0,
        "proxy_residual_dyaw_mae": float(residual_err.mean().item()),
        "proxy_residual_dyaw_sign_match_rate": float(
            ((torch.sign(proxy_residual[torch.abs(proxy_residual) > 1.0e-6]) == torch.sign(y[torch.abs(proxy_residual) > 1.0e-6])).float().mean().item())
            if bool(torch.any(torch.abs(proxy_residual) > 1.0e-6))
            else 0.0
        ),
        "proxy_residual_dyaw_corr": float(np.corrcoef(proxy_residual.detach().cpu().numpy(), y.detach().cpu().numpy())[0, 1]) if x.shape[0] > 1 else 0.0,
    }


def _subset_summary(data: np.lib.npyio.NpzFile, idx: np.ndarray) -> dict[str, Any]:
    if idx.size == 0:
        return {"rows": 0}
    out = {"rows": int(idx.size)}
    for key in ("yaw_observable", "yaw_positive_focus", "yaw_entry_feasible", "near_basin_shell", "visual_observable"):
        if key in data.files:
            arr = np.asarray(data[key], dtype=np.float32)
            out[f"{key}_rows"] = int(np.count_nonzero(arr[idx] > 0.5))
            out[f"{key}_rate"] = float(np.mean(arr[idx] > 0.5))
    if "yaw_stratum" in data.files:
        strata = np.asarray(data["yaw_stratum"]).astype(str)
        out["stratum_counts"] = {str(name): int(np.count_nonzero(strata[idx] == str(name))) for name in sorted(set(strata[idx].tolist()))}
    return out


def train(
    dataset_npz: Path,
    *,
    output_ckpt: Path,
    output_json: Path,
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 1.0e-3,
    val_ratio: float = 0.2,
    seed: int = 7,
    hidden_dim: int = 96,
    observable_loss_weight: float = 0.25,
) -> dict[str, Any]:
    data = np.load(dataset_npz, allow_pickle=True)
    x_np = np.asarray(data["features"], dtype=np.float32)
    y_np = np.asarray(data["dyaw"], dtype=np.float32)
    obs_np = np.asarray(data["yaw_observable"], dtype=np.float32)
    ep_np = np.asarray(data["episode_idx"], dtype=np.int64)
    sample_weight_np = np.asarray(data["sample_weight"], dtype=np.float32) if "sample_weight" in data.files else np.ones_like(obs_np, dtype=np.float32)
    if x_np.ndim != 2 or x_np.shape[0] == 0:
        raise RuntimeError(f"empty frame yaw dataset: {dataset_npz}")
    if x_np.shape[1] != len(FRAME_YAW_FEATURE_NAMES):
        raise RuntimeError(f"feature_dim mismatch: {x_np.shape[1]} != {len(FRAME_YAW_FEATURE_NAMES)}")

    torch.manual_seed(int(seed))
    train_idx, val_idx, split_source = _split_from_dataset(data, ep_np, val_ratio=float(val_ratio), seed=int(seed))
    x = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.float32)
    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    sample_weight = torch.as_tensor(sample_weight_np, dtype=torch.float32)

    train_ds = TensorDataset(x[train_idx], y[train_idx], obs[train_idx], sample_weight[train_idx])
    sampler = WeightedRandomSampler(
        weights=torch.clamp(sample_weight[train_idx], min=1.0e-6),
        num_samples=int(train_idx.size),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    loader = DataLoader(train_ds, batch_size=int(batch_size), sampler=sampler)
    model = FrameYawEstimatorNet(feature_dim=x_np.shape[1], hidden_dim=int(hidden_dim))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    obs_pos = float(np.count_nonzero(obs_np[train_idx] > 0.5))
    obs_neg = float(max(train_idx.size - obs_pos, 0))
    pos_weight = torch.tensor([min(obs_neg / max(obs_pos, 1.0), 50.0)], dtype=torch.float32)

    for _epoch in range(int(epochs)):
        model.train()
        for xb, yb, ob, wb in loader:
            out = model(xb)
            wb = wb / torch.clamp(torch.mean(wb), min=1.0e-6)
            dyaw_loss = torch.mean(F.smooth_l1_loss(out["dyaw"], yb, reduction="none") * wb)
            obs_loss_raw = F.binary_cross_entropy_with_logits(out["yaw_observable_logit"], ob, pos_weight=pos_weight, reduction="none")
            obs_loss = torch.mean(obs_loss_raw * wb)
            conf_target = torch.exp(-torch.abs(out["dyaw"].detach() - yb) / 0.08).clamp(0.0, 1.0)
            conf_loss = torch.mean(F.binary_cross_entropy_with_logits(out["confidence_logit"], conf_target, reduction="none") * wb)
            loss = dyaw_loss + float(observable_loss_weight) * obs_loss + 0.10 * conf_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    train_metrics = _metrics(model, x[train_idx], y[train_idx], obs[train_idx])
    val_metrics = _metrics(model, x[val_idx], y[val_idx], obs[val_idx]) if val_idx.size else {"rows": 0}
    train_metrics.update(_proxy_metrics_from_features(x[train_idx], y[train_idx]))
    val_metrics.update(_proxy_metrics_from_features(x[val_idx], y[val_idx]) if val_idx.size else {"rows": 0})
    with torch.no_grad():
        train_prob = torch.sigmoid(model(x[train_idx])["yaw_observable_logit"]) if train_idx.size else torch.tensor([])
        val_prob = torch.sigmoid(model(x[val_idx])["yaw_observable_logit"]) if val_idx.size else torch.tensor([])
    train_metrics.update({f"threshold_{k}": v for k, v in _threshold_sweep(train_prob, obs[train_idx]).items()})
    val_metrics.update({f"threshold_{k}": v for k, v in _threshold_sweep(val_prob, obs[val_idx]).items()})
    calibrated_threshold = float(val_metrics.get("threshold_best_threshold", 0.5))
    report = {
        "schema_version": "frame_yaw_estimator_train_v1",
        "dataset_npz": str(dataset_npz.resolve()),
        "output_ckpt": str(output_ckpt.resolve()),
        "rows": int(x_np.shape[0]),
        "feature_dim": int(x_np.shape[1]),
        "feature_names": list(FRAME_YAW_FEATURE_NAMES),
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "split_source": str(split_source),
        "train_split_summary": _subset_summary(data, train_idx),
        "val_split_summary": _subset_summary(data, val_idx),
        "yaw_observable_pos_weight": float(pos_weight.item()),
        "yaw_observable_threshold_default": 0.5,
        "yaw_observable_threshold_working_point": calibrated_threshold,
        "yaw_observable_threshold_working_point_source": "val.threshold_best_threshold",
        "train": train_metrics,
        "val": val_metrics,
        "baseline_reference": {
            "proxy_image_axis_yaw": "diagnostic_only",
            "proxy_residual_dyaw": "runtime_available_diagnostic_not_frozen_yaw",
            "label_target": "jaw_local_privileged_dyaw",
        },
        "runtime_policy": {
            "uses_privileged_runtime": False,
            "yaw_apply_enabled": False,
            "intended_use": "shadow_estimator_until_audit_passes",
        },
    }
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_frame_yaw_checkpoint(output_ckpt, model, metadata=report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train C2C v2 frame yaw estimator from a built NPZ dataset.")
    ap.add_argument("--dataset_npz", type=Path, required=True)
    ap.add_argument("--output_ckpt", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/frame_yaw_estimator.pt"))
    ap.add_argument("--output_json", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/frame_yaw_estimator_train.json"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden_dim", type=int, default=96)
    args = ap.parse_args()
    report = train(
        args.dataset_npz,
        output_ckpt=args.output_ckpt,
        output_json=args.output_json,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
    )
    print(json.dumps({"train": report["train"], "val": report["val"], "output_ckpt": report["output_ckpt"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
