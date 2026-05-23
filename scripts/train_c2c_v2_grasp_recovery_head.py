#!/usr/bin/env python3
"""Train a recovery-aware grasp head for large-bias near-grasp correction."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import DepthLocalizerJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_localizer import GraspRecoveryHeadNet


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _wrap_yaw(x: torch.Tensor, period: float = math.pi / 2.0) -> torch.Tensor:
    return torch.remainder(x + 0.5 * period, period) - 0.5 * period


def _compute_output_scales(records: list[dict]) -> tuple[float, float, float]:
    xs, ys, yaws = [], [], []
    for r in records:
        xs.append(abs(float(r.get("recovery_target_dx", r.get("trace_error_dx", 0.0)))))
        ys.append(abs(float(r.get("recovery_target_dy", r.get("trace_error_dy", 0.0)))))
        yaws.append(abs(float(r.get("recovery_target_dyaw", r.get("trace_error_dyaw", 0.0)))))
    if not xs:
        return 0.01, 0.01, 0.2
    x_scale = float(np.clip(max(np.percentile(np.asarray(xs, dtype=np.float32), 95) * 1.10, 5e-4), 5e-4, 0.02))
    y_scale = float(np.clip(max(np.percentile(np.asarray(ys, dtype=np.float32), 95) * 1.10, 5e-4), 5e-4, 0.03))
    yaw_scale = float(np.clip(max(np.percentile(np.asarray(yaws, dtype=np.float32), 95) * 1.10, 0.05), 0.05, 0.9))
    return x_scale, y_scale, yaw_scale


def _compute_large_bias_threshold(records: list[dict], quantile: float) -> float:
    scores = np.asarray([float(r.get("planner_bias_score", 0.0)) for r in records], dtype=np.float32)
    if scores.size == 0:
        return 0.0
    return float(np.quantile(scores, float(np.clip(quantile, 0.0, 1.0))))


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    image = torch.stack([item["image_rgbd"] for item in batch], dim=0)
    frame = torch.tensor(
        [
            [
                float(item.get("frame_center_u", 0.5)),
                float(item.get("frame_center_v", 0.5)),
                float(item.get("frame_axis_pos_u", 0.5)),
                float(item.get("frame_axis_pos_v", 0.5)),
                float(item.get("frame_axis_neg_u", 0.5)),
                float(item.get("frame_axis_neg_v", 0.5)),
                float(item.get("frame_confidence", 0.0)),
                float(item.get("frame_observability", 0.0)),
                float(item.get("frame_completeness", 0.0)),
                float(item.get("frame_border_touch", 1.0)),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    proprio = torch.tensor(
        [
            (list(item.get("proprio", [])) + [0.0] * 15)[:15]
            for item in batch
        ],
        dtype=torch.float32,
    )
    planner_prior = torch.tensor(
        [
            (list(item.get("planner_prior_delta", [])) + [0.0] * 6)[:6]
            for item in batch
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [
            [
                float(item.get("recovery_target_dx", item.get("trace_error_dx", 0.0))),
                float(item.get("recovery_target_dy", item.get("trace_error_dy", 0.0))),
                float(item.get("recovery_target_dyaw", item.get("trace_error_dyaw", 0.0))),
                float(item.get("recovery_needed", 0.0)),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    sample_weight = torch.tensor(
        [
            float(item.get("planner_bias_score", 0.0))
            for item in batch
        ],
        dtype=torch.float32,
    )
    bucket_weight = torch.tensor(
        [
            float(item.get("recovery_bucket_weight", 1.0))
            for item in batch
        ],
        dtype=torch.float32,
    )
    augmented = torch.tensor(
        [float(item.get("is_augmented", 0.0)) for item in batch],
        dtype=torch.float32,
    )
    recovery_needed = torch.tensor(
        [float(item.get("recovery_needed", 0.0)) for item in batch],
        dtype=torch.float32,
    )
    return {
        "image": image,
        "frame": frame,
        "proprio": proprio,
        "planner_prior": planner_prior,
        "labels": labels,
        "sample_weight": sample_weight,
        "bucket_weight": bucket_weight,
        "is_augmented": augmented,
        "recovery_needed": recovery_needed,
    }


def _split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    episodes = sorted({int(r["episode_idx"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val_eps = set(episodes[:n_val])
    return [r for r in records if int(r["episode_idx"]) not in val_eps], [r for r in records if int(r["episode_idx"]) in val_eps]


def _filter(records: list[dict]) -> list[dict]:
    filtered = []
    for r in records:
        if str(r.get("view_name", "")) != "wrist":
            continue
        if float(r.get("trace_error_confidence", 0.0)) <= 0.0:
            continue
        if not r.get("rgb_path") or not r.get("depth_path"):
            continue
        filtered.append(r)
    return filtered


@torch.no_grad()
def _evaluate(model: GraspRecoveryHeadNet, loader: DataLoader, device: torch.device, *, large_bias_threshold: float) -> dict[str, float]:
    model.eval()
    all_gain = []
    all_gain_large = []
    all_error_norm = []
    all_post_norm = []
    all_pred_xy_cos = []
    all_pred_dyaw_sign = []
    all_conf = []
    all_needed = []
    all_needed_pred = []
    for batch in loader:
        out = model(
            batch["image"].to(device),
            batch["frame"].to(device),
            batch["proprio"].to(device),
            batch["planner_prior"].to(device),
        )
        labels = batch["labels"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        pred = torch.stack([out["dx"], out["dy"], out["dyaw"]], dim=-1)
        tgt = labels[:, :3]
        err_norm = torch.linalg.norm(torch.stack([tgt[:, 0], tgt[:, 1], 0.04 * tgt[:, 2]], dim=-1), dim=-1)
        post = tgt - pred
        post_norm = torch.linalg.norm(torch.stack([post[:, 0], post[:, 1], 0.04 * post[:, 2]], dim=-1), dim=-1)
        gain = err_norm - post_norm
        large = sample_weight >= float(large_bias_threshold)
        xy_cos = F.cosine_similarity(pred[:, :2], tgt[:, :2], dim=-1)
        all_gain.extend(gain.detach().cpu().tolist())
        all_error_norm.extend(err_norm.detach().cpu().tolist())
        all_post_norm.extend(post_norm.detach().cpu().tolist())
        all_pred_xy_cos.extend(xy_cos.detach().cpu().tolist())
        all_pred_dyaw_sign.extend((torch.sign(pred[:, 2]) == torch.sign(tgt[:, 2])).float().detach().cpu().tolist())
        all_conf.extend(torch.sigmoid(out["confidence_logit"]).detach().cpu().tolist())
        all_needed.extend(labels[:, 3].detach().cpu().tolist())
        all_needed_pred.extend((torch.sigmoid(out["confidence_logit"]) > 0.5).float().detach().cpu().tolist())
        if torch.any(large):
            all_gain_large.extend(gain[large].detach().cpu().tolist())
    return {
        "recovery_gain_mean": float(np.mean(all_gain)) if all_gain else 0.0,
        "recovery_gain_median": float(np.median(all_gain)) if all_gain else 0.0,
        "recovery_improved_rate": float(np.mean(np.asarray(all_gain) > 0.0)) if all_gain else 0.0,
        "recovery_gain_large_mean": float(np.mean(all_gain_large)) if all_gain_large else 0.0,
        "recovery_improved_rate_large": float(np.mean(np.asarray(all_gain_large) > 0.0)) if all_gain_large else 0.0,
        "error_norm_mean": float(np.mean(all_error_norm)) if all_error_norm else 0.0,
        "post_norm_mean": float(np.mean(all_post_norm)) if all_post_norm else 0.0,
        "xy_cosine_mean": float(np.mean(all_pred_xy_cos)) if all_pred_xy_cos else 0.0,
        "yaw_sign_match_rate": float(np.mean(all_pred_dyaw_sign)) if all_pred_dyaw_sign else 0.0,
        "confidence_mean": float(np.mean(all_conf)) if all_conf else 0.0,
        "recovery_needed_rate": float(np.mean(all_needed)) if all_needed else 0.0,
        "recovery_pred_needed_rate": float(np.mean(all_needed_pred)) if all_needed_pred else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/grasp_recovery_head_v1"))
    ap.add_argument("--init_checkpoint", type=Path, default=None, help="Optional checkpoint to resume from.")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--bias_quantile", type=float, default=0.7)
    ap.add_argument("--augmented_loss_mult", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    _seed(args.seed)
    full = DepthLocalizerJsonlDataset(args.dataset)
    records = _filter(full.records)
    train_records, val_records = _split(records, args.val_fraction, args.seed)
    if not train_records or not val_records:
        raise RuntimeError("Need both train and val records for recovery training")

    x_scale, y_scale, yaw_scale = _compute_output_scales(train_records)
    large_bias_threshold = _compute_large_bias_threshold(train_records, args.bias_quantile)

    train_ds = DepthLocalizerJsonlDataset(args.dataset, records=train_records)
    val_ds = DepthLocalizerJsonlDataset(args.dataset, records=val_records)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    model = GraspRecoveryHeadNet(
        x_output_scale=x_scale,
        y_output_scale=y_scale,
        yaw_output_scale=yaw_scale,
    ).to(args.device)

    best_metric = -1e9
    best_state = None
    history: list[dict] = []

    if args.init_checkpoint is not None:
        init_ckpt = torch.load(args.init_checkpoint, map_location=args.device)
        init_state = dict(init_ckpt["model_state_dict"])
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        if missing:
            print(f"[resume] missing keys: {sorted(missing)}")
        if unexpected:
            print(f"[resume] unexpected keys: {sorted(unexpected)}")
        best_metric = float(init_ckpt.get("best_metric", best_metric))
        history = list(init_ckpt.get("history", []))
        best_state = {
            "model_state_dict": model.state_dict(),
            "config": dict(init_ckpt.get("config", {})),
            "history": history,
            "best_metric": float(best_metric),
            "seed": int(args.seed),
        }

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(int(args.epochs)):
        model.train()
        loss_meter = []
        gain_meter = []
        large_gain_meter = []
        for batch in train_loader:
            out = model(
                batch["image"].to(args.device),
                batch["frame"].to(args.device),
                batch["proprio"].to(args.device),
                batch["planner_prior"].to(args.device),
            )
            labels = batch["labels"].to(args.device)
            sample_weight = batch["sample_weight"].to(args.device)
            bucket_weight = batch["bucket_weight"].to(args.device)
            augmented = batch["is_augmented"].to(args.device)
            needed = batch["recovery_needed"].to(args.device)
            pred_xy = torch.stack([out["dx"], out["dy"]], dim=-1)
            tgt_xy = labels[:, :2]
            tgt_yaw = labels[:, 2]
            pred_yaw = out["dyaw"]
            err_weight = 1.0 + 3.0 * torch.clamp(sample_weight / max(large_bias_threshold, 1e-6), 0.0, 2.0)
            err_weight = err_weight * torch.clamp(bucket_weight, 0.5, 3.0)
            if float(args.augmented_loss_mult) != 1.0:
                err_weight = err_weight * torch.where(
                    augmented > 0.5,
                    torch.full_like(err_weight, float(args.augmented_loss_mult)),
                    torch.ones_like(err_weight),
                )
            large_mask = sample_weight >= large_bias_threshold
            xy_loss = F.smooth_l1_loss(pred_xy, tgt_xy, reduction="none").mean(dim=-1)
            yaw_loss = F.smooth_l1_loss(_wrap_yaw(pred_yaw - tgt_yaw), torch.zeros_like(tgt_yaw), reduction="none")
            conf_loss = F.binary_cross_entropy_with_logits(out["confidence_logit"], needed, reduction="none")
            loss = (xy_loss + 1.5 * yaw_loss + 0.3 * conf_loss) * err_weight
            loss = loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                error_norm = torch.linalg.norm(torch.stack([tgt_xy[:, 0], tgt_xy[:, 1], 0.04 * tgt_yaw], dim=-1), dim=-1)
                post = torch.stack([tgt_xy[:, 0] - out["dx"], tgt_xy[:, 1] - out["dy"], 0.04 * (_wrap_yaw(tgt_yaw - pred_yaw))], dim=-1)
                post_norm = torch.linalg.norm(post, dim=-1)
                gain = error_norm - post_norm
                loss_meter.append(float(loss.item()))
                gain_meter.append(float(torch.mean(gain).item()))
                if torch.any(large_mask):
                    large_gain_meter.append(float(torch.mean(gain[large_mask]).item()))
        val_metrics = _evaluate(model, val_loader, torch.device(args.device), large_bias_threshold=large_bias_threshold)
        metric = float(val_metrics["recovery_gain_large_mean"] + 0.25 * val_metrics["recovery_gain_mean"])
        history.append(
            {
                "epoch": len(history),
                "train_loss_mean": float(np.mean(loss_meter)) if loss_meter else 0.0,
                "train_gain_mean": float(np.mean(gain_meter)) if gain_meter else 0.0,
                "train_large_gain_mean": float(np.mean(large_gain_meter)) if large_gain_meter else 0.0,
                **val_metrics,
            }
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model_type": "grasp_recovery",
                    "image_in_channels": 6,
                    "image_hidden_dim": 128,
                    "frame_feature_dim": 10,
                    "proprio_dim": 15,
                    "planner_prior_dim": 6,
                    "x_output_scale": float(x_scale),
                    "y_output_scale": float(y_scale),
                    "yaw_output_scale": float(yaw_scale),
                    "large_bias_threshold": float(large_bias_threshold),
                    "bias_quantile": float(args.bias_quantile),
                },
                "history": history,
                "best_metric": float(best_metric),
                "seed": int(args.seed),
            }

    if best_state is None:
        best_state = {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_type": "grasp_recovery",
                "image_in_channels": 6,
                "image_hidden_dim": 128,
                "frame_feature_dim": 10,
                "proprio_dim": 15,
                "planner_prior_dim": 6,
                "x_output_scale": float(x_scale),
                "y_output_scale": float(y_scale),
                "yaw_output_scale": float(yaw_scale),
                "large_bias_threshold": float(large_bias_threshold),
                "bias_quantile": float(args.bias_quantile),
            },
            "history": history,
            "best_metric": float(best_metric),
            "seed": int(args.seed),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "best.pt"
    torch.save(best_state, ckpt_path)
    report_path = args.output_dir / "train_report.json"
    report_path.write_text(json.dumps({"history": history, "best_metric": best_metric, "checkpoint": str(ckpt_path)}, indent=2, sort_keys=True), encoding="utf-8")
    print(ckpt_path)
    print(report_path)


if __name__ == "__main__":
    main()
