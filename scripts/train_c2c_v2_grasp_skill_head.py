#!/usr/bin/env python3
"""Train the C2C v2 GraspSkillHead on close-index anchored successful windows."""

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
from prismatic.robot.coarse2contact_v2.learned_localizer import GraspSkillHeadNet


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _wrap_yaw(x: torch.Tensor, period: float = math.pi / 2.0) -> torch.Tensor:
    return torch.remainder(x + 0.5 * period, period) - 0.5 * period


def _yaw_to_vec(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def _symmetric_endpoint_loss(
    pred_dyaw: torch.Tensor,
    center_uv: torch.Tensor,
    axis_pos_uv: torch.Tensor,
    axis_neg_uv: torch.Tensor,
    jaw_reference_yaw: torch.Tensor,
) -> torch.Tensor:
    abs_yaw = jaw_reference_yaw + pred_dyaw
    dir_xy = torch.stack([torch.cos(abs_yaw), torch.sin(abs_yaw)], dim=-1)
    half_len = 0.5 * torch.linalg.norm(axis_pos_uv - axis_neg_uv, dim=-1, keepdim=True)
    pred_pos = center_uv + dir_xy * half_len
    pred_neg = center_uv - dir_xy * half_len
    direct = torch.abs(pred_pos - axis_pos_uv).mean(dim=-1) + torch.abs(pred_neg - axis_neg_uv).mean(dim=-1)
    swapped = torch.abs(pred_pos - axis_neg_uv).mean(dim=-1) + torch.abs(pred_neg - axis_pos_uv).mean(dim=-1)
    return torch.minimum(direct, swapped).mean()


def _compute_output_scales(records: list[dict]) -> tuple[float, float]:
    xs = []
    ys = []
    for r in records:
        if float(r.get("xyyaw_supervision_mask", 0.0)) <= 0.5:
            continue
        xs.append(abs(float(r.get("label_dx", 0.0))))
        ys.append(abs(float(r.get("label_dy", 0.0))))
    if not xs or not ys:
        return 0.003, 0.003
    x_scale = float(np.percentile(np.asarray(xs, dtype=np.float32), 95))
    y_scale = float(np.percentile(np.asarray(ys, dtype=np.float32), 95))
    x_scale = float(np.clip(max(x_scale * 1.10, 5e-4), 5e-4, 0.01))
    y_scale = float(np.clip(max(y_scale * 1.10, 5e-4), 5e-4, 0.02))
    return x_scale, y_scale


def _compute_yaw_observable_pos_weight(records: list[dict]) -> float:
    total = 0
    pos = 0
    for r in records:
        if float(r.get("xyyaw_supervision_mask", 0.0)) <= 0.5:
            continue
        total += 1
        pos += int(float(r.get("yaw_observable_target", 0.0)) > 0.5)
    neg = max(total - pos, 1)
    if pos <= 0:
        return 1.0
    return float(np.clip(neg / max(pos, 1), 1.0, 8.0))


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
                float(item.get("label_confidence", 0.0)),
                float(item.get("label_observability", 0.0)),
                float(item.get("frame_completeness", 0.0)),
                float(item.get("frame_border_touch", 1.0)),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    proprio_rows = []
    for item in batch:
        proprio = list(item.get("proprio", []))
        if len(proprio) < 15:
            proprio = proprio + [0.0] * (15 - len(proprio))
        proprio_rows.append(proprio[:15])
    proprio = torch.tensor(proprio_rows, dtype=torch.float32)
    labels = torch.tensor(
        [
            [
                float(item.get("label_dx", 0.0)),
                float(item.get("label_dy", 0.0)),
                float(item.get("label_descend_amount", item.get("label_dz", 0.0))),
                float(item.get("label_dyaw", 0.0)),
                float(item.get("ready_to_close", 0.0)),
                float(item.get("label_confidence", 1.0)),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [
            [
                float(item.get("xyyaw_supervision_mask", 0.0)),
                float(item.get("yaw_observable_target", 0.0)),
                float(item.get("yaw_strong_supervision_mask", 0.0)),
                float(item.get("z_supervision_mask", 1.0)),
                float(item.get("ready_supervision_mask", 0.0)),
                float(item.get("near_grasp_basin", 0.0)),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    yaw_geom = torch.tensor(
        [
            [
                float(item.get("priv_frame_center_u", item.get("frame_center_u", 0.5))),
                float(item.get("priv_frame_center_v", item.get("frame_center_v", 0.5))),
                float(item.get("priv_frame_axis_pos_u", item.get("frame_axis_pos_u", 0.5))),
                float(item.get("priv_frame_axis_pos_v", item.get("frame_axis_pos_v", 0.5))),
                float(item.get("priv_frame_axis_neg_u", item.get("frame_axis_neg_u", 0.5))),
                float(item.get("priv_frame_axis_neg_v", item.get("frame_axis_neg_v", 0.5))),
                float(item.get("jaw_reference_yaw", 0.0)),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    return {"image": image, "frame": frame, "proprio": proprio, "labels": labels, "masks": masks, "yaw_geom": yaw_geom}


def _split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    episodes = sorted({int(r["episode_idx"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val_eps = set(episodes[:n_val])
    return [r for r in records if int(r["episode_idx"]) not in val_eps], [r for r in records if int(r["episode_idx"]) in val_eps]


def _filter_records(records: list[dict]) -> list[dict]:
    filtered = []
    for r in records:
        if str(r.get("view_name", "")) != "wrist":
            continue
        if float(r.get("z_supervision_mask", 0.0)) <= 0.0 and float(r.get("ready_supervision_mask", 0.0)) <= 0.0 and float(r.get("xyyaw_supervision_mask", 0.0)) <= 0.0:
            continue
        filtered.append(r)
    return filtered


@torch.no_grad()
def _eval(model: GraspSkillHeadNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    xy = []
    z = []
    yaw = []
    yaw_axis = []
    yaw_endpoint = []
    yaw_obs_acc = []
    yaw_obs_prec = []
    yaw_obs_recall = []
    yaw_obs_pred_rate = []
    yaw_obs_tgt_rate = []
    ready_acc = []
    basin_rate = []
    ready_pred = []
    for batch in loader:
        out = model(batch["image"].to(device), batch["frame"].to(device), batch["proprio"].to(device))
        labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)
        yaw_geom = batch["yaw_geom"].to(device)
        pred_xy = torch.stack([out["dx"], out["dy"]], dim=-1)
        xy_mask = masks[:, 0] > 0.5
        yaw_observable_target = masks[:, 1]
        yaw_active_mask = yaw_observable_target > 0.5
        yaw_strong_mask = masks[:, 2] > 0.5
        z_mask = masks[:, 3] > 0.5
        ready_mask = masks[:, 4] > 0.5
        basin_rate.append(float(torch.mean(masks[:, 5]).item()))
        if torch.any(xy_mask):
            pred_obs = torch.sigmoid(out["yaw_observable_logit"][xy_mask])
            tgt_obs = yaw_observable_target[xy_mask]
            pred_flag = pred_obs > 0.5
            tgt_flag = tgt_obs > 0.5
            yaw_obs_acc.append(float((pred_flag == tgt_flag).float().mean().item()))
            yaw_obs_pred_rate.append(float(pred_flag.float().mean().item()))
            yaw_obs_tgt_rate.append(float(tgt_flag.float().mean().item()))
            tp = float(torch.logical_and(pred_flag, tgt_flag).float().sum().item())
            fp = float(torch.logical_and(pred_flag, torch.logical_not(tgt_flag)).float().sum().item())
            fn = float(torch.logical_and(torch.logical_not(pred_flag), tgt_flag).float().sum().item())
            yaw_obs_prec.append(tp / max(tp + fp, 1.0))
            yaw_obs_recall.append(tp / max(tp + fn, 1.0))
        if torch.any(xy_mask):
            xy.append(torch.linalg.norm(pred_xy[xy_mask] - labels[xy_mask, :2], dim=-1).mean().item())
        if torch.any(yaw_active_mask):
            yaw.append(torch.abs(_wrap_yaw(out["dyaw"][yaw_active_mask] - labels[yaw_active_mask, 3])).mean().item())
            pred_yaw_vec = torch.stack([out["yaw_dir_x"], out["yaw_dir_y"]], dim=-1)
            tgt_yaw_vec = _yaw_to_vec(labels[:, 3])
            yaw_axis.append(F.cosine_similarity(pred_yaw_vec[yaw_active_mask], tgt_yaw_vec[yaw_active_mask], dim=-1).mean().item())
            if torch.any(yaw_strong_mask):
                yaw_endpoint.append(
                    float(
                        _symmetric_endpoint_loss(
                            out["dyaw"][yaw_strong_mask],
                            yaw_geom[yaw_strong_mask, 0:2],
                            yaw_geom[yaw_strong_mask, 2:4],
                            yaw_geom[yaw_strong_mask, 4:6],
                            yaw_geom[yaw_strong_mask, 6],
                        ).item()
                    )
                )
        if torch.any(z_mask):
            z.append(torch.abs(out["dz"][z_mask] - labels[z_mask, 2]).mean().item())
        ready = torch.sigmoid(out["ready_to_close_logit"])
        if torch.any(ready_mask):
            ready_acc.append(((ready[ready_mask] > 0.5).float() == (labels[ready_mask, 4] > 0.5).float()).float().mean().item())
            ready_pred.append(float(torch.mean(ready[ready_mask]).item()))
    return {
        "xy_l2": float(np.mean(xy)) if xy else 0.0,
        "z_l1": float(np.mean(z)) if z else 0.0,
        "yaw_l1": float(np.mean(yaw)) if yaw else 0.0,
        "yaw_axis_cosine": float(np.mean(yaw_axis)) if yaw_axis else 0.0,
        "yaw_endpoint_l1": float(np.mean(yaw_endpoint)) if yaw_endpoint else 0.0,
        "yaw_observable_accuracy": float(np.mean(yaw_obs_acc)) if yaw_obs_acc else 0.0,
        "yaw_observable_precision": float(np.mean(yaw_obs_prec)) if yaw_obs_prec else 0.0,
        "yaw_observable_recall": float(np.mean(yaw_obs_recall)) if yaw_obs_recall else 0.0,
        "yaw_observable_pred_rate": float(np.mean(yaw_obs_pred_rate)) if yaw_obs_pred_rate else 0.0,
        "yaw_observable_target_rate": float(np.mean(yaw_obs_tgt_rate)) if yaw_obs_tgt_rate else 0.0,
        "ready_accuracy": float(np.mean(ready_acc)) if ready_acc else 0.0,
        "ready_pred_mean": float(np.mean(ready_pred)) if ready_pred else 0.0,
        "near_grasp_basin_rate": float(np.mean(basin_rate)) if basin_rate else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/grasp_skill_head"))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    _seed(args.seed)
    full = DepthLocalizerJsonlDataset(args.dataset)
    filtered_records = _filter_records(full.records)
    train_records, val_records = _split(filtered_records, args.val_fraction, args.seed)
    train_ds = DepthLocalizerJsonlDataset(args.dataset, records=train_records)
    val_ds = DepthLocalizerJsonlDataset(args.dataset, records=val_records)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    x_output_scale, y_output_scale = _compute_output_scales(train_records)
    yaw_observable_pos_weight = _compute_yaw_observable_pos_weight(train_records)
    model = GraspSkillHeadNet(x_output_scale=x_output_scale, y_output_scale=y_output_scale).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    train_ready_rate = float(np.mean([float(r.get("ready_to_close", 0.0)) > 0.5 for r in train_records if float(r.get("ready_supervision_mask", 0.0)) > 0.5])) if train_records else 0.0
    val_ready_rate = float(np.mean([float(r.get("ready_to_close", 0.0)) > 0.5 for r in val_records if float(r.get("ready_supervision_mask", 0.0)) > 0.5])) if val_records else 0.0
    report = {
        "dataset": str(args.dataset),
        "epochs": [],
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "train_ready_positive_rate": train_ready_rate,
        "val_ready_positive_rate": val_ready_rate,
        "train_yaw_observable_rate": float(np.mean([float(r.get("yaw_observable_target", 0.0)) for r in train_records])) if train_records else 0.0,
        "val_yaw_observable_rate": float(np.mean([float(r.get("yaw_observable_target", 0.0)) for r in val_records])) if val_records else 0.0,
        "yaw_observable_pos_weight": yaw_observable_pos_weight,
        "x_output_scale": x_output_scale,
        "y_output_scale": y_output_scale,
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            out = model(batch["image"].to(args.device), batch["frame"].to(args.device), batch["proprio"].to(args.device))
            labels = batch["labels"].to(args.device)
            masks = batch["masks"].to(args.device)
            yaw_geom = batch["yaw_geom"].to(args.device)
            pred_xy = torch.stack([out["dx"], out["dy"]], dim=-1)
            xy_mask = masks[:, 0] > 0.5
            yaw_observable_target = masks[:, 1]
            yaw_active_mask = yaw_observable_target > 0.5
            yaw_strong_mask = masks[:, 2] > 0.5
            z_mask = masks[:, 3] > 0.5
            ready_mask = masks[:, 4] > 0.5
            loss_xy = torch.tensor(0.0, device=args.device)
            loss_yaw = torch.tensor(0.0, device=args.device)
            loss_yaw_observable = torch.tensor(0.0, device=args.device)
            loss_z = torch.tensor(0.0, device=args.device)
            loss_ready = torch.tensor(0.0, device=args.device)
            if torch.any(xy_mask):
                loss_xy = F.smooth_l1_loss(pred_xy[xy_mask], labels[xy_mask, :2])
                loss_yaw_observable = F.binary_cross_entropy_with_logits(
                    out["yaw_observable_logit"][xy_mask],
                    yaw_observable_target[xy_mask],
                    pos_weight=torch.tensor(yaw_observable_pos_weight, device=args.device),
                )
            if torch.any(yaw_active_mask):
                pred_yaw_vec = torch.stack([out["yaw_dir_x"], out["yaw_dir_y"]], dim=-1)
                tgt_yaw_vec = _yaw_to_vec(labels[:, 3])
                loss_yaw_vec = F.smooth_l1_loss(pred_yaw_vec[yaw_active_mask], tgt_yaw_vec[yaw_active_mask])
                loss_yaw_scalar = F.smooth_l1_loss(
                    _wrap_yaw(out["dyaw"][yaw_active_mask] - labels[yaw_active_mask, 3]),
                    torch.zeros_like(labels[yaw_active_mask, 3]),
                )
                loss_yaw_endpoint = torch.tensor(0.0, device=args.device)
                if torch.any(yaw_strong_mask):
                    loss_yaw_endpoint = _symmetric_endpoint_loss(
                        out["dyaw"][yaw_strong_mask],
                        yaw_geom[yaw_strong_mask, 0:2],
                        yaw_geom[yaw_strong_mask, 2:4],
                        yaw_geom[yaw_strong_mask, 4:6],
                        yaw_geom[yaw_strong_mask, 6],
                    )
                loss_yaw = 0.65 * loss_yaw_vec + 0.20 * loss_yaw_scalar + 0.15 * loss_yaw_endpoint
            if torch.any(z_mask):
                loss_z = F.smooth_l1_loss(out["dz"][z_mask], labels[z_mask, 2])
            if torch.any(ready_mask):
                loss_ready = F.binary_cross_entropy_with_logits(out["ready_to_close_logit"][ready_mask], labels[ready_mask, 4])
            conf_target = torch.clamp(0.2 + 0.35 * masks[:, 0] + 0.25 * masks[:, 1] + 0.20 * masks[:, 2], 0.0, 1.0)
            loss_conf = F.binary_cross_entropy_with_logits(out["confidence_logit"], conf_target)
            loss = loss_xy + loss_z + 0.5 * loss_yaw + 0.35 * loss_yaw_observable + 0.5 * loss_ready + 0.25 * loss_conf
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        metrics = _eval(model, val_loader, torch.device(args.device))
        score = (
            metrics["xy_l2"]
            + metrics["z_l1"]
            + 0.20 * metrics["yaw_l1"]
            + 0.05 * metrics["yaw_endpoint_l1"]
            + 0.10 * max(0.0, 0.80 - metrics["yaw_observable_precision"])
            + 0.10 * max(0.0, 0.80 - metrics["yaw_observable_recall"])
            - 0.05 * metrics["yaw_axis_cosine"]
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, **metrics}
        report["epochs"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score < best:
            best = score
            torch.save(
                {
                    "model_type": "grasp_skill_head",
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "image_in_channels": 6,
                        "frame_feature_dim": 10,
                        "proprio_dim": 15,
                        "x_output_scale": x_output_scale,
                        "y_output_scale": y_output_scale,
                    },
                    "report": report,
                },
                args.output_dir / "best.pt",
            )
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
