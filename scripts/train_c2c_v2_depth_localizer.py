#!/usr/bin/env python3
"""Train the Coarse2Contact v2 DepthGeometryLocalizer."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import DepthLocalizerJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_localizer import DepthGeometryLocalizerNet, _make_gaussian_heatmap, _softargmax_2d


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _wrap_symmetry(delta: torch.Tensor, period: float = math.pi / 2.0) -> torch.Tensor:
    if period <= 0:
        return delta
    return torch.remainder(delta + 0.5 * period, period) - 0.5 * period


def _encode_dofs(dofs: list[str], vocab: dict[str, int]) -> torch.Tensor:
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for dof in dofs:
        idx = vocab.get(str(dof))
        if idx is not None:
            vec[int(idx)] = 1.0
    return vec


def _collate(batch: list[dict], vocab: dict[str, dict[str, int]]) -> dict[str, torch.Tensor]:
    images = torch.stack([item["image_rgbd"] for item in batch], dim=0)
    skill_ids = torch.tensor([vocab["skill_type"].get(item["skill_type"], vocab["skill_type"].get("<unk>", 0)) for item in batch], dtype=torch.long)
    stage_ids = torch.tensor([vocab["stage_name"].get(item["stage_name"], vocab["stage_name"].get("<unk>", 0)) for item in batch], dtype=torch.long)
    target_ids = torch.tensor([vocab["entity"].get(item["target_entity"], vocab["entity"].get("<unk>", 0)) for item in batch], dtype=torch.long)
    reference_ids = torch.tensor([vocab["entity"].get(item["reference_entity"], vocab["entity"].get("<unk>", 0)) for item in batch], dtype=torch.long)
    dof_vecs = torch.stack([_encode_dofs(item.get("controlled_dofs_vec", []), vocab["controlled_dofs"]) for item in batch], dim=0)
    labels = torch.tensor([[item["label_dx"], item["label_dy"], item.get("label_dz", 0.0), item["label_dyaw"], item["label_confidence"]] for item in batch], dtype=torch.float32)
    positive_mask = torch.tensor([1.0 if float(item["label_confidence"]) > 0.0 else 0.0 for item in batch], dtype=torch.float32)
    heatmap_size = int(batch[0].get("heatmap_size", 16) or 16)
    heatmap_sigma = float(batch[0].get("heatmap_sigma_px", 1.5) or 1.5)
    heatmap_pos_weight = float(batch[0].get("heatmap_pos_weight", 8.0) or 8.0)
    center_uv = torch.tensor([[float(item.get("frame_center_u", 0.5)), float(item.get("frame_center_v", 0.5))] for item in batch], dtype=torch.float32)
    axis_pos_uv = torch.tensor([[float(item.get("frame_axis_pos_u", item.get("frame_axis_u", 0.5))), float(item.get("frame_axis_pos_v", item.get("frame_axis_v", 0.5)))] for item in batch], dtype=torch.float32)
    axis_neg_uv = torch.tensor([[float(item.get("frame_axis_neg_u", 0.5)), float(item.get("frame_axis_neg_v", 0.5))] for item in batch], dtype=torch.float32)
    axis_dir_uv = torch.tensor([[float(item.get("frame_axis_dir_x", 1.0)), float(item.get("frame_axis_dir_y", 0.0))] for item in batch], dtype=torch.float32)
    center_heatmap = torch.stack([
        torch.from_numpy(_make_gaussian_heatmap((float(item.get("frame_center_u", 0.5)), float(item.get("frame_center_v", 0.5))), size=heatmap_size, sigma=heatmap_sigma, valid=bool(item.get("sample_kind") == "positive"))).float()
        for item in batch
    ], dim=0).unsqueeze(1)
    axis_pos_heatmap = torch.stack([
        torch.from_numpy(_make_gaussian_heatmap((float(item.get("frame_axis_pos_u", item.get("frame_axis_u", 0.5))), float(item.get("frame_axis_pos_v", item.get("frame_axis_v", 0.5)))), size=heatmap_size, sigma=heatmap_sigma, valid=bool(item.get("sample_kind") == "positive"))).float()
        for item in batch
    ], dim=0).unsqueeze(1)
    axis_neg_heatmap = torch.stack([
        torch.from_numpy(_make_gaussian_heatmap((float(item.get("frame_axis_neg_u", 0.5)), float(item.get("frame_axis_neg_v", 0.5))), size=heatmap_size, sigma=heatmap_sigma, valid=bool(item.get("sample_kind") == "positive"))).float()
        for item in batch
    ], dim=0).unsqueeze(1)
    target_yaw = torch.atan2(axis_pos_uv[:, 1] - axis_neg_uv[:, 1], axis_pos_uv[:, 0] - axis_neg_uv[:, 0])
    return {
        "image_rgbd": images,
        "skill_type_id": skill_ids,
        "stage_id": stage_ids,
        "target_entity_id": target_ids,
        "reference_entity_id": reference_ids,
        "dof_vec": dof_vecs,
        "labels": labels,
        "positive_mask": positive_mask,
        "center_uv": center_uv,
        "axis_pos_uv": axis_pos_uv,
        "axis_neg_uv": axis_neg_uv,
        "axis_dir_uv": axis_dir_uv,
        "target_yaw": target_yaw,
        "center_heatmap": center_heatmap,
        "axis_pos_heatmap": axis_pos_heatmap,
        "axis_neg_heatmap": axis_neg_heatmap,
        "heatmap_size": torch.tensor([heatmap_size], dtype=torch.long),
        "heatmap_xy_range_m": torch.tensor([float(batch[0].get("heatmap_xy_range_m", 0.04) or 0.04)], dtype=torch.float32),
        "heatmap_pos_weight": torch.tensor([heatmap_pos_weight], dtype=torch.float32),
    }


def _evaluate(model: DepthGeometryLocalizerNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mae_dx = []
    mae_dy = []
    mae_dz = []
    mae_yaw = []
    center_uv_err = []
    axis_uv_err = []
    conf_correct = []
    neg_conf = []
    pos_conf = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image_rgbd"].to(device)
            skill_type_id = batch["skill_type_id"].to(device)
            stage_id = batch["stage_id"].to(device)
            target_id = batch["target_entity_id"].to(device)
            reference_id = batch["reference_entity_id"].to(device)
            dof_vec = batch["dof_vec"].to(device)
            labels = batch["labels"].to(device)
            positive_mask = batch["positive_mask"].to(device)
            if getattr(model, "prediction_mode", "regression") == "heatmap":
                out = model(image, skill_type_id, stage_id, target_id, reference_id, dof_vec)
                center_x, center_y, _ = _softargmax_2d(out["center_heatmap_logits"])
                axis_pos_logits = out["axis_pos_heatmap_logits"] if "axis_pos_heatmap_logits" in out else out["axis_heatmap_logits"]
                axis_pos_x, axis_pos_y, _ = _softargmax_2d(axis_pos_logits)
                axis_neg_logits = out.get("axis_neg_heatmap_logits", None)
                if axis_neg_logits is not None:
                    axis_neg_x, axis_neg_y, _ = _softargmax_2d(axis_neg_logits)
                else:
                    axis_neg_x, axis_neg_y = axis_pos_x, axis_pos_y
                center_uv = torch.stack([center_x[:, 0], center_y[:, 0]], dim=-1)
                axis_pos_uv = torch.stack([axis_pos_x[:, 0], axis_pos_y[:, 0]], dim=-1)
                axis_neg_uv = torch.stack([axis_neg_x[:, 0], axis_neg_y[:, 0]], dim=-1)
                target_yaw = batch["target_yaw"].to(device)
                yaw_pred = torch.atan2(axis_pos_uv[:, 1] - axis_neg_uv[:, 1], axis_pos_uv[:, 0] - axis_neg_uv[:, 0])
                xy_range = batch["heatmap_xy_range_m"].to(device).view(-1, 1)
                pred_dxdy = (center_uv - 0.5) * 2.0 * xy_range
                target_dxdy = labels[:, :2]
                mae_dx.append((pred_dxdy[:, 0] - target_dxdy[:, 0]).abs().mean().item())
                mae_dy.append((pred_dxdy[:, 1] - target_dxdy[:, 1]).abs().mean().item())
                mae_dz.append((out["confidence_logit"] * 0.0).abs().mean().item())
                mae_yaw.append(_wrap_symmetry(yaw_pred - target_yaw).abs().mean().item())
                center_uv_err.append((center_uv - batch["center_uv"].to(device)).abs().mean().item())
                axis_uv_err.append((axis_pos_uv - batch["axis_pos_uv"].to(device)).abs().mean().item())
                axis_uv_err.append((axis_neg_uv - batch["axis_neg_uv"].to(device)).abs().mean().item())
                conf = torch.sigmoid(out["confidence_logit"])
            else:
                pred = model(image, skill_type_id, stage_id, target_id, reference_id, dof_vec)
                dxdy = pred[:, :2] - labels[:, :2]
                dz_err = pred[:, 2] - labels[:, 2]
                yaw_idx = 3 if pred.shape[-1] >= 5 else 2
                conf_idx = 4 if pred.shape[-1] >= 5 else 3
                yaw_err = _wrap_symmetry(pred[:, yaw_idx] - labels[:, 3])
                conf = torch.sigmoid(pred[:, conf_idx])
                mae_dx.append(dxdy[:, 0].abs().mean().item())
                mae_dy.append(dxdy[:, 1].abs().mean().item())
                mae_dz.append(dz_err.abs().mean().item())
                mae_yaw.append(yaw_err.abs().mean().item())
            conf_correct.append(((conf > 0.5).float() == positive_mask).float().mean().item())
            if torch.any(positive_mask > 0.5):
                pos_conf.append(conf[positive_mask > 0.5].mean().item())
            if torch.any(positive_mask < 0.5):
                neg_conf.append(conf[positive_mask < 0.5].mean().item())
    return {
        "mae_dx": float(np.mean(mae_dx)) if mae_dx else 0.0,
        "mae_dy": float(np.mean(mae_dy)) if mae_dy else 0.0,
        "mae_dz": float(np.mean(mae_dz)) if mae_dz else 0.0,
        "mae_yaw": float(np.mean(mae_yaw)) if mae_yaw else 0.0,
        "center_uv_error": float(np.mean(center_uv_err)) if center_uv_err else 0.0,
        "axis_uv_error": float(np.mean(axis_uv_err)) if axis_uv_err else 0.0,
        "confidence_accuracy": float(np.mean(conf_correct)) if conf_correct else 0.0,
        "positive_confidence": float(np.mean(pos_conf)) if pos_conf else 0.0,
        "negative_confidence": float(np.mean(neg_conf)) if neg_conf else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/depth_localizer"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    _seed_everything(args.seed)
    dataset = DepthLocalizerJsonlDataset(args.dataset)
    vocab = dataset.build_vocab()

    episodes = sorted({int(r["episode_idx"]) for r in dataset.records})
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * float(args.val_fraction))))
    val_episodes = set(episodes[:val_count])
    train_records = [r for r in dataset.records if int(r["episode_idx"]) not in val_episodes]
    val_records = [r for r in dataset.records if int(r["episode_idx"]) in val_episodes]
    train_ds = DepthLocalizerJsonlDataset(args.dataset, records=train_records)
    val_ds = DepthLocalizerJsonlDataset(args.dataset, records=val_records)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda batch: _collate(batch, vocab))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: _collate(batch, vocab))

    model = DepthGeometryLocalizerNet.from_vocab(
        vocab,
        prediction_mode="heatmap",
        heatmap_size=32,
        heatmap_sigma=0.9,
        heatmap_xy_range_m=0.04,
        heatmap_channels=3,
        heatmap_pos_weight=8.0,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "dataset": str(args.dataset),
        "vocab": vocab,
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = defaultdict(float)
        steps = 0
        for batch in train_loader:
            image = batch["image_rgbd"].to(args.device)
            skill_type_id = batch["skill_type_id"].to(args.device)
            stage_id = batch["stage_id"].to(args.device)
            target_id = batch["target_entity_id"].to(args.device)
            reference_id = batch["reference_entity_id"].to(args.device)
            dof_vec = batch["dof_vec"].to(args.device)
            labels = batch["labels"].to(args.device)
            positive_mask = batch["positive_mask"].to(args.device)
            pos = positive_mask > 0.5
            neg = ~pos
            loss_center_uv = torch.tensor(0.0, device=args.device)
            loss_axis_uv = torch.tensor(0.0, device=args.device)
            loss_mid_uv = torch.tensor(0.0, device=args.device)
            loss_axis_dir = torch.tensor(0.0, device=args.device)
            loss_xy = torch.tensor(0.0, device=args.device)
            loss_z = torch.tensor(0.0, device=args.device)
            loss_yaw = torch.tensor(0.0, device=args.device)
            loss_conf = torch.tensor(0.0, device=args.device)
            abstain_loss = torch.tensor(0.0, device=args.device)

            if getattr(model, "prediction_mode", "regression") == "heatmap":
                out = model(image, skill_type_id, stage_id, target_id, reference_id, dof_vec)
                center_logits = out["center_heatmap_logits"]
                axis_pos_logits = out["axis_pos_heatmap_logits"]
                axis_neg_logits = out.get("axis_neg_heatmap_logits", None)
                conf_logits = out["confidence_logit"]
                center_target = batch["center_heatmap"].to(args.device)
                axis_pos_target = batch["axis_pos_heatmap"].to(args.device)
                axis_neg_target = batch["axis_neg_heatmap"].to(args.device)
                pos_weight = batch["heatmap_pos_weight"].to(args.device).view(1)
                heatmap_loss = F.binary_cross_entropy_with_logits(center_logits, center_target, pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(axis_pos_logits, axis_pos_target, pos_weight=pos_weight)
                if axis_neg_logits is not None:
                    heatmap_loss = heatmap_loss + F.binary_cross_entropy_with_logits(axis_neg_logits, axis_neg_target, pos_weight=pos_weight)
                center_x, center_y, _ = _softargmax_2d(center_logits)
                axis_pos_x, axis_pos_y, _ = _softargmax_2d(axis_pos_logits)
                if axis_neg_logits is not None:
                    axis_neg_x, axis_neg_y, _ = _softargmax_2d(axis_neg_logits)
                else:
                    axis_neg_x, axis_neg_y = axis_pos_x, axis_pos_y
                center_uv = torch.stack([center_x[:, 0], center_y[:, 0]], dim=-1)
                axis_pos_uv = torch.stack([axis_pos_x[:, 0], axis_pos_y[:, 0]], dim=-1)
                axis_neg_uv = torch.stack([axis_neg_x[:, 0], axis_neg_y[:, 0]], dim=-1)
                xy_range = batch["heatmap_xy_range_m"].to(args.device).view(-1, 1)
                pred_dxdy = (center_uv - 0.5) * 2.0 * xy_range
                target_dxdy = labels[:, :2]
                loss_xy = torch.tensor(0.0, device=args.device)
                if torch.any(pos):
                    loss_xy = F.smooth_l1_loss(pred_dxdy[pos], target_dxdy[pos])
                target_center_uv = batch["center_uv"].to(args.device)
                target_axis_pos_uv = batch["axis_pos_uv"].to(args.device)
                target_axis_neg_uv = batch["axis_neg_uv"].to(args.device)
                target_axis_dir_uv = F.normalize(batch["axis_dir_uv"].to(args.device), dim=-1)
                loss_center_uv = torch.tensor(0.0, device=args.device)
                loss_axis_uv = torch.tensor(0.0, device=args.device)
                loss_mid_uv = torch.tensor(0.0, device=args.device)
                loss_axis_dir = torch.tensor(0.0, device=args.device)
                if torch.any(pos):
                    loss_center_uv = F.smooth_l1_loss(center_uv[pos], target_center_uv[pos])
                    loss_axis_uv = F.smooth_l1_loss(axis_pos_uv[pos], target_axis_pos_uv[pos]) + F.smooth_l1_loss(axis_neg_uv[pos], target_axis_neg_uv[pos])
                    loss_mid_uv = F.smooth_l1_loss(0.5 * (axis_pos_uv[pos] + axis_neg_uv[pos]), target_center_uv[pos])
                    pred_axis_dir = F.normalize(out["axis_dir_xy"], dim=-1)
                    loss_axis_dir = torch.mean(1.0 - torch.sum(pred_axis_dir[pos] * target_axis_dir_uv[pos], dim=-1))
                yaw_pred = torch.atan2(out["axis_dir_xy"][:, 1], out["axis_dir_xy"][:, 0])
                loss_yaw = torch.tensor(0.0, device=args.device)
                if torch.any(pos):
                    loss_yaw = torch.mean(torch.abs(_wrap_symmetry(yaw_pred[pos] - batch["target_yaw"].to(args.device)[pos])))
                loss_z = torch.tensor(0.0, device=args.device)
                if torch.any(pos):
                    loss_z = F.smooth_l1_loss(out["dz_pred"][pos, 0], labels[pos, 2])
                loss_conf = F.binary_cross_entropy_with_logits(conf_logits, positive_mask)
                conf_prob = torch.sigmoid(conf_logits)
                abstain_loss = torch.mean(conf_prob[neg]) if torch.any(neg) else torch.tensor(0.0, device=args.device)
                loss = heatmap_loss + 0.75 * loss_xy + 0.5 * loss_center_uv + 0.5 * loss_axis_uv + 0.25 * loss_mid_uv + 0.75 * loss_axis_dir + 0.75 * loss_yaw + 1.0 * loss_z + 0.5 * loss_conf + 0.25 * abstain_loss
                running["heatmap"] += float(heatmap_loss.item())
            else:
                pred = model(image, skill_type_id, stage_id, target_id, reference_id, dof_vec)
                loss_xy = torch.tensor(0.0, device=args.device)
                if torch.any(pos):
                    loss_xy = F.smooth_l1_loss(pred[pos, :2], labels[pos, :2])
                loss_z = torch.tensor(0.0, device=args.device)
                if torch.any(pos):
                    loss_z = F.smooth_l1_loss(pred[pos, 2], labels[pos, 2])
                yaw_idx = 3 if pred.shape[-1] >= 5 else 2
                conf_idx = 4 if pred.shape[-1] >= 5 else 3
                yaw_diff = _wrap_symmetry(pred[:, yaw_idx] - labels[:, 3])
                loss_yaw = torch.mean(torch.abs(yaw_diff))
                conf_logits = pred[:, conf_idx]
                conf_target = positive_mask
                loss_conf = F.binary_cross_entropy_with_logits(conf_logits, conf_target)
                conf_prob = torch.sigmoid(conf_logits)
                abstain_loss = torch.mean(conf_prob[neg]) if torch.any(neg) else torch.tensor(0.0, device=args.device)
                loss = loss_xy + 0.5 * loss_z + 0.5 * loss_yaw + 0.5 * loss_conf + 0.25 * abstain_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["loss"] += float(loss.item())
            running["loss_xy"] += float(loss_xy.item())
            running["loss_center_uv"] += float(loss_center_uv.item())
            running["loss_axis_uv"] += float(loss_axis_uv.item())
            running["loss_mid_uv"] += float(loss_mid_uv.item())
            running["loss_axis_dir"] += float(loss_axis_dir.item())
            running["loss_z"] += float(loss_z.item())
            running["loss_yaw"] += float(loss_yaw.item())
            running["loss_conf"] += float(loss_conf.item())
            running["abstain"] += float(abstain_loss.item())
            steps += 1

        train_metrics = {k: v / max(steps, 1) for k, v in running.items()}
        val_metrics = _evaluate(model, val_loader, torch.device(args.device))
        epoch_report = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        report["epochs"].append(epoch_report)
        val_loss = val_metrics["mae_dx"] + val_metrics["mae_dy"] + val_metrics["mae_dz"] + val_metrics["mae_yaw"]
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = args.output_dir / "depth_localizer_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "image_in_channels": 6,
                        "image_hidden_dim": 128,
                        "embed_dim": 24,
                        "output_dim": 5,
                        "prediction_mode": getattr(model, "prediction_mode", "regression"),
                        "heatmap_size": int(getattr(model, "heatmap_size", 16)),
                        "heatmap_sigma": float(getattr(model, "heatmap_sigma", 1.5)),
                        "heatmap_xy_range_m": float(getattr(model, "heatmap_xy_range_m", 0.04)),
                        "heatmap_channels": int(getattr(model, "heatmap_channels", 3)),
                        "heatmap_pos_weight": float(getattr(model, "heatmap_pos_weight", 8.0)),
                    },
                    "metrics": val_metrics,
                },
                ckpt_path,
            )

    (args.output_dir / "depth_localizer_train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output_dir / "depth_localizer_best.pt")


if __name__ == "__main__":
    main()
