#!/usr/bin/env python3
"""Train the C2C v2 RingFrameLocalizer on privileged offline frame labels."""

from __future__ import annotations

import argparse
import json
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
from prismatic.robot.coarse2contact_v2.learned_localizer import (
    RingFrameLocalizerNet,
    _make_gaussian_heatmap,
    _softargmax_2d,
)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    heatmap_size = int(batch[0].get("heatmap_size", 32) or 32)
    sigma = float(batch[0].get("heatmap_sigma_px", 0.9) or 0.9)
    image = torch.stack([item["image_rgbd"] for item in batch], dim=0)
    positive = torch.tensor([1.0 if item.get("sample_kind") == "positive" else 0.0 for item in batch], dtype=torch.float32)
    confidence = torch.tensor([float(item.get("label_confidence", 0.0)) for item in batch], dtype=torch.float32)
    center_uv = torch.tensor([[float(item.get("frame_center_u", 0.5)), float(item.get("frame_center_v", 0.5))] for item in batch], dtype=torch.float32)
    axis_pos_uv = torch.tensor([[float(item.get("frame_axis_pos_u", 0.5)), float(item.get("frame_axis_pos_v", 0.5))] for item in batch], dtype=torch.float32)
    axis_neg_uv = torch.tensor([[float(item.get("frame_axis_neg_u", 0.5)), float(item.get("frame_axis_neg_v", 0.5))] for item in batch], dtype=torch.float32)
    center_heat = torch.stack(
        [
            torch.from_numpy(
                _make_gaussian_heatmap(
                    (float(item.get("frame_center_u", 0.5)), float(item.get("frame_center_v", 0.5))),
                    size=heatmap_size,
                    sigma=sigma,
                    valid=item.get("sample_kind") == "positive",
                )
            )
            for item in batch
        ],
        dim=0,
    ).unsqueeze(1)
    axis_pos_heat = torch.stack(
        [
            torch.from_numpy(
                _make_gaussian_heatmap(
                    (float(item.get("frame_axis_pos_u", 0.5)), float(item.get("frame_axis_pos_v", 0.5))),
                    size=heatmap_size,
                    sigma=sigma,
                    valid=item.get("sample_kind") == "positive",
                )
            )
            for item in batch
        ],
        dim=0,
    ).unsqueeze(1)
    axis_neg_heat = torch.stack(
        [
            torch.from_numpy(
                _make_gaussian_heatmap(
                    (float(item.get("frame_axis_neg_u", 0.5)), float(item.get("frame_axis_neg_v", 0.5))),
                    size=heatmap_size,
                    sigma=sigma,
                    valid=item.get("sample_kind") == "positive",
                )
            )
            for item in batch
        ],
        dim=0,
    ).unsqueeze(1)
    return {
        "image": image,
        "positive": positive,
        "confidence": confidence,
        "center_uv": center_uv,
        "axis_pos_uv": axis_pos_uv,
        "axis_neg_uv": axis_neg_uv,
        "center_heat": center_heat.float(),
        "axis_pos_heat": axis_pos_heat.float(),
        "axis_neg_heat": axis_neg_heat.float(),
    }


def _split_by_episode(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
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
        if r.get("sample_kind") == "positive" and float(r.get("frame_completeness", 0.0)) < 0.5:
            continue
        filtered.append(r)
    return filtered


@torch.no_grad()
def _evaluate(model: RingFrameLocalizerNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    visible_acc = []
    pos_center_err = []
    pos_axis_err = []
    pos_conf = []
    neg_conf = []
    for batch in loader:
        image = batch["image"].to(device)
        positive = batch["positive"].to(device)
        out = model(image)
        visible = torch.sigmoid(out["visible_logit"])
        confidence = torch.sigmoid(out["confidence_logit"])
        cx, cy, _ = _softargmax_2d(out["center_heatmap_logits"])
        px, py, _ = _softargmax_2d(out["axis_pos_heatmap_logits"])
        nx, ny, _ = _softargmax_2d(out["axis_neg_heatmap_logits"])
        pred_center = torch.stack([cx[:, 0], cy[:, 0]], dim=-1)
        pred_axis_pos = torch.stack([px[:, 0], py[:, 0]], dim=-1)
        pred_axis_neg = torch.stack([nx[:, 0], ny[:, 0]], dim=-1)
        visible_acc.append(((visible > 0.5).float() == positive).float().mean().item())
        pos = positive > 0.5
        if torch.any(pos):
            center_gt = batch["center_uv"].to(device)
            axis_pos_gt = batch["axis_pos_uv"].to(device)
            axis_neg_gt = batch["axis_neg_uv"].to(device)
            pos_center_err.append(torch.linalg.norm(pred_center[pos] - center_gt[pos], dim=-1).mean().item())
            axis_err = 0.5 * (
                torch.linalg.norm(pred_axis_pos[pos] - axis_pos_gt[pos], dim=-1)
                + torch.linalg.norm(pred_axis_neg[pos] - axis_neg_gt[pos], dim=-1)
            )
            pos_axis_err.append(axis_err.mean().item())
            pos_conf.append(confidence[pos].mean().item())
        if torch.any(~pos):
            neg_conf.append(confidence[~pos].mean().item())
    return {
        "visible_accuracy": float(np.mean(visible_acc)) if visible_acc else 0.0,
        "positive_center_uv_l2": float(np.mean(pos_center_err)) if pos_center_err else 0.0,
        "positive_axis_uv_l2": float(np.mean(pos_axis_err)) if pos_axis_err else 0.0,
        "positive_confidence": float(np.mean(pos_conf)) if pos_conf else 0.0,
        "negative_confidence": float(np.mean(neg_conf)) if neg_conf else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/ring_frame_localizer"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    _seed(args.seed)
    full = DepthLocalizerJsonlDataset(args.dataset)
    filtered_records = _filter_records(full.records)
    train_records, val_records = _split_by_episode(filtered_records, args.val_fraction, args.seed)
    train_ds = DepthLocalizerJsonlDataset(args.dataset, records=train_records)
    val_ds = DepthLocalizerJsonlDataset(args.dataset, records=val_records)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    model = RingFrameLocalizerNet(heatmap_size=32).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    report = {"dataset": str(args.dataset), "epochs": [], "num_train": len(train_ds), "num_val": len(val_ds)}

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(args.device)
            positive = batch["positive"].to(args.device)
            out = model(image)
            heat_target = torch.cat(
                [batch["center_heat"], batch["axis_pos_heat"], batch["axis_neg_heat"]],
                dim=1,
            ).to(args.device)
            heat_logits = torch.cat(
                [out["center_heatmap_logits"], out["axis_pos_heatmap_logits"], out["axis_neg_heatmap_logits"]],
                dim=1,
            )
            pos = positive > 0.5
            heat_loss = torch.tensor(0.0, device=args.device)
            if torch.any(pos):
                heat_loss = F.binary_cross_entropy_with_logits(heat_logits[pos], heat_target[pos])
            visible_loss = F.binary_cross_entropy_with_logits(out["visible_logit"], positive)
            conf_loss = F.binary_cross_entropy_with_logits(out["confidence_logit"], positive)
            loss = heat_loss + 0.7 * visible_loss + 0.5 * conf_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        metrics = _evaluate(model, val_loader, torch.device(args.device))
        val_score = metrics["positive_center_uv_l2"] + metrics["positive_axis_uv_l2"] + max(0.0, metrics["negative_confidence"] - 0.25)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, **metrics}
        report["epochs"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_score < best:
            best = val_score
            torch.save(
                {
                    "model_type": "ring_frame_localizer",
                    "model_state_dict": model.state_dict(),
                    "config": {"heatmap_size": 32, "image_in_channels": 6},
                    "report": report,
                },
                args.output_dir / "best.pt",
            )
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
