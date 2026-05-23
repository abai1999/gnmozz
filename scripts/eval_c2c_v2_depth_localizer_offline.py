#!/usr/bin/env python3
"""Offline evaluation for the Coarse2Contact v2 depth localizer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import DepthLocalizerJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_localizer import DepthGeometryLocalizerNet
from scripts.train_c2c_v2_depth_localizer import _collate, _wrap_symmetry


def _evaluate(model: DepthGeometryLocalizerNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mae_dx = []
    mae_dy = []
    mae_dz = []
    mae_yaw = []
    center_uv_err = []
    axis_pos_uv_err = []
    axis_neg_uv_err = []
    axis_dir_err = []
    conf_acc = []
    pos_conf = []
    neg_conf = []
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
            out = model.predict(image, skill_type_id, stage_id, target_id, reference_id, dof_vec)
            dxdy = torch.stack([out["dx"], out["dy"]], dim=-1) - labels[:, :2]
            dz_err = out["dz"] - labels[:, 2]
            yaw_err = _wrap_symmetry(out["dyaw"] - labels[:, 3])
            conf = out["confidence"]
            mae_dx.append(dxdy[:, 0].abs().mean().item())
            mae_dy.append(dxdy[:, 1].abs().mean().item())
            mae_dz.append(dz_err.abs().mean().item())
            mae_yaw.append(yaw_err.abs().mean().item())
            if "center_u" in out and "axis_u" in out:
                center_uv = torch.stack([out["center_u"], out["center_v"]], dim=-1)
                axis_uv = torch.stack([out["axis_u"], out["axis_v"]], dim=-1)
                center_uv_err.append((center_uv - batch["center_uv"].to(device)).abs().mean().item())
                if "axis_pos_uv" in batch:
                    axis_pos_uv_err.append((axis_uv - batch["axis_pos_uv"].to(device)).abs().mean().item())
                    axis_neg_uv_err.append((torch.stack([out.get("axis_neg_u", out["axis_u"]), out.get("axis_neg_v", out["axis_v"])], dim=-1) - batch["axis_neg_uv"].to(device)).abs().mean().item())
                    if "axis_dir_x" in out and "axis_dir_y" in out:
                        pred_dir = torch.nn.functional.normalize(torch.stack([out["axis_dir_x"], out["axis_dir_y"]], dim=-1), dim=-1)
                        axis_dir_err.append((1.0 - torch.sum(pred_dir * torch.nn.functional.normalize(batch["axis_dir_uv"].to(device), dim=-1), dim=-1)).mean().item())
                else:
                    axis_pos_uv_err.append((axis_uv - batch["axis_uv"].to(device)).abs().mean().item())
            conf_acc.append(((conf > 0.5).float() == positive_mask).float().mean().item())
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
        "axis_pos_uv_error": float(np.mean(axis_pos_uv_err)) if axis_pos_uv_err else 0.0,
        "axis_neg_uv_error": float(np.mean(axis_neg_uv_err)) if axis_neg_uv_err else 0.0,
        "axis_dir_error": float(np.mean(axis_dir_err)) if axis_dir_err else 0.0,
        "axis_uv_error": float(np.mean(axis_pos_uv_err + axis_neg_uv_err)) if (axis_pos_uv_err or axis_neg_uv_err) else 0.0,
        "confidence_accuracy": float(np.mean(conf_acc)) if conf_acc else 0.0,
        "positive_confidence": float(np.mean(pos_conf)) if pos_conf else 0.0,
        "negative_confidence": float(np.mean(neg_conf)) if neg_conf else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    dataset = DepthLocalizerJsonlDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: _collate(batch, vocab))
    model = DepthGeometryLocalizerNet.from_vocab(vocab, **ckpt.get("config", {})).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = _evaluate(model, loader, torch.device(args.device))
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "metrics": metrics,
    }
    out_path = args.output_root / "depth_localizer_eval.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
