#!/usr/bin/env python3
"""Create overlay sheets for a trained RingFrameLocalizer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import DepthLocalizerJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_localizer import RingFrameLocalizerNet, _softargmax_2d


def _collate(batch: list[dict]) -> dict:
    return {"records": batch, "image": torch.stack([r["image_rgbd"] for r in batch], dim=0)}


def _denorm_rgb(t: torch.Tensor) -> Image.Image:
    arr = t[:3].permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _draw_marker(draw: ImageDraw.ImageDraw, uv: tuple[float, float], size: tuple[int, int], color: str, r: int = 4) -> None:
    x = float(uv[0]) * (size[0] - 1)
    y = float(uv[1]) * (size[1] - 1)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/ring_frame_overlay"))
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--view_name", type=str, default="wrist")
    ap.add_argument("--sample_kind", type=str, default="positive", choices=["positive", "negative", "all"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = DepthLocalizerJsonlDataset(args.dataset)
    rng = random.Random(args.seed)
    records = []
    for r in ds.records:
        if args.view_name and str(r.get("view_name", "")) != args.view_name:
            continue
        if args.sample_kind != "all" and str(r.get("sample_kind", "")) != args.sample_kind:
            continue
        records.append(r)
    rng.shuffle(records)
    records = records[: max(1, int(args.num_samples))]
    sub = DepthLocalizerJsonlDataset(args.dataset, records=records)
    loader = DataLoader(sub, batch_size=32, shuffle=False, num_workers=0, collate_fn=_collate)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = RingFrameLocalizerNet(**ckpt.get("config", {})).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    metrics = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(args.device)
            out = model(image)
            cx, cy, _ = _softargmax_2d(out["center_heatmap_logits"])
            px, py, _ = _softargmax_2d(out["axis_pos_heatmap_logits"])
            nx, ny, _ = _softargmax_2d(out["axis_neg_heatmap_logits"])
            visible = torch.sigmoid(out["visible_logit"]).detach().cpu().numpy()
            conf = torch.sigmoid(out["confidence_logit"]).detach().cpu().numpy()
            for i, record in enumerate(batch["records"]):
                img = _denorm_rgb(batch["image"][i])
                draw = ImageDraw.Draw(img)
                size = img.size
                gt_center = (float(record.get("frame_center_u", 0.5)), float(record.get("frame_center_v", 0.5)))
                gt_pos = (float(record.get("frame_axis_pos_u", 0.5)), float(record.get("frame_axis_pos_v", 0.5)))
                gt_neg = (float(record.get("frame_axis_neg_u", 0.5)), float(record.get("frame_axis_neg_v", 0.5)))
                pred_center = (float(cx[i, 0].item()), float(cy[i, 0].item()))
                pred_pos = (float(px[i, 0].item()), float(py[i, 0].item()))
                pred_neg = (float(nx[i, 0].item()), float(ny[i, 0].item()))
                _draw_marker(draw, gt_center, size, "lime", r=5)
                _draw_marker(draw, pred_center, size, "red", r=3)
                _draw_marker(draw, gt_pos, size, "cyan", r=4)
                _draw_marker(draw, gt_neg, size, "cyan", r=4)
                _draw_marker(draw, pred_pos, size, "yellow", r=3)
                _draw_marker(draw, pred_neg, size, "yellow", r=3)
                draw.line(
                    (
                        pred_pos[0] * (size[0] - 1),
                        pred_pos[1] * (size[1] - 1),
                        pred_neg[0] * (size[0] - 1),
                        pred_neg[1] * (size[1] - 1),
                    ),
                    fill="yellow",
                    width=2,
                )
                draw.text((4, 4), f"v={visible[i]:.2f} c={conf[i]:.2f} {record.get('sample_kind')}", fill="white")
                tiles.append(img)
                if record.get("sample_kind") == "positive":
                    metrics.append(float(np.linalg.norm(np.asarray(pred_center) - np.asarray(gt_center))))
    if not tiles:
        raise RuntimeError("No overlay tiles created")
    tile_w, tile_h = tiles[0].size
    cols = 8
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * tile_w, (idx // cols) * tile_h))
    sheet_path = args.output_dir / "ring_frame_overlay_sheet.jpg"
    sheet.save(sheet_path, quality=92)
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "num_samples": len(tiles),
        "positive_center_uv_l2_mean": float(np.mean(metrics)) if metrics else 0.0,
        "overlay_path": str(sheet_path),
        "legend": "green/cyan=label center/axis endpoints, red/yellow=prediction center/axis endpoints",
    }
    (args.output_dir / "ring_frame_overlay_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
