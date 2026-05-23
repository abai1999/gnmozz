#!/usr/bin/env python3
"""Audit recovery failure morphology buckets for Coarse2Contact v2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import DepthLocalizerJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_localizer import load_grasp_recovery_checkpoint
from prismatic.robot.coarse2contact_v2.recovery_augmentation import failure_morphology_bucket


def _collate(batch: list[dict]) -> dict[str, torch.Tensor | list[dict]]:
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
                float(item.get("recovery_target_dx", item.get("label_dx", 0.0))),
                float(item.get("recovery_target_dy", item.get("label_dy", 0.0))),
                float(item.get("recovery_target_dyaw", item.get("label_dyaw", 0.0))),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    return {"image": image, "frame": frame, "proprio": proprio, "planner_prior": planner_prior, "labels": labels, "records": batch}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional recovery checkpoint to score buckets.")
    ap.add_argument("--output", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_bucket_audit.json"))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--xy_threshold", type=float, default=0.06)
    ap.add_argument("--yaw_threshold", type=float, default=0.15)
    args = ap.parse_args()

    full = DepthLocalizerJsonlDataset(args.dataset)
    records = [dict(r) for r in full.records]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[failure_morphology_bucket(record, xy_threshold=args.xy_threshold, yaw_threshold=args.yaw_threshold)].append(record)

    score_by_row: dict[tuple[int, int], dict[str, float]] = {}
    if args.checkpoint is not None:
        model, _ = load_grasp_recovery_checkpoint(args.checkpoint, map_location=args.device)
        model = model.to(args.device)
        model.eval()
        loader = DataLoader(full, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
        for batch in loader:
            out = model(
                batch["image"].to(args.device),
                batch["frame"].to(args.device),
                batch["proprio"].to(args.device),
                batch["planner_prior"].to(args.device),
            )
            pred = torch.stack([out["dx"], out["dy"], out["dyaw"]], dim=-1).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            for i, record in enumerate(batch["records"]):
                tgt = labels[i]
                err = np.linalg.norm(np.array([tgt[0], tgt[1], 0.04 * tgt[2]], dtype=np.float32))
                post = tgt - pred[i]
                postn = np.linalg.norm(np.array([post[0], post[1], 0.04 * post[2]], dtype=np.float32))
                score_by_row[(int(record["episode_idx"]), int(record["step_idx"]))] = {
                    "gain": float(err - postn),
                    "xy_cosine": float(np.dot(pred[i, :2], tgt[:2]) / (np.linalg.norm(pred[i, :2]) * np.linalg.norm(tgt[:2]) + 1e-9)),
                    "yaw_sign_match": float(np.sign(pred[i, 2]) == np.sign(tgt[2]) or abs(pred[i, 2]) < 1e-6 or abs(tgt[2]) < 1e-6),
                    "error_norm": float(err),
                }

    bucket_items: list[dict[str, object]] = []
    for bucket_name, rows in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        dx = np.asarray([float(r.get("recovery_target_dx", r.get("trace_error_dx", 0.0)) or 0.0) for r in rows], dtype=np.float32)
        dy = np.asarray([float(r.get("recovery_target_dy", r.get("trace_error_dy", 0.0)) or 0.0) for r in rows], dtype=np.float32)
        dyaw = np.asarray([float(r.get("recovery_target_dyaw", r.get("trace_error_dyaw", 0.0)) or 0.0) for r in rows], dtype=np.float32)
        bias = np.asarray([float(r.get("planner_bias_score", 0.0)) for r in rows], dtype=np.float32)
        gains = []
        xy_cos = []
        yaw_match = []
        for r in rows:
            key = (int(r["episode_idx"]), int(r["step_idx"]))
            if key in score_by_row:
                gains.append(score_by_row[key]["gain"])
                xy_cos.append(score_by_row[key]["xy_cosine"])
                yaw_match.append(score_by_row[key]["yaw_sign_match"])
        bucket_items.append(
            {
                "bucket": bucket_name,
                "count": int(len(rows)),
                "xy_norm_median": float(np.median(np.hypot(dx, dy))) if rows else 0.0,
                "yaw_abs_median": float(np.median(np.abs(dyaw))) if rows else 0.0,
                "planner_bias_score_median": float(np.median(bias)) if rows else 0.0,
                "planner_bias_score_mean": float(np.mean(bias)) if rows else 0.0,
                "recovery_gain_mean": float(np.mean(gains)) if gains else None,
                "recovery_gain_median": float(np.median(gains)) if gains else None,
                "recovery_improved_rate": float(np.mean(np.asarray(gains) > 0.0)) if gains else None,
                "xy_cosine_mean": float(np.mean(xy_cos)) if xy_cos else None,
                "yaw_sign_match_rate": float(np.mean(yaw_match)) if yaw_match else None,
                "example_episode_idx": int(rows[0]["episode_idx"]) if rows else -1,
                "example_step_idx": int(rows[0]["step_idx"]) if rows else -1,
            }
        )

    hard_buckets = [b for b in bucket_items if b.get("recovery_gain_mean") is not None]
    hard_buckets = sorted(hard_buckets, key=lambda item: (float(item["recovery_gain_mean"]), float(item["recovery_improved_rate"] or 0.0)))
    report = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
        "xy_threshold": float(args.xy_threshold),
        "yaw_threshold": float(args.yaw_threshold),
        "rows": len(records),
        "buckets": bucket_items,
        "hard_buckets": hard_buckets[: max(1, min(2, len(hard_buckets)))],
        "bucket_order_by_gain": [item["bucket"] for item in hard_buckets],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
