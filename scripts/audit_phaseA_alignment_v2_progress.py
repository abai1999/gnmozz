#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2
from scripts.train_phaseA_alignment_v2_boundary_progress import AlignmentV2Dataset, forward_model


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def label_audit(data: dict[str, np.ndarray]) -> dict:
    source = np.asarray(data["source_name"]).astype(str)
    episode = np.asarray(data["episode_index"], dtype=np.int64)
    axis = np.asarray(data["alignment_v2_axis_block_label"], dtype=np.int64)
    mask = np.asarray(data["alignment_v2_progress_mask"], dtype=np.float32) > 0.5
    label = np.asarray(data["alignment_v2_progress_label"], dtype=np.float32) > 0.5
    amb = np.asarray(data.get("alignment_v2_progress_ambiguous_mask", np.zeros_like(label, dtype=np.float32)), dtype=np.float32) > 0.5
    metrics = np.asarray(data["teacher_metrics_norm"], dtype=np.float32)
    weighted = np.asarray(data.get("alignment_v2_weighted_sum_norm", 0.45 * metrics[:, 0] + 0.30 * metrics[:, 1] + 0.25 * metrics[:, 2]), dtype=np.float32)
    max_axis = np.asarray(data.get("alignment_v2_max_axis_norm", np.max(metrics, axis=1)), dtype=np.float32)

    def bucket(mask_local: np.ndarray) -> dict:
        return {
            "rows": int(np.sum(mask_local)),
            "progress_rows": int(np.sum(mask_local & mask)),
            "positive_rows": int(np.sum(mask_local & mask & label)),
            "negative_rows": int(np.sum(mask_local & mask & ~label)),
            "ambiguous_rows": int(np.sum(mask_local & amb)),
            "xy_norm": _summary(metrics[mask_local, 0]),
            "z_norm": _summary(metrics[mask_local, 1]),
            "yaw_norm": _summary(metrics[mask_local, 2]),
            "weighted_sum": _summary(weighted[mask_local]),
            "max_axis": _summary(max_axis[mask_local]),
        }

    by_source = {s: bucket(source == s) for s in sorted(np.unique(source).tolist())}
    by_axis = {str(a): bucket(axis == a) for a in sorted(np.unique(axis).tolist())}
    by_episode = {str(int(e)): bucket(episode == e) for e in sorted(np.unique(episode).tolist())}
    return {
        "overall": bucket(np.ones((source.shape[0],), dtype=bool)),
        "by_source": by_source,
        "by_axis": by_axis,
        "by_episode": by_episode,
        "top_negative_episodes": sorted(
            ((k, v["negative_rows"]) for k, v in by_episode.items()),
            key=lambda kv: (-kv[1], int(kv[0])),
        )[:10],
    }


@torch.no_grad()
def prediction_audit(dataset_path: Path, ckpt_path: Path, batch_size: int) -> dict:
    ds = AlignmentV2Dataset(str(dataset_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StudentHandoffStateHeadV2().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    margins = []
    labels = []
    masks = []
    source_names = []
    for start in range(0, len(ds), batch_size):
        items = [ds[i] for i in range(start, min(len(ds), start + batch_size))]
        batch = {}
        for key in items[0].keys():
            if key == "source_name":
                batch[key] = [x[key] for x in items]
            else:
                batch[key] = torch.stack([x[key] for x in items], dim=0)
        out = forward_model(model, batch, device)
        progress_logit = out.get("progress_logit")
        if progress_logit is None:
            pred = out["pred_metrics_norm"]
            teacher = batch["teacher_metrics_norm"].to(device=device, dtype=torch.float32)
            progress_logit = (0.45 * teacher[:, 0] + 0.30 * teacher[:, 1] + 0.25 * teacher[:, 2]) - (
                0.45 * pred[:, 0] + 0.30 * pred[:, 1] + 0.25 * pred[:, 2]
            )
        margins.append(progress_logit.detach().cpu().numpy())
        labels.append(batch["progress_label"].numpy())
        masks.append(batch["progress_mask"].numpy())
        source_names.extend(batch["source_name"])
    margin = np.concatenate(margins)
    label = np.concatenate(labels) > 0.5
    mask = np.concatenate(masks) > 0.5
    pred = margin > 0.0
    src = np.asarray(source_names).astype(str)
    pos = mask & label
    neg = mask & ~label
    best = {"balanced_acc": 0.0, "sign": 1, "threshold": 0.0, "pos_recall": 0.0, "neg_recall": 0.0}
    if np.any(mask):
        masked_margin = margin[mask]
        masked_label = label[mask]
        thresholds = np.percentile(masked_margin, np.linspace(0, 100, 401))
        for sign in (1, -1):
            signed = sign * masked_margin
            for threshold in thresholds:
                pred_cal = signed > float(threshold)
                pos_cal = masked_label
                neg_cal = ~masked_label
                pos_recall = float(np.mean(pred_cal[pos_cal])) if np.any(pos_cal) else 0.0
                neg_recall = float(np.mean(~pred_cal[neg_cal])) if np.any(neg_cal) else 0.0
                balanced = 0.5 * (pos_recall + neg_recall)
                if balanced > best["balanced_acc"]:
                    best = {
                        "balanced_acc": float(balanced),
                        "sign": int(sign),
                        "threshold": float(threshold),
                        "pos_recall": float(pos_recall),
                        "neg_recall": float(neg_recall),
                    }
    return {
        "checkpoint": str(ckpt_path),
        "progress_rows": int(np.sum(mask)),
        "pos_recall": float(np.mean(pred[pos])) if np.any(pos) else 0.0,
        "neg_recall": float(np.mean(~pred[neg])) if np.any(neg) else 0.0,
        "balanced_acc": float(0.5 * ((np.mean(pred[pos]) if np.any(pos) else 0.0) + (np.mean(~pred[neg]) if np.any(neg) else 0.0))),
        "calibrated_best": best,
        "margin_positive": _summary(margin[pos]),
        "margin_negative": _summary(margin[neg]),
        "by_source": {
            s: {
                "rows": int(np.sum(mask & (src == s))),
                "pos_recall": float(np.mean(pred[mask & (src == s) & label])) if np.any(mask & (src == s) & label) else 0.0,
                "neg_recall": float(np.mean(~pred[mask & (src == s) & ~label])) if np.any(mask & (src == s) & ~label) else 0.0,
            }
            for s in sorted(np.unique(src).tolist())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.dataset_npz, allow_pickle=False)
    data = {k: np.asarray(raw[k]) for k in raw.files}
    label = label_audit(data)
    (args.output_dir / "progress_label_audit.json").write_text(json.dumps(label, indent=2))
    report = {"label_audit": label}
    if args.ckpt is not None and args.ckpt.exists():
        pred = prediction_audit(args.dataset_npz, args.ckpt, args.batch_size)
        (args.output_dir / "progress_prediction_audit.json").write_text(json.dumps(pred, indent=2))
        report["prediction_audit"] = pred
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
