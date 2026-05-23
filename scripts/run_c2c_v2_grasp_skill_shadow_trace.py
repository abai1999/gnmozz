#!/usr/bin/env python3
"""Episode-ordered shadow rollout traces for the learned GraspSkillHead."""

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
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import DepthLocalizerJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_localizer import load_grasp_skill_head_checkpoint


def _wrap_yaw_np(x: np.ndarray | float, period: float = math.pi / 2.0) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    wrapped = np.remainder(arr + 0.5 * period, period) - 0.5 * period
    if np.isscalar(x):
        return float(wrapped)
    return wrapped


def _safe_corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _apply_continuous_yaw_gate(
    rows: list[dict],
    *,
    threshold: float,
    stability_steps: int,
    geometry_score_threshold: float,
    geometry_confidence_threshold: float,
    geometry_abs_rel_yaw_threshold: float,
) -> list[dict]:
    stability_steps = max(int(stability_steps), 1)
    streak = 0
    for row in rows:
        if float(row.get("pred_yaw_observable", 0.0)) > float(threshold):
            streak += 1
        else:
            streak = 0
        row["pred_yaw_stable_count"] = int(streak)
        row["pred_yaw_active_stable"] = bool(streak >= stability_steps)
        geom_score = float(row.get("yaw_observability_score", 0.0))
        geom_conf = float(row.get("frame_confidence", 0.0))
        geom_rel_yaw = abs(float(row.get("frame_rel_yaw", 0.0)))
        row["frame_geometry_consistent"] = bool(
            geom_score >= float(geometry_score_threshold)
            and geom_conf >= float(geometry_confidence_threshold)
            and geom_rel_yaw <= float(geometry_abs_rel_yaw_threshold)
        )
        row["pred_yaw_active"] = bool(row["pred_yaw_active_stable"] and row["frame_geometry_consistent"])
        row["pred_yaw_abstained"] = bool(not row["pred_yaw_active"])
    return rows


def _image_size(path: str, cache: dict[str, tuple[int, int]]) -> tuple[int, int]:
    key = str(path)
    if key not in cache:
        with Image.open(key) as img:
            cache[key] = tuple(int(x) for x in img.size)
    return cache[key]


def _filter_records(records: list[dict], *, episode_indices: set[int] | None = None) -> list[dict]:
    filtered = []
    for r in records:
        if str(r.get("view_name", "")) != "wrist":
            continue
        step = int(r.get("steps_to_close", -1))
        if step < 1 or step > 12:
            continue
        ep = int(r.get("episode_idx", -1))
        if episode_indices is not None and ep not in episode_indices:
            continue
        filtered.append(r)
    return filtered


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
                float(item.get("label_confidence", 0.0)),
                float(item.get("label_observability", 0.0)),
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
    return {"image": image, "frame": frame, "proprio": proprio, "records": batch}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--episode_indices", type=str, default="5,8,19")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--yaw_observable_threshold", type=float, default=0.7)
    ap.add_argument("--yaw_observable_stability_steps", type=int, default=2)
    ap.add_argument("--yaw_geometry_min_score", type=float, default=0.60)
    ap.add_argument("--yaw_geometry_min_confidence", type=float, default=0.33)
    ap.add_argument("--yaw_geometry_max_abs_rel_yaw", type=float, default=0.02)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode_indices = {int(x.strip()) for x in str(args.episode_indices).split(",") if x.strip()}
    full = DepthLocalizerJsonlDataset(args.dataset)
    records = _filter_records(full.records, episode_indices=episode_indices)
    records = sorted(records, key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step_idx", -1))))
    ds = DepthLocalizerJsonlDataset(args.dataset, records=records)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    model, _ = load_grasp_skill_head_checkpoint(args.checkpoint, map_location=args.device)
    model = model.to(args.device)
    model.eval()

    traces_by_ep: dict[int, list[dict]] = defaultdict(list)
    image_size_cache: dict[str, tuple[int, int]] = {}
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(args.device), batch["frame"].to(args.device), batch["proprio"].to(args.device))
            pred_dx = out["dx"].detach().cpu().numpy()
            pred_dy = out["dy"].detach().cpu().numpy()
            pred_dz = out["dz"].detach().cpu().numpy()
            pred_dyaw = out["dyaw"].detach().cpu().numpy()
            pred_ready = torch.sigmoid(out["ready_to_close_logit"]).detach().cpu().numpy()
            pred_conf = torch.sigmoid(out["confidence_logit"]).detach().cpu().numpy()
            pred_yaw_observable = torch.sigmoid(out["yaw_observable_logit"]).detach().cpu().numpy()
            pred_yaw_dir_x = out["yaw_dir_x"].detach().cpu().numpy()
            pred_yaw_dir_y = out["yaw_dir_y"].detach().cpu().numpy()

            for i, record in enumerate(batch["records"]):
                ep = int(record.get("episode_idx", -1))
                frame_axis_yaw = float(math.atan2(float(record.get("frame_axis_dir_y", 0.0)), float(record.get("frame_axis_dir_x", 1.0))))
                jaw_ref_yaw = float(record.get("jaw_reference_yaw", 0.0))
                rel_frame_yaw = float(_wrap_yaw_np(frame_axis_yaw - jaw_ref_yaw))
                rgb_path = str(record.get("rgb_path", ""))
                img_w, img_h = _image_size(rgb_path, image_size_cache) if rgb_path else (128, 128)
                x0, y0, x1, y1 = [int(v) for v in record.get("roi_box", [0, 0, img_w, img_h])]
                roi_w = max(1, x1 - x0)
                roi_h = max(1, y1 - y0)
                frame_center_full_u = (float(x0) + float(record.get("frame_center_u", 0.5)) * float(max(roi_w - 1, 1))) / float(max(img_w - 1, 1))
                frame_center_full_v = (float(y0) + float(record.get("frame_center_v", 0.5)) * float(max(roi_h - 1, 1))) / float(max(img_h - 1, 1))
                jaw_ref_crop_u = (float(record.get("jaw_reference_u", 0.5)) * float(max(img_w - 1, 1)) - float(x0)) / float(max(roi_w - 1, 1))
                jaw_ref_crop_v = (float(record.get("jaw_reference_v", 0.5)) * float(max(img_h - 1, 1)) - float(y0)) / float(max(roi_h - 1, 1))
                row = {
                    "episode_idx": ep,
                    "step_idx": int(record.get("step_idx", -1)),
                    "steps_to_close": int(record.get("steps_to_close", -1)),
                    "near_grasp_basin": bool(float(record.get("near_grasp_basin", 0.0)) > 0.5),
                    "xyyaw_supervision_mask": bool(float(record.get("xyyaw_supervision_mask", 0.0)) > 0.5),
                    "yaw_observable_target": bool(float(record.get("yaw_observable_target", 0.0)) > 0.5),
                    "yaw_low_observability": bool(float(record.get("yaw_low_observability", 0.0)) > 0.5),
                    "yaw_near_zero": bool(float(record.get("yaw_near_zero", 0.0)) > 0.5),
                    "yaw_observability_score": float(record.get("yaw_observability_score", 0.0)),
                    "frame_center_u": float(record.get("frame_center_u", 0.5)),
                    "frame_center_v": float(record.get("frame_center_v", 0.5)),
                    "frame_center_full_u": frame_center_full_u,
                    "frame_center_full_v": frame_center_full_v,
                    "frame_axis_dir_x": float(record.get("frame_axis_dir_x", 1.0)),
                    "frame_axis_dir_y": float(record.get("frame_axis_dir_y", 0.0)),
                    "frame_confidence": float(record.get("frame_confidence", 0.0)),
                    "frame_completeness": float(record.get("frame_completeness", 0.0)),
                    "roi_box": [x0, y0, x1, y1],
                    "jaw_reference_u": float(record.get("jaw_reference_u", 0.5)),
                    "jaw_reference_v": float(record.get("jaw_reference_v", 0.5)),
                    "jaw_reference_yaw": jaw_ref_yaw,
                    "jaw_reference_crop_u": jaw_ref_crop_u,
                    "jaw_reference_crop_v": jaw_ref_crop_v,
                    "frame_rel_crop_u": float(record.get("frame_center_u", 0.5)) - jaw_ref_crop_u,
                    "frame_rel_crop_v": float(record.get("frame_center_v", 0.5)) - jaw_ref_crop_v,
                    "frame_rel_full_u": frame_center_full_u - float(record.get("jaw_reference_u", 0.5)),
                    "frame_rel_full_v": frame_center_full_v - float(record.get("jaw_reference_v", 0.5)),
                    "frame_rel_yaw": rel_frame_yaw,
                    "pred_dx": float(pred_dx[i]),
                    "pred_dy": float(pred_dy[i]),
                    "pred_dz": float(pred_dz[i]),
                    "pred_dyaw": float(pred_dyaw[i]),
                    "pred_xy_norm": float(np.linalg.norm(np.asarray([pred_dx[i], pred_dy[i]], dtype=np.float64))),
                    "pred_ready": float(pred_ready[i]),
                    "pred_confidence": float(pred_conf[i]),
                    "pred_yaw_observable": float(pred_yaw_observable[i]),
                    "pred_yaw_abstained": bool(float(pred_yaw_observable[i]) <= float(args.yaw_observable_threshold)),
                    "pred_yaw_dir_x": float(pred_yaw_dir_x[i]),
                    "pred_yaw_dir_y": float(pred_yaw_dir_y[i]),
                    "tgt_dx": float(record.get("label_dx", 0.0)),
                    "tgt_dy": float(record.get("label_dy", 0.0)),
                    "tgt_dz": float(record.get("label_descend_amount", record.get("label_dz", 0.0))),
                    "tgt_dyaw": float(record.get("label_dyaw", 0.0)),
                    "tgt_xy_norm": float(np.linalg.norm(np.asarray([record.get("label_dx", 0.0), record.get("label_dy", 0.0)], dtype=np.float64))),
                    "tgt_ready": float(record.get("ready_to_close", 0.0)),
                    "rgb_path": str(record.get("rgb_path", "")),
                }
                traces_by_ep[ep].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_episode = {}
    for ep, rows in sorted(traces_by_ep.items()):
        rows = sorted(rows, key=lambda r: r["step_idx"])
        rows = _apply_continuous_yaw_gate(
            rows,
            threshold=args.yaw_observable_threshold,
            stability_steps=args.yaw_observable_stability_steps,
            geometry_score_threshold=args.yaw_geometry_min_score,
            geometry_confidence_threshold=args.yaw_geometry_min_confidence,
            geometry_abs_rel_yaw_threshold=args.yaw_geometry_max_abs_rel_yaw,
        )
        trace_path = args.output_dir / f"ep{ep:03d}_grasp_skill_shadow_trace.jsonl"
        with open(trace_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        xyyaw_rows = [r for r in rows if r["xyyaw_supervision_mask"]]
        pred_dx = [r["pred_dx"] for r in xyyaw_rows]
        pred_dy = [r["pred_dy"] for r in xyyaw_rows]
        pred_dyaw = [r["pred_dyaw"] for r in xyyaw_rows]
        tgt_dx = [r["tgt_dx"] for r in xyyaw_rows]
        tgt_dy = [r["tgt_dy"] for r in xyyaw_rows]
        tgt_dyaw = [r["tgt_dyaw"] for r in xyyaw_rows]
        frame_rel_u = [r["frame_rel_crop_u"] for r in xyyaw_rows]
        frame_rel_v = [r["frame_rel_crop_v"] for r in xyyaw_rows]
        frame_rel_yaw = [r["frame_rel_yaw"] for r in xyyaw_rows]
        pred_xy_norm = [r["pred_xy_norm"] for r in xyyaw_rows]
        tgt_xy_norm = [r["tgt_xy_norm"] for r in xyyaw_rows]
        yaw_obs_target = [float(r["yaw_observable_target"]) for r in xyyaw_rows]
        pred_yaw_obs = [float(r["pred_yaw_observable"]) for r in xyyaw_rows]
        pred_active_rows = [r for r in xyyaw_rows if r["pred_yaw_active"]]
        pred_active_pred_dyaw = [r["pred_dyaw"] for r in pred_active_rows]
        pred_active_tgt_dyaw = [r["tgt_dyaw"] for r in pred_active_rows]
        target_active_rows = [r for r in xyyaw_rows if r["yaw_observable_target"]]
        target_active_pred_dyaw = [r["pred_dyaw"] for r in target_active_rows]
        target_active_tgt_dyaw = [r["tgt_dyaw"] for r in target_active_rows]
        both_active_rows = [r for r in xyyaw_rows if r["pred_yaw_active"] and r["yaw_observable_target"]]
        both_active_pred_dyaw = [r["pred_dyaw"] for r in both_active_rows]
        both_active_tgt_dyaw = [r["tgt_dyaw"] for r in both_active_rows]
        fixed_bias_warning = bool(
            len(xyyaw_rows) >= 3
            and (
                (np.std(np.asarray(pred_dx, dtype=np.float64)) < 0.15 * max(float(np.std(np.asarray(tgt_dx, dtype=np.float64))), 1e-9))
                or (np.std(np.asarray(pred_dy, dtype=np.float64)) < 0.15 * max(float(np.std(np.asarray(tgt_dy, dtype=np.float64))), 1e-9))
            )
        )
        per_episode[str(ep)] = {
            "trace_path": str(trace_path),
            "num_rows": len(rows),
            "num_xyyaw_rows": len(xyyaw_rows),
            "pred_dx_std": float(np.std(np.asarray(pred_dx, dtype=np.float64))) if pred_dx else float("nan"),
            "pred_dy_std": float(np.std(np.asarray(pred_dy, dtype=np.float64))) if pred_dy else float("nan"),
            "pred_dyaw_std": float(np.std(np.asarray(pred_dyaw, dtype=np.float64))) if pred_dyaw else float("nan"),
            "corr_pred_tgt_dx": _safe_corr(pred_dx, tgt_dx),
            "corr_pred_tgt_dy": _safe_corr(pred_dy, tgt_dy),
            "corr_pred_tgt_dyaw": _safe_corr(pred_dyaw, tgt_dyaw),
            "corr_pred_tgt_dyaw_pred_gated": _safe_corr(pred_active_pred_dyaw, pred_active_tgt_dyaw),
            "corr_pred_tgt_dyaw_target_active": _safe_corr(target_active_pred_dyaw, target_active_tgt_dyaw),
            "corr_pred_tgt_dyaw_both_active": _safe_corr(both_active_pred_dyaw, both_active_tgt_dyaw),
            "corr_pred_frame_u": _safe_corr(pred_dx, frame_rel_u),
            "corr_pred_frame_v": _safe_corr(pred_dy, frame_rel_v),
            "corr_pred_frame_yaw": _safe_corr(pred_dyaw, frame_rel_yaw),
            "pred_yaw_observable_mean": float(np.mean(np.asarray(pred_yaw_obs, dtype=np.float64))) if pred_yaw_obs else float("nan"),
            "yaw_observable_target_rate": float(np.mean(np.asarray(yaw_obs_target, dtype=np.float64))) if yaw_obs_target else float("nan"),
            "pred_yaw_active_rate": float(len(pred_active_rows) / max(len(xyyaw_rows), 1)),
            "pred_yaw_active_stable_rate": float(np.mean([float(r["pred_yaw_active_stable"]) for r in xyyaw_rows])) if xyyaw_rows else float("nan"),
            "frame_geometry_consistent_rate": float(np.mean([float(r["frame_geometry_consistent"]) for r in xyyaw_rows])) if xyyaw_rows else float("nan"),
            "num_pred_yaw_active": len(pred_active_rows),
            "num_pred_yaw_active_stable": len([r for r in xyyaw_rows if r.get("pred_yaw_active_stable")]),
            "num_target_yaw_active": len(target_active_rows),
            "num_both_yaw_active": len(both_active_rows),
            "pred_xy_norm_p50": float(np.median(np.asarray(pred_xy_norm, dtype=np.float64))) if pred_xy_norm else float("nan"),
            "tgt_xy_norm_p50": float(np.median(np.asarray(tgt_xy_norm, dtype=np.float64))) if tgt_xy_norm else float("nan"),
            "fixed_bias_warning": fixed_bias_warning,
        }

    overall = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "episode_indices": sorted(int(x) for x in episode_indices),
        "yaw_observable_threshold": float(args.yaw_observable_threshold),
        "yaw_observable_stability_steps": int(args.yaw_observable_stability_steps),
        "yaw_geometry_min_score": float(args.yaw_geometry_min_score),
        "yaw_geometry_min_confidence": float(args.yaw_geometry_min_confidence),
        "yaw_geometry_max_abs_rel_yaw": float(args.yaw_geometry_max_abs_rel_yaw),
        "yaw_primary_metric_name": "yaw_axis_cosine_mean",
        "episodes": per_episode,
        "note": "Episode-ordered shadow traces for grasp skill. Use pred-gated yaw metrics to judge whether the model abstains on low-observability frames and only varies dyaw on frames it considers observable. pred_yaw_active requires threshold plus continuous stability, then a geometry-consistency intersection gate.",
    }
    summary_path = args.output_dir / "shadow_trace_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
