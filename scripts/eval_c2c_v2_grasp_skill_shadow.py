#!/usr/bin/env python3
"""Offline shadow evaluation for C2C v2 GraspSkillHead."""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import DepthLocalizerJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_localizer import load_grasp_skill_head_checkpoint


def _wrap_yaw_np(x: np.ndarray, period: float = math.pi / 2.0) -> np.ndarray:
    return np.remainder(x + 0.5 * period, period) - 0.5 * period


def _yaw_to_vec_np(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)


def _filter_records(records: list[dict]) -> list[dict]:
    filtered = []
    for r in records:
        if str(r.get("view_name", "")) != "wrist":
            continue
        if float(r.get("xyyaw_supervision_mask", 0.0)) <= 0.0 and float(r.get("z_supervision_mask", 0.0)) <= 0.0 and float(r.get("ready_supervision_mask", 0.0)) <= 0.0:
            continue
        filtered.append(r)
    return filtered


def _split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    episodes = sorted({int(r["episode_idx"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val_eps = set(episodes[:n_val])
    return [r for r in records if int(r["episode_idx"]) not in val_eps], [r for r in records if int(r["episode_idx"]) in val_eps]


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
    labels = torch.tensor(
        [
            [
                float(item.get("label_dx", 0.0)),
                float(item.get("label_dy", 0.0)),
                float(item.get("label_descend_amount", item.get("label_dz", 0.0))),
                float(item.get("label_dyaw", 0.0)),
                float(item.get("ready_to_close", 0.0)),
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
                float(item.get("z_supervision_mask", 1.0)),
                float(item.get("ready_supervision_mask", 0.0)),
                float(item.get("near_grasp_basin", 0.0)),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    return {"image": image, "frame": frame, "proprio": proprio, "labels": labels, "masks": masks, "records": batch}


def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


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
    sorted_rows = sorted(rows, key=lambda r: (int(r["episode_idx"]), int(r["step_idx"])))
    streak_by_episode: dict[int, int] = defaultdict(int)
    for row in sorted_rows:
        ep = int(row["episode_idx"])
        pred_obs = float(row.get("pred_yaw_observable", 0.0))
        if pred_obs > float(threshold):
            streak_by_episode[ep] = streak_by_episode[ep] + 1
        else:
            streak_by_episode[ep] = 0
        row["pred_yaw_stable_count"] = int(streak_by_episode[ep])
        row["pred_yaw_active_stable"] = bool(streak_by_episode[ep] >= stability_steps)
        geom_score = float(row.get("yaw_observability_score", 0.0))
        geom_conf = float(row.get("frame_confidence", 0.0))
        geom_rel_yaw = abs(float(row.get("frame_rel_yaw", 0.0)))
        row["frame_geometry_consistent"] = bool(
            geom_score >= float(geometry_score_threshold)
            and geom_conf >= float(geometry_confidence_threshold)
            and geom_rel_yaw <= float(geometry_abs_rel_yaw_threshold)
        )
        row["pred_yaw_active"] = bool(row["pred_yaw_active_stable"] and row["frame_geometry_consistent"])
        row["pred_yaw_active_pregeom"] = bool(row["pred_yaw_active_stable"])
    return sorted_rows


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_skill_shadow_eval.json"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--yaw_observable_threshold", type=float, default=0.7)
    ap.add_argument("--yaw_observable_stability_steps", type=int, default=2)
    ap.add_argument("--yaw_geometry_min_score", type=float, default=0.60)
    ap.add_argument("--yaw_geometry_min_confidence", type=float, default=0.33)
    ap.add_argument("--yaw_geometry_max_abs_rel_yaw", type=float, default=0.02)
    args = ap.parse_args()

    full = DepthLocalizerJsonlDataset(args.dataset)
    filtered = _filter_records(full.records)
    _, val_records = _split(filtered, args.val_fraction, args.seed)
    val_ds = DepthLocalizerJsonlDataset(args.dataset, records=val_records)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    model, _ = load_grasp_skill_head_checkpoint(args.checkpoint, map_location=args.device)
    model = model.to(args.device)
    model.eval()

    rows: list[dict] = []

    for batch in loader:
        out = model(batch["image"].to(args.device), batch["frame"].to(args.device), batch["proprio"].to(args.device))
        pred_dx = out["dx"].detach().cpu().numpy()
        pred_dy = out["dy"].detach().cpu().numpy()
        pred_dz = out["dz"].detach().cpu().numpy()
        pred_dyaw = out["dyaw"].detach().cpu().numpy()
        pred_yaw_dir_x = out["yaw_dir_x"].detach().cpu().numpy()
        pred_yaw_dir_y = out["yaw_dir_y"].detach().cpu().numpy()
        pred_yaw_observable = torch.sigmoid(out["yaw_observable_logit"]).detach().cpu().numpy()
        pred_ready = torch.sigmoid(out["ready_to_close_logit"]).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        masks = batch["masks"].detach().cpu().numpy()
        for i, record in enumerate(batch["records"]):
            rows.append(
                {
                    "episode_idx": int(record.get("episode_idx", -1)),
                    "step_idx": int(record.get("step_idx", -1)),
                    "steps_to_close": int(record.get("steps_to_close", -1)),
                    "near_grasp_basin": float(masks[i, 4]),
                    "xyyaw_mask": float(masks[i, 0]),
                    "yaw_observable_target": float(masks[i, 1]),
                    "z_mask": float(masks[i, 2]),
                    "ready_mask": float(masks[i, 3]),
                    "pred_dx": float(pred_dx[i]),
                    "pred_dy": float(pred_dy[i]),
                    "pred_dz": float(pred_dz[i]),
                    "pred_dyaw": float(pred_dyaw[i]),
                    "pred_yaw_observable": float(pred_yaw_observable[i]),
                    "pred_ready": float(pred_ready[i]),
                    "pred_yaw_dir_x": float(pred_yaw_dir_x[i]),
                    "pred_yaw_dir_y": float(pred_yaw_dir_y[i]),
                    "tgt_dx": float(labels[i, 0]),
                    "tgt_dy": float(labels[i, 1]),
                    "tgt_dz": float(labels[i, 2]),
                    "tgt_dyaw": float(labels[i, 3]),
                    "tgt_ready": float(labels[i, 4]),
                }
            )

    rows = _apply_continuous_yaw_gate(
        rows,
        threshold=args.yaw_observable_threshold,
        stability_steps=args.yaw_observable_stability_steps,
        geometry_score_threshold=args.yaw_geometry_min_score,
        geometry_confidence_threshold=args.yaw_geometry_min_confidence,
        geometry_abs_rel_yaw_threshold=args.yaw_geometry_max_abs_rel_yaw,
    )

    xy_cosines = []
    descend_l1 = []
    yaw_sign_match = []
    yaw_axis_cosines = []
    yaw_sign_match_gated = []
    yaw_axis_cosines_gated = []
    yaw_obs_acc = []
    yaw_obs_prec = []
    yaw_obs_recall = []
    yaw_obs_pred_rate = []
    yaw_obs_pred_rate_pregeom = []
    yaw_obs_tgt_rate = []
    yaw_obs_tp = 0.0
    yaw_obs_fp = 0.0
    yaw_obs_fn = 0.0
    ready_probs = []
    ready_targets = []
    near_grasp_basin_rate = []
    step_buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        near_grasp_basin_rate.append(float(row["near_grasp_basin"]))
        if row["xyyaw_mask"] > 0.5:
            tgt_xy = np.asarray([row["tgt_dx"], row["tgt_dy"]], dtype=np.float64)
            pred_xy = np.asarray([row["pred_dx"], row["pred_dy"]], dtype=np.float64)
            cos = _safe_cosine(pred_xy, tgt_xy)
            if np.isfinite(cos):
                xy_cosines.append(cos)
            tgt_obs = float(row["yaw_observable_target"]) > 0.5
            pred_obs = bool(row["pred_yaw_active"]) if "pred_yaw_active" in row else False
            pred_obs_pregeom = bool(row.get("pred_yaw_active_stable", False))
            geom_consistent = bool(row.get("frame_geometry_consistent", False))
            yaw_obs_acc.append(float(pred_obs == tgt_obs))
            yaw_obs_pred_rate.append(float(pred_obs))
            yaw_obs_pred_rate_pregeom.append(float(pred_obs_pregeom))
            yaw_obs_tgt_rate.append(float(tgt_obs))
            yaw_obs_tp += float(pred_obs and tgt_obs)
            yaw_obs_fp += float(pred_obs and (not tgt_obs))
            yaw_obs_fn += float((not pred_obs) and tgt_obs)
        if row["yaw_observable_target"] > 0.5:
            pred_yaw_vec = np.asarray([row["pred_yaw_dir_x"], row["pred_yaw_dir_y"]], dtype=np.float64)
            tgt_yaw_vec = _yaw_to_vec_np(np.asarray([row["tgt_dyaw"]], dtype=np.float64))[0]
            yaw_axis_cos = _safe_cosine(pred_yaw_vec, tgt_yaw_vec)
            if np.isfinite(yaw_axis_cos):
                yaw_axis_cosines.append(yaw_axis_cos)
                if bool(row["pred_yaw_active"]):
                    yaw_axis_cosines_gated.append(yaw_axis_cos)
        if row["z_mask"] > 0.5:
            descend_l1.append(float(abs(row["pred_dz"] - row["tgt_dz"])))
        if row["yaw_observable_target"] > 0.5 and abs(row["tgt_dyaw"]) > 1e-6:
            sign_match = float(np.sign(row["pred_dyaw"]) == np.sign(row["tgt_dyaw"]))
            yaw_sign_match.append(sign_match)
            if bool(row["pred_yaw_active"]):
                yaw_sign_match_gated.append(sign_match)
        if row["ready_mask"] > 0.5:
            ready_probs.append(float(row["pred_ready"]))
            ready_targets.append(float(row["tgt_ready"]))
        step = int(row["steps_to_close"])
        step_buckets[step]["pred_dx"].append(float(row["pred_dx"]))
        step_buckets[step]["pred_dy"].append(float(row["pred_dy"]))
        step_buckets[step]["pred_dz"].append(float(row["pred_dz"]))
        step_buckets[step]["pred_dyaw"].append(float(row["pred_dyaw"]))
        step_buckets[step]["pred_yaw_observable"].append(float(row["pred_yaw_observable"]))
        step_buckets[step]["pred_yaw_active"].append(float(bool(row["pred_yaw_active"])))
        step_buckets[step]["pred_yaw_stable_count"].append(float(row["pred_yaw_stable_count"]))
        step_buckets[step]["pred_ready"].append(float(row["pred_ready"]))
        step_buckets[step]["tgt_dx"].append(float(row["tgt_dx"]))
        step_buckets[step]["tgt_dy"].append(float(row["tgt_dy"]))
        step_buckets[step]["tgt_dz"].append(float(row["tgt_dz"]))
        step_buckets[step]["tgt_dyaw"].append(float(row["tgt_dyaw"]))
        step_buckets[step]["tgt_ready"].append(float(row["tgt_ready"]))
        step_buckets[step]["xyyaw_mask"].append(float(row["xyyaw_mask"]))
        step_buckets[step]["yaw_observable_target"].append(float(row["yaw_observable_target"]))
        step_buckets[step]["z_mask"].append(float(row["z_mask"]))
        step_buckets[step]["ready_mask"].append(float(row["ready_mask"]))
        step_buckets[step]["near_grasp_basin"].append(float(row["near_grasp_basin"]))

    bucket_summary = {}
    for step, stats in sorted(step_buckets.items()):
        bucket_summary[str(step)] = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    ready_targets_np = np.asarray(ready_targets, dtype=np.float32)
    ready_probs_np = np.asarray(ready_probs, dtype=np.float32)
    pred_yaw_active_rate = float(np.mean([float(r["pred_yaw_active"]) for r in rows if r["xyyaw_mask"] > 0.5])) if rows else 0.0
    pred_yaw_active_stable_rate = float(np.mean([float(r["pred_yaw_active_stable"]) for r in rows if r["xyyaw_mask"] > 0.5])) if rows else 0.0
    frame_geometry_consistent_rate = float(np.mean([float(r["frame_geometry_consistent"]) for r in rows if r["xyyaw_mask"] > 0.5])) if rows else 0.0
    pred_yaw_prob_mean = float(np.mean([float(r["pred_yaw_observable"]) for r in rows if r["xyyaw_mask"] > 0.5])) if rows else 0.0
    report = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "num_val": len(val_records),
        "yaw_observable_threshold": float(args.yaw_observable_threshold),
        "yaw_observable_stability_steps": int(args.yaw_observable_stability_steps),
        "yaw_primary_metric_name": "yaw_axis_cosine_mean",
        "yaw_primary_metric_value": float(np.mean(yaw_axis_cosines_gated)) if yaw_axis_cosines_gated else float("nan"),
        "xy_direction_cosine_mean": float(np.nanmean(np.asarray(xy_cosines, dtype=np.float32))) if xy_cosines else float("nan"),
        "xy_direction_cosine_p50": float(np.nanmedian(np.asarray(xy_cosines, dtype=np.float32))) if xy_cosines else float("nan"),
        "descend_amount_l1_mean": float(np.mean(descend_l1)) if descend_l1 else float("nan"),
        "yaw_axis_cosine_mean": float(np.mean(yaw_axis_cosines)) if yaw_axis_cosines else float("nan"),
        "yaw_axis_cosine_mean_gated": float(np.mean(yaw_axis_cosines_gated)) if yaw_axis_cosines_gated else float("nan"),
        "yaw_observable_accuracy": float(np.mean(yaw_obs_acc)) if yaw_obs_acc else float("nan"),
        "yaw_observable_precision": float(yaw_obs_tp / max(yaw_obs_tp + yaw_obs_fp, 1.0)),
        "yaw_observable_recall": float(yaw_obs_tp / max(yaw_obs_tp + yaw_obs_fn, 1.0)),
        "yaw_observable_pred_rate": pred_yaw_active_rate,
        "yaw_observable_pred_rate_stable": pred_yaw_active_stable_rate,
        "yaw_geometry_consistent_rate": frame_geometry_consistent_rate,
        "yaw_observable_pred_rate_pregeom": float(np.mean(yaw_obs_pred_rate_pregeom)) if yaw_obs_pred_rate_pregeom else float("nan"),
        "yaw_observable_prob_mean": pred_yaw_prob_mean,
        "yaw_observable_target_rate": float(np.mean(yaw_obs_tgt_rate)) if yaw_obs_tgt_rate else float("nan"),
        "yaw_sign_match_rate_secondary": float(np.mean(yaw_sign_match)) if yaw_sign_match else float("nan"),
        "yaw_sign_match_rate_gated": float(np.mean(yaw_sign_match_gated)) if yaw_sign_match_gated else float("nan"),
        "yaw_sign_match_rate": float(np.mean(yaw_sign_match)) if yaw_sign_match else float("nan"),
        "yaw_geometry_min_score": float(args.yaw_geometry_min_score),
        "yaw_geometry_min_confidence": float(args.yaw_geometry_min_confidence),
        "yaw_geometry_max_abs_rel_yaw": float(args.yaw_geometry_max_abs_rel_yaw),
        "yaw_observable_threshold_geom": float(args.yaw_geometry_min_score),
        "yaw_observable_min_confidence_geom": float(args.yaw_geometry_min_confidence),
        "yaw_observable_max_abs_rel_yaw_geom": float(args.yaw_geometry_max_abs_rel_yaw),
        "ready_target_positive_rate": float(np.mean(ready_targets_np > 0.5)) if ready_targets else 0.0,
        "ready_pred_mean": float(np.mean(ready_probs_np)) if ready_probs else 0.0,
        "ready_pred_p95": float(np.percentile(ready_probs_np, 95)) if ready_probs else 0.0,
        "ready_label_degenerate": bool(np.all(ready_targets_np < 0.5)) if ready_targets else True,
        "near_grasp_basin_rate": float(np.mean(np.asarray(near_grasp_basin_rate, dtype=np.float32))) if near_grasp_basin_rate else 0.0,
        "steps_to_close_bucket_means": bucket_summary,
        "decision_note": "Use yaw_axis_cosine_mean together with yaw_observable_precision/recall. yaw_sign_match_rate is secondary only; near-zero and low-observability yaw frames should prefer abstain over forced control. pred_yaw_active now requires a threshold plus continuous stability, and then a geometry-consistency intersection gate.",
        "note": "Shadow-only offline comparison. If ready labels are degenerate, treat ready_to_close as diagnostic-invalid until dataset semantics are repaired.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
