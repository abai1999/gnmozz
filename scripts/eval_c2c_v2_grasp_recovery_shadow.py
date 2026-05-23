#!/usr/bin/env python3
"""Offline recovery-capability shadow evaluation for Coarse2Contact v2 grasp recovery traces."""

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
from prismatic.robot.coarse2contact_v2.learned_localizer import load_grasp_recovery_checkpoint, load_grasp_skill_head_checkpoint


def _safe_corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


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
    masks = torch.tensor(
        [
            [
                float(item.get("planner_bias_score", 0.0)),
                float(item.get("recovery_needed", 0.0)),
                float(item.get("frame_observability", 0.0)),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    return {"image": image, "frame": frame, "proprio": proprio, "planner_prior": planner_prior, "labels": labels, "masks": masks, "records": batch}


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
        if float(r.get("trace_error_confidence", r.get("label_confidence", 0.0))) <= 0.0:
            continue
        filtered.append(r)
    return filtered


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_shadow_eval.json"))
    ap.add_argument("--trace_output", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_shadow_trace.jsonl"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--large_bias_quantile", type=float, default=0.70)
    ap.add_argument("--model_kind", type=str, default="auto", choices=["auto", "grasp_skill", "recovery"])
    args = ap.parse_args()

    full = DepthLocalizerJsonlDataset(args.dataset)
    filtered = _filter_records(full.records)
    train_records, val_records = _split_by_episode(filtered, args.val_fraction, args.seed)
    val_ds = DepthLocalizerJsonlDataset(args.dataset, records=val_records)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_kind = str(args.model_kind)
    if model_kind == "auto":
        model_kind = str(ckpt.get("config", {}).get("model_type", "grasp_skill"))
    if model_kind == "grasp_recovery" or model_kind == "recovery":
        model, _ = load_grasp_recovery_checkpoint(args.checkpoint, map_location=args.device)
    else:
        model, _ = load_grasp_skill_head_checkpoint(args.checkpoint, map_location=args.device)
    model = model.to(args.device)
    model.eval()
    is_recovery_model = model_kind in {"grasp_recovery", "recovery"}

    rows: list[dict] = []
    trajectory_rows: dict[str, list[dict]] = defaultdict(list)
    for batch in loader:
        if model_kind in {"grasp_recovery", "recovery"}:
            out = model(
                batch["image"].to(args.device),
                batch["frame"].to(args.device),
                batch["proprio"].to(args.device),
                batch["planner_prior"].to(args.device),
            )
        else:
            out = model(batch["image"].to(args.device), batch["frame"].to(args.device), batch["proprio"].to(args.device))
        pred_dx = out["dx"].detach().cpu().numpy()
        pred_dy = out["dy"].detach().cpu().numpy()
        pred_dyaw = out["dyaw"].detach().cpu().numpy()
        pred_conf = torch.sigmoid(out["confidence_logit"]).detach().cpu().numpy()
        if is_recovery_model:
            pred_ready = pred_conf.copy()
            pred_yaw_obs = np.full_like(pred_conf, np.nan, dtype=np.float32)
        else:
            pred_ready = torch.sigmoid(out["ready_to_close_logit"]).detach().cpu().numpy()
            pred_yaw_obs = torch.sigmoid(out["yaw_observable_logit"]).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        masks = batch["masks"].detach().cpu().numpy()
        for i, record in enumerate(batch["records"]):
            error = labels[i]
            pred = np.array([pred_dx[i], pred_dy[i], pred_dyaw[i]], dtype=np.float32)
            error_xy = error[:2]
            pred_xy = pred[:2]
            error_norm = float(np.linalg.norm(np.array([error[0], error[1], 0.04 * error[2]], dtype=np.float32)))
            post_error = error - pred
            post_norm = float(np.linalg.norm(np.array([post_error[0], post_error[1], 0.04 * post_error[2]], dtype=np.float32)))
            gain = float(error_norm - post_norm)
            row = {
                "episode_idx": int(record.get("episode_idx", -1)),
                "step_idx": int(record.get("step_idx", -1)),
                "trajectory_id": str(record.get("trajectory_id", f"ep{int(record.get('episode_idx', -1)):03d}")),
                "trajectory_step": int(record.get("trajectory_step", 0)),
                "trajectory_len": int(record.get("trajectory_len", 1)),
                "trajectory_phase": str(record.get("trajectory_phase", "")),
                "phase_name": str(record.get("phase_name", "")),
                "planner_bias_score": float(record.get("planner_bias_score", 0.0)),
                "planner_bias_xy": float(record.get("planner_bias_xy", 0.0)),
                "planner_bias_yaw": float(record.get("planner_bias_yaw", 0.0)),
                "planner_bias_dyaw": float(record.get("planner_bias_dyaw", 0.0)),
                "trace_error_valid": bool(record.get("trace_error_valid", False)),
                "trace_error_confidence": float(record.get("trace_error_confidence", 0.0)),
                "trace_error_observability": float(record.get("trace_error_observability", 0.0)),
                "trace_error_dx": float(error[0]),
                "trace_error_dy": float(error[1]),
                "trace_error_dyaw": float(error[2]),
                "pred_dx": float(pred[0]),
                "pred_dy": float(pred[1]),
                "pred_dyaw": float(pred[2]),
                "pred_ready": float(pred_ready[i]),
                "pred_confidence": float(pred_conf[i]),
                "pred_yaw_observable": float(pred_yaw_obs[i]),
                "planner_prior_dx": float(batch["planner_prior"][i, 0].item()),
                "planner_prior_dy": float(batch["planner_prior"][i, 1].item()),
                "planner_prior_dyaw": float(batch["planner_prior"][i, 5].item()),
                "recovery_gain": gain,
                "recovery_error_norm": error_norm,
                "recovery_post_norm": post_norm,
                "recovery_improved": bool(gain > 0.0),
                "xy_cosine": _safe_cosine(pred_xy, error_xy),
                "yaw_sign_match": bool(np.sign(float(pred[2])) == np.sign(float(error[2])) or abs(float(error[2])) < 1e-6 or abs(float(pred[2])) < 1e-6),
                "error_norm_xy": float(np.linalg.norm(error_xy)),
                "pred_norm_xy": float(np.linalg.norm(pred_xy)),
                "error_abs_yaw": float(abs(error[2])),
                "pred_abs_yaw": float(abs(pred[2])),
                "recovery_target_dx": float(record.get("recovery_target_dx", error[0])),
                "recovery_target_dy": float(record.get("recovery_target_dy", error[1])),
                "recovery_target_dyaw": float(record.get("recovery_target_dyaw", error[2])),
                "recovery_target_kind": str(record.get("recovery_target_kind", "trace_local_geometry_error")),
                "planner_bias_source": str(record.get("recovery_bias_source", "")),
                "source_trace_path": str(record.get("source_trace_path", "")),
                "mp4_path": str(record.get("mp4_path", "")),
                "rgb_path": str(record.get("rgb_path", "")),
                "depth_path": str(record.get("depth_path", "")),
                "roi_box": list(record.get("roi_box", [])),
                "frame_confidence": float(record.get("frame_confidence", 0.0)),
                "frame_observability": float(record.get("frame_observability", 0.0)),
                "frame_axis_strength": float(record.get("frame_axis_strength", 0.0)),
                "frame_completeness": float(record.get("frame_completeness", 0.0)),
                "uses_privileged_target": False,
                "uses_rlbench_mask_runtime": False,
            }
            rows.append(row)
            trajectory_rows[row["trajectory_id"]].append(row)

    if not rows:
        raise RuntimeError("No recovery shadow rows available after filtering")

    bias_scores = np.asarray([float(r["planner_bias_score"]) for r in rows], dtype=np.float32)
    large_bias_threshold = float(np.quantile(bias_scores, float(np.clip(args.large_bias_quantile, 0.0, 1.0))))
    large_bias_rows = [r for r in rows if float(r["planner_bias_score"]) >= large_bias_threshold]

    def _summarize(subset: list[dict]) -> dict:
        if not subset:
            return {}
        gains = np.asarray([r["recovery_gain"] for r in subset], dtype=np.float32)
        xy_cos = np.asarray([r["xy_cosine"] for r in subset if np.isfinite(r["xy_cosine"])], dtype=np.float32)
        pred_dx = np.asarray([r["pred_dx"] for r in subset], dtype=np.float32)
        pred_dy = np.asarray([r["pred_dy"] for r in subset], dtype=np.float32)
        pred_dyaw = np.asarray([r["pred_dyaw"] for r in subset], dtype=np.float32)
        tgt_dx = np.asarray([r["trace_error_dx"] for r in subset], dtype=np.float32)
        tgt_dy = np.asarray([r["trace_error_dy"] for r in subset], dtype=np.float32)
        tgt_dyaw = np.asarray([r["trace_error_dyaw"] for r in subset], dtype=np.float32)
        return {
            "count": int(len(subset)),
            "recovery_gain_mean": float(np.mean(gains)),
            "recovery_gain_median": float(np.median(gains)),
            "recovery_improved_rate": float(np.mean(gains > 0.0)),
            "xy_cosine_mean": float(np.mean(xy_cos)) if xy_cos.size else float("nan"),
            "xy_cosine_median": float(np.median(xy_cos)) if xy_cos.size else float("nan"),
            "pred_dx_mean": float(np.mean(pred_dx)),
            "pred_dy_mean": float(np.mean(pred_dy)),
            "pred_dyaw_mean": float(np.mean(pred_dyaw)),
            "tgt_dx_mean": float(np.mean(tgt_dx)),
            "tgt_dy_mean": float(np.mean(tgt_dy)),
            "tgt_dyaw_mean": float(np.mean(tgt_dyaw)),
            "yaw_sign_match_rate": float(np.mean([bool(r["yaw_sign_match"]) for r in subset])),
        }

    trajectory_summaries = []
    for traj_id, traj in sorted(trajectory_rows.items()):
        traj = sorted(traj, key=lambda r: int(r["trajectory_step"]))
        error_norms = np.asarray([r["recovery_error_norm"] for r in traj], dtype=np.float32)
        post_norms = np.asarray([r["recovery_post_norm"] for r in traj], dtype=np.float32)
        gains = error_norms - post_norms
        trajectory_summaries.append(
            {
                "trajectory_id": traj_id,
                "episode_idx": int(traj[0]["episode_idx"]),
                "num_steps": int(len(traj)),
                "error_norm_start": float(error_norms[0]),
                "error_norm_end": float(error_norms[-1]),
                "error_norm_min": float(np.min(error_norms)),
                "error_norm_max": float(np.max(error_norms)),
                "gain_mean": float(np.mean(gains)),
                "gain_rate": float(np.mean(gains > 0.0)),
                "bias_score_max": float(np.max([r["planner_bias_score"] for r in traj])),
                "bias_score_start": float(traj[0]["planner_bias_score"]),
                "bias_score_end": float(traj[-1]["planner_bias_score"]),
                "trajectory_improved": bool(float(error_norms[-1]) < float(error_norms[0])),
            }
        )

    trajectory_improved_rate = float(np.mean([t["trajectory_improved"] for t in trajectory_summaries])) if trajectory_summaries else 0.0
    loss_like = float(np.mean([r["recovery_post_norm"] for r in rows]))
    report = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "num_rows": int(len(rows)),
        "num_trajectories": int(len(trajectory_summaries)),
        "large_bias_quantile": float(args.large_bias_quantile),
        "large_bias_threshold": float(large_bias_threshold),
        "summary_all": _summarize(rows),
        "summary_large_bias": _summarize(large_bias_rows),
        "trajectory_improved_rate": trajectory_improved_rate,
        "trajectory_summaries": trajectory_summaries,
        "loss_like_post_norm_mean": loss_like,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.trace_output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(args.output)
    print(args.trace_output)


if __name__ == "__main__":
    main()
