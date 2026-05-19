#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismatic.models.depth_force_local_proposal_policy import DepthForceLocalProposalPolicy
from prismatic.vla.datasets.depth_force_local_proposal_dataset import DepthForceLocalProposalDataset


ACTION_SCALE = torch.tensor([0.008, 0.008, 0.006, 0.06, 0.06, 0.12], dtype=torch.float32)
YAW_BUCKETS = np.asarray([0.01, 0.05, 0.09], dtype=np.float32)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("prismatic"):
        return {k: _jsonable(v) for k, v in obj.__dict__.items()}
    return obj


def _weighted_l1(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=a.device, dtype=a.dtype).view(*([1] * (a.ndim - 1)), -1)
    return torch.sum(torch.abs((a - b) / torch.clamp(scale, min=1e-6)), dim=-1)


def _stats(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _group_summary(rows: np.ndarray, metrics: dict[str, np.ndarray]) -> dict[str, float]:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return {"rows": 0}
    out = {"rows": int(idx.size)}
    for key, arr in metrics.items():
        out[key] = float(np.mean(np.asarray(arr)[idx]))
    return out


@torch.no_grad()
def _predict(model: DepthForceLocalProposalPolicy, loader: DataLoader, *, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    proposals_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    target_best_safe_all: list[np.ndarray] = []
    target_geom_all: list[np.ndarray] = []
    target_best_safe_idx_all: list[np.ndarray] = []
    row_idx_all: list[np.ndarray] = []
    target_yaw_all: list[np.ndarray] = []

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(
            front_rgb=batch["front_rgb"],
            wrist_rgb=batch["wrist_rgb"],
            wrist_depth=batch["wrist_depth"],
            force_history=batch["force_history"],
            proprio=batch["proprio"],
            planner_base_action_local=batch["planner_base_action_local"],
            stage_token=batch.get("stage_token"),
            contact_phase=batch.get("contact_phase"),
            depth_proximity=batch.get("depth_proximity"),
            gripper_state=batch.get("gripper_state"),
        )
        proposals_all.append(outputs["proposal_actions_local"].detach().cpu().numpy())
        scores_all.append(outputs["proposal_scores"].detach().cpu().numpy())
        target_best_safe_all.append(batch["proposal_safe_target_action_local"].detach().cpu().numpy())
        target_geom_all.append(batch["proposal_actions_local"][torch.arange(batch["proposal_actions_local"].shape[0], device=device), batch["proposal_best_safe_index"].long()].detach().cpu().numpy())
        target_best_safe_idx_all.append(batch["proposal_best_safe_index"].detach().cpu().numpy())
        row_idx_all.append(batch["row_index"].detach().cpu().numpy())
        target_yaw_all.append(batch["proposal_target_delta_local"][:, 5].detach().cpu().numpy())

    return {
        "proposal_actions": np.concatenate(proposals_all, axis=0),
        "proposal_scores": np.concatenate(scores_all, axis=0),
        "target_best_safe_action": np.concatenate(target_best_safe_all, axis=0),
        "target_geom_action": np.concatenate(target_geom_all, axis=0),
        "target_best_safe_index": np.concatenate(target_best_safe_idx_all, axis=0),
        "row_index": np.concatenate(row_idx_all, axis=0),
        "target_yaw": np.concatenate(target_yaw_all, axis=0),
    }


def _row_metrics(pred: np.ndarray, target: np.ndarray, scale: np.ndarray) -> dict[str, np.ndarray]:
    scale = np.asarray(scale, dtype=np.float32).reshape(1, 1, 6)
    pred_norm = np.linalg.norm(pred, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    pred_normed = np.linalg.norm(pred / np.clip(scale, 1e-6, None), axis=-1)
    target_normed = np.linalg.norm(target / np.clip(scale[:, :1, :], 1e-6, None), axis=-1)
    pred_xyz = np.linalg.norm(pred[..., :3], axis=-1)
    target_xyz = np.linalg.norm(target[..., :3], axis=-1)
    pred_yaw = np.abs(pred[..., 5])
    target_yaw = np.abs(target[..., 5])
    pairwise = []
    collapse = np.zeros((pred.shape[0],), dtype=np.float32)
    near_zero = np.zeros((pred.shape[0],), dtype=np.float32)
    bucket_count = np.zeros((pred.shape[0],), dtype=np.float32)
    for i in range(pred.shape[0]):
        d = pred[i] / np.clip(scale[0], 1e-6, None)
        if d.shape[0] > 1:
            pdist = np.linalg.norm(d[:, None, :] - d[None, :, :], axis=-1)
            tri = pdist[np.triu_indices(d.shape[0], k=1)]
            pairwise.append(float(np.mean(tri)) if tri.size > 0 else 0.0)
            collapse[i] = float((np.mean(tri) < 0.35) if tri.size > 0 else 1.0)
        else:
            pairwise.append(0.0)
            collapse[i] = 1.0
        near_zero[i] = float(np.mean(np.linalg.norm(d, axis=-1) <= 0.25))
        yaw_bins = np.digitize(np.abs(pred[i, :, 5]), YAW_BUCKETS, right=False)
        bucket_count[i] = float(len(np.unique(yaw_bins)))
    return {
        "pred_action_norm": pred_norm,
        "pred_action_normed": pred_normed,
        "target_action_norm": target_norm,
        "target_action_normed": target_normed,
        "pred_xyz_norm": pred_xyz,
        "target_xyz_norm": target_xyz,
        "pred_yaw_abs": pred_yaw,
        "target_yaw_abs": target_yaw,
        "pred_pairwise_distance": np.asarray(pairwise, dtype=np.float32),
        "proposal_collapse_rate": collapse,
        "proposal_near_zero_rate": near_zero,
        "proposal_unique_yaw_bucket_count": bucket_count,
        "pred_to_best_safe_l2": np.linalg.norm(pred - target, axis=-1).min(axis=1).astype(np.float32),
        "pred_to_best_safe_xyz_l2": np.linalg.norm(pred[..., :3] - target[..., :3], axis=-1).min(axis=1).astype(np.float32),
        "pred_to_best_safe_yaw_l1": np.abs(pred[..., 5] - target[..., 5]).min(axis=1).astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = DepthForceLocalProposalDataset(args.dataset_npz)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    inferred_hidden = int(ckpt["model_state_dict"]["proposal_head.0.weight"].shape[0])
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    model = DepthForceLocalProposalPolicy(
        proposal_count=int(ckpt.get("proposal_count", 8)),
        state_dim=int(ckpt.get("state_dim", 384)),
        hidden_dim=int(ckpt.get("hidden_dim", inferred_hidden)),
        **model_kwargs,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model = model.to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pred = _predict(model, loader, device=device)
    scale = ACTION_SCALE.cpu().numpy()
    metrics = _row_metrics(pred["proposal_actions"], pred["target_best_safe_action"][:, None, :], scale)
    n = len(dataset)
    episodes = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    yaw_aug = np.asarray(dataset.data.get("yaw_augmentation_applied", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(dataset.data.get("yaw_opportunity_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    original = ~yaw_aug
    non_yaw = ~yaw_opp
    weak_episodes = np.isin(episodes, np.asarray([1, 8, 10, 19], dtype=np.int64))
    strong_episodes = np.isin(episodes, np.asarray([5, 16, 17, 20], dtype=np.int64))

    report = {
        "dataset_npz": str(args.dataset_npz),
        "checkpoint": str(args.checkpoint),
        "model_kwargs": _jsonable(model_kwargs),
        "rows": n,
        "all_rows": {
            "pred_action_norm_mean": float(np.mean(metrics["pred_action_norm"])),
            "pred_action_norm_p95": float(np.percentile(metrics["pred_action_norm"], 95)),
            "pred_action_normed_mean": float(np.mean(metrics["pred_action_normed"])),
            "target_action_norm_mean": float(np.mean(metrics["target_action_norm"])),
            "target_action_norm_p95": float(np.percentile(metrics["target_action_norm"], 95)),
            "target_action_normed_mean": float(np.mean(metrics["target_action_normed"])),
            "pred_xyz_norm_mean": float(np.mean(metrics["pred_xyz_norm"])),
            "target_xyz_norm_mean": float(np.mean(metrics["target_xyz_norm"])),
            "pred_yaw_abs_mean": float(np.mean(metrics["pred_yaw_abs"])),
            "target_yaw_abs_mean": float(np.mean(metrics["target_yaw_abs"])),
            "pred_to_best_safe_l2_mean": float(np.mean(metrics["pred_to_best_safe_l2"])),
            "pred_to_best_safe_xyz_l2_mean": float(np.mean(metrics["pred_to_best_safe_xyz_l2"])),
            "pred_to_best_safe_yaw_l1_mean": float(np.mean(metrics["pred_to_best_safe_yaw_l1"])),
            "proposal_pairwise_distance_mean": float(np.mean(metrics["pred_pairwise_distance"])),
            "proposal_pairwise_distance_p10": float(np.percentile(metrics["pred_pairwise_distance"], 10)),
            "proposal_collapse_rate": float(np.mean(metrics["proposal_collapse_rate"])),
            "proposal_near_zero_rate": float(np.mean(metrics["proposal_near_zero_rate"])),
            "proposal_unique_yaw_bucket_count_mean": float(np.mean(metrics["proposal_unique_yaw_bucket_count"])),
            "proposal_unique_yaw_bucket_count_p50": float(np.percentile(metrics["proposal_unique_yaw_bucket_count"], 50)),
            "target_best_safe_geom_gain_mean": float(np.mean(pred["target_best_safe_action"][:, 0] * 0 + np.asarray(dataset.data["proposal_geometry_gain"], dtype=np.float32)[np.arange(n), np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64)])),
            "target_geom_action_geom_gain_mean": float(np.mean(np.asarray(dataset.data["proposal_geometry_gain"], dtype=np.float32)[np.arange(n), np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64)])),
            "target_geom_action_risk_delta_mean": float(np.mean(np.asarray(dataset.data["proposal_risk_delta"], dtype=np.float32)[np.arange(n), np.asarray(dataset.data["proposal_best_safe_index"], dtype=np.int64)])),
        },
        "original_rows": _group_summary(np.where(original)[0], metrics),
        "yaw_augmented_rows": _group_summary(np.where(yaw_aug)[0], metrics),
        "yaw_opportunity_rows": _group_summary(np.where(yaw_opp)[0], metrics),
        "non_yaw_rows": _group_summary(np.where(non_yaw)[0], metrics),
        "weak_episodes": _group_summary(np.where(weak_episodes)[0], metrics),
        "strong_episodes": _group_summary(np.where(strong_episodes)[0], metrics),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
