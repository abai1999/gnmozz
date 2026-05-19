#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from prismatic.models.depth_force_contact_policy import DepthForceLocalContactPolicy
from scripts.train_depth_force_mode_first_geometry_risk_policy import PrivilegedGeometryCandidateDataset, evaluate


def _eval_indices(
    model: DepthForceLocalContactPolicy,
    dataset: PrivilegedGeometryCandidateDataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    if indices.size == 0:
        return {"rows": 0.0}
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, device, args)
    metrics["rows"] = float(indices.size)
    return metrics


def _group_indices(dataset: PrivilegedGeometryCandidateDataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    yaw_aug = np.asarray(dataset.data.get("yaw_augmentation_applied", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(dataset.data.get("yaw_opportunity_label", np.zeros((dataset.length,), dtype=np.float32)), dtype=np.float32) > 0.5
    return {
        "all_rows": np.asarray(indices, dtype=np.int64),
        "original_rows": np.asarray(indices[~yaw_aug[indices]], dtype=np.int64),
        "yaw_augmented_rows": np.asarray(indices[yaw_aug[indices]], dtype=np.int64),
        "yaw_opportunity_rows": np.asarray(indices[yaw_opp[indices]], dtype=np.int64),
        "non_yaw_rows": np.asarray(indices[~yaw_opp[indices]], dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_normalized_costs", action="store_true", default=False)
    args = ap.parse_args()
    defaults = {
        "keep_yaw_abs": 0.02,
        "small_yaw_abs": 0.05,
        "large_yaw_abs": 0.09,
        "switch_margin": 0.001,
        "mode_weight": 0.75,
        "total_rank_weight": 1.0,
        "geometry_rank_weight": 1.0,
        "risk_rank_weight": 0.5,
        "value_weight": 0.2,
        "total_ce_weight": 0.5,
        "geometry_ce_weight": 0.35,
        "switch_weight": 0.4,
        "progress_weight": 0.2,
        "residual_weight": 0.05,
        "yaw_pair_weight": 0.15,
        "yaw_sign_weight": 0.15,
        "small_over_large_yaw_weight": 0.15,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = PrivilegedGeometryCandidateDataset(args.dataset_npz)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    model = DepthForceLocalContactPolicy().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    eps = np.asarray(dataset.data.get("episode_index", np.zeros((dataset.length,), dtype=np.int64)), dtype=np.int64)
    base_indices = np.arange(dataset.length, dtype=np.int64)
    groups = _group_indices(dataset, base_indices)

    report: dict[str, object] = {
        "dataset_npz": args.dataset_npz,
        "checkpoint_path": args.checkpoint_path,
        "rows": int(dataset.length),
        "episodes": int(len(np.unique(eps))),
        "use_normalized_costs": bool(args.use_normalized_costs),
        "groups": {},
        "episodes_by_id": {},
    }

    for gname, idx in groups.items():
        report["groups"][gname] = _eval_indices(model, dataset, idx, args.batch_size, device, args)

    for ep in sorted(int(x) for x in np.unique(eps)):
        idx = np.where(eps == ep)[0]
        report["episodes_by_id"][str(ep)] = _eval_indices(model, dataset, idx, args.batch_size, device, args)

    # Diagnostic rankings for the worst/best episodes by geometry improvement and yaw opportunity.
    ep_rows = []
    for ep_str, metrics in report["episodes_by_id"].items():
        ep_rows.append(
            {
                "episode": int(ep_str),
                "rows": float(metrics.get("rows", 0.0)),
                "selected_geometry_improves_rate": float(metrics.get("selected_geometry_improves_rate", 0.0)),
                "selected_total_improves_rate": float(metrics.get("selected_total_improves_rate", 0.0)),
                "risk_non_increase_rate": float(metrics.get("risk_non_increase_rate", 0.0)),
                "yaw_opportunity_selected_rate": float(metrics.get("yaw_opportunity_selected_rate", 0.0)),
                "geometry_regret_delta_mean": float(metrics.get("geometry_regret_delta_mean", 0.0)),
                "total_regret_delta_mean": float(metrics.get("total_regret_delta_mean", 0.0)),
            }
        )
    ep_rows_sorted_geom = sorted(ep_rows, key=lambda r: (r["selected_geometry_improves_rate"], r["geometry_regret_delta_mean"], -r["episode"]))
    ep_rows_sorted_yaw = sorted(ep_rows, key=lambda r: (r["yaw_opportunity_selected_rate"], r["selected_geometry_improves_rate"], -r["episode"]))
    report["worst_episodes_by_geometry"] = ep_rows_sorted_geom[:5]
    report["best_episodes_by_geometry"] = list(reversed(ep_rows_sorted_geom[-5:]))
    report["worst_episodes_by_yaw"] = ep_rows_sorted_yaw[:5]
    report["best_episodes_by_yaw"] = list(reversed(ep_rows_sorted_yaw[-5:]))

    out_json = Path(args.output_json) if args.output_json else Path(args.checkpoint_path).with_name("geoyaw_offline_stress_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
