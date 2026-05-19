"""
offline_basin_probe.py

Probe whether a trained alignment controller predicts basin corrections in the
same direction as the stored delta_basin_target on a held-out collector shard.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prismatic.models.residual_controller import ResidualController
from prismatic.vla.datasets.residual_rlbench_dataset import ResidualRLBenchDataset


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser(description="Offline basin correction probe")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ResidualRLBenchDataset(args.data_dir, stage_role_filter="align")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ResidualController(
        pose_output_mode=ckpt.get("pose_output_mode", "raw"),
        pose_use_depth=ckpt.get("pose_use_depth", True),
        pose_use_force=ckpt.get("pose_use_force", True),
        pose_use_proprio=ckpt.get("pose_use_proprio", True),
        pose_use_action=ckpt.get("pose_use_action", True),
        fire_only_head=ckpt.get("fire_only_head", False),
        ready_use_context=ckpt.get("ready_use_context", True),
        ready_use_gripper_context=ckpt.get("ready_use_gripper_context", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    cosines = []
    norm_ratios = []
    sign_hits = []
    target_norms = []
    pred_norms = []
    basin_distances = []
    planner_close_intents = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["wrist_depth"].to(device),
                batch["ft_hist"].to(device),
                batch["proprio"].to(device),
                batch["base_action"].to(device),
                batch["step_idx"].to(device),
                phase_id=batch["phase_id"].to(device),
                phase_age=batch["phase_age"].to(device),
                steps_since_last_replan=batch["steps_since_last_replan"].to(device),
                gripper_context=batch["gripper_context"].to(device),
                return_aux=True,
            )
            pred = outputs["delta_pose"].float().cpu().numpy()
            target = batch["delta_basin_target"].float().cpu().numpy()
            basin = batch["basin_distance"].float().cpu().numpy()
            intent = batch["planner_close_intent"].float().cpu().numpy()

            for p, t, bdist, pintent in zip(pred, target, basin, intent):
                p4 = np.asarray([p[0], p[1], p[2], p[5]], dtype=np.float32)
                t4 = np.asarray([t[0], t[1], t[2], t[5]], dtype=np.float32)
                cosines.append(cosine(p4, t4))
                target_norm = float(np.linalg.norm(t4))
                pred_norm = float(np.linalg.norm(p4))
                norm_ratios.append(pred_norm / max(target_norm, 1e-8))
                sign_hit = 0.0
                sign_dims = 0
                for idx in (0, 1, 5):
                    if abs(float(t[idx])) > 1e-6:
                        sign_dims += 1
                        sign_hit += float(np.sign(p[idx]) == np.sign(t[idx]))
                sign_hits.append(sign_hit / max(sign_dims, 1))
                target_norms.append(target_norm)
                pred_norms.append(pred_norm)
                basin_distances.append(float(bdist))
                planner_close_intents.append(float(pintent))

    def stat(arr):
        arr = np.asarray(arr, dtype=np.float32)
        return {
            "count": int(arr.size),
            "p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
            "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "mean": float(arr.mean()) if arr.size else 0.0,
        }

    result = {
        "data_dir": str(args.data_dir),
        "ckpt": str(args.ckpt),
        "dataset_view": dataset.meta.get("dataset_view"),
        "cosine_stats": stat(cosines),
        "norm_ratio_stats": stat(norm_ratios),
        "sign_agreement_stats": stat(sign_hits),
        "target_norm_stats": stat(target_norms),
        "pred_norm_stats": stat(pred_norms),
        "actual_basin_distance_stats": stat(basin_distances),
        "planner_close_intent_rate": float(np.mean(planner_close_intents)) if planner_close_intents else 0.0,
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
