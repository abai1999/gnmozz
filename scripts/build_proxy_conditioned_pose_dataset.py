"""
build_proxy_conditioned_pose_dataset.py

Materialize a pose-candidate dataset whose target-context fields come from a
learned visual target-delta proxy instead of privileged teacher deltas. Labels
and candidate oracle scores are kept unchanged. This lets us train a scorer
under the same noisy target context it will see with LearnedTargetProvider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from prismatic.models.target_delta_predictor import TargetDeltaPredictor


class ProxyDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data
        self.n = int(data["front_rgb"].shape[0])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        depth = self.data["wrist_depth"][idx].astype(np.float32)
        if depth.ndim == 2:
            depth = depth[None, ...]
        return {
            "front_rgb": torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0,
            "wrist_rgb": torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0,
            "wrist_depth": torch.from_numpy(depth),
            "proprio": torch.from_numpy(self.data["proprio"][idx].astype(np.float32)),
            "gripper_context": torch.from_numpy(self.data["gripper_context"][idx].astype(np.float32)),
            "has_object_in_hand": torch.tensor(float(self.data["has_object_in_hand"][idx]), dtype=torch.float32),
            "substage_id": torch.tensor(int(self.data["substage_id"][idx]), dtype=torch.long),
            "contact_state": torch.tensor(int(self.data["contact_state"][idx]), dtype=torch.long),
            "stage_target_mode": torch.tensor(int(self.data["stage_target_mode"][idx]), dtype=torch.long),
        }


def sign_bucket_arr(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.int64)
    out[x > eps] = 1
    out[x < -eps] = -1
    return out


def basin_distance(delta: np.ndarray, r_xy: float, r_z: float, r_yaw: float, r_tilt: float) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float32)
    e_xy = np.linalg.norm(arr[:, :2], axis=1)
    e_z = np.abs(arr[:, 2])
    e_yaw = np.abs(arr[:, 5])
    terms = [
        e_xy / max(float(r_xy), 1e-6),
        e_z / max(float(r_z), 1e-6),
        e_yaw / max(float(r_yaw), 1e-6),
    ]
    if np.isfinite(float(r_tilt)) and float(r_tilt) > 0:
        terms.append(np.linalg.norm(arr[:, 3:5], axis=1) / max(float(r_tilt), 1e-6))
    return np.maximum.reduce(terms).astype(np.float32)


def basin_bins(dist: np.ndarray) -> np.ndarray:
    out = np.full(dist.shape, 3, dtype=np.int64)
    out[dist <= 1.2] = 2
    out[dist <= 1.05] = 1
    out[dist <= 0.9] = 0
    return out


def predict_proxy(model: TargetDeltaPredictor, data: dict[str, np.ndarray], batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(ProxyDataset(data), batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(
                front_rgb=batch["front_rgb"].to(device=device, dtype=torch.float32),
                wrist_rgb=batch["wrist_rgb"].to(device=device, dtype=torch.float32),
                wrist_depth=batch["wrist_depth"].to(device=device, dtype=torch.float32),
                proprio=batch["proprio"].to(device=device, dtype=torch.float32),
                gripper_context=batch["gripper_context"].to(device=device, dtype=torch.float32),
                has_object_in_hand=batch["has_object_in_hand"].to(device=device, dtype=torch.float32),
                substage_id=batch["substage_id"].to(device=device),
                contact_state=batch["contact_state"].to(device=device),
                stage_target_mode=batch["stage_target_mode"].to(device=device),
            )
            preds.append(pred.float().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 6), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--target_ckpt", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--basin_radius_xy", type=float, default=0.008)
    parser.add_argument("--basin_radius_z", type=float, default=0.010)
    parser.add_argument("--basin_radius_yaw", type=float, default=0.050)
    parser.add_argument("--basin_radius_tilt", type=float, default=-1.0)
    args = parser.parse_args()

    input_path = Path(args.input_npz)
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_npz = np.load(input_path, allow_pickle=False)
    data = {k: raw_npz[k] for k in raw_npz.files}

    ckpt = torch.load(args.target_ckpt, map_location="cpu")
    model = TargetDeltaPredictor()
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    proxy_delta = predict_proxy(model, data, batch_size=int(args.batch_size), device=device)
    teacher_delta = np.asarray(data.get("target_delta_teacher", data["current_delta_basin_target"]), dtype=np.float32)
    proxy_dist = basin_distance(
        proxy_delta,
        r_xy=float(args.basin_radius_xy),
        r_z=float(args.basin_radius_z),
        r_yaw=float(args.basin_radius_yaw),
        r_tilt=float(args.basin_radius_tilt),
    )

    out = dict(data)
    out["teacher_current_delta_basin_target"] = np.asarray(data["current_delta_basin_target"], dtype=np.float32)
    out["proxy_delta_pred"] = proxy_delta.astype(np.float32)
    out["proxy_delta_error"] = (proxy_delta - teacher_delta).astype(np.float32)
    out["current_delta_basin_target"] = proxy_delta.astype(np.float32)
    out["current_basin_distance"] = proxy_dist.astype(np.float32)
    out["current_dx_sign"] = sign_bucket_arr(proxy_delta[:, 0])
    out["current_dy_sign"] = sign_bucket_arr(proxy_delta[:, 1])
    out["current_dyaw_sign"] = sign_bucket_arr(proxy_delta[:, 5])
    out["basin_distance_bin"] = basin_bins(proxy_dist)
    out["target_context_source"] = np.asarray(["learned_proxy"] * proxy_delta.shape[0])
    np.savez_compressed(output_path, **out)

    mae = np.mean(np.abs(proxy_delta - teacher_delta), axis=0).tolist() if proxy_delta.size else [0.0] * 6
    meta = {
        "input_npz": str(input_path),
        "target_ckpt": str(args.target_ckpt),
        "output_npz": str(output_path),
        "num_rows": int(proxy_delta.shape[0]),
        "proxy_delta_mae_xyzrpy": [float(x) for x in mae],
        "proxy_basin_distance_mean": float(np.mean(proxy_dist)) if proxy_dist.size else 0.0,
        "proxy_basin_distance_p90": float(np.percentile(proxy_dist, 90)) if proxy_dist.size else 0.0,
        "basin_radius_xy": float(args.basin_radius_xy),
        "basin_radius_z": float(args.basin_radius_z),
        "basin_radius_yaw": float(args.basin_radius_yaw),
        "basin_radius_tilt": float(args.basin_radius_tilt),
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
