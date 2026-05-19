"""
train_target_delta_predictor.py

Train a lightweight deployment-time target-delta predictor from teacher support rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from prismatic.models.target_delta_predictor import TargetDeltaPredictor

try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.set_num_interop_threads(max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "8")) // 2)))
except Exception:
    pass


class TargetDeltaDataset(Dataset):
    def __init__(self, npz_path: str):
        self.path = str(npz_path)
        # npz files are zip containers. Reading arrays lazily from the handle in
        # every __getitem__ call repeatedly pays decompression/lookup overhead,
        # which makes this small proxy experiment unexpectedly slow. Materialize
        # the required arrays once; the 20260419a candidate set is only a few GB
        # at most and easily fits in RAM on the training server.
        with np.load(npz_path, allow_pickle=False) as data:
            def _pick(*names, default=None):
                for name in names:
                    if name in data.files:
                        return np.asarray(data[name])
                if default is not None:
                    return np.asarray(default)
                raise KeyError(f"missing required field from: {names}")

            self.data = {
                "front_rgb": np.asarray(_pick("front_rgb")),
                "wrist_rgb": np.asarray(_pick("wrist_rgb")),
                "wrist_depth": np.asarray(_pick("wrist_depth")),
                "proprio": np.asarray(_pick("proprio")),
                "gripper_context": np.asarray(_pick("gripper_context")),
                "has_object_in_hand": np.asarray(_pick("has_object_in_hand", "phase_id", default=np.zeros((1,), dtype=np.float32))),
                "substage_id": np.asarray(_pick("substage_id", "phase_id", default=np.zeros((1,), dtype=np.int64))),
                "contact_state": np.asarray(_pick("contact_state", "stage_bucket_id", default=np.zeros((1,), dtype=np.int64))),
                "stage_target_mode": np.asarray(_pick("stage_target_mode", "phase_id", default=np.zeros((1,), dtype=np.int64))),
                "target_delta_teacher": np.asarray(_pick("target_delta_teacher", "teacher_target_delta_local_6d")),
                "handoff_metric_xy_error": np.asarray(
                    _pick("handoff_metric_xy_error", "teacher_truth_handoff_metric_xy_error", default=np.zeros((1,), dtype=np.float32))
                ),
                "handoff_metric_abs_z_error": np.asarray(
                    _pick("handoff_metric_abs_z_error", "teacher_truth_handoff_metric_abs_z_error", default=np.zeros((1,), dtype=np.float32))
                ),
                "handoff_metric_yaw_error": np.asarray(
                    _pick("handoff_metric_yaw_error", "teacher_truth_handoff_metric_yaw_error", default=np.zeros((1,), dtype=np.float32))
                ),
                "handoff_ready_target": np.asarray(
                    _pick("handoff_ready_target", "teacher_truth_handoff_ready", default=np.zeros((1,), dtype=np.float32))
                ),
            }
            self.data["handoff_threshold_xy_error"] = np.asarray(
                data["handoff_threshold_xy_error"] if "handoff_threshold_xy_error" in data.files else np.full((self.data["target_delta_teacher"].shape[0],), np.nan, dtype=np.float32)
            )
            self.data["handoff_threshold_abs_z_error"] = np.asarray(
                data["handoff_threshold_abs_z_error"] if "handoff_threshold_abs_z_error" in data.files else np.full((self.data["target_delta_teacher"].shape[0],), np.nan, dtype=np.float32)
            )
            self.data["handoff_threshold_yaw_error"] = np.asarray(
                data["handoff_threshold_yaw_error"] if "handoff_threshold_yaw_error" in data.files else np.full((self.data["target_delta_teacher"].shape[0],), np.nan, dtype=np.float32)
            )
        self.length = int(self.data["target_delta_teacher"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        proprio = torch.from_numpy(self.data["proprio"][idx].astype(np.float32))
        gripper_context = torch.from_numpy(self.data["gripper_context"][idx].astype(np.float32))
        if gripper_context.ndim == 1:
            if gripper_context.shape[0] < 3:
                pad = torch.zeros((3 - gripper_context.shape[0],), dtype=torch.float32)
                gripper_context = torch.cat([gripper_context, pad], dim=0)
            elif gripper_context.shape[0] > 3:
                gripper_context = gripper_context[:3]
        target = torch.from_numpy(self.data["target_delta_teacher"][idx].astype(np.float32))
        handoff_metrics = torch.tensor(
            [
                float(self.data["handoff_metric_xy_error"][idx]),
                float(self.data["handoff_metric_abs_z_error"][idx]),
                float(self.data["handoff_metric_yaw_error"][idx]),
                0.0,
            ],
            dtype=torch.float32,
        )
        handoff_ready = torch.tensor(float(self.data["handoff_ready_target"][idx]), dtype=torch.float32)
        rel_xy = float(self.data["handoff_threshold_xy_error"][idx])
        rel_z = float(self.data["handoff_threshold_abs_z_error"][idx])
        rel_yaw = float(self.data["handoff_threshold_yaw_error"][idx])
        xy = float(self.data["handoff_metric_xy_error"][idx])
        z = float(self.data["handoff_metric_abs_z_error"][idx])
        yaw = float(self.data["handoff_metric_yaw_error"][idx])
        near_ready = False
        if np.isfinite(rel_xy) and np.isfinite(rel_z):
            yaw_ok = True if (not np.isfinite(rel_yaw) or rel_yaw < 0.0) else (yaw <= rel_yaw * 1.5)
            near_ready = bool(
                z <= rel_z * 1.5
                and yaw_ok
                and xy >= rel_xy * 1.0
                and xy <= rel_xy * 3.0
            )
        sample_weight = 1.0
        if near_ready:
            sample_weight = 3.0
        if handoff_ready.item() > 0.5:
            sample_weight = max(sample_weight, 4.0)
        return {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "proprio": proprio,
            "gripper_context": gripper_context,
            "has_object_in_hand": torch.tensor(float(self.data["has_object_in_hand"][idx]), dtype=torch.float32),
            "substage_id": torch.tensor(int(self.data["substage_id"][idx]), dtype=torch.long),
            "contact_state": torch.tensor(int(self.data["contact_state"][idx]), dtype=torch.long),
            "stage_target_mode": torch.tensor(int(self.data["stage_target_mode"][idx]), dtype=torch.long),
            "target_delta_teacher": target,
            "handoff_metrics_teacher": handoff_metrics,
            "handoff_ready_target": handoff_ready,
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }


def _normalize_gripper_context(batch_gc: torch.Tensor) -> torch.Tensor:
    if batch_gc.ndim == 1:
        batch_gc = batch_gc.unsqueeze(0)
    if batch_gc.shape[-1] < 3:
        pad = torch.zeros((*batch_gc.shape[:-1], 3 - batch_gc.shape[-1]), device=batch_gc.device, dtype=batch_gc.dtype)
        batch_gc = torch.cat([batch_gc, pad], dim=-1)
    elif batch_gc.shape[-1] > 3:
        batch_gc = batch_gc[..., :3]
    return batch_gc.contiguous()


def step_model(model, batch, device):
    outputs = model(
        front_rgb=batch["front_rgb"].to(device=device, dtype=torch.float32),
        wrist_rgb=batch["wrist_rgb"].to(device=device, dtype=torch.float32),
        wrist_depth=batch["wrist_depth"].to(device=device, dtype=torch.float32),
        proprio=batch["proprio"].to(device=device, dtype=torch.float32),
        gripper_context=_normalize_gripper_context(batch["gripper_context"].to(device=device, dtype=torch.float32)),
        has_object_in_hand=batch["has_object_in_hand"].to(device=device, dtype=torch.float32),
        substage_id=batch["substage_id"].to(device=device),
        contact_state=batch["contact_state"].to(device=device),
        stage_target_mode=batch["stage_target_mode"].to(device=device),
        return_aux=True,
    )
    pred = outputs["target_delta"]
    target = batch["target_delta_teacher"].to(device=device, dtype=torch.float32)
    handoff_metrics_target = batch["handoff_metrics_teacher"].to(device=device, dtype=torch.float32)
    handoff_ready_target = batch["handoff_ready_target"].to(device=device, dtype=torch.float32)
    sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
    weights = torch.tensor([2.0, 2.0, 1.5, 1.0, 1.0, 1.2], device=device, dtype=torch.float32)
    delta_loss = F.smooth_l1_loss(pred, target, reduction="none")
    delta_loss = (delta_loss * weights.view(1, -1)).mean(dim=-1)
    metric_weights = torch.tensor([2.0, 1.5, 1.2, 0.0], device=device, dtype=torch.float32)
    metric_pred = outputs["handoff_metrics"]
    metric_loss = F.smooth_l1_loss(metric_pred, handoff_metrics_target, reduction="none")
    metric_loss = (metric_loss * metric_weights.view(1, -1)).mean(dim=-1)
    ready_logit = outputs["handoff_ready_logit"]
    ready_loss = F.binary_cross_entropy_with_logits(ready_logit, handoff_ready_target, reduction="none")
    loss = ((delta_loss + 0.75 * metric_loss + 0.5 * ready_loss) * sample_weight).mean()
    with torch.no_grad():
        mae = torch.mean(torch.abs(pred - target), dim=0)
        handoff_mae = torch.mean(torch.abs(metric_pred[:, :3] - handoff_metrics_target[:, :3]), dim=0)
        ready_prob = torch.sigmoid(ready_logit)
        ready_acc = torch.mean(((ready_prob > 0.5).float() == (handoff_ready_target > 0.5).float()).float())
    return pred, target, loss, mae, handoff_mae, ready_acc


def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = torch.zeros(6, dtype=torch.float64)
    total_handoff_mae = torch.zeros(3, dtype=torch.float64)
    total_ready_acc = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            _, _, loss, mae, handoff_mae, ready_acc = step_model(model, batch, device)
            bsz = batch["target_delta_teacher"].shape[0]
            total_loss += float(loss.item()) * bsz
            total_mae += mae.double().cpu() * bsz
            total_handoff_mae += handoff_mae.double().cpu() * bsz
            total_ready_acc += float(ready_acc.item()) * bsz
            n += bsz
    return {
        "loss": total_loss / max(n, 1),
        "mae_xyzrpy": (total_mae / max(n, 1)).tolist(),
        "handoff_mae_xyz": (total_handoff_mae / max(n, 1)).tolist(),
        "handoff_ready_acc": total_ready_acc / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = TargetDeltaDataset(args.dataset_npz)
    n_val = max(1, int(len(dataset) * float(args.val_ratio)))
    n_train = max(len(dataset) - n_val, 1)
    train_set, val_set = random_split(dataset, [n_train, len(dataset) - n_train], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TargetDeltaPredictor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            _, _, loss, _, _, _ = step_model(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = batch["target_delta_teacher"].shape[0]
            running += float(loss.item()) * bsz
            count += bsz
        train_loss = running / max(count, 1)
        val_stats = evaluate_model(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **val_stats}
        history.append(row)
        print(json.dumps(row))
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset_npz": str(args.dataset_npz),
                    "predicts_handoff_aux": True,
                    "history": history,
                    "val_stats": val_stats,
                },
                output_dir / "target_delta_predictor_best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_npz": str(args.dataset_npz),
            "predicts_handoff_aux": True,
            "history": history,
            "best_val_loss": best_val,
        },
        output_dir / "target_delta_predictor_final.pt",
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
