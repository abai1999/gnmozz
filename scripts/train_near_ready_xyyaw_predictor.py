"""
Train a lightweight near-ready xy+yaw predictor from truth-diag support rows.
This is an offline diagnostic head only; it does not alter runtime control.
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
from torch.utils.data import DataLoader, Dataset, random_split

from prismatic.models.near_ready_xyyaw_predictor import NearReadyXYYawPredictor

try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.set_num_interop_threads(max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "8")) // 2)))
except Exception:
    pass


class NearReadyDataset(Dataset):
    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as data:
            self.data = {k: np.asarray(data[k]) for k in data.files}
        self.length = int(self.data["teacher_truth_ready_target"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        front = torch.from_numpy(self.data["front_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        wrist = torch.from_numpy(self.data["wrist_rgb"][idx].astype(np.float32).transpose(2, 0, 1)) / 255.0
        depth = torch.from_numpy(self.data["wrist_depth"][idx].astype(np.float32))
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        return {
            "front_rgb": front,
            "wrist_rgb": wrist,
            "wrist_depth": depth,
            "proprio": torch.from_numpy(self.data["proprio"][idx].astype(np.float32)),
            "gripper_context": torch.from_numpy(self.data["gripper_context"][idx].astype(np.float32)),
            "runtime_xyyaw_norm": torch.from_numpy(self.data["runtime_xyyaw_norm"][idx].astype(np.float32)),
            "teacher_xyyaw_norm": torch.from_numpy(self.data["teacher_truth_xyyaw_norm"][idx].astype(np.float32)),
            "teacher_ready_target": torch.tensor(float(self.data["teacher_truth_ready_target"][idx]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.data["sample_weight"][idx]), dtype=torch.float32),
            "hard_negative_target": torch.tensor(float(self.data["hard_negative_target"][idx]), dtype=torch.float32),
            "substage_id": torch.tensor(int(self.data["substage_id"][idx]), dtype=torch.long),
            "contact_state": torch.tensor(int(self.data["contact_state"][idx]), dtype=torch.long),
            "stage_target_mode": torch.tensor(int(self.data["stage_target_mode"][idx]), dtype=torch.long),
        }


def make_stratified_split(dataset: NearReadyDataset, val_ratio: float, seed: int):
    labels = np.asarray(dataset.data["teacher_truth_ready_target"], dtype=np.float32)
    all_idx = np.arange(labels.shape[0], dtype=np.int64)
    pos_idx = all_idx[labels > 0.5]
    neg_idx = all_idx[labels <= 0.5]
    rng = np.random.default_rng(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_total = int(all_idx.shape[0])
    n_val = max(1, int(n_total * float(val_ratio)))
    n_pos_total = int(pos_idx.shape[0])
    n_pos_val = 0
    if n_pos_total > 0:
        n_pos_val = max(1, int(round(n_pos_total * float(val_ratio))))
        n_pos_val = min(n_pos_val, n_pos_total)
    n_neg_val = max(0, min(int(neg_idx.shape[0]), n_val - n_pos_val))
    if n_neg_val + n_pos_val < n_val:
        spill = n_val - (n_neg_val + n_pos_val)
        extra_neg = neg_idx[n_neg_val : n_neg_val + spill]
        val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val], extra_neg], axis=0)
    else:
        val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]], axis=0)
    train_mask = np.ones((n_total,), dtype=bool)
    train_mask[val_idx] = False
    train_idx = all_idx[train_mask]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.tolist(), val_idx.tolist()


def forward_loss(model, batch, device):
    outputs = model(
        front_rgb=batch["front_rgb"].to(device=device, dtype=torch.float32),
        wrist_rgb=batch["wrist_rgb"].to(device=device, dtype=torch.float32),
        wrist_depth=batch["wrist_depth"].to(device=device, dtype=torch.float32),
        proprio=batch["proprio"].to(device=device, dtype=torch.float32),
        gripper_context=batch["gripper_context"].to(device=device, dtype=torch.float32),
        runtime_xyyaw_norm=batch["runtime_xyyaw_norm"].to(device=device, dtype=torch.float32),
        substage_id=batch["substage_id"].to(device=device),
        contact_state=batch["contact_state"].to(device=device),
        stage_target_mode=batch["stage_target_mode"].to(device=device),
    )
    pred_xyyaw = outputs["xyyaw_norm"]
    pred_ready = outputs["ready_logit"]
    tgt_xyyaw = batch["teacher_xyyaw_norm"].to(device=device, dtype=torch.float32)
    tgt_ready = batch["teacher_ready_target"].to(device=device, dtype=torch.float32)
    sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
    hard_negative = batch["hard_negative_target"].to(device=device, dtype=torch.float32)

    metric_loss = F.smooth_l1_loss(pred_xyyaw, tgt_xyyaw, reduction="none").mean(dim=-1)
    ready_loss = F.binary_cross_entropy_with_logits(pred_ready, tgt_ready, reduction="none")
    weighted = metric_loss + 0.5 * ready_loss + 0.25 * hard_negative * ready_loss
    loss = (weighted * sample_weight).mean()

    with torch.no_grad():
        mae = torch.mean(torch.abs(pred_xyyaw - tgt_xyyaw), dim=0)
        ready_prob = torch.sigmoid(pred_ready)
        ready_acc = torch.mean(((ready_prob > 0.5).float() == (tgt_ready > 0.5).float()).float())
        runtime_mae = torch.mean(
            torch.abs(batch["runtime_xyyaw_norm"].to(device=device, dtype=torch.float32) - tgt_xyyaw),
            dim=0,
        )
    return loss, mae, runtime_mae, ready_acc


def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = torch.zeros(2, dtype=torch.float64)
    total_runtime_mae = torch.zeros(2, dtype=torch.float64)
    total_ready_acc = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            loss, mae, runtime_mae, ready_acc = forward_loss(model, batch, device)
            bsz = batch["teacher_ready_target"].shape[0]
            total_loss += float(loss.item()) * bsz
            total_mae += mae.double().cpu() * bsz
            total_runtime_mae += runtime_mae.double().cpu() * bsz
            total_ready_acc += float(ready_acc.item()) * bsz
            n += bsz
    return {
        "loss": total_loss / max(n, 1),
        "mae_xyyaw_norm": (total_mae / max(n, 1)).tolist(),
        "runtime_baseline_mae_xyyaw_norm": (total_runtime_mae / max(n, 1)).tolist(),
        "ready_acc": total_ready_acc / max(n, 1),
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

    dataset = NearReadyDataset(args.dataset_npz)
    train_idx, val_idx = make_stratified_split(dataset, args.val_ratio, args.seed)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NearReadyXYYawPredictor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []
    split_info = {
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "train_ready_positive": int(np.sum(labels := np.asarray(dataset.data["teacher_truth_ready_target"], dtype=np.float32)[train_idx] > 0.5)),
        "val_ready_positive": int(np.sum(np.asarray(dataset.data["teacher_truth_ready_target"], dtype=np.float32)[val_idx] > 0.5)),
    }
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            loss, _, _, _ = forward_loss(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsz = batch["teacher_ready_target"].shape[0]
            total += float(loss.item()) * bsz
            count += bsz
        train_loss = total / max(count, 1)
        val_stats = eval_model(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **val_stats}
        history.append(row)
        print(json.dumps(row))
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset_npz": str(args.dataset_npz),
                    "history": history,
                    "val_stats": val_stats,
                },
                output_dir / "near_ready_xyyaw_predictor_best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_npz": str(args.dataset_npz),
            "history": history,
            "val_stats": history[-1] if history else {},
        },
        output_dir / "near_ready_xyyaw_predictor_final.pt",
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
