#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2


class AlignmentV4Dataset(Dataset):
    def __init__(self, npz_path: str):
        raw = np.load(npz_path, allow_pickle=False)
        self.data = {k: np.asarray(raw[k]) for k in raw.files}
        self.length = int(self.data["episode_index"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict:
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
            "proxy_current_delta_basin_target": torch.from_numpy(self.data["proxy_current_delta_basin_target"][idx].astype(np.float32)),
            "temporal_action_summary": torch.from_numpy(self.data["temporal_action_summary"][idx].astype(np.float32)),
            "current_dx_sign": torch.tensor(int(self.data["current_dx_sign"][idx]), dtype=torch.long),
            "current_dy_sign": torch.tensor(int(self.data["current_dy_sign"][idx]), dtype=torch.long),
            "current_dyaw_sign": torch.tensor(int(self.data["current_dyaw_sign"][idx]), dtype=torch.long),
            "basin_distance_bin": torch.tensor(int(self.data["basin_distance_bin"][idx]), dtype=torch.long),
            "substage_id": torch.tensor(int(self.data["substage_id"][idx]), dtype=torch.long),
            "contact_state": torch.tensor(int(self.data["contact_state"][idx]), dtype=torch.long),
            "stage_target_mode": torch.tensor(int(self.data["stage_target_mode"][idx]), dtype=torch.long),
            "residual_target": torch.from_numpy(self.data["alignment_v4_residual_target"][idx].astype(np.float32)),
            "residual_mask": torch.tensor(float(self.data["alignment_v4_residual_mask"][idx]), dtype=torch.float32),
            "confidence_target": torch.tensor(float(self.data["alignment_v4_residual_confidence_target"][idx]), dtype=torch.float32),
            "improvement_label": torch.tensor(float(self.data["alignment_v4_improvement_label"][idx]), dtype=torch.float32),
            "closeability_label": torch.tensor(float(self.data["alignment_v4_closeability_label"][idx]), dtype=torch.float32),
            "progress_label": torch.tensor(float(self.data.get("alignment_v4_progress_label", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "progress_mask": torch.tensor(float(self.data.get("alignment_v4_progress_mask", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "focus_mask": torch.tensor(float(self.data.get("alignment_v4_focus_mask", np.zeros((self.length,), dtype=np.float32))[idx]), dtype=torch.float32),
            "cost_before": torch.tensor(float(self.data["alignment_v4_cost_before"][idx]), dtype=torch.float32),
            "target_cost_after": torch.tensor(float(self.data["alignment_v4_cost_after_target"][idx]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.data["sample_weight"][idx]), dtype=torch.float32),
        }


def split_by_episode(dataset: AlignmentV4Dataset, val_ratio: float, seed: int):
    eps = np.asarray(dataset.data["episode_index"], dtype=np.int64)
    uniq = np.unique(eps)
    rng = np.random.default_rng(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    val_n = max(1, min(uniq.size - 1, int(round(uniq.size * val_ratio))))
    val_eps = set(int(x) for x in shuffled[:val_n])
    train_idx = np.where(~np.isin(eps, sorted(val_eps)))[0].tolist()
    val_idx = np.where(np.isin(eps, sorted(val_eps)))[0].tolist()
    return train_idx, val_idx, sorted(val_eps)


def make_loader(dataset: AlignmentV4Dataset, indices: list[int], batch_size: int, weighted: bool, shuffle: bool):
    subset = Subset(dataset, indices)
    if weighted:
        weights = np.asarray(dataset.data["sample_weight"], dtype=np.float32)[indices]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(indices), replacement=True)
        return DataLoader(subset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def forward_batch(model, batch, device):
    return model(
        front_rgb=batch["front_rgb"].to(device),
        wrist_rgb=batch["wrist_rgb"].to(device),
        wrist_depth=batch["wrist_depth"].to(device),
        proprio=batch["proprio"].to(device),
        gripper_context=batch["gripper_context"].to(device),
        proxy_current_delta_basin_target=batch["proxy_current_delta_basin_target"].to(device),
        current_dx_sign=batch["current_dx_sign"].to(device),
        current_dy_sign=batch["current_dy_sign"].to(device),
        current_dyaw_sign=batch["current_dyaw_sign"].to(device),
        basin_distance_bin=batch["basin_distance_bin"].to(device),
        substage_id=batch["substage_id"].to(device),
        contact_state=batch["contact_state"].to(device),
        stage_target_mode=batch["stage_target_mode"].to(device),
        temporal_action_summary=batch["temporal_action_summary"].to(device),
    )


def compute_loss(model, batch, device, args):
    out = forward_batch(model, batch, device)
    pred = out["residual_delta_local"]
    target = batch["residual_target"].to(device)
    mask = batch["residual_mask"].to(device).float()
    weight = batch["sample_weight"].to(device).float()
    residual_l1 = F.smooth_l1_loss(pred, target, reduction="none")
    xyz_loss = residual_l1[:, :3].mean(dim=-1)
    yaw_loss = residual_l1[:, 3]
    conf_logit = out["residual_confidence_logit"]
    conf_target = batch["confidence_target"].to(device).float()
    conf_loss = F.binary_cross_entropy_with_logits(conf_logit, conf_target, reduction="none")
    close_logit = out["closeability_logit"]
    close_target = batch["closeability_label"].to(device).float()
    close_loss = F.binary_cross_entropy_with_logits(close_logit, close_target, reduction="none")
    progress_logit = out["progress_logit"]
    progress_target = batch["progress_label"].to(device).float()
    progress_mask = batch["progress_mask"].to(device).float()
    progress_loss = F.binary_cross_entropy_with_logits(progress_logit, progress_target, reduction="none")
    loss_vec = (
        float(args.lambda_xyz) * xyz_loss
        + float(args.lambda_yaw) * yaw_loss
        + float(args.lambda_confidence) * conf_loss
        + float(args.lambda_closeability) * close_loss
        + float(args.lambda_progress) * progress_loss * progress_mask
    )
    loss = (loss_vec * mask * weight).sum() / torch.clamp((mask * weight).sum(), min=1.0)
    return loss


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    improve_count = 0
    close_probs: list[float] = []
    close_labels: list[float] = []
    progress_probs: list[float] = []
    progress_labels: list[float] = []
    progress_masks: list[float] = []
    confidence_probs: list[float] = []
    confidence_labels: list[float] = []
    for batch in loader:
        out = forward_batch(model, batch, device)
        pred = out["residual_delta_local"].float()
        target = batch["residual_target"].to(device).float()
        delta = batch["proxy_current_delta_basin_target"].to(device).float()
        before = batch["cost_before"].to(device).float()
        close_label = batch["closeability_label"].to(device).float()
        close_prob = torch.sigmoid(out["closeability_logit"].float())
        progress_label = batch["progress_label"].to(device).float()
        progress_mask = batch["progress_mask"].to(device).float()
        progress_prob = torch.sigmoid(out["progress_logit"].float())
        conf_prob = torch.sigmoid(out["residual_confidence_logit"].float())
        conf_label = batch["improvement_label"].to(device).float()
        # Approximate rollout in the same local-delta convention used by the dataset builder.
        after_delta = delta.clone()
        after_delta[:, :3] = after_delta[:, :3] - pred[:, :3]
        after_delta[:, 5] = after_delta[:, 5] - pred[:, 3]
        xy = torch.linalg.norm(after_delta[:, :2], dim=-1) / 0.0085
        z = torch.abs(after_delta[:, 2]) / 0.0035
        yaw = torch.abs(after_delta[:, 5]) / 0.1243404
        after_cost = 0.45 * xy + 0.30 * z + 0.25 * yaw
        improved = after_cost < before
        bsz = int(pred.shape[0])
        count += bsz
        improve_count += int(improved.sum().item())
        xyz_mae = torch.abs(pred[:, :3] - target[:, :3]).mean().item()
        yaw_mae = torch.abs(pred[:, 3] - target[:, 3]).mean().item()
        dir_acc = (torch.sign(pred[:, :3]) == torch.sign(target[:, :3])).float().mean().item()
        sums["residual_xyz_mae"] = sums.get("residual_xyz_mae", 0.0) + xyz_mae * bsz
        sums["residual_yaw_mae"] = sums.get("residual_yaw_mae", 0.0) + yaw_mae * bsz
        sums["xyz_direction_acc"] = sums.get("xyz_direction_acc", 0.0) + dir_acc * bsz
        sums["predicted_after_cost_mean"] = sums.get("predicted_after_cost_mean", 0.0) + float(after_cost.mean().item()) * bsz
        sums["cost_before_mean"] = sums.get("cost_before_mean", 0.0) + float(before.mean().item()) * bsz
        sums["focus_improvement_rate"] = sums.get("focus_improvement_rate", 0.0) + float(improved[batch["focus_mask"].to(device) > 0.5].float().mean().item() if torch.any(batch["focus_mask"].to(device) > 0.5) else 0.0) * bsz
        close_probs.extend(close_prob.detach().cpu().numpy().tolist())
        close_labels.extend(close_label.detach().cpu().numpy().tolist())
        progress_probs.extend(progress_prob.detach().cpu().numpy().tolist())
        progress_labels.extend(progress_label.detach().cpu().numpy().tolist())
        progress_masks.extend(progress_mask.detach().cpu().numpy().tolist())
        confidence_probs.extend(conf_prob.detach().cpu().numpy().tolist())
        confidence_labels.extend(conf_label.detach().cpu().numpy().tolist())
    metrics = {k: float(v / max(count, 1)) for k, v in sums.items()}
    metrics["predicted_improvement_rate"] = float(improve_count / max(count, 1))
    probs = np.asarray(close_probs, dtype=np.float32)
    labels = np.asarray(close_labels, dtype=np.float32) > 0.5
    if probs.size and np.any(labels) and np.any(~labels):
        best = {"balanced_acc": -1.0, "threshold": 0.5, "pos_recall": 0.0, "neg_recall": 0.0}
        for thr in np.unique(np.quantile(probs, np.linspace(0.0, 1.0, 101))):
            pred = probs >= float(thr)
            pos = float(np.mean(pred[labels])) if np.any(labels) else 0.0
            neg = float(np.mean(~pred[~labels])) if np.any(~labels) else 0.0
            ba = 0.5 * (pos + neg)
            if ba > best["balanced_acc"]:
                best = {"balanced_acc": float(ba), "threshold": float(thr), "pos_recall": pos, "neg_recall": neg}
        metrics["closeability_calibrated_balanced_acc"] = best["balanced_acc"]
        metrics["closeability_calibrated_threshold"] = best["threshold"]
        metrics["closeability_calibrated_pos_recall"] = best["pos_recall"]
        metrics["closeability_calibrated_neg_recall"] = best["neg_recall"]
    else:
        metrics["closeability_calibrated_balanced_acc"] = 0.5
        metrics["closeability_calibrated_threshold"] = 0.5
        metrics["closeability_calibrated_pos_recall"] = 0.0
        metrics["closeability_calibrated_neg_recall"] = 0.0
    progress_probs_np = np.asarray(progress_probs, dtype=np.float32)
    progress_labels_np = np.asarray(progress_labels, dtype=np.float32) > 0.5
    progress_masks_np = np.asarray(progress_masks, dtype=np.float32) > 0.5
    if np.any(progress_masks_np):
        p_probs = progress_probs_np[progress_masks_np]
        p_labels = progress_labels_np[progress_masks_np]
        if p_probs.size > 0 and np.any(p_labels) and np.any(~p_labels):
            best = {"balanced_acc": -1.0, "threshold": 0.5, "pos_recall": 0.0, "neg_recall": 0.0}
            for thr in np.unique(np.quantile(p_probs, np.linspace(0.0, 1.0, 101))):
                pred_np = p_probs >= float(thr)
                pos = float(np.mean(pred_np[p_labels])) if np.any(p_labels) else 0.0
                neg = float(np.mean(~pred_np[~p_labels])) if np.any(~p_labels) else 0.0
                ba = 0.5 * (pos + neg)
                if ba > best["balanced_acc"]:
                    best = {"balanced_acc": float(ba), "threshold": float(thr), "pos_recall": pos, "neg_recall": neg}
            metrics["progress_balanced_acc"] = best["balanced_acc"]
            metrics["progress_calibrated_threshold"] = best["threshold"]
            metrics["progress_pos_recall"] = best["pos_recall"]
            metrics["progress_neg_recall"] = best["neg_recall"]
        else:
            metrics["progress_balanced_acc"] = 0.5
            metrics["progress_calibrated_threshold"] = 0.5
            metrics["progress_pos_recall"] = 0.0
            metrics["progress_neg_recall"] = 0.0
    else:
        metrics["progress_balanced_acc"] = 0.5
        metrics["progress_calibrated_threshold"] = 0.5
        metrics["progress_pos_recall"] = 0.0
        metrics["progress_neg_recall"] = 0.0
    conf_probs_np = np.asarray(confidence_probs, dtype=np.float32)
    conf_labels_np = np.asarray(confidence_labels, dtype=np.float32) > 0.5
    if conf_probs_np.size > 0 and np.any(conf_labels_np) and np.any(~conf_labels_np):
        best = {"balanced_acc": -1.0, "threshold": 0.5, "pos_recall": 0.0, "neg_recall": 0.0}
        for thr in np.unique(np.quantile(conf_probs_np, np.linspace(0.0, 1.0, 101))):
            pred_np = conf_probs_np >= float(thr)
            pos = float(np.mean(pred_np[conf_labels_np])) if np.any(conf_labels_np) else 0.0
            neg = float(np.mean(~pred_np[~conf_labels_np])) if np.any(~conf_labels_np) else 0.0
            ba = 0.5 * (pos + neg)
            if ba > best["balanced_acc"]:
                best = {"balanced_acc": float(ba), "threshold": float(thr), "pos_recall": pos, "neg_recall": neg}
        metrics["confidence_balanced_acc"] = best["balanced_acc"]
        metrics["confidence_calibrated_threshold"] = best["threshold"]
        metrics["confidence_pos_recall"] = best["pos_recall"]
        metrics["confidence_neg_recall"] = best["neg_recall"]
    else:
        metrics["confidence_balanced_acc"] = 0.5
        metrics["confidence_calibrated_threshold"] = 0.5
        metrics["confidence_pos_recall"] = 0.0
        metrics["confidence_neg_recall"] = 0.0
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--init_ckpt", default="")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--weighted_sampling", action="store_true")
    ap.add_argument("--lambda_xyz", type=float, default=1.0)
    ap.add_argument("--lambda_yaw", type=float, default=0.5)
    ap.add_argument("--lambda_confidence", type=float, default=0.25)
    ap.add_argument("--lambda_closeability", type=float, default=0.15)
    ap.add_argument("--lambda_progress", type=float, default=0.10)
    ap.add_argument("--gate_improvement_rate", type=float, default=0.60)
    ap.add_argument("--gate_xyz_direction_acc", type=float, default=0.55)
    ap.add_argument("--gate_closeability_balanced_acc", type=float, default=0.58)
    ap.add_argument("--gate_progress_balanced_acc", type=float, default=0.55)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = AlignmentV4Dataset(args.dataset_npz)
    train_idx, val_idx, val_eps = split_by_episode(dataset, float(args.val_ratio), int(args.seed))
    train_loader = make_loader(dataset, train_idx, int(args.batch_size), bool(args.weighted_sampling), True)
    val_loader = make_loader(dataset, val_idx, int(args.batch_size), False, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StudentHandoffStateHeadV2().to(device)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    history = []
    best_score = -1e9
    best_path = out_dir / "student_handoff_state_head_v2_alignment_v4_best_residual_policy.pt"
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        rows = 0
        for batch in train_loader:
            loss = compute_loss(model, batch, device, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bsz = int(batch["residual_target"].shape[0])
            total += float(loss.item()) * bsz
            rows += bsz
        metrics = evaluate(model, val_loader, device, args)
        score = (
            metrics.get("predicted_improvement_rate", 0.0)
            + 0.5 * metrics.get("xyz_direction_acc", 0.0)
            - 20.0 * metrics.get("residual_xyz_mae", 0.0)
            - 2.0 * metrics.get("residual_yaw_mae", 0.0)
        )
        rec = {"epoch": epoch, "train_loss": total / max(rows, 1), **metrics}
        history.append(rec)
        print(json.dumps(rec))
        if score > best_score:
            best_score = score
            torch.save({"model_state_dict": model.state_dict(), "metrics": metrics, "args": vars(args)}, best_path)
    final = evaluate(model, val_loader, device, args)
    decision = "shadow_candidate" if (
        final.get("predicted_improvement_rate", 0.0) >= float(args.gate_improvement_rate)
        and final.get("xyz_direction_acc", 0.0) >= float(args.gate_xyz_direction_acc)
        and final.get("closeability_calibrated_balanced_acc", 0.0) >= float(args.gate_closeability_balanced_acc)
        and final.get("progress_balanced_acc", 0.0) >= float(args.gate_progress_balanced_acc)
    ) else "offline_blocked"
    report = {
        "decision": decision,
        "best_ckpt": str(best_path),
        "dataset_npz": str(args.dataset_npz),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "val_episodes": [int(x) for x in val_eps],
        **final,
    }
    (out_dir / "alignment_v4_train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "alignment_v4_gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
