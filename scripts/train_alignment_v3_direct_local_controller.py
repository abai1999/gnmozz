#!/usr/bin/env python3
"""Train alignment_v3_direct_local_controller for shadow-only evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from prismatic.models.alignment_v3_direct_local_controller import AlignmentV3DirectLocalController
from prismatic.vla.datasets.alignment_v3_direct_local_dataset import AlignmentV3DirectLocalDataset


def _bce_or_zero(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.float()
    if targets.numel() == 0:
        return logits.sum() * 0.0
    # If a batch has only one class, BCE is still valid; keep it simple.
    return F.binary_cross_entropy_with_logits(logits, targets)


def _shadow_metrics(batch: dict, out: dict, noop_pos_epsilon: float, noop_yaw_epsilon: float) -> dict[str, float]:
    cur = batch["current_to_target_delta_local"].float()
    pred = out["direct_residual_6d"].float()
    post = cur - pred
    cur_xy = torch.linalg.norm(cur[:, :2], dim=-1)
    cur_z = torch.abs(cur[:, 2])
    cur_yaw = torch.abs(cur[:, 5])
    post_xy = torch.linalg.norm(post[:, :2], dim=-1)
    post_z = torch.abs(post[:, 2])
    post_yaw = torch.abs(post[:, 5])

    xy_improve = (post_xy < cur_xy).float()
    z_improve = (post_z < cur_z).float()
    yaw_improve = (post_yaw < cur_yaw).float()
    all_improve = (xy_improve * z_improve * yaw_improve).float()
    pos_norm = torch.linalg.norm(pred[:, :3], dim=-1)
    yaw_abs = torch.abs(pred[:, 5])
    near_noop = ((pos_norm <= float(noop_pos_epsilon)) & (yaw_abs <= float(noop_yaw_epsilon))).float()
    teacher_noop = None
    if "target_residual_local_6d" in batch:
        teacher = batch["target_residual_local_6d"].float()
        teacher_pos = torch.linalg.norm(teacher[:, :3], dim=-1)
        teacher_yaw = torch.abs(teacher[:, 5])
        teacher_noop = ((teacher_pos <= float(noop_pos_epsilon)) & (teacher_yaw <= float(noop_yaw_epsilon))).float()
    return {
        "xy_improved_rate": float(xy_improve.mean().item()),
        "z_improved_rate": float(z_improve.mean().item()),
        "yaw_improved_rate": float(yaw_improve.mean().item()),
        "all_improved_rate": float(all_improve.mean().item()),
        "pred_pos_norm_mean": float(pos_norm.mean().item()),
        "pred_yaw_abs_mean": float(yaw_abs.mean().item()),
        "near_noop_rate": float(near_noop.mean().item()),
        "teacher_noop_rate": float(teacher_noop.mean().item()) if teacher_noop is not None else 0.0,
    }


def _shadow_metrics_by_bucket(batch: dict, out: dict, noop_pos_epsilon: float, noop_yaw_epsilon: float) -> dict[str, dict[str, float]]:
    buckets = batch.get("stage_bucket", [])
    if isinstance(buckets, str):
        buckets = [buckets]
    result: dict[str, dict[str, float]] = {}
    for bucket in sorted(set(str(b) for b in buckets)):
        indices = [i for i, b in enumerate(buckets) if str(b) == bucket]
        if not indices:
            continue
        idx = torch.tensor(indices, device=batch["current_to_target_delta_local"].device, dtype=torch.long)
        sub_batch = dict(batch)
        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and val.shape[:1] == batch["current_to_target_delta_local"].shape[:1]:
                sub_batch[key] = val.index_select(0, idx)
        sub_out = {}
        for key, val in out.items():
            if isinstance(val, torch.Tensor) and val.shape[:1] == batch["current_to_target_delta_local"].shape[:1]:
                sub_out[key] = val.index_select(0, idx)
            else:
                sub_out[key] = val
        metrics = _shadow_metrics(sub_batch, sub_out, noop_pos_epsilon, noop_yaw_epsilon)
        metrics["rows"] = float(len(indices))
        result[bucket] = metrics
    return result


def _compute_loss(batch: dict, out: dict, yaw_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    target_residual = batch["target_residual_local_4d"].float()
    pred_residual = out["direct_residual_4d"].float()
    target_post = torch.stack(
        [
            batch["target_post_xy_error"].float(),
            batch["target_post_z_error"].float(),
            batch["target_post_yaw_error"].float(),
        ],
        dim=-1,
    )
    pred_post = out["shadow_post_xyz_yaw"].float()
    overshoot = batch["overshoot_proxy"].float()
    invalid = batch["invalid_risk_proxy"].float()
    conf_target = (
        batch["target_improves_xy"].float()
        * batch["target_improves_z"].float()
        * batch["target_improves_yaw"].float()
    )

    l_residual = F.smooth_l1_loss(pred_residual[:, :3], target_residual[:, :3])
    l_residual_yaw = F.smooth_l1_loss(pred_residual[:, 3], target_residual[:, 3])
    l_post_xy = F.smooth_l1_loss(pred_post[:, 0], target_post[:, 0])
    l_post_z = F.smooth_l1_loss(pred_post[:, 1], target_post[:, 1])
    l_post_yaw = F.smooth_l1_loss(pred_post[:, 2], target_post[:, 2])
    l_risk = _bce_or_zero(out["risk_logit"].float(), torch.maximum(overshoot, invalid))
    l_conf = _bce_or_zero(out["confidence_logit"].float(), conf_target)
    l_norm = pred_residual.pow(2).mean()

    loss = (
        1.0 * l_post_xy
        + 1.0 * l_post_z
        + yaw_weight * l_post_yaw
        + 1.0 * l_residual
        + yaw_weight * l_residual_yaw
        + 0.5 * l_risk
        + 0.25 * l_conf
        + 0.1 * l_norm
    )
    return loss, {
        "loss": float(loss.item()),
        "l_post_xy": float(l_post_xy.item()),
        "l_post_z": float(l_post_z.item()),
        "l_post_yaw": float(l_post_yaw.item()),
        "l_residual": float(l_residual.item()),
        "l_residual_yaw": float(l_residual_yaw.item()),
        "l_risk": float(l_risk.item()),
        "l_conf": float(l_conf.item()),
        "l_norm": float(l_norm.item()),
    }


def _move(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    yaw_weight: float = 1.5,
    noop_pos_epsilon: float = 1e-4,
    noop_yaw_epsilon: float = 1e-4,
):
    train = optimizer is not None
    model.train(train)
    rows = []
    shadow_rows = []
    for batch in loader:
        batch = _move(batch, device)
        with torch.set_grad_enabled(train):
            out = model(
                wrist_depth=batch["wrist_depth"],
                force_history=batch["force_history"],
                proprio=batch["proprio"],
                planner_action_local=batch["planner_action_local"],
                current_to_target_delta_local=batch["current_to_target_delta_local"],
            )
            loss, loss_stats = _compute_loss(batch, out, yaw_weight=yaw_weight)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        rows.append(loss_stats)
        shadow_rows.append(_shadow_metrics(batch, out, noop_pos_epsilon, noop_yaw_epsilon))
        shadow_rows[-1]["per_bucket"] = _shadow_metrics_by_bucket(batch, out, noop_pos_epsilon, noop_yaw_epsilon)

    def _mean(key, seq):
        return float(np.mean([x[key] for x in seq])) if seq else 0.0

    def _bucket_mean(bucket: str, key: str) -> float:
        vals = []
        weights = []
        for row in shadow_rows:
            per_bucket = row.get("per_bucket", {})
            if bucket not in per_bucket:
                continue
            vals.append(float(per_bucket[bucket][key]))
            weights.append(float(per_bucket[bucket].get("rows", 1.0)))
        return float(np.average(vals, weights=weights)) if vals else 0.0

    summary = {
        "loss": _mean("loss", rows),
        "l_post_xy": _mean("l_post_xy", rows),
        "l_post_z": _mean("l_post_z", rows),
        "l_post_yaw": _mean("l_post_yaw", rows),
        "l_residual": _mean("l_residual", rows),
        "l_residual_yaw": _mean("l_residual_yaw", rows),
        "l_risk": _mean("l_risk", rows),
        "l_conf": _mean("l_conf", rows),
        "l_norm": _mean("l_norm", rows),
        "shadow_xy_improved_rate": _mean("xy_improved_rate", shadow_rows),
        "shadow_z_improved_rate": _mean("z_improved_rate", shadow_rows),
        "shadow_yaw_improved_rate": _mean("yaw_improved_rate", shadow_rows),
        "shadow_all_improved_rate": _mean("all_improved_rate", shadow_rows),
        "pred_pos_norm_mean": _mean("pred_pos_norm_mean", shadow_rows),
        "pred_yaw_abs_mean": _mean("pred_yaw_abs_mean", shadow_rows),
        "near_noop_rate": _mean("near_noop_rate", shadow_rows),
        "teacher_noop_rate": _mean("teacher_noop_rate", shadow_rows),
    }
    for bucket in ("near_alignment", "micro_contact_refine"):
        prefix = f"{bucket}_"
        for key in (
            "xy_improved_rate",
            "z_improved_rate",
            "yaw_improved_rate",
            "all_improved_rate",
            "pred_pos_norm_mean",
            "pred_yaw_abs_mean",
            "near_noop_rate",
            "teacher_noop_rate",
        ):
            summary[prefix + key] = _bucket_mean(bucket, key)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--yaw_weight", type=float, default=2.0)
    parser.add_argument("--max_pos", type=float, default=0.0200)
    parser.add_argument("--max_yaw", type=float, default=0.0100)
    parser.add_argument("--disable_planner_action", action="store_true", default=False)
    parser.add_argument("--disable_force", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overfit_sanity", action="store_true", default=False)
    parser.add_argument("--noop_pos_epsilon", type=float, default=1e-4)
    parser.add_argument("--noop_yaw_epsilon", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_planner_action = not bool(args.disable_planner_action)
    use_force = not bool(args.disable_force)

    full_ds = AlignmentV3DirectLocalDataset(str(args.dataset_npz), stage_bucket_filter=["near_alignment", "micro_contact_refine"])
    if args.overfit_sanity:
        n_small = min(64, len(full_ds))
        indices = torch.randperm(len(full_ds))[:n_small].tolist()
        train_ds = torch.utils.data.Subset(full_ds, indices)
        val_ds = torch.utils.data.Subset(full_ds, indices)
    else:
        n_val = max(1, int(len(full_ds) * 0.15))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = AlignmentV3DirectLocalController(
        max_pos=args.max_pos,
        max_yaw=args.max_yaw,
        use_planner_action=use_planner_action,
        use_force=use_force,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_score = -1.0
    for epoch in range(args.epochs):
        train_stats = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            yaw_weight=args.yaw_weight,
            noop_pos_epsilon=args.noop_pos_epsilon,
            noop_yaw_epsilon=args.noop_yaw_epsilon,
        )
        val_stats = _run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            yaw_weight=args.yaw_weight,
            noop_pos_epsilon=args.noop_pos_epsilon,
            noop_yaw_epsilon=args.noop_yaw_epsilon,
        )
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})
        composite = val_stats["shadow_all_improved_rate"] + 0.25 * val_stats["shadow_yaw_improved_rate"]
        print(
            f"epoch {epoch+1:03d} "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"shadow(xyz/yaw/all)=({val_stats['shadow_xy_improved_rate']:.3f},"
            f"{val_stats['shadow_z_improved_rate']:.3f},"
            f"{val_stats['shadow_yaw_improved_rate']:.3f},"
            f"{val_stats['shadow_all_improved_rate']:.3f})"
        )
        if composite > best_score:
            best_score = composite
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "composite_shadow_score": best_score,
                    "max_pos": args.max_pos,
                    "max_yaw": args.max_yaw,
                    "use_planner_action": use_planner_action,
                    "use_force": use_force,
                },
                output_dir / "alignment_v3_direct_local_best.pt",
            )

    all_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    final_shadow = _run_epoch(
        model,
        all_loader,
        device,
        optimizer=None,
        yaw_weight=args.yaw_weight,
        noop_pos_epsilon=args.noop_pos_epsilon,
        noop_yaw_epsilon=args.noop_yaw_epsilon,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "composite_shadow_score": best_score,
            "max_pos": args.max_pos,
            "max_yaw": args.max_yaw,
            "use_planner_action": use_planner_action,
            "use_force": use_force,
        },
        output_dir / "alignment_v3_direct_local_final.pt",
    )
    (output_dir / "train_history.json").write_text(
        json.dumps(
            {
                "dataset_npz": str(args.dataset_npz),
                "epochs": args.epochs,
                "yaw_weight": args.yaw_weight,
                "max_pos": args.max_pos,
                "max_yaw": args.max_yaw,
                "use_planner_action": use_planner_action,
                "use_force": use_force,
                "noop_pos_epsilon": args.noop_pos_epsilon,
                "noop_yaw_epsilon": args.noop_yaw_epsilon,
                "history": history,
                "final_shadow_eval": final_shadow,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[done] best composite shadow score={best_score:.4f}")
    print(json.dumps(final_shadow, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
