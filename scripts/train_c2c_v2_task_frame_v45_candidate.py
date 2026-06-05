#!/usr/bin/env python3
"""Train the v45 joint task-frame dz/dyaw candidate for C2C v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.task_frame_v45_candidate import (  # noqa: E402
    TASK_FRAME_V45_RISK_CLASSES,
    TaskFrameV45CandidateNet,
    load_task_frame_v45_candidate_checkpoint,
    save_task_frame_v45_candidate_checkpoint,
    task_frame_v45_candidate_feature_vector,
    task_frame_v45_candidate_labels_from_row,
)
from prismatic.robot.coarse2contact_v2.task_frame_readiness import TASK_FRAME_READINESS_FEATURE_NAMES  # noqa: E402
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import (  # noqa: E402
    split_records_by_source_root,
    source_eval_root_key,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in paths:
        files = sorted(item.glob("*.jsonl")) if item.is_dir() else [item]
        for path in files:
            rows.extend(_read_jsonl(path))
    rows.sort(key=lambda row: (str(row.get("source_eval_root", "")), int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1)))))
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _risk_reason_target(row: Mapping[str, Any]) -> str:
    labels = row.get("offline_labels", {})
    labels = labels if isinstance(labels, Mapping) else {}
    if bool(row.get("wrist_is_occluded", False)) or bool(row.get("wrist_is_low_visibility", False)):
        return "low_visibility"
    force_norm = _safe_float(row.get("grasp_contact_rule_force_norm", row.get("force_norm", 0.0)), 0.0)
    if bool(row.get("grasp_contact_rule_contact_confirmed", False)) or force_norm > 0.18:
        return "force_guard"
    if bool(labels.get("yaw_ambiguous", labels.get("yaw_ambiguous_label", False))):
        return "direction_conflict"
    if not bool(labels.get("z_ready", labels.get("z_ready_label", False))) or not bool(labels.get("yaw_ready", labels.get("yaw_ready_label", False))):
        return "insufficient_support"
    return "normal"


def _step_scale_target(row: Mapping[str, Any]) -> float:
    labels = row.get("offline_labels", {})
    labels = labels if isinstance(labels, Mapping) else {}
    dz = abs(_safe_float(labels.get("dz", row.get("privileged_dz", 0.0)), 0.0))
    dyaw = abs(_safe_float(labels.get("dyaw", row.get("privileged_dyaw", 0.0)), 0.0))
    z_ready = bool(labels.get("z_ready", labels.get("z_ready_label", False)))
    yaw_ready = bool(labels.get("yaw_ready", labels.get("yaw_ready_label", False)))
    step = float(min(np.clip(1.0 - dz / 0.020, 0.05, 1.0), np.clip(1.0 - dyaw / 0.140, 0.05, 1.0)))
    if not z_ready:
        step *= 0.85
    if not yaw_ready:
        step *= 0.75
    if bool(row.get("wrist_is_occluded", False)) or bool(row.get("wrist_is_low_visibility", False)):
        step *= 0.80
    if bool(row.get("grasp_contact_rule_contact_confirmed", False)):
        step *= 0.75
    return float(np.clip(step, 0.05, 1.0))


def _split_rows(rows: list[dict[str, Any]], *, val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = split_records_by_source_root(rows, split_mode="root", val_fraction=val_fraction, test_fraction=0.0, seed=seed)
    return list(split.train_records), list(split.val_records)


def _build_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    dz: list[float] = []
    dyaw: list[float] = []
    z_ready: list[float] = []
    z_conf: list[float] = []
    yaw_ready: list[float] = []
    yaw_conf: list[float] = []
    yaw_ambiguous: list[float] = []
    step_scale: list[float] = []
    risk_reason: list[int] = []
    weights: list[float] = []
    kept: list[dict[str, Any]] = []
    risk_index = {name: idx for idx, name in enumerate(TASK_FRAME_V45_RISK_CLASSES)}
    for row in rows:
        labels = task_frame_v45_candidate_labels_from_row(row)
        if labels is None:
            continue
        feat = task_frame_v45_candidate_feature_vector(row)
        if not np.all(np.isfinite(feat)):
            continue
        features.append(feat)
        dz.append(float(labels["dz"]))
        dyaw.append(float(labels["dyaw"]))
        z_ready.append(float(labels["z_ready"]))
        z_conf.append(float(labels["z_confidence"]))
        yaw_ready.append(float(labels["yaw_ready"]))
        yaw_conf.append(float(labels["yaw_confidence"]))
        yaw_ambiguous.append(float(labels["yaw_ambiguous"]))
        step_scale.append(float(_step_scale_target(row)))
        risk_reason.append(int(risk_index.get(str(labels["risk_reason"]), 0)))
        weight = 1.0
        bucket = str(row.get("failure_bucket", row.get("bucket", "")) or "")
        obs_bucket = str(row.get("observability_bucket", "") or "")
        if bucket in {"large_xy_large_yaw", "small_xy_large_yaw"}:
            weight *= 1.6
        if obs_bucket in {"occluded", "low_observability", "low_visibility", "partial_observable", "partial_observation"}:
            weight *= 1.7
        if bool(row.get("grasp_probe_active", False)):
            weight *= 1.4
        weights.append(float(weight))
        kept.append(dict(row))
    if not features:
        raise RuntimeError("no v45 candidate rows available")
    x = np.stack(features).astype(np.float32)
    y = {
        "dz": np.asarray(dz, dtype=np.float32),
        "dyaw": np.asarray(dyaw, dtype=np.float32),
        "z_ready": np.asarray(z_ready, dtype=np.float32),
        "z_conf": np.asarray(z_conf, dtype=np.float32),
        "yaw_ready": np.asarray(yaw_ready, dtype=np.float32),
        "yaw_conf": np.asarray(yaw_conf, dtype=np.float32),
        "yaw_ambiguous": np.asarray(yaw_ambiguous, dtype=np.float32),
        "step_scale": np.asarray(step_scale, dtype=np.float32),
        "risk_reason": np.asarray(risk_reason, dtype=np.int64),
    }
    return x, y, np.asarray(weights, dtype=np.float32), kept


def _metrics(model: TaskFrameV45CandidateNet, x: torch.Tensor, y: dict[str, torch.Tensor]) -> dict[str, Any]:
    if x.numel() == 0:
        return {"rows": 0}
    model.eval()
    with torch.no_grad():
        out = model(x)
    dz_err = torch.abs(out["dz"] - y["dz"])
    dyaw_err = torch.abs(out["dyaw"] - y["dyaw"])
    z_ready_pred = out["z_ready_probability"] >= 0.5
    yaw_ready_pred = out["yaw_ready_probability"] >= 0.5
    yaw_ambig_pred = out["yaw_ambiguous_probability"] >= 0.5
    return {
        "rows": int(x.shape[0]),
        "dz_mae": float(dz_err.mean().item()),
        "dyaw_mae": float(dyaw_err.mean().item()),
        "dz_sign_match": float((torch.sign(out["dz"]) == torch.sign(y["dz"])).float().mean().item()),
        "dyaw_sign_match": float((torch.sign(out["dyaw"]) == torch.sign(y["dyaw"])).float().mean().item()),
        "z_ready_precision": float(((z_ready_pred & (y["z_ready"] >= 0.5)).float().sum() / torch.clamp(z_ready_pred.float().sum(), min=1.0)).item()),
        "z_ready_recall": float(((z_ready_pred & (y["z_ready"] >= 0.5)).float().sum() / torch.clamp((y["z_ready"] >= 0.5).float().sum(), min=1.0)).item()),
        "yaw_ready_precision": float(((yaw_ready_pred & (y["yaw_ready"] >= 0.5)).float().sum() / torch.clamp(yaw_ready_pred.float().sum(), min=1.0)).item()),
        "yaw_ready_recall": float(((yaw_ready_pred & (y["yaw_ready"] >= 0.5)).float().sum() / torch.clamp((y["yaw_ready"] >= 0.5).float().sum(), min=1.0)).item()),
        "yaw_ambiguous_rate": float((y["yaw_ambiguous"] >= 0.5).float().mean().item()),
        "step_scale_mae": float(torch.abs(out["step_scale"] - y["step_scale"]).mean().item()),
        "risk_accuracy": float((torch.argmax(out["risk_reason_probability"], dim=-1) == y["risk_reason"]).float().mean().item()),
    }


def train(
    dataset_jsonl: list[Path],
    *,
    output_checkpoint: Path,
    output_json: Path,
    val_fraction: float = 0.2,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    seed: int = 7,
    hidden_dim: int = 96,
    z_gain: float = 0.35,
    yaw_gain: float = 0.25,
    max_z_step: float = 0.0030,
    max_yaw_step: float = 0.020,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    rows = _load_rows(dataset_jsonl)
    train_rows, val_rows = _split_rows(rows, val_fraction=val_fraction, seed=seed)
    x_train, y_train, w_train, kept_train = _build_arrays(train_rows)
    x_val, y_val, w_val, kept_val = _build_arrays(val_rows)
    feature_mean = np.mean(x_train.astype(np.float64), axis=0).astype(np.float32)
    feature_std = np.std(x_train.astype(np.float64), axis=0).astype(np.float32)
    feature_std[~np.isfinite(feature_std) | (np.abs(feature_std) < 1.0e-6)] = 1.0
    model = TaskFrameV45CandidateNet(
        feature_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in y_train.items()}
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32)
    y_val_t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in y_val.items()}
    sampler = WeightedRandomSampler(weights=torch.as_tensor(w_train, dtype=torch.float32), num_samples=len(w_train), replacement=True)
    loader = DataLoader(TensorDataset(x_train_t, y_train_t["dz"], y_train_t["dyaw"], y_train_t["z_ready"], y_train_t["z_conf"], y_train_t["yaw_ready"], y_train_t["yaw_conf"], y_train_t["yaw_ambiguous"], y_train_t["step_scale"], y_train_t["risk_reason"]), batch_size=batch_size, sampler=sampler, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    ce_weight = torch.tensor([1.0, 1.2, 1.2, 1.4, 1.1], dtype=torch.float32, device=device)
    for _ in range(int(epochs)):
        model.train()
        for batch in loader:
            x = batch[0].to(device=device)
            dz_t = batch[1].to(device=device)
            dyaw_t = batch[2].to(device=device)
            z_ready_t = batch[3].to(device=device)
            z_conf_t = batch[4].to(device=device)
            yaw_ready_t = batch[5].to(device=device)
            yaw_conf_t = batch[6].to(device=device)
            yaw_ambiguous_t = batch[7].to(device=device)
            step_scale_t = batch[8].to(device=device)
            risk_reason_t = batch[9].to(device=device).long()
            out = model(x)
            pred_z_step = torch.clamp(out["dz"] * out["step_scale"] * float(z_gain), -float(max_z_step), float(max_z_step))
            pred_yaw_step = torch.clamp(out["dyaw"] * out["step_scale"] * float(yaw_gain), -float(max_yaw_step), float(max_yaw_step))
            post_dz = dz_t - pred_z_step
            post_dyaw = dyaw_t - pred_yaw_step
            z_contraction = F.relu(torch.abs(post_dz) - 0.90 * torch.abs(dz_t))
            yaw_contraction = F.relu(torch.abs(post_dyaw) - 0.90 * torch.abs(dyaw_t))
            z_overshoot = F.relu(-(pred_z_step * dz_t)) + F.relu(torch.abs(pred_z_step) - torch.abs(dz_t))
            yaw_overshoot = F.relu(-(pred_yaw_step * dyaw_t)) + F.relu(torch.abs(pred_yaw_step) - torch.abs(dyaw_t))
            loss = (
                2.0 * F.smooth_l1_loss(out["dz"], dz_t)
                + 2.0 * F.smooth_l1_loss(out["dyaw"], dyaw_t)
                + 0.8 * F.binary_cross_entropy(out["z_ready_probability"], z_ready_t)
                + 0.6 * F.binary_cross_entropy(out["z_confidence_probability"], z_conf_t)
                + 0.9 * F.binary_cross_entropy(out["yaw_ready_probability"], yaw_ready_t)
                + 0.7 * F.binary_cross_entropy(out["yaw_confidence_probability"], yaw_conf_t)
                + 0.6 * F.binary_cross_entropy(out["yaw_ambiguous_probability"], yaw_ambiguous_t)
                + 0.8 * F.mse_loss(out["step_scale"], step_scale_t)
                + 0.5 * F.cross_entropy(out["risk_reason_logits"], risk_reason_t, weight=ce_weight)
                + 1.0 * torch.mean(z_contraction)
                + 1.0 * torch.mean(yaw_contraction)
                + 0.8 * torch.mean(z_overshoot)
                + 0.8 * torch.mean(yaw_overshoot)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    val_metrics = _metrics(model, x_val_t.to(device=device), {k: v.to(device=device) for k, v in y_val_t.items()})
    train_metrics = _metrics(model, x_train_t.to(device=device), {k: v.to(device=device) for k, v in y_train_t.items()})
    metadata = {
        "schema_version": "c2c_v2_task_frame_v45_candidate_report_v1",
        "feature_names": list(TASK_FRAME_READINESS_FEATURE_NAMES),
        "feature_dim": int(x_train.shape[1]),
        "hidden_dim": int(hidden_dim),
        "z_gain": float(z_gain),
        "yaw_gain": float(yaw_gain),
        "max_z_step": float(max_z_step),
        "max_yaw_step": float(max_yaw_step),
        "device": str(device),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "train_source_eval_roots": sorted({source_eval_root_key(r) for r in kept_train}),
        "val_source_eval_roots": sorted({source_eval_root_key(r) for r in kept_val}),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_task_frame_v45_candidate_checkpoint(output_checkpoint, model, metadata=metadata)
    output_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    ap.add_argument("--output_checkpoint", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden_dim", type=int, default=96)
    ap.add_argument("--z_gain", type=float, default=0.35)
    ap.add_argument("--yaw_gain", type=float, default=0.25)
    ap.add_argument("--max_z_step", type=float, default=0.0030)
    ap.add_argument("--max_yaw_step", type=float, default=0.020)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train(
        list(args.dataset_jsonl),
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        val_fraction=float(args.val_fraction),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        z_gain=float(args.z_gain),
        yaw_gain=float(args.yaw_gain),
        max_z_step=float(args.max_z_step),
        max_yaw_step=float(args.max_yaw_step),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
