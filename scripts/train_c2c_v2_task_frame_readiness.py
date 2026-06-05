#!/usr/bin/env python3
"""Train non-privileged task-frame Z/Yaw readiness heads for C2C v2."""

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

from prismatic.robot.coarse2contact_v2.task_frame_readiness import (  # noqa: E402
    TASK_FRAME_READINESS_FEATURE_NAMES,
    TaskFrameReadinessNet,
    load_task_frame_readiness_checkpoint,
    save_task_frame_readiness_checkpoint,
    task_frame_readiness_feature_vector,
    task_frame_yaw_label_from_row,
    task_frame_z_label_from_row,
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
    rows.sort(key=lambda row: (str(row.get("source_eval_root", "")), int(row.get("episode_idx", -1)), int(row.get("step", row.get("step_idx", -1)))))
    return rows


def _root_key(row: Mapping[str, Any]) -> str:
    root = str(row.get("source_eval_root", "") or "").strip()
    if root:
        return root
    trace_path = str(row.get("trace_path", row.get("source_trace_path", "")) or "").strip()
    if trace_path:
        return trace_path.rsplit("/", 1)[0]
    return f"ep{int(row.get('episode_idx', -1)):03d}"


def _split_rows(rows: list[dict[str, Any]], *, split_mode: str, val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    mode = str(split_mode).lower().strip()
    if mode not in {"root", "episode"}:
        raise ValueError(f"invalid split_mode: {split_mode}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _root_key(row) if mode == "root" else f"ep{int(row.get('episode_idx', -1)):03d}"
        groups.setdefault(key, []).append(dict(row))
    keys = sorted(groups.keys())
    rng = np.random.default_rng(int(seed))
    rng.shuffle(keys)
    if len(keys) <= 1:
        return rows, rows[: max(1, len(rows) // 5)]
    n_val = max(1, int(round(len(keys) * float(val_fraction))))
    val_keys = set(keys[:n_val])
    train = [row for key, group in groups.items() if key not in val_keys for row in group]
    val = [row for key, group in groups.items() if key in val_keys for row in group]
    if not train:
        train, val = val, train
    return train, val


def _head_labels(row: Mapping[str, Any], head: str) -> tuple[np.ndarray, dict[str, float]]:
    if head == "z":
        ready, observable, near, contact, valid = task_frame_z_label_from_row(row)
        y = np.asarray([ready, observable, near, contact], dtype=np.float32)
        meta = {
            "z_ready": float(ready),
            "z_observable": float(observable),
            "z_near_alignment": float(near),
            "z_contact_or_depth_ready": float(contact),
            "label_valid": float(valid),
        }
        return y, meta
    ready, observable, ambiguous, unobservable, valid = task_frame_yaw_label_from_row(row)
    y = np.asarray([ready, observable, ambiguous], dtype=np.float32)
    meta = {
        "yaw_ready": float(ready),
        "yaw_observable": float(observable),
        "yaw_ambiguous": float(ambiguous),
        "yaw_unobservable": float(unobservable),
        "label_valid": float(valid),
    }
    return y, meta


def _sample_weight(row: Mapping[str, Any], head: str, y: np.ndarray) -> float:
    weight = 1.0
    bucket = str(row.get("failure_bucket", row.get("bucket", "")) or "")
    obs_bucket = str(row.get("observability_bucket", "") or "")
    if bucket in {"large_xy_large_yaw", "small_xy_large_yaw"}:
        weight *= 1.5
    if obs_bucket in {"occluded", "low_observability", "low_visibility", "partial_observable", "partial_observation"}:
        weight *= 1.6
    if head == "yaw":
        alias = str(row.get("alias_drift_decision", "") or "")
        if alias == "frame_drift_abstain":
            weight *= 1.8
        if alias == "unknown":
            weight *= 1.3
    if bool(row.get("grasp_probe_active", False)):
        weight *= 1.4
    return float(weight)


def _build_arrays(rows: list[dict[str, Any]], *, head: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[float] = []
    kept: list[dict[str, Any]] = []
    for row in rows:
        x = task_frame_readiness_feature_vector(row)
        y, _meta = _head_labels(row, head)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            continue
        features.append(x)
        labels.append(y)
        weights.append(_sample_weight(row, head, y))
        kept.append(dict(row))
    if not features:
        return (
            np.zeros((0, len(TASK_FRAME_READINESS_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
        )
    return np.stack(features).astype(np.float32), np.stack(labels).astype(np.float32), np.asarray(weights, dtype=np.float32), kept


def _metrics(model: TaskFrameReadinessNet, x: torch.Tensor, y: torch.Tensor, *, head: str, threshold: float) -> dict[str, Any]:
    if x.numel() == 0:
        return {"rows": 0}
    model.eval()
    with torch.no_grad():
        out = model(x)
    if head == "z":
        ready_prob = out["z_ready_probability"]
        observable_prob = out["z_observable_probability"]
        near_prob = out["z_near_alignment_probability"]
        contact_prob = out["z_contact_or_depth_ready_probability"]
        ready_pred = ready_prob >= threshold
        ready_target = y[:, 0] >= 0.5
        observable_target = y[:, 1] >= 0.5
        near_target = y[:, 2] >= 0.5
        contact_target = y[:, 3] >= 0.5
        return {
            "rows": int(x.shape[0]),
            "ready_precision": float(((ready_pred & ready_target).float().sum() / torch.clamp(ready_pred.float().sum(), min=1.0)).item()),
            "ready_recall": float(((ready_pred & ready_target).float().sum() / torch.clamp(ready_target.float().sum(), min=1.0)).item()),
            "ready_rate": float(ready_pred.float().mean().item()),
            "ready_target_rate": float(ready_target.float().mean().item()),
            "observable_precision": float((((observable_prob >= threshold) & observable_target).float().sum() / torch.clamp((observable_prob >= threshold).float().sum(), min=1.0)).item()),
            "near_alignment_rate": float(near_target.float().mean().item()),
            "contact_ready_rate": float(contact_target.float().mean().item()),
            "ready_probability_mean": float(ready_prob.mean().item()),
            "observable_probability_mean": float(observable_prob.mean().item()),
            "near_probability_mean": float(near_prob.mean().item()),
            "contact_probability_mean": float(contact_prob.mean().item()),
        }
    ready_prob = out["yaw_ready_probability"]
    observable_prob = out["yaw_observable_probability"]
    ambiguous_prob = out["yaw_ambiguous_probability"]
    ready_pred = ready_prob >= threshold
    ready_target = y[:, 0] >= 0.5
    observable_target = y[:, 1] >= 0.5
    ambiguous_target = y[:, 2] >= 0.5
    return {
        "rows": int(x.shape[0]),
        "ready_precision": float(((ready_pred & ready_target).float().sum() / torch.clamp(ready_pred.float().sum(), min=1.0)).item()),
        "ready_recall": float(((ready_pred & ready_target).float().sum() / torch.clamp(ready_target.float().sum(), min=1.0)).item()),
        "ready_rate": float(ready_pred.float().mean().item()),
        "ready_target_rate": float(ready_target.float().mean().item()),
        "observable_precision": float((((observable_prob >= threshold) & observable_target).float().sum() / torch.clamp((observable_prob >= threshold).float().sum(), min=1.0)).item()),
        "ambiguous_rate": float(ambiguous_target.float().mean().item()),
        "ready_probability_mean": float(ready_prob.mean().item()),
        "observable_probability_mean": float(observable_prob.mean().item()),
        "ambiguous_probability_mean": float(ambiguous_prob.mean().item()),
    }


def _threshold_sweep(prob: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if prob.numel() == 0:
        return {"best_threshold": 0.5, "best_precision": 0.0, "best_recall": 0.0, "best_f1": 0.0}
    best = {"best_threshold": 0.5, "best_precision": 0.0, "best_recall": 0.0, "best_f1": 0.0, "best_accuracy": 0.0}
    thresholds = torch.linspace(0.05, 0.95, 37, dtype=torch.float32)
    truth = target >= 0.5
    for thr in thresholds:
        pred = prob >= thr
        tp = int(torch.count_nonzero(pred & truth).item())
        fp = int(torch.count_nonzero(pred & ~truth).item())
        fn = int(torch.count_nonzero(~pred & truth).item())
        tn = int(torch.count_nonzero(~pred & ~truth).item())
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2.0 * precision * recall / max(precision + recall, 1.0e-9))
        accuracy = float((pred == truth).float().mean().item())
        if f1 > best["best_f1"] or (f1 == best["best_f1"] and precision > best["best_precision"]):
            best = {
                "best_threshold": float(thr.item()),
                "best_precision": precision,
                "best_recall": recall,
                "best_f1": f1,
                "best_accuracy": accuracy,
            }
    return best


def train(
    dataset_jsonl: list[Path],
    *,
    head: str,
    output_checkpoint: Path,
    output_json: Path,
    split_mode: str = "root",
    val_fraction: float = 0.2,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    seed: int = 7,
    hidden_dim: int = 96,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    rows = _load_rows(dataset_jsonl)
    train_rows, val_rows = _split_rows(rows, split_mode=split_mode, val_fraction=val_fraction, seed=seed)
    x_train, y_train, w_train, kept_train = _build_arrays(train_rows, head=head)
    x_val, y_val, w_val, kept_val = _build_arrays(val_rows, head=head)
    if x_train.size == 0 or x_val.size == 0:
        raise RuntimeError("need non-empty train and val readiness rows")
    feature_mean = np.mean(x_train.astype(np.float64), axis=0).astype(np.float32)
    feature_std = np.std(x_train.astype(np.float64), axis=0).astype(np.float32)
    feature_std[~np.isfinite(feature_std) | (np.abs(feature_std) < 1.0e-6)] = 1.0
    model = TaskFrameReadinessNet(
        head_type=head,
        feature_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    train_x = torch.as_tensor(x_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    train_w = torch.as_tensor(w_train, dtype=torch.float32)
    val_x = torch.as_tensor(x_val, dtype=torch.float32)
    val_y = torch.as_tensor(y_val, dtype=torch.float32)
    sampler = WeightedRandomSampler(
        weights=torch.clamp(train_w, min=1.0e-6),
        num_samples=int(train_w.shape[0]),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    loader = DataLoader(TensorDataset(train_x, train_y, train_w), batch_size=int(batch_size), sampler=sampler)
    pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("inf")
    best_epoch = 0
    best_threshold = 0.5
    best_val_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    threshold = 0.5
    for epoch in range(int(epochs)):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            out = model(xb)
            if head == "z":
                ready = out["z_ready_logit"]
                observable = out["z_observable_logit"]
                near = out["z_near_alignment_logit"]
                contact = out["z_contact_or_depth_ready_logit"]
                conf = out["z_confidence_logit"]
                loss_ready = F.binary_cross_entropy_with_logits(ready, yb[:, 0], pos_weight=pos_weight, reduction="none")
                loss_observable = F.binary_cross_entropy_with_logits(observable, yb[:, 1], pos_weight=pos_weight, reduction="none")
                loss_near = F.binary_cross_entropy_with_logits(near, yb[:, 2], pos_weight=pos_weight, reduction="none")
                loss_contact = F.binary_cross_entropy_with_logits(contact, yb[:, 3], pos_weight=pos_weight, reduction="none")
                conf_target = torch.exp(-torch.abs(yb[:, 0] - 1.0))
                loss_conf = F.binary_cross_entropy_with_logits(conf, conf_target, reduction="none")
                loss = (loss_ready + 0.6 * loss_observable + 0.4 * loss_near + 0.5 * loss_contact + 0.2 * loss_conf) * wb
            else:
                ready = out["yaw_ready_logit"]
                observable = out["yaw_observable_logit"]
                ambiguous = out["yaw_ambiguous_logit"]
                conf = out["yaw_confidence_logit"]
                loss_ready = F.binary_cross_entropy_with_logits(ready, yb[:, 0], pos_weight=pos_weight, reduction="none")
                loss_observable = F.binary_cross_entropy_with_logits(observable, yb[:, 1], pos_weight=pos_weight, reduction="none")
                loss_ambiguous = F.binary_cross_entropy_with_logits(ambiguous, yb[:, 2], reduction="none")
                conf_target = torch.exp(-torch.abs(yb[:, 0] - 1.0))
                loss_conf = F.binary_cross_entropy_with_logits(conf, conf_target, reduction="none")
                loss = (loss_ready + 0.6 * loss_observable + 0.4 * loss_ambiguous + 0.2 * loss_conf) * wb
            loss = torch.sum(loss) / torch.clamp(torch.sum(wb), min=1.0e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        with torch.no_grad():
            model.eval()
            if head == "z":
                val_ready_prob = model(val_x.to(device))["z_ready_probability"].detach().cpu()
                sweep = _threshold_sweep(val_ready_prob, val_y[:, 0])
                threshold = float(sweep["best_threshold"])
            else:
                val_ready_prob = model(val_x.to(device))["yaw_ready_probability"].detach().cpu()
                sweep = _threshold_sweep(val_ready_prob, val_y[:, 0])
                threshold = float(sweep["best_threshold"])
        val_metrics = _metrics(model, val_x.to(device), val_y.to(device), head=head, threshold=threshold)
        val_score = float(1.0 - val_metrics.get("ready_precision", 0.0) + 0.5 * (1.0 - val_metrics.get("ready_recall", 0.0)))
        history.append({"epoch": int(epoch + 1), "val": val_metrics, "threshold": float(threshold), "score": float(val_score)})
        if val_score < best_score:
            best_score = float(val_score)
            best_epoch = int(epoch + 1)
            best_threshold = float(threshold)
            best_val_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = {
                "schema_version": "c2c_v2_task_frame_readiness_checkpoint_v1",
                "head_type": str(head),
                "feature_names": list(TASK_FRAME_READINESS_FEATURE_NAMES),
                "feature_dim": int(x_train.shape[1]),
                "hidden_dim": int(hidden_dim),
                "model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
                "metadata": {
                    "best_epoch": int(best_epoch),
                    "val_threshold": float(best_threshold),
                    "val_metrics": val_metrics,
                    "split_mode": str(split_mode),
                    "train_rows": int(x_train.shape[0]),
                    "val_rows": int(x_val.shape[0]),
                },
            }
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_checkpoint)

    model.load_state_dict(best_state)
    final_val_metrics = best_val_metrics if best_val_metrics else _metrics(model, val_x.to(device), val_y.to(device), head=head, threshold=best_threshold)
    report = {
        "schema_version": "c2c_v2_task_frame_readiness_train_v1",
        "head_type": str(head),
        "dataset_jsonl": [str(path.resolve()) for path in dataset_jsonl],
        "output_checkpoint": str(output_checkpoint.resolve()),
        "rows": int(len(rows)),
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "feature_names": list(TASK_FRAME_READINESS_FEATURE_NAMES),
        "split_mode": str(split_mode),
        "val_fraction": float(val_fraction),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "train_source_eval_roots": sorted({str(row.get("source_eval_root", "")) for row in kept_train}),
        "val_source_eval_roots": sorted({str(row.get("source_eval_root", "")) for row in kept_val}),
        "train_metrics": _metrics(model, train_x.to(device), train_y.to(device), head=head, threshold=threshold),
        "val_metrics": final_val_metrics,
        "history": history,
        "runtime_policy": {
            "uses_privileged_runtime": False,
            "strict_handoff_only": True,
            "open_close_authority": False,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train C2C v2 task-frame Z/Yaw readiness heads.")
    ap.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    ap.add_argument("--head", type=str, required=True, choices=["z", "yaw"])
    ap.add_argument("--output_checkpoint", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--split_mode", type=str, default="root", choices=["root", "episode"])
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden_dim", type=int, default=96)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    report = train(
        [Path(p) for p in args.dataset_jsonl],
        head=str(args.head),
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        split_mode=str(args.split_mode),
        val_fraction=float(args.val_fraction),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        device=str(args.device),
    )
    print(json.dumps({"head_type": report["head_type"], "best_epoch": report["best_epoch"], "best_threshold": report["best_threshold"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
