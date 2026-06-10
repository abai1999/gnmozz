#!/usr/bin/env python3
"""Train a precision-aware yaw-control permission selector for v46.

The v46 yaw observability head currently has a poor precision/recall tradeoff:
raw recall weighting creates many ambiguous/unobservable false positives, while
strict thresholding kills recall.  This script trains a small selector on top of
runtime-visible v46 model outputs.  It is a permission head only: it does not
output actions, does not affect close authority, and uses privileged labels
only as offline train/eval targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.runtime_xy_residual import (  # noqa: E402
    RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
)
from prismatic.robot.coarse2contact_v2.task_frame_v46_alignment import load_task_frame_v46_alignment_checkpoint  # noqa: E402
from prismatic.robot.coarse2contact_v2.task_frame_v46_alignment import task_frame_v46_yaw_selector_feature_names  # noqa: E402
from prismatic.robot.coarse2contact_v2.task_frame_v46_alignment import task_frame_v46_spatial_moment_features  # noqa: E402
from prismatic.robot.coarse2contact_v2.task_frame_readiness import TASK_FRAME_READINESS_FEATURE_NAMES  # noqa: E402
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import source_eval_root_key, split_records_by_source_root  # noqa: E402
from scripts.train_c2c_v2_task_frame_v46_alignment import _build_arrays, _load_rows, _normalize_row_metadata  # noqa: E402


YAW_SELECTOR_FEATURE_NAMES = (
    "v46_yaw_observable_score",
    "v46_yaw_confidence",
    "v46_yaw_ambiguous_score",
    "v46_yaw_step_scale",
    "v46_pred_dyaw",
    "v46_abs_pred_dyaw",
    "v46_yaw_hypothesis_gap",
    "v46_near_field_confidence",
    "v46_xy_observable_score",
    "v46_z_observable_score",
    "v46_xy_confidence",
    "v46_z_confidence",
    "v46_xy_step_scale",
    "v46_z_step_scale",
)


def _feature_names(*, include_scalar_features: bool, include_spatial_moment_features: bool) -> list[str]:
    return list(
        task_frame_v46_yaw_selector_feature_names(
            include_scalar_features=include_scalar_features,
            include_spatial_moment_features=include_spatial_moment_features,
        )
    )


class YawControlSelectorNet(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).reshape(-1)


def _yaw_control_target(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    observable = np.asarray(arrays["observability"], dtype=np.float32)
    ambiguous = np.asarray(arrays["yaw_ambiguous"], dtype=np.float32).reshape(-1)
    return ((observable[:, 2] >= 0.5) & (ambiguous < 0.5)).astype(np.float32)


def _extract_features(
    v46_model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    device: str,
    include_scalar_features: bool,
    include_spatial_moment_features: bool,
) -> np.ndarray:
    features: list[np.ndarray] = []
    v46_model.eval()
    n = int(np.asarray(arrays["residual"]).shape[0])
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            end = min(start + int(batch_size), n)
            image = torch.as_tensor(arrays["image"][start:end], dtype=torch.float32, device=device)
            out = v46_model(
                image,
                torch.as_tensor(arrays["scalar"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays["history"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays["proprio"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays["planner"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays.get("command_6d", np.zeros((n, 6), dtype=np.float32))[start:end], dtype=torch.float32, device=device),
            )
            yaw_probs = torch.softmax(out["yaw_hypothesis_logits"], dim=-1)
            yaw_top2 = torch.topk(yaw_probs, k=min(2, yaw_probs.shape[-1]), dim=-1).values
            if yaw_top2.shape[-1] == 1:
                gap = torch.ones_like(yaw_top2[:, 0])
            else:
                gap = yaw_top2[:, 0] - yaw_top2[:, 1]
            pred_dyaw = out["dyaw"]
            batch = torch.stack(
                [
                    out["axis_observability"][:, 2],
                    out["axis_confidence"][:, 2],
                    out["yaw_ambiguous"],
                    out["axis_step_scale"][:, 2],
                    pred_dyaw,
                    torch.abs(pred_dyaw),
                    gap,
                    out["near_field_confidence"],
                    out["axis_observability"][:, 0],
                    out["axis_observability"][:, 1],
                    out["axis_confidence"][:, 0],
                    out["axis_confidence"][:, 1],
                    out["axis_step_scale"][:, 0],
                    out["axis_step_scale"][:, 1],
                ],
                dim=-1,
            )
            batch_np = batch.detach().cpu().numpy().astype(np.float32)
            if bool(include_scalar_features):
                scalar = np.asarray(arrays["scalar"][start:end], dtype=np.float32)
                batch_np = np.concatenate([batch_np, scalar], axis=-1).astype(np.float32)
            if bool(include_spatial_moment_features):
                spatial = task_frame_v46_spatial_moment_features(image).detach().cpu().numpy().astype(np.float32)
                batch_np = np.concatenate([batch_np, spatial], axis=-1).astype(np.float32)
            features.append(batch_np)
    return np.concatenate(features, axis=0).astype(np.float32)


def _metrics(scores: np.ndarray, target: np.ndarray, *, threshold: float) -> dict[str, Any]:
    pred = np.asarray(scores >= float(threshold), dtype=bool)
    truth = np.asarray(target >= 0.5, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    fpr = float(fp / max(1, fp + tn))
    f1 = float(2.0 * precision * recall / max(1.0e-9, precision + recall))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "f1": f1,
        "predicted_positive_rate": float(np.mean(pred)) if pred.size else 0.0,
        "target_positive_rate": float(np.mean(truth)) if truth.size else 0.0,
    }


def _threshold_sweep(scores: np.ndarray, target: np.ndarray, *, min_precision: float, max_false_positive_rate: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = sorted(set(float(x) for x in np.linspace(0.01, 0.99, 99)) | set(float(x) for x in scores.tolist()))
    rows = [_metrics(scores, target, threshold=t) for t in candidates]
    feasible = [
        row
        for row in rows
        if float(row["precision"]) >= float(min_precision)
        and float(row["false_positive_rate"]) <= float(max_false_positive_rate)
        and int(row["tp"]) > 0
    ]
    if feasible:
        selected = sorted(feasible, key=lambda row: (-float(row["recall"]), -float(row["precision"]), int(row["fp"])))[0]
        selected = dict(selected, selection_reason="max_recall_under_precision_fpr_gate")
    else:
        selected = sorted(rows, key=lambda row: (-float(row["f1"]), -float(row["precision"]), int(row["fp"])))[0]
        selected = dict(selected, selection_reason="best_f1_no_feasible_precision_fpr_gate")
    return rows, selected


def _fit_normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return mean, std


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = source_eval_root_key(row)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def train(
    dataset_jsonl: list[Path],
    *,
    v46_checkpoint: Path,
    output_checkpoint: Path,
    output_json: Path,
    split_mode: str = "root",
    val_fraction: float = 0.2,
    epochs: int = 200,
    lr: float = 1.0e-3,
    hidden_dim: int = 24,
    positive_weight: float = 6.0,
    negative_weight: float = 4.0,
    min_precision: float = 0.80,
    max_false_positive_rate: float = 0.005,
    include_scalar_features: bool = True,
    include_spatial_moment_features: bool = True,
    seed: int = 7,
    image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    history_window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    max_abs_xy_label: float = 0.080,
    max_abs_z_label: float = 0.080,
    max_abs_yaw_label: float = 0.350,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    rows = [_normalize_row_metadata(row) for row in _load_rows(dataset_jsonl)]
    split = split_records_by_source_root(rows, split_mode=split_mode, val_fraction=val_fraction, test_fraction=0.0, seed=seed)
    if not split.val_records:
        raise RuntimeError("yaw-control selector validation split is empty")
    calibration, v46_metadata = load_task_frame_v46_alignment_checkpoint(v46_checkpoint, map_location="cpu")
    v46_model = calibration.model.to(device)

    def build(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        arrays, kept = _build_arrays(
            records,
            image_crop_size=image_crop_size,
            image_resize_size=image_resize_size,
            history_window_size=history_window_size,
            proprio_dim=int(calibration.proprio_dim),
            planner_prior_dim=int(calibration.planner_prior_dim),
            max_abs_xy_label=max_abs_xy_label,
            max_abs_z_label=max_abs_z_label,
            max_abs_yaw_label=max_abs_yaw_label,
        )
        return (
            _extract_features(
                v46_model,
                arrays,
                batch_size=256,
                device=device,
                include_scalar_features=include_scalar_features,
                include_spatial_moment_features=include_spatial_moment_features,
            ),
            _yaw_control_target(arrays),
            kept,
        )

    train_x, train_y, train_kept = build([dict(r) for r in split.train_records])
    val_x, val_y, val_kept = build([dict(r) for r in split.val_records])
    mean, std = _fit_normalizer(train_x)
    train_xn = (train_x - mean.reshape(1, -1)) / std.reshape(1, -1)
    val_xn = (val_x - mean.reshape(1, -1)) / std.reshape(1, -1)
    selector = YawControlSelectorNet(feature_dim=train_xn.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(selector.parameters(), lr=float(lr), weight_decay=1.0e-4)
    x_t = torch.as_tensor(train_xn, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    weights = torch.where(y_t >= 0.5, torch.full_like(y_t, float(positive_weight)), torch.full_like(y_t, float(negative_weight)))
    for _ in range(int(epochs)):
        logits = selector(x_t)
        loss = (F.binary_cross_entropy_with_logits(logits, y_t, reduction="none") * weights).sum() / torch.clamp(weights.sum(), min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    selector.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(selector(torch.as_tensor(train_xn, dtype=torch.float32, device=device))).detach().cpu().numpy()
        val_scores = torch.sigmoid(selector(torch.as_tensor(val_xn, dtype=torch.float32, device=device))).detach().cpu().numpy()
    sweep, selected = _threshold_sweep(val_scores, val_y, min_precision=min_precision, max_false_positive_rate=max_false_positive_rate)
    selected_threshold = float(selected["threshold"])
    report = {
        "schema_version": "c2c_v2_task_frame_v46_yaw_control_selector_report_v1",
        "model": "v46_yaw_control_permission_selector",
        "dataset_jsonl": [str(path) for path in dataset_jsonl],
        "v46_checkpoint": str(v46_checkpoint),
        "v46_checkpoint_metadata_summary": {
            "model": dict(v46_metadata or {}).get("model", ""),
            "train_rows": dict(v46_metadata or {}).get("train_rows", 0),
            "val_rows": dict(v46_metadata or {}).get("val_rows", 0),
        },
        "feature_names": _feature_names(
            include_scalar_features=include_scalar_features,
            include_spatial_moment_features=include_spatial_moment_features,
        ),
        "include_scalar_features": bool(include_scalar_features),
        "include_spatial_moment_features": bool(include_spatial_moment_features),
        "split_mode": str(split.split_mode),
        "train_rows": int(train_x.shape[0]),
        "val_rows": int(val_x.shape[0]),
        "train_positive_rows": int(np.sum(train_y >= 0.5)),
        "val_positive_rows": int(np.sum(val_y >= 0.5)),
        "train_source_eval_roots": list(split.train_source_eval_roots),
        "val_source_eval_roots": list(split.val_source_eval_roots),
        "train_source_eval_root_counts": _source_counts(train_kept),
        "val_source_eval_root_counts": _source_counts(val_kept),
        "positive_weight": float(positive_weight),
        "negative_weight": float(negative_weight),
        "min_precision_gate": float(min_precision),
        "max_false_positive_rate_gate": float(max_false_positive_rate),
        "selected_threshold": selected_threshold,
        "selected_val_metrics": selected,
        "train_metrics_at_selected_threshold": _metrics(train_scores, train_y, threshold=selected_threshold),
        "val_threshold_sweep_best_precision_top10": sorted(sweep, key=lambda row: (-float(row["precision"]), -float(row["recall"]), int(row["fp"])))[:10],
        "val_threshold_sweep_best_f1_top10": sorted(sweep, key=lambda row: (-float(row["f1"]), -float(row["precision"]), int(row["fp"])))[:10],
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_yaw_control_targets_only",
        "close_control_allowed": False,
        "upgrade_gate": "pending_random_heldout_runtime_yaw_control_and_three_axis_contraction",
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "c2c_v2_task_frame_v46_yaw_control_selector_checkpoint_v1",
            "model_type": "v46_yaw_control_permission_selector",
            "feature_names": _feature_names(
                include_scalar_features=include_scalar_features,
                include_spatial_moment_features=include_spatial_moment_features,
            ),
            "feature_mean": mean,
            "feature_std": std,
            "selected_threshold": selected_threshold,
            "model_state_dict": selector.state_dict(),
            "hidden_dim": int(hidden_dim),
            "metadata": report,
        },
        output_checkpoint,
    )
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a precision-aware v46 yaw-control permission selector.")
    parser.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--v46_checkpoint", type=Path, required=True)
    parser.add_argument("--output_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--split_mode", type=str, default="root", choices=("root", "episode", "auto"))
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden_dim", type=int, default=24)
    parser.add_argument("--positive_weight", type=float, default=6.0)
    parser.add_argument("--negative_weight", type=float, default=4.0)
    parser.add_argument("--min_precision", type=float, default=0.80)
    parser.add_argument("--max_false_positive_rate", type=float, default=0.005)
    parser.add_argument("--include_scalar_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_spatial_moment_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image_crop_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE)
    parser.add_argument("--image_resize_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE)
    parser.add_argument("--history_window_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW)
    parser.add_argument("--max_abs_xy_label", type=float, default=0.080)
    parser.add_argument("--max_abs_z_label", type=float, default=0.080)
    parser.add_argument("--max_abs_yaw_label", type=float, default=0.350)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train(
        list(args.dataset_jsonl),
        v46_checkpoint=args.v46_checkpoint,
        output_checkpoint=args.output_checkpoint,
        output_json=args.output_json,
        split_mode=str(args.split_mode),
        val_fraction=float(args.val_fraction),
        epochs=int(args.epochs),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        positive_weight=float(args.positive_weight),
        negative_weight=float(args.negative_weight),
        min_precision=float(args.min_precision),
        max_false_positive_rate=float(args.max_false_positive_rate),
        include_scalar_features=bool(args.include_scalar_features),
        include_spatial_moment_features=bool(args.include_spatial_moment_features),
        seed=int(args.seed),
        image_crop_size=int(args.image_crop_size),
        image_resize_size=int(args.image_resize_size),
        history_window_size=int(args.history_window_size),
        max_abs_xy_label=float(args.max_abs_xy_label),
        max_abs_z_label=float(args.max_abs_z_label),
        max_abs_yaw_label=float(args.max_abs_yaw_label),
        device=str(args.device),
    )
    print(json.dumps({"selected_val_metrics": report["selected_val_metrics"], "train_metrics": report["train_metrics_at_selected_threshold"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
