#!/usr/bin/env python3
"""Evaluate a v46 yaw-control permission selector on offline manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.runtime_xy_residual import (  # noqa: E402
    RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
)
from prismatic.robot.coarse2contact_v2.task_frame_v46_alignment import load_task_frame_v46_alignment_checkpoint  # noqa: E402
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import source_eval_root_key  # noqa: E402
from scripts.train_c2c_v2_task_frame_v46_alignment import _build_arrays, _load_rows, _normalize_row_metadata  # noqa: E402
from scripts.train_c2c_v2_task_frame_v46_yaw_control_selector import (  # noqa: E402
    YawControlSelectorNet,
    _extract_features,
    _metrics,
    _yaw_control_target,
)


def _counts_by(rows: list[dict[str, Any]], pred: np.ndarray, target: np.ndarray, *, key: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row, p, t in zip(rows, pred, target):
        if key == "source_eval_root":
            name = source_eval_root_key(row)
        elif key == "episode_idx":
            name = f"ep{int(row.get('episode_idx', -1)):03d}"
        else:
            name = str(row.get(key, "unknown") or "unknown")
        truth = bool(float(t) >= 0.5)
        guess = bool(float(p) >= 0.5)
        out[name]["rows"] += 1
        if truth:
            out[name]["target_positive"] += 1
        if guess:
            out[name]["predicted_positive"] += 1
        if truth and guess:
            out[name]["tp"] += 1
        elif (not truth) and guess:
            out[name]["fp"] += 1
        elif truth and not guess:
            out[name]["fn"] += 1
        else:
            out[name]["tn"] += 1
    return {name: dict(counter) for name, counter in sorted(out.items())}


def evaluate(
    dataset_jsonl: list[Path],
    *,
    selector_checkpoint: Path,
    v46_checkpoint: Path | None,
    output_json: Path,
    image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    history_window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    max_abs_xy_label: float = 0.080,
    max_abs_z_label: float = 0.080,
    max_abs_yaw_label: float = 0.350,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    payload = torch.load(selector_checkpoint, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    v46_path = Path(str(v46_checkpoint or metadata.get("v46_checkpoint", "")))
    calibration, _ = load_task_frame_v46_alignment_checkpoint(v46_path, map_location="cpu")
    arrays, kept = _build_arrays(
        [_normalize_row_metadata(row) for row in _load_rows(dataset_jsonl)],
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
        history_window_size=history_window_size,
        proprio_dim=int(calibration.proprio_dim),
        planner_prior_dim=int(calibration.planner_prior_dim),
        max_abs_xy_label=max_abs_xy_label,
        max_abs_z_label=max_abs_z_label,
        max_abs_yaw_label=max_abs_yaw_label,
    )
    include_scalar = bool(metadata.get("include_scalar_features", True))
    include_spatial = bool(metadata.get("include_spatial_moment_features", False))
    features = _extract_features(
        calibration.model.to(device),
        arrays,
        batch_size=256,
        device=device,
        include_scalar_features=include_scalar,
        include_spatial_moment_features=include_spatial,
    )
    mean = np.asarray(payload["feature_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(payload["feature_std"], dtype=np.float32).reshape(1, -1)
    selector = YawControlSelectorNet(feature_dim=features.shape[1], hidden_dim=int(payload.get("hidden_dim", 24))).to(device)
    selector.load_state_dict(payload["model_state_dict"], strict=True)
    selector.eval()
    with torch.no_grad():
        scores = torch.sigmoid(
            selector(torch.as_tensor((features - mean) / std, dtype=torch.float32, device=device))
        ).detach().cpu().numpy()
    target = _yaw_control_target(arrays)
    threshold = float(payload.get("selected_threshold", metadata.get("selected_threshold", 0.5)))
    pred_binary = (scores >= threshold).astype(np.float32)
    report = {
        "schema_version": "c2c_v2_task_frame_v46_yaw_control_selector_eval_v1",
        "dataset_jsonl": [str(path) for path in dataset_jsonl],
        "selector_checkpoint": str(selector_checkpoint),
        "v46_checkpoint": str(v46_path),
        "rows": int(scores.shape[0]),
        "threshold": threshold,
        "overall": _metrics(scores, target, threshold=threshold),
        "by_yaw_observability_class": _counts_by(kept, pred_binary, target, key="yaw_observability_class"),
        "by_episode": _counts_by(kept, pred_binary, target, key="episode_idx"),
        "by_source_eval_root": _counts_by(kept, pred_binary, target, key="source_eval_root"),
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_eval": True,
        "privileged_label_boundary": "offline_yaw_control_targets_only",
        "close_control_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a v46 yaw-control permission selector.")
    parser.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--selector_checkpoint", type=Path, required=True)
    parser.add_argument("--v46_checkpoint", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, required=True)
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
    report = evaluate(
        list(args.dataset_jsonl),
        selector_checkpoint=args.selector_checkpoint,
        v46_checkpoint=args.v46_checkpoint,
        output_json=args.output_json,
        image_crop_size=int(args.image_crop_size),
        image_resize_size=int(args.image_resize_size),
        history_window_size=int(args.history_window_size),
        max_abs_xy_label=float(args.max_abs_xy_label),
        max_abs_z_label=float(args.max_abs_z_label),
        max_abs_yaw_label=float(args.max_abs_yaw_label),
        device=str(args.device),
    )
    print(json.dumps({"rows": report["rows"], "overall": report["overall"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
