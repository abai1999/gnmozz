#!/usr/bin/env python3
"""Offline validation for v46 task-frame alignment checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from scripts.train_c2c_v2_task_frame_v46_alignment import (  # noqa: E402
    _build_arrays,
    _controller_bounded_step,
    _effect_aware_xy_step,
    _load_rows,
    _metrics,
    _normalize_row_metadata,
)
from scripts.train_c2c_v2_task_frame_v46_yaw_control_selector import (  # noqa: E402
    YawControlSelectorNet,
    _extract_features as _extract_yaw_selector_features,
    _yaw_control_target,
)


def _slice_arrays(arrays: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    return {key: value[idx] for key, value in arrays.items()}


def _metrics_for_groups(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    kept: list[dict[str, Any]],
    *,
    device: str,
    key: str,
    max_groups: int = 64,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(kept):
        if key == "source_eval_root":
            name = source_eval_root_key(row)
        elif key == "episode_idx":
            name = f"ep{int(row.get('episode_idx', -1)):03d}"
        else:
            name = str(row.get(key, "unknown") or "unknown")
        groups[name].append(idx)
    out: dict[str, Any] = {}
    for name, indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[: int(max_groups)]:
        out[name] = _metrics(model, _slice_arrays(arrays, indices), device=device)
    return out


def _binary_metrics(scores: np.ndarray, target: np.ndarray, *, threshold: float) -> dict[str, Any]:
    pred = np.asarray(scores >= float(threshold), dtype=bool)
    truth = np.asarray(target >= 0.5, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    fpr = float(fp / max(1, fp + tn))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "predicted_positive_rate": float(np.mean(pred)) if pred.size else 0.0,
        "target_positive_rate": float(np.mean(truth)) if truth.size else 0.0,
    }


def _selector_counts_by(
    kept: list[dict[str, Any]],
    scores: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    key: str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(kept):
        if key == "source_eval_root":
            name = source_eval_root_key(row)
        elif key == "episode_idx":
            name = f"ep{int(row.get('episode_idx', -1)):03d}"
        else:
            name = str(row.get(key, "unknown") or "unknown")
        groups[name].append(idx)
    out: dict[str, Any] = {}
    for name, indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:64]:
        idx = np.asarray(indices, dtype=np.int64)
        out[name] = _binary_metrics(np.asarray(scores)[idx], np.asarray(target)[idx], threshold=threshold)
        out[name]["rows"] = int(idx.size)
    return out


def _compute_yaw_selector_outputs(
    selector_checkpoint: Path,
    *,
    v46_model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    device: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    payload = torch.load(selector_checkpoint, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    include_scalar = bool(metadata.get("include_scalar_features", True))
    include_spatial = bool(metadata.get("include_spatial_moment_features", False))
    features = _extract_yaw_selector_features(
        v46_model,
        arrays,
        batch_size=256,
        device=device,
        include_scalar_features=include_scalar,
        include_spatial_moment_features=include_spatial,
    )
    mean = np.asarray(payload["feature_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(payload["feature_std"], dtype=np.float32).reshape(1, -1)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    selector = YawControlSelectorNet(feature_dim=features.shape[1], hidden_dim=int(payload.get("hidden_dim", 24))).to(device)
    selector.load_state_dict(payload["model_state_dict"], strict=True)
    selector.eval()
    with torch.no_grad():
        scores = torch.sigmoid(
            selector(torch.as_tensor((features - mean) / std, dtype=torch.float32, device=device))
        ).detach().cpu().numpy()
    target = _yaw_control_target(arrays)
    threshold = float(payload.get("selected_threshold", metadata.get("selected_threshold", 0.5)))
    meta = {
        "selector_checkpoint": str(selector_checkpoint),
        "threshold": threshold,
        "include_scalar_features": include_scalar,
        "include_spatial_moment_features": include_spatial,
    }
    return meta, scores, target, threshold


def _yaw_selector_report(
    selector_checkpoint: Path,
    *,
    v46_model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    kept: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    meta, scores, target, threshold = _compute_yaw_selector_outputs(
        selector_checkpoint,
        v46_model=v46_model,
        arrays=arrays,
        device=device,
    )
    overall = _binary_metrics(scores, target, threshold=threshold)
    return {
        "schema_version": "c2c_v2_task_frame_v46_yaw_selector_alignment_eval_v1",
        **meta,
        "overall": overall,
        "by_yaw_observability_class": _selector_counts_by(kept, scores, target, threshold=threshold, key="yaw_observability_class"),
        "by_episode": _selector_counts_by(kept, scores, target, threshold=threshold, key="episode_idx"),
        "by_source_eval_root": _selector_counts_by(kept, scores, target, threshold=threshold, key="source_eval_root"),
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_eval": True,
        "privileged_label_boundary": "offline_yaw_control_targets_only",
        "close_control_allowed": False,
    }


def _write_prediction_dump(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    kept: list[dict[str, Any]],
    *,
    output_jsonl: Path,
    device: str,
    yaw_selector_scores: np.ndarray | None = None,
    yaw_selector_target: np.ndarray | None = None,
    yaw_selector_threshold: float | None = None,
) -> None:
    model.eval()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        out = model(
            torch.as_tensor(arrays["image"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["scalar"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["history"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["proprio"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays["planner"], dtype=torch.float32, device=device),
            torch.as_tensor(arrays.get("command_6d", np.zeros((arrays["residual"].shape[0], 6), dtype=np.float32)), dtype=torch.float32, device=device),
        )
    target = torch.as_tensor(arrays["residual"], dtype=torch.float32, device=device)
    pred = torch.stack([out["dx"], out["dy"], out["dz"], out["dyaw"]], dim=-1)
    step_scale = out["axis_step_scale"]
    step = _controller_bounded_step(pred, out, use_predicted_gate=True)
    planner_xy = torch.as_tensor(arrays["planner"][:, :2], dtype=torch.float32, device=device)
    effect_step_xy = _effect_aware_xy_step(target[:, :2], planner_xy, out["xy_control_effect"])
    pred_effect_step_xy = _effect_aware_xy_step(pred[:, :2], planner_xy, out["xy_control_effect"])
    effect_delta_xy = torch.bmm(out["xy_control_effect"], (planner_xy + effect_step_xy).unsqueeze(-1)).squeeze(-1)
    pred_effect_delta_xy = torch.bmm(out["xy_control_effect"], (planner_xy + pred_effect_step_xy).unsqueeze(-1)).squeeze(-1)
    effect_post_xy = target[:, :2] + effect_delta_xy
    pred_effect_post_xy = target[:, :2] + pred_effect_delta_xy
    post = target - step
    pre_norm = torch.linalg.norm(target[:, :3], dim=-1) + torch.abs(target[:, 3])
    post_norm = torch.linalg.norm(post[:, :3], dim=-1) + torch.abs(post[:, 3])
    yaw_probs = torch.softmax(out["yaw_hypothesis_logits"], dim=-1)
    yaw_order = torch.argsort(yaw_probs, dim=-1, descending=True)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    step_np = step.detach().cpu().numpy()
    post_np = post.detach().cpu().numpy()
    effect_step_xy_np = effect_step_xy.detach().cpu().numpy()
    effect_post_xy_np = effect_post_xy.detach().cpu().numpy()
    pred_effect_step_xy_np = pred_effect_step_xy.detach().cpu().numpy()
    pred_effect_post_xy_np = pred_effect_post_xy.detach().cpu().numpy()
    xy_effect_np = out["xy_control_effect"].detach().cpu().numpy()
    command_delta_np = out["command_delta"].detach().cpu().numpy()
    command_logvar_np = out["command_logvar"].detach().cpu().numpy()
    command_support_np = out["command_support"].detach().cpu().numpy()
    command_post_np = (target + out["command_delta"]).detach().cpu().numpy()
    step_scale_np = step_scale.detach().cpu().numpy()
    axis_conf_np = out["axis_confidence"].detach().cpu().numpy()
    axis_obs_np = out["axis_observability"].detach().cpu().numpy()
    near_field_np = out["near_field_confidence"].detach().cpu().numpy()
    yaw_probs_np = yaw_probs.detach().cpu().numpy()
    yaw_order_np = yaw_order.detach().cpu().numpy()
    pre_norm_np = pre_norm.detach().cpu().numpy()
    post_norm_np = post_norm.detach().cpu().numpy()
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for idx, row in enumerate(kept):
            best = int(yaw_order_np[idx, 0]) if yaw_order_np.shape[1] else 0
            second = int(yaw_order_np[idx, 1]) if yaw_order_np.shape[1] > 1 else best
            payload = {
                "episode_idx": int(row.get("episode_idx", -1)),
                "step_idx": int(row.get("step_idx", row.get("step", -1))),
                "source_eval_root": source_eval_root_key(row),
                "sequence_id": str(row.get("sequence_id", "")),
                "trace_path": str(row.get("trace_path", row.get("source_trace_path", ""))),
                "runtime_obs_path": str(row.get("runtime_obs_path", row.get("source_runtime_obs_path", ""))),
                "source_trace_path": str(row.get("source_trace_path", row.get("trace_path", ""))),
                "source_runtime_obs_path": str(row.get("source_runtime_obs_path", row.get("runtime_obs_path", ""))),
                "obs_pointer": row.get("obs_pointer", {}),
                "stage": str(row.get("stage", row.get("stage_name", row.get("source_c2c_stage", "")))),
                "stage_name": str(row.get("stage_name", row.get("source_c2c_stage", row.get("stage", "")))),
                "failure_bucket": str(row.get("failure_bucket", "unknown") or "unknown"),
                "visual_observability_class": str(row.get("visual_observability_class", "unknown") or "unknown"),
                "yaw_observability_class": str(row.get("yaw_observability_class", "unknown") or "unknown"),
                "grasp_probe_local_command_local_6d": row.get("grasp_probe_local_command_local_6d", row.get("planner_local_delta_6d", row.get("planner_prior_delta", [0.0] * 6))),
                "pre_clip_action_local_6d": row.get("pre_clip_action_local_6d", row.get("planner_local_delta_6d", row.get("planner_prior_delta", [0.0] * 6))),
                "planner_local_delta_6d": row.get("planner_local_delta_6d", row.get("planner_prior_delta", [0.0] * 6)),
                "target": target_np[idx].tolist(),
                "pred": pred_np[idx].tolist(),
                "bounded_step": step_np[idx].tolist(),
                "post": post_np[idx].tolist(),
                "xy_control_effect": xy_effect_np[idx].tolist(),
                "effect_aware_xy_step": effect_step_xy_np[idx].tolist(),
                "effect_aware_xy_post": effect_post_xy_np[idx].tolist(),
                "predicted_effect_aware_xy_step": pred_effect_step_xy_np[idx].tolist(),
                "predicted_effect_aware_xy_post": pred_effect_post_xy_np[idx].tolist(),
                "command_transition_delta": command_delta_np[idx].tolist(),
                "command_transition_logvar": command_logvar_np[idx].tolist(),
                "command_transition_support": float(command_support_np[idx]),
                "command_transition_post": command_post_np[idx].tolist(),
                "pre_norm": float(pre_norm_np[idx]),
                "post_norm": float(post_norm_np[idx]),
                "contracted": bool(post_norm_np[idx] < pre_norm_np[idx]),
                "axis_step_scale": step_scale_np[idx].tolist(),
                "axis_confidence": axis_conf_np[idx].tolist(),
                "axis_observability": axis_obs_np[idx].tolist(),
                "near_field_confidence": float(near_field_np[idx]),
                "yaw_hypothesis_index": best,
                "yaw_hypothesis_gap": float(yaw_probs_np[idx, best] - yaw_probs_np[idx, second]),
                "yaw_hypothesis_probs": yaw_probs_np[idx].tolist(),
                "uses_privileged_runtime": False,
                "uses_privileged_label_for_eval": True,
                "privileged_label_boundary": "offline_prediction_dump_labels_only",
            }
            if yaw_selector_scores is not None and yaw_selector_threshold is not None:
                score = float(np.asarray(yaw_selector_scores, dtype=np.float32).reshape(-1)[idx])
                selector_target = float(np.asarray(yaw_selector_target, dtype=np.float32).reshape(-1)[idx]) if yaw_selector_target is not None else float("nan")
                payload["task_frame_v46_yaw_selector_score"] = score
                payload["task_frame_v46_yaw_selector_threshold"] = float(yaw_selector_threshold)
                payload["task_frame_v46_yaw_selector_allowed"] = bool(score >= float(yaw_selector_threshold))
                payload["task_frame_v46_yaw_selector_target"] = selector_target
                payload["task_frame_v46_yaw_selector_close_control_allowed"] = False
                payload["task_frame_v46_yaw_selector_uses_privileged_runtime"] = False
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def evaluate(
    dataset_jsonl: list[Path],
    *,
    checkpoint: Path,
    output_json: Path,
    image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    history_window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    max_abs_xy_label: float = 0.080,
    max_abs_z_label: float = 0.080,
    max_abs_yaw_label: float = 0.350,
    near_field_xy_radius: float = 0.060,
    near_field_z_radius: float = 0.040,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    prediction_dump_jsonl: Path | None = None,
    yaw_selector_checkpoint: Path | None = None,
) -> dict[str, Any]:
    calibration, metadata = load_task_frame_v46_alignment_checkpoint(checkpoint, map_location="cpu")
    model = calibration.model.to(device)
    rows = [_normalize_row_metadata(row) for row in _load_rows(dataset_jsonl)]
    arrays, kept = _build_arrays(
        rows,
        image_crop_size=image_crop_size,
        image_resize_size=image_resize_size,
        history_window_size=history_window_size,
        proprio_dim=int(calibration.proprio_dim),
        planner_prior_dim=int(calibration.planner_prior_dim),
        max_abs_xy_label=max_abs_xy_label,
        max_abs_z_label=max_abs_z_label,
        max_abs_yaw_label=max_abs_yaw_label,
        near_field_xy_radius=near_field_xy_radius,
        near_field_z_radius=near_field_z_radius,
    )
    report = {
        "schema_version": "c2c_v2_task_frame_v46_alignment_eval_v1",
        "model": "v46_unified_task_frame_alignment_candidate",
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "dataset_jsonl": [str(path) for path in dataset_jsonl],
        "input_rows": len(rows),
        "eval_rows": int(arrays["image"].shape[0]),
        "source_eval_roots": sorted({source_eval_root_key(row) for row in kept}),
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_eval": True,
        "privileged_label_boundary": "offline_labels_only",
        "near_field_xy_radius": float(near_field_xy_radius),
        "near_field_z_radius": float(near_field_z_radius),
        "overall": _metrics(model, arrays, device=device),
        "by_failure_bucket": _metrics_for_groups(model, arrays, kept, device=device, key="failure_bucket"),
        "by_episode": _metrics_for_groups(model, arrays, kept, device=device, key="episode_idx"),
        "by_yaw_observability_class": _metrics_for_groups(model, arrays, kept, device=device, key="yaw_observability_class"),
        "by_visual_observability_class": _metrics_for_groups(model, arrays, kept, device=device, key="visual_observability_class"),
        "by_source_eval_root": _metrics_for_groups(model, arrays, kept, device=device, key="source_eval_root", max_groups=32),
        "upgrade_gate": "pending_closed_loop_random_holdout_and_insert_success",
    }
    if yaw_selector_checkpoint is not None:
        _, selector_scores, selector_target, selector_threshold = _compute_yaw_selector_outputs(
            yaw_selector_checkpoint,
            v46_model=model,
            arrays=arrays,
            device=device,
        )
        selector_report = _yaw_selector_report(
            yaw_selector_checkpoint,
            v46_model=model,
            arrays=arrays,
            kept=kept,
            device=device,
        )
        report["yaw_selector_checkpoint"] = str(yaw_selector_checkpoint)
        report["yaw_selector"] = selector_report
        selector_overall = dict(selector_report.get("overall", {}))
        report["overall"]["yaw_selector_control_predicted_rate"] = float(selector_overall.get("predicted_positive_rate", 0.0))
        report["overall"]["yaw_selector_control_target_rate"] = float(selector_overall.get("target_positive_rate", 0.0))
        report["overall"]["yaw_selector_precision"] = float(selector_overall.get("precision", 0.0))
        report["overall"]["yaw_selector_recall"] = float(selector_overall.get("recall", 0.0))
        report["overall"]["yaw_selector_false_positive_rate"] = float(selector_overall.get("false_positive_rate", 0.0))
        report["overall"]["yaw_selector_false_positives"] = int(selector_overall.get("fp", 0))
    else:
        selector_scores = None
        selector_target = None
        selector_threshold = None
    if prediction_dump_jsonl is not None:
        _write_prediction_dump(
            model,
            arrays,
            kept,
            output_jsonl=prediction_dump_jsonl,
            device=device,
            yaw_selector_scores=selector_scores,
            yaw_selector_target=selector_target,
            yaw_selector_threshold=selector_threshold,
        )
        report["prediction_dump_jsonl"] = str(prediction_dump_jsonl)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a v46 task-frame alignment checkpoint on an offline manifest.")
    parser.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--image_crop_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE)
    parser.add_argument("--image_resize_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE)
    parser.add_argument("--history_window_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW)
    parser.add_argument("--max_abs_xy_label", type=float, default=0.080)
    parser.add_argument("--max_abs_z_label", type=float, default=0.080)
    parser.add_argument("--max_abs_yaw_label", type=float, default=0.350)
    parser.add_argument("--near_field_xy_radius", type=float, default=0.060)
    parser.add_argument("--near_field_z_radius", type=float, default=0.040)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction_dump_jsonl", type=Path, default=None)
    parser.add_argument("--yaw_selector_checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        list(args.dataset_jsonl),
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        image_crop_size=int(args.image_crop_size),
        image_resize_size=int(args.image_resize_size),
        history_window_size=int(args.history_window_size),
        max_abs_xy_label=float(args.max_abs_xy_label),
        max_abs_z_label=float(args.max_abs_z_label),
        max_abs_yaw_label=float(args.max_abs_yaw_label),
        near_field_xy_radius=float(args.near_field_xy_radius),
        near_field_z_radius=float(args.near_field_z_radius),
        device=str(args.device),
        prediction_dump_jsonl=args.prediction_dump_jsonl,
        yaw_selector_checkpoint=args.yaw_selector_checkpoint,
    )
    print(json.dumps({"eval_rows": report["eval_rows"], "overall": report["overall"], "upgrade_gate": report["upgrade_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
