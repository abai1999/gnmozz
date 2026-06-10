#!/usr/bin/env python3
"""Audit seed-label vs v46-model yaw observability without executing RLBench.

This audit is meant for the current v46 stage: some offline/source rows are
selected as yaw-visible candidates, but runtime execution can still show the
v46 head declaring yaw ambiguous/unobservable.  Running the environment for
every candidate is expensive, so this script reuses saved runtime observations
and the current v46 checkpoint to measure whether the model head itself agrees
with the label-side yaw-control evidence.

Privileged residuals/labels are used only as audit targets.  The model inputs
come from runtime-visible RGBD/proprio/planner/history fields through the same
array builder used by v46 training/evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

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
    _load_rows,
    _normalize_row_metadata,
)


DEFAULT_SWEEP_THRESHOLDS = (0.30, 0.50, 0.70, 0.80, 0.90, 0.95)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(float(value) > 0.5)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(default)


def _seed_enriched_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose command-sweep runtime_trace_fields as normal label fields.

    Command-sweep specs intentionally keep source labels under
    ``runtime_trace_fields`` because candidate commands do not have transition
    labels yet.  For this audit we need those seed labels to build the same
    runtime-visible inputs and compare model observability against label-side
    yaw-control evidence.
    """

    out = _normalize_row_metadata(row)
    runtime = out.get("runtime_trace_fields", {})
    runtime = runtime if isinstance(runtime, Mapping) else {}
    for axis in ("dx", "dy", "dz", "dyaw"):
        label_key = f"task_frame_v46_label_{axis}"
        if f"privileged_{axis}" not in out and label_key in runtime:
            out[f"privileged_{axis}"] = runtime.get(label_key)
    if "yaw_observability_class" not in out:
        value = runtime.get("task_frame_v46_yaw_observability_class", runtime.get("yaw_observability_class", ""))
        if value:
            out["yaw_observability_class"] = value
    if "yaw_observable" not in out and "task_frame_v46_label_yaw_observable" in runtime:
        out["yaw_observable"] = _safe_bool(runtime.get("task_frame_v46_label_yaw_observable"), False)
        out["yaw_control_observable"] = bool(out["yaw_observable"])
    if "yaw_ambiguous" not in out and "task_frame_v46_label_yaw_ambiguous" in runtime:
        out["yaw_ambiguous"] = _safe_bool(runtime.get("task_frame_v46_label_yaw_ambiguous"), False)
    if "failure_bucket" not in out and "failure_bucket" in runtime:
        out["failure_bucket"] = runtime.get("failure_bucket")
    if "visual_observability_class" not in out and "visual_observability_class" in runtime:
        out["visual_observability_class"] = runtime.get("visual_observability_class")
    return out


def _seed_control_from_row(row: Mapping[str, Any]) -> bool:
    yaw_class = str(row.get("yaw_observability_class", "") or "").strip().lower()
    observable = _safe_bool(row.get("yaw_observable", row.get("yaw_control_observable", yaw_class == "observable")), yaw_class == "observable")
    ambiguous = _safe_bool(row.get("yaw_ambiguous", yaw_class in {"ambiguous", "unobservable"}), yaw_class in {"ambiguous", "unobservable"})
    if yaw_class == "unobservable":
        observable = False
    if yaw_class == "ambiguous":
        ambiguous = True
    return bool(observable and not ambiguous)


def _model_block_reason(
    *,
    pred_observable: bool,
    pred_confidence: float,
    pred_ambiguous: bool,
    pred_step_scale: float,
    confidence_threshold: float,
    step_threshold: float,
) -> str:
    if not pred_observable:
        return "model_yaw_not_observable"
    if pred_confidence < float(confidence_threshold):
        return "model_yaw_low_confidence"
    if pred_ambiguous:
        return "model_yaw_ambiguous"
    if pred_step_scale < float(step_threshold):
        return "model_yaw_low_step_scale"
    return "ready"


def _dedupe_sorted(values: list[float] | tuple[float, ...]) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values):
        if not out or abs(float(value) - float(out[-1])) > 1.0e-9:
            out.append(float(value))
    return out


def _confusion_metrics(*, tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    false_positive_rate = float(fp / max(1, fp + tn))
    f1 = float(2.0 * precision * recall / max(1.0e-9, precision + recall))
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
    }


def audit_model_observability(
    dataset_jsonl: list[Path],
    *,
    checkpoint: Path,
    output_json: Path,
    output_jsonl: Path | None = None,
    image_crop_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE,
    image_resize_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE,
    history_window_size: int = RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW,
    max_abs_xy_label: float = 0.080,
    max_abs_z_label: float = 0.080,
    max_abs_yaw_label: float = 0.350,
    yaw_observable_threshold: float = 0.5,
    yaw_confidence_threshold: float = 0.45,
    yaw_ambiguous_threshold: float = 0.5,
    yaw_step_scale_threshold: float = 0.05,
    sweep_yaw_observable_thresholds: tuple[float, ...] | list[float] = (),
    sweep_yaw_confidence_thresholds: tuple[float, ...] | list[float] = (),
    sweep_yaw_ambiguous_thresholds: tuple[float, ...] | list[float] = (),
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    calibration, metadata = load_task_frame_v46_alignment_checkpoint(checkpoint, map_location="cpu")
    rows = [_seed_enriched_row(row) for row in _load_rows(dataset_jsonl)]
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
    )
    model = calibration.model.to(device)
    model.eval()
    counters: Counter[str] = Counter()
    by_yaw_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_root: dict[str, Counter[str]] = defaultdict(Counter)
    enriched_rows: list[dict[str, Any]] = []
    raw_scores: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, int(arrays["residual"].shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(arrays["residual"].shape[0]))
            out = model(
                torch.as_tensor(arrays["image"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays["scalar"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays["history"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays["proprio"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays["planner"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arrays.get("command_6d", np.zeros((end - start, 6), dtype=np.float32))[start:end], dtype=torch.float32, device=device),
            )
            residual = arrays["residual"][start:end]
            axis_observable = out["axis_observability"].detach().cpu().numpy()
            axis_confidence = out["axis_confidence"].detach().cpu().numpy()
            step_scale = out["axis_step_scale"].detach().cpu().numpy()
            yaw_ambiguous = out["yaw_ambiguous"].detach().cpu().numpy()
            pred = torch.stack([out["dx"], out["dy"], out["dz"], out["dyaw"]], dim=-1).detach().cpu().numpy()
            for local_idx, row in enumerate(kept[start:end]):
                idx = start + local_idx
                counters["rows"] += 1
                seed_control = _seed_control_from_row(row)
                yaw_class = str(row.get("yaw_observability_class", "unknown") or "unknown").strip().lower() or "unknown"
                pred_observable = bool(axis_observable[local_idx, 2] >= float(yaw_observable_threshold))
                pred_observable_score = float(axis_observable[local_idx, 2])
                pred_confidence = float(axis_confidence[local_idx, 2])
                pred_ambiguous_score = float(yaw_ambiguous[local_idx])
                pred_ambiguous = bool(pred_ambiguous_score >= float(yaw_ambiguous_threshold))
                pred_step_scale = float(step_scale[local_idx, 2])
                block_reason = _model_block_reason(
                    pred_observable=pred_observable,
                    pred_confidence=pred_confidence,
                    pred_ambiguous=pred_ambiguous,
                    pred_step_scale=pred_step_scale,
                    confidence_threshold=yaw_confidence_threshold,
                    step_threshold=yaw_step_scale_threshold,
                )
                model_control = bool(block_reason == "ready")
                if seed_control:
                    counters["seed_yaw_control_rows"] += 1
                if model_control:
                    counters["model_yaw_control_rows"] += 1
                if seed_control and not model_control:
                    counters["seed_control_to_model_noncontrol"] += 1
                if (not seed_control) and model_control:
                    counters["seed_noncontrol_to_model_control"] += 1
                counters[f"block_{block_reason}"] += 1
                for group in (by_yaw_class[yaw_class], by_root[source_eval_root_key(row)]):
                    group["rows"] += 1
                    if seed_control:
                        group["seed_yaw_control_rows"] += 1
                    if model_control:
                        group["model_yaw_control_rows"] += 1
                    if seed_control and not model_control:
                        group["seed_control_to_model_noncontrol"] += 1
                    group[f"block_{block_reason}"] += 1
                raw_scores.append(
                    {
                        "seed_yaw_control": bool(seed_control),
                        "yaw_observable_score": pred_observable_score,
                        "yaw_confidence": pred_confidence,
                        "yaw_ambiguous_score": pred_ambiguous_score,
                        "yaw_step_scale": pred_step_scale,
                    }
                )
                enriched_rows.append(
                    {
                        "source_eval_root": source_eval_root_key(row),
                        "episode_idx": int(row.get("episode_idx", -1)),
                        "step_idx": int(row.get("step_idx", row.get("step", -1))),
                        "yaw_observability_class": yaw_class,
                        "seed_yaw_control": bool(seed_control),
                        "model_yaw_control": bool(model_control),
                        "model_yaw_block_reason": block_reason,
                        "model_yaw_observable": bool(pred_observable),
                        "model_yaw_observable_score": pred_observable_score,
                        "model_yaw_confidence": pred_confidence,
                        "model_yaw_ambiguous": bool(pred_ambiguous),
                        "model_yaw_ambiguous_score": pred_ambiguous_score,
                        "model_yaw_step_scale": pred_step_scale,
                        "label_dyaw": float(residual[local_idx, 3]),
                        "pred_dyaw": float(pred[local_idx, 3]),
                    }
                )

    for key in (
        "seed_yaw_control_rows",
        "model_yaw_control_rows",
        "seed_control_to_model_noncontrol",
        "seed_noncontrol_to_model_control",
    ):
        counters.setdefault(key, 0)
    total = max(1, int(counters["rows"]))
    seed_total = max(1, int(counters["seed_yaw_control_rows"]))
    observable_thresholds = _dedupe_sorted(list(sweep_yaw_observable_thresholds or ()) or list(DEFAULT_SWEEP_THRESHOLDS))
    confidence_thresholds = _dedupe_sorted(list(sweep_yaw_confidence_thresholds or ()) or [float(yaw_confidence_threshold)])
    ambiguous_thresholds = _dedupe_sorted(list(sweep_yaw_ambiguous_thresholds or ()) or [float(yaw_ambiguous_threshold)])
    threshold_sweep: list[dict[str, Any]] = []
    for obs_thr in observable_thresholds:
        for conf_thr in confidence_thresholds:
            for amb_thr in ambiguous_thresholds:
                tp = fp = fn = tn = 0
                for row in raw_scores:
                    pred_control = bool(
                        float(row["yaw_observable_score"]) >= float(obs_thr)
                        and float(row["yaw_confidence"]) >= float(conf_thr)
                        and float(row["yaw_ambiguous_score"]) < float(amb_thr)
                        and float(row["yaw_step_scale"]) >= float(yaw_step_scale_threshold)
                    )
                    target_control = bool(row["seed_yaw_control"])
                    if pred_control and target_control:
                        tp += 1
                    elif pred_control and not target_control:
                        fp += 1
                    elif (not pred_control) and target_control:
                        fn += 1
                    else:
                        tn += 1
                payload = {
                    "yaw_observable_threshold": float(obs_thr),
                    "yaw_confidence_threshold": float(conf_thr),
                    "yaw_ambiguous_threshold": float(amb_thr),
                    **_confusion_metrics(tp=tp, fp=fp, fn=fn, tn=tn),
                }
                threshold_sweep.append(payload)
    threshold_sweep.sort(key=lambda row: (-float(row["precision"]), -float(row["recall"]), float(row["fp"])))
    summary = {
        "schema_version": "c2c_v2_task_frame_yaw_model_observability_audit_v1",
        "dataset_jsonl": [str(path) for path in dataset_jsonl],
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": dict(metadata or {}),
        "rows": int(counters["rows"]),
        "counters": dict(counters),
        "rates": {
            "seed_yaw_control": float(counters["seed_yaw_control_rows"] / total),
            "model_yaw_control": float(counters["model_yaw_control_rows"] / total),
            "seed_control_to_model_noncontrol": float(counters["seed_control_to_model_noncontrol"] / seed_total),
            "seed_noncontrol_to_model_control": float(counters["seed_noncontrol_to_model_control"] / total),
        },
        "by_yaw_observability_class": {key: dict(value) for key, value in sorted(by_yaw_class.items())},
        "by_source_eval_root_top20": {key: dict(value) for key, value in sorted(by_root.items(), key=lambda item: (-item[1]["rows"], item[0]))[:20]},
        "thresholds": {
            "yaw_observable": float(yaw_observable_threshold),
            "yaw_confidence": float(yaw_confidence_threshold),
            "yaw_ambiguous": float(yaw_ambiguous_threshold),
            "yaw_step_scale": float(yaw_step_scale_threshold),
        },
        "threshold_sweep": threshold_sweep,
        "threshold_sweep_best_precision_top10": threshold_sweep[:10],
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_audit": True,
        "privileged_label_boundary": "offline_seed_labels_only_for_audit_targets",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for row in enriched_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary["output_jsonl"] = str(output_jsonl)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v46 model yaw observability against seed labels.")
    parser.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--image_crop_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_CROP_SIZE)
    parser.add_argument("--image_resize_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_IMAGE_SIZE)
    parser.add_argument("--history_window_size", type=int, default=RUNTIME_XY_SPATIAL_TEMPORAL_HISTORY_WINDOW)
    parser.add_argument("--max_abs_xy_label", type=float, default=0.080)
    parser.add_argument("--max_abs_z_label", type=float, default=0.080)
    parser.add_argument("--max_abs_yaw_label", type=float, default=0.350)
    parser.add_argument("--yaw_observable_threshold", type=float, default=0.5)
    parser.add_argument("--yaw_confidence_threshold", type=float, default=0.45)
    parser.add_argument("--yaw_ambiguous_threshold", type=float, default=0.5)
    parser.add_argument("--yaw_step_scale_threshold", type=float, default=0.05)
    parser.add_argument("--sweep_yaw_observable_threshold", action="append", type=float, default=[])
    parser.add_argument("--sweep_yaw_confidence_threshold", action="append", type=float, default=[])
    parser.add_argument("--sweep_yaw_ambiguous_threshold", action="append", type=float, default=[])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_model_observability(
        list(args.dataset_jsonl),
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        output_jsonl=args.output_jsonl,
        image_crop_size=int(args.image_crop_size),
        image_resize_size=int(args.image_resize_size),
        history_window_size=int(args.history_window_size),
        max_abs_xy_label=float(args.max_abs_xy_label),
        max_abs_z_label=float(args.max_abs_z_label),
        max_abs_yaw_label=float(args.max_abs_yaw_label),
        yaw_observable_threshold=float(args.yaw_observable_threshold),
        yaw_confidence_threshold=float(args.yaw_confidence_threshold),
        yaw_ambiguous_threshold=float(args.yaw_ambiguous_threshold),
        yaw_step_scale_threshold=float(args.yaw_step_scale_threshold),
        sweep_yaw_observable_thresholds=tuple(float(v) for v in args.sweep_yaw_observable_threshold),
        sweep_yaw_confidence_thresholds=tuple(float(v) for v in args.sweep_yaw_confidence_threshold),
        sweep_yaw_ambiguous_thresholds=tuple(float(v) for v in args.sweep_yaw_ambiguous_threshold),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    print(json.dumps({k: summary[k] for k in ("rows", "counters", "rates")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
