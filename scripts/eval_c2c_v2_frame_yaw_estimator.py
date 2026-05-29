#!/usr/bin/env python3
"""Evaluate a C2C v2 frame-yaw estimator on a focused diagnostic slice.

The intended use is to compare a learned frame-yaw estimator against the
legacy image/PCA yaw proxy on a narrow offline-only slice such as:

    visual_observable && near_basin_shell && yaw_entry_feasible

This script never changes runtime policy.  It only reads relabels, optionally
loads a checkpoint, and writes an offline evaluation report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.frame_yaw_estimator import (
    FRAME_YAW_FEATURE_NAMES,
    FrameYawEstimatorNet,
    frame_yaw_feature_vector,
    load_frame_yaw_checkpoint,
    resolve_yaw_observable_threshold,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    return value if isinstance(value, Mapping) else {}


def _visual_class(row: Mapping[str, Any]) -> str:
    obs = _mapping(row, "obs_t")
    return str(obs.get("visual_observability_class", row.get("visual_observability_class", "")))


def _proxy_image_axis_yaw(row: Mapping[str, Any]) -> float:
    proxy = _mapping(row, "proxy_local_geometry_error")
    if "image_axis_yaw" in proxy:
        return _safe_float(proxy.get("image_axis_yaw"), 0.0)
    return _safe_float(proxy.get("dyaw", 0.0), 0.0)


def _proxy_residual_yaw(row: Mapping[str, Any]) -> float:
    proxy = _mapping(row, "proxy_local_geometry_error")
    return _safe_float(proxy.get("dyaw", 0.0), 0.0)


def _true_dyaw(row: Mapping[str, Any]) -> float:
    residual = _mapping(row, "true_basin_error_t")
    return _safe_float(residual.get("dyaw", row.get("privileged_dyaw", float("nan"))), float("nan"))


def _yaw_observable_target(row: Mapping[str, Any]) -> float:
    return 1.0 if _safe_bool(row.get("yaw_control_observable", row.get("yaw_observable", False))) else 0.0


def _yaw_entry_feasible(row: Mapping[str, Any]) -> bool:
    if "yaw_entry_feasible" in row:
        return _safe_bool(row.get("yaw_entry_feasible", False))
    yaw_abs = abs(_true_dyaw(row))
    return bool(np.isfinite(yaw_abs) and yaw_abs <= 0.08 + 1.0e-9)


def _near_basin(row: Mapping[str, Any]) -> bool:
    return _safe_bool(row.get("near_basin_shell", False))


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    stage_name: str,
    skill_type: str,
    visual_only: bool,
    near_basin_only: bool,
    min_frame_observability: float | None,
    require_not_wrist_occluded: bool,
    candidate_jsonl: Path | None,
) -> list[dict[str, Any]]:
    candidate_keys: set[tuple[int, int]] | None = None
    if candidate_jsonl is not None:
        candidate_keys = {
            (int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1))))
            for row in _read_jsonl(candidate_jsonl)
        }
    selected: list[dict[str, Any]] = []
    for row in rows:
        if candidate_keys is not None and (int(row.get("episode_idx", -1)), int(row.get("step_idx", row.get("step", -1)))) not in candidate_keys:
            continue
        if stage_name and str(row.get("stage_name", "")) != stage_name:
            continue
        if skill_type and str(row.get("skill_type", "")) != skill_type:
            continue
        if visual_only and _visual_class(row) != "visual_observable":
            continue
        if near_basin_only and not _near_basin(row):
            continue
        if min_frame_observability is not None:
            obs = _safe_float(
                row.get("yaw_observability_frame_observability", row.get("source_frame_observability", _mapping(row, "obs_t").get("frame_observability", 0.0))),
                0.0,
            )
            if obs < float(min_frame_observability) - 1.0e-12:
                continue
        if require_not_wrist_occluded and _safe_bool(row.get("yaw_observability_wrist_occluded", row.get("wrist_is_occluded", False))):
            continue
        selected.append(row)
    return selected


def _corr(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) <= 1.0e-12 or float(np.std(bb)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _sign_match(pred: Iterable[float], target: Iterable[float]) -> float:
    pairs = [
        (p, t)
        for p, t in zip(pred, target)
        if np.isfinite(p) and np.isfinite(t) and abs(float(p)) > 1.0e-6 and abs(float(t)) > 1.0e-6
    ]
    if not pairs:
        return 0.0
    return float(np.mean([np.sign(p) == np.sign(t) for p, t in pairs]))


def _metrics(
    pred_dyaw: np.ndarray,
    target_dyaw: np.ndarray,
    pred_obs_prob: np.ndarray | None,
    target_obs: np.ndarray,
    *,
    yaw_observable_threshold: float = 0.5,
) -> dict[str, Any]:
    mask = np.isfinite(pred_dyaw) & np.isfinite(target_dyaw)
    abs_err = np.abs(pred_dyaw[mask] - target_dyaw[mask]) if np.any(mask) else np.asarray([], dtype=np.float32)
    out = {
        "rows": int(target_dyaw.shape[0]),
        "dyaw_mae": float(np.mean(abs_err)) if abs_err.size else 0.0,
        "dyaw_p95_abs_error": float(np.percentile(abs_err, 95)) if abs_err.size else 0.0,
        "dyaw_corr": _corr(pred_dyaw, target_dyaw),
        "dyaw_sign_match_rate": _sign_match(pred_dyaw, target_dyaw),
        "target_obs_rate": float(np.mean(target_obs > 0.5)) if target_obs.size else 0.0,
        "yaw_observable_threshold": float(yaw_observable_threshold),
    }
    if pred_obs_prob is not None and pred_obs_prob.size:
        pred_obs = pred_obs_prob >= float(yaw_observable_threshold)
        target_obs_bool = target_obs >= 0.5
        tp = int(np.count_nonzero(pred_obs & target_obs_bool))
        fp = int(np.count_nonzero(pred_obs & ~target_obs_bool))
        fn = int(np.count_nonzero(~pred_obs & target_obs_bool))
        tn = int(np.count_nonzero(~pred_obs & ~target_obs_bool))
        recall = float(tp / max(tp + fn, 1))
        precision = float(tp / max(tp + fp, 1))
        specificity = float(tn / max(tn + fp, 1))
        out.update(
            {
                "yaw_observable_accuracy": float(np.mean(pred_obs == target_obs_bool)) if target_obs_bool.size else 0.0,
                "yaw_observable_precision": precision,
                "yaw_observable_recall": recall,
                "yaw_observable_specificity": specificity,
                "yaw_observable_balanced_accuracy": float(0.5 * (recall + specificity)),
                "predicted_yaw_observable_rate": float(np.mean(pred_obs)) if pred_obs.size else 0.0,
            }
        )
    return out


def _group_metrics(
    rows: list[dict[str, Any]],
    *,
    pred_dyaw: np.ndarray,
    pred_obs_prob: np.ndarray | None,
    yaw_observable_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    by_episode: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_episode[int(row.get("episode_idx", -1))].append(idx)
    out: list[dict[str, Any]] = []
    for ep, idxs in sorted(by_episode.items()):
        target = np.asarray([_true_dyaw(rows[i]) for i in idxs], dtype=np.float32)
        obs = np.asarray([_yaw_observable_target(rows[i]) for i in idxs], dtype=np.float32)
        pred_obs = pred_obs_prob[idxs] if pred_obs_prob is not None else None
        out.append(
            {
                "episode_idx": int(ep),
                "rows": int(len(idxs)),
                "visual_rows": int(sum(_visual_class(rows[i]) == "visual_observable" for i in idxs)),
                "near_basin_rows": int(sum(_near_basin(rows[i]) for i in idxs)),
                "yaw_entry_feasible_rows": int(sum(_yaw_entry_feasible(rows[i]) for i in idxs)),
                "proxy_image_axis_yaw_mae": float(np.mean(np.abs(np.asarray([_proxy_image_axis_yaw(rows[i]) for i in idxs]) - target))) if idxs else 0.0,
                "proxy_image_axis_yaw_corr": _corr([_proxy_image_axis_yaw(rows[i]) for i in idxs], target),
                "estimator_dyaw_mae": float(np.mean(np.abs(pred_dyaw[idxs] - target))) if idxs else 0.0,
                "estimator_dyaw_corr": _corr(pred_dyaw[idxs], target),
                "estimator_dyaw_sign_match": _sign_match(pred_dyaw[idxs], target),
                "yaw_observable_rate": float(np.mean(obs > 0.5)) if obs.size else 0.0,
                "yaw_observable_accuracy": float(np.mean((pred_obs >= float(yaw_observable_threshold)) == (obs >= 0.5))) if pred_obs is not None else float("nan"),
            }
        )
    return out


def evaluate(
    rows: list[dict[str, Any]],
    *,
    checkpoint: Path | None,
    yaw_observable_threshold: float | None = None,
    stage_name: str = "RING_GRASP_ALIGN",
    skill_type: str = "precision_grasp",
    visual_only: bool = True,
    near_basin_only: bool = True,
    min_frame_observability: float | None = 0.02,
    require_not_wrist_occluded: bool = True,
    candidate_jsonl: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = _select_rows(
        rows,
        stage_name=stage_name,
        skill_type=skill_type,
        visual_only=visual_only,
        near_basin_only=near_basin_only,
        min_frame_observability=min_frame_observability,
        require_not_wrist_occluded=require_not_wrist_occluded,
        candidate_jsonl=candidate_jsonl,
    )
    if not selected:
        report = {
            "schema_version": "frame_yaw_estimator_focused_eval_v1",
            "selection": {
                "stage_name": stage_name,
                "skill_type": skill_type,
                "visual_only": bool(visual_only),
                "near_basin_only": bool(near_basin_only),
                "min_frame_observability": None if min_frame_observability is None else float(min_frame_observability),
                "require_not_wrist_occluded": bool(require_not_wrist_occluded),
                "candidate_jsonl": None if candidate_jsonl is None else str(candidate_jsonl.resolve()),
                "predicate": "visual_observable && near_basin_shell && yaw_entry_feasible",
            },
            "overall": {"rows": 0},
            "rows": [],
        }
        return selected, report

    features = np.stack([frame_yaw_feature_vector(r) for r in selected]).astype(np.float32)
    target = np.asarray([_true_dyaw(r) for r in selected], dtype=np.float32)
    target_obs = np.asarray([_yaw_observable_target(r) for r in selected], dtype=np.float32)
    proxy_image_axis = np.asarray([_proxy_image_axis_yaw(r) for r in selected], dtype=np.float32)
    proxy_residual = np.asarray([_proxy_residual_yaw(r) for r in selected], dtype=np.float32)

    model_metrics: dict[str, Any] | None = None
    model_metrics_legacy: dict[str, Any] | None = None
    pred_dyaw = np.full_like(target, np.nan, dtype=np.float32)
    pred_obs_prob: np.ndarray | None = None
    model_meta: dict[str, Any] = {}
    resolved_threshold = 0.5
    threshold_source = "default_0.5"
    if checkpoint is not None:
        model, model_meta = load_frame_yaw_checkpoint(checkpoint, map_location="cpu")
        if int(model.feature_dim) != int(features.shape[1]):
            raise RuntimeError(f"feature_dim mismatch: checkpoint={model.feature_dim} dataset={features.shape[1]}")
        resolved_threshold = float(yaw_observable_threshold) if yaw_observable_threshold is not None else resolve_yaw_observable_threshold(model_meta, default=0.5)
        threshold_source = "cli_override" if yaw_observable_threshold is not None else "checkpoint_metadata"
        with torch.no_grad():
            out = model(torch.as_tensor(features, dtype=torch.float32))
        pred_dyaw = out["dyaw"].detach().cpu().numpy().astype(np.float32)
        pred_obs_prob = out["yaw_observable_probability"].detach().cpu().numpy().astype(np.float32)
        model_metrics = _metrics(pred_dyaw, target, pred_obs_prob, target_obs, yaw_observable_threshold=resolved_threshold)
        model_metrics_legacy = _metrics(pred_dyaw, target, pred_obs_prob, target_obs, yaw_observable_threshold=0.5)

    proxy_metrics = _metrics(proxy_image_axis, target, None, target_obs)
    proxy_residual_metrics = _metrics(proxy_residual, target, None, target_obs)

    rows_out: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        rows_out.append(
            {
                "episode_idx": int(row.get("episode_idx", -1)),
                "step_idx": int(row.get("step_idx", row.get("step", -1))),
                "stage_name": str(row.get("stage_name", "")),
                "skill_type": str(row.get("skill_type", "")),
                "visual_observability_class": _visual_class(row),
                "yaw_observable_target": float(target_obs[idx]),
                "true_dyaw": float(target[idx]),
                "proxy_image_axis_yaw": float(proxy_image_axis[idx]),
                "proxy_residual_yaw": float(proxy_residual[idx]),
                "estimator_dyaw": float(pred_dyaw[idx]) if np.isfinite(pred_dyaw[idx]) else float("nan"),
                "estimator_yaw_observable_probability": float(pred_obs_prob[idx]) if pred_obs_prob is not None else float("nan"),
                "estimator_yaw_observable": bool(pred_obs_prob[idx] >= resolved_threshold) if pred_obs_prob is not None else False,
                "estimator_yaw_observable_threshold": float(resolved_threshold),
                "yaw_entry_feasible": bool(_yaw_entry_feasible(row)),
                "near_basin_shell": bool(_near_basin(row)),
                "frame_observability": float(_safe_float(row.get("yaw_observability_frame_observability", row.get("source_frame_observability", 0.0)), 0.0)),
                "primary_blocker": str(row.get("yaw_observability_primary_blocker", "")),
                "diagnosis_label": str(row.get("yaw_observability_primary_blocker", "")),
            }
        )

    report = {
        "schema_version": "frame_yaw_estimator_focused_eval_v1",
        "selection": {
            "stage_name": stage_name,
            "skill_type": skill_type,
            "visual_only": bool(visual_only),
            "near_basin_only": bool(near_basin_only),
            "min_frame_observability": None if min_frame_observability is None else float(min_frame_observability),
            "require_not_wrist_occluded": bool(require_not_wrist_occluded),
            "candidate_jsonl": None if candidate_jsonl is None else str(candidate_jsonl.resolve()),
            "predicate": "visual_observable && near_basin_shell && yaw_entry_feasible",
            "pca_yaw_is_diagnostic_only": True,
            "yaw_observable_threshold_working_point": float(resolved_threshold),
            "yaw_observable_threshold_working_point_source": threshold_source,
        },
        "overall": {
            "rows": int(len(selected)),
            "episodes": int(len({int(r.get("episode_idx", -1)) for r in selected})),
            "visual_rows": int(sum(_visual_class(r) == "visual_observable" for r in selected)),
            "near_basin_rows": int(sum(_near_basin(r) for r in selected)),
            "yaw_entry_feasible_rows": int(sum(_yaw_entry_feasible(r) for r in selected)),
            "proxy_image_axis_yaw_mae": proxy_metrics["dyaw_mae"],
            "proxy_image_axis_yaw_sign_match_rate": proxy_metrics["dyaw_sign_match_rate"],
            "proxy_image_axis_yaw_corr": proxy_metrics["dyaw_corr"],
            "proxy_residual_yaw_mae": proxy_residual_metrics["dyaw_mae"],
            "proxy_residual_yaw_sign_match_rate": proxy_residual_metrics["dyaw_sign_match_rate"],
            "proxy_residual_yaw_corr": proxy_residual_metrics["dyaw_corr"],
            "target_yaw_observable_rate": proxy_metrics["target_obs_rate"],
            "yaw_observable_threshold_working_point": float(resolved_threshold),
            "yaw_observable_threshold_working_point_source": threshold_source,
            "candidate_episode_counts": dict(Counter(int(r.get("episode_idx", -1)) for r in selected)),
        },
        "proxy": proxy_metrics,
        "proxy_residual": proxy_residual_metrics,
        "model": model_metrics or {"rows": 0, "checkpoint": None},
        "model_reference_0_5": model_metrics_legacy or {"rows": 0, "checkpoint": None},
        "by_episode": _group_metrics(selected, pred_dyaw=pred_dyaw, pred_obs_prob=pred_obs_prob, yaw_observable_threshold=resolved_threshold),
        "rows": rows_out,
        "model_metadata": model_meta,
    }
    return selected, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a C2C v2 frame-yaw estimator on a focused diagnostic slice.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument("--candidate_jsonl", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--yaw_observable_threshold", type=float, default=None)
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/frame_yaw_focused_eval"))
    ap.add_argument("--stage_name", type=str, default="RING_GRASP_ALIGN")
    ap.add_argument("--skill_type", type=str, default="precision_grasp")
    ap.add_argument("--visual_only", action="store_true", default=True)
    ap.add_argument("--near_basin_only", action="store_true", default=True)
    ap.add_argument("--min_frame_observability", type=float, default=0.02)
    ap.add_argument("--require_not_wrist_occluded", action="store_true", default=True)
    args = ap.parse_args()

    rows = _read_jsonl(args.relabel_jsonl)
    selected, report = evaluate(
        rows,
        checkpoint=args.checkpoint,
        yaw_observable_threshold=args.yaw_observable_threshold,
        stage_name=str(args.stage_name),
        skill_type=str(args.skill_type),
        visual_only=bool(args.visual_only),
        near_basin_only=bool(args.near_basin_only),
        min_frame_observability=args.min_frame_observability,
        require_not_wrist_occluded=bool(args.require_not_wrist_occluded),
        candidate_jsonl=args.candidate_jsonl,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "frame_yaw_focused_eval_rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as handle:
        for row in report["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report["selected_rows_jsonl"] = str(rows_path)
    report["source_jsonl"] = str(args.relabel_jsonl.resolve())
    if args.candidate_jsonl is not None:
        report["candidate_jsonl"] = str(args.candidate_jsonl.resolve())
    if args.checkpoint is not None:
        report["checkpoint"] = str(args.checkpoint.resolve())
    out_json = output_dir / "frame_yaw_focused_eval.json"
    out_md = output_dir / "frame_yaw_focused_eval.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Frame Yaw Focused Eval",
        "",
        f"- source: `{args.relabel_jsonl}`",
        f"- rows: `{report['overall']['rows']}`",
        f"- episodes: `{report['overall']['episodes']}`",
        f"- visual_rows: `{report['overall']['visual_rows']}`",
        f"- near_basin_rows: `{report['overall']['near_basin_rows']}`",
        f"- yaw_entry_feasible_rows: `{report['overall']['yaw_entry_feasible_rows']}`",
        f"- proxy_image_axis_yaw_mae: `{report['overall']['proxy_image_axis_yaw_mae']:.6f}`",
        f"- proxy_image_axis_yaw_sign_match_rate: `{report['overall']['proxy_image_axis_yaw_sign_match_rate']:.3f}`",
        f"- proxy_image_axis_yaw_corr: `{report['overall']['proxy_image_axis_yaw_corr']:.3f}`",
        f"- proxy_residual_yaw_mae: `{report['overall']['proxy_residual_yaw_mae']:.6f}`",
        f"- proxy_residual_yaw_sign_match_rate: `{report['overall']['proxy_residual_yaw_sign_match_rate']:.3f}`",
        f"- proxy_residual_yaw_corr: `{report['overall']['proxy_residual_yaw_corr']:.3f}`",
        f"- target_yaw_observable_rate: `{report['overall']['target_yaw_observable_rate']:.3f}`",
        f"- yaw_observable_threshold_working_point: `{report['overall']['yaw_observable_threshold_working_point']:.3f}`",
        f"- yaw_observable_threshold_working_point_source: `{report['overall']['yaw_observable_threshold_working_point_source']}`",
        f"- pca_yaw_is_diagnostic_only: `true`",
    ]
    if report["model"] and int(report["model"].get("rows", 0)) > 0:
        lines.extend(
            [
                "",
                "## Model",
                f"- yaw_observable_threshold: `{report['model']['yaw_observable_threshold']:.3f}`",
                f"- dyaw_mae: `{report['model']['dyaw_mae']:.6f}`",
                f"- dyaw_sign_match_rate: `{report['model']['dyaw_sign_match_rate']:.3f}`",
                f"- dyaw_corr: `{report['model']['dyaw_corr']:.3f}`",
                f"- yaw_observable_accuracy: `{report['model']['yaw_observable_accuracy']:.3f}`",
                f"- yaw_observable_balanced_accuracy: `{report['model']['yaw_observable_balanced_accuracy']:.3f}`",
            ]
        )
    if report["model_reference_0_5"] and int(report["model_reference_0_5"].get("rows", 0)) > 0:
        lines.extend(
            [
                "",
                "## Model At 0.5",
                f"- yaw_observable_threshold: `{report['model_reference_0_5']['yaw_observable_threshold']:.3f}`",
                f"- yaw_observable_accuracy: `{report['model_reference_0_5']['yaw_observable_accuracy']:.3f}`",
                f"- yaw_observable_balanced_accuracy: `{report['model_reference_0_5']['yaw_observable_balanced_accuracy']:.3f}`",
            ]
        )
    lines.extend(["", "## By Episode"])
    for item in report["by_episode"]:
        lines.append(
            f"- ep{int(item['episode_idx']):03d}: rows={item['rows']}, near={item['near_basin_rows']}, "
            f"proxy_mae={item['proxy_image_axis_yaw_mae']:.3f}, est_mae={item['estimator_dyaw_mae']:.3f}, "
            f"est_sign={item['estimator_dyaw_sign_match']:.3f}"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    print(out_json)
    print(out_md)
    print(rows_path)


if __name__ == "__main__":
    main()
