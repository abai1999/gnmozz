#!/usr/bin/env python3
"""Leave-one-source-root validation for the v46 command-candidate ranker.

This utility repeatedly trains the candidate ranker while forcing exactly one
source_eval_root into validation.  It is intended for small/medium executed
command-sweep pools where worst-root behavior matters more than a random split.
Runtime inputs remain non-privileged; offline pre/post residuals are labels only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import source_eval_root_key  # noqa: E402
from scripts.train_c2c_v2_task_frame_v46_alignment import _load_rows, _normalize_row_metadata  # noqa: E402
from scripts.train_c2c_v2_task_frame_v46_candidate_ranker import train  # noqa: E402


def _safe_slug(value: str, *, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_")
    if not slug:
        slug = "root"
    return slug[-max_len:]


def _source_roots_from_dataset(dataset_jsonl: list[Path]) -> list[str]:
    rows = [_normalize_row_metadata(row) for row in _load_rows(dataset_jsonl)]
    return sorted({source_eval_root_key(row) for row in rows})


def _numeric_values(reports: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for report in reports:
        current: Any = report
        for part in path:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if isinstance(current, (int, float)) and np.isfinite(float(current)):
            values.append(float(current))
    return values


def _metric_stats(reports: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    values = _numeric_values(reports, ("val_metrics", metric))
    if not values:
        return {"mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def _flag_rate(group_details: dict[str, Any], *, selected_index: int, zero_index: int, mode: str) -> float | None:
    if not group_details:
        return None
    hits = 0
    total = 0
    for detail in group_details.values():
        selected = detail.get("selected_flags")
        zero = detail.get("zero_flags")
        if not isinstance(selected, (list, tuple)) or not isinstance(zero, (list, tuple)):
            continue
        if len(selected) <= selected_index or len(zero) <= zero_index:
            continue
        selected_flag = bool(selected[selected_index])
        zero_flag = bool(zero[zero_index])
        if mode == "beats":
            hit = selected_flag and not zero_flag
        elif mode == "worse":
            hit = (not selected_flag) and zero_flag
        else:
            raise ValueError(f"unknown flag comparison mode: {mode}")
        hits += int(hit)
        total += 1
    if total <= 0:
        return None
    return float(hits / total)


def _axis_zero_comparison(val_metrics: dict[str, Any], axis: str, flag_index: int) -> dict[str, float | None]:
    top1 = float(val_metrics.get(f"top1_{axis}_contraction", 0.0) or 0.0)
    zero = float(val_metrics.get(f"zero_{axis}_contraction", 0.0) or 0.0)
    details = dict(val_metrics.get("group_details", {}) or {})
    return {
        "top1_minus_zero": float(top1 - zero),
        "top1_beats_zero_rate": _flag_rate(details, selected_index=flag_index, zero_index=flag_index, mode="beats"),
        "top1_worse_than_zero_rate": _flag_rate(details, selected_index=flag_index, zero_index=flag_index, mode="worse"),
    }


def _worst_fold(fold_summaries: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    if not fold_summaries:
        return None
    return min(fold_summaries, key=lambda item: float(item.get(metric, 0.0) or 0.0))


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "top1_best_score_match",
        "top1_xy_contraction",
        "top1_z_contraction",
        "top1_yaw_contraction",
        "top1_combined_contraction",
        "zero_xy_contraction",
        "zero_z_contraction",
        "zero_yaw_contraction",
        "zero_combined_contraction",
    )
    fold_summaries = []
    for report in reports:
        val_roots = list(report.get("val_source_eval_roots", []))
        val_metrics = dict(report.get("val_metrics", {}))
        xy_zero = _axis_zero_comparison(val_metrics, "xy", 0)
        z_zero = _axis_zero_comparison(val_metrics, "z", 1)
        yaw_zero = _axis_zero_comparison(val_metrics, "yaw", 2)
        combined_zero = _axis_zero_comparison(val_metrics, "combined", 3)
        fold_summaries.append(
            {
                "val_source_eval_roots": val_roots,
                "val_groups": int(val_metrics.get("groups", 0) or 0),
                "val_rows": int(val_metrics.get("rows", report.get("val_rows", 0)) or 0),
                "top1_best_score_match": float(val_metrics.get("top1_best_score_match", 0.0) or 0.0),
                "top1_xy_contraction": float(val_metrics.get("top1_xy_contraction", 0.0) or 0.0),
                "top1_z_contraction": float(val_metrics.get("top1_z_contraction", 0.0) or 0.0),
                "top1_yaw_contraction": float(val_metrics.get("top1_yaw_contraction", 0.0) or 0.0),
                "top1_combined_contraction": float(val_metrics.get("top1_combined_contraction", 0.0) or 0.0),
                "top1_minus_zero_xy_contraction": xy_zero["top1_minus_zero"],
                "top1_minus_zero_z_contraction": z_zero["top1_minus_zero"],
                "top1_minus_zero_yaw_contraction": yaw_zero["top1_minus_zero"],
                "top1_minus_zero_combined_contraction": combined_zero["top1_minus_zero"],
                "top1_beats_zero_xy_rate": xy_zero["top1_beats_zero_rate"],
                "top1_beats_zero_z_rate": z_zero["top1_beats_zero_rate"],
                "top1_beats_zero_yaw_rate": yaw_zero["top1_beats_zero_rate"],
                "top1_beats_zero_combined_rate": combined_zero["top1_beats_zero_rate"],
                "top1_worse_than_zero_xy_rate": xy_zero["top1_worse_than_zero_rate"],
                "top1_worse_than_zero_z_rate": z_zero["top1_worse_than_zero_rate"],
                "top1_worse_than_zero_yaw_rate": yaw_zero["top1_worse_than_zero_rate"],
                "top1_worse_than_zero_combined_rate": combined_zero["top1_worse_than_zero_rate"],
                "selected_candidate_counts": dict(val_metrics.get("selected_candidate_counts", {})),
                "oracle_candidate_counts": dict(val_metrics.get("oracle_candidate_counts", {})),
            }
        )
    summary = {
        "schema_version": "c2c_v2_task_frame_v46_candidate_ranker_loo_summary_v1",
        "model": "v46_unified_task_frame_alignment_candidate_ranker",
        "folds": int(len(reports)),
        "metric_stats": {metric: _metric_stats(reports, metric) for metric in metrics},
        "fold_summaries": fold_summaries,
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": True,
        "privileged_label_boundary": "offline_pre_post_transition_labels_only",
        "upgrade_gate": "pending_large_random_holdout_and_closed_loop_insert_success",
    }
    summary["worst_folds"] = {
        "top1_best_score_match": _worst_fold(fold_summaries, "top1_best_score_match"),
        "top1_xy_contraction": _worst_fold(fold_summaries, "top1_xy_contraction"),
        "top1_z_contraction": _worst_fold(fold_summaries, "top1_z_contraction"),
        "top1_yaw_contraction": _worst_fold(fold_summaries, "top1_yaw_contraction"),
        "top1_combined_contraction": _worst_fold(fold_summaries, "top1_combined_contraction"),
        "top1_minus_zero_combined_contraction": _worst_fold(fold_summaries, "top1_minus_zero_combined_contraction"),
        "top1_minus_zero_yaw_contraction": _worst_fold(fold_summaries, "top1_minus_zero_yaw_contraction"),
    }
    return summary


def run_leave_one_source_root(
    dataset_jsonl: list[Path],
    *,
    output_dir: Path,
    output_json: Path,
    max_roots: int = 0,
    epochs: int = 30,
    lr: float = 5.0e-4,
    seed: int = 7,
    rank_score_mode: str = "yaw_collateral",
    zero_guard_margin: float = 0.050,
    zero_guard_weight: float = 1.000,
    support_score_penalty: float = 0.250,
    outcome_loss_weight: float = 1.000,
    pairwise_margin: float = 0.050,
    pairwise_weight: float = 0.000,
    pairwise_zero_margin: float | None = None,
    command_feature_mode: str = "raw6",
    device: str = "cpu",
    image_hidden_dim: int = 128,
    fusion_hidden_dim: int = 128,
) -> dict[str, Any]:
    roots = _source_roots_from_dataset(dataset_jsonl)
    if max_roots > 0:
        roots = roots[: int(max_roots)]
    if len(roots) < 2:
        raise RuntimeError("leave-one-source-root validation needs at least two source roots")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for fold_idx, root in enumerate(roots):
        slug = f"{fold_idx:03d}_{_safe_slug(root)}"
        report = train(
            dataset_jsonl,
            output_checkpoint=output_dir / "checkpoints" / f"{slug}.pt",
            output_json=output_dir / "reports" / f"{slug}.json",
            val_fraction=0.0,
            test_fraction=0.0,
            split_mode="root",
            epochs=int(epochs),
            lr=float(lr),
            seed=int(seed) + int(fold_idx),
            image_hidden_dim=int(image_hidden_dim),
            fusion_hidden_dim=int(fusion_hidden_dim),
            rank_score_mode=str(rank_score_mode),
            zero_guard_margin=float(zero_guard_margin),
            zero_guard_weight=float(zero_guard_weight),
            support_score_penalty=float(support_score_penalty),
            outcome_loss_weight=float(outcome_loss_weight),
            pairwise_margin=float(pairwise_margin),
            pairwise_weight=float(pairwise_weight),
            pairwise_zero_margin=pairwise_zero_margin,
            command_feature_mode=str(command_feature_mode),
            val_source_eval_roots={root},
            device=str(device),
        )
        reports.append(report)

    summary = _aggregate_reports(reports)
    summary["dataset_jsonl"] = [str(path) for path in dataset_jsonl]
    summary["source_eval_roots"] = roots
    summary["output_dir"] = str(output_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--max_roots", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rank_score_mode", type=str, default="yaw_collateral", choices=("residual", "axis_balanced", "yaw_collateral", "outcome_utility"))
    parser.add_argument("--zero_guard_margin", type=float, default=0.050)
    parser.add_argument("--zero_guard_weight", type=float, default=1.000)
    parser.add_argument("--support_score_penalty", type=float, default=0.250)
    parser.add_argument("--outcome_loss_weight", type=float, default=1.000)
    parser.add_argument("--pairwise_margin", type=float, default=0.050)
    parser.add_argument("--pairwise_weight", type=float, default=0.000)
    parser.add_argument("--pairwise_zero_margin", type=float, default=-1.0)
    parser.add_argument("--command_feature_mode", type=str, default="raw6", choices=("raw6", "typed16"))
    parser.add_argument("--image_hidden_dim", type=int, default=128)
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_leave_one_source_root(
        list(args.dataset_jsonl),
        output_dir=args.output_dir,
        output_json=args.output_json,
        max_roots=int(args.max_roots),
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
        rank_score_mode=str(args.rank_score_mode),
        zero_guard_margin=float(args.zero_guard_margin),
        zero_guard_weight=float(args.zero_guard_weight),
        support_score_penalty=float(args.support_score_penalty),
        outcome_loss_weight=float(args.outcome_loss_weight),
        pairwise_margin=float(args.pairwise_margin),
        pairwise_weight=float(args.pairwise_weight),
        pairwise_zero_margin=None if float(args.pairwise_zero_margin) < 0.0 else float(args.pairwise_zero_margin),
        command_feature_mode=str(args.command_feature_mode),
        image_hidden_dim=int(args.image_hidden_dim),
        fusion_hidden_dim=int(args.fusion_hidden_dim),
        device=str(args.device),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
