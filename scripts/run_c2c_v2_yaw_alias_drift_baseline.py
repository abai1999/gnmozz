#!/usr/bin/env python3
"""Train and evaluate a tiny numpy baseline for yaw alias correction.

This script is intentionally small and torch-free:

* train on calibration-positive stable alias slices only,
* hold out frame-drift hard cases such as ep6,
* compare a learned ridge regressor against the symmetry-aware alias baseline.

The point is not to build the final model.  The point is to check whether a
very small estimator can learn the stable alias correction while still failing
cleanly on jump-heavy frame-drift cases.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _wrap_yaw_to_symmetry(yaw: float, period: float = float(np.pi / 2.0)) -> float:
    if not (np.isfinite(yaw) and np.isfinite(period) and period > 0.0):
        return float("nan")
    half = 0.5 * float(period)
    return float(((float(yaw) + half) % float(period)) - half)


def _symmetry_aware_yaw(raw_yaw: float) -> float:
    if not np.isfinite(raw_yaw):
        return float("nan")
    return float(-_wrap_yaw_to_symmetry(raw_yaw))


def _step_diff(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(((float(a) - float(b) + math.pi) % (2.0 * math.pi)) - math.pi)


def _proxy(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row, "proxy_local_geometry_error")


def _planner_local_delta(row: Mapping[str, Any]) -> np.ndarray:
    planner = _mapping(row, "planner_prior")
    local = planner.get("local_delta_6d", row.get("planner_local_delta_6d", [0.0] * 6))
    try:
        arr = np.asarray(local, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.float32)
    arr = np.pad(arr, (0, max(0, 6 - arr.size)))[:6]
    arr[~np.isfinite(arr)] = 0.0
    return arr.astype(np.float32)


def _row_features(row: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    proxy = _proxy(row)
    raw_proxy_yaw = _safe_float(
        proxy.get(
            "image_axis_yaw",
            row.get("proxy_yaw", proxy.get("dyaw", float("nan"))),
        ),
        float("nan"),
    )
    symm_proxy_yaw = _safe_float(row.get("best_symmetry_alias_yaw", float("nan")), float("nan"))
    if not np.isfinite(symm_proxy_yaw):
        symm_proxy_yaw = _symmetry_aware_yaw(raw_proxy_yaw)
    proxy_conf = _safe_float(proxy.get("confidence", row.get("frame_confidence", 0.0)), 0.0)
    proxy_obs = _safe_float(proxy.get("observability", row.get("frame_observability", 0.0)), 0.0)
    proxy_fit_residual = _safe_float(proxy.get("fit_residual", row.get("best_symmetry_alias_abs_error", 0.0)), 0.0)
    proxy_inlier_ratio = _safe_float(proxy.get("inlier_ratio", 1.0 - proxy_fit_residual), 0.0)
    frame_confidence = _safe_float(row.get("yaw_observability_frame_confidence", row.get("source_frame_confidence", 0.0)), 0.0)
    frame_observability = _safe_float(row.get("yaw_observability_frame_observability", row.get("source_frame_observability", 0.0)), 0.0)
    frame_axis_strength = _safe_float(row.get("yaw_observability_frame_axis_strength", row.get("source_frame_axis_strength", 0.0)), 0.0)
    wide_ring_visible = 1.0 if _safe_bool(row.get("yaw_observability_wide_ring_visible", row.get("wide_ring_visible", False))) else 0.0
    wrist_occluded = 1.0 if _safe_bool(row.get("yaw_observability_wrist_occluded", row.get("wrist_is_occluded", False))) else 0.0
    visual_observable = 1.0 if str(row.get("visual_observability_class", "")) == "visual_observable" else 0.0
    planner_local = _planner_local_delta(row)
    xy_error = _safe_float(row.get("xy_error", float("nan")), float("nan"))
    if not np.isfinite(xy_error):
        true = _mapping(row, "true_basin_error_t")
        dx = _safe_float(true.get("dx", row.get("privileged_dx", float("nan"))), float("nan"))
        dy = _safe_float(true.get("dy", row.get("privileged_dy", float("nan"))), float("nan"))
        xy_error = float(np.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else 0.0
    values = np.asarray(
        [
            1.0,
            raw_proxy_yaw,
            symm_proxy_yaw,
            proxy_conf,
            proxy_obs,
            proxy_fit_residual,
            proxy_inlier_ratio,
            frame_confidence,
            frame_observability,
            frame_axis_strength,
            wide_ring_visible,
            wrist_occluded,
            visual_observable,
            float(planner_local[0]),
            float(planner_local[1]),
            float(planner_local[2]),
            float(planner_local[5]),
            xy_error,
        ],
        dtype=np.float64,
    )
    meta = {
        "raw_proxy_yaw": raw_proxy_yaw,
        "symmetry_aware_proxy_yaw": symm_proxy_yaw,
        "proxy_confidence": proxy_conf,
        "proxy_observability": proxy_obs,
        "frame_confidence": frame_confidence,
        "frame_observability": frame_observability,
        "frame_axis_strength": frame_axis_strength,
    }
    return values, meta


def _select_rows(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    ep = int(report.get("episode_idx", -1))
    bucket = str(report.get("failure_bucket", ""))
    blocker = str(report.get("primary_blocker", ""))
    stage = str(report.get("stage_name", "RING_GRASP_ALIGN"))
    skill = str(report.get("skill_type", "precision_grasp"))
    step_idxs = {int(v) for v in report.get("selected_step_idxs", []) if isinstance(v, (int, float, np.integer, np.floating))}
    selected = [
        row
        for row in rows
        if int(row.get("episode_idx", -1)) == ep
        and str(row.get("failure_bucket", "")) == bucket
        and str(row.get("yaw_observability_primary_blocker", "")) == blocker
        and str(row.get("stage_name", "")) == stage
        and (
            str(row.get("skill_type", row.get("skill_name", ""))) == skill
            or str(row.get("skill_type", row.get("skill_name", ""))).startswith(skill)
            or skill.startswith(str(row.get("skill_type", row.get("skill_name", ""))))
        )
        and (not step_idxs or int(row.get("step_idx", row.get("step", -1))) in step_idxs)
    ]
    selected.sort(key=lambda r: int(r.get("step_idx", r.get("step", -1))))
    return selected


def _stack(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    symm_targets: list[float] = []
    meta: list[dict[str, float]] = []
    for row in rows:
        feats, info = _row_features(row)
        true = _mapping(row, "true_basin_error_t")
        dyaw = _safe_float(
            true.get(
                "dyaw",
                row.get("privileged_dyaw", row.get("privileged_yaw", float("nan"))),
            ),
            float("nan"),
        )
        if not np.isfinite(dyaw):
            continue
        features.append(feats)
        targets.append(float(dyaw))
        symm_targets.append(float(_symmetry_aware_yaw(info["raw_proxy_yaw"])))
        meta.append(info)
    if features:
        return np.stack(features).astype(np.float64), np.asarray(targets, dtype=np.float64), np.asarray(symm_targets, dtype=np.float64), meta
    return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), meta


def _standardize(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    return (x_train - mu) / sigma, (x_other - mu) / sigma, np.stack([mu, sigma], axis=0)


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1.0e-2) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    eye = np.eye(x_aug.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    w = np.linalg.solve(x_aug.T @ x_aug + float(ridge) * eye, x_aug.T @ y)
    return w


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    if x.size == 0 or w.size == 0:
        return np.zeros((x.shape[0],), dtype=np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return x_aug @ w


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    mask = np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def _corr(pred: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(target)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    p = pred[mask]
    t = target[mask]
    if float(np.std(p)) < 1.0e-12 or float(np.std(t)) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def _sign_match(pred: np.ndarray, target: np.ndarray) -> float:
    pairs = [
        (float(p), float(t))
        for p, t in zip(pred, target)
        if np.isfinite(p) and np.isfinite(t) and abs(float(p)) > 1.0e-6 and abs(float(t)) > 1.0e-6
    ]
    if not pairs:
        return 0.0
    return float(np.mean([np.sign(p) == np.sign(t) for p, t in pairs]))


def _jump_points(values: np.ndarray, threshold: float = 0.40) -> list[dict[str, float]]:
    jumps: list[dict[str, float]] = []
    for i in range(1, int(values.shape[0])):
        prev = float(values[i - 1])
        cur = float(values[i])
        if not (np.isfinite(prev) and np.isfinite(cur)):
            continue
        delta = abs(_step_diff(cur, prev))
        if delta >= float(threshold):
            jumps.append({"prev_idx": float(i - 1), "idx": float(i), "delta": float(delta)})
    return jumps


def run_baseline(
    *,
    relabel_jsonl: Path,
    train_reports: list[Path],
    holdout_reports: list[Path],
    output_dir: Path,
    ridge: float = 1.0e-2,
) -> dict[str, Any]:
    rows = _read_jsonl(relabel_jsonl)
    train_selected: list[dict[str, Any]] = []
    holdout_selected: list[dict[str, Any]] = []
    train_slices: list[dict[str, Any]] = []
    holdout_slices: list[dict[str, Any]] = []
    for report_path in train_reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = _select_rows(rows, report)
        train_selected.extend(selected)
        train_slices.append(
            {
                "report_path": str(report_path.resolve()),
                "episode_idx": int(report.get("episode_idx", -1)),
                "failure_bucket": str(report.get("failure_bucket", "")),
                "primary_blocker": str(report.get("primary_blocker", "")),
                "rows": int(len(selected)),
            }
        )
    for report_path in holdout_reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = _select_rows(rows, report)
        holdout_selected.extend(selected)
        holdout_slices.append(
            {
                "report_path": str(report_path.resolve()),
                "episode_idx": int(report.get("episode_idx", -1)),
                "failure_bucket": str(report.get("failure_bucket", "")),
                "primary_blocker": str(report.get("primary_blocker", "")),
                "rows": int(len(selected)),
            }
        )

    x_train_raw, y_train, symm_train, train_meta = _stack(train_selected)
    x_hold_raw, y_hold, symm_hold, hold_meta = _stack(holdout_selected)
    if x_train_raw.size == 0:
        raise RuntimeError("No training rows selected for alias baseline")

    x_train, x_hold, stats = _standardize(x_train_raw, x_hold_raw if x_hold_raw.size else np.zeros((0, x_train_raw.shape[1]), dtype=np.float64))
    w = _fit_ridge(x_train, y_train, ridge=float(ridge))
    pred_train = _predict(x_train, w)
    pred_hold = _predict(x_hold, w) if x_hold.size else np.zeros((0,), dtype=np.float64)

    train_bias = float(np.mean(symm_train - y_train)) if y_train.size else 0.0
    hold_bias = float(np.mean(symm_hold - y_hold)) if y_hold.size else 0.0
    symm_train_bc = symm_train - train_bias
    symm_hold_bc = symm_hold - train_bias

    train_report = {
        "rows": int(y_train.shape[0]),
        "raw_proxy_mae": _mae(np.asarray([m["raw_proxy_yaw"] for m in train_meta], dtype=np.float64), y_train),
        "symmetry_aware_mae": _mae(symm_train, y_train),
        "symmetry_aware_bias": train_bias,
        "symmetry_aware_bias_corrected_mae": _mae(symm_train_bc, y_train),
        "learned_mae": _mae(pred_train, y_train),
        "learned_corr": _corr(pred_train, y_train),
        "learned_sign_match": _sign_match(pred_train, y_train),
        "learned_jump_count": int(len(_jump_points(pred_train))),
    }
    hold_report = {
        "rows": int(y_hold.shape[0]),
        "raw_proxy_mae": _mae(np.asarray([m["raw_proxy_yaw"] for m in hold_meta], dtype=np.float64), y_hold) if y_hold.size else 0.0,
        "symmetry_aware_mae": _mae(symm_hold, y_hold) if y_hold.size else 0.0,
        "symmetry_aware_bias": train_bias,
        "symmetry_aware_bias_corrected_mae": _mae(symm_hold_bc, y_hold) if y_hold.size else 0.0,
        "learned_mae": _mae(pred_hold, y_hold) if y_hold.size else 0.0,
        "learned_corr": _corr(pred_hold, y_hold) if y_hold.size else 0.0,
        "learned_sign_match": _sign_match(pred_hold, y_hold) if y_hold.size else 0.0,
        "learned_jump_count": int(len(_jump_points(pred_hold))),
    }

    row_jsonl = output_dir / "yaw_alias_drift_baseline_rows.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(row_jsonl, "w", encoding="utf-8") as handle:
        for split_name, selected, feats, target, symm, pred in (
            ("train", train_selected, x_train_raw, y_train, symm_train, pred_train),
            ("holdout", holdout_selected, x_hold_raw, y_hold, symm_hold, pred_hold),
        ):
            for row, feat_vec, tgt, symm_yaw, pred_yaw in zip(selected, feats, target, symm, pred):
                proxy = _proxy(row)
                handle.write(
                    json.dumps(
                        {
                            "split": split_name,
                            "episode_idx": int(row.get("episode_idx", -1)),
                            "step_idx": int(row.get("step_idx", row.get("step", -1))),
                            "failure_bucket": str(row.get("failure_bucket", "")),
                            "primary_blocker": str(row.get("yaw_observability_primary_blocker", "")),
                            "raw_proxy_yaw": float(proxy.get("image_axis_yaw", proxy.get("dyaw", float("nan")))),
                            "symmetry_aware_proxy_yaw": float(symm_yaw),
                            "true_dyaw": float(tgt),
                            "predicted_dyaw": float(pred_yaw),
                            "residual_raw_proxy": float(_safe_float(proxy.get("image_axis_yaw", proxy.get("dyaw", float("nan")))) - float(tgt)),
                            "residual_symmetry_aware": float(symm_yaw - float(tgt)),
                            "residual_predicted": float(pred_yaw - float(tgt)),
                            "frame_observability": float(_safe_float(row.get("yaw_observability_frame_observability", row.get("source_frame_observability", 0.0)), 0.0)),
                            "frame_confidence": float(_safe_float(row.get("yaw_observability_frame_confidence", row.get("source_frame_confidence", 0.0)), 0.0)),
                            "jump_hint": bool(abs(float(_safe_float(proxy.get("image_axis_yaw", proxy.get("dyaw", float("nan")))) - float(symm_yaw))) > 0.2),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    report = {
        "schema_version": "yaw_alias_drift_baseline_v1",
        "relabel_jsonl": str(relabel_jsonl.resolve()),
        "train_reports": [str(path.resolve()) for path in train_reports],
        "holdout_reports": [str(path.resolve()) for path in holdout_reports],
        "train_slices": train_slices,
        "holdout_slices": holdout_slices,
        "feature_names": [
            "bias",
            "raw_proxy_yaw",
            "symmetry_aware_proxy_yaw",
            "proxy_confidence",
            "proxy_observability",
            "proxy_fit_residual",
            "proxy_inlier_ratio",
            "frame_confidence",
            "frame_observability",
            "frame_axis_strength",
            "wide_ring_visible",
            "wrist_occluded",
            "visual_observable",
            "planner_local_dx",
            "planner_local_dy",
            "planner_local_dz",
            "planner_local_dyaw",
            "xy_error",
        ],
        "ridge": float(ridge),
        "standardization": {
            "feature_mean": stats[0].tolist() if stats.size else [],
            "feature_std": stats[1].tolist() if stats.size else [],
        },
        "weights": w.tolist(),
        "train": train_report,
        "holdout": hold_report,
        "holdout_jump_points_raw_proxy": len(_jump_points(np.asarray([m["raw_proxy_yaw"] for m in hold_meta], dtype=np.float64))),
        "holdout_jump_points_predicted": len(_jump_points(pred_hold)),
        "holdout_jump_points_symmetry_aware": len(_jump_points(symm_hold)),
        "train_bias_from_symmetry_aware": train_bias,
        "holdout_bias_from_symmetry_aware": hold_bias,
        "rows_jsonl": str(row_jsonl.resolve()),
    }

    out_json = output_dir / "yaw_alias_drift_baseline_report.json"
    out_md = output_dir / "yaw_alias_drift_baseline_report.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Yaw Alias Drift Baseline",
        "",
        f"- train reports: `{len(train_reports)}`",
        f"- holdout reports: `{len(holdout_reports)}`",
        f"- train rows: `{train_report['rows']}`",
        f"- holdout rows: `{hold_report['rows']}`",
        f"- train symmetry-aware MAE: `{train_report['symmetry_aware_mae']:.6f}`",
        f"- train learned MAE: `{train_report['learned_mae']:.6f}`",
        f"- holdout symmetry-aware MAE: `{hold_report['symmetry_aware_mae']:.6f}`",
        f"- holdout learned MAE: `{hold_report['learned_mae']:.6f}`",
        f"- holdout raw proxy MAE: `{hold_report['raw_proxy_mae']:.6f}`",
        f"- holdout raw jump count: `{report['holdout_jump_points_raw_proxy']}`",
        f"- holdout symmetry-aware jump count: `{report['holdout_jump_points_symmetry_aware']}`",
        f"- holdout predicted jump count: `{report['holdout_jump_points_predicted']}`",
        "",
        "## Interpretation",
        "- stable alias rows should be nearly solved by the learned baseline if the symmetry-aware semantics are right.",
        "- the ep6 hard case remains a holdout; if it still has high MAE or jump-heavy raw proxy behavior, that is evidence of frame drift rather than a fixed alias.",
    ]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a tiny yaw alias drift baseline on stable alias positives and hard-case holdout.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument("--train_report", type=Path, action="append", required=True)
    ap.add_argument("--holdout_report", type=Path, action="append", required=True)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/yaw_alias_drift_baseline"),
    )
    ap.add_argument("--ridge", type=float, default=1.0e-2)
    args = ap.parse_args()

    report = run_baseline(
        relabel_jsonl=args.relabel_jsonl,
        train_reports=list(args.train_report),
        holdout_reports=list(args.holdout_report),
        output_dir=args.output_dir.resolve(),
        ridge=float(args.ridge),
    )
    print(json.dumps({"train": report["train"], "holdout": report["holdout"], "rows_jsonl": report["rows_jsonl"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
