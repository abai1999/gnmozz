#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _as_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _teacher_norms(row: dict) -> tuple[float, float, float] | None:
    metrics = dict(row.get("teacher_truth_handoff_metrics", {}) or {})
    thresholds = dict(row.get("handoff_release_metric_thresholds_provider", {}) or {})
    xy = _as_float(metrics.get("xy_error", np.nan))
    z = _as_float(metrics.get("abs_z_error", np.nan))
    yaw = _as_float(metrics.get("yaw_error", np.nan))
    xy_thr = max(_as_float(thresholds.get("xy_error", np.nan)), 1e-6)
    z_thr = max(_as_float(thresholds.get("abs_z_error", np.nan)), 1e-6)
    yaw_thr = max(_as_float(thresholds.get("yaw_error", np.nan)), 1e-6)
    vals = (xy / xy_thr, z / z_thr, yaw / yaw_thr)
    if not np.all(np.isfinite(vals)):
        return None
    return tuple(float(x) for x in vals)


def _teacher_score(norms: tuple[float, float, float]) -> tuple[float, float]:
    weighted = 0.45 * norms[0] + 0.30 * norms[1] + 0.25 * norms[2]
    max_axis = max(norms)
    return float(weighted), float(max_axis)


def _pair_metrics(pairs: list[dict]) -> dict:
    if not pairs:
        return {
            "pairs": 0,
            "pos_recall": 0.0,
            "neg_recall": 0.0,
            "balanced_acc": 0.0,
            "pred_positive_rate": 0.0,
            "decision_flip_rate": 0.0,
            "mean_score_delta": 0.0,
        }
    labels = np.asarray([float(p["label"]) for p in pairs], dtype=np.float32)
    preds = np.asarray([float(p["pred"]) for p in pairs], dtype=np.float32)
    score_delta = np.asarray([float(p["score_delta"]) for p in pairs], dtype=np.float32)
    pos = labels > 0.5
    neg = ~pos
    pos_recall = float(np.mean(preds[pos] > 0.5)) if np.any(pos) else 0.0
    neg_recall = float(np.mean(preds[neg] <= 0.5)) if np.any(neg) else 0.0
    flips = 0
    flip_den = 0
    prev = None
    for pred in preds.tolist():
        if prev is not None:
            flip_den += 1
            flips += int(int(pred) != int(prev))
        prev = int(pred)
    return {
        "pairs": int(len(pairs)),
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
        "balanced_acc": 0.5 * (pos_recall + neg_recall),
        "pred_positive_rate": float(np.mean(preds > 0.5)),
        "decision_flip_rate": float(flips / max(flip_den, 1)),
        "mean_score_delta": float(np.mean(score_delta)),
        "score_delta_p10": float(np.percentile(score_delta, 10)),
        "score_delta_p50": float(np.percentile(score_delta, 50)),
        "score_delta_p90": float(np.percentile(score_delta, 90)),
    }


def analyze_trace_dir(trace_dir: Path, closeness_margin: float, teacher_delta_margin: float) -> dict:
    overall_pairs: list[dict] = []
    close_intent_pairs: list[dict] = []
    focus_eps = {18, 34, 45, 46}
    per_episode: dict[str, dict] = {}

    for trace_path in sorted((trace_dir / "gripper_traces").glob("ep*_gripper_trace.jsonl")):
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        ep_pairs: list[dict] = []
        ep_close_pairs: list[dict] = []
        prev_row = None
        for row in rows:
            if prev_row is None:
                prev_row = row
                continue
            prev_score = _as_float(prev_row.get("handoff_pred_closeness_score", np.nan))
            cur_score = _as_float(row.get("handoff_pred_closeness_score", np.nan))
            prev_norms = _teacher_norms(prev_row)
            cur_norms = _teacher_norms(row)
            if not (np.isfinite(prev_score) and np.isfinite(cur_score) and prev_norms is not None and cur_norms is not None):
                prev_row = row
                continue
            prev_weighted, prev_max = _teacher_score(prev_norms)
            cur_weighted, cur_max = _teacher_score(cur_norms)
            delta_sum = prev_weighted - cur_weighted
            delta_max = prev_max - cur_max
            if (
                abs(delta_sum) < float(teacher_delta_margin)
                or abs(delta_max) < float(teacher_delta_margin)
                or np.sign(delta_sum) != np.sign(delta_max)
            ):
                prev_row = row
                continue
            label = 1.0 if delta_sum > 0.0 else 0.0
            score_delta = float(cur_score - prev_score)
            pred = 1.0 if score_delta > float(closeness_margin) else 0.0
            pair = {
                "label": label,
                "pred": pred,
                "score_delta": score_delta,
                "teacher_delta_weighted": float(delta_sum),
                "teacher_delta_max": float(delta_max),
                "close_intent": _as_bool(row.get("refiner_alignment_planner_close_intent", False)),
            }
            ep_pairs.append(pair)
            overall_pairs.append(pair)
            if pair["close_intent"]:
                ep_close_pairs.append(pair)
                close_intent_pairs.append(pair)
            prev_row = row

        ep_key = trace_path.stem[:5]
        ep_num = int(trace_path.stem[2:5])
        per_episode[ep_key] = {
            "episode_index": ep_num,
            "overall": _pair_metrics(ep_pairs),
            "close_intent": _pair_metrics(ep_close_pairs),
            "focus_episode": bool(ep_num in focus_eps),
        }

    focus = {k: v for k, v in per_episode.items() if v["focus_episode"]}
    return {
        "closeness_margin_threshold": float(closeness_margin),
        "teacher_delta_margin": float(teacher_delta_margin),
        "overall": _pair_metrics(overall_pairs),
        "close_intent_only": _pair_metrics(close_intent_pairs),
        "episodes": per_episode,
        "focus_episodes": focus,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--closeness_margin_threshold", type=float, required=True)
    ap.add_argument("--teacher_delta_margin", type=float, default=0.10)
    args = ap.parse_args()

    report = analyze_trace_dir(
        trace_dir=args.trace_dir,
        closeness_margin=float(args.closeness_margin_threshold),
        teacher_delta_margin=float(args.teacher_delta_margin),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
