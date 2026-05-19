#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _parse_csv_ints(text: str | None) -> set[int]:
    out: set[int] = set()
    if not text:
        return out
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def _yaw_bucket(abs_yaw: float, keep_abs: float, small_abs: float, large_abs: float) -> str:
    if abs_yaw < keep_abs:
        return "no_yaw"
    if abs_yaw < small_abs:
        return "small_yaw"
    if abs_yaw < large_abs:
        return "medium_yaw"
    return "large_yaw"


def _axis_dominance(action: np.ndarray) -> str:
    xy = float(np.hypot(float(action[0]), float(action[1])))
    z = float(abs(float(action[2])))
    yaw = float(abs(float(action[5])))
    axis = int(np.argmax(np.asarray([xy, z, yaw], dtype=np.float32)))
    return ("xy", "z", "yaw")[axis]


def _finite_or(x: np.ndarray, default: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.where(np.isfinite(x), x, float(default))
    return x


def _classify_row(
    pred_action: np.ndarray,
    oracle_action: np.ndarray,
    baseline_action: np.ndarray,
    oracle_scores: np.ndarray,
    candidate_mask: np.ndarray,
    mode: int,
    pred_regret: float,
    baseline_regret: float,
    keep_yaw_abs: float,
    small_yaw_abs: float,
    large_yaw_abs: float,
    candidate_score_std_min: float,
    oracle_baseline_gap_min: float,
) -> list[str]:
    labels: list[str] = []

    valid = candidate_mask > 0.5
    score = oracle_scores[valid]
    score = score[np.isfinite(score)]
    score_std = float(np.std(score)) if score.size else 0.0
    score_gap = float(np.max(score) - np.min(score)) if score.size else 0.0
    if score_std < float(candidate_score_std_min) or score_gap < float(oracle_baseline_gap_min):
        labels.append("candidate_bank_missing")
        return labels

    if int(mode) == 0 and math.isfinite(pred_regret) and math.isfinite(baseline_regret) and pred_regret > baseline_regret + 1e-6:
        labels.append("mode_keep_failure")
    elif int(mode) == 1 and math.isfinite(pred_regret) and math.isfinite(baseline_regret) and pred_regret > baseline_regret + 1e-6:
        labels.append("mode_apply_failure")

    pred_yaw = float(abs(float(pred_action[5])))
    oracle_yaw = float(abs(float(oracle_action[5])))
    base_yaw = float(abs(float(baseline_action[5])))

    pred_bucket = _yaw_bucket(pred_yaw, keep_yaw_abs, small_yaw_abs, large_yaw_abs)
    oracle_bucket = _yaw_bucket(oracle_yaw, keep_yaw_abs, small_yaw_abs, large_yaw_abs)
    base_bucket = _yaw_bucket(base_yaw, keep_yaw_abs, small_yaw_abs, large_yaw_abs)
    oracle_axis = _axis_dominance(oracle_action)
    pred_axis = _axis_dominance(pred_action)

    if pred_bucket == "large_yaw" and oracle_bucket in {"no_yaw", "small_yaw"}:
        labels.append("large_yaw_overuse")
        labels.append("small_vs_large_yaw")
        if oracle_axis == "xy":
            labels.append("xy_over_yaw")
        elif oracle_axis == "z":
            labels.append("z_over_yaw")
        return labels

    if pred_bucket in {"medium_yaw", "large_yaw"} and oracle_bucket in {"no_yaw", "small_yaw"}:
        labels.append("yaw_not_needed_but_selected")
        if oracle_axis == "xy":
            labels.append("xy_over_yaw")
        elif oracle_axis == "z":
            labels.append("z_over_yaw")

    if pred_bucket in {"medium_yaw", "large_yaw"} and oracle_bucket in {"medium_yaw", "large_yaw"}:
        if np.sign(float(pred_action[5])) != np.sign(float(oracle_action[5])) and abs(float(oracle_action[5])) >= float(small_yaw_abs):
            labels.append("wrong_yaw_sign")

    if oracle_bucket in {"medium_yaw", "large_yaw"} and pred_bucket in {"no_yaw", "small_yaw"}:
        labels.append("yaw_needed_but_not_selected")

    if not labels:
        if pred_axis != oracle_axis:
            labels.append(f"{pred_axis}_vs_{oracle_axis}")
        else:
            labels.append("generic_negative")

    if base_bucket != oracle_bucket and oracle_bucket in {"medium_yaw", "large_yaw"}:
        labels.append("baseline_yaw_mismatch")

    return labels


def _load_npz_dict(path: str) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    return {k: np.asarray(raw[k]) for k in raw.files}


def _episode_row_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    changed = sum(bool(r["changed"]) for r in rows)
    better = sum(bool(r["better"]) for r in rows)
    worse = sum(bool(r["worse"]) for r in rows)
    ties = sum(bool(r["tie"]) for r in rows)
    deltas = [float(r["regret_delta"]) for r in rows if math.isfinite(float(r["regret_delta"]))]
    failure_types = Counter()
    for r in rows:
        if bool(r["worse"]):
            failure_types[str(r["primary_failure_type"])] += 1
    dominant = failure_types.most_common(1)[0][0] if failure_types else "none"
    return {
        "changed_frames": int(changed),
        "better_frames": int(better),
        "worse_frames": int(worse),
        "tie_frames": int(ties),
        "better_rate": float(better / max(changed, 1)),
        "worse_rate": float(worse / max(changed, 1)),
        "regret_delta_mean_baseline_minus_pred": float(np.mean(deltas)) if deltas else float("nan"),
        "regret_delta_p50": float(np.median(deltas)) if deltas else float("nan"),
        "regret_delta_p10": float(np.percentile(deltas, 10)) if deltas else float("nan"),
        "dominant_failure_type": dominant,
        "failure_type_counts": dict(failure_types),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    ap.add_argument("--small_yaw_abs", type=float, default=0.05)
    ap.add_argument("--large_yaw_abs", type=float, default=0.09)
    ap.add_argument("--candidate_score_std_min", type=float, default=0.5)
    ap.add_argument("--oracle_baseline_gap_min", type=float, default=1.0)
    ap.add_argument("--negative_only", action="store_true", default=False)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_reports: list[dict[str, object]] = []
    episode_rows_out: list[dict[str, object]] = []
    all_episode_reports: list[dict[str, object]] = []
    overall_rows = 0
    overall_changed = 0
    overall_better = 0
    overall_worse = 0
    overall_tie = 0
    overall_failure_types = Counter()
    overall_yaw_buckets = Counter()

    for support_path in args.support_npz:
        batch_label = Path(support_path).parent.name
        data = _load_npz_dict(support_path)
        gate_open = np.asarray(data.get("b2_candidate_shadow_gate_open", np.zeros((len(next(iter(data.values()))),), dtype=np.float32)), dtype=np.float32) > 0.5
        changed = np.asarray(data.get("b2_candidate_shadow_changed", np.zeros_like(gate_open, dtype=np.float32)), dtype=np.float32) > 0.5
        regret_delta = np.asarray(data.get("b2_candidate_shadow_regret_delta", np.full_like(gate_open, np.nan, dtype=np.float32)), dtype=np.float32)
        valid = gate_open & changed & np.isfinite(regret_delta)

        if not np.any(valid):
            continue

        episode_index = np.asarray(data.get("episode_index", np.full_like(regret_delta, -1, dtype=np.int64)), dtype=np.int64)
        candidate_actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
        candidate_mask = np.asarray(data.get("candidate_mask", np.ones(candidate_actions.shape[:2], dtype=np.float32)), dtype=np.float32)
        pred_idx = np.asarray(data.get("pred_candidate_index", data.get("b2_candidate_shadow_pred_index", np.full_like(episode_index, -1))), dtype=np.int64)
        baseline_idx = np.asarray(data.get("runtime_selected_candidate_index", data.get("b2_candidate_shadow_baseline_index", np.full_like(episode_index, -1))), dtype=np.int64)
        best_idx = np.asarray(data.get("oracle_candidate_index", data.get("b2_candidate_shadow_best_index", np.full_like(episode_index, -1))), dtype=np.int64)
        mode = np.asarray(data.get("b2_candidate_shadow_mode", np.full_like(episode_index, -1)), dtype=np.int64)
        oracle_scores = np.asarray(data["candidate_oracle_score"], dtype=np.float32)
        pred_regret = np.asarray(data.get("b2_candidate_shadow_pred_regret", np.full_like(regret_delta, np.nan, dtype=np.float32)), dtype=np.float32)
        base_regret = np.asarray(data.get("b2_candidate_shadow_baseline_regret", np.full_like(regret_delta, np.nan, dtype=np.float32)), dtype=np.float32)

        unique_eps = sorted(int(x) for x in np.unique(episode_index[valid]))
        batch_episode_rows: list[dict[str, object]] = []
        for ep in unique_eps:
            ep_mask = valid & (episode_index == ep)
            ep_rows: list[dict[str, object]] = []
            ep_indices = np.where(ep_mask)[0]
            for i in ep_indices:
                if pred_idx[i] < 0 or baseline_idx[i] < 0 or best_idx[i] < 0:
                    continue
                cand_ok = (
                    pred_idx[i] < candidate_actions.shape[1]
                    and baseline_idx[i] < candidate_actions.shape[1]
                    and best_idx[i] < candidate_actions.shape[1]
                )
                if not cand_ok:
                    continue
                pred_action = candidate_actions[i, pred_idx[i]]
                baseline_action = candidate_actions[i, baseline_idx[i]]
                oracle_action = candidate_actions[i, best_idx[i]]
                labels = _classify_row(
                    pred_action=pred_action,
                    oracle_action=oracle_action,
                    baseline_action=baseline_action,
                    oracle_scores=oracle_scores[i],
                    candidate_mask=candidate_mask[i],
                    mode=int(mode[i]),
                    pred_regret=float(pred_regret[i]),
                    baseline_regret=float(base_regret[i]),
                    keep_yaw_abs=float(args.keep_yaw_abs),
                    small_yaw_abs=float(args.small_yaw_abs),
                    large_yaw_abs=float(args.large_yaw_abs),
                    candidate_score_std_min=float(args.candidate_score_std_min),
                    oracle_baseline_gap_min=float(args.oracle_baseline_gap_min),
                )
                primary = labels[0] if labels else "generic_negative"
                row = {
                    "batch_label": batch_label,
                    "episode_index": int(ep),
                    "row_index": int(i),
                    "changed": True,
                    "better": bool(regret_delta[i] > 1e-6),
                    "worse": bool(regret_delta[i] < -1e-6),
                    "tie": bool(abs(float(regret_delta[i])) <= 1e-6),
                    "regret_delta": float(regret_delta[i]),
                    "pred_regret": float(pred_regret[i]) if math.isfinite(float(pred_regret[i])) else float("nan"),
                    "baseline_regret": float(base_regret[i]) if math.isfinite(float(base_regret[i])) else float("nan"),
                    "shadow_mode": int(mode[i]),
                    "pred_index": int(pred_idx[i]),
                    "baseline_index": int(baseline_idx[i]),
                    "oracle_index": int(best_idx[i]),
                    "pred_yaw_abs": float(abs(float(pred_action[5]))),
                    "baseline_yaw_abs": float(abs(float(baseline_action[5]))),
                    "oracle_yaw_abs": float(abs(float(oracle_action[5]))),
                    "pred_yaw_bucket": _yaw_bucket(abs(float(pred_action[5])), float(args.keep_yaw_abs), float(args.small_yaw_abs), float(args.large_yaw_abs)),
                    "baseline_yaw_bucket": _yaw_bucket(abs(float(baseline_action[5])), float(args.keep_yaw_abs), float(args.small_yaw_abs), float(args.large_yaw_abs)),
                    "oracle_yaw_bucket": _yaw_bucket(abs(float(oracle_action[5])), float(args.keep_yaw_abs), float(args.small_yaw_abs), float(args.large_yaw_abs)),
                    "primary_failure_type": primary,
                    "failure_types": labels,
                    "oracle_score_std": float(np.std(_finite_or(oracle_scores[i][candidate_mask[i] > 0.5], -1e9))),
                    "candidate_count": int(np.sum(candidate_mask[i] > 0.5)),
                    "has_yaw_candidate": bool(np.any(np.abs(candidate_actions[i, :, 5]) > float(args.keep_yaw_abs))),
                    "oracle_has_yaw": bool(abs(float(oracle_action[5])) > float(args.keep_yaw_abs)),
                    "baseline_has_yaw": bool(abs(float(baseline_action[5])) > float(args.keep_yaw_abs)),
                    "pred_has_yaw": bool(abs(float(pred_action[5])) > float(args.keep_yaw_abs)),
                }
                ep_rows.append(row)
            if not ep_rows:
                continue
            summary = _episode_row_summary(ep_rows)
            summary.update({
                "batch_label": batch_label,
                "episode_index": int(ep),
                "trace_file": str(Path(support_path).name),
                "num_rows_considered": int(len(ep_rows)),
            })
            batch_episode_rows.append(summary)
            all_episode_reports.append(summary)
            overall_rows += len(ep_rows)
            overall_changed += summary["changed_frames"]
            overall_better += summary["better_frames"]
            overall_worse += summary["worse_frames"]
            overall_tie += summary["tie_frames"]
            for r in ep_rows:
                if bool(r["worse"]):
                    overall_failure_types[str(r["primary_failure_type"])] += 1
                    overall_yaw_buckets[str(r["pred_yaw_bucket"])] += 1
                    episode_rows_out.append(r)

        batch_report = {
            "batch_label": batch_label,
            "support_npz": str(support_path),
            "episodes": batch_episode_rows,
            "overall": _episode_row_summary([r for r in episode_rows_out if r["batch_label"] == batch_label]),
        }
        batch_reports.append(batch_report)

    overall = {
        "rows": int(overall_rows),
        "changed_frames": int(overall_changed),
        "better_frames": int(overall_better),
        "worse_frames": int(overall_worse),
        "tie_frames": int(overall_tie),
        "better_rate": float(overall_better / max(overall_changed, 1)),
        "worse_rate": float(overall_worse / max(overall_changed, 1)),
        "failure_type_counts": dict(overall_failure_types),
        "pred_yaw_bucket_counts_in_worse": dict(overall_yaw_buckets),
    }
    report = {"overall": overall, "batches": batch_reports, "episodes": all_episode_reports}
    (out_dir / "alignment_v4b_negative_episode_triage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # frame-level detail for supplement construction
    frame_path = out_dir / "alignment_v4b_negative_episode_triage_frames.jsonl"
    with frame_path.open("w", encoding="utf-8") as f:
        for row in episode_rows_out:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
