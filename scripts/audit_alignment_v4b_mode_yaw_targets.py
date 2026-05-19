#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _yaw_abs_at(actions: np.ndarray, idx: np.ndarray) -> np.ndarray:
    row = np.arange(actions.shape[0], dtype=np.int64)
    clipped = np.clip(idx.astype(np.int64), 0, actions.shape[1] - 1)
    return np.abs(actions[row, clipped, 5].astype(np.float32))


def _bool_rate(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(np.float32))) if mask.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--thresholds", type=str, default="0.02,0.035,0.05,0.06,0.075,0.10,0.12")
    args = ap.parse_args()

    raw = np.load(args.dataset_npz, allow_pickle=False)
    data = {k: np.asarray(raw[k]) for k in raw.files}
    actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    mask = np.asarray(data["candidate_mask"], dtype=np.float32) > 0.5
    scores = np.asarray(data["candidate_oracle_score"], dtype=np.float32)
    best_idx = np.asarray(data["candidate_best_index"], dtype=np.int64)
    baseline_idx = np.asarray(data["candidate_baseline_index"], dtype=np.int64)

    row = np.arange(actions.shape[0], dtype=np.int64)
    best_yaw = _yaw_abs_at(actions, best_idx)
    baseline_yaw = _yaw_abs_at(actions, baseline_idx)
    oracle_gap = scores[row, best_idx] - scores[row, baseline_idx]
    candidate_yaw_abs = np.abs(actions[:, :, 5])
    yaw_candidates_any = np.any((candidate_yaw_abs > 1e-4) & mask, axis=1)
    improving_mask = scores > scores[row, baseline_idx][:, None] + 1e-6
    improving_yaw_any = np.any((candidate_yaw_abs > 1e-4) & mask & improving_mask, axis=1)

    threshold_sweep = {}
    for t in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        best_apply = best_yaw > t
        baseline_apply = baseline_yaw > t
        improving_apply = np.any((candidate_yaw_abs > t) & mask & improving_mask, axis=1)
        threshold_sweep[str(t)] = {
            "best_apply_rate": _bool_rate(best_apply),
            "baseline_apply_rate": _bool_rate(baseline_apply),
            "improving_apply_rate": _bool_rate(improving_apply),
            "best_vs_baseline_mode_disagree_rate": _bool_rate(best_apply != baseline_apply),
        }

    summary = {
        "rows": int(actions.shape[0]),
        "episodes": int(np.unique(data["episode_index"]).size) if "episode_index" in data else 0,
        "candidate_yaw_nonzero_ratio": float(np.mean((candidate_yaw_abs > 1e-4)[mask])),
        "best_candidate_yaw_abs_mean": float(np.mean(best_yaw)),
        "best_candidate_yaw_abs_p90": float(np.percentile(best_yaw, 90)),
        "best_candidate_yaw_nonzero_ratio": _bool_rate(best_yaw > 1e-4),
        "baseline_candidate_yaw_abs_mean": float(np.mean(baseline_yaw)),
        "baseline_candidate_yaw_abs_p90": float(np.percentile(baseline_yaw, 90)),
        "baseline_candidate_yaw_nonzero_ratio": _bool_rate(baseline_yaw > 1e-4),
        "rows_with_any_yaw_candidate": _bool_rate(yaw_candidates_any),
        "rows_with_improving_yaw_candidate": _bool_rate(improving_yaw_any),
        "oracle_baseline_gap_mean": float(np.mean(oracle_gap)),
        "oracle_baseline_gap_p50": float(np.percentile(oracle_gap, 50)),
        "oracle_baseline_gap_p90": float(np.percentile(oracle_gap, 90)),
        "threshold_sweep": threshold_sweep,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
