#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _hist(arr: np.ndarray) -> dict[str, int]:
    uniq, cnt = np.unique(arr, return_counts=True)
    return {str(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}


def _safe_argmin(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, cost, np.inf)
    return np.argmin(masked, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    data = {k: np.asarray(v) for k, v in np.load(args.dataset_npz, allow_pickle=False).items()}
    if "candidate_risk_cost" not in data:
        raise SystemExit("dataset does not contain candidate_risk_cost")

    risk = np.asarray(data["candidate_risk_cost"], dtype=np.float32)
    geom = np.asarray(data["candidate_geometry_cost"], dtype=np.float32)
    total = np.asarray(data["candidate_total_cost"], dtype=np.float32)
    mask = np.asarray(data.get("candidate_mask", np.ones_like(total, dtype=np.float32)), dtype=np.float32) > 0.5
    best_total = np.asarray(data["candidate_best_index"], dtype=np.int64)
    baseline = np.asarray(data["candidate_baseline_index"], dtype=np.int64)
    rows = np.arange(total.shape[0], dtype=np.int64)

    best_risk = _safe_argmin(risk, mask)
    best_geom = np.asarray(data.get("candidate_best_geometry_index", data.get("best_geometry_candidate_index", best_total)), dtype=np.int64)

    selected_risk = risk[rows, best_total]
    baseline_risk = risk[rows, baseline]
    selected_geom = geom[rows, best_total]
    baseline_geom = geom[rows, baseline]
    risk_best = risk[rows, best_risk]
    geom_best_risk = risk[rows, best_geom]
    best_risk_gap = baseline_risk - risk_best
    selected_risk_gap = baseline_risk - selected_risk

    risk_std = np.nanstd(np.where(mask, risk, np.nan), axis=1)
    risk_range = np.nanmax(np.where(mask, risk, np.nan), axis=1) - np.nanmin(np.where(mask, risk, np.nan), axis=1)
    risk_std = np.nan_to_num(risk_std, nan=0.0)
    risk_range = np.nan_to_num(risk_range, nan=0.0)

    report = {
        "dataset_npz": str(args.dataset_npz),
        "rows": int(total.shape[0]),
        "episodes": int(np.unique(np.asarray(data["episode_index"], dtype=np.int64)).size) if "episode_index" in data else 0,
        "candidate_count": int(total.shape[1]),
        "candidate_risk_cost_std": _summary_stats(risk_std),
        "candidate_risk_cost_range": _summary_stats(risk_range),
        "candidate_risk_dependent_rate": float(np.mean(risk_std > 1e-6)),
        "risk_best_equals_baseline_rate": float(np.mean(best_risk == baseline)),
        "risk_best_equals_geometry_best_rate": float(np.mean(best_risk == best_geom)),
        "geometry_best_equals_baseline_rate": float(np.mean(best_geom == baseline)),
        "selected_risk_nonincrease_rate": float(np.mean(selected_risk <= baseline_risk + 1e-6)),
        "selected_risk_improves_rate": float(np.mean(selected_risk < baseline_risk - 1e-6)),
        "selected_risk_gap": _summary_stats(selected_risk_gap),
        "risk_best_gap": _summary_stats(best_risk_gap),
        "selected_geom_gap": _summary_stats(baseline_geom - selected_geom),
        "risk_best_vs_geometry_best_gap": _summary_stats(risk[rows, best_risk] - geom[rows, best_geom]),
        "best_risk_index_hist_top": dict(sorted(_hist(best_risk).items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "best_geom_index_hist_top": dict(sorted(_hist(best_geom).items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "per_episode": {},
    }

    if "episode_index" in data:
        ep_arr = np.asarray(data["episode_index"], dtype=np.int64)
        for ep in sorted(int(x) for x in np.unique(ep_arr)):
            m = ep_arr == ep
            report["per_episode"][str(ep)] = {
                "rows": int(np.sum(m)),
                "candidate_risk_dependent_rate": float(np.mean(risk_std[m] > 1e-6)) if np.any(m) else 0.0,
                "risk_best_equals_baseline_rate": float(np.mean(best_risk[m] == baseline[m])) if np.any(m) else 0.0,
                "risk_best_equals_geometry_best_rate": float(np.mean(best_risk[m] == best_geom[m])) if np.any(m) else 0.0,
                "selected_risk_nonincrease_rate": float(np.mean(selected_risk[m] <= baseline_risk[m] + 1e-6)) if np.any(m) else 0.0,
                "selected_risk_gap": _summary_stats(selected_risk_gap[m]) if np.any(m) else _summary_stats(np.array([])),
                "risk_best_gap": _summary_stats(best_risk_gap[m]) if np.any(m) else _summary_stats(np.array([])),
            }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
