#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: np.asarray(raw[k]) for k in raw.files}


def _pick(arrs: dict[str, np.ndarray], *keys: str, default: np.ndarray | None = None) -> np.ndarray:
    for key in keys:
        if key in arrs:
            return np.asarray(arrs[key])
    if default is not None:
        return np.asarray(default)
    raise KeyError(f"none of the keys exist: {keys}")


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
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


def _mask_summary(
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    yaw_presence_threshold: float,
    yaw_match_tol: float,
) -> dict[str, object]:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return {"rows": 0}

    actions = np.asarray(_pick(data, "proposal_actions_local", "candidate_actions_local"), dtype=np.float32)[idx]
    geom_gain = np.asarray(_pick(data, "proposal_geometry_gain", "candidate_geometry_gain"), dtype=np.float32)[idx]
    risk_delta = np.asarray(_pick(data, "proposal_risk_delta", "candidate_risk_delta"), dtype=np.float32)[idx]
    pareto_mask = np.asarray(_pick(data, "proposal_pareto_mask", "candidate_pareto_mask"), dtype=np.float32)[idx] > 0.5
    budget_mask = np.asarray(_pick(data, "proposal_budget_mask", "candidate_budget_mask"), dtype=np.float32)[idx] > 0.5
    best_safe_idx = np.asarray(_pick(data, "proposal_best_safe_index", "candidate_best_safe_index"), dtype=np.int64)[idx]
    geom_top1_idx = np.asarray(_pick(data, "proposal_geom_top1_index", "candidate_geom_top1_index"), dtype=np.int64)[idx]
    baseline_idx = np.asarray(_pick(data, "proposal_baseline_index", "candidate_baseline_index"), dtype=np.int64)[idx]
    target_delta = np.asarray(_pick(data, "proposal_target_delta_local", "candidate_target_delta_local"), dtype=np.float32)[idx]

    row = np.arange(idx.size, dtype=np.int64)
    base_geom = geom_gain[row, baseline_idx]
    base_risk = risk_delta[row, baseline_idx]
    utility = geom_gain - np.maximum(risk_delta, 0.0)
    utility_best_idx = np.argmax(utility, axis=1)
    risk_best_idx = np.argmin(risk_delta, axis=1)
    geometry_best_idx = np.argmax(np.where(np.isfinite(geom_gain), geom_gain, -np.inf), axis=1)

    target_yaw = target_delta[:, 5]
    yaw_opportunity = np.abs(target_yaw) > float(yaw_presence_threshold)
    cand_yaw_abs = np.abs(actions[:, :, 5])
    target_yaw_abs = np.abs(target_yaw)[:, None]
    yaw_match_mask = np.abs(cand_yaw_abs - target_yaw_abs) <= float(yaw_match_tol)
    yaw_sign_mask = np.zeros_like(yaw_match_mask, dtype=bool)
    yaw_sign_mask[yaw_opportunity] = (
        ((actions[yaw_opportunity, :, 5] > 0) & (target_yaw[yaw_opportunity, None] > 0))
        | ((actions[yaw_opportunity, :, 5] < 0) & (target_yaw[yaw_opportunity, None] < 0))
    )

    safe_count = np.sum(pareto_mask, axis=1)
    yaw_match_count = np.sum(yaw_match_mask, axis=1)

    best_safe_is_pareto = pareto_mask[row, best_safe_idx]
    best_safe_has_yaw_match = yaw_match_mask[row, best_safe_idx]
    best_safe_is_utility_best = best_safe_idx == utility_best_idx
    pareto_has_yaw_match = np.any(pareto_mask & yaw_match_mask, axis=1)
    yaw_match_is_pareto = np.zeros((idx.size,), dtype=bool)
    for i in range(idx.size):
        if not np.any(yaw_match_mask[i]):
            continue
        candidates = np.where(yaw_match_mask[i])[0]
        # Among yaw-match candidates, choose the one with highest utility.
        best_yaw_candidate = candidates[int(np.argmax(utility[i, candidates]))]
        yaw_match_is_pareto[i] = bool(pareto_mask[i, best_yaw_candidate])
    utility_best_is_pareto = pareto_mask[row, utility_best_idx]
    geometry_best_is_best_safe = geometry_best_idx == best_safe_idx
    risk_best_is_best_safe = risk_best_idx == best_safe_idx
    best_safe_is_utility_best = best_safe_idx == utility_best_idx
    best_safe_is_pareto = pareto_mask[row, best_safe_idx]

    best_safe_yaw_match_all = best_safe_has_yaw_match
    best_safe_yaw_sign_all = np.zeros((idx.size,), dtype=bool)
    best_safe_yaw_sign_all[yaw_opportunity] = yaw_sign_mask[yaw_opportunity, best_safe_idx[yaw_opportunity]]
    best_safe_yaw_match_on_yaw = best_safe_has_yaw_match[yaw_opportunity]
    best_safe_yaw_sign_on_yaw = best_safe_yaw_sign_all[yaw_opportunity]
    pareto_has_yaw_match_on_yaw = pareto_has_yaw_match[yaw_opportunity]
    yaw_match_is_pareto_on_yaw = yaw_match_is_pareto[yaw_opportunity]

    out = {
        "rows": int(idx.size),
        "yaw_opportunity_rows": int(np.sum(yaw_opportunity)),
        "best_safe_is_pareto_rate": float(np.mean(best_safe_is_pareto)),
        "best_safe_has_yaw_match_rate": float(np.mean(best_safe_yaw_match_all)),
        "best_safe_has_yaw_match_rate_on_yaw_opportunity": float(np.mean(best_safe_yaw_match_on_yaw)) if np.any(yaw_opportunity) else 0.0,
        "best_safe_has_yaw_sign_rate_on_yaw_opportunity": float(np.mean(best_safe_yaw_sign_on_yaw)) if np.any(yaw_opportunity) else 0.0,
        "best_safe_is_utility_best_rate": float(np.mean(best_safe_is_utility_best)),
        "pareto_has_yaw_match_rate": float(np.mean(pareto_has_yaw_match)),
        "pareto_has_yaw_match_rate_on_yaw_opportunity": float(np.mean(pareto_has_yaw_match_on_yaw)) if np.any(yaw_opportunity) else 0.0,
        "yaw_match_is_pareto_rate": float(np.mean(yaw_match_is_pareto)),
        "yaw_match_is_pareto_rate_on_yaw_opportunity": float(np.mean(yaw_match_is_pareto_on_yaw)) if np.any(yaw_opportunity) else 0.0,
        "utility_best_is_pareto_rate": float(np.mean(utility_best_is_pareto)),
        "geometry_best_is_best_safe_rate": float(np.mean(geometry_best_is_best_safe)),
        "risk_best_is_best_safe_rate": float(np.mean(risk_best_is_best_safe)),
        "num_pareto_per_row_mean": float(np.mean(safe_count)),
        "num_pareto_per_row_std": float(np.std(safe_count)),
        "num_yaw_match_per_row_mean": float(np.mean(yaw_match_count)),
        "num_yaw_match_per_row_std": float(np.std(yaw_match_count)),
        "best_safe_geom_gain": _summary_stats(geom_gain[row, best_safe_idx]),
        "best_safe_risk_delta": _summary_stats(risk_delta[row, best_safe_idx]),
        "utility_best_geom_gain": _summary_stats(geom_gain[row, utility_best_idx]),
        "utility_best_risk_delta": _summary_stats(risk_delta[row, utility_best_idx]),
        "geometry_best_geom_gain": _summary_stats(geom_gain[row, geometry_best_idx]),
        "geometry_best_risk_delta": _summary_stats(risk_delta[row, geometry_best_idx]),
        "risk_best_geom_gain": _summary_stats(geom_gain[row, risk_best_idx]),
        "risk_best_risk_delta": _summary_stats(risk_delta[row, risk_best_idx]),
        "best_safe_vs_geometry_best_gap_mean": float(np.mean((geom_gain[row, best_safe_idx] - geom_gain[row, geometry_best_idx]))),
        "best_safe_vs_utility_best_gap_mean": float(np.mean((geom_gain[row, best_safe_idx] - geom_gain[row, utility_best_idx]))),
        "best_safe_vs_risk_best_gap_mean": float(np.mean((geom_gain[row, best_safe_idx] - geom_gain[row, risk_best_idx]))),
        "best_safe_vs_geometry_best_risk_gap_mean": float(np.mean((risk_delta[row, best_safe_idx] - risk_delta[row, geometry_best_idx]))),
        "best_safe_vs_utility_best_risk_gap_mean": float(np.mean((risk_delta[row, best_safe_idx] - risk_delta[row, utility_best_idx]))),
        "best_safe_vs_risk_best_risk_gap_mean": float(np.mean((risk_delta[row, best_safe_idx] - risk_delta[row, risk_best_idx]))),
        "best_safe_is_baseline_rate": float(np.mean(best_safe_idx == baseline_idx)),
        "geometry_best_is_baseline_rate": float(np.mean(geometry_best_idx == baseline_idx)),
        "utility_best_is_baseline_rate": float(np.mean(utility_best_idx == baseline_idx)),
        "risk_best_is_baseline_rate": float(np.mean(risk_best_idx == baseline_idx)),
        "baseline_geom_gain": _summary_stats(base_geom),
        "baseline_risk_delta": _summary_stats(base_risk),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal_cache_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--yaw_presence_threshold", type=float, default=0.0025)
    ap.add_argument("--yaw_match_tol", type=float, default=0.0015)
    ap.add_argument("--weak_episodes", type=str, default="1,8,10,19")
    ap.add_argument("--strong_episodes", type=str, default="5,16,17,20")
    args = ap.parse_args()

    data = _load_npz(Path(args.proposal_cache_npz))
    n = int(np.asarray(_pick(data, "proposal_actions_local", "candidate_actions_local")).shape[0])
    eps = np.asarray(data.get("episode_index", np.zeros((n,), dtype=np.int64)), dtype=np.int64)
    yaw_target = np.asarray(_pick(data, "proposal_target_delta_local", "candidate_target_delta_local"), dtype=np.float32)[:, 5]
    yaw_opp = np.abs(yaw_target) > float(args.yaw_presence_threshold)
    weak_eps = {int(x) for x in args.weak_episodes.split(",") if str(x).strip()}
    strong_eps = {int(x) for x in args.strong_episodes.split(",") if str(x).strip()}

    groups = {
        "all_rows": np.arange(n, dtype=np.int64),
        "yaw_opportunity_rows": np.where(yaw_opp)[0],
        "non_yaw_rows": np.where(~yaw_opp)[0],
        "weak_episodes": np.where(np.isin(eps, sorted(weak_eps)))[0],
        "strong_episodes": np.where(np.isin(eps, sorted(strong_eps)))[0],
    }
    report: dict[str, object] = {
        "proposal_cache_npz": args.proposal_cache_npz,
        "rows": n,
        "episodes": int(np.unique(eps).size),
        "yaw_presence_threshold": float(args.yaw_presence_threshold),
        "yaw_match_tol": float(args.yaw_match_tol),
        "groups": {},
    }
    for gname, idx in groups.items():
        report["groups"][gname] = _mask_summary(
            data,
            np.asarray(idx, dtype=np.int64),
            yaw_presence_threshold=float(args.yaw_presence_threshold),
            yaw_match_tol=float(args.yaw_match_tol),
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
