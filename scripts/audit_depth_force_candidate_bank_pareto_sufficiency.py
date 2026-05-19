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


def _frontier_size(gain: np.ndarray, risk_delta: np.ndarray, mask: np.ndarray) -> int:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0
    g = np.asarray(gain[idx], dtype=np.float32)
    r = np.asarray(risk_delta[idx], dtype=np.float32)
    frontier = np.ones((idx.size,), dtype=bool)
    eps = 1e-6
    for i in range(idx.size):
        if not frontier[i]:
            continue
        dominated = (g >= g[i] - eps) & (r <= r[i] + eps) & ((g > g[i] + eps) | (r < r[i] - eps))
        dominated[i] = False
        if np.any(dominated):
            frontier[i] = False
    return int(np.sum(frontier))


def _row_stats(
    geom_row: np.ndarray,
    risk_row: np.ndarray,
    mask_row: np.ndarray,
    baseline_idx: int,
    geo_margin: float,
    risk_budget: float,
) -> dict[str, float | int]:
    geom = np.asarray(geom_row, dtype=np.float32).reshape(-1)
    risk = np.asarray(risk_row, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask_row, dtype=bool).reshape(-1)
    if geom.shape != risk.shape or geom.shape != mask.shape:
        raise ValueError(f"geometry/risk/mask shape mismatch: {geom.shape} vs {risk.shape} vs {mask.shape}")
    base_i = int(baseline_idx)
    if not (0 <= base_i < geom.shape[0]):
        base_i = int(np.argmax(mask)) if np.any(mask) else 0
    valid = mask & np.isfinite(geom) & np.isfinite(risk)
    if not np.any(valid):
        return {
            "pareto_feasible": 0,
            "missing_compromise": 1,
            "best_safe_idx": base_i,
            "best_safe_available": 0,
            "best_safe_geom_gain": 0.0,
            "best_safe_risk_delta": 0.0,
            "geom_top1_idx": base_i,
            "geom_top1_geom_gain": 0.0,
            "geom_top1_risk_delta": 0.0,
            "geom_top1_safe": 0,
            "geom_top1_risk_increase": 0,
            "geom_top1_risk_over_budget": 0,
            "frontier_size": 0,
            "safe_count": 0,
            "baseline_geom": float(geom[base_i]),
            "baseline_risk": float(risk[base_i]),
        }
    base_geom = float(geom[base_i])
    base_risk = float(risk[base_i])
    geom_gain = base_geom - geom
    risk_delta = risk - base_risk
    safe = valid & (geom_gain > float(geo_margin)) & (risk_delta <= float(risk_budget))
    safe_idx = np.where(safe)[0]
    pareto_feasible = int(safe_idx.size > 0)
    missing_compromise = int(safe_idx.size == 0)
    if safe_idx.size > 0:
        order = np.lexsort((risk_delta[safe_idx], -geom_gain[safe_idx]))
        best_safe = int(safe_idx[order[0]])
        best_safe_geom_gain = float(geom_gain[best_safe])
        best_safe_risk_delta = float(risk_delta[best_safe])
    else:
        best_safe = base_i
        best_safe_geom_gain = 0.0
        best_safe_risk_delta = 0.0
    geom_top1 = int(np.argmin(np.where(valid, geom, np.inf)))
    geom_top1_geom_gain = float(geom_gain[geom_top1])
    geom_top1_risk_delta = float(risk_delta[geom_top1])
    frontier_size = _frontier_size(geom_gain, risk_delta, valid)
    return {
        "pareto_feasible": pareto_feasible,
        "missing_compromise": missing_compromise,
        "best_safe_idx": best_safe,
        "best_safe_available": int(safe_idx.size > 0),
        "best_safe_geom_gain": best_safe_geom_gain,
        "best_safe_risk_delta": best_safe_risk_delta,
        "geom_top1_idx": geom_top1,
        "geom_top1_geom_gain": geom_top1_geom_gain,
        "geom_top1_risk_delta": geom_top1_risk_delta,
        "geom_top1_safe": int(bool(safe[geom_top1])),
        "geom_top1_risk_increase": int(geom_top1_risk_delta > 1e-6),
        "geom_top1_risk_over_budget": int(geom_top1_risk_delta > float(risk_budget)),
        "frontier_size": frontier_size,
        "safe_count": int(np.sum(safe)),
        "baseline_geom": base_geom,
        "baseline_risk": base_risk,
    }


def _group_summary(
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    geo_margin: float,
    risk_budget: float,
) -> dict[str, object]:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return {"rows": 0}
    geom = np.asarray(_pick(data, "candidate_privileged_geometry_cost", "candidate_geometry_cost"), dtype=np.float32)[idx]
    risk = np.asarray(_pick(data, "candidate_future_risk_score", "candidate_risk_cost", "candidate_total_cost"), dtype=np.float32)[idx]
    mask = np.asarray(_pick(data, "candidate_future_risk_mask", "candidate_mask", default=np.ones_like(risk)), dtype=np.float32)[idx] > 0.5
    baseline_idx = np.asarray(
        _pick(
            data,
            "candidate_future_risk_baseline_index",
            "candidate_baseline_index",
            "baseline_candidate_index",
            default=np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.int64),
        ),
        dtype=np.int64,
    )[idx]
    stats = [_row_stats(geom[i], risk[i], mask[i], int(baseline_idx[i]), geo_margin, risk_budget) for i in range(idx.size)]
    geom_top1_idx = np.asarray([int(s["geom_top1_idx"]) for s in stats], dtype=np.int64)
    best_safe_idx = np.asarray([int(s["best_safe_idx"]) for s in stats], dtype=np.int64)
    baseline_idx = np.asarray([int(np.clip(baseline_idx[i], 0, geom.shape[1] - 1)) for i in range(idx.size)], dtype=np.int64)
    candidate_actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    if candidate_actions.ndim == 2:
        candidate_actions = np.broadcast_to(candidate_actions[None, :, :], (data["candidate_actions_local"].shape[0], candidate_actions.shape[0], 6)).copy()
    if candidate_actions.ndim != 3:
        raise ValueError(f"candidate_actions_local must have shape (N,C,6) or (C,6); got {candidate_actions.shape}")
    cand_kind = data.get("candidate_kind", None)
    if cand_kind is not None:
        cand_kind = np.asarray(cand_kind)
        if cand_kind.ndim == 1:
            cand_kind = np.broadcast_to(cand_kind[None, :], (data["candidate_actions_local"].shape[0], cand_kind.shape[0])).copy()
        if cand_kind.shape[0] == data["candidate_actions_local"].shape[0]:
            cand_kind = cand_kind[idx]
        else:
            cand_kind = None

    row_ids = np.arange(idx.size, dtype=np.int64)
    geom_gain = np.asarray([s["best_safe_geom_gain"] for s in stats], dtype=np.float32)
    geom_top1_gain = np.asarray([s["geom_top1_geom_gain"] for s in stats], dtype=np.float32)
    risk_delta = np.asarray([s["best_safe_risk_delta"] for s in stats], dtype=np.float32)
    geom_top1_risk_delta = np.asarray([s["geom_top1_risk_delta"] for s in stats], dtype=np.float32)
    safe_count = np.asarray([s["safe_count"] for s in stats], dtype=np.float32)
    frontier_size = np.asarray([s["frontier_size"] for s in stats], dtype=np.float32)
    missing_compromise = np.asarray([s["missing_compromise"] for s in stats], dtype=np.float32)
    pareto_feasible = np.asarray([s["pareto_feasible"] for s in stats], dtype=np.float32)
    geom_top1_safe = np.asarray([s["geom_top1_safe"] for s in stats], dtype=np.float32)
    geom_top1_risk_increase = np.asarray([s["geom_top1_risk_increase"] for s in stats], dtype=np.float32)
    geom_top1_risk_over_budget = np.asarray([s["geom_top1_risk_over_budget"] for s in stats], dtype=np.float32)
    base_geom = np.asarray([s["baseline_geom"] for s in stats], dtype=np.float32)
    base_risk = np.asarray([s["baseline_risk"] for s in stats], dtype=np.float32)
    geom_row_sel = geom[row_ids, geom_top1_idx]
    base_row_sel = geom[row_ids, baseline_idx]
    risk_row_sel = risk[row_ids, geom_top1_idx]
    base_risk_sel = risk[row_ids, baseline_idx]
    best_safe_geom = geom[row_ids, best_safe_idx]
    best_safe_risk = risk[row_ids, best_safe_idx]
    best_safe_yaw = np.abs(candidate_actions[idx, best_safe_idx, 5]) > 0.02 if candidate_actions.ndim == 3 else np.zeros_like(best_safe_geom, dtype=bool)
    geom_top1_yaw = np.abs(candidate_actions[idx, geom_top1_idx, 5]) > 0.02 if candidate_actions.ndim == 3 else np.zeros_like(best_safe_geom, dtype=bool)
    baseline_yaw = np.abs(candidate_actions[idx, baseline_idx, 5]) > 0.02 if candidate_actions.ndim == 3 else np.zeros_like(best_safe_geom, dtype=bool)

    def _kind_hist(sel_idx: np.ndarray) -> dict[str, int]:
        if cand_kind is None:
            return {}
        kinds = cand_kind[row_ids, sel_idx]
        uniq, cnt = np.unique(kinds.astype("U32"), return_counts=True)
        return {str(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}

    out = {
        "rows": int(idx.size),
        "pareto_feasible_rate": float(np.mean(pareto_feasible)),
        "missing_compromise_rate": float(np.mean(missing_compromise)),
        "frontier_size": _summary_stats(frontier_size),
        "safe_count": _summary_stats(safe_count),
        "best_safe_geometry_gain": _summary_stats(geom_gain),
        "best_safe_risk_delta": _summary_stats(risk_delta),
        "geom_top1_geometry_gain": _summary_stats(geom_top1_gain),
        "geom_top1_risk_delta": _summary_stats(geom_top1_risk_delta),
        "geom_top1_risk_increase_rate": float(np.mean(geom_top1_risk_increase)),
        "geom_top1_risk_over_budget_rate": float(np.mean(geom_top1_risk_over_budget)),
        "geom_top1_safe_rate": float(np.mean(geom_top1_safe)),
        "best_safe_is_baseline_rate": float(np.mean(best_safe_idx == baseline_idx)),
        "best_safe_is_geom_top1_rate": float(np.mean(best_safe_idx == geom_top1_idx)),
        "geometry_top1_gain_vs_baseline_mean": float(np.mean(base_geom - geom_row_sel)),
        "geometry_top1_risk_delta_vs_baseline_mean": float(np.mean(risk_row_sel - base_risk_sel)),
        "best_safe_gain_vs_baseline_mean": float(np.mean(base_geom - best_safe_geom)),
        "best_safe_risk_delta_vs_baseline_mean": float(np.mean(best_safe_risk - base_risk_sel)),
        "geometry_retention_vs_top1": float(np.mean((base_geom - best_safe_geom) / np.maximum(base_geom - geom_row_sel, 1e-6))),
        "risk_reduction_vs_top1": float(np.mean(np.maximum(risk_row_sel - best_safe_risk, 0.0))),
        "geom_top1_yaw_rate": float(np.mean(geom_top1_yaw)),
        "best_safe_yaw_rate": float(np.mean(best_safe_yaw)),
        "baseline_yaw_rate": float(np.mean(baseline_yaw)),
        "geom_top1_yaw_gap": float(np.mean(np.abs(candidate_actions[idx, geom_top1_idx, 5]) - np.abs(candidate_actions[idx, baseline_idx, 5]))),
        "best_safe_yaw_gap": float(np.mean(np.abs(candidate_actions[idx, best_safe_idx, 5]) - np.abs(candidate_actions[idx, baseline_idx, 5]))),
        "geom_top1_geometry_improve_rate": float(np.mean((base_geom - geom_row_sel) > 1e-6)),
        "best_safe_geometry_improve_rate": float(np.mean((base_geom - best_safe_geom) > 1e-6)),
        "candidate_kind_best_safe_hist": _kind_hist(best_safe_idx),
        "candidate_kind_geom_top1_hist": _kind_hist(geom_top1_idx),
        "candidate_kind_baseline_hist": _kind_hist(baseline_idx),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--geo_margin", type=float, default=0.0)
    ap.add_argument("--risk_budget", type=float, default=0.05)
    ap.add_argument("--weak_episodes", type=str, default="1,8,10,19")
    ap.add_argument("--strong_episodes", type=str, default="5,16,17,20")
    args = ap.parse_args()

    data = _load_npz(Path(args.dataset_npz))
    n = int(data["candidate_actions_local"].shape[0])
    eps = np.asarray(data.get("episode_index", np.zeros((n,), dtype=np.int64)), dtype=np.int64)
    yaw_aug = np.asarray(data.get("yaw_augmentation_applied", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(data.get("yaw_opportunity_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    weak_eps = {int(x) for x in args.weak_episodes.split(",") if str(x).strip()}
    strong_eps = {int(x) for x in args.strong_episodes.split(",") if str(x).strip()}

    groups = {
        "all_rows": np.arange(n, dtype=np.int64),
        "original_rows": np.arange(n, dtype=np.int64)[~yaw_aug],
        "yaw_augmented_rows": np.arange(n, dtype=np.int64)[yaw_aug],
        "yaw_opportunity_rows": np.arange(n, dtype=np.int64)[yaw_opp],
        "non_yaw_rows": np.arange(n, dtype=np.int64)[~yaw_opp],
        "weak_episodes": np.where(np.isin(eps, sorted(weak_eps)))[0],
        "strong_episodes": np.where(np.isin(eps, sorted(strong_eps)))[0],
    }

    report: dict[str, object] = {
        "dataset_npz": args.dataset_npz,
        "rows": n,
        "episodes": int(np.unique(eps).size),
        "geo_margin": float(args.geo_margin),
        "risk_budget": float(args.risk_budget),
        "groups": {},
        "episodes_by_id": {},
    }
    for gname, idx in groups.items():
        report["groups"][gname] = _group_summary(data, np.asarray(idx, dtype=np.int64), geo_margin=float(args.geo_margin), risk_budget=float(args.risk_budget))

    for ep in sorted(int(x) for x in np.unique(eps)):
        report["episodes_by_id"][str(ep)] = _group_summary(
            data,
            np.where(eps == ep)[0],
            geo_margin=float(args.geo_margin),
            risk_budget=float(args.risk_budget),
        )

    out_json = Path(args.output_json) if args.output_json else Path(args.dataset_npz).with_name("candidate_bank_pareto_sufficiency_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
