#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _safe_second_best(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(cost, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    masked = np.where(mask, arr, np.inf)
    # Replace best with +inf and take the min again.
    best_idx = np.argmin(masked, axis=1)
    row = np.arange(arr.shape[0])
    masked[row, best_idx] = np.inf
    return np.min(masked, axis=1)


def _hist(arr: np.ndarray) -> dict[str, int]:
    uniq, cnt = np.unique(arr, return_counts=True)
    out = {}
    for u, c in zip(uniq.tolist(), cnt.tolist()):
        out[str(u)] = int(c)
    return out


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32)
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


def _best_gap_summary(cost: np.ndarray, mask: np.ndarray, best_idx: np.ndarray, baseline_idx: np.ndarray) -> dict[str, float]:
    row = np.arange(cost.shape[0])
    best = cost[row, best_idx]
    base = cost[row, baseline_idx]
    second = _safe_second_best(cost, mask)
    best_base = base - best
    best_second = second - best
    total_std = np.std(np.where(mask, cost, np.nan), axis=1)
    total_std = np.nan_to_num(total_std, nan=0.0)
    total_range = np.nanmax(np.where(mask, cost, np.nan), axis=1) - np.nanmin(np.where(mask, cost, np.nan), axis=1)
    total_range = np.nan_to_num(total_range, nan=0.0)
    return {
        "best_vs_baseline_mean": float(np.mean(best_base)),
        "best_vs_baseline_p50": float(np.percentile(best_base, 50)),
        "best_vs_baseline_p90": float(np.percentile(best_base, 90)),
        "best_vs_baseline_zero_frac": float(np.mean(np.abs(best_base) < 1e-4)),
        "best_vs_second_mean": float(np.mean(best_second)),
        "best_vs_second_p50": float(np.percentile(best_second, 50)),
        "best_vs_second_p90": float(np.percentile(best_second, 90)),
        "best_vs_second_zero_frac": float(np.mean(np.abs(best_second) < 1e-4)),
        "total_cost_std_mean": float(np.mean(total_std)),
        "total_cost_std_p50": float(np.percentile(total_std, 50)),
        "total_cost_std_p90": float(np.percentile(total_std, 90)),
        "total_cost_range_mean": float(np.mean(total_range)),
        "total_cost_range_p50": float(np.percentile(total_range, 50)),
        "total_cost_range_p90": float(np.percentile(total_range, 90)),
    }


def _mode_hist_per_label(modes: np.ndarray, label: np.ndarray) -> dict[str, int]:
    mask = np.asarray(label, dtype=np.float32) > 0.5
    if not np.any(mask):
        return {}
    return _hist(modes[mask])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    data = {k: np.asarray(v) for k, v in np.load(args.dataset_npz, allow_pickle=False).items()}
    cost = np.asarray(data["candidate_total_cost"], dtype=np.float32)
    geom = np.asarray(data["candidate_geometry_cost"], dtype=np.float32)
    risk = np.asarray(data["candidate_risk_cost"], dtype=np.float32)
    mask = np.asarray(data.get("candidate_mask", np.ones_like(cost, dtype=np.float32)), dtype=np.float32) > 0.5
    best = np.asarray(data["candidate_best_index"], dtype=np.int64)
    best_geom_idx = np.asarray(
        data.get("best_geometry_candidate_index", data.get("candidate_best_geometry_index", best)),
        dtype=np.int64,
    )
    baseline = np.asarray(data["candidate_baseline_index"], dtype=np.int64)
    target_mode = np.asarray(data.get("candidate_target_mode", np.full((cost.shape[0],), "unknown", dtype="U32"))).astype(str)
    candidate_kind = np.asarray(data.get("candidate_kind", np.full(cost.shape, "unknown", dtype="U16"))).astype(str)
    row = np.arange(cost.shape[0], dtype=np.int64)
    best_cost = cost[row, best]
    base_cost = cost[row, baseline]
    best_geom = geom[row, best]
    geom_best_geom = geom[row, best_geom_idx]
    geom_best_risk = risk[row, best_geom_idx]
    geom_best_total = cost[row, best_geom_idx]
    base_geom = geom[row, baseline]
    best_risk = risk[row, best]
    base_risk = risk[row, baseline]
    total_improve = base_cost - best_cost
    geom_improve = base_geom - best_geom
    geom_best_improve = base_geom - geom_best_geom
    risk_improve = base_risk - best_risk
    second = _safe_second_best(cost, mask)

    report = {
        "dataset_npz": str(args.dataset_npz),
        "rows": int(cost.shape[0]),
        "episodes": int(np.unique(np.asarray(data["episode_index"], dtype=np.int64)).size) if "episode_index" in data else 0,
        "candidate_count": int(cost.shape[1]),
        "candidate_kind_hist": _hist(candidate_kind.reshape(-1)),
        "target_mode_hist": _hist(target_mode),
        "best_candidate_index_hist": _hist(best),
        "baseline_candidate_index_hist_top": dict(sorted(_hist(baseline).items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "best_vs_baseline_gap": _summary_stats(base_cost - best_cost),
        "best_vs_second_gap": _summary_stats(second - best_cost),
        "total_cost": _summary_stats(cost[mask]),
        "geometry_cost": _summary_stats(geom[mask]),
        "risk_cost": _summary_stats(risk[mask]),
        "best_cost": _summary_stats(best_cost),
        "baseline_cost": _summary_stats(base_cost),
        "total_improvement": _summary_stats(total_improve),
        "geometry_improvement": _summary_stats(geom_improve),
        "risk_improvement": _summary_stats(risk_improve),
        "best_is_baseline_rate": float(np.mean(best == baseline)),
        "best_geometry_is_baseline_rate": float(np.mean(best_geom_idx == baseline)),
        "best_is_yaw_rate": float(
            np.mean(np.abs(np.asarray(data["candidate_actions_local"], dtype=np.float32)[row, best, 5]) > 0.02)
        )
        if "candidate_actions_local" in data
        else 0.0,
        "best_geometry_is_yaw_rate": float(
            np.mean(np.abs(np.asarray(data["candidate_actions_local"], dtype=np.float32)[row, best_geom_idx, 5]) > 0.02)
        )
        if "candidate_actions_local" in data
        else 0.0,
        "geometry_best_vs_baseline_gap": _summary_stats(geom_best_improve),
        "geometry_best_vs_second_gap": _best_gap_summary(geom, mask, best_geom_idx, baseline),
        "geometry_risk_conflict": {
            "geometry_best_risk_cost": _summary_stats(geom_best_risk),
            "risk_aware_best_risk_cost": _summary_stats(best_risk),
            "geometry_best_total_cost": _summary_stats(geom_best_total),
            "risk_aware_best_total_cost": _summary_stats(best_cost),
            "geometry_best_risk_delta_vs_baseline": _summary_stats(base_risk - geom_best_risk),
            "risk_aware_best_geometry_delta_vs_baseline": _summary_stats(base_geom - best_geom),
        },
        "yaw_opportunity": {
            "count": int(np.sum(np.asarray(data.get("yaw_opportunity_label", np.zeros((cost.shape[0],), dtype=np.float32))) > 0.5)),
            "rate": float(np.mean(np.asarray(data.get("yaw_opportunity_label", np.zeros((cost.shape[0],), dtype=np.float32))) > 0.5)),
        },
        "per_label": {
            key: {
                "rows": int(np.sum(np.asarray(data[key], dtype=np.float32) > 0.5)) if key in data else 0,
                "best_is_baseline_rate": float(np.mean(best[np.asarray(data[key], dtype=np.float32) > 0.5] == baseline[np.asarray(data[key], dtype=np.float32) > 0.5])) if key in data and np.any(np.asarray(data[key], dtype=np.float32) > 0.5) else 0.0,
                "best_vs_baseline_gap": _summary_stats((base_cost - best_cost)[np.asarray(data[key], dtype=np.float32) > 0.5]) if key in data and np.any(np.asarray(data[key], dtype=np.float32) > 0.5) else _summary_stats(np.array([])),
                "total_improvement": _summary_stats(total_improve[np.asarray(data[key], dtype=np.float32) > 0.5]) if key in data and np.any(np.asarray(data[key], dtype=np.float32) > 0.5) else _summary_stats(np.array([])),
            }
            for key in [
                "contact_label",
                "force_spike_label",
                "jam_label",
                "motion_stall_label",
                "kinematic_invalid_label",
                "action_range_invalid_label",
                "near_depth_label",
            ]
            if key in data
        },
        "per_mode": {},
        "per_episode": {},
    }

    if "candidate_actions_local" in data:
        cand = np.asarray(data["candidate_actions_local"], dtype=np.float32)
        best_actions = cand[row, best]
        baseline_actions = cand[row, baseline]
        report["best_action_norm"] = _summary_stats(np.linalg.norm(best_actions, axis=1))
        report["baseline_action_norm"] = _summary_stats(np.linalg.norm(baseline_actions, axis=1))
        report["best_action_xyz_norm"] = _summary_stats(np.linalg.norm(best_actions[:, :3], axis=1))
        report["best_action_yaw_abs"] = _summary_stats(np.abs(best_actions[:, 5]))
        report["baseline_action_yaw_abs"] = _summary_stats(np.abs(baseline_actions[:, 5]))
        geom_best_actions = cand[row, best_geom_idx]
        report["best_geometry_action_norm"] = _summary_stats(np.linalg.norm(geom_best_actions, axis=1))
        report["best_geometry_action_yaw_abs"] = _summary_stats(np.abs(geom_best_actions[:, 5]))
        yaw_mask = np.abs(cand[:, :, 5]) > 0.02
        yaw_best_cost = np.min(np.where(yaw_mask, geom, np.inf), axis=1)
        no_yaw_best_cost = np.min(np.where(~yaw_mask, geom, np.inf), axis=1)
        yaw_improve = no_yaw_best_cost - yaw_best_cost
        finite_yaw = np.isfinite(yaw_improve)
        report["yaw_opportunity"].update(
            {
                "yaw_candidate_improves_rate": float(np.mean(yaw_improve[finite_yaw] > 1e-6)) if np.any(finite_yaw) else 0.0,
                "yaw_candidate_gap": _summary_stats(yaw_improve[finite_yaw]) if np.any(finite_yaw) else _summary_stats(np.array([])),
                "small_yaw_best_rate": float(
                    np.mean((np.abs(geom_best_actions[:, 5]) > 0.02) & (np.abs(geom_best_actions[:, 5]) <= 0.07))
                ),
                "large_yaw_best_rate": float(np.mean(np.abs(geom_best_actions[:, 5]) > 0.07)),
            }
        )

    for mode in sorted(set(target_mode.tolist())):
        m = target_mode == mode
        if not np.any(m):
            continue
        report["per_mode"][mode] = {
            "rows": int(np.sum(m)),
            "best_is_baseline_rate": float(np.mean(best[m] == baseline[m])),
            "best_vs_baseline_gap": _summary_stats((base_cost - best_cost)[m]),
            "total_improvement": _summary_stats(total_improve[m]),
            "geometry_improvement": _summary_stats(geom_improve[m]),
            "geometry_best_improvement": _summary_stats(geom_best_improve[m]),
            "risk_improvement": _summary_stats(risk_improve[m]),
            "best_geometry_is_yaw_rate": float(
                np.mean(np.abs(np.asarray(data["candidate_actions_local"], dtype=np.float32)[row[m], best_geom_idx[m], 5]) > 0.02)
            )
            if "candidate_actions_local" in data
            else 0.0,
            "candidate_kind_hist": _hist(candidate_kind[m].reshape(-1)),
        }
    if "episode_index" in data:
        for ep in sorted(int(x) for x in np.unique(np.asarray(data["episode_index"], dtype=np.int64))):
            m = np.asarray(data["episode_index"], dtype=np.int64) == ep
            report["per_episode"][str(ep)] = {
                "rows": int(np.sum(m)),
                "best_is_baseline_rate": float(np.mean(best[m] == baseline[m])),
                "best_vs_baseline_gap": _summary_stats((base_cost - best_cost)[m]),
                "total_improvement": _summary_stats(total_improve[m]),
                "geometry_improvement": _summary_stats(geom_improve[m]),
                "geometry_best_improvement": _summary_stats(geom_best_improve[m]),
                "risk_improvement": _summary_stats(risk_improve[m]),
                "best_geometry_is_yaw_rate": float(
                    np.mean(np.abs(np.asarray(data["candidate_actions_local"], dtype=np.float32)[row[m], best_geom_idx[m], 5]) > 0.02)
                )
                if "candidate_actions_local" in data
                else 0.0,
                "target_mode_hist": _hist(target_mode[m]),
            }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
