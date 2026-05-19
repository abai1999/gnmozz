#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def _safe_load(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: np.asarray(raw[k]) for k in raw.files}


def _pick(arrs: dict[str, np.ndarray], *keys: str, default: np.ndarray | None = None) -> np.ndarray:
    for key in keys:
        if key in arrs:
            return np.asarray(arrs[key])
    if default is not None:
        return np.asarray(default)
    raise KeyError(f"none of the keys exist: {keys}")


def _episode_summary(data: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, object]:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return {"rows": 0}
    candidate_future_risk = np.asarray(data["candidate_future_risk_score"], dtype=np.float32)[idx]
    candidate_future_risk_delta = np.asarray(data["candidate_future_risk_delta"], dtype=np.float32)[idx]
    candidate_future_risk_mask = np.asarray(data.get("candidate_future_risk_mask", np.ones_like(candidate_future_risk)), dtype=np.float32)[idx] > 0.5
    baseline_idx = np.asarray(data["candidate_future_risk_baseline_index"], dtype=np.int64)[idx]
    geom_idx = np.asarray(data["candidate_future_risk_geom_index"], dtype=np.int64)[idx]
    best_idx = np.asarray(data["candidate_future_risk_best_index"], dtype=np.int64)[idx]
    candidate_actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)[idx]
    yaw_opp = np.asarray(data.get("yaw_opportunity_label", np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.float32)), dtype=np.float32)[idx] > 0.5
    yaw_aug = np.asarray(data.get("yaw_augmentation_applied", np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.float32)), dtype=np.float32)[idx] > 0.5

    def _row_metric(sel_idx: np.ndarray, base_idx: np.ndarray) -> dict[str, float]:
        r = np.arange(sel_idx.shape[0], dtype=np.int64)
        sel_risk = candidate_future_risk[r, sel_idx]
        base_risk = candidate_future_risk[r, base_idx]
        sel_delta = candidate_future_risk_delta[r, sel_idx]
        base_delta = candidate_future_risk_delta[r, base_idx]
        geom_sel = np.asarray(
            data.get("candidate_privileged_geometry_cost", data.get("candidate_geometry_cost")),
            dtype=np.float32,
        )[idx][r, sel_idx]
        geom_base = np.asarray(
            data.get("candidate_privileged_geometry_cost", data.get("candidate_geometry_cost")),
            dtype=np.float32,
        )[idx][r, base_idx]
        yaw_sel = np.abs(candidate_actions[r, sel_idx, 5]) > 0.02
        yaw_base = np.abs(candidate_actions[r, base_idx, 5]) > 0.02
        return {
            "risk_mean": float(np.mean(sel_risk)),
            "risk_delta_mean": float(np.mean(sel_delta)),
            "risk_nonincrease_rate": float(np.mean(sel_risk <= base_risk + 1e-6)),
            "geometry_improve_rate": float(np.mean(geom_sel < geom_base - 1e-6)),
            "geometry_regret_delta_mean": float(np.mean(geom_base - geom_sel)),
            "yaw_selected_rate": float(np.mean(yaw_sel)),
            "yaw_nonzero_rate": float(np.mean(yaw_sel)),
            "baseline_yaw_rate": float(np.mean(yaw_base)),
        }

    return {
        "rows": int(idx.size),
        "future_risk_score": _summary_stats(candidate_future_risk.reshape(-1)),
        "future_risk_delta": _summary_stats(candidate_future_risk_delta.reshape(-1)),
        "candidate_future_risk_dependent_rate": float(np.mean(np.std(candidate_future_risk, axis=1) > 1e-6)),
        "candidate_future_risk_range_mean": float(np.mean(np.max(candidate_future_risk, axis=1) - np.min(candidate_future_risk, axis=1))),
        "candidate_future_risk_range_p90": float(np.percentile(np.max(candidate_future_risk, axis=1) - np.min(candidate_future_risk, axis=1), 90)),
        "best_is_baseline_rate": float(np.mean(best_idx == baseline_idx)),
        "best_is_geometry_rate": float(np.mean(best_idx == geom_idx)),
        "geometry_selected_risk_increase_rate": float(np.mean(candidate_future_risk[np.arange(idx.size), geom_idx] > candidate_future_risk[np.arange(idx.size), baseline_idx] + 1e-6)),
        "geometry_selected_risk_nonincrease_rate": float(np.mean(candidate_future_risk[np.arange(idx.size), geom_idx] <= candidate_future_risk[np.arange(idx.size), baseline_idx] + 1e-6)),
        "best_vs_baseline_risk_delta_mean": float(np.mean(candidate_future_risk[np.arange(idx.size), baseline_idx] - candidate_future_risk[np.arange(idx.size), best_idx])),
        "best_vs_geom_risk_delta_mean": float(np.mean(candidate_future_risk[np.arange(idx.size), geom_idx] - candidate_future_risk[np.arange(idx.size), best_idx])),
        "future_risk_contact_rate": float(np.mean(np.asarray(data.get("future_contact_label", np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.float32)))[idx] > 0.5)),
        "future_risk_spike_rate": float(np.mean(np.asarray(data.get("future_force_spike_label", np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.float32)))[idx] > 0.5)),
        "future_risk_jam_rate": float(np.mean(np.asarray(data.get("future_jam_label", np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.float32)))[idx] > 0.5)),
        "future_risk_stall_rate": float(np.mean(np.asarray(data.get("future_motion_stall_label", np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.float32)))[idx] > 0.5)),
        "future_risk_kin_invalid_rate": float(np.mean(np.asarray(data.get("future_kinematic_invalid_label", np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.float32)))[idx] > 0.5)),
        "future_risk_action_invalid_rate": float(np.mean(np.asarray(data.get("future_action_range_invalid_label", np.zeros((data["candidate_actions_local"].shape[0],), dtype=np.float32)))[idx] > 0.5)),
        "yaw_opportunity_rate": float(np.mean(yaw_opp)),
        "yaw_augmentation_rate": float(np.mean(yaw_aug)),
        "geom_selected": _row_metric(geom_idx, baseline_idx),
        "best_selected": _row_metric(best_idx, baseline_idx),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()

    data = _safe_load(Path(args.dataset_npz))
    n = int(data["candidate_actions_local"].shape[0])
    eps = np.asarray(data.get("episode_index", np.zeros((n,), dtype=np.int64)), dtype=np.int64)
    yaw_aug = np.asarray(data.get("yaw_augmentation_applied", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(data.get("yaw_opportunity_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5

    groups = {
        "all_rows": np.arange(n, dtype=np.int64),
        "original_rows": np.arange(n, dtype=np.int64)[~yaw_aug],
        "yaw_augmented_rows": np.arange(n, dtype=np.int64)[yaw_aug],
        "yaw_opportunity_rows": np.arange(n, dtype=np.int64)[yaw_opp],
        "non_yaw_rows": np.arange(n, dtype=np.int64)[~yaw_opp],
    }

    report: dict[str, object] = {
        "dataset_npz": args.dataset_npz,
        "rows": n,
        "episodes": int(np.unique(eps).size),
        "margin": float(args.margin),
        "groups": {},
        "episodes_by_id": {},
    }

    for gname, idx in groups.items():
        report["groups"][gname] = _episode_summary(data, idx)

    for ep in sorted(int(x) for x in np.unique(eps)):
        report["episodes_by_id"][str(ep)] = _episode_summary(data, np.where(eps == ep)[0])

    # Candidate-kind audit
    cand_kind = np.asarray(data.get("candidate_kind", np.zeros_like(data["candidate_actions_local"][..., 0], dtype="U16")))
    cand_kind = cand_kind.astype("U16")
    flat_kind = cand_kind.reshape(-1)
    fut_risk = np.asarray(data["candidate_future_risk_score"], dtype=np.float32)
    fut_mask = np.asarray(data.get("candidate_future_risk_mask", np.ones_like(fut_risk)), dtype=np.float32) > 0.5
    cand_actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    best_idx = np.asarray(data["candidate_future_risk_best_index"], dtype=np.int64)
    geom_idx = np.asarray(data["candidate_future_risk_geom_index"], dtype=np.int64)
    base_idx = np.asarray(data["candidate_future_risk_baseline_index"], dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)

    kind_report: dict[str, object] = {}
    for kind in sorted(set(str(k) for k in flat_kind.tolist())):
        if not kind:
            continue
        mask = np.asarray(cand_kind == kind)
        if not np.any(mask):
            continue
        kind_report[kind] = {
            "count": int(np.sum(mask)),
            "rate": float(np.mean(mask)),
            "future_risk_mean": float(np.mean(fut_risk[mask])),
            "future_risk_std": float(np.std(fut_risk[mask])),
            "future_risk_p90": float(np.percentile(fut_risk[mask], 90)),
            "candidate_risk_mask_rate": float(np.mean(fut_mask[mask])),
        }
    report["candidate_kind"] = kind_report
    report["candidate_future_risk_mask_rate"] = float(np.mean(fut_mask))
    report["candidate_future_risk_best_equals_baseline_rate"] = float(np.mean(best_idx == base_idx))
    report["candidate_future_risk_best_equals_geometry_rate"] = float(np.mean(best_idx == geom_idx))
    report["candidate_future_risk_geometry_vs_baseline_gap_mean"] = float(
        np.mean(fut_risk[rows, geom_idx] - fut_risk[rows, base_idx])
    )
    report["candidate_future_risk_geometry_vs_baseline_gap_p90"] = float(
        np.percentile(fut_risk[rows, geom_idx] - fut_risk[rows, base_idx], 90)
    )
    report["candidate_future_risk_geometry_vs_best_gap_mean"] = float(
        np.mean(fut_risk[rows, geom_idx] - fut_risk[rows, best_idx])
    )
    report["candidate_future_risk_baseline_vs_best_gap_mean"] = float(
        np.mean(fut_risk[rows, base_idx] - fut_risk[rows, best_idx])
    )

    out_json = Path(args.output_json) if args.output_json else Path(args.dataset_npz).with_name("candidate_future_risk_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
