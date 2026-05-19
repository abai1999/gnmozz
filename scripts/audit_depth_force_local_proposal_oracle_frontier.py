#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_depth_force_candidate_cost_dataset import _load_npz, _risk_join, _safe_target_actions
from local_proposal_utils import (
    LocalProposalConfig,
    evaluate_state_conditioned_proposals,
    make_state_conditioned_proposals,
    select_best_indices,
    select_planner_action,
    select_pose_fields,
    summary_stats,
)
from build_pose_candidate_dataset import pose_delta_local_between


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


def _summary_row(
    geom: np.ndarray,
    risk: np.ndarray,
    candidate_actions: np.ndarray,
    *,
    baseline_idx: int,
    geo_margin: float,
    risk_budget: float,
    soft_alpha: float,
    target_delta_yaw: np.ndarray,
) -> dict[str, float | int]:
    geom = np.asarray(geom, dtype=np.float32).reshape(-1)
    risk = np.asarray(risk, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate_actions, dtype=np.float32).reshape(-1, 6)
    base_i = int(np.clip(int(baseline_idx), 0, geom.shape[0] - 1))
    base_geom = float(geom[base_i])
    base_risk = float(risk[base_i])
    geom_gain = base_geom - geom
    risk_delta = risk - base_risk
    safe_mask = (geom_gain > float(geo_margin)) & (risk_delta <= float(risk_budget))
    utility = geom_gain - float(soft_alpha) * np.maximum(risk_delta, 0.0)
    utility = np.where(np.isfinite(utility), utility, -1e9)
    safe_utility = np.where(safe_mask, utility, -1e9)
    budget_mask = risk_delta <= float(risk_budget)
    budget_utility = np.where(budget_mask, utility, -1e9)
    geom_top1 = int(np.argmin(np.where(np.isfinite(geom), geom, np.inf)))
    best_safe = int(np.argmax(safe_utility)) if np.any(safe_mask) else base_i
    best_soft = int(np.argmax(utility))
    best_budget = int(np.argmax(budget_utility)) if np.any(budget_mask) else base_i
    frontier = _frontier_size(geom_gain, risk_delta, np.isfinite(geom) & np.isfinite(risk))
    best_safe_yaw = abs(float(cand[best_safe, 5])) > 0.02
    geom_top1_yaw = abs(float(cand[geom_top1, 5])) > 0.02
    best_soft_yaw = abs(float(cand[best_soft, 5])) > 0.02
    best_budget_yaw = abs(float(cand[best_budget, 5])) > 0.02
    tgt_yaw = float(target_delta_yaw.reshape(-1)[0])
    tgt_sign = 0
    if tgt_yaw > 1e-6:
        tgt_sign = 1
    elif tgt_yaw < -1e-6:
        tgt_sign = -1

    def _sign_match(val: float) -> bool:
        if abs(tgt_yaw) <= 0.02:
            return False
        s = 1 if val > 1e-6 else -1 if val < -1e-6 else 0
        return s == tgt_sign and s != 0

    return {
        "baseline_idx": base_i,
        "geom_top1_idx": geom_top1,
        "best_safe_idx": best_safe,
        "best_soft_idx": best_soft,
        "best_budget_idx": best_budget,
        "baseline_geom": base_geom,
        "baseline_risk": base_risk,
        "geom_top1_geom": float(geom[geom_top1]),
        "geom_top1_risk": float(risk[geom_top1]),
        "best_safe_geom": float(geom[best_safe]),
        "best_safe_risk": float(risk[best_safe]),
        "best_soft_geom": float(geom[best_soft]),
        "best_soft_risk": float(risk[best_soft]),
        "best_budget_geom": float(geom[best_budget]),
        "best_budget_risk": float(risk[best_budget]),
        "geom_top1_geom_gain": float(geom_gain[geom_top1]),
        "best_safe_geom_gain": float(geom_gain[best_safe]),
        "best_soft_geom_gain": float(geom_gain[best_soft]),
        "best_budget_geom_gain": float(geom_gain[best_budget]),
        "geom_top1_risk_delta": float(risk_delta[geom_top1]),
        "best_safe_risk_delta": float(risk_delta[best_safe]),
        "best_soft_risk_delta": float(risk_delta[best_soft]),
        "best_budget_risk_delta": float(risk_delta[best_budget]),
        "pareto_feasible": int(np.any(safe_mask)),
        "missing_compromise": int(not np.any(safe_mask)),
        "safe_count": int(np.sum(safe_mask)),
        "frontier_size": int(frontier),
        "geom_top1_safe": int(bool(safe_mask[geom_top1])),
        "best_safe_is_baseline": int(best_safe == base_i),
        "best_safe_is_geom_top1": int(best_safe == geom_top1),
        "geom_top1_risk_increase": int(risk_delta[geom_top1] > 1e-6),
        "best_safe_risk_increase": int(risk_delta[best_safe] > 1e-6),
        "best_soft_risk_increase": int(risk_delta[best_soft] > 1e-6),
        "best_budget_risk_increase": int(risk_delta[best_budget] > 1e-6),
        "geom_top1_yaw": int(geom_top1_yaw),
        "best_safe_yaw": int(best_safe_yaw),
        "best_soft_yaw": int(best_soft_yaw),
        "best_budget_yaw": int(best_budget_yaw),
        "geom_top1_correct_yaw_sign": int(_sign_match(float(cand[geom_top1, 5]))),
        "best_safe_correct_yaw_sign": int(_sign_match(float(cand[best_safe, 5]))),
        "best_soft_correct_yaw_sign": int(_sign_match(float(cand[best_soft, 5]))),
        "best_budget_correct_yaw_sign": int(_sign_match(float(cand[best_budget, 5]))),
    }


def _group_summary(
    data: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    geo_margin: float,
    risk_budget: float,
    soft_alpha: float,
) -> dict[str, object]:
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return {"rows": 0}

    current_pose = np.asarray(data["current_pose_7d"], dtype=np.float32)[idx]
    target_pose = np.asarray(
        data.get("privileged_motion_target_pose_7d", data.get("privileged_basin_center_pose_7d")),
        dtype=np.float32,
    )[idx]
    planner = select_planner_action(data)[idx]
    contact = np.asarray(labels["contact_label"], dtype=np.float32)[idx]
    force_spike = np.asarray(labels["force_spike_label"], dtype=np.float32)[idx]
    jam = np.asarray(labels["jam_label"], dtype=np.float32)[idx]
    stall = np.asarray(labels["motion_stall_label"], dtype=np.float32)[idx]
    near_depth = np.asarray(labels["near_depth_label"], dtype=np.float32)[idx]
    kin_invalid = np.asarray(labels["kinematic_invalid_label"], dtype=np.float32)[idx]
    action_invalid = np.asarray(labels["action_range_invalid_label"], dtype=np.float32)[idx]
    gripper_state = np.asarray(data.get("gripper_state", np.zeros((data["current_pose_7d"].shape[0],), dtype=np.float32)), dtype=np.float32)[idx]

    target_delta = np.asarray(data.get("privileged_current_delta_basin_target"), dtype=np.float32)[idx]
    target_delta_yaw = target_delta[:, 5]
    if "privileged_current_delta_basin_target" not in data:
        target_delta = np.stack([pose_delta_local_between(current_pose[i], target_pose[i]) for i in range(idx.size)], axis=0).astype(np.float32)
        target_delta_yaw = target_delta[:, 5]

    safe_target, target_mode = _safe_target_actions(
        planner,
        contact=contact,
        force_spike=force_spike,
        jam=jam,
        motion_stall=stall,
        near_depth=near_depth,
        kin_invalid=kin_invalid,
        action_range_invalid=action_invalid,
    )
    cfg = LocalProposalConfig()

    proposal_actions: list[np.ndarray] = []
    proposal_kind: list[np.ndarray] = []
    proposal_family: list[np.ndarray] = []
    for row_i in range(idx.size):
        acts, kinds, families = make_state_conditioned_proposals(
            planner[row_i],
            target_delta[row_i],
            safe_target[row_i],
            cfg=cfg,
            row_seed=int(idx[row_i] * 1009 + 17),
        )
        proposal_actions.append(acts)
        proposal_kind.append(kinds)
        proposal_family.append(families)

    proposal_actions_arr = np.stack(proposal_actions, axis=0)
    proposal_kind_arr = np.stack(proposal_kind, axis=0)
    proposal_family_arr = np.stack(proposal_family, axis=0)
    scored = evaluate_state_conditioned_proposals(
        current_pose=current_pose,
        target_pose=target_pose,
        candidate_actions=proposal_actions_arr,
        contact=contact,
        force_spike=force_spike,
        jam=jam,
        motion_stall=stall,
        near_depth=near_depth,
        kin_invalid=kin_invalid,
        action_range_invalid=action_invalid,
        gripper_state=gripper_state,
    )
    geom = scored["candidate_geometry_cost"]
    risk = scored["candidate_risk_cost"]
    best = select_best_indices(
        geom,
        risk,
        baseline_index=0,
        geo_margin=geo_margin,
        risk_budget=risk_budget,
        soft_alpha=soft_alpha,
    )

    row_ids = np.arange(idx.size, dtype=np.int64)
    base_geom = geom[row_ids, best["baseline_index"]]
    base_risk = risk[row_ids, best["baseline_index"]]
    geom_top1 = best["geom_top1_index"]
    best_safe = best["best_safe_index"]
    best_soft = best["best_soft_index"]
    best_budget = best["best_budget_index"]

    def _yaw_rate(sel: np.ndarray) -> np.ndarray:
        return np.abs(proposal_actions_arr[row_ids, sel, 5]) > 0.02

    def _correct_yaw(sel: np.ndarray) -> np.ndarray:
        sel_yaw = proposal_actions_arr[row_ids, sel, 5]
        tgt = target_delta_yaw
        out = np.zeros_like(sel_yaw, dtype=bool)
        for i, (sy, ty) in enumerate(zip(sel_yaw.tolist(), tgt.tolist())):
            if abs(ty) <= 0.02:
                out[i] = False
            else:
                out[i] = (sy > 0 and ty > 0) or (sy < 0 and ty < 0)
        return out

    def _metric(sel: np.ndarray) -> dict[str, float]:
        sel_geom = geom[row_ids, sel]
        sel_risk = risk[row_ids, sel]
        return {
            "geometry_improve_rate": float(np.mean(base_geom > sel_geom + 1e-6)),
            "geometry_retention": float(np.mean((base_geom - sel_geom) / np.maximum(base_geom - geom[row_ids, geom_top1], 1e-6))),
            "risk_nonincrease_rate": float(np.mean(sel_risk <= base_risk + 1e-6)),
            "risk_delta_mean": float(np.mean(sel_risk - base_risk)),
            "geometry_gain_mean": float(np.mean(base_geom - sel_geom)),
            "yaw_selected_rate": float(np.mean(_yaw_rate(sel))),
            "correct_yaw_sign_rate": float(np.mean(_correct_yaw(sel))),
            "safe_rate": float(np.mean((base_geom - sel_geom > geo_margin) & ((sel_risk - base_risk) <= risk_budget))),
        }

    geom_gain = base_geom - geom[row_ids, geom_top1]
    risk_delta = risk[row_ids, geom_top1] - base_risk
    safe_mask = best["safe_mask"] > 0.5
    out = {
        "rows": int(idx.size),
        "candidate_count": int(proposal_actions_arr.shape[1]),
        "proposal_family_hist": {
            k: int(v) for k, v in zip(*np.unique(proposal_family_arr.reshape(-1), return_counts=True))
        },
        "proposal_kind_hist": {
            k: int(v) for k, v in zip(*np.unique(proposal_kind_arr.reshape(-1), return_counts=True))
        },
        "pareto_feasible_rate": float(np.mean(best["pareto_feasible"])),
        "missing_compromise_rate": float(np.mean(best["missing_compromise"])),
        "frontier_size": summary_stats(best["frontier_size"]),
        "safe_count": summary_stats(best["safe_count"]),
        "geom_top1_geometry_gain": summary_stats(geom_gain),
        "geom_top1_risk_delta": summary_stats(risk_delta),
        "geom_top1_risk_increase_rate": float(np.mean(best["geom_top1_risk_increase"])),
        "geom_top1_safe_rate": float(np.mean(best["geom_top1_safe"])),
        "best_safe_geometry_gain": summary_stats(base_geom - geom[row_ids, best_safe]),
        "best_safe_risk_delta": summary_stats(risk[row_ids, best_safe] - base_risk),
        "best_soft_geometry_gain": summary_stats(base_geom - geom[row_ids, best_soft]),
        "best_soft_risk_delta": summary_stats(risk[row_ids, best_soft] - base_risk),
        "best_budget_geometry_gain": summary_stats(base_geom - geom[row_ids, best_budget]),
        "best_budget_risk_delta": summary_stats(risk[row_ids, best_budget] - base_risk),
        "best_safe_is_baseline_rate": float(np.mean(best_safe == best["baseline_index"])),
        "best_safe_is_geom_top1_rate": float(np.mean(best_safe == geom_top1)),
        "geom_top1_yaw_rate": float(np.mean(_yaw_rate(geom_top1))),
        "best_safe_yaw_rate": float(np.mean(_yaw_rate(best_safe))),
        "best_soft_yaw_rate": float(np.mean(_yaw_rate(best_soft))),
        "best_budget_yaw_rate": float(np.mean(_yaw_rate(best_budget))),
        "geom_top1_correct_yaw_sign_rate": float(np.mean(_correct_yaw(geom_top1))),
        "best_safe_correct_yaw_sign_rate": float(np.mean(_correct_yaw(best_safe))),
        "best_soft_correct_yaw_sign_rate": float(np.mean(_correct_yaw(best_soft))),
        "best_budget_correct_yaw_sign_rate": float(np.mean(_correct_yaw(best_budget))),
        "geom_top1": _metric(geom_top1),
        "best_safe": _metric(best_safe),
        "best_soft": _metric(best_soft),
        "best_budget": _metric(best_budget),
        "candidate_geometry_gain_mean": float(np.mean(best["geometry_gain"])),
        "candidate_risk_delta_mean": float(np.mean(best["risk_delta"])),
        "candidate_safe_mask_rate": float(np.mean(safe_mask)),
        "candidate_budget_mask_rate": float(np.mean(best["budget_mask"])),
        "candidate_target_mode_hist": {
            k: int(v) for k, v in zip(*np.unique(np.asarray(target_mode, dtype="U32"), return_counts=True))
        },
        "proposal_family_hist": {
            k: int(v) for k, v in zip(*np.unique(proposal_family_arr.reshape(-1), return_counts=True))
        },
        "proposal_kind_hist": {
            k: int(v) for k, v in zip(*np.unique(proposal_kind_arr.reshape(-1), return_counts=True))
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--risk_npz", default="")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--geo_margin", type=float, default=0.0)
    ap.add_argument("--risk_budget", type=float, default=0.05)
    ap.add_argument("--soft_alpha", type=float, default=0.3)
    ap.add_argument("--groups", type=str, default="all_rows,original_rows,yaw_augmented_rows,yaw_opportunity_rows,non_yaw_rows")
    args = ap.parse_args()

    support = _load_npz(Path(args.support_npz))
    risk = _load_npz(Path(args.risk_npz)) if args.risk_npz else None
    data = dict(support) if risk is None else {**support, **risk}
    labels = _risk_join(support, risk)

    report = {"dataset_npz": args.support_npz, "risk_npz": args.risk_npz, "groups": {}, "episodes_by_id": {}}
    n = int(data["current_pose_7d"].shape[0])
    eps = np.asarray(data.get("episode_index", np.zeros((n,), dtype=np.int64)), dtype=np.int64)
    yaw_aug = np.asarray(data.get("yaw_augmentation_applied", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(data.get("yaw_opportunity_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    group_idx = {
        "all_rows": np.arange(n, dtype=np.int64),
        "original_rows": np.arange(n, dtype=np.int64)[~yaw_aug],
        "yaw_augmented_rows": np.arange(n, dtype=np.int64)[yaw_aug],
        "yaw_opportunity_rows": np.arange(n, dtype=np.int64)[yaw_opp],
        "non_yaw_rows": np.arange(n, dtype=np.int64)[~yaw_opp],
    }

    for gname in [x.strip() for x in args.groups.split(",") if x.strip()]:
        if gname in group_idx:
            report["groups"][gname] = _group_summary(
                data,
                labels,
                group_idx[gname],
                geo_margin=float(args.geo_margin),
                risk_budget=float(args.risk_budget),
                soft_alpha=float(args.soft_alpha),
            )

    for ep in sorted(int(x) for x in np.unique(eps)):
        report["episodes_by_id"][str(ep)] = _group_summary(
            data,
            labels,
            np.where(eps == ep)[0],
            geo_margin=float(args.geo_margin),
            risk_budget=float(args.risk_budget),
            soft_alpha=float(args.soft_alpha),
        )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
