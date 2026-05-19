#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_depth_force_local_proposal_oracle_frontier import _group_summary
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--risk_npz", default="")
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--geo_margin", type=float, default=0.0)
    ap.add_argument("--risk_budget", type=float, default=0.05)
    ap.add_argument("--soft_alpha", type=float, default=0.3)
    args = ap.parse_args()

    support = _load_npz(Path(args.support_npz))
    risk = _load_npz(Path(args.risk_npz)) if args.risk_npz else None
    labels = _risk_join(support, risk)
    current_pose, target_pose, target_source = select_pose_fields(support)
    planner = select_planner_action(support)
    n = int(current_pose.shape[0])
    if planner.shape[0] != n:
        if planner.shape[0] == 1:
            planner = np.repeat(planner, n, axis=0)
        else:
            raise ValueError("planner action row count mismatch")

    contact = np.asarray(labels["contact_label"], dtype=np.float32)
    force_spike = np.asarray(labels["force_spike_label"], dtype=np.float32)
    jam = np.asarray(labels["jam_label"], dtype=np.float32)
    stall = np.asarray(labels["motion_stall_label"], dtype=np.float32)
    near_depth = np.asarray(labels["near_depth_label"], dtype=np.float32)
    kin_invalid = np.asarray(labels["kinematic_invalid_label"], dtype=np.float32)
    action_invalid = np.asarray(labels["action_range_invalid_label"], dtype=np.float32)
    gripper_state = np.asarray(support.get("gripper_state", np.zeros((n,), dtype=np.float32)), dtype=np.float32)

    if "privileged_current_delta_basin_target" in support:
        target_delta = np.asarray(support["privileged_current_delta_basin_target"], dtype=np.float32)
    else:
        target_delta = np.stack([pose_delta_local_between(current_pose[i], target_pose[i]) for i in range(n)], axis=0).astype(np.float32)

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
    for row_i in range(n):
        acts, kinds, families = make_state_conditioned_proposals(
            planner[row_i],
            target_delta[row_i],
            safe_target[row_i],
            cfg=cfg,
            row_seed=int((int(support.get("episode_index", np.arange(n))[row_i]) + 1) * 1009 + int(support.get("step_index", np.arange(n))[row_i]) * 17),
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
    risk_cost = scored["candidate_risk_cost"]
    best = select_best_indices(
        geom,
        risk_cost,
        baseline_index=0,
        geo_margin=float(args.geo_margin),
        risk_budget=float(args.risk_budget),
        soft_alpha=float(args.soft_alpha),
    )

    row_ids = np.arange(n, dtype=np.int64)
    base_geom = geom[row_ids, best["baseline_index"]]
    base_risk = risk_cost[row_ids, best["baseline_index"]]
    geom_top1 = best["geom_top1_index"]
    best_safe = best["best_safe_index"]
    best_soft = best["best_soft_index"]
    best_budget = best["best_budget_index"]

    out: dict[str, np.ndarray] = {}
    for key in [
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "gripper_context",
        "force_history",
        "force_history_raw",
        "force_history_normalized",
        "ft_hist",
        "gripper_touch_forces",
        "force_norm",
        "torque_norm",
        "force_delta_norm",
        "torque_delta_norm",
        "depth_proximity",
        "gripper_state",
        "planner_base_action_local_raw",
        "planner_base_action_7d_raw",
        "planner_base_action_local",
        "executed_action_local",
        "executed_action_7d",
        "stage_token",
        "phase_id",
        "substage_id",
        "contact_state",
        "planner_close_intent",
        "stage_target_mode",
        "episode_index",
        "step_index",
        "privileged_current_pose_7d",
        "privileged_motion_target_pose_7d",
        "privileged_basin_center_pose_7d",
        "privileged_pregrasp_target_pose_7d",
        "privileged_grasp_commit_target_pose_7d",
        "privileged_current_delta_basin_target",
        "privileged_target_provider_source",
        "privileged_target_provider_uses_privileged",
    ]:
        if key in support:
            out[key] = np.asarray(support[key])

    for key, value in labels.items():
        out[key] = np.asarray(value).astype(np.float32)

    out["proposal_actions_local"] = proposal_actions_arr.astype(np.float32)
    out["proposal_kind"] = proposal_kind_arr.astype("U32")
    out["proposal_family"] = proposal_family_arr.astype("U32")
    out["proposal_geometry_cost"] = geom.astype(np.float32)
    out["proposal_risk_cost"] = risk_cost.astype(np.float32)
    out["proposal_geometry_gain"] = best["geometry_gain"].astype(np.float32)
    out["proposal_risk_delta"] = best["risk_delta"].astype(np.float32)
    out["proposal_pareto_mask"] = best["safe_mask"].astype(np.float32)
    out["proposal_budget_mask"] = best["budget_mask"].astype(np.float32)
    out["proposal_baseline_index"] = best["baseline_index"].astype(np.int64)
    out["proposal_geom_top1_index"] = best["geom_top1_index"].astype(np.int64)
    out["proposal_best_safe_index"] = best["best_safe_index"].astype(np.int64)
    out["proposal_best_soft_index"] = best["best_soft_index"].astype(np.int64)
    out["proposal_best_budget_index"] = best["best_budget_index"].astype(np.int64)
    out["proposal_target_mode"] = np.asarray(target_mode, dtype="U32")
    out["proposal_target_source"] = np.asarray([target_source] * n, dtype="U64")
    out["proposal_target_delta_local"] = target_delta.astype(np.float32)
    out["proposal_safe_target_action_local"] = safe_target.astype(np.float32)
    out["proposal_safe_mask_rate"] = np.asarray(best["safe_mask"].mean(), dtype=np.float32)
    out["proposal_candidate_count"] = np.asarray(proposal_actions_arr.shape[1], dtype=np.int64)

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    eps = np.asarray(support.get("episode_index", np.zeros((n,), dtype=np.int64)), dtype=np.int64)
    yaw_aug = np.asarray(support.get("yaw_augmentation_applied", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    yaw_opp = np.asarray(support.get("yaw_opportunity_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    report = {
        "dataset_npz": args.support_npz,
        "risk_npz": args.risk_npz,
        "rows": n,
        "episodes": int(np.unique(eps).size),
        "candidate_count": int(proposal_actions_arr.shape[1]),
        "geo_margin": float(args.geo_margin),
        "risk_budget": float(args.risk_budget),
        "soft_alpha": float(args.soft_alpha),
        "proposal_family_hist": {
            k: int(v) for k, v in zip(*np.unique(proposal_family_arr.reshape(-1), return_counts=True))
        },
        "proposal_kind_hist": {
            k: int(v) for k, v in zip(*np.unique(proposal_kind_arr.reshape(-1), return_counts=True))
        },
        "groups": {},
        "episodes_by_id": {},
        "global": {
            "pareto_feasible_rate": float(np.mean(best["pareto_feasible"])),
            "missing_compromise_rate": float(np.mean(best["missing_compromise"])),
            "geom_top1_geometry_gain": summary_stats(best["geometry_gain"]),
            "geom_top1_risk_delta": summary_stats(best["risk_delta"]),
            "geom_top1_risk_increase_rate": float(np.mean(best["geom_top1_risk_increase"])),
            "geom_top1_safe_rate": float(np.mean(best["geom_top1_safe"])),
            "best_safe_is_baseline_rate": float(np.mean(best["best_safe_is_baseline"])),
            "best_safe_is_geom_top1_rate": float(np.mean(best["best_safe_is_geom_top1"])),
            "geom_top1_yaw_rate": float(np.mean(np.abs(proposal_actions_arr[np.arange(n), best["geom_top1_index"], 5]) > 0.02)),
            "best_safe_yaw_rate": float(np.mean(np.abs(proposal_actions_arr[np.arange(n), best["best_safe_index"], 5]) > 0.02)),
            "geom_top1_correct_yaw_sign_rate": float(
                np.mean(
                    np.asarray(
                        [
                            abs(float(target_delta[i, 5])) > 0.02
                            and ((proposal_actions_arr[i, best["geom_top1_index"][i], 5] > 0 and target_delta[i, 5] > 0)
                                 or (proposal_actions_arr[i, best["geom_top1_index"][i], 5] < 0 and target_delta[i, 5] < 0))
                            for i in range(n)
                        ],
                        dtype=np.float32,
                    )
                )
            ),
            "best_safe_correct_yaw_sign_rate": float(
                np.mean(
                    np.asarray(
                        [
                            abs(float(target_delta[i, 5])) > 0.02
                            and ((proposal_actions_arr[i, best["best_safe_index"][i], 5] > 0 and target_delta[i, 5] > 0)
                                 or (proposal_actions_arr[i, best["best_safe_index"][i], 5] < 0 and target_delta[i, 5] < 0))
                            for i in range(n)
                        ],
                        dtype=np.float32,
                    )
                )
            ),
        },
    }

    group_idx = {
        "all_rows": np.arange(n, dtype=np.int64),
        "original_rows": np.arange(n, dtype=np.int64)[~yaw_aug],
        "yaw_augmented_rows": np.arange(n, dtype=np.int64)[yaw_aug],
        "yaw_opportunity_rows": np.arange(n, dtype=np.int64)[yaw_opp],
        "non_yaw_rows": np.arange(n, dtype=np.int64)[~yaw_opp],
    }
    for gname, idx in group_idx.items():
        report["groups"][gname] = _group_summary(
            {**support, **{"candidate_actions_local": proposal_actions_arr}},
            labels,
            idx,
            geo_margin=float(args.geo_margin),
            risk_budget=float(args.risk_budget),
            soft_alpha=float(args.soft_alpha),
        )
    for ep in sorted(int(x) for x in np.unique(eps)):
        report["episodes_by_id"][str(ep)] = _group_summary(
            {**support, **{"candidate_actions_local": proposal_actions_arr}},
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
