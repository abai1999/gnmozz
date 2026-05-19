"""
audit_candidate_oracles.py

Method-level audit for candidate action sets and oracle objectives on a controlled
planner-only rollout.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from scripts.collect_residual_data import load_planner
from scripts.collect_stage_refiner_data import (
    build_reference_preclose_segment,
    choose_pregrasp_reset_index,
    compute_basin_center_pose,
    make_env,
    reconstruct_expert_action_7d,
    pose_delta_local_between,
)
from scripts.evaluate_rlbench import delta_to_absolute, predict_actions, process_obs
from scripts.build_pose_candidate_dataset import (
    build_local_perturb_offsets,
    build_action_primitives,
    apply_local_offset_to_pose,
    compute_basin_metrics,
    score_candidate_approach_funnel,
)


def candidate_offsets(xy_values, z_values, yaw_values, include_diagonals=True, include_combo=True):
    offsets = [np.zeros(6, dtype=np.float32)]
    offsets.extend(build_local_perturb_offsets(xy_values, z_values, yaw_values, include_diagonals=include_diagonals))
    if include_combo:
        seen = {tuple(np.round(x, 6).tolist()) for x in offsets}
        combo_xy = [float(max(xy_values))] if xy_values else []
        combo_yaw = [float(min(yaw_values))] if yaw_values else []
        for mag in combo_xy:
            for yaw in combo_yaw:
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        for syaw in (-1.0, 1.0):
                            arr = np.asarray([sx * mag, sy * mag, 0.0, 0.0, 0.0, syaw * yaw], dtype=np.float32)
                            key = tuple(np.round(arr, 6).tolist())
                            if key not in seen:
                                seen.add(key)
                                offsets.append(arr)
    return np.stack(offsets, axis=0).astype(np.float32)


def primitive_candidates():
    return np.stack(
        build_action_primitives(
            xy_small=0.004,
            xy_large=0.008,
            z_small=0.004,
            yaw_small=0.03,
            include_descend=True,
            include_combos=True,
        ),
        axis=0,
    ).astype(np.float32)


def one_step_oracle(current_pose_7d, basin_center_pose, candidates, r_xy, r_z, r_yaw):
    current_delta = pose_delta_local_between(current_pose_7d, basin_center_pose)
    current_dist, _, _, _ = compute_basin_metrics(current_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)
    best_idx = 0
    best_score = -1e9
    rows = []
    for i, cand in enumerate(candidates):
        next_pose = apply_local_offset_to_pose(current_pose_7d, cand)
        next_delta = pose_delta_local_between(next_pose, basin_center_pose)
        next_dist, _, _, _ = compute_basin_metrics(next_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)
        improve = float(current_dist - next_dist)
        in_basin = float(next_dist <= 1.0)
        score = improve + 0.5 * in_basin
        rows.append((i, float(next_dist), improve, score))
        if score > best_score:
            best_score = score
            best_idx = i
    return {
        "best_idx": int(best_idx),
        "current_dist": float(current_dist),
        "rows": rows,
    }


def one_step_funnel_oracle(
    current_pose_7d,
    basin_center_pose,
    reference_anchor_pose,
    candidates,
    base_action_local,
    depth_proximity,
    r_xy,
    r_z,
    r_yaw,
):
    current_delta = pose_delta_local_between(current_pose_7d, basin_center_pose)
    best_idx = 0
    best_score = -1e9
    rows = []
    for i, cand in enumerate(candidates):
        next_pose = apply_local_offset_to_pose(current_pose_7d, cand)
        next_delta = pose_delta_local_between(next_pose, basin_center_pose)
        next_dist, _, _, _ = compute_basin_metrics(next_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)
        improve = float(compute_basin_metrics(current_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)[0] - next_dist)
        score, details = score_candidate_approach_funnel(
            current_pose_7d=current_pose_7d,
            next_pose_7d=next_pose,
            current_delta=current_delta,
            next_delta=next_delta,
            candidate_local=cand,
            reference_anchor_pose_7d=reference_anchor_pose,
            base_action_local=base_action_local,
            depth_proximity=depth_proximity,
            r_xy=r_xy,
            r_z=r_z,
            r_yaw=r_yaw,
        )
        rows.append((i, float(next_dist), improve, float(score), details))
        if score > best_score:
            best_score = score
            best_idx = i
    return {
        "best_idx": int(best_idx),
        "current_dist": float(compute_basin_metrics(current_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)[0]),
        "rows": rows,
    }


def short_horizon_oracle(current_pose_7d, basin_center_pose, candidates, r_xy, r_z, r_yaw, horizon=2):
    current_delta = pose_delta_local_between(current_pose_7d, basin_center_pose)
    current_dist, _, _, _ = compute_basin_metrics(current_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)
    best_first = 0
    best_value = 1e9
    for i, cand in enumerate(candidates):
        pose1 = apply_local_offset_to_pose(current_pose_7d, cand)
        if horizon <= 1:
            delta1 = pose_delta_local_between(pose1, basin_center_pose)
            dist1, _, _, _ = compute_basin_metrics(delta1, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)
            value = float(dist1)
        else:
            value = 1e9
            for cand2 in candidates:
                pose2 = apply_local_offset_to_pose(pose1, cand2)
                delta2 = pose_delta_local_between(pose2, basin_center_pose)
                dist2, _, _, _ = compute_basin_metrics(delta2, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)
                value = min(value, float(dist2))
        if value < best_value:
            best_value = value
            best_first = i
    return {
        "best_idx": int(best_first),
        "current_dist": float(current_dist),
        "best_value": float(best_value),
    }


def short_horizon_funnel_oracle(
    current_pose_7d,
    basin_center_pose,
    reference_anchor_pose,
    candidates,
    base_action_local,
    depth_proximity,
    r_xy,
    r_z,
    r_yaw,
    horizon=4,
    gamma=0.9,
):
    best_first = 0
    best_value = -1e9
    for i, cand in enumerate(candidates):
        pose_t = np.asarray(current_pose_7d, dtype=np.float32).copy()
        delta_t = pose_delta_local_between(pose_t, basin_center_pose)
        cand_t = np.asarray(cand, dtype=np.float32).copy()
        value = 0.0
        for t in range(max(int(horizon), 1)):
            pose_next = apply_local_offset_to_pose(pose_t, cand_t)
            delta_next = pose_delta_local_between(pose_next, basin_center_pose)
            score_t, _ = score_candidate_approach_funnel(
                current_pose_7d=pose_t,
                next_pose_7d=pose_next,
                current_delta=delta_t,
                next_delta=delta_next,
                candidate_local=cand_t,
                reference_anchor_pose_7d=reference_anchor_pose,
                base_action_local=base_action_local,
                depth_proximity=depth_proximity,
                r_xy=r_xy,
                r_z=r_z,
                r_yaw=r_yaw,
            )
            value += (float(gamma) ** t) * float(score_t)
            pose_t = pose_next
            delta_t = delta_next
            if t + 1 < int(horizon):
                best_future = -1e9
                best_future_cand = candidates[0]
                for cand2 in candidates:
                    pose_future = apply_local_offset_to_pose(pose_t, cand2)
                    delta_future = pose_delta_local_between(pose_future, basin_center_pose)
                    score_future, _ = score_candidate_approach_funnel(
                        current_pose_7d=pose_t,
                        next_pose_7d=pose_future,
                        current_delta=delta_t,
                        next_delta=delta_future,
                        candidate_local=cand2,
                        reference_anchor_pose_7d=reference_anchor_pose,
                        base_action_local=base_action_local,
                        depth_proximity=depth_proximity,
                        r_xy=r_xy,
                        r_z=r_z,
                        r_yaw=r_yaw,
                    )
                    if score_future > best_future:
                        best_future = float(score_future)
                        best_future_cand = np.asarray(cand2, dtype=np.float32)
                cand_t = best_future_cand
        if value > best_value:
            best_value = value
            best_first = i
    return {
        "best_idx": int(best_first),
        "best_value": float(best_value),
    }


def hist_dict(values):
    uniq, cnt = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, cnt)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--episode_idx", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=45)
    parser.add_argument("--pregrasp_window", type=int, default=16)
    parser.add_argument("--planner_no_depth", action="store_true", default=True)
    parser.add_argument("--planner_no_force", action="store_true", default=True)
    parser.add_argument("--basin_center_mode", type=str, default="success_region_proxy")
    parser.add_argument("--success_region_window", type=int, default=8)
    parser.add_argument("--success_region_exclude_last", type=int, default=1)
    parser.add_argument("--support_basin_max", type=float, default=6.584)
    parser.add_argument("--horizon_k", type=int, default=4)
    parser.add_argument("--discount_gamma", type=float, default=0.9)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    base_candidates = candidate_offsets([0.004, 0.008], [], [0.03, 0.05], include_diagonals=True, include_combo=False)
    expanded_candidates = candidate_offsets([0.004, 0.008], [0.004], [0.03, 0.05], include_diagonals=True, include_combo=True)
    primitive_cands = primitive_candidates()

    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        args.checkpoint_dir,
        "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
        "pretrained_models/configs/config.json",
        use_depth=not args.planner_no_depth,
        use_force=not args.planner_no_force,
    )

    ep_path = Path(args.data_root) / args.task_name / "train" / "episodes" / f"episode{args.episode_idx}"
    npz_data = dict(np.load(ep_path / "model_inputs.npz"))
    with open(ep_path / "low_dim_obs.pkl", "rb") as f:
        demo_obs = pickle.load(f)

    rollout_offset = choose_pregrasp_reset_index(npz_data, window=args.pregrasp_window, open_threshold=0.5)
    reference_candidates, reference_close_idx, reference_anchor_idx = build_reference_preclose_segment(
        npz_data, start_idx=rollout_offset, open_threshold=0.5
    )
    basin_center_pose = compute_basin_center_pose(
        npz_data,
        reference_candidates,
        center_k=3,
        mode=args.basin_center_mode,
        close_idx=reference_close_idx,
        success_region_window=args.success_region_window,
        success_region_exclude_last=args.success_region_exclude_last,
    )
    anchor_pose = np.asarray(npz_data["gripper_pose"], dtype=np.float32)[reference_anchor_idx]

    env, task = make_env(args.task_name, Path(args.data_root))
    action_queue = []
    rows = []
    try:
        descs, obs = task.reset_to_demo(demo_obs)
        for i in range(rollout_offset):
            expert_action = reconstruct_expert_action_7d(npz_data, i)
            abs_action = delta_to_absolute(expert_action, obs.gripper_pose)
            obs, _, terminate = task.step(abs_action)
            if terminate:
                break

        instruction = descs[0] if isinstance(descs, list) else str(descs)
        force_buffer = []
        for step in range(args.max_steps):
            front_pil, wrist_pil, proprio, _, _, depth_tensor_96, _ = process_obs(
                obs, norm_stats, force_buffer, use_depth=True, use_force=False, depth_max=1.0
            )
            if not action_queue:
                actions = predict_actions(
                    vla, processor, action_head, proprio_projector,
                    front_pil, wrist_pil, proprio, None, None, instruction, unnorm_key="rlbench"
                )
                action_queue = [np.asarray(a, dtype=np.float32) for a in actions]
            _ = action_queue.pop(0)

            delta_local = pose_delta_local_between(obs.gripper_pose, basin_center_pose)
            basin_distance, e_xy, e_z, e_yaw = compute_basin_metrics(delta_local, r_xy=0.008, r_z=0.01, r_yaw=0.05)
            in_support = bool(basin_distance <= float(args.support_basin_max))

            one_base = one_step_oracle(obs.gripper_pose, basin_center_pose, base_candidates, 0.008, 0.01, 0.05)
            one_exp = one_step_oracle(obs.gripper_pose, basin_center_pose, expanded_candidates, 0.008, 0.01, 0.05)
            sh_base = short_horizon_oracle(obs.gripper_pose, basin_center_pose, base_candidates, 0.008, 0.01, 0.05, horizon=2)
            sh_exp = short_horizon_oracle(obs.gripper_pose, basin_center_pose, expanded_candidates, 0.008, 0.01, 0.05, horizon=2)
            depth_prox = None
            if depth_tensor_96 is not None:
                depth_arr = np.asarray(depth_tensor_96, dtype=np.float32).reshape(-1)
                depth_arr = depth_arr[np.isfinite(depth_arr)]
                if depth_arr.size > 0:
                    depth_prox = float(np.percentile(depth_arr, 5.0))
            current_base_local = action_queue[0][:6] if action_queue else np.zeros(6, dtype=np.float32)
            funnel_one = one_step_funnel_oracle(
                obs.gripper_pose, basin_center_pose, anchor_pose, primitive_cands, current_base_local, depth_prox, 0.008, 0.01, 0.05
            )
            funnel_short = short_horizon_funnel_oracle(
                obs.gripper_pose, basin_center_pose, anchor_pose, primitive_cands, current_base_local, depth_prox, 0.008, 0.01, 0.05, horizon=args.horizon_k, gamma=args.discount_gamma
            )

            rows.append(
                {
                    "step": int(step),
                    "basin_distance": float(basin_distance),
                    "e_xy": float(e_xy),
                    "e_z": float(e_z),
                    "e_yaw": float(e_yaw),
                    "in_support": in_support,
                    "one_step_base_idx": int(one_base["best_idx"]),
                    "one_step_exp_idx": int(one_exp["best_idx"]),
                    "short_base_idx": int(sh_base["best_idx"]),
                    "short_exp_idx": int(sh_exp["best_idx"]),
                    "funnel_one_idx": int(funnel_one["best_idx"]),
                    "funnel_short_idx": int(funnel_short["best_idx"]),
                    "depth_proximity": None if depth_prox is None else float(depth_prox),
                }
            )

            next_action = reconstruct_expert_action_7d(npz_data, rollout_offset + step) if (rollout_offset + step) < len(npz_data["action_targets"]) else None
            if next_action is None:
                break
            abs_action = delta_to_absolute(next_action, obs.gripper_pose)
            obs, reward, terminate = task.step(abs_action)
            if terminate or reward > 0:
                break
    finally:
        env.shutdown()

    support_rows = [r for r in rows if r["in_support"]]
    def disagree(a, b, subset):
        if not subset:
            return 0.0
        return float(sum(int(r[a] != r[b]) for r in subset) / len(subset))

    out = {
        "episode_idx": int(args.episode_idx),
        "reference_close_idx": int(reference_close_idx),
        "reference_anchor_idx": int(reference_anchor_idx),
        "num_rows": int(len(rows)),
        "num_support_rows": int(len(support_rows)),
        "candidate_counts": {
            "base": int(len(base_candidates)),
            "expanded": int(len(expanded_candidates)),
            "primitive": int(len(primitive_cands)),
        },
        "oracle_config": {
            "horizon_k": int(args.horizon_k),
            "discount_gamma": float(args.discount_gamma),
            "funnel_axis": "demo_anchor_axis",
        },
        "all_rows_disagreement": {
            "base_one_vs_short": disagree("one_step_base_idx", "short_base_idx", rows),
            "funnel_one_vs_short": disagree("funnel_one_idx", "funnel_short_idx", rows),
            "base_one_vs_funnel_short": disagree("one_step_base_idx", "funnel_short_idx", rows),
            "exp_one_vs_funnel_short": disagree("one_step_exp_idx", "funnel_short_idx", rows),
        },
        "support_rows_disagreement": {
            "base_one_vs_short": disagree("one_step_base_idx", "short_base_idx", support_rows),
            "funnel_one_vs_short": disagree("funnel_one_idx", "funnel_short_idx", support_rows),
            "base_one_vs_funnel_short": disagree("one_step_base_idx", "funnel_short_idx", support_rows),
            "exp_one_vs_funnel_short": disagree("one_step_exp_idx", "funnel_short_idx", support_rows),
        },
        "support_histograms": {
            "one_step_base": hist_dict([r["one_step_base_idx"] for r in support_rows]),
            "short_base": hist_dict([r["short_base_idx"] for r in support_rows]),
            "one_step_exp": hist_dict([r["one_step_exp_idx"] for r in support_rows]),
            "short_exp": hist_dict([r["short_exp_idx"] for r in support_rows]),
            "funnel_one": hist_dict([r["funnel_one_idx"] for r in support_rows]),
            "funnel_short": hist_dict([r["funnel_short_idx"] for r in support_rows]),
        },
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
