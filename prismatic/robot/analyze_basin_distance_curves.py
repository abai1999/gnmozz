"""
analyze_basin_distance_curves.py

Replay a controlled pre-grasp reset rollout and compare planner-only vs
pose-alignment runtime on distances to the current reference anchor / basin
center proxy.
"""

import argparse
import json
import pickle
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from prismatic.robot.residual_safety import ResidualSafety
from prismatic.robot.stage_manager import StagePhase
from scripts.collect_residual_data import load_planner
from scripts.collect_stage_refiner_data import (
    build_reference_preclose_segment,
    choose_pregrasp_reset_index,
    compute_basin_metrics,
    compute_basin_center_pose,
    make_env,
    make_gripper_context,
    match_reference_index,
    pose_delta_local_between,
    reconstruct_expert_action_7d,
)
from scripts.build_pose_candidate_dataset import (
    basin_distance_bin,
    improvement_tiers,
    apply_local_offset_to_pose,
    score_candidate_approach_funnel,
    compute_funnel_cost,
)
from scripts.evaluate_rlbench import (
    _load_optional_controller,
    delta_to_absolute,
    predict_actions,
    process_obs,
)
from prismatic.robot.stage_aware_refiner import StageAwareRefiner


def sign_bucket(val: float, eps: float) -> int:
    if val > eps:
        return 1
    if val < -eps:
        return -1
    return 0


def pose_distance_metrics(current_pose_7d, target_pose_7d, r_xy=0.008, r_z=0.01, r_yaw=0.05):
    current = np.asarray(current_pose_7d, dtype=np.float32)
    target = np.asarray(target_pose_7d, dtype=np.float32)
    dpos = target[:3] - current[:3]
    xy = float(np.linalg.norm(dpos[:2]))
    z = float(abs(dpos[2]))
    # lightweight yaw proxy: quat diff converted to rotvec, keep z
    from scipy.spatial.transform import Rotation

    r_cur = Rotation.from_quat(current[3:7])
    r_tgt = Rotation.from_quat(target[3:7])
    drot = (r_tgt * r_cur.inv()).as_rotvec().astype(np.float32)
    yaw = float(abs(drot[2]))
    basin_distance = float(max(xy / max(r_xy, 1e-8), z / max(r_z, 1e-8), yaw / max(r_yaw, 1e-8)))
    return {
        "xy": xy,
        "z": z,
        "yaw": yaw,
        "basin_distance": basin_distance,
    }


def summarize_curve(values):
    arr = np.asarray(values, dtype=np.float32)
    return {
        "start": float(arr[0]) if arr.size else None,
        "min": float(arr.min()) if arr.size else None,
        "final": float(arr[-1]) if arr.size else None,
        "p50": float(np.median(arr)) if arr.size else None,
    }


def scorer_oracle(
    current_pose_7d,
    basin_center_pose,
    reference_anchor_pose,
    candidate_actions_local,
    base_action_local,
    depth_proximity,
    planner_close_intent=True,
    r_xy=0.008,
    r_z=0.01,
    r_yaw=0.05,
    horizon_k=4,
    gamma=0.9,
    no_intent_hold_basin_distance=1.2,
    no_intent_hold_abs_z_threshold=0.025,
    no_intent_noop_bonus=8.0,
    no_intent_motion_penalty=4.0,
    primitive_xy_small=0.004,
):
    best_idx = -1
    best_score = -1e9
    current_delta = pose_delta_local_between(current_pose_7d, basin_center_pose)
    current_dist, _, _, _ = compute_basin_metrics(current_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)
    current_axis = pose_delta_local_between(current_pose_7d, reference_anchor_pose)
    current_funnel_cost, funnel_terms = compute_funnel_cost(
        current_delta,
        current_axis,
        r_xy=r_xy,
        r_z=r_z,
        r_yaw=r_yaw,
    )
    next_dists = []
    oracle_scores = []
    for j, cand in enumerate(candidate_actions_local):
        pose_t = np.asarray(current_pose_7d, dtype=np.float32).copy()
        delta_t = current_delta.copy()
        cand_t = np.asarray(cand, dtype=np.float32).copy()
        total_score = 0.0
        next_dist = current_dist
        for t in range(max(int(horizon_k), 1)):
            pose_next = apply_local_offset_to_pose(pose_t, cand_t)
            delta_next = pose_delta_local_between(pose_next, basin_center_pose)
            next_dist, _, _, _ = compute_basin_metrics(delta_next, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw)
            step_score, _ = score_candidate_approach_funnel(
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
            total_score += (float(gamma) ** t) * float(step_score)
            pose_t = pose_next
            delta_t = delta_next
            if t + 1 < int(horizon_k):
                best_future_score = -1e9
                best_future_cand = candidate_actions_local[0]
                for cand2 in candidate_actions_local:
                    pose_future = apply_local_offset_to_pose(pose_t, cand2)
                    delta_future = pose_delta_local_between(pose_future, basin_center_pose)
                    future_score, _ = score_candidate_approach_funnel(
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
                    if future_score > best_future_score:
                        best_future_score = float(future_score)
                        best_future_cand = np.asarray(cand2, dtype=np.float32)
                cand_t = best_future_cand
        improve = float(current_dist - next_dist)
        if (
            not bool(planner_close_intent)
            and (
                current_dist <= float(no_intent_hold_basin_distance)
                or abs(float(current_delta[2])) >= float(no_intent_hold_abs_z_threshold)
            )
        ):
            cand_norm = float(np.linalg.norm(np.asarray(cand, dtype=np.float32).reshape(6)))
            if cand_norm < 1e-8:
                total_score += float(no_intent_noop_bonus)
            else:
                total_score -= float(no_intent_motion_penalty) * cand_norm / max(float(primitive_xy_small), 1e-6)
        next_dists.append(float(next_dist))
        oracle_scores.append(float(total_score))
        if total_score > best_score:
            best_score = float(total_score)
            best_idx = j
    return best_idx, float(best_score), float(current_dist), next_dists, oracle_scores, float(current_funnel_cost), funnel_terms


def scorer_forward_debug(controller, wrist_depth, proprio, base_action_local, step_idx, gripper_context):
    device = next(controller.parameters()).device
    dtype = next(controller.parameters()).dtype
    wd = torch.from_numpy(np.asarray(wrist_depth, dtype=np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
    pr = torch.from_numpy(np.asarray(proprio, dtype=np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
    ba = torch.from_numpy(np.asarray(base_action_local, dtype=np.float32)).reshape(1, 6).to(device=device, dtype=dtype)
    gc = torch.from_numpy(np.asarray(gripper_context, dtype=np.float32)).reshape(1, 3).to(device=device, dtype=dtype)
    si = torch.tensor([int(step_idx)], device=device, dtype=torch.long)
    phase_id = torch.tensor([1], device=device, dtype=torch.long)
    phase_age = torch.tensor([0.0], device=device, dtype=dtype)
    since_replan = torch.tensor([0.0], device=device, dtype=dtype)
    current_delta = np.asarray(getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)), dtype=np.float32)
    basin_distance = float(getattr(controller, "_runtime_current_basin_distance", 3.0))
    if basin_distance <= 0.9:
        basin_bin = 0
    elif basin_distance <= 1.05:
        basin_bin = 1
    elif basin_distance <= 1.2:
        basin_bin = 2
    else:
        basin_bin = 3
    cur_delta_t = torch.from_numpy(current_delta).reshape(1, 6).to(device=device, dtype=dtype)
    dxs = torch.tensor([sign_bucket(float(current_delta[0]), 1e-4)], device=device, dtype=torch.long)
    dys = torch.tensor([sign_bucket(float(current_delta[1]), 1e-4)], device=device, dtype=torch.long)
    dyaws = torch.tensor([sign_bucket(float(current_delta[5]), 1e-3)], device=device, dtype=torch.long)
    basin_bin_t = torch.tensor([basin_bin], device=device, dtype=torch.long)
    candidate_actions = controller._candidate_actions_local.unsqueeze(0).to(device=device, dtype=dtype)
    candidate_group_index = controller._candidate_group_index.unsqueeze(0).to(device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = controller(
            wd,
            pr,
            ba,
            gc,
            si,
            candidate_actions,
            phase_id=phase_id,
            phase_age=phase_age,
            steps_since_last_replan=since_replan,
            current_delta_basin_target=cur_delta_t,
            current_dx_sign=dxs,
            current_dy_sign=dys,
            current_dyaw_sign=dyaws,
            basin_distance_bin=basin_bin_t,
            return_aux=True,
        )
        scores = outputs["candidate_scores"].float()
        group_logits = outputs["group_logits"].float()
        pred_group = group_logits.argmax(dim=-1)
        group_mask = candidate_group_index.eq(pred_group.unsqueeze(1))
        pred = scores.masked_fill(~group_mask, -1e9).argmax(dim=-1)
        probs = torch.softmax(scores, dim=-1).squeeze(0).cpu().numpy()
        group_probs = torch.softmax(group_logits, dim=-1).squeeze(0).cpu().numpy()
    topk_idx = np.argsort(-probs)[:5].astype(np.int64)
    return {
        "candidate_scores": scores.squeeze(0).cpu().numpy().astype(np.float32),
        "candidate_probs": probs.astype(np.float32),
        "group_logits": group_logits.squeeze(0).cpu().numpy().astype(np.float32),
        "group_probs": group_probs.astype(np.float32),
        "pred_candidate_index": int(pred.item()),
        "pred_group_index": int(pred_group.item()),
        "topk_candidate_index": topk_idx,
        "topk_candidate_prob": probs[topk_idx].astype(np.float32),
        "input_current_dx_sign": int(dxs.item()),
        "input_current_dy_sign": int(dys.item()),
        "input_current_dyaw_sign": int(dyaws.item()),
        "input_basin_distance_bin": int(basin_bin),
    }


def run_mode(
    mode_name,
    checkpoint_dir,
    alignment_ckpt,
    close_trigger_ckpt,
    task_name,
    data_root,
    episode_idx,
    max_steps,
    pregrasp_window,
    planner_no_depth,
    planner_no_force,
    basin_center_mode,
    success_region_window,
    success_region_exclude_last,
    hard_mining_rows=None,
    support_state_rows=None,
    trigger_state_rows=None,
    save_frames=False,
):
    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        checkpoint_dir,
        "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
        "pretrained_models/configs/config.json",
        use_depth=not planner_no_depth,
        use_force=not planner_no_force,
    )

    episodes_dir = Path(data_root) / task_name / "train" / "episodes"
    ep_name = f"episode{episode_idx}"
    ep_path = episodes_dir / ep_name
    npz_data = dict(np.load(ep_path / "model_inputs.npz"))
    with open(ep_path / "low_dim_obs.pkl", "rb") as f:
        demo_obs = pickle.load(f)

    rollout_offset = choose_pregrasp_reset_index(npz_data, window=pregrasp_window, open_threshold=0.5)
    reference_candidates, reference_close_idx, reference_anchor_idx = build_reference_preclose_segment(
        npz_data,
        start_idx=rollout_offset,
        open_threshold=0.5,
    )
    basin_center_pose = compute_basin_center_pose(
        npz_data,
        reference_candidates,
        center_k=3,
        mode=basin_center_mode,
        close_idx=reference_close_idx,
        success_region_window=success_region_window,
        success_region_exclude_last=success_region_exclude_last,
    )
    anchor_pose = np.asarray(npz_data["gripper_pose"], dtype=np.float32)[reference_anchor_idx]

    env, task = make_env(task_name, Path(data_root))
    refiner = None
    safety = ResidualSafety(max_residual_pos=0.01, max_residual_rot=0.03)
    if mode_name == "pose_alignment_only_basin":
        controller = _load_optional_controller(alignment_ckpt)
        close_trigger_controller = _load_optional_controller(close_trigger_ckpt) if close_trigger_ckpt else None
        refiner = StageAwareRefiner(
            mode="alignment",
            alignment_controller=controller,
            close_trigger_controller=close_trigger_controller,
            max_residual_pos=0.01,
            max_residual_rot=0.03,
            learned_residual_scale=1.0,
            max_alignment_corrections_per_window=20,
            require_close_intent_for_alignment=False,
            enable_alignment_pose=True,
            use_pose_alpha=False,
            enable_readiness_gripper=bool(close_trigger_controller),
        )
    try:
        descs, obs = task.reset_to_demo(demo_obs)
        for i in range(rollout_offset):
            expert_action = reconstruct_expert_action_7d(npz_data, i)
            abs_action = delta_to_absolute(expert_action, obs.gripper_pose)
            obs, _, terminate = task.step(abs_action)
            if terminate:
                break

        instruction = descs[0] if isinstance(descs, list) else str(descs)
        action_queue = []
        chunk_step = 0
        force_buffer = deque(maxlen=256)
        curves = []
        frames = []
        close_before_basin_count = 0
        for step_idx in range(max_steps):
            if save_frames:
                frames.append(np.asarray(obs.front_rgb, dtype=np.uint8))
            metrics_anchor = pose_distance_metrics(obs.gripper_pose, anchor_pose)
            metrics_basin = pose_distance_metrics(obs.gripper_pose, basin_center_pose)
            row = {
                "step": int(step_idx),
                "anchor_xy": metrics_anchor["xy"],
                "anchor_z": metrics_anchor["z"],
                "anchor_yaw": metrics_anchor["yaw"],
                "basin_xy": metrics_basin["xy"],
                "basin_z": metrics_basin["z"],
                "basin_yaw": metrics_basin["yaw"],
                "basin_distance": metrics_basin["basin_distance"],
                "gripper_open": float(obs.gripper_open),
            }

            front_pil, wrist_pil, proprio, depth_tensor, force_hist, depth_tensor_96, raw_force = process_obs(
                obs,
                norm_stats,
                force_buffer,
                use_depth=True,
                use_force=False,
                depth_max=1.0,
            )

            if len(action_queue) == 0:
                actions = predict_actions(
                    vla,
                    processor,
                    action_head,
                    proprio_projector,
                    front_pil,
                    wrist_pil,
                    proprio,
                    None,
                    None,
                    instruction,
                    unnorm_key="rlbench",
                )
                max_chunk = refiner.get_chunk_size() if refiner is not None else len(actions)
                action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), max_chunk))]
                chunk_step = 0

            delta_action = action_queue.pop(0)
            base_delta_action = np.asarray(delta_action, dtype=np.float32).copy()
            planner_nominal_pose = delta_to_absolute(delta_action, obs.gripper_pose)
            planner_nominal_metrics = pose_distance_metrics(planner_nominal_pose, basin_center_pose)
            row["planner_nominal_basin_xy"] = planner_nominal_metrics["xy"]
            row["planner_nominal_basin_z"] = planner_nominal_metrics["z"]
            row["planner_nominal_basin_yaw"] = planner_nominal_metrics["yaw"]
            row["planner_nominal_basin_distance"] = planner_nominal_metrics["basin_distance"]
            future_action_queue = action_queue[:4]
            gripper_context = make_gripper_context(delta_action, future_action_queue)
            planner_close_intent = bool(float(np.min(gripper_context[:2])) <= 0.5)
            row["planner_close_intent"] = planner_close_intent
            depth_prox = None
            if depth_tensor_96 is not None:
                depth_arr = np.asarray(depth_tensor_96, dtype=np.float32).reshape(-1)
                depth_arr = depth_arr[np.isfinite(depth_arr)]
                if depth_arr.size > 0:
                    depth_prox = float(np.percentile(depth_arr, 5.0))
            if trigger_state_rows is not None and float(obs.gripper_open) >= 0.5:
                current_delta_trigger = pose_delta_local_between(obs.gripper_pose, basin_center_pose).astype(np.float32)
                trigger_state_rows.append(
                    {
                        "mode_index": np.asarray(0 if mode_name == "planner_only" else 1, dtype=np.int64),
                        "wrist_depth": depth_tensor_96.detach().cpu().numpy().astype(np.float32),
                        "proprio": np.asarray(proprio, dtype=np.float32),
                        "base_action": np.asarray(base_delta_action[:6], dtype=np.float32),
                        "gripper_context": np.asarray(gripper_context, dtype=np.float32),
                        "rollout_gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
                        "planner_close_intent": np.asarray(float(planner_close_intent), dtype=np.float32),
                        "depth_proximity": np.asarray(float(depth_prox) if depth_prox is not None else np.nan, dtype=np.float32),
                        "wrist_depth_median": np.asarray(
                            float(np.median(depth_tensor_96.detach().cpu().numpy())) if depth_tensor_96 is not None else np.nan,
                            dtype=np.float32,
                        ),
                        "step_idx": np.asarray(int(step_idx), dtype=np.int64),
                        "phase_id": np.asarray(1, dtype=np.int64),
                        "phase_age": np.asarray(float(step_idx), dtype=np.float32),
                        "steps_since_last_replan": np.asarray(float(chunk_step), dtype=np.float32),
                        "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                        "basin_center_pose_7d": np.asarray(basin_center_pose, dtype=np.float32),
                        "reference_anchor_pose_7d": np.asarray(anchor_pose, dtype=np.float32),
                        "current_delta_basin_target": current_delta_trigger,
                        "current_basin_distance": np.asarray(float(metrics_basin["basin_distance"]), dtype=np.float32),
                        "current_basin_xy": np.asarray(float(metrics_basin["xy"]), dtype=np.float32),
                        "current_basin_z": np.asarray(float(metrics_basin["z"]), dtype=np.float32),
                        "current_basin_yaw": np.asarray(float(metrics_basin["yaw"]), dtype=np.float32),
                        "current_dx_sign": np.asarray(int(sign_bucket(float(current_delta_trigger[0]), 1e-4)), dtype=np.int64),
                        "current_dy_sign": np.asarray(int(sign_bucket(float(current_delta_trigger[1]), 1e-4)), dtype=np.int64),
                        "current_dyaw_sign": np.asarray(int(sign_bucket(float(current_delta_trigger[5]), 1e-3)), dtype=np.int64),
                        "basin_distance_bin": np.asarray(int(basin_distance_bin(float(metrics_basin["basin_distance"]))), dtype=np.int64),
                    }
                )
            if refiner is not None:
                if getattr(refiner.alignment_controller, "_controller_type", "") == "pose_field_scorer":
                    candidate_actions = refiner.alignment_controller._candidate_actions_local.cpu().numpy()
                    matched_ref_idx = match_reference_index(
                        npz_data,
                        obs.gripper_pose,
                        reference_candidates,
                        rot_weight=0.05,
                    )
                    delta_local = pose_delta_local_between(obs.gripper_pose, basin_center_pose)
                    basin_distance, _, _, _ = compute_basin_metrics(
                        delta_local,
                        r_xy=0.008,
                        r_z=0.01,
                        r_yaw=0.05,
                    )
                    current_axis = pose_delta_local_between(obs.gripper_pose, anchor_pose)
                    current_funnel_cost, funnel_terms = compute_funnel_cost(
                        delta_local,
                        current_axis,
                        r_xy=0.008,
                        r_z=0.01,
                        r_yaw=0.05,
                    )
                    refiner.alignment_controller._runtime_current_delta_basin_target = delta_local.astype(np.float32)
                    refiner.alignment_controller._runtime_current_basin_distance = float(basin_distance)
                    oracle_idx, oracle_score, oracle_current_dist, oracle_next_dists, oracle_scores, current_funnel_cost_value, current_funnel_terms = scorer_oracle(
                        obs.gripper_pose,
                        basin_center_pose,
                        anchor_pose,
                        candidate_actions,
                        np.asarray(delta_action[:6], dtype=np.float32),
                        depth_prox,
                        planner_close_intent=planner_close_intent,
                    )
                    row["matched_ref_idx"] = int(matched_ref_idx)
                    row["runtime_delta_basin_target"] = delta_local.tolist()
                    row["runtime_basin_distance"] = float(basin_distance)
                    row["runtime_funnel_cost"] = float(current_funnel_cost_value)
                    row["runtime_funnel_terms"] = current_funnel_terms
                    row["oracle_candidate_index"] = int(oracle_idx)
                    row["oracle_improvement"] = float(oracle_current_dist - oracle_next_dists[oracle_idx])
                    row["oracle_score"] = float(oracle_score)
                    row["oracle_current_basin_distance"] = float(oracle_current_dist)
                    row["input_current_dx_sign"] = int(sign_bucket(float(delta_local[0]), 1e-4))
                    row["input_current_dy_sign"] = int(sign_bucket(float(delta_local[1]), 1e-4))
                    row["input_current_dyaw_sign"] = int(sign_bucket(float(delta_local[5]), 1e-3))
                    row["input_basin_distance_bin"] = int(basin_distance_bin(float(basin_distance)))
                    debug_pred = scorer_forward_debug(
                        refiner.alignment_controller,
                        depth_tensor_96.detach().cpu().numpy().astype(np.float32),
                        proprio,
                        delta_action[:6],
                        chunk_step,
                        gripper_context,
                    )
                    row["debug_pred_candidate_index"] = int(debug_pred["pred_candidate_index"])
                    row["debug_pred_group_index"] = int(debug_pred["pred_group_index"])
                    row["debug_topk_candidate_index"] = debug_pred["topk_candidate_index"].tolist()
                    row["debug_topk_candidate_prob"] = debug_pred["topk_candidate_prob"].tolist()
                delta_action = refiner.step(
                    a_base_7d=delta_action,
                    step_idx=chunk_step,
                    force_reading=raw_force,
                    gripper_z=float(obs.gripper_pose[2]),
                    wrist_depth=depth_tensor_96,
                    ft_hist=force_hist,
                    proprio=proprio,
                    gripper_pose=obs.gripper_pose,
                    gripper_open=float(obs.gripper_open),
                    future_gripper_actions=[float(a[6]) for a in future_action_queue],
                )
                stats = refiner.get_stats()
                row["alignment_gate_open"] = bool(stats.get("current_alignment_gate_open", False))
                row["alignment_blocked_reason"] = str(stats.get("current_alignment_blocked_reason", "unknown"))
                row["executed_delta_norm_mean_so_far"] = float(stats.get("executed_delta_norm_mean", 0.0))
                row["correction_count_so_far"] = int(stats.get("alignment_correction_count", 0))
                row["trigger_prob"] = float(stats.get("current_trigger_prob", 0.0))
                row["trigger_basin_positive"] = float(stats.get("current_basin_positive", 0.0))
                row["ready_positive_rate_so_far"] = float(stats.get("ready_positive_rate", 0.0))
                row["ready_to_close_prob_mean_so_far"] = float(stats.get("ready_to_close_prob_mean", 0.0))
                row["readiness_eval_count_so_far"] = int(stats.get("readiness_eval_count", 0))
                row["readiness_close_override_count_so_far"] = int(stats.get("readiness_close_override_count", 0))
                row["readiness_open_override_count_so_far"] = int(stats.get("readiness_open_override_count", 0))
                row["readiness_hold_override_count_so_far"] = int(stats.get("readiness_hold_override_count", 0))
                row["readiness_heads_missing_count_so_far"] = int(stats.get("readiness_heads_missing_count", 0))
                row["last_scorer_candidate_index"] = int(stats.get("last_scorer_candidate_index", -1))
                row["last_scorer_group_index"] = int(stats.get("last_scorer_group_index", -1))
                row["scorer_pred_hist"] = stats.get("scorer_pred_hist", {})
                row["scorer_group_hist"] = stats.get("scorer_group_hist", {})
                if "oracle_candidate_index" in row:
                    runtime_selected_idx = int(row["last_scorer_candidate_index"])
                    row["runtime_selected_candidate_index"] = runtime_selected_idx
                    row["selected_matches_oracle"] = bool(runtime_selected_idx == row["oracle_candidate_index"])
                    dump_support_state = (
                        support_state_rows is not None
                        and int(stats.get("phase_id", 1)) == int(StagePhase.ALIGN)
                        and float(obs.gripper_open) >= 0.5
                    )
                    if dump_support_state:
                        current_delta = np.asarray(row["runtime_delta_basin_target"], dtype=np.float32)
                        support_state_rows.append(
                            {
                                "wrist_depth": depth_tensor_96.detach().cpu().numpy().astype(np.float32),
                                "proprio": np.asarray(proprio, dtype=np.float32),
                                "base_action": np.asarray(base_delta_action[:6], dtype=np.float32),
                                "gripper_context": np.asarray(gripper_context, dtype=np.float32),
                                "rollout_gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
                                "planner_close_intent": np.asarray(float(planner_close_intent), dtype=np.float32),
                                "depth_proximity": np.asarray(float(depth_prox) if depth_prox is not None else np.nan, dtype=np.float32),
                                "wrist_depth_median": np.asarray(float(np.median(depth_tensor_96.detach().cpu().numpy())) if depth_tensor_96 is not None else np.nan, dtype=np.float32),
                                "step_idx": int(chunk_step),
                                "phase_id": int(stats.get("phase_id", 1)),
                                "phase_age": float(stats.get("phase_age", 0.0)),
                                "steps_since_last_replan": float(stats.get("steps_since_last_replan", 0.0)),
                                "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                                "basin_center_pose_7d": np.asarray(basin_center_pose, dtype=np.float32),
                                "reference_anchor_pose_7d": np.asarray(anchor_pose, dtype=np.float32),
                                "current_delta_basin_target": current_delta,
                                "current_basin_distance": float(row["runtime_basin_distance"]),
                                "current_dx_sign": int(row["input_current_dx_sign"]),
                                "current_dy_sign": int(row["input_current_dy_sign"]),
                                "current_dyaw_sign": int(row["input_current_dyaw_sign"]),
                                "basin_distance_bin": int(row["input_basin_distance_bin"]),
                                "candidate_actions_local": np.asarray(candidate_actions, dtype=np.float32),
                                "candidate_group_index": np.asarray(
                                    refiner.alignment_controller._candidate_group_index.cpu().numpy(), dtype=np.int64
                                ),
                                "candidate_next_basin_distance": np.asarray(oracle_next_dists, dtype=np.float32),
                                "candidate_improvement": np.asarray([float(row["runtime_basin_distance"]) - float(x) for x in oracle_next_dists], dtype=np.float32),
                                "candidate_oracle_score": np.asarray(oracle_scores, dtype=np.float32),
                                "candidate_basin_positive": (np.asarray(oracle_next_dists, dtype=np.float32) <= 1.0).astype(np.float32),
                                "candidate_tier": improvement_tiers(
                                    np.asarray(oracle_scores, dtype=np.float32),
                                    (np.asarray(oracle_next_dists, dtype=np.float32) <= 1.0).astype(np.float32),
                                ).astype(np.int64),
                                "best_candidate_index": int(row["oracle_candidate_index"]),
                                "best_group_index": int(
                                    refiner.alignment_controller._candidate_group_index.cpu().numpy()[row["oracle_candidate_index"]]
                                ),
                                "support_source_index": int(matched_ref_idx),
                                "runtime_selected_candidate_index": int(runtime_selected_idx),
                                "runtime_selected_group_index": int(row["last_scorer_group_index"]),
                                "runtime_selected_matches_oracle": np.asarray(
                                    float(runtime_selected_idx == row["oracle_candidate_index"]), dtype=np.float32
                                ),
                                "pred_candidate_index": int(debug_pred["pred_candidate_index"]),
                                "pred_group_index": int(debug_pred["pred_group_index"]),
                                "topk_candidate_index": np.asarray(debug_pred["topk_candidate_index"], dtype=np.int64),
                                "topk_candidate_prob": np.asarray(debug_pred["topk_candidate_prob"], dtype=np.float32),
                                "candidate_scores": np.asarray(debug_pred["candidate_scores"], dtype=np.float32),
                                "candidate_probs": np.asarray(debug_pred["candidate_probs"], dtype=np.float32),
                                "group_probs": np.asarray(debug_pred["group_probs"], dtype=np.float32),
                            }
                        )
                    if (
                        hard_mining_rows is not None
                        and row["alignment_gate_open"]
                        and row["oracle_score"] > 1e-6
                        and runtime_selected_idx >= 0
                        and not row["selected_matches_oracle"]
                    ):
                        current_delta = np.asarray(row["runtime_delta_basin_target"], dtype=np.float32)
                        hard_mining_rows.append(
                            {
                                "wrist_depth": depth_tensor_96.detach().cpu().numpy().astype(np.float32),
                                "proprio": np.asarray(proprio, dtype=np.float32),
                                "base_action": np.asarray(base_delta_action[:6], dtype=np.float32),
                                "gripper_context": np.asarray(gripper_context, dtype=np.float32),
                                "rollout_gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
                                "planner_close_intent": np.asarray(float(planner_close_intent), dtype=np.float32),
                                "depth_proximity": np.asarray(float(depth_prox) if depth_prox is not None else np.nan, dtype=np.float32),
                                "wrist_depth_median": np.asarray(float(np.median(depth_tensor_96.detach().cpu().numpy())) if depth_tensor_96 is not None else np.nan, dtype=np.float32),
                                "step_idx": int(chunk_step),
                                "phase_id": int(stats.get("phase_id", 0)),
                                "phase_age": float(stats.get("phase_age", 0.0)),
                                "steps_since_last_replan": float(stats.get("steps_since_last_replan", 0.0)),
                                "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                                "basin_center_pose_7d": np.asarray(basin_center_pose, dtype=np.float32),
                                "reference_anchor_pose_7d": np.asarray(anchor_pose, dtype=np.float32),
                                "current_delta_basin_target": current_delta,
                                "current_basin_distance": float(row["runtime_basin_distance"]),
                                "current_dx_sign": int(np.sign(current_delta[0])) if abs(float(current_delta[0])) > 1e-4 else 0,
                                "current_dy_sign": int(np.sign(current_delta[1])) if abs(float(current_delta[1])) > 1e-4 else 0,
                                "current_dyaw_sign": int(np.sign(current_delta[5])) if abs(float(current_delta[5])) > 1e-3 else 0,
                                "basin_distance_bin": int(basin_distance_bin(float(row["runtime_basin_distance"]))),
                                "candidate_actions_local": np.asarray(candidate_actions, dtype=np.float32),
                                "candidate_group_index": np.asarray(
                                    refiner.alignment_controller._candidate_group_index.cpu().numpy(), dtype=np.int64
                                ),
                                "candidate_next_basin_distance": np.asarray(oracle_next_dists, dtype=np.float32),
                                "candidate_improvement": np.asarray([float(row["runtime_basin_distance"]) - float(x) for x in oracle_next_dists], dtype=np.float32),
                                "candidate_oracle_score": np.asarray(oracle_scores, dtype=np.float32),
                                "candidate_basin_positive": (np.asarray(oracle_next_dists, dtype=np.float32) <= 1.0).astype(np.float32),
                                "candidate_tier": improvement_tiers(
                                    np.asarray(oracle_scores, dtype=np.float32),
                                    (np.asarray(oracle_next_dists, dtype=np.float32) <= 1.0).astype(np.float32),
                                ).astype(np.int64),
                                "best_candidate_index": int(row["oracle_candidate_index"]),
                                "best_group_index": int(
                                    refiner.alignment_controller._candidate_group_index.cpu().numpy()[row["oracle_candidate_index"]]
                                ),
                                "support_source_index": int(matched_ref_idx),
                                "runtime_selected_candidate_index": int(runtime_selected_idx),
                                "runtime_selected_group_index": int(row["last_scorer_group_index"]),
                            }
                        )
            else:
                row["alignment_gate_open"] = False
                row["alignment_blocked_reason"] = "planner_only"
                row["executed_delta_norm_mean_so_far"] = 0.0
                row["correction_count_so_far"] = 0

            curves.append(row)
            abs_action = delta_to_absolute(delta_action, obs.gripper_pose)
            row["workspace_violation"] = float(safety.workspace_violation(abs_action[:3]))
            row["exec_gripper_raw"] = float(delta_action[6])
            if float(delta_action[6]) <= 0.5 and metrics_basin["basin_distance"] > 1.0:
                close_before_basin_count += 1
            row["close_before_basin_count_so_far"] = int(close_before_basin_count)
            try:
                obs, reward, terminate = task.step(abs_action)
            except Exception:
                row["invalid_action"] = True
                break
            row["reward"] = float(reward)
            row["terminate"] = bool(terminate)
            if terminate or reward > 0:
                break
            chunk_step += 1
    finally:
        env.shutdown()

    summary = {
        "mode": mode_name,
        "episode_idx": episode_idx,
        "reference_close_idx": int(reference_close_idx),
        "reference_anchor_idx": int(reference_anchor_idx),
        "anchor_xy_curve": summarize_curve([x["anchor_xy"] for x in curves]),
        "basin_xy_curve": summarize_curve([x["basin_xy"] for x in curves]),
        "basin_distance_curve": summarize_curve([x["basin_distance"] for x in curves]),
        "workspace_violation_curve": summarize_curve([x["workspace_violation"] for x in curves]),
        "planner_nominal_basin_distance_curve": summarize_curve([x["planner_nominal_basin_distance"] for x in curves]),
        "steps": len(curves),
        "final_block_reason": curves[-1]["alignment_blocked_reason"] if curves else None,
        "correction_count": curves[-1]["correction_count_so_far"] if curves else 0,
        "close_before_basin_count": curves[-1]["close_before_basin_count_so_far"] if curves else 0,
    }
    return {"summary": summary, "curves": curves, "frames": frames if save_frames else None}


def write_hard_mining_rows(rows, output_npz: Path):
    if not rows:
        return None
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    stacked = {}
    for key in keys:
        vals = [row[key] for row in rows]
        stacked[key] = np.stack(vals, axis=0)
    np.savez_compressed(output_npz, **stacked)
    return output_npz


def write_mode_gifs(data, output_dir: Path, fps: int = 6):
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode_name, mode_data in data.items():
        frames = mode_data.get("frames")
        if not frames:
            continue
        path = output_dir / f"{mode_name}.gif"
        imageio.mimsave(path, frames, fps=fps)


def strip_frames_for_json(data):
    cleaned = {}
    for mode_name, mode_data in data.items():
        cleaned[mode_name] = {
            "summary": mode_data.get("summary", {}),
            "curves": mode_data.get("curves", []),
        }
    return cleaned


def render_plot(data, output_png: Path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    for mode_name, color in [("planner_only", "tab:blue"), ("pose_alignment_only_basin", "tab:orange")]:
        curves = data[mode_name]["curves"]
        steps = [r["step"] for r in curves]
        axes[0].plot(steps, [r["anchor_xy"] for r in curves], label=f"{mode_name}: anchor_xy", color=color, linestyle="-")
        axes[0].plot(steps, [r["basin_xy"] for r in curves], label=f"{mode_name}: basin_xy", color=color, linestyle="--")
        axes[1].plot(steps, [r["basin_distance"] for r in curves], label=mode_name, color=color)
        axes[1].plot(steps, [r.get("planner_nominal_basin_distance", 0.0) for r in curves], label=f"{mode_name}: planner_nominal_basin", color=color, linestyle=":")
        axes[2].plot(steps, [1.0 if r.get("planner_close_intent", False) else 0.0 for r in curves], label=f"{mode_name}: close_intent", color=color, linestyle="-")
        axes[2].plot(steps, [r.get("close_before_basin_count_so_far", 0) for r in curves], label=f"{mode_name}: close_before_basin", color=color, linestyle="--")
        axes[3].plot(steps, [r.get("workspace_violation", 0.0) for r in curves], label=f"{mode_name}: workspace_violation", color=color, linestyle="-")
        if mode_name == "pose_alignment_only_basin":
            axes[3].plot(steps, [r.get("executed_delta_norm_mean_so_far", 0.0) for r in curves], label=f"{mode_name}: executed_delta_mean", color="tab:red", linestyle="--")
            axes[3].plot(steps, [r.get("runtime_funnel_cost", 0.0) for r in curves], label=f"{mode_name}: funnel_cost", color="tab:green", linestyle=":")
    axes[0].set_ylabel("EE Distance (m)")
    axes[0].set_title("EE to Reference Anchor / Basin Center")
    axes[1].set_ylabel("Basin Distance")
    axes[1].set_title("Basin Distance Trajectory")
    axes[2].set_ylabel("Intent / Count")
    axes[2].set_title("Planner Close Intent / Close Before Basin")
    axes[3].set_ylabel("Magnitude")
    axes[3].set_title("Workspace Violation / Executed Delta")
    axes[3].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze EE distance curves to reference/basin center.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--alignment_ckpt", type=str, required=True)
    parser.add_argument("--close_trigger_ckpt", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--pregrasp_window", type=int, default=16)
    parser.add_argument("--basin_center_mode", type=str, default="success_region_proxy", choices=["lastk_mean", "success_region_proxy"])
    parser.add_argument("--success_region_window", type=int, default=8)
    parser.add_argument("--success_region_exclude_last", type=int, default=1)
    parser.add_argument("--planner_no_depth", action="store_true", default=True)
    parser.add_argument("--planner_no_force", action="store_true", default=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_png", type=str, default=None)
    parser.add_argument("--hard_mining_output_npz", type=str, default=None)
    parser.add_argument("--support_states_output_npz", type=str, default=None)
    parser.add_argument("--trigger_states_output_npz", type=str, default=None)
    parser.add_argument("--output_gif_dir", type=str, default=None)
    args = parser.parse_args()

    hard_mining_rows = []
    support_state_rows = []
    trigger_state_rows = []
    out = {
            "planner_only": run_mode(
            "planner_only",
            args.checkpoint_dir,
            args.alignment_ckpt,
            args.close_trigger_ckpt,
            args.task_name,
            args.data_root,
            args.episode_idx,
            args.max_steps,
            args.pregrasp_window,
            args.planner_no_depth,
            args.planner_no_force,
            args.basin_center_mode,
            args.success_region_window,
            args.success_region_exclude_last,
            hard_mining_rows=None,
            trigger_state_rows=trigger_state_rows,
            save_frames=bool(args.output_gif_dir),
        ),
        "pose_alignment_only_basin": run_mode(
            "pose_alignment_only_basin",
            args.checkpoint_dir,
            args.alignment_ckpt,
            args.close_trigger_ckpt,
            args.task_name,
            args.data_root,
            args.episode_idx,
            args.max_steps,
            args.pregrasp_window,
            args.planner_no_depth,
            args.planner_no_force,
            args.basin_center_mode,
            args.success_region_window,
            args.success_region_exclude_last,
            hard_mining_rows=hard_mining_rows,
            support_state_rows=support_state_rows,
            trigger_state_rows=trigger_state_rows,
            save_frames=bool(args.output_gif_dir),
        ),
    }
    json_out = strip_frames_for_json(out)
    print(json.dumps(json_out, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    if args.output_png:
        render_plot(out, Path(args.output_png))
    if args.hard_mining_output_npz:
        write_hard_mining_rows(hard_mining_rows, Path(args.hard_mining_output_npz))
    if args.support_states_output_npz:
        write_hard_mining_rows(support_state_rows, Path(args.support_states_output_npz))
    if args.trigger_states_output_npz:
        write_hard_mining_rows(trigger_state_rows, Path(args.trigger_states_output_npz))
    if args.output_gif_dir:
        write_mode_gifs(out, Path(args.output_gif_dir))


if __name__ == "__main__":
    main()
