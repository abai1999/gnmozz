"""
online_basin_offset_probe.py

Construct synthetic local offsets around a pre-grasp reset observation and probe
whether the alignment controller predicts corrections toward the reference basin
center.
"""

import argparse
import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch

from prismatic.vla.constants import FORCE_DIM, FORCE_HISTORY_LEN
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from prismatic.robot.stage_manager import StageManager
from prismatic.robot.stage_manager import StagePhase
from prismatic.robot.stage_aware_refiner import StageAwareRefiner
from scripts.collect_residual_data import load_planner
from scripts.collect_stage_refiner_data import (
    build_reference_preclose_segment,
    choose_pregrasp_reset_index,
    compute_basin_center_pose,
    compute_basin_metrics,
    interaction_role_to_code,
    make_env,
    make_gripper_context,
    match_reference_index,
    planner_close_intent_from_context,
    pose_delta_local_between,
    reconstruct_expert_action_7d,
)
from scripts.evaluate_rlbench import delta_to_absolute, load_residual_controller, process_obs, predict_actions


def apply_local_offset(pose_7d: np.ndarray, delta_local_6d: np.ndarray) -> np.ndarray:
    delta_world = local_delta_to_world(delta_local_6d.astype(np.float32), pose_7d[3:7])
    out = pose_7d.copy().astype(np.float32)
    out[:6] += delta_world[:6]
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def collect_probe_state(
    task,
    obs,
    instruction: str,
    npz_data,
    vla,
    processor,
    action_head,
    proprio_projector,
    norm_stats,
    rollout_offset: int,
    reference_candidates: np.ndarray,
    reference_anchor_idx: int,
    basin_center_pose: np.ndarray,
    probe_mode: str,
    open_gripper_threshold: float,
    planner_close_threshold: float,
):
    force_buffer = deque(maxlen=256)
    manager = StageManager()
    action_queue = []
    chosen = None

    for rollout_step in range(80):
        front_pil, wrist_pil, proprio, _, force_hist, depth_tensor_96, _ = process_obs(
            obs,
            norm_stats,
            force_buffer,
            use_depth=True,
            use_force=False,
            depth_max=1.0,
        )
        if force_hist is None:
            force_hist = torch.zeros((FORCE_HISTORY_LEN, FORCE_DIM), dtype=torch.float32)

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
            action_queue = [np.asarray(a, dtype=np.float32) for a in actions]

        base_action = np.asarray(action_queue.pop(0), dtype=np.float32)
        future_gripper_actions = action_queue[:4]
        depth_prox = StageAwareRefiner.compute_depth_proximity(depth_tensor_96)
        manager.update(
            force_reading=np.zeros(6, dtype=np.float32),
            gripper_pose=obs.gripper_pose,
            gripper_open=float(obs.gripper_open),
            depth_proximity=depth_prox,
            base_action=base_action,
        )
        gripper_context = make_gripper_context(base_action, future_gripper_actions)
        planner_close_intent, planner_close_intent_strength = planner_close_intent_from_context(
            gripper_context,
            threshold=planner_close_threshold,
        )
        matched_ref_idx = match_reference_index(
            npz_data,
            obs.gripper_pose,
            reference_candidates,
            rot_weight=0.05,
        )
        frames_to_reference_trigger = (
            int(reference_anchor_idx - matched_ref_idx)
            if matched_ref_idx >= 0 and reference_anchor_idx >= 0
            else -1
        )
        delta_local = pose_delta_local_between(obs.gripper_pose, basin_center_pose)
        basin_distance, e_xy, e_z, e_yaw = compute_basin_metrics(
            delta_local,
            r_xy=0.008,
            r_z=0.01,
            r_yaw=0.05,
        )
        support_aligned = bool(
            int(manager.phase) == int(StagePhase.ALIGN)
            and float(obs.gripper_open) >= open_gripper_threshold
            and planner_close_intent
        )
        reset_state = rollout_step == 0
        choose = support_aligned if probe_mode == "support_aligned" else reset_state
        if choose:
            chosen = {
                "obs": obs,
                "proprio": proprio,
                "force_hist": force_hist,
                "depth_tensor_96": depth_tensor_96,
                "base_action": base_action,
                "gripper_context": gripper_context,
                "probe_info": {
                    "probe_mode": probe_mode,
                    "rollout_offset": int(rollout_offset),
                    "rollout_step": int(rollout_step),
                    "phase": StagePhase(int(manager.phase)).name,
                    "phase_id": int(manager.phase),
                    "gripper_open": float(obs.gripper_open),
                    "depth_proximity": None if depth_prox is None else float(depth_prox),
                    "planner_close_intent": bool(planner_close_intent),
                    "planner_close_intent_strength": float(planner_close_intent_strength),
                    "gripper_context": gripper_context.tolist(),
                    "matched_ref_idx": int(matched_ref_idx),
                    "frames_to_reference_trigger": int(frames_to_reference_trigger),
                    "basin_distance": float(basin_distance),
                    "e_xy": float(e_xy),
                    "e_z": float(e_z),
                    "e_yaw": float(e_yaw),
                    "delta_local": delta_local.tolist(),
                    "support_aligned": bool(support_aligned),
                },
            }
            break

        abs_action = delta_to_absolute(base_action, obs.gripper_pose)
        obs, _, terminate = task.step(abs_action)
        if terminate:
            break

    return chosen


def main():
    parser = argparse.ArgumentParser(description="Online static basin offset probe")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--alignment_ckpt", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--pregrasp_window", type=int, default=16)
    parser.add_argument("--open_gripper_threshold", type=float, default=0.5)
    parser.add_argument("--planner_close_threshold", type=float, default=0.5)
    parser.add_argument(
        "--probe_mode",
        type=str,
        default="support_aligned",
        choices=["reset_state", "support_aligned"],
    )
    parser.add_argument("--basin_center_mode", type=str, default="success_region_proxy", choices=["lastk_mean", "success_region_proxy"])
    parser.add_argument("--success_region_window", type=int, default=8)
    parser.add_argument("--success_region_exclude_last", type=int, default=1)
    parser.add_argument("--planner_no_depth", action="store_true", default=True)
    parser.add_argument("--planner_no_force", action="store_true", default=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        args.checkpoint_dir,
        "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
        "pretrained_models/configs/config.json",
        use_depth=not args.planner_no_depth,
        use_force=not args.planner_no_force,
    )
    controller = load_residual_controller(args.alignment_ckpt)

    episodes_dir = Path(args.data_root) / args.task_name / "train" / "episodes"
    ep_name = f"episode{args.episode_idx}"
    ep_path = episodes_dir / ep_name
    npz_data = dict(np.load(ep_path / "model_inputs.npz"))
    with open(ep_path / "low_dim_obs.pkl", "rb") as f:
        demo_obs = pickle.load(f)

    env, task = make_env(args.task_name, Path(args.data_root))
    try:
        descs, obs = task.reset_to_demo(demo_obs)
        rollout_offset = choose_pregrasp_reset_index(
            npz_data,
            window=args.pregrasp_window,
            open_threshold=args.open_gripper_threshold,
        )
        for i in range(rollout_offset):
            expert_action = reconstruct_expert_action_7d(npz_data, i)
            abs_action = delta_to_absolute(expert_action, obs.gripper_pose)
            obs, _, terminate = task.step(abs_action)
            if terminate:
                break

        reference_candidates, reference_close_idx, reference_anchor_idx = build_reference_preclose_segment(
            npz_data,
            start_idx=rollout_offset,
            open_threshold=args.open_gripper_threshold,
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
        instruction = descs[0] if isinstance(descs, list) else str(descs)
        chosen = collect_probe_state(
            task=task,
            obs=obs,
            instruction=instruction,
            npz_data=npz_data,
            vla=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            norm_stats=norm_stats,
            rollout_offset=rollout_offset,
            reference_candidates=reference_candidates,
            reference_anchor_idx=reference_anchor_idx,
            basin_center_pose=basin_center_pose,
            probe_mode=args.probe_mode,
            open_gripper_threshold=args.open_gripper_threshold,
            planner_close_threshold=args.planner_close_threshold,
        )
        if chosen is None:
            raise RuntimeError(f"Failed to find a probe state for mode={args.probe_mode}")
        obs = chosen["obs"]
        proprio = chosen["proprio"]
        force_hist = chosen["force_hist"]
        depth_tensor_96 = chosen["depth_tensor_96"]
        base_action = chosen["base_action"]
        gripper_context = chosen["gripper_context"]
        probe_info = chosen["probe_info"]

        offsets = [
            ("dx_p5mm", np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            ("dx_m5mm", np.array([-0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            ("dx_p10mm", np.array([0.010, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            ("dy_p5mm", np.array([0.0, 0.005, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            ("dy_m5mm", np.array([0.0, -0.005, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            ("yaw_p0.05", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05], dtype=np.float32)),
            ("yaw_m0.05", np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.05], dtype=np.float32)),
        ]

        device = next(controller.parameters()).device
        dtype = next(controller.parameters()).dtype
        results = []
        with torch.no_grad():
            for name, offset_local in offsets:
                synth_pose = apply_local_offset(obs.gripper_pose, offset_local)
                proprio_synth = proprio.copy()
                proprio_synth[7:14] = synth_pose
                base_action_local = world_delta_to_local(base_action[:6], synth_pose[3:7])
                outputs = controller(
                    torch.from_numpy(depth_tensor_96.numpy()).unsqueeze(0).to(device=device, dtype=dtype),
                    force_hist.unsqueeze(0).to(device=device, dtype=dtype),
                    torch.from_numpy(proprio_synth).unsqueeze(0).to(device=device, dtype=dtype),
                    torch.from_numpy(base_action_local.astype(np.float32)).unsqueeze(0).to(device=device, dtype=dtype),
                    torch.tensor([0], device=device, dtype=torch.long),
                    phase_id=torch.tensor([int(StagePhase.ALIGN)], device=device, dtype=torch.long),
                    phase_age=torch.tensor([1.0], device=device, dtype=dtype),
                    steps_since_last_replan=torch.tensor([0.0], device=device, dtype=dtype),
                    gripper_context=torch.from_numpy(gripper_context).unsqueeze(0).to(device=device, dtype=dtype),
                    return_aux=True,
                )
                pred = outputs["delta_pose"].squeeze(0).float().cpu().numpy()
                target = pose_delta_local_between(synth_pose, basin_center_pose)
                pred4 = np.asarray([pred[0], pred[1], pred[2], pred[5]], dtype=np.float32)
                tgt4 = np.asarray([target[0], target[1], target[2], target[5]], dtype=np.float32)
                sign_hits = []
                for idx in (0, 1, 5):
                    if abs(float(target[idx])) > 1e-6:
                        sign_hits.append(float(np.sign(pred[idx]) == np.sign(target[idx])))
                results.append(
                    {
                        "offset_name": name,
                        "offset_local": offset_local.tolist(),
                        "target_local": target.tolist(),
                        "pred_local": pred.tolist(),
                        "cosine": cosine(pred4, tgt4),
                        "norm_ratio": float(np.linalg.norm(pred4) / max(np.linalg.norm(tgt4), 1e-8)),
                        "sign_agreement": float(np.mean(sign_hits)) if sign_hits else 0.0,
                    }
                )
    finally:
        env.shutdown()

    summary = {
        "task_name": args.task_name,
        "episode_idx": args.episode_idx,
        "interaction_role": "pre_grasp",
        "probe_info": probe_info,
        "results": results,
        "cosine_mean": float(np.mean([r["cosine"] for r in results])) if results else 0.0,
        "sign_agreement_mean": float(np.mean([r["sign_agreement"] for r in results])) if results else 0.0,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
