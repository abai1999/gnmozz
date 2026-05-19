"""
collect_residual_planner_state.py

Collect planner-state aligned residual supervision by rolling out the frozen planner
from stored RLBench demo initial states and pairing planner-state observations with
step-aligned expert actions from the same episode.
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
_HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/mnt/ssd/guoning/hf-cache")
os.environ.setdefault("HF_HOME", _HF_CACHE_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from prismatic.vla.constants import (
    FORCE_DIM,
    NUM_ACTIONS_CHUNK,
)
from prismatic.robot.contact_refiner import ContactRefiner
from prismatic.robot.residual_transforms import world_delta_to_local
from scripts.collect_residual_data import load_planner
from scripts.evaluate_rlbench import (
    TASK_MAP,
    _lazy_import_tasks,
    delta_to_absolute,
    load_residual_controller,
    predict_actions,
    process_obs,
)


def compute_phase_label_from_state(force_history, force_reading, gripper_z, depth_tensor_96,
                                   force_threshold=0.5, jam_threshold=3.0,
                                   z_threshold=0.90, depth_threshold=0.15):
    force_mag = 0.0 if force_reading is None else np.linalg.norm(force_reading[:3])
    prev_force = 0.0 if len(force_history) < 2 else np.linalg.norm(np.asarray(force_history[-2])[:3])
    depth_prox = None
    if depth_tensor_96 is not None:
        depth_arr = depth_tensor_96.numpy() if hasattr(depth_tensor_96, "numpy") else np.asarray(depth_tensor_96)
        valid = depth_arr[np.isfinite(depth_arr)]
        if valid.size > 0:
            depth_prox = float(np.percentile(valid, 5.0))

    if force_mag > jam_threshold:
        return 4
    if force_mag > force_threshold:
        return 2 if prev_force <= force_threshold else 3
    if (gripper_z is not None and gripper_z < z_threshold) or (
        depth_prox is not None and depth_prox < depth_threshold
    ):
        return 1
    return 0


def phase_to_contact_mask(phase_label):
    return 0 if phase_label == 0 else (1 if phase_label == 1 else 2)


def clip_delta(delta, clip_pos, clip_rot):
    out = delta.astype(np.float32).copy()
    pos_norm = np.linalg.norm(out[:3])
    if pos_norm > clip_pos:
        out[:3] = out[:3] * (clip_pos / max(pos_norm, 1e-8))
    rot_norm = np.linalg.norm(out[3:6])
    if rot_norm > clip_rot:
        out[3:6] = out[3:6] * (clip_rot / max(rot_norm, 1e-8))
    return out


def make_env(task_name, dataset_root):
    _lazy_import_tasks()
    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    if hasattr(obs_config, "gripper_touch_forces"):
        obs_config.gripper_touch_forces = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode, dataset_root=str(dataset_root), obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(TASK_MAP[task_name])
    return env, task


def flush_shard(output_dir, shard_idx, buffers):
    shard_path = output_dir / f"residual_shard_{shard_idx:04d}.npz"
    np.savez_compressed(
        shard_path,
        wrist_depth=np.array(buffers["wrist_depth"], dtype=np.float32),
        ft_hist=np.array(buffers["ft_hist"], dtype=np.float32),
        proprio=np.array(buffers["proprio"], dtype=np.float32),
        base_action=np.array(buffers["base_action"], dtype=np.float32),
        step_idx=np.array(buffers["step_idx"], dtype=np.int64),
        delta_target=np.array(buffers["delta_target"], dtype=np.float32),
        contact_mask=np.array(buffers["contact_mask"], dtype=np.int64),
        phase_label=np.array(buffers["phase_label"], dtype=np.int64),
    )
    print(f"  [shard {shard_idx}] Saved {len(buffers['delta_target'])} samples to {shard_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect planner-state residual training data")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, default="data/rlbench_data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vlm_path", type=str, default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", type=str, default="pretrained_models/configs/config.json")
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.set_defaults(planner_use_depth=None, planner_use_force=None)
    parser.add_argument("--delta_clip_pos", type=float, default=0.01)
    parser.add_argument("--delta_clip_rot", type=float, default=0.05)
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--max_rollout_steps", type=int, default=-1)
    parser.add_argument("--use_rule_reflex", action="store_true", default=False)
    parser.add_argument("--use_learned_residual", action="store_true", default=False)
    parser.add_argument("--residual_ckpt", type=str, default=None)
    args = parser.parse_args()

    if args.planner_use_depth is None:
        args.planner_use_depth = args.use_depth
    if args.planner_use_force is None:
        args.planner_use_force = args.use_force

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[planner_state_collect] Loading frozen planner...")
    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        args.checkpoint_dir,
        args.vlm_path,
        args.config_path,
        args.planner_use_depth,
        args.planner_use_force,
    )

    episodes_dir = Path(args.data_root) / args.task_name / "train" / "episodes"
    assert episodes_dir.exists(), f"Episodes dir not found: {episodes_dir}"
    ep_dirs = sorted(
        [d for d in os.listdir(episodes_dir) if d.startswith("episode")],
        key=lambda x: int(x.replace("episode", "")),
    )
    if args.max_episodes > 0:
        ep_dirs = ep_dirs[:args.max_episodes]

    env, task = make_env(args.task_name, Path(args.data_root))
    refiner = None
    if args.use_learned_residual:
        if not args.residual_ckpt:
            raise ValueError("--use_learned_residual requires --residual_ckpt")
        refiner = ContactRefiner(
            mode="learned_residual",
            residual_controller=load_residual_controller(args.residual_ckpt),
        )
    elif args.use_rule_reflex:
        refiner = ContactRefiner(mode="rule_reflex")

    buffers = {
        "wrist_depth": [],
        "ft_hist": [],
        "proprio": [],
        "base_action": [],
        "step_idx": [],
        "delta_target": [],
        "contact_mask": [],
        "phase_label": [],
    }
    shard_idx = 0
    total_samples = 0
    phase_counts = {0: 0, 1: 0, 2: 0}
    phase_label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    try:
        for ep_i, ep_name in enumerate(ep_dirs):
            ep_num = int(ep_name.replace("episode", ""))
            ep_path = episodes_dir / ep_name
            npz_path = ep_path / "model_inputs.npz"
            if not npz_path.exists():
                continue

            npz_data = dict(np.load(npz_path))
            expert_actions = npz_data["action_targets"].astype(np.float32)
            T = expert_actions.shape[0]
            if T < NUM_ACTIONS_CHUNK:
                continue

            low_dim_obs_path = ep_path / "low_dim_obs.pkl"
            assert low_dim_obs_path.exists(), f"Missing low_dim_obs.pkl for {ep_path}"
            with open(low_dim_obs_path, "rb") as f:
                demo_obs = pickle.load(f)
            descs, obs = task.reset_to_demo(demo_obs)

            desc_path = ep_path / "variation_descriptions.pkl"
            if desc_path.exists():
                with open(desc_path, "rb") as f:
                    descs_from_disk = pickle.load(f)
                instruction = descs_from_disk[0] if isinstance(descs_from_disk, list) else str(descs_from_disk)
            else:
                instruction = descs[0]

            force_buffer = deque(maxlen=256)
            action_queue = []
            chunk_step = 0
            max_steps = T if args.max_rollout_steps <= 0 else min(T, args.max_rollout_steps)
            if refiner is not None:
                refiner.reset()

            t0 = time.time()
            for rollout_step in range(max_steps):
                front_pil, wrist_pil, proprio, depth_tensor_224, force_hist, depth_tensor_96, raw_force = process_obs(
                    obs,
                    norm_stats,
                    force_buffer,
                    use_depth=args.use_depth,
                    use_force=args.use_force,
                    depth_max=1.0,
                )

                if len(action_queue) == 0:
                    if refiner is not None:
                        refiner.trigger.update(
                            force_reading=raw_force,
                            gripper_z=float(obs.gripper_pose[2]),
                            depth_proximity=refiner.compute_depth_proximity(depth_tensor_96),
                        )
                    actions = predict_actions(
                        vla,
                        processor,
                        action_head,
                        proprio_projector,
                        front_pil,
                        wrist_pil,
                        proprio,
                        depth_tensor_224 if args.planner_use_depth else None,
                        force_hist if args.planner_use_force else None,
                        instruction,
                        unnorm_key="rlbench",
                    )
                    max_chunk = refiner.get_chunk_size() if refiner is not None else len(actions)
                    action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), max_chunk))]
                    chunk_step = 0

                base_action = action_queue.pop(0)
                expert_action = expert_actions[min(rollout_step, T - 1)]
                current_quat = obs.gripper_pose[3:7]
                base_action_local = world_delta_to_local(base_action[:6], current_quat)
                delta_target = clip_delta(
                    world_delta_to_local(expert_action[:6] - base_action[:6], current_quat),
                    clip_pos=args.delta_clip_pos,
                    clip_rot=args.delta_clip_rot,
                )

                buffers["wrist_depth"].append(
                    depth_tensor_96.numpy().astype(np.float32)
                    if depth_tensor_96 is not None
                    else np.zeros((1, 96, 96), dtype=np.float32)
                )
                buffers["ft_hist"].append(
                    force_hist.numpy().astype(np.float32)
                    if force_hist is not None
                    else np.zeros((32, FORCE_DIM), dtype=np.float32)
                )
                buffers["proprio"].append(proprio.astype(np.float32))
                buffers["base_action"].append(base_action_local.astype(np.float32))
                buffers["step_idx"].append(chunk_step)
                buffers["delta_target"].append(delta_target)
                phase_label = compute_phase_label_from_state(
                    list(force_buffer), raw_force, float(obs.gripper_pose[2]), depth_tensor_96
                )
                c_mask = phase_to_contact_mask(phase_label)
                buffers["contact_mask"].append(c_mask)
                buffers["phase_label"].append(phase_label)
                phase_counts[c_mask] += 1
                phase_label_counts[phase_label] += 1
                total_samples += 1

                exec_action = base_action
                if refiner is not None:
                    exec_action = refiner.step(
                        a_base_7d=base_action,
                        step_idx=chunk_step,
                        force_reading=raw_force,
                        gripper_z=float(obs.gripper_pose[2]),
                        wrist_depth=depth_tensor_96,
                        ft_hist=force_hist,
                        proprio=proprio,
                    )

                abs_action = delta_to_absolute(exec_action, obs.gripper_pose)
                if refiner is not None:
                    abs_action[:3] = refiner.safety.clamp_workspace(abs_action[:3])
                try:
                    obs, reward, terminate = task.step(abs_action)
                except Exception as e:
                    print(f"[planner_state_collect] Episode {ep_name} step {rollout_step}: {type(e).__name__}, stopping rollout")
                    break

                chunk_step += 1
                if refiner is not None and refiner.should_replan():
                    if len(action_queue) > 0:
                        refiner.note_replan()
                    action_queue.clear()
                if reward > 0 or terminate:
                    break

                if len(buffers["delta_target"]) >= args.shard_size:
                    flush_shard(output_dir, shard_idx, buffers)
                    shard_idx += 1
                    for key in buffers:
                        buffers[key].clear()

            elapsed = time.time() - t0
            print(
                f"[planner_state_collect] Episode {ep_i + 1}/{len(ep_dirs)} ({ep_name}): "
                f"{max_steps} rollout steps max, elapsed={elapsed:.1f}s, total_samples={total_samples}"
            )
    finally:
        env.shutdown()

    if len(buffers["delta_target"]) > 0:
        flush_shard(output_dir, shard_idx, buffers)

    meta = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "task_name": args.task_name,
        "collector_mode": "planner_state_aligned_residual" if refiner is not None else "planner_state_aligned",
        "label_frame": "ee_local",
        "num_episodes": len(ep_dirs),
        "total_samples": total_samples,
        "delta_clip_pos": args.delta_clip_pos,
        "delta_clip_rot": args.delta_clip_rot,
        "phase_counts": {
            "free": phase_counts[0],
            "pre_contact": phase_counts[1],
            "contact": phase_counts[2],
        },
        "phase_label_counts": {
            "free": phase_label_counts[0],
            "near_alignment": phase_label_counts[1],
            "first_contact": phase_label_counts[2],
            "sliding_contact": phase_label_counts[3],
            "jam": phase_label_counts[4],
        },
    }
    with open(output_dir / "residual_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[planner_state_collect] Done. Metadata saved to {output_dir / 'residual_meta.json'}")


if __name__ == "__main__":
    main()
