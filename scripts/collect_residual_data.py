"""
collect_residual_data.py

Offline residual dataset collection:
1. Load frozen planner checkpoint
2. For each expert episode, at strided timesteps, run planner to get 8-step chunk
3. For each j in 0..7, pair planner's step-j action with expert's t+j action
4. Convert base and target residual into EE-local frame
5. Save per-shard .npz files + metadata, including richer phase labels

Usage:
    python scripts/collect_residual_data.py \
        --checkpoint_dir outputs/insert_long_train/run--30000_chkpt \
        --task_name insert_onto_square_peg \
        --output_dir data/residual_data/insert_onto_square_peg \
        --stride 4 \
        --delta_clip_pos 0.01 \
        --delta_clip_rot 0.05
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
from PIL import Image
from scipy.spatial.transform import Rotation

# Ensure PYTHONPATH includes workspace root
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

from prismatic.vla.constants import (
    ACTION_DIM,
    FORCE_DIM,
    FORCE_HISTORY_LEN,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.robot.residual_transforms import world_delta_to_local


def load_planner(checkpoint_dir, vlm_path, config_path, use_depth, use_force):
    """Load the frozen planner using the same logic as evaluate_rlbench.py."""
    # Import here to avoid circular imports and CoppeliaSim setup
    from scripts.evaluate_rlbench import load_checkpoint
    return load_checkpoint(checkpoint_dir, vlm_path, config_path, use_depth, use_force)


def predict_chunk_for_obs(
    vla, processor, action_head, proprio_projector, norm_stats,
    front_pil, wrist_pil, proprio, depth_tensor, force_history,
    instruction, device,
):
    """Run planner inference and return unnormalized action chunk (8,7)."""
    from scripts.evaluate_rlbench import predict_actions
    actions = predict_actions(
        vla, processor, action_head, proprio_projector,
        front_pil, wrist_pil, proprio, depth_tensor, force_history,
        instruction, unnorm_key="rlbench",
    )
    return actions  # (8, 7) np array


def load_depth_96(ep_path, frame_idx):
    """Load wrist depth and resize to 96x96, returning (1, 96, 96) float32."""
    depth_img_path = ep_path / "wrist_depth" / f"{frame_idx}.png"
    if not depth_img_path.exists():
        return np.zeros((1, 96, 96), dtype=np.float32)
    depth_img = Image.open(depth_img_path)
    depth = np.array(depth_img, dtype=np.float32)
    if depth.ndim == 3:
        depth = (depth[:, :, 0] * 65536 + depth[:, :, 1] * 256 + depth[:, :, 2]) / (256.0**3 - 1)
    else:
        depth = depth / 255.0
    depth = np.clip(depth, 0.0, 1.0)
    # Resize to 96x96
    depth_pil = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
    depth_pil = depth_pil.resize((96, 96), Image.BILINEAR)
    depth_96 = np.array(depth_pil, dtype=np.float32) / 255.0
    return depth_96[np.newaxis, :, :]  # (1, 96, 96)


def load_depth_224(ep_path, frame_idx):
    """Load wrist depth at 224x224 for planner input."""
    depth_img_path = ep_path / "wrist_depth" / f"{frame_idx}.png"
    if not depth_img_path.exists():
        return None
    depth_img = Image.open(depth_img_path)
    depth = np.array(depth_img, dtype=np.float32)
    if depth.ndim == 3:
        depth = (depth[:, :, 0] * 65536 + depth[:, :, 1] * 256 + depth[:, :, 2]) / (256.0**3 - 1)
    else:
        depth = depth / 255.0
    depth = np.clip(depth, 0.0, 1.0)
    depth_pil = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
    depth_pil = depth_pil.resize((224, 224), Image.BILINEAR)
    depth_tensor = torch.from_numpy(np.array(depth_pil, dtype=np.float32) / 255.0).unsqueeze(0)
    return depth_tensor  # (1, 224, 224)


def get_force_history(forces, frame_idx, force_mean, force_std, history_len=32):
    """Extract z-score normalized force history ending at frame_idx. Returns (history_len, 6)."""
    T = forces.shape[0]
    history = np.zeros((history_len, FORCE_DIM), dtype=np.float32)
    for i in range(history_len):
        t = frame_idx - (history_len - 1 - i)
        t = max(0, min(t, T - 1))
        history[i] = forces[t]
    history = (history - force_mean) / force_std
    return history


def compute_phase_label(forces_raw, frame_idx, gripper_z=None, depth_96=None,
                        force_threshold=0.5, jam_threshold=3.0, z_threshold=0.90,
                        depth_threshold=0.15):
    """Heuristic phase label: 0=free, 1=near_alignment, 2=first_contact, 3=sliding_contact, 4=jam."""
    force_mag = np.linalg.norm(forces_raw[frame_idx])
    prev_force = np.linalg.norm(forces_raw[max(frame_idx - 1, 0)])
    depth_prox = None
    if depth_96 is not None:
        valid = depth_96[np.isfinite(depth_96)]
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


def main():
    parser = argparse.ArgumentParser(description="Collect residual training data")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to frozen planner checkpoint")
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, default="data/rlbench_data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save residual shards")
    parser.add_argument("--vlm_path", type=str,
                        default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", type=str,
                        default="pretrained_models/configs/config.json")
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.set_defaults(planner_use_depth=None, planner_use_force=None)
    parser.add_argument("--stride", type=int, default=4,
                        help="Stride for sampling starting timesteps within each episode")
    parser.add_argument("--delta_clip_pos", type=float, default=0.01,
                        help="Max residual position clip in meters")
    parser.add_argument("--delta_clip_rot", type=float, default=0.05,
                        help="Max residual rotation clip in radians")
    parser.add_argument("--shard_size", type=int, default=10000,
                        help="Number of samples per shard file")
    parser.add_argument("--max_episodes", type=int, default=-1,
                        help="Max episodes to process (-1 for all)")
    args = parser.parse_args()

    if args.planner_use_depth is None:
        args.planner_use_depth = args.use_depth
    if args.planner_use_force is None:
        args.planner_use_force = args.use_force

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Load planner ──
    print("[collect] Loading frozen planner...")
    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        args.checkpoint_dir, args.vlm_path, args.config_path,
        args.planner_use_depth, args.planner_use_force,
    )

    # ── Load dataset statistics ──
    rlbench_stats = norm_stats.get("rlbench", norm_stats)
    action_q01 = np.array(rlbench_stats["action"]["q01"], dtype=np.float32)
    action_q99 = np.array(rlbench_stats["action"]["q99"], dtype=np.float32)
    if "force" in rlbench_stats:
        force_mean = np.array(rlbench_stats["force"]["mean"], dtype=np.float32)
        force_std = np.maximum(np.array(rlbench_stats["force"]["std"], dtype=np.float32), 1e-6)
    else:
        force_mean = np.zeros(FORCE_DIM, dtype=np.float32)
        force_std = np.ones(FORCE_DIM, dtype=np.float32)

    # ── Find episodes ──
    episodes_dir = Path(args.data_root) / args.task_name / "train" / "episodes"
    assert episodes_dir.exists(), f"Episodes dir not found: {episodes_dir}"

    ep_dirs = sorted(
        [d for d in os.listdir(episodes_dir) if d.startswith("episode")],
        key=lambda x: int(x.replace("episode", "")),
    )
    if args.max_episodes > 0:
        ep_dirs = ep_dirs[:args.max_episodes]

    print(f"[collect] Processing {len(ep_dirs)} episodes from {episodes_dir}")

    # ── Output directory ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collection buffers ──
    buf_wrist_depth = []
    buf_ft_hist = []
    buf_proprio = []
    buf_base_action = []
    buf_step_idx = []
    buf_delta_target = []
    buf_contact_mask = []
    buf_phase_label = []

    shard_idx = 0
    total_samples = 0
    phase_counts = {0: 0, 1: 0, 2: 0}
    phase_label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for ep_i, ep_name in enumerate(ep_dirs):
        ep_path = episodes_dir / ep_name
        npz_path = ep_path / "model_inputs.npz"
        if not npz_path.exists():
            continue

        npz_data = dict(np.load(npz_path))
        T = npz_data["action_targets"].shape[0]
        if T < NUM_ACTIONS_CHUNK:
            continue

        # Expert actions unnormalized (already stored as normalized in npz? No — action_targets are raw)
        expert_actions = npz_data["action_targets"]  # (T, 7)
        forces_raw = npz_data.get("gripper_touch_forces",
                                   np.zeros((T, FORCE_DIM), dtype=np.float32))
        proprios = npz_data["proprio"]  # (T, 15)

        # Load language description
        desc_path = ep_path / "variation_descriptions.pkl"
        if desc_path.exists():
            with open(desc_path, "rb") as f:
                descs = pickle.load(f)
            instruction = descs[0] if isinstance(descs, list) else str(descs)
        else:
            instruction = args.task_name.replace("_", " ")

        # Iterate over starting timesteps
        max_start = T - NUM_ACTIONS_CHUNK
        starts = list(range(0, max_start + 1, args.stride))

        t0 = time.time()
        for t in starts:
            # ── Planner inference at timestep t ──
            front_pil = Image.open(ep_path / "front_rgb" / f"{t}.png").convert("RGB")
            wrist_pil = Image.open(ep_path / "wrist_rgb" / f"{t}.png").convert("RGB")
            proprio_t = proprios[t].astype(np.float32)

            depth_tensor_224 = load_depth_224(ep_path, t) if args.use_depth else None

            # Force history for planner
            force_hist_planner = None
            if args.use_force:
                fh = get_force_history(forces_raw, t, force_mean, force_std, FORCE_HISTORY_LEN)
                force_hist_planner = torch.from_numpy(fh)

            planned_chunk = predict_chunk_for_obs(
                vla, processor, action_head, proprio_projector, norm_stats,
                front_pil,
                wrist_pil,
                proprio_t,
                depth_tensor_224 if args.planner_use_depth else None,
                force_hist_planner if args.planner_use_force else None,
                instruction, device,
            )  # (8, 7) unnormalized

            # ── For each step j in chunk ──
            for j in range(NUM_ACTIONS_CHUNK):
                tj = t + j
                if tj >= T:
                    break

                # Expert action at t+j
                expert_action_tj = expert_actions[tj]  # (7,)
                proprio_tj = proprios[tj].astype(np.float32)

                # Base action from planner chunk step j
                base_action_j = planned_chunk[j]  # (7,)

                # Residual target: expert - base (6D pose only)
                delta = expert_action_tj[:6] - base_action_j[:6]
                current_quat = proprio_tj[10:14]
                delta = world_delta_to_local(delta, current_quat)
                base_action_local = world_delta_to_local(base_action_j[:6], current_quat)

                # Clip residual
                pos_norm = np.linalg.norm(delta[:3])
                if pos_norm > args.delta_clip_pos:
                    delta[:3] = delta[:3] * (args.delta_clip_pos / max(pos_norm, 1e-8))
                rot_norm = np.linalg.norm(delta[3:6])
                if rot_norm > args.delta_clip_rot:
                    delta[3:6] = delta[3:6] * (args.delta_clip_rot / max(rot_norm, 1e-8))

                # Obs at t+j for residual controller
                wrist_depth_96 = load_depth_96(ep_path, tj)
                ft_hist_tj = get_force_history(forces_raw, tj, force_mean, force_std, FORCE_HISTORY_LEN)

                # Contact mask
                gripper_z = proprio_tj[9] if len(proprio_tj) > 9 else None  # gripper_pose z
                phase_label = compute_phase_label(forces_raw, tj, gripper_z, wrist_depth_96.squeeze(0))
                c_mask = phase_to_contact_mask(phase_label)

                # Append to buffers
                buf_wrist_depth.append(wrist_depth_96)
                buf_ft_hist.append(ft_hist_tj)
                buf_proprio.append(proprio_tj)
                buf_base_action.append(base_action_local.astype(np.float32))
                buf_step_idx.append(j)
                buf_delta_target.append(delta.astype(np.float32))
                buf_contact_mask.append(c_mask)
                buf_phase_label.append(phase_label)

                phase_counts[c_mask] += 1
                phase_label_counts[phase_label] += 1
                total_samples += 1

            # Flush shard if buffer is large enough
            if len(buf_delta_target) >= args.shard_size:
                shard_path = output_dir / f"residual_shard_{shard_idx:04d}.npz"
                np.savez_compressed(
                    shard_path,
                    wrist_depth=np.array(buf_wrist_depth, dtype=np.float32),
                    ft_hist=np.array(buf_ft_hist, dtype=np.float32),
                    proprio=np.array(buf_proprio, dtype=np.float32),
                    base_action=np.array(buf_base_action, dtype=np.float32),
                    step_idx=np.array(buf_step_idx, dtype=np.int64),
                    delta_target=np.array(buf_delta_target, dtype=np.float32),
                    contact_mask=np.array(buf_contact_mask, dtype=np.int64),
                    phase_label=np.array(buf_phase_label, dtype=np.int64),
                )
                print(f"  [shard {shard_idx}] Saved {len(buf_delta_target)} samples to {shard_path}")
                shard_idx += 1
                buf_wrist_depth.clear()
                buf_ft_hist.clear()
                buf_proprio.clear()
                buf_base_action.clear()
                buf_step_idx.clear()
                buf_delta_target.clear()
                buf_contact_mask.clear()
                buf_phase_label.clear()

        elapsed = time.time() - t0
        print(f"[collect] Episode {ep_i+1}/{len(ep_dirs)} ({ep_name}): "
              f"{len(starts)} chunks, {elapsed:.1f}s, total={total_samples}")

    # ── Flush remaining buffer ──
    if len(buf_delta_target) > 0:
        shard_path = output_dir / f"residual_shard_{shard_idx:04d}.npz"
        np.savez_compressed(
            shard_path,
            wrist_depth=np.array(buf_wrist_depth, dtype=np.float32),
            ft_hist=np.array(buf_ft_hist, dtype=np.float32),
            proprio=np.array(buf_proprio, dtype=np.float32),
            base_action=np.array(buf_base_action, dtype=np.float32),
            step_idx=np.array(buf_step_idx, dtype=np.int64),
            delta_target=np.array(buf_delta_target, dtype=np.float32),
            contact_mask=np.array(buf_contact_mask, dtype=np.int64),
            phase_label=np.array(buf_phase_label, dtype=np.int64),
        )
        print(f"  [shard {shard_idx}] Saved {len(buf_delta_target)} samples to {shard_path}")

    # ── Save metadata ──
    meta = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "task_name": args.task_name,
        "num_episodes": len(ep_dirs),
        "total_samples": total_samples,
        "stride": args.stride,
        "collector_mode": "demo_state_local_frame",
        "label_frame": "ee_local",
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

    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"  Total samples:  {total_samples}")
    print(f"  Free-space:     {phase_counts[0]}")
    print(f"  Pre-contact:    {phase_counts[1]}")
    print(f"  Contact:        {phase_counts[2]}")
    print(f"  Shards:         {shard_idx + 1}")
    print(f"  Output:         {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
