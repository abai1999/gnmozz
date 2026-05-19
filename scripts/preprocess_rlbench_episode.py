"""
preprocess_rlbench_episode.py

Generates model_inputs.npz, phase_ids.npy, and dataset_statistics.json for RLBench tasks
that only have raw observation data (low_dim_obs.pkl + images).

Usage:
    LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python scripts/preprocess_rlbench_episode.py \
        --task plug_charger_in_power_supply

Produces per-episode:
    model_inputs.npz  (joint_positions, gripper_pose, gripper_open, gripper_touch_forces, proprio, action_targets)
    phase_ids.npy     (uniform phase 0, since no phase annotation exists)

Produces per-task:
    dataset_statistics.json  (action q01/q99, proprio q01/q99, force mean/std)
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def process_episode(ep_path: Path) -> dict:
    """Process a single episode directory, return arrays dict or None on failure."""
    ldo_path = ep_path / "low_dim_obs.pkl"
    if not ldo_path.exists():
        return None

    with open(ldo_path, "rb") as f:
        demo = pickle.load(f)

    T = len(demo)
    joint_positions = np.zeros((T, 7), dtype=np.float32)
    gripper_pose = np.zeros((T, 7), dtype=np.float32)
    gripper_open = np.zeros((T, 1), dtype=np.float32)
    gripper_touch_forces = np.zeros((T, 6), dtype=np.float32)

    for t, obs in enumerate(demo):
        joint_positions[t] = obs.joint_positions
        gripper_pose[t] = obs.gripper_pose
        gripper_open[t] = obs.gripper_open
        gripper_touch_forces[t] = obs.gripper_touch_forces

    # proprio = [joint_positions, gripper_pose, gripper_open]  (15D)
    proprio = np.concatenate([joint_positions, gripper_pose, gripper_open], axis=-1)

    # Compute 7D action_targets: [delta_pos(3), delta_rotvec(3), next_gripper(1)]
    action_targets = np.zeros((T, 7), dtype=np.float32)
    for t in range(T - 1):
        # Delta position
        action_targets[t, :3] = gripper_pose[t + 1, :3] - gripper_pose[t, :3]
        # Delta rotation: (R1 * R0.inv()).as_rotvec(), quaternion is xyzw
        r0 = Rotation.from_quat(gripper_pose[t, 3:7])
        r1 = Rotation.from_quat(gripper_pose[t + 1, 3:7])
        action_targets[t, 3:6] = (r1 * r0.inv()).as_rotvec()
        # Next gripper state
        action_targets[t, 6] = gripper_open[t + 1, 0]
    # Last frame copies from second-to-last
    action_targets[-1] = action_targets[-2] if T > 1 else 0.0

    # Phase IDs: no phase annotation, assign uniform 0
    phase_ids = np.zeros(T, dtype=np.int64)

    return {
        "joint_positions": joint_positions,
        "gripper_pose": gripper_pose,
        "gripper_open": gripper_open,
        "gripper_touch_forces": gripper_touch_forces,
        "proprio": proprio,
        "action_targets": action_targets,
        "phase_ids": phase_ids,
    }


def compute_dataset_statistics(all_actions, all_proprio, all_forces):
    """Compute q01/q99 for actions and proprio, mean/std for forces."""
    actions = np.concatenate(all_actions, axis=0)
    proprio = np.concatenate(all_proprio, axis=0)
    forces = np.concatenate(all_forces, axis=0)

    stats = {
        "rlbench": {
            "action": {
                "q01": np.percentile(actions, 1, axis=0).tolist(),
                "q99": np.percentile(actions, 99, axis=0).tolist(),
                "mean": np.mean(actions, axis=0).tolist(),
                "std": np.std(actions, axis=0).tolist(),
                "min": np.min(actions, axis=0).tolist(),
                "max": np.max(actions, axis=0).tolist(),
            },
            "proprio": {
                "q01": np.percentile(proprio, 1, axis=0).tolist(),
                "q99": np.percentile(proprio, 99, axis=0).tolist(),
                "mean": np.mean(proprio, axis=0).tolist(),
                "std": np.std(proprio, axis=0).tolist(),
            },
            "force": {
                "mean": np.mean(forces, axis=0).tolist(),
                "std": np.std(forces, axis=0).tolist(),
            },
        }
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/rlbench_data")
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    episodes_dir = Path(args.data_root) / args.task / "train" / "episodes"
    ep_dirs = sorted(
        [d for d in episodes_dir.iterdir() if d.is_dir() and d.name.startswith("episode")],
        key=lambda d: int(d.name.replace("episode", "")),
    )
    print(f"Found {len(ep_dirs)} episodes for task '{args.task}'")

    all_actions, all_proprio, all_forces = [], [], []
    processed = 0

    for ep_path in ep_dirs:
        result = process_episode(ep_path)
        if result is None:
            print(f"  Skipping {ep_path.name} (no low_dim_obs.pkl)")
            continue

        # Save model_inputs.npz
        np.savez(
            ep_path / "model_inputs.npz",
            joint_positions=result["joint_positions"],
            gripper_pose=result["gripper_pose"],
            gripper_open=result["gripper_open"],
            gripper_touch_forces=result["gripper_touch_forces"],
            proprio=result["proprio"],
            action_targets=result["action_targets"],
        )

        # Save phase_ids.npy
        np.save(ep_path / "phase_ids.npy", result["phase_ids"])

        all_actions.append(result["action_targets"])
        all_proprio.append(result["proprio"])
        all_forces.append(result["gripper_touch_forces"])
        processed += 1

        if processed % 10 == 0:
            print(f"  Processed {processed}/{len(ep_dirs)} episodes")

    print(f"Processed {processed} episodes total")

    # Compute and save dataset statistics
    if all_actions:
        stats = compute_dataset_statistics(all_actions, all_proprio, all_forces)
        stats_path = episodes_dir / "dataset_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved dataset_statistics.json to {stats_path}")


if __name__ == "__main__":
    main()
