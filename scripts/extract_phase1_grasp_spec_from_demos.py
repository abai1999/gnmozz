"""
Extract a task-level phase-1 grasp-frame spec from RLBench demos.

The script replays demos until the first expert close, reads privileged object and
gripper poses, and estimates an object-frame grasp offset plus a pregrasp hover
offset suitable for privileged teacher construction.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
os.environ.setdefault("COPPELIASIM_ROOT", os.path.expanduser("~/CoppeliaSim"))
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.environ["COPPELIASIM_ROOT"])
os.environ.setdefault("QT_PLUGIN_PATH", os.environ["COPPELIASIM_ROOT"])
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from evaluate_rlbench import TASK_MAP, _lazy_import_tasks, delta_to_absolute, resolve_live_target_handle


def transform_object_to_gripper(object_pose_7d: np.ndarray, gripper_pose_7d: np.ndarray) -> np.ndarray:
    object_pose = np.asarray(object_pose_7d, dtype=np.float32).reshape(7)
    gripper_pose = np.asarray(gripper_pose_7d, dtype=np.float32).reshape(7)
    r_obj = Rotation.from_quat(object_pose[3:7])
    r_grip = Rotation.from_quat(gripper_pose[3:7])
    rel_pos = r_obj.inv().apply(gripper_pose[:3] - object_pose[:3]).astype(np.float32)
    rel_quat = (r_obj.inv() * r_grip).as_quat().astype(np.float32)
    return np.concatenate([rel_pos, rel_quat], axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--episodes_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--close_open_threshold", type=float, default=0.5)
    parser.add_argument("--hover_offset_z", type=float, default=0.01)
    args = parser.parse_args()

    _lazy_import_tasks()
    task_cls = TASK_MAP[args.task_name]
    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    env = Environment(
        MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()),
        obs_config=obs_config,
        headless=True,
    )
    env.launch()
    task = env.get_task(task_cls)

    offsets = []
    episode_paths = sorted(Path(args.episodes_dir).glob("episode*"))[: args.max_episodes]
    for ep_path in episode_paths:
        ldo_path = ep_path / "low_dim_obs.pkl"
        model_npz = ep_path / "model_inputs.npz"
        if not ldo_path.exists() or not model_npz.exists():
            continue
        with open(ldo_path, "rb") as f:
            demo_obs = pickle.load(f)
        npz_data = np.load(model_npz)
        descs, obs = task.reset_to_demo(demo_obs)
        live_target_handle = resolve_live_target_handle(task)
        if live_target_handle is None:
            continue
        T = int(npz_data["action_targets"].shape[0])
        close_step = None
        for t in range(max(T - 1, 1)):
            if float(npz_data["action_targets"][t, 6]) <= float(args.close_open_threshold):
                close_step = t
                break
        if close_step is None:
            continue
        for t in range(close_step):
            abs_action = delta_to_absolute(np.asarray(npz_data["action_targets"][t], dtype=np.float32), obs.gripper_pose)
            obs, _, terminate = task.step(abs_action)
            if terminate:
                break
        try:
            object_pose = np.asarray(live_target_handle.get_pose(), dtype=np.float32).reshape(7)
        except Exception:
            continue
        offsets.append(transform_object_to_gripper(object_pose, np.asarray(obs.gripper_pose, dtype=np.float32)))

    env.shutdown()
    if not offsets:
        raise RuntimeError("No usable demo offsets extracted.")
    offsets_arr = np.stack(offsets, axis=0)
    median_pos = np.median(offsets_arr[:, :3], axis=0).astype(np.float32)
    median_quat = Rotation.from_quat(offsets_arr[:, 3:7]).mean().as_quat().astype(np.float32)
    spec = {
        "name": f"{args.task_name}_phase1_demo_extracted",
        "grasp_offset_local_7d": np.concatenate([median_pos, median_quat], axis=0).astype(np.float32).tolist(),
        "pregrasp_hover_offset_local_6d": [0.0, 0.0, float(args.hover_offset_z), 0.0, 0.0, 0.0],
        "close_xy_threshold": 0.02,
        "close_abs_z_threshold": 0.02,
        "close_yaw_threshold": 0.12,
        "orientation_rescue_xy_threshold": 0.005,
        "orientation_rescue_angle_threshold_deg": 5.0,
        "commit_switch_xy_threshold": 0.01,
        "commit_switch_z_threshold": 0.02,
        "commit_switch_yaw_threshold": 0.12,
        "num_episodes_used": int(offsets_arr.shape[0]),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
