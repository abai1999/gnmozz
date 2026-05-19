"""
Extract demo-derived stage handoff specs from RLBench official demos.

This is intentionally stage-spec oriented, not controller oriented: it estimates
when a task-stage is ready to hand control back to the planner, then writes a
task JSON consumed by StageTargetProvider.
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

from evaluate_rlbench import TASK_MAP, _lazy_import_tasks, resolve_live_target_handle
from prismatic.robot.stage_target_provider import apply_object_frame_offset, pose_delta_local_between


def _episode_sort_key(path: Path) -> tuple[int, str]:
    name = path.name.replace("episode", "")
    try:
        return int(name), path.name
    except ValueError:
        return 10**9, path.name


def _find_episode_dir(data_root: Path, task_name: str, split: str) -> Path:
    candidates = [
        data_root / task_name / split / "episodes",
        data_root / task_name / "all_variations" / split / "episodes",
        data_root / task_name / "episodes",
        data_root,
    ]
    for path in candidates:
        if path.exists() and any(path.glob("episode*/low_dim_obs.pkl")):
            return path
    raise FileNotFoundError(f"Could not find RLBench episodes under {data_root} for {task_name}/{split}")


def _first_close_step(actions: np.ndarray, threshold: float) -> int | None:
    for idx in range(int(actions.shape[0])):
        if float(actions[idx, 6]) <= float(threshold):
            return int(idx)
    return None


def _first_demo_close_step(demo_obs, threshold: float) -> int | None:
    for idx, obs in enumerate(demo_obs):
        if float(getattr(obs, "gripper_open", 1.0)) <= float(threshold):
            return int(idx)
    return None


def _object_to_gripper_offset(object_pose_7d: np.ndarray, gripper_pose_7d: np.ndarray) -> np.ndarray:
    object_pose = np.asarray(object_pose_7d, dtype=np.float32).reshape(7)
    gripper_pose = np.asarray(gripper_pose_7d, dtype=np.float32).reshape(7)
    r_obj = Rotation.from_quat(object_pose[3:7])
    r_grip = Rotation.from_quat(gripper_pose[3:7])
    rel_pos = r_obj.inv().apply(gripper_pose[:3] - object_pose[:3]).astype(np.float32)
    rel_quat = (r_obj.inv() * r_grip).as_quat().astype(np.float32)
    return np.concatenate([rel_pos, rel_quat], axis=0).astype(np.float32)


def _make_obs_config() -> ObservationConfig:
    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    return obs_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, default="data/rlbench_data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--episodes_dir", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--max_episodes", type=int, default=25)
    parser.add_argument("--close_open_threshold", type=float, default=0.5)
    parser.add_argument("--window_before_close", type=int, default=0)
    parser.add_argument("--window_after_close", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--threshold_scale", type=float, default=1.10)
    parser.add_argument("--min_xy_threshold", type=float, default=0.003)
    parser.add_argument("--min_z_threshold", type=float, default=0.001)
    parser.add_argument("--max_xy_threshold", type=float, default=0.020)
    parser.add_argument("--max_z_threshold", type=float, default=0.020)
    parser.add_argument("--allow_loose_thresholds", action="store_true", default=False)
    parser.add_argument("--disable_yaw_threshold", action="store_true", default=True)
    parser.add_argument("--enable_yaw_threshold", dest="disable_yaw_threshold", action="store_false")
    parser.add_argument("--substage_id", type=int, default=1)
    parser.add_argument("--target_role", type=str, default="pregrasp_close")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    episodes_dir = Path(args.episodes_dir) if args.episodes_dir else _find_episode_dir(data_root, args.task_name, args.split)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path("configs") / "stage_handoff_specs" / f"{args.task_name}.json"
    )

    _lazy_import_tasks()
    task_cls = TASK_MAP[args.task_name]
    env = Environment(
        MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()),
        obs_config=_make_obs_config(),
        headless=True,
    )
    env.launch()
    task = env.get_task(task_cls)

    close_records = []
    window_metrics = []
    used_eps = []
    episode_paths = sorted(episodes_dir.glob("episode*"), key=_episode_sort_key)
    try:
        for ep_path in episode_paths:
            ldo_path = ep_path / "low_dim_obs.pkl"
            npz_path = ep_path / "model_inputs.npz"
            if not ldo_path.exists() or not npz_path.exists():
                continue
            with ldo_path.open("rb") as f:
                demo_obs = pickle.load(f)
            npz_data = np.load(npz_path)
            close_step = None
            if "action_targets" in npz_data:
                action_targets = np.asarray(npz_data["action_targets"], dtype=np.float32)
                if action_targets.ndim == 2 and action_targets.shape[1] == 7:
                    close_step = _first_close_step(action_targets, args.close_open_threshold)
            if close_step is None:
                close_step = _first_demo_close_step(demo_obs, args.close_open_threshold)
            if close_step is None:
                continue
            _, obs = task.reset_to_demo(demo_obs)
            live_target_handle = resolve_live_target_handle(task)
            if live_target_handle is None:
                continue
            try:
                object_pose = np.asarray(live_target_handle.get_pose(), dtype=np.float32).reshape(7)
            except Exception:
                continue
            close_obs_idx = min(close_step, len(demo_obs) - 1)
            close_obs = demo_obs[close_obs_idx]
            close_gripper_pose = np.asarray(close_obs.gripper_pose, dtype=np.float32).reshape(7)
            close_records.append((object_pose, close_gripper_pose))

            start = max(0, close_step - int(args.window_before_close))
            end = min(len(demo_obs) - 1, close_step + int(args.window_after_close))
            offset = _object_to_gripper_offset(object_pose, close_gripper_pose)
            target_pose = apply_object_frame_offset(object_pose, offset)
            for obs_idx in range(start, end + 1):
                cur_obs = demo_obs[obs_idx]
                delta = pose_delta_local_between(np.asarray(cur_obs.gripper_pose, dtype=np.float32), target_pose)
                window_metrics.append(
                    {
                        "xy_error": float(np.linalg.norm(delta[:2])),
                        "abs_z_error": float(abs(delta[2])),
                        "yaw_error": float(abs(delta[5])),
                        "tilt_error": float(np.linalg.norm(delta[3:5])),
                    }
                )
            used_eps.append(ep_path.name)
            if len(used_eps) >= int(args.max_episodes):
                break
    finally:
        env.shutdown()

    if not window_metrics:
        raise RuntimeError(f"No usable handoff metrics extracted from {episodes_dir}")
    if not close_records:
        raise RuntimeError(f"No usable close offsets extracted from {episodes_dir}")
    offsets = np.stack([_object_to_gripper_offset(obj, grip) for obj, grip in close_records], axis=0)
    median_offset_pos = np.median(offsets[:, :3], axis=0).astype(np.float32)
    median_offset_quat = Rotation.from_quat(offsets[:, 3:7]).mean().as_quat().astype(np.float32)
    median_offset = np.concatenate([median_offset_pos, median_offset_quat], axis=0).astype(np.float32)

    def q_metric(name: str) -> float:
        values = np.asarray([m[name] for m in window_metrics], dtype=np.float32)
        return float(np.quantile(values, float(args.quantile)) * float(args.threshold_scale))

    xy_threshold = max(q_metric("xy_error"), float(args.min_xy_threshold))
    z_threshold = max(q_metric("abs_z_error"), float(args.min_z_threshold))
    yaw_threshold = -1.0 if bool(args.disable_yaw_threshold) else q_metric("yaw_error")
    tilt_threshold = -1.0
    if not bool(args.allow_loose_thresholds):
        if xy_threshold > float(args.max_xy_threshold) or z_threshold > float(args.max_z_threshold):
            raise RuntimeError(
                "Extracted handoff spec is too loose for safe runtime use: "
                f"xy={xy_threshold:.6f} max={float(args.max_xy_threshold):.6f}, "
                f"z={z_threshold:.6f} max={float(args.max_z_threshold):.6f}. "
                "This usually means the extraction target frame does not match the runtime provider target. "
                "Use --allow_loose_thresholds only for offline analysis, not runtime gating."
            )

    stage_spec = {
        "name": f"{args.task_name}_{args.target_role}_demo_q{int(args.quantile * 100)}",
        "task_name": str(args.task_name),
        "substage_id": int(args.substage_id),
        "target_frame": "object_grasp_target",
        "target_role": str(args.target_role),
        "target_offset_local_7d": median_offset.tolist(),
        "release_thresholds": {
            "xy_error": float(xy_threshold),
            "abs_z_error": float(z_threshold),
            "yaw_error": float(yaw_threshold),
            "tilt_error": float(tilt_threshold),
        },
        "optimization_thresholds": {
            "xy_error": float(xy_threshold),
            "abs_z_error": float(z_threshold),
            "yaw_error": float(yaw_threshold),
            "tilt_error": float(tilt_threshold),
        },
        "metric_thresholds": {
            "xy_error": float(xy_threshold),
            "abs_z_error": float(z_threshold),
            "yaw_error": float(yaw_threshold),
            "tilt_error": float(tilt_threshold),
        },
        "min_stable_frames": 1,
        "required_gripper_state": "open",
        "required_contact_state": "any",
        "source": "official_demo",
        "uses_privileged": True,
        "num_episodes_used": int(len(used_eps)),
        "num_metric_rows": int(len(window_metrics)),
        "quantile": float(args.quantile),
        "threshold_scale": float(args.threshold_scale),
        "episodes_dir": str(episodes_dir),
        "episode_names": used_eps[:50],
    }
    output = {
        "task_name": str(args.task_name),
        "version": 1,
        "description": "Demo-derived task-stage handoff thresholds. Alignment uses these as provider-owned handoff labels.",
        "stages": [stage_spec],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2))
    print(json.dumps(stage_spec, indent=2))


if __name__ == "__main__":
    main()
