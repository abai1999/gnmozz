"""
collect_dagger_rollouts.py

Collect planner-state DAgger-lite samples with generic stage/event annotations.
"""

import argparse
import json
import os
import pickle
import sys
from collections import deque
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
_COPPELIASIM_ROOT = os.path.expanduser("~/CoppeliaSim")
_CONDA_LIBSTDCXX = os.path.expanduser("~/my_conda_envs/vla-adapter/lib/libstdc++.so.6")
os.environ.setdefault("COPPELIASIM_ROOT", _COPPELIASIM_ROOT)
_base_ld = f"{_COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
_existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
if not _existing_ld:
    os.environ["LD_LIBRARY_PATH"] = _base_ld
elif _COPPELIASIM_ROOT not in _existing_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{_base_ld}:{_existing_ld}"
os.environ.setdefault("LD_PRELOAD", _CONDA_LIBSTDCXX)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _COPPELIASIM_ROOT)
os.environ.setdefault("QT_PLUGIN_PATH", _COPPELIASIM_ROOT)
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
_HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/mnt/ssd/guoning/hf-cache")
os.environ.setdefault("HF_HOME", _HF_CACHE_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from rlbench.backend.exceptions import InvalidActionError

from prismatic.robot.stage_aware_refiner import StageAwareRefiner
from prismatic.robot.stage_manager import FailureMode, StageManager, StagePhase
from scripts.collect_residual_data import load_planner
from scripts.collect_stage_refiner_data import make_env, maybe_reset_to_subgoal, reconstruct_expert_action_7d
from scripts.evaluate_rlbench import TASK_MAP, _lazy_import_tasks, delta_to_absolute, predict_actions, process_obs


def event_code_from_state(phase_id: int, transitioned: bool, failure_mode: int) -> int:
    if failure_mode != int(FailureMode.NONE) or phase_id >= int(StagePhase.RECOVER):
        return 3
    if transitioned:
        return 2
    if phase_id == int(StagePhase.ALIGN):
        return 0
    if phase_id >= int(StagePhase.INTERACT):
        return 1
    return 4


def summarize_phase_distribution(values):
    if len(values) == 0:
        return {}
    arr = np.asarray(values, dtype=np.int64)
    return {int(k): int((arr == k).sum()) for k in np.unique(arr)}


def flush_dagger_shard(output_dir, shard_idx, buffers):
    shard_path = output_dir / f"dagger_shard_{shard_idx:04d}.npz"
    phase_summary = summarize_phase_distribution(buffers["phase_id"])
    event_summary = summarize_phase_distribution(buffers["event_code"])
    failure_summary = summarize_phase_distribution(buffers["failure_mode"])
    np.savez_compressed(
        shard_path,
        front_rgb=np.asarray(buffers["front_rgb"], dtype=np.uint8),
        wrist_rgb=np.asarray(buffers["wrist_rgb"], dtype=np.uint8),
        wrist_depth=np.asarray(buffers["wrist_depth"], dtype=np.float32),
        force_history=np.asarray(buffers["force_history"], dtype=np.float32),
        proprio=np.asarray(buffers["proprio"], dtype=np.float32),
        action_chunk=np.asarray(buffers["action_chunk"], dtype=np.float32),
        language=np.asarray(buffers["language"], dtype=object),
        phase_id=np.asarray(buffers["phase_id"], dtype=np.int64),
        step_idx=np.asarray(buffers["step_idx"], dtype=np.int64),
        event_code=np.asarray(buffers["event_code"], dtype=np.int64),
        transition_flag=np.asarray(buffers["transition_flag"], dtype=np.int64),
        failure_mode=np.asarray(buffers["failure_mode"], dtype=np.int64),
        subgoal_progress=np.asarray(buffers["subgoal_progress"], dtype=np.float32),
    )
    print(f"  [dagger shard {shard_idx}] saved {len(buffers['action_chunk'])} samples -> {shard_path}")
    print(f"    phase_counts={phase_summary} event_counts={event_summary} failure_counts={failure_summary}")
    total = max(len(buffers["phase_id"]), 1)
    dominant = max(phase_summary.values(), default=0) / total
    if dominant > 0.9:
        print(f"    WARNING: shard {shard_idx} is phase-skewed (dominant_fraction={dominant:.3f})")


def normalize_action(action, q01, q99):
    action = action.astype(np.float32).copy()
    mask = (q99 - q01) > 1e-8
    out = np.zeros_like(action)
    out[mask] = 2.0 * (action[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
    return np.clip(out, -1.0, 1.0)


def make_recovery_action_chunk(base_action, force_reading, q01, q99, chunk_len=8):
    """Build a safety-oriented target chunk for invalid-action hard negatives."""
    hold = np.zeros_like(q01, dtype=np.float32)
    recovery = hold.copy()
    if base_action is not None and len(base_action) > 6:
        hold[6] = float(base_action[6])
        recovery[6] = float(base_action[6])

    if force_reading is not None:
        force = np.asarray(force_reading, dtype=np.float32)
        fxyz = force[:3]
        tau = force[3:6] if force.shape[0] >= 6 else np.zeros(3, dtype=np.float32)
        fxy = fxyz[:2]
        lateral_mag = float(np.linalg.norm(fxy))
        if lateral_mag > 2.0:
            recovery[:2] = (-fxy / max(lateral_mag, 1e-8)) * 0.002
        if abs(float(fxyz[2])) > 2.5:
            recovery[2] = 0.0015
        if np.linalg.norm(tau) > 1.5 and abs(float(tau[2])) > 1e-6:
            recovery[5] = -np.sign(float(tau[2])) * 0.02

    recovery[:3] = np.clip(recovery[:3], -0.006, 0.006)
    recovery[3:6] = np.clip(recovery[3:6], -0.03, 0.03)
    chunk = np.zeros((chunk_len, q01.shape[0]), dtype=np.float32)
    chunk[:] = normalize_action(hold, q01, q99)
    chunk[0] = normalize_action(recovery, q01, q99)
    return chunk


def main():
    parser = argparse.ArgumentParser(description="Collect planner-state DAgger rollouts")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--data_root", type=str, default="data/rlbench_data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vlm_path", type=str, default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", type=str, default="pretrained_models/configs/config.json")
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--no_depth", dest="use_depth", action="store_false")
    parser.add_argument("--use_force", action="store_true", default=False)
    parser.add_argument("--no_force", dest="use_force", action="store_false")
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.set_defaults(planner_use_depth=None, planner_use_force=None)
    parser.add_argument("--subgoal_reset_mode", type=str, default="none", choices=["none", "demo_subgoal", "oracle_subgoal"])
    parser.add_argument("--subgoal_stage", type=str, default="ALIGN", choices=["ALIGN", "INTERACT"])
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--max_rollout_steps", type=int, default=200)
    parser.add_argument(
        "--use_safety_supervisor",
        action="store_true",
        default=False,
        help="Roll out planner actions through StageAwareRefiner(mode=safety_only).",
    )
    args = parser.parse_args()

    if args.planner_use_depth is None:
        args.planner_use_depth = False
    if args.planner_use_force is None:
        args.planner_use_force = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _lazy_import_tasks()
    assert args.task_name in TASK_MAP
    vla, processor, action_head, proprio_projector, norm_stats = load_planner(
        args.checkpoint_dir,
        args.vlm_path,
        args.config_path,
        args.planner_use_depth,
        args.planner_use_force,
    )
    rlbench_stats = norm_stats.get("rlbench", norm_stats)
    q01 = np.asarray(rlbench_stats["action"]["q01"], dtype=np.float32)
    q99 = np.asarray(rlbench_stats["action"]["q99"], dtype=np.float32)

    episodes_dir = Path(args.data_root) / args.task_name / "train" / "episodes"
    ep_dirs = sorted([d for d in os.listdir(episodes_dir) if d.startswith("episode")], key=lambda x: int(x.replace("episode", "")))
    if args.max_episodes > 0:
        ep_dirs = ep_dirs[:args.max_episodes]
    env, task = make_env(args.task_name, Path(args.data_root))
    target_stage = StagePhase[args.subgoal_stage]
    safety_refiner = StageAwareRefiner(mode="safety_only") if args.use_safety_supervisor else None

    buffers = {k: [] for k in [
        "front_rgb", "wrist_rgb", "wrist_depth", "proprio", "action_chunk", "language",
        "force_history",
        "phase_id", "step_idx", "event_code", "transition_flag", "failure_mode", "subgoal_progress"
    ]}
    shard_idx = 0
    total_samples = 0
    progress_sum = 0.0
    phase_counts = {int(phase): 0 for phase in StagePhase}
    event_counts = {}
    failure_counts = {}

    def append_sample(
        front_pil,
        wrist_pil,
        depth_tensor_96,
        proprio,
        npz_data,
        rollout_step,
        instruction,
        phase,
        transitioned,
        failure_mode,
        step_idx,
        subgoal_progress,
        action_chunk_override=None,
    ):
        nonlocal total_samples, progress_sum
        buffers["front_rgb"].append(np.asarray(front_pil, dtype=np.uint8))
        buffers["wrist_rgb"].append(np.asarray(wrist_pil, dtype=np.uint8))
        buffers["wrist_depth"].append(
            depth_tensor_96.numpy().astype(np.float32)
            if depth_tensor_96 is not None else np.zeros((1, 96, 96), dtype=np.float32)
        )
        buffers["force_history"].append(
            force_hist.numpy().astype(np.float32)
            if force_hist is not None else np.zeros((32, 6), dtype=np.float32)
        )
        buffers["proprio"].append(proprio.astype(np.float32))
        if action_chunk_override is not None:
            action_chunk = np.asarray(action_chunk_override, dtype=np.float32)
        else:
            action_chunk = np.zeros((8, q01.shape[0]), dtype=np.float32)
            for j in range(8):
                t = min(rollout_step + j, T - 1)
                action_chunk[j] = normalize_action(reconstruct_expert_action_7d(npz_data, t), q01, q99)
        buffers["action_chunk"].append(action_chunk)
        buffers["language"].append(instruction)
        buffers["phase_id"].append(int(phase))
        buffers["step_idx"].append(step_idx)
        event_code = event_code_from_state(int(phase), transitioned, failure_mode)
        buffers["event_code"].append(event_code)
        buffers["transition_flag"].append(int(transitioned))
        buffers["failure_mode"].append(failure_mode)
        buffers["subgoal_progress"].append(float(subgoal_progress))
        total_samples += 1
        progress_sum += float(subgoal_progress)
        phase_counts[int(phase)] = phase_counts.get(int(phase), 0) + 1
        event_counts[event_code] = event_counts.get(event_code, 0) + 1
        failure_counts[failure_mode] = failure_counts.get(failure_mode, 0) + 1

    try:
        for ep_name in ep_dirs:
            ep_path = episodes_dir / ep_name
            npz_path = ep_path / "model_inputs.npz"
            if not npz_path.exists():
                continue
            npz_data = dict(np.load(npz_path))
            T = npz_data["action_targets"].shape[0]
            low_dim_obs_path = ep_path / "low_dim_obs.pkl"
            if not low_dim_obs_path.exists():
                continue
            with open(low_dim_obs_path, "rb") as f:
                demo_obs = pickle.load(f)
            descs, obs, rollout_offset = maybe_reset_to_subgoal(
                task, demo_obs, ep_path, npz_data, target_stage, args.subgoal_reset_mode
            )
            instruction = descs[0] if isinstance(descs, list) else str(descs)

            force_buffer = deque(maxlen=256)
            action_queue = []
            chunk_step = 0
            if safety_refiner is not None:
                safety_refiner.reset()
                manager = safety_refiner.manager
            else:
                manager = StageManager()
            prev_phase = manager.phase
            for rollout_step in range(rollout_offset, min(T, args.max_rollout_steps)):
                front_pil, wrist_pil, proprio, depth_tensor_224, force_hist, depth_tensor_96, raw_force = process_obs(
                    obs,
                    norm_stats,
                    force_buffer,
                    use_depth=args.use_depth,
                    use_force=args.use_force,
                    depth_max=1.0,
                )
                if len(action_queue) == 0:
                    actions = predict_actions(
                        vla, processor, action_head, proprio_projector,
                        front_pil, wrist_pil, proprio,
                        depth_tensor_224 if args.planner_use_depth else None,
                        force_hist if args.planner_use_force else None,
                        instruction,
                        unnorm_key="rlbench",
                    )
                    max_chunk = safety_refiner.get_chunk_size() if safety_refiner is not None else len(actions)
                    action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), max_chunk))]
                    chunk_step = 0

                base_action = action_queue.pop(0)
                if safety_refiner is not None:
                    exec_action = safety_refiner.step(
                        a_base_7d=base_action,
                        step_idx=chunk_step,
                        force_reading=raw_force,
                        gripper_z=float(obs.gripper_pose[2]),
                        wrist_depth=depth_tensor_96,
                        ft_hist=force_hist,
                        proprio=proprio,
                        gripper_pose=obs.gripper_pose,
                        gripper_open=float(obs.gripper_open),
                    )
                    phase = manager.phase
                    transitioned = manager.last_transitioned
                else:
                    exec_action = base_action
                    depth_prox = None if depth_tensor_96 is None else float(np.percentile(depth_tensor_96.numpy(), 5.0))
                    phase = manager.update(
                        force_reading=raw_force,
                        gripper_pose=obs.gripper_pose,
                        gripper_open=float(obs.gripper_open),
                        depth_proximity=depth_prox,
                        base_action=base_action,
                    )
                    transitioned = phase != prev_phase
                prev_phase = phase
                stalled = manager.no_progress_steps >= manager.no_progress_window
                failure_mode = int(manager.failure_mode)
                keep_sample = (
                    phase in (StagePhase.ALIGN, StagePhase.INTERACT, StagePhase.RECOVER)
                    or transitioned
                    or failure_mode != int(FailureMode.NONE)
                    or stalled
                )
                if keep_sample:
                    append_sample(
                        front_pil, wrist_pil, depth_tensor_96, proprio, npz_data, rollout_step,
                        instruction, phase, transitioned, failure_mode, chunk_step,
                        manager.get_subgoal_progress(),
                    )

                abs_action = delta_to_absolute(exec_action, obs.gripper_pose)
                if safety_refiner is not None:
                    abs_action[:3] = safety_refiner.safety.clamp_workspace(abs_action[:3])
                try:
                    obs, reward, terminate = task.step(abs_action)
                except InvalidActionError as exc:
                    if safety_refiner is not None:
                        recovery_delta = safety_refiner.on_invalid_action(base_action, raw_force)
                        hard_phase = safety_refiner.manager.phase
                    else:
                        hard_phase = manager.note_invalid_action()
                        recovery_delta = None
                    prev_phase = hard_phase
                    recovery_chunk = make_recovery_action_chunk(base_action, raw_force, q01, q99)
                    append_sample(
                        front_pil, wrist_pil, depth_tensor_96, proprio, npz_data, rollout_step,
                        instruction, hard_phase, True, int(FailureMode.INVALID_ACTION), chunk_step,
                        manager.get_subgoal_progress(),
                        action_chunk_override=recovery_chunk,
                    )
                    print(f"  [dagger hard-negative] {ep_name} step={rollout_step}: InvalidActionError -> RECOVER ({exc})")
                    action_queue.clear()
                    chunk_step = 0
                    if recovery_delta is not None:
                        recovery_abs = delta_to_absolute(recovery_delta, obs.gripper_pose)
                        recovery_abs[:3] = safety_refiner.safety.clamp_workspace(recovery_abs[:3])
                        try:
                            obs, reward, terminate = task.step(recovery_abs)
                        except InvalidActionError:
                            pass
                    if len(buffers["action_chunk"]) >= args.shard_size:
                        flush_dagger_shard(output_dir, shard_idx, buffers)
                        shard_idx += 1
                        for key in buffers:
                            buffers[key].clear()
                    continue
                chunk_step += 1
                if safety_refiner is not None and safety_refiner.should_replan():
                    safety_refiner.note_replan()
                    action_queue.clear()
                    chunk_step = 0
                if reward > 0 or terminate:
                    break

                if len(buffers["action_chunk"]) >= args.shard_size:
                    flush_dagger_shard(output_dir, shard_idx, buffers)
                    shard_idx += 1
                    for key in buffers:
                        buffers[key].clear()
            print(f"[dagger_collect] {ep_name}: total_samples={total_samples}")
    finally:
        env.shutdown()

    if len(buffers["action_chunk"]) > 0:
        flush_dagger_shard(output_dir, shard_idx, buffers)

    meta = {
        "checkpoint_dir": args.checkpoint_dir,
        "task_name": args.task_name,
        "subgoal_reset_mode": args.subgoal_reset_mode,
        "subgoal_stage": args.subgoal_stage,
        "rollout_policy": "planner+safety" if args.use_safety_supervisor else "planner_only",
        "total_samples": total_samples,
        "phase_counts": phase_counts,
        "event_counts": event_counts,
        "failure_counts": failure_counts,
        "avg_subgoal_progress": progress_sum / max(total_samples, 1),
    }
    with open(output_dir / "dagger_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
