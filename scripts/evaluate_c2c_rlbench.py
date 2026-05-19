#!/usr/bin/env python
"""Standalone RLBench evaluator for Coarse2Contact."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

_HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/mnt/ssd/guoning/hf-cache")
os.environ.setdefault("HF_HOME", _HF_CACHE_ROOT)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_CACHE_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_HF_CACHE_ROOT, "transformers"))
os.environ.setdefault("TORCH_HOME", os.path.join(_HF_CACHE_ROOT, "torch"))
os.environ.setdefault("TIMM_HOME", os.path.join(_HF_CACHE_ROOT, "timm"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    from moviepy import ImageSequenceClip
from PIL import Image
from scipy.spatial.transform import Rotation

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_utils import no_init_weights

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.action_heads_paper_faithful import PaperFaithfulL1RegressionActionHead
from prismatic.models.load import load as load_vlm
from prismatic.models.projectors import ProprioProjector
from prismatic.robot.coarse2contact import Coarse2ContactSupervisor
from prismatic.robot.residual_transforms import world_delta_to_local
from prismatic.vla.constants import ACTION_DIM, FORCE_DIM, FORCE_HISTORY_LEN, PROPRIO_DIM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TASK_MAP: dict[str, object] = {}


def _jsonable_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(v) for v in value]
    return value


def _lazy_import_tasks() -> None:
    global TASK_MAP
    if TASK_MAP:
        return
    from rlbench.tasks import InsertOntoSquarePeg, PlugChargerInPowerSupply, PushButton, StackBlocks

    TASK_MAP.update(
        {
            "insert_onto_square_peg": InsertOntoSquarePeg,
            "stack_blocks": StackBlocks,
            "plug_charger_in_power_supply": PlugChargerInPowerSupply,
            "push_button": PushButton,
        }
    )


def _rename_state_dict(state_dict):
    replace_map = [
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"),
        ("gamma", "scale_factor"),
    ]
    new_sd = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replace_map:
            if old in new_key:
                new_key = new_key.replace(old, new)
        new_sd[new_key] = value
    return new_sd


def _clean_state_dict(state_dict):
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint(
    checkpoint_dir,
    vlm_path="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
    config_path="pretrained_models/configs/config.json",
    use_depth=True,
    use_force=True,
):
    checkpoint_dir = Path(checkpoint_dir)
    print(f"[c2c] Loading checkpoint from {checkpoint_dir}", flush=True)
    load_t0 = time.perf_counter()

    def _mark(label: str) -> None:
        print(f"[c2c][load] {label}: {time.perf_counter() - load_t0:.1f}s", flush=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    config = AutoConfig.from_pretrained(config_path, local_files_only=True, trust_remote_code=False)
    with no_init_weights():
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
    _mark("built eval shell")

    merged_eval_state = checkpoint_dir / "merged_eval_vla_state.pt"
    if merged_eval_state.exists() and os.environ.get("EVAL_USE_MERGED_STATE", "1").lower() not in {"0", "false", "no", "off"}:
        merged_state = torch.load(merged_eval_state, map_location="cpu", weights_only=True)
        state_dict = merged_state.get("model_state_dict", merged_state) if isinstance(merged_state, dict) else merged_state
        vla.load_state_dict(_clean_state_dict(state_dict), strict=False)
        _mark("loaded merged eval state")
    else:
        vlm = load_vlm(vlm_path, hf_token="", load_for_training=False)
        _mark("loaded base vlm")
        vla.load_state_dict(_rename_state_dict(vlm.state_dict()), strict=False)
        del vlm
        _mark("copied base weights")
        lora_adapter_dir = checkpoint_dir / "lora_adapter"
        if lora_adapter_dir.exists():
            vla = PeftModel.from_pretrained(vla, str(lora_adapter_dir), local_files_only=True)
            _mark("loaded lora")
            if os.environ.get("EVAL_MERGE_LORA", "0").lower() in {"1", "true", "yes", "on"}:
                vla = vla.merge_and_unload()
                _mark("merged lora")

    vla_core = getattr(getattr(vla, "base_model", None), "model", vla)
    aq_paths = sorted(glob.glob(str(checkpoint_dir / "action_queries--*.pt")))
    if not aq_paths:
        raise FileNotFoundError(f"No action_queries checkpoint found in {checkpoint_dir}")
    aq_state = torch.load(aq_paths[0], map_location="cpu", weights_only=True)
    vla_core.action_queries.weight.data.copy_(aq_state["action_queries.weight"])
    vla_core.vision_backbone.set_num_images_in_input(2)
    vla = vla.to(torch.bfloat16).to(DEVICE).eval()
    _mark("moved vla to device")

    stats_path = checkpoint_dir / "dataset_statistics.json"
    with open(stats_path) as handle:
        norm_stats = json.load(handle)
    vla.norm_stats = norm_stats
    vla_core.norm_stats = norm_stats

    processor = AutoProcessor.from_pretrained(str(checkpoint_dir), trust_remote_code=True, local_files_only=True)
    _mark("loaded processor")

    pp_paths = sorted(glob.glob(str(checkpoint_dir / "proprio_projector--*checkpoint.pt")))
    if not pp_paths:
        raise FileNotFoundError(f"No proprio_projector checkpoint found in {checkpoint_dir}")
    llm_dim = int(getattr(vla_core, "llm_dim"))
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
    proprio_projector.load_state_dict(_clean_state_dict(torch.load(pp_paths[0], map_location="cpu", weights_only=True)))
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE).eval()
    _mark("loaded proprio projector")

    ah_paths = sorted(glob.glob(str(checkpoint_dir / "action_head--*checkpoint.pt")))
    if not ah_paths:
        raise FileNotFoundError(f"No action_head checkpoint found in {checkpoint_dir}")
    planner_head_cfg_path = checkpoint_dir / "planner_head_config.json"
    planner_core_variant = "current"
    if planner_head_cfg_path.exists():
        try:
            planner_core_variant = str(json.loads(planner_head_cfg_path.read_text()).get("planner_core_variant", "current"))
        except Exception:
            planner_core_variant = "current"
    action_head_cls = PaperFaithfulL1RegressionActionHead if planner_core_variant == "paper_faithful" else L1RegressionActionHead
    action_head = action_head_cls(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=ACTION_DIM,
        use_pro_version=True,
        use_depth=use_depth,
        use_force=use_force,
    )
    action_head.load_state_dict(_clean_state_dict(torch.load(ah_paths[0], map_location="cpu", weights_only=True)), strict=False)
    action_head = action_head.to(torch.bfloat16).to(DEVICE).eval()
    _mark("loaded action head")
    return vla, processor, action_head, proprio_projector, norm_stats


def _normalize_depth_array(depth, depth_max=1.0):
    if depth is None:
        return None
    depth_arr = np.asarray(depth, dtype=np.float32).copy()
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[:, :, 0]
    depth_norm = np.clip(depth_arr / float(depth_max), 0.0, 1.0)
    return depth_norm.astype(np.float32)


def _depth_tensor_from_norm(depth_norm, size):
    if depth_norm is None:
        return None
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_pil = Image.fromarray(depth_uint8, mode="L").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.array(depth_pil, dtype=np.float32) / 255.0).unsqueeze(0)


def process_obs(obs, norm_stats, force_buffer, use_depth=True, use_force=True, depth_max=1.0):
    front_pil = Image.fromarray(obs.front_rgb).convert("RGB")
    wrist_pil = Image.fromarray(obs.wrist_rgb).convert("RGB")
    proprio = np.concatenate([obs.joint_positions, obs.gripper_pose, [float(obs.gripper_open)]]).astype(np.float32)

    depth_tensor_224 = None
    if use_depth and obs.wrist_depth is not None:
        depth_norm = _normalize_depth_array(obs.wrist_depth, depth_max=depth_max)
        depth_tensor_224 = _depth_tensor_from_norm(depth_norm, 224)

    force_history = None
    raw_force = None
    if use_force:
        raw_force = (
            obs.gripper_touch_forces.astype(np.float32)
            if obs.gripper_touch_forces is not None
            else np.zeros(FORCE_DIM, dtype=np.float32)
        )
        force_buffer.append(raw_force)
        history = np.zeros((FORCE_HISTORY_LEN, FORCE_DIM), dtype=np.float32)
        buf = list(force_buffer)
        for i in range(FORCE_HISTORY_LEN):
            idx = len(buf) - FORCE_HISTORY_LEN + i
            if idx >= 0:
                history[i] = buf[idx]
        rlbench_stats = norm_stats.get("rlbench", norm_stats)
        if "force" in rlbench_stats:
            force_mean = np.array(rlbench_stats["force"]["mean"], dtype=np.float32)
            force_std = np.maximum(np.array(rlbench_stats["force"]["std"], dtype=np.float32), 1e-6)
            history = (history - force_mean) / force_std
        force_history = torch.from_numpy(history)
    return front_pil, wrist_pil, proprio, depth_tensor_224, force_history, raw_force


def predict_actions(
    vla,
    processor,
    action_head,
    proprio_projector,
    front_pil,
    wrist_pil,
    proprio,
    depth_tensor,
    force_history,
    instruction,
    unnorm_key="rlbench",
):
    prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
        "You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nWhat action should the robot take to {instruction.lower()}?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs_front = processor(prompt, front_pil).to(DEVICE, dtype=torch.bfloat16)
    inputs_wrist = processor(prompt, wrist_pil).to(DEVICE, dtype=torch.bfloat16)
    inputs_front["pixel_values"] = torch.cat([inputs_front["pixel_values"], inputs_wrist["pixel_values"]], dim=1)
    depth = depth_tensor.unsqueeze(0).to(DEVICE, dtype=torch.bfloat16) if depth_tensor is not None else None
    force = force_history.unsqueeze(0).to(DEVICE, dtype=torch.bfloat16) if force_history is not None else None
    with torch.no_grad():
        actions, _ = vla.predict_action(
            **inputs_front,
            unnorm_key=unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            depth=depth,
            force_history=force,
            use_film=False,
        )
    return actions


def delta_to_absolute(delta_action, current_gripper_pose):
    delta_pos = delta_action[:3]
    delta_rv = delta_action[3:6]
    gripper_raw = delta_action[6]
    current_pos = current_gripper_pose[:3]
    current_quat = current_gripper_pose[3:7]
    new_pos = current_pos + delta_pos
    new_quat = (Rotation.from_rotvec(delta_rv) * Rotation.from_quat(current_quat)).as_quat()
    gripper_cmd = 1.0 if gripper_raw > 0.5 else 0.0
    return np.concatenate([new_pos, new_quat, [gripper_cmd]]).astype(np.float32)


def compute_wrist_visibility_stats(
    wrist_depth: np.ndarray | None,
    *,
    near_threshold: float = 0.04,
    occluded_valid_ratio_threshold: float = 0.60,
    occluded_near_fraction_threshold: float = 0.45,
):
    if wrist_depth is None:
        return 0.0, 1.0, True, True
    depth = np.asarray(wrist_depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    finite_mask = np.logical_and(np.isfinite(depth), depth > 1.0e-6)
    valid_ratio = float(np.mean(finite_mask)) if depth.size > 0 else 0.0
    near_fraction = float(np.mean(np.logical_and(finite_mask, depth < float(near_threshold)))) if depth.size > 0 else 1.0
    is_occluded = bool(valid_ratio < float(occluded_valid_ratio_threshold) or near_fraction > float(occluded_near_fraction_threshold))
    is_low_visibility = bool(valid_ratio < float(occluded_valid_ratio_threshold + 0.10) or near_fraction > float(max(0.30, occluded_near_fraction_threshold - 0.10)))
    return valid_ratio, near_fraction, is_occluded, is_low_visibility


def maybe_apply_workspace_filter(abs_action, safety, mode="diagnostic", tolerance=0.0):
    if safety is None:
        return abs_action, 0.0
    violation = safety.workspace_violation(abs_action[:3])
    should_clamp = mode == "hard" or (mode == "tolerance" and violation > float(tolerance))
    if should_clamp:
        abs_action = abs_action.copy()
        abs_action[:3] = safety.clamp_workspace(abs_action[:3])
    return abs_action, violation


def build_c2c_supervisor(args):
    if args.mode == "planner_only":
        return None
    return Coarse2ContactSupervisor(
        mode="depth_shadow" if args.mode == "depth_shadow" else args.mode,
        shadow_only=bool(args.shadow_only or args.mode == "depth_shadow"),
        visual_xy_threshold=float(args.coarse2contact_visual_xy_threshold),
        visual_yaw_threshold=float(args.coarse2contact_visual_yaw_threshold),
        visual_precontact_depth_threshold=float(args.coarse2contact_precontact_depth_threshold),
        visual_contact_depth_threshold=float(args.coarse2contact_contact_depth_threshold),
        max_xy_step=float(args.coarse2contact_max_xy_step),
        max_yaw_step=float(args.coarse2contact_max_yaw_step),
        force_contact_threshold=float(args.coarse2contact_force_contact_threshold),
        force_delta_contact_threshold=float(args.coarse2contact_force_delta_contact_threshold),
        force_jam_threshold=float(args.coarse2contact_force_jam_threshold),
        force_torque_threshold=float(args.coarse2contact_force_torque_threshold),
        force_spike_threshold=float(args.coarse2contact_force_spike_threshold),
        backoff_m=float(args.coarse2contact_backoff_m),
        lateral_m=float(args.coarse2contact_lateral_m),
        chunk_size=int(args.coarse2contact_chunk_size),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Coarse2Contact RLBench evaluator")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--mode", type=str, default="planner_only", choices=["planner_only", "depth_shadow", "depth_apply", "force_reflex", "depth_force"])
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--episode_indices", type=str, default="5,8,19")
    parser.add_argument("--max_steps", type=int, default=320)
    parser.add_argument("--eval_seed", type=int, default=3407)
    parser.add_argument("--depth_max", type=float, default=1.0)
    parser.add_argument("--output_root", type=str, default="runtime_artifacts/coarse2contact/eval_3ep")
    parser.add_argument("--name_suffix", type=str, default="coarse2contact_eval_3ep")
    parser.add_argument("--shadow_only", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="record_video", action="store_false")
    parser.add_argument("--write_episode_videos", action="store_true", default=True)
    parser.add_argument("--no_episode_videos", dest="write_episode_videos", action="store_false")
    parser.add_argument("--record_gripper_trace", action="store_true", default=True)
    parser.add_argument("--no_gripper_trace", dest="record_gripper_trace", action="store_false")
    parser.add_argument("--no_best_gif", action="store_true", default=True)
    parser.add_argument("--coarse2contact_shadow_only", action="store_true", default=False)
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.set_defaults(planner_use_depth=False, planner_use_force=False)
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--workspace_clamp_mode", type=str, default="diagnostic", choices=["diagnostic", "hard", "tolerance"])
    parser.add_argument("--workspace_clamp_tolerance", type=float, default=0.0)
    parser.add_argument("--coarse2contact_chunk_size", type=int, default=4)
    parser.add_argument("--coarse2contact_precontact_depth_threshold", type=float, default=0.20)
    parser.add_argument("--coarse2contact_contact_depth_threshold", type=float, default=0.035)
    parser.add_argument("--coarse2contact_visual_xy_threshold", type=float, default=0.0015)
    parser.add_argument("--coarse2contact_visual_yaw_threshold", type=float, default=0.0349)
    parser.add_argument("--coarse2contact_max_xy_step", type=float, default=0.0005)
    parser.add_argument("--coarse2contact_max_yaw_step", type=float, default=0.0087)
    parser.add_argument("--coarse2contact_force_contact_threshold", type=float, default=0.18)
    parser.add_argument("--coarse2contact_force_delta_contact_threshold", type=float, default=0.05)
    parser.add_argument("--coarse2contact_force_jam_threshold", type=float, default=0.55)
    parser.add_argument("--coarse2contact_force_torque_threshold", type=float, default=0.12)
    parser.add_argument("--coarse2contact_force_spike_threshold", type=float, default=1.0)
    parser.add_argument("--coarse2contact_backoff_m", type=float, default=0.003)
    parser.add_argument("--coarse2contact_lateral_m", type=float, default=0.0015)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> float:
    _lazy_import_tasks()
    if args.task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {sorted(TASK_MAP)}")

    vla, processor, action_head, proprio_projector, norm_stats = load_checkpoint(
        args.checkpoint_dir,
        use_depth=args.planner_use_depth,
        use_force=args.planner_use_force,
    )
    if bool(args.coarse2contact_shadow_only):
        args.shadow_only = True
    coarse2contact = build_c2c_supervisor(args)

    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.task_low_dim_state = True
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    if hasattr(obs_config, "gripper_touch_forces"):
        obs_config.gripper_touch_forces = True

    env = Environment(
        MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()),
        obs_config=obs_config,
        headless=True,
    )
    env.launch()
    task = env.get_task(TASK_MAP[args.task_name])

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    trace_dir = output_dir / "gripper_traces"
    if args.record_video and args.write_episode_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
    if args.record_gripper_trace:
        trace_dir.mkdir(parents=True, exist_ok=True)

    episode_indices = [int(x.strip()) for x in str(args.episode_indices).split(",") if x.strip()] if args.episode_indices else list(range(args.num_episodes))
    results = {
        "mode": args.mode,
        "task_name": args.task_name,
        "episode_indices": episode_indices,
        "successes": [],
        "episode_lengths": [],
        "invalid_action_counts": [],
        "stage_stats": [],
        "video_paths": [],
        "gripper_trace_paths": [],
        "uses_privileged_target": False,
    }

    for loop_idx, ep_idx in enumerate(episode_indices):
        print(f"\n[c2c] Episode {loop_idx + 1}/{len(episode_indices)} (ep={ep_idx})", flush=True)
        if args.eval_seed is not None:
            seed = int(args.eval_seed) + int(ep_idx)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        descriptions, obs = task.reset()
        instruction = descriptions[0] if descriptions else f"complete {args.task_name.replace('_', ' ')}"
        force_buffer: deque[np.ndarray] = deque(maxlen=FORCE_HISTORY_LEN)
        action_queue: list[np.ndarray] = []
        frames = []
        gripper_trace = []
        invalid_action_count = 0
        workspace_violation_count = 0
        workspace_violation_max = 0.0
        success = False
        reward = 0.0
        terminate = False
        if coarse2contact is not None:
            coarse2contact.reset()

        for step_idx in range(args.max_steps):
            if args.record_video:
                frames.append(obs.front_rgb.copy())

            front_pil, wrist_pil, proprio, depth_tensor, force_hist, raw_force = process_obs(
                obs,
                norm_stats,
                force_buffer,
                use_depth=args.use_depth,
                use_force=args.use_force,
                depth_max=args.depth_max,
            )
            wrist_valid_depth_ratio, wrist_depth_near_fraction, wrist_is_occluded, wrist_is_low_visibility = compute_wrist_visibility_stats(obs.wrist_depth)

            if not action_queue:
                actions = predict_actions(
                    vla,
                    processor,
                    action_head,
                    proprio_projector,
                    front_pil,
                    wrist_pil,
                    proprio,
                    depth_tensor if args.planner_use_depth else None,
                    force_hist if args.planner_use_force else None,
                    instruction,
                )
                max_chunk = coarse2contact.get_chunk_size() if coarse2contact is not None else len(actions)
                action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), max_chunk))]

            delta_action = action_queue.pop(0)
            base_delta_action = delta_action.copy()
            planner_chunk_local = world_delta_to_local(base_delta_action[:6], obs.gripper_pose[3:7]).astype(np.float32)
            trace_entry = {
                "step": int(step_idx),
                "planner_action_world_6d": _jsonable_value(np.asarray(base_delta_action[:6], dtype=np.float32)),
                "planner_action_world_8d": _jsonable_value(np.asarray(base_delta_action, dtype=np.float32)),
                "planner_chunk_world_6d": _jsonable_value(np.asarray(base_delta_action[:6], dtype=np.float32)),
                "planner_chunk_local_6d": _jsonable_value(planner_chunk_local),
                "base_gripper_raw": float(base_delta_action[6]),
                "obs_gripper_open": float(obs.gripper_open),
                "wrist_valid_depth_ratio": float(wrist_valid_depth_ratio),
                "wrist_depth_near_fraction": float(wrist_depth_near_fraction),
                "wrist_is_occluded": bool(wrist_is_occluded),
                "wrist_is_low_visibility": bool(wrist_is_low_visibility),
                "invalid_action": False,
                "invalid_action_flag": False,
                "coarse2contact_phase": "COARSE",
                "planner_reaches_precontact": False,
                "planner_reaches_preinsert": False,
                "uses_privileged_target": False,
                "retry_id": 0,
                "depth_conf": None,
                "depth_obs_quality": None,
                "phase_owner": "planner",
                "phase_reason": "planner",
                "raw_wrench": _jsonable_value(np.asarray(raw_force if raw_force is not None else np.zeros(6, dtype=np.float32), dtype=np.float32)),
                "filtered_wrench": _jsonable_value(np.zeros(6, dtype=np.float32)),
                "mp4_path": None,
            }

            if coarse2contact is not None:
                delta_action = coarse2contact.step(
                    delta_action,
                    force_reading=raw_force,
                    gripper_z=float(obs.gripper_pose[2]),
                    wrist_depth=obs.wrist_depth,
                    proprio=proprio,
                )
                trace_entry.update(_jsonable_value(coarse2contact.get_last_trace()))

            abs_action = delta_to_absolute(delta_action, obs.gripper_pose)
            safety = coarse2contact.safety if coarse2contact is not None else None
            abs_action, workspace_violation = maybe_apply_workspace_filter(
                abs_action,
                safety,
                mode=args.workspace_clamp_mode,
                tolerance=args.workspace_clamp_tolerance,
            )
            if workspace_violation > 0.0:
                workspace_violation_count += 1
                workspace_violation_max = max(workspace_violation_max, float(workspace_violation))
            trace_entry["pre_clip_action_world_6d"] = _jsonable_value(np.asarray(abs_action[:6], dtype=np.float32))
            trace_entry["commanded_action_world_8d"] = _jsonable_value(abs_action.astype(np.float32))

            executed_action = abs_action
            recovery_applied = False
            try:
                obs, reward, terminate = task.step(abs_action)
            except InvalidActionError as exc:
                trace_entry["invalid_action"] = True
                trace_entry["invalid_action_flag"] = True
                trace_entry["invalid_error"] = type(exc).__name__
                invalid_action_count += 1
                action_queue.clear()
                if coarse2contact is not None:
                    recovery_abs = coarse2contact.build_invalid_action_recovery_absolute(
                        obs.gripper_pose,
                        obs.gripper_open,
                        force_reading=raw_force,
                        proprio=proprio,
                    )
                    recovery_abs, workspace_violation = maybe_apply_workspace_filter(
                        recovery_abs,
                        coarse2contact.safety,
                        mode=args.workspace_clamp_mode,
                        tolerance=args.workspace_clamp_tolerance,
                    )
                    executed_action = recovery_abs
                    trace_entry["coarse2contact_invalid_action_recovery"] = True
                    trace_entry.update(
                        {
                            "coarse2contact_invalid_action_recovery_phase": str(coarse2contact.get_last_trace().get("recovery_phase", "IDLE")),
                            "coarse2contact_invalid_action_recovery_primitive": str(coarse2contact.get_last_trace().get("recovery_primitive", "none")),
                            "coarse2contact_invalid_action_recovery_reason": str(coarse2contact.get_last_trace().get("force_reflex_reason", "invalid_action")),
                        }
                    )
                    trace_entry["retry_id"] = int(coarse2contact.get_last_trace().get("retry_id", trace_entry["retry_id"]))
                    try:
                        obs, reward, terminate = task.step(recovery_abs)
                        recovery_applied = True
                    except InvalidActionError:
                        gripper_trace.append(trace_entry)
                        continue
                else:
                    gripper_trace.append(trace_entry)
                    continue

            trace_entry["reward"] = float(reward)
            trace_entry["terminate"] = bool(terminate)
            trace_entry["filtered_wrench_6d"] = _jsonable_value(np.asarray(coarse2contact.get_last_trace().get("filtered_wrench_6d", np.zeros(6)), dtype=np.float32)) if coarse2contact is not None else _jsonable_value(np.zeros(6, dtype=np.float32))
            trace_entry["raw_wrench_6d"] = _jsonable_value(np.asarray(raw_force if raw_force is not None else np.zeros(6, dtype=np.float32), dtype=np.float32))
            if coarse2contact is not None:
                trace_entry["depth_conf"] = float(coarse2contact.get_last_trace().get("depth_conf", 0.0))
                trace_entry["depth_obs_quality"] = float(coarse2contact.get_last_trace().get("depth_obs_quality", 0.0))
                trace_entry["phase_owner"] = str(coarse2contact.get_last_trace().get("phase_owner", "planner"))
                trace_entry["phase_reason"] = str(coarse2contact.get_last_trace().get("phase_reason", "planner"))
                trace_entry["retry_id"] = int(coarse2contact.get_last_trace().get("retry_id", 0))
                trace_entry["invalid_action_flag"] = bool(coarse2contact.get_last_trace().get("invalid_action_flag", trace_entry["invalid_action"]))
                trace_entry["planner_reaches_precontact"] = bool(coarse2contact.get_last_trace().get("planner_reaches_precontact", trace_entry["planner_reaches_precontact"]))
                trace_entry["planner_reaches_preinsert"] = bool(coarse2contact.get_last_trace().get("planner_reaches_preinsert", trace_entry["planner_reaches_preinsert"]))
                trace_entry["pre_clip_action_world_6d"] = _jsonable_value(np.asarray(coarse2contact.get_last_trace().get("pre_clip_action_world_6d", abs_action[:6]), dtype=np.float32))
                trace_entry["post_clip_action_world_6d"] = _jsonable_value(np.asarray(coarse2contact.get_last_trace().get("post_clip_action_world_6d", abs_action[:6]), dtype=np.float32))
                trace_entry["executed_action_world_6d"] = _jsonable_value(np.asarray(executed_action[:6], dtype=np.float32))
                trace_entry["executed_action_world_8d"] = _jsonable_value(np.asarray(executed_action, dtype=np.float32))
                trace_entry["planner_action_world"] = _jsonable_value(np.asarray(base_delta_action[:6], dtype=np.float32))
                trace_entry["raw_wrench"] = _jsonable_value(np.asarray(coarse2contact.get_last_trace().get("raw_wrench", raw_force if raw_force is not None else np.zeros(6, dtype=np.float32)), dtype=np.float32))
                trace_entry["filtered_wrench"] = _jsonable_value(np.asarray(coarse2contact.get_last_trace().get("filtered_wrench", np.zeros(6, dtype=np.float32)), dtype=np.float32))
            else:
                trace_entry["post_clip_action_world_6d"] = _jsonable_value(np.asarray(abs_action[:6], dtype=np.float32))
                trace_entry["executed_action_world_6d"] = _jsonable_value(np.asarray(executed_action[:6], dtype=np.float32))
                trace_entry["planner_action_world"] = _jsonable_value(np.asarray(base_delta_action[:6], dtype=np.float32))
            trace_entry["executed_action_world_8d"] = _jsonable_value(np.asarray(executed_action, dtype=np.float32))
            trace_entry["invalid_action_recovery_executed"] = bool(recovery_applied)
            trace_entry["final_action_world_6d"] = _jsonable_value(np.asarray(delta_action[:6], dtype=np.float32))
            gripper_trace.append(trace_entry)

            if reward > 0.0:
                success = True
            if terminate:
                break

        episode_mp4_path = None
        if args.record_video and args.write_episode_videos and len(frames) > 1:
            status = "succ" if success else "fail"
            episode_mp4_path = str(video_dir / f"ep{ep_idx:03d}_{status}.mp4")
            clip = ImageSequenceClip(frames, fps=20)
            clip.write_videofile(episode_mp4_path, fps=20, codec="libx264", bitrate="3000k", logger=None)
            results["video_paths"].append(episode_mp4_path)
            for item in gripper_trace:
                item["mp4_path"] = episode_mp4_path

        if args.record_gripper_trace:
            trace_path = trace_dir / f"ep{ep_idx:03d}_gripper_trace.jsonl"
            with open(trace_path, "w") as handle:
                for item in gripper_trace:
                    handle.write(json.dumps(item, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            results["gripper_trace_paths"].append(str(trace_path))

        c2c_stats = coarse2contact.get_stats() if coarse2contact is not None else {}
        stage_stat = {
            "episode_index": int(ep_idx),
            "success": bool(success),
            "reward": float(reward),
            "episode_length": int(len(gripper_trace)),
            "invalid_action_count": int(invalid_action_count),
            "workspace_violation_count": int(workspace_violation_count),
            "workspace_violation_max": float(workspace_violation_max),
            "coarse2contact_phase": str(c2c_stats.get("coarse2contact_phase", "off")),
            "coarse2contact_correction_count": int(c2c_stats.get("coarse2contact_correction_count", 0)),
            "coarse2contact_recovery_count": int(c2c_stats.get("coarse2contact_recovery_count", 0)),
            "coarse2contact_invalid_action_count": int(c2c_stats.get("coarse2contact_invalid_action_count", 0)),
            "coarse2contact_precontact_count": int(c2c_stats.get("coarse2contact_precontact_count", 0)),
            "coarse2contact_preinsert_count": int(c2c_stats.get("coarse2contact_preinsert_count", 0)),
            "coarse2contact_reaches_precontact": bool(int(c2c_stats.get("coarse2contact_precontact_count", 0)) > 0),
            "coarse2contact_reaches_preinsert": bool(int(c2c_stats.get("coarse2contact_preinsert_count", 0)) > 0),
            "uses_privileged_target": False,
            "mp4_path": episode_mp4_path,
        }
        results["successes"].append(bool(success))
        results["episode_lengths"].append(int(len(gripper_trace)))
        results["invalid_action_counts"].append(int(invalid_action_count))
        results["stage_stats"].append(stage_stat)

    results["success_rate"] = float(np.mean(results["successes"])) if results["successes"] else 0.0
    results["avg_episode_length"] = float(np.mean(results["episode_lengths"])) if results["episode_lengths"] else 0.0
    results["planner_reaches_precontact_count"] = int(sum(int(s.get("coarse2contact_precontact_count", 0)) for s in results["stage_stats"]))
    results["planner_reaches_preinsert_count"] = int(sum(int(s.get("coarse2contact_preinsert_count", 0)) for s in results["stage_stats"]))
    results["coarse2contact_correction_count"] = int(sum(int(s.get("coarse2contact_correction_count", 0)) for s in results["stage_stats"]))
    results["coarse2contact_recovery_count"] = int(sum(int(s.get("coarse2contact_recovery_count", 0)) for s in results["stage_stats"]))
    results["coarse2contact_invalid_action_count"] = int(sum(int(s.get("coarse2contact_invalid_action_count", 0)) for s in results["stage_stats"]))

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as handle:
        json.dump(_jsonable_value(results), handle, indent=2)
    print(f"\n[c2c] Saved results to {results_path}", flush=True)
    env.shutdown()
    return float(results["success_rate"])


def main() -> int:
    args = parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
