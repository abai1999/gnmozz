"""
evaluate_rlbench.py

Evaluate a VLA-Adapter checkpoint on RLBench tasks in CoppeliaSim.
Supports optional residual/reflex refinement, video recording, and best-success GIF export.
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

# ── CoppeliaSim / PyRep environment setup (must happen before rlbench import) ─
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
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    from moviepy import ImageSequenceClip
from PIL import Image
from scipy.spatial.transform import Rotation

# ── RLBench imports ──────────────────────────────────────────────────────────
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

# ── Project imports ──────────────────────────────────────────────────────────
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load as load_vlm
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.action_heads_paper_faithful import PaperFaithfulL1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.models.residual_controller import ResidualController
from prismatic.robot.contact_refiner import ContactRefiner
from prismatic.robot.gripper_supervisor import GripperSupervisor
from prismatic.robot.stage_manager import StageManager, StagePhase
from prismatic.robot.stage_aware_refiner import StageAwareRefiner
os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
from prismatic.vla.constants import (
    ACTION_DIM,
    FORCE_DIM,
    FORCE_HISTORY_LEN,
    PROPRIO_DIM,
)

TASK_MAP = {}


def _lazy_import_tasks():
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


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    for k, v in state_dict.items():
        new_k = k
        for old, new in replace_map:
            if old in new_k:
                new_k = new_k.replace(old, new)
        new_sd[new_k] = v
    return new_sd


def _clean_state_dict(state_dict):
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_planner_head_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "planner_head_config.json"
    if not config_path.exists():
        return {"action_head_type": "l1_resnet", "planner_core_variant": "current"}
    with open(config_path) as f:
        return json.load(f)


def load_checkpoint(
    checkpoint_dir,
    vlm_path="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
    config_path="pretrained_models/configs/config.json",
    use_depth=True,
    use_force=True,
):
    checkpoint_dir = Path(checkpoint_dir)
    print(f"[eval] Loading checkpoint from {checkpoint_dir}")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vlm = load_vlm(vlm_path, hf_token="", load_for_training=False)
    config = AutoConfig.from_pretrained(config_path)
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)

    renamed_sd = _rename_state_dict(vlm.state_dict())
    vla.load_state_dict(renamed_sd, strict=False)
    del vlm

    lora_adapter_dir = checkpoint_dir / "lora_adapter"
    if lora_adapter_dir.exists():
        vla = PeftModel.from_pretrained(vla, str(lora_adapter_dir))
        vla = vla.merge_and_unload()

    aq_paths = sorted(glob.glob(str(checkpoint_dir / "action_queries--*.pt")))
    if aq_paths:
        aq_state = torch.load(aq_paths[0], map_location="cpu", weights_only=True)
        vla.action_queries.weight.data.copy_(aq_state["action_queries.weight"])
        print(f"[eval] Loaded action_queries from {aq_paths[0]}")
    else:
        raise FileNotFoundError(
            f"No action_queries checkpoint found in {checkpoint_dir}. "
            "Training must save action_queries--*.pt for valid evaluation."
        )

    vla.vision_backbone.set_num_images_in_input(2)
    vla = vla.to(torch.bfloat16).to(DEVICE).eval()

    stats_path = checkpoint_dir / "dataset_statistics.json"
    assert stats_path.exists(), f"dataset_statistics.json not found in {checkpoint_dir}"
    with open(stats_path) as f:
        norm_stats = json.load(f)
    vla.norm_stats = norm_stats

    processor = AutoProcessor.from_pretrained(str(checkpoint_dir), trust_remote_code=True)

    pp_paths = sorted(glob.glob(str(checkpoint_dir / "proprio_projector--*checkpoint.pt")))
    assert pp_paths, f"No proprio_projector checkpoint found in {checkpoint_dir}"
    proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
    proprio_projector.load_state_dict(_clean_state_dict(torch.load(pp_paths[0], weights_only=True)))
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE).eval()

    ah_paths = sorted(glob.glob(str(checkpoint_dir / "action_head--*checkpoint.pt")))
    assert ah_paths, f"No action_head checkpoint found in {checkpoint_dir}"
    planner_head_config = _load_planner_head_config(checkpoint_dir)
    action_head_type = planner_head_config.get("action_head_type", "l1_resnet")
    if action_head_type == "paper_faithful_l1_resnet" or planner_head_config.get("planner_core_variant") == "paper_faithful":
        action_head_cls = PaperFaithfulL1RegressionActionHead
    else:
        action_head_cls = L1RegressionActionHead
    action_head = action_head_cls(
        input_dim=vla.llm_dim,
        hidden_dim=vla.llm_dim,
        action_dim=ACTION_DIM,
        use_pro_version=True,
        use_depth=use_depth,
        use_force=use_force,
    )
    action_head.load_state_dict(_clean_state_dict(torch.load(ah_paths[0], weights_only=True)), strict=False)
    action_head = action_head.to(torch.bfloat16).to(DEVICE).eval()

    print("[eval] All components loaded successfully.")
    return vla, processor, action_head, proprio_projector, norm_stats


def load_residual_controller(residual_ckpt):
    model = ResidualController().to(DEVICE)
    ckpt = torch.load(residual_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.pose_output_mode = ckpt.get("pose_output_mode", "gated")
    readiness_keys = (
        "gripper_context_encoder.0.weight",
        "gripper_context_encoder.0.bias",
        "gripper_context_encoder.2.weight",
        "gripper_context_encoder.2.bias",
        "ready_head.weight",
        "ready_head.bias",
        "gripper_head.weight",
        "gripper_head.bias",
    )
    model._readiness_heads_loaded = all(k in state_dict for k in readiness_keys)
    if missing or unexpected:
        print(f"[eval] Residual checkpoint compatibility: missing={missing}, unexpected={unexpected}")
    if not model._readiness_heads_loaded:
        print("[eval] Residual checkpoint has no trained readiness/gripper heads; gripper readiness overrides disabled.")
    model = model.to(DEVICE).eval()
    print(f"[eval] Loaded residual controller from {residual_ckpt}")
    return model


def _load_optional_controller(path):
    if path is None:
        return None
    return load_residual_controller(path)


def _normalize_depth_array(depth, depth_max=1.0):
    if depth is None:
        return None
    depth_arr = depth.copy()
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[:, :, 0]
    depth_norm = np.clip(depth_arr / depth_max, 0.0, 1.0)
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

    proprio = np.concatenate(
        [obs.joint_positions, obs.gripper_pose, [float(obs.gripper_open)]]
    ).astype(np.float32)

    depth_tensor_224 = None
    depth_tensor_96 = None
    if use_depth and obs.wrist_depth is not None:
        depth_norm = _normalize_depth_array(obs.wrist_depth, depth_max=depth_max)
        depth_tensor_224 = _depth_tensor_from_norm(depth_norm, 224)
        depth_tensor_96 = _depth_tensor_from_norm(depth_norm, 96)

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

    return front_pil, wrist_pil, proprio, depth_tensor_224, force_history, depth_tensor_96, raw_force


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
    inputs_front["pixel_values"] = torch.cat(
        [inputs_front["pixel_values"], inputs_wrist["pixel_values"]], dim=1
    )

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
    r_current = Rotation.from_quat(current_quat)
    r_delta = Rotation.from_rotvec(delta_rv)
    new_quat = (r_delta * r_current).as_quat()

    gripper_cmd = 1.0 if gripper_raw > 0.5 else 0.0
    return np.concatenate([new_pos, new_quat, [gripper_cmd]])


def safe_recovery_absolute(current_gripper_pose, gripper_open, lift=0.0, safety=None):
    """Build an absolute recovery target that does not toggle the gripper."""
    pose = np.asarray(current_gripper_pose, dtype=np.float32).copy()
    pose[2] += float(lift)
    gripper_cmd = 1.0 if float(gripper_open) > 0.5 else 0.0
    return np.concatenate([pose[:7], [gripper_cmd]]).astype(np.float32)


def maybe_apply_workspace_filter(abs_action, safety, mode="diagnostic", tolerance=0.0):
    """Optionally clamp workspace while always returning raw violation distance."""
    violation = safety.workspace_violation(abs_action[:3])
    should_clamp = mode == "hard" or (mode == "tolerance" and violation > float(tolerance))
    if should_clamp:
        abs_action = abs_action.copy()
        abs_action[:3] = safety.clamp_workspace(abs_action[:3])
    return abs_action, violation


def build_refiner(args):
    if args.use_stage_aware_refiner:
        alignment_controller = _load_optional_controller(args.alignment_ckpt)
        contact_controller = _load_optional_controller(args.contact_ckpt)
        return StageAwareRefiner(
            mode=args.stage_refiner_mode,
            alignment_controller=alignment_controller,
            contact_controller=contact_controller,
            require_close_intent_for_alignment=args.require_close_intent_for_alignment,
            enable_alignment_pose=args.enable_alignment_pose,
            use_pose_alpha=args.use_pose_alpha,
            enable_readiness_gripper=args.enable_readiness_gripper,
            readiness_close_threshold=args.readiness_close_threshold,
            gripper_override_confidence=args.gripper_override_confidence,
        )
    if args.use_learned_residual:
        if not args.residual_ckpt:
            raise ValueError("--use_learned_residual requires --residual_ckpt")
        residual_controller = load_residual_controller(args.residual_ckpt)
        mode = "full" if args.use_rule_reflex else "learned_residual"
        return ContactRefiner(mode=mode, residual_controller=residual_controller)
    if args.use_rule_reflex:
        return ContactRefiner(mode="rule_reflex")
    return None


def build_gripper_supervisor(args):
    if not getattr(args, "use_gripper_supervisor", False):
        return None
    return GripperSupervisor(
        close_threshold=args.gripper_close_threshold,
        open_threshold=args.gripper_open_threshold,
        near_depth_threshold=args.gripper_near_depth_threshold,
        close_lookahead=args.gripper_close_lookahead,
        min_close_votes=args.gripper_min_close_votes,
        min_hold_steps=args.gripper_min_hold_steps,
        release_open_votes=args.gripper_release_open_votes,
        allow_release=args.gripper_allow_release,
    )


def evaluate(args, preloaded_components=None):
    _lazy_import_tasks()
    assert args.task_name in TASK_MAP, f"Unknown task: {args.task_name}. Available: {list(TASK_MAP)}"

    if preloaded_components is None:
        vla, processor, action_head, proprio_projector, norm_stats = load_checkpoint(
            args.checkpoint_dir,
            vlm_path=args.vlm_path,
            config_path=args.config_path,
            use_depth=args.planner_use_depth,
            use_force=args.planner_use_force,
        )
    else:
        vla, processor, action_head, proprio_projector, norm_stats = preloaded_components
    refiner = build_refiner(args)
    gripper_supervisor = build_gripper_supervisor(args)

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

    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    task_cls = TASK_MAP[args.task_name]
    task = env.get_task(task_cls)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    if args.record_video:
        video_dir.mkdir(exist_ok=True)
    gripper_trace_dir = output_dir / "gripper_traces"
    if args.record_gripper_trace:
        gripper_trace_dir.mkdir(exist_ok=True)

    results = {
        "successes": [],
        "episode_lengths": [],
        "stage_stats": [],
        "invalid_action_counts": [],
    }
    if refiner is not None:
        results["refiner_stats"] = []
    if gripper_supervisor is not None:
        results["gripper_supervisor_stats"] = []
    if args.record_gripper_trace:
        results["gripper_trace_paths"] = []

    best_success_frames = None
    best_success_len = float("inf")
    best_fail_frames = None
    best_fail_len = 0

    for ep_idx in range(args.num_episodes):
        print(f"\n--- Episode {ep_idx + 1}/{args.num_episodes} ---")
        if args.eval_seed is not None:
            ep_seed = int(args.eval_seed) + ep_idx
            random.seed(ep_seed)
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)
        descriptions, obs = task.reset()
        instruction = descriptions[0]
        if ep_idx == 0:
            print(f"Task: {args.task_name}  Instruction: {instruction}")

        if refiner is not None:
            refiner.reset()
        if gripper_supervisor is not None:
            gripper_supervisor.reset()

        force_buffer = deque(maxlen=256)
        action_queue = []
        frames = []
        success = False
        step_count = 0
        chunk_step = 0
        stage_tracker = StageManager()
        phase_hist = {int(phase): 0 for phase in StagePhase}
        failure_hist = {}
        align_entered = False
        interact_entered = False
        recover_entered = False
        invalid_action_count = 0
        consecutive_invalid_actions = 0
        first_close_step = None
        close_before_ready_count = 0
        close_before_basin_count = 0
        reopen_after_close_count = 0
        gripper_flip_count = 0
        last_abs_gripper_cmd = None
        ever_closed_cmd = False
        close_pose_z = None
        grasp_lift_proxy = False
        workspace_violation_count = 0
        workspace_violation_sum = 0.0
        workspace_violation_max = 0.0
        gripper_trace = []

        for step_idx in range(args.max_steps):
            if args.record_video:
                frames.append(obs.front_rgb.copy())

            front_pil, wrist_pil, proprio, depth_tensor, force_hist, depth_tensor_96, raw_force = process_obs(
                obs,
                norm_stats,
                force_buffer,
                use_depth=args.use_depth,
                use_force=args.use_force,
                depth_max=args.depth_max,
            )

            if len(action_queue) == 0:
                if refiner is not None and not isinstance(refiner, StageAwareRefiner):
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
                    depth_tensor if args.planner_use_depth else None,
                    force_hist if args.planner_use_force else None,
                    instruction,
                    unnorm_key="rlbench",
                )
                max_chunk = refiner.get_chunk_size() if refiner is not None else len(actions)
                action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), max_chunk))]
                chunk_step = 0
                if step_idx < 40 or step_idx % 50 == 0:
                    print(
                        f"  Step {step_idx} chunk[0]: delta_xyz=[{action_queue[0][0]:+.6f},{action_queue[0][1]:+.6f},{action_queue[0][2]:+.6f}] "
                        f"rot=[{action_queue[0][3]:+.6f},{action_queue[0][4]:+.6f},{action_queue[0][5]:+.6f}] grip={action_queue[0][6]:+.4f}"
                    )

            delta_action = action_queue.pop(0)
            base_delta_action = delta_action.copy()
            future_gripper_actions = [
                float(a[6]) for a in action_queue[: args.alignment_gate_lookahead]
            ]
            depth_proximity = StageAwareRefiner.compute_depth_proximity(depth_tensor_96)
            trace_entry = {
                "step": int(step_idx),
                "base_gripper_raw": float(base_delta_action[6]),
                "future_gripper_actions": list(future_gripper_actions),
                "obs_gripper_open": float(obs.gripper_open),
                "depth_proximity": None if depth_proximity is None else float(depth_proximity),
                "near_target_for_gripper": bool(
                    depth_proximity is not None
                    and np.isfinite(depth_proximity)
                    and float(depth_proximity) < args.gripper_near_depth_threshold
                ),
                "phase_before": int(stage_tracker.phase),
                "invalid_action": False,
            }
            if refiner is not None:
                step_kwargs = dict(
                    a_base_7d=delta_action,
                    step_idx=chunk_step,
                    force_reading=raw_force,
                    gripper_z=float(obs.gripper_pose[2]),
                    wrist_depth=depth_tensor_96,
                    ft_hist=force_hist,
                    proprio=proprio,
                )
                if isinstance(refiner, StageAwareRefiner):
                    step_kwargs["gripper_pose"] = obs.gripper_pose
                    step_kwargs["gripper_open"] = float(obs.gripper_open)
                    step_kwargs["future_gripper_actions"] = future_gripper_actions
                delta_action = refiner.step(**step_kwargs)
                if isinstance(refiner, StageAwareRefiner):
                    refiner_stats_snapshot = refiner.get_stats()
                    trace_entry["refiner_phase_after"] = int(refiner_stats_snapshot.get("phase_id", -1))
                    trace_entry["refiner_ready_to_close_prob_mean"] = float(
                        refiner_stats_snapshot.get("ready_to_close_prob_mean", 0.0)
                    )
                    trace_entry["refiner_current_trigger_prob"] = float(
                        refiner_stats_snapshot.get("current_trigger_prob", 0.0)
                    )
                    trace_entry["refiner_current_basin_positive"] = float(
                        refiner_stats_snapshot.get("current_basin_positive", 0.0)
                    )
                    trace_entry["refiner_readiness_eval_count"] = int(
                        refiner_stats_snapshot.get("readiness_eval_count", 0)
                    )
            trace_entry["after_refiner_gripper_raw"] = float(delta_action[6])

            if gripper_supervisor is not None:
                delta_action = gripper_supervisor.step(
                    delta_action,
                    depth_proximity=depth_proximity,
                    gripper_open=float(obs.gripper_open),
                    future_gripper_actions=future_gripper_actions,
                    phase_id=int(stage_tracker.phase),
                )
                trace_entry.update(gripper_supervisor.get_last_trace())
            trace_entry["exec_gripper_raw"] = float(delta_action[6])

            stage_tracker.update(
                force_reading=raw_force,
                gripper_pose=obs.gripper_pose,
                gripper_open=float(obs.gripper_open),
                depth_proximity=depth_proximity,
                base_action=delta_action,
            )
            trace_entry["phase_after"] = int(stage_tracker.phase)
            phase_hist[int(stage_tracker.phase)] += 1
            failure_code = int(stage_tracker.failure_mode)
            failure_hist[failure_code] = failure_hist.get(failure_code, 0) + 1
            align_entered = align_entered or stage_tracker.max_phase_reached >= int(StagePhase.ALIGN)
            interact_entered = interact_entered or stage_tracker.max_phase_reached >= int(StagePhase.INTERACT)
            recover_entered = recover_entered or stage_tracker.max_phase_reached >= int(StagePhase.RECOVER)

            abs_action = delta_to_absolute(delta_action, obs.gripper_pose)
            trace_entry["abs_gripper_cmd"] = float(abs_action[7])
            abs_gripper_cmd = float(abs_action[7])
            if last_abs_gripper_cmd is not None and abs(abs_gripper_cmd - last_abs_gripper_cmd) > 0.5:
                gripper_flip_count += 1
            if abs_gripper_cmd <= 0.5:
                if first_close_step is None:
                    first_close_step = int(step_idx)
                    close_pose_z = float(obs.gripper_pose[2])
                ever_closed_cmd = True
                if not bool(trace_entry.get("near_target_for_gripper", False)):
                    close_before_ready_count += 1
                if float(trace_entry.get("refiner_current_basin_positive", 0.0)) < 0.5:
                    close_before_basin_count += 1
            elif ever_closed_cmd:
                reopen_after_close_count += 1
            last_abs_gripper_cmd = abs_gripper_cmd
            trace_entry["first_close_step_so_far"] = -1 if first_close_step is None else int(first_close_step)
            trace_entry["close_before_ready_count_so_far"] = int(close_before_ready_count)
            trace_entry["close_before_basin_count_so_far"] = int(close_before_basin_count)
            trace_entry["reopen_after_close_count_so_far"] = int(reopen_after_close_count)
            trace_entry["gripper_flip_count_so_far"] = int(gripper_flip_count)
            if refiner is not None:
                abs_action, workspace_violation = maybe_apply_workspace_filter(
                    abs_action,
                    refiner.safety,
                    mode=args.workspace_clamp_mode,
                    tolerance=args.workspace_clamp_tolerance,
                )
                if workspace_violation > 0:
                    workspace_violation_count += 1
                    workspace_violation_sum += workspace_violation
                    workspace_violation_max = max(workspace_violation_max, workspace_violation)

            try:
                obs, reward, terminate = task.step(abs_action)
            except InvalidActionError as e:
                trace_entry["invalid_action"] = True
                trace_entry["invalid_error"] = type(e).__name__
                if args.record_gripper_trace:
                    gripper_trace.append(trace_entry)
                invalid_action_count += 1
                consecutive_invalid_actions += 1
                invalid_phase = stage_tracker.note_invalid_action()
                phase_hist[int(invalid_phase)] += 1
                failure_code = int(stage_tracker.failure_mode)
                failure_hist[failure_code] = failure_hist.get(failure_code, 0) + 1
                align_entered = align_entered or stage_tracker.max_phase_reached >= int(StagePhase.ALIGN)
                interact_entered = interact_entered or stage_tracker.max_phase_reached >= int(StagePhase.INTERACT)
                recover_entered = recover_entered or stage_tracker.max_phase_reached >= int(StagePhase.RECOVER)
                step_count = step_idx + 1
                print(f"  Step {step_idx}: action execution failed ({type(e).__name__}), marking RECOVER and replanning")
                action_queue.clear()
                chunk_step = 0

                if refiner is not None and hasattr(refiner, "on_invalid_action"):
                    _ = refiner.on_invalid_action(delta_action, raw_force)
                    lift = args.invalid_recovery_lift if consecutive_invalid_actions >= args.invalid_recovery_lift_after else 0.0
                    recovery_abs = safe_recovery_absolute(
                        obs.gripper_pose,
                        obs.gripper_open,
                        lift=lift,
                        safety=refiner.safety,
                    )
                    recovery_abs, recovery_violation = maybe_apply_workspace_filter(
                        recovery_abs,
                        refiner.safety,
                        mode=args.workspace_clamp_mode,
                        tolerance=args.workspace_clamp_tolerance,
                    )
                    if recovery_violation > 0:
                        workspace_violation_count += 1
                        workspace_violation_sum += recovery_violation
                        workspace_violation_max = max(workspace_violation_max, recovery_violation)
                    try:
                        obs, reward, terminate = task.step(recovery_abs)
                    except Exception as recovery_e:
                        print(
                            f"  Step {step_idx}: invalid-action recovery failed "
                            f"({type(recovery_e).__name__}); forcing replan"
                        )
                        continue
                    consecutive_invalid_actions = 0
                    if reward > 0:
                        success = True
                        if args.record_video:
                            frames.append(obs.front_rgb.copy())
                        print(f"  SUCCESS during invalid-action recovery at step {step_count}")
                        break
                continue
            except Exception as e:
                trace_entry["invalid_action"] = True
                trace_entry["invalid_error"] = type(e).__name__
                if args.record_gripper_trace:
                    gripper_trace.append(trace_entry)
                print(f"  Step {step_idx}: action execution failed ({type(e).__name__}), skipping step")
                action_queue.clear()
                consecutive_invalid_actions += 1
                step_count = step_idx + 1
                continue

            step_count = step_idx + 1
            consecutive_invalid_actions = 0
            chunk_step += 1
            trace_entry["reward"] = float(reward)
            trace_entry["terminate"] = bool(terminate)
            trace_entry["next_obs_gripper_open"] = float(obs.gripper_open)
            if (
                ever_closed_cmd
                and close_pose_z is not None
                and float(obs.gripper_open) < 0.5
                and float(obs.gripper_pose[2]) - float(close_pose_z) > 0.02
            ):
                grasp_lift_proxy = True
            if args.record_gripper_trace:
                gripper_trace.append(trace_entry)

            if refiner is not None and refiner.should_replan():
                refiner.note_replan()
                action_queue.clear()

            if reward > 0:
                success = True
                if args.record_video:
                    frames.append(obs.front_rgb.copy())
                print(f"  SUCCESS at step {step_count}")
                break

        if not success:
            print(f"  FAILED after {step_count} steps")

        results["successes"].append(success)
        results["episode_lengths"].append(step_count)
        results["invalid_action_counts"].append(invalid_action_count)
        results["stage_stats"].append(
            {
                "phase_counts": phase_hist,
                "failure_counts": failure_hist,
                "invalid_action_count": invalid_action_count,
                "transition_count": stage_tracker.transition_count,
                "phase_transition_rate": stage_tracker.transition_count / max(step_count, 1),
                "max_phase_reached": stage_tracker.max_phase_reached,
                "subgoal_progress": stage_tracker.get_subgoal_progress(),
                "align_entry": align_entered,
                "interact_entry": interact_entered,
                "recover_entry": recover_entered,
                "final_failure_mode": int(stage_tracker.failure_mode),
                "final_failure_mode_name": stage_tracker.get_failure_mode_name(),
                "workspace_violation_count": workspace_violation_count,
                "workspace_violation_mean": workspace_violation_sum / max(workspace_violation_count, 1),
                "workspace_violation_max": workspace_violation_max,
                "first_close_step": -1 if first_close_step is None else int(first_close_step),
                "close_before_ready_count": int(close_before_ready_count),
                "close_before_basin_count": int(close_before_basin_count),
                "reopen_after_close_count": int(reopen_after_close_count),
                "gripper_flip_count": int(gripper_flip_count),
                "grasp_lift_proxy": bool(grasp_lift_proxy),
                "refiner_phase_id": int(refiner.get_stats().get("phase_id", -1))
                if isinstance(refiner, StageAwareRefiner) else None,
                "refiner_max_phase_reached": int(refiner.get_stats().get("max_phase_reached", -1))
                if isinstance(refiner, StageAwareRefiner) else None,
                "refiner_subgoal_progress": float(refiner.get_stats().get("subgoal_progress", 0.0))
                if isinstance(refiner, StageAwareRefiner) else None,
            }
        )
        if refiner is not None:
            results["refiner_stats"].append(refiner.get_stats())
        if gripper_supervisor is not None:
            results["gripper_supervisor_stats"].append(gripper_supervisor.get_stats())
        if args.record_gripper_trace:
            trace_path = gripper_trace_dir / f"ep{ep_idx:03d}_gripper_trace.jsonl"
            with open(trace_path, "w") as f:
                for item in gripper_trace:
                    f.write(json.dumps(item) + "\n")
            results["gripper_trace_paths"].append(str(trace_path))

        if args.record_video and len(frames) > 1:
            status = "succ" if success else "fail"
            clip = ImageSequenceClip(frames, fps=20)
            vid_path = str(video_dir / f"ep{ep_idx:03d}_{status}.mp4")
            clip.write_videofile(vid_path, fps=20, codec="libx264", bitrate="3000k", logger=None)

            if success and len(frames) < best_success_len:
                best_success_len = len(frames)
                best_success_frames = frames
            elif not success and step_count > best_fail_len:
                best_fail_len = step_count
                best_fail_frames = list(frames)

    gif_frames = best_success_frames if best_success_frames is not None else best_fail_frames
    gif_label = "best_success" if best_success_frames is not None else "best_fail"
    if gif_frames is not None and len(gif_frames) > 1:
        gif_path = str(output_dir / f"{gif_label}.gif")
        clip = ImageSequenceClip(gif_frames, fps=10)
        clip.write_gif(gif_path, fps=10)
        print(f"Best GIF saved to: {gif_path}")

    n_success = sum(results["successes"])
    n_total = len(results["successes"])
    success_rate = n_success / max(n_total, 1)
    avg_len = float(np.mean(results["episode_lengths"])) if results["episode_lengths"] else 0.0

    total_phase_counts = {}
    total_failure_counts = {}
    for stage_stat in results["stage_stats"]:
        for key, value in stage_stat.get("phase_counts", {}).items():
            total_phase_counts[str(key)] = total_phase_counts.get(str(key), 0) + int(value)
        for key, value in stage_stat.get("failure_counts", {}).items():
            total_failure_counts[str(key)] = total_failure_counts.get(str(key), 0) + int(value)

    results["success_rate"] = success_rate
    results["avg_episode_length"] = avg_len
    results["invalid_action_count"] = int(sum(results["invalid_action_counts"]))
    results["avg_invalid_action_count"] = float(np.mean(results["invalid_action_counts"])) if results["invalid_action_counts"] else 0.0
    results["total_phase_counts"] = total_phase_counts
    results["total_failure_counts"] = total_failure_counts
    results["align_entry_rate"] = float(np.mean([s["align_entry"] for s in results["stage_stats"]])) if results["stage_stats"] else 0.0
    results["interact_entry_rate"] = float(np.mean([s["interact_entry"] for s in results["stage_stats"]])) if results["stage_stats"] else 0.0
    results["recover_entry_rate"] = float(np.mean([s["recover_entry"] for s in results["stage_stats"]])) if results["stage_stats"] else 0.0
    results["avg_phase_transition_rate"] = float(np.mean([s["phase_transition_rate"] for s in results["stage_stats"]])) if results["stage_stats"] else 0.0
    results["workspace_violation_count"] = int(sum(s.get("workspace_violation_count", 0) for s in results["stage_stats"]))
    violation_counts = [s.get("workspace_violation_count", 0) for s in results["stage_stats"]]
    violation_weighted = [
        s.get("workspace_violation_mean", 0.0) * s.get("workspace_violation_count", 0)
        for s in results["stage_stats"]
    ]
    results["workspace_violation_mean"] = float(sum(violation_weighted) / max(sum(violation_counts), 1))
    results["workspace_violation_max"] = float(max([s.get("workspace_violation_max", 0.0) for s in results["stage_stats"]] or [0.0]))
    results["workspace_clamp_mode"] = args.workspace_clamp_mode
    results["workspace_clamp_tolerance"] = args.workspace_clamp_tolerance
    results["first_close_steps"] = [int(s.get("first_close_step", -1)) for s in results["stage_stats"]]
    results["close_before_ready_count"] = int(sum(s.get("close_before_ready_count", 0) for s in results["stage_stats"]))
    results["close_before_basin_count"] = int(sum(s.get("close_before_basin_count", 0) for s in results["stage_stats"]))
    results["reopen_after_close_count"] = int(sum(s.get("reopen_after_close_count", 0) for s in results["stage_stats"]))
    results["gripper_flip_count"] = int(sum(s.get("gripper_flip_count", 0) for s in results["stage_stats"]))
    results["grasp_lift_proxy_count"] = int(sum(1 for s in results["stage_stats"] if s.get("grasp_lift_proxy", False)))
    if gripper_supervisor is not None:
        sup_stats = results.get("gripper_supervisor_stats", [])
        for key in [
            "gripper_close_commit_count",
            "gripper_blocked_early_close_count",
            "gripper_hold_override_count",
            "gripper_release_count",
            "gripper_raw_close_before_near_count",
            "gripper_raw_reopen_during_hold_count",
            "gripper_command_flip_count",
        ]:
            results[key] = int(sum(s.get(key, 0) for s in sup_stats))
    if refiner is not None:
        ref_stats = results.get("refiner_stats", [])
        for key in [
            "correction_count",
            "alignment_correction_count",
            "contact_correction_count",
            "replan_count",
            "alignment_gate_block_count",
            "alignment_window_corrections",
            "readiness_eval_count",
            "preclose_alignment_correction_count",
            "ready_positive_rate",
            "basin_positive_rate",
            "readiness_close_override_count",
            "readiness_open_override_count",
            "readiness_hold_override_count",
            "readiness_heads_missing_count",
        ]:
            if key.endswith("_rate"):
                values = [float(s.get(key, 0.0)) for s in ref_stats]
                results[key] = float(sum(values) / max(len(values), 1))
            else:
                results[key] = int(sum(s.get(key, 0) for s in ref_stats))
        for key in [
            "alpha_mean",
            "residual_pos_norm_mean",
            "ready_to_close_prob_mean",
            "trigger_prob_mean",
        ]:
            values = [float(s.get(key, 0.0)) for s in ref_stats]
            results[key] = float(sum(values) / max(len(values), 1))
    results["use_gripper_supervisor"] = bool(gripper_supervisor is not None)
    results["checkpoint"] = str(args.checkpoint_dir)
    results["task_name"] = args.task_name
    results["eval_seed"] = args.eval_seed
    base_mode = (
        f"stage_aware:{args.stage_refiner_mode}" if args.use_stage_aware_refiner else
        "full" if (args.use_learned_residual and args.use_rule_reflex) else
        "learned_residual" if args.use_learned_residual else
        "rule_reflex" if args.use_rule_reflex else
        "planner_only"
    )
    if gripper_supervisor is not None:
        base_mode = f"{base_mode}+gripper_supervisor"
    results["mode"] = base_mode

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Task:         {args.task_name}")
    print(f"Checkpoint:   {args.checkpoint_dir}")
    print(f"Mode:         {results['mode']}")
    print(f"Success Rate: {success_rate:.1%}  ({n_success}/{n_total})")
    print(f"Avg Length:   {avg_len:.1f} steps")
    print(f"InvalidAct:   {results['invalid_action_count']} total")
    print(f"Results:      {results_path}")
    print(f"{'=' * 50}")

    env.shutdown()
    return success_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA-Adapter checkpoint on RLBench")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--num_episodes", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--no_depth", dest="use_depth", action="store_false")
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--no_force", dest="use_force", action="store_false")
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="record_video", action="store_false")
    parser.add_argument(
        "--depth_max",
        type=float,
        default=1.0,
        help="Max depth for normalization to [0,1]. Use 1.0 for depth buffer (default), 2.55 for meters.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--vlm_path",
        type=str,
        default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
    )
    parser.add_argument("--config_path", type=str, default="pretrained_models/configs/config.json")
    parser.add_argument("--use_rule_reflex", action="store_true", default=False)
    parser.add_argument("--use_learned_residual", action="store_true", default=False)
    parser.add_argument("--residual_ckpt", type=str, default=None)
    parser.add_argument("--use_stage_aware_refiner", action="store_true", default=False)
    parser.add_argument(
        "--stage_refiner_mode",
        type=str,
        default="full",
        choices=["planner_only", "safety_only", "alignment", "contact", "full"],
    )
    parser.add_argument("--alignment_ckpt", type=str, default=None)
    parser.add_argument("--contact_ckpt", type=str, default=None)
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.add_argument(
        "--invalid_recovery_lift",
        type=float,
        default=0.008,
        help="Meters to lift the end effector for safe recovery after repeated InvalidActionError.",
    )
    parser.add_argument(
        "--invalid_recovery_lift_after",
        type=int,
        default=1,
        help="Start lifting instead of pure hold after this many consecutive InvalidActionError events.",
    )
    parser.add_argument(
        "--workspace_clamp_mode",
        type=str,
        default="diagnostic",
        choices=["off", "diagnostic", "tolerance", "hard"],
        help="diagnostic records workspace violations without modifying the action; hard clamps before env.step.",
    )
    parser.add_argument(
        "--workspace_clamp_tolerance",
        type=float,
        default=0.01,
        help="Clamp only when workspace violation exceeds this distance in tolerance mode.",
    )
    parser.add_argument(
        "--alignment_gate_lookahead",
        type=int,
        default=4,
        help="Number of queued future actions used to detect imminent gripper closing for AlignmentRefiner.",
    )
    parser.add_argument(
        "--require_close_intent_for_alignment",
        dest="require_close_intent_for_alignment",
        action="store_true",
        default=True,
        help="If set, AlignmentRefiner only runs when the planner's gripper plan contains a close command.",
    )
    parser.add_argument(
        "--allow_alignment_without_close_intent",
        dest="require_close_intent_for_alignment",
        action="store_false",
        help="Allow AlignmentRefiner to run outside the planner's close-intent window.",
    )
    parser.add_argument("--enable_alignment_pose", action="store_true", default=True)
    parser.add_argument("--disable_alignment_pose", dest="enable_alignment_pose", action="store_false")
    parser.add_argument("--use_pose_alpha", action="store_true", default=True)
    parser.add_argument("--disable_pose_alpha", dest="use_pose_alpha", action="store_false")
    parser.add_argument("--enable_readiness_gripper", action="store_true", default=False)
    parser.add_argument("--readiness_close_threshold", type=float, default=0.65)
    parser.add_argument("--gripper_override_confidence", type=float, default=0.60)
    parser.add_argument("--use_gripper_supervisor", action="store_true", default=False)
    parser.add_argument("--record_gripper_trace", action="store_true", default=True)
    parser.add_argument("--no_gripper_trace", dest="record_gripper_trace", action="store_false")
    parser.add_argument("--gripper_close_threshold", type=float, default=0.5)
    parser.add_argument("--gripper_open_threshold", type=float, default=0.8)
    parser.add_argument("--gripper_near_depth_threshold", type=float, default=0.08)
    parser.add_argument("--gripper_close_lookahead", type=int, default=4)
    parser.add_argument("--gripper_min_close_votes", type=int, default=2)
    parser.add_argument("--gripper_min_hold_steps", type=int, default=40)
    parser.add_argument("--gripper_release_open_votes", type=int, default=3)
    parser.add_argument("--gripper_allow_release", action="store_true", default=True)
    parser.add_argument("--gripper_no_release", dest="gripper_allow_release", action="store_false")
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.set_defaults(planner_use_depth=None, planner_use_force=None)

    args = parser.parse_args()

    if args.planner_use_depth is None:
        args.planner_use_depth = args.use_depth
    if args.planner_use_force is None:
        args.planner_use_force = args.use_force

    if args.output_dir is None:
        ckpt_name = Path(args.checkpoint_dir).name
        args.output_dir = f"eval_logs/{args.task_name}/{ckpt_name}"

    evaluate(args)


if __name__ == "__main__":
    main()
