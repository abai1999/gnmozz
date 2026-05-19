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
import time
from collections import Counter, deque
from pathlib import Path

# ── CoppeliaSim / PyRep environment setup (must happen before rlbench import) ─
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
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
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
from transformers.modeling_utils import no_init_weights

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.load import load as load_vlm
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.action_heads_paper_faithful import PaperFaithfulL1RegressionActionHead
from prismatic.models.alignment_diffusion_refiner import AlignmentDiffusionRefiner
from prismatic.models.alignment_tc_diffusion_refiner import TargetConditionedAlignmentDiffusionRefiner
from prismatic.models.alignment_tc_student_vnext import AlignmentTCStudentVNext
from prismatic.models.alignment_v3_direct_local_controller import AlignmentV3DirectLocalController
from prismatic.models.alignment_v4_direct_local_controller import AlignmentV4DirectLocalController
from prismatic.models.near_ready_xyyaw_predictor import NearReadyXYYawPredictor
from prismatic.models.pose_field_scorer import PoseFieldScorer
from prismatic.models.depth_force_local_proposal_policy import DepthForceLocalProposalPolicy
from prismatic.models.near_ready_residual_adapter import (
    NearReadyGroupResidualAdapter,
    NearReadyResidualScoreAdapter,
)
from prismatic.models.projectors import ProprioProjector
from prismatic.models.residual_controller import ResidualController
from prismatic.models.student_candidate_evaluator_v2 import StudentCandidateEvaluatorV2
from prismatic.models.student_group_selector_v2 import StudentGroupSelectorV2
from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2
from prismatic.robot.coarse2contact import Coarse2ContactSupervisor
from prismatic.robot.contact_refiner import ContactRefiner
from prismatic.robot.gripper_supervisor import GripperSupervisor
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from prismatic.robot.stage_manager import ContactState, StageManager, StagePhase, StageSubgoal, StageTargetMode
from prismatic.robot.stage_target_provider import (
    TeacherPhase1TargetResult,
    apply_yaw_symmetry_to_delta,
    build_phase1_teacher_targets,
    build_stage_target_provider,
    load_phase1_grasp_spec,
    select_phase1_teacher_target,
)
from prismatic.robot.stage_aware_refiner import StageAwareRefiner
os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
from prismatic.vla.constants import (
    ACTION_DIM,
    FORCE_DIM,
    FORCE_HISTORY_LEN,
    PROPRIO_DIM,
)
from build_pose_candidate_dataset import (
    apply_local_offset_to_pose,
    build_action_primitives,
    build_orientation_rescue_primitives,
    candidate_group_key,
    improvement_tiers,
    sign_bucket,
)

TASK_MAP = {}


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


def shadow_residual_audit(
    current_delta_local,
    predicted_residual_local,
    *,
    xy_threshold: float,
    abs_z_threshold: float,
    yaw_threshold: float,
    yaw_symmetry_period: float = -1.0,
):
    current = np.asarray(current_delta_local, dtype=np.float32).reshape(-1)
    residual = np.asarray(predicted_residual_local, dtype=np.float32).reshape(-1)
    if current.size < 6 or residual.size < 4:
        return {}
    if not np.all(np.isfinite(current[:6])) or not np.all(np.isfinite(residual[:4])):
        return {}
    before = apply_yaw_symmetry_to_delta(current[:6], yaw_symmetry_period).reshape(-1)
    after = before.copy()
    after[:3] = after[:3] - residual[:3]
    after[5] = after[5] - residual[3]
    after = apply_yaw_symmetry_to_delta(after, yaw_symmetry_period).reshape(-1)
    xy_thr = max(float(xy_threshold), 1e-6)
    z_thr = max(float(abs_z_threshold), 1e-6)
    yaw_thr = max(float(yaw_threshold), 1e-6) if float(yaw_threshold) > 0.0 else 1.0

    def _cost(delta6):
        xy = float(np.linalg.norm(delta6[:2])) / xy_thr
        z = float(abs(delta6[2])) / z_thr
        yaw = float(abs(delta6[5])) / yaw_thr if float(yaw_threshold) > 0.0 else 0.0
        return 0.45 * xy + 0.30 * z + 0.25 * yaw

    cost_before = _cost(before)
    cost_after = _cost(after)
    return {
        "shadow_residual_valid": True,
        "shadow_residual_cost_before": float(cost_before),
        "shadow_residual_cost_after": float(cost_after),
        "shadow_residual_cost_delta": float(cost_before - cost_after),
        "shadow_residual_improves": bool(cost_after < cost_before),
        "shadow_residual_next_xy_error": float(np.linalg.norm(after[:2])),
        "shadow_residual_next_abs_z_error": float(abs(after[2])),
        "shadow_residual_next_yaw_error": float(abs(after[5])),
        "shadow_residual_next_delta_local": after.astype(np.float32).tolist(),
    }


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.set_num_interop_threads(max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "8")) // 2)))
except Exception:
    pass


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


def load_checkpoint(
    checkpoint_dir,
    vlm_path="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
    config_path="pretrained_models/configs/config.json",
    use_depth=True,
    use_force=True,
):
    checkpoint_dir = Path(checkpoint_dir)
    print(f"[eval] Loading checkpoint from {checkpoint_dir}")
    load_t0 = time.perf_counter()

    def _mark(label: str) -> None:
        print(f"[eval][load] {label}: {time.perf_counter() - load_t0:.1f}s", flush=True)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    config = AutoConfig.from_pretrained(config_path, local_files_only=True, trust_remote_code=False)
    with no_init_weights():
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
    _mark("built eval VLA shell")

    merged_eval_state = checkpoint_dir / "merged_eval_vla_state.pt"
    if merged_eval_state.exists() and os.environ.get("EVAL_USE_MERGED_STATE", "1").lower() not in {"0", "false", "no", "off"}:
        merged_state = torch.load(merged_eval_state, map_location="cpu", weights_only=True)
        state_dict = merged_state.get("model_state_dict", merged_state) if isinstance(merged_state, dict) else merged_state
        missing, unexpected = vla.load_state_dict(_clean_state_dict(state_dict), strict=False)
        print(
            f"[eval] Loaded merged eval VLA state from {merged_eval_state} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        _mark("loaded merged eval VLA state")
    else:
        vlm = load_vlm(vlm_path, hf_token="", load_for_training=False)
        _mark("loaded base VLM")
        renamed_sd = _rename_state_dict(vlm.state_dict())
        vla.load_state_dict(renamed_sd, strict=False)
        del vlm
        _mark("copied base VLM weights")

        lora_adapter_dir = checkpoint_dir / "lora_adapter"
        if lora_adapter_dir.exists():
            vla = PeftModel.from_pretrained(vla, str(lora_adapter_dir), local_files_only=True)
            _mark("loaded LoRA adapter")
            if os.environ.get("EVAL_MERGE_LORA", "0").lower() in {"1", "true", "yes", "on"}:
                vla = vla.merge_and_unload()
                _mark("merged LoRA adapter")

    vla_core = getattr(getattr(vla, "base_model", None), "model", vla)

    aq_paths = sorted(glob.glob(str(checkpoint_dir / "action_queries--*.pt")))
    if aq_paths:
        aq_state = torch.load(aq_paths[0], map_location="cpu", weights_only=True)
        vla_core.action_queries.weight.data.copy_(aq_state["action_queries.weight"])
        print(f"[eval] Loaded action_queries from {aq_paths[0]}")
    else:
        raise FileNotFoundError(
            f"No action_queries checkpoint found in {checkpoint_dir}. "
            "Training must save action_queries--*.pt for valid evaluation."
        )

    vla_core.vision_backbone.set_num_images_in_input(2)
    vla = vla.to(torch.bfloat16).to(DEVICE).eval()
    _mark("moved VLA to device")

    stats_path = checkpoint_dir / "dataset_statistics.json"
    assert stats_path.exists(), f"dataset_statistics.json not found in {checkpoint_dir}"
    with open(stats_path) as f:
        norm_stats = json.load(f)
    vla.norm_stats = norm_stats
    vla_core.norm_stats = norm_stats

    processor = AutoProcessor.from_pretrained(
        str(checkpoint_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    _mark("loaded processor")

    pp_paths = sorted(glob.glob(str(checkpoint_dir / "proprio_projector--*checkpoint.pt")))
    assert pp_paths, f"No proprio_projector checkpoint found in {checkpoint_dir}"
    llm_dim = int(getattr(vla_core, "llm_dim"))
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
    proprio_projector.load_state_dict(
        _clean_state_dict(torch.load(pp_paths[0], map_location="cpu", weights_only=True))
    )
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE).eval()
    _mark("loaded proprio projector")

    ah_paths = sorted(glob.glob(str(checkpoint_dir / "action_head--*checkpoint.pt")))
    assert ah_paths, f"No action_head checkpoint found in {checkpoint_dir}"
    planner_head_cfg_path = checkpoint_dir / "planner_head_config.json"
    planner_core_variant = "current"
    if planner_head_cfg_path.exists():
        try:
            planner_head_cfg = json.loads(planner_head_cfg_path.read_text())
            planner_core_variant = str(planner_head_cfg.get("planner_core_variant", "current"))
        except Exception as exc:
            print(f"[eval] Warning: failed to read planner head config {planner_head_cfg_path}: {exc}")
            planner_core_variant = "current"
    action_head_cls = (
        PaperFaithfulL1RegressionActionHead
        if planner_core_variant == "paper_faithful"
        else L1RegressionActionHead
    )
    print(f"[eval] Loading action head variant: {planner_core_variant}")
    action_head = action_head_cls(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=ACTION_DIM,
        use_pro_version=True,
        use_depth=use_depth,
        use_force=use_force,
    )
    action_head.load_state_dict(
        _clean_state_dict(torch.load(ah_paths[0], map_location="cpu", weights_only=True)),
        strict=False,
    )
    action_head = action_head.to(torch.bfloat16).to(DEVICE).eval()
    _mark("loaded action head")

    print("[eval] All components loaded successfully.")
    return vla, processor, action_head, proprio_projector, norm_stats


def load_residual_controller(residual_ckpt):
    ckpt = torch.load(residual_ckpt, map_location="cpu")
    model = ResidualController(
        pose_output_mode=ckpt.get("pose_output_mode", "gated"),
        pose_use_depth=ckpt.get("pose_use_depth", True),
        pose_use_force=ckpt.get("pose_use_force", True),
        pose_use_proprio=ckpt.get("pose_use_proprio", True),
        pose_use_action=ckpt.get("pose_use_action", True),
        fire_only_head=ckpt.get("fire_only_head", False),
        ready_use_context=ckpt.get("ready_use_context", True),
        ready_use_gripper_context=ckpt.get("ready_use_gripper_context", True),
    ).to(DEVICE)
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
    )
    if not model.fire_only_head:
        readiness_keys = readiness_keys + (
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


def load_pose_field_scorer(scorer_ckpt):
    ckpt = torch.load(scorer_ckpt, map_location="cpu")
    dataset_npz = ckpt.get("dataset_npz")
    if dataset_npz is None:
        raise ValueError(f"Pose-field scorer checkpoint missing dataset_npz: {scorer_ckpt}")
    dataset = np.load(dataset_npz)
    candidate_actions = dataset["candidate_actions_local"][0].astype(np.float32)
    candidate_group_index = dataset["candidate_group_index"][0].astype(np.int64)
    candidate_mask = dataset["candidate_mask"][0].astype(np.float32) if "candidate_mask" in dataset.files else np.ones((candidate_actions.shape[0],), dtype=np.float32)
    candidate_kind = dataset["candidate_kind"][0] if "candidate_kind" in dataset.files else np.asarray(["base"] * candidate_actions.shape[0])
    current_basin_distance = dataset["current_basin_distance"].astype(np.float32)
    num_candidate_groups = int(ckpt.get("num_candidate_groups", int(np.max(candidate_group_index)) + 1))
    model = PoseFieldScorer(
        use_depth=ckpt.get("use_depth", True),
        use_base_action=ckpt.get("use_base_action", True),
        use_proprio=ckpt.get("use_proprio", True),
        use_target_context=ckpt.get("use_target_context", True),
        use_front_rgb=ckpt.get("use_front_rgb", False),
        use_wrist_rgb=ckpt.get("use_wrist_rgb", False),
        num_candidate_groups=num_candidate_groups,
        fire_only_head=ckpt.get("fire_only_head", True),
    ).to(DEVICE)
    state_dict = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    legacy_missing = set(missing or [])
    legacy_step_scale_missing = any(k.startswith("step_scale_head.") for k in legacy_missing)
    if legacy_step_scale_missing:
        # Older 20260419a-family checkpoints predate the step-scale head.
        # Leaving this head at random init perturbs runtime action magnitudes
        # and changes controller behavior even when all main scorer weights
        # match. Neutralize it so candidate_step_scale defaults to 1.0.
        for module in model.step_scale_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
    model._controller_type = "pose_field_scorer"
    model._selection_policy = str(ckpt.get("selection_policy", "two_stage_group"))
    readiness_keys = (
        "ready_head.weight",
        "ready_head.bias",
    )
    model._readiness_heads_loaded = all(k in state_dict for k in readiness_keys)
    model._candidate_actions_local = torch.from_numpy(candidate_actions)
    model._candidate_group_index = torch.from_numpy(candidate_group_index)
    model._runtime_candidate_mask = np.asarray(candidate_mask, dtype=np.float32)
    model._teacher_candidate_kind = np.asarray(candidate_kind)
    has_tilt_candidates = bool(np.any(np.abs(candidate_actions[:, 3:5]) > 1e-8))
    model._ignore_tilt_alignment = not has_tilt_candidates
    model._support_gate_requires_target = bool(ckpt.get("use_target_context", True))
    model._close_veto_requires_target = bool(ckpt.get("use_target_context", True))
    model._runtime_current_delta_basin_target = np.zeros(6, dtype=np.float32)
    model._runtime_current_basin_distance = 3.0
    model._support_basin_distance_max = float(np.percentile(current_basin_distance, 95))
    if "current_delta_basin_target" in dataset.files:
        support_delta = dataset["current_delta_basin_target"].astype(np.float32)
        # Runtime target context must stay inside the distribution seen by the
        # scorer. This is not a normalization change; it is a conservative clamp
        # to prevent far-out privileged/proxy deltas from driving the context MLP
        # into an extrapolation regime during closed-loop recovery.
        model._target_context_clip_abs = np.maximum(
            np.percentile(np.abs(support_delta), 99, axis=0).astype(np.float32),
            np.asarray([0.02, 0.02, 0.02, 0.05, 0.05, 0.05], dtype=np.float32),
        )
        model._support_inner_xy_max = float(np.percentile(np.linalg.norm(support_delta[:, :2], axis=1), 95))
        model._support_inner_abs_z_max = float(np.percentile(np.abs(support_delta[:, 2]), 95))
        model._support_inner_tilt_max = (
            float(np.percentile(np.linalg.norm(support_delta[:, 3:5], axis=1), 95))
            if has_tilt_candidates else np.inf
        )
        model._support_inner_yaw_max = float(np.percentile(np.abs(support_delta[:, 5]), 95))
    else:
        model._support_inner_xy_max = np.inf
        model._support_inner_abs_z_max = np.inf
        model._support_inner_tilt_max = np.inf
        model._support_inner_yaw_max = np.inf
    model._runtime_target_pose_7d = None
    if "reference_anchor_pose_7d" in dataset.files:
        anchor_poses = dataset["reference_anchor_pose_7d"].astype(np.float32)
        anchor_pos = np.median(anchor_poses[:, :3], axis=0).astype(np.float32)
        anchor_quat = Rotation.from_quat(anchor_poses[:, 3:7]).mean().as_quat().astype(np.float32)
        model._reference_anchor_pose_7d = np.concatenate([anchor_pos, anchor_quat], axis=0).astype(np.float32)
    else:
        model._reference_anchor_pose_7d = None
    if "basin_center_pose_7d" in dataset.files:
        basin_poses = dataset["basin_center_pose_7d"].astype(np.float32)
        basin_pos = np.median(basin_poses[:, :3], axis=0).astype(np.float32)
        basin_quat = Rotation.from_quat(basin_poses[:, 3:7]).mean().as_quat().astype(np.float32)
        model._canonical_basin_center_pose_7d = np.concatenate([basin_pos, basin_quat], axis=0).astype(np.float32)
    else:
        model._canonical_basin_center_pose_7d = None
    ready_meta = {}
    if isinstance(ckpt.get("dataset_meta"), dict):
        ready_meta = ckpt.get("dataset_meta", {}).get("ready_label", {}) or {}
    if not ready_meta:
        meta_path = Path(dataset_npz).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                ready_meta = json.loads(meta_path.read_text()).get("ready_label", {}) or {}
            except Exception:
                ready_meta = {}
    model._ready_band_xy_threshold = float(ready_meta.get("xy_threshold", 0.010))
    model._ready_band_abs_z_threshold = float(ready_meta.get("abs_z_threshold", 0.020))
    model._ready_band_yaw_threshold = float(ready_meta.get("yaw_threshold", -1.0))
    model._ready_band_basin_distance_threshold = float(ready_meta.get("basin_distance_threshold", 1.0))
    if missing or unexpected:
        print(f"[eval] Pose-field checkpoint compatibility: missing={missing}, unexpected={unexpected}")
    if not model._readiness_heads_loaded:
        print("[eval] Pose-field scorer has no trained readiness head; learned close fire disabled.")
    model = model.to(DEVICE).eval()
    print(
        f"[eval] Loaded pose-field scorer from {scorer_ckpt} "
        f"(candidates={candidate_actions.shape[0]}, groups={num_candidate_groups})"
    )
    return model


def load_depth_force_local_proposal_policy(policy_ckpt):
    ckpt = torch.load(policy_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    state_dim = int(ckpt.get("state_dim", 384))
    proposal_count = int(ckpt.get("proposal_count", 8))
    model = DepthForceLocalProposalPolicy(
        proposal_count=proposal_count,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        **model_kwargs,
    ).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    load_summary = {
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "loaded_keys": sorted(set(model.state_dict().keys()) & set(state_dict.keys())),
        "newly_initialized_keys": sorted(missing),
    }
    print(
        "[eval] Loaded depth-force local proposal policy "
        f"from {policy_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing or unexpected:
        print(f"[eval] Depth-force proposal compatibility: missing={missing}, unexpected={unexpected}")
    model = model.to(DEVICE).eval()
    # Keep the legacy pose-field scorer path and stage gates, but mark this as a
    # self-generated proposal policy so runtime can take the thin adapter path.
    model._controller_type = "pose_field_scorer"
    model._runtime_policy_type = "depth_force_local_proposal_policy"
    model._selection_mode = str(ckpt.get("selection_mode", "layered_multi"))
    model._selection_weights = {
        "safe": float(ckpt.get("multi_utility_w_safe", 1.0)),
        "pareto": float(ckpt.get("multi_utility_w_pareto", 1.0)),
        "yaw": float(ckpt.get("multi_utility_w_yaw", 1.0)),
        "geom": float(ckpt.get("multi_utility_w_geom", 0.5)),
        "risk": float(ckpt.get("multi_utility_w_risk", 0.5)),
    }
    model._checkpoint_path = str(policy_ckpt)
    model._load_summary = load_summary
    # --- canonical target context for the target provider ---
    # The fixed K=8 proposal cache carries target poses; use their median as
    # the canonical basin centre so that the target provider can compute a
    # meaningful motion_target_delta_local at runtime.
    _cache_path = ckpt.get("cache_npz") or ckpt.get("proposal_cache_npz")
    if _cache_path is None:
        _cache_path = "/home/guoning/code/VLA/runtime_artifacts/depth_force_contact/fixed_k8_from_v5_proposal_cache_20260503a.npz"
    if Path(str(_cache_path)).exists():
        print(f"[eval] Loading canonical target context from {_cache_path}", flush=True)
        _cache = np.load(str(_cache_path), allow_pickle=True)
        if "target_pose_7d" in _cache.files:
            tp = np.asarray(_cache["target_pose_7d"], dtype=np.float32)
            basin_pos = np.median(tp[:, :3], axis=0).astype(np.float32)
            basin_quat = Rotation.from_quat(tp[:, 3:7]).mean().as_quat().astype(np.float32)
            model._canonical_basin_center_pose_7d = np.concatenate([basin_pos, basin_quat])
        if "reference_anchor_pose_7d" in _cache.files:
            ap = np.asarray(_cache["reference_anchor_pose_7d"], dtype=np.float32)
            anchor_pos = np.median(ap[:, :3], axis=0).astype(np.float32)
            anchor_quat = Rotation.from_quat(ap[:, 3:7]).mean().as_quat().astype(np.float32)
            model._reference_anchor_pose_7d = np.concatenate([anchor_pos, anchor_quat])
        model._support_inner_xy_max = np.inf
        model._support_inner_abs_z_max = np.inf
        model._support_inner_tilt_max = np.inf
        model._support_inner_yaw_max = np.inf
        # Store canonical basin centre directly on the model; v2 shadow code
        # in the refiner will read it via getattr(controller, ...).
        # (PyTorch nn.Module.__setattr__ may intercept plain ndarray assignment,
        #  so use object.__setattr__ as a safety measure.)
        object.__setattr__(model, "_canonical_basin_center_pose_7d",
                           np.concatenate([basin_pos, basin_quat]).astype(np.float32))
        print(f"[eval] canonical basin seeded: {basin_pos}", flush=True)
    return model


def augment_runtime_pose_field_candidate_bank(controller, yaw_probe_values=()):
    if controller is None or getattr(controller, "_controller_type", "") != "pose_field_scorer":
        return 0
    has_teacher_bank = hasattr(controller, "_teacher_candidate_actions_local")
    if has_teacher_bank:
        base_actions = np.asarray(controller._teacher_candidate_actions_local, dtype=np.float32)
        base_groups = np.asarray(controller._teacher_candidate_group_index, dtype=np.int64)
    else:
        base_actions = np.asarray(controller._candidate_actions_local.detach().cpu().numpy(), dtype=np.float32)
        base_groups = np.asarray(controller._candidate_group_index.detach().cpu().numpy(), dtype=np.int64)
    base_mask = np.asarray(
        getattr(controller, "_runtime_candidate_mask", np.ones((base_actions.shape[0],), dtype=np.float32)),
        dtype=np.float32,
    )
    base_kind = np.asarray(
        getattr(controller, "_teacher_candidate_kind", np.asarray(["base"] * base_actions.shape[0])),
        dtype=object,
    )
    group_by_key = {}
    for cand, grp in zip(base_actions, base_groups):
        key = candidate_group_key(cand)
        if key not in group_by_key:
            group_by_key[key] = int(grp)

    new_actions = []
    new_groups = []
    new_kinds = []
    for mag in yaw_probe_values:
        m = abs(float(mag))
        if m <= 0.0:
            continue
        for sign in (-1.0, 1.0):
            cand = np.zeros((6,), dtype=np.float32)
            cand[5] = sign * m
            if np.any(np.all(np.isclose(base_actions, cand[None, :], atol=1e-6), axis=1)):
                continue
            grp = group_by_key.get(candidate_group_key(cand), -1)
            if grp < 0:
                continue
            new_actions.append(cand)
            new_groups.append(grp)
            new_kinds.append("runtime_yaw_probe")
    if not new_actions:
        return 0

    aug_actions = np.concatenate([base_actions, np.stack(new_actions, axis=0)], axis=0).astype(np.float32)
    aug_groups = np.concatenate([base_groups, np.asarray(new_groups, dtype=np.int64)], axis=0)
    aug_mask = np.concatenate([base_mask, np.ones((len(new_actions),), dtype=np.float32)], axis=0)
    aug_kind = np.concatenate([base_kind, np.asarray(new_kinds, dtype=object)], axis=0)
    controller._candidate_actions_local = torch.from_numpy(aug_actions)
    controller._candidate_group_index = torch.from_numpy(aug_groups)
    controller._runtime_candidate_mask = aug_mask.astype(np.float32)
    controller._teacher_candidate_kind = aug_kind
    if has_teacher_bank:
        controller._teacher_candidate_actions_local = aug_actions
        controller._teacher_candidate_group_index = aug_groups
        controller._teacher_candidate_base_mask = aug_mask.astype(np.float32)
    controller._runtime_candidate_yaw_probe_values = tuple(float(v) for v in yaw_probe_values)
    controller._runtime_candidate_yaw_augmented = True
    print(
        f"[eval] Augmented runtime scorer candidate bank with {len(new_actions)} yaw candidates "
        f"-> total={aug_actions.shape[0]}"
    )
    return int(len(new_actions))


def maybe_load_near_ready_xyyaw_predictor(ckpt_path):
    if not ckpt_path:
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", {})
    if any(k.startswith("uncertainty_head.") or k.startswith("band_head.") for k in state_dict.keys()):
        return None
    if not any(k.startswith("metric_head.") or k.startswith("ready_head.") for k in state_dict.keys()):
        return None
    model = NearReadyXYYawPredictor().to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] Near-ready rerank compatibility: missing={missing}, unexpected={unexpected}")
    model = model.to(DEVICE).eval()
    print(f"[eval] Loaded near-ready xy+yaw predictor from {ckpt_path}")
    return model


def load_near_ready_residual_adapter(adapter_ckpt):
    ckpt = torch.load(adapter_ckpt, map_location="cpu")
    model = NearReadyResidualScoreAdapter(
        clip_rho=float(ckpt.get("clip_rho", 0.35)),
    ).to(DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] Near-ready residual adapter compatibility: missing={missing}, unexpected={unexpected}")
    model = model.to(DEVICE).eval()
    model._controller_type = "near_ready_residual_adapter"
    model._clip_rho = float(ckpt.get("clip_rho", 0.35))
    model._rank_margin = float(ckpt.get("rank_margin", 0.35))
    model._reachability_audit = ckpt.get("reachability_audit", {})
    print(f"[eval] Loaded near-ready residual adapter from {adapter_ckpt}")
    return model


def load_near_ready_group_residual_adapter(adapter_ckpt):
    ckpt = torch.load(adapter_ckpt, map_location="cpu")
    num_groups = int(ckpt.get("num_groups", 37))
    model = NearReadyGroupResidualAdapter(
        num_groups=num_groups,
        clip_rho=float(ckpt.get("clip_rho_g", ckpt.get("clip_rho", 0.35))),
    ).to(DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] Near-ready group residual adapter compatibility: missing={missing}, unexpected={unexpected}")
    model = model.to(DEVICE).eval()
    model._controller_type = "near_ready_group_residual_adapter"
    model._clip_rho_g = float(ckpt.get("clip_rho_g", ckpt.get("clip_rho", 0.35)))
    model._rank_margin_g = float(ckpt.get("rank_margin_g", ckpt.get("rank_margin", 0.35)))
    model._reachability_audit = ckpt.get("reachability_audit", {})
    print(f"[eval] Loaded near-ready group residual adapter from {adapter_ckpt}")
    return model


def load_student_group_selector_shadow(selector_ckpt, handoff_ckpt=None):
    ckpt = torch.load(selector_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    head_weight = state_dict.get("head.2.weight", None)
    num_groups = int(head_weight.shape[0]) if head_weight is not None else 37
    group_model = StudentGroupSelectorV2(num_groups=num_groups).to(DEVICE)
    missing, unexpected = group_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] B1 group selector compatibility: missing={missing}, unexpected={unexpected}")
    handoff_path = handoff_ckpt or ckpt.get("handoff_state_ckpt")
    if not handoff_path:
        raise ValueError(
            "B1 shadow selector needs a handoff-state ckpt. Provide "
            "--student_group_selector_handoff_ckpt or train ckpt metadata with handoff_state_ckpt."
        )
    handoff = torch.load(handoff_path, map_location="cpu")
    handoff_model = StudentHandoffStateHeadV2().to(DEVICE)
    missing_h, unexpected_h = handoff_model.load_state_dict(handoff.get("model_state_dict", handoff), strict=False)
    if missing_h or unexpected_h:
        print(f"[eval] B1 shadow handoff compatibility: missing={missing_h}, unexpected={unexpected_h}")
    group_model.eval()
    handoff_model.eval()
    for p in group_model.parameters():
        p.requires_grad = False
    for p in handoff_model.parameters():
        p.requires_grad = False
    print(f"[eval] Loaded B1 group selector shadow from {selector_ckpt}")
    print(f"[eval] Loaded B1 handoff latent source from {handoff_path}")
    return {
        "group_model": group_model,
        "handoff_model": handoff_model,
        "selector_ckpt": str(selector_ckpt),
        "handoff_ckpt": str(handoff_path),
    }


def load_b1_apply_gate_shadow(gate_ckpt):
    from build_b1_apply_gate_dataset import row_features
    from train_b1_apply_gate import ApplyGateMLP

    ckpt = torch.load(gate_ckpt, map_location="cpu")
    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    feature_names = [str(v) for v in ckpt.get("feature_names", [])]
    hidden_dim = int(ckpt.get("hidden_dim", 32))
    model = ApplyGateMLP(feature_mean.shape[1], hidden_dim=hidden_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def predict(row: dict) -> float:
        feat = np.asarray(row_features(row, feature_names if feature_names else None), dtype=np.float32)[None, :]
        feat = (feat - feature_mean) / feature_std
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(feat).to(DEVICE))).item()
        return float(prob)

    print(f"[eval] Loaded B1 apply gate shadow from {gate_ckpt}")
    return {
        "model": model,
        "threshold": float(ckpt.get("threshold", 0.5)),
        "selection_rule": str(ckpt.get("selection_rule", "threshold_only")),
        "veto_runtime_yaw_norm_gt": float(ckpt.get("veto_runtime_yaw_norm_gt", np.inf)),
        "veto_pred_uncertainty_gt": float(ckpt.get("veto_pred_uncertainty_gt", np.inf)),
        "veto_group_margin_lt": float(ckpt.get("veto_group_margin_lt", -np.inf)),
        "veto_baseline_groups": [int(v) for v in ckpt.get("veto_baseline_groups", [])],
        "require_runtime_yaw_norm_ge": float(ckpt.get("require_runtime_yaw_norm_ge", -np.inf)),
        "require_runtime_yaw_norm_le": float(ckpt.get("require_runtime_yaw_norm_le", np.inf)),
        "require_group_margin_ge": float(ckpt.get("require_group_margin_ge", -np.inf)),
        "require_uncertainty_ge_or_yaw_norm_le": [
            float(v) for v in ckpt.get("require_uncertainty_ge_or_yaw_norm_le", [])
        ],
        "veto_pred_group_if_yaw_norm_le_and_uncertainty_lt": [
            float(v) for v in ckpt.get("veto_pred_group_if_yaw_norm_le_and_uncertainty_lt", [])
        ],
        "predict": predict,
        "gate_ckpt": str(gate_ckpt),
    }


def load_student_candidate_evaluator_shadow(candidate_ckpt, handoff_ckpt=None, mode_input_path="summary_only"):
    ckpt = torch.load(candidate_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    yaw_mode_classes = int(ckpt.get("yaw_mode_num_classes", 2))
    model = StudentCandidateEvaluatorV2(yaw_mode_classes=yaw_mode_classes).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] B2 candidate evaluator compatibility: missing={missing}, unexpected={unexpected}")
    if hasattr(model, "set_mode_input_path"):
        model.set_mode_input_path(str(mode_input_path))

    handoff_path = handoff_ckpt or ckpt.get("handoff_state_ckpt")
    if not handoff_path:
        raise ValueError(
            "B2 candidate shadow needs a handoff-state ckpt. Provide "
            "--student_candidate_evaluator_handoff_ckpt or train ckpt metadata with handoff_state_ckpt."
        )
    handoff = torch.load(handoff_path, map_location="cpu")
    handoff_model = StudentHandoffStateHeadV2().to(DEVICE)
    missing_h, unexpected_h = handoff_model.load_state_dict(handoff.get("model_state_dict", handoff), strict=False)
    if missing_h or unexpected_h:
        print(f"[eval] B2 shadow handoff compatibility: missing={missing_h}, unexpected={unexpected_h}")
    model.eval()
    handoff_model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in handoff_model.parameters():
        p.requires_grad = False
    print(f"[eval] Loaded B2 candidate evaluator shadow from {candidate_ckpt}")
    print(f"[eval] Loaded B2 handoff latent source from {handoff_path}")
    return {
        "candidate_model": model,
        "handoff_model": handoff_model,
        "candidate_ckpt": str(candidate_ckpt),
        "handoff_ckpt": str(handoff_path),
        "yaw_mode_num_classes": int(yaw_mode_classes),
        "yaw_keep_abs": float(ckpt.get("yaw_keep_abs", 0.035)),
        "mode_input_path": str(mode_input_path),
        "shadow_policy": str(ckpt.get("shadow_policy", "b2_mode_gated")),
    }


def load_alignment_v3_shadow_controller(controller_ckpt):
    ckpt = torch.load(controller_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    max_pos = float(ckpt.get("max_pos", 0.0125))
    max_yaw = float(ckpt.get("max_yaw", 0.0020))
    model = AlignmentV3DirectLocalController(max_pos=max_pos, max_yaw=max_yaw).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] alignment v3 shadow compatibility: missing={missing}, unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[eval] Loaded alignment v3 shadow controller from {controller_ckpt}")
    return {
        "model": model,
        "controller_ckpt": str(controller_ckpt),
        "max_pos": max_pos,
        "max_yaw": max_yaw,
        "controller_type": "alignment_v3_direct_local_controller",
    }


def load_alignment_v4_shadow_controller(controller_ckpt):
    ckpt = torch.load(controller_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    max_pos = float(ckpt.get("max_pos", 0.015))
    max_yaw = float(ckpt.get("max_yaw", 0.006))
    use_front_rgb = bool(ckpt.get("use_front_rgb", False))
    use_planner_action = bool(ckpt.get("use_planner_action", True))
    use_force = bool(ckpt.get("use_force", True))
    model = AlignmentV4DirectLocalController(
        max_pos=max_pos,
        max_yaw=max_yaw,
        use_front_rgb=use_front_rgb,
        use_planner_action=use_planner_action,
        use_force=use_force,
    ).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] alignment v4 shadow compatibility: missing={missing}, unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[eval] Loaded alignment v4 shadow controller from {controller_ckpt}")
    return {
        "model": model,
        "controller_ckpt": str(controller_ckpt),
        "max_pos": max_pos,
        "max_yaw": max_yaw,
        "use_front_rgb": use_front_rgb,
        "use_planner_action": use_planner_action,
        "use_force": use_force,
        "controller_type": "alignment_v4_direct_local_controller",
    }


def load_alignment_diffusion_controller(controller_ckpt, *, horizon=None, max_pos_step=None, max_yaw_step=None):
    ckpt = torch.load(controller_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_horizon = int(horizon if horizon is not None else ckpt.get("horizon", 8))
    model_max_pos = float(max_pos_step if max_pos_step is not None else ckpt.get("max_pos_step", 0.0015))
    model_max_yaw = float(max_yaw_step if max_yaw_step is not None else ckpt.get("max_yaw_step", 0.0060))
    model = AlignmentDiffusionRefiner(
        horizon=model_horizon,
        max_pos_step=model_max_pos,
        max_yaw_step=model_max_yaw,
        use_front_rgb=bool(ckpt.get("use_front_rgb", False)),
        use_wrist_rgb=bool(ckpt.get("use_wrist_rgb", True)),
        use_wrist_depth=bool(ckpt.get("use_wrist_depth", True)),
        use_force=bool(ckpt.get("use_force", True)),
        use_planner_action=bool(ckpt.get("use_planner_action", True)),
    ).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] alignment diffusion compatibility: missing={missing}, unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[eval] Loaded alignment diffusion refiner from {controller_ckpt}")
    return {
        "model": model,
        "controller_ckpt": str(controller_ckpt),
        "horizon": model_horizon,
        "max_pos_step": model_max_pos,
        "max_yaw_step": model_max_yaw,
        "controller_type": "alignment_diffusion_refiner",
    }


def load_alignment_tc_diffusion_controller(controller_ckpt, *, horizon=None, max_pos_step=None, max_yaw_step=None):
    ckpt = torch.load(controller_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_horizon = int(horizon if horizon is not None else ckpt.get("horizon", 8))
    model_max_pos = float(max_pos_step if max_pos_step is not None else ckpt.get("max_pos_step", 0.0015))
    model_max_yaw = float(max_yaw_step if max_yaw_step is not None else ckpt.get("max_yaw_step", 0.0060))
    model = TargetConditionedAlignmentDiffusionRefiner(
        horizon=model_horizon,
        max_pos_step=model_max_pos,
        max_yaw_step=model_max_yaw,
        use_front_rgb=bool(ckpt.get("use_front_rgb", False)),
        use_wrist_rgb=bool(ckpt.get("use_wrist_rgb", True)),
        use_wrist_depth=bool(ckpt.get("use_wrist_depth", True)),
        use_force=bool(ckpt.get("use_force", True)),
        use_planner_action=bool(ckpt.get("use_planner_action", True)),
    ).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] target-conditioned diffusion compatibility: missing={missing}, unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[eval] Loaded target-conditioned diffusion refiner from {controller_ckpt}")
    return {
        "model": model,
        "controller_ckpt": str(controller_ckpt),
        "horizon": model_horizon,
        "max_pos_step": model_max_pos,
        "max_yaw_step": model_max_yaw,
        "controller_type": "alignment_tc_diffusion_refiner",
    }


def load_alignment_tc_student_vnext_controller(controller_ckpt, *, horizon=None, max_pos_step=None, max_yaw_step=None):
    ckpt = torch.load(controller_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_horizon = int(horizon if horizon is not None else ckpt.get("horizon", 8))
    model_max_pos = float(max_pos_step if max_pos_step is not None else ckpt.get("max_pos_step", 0.0015))
    model_max_yaw = float(max_yaw_step if max_yaw_step is not None else ckpt.get("max_yaw_step", 0.0060))
    model_y_bridge = float(ckpt.get("y_bridge_max_step", model_max_pos * 0.85))
    model = AlignmentTCStudentVNext(
        horizon=model_horizon,
        max_pos_step=model_max_pos,
        max_yaw_step=model_max_yaw,
        y_bridge_max_step=model_y_bridge,
        use_front_rgb=bool(ckpt.get("use_front_rgb", False)),
        use_wrist_rgb=bool(ckpt.get("use_wrist_rgb", True)),
        use_wrist_depth=bool(ckpt.get("use_wrist_depth", True)),
        use_force=bool(ckpt.get("use_force", True)),
        use_planner_action=bool(ckpt.get("use_planner_action", True)),
    ).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] target-conditioned student vNext compatibility: missing={missing}, unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[eval] Loaded target-conditioned student vNext from {controller_ckpt}")
    return {
        "model": model,
        "controller_ckpt": str(controller_ckpt),
        "horizon": model_horizon,
        "max_pos_step": model_max_pos,
        "max_yaw_step": model_max_yaw,
        "y_bridge_max_step": model_y_bridge,
        "controller_type": "alignment_tc_student_vnext",
    }


def _active_alignment_like_controller(refiner):
    if refiner is None:
        return None
    for attr_name in (
        "alignment_tc_student_vnext_controller",
        "alignment_tc_diffusion_controller",
        "alignment_diffusion_controller",
        "alignment_v4_shadow_controller",
        "alignment_controller",
    ):
        controller = getattr(refiner, attr_name, None)
        if controller is not None:
            return controller
    return None


def _resolve_optional_controller_path(path):
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        for candidate_name in (
            "pose_field_scorer_best_fire_tradeoff.pt",
            "pose_field_scorer_final.pt",
            "pose_field_scorer_best_pose.pt",
            "near_ready_residual_adapter_best.pt",
            "near_ready_residual_adapter_final.pt",
            "near_ready_group_residual_adapter_best.pt",
            "near_ready_group_residual_adapter_final.pt",
            "residual_final.pt",
        ):
            candidate = p / candidate_name
            if candidate.exists():
                return str(candidate)
        return str(p)
    if p.name == "pose_field_scorer_final.pt":
        tradeoff = p.with_name("pose_field_scorer_best_fire_tradeoff.pt")
        if tradeoff.exists():
            return str(tradeoff)
    return str(p)


def _load_optional_controller(path):
    if path is None:
        return None
    path = _resolve_optional_controller_path(path)
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    if (
        isinstance(ckpt, dict)
        and "proposal_count" in ckpt
        and "model_kwargs" in ckpt
        and any(k.startswith("multi_head_score_head.") or k.startswith("proposal_head.") for k in state_dict.keys())
    ):
        return load_depth_force_local_proposal_policy(path)
    module_type = ckpt.get("module_type", "")
    if module_type == "pose_field_scorer":
        return load_pose_field_scorer(path)
    if module_type == "near_ready_residual_adapter":
        return load_near_ready_residual_adapter(path)
    if module_type == "near_ready_group_residual_adapter":
        return load_near_ready_group_residual_adapter(path)
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


def apply_executed_local_delta_to_pose(pose_7d, delta_local_6d):
    """Roll out the same local-delta convention used by RLBench execution.

    Translation and rotvec are first converted from EE-local to world, then the
    world rotvec is left-multiplied onto the current gripper quaternion. This
    mirrors ``local_delta_to_world(...)`` followed by ``delta_to_absolute(...)``;
    using a different convention in the oracle scorer can invert yaw decisions.
    """
    pose = np.asarray(pose_7d, dtype=np.float32).copy().reshape(7)
    delta = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    delta_world = local_delta_to_world(delta, pose[3:7]).astype(np.float32)
    pose[:3] = pose[:3] + delta_world[:3]
    r_cur = Rotation.from_quat(pose[3:7])
    r_delta = Rotation.from_rotvec(delta_world[3:6])
    pose[3:7] = (r_delta * r_cur).as_quat().astype(np.float32)
    return pose.astype(np.float32)


def resolve_live_target_handle(task):
    task_obj = getattr(task, "_task", task)
    if hasattr(task_obj, "_square_ring"):
        return task_obj._square_ring
    graspables = getattr(task_obj, "_graspable_objects", None)
    if graspables:
        return graspables[0]
    return None


def safe_live_target_pose_7d(live_target_handle):
    if live_target_handle is None:
        return None
    try:
        return np.asarray(live_target_handle.get_pose(), dtype=np.float32).reshape(7)
    except Exception:
        return None


def safe_task_low_dim_pose_7d(obs):
    low = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).reshape(-1)
    if low.size >= 7 and np.all(np.isfinite(low[:7])):
        return low[:7].astype(np.float32)
    return None


def safe_task_low_dim_pose_7d_from_task(task):
    if task is None:
        return None
    try:
        low = np.asarray(task.get_low_dim_state(), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if low.size >= 7 and np.all(np.isfinite(low[:7])):
        return low[:7].astype(np.float32)
    return None


def build_object_center_target_pose(current_object_pose_7d, anchor_pose_7d):
    current_object_pose = np.asarray(current_object_pose_7d, dtype=np.float32).reshape(7)
    anchor_pose = np.asarray(anchor_pose_7d, dtype=np.float32).reshape(7)
    target_pose = anchor_pose.copy()
    target_pose[:2] = current_object_pose[:2]
    # For the current grasp-style pre-contact stage, object z is effectively stable
    # across randomized placements, so keeping the canonical anchor z is a robust
    # task-agnostic approximation of the "above-object" grasp target.
    target_pose[2] = float(anchor_pose[2])
    return target_pose.astype(np.float32)


def pose_delta_local_between(current_pose_7d, target_pose_7d):
    current_pose_7d = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
    target_pose_7d = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
    delta_pos_world = target_pose_7d[:3] - current_pose_7d[:3]
    r_cur = Rotation.from_quat(current_pose_7d[3:7])
    r_tgt = Rotation.from_quat(target_pose_7d[3:7])
    delta_rot = (r_tgt * r_cur.inv()).as_rotvec().astype(np.float32)
    delta_pos_local = r_cur.inv().apply(delta_pos_world.astype(np.float32)).astype(np.float32)
    return np.concatenate([delta_pos_local, delta_rot], axis=0).astype(np.float32)


def compute_basin_metrics(delta_basin_target, r_xy=0.008, r_z=0.01, r_yaw=0.05, r_tilt=0.12):
    delta_arr = np.asarray(delta_basin_target, dtype=np.float32).reshape(-1)
    e_xy = float(np.linalg.norm(delta_arr[:2])) if delta_arr.size >= 2 else 0.0
    e_z = float(abs(delta_arr[2])) if delta_arr.size >= 3 else 0.0
    e_tilt = float(np.linalg.norm(delta_arr[3:5])) if delta_arr.size >= 5 else 0.0
    e_yaw = float(abs(delta_arr[5])) if delta_arr.size >= 6 else 0.0
    use_tilt = bool(np.isfinite(float(r_tilt)) and float(r_tilt) > 0.0)
    basin_distance = max(
        e_xy / max(float(r_xy), 1e-6),
        e_z / max(float(r_z), 1e-6),
        (e_tilt / max(float(r_tilt), 1e-6)) if use_tilt else 0.0,
        e_yaw / max(float(r_yaw), 1e-6),
    )
    return basin_distance, e_xy, e_z, e_yaw, e_tilt


def load_grasp_basin_profile(args):
    profile_path = getattr(args, "grasp_basin_profile_json", None)
    if not profile_path:
        return None
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"grasp basin profile not found: {path}")
    profile = json.loads(path.read_text())
    profile = dict(profile) if isinstance(profile, dict) else {}
    overrides = {
        "teacher_close_xy_threshold": profile.get("close_xy_threshold"),
        "teacher_close_abs_z_threshold": profile.get("close_abs_z_threshold"),
        "teacher_close_yaw_threshold": profile.get("close_yaw_threshold"),
        "teacher_grasp_xy_threshold": profile.get("grasp_xy_threshold"),
        "teacher_grasp_abs_z_threshold": profile.get("grasp_abs_z_threshold"),
        "teacher_grasp_yaw_threshold": profile.get("grasp_yaw_threshold"),
        "teacher_grasp_ready_threshold": profile.get("grasp_ready_threshold"),
        "teacher_close_contact_depth_threshold": profile.get("close_contact_depth_threshold"),
        "teacher_close_basin_distance_threshold": profile.get("close_basin_distance_threshold"),
    }
    close_abs_z_stats = profile.get("close_abs_z_stats") if isinstance(profile.get("close_abs_z_stats"), dict) else {}
    close_xy_stats = profile.get("close_xy_stats") if isinstance(profile.get("close_xy_stats"), dict) else {}
    close_yaw_stats = profile.get("close_yaw_stats") if isinstance(profile.get("close_yaw_stats"), dict) else {}
    grasp_abs_z_stats = close_abs_z_stats
    min_close_abs_z = float(profile.get("min_close_abs_z_threshold", 0.006) or 0.006)
    demo_close_abs_z_p90 = close_abs_z_stats.get("p90")
    if demo_close_abs_z_p90 is not None:
        try:
            overrides["teacher_close_abs_z_threshold"] = min(
                float(overrides["teacher_close_abs_z_threshold"] or 0.012),
                max(float(demo_close_abs_z_p90) * 2.0, min_close_abs_z),
            )
        except Exception:
            pass
    demo_close_xy_p90 = close_xy_stats.get("p90")
    if demo_close_xy_p90 is not None:
        try:
            overrides["teacher_close_xy_threshold"] = min(
                float(overrides["teacher_close_xy_threshold"] or 0.004),
                max(float(demo_close_xy_p90) * 2.0, 0.001),
            )
        except Exception:
            pass
    demo_close_yaw_p90 = close_yaw_stats.get("p90")
    if demo_close_yaw_p90 is not None:
        try:
            overrides["teacher_close_yaw_threshold"] = min(
                float(overrides["teacher_close_yaw_threshold"] or 0.04),
                max(float(demo_close_yaw_p90) * 2.0, 0.005),
            )
        except Exception:
            pass
    demo_grasp_abs_z_p90 = grasp_abs_z_stats.get("p90")
    if demo_grasp_abs_z_p90 is not None:
        try:
            overrides["teacher_grasp_abs_z_threshold"] = min(
                float(overrides["teacher_grasp_abs_z_threshold"] or 0.012),
                max(float(demo_grasp_abs_z_p90) * 2.0, 0.003),
            )
        except Exception:
            pass
    demo_grasp_xy_p90 = close_xy_stats.get("p90")
    if demo_grasp_xy_p90 is not None:
        try:
            overrides["teacher_grasp_xy_threshold"] = min(
                float(overrides["teacher_grasp_xy_threshold"] or 0.004),
                max(float(demo_grasp_xy_p90) * 2.0, 0.001),
            )
        except Exception:
            pass
    demo_grasp_yaw_p90 = close_yaw_stats.get("p90")
    if demo_grasp_yaw_p90 is not None:
        try:
            overrides["teacher_grasp_yaw_threshold"] = min(
                float(overrides["teacher_grasp_yaw_threshold"] or 0.04),
                max(float(demo_grasp_yaw_p90) * 2.0, 0.005),
            )
        except Exception:
            pass
    for key, value in overrides.items():
        if value is None:
            continue
        try:
            parsed = float(value)
            if key in {"teacher_close_abs_z_threshold", "teacher_grasp_abs_z_threshold"}:
                parsed = min(parsed, 0.012)
            setattr(args, key, parsed)
        except Exception:
            continue
    if "profile_source" in profile:
        setattr(args, "teacher_close_basin_source", str(profile["profile_source"]))
    else:
        setattr(args, "teacher_close_basin_source", str(path))
    npz_path = path.with_suffix(".npz")
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                phase = np.asarray(npz["phase"]).astype(str) if "phase" in npz.files else None
                delta = (
                    np.asarray(npz["object_to_gripper_delta_local_6d"], dtype=np.float32)
                    if "object_to_gripper_delta_local_6d" in npz.files
                    else None
                )
                if phase is not None and delta is not None:
                    mask = phase == "close"
                    close_delta = delta[mask]
                    close_delta = close_delta[np.all(np.isfinite(close_delta), axis=1)]
                    if close_delta.size > 0:
                        profile["close_object_to_gripper_delta_local_6d_median"] = (
                            np.nanmedian(close_delta, axis=0).astype(np.float32).tolist()
                        )
        except Exception:
            pass
    setattr(args, "_grasp_basin_profile", profile)
    return profile


def compute_tc_close_contact_contract(
    *,
    xy_val: float | None,
    abs_z_val: float | None,
    yaw_val: float | None,
    depth_proximity: float | None,
    stage_contact_state: int | None,
    refiner_contact_state: int | None,
    close_xy_threshold: float,
    close_abs_z_threshold: float,
    close_yaw_threshold: float,
    close_contact_depth_threshold: float,
    low_visibility: bool = False,
    occluded: bool = False,
) -> dict[str, float | bool]:
    xy_ready = bool(xy_val is not None and float(xy_val) <= float(close_xy_threshold))
    z_ready = bool(abs_z_val is not None and float(abs_z_val) <= float(close_abs_z_threshold))
    yaw_ready = bool(close_yaw_threshold < 0.0 or (yaw_val is not None and float(yaw_val) <= float(close_yaw_threshold)))
    geometry_ready = bool(xy_ready and z_ready and yaw_ready)

    stage_ready = False
    stage_contact_candidates = []
    if stage_contact_state is not None:
        stage_contact_candidates.append(int(stage_contact_state))
    if refiner_contact_state is not None:
        stage_contact_candidates.append(int(refiner_contact_state))
    if stage_contact_candidates:
        stage_ready = bool(max(stage_contact_candidates) >= int(ContactState.NEAR_CONTACT))

    depth_ready = bool(
        depth_proximity is not None
        and np.isfinite(float(depth_proximity))
        and float(depth_proximity) <= float(close_contact_depth_threshold)
    )
    visibility_ok = bool(not low_visibility and not occluded)
    contact_ready = bool(geometry_ready and (stage_ready or depth_ready))
    confidence = 1.0
    if occluded:
        confidence -= 0.35
    if low_visibility:
        confidence -= 0.20
    if not stage_ready and not depth_ready:
        confidence -= 0.15
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return {
        "xy_ready": bool(xy_ready),
        "z_ready": bool(z_ready),
        "yaw_ready": bool(yaw_ready),
        "geometry_ready": bool(geometry_ready),
        "stage_ready": bool(stage_ready),
        "depth_ready": bool(depth_ready),
        "visibility_ok": bool(visibility_ok),
        "contact_ready": bool(contact_ready),
        "confidence": float(confidence),
    }


def compute_tc_grasp_expert_contract(
    *,
    current_pose_7d: np.ndarray,
    object_pose_7d: np.ndarray | None,
    gripper_open: float | None,
    depth_proximity: float | None,
    stage_contact_state: int | None,
    refiner_contact_state: int | None,
    close_xy_threshold: float,
    close_abs_z_threshold: float,
    close_yaw_threshold: float,
    close_contact_depth_threshold: float,
    low_visibility: bool = False,
    occluded: bool = False,
    grasp_ready_threshold: float = 0.55,
    desired_object_delta_local_6d: np.ndarray | None = None,
) -> dict[str, float | bool | np.ndarray]:
    current_pose = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
    object_pose = None if object_pose_7d is None else np.asarray(object_pose_7d, dtype=np.float32).reshape(7)
    if object_pose is None or object_pose.size < 7 or not np.all(np.isfinite(object_pose[:7])):
        nan_pose = np.full((7,), np.nan, dtype=np.float32)
        return {
            "gripper_finger_pose_7d": current_pose.astype(np.float32),
            "object_in_finger_region": False,
            "finger_object_lateral_error": float("nan"),
            "finger_object_height_overlap": float("nan"),
            "finger_object_yaw_error": float("nan"),
            "gripper_aperture_ready": bool(gripper_open is None or float(gripper_open) >= 0.5),
            "contact_ready": False,
            "depth_ready": False,
            "stage_ready": False,
            "visibility_ok": bool(not low_visibility and not occluded),
            "grasp_ready": False,
            "grasp_readiness_score": 0.0,
            "grasp_readiness_reason": "missing_object_pose",
            "object_pose_7d": nan_pose,
            "finger_xy_threshold": float(max(float(close_xy_threshold) * 1.25, 1e-4)),
            "finger_abs_z_threshold": float(max(float(close_abs_z_threshold) * 1.20, 1e-4)),
            "finger_yaw_threshold": float(max(float(close_yaw_threshold) * 1.20 if float(close_yaw_threshold) >= 0.0 else 0.05, 1e-4)),
        }

    delta = pose_delta_local_between(current_pose, object_pose)
    desired_delta = None
    if desired_object_delta_local_6d is not None:
        desired_delta = np.asarray(desired_object_delta_local_6d, dtype=np.float32).reshape(-1)[:6]
        if desired_delta.size < 6 or not np.all(np.isfinite(desired_delta[:6])):
            desired_delta = None
    error_delta = delta.copy()
    if desired_delta is not None:
        error_delta[:6] = delta[:6] - desired_delta[:6]
        error_delta[5] = float((float(error_delta[5]) + np.pi) % (2.0 * np.pi) - np.pi)
    xy = float(np.linalg.norm(error_delta[:2]))
    abs_z = float(abs(error_delta[2]))
    yaw = float(abs(error_delta[5]))
    finger_xy_threshold = float(max(float(close_xy_threshold) * 1.25, 1e-4))
    finger_abs_z_threshold = float(max(float(close_abs_z_threshold) * 1.20, 1e-4))
    finger_yaw_threshold = float(
        max(float(close_yaw_threshold) * 1.20 if float(close_yaw_threshold) >= 0.0 else 0.05, 1e-4)
    )
    gripper_aperture_ready = bool(gripper_open is None or float(gripper_open) >= 0.5)
    stage_candidates = []
    if stage_contact_state is not None:
        stage_candidates.append(int(stage_contact_state))
    if refiner_contact_state is not None:
        stage_candidates.append(int(refiner_contact_state))
    stage_ready = bool(stage_candidates and max(stage_candidates) >= int(ContactState.NEAR_CONTACT))
    depth_ready = bool(
        depth_proximity is not None
        and np.isfinite(float(depth_proximity))
        and float(depth_proximity) <= float(close_contact_depth_threshold)
    )
    contact_ready = bool(stage_ready or depth_ready)
    object_in_finger_region = bool(
        xy <= finger_xy_threshold and abs_z <= finger_abs_z_threshold and yaw <= finger_yaw_threshold
    )
    lateral_score = float(np.clip(1.0 - xy / max(finger_xy_threshold, 1e-6), 0.0, 1.0))
    height_score = float(np.clip(1.0 - abs_z / max(finger_abs_z_threshold, 1e-6), 0.0, 1.0))
    yaw_score = float(np.clip(1.0 - yaw / max(finger_yaw_threshold, 1e-6), 0.0, 1.0))
    geometry_score = float(0.45 * lateral_score + 0.35 * height_score + 0.20 * yaw_score)
    visibility_score = 1.0
    if occluded:
        visibility_score -= 0.25
    if low_visibility:
        visibility_score -= 0.15
    if not stage_ready and not depth_ready:
        visibility_score -= 0.10
    visibility_score = float(np.clip(visibility_score, 0.0, 1.0))
    contact_score = 1.0 if contact_ready else 0.0
    aperture_score = 1.0 if gripper_aperture_ready else 0.0
    grasp_readiness_score = float(
        np.clip(
            0.40 * geometry_score + 0.25 * contact_score + 0.20 * aperture_score + 0.15 * visibility_score,
            0.0,
            1.0,
        )
    )
    # Grasp readiness is a hard close-basin contract. Earlier versions treated
    # object_in_finger_region as soft, which allowed close labels outside the
    # actual attach basin and produced failed positive supervision.
    grasp_ready = bool(
        gripper_aperture_ready
        and contact_ready
        and object_in_finger_region
        and grasp_readiness_score >= float(grasp_ready_threshold)
    )
    if not object_in_finger_region:
        reason = "finger_region_soft"
    elif not gripper_aperture_ready:
        reason = "aperture"
    elif not contact_ready:
        reason = "contact"
    elif grasp_readiness_score < float(grasp_ready_threshold):
        reason = "score"
    else:
        reason = "ready"
    return {
        "gripper_finger_pose_7d": current_pose.astype(np.float32),
        "object_pose_7d": object_pose.astype(np.float32),
        "object_to_gripper_delta_local_6d": delta.astype(np.float32),
        "object_to_gripper_error_delta_local_6d": error_delta.astype(np.float32),
        "desired_object_to_gripper_delta_local_6d": (
            np.asarray(desired_delta, dtype=np.float32) if desired_delta is not None else np.full((6,), np.nan, dtype=np.float32)
        ),
        "object_in_finger_region": bool(object_in_finger_region),
        "finger_object_lateral_error": float(xy),
        "finger_object_height_overlap": float(np.clip(height_score, 0.0, 1.0)),
        "finger_object_yaw_error": float(yaw),
        "gripper_aperture_ready": bool(gripper_aperture_ready),
        "contact_ready": bool(contact_ready),
        "depth_ready": bool(depth_ready),
        "stage_ready": bool(stage_ready),
        "visibility_ok": bool(not low_visibility and not occluded),
        "grasp_ready": bool(grasp_ready),
        "grasp_readiness_score": float(grasp_readiness_score),
        "grasp_readiness_reason": str(reason),
        "finger_xy_threshold": float(finger_xy_threshold),
        "finger_abs_z_threshold": float(finger_abs_z_threshold),
        "finger_yaw_threshold": float(finger_yaw_threshold),
    }


def select_tc_expert_sequence_name(
    *,
    teacher_grasp_verified_now: bool,
    teacher_close_ready_all: bool,
    teacher_close_contact_ready: bool,
    teacher_close_yaw_ok: bool,
    teacher_yaw_correct_ready: bool,
    teacher_enter_finger_region_ready: bool,
    teacher_xy_correct_ready: bool,
    teacher_preclose_descend_ready: bool,
    teacher_alignment_ready_now: bool,
    teacher_verify_active_now: bool,
    teacher_close_hold_remaining: int,
    teacher_close_attempted_this_cycle: bool,
    teacher_yaw_imitation_enabled: bool,
    teacher_grasp_ready: bool,
) -> str:
    if teacher_grasp_verified_now:
        return "verified_success"
    if teacher_verify_active_now or int(teacher_close_hold_remaining) > 0:
        return "lift_verify"
    if bool(teacher_close_attempted_this_cycle) and not teacher_grasp_verified_now:
        return "settle"
    if teacher_close_ready_all:
        return "close"
    if teacher_yaw_correct_ready and teacher_yaw_imitation_enabled:
        return "yaw_correct"
    if teacher_enter_finger_region_ready:
        return "enter_finger_region"
    if teacher_xy_correct_ready:
        return "xy_correct"
    if teacher_preclose_descend_ready:
        return "descend_z"
    if teacher_alignment_ready_now or teacher_grasp_ready:
        return "align_xy"
    return "align_xy"


def _tc_teacher_action4(action6: np.ndarray) -> np.ndarray:
    action = np.asarray(action6, dtype=np.float32).reshape(-1)
    out = np.zeros((4,), dtype=np.float32)
    out[: min(3, action.size)] = action[: min(3, action.size)]
    if action.size >= 6:
        out[3] = action[5]
    return out


def _clip_tc_teacher_action(action6: np.ndarray, *, max_pos: float, max_yaw: float) -> np.ndarray:
    out = np.asarray(action6, dtype=np.float32).reshape(6).copy()
    out[3:5] = 0.0
    pos_norm = float(np.linalg.norm(out[:3]))
    if pos_norm > float(max_pos) > 0.0:
        out[:3] *= float(max_pos) / max(pos_norm, 1e-8)
    out[5] = float(np.clip(out[5], -float(max_yaw), float(max_yaw)))
    return out.astype(np.float32)


def _build_tc_hard_stage_action(
    *,
    phase: str,
    current_delta_local_6d: np.ndarray | None,
    grasp_delta_local_6d: np.ndarray | None,
    yaw_raw_val: float | None,
    yaw_control_sign: float,
    bucket: str,
    args,
) -> np.ndarray:
    phase = str(phase or "align_xy_yaw")
    current_delta = (
        np.asarray(current_delta_local_6d, dtype=np.float32).reshape(6)
        if current_delta_local_6d is not None
        else np.zeros((6,), dtype=np.float32)
    )
    grasp_delta = (
        np.asarray(grasp_delta_local_6d, dtype=np.float32).reshape(6)
        if grasp_delta_local_6d is not None
        else current_delta
    )
    if bucket == "micro_contact_refine":
        max_pos = float(getattr(args, "alignment_tc_teacher_micro_max_pos_step", 0.0015))
        max_yaw = float(getattr(args, "alignment_tc_teacher_micro_max_yaw_step", 0.008))
    elif bucket == "near_contact_refine":
        max_pos = float(getattr(args, "alignment_tc_teacher_near_max_pos_step", 0.0030))
        max_yaw = float(getattr(args, "alignment_tc_teacher_near_max_yaw_step", 0.012))
    else:
        max_pos = float(getattr(args, "alignment_tc_teacher_broad_max_pos_step", 0.0045))
        max_yaw = float(getattr(args, "alignment_tc_teacher_broad_max_yaw_step", 0.018))
    yaw_gain = max(float(getattr(args, "teacher_smooth_kp_yaw", 0.35)), 0.80)
    xy_gain = max(float(getattr(args, "teacher_smooth_kp_xy", 0.45)), 0.80)
    z_gain = max(float(getattr(args, "teacher_smooth_kp_z", 0.55)), 0.70)
    out = np.zeros((6,), dtype=np.float32)
    if phase == "yaw_correct":
        yaw_val = 0.0 if yaw_raw_val is None or not np.isfinite(float(yaw_raw_val)) else float(yaw_raw_val)
        out[:2] = np.clip(0.50 * xy_gain * current_delta[:2], -0.50 * max_pos, 0.50 * max_pos)
        out[2] = float(np.clip(0.50 * z_gain * current_delta[2], -0.50 * max_pos, 0.50 * max_pos))
        out[5] = float(np.clip(float(yaw_control_sign) * yaw_gain * yaw_val, -max_yaw, max_yaw))
    elif phase == "enter_finger_region":
        out[:2] = np.clip(xy_gain * grasp_delta[:2], -max_pos, max_pos)
        out[2] = float(np.clip(0.25 * z_gain * grasp_delta[2], -0.5 * max_pos, 0.5 * max_pos))
    elif phase == "xy_correct":
        out[:2] = np.clip(xy_gain * current_delta[:2], -max_pos, max_pos)
    elif phase == "descend_z":
        out[2] = float(np.clip(z_gain * grasp_delta[2], -max_pos, max_pos))
    else:
        out[:3] = np.clip(
            np.asarray(
                [xy_gain * current_delta[0], xy_gain * current_delta[1], z_gain * current_delta[2]],
                dtype=np.float32,
            ),
            -max_pos,
            max_pos,
        )
        out[5] = float(np.clip(yaw_gain * float(yaw_control_sign) * float(current_delta[5]), -max_yaw, max_yaw))
    return _clip_tc_teacher_action(out, max_pos=max_pos, max_yaw=max_yaw)


def build_tc_privileged_teacher_candidates(
    delta_local_6d: np.ndarray,
    planner_action_local_6d: np.ndarray,
    *,
    bucket: str,
    yaw_sign: float,
    phase: str,
    yaw_imitation_enabled: bool,
    args,
) -> tuple[np.ndarray, list[str], float, float]:
    delta = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    planner = np.asarray(planner_action_local_6d, dtype=np.float32).reshape(-1)
    planner6 = np.zeros((6,), dtype=np.float32)
    planner6[: min(6, planner.size)] = planner[: min(6, planner.size)]
    phase = str(phase or "align_xy_yaw")
    if bucket == "micro_contact_refine":
        max_pos = float(getattr(args, "alignment_tc_teacher_micro_max_pos_step", 0.0015))
        max_yaw = float(getattr(args, "alignment_tc_teacher_micro_max_yaw_step", 0.008))
        xy_probes = [0.0008, 0.0012, max_pos]
        z_probes = [0.0006, 0.0010, max_pos]
        yaw_probes = [0.003, 0.006, max_yaw]
        k_xy, k_z, k_yaw = 0.32, 0.14, 0.30
    elif bucket == "near_contact_refine":
        max_pos = float(getattr(args, "alignment_tc_teacher_near_max_pos_step", 0.0030))
        max_yaw = float(getattr(args, "alignment_tc_teacher_near_max_yaw_step", 0.012))
        xy_probes = [0.0015, 0.0022, max_pos]
        z_probes = [0.0012, 0.0020, max_pos]
        yaw_probes = [0.006, 0.010, max_yaw] if bool(yaw_imitation_enabled) else []
        k_xy, k_z, k_yaw = 0.38, 0.18, 0.30
    else:
        max_pos = float(getattr(args, "alignment_tc_teacher_broad_max_pos_step", 0.0045))
        max_yaw = float(getattr(args, "alignment_tc_teacher_broad_max_yaw_step", 0.018))
        xy_probes = [0.0020, 0.0032, max_pos]
        z_probes = [0.0015, 0.0030, max_pos]
        yaw_probes = [0.008, 0.014, max_yaw] if bool(yaw_imitation_enabled) else []
        k_xy, k_z, k_yaw = 0.42, 0.30, 0.32
    candidates: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, action) -> None:
        candidates.append(_clip_tc_teacher_action(action, max_pos=max_pos, max_yaw=max_yaw))
        names.append(name)

    add("noop", np.zeros((6,), dtype=np.float32))
    add("planner", planner6)
    for scale in (0.75, 1.0, 1.35):
        servo = np.zeros((6,), dtype=np.float32)
        servo[:2] = float(scale) * k_xy * delta[:2]
        servo[2] = float(scale) * k_z * delta[2]
        servo[5] = float(scale) * yaw_sign * k_yaw * delta[5] if bool(yaw_imitation_enabled) else 0.0
        add(f"servo_{scale:.2f}", servo)
    for mag in xy_probes:
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                a = np.zeros((6,), dtype=np.float32)
                a[axis] = sign * float(mag)
                add(f"xy_axis{axis}_{sign:+.0f}_{mag:g}", a)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            a = np.zeros((6,), dtype=np.float32)
            a[0] = sx * float(mag) / np.sqrt(2.0)
            a[1] = sy * float(mag) / np.sqrt(2.0)
            add(f"xy_diag_{sx:+d}_{sy:+d}_{mag:g}", a)
    for mag in z_probes:
        for sign in (-1.0, 1.0):
            a = np.zeros((6,), dtype=np.float32)
            a[2] = sign * float(mag)
            add(f"z_{sign:+.0f}_{mag:g}", a)
    for mag in yaw_probes:
        for sign in (-1.0, 1.0):
            a = np.zeros((6,), dtype=np.float32)
            a[5] = sign * float(mag)
            add(f"yaw_{sign:+.0f}_{mag:g}", a)
    for scale in (0.6, 1.0):
        a = np.zeros((6,), dtype=np.float32)
        a[:2] = float(scale) * k_xy * delta[:2]
        a[5] = float(scale) * yaw_sign * k_yaw * delta[5] if bool(yaw_imitation_enabled) else 0.0
        add(f"xy_yaw_{scale:.1f}", a)
        a = np.zeros((6,), dtype=np.float32)
        a[:3] = float(scale) * np.asarray([k_xy * delta[0], k_xy * delta[1], k_z * delta[2]], dtype=np.float32)
        add(f"xyz_{scale:.1f}", a)
    return np.stack(candidates, axis=0).astype(np.float32), names, float(max_pos), float(max_yaw)


def select_tc_privileged_teacher_action(
    *,
    current_pose_7d: np.ndarray,
    target_pose_7d: np.ndarray,
    current_delta_local_6d: np.ndarray,
    planner_action_local_6d: np.ndarray,
    object_pose_7d: np.ndarray | None = None,
    safety,
    bucket: str,
    yaw_sign: float,
    yaw_imitation_enabled: bool,
    phase: str,
    grasp_readiness_score: float | None = None,
    args,
) -> dict:
    candidates, names, max_pos, max_yaw = build_tc_privileged_teacher_candidates(
        current_delta_local_6d,
        planner_action_local_6d,
        bucket=bucket,
        yaw_sign=yaw_sign,
        phase=phase,
        yaw_imitation_enabled=yaw_imitation_enabled,
        args=args,
    )
    current_pose = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
    target_pose = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
    current_delta = np.asarray(current_delta_local_6d, dtype=np.float32).reshape(6)
    pre_xy = float(np.linalg.norm(current_delta[:2]))
    pre_z = float(abs(current_delta[2]))
    pre_yaw = float(abs(current_delta[5]))
    already_close = bool(
        pre_xy <= float(getattr(args, "alignment_tc_teacher_already_close_xy", 0.0025))
        and pre_z <= float(getattr(args, "alignment_tc_teacher_already_close_abs_z", 0.0030))
        and pre_yaw <= float(getattr(args, "alignment_tc_teacher_already_close_yaw", 0.015))
    )
    min_pos_step = (
        float(getattr(args, "alignment_tc_teacher_micro_min_pos_step", 0.0007))
        if bucket == "micro_contact_refine"
        else float(getattr(args, "alignment_tc_teacher_near_min_pos_step", 0.0012))
        if bucket == "near_contact_refine"
        else float(getattr(args, "alignment_tc_teacher_broad_min_pos_step", 0.0015))
    )
    current_workspace = float(safety.workspace_violation(current_pose[:3])) if safety is not None else 0.0
    horizon = int(getattr(args, "alignment_tc_teacher_short_horizon", 3))
    best = None
    rows = []
    valid_rows = []
    improvement_constraint = "already_close"
    improve_eps = 1e-6
    phase = str(phase or "align_xy")
    if phase in {"micro_contact_refine", "align_xy", "align_xy_yaw"}:
        improvement_constraint = "micro_xy_or_yaw"
    elif phase in {"near_contact_refine", "descend_z"}:
        improvement_constraint = "near_xy_or_z"
    elif phase in {"close", "settle", "lift_verify"}:
        improvement_constraint = "close_ready"
    else:
        improvement_constraint = "broad_any"
    for idx, base_action in enumerate(candidates):
        chosen = None
        chosen_scale = 0.0
        chosen_delta_violation = 0.0
        chosen_variant = "none"
        for scale in (1.0, 0.85, 0.7, 0.5, 0.35, 0.2, 0.0):
            base_scaled = np.asarray(base_action, dtype=np.float32).copy() * float(scale)
            variants = [("full", base_scaled)]
            if abs(float(base_scaled[2])) > 1e-9:
                no_z = base_scaled.copy()
                no_z[2] = 0.0
                variants.append(("preserve_xy_yaw_no_z", no_z))
                half_z = base_scaled.copy()
                half_z[2] *= 0.5
                variants.append(("preserve_xy_yaw_half_z", half_z))
            for variant_name, action in variants:
                next_pose = apply_executed_local_delta_to_pose(current_pose, action)
                next_workspace = float(safety.workspace_violation(next_pose[:3])) if safety is not None else 0.0
                delta_violation = max(0.0, next_workspace - current_workspace)
                if delta_violation <= float(getattr(args, "alignment_tc_teacher_workspace_delta_tolerance", 1e-5)) or scale == 0.0:
                    chosen = action
                    chosen_scale = float(scale)
                    chosen_delta_violation = float(delta_violation)
                    chosen_variant = str(variant_name)
                    break
            if chosen is not None:
                break
        if chosen is None:
            chosen = np.zeros((6,), dtype=np.float32)
            chosen_variant = "fallback_zero"
        pose = current_pose.copy()
        post_series = []
        max_delta_violation = chosen_delta_violation
        for _ in range(max(horizon, 1)):
            pose = apply_executed_local_delta_to_pose(pose, chosen)
            d = pose_delta_local_between(pose, target_pose)
            d = apply_yaw_symmetry_to_delta(d, np.pi / 2.0)
            post_series.append([float(np.linalg.norm(d[:2])), float(abs(d[2])), float(abs(d[5]))])
            if safety is not None:
                max_delta_violation = max(
                    max_delta_violation,
                    max(0.0, float(safety.workspace_violation(pose[:3])) - current_workspace),
                )
        posts = np.asarray(post_series, dtype=np.float32)
        final_xy, final_z, final_yaw = [float(v) for v in posts[-1]]
        worsen_xy = max(0.0, final_xy - pre_xy)
        worsen_z = max(0.0, final_z - pre_z)
        worsen_yaw = max(0.0, final_yaw - pre_yaw)
        improve_count = int(final_xy < pre_xy) + int(final_z < pre_z) + int(final_yaw < pre_yaw)
        pos_norm = float(np.linalg.norm(chosen[:3]))
        yaw_abs = float(abs(float(chosen[5])))
        action_norm = float(pos_norm + 0.25 * yaw_abs)
        bucket_micro = bucket == "micro_contact_refine"
        bucket_near = bucket == "near_contact_refine"
        noop_like = bool(pos_norm + yaw_abs < 1e-8)
        improve_xy = bool(final_xy < pre_xy - improve_eps)
        improve_z = bool(final_z < pre_z - improve_eps)
        improve_yaw = bool(final_yaw < pre_yaw - improve_eps)
        if already_close:
            constraint_ok = True
            constraint_reason = "already_close"
        elif phase in {"close", "settle", "lift_verify"}:
            constraint_ok = bool(improve_xy or improve_z or improve_yaw or noop_like)
            constraint_reason = "close_ready"
        elif bucket_micro:
            constraint_ok = bool(improve_xy or improve_yaw)
            constraint_reason = "micro_xy_or_yaw"
        elif bucket_near:
            constraint_ok = bool(improve_xy or improve_z)
            constraint_reason = "near_xy_or_z"
        else:
            constraint_ok = bool(improve_xy or improve_z or improve_yaw)
            constraint_reason = "broad_any"
        too_small_pos = max(0.0, float(min_pos_step) - pos_norm) if not already_close else 0.0
        no_xy_improve_penalty = (
            0.0045
            if (pre_xy > float(getattr(args, "alignment_tc_teacher_already_close_xy", 0.0025)) and final_xy >= pre_xy)
            else 0.0
        )
        noop_penalty = 0.050 if (noop_like and not already_close) else 0.0
        if phase == "descend_z":
            xy_weight = 2.15 if bucket_micro else 1.95 if bucket_near else 1.70
            z_weight = 1.30 if bucket_micro else 1.20 if bucket_near else 1.05
            yaw_weight = 1.55 if bucket_micro else 1.35 if bucket_near else 1.20
            worsen_xy_weight = 4.50 if bucket_micro else 4.00 if bucket_near else 3.20
            worsen_z_weight = 2.10
            worsen_yaw_weight = 1.85 if bucket_micro else 1.65 if bucket_near else 1.45
            z_only_penalty = 0.0025 if (improve_z and not (improve_xy or improve_yaw)) else 0.0
        elif phase == "yaw_correct":
            xy_weight = 2.45 if bucket_micro else 2.05 if bucket_near else 1.75
            z_weight = 0.92 if bucket_micro else 0.95 if bucket_near else 0.90
            yaw_weight = 3.10 if bucket_micro else 2.55 if bucket_near else 2.10
            worsen_xy_weight = 4.60 if bucket_micro else 4.00 if bucket_near else 3.20
            worsen_z_weight = 1.60
            worsen_yaw_weight = 3.10 if bucket_micro else 2.65 if bucket_near else 2.20
            z_only_penalty = 0.0015 if (improve_z and not (improve_xy or improve_yaw)) else 0.0
        elif phase in {"close", "settle", "lift_verify"}:
            xy_weight = 1.85 if bucket_micro else 1.65 if bucket_near else 1.45
            z_weight = 1.25 if bucket_micro else 1.15 if bucket_near else 1.05
            yaw_weight = 1.85 if bucket_micro else 1.55 if bucket_near else 1.25
            worsen_xy_weight = 5.00 if bucket_micro else 4.20 if bucket_near else 3.50
            worsen_z_weight = 2.20
            worsen_yaw_weight = 2.60 if bucket_micro else 2.10 if bucket_near else 1.80
            z_only_penalty = 0.0010 if (improve_z and not (improve_xy or improve_yaw)) else 0.0
        else:
            xy_weight = 3.15 if bucket_micro else 2.45 if bucket_near else 2.00
            z_weight = 0.88 if bucket_micro else 0.95 if bucket_near else 0.90
            yaw_weight = 2.30 if bucket_micro else 1.85 if bucket_near else 1.45
            worsen_xy_weight = 5.50 if bucket_micro else 4.60 if bucket_near else 3.60
            worsen_z_weight = 1.55
            worsen_yaw_weight = 2.15 if bucket_micro else 1.85 if bucket_near else 1.60
            z_only_penalty = 0.0040 if (improve_z and not (improve_xy or improve_yaw)) else 0.0
        score = (
            xy_weight * final_xy
            + z_weight * final_z
            + yaw_weight * final_yaw
            + worsen_xy_weight * worsen_xy
            + worsen_z_weight * worsen_z
            + worsen_yaw_weight * worsen_yaw
            + 8.00 * max_delta_violation
            + 0.025 * action_norm
            + 4.0 * too_small_pos
            + no_xy_improve_penalty
            + noop_penalty
            + z_only_penalty
            - 0.0025 * max(improve_count - 1, 0)
            - (0.0020 if final_xy < pre_xy else 0.0)
        )
        if object_pose_7d is not None and np.all(np.isfinite(np.asarray(object_pose_7d, dtype=np.float32).reshape(-1)[:7])):
            grasp_next = compute_tc_grasp_expert_contract(
                current_pose_7d=apply_executed_local_delta_to_pose(current_pose, chosen),
                object_pose_7d=object_pose_7d,
                gripper_open=1.0,
                depth_proximity=None,
                stage_contact_state=None,
                refiner_contact_state=None,
                close_xy_threshold=float(getattr(args, "teacher_grasp_xy_threshold", getattr(args, "teacher_close_xy_threshold", 0.008))),
                close_abs_z_threshold=float(getattr(args, "teacher_grasp_abs_z_threshold", getattr(args, "teacher_close_abs_z_threshold", 0.010))),
                close_yaw_threshold=float(getattr(args, "teacher_grasp_yaw_threshold", getattr(args, "teacher_close_yaw_threshold", 0.05))),
                close_contact_depth_threshold=float(getattr(args, "teacher_close_contact_depth_threshold", 0.020)),
                low_visibility=False,
                occluded=False,
                grasp_ready_threshold=float(getattr(args, "teacher_grasp_ready_threshold", 0.55)),
            )
            readiness_score = float(grasp_next["grasp_readiness_score"])
            grasp_bonus = (
                1.6 * readiness_score
                if phase in {"close", "settle", "lift_verify"}
                else 1.2 * readiness_score
                if phase == "yaw_correct"
                else 1.0 * readiness_score
            )
            score += grasp_bonus
        row = {
            "idx": int(idx),
            "name": names[idx],
            "action": chosen.astype(np.float32),
            "score": float(score),
            "scale": float(chosen_scale),
            "variant": str(chosen_variant),
            "workspace_delta_violation": float(max_delta_violation),
            "final": (final_xy, final_z, final_yaw),
            "improve_count": int(improve_count),
            "already_close": bool(already_close),
            "pos_norm": float(pos_norm),
            "improve_xy": bool(improve_xy),
            "improve_z": bool(improve_z),
            "improve_yaw": bool(improve_yaw),
            "constraint_ok": bool(constraint_ok),
            "constraint_reason": str(constraint_reason),
            "grasp_readiness_score": float(grasp_readiness_score) if grasp_readiness_score is not None else float("nan"),
            "sequence_name": str(phase),
        }
        rows.append(row)
        if constraint_ok:
            valid_rows.append(row)
        if best is None or row["score"] < best["score"]:
            best = row
    selected_from_valid = bool(valid_rows)
    if valid_rows:
        best = min(valid_rows, key=lambda r: r["score"])
    elif best is None:
        best = rows[0]
    selected_constraint_ok = bool(best.get("constraint_ok", False))
    return {
        "action": np.asarray(best["action"], dtype=np.float32),
        "candidate_index": int(best["idx"]),
        "candidate_name": str(best["name"]),
        "score": float(best["score"]),
        "workspace_delta_violation": float(best["workspace_delta_violation"]),
        "scale": float(best["scale"]),
        "variant": str(best.get("variant", "unknown")),
        "final": tuple(float(x) for x in best["final"]),
        "improve_count": int(best["improve_count"]),
        "already_close": bool(best.get("already_close", already_close)),
        "pos_norm": float(best.get("pos_norm", 0.0)),
        "candidate_count": int(len(rows)),
        "valid_candidate_count": int(len(valid_rows)),
        "selected_from_valid": bool(selected_from_valid),
        "selected_constraint_ok": bool(selected_constraint_ok),
        "constraint_reason": str(best.get("constraint_reason", improvement_constraint)),
        "max_pos": float(max_pos),
        "max_yaw": float(max_yaw),
        "expert_sequence_name": str(phase),
        "expert_sequence_score": float(best["score"]),
        "expert_sequence_verified": False,
    }


def build_teacher_candidate_bank(
    xy_small=0.004,
    xy_large=0.008,
    z_small=0.004,
    yaw_small=0.03,
    yaw_probe_values=(),
    rescue_pitch_small=0.04,
    rescue_roll_small=0.04,
):
    base_actions = build_action_primitives(
        xy_small=xy_small,
        xy_large=xy_large,
        z_small=z_small,
        yaw_small=yaw_small,
        yaw_probe_values=yaw_probe_values,
        include_descend=True,
        include_combos=True,
        include_tilt=False,
    )
    rescue_actions = build_orientation_rescue_primitives(
        pitch_small=rescue_pitch_small,
        roll_small=rescue_roll_small,
        xy_small=xy_small,
        coupled_xy_tilt=True,
    )
    all_actions = np.stack(base_actions + rescue_actions, axis=0).astype(np.float32)
    kinds = np.asarray(["base"] * len(base_actions) + ["rescue"] * len(rescue_actions))
    group_keys = [candidate_group_key(c) for c in all_actions]
    unique_group_keys = sorted(set(group_keys))
    group_key_to_idx = {key: idx for idx, key in enumerate(unique_group_keys)}
    group_index = np.asarray([group_key_to_idx[key] for key in group_keys], dtype=np.int64)
    base_mask = np.asarray([kind == "base" for kind in kinds], dtype=np.float32)
    return all_actions, group_index, kinds, base_mask


def build_teacher_candidate_mask(
    candidate_kind: np.ndarray,
    *,
    enable_orientation_rescue: bool,
):
    mask = np.asarray(candidate_kind == "base", dtype=np.float32)
    if enable_orientation_rescue:
        mask = np.ones_like(mask, dtype=np.float32)
    return mask.astype(np.float32)


def score_candidate_grasp_aware(
    *,
    current_pose_7d,
    next_pose_7d,
    pregrasp_target_pose_7d,
    grasp_commit_target_pose_7d,
    use_commit_target: bool,
    candidate_local,
    candidate_kind: str,
    base_action_local=None,
    support_inner: bool,
    support_outer: bool,
    current_best_stall_count: int,
    orientation_rescue_active: bool,
    close_xy_threshold: float,
    close_abs_z_threshold: float,
    close_yaw_threshold: float,
    r_xy: float = 0.008,
    r_z: float = 0.010,
    r_yaw: float = 0.05,
    r_tilt: float = 0.10,
    z_collision_margin: float = 0.004,
    xy_safe_for_descend: float = 0.010,
    w_xy: float = 1.0,
    w_z_pregrasp: float = 0.8,
    w_z_commit: float = 1.2,
    w_yaw: float = 0.6,
    w_tilt: float = 0.4,
    lambda_col: float = 8.0,
    lambda_support: float = 1.5,
    lambda_close: float = 2.5,
    lambda_noop: float = 0.75,
    lambda_stall: float = 1.25,
    lambda_rescue_improve: float = 1.5,
    lambda_base_when_rescue: float = 1.0,
    rescue_improve_eps: float = 1e-4,
    cand_norm_eps: float = 1e-8,
    yaw_symmetry_period: float = -1.0,
):
    active_target = np.asarray(
        grasp_commit_target_pose_7d if use_commit_target else pregrasp_target_pose_7d,
        dtype=np.float32,
    )
    delta_cur = apply_yaw_symmetry_to_delta(
        pose_delta_local_between(np.asarray(current_pose_7d, dtype=np.float32), active_target),
        yaw_symmetry_period,
    )
    delta_next = apply_yaw_symmetry_to_delta(
        pose_delta_local_between(np.asarray(next_pose_7d, dtype=np.float32), active_target),
        yaw_symmetry_period,
    )
    current_tilt = float(np.linalg.norm(delta_cur[3:5]))
    xy = float(np.linalg.norm(delta_next[:2]))
    z = float(abs(delta_next[2]))
    tilt = float(np.linalg.norm(delta_next[3:5]))
    yaw = float(abs(delta_next[5]))
    w_z = float(w_z_commit if use_commit_target else w_z_pregrasp)
    use_yaw_cost = bool(float(close_yaw_threshold) >= 0.0)
    pose_cost = (
        float(w_xy) * (xy / max(float(r_xy), 1e-6)) ** 2
        + float(w_z) * (z / max(float(r_z), 1e-6)) ** 2
        + (float(w_yaw) * (yaw / max(float(r_yaw), 1e-6)) ** 2 if use_yaw_cost else 0.0)
        + ((float(w_tilt) * (tilt / max(float(r_tilt), 1e-6)) ** 2) if orientation_rescue_active else 0.0)
    )
    commit_target_world_z = float(np.asarray(grasp_commit_target_pose_7d, dtype=np.float32)[2])
    too_low_without_xy = bool(
        float(np.asarray(next_pose_7d, dtype=np.float32)[2]) < (commit_target_world_z + float(z_collision_margin))
        and xy > float(xy_safe_for_descend)
    )
    collision_penalty = float(lambda_col) if too_low_without_xy else 0.0
    support_bonus = float(lambda_support) if support_inner else (0.5 * float(lambda_support) if support_outer else 0.0)
    close_ready_bonus = float(lambda_close) if (
        xy <= float(close_xy_threshold)
        and z <= float(close_abs_z_threshold)
        and (not use_yaw_cost or yaw <= float(close_yaw_threshold))
    ) else 0.0
    cand_norm = float(np.linalg.norm(np.asarray(candidate_local, dtype=np.float32)))
    noop_penalty = float(lambda_noop) if cand_norm < float(cand_norm_eps) else 0.0
    stall_penalty = float(lambda_stall) if (int(current_best_stall_count) >= 2 and cand_norm < float(cand_norm_eps)) else 0.0
    rescue_improvement = max(current_tilt - tilt, 0.0)
    rescue_bonus = 0.0
    base_rescue_penalty = 0.0
    if orientation_rescue_active:
        rescue_bonus = float(lambda_rescue_improve) * (rescue_improvement / max(float(r_tilt), 1e-6))
        if str(candidate_kind) != "rescue" and rescue_improvement <= float(rescue_improve_eps):
            base_rescue_penalty = float(lambda_base_when_rescue)
    score = (
        -pose_cost
        - collision_penalty
        - noop_penalty
        - stall_penalty
        - base_rescue_penalty
        + support_bonus
        + close_ready_bonus
        + rescue_bonus
    )
    details = {
        "current_tilt": float(current_tilt),
        "xy": float(xy),
        "z": float(z),
        "yaw": float(yaw),
        "use_yaw_cost": bool(use_yaw_cost),
        "tilt": float(tilt),
        "pose_cost": float(pose_cost),
        "collision_penalty": float(collision_penalty),
        "support_bonus": float(support_bonus),
        "close_ready_bonus": float(close_ready_bonus),
        "noop_penalty": float(noop_penalty),
        "stall_penalty": float(stall_penalty),
        "rescue_bonus": float(rescue_bonus),
        "base_rescue_penalty": float(base_rescue_penalty),
        "score": float(score),
    }
    return float(score), details


def scorer_oracle(
    current_pose_7d,
    pregrasp_target_pose,
    grasp_commit_target_pose,
    candidate_actions_local,
    base_action_local,
    depth_proximity,
    candidate_mask=None,
    candidate_kind=None,
    orientation_rescue_active=False,
    close_xy_threshold=0.020,
    close_abs_z_threshold=0.020,
    close_yaw_threshold=0.12,
    commit_switch_xy_threshold=0.010,
    commit_switch_z_threshold=0.020,
    commit_switch_yaw_threshold=0.12,
    stall_count=0,
    planner_close_intent=True,
    r_xy=0.008,
    r_z=0.01,
    r_yaw=0.05,
    r_tilt=0.10,
    horizon_k=4,
    gamma=0.9,
    yaw_symmetry_period=-1.0,
):
    best_idx = -1
    best_score = -1e9
    current_pre = apply_yaw_symmetry_to_delta(
        pose_delta_local_between(current_pose_7d, pregrasp_target_pose),
        yaw_symmetry_period,
    )
    commit_yaw_ok = True
    if float(commit_switch_yaw_threshold) >= 0.0:
        commit_yaw_ok = float(abs(current_pre[5])) <= float(commit_switch_yaw_threshold)
    use_commit_target = bool(
        float(np.linalg.norm(current_pre[:2])) <= float(commit_switch_xy_threshold)
        and float(abs(current_pre[2])) <= float(commit_switch_z_threshold)
        and commit_yaw_ok
    )
    active_target = np.asarray(grasp_commit_target_pose if use_commit_target else pregrasp_target_pose, dtype=np.float32)
    current_delta = apply_yaw_symmetry_to_delta(
        pose_delta_local_between(current_pose_7d, active_target),
        yaw_symmetry_period,
    )
    current_dist, _, _, _, _ = compute_basin_metrics(current_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw, r_tilt=r_tilt)
    next_dists = []
    oracle_scores = []
    if candidate_mask is None:
        candidate_mask = np.ones((len(candidate_actions_local),), dtype=np.float32)
    if candidate_kind is None:
        candidate_kind = np.asarray(["base"] * len(candidate_actions_local))
    for j, cand in enumerate(candidate_actions_local):
        if float(candidate_mask[j]) <= 0.5:
            next_dists.append(float(current_dist))
            oracle_scores.append(-1e9)
            continue
        pose_t = np.asarray(current_pose_7d, dtype=np.float32).copy()
        cand_t = np.asarray(cand, dtype=np.float32).copy()
        total_score = 0.0
        next_dist = current_dist
        next_details = None
        for t in range(max(int(horizon_k), 1)):
            pose_next = apply_executed_local_delta_to_pose(pose_t, cand_t)
            step_pre = apply_yaw_symmetry_to_delta(
                pose_delta_local_between(pose_t, pregrasp_target_pose),
                yaw_symmetry_period,
            )
            step_yaw_ok = True
            if float(commit_switch_yaw_threshold) >= 0.0:
                step_yaw_ok = float(abs(step_pre[5])) <= float(commit_switch_yaw_threshold)
            step_use_commit = bool(
                float(np.linalg.norm(step_pre[:2])) <= float(commit_switch_xy_threshold)
                and float(abs(step_pre[2])) <= float(commit_switch_z_threshold)
                and step_yaw_ok
            )
            delta_next = apply_yaw_symmetry_to_delta(
                pose_delta_local_between(
                    pose_next,
                    np.asarray(grasp_commit_target_pose if step_use_commit else pregrasp_target_pose, dtype=np.float32),
                ),
                yaw_symmetry_period,
            )
            next_dist, _, _, _, _ = compute_basin_metrics(delta_next, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw, r_tilt=r_tilt)
            step_score, next_details = score_candidate_grasp_aware(
                current_pose_7d=pose_t,
                next_pose_7d=pose_next,
                pregrasp_target_pose_7d=pregrasp_target_pose,
                grasp_commit_target_pose_7d=grasp_commit_target_pose,
                use_commit_target=step_use_commit,
                candidate_local=cand_t,
                candidate_kind=str(candidate_kind[j]),
                base_action_local=base_action_local,
                support_inner=bool(next_dist <= 1.0),
                support_outer=bool(next_dist <= 2.0),
                current_best_stall_count=stall_count,
                orientation_rescue_active=bool(orientation_rescue_active),
                close_xy_threshold=close_xy_threshold,
                close_abs_z_threshold=close_abs_z_threshold,
                close_yaw_threshold=close_yaw_threshold if bool(yaw_imitation_enabled) else -1.0,
                r_xy=r_xy,
                r_z=r_z,
                r_yaw=r_yaw,
                r_tilt=r_tilt,
                yaw_symmetry_period=yaw_symmetry_period,
            )
            total_score += (float(gamma) ** t) * float(step_score)
            pose_t = pose_next
            if t + 1 < int(horizon_k):
                best_future_score = -1e9
                best_future_cand = np.asarray(cand_t, dtype=np.float32)
                for j2, cand2 in enumerate(candidate_actions_local):
                    if float(candidate_mask[j2]) <= 0.5:
                        continue
                    pose_future = apply_executed_local_delta_to_pose(pose_t, cand2)
                    future_pre = apply_yaw_symmetry_to_delta(
                        pose_delta_local_between(pose_t, pregrasp_target_pose),
                        yaw_symmetry_period,
                    )
                    future_yaw_ok = True
                    if float(commit_switch_yaw_threshold) >= 0.0:
                        future_yaw_ok = float(abs(future_pre[5])) <= float(commit_switch_yaw_threshold)
                    future_use_commit = bool(
                        float(np.linalg.norm(future_pre[:2])) <= float(commit_switch_xy_threshold)
                        and float(abs(future_pre[2])) <= float(commit_switch_z_threshold)
                        and future_yaw_ok
                    )
                    future_score, _ = score_candidate_grasp_aware(
                        current_pose_7d=pose_t,
                        next_pose_7d=pose_future,
                        pregrasp_target_pose_7d=pregrasp_target_pose,
                        grasp_commit_target_pose_7d=grasp_commit_target_pose,
                        use_commit_target=future_use_commit,
                        candidate_local=cand2,
                        candidate_kind=str(candidate_kind[j2]),
                        base_action_local=base_action_local,
                        support_inner=False,
                        support_outer=False,
                        current_best_stall_count=stall_count,
                        orientation_rescue_active=bool(orientation_rescue_active),
                        close_xy_threshold=close_xy_threshold,
                        close_abs_z_threshold=close_abs_z_threshold,
                        close_yaw_threshold=close_yaw_threshold if bool(yaw_imitation_enabled) else -1.0,
                        r_xy=r_xy,
                        r_z=r_z,
                        r_yaw=r_yaw,
                        r_tilt=r_tilt,
                        yaw_symmetry_period=yaw_symmetry_period,
                    )
                    if future_score > best_future_score:
                        best_future_score = float(future_score)
                        best_future_cand = np.asarray(cand2, dtype=np.float32)
                cand_t = best_future_cand
        next_dists.append(float(next_dist))
        oracle_scores.append(float(total_score))
        if total_score > best_score:
            best_score = float(total_score)
            best_idx = j
    return best_idx, float(best_score), float(current_dist), next_dists, oracle_scores


def scorer_forward_debug(controller, front_rgb, wrist_rgb, wrist_depth, proprio, base_action_local, step_idx, gripper_context):
    candidate_actions_src = getattr(controller, "_teacher_candidate_actions_local", None)
    candidate_group_index_src = getattr(controller, "_teacher_candidate_group_index", None)
    if candidate_actions_src is None:
        candidate_actions_src = controller._candidate_actions_local.detach().cpu().numpy()
    if candidate_group_index_src is None:
        candidate_group_index_src = controller._candidate_group_index.detach().cpu().numpy()

    def _empty_debug(reason: str):
        candidate_actions_np = np.asarray(candidate_actions_src, dtype=np.float32)
        candidate_group_index_np = np.asarray(candidate_group_index_src, dtype=np.int64).reshape(-1)
        num_candidates = int(candidate_actions_np.shape[0]) if candidate_actions_np.ndim >= 1 else 0
        num_groups = int(candidate_group_index_np.max()) + 1 if candidate_group_index_np.size > 0 else 0
        return {
            "pred_candidate_index": -1,
            "pred_group_index": -1,
            "topk_candidate_index": np.zeros((0,), dtype=np.int64),
            "topk_candidate_prob": np.zeros((0,), dtype=np.float32),
            "candidate_scores": np.full((num_candidates,), np.nan, dtype=np.float32),
            "candidate_probs": np.full((num_candidates,), np.nan, dtype=np.float32),
            "group_probs": np.full((num_groups,), np.nan, dtype=np.float32),
            "debug_error": reason,
        }

    if (
        front_rgb is None
        or wrist_rgb is None
        or wrist_depth is None
        or proprio is None
        or base_action_local is None
        or gripper_context is None
    ):
        return _empty_debug("missing_debug_input")

    device = next(controller.parameters()).device
    dtype = next(controller.parameters()).dtype
    try:
        front_arr = np.asarray(front_rgb, dtype=np.float32)
        wrist_arr = np.asarray(wrist_rgb, dtype=np.float32)
        if front_arr.ndim == 3 and front_arr.shape[-1] == 3:
            front_arr = np.transpose(front_arr, (2, 0, 1))
        if wrist_arr.ndim == 3 and wrist_arr.shape[-1] == 3:
            wrist_arr = np.transpose(wrist_arr, (2, 0, 1))
        fr = torch.from_numpy(front_arr).unsqueeze(0).to(device=device, dtype=dtype)
        wr = torch.from_numpy(wrist_arr).unsqueeze(0).to(device=device, dtype=dtype)
        if float(fr.max().item()) > 1.5:
            fr = fr / 255.0
        if float(wr.max().item()) > 1.5:
            wr = wr / 255.0
        wd = torch.from_numpy(np.asarray(wrist_depth, dtype=np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
        pr = torch.from_numpy(np.asarray(proprio, dtype=np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
        ba = torch.from_numpy(np.asarray(base_action_local, dtype=np.float32)).reshape(1, 6).to(device=device, dtype=dtype)
        gc = torch.from_numpy(np.asarray(gripper_context, dtype=np.float32)).reshape(1, 3).to(device=device, dtype=dtype)
        si = torch.tensor([int(step_idx)], device=device, dtype=torch.long)
        phase_id = torch.tensor([1], device=device, dtype=torch.long)
        phase_age = torch.tensor([0.0], device=device, dtype=dtype)
        since_replan = torch.tensor([0.0], device=device, dtype=dtype)
        if bool(getattr(controller, "use_target_context", True)):
            current_delta = np.asarray(
                getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                dtype=np.float32,
            )
            clip_abs = getattr(controller, "_target_context_clip_abs", None)
            if clip_abs is not None:
                current_delta = np.clip(
                    current_delta,
                    -np.asarray(clip_abs, dtype=np.float32),
                    np.asarray(clip_abs, dtype=np.float32),
                ).astype(np.float32)
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
        else:
            cur_delta_t = None
            dxs = None
            dys = None
            dyaws = None
            basin_bin_t = None
        candidate_actions = torch.from_numpy(np.asarray(candidate_actions_src, dtype=np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
        candidate_group_index = torch.from_numpy(np.asarray(candidate_group_index_src, dtype=np.int64)).unsqueeze(0).to(device=device, dtype=torch.long)
        candidate_mask = getattr(controller, "_runtime_candidate_mask", None)
        if candidate_mask is not None:
            candidate_mask = torch.from_numpy(np.asarray(candidate_mask, dtype=np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            outputs = controller(
                fr,
                wr,
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
                candidate_mask=candidate_mask,
                return_aux=True,
            )
            scores = outputs["candidate_scores"].float()
            group_logits = outputs["group_logits"].float()
            pred_group = group_logits.argmax(dim=-1)
            valid_mask = candidate_mask > 0.5 if candidate_mask is not None else torch.ones_like(scores, dtype=torch.bool)
            group_valid = []
            for group_id in range(group_logits.shape[1]):
                group_valid.append(torch.any(candidate_group_index.eq(group_id) & valid_mask, dim=1))
            group_valid = torch.stack(group_valid, dim=1)
            pred_group = group_logits.masked_fill(~group_valid, -1e9).argmax(dim=-1)
            group_mask = candidate_group_index.eq(pred_group.unsqueeze(1)) & valid_mask
            pred = scores.masked_fill(~group_mask, -1e9).argmax(dim=-1)
            cand_probs = torch.softmax(scores.masked_fill(~valid_mask, -1e9), dim=-1)
            topk = min(5, cand_probs.shape[-1])
            topk_prob, topk_idx = torch.topk(cand_probs, k=topk, dim=-1)
        return {
            "pred_candidate_index": int(pred.item()),
            "pred_group_index": int(pred_group.item()),
            "topk_candidate_index": topk_idx.squeeze(0).cpu().numpy(),
            "topk_candidate_prob": topk_prob.squeeze(0).cpu().numpy(),
            "candidate_scores": scores.squeeze(0).cpu().numpy(),
            "candidate_probs": cand_probs.squeeze(0).cpu().numpy(),
            "group_probs": torch.softmax(group_logits, dim=-1).squeeze(0).cpu().numpy(),
        }
    except Exception as exc:
        return _empty_debug(f"scorer_forward_debug_exception:{type(exc).__name__}")


def write_rows_npz(rows, output_npz: Path):
    if not rows:
        return None
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    required_keys = {
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "base_action",
        "gripper_context",
        "planner_close_intent",
        "depth_proximity",
        "wrist_depth_median",
        "step_idx",
        "phase_id",
        "phase_age",
        "steps_since_last_replan",
        "candidate_actions_local",
        "candidate_group_index",
        "candidate_mask",
        "candidate_next_basin_distance",
        "candidate_improvement",
        "candidate_oracle_score",
        "candidate_tier",
        "current_delta_basin_target",
        "current_basin_distance",
        "current_dx_sign",
        "current_dy_sign",
        "current_dyaw_sign",
        "basin_distance_bin",
        "best_candidate_index",
        "best_group_index",
        "ready_to_close_target",
    }
    stacked = {}
    skipped_keys = {}
    for key in keys:
        values = [np.asarray(row[key]) for row in rows]
        shapes = sorted({tuple(v.shape) for v in values})
        if len(shapes) == 1:
            stacked[key] = np.stack(values, axis=0)
            continue
        # Strings / scalar objects can still be stored as a flat array safely.
        if all(v.shape == () for v in values):
            stacked[key] = np.asarray(values)
            continue
        skipped_keys[key] = shapes
        if key in required_keys:
            raise ValueError(
                f"support row key '{key}' has inconsistent shapes: {shapes}"
            )
        print(
            f"[write_rows_npz] skipping optional key '{key}' due to inconsistent shapes: {shapes}"
        )
    if skipped_keys:
        stacked["__skipped_inconsistent_keys__"] = np.asarray(
            [f"{k}:{shapes}" for k, shapes in sorted(skipped_keys.items())]
        )
    np.savez_compressed(output_npz, **stacked)
    return output_npz


def _alignment_diffusion_action4d_from_local(action_local_6d: np.ndarray) -> np.ndarray:
    action = np.asarray(action_local_6d, dtype=np.float32).reshape(-1)
    out = np.zeros((4,), dtype=np.float32)
    if action.size >= 3:
        out[:3] = action[:3]
    if action.size >= 6:
        out[3] = action[5]
    elif action.size >= 4:
        out[3] = action[3]
    return out


def _alignment_diffusion_bucket(depth_proximity: float, force_norm: float, phase_id: int, planner_close_intent: bool) -> str:
    if (
        (np.isfinite(depth_proximity) and depth_proximity <= 0.045)
        or force_norm >= 0.45
        or phase_id in (int(StagePhase.ALIGN), int(StagePhase.INTERACT))
    ):
        return "micro_contact_refine"
    if (
        (np.isfinite(depth_proximity) and depth_proximity <= 0.085)
        or force_norm >= 0.18
        or planner_close_intent
        or phase_id == int(StagePhase.RECOVER)
    ):
        return "near_contact_refine"
    return "far"


def _alignment_diffusion_near_gate(row: dict[str, np.ndarray]) -> bool:
    return str(row.get("stage_bucket", "far")) in {"near_contact_refine", "micro_contact_refine"}


def _target_delta_bucket_from_privileged_row(
    row: dict[str, np.ndarray],
    *,
    near_xy_threshold: float,
    near_abs_z_threshold: float,
    near_yaw_threshold: float,
    micro_xy_threshold: float,
    micro_abs_z_threshold: float,
    micro_yaw_threshold: float,
) -> tuple[str, np.ndarray | None, tuple[float, float, float]]:
    delta = None
    for key in (
        "privileged_current_to_target_delta_local",
        "privileged_current_delta_basin_target",
        "motion_target_delta_local",
    ):
        if key in row:
            arr = np.asarray(row[key], dtype=np.float32).reshape(-1)
            if arr.size >= 6 and np.all(np.isfinite(arr[:6])):
                delta = arr[:6].copy()
                break
    if delta is None:
        return "missing_privileged_delta", None, (np.inf, np.inf, np.inf)
    xy = float(np.linalg.norm(delta[:2]))
    abs_z = float(abs(delta[2]))
    yaw = float(abs(delta[5]))
    if xy <= float(micro_xy_threshold) and abs_z <= float(micro_abs_z_threshold) and yaw <= float(micro_yaw_threshold):
        return "micro_contact_refine", delta, (xy, abs_z, yaw)
    if xy <= float(near_xy_threshold) and abs_z <= float(near_abs_z_threshold) and yaw <= float(near_yaw_threshold):
        return "near_contact_refine", delta, (xy, abs_z, yaw)
    return "outside_target_delta_window", delta, (xy, abs_z, yaw)


def _alignment_diffusion_action_noise(rng, shape: tuple[int, ...], std: float) -> np.ndarray:
    if std <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    return rng.normal(0.0, float(std), size=shape).astype(np.float32)


def _alignment_diffusion_augment_row(
    row: dict[str, np.ndarray],
    *,
    rng,
    aug_id: int,
    rgb_noise_std: float,
    depth_noise_std: float,
    force_noise_std: float,
    proprio_noise_std: float,
    action_xy_noise_std: float,
    action_z_noise_std: float,
    action_yaw_noise_std: float,
    sample_weight_scale: float,
) -> dict[str, np.ndarray]:
    out = {k: (np.asarray(v).copy() if hasattr(v, "shape") else v) for k, v in row.items()}
    if "front_rgb" in out:
        rgb = np.asarray(out["front_rgb"], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.5:
            rgb = rgb / 255.0
        rgb = np.clip(rgb + _alignment_diffusion_action_noise(rng, rgb.shape, rgb_noise_std), 0.0, 1.0)
        out["front_rgb"] = (rgb * 255.0).astype(np.uint8) if row["front_rgb"].dtype == np.uint8 else rgb.astype(np.float32)
    if "wrist_rgb" in out:
        rgb = np.asarray(out["wrist_rgb"], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.5:
            rgb = rgb / 255.0
        rgb = np.clip(rgb + _alignment_diffusion_action_noise(rng, rgb.shape, rgb_noise_std), 0.0, 1.0)
        out["wrist_rgb"] = (rgb * 255.0).astype(np.uint8) if row["wrist_rgb"].dtype == np.uint8 else rgb.astype(np.float32)
    if "wrist_depth" in out:
        depth = np.asarray(out["wrist_depth"], dtype=np.float32)
        out["wrist_depth"] = np.clip(depth + _alignment_diffusion_action_noise(rng, depth.shape, depth_noise_std), 0.0, 1.0).astype(np.float32)
    if "force_history" in out:
        force = np.asarray(out["force_history"], dtype=np.float32)
        out["force_history"] = (force + _alignment_diffusion_action_noise(rng, force.shape, force_noise_std)).astype(np.float32)
    if "proprio" in out:
        proprio = np.asarray(out["proprio"], dtype=np.float32)
        out["proprio"] = (proprio + _alignment_diffusion_action_noise(rng, proprio.shape, proprio_noise_std)).astype(np.float32)
    if "planner_action_local" in out:
        action = np.asarray(out["planner_action_local"], dtype=np.float32).copy()
        action[:2] += _alignment_diffusion_action_noise(rng, (2,), action_xy_noise_std)
        action[2] += float(_alignment_diffusion_action_noise(rng, (1,), action_z_noise_std)[0])
        if action.shape[0] > 5:
            action[5] += float(_alignment_diffusion_action_noise(rng, (1,), action_yaw_noise_std)[0])
        out["planner_action_local"] = action.astype(np.float32)
    if "executed_action_local" in out:
        action = np.asarray(out["executed_action_local"], dtype=np.float32).copy()
        action[:2] += _alignment_diffusion_action_noise(rng, (2,), action_xy_noise_std)
        action[2] += float(_alignment_diffusion_action_noise(rng, (1,), action_z_noise_std)[0])
        if action.shape[0] > 5:
            action[5] += float(_alignment_diffusion_action_noise(rng, (1,), action_yaw_noise_std)[0])
        out["executed_action_local"] = action.astype(np.float32)
    out["data_aug_id"] = np.asarray(int(aug_id), dtype=np.int64)
    out["sample_weight"] = np.asarray(float(out.get("sample_weight", 1.0)) * float(sample_weight_scale), dtype=np.float32)
    return out


def _alignment_diffusion_finalize_episode_rows(
    rows: list[dict[str, np.ndarray]],
    *,
    horizon: int,
    micro_depth_threshold: float,
    near_depth_threshold: float,
    contact_force_threshold: float,
    high_force_threshold: float,
    force_spike_threshold: float,
    action_floor_xy: float,
    action_floor_z: float,
    action_floor_yaw: float,
    xy_gain_margin: float,
    z_gain_margin: float,
    yaw_gain_margin: float,
    augment_copies: int,
    rgb_noise_std: float,
    depth_noise_std: float,
    force_noise_std: float,
    proprio_noise_std: float,
    action_xy_noise_std: float,
    action_z_noise_std: float,
    action_yaw_noise_std: float,
    sample_weight_near: float,
    sample_weight_micro: float,
    sample_weight_stop: float,
    sample_weight_risk: float,
    use_privileged_target_delta_gate: bool,
    near_target_xy_threshold: float,
    near_target_abs_z_threshold: float,
    near_target_yaw_threshold: float,
    micro_target_xy_threshold: float,
    micro_target_abs_z_threshold: float,
    micro_target_yaw_threshold: float,
    rng,
) -> list[dict[str, np.ndarray]]:
    def _scalar(x, default=0.0) -> float:
        try:
            return float(np.asarray(x).reshape(()))
        except Exception:
            return float(default)

    out: list[dict[str, np.ndarray]] = []
    n = len(rows)
    for i, row in enumerate(rows):
        geom_bucket = None
        geom_delta = None
        geom_metrics = (np.inf, np.inf, np.inf)
        if use_privileged_target_delta_gate:
            geom_bucket, geom_delta, geom_metrics = _target_delta_bucket_from_privileged_row(
                row,
                near_xy_threshold=near_target_xy_threshold,
                near_abs_z_threshold=near_target_abs_z_threshold,
                near_yaw_threshold=near_target_yaw_threshold,
                micro_xy_threshold=micro_target_xy_threshold,
                micro_abs_z_threshold=micro_target_abs_z_threshold,
                micro_yaw_threshold=micro_target_yaw_threshold,
            )
            if geom_bucket not in {"near_contact_refine", "micro_contact_refine"}:
                continue
        elif not _alignment_diffusion_near_gate(row):
            continue
        future = rows[i + 1 : min(n, i + 1 + int(horizon))]
        if len(future) == 0:
            continue
        cur = np.asarray(row["executed_action_local"], dtype=np.float32).reshape(-1)
        cur4 = _alignment_diffusion_action4d_from_local(cur)
        future4 = np.stack(
            [_alignment_diffusion_action4d_from_local(np.asarray(fr["executed_action_local"], dtype=np.float32)) for fr in future],
            axis=0,
        )
        if future4.shape[0] < int(horizon):
            pad = np.repeat(future4[-1:,...], int(horizon) - future4.shape[0], axis=0)
            future4 = np.concatenate([future4, pad], axis=0)
        teacher_action4 = np.asarray(
            row.get("teacher_residual_action_4d", np.zeros((4,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)[:4]
        if teacher_action4.shape[0] < 4:
            teacher_action4 = np.pad(teacher_action4, (0, 4 - teacher_action4.shape[0]), mode="constant").astype(np.float32)
        teacher_future4_rows = [
            np.asarray(fr.get("teacher_residual_action_4d", teacher_action4), dtype=np.float32).reshape(-1)[:4]
            for fr in future
        ]
        teacher_future4 = np.stack(
            [
                np.pad(v, (0, max(0, 4 - v.shape[0])), mode="constant")[:4].astype(np.float32)
                for v in teacher_future4_rows
            ],
            axis=0,
        )
        if teacher_future4.shape[0] < int(horizon):
            pad = np.repeat(teacher_future4[-1:,...], int(horizon) - teacher_future4.shape[0], axis=0)
            teacher_future4 = np.concatenate([teacher_future4, pad], axis=0)
        teacher_traj4 = np.concatenate([teacher_action4.reshape(1, 4), teacher_future4], axis=0)[: int(horizon)]
        if teacher_traj4.shape[0] < int(horizon):
            pad = np.repeat(teacher_traj4[-1:,...], int(horizon) - teacher_traj4.shape[0], axis=0)
            teacher_traj4 = np.concatenate([teacher_traj4, pad], axis=0)
        teacher_traj6 = np.zeros((int(horizon), 6), dtype=np.float32)
        teacher_traj6[:, :3] = teacher_traj4[:, :3]
        teacher_traj6[:, 5] = teacher_traj4[:, 3]
        future_mean = np.mean(np.abs(future4), axis=0)
        future_max = np.max(np.abs(future4), axis=0)
        future_invalid = any(bool(fr.get("invalid_action", False)) for fr in future)
        future_force_spike = any(
            bool(_scalar(fr.get("force_spike", 0.0)) > 0.5)
            or _scalar(fr.get("force_norm", 0.0)) > float(high_force_threshold)
            or abs(_scalar(fr.get("force_norm", 0.0)) - _scalar(fr.get("prev_force_norm", 0.0))) > float(force_spike_threshold)
            for fr in future
        )
        future_stall = any(bool(_scalar(fr.get("motion_stall", False)) > 0.5) for fr in future)
        future_workspace_violation = any(_scalar(fr.get("workspace_violation", 0.0)) > 0.0 for fr in future)
        current_force = _scalar(row.get("force_norm", 0.0))
        current_depth = _scalar(row.get("depth_proximity", np.nan), np.nan)
        current_bucket = str(geom_bucket or row.get("stage_bucket", "far"))
        current_xy = float(np.linalg.norm(cur4[:2]))
        current_z = float(abs(cur4[2]))
        current_yaw = float(abs(cur4[3]))
        future_xy = float(future_mean[:2].mean())
        future_z = float(future_mean[2])
        future_yaw = float(future_mean[3])
        improvement_xy = float((current_xy - future_xy) > float(xy_gain_margin) and future_xy > float(action_floor_xy))
        improvement_z = float((current_z - future_z) > float(z_gain_margin) and future_z > float(action_floor_z))
        improvement_yaw = float((current_yaw - future_yaw) > float(yaw_gain_margin) and future_yaw > float(action_floor_yaw))
        progress_label = np.asarray([improvement_xy, improvement_z, improvement_yaw], dtype=np.float32)
        risk_label = np.asarray(
            float(
                future_invalid
                or future_force_spike
                or future_stall
                or future_workspace_violation
                or _scalar(row.get("invalid_action", False)) > 0.5
                or _scalar(row.get("force_spike", False)) > 0.5
                or _scalar(row.get("workspace_violation", 0.0)) > 0.0
            ),
            dtype=np.float32,
        ).reshape(1)
        very_close_abstain = bool(
            (current_bucket == "micro_contact_refine")
            and np.isfinite(current_depth)
            and current_depth <= float(micro_depth_threshold)
            and current_force <= float(contact_force_threshold)
            and current_xy <= float(action_floor_xy)
            and current_z <= float(action_floor_z)
            and current_yaw <= float(action_floor_yaw)
        )
        low_gain = bool((improvement_xy + improvement_z + improvement_yaw) <= 0.0)
        risk_high = bool(risk_label.reshape(-1)[0] > 0.5)
        stop_reason = "continue"
        if risk_high:
            stop_reason = "risk_high"
        elif very_close_abstain:
            stop_reason = "very_close_abstain"
        elif low_gain:
            stop_reason = "low_gain"
        stop_label = np.asarray(float(stop_reason != "continue"), dtype=np.float32).reshape(1)
        no_op_reason = stop_reason
        if risk_high:
            risk_reason = "future_invalid_or_force_spike"
            risk_level = 3
        elif very_close_abstain:
            risk_reason = "micro_contact_stable"
            risk_level = 2
        elif low_gain:
            risk_reason = "low_gain"
            risk_level = 1
        else:
            risk_reason = "safe"
            risk_level = 0
        close_action = str(row.get("teacher_close_action", "continue_align"))
        close_success = bool(_scalar(row.get("close_success_label", 0.0)) > 0.5)
        teacher_verified = bool(_scalar(row.get("teacher_grasp_verified", 0.0)) > 0.5)
        if close_action == "close_now":
            stop_reason = "close_now"
            no_op_reason = "close_now"
            stop_label = np.asarray(1.0, dtype=np.float32).reshape(1)
            risk_label = np.asarray(0.0, dtype=np.float32).reshape(1)
            risk_reason = "safe_close"
            risk_level = 0
        elif close_action == "verify_lift":
            stop_reason = "verify_lift"
            no_op_reason = "verify_lift"
            stop_label = np.asarray(1.0, dtype=np.float32).reshape(1)
            risk_label = np.asarray(0.0, dtype=np.float32).reshape(1)
            risk_reason = "verify_lift"
            risk_level = 0
        elif close_action == "verified_success" or teacher_verified or close_success:
            stop_reason = "verified_success"
            no_op_reason = "verified_success"
            stop_label = np.asarray(1.0, dtype=np.float32).reshape(1)
            risk_label = np.asarray(0.0, dtype=np.float32).reshape(1)
            risk_reason = "verified_success"
            risk_level = 0
        elif close_action == "retry":
            stop_reason = "retry"
            no_op_reason = "retry"
            stop_label = np.asarray(1.0, dtype=np.float32).reshape(1)
            risk_label = np.asarray(1.0, dtype=np.float32).reshape(1)
            risk_reason = str(row.get("close_failure_reason", "verify_failed"))
            risk_level = 2
        residual4 = future4 - cur4.reshape(1, -1)
        residual6 = np.zeros((int(horizon), 6), dtype=np.float32)
        residual6[:, :3] = residual4[:, :3]
        residual6[:, 5] = residual4[:, 3]
        base_weight = sample_weight_micro if current_bucket == "micro_contact_refine" else sample_weight_near
        if stop_reason != "continue":
            base_weight *= sample_weight_stop if stop_reason != "risk_high" else sample_weight_risk
        base_row = dict(row)
        base_row.update(
            {
                "stage_bucket": np.asarray(current_bucket),
                "target_delta_gate_bucket": np.asarray(str(geom_bucket or "depth_phase_gate")),
                "target_delta_gate_xy": np.asarray(float(geom_metrics[0]), dtype=np.float32),
                "target_delta_gate_abs_z": np.asarray(float(geom_metrics[1]), dtype=np.float32),
                "target_delta_gate_yaw": np.asarray(float(geom_metrics[2]), dtype=np.float32),
                "target_delta_gate_used": np.asarray(float(use_privileged_target_delta_gate), dtype=np.float32),
                "teacher_target_delta_local_6d": np.asarray(
                    geom_delta if geom_delta is not None else row.get("privileged_current_to_target_delta_local", np.full((6,), np.nan, dtype=np.float32)),
                    dtype=np.float32,
                ),
                "residual_trajectory_4d": residual4.astype(np.float32),
                "residual_trajectory_6d": residual6.astype(np.float32),
                "teacher_residual_trajectory_4d": teacher_traj4.astype(np.float32),
                "teacher_residual_trajectory_6d": teacher_traj6.astype(np.float32),
                "progress_label": progress_label,
                "risk_label": risk_label,
                "stop_label": stop_label,
                "stop_reason": np.asarray(stop_reason),
                "no_op_reason": np.asarray(no_op_reason),
                "risk_reason": np.asarray(risk_reason),
                "risk_level": np.asarray(int(risk_level), dtype=np.int64),
                "future_mean_action_4d": future_mean.astype(np.float32),
                "future_max_action_4d": future_max.astype(np.float32),
                "future_gain_xy": np.asarray(float(current_xy - future_xy), dtype=np.float32),
                "future_gain_z": np.asarray(float(current_z - future_z), dtype=np.float32),
                "future_gain_yaw": np.asarray(float(current_yaw - future_yaw), dtype=np.float32),
                "future_invalid": np.asarray(float(future_invalid), dtype=np.float32),
                "future_force_spike": np.asarray(float(future_force_spike), dtype=np.float32),
                "future_stall": np.asarray(float(future_stall), dtype=np.float32),
                "future_workspace_violation": np.asarray(float(future_workspace_violation), dtype=np.float32),
                "sample_weight": np.asarray(float(base_weight), dtype=np.float32),
                "data_aug_id": np.asarray(0, dtype=np.int64),
            }
        )
        out.append(base_row)
        for aug_id in range(1, int(augment_copies) + 1):
            aug = _alignment_diffusion_augment_row(
                base_row,
                rng=rng,
                aug_id=aug_id,
                rgb_noise_std=rgb_noise_std,
                depth_noise_std=depth_noise_std,
                force_noise_std=force_noise_std,
                proprio_noise_std=proprio_noise_std,
                action_xy_noise_std=action_xy_noise_std,
                action_z_noise_std=action_z_noise_std,
                action_yaw_noise_std=action_yaw_noise_std,
                sample_weight_scale=0.75,
            )
            out.append(aug)
    return out


def _force_history_from_buffer(force_buffer, history_len: int = FORCE_HISTORY_LEN, force_dim: int = FORCE_DIM):
    history = np.zeros((history_len, force_dim), dtype=np.float32)
    buf = list(force_buffer)
    for i in range(history_len):
        idx = len(buf) - history_len + i
        if idx >= 0:
            value = np.asarray(buf[idx], dtype=np.float32).reshape(-1)
            history[i, : min(force_dim, value.shape[0])] = value[:force_dim]
    return history


def _depth_force_contact_state(depth_proximity, raw_force, near_threshold: float, contact_force: float, jam_force: float):
    force = np.asarray(raw_force if raw_force is not None else np.zeros(FORCE_DIM, dtype=np.float32), dtype=np.float32).reshape(-1)
    force_xyz = force[:3] if force.size >= 3 else force
    torque = force[3:6] if force.size >= 6 else np.zeros(3, dtype=np.float32)
    force_norm = float(np.linalg.norm(force_xyz))
    torque_norm = float(np.linalg.norm(torque))
    if force_norm > float(jam_force) or torque_norm > float(jam_force):
        return 3
    if force_norm > float(contact_force):
        return 2
    if depth_proximity is not None and np.isfinite(depth_proximity) and float(depth_proximity) < float(near_threshold):
        return 1
    return 0


def append_depth_force_clean_support_row(
    rows,
    *,
    ep_idx: int,
    step_idx: int,
    chunk_step: int,
    obs,
    proprio,
    depth_tensor_96,
    force_hist,
    force_buffer,
    raw_force,
    base_delta_action,
    delta_action,
    planner_base_action_local_raw,
    planner_base_action_7d_raw,
    depth_proximity,
    wrist_valid_depth_ratio: float,
    wrist_depth_near_fraction: float,
    wrist_is_occluded: bool,
    wrist_is_low_visibility: bool,
    stage_tracker,
    abs_gripper_cmd: float,
    privileged_label: dict | None,
    args,
):
    if rows is None:
        return
    wrist_depth = (
        depth_tensor_96.detach().cpu().numpy().astype(np.float32)
        if depth_tensor_96 is not None
        else np.zeros((1, 96, 96), dtype=np.float32)
    )
    if wrist_depth.ndim == 3 and wrist_depth.shape[0] == 1:
        wrist_depth_for_stats = wrist_depth[0]
    else:
        wrist_depth_for_stats = wrist_depth
    valid_depth = wrist_depth_for_stats[np.isfinite(wrist_depth_for_stats)]
    wrist_depth_median = float(np.median(valid_depth)) if valid_depth.size > 0 else np.nan
    force_history_raw = _force_history_from_buffer(force_buffer)
    force_history_normalized = (
        force_hist.detach().cpu().numpy().astype(np.float32)
        if force_hist is not None
        else force_history_raw.copy()
    )
    force = np.asarray(raw_force if raw_force is not None else np.zeros(FORCE_DIM, dtype=np.float32), dtype=np.float32).reshape(-1)
    if force.shape[0] < FORCE_DIM:
        padded = np.zeros((FORCE_DIM,), dtype=np.float32)
        padded[: force.shape[0]] = force
        force = padded
    force = force[:FORCE_DIM]
    executed_action_local = world_delta_to_local(
        np.asarray(delta_action[:6], dtype=np.float32),
        np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
    ).astype(np.float32)
    contact_state = _depth_force_contact_state(
        depth_proximity,
        force,
        near_threshold=float(args.depth_force_clean_near_depth_threshold),
        contact_force=float(args.depth_force_clean_contact_force_threshold),
        jam_force=float(args.depth_force_clean_jam_force_threshold),
    )
    row = {
            "mode_index": np.asarray(0, dtype=np.int64),
            "episode_index": np.asarray(int(ep_idx), dtype=np.int64),
            "step_index": np.asarray(int(step_idx), dtype=np.int64),
            "step_idx": np.asarray(int(chunk_step), dtype=np.int64),
            "rollout_step": np.asarray(int(step_idx), dtype=np.int64),
            "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
            "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
            "wrist_depth": wrist_depth.astype(np.float32),
            "proprio": np.asarray(proprio, dtype=np.float32),
            "base_action": np.asarray(base_delta_action[:6], dtype=np.float32),
            "planner_base_action_local_raw": np.asarray(planner_base_action_local_raw, dtype=np.float32),
            "planner_base_action_7d_raw": np.asarray(planner_base_action_7d_raw, dtype=np.float32),
            "executed_action_local": executed_action_local,
            "executed_action_7d": np.asarray(delta_action, dtype=np.float32),
            "force_history": force_history_raw.astype(np.float32),
            "force_history_raw": force_history_raw.astype(np.float32),
            "force_history_normalized": force_history_normalized.astype(np.float32),
            "ft_hist": force_history_raw.astype(np.float32),
            "gripper_touch_forces": force.astype(np.float32),
            "force_norm": np.asarray(float(np.linalg.norm(force[:3])), dtype=np.float32),
            "torque_norm": np.asarray(float(np.linalg.norm(force[3:6])), dtype=np.float32),
            "gripper_context": np.asarray(
                [
                    float(obs.gripper_open),
                    float(np.clip(base_delta_action[6], -1.0, 1.0)),
                    float(depth_proximity) if depth_proximity is not None else np.nan,
                ],
                dtype=np.float32,
            ),
            "gripper_state": np.asarray(float(obs.gripper_open), dtype=np.float32),
            "rollout_gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
            "planner_close_intent": np.asarray(float(np.clip(base_delta_action[6], -1.0, 1.0) <= 0.5), dtype=np.float32),
            "abs_gripper_cmd": np.asarray(float(abs_gripper_cmd), dtype=np.float32),
            "depth_proximity": np.asarray(float(depth_proximity) if depth_proximity is not None else np.nan, dtype=np.float32),
            "wrist_depth_median": np.asarray(wrist_depth_median, dtype=np.float32),
            "wrist_valid_depth_ratio": np.asarray(float(wrist_valid_depth_ratio), dtype=np.float32),
            "wrist_depth_near_fraction": np.asarray(float(wrist_depth_near_fraction), dtype=np.float32),
            "is_occluded": np.asarray(float(wrist_is_occluded), dtype=np.float32),
            "is_low_visibility": np.asarray(float(wrist_is_low_visibility), dtype=np.float32),
            "phase_id": np.asarray(int(stage_tracker.phase), dtype=np.int64),
            "substage_id": np.asarray(int(stage_tracker.substage), dtype=np.int64),
            "stage_token": np.asarray(int(stage_tracker.substage), dtype=np.int64),
            "has_object_in_hand": np.asarray(float(stage_tracker.has_object_in_hand), dtype=np.float32),
            "contact_state": np.asarray(int(contact_state), dtype=np.int64),
            "stage_tracker_contact_state": np.asarray(int(stage_tracker.contact_state), dtype=np.int64),
            "stage_target_mode": np.asarray(int(stage_tracker.stage_target_mode), dtype=np.int64),
            "phase_age": np.asarray(float(getattr(stage_tracker, "phase_age", 0.0)), dtype=np.float32),
            "steps_since_last_replan": np.asarray(float(chunk_step), dtype=np.float32),
            "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
            "zone_state": np.asarray("planner_only_clean"),
            "depth_force_clean_support": np.asarray(1.0, dtype=np.float32),
            "ready_to_close_target": np.asarray(0.0, dtype=np.float32),
            "post_close_stability_proxy": np.asarray(0.0, dtype=np.float32),
            "grasp_lift_proxy": np.asarray(0.0, dtype=np.float32),
            "grasp_verified_target": np.asarray(0.0, dtype=np.float32),
            "retry_required_target": np.asarray(0.0, dtype=np.float32),
            "reopen_after_trigger": np.asarray(0.0, dtype=np.float32),
            "invalid_after_trigger": np.asarray(0.0, dtype=np.float32),
            "planner_close_too_early": np.asarray(0.0, dtype=np.float32),
    }
    if privileged_label:
        row.update(privileged_label)
    rows.append(row)


def build_depth_force_label_only_privileged_fields(
    *,
    obs,
    stage_tracker,
    task_name: str,
    live_target_handle,
    target_mode: str = "active",
) -> dict:
    nan_pose = np.full((7,), np.nan, dtype=np.float32)
    nan_delta = np.full((6,), np.nan, dtype=np.float32)
    current_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    object_pose = safe_live_target_pose_7d(live_target_handle)
    if object_pose is None:
        return {
            "privileged_current_pose_7d": current_pose,
            "privileged_motion_target_pose_7d": nan_pose,
            "privileged_basin_center_pose_7d": nan_pose,
            "privileged_pregrasp_target_pose_7d": nan_pose,
            "privileged_grasp_commit_target_pose_7d": nan_pose,
            "privileged_object_anchor_pose_7d": nan_pose,
            "privileged_current_delta_basin_target": nan_delta,
            "privileged_target_provider_source": np.asarray("missing_live_target"),
            "privileged_target_provider_uses_privileged": np.asarray(0.0, dtype=np.float32),
        }
    try:
        grasp_spec = load_phase1_grasp_spec(task_name)
        pregrasp_target, grasp_commit_target = build_phase1_teacher_targets(object_pose, grasp_spec)
        insert_target = safe_task_low_dim_pose_7d(obs)
        if str(target_mode) in ("insert_commit", "insert", "task_success_centre"):
            active_target = insert_target if insert_target is not None else grasp_commit_target
            use_commit_target = True
        elif str(target_mode) == "commit":
            active_target = grasp_commit_target
            use_commit_target = True
        elif str(target_mode) == "pregrasp":
            active_target = pregrasp_target
            use_commit_target = False
        else:
            active_target, use_commit_target = select_phase1_teacher_target(
                current_gripper_pose=current_pose,
                pregrasp_target_pose_7d=pregrasp_target,
                grasp_commit_target_pose_7d=grasp_commit_target,
                grasp_spec=grasp_spec,
            )
        delta_raw = pose_delta_local_between(current_pose, active_target)
        delta_folded = apply_yaw_symmetry_to_delta(delta_raw, np.pi / 2.0)
        # Square rings are symmetric under 90-degree yaw rotations; this folds
        # label yaw without exposing privileged pose to the runtime policy.
        if str(target_mode) in ("insert_commit", "insert", "task_success_centre"):
            source = "label_only_insert_commit"
        else:
            source = "label_only_phase1_grasp_commit" if use_commit_target else "label_only_phase1_pregrasp"
        if str(target_mode) in ("commit", "pregrasp", "insert_commit", "insert", "task_success_centre"):
            source = f"{source}__forced_{target_mode}"
        return {
            "privileged_current_pose_7d": current_pose,
            "privileged_motion_target_pose_7d": np.asarray(active_target, dtype=np.float32),
            "privileged_basin_center_pose_7d": np.asarray(active_target, dtype=np.float32),
            "privileged_pregrasp_target_pose_7d": np.asarray(pregrasp_target, dtype=np.float32),
            "privileged_grasp_commit_target_pose_7d": np.asarray(grasp_commit_target, dtype=np.float32),
            "privileged_insert_target_pose_7d": np.asarray(insert_target if insert_target is not None else nan_pose, dtype=np.float32),
            "privileged_object_anchor_pose_7d": np.asarray(object_pose, dtype=np.float32),
            "privileged_current_delta_basin_target": np.asarray(delta_folded, dtype=np.float32),
            "privileged_current_delta_basin_target_raw": np.asarray(delta_raw, dtype=np.float32),
            "privileged_current_delta_basin_target_folded": np.asarray(delta_folded, dtype=np.float32),
            "privileged_target_provider_source": np.asarray(source),
            "privileged_target_provider_uses_privileged": np.asarray(1.0, dtype=np.float32),
        }
    except Exception:
        return {
            "privileged_current_pose_7d": current_pose,
            "privileged_motion_target_pose_7d": nan_pose,
            "privileged_basin_center_pose_7d": nan_pose,
            "privileged_pregrasp_target_pose_7d": nan_pose,
            "privileged_grasp_commit_target_pose_7d": nan_pose,
            "privileged_insert_target_pose_7d": nan_pose,
            "privileged_object_anchor_pose_7d": np.asarray(object_pose, dtype=np.float32),
            "privileged_current_delta_basin_target": nan_delta,
            "privileged_current_delta_basin_target_raw": nan_delta,
            "privileged_current_delta_basin_target_folded": nan_delta,
            "privileged_target_provider_source": np.asarray("label_exception"),
            "privileged_target_provider_uses_privileged": np.asarray(0.0, dtype=np.float32),
        }


def compute_wrist_visibility_stats(
    wrist_depth_96: np.ndarray | None,
    *,
    near_threshold: float = 0.04,
    occluded_valid_ratio_threshold: float = 0.60,
    occluded_near_fraction_threshold: float = 0.45,
):
    if wrist_depth_96 is None:
        return 0.0, 1.0, True, True
    depth = np.asarray(wrist_depth_96, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    finite_mask = np.isfinite(depth)
    valid_ratio = float(np.mean(finite_mask)) if depth.size > 0 else 0.0
    near_fraction = (
        float(np.mean(np.logical_and(finite_mask, depth < float(near_threshold))))
        if depth.size > 0
        else 1.0
    )
    is_occluded = bool(
        valid_ratio < float(occluded_valid_ratio_threshold)
        or near_fraction > float(occluded_near_fraction_threshold)
    )
    is_low_visibility = bool(
        valid_ratio < float(occluded_valid_ratio_threshold + 0.10)
        or near_fraction > float(max(0.30, occluded_near_fraction_threshold - 0.10))
    )
    return valid_ratio, near_fraction, is_occluded, is_low_visibility


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
        if (
            alignment_controller is not None
            and getattr(alignment_controller, "_controller_type", "") == "pose_field_scorer"
            and getattr(args, "near_ready_alignment_ckpt", None)
        ):
            near_ready_alignment_controller = _load_optional_controller(args.near_ready_alignment_ckpt)
            if (
                near_ready_alignment_controller is not None
                and getattr(near_ready_alignment_controller, "_controller_type", "") == "pose_field_scorer"
            ):
                base_actions = np.asarray(getattr(alignment_controller, "_candidate_actions_local", torch.empty(0)).cpu().numpy(), dtype=np.float32)
                nr_actions = np.asarray(getattr(near_ready_alignment_controller, "_candidate_actions_local", torch.empty(0)).cpu().numpy(), dtype=np.float32)
                if base_actions.shape == nr_actions.shape and np.allclose(base_actions, nr_actions, atol=1e-6):
                    alignment_controller._near_ready_alignment_controller = near_ready_alignment_controller
                    alignment_controller._near_ready_alignment_xy_multiplier = 2.5
                    alignment_controller._near_ready_alignment_z_multiplier = 4.0
                    alignment_controller._near_ready_alignment_yaw_multiplier = 2.0
                    print(f"[eval] Loaded near-ready specialist scorer from {args.near_ready_alignment_ckpt}")
                else:
                    print(
                        "[eval] Near-ready specialist scorer candidate bank mismatch; "
                        "specialist scorer disabled."
                    )
        if (
            alignment_controller is not None
            and getattr(alignment_controller, "_controller_type", "") == "pose_field_scorer"
            and getattr(args, "handoff_provider_ckpt", None)
        ):
            near_ready_rerank_model = maybe_load_near_ready_xyyaw_predictor(args.handoff_provider_ckpt)
            if near_ready_rerank_model is not None:
                alignment_controller._near_ready_rerank_model = near_ready_rerank_model
                alignment_controller._near_ready_rerank_topk = 5
                alignment_controller._near_ready_rerank_xy_multiplier = 4.0
                alignment_controller._near_ready_rerank_z_multiplier = 3.0
                alignment_controller._near_ready_rerank_yaw_multiplier = 2.5
                alignment_controller._near_ready_rerank_weight_xy = 1.0
                alignment_controller._near_ready_rerank_weight_yaw = 0.75
                alignment_controller._near_ready_rerank_weight_score = 0.05
        if (
            alignment_controller is not None
            and getattr(alignment_controller, "_controller_type", "") == "pose_field_scorer"
            and getattr(args, "residual_score_adapter_ckpt", None)
        ):
            residual_adapter = _load_optional_controller(args.residual_score_adapter_ckpt)
            if (
                residual_adapter is not None
                and getattr(residual_adapter, "_controller_type", "")
                in ("near_ready_residual_adapter", "near_ready_group_residual_adapter")
            ):
                alignment_controller._near_ready_residual_adapter = residual_adapter
                alignment_controller._near_ready_residual_xy_multiplier = 4.0
                alignment_controller._near_ready_residual_z_multiplier = 2.0
                alignment_controller._near_ready_residual_yaw_multiplier = 2.0
                alignment_controller._near_ready_residual_gate_requires_support = True
                adapter_kind = str(getattr(residual_adapter, "_controller_type", "near_ready_residual_adapter"))
                print(f"[eval] Loaded {adapter_kind} from {args.residual_score_adapter_ckpt}")
        if (
            alignment_controller is not None
            and getattr(alignment_controller, "_controller_type", "") == "pose_field_scorer"
            and getattr(args, "student_group_selector_shadow_ckpt", None)
        ):
            alignment_controller._student_group_selector_shadow = load_student_group_selector_shadow(
                args.student_group_selector_shadow_ckpt,
                handoff_ckpt=getattr(args, "student_group_selector_handoff_ckpt", None),
            )
            alignment_controller._student_group_selector_shadow_only = True
            if getattr(args, "student_b1_apply_gate_shadow_ckpt", None):
                alignment_controller._student_b1_apply_gate_shadow = load_b1_apply_gate_shadow(
                    args.student_b1_apply_gate_shadow_ckpt
                )
            alignment_controller._student_group_selector_bounded = bool(
                getattr(args, "enable_b1_group_selector_bounded", False)
            )
        if (
            alignment_controller is not None
            and getattr(alignment_controller, "_controller_type", "") == "pose_field_scorer"
            and getattr(args, "student_candidate_evaluator_shadow_ckpt", None)
        ):
            alignment_controller._student_candidate_evaluator_shadow = load_student_candidate_evaluator_shadow(
                args.student_candidate_evaluator_shadow_ckpt,
                handoff_ckpt=getattr(args, "student_candidate_evaluator_handoff_ckpt", None),
                mode_input_path=getattr(args, "student_candidate_evaluator_mode_input_path", "summary_only"),
            )
            alignment_controller._student_candidate_evaluator_shadow_only = True
            alignment_controller._b2_candidate_shadow_yaw_probe_values = tuple(
                float(v.strip())
                for v in str(getattr(args, "b2_candidate_shadow_yaw_probe_values", "") or "").split(",")
                if v.strip()
            )
            runtime_yaw_probe_values = tuple(
                float(v.strip())
                for v in str(getattr(args, "runtime_candidate_yaw_probe_values", "") or "").split(",")
                if v.strip()
            )
            if runtime_yaw_probe_values:
                augment_runtime_pose_field_candidate_bank(alignment_controller, runtime_yaw_probe_values)
        alignment_v3_shadow_controller = None
        if getattr(args, "alignment_v3_shadow_ckpt", None):
            alignment_v3_shadow_controller = load_alignment_v3_shadow_controller(args.alignment_v3_shadow_ckpt)
        alignment_v4_shadow_controller = None
        if getattr(args, "alignment_v4_shadow_ckpt", None):
            alignment_v4_shadow_controller = load_alignment_v4_shadow_controller(args.alignment_v4_shadow_ckpt)
        alignment_diffusion_controller = None
        if getattr(args, "alignment_diffusion_ckpt", None):
            alignment_diffusion_controller = load_alignment_diffusion_controller(
                args.alignment_diffusion_ckpt,
                horizon=getattr(args, "alignment_diffusion_horizon", 8),
                max_pos_step=getattr(args, "alignment_diffusion_max_pos_step", 0.0015),
                max_yaw_step=getattr(args, "alignment_diffusion_max_yaw_step", 0.0060),
            )
        alignment_tc_diffusion_controller = None
        if getattr(args, "alignment_tc_diffusion_ckpt", None):
            alignment_tc_diffusion_controller = load_alignment_tc_diffusion_controller(
                args.alignment_tc_diffusion_ckpt,
                horizon=getattr(args, "alignment_diffusion_horizon", 8),
                max_pos_step=getattr(args, "alignment_diffusion_max_pos_step", 0.0015),
                max_yaw_step=getattr(args, "alignment_diffusion_max_yaw_step", 0.0060),
            )
        alignment_tc_student_vnext_controller = None
        if getattr(args, "alignment_tc_student_vnext_ckpt", None):
            alignment_tc_student_vnext_controller = load_alignment_tc_student_vnext_controller(
                args.alignment_tc_student_vnext_ckpt,
                horizon=getattr(args, "alignment_diffusion_horizon", 8),
                max_pos_step=getattr(args, "alignment_diffusion_max_pos_step", 0.0015),
                max_yaw_step=getattr(args, "alignment_diffusion_max_yaw_step", 0.0060),
            )
        if alignment_controller is not None:
            disable_close_contract = bool(
                getattr(args, "disable_learned_target_close_stage_orientation_contract", False)
                or (
                    alignment_v3_shadow_controller is not None
                    and not getattr(args, "enable_learned_target_close_stage_orientation_contract_for_v3", False)
                )
            )
            if disable_close_contract:
                alignment_controller._disable_learned_target_close_stage_orientation_contract = True
        use_internal_readiness = bool(
            alignment_controller is not None
            and getattr(alignment_controller, "_controller_type", "") == "pose_field_scorer"
            and bool(getattr(alignment_controller, "_readiness_heads_loaded", False))
        )
        return StageAwareRefiner(
            mode=args.stage_refiner_mode,
            alignment_controller=alignment_controller,
            alignment_v3_shadow_controller=alignment_v3_shadow_controller["model"] if alignment_v3_shadow_controller else None,
            alignment_v4_shadow_controller=alignment_v4_shadow_controller["model"] if alignment_v4_shadow_controller else None,
            alignment_v4_shadow_micro_only=getattr(args, "alignment_v4_shadow_micro_only", False),
            alignment_diffusion_controller=alignment_diffusion_controller["model"] if alignment_diffusion_controller else None,
            alignment_tc_diffusion_controller=alignment_tc_diffusion_controller["model"] if alignment_tc_diffusion_controller else None,
            alignment_tc_student_vnext_controller=alignment_tc_student_vnext_controller["model"] if alignment_tc_student_vnext_controller else None,
            alignment_tc_student_vnext_collector_like=getattr(
                args, "alignment_tc_student_vnext_collector_like", False
            ),
            enable_alignment_tc_student_vnext_ready_gate=getattr(
                args, "enable_alignment_tc_student_vnext_ready_gate", False
            ),
            alignment_tc_student_vnext_close_ready_threshold=getattr(
                args, "alignment_tc_student_vnext_close_ready_threshold", 0.5
            ),
            alignment_tc_student_vnext_handoff_ready_threshold=getattr(
                args, "alignment_tc_student_vnext_handoff_ready_threshold", 0.5
            ),
            enable_alignment_diffusion_shadow=getattr(args, "enable_alignment_diffusion_shadow", False),
            enable_alignment_diffusion_apply=getattr(args, "enable_alignment_diffusion_apply", False),
            alignment_diffusion_horizon=getattr(args, "alignment_diffusion_horizon", 8),
            alignment_diffusion_num_samples=getattr(args, "alignment_diffusion_num_samples", 16),
            alignment_diffusion_apply_mode=getattr(args, "alignment_diffusion_apply_mode", "additive"),
            alignment_diffusion_max_pos_step=getattr(args, "alignment_diffusion_max_pos_step", 0.0015),
            alignment_diffusion_max_yaw_step=getattr(args, "alignment_diffusion_max_yaw_step", 0.0060),
            alignment_diffusion_risk_threshold=getattr(args, "alignment_diffusion_risk_threshold", 0.65),
            alignment_diffusion_trigger_mode=getattr(args, "alignment_diffusion_trigger_mode", "near_contact_stall"),
            alignment_diffusion_execute_steps=getattr(args, "alignment_diffusion_execute_steps", 1),
            enable_alignment_tc_diffusion_shadow=getattr(args, "enable_alignment_tc_diffusion_shadow", False),
            enable_alignment_tc_diffusion_apply=getattr(args, "enable_alignment_tc_diffusion_apply", False),
            alignment_tc_diffusion_num_samples=getattr(args, "alignment_tc_diffusion_num_samples", 8),
            alignment_tc_diffusion_top_k=getattr(args, "alignment_tc_diffusion_top_k", 3),
            alignment_tc_diffusion_confidence_threshold=getattr(
                args, "alignment_tc_diffusion_confidence_threshold", 0.55
            ),
            alignment_tc_diffusion_risk_threshold=getattr(args, "alignment_tc_diffusion_risk_threshold", 0.65),
            alignment_tc_diffusion_soft_clamp=getattr(args, "alignment_tc_diffusion_soft_clamp", False),
            alignment_tc_diffusion_workspace_tolerance=getattr(
                args, "alignment_tc_diffusion_workspace_tolerance", 0.0
            ),
            alignment_tc_diffusion_workspace_soft_clamp=getattr(
                args, "alignment_tc_diffusion_workspace_soft_clamp", False
            ),
            alignment_tc_diffusion_execute_steps=getattr(args, "alignment_tc_diffusion_execute_steps", 1),
            enable_alignment_tc_student_vnext_shadow=getattr(args, "enable_alignment_tc_student_vnext_shadow", False),
            enable_alignment_tc_student_vnext_apply=getattr(args, "enable_alignment_tc_student_vnext_apply", False),
            alignment_tc_student_vnext_corridor_json=getattr(args, "alignment_tc_student_vnext_corridor_json", None),
            enable_alignment_v4_apply=getattr(args, "enable_alignment_v4_apply", False),
            alignment_v4_apply_scale=getattr(args, "alignment_v4_apply_scale", 1.0),
            alignment_v4_apply_max_pos=getattr(args, "alignment_v4_apply_max_pos", 0.0010),
            alignment_v4_apply_max_yaw=getattr(args, "alignment_v4_apply_max_yaw", 0.0040),
            alignment_v4_apply_only_when_gate_pass=getattr(
                args, "alignment_v4_apply_only_when_gate_pass", True
            ),
            alignment_v4_apply_require_improve=getattr(args, "alignment_v4_apply_require_improve", False),
            alignment_v4_apply_micro_only=getattr(args, "alignment_v4_apply_micro_only", False),
            enable_alignment_v3_apply=getattr(args, "enable_alignment_v3_apply", False),
            alignment_v3_apply_scale=getattr(args, "alignment_v3_apply_scale", 1.0),
            alignment_v3_apply_max_pos=getattr(args, "alignment_v3_apply_max_pos", 0.0020),
            alignment_v3_apply_max_yaw=getattr(args, "alignment_v3_apply_max_yaw", 0.0020),
            alignment_v3_apply_only_when_gate_pass=getattr(
                args, "alignment_v3_apply_only_when_gate_pass", True
            ),
            alignment_v3_apply_require_improve=getattr(args, "alignment_v3_apply_require_improve", False),
            contact_controller=contact_controller,
            max_residual_pos=args.max_residual_pos,
            max_residual_rot=args.max_residual_rot,
            learned_residual_scale=args.learned_residual_scale,
            max_alignment_corrections_per_window=args.max_alignment_corrections_per_window,
            require_close_intent_for_alignment=args.require_close_intent_for_alignment,
            require_close_intent_for_refine_band=getattr(args, "require_close_intent_for_refine_band", False),
            enable_alignment_pose=args.enable_alignment_pose,
            use_pose_alpha=args.use_pose_alpha,
            enable_outer_rescue=args.enable_outer_rescue,
            outer_rescue_xy_scale=args.outer_rescue_xy_scale,
            outer_rescue_abs_z_scale=args.outer_rescue_abs_z_scale,
            outer_rescue_yaw_scale=args.outer_rescue_yaw_scale,
            outer_rescue_min_xy=args.outer_rescue_min_xy,
            outer_rescue_min_abs_z=args.outer_rescue_min_abs_z,
            outer_rescue_min_yaw=args.outer_rescue_min_yaw,
            enable_alignment_close_veto=args.enable_alignment_close_veto,
            close_veto_xy_threshold=args.close_veto_xy_threshold,
            close_veto_abs_z_threshold=args.close_veto_abs_z_threshold,
            close_veto_yaw_threshold=args.close_veto_yaw_threshold,
            close_veto_ready_streak_frames=args.close_veto_ready_streak_frames,
            close_veto_settle_steps=args.close_veto_settle_steps,
            close_latch_enabled=args.close_latch_enabled,
            close_latch_steps=args.close_latch_steps,
            alignment_takeover_until_close_ready=args.alignment_takeover_until_close_ready,
            alignment_assist_xy_scale=args.alignment_assist_xy_scale,
            alignment_assist_abs_z_scale=args.alignment_assist_abs_z_scale,
            alignment_assist_yaw_scale=args.alignment_assist_yaw_scale,
            alignment_assist_base_scale=args.alignment_assist_base_scale,
            alignment_assist_close_block_base_scale=args.alignment_assist_close_block_base_scale,
            alignment_takeover_xy_scale=args.alignment_takeover_xy_scale,
            alignment_takeover_abs_z_scale=args.alignment_takeover_abs_z_scale,
            alignment_takeover_yaw_scale=args.alignment_takeover_yaw_scale,
            alignment_takeover_motion_xy_threshold=args.alignment_takeover_motion_xy_threshold,
            alignment_takeover_motion_abs_z_threshold=args.alignment_takeover_motion_abs_z_threshold,
            alignment_takeover_close_block_xy_threshold=args.alignment_takeover_close_block_xy_threshold,
            alignment_takeover_close_block_abs_z_threshold=args.alignment_takeover_close_block_abs_z_threshold,
            alignment_zone_hysteresis=args.alignment_zone_hysteresis,
            alignment_candidate_hold_steps=args.alignment_candidate_hold_steps,
            alignment_candidate_switch_margin=args.alignment_candidate_switch_margin,
            alignment_action_lowpass_alpha=args.alignment_action_lowpass_alpha,
            enable_alignment_physical_mask=bool(getattr(args, "enable_alignment_physical_mask", False)),
            enable_alignment_low_conf_noop=bool(getattr(args, "enable_alignment_low_conf_noop", False)),
            skip_alignment_when_close_ready=args.skip_alignment_when_close_ready,
            skip_alignment_ready_xy_threshold=args.skip_alignment_ready_xy_threshold,
            skip_alignment_ready_abs_z_threshold=args.skip_alignment_ready_abs_z_threshold,
            skip_alignment_ready_yaw_threshold=args.skip_alignment_ready_yaw_threshold,
            enable_readiness_gripper=bool(args.enable_readiness_gripper or use_internal_readiness),
            readiness_close_threshold=args.readiness_close_threshold,
            gripper_override_confidence=args.gripper_override_confidence,
            b1_group_shadow_gate_mode=getattr(args, "b1_group_shadow_gate_mode", "broad"),
            b2_candidate_shadow_gate_mode=getattr(args, "b2_candidate_shadow_gate_mode", "broad"),
            enable_b2_candidate_bounded_v0=getattr(args, "enable_b2_candidate_bounded_v0", False),
            b2_candidate_apply_conf_threshold=getattr(args, "b2_candidate_apply_conf_threshold", 0.431),
            b2_candidate_apply_margin_threshold=getattr(args, "b2_candidate_apply_margin_threshold", 0.010),
            close_veto_runtime_geometry_fallback_for_bounded=getattr(
                args,
                "close_veto_runtime_geometry_fallback_for_bounded",
                False,
            ),
            enable_bounded_auto_close_on_alignment=getattr(args, "enable_bounded_auto_close_on_alignment", False),
            bounded_auto_close_stable_frames=getattr(args, "bounded_auto_close_stable_frames", 1),
            bounded_auto_close_xy_threshold=getattr(args, "bounded_auto_close_xy_threshold", -1.0),
            bounded_auto_close_abs_z_threshold=getattr(args, "bounded_auto_close_abs_z_threshold", -1.0),
            bounded_auto_close_yaw_threshold=getattr(args, "bounded_auto_close_yaw_threshold", -1.0),
            enable_force_close_after_b2_eval=getattr(args, "enable_force_close_after_b2_eval", False),
            enable_alignment_near_zone_gate=getattr(args, "enable_alignment_near_zone_gate", False),
            alignment_near_zone_xy_threshold=getattr(args, "alignment_near_zone_xy_threshold", 0.05),
            alignment_near_zone_z_threshold=getattr(args, "alignment_near_zone_z_threshold", 0.10),
            enable_v2_nearzone_assist=getattr(args, "enable_v2_nearzone_assist", False),
            v2_assist_scale_cap=getattr(args, "v2_assist_scale_cap", 0.25),
            v2_assist_max_pos=getattr(args, "v2_assist_max_pos", 0.0025),
            v2_assist_max_rot=getattr(args, "v2_assist_max_rot", 0.015),
            v2_assist_only_when_gate_pass=getattr(args, "v2_assist_only_when_gate_pass", True),
            enable_v2_predictor_micro_assist_apply=getattr(
                args, "enable_v2_predictor_micro_assist_apply", False
            ),
            v2_predictor_micro_assist_scale_cap=getattr(
                args, "v2_predictor_micro_assist_scale_cap", 0.05
            ),
            v2_predictor_micro_assist_max_pos=getattr(
                args, "v2_predictor_micro_assist_max_pos", 0.00075
            ),
            v2_predictor_micro_assist_max_rot=getattr(
                args, "v2_predictor_micro_assist_max_rot", 0.0040
            ),
            v2_predictor_micro_assist_only_when_gate_pass=getattr(
                args, "v2_predictor_micro_assist_only_when_gate_pass", True
            ),
            enable_target_delta_servo_shadow=getattr(args, "enable_target_delta_servo_shadow", False),
            enable_target_delta_servo_apply=getattr(args, "enable_target_delta_servo_apply", False),
            target_delta_servo_bypass_gates=getattr(args, "target_delta_servo_bypass_gates", False),
            target_delta_servo_apply_once_per_episode=getattr(args, "target_delta_servo_apply_once_per_episode", False),
            target_delta_servo_source=getattr(args, "target_delta_servo_source", "predictor"),
            target_delta_servo_k_xy=getattr(args, "target_delta_servo_k_xy", 0.08),
            target_delta_servo_k_z=getattr(args, "target_delta_servo_k_z", 0.06),
            target_delta_servo_k_yaw=getattr(args, "target_delta_servo_k_yaw", 0.04),
            target_delta_servo_max_pos=getattr(args, "target_delta_servo_max_pos", 0.0010),
            target_delta_servo_max_yaw=getattr(args, "target_delta_servo_max_yaw", 0.0040),
            target_delta_servo_apply_xy_threshold=getattr(args, "target_delta_servo_apply_xy_threshold", 0.03),
            target_delta_servo_apply_abs_z_threshold=getattr(args, "target_delta_servo_apply_abs_z_threshold", 0.07),
            target_delta_servo_apply_yaw_threshold=getattr(args, "target_delta_servo_apply_yaw_threshold", 0.25),
            enable_phase1_force_reflex=getattr(args, "phase1_force_reflex_enable", False),
            phase1_force_contact_fz_threshold=getattr(args, "phase1_force_contact_fz_threshold", 0.35),
            phase1_force_force_norm_threshold=getattr(args, "phase1_force_force_norm_threshold", 0.50),
            phase1_force_high_fz_threshold=getattr(args, "phase1_force_high_fz_threshold", 2.5),
            phase1_force_lateral_threshold=getattr(args, "phase1_force_lateral_threshold", 1.5),
            phase1_force_torque_threshold=getattr(args, "phase1_force_torque_threshold", 1.0),
            phase1_force_spike_threshold=getattr(args, "phase1_force_spike_threshold", 0.75),
            phase1_force_backoff_mm=getattr(args, "phase1_force_backoff_mm", 0.0015),
            phase1_close_confirm_steps=getattr(args, "phase1_close_confirm_steps", 2),
            phase1_close_fail_steps=getattr(args, "phase1_close_fail_steps", 6),
            phase1_post_contact_hold_steps=getattr(args, "phase1_post_contact_hold_steps", 8),
            phase1_reopen_cooldown_steps=getattr(args, "phase1_reopen_cooldown_steps", 4),
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


def build_coarse2contact_supervisor(args):
    mode = str(getattr(args, "coarse2contact_mode", "off"))
    if mode == "off":
        return None
    return Coarse2ContactSupervisor(
        mode=mode,
        shadow_only=bool(getattr(args, "coarse2contact_shadow_only", False)),
        visual_xy_threshold=float(getattr(args, "coarse2contact_visual_xy_threshold", 0.0015)),
        visual_yaw_threshold=float(getattr(args, "coarse2contact_visual_yaw_threshold", 0.0349)),
        visual_precontact_depth_threshold=float(getattr(args, "coarse2contact_precontact_depth_threshold", 0.20)),
        visual_contact_depth_threshold=float(getattr(args, "coarse2contact_contact_depth_threshold", 0.035)),
        max_xy_step=float(getattr(args, "coarse2contact_max_xy_step", 0.0005)),
        max_yaw_step=float(getattr(args, "coarse2contact_max_yaw_step", 0.0087)),
        force_contact_threshold=float(getattr(args, "coarse2contact_force_contact_threshold", 0.18)),
        force_delta_contact_threshold=float(getattr(args, "coarse2contact_force_delta_contact_threshold", 0.05)),
        force_jam_threshold=float(getattr(args, "coarse2contact_force_jam_threshold", 0.55)),
        force_torque_threshold=float(getattr(args, "coarse2contact_force_torque_threshold", 0.12)),
        force_spike_threshold=float(getattr(args, "coarse2contact_force_spike_threshold", 1.0)),
        backoff_m=float(getattr(args, "coarse2contact_backoff_m", 0.003)),
        lateral_m=float(getattr(args, "coarse2contact_lateral_m", 0.0015)),
        chunk_size=int(getattr(args, "coarse2contact_chunk_size", 4)),
    )


def evaluate(args, preloaded_components=None):
    _lazy_import_tasks()
    assert args.task_name in TASK_MAP, f"Unknown task: {args.task_name}. Available: {list(TASK_MAP)}"
    load_grasp_basin_profile(args)

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
    coarse2contact = build_coarse2contact_supervisor(args)
    if coarse2contact is not None and refiner is not None:
        raise ValueError(
            "Coarse2Contact runtime is an independent first-stage scaffold. "
            "Run it with --coarse2contact_mode and without legacy refiner/alignment/residual modes."
        )
    # --- v2 target-conditioned alignment shadow ---
    v2_shadow_ckpt = getattr(args, "v2_alignment_shadow_ckpt", None)
    if v2_shadow_ckpt and refiner is not None and hasattr(refiner, "alignment_controller"):
        from prismatic.models.target_conditioned_alignment_policy import TargetConditionedAlignmentPolicy
        print(f"[eval] Loading v2 alignment shadow from {v2_shadow_ckpt}", flush=True)
        v2_ckpt = torch.load(v2_shadow_ckpt, map_location="cpu", weights_only=False)
        v2_model = TargetConditionedAlignmentPolicy(proposal_count=8).to(DEVICE).eval()
        v2_model.load_state_dict(v2_ckpt["model_state_dict"], strict=False)
        refiner._v2_shadow = v2_model
        print(f"[eval] v2 shadow loaded (top1={v2_ckpt.get('val_top1', '?')})", flush=True)
    # --- v2 target delta predictor ---
    v2_td_ckpt = getattr(args, "v2_target_delta_provider_ckpt", None)
    if v2_td_ckpt and refiner is not None:
        from prismatic.models.target_delta_predictor import TargetDeltaPredictor
        print(f"[eval] Loading v2 target delta predictor from {v2_td_ckpt}", flush=True)
        v2_td_payload = torch.load(v2_td_ckpt, map_location="cpu", weights_only=False)
        v2_td_state = v2_td_payload["model_state_dict"]
        is_legacy_motion_ckpt = "delta_head.weight" not in v2_td_state and "head.4.weight" in v2_td_state
        refiner._v2_target_delta_predictor = TargetDeltaPredictor(legacy_output_head=is_legacy_motion_ckpt)
        missing, unexpected = refiner._v2_target_delta_predictor.load_state_dict(v2_td_state, strict=False)
        refiner._v2_target_delta_predictor.to(DEVICE).eval()
        print(
            f"[eval] v2 target delta predictor loaded (legacy={is_legacy_motion_ckpt}, "
            f"missing={len(missing)}, unexpected={len(unexpected)})",
            flush=True,
        )
    gripper_supervisor = build_gripper_supervisor(args)
    # A trained PoseFieldScorer carries the exact candidate bank it was trained
    # with in its dataset npz.  Do not overwrite that bank during normal eval:
    # changing the runtime candidates silently breaks the scorer/controller
    # contract (for example, xy-micro checkpoints need their 61-action bank).
    # The hand-built teacher bank is only for oracle-executed collectors, where
    # the privileged teacher itself defines the candidate supervision source.
    if (
        isinstance(refiner, StageAwareRefiner)
        and getattr(refiner.alignment_controller, "_controller_type", "") == "pose_field_scorer"
        and (
            bool(getattr(args, "oracle_executed_align_collect", False))
            or bool(getattr(args, "oracle_executed_pregrasp_collect", False))
            or bool(getattr(args, "use_legacy_teacher_candidate_bank_for_scorer", False))
        )
    ):
        yaw_probe_values = tuple(
            float(v.strip())
            for v in str(getattr(args, "teacher_candidate_yaw_probe_values", "") or "").split(",")
            if v.strip()
        )
        teacher_candidate_actions, teacher_candidate_group_index, teacher_candidate_kind, teacher_candidate_base_mask = build_teacher_candidate_bank(
            xy_small=float(getattr(args, "primitive_xy_small", 0.004)),
            xy_large=float(getattr(args, "primitive_xy_large", 0.008)),
            z_small=float(getattr(args, "primitive_z_small", 0.004)),
            yaw_small=float(getattr(args, "primitive_yaw_small", 0.03)),
            yaw_probe_values=yaw_probe_values,
            rescue_pitch_small=float(getattr(args, "teacher_rescue_pitch_small", 0.04)),
            rescue_roll_small=float(getattr(args, "teacher_rescue_roll_small", 0.04)),
        )
        controller = refiner.alignment_controller
        controller._teacher_candidate_actions_local = teacher_candidate_actions
        controller._teacher_candidate_group_index = teacher_candidate_group_index
        controller._teacher_candidate_kind = teacher_candidate_kind
        controller._teacher_candidate_base_mask = teacher_candidate_base_mask
        controller._runtime_candidate_mask = teacher_candidate_base_mask.copy()
        controller._teacher_pregrasp_stall_count = 0
        runtime_yaw_probe_values = tuple(
            float(v.strip())
            for v in str(getattr(args, "runtime_candidate_yaw_probe_values", "") or "").split(",")
            if v.strip()
        )
        if runtime_yaw_probe_values:
            augment_runtime_pose_field_candidate_bank(controller, runtime_yaw_probe_values)

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

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete(),
    )

    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    task_cls = TASK_MAP[args.task_name]
    task = env.get_task(task_cls)
    stage_target_provider = build_stage_target_provider(
        getattr(args, "target_provider_mode", "legacy_auto"),
        ckpt_path=getattr(args, "target_provider_ckpt", None),
        handoff_ckpt_path=getattr(args, "handoff_provider_ckpt", None),
    )
    student_vnext_ckpt = getattr(args, "alignment_tc_student_vnext_ckpt", None)
    if student_vnext_ckpt:
        target_provider_mode = getattr(args, "target_provider_mode", "legacy_auto")
        if target_provider_mode in ("legacy_auto", "teacher_oracle"):
            raise ValueError(
                "Student vNext evaluation must not use teacher_oracle/legacy_auto target providers. "
                "Use learned or canonical_fallback, and keep --enforce_no_privileged_runtime enabled."
            )
        if not bool(getattr(args, "enforce_no_privileged_runtime", False)):
            raise ValueError(
                "Student vNext evaluation requires --enforce_no_privileged_runtime to avoid privileged runtime targets."
            )
        if bool(getattr(args, "record_teacher_truth_metrics", False)):
            raise ValueError(
                "Student vNext evaluation must not record teacher truth metrics; that path is privileged."
            )
    teacher_truth_provider = (
        build_stage_target_provider("teacher_oracle")
        if bool(getattr(args, "record_teacher_truth_metrics", False))
        else None
    )
    if (
        bool(getattr(args, "enforce_no_privileged_runtime", False))
        and getattr(args, "target_provider_mode", "legacy_auto") in ("legacy_auto", "teacher_oracle")
    ):
        raise ValueError(
            "--enforce_no_privileged_runtime was requested, but target_provider_mode is "
            f"{getattr(args, 'target_provider_mode', 'legacy_auto')!r}. Use learned or canonical_fallback."
        )
    if (
        args.use_stage_aware_refiner
        and getattr(args, "stage_refiner_mode", None) == "alignment"
        and getattr(args, "target_provider_mode", "legacy_auto") in ("legacy_auto", "teacher_oracle")
    ):
        print(
            "[eval] WARNING: running with privileged teacher target provider; "
            "treat this as oracle-target upper bound, not deployment mainline."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    if args.record_video and args.write_episode_videos:
        video_dir.mkdir(exist_ok=True)
    gripper_trace_dir = output_dir / "gripper_traces"
    if args.record_gripper_trace:
        gripper_trace_dir.mkdir(exist_ok=True)

    results = {
        "successes": [],
        "episode_lengths": [],
        "stage_stats": [],
        "invalid_action_counts": [],
        "target_provider_mode": str(getattr(args, "target_provider_mode", "legacy_auto")),
        "target_provider_ckpt": getattr(args, "target_provider_ckpt", None),
        "handoff_provider_ckpt": getattr(args, "handoff_provider_ckpt", None),
        "enforce_no_privileged_runtime": bool(getattr(args, "enforce_no_privileged_runtime", False)),
        "coarse2contact_mode": str(getattr(args, "coarse2contact_mode", "off")),
        "uses_privileged_target": False,
        "video_paths": [],
    }
    if refiner is not None:
        results["refiner_stats"] = []
    if coarse2contact is not None:
        results["coarse2contact_stats"] = []
    if gripper_supervisor is not None:
        results["gripper_supervisor_stats"] = []
    if args.record_gripper_trace:
        results["gripper_trace_paths"] = []
    if args.episode_indices:
        episode_indices = [int(x.strip()) for x in str(args.episode_indices).split(",") if x.strip()]
    else:
        episode_indices = list(range(args.num_episodes))
    results["episode_indices"] = [int(x) for x in episode_indices]
    results["collector_like_demo_reset"] = bool(getattr(args, "collector_like_demo_reset", False))
    support_state_rows = [] if (args.support_states_output_npz or args.fire_distill_output_npz) else None
    alignment_diffusion_rows = [] if args.alignment_diffusion_raw_output_npz else None

    best_success_frames = None
    best_success_len = float("inf")
    best_fail_frames = None
    best_fail_len = 0
    track_best_gif = bool(args.write_best_gif)

    for loop_idx, ep_idx in enumerate(episode_indices):
        print(f"\n--- Episode {loop_idx + 1}/{len(episode_indices)} ---")
        if coarse2contact is not None:
            coarse2contact.reset()
        if args.eval_seed is not None:
            ep_seed = int(args.eval_seed) + ep_idx
            random.seed(ep_seed)
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)
        if bool(getattr(args, "collector_like_demo_reset", False)):
            try:
                demos = task.get_demos(
                    1,
                    live_demos=True,
                    random_selection=False,
                    from_episode_number=int(ep_idx),
                    max_attempts=max(10, int(getattr(args, "demo_max_attempts", 10))),
                )
            except Exception as exc:
                demos = []
                print(f"  failed to load demo for episode {ep_idx}: {exc}", flush=True)
            if not demos:
                results["stage_stats"].append(
                    {
                        "episode_index": int(ep_idx),
                        "phase_counts": {int(phase): 0 for phase in StagePhase},
                        "failure_counts": {},
                        "invalid_action_count": 0,
                        "invalid_action_exception_count": 0,
                        "transition_count": 0,
                        "phase_transition_rate": 0.0,
                        "max_phase_reached": 0,
                        "subgoal_progress": {},
                        "align_entry": False,
                        "interact_entry": False,
                        "recover_entry": False,
                        "final_failure_mode": -1,
                        "final_failure_mode_name": "missing_demo",
                        "workspace_violation_count": 0,
                        "workspace_violation_mean": 0.0,
                        "workspace_violation_max": 0.0,
                        "first_close_step": -1,
                        "close_before_ready_count": 0,
                        "close_before_basin_count": 0,
                        "reopen_after_close_count": 0,
                        "gripper_flip_count": 0,
                        "grasp_lift_proxy": False,
                        "oracle_executed_alignment_count": 0,
                        "oracle_executed_noop_count": 0,
                        "oracle_pregrasp_success": False,
                        "oracle_pregrasp_close_count": 0,
                        "oracle_pregrasp_retry_count": 0,
                        "best_hold_run": 0,
                        "fire_anchor_step": -1,
                        "substage_id": -1,
                        "substage": "missing_demo",
                        "has_object_in_hand": False,
                        "phase1_verified_grasp_reached": False,
                        "phase1_grasp_contact_confirmed": False,
                        "phase1_close_command_source": "missing_demo",
                        "phase1_reopen_reason": "missing_demo",
                        "phase1_force_spike_count": 0,
                        "phase1_jam_detected_count": 0,
                        "contact_state_id": -1,
                        "contact_state": "missing_demo",
                        "stage_target_mode_id": -1,
                        "stage_target_mode": "missing_demo",
                        "refiner_phase_id": -1,
                        "refiner_max_phase_reached": -1,
                        "has_object_in_hand_entered": False,
                        "phase2_entered_after_student": False,
                        "phase1_close_hold_active": False,
                        "phase1_close_hold_remaining": 0,
                        "phase1_force_reflex_active": False,
                        "phase1_force_reflex_reason": "missing_demo",
                    }
                )
                continue
            descriptions, obs = task.reset_to_demo(demos[0])
        else:
            descriptions, obs = task.reset()
        instruction = descriptions[0]
        live_target_handle = resolve_live_target_handle(task)
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
        success_reported = False
        step_count = 0
        chunk_step = 0
        stage_tracker = StageManager()
        phase_hist = {int(phase): 0 for phase in StagePhase}
        failure_hist = {}
        align_entered = False
        interact_entered = False
        recover_entered = False
        invalid_action_count = 0
        invalid_action_exception_count = 0
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
        oracle_executed_alignment_count = 0
        oracle_executed_noop_count = 0
        oracle_pregrasp_success = False
        oracle_pregrasp_close_count = 0
        oracle_pregrasp_retry_count = 0
        teacher_close_attempt_step = None
        teacher_grasp_verified_step = None
        teacher_retry_anchor_steps = []
        teacher_close_hold_remaining = 0
        teacher_retry_steps_remaining = 0
        teacher_retry_budget_remaining = int(args.teacher_max_retries)
        teacher_close_attempted_this_cycle = False
        teacher_verify_lift_streak = 0
        teacher_last_close_object_z = None
        teacher_last_close_object_pose = None
        teacher_last_close_gripper_pose = None
        teacher_yaw_correct_latched = False
        teacher_alignment_handoff_active = False
        teacher_planner_verify_remaining = 0
        teacher_realign_cooldown_remaining = 0
        teacher_smooth_action_local = None
        teacher_last_candidate_idx = None
        teacher_last_candidate_score = -1.0e9
        teacher_candidate_hold_remaining = 0
        teacher_planner_close_latch_remaining = 0
        gripper_trace = []
        support_row_start = len(support_state_rows) if support_state_rows is not None else None
        alignment_diffusion_episode_rows = [] if alignment_diffusion_rows is not None else None

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
            depth_proximity = None
            if depth_tensor_96 is not None:
                depth_arr = depth_tensor_96.detach().cpu().numpy().astype(np.float32)
                valid_depth = depth_arr[np.isfinite(depth_arr)]
                if valid_depth.size > 0:
                    depth_proximity = float(np.percentile(valid_depth, 5.0))

            if len(action_queue) == 0:
                if refiner is not None and not isinstance(refiner, StageAwareRefiner):
                    refiner.trigger.update(
                        force_reading=raw_force,
                        gripper_z=float(obs.gripper_pose[2]),
                        depth_proximity=depth_proximity,
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
                if coarse2contact is not None:
                    max_chunk = coarse2contact.get_chunk_size()
                elif refiner is not None:
                    max_chunk = refiner.get_chunk_size()
                else:
                    max_chunk = len(actions)
                action_queue = [np.asarray(actions[i], dtype=np.float32) for i in range(min(len(actions), max_chunk))]
                chunk_step = 0
                if step_idx < 40 or step_idx % 50 == 0:
                    print(
                        f"  Step {step_idx} chunk[0]: delta_xyz=[{action_queue[0][0]:+.6f},{action_queue[0][1]:+.6f},{action_queue[0][2]:+.6f}] "
                        f"rot=[{action_queue[0][3]:+.6f},{action_queue[0][4]:+.6f},{action_queue[0][5]:+.6f}] grip={action_queue[0][6]:+.4f}"
                    )

            delta_action = action_queue.pop(0)
            base_delta_action = delta_action.copy()
            base_action_local = world_delta_to_local(base_delta_action[:6], obs.gripper_pose[3:7])
            planner_base_action_local_raw = np.asarray(base_action_local, dtype=np.float32).copy()
            planner_base_action_7d_raw = np.asarray(base_delta_action, dtype=np.float32).copy()
            future_gripper_actions = [
                float(a[6]) for a in action_queue[: args.alignment_gate_lookahead]
            ]
            target_result = None
            handoff_result = None
            runtime_target_pose = None
            current_delta_basin = None
            pregrasp_target_pose = None
            grasp_commit_target_pose = None
            grasp_spec_name = "none"
            use_commit_target = False
            orientation_rescue_enabled = False
            anchor_pose = None
            gripper_context_arr = np.asarray(
                [
                    float(obs.gripper_open),
                    float(np.clip(base_delta_action[6], -1.0, 1.0)),
                    float(depth_proximity) if depth_proximity is not None else np.nan,
                ],
                dtype=np.float32,
            )
            active_alignment_controller = _active_alignment_like_controller(refiner)
            if isinstance(refiner, StageAwareRefiner) and active_alignment_controller is not None:
                controller = active_alignment_controller
                anchor_pose = getattr(controller, "_reference_anchor_pose_7d", None)
                target_result = stage_target_provider.resolve(
                    current_gripper_pose=np.asarray(obs.gripper_pose, dtype=np.float32),
                    manager=refiner.manager,
                    controller=controller,
                    live_target_handle=live_target_handle,
                    obs=obs,
                    task=task,
                    gripper_context=gripper_context_arr,
                )
                if bool(getattr(args, "enforce_no_privileged_runtime", False)):
                    privileged_target = bool(getattr(target_result, "uses_privileged_target", False))
                    privileged_handoff = bool(getattr(target_result, "handoff_uses_privileged", False))
                    if privileged_target or privileged_handoff:
                        raise RuntimeError(
                            "Privileged runtime target leaked into a no-privileged evaluation: "
                            f"provider={getattr(target_result, 'provider_name', 'unknown')}, "
                            f"source={getattr(target_result, 'source', 'unknown')}, "
                            f"uses_privileged_target={privileged_target}, "
                            f"handoff_uses_privileged={privileged_handoff}. "
                            "This run is an oracle upper bound, not a student mainline."
                        )
                runtime_target_pose = getattr(target_result, "motion_target_pose_7d", target_result.target_pose_7d)
                current_delta_basin = getattr(target_result, "motion_target_delta_local", target_result.target_delta_local)
                pregrasp_target_pose = getattr(target_result, "pregrasp_target_pose_7d", None)
                grasp_commit_target_pose = getattr(target_result, "grasp_commit_target_pose_7d", None)
                grasp_spec_name = getattr(target_result, "grasp_spec_name", "none")
                use_commit_target = bool(getattr(target_result, "use_commit_target", False))
                orientation_rescue_enabled = bool(getattr(target_result, "orientation_rescue_enabled", False))
                controller._runtime_target_provider_name = str(target_result.provider_name)
                controller._runtime_target_provider_source = str(target_result.source)
                controller._runtime_target_uses_privileged = bool(target_result.uses_privileged_target)
                controller._runtime_stage_target_mode = int(target_result.stage_target_mode)
                # Only overwrite with non-None values so that a pre-seeded
                # canonical basin centre (from load_depth_force_local_proposal_policy)
                # survives across steps where the target provider returns None.
                if runtime_target_pose is not None:
                    controller._runtime_motion_target_pose_7d = np.asarray(runtime_target_pose, dtype=np.float32).copy()
                    controller._runtime_motion_target_pose_source = str(target_result.source)
                elif not hasattr(controller, "_runtime_motion_target_pose_7d") or controller._runtime_motion_target_pose_7d is None:
                    _basin = getattr(controller, "_canonical_basin_center_pose_7d", None)
                    if _basin is not None:
                        controller._runtime_motion_target_pose_7d = np.asarray(_basin, dtype=np.float32).copy()
                        controller._runtime_motion_target_pose_source = "canonical_basin_seeded"
                if current_delta_basin is not None:
                    controller._runtime_motion_target_delta_local = np.asarray(current_delta_basin, dtype=np.float32).copy()
                    controller._runtime_motion_target_delta_source = str(target_result.source)
                controller._runtime_pregrasp_target_pose_7d = (
                    None if pregrasp_target_pose is None else np.asarray(pregrasp_target_pose, dtype=np.float32).copy()
                )
                controller._runtime_grasp_commit_target_pose_7d = (
                    None if grasp_commit_target_pose is None else np.asarray(grasp_commit_target_pose, dtype=np.float32).copy()
                )
                controller._runtime_grasp_commit_edge_pair_index = int(
                    getattr(target_result, "grasp_commit_edge_pair_index", -1)
                )
                controller._runtime_grasp_commit_edge_pair_family = int(
                    getattr(target_result, "grasp_commit_edge_pair_family", -1)
                )
                controller._runtime_grasp_commit_edge_pair_yaw_error = float(
                    getattr(target_result, "grasp_commit_edge_pair_yaw_error", np.nan)
                )
                controller._runtime_grasp_spec_name = str(grasp_spec_name)
                controller._runtime_use_commit_target = bool(use_commit_target)
                controller._runtime_orientation_rescue_enabled = bool(orientation_rescue_enabled)
                if current_delta_basin is not None:
                    current_basin_distance, _, _, _, _ = compute_basin_metrics(current_delta_basin)
                    controller._runtime_current_delta_basin_target = np.asarray(current_delta_basin, dtype=np.float32).copy()
                    controller._runtime_current_basin_distance = float(current_basin_distance)
                # Do not zero out a previously valid delta; v2 shadow needs it.
                # Only clear if we have evidence the target moved out of range.
                planner_close_intent_estimate = bool(
                    refiner._planner_close_intent(base_delta_action, future_gripper_actions)
                )
                handoff_result = stage_target_provider.resolve_handoff_only(
                    current_gripper_pose=np.asarray(obs.gripper_pose, dtype=np.float32),
                    manager=refiner.manager,
                    controller=controller,
                    live_target_handle=live_target_handle,
                    obs=obs,
                    task=task,
                    gripper_context=gripper_context_arr,
                )
                handoff_aux = dict((handoff_result or {}).get("handoff_aux", {}) or {})
                predicted_handoff_ready = bool((handoff_result or {}).get("handoff_ready", False))
                handoff_shadow_only = bool(getattr(args, "student_handoff_shadow_only", False))
                handoff_shadow_blocks_apply = bool(handoff_shadow_only)
                applied_handoff_ready = bool(
                    False if handoff_shadow_blocks_apply else (predicted_handoff_ready if planner_close_intent_estimate else False)
                )
                controller._runtime_handoff_ready_pred = bool(predicted_handoff_ready)
                controller._runtime_handoff_ready = applied_handoff_ready
                controller._runtime_handoff_ready_applied = bool(applied_handoff_ready)
                controller._runtime_handoff_metrics = dict((handoff_result or {}).get("handoff_metrics", {}) or {})
                controller._runtime_handoff_metric_thresholds = dict(
                    (handoff_result or {}).get("handoff_metric_thresholds", {}) or {}
                )
                controller._runtime_handoff_release_metric_thresholds = dict(
                    (handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}
                )
                controller._runtime_handoff_optimization_metric_thresholds = dict(
                    (handoff_result or {}).get("handoff_optimization_metric_thresholds", {}) or {}
                )
                controller._runtime_handoff_target_pose_7d = None
                controller._runtime_handoff_spec_name = str((handoff_result or {}).get("handoff_spec_name", "none"))
                controller._runtime_handoff_target_role = str((handoff_result or {}).get("handoff_target_role", "none"))
                controller._runtime_handoff_uses_privileged = bool((handoff_result or {}).get("handoff_uses_privileged", False))
                provider_min_stable = int((handoff_result or {}).get("handoff_min_stable_frames", 1))
                # The provider/spec may request a minimum stability window, while
                # the eval/runtime config can only make close release more
                # conservative.  This keeps provider booleanization unchanged but
                # lets close-veto smoke tests require consecutive ready frames.
                controller._runtime_handoff_min_stable_frames = max(
                    provider_min_stable,
                    int(getattr(args, "close_veto_ready_streak_frames", 1)),
                )
                controller._runtime_handoff_aux = dict(handoff_aux)
                controller._runtime_handoff_shadow_only = bool(handoff_shadow_only)
                controller._runtime_handoff_shadow_blocks_apply = bool(handoff_shadow_blocks_apply)
                if current_delta_basin is not None:
                    current_basin_distance, current_basin_xy, current_basin_z, current_basin_yaw, current_basin_tilt = compute_basin_metrics(
                        current_delta_basin
                    )
                    controller._runtime_current_delta_basin_target = current_delta_basin.astype(np.float32)
                    controller._runtime_current_basin_distance = float(current_basin_distance)
                    controller._runtime_target_pose_7d = (
                        None if runtime_target_pose is None else np.asarray(runtime_target_pose, dtype=np.float32).copy()
                    )
                else:
                    current_delta_basin = None
                    current_basin_distance = None
                    current_basin_xy = None
                    current_basin_z = None
                    current_basin_yaw = None
                    current_basin_tilt = None
                    controller._runtime_target_pose_7d = None
                    controller._runtime_current_delta_basin_target = np.zeros(6, dtype=np.float32)
                    controller._runtime_current_basin_distance = np.inf
                teacher_truth_result = None
                teacher_truth_handoff = None
                teacher_truth_basin_distance = None
                teacher_truth_basin_xy = None
                teacher_truth_basin_z = None
                teacher_truth_basin_yaw = None
                teacher_truth_basin_tilt = None
                teacher_truth_delta = None
                if teacher_truth_provider is not None:
                    teacher_truth_result = teacher_truth_provider.resolve(
                        current_gripper_pose=np.asarray(obs.gripper_pose, dtype=np.float32),
                        manager=refiner.manager,
                        controller=controller,
                        live_target_handle=live_target_handle,
                        obs=obs,
                        task=task,
                        gripper_context=gripper_context_arr,
                    )
                    teacher_truth_handoff = teacher_truth_provider.resolve_handoff_only(
                        current_gripper_pose=np.asarray(obs.gripper_pose, dtype=np.float32),
                        manager=refiner.manager,
                        controller=controller,
                        live_target_handle=live_target_handle,
                        obs=obs,
                        task=task,
                        gripper_context=gripper_context_arr,
                    )
                    teacher_truth_delta = getattr(
                        teacher_truth_result,
                        "motion_target_delta_local",
                        getattr(teacher_truth_result, "target_delta_local", None),
                    )
                    if teacher_truth_delta is not None:
                        (
                            teacher_truth_basin_distance,
                            teacher_truth_basin_xy,
                            teacher_truth_basin_z,
                            teacher_truth_basin_yaw,
                            teacher_truth_basin_tilt,
                        ) = compute_basin_metrics(teacher_truth_delta)
            else:
                current_delta_basin = None
                current_basin_distance = None
                current_basin_xy = None
                current_basin_z = None
                current_basin_yaw = None
                teacher_truth_result = None
                teacher_truth_handoff = None
                teacher_truth_basin_distance = None
                teacher_truth_basin_xy = None
                teacher_truth_basin_z = None
                teacher_truth_basin_yaw = None
                teacher_truth_basin_tilt = None
                teacher_truth_delta = None
            teacher_delta_for_dataset = (
                None
                if teacher_truth_delta is None
                else np.asarray(teacher_truth_delta, dtype=np.float32).reshape(-1)
            )
            if (
                refiner is not None
                and active_alignment_controller is not None
                and getattr(args, "record_teacher_truth_metrics", False)
            ):
                active_alignment_controller._runtime_teacher_current_delta_basin_target = (
                    None
                    if teacher_delta_for_dataset is None
                    else np.asarray(teacher_delta_for_dataset, dtype=np.float32).copy()
                )
            depth_proximity = StageAwareRefiner.compute_depth_proximity(depth_tensor_96)
            trace_entry = {
                "step": int(step_idx),
                "base_gripper_raw": float(base_delta_action[6]),
                "planner_chunk_world_6d": _jsonable_value(
                    np.asarray(base_delta_action[:6], dtype=np.float32)
                ),
                "planner_chunk_local_6d": _jsonable_value(
                    np.asarray(planner_base_action_local_raw, dtype=np.float32)
                ),
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
                "alignment_runtime_basin_distance": None
                if current_basin_distance is None
                else float(current_basin_distance),
                "alignment_runtime_basin_xy": None if current_basin_xy is None else float(current_basin_xy),
                "alignment_runtime_basin_z": None if current_basin_z is None else float(current_basin_z),
                "alignment_runtime_basin_yaw": None if current_basin_yaw is None else float(current_basin_yaw),
                "coarse2contact_phase": "COARSE",
                "planner_reaches_precontact": bool(
                    depth_proximity is not None
                    and np.isfinite(float(depth_proximity))
                    and float(depth_proximity) <= float(getattr(args, "coarse2contact_precontact_depth_threshold", 0.20))
                ),
                "planner_reaches_preinsert": bool(
                    depth_proximity is not None
                    and np.isfinite(float(depth_proximity))
                    and float(depth_proximity) <= float(getattr(args, "coarse2contact_contact_depth_threshold", 0.035))
                ),
                "visual_error_xy": None,
                "visual_error_z": None if depth_proximity is None else float(depth_proximity),
                "visual_error_yaw": None,
                "contact_state": "free",
                "force_reflex_reason": "none",
                "recovery_primitive": "none",
                "uses_privileged_target": False,
                "mp4_path": None,
            }
            if target_result is not None:
                if handoff_aux:
                    current_delta_for_shadow = np.asarray(
                        getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                        dtype=np.float32,
                    ).reshape(-1)
                    predicted_residual_local = np.asarray(
                        [
                            float(handoff_aux.get("pred_residual_dx", np.nan)),
                            float(handoff_aux.get("pred_residual_dy", np.nan)),
                            float(handoff_aux.get("pred_residual_dz", np.nan)),
                            float(handoff_aux.get("pred_residual_dyaw", np.nan)),
                        ],
                        dtype=np.float32,
                    )
                    release_thresholds = (handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}
                    shadow_audit = shadow_residual_audit(
                        current_delta_for_shadow,
                        predicted_residual_local,
                        xy_threshold=float(release_thresholds.get("xy_error", 0.0085)),
                        abs_z_threshold=float(release_thresholds.get("abs_z_error", 0.0035)),
                        yaw_threshold=float(release_thresholds.get("yaw_error", 0.1243404)),
                        yaw_symmetry_period=float((handoff_result or {}).get("yaw_symmetry_period", -1.0)),
                    )
                    handoff_aux.update(shadow_audit)
                trace_entry["handoff_ready_provider"] = bool((handoff_result or {}).get("handoff_ready", False))
                trace_entry["handoff_metrics_provider"] = dict((handoff_result or {}).get("handoff_metrics", {}) or {})
                trace_entry["handoff_metric_thresholds_provider"] = dict(
                    (handoff_result or {}).get("handoff_metric_thresholds", {}) or {}
                )
                trace_entry["handoff_release_metric_thresholds_provider"] = dict(
                    (handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}
                )
                trace_entry["handoff_optimization_metric_thresholds_provider"] = dict(
                    (handoff_result or {}).get("handoff_optimization_metric_thresholds", {}) or {}
                )
                trace_entry["motion_target_provider_source"] = str(getattr(target_result, "source", "none"))
                trace_entry["handoff_spec_name"] = str((handoff_result or {}).get("handoff_spec_name", "none"))
                trace_entry["handoff_target_role"] = str((handoff_result or {}).get("handoff_target_role", "none"))
                trace_entry["handoff_aux_provider"] = dict(handoff_aux)
                trace_entry["handoff_shadow_only"] = bool(handoff_shadow_only)
                trace_entry["handoff_ready_pred"] = bool(predicted_handoff_ready)
            if teacher_truth_provider is not None:
                trace_entry["teacher_truth_provider_source"] = (
                    "none" if teacher_truth_result is None else str(getattr(teacher_truth_result, "source", "none"))
                )
                trace_entry["privileged_current_pose_7d"] = np.asarray(obs.gripper_pose, dtype=np.float32)
                trace_entry["privileged_motion_target_pose_7d"] = np.asarray(
                    getattr(teacher_truth_result, "motion_target_pose_7d", np.full((7,), np.nan, dtype=np.float32)),
                    dtype=np.float32,
                )
                trace_entry["privileged_basin_center_pose_7d"] = np.asarray(
                    getattr(teacher_truth_result, "target_pose_7d", np.full((7,), np.nan, dtype=np.float32)),
                    dtype=np.float32,
                )
                trace_entry["privileged_current_delta_basin_target"] = np.asarray(
                    np.asarray(teacher_truth_delta, dtype=np.float32).reshape(-1)
                    if teacher_truth_delta is not None
                    else np.full((6,), np.nan, dtype=np.float32),
                    dtype=np.float32,
                )
                trace_entry["privileged_target_provider_source"] = np.asarray(
                    str(getattr(teacher_truth_result, "source", "none"))
                )
                trace_entry["privileged_target_provider_uses_privileged"] = np.asarray(
                    float(getattr(teacher_truth_result, "uses_privileged_target", False)),
                    dtype=np.float32,
                )
                trace_entry["teacher_truth_basin_distance"] = (
                    None if teacher_truth_basin_distance is None else float(teacher_truth_basin_distance)
                )
                trace_entry["teacher_truth_basin_xy"] = (
                    None if teacher_truth_basin_xy is None else float(teacher_truth_basin_xy)
                )
                trace_entry["teacher_truth_basin_z"] = (
                    None if teacher_truth_basin_z is None else float(teacher_truth_basin_z)
                )
                trace_entry["teacher_truth_basin_yaw"] = (
                    None if teacher_truth_basin_yaw is None else float(teacher_truth_basin_yaw)
                )
                trace_entry["teacher_truth_basin_tilt"] = (
                    None if teacher_truth_basin_tilt is None else float(teacher_truth_basin_tilt)
                )
                trace_entry["teacher_truth_handoff_ready"] = bool(
                    (teacher_truth_handoff or {}).get("handoff_ready", False)
                )
                trace_entry["teacher_truth_handoff_metrics"] = dict(
                    (teacher_truth_handoff or {}).get("handoff_metrics", {}) or {}
                )
                trace_entry["teacher_truth_handoff_spec_name"] = str(
                    (teacher_truth_handoff or {}).get("handoff_spec_name", "none")
                )
            wrist_valid_depth_ratio, wrist_depth_near_fraction, wrist_is_occluded, wrist_is_low_visibility = compute_wrist_visibility_stats(
                depth_tensor_96.detach().cpu().numpy().astype(np.float32) if depth_tensor_96 is not None else None
            )
            trace_entry["wrist_valid_depth_ratio"] = float(wrist_valid_depth_ratio)
            trace_entry["wrist_depth_near_fraction"] = float(wrist_depth_near_fraction)
            trace_entry["wrist_is_occluded"] = bool(wrist_is_occluded)
            trace_entry["wrist_is_low_visibility"] = bool(wrist_is_low_visibility)
            refiner_stats_snapshot = None
            oracle_collect_active = False
            oracle_collect_idx = -1
            oracle_collect_score = 0.0
            oracle_collect_current_dist = None
            oracle_collect_next_dists = None
            oracle_collect_scores = None
            oracle_collect_action_local = None
            oracle_collect_candidate_actions = None
            oracle_collect_debug_pred = None
            teacher_live_object_pose = safe_live_target_pose_7d(live_target_handle)
            teacher_close_ready_now = False
            teacher_alignment_ready_now = False
            teacher_verify_active_now = bool(teacher_close_hold_remaining > 0)
            teacher_retry_active_now = bool(teacher_retry_steps_remaining > 0)
            teacher_grasp_verified_now = False
            teacher_retry_required_now = False
            teacher_enable_orientation_rescue = False
            teacher_current_delta_for_control = None
            tc_close_verify_collect_enabled = bool(
                getattr(args, "alignment_tc_privileged_teacher_collect", False)
                and getattr(args, "alignment_tc_privileged_teacher_close_enabled", False)
                and getattr(args, "alignment_tc_teacher_close_verify", False)
            )
            if refiner is not None:
                step_kwargs = dict(
                    a_base_7d=delta_action,
                    step_idx=chunk_step,
                    force_reading=raw_force,
                    gripper_z=float(obs.gripper_pose[2]),
                    front_rgb=np.asarray(obs.front_rgb, dtype=np.uint8),
                    wrist_rgb=np.asarray(obs.wrist_rgb, dtype=np.uint8),
                    wrist_depth=depth_tensor_96,
                    ft_hist=force_hist,
                    proprio=proprio,
                )
                if isinstance(refiner, StageAwareRefiner):
                    step_kwargs["gripper_pose"] = obs.gripper_pose
                    step_kwargs["gripper_open"] = float(obs.gripper_open)
                    step_kwargs["future_gripper_actions"] = future_gripper_actions
                if (
                    isinstance(refiner, StageAwareRefiner)
                    and bool(getattr(args, "alignment_tc_privileged_teacher_collect", False))
                    and not bool(oracle_pregrasp_success)
                ):
                    label_privileged = build_depth_force_label_only_privileged_fields(
                        obs=obs,
                        stage_tracker=stage_tracker,
                        task_name=str(args.task_name),
                        live_target_handle=live_target_handle,
                        target_mode=str(getattr(args, "alignment_tc_teacher_target_mode", "commit")),
                    )
                    tc_teacher_delta_folded = np.asarray(
                        label_privileged.get(
                            "privileged_current_delta_basin_target_folded",
                            label_privileged.get(
                                "privileged_current_delta_basin_target",
                                np.full((6,), np.nan, dtype=np.float32),
                            ),
                        ),
                        dtype=np.float32,
                    ).reshape(-1)[:6]
                    tc_teacher_delta_raw = np.asarray(
                        label_privileged.get(
                            "privileged_current_delta_basin_target_raw",
                            tc_teacher_delta_folded,
                        ),
                        dtype=np.float32,
                    ).reshape(-1)[:6]
                    tc_teacher_target_pose = np.asarray(
                        label_privileged.get(
                            "privileged_motion_target_pose_7d",
                            np.full((7,), np.nan, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    )
                    tc_finite_delta = bool(tc_teacher_delta_raw.size >= 6 and np.all(np.isfinite(tc_teacher_delta_raw[:6])))
                    tc_xy = float(np.linalg.norm(tc_teacher_delta_raw[:2])) if tc_finite_delta else np.inf
                    tc_abs_z = float(abs(tc_teacher_delta_raw[2])) if tc_finite_delta else np.inf
                    tc_yaw_raw = float(abs(tc_teacher_delta_raw[5])) if tc_finite_delta else np.inf
                    tc_yaw_folded = float(abs(tc_teacher_delta_folded[5])) if tc_finite_delta else np.inf
                    tc_broad_gate = bool(
                        tc_finite_delta
                        and tc_xy <= float(getattr(args, "alignment_tc_teacher_broad_xy_threshold", 0.060))
                        and tc_abs_z <= float(getattr(args, "alignment_tc_teacher_broad_abs_z_threshold", 0.120))
                        and tc_yaw_folded <= float(getattr(args, "alignment_tc_teacher_broad_yaw_threshold", 0.60))
                    )
                    tc_gripper_open = bool(float(obs.gripper_open) >= 0.5)
                    tc_gate_open = bool(tc_broad_gate and tc_gripper_open)
                    trace_entry["alignment_tc_teacher_collect_enabled"] = True
                    trace_entry["alignment_tc_teacher_target_source"] = str(
                        np.asarray(label_privileged.get("privileged_target_provider_source", "none")).item()
                    )
                    trace_entry["alignment_tc_teacher_delta_local"] = tc_teacher_delta_folded.astype(np.float32).tolist()
                    trace_entry["alignment_tc_teacher_delta_local_raw"] = tc_teacher_delta_raw.astype(np.float32).tolist()
                    trace_entry["alignment_tc_teacher_delta_local_folded"] = tc_teacher_delta_folded.astype(np.float32).tolist()
                    trace_entry["alignment_tc_teacher_target_pose_7d"] = tc_teacher_target_pose.astype(np.float32).tolist()
                    trace_entry["alignment_tc_teacher_broad_gate"] = bool(tc_broad_gate)
                    trace_entry["alignment_tc_teacher_gate_open"] = bool(tc_gate_open)
                    trace_entry["alignment_tc_teacher_gate_reason"] = (
                        "open"
                        if tc_gate_open
                        else (
                            "missing_commit_delta"
                            if not tc_finite_delta
                            else "gripper_not_open"
                            if not tc_gripper_open
                            else "outside_broad_near"
                        )
                    )
                    trace_entry["alignment_tc_teacher_xy"] = float(tc_xy)
                    trace_entry["alignment_tc_teacher_abs_z"] = float(tc_abs_z)
                    trace_entry["alignment_tc_teacher_yaw"] = float(tc_yaw_folded)
                    trace_entry["alignment_tc_teacher_yaw_raw"] = float(tc_yaw_raw)
                    trace_entry["alignment_tc_teacher_yaw_folded"] = float(tc_yaw_folded)
                    trace_entry["alignment_tc_teacher_broad_xy_threshold"] = float(
                        getattr(args, "alignment_tc_teacher_broad_xy_threshold", 0.060)
                    )
                    trace_entry["alignment_tc_teacher_broad_abs_z_threshold"] = float(
                        getattr(args, "alignment_tc_teacher_broad_abs_z_threshold", 0.120)
                    )
                    trace_entry["alignment_tc_teacher_broad_yaw_threshold"] = float(
                        getattr(args, "alignment_tc_teacher_broad_yaw_threshold", 0.60)
                    )
                    tc_close_enabled = bool(getattr(args, "alignment_tc_privileged_teacher_close_enabled", False))
                    tc_close_verify_enabled = bool(getattr(args, "alignment_tc_teacher_close_verify", False))
                    tc_close_xy_threshold = float(getattr(args, "teacher_close_xy_threshold", 0.006))
                    tc_close_abs_z_threshold = float(getattr(args, "teacher_close_abs_z_threshold", 0.005))
                    tc_close_yaw_threshold = float(getattr(args, "teacher_close_yaw_threshold", 0.12))
                    grasp_basin_profile = getattr(args, "_grasp_basin_profile", None)
                    if grasp_basin_profile:
                        tc_close_xy_threshold_eff = float(tc_close_xy_threshold)
                        tc_close_abs_z_threshold_eff = float(tc_close_abs_z_threshold)
                        tc_close_yaw_threshold_eff = float(tc_close_yaw_threshold)
                    else:
                        tc_close_xy_threshold_eff = float(max(tc_close_xy_threshold, 0.0140))
                        tc_close_abs_z_threshold_eff = float(max(tc_close_abs_z_threshold, 0.0120))
                        tc_close_yaw_threshold_eff = float(max(tc_close_yaw_threshold, 0.060))
                    tc_close_yaw_ok = bool(tc_close_yaw_threshold < 0.0 or tc_yaw_folded <= tc_close_yaw_threshold)
                    tc_close_depth_ready = bool(
                        depth_proximity is not None
                        and np.isfinite(float(depth_proximity))
                        and float(depth_proximity) <= float(getattr(args, "teacher_close_contact_depth_threshold", 0.020))
                    )
                    tc_close_stage_ready = bool(
                        int(getattr(stage_tracker, "contact_state", 0)) >= int(ContactState.NEAR_CONTACT)
                        or int(getattr(stage_tracker, "contact_state", 0)) >= int(ContactState.IN_CONTACT)
                    )
                    tc_close_geometry_ready = bool(
                        tc_xy <= tc_close_xy_threshold_eff
                        and tc_abs_z <= tc_close_abs_z_threshold_eff
                        and tc_close_yaw_ok
                    )
                    tc_close_visibility_ok = bool(not wrist_is_low_visibility and not wrist_is_occluded)
                    tc_close_contact_ready = bool(
                        tc_close_geometry_ready and (tc_close_stage_ready or tc_close_depth_ready)
                    )
                    teacher_close_stage_contact_state = int(getattr(stage_tracker, "contact_state", 0))
                    teacher_close_refiner_contact_state = int(
                        refiner_stats_snapshot.get("contact_state_id", teacher_close_stage_contact_state)
                        if refiner_stats_snapshot is not None
                        else teacher_close_stage_contact_state
                    )
                    teacher_grasp_contract = compute_tc_grasp_expert_contract(
                        current_pose_7d=np.asarray(obs.gripper_pose, dtype=np.float32),
                        object_pose_7d=teacher_live_object_pose,
                        gripper_open=float(obs.gripper_open),
                        depth_proximity=depth_proximity,
                        stage_contact_state=teacher_close_stage_contact_state,
                        refiner_contact_state=teacher_close_refiner_contact_state,
                        close_xy_threshold=float(getattr(args, "teacher_grasp_xy_threshold", tc_close_xy_threshold)),
                        close_abs_z_threshold=float(getattr(args, "teacher_grasp_abs_z_threshold", tc_close_abs_z_threshold)),
                        close_yaw_threshold=float(getattr(args, "teacher_grasp_yaw_threshold", tc_close_yaw_threshold)),
                        close_contact_depth_threshold=float(
                            getattr(args, "teacher_close_contact_depth_threshold", 0.020)
                        ),
                        low_visibility=bool(wrist_is_low_visibility),
                        occluded=bool(wrist_is_occluded),
                        grasp_ready_threshold=float(getattr(args, "teacher_grasp_ready_threshold", 0.55)),
                        desired_object_delta_local_6d=(
                            np.asarray(grasp_basin_profile.get("close_object_to_gripper_delta_local_6d_median"), dtype=np.float32)
                            if isinstance(grasp_basin_profile, dict)
                            and grasp_basin_profile.get("close_object_to_gripper_delta_local_6d_median") is not None
                            else None
                        ),
                    )
                    teacher_gripper_finger_pose_7d = np.asarray(
                        teacher_grasp_contract["gripper_finger_pose_7d"],
                        dtype=np.float32,
                    ).reshape(7)
                    teacher_grasp_ready = bool(teacher_grasp_contract["grasp_ready"])
                    teacher_grasp_readiness_score = float(teacher_grasp_contract["grasp_readiness_score"])
                    teacher_object_in_finger_region = bool(teacher_grasp_contract["object_in_finger_region"])
                    teacher_finger_object_lateral_error = float(teacher_grasp_contract["finger_object_lateral_error"])
                    teacher_finger_object_height_overlap = float(teacher_grasp_contract["finger_object_height_overlap"])
                    teacher_finger_object_yaw_error = float(teacher_grasp_contract["finger_object_yaw_error"])
                    teacher_grasp_contact_ready = bool(teacher_grasp_contract["contact_ready"])
                    teacher_close_basin_source = str(getattr(args, "teacher_close_basin_source", "heuristic"))
                    teacher_close_xy_val = float(tc_xy) if tc_finite_delta else None
                    teacher_close_abs_z_val = float(tc_abs_z) if tc_finite_delta else None
                    teacher_close_yaw_raw_val = float(tc_yaw_raw) if tc_finite_delta else None
                    teacher_close_yaw_val = float(tc_yaw_folded) if tc_finite_delta else None
                    teacher_object_to_gripper_delta_local_6d = None
                    teacher_object_to_gripper_error_delta_local_6d = None
                    if teacher_live_object_pose is not None:
                        teacher_object_to_gripper_delta_local_6d = pose_delta_local_between(
                            np.asarray(obs.gripper_pose, dtype=np.float32),
                            np.asarray(teacher_live_object_pose, dtype=np.float32),
                        )
                    teacher_object_to_gripper_error_delta_local_6d = np.asarray(
                        teacher_grasp_contract.get(
                            "object_to_gripper_error_delta_local_6d",
                            np.full((6,), np.nan, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ).reshape(6)
                    if np.all(np.isfinite(teacher_object_to_gripper_error_delta_local_6d)):
                        teacher_close_xy_val = float(np.linalg.norm(teacher_object_to_gripper_error_delta_local_6d[:2]))
                        teacher_close_abs_z_val = float(abs(teacher_object_to_gripper_error_delta_local_6d[2]))
                        teacher_close_yaw_raw_val = float(abs(teacher_object_to_gripper_error_delta_local_6d[5]))
                        teacher_close_yaw_val = float(abs(teacher_object_to_gripper_error_delta_local_6d[5]))
                        tc_xy = teacher_close_xy_val
                        tc_abs_z = teacher_close_abs_z_val
                        tc_yaw_raw = teacher_close_yaw_raw_val
                        tc_yaw_folded = teacher_close_yaw_val
                    teacher_direct_control_delta_local_6d = (
                        teacher_object_to_gripper_error_delta_local_6d.astype(np.float32)
                        if teacher_object_to_gripper_error_delta_local_6d is not None
                        and np.all(np.isfinite(teacher_object_to_gripper_error_delta_local_6d))
                        else tc_teacher_delta_raw.astype(np.float32)
                    )
                    tc_close_contact_ready = bool(
                        tc_close_contact_ready and teacher_grasp_ready and teacher_object_in_finger_region
                    )
                    tc_close_ready_all = bool(
                        tc_close_enabled
                        and tc_finite_delta
                        and tc_broad_gate
                        and tc_gripper_open
                        and tc_close_contact_ready
                        and tc_close_geometry_ready
                        and tc_close_yaw_ok
                        and teacher_grasp_ready
                        and teacher_object_in_finger_region
                    )
                    tc_close_ready_now = bool(
                        tc_close_enabled
                        and tc_finite_delta
                        and tc_broad_gate
                        and tc_gripper_open
                        and tc_xy <= tc_close_xy_threshold
                        and tc_abs_z <= tc_close_abs_z_threshold
                        and tc_close_yaw_ok
                        and tc_close_contact_ready
                        and teacher_grasp_ready
                        and teacher_object_in_finger_region
                    )
                    if not tc_close_enabled:
                        tc_close_reason = "disabled"
                    elif not tc_finite_delta:
                        tc_close_reason = "missing_commit_delta"
                    elif not tc_gripper_open:
                        tc_close_reason = "gripper_not_open"
                    elif not tc_broad_gate:
                        tc_close_reason = "outside_broad_near"
                    elif tc_xy > tc_close_xy_threshold:
                        tc_close_reason = "xy"
                    elif tc_abs_z > tc_close_abs_z_threshold:
                        tc_close_reason = "z"
                    elif not tc_close_yaw_ok:
                        tc_close_reason = "yaw"
                    elif not tc_close_contact_ready:
                        tc_close_reason = (
                            "contact_geometry"
                            if not tc_close_geometry_ready
                            else "contact_stage"
                            if not tc_close_stage_ready
                            else "contact_depth"
                        )
                    elif not teacher_object_in_finger_region:
                        tc_close_reason = "finger_region"
                    elif not teacher_grasp_ready:
                        tc_close_reason = "grasp_region"
                    else:
                        tc_close_reason = "ok"
                    trace_entry["teacher_close_ready"] = bool(tc_close_ready_now)
                    trace_entry["teacher_close_ready_all"] = bool(tc_close_ready_all)
                    trace_entry["teacher_close_gate_reason"] = str(tc_close_reason)
                    trace_entry["teacher_close_failure_reason"] = str(tc_close_reason if tc_close_reason != "ok" else "none")
                    trace_entry["teacher_close_xy"] = float(tc_xy) if tc_finite_delta else None
                    trace_entry["teacher_close_abs_z"] = float(tc_abs_z) if tc_finite_delta else None
                    trace_entry["teacher_close_yaw"] = float(tc_yaw_folded) if tc_finite_delta else None
                    trace_entry["teacher_close_yaw_raw"] = float(tc_yaw_raw) if tc_finite_delta else None
                    trace_entry["teacher_close_yaw_folded"] = float(tc_yaw_folded) if tc_finite_delta else None
                    trace_entry["teacher_close_yaw_threshold"] = float(tc_close_yaw_threshold)
                    trace_entry["teacher_close_yaw_ok"] = bool(tc_close_yaw_ok)
                    trace_entry["teacher_close_contact_ready_by_depth"] = bool(tc_close_depth_ready)
                    trace_entry["teacher_close_contact_ready_by_stage"] = bool(tc_close_stage_ready)
                    trace_entry["teacher_close_contact_ready_by_geometry"] = bool(tc_close_geometry_ready)
                    trace_entry["teacher_close_contact_visibility_ok"] = bool(tc_close_visibility_ok)
                    trace_entry["teacher_close_basin_source"] = str(teacher_close_basin_source)
                    trace_entry["teacher_close_failure_reason"] = "none"
                    trace_entry["teacher_close_contact_confidence"] = float(
                        np.clip(
                            1.0
                            - (0.35 if wrist_is_occluded else 0.0)
                            - (0.20 if wrist_is_low_visibility else 0.0)
                            - (0.15 if not tc_close_stage_ready and not tc_close_depth_ready else 0.0),
                            0.0,
                            1.0,
                        )
                    )
                    trace_entry["teacher_gripper_finger_pose_7d"] = teacher_gripper_finger_pose_7d.tolist()
                    trace_entry["teacher_object_in_finger_region"] = bool(teacher_object_in_finger_region)
                    trace_entry["teacher_object_to_gripper_error_delta_local_6d"] = (
                        teacher_object_to_gripper_error_delta_local_6d.astype(np.float32).tolist()
                        if teacher_object_to_gripper_error_delta_local_6d is not None
                        and np.all(np.isfinite(teacher_object_to_gripper_error_delta_local_6d))
                        else np.full((6,), np.nan, dtype=np.float32).tolist()
                    )
                    trace_entry["teacher_desired_object_to_gripper_delta_local_6d"] = np.asarray(
                        teacher_grasp_contract.get(
                            "desired_object_to_gripper_delta_local_6d",
                            np.full((6,), np.nan, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ).reshape(6).tolist()
                    trace_entry["teacher_finger_object_lateral_error"] = float(teacher_finger_object_lateral_error)
                    trace_entry["teacher_finger_object_height_overlap"] = float(teacher_finger_object_height_overlap)
                    trace_entry["teacher_finger_object_yaw_error"] = float(teacher_finger_object_yaw_error)
                    trace_entry["teacher_grasp_contact_ready"] = bool(teacher_grasp_contact_ready)
                    trace_entry["teacher_grasp_ready"] = bool(teacher_grasp_ready)
                    trace_entry["teacher_grasp_readiness_score"] = float(teacher_grasp_readiness_score)
                    trace_entry["teacher_grasp_readiness_reason"] = str(teacher_grasp_contract["grasp_readiness_reason"])
                    teacher_yaw_imitation_enabled = bool(
                        getattr(args, "alignment_tc_teacher_yaw_imitation_enabled", True)
                    )
                    teacher_yaw_control_sign = float(
                        getattr(args, "alignment_tc_teacher_yaw_control_sign", 1.0)
                    )
                    if not teacher_yaw_imitation_enabled:
                        teacher_yaw_control_sign = 0.0
                    if (
                        tc_xy <= float(getattr(args, "alignment_tc_raw_micro_xy_threshold", 0.010))
                        and tc_abs_z <= float(getattr(args, "alignment_tc_raw_micro_abs_z_threshold", 0.030))
                        and tc_yaw_folded <= float(getattr(args, "alignment_tc_raw_micro_yaw_threshold", 0.18))
                    ):
                        tc_teacher_bucket = "micro_contact_refine"
                    elif (
                        tc_xy <= float(getattr(args, "alignment_tc_raw_near_xy_threshold", 0.030))
                        and tc_abs_z <= float(getattr(args, "alignment_tc_raw_near_abs_z_threshold", 0.070))
                        and tc_yaw_folded <= float(getattr(args, "alignment_tc_raw_near_yaw_threshold", 0.35))
                    ):
                        tc_teacher_bucket = "near_contact_refine"
                    else:
                        tc_teacher_bucket = "broad_near"
                    teacher_yaw_close_threshold = float(
                        tc_close_yaw_threshold_eff if tc_close_yaw_threshold_eff >= 0.0 else 0.04
                    )
                    if not tc_gate_open:
                        teacher_yaw_correct_latched = False
                    elif teacher_close_yaw_raw_val is not None and np.isfinite(float(teacher_close_yaw_raw_val)):
                        _yaw_val = float(teacher_close_yaw_raw_val)
                        if _yaw_val > max(teacher_yaw_close_threshold + 0.005, teacher_yaw_close_threshold * 1.10):
                            teacher_yaw_correct_latched = False
                        elif _yaw_val <= teacher_yaw_close_threshold:
                            teacher_yaw_correct_latched = True
                    teacher_yaw_correct_entry_ready = bool(
                        tc_gate_open
                        and teacher_close_yaw_raw_val is not None
                        and float(teacher_close_yaw_raw_val) > teacher_yaw_close_threshold
                        and float(obs.gripper_open) >= 0.5
                    )
                    teacher_yaw_correct_ready = bool(
                        teacher_yaw_correct_entry_ready and not bool(teacher_yaw_correct_latched)
                    )
                    teacher_descend_handoff_z_threshold = float(
                        max(tc_close_abs_z_threshold_eff * 30.0, 0.200)
                    )
                    teacher_enter_finger_region_ready = bool(
                        tc_gate_open
                        and
                        teacher_grasp_contact_ready
                        and not teacher_object_in_finger_region
                        and not teacher_yaw_correct_ready
                        and teacher_close_xy_val is not None
                        and teacher_close_xy_val <= max(tc_close_xy_threshold_eff * 80.0, 0.380)
                        and teacher_close_abs_z_val is not None
                        and teacher_close_abs_z_val <= teacher_descend_handoff_z_threshold
                    )
                    teacher_xy_correct_ready = bool(
                        tc_gate_open
                        and not bool(teacher_alignment_handoff_active)
                        and float(obs.gripper_open) >= 0.5
                        and not teacher_yaw_correct_ready
                        and not teacher_enter_finger_region_ready
                        and teacher_close_xy_val is not None
                        and teacher_close_xy_val > tc_close_xy_threshold_eff
                        and teacher_close_abs_z_val is not None
                        and teacher_close_abs_z_val <= teacher_descend_handoff_z_threshold
                        and tc_close_yaw_ok
                        and teacher_grasp_contact_ready
                    )
                    teacher_preclose_descend_ready = bool(
                        tc_gate_open
                        and
                        not bool(teacher_alignment_handoff_active)
                        and float(obs.gripper_open) >= 0.5
                        and teacher_object_in_finger_region
                        and teacher_close_xy_val is not None
                        and teacher_close_abs_z_val is not None
                        and teacher_close_xy_val <= max(tc_close_xy_threshold_eff * 1.5, 0.008)
                        and teacher_close_abs_z_val > tc_close_abs_z_threshold_eff
                        and tc_close_yaw_ok
                        and teacher_grasp_ready
                        and teacher_close_abs_z_val <= teacher_descend_handoff_z_threshold
                    )
                    trace_entry["teacher_yaw_correct_ready"] = bool(teacher_yaw_correct_ready)
                    trace_entry["teacher_yaw_correct_latched"] = bool(teacher_yaw_correct_latched)
                    trace_entry["teacher_enter_finger_region_ready"] = bool(teacher_enter_finger_region_ready)
                    trace_entry["teacher_xy_correct_ready"] = bool(teacher_xy_correct_ready)
                    teacher_expert_sequence_name = str(
                        select_tc_expert_sequence_name(
                            teacher_grasp_verified_now=False,
                            teacher_close_ready_all=bool(tc_close_ready_all),
                            teacher_close_contact_ready=bool(tc_close_contact_ready),
                            teacher_close_yaw_ok=bool(tc_close_yaw_ok),
                            teacher_yaw_correct_ready=bool(teacher_yaw_correct_ready),
                            teacher_enter_finger_region_ready=bool(teacher_enter_finger_region_ready),
                            teacher_xy_correct_ready=bool(teacher_xy_correct_ready),
                            teacher_preclose_descend_ready=bool(teacher_preclose_descend_ready),
                            teacher_alignment_ready_now=bool(tc_close_ready_all or tc_close_contact_ready),
                            teacher_verify_active_now=bool(teacher_close_hold_remaining > 0),
                            teacher_close_hold_remaining=int(teacher_close_hold_remaining),
                            teacher_close_attempted_this_cycle=bool(teacher_close_attempted_this_cycle),
                            teacher_yaw_imitation_enabled=bool(
                                getattr(args, "alignment_tc_teacher_yaw_imitation_enabled", True)
                            ),
                            teacher_grasp_ready=bool(teacher_grasp_ready),
                        )
                    )
                    trace_entry["expert_sequence_name"] = str(teacher_expert_sequence_name)
                    trace_entry["expert_sequence_score"] = float(teacher_grasp_readiness_score)
                    trace_entry["expert_sequence_verified"] = False
                    trace_entry["teacher_close_action"] = "continue_align"
                    trace_entry["teacher_grasp_verified_now"] = False
                    trace_entry["teacher_verify_active"] = bool(teacher_close_hold_remaining > 0)
                    trace_entry["teacher_retry_active"] = bool(teacher_retry_steps_remaining > 0)
                    trace_entry["close_failure_reason"] = "none"
                    teacher_motion_phase = str(trace_entry["expert_sequence_name"])
                    if teacher_motion_phase == "align_xy":
                        teacher_motion_phase = "align_xy_yaw"
                    trace_entry["teacher_tc_motion_phase"] = str(teacher_motion_phase)
                    if tc_close_enabled and teacher_retry_steps_remaining > 0:
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = np.asarray(
                            [0.0, 0.0, float(args.teacher_retry_lift), 0.0, 0.0, 0.0],
                            dtype=np.float32,
                        )
                        delta_action[6] = 1.0
                        teacher_retry_steps_remaining -= 1
                        teacher_retry_active_now = True
                        teacher_retry_required_now = True
                        teacher_alignment_handoff_active = False
                        trace_entry["teacher_gripper_override"] = "tc_retry_open"
                        trace_entry["teacher_close_action"] = "retry"
                        trace_entry["close_failure_reason"] = "retry"
                        teacher_close_attempted_this_cycle = False
                        teacher_verify_lift_streak = 0
                        oracle_collect_active = True
                        oracle_collect_action_local = np.asarray(
                            [0.0, 0.0, float(args.teacher_retry_lift), 0.0, 0.0, 0.0],
                            dtype=np.float32,
                        )
                        trace_entry["teacher_motion_controller"] = "tc_privileged_close_retry"
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif tc_close_enabled and teacher_realign_cooldown_remaining > 0:
                        delta_action = base_delta_action.copy()
                        delta_action[6] = 1.0
                        teacher_realign_cooldown_remaining -= 1
                        trace_entry["teacher_gripper_override"] = "tc_realign_cooldown_force_open"
                        trace_entry["teacher_close_action"] = "retry"
                        trace_entry["close_failure_reason"] = "realign_cooldown"
                        teacher_close_attempted_this_cycle = False
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif tc_close_enabled and tc_close_verify_enabled and teacher_close_hold_remaining > 0:
                        delta_action = base_delta_action.copy()
                        verify_hold_steps = max(int(getattr(args, "teacher_verify_hold_steps", 1)), 1)
                        verify_settle_steps = max(int(getattr(args, "teacher_verify_settle_steps", 4)), 0)
                        verify_elapsed_steps = max(verify_hold_steps - 1 - int(teacher_close_hold_remaining), 0)
                        if verify_elapsed_steps < verify_settle_steps:
                            delta_action[:6] = np.zeros((6,), dtype=np.float32)
                            trace_entry["teacher_close_settle_phase"] = "settle"
                            trace_entry["teacher_close_action"] = "settle"
                            teacher_motion_phase = "settle"
                        else:
                            verify_lift = max(float(args.teacher_retry_lift) * 0.5, 0.004)
                            delta_action[:6] = np.asarray([0.0, 0.0, verify_lift, 0.0, 0.0, 0.0], dtype=np.float32)
                            trace_entry["teacher_close_settle_phase"] = "lift_verify"
                            trace_entry["teacher_close_action"] = "verify_lift"
                            teacher_motion_phase = "lift_verify"
                        delta_action[6] = 0.0
                        teacher_close_hold_remaining -= 1
                        teacher_verify_active_now = True
                        trace_entry["teacher_gripper_override"] = "tc_verify_hold_close"
                        oracle_collect_active = True
                        oracle_collect_action_local = np.asarray(
                            delta_action[:6],
                            dtype=np.float32,
                        )
                        trace_entry["teacher_motion_controller"] = "tc_privileged_close_verify"
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif tc_close_ready_now:
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = np.zeros((6,), dtype=np.float32)
                        delta_action[6] = 0.0
                        teacher_close_hold_remaining = (
                            max(int(args.teacher_verify_hold_steps) - 1, 0)
                            if tc_close_verify_enabled
                            else 0
                        )
                        oracle_pregrasp_close_count += 1
                        if teacher_close_attempt_step is None:
                            teacher_close_attempt_step = int(step_idx)
                        teacher_close_attempted_this_cycle = True
                        teacher_verify_lift_streak = 0
                        teacher_last_close_object_z = (
                            None if teacher_live_object_pose is None else float(teacher_live_object_pose[2])
                        )
                        if teacher_live_object_pose is not None:
                            teacher_last_close_object_pose = np.asarray(teacher_live_object_pose, dtype=np.float32).copy()
                        else:
                            low_dim_pose = safe_task_low_dim_pose_7d(obs)
                            teacher_last_close_object_pose = (
                                None if low_dim_pose is None else np.asarray(low_dim_pose, dtype=np.float32).copy()
                            )
                            if teacher_last_close_object_z is None and low_dim_pose is not None:
                                teacher_last_close_object_z = float(low_dim_pose[2])
                        teacher_last_close_gripper_pose = np.asarray(obs.gripper_pose, dtype=np.float32).copy()
                        teacher_verify_active_now = bool(tc_close_verify_enabled)
                        if teacher_live_object_pose is not None:
                            teacher_object_to_gripper_delta_local_6d = pose_delta_local_between(
                                np.asarray(obs.gripper_pose, dtype=np.float32),
                                np.asarray(teacher_live_object_pose, dtype=np.float32),
                            )
                        elif teacher_last_close_object_pose is not None:
                            teacher_object_to_gripper_delta_local_6d = pose_delta_local_between(
                                np.asarray(obs.gripper_pose, dtype=np.float32),
                                np.asarray(teacher_last_close_object_pose, dtype=np.float32),
                            )
                        else:
                            teacher_object_to_gripper_delta_local_6d = np.full((6,), np.nan, dtype=np.float32)
                        demo_basin_distance, _, _, _, _ = compute_basin_metrics(
                            teacher_object_to_gripper_delta_local_6d,
                            r_xy=float(tc_close_xy_threshold),
                            r_z=float(tc_close_abs_z_threshold),
                            r_yaw=float(tc_close_yaw_threshold if tc_close_yaw_threshold > 0.0 else 0.05),
                        )
                        trace_entry["teacher_attached_after_close"] = bool(False)
                        trace_entry["teacher_grasped_object_count"] = 0
                        trace_entry["teacher_grasped_target_handle_match"] = bool(False)
                        trace_entry["teacher_object_to_gripper_delta_local_6d"] = teacher_object_to_gripper_delta_local_6d.tolist()
                        trace_entry["teacher_demo_basin_distance"] = float(demo_basin_distance)
                        trace_entry["teacher_gripper_override"] = "tc_teacher_close"
                        trace_entry["teacher_close_action"] = "close_now"
                        oracle_collect_active = True
                        oracle_collect_action_local = np.zeros((6,), dtype=np.float32)
                        trace_entry["teacher_motion_controller"] = "tc_privileged_close_now"
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_expert_sequence_name == "yaw_correct":
                        stage_action_local = _build_tc_hard_stage_action(
                            phase="yaw_correct",
                            current_delta_local_6d=teacher_direct_control_delta_local_6d,
                            grasp_delta_local_6d=teacher_object_to_gripper_delta_local_6d,
                            yaw_raw_val=teacher_close_yaw_raw_val,
                            yaw_control_sign=teacher_yaw_control_sign,
                            bucket=tc_teacher_bucket,
                            args=args,
                        )
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = local_delta_to_world(
                            stage_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action[6] = 1.0
                        oracle_collect_active = True
                        oracle_collect_idx = -1
                        oracle_collect_score = 0.0
                        oracle_collect_action_local = stage_action_local.astype(np.float32)
                        oracle_collect_current_dist = float(current_basin_distance if current_basin_distance is not None else 0.0)
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [0.0]
                        oracle_collect_candidate_actions = stage_action_local.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        trace_entry["teacher_gripper_override"] = "hard_stage_yaw_correct"
                        trace_entry["teacher_close_action"] = "yaw_correct"
                        trace_entry["teacher_motion_controller"] = "tc_privileged_yaw_correct"
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = -1
                        trace_entry["oracle_score"] = 0.0
                        trace_entry["oracle_action_local"] = stage_action_local.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_expert_sequence_name == "enter_finger_region":
                        grasp_delta_for_control = (
                            np.asarray(teacher_object_to_gripper_delta_local_6d, dtype=np.float32)
                            if teacher_object_to_gripper_delta_local_6d is not None
                            else np.asarray(teacher_direct_control_delta_local_6d, dtype=np.float32)
                        )
                        stage_action_local = _build_tc_hard_stage_action(
                            phase="enter_finger_region",
                            current_delta_local_6d=teacher_direct_control_delta_local_6d,
                            grasp_delta_local_6d=grasp_delta_for_control,
                            yaw_raw_val=teacher_close_yaw_raw_val,
                            yaw_control_sign=teacher_yaw_control_sign,
                            bucket=tc_teacher_bucket,
                            args=args,
                        )
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = local_delta_to_world(
                            stage_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action[6] = 1.0
                        oracle_collect_active = True
                        oracle_collect_idx = -1
                        oracle_collect_score = 0.0
                        oracle_collect_action_local = stage_action_local.astype(np.float32)
                        oracle_collect_current_dist = float(current_basin_distance if current_basin_distance is not None else 0.0)
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [0.0]
                        oracle_collect_candidate_actions = stage_action_local.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        trace_entry["teacher_gripper_override"] = "hard_stage_enter_finger_region"
                        trace_entry["teacher_close_action"] = "enter_finger_region"
                        trace_entry["teacher_motion_controller"] = "tc_privileged_enter_finger_region"
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = -1
                        trace_entry["oracle_score"] = 0.0
                        trace_entry["oracle_action_local"] = stage_action_local.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_expert_sequence_name == "xy_correct":
                        stage_action_local = _build_tc_hard_stage_action(
                            phase="xy_correct",
                            current_delta_local_6d=teacher_direct_control_delta_local_6d,
                            grasp_delta_local_6d=teacher_object_to_gripper_delta_local_6d,
                            yaw_raw_val=teacher_close_yaw_raw_val,
                            yaw_control_sign=teacher_yaw_control_sign,
                            bucket=tc_teacher_bucket,
                            args=args,
                        )
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = local_delta_to_world(
                            stage_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action[6] = 1.0
                        oracle_collect_active = True
                        oracle_collect_idx = -1
                        oracle_collect_score = 0.0
                        oracle_collect_action_local = stage_action_local.astype(np.float32)
                        oracle_collect_current_dist = float(current_basin_distance if current_basin_distance is not None else 0.0)
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [0.0]
                        oracle_collect_candidate_actions = stage_action_local.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        trace_entry["teacher_gripper_override"] = "hard_stage_xy_correct"
                        trace_entry["teacher_close_action"] = "xy_correct"
                        trace_entry["teacher_motion_controller"] = "tc_privileged_xy_correct"
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = -1
                        trace_entry["oracle_score"] = 0.0
                        trace_entry["oracle_action_local"] = stage_action_local.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_expert_sequence_name == "xy_correct":
                        stage_action_local = _build_tc_hard_stage_action(
                            phase="xy_correct",
                            current_delta_local_6d=current_delta_basin,
                            grasp_delta_local_6d=teacher_object_to_gripper_delta_local_6d,
                            yaw_raw_val=teacher_close_yaw_raw_val,
                            yaw_control_sign=teacher_yaw_control_sign,
                            bucket=tc_teacher_bucket,
                            args=args,
                        )
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = local_delta_to_world(
                            stage_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action[6] = 1.0
                        oracle_collect_active = True
                        oracle_collect_idx = -1
                        oracle_collect_score = 0.0
                        oracle_collect_action_local = stage_action_local.astype(np.float32)
                        oracle_collect_current_dist = float(current_basin_distance if current_basin_distance is not None else 0.0)
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [0.0]
                        oracle_collect_candidate_actions = stage_action_local.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        trace_entry["teacher_gripper_override"] = "hard_stage_xy_correct"
                        trace_entry["teacher_close_action"] = "xy_correct"
                        trace_entry["teacher_motion_controller"] = "tc_privileged_xy_correct"
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = -1
                        trace_entry["oracle_score"] = 0.0
                        trace_entry["oracle_action_local"] = stage_action_local.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_expert_sequence_name == "descend_z":
                        stage_action_local = _build_tc_hard_stage_action(
                            phase="descend_z",
                            current_delta_local_6d=teacher_direct_control_delta_local_6d,
                            grasp_delta_local_6d=teacher_direct_control_delta_local_6d,
                            yaw_raw_val=teacher_close_yaw_raw_val,
                            yaw_control_sign=teacher_yaw_control_sign,
                            bucket=tc_teacher_bucket,
                            args=args,
                        )
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = local_delta_to_world(
                            stage_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action[6] = 1.0
                        oracle_collect_active = True
                        oracle_collect_idx = -1
                        oracle_collect_score = 0.0
                        oracle_collect_action_local = stage_action_local.astype(np.float32)
                        oracle_collect_current_dist = float(current_basin_distance if current_basin_distance is not None else 0.0)
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [0.0]
                        oracle_collect_candidate_actions = stage_action_local.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        trace_entry["teacher_gripper_override"] = "hard_stage_descend_z"
                        trace_entry["teacher_close_action"] = "descend_z"
                        trace_entry["teacher_motion_controller"] = "tc_privileged_descend_z"
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = -1
                        trace_entry["oracle_score"] = 0.0
                        trace_entry["oracle_action_local"] = stage_action_local.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif tc_gate_open:
                        teacher_current_delta_for_control = tc_teacher_delta_raw.copy()
                        if (
                            tc_xy <= float(getattr(args, "alignment_tc_raw_micro_xy_threshold", 0.010))
                            and tc_abs_z <= float(getattr(args, "alignment_tc_raw_micro_abs_z_threshold", 0.030))
                            and tc_yaw_folded <= float(getattr(args, "alignment_tc_raw_micro_yaw_threshold", 0.18))
                        ):
                            tc_teacher_bucket = "micro_contact_refine"
                        elif (
                            tc_xy <= float(getattr(args, "alignment_tc_raw_near_xy_threshold", 0.030))
                            and tc_abs_z <= float(getattr(args, "alignment_tc_raw_near_abs_z_threshold", 0.070))
                            and tc_yaw_folded <= float(getattr(args, "alignment_tc_raw_near_yaw_threshold", 0.35))
                        ):
                            tc_teacher_bucket = "near_contact_refine"
                        else:
                            tc_teacher_bucket = "broad_near"
                        teacher_selection = select_tc_privileged_teacher_action(
                            current_pose_7d=np.asarray(obs.gripper_pose, dtype=np.float32),
                            target_pose_7d=tc_teacher_target_pose,
                            current_delta_local_6d=tc_teacher_delta_raw,
                            planner_action_local_6d=planner_base_action_local_raw,
                            object_pose_7d=np.asarray(teacher_live_object_pose, dtype=np.float32)
                            if teacher_live_object_pose is not None
                            else None,
                            safety=refiner.safety,
                            bucket=tc_teacher_bucket,
                            yaw_sign=teacher_yaw_control_sign,
                            yaw_imitation_enabled=teacher_yaw_imitation_enabled,
                            phase=teacher_motion_phase,
                            grasp_readiness_score=float(teacher_grasp_readiness_score),
                            args=args,
                        )
                        p_action = np.asarray(teacher_selection["action"], dtype=np.float32)
                        oracle_collect_action_world = local_delta_to_world(
                            p_action,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = oracle_collect_action_world
                        delta_action[6] = (
                            1.0
                            if bool(getattr(args, "alignment_tc_teacher_force_open_until_close_ready", True))
                            else float(base_delta_action[6])
                        )
                        oracle_collect_active = True
                        oracle_collect_idx = int(teacher_selection["candidate_index"])
                        oracle_collect_score = float(-teacher_selection["score"])
                        oracle_collect_action_local = p_action.astype(np.float32)
                        oracle_collect_current_dist = float(np.linalg.norm(tc_teacher_delta_raw[:3]))
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [float(-teacher_selection["score"])]
                        oracle_collect_candidate_actions = p_action.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        if float(np.linalg.norm(p_action)) < 1e-8:
                            oracle_executed_noop_count += 1
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = int(teacher_selection["candidate_index"])
                        trace_entry["oracle_score"] = float(-teacher_selection["score"])
                        trace_entry["oracle_action_local"] = p_action.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        trace_entry["teacher_motion_controller"] = "tc_privileged_commit_short_horizon"
                        trace_entry["alignment_tc_teacher_bucket"] = str(tc_teacher_bucket)
                        trace_entry["alignment_tc_teacher_candidate_name"] = str(teacher_selection["candidate_name"])
                        trace_entry["alignment_tc_teacher_candidate_count"] = int(teacher_selection["candidate_count"])
                        trace_entry["alignment_tc_teacher_candidate_scale"] = float(teacher_selection["scale"])
                        trace_entry["alignment_tc_teacher_candidate_variant"] = str(teacher_selection["variant"])
                        trace_entry["alignment_tc_teacher_candidate_score"] = float(teacher_selection["score"])
                        trace_entry["alignment_tc_teacher_candidate_final_xy"] = float(teacher_selection["final"][0])
                        trace_entry["alignment_tc_teacher_candidate_final_z"] = float(teacher_selection["final"][1])
                        trace_entry["alignment_tc_teacher_candidate_final_yaw"] = float(teacher_selection["final"][2])
                        trace_entry["alignment_tc_teacher_candidate_improve_count"] = int(teacher_selection["improve_count"])
                        trace_entry["alignment_tc_teacher_already_close"] = bool(teacher_selection["already_close"])
                        trace_entry["alignment_tc_teacher_pos_norm"] = float(teacher_selection["pos_norm"])
                        trace_entry["alignment_tc_teacher_valid_candidate_count"] = int(
                            teacher_selection["valid_candidate_count"]
                        )
                        trace_entry["alignment_tc_teacher_selected_from_valid"] = bool(
                            teacher_selection["selected_from_valid"]
                        )
                        trace_entry["alignment_tc_teacher_selected_constraint_ok"] = bool(
                            teacher_selection["selected_constraint_ok"]
                        )
                        trace_entry["alignment_tc_teacher_constraint_reason"] = str(
                            teacher_selection["constraint_reason"]
                        )
                        trace_entry["alignment_tc_teacher_workspace_delta_violation_pred"] = float(
                            teacher_selection["workspace_delta_violation"]
                        )
                        trace_entry["alignment_tc_teacher_max_pos"] = float(teacher_selection["max_pos"])
                        trace_entry["alignment_tc_teacher_max_yaw"] = float(teacher_selection["max_yaw"])
                        trace_entry["teacher_gripper_override"] = (
                            "tc_privileged_force_open"
                            if bool(getattr(args, "alignment_tc_teacher_force_open_until_close_ready", True))
                            else "tc_privileged_motion_passthrough_gripper"
                        )
                        trace_entry["teacher_smoothed_action_local"] = p_action.astype(np.float32).tolist()
                        trace_entry["planner_base_action_local_raw"] = planner_base_action_local_raw.tolist()
                        trace_entry["residual_label_local"] = (
                            p_action - planner_base_action_local_raw
                        ).astype(np.float32).tolist()
                        trace_entry["alignment_tc_teacher_action_local"] = p_action.astype(np.float32).tolist()
                        trace_entry["alignment_tc_teacher_action_world"] = oracle_collect_action_world.astype(np.float32).tolist()
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    else:
                        delta_action = refiner.step(**step_kwargs)
                        refiner_stats_snapshot = refiner.get_stats()
                elif (
                    isinstance(refiner, StageAwareRefiner)
                    and (
                        bool(getattr(args, "oracle_executed_pregrasp_collect", False))
                        or bool(getattr(args, "oracle_executed_align_collect", False))
                    )
                    and runtime_target_pose is not None
                    and anchor_pose is not None
                    and int(refiner.manager.phase) == int(StagePhase.ALIGN)
                    and not bool(oracle_pregrasp_success)
                ):
                    teacher_force_open_in_align = bool(getattr(args, "oracle_executed_pregrasp_collect", False))
                    controller = _active_alignment_like_controller(refiner)
                    oracle_gate = refiner._alignment_gate_decision(
                        depth_proximity,
                        float(obs.gripper_open),
                        a_base_7d=base_delta_action,
                        future_gripper_actions=future_gripper_actions,
                        controller=controller,
                        step_idx=chunk_step,
                    )
                    teacher_close_xy_val = None if current_basin_xy is None else float(current_basin_xy)
                    teacher_close_abs_z_val = None if current_basin_z is None else abs(float(current_basin_z))
                    teacher_close_dist_val = None if current_basin_distance is None else float(current_basin_distance)
                    teacher_pregrasp_target = getattr(controller, "_runtime_pregrasp_target_pose_7d", None)
                    teacher_commit_target = getattr(controller, "_runtime_grasp_commit_target_pose_7d", None)
                    teacher_use_commit_target_now = bool(getattr(controller, "_runtime_use_commit_target", False))
                    # Until we have a demo-derived grasp orientation, pitch/roll/yaw
                    # orientation targets are uncalibrated. Keep the fixed 41-way
                    # candidate table for schema compatibility, but mask runtime
                    # teacher control back to the v6-style XYZ + minimal yaw base set.
                    teacher_orientation_rescue_enabled = False
                    teacher_active_target = np.asarray(runtime_target_pose, dtype=np.float32)
                    teacher_active_delta = None if runtime_target_pose is None else pose_delta_local_between(
                        np.asarray(obs.gripper_pose, dtype=np.float32),
                        teacher_active_target,
                    )
                    teacher_commit_delta = None
                    teacher_commit_xy_val = None
                    teacher_commit_abs_z_val = None
                    teacher_commit_yaw_val = None
                    if teacher_commit_target is not None:
                        teacher_commit_delta = pose_delta_local_between(
                            np.asarray(obs.gripper_pose, dtype=np.float32),
                            np.asarray(teacher_commit_target, dtype=np.float32),
                        )
                        teacher_commit_xy_val = float(np.linalg.norm(teacher_commit_delta[:2]))
                        teacher_commit_abs_z_val = float(abs(teacher_commit_delta[2]))
                        teacher_commit_yaw_val = float(abs(teacher_commit_delta[5]))
                    teacher_orientation_error_deg = 0.0 if teacher_active_delta is None else float(
                        np.degrees(np.linalg.norm(np.asarray(teacher_active_delta[3:5], dtype=np.float32)))
                    )
                    teacher_enable_orientation_rescue = False
                    teacher_candidate_actions_all = np.asarray(
                        getattr(controller, "_teacher_candidate_actions_local", controller._candidate_actions_local.detach().cpu().numpy()),
                        dtype=np.float32,
                    )
                    teacher_candidate_group_index_all = np.asarray(
                        getattr(controller, "_teacher_candidate_group_index", controller._candidate_group_index.detach().cpu().numpy()),
                        dtype=np.int64,
                    )
                    teacher_candidate_kind = np.asarray(
                        getattr(controller, "_teacher_candidate_kind", np.asarray(["base"] * teacher_candidate_actions_all.shape[0])),
                    )
                    teacher_candidate_mask = build_teacher_candidate_mask(
                        teacher_candidate_kind,
                        enable_orientation_rescue=teacher_enable_orientation_rescue,
                    )
                    controller._runtime_candidate_mask = teacher_candidate_mask.astype(np.float32)
                    close_xy_threshold = min(
                        float(getattr(target_result, "close_xy_threshold", getattr(args, "teacher_close_xy_threshold", 0.008))),
                        float(getattr(args, "teacher_close_xy_threshold", 0.008)),
                    )
                    close_abs_z_threshold = min(
                        float(getattr(target_result, "close_abs_z_threshold", getattr(args, "teacher_close_abs_z_threshold", 0.008))),
                        float(getattr(args, "teacher_close_abs_z_threshold", 0.008)),
                    )
                    close_yaw_threshold = min(
                        float(getattr(target_result, "close_yaw_threshold", getattr(args, "teacher_close_yaw_threshold", 0.12))),
                        float(getattr(args, "teacher_close_yaw_threshold", 0.12)),
                    )
                    teacher_yaw_symmetry_period = float(getattr(target_result, "yaw_symmetry_period", -1.0))
                    teacher_current_delta_for_control = None
                    if current_delta_basin is not None:
                        teacher_current_delta_for_control = apply_yaw_symmetry_to_delta(
                            np.asarray(current_delta_basin, dtype=np.float32),
                            teacher_yaw_symmetry_period,
                        )
                    teacher_close_yaw_raw_val = (
                        None if current_delta_basin is None else abs(float(np.asarray(current_delta_basin, dtype=np.float32)[5]))
                    )
                    teacher_close_yaw_val = (
                        None
                        if teacher_current_delta_for_control is None
                        else abs(float(np.asarray(teacher_current_delta_for_control, dtype=np.float32)[5]))
                    )
                    teacher_close_yaw_ok = bool(
                        close_yaw_threshold < 0.0
                        or (
                            teacher_close_yaw_val is not None
                            and teacher_close_yaw_val <= close_yaw_threshold
                        )
                    )
                    teacher_close_contract = compute_tc_close_contact_contract(
                        xy_val=teacher_close_xy_val,
                        abs_z_val=teacher_close_abs_z_val,
                        yaw_val=teacher_close_yaw_val,
                        depth_proximity=depth_proximity,
                        stage_contact_state=teacher_close_stage_contact_state,
                        refiner_contact_state=teacher_close_refiner_contact_state,
                        close_xy_threshold=close_xy_threshold,
                        close_abs_z_threshold=close_abs_z_threshold,
                        close_yaw_threshold=close_yaw_threshold,
                        close_contact_depth_threshold=float(
                            getattr(args, "teacher_close_contact_depth_threshold", 0.020)
                        ),
                        low_visibility=bool(wrist_is_low_visibility),
                        occluded=bool(wrist_is_occluded),
                    )
                    teacher_close_contact_ready = bool(teacher_close_contract["contact_ready"])
                    teacher_close_contact_ready_by_depth = bool(teacher_close_contract["depth_ready"])
                    teacher_close_contact_ready_by_stage = bool(teacher_close_contract["stage_ready"])
                    teacher_close_contact_ready_by_geometry = bool(teacher_close_contract["geometry_ready"])
                    teacher_close_contact_visibility_ok = bool(teacher_close_contract["visibility_ok"])
                    teacher_close_contact_confidence = float(teacher_close_contract["confidence"])
                    trace_entry["teacher_gripper_finger_pose_7d"] = teacher_gripper_finger_pose_7d.astype(np.float32).tolist()
                    trace_entry["teacher_object_in_finger_region"] = bool(teacher_object_in_finger_region)
                    trace_entry["teacher_finger_object_lateral_error"] = float(teacher_finger_object_lateral_error)
                    trace_entry["teacher_finger_object_height_overlap"] = float(teacher_finger_object_height_overlap)
                    trace_entry["teacher_finger_object_yaw_error"] = float(teacher_finger_object_yaw_error)
                    trace_entry["teacher_grasp_contact_ready"] = bool(teacher_grasp_contact_ready)
                    trace_entry["teacher_grasp_ready"] = bool(teacher_grasp_ready)
                    trace_entry["teacher_grasp_readiness_score"] = float(teacher_grasp_readiness_score)
                    trace_entry["teacher_grasp_readiness_reason"] = str(teacher_grasp_contract["grasp_readiness_reason"])
                    trace_entry["teacher_yaw_imitation_enabled"] = bool(teacher_yaw_imitation_enabled)
                    teacher_yaw_close_threshold = float(
                        close_yaw_threshold if close_yaw_threshold >= 0.0 else 0.04
                    )
                    if not tc_gate_open:
                        teacher_yaw_correct_latched = False
                    elif teacher_close_yaw_raw_val is not None and np.isfinite(float(teacher_close_yaw_raw_val)):
                        _yaw_val = float(teacher_close_yaw_raw_val)
                        if _yaw_val > max(teacher_yaw_close_threshold + 0.005, teacher_yaw_close_threshold * 1.10):
                            teacher_yaw_correct_latched = False
                        elif _yaw_val <= teacher_yaw_close_threshold:
                            teacher_yaw_correct_latched = True
                    teacher_yaw_correct_entry_ready = bool(
                        tc_gate_open
                        and teacher_close_yaw_raw_val is not None
                        and float(teacher_close_yaw_raw_val) > teacher_yaw_close_threshold
                        and float(obs.gripper_open) >= 0.5
                    )
                    teacher_yaw_correct_ready = bool(
                        teacher_yaw_correct_entry_ready and not bool(teacher_yaw_correct_latched)
                    )
                    teacher_descend_handoff_z_threshold = float(
                        max(tc_close_abs_z_threshold_eff * 30.0, 0.200)
                    )
                    teacher_enter_finger_region_ready = bool(
                        tc_gate_open
                        and
                        teacher_grasp_contact_ready
                        and not teacher_object_in_finger_region
                        and not teacher_yaw_correct_ready
                        and teacher_close_xy_val is not None
                        and teacher_close_xy_val <= max(tc_close_xy_threshold_eff * 80.0, 0.380)
                        and teacher_close_abs_z_val is not None
                        and teacher_close_abs_z_val <= teacher_descend_handoff_z_threshold
                    )
                    teacher_xy_correct_ready = bool(
                        tc_gate_open
                        and not bool(teacher_alignment_handoff_active)
                        and float(obs.gripper_open) >= 0.5
                        and not teacher_yaw_correct_ready
                        and not teacher_enter_finger_region_ready
                        and teacher_close_xy_val is not None
                        and teacher_close_xy_val > tc_close_xy_threshold_eff
                        and teacher_close_abs_z_val is not None
                        and teacher_close_abs_z_val <= teacher_descend_handoff_z_threshold
                        and teacher_close_yaw_ok
                        and teacher_grasp_contact_ready
                    )
                    trace_entry["teacher_yaw_correct_ready"] = bool(teacher_yaw_correct_ready)
                    trace_entry["teacher_yaw_correct_latched"] = bool(teacher_yaw_correct_latched)
                    trace_entry["teacher_enter_finger_region_ready"] = bool(teacher_enter_finger_region_ready)
                    trace_entry["teacher_xy_correct_ready"] = bool(teacher_xy_correct_ready)
                    planner_close_intent_now = bool(oracle_gate.get("planner_close_intent", False))
                    if (
                        planner_close_intent_now
                        and float(obs.gripper_open) >= 0.5
                        and int(teacher_planner_verify_remaining) == 0
                        and teacher_last_close_object_z is None
                    ):
                        teacher_planner_close_latch_remaining = max(
                            int(getattr(args, "teacher_planner_close_latch_steps", 48)),
                            0,
                        )
                    elif teacher_planner_close_latch_remaining > 0:
                        teacher_planner_close_latch_remaining -= 1
                    teacher_alignment_ready_now = bool(
                        current_delta_basin is not None
                        and teacher_close_xy_val is not None
                        and teacher_close_abs_z_val is not None
                        and teacher_close_xy_val <= max(close_xy_threshold * 0.70, 0.0100)
                        and teacher_close_abs_z_val <= max(close_abs_z_threshold * 0.60, 0.0040)
                        and teacher_close_yaw_ok
                        and teacher_close_contact_ready
                        and teacher_grasp_ready
                    )
                    if teacher_alignment_ready_now:
                        teacher_alignment_handoff_active = bool(getattr(args, "teacher_planner_close_handoff", True))
                    elif (
                        bool(teacher_alignment_handoff_active)
                        and teacher_last_close_object_z is None
                        and int(teacher_planner_verify_remaining) == 0
                        and teacher_close_xy_val is not None
                        and teacher_close_abs_z_val is not None
                        and (
                            teacher_close_xy_val > float(getattr(args, "teacher_handoff_revoke_xy_threshold", 0.012))
                            or teacher_close_abs_z_val > float(getattr(args, "teacher_handoff_revoke_abs_z_threshold", 0.025))
                            or (
                                close_yaw_threshold >= 0.0
                                and teacher_close_yaw_val is not None
                                and teacher_close_yaw_val > max(close_yaw_threshold * 1.5, close_yaw_threshold + 0.05)
                            )
                        )
                    ):
                        teacher_alignment_handoff_active = False
                        teacher_smooth_action_local = None
                        teacher_last_candidate_idx = None
                        teacher_candidate_hold_remaining = 0
                        teacher_planner_close_latch_remaining = 0
                    if teacher_realign_cooldown_remaining > 0:
                        teacher_alignment_handoff_active = False
                    planner_assisted_close_ready = bool(
                        planner_close_intent_now
                        and teacher_close_xy_val is not None
                        and teacher_close_abs_z_val is not None
                        and teacher_close_xy_val <= max(tc_close_xy_threshold_eff, 0.010)
                        and teacher_close_abs_z_val <= max(tc_close_abs_z_threshold_eff, 0.012)
                        and teacher_close_yaw_ok
                        and float(depth_proximity) <= 0.040
                    )
                    teacher_preclose_descend_ready = bool(
                        tc_gate_open
                        and
                        not bool(teacher_alignment_handoff_active)
                        and float(obs.gripper_open) >= 0.5
                        and teacher_close_xy_val is not None
                        and teacher_close_abs_z_val is not None
                        and teacher_close_xy_val <= max(tc_close_xy_threshold_eff * 1.5, 0.008)
                        and teacher_close_abs_z_val > tc_close_abs_z_threshold_eff
                        and teacher_close_yaw_ok
                        and teacher_grasp_ready
                        and teacher_close_abs_z_val <= teacher_descend_handoff_z_threshold
                    )
                    teacher_expert_sequence_name = select_tc_expert_sequence_name(
                        teacher_grasp_verified_now=False,
                        teacher_close_ready_all=bool(teacher_close_ready_all),
                        teacher_close_contact_ready=bool(teacher_close_contact_ready),
                        teacher_close_yaw_ok=bool(teacher_close_yaw_ok),
                        teacher_yaw_correct_ready=bool(teacher_yaw_correct_ready),
                        teacher_enter_finger_region_ready=bool(teacher_enter_finger_region_ready),
                        teacher_xy_correct_ready=bool(teacher_xy_correct_ready),
                        teacher_preclose_descend_ready=bool(teacher_preclose_descend_ready),
                        teacher_alignment_ready_now=bool(teacher_alignment_ready_now),
                        teacher_verify_active_now=bool(teacher_verify_active_now),
                        teacher_close_hold_remaining=int(teacher_close_hold_remaining),
                        teacher_close_attempted_this_cycle=bool(teacher_close_attempted_this_cycle),
                        teacher_yaw_imitation_enabled=bool(
                            getattr(args, "alignment_tc_teacher_yaw_imitation_enabled", True)
                        ),
                        teacher_grasp_ready=bool(teacher_grasp_ready),
                    )
                    trace_entry["expert_sequence_name"] = str(teacher_expert_sequence_name)
                    trace_entry["expert_sequence_score"] = float(teacher_grasp_readiness_score)
                    trace_entry["expert_sequence_verified"] = False
                    teacher_motion_phase = str(teacher_expert_sequence_name)
                    if teacher_motion_phase == "align_xy":
                        teacher_motion_phase = "align_xy_yaw"
                    trace_entry["teacher_tc_motion_phase"] = str(teacher_motion_phase)
                    teacher_close_reason = "ok"
                    teacher_close_ready_now = bool(
                        not bool(getattr(args, "teacher_planner_close_handoff", True))
                        and
                        current_delta_basin is not None
                        and float(obs.gripper_open) >= 0.5
                        and bool(oracle_gate.get("near_target", False))
                        and teacher_close_xy_val is not None
                        and teacher_close_abs_z_val is not None
                        and not bool(teacher_close_attempted_this_cycle)
                        and teacher_close_xy_val <= tc_close_xy_threshold_eff
                        and teacher_close_yaw_ok
                        and teacher_close_contact_ready
                        and teacher_object_in_finger_region
                        and teacher_grasp_ready
                        and teacher_close_abs_z_val <= tc_close_abs_z_threshold_eff
                    )
                    if current_delta_basin is None:
                        teacher_close_reason = "no_delta"
                    elif float(obs.gripper_open) < 0.5:
                        teacher_close_reason = "gripper_not_open"
                    elif bool(getattr(args, "teacher_planner_close_handoff", True)) and teacher_alignment_ready_now:
                        teacher_close_reason = "planner_handoff"
                    elif not bool(oracle_gate.get("near_target", False)):
                        teacher_close_reason = "near_target"
                    elif teacher_close_xy_val is None:
                        teacher_close_reason = "xy_missing"
                    elif teacher_close_xy_val > tc_close_xy_threshold_eff:
                        teacher_close_reason = "xy"
                    elif teacher_close_abs_z_val is None:
                        teacher_close_reason = "z_missing"
                    elif teacher_close_abs_z_val > tc_close_abs_z_threshold_eff:
                        teacher_close_reason = "z"
                    elif not teacher_close_yaw_ok:
                        teacher_close_reason = "yaw"
                    elif not teacher_close_contact_ready:
                        teacher_close_reason = (
                            "contact_geometry"
                            if not teacher_close_contact_ready_by_geometry
                            else "contact_stage"
                            if not teacher_close_contact_ready_by_stage
                            else "contact_depth"
                        )
                    elif not teacher_object_in_finger_region:
                        teacher_close_reason = "finger_region"
                    elif not teacher_grasp_ready:
                        teacher_close_reason = "grasp_region"
                    elif (
                        float(args.teacher_close_basin_distance_threshold) > 0.0
                        and (teacher_close_dist_val is None or teacher_close_dist_val > float(args.teacher_close_basin_distance_threshold))
                    ):
                        teacher_close_reason = "basin_distance"
                    teacher_close_ready_all = bool(
                        teacher_close_ready_now
                        and teacher_close_contact_ready
                        and teacher_close_contact_ready_by_geometry
                        and (teacher_close_contact_ready_by_stage or teacher_close_contact_ready_by_depth)
                    )
                    trace_entry["teacher_close_ready"] = bool(teacher_close_ready_now)
                    trace_entry["teacher_close_ready_all"] = bool(teacher_close_ready_all)
                    trace_entry["teacher_close_gate_reason"] = str(teacher_close_reason)
                    trace_entry["teacher_close_xy"] = teacher_close_xy_val
                    trace_entry["teacher_close_abs_z"] = teacher_close_abs_z_val
                    trace_entry["teacher_close_yaw"] = teacher_close_yaw_val
                    trace_entry["teacher_close_yaw_raw"] = teacher_close_yaw_raw_val
                    trace_entry["teacher_close_yaw_folded"] = teacher_close_yaw_val
                    trace_entry["teacher_close_yaw_threshold"] = float(close_yaw_threshold)
                    trace_entry["teacher_close_yaw_ok"] = bool(teacher_close_yaw_ok)
                    trace_entry["teacher_close_contact_ready"] = bool(teacher_close_contact_ready)
                    trace_entry["teacher_close_contact_ready_by_depth"] = bool(teacher_close_contact_ready_by_depth)
                    trace_entry["teacher_close_contact_ready_by_stage"] = bool(teacher_close_contact_ready_by_stage)
                    trace_entry["teacher_close_contact_ready_by_geometry"] = bool(teacher_close_contact_ready_by_geometry)
                    trace_entry["teacher_close_contact_visibility_ok"] = bool(teacher_close_contact_visibility_ok)
                    trace_entry["teacher_close_contact_confidence"] = float(teacher_close_contact_confidence)
                    trace_entry["teacher_yaw_symmetry_period"] = float(teacher_yaw_symmetry_period)
                    trace_entry["teacher_grasp_commit_edge_pair_index"] = int(
                        getattr(target_result, "grasp_commit_edge_pair_index", -1)
                    )
                    trace_entry["teacher_grasp_commit_edge_pair_family"] = int(
                        getattr(target_result, "grasp_commit_edge_pair_family", -1)
                    )
                    trace_entry["teacher_grasp_commit_edge_pair_yaw_error"] = float(
                        getattr(target_result, "grasp_commit_edge_pair_yaw_error", np.nan)
                    )
                    trace_entry["teacher_commit_close_xy"] = teacher_commit_xy_val
                    trace_entry["teacher_commit_close_abs_z"] = teacher_commit_abs_z_val
                    trace_entry["teacher_commit_close_yaw"] = teacher_commit_yaw_val
                    trace_entry["teacher_planner_assisted_close_ready"] = bool(planner_assisted_close_ready)
                    trace_entry["teacher_alignment_ready"] = bool(teacher_alignment_ready_now)
                    trace_entry["teacher_alignment_handoff_active"] = bool(teacher_alignment_handoff_active)
                    trace_entry["teacher_planner_close_latch_remaining"] = int(teacher_planner_close_latch_remaining)
                    trace_entry["teacher_realign_cooldown_remaining"] = int(teacher_realign_cooldown_remaining)
                    trace_entry["teacher_planner_verify_remaining"] = int(teacher_planner_verify_remaining)
                    trace_entry["teacher_preclose_descend_ready"] = bool(teacher_preclose_descend_ready)
                    trace_entry["teacher_close_basin_distance"] = teacher_close_dist_val
                    trace_entry["teacher_verify_active"] = bool(teacher_verify_active_now)
                    trace_entry["teacher_retry_active"] = bool(teacher_retry_active_now)
                    trace_entry["teacher_orientation_rescue_active"] = bool(teacher_enable_orientation_rescue)
                    trace_entry["teacher_orientation_rescue_enabled"] = bool(teacher_orientation_rescue_enabled)
                    trace_entry["teacher_orientation_error_deg"] = float(teacher_orientation_error_deg)
                    trace_entry["teacher_use_commit_target"] = bool(teacher_use_commit_target_now)
                    if teacher_preclose_descend_ready and not teacher_close_ready_now:
                        teacher_motion_phase = "descend_z"
                    elif teacher_close_ready_now:
                        teacher_motion_phase = "close_ready"
                    else:
                        teacher_motion_phase = "align_xy_yaw"
                    trace_entry["teacher_motion_phase"] = str(teacher_motion_phase)
                    teacher_oracle_motion_gate = bool(
                        bool(oracle_gate.get("near_target", False))
                        and bool(oracle_gate.get("gripper_still_open", False))
                        and bool(oracle_gate.get("close_requirement_satisfied", False))
                        and not bool(teacher_alignment_handoff_active)
                        and int(teacher_realign_cooldown_remaining) == 0
                        and (
                            not bool(getattr(args, "teacher_require_alignment_ready_for_motion_gate", False))
                            or bool(teacher_alignment_ready_now)
                        )
                        and current_delta_basin is not None
                        and teacher_close_xy_val is not None
                        and teacher_close_abs_z_val is not None
                        and (
                            bool(planner_close_intent_now)
                            or (
                                teacher_close_xy_val <= float(getattr(args, "teacher_motion_entry_xy_threshold", 0.040))
                                and teacher_close_abs_z_val <= float(getattr(args, "teacher_motion_entry_abs_z_threshold", 0.120))
                            )
                        )
                    )
                    trace_entry["teacher_motion_gate_open"] = bool(teacher_oracle_motion_gate)
                    trace_entry["teacher_motion_entry_xy_threshold"] = float(getattr(args, "teacher_motion_entry_xy_threshold", 0.040))
                    trace_entry["teacher_motion_entry_abs_z_threshold"] = float(getattr(args, "teacher_motion_entry_abs_z_threshold", 0.120))
                    trace_entry["teacher_motion_gate_require_alignment_ready"] = bool(
                        getattr(args, "teacher_require_alignment_ready_for_motion_gate", False)
                    )
                    trace_entry["teacher_handoff_revoke_xy_threshold"] = float(getattr(args, "teacher_handoff_revoke_xy_threshold", 0.012))
                    trace_entry["teacher_handoff_revoke_abs_z_threshold"] = float(getattr(args, "teacher_handoff_revoke_abs_z_threshold", 0.025))
                    if teacher_retry_steps_remaining > 0:
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = np.asarray([0.0, 0.0, float(args.teacher_retry_lift), 0.0, 0.0, 0.0], dtype=np.float32)
                        delta_action[6] = 1.0
                        trace_entry["teacher_gripper_override"] = "retry_open"
                        teacher_retry_steps_remaining -= 1
                        teacher_retry_active_now = True
                        teacher_retry_required_now = True
                        teacher_alignment_handoff_active = False
                        teacher_smooth_action_local = None
                        teacher_last_candidate_idx = None
                        teacher_candidate_hold_remaining = 0
                        teacher_planner_close_latch_remaining = 0
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_realign_cooldown_remaining > 0:
                        delta_action = base_delta_action.copy()
                        delta_action[6] = 1.0
                        teacher_realign_cooldown_remaining -= 1
                        teacher_alignment_handoff_active = False
                        trace_entry["teacher_gripper_override"] = "realign_cooldown_force_open"
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_close_hold_remaining > 0:
                        delta_action = base_delta_action.copy()
                        verify_hold_steps = max(int(getattr(args, "teacher_verify_hold_steps", 1)), 1)
                        verify_settle_steps = max(int(getattr(args, "teacher_verify_settle_steps", 4)), 0)
                        verify_elapsed_steps = max(verify_hold_steps - 1 - int(teacher_close_hold_remaining), 0)
                        if verify_elapsed_steps < verify_settle_steps:
                            delta_action[:6] = np.zeros((6,), dtype=np.float32)
                            trace_entry["teacher_close_settle_phase"] = "settle"
                            trace_entry["teacher_close_action"] = "settle"
                        else:
                            verify_lift = max(float(args.teacher_retry_lift) * 0.5, 0.004)
                            delta_action[:6] = np.asarray([0.0, 0.0, verify_lift, 0.0, 0.0, 0.0], dtype=np.float32)
                            trace_entry["teacher_close_settle_phase"] = "lift_verify"
                            trace_entry["teacher_close_action"] = "verify_lift"
                        delta_action[6] = 0.0
                        trace_entry["teacher_gripper_override"] = "verify_hold_close"
                        teacher_close_hold_remaining -= 1
                        teacher_verify_active_now = True
                        refiner_stats_snapshot = refiner.get_stats()
                    elif (
                        bool(getattr(args, "teacher_planner_close_settle", True))
                        and teacher_last_close_object_z is not None
                        and int(teacher_planner_verify_remaining) > 0
                    ):
                        delta_action = base_delta_action.copy()
                        verify_lift_steps = max(int(args.teacher_planner_verify_lift_steps), 0)
                        verify_settle_steps = max(int(getattr(args, "teacher_verify_settle_steps", 4)), 0)
                        verify_elapsed_steps = max(int(args.teacher_planner_verify_steps) - int(teacher_planner_verify_remaining), 0)
                        if verify_elapsed_steps < verify_settle_steps and int(teacher_planner_verify_remaining) > verify_lift_steps:
                            delta_action[:6] = np.zeros((6,), dtype=np.float32)
                            trace_entry["teacher_close_settle_phase"] = "settle"
                            trace_entry["teacher_close_action"] = "settle"
                        else:
                            verify_lift = max(float(args.teacher_retry_lift) * 0.5, 0.004)
                            delta_action[:6] = np.asarray([0.0, 0.0, verify_lift, 0.0, 0.0, 0.0], dtype=np.float32)
                            trace_entry["teacher_close_settle_phase"] = "lift_verify"
                            trace_entry["teacher_close_action"] = "verify_lift"
                        delta_action[6] = 0.0
                        trace_entry["teacher_gripper_override"] = "planner_close_settle_verify"
                        teacher_verify_active_now = True
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_alignment_handoff_active:
                        delta_action = base_delta_action.copy()
                        planner_handoff_gripper_raw = float(delta_action[6])
                        latch_released_close = bool(
                            teacher_alignment_ready_now
                            and int(teacher_planner_close_latch_remaining) > 0
                            and float(obs.gripper_open) >= 0.5
                        )
                        if (
                            bool(getattr(args, "teacher_planner_close_settle", True))
                            and (
                                planner_handoff_gripper_raw <= float(getattr(args, "teacher_planner_close_gripper_threshold", 0.5))
                                or latch_released_close
                            )
                            and bool(teacher_alignment_ready_now)
                        ):
                            delta_action[:6] = np.zeros((6,), dtype=np.float32)
                            delta_action[6] = 0.0
                            trace_entry["teacher_gripper_override"] = (
                                "planner_close_latch_release"
                                if latch_released_close and planner_handoff_gripper_raw > float(getattr(args, "teacher_planner_close_gripper_threshold", 0.5))
                                else "planner_close_settle_start"
                            )
                            teacher_planner_close_latch_remaining = 0
                        else:
                            trace_entry["teacher_gripper_override"] = "planner_handoff_after_alignment"
                        trace_entry["teacher_handoff_planner_gripper_raw"] = float(planner_handoff_gripper_raw)
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_close_ready_now:
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = np.zeros((6,), dtype=np.float32)
                        delta_action[6] = 0.0
                        trace_entry["teacher_gripper_override"] = "teacher_close"
                        teacher_close_hold_remaining = max(int(args.teacher_verify_hold_steps) - 1, 0)
                        oracle_pregrasp_close_count += 1
                        if teacher_close_attempt_step is None:
                            teacher_close_attempt_step = int(step_idx)
                        teacher_last_close_object_z = (
                            None if teacher_live_object_pose is None else float(teacher_live_object_pose[2])
                        )
                        if teacher_live_object_pose is not None:
                            teacher_last_close_object_pose = np.asarray(teacher_live_object_pose, dtype=np.float32).copy()
                        else:
                            low_dim_pose = safe_task_low_dim_pose_7d(obs)
                            teacher_last_close_object_pose = (
                                None if low_dim_pose is None else np.asarray(low_dim_pose, dtype=np.float32).copy()
                            )
                            if teacher_last_close_object_z is None and low_dim_pose is not None:
                                teacher_last_close_object_z = float(low_dim_pose[2])
                        teacher_last_close_gripper_pose = np.asarray(obs.gripper_pose, dtype=np.float32).copy()
                        teacher_verify_active_now = True
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_expert_sequence_name == "yaw_correct":
                        stage_action_local = _build_tc_hard_stage_action(
                            phase="yaw_correct",
                            current_delta_local_6d=current_delta_basin,
                            grasp_delta_local_6d=teacher_object_to_gripper_delta_local_6d,
                            yaw_raw_val=teacher_close_yaw_raw_val,
                            yaw_control_sign=teacher_yaw_control_sign,
                            bucket=tc_teacher_bucket,
                            args=args,
                        )
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = local_delta_to_world(
                            stage_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action[6] = 1.0
                        oracle_collect_active = True
                        oracle_collect_idx = -1
                        oracle_collect_score = 0.0
                        oracle_collect_action_local = stage_action_local.astype(np.float32)
                        oracle_collect_current_dist = float(current_basin_distance if current_basin_distance is not None else 0.0)
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [0.0]
                        oracle_collect_candidate_actions = stage_action_local.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        trace_entry["teacher_gripper_override"] = "hard_stage_yaw_correct"
                        trace_entry["teacher_close_action"] = "yaw_correct"
                        trace_entry["teacher_motion_controller"] = "tc_privileged_yaw_correct"
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = -1
                        trace_entry["oracle_score"] = 0.0
                        trace_entry["oracle_action_local"] = stage_action_local.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_expert_sequence_name == "enter_finger_region":
                        grasp_delta_for_control = (
                            np.asarray(teacher_object_to_gripper_delta_local_6d, dtype=np.float32)
                            if teacher_object_to_gripper_delta_local_6d is not None
                            else np.asarray(current_delta_basin, dtype=np.float32)
                        )
                        stage_action_local = _build_tc_hard_stage_action(
                            phase="enter_finger_region",
                            current_delta_local_6d=current_delta_basin,
                            grasp_delta_local_6d=grasp_delta_for_control,
                            yaw_raw_val=teacher_close_yaw_raw_val,
                            yaw_control_sign=teacher_yaw_control_sign,
                            bucket=tc_teacher_bucket,
                            args=args,
                        )
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = local_delta_to_world(
                            stage_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action[6] = 1.0
                        oracle_collect_active = True
                        oracle_collect_idx = -1
                        oracle_collect_score = 0.0
                        oracle_collect_action_local = stage_action_local.astype(np.float32)
                        oracle_collect_current_dist = float(current_basin_distance if current_basin_distance is not None else 0.0)
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [0.0]
                        oracle_collect_candidate_actions = stage_action_local.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        trace_entry["teacher_gripper_override"] = "hard_stage_enter_finger_region"
                        trace_entry["teacher_close_action"] = "enter_finger_region"
                        trace_entry["teacher_motion_controller"] = "tc_privileged_enter_finger_region"
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = -1
                        trace_entry["oracle_score"] = 0.0
                        trace_entry["oracle_action_local"] = stage_action_local.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif teacher_expert_sequence_name == "descend_z":
                        stage_action_local = _build_tc_hard_stage_action(
                            phase="descend_z",
                            current_delta_local_6d=current_delta_basin,
                            grasp_delta_local_6d=teacher_object_to_gripper_delta_local_6d,
                            yaw_raw_val=teacher_close_yaw_raw_val,
                            yaw_control_sign=teacher_yaw_control_sign,
                            bucket=tc_teacher_bucket,
                            args=args,
                        )
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = local_delta_to_world(
                            stage_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action[6] = 1.0
                        oracle_collect_active = True
                        oracle_collect_idx = -1
                        oracle_collect_score = 0.0
                        oracle_collect_action_local = stage_action_local.astype(np.float32)
                        oracle_collect_current_dist = float(current_basin_distance if current_basin_distance is not None else 0.0)
                        oracle_collect_next_dists = [float(oracle_collect_current_dist)]
                        oracle_collect_scores = [0.0]
                        oracle_collect_candidate_actions = stage_action_local.reshape(1, 6).astype(np.float32)
                        oracle_executed_alignment_count += 1
                        trace_entry["teacher_gripper_override"] = "hard_stage_descend_z"
                        trace_entry["teacher_close_action"] = "descend_z"
                        trace_entry["teacher_motion_controller"] = "tc_privileged_descend_z"
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = -1
                        trace_entry["oracle_score"] = 0.0
                        trace_entry["oracle_action_local"] = stage_action_local.astype(np.float32).tolist()
                        trace_entry["teacher_yaw_control_sign"] = float(teacher_yaw_control_sign)
                        refiner.manager.update(
                            force_reading=raw_force,
                            gripper_pose=obs.gripper_pose,
                            gripper_open=float(obs.gripper_open),
                            depth_proximity=depth_proximity,
                            base_action=delta_action,
                        )
                        refiner_stats_snapshot = refiner.get_stats()
                    elif int(refiner._residual_cooldown) == 0 and teacher_oracle_motion_gate:
                        oracle_collect_candidate_actions = teacher_candidate_actions_all
                        oracle_collect_idx, oracle_collect_score, oracle_collect_current_dist, oracle_collect_next_dists, oracle_collect_scores = scorer_oracle(
                            current_pose_7d=np.asarray(obs.gripper_pose, dtype=np.float32),
                            pregrasp_target_pose=np.asarray(runtime_target_pose, dtype=np.float32),
                            grasp_commit_target_pose=np.asarray(runtime_target_pose, dtype=np.float32),
                            candidate_actions_local=oracle_collect_candidate_actions,
                            base_action_local=np.asarray(planner_base_action_local_raw, dtype=np.float32),
                            depth_proximity=depth_proximity,
                            candidate_mask=teacher_candidate_mask,
                            candidate_kind=teacher_candidate_kind,
                            close_xy_threshold=float(getattr(args, "teacher_close_xy_threshold", 0.020)),
                            close_abs_z_threshold=float(getattr(args, "teacher_close_abs_z_threshold", 0.020)),
                            close_yaw_threshold=float(getattr(args, "teacher_close_yaw_threshold", 0.12)),
                            commit_switch_xy_threshold=float(getattr(args, "teacher_commit_switch_xy_threshold", 0.010)),
                            commit_switch_z_threshold=float(getattr(args, "teacher_commit_switch_z_threshold", 0.020)),
                            commit_switch_yaw_threshold=float(getattr(args, "teacher_commit_switch_yaw_threshold", 0.12)),
                            stall_count=int(getattr(controller, "_teacher_pregrasp_stall_count", 0)),
                            orientation_rescue_active=bool(teacher_enable_orientation_rescue),
                            yaw_symmetry_period=float(teacher_yaw_symmetry_period),
                        )
                        oracle_collect_action_local = np.asarray(
                            oracle_collect_candidate_actions[int(oracle_collect_idx)], dtype=np.float32
                        ).copy()
                        raw_oracle_idx = int(oracle_collect_idx)
                        raw_oracle_score = float(oracle_collect_score)
                        selected_idx = raw_oracle_idx
                        if (
                            teacher_last_candidate_idx is not None
                            and int(teacher_candidate_hold_remaining) > 0
                            and 0 <= int(teacher_last_candidate_idx) < len(oracle_collect_scores)
                        ):
                            held_score = float(oracle_collect_scores[int(teacher_last_candidate_idx)])
                            if held_score + float(args.teacher_candidate_switch_margin) >= raw_oracle_score:
                                selected_idx = int(teacher_last_candidate_idx)
                                oracle_collect_idx = selected_idx
                                oracle_collect_score = held_score
                                oracle_collect_action_local = np.asarray(
                                    oracle_collect_candidate_actions[selected_idx], dtype=np.float32
                                ).copy()
                                teacher_candidate_hold_remaining -= 1
                                trace_entry["teacher_candidate_hysteresis"] = "held"
                            else:
                                teacher_candidate_hold_remaining = max(int(args.teacher_candidate_hold_steps) - 1, 0)
                                trace_entry["teacher_candidate_hysteresis"] = "switched_margin"
                        else:
                            teacher_candidate_hold_remaining = max(int(args.teacher_candidate_hold_steps) - 1, 0)
                            trace_entry["teacher_candidate_hysteresis"] = "new"
                        teacher_last_candidate_idx = int(oracle_collect_idx)
                        teacher_last_candidate_score = float(oracle_collect_score)

                        use_far_step = bool(
                            teacher_close_xy_val is not None
                            and teacher_close_abs_z_val is not None
                            and (
                                teacher_close_xy_val > float(args.teacher_near_smooth_xy_threshold)
                                or teacher_close_abs_z_val > float(args.teacher_near_smooth_abs_z_threshold)
                            )
                        )
                        action_scale = float(args.teacher_far_action_scale) if use_far_step else 1.0
                        if bool(getattr(args, "teacher_use_continuous_smooth_control", True)) and current_delta_basin is not None:
                            current_delta_arr = (
                                np.asarray(teacher_current_delta_for_control, dtype=np.float32)
                                if teacher_current_delta_for_control is not None
                                else np.asarray(current_delta_basin, dtype=np.float32)
                            )
                            max_step = float(args.teacher_far_max_step if use_far_step else args.teacher_near_max_step)
                            max_yaw_step = float(getattr(args, "teacher_max_yaw_step", 0.035))
                            p_action = np.zeros((6,), dtype=np.float32)
                            p_action[:2] = np.clip(
                                float(args.teacher_smooth_kp_xy) * current_delta_arr[:2],
                                -max_step,
                                max_step,
                            )
                            p_action[2] = float(np.clip(
                                float(args.teacher_smooth_kp_z) * current_delta_arr[2],
                                -max_step,
                                max_step,
                            ))
                            yaw_sign = -1.0 if bool(getattr(args, "teacher_invert_yaw_control", True)) else 1.0
                            p_action[5] = float(np.clip(
                                yaw_sign * float(getattr(args, "teacher_smooth_kp_yaw", 0.50)) * current_delta_arr[5],
                                -max_yaw_step,
                                max_yaw_step,
                            ))
                            xy_deadband = float(args.teacher_smooth_xy_deadband)
                            z_deadband = float(args.teacher_smooth_z_deadband)
                            yaw_deadband = float(getattr(args, "teacher_smooth_yaw_deadband", 0.025))
                            p_action[:2] = np.where(np.abs(current_delta_arr[:2]) < xy_deadband, 0.0, p_action[:2])
                            if abs(float(current_delta_arr[2])) < z_deadband:
                                p_action[2] = 0.0
                            if abs(float(current_delta_arr[5])) < yaw_deadband:
                                p_action[5] = 0.0
                            oracle_collect_action_local = p_action
                            trace_entry["teacher_motion_controller"] = "continuous_p_smooth"
                        else:
                            max_step = float(args.teacher_far_max_step if action_scale > 1.0 else args.teacher_near_max_step)
                            oracle_collect_action_local = np.asarray(oracle_collect_action_local, dtype=np.float32) * float(action_scale)
                            trace_entry["teacher_motion_controller"] = "candidate_smooth"
                        if teacher_smooth_action_local is None:
                            smoothed_action_local = oracle_collect_action_local.copy()
                        else:
                            alpha = float(args.teacher_action_lowpass_alpha)
                            smoothed_action_local = (
                                alpha * oracle_collect_action_local
                                + (1.0 - alpha) * np.asarray(teacher_smooth_action_local, dtype=np.float32)
                            ).astype(np.float32)
                        smoothed_action_local[:3] = np.clip(smoothed_action_local[:3], -max_step, max_step)
                        smoothed_action_local[3:5] = np.clip(smoothed_action_local[3:5], -0.03, 0.03)
                        smoothed_action_local[5] = np.clip(
                            smoothed_action_local[5],
                            -float(getattr(args, "teacher_max_yaw_step", 0.035)),
                            float(getattr(args, "teacher_max_yaw_step", 0.035)),
                        )
                        teacher_smooth_action_local = smoothed_action_local.copy()
                        trace_entry["teacher_raw_candidate_index"] = int(raw_oracle_idx)
                        trace_entry["teacher_raw_oracle_score"] = float(raw_oracle_score)
                        trace_entry["teacher_action_scale"] = float(action_scale)
                        trace_entry["teacher_smoothed_action_local"] = smoothed_action_local.tolist()
                        oracle_collect_action_local = smoothed_action_local
                        oracle_collect_action_world = local_delta_to_world(
                            oracle_collect_action_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        delta_action = base_delta_action.copy()
                        delta_action[:6] = oracle_collect_action_world
                        if teacher_force_open_in_align:
                            delta_action[6] = 1.0
                            trace_entry["teacher_gripper_override"] = "align_force_open"
                        oracle_collect_active = True
                        oracle_executed_alignment_count += 1
                        if float(np.linalg.norm(oracle_collect_action_local)) < 1e-8:
                            oracle_executed_noop_count += 1
                        trace_entry["oracle_executed_align"] = True
                        trace_entry["oracle_candidate_index"] = int(oracle_collect_idx)
                        trace_entry["oracle_score"] = float(oracle_collect_score)
                        trace_entry["oracle_action_local"] = oracle_collect_action_local.tolist()
                        trace_entry["teacher_candidate_mask"] = teacher_candidate_mask.astype(np.float32).tolist()
                        trace_entry["planner_base_action_local_raw"] = planner_base_action_local_raw.tolist()
                        trace_entry["residual_label_local"] = (
                            oracle_collect_action_local - planner_base_action_local_raw
                        ).astype(np.float32).tolist()
                        if float(np.linalg.norm(oracle_collect_action_local)) < 1e-8:
                            controller._teacher_pregrasp_stall_count = int(getattr(controller, "_teacher_pregrasp_stall_count", 0)) + 1
                        else:
                            controller._teacher_pregrasp_stall_count = 0
                        refiner_stats_snapshot = refiner.get_stats()
                        oracle_collect_debug_pred = scorer_forward_debug(
                            controller,
                            np.asarray(obs.front_rgb, dtype=np.uint8),
                            np.asarray(obs.wrist_rgb, dtype=np.uint8),
                            depth_tensor_96.detach().cpu().numpy().astype(np.float32),
                            np.asarray(proprio, dtype=np.float32),
                            np.asarray(planner_base_action_local_raw, dtype=np.float32),
                            chunk_step,
                            gripper_context_arr,
                        )
                    else:
                        delta_action = refiner.step(**step_kwargs)
                        if teacher_force_open_in_align:
                            delta_action = np.asarray(delta_action, dtype=np.float32).copy()
                            delta_action[6] = 1.0
                            trace_entry["teacher_gripper_override"] = "fallback_force_open"
                        refiner_stats_snapshot = refiner.get_stats()
                else:
                    delta_action = refiner.step(**step_kwargs)
                    if isinstance(refiner, StageAwareRefiner):
                        refiner_stats_snapshot = refiner.get_stats()
                if isinstance(refiner, StageAwareRefiner) and refiner_stats_snapshot is not None:
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
                    trace_entry["refiner_alignment_gate_open"] = bool(
                        refiner_stats_snapshot.get("current_alignment_gate_open", False)
                    )
                    trace_entry["refiner_alignment_blocked_reason"] = str(
                        refiner_stats_snapshot.get("current_alignment_blocked_reason", "unknown")
                    )
                    trace_entry["refiner_alignment_depth_proximity"] = refiner_stats_snapshot.get(
                        "current_alignment_depth_proximity", None
                    )
                    trace_entry["refiner_alignment_near_target"] = bool(
                        refiner_stats_snapshot.get("current_alignment_near_target", False)
                    )
                    trace_entry["refiner_alignment_gripper_open"] = refiner_stats_snapshot.get(
                        "current_alignment_gripper_open", None
                    )
                    trace_entry["refiner_alignment_gripper_still_open"] = bool(
                        refiner_stats_snapshot.get("current_alignment_gripper_still_open", False)
                    )
                    trace_entry["refiner_alignment_planner_close_intent"] = bool(
                        refiner_stats_snapshot.get("current_alignment_planner_close_intent", False)
                    )
                    trace_entry["refiner_alignment_close_requirement_satisfied"] = bool(
                        refiner_stats_snapshot.get("current_alignment_close_requirement_satisfied", False)
                    )
                    trace_entry["refiner_alignment_window_active"] = bool(
                        refiner_stats_snapshot.get("current_alignment_window_active", False)
                    )
                    trace_entry["refiner_alignment_short_window_available"] = bool(
                        refiner_stats_snapshot.get("current_alignment_short_window_available", False)
                    )
                    trace_entry["refiner_current_alignment_support_satisfied"] = bool(
                        refiner_stats_snapshot.get("current_alignment_support_satisfied", False)
                    )
                    trace_entry["refiner_current_alignment_support_inner_satisfied"] = bool(
                        refiner_stats_snapshot.get("current_alignment_support_inner_satisfied", False)
                    )
                    trace_entry["refiner_current_alignment_support_outer_satisfied"] = bool(
                        refiner_stats_snapshot.get("current_alignment_support_outer_satisfied", False)
                    )
                    trace_entry["refiner_current_alignment_support_soft_override"] = bool(
                        refiner_stats_snapshot.get("current_alignment_support_soft_override", False)
                    )
                    trace_entry["refiner_current_alignment_support_soft_override_reason"] = str(
                        refiner_stats_snapshot.get("current_alignment_support_soft_override_reason", "none")
                    )
                    trace_entry["refiner_current_alignment_use_outer_rescue"] = bool(
                        refiner_stats_snapshot.get("current_alignment_use_outer_rescue", False)
                    )
                    trace_entry["refiner_current_alignment_refine_band_satisfied"] = bool(
                        refiner_stats_snapshot.get("current_alignment_refine_band_satisfied", False)
                    )
                    trace_entry["refiner_current_alignment_takeover_band_satisfied"] = bool(
                        refiner_stats_snapshot.get("current_alignment_takeover_band_satisfied", False)
                    )
                    trace_entry["refiner_current_alignment_takeover_active"] = bool(
                        refiner_stats_snapshot.get("current_alignment_takeover_active", False)
                    )
                    trace_entry["refiner_current_close_veto_blocked"] = bool(
                        refiner_stats_snapshot.get("current_close_veto_blocked", False)
                    )
                    trace_entry["refiner_current_close_veto_ready"] = bool(
                        refiner_stats_snapshot.get("current_close_veto_ready", False)
                    )
                    trace_entry["refiner_current_close_veto_ready_streak"] = int(
                        refiner_stats_snapshot.get("current_close_veto_ready_streak", 0)
                    )
                    trace_entry["refiner_current_close_veto_settle_remaining"] = int(
                        refiner_stats_snapshot.get("current_close_veto_settle_remaining", 0)
                    )
                    trace_entry["refiner_current_close_latch_remaining"] = int(
                        refiner_stats_snapshot.get("current_close_latch_remaining", 0)
                    )
                    trace_entry["refiner_current_phase1_post_close_hold_remaining"] = int(
                        refiner_stats_snapshot.get("current_phase1_post_close_hold_remaining", 0)
                    )
                    trace_entry["refiner_current_phase1_post_close_hold_active"] = bool(
                        refiner_stats_snapshot.get("current_phase1_post_close_hold_active", False)
                    )
                    trace_entry["refiner_current_student_ready_gate_active"] = bool(
                        refiner_stats_snapshot.get("current_student_ready_gate_active", False)
                    )
                    trace_entry["refiner_current_student_close_ready_logit"] = float(
                        refiner_stats_snapshot.get("current_student_close_ready_logit", np.nan)
                    )
                    trace_entry["refiner_current_student_close_ready_prob"] = float(
                        refiner_stats_snapshot.get("current_student_close_ready_prob", np.nan)
                    )
                    trace_entry["refiner_current_student_close_ready_pred"] = bool(
                        refiner_stats_snapshot.get("current_student_close_ready_pred", False)
                    )
                    trace_entry["refiner_current_student_close_ready"] = bool(
                        refiner_stats_snapshot.get("current_student_close_ready", False)
                    )
                    trace_entry["refiner_current_student_close_ready_applied"] = bool(
                        refiner_stats_snapshot.get("current_student_close_ready_applied", False)
                    )
                    trace_entry["refiner_current_student_handoff_ready_logit"] = float(
                        refiner_stats_snapshot.get("current_student_handoff_ready_logit", np.nan)
                    )
                    trace_entry["refiner_current_student_handoff_ready_prob"] = float(
                        refiner_stats_snapshot.get("current_student_handoff_ready_prob", np.nan)
                    )
                    trace_entry["refiner_current_student_handoff_ready_pred"] = bool(
                        refiner_stats_snapshot.get("current_student_handoff_ready_pred", False)
                    )
                    trace_entry["refiner_current_student_handoff_ready"] = bool(
                        refiner_stats_snapshot.get("current_student_handoff_ready", False)
                    )
                    trace_entry["refiner_current_student_handoff_ready_applied"] = bool(
                        refiner_stats_snapshot.get("current_student_handoff_ready_applied", False)
                    )
                    close_state_machine = dict(refiner_stats_snapshot.get("close_state_machine", {}) or {})
                    trace_entry["refiner_close_state_machine"] = close_state_machine
                    for key in (
                        "close_state",
                        "close_action_decision",
                        "close_blocked_reason",
                    ):
                        trace_entry[f"refiner_{key}"] = str(refiner_stats_snapshot.get(key, "unknown"))
                    for key in (
                        "close_runtime_geometry_ready",
                        "close_handoff_ready_pred",
                        "close_handoff_ready_applied",
                        "close_handoff_shadow_blocks_apply",
                        "close_fallback_enabled",
                        "close_fallback_used",
                    ):
                        trace_entry[f"refiner_{key}"] = bool(refiner_stats_snapshot.get(key, False))
                    for key in (
                        "xy_error",
                        "abs_z_error",
                        "yaw_error",
                        "xy_threshold",
                        "abs_z_threshold",
                        "yaw_threshold",
                    ):
                        trace_entry[f"refiner_close_{key}"] = close_state_machine.get(key, None)
                    trace_entry["refiner_bounded_auto_close_ready"] = bool(
                        refiner_stats_snapshot.get("current_bounded_auto_close_ready", False)
                    )
                    trace_entry["refiner_bounded_auto_close_applied"] = bool(
                        refiner_stats_snapshot.get("current_bounded_auto_close_applied", False)
                    )
                    trace_entry["refiner_bounded_auto_close_ready_streak"] = int(
                        refiner_stats_snapshot.get("current_bounded_auto_close_ready_streak", 0)
                    )
                    trace_entry["refiner_force_close_after_b2_applied"] = bool(
                        refiner_stats_snapshot.get("current_force_close_after_b2_applied", False)
                    )
                    trace_entry["refiner_close_intent_shadow_would_auto_close"] = bool(
                        refiner_stats_snapshot.get("current_close_intent_shadow_would_auto_close", False)
                    )
                    trace_entry["refiner_close_intent_shadow_reason"] = str(
                        refiner_stats_snapshot.get("current_close_intent_shadow_reason", "unknown")
                    )
                    trace_entry["refiner_close_intent_shadow_blocking_axis"] = str(
                        refiner_stats_snapshot.get("current_close_intent_shadow_blocking_axis", "unknown")
                    )
                    trace_entry["refiner_close_intent_shadow_confidence"] = float(
                        refiner_stats_snapshot.get("current_close_intent_shadow_confidence", 0.0)
                    )
                    trace_entry["refiner_close_intent_shadow_candidate_count"] = int(
                        refiner_stats_snapshot.get("close_intent_shadow_candidate_count", 0)
                    )
                    trace_entry["refiner_current_handoff_ready"] = bool(
                        refiner_stats_snapshot.get("current_handoff_ready", False)
                    )
                    trace_entry["refiner_current_gripper_fsm_state"] = str(
                        refiner_stats_snapshot.get("gripper_fsm_state", "unknown")
                    )
                    trace_entry["phase1_close_arbiter_state"] = str(
                        refiner_stats_snapshot.get("phase1_close_arbiter_state", "APPROACH")
                    )
                    trace_entry["phase1_close_command_source"] = str(
                        refiner_stats_snapshot.get("phase1_close_command_source", "none")
                    )
                    trace_entry["phase1_grasp_contact_confirmed"] = bool(
                        refiner_stats_snapshot.get("phase1_grasp_contact_confirmed", False)
                    )
                    trace_entry["phase1_force_reflex_active"] = bool(
                        refiner_stats_snapshot.get("phase1_force_reflex_active", False)
                    )
                    trace_entry["phase1_force_reflex_reason"] = str(
                        refiner_stats_snapshot.get("phase1_force_reflex_reason", "none")
                    )
                    trace_entry["phase1_force_backoff_applied"] = bool(
                        refiner_stats_snapshot.get("phase1_force_backoff_applied", False)
                    )
                    trace_entry["phase1_reopen_reason"] = str(
                        refiner_stats_snapshot.get("phase1_reopen_reason", "none")
                    )
                    trace_entry["phase1_close_hold_active"] = bool(
                        refiner_stats_snapshot.get("phase1_close_hold_active", False)
                    )
                    trace_entry["phase1_close_hold_remaining"] = int(
                        refiner_stats_snapshot.get("phase1_close_hold_remaining", 0)
                    )
                    trace_entry["refiner_zone_state"] = str(refiner_stats_snapshot.get("zone_state", "planner_only"))
                    trace_entry["refiner_alignment_active"] = bool(refiner_stats_snapshot.get("alignment_active", False))
                    trace_entry["refiner_alignment_takeover_active"] = bool(refiner_stats_snapshot.get("alignment_takeover_active", False))
                    trace_entry["refiner_alignment_near_zone_gate_enabled"] = bool(
                        refiner_stats_snapshot.get("current_near_zone_gate_enabled", False)
                    )
                    trace_entry["refiner_alignment_near_zone_gate_pass"] = bool(
                        refiner_stats_snapshot.get("current_near_zone_gate_pass", False)
                    )
                    trace_entry["refiner_alignment_near_zone_xy_error"] = refiner_stats_snapshot.get(
                        "current_near_zone_xy_error", None
                    )
                    trace_entry["refiner_alignment_near_zone_z_error"] = refiner_stats_snapshot.get(
                        "current_near_zone_z_error", None
                    )
                    trace_entry["refiner_alignment_near_zone_block_reason"] = str(
                        refiner_stats_snapshot.get("current_near_zone_block_reason", "disabled")
                    )
                    # v3 direct-local shadow fields
                    trace_entry["refiner_alignment_v3_shadow_enabled"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_enabled", False)
                    )
                    trace_entry["refiner_alignment_v3_shadow_active"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_active", False)
                    )
                    trace_entry["refiner_alignment_v3_shadow_eval_count"] = int(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_eval_count", 0)
                    )
                    trace_entry["refiner_alignment_v3_shadow_gate_pass_count"] = int(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_gate_pass_count", 0)
                    )
                    trace_entry["refiner_alignment_v3_shadow_gate_pass"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_gate_pass", False)
                    )
                    trace_entry["refiner_alignment_v3_shadow_source"] = str(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_source", "none")
                    )
                    trace_entry["refiner_alignment_v3_shadow_block_reason"] = str(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_block_reason", "disabled")
                    )
                    trace_entry["refiner_alignment_v3_shadow_cur_xy"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_cur_xy", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_cur_z"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_cur_z", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_cur_yaw"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_cur_yaw", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_post_xy"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_post_xy", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_post_z"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_post_z", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_post_yaw"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_post_yaw", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_xy_improved"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_xy_improved", False)
                    )
                    trace_entry["refiner_alignment_v3_shadow_z_improved"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_z_improved", False)
                    )
                    trace_entry["refiner_alignment_v3_shadow_yaw_improved"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_yaw_improved", False)
                    )
                    trace_entry["refiner_alignment_v3_shadow_all_improved"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_all_improved", False)
                    )
                    trace_entry["refiner_alignment_v3_shadow_pred_residual_4d"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_pred_residual_4d", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_pred_residual_6d"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_pred_residual_6d", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_pred_pos_norm"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_pred_pos_norm", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_pred_yaw_abs"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_pred_yaw_abs", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_risk_logit"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_risk_logit", None)
                    )
                    trace_entry["refiner_alignment_v3_shadow_confidence_logit"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_shadow_confidence_logit", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_enabled"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_enabled", False)
                    )
                    trace_entry["refiner_alignment_v4_shadow_active"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_active", False)
                    )
                    trace_entry["refiner_alignment_v4_shadow_eval_count"] = int(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_eval_count", 0)
                    )
                    trace_entry["refiner_alignment_v4_shadow_source"] = str(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_source", "none")
                    )
                    trace_entry["refiner_alignment_v4_shadow_block_reason"] = str(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_block_reason", "disabled")
                    )
                    trace_entry["refiner_alignment_v4_shadow_cur_xy"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_cur_xy", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_cur_z"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_cur_z", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_cur_yaw"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_cur_yaw", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_post_xy"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_post_xy", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_post_z"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_post_z", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_post_yaw"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_post_yaw", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_xy_improved"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_xy_improved", False)
                    )
                    trace_entry["refiner_alignment_v4_shadow_z_improved"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_z_improved", False)
                    )
                    trace_entry["refiner_alignment_v4_shadow_yaw_improved"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_yaw_improved", False)
                    )
                    trace_entry["refiner_alignment_v4_shadow_all_improved"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_all_improved", False)
                    )
                    trace_entry["refiner_alignment_v4_shadow_pred_residual_4d"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_pred_residual_4d", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_pred_residual_6d"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_pred_residual_6d", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_pred_post_xyz_yaw"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_pred_post_xyz_yaw", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_pred_reduction_xyz"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_pred_reduction_xyz", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_pred_pos_norm"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_pred_pos_norm", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_pred_yaw_abs"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_pred_yaw_abs", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_risk_logit"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_risk_logit", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_confidence_logit"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_confidence_logit", None)
                    )
                    trace_entry["refiner_alignment_v4_shadow_policy_mode"] = str(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_policy_mode", "unknown")
                    )
                    trace_entry["refiner_alignment_v4_shadow_stage_bucket"] = str(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_stage_bucket", "unknown")
                    )
                    trace_entry["refiner_alignment_v4_shadow_micro_gate"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_micro_gate", False)
                    )
                    trace_entry["refiner_alignment_v4_shadow_micro_only"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_shadow_micro_only", False)
                    )
                    trace_entry["refiner_alignment_v4_apply_enabled"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_enabled", False)
                    )
                    trace_entry["refiner_alignment_v4_apply_count"] = int(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_count", 0) or 0
                    )
                    trace_entry["refiner_alignment_v4_apply_applied"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_applied", False)
                    )
                    trace_entry["refiner_alignment_v4_apply_block_reason"] = str(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_block_reason", "disabled")
                    )
                    trace_entry["refiner_alignment_v4_apply_local_delta"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_local_delta", None)
                    )
                    trace_entry["refiner_alignment_v4_apply_world_delta"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_world_delta", None)
                    )
                    trace_entry["refiner_alignment_v4_apply_pos_norm"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_pos_norm", None)
                    )
                    trace_entry["refiner_alignment_v4_apply_yaw_abs"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_yaw_abs", None)
                    )
                    trace_entry["refiner_alignment_v4_apply_stage_bucket"] = str(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_stage_bucket", "unknown")
                    )
                    trace_entry["refiner_alignment_v4_apply_micro_gate"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_micro_gate", False)
                    )
                    trace_entry["refiner_alignment_v4_apply_micro_only"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v4_apply_micro_only", False)
                    )
                    for _key in (
                        "enabled",
                        "shadow_enabled",
                        "apply_enabled",
                        "eval_count",
                        "active_count",
                        "apply_count",
                        "active",
                        "applied",
                        "block_reason",
                        "trigger_mode",
                        "stage_bucket",
                        "safety_reject",
                        "selected_index",
                        "num_samples",
                        "candidate_diversity",
                        "candidate_score",
                        "risk_prob",
                        "stop_prob",
                        "progress_logits",
                        "first_residual_4d",
                        "first_residual_6d",
                        "local_delta",
                        "world_delta",
                        "pos_norm",
                        "yaw_abs",
                        "workspace_violation",
                        "workspace_projected",
                        "workspace_project_reason",
                        "phase1_bridge_blend_active",
                        "phase1_bridge_blend_reason",
                        "phase1_bridge_blended_base_world_6d",
                        "phase1_bridge_blended_base_local_6d",
                        "phase1_bridge_basin_bias_local_6d",
                        "phase1_bridge_taskspace_yaw_error",
                        "phase1_bridge_taskspace_yaw_target_source",
                        "phase1_bridge_direct_yaw_rescue_local_6d",
                        "phase1_bridge_yaw_holdoff_active",
                        "phase1_bridge_yaw_holdoff_reason",
                        "controller_type",
                        "target_confidence",
                        "pred_target_delta_6d",
                        "pred_target_delta_norm",
                        "pred_target_yaw_abs",
                        "target_action_sign_agreement",
                        "low_confidence",
                        "soft_clamp",
                        "scale_down",
                        "hard_reject",
                        "phase1_bridge_soft_apply",
                        "phase1_bridge_soft_apply_reason",
                        "latency_total_ms",
                        "top_k",
                    ):
                        _stat_key = f"current_alignment_diffusion_{_key}"
                        _trace_key = f"refiner_alignment_diffusion_{_key}"
                        _val = refiner_stats_snapshot.get(_stat_key, None)
                        if isinstance(_val, (bool, str)):
                            trace_entry[_trace_key] = _val
                        elif _key in (
                            "eval_count",
                            "active_count",
                            "apply_count",
                            "selected_index",
                            "num_samples",
                            "top_k",
                        ):
                            trace_entry[_trace_key] = int(_val or 0)
                        else:
                            trace_entry[_trace_key] = _jsonable_value(_val)
                    trace_entry["refiner_alignment_v3_apply_enabled"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_enabled", False)
                    )
                    trace_entry["refiner_alignment_v3_apply_scale"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_scale", None)
                    )
                    trace_entry["refiner_alignment_v3_apply_max_pos"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_max_pos", None)
                    )
                    trace_entry["refiner_alignment_v3_apply_max_yaw"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_max_yaw", None)
                    )
                    trace_entry["refiner_alignment_v3_apply_only_when_gate_pass"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_only_when_gate_pass", True)
                    )
                    trace_entry["refiner_alignment_v3_apply_require_improve"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_require_improve", False)
                    )
                    trace_entry["refiner_alignment_v3_apply_count"] = int(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_count", 0) or 0
                    )
                    trace_entry["refiner_alignment_v3_apply_applied"] = bool(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_applied", False)
                    )
                    trace_entry["refiner_alignment_v3_apply_block_reason"] = str(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_block_reason", "disabled")
                    )
                    trace_entry["refiner_alignment_v3_apply_local_delta"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_local_delta", None)
                    )
                    trace_entry["refiner_alignment_v3_apply_world_delta"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_world_delta", None)
                    )
                    trace_entry["refiner_alignment_v3_apply_pos_norm"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_pos_norm", None)
                    )
                    trace_entry["refiner_alignment_v3_apply_yaw_abs"] = _jsonable_value(
                        refiner_stats_snapshot.get("current_alignment_v3_apply_yaw_abs", None)
                    )
                    # v2 shadow fields
                    trace_entry["refiner_v2_shadow_active"] = bool(
                        refiner_stats_snapshot.get("current_v2_shadow_active", False)
                    )
                    trace_entry["refiner_v2_selected_candidate_index"] = int(
                        refiner_stats_snapshot.get("current_v2_selected_candidate_index", -1) or -1
                    )
                    trace_entry["refiner_v2_gate_pass"] = bool(
                        refiner_stats_snapshot.get("current_v2_gate_pass", False)
                    )
                    trace_entry["refiner_v2_stage_bucket"] = str(
                        refiner_stats_snapshot.get("current_v2_stage_bucket", "unknown")
                    )
                    def _jsonable_trace_value(value):
                        if isinstance(value, np.ndarray):
                            return value.tolist()
                        if isinstance(value, np.generic):
                            return value.item()
                        return value

                    trace_entry["refiner_v2_selected_delta"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_selected_delta", None)
                    )
                    trace_entry["refiner_v2_geometry_improvement"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_geometry_improvement", None)
                    )
                    trace_entry["refiner_v2_topk_indices"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_topk_indices", None)
                    )
                    trace_entry["refiner_v2_selected_post_xy"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_selected_post_xy", None)
                    )
                    trace_entry["refiner_v2_selected_post_z"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_selected_post_z", None)
                    )
                    trace_entry["refiner_v2_selected_post_yaw"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_selected_post_yaw", None)
                    )
                    trace_entry["refiner_v2_apply_gate_pass"] = bool(
                        refiner_stats_snapshot.get("current_v2_apply_gate_pass", False)
                    )
                    trace_entry["refiner_v2_apply_block_reason"] = str(
                        refiner_stats_snapshot.get("current_v2_apply_block_reason", "disabled")
                    )
                    trace_entry["refiner_v2_apply_assist_scale"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_apply_assist_scale", 0.0)
                    )
                    trace_entry["refiner_v2_apply_local_delta"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_apply_local_delta", None)
                    )
                    trace_entry["refiner_v2_apply_world_delta"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_apply_world_delta", None)
                    )
                    # v2 delta source trace
                    trace_entry["refiner_v2_delta_source"] = str(
                        refiner_stats_snapshot.get("v2_delta_source", "none")
                    )
                    trace_entry["refiner_v2_delta_norm"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("v2_delta_norm", 0.0)
                    )
                    trace_entry["refiner_v2_post_candidate_delta"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_post_delta", None)
                    )
                    trace_entry["refiner_v2_cur_xy"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("v2_cur_xy", None)
                    )
                    trace_entry["refiner_v2_cur_z"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("v2_cur_z", None)
                    )
                    trace_entry["refiner_v2_cur_yaw"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("v2_cur_yaw", None)
                    )
                    trace_entry["refiner_v2_gripper_pose_present"] = bool(
                        refiner_stats_snapshot.get("v2_gripper_pose_present", False)
                    )
                    trace_entry["refiner_v2_motion_target_pose_present"] = bool(
                        refiner_stats_snapshot.get("v2_motion_target_pose_present", False)
                    )
                    trace_entry["refiner_v2_motion_target_delta_present"] = bool(
                        refiner_stats_snapshot.get("v2_motion_target_delta_present", False)
                    )
                    trace_entry["refiner_v2_current_delta_basin_present"] = bool(
                        refiner_stats_snapshot.get("v2_current_delta_basin_present", False)
                    )
                    trace_entry["refiner_v2_predictor_micro_assist_enabled"] = bool(
                        refiner_stats_snapshot.get("v2_predictor_micro_assist_enabled", False)
                    )
                    trace_entry["refiner_v2_predictor_micro_assist_eval_count"] = int(
                        refiner_stats_snapshot.get("v2_predictor_micro_assist_eval_count", 0) or 0
                    )
                    trace_entry["refiner_v2_predictor_micro_assist_apply_count"] = int(
                        refiner_stats_snapshot.get("v2_predictor_micro_assist_apply_count", 0) or 0
                    )
                    trace_entry["refiner_v2_predictor_micro_assist_applied"] = bool(
                        refiner_stats_snapshot.get("current_v2_predictor_micro_assist_applied", False)
                    )
                    trace_entry["refiner_v2_predictor_micro_assist_block_reason"] = str(
                        refiner_stats_snapshot.get("current_v2_predictor_micro_assist_block_reason", "disabled")
                    )
                    trace_entry["refiner_v2_predictor_micro_assist_pos_norm"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_predictor_micro_assist_pos_norm", 0.0)
                    )
                    trace_entry["refiner_v2_predictor_micro_assist_rot_norm"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_v2_predictor_micro_assist_rot_norm", 0.0)
                    )
                    trace_entry["refiner_v2_predictor_micro_assist_source"] = str(
                        refiner_stats_snapshot.get("current_v2_predictor_micro_assist_source", "none")
                    )
                    trace_entry["refiner_target_delta_servo_enabled"] = bool(
                        refiner_stats_snapshot.get("current_target_delta_servo_enabled", False)
                    )
                    trace_entry["refiner_target_delta_servo_shadow_enabled"] = bool(
                        refiner_stats_snapshot.get("current_target_delta_servo_shadow_enabled", False)
                    )
                    trace_entry["refiner_target_delta_servo_apply_enabled"] = bool(
                        refiner_stats_snapshot.get("current_target_delta_servo_apply_enabled", False)
                    )
                    trace_entry["refiner_target_delta_servo_bypass_gates"] = bool(
                        refiner_stats_snapshot.get("current_target_delta_servo_bypass_gates", False)
                    )
                    trace_entry["refiner_target_delta_servo_apply_once_per_episode"] = bool(
                        refiner_stats_snapshot.get("current_target_delta_servo_apply_once_per_episode", False)
                    )
                    trace_entry["refiner_target_delta_servo_source_mode"] = str(
                        refiner_stats_snapshot.get("current_target_delta_servo_source_mode", "predictor")
                    )
                    trace_entry["refiner_target_delta_servo_k_xy"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_k_xy", 0.08)
                    )
                    trace_entry["refiner_target_delta_servo_k_z"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_k_z", 0.06)
                    )
                    trace_entry["refiner_target_delta_servo_k_yaw"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_k_yaw", 0.04)
                    )
                    trace_entry["refiner_target_delta_servo_max_pos"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_max_pos", 0.0010)
                    )
                    trace_entry["refiner_target_delta_servo_max_yaw"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_max_yaw", 0.0040)
                    )
                    trace_entry["refiner_target_delta_servo_eval_count"] = int(
                        refiner_stats_snapshot.get("current_target_delta_servo_eval_count", 0) or 0
                    )
                    trace_entry["refiner_target_delta_servo_apply_count"] = int(
                        refiner_stats_snapshot.get("current_target_delta_servo_apply_count", 0) or 0
                    )
                    trace_entry["refiner_target_delta_servo_applied"] = bool(
                        refiner_stats_snapshot.get("current_target_delta_servo_applied", False)
                    )
                    trace_entry["refiner_target_delta_servo_block_reason"] = str(
                        refiner_stats_snapshot.get("current_target_delta_servo_block_reason", "disabled")
                    )
                    trace_entry["refiner_target_delta_servo_source"] = str(
                        refiner_stats_snapshot.get("current_target_delta_servo_source", "none")
                    )
                    trace_entry["refiner_target_delta_servo_local_delta"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_local_delta", None)
                    )
                    trace_entry["refiner_target_delta_servo_world_delta"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_world_delta", None)
                    )
                    trace_entry["refiner_target_delta_servo_pos_norm"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_pos_norm", 0.0)
                    )
                    trace_entry["refiner_target_delta_servo_rot_norm"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_rot_norm", 0.0)
                    )
                    trace_entry["refiner_target_delta_servo_cur_xy"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_cur_xy", None)
                    )
                    trace_entry["refiner_target_delta_servo_cur_z"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_cur_z", None)
                    )
                    trace_entry["refiner_target_delta_servo_cur_yaw"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_cur_yaw", None)
                    )
                    trace_entry["refiner_target_delta_servo_post_xy"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_post_xy", None)
                    )
                    trace_entry["refiner_target_delta_servo_post_z"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_post_z", None)
                    )
                    trace_entry["refiner_target_delta_servo_post_yaw"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_target_delta_servo_post_yaw", None)
                    )
                    trace_entry["refiner_target_delta_servo_gate_pass"] = bool(
                        refiner_stats_snapshot.get("current_target_delta_servo_gate_pass", False)
                    )
                    trace_entry["refiner_raw_delta_pose_local"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_raw_delta_pose_local", None)
                    )
                    trace_entry["refiner_preclip_delta_pose_local"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_preclip_delta_pose_local", None)
                    )
                    trace_entry["refiner_clipped_delta_pose_local"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_clipped_delta_pose_local", None)
                    )
                    trace_entry["refiner_delta_pose_world"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_delta_pose_world", None)
                    )
                    trace_entry["refiner_raw_residual_pos_norm"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_raw_residual_pos_norm", None)
                    )
                    trace_entry["refiner_preclip_residual_pos_norm"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_preclip_residual_pos_norm", None)
                    )
                    trace_entry["refiner_clipped_residual_pos_norm"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_clipped_residual_pos_norm", None)
                    )
                    trace_entry["refiner_raw_residual_yaw_abs"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_raw_residual_yaw_abs", None)
                    )
                    trace_entry["refiner_preclip_residual_yaw_abs"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_preclip_residual_yaw_abs", None)
                    )
                    trace_entry["refiner_clipped_residual_yaw_abs"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_clipped_residual_yaw_abs", None)
                    )
                    trace_entry["refiner_learned_residual_scale"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_learned_residual_scale", None)
                    )
                    trace_entry["refiner_v2_assist_enabled"] = bool(
                        refiner_stats_snapshot.get("v2_assist_enabled", False)
                    )
                    trace_entry["refiner_v2_assist_applied"] = bool(
                        refiner_stats_snapshot.get("v2_assist_applied", False)
                    )
                    trace_entry["refiner_v2_assist_apply_count"] = int(
                        refiner_stats_snapshot.get("v2_assist_apply_count", 0)
                    )
                    trace_entry["refiner_current_handoff_metrics"] = dict(
                        refiner_stats_snapshot.get("current_handoff_metrics", {}) or {}
                    )
                    trace_entry["refiner_current_handoff_spec_name"] = refiner_stats_snapshot.get(
                        "current_handoff_spec_name", None
                    )
                    trace_entry["refiner_current_handoff_target_role"] = refiner_stats_snapshot.get(
                        "current_handoff_target_role", None
                    )
                    trace_entry["refiner_current_delta_basin_target"] = _jsonable_trace_value(
                        refiner_stats_snapshot.get("current_delta_basin_target", None)
                    )
                    active_alignment_controller = _active_alignment_like_controller(refiner)
                    _mt_delta = (
                        None if active_alignment_controller is None
                        else getattr(active_alignment_controller, "_runtime_motion_target_delta_local", None)
                    )
                    trace_entry["refiner_motion_target_delta_local"] = (
                        _mt_delta.tolist() if hasattr(_mt_delta, "tolist") else _mt_delta
                    )
                    trace_entry["refiner_has_canonical_basin"] = (
                        False
                        if active_alignment_controller is None
                        else getattr(active_alignment_controller, "_canonical_basin_center_pose_7d", None) is not None
                    )
                    trace_entry["runtime_motion_target_pose_source"] = (
                        None if active_alignment_controller is None
                        else getattr(active_alignment_controller, "_runtime_motion_target_pose_source", "unknown")
                    )
                    trace_entry["runtime_motion_target_delta_source"] = (
                        None if active_alignment_controller is None
                        else getattr(active_alignment_controller, "_runtime_motion_target_delta_source", "unknown")
                    )
                    trace_entry["refiner_current_basin_distance_runtime"] = refiner_stats_snapshot.get(
                        "current_basin_distance_runtime", None
                    )
                    trace_entry["refiner_current_basin_xy_runtime"] = refiner_stats_snapshot.get(
                        "current_basin_xy_runtime", None
                    )
                    trace_entry["refiner_current_basin_abs_z_runtime"] = refiner_stats_snapshot.get(
                        "current_basin_abs_z_runtime", None
                    )
                    trace_entry["refiner_current_basin_yaw_runtime"] = refiner_stats_snapshot.get(
                        "current_basin_yaw_runtime", None
                    )
                    trace_entry["refiner_substage_id"] = int(refiner_stats_snapshot.get("substage_id", 0))
                    trace_entry["refiner_substage"] = refiner_stats_snapshot.get("substage", "TRANSIT")
                    trace_entry["refiner_has_object_in_hand"] = bool(refiner_stats_snapshot.get("has_object_in_hand", False))
                    trace_entry["refiner_contact_state_id"] = int(refiner_stats_snapshot.get("contact_state_id", 0))
                    trace_entry["refiner_contact_state"] = refiner_stats_snapshot.get("contact_state", "FREE_SPACE")
                    trace_entry["refiner_stage_target_mode_id"] = int(refiner_stats_snapshot.get("stage_target_mode_id", 0))
                    trace_entry["refiner_stage_target_mode"] = refiner_stats_snapshot.get("stage_target_mode", "NONE")
                    trace_entry["refiner_target_provider_name"] = refiner_stats_snapshot.get("current_target_provider_name", None)
                    trace_entry["refiner_target_provider_source"] = refiner_stats_snapshot.get("current_target_provider_source", None)
                    trace_entry["refiner_target_uses_privileged_runtime"] = bool(
                        refiner_stats_snapshot.get("current_target_uses_privileged_runtime", False)
                    )
                    trace_entry["refiner_last_scorer_candidate_index"] = int(
                        refiner_stats_snapshot.get("last_scorer_candidate_index", -1)
                    )
                    trace_entry["refiner_last_scorer_group_index"] = int(
                        refiner_stats_snapshot.get("last_scorer_group_index", -1)
                    )
                    trace_entry["refiner_last_selected_step_scale"] = float(
                        refiner_stats_snapshot.get("last_selected_step_scale", 1.0)
                    )
                    trace_entry["refiner_last_handoff_yaw_priority_active"] = bool(
                        refiner_stats_snapshot.get("last_handoff_yaw_priority_active", False)
                    )
                    trace_entry["refiner_last_handoff_axis_priority"] = str(
                        refiner_stats_snapshot.get("last_handoff_axis_priority", "none")
                    )
                    trace_entry["refiner_last_near_ready_rerank_gate_open"] = bool(
                        refiner_stats_snapshot.get("last_near_ready_rerank_gate_open", False)
                    )
                    trace_entry["refiner_last_near_ready_rerank_applied"] = bool(
                        refiner_stats_snapshot.get("last_near_ready_rerank_applied", False)
                    )
                    trace_entry["refiner_last_near_ready_rerank_changed"] = bool(
                        refiner_stats_snapshot.get("last_near_ready_rerank_changed", False)
                    )
                    trace_entry["refiner_last_near_ready_rerank_prev_index"] = int(
                        refiner_stats_snapshot.get("last_near_ready_rerank_prev_index", -1)
                    )
                    trace_entry["refiner_last_near_ready_rerank_new_index"] = int(
                        refiner_stats_snapshot.get("last_near_ready_rerank_new_index", -1)
                    )
                    trace_entry["refiner_last_near_ready_rerank_topk"] = int(
                        refiner_stats_snapshot.get("last_near_ready_rerank_topk", 0)
                    )
                    trace_entry["refiner_last_near_ready_rerank_cur_xy"] = refiner_stats_snapshot.get(
                        "last_near_ready_rerank_cur_xy", None
                    )
                    trace_entry["refiner_last_near_ready_rerank_cur_z"] = refiner_stats_snapshot.get(
                        "last_near_ready_rerank_cur_z", None
                    )
                    trace_entry["refiner_last_near_ready_rerank_cur_yaw"] = refiner_stats_snapshot.get(
                        "last_near_ready_rerank_cur_yaw", None
                    )
                    trace_entry["refiner_last_near_ready_rerank_gate_xy_max"] = refiner_stats_snapshot.get(
                        "last_near_ready_rerank_gate_xy_max", None
                    )
                    trace_entry["refiner_last_near_ready_rerank_gate_z_max"] = refiner_stats_snapshot.get(
                        "last_near_ready_rerank_gate_z_max", None
                    )
                    trace_entry["refiner_last_near_ready_rerank_gate_yaw_max"] = refiner_stats_snapshot.get(
                        "last_near_ready_rerank_gate_yaw_max", None
                    )
                    trace_entry["refiner_last_near_ready_specialist_gate_open"] = bool(
                        refiner_stats_snapshot.get("last_near_ready_specialist_gate_open", False)
                    )
                    trace_entry["refiner_last_near_ready_specialist_active"] = bool(
                        refiner_stats_snapshot.get("last_near_ready_specialist_active", False)
                    )
                    trace_entry["refiner_last_near_ready_specialist_cur_xy"] = refiner_stats_snapshot.get(
                        "last_near_ready_specialist_cur_xy", None
                    )
                    trace_entry["refiner_last_near_ready_specialist_cur_z"] = refiner_stats_snapshot.get(
                        "last_near_ready_specialist_cur_z", None
                    )
                    trace_entry["refiner_last_near_ready_specialist_cur_yaw"] = refiner_stats_snapshot.get(
                        "last_near_ready_specialist_cur_yaw", None
                    )
                    trace_entry["refiner_last_near_ready_specialist_gate_xy_max"] = refiner_stats_snapshot.get(
                        "last_near_ready_specialist_gate_xy_max", None
                    )
                    trace_entry["refiner_last_near_ready_specialist_gate_z_max"] = refiner_stats_snapshot.get(
                        "last_near_ready_specialist_gate_z_max", None
                    )
                    trace_entry["refiner_last_near_ready_specialist_gate_yaw_max"] = refiner_stats_snapshot.get(
                        "last_near_ready_specialist_gate_yaw_max", None
                    )
                    trace_entry["refiner_near_ready_group_residual_eval_count"] = int(
                        refiner_stats_snapshot.get("near_ready_group_residual_eval_count", 0)
                    )
                    trace_entry["refiner_near_ready_group_residual_gate_pass_count"] = int(
                        refiner_stats_snapshot.get("near_ready_group_residual_gate_pass_count", 0)
                    )
                    trace_entry["refiner_near_ready_group_residual_apply_count"] = int(
                        refiner_stats_snapshot.get("near_ready_group_residual_apply_count", 0)
                    )
                    trace_entry["refiner_near_ready_group_residual_change_count"] = int(
                        refiner_stats_snapshot.get("near_ready_group_residual_change_count", 0)
                    )
                    trace_entry["refiner_last_near_ready_group_residual_gate_open"] = bool(
                        refiner_stats_snapshot.get("last_near_ready_group_residual_gate_open", False)
                    )
                    trace_entry["refiner_last_near_ready_group_residual_applied"] = bool(
                        refiner_stats_snapshot.get("last_near_ready_group_residual_applied", False)
                    )
                    trace_entry["refiner_last_near_ready_group_residual_changed"] = bool(
                        refiner_stats_snapshot.get("last_near_ready_group_residual_changed", False)
                    )
                    trace_entry["refiner_last_near_ready_group_residual_prev_group"] = int(
                        refiner_stats_snapshot.get("last_near_ready_group_residual_prev_group", -1)
                    )
                    trace_entry["refiner_last_near_ready_group_residual_new_group"] = int(
                        refiner_stats_snapshot.get("last_near_ready_group_residual_new_group", -1)
                    )
                    trace_entry["b1_group_shadow_eval_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_eval_count", 0)
                    )
                    trace_entry["b1_group_shadow_gate_mode"] = str(
                        refiner_stats_snapshot.get("b1_group_shadow_gate_mode", "broad")
                    )
                    trace_entry["b1_group_shadow_gate_pass_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_gate_pass_count", 0)
                    )
                    trace_entry["b1_group_shadow_change_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_change_count", 0)
                    )
                    trace_entry["b1_group_shadow_disagreement_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_disagreement_count", 0)
                    )
                    trace_entry["b1_group_shadow_teacher_group_valid_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_teacher_group_valid_count", 0)
                    )
                    trace_entry["b1_group_shadow_cost_valid_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_cost_valid_count", 0)
                    )
                    trace_entry["b1_group_shadow_cost_improve_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_cost_improve_count", 0)
                    )
                    trace_entry["b1_group_shadow_cost_worse_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_cost_worse_count", 0)
                    )
                    trace_entry["b1_group_shadow_regret_delta_sum"] = float(
                        refiner_stats_snapshot.get("b1_group_shadow_regret_delta_sum", 0.0)
                    )
                    trace_entry["b1_group_shadow_close_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_close_count", 0)
                    )
                    trace_entry["b1_group_shadow_close_change_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_close_change_count", 0)
                    )
                    trace_entry["b1_group_shadow_close_cost_valid_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_close_cost_valid_count", 0)
                    )
                    trace_entry["b1_group_shadow_close_cost_improve_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_close_cost_improve_count", 0)
                    )
                    trace_entry["b1_group_shadow_close_cost_worse_count"] = int(
                        refiner_stats_snapshot.get("b1_group_shadow_close_cost_worse_count", 0)
                    )
                    trace_entry["b1_group_shadow_close_regret_delta_sum"] = float(
                        refiner_stats_snapshot.get("b1_group_shadow_close_regret_delta_sum", 0.0)
                    )
                    trace_entry["b1_apply_gate_eval_count"] = int(
                        refiner_stats_snapshot.get("b1_apply_gate_eval_count", 0)
                    )
                    trace_entry["b1_apply_gate_apply_count"] = int(
                        refiner_stats_snapshot.get("b1_apply_gate_apply_count", 0)
                    )
                    trace_entry["b1_apply_gate_cost_valid_count"] = int(
                        refiner_stats_snapshot.get("b1_apply_gate_cost_valid_count", 0)
                    )
                    trace_entry["b1_apply_gate_cost_improve_count"] = int(
                        refiner_stats_snapshot.get("b1_apply_gate_cost_improve_count", 0)
                    )
                    trace_entry["b1_apply_gate_cost_worse_count"] = int(
                        refiner_stats_snapshot.get("b1_apply_gate_cost_worse_count", 0)
                    )
                    trace_entry["b1_apply_gate_regret_delta_sum"] = float(
                        refiner_stats_snapshot.get("b1_apply_gate_regret_delta_sum", 0.0)
                    )
                    trace_entry["b1_apply_gate_close_cost_valid_count"] = int(
                        refiner_stats_snapshot.get("b1_apply_gate_close_cost_valid_count", 0)
                    )
                    trace_entry["b1_apply_gate_close_cost_improve_count"] = int(
                        refiner_stats_snapshot.get("b1_apply_gate_close_cost_improve_count", 0)
                    )
                    trace_entry["b1_apply_gate_close_cost_worse_count"] = int(
                        refiner_stats_snapshot.get("b1_apply_gate_close_cost_worse_count", 0)
                    )
                    trace_entry["b1_apply_gate_close_regret_delta_sum"] = float(
                        refiner_stats_snapshot.get("b1_apply_gate_close_regret_delta_sum", 0.0)
                    )
                    trace_entry["b1_group_shadow_gate_open"] = bool(
                        refiner_stats_snapshot.get("last_b1_group_shadow_gate_open", False)
                    )
                    trace_entry["b1_group_shadow_pred_group"] = int(
                        refiner_stats_snapshot.get("last_b1_group_shadow_pred_group", -1)
                    )
                    trace_entry["b1_group_shadow_baseline_group"] = int(
                        refiner_stats_snapshot.get("last_b1_group_shadow_baseline_group", -1)
                    )
                    trace_entry["b1_group_shadow_teacher_group"] = int(
                        refiner_stats_snapshot.get("last_b1_group_shadow_teacher_group", -1)
                    )
                    trace_entry["b1_group_shadow_changed"] = bool(
                        refiner_stats_snapshot.get("last_b1_group_shadow_changed", False)
                    )
                    trace_entry["b1_group_shadow_teacher_disagree"] = bool(
                        refiner_stats_snapshot.get("last_b1_group_shadow_teacher_disagree", False)
                    )
                    trace_entry["b1_group_shadow_teacher_group_valid"] = bool(
                        refiner_stats_snapshot.get("last_b1_group_shadow_teacher_group_valid", False)
                    )
                    trace_entry["b1_group_shadow_close_neighborhood"] = bool(
                        refiner_stats_snapshot.get("last_b1_group_shadow_close_neighborhood", False)
                    )
                    trace_entry["b1_group_shadow_close_group_changed"] = bool(
                        refiner_stats_snapshot.get("last_b1_group_shadow_close_group_changed", False)
                    )
                    trace_entry["b1_group_shadow_margin"] = float(
                        refiner_stats_snapshot.get("last_b1_group_shadow_margin", np.nan)
                    )
                    trace_entry["b1_group_shadow_teacher_best_cost"] = float(
                        refiner_stats_snapshot.get("last_b1_group_shadow_teacher_best_cost", np.nan)
                    )
                    trace_entry["b1_group_shadow_baseline_group_cost"] = float(
                        refiner_stats_snapshot.get("last_b1_group_shadow_baseline_group_cost", np.nan)
                    )
                    trace_entry["b1_group_shadow_pred_group_cost"] = float(
                        refiner_stats_snapshot.get("last_b1_group_shadow_pred_group_cost", np.nan)
                    )
                    trace_entry["b1_group_shadow_baseline_group_regret"] = float(
                        refiner_stats_snapshot.get("last_b1_group_shadow_baseline_group_regret", np.nan)
                    )
                    trace_entry["b1_group_shadow_pred_group_regret"] = float(
                        refiner_stats_snapshot.get("last_b1_group_shadow_pred_group_regret", np.nan)
                    )
                    trace_entry["b1_group_shadow_regret_delta"] = float(
                        refiner_stats_snapshot.get("last_b1_group_shadow_regret_delta", np.nan)
                    )
                    trace_entry["b1_apply_gate_prob"] = float(
                        refiner_stats_snapshot.get("last_b1_apply_gate_prob", np.nan)
                    )
                    trace_entry["b1_apply_gate_threshold"] = float(
                        refiner_stats_snapshot.get("last_b1_apply_gate_threshold", np.nan)
                    )
                    trace_entry["b1_apply_gate_apply"] = bool(
                        refiner_stats_snapshot.get("last_b1_apply_gate_apply", False)
                    )
                    trace_entry["b1_apply_gate_vetoed"] = bool(
                        refiner_stats_snapshot.get("last_b1_apply_gate_vetoed", False)
                    )
                    trace_entry["b1_group_bounded_applied"] = bool(
                        refiner_stats_snapshot.get("last_b1_group_bounded_applied", False)
                    )
                    trace_entry["b1_group_bounded_prev_group"] = int(
                        refiner_stats_snapshot.get("last_b1_group_bounded_prev_group", -1)
                    )
                    trace_entry["b1_group_bounded_new_group"] = int(
                        refiner_stats_snapshot.get("last_b1_group_bounded_new_group", -1)
                    )
                    for key in (
                        "b2_candidate_shadow_eval_count",
                        "b2_candidate_shadow_gate_pass_count",
                        "b2_candidate_shadow_change_count",
                        "b2_candidate_shadow_cost_valid_count",
                        "b2_candidate_shadow_cost_improve_count",
                        "b2_candidate_shadow_cost_worse_count",
                        "b2_candidate_shadow_mode_keep_count",
                        "b2_candidate_shadow_mode_apply_count",
                        "b2_candidate_shadow_close_count",
                        "b2_candidate_shadow_yaw_needed_count",
                        "b2_candidate_shadow_yaw_keep_count",
                        "b2_candidate_shadow_teacher_ready_count",
                        "b2_candidate_shadow_xy_block_count",
                        "b2_candidate_shadow_nearish_count",
                        "b2_candidate_shadow_keep_baseline_forced_count",
                        "b2_candidate_bounded_eval_count",
                        "b2_candidate_bounded_gate_pass_count",
                        "b2_candidate_bounded_apply_count",
                        "b2_candidate_bounded_change_count",
                    ):
                        trace_entry[key] = int(refiner_stats_snapshot.get(key, 0))
                    trace_entry["b2_candidate_shadow_gate_mode"] = str(
                        refiner_stats_snapshot.get("b2_candidate_shadow_gate_mode", "broad")
                    )
                    trace_entry["b2_candidate_shadow_regret_delta_sum"] = float(
                        refiner_stats_snapshot.get("b2_candidate_shadow_regret_delta_sum", 0.0)
                    )
                    for key in (
                        "gate_open",
                        "close_neighborhood",
                        "changed",
                        "yaw_needed",
                        "yaw_keep",
                        "teacher_ready",
                        "xy_block",
                        "nearish_runtime",
                        "keep_baseline_forced",
                    ):
                        trace_entry[f"b2_candidate_shadow_{key}"] = bool(
                            refiner_stats_snapshot.get(f"last_b2_candidate_shadow_{key}", False)
                        )
                    for key in (
                        "gate_open",
                        "applied",
                        "changed",
                    ):
                        trace_entry[f"b2_candidate_bounded_{key}"] = bool(
                            refiner_stats_snapshot.get(f"last_b2_candidate_bounded_{key}", False)
                        )
                    for key in (
                        "mode",
                        "baseline_index",
                        "pred_index",
                        "best_index",
                        "runtime_scope_size",
                        "small_yaw_scope_size",
                        "large_yaw_scope_size",
                        "probe_count",
                    ):
                        trace_entry[f"b2_candidate_shadow_{key}"] = int(
                            refiner_stats_snapshot.get(f"last_b2_candidate_shadow_{key}", -1)
                        )
                    for key in (
                        "prev_index",
                        "new_index",
                    ):
                        trace_entry[f"b2_candidate_bounded_{key}"] = int(
                            refiner_stats_snapshot.get(f"last_b2_candidate_bounded_{key}", -1)
                        )
                    for key in (
                        "mode_confidence",
                        "mode_margin",
                        "best_cost",
                        "baseline_cost",
                        "pred_cost",
                        "baseline_regret",
                        "pred_regret",
                        "regret_delta",
                    ):
                        trace_entry[f"b2_candidate_shadow_{key}"] = float(
                            refiner_stats_snapshot.get(f"last_b2_candidate_shadow_{key}", np.nan)
                        )
                    for key in (
                        "mode_confidence",
                        "mode_margin",
                    ):
                        trace_entry[f"b2_candidate_bounded_{key}"] = float(
                            refiner_stats_snapshot.get(f"last_b2_candidate_bounded_{key}", np.nan)
                        )
                    for key in (
                        "candidate_actions_local",
                        "candidate_scope_mask",
                        "candidate_valid_mask",
                        "candidate_cost",
                        "candidate_oracle_score",
                    ):
                        value = refiner_stats_snapshot.get(f"last_b2_candidate_shadow_{key}", None)
                        if value is not None:
                            trace_entry[f"b2_candidate_shadow_{key}"] = value
                    trace_entry["refiner_alignment_gate_block_reason_counts"] = refiner_stats_snapshot.get(
                        "alignment_gate_block_reason_counts", {}
                    )
                    controller_for_support = active_alignment_controller
                    support_motion_target_pose = None
                    support_motion_target_delta = None
                    if runtime_target_pose is not None:
                        support_motion_target_pose = np.asarray(runtime_target_pose, dtype=np.float32)
                    elif getattr(controller_for_support, "_runtime_motion_target_pose_7d", None) is not None:
                        support_motion_target_pose = np.asarray(
                            getattr(controller_for_support, "_runtime_motion_target_pose_7d"), dtype=np.float32
                        )
                    elif getattr(controller_for_support, "_canonical_basin_center_pose_7d", None) is not None:
                        support_motion_target_pose = np.asarray(
                            getattr(controller_for_support, "_canonical_basin_center_pose_7d"), dtype=np.float32
                        )
                    if current_delta_basin is not None:
                        support_motion_target_delta = np.asarray(current_delta_basin, dtype=np.float32)
                    elif getattr(controller_for_support, "_runtime_motion_target_delta_local", None) is not None:
                        support_motion_target_delta = np.asarray(
                            getattr(controller_for_support, "_runtime_motion_target_delta_local"), dtype=np.float32
                        )
                    else:
                        support_motion_target_delta = np.zeros((6,), dtype=np.float32)
                    if support_motion_target_pose is None:
                        support_motion_target_pose = np.asarray(
                            apply_local_offset_to_pose(np.asarray(obs.gripper_pose, dtype=np.float32), support_motion_target_delta),
                            dtype=np.float32,
                        )

                    dump_support_state = (
                        support_state_rows is not None
                        and int(refiner_stats_snapshot.get("phase_id", 1)) == int(StagePhase.ALIGN)
                        and float(obs.gripper_open) >= 0.5
                        and support_motion_target_pose is not None
                    )
                    if dump_support_state:
                        if oracle_collect_active:
                            candidate_actions = np.asarray(oracle_collect_candidate_actions, dtype=np.float32)
                            candidate_mask = np.asarray(teacher_candidate_mask, dtype=np.float32)
                            candidate_kind = np.asarray(teacher_candidate_kind)
                            candidate_group_index_row = np.asarray(teacher_candidate_group_index_all, dtype=np.int64)
                            oracle_idx = int(oracle_collect_idx)
                            oracle_score = float(oracle_collect_score)
                            oracle_current_dist = float(oracle_collect_current_dist)
                            oracle_next_dists = np.asarray(oracle_collect_next_dists, dtype=np.float32)
                            oracle_scores = np.asarray(oracle_collect_scores, dtype=np.float32)
                            debug_pred = oracle_collect_debug_pred
                        else:
                            controller_for_support = _active_alignment_like_controller(refiner)
                            candidate_actions_src = refiner_stats_snapshot.get("last_b2_candidate_shadow_candidate_actions_local", None)
                            if candidate_actions_src is None and controller_for_support is not None:
                                candidate_actions_src = getattr(controller_for_support, "_teacher_candidate_actions_local", None)
                                if candidate_actions_src is None and getattr(controller_for_support, "_candidate_actions_local", None) is not None:
                                    candidate_actions_src = controller_for_support._candidate_actions_local.detach().cpu().numpy()
                            if candidate_actions_src is None:
                                candidate_actions_src = np.zeros((1, 6), dtype=np.float32)
                            candidate_actions = np.asarray(candidate_actions_src, dtype=np.float32)

                            candidate_mask_src = refiner_stats_snapshot.get("last_b2_candidate_shadow_candidate_valid_mask", None)
                            if candidate_mask_src is None:
                                candidate_mask_src = refiner_stats_snapshot.get("last_b2_candidate_shadow_candidate_scope_mask", None)
                            if candidate_mask_src is None and controller_for_support is not None:
                                candidate_mask_src = getattr(controller_for_support, "_runtime_candidate_mask", None)
                            if candidate_mask_src is None:
                                candidate_mask_src = np.ones((candidate_actions.shape[0],), dtype=np.float32)
                            candidate_mask = np.asarray(candidate_mask_src, dtype=np.float32)

                            candidate_kind = np.asarray(["base"] * candidate_actions.shape[0])
                            candidate_group_index_row = np.asarray(
                                refiner_stats_snapshot.get(
                                    "last_b2_candidate_shadow_candidate_group_index",
                                    np.zeros((candidate_actions.shape[0],), dtype=np.int64),
                                ),
                                dtype=np.int64,
                            )
                            oracle_idx, oracle_score, oracle_current_dist, oracle_next_dists, oracle_scores = scorer_oracle(
                                current_pose_7d=np.asarray(obs.gripper_pose, dtype=np.float32),
                                pregrasp_target_pose=np.asarray(support_motion_target_pose, dtype=np.float32),
                                grasp_commit_target_pose=np.asarray(support_motion_target_pose, dtype=np.float32),
                                candidate_actions_local=candidate_actions,
                                base_action_local=np.asarray(planner_base_action_local_raw, dtype=np.float32),
                                depth_proximity=depth_proximity,
                                candidate_mask=candidate_mask,
                                candidate_kind=candidate_kind,
                                close_xy_threshold=float(getattr(args, "teacher_close_xy_threshold", 0.020)),
                                close_abs_z_threshold=float(getattr(args, "teacher_close_abs_z_threshold", 0.020)),
                                close_yaw_threshold=float(getattr(args, "teacher_close_yaw_threshold", 0.12)),
                                commit_switch_xy_threshold=float(getattr(args, "teacher_commit_switch_xy_threshold", 0.010)),
                                commit_switch_z_threshold=float(getattr(args, "teacher_commit_switch_z_threshold", 0.020)),
                                commit_switch_yaw_threshold=float(getattr(args, "teacher_commit_switch_yaw_threshold", 0.12)),
                                stall_count=int(
                                    getattr(controller_for_support, "_teacher_pregrasp_stall_count", 0)
                                    if controller_for_support is not None
                                    else 0
                                ),
                                orientation_rescue_active=bool(teacher_enable_orientation_rescue),
                                yaw_symmetry_period=float(getattr(target_result, "yaw_symmetry_period", -1.0)),
                            )
                            debug_pred = (
                                scorer_forward_debug(
                                    controller_for_support,
                                    np.asarray(obs.front_rgb, dtype=np.uint8),
                                    np.asarray(obs.wrist_rgb, dtype=np.uint8),
                                    depth_tensor_96.detach().cpu().numpy().astype(np.float32),
                                    np.asarray(proprio, dtype=np.float32),
                                    np.asarray(planner_base_action_local_raw, dtype=np.float32),
                                    chunk_step,
                                    gripper_context_arr,
                                )
                                if controller_for_support is not None
                                else None
                            )
                        if debug_pred is None:
                            debug_pred = (
                                scorer_forward_debug(
                                    controller_for_support,
                                    np.asarray(obs.front_rgb, dtype=np.uint8),
                                    np.asarray(obs.wrist_rgb, dtype=np.uint8),
                                    depth_tensor_96.detach().cpu().numpy().astype(np.float32),
                                    np.asarray(proprio, dtype=np.float32),
                                    np.asarray(planner_base_action_local_raw, dtype=np.float32),
                                    chunk_step,
                                    gripper_context_arr,
                                )
                                if controller_for_support is not None
                                else None
                            )
                        current_delta = np.asarray(
                            support_motion_target_delta,
                            dtype=np.float32,
                        )
                        oracle_action_local_row = np.asarray(candidate_actions[int(oracle_idx)], dtype=np.float32)
                        residual_label_local = (
                            oracle_action_local_row - np.asarray(planner_base_action_local_raw, dtype=np.float32)
                        ).astype(np.float32)
                        residual_label_world = local_delta_to_world(
                            residual_label_local,
                            np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                        ).astype(np.float32)
                        support_state_rows.append(
                            {
                                "mode_index": np.asarray(1, dtype=np.int64),
                                "episode_index": np.asarray(int(ep_idx), dtype=np.int64),
                                "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
                                "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
                                "wrist_depth": depth_tensor_96.detach().cpu().numpy().astype(np.float32),
                                "force_history": np.asarray(
                                    force_hist.detach().cpu().numpy().astype(np.float32)
                                    if force_hist is not None
                                    else np.zeros((32, 6), dtype=np.float32),
                                    dtype=np.float32,
                                ),
                                "ft_hist": np.asarray(
                                    force_hist.detach().cpu().numpy().astype(np.float32)
                                    if force_hist is not None
                                    else np.zeros((32, 6), dtype=np.float32),
                                    dtype=np.float32,
                                ),
                                "gripper_touch_forces": np.asarray(
                                    raw_force if raw_force is not None else np.zeros((6,), dtype=np.float32),
                                    dtype=np.float32,
                                ),
                                "force_norm": np.asarray(
                                    float(np.linalg.norm(raw_force)) if raw_force is not None else 0.0,
                                    dtype=np.float32,
                                ),
                                "proprio": np.asarray(proprio, dtype=np.float32),
                                "base_action": np.asarray(base_delta_action[:6], dtype=np.float32),
                                "planner_base_action_local_raw": np.asarray(planner_base_action_local_raw, dtype=np.float32),
                                "planner_base_action_7d_raw": np.asarray(planner_base_action_7d_raw, dtype=np.float32),
                                "planner_base_action_gain": np.asarray(1.0, dtype=np.float32),
                                "planner_base_action_is_raw": np.asarray(1.0, dtype=np.float32),
                                "planner_base_action_frame_is_local": np.asarray(1.0, dtype=np.float32),
                                "gripper_context": np.asarray(
                                    gripper_context_arr,
                                    dtype=np.float32,
                                ),
                                "rollout_gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
                                "planner_close_intent": np.asarray(
                                    float(np.clip(base_delta_action[6], -1.0, 1.0) <= 0.5),
                                    dtype=np.float32,
                                ),
                                "depth_proximity": np.asarray(
                                    float(depth_proximity) if depth_proximity is not None else np.nan,
                                    dtype=np.float32,
                                ),
                                "wrist_depth_median": np.asarray(
                                    float(np.median(depth_tensor_96.detach().cpu().numpy())),
                                    dtype=np.float32,
                                ),
                                "step_idx": np.asarray(int(chunk_step), dtype=np.int64),
                                "rollout_step": np.asarray(int(step_idx), dtype=np.int64),
                                "phase_id": np.asarray(int(refiner_stats_snapshot.get("phase_id", 1)), dtype=np.int64),
                                "substage_id": np.asarray(int(refiner_stats_snapshot.get("substage_id", 0)), dtype=np.int64),
                                "has_object_in_hand": np.asarray(
                                    float(refiner_stats_snapshot.get("has_object_in_hand", False)), dtype=np.float32
                                ),
                                "contact_state": np.asarray(
                                    int(refiner_stats_snapshot.get("contact_state_id", 0)), dtype=np.int64
                                ),
                                "zone_state": np.asarray(str(refiner_stats_snapshot.get("zone_state", "planner_only"))),
                                "stage_target_mode": np.asarray(
                                    int(refiner_stats_snapshot.get("stage_target_mode_id", 0)), dtype=np.int64
                                ),
                                "phase_age": np.asarray(float(refiner_stats_snapshot.get("phase_age", 0.0)), dtype=np.float32),
                                "steps_since_last_replan": np.asarray(
                                    float(refiner_stats_snapshot.get("steps_since_last_replan", 0.0)),
                                    dtype=np.float32,
                                ),
                                "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                                "motion_target_pose_7d": np.asarray(support_motion_target_pose, dtype=np.float32),
                                "basin_center_pose_7d": np.asarray(support_motion_target_pose, dtype=np.float32),
                                "reference_anchor_pose_7d": np.asarray(
                                    anchor_pose if anchor_pose is not None else support_motion_target_pose,
                                    dtype=np.float32,
                                ),
                                "pregrasp_target_pose_7d": np.asarray(
                                    getattr(
                                        controller_for_support,
                                        "_runtime_pregrasp_target_pose_7d",
                                        support_motion_target_pose,
                                    )
                                    if controller_for_support is not None
                                    else support_motion_target_pose,
                                    dtype=np.float32,
                                ),
                                "grasp_commit_target_pose_7d": np.asarray(
                                    getattr(
                                        controller_for_support,
                                        "_runtime_grasp_commit_target_pose_7d",
                                        support_motion_target_pose,
                                    )
                                    if controller_for_support is not None
                                    else support_motion_target_pose,
                                    dtype=np.float32,
                                ),
                                "target_delta_teacher": np.asarray(
                                    np.asarray(teacher_delta_for_dataset, dtype=np.float32)
                                    if teacher_delta_for_dataset is not None
                                    else np.asarray(current_delta, dtype=np.float32),
                                    dtype=np.float32,
                                ),
                                "proxy_current_delta_basin_target": np.asarray(current_delta, dtype=np.float32),
                                "teacher_current_delta_basin_target": np.asarray(
                                    np.asarray(teacher_delta_for_dataset, dtype=np.float32)
                                    if teacher_delta_for_dataset is not None
                                    else np.asarray(current_delta, dtype=np.float32),
                                    dtype=np.float32,
                                ),
                                "motion_target_delta_local": np.asarray(current_delta, dtype=np.float32),
                                "teacher_source": np.asarray(
                                    float(str(getattr(target_result, "source", "none")) == "teacher_live_target"),
                                    dtype=np.float32,
                                ),
                                "target_provider_name": np.asarray(str(getattr(target_result, "provider_name", "none"))),
                                "target_provider_source": np.asarray(str(getattr(target_result, "source", "none"))),
                                "target_provider_uses_privileged": np.asarray(
                                    float(bool(getattr(target_result, "uses_privileged_target", False))),
                                    dtype=np.float32,
                                ),
                                "handoff_ready_target": np.asarray(
                                    float(getattr(target_result, "handoff_ready", False)), dtype=np.float32
                                ),
                                "teacher_truth_basin_distance": np.asarray(
                                    np.nan if teacher_truth_basin_distance is None else float(teacher_truth_basin_distance),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_basin_xy": np.asarray(
                                    np.nan if teacher_truth_basin_xy is None else float(teacher_truth_basin_xy),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_basin_z": np.asarray(
                                    np.nan if teacher_truth_basin_z is None else float(teacher_truth_basin_z),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_basin_yaw": np.asarray(
                                    np.nan if teacher_truth_basin_yaw is None else float(teacher_truth_basin_yaw),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_basin_tilt": np.asarray(
                                    np.nan if teacher_truth_basin_tilt is None else float(teacher_truth_basin_tilt),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_ready": np.asarray(
                                    float((teacher_truth_handoff or {}).get("handoff_ready", False)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_spec_name": np.asarray(
                                    str((teacher_truth_handoff or {}).get("handoff_spec_name", "none"))
                                ),
                                "teacher_truth_handoff_metric_xy_error": np.asarray(
                                    float(((teacher_truth_handoff or {}).get("handoff_metrics", {}) or {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_metric_abs_z_error": np.asarray(
                                    float(((teacher_truth_handoff or {}).get("handoff_metrics", {}) or {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_metric_yaw_error": np.asarray(
                                    float(((teacher_truth_handoff or {}).get("handoff_metrics", {}) or {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_metric_tilt_error": np.asarray(
                                    float(((teacher_truth_handoff or {}).get("handoff_metrics", {}) or {}).get("tilt_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_release_threshold_xy_error": np.asarray(
                                    float(((teacher_truth_handoff or {}).get("handoff_release_metric_thresholds", {}) or {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_release_threshold_abs_z_error": np.asarray(
                                    float(((teacher_truth_handoff or {}).get("handoff_release_metric_thresholds", {}) or {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_release_threshold_yaw_error": np.asarray(
                                    float(((teacher_truth_handoff or {}).get("handoff_release_metric_thresholds", {}) or {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_release_threshold_tilt_error": np.asarray(
                                    float(((teacher_truth_handoff or {}).get("handoff_release_metric_thresholds", {}) or {}).get("tilt_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_metric_xy_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_metric_abs_z_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_metric_yaw_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_metric_tilt_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("tilt_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_threshold_xy_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metric_thresholds", {}) or {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_threshold_abs_z_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metric_thresholds", {}) or {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_threshold_yaw_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metric_thresholds", {}) or {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_threshold_tilt_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metric_thresholds", {}) or {}).get("tilt_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_release_threshold_xy_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_release_threshold_abs_z_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_release_threshold_yaw_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_release_threshold_tilt_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}).get("tilt_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_optimization_threshold_xy_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_optimization_metric_thresholds", {}) or {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_optimization_threshold_abs_z_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_optimization_metric_thresholds", {}) or {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_optimization_threshold_yaw_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_optimization_metric_thresholds", {}) or {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_optimization_threshold_tilt_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_optimization_metric_thresholds", {}) or {}).get("tilt_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_metric_xy_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_metric_abs_z_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_metric_yaw_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_ready": np.asarray(
                                    # Deprecated alias kept for older consumers; use
                                    # runtime_handoff_ready_pred/applied in new code.
                                    float((handoff_result or {}).get("handoff_ready", False)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_ready_pred": np.asarray(
                                    float(predicted_handoff_ready),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_ready_applied": np.asarray(
                                    float(applied_handoff_ready),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_only": np.asarray(
                                    float(
                                        bool(
                                            getattr(
                                                controller_for_support,
                                                "_runtime_handoff_shadow_only",
                                                False,
                                            )
                                        )
                                    ),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_metric_valid": np.asarray(
                                    float(
                                        bool((handoff_result or {}).get("handoff_metric_valid", False))
                                        and np.isfinite(float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("xy_error", np.nan)))
                                        and np.isfinite(float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("abs_z_error", np.nan)))
                                        and np.isfinite(float(((handoff_result or {}).get("handoff_metrics", {}) or {}).get("yaw_error", np.nan)))
                                    ),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_release_threshold_xy_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_release_threshold_abs_z_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_release_threshold_yaw_error": np.asarray(
                                    float(((handoff_result or {}).get("handoff_release_metric_thresholds", {}) or {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_pred_xy_norm": np.asarray(
                                    float(handoff_aux.get("pred_xy_norm", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_pred_abs_z_norm": np.asarray(
                                    float(handoff_aux.get("pred_abs_z_norm", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_pred_yaw_norm": np.asarray(
                                    float(handoff_aux.get("pred_yaw_norm", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_pred_band_index": np.asarray(
                                    int(handoff_aux.get("pred_band_index", -1)),
                                    dtype=np.int64,
                                ),
                                "runtime_handoff_pred_ready_prob": np.asarray(
                                    float(handoff_aux.get("pred_ready_prob", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_pred_uncertainty": np.asarray(
                                    float(handoff_aux.get("pred_uncertainty", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_pred_residual_local": np.asarray(
                                    [
                                        float(handoff_aux.get("pred_residual_dx", np.nan)),
                                        float(handoff_aux.get("pred_residual_dy", np.nan)),
                                        float(handoff_aux.get("pred_residual_dz", np.nan)),
                                        float(handoff_aux.get("pred_residual_dyaw", np.nan)),
                                    ],
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_pred_residual_confidence": np.asarray(
                                    float(handoff_aux.get("pred_residual_confidence", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_valid": np.asarray(
                                    float(bool(handoff_aux.get("shadow_residual_valid", False))),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_improves": np.asarray(
                                    float(bool(handoff_aux.get("shadow_residual_improves", False))),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_cost_before": np.asarray(
                                    float(handoff_aux.get("shadow_residual_cost_before", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_cost_after": np.asarray(
                                    float(handoff_aux.get("shadow_residual_cost_after", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_cost_delta": np.asarray(
                                    float(handoff_aux.get("shadow_residual_cost_delta", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_next_xy_error": np.asarray(
                                    float(handoff_aux.get("shadow_residual_next_xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_next_abs_z_error": np.asarray(
                                    float(handoff_aux.get("shadow_residual_next_abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_next_yaw_error": np.asarray(
                                    float(handoff_aux.get("shadow_residual_next_yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "runtime_handoff_shadow_residual_next_delta_local": np.asarray(
                                    handoff_aux.get("shadow_residual_next_delta_local", [np.nan] * 6),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_metric_xy_error": np.asarray(
                                    float((teacher_truth_handoff or {}).get("handoff_metrics", {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_metric_abs_z_error": np.asarray(
                                    float((teacher_truth_handoff or {}).get("handoff_metrics", {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_metric_yaw_error": np.asarray(
                                    float((teacher_truth_handoff or {}).get("handoff_metrics", {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_release_threshold_xy_error": np.asarray(
                                    float((teacher_truth_handoff or {}).get("handoff_release_metric_thresholds", {}).get("xy_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_release_threshold_abs_z_error": np.asarray(
                                    float((teacher_truth_handoff or {}).get("handoff_release_metric_thresholds", {}).get("abs_z_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "teacher_truth_handoff_release_threshold_yaw_error": np.asarray(
                                    float((teacher_truth_handoff or {}).get("handoff_release_metric_thresholds", {}).get("yaw_error", np.nan)),
                                    dtype=np.float32,
                                ),
                                "handoff_spec_name": np.asarray(str(getattr(target_result, "handoff_spec_name", "none"))),
                                "handoff_target_role": np.asarray(str(getattr(target_result, "handoff_target_role", "none"))),
                                "handoff_uses_privileged": np.asarray(
                                    float(getattr(target_result, "handoff_uses_privileged", False)), dtype=np.float32
                                ),
                                "handoff_target_pose_7d": np.asarray(
                                    getattr(target_result, "handoff_target_pose_7d", runtime_target_pose),
                                    dtype=np.float32,
                                ),
                                "oracle_executed": np.asarray(float(oracle_collect_active), dtype=np.float32),
                                "executed_action_source": np.asarray(
                                    "oracle_executed_align" if oracle_collect_active else "learned_or_planner_rollout"
                                ),
                                "current_delta_basin_target": current_delta,
                                "current_basin_distance": np.asarray(float(current_basin_distance), dtype=np.float32),
                                "current_dx_sign": np.asarray(sign_bucket(float(current_delta[0]), 1e-4), dtype=np.int64),
                                "current_dy_sign": np.asarray(sign_bucket(float(current_delta[1]), 1e-4), dtype=np.int64),
                                "current_dyaw_sign": np.asarray(sign_bucket(float(current_delta[5]), 1e-3), dtype=np.int64),
                                "basin_distance_bin": np.asarray(
                                    0 if float(current_basin_distance) <= 0.9 else 1 if float(current_basin_distance) <= 1.05 else 2 if float(current_basin_distance) <= 1.2 else 3,
                                    dtype=np.int64,
                                ),
                                "candidate_actions_local": np.asarray(candidate_actions, dtype=np.float32),
                                "candidate_group_index": candidate_group_index_row.astype(np.int64),
                                "candidate_mask": candidate_mask.astype(np.float32),
                                "candidate_kind": candidate_kind,
                                "candidate_next_basin_distance": np.asarray(oracle_next_dists, dtype=np.float32),
                                "candidate_improvement": np.asarray(
                                    [float(current_basin_distance) - float(x) for x in oracle_next_dists],
                                    dtype=np.float32,
                                ),
                                "candidate_oracle_score": np.asarray(oracle_scores, dtype=np.float32),
                                "candidate_basin_positive": (np.asarray(oracle_next_dists, dtype=np.float32) <= 1.0).astype(np.float32),
                                "candidate_tier": improvement_tiers(
                                    np.asarray(oracle_scores, dtype=np.float32),
                                    (np.asarray(oracle_next_dists, dtype=np.float32) <= 1.0).astype(np.float32),
                                ).astype(np.int64),
                                "best_candidate_index": np.asarray(int(oracle_idx), dtype=np.int64),
                                "oracle_candidate_index": np.asarray(int(oracle_idx), dtype=np.int64),
                                "oracle_action_local": oracle_action_local_row.astype(np.float32),
                                "oracle_action_frame_is_local": np.asarray(1.0, dtype=np.float32),
                                "executed_action_local": oracle_action_local_row.astype(np.float32)
                                if oracle_collect_active
                                else np.asarray(
                                    world_delta_to_local(delta_action[:6], obs.gripper_pose[3:7]),
                                    dtype=np.float32,
                                ),
                                "residual_label_local": residual_label_local,
                                "residual_label_world": residual_label_world,
                                "best_group_index": np.asarray(
                                    int(candidate_group_index_row[oracle_idx]),
                                    dtype=np.int64,
                                ),
                                "support_source_index": np.asarray(-1, dtype=np.int64),
                                "runtime_selected_candidate_index": np.asarray(
                                    int(refiner_stats_snapshot.get("last_scorer_candidate_index", -1)),
                                    dtype=np.int64,
                                ),
                                "runtime_selected_group_index": np.asarray(
                                    int(refiner_stats_snapshot.get("last_scorer_group_index", -1)),
                                    dtype=np.int64,
                                ),
                                "executed_motion_local": np.asarray(
                                    np.asarray(world_delta_to_local(delta_action[:6], obs.gripper_pose[3:7]), dtype=np.float32),
                                    dtype=np.float32,
                                ),
                                "runtime_selected_matches_oracle": np.asarray(
                                    float(int(refiner_stats_snapshot.get("last_scorer_candidate_index", -1)) == int(oracle_idx)),
                                    dtype=np.float32,
                                ),
                                "pred_candidate_index": np.asarray(int(debug_pred["pred_candidate_index"]), dtype=np.int64),
                                "pred_group_index": np.asarray(int(debug_pred["pred_group_index"]), dtype=np.int64),
                                "topk_candidate_index": np.asarray(debug_pred["topk_candidate_index"], dtype=np.int64),
                                "topk_candidate_prob": np.asarray(debug_pred["topk_candidate_prob"], dtype=np.float32),
                                "candidate_scores": np.asarray(debug_pred["candidate_scores"], dtype=np.float32),
                                "candidate_probs": np.asarray(debug_pred["candidate_probs"], dtype=np.float32),
                                "group_probs": np.asarray(debug_pred["group_probs"], dtype=np.float32),
                                "b2_candidate_shadow_gate_open": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_gate_open", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_close_neighborhood": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_close_neighborhood", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_changed": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_changed", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_yaw_needed": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_yaw_needed", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_yaw_keep": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_yaw_keep", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_teacher_ready": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_teacher_ready", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_xy_block": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_xy_block", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_nearish_runtime": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_nearish_runtime", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_keep_baseline_forced": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_keep_baseline_forced", False)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_mode": np.asarray(
                                    int(refiner_stats_snapshot.get("last_b2_candidate_shadow_mode", -1)),
                                    dtype=np.int64,
                                ),
                                "b2_candidate_shadow_baseline_index": np.asarray(
                                    int(refiner_stats_snapshot.get("last_b2_candidate_shadow_baseline_index", -1)),
                                    dtype=np.int64,
                                ),
                                "b2_candidate_shadow_pred_index": np.asarray(
                                    int(refiner_stats_snapshot.get("last_b2_candidate_shadow_pred_index", -1)),
                                    dtype=np.int64,
                                ),
                                "b2_candidate_shadow_best_index": np.asarray(
                                    int(refiner_stats_snapshot.get("last_b2_candidate_shadow_best_index", -1)),
                                    dtype=np.int64,
                                ),
                                "b2_candidate_shadow_runtime_scope_size": np.asarray(
                                    int(refiner_stats_snapshot.get("last_b2_candidate_shadow_runtime_scope_size", 0)),
                                    dtype=np.int64,
                                ),
                                "b2_candidate_shadow_small_yaw_scope_size": np.asarray(
                                    int(refiner_stats_snapshot.get("last_b2_candidate_shadow_small_yaw_scope_size", 0)),
                                    dtype=np.int64,
                                ),
                                "b2_candidate_shadow_large_yaw_scope_size": np.asarray(
                                    int(refiner_stats_snapshot.get("last_b2_candidate_shadow_large_yaw_scope_size", 0)),
                                    dtype=np.int64,
                                ),
                                "b2_candidate_shadow_probe_count": np.asarray(
                                    int(refiner_stats_snapshot.get("last_b2_candidate_shadow_probe_count", 0)),
                                    dtype=np.int64,
                                ),
                                "b2_candidate_shadow_mode_confidence": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_mode_confidence", np.nan)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_mode_margin": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_mode_margin", np.nan)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_best_cost": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_best_cost", np.nan)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_baseline_cost": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_baseline_cost", np.nan)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_pred_cost": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_pred_cost", np.nan)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_baseline_regret": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_baseline_regret", np.nan)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_pred_regret": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_pred_regret", np.nan)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_regret_delta": np.asarray(
                                    float(refiner_stats_snapshot.get("last_b2_candidate_shadow_regret_delta", np.nan)),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_candidate_actions_local": np.asarray(
                                    refiner_stats_snapshot.get("last_b2_candidate_shadow_candidate_actions_local", []),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_candidate_scope_mask": np.asarray(
                                    refiner_stats_snapshot.get("last_b2_candidate_shadow_candidate_scope_mask", []),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_candidate_valid_mask": np.asarray(
                                    refiner_stats_snapshot.get("last_b2_candidate_shadow_candidate_valid_mask", []),
                                    dtype=np.float32,
                                ),
                                "b2_candidate_shadow_candidate_oracle_score": np.asarray(
                                    refiner_stats_snapshot.get("last_b2_candidate_shadow_candidate_oracle_score", []),
                                    dtype=np.float32,
                                ),
                                "ready_to_close_target": np.asarray(
                                    float(
                                        getattr(
                                            target_result,
                                            "handoff_ready",
                                            teacher_alignment_ready_now
                                            if bool(getattr(args, "teacher_planner_close_handoff", True))
                                            else teacher_close_ready_now,
                                        )
                                    )
                                    if bool(getattr(args, "oracle_executed_pregrasp_collect", False)) else 0.0,
                                    dtype=np.float32,
                                ),
                                "post_close_stability_proxy": np.asarray(0.0, dtype=np.float32),
                                "grasp_lift_proxy": np.asarray(0.0, dtype=np.float32),
                                "grasp_verified_target": np.asarray(0.0, dtype=np.float32),
                                "retry_required_target": np.asarray(float(teacher_retry_required_now), dtype=np.float32),
                                "reopen_after_trigger": np.asarray(0.0, dtype=np.float32),
                                "invalid_after_trigger": np.asarray(0.0, dtype=np.float32),
                                "planner_close_too_early": np.asarray(0.0, dtype=np.float32),
                                "teacher_close_ready_now": np.asarray(float(teacher_close_ready_now), dtype=np.float32),
                                "teacher_alignment_ready_now": np.asarray(float(teacher_alignment_ready_now), dtype=np.float32),
                                "teacher_alignment_handoff_active": np.asarray(float(teacher_alignment_handoff_active), dtype=np.float32),
                                "teacher_verify_active_now": np.asarray(float(teacher_verify_active_now), dtype=np.float32),
                                "teacher_retry_active_now": np.asarray(float(teacher_retry_active_now), dtype=np.float32),
                                "orientation_rescue_active": np.asarray(float(teacher_enable_orientation_rescue), dtype=np.float32),
                                "contact_onset": np.asarray(
                                    float(refiner_stats_snapshot.get("contact_state_id", 0) >= int(ContactState.IN_CONTACT)),
                                    dtype=np.float32,
                                ),
                                "post_contact_outcome": np.asarray(0.0, dtype=np.float32),
                                "wrist_valid_depth_ratio": np.asarray(float(wrist_valid_depth_ratio), dtype=np.float32),
                                "wrist_depth_near_fraction": np.asarray(float(wrist_depth_near_fraction), dtype=np.float32),
                                "is_occluded": np.asarray(float(wrist_is_occluded), dtype=np.float32),
                                "is_low_visibility": np.asarray(float(wrist_is_low_visibility), dtype=np.float32),
                            }
                        )
            if coarse2contact is not None:
                delta_action = coarse2contact.step(
                    delta_action,
                    force_reading=raw_force,
                    gripper_z=float(obs.gripper_pose[2]),
                    wrist_depth=obs.wrist_depth,
                    proprio=proprio,
                )
                trace_entry.update(coarse2contact.get_last_trace())
                trace_entry["after_coarse2contact_gripper_raw"] = float(delta_action[6])
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
            active_alignment_controller = _active_alignment_like_controller(refiner)
            if (
                isinstance(refiner, StageAwareRefiner)
                and active_alignment_controller is not None
                and float(obs.gripper_open) >= refiner.alignment_open_threshold
                and float(delta_action[6]) <= refiner.alignment_close_command_threshold
            ):
                # Last-line guard: every final close command must pass the same
                # provider-owned handoff geometry, regardless of which phase or
                # supervisor branch produced it.
                before_gate_grip = float(delta_action[6])
                delta_action = refiner._apply_alignment_close_veto(
                    delta_action,
                    active_alignment_controller,
                    float(obs.gripper_open),
                    step_idx=chunk_step,
                )
                trace_entry["final_close_gate_input_gripper_raw"] = before_gate_grip
                trace_entry["final_close_gate_output_gripper_raw"] = float(delta_action[6])
                trace_entry["final_close_gate_blocked"] = bool(
                    float(delta_action[6]) > refiner.alignment_close_command_threshold
                    and before_gate_grip <= refiner.alignment_close_command_threshold
                )
            trace_entry["exec_gripper_raw"] = float(delta_action[6])
            trace_entry["final_action_world_7d"] = _jsonable_value(np.asarray(delta_action, dtype=np.float32))
            trace_entry["final_action_local_6d"] = _jsonable_value(
                world_delta_to_local(
                    np.asarray(delta_action[:6], dtype=np.float32),
                    np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                )
            )

            stage_gripper_open = float(obs.gripper_open)
            if bool(getattr(args, "oracle_executed_pregrasp_collect", False)) and not oracle_pregrasp_success:
                if teacher_close_hold_remaining > 0 or teacher_retry_steps_remaining > 0 or teacher_last_close_object_z is not None:
                    stage_gripper_open = 1.0
                    trace_entry["teacher_stage_align_freeze"] = True
            stage_tracker.update(
                force_reading=raw_force,
                gripper_pose=obs.gripper_pose,
                gripper_open=stage_gripper_open,
                depth_proximity=depth_proximity,
                base_action=delta_action,
            )
            if not np.all(np.isfinite(np.asarray(delta_action, dtype=np.float32).reshape(-1))):
                trace_entry["delta_action_nonfinite"] = True
                trace_entry["delta_action_nonfinite_before_fallback"] = _jsonable_value(delta_action)
                delta_action = np.zeros_like(np.asarray(delta_action, dtype=np.float32))
                if delta_action.size >= 7:
                    delta_action[6] = 1.0
                trace_entry["delta_action_nonfinite_fallback_applied"] = True
            else:
                trace_entry["delta_action_nonfinite"] = False
                trace_entry["delta_action_nonfinite_fallback_applied"] = False
            trace_entry["phase_after"] = int(stage_tracker.phase)
            phase_hist[int(stage_tracker.phase)] += 1
            failure_code = int(stage_tracker.failure_mode)
            failure_hist[failure_code] = failure_hist.get(failure_code, 0) + 1
            align_entered = align_entered or stage_tracker.max_phase_reached >= int(StagePhase.ALIGN)
            interact_entered = interact_entered or stage_tracker.max_phase_reached >= int(StagePhase.INTERACT)
            recover_entered = recover_entered or stage_tracker.max_phase_reached >= int(StagePhase.RECOVER)

            abs_action = delta_to_absolute(delta_action, obs.gripper_pose)
            refiner_delta_action_world_pre_workspace = np.asarray(delta_action[:6], dtype=np.float32).copy()
            refiner_delta_action_local_pre_workspace = world_delta_to_local(
                refiner_delta_action_world_pre_workspace,
                np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
            ).astype(np.float32)
            trace_entry["refiner_final_delta_world_pre_workspace_6d"] = _jsonable_value(
                refiner_delta_action_world_pre_workspace
            )
            trace_entry["refiner_final_delta_local_pre_workspace_6d"] = _jsonable_value(
                refiner_delta_action_local_pre_workspace
            )
            trace_entry["abs_gripper_cmd"] = float(abs_action[7])
            abs_gripper_cmd = float(abs_action[7])
            if (
                bool(getattr(args, "oracle_executed_pregrasp_collect", False))
                and bool(getattr(args, "teacher_planner_close_handoff", True))
                and bool(getattr(args, "teacher_planner_close_settle", True))
                and bool(teacher_alignment_handoff_active)
                and teacher_last_close_object_z is None
                and abs_gripper_cmd <= 0.5
            ):
                teacher_last_close_object_z = (
                    None if teacher_live_object_pose is None else float(teacher_live_object_pose[2])
                )
                teacher_last_close_gripper_pose = np.asarray(obs.gripper_pose, dtype=np.float32).copy()
                teacher_planner_verify_remaining = int(args.teacher_planner_verify_steps)
                if teacher_close_attempt_step is None:
                    teacher_close_attempt_step = int(step_idx)
                oracle_pregrasp_close_count += 1
                trace_entry["teacher_planner_close_detected"] = True
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
            final_applied_pose_7d = np.asarray(abs_action[:7], dtype=np.float32).copy()
            final_applied_delta_local = pose_delta_local_between(
                np.asarray(obs.gripper_pose, dtype=np.float32),
                final_applied_pose_7d,
            ).astype(np.float32)
            final_applied_delta_world = local_delta_to_world(
                final_applied_delta_local,
                np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
            ).astype(np.float32)
            trace_entry["final_applied_pose_7d"] = _jsonable_value(final_applied_pose_7d)
            trace_entry["final_applied_delta_local_6d"] = _jsonable_value(final_applied_delta_local)
            trace_entry["final_applied_delta_world_6d"] = _jsonable_value(final_applied_delta_world)
            trace_entry["final_workspace_violation_abs"] = float(workspace_violation if refiner is not None else 0.0)

            alignment_diffusion_step_row = None
            if alignment_diffusion_episode_rows is not None:
                executed_action_local = world_delta_to_local(
                    np.asarray(delta_action[:6], dtype=np.float32),
                    np.asarray(obs.gripper_pose[3:7], dtype=np.float32),
                ).astype(np.float32)
                nan_pose_7d = np.full((7,), np.nan, dtype=np.float32)
                nan_delta_6d = np.full((6,), np.nan, dtype=np.float32)
                teacher_result_for_raw = locals().get("teacher_truth_result", None)
                teacher_delta_for_raw = locals().get("teacher_truth_delta", None)
                label_only_privileged_for_raw = None
                raw_target_mode = str(getattr(args, "alignment_tc_raw_privileged_target_mode", "active"))
                force_label_only_target_for_raw = bool(
                    raw_target_mode in ("commit", "pregrasp", "insert_commit", "insert", "task_success_centre")
                    or bool(getattr(args, "alignment_tc_privileged_teacher_collect", False))
                )
                if (
                    (teacher_result_for_raw is None or force_label_only_target_for_raw)
                    and bool(getattr(args, "record_teacher_truth_metrics", False))
                ):
                    label_only_privileged_for_raw = build_depth_force_label_only_privileged_fields(
                        obs=obs,
                        stage_tracker=stage_tracker,
                        task_name=str(args.task_name),
                        live_target_handle=live_target_handle,
                        target_mode=raw_target_mode,
                    )
                if force_label_only_target_for_raw and label_only_privileged_for_raw is not None:
                    teacher_result_for_raw = None
                    teacher_delta_for_raw = None
                privileged_motion_target_pose = (
                    np.asarray(
                        getattr(teacher_result_for_raw, "motion_target_pose_7d", nan_pose_7d),
                        dtype=np.float32,
                    )
                    if teacher_result_for_raw is not None
                    else np.asarray(
                        (label_only_privileged_for_raw or {}).get("privileged_motion_target_pose_7d", nan_pose_7d),
                        dtype=np.float32,
                    )
                )
                privileged_basin_center_pose = (
                    np.asarray(getattr(teacher_result_for_raw, "target_pose_7d", nan_pose_7d), dtype=np.float32)
                    if teacher_result_for_raw is not None
                    else np.asarray(
                        (label_only_privileged_for_raw or {}).get("privileged_basin_center_pose_7d", nan_pose_7d),
                        dtype=np.float32,
                    )
                )
                privileged_current_delta_raw = (
                    np.asarray(teacher_delta_for_raw, dtype=np.float32).reshape(-1)[:6]
                    if teacher_delta_for_raw is not None
                    else np.asarray(
                        (label_only_privileged_for_raw or {}).get(
                            "privileged_current_delta_basin_target_raw",
                            (label_only_privileged_for_raw or {}).get("privileged_current_delta_basin_target", nan_delta_6d),
                        ),
                        dtype=np.float32,
                    ).reshape(-1)[:6]
                )
                privileged_current_delta = np.asarray(
                    (
                        (label_only_privileged_for_raw or {}).get(
                            "privileged_current_delta_basin_target_folded",
                            (label_only_privileged_for_raw or {}).get(
                                "privileged_current_delta_basin_target",
                                apply_yaw_symmetry_to_delta(np.asarray(privileged_current_delta_raw, dtype=np.float32), np.pi / 2.0),
                            ),
                        )
                    ),
                    dtype=np.float32,
                ).reshape(-1)[:6]
                live_object_pose_for_raw = (
                    np.asarray(teacher_live_object_pose, dtype=np.float32)
                    if "teacher_live_object_pose" in locals() and teacher_live_object_pose is not None
                    else np.asarray(
                        (label_only_privileged_for_raw or {}).get("privileged_object_anchor_pose_7d", nan_pose_7d),
                        dtype=np.float32,
                    )
                )
                privileged_source_for_raw = str(
                    getattr(
                        teacher_result_for_raw,
                        "source",
                        np.asarray((label_only_privileged_for_raw or {}).get("privileged_target_provider_source", "none")).item(),
                    )
                )
                privileged_uses_for_raw = float(
                    getattr(
                        teacher_result_for_raw,
                        "uses_privileged_target",
                        float(
                            np.asarray(
                                (label_only_privileged_for_raw or {}).get(
                                    "privileged_target_provider_uses_privileged",
                                    np.asarray(0.0, dtype=np.float32),
                                )
                            ).reshape(-1)[0]
                        ),
                    )
                )
                teacher_action_local_for_raw = np.zeros((6,), dtype=np.float32)
                teacher_collect_active_for_raw = bool(oracle_collect_active)
                teacher_action_source_for_raw = "none"
                if oracle_collect_action_local is not None:
                    teacher_action_local_for_raw = np.asarray(oracle_collect_action_local, dtype=np.float32).reshape(-1)[:6]
                    if teacher_action_local_for_raw.shape[0] < 6:
                        teacher_action_local_for_raw = np.pad(
                            teacher_action_local_for_raw,
                            (0, 6 - teacher_action_local_for_raw.shape[0]),
                            mode="constant",
                        ).astype(np.float32)
                    teacher_action_source_for_raw = str(
                        trace_entry.get(
                            "expert_sequence_name",
                            trace_entry.get(
                                "teacher_motion_controller",
                                "oracle_collect" if teacher_collect_active_for_raw else "teacher_shadow",
                            ),
                        )
                    )
                teacher_action4_for_raw = _alignment_diffusion_action4d_from_local(
                    teacher_action_local_for_raw
                ).astype(np.float32)
                pre_delta_for_raw = np.asarray(privileged_current_delta, dtype=np.float32).reshape(-1)
                if pre_delta_for_raw.size >= 6 and np.all(np.isfinite(pre_delta_for_raw[:6])):
                    teacher_pre_xy_for_raw = float(np.linalg.norm(pre_delta_for_raw[:2]))
                    teacher_pre_z_for_raw = float(abs(pre_delta_for_raw[2]))
                    teacher_pre_yaw_for_raw = float(abs(pre_delta_for_raw[5]))
                    teacher_pre_yaw_raw_for_raw = float(abs(privileged_current_delta_raw[5]))
                    teacher_pre_yaw_folded_for_raw = float(abs(privileged_current_delta[5]))
                else:
                    teacher_pre_xy_for_raw = float("nan")
                    teacher_pre_z_for_raw = float("nan")
                    teacher_pre_yaw_for_raw = float("nan")
                    teacher_pre_yaw_raw_for_raw = float("nan")
                    teacher_pre_yaw_folded_for_raw = float("nan")
                workspace_violation_abs_for_raw = float(workspace_violation if refiner is not None else 0.0)
                current_workspace_violation_abs_for_raw = 0.0
                workspace_violation_delta_for_raw = workspace_violation_abs_for_raw
                if refiner is not None:
                    current_workspace_violation_abs_for_raw = float(
                        refiner.safety.workspace_violation(np.asarray(obs.gripper_pose[:3], dtype=np.float32))
                    )
                    if bool(getattr(args, "alignment_tc_privileged_teacher_collect", False)):
                        workspace_violation_delta_for_raw = max(
                            0.0,
                            workspace_violation_abs_for_raw - current_workspace_violation_abs_for_raw,
                        )
                alignment_diffusion_step_row = {
                    "episode_index": np.asarray(int(ep_idx), dtype=np.int64),
                    "step_index": np.asarray(int(step_idx), dtype=np.int64),
                    "rollout_step": np.asarray(int(step_idx), dtype=np.int64),
                    "chunk_step": np.asarray(int(chunk_step), dtype=np.int64),
                    "task_name": np.asarray(str(args.task_name)),
                    "front_rgb": np.asarray(obs.front_rgb, dtype=np.uint8),
                    "wrist_rgb": np.asarray(obs.wrist_rgb, dtype=np.uint8),
                    "wrist_depth": depth_tensor_96.detach().cpu().numpy().astype(np.float32)
                    if depth_tensor_96 is not None
                    else np.zeros((1, 96, 96), dtype=np.float32),
                    "force_history": np.asarray(
                        force_hist.detach().cpu().numpy().astype(np.float32)
                        if force_hist is not None
                        else np.zeros((32, FORCE_DIM), dtype=np.float32),
                        dtype=np.float32,
                    ),
                    "proprio": np.asarray(proprio, dtype=np.float32),
                    "planner_action_local": np.asarray(planner_base_action_local_raw, dtype=np.float32),
                    "executed_action_local": executed_action_local.astype(np.float32),
                    "executed_action_local_4d": _alignment_diffusion_action4d_from_local(executed_action_local).astype(
                        np.float32
                    ),
                    "teacher_collect_active": np.asarray(float(teacher_collect_active_for_raw), dtype=np.float32),
                    "teacher_action_source": np.asarray(teacher_action_source_for_raw),
                    "teacher_residual_action_6d": teacher_action_local_for_raw.astype(np.float32),
                    "teacher_residual_action_4d": teacher_action4_for_raw.astype(np.float32),
                    "teacher_pre_xy": np.asarray(float(teacher_pre_xy_for_raw), dtype=np.float32),
                    "teacher_pre_z": np.asarray(float(teacher_pre_z_for_raw), dtype=np.float32),
                    "teacher_pre_yaw": np.asarray(float(teacher_pre_yaw_for_raw), dtype=np.float32),
                    "teacher_pre_yaw_raw": np.asarray(float(teacher_pre_yaw_raw_for_raw), dtype=np.float32),
                    "teacher_pre_yaw_folded": np.asarray(float(teacher_pre_yaw_folded_for_raw), dtype=np.float32),
                    "alignment_tc_teacher_candidate_name": np.asarray(
                        str(trace_entry.get("alignment_tc_teacher_candidate_name", "unknown"))
                    ),
                    "alignment_tc_teacher_candidate_variant": np.asarray(
                        str(trace_entry.get("alignment_tc_teacher_candidate_variant", "unknown"))
                    ),
                    "alignment_tc_teacher_candidate_count": np.asarray(
                        int(trace_entry.get("alignment_tc_teacher_candidate_count", 0)),
                        dtype=np.int64,
                    ),
                    "alignment_tc_teacher_candidate_scale": np.asarray(
                        float(trace_entry.get("alignment_tc_teacher_candidate_scale", np.nan)),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_candidate_score": np.asarray(
                        float(trace_entry.get("alignment_tc_teacher_candidate_score", np.nan)),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_candidate_improve_count": np.asarray(
                        int(trace_entry.get("alignment_tc_teacher_candidate_improve_count", 0)),
                        dtype=np.int64,
                    ),
                    "alignment_tc_teacher_candidate_final_xy": np.asarray(
                        float(trace_entry.get("alignment_tc_teacher_candidate_final_xy", np.nan)),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_candidate_final_z": np.asarray(
                        float(trace_entry.get("alignment_tc_teacher_candidate_final_z", np.nan)),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_candidate_final_yaw": np.asarray(
                        float(trace_entry.get("alignment_tc_teacher_candidate_final_yaw", np.nan)),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_candidate_workspace_delta_violation_pred": np.asarray(
                        float(trace_entry.get("alignment_tc_teacher_workspace_delta_violation_pred", np.nan)),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_already_close": np.asarray(
                        float(bool(trace_entry.get("alignment_tc_teacher_already_close", False))),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_valid_candidate_count": np.asarray(
                        int(trace_entry.get("alignment_tc_teacher_valid_candidate_count", 0)),
                        dtype=np.int64,
                    ),
                    "alignment_tc_teacher_selected_from_valid": np.asarray(
                        float(bool(trace_entry.get("alignment_tc_teacher_selected_from_valid", False))),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_selected_constraint_ok": np.asarray(
                        float(bool(trace_entry.get("alignment_tc_teacher_selected_constraint_ok", False))),
                        dtype=np.float32,
                    ),
                    "alignment_tc_teacher_constraint_reason": np.asarray(
                        str(trace_entry.get("alignment_tc_teacher_constraint_reason", "unknown"))
                    ),
                    "post_xy": np.asarray(np.nan, dtype=np.float32),
                    "post_z": np.asarray(np.nan, dtype=np.float32),
                    "post_yaw": np.asarray(np.nan, dtype=np.float32),
                    "teacher_improves_xy": np.asarray(0.0, dtype=np.float32),
                    "teacher_improves_z": np.asarray(0.0, dtype=np.float32),
                    "teacher_improves_yaw": np.asarray(0.0, dtype=np.float32),
                    "teacher_improves_two_axis": np.asarray(0.0, dtype=np.float32),
                    "teacher_all_improves": np.asarray(0.0, dtype=np.float32),
                    "success_label": np.asarray(0.0, dtype=np.float32),
                    "teacher_close_ready": np.asarray(
                        float(bool(trace_entry.get("teacher_close_ready", False))),
                        dtype=np.float32,
                    ),
                    "teacher_close_ready_all": np.asarray(
                        float(bool(trace_entry.get("teacher_close_ready_all", False))),
                        dtype=np.float32,
                    ),
                    "teacher_close_action": np.asarray(str(trace_entry.get("teacher_close_action", "continue_align"))),
                    "teacher_close_gate_reason": np.asarray(
                        str(trace_entry.get("teacher_close_gate_reason", "unknown"))
                    ),
                    "teacher_close_contact_ready": np.asarray(
                        float(
                            bool(
                                trace_entry.get(
                                    "teacher_close_contact_ready",
                                    bool(trace_entry.get("teacher_close_contact_ready_by_geometry", False))
                                    and (
                                        bool(trace_entry.get("teacher_close_contact_ready_by_stage", False))
                                        or bool(trace_entry.get("teacher_close_contact_ready_by_depth", False))
                                    ),
                                )
                            )
                        ),
                        dtype=np.float32,
                    ),
                    "teacher_close_contact_ready_by_depth": np.asarray(
                        float(bool(trace_entry.get("teacher_close_contact_ready_by_depth", False))),
                        dtype=np.float32,
                    ),
                    "teacher_close_contact_ready_by_stage": np.asarray(
                        float(bool(trace_entry.get("teacher_close_contact_ready_by_stage", False))),
                        dtype=np.float32,
                    ),
                    "teacher_close_contact_ready_by_geometry": np.asarray(
                        float(bool(trace_entry.get("teacher_close_contact_ready_by_geometry", False))),
                        dtype=np.float32,
                    ),
                    "teacher_close_contact_visibility_ok": np.asarray(
                        float(bool(trace_entry.get("teacher_close_contact_visibility_ok", False))),
                        dtype=np.float32,
                    ),
                    "teacher_close_contact_confidence": np.asarray(
                        float(trace_entry.get("teacher_close_contact_confidence", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_gripper_finger_pose_7d": np.asarray(
                        trace_entry.get("teacher_gripper_finger_pose_7d", np.asarray(obs.gripper_pose, dtype=np.float32)),
                        dtype=np.float32,
                    ),
                    "teacher_object_in_finger_region": np.asarray(
                        float(bool(trace_entry.get("teacher_object_in_finger_region", False))),
                        dtype=np.float32,
                    ),
                    "teacher_finger_object_lateral_error": np.asarray(
                        float(trace_entry.get("teacher_finger_object_lateral_error", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_finger_object_height_overlap": np.asarray(
                        float(trace_entry.get("teacher_finger_object_height_overlap", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_finger_object_yaw_error": np.asarray(
                        float(trace_entry.get("teacher_finger_object_yaw_error", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_grasp_contact_ready": np.asarray(
                        float(bool(trace_entry.get("teacher_grasp_contact_ready", False))),
                        dtype=np.float32,
                    ),
                    "teacher_grasp_ready": np.asarray(
                        float(bool(trace_entry.get("teacher_grasp_ready", False))),
                        dtype=np.float32,
                    ),
                    "teacher_grasp_readiness_score": np.asarray(
                        float(trace_entry.get("teacher_grasp_readiness_score", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_grasp_readiness_reason": np.asarray(
                        str(trace_entry.get("teacher_grasp_readiness_reason", "unknown"))
                    ),
                    "teacher_attached_after_close": np.asarray(
                        float(bool(trace_entry.get("teacher_attached_after_close", False))),
                        dtype=np.float32,
                    ),
                    "teacher_grasped_object_count": np.asarray(
                        int(trace_entry.get("teacher_grasped_object_count", 0)),
                        dtype=np.int64,
                    ),
                    "teacher_grasped_target_handle_match": np.asarray(
                        float(bool(trace_entry.get("teacher_grasped_target_handle_match", False))),
                        dtype=np.float32,
                    ),
                    "teacher_object_to_gripper_delta_local_6d": np.asarray(
                        trace_entry.get(
                            "teacher_object_to_gripper_delta_local_6d",
                            np.full((6,), np.nan, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ),
                    "teacher_demo_basin_distance": np.asarray(
                        float(trace_entry.get("teacher_demo_basin_distance", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_close_basin_source": np.asarray(
                        str(trace_entry.get("teacher_close_basin_source", "heuristic"))
                    ),
                    "teacher_close_failure_reason": np.asarray(
                        str(trace_entry.get("teacher_close_failure_reason", "unknown"))
                    ),
                    "teacher_close_yaw_raw": np.asarray(
                        float(trace_entry.get("teacher_close_yaw_raw", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_close_yaw_folded": np.asarray(
                        float(trace_entry.get("teacher_close_yaw_folded", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_yaw_control_sign": np.asarray(
                        float(trace_entry.get("teacher_yaw_control_sign", np.nan)),
                        dtype=np.float32,
                    ),
                    "yaw_imitation_enabled": np.asarray(
                        float(bool(trace_entry.get("teacher_yaw_imitation_enabled", True))),
                        dtype=np.float32,
                    ),
                    "teacher_tc_motion_phase": np.asarray(
                        str(trace_entry.get("teacher_tc_motion_phase", "align_xy_yaw"))
                    ),
                    "expert_sequence_name": np.asarray(
                        str(trace_entry.get("expert_sequence_name", "align_xy"))
                    ),
                    "expert_sequence_score": np.asarray(
                        float(trace_entry.get("expert_sequence_score", np.nan)),
                        dtype=np.float32,
                    ),
                    "expert_sequence_verified": np.asarray(
                        float(bool(trace_entry.get("expert_sequence_verified", False))),
                        dtype=np.float32,
                    ),
                    "teacher_motion_phase": np.asarray(str(trace_entry.get("teacher_motion_phase", "align_xy_yaw"))),
                    "teacher_grasp_verified": np.asarray(
                        float(bool(trace_entry.get("teacher_grasp_verified_now", False))),
                        dtype=np.float32,
                    ),
                    "teacher_object_lift": np.asarray(
                        float(trace_entry.get("teacher_object_lift", np.nan)),
                        dtype=np.float32,
                    ),
                    "teacher_object_follows_gripper": np.asarray(
                        float(bool(trace_entry.get("teacher_object_follows_gripper", False))),
                        dtype=np.float32,
                    ),
                    "teacher_close_settle_phase": np.asarray(
                        str(trace_entry.get("teacher_close_settle_phase", "none"))
                    ),
                    "teacher_verify_lift_streak": np.asarray(
                        int(trace_entry.get("teacher_verify_lift_streak", 0)),
                        dtype=np.int64,
                    ),
                    "teacher_close_attempt_step": np.asarray(
                        -1 if teacher_close_attempt_step is None else int(teacher_close_attempt_step),
                        dtype=np.int64,
                    ),
                    "teacher_grasp_verified_step": np.asarray(
                        -1 if teacher_grasp_verified_step is None else int(teacher_grasp_verified_step),
                        dtype=np.int64,
                    ),
                    "close_success_label": np.asarray(float(oracle_pregrasp_success), dtype=np.float32),
                    "close_failure_reason": np.asarray(str(trace_entry.get("close_failure_reason", "none"))),
                    "time_to_close": np.asarray(
                        -1 if teacher_close_attempt_step is None else int(teacher_close_attempt_step) - int(step_idx),
                        dtype=np.int64,
                    ),
                    "time_to_verified_grasp": np.asarray(
                        -1 if teacher_grasp_verified_step is None else int(teacher_grasp_verified_step) - int(step_idx),
                        dtype=np.int64,
                    ),
                    "planner_action_local_4d": _alignment_diffusion_action4d_from_local(
                        np.asarray(planner_base_action_local_raw, dtype=np.float32)
                    ).astype(np.float32),
                    "base_action_world": np.asarray(delta_action[:6], dtype=np.float32),
                    "gripper_context": np.asarray(gripper_context_arr, dtype=np.float32),
                    "current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                    "gripper_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                    "privileged_current_pose_7d": np.asarray(obs.gripper_pose, dtype=np.float32),
                    "privileged_motion_target_pose_7d": privileged_motion_target_pose.astype(np.float32),
                    "privileged_basin_center_pose_7d": privileged_basin_center_pose.astype(np.float32),
                    "privileged_object_anchor_pose_7d": live_object_pose_for_raw.astype(np.float32),
                    "privileged_current_to_target_delta_local": privileged_current_delta.astype(np.float32),
                    "privileged_current_delta_basin_target": privileged_current_delta.astype(np.float32),
                    "privileged_target_provider_source": np.asarray(privileged_source_for_raw),
                    "privileged_target_provider_uses_privileged": np.asarray(
                        privileged_uses_for_raw,
                        dtype=np.float32,
                    ),
                    "current_gripper_open": np.asarray(float(obs.gripper_open), dtype=np.float32),
                    "depth_proximity": np.asarray(
                        float(depth_proximity) if depth_proximity is not None else np.nan,
                        dtype=np.float32,
                    ),
                    "wrist_depth_median": np.asarray(
                        float(np.median(depth_tensor_96.detach().cpu().numpy())),
                        dtype=np.float32,
                    ),
                    "wrist_valid_depth_ratio": np.asarray(float(wrist_valid_depth_ratio), dtype=np.float32),
                    "wrist_depth_near_fraction": np.asarray(float(wrist_depth_near_fraction), dtype=np.float32),
                    "is_occluded": np.asarray(float(wrist_is_occluded), dtype=np.float32),
                    "is_low_visibility": np.asarray(float(wrist_is_low_visibility), dtype=np.float32),
                    "force_norm": np.asarray(
                        float(np.linalg.norm(raw_force[:3])) if raw_force is not None else 0.0,
                        dtype=np.float32,
                    ),
                    "prev_force_norm": np.asarray(
                        float(np.linalg.norm(force_buffer[-2][:3]))
                        if force_buffer is not None and len(force_buffer) >= 2 and force_buffer[-2] is not None
                        else float(np.linalg.norm(raw_force[:3])) if raw_force is not None else 0.0,
                        dtype=np.float32,
                    ),
                    "force_spike": np.asarray(
                        float(
                            raw_force is not None
                            and (
                                np.linalg.norm(raw_force[:3]) > float(args.alignment_diffusion_raw_high_force_threshold)
                                or np.linalg.norm(raw_force[:3])
                                - float(
                                    np.linalg.norm(force_buffer[-2][:3])
                                    if force_buffer is not None and len(force_buffer) >= 2 and force_buffer[-2] is not None
                                    else np.linalg.norm(raw_force[:3])
                                )
                                > float(args.alignment_diffusion_raw_force_spike_threshold)
                            )
                        ),
                        dtype=np.float32,
                    ),
                    "planner_close_intent": np.asarray(
                        float(np.clip(base_delta_action[6], -1.0, 1.0) <= 0.5),
                        dtype=np.float32,
                    ),
                    "phase_id": np.asarray(int(stage_tracker.phase), dtype=np.int64),
                    "substage_id": np.asarray(int(getattr(stage_tracker, "substage", 0)), dtype=np.int64),
                    "max_phase_reached": np.asarray(int(stage_tracker.max_phase_reached), dtype=np.int64),
                    "workspace_violation": np.asarray(float(workspace_violation_delta_for_raw), dtype=np.float32),
                    "workspace_violation_abs": np.asarray(float(workspace_violation_abs_for_raw), dtype=np.float32),
                    "current_workspace_violation_abs": np.asarray(
                        float(current_workspace_violation_abs_for_raw),
                        dtype=np.float32,
                    ),
                    "invalid_action": np.asarray(float(False), dtype=np.float32),
                    "reward": np.asarray(np.nan, dtype=np.float32),
                    "terminate": np.asarray(0.0, dtype=np.float32),
                    "next_gripper_open": np.asarray(np.nan, dtype=np.float32),
                    "sample_weight": np.asarray(
                        float(args.alignment_diffusion_raw_sample_weight_micro)
                        if (
                            np.isfinite(float(depth_proximity) if depth_proximity is not None else np.nan)
                            and float(depth_proximity) <= float(args.alignment_diffusion_raw_micro_depth_threshold)
                        )
                        else float(args.alignment_diffusion_raw_sample_weight_near),
                        dtype=np.float32,
                    ),
                    "stage_bucket": np.asarray(
                        _alignment_diffusion_bucket(
                            float(depth_proximity) if depth_proximity is not None else np.nan,
                            float(np.linalg.norm(raw_force[:3])) if raw_force is not None else 0.0,
                            int(stage_tracker.phase),
                            bool(np.clip(base_delta_action[6], -1.0, 1.0) <= 0.5),
                        )
                    ),
                    "contact_state_hint": np.asarray(
                        int(getattr(stage_tracker, "contact_state", 0)),
                        dtype=np.int64,
                    ),
                }

            if bool(getattr(args, "depth_force_clean_support", False)):
                # Historically this support row writer only ran in refiner-free
                # clean-support modes. For from-scratch rollout collection we also
                # allow explicit refiner-backed collection so the raw trace can
                # keep richer near/micro context while still materializing support
                # tensors for downstream teacher/dataset builders.
                support_ok_with_refiner = bool(getattr(args, "depth_force_clean_support_with_refiner", False))
                if refiner is None or support_ok_with_refiner:
                    privileged_label = None
                    if bool(getattr(args, "depth_force_clean_privileged_labels", False)):
                        privileged_label = build_depth_force_label_only_privileged_fields(
                            obs=obs,
                            stage_tracker=stage_tracker,
                            task_name=str(args.task_name),
                            live_target_handle=live_target_handle,
                        )
                    append_depth_force_clean_support_row(
                        support_state_rows,
                        ep_idx=int(ep_idx),
                        step_idx=int(step_idx),
                        chunk_step=int(chunk_step),
                        obs=obs,
                        proprio=proprio,
                        depth_tensor_96=depth_tensor_96,
                        force_hist=force_hist,
                        force_buffer=force_buffer,
                        raw_force=raw_force,
                        base_delta_action=base_delta_action,
                        delta_action=delta_action,
                        planner_base_action_local_raw=planner_base_action_local_raw,
                        planner_base_action_7d_raw=planner_base_action_7d_raw,
                        depth_proximity=depth_proximity,
                        wrist_valid_depth_ratio=float(wrist_valid_depth_ratio),
                        wrist_depth_near_fraction=float(wrist_depth_near_fraction),
                        wrist_is_occluded=bool(wrist_is_occluded),
                        wrist_is_low_visibility=bool(wrist_is_low_visibility),
                        stage_tracker=stage_tracker,
                        abs_gripper_cmd=float(abs_gripper_cmd),
                        privileged_label=privileged_label,
                        args=args,
                    )

            try:
                obs, reward, terminate = task.step(abs_action)
            except InvalidActionError as e:
                coarse2contact_recovery_abs = None
                if alignment_diffusion_step_row is not None:
                    alignment_diffusion_step_row["invalid_action"] = np.asarray(1.0, dtype=np.float32)
                    alignment_diffusion_step_row["reward"] = np.asarray(0.0, dtype=np.float32)
                    alignment_diffusion_step_row["terminate"] = np.asarray(0.0, dtype=np.float32)
                    alignment_diffusion_step_row["next_gripper_open"] = np.asarray(float(obs.gripper_open), dtype=np.float32)
                    alignment_diffusion_episode_rows.append(alignment_diffusion_step_row)
                trace_entry["invalid_action"] = True
                trace_entry["invalid_error"] = type(e).__name__
                trace_entry["invalid_action_count_so_far"] = int(invalid_action_count + 1)
                trace_entry["recovery_primitive"] = trace_entry.get("recovery_primitive", "none")
                trace_entry["recovery_phase"] = trace_entry.get("recovery_phase", "IDLE")
                if args.record_gripper_trace:
                    gripper_trace.append(trace_entry)
                invalid_action_count += 1
                invalid_action_exception_count += 1
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

                if coarse2contact is not None and hasattr(coarse2contact, "build_invalid_action_recovery_absolute"):
                    coarse2contact_recovery_abs = coarse2contact.build_invalid_action_recovery_absolute(
                        obs.gripper_pose,
                        obs.gripper_open,
                        force_reading=raw_force,
                        proprio=proprio,
                    )
                    trace_entry["coarse2contact_invalid_action_recovery"] = True
                    trace_entry["coarse2contact_invalid_action_recovery_phase"] = str(
                        coarse2contact.get_last_trace().get("recovery_phase", "IDLE")
                    )
                    trace_entry["coarse2contact_invalid_action_recovery_primitive"] = str(
                        coarse2contact.get_last_trace().get("recovery_primitive", "none")
                    )
                    trace_entry["coarse2contact_invalid_action_recovery_reason"] = str(
                        coarse2contact.get_last_trace().get("force_reflex_reason", "invalid_action")
                    )
                    recovery_abs = coarse2contact_recovery_abs
                    recovery_abs, recovery_violation = maybe_apply_workspace_filter(
                        recovery_abs,
                        coarse2contact.safety,
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
                        if not success_reported:
                            print(f"  SUCCESS during invalid-action recovery at step {step_count}")
                            success_reported = True
                        if not bool(args.run_full_horizon_on_success):
                            break
                continue
            except Exception as e:
                if alignment_diffusion_step_row is not None:
                    alignment_diffusion_step_row["invalid_action"] = np.asarray(1.0, dtype=np.float32)
                    alignment_diffusion_step_row["reward"] = np.asarray(0.0, dtype=np.float32)
                    alignment_diffusion_step_row["terminate"] = np.asarray(0.0, dtype=np.float32)
                    alignment_diffusion_step_row["next_gripper_open"] = np.asarray(float(obs.gripper_open), dtype=np.float32)
                    alignment_diffusion_episode_rows.append(alignment_diffusion_step_row)
                trace_entry["invalid_action"] = True
                trace_entry["invalid_error"] = type(e).__name__
                trace_entry["invalid_action_count_so_far"] = int(invalid_action_count + 1)
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
            if alignment_diffusion_step_row is not None:
                alignment_diffusion_step_row["reward"] = np.asarray(float(reward), dtype=np.float32)
                alignment_diffusion_step_row["terminate"] = np.asarray(float(bool(terminate)), dtype=np.float32)
                alignment_diffusion_step_row["next_gripper_open"] = np.asarray(float(obs.gripper_open), dtype=np.float32)
                alignment_diffusion_step_row["success_label"] = np.asarray(float(reward > 0), dtype=np.float32)
                target_pose_for_post = np.asarray(
                    alignment_diffusion_step_row.get(
                        "privileged_motion_target_pose_7d",
                        np.full((7,), np.nan, dtype=np.float32),
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                if target_pose_for_post.size >= 7 and np.all(np.isfinite(target_pose_for_post[:7])):
                    post_delta = pose_delta_local_between(
                        np.asarray(obs.gripper_pose, dtype=np.float32),
                        target_pose_for_post[:7],
                    )
                    post_delta = apply_yaw_symmetry_to_delta(post_delta, np.pi / 2.0)
                    post_xy = float(np.linalg.norm(post_delta[:2]))
                    post_z = float(abs(post_delta[2]))
                    post_yaw = float(abs(post_delta[5]))
                    pre_xy = float(np.asarray(alignment_diffusion_step_row.get("teacher_pre_xy", np.nan)).reshape(()))
                    pre_z = float(np.asarray(alignment_diffusion_step_row.get("teacher_pre_z", np.nan)).reshape(()))
                    pre_yaw = float(np.asarray(alignment_diffusion_step_row.get("teacher_pre_yaw", np.nan)).reshape(()))
                    improves_xy = float(np.isfinite(pre_xy) and post_xy < pre_xy)
                    improves_z = float(np.isfinite(pre_z) and post_z < pre_z)
                    improves_yaw = float(np.isfinite(pre_yaw) and post_yaw < pre_yaw)
                    improve_count = int(improves_xy + improves_z + improves_yaw)
                    alignment_diffusion_step_row["post_xy"] = np.asarray(post_xy, dtype=np.float32)
                    alignment_diffusion_step_row["post_z"] = np.asarray(post_z, dtype=np.float32)
                    alignment_diffusion_step_row["post_yaw"] = np.asarray(post_yaw, dtype=np.float32)
                    alignment_diffusion_step_row["teacher_improves_xy"] = np.asarray(improves_xy, dtype=np.float32)
                    alignment_diffusion_step_row["teacher_improves_z"] = np.asarray(improves_z, dtype=np.float32)
                    alignment_diffusion_step_row["teacher_improves_yaw"] = np.asarray(improves_yaw, dtype=np.float32)
                    alignment_diffusion_step_row["teacher_improves_two_axis"] = np.asarray(
                        float(improve_count >= 2),
                        dtype=np.float32,
                    )
                    alignment_diffusion_step_row["teacher_all_improves"] = np.asarray(
                        float(improve_count >= 3),
                        dtype=np.float32,
                    )
                alignment_diffusion_episode_rows.append(alignment_diffusion_step_row)
            teacher_live_object_pose_after = safe_live_target_pose_7d(live_target_handle)
            teacher_verify_object_pose_source = "live_handle" if teacher_live_object_pose_after is not None else "missing"
            if teacher_live_object_pose_after is None:
                teacher_live_object_pose_after = safe_task_low_dim_pose_7d_from_task(task)
                if teacher_live_object_pose_after is not None:
                    teacher_verify_object_pose_source = "task_low_dim_state(task)"
            if teacher_live_object_pose_after is None:
                teacher_live_object_pose_after = safe_task_low_dim_pose_7d(obs)
                if teacher_live_object_pose_after is not None:
                    teacher_verify_object_pose_source = "task_low_dim_state(obs)"
            if (
                (
                    bool(getattr(args, "oracle_executed_pregrasp_collect", False))
                    or bool(tc_close_verify_collect_enabled)
                )
                and teacher_last_close_object_z is not None
            ):
                grasped_objects = []
                try:
                    grasped_objects = list(getattr(getattr(task, "robot", None), "gripper", None).get_grasped_objects())  # type: ignore[union-attr]
                except Exception:
                    grasped_objects = []
                object_grasped = bool(
                    live_target_handle is not None
                    and any(
                        getattr(obj, "get_handle", lambda: None)() == getattr(live_target_handle, "get_handle", lambda: None)()
                        for obj in grasped_objects
                    )
                )
                object_pose_available = bool(teacher_live_object_pose_after is not None)
                object_lift = float("nan")
                object_follow = False
                verify_fail_reason = "missing_object_pose"
                verify_follow_distance = float("nan")
                verify_object_source = str(teacher_verify_object_pose_source)
                verify_object_is_grasped = bool(object_grasped)
                grasped_object_count = int(len(grasped_objects))
                teacher_object_to_gripper_delta_local_6d = np.full((6,), np.nan, dtype=np.float32)
                if object_pose_available:
                    object_lift = float(teacher_live_object_pose_after[2] - float(teacher_last_close_object_z))
                    verify_object_source = str(teacher_verify_object_pose_source)
                    verify_follow_distance = float(
                        np.linalg.norm(
                            np.asarray(teacher_live_object_pose_after[:3], dtype=np.float32)
                            - np.asarray(obs.gripper_pose[:3], dtype=np.float32)
                        )
                    )
                    object_follow = bool(
                        verify_follow_distance <= float(args.teacher_verify_follow_distance)
                    )
                    teacher_object_to_gripper_delta_local_6d = pose_delta_local_between(
                        np.asarray(obs.gripper_pose, dtype=np.float32),
                        np.asarray(teacher_live_object_pose_after, dtype=np.float32),
                    )
                elif object_grasped:
                    object_lift = float(float(obs.gripper_pose[2]) - float(teacher_last_close_gripper_pose[2]))
                    verify_follow_distance = 0.0
                    object_follow = True
                    verify_fail_reason = "grasped_proxy"
                    verify_object_source = "grasped_proxy"
                    if teacher_last_close_object_pose is not None:
                        teacher_object_to_gripper_delta_local_6d = pose_delta_local_between(
                            np.asarray(obs.gripper_pose, dtype=np.float32),
                            np.asarray(teacher_last_close_object_pose, dtype=np.float32),
                        )
                if object_lift > float(args.teacher_verify_lift_threshold) and object_follow:
                    teacher_verify_lift_streak += 1
                else:
                    teacher_verify_lift_streak = 0
                teacher_grasp_verified_now = bool(
                    teacher_verify_lift_streak >= int(getattr(args, "teacher_verify_min_consecutive_lift_steps", 2))
                    and object_follow
                )
                verify_fail_reason = "none" if teacher_grasp_verified_now else (
                    "lift_too_small" if object_lift <= float(args.teacher_verify_lift_threshold) else "follow_too_far"
                )
                trace_entry["teacher_verify_object_pose_available"] = bool(object_pose_available or object_grasped)
                trace_entry["teacher_verify_object_pose_source"] = str(verify_object_source)
                trace_entry["teacher_verify_object_is_grasped"] = bool(verify_object_is_grasped)
                trace_entry["teacher_verify_fail_reason"] = str(verify_fail_reason)
                trace_entry["teacher_object_lift"] = float(object_lift)
                trace_entry["teacher_object_follows_gripper"] = bool(object_follow)
                trace_entry["teacher_verify_follow_distance"] = float(verify_follow_distance)
                trace_entry["teacher_verify_lift_streak"] = int(teacher_verify_lift_streak)
                trace_entry["teacher_grasp_verified_now"] = bool(teacher_grasp_verified_now)
                trace_entry["teacher_attached_after_close"] = bool(object_grasped)
                trace_entry["teacher_grasped_object_count"] = int(grasped_object_count)
                trace_entry["teacher_grasped_target_handle_match"] = bool(object_grasped)
                trace_entry["teacher_object_to_gripper_delta_local_6d"] = teacher_object_to_gripper_delta_local_6d.tolist()
                trace_entry["teacher_demo_basin_distance"] = float(
                    compute_basin_metrics(
                        teacher_object_to_gripper_delta_local_6d,
                        r_xy=float(getattr(args, "teacher_grasp_xy_threshold", getattr(args, "teacher_close_xy_threshold", 0.006))),
                        r_z=float(getattr(args, "teacher_grasp_abs_z_threshold", getattr(args, "teacher_close_abs_z_threshold", 0.010))),
                        r_yaw=float(getattr(args, "teacher_grasp_yaw_threshold", getattr(args, "teacher_close_yaw_threshold", 0.05))),
                    )[0]
                )
                trace_entry["teacher_close_failure_reason"] = str(verify_fail_reason)
                if alignment_diffusion_episode_rows:
                    current_raw_row = alignment_diffusion_episode_rows[-1]
                    if int(np.asarray(current_raw_row.get("step_index", -1)).reshape(())) == int(step_idx):
                        current_raw_row["teacher_object_lift"] = np.asarray(float(object_lift), dtype=np.float32)
                        current_raw_row["teacher_object_follows_gripper"] = np.asarray(
                            float(bool(object_follow)),
                            dtype=np.float32,
                        )
                        current_raw_row["teacher_verify_lift_streak"] = np.asarray(
                            int(teacher_verify_lift_streak),
                            dtype=np.int64,
                        )
                        current_raw_row["teacher_grasp_verified"] = np.asarray(
                            float(bool(teacher_grasp_verified_now)),
                            dtype=np.float32,
                        )
                        current_raw_row["teacher_attached_after_close"] = np.asarray(
                            float(bool(object_grasped)),
                            dtype=np.float32,
                        )
                        current_raw_row["teacher_grasped_object_count"] = np.asarray(
                            int(grasped_object_count),
                            dtype=np.int64,
                        )
                        current_raw_row["teacher_grasped_target_handle_match"] = np.asarray(
                            float(bool(object_grasped)),
                            dtype=np.float32,
                        )
                        current_raw_row["teacher_object_to_gripper_delta_local_6d"] = np.asarray(
                            teacher_object_to_gripper_delta_local_6d,
                            dtype=np.float32,
                        )
                        current_raw_row["teacher_demo_basin_distance"] = np.asarray(
                            float(trace_entry.get("teacher_demo_basin_distance", np.nan)),
                            dtype=np.float32,
                        )
                        current_raw_row["teacher_close_basin_source"] = np.asarray(
                            str(trace_entry.get("teacher_close_basin_source", "heuristic"))
                        )
                        current_raw_row["teacher_close_failure_reason"] = np.asarray(
                            str(trace_entry.get("teacher_close_failure_reason", verify_fail_reason))
                        )
                        current_raw_row["teacher_verify_object_pose_available"] = np.asarray(
                            float(bool(object_pose_available)),
                            dtype=np.float32,
                        )
                        current_raw_row["teacher_verify_object_pose_source"] = np.asarray(
                            str(verify_object_source)
                        )
                        current_raw_row["teacher_verify_object_is_grasped"] = np.asarray(
                            float(bool(verify_object_is_grasped)),
                            dtype=np.float32,
                        )
                        current_raw_row["teacher_verify_follow_distance"] = np.asarray(
                            float(verify_follow_distance),
                            dtype=np.float32,
                        )
                if teacher_grasp_verified_now and not oracle_pregrasp_success:
                    oracle_pregrasp_success = True
                    teacher_grasp_verified_step = int(step_idx)
                    teacher_alignment_handoff_active = False
                    teacher_last_close_object_z = None
                    teacher_close_hold_remaining = 0
                    teacher_planner_verify_remaining = 0
                    teacher_retry_steps_remaining = 0
                    teacher_realign_cooldown_remaining = 0
                    teacher_close_attempted_this_cycle = False
                    teacher_verify_lift_streak = 0
                    teacher_smooth_action_local = None
                    teacher_last_candidate_idx = None
                    teacher_candidate_hold_remaining = 0
                    teacher_planner_close_latch_remaining = 0
                    action_queue.clear()
                    chunk_step = 0
                    if isinstance(refiner, StageAwareRefiner):
                        refiner._residual_cooldown = 0
                    trace_entry["expert_sequence_verified"] = True
                    trace_entry["expert_sequence_name"] = "verified_success"
                    trace_entry["teacher_post_success_replan_handoff"] = True
                if teacher_planner_verify_remaining > 0:
                    teacher_planner_verify_remaining -= 1
            if (
                (
                    bool(getattr(args, "oracle_executed_pregrasp_collect", False))
                    or bool(tc_close_verify_collect_enabled)
                )
                and not oracle_pregrasp_success
                and teacher_last_close_object_z is not None
                and teacher_close_hold_remaining == 0
                and teacher_retry_steps_remaining == 0
                and teacher_planner_verify_remaining == 0
            ):
                if teacher_retry_budget_remaining > 0:
                    teacher_retry_budget_remaining -= 1
                    teacher_retry_steps_remaining = int(args.teacher_retry_steps)
                    teacher_realign_cooldown_remaining = int(args.teacher_retry_realign_cooldown_steps)
                    teacher_alignment_handoff_active = False
                    teacher_smooth_action_local = None
                    teacher_last_candidate_idx = None
                    teacher_candidate_hold_remaining = 0
                    teacher_planner_close_latch_remaining = 0
                    teacher_close_attempted_this_cycle = False
                    teacher_verify_lift_streak = 0
                    teacher_retry_anchor_steps.append(int(step_idx))
                    oracle_pregrasp_retry_count += 1
                    teacher_retry_required_now = True
                    trace_entry["teacher_retry_scheduled"] = True
                    teacher_last_close_object_z = None
                else:
                    trace_entry["teacher_retry_exhausted"] = True
            if (
                ever_closed_cmd
                and close_pose_z is not None
                and float(obs.gripper_open) < 0.5
                and float(obs.gripper_pose[2]) - float(close_pose_z) > 0.02
            ):
                grasp_lift_proxy = True
            if alignment_diffusion_episode_rows:
                current_raw_row = alignment_diffusion_episode_rows[-1]
                if int(np.asarray(current_raw_row.get("step_index", -1)).reshape(())) == int(step_idx):
                    current_raw_row["teacher_grasp_verified"] = np.asarray(
                        float(teacher_grasp_verified_now),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_verify_object_pose_available"] = np.asarray(
                        float(bool(trace_entry.get("teacher_verify_object_pose_available", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_verify_object_pose_source"] = np.asarray(
                        str(trace_entry.get("teacher_verify_object_pose_source", "missing"))
                    )
                    current_raw_row["teacher_verify_object_is_grasped"] = np.asarray(
                        float(bool(trace_entry.get("teacher_verify_object_is_grasped", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_verify_fail_reason"] = np.asarray(
                        str(trace_entry.get("teacher_verify_fail_reason", "unknown"))
                    )
                    current_raw_row["teacher_object_lift"] = np.asarray(
                        float(trace_entry.get("teacher_object_lift", np.nan)),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_object_follows_gripper"] = np.asarray(
                        float(bool(trace_entry.get("teacher_object_follows_gripper", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_verify_follow_distance"] = np.asarray(
                        float(trace_entry.get("teacher_verify_follow_distance", np.nan)),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_settle_phase"] = np.asarray(
                        str(trace_entry.get("teacher_close_settle_phase", "none"))
                    )
                    current_raw_row["teacher_verify_lift_streak"] = np.asarray(
                        int(teacher_verify_lift_streak),
                        dtype=np.int64,
                    )
                    current_raw_row["teacher_close_attempt_step"] = np.asarray(
                        -1 if teacher_close_attempt_step is None else int(teacher_close_attempt_step),
                        dtype=np.int64,
                    )
                    current_raw_row["teacher_grasp_verified_step"] = np.asarray(
                        -1 if teacher_grasp_verified_step is None else int(teacher_grasp_verified_step),
                        dtype=np.int64,
                    )
                    current_raw_row["expert_sequence_name"] = np.asarray(
                        str(trace_entry.get("expert_sequence_name", "unknown"))
                    )
                    current_raw_row["expert_sequence_score"] = np.asarray(
                        float(trace_entry.get("expert_sequence_score", np.nan)),
                        dtype=np.float32,
                    )
                    current_raw_row["expert_sequence_verified"] = np.asarray(
                        float(bool(trace_entry.get("expert_sequence_verified", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_ready"] = np.asarray(
                        float(bool(trace_entry.get("teacher_close_ready", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_ready_all"] = np.asarray(
                        float(bool(trace_entry.get("teacher_close_ready_all", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_gate_reason"] = np.asarray(
                        str(trace_entry.get("teacher_close_gate_reason", "unknown"))
                    )
                    current_raw_row["teacher_close_contact_ready"] = np.asarray(
                        float(
                            bool(
                                trace_entry.get(
                                    "teacher_close_contact_ready",
                                    bool(trace_entry.get("teacher_close_contact_ready_by_geometry", False))
                                    and (
                                        bool(trace_entry.get("teacher_close_contact_ready_by_stage", False))
                                        or bool(trace_entry.get("teacher_close_contact_ready_by_depth", False))
                                    ),
                                )
                            )
                        ),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_contact_ready_by_depth"] = np.asarray(
                        float(bool(trace_entry.get("teacher_close_contact_ready_by_depth", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_contact_ready_by_stage"] = np.asarray(
                        float(bool(trace_entry.get("teacher_close_contact_ready_by_stage", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_contact_ready_by_geometry"] = np.asarray(
                        float(bool(trace_entry.get("teacher_close_contact_ready_by_geometry", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_contact_visibility_ok"] = np.asarray(
                        float(bool(trace_entry.get("teacher_close_contact_visibility_ok", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_contact_confidence"] = np.asarray(
                        float(trace_entry.get("teacher_close_contact_confidence", np.nan)),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_grasp_contact_ready"] = np.asarray(
                        float(bool(trace_entry.get("teacher_grasp_contact_ready", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_grasp_ready"] = np.asarray(
                        float(bool(trace_entry.get("teacher_grasp_ready", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_grasp_readiness_score"] = np.asarray(
                        float(trace_entry.get("teacher_grasp_readiness_score", np.nan)),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_grasp_readiness_reason"] = np.asarray(
                        str(trace_entry.get("teacher_grasp_readiness_reason", "unknown"))
                    )
                    current_raw_row["teacher_attached_after_close"] = np.asarray(
                        float(bool(trace_entry.get("teacher_attached_after_close", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_grasped_object_count"] = np.asarray(
                        int(trace_entry.get("teacher_grasped_object_count", 0)),
                        dtype=np.int64,
                    )
                    current_raw_row["teacher_grasped_target_handle_match"] = np.asarray(
                        float(bool(trace_entry.get("teacher_grasped_target_handle_match", False))),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_object_to_gripper_delta_local_6d"] = np.asarray(
                        trace_entry.get(
                            "teacher_object_to_gripper_delta_local_6d",
                            np.full((6,), np.nan, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_demo_basin_distance"] = np.asarray(
                        float(trace_entry.get("teacher_demo_basin_distance", np.nan)),
                        dtype=np.float32,
                    )
                    current_raw_row["teacher_close_basin_source"] = np.asarray(
                        str(trace_entry.get("teacher_close_basin_source", "heuristic"))
                    )
                    current_raw_row["teacher_close_failure_reason"] = np.asarray(
                        str(trace_entry.get("teacher_close_failure_reason", "unknown"))
                    )
                    current_raw_row["teacher_grasp_commit_edge_pair_index"] = np.asarray(
                        int(trace_entry.get("teacher_grasp_commit_edge_pair_index", -1)),
                        dtype=np.int64,
                    )
                    current_raw_row["teacher_grasp_commit_edge_pair_family"] = np.asarray(
                        int(trace_entry.get("teacher_grasp_commit_edge_pair_family", -1)),
                        dtype=np.int64,
                    )
                    current_raw_row["teacher_grasp_commit_edge_pair_yaw_error"] = np.asarray(
                        float(trace_entry.get("teacher_grasp_commit_edge_pair_yaw_error", np.nan)),
                        dtype=np.float32,
                    )
                    current_raw_row["close_success_label"] = np.asarray(
                        float(teacher_grasp_verified_step is not None),
                        dtype=np.float32,
                    )
                    current_raw_row["time_to_close"] = np.asarray(
                        -1 if teacher_close_attempt_step is None else int(teacher_close_attempt_step) - int(step_idx),
                        dtype=np.int64,
                    )
                    current_raw_row["time_to_verified_grasp"] = np.asarray(
                        -1 if teacher_grasp_verified_step is None else int(teacher_grasp_verified_step) - int(step_idx),
                        dtype=np.int64,
                    )
                    if bool(teacher_grasp_verified_now):
                        current_raw_row["teacher_close_action"] = np.asarray("verified_success")
                        current_raw_row["close_failure_reason"] = np.asarray("none")
                    elif bool(teacher_retry_required_now):
                        current_raw_row["teacher_close_action"] = np.asarray("retry")
                        current_raw_row["close_failure_reason"] = np.asarray("verify_failed")
            if args.record_gripper_trace:
                gripper_trace.append(trace_entry)

            if refiner is not None and refiner.should_replan():
                refiner.note_replan()
                action_queue.clear()

            if bool(getattr(args, "oracle_executed_pregrasp_collect", False)) and oracle_pregrasp_success:
                success = True
                if args.record_video:
                    frames.append(obs.front_rgb.copy())
                if not success_reported:
                    print(f"  PREGRASP SUCCESS at step {step_count}")
                    success_reported = True
                if not bool(args.run_full_horizon_on_success):
                    break
            elif bool(tc_close_verify_collect_enabled) and teacher_grasp_verified_step is not None:
                success = True
                if args.record_video:
                    frames.append(obs.front_rgb.copy())
                if not success_reported:
                    print(f"  VERIFIED GRASP SUCCESS at step {step_count}")
                    success_reported = True
                if not bool(args.run_full_horizon_on_success):
                    break

            if not bool(tc_close_verify_collect_enabled) and reward > 0:
                success = True
                if args.record_video:
                    frames.append(obs.front_rgb.copy())
                if not success_reported:
                    print(f"  SUCCESS at step {step_count}")
                    success_reported = True
                if not bool(args.run_full_horizon_on_success):
                    break

        if not success:
            print(f"  FAILED after {step_count} steps")

        if alignment_diffusion_rows is not None and alignment_diffusion_episode_rows is not None:
            episode_rng = np.random.default_rng(int(args.eval_seed or 0) + int(ep_idx) * 9973 + 17)
            finalized_rows = _alignment_diffusion_finalize_episode_rows(
                alignment_diffusion_episode_rows,
                horizon=int(args.alignment_diffusion_raw_horizon),
                micro_depth_threshold=float(args.alignment_diffusion_raw_micro_depth_threshold),
                near_depth_threshold=float(args.alignment_diffusion_raw_near_depth_threshold),
                contact_force_threshold=float(args.alignment_diffusion_raw_contact_force_threshold),
                high_force_threshold=float(args.alignment_diffusion_raw_high_force_threshold),
                force_spike_threshold=float(args.alignment_diffusion_raw_force_spike_threshold),
                action_floor_xy=float(args.alignment_diffusion_raw_action_floor_xy),
                action_floor_z=float(args.alignment_diffusion_raw_action_floor_z),
                action_floor_yaw=float(args.alignment_diffusion_raw_action_floor_yaw),
                xy_gain_margin=float(args.alignment_diffusion_raw_xy_gain_margin),
                z_gain_margin=float(args.alignment_diffusion_raw_z_gain_margin),
                yaw_gain_margin=float(args.alignment_diffusion_raw_yaw_gain_margin),
                augment_copies=int(args.alignment_diffusion_raw_augment_copies),
                rgb_noise_std=float(args.alignment_diffusion_raw_rgb_noise_std),
                depth_noise_std=float(args.alignment_diffusion_raw_depth_noise_std),
                force_noise_std=float(args.alignment_diffusion_raw_force_noise_std),
                proprio_noise_std=float(args.alignment_diffusion_raw_proprio_noise_std),
                action_xy_noise_std=float(args.alignment_diffusion_raw_action_xy_noise_std),
                action_z_noise_std=float(args.alignment_diffusion_raw_action_z_noise_std),
                action_yaw_noise_std=float(args.alignment_diffusion_raw_action_yaw_noise_std),
                sample_weight_near=float(args.alignment_diffusion_raw_sample_weight_near),
                sample_weight_micro=float(args.alignment_diffusion_raw_sample_weight_micro),
                sample_weight_stop=float(args.alignment_diffusion_raw_sample_weight_stop),
                sample_weight_risk=float(args.alignment_diffusion_raw_sample_weight_risk),
                use_privileged_target_delta_gate=bool(args.alignment_tc_raw_use_privileged_delta_gate),
                near_target_xy_threshold=float(args.alignment_tc_raw_near_xy_threshold),
                near_target_abs_z_threshold=float(args.alignment_tc_raw_near_abs_z_threshold),
                near_target_yaw_threshold=float(args.alignment_tc_raw_near_yaw_threshold),
                micro_target_xy_threshold=float(args.alignment_tc_raw_micro_xy_threshold),
                micro_target_abs_z_threshold=float(args.alignment_tc_raw_micro_abs_z_threshold),
                micro_target_yaw_threshold=float(args.alignment_tc_raw_micro_yaw_threshold),
                rng=episode_rng,
            )
            if bool(tc_close_verify_collect_enabled):
                for row in finalized_rows:
                    row_step = int(np.asarray(row.get("rollout_step", -1)).reshape(()))
                    close_delta = (
                        int(teacher_close_attempt_step) - row_step
                        if teacher_close_attempt_step is not None and int(teacher_close_attempt_step) >= 0
                        else -1
                    )
                    verified_delta = (
                        int(teacher_grasp_verified_step) - row_step
                        if teacher_grasp_verified_step is not None and int(teacher_grasp_verified_step) >= 0
                        else -1
                    )
                    row["teacher_close_attempt_step"] = np.asarray(
                        -1 if teacher_close_attempt_step is None else int(teacher_close_attempt_step),
                        dtype=np.int64,
                    )
                    row["teacher_grasp_verified_step"] = np.asarray(
                        -1 if teacher_grasp_verified_step is None else int(teacher_grasp_verified_step),
                        dtype=np.int64,
                    )
                    row["expert_sequence_name"] = np.asarray(
                        str(trace_entry.get("expert_sequence_name", "unknown"))
                    )
                    row["expert_sequence_score"] = np.asarray(
                        float(trace_entry.get("expert_sequence_score", np.nan)),
                        dtype=np.float32,
                    )
                    row["expert_sequence_verified"] = np.asarray(
                        float(bool(trace_entry.get("expert_sequence_verified", False))),
                        dtype=np.float32,
                    )
                    row["teacher_object_lift"] = np.asarray(
                        float(trace_entry.get("teacher_object_lift", np.nan)),
                        dtype=np.float32,
                    )
                    row["teacher_object_follows_gripper"] = np.asarray(
                        float(bool(trace_entry.get("teacher_object_follows_gripper", False))),
                        dtype=np.float32,
                    )
                    row["teacher_verify_object_pose_available"] = np.asarray(
                        float(bool(trace_entry.get("teacher_verify_object_pose_available", False))),
                        dtype=np.float32,
                    )
                    row["teacher_verify_object_pose_source"] = np.asarray(
                        str(trace_entry.get("teacher_verify_object_pose_source", "missing"))
                    )
                    row["teacher_verify_object_is_grasped"] = np.asarray(
                        float(bool(trace_entry.get("teacher_verify_object_is_grasped", False))),
                        dtype=np.float32,
                    )
                    row["teacher_verify_follow_distance"] = np.asarray(
                        float(trace_entry.get("teacher_verify_follow_distance", np.nan)),
                        dtype=np.float32,
                    )
                    row["teacher_verify_fail_reason"] = np.asarray(
                        str(trace_entry.get("teacher_verify_fail_reason", "unknown"))
                    )
                    row["teacher_close_settle_phase"] = np.asarray(
                        str(trace_entry.get("teacher_close_settle_phase", "none"))
                    )
                    row["teacher_close_ready"] = np.asarray(
                        float(bool(row.get("teacher_close_ready", trace_entry.get("teacher_close_ready", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_close_ready_all"] = np.asarray(
                        float(bool(row.get("teacher_close_ready_all", trace_entry.get("teacher_close_ready_all", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_close_gate_reason"] = np.asarray(
                        str(row.get("teacher_close_gate_reason", trace_entry.get("teacher_close_gate_reason", "unknown")))
                    )
                    row["teacher_close_contact_ready"] = np.asarray(
                        float(
                            bool(
                                row.get(
                                    "teacher_close_contact_ready",
                                    trace_entry.get(
                                        "teacher_close_contact_ready",
                                        bool(trace_entry.get("teacher_close_contact_ready_by_geometry", False))
                                        and (
                                            bool(trace_entry.get("teacher_close_contact_ready_by_stage", False))
                                            or bool(trace_entry.get("teacher_close_contact_ready_by_depth", False))
                                        ),
                                    ),
                                )
                            )
                        ),
                        dtype=np.float32,
                    )
                    row["teacher_close_contact_ready_by_depth"] = np.asarray(
                        float(bool(row.get("teacher_close_contact_ready_by_depth", trace_entry.get("teacher_close_contact_ready_by_depth", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_close_contact_ready_by_stage"] = np.asarray(
                        float(bool(row.get("teacher_close_contact_ready_by_stage", trace_entry.get("teacher_close_contact_ready_by_stage", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_close_contact_ready_by_geometry"] = np.asarray(
                        float(bool(row.get("teacher_close_contact_ready_by_geometry", trace_entry.get("teacher_close_contact_ready_by_geometry", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_close_contact_visibility_ok"] = np.asarray(
                        float(bool(row.get("teacher_close_contact_visibility_ok", trace_entry.get("teacher_close_contact_visibility_ok", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_close_contact_confidence"] = np.asarray(
                        float(row.get("teacher_close_contact_confidence", trace_entry.get("teacher_close_contact_confidence", np.nan))),
                        dtype=np.float32,
                    )
                    row["teacher_grasp_contact_ready"] = np.asarray(
                        float(bool(row.get("teacher_grasp_contact_ready", trace_entry.get("teacher_grasp_contact_ready", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_grasp_ready"] = np.asarray(
                        float(bool(row.get("teacher_grasp_ready", trace_entry.get("teacher_grasp_ready", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_grasp_readiness_score"] = np.asarray(
                        float(row.get("teacher_grasp_readiness_score", trace_entry.get("teacher_grasp_readiness_score", np.nan))),
                        dtype=np.float32,
                    )
                    row["teacher_grasp_readiness_reason"] = np.asarray(
                        str(row.get("teacher_grasp_readiness_reason", trace_entry.get("teacher_grasp_readiness_reason", "unknown")))
                    )
                    row["teacher_attached_after_close"] = np.asarray(
                        float(bool(row.get("teacher_attached_after_close", trace_entry.get("teacher_attached_after_close", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_grasped_object_count"] = np.asarray(
                        int(row.get("teacher_grasped_object_count", trace_entry.get("teacher_grasped_object_count", 0))),
                        dtype=np.int64,
                    )
                    row["teacher_grasped_target_handle_match"] = np.asarray(
                        float(bool(row.get("teacher_grasped_target_handle_match", trace_entry.get("teacher_grasped_target_handle_match", False)))),
                        dtype=np.float32,
                    )
                    row["teacher_object_to_gripper_delta_local_6d"] = np.asarray(
                        row.get(
                            "teacher_object_to_gripper_delta_local_6d",
                            trace_entry.get(
                                "teacher_object_to_gripper_delta_local_6d",
                                np.full((6,), np.nan, dtype=np.float32),
                            ),
                        ),
                        dtype=np.float32,
                    )
                    row["teacher_demo_basin_distance"] = np.asarray(
                        float(row.get("teacher_demo_basin_distance", trace_entry.get("teacher_demo_basin_distance", np.nan))),
                        dtype=np.float32,
                    )
                    row["teacher_close_basin_source"] = np.asarray(
                        str(row.get("teacher_close_basin_source", trace_entry.get("teacher_close_basin_source", "heuristic")))
                    )
                    row["teacher_close_failure_reason"] = np.asarray(
                        str(row.get("teacher_close_failure_reason", trace_entry.get("teacher_close_failure_reason", "unknown")))
                    )
                    row["teacher_grasp_commit_edge_pair_index"] = np.asarray(
                        int(row.get("teacher_grasp_commit_edge_pair_index", trace_entry.get("teacher_grasp_commit_edge_pair_index", -1))),
                        dtype=np.int64,
                    )
                    row["teacher_grasp_commit_edge_pair_family"] = np.asarray(
                        int(row.get("teacher_grasp_commit_edge_pair_family", trace_entry.get("teacher_grasp_commit_edge_pair_family", -1))),
                        dtype=np.int64,
                    )
                    row["teacher_grasp_commit_edge_pair_yaw_error"] = np.asarray(
                        float(row.get("teacher_grasp_commit_edge_pair_yaw_error", trace_entry.get("teacher_grasp_commit_edge_pair_yaw_error", np.nan))),
                        dtype=np.float32,
                    )
                    row["teacher_gripper_finger_pose_7d"] = np.asarray(
                        trace_entry.get("teacher_gripper_finger_pose_7d", np.asarray(obs.gripper_pose, dtype=np.float32)),
                        dtype=np.float32,
                    )
                    row["teacher_object_in_finger_region"] = np.asarray(
                        float(bool(trace_entry.get("teacher_object_in_finger_region", False))),
                        dtype=np.float32,
                    )
                    row["teacher_finger_object_lateral_error"] = np.asarray(
                        float(trace_entry.get("teacher_finger_object_lateral_error", np.nan)),
                        dtype=np.float32,
                    )
                    row["teacher_finger_object_height_overlap"] = np.asarray(
                        float(trace_entry.get("teacher_finger_object_height_overlap", np.nan)),
                        dtype=np.float32,
                    )
                    row["teacher_finger_object_yaw_error"] = np.asarray(
                        float(trace_entry.get("teacher_finger_object_yaw_error", np.nan)),
                        dtype=np.float32,
                    )
                    row["teacher_grasp_contact_ready"] = np.asarray(
                        float(bool(trace_entry.get("teacher_grasp_contact_ready", False))),
                        dtype=np.float32,
                    )
                    row["teacher_grasp_ready"] = np.asarray(
                        float(bool(trace_entry.get("teacher_grasp_ready", False))),
                        dtype=np.float32,
                    )
                    row["teacher_grasp_readiness_score"] = np.asarray(
                        float(trace_entry.get("teacher_grasp_readiness_score", np.nan)),
                        dtype=np.float32,
                    )
                    row["teacher_grasp_readiness_reason"] = np.asarray(
                        str(trace_entry.get("teacher_grasp_readiness_reason", "unknown"))
                    )
                    row["close_success_label"] = np.asarray(float(teacher_grasp_verified_step is not None), dtype=np.float32)
                    row["time_to_close"] = np.asarray(close_delta, dtype=np.int64)
                    row["time_to_verified_grasp"] = np.asarray(verified_delta, dtype=np.int64)
                    if teacher_grasp_verified_step is not None and 0 <= verified_delta <= 1:
                        row["teacher_grasp_verified"] = np.asarray(1.0, dtype=np.float32)
                        row["teacher_verify_lift_streak"] = np.asarray(
                            int(max(teacher_verify_lift_streak, int(getattr(args, "teacher_verify_min_consecutive_lift_steps", 2)))),
                            dtype=np.int64,
                        )
                        row["teacher_close_action"] = np.asarray("verified_success")
                        row["stop_label"] = np.asarray(1.0, dtype=np.float32).reshape(1)
                        row["stop_reason"] = np.asarray("verified_success")
                        row["no_op_reason"] = np.asarray("verified_success")
                        row["risk_label"] = np.asarray(0.0, dtype=np.float32).reshape(1)
                        row["risk_reason"] = np.asarray("verified_success")
                    elif close_delta == 0:
                        row["teacher_close_action"] = np.asarray("close_now")
                        row["stop_label"] = np.asarray(1.0, dtype=np.float32).reshape(1)
                        row["stop_reason"] = np.asarray("close_now")
                        row["no_op_reason"] = np.asarray("close_now")
                        row["risk_label"] = np.asarray(0.0, dtype=np.float32).reshape(1)
                        row["risk_reason"] = np.asarray("safe_close")
            alignment_diffusion_rows.extend(finalized_rows)

        if support_state_rows is not None and support_row_start is not None:
            closed_steps = [bool(item.get("abs_gripper_cmd", 1.0) < 0.5) for item in gripper_trace]
            best_hold_run = 0
            best_hold_end = -1
            hold_run = 0
            for idx, closed in enumerate(closed_steps):
                if closed:
                    hold_run += 1
                    if hold_run > best_hold_run:
                        best_hold_run = hold_run
                        best_hold_end = idx
                else:
                    hold_run = 0
            best_hold_start_step = None
            if best_hold_run > 0 and best_hold_end >= 0:
                best_hold_start_step = int(gripper_trace[best_hold_end - best_hold_run + 1]["step"])
            fire_anchor_step = teacher_close_attempt_step if bool(getattr(args, "oracle_executed_pregrasp_collect", False)) else None
            if bool(getattr(args, "oracle_executed_pregrasp_collect", False)):
                for row in support_state_rows[support_row_start:]:
                    row_step = int(row["rollout_step"])
                    close_delta = (
                        (int(teacher_close_attempt_step) - row_step)
                        if teacher_close_attempt_step is not None and int(teacher_close_attempt_step) >= 0
                        else None
                    )
                    verified_delta = (
                        (int(teacher_grasp_verified_step) - row_step)
                        if teacher_grasp_verified_step is not None and int(teacher_grasp_verified_step) >= 0
                        else None
                    )
                    retry_delta = None
                    if teacher_retry_anchor_steps:
                        retry_delta = min((int(s) - row_step) for s in teacher_retry_anchor_steps)
                    ready_local = bool(close_delta is not None and 0 <= int(close_delta) <= int(args.fire_positive_window))
                    grasp_verified_local = bool(
                        oracle_pregrasp_success
                        and verified_delta is not None
                        and 0 <= int(verified_delta) <= 1
                    )
                    retry_local = bool(retry_delta is not None and 0 <= int(retry_delta) <= int(args.fire_negative_gap))
                    row["ready_to_close_target"] = np.asarray(float(ready_local), dtype=np.float32)
                    row["post_close_stability_proxy"] = np.asarray(float(oracle_pregrasp_success), dtype=np.float32)
                    row["grasp_lift_proxy"] = np.asarray(float(oracle_pregrasp_success), dtype=np.float32)
                    row["grasp_verified_target"] = np.asarray(float(grasp_verified_local), dtype=np.float32)
                    row["retry_required_target"] = np.asarray(float(retry_local), dtype=np.float32)
                    row["reopen_after_trigger"] = np.asarray(float(retry_local), dtype=np.float32)
                    row["invalid_after_trigger"] = np.asarray(float(0.0), dtype=np.float32)
                    row["planner_close_too_early"] = np.asarray(float(0.0), dtype=np.float32)
            else:
                positive_fire_episode = bool(grasp_lift_proxy or best_hold_run >= int(args.fire_hold_min_steps))
                fire_anchor_step = best_hold_start_step if positive_fire_episode else first_close_step
                for row in support_state_rows[support_row_start:]:
                    row_step = int(row["rollout_step"])
                    anchor_delta = (int(fire_anchor_step) - row_step) if fire_anchor_step is not None and int(fire_anchor_step) >= 0 else None
                    close_delta = (int(first_close_step) - row_step) if first_close_step is not None and int(first_close_step) >= 0 else None
                    ready_local = bool(
                        positive_fire_episode
                        and anchor_delta is not None
                        and 0 <= int(anchor_delta) <= int(args.fire_positive_window)
                    )
                    early_close_local = bool(
                        first_close_step is not None
                        and close_before_basin_count > 0
                        and close_delta is not None
                        and 0 <= int(close_delta) <= int(args.fire_negative_gap)
                    )
                    reopen_local = bool(
                        first_close_step is not None
                        and reopen_after_close_count > 0
                        and close_delta is not None
                        and 0 <= int(close_delta) <= int(args.fire_negative_gap)
                    )
                    invalid_local = bool(
                        first_close_step is not None
                        and invalid_action_count > 0
                        and close_delta is not None
                        and 0 <= int(close_delta) <= int(args.fire_negative_gap)
                    )
                    row["ready_to_close_target"] = np.asarray(
                        float(ready_local),
                        dtype=np.float32,
                    )
                    row["post_close_stability_proxy"] = np.asarray(float(ready_local), dtype=np.float32)
                    row["grasp_lift_proxy"] = np.asarray(float(ready_local and grasp_lift_proxy), dtype=np.float32)
                    row["reopen_after_trigger"] = np.asarray(
                        float(reopen_local),
                        dtype=np.float32,
                    )
                    row["invalid_after_trigger"] = np.asarray(
                        float(invalid_local),
                        dtype=np.float32,
                    )
                    row["planner_close_too_early"] = np.asarray(
                        float(early_close_local),
                        dtype=np.float32,
                    )

        episode_invalid_action_trace_count = (
            int(sum(1 for item in gripper_trace if bool(item.get("invalid_action", False))))
            if args.record_gripper_trace
            else int(invalid_action_count)
        )

        results["successes"].append(success)
        results["episode_lengths"].append(step_count)
        results["invalid_action_counts"].append(episode_invalid_action_trace_count)
        refiner_snapshot = refiner.get_stats() if isinstance(refiner, StageAwareRefiner) else {}
        coarse2contact_snapshot = coarse2contact.get_stats() if coarse2contact is not None else {}
        results["stage_stats"].append(
            {
                "episode_index": int(ep_idx),
                "phase_counts": phase_hist,
                "failure_counts": failure_hist,
                "invalid_action_count": episode_invalid_action_trace_count,
                "invalid_action_exception_count": int(invalid_action_exception_count),
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
                "oracle_executed_alignment_count": int(oracle_executed_alignment_count),
                "oracle_executed_noop_count": int(oracle_executed_noop_count),
                "oracle_pregrasp_success": bool(oracle_pregrasp_success),
                "oracle_pregrasp_close_count": int(oracle_pregrasp_close_count),
                "oracle_pregrasp_retry_count": int(oracle_pregrasp_retry_count),
                "teacher_close_attempt_step": -1 if teacher_close_attempt_step is None else int(teacher_close_attempt_step),
                "teacher_grasp_verified_step": -1 if teacher_grasp_verified_step is None else int(teacher_grasp_verified_step),
                "best_hold_run": int(best_hold_run) if support_state_rows is not None and support_row_start is not None else 0,
                "fire_anchor_step": -1 if (support_state_rows is None or support_row_start is None or fire_anchor_step is None) else int(fire_anchor_step),
                "substage_id": int(stage_tracker.substage),
                "substage": stage_tracker.substage.name,
                "has_object_in_hand": bool(stage_tracker.has_object_in_hand),
                "phase1_verified_grasp_reached": bool(
                    stage_tracker.has_object_in_hand
                    or teacher_grasp_verified_step is not None
                    or grasp_lift_proxy
                ),
                "phase1_grasp_contact_confirmed": bool(
                    refiner_snapshot.get("phase1_grasp_contact_confirmed", False)
                ) if isinstance(refiner, StageAwareRefiner) else False,
                "phase1_close_command_source": str(
                    refiner_snapshot.get("phase1_close_command_source", "none")
                ) if isinstance(refiner, StageAwareRefiner) else "none",
                "phase1_reopen_reason": str(
                    refiner_snapshot.get("phase1_reopen_reason", "none")
                ) if isinstance(refiner, StageAwareRefiner) else "none",
                "phase1_force_spike_count": int(
                    refiner_snapshot.get("phase1_force_spike_count", 0)
                ) if isinstance(refiner, StageAwareRefiner) else 0,
                "phase1_jam_detected_count": int(
                    refiner_snapshot.get("phase1_jam_detected_count", 0)
                ) if isinstance(refiner, StageAwareRefiner) else 0,
                "contact_state_id": int(stage_tracker.contact_state),
                "contact_state": stage_tracker.contact_state.name,
                "coarse2contact_phase": str(coarse2contact_snapshot.get("coarse2contact_phase", "off")),
                "coarse2contact_precontact_count": int(
                    coarse2contact_snapshot.get("coarse2contact_precontact_count", 0)
                ),
                "coarse2contact_reaches_precontact": bool(
                    int(coarse2contact_snapshot.get("coarse2contact_precontact_count", 0)) > 0
                ),
                "coarse2contact_preinsert_count": int(
                    coarse2contact_snapshot.get("coarse2contact_preinsert_count", 0)
                ),
                "coarse2contact_reaches_preinsert": bool(
                    int(coarse2contact_snapshot.get("coarse2contact_preinsert_count", 0)) > 0
                ),
                "coarse2contact_correction_count": int(
                    coarse2contact_snapshot.get("coarse2contact_correction_count", 0)
                ),
                "coarse2contact_recovery_count": int(
                    coarse2contact_snapshot.get("coarse2contact_recovery_count", 0)
                ),
                "coarse2contact_invalid_action_count": int(
                    coarse2contact_snapshot.get("coarse2contact_invalid_action_count", 0)
                ),
                "uses_privileged_target": False,
                "stage_target_mode_id": int(stage_tracker.stage_target_mode),
                "stage_target_mode": stage_tracker.stage_target_mode.name,
                "refiner_phase_id": int(refiner_snapshot.get("phase_id", -1))
                if isinstance(refiner, StageAwareRefiner) else None,
                "refiner_max_phase_reached": int(refiner_snapshot.get("max_phase_reached", -1))
                if isinstance(refiner, StageAwareRefiner) else None,
                "refiner_subgoal_progress": float(refiner_snapshot.get("subgoal_progress", 0.0))
                if isinstance(refiner, StageAwareRefiner) else None,
                "target_provider_name": None if not isinstance(refiner, StageAwareRefiner) else refiner_snapshot.get("current_target_provider_name", None),
                "target_provider_source": None if not isinstance(refiner, StageAwareRefiner) else refiner_snapshot.get("current_target_provider_source", None),
                "target_uses_privileged_runtime": False if not isinstance(refiner, StageAwareRefiner) else bool(refiner_snapshot.get("current_target_uses_privileged_runtime", False)),
                "handoff_ready": False if not isinstance(refiner, StageAwareRefiner) else bool(refiner_snapshot.get("current_handoff_ready", False)),
                "handoff_metrics": {} if not isinstance(refiner, StageAwareRefiner) else dict(refiner_snapshot.get("current_handoff_metrics", {}) or {}),
                "handoff_spec_name": None if not isinstance(refiner, StageAwareRefiner) else refiner_snapshot.get("current_handoff_spec_name", None),
                "handoff_target_role": None if not isinstance(refiner, StageAwareRefiner) else refiner_snapshot.get("current_handoff_target_role", None),
                "near_ready_group_residual_gate_pass_count": int(refiner_snapshot.get("near_ready_group_residual_gate_pass_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "near_ready_group_residual_apply_count": int(refiner_snapshot.get("near_ready_group_residual_apply_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "near_ready_group_residual_change_count": int(refiner_snapshot.get("near_ready_group_residual_change_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_gate_pass_count": int(refiner_snapshot.get("b1_group_shadow_gate_pass_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_gate_mode": str(refiner_snapshot.get("b1_group_shadow_gate_mode", "broad"))
                if isinstance(refiner, StageAwareRefiner) else "broad",
                "b1_group_shadow_change_count": int(refiner_snapshot.get("b1_group_shadow_change_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_disagreement_count": int(refiner_snapshot.get("b1_group_shadow_disagreement_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_teacher_group_valid_count": int(refiner_snapshot.get("b1_group_shadow_teacher_group_valid_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_cost_valid_count": int(refiner_snapshot.get("b1_group_shadow_cost_valid_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_cost_improve_count": int(refiner_snapshot.get("b1_group_shadow_cost_improve_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_cost_worse_count": int(refiner_snapshot.get("b1_group_shadow_cost_worse_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_regret_delta_sum": float(refiner_snapshot.get("b1_group_shadow_regret_delta_sum", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "b1_group_shadow_close_count": int(refiner_snapshot.get("b1_group_shadow_close_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_close_change_count": int(refiner_snapshot.get("b1_group_shadow_close_change_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_close_cost_valid_count": int(refiner_snapshot.get("b1_group_shadow_close_cost_valid_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_close_cost_improve_count": int(refiner_snapshot.get("b1_group_shadow_close_cost_improve_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_close_cost_worse_count": int(refiner_snapshot.get("b1_group_shadow_close_cost_worse_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_shadow_close_regret_delta_sum": float(refiner_snapshot.get("b1_group_shadow_close_regret_delta_sum", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "b2_candidate_shadow_eval_count": int(refiner_snapshot.get("b2_candidate_shadow_eval_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_gate_pass_count": int(refiner_snapshot.get("b2_candidate_shadow_gate_pass_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_gate_mode": str(refiner_snapshot.get("b2_candidate_shadow_gate_mode", "broad"))
                if isinstance(refiner, StageAwareRefiner) else "broad",
                "b2_candidate_shadow_change_count": int(refiner_snapshot.get("b2_candidate_shadow_change_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_cost_valid_count": int(refiner_snapshot.get("b2_candidate_shadow_cost_valid_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_cost_improve_count": int(refiner_snapshot.get("b2_candidate_shadow_cost_improve_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_cost_worse_count": int(refiner_snapshot.get("b2_candidate_shadow_cost_worse_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_regret_delta_sum": float(refiner_snapshot.get("b2_candidate_shadow_regret_delta_sum", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "b2_candidate_shadow_mode_keep_count": int(refiner_snapshot.get("b2_candidate_shadow_mode_keep_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_mode_apply_count": int(refiner_snapshot.get("b2_candidate_shadow_mode_apply_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_close_count": int(refiner_snapshot.get("b2_candidate_shadow_close_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_yaw_needed_count": int(refiner_snapshot.get("b2_candidate_shadow_yaw_needed_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_yaw_keep_count": int(refiner_snapshot.get("b2_candidate_shadow_yaw_keep_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_teacher_ready_count": int(refiner_snapshot.get("b2_candidate_shadow_teacher_ready_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_xy_block_count": int(refiner_snapshot.get("b2_candidate_shadow_xy_block_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_nearish_count": int(refiner_snapshot.get("b2_candidate_shadow_nearish_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_shadow_keep_baseline_forced_count": int(refiner_snapshot.get("b2_candidate_shadow_keep_baseline_forced_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_bounded_eval_count": int(refiner_snapshot.get("b2_candidate_bounded_eval_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_bounded_gate_pass_count": int(refiner_snapshot.get("b2_candidate_bounded_gate_pass_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_bounded_apply_count": int(refiner_snapshot.get("b2_candidate_bounded_apply_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b2_candidate_bounded_change_count": int(refiner_snapshot.get("b2_candidate_bounded_change_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_eval_count": int(refiner_snapshot.get("b1_apply_gate_eval_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_apply_count": int(refiner_snapshot.get("b1_apply_gate_apply_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_cost_valid_count": int(refiner_snapshot.get("b1_apply_gate_cost_valid_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_cost_improve_count": int(refiner_snapshot.get("b1_apply_gate_cost_improve_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_cost_worse_count": int(refiner_snapshot.get("b1_apply_gate_cost_worse_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_regret_delta_sum": float(refiner_snapshot.get("b1_apply_gate_regret_delta_sum", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "b1_apply_gate_close_cost_valid_count": int(refiner_snapshot.get("b1_apply_gate_close_cost_valid_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_close_cost_improve_count": int(refiner_snapshot.get("b1_apply_gate_close_cost_improve_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_close_cost_worse_count": int(refiner_snapshot.get("b1_apply_gate_close_cost_worse_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_apply_gate_close_regret_delta_sum": float(refiner_snapshot.get("b1_apply_gate_close_regret_delta_sum", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "b1_group_bounded_apply_count": int(refiner_snapshot.get("b1_group_bounded_apply_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "b1_group_bounded_change_count": int(refiner_snapshot.get("b1_group_bounded_change_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "alignment_diffusion_controller_type": str(refiner_snapshot.get("current_alignment_diffusion_controller_type", "none"))
                if isinstance(refiner, StageAwareRefiner) else "none",
                "alignment_diffusion_eval_count": int(refiner_snapshot.get("current_alignment_diffusion_eval_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "alignment_diffusion_active_count": int(refiner_snapshot.get("current_alignment_diffusion_active_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "alignment_diffusion_apply_count": int(refiner_snapshot.get("current_alignment_diffusion_apply_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "alignment_diffusion_block_reason_hist": dict(refiner_snapshot.get("alignment_diffusion_block_reason_hist", {}) or {})
                if isinstance(refiner, StageAwareRefiner) else {},
                "alignment_diffusion_phase_hist": dict(refiner_snapshot.get("alignment_diffusion_phase_hist", {}) or {})
                if isinstance(refiner, StageAwareRefiner) else {},
                "alignment_diffusion_bucket_hist": dict(refiner_snapshot.get("alignment_diffusion_bucket_hist", {}) or {})
                if isinstance(refiner, StageAwareRefiner) else {},
                "alignment_diffusion_controller_type_hist": dict(refiner_snapshot.get("alignment_diffusion_controller_type_hist", {}) or {})
                if isinstance(refiner, StageAwareRefiner) else {},
                "alignment_diffusion_confidence_mean": float(refiner_snapshot.get("alignment_diffusion_confidence_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_risk_prob_mean": float(refiner_snapshot.get("alignment_diffusion_risk_prob_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_stop_prob_mean": float(refiner_snapshot.get("alignment_diffusion_stop_prob_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_scale_down_mean": float(refiner_snapshot.get("alignment_diffusion_scale_down_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_pos_norm_mean": float(refiner_snapshot.get("alignment_diffusion_pos_norm_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_yaw_abs_mean": float(refiner_snapshot.get("alignment_diffusion_yaw_abs_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_pred_target_delta_norm_mean": float(refiner_snapshot.get("alignment_diffusion_pred_target_delta_norm_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_pred_target_yaw_abs_mean": float(refiner_snapshot.get("alignment_diffusion_pred_target_yaw_abs_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_target_action_sign_agreement_mean": float(refiner_snapshot.get("alignment_diffusion_target_action_sign_agreement_mean", 0.0))
                if isinstance(refiner, StageAwareRefiner) else 0.0,
                "alignment_diffusion_low_confidence_count": int(refiner_snapshot.get("alignment_diffusion_low_confidence_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "alignment_diffusion_soft_clamp_count": int(refiner_snapshot.get("alignment_diffusion_soft_clamp_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "alignment_diffusion_hard_reject_count": int(refiner_snapshot.get("alignment_diffusion_hard_reject_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "alignment_diffusion_safety_reject_count": int(refiner_snapshot.get("alignment_diffusion_safety_reject_count", 0))
                if isinstance(refiner, StageAwareRefiner) else 0,
                "has_object_in_hand_entered": bool(stage_tracker.has_object_in_hand),
                "phase2_entered_after_student": bool(
                    stage_tracker.has_object_in_hand
                    and int(dict(refiner_snapshot.get("alignment_diffusion_phase_hist", {}) or {}).get("insert_commit", 0)) > 0
                ) if isinstance(refiner, StageAwareRefiner) else False,
                "insert_commit_steps_after_bridge": int(
                    dict(refiner_snapshot.get("alignment_diffusion_phase_hist", {}) or {}).get("insert_commit", 0)
                ) if isinstance(refiner, StageAwareRefiner) else 0,
            }
        )
        episode_mp4_path = None
        if args.record_video and args.write_episode_videos and len(frames) > 1:
            status = "succ" if success else "fail"
            episode_mp4_path = str(video_dir / f"ep{ep_idx:03d}_{status}.mp4")
            results["video_paths"].append(episode_mp4_path)
            if args.record_gripper_trace:
                for item in gripper_trace:
                    item["mp4_path"] = episode_mp4_path
            if coarse2contact is not None:
                coarse2contact._last_trace["mp4_path"] = episode_mp4_path

        if refiner is not None:
            results["refiner_stats"].append(refiner.get_stats())
        if coarse2contact is not None:
            results["coarse2contact_stats"].append(coarse2contact.get_stats())
        if gripper_supervisor is not None:
            results["gripper_supervisor_stats"].append(_jsonable_value(gripper_supervisor.get_stats()))
        if args.record_gripper_trace:
            trace_path = gripper_trace_dir / f"ep{ep_idx:03d}_gripper_trace.jsonl"
            with open(trace_path, "w") as f:
                for item in gripper_trace:
                    f.write(json.dumps(item, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o) if hasattr(o, "item") else str(o)) + "\n")
            results["gripper_trace_paths"].append(str(trace_path))

        if args.record_video and len(frames) > 1:
            if args.write_episode_videos:
                clip = ImageSequenceClip(frames, fps=20)
                vid_path = episode_mp4_path or str(video_dir / f"ep{ep_idx:03d}_{'succ' if success else 'fail'}.mp4")
                clip.write_videofile(vid_path, fps=20, codec="libx264", bitrate="3000k", logger=None)

            if track_best_gif:
                if success and len(frames) < best_success_len:
                    best_success_len = len(frames)
                    best_success_frames = frames
                elif not success and step_count > best_fail_len:
                    best_fail_len = step_count
                    best_fail_frames = list(frames)

    gif_frames = best_success_frames if best_success_frames is not None else best_fail_frames
    gif_label = "best_success" if best_success_frames is not None else "best_fail"
    if track_best_gif and gif_frames is not None and len(gif_frames) > 1:
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
    results["planner_reaches_precontact_rate"] = (
        float(np.mean([s.get("coarse2contact_reaches_precontact", False) for s in results["stage_stats"]]))
        if results["stage_stats"]
        else 0.0
    )
    results["planner_reaches_preinsert_rate"] = (
        float(np.mean([s.get("coarse2contact_reaches_preinsert", False) for s in results["stage_stats"]]))
        if results["stage_stats"]
        else 0.0
    )
    results["coarse2contact_precontact_count"] = int(
        sum(int(s.get("coarse2contact_precontact_count", 0)) for s in results["stage_stats"])
    )
    results["planner_reaches_preinsert_count"] = int(
        sum(int(s.get("coarse2contact_preinsert_count", 0)) for s in results["stage_stats"])
    )
    results["coarse2contact_correction_count"] = int(
        sum(int(s.get("coarse2contact_correction_count", 0)) for s in results["stage_stats"])
    )
    results["coarse2contact_recovery_count"] = int(
        sum(int(s.get("coarse2contact_recovery_count", 0)) for s in results["stage_stats"])
    )
    results["coarse2contact_invalid_action_count"] = int(
        sum(int(s.get("coarse2contact_invalid_action_count", 0)) for s in results["stage_stats"])
    )
    results["uses_privileged_target"] = False
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
    results["oracle_executed_alignment_count"] = int(sum(s.get("oracle_executed_alignment_count", 0) for s in results["stage_stats"]))
    results["oracle_executed_noop_count"] = int(sum(s.get("oracle_executed_noop_count", 0) for s in results["stage_stats"]))
    results["oracle_pregrasp_success_count"] = int(sum(1 for s in results["stage_stats"] if s.get("oracle_pregrasp_success", False)))
    results["oracle_pregrasp_close_count"] = int(sum(s.get("oracle_pregrasp_close_count", 0) for s in results["stage_stats"]))
    results["oracle_pregrasp_retry_count"] = int(sum(s.get("oracle_pregrasp_retry_count", 0) for s in results["stage_stats"]))
    results["target_uses_privileged_runtime_count"] = int(
        sum(1 for s in results["stage_stats"] if s.get("target_uses_privileged_runtime", False))
    )
    provider_hist = {}
    provider_source_hist = {}
    substage_hist = {}
    for s in results["stage_stats"]:
        provider = str(s.get("target_provider_name", "None"))
        provider_hist[provider] = provider_hist.get(provider, 0) + 1
        provider_source = str(s.get("target_provider_source", "None"))
        provider_source_hist[provider_source] = provider_source_hist.get(provider_source, 0) + 1
        substage = str(s.get("substage", "TRANSIT"))
        substage_hist[substage] = substage_hist.get(substage, 0) + 1
    results["target_provider_hist"] = provider_hist
    results["target_provider_source_hist"] = provider_source_hist
    results["final_substage_hist"] = substage_hist
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
            "outer_rescue_decision_count",
            "outer_rescue_correction_count",
            "outer_rescue_noop_count",
            "outer_rescue_handoff_count",
            "readiness_eval_count",
            "preclose_alignment_correction_count",
            "ready_positive_rate",
            "basin_positive_rate",
            "close_veto_block_count",
            "close_veto_pass_count",
            "close_veto_settle_count",
            "alignment_takeover_count",
            "close_latch_set_count",
            "close_latch_release_count",
            "close_latch_expire_count",
            "readiness_close_override_count",
            "readiness_open_override_count",
            "readiness_hold_override_count",
            "readiness_heads_missing_count",
            "near_ready_group_residual_eval_count",
            "near_ready_group_residual_gate_pass_count",
            "near_ready_group_residual_apply_count",
            "near_ready_group_residual_change_count",
            "b1_group_shadow_eval_count",
            "b1_group_shadow_gate_pass_count",
            "b1_group_shadow_change_count",
            "b1_group_shadow_disagreement_count",
            "b1_group_shadow_teacher_group_valid_count",
            "b1_group_shadow_cost_valid_count",
            "b1_group_shadow_cost_improve_count",
            "b1_group_shadow_cost_worse_count",
            "b1_group_shadow_regret_delta_sum",
            "b1_group_shadow_close_count",
            "b1_group_shadow_close_change_count",
            "b1_group_shadow_close_cost_valid_count",
            "b1_group_shadow_close_cost_improve_count",
            "b1_group_shadow_close_cost_worse_count",
            "b1_group_shadow_close_regret_delta_sum",
            "b2_candidate_shadow_eval_count",
            "b2_candidate_shadow_gate_pass_count",
            "b2_candidate_shadow_change_count",
            "b2_candidate_shadow_cost_valid_count",
            "b2_candidate_shadow_cost_improve_count",
            "b2_candidate_shadow_cost_worse_count",
            "b2_candidate_shadow_regret_delta_sum",
            "b2_candidate_shadow_mode_keep_count",
            "b2_candidate_shadow_mode_apply_count",
            "b2_candidate_shadow_close_count",
            "b2_candidate_shadow_yaw_needed_count",
            "b2_candidate_shadow_yaw_keep_count",
            "b2_candidate_shadow_teacher_ready_count",
            "b2_candidate_shadow_xy_block_count",
            "b2_candidate_shadow_nearish_count",
            "b2_candidate_shadow_keep_baseline_forced_count",
            "b2_candidate_bounded_eval_count",
            "b2_candidate_bounded_gate_pass_count",
            "b2_candidate_bounded_apply_count",
            "b2_candidate_bounded_change_count",
        ]:
            if key.endswith("_rate"):
                values = [float(s.get(key, 0.0)) for s in ref_stats]
                results[key] = float(sum(values) / max(len(values), 1))
            elif key.endswith("_sum"):
                results[key] = float(sum(float(s.get(key, 0.0)) for s in ref_stats))
            else:
                results[key] = int(sum(s.get(key, 0) for s in ref_stats))
        for key in [
            "alpha_mean",
            "residual_pos_norm_mean",
            "raw_delta_norm_mean",
            "executed_delta_norm_mean",
            "clip_hit_rate",
            "planner_close_intent_rate",
            "ready_to_close_prob_mean",
            "trigger_prob_mean",
        ]:
            values = [float(s.get(key, 0.0)) for s in ref_stats]
            results[key] = float(sum(values) / max(len(values), 1))
        alignment_hist_keys = [
            "alignment_diffusion_block_reason_hist",
            "alignment_diffusion_phase_hist",
            "alignment_diffusion_bucket_hist",
            "alignment_diffusion_controller_type_hist",
        ]
        for key in alignment_hist_keys:
            merged_hist = {}
            for s in ref_stats:
                for hist_key, hist_value in dict(s.get(key, {}) or {}).items():
                    merged_hist[str(hist_key)] = int(merged_hist.get(str(hist_key), 0)) + int(hist_value)
            results[key] = merged_hist
        for key in [
            "alignment_diffusion_eval_count",
            "alignment_diffusion_active_count",
            "alignment_diffusion_apply_count",
            "alignment_diffusion_low_confidence_count",
            "alignment_diffusion_soft_clamp_count",
            "alignment_diffusion_hard_reject_count",
            "alignment_diffusion_safety_reject_count",
        ]:
            results[key] = int(sum(int(s.get(key, 0)) for s in ref_stats))
        for key in [
            "alignment_diffusion_confidence_mean",
            "alignment_diffusion_risk_prob_mean",
            "alignment_diffusion_stop_prob_mean",
            "alignment_diffusion_scale_down_mean",
            "alignment_diffusion_pos_norm_mean",
            "alignment_diffusion_yaw_abs_mean",
            "alignment_diffusion_pred_target_delta_norm_mean",
            "alignment_diffusion_pred_target_yaw_abs_mean",
            "alignment_diffusion_target_action_sign_agreement_mean",
        ]:
            values = [float(s.get(key, 0.0)) for s in ref_stats]
            results[key] = float(sum(values) / max(len(values), 1))
        # Fallback: some shadow runs can miss the scalar counter propagation while
        # still recording consistent histograms. Recover the counters from the
        # merged histograms so downstream diagnosis remains trustworthy.
        if int(results.get("alignment_diffusion_eval_count", 0)) <= 0:
            total_from_hist = int(sum(int(v) for v in results.get("alignment_diffusion_block_reason_hist", {}).values()))
            if total_from_hist > 0:
                results["alignment_diffusion_eval_count"] = total_from_hist
        if int(results.get("alignment_diffusion_active_count", 0)) <= 0:
            block_hist = dict(results.get("alignment_diffusion_block_reason_hist", {}) or {})
            total_from_hist = int(sum(int(v) for v in block_hist.values()))
            blocked_pre_model = int(block_hist.get("trigger", 0)) + int(block_hist.get("coarse", 0))
            recovered_active = max(total_from_hist - blocked_pre_model, 0)
            if recovered_active > 0:
                results["alignment_diffusion_active_count"] = recovered_active
        if int(results.get("alignment_diffusion_apply_count", 0)) <= 0:
            block_hist = dict(results.get("alignment_diffusion_block_reason_hist", {}) or {})
            if int(block_hist.get("applied", 0)) > 0:
                results["alignment_diffusion_apply_count"] = int(block_hist.get("applied", 0))
        controller_hist = dict(results.get("alignment_diffusion_controller_type_hist", {}) or {})
        has_vnext = (
            any("alignment_tc_student_vnext" in str(s.get("alignment_diffusion_controller_type", "")) for s in ref_stats)
            or int(controller_hist.get("alignment_tc_student_vnext", 0)) > 0
        )
        if has_vnext:
            alias_map = {
                "alignment_tc_student_vnext_eval_count": "alignment_diffusion_eval_count",
                "alignment_tc_student_vnext_active_count": "alignment_diffusion_active_count",
                "alignment_tc_student_vnext_apply_count": "alignment_diffusion_apply_count",
                "alignment_tc_student_vnext_low_confidence_count": "alignment_diffusion_low_confidence_count",
                "alignment_tc_student_vnext_soft_clamp_count": "alignment_diffusion_soft_clamp_count",
                "alignment_tc_student_vnext_hard_reject_count": "alignment_diffusion_hard_reject_count",
                "alignment_tc_student_vnext_safety_reject_count": "alignment_diffusion_safety_reject_count",
                "alignment_tc_student_vnext_block_reason_hist": "alignment_diffusion_block_reason_hist",
                "alignment_tc_student_vnext_phase_hist": "alignment_diffusion_phase_hist",
                "alignment_tc_student_vnext_bucket_hist": "alignment_diffusion_bucket_hist",
                "alignment_tc_student_vnext_controller_type_hist": "alignment_diffusion_controller_type_hist",
                "alignment_tc_student_vnext_confidence_mean": "alignment_diffusion_confidence_mean",
                "alignment_tc_student_vnext_risk_prob_mean": "alignment_diffusion_risk_prob_mean",
                "alignment_tc_student_vnext_stop_prob_mean": "alignment_diffusion_stop_prob_mean",
                "alignment_tc_student_vnext_scale_down_mean": "alignment_diffusion_scale_down_mean",
                "alignment_tc_student_vnext_pos_norm_mean": "alignment_diffusion_pos_norm_mean",
                "alignment_tc_student_vnext_yaw_abs_mean": "alignment_diffusion_yaw_abs_mean",
                "alignment_tc_student_vnext_pred_target_delta_norm_mean": "alignment_diffusion_pred_target_delta_norm_mean",
                "alignment_tc_student_vnext_pred_target_yaw_abs_mean": "alignment_diffusion_pred_target_yaw_abs_mean",
                "alignment_tc_student_vnext_target_action_sign_agreement_mean": "alignment_diffusion_target_action_sign_agreement_mean",
            }
            for out_key, src_key in alias_map.items():
                results[out_key] = results.get(src_key)
    results["phase1_verified_grasp_reached_count"] = int(
        sum(1 for s in results["stage_stats"] if s.get("phase1_verified_grasp_reached", False))
    )
    results["phase1_grasp_contact_confirmed_count"] = int(
        sum(1 for s in results["stage_stats"] if s.get("phase1_grasp_contact_confirmed", False))
    )
    results["has_object_in_hand_entered_count"] = int(
        sum(1 for s in results["stage_stats"] if s.get("has_object_in_hand_entered", False))
    )
    results["phase2_entered_after_student_count"] = int(
        sum(1 for s in results["stage_stats"] if s.get("phase2_entered_after_student", False))
    )
    results["phase1_force_spike_count"] = int(sum(int(s.get("phase1_force_spike_count", 0)) for s in results["stage_stats"]))
    results["phase1_jam_detected_count"] = int(sum(int(s.get("phase1_jam_detected_count", 0)) for s in results["stage_stats"]))
    close_source_hist = {}
    reopen_reason_hist = {}
    for s in results["stage_stats"]:
        close_source = str(s.get("phase1_close_command_source", "none"))
        close_source_hist[close_source] = int(close_source_hist.get(close_source, 0)) + 1
        reopen_reason = str(s.get("phase1_reopen_reason", "none"))
        reopen_reason_hist[reopen_reason] = int(reopen_reason_hist.get(reopen_reason, 0)) + 1
    results["phase1_close_command_source_hist"] = close_source_hist
    results["phase1_reopen_reason_hist"] = reopen_reason_hist
    insert_commit_steps_after_bridge = [int(s.get("insert_commit_steps_after_bridge", 0)) for s in results["stage_stats"]]
    results["insert_commit_steps_after_bridge_total"] = int(sum(insert_commit_steps_after_bridge))
    results["insert_commit_steps_after_bridge_mean"] = float(sum(insert_commit_steps_after_bridge) / max(len(insert_commit_steps_after_bridge), 1))
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
    if coarse2contact is not None:
        base_mode = f"coarse2contact:{coarse2contact.mode}"
    results["mode"] = base_mode

    if args.support_states_output_npz:
        write_rows_npz(support_state_rows or [], Path(args.support_states_output_npz))
    if args.fire_distill_output_npz:
        write_rows_npz(support_state_rows or [], Path(args.fire_distill_output_npz))
    if args.alignment_diffusion_raw_output_npz:
        raw_out = write_rows_npz(alignment_diffusion_rows or [], Path(args.alignment_diffusion_raw_output_npz))
        results["alignment_diffusion_raw_output_npz"] = str(raw_out) if raw_out is not None else None
        results["alignment_diffusion_raw_rows"] = int(len(alignment_diffusion_rows or []))
        if alignment_diffusion_rows:
            bucket_counts = Counter(str(row.get("stage_bucket", "far")) for row in alignment_diffusion_rows)
            stop_counts = Counter(str(row.get("stop_reason", "continue")) for row in alignment_diffusion_rows)
            risk_reason_counts = Counter(str(row.get("risk_reason", "safe")) for row in alignment_diffusion_rows)
            no_op_counts = Counter(str(row.get("no_op_reason", "continue")) for row in alignment_diffusion_rows)
            aug_counts = Counter(int(np.asarray(row.get("data_aug_id", 0)).reshape(())) for row in alignment_diffusion_rows)
            risk_rate = float(np.mean([float(np.asarray(row.get("risk_label", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]) for row in alignment_diffusion_rows]))
            stop_rate = float(np.mean([float(np.asarray(row.get("stop_label", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]) for row in alignment_diffusion_rows]))
            sample_weight_mean = float(np.mean([float(np.asarray(row.get("sample_weight", 1.0)).reshape(())) for row in alignment_diffusion_rows]))
            raw_report = {
                "output_npz": str(raw_out) if raw_out is not None else None,
                "rows": int(len(alignment_diffusion_rows)),
                "episodes": int(len(episode_indices)),
                "bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
                "stop_reason_counts": {k: int(v) for k, v in stop_counts.items()},
                "risk_reason_counts": {k: int(v) for k, v in risk_reason_counts.items()},
                "no_op_reason_counts": {k: int(v) for k, v in no_op_counts.items()},
                "data_aug_counts": {str(k): int(v) for k, v in aug_counts.items()},
                "risk_rate": risk_rate,
                "stop_rate": stop_rate,
                "sample_weight_mean": sample_weight_mean,
            }
            results["alignment_diffusion_raw_report"] = raw_report
            if args.alignment_diffusion_raw_report_json:
                report_path = Path(args.alignment_diffusion_raw_report_json)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(raw_report, indent=2), encoding="utf-8")

    results = _jsonable_value(results)
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Task:         {args.task_name}")
    print(f"Checkpoint:   {args.checkpoint_dir}")
    print(f"Mode:         {results['mode']}")
    print(f"TargetProv:   {results.get('target_provider_mode', 'legacy_auto')}")
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
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--run_full_horizon_on_success", action="store_true", default=True)
    parser.add_argument("--stop_on_success", dest="run_full_horizon_on_success", action="store_false")
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--no_depth", dest="use_depth", action="store_false")
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--no_force", dest="use_force", action="store_false")
    parser.add_argument(
        "--coarse2contact_mode",
        type=str,
        default="off",
        choices=["off", "depth_shadow", "depth_apply", "force_reflex", "depth_force"],
    )
    parser.add_argument("--coarse2contact_shadow_only", action="store_true", default=False)
    parser.add_argument("--coarse2contact_chunk_size", type=int, default=4)
    parser.add_argument("--coarse2contact_precontact_depth_threshold", type=float, default=0.20)
    parser.add_argument("--coarse2contact_contact_depth_threshold", type=float, default=0.035)
    parser.add_argument("--coarse2contact_visual_xy_threshold", type=float, default=0.0015)
    parser.add_argument("--coarse2contact_visual_yaw_threshold", type=float, default=0.0349)
    parser.add_argument("--coarse2contact_max_xy_step", type=float, default=0.0005)
    parser.add_argument("--coarse2contact_max_z_step", type=float, default=0.0005)
    parser.add_argument("--coarse2contact_max_yaw_step", type=float, default=0.0087)
    parser.add_argument("--coarse2contact_force_contact_threshold", type=float, default=0.18)
    parser.add_argument("--coarse2contact_force_delta_contact_threshold", type=float, default=0.05)
    parser.add_argument("--coarse2contact_force_jam_threshold", type=float, default=0.55)
    parser.add_argument("--coarse2contact_force_torque_threshold", type=float, default=0.12)
    parser.add_argument("--coarse2contact_force_spike_threshold", type=float, default=1.0)
    parser.add_argument("--coarse2contact_backoff_m", type=float, default=0.003)
    parser.add_argument("--coarse2contact_lateral_m", type=float, default=0.0015)
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="record_video", action="store_false")
    parser.add_argument("--write_episode_videos", action="store_true", default=True)
    parser.add_argument("--no_episode_videos", dest="write_episode_videos", action="store_false")
    parser.add_argument("--write_best_gif", action="store_true", default=False)
    parser.add_argument("--no_best_gif", dest="write_best_gif", action="store_false")
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
    parser.add_argument("--alignment_v3_shadow_ckpt", type=str, default=None)
    parser.add_argument("--alignment_v4_shadow_ckpt", type=str, default=None)
    parser.add_argument("--alignment_v4_shadow_micro_only", action="store_true", default=False)
    parser.add_argument("--alignment_diffusion_ckpt", type=str, default=None)
    parser.add_argument("--enable_alignment_diffusion_shadow", action="store_true", default=False)
    parser.add_argument("--enable_alignment_diffusion_apply", action="store_true", default=False)
    parser.add_argument("--alignment_diffusion_horizon", type=int, default=8)
    parser.add_argument("--alignment_diffusion_num_samples", type=int, default=16)
    parser.add_argument(
        "--alignment_diffusion_apply_mode",
        type=str,
        default="additive",
        choices=["additive", "blend", "rewrite_micro"],
    )
    parser.add_argument("--alignment_diffusion_max_pos_step", type=float, default=0.0015)
    parser.add_argument("--alignment_diffusion_max_yaw_step", type=float, default=0.0060)
    parser.add_argument("--alignment_diffusion_risk_threshold", type=float, default=0.65)
    parser.add_argument(
        "--alignment_diffusion_trigger_mode",
        type=str,
        default="near_contact_stall",
        choices=["near_contact_stall", "align_phase", "depth_only"],
    )
    parser.add_argument("--alignment_diffusion_execute_steps", type=int, default=1)
    parser.add_argument("--alignment_tc_diffusion_ckpt", type=str, default=None)
    parser.add_argument("--enable_alignment_tc_diffusion_shadow", action="store_true", default=False)
    parser.add_argument("--enable_alignment_tc_diffusion_apply", action="store_true", default=False)
    parser.add_argument("--alignment_tc_diffusion_num_samples", type=int, default=8)
    parser.add_argument("--alignment_tc_diffusion_top_k", type=int, default=3)
    parser.add_argument("--alignment_tc_diffusion_confidence_threshold", type=float, default=0.55)
    parser.add_argument("--alignment_tc_diffusion_risk_threshold", type=float, default=0.65)
    parser.add_argument("--alignment_tc_diffusion_soft_clamp", action="store_true", default=False)
    parser.add_argument("--alignment_tc_diffusion_workspace_tolerance", type=float, default=0.0)
    parser.add_argument("--alignment_tc_diffusion_workspace_soft_clamp", action="store_true", default=False)
    parser.add_argument("--alignment_tc_diffusion_execute_steps", type=int, default=1)
    parser.add_argument("--alignment_tc_student_vnext_ckpt", type=str, default=None)
    parser.add_argument("--enable_alignment_tc_student_vnext_shadow", action="store_true", default=False)
    parser.add_argument("--enable_alignment_tc_student_vnext_apply", action="store_true", default=False)
    parser.add_argument("--alignment_tc_student_vnext_corridor_json", type=str, default=None)
    parser.add_argument("--alignment_tc_student_vnext_collector_like", action="store_true", default=False)
    parser.add_argument("--enable_alignment_tc_student_vnext_ready_gate", action="store_true", default=False)
    parser.add_argument("--alignment_tc_student_vnext_close_ready_threshold", type=float, default=0.5)
    parser.add_argument("--alignment_tc_student_vnext_handoff_ready_threshold", type=float, default=0.5)
    parser.add_argument("--phase1_force_reflex_enable", action="store_true", default=False)
    parser.add_argument("--phase1_force_contact_fz_threshold", type=float, default=0.35)
    parser.add_argument("--phase1_force_force_norm_threshold", type=float, default=0.50)
    parser.add_argument("--phase1_force_high_fz_threshold", type=float, default=2.5)
    parser.add_argument("--phase1_force_lateral_threshold", type=float, default=1.5)
    parser.add_argument("--phase1_force_torque_threshold", type=float, default=1.0)
    parser.add_argument("--phase1_force_spike_threshold", type=float, default=0.75)
    parser.add_argument("--phase1_force_backoff_mm", type=float, default=0.0015)
    parser.add_argument("--phase1_close_confirm_steps", type=int, default=2)
    parser.add_argument("--phase1_close_fail_steps", type=int, default=6)
    parser.add_argument("--phase1_post_contact_hold_steps", type=int, default=8)
    parser.add_argument("--phase1_reopen_cooldown_steps", type=int, default=4)
    parser.add_argument("--enable_alignment_v4_apply", action="store_true", default=False)
    parser.add_argument("--alignment_v4_apply_scale", type=float, default=1.0)
    parser.add_argument("--alignment_v4_apply_max_pos", type=float, default=0.0010)
    parser.add_argument("--alignment_v4_apply_max_yaw", type=float, default=0.0040)
    parser.add_argument("--alignment_v4_apply_only_when_gate_pass", action="store_true", default=True)
    parser.add_argument(
        "--alignment_v4_apply_without_gate_pass",
        dest="alignment_v4_apply_only_when_gate_pass",
        action="store_false",
    )
    parser.add_argument("--alignment_v4_apply_require_improve", action="store_true", default=False)
    parser.add_argument("--alignment_v4_apply_micro_only", action="store_true", default=False)
    parser.add_argument("--enable_alignment_v3_apply", action="store_true", default=False)
    parser.add_argument("--alignment_v3_apply_scale", type=float, default=1.0)
    parser.add_argument("--alignment_v3_apply_max_pos", type=float, default=0.0020)
    parser.add_argument("--alignment_v3_apply_max_yaw", type=float, default=0.0020)
    parser.add_argument("--alignment_v3_apply_only_when_gate_pass", action="store_true", default=True)
    parser.add_argument(
        "--alignment_v3_apply_without_gate_pass",
        dest="alignment_v3_apply_only_when_gate_pass",
        action="store_false",
    )
    parser.add_argument("--alignment_v3_apply_require_improve", action="store_true", default=False)
    parser.add_argument(
        "--disable_learned_target_close_stage_orientation_contract",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--enable_learned_target_close_stage_orientation_contract_for_v3",
        action="store_true",
        default=False,
        help=(
            "Keep the canonical close-stage orientation contract even when "
            "alignment_v3 shadow is enabled. By default, v3 shadow uses raw "
            "learned target deltas to stay closer to the direct-control "
            "teacher contract."
        ),
    )
    parser.add_argument("--contact_ckpt", type=str, default=None)
    parser.add_argument(
        "--target_provider_mode",
        type=str,
        default="legacy_auto",
        choices=["legacy_auto", "teacher_oracle", "learned", "canonical_fallback"],
    )
    parser.add_argument("--target_provider_ckpt", type=str, default=None)
    parser.add_argument("--handoff_provider_ckpt", type=str, default=None)
    parser.add_argument("--student_handoff_shadow_only", action="store_true", default=False)
    parser.add_argument("--student_group_selector_shadow_ckpt", type=str, default=None)
    parser.add_argument("--student_group_selector_handoff_ckpt", type=str, default=None)
    parser.add_argument("--student_b1_apply_gate_shadow_ckpt", type=str, default=None)
    parser.add_argument("--enable_b1_group_selector_bounded", action="store_true", default=False)
    parser.add_argument(
        "--b1_group_shadow_gate_mode",
        type=str,
        default="broad",
        choices=["broad", "close_only"],
        help="B1 shadow diagnostic gate. close_only only records B1 predictions in close-neighborhood frames.",
    )
    parser.add_argument("--student_candidate_evaluator_shadow_ckpt", type=str, default=None)
    parser.add_argument("--student_candidate_evaluator_handoff_ckpt", type=str, default=None)
    parser.add_argument("--enable_b2_candidate_bounded_v0", action="store_true", default=False)
    parser.add_argument("--b2_candidate_apply_conf_threshold", type=float, default=0.431)
    parser.add_argument("--b2_candidate_apply_margin_threshold", type=float, default=0.010)
    parser.add_argument(
        "--close_veto_runtime_geometry_fallback_for_bounded",
        action="store_true",
        default=False,
        help=(
            "Bounded-only close-veto fallback: when handoff spec is active and handoff ready is shadow-blocked, "
            "allow conservative runtime geometry thresholds to decide close-ready on close-intent steps."
        ),
    )
    parser.add_argument("--enable_bounded_auto_close_on_alignment", action="store_true", default=False)
    parser.add_argument("--bounded_auto_close_stable_frames", type=int, default=1)
    parser.add_argument("--bounded_auto_close_xy_threshold", type=float, default=-1.0)
    parser.add_argument("--bounded_auto_close_abs_z_threshold", type=float, default=-1.0)
    parser.add_argument("--bounded_auto_close_yaw_threshold", type=float, default=-1.0)
    parser.add_argument("--enable_force_close_after_b2_eval", action="store_true", default=False)
    parser.add_argument("--enable_alignment_near_zone_gate", action="store_true", default=False)
    parser.add_argument("--alignment_near_zone_xy_threshold", type=float, default=0.05)
    parser.add_argument("--alignment_near_zone_z_threshold", type=float, default=0.10)
    parser.add_argument("--v2_alignment_shadow_ckpt", type=str, default=None)
    parser.add_argument("--v2_target_delta_provider_ckpt", type=str, default=None)
    parser.add_argument("--enable_v2_nearzone_assist", action="store_true", default=False)
    parser.add_argument("--v2_assist_scale_cap", type=float, default=0.25)
    parser.add_argument("--v2_assist_max_pos", type=float, default=0.0025)
    parser.add_argument("--v2_assist_max_rot", type=float, default=0.015)
    parser.add_argument("--v2_assist_only_when_gate_pass", action="store_true", default=True)
    parser.add_argument("--enable_v2_predictor_micro_assist_apply", action="store_true", default=False)
    parser.add_argument("--v2_predictor_micro_assist_scale_cap", type=float, default=0.05)
    parser.add_argument("--v2_predictor_micro_assist_max_pos", type=float, default=0.00075)
    parser.add_argument("--v2_predictor_micro_assist_max_rot", type=float, default=0.0040)
    parser.add_argument("--v2_predictor_micro_assist_only_when_gate_pass", action="store_true", default=True)
    parser.add_argument("--enable_target_delta_servo_shadow", action="store_true", default=False)
    parser.add_argument("--enable_target_delta_servo_apply", action="store_true", default=False)
    parser.add_argument("--target_delta_servo_bypass_gates", action="store_true", default=False)
    parser.add_argument("--target_delta_servo_apply_once_per_episode", action="store_true", default=False)
    parser.add_argument(
        "--target_delta_servo_source",
        type=str,
        default="predictor",
        choices=["predictor", "privileged_replay", "basin_diagnostic", "zero_diagnostic"],
    )
    parser.add_argument("--target_delta_servo_k_xy", type=float, default=0.08)
    parser.add_argument("--target_delta_servo_k_z", type=float, default=0.06)
    parser.add_argument("--target_delta_servo_k_yaw", type=float, default=0.04)
    parser.add_argument("--target_delta_servo_max_pos", type=float, default=0.0010)
    parser.add_argument("--target_delta_servo_max_yaw", type=float, default=0.0040)
    parser.add_argument("--target_delta_servo_apply_xy_threshold", type=float, default=0.03)
    parser.add_argument("--target_delta_servo_apply_abs_z_threshold", type=float, default=0.07)
    parser.add_argument("--target_delta_servo_apply_yaw_threshold", type=float, default=0.25)
    parser.add_argument(
        "--student_candidate_evaluator_mode_input_path",
        type=str,
        default="summary_only",
        choices=["summary_only", "hybrid"],
        help="Mode-head input path for B2 candidate evaluator shadow checkpoints.",
    )
    parser.add_argument(
        "--b2_candidate_shadow_gate_mode",
        type=str,
        default="broad",
        choices=["broad", "close_only", "nearish_only"],
        help="B2 shadow diagnostic gate. nearish_only restricts B2 shadow to z-aware late-pregrasp frames.",
    )
    parser.add_argument(
        "--b2_candidate_shadow_yaw_probe_values",
        type=str,
        default="",
        help="Comma-separated pure-yaw probes appended only inside B2 shadow diagnostics; never changes controller candidates.",
    )
    parser.add_argument(
        "--runtime_candidate_yaw_probe_values",
        type=str,
        default="",
        help="Comma-separated pure-yaw actions appended into the real runtime scorer candidate bank, e.g. '0.06,0.12'.",
    )
    parser.add_argument("--near_ready_alignment_ckpt", type=str, default=None)
    parser.add_argument("--residual_score_adapter_ckpt", type=str, default=None)
    parser.add_argument(
        "--record_teacher_truth_metrics",
        action="store_true",
        default=False,
        help=(
            "Record privileged teacher-oracle xy/z/yaw and handoff metrics to gripper traces for diagnosis only. "
            "These metrics are not fed back into runtime control."
        ),
    )
    parser.add_argument(
        "--enforce_no_privileged_runtime",
        action="store_true",
        default=False,
        help=(
            "Fail fast if the alignment runtime receives simulator/object-pose privileged "
            "targets or privileged handoff readiness. Use this for deployment/student mainline evals."
        ),
    )
    parser.add_argument(
        "--allow_privileged_runtime",
        dest="enforce_no_privileged_runtime",
        action="store_false",
        help="Allow teacher-oracle runtime targets; these runs must be reported as oracle upper bounds.",
    )
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
    parser.add_argument(
        "--require_close_intent_for_refine_band",
        action="store_true",
        default=False,
        help="Require close intent before the late refine/takeover alignment bands may open.",
    )
    parser.add_argument("--enable_alignment_pose", action="store_true", default=True)
    parser.add_argument("--disable_alignment_pose", dest="enable_alignment_pose", action="store_false")
    parser.add_argument("--use_pose_alpha", action="store_true", default=True)
    parser.add_argument("--disable_pose_alpha", dest="use_pose_alpha", action="store_false")
    parser.add_argument("--learned_residual_scale", type=float, default=0.25)
    parser.add_argument("--max_residual_pos", type=float, default=0.0025)
    parser.add_argument("--max_residual_rot", type=float, default=0.015)
    parser.add_argument("--max_alignment_corrections_per_window", type=int, default=5)
    parser.add_argument("--enable_outer_rescue", action="store_true", default=True)
    parser.add_argument("--disable_outer_rescue", dest="enable_outer_rescue", action="store_false")
    parser.add_argument("--outer_rescue_xy_scale", type=float, default=2.0)
    parser.add_argument("--outer_rescue_abs_z_scale", type=float, default=2.0)
    parser.add_argument("--outer_rescue_yaw_scale", type=float, default=2.0)
    parser.add_argument("--outer_rescue_min_xy", type=float, default=0.05)
    parser.add_argument("--outer_rescue_min_abs_z", type=float, default=0.18)
    parser.add_argument("--outer_rescue_min_yaw", type=float, default=0.30)
    parser.add_argument("--enable_alignment_close_veto", action="store_true", default=True)
    parser.add_argument("--disable_alignment_close_veto", dest="enable_alignment_close_veto", action="store_false")
    parser.add_argument("--close_veto_xy_threshold", type=float, default=0.010)
    parser.add_argument("--close_veto_abs_z_threshold", type=float, default=0.020)
    parser.add_argument("--close_veto_yaw_threshold", type=float, default=-1.0)
    parser.add_argument("--close_veto_ready_streak_frames", type=int, default=1)
    parser.add_argument("--close_veto_settle_steps", type=int, default=0)
    parser.add_argument("--close_latch_enabled", action="store_true", default=False)
    parser.add_argument("--disable_close_latch", dest="close_latch_enabled", action="store_false")
    parser.add_argument("--close_latch_steps", type=int, default=24)
    parser.add_argument("--alignment_takeover_until_close_ready", action="store_true", default=True)
    parser.add_argument("--disable_alignment_takeover_until_close_ready", dest="alignment_takeover_until_close_ready", action="store_false")
    parser.add_argument("--alignment_assist_xy_scale", type=float, default=10.0)
    parser.add_argument("--alignment_assist_abs_z_scale", type=float, default=40.0)
    parser.add_argument("--alignment_assist_yaw_scale", type=float, default=4.0)
    parser.add_argument("--alignment_assist_base_scale", type=float, default=0.55)
    parser.add_argument("--alignment_assist_close_block_base_scale", type=float, default=0.25)
    parser.add_argument("--alignment_takeover_xy_scale", type=float, default=2.0)
    parser.add_argument("--alignment_takeover_abs_z_scale", type=float, default=6.0)
    parser.add_argument("--alignment_takeover_yaw_scale", type=float, default=1.5)
    parser.add_argument("--alignment_takeover_motion_xy_threshold", type=float, default=0.020)
    parser.add_argument("--alignment_takeover_motion_abs_z_threshold", type=float, default=0.030)
    parser.add_argument("--alignment_takeover_close_block_xy_threshold", type=float, default=0.025)
    parser.add_argument("--alignment_takeover_close_block_abs_z_threshold", type=float, default=0.040)
    parser.add_argument("--alignment_zone_hysteresis", type=float, default=1.15)
    parser.add_argument("--alignment_candidate_hold_steps", type=int, default=4)
    parser.add_argument("--alignment_candidate_switch_margin", type=float, default=0.35)
    parser.add_argument("--alignment_action_lowpass_alpha", type=float, default=0.65)
    parser.add_argument("--skip_alignment_when_close_ready", action="store_true", default=False)
    parser.add_argument("--no_skip_alignment_when_close_ready", dest="skip_alignment_when_close_ready", action="store_false")
    parser.add_argument("--skip_alignment_ready_xy_threshold", type=float, default=-1.0)
    parser.add_argument("--skip_alignment_ready_abs_z_threshold", type=float, default=-1.0)
    parser.add_argument("--skip_alignment_ready_yaw_threshold", type=float, default=-1.0)
    parser.add_argument("--enable_alignment_physical_mask", action="store_true", default=False)
    parser.add_argument("--disable_alignment_physical_mask", dest="enable_alignment_physical_mask", action="store_false")
    parser.add_argument("--enable_alignment_low_conf_noop", action="store_true", default=False)
    parser.add_argument("--disable_alignment_low_conf_noop", dest="enable_alignment_low_conf_noop", action="store_false")
    parser.add_argument("--use_legacy_teacher_candidate_bank_for_scorer", action="store_true", default=False)
    parser.add_argument("--no_legacy_teacher_candidate_bank_for_scorer", dest="use_legacy_teacher_candidate_bank_for_scorer", action="store_false")
    parser.add_argument(
        "--teacher_candidate_yaw_probe_values",
        type=str,
        default="",
        help="Optional extra pure-yaw probe magnitudes for the runtime/teacher candidate bank, e.g. '0.06,0.12'.",
    )
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
    parser.add_argument("--episode_indices", type=str, default=None)
    parser.add_argument(
        "--collector_like_demo_reset",
        action="store_true",
        default=False,
        help="Reset each episode from the matching demo episode number instead of task.reset().",
    )
    parser.add_argument("--demo_max_attempts", type=int, default=10)
    parser.add_argument("--support_states_output_npz", type=str, default=None)
    parser.add_argument("--alignment_diffusion_raw_output_npz", type=str, default=None)
    parser.add_argument("--alignment_diffusion_raw_report_json", type=str, default=None)
    parser.add_argument("--alignment_diffusion_raw_horizon", type=int, default=8)
    parser.add_argument("--alignment_diffusion_raw_near_depth_threshold", type=float, default=0.085)
    parser.add_argument("--alignment_diffusion_raw_micro_depth_threshold", type=float, default=0.045)
    parser.add_argument("--alignment_diffusion_raw_contact_force_threshold", type=float, default=0.18)
    parser.add_argument("--alignment_diffusion_raw_high_force_threshold", type=float, default=0.45)
    parser.add_argument("--alignment_diffusion_raw_force_spike_threshold", type=float, default=0.10)
    parser.add_argument("--alignment_diffusion_raw_action_floor_xy", type=float, default=0.0008)
    parser.add_argument("--alignment_diffusion_raw_action_floor_z", type=float, default=0.0008)
    parser.add_argument("--alignment_diffusion_raw_action_floor_yaw", type=float, default=0.003)
    parser.add_argument("--alignment_diffusion_raw_xy_gain_margin", type=float, default=0.00015)
    parser.add_argument("--alignment_diffusion_raw_z_gain_margin", type=float, default=0.00015)
    parser.add_argument("--alignment_diffusion_raw_yaw_gain_margin", type=float, default=0.0008)
    parser.add_argument("--alignment_diffusion_raw_augment_copies", type=int, default=1)
    parser.add_argument("--alignment_diffusion_raw_rgb_noise_std", type=float, default=0.015)
    parser.add_argument("--alignment_diffusion_raw_depth_noise_std", type=float, default=0.010)
    parser.add_argument("--alignment_diffusion_raw_force_noise_std", type=float, default=0.015)
    parser.add_argument("--alignment_diffusion_raw_proprio_noise_std", type=float, default=0.002)
    parser.add_argument("--alignment_diffusion_raw_action_xy_noise_std", type=float, default=0.00015)
    parser.add_argument("--alignment_diffusion_raw_action_z_noise_std", type=float, default=0.00010)
    parser.add_argument("--alignment_diffusion_raw_action_yaw_noise_std", type=float, default=0.00060)
    parser.add_argument("--alignment_diffusion_raw_sample_weight_near", type=float, default=1.0)
    parser.add_argument("--alignment_diffusion_raw_sample_weight_micro", type=float, default=1.5)
    parser.add_argument("--alignment_diffusion_raw_sample_weight_stop", type=float, default=0.85)
    parser.add_argument("--alignment_diffusion_raw_sample_weight_risk", type=float, default=0.60)
    parser.add_argument(
        "--alignment_tc_raw_privileged_target_mode",
        type=str,
        default="active",
        choices=["active", "pregrasp", "commit", "insert_commit", "insert", "task_success_centre"],
        help="Label-only privileged target for target-conditioned alignment data. Use commit for contact/insert training.",
    )
    parser.add_argument(
        "--alignment_tc_raw_use_privileged_delta_gate",
        action="store_true",
        default=False,
        help="Filter raw near/micro rows by privileged target-relative delta instead of depth/phase buckets.",
    )
    parser.add_argument("--alignment_tc_raw_near_xy_threshold", type=float, default=0.030)
    parser.add_argument("--alignment_tc_raw_near_abs_z_threshold", type=float, default=0.070)
    parser.add_argument("--alignment_tc_raw_near_yaw_threshold", type=float, default=0.35)
    parser.add_argument("--alignment_tc_raw_micro_xy_threshold", type=float, default=0.010)
    parser.add_argument("--alignment_tc_raw_micro_abs_z_threshold", type=float, default=0.030)
    parser.add_argument("--alignment_tc_raw_micro_yaw_threshold", type=float, default=0.18)
    parser.add_argument(
        "--depth_force_clean_support",
        action="store_true",
        default=False,
        help="Write planner-only RGB-D/force/action support rows for the depth-force local contact policy.",
    )
    parser.add_argument(
        "--depth_force_clean_support_with_refiner",
        action="store_true",
        default=False,
        help="Allow clean-support row capture even when a refiner is active.",
    )
    parser.add_argument(
        "--depth_force_clean_privileged_labels",
        action="store_true",
        default=False,
        help="Add label-only privileged target pose fields to depth-force clean support rows without changing planner actions.",
    )
    parser.add_argument("--depth_force_clean_near_depth_threshold", type=float, default=0.08)
    parser.add_argument("--depth_force_clean_contact_force_threshold", type=float, default=0.5)
    parser.add_argument("--depth_force_clean_jam_force_threshold", type=float, default=3.0)
    parser.add_argument("--fire_distill_output_npz", type=str, default=None)
    parser.add_argument("--fire_positive_window", type=int, default=1)
    parser.add_argument("--fire_hold_min_steps", type=int, default=50)
    parser.add_argument("--fire_negative_gap", type=int, default=8)
    parser.add_argument("--oracle_executed_align_collect", action="store_true", default=False)
    parser.add_argument("--oracle_executed_pregrasp_collect", action="store_true", default=False)
    parser.add_argument(
        "--alignment_tc_privileged_teacher_collect",
        action="store_true",
        default=False,
        help="Collect vNext target-conditioned data by letting a privileged commit-target teacher control broad-near alignment.",
    )
    parser.add_argument(
        "--alignment_tc_teacher_target_mode",
        type=str,
        default="commit",
        choices=["commit", "pregrasp", "insert_commit", "insert", "task_success_centre"],
        help="Which privileged teacher target to use for target-conditioned collection.",
    )
    parser.add_argument(
        "--alignment_tc_privileged_teacher_close_enabled",
        action="store_true",
        default=False,
        help="Allow the target-conditioned privileged teacher to close and verify grasp once the commit basin is reached.",
    )
    parser.add_argument(
        "--alignment_tc_teacher_close_verify",
        action="store_true",
        default=False,
        help="Run settle/lift verification after close-enabled target-conditioned teacher closes.",
    )
    parser.add_argument(
        "--alignment_tc_teacher_force_open_until_close_ready",
        action="store_true",
        default=True,
        help="Keep the gripper open during target-conditioned teacher motion until the close basin is reached.",
    )
    parser.add_argument(
        "--alignment_tc_teacher_no_force_open_until_close_ready",
        dest="alignment_tc_teacher_force_open_until_close_ready",
        action="store_false",
    )
    parser.add_argument("--alignment_tc_teacher_broad_xy_threshold", type=float, default=0.060)
    parser.add_argument("--alignment_tc_teacher_broad_abs_z_threshold", type=float, default=0.120)
    parser.add_argument("--alignment_tc_teacher_broad_yaw_threshold", type=float, default=0.60)
    parser.add_argument("--alignment_tc_teacher_k_xy", type=float, default=0.20)
    parser.add_argument("--alignment_tc_teacher_k_z", type=float, default=0.14)
    parser.add_argument("--alignment_tc_teacher_k_yaw", type=float, default=0.12)
    parser.add_argument("--alignment_tc_teacher_max_pos_step", type=float, default=0.0015)
    parser.add_argument("--alignment_tc_teacher_max_yaw_step", type=float, default=0.006)
    parser.add_argument("--alignment_tc_teacher_broad_max_pos_step", type=float, default=0.0045)
    parser.add_argument("--alignment_tc_teacher_broad_max_yaw_step", type=float, default=0.018)
    parser.add_argument("--alignment_tc_teacher_near_max_pos_step", type=float, default=0.0030)
    parser.add_argument("--alignment_tc_teacher_near_max_yaw_step", type=float, default=0.012)
    parser.add_argument("--alignment_tc_teacher_micro_max_pos_step", type=float, default=0.0015)
    parser.add_argument("--alignment_tc_teacher_micro_max_yaw_step", type=float, default=0.008)
    parser.add_argument("--alignment_tc_teacher_short_horizon", type=int, default=3)
    parser.add_argument("--alignment_tc_teacher_workspace_delta_tolerance", type=float, default=0.00001)
    parser.add_argument("--alignment_tc_teacher_already_close_xy", type=float, default=0.0025)
    parser.add_argument("--alignment_tc_teacher_already_close_abs_z", type=float, default=0.0030)
    parser.add_argument("--alignment_tc_teacher_already_close_yaw", type=float, default=0.015)
    parser.add_argument("--alignment_tc_teacher_broad_min_pos_step", type=float, default=0.0015)
    parser.add_argument("--alignment_tc_teacher_near_min_pos_step", type=float, default=0.0012)
    parser.add_argument("--alignment_tc_teacher_micro_min_pos_step", type=float, default=0.0007)
    parser.add_argument("--alignment_tc_teacher_yaw_control_sign", type=float, default=-1.0)
    parser.add_argument("--alignment_tc_teacher_yaw_imitation_enabled", action="store_true", default=True)
    parser.add_argument(
        "--alignment_tc_teacher_no_yaw_imitation_enabled",
        dest="alignment_tc_teacher_yaw_imitation_enabled",
        action="store_false",
    )
    parser.add_argument("--teacher_grasp_ready_threshold", type=float, default=0.55)
    parser.add_argument("--teacher_grasp_xy_threshold", type=float, default=0.006)
    parser.add_argument("--teacher_grasp_abs_z_threshold", type=float, default=0.010)
    parser.add_argument("--teacher_grasp_yaw_threshold", type=float, default=0.05)
    parser.add_argument("--grasp_basin_profile_json", type=str, default=None)
    parser.add_argument("--alignment_tc_teacher_invert_yaw_control", action="store_true", default=False)
    parser.add_argument(
        "--alignment_tc_teacher_no_invert_yaw_control",
        dest="alignment_tc_teacher_invert_yaw_control",
        action="store_false",
    )
    parser.add_argument("--teacher_close_xy_threshold", type=float, default=0.006)
    parser.add_argument("--teacher_close_abs_z_threshold", type=float, default=0.005)
    parser.add_argument("--teacher_close_yaw_threshold", type=float, default=0.12)
    parser.add_argument("--teacher_close_orientation_threshold_deg", type=float, default=12.0)
    parser.add_argument("--teacher_close_basin_distance_threshold", type=float, default=-1.0)
    parser.add_argument("--teacher_close_contact_depth_threshold", type=float, default=0.022)
    parser.add_argument("--teacher_commit_switch_xy_threshold", type=float, default=0.012)
    parser.add_argument("--teacher_commit_switch_z_threshold", type=float, default=0.012)
    parser.add_argument("--teacher_commit_switch_yaw_threshold", type=float, default=0.20)
    parser.add_argument("--teacher_orientation_rescue_xy_threshold", type=float, default=0.015)
    parser.add_argument("--teacher_orientation_rescue_angle_threshold_deg", type=float, default=8.0)
    parser.add_argument("--teacher_rescue_pitch_small", type=float, default=0.04)
    parser.add_argument("--teacher_rescue_roll_small", type=float, default=0.04)
    parser.add_argument("--teacher_verify_hold_steps", type=int, default=8)
    parser.add_argument("--teacher_verify_settle_steps", type=int, default=4)
    parser.add_argument("--teacher_verify_lift_threshold", type=float, default=0.012)
    parser.add_argument("--teacher_verify_follow_distance", type=float, default=0.06)
    parser.add_argument("--teacher_verify_min_consecutive_lift_steps", type=int, default=2)
    parser.add_argument("--teacher_retry_lift", type=float, default=0.008)
    parser.add_argument("--teacher_retry_steps", type=int, default=4)
    parser.add_argument("--teacher_max_retries", type=int, default=2)
    parser.add_argument("--teacher_planner_close_handoff", action="store_true", default=True)
    parser.add_argument("--teacher_no_planner_close_handoff", dest="teacher_planner_close_handoff", action="store_false")
    parser.add_argument("--teacher_motion_entry_xy_threshold", type=float, default=0.040)
    parser.add_argument("--teacher_motion_entry_abs_z_threshold", type=float, default=0.120)
    parser.add_argument("--teacher_require_alignment_ready_for_motion_gate", action="store_true", default=False)
    parser.add_argument("--teacher_no_require_alignment_ready_for_motion_gate", dest="teacher_require_alignment_ready_for_motion_gate", action="store_false")
    parser.add_argument("--teacher_handoff_revoke_xy_threshold", type=float, default=0.012)
    parser.add_argument("--teacher_handoff_revoke_abs_z_threshold", type=float, default=0.025)
    parser.add_argument("--teacher_candidate_hold_steps", type=int, default=4)
    parser.add_argument("--teacher_candidate_switch_margin", type=float, default=0.75)
    parser.add_argument("--teacher_action_lowpass_alpha", type=float, default=0.65)
    parser.add_argument("--teacher_use_continuous_smooth_control", action="store_true", default=True)
    parser.add_argument("--teacher_no_continuous_smooth_control", dest="teacher_use_continuous_smooth_control", action="store_false")
    parser.add_argument("--teacher_smooth_kp_xy", type=float, default=0.45)
    parser.add_argument("--teacher_smooth_kp_z", type=float, default=0.55)
    parser.add_argument("--teacher_smooth_kp_yaw", type=float, default=0.35)
    parser.add_argument("--teacher_invert_yaw_control", action="store_true", default=True)
    parser.add_argument("--teacher_no_invert_yaw_control", dest="teacher_invert_yaw_control", action="store_false")
    parser.add_argument("--teacher_smooth_xy_deadband", type=float, default=0.0015)
    parser.add_argument("--teacher_smooth_z_deadband", type=float, default=0.0015)
    parser.add_argument("--teacher_smooth_yaw_deadband", type=float, default=0.010)
    parser.add_argument("--teacher_max_yaw_step", type=float, default=0.020)
    parser.add_argument("--teacher_near_smooth_xy_threshold", type=float, default=0.020)
    parser.add_argument("--teacher_near_smooth_abs_z_threshold", type=float, default=0.060)
    parser.add_argument("--teacher_far_action_scale", type=float, default=1.5)
    parser.add_argument("--teacher_near_max_step", type=float, default=0.004)
    parser.add_argument("--teacher_far_max_step", type=float, default=0.006)
    parser.add_argument("--teacher_planner_verify_steps", type=int, default=10)
    parser.add_argument("--teacher_planner_verify_lift_steps", type=int, default=5)
    parser.add_argument("--teacher_planner_close_settle", action="store_true", default=True)
    parser.add_argument("--teacher_no_planner_close_settle", dest="teacher_planner_close_settle", action="store_false")
    parser.add_argument("--teacher_planner_close_gripper_threshold", type=float, default=0.5)
    parser.add_argument("--teacher_planner_close_latch_steps", type=int, default=48)
    parser.add_argument("--teacher_retry_realign_cooldown_steps", type=int, default=8)
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
