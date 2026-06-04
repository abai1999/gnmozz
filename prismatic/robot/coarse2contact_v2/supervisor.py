"""Stage-owned Coarse2Contact v2 runtime supervisor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from prismatic.robot.residual_safety import ResidualSafety
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local

from .basin_recovery import BasinRecoveryConfig, BasinRecoverySupervisor
from .basin_state import BasinStateCalibration, CalibratedGraspBasinEstimator, EstimatedBasinError, load_basin_state_calibration_report
from .controllers import GuardedSlideController, GraspContactController, RecoveryFSM, RecoveryPhase, SkillControlResult
from .learned_force import LearnedForceClassifierAdapter
from .localizers import LocalGeometryError, RingGraspLocalizer, RingSpokeAlignLocalizer
from .specs import PrecisionSkillSpec, PrecisionTaskSpec, StageTransition
from .alignment_takeover import AlignmentTakeoverConfig, AlignmentReadiness, TaskFrameResidualEstimate, evaluate_alignment_readiness
from .takeover_contract import FrameResidual, ObservabilityDecision, TakeoverThresholds, decide_takeover_tier


def _obs_get(observation: Any, key: str, default: Any = None) -> Any:
    if observation is None:
        return default
    if isinstance(observation, Mapping):
        return observation.get(key, default)
    return getattr(observation, key, default)


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


def _local_geometry_semantics(error: LocalGeometryError, *, skill_type: str) -> dict[str, Any]:
    if skill_type == "precision_grasp":
        return {
            "schema_version": "grasp_frame_residual_estimate_v1",
            "estimate_status": "calibrated_proxy",
            "visual_evidence": {
                "dx_source": "mask_centroid_image_offset",
                "dy_source": "mask_centroid_image_offset",
                "dz_source": "mask_median_depth_proxy",
                "dyaw_source": "image_axis_diagnostic_only",
                "image_axis_yaw": float(error.image_axis_yaw),
                "yaw_reason": str(error.yaw_reason),
            },
            "contract_aligned_estimated_residual": False,
            "invalid_reason": "" if bool(error.valid) else str(error.reason),
            "z_semantics_status": "proxy_not_descend_progress",
            "yaw_control_status": "abstain" if not bool(error.yaw_valid) else "diagnostic_only",
        }
    return {
        "schema_version": "frame_residual_estimate_v1",
        "estimate_status": "localizer_output",
        "contract_aligned_estimated_residual": bool(error.valid),
        "invalid_reason": "" if bool(error.valid) else str(error.reason),
    }


def _runtime_contract_decision_from_estimate(
    est: EstimatedBasinError | None,
    *,
    precision_row: bool,
    requires_yaw_observability: bool,
    pullback_xy_threshold: float,
) -> dict[str, Any]:
    if est is None:
        return {
            "takeover_tier": "invalid",
            "pullback_allowed": False,
            "micro_entry_ready": False,
            "close_ready_ready": False,
            "runtime_proxy_contract": True,
            "contract_source": "estimated_basin_error_missing",
        }
    residual = FrameResidual(
        dx=float(est.dx),
        dy=float(est.dy),
        dz=float(est.dz),
        dyaw=float(est.dyaw if est.yaw_valid else 0.0),
        reference_frame=str(est.reference_entity),
        target_frame=str(est.target_entity),
        z_semantics="runtime_estimated_basin_error",
        source=str(est.source),
    )
    visual_class = "prior_only" if str(est.reason) in {"prior_only", "prior_only_reacquire", "reacquire_needed"} else "runtime_estimated"
    observability = ObservabilityDecision(
        visual_observability_class=visual_class,
        yaw_observability_class="observable" if bool(est.yaw_valid) else "unobservable",
        yaw_observable=bool(est.yaw_valid),
        reacquire_needed=bool(visual_class == "prior_only" or not est.valid),
        reason="runtime_yaw_valid" if bool(est.yaw_valid) else str(est.reason or "yaw_abstain"),
    )
    decision = decide_takeover_tier(
        residual,
        observability,
        precision_row=bool(precision_row),
        requires_yaw_observability=bool(requires_yaw_observability),
        xy_contracted=bool(est.pullback_ready(xy_threshold=pullback_xy_threshold, min_frame_consistency=0.0)),
        thresholds=TakeoverThresholds(coarse_xy=float(pullback_xy_threshold), outer_xy=max(float(pullback_xy_threshold), 0.120), frontier_xy=max(float(pullback_xy_threshold), 0.180)),
    ).to_dict()
    decision.update(
        {
            "runtime_proxy_contract": True,
            "contract_source": "estimated_basin_error",
            "residual_source": str(est.source),
            "yaw_control_source": "estimated_basin_error_yaw" if bool(est.yaw_valid) else "abstain",
        }
    )
    return decision


def _skill_requires_axis(skill: PrecisionSkillSpec | None, axis: str) -> bool:
    if skill is None:
        return False
    dofs = {str(dof) for dof in skill.controlled_dofs}
    if axis == "z":
        return bool("z" in dofs or "dz" in dofs or "z_micro" in dofs)
    if axis == "yaw":
        return bool("yaw" in dofs or "yaw_micro" in dofs or bool(skill.requires_yaw_observability))
    return bool(axis in dofs or f"{axis}_micro" in dofs)


def _alignment_readiness_from_estimate(
    est: EstimatedBasinError | None,
    skill: PrecisionSkillSpec | None,
    *,
    min_frame_consistency: float = 0.20,
) -> AlignmentReadiness:
    """Strict task-frame handoff predicate derived from runtime estimates.

    This is the single supervisor-side predicate for planner gripper handoff.
    The older EstimatedBasinError.close_ready() remains available as a
    diagnostic bucket, but it must not silently drop yaw/z requirements when
    the task-frame contract asks for those axes.
    """

    if est is None or skill is None:
        residual = TaskFrameResidualEstimate(
            skill_id="" if skill is None else str(skill.skill_type),
            stage_name="" if est is None else str(est.stage_name),
            reference_frame="" if skill is None else str(skill.reference_entity),
            target_frame="" if skill is None else str(skill.target_entity),
            active_dofs=tuple(() if skill is None else skill.controlled_dofs),
            dx=float("inf"),
            dy=float("inf"),
            dz=float("inf"),
            dyaw=float("inf"),
            axis_validity={"x": False, "y": False, "z": False, "yaw": False},
            axis_confidence={"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            observability=0.0,
            frame_consistency=0.0,
            abstain_reason="missing_estimate" if est is None else "missing_skill",
            uses_privileged_runtime=False,
        )
        cfg = AlignmentTakeoverConfig(
            xy_threshold=float(skill.xy_tolerance if skill is not None else 0.005),
            z_threshold=float(skill.z_tolerance if skill is not None else 0.010),
            yaw_threshold=float(skill.yaw_tolerance if skill is not None else 0.03),
            min_observability=float(max(skill.confidence_threshold if skill is not None else 0.30, 0.30)),
            min_frame_consistency=float(min_frame_consistency),
            z_required=True,
            yaw_required=True,
        )
        return evaluate_alignment_readiness(residual, cfg)

    residual = TaskFrameResidualEstimate(
        skill_id=str(skill.skill_type),
        stage_name=str(est.stage_name),
        reference_frame=str(skill.reference_entity or est.reference_entity),
        target_frame=str(skill.target_entity or est.target_entity),
        active_dofs=tuple(str(dof) for dof in skill.controlled_dofs),
        dx=float(est.dx),
        dy=float(est.dy),
        dz=float(est.dz),
        dyaw=float(est.dyaw),
        axis_validity={
            "x": bool(est.x_valid),
            "y": bool(est.y_valid),
            "z": bool(est.z_valid),
            "yaw": bool(est.yaw_valid),
        },
        axis_confidence={
            "x": float(est.x_confidence),
            "y": float(est.y_confidence),
            "z": float(est.z_confidence),
            "yaw": float(est.yaw_confidence),
        },
        observability=float(est.confidence),
        frame_consistency=float(est.frame_consistency),
        abstain_reason=str(est.reason) if str(est.reason) in {"prior_only", "prior_only_reacquire", "reacquire_needed"} else "",
        z_semantics=str(skill.z_semantics or "task_approach_axis_residual"),
        yaw_semantics="task_frame_yaw_residual",
        source=str(est.source),
        uses_privileged_runtime=False,
    )
    cfg = AlignmentTakeoverConfig(
        xy_threshold=float(skill.xy_tolerance),
        z_threshold=float(skill.z_tolerance),
        yaw_threshold=float(skill.yaw_tolerance),
        min_observability=float(max(float(skill.confidence_threshold), 0.30)),
        min_frame_consistency=float(min_frame_consistency),
        z_required=bool(_skill_requires_axis(skill, "z")),
        yaw_required=bool(_skill_requires_axis(skill, "yaw")),
    )
    return evaluate_alignment_readiness(residual, cfg)


@dataclass
class PrecisionObservationBundle:
    front_rgb: Optional[np.ndarray] = None
    wrist_rgb: Optional[np.ndarray] = None
    wrist_depth: Optional[np.ndarray] = None
    force_reading: Optional[np.ndarray] = None
    force_history: Optional[np.ndarray] = None
    proprio: Optional[np.ndarray] = None
    gripper_pose: Optional[np.ndarray] = None
    gripper_open: float = 1.0
    instruction: str = ""

    @classmethod
    def from_observation(cls, observation: Any, *, instruction: str = "") -> "PrecisionObservationBundle":
        proprio = _obs_get(observation, "proprio", None)
        if proprio is None:
            joint_positions = _obs_get(observation, "joint_positions", None)
            gripper_pose = _obs_get(observation, "gripper_pose", None)
            gripper_open = float(_obs_get(observation, "gripper_open", 1.0))
            if joint_positions is not None and gripper_pose is not None:
                proprio = np.concatenate(
                    [np.asarray(joint_positions, dtype=np.float32), np.asarray(gripper_pose, dtype=np.float32), [gripper_open]]
                ).astype(np.float32)
        else:
            proprio = np.asarray(proprio, dtype=np.float32)
        front_rgb = _obs_get(observation, "front_rgb", None)
        wrist_rgb = _obs_get(observation, "wrist_rgb", None)
        wrist_depth = _obs_get(observation, "wrist_depth", None)
        force_reading = _obs_get(observation, "gripper_touch_forces", None)
        force_history = _obs_get(observation, "force_history", None)
        gripper_pose = _obs_get(observation, "gripper_pose", None)
        return cls(
            front_rgb=np.asarray(front_rgb) if front_rgb is not None else None,
            wrist_rgb=np.asarray(wrist_rgb) if wrist_rgb is not None else None,
            wrist_depth=np.asarray(wrist_depth) if wrist_depth is not None else None,
            force_reading=np.asarray(force_reading) if force_reading is not None else None,
            force_history=np.asarray(force_history) if force_history is not None else None,
            proprio=proprio,
            gripper_pose=np.asarray(gripper_pose) if gripper_pose is not None else None,
            gripper_open=float(_obs_get(observation, "gripper_open", 1.0)),
            instruction=instruction,
        )


class PrecisionSkillSupervisor:
    """Task-spec driven, non-privileged local skill supervisor."""

    def __init__(
        self,
        task_spec: Optional[PrecisionTaskSpec],
        *,
        mode: str = "full_owner_by_stage",
        shadow_only: bool = False,
        grasp_localizer: Any = None,
        spoke_localizer: Any = None,
        force_classifier: Any = None,
        basin_recovery_config: BasinRecoveryConfig | None = None,
        max_xy_step: float = 0.0010,
        max_yaw_step: float = 0.035,
        max_dz_step: float = 0.0010,
    ) -> None:
        self.task_spec = task_spec
        self.mode = str(mode)
        self.shadow_only = bool(shadow_only or mode == "c2c_stage_shadow")
        self.max_xy_step = float(max_xy_step)
        self.max_yaw_step = float(max_yaw_step)
        self.max_dz_step = float(max_dz_step)
        self.stage_names = list(task_spec.stage_names()) if task_spec is not None else []
        self._default_stage = task_spec.default_stage if task_spec is not None else ""
        self.stage_index = 0
        self.stage_age = 0
        self._last_trace: dict[str, Any] = {}
        self._force_history: deque[np.ndarray] = deque(maxlen=5)
        self.safety = ResidualSafety(max_residual_pos=max_xy_step, max_residual_rot=max_yaw_step, max_delta_pos=0.025, max_delta_rot=0.10)
        self.grasp_localizer = grasp_localizer if grasp_localizer is not None else RingGraspLocalizer(shadow_only=True)
        self.spoke_localizer = spoke_localizer if spoke_localizer is not None else RingSpokeAlignLocalizer(shadow_only=True)
        self.force_classifier = force_classifier
        self.grasp_controller = GraspContactController(safety=self.safety, max_xy_step=max_xy_step, max_yaw_step=max_yaw_step)
        self.slide_controller = GuardedSlideController(safety=self.safety, max_xy_step=max_xy_step, max_yaw_step=max_yaw_step, max_z_step=max_dz_step)
        self.recovery = RecoveryFSM(safety=self.safety, backoff_m=0.003, lateral_m=0.0015, unload_m=0.0010, yaw_rad=max_yaw_step)
        self.basin_recovery = BasinRecoverySupervisor(config=basin_recovery_config)
        self._basin_state_estimator = CalibratedGraspBasinEstimator()
        self._skill_entry_z: dict[str, Optional[float]] = {"precision_grasp": None, "precision_align": None}
        self._resume_stage_name: Optional[str] = None
        self._force_feature_history: deque[np.ndarray] = deque(maxlen=12)
        self._prev_force_raw = np.zeros(6, dtype=np.float32)
        self._prev_gripper_z: Optional[float] = None
        self._grasp_ready_stable_frames = 0
        self._last_basin_record: dict[str, Any] = {}
        self._last_basin_visual_error: Optional[np.ndarray] = None
        self._last_estimated_basin_error: Optional[EstimatedBasinError] = None
        self._last_basin_state_calibration: Optional[BasinStateCalibration] = None
        self._last_current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._precision_gate_active: dict[str, bool] = {"precision_grasp": False, "precision_align": False}
        self._precision_gate_stable_frames: dict[str, int] = {"precision_grasp": 0, "precision_align": 0}
        self._last_precision_gate_info: dict[str, Any] = {}

    def reset(self) -> None:
        self.stage_index = 0
        self.stage_age = 0
        self._last_trace = {}
        self._force_history.clear()
        self._force_feature_history.clear()
        self._prev_force_raw = np.zeros(6, dtype=np.float32)
        self._prev_gripper_z = None
        self._skill_entry_z = {"precision_grasp": None, "precision_align": None}
        self._resume_stage_name = None
        self._grasp_ready_stable_frames = 0
        self.grasp_controller.reset()
        self.slide_controller.reset()
        self.recovery.reset()
        self.basin_recovery.reset()
        if self.task_spec is not None and self._default_stage and self._default_stage in self.stage_names:
            self.stage_index = self.stage_names.index(self._default_stage)
        self._last_basin_record = {}
        self._last_basin_visual_error = None
        self._last_estimated_basin_error = None
        self._last_basin_state_calibration = None
        self._last_current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._precision_gate_active = {"precision_grasp": False, "precision_align": False}
        self._precision_gate_stable_frames = {"precision_grasp": 0, "precision_align": 0}
        self._last_precision_gate_info = {}

    def current_stage(self) -> Optional[str]:
        if self.task_spec is None or not self.stage_names:
            return None
        return self.stage_names[min(self.stage_index, len(self.stage_names) - 1)]

    def current_stage_spec(self) -> Optional[StageTransition]:
        stage_name = self.current_stage()
        if self.task_spec is None or stage_name is None:
            return None
        return self.task_spec.get_stage(stage_name)

    def _skill_for_name(self, skill_name: Optional[str]) -> Optional[PrecisionSkillSpec]:
        if self.task_spec is None or not skill_name:
            return None
        return self.task_spec.skills.get(skill_name)

    def _skill_for_type(self, skill_type: str) -> Optional[PrecisionSkillSpec]:
        if self.task_spec is None:
            return None
        return self.task_spec.skill_by_type(skill_type)

    @staticmethod
    def _extract_current_quat(proprio) -> np.ndarray:
        if proprio is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        arr = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if arr.size >= 14:
            return arr[10:14].copy()
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    @staticmethod
    def _pose_to_abs_action(current_gripper_pose: np.ndarray, delta_local: np.ndarray, gripper_open: float) -> np.ndarray:
        pose = np.asarray(current_gripper_pose, dtype=np.float32).copy().reshape(7)
        delta = np.asarray(delta_local, dtype=np.float32).copy().reshape(6)
        delta_world = local_delta_to_world(delta, pose[3:7]).astype(np.float32)
        pose[:3] = pose[:3] + delta_world[:3]
        r_cur = Rotation.from_quat(pose[3:7])
        r_delta = Rotation.from_rotvec(delta_world[3:6])
        pose[3:7] = (r_delta * r_cur).as_quat().astype(np.float32)
        gripper_cmd = 1.0 if float(gripper_open) > 0.5 else 0.0
        return np.concatenate([pose[:7], [gripper_cmd]]).astype(np.float32)

    def _skill_context_for_stage(self, stage: StageTransition) -> tuple[Optional[PrecisionSkillSpec], str]:
        skill = self._skill_for_name(stage.skill_name)
        if skill is not None:
            return skill, skill.skill_type
        if stage.owner == "c2c_depth":
            skill = self._skill_for_type("precision_grasp") or self._skill_for_type("precision_align")
            return skill, skill.skill_type if skill is not None else ""
        if stage.owner == "c2c_force":
            skill = self._skill_for_type("guarded_slide") or self._skill_for_type("precision_grasp")
            return skill, skill.skill_type if skill is not None else ""
        if stage.owner == "c2c_recovery":
            skill = self._skill_for_type("recovery")
            return skill, skill.skill_type if skill is not None else ""
        return None, ""

    def _can_apply_stage(self, stage: StageTransition) -> bool:
        if self.shadow_only or self.task_spec is None:
            return False
        skill, skill_type = self._skill_context_for_stage(stage)
        if self.mode == "planner_only":
            return False
        if self.mode == "grasp_depth_apply":
            return stage.owner == "c2c_depth" and skill_type == "precision_grasp"
        if self.mode == "spoke_depth_apply":
            return stage.owner == "c2c_depth" and skill_type == "precision_align"
        if self.mode == "force_recovery":
            return stage.owner in {"c2c_force", "c2c_recovery"} and skill_type in {"guarded_slide", "recovery", "precision_grasp"}
        if self.mode == "basin_recovery_only":
            return False
        return stage.owner in {"c2c_depth", "c2c_force", "c2c_recovery"}

    def _basin_recovery_mode_enabled(self) -> bool:
        return self.mode in {"basin_recovery_shadow", "basin_recovery_only"}

    def _basin_recovery_apply_enabled(self) -> bool:
        return self.mode == "basin_recovery_only" and not self.shadow_only

    def _basin_target_stage(self, stage: StageTransition, active_skill: Optional[PrecisionSkillSpec]) -> bool:
        if active_skill is not None and active_skill.skill_type == "precision_grasp":
            return True
        if stage.name == "RECOVER" and (self._resume_stage_name or stage.recover_to_stage or "").startswith("RING_GRASP"):
            return True
        return False

    @staticmethod
    def _stage_gate_key(stage: StageTransition, active_skill: Optional[PrecisionSkillSpec]) -> Optional[str]:
        if active_skill is not None and active_skill.skill_type in {"precision_grasp", "precision_align"}:
            return active_skill.skill_type
        stage_name = str(stage.name or "")
        resume_name = str(stage.recover_to_stage or "")
        if stage_name.startswith("RING_GRASP") or resume_name.startswith("RING_GRASP"):
            return "precision_grasp"
        if stage_name.startswith("RING_SPOKE") or stage_name.startswith("SLIDE_ON_SPOKE") or resume_name.startswith("RING_SPOKE"):
            return "precision_align"
        return None

    def _reset_inactive_precision_gates(self, current_key: Optional[str]) -> None:
        for key in list(self._precision_gate_active.keys()):
            if key != current_key:
                self._precision_gate_active[key] = False
                self._precision_gate_stable_frames[key] = 0

    def _precision_gate_status(
        self,
        *,
        stage: StageTransition,
        active_skill: Optional[PrecisionSkillSpec],
        local_base: np.ndarray,
        grasp_error: LocalGeometryError,
        spoke_error: LocalGeometryError,
        estimated_basin_error: Optional[EstimatedBasinError] = None,
    ) -> tuple[bool, dict[str, Any]]:
        key = self._stage_gate_key(stage, active_skill)
        info = {
            "skill": key or "none",
            "active": True,
            "reason": "not_precision_stage",
            "stable_frames": 0,
            "required_frames": 0,
            "pullback_gate_ready": False,
            "micro_entry_ready": False,
            "close_ready": False,
            "pullback_block_reason": "not_precision_stage",
            "micro_entry_block_reason": "not_precision_stage",
            "close_block_reason": "not_precision_stage",
            "takeover_gate_kind": "none",
        }
        if key is None:
            self._last_precision_gate_info = dict(info)
            return True, info

        self._reset_inactive_precision_gates(key)
        skill = active_skill if active_skill is not None and active_skill.skill_type == key else self._skill_for_type(key)
        if skill is None:
            info.update(active=False, reason="missing_skill")
            self._last_precision_gate_info = dict(info)
            return False, info
        info["skill_name"] = str(skill.name)

        flags = dict(self.task_spec.runtime_flags if self.task_spec is not None else {})
        required_frames = int(flags.get("precision_takeover_stable_frames", 3) or 3)
        error_xy_max = float(flags.get(f"{key}_takeover_error_xy_max", 0.018 if key == "precision_grasp" else 0.022))
        depth_max = float(flags.get(f"{key}_takeover_depth_max", 0.12 if key == "precision_grasp" else 0.10))
        conf_min = float(flags.get(f"{key}_takeover_confidence_min", max(skill.shadow_confidence, 0.28 if key == "precision_grasp" else 0.012)))
        obs_min = float(flags.get(f"{key}_takeover_observability_min", 0.0012 if key == "precision_grasp" else 0.0008))

        info["required_frames"] = required_frames
        info["takeover_depth_max"] = depth_max
        if self._precision_gate_active.get(key, False):
            info.update(
                active=True,
                reason="latched",
                stable_frames=int(self._precision_gate_stable_frames.get(key, required_frames)),
                pullback_gate_ready=True,
                pullback_block_reason="ready",
                takeover_gate_kind="pullback",
            )
            self._last_precision_gate_info = dict(info)
            return True, info

        if stage.owner == "planner":
            self._precision_gate_stable_frames[key] = 0
            info.update(active=False, reason="planner_owned_stage")
            self._last_precision_gate_info = dict(info)
            return False, info

        local_error = grasp_error if key == "precision_grasp" else spoke_error
        est = estimated_basin_error if estimated_basin_error is not None else None
        if est is not None:
            if not est.has_trusted_control_axis:
                self._precision_gate_stable_frames[key] = 0
                info.update(
                    active=False,
                    reason="no_trusted_control_axis",
                    absolute_nearfield_depth=float("inf"),
                    nearfield_depth_max=float(depth_max),
                    target_xy_error=float("inf"),
                    target_xy_error_max=float(error_xy_max),
                    localizer_visible=False,
                    depth_nearfield=False,
                    target_nearfield=False,
                    pullback_gate_ready=False,
                    micro_entry_ready=False,
                    close_ready=False,
                    pullback_block_reason="no_trusted_control_axis",
                    micro_entry_block_reason="no_trusted_control_axis",
                    close_block_reason="no_trusted_control_axis",
                    takeover_gate_kind="none",
                )
                self._last_precision_gate_info = dict(info)
                return False, info
            visible = bool(est.confidence >= conf_min and est.frame_consistency >= self.basin_recovery.config.visual_observability_threshold)
            error_xy = float(np.hypot(float(est.dx), float(est.dy))) if np.isfinite(est.dx) and np.isfinite(est.dy) else float("inf")
            error_depth = float(abs(est.dz)) if np.isfinite(est.dz) else float("inf")
            target_near = bool(error_xy <= error_xy_max and (est.x_valid or est.y_valid))
            depth_near = bool(error_depth <= depth_max and est.z_valid)
            pullback_ready = bool(
                visible
                and est.pullback_ready(
                    xy_threshold=error_xy_max,
                    min_frame_consistency=self.basin_recovery.config.visual_observability_threshold,
                )
            )
            micro_xy_max = float(getattr(self.basin_recovery.config, "micro_entry_xy_threshold", error_xy_max))
            micro_z_max = float(getattr(self.basin_recovery.config, "micro_entry_z_threshold", depth_max))
            micro_entry_ready = bool(
                pullback_ready
                and error_xy <= micro_xy_max
                and depth_near
                and error_depth <= micro_z_max
            )
            alignment_readiness = _alignment_readiness_from_estimate(
                est,
                skill,
                min_frame_consistency=self.basin_recovery.config.visual_observability_threshold,
            )
            close_ready = bool(alignment_readiness.alignment_ready_for_handoff)
            pullback_blocks: list[str] = []
            if not visible:
                pullback_blocks.append("visibility")
            est_pullback_reason = est.pullback_block_reason(
                xy_threshold=error_xy_max,
                min_frame_consistency=self.basin_recovery.config.visual_observability_threshold,
            )
            if est_pullback_reason != "ready":
                pullback_blocks.extend(est_pullback_reason.split("+"))
            micro_blocks = list(pullback_blocks)
            if error_xy > micro_xy_max:
                micro_blocks.append("xy")
            if not depth_near or error_depth > micro_z_max:
                micro_blocks.append("z")
            close_blocks: list[str] = []
            if not close_ready:
                if not est.valid:
                    close_blocks.append("invalid_estimate")
                if error_xy > float(skill.xy_tolerance) or not est.xy_valid:
                    close_blocks.append("xy")
                if error_depth > float(skill.z_tolerance) or not est.z_valid:
                    close_blocks.append("z")
                close_blocks.extend(str(alignment_readiness.block_reason).split("+"))
        else:
            visible = bool(local_error.valid and local_error.confidence >= conf_min and local_error.observability >= obs_min)
            error_xy = float(np.hypot(float(local_error.dx), float(local_error.dy))) if np.isfinite(local_error.dx) and np.isfinite(local_error.dy) else float("inf")
            error_depth = float(local_error.dz) if np.isfinite(local_error.dz) else float("inf")
            target_near = bool(error_xy <= error_xy_max)
            depth_near = bool(error_depth <= depth_max)
            pullback_ready = bool(visible and depth_near and target_near)
            micro_entry_ready = bool(pullback_ready)
            close_ready = False
            pullback_blocks = []
            if not visible:
                pullback_blocks.append("visibility")
            if not target_near:
                pullback_blocks.append("xy_outside_pullback_window")
            if not depth_near:
                pullback_blocks.append("z")
            micro_blocks = list(pullback_blocks)
            close_blocks = ["estimated_basin_error_required"]

        if pullback_ready:
            self._precision_gate_stable_frames[key] = int(self._precision_gate_stable_frames.get(key, 0)) + 1
            info_reason = "stable_pullback_window"
        else:
            self._precision_gate_stable_frames[key] = 0
            if not visible:
                info_reason = "waiting_visible_confident_localizer"
            elif not target_near:
                info_reason = "waiting_target_near_precision_basin"
            elif not depth_near:
                info_reason = "waiting_absolute_nearfield_depth"
            else:
                info_reason = "waiting_pullback_gate"

        stable_frames = int(self._precision_gate_stable_frames.get(key, 0))
        info.update(
            absolute_nearfield_depth=float(error_depth),
            nearfield_depth_max=float(depth_max),
            target_xy_error=float(error_xy),
            target_xy_error_max=float(error_xy_max),
            localizer_visible=bool(visible),
            depth_nearfield=bool(depth_near),
            target_nearfield=bool(target_near),
            pullback_gate_ready=bool(pullback_ready),
            micro_entry_ready=bool(micro_entry_ready),
            close_ready=bool(close_ready),
            pullback_block_reason="+".join(dict.fromkeys(pullback_blocks)) if pullback_blocks else "ready",
            micro_entry_block_reason="+".join(dict.fromkeys(micro_blocks)) if micro_blocks else "ready",
            close_block_reason="+".join(dict.fromkeys(part for part in close_blocks if part)) if close_blocks else "ready",
            takeover_gate_kind="pullback" if pullback_ready else "none",
        )
        if stable_frames >= required_frames:
            self._precision_gate_active[key] = True
            info.update(active=True, reason="takeover_gate_armed", stable_frames=stable_frames)
        else:
            info.update(active=False, reason=info_reason, stable_frames=stable_frames)
        self._last_precision_gate_info = dict(info)
        return bool(info["active"]), info

    def _make_live_basin_record(
        self,
        *,
        stage: StageTransition,
        grasp_error: LocalGeometryError,
        bundle: PrecisionObservationBundle,
        robot_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        wrist_valid_depth_ratio = float(robot_state.get("wrist_valid_depth_ratio", 0.0) or 0.0)
        wrist_depth_near_fraction = float(robot_state.get("wrist_depth_near_fraction", 0.0) or 0.0)
        wrist_is_occluded = bool(robot_state.get("wrist_is_occluded", False))
        wrist_is_low_visibility = bool(robot_state.get("wrist_is_low_visibility", False))
        wide = self._wide_ring_glimpse(bundle)
        axis_strength = float(np.clip(1.0 - max(float(grasp_error.fit_residual), 0.0), 0.0, 1.0))
        conf = float(max(grasp_error.confidence, 0.0))
        obs = float(max(grasp_error.observability, 0.0))
        if wrist_is_low_visibility:
            conf *= 0.4
            obs *= 0.6
        if wrist_is_occluded:
            conf *= 0.2
            obs *= 0.4
            axis_strength *= 0.5
        if wide["visible"] and conf <= 0.0:
            conf = max(conf, 0.08)
            obs = max(obs, float(wide["observability"]))
        return {
            "stage_name": stage.name,
            "frame_confidence": float(conf),
            "frame_observability": float(obs),
            "frame_axis_strength": float(axis_strength),
            "frame_completeness": float(np.clip(obs * (1.0 - 0.5 * float(wrist_is_occluded)), 0.0, 1.0)),
            "frame_border_touch": 1.0 if wrist_is_low_visibility else 0.0,
            "trace_error_confidence": float(conf),
            "wrist_valid_depth_ratio": wrist_valid_depth_ratio,
            "wrist_depth_near_fraction": wrist_depth_near_fraction,
            "wrist_is_occluded": wrist_is_occluded,
            "wrist_is_low_visibility": wrist_is_low_visibility,
            "gripper_open": float(bundle.gripper_open),
            "wide_ring_visible": bool(wide["visible"]),
            "wide_ring_dx": float(wide["dx"]),
            "wide_ring_dy": float(wide["dy"]),
            "wide_ring_observability": float(wide["observability"]),
        }

    def _estimated_basin_error(
        self,
        *,
        stage: StageTransition,
        grasp_error: LocalGeometryError,
        robot_state: Mapping[str, Any],
        active_skill: Optional[PrecisionSkillSpec],
    ) -> EstimatedBasinError:
        skill = active_skill
        if skill is None:
            skill = self._skill_for_type("precision_grasp") or self._skill_for_type("precision_align")
        if skill is None:
            return EstimatedBasinError(
                valid=False,
                confidence=0.0,
                dx=0.0,
                dy=0.0,
                dz=0.0,
                dyaw=0.0,
                x_valid=False,
                y_valid=False,
                z_valid=False,
                yaw_valid=False,
                x_confidence=0.0,
                y_confidence=0.0,
                z_confidence=0.0,
                yaw_confidence=0.0,
                frame_consistency=0.0,
                source="no_skill",
                reason="no_skill",
                stage_name=stage.name,
            )
        runtime_calibration = None
        if self.task_spec is not None:
            runtime_calibration = self.task_spec.runtime_flags.get("basin_state_calibration")
        if runtime_calibration is not None:
            if isinstance(runtime_calibration, BasinStateCalibration):
                calibration = runtime_calibration
            elif isinstance(runtime_calibration, str):
                calibration = load_basin_state_calibration_report(runtime_calibration) or BasinStateCalibration.from_dict({})
            elif isinstance(runtime_calibration, Mapping) and ("axis_summary" in runtime_calibration or "recommendation" in runtime_calibration):
                calibration = BasinStateCalibration.from_report(runtime_calibration)
            else:
                calibration = BasinStateCalibration.from_dict(runtime_calibration if isinstance(runtime_calibration, Mapping) else {})
            self._last_basin_state_calibration = calibration
            estimator = CalibratedGraspBasinEstimator(calibration)
        else:
            self._last_basin_state_calibration = self._basin_state_estimator.calibration
            estimator = self._basin_state_estimator
        estimated = estimator.estimate(grasp_error, robot_state, self.task_spec, skill, stage_name=stage.name)
        return estimated

    def _basin_visual_error(self, grasp_error: LocalGeometryError) -> Optional[np.ndarray]:
        if not np.isfinite(grasp_error.dx) or not np.isfinite(grasp_error.dy):
            return None
        if grasp_error.observability <= 0.0 and grasp_error.confidence <= 0.0:
            return None
        dyaw = float(grasp_error.dyaw) if bool(getattr(grasp_error, "yaw_valid", True)) else 0.0
        if not np.isfinite(dyaw):
            dyaw = 0.0
        return np.asarray([float(grasp_error.dx), float(grasp_error.dy), dyaw], dtype=np.float32)

    @staticmethod
    def _wide_ring_glimpse(bundle: PrecisionObservationBundle) -> dict[str, Any]:
        rgb = bundle.wrist_rgb if bundle.wrist_rgb is not None else bundle.front_rgb
        if rgb is None:
            return {"visible": False, "dx": 0.0, "dy": 0.0, "observability": 0.0}
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return {"visible": False, "dx": 0.0, "dy": 0.0, "observability": 0.0}
        b = arr[..., 2]
        r = arr[..., 0]
        g = arr[..., 1]
        mask = (b > 90.0) & (b > 1.15 * r) & (b > 1.10 * g)
        mask = ndi.binary_opening(mask, iterations=1)
        mask = ndi.binary_closing(mask, iterations=1)
        if not np.any(mask):
            return {"visible": False, "dx": 0.0, "dy": 0.0, "observability": 0.0}
        labeled, num = ndi.label(mask)
        if num > 1:
            counts = ndi.sum(mask.astype(np.int32), labeled, index=np.arange(1, num + 1))
            keep = int(np.argmax(counts) + 1)
            mask = labeled == keep
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return {"visible": False, "dx": 0.0, "dy": 0.0, "observability": 0.0}
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        h, w = mask.shape
        depth_med = 0.25
        if bundle.wrist_depth is not None:
            depth_arr = np.asarray(bundle.wrist_depth, dtype=np.float32)
            if depth_arr.ndim == 3:
                depth_arr = depth_arr[..., 0]
            vals = depth_arr[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                depth_med = float(np.median(vals))
        fx = max(w * 1.15, 1.0)
        fy = max(h * 1.15, 1.0)
        dx = (cx - 0.5 * (w - 1)) * (depth_med / fx)
        dy = (cy - 0.5 * (h - 1)) * (depth_med / fy)
        obs = float(np.count_nonzero(mask) / max(mask.size, 1))
        return {"visible": bool(obs >= 2.0e-4), "dx": float(dx), "dy": float(dy), "observability": float(obs)}

    def _low_level_error_for_stage(
        self,
        stage: StageTransition,
        observation: PrecisionObservationBundle,
        robot_state: Mapping[str, Any],
    ) -> tuple[LocalGeometryError, LocalGeometryError, Optional[PrecisionSkillSpec]]:
        skill, skill_type = self._skill_context_for_stage(stage)
        grasp_skill = self._skill_for_type("precision_grasp") or skill
        align_skill = self._skill_for_type("precision_align") or skill
        if grasp_skill is None:
            dummy = LocalGeometryError(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "no_grasp_skill")
            grasp_error = dummy
        else:
            grasp_error = self.grasp_localizer.localize(observation, robot_state, self.task_spec, grasp_skill, stage_name=stage.name)
        if align_skill is None:
            dummy = LocalGeometryError(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "no_align_skill")
            align_error = dummy
        else:
            align_error = self.spoke_localizer.localize(observation, robot_state, self.task_spec, align_skill, stage_name=stage.name)
        return grasp_error, align_error, skill

    def _evaluate_condition(
        self,
        condition: str,
        *,
        stage: StageTransition,
        observation: PrecisionObservationBundle,
        robot_state: Mapping[str, Any],
        grasp_error: LocalGeometryError,
        spoke_error: LocalGeometryError,
        estimated_basin_error: Optional[EstimatedBasinError],
        grasp_ctrl: Optional[SkillControlResult],
        slide_ctrl: Optional[SkillControlResult],
    ) -> bool:
        force_vec = np.asarray(observation.force_reading if observation.force_reading is not None else np.zeros(6, dtype=np.float32), dtype=np.float32).reshape(-1)[:6]
        force_norm = float(np.linalg.norm(force_vec[:3]))
        torque_norm = float(np.linalg.norm(force_vec[3:]))
        gripper_open = float(observation.gripper_open)
        gripper_z = float(observation.gripper_pose[2]) if observation.gripper_pose is not None and len(observation.gripper_pose) >= 3 else float("nan")
        entry_grasp_z = self._skill_entry_z.get("precision_grasp")
        entry_align_z = self._skill_entry_z.get("precision_align")
        if condition == "always":
            return True
        if condition == "grasp_visible":
            return grasp_error.valid and grasp_error.confidence >= max(0.1, self._skill_for_type("precision_grasp").shadow_confidence if self._skill_for_type("precision_grasp") else 0.1)
        if condition == "grasp_aligned":
            skill = self._skill_for_type("precision_grasp")
            if skill is None:
                return False
            est = estimated_basin_error
            if est is None:
                return False
            ready_now = bool(
                _alignment_readiness_from_estimate(
                    est,
                    skill,
                    min_frame_consistency=0.20,
                ).alignment_ready_for_handoff
            )
            if ready_now:
                self._grasp_ready_stable_frames += 1
            else:
                self._grasp_ready_stable_frames = 0
            required = int(self.task_spec.runtime_flags.get("grasp_ready_stable_frames", 4)) if self.task_spec is not None else 4
            return bool(ready_now and self._grasp_ready_stable_frames >= max(required, 1))
        if condition == "grasp_contacted":
            return bool(force_norm >= self.slide_controller.contact_threshold or (grasp_ctrl is not None and grasp_ctrl.contact_confirmed))
        if condition == "grasp_stable":
            return bool(grasp_ctrl is not None and grasp_ctrl.stable and gripper_open < 0.5)
        if condition == "lifted":
            return bool(np.isfinite(gripper_z) and np.isfinite(entry_grasp_z) and (gripper_z - entry_grasp_z) >= 0.018 and self._skill_entry_z.get("precision_grasp") is not None)
        if condition == "spoke_visible":
            return spoke_error.valid and spoke_error.confidence >= max(0.1, self._skill_for_type("precision_align").shadow_confidence if self._skill_for_type("precision_align") else 0.1)
        if condition == "spoke_aligned":
            skill = self._skill_for_type("precision_align")
            if skill is None:
                return False
            return (
                spoke_error.valid
                and spoke_error.confidence >= skill.confidence_threshold
                and abs(spoke_error.dx) <= skill.xy_tolerance
                and abs(spoke_error.dy) <= skill.xy_tolerance
                and abs(spoke_error.dyaw) <= skill.yaw_tolerance
            )
        if condition == "slide_contact":
            return bool(force_norm >= self.slide_controller.contact_threshold or (slide_ctrl is not None and slide_ctrl.contact_confirmed))
        if condition == "slide_seated":
            return bool(slide_ctrl is not None and slide_ctrl.state_name == "SLIDE" and force_norm <= self.slide_controller.contact_threshold * 0.8)
        if condition == "recover_complete":
            return self.recovery.phase == RecoveryPhase.IDLE
        if condition == "force_overload":
            return bool(force_norm > self.safety.force_stop_threshold or torque_norm > self.safety.torque_threshold)
        if condition == "invalid_action":
            return bool(robot_state.get("invalid_action_flag", False))
        if condition == "timeout":
            return stage.max_steps > 0 and self.stage_age >= stage.max_steps
        return False

    def _grasp_contact_rule_table(
        self,
        *,
        stage: StageTransition,
        observation: PrecisionObservationBundle,
        grasp_error: LocalGeometryError,
        estimated_basin_error: Optional[EstimatedBasinError],
        grasp_ctrl: Optional[SkillControlResult],
    ) -> dict[str, Any]:
        skill = self._skill_for_name("grasp_contact_ring") or self._skill_for_type("precision_grasp")
        force_vec = np.asarray(observation.force_reading if observation.force_reading is not None else np.zeros(6, dtype=np.float32), dtype=np.float32).reshape(-1)[:6]
        force_norm = float(np.linalg.norm(force_vec[:3]))
        force_threshold = float(self.slide_controller.contact_threshold)
        conf_threshold = float(max(skill.confidence_threshold if skill is not None else 0.35, 0.35))
        xy_tol = float(skill.xy_tolerance if skill is not None else 0.004)
        yaw_tol = float(skill.yaw_tolerance if skill is not None else 0.14)
        est = estimated_basin_error
        alignment_readiness = _alignment_readiness_from_estimate(est, skill, min_frame_consistency=0.20)
        close_ready = bool(alignment_readiness.alignment_ready_for_handoff)
        force_ready = bool(force_norm <= force_threshold)
        contact_confirmed = bool(grasp_ctrl.contact_confirmed if grasp_ctrl is not None else force_norm > force_threshold)
        gripper_override = None if grasp_ctrl is None else grasp_ctrl.gripper_override
        close_triggered = bool(gripper_override is not None and float(gripper_override) <= 0.0)
        rules = [
            {
                "rule": "stage_is_RING_GRASP_CONTACT",
                "passed": bool(stage.name == "RING_GRASP_CONTACT"),
                "value": stage.name,
                "threshold": "RING_GRASP_CONTACT",
            },
            {
                "rule": "geometry_confident_and_aligned",
                "passed": bool(close_ready),
                "value": {
                    "confidence": float(est.confidence if est is not None else grasp_error.confidence),
                    "dx": float(est.dx if est is not None else grasp_error.dx),
                    "dy": float(est.dy if est is not None else grasp_error.dy),
                    "dz": float(est.dz if est is not None else grasp_error.dz),
                    "dyaw": float(est.dyaw if est is not None else grasp_error.dyaw),
                },
                "threshold": {
                    "confidence": conf_threshold,
                    "xy_tolerance": xy_tol,
                    "z_tolerance": float(skill.z_tolerance if skill is not None else 0.010),
                    "yaw_tolerance": yaw_tol,
                    "alignment_handoff_block_reason": str(alignment_readiness.block_reason),
                },
            },
            {
                "rule": "force_below_contact_threshold",
                "passed": bool(force_ready),
                "value": float(force_norm),
                "threshold": float(force_threshold),
            },
            {
                "rule": "contact_confirmed",
                "passed": bool(contact_confirmed),
                "value": bool(contact_confirmed),
                "threshold": True,
            },
            {
                "rule": "gripper_override_close",
                "passed": bool(close_triggered),
                "value": None if gripper_override is None else float(gripper_override),
                "threshold": 0.0,
            },
        ]
        return {
            "decision": "close" if close_triggered else "hold",
            "reason": "contact_close" if close_triggered else ("contact_hold" if stage.name == "RING_GRASP_CONTACT" else "not_contact_stage"),
            "force_norm": float(force_norm),
            "force_threshold": float(force_threshold),
            "gripper_override": None if gripper_override is None else float(gripper_override),
            "contact_confirmed": bool(contact_confirmed),
            "stable": bool(grasp_ctrl.stable if grasp_ctrl is not None else False),
            "rules": rules,
        }

    def _maybe_transition(
        self,
        *,
        observation: PrecisionObservationBundle,
        robot_state: Mapping[str, Any],
        grasp_error: LocalGeometryError,
        spoke_error: LocalGeometryError,
        estimated_basin_error: Optional[EstimatedBasinError],
        grasp_ctrl: Optional[SkillControlResult],
        slide_ctrl: Optional[SkillControlResult],
    ) -> None:
        if self.task_spec is None or not self.stage_names:
            return
        stage = self.current_stage_spec()
        if stage is None or self.stage_age < stage.min_steps:
            return
        if self.recovery.phase != RecoveryPhase.IDLE:
            if stage.name != "RECOVER":
                self._resume_stage_name = stage.recover_to_stage or stage.name
            self._set_stage("RECOVER")
            return
        if stage.name == "RECOVER" and self._resume_stage_name:
            resume = self._resume_stage_name
            self._resume_stage_name = None
            self._set_stage(resume)
            return
        if self._evaluate_condition(
            stage.transition_on,
            stage=stage,
            observation=observation,
            robot_state=robot_state,
            grasp_error=grasp_error,
            spoke_error=spoke_error,
            estimated_basin_error=estimated_basin_error,
            grasp_ctrl=grasp_ctrl,
            slide_ctrl=slide_ctrl,
        ):
            if stage.next_stage is not None:
                self._set_stage(stage.next_stage)
            return
        if stage.max_steps > 0 and self.stage_age >= stage.max_steps and stage.recover_to_stage is not None:
            self._resume_stage_name = stage.recover_to_stage
            self._set_stage(stage.recover_to_stage)

    def _set_stage(self, stage_name: str) -> None:
        if stage_name not in self.stage_names:
            return
        self.stage_index = self.stage_names.index(stage_name)
        self.stage_age = 0
        stage = self.current_stage_spec()
        if stage is not None:
            skill = self._skill_for_name(stage.skill_name)
            if skill is not None:
                self._skill_entry_z[skill.skill_type] = None
            if stage.name in {"COARSE_TO_RING", "RING_GRASP_ALIGN", "RING_GRASP_CONTACT"}:
                self._grasp_ready_stable_frames = 0
            if stage.name == "RECOVER":
                # Preserve the target resume stage until recovery completes.
                pass
        stage_key = self._stage_gate_key(stage, self._skill_for_name(stage.skill_name) if stage is not None else None) if stage is not None else None
        self._reset_inactive_precision_gates(stage_key)

    def _trace_force(self, force_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._force_history.append(np.asarray(force_vec, dtype=np.float32).reshape(-1)[:6])
        if not self._force_history:
            return np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)
        stacked = np.stack(list(self._force_history), axis=0)
        filtered = np.mean(stacked, axis=0).astype(np.float32)
        return np.asarray(force_vec, dtype=np.float32).reshape(-1)[:6], filtered

    def _build_force_sequence_features(
        self,
        *,
        raw_force: np.ndarray,
        filtered_force: np.ndarray,
        gripper_open: float,
        gripper_z: float,
        invalid_action_flag: bool,
        planner_delta_7d: np.ndarray,
    ) -> torch.Tensor:
        delta_force = np.asarray(raw_force, dtype=np.float32).reshape(6) - np.asarray(self._prev_force_raw, dtype=np.float32).reshape(6)
        z_progress = 0.0 if self._prev_gripper_z is None or not np.isfinite(gripper_z) else float(gripper_z - self._prev_gripper_z)
        action_hist = np.asarray(planner_delta_7d, dtype=np.float32).reshape(-1)[:6]
        feat = np.concatenate(
            [
                np.asarray(raw_force, dtype=np.float32).reshape(6),
                np.asarray(filtered_force, dtype=np.float32).reshape(6),
                delta_force.reshape(6),
                np.asarray([float(gripper_open)], dtype=np.float32),
                np.asarray([float(z_progress)], dtype=np.float32),
                np.asarray([1.0 if invalid_action_flag else 0.0], dtype=np.float32),
                action_hist.reshape(6),
            ],
            axis=0,
        ).astype(np.float32)
        self._force_feature_history.append(feat)
        self._prev_force_raw = np.asarray(raw_force, dtype=np.float32).reshape(6)
        self._prev_gripper_z = float(gripper_z) if np.isfinite(gripper_z) else self._prev_gripper_z
        history = list(self._force_feature_history)
        if not history:
            history = [np.zeros(27, dtype=np.float32)]
        seq = np.stack(history, axis=0).astype(np.float32)
        return torch.from_numpy(seq).unsqueeze(0)

    def _trace_planner_only(self, planner_delta_7d: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
        base = np.asarray(planner_delta_7d, dtype=np.float32).copy()
        return base

    def _sync_recovery_from_control_result(self, ctrl: Optional[SkillControlResult], stage: StageTransition) -> None:
        if ctrl is None:
            return
        try:
            phase = RecoveryPhase(ctrl.state_name)
        except Exception:
            return
        if phase == RecoveryPhase.IDLE:
            return
        self.recovery.phase = phase
        self.recovery.cycle_id = max(self.recovery.cycle_id, int(ctrl.recovery_cycle_id))
        self._resume_stage_name = stage.recover_to_stage or self._resume_stage_name or stage.name

    def build_invalid_action_recovery_absolute(
        self,
        current_gripper_pose: np.ndarray,
        gripper_open: float,
        *,
        force_reading: Any = None,
        proprio: Any = None,
    ) -> np.ndarray:
        if self._basin_recovery_mode_enabled() and self._last_basin_record and bool(self._last_precision_gate_info.get("active", False)):
            decision = self.basin_recovery.step(
                record=self._last_basin_record,
                planner_prior_state=np.zeros(6, dtype=np.float32),
                estimated_basin_error=self._last_estimated_basin_error,
                visual_error_state=self._last_basin_visual_error,
                model_prediction=None,
                allow_eval_line_search=False,
            )
            abs_action = self._pose_to_abs_action(current_gripper_pose, decision.local_action_6d, gripper_open)
            if self._last_trace:
                self._last_trace.update(decision.to_trace())
                self._last_trace["force_skill_state"] = decision.mode.value
                self._last_trace["phase_owner"] = "basin_recovery"
                self._last_trace["phase_reason"] = decision.reason
                self._last_trace["local_correction_owner"] = "basin_recovery"
                self._last_trace["local_correction_local_6d"] = np.asarray(decision.local_action_6d, dtype=np.float32).tolist()
                self._last_trace["basin_recovery_invalid_action"] = True
            return abs_action
        local_base = np.zeros(6, dtype=np.float32)
        rec = self.recovery.step(force_reading=force_reading, local_base=local_base, invalid_action=True, jam=False)
        return self._pose_to_abs_action(current_gripper_pose, rec.delta_local, gripper_open)

    def step(
        self,
        planner_delta_7d: np.ndarray,
        observation: Any,
        robot_state: Optional[Mapping[str, Any]] = None,
        task_spec: Optional[PrecisionTaskSpec] = None,
        current_instruction: str = "",
    ) -> np.ndarray:
        if task_spec is not None:
            self.task_spec = task_spec
            if not self.stage_names and self.task_spec is not None:
                self.stage_names = list(self.task_spec.stage_names())
        bundle = observation if isinstance(observation, PrecisionObservationBundle) else PrecisionObservationBundle.from_observation(observation, instruction=current_instruction)
        robot_state = dict(robot_state or {})
        robot_state.setdefault("planner_delta_7d", np.asarray(planner_delta_7d, dtype=np.float32).tolist())
        planner_delta_7d = np.asarray(planner_delta_7d, dtype=np.float32).reshape(-1)
        current_quat = self._extract_current_quat(bundle.proprio)

        if self.task_spec is None or not self.stage_names:
            local_base = world_delta_to_local(planner_delta_7d[:6], current_quat).astype(np.float32)
            self._last_trace = {
                "c2c_v2_stage": "planner_only",
                "c2c_v2_skill_type": "none",
                "c2c_v2_owner": "planner",
                "c2c_v2_controlled_dofs": [],
                "c2c_v2_target_entity": "none",
                "c2c_v2_reference_entity": "none",
                "local_geometry_error": {},
                "localizer_confidence": 0.0,
                "localizer_abstained": True,
                "force_skill_state": "planner_only",
                "recovery_cycle_id": 0,
                "uses_privileged_target": False,
                "uses_rlbench_mask_runtime": False,
                "phase_owner": "planner",
                "phase_reason": "no_task_spec",
                "invalid_action_flag": False,
                "planner_action_world": planner_delta_7d[:6].tolist(),
                "planner_chunk_local_6d": local_base.tolist(),
                "local_command_local_6d": local_base.tolist(),
                "local_residual_vs_planner_local_6d": np.zeros(6, dtype=np.float32).tolist(),
                "pre_clip_action_world_6d": planner_delta_7d[:6].tolist(),
                "post_clip_action_world_6d": planner_delta_7d[:6].tolist(),
                "executed_action_world_6d": planner_delta_7d[:6].tolist(),
                "local_correction_local_6d": np.zeros(6, dtype=np.float32).tolist(),
                "local_correction_owner": "planner",
                "grasp_gripper_override": None,
                "slide_gripper_override": None,
                "c2c_stage_age": 0,
                "planner_reaches_precontact": False,
                "planner_reaches_preinsert": False,
                "raw_wrench": np.zeros(6, dtype=np.float32).tolist(),
                "filtered_wrench": np.zeros(6, dtype=np.float32).tolist(),
            }
            return planner_delta_7d.copy()

        if self.current_stage() is None:
            self._set_stage(self._default_stage or self.stage_names[0])

        stage = self.current_stage_spec()
        if stage is None:
            stage = self.task_spec.get_stage(self._default_stage or self.stage_names[0])

        self._last_current_quat = current_quat.copy()
        local_base = world_delta_to_local(planner_delta_7d[:6], current_quat).astype(np.float32)
        local_out = local_base.copy()
        active_skill, active_skill_type = self._skill_context_for_stage(stage)
        grasp_error, spoke_error, _ = self._low_level_error_for_stage(stage, bundle, robot_state)

        raw_force = np.asarray(bundle.force_reading if bundle.force_reading is not None else np.zeros(6, dtype=np.float32), dtype=np.float32).reshape(-1)[:6]
        raw_force, filtered_force = self._trace_force(raw_force)

        grasp_ctrl: Optional[SkillControlResult] = None
        slide_ctrl: Optional[SkillControlResult] = None
        invalid_action_flag = bool(robot_state.get("invalid_action_flag", False))
        force_model_probs = None
        force_model_triggered = False
        if self.force_classifier is not None:
            gripper_z_now = float(bundle.gripper_pose[2]) if bundle.gripper_pose is not None and len(bundle.gripper_pose) >= 3 else float("nan")
            force_seq = self._build_force_sequence_features(
                raw_force=raw_force,
                filtered_force=filtered_force,
                gripper_open=bundle.gripper_open,
                gripper_z=gripper_z_now,
                invalid_action_flag=invalid_action_flag,
                planner_delta_7d=planner_delta_7d,
            )
            try:
                skill_token = active_skill_type or ""
                force_model_probs = self.force_classifier.predict(
                    force_seq,
                    skill_type=skill_token,
                    stage_name=stage.name,
                    lengths=torch.tensor([force_seq.shape[1]], dtype=torch.long),
                )
                recovery_prob = float(force_model_probs.get("recovery_needed", torch.tensor([0.0]))[0].item())
                contact_prob = float(force_model_probs.get("contact", torch.tensor([0.0]))[0].item())
                jam_prob = float(force_model_probs.get("jam", torch.tensor([0.0]))[0].item())
                force_model_triggered = bool(recovery_prob >= 0.5 or jam_prob >= 0.6)
            except Exception:
                force_model_probs = None
                force_model_triggered = False

        # Record generic entry depths for high-precision skills.
        if active_skill is not None and active_skill.skill_type in self._skill_entry_z and self._skill_entry_z[active_skill.skill_type] is None:
            if bundle.gripper_pose is not None and len(bundle.gripper_pose) >= 3:
                self._skill_entry_z[active_skill.skill_type] = float(bundle.gripper_pose[2])

        estimated_basin_error = self._estimated_basin_error(
            stage=stage,
            grasp_error=grasp_error,
            robot_state=robot_state,
            active_skill=active_skill,
        )
        self._last_estimated_basin_error = estimated_basin_error

        gate_active, gate_info = self._precision_gate_status(
            stage=stage,
            active_skill=active_skill,
            local_base=local_base,
            grasp_error=grasp_error,
            spoke_error=spoke_error,
            estimated_basin_error=estimated_basin_error,
        )

        basin_decision = None
        if self._basin_recovery_mode_enabled() and self._basin_target_stage(stage, active_skill):
            self._last_basin_record = self._make_live_basin_record(stage=stage, grasp_error=grasp_error, bundle=bundle, robot_state=robot_state)
            self._last_basin_visual_error = self._basin_visual_error(grasp_error)
            basin_controller = self.basin_recovery
            if not gate_active:
                basin_controller = BasinRecoverySupervisor(config=self.basin_recovery.config)
            elif str(gate_info.get("reason", "")) == "takeover_gate_armed":
                self.basin_recovery.reset()
                basin_controller = self.basin_recovery
            basin_decision = basin_controller.step(
                record=self._last_basin_record,
                planner_prior_state=local_base,
                estimated_basin_error=estimated_basin_error,
                visual_error_state=self._last_basin_visual_error,
                model_prediction=None,
                allow_eval_line_search=False,
            )

        stage_apply_allowed = bool(gate_active and self._can_apply_stage(stage))
        basin_apply_allowed = bool(gate_active and basin_decision is not None and self._basin_recovery_apply_enabled())

        if basin_apply_allowed:
            local_out = np.asarray(basin_decision.local_action_6d, dtype=np.float32).reshape(6)
        elif stage.owner == "c2c_depth" and active_skill is not None and stage_apply_allowed:
            basin_error = estimated_basin_error if active_skill.skill_type == "precision_grasp" else None
            if basin_error is not None and basin_error.valid and basin_error.confidence >= active_skill.apply_confidence:
                if "x" in active_skill.controlled_dofs and basin_error.x_valid:
                    local_out[0] = float(np.clip(-basin_error.dx, -active_skill.max_xy_step, active_skill.max_xy_step))
                if "y" in active_skill.controlled_dofs and basin_error.y_valid:
                    local_out[1] = float(np.clip(-basin_error.dy, -active_skill.max_xy_step, active_skill.max_xy_step))
                if ("dz" in active_skill.controlled_dofs or active_skill.skill_type == "precision_grasp") and basin_error.z_valid:
                    local_out[2] = float(np.clip(-basin_error.dz, -active_skill.max_dz_step, active_skill.max_dz_step))
                if "yaw" in active_skill.controlled_dofs and basin_error.yaw_valid:
                    local_out[5] = float(np.clip(-0.7 * basin_error.dyaw, -active_skill.max_yaw_step, active_skill.max_yaw_step))
                if "x_micro" in active_skill.controlled_dofs and basin_error.x_valid:
                    local_out[0] = float(np.clip(local_out[0] - 0.5 * basin_error.dx, -active_skill.max_xy_step, active_skill.max_xy_step))
                if "y_micro" in active_skill.controlled_dofs and basin_error.y_valid:
                    local_out[1] = float(np.clip(local_out[1] - 0.5 * basin_error.dy, -active_skill.max_xy_step, active_skill.max_xy_step))
                if "yaw_micro" in active_skill.controlled_dofs and basin_error.yaw_valid:
                    local_out[5] = float(np.clip(local_out[5] - 0.5 * basin_error.dyaw, -active_skill.max_yaw_step, active_skill.max_yaw_step))
        elif stage.owner == "c2c_force" and active_skill is not None and stage_apply_allowed:
            if active_skill.skill_type == "precision_grasp":
                grasp_ctrl = self.grasp_controller.step(
                    error=grasp_error,
                    force_reading=filtered_force,
                    local_base=local_base,
                    gripper_open=bundle.gripper_open,
                    invalid_action=invalid_action_flag,
                )
                self._sync_recovery_from_control_result(grasp_ctrl, stage)
                local_out = local_base + grasp_ctrl.delta_local
                if grasp_ctrl.gripper_override is not None:
                    bundle.gripper_open = float(grasp_ctrl.gripper_override)
            else:
                slide_ctrl = self.slide_controller.step(
                    error=spoke_error,
                    force_reading=filtered_force,
                    local_base=local_base,
                    invalid_action=invalid_action_flag,
                )
                self._sync_recovery_from_control_result(slide_ctrl, stage)
                local_out = local_base + slide_ctrl.delta_local
        elif stage_apply_allowed and (stage.owner == "c2c_recovery" or self.recovery.phase != RecoveryPhase.IDLE or force_model_triggered):
            jam_flag = bool((grasp_ctrl and grasp_ctrl.contact_confirmed) or (slide_ctrl and slide_ctrl.contact_confirmed) or force_model_triggered)
            if stage.name != "RECOVER" and stage.recover_to_stage:
                self._resume_stage_name = stage.recover_to_stage
            rec = self.recovery.step(force_reading=filtered_force, local_base=local_base, invalid_action=invalid_action_flag, jam=jam_flag)
            local_out = local_base + rec.delta_local

        if self.shadow_only or not stage_apply_allowed:
            local_out = local_base.copy()
            if not self.shadow_only and stage.owner == "c2c_force" and active_skill is not None:
                # keep the force controller state but drop the command for planning-only modes
                local_out = local_base.copy()
        elif basin_decision is not None and not basin_apply_allowed:
            local_out = local_base.copy()

        world_out = local_delta_to_world(local_out, current_quat).astype(np.float32)
        final_action = planner_delta_7d.copy()
        final_action[:6] = world_out[:6]
        if not self.shadow_only and stage.owner in {"c2c_force", "c2c_recovery"} and bundle.gripper_open is not None:
            final_action[6] = 1.0 if float(bundle.gripper_open) > 0.5 else 0.0
        final_action = self.safety.clip_final_action(final_action)

        self._maybe_transition(
            observation=bundle,
            robot_state={**robot_state, "invalid_action_flag": invalid_action_flag},
            grasp_error=grasp_error,
            spoke_error=spoke_error,
            estimated_basin_error=estimated_basin_error,
            grasp_ctrl=grasp_ctrl,
            slide_ctrl=slide_ctrl,
        )
        self.stage_age += 1

        localizer_active = grasp_error if active_skill is not None and active_skill.skill_type == "precision_grasp" else spoke_error
        force_ctrl = grasp_ctrl or slide_ctrl
        force_state = force_ctrl.state_name if force_ctrl is not None else (self.recovery.phase.value if self.recovery.phase != RecoveryPhase.IDLE else "planner")
        grasp_close_trace = self._grasp_contact_rule_table(
            stage=stage,
            observation=bundle,
            grasp_error=grasp_error,
            estimated_basin_error=estimated_basin_error,
            grasp_ctrl=grasp_ctrl,
        )
        runtime_contract_decision = _runtime_contract_decision_from_estimate(
            estimated_basin_error,
            precision_row=bool(active_skill is not None and active_skill.skill_type in {"precision_grasp", "precision_align"}),
            requires_yaw_observability=bool(active_skill is not None and "yaw" in set(active_skill.controlled_dofs)),
            pullback_xy_threshold=float(gate_info.get("target_xy_error_max", active_skill.xy_tolerance if active_skill is not None else 0.005)),
        )
        strict_alignment_readiness = _alignment_readiness_from_estimate(
            estimated_basin_error,
            active_skill,
            min_frame_consistency=0.20,
        )
        legacy_basin_close_ready = bool(
            estimated_basin_error.close_ready(
                xy_threshold=float(active_skill.xy_tolerance if active_skill is not None else 0.005),
                z_threshold=float(active_skill.z_tolerance if active_skill is not None else 0.010),
                yaw_threshold=float(active_skill.yaw_tolerance if active_skill is not None else 0.03),
                yaw_required=bool(estimated_basin_error.yaw_valid) if estimated_basin_error is not None else False,
                min_frame_consistency=0.20,
            )
            if estimated_basin_error is not None
            else False
        )
        localizer_semantics = _local_geometry_semantics(localizer_active, skill_type=active_skill.skill_type if active_skill is not None else "none")

        self._last_trace = {
            "c2c_v2_stage": stage.name,
            "c2c_v2_skill_type": active_skill.skill_type if active_skill is not None else "none",
            "c2c_v2_owner": stage.owner,
            "c2c_v2_controlled_dofs": list(active_skill.controlled_dofs) if active_skill is not None else [],
            "c2c_v2_target_entity": active_skill.target_entity if active_skill is not None else "none",
            "c2c_v2_reference_entity": active_skill.reference_entity if active_skill is not None else "none",
            "local_geometry_error": {
                "grasp": _jsonable_value(grasp_error.__dict__),
                "spoke": _jsonable_value(spoke_error.__dict__),
            },
            "runtime_frame_residual_estimate": _jsonable_value(localizer_semantics),
            "estimated_basin_error": _jsonable_value(
                estimated_basin_error.to_trace(
                    xy_threshold=float(active_skill.xy_tolerance if active_skill is not None else 0.005),
                    pullback_xy_threshold=float(gate_info.get("target_xy_error_max", active_skill.xy_tolerance if active_skill is not None else 0.005)),
                    z_threshold=float(active_skill.z_tolerance if active_skill is not None else 0.010),
                    yaw_threshold=float(active_skill.yaw_tolerance if active_skill is not None else 0.03),
                    yaw_required=bool(estimated_basin_error.yaw_valid),
                    min_frame_consistency=0.20,
                )
                if estimated_basin_error is not None
                else {}
            ),
            "localizer_confidence": float(localizer_active.confidence if localizer_active is not None else 0.0),
            "localizer_abstained": bool(localizer_active is None or not localizer_active.valid),
            "force_skill_state": force_state,
            "force_model_recovery_needed": float(force_model_probs["recovery_needed"][0].item()) if force_model_probs is not None and "recovery_needed" in force_model_probs else 0.0,
            "force_model_contact": float(force_model_probs["contact"][0].item()) if force_model_probs is not None and "contact" in force_model_probs else 0.0,
            "force_model_jam": float(force_model_probs["jam"][0].item()) if force_model_probs is not None and "jam" in force_model_probs else 0.0,
            "force_model_misgrasp": float(force_model_probs["misgrasp"][0].item()) if force_model_probs is not None and "misgrasp" in force_model_probs else 0.0,
            "force_model_slip": float(force_model_probs["slip"][0].item()) if force_model_probs is not None and "slip" in force_model_probs else 0.0,
            "force_model_triggered": force_model_triggered,
            "recovery_cycle_id": int((force_ctrl.recovery_cycle_id if force_ctrl is not None else self.recovery.cycle_id)),
            "basin_pullback_variant": str(self.basin_recovery.config.variant_name),
            "basin_visual_gain": float(self.basin_recovery.config.visual_gain),
            "basin_max_pullback_xy_step": float(self.basin_recovery.config.max_pullback_xy_step),
            "basin_max_recovery_steps": int(self.basin_recovery.config.max_recovery_steps),
            "uses_privileged_target": False,
            "uses_rlbench_mask_runtime": False,
            "basin_axis_validity": estimated_basin_error.axis_validity if estimated_basin_error is not None else {"x": False, "y": False, "z": False, "yaw": False},
            "basin_axis_confidence": estimated_basin_error.axis_confidence if estimated_basin_error is not None else {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "basin_pullback_ready_axes": list(estimated_basin_error.pullback_ready_axes) if estimated_basin_error is not None else [],
            "basin_control_gate_axes": list(estimated_basin_error.trusted_control_axes) if estimated_basin_error is not None else [],
            "basin_axis_policy": {
                "x": str(self._last_basin_state_calibration.x.policy if self._last_basin_state_calibration is not None else "abstain"),
                "y": str(self._last_basin_state_calibration.y.policy if self._last_basin_state_calibration is not None else "abstain"),
                "z": str(self._last_basin_state_calibration.z.policy if self._last_basin_state_calibration is not None else "abstain"),
                "yaw": str(self._last_basin_state_calibration.yaw.policy if self._last_basin_state_calibration is not None else "abstain"),
            },
            "basin_axis_source": str(estimated_basin_error.source if estimated_basin_error is not None else "none"),
            "basin_frame_consistency": float(estimated_basin_error.frame_consistency if estimated_basin_error is not None else 0.0),
            "basin_pullback_gate_ready": bool(gate_info.get("pullback_gate_ready", False)),
            "basin_pullback_block_reason": str(gate_info.get("pullback_block_reason", "not_evaluated")),
            "basin_micro_entry_ready": bool(gate_info.get("micro_entry_ready", False)),
            "basin_micro_entry_block_reason": str(gate_info.get("micro_entry_block_reason", "not_evaluated")),
            "basin_entry_gate_ready": bool(gate_info.get("micro_entry_ready", False)),
            "basin_close_ready": bool(strict_alignment_readiness.alignment_ready_for_handoff),
            "basin_close_ready_diagnostic_legacy": bool(legacy_basin_close_ready),
            "basin_close_block_reason": str(strict_alignment_readiness.block_reason),
            "alignment_ready_for_handoff": bool(strict_alignment_readiness.alignment_ready_for_handoff),
            "alignment_handoff_block_reason": str(strict_alignment_readiness.block_reason),
            "alignment_xy_ready": bool(strict_alignment_readiness.xy_ready),
            "alignment_z_ready": bool(strict_alignment_readiness.z_ready),
            "alignment_yaw_ready": bool(strict_alignment_readiness.yaw_ready),
            "alignment_observability_ready": bool(strict_alignment_readiness.observability_ready),
            "alignment_frame_consistency_ready": bool(strict_alignment_readiness.frame_consistency_ready),
            "runtime_takeover_contract": _jsonable_value(runtime_contract_decision),
            "runtime_takeover_tier": str(runtime_contract_decision.get("takeover_tier", "invalid")),
            "runtime_takeover_contract_source": str(runtime_contract_decision.get("contract_source", "unknown")),
            "runtime_gate_contract_pullback_consistent": bool(
                bool(gate_info.get("pullback_gate_ready", False)) == bool(runtime_contract_decision.get("pullback_allowed", False))
            ),
            "runtime_gate_contract_micro_consistent": bool(
                bool(gate_info.get("micro_entry_ready", False)) == bool(runtime_contract_decision.get("micro_entry_ready", False))
            ),
            "runtime_gate_contract_close_consistent": bool(
                bool(strict_alignment_readiness.alignment_ready_for_handoff) == bool(runtime_contract_decision.get("close_ready_ready", False))
            ),
            "phase_owner": stage.owner if stage_apply_allowed else "planner",
            "phase_reason": stage.transition_on if stage_apply_allowed else str(gate_info.get("reason", "waiting_precision_gate")),
            "invalid_action_flag": invalid_action_flag,
            "planner_action_world": planner_delta_7d[:6].tolist(),
            "planner_chunk_local_6d": local_base.astype(np.float32).tolist(),
            "local_command_local_6d": local_out.astype(np.float32).tolist(),
            "local_residual_vs_planner_local_6d": (local_out - local_base).astype(np.float32).tolist(),
            "pre_clip_action_world_6d": world_out[:6].tolist(),
            "post_clip_action_world_6d": final_action[:6].tolist(),
            "executed_action_world_6d": final_action[:6].tolist(),
            "local_correction_local_6d": (local_out - local_base).astype(np.float32).tolist(),
            "local_correction_owner": stage.owner if stage_apply_allowed else "planner",
            "grasp_gripper_override": None if grasp_ctrl is None else grasp_ctrl.gripper_override,
            "slide_gripper_override": None if slide_ctrl is None else slide_ctrl.gripper_override,
            "grasp_contact_rule_table": grasp_close_trace["rules"],
            "grasp_contact_rule_decision": grasp_close_trace["decision"],
            "grasp_contact_rule_reason": grasp_close_trace["reason"],
            "grasp_contact_rule_force_norm": grasp_close_trace["force_norm"],
            "grasp_contact_rule_force_threshold": grasp_close_trace["force_threshold"],
            "grasp_contact_rule_gripper_override": grasp_close_trace["gripper_override"],
            "grasp_contact_rule_contact_confirmed": grasp_close_trace["contact_confirmed"],
            "grasp_contact_rule_stable": grasp_close_trace["stable"],
            "c2c_stage_age": int(self.stage_age),
            "c2c_gate_skill": str(gate_info.get("skill", "none")),
            "c2c_gate_skill_name": str(gate_info.get("skill_name", "none")),
            "c2c_gate_active": bool(gate_active),
            "c2c_gate_reason": str(gate_info.get("reason", "none")),
            "c2c_gate_stable_frames": int(gate_info.get("stable_frames", 0)),
            "c2c_gate_required_frames": int(gate_info.get("required_frames", 0)),
            "c2c_gate_nearfield_depth_m": float(gate_info.get("absolute_nearfield_depth", float("nan"))),
            "c2c_gate_nearfield_depth_max": float(gate_info.get("nearfield_depth_max", float("nan"))),
            "c2c_gate_target_xy_error": float(gate_info.get("target_xy_error", float("nan"))),
            "c2c_gate_target_xy_error_max": float(gate_info.get("target_xy_error_max", float("nan"))),
            "c2c_gate_localizer_visible": bool(gate_info.get("localizer_visible", False)),
            "c2c_gate_depth_nearfield": bool(gate_info.get("depth_nearfield", False)),
            "c2c_gate_target_nearfield": bool(gate_info.get("target_nearfield", False)),
            "c2c_gate_pullback_ready": bool(gate_info.get("pullback_gate_ready", False)),
            "c2c_gate_micro_entry_ready": bool(gate_info.get("micro_entry_ready", False)),
            "c2c_gate_close_ready": bool(gate_info.get("close_ready", False)),
            "c2c_gate_pullback_block_reason": str(gate_info.get("pullback_block_reason", "not_evaluated")),
            "c2c_gate_micro_entry_block_reason": str(gate_info.get("micro_entry_block_reason", "not_evaluated")),
            "c2c_gate_close_block_reason": str(gate_info.get("close_block_reason", "not_evaluated")),
            "c2c_gate_takeover_kind": str(gate_info.get("takeover_gate_kind", "none")),
            "planner_reaches_precontact": bool(
                estimated_basin_error is not None
                and estimated_basin_error.valid
                and estimated_basin_error.xy_valid
                and estimated_basin_error.confidence >= (
                    self._skill_for_type("precision_grasp").shadow_confidence if self._skill_for_type("precision_grasp") else 0.2
                )
            ),
            "planner_reaches_preinsert": bool(
                spoke_error.valid and spoke_error.confidence >= (self._skill_for_type("precision_align").shadow_confidence if self._skill_for_type("precision_align") else 0.2)
            ),
            "raw_wrench": raw_force.tolist(),
            "filtered_wrench": filtered_force.tolist(),
            "raw_wrench_6d": raw_force.tolist(),
            "filtered_wrench_6d": filtered_force.tolist(),
        }
        if basin_decision is not None:
            self._last_trace.update(basin_decision.to_trace())
            if basin_apply_allowed:
                self._last_trace["phase_owner"] = "basin_recovery"
                self._last_trace["phase_reason"] = basin_decision.reason
                self._last_trace["local_correction_owner"] = "basin_recovery"
                command_local = np.asarray(basin_decision.local_action_6d, dtype=np.float32).reshape(6)
                residual_local = (command_local - local_base).astype(np.float32)
                self._last_trace["local_command_local_6d"] = command_local.tolist()
                self._last_trace["local_residual_vs_planner_local_6d"] = residual_local.tolist()
                self._last_trace["local_correction_local_6d"] = residual_local.tolist()
                self._last_trace["force_skill_state"] = basin_decision.mode.value
        corr_vec = np.asarray(self._last_trace.get("local_residual_vs_planner_local_6d", self._last_trace.get("local_correction_local_6d", [0.0] * 6)), dtype=np.float32).reshape(-1)
        if corr_vec.size >= 6 and float(np.linalg.norm(corr_vec[:6])) <= 1.0e-9:
            self._last_trace["phase_owner"] = "planner"
            self._last_trace["local_correction_owner"] = "planner"
            if bool(self._last_trace.get("c2c_gate_active", False)):
                self._last_trace["phase_reason"] = f"{self._last_trace.get('c2c_gate_reason', 'gate_active')}+planner_passthrough"
        return final_action

    def get_last_trace(self) -> dict[str, Any]:
        return dict(self._last_trace)
