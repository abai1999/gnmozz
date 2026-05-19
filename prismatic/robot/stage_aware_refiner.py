"""
stage_aware_refiner.py

Generic stage-aware low-level refiner for frozen chunk planners.
"""

import json
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from prismatic.robot.residual_safety import ResidualSafety
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from prismatic.robot.stage_manager import ContactState, StageManager, StagePhase, StageSubgoal, StageTargetMode
from prismatic.robot.stage_target_provider import pose_delta_local_between, wrap_yaw_to_symmetry


class StageAwareRefiner:
    """Task-agnostic runtime refiner with stage-aware controller routing."""

    def __init__(
        self,
        mode: str = "full",
        alignment_controller=None,
        contact_controller=None,
        close_trigger_controller=None,
        workspace_min: Optional[list] = None,
        workspace_max: Optional[list] = None,
        max_residual_pos: float = 0.0025,
        max_residual_rot: float = 0.015,
        force_stop_threshold: float = 5.0,
        backoff_distance: float = 0.002,
        backoff_force_threshold: float = 3.0,
        learned_residual_scale: float = 0.25,
        invalid_cooldown_steps: int = 8,
        require_pregrasp_alignment_gate: bool = True,
        alignment_depth_threshold: float = 0.08,
        alignment_open_threshold: float = 0.5,
        alignment_close_command_threshold: float = 0.50,
        max_alignment_corrections_per_window: int = 5,
        require_close_intent_for_alignment: bool = True,
        enable_alignment_pose: bool = True,
        use_pose_alpha: bool = True,
        enable_outer_rescue: bool = True,
        outer_rescue_xy_scale: float = 2.0,
        outer_rescue_abs_z_scale: float = 2.0,
        outer_rescue_yaw_scale: float = 2.0,
        outer_rescue_min_xy: float = 0.05,
        outer_rescue_min_abs_z: float = 0.18,
        outer_rescue_min_yaw: float = 0.30,
        enable_alignment_close_veto: bool = True,
        close_veto_xy_threshold: float = 0.010,
        close_veto_abs_z_threshold: float = 0.020,
        close_veto_yaw_threshold: float = -1.0,
        close_veto_ready_streak_frames: int = 1,
        close_veto_settle_steps: int = 0,
        close_latch_enabled: bool = False,
        close_latch_steps: int = 24,
        alignment_takeover_until_close_ready: bool = True,
        require_close_intent_for_refine_band: bool = False,
        alignment_assist_xy_scale: float = 10.0,
        alignment_assist_abs_z_scale: float = 40.0,
        alignment_assist_yaw_scale: float = 4.0,
        alignment_assist_base_scale: float = 0.55,
        alignment_assist_close_block_base_scale: float = 0.25,
        alignment_takeover_xy_scale: float = 2.0,
        alignment_takeover_abs_z_scale: float = 6.0,
        alignment_takeover_yaw_scale: float = 1.5,
        alignment_takeover_motion_xy_threshold: float = 0.020,
        alignment_takeover_motion_abs_z_threshold: float = 0.030,
        alignment_takeover_close_block_xy_threshold: float = 0.025,
        alignment_takeover_close_block_abs_z_threshold: float = 0.040,
        alignment_zone_hysteresis: float = 1.15,
        alignment_candidate_hold_steps: int = 4,
        alignment_candidate_switch_margin: float = 0.35,
        alignment_action_lowpass_alpha: float = 0.65,
        enable_near_handoff_z_correction: bool = True,
        near_handoff_z_xy_multiplier: float = 1.6,
        near_handoff_z_yaw_multiplier: float = 1.2,
        near_handoff_z_gate_multiplier: float = 4.0,
        near_handoff_z_min_step: float = 0.0025,
        near_handoff_z_max_step: float = 0.0040,
        near_handoff_z_blend: float = 0.75,
        enable_final_handoff_polish: bool = True,
        final_handoff_polish_xy_multiplier: float = 2.0,
        final_handoff_polish_yaw_multiplier: float = 1.75,
        final_handoff_polish_z_multiplier: float = 3.0,
        final_handoff_polish_max_xy_step: float = 0.0045,
        final_handoff_polish_max_z_step: float = 0.0045,
        final_handoff_polish_max_yaw_step: float = 0.03,
        final_handoff_polish_scale_cap: float = 0.85,
        enable_handoff_xy_micro_polish: bool = False,
        handoff_xy_micro_stall_frames: int = 3,
        handoff_xy_micro_window_steps: int = 4,
        handoff_xy_micro_improve_eps: float = 0.0004,
        handoff_xy_micro_max_xy_step: float = 0.0025,
        handoff_xy_micro_scale_cap: float = 0.70,
        enable_alignment_physical_mask: bool = False,
        enable_alignment_low_conf_noop: bool = False,
        enable_handoff_yaw_priority: bool = True,
        enable_handoff_xy_priority: bool = False,
        handoff_xy_priority_z_multiplier: float = 1.25,
        handoff_xy_priority_yaw_multiplier: float = 1.25,
        handoff_xy_priority_gate_multiplier: float = 2.5,
        handoff_xy_priority_max_xy_step: float = 0.0025,
        skip_alignment_when_close_ready: bool = False,
        skip_alignment_ready_xy_threshold: float = -1.0,
        skip_alignment_ready_abs_z_threshold: float = -1.0,
        skip_alignment_ready_yaw_threshold: float = -1.0,
        enable_readiness_gripper: bool = False,
        enable_alignment_conditioned_fire: bool = False,
        alignment_fire_xy_threshold: float = 0.003,
        alignment_fire_abs_z_threshold: float = 0.055,
        alignment_fire_yaw_threshold: float = -1.0,
        readiness_close_threshold: float = 0.65,
        gripper_override_confidence: float = 0.60,
        fire_hysteresis_frames: int = 2,
        verify_contact_steps: int = 4,
        verify_force_threshold: float = 0.5,
        b1_group_shadow_gate_mode: str = "broad",
        b2_candidate_shadow_gate_mode: str = "broad",
        enable_b2_candidate_bounded_v0: bool = False,
        b2_candidate_apply_conf_threshold: float = 0.431,
        b2_candidate_apply_margin_threshold: float = 0.010,
        close_veto_runtime_geometry_fallback_for_bounded: bool = False,
        enable_bounded_auto_close_on_alignment: bool = False,
        bounded_auto_close_stable_frames: int = 1,
        bounded_auto_close_xy_threshold: float = -1.0,
        bounded_auto_close_abs_z_threshold: float = -1.0,
        bounded_auto_close_yaw_threshold: float = -1.0,
        enable_force_close_after_b2_eval: bool = False,
        enable_alignment_near_zone_gate: bool = False,
        alignment_near_zone_xy_threshold: float = 0.05,
        alignment_near_zone_z_threshold: float = 0.10,
        enable_v2_nearzone_assist: bool = False,
        v2_assist_scale_cap: float = 0.25,
        v2_assist_max_pos: float = 0.0025,
        v2_assist_max_rot: float = 0.015,
        v2_assist_only_when_gate_pass: bool = True,
        enable_v2_predictor_micro_assist_apply: bool = False,
        v2_predictor_micro_assist_scale_cap: float = 0.05,
        v2_predictor_micro_assist_max_pos: float = 0.00075,
        v2_predictor_micro_assist_max_rot: float = 0.0040,
        v2_predictor_micro_assist_only_when_gate_pass: bool = True,
        enable_target_delta_servo_shadow: bool = False,
        enable_target_delta_servo_apply: bool = False,
        target_delta_servo_bypass_gates: bool = False,
        target_delta_servo_apply_once_per_episode: bool = False,
        target_delta_servo_source: str = "predictor",
        target_delta_servo_k_xy: float = 0.08,
        target_delta_servo_k_z: float = 0.06,
        target_delta_servo_k_yaw: float = 0.04,
        target_delta_servo_max_pos: float = 0.0010,
        target_delta_servo_max_yaw: float = 0.0040,
        target_delta_servo_apply_xy_threshold: float = 0.03,
        target_delta_servo_apply_abs_z_threshold: float = 0.07,
        target_delta_servo_apply_yaw_threshold: float = 0.25,
        alignment_v3_shadow_controller=None,
        alignment_v4_shadow_controller=None,
        alignment_v4_shadow_micro_only: bool = False,
        alignment_diffusion_controller=None,
        alignment_tc_diffusion_controller=None,
        alignment_tc_student_vnext_controller=None,
        alignment_tc_student_vnext_collector_like: bool = False,
        enable_alignment_tc_student_vnext_ready_gate: bool = False,
        alignment_tc_student_vnext_close_ready_threshold: float = 0.5,
        alignment_tc_student_vnext_handoff_ready_threshold: float = 0.5,
        enable_alignment_diffusion_shadow: bool = False,
        enable_alignment_diffusion_apply: bool = False,
        alignment_diffusion_horizon: int = 8,
        alignment_diffusion_num_samples: int = 16,
        alignment_diffusion_apply_mode: str = "additive",
        alignment_diffusion_max_pos_step: float = 0.0015,
        alignment_diffusion_max_yaw_step: float = 0.0060,
        alignment_diffusion_risk_threshold: float = 0.65,
        alignment_diffusion_trigger_mode: str = "near_contact_stall",
        alignment_diffusion_execute_steps: int = 1,
        enable_alignment_tc_diffusion_shadow: bool = False,
        enable_alignment_tc_diffusion_apply: bool = False,
        alignment_tc_diffusion_num_samples: int = 8,
        alignment_tc_diffusion_top_k: int = 3,
        alignment_tc_diffusion_confidence_threshold: float = 0.55,
        alignment_tc_diffusion_risk_threshold: float = 0.65,
        alignment_tc_diffusion_soft_clamp: bool = False,
        alignment_tc_diffusion_workspace_tolerance: float = 0.0,
        alignment_tc_diffusion_workspace_soft_clamp: bool = False,
        alignment_tc_diffusion_execute_steps: int = 1,
        enable_phase1_bridge_support_soft_override: bool = True,
        enable_phase1_bridge_risk_soft_apply: bool = True,
        phase1_bridge_soft_apply_max_pos: float = 0.00055,
        phase1_bridge_soft_apply_max_yaw: float = 0.0018,
        phase1_bridge_workspace_project_max_pos: float = 0.0065,
        phase1_bridge_workspace_project_max_yaw: float = 0.0085,
        phase1_bridge_workspace_project_first: bool = True,
        phase1_bridge_planner_x_blend: float = 0.50,
        phase1_bridge_planner_z_blend: float = 0.35,
        phase1_bridge_planner_yaw_blend: float = 0.08,
        phase1_bridge_basin_bias_xy_scale: float = 0.36,
        phase1_bridge_basin_bias_z_scale: float = 0.28,
        phase1_bridge_basin_bias_yaw_scale: float = 0.07,
        phase1_bridge_basin_bias_yaw_direct_scale: float = 0.40,
        phase1_bridge_basin_bias_max_pos: float = 0.0025,
        phase1_bridge_basin_bias_max_yaw: float = 0.0050,
        phase1_bridge_basin_bias_yaw_direct_max: float = 0.0055,
        phase1_bridge_yaw_holdoff_xy_threshold: float = 0.009,
        phase1_bridge_yaw_holdoff_abs_z_threshold: float = 0.010,
        phase1_bridge_yaw_holdoff_blend_scale: float = 0.07,
        enable_phase1_force_reflex: bool = False,
        phase1_force_contact_fz_threshold: float = 0.35,
        phase1_force_force_norm_threshold: float = 0.50,
        phase1_force_high_fz_threshold: float = 2.5,
        phase1_force_lateral_threshold: float = 1.5,
        phase1_force_torque_threshold: float = 1.0,
        phase1_force_spike_threshold: float = 0.75,
        phase1_force_backoff_mm: float = 0.0015,
        phase1_close_confirm_steps: int = 2,
        phase1_close_fail_steps: int = 3,
        phase1_post_contact_hold_steps: int = 8,
        phase1_reopen_cooldown_steps: int = 4,
        enable_alignment_tc_student_vnext_shadow: bool = False,
        enable_alignment_tc_student_vnext_apply: bool = False,
        alignment_tc_student_vnext_corridor_json: Optional[str] = None,
        enable_alignment_v4_apply: bool = False,
        alignment_v4_apply_scale: float = 1.0,
        alignment_v4_apply_max_pos: float = 0.0010,
        alignment_v4_apply_max_yaw: float = 0.0040,
        alignment_v4_apply_only_when_gate_pass: bool = True,
        alignment_v4_apply_require_improve: bool = False,
        alignment_v4_apply_micro_only: bool = False,
        enable_alignment_v3_apply: bool = False,
        alignment_v3_apply_scale: float = 1.0,
        alignment_v3_apply_max_pos: float = 0.0020,
        alignment_v3_apply_max_yaw: float = 0.0020,
        alignment_v3_apply_only_when_gate_pass: bool = True,
        alignment_v3_apply_require_improve: bool = False,
        stage_manager: Optional[StageManager] = None,
    ):
        assert mode in ("planner_only", "safety_only", "alignment", "contact", "full"), f"Unknown mode: {mode}"
        self.mode = mode
        self.alignment_controller = alignment_controller
        self.contact_controller = contact_controller
        self.close_trigger_controller = close_trigger_controller
        self.manager = stage_manager if stage_manager is not None else StageManager()
        self.safety = ResidualSafety(
            max_residual_pos=max_residual_pos,
            max_residual_rot=max_residual_rot,
            force_stop_threshold=force_stop_threshold,
            backoff_distance=backoff_distance,
            backoff_force_threshold=backoff_force_threshold,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
        )
        self._correction_count = 0
        self._replan_count = 0
        self._alpha_sum = 0.0
        self._residual_norm_sum = 0.0
        self._alignment_count = 0
        self._contact_count = 0
        self.learned_residual_scale = learned_residual_scale
        self.invalid_cooldown_steps = invalid_cooldown_steps
        self._residual_cooldown = 0
        self.require_pregrasp_alignment_gate = require_pregrasp_alignment_gate
        self.alignment_depth_threshold = alignment_depth_threshold
        self.alignment_open_threshold = alignment_open_threshold
        self.alignment_close_command_threshold = alignment_close_command_threshold
        self.max_alignment_corrections_per_window = max_alignment_corrections_per_window
        self.require_close_intent_for_alignment = require_close_intent_for_alignment
        self.enable_alignment_pose = enable_alignment_pose
        self.use_pose_alpha = use_pose_alpha
        self.enable_outer_rescue = bool(enable_outer_rescue)
        self.outer_rescue_xy_scale = float(outer_rescue_xy_scale)
        self.outer_rescue_abs_z_scale = float(outer_rescue_abs_z_scale)
        self.outer_rescue_yaw_scale = float(outer_rescue_yaw_scale)
        self.outer_rescue_min_xy = float(outer_rescue_min_xy)
        self.outer_rescue_min_abs_z = float(outer_rescue_min_abs_z)
        self.outer_rescue_min_yaw = float(outer_rescue_min_yaw)
        self.enable_alignment_close_veto = bool(enable_alignment_close_veto)
        self.close_veto_xy_threshold = float(close_veto_xy_threshold)
        self.close_veto_abs_z_threshold = float(close_veto_abs_z_threshold)
        self.close_veto_yaw_threshold = float(close_veto_yaw_threshold)
        self.close_veto_ready_streak_frames = max(int(close_veto_ready_streak_frames), 1)
        self.close_veto_settle_steps = max(int(close_veto_settle_steps), 0)
        self.close_latch_enabled = bool(close_latch_enabled)
        self.close_latch_steps = max(int(close_latch_steps), 0)
        self.alignment_takeover_until_close_ready = bool(alignment_takeover_until_close_ready)
        self.require_close_intent_for_refine_band = bool(require_close_intent_for_refine_band)
        self.alignment_assist_xy_scale = float(alignment_assist_xy_scale)
        self.alignment_assist_abs_z_scale = float(alignment_assist_abs_z_scale)
        self.alignment_assist_yaw_scale = float(alignment_assist_yaw_scale)
        self.alignment_assist_base_scale = float(np.clip(alignment_assist_base_scale, 0.0, 1.0))
        self.alignment_assist_close_block_base_scale = float(np.clip(alignment_assist_close_block_base_scale, 0.0, 1.0))
        self.alignment_takeover_xy_scale = float(alignment_takeover_xy_scale)
        self.alignment_takeover_abs_z_scale = float(alignment_takeover_abs_z_scale)
        self.alignment_takeover_yaw_scale = float(alignment_takeover_yaw_scale)
        self.alignment_takeover_motion_xy_threshold = float(alignment_takeover_motion_xy_threshold)
        self.alignment_takeover_motion_abs_z_threshold = float(alignment_takeover_motion_abs_z_threshold)
        self.alignment_takeover_close_block_xy_threshold = float(alignment_takeover_close_block_xy_threshold)
        self.alignment_takeover_close_block_abs_z_threshold = float(alignment_takeover_close_block_abs_z_threshold)
        self.alignment_zone_hysteresis = max(float(alignment_zone_hysteresis), 1.0)
        self.alignment_candidate_hold_steps = max(int(alignment_candidate_hold_steps), 1)
        self.alignment_candidate_switch_margin = float(alignment_candidate_switch_margin)
        self.alignment_action_lowpass_alpha = float(np.clip(alignment_action_lowpass_alpha, 0.0, 0.999))
        self.enable_near_handoff_z_correction = bool(enable_near_handoff_z_correction)
        self.near_handoff_z_xy_multiplier = float(max(near_handoff_z_xy_multiplier, 1.0))
        self.near_handoff_z_yaw_multiplier = float(max(near_handoff_z_yaw_multiplier, 1.0))
        self.near_handoff_z_gate_multiplier = float(max(near_handoff_z_gate_multiplier, 1.0))
        self.near_handoff_z_min_step = float(max(near_handoff_z_min_step, 0.0))
        self.near_handoff_z_max_step = float(max(near_handoff_z_max_step, self.near_handoff_z_min_step))
        self.near_handoff_z_blend = float(np.clip(near_handoff_z_blend, 0.0, 1.0))
        self._near_handoff_z_correction_count = 0
        self.enable_final_handoff_polish = bool(enable_final_handoff_polish)
        self.final_handoff_polish_xy_multiplier = float(max(final_handoff_polish_xy_multiplier, 1.0))
        self.final_handoff_polish_yaw_multiplier = float(max(final_handoff_polish_yaw_multiplier, 1.0))
        self.final_handoff_polish_z_multiplier = float(max(final_handoff_polish_z_multiplier, 1.0))
        self.final_handoff_polish_max_xy_step = float(max(final_handoff_polish_max_xy_step, 0.0))
        self.final_handoff_polish_max_z_step = float(max(final_handoff_polish_max_z_step, 0.0))
        self.final_handoff_polish_max_yaw_step = float(max(final_handoff_polish_max_yaw_step, 0.0))
        self.final_handoff_polish_scale_cap = float(np.clip(final_handoff_polish_scale_cap, 0.05, 1.0))
        self._final_handoff_polish_count = 0
        self.enable_handoff_xy_micro_polish = bool(enable_handoff_xy_micro_polish)
        self.handoff_xy_micro_stall_frames = max(int(handoff_xy_micro_stall_frames), 1)
        self.handoff_xy_micro_window_steps = max(int(handoff_xy_micro_window_steps), 1)
        self.handoff_xy_micro_improve_eps = float(max(handoff_xy_micro_improve_eps, 0.0))
        self.handoff_xy_micro_max_xy_step = float(max(handoff_xy_micro_max_xy_step, 1e-4))
        self.handoff_xy_micro_scale_cap = float(np.clip(handoff_xy_micro_scale_cap, 0.05, 1.0))
        self._handoff_xy_micro_stall_streak = 0
        self._handoff_xy_micro_prev_xy = np.inf
        self._handoff_xy_micro_remaining = 0
        self._handoff_xy_micro_trigger_count = 0
        self.enable_alignment_physical_mask = bool(enable_alignment_physical_mask)
        self.enable_alignment_low_conf_noop = bool(enable_alignment_low_conf_noop)
        self.enable_handoff_yaw_priority = bool(enable_handoff_yaw_priority)
        self.enable_handoff_xy_priority = bool(enable_handoff_xy_priority)
        self.handoff_xy_priority_z_multiplier = float(max(handoff_xy_priority_z_multiplier, 1.0))
        self.handoff_xy_priority_yaw_multiplier = float(max(handoff_xy_priority_yaw_multiplier, 1.0))
        self.handoff_xy_priority_gate_multiplier = float(max(handoff_xy_priority_gate_multiplier, 1.0))
        self.handoff_xy_priority_max_xy_step = float(max(handoff_xy_priority_max_xy_step, 1e-4))
        self.skip_alignment_when_close_ready = bool(skip_alignment_when_close_ready)
        self.skip_alignment_ready_xy_threshold = float(skip_alignment_ready_xy_threshold)
        self.skip_alignment_ready_abs_z_threshold = float(skip_alignment_ready_abs_z_threshold)
        self.skip_alignment_ready_yaw_threshold = float(skip_alignment_ready_yaw_threshold)
        self.enable_readiness_gripper = enable_readiness_gripper
        self.enable_alignment_conditioned_fire = bool(enable_alignment_conditioned_fire)
        self.alignment_fire_xy_threshold = float(alignment_fire_xy_threshold)
        self.alignment_fire_abs_z_threshold = float(alignment_fire_abs_z_threshold)
        self.alignment_fire_yaw_threshold = float(alignment_fire_yaw_threshold)
        self.readiness_close_threshold = float(readiness_close_threshold)
        self.gripper_override_confidence = float(gripper_override_confidence)
        self.fire_hysteresis_frames = int(fire_hysteresis_frames)
        self.verify_contact_steps = int(verify_contact_steps)
        self.verify_force_threshold = float(verify_force_threshold)
        if b1_group_shadow_gate_mode not in ("broad", "close_only"):
            raise ValueError(f"Unknown b1_group_shadow_gate_mode={b1_group_shadow_gate_mode}")
        self.b1_group_shadow_gate_mode = str(b1_group_shadow_gate_mode)
        if b2_candidate_shadow_gate_mode not in ("broad", "close_only", "nearish_only"):
            raise ValueError(f"Unknown b2_candidate_shadow_gate_mode={b2_candidate_shadow_gate_mode}")
        self.b2_candidate_shadow_gate_mode = str(b2_candidate_shadow_gate_mode)
        self.enable_b2_candidate_bounded_v0 = bool(enable_b2_candidate_bounded_v0)
        self.b2_candidate_apply_conf_threshold = float(np.clip(b2_candidate_apply_conf_threshold, 0.0, 1.0))
        self.b2_candidate_apply_margin_threshold = float(max(b2_candidate_apply_margin_threshold, 0.0))
        self.close_veto_runtime_geometry_fallback_for_bounded = bool(close_veto_runtime_geometry_fallback_for_bounded)
        self.enable_bounded_auto_close_on_alignment = bool(enable_bounded_auto_close_on_alignment)
        self.bounded_auto_close_stable_frames = max(int(bounded_auto_close_stable_frames), 1)
        self.bounded_auto_close_xy_threshold = float(bounded_auto_close_xy_threshold)
        self.bounded_auto_close_abs_z_threshold = float(bounded_auto_close_abs_z_threshold)
        self.bounded_auto_close_yaw_threshold = float(bounded_auto_close_yaw_threshold)
        self.enable_force_close_after_b2_eval = bool(enable_force_close_after_b2_eval)
        self.enable_alignment_near_zone_gate = bool(enable_alignment_near_zone_gate)
        self.alignment_near_zone_xy_threshold = float(alignment_near_zone_xy_threshold)
        self.alignment_near_zone_z_threshold = float(alignment_near_zone_z_threshold)
        self.enable_v2_nearzone_assist = bool(enable_v2_nearzone_assist)
        self.v2_assist_scale_cap = float(v2_assist_scale_cap)
        self.v2_assist_max_pos = float(v2_assist_max_pos)
        self.v2_assist_max_rot = float(v2_assist_max_rot)
        self.v2_assist_only_when_gate_pass = bool(v2_assist_only_when_gate_pass)
        self.enable_v2_predictor_micro_assist_apply = bool(enable_v2_predictor_micro_assist_apply)
        self.v2_predictor_micro_assist_scale_cap = float(v2_predictor_micro_assist_scale_cap)
        self.v2_predictor_micro_assist_max_pos = float(v2_predictor_micro_assist_max_pos)
        self.v2_predictor_micro_assist_max_rot = float(v2_predictor_micro_assist_max_rot)
        self.v2_predictor_micro_assist_only_when_gate_pass = bool(v2_predictor_micro_assist_only_when_gate_pass)
        self.enable_target_delta_servo_shadow = bool(enable_target_delta_servo_shadow)
        self.enable_target_delta_servo_apply = bool(enable_target_delta_servo_apply)
        self.target_delta_servo_bypass_gates = bool(target_delta_servo_bypass_gates)
        self.target_delta_servo_apply_once_per_episode = bool(target_delta_servo_apply_once_per_episode)
        self.target_delta_servo_source = str(target_delta_servo_source)
        self.target_delta_servo_k_xy = float(target_delta_servo_k_xy)
        self.target_delta_servo_k_z = float(target_delta_servo_k_z)
        self.target_delta_servo_k_yaw = float(target_delta_servo_k_yaw)
        self.target_delta_servo_max_pos = float(target_delta_servo_max_pos)
        self.target_delta_servo_max_yaw = float(target_delta_servo_max_yaw)
        self.target_delta_servo_apply_xy_threshold = float(target_delta_servo_apply_xy_threshold)
        self.target_delta_servo_apply_abs_z_threshold = float(target_delta_servo_apply_abs_z_threshold)
        self.target_delta_servo_apply_yaw_threshold = float(target_delta_servo_apply_yaw_threshold)
        self.alignment_v3_shadow_controller = alignment_v3_shadow_controller
        self.alignment_v4_shadow_controller = alignment_v4_shadow_controller
        self.alignment_v4_shadow_micro_only = bool(alignment_v4_shadow_micro_only)
        self.alignment_diffusion_controller = alignment_diffusion_controller
        self.alignment_tc_diffusion_controller = alignment_tc_diffusion_controller
        self.alignment_tc_student_vnext_controller = alignment_tc_student_vnext_controller
        self.alignment_tc_student_vnext_collector_like = bool(alignment_tc_student_vnext_collector_like)
        self.enable_alignment_tc_student_vnext_ready_gate = bool(enable_alignment_tc_student_vnext_ready_gate)
        self.alignment_tc_student_vnext_close_ready_threshold = float(
            alignment_tc_student_vnext_close_ready_threshold
        )
        self.alignment_tc_student_vnext_handoff_ready_threshold = float(
            alignment_tc_student_vnext_handoff_ready_threshold
        )
        self.enable_alignment_diffusion_shadow = bool(enable_alignment_diffusion_shadow)
        self.enable_alignment_diffusion_apply = bool(enable_alignment_diffusion_apply)
        self.alignment_diffusion_horizon = max(int(alignment_diffusion_horizon), 1)
        self.alignment_diffusion_num_samples = max(int(alignment_diffusion_num_samples), 1)
        if alignment_diffusion_apply_mode not in ("additive", "blend", "rewrite_micro"):
            raise ValueError(f"Unknown alignment_diffusion_apply_mode={alignment_diffusion_apply_mode}")
        self.alignment_diffusion_apply_mode = str(alignment_diffusion_apply_mode)
        self.alignment_diffusion_max_pos_step = float(max(alignment_diffusion_max_pos_step, 0.0))
        self.alignment_diffusion_max_yaw_step = float(max(alignment_diffusion_max_yaw_step, 0.0))
        self.alignment_diffusion_risk_threshold = float(np.clip(alignment_diffusion_risk_threshold, 0.0, 1.0))
        if alignment_diffusion_trigger_mode not in ("near_contact_stall", "align_phase", "depth_only"):
            raise ValueError(f"Unknown alignment_diffusion_trigger_mode={alignment_diffusion_trigger_mode}")
        self.alignment_diffusion_trigger_mode = str(alignment_diffusion_trigger_mode)
        self.alignment_diffusion_execute_steps = max(int(alignment_diffusion_execute_steps), 1)
        self.enable_alignment_tc_diffusion_shadow = bool(enable_alignment_tc_diffusion_shadow)
        self.enable_alignment_tc_diffusion_apply = bool(enable_alignment_tc_diffusion_apply)
        self.alignment_tc_diffusion_num_samples = max(int(alignment_tc_diffusion_num_samples), 1)
        self.alignment_tc_diffusion_top_k = max(int(alignment_tc_diffusion_top_k), 1)
        self.alignment_tc_diffusion_confidence_threshold = float(
            np.clip(alignment_tc_diffusion_confidence_threshold, 0.0, 1.0)
        )
        self.alignment_tc_diffusion_risk_threshold = float(np.clip(alignment_tc_diffusion_risk_threshold, 0.0, 1.0))
        self.alignment_tc_diffusion_soft_clamp = bool(alignment_tc_diffusion_soft_clamp)
        self.alignment_tc_diffusion_workspace_tolerance = float(max(alignment_tc_diffusion_workspace_tolerance, 0.0))
        self.alignment_tc_diffusion_workspace_soft_clamp = bool(alignment_tc_diffusion_workspace_soft_clamp)
        self.alignment_tc_diffusion_execute_steps = max(int(alignment_tc_diffusion_execute_steps), 1)
        self.enable_phase1_bridge_support_soft_override = bool(enable_phase1_bridge_support_soft_override)
        self.enable_phase1_bridge_risk_soft_apply = bool(enable_phase1_bridge_risk_soft_apply)
        self.phase1_bridge_soft_apply_max_pos = float(max(phase1_bridge_soft_apply_max_pos, 0.0))
        self.phase1_bridge_soft_apply_max_yaw = float(max(phase1_bridge_soft_apply_max_yaw, 0.0))
        self.phase1_bridge_workspace_project_max_pos = float(max(phase1_bridge_workspace_project_max_pos, 0.0))
        self.phase1_bridge_workspace_project_max_yaw = float(max(phase1_bridge_workspace_project_max_yaw, 0.0))
        self.phase1_bridge_workspace_project_first = bool(phase1_bridge_workspace_project_first)
        self.phase1_bridge_planner_x_blend = float(np.clip(phase1_bridge_planner_x_blend, 0.0, 1.0))
        self.phase1_bridge_planner_z_blend = float(np.clip(phase1_bridge_planner_z_blend, 0.0, 1.0))
        self.phase1_bridge_planner_yaw_blend = float(np.clip(phase1_bridge_planner_yaw_blend, 0.0, 1.0))
        self.phase1_bridge_basin_bias_xy_scale = float(max(phase1_bridge_basin_bias_xy_scale, 0.0))
        self.phase1_bridge_basin_bias_z_scale = float(max(phase1_bridge_basin_bias_z_scale, 0.0))
        self.phase1_bridge_basin_bias_yaw_scale = float(max(phase1_bridge_basin_bias_yaw_scale, 0.0))
        self.phase1_bridge_basin_bias_yaw_direct_scale = float(max(phase1_bridge_basin_bias_yaw_direct_scale, 0.0))
        self.phase1_bridge_basin_bias_max_pos = float(max(phase1_bridge_basin_bias_max_pos, 0.0))
        self.phase1_bridge_basin_bias_max_yaw = float(max(phase1_bridge_basin_bias_max_yaw, 0.0))
        self.phase1_bridge_basin_bias_yaw_direct_max = float(max(phase1_bridge_basin_bias_yaw_direct_max, 0.0))
        self.phase1_bridge_yaw_holdoff_xy_threshold = float(max(phase1_bridge_yaw_holdoff_xy_threshold, 0.0))
        self.phase1_bridge_yaw_holdoff_abs_z_threshold = float(max(phase1_bridge_yaw_holdoff_abs_z_threshold, 0.0))
        self.phase1_bridge_yaw_holdoff_blend_scale = float(np.clip(phase1_bridge_yaw_holdoff_blend_scale, 0.0, 1.0))
        self.enable_phase1_force_reflex = bool(enable_phase1_force_reflex)
        self.phase1_force_contact_fz_threshold = float(max(phase1_force_contact_fz_threshold, 0.0))
        self.phase1_force_force_norm_threshold = float(max(phase1_force_force_norm_threshold, 0.0))
        self.phase1_force_high_fz_threshold = float(max(phase1_force_high_fz_threshold, 0.0))
        self.phase1_force_lateral_threshold = float(max(phase1_force_lateral_threshold, 0.0))
        self.phase1_force_torque_threshold = float(max(phase1_force_torque_threshold, 0.0))
        self.phase1_force_spike_threshold = float(max(phase1_force_spike_threshold, 0.0))
        self.phase1_force_backoff_mm = float(max(phase1_force_backoff_mm, 0.0))
        self.phase1_close_confirm_steps = max(int(phase1_close_confirm_steps), 1)
        self.phase1_close_fail_steps = max(int(phase1_close_fail_steps), 1)
        self.phase1_post_contact_hold_steps = max(int(phase1_post_contact_hold_steps), 1)
        self.phase1_reopen_cooldown_steps = max(int(phase1_reopen_cooldown_steps), 1)
        self.enable_alignment_tc_student_vnext_shadow = bool(enable_alignment_tc_student_vnext_shadow)
        self.enable_alignment_tc_student_vnext_apply = bool(enable_alignment_tc_student_vnext_apply)
        self.alignment_tc_student_vnext_corridor_json = alignment_tc_student_vnext_corridor_json
        self._alignment_tc_student_vnext_corridor = None
        if alignment_tc_student_vnext_corridor_json:
            try:
                with open(alignment_tc_student_vnext_corridor_json, "r", encoding="utf-8") as f:
                    self._alignment_tc_student_vnext_corridor = json.load(f).get("runtime_corridor", None)
            except Exception:
                self._alignment_tc_student_vnext_corridor = None
        self.enable_alignment_v4_apply = bool(enable_alignment_v4_apply)
        self.alignment_v4_apply_scale = float(alignment_v4_apply_scale)
        self.alignment_v4_apply_max_pos = float(alignment_v4_apply_max_pos)
        self.alignment_v4_apply_max_yaw = float(alignment_v4_apply_max_yaw)
        self.alignment_v4_apply_only_when_gate_pass = bool(alignment_v4_apply_only_when_gate_pass)
        self.alignment_v4_apply_require_improve = bool(alignment_v4_apply_require_improve)
        self.alignment_v4_apply_micro_only = bool(alignment_v4_apply_micro_only)
        self.enable_alignment_v3_apply = bool(enable_alignment_v3_apply)
        self.alignment_v3_apply_scale = float(alignment_v3_apply_scale)
        self.alignment_v3_apply_max_pos = float(alignment_v3_apply_max_pos)
        self.alignment_v3_apply_max_yaw = float(alignment_v3_apply_max_yaw)
        self.alignment_v3_apply_only_when_gate_pass = bool(alignment_v3_apply_only_when_gate_pass)
        self.alignment_v3_apply_require_improve = bool(alignment_v3_apply_require_improve)
        self._alignment_v4_shadow_eval_count = 0
        self._alignment_v4_shadow_improve_count = 0
        self._alignment_v4_shadow_all_improve_count = 0
        self._last_alignment_v4_shadow_active = False
        self._last_alignment_v4_shadow_source = "none"
        self._last_alignment_v4_shadow_block_reason = "disabled"
        self._last_alignment_v4_shadow_cur_xy = None
        self._last_alignment_v4_shadow_cur_z = None
        self._last_alignment_v4_shadow_cur_yaw = None
        self._last_alignment_v4_shadow_post_xy = None
        self._last_alignment_v4_shadow_post_z = None
        self._last_alignment_v4_shadow_post_yaw = None
        self._last_alignment_v4_shadow_xy_improved = False
        self._last_alignment_v4_shadow_z_improved = False
        self._last_alignment_v4_shadow_yaw_improved = False
        self._last_alignment_v4_shadow_all_improved = False
        self._last_alignment_v4_shadow_pred_residual_4d = None
        self._last_alignment_v4_shadow_pred_residual_6d = None
        self._last_alignment_v4_shadow_pred_post_xyz_yaw = None
        self._last_alignment_v4_shadow_pred_reduction_xyz = None
        self._last_alignment_v4_shadow_pred_pos_norm = 0.0
        self._last_alignment_v4_shadow_pred_yaw_abs = 0.0
        self._last_alignment_v4_shadow_risk_logit = 0.0
        self._last_alignment_v4_shadow_confidence_logit = 0.0
        self._last_alignment_v4_shadow_policy_mode = "unknown"
        self._last_alignment_v4_shadow_stage_bucket = "unknown"
        self._last_alignment_v4_shadow_micro_gate = False
        self._alignment_v4_apply_count = 0
        self._last_alignment_v4_apply_applied = False
        self._last_alignment_v4_apply_block_reason = "disabled"
        self._last_alignment_v4_apply_local_delta = None
        self._last_alignment_v4_apply_world_delta = None
        self._last_alignment_v4_apply_pos_norm = 0.0
        self._last_alignment_v4_apply_yaw_abs = 0.0
        self._last_alignment_v4_apply_stage_bucket = "unknown"
        self._last_alignment_v4_apply_micro_gate = False
        self._alignment_diffusion_eval_count = 0
        self._alignment_diffusion_active_count = 0
        self._alignment_diffusion_apply_count = 0
        self._alignment_diffusion_block_reason_hist = {}
        self._alignment_diffusion_phase_hist = {}
        self._alignment_diffusion_bucket_hist = {}
        self._alignment_diffusion_controller_type_hist = {}
        self._alignment_diffusion_confidence_sum = 0.0
        self._alignment_diffusion_risk_prob_sum = 0.0
        self._alignment_diffusion_stop_prob_sum = 0.0
        self._alignment_diffusion_scale_down_sum = 0.0
        self._alignment_diffusion_pos_norm_sum = 0.0
        self._alignment_diffusion_yaw_abs_sum = 0.0
        self._alignment_diffusion_pred_target_delta_norm_sum = 0.0
        self._alignment_diffusion_pred_target_yaw_abs_sum = 0.0
        self._alignment_diffusion_target_action_sign_agreement_sum = 0.0
        self._alignment_diffusion_low_confidence_count = 0
        self._alignment_diffusion_soft_clamp_count = 0
        self._alignment_diffusion_hard_reject_count = 0
        self._alignment_diffusion_safety_reject_count = 0
        self._last_alignment_diffusion_enabled = False
        self._last_alignment_diffusion_active = False
        self._last_alignment_diffusion_applied = False
        self._last_alignment_diffusion_block_reason = "disabled"
        self._last_alignment_diffusion_trigger_mode = str(self.alignment_diffusion_trigger_mode)
        self._last_alignment_diffusion_phase_name = "unknown"
        self._last_alignment_diffusion_stage_bucket = "unknown"
        self._last_alignment_diffusion_safety_reject = False
        self._last_alignment_diffusion_selected_index = -1
        self._last_alignment_diffusion_num_samples = int(self.alignment_diffusion_num_samples)
        self._last_alignment_diffusion_candidate_diversity = 0.0
        self._last_alignment_diffusion_candidate_score = 0.0
        self._last_alignment_diffusion_risk_prob = 0.0
        self._last_alignment_diffusion_stop_prob = 0.0
        self._last_alignment_diffusion_progress_logits = None
        self._last_alignment_diffusion_first_residual_4d = None
        self._last_alignment_diffusion_first_residual_6d = None
        self._last_alignment_diffusion_local_delta = None
        self._last_alignment_diffusion_world_delta = None
        self._last_alignment_diffusion_pos_norm = 0.0
        self._last_alignment_diffusion_yaw_abs = 0.0
        self._last_alignment_diffusion_workspace_violation = 0.0
        self._last_alignment_diffusion_workspace_projected = False
        self._last_alignment_diffusion_workspace_project_reason = "none"
        self._last_alignment_diffusion_phase1_bridge_blend_active = False
        self._last_alignment_diffusion_phase1_bridge_blend_reason = "none"
        self._last_alignment_diffusion_phase1_bridge_blended_base_world = None
        self._last_alignment_diffusion_phase1_bridge_blended_base_local = None
        self._last_alignment_diffusion_phase1_bridge_basin_bias_local = None
        self._last_alignment_diffusion_phase1_bridge_taskspace_yaw_error = 0.0
        self._last_alignment_diffusion_phase1_bridge_taskspace_yaw_target_source = "none"
        self._last_alignment_diffusion_phase1_bridge_direct_yaw_rescue_local = None
        self._last_alignment_diffusion_phase1_bridge_yaw_holdoff_active = False
        self._last_alignment_diffusion_phase1_bridge_yaw_holdoff_reason = "none"
        self._phase1_close_arbiter_state = "APPROACH"
        self._phase1_close_command_source = "none"
        self._phase1_grasp_contact_confirmed = False
        self._phase1_force_reflex_active = False
        self._phase1_force_reflex_reason = "none"
        self._phase1_force_backoff_applied = False
        self._phase1_reopen_reason = "none"
        self._phase1_close_hold_active = False
        self._phase1_close_hold_remaining = 0
        self._phase1_close_confirm_streak = 0
        self._phase1_close_fail_remaining = 0
        self._phase1_reopen_cooldown_remaining = 0
        self._phase1_force_spike_count = 0
        self._phase1_jam_detected_count = 0
        self._phase1_grasp_contact_confirmed_count = 0
        self._phase1_reopen_count = 0
        self._phase1_force_reflex_activation_count = 0
        self._phase1_force_prev_norm = None
        self._phase1_prev_basin_metrics = None
        self._last_alignment_diffusion_controller_type = "none"
        self._last_alignment_diffusion_target_confidence = 0.0
        self._last_alignment_diffusion_pred_target_delta_6d = None
        self._last_alignment_diffusion_pred_target_delta_norm = 0.0
        self._last_alignment_diffusion_pred_target_yaw_abs = 0.0
        self._last_alignment_diffusion_target_action_sign_agreement = 0.0
        self._last_alignment_diffusion_low_confidence = False
        self._last_alignment_diffusion_soft_clamp = False
        self._last_alignment_diffusion_scale_down = 1.0
        self._last_alignment_diffusion_hard_reject = False
        self._last_alignment_diffusion_phase1_bridge_soft_apply = False
        self._last_alignment_diffusion_phase1_bridge_soft_apply_reason = "none"
        self._last_alignment_diffusion_latency_total_ms = 0.0
        self._last_alignment_diffusion_top_k = int(self.alignment_tc_diffusion_top_k)
        self._v2_assist_apply_count = 0
        self._last_v2_assist_applied = False
        self._v2_predictor_micro_assist_eval_count = 0
        self._v2_predictor_micro_assist_apply_count = 0
        self._last_v2_predictor_micro_assist_applied = False
        self._last_v2_predictor_micro_assist_block_reason = "disabled"
        self._last_v2_predictor_micro_assist_pos_norm = 0.0
        self._last_v2_predictor_micro_assist_rot_norm = 0.0
        self._last_v2_predictor_micro_assist_source = "none"
        self._target_delta_servo_eval_count = 0
        self._target_delta_servo_apply_count = 0
        self._last_target_delta_servo_applied = False
        self._last_target_delta_servo_block_reason = "disabled"
        self._last_target_delta_servo_source = "none"
        self._last_target_delta_servo_local_delta = None
        self._last_target_delta_servo_world_delta = None
        self._last_target_delta_servo_pos_norm = 0.0
        self._last_target_delta_servo_rot_norm = 0.0
        self._last_target_delta_servo_cur_xy = None
        self._last_target_delta_servo_cur_z = None
        self._last_target_delta_servo_cur_yaw = None
        self._last_target_delta_servo_post_xy = None
        self._last_target_delta_servo_post_z = None
        self._last_target_delta_servo_post_yaw = None
        self._last_target_delta_servo_gate_pass = False
        self._near_zone_gate_eval_count = 0
        self._near_zone_gate_pass_count = 0
        self._near_zone_gate_block_count = 0
        self._last_near_zone_gate_pass = False
        self._last_near_zone_xy_error = np.nan
        self._last_near_zone_z_error = np.nan
        self._last_near_zone_block_reason = "disabled"
        self._last_nz_blocked_zone = False
        self._last_v2_delta_source = "none"
        self._last_v2_delta_norm = 0.0
        self._last_v2_gripper_pose_present = False
        self._last_v2_mt_pose_present = False
        self._last_v2_mt_delta_present = False
        self._last_v2_cdb_present = False
        self._last_raw_delta_pose_local = None
        self._last_preclip_delta_pose_local = None
        self._last_clipped_delta_pose_local = None
        self._last_delta_pose_world = None
        self._last_raw_residual_pos_norm = 0.0
        self._last_preclip_residual_pos_norm = 0.0
        self._last_clipped_residual_pos_norm = 0.0
        self._last_raw_residual_yaw_abs = 0.0
        self._last_preclip_residual_yaw_abs = 0.0
        self._last_clipped_residual_yaw_abs = 0.0
        self._last_v2_selected_post_xy = None
        self._last_v2_selected_post_z = None
        self._last_v2_selected_post_yaw = None
        self._last_v2_apply_gate_pass = False
        self._last_v2_apply_block_reason = "disabled"
        self._last_v2_apply_assist_scale = 0.0
        self._last_v2_apply_local_delta = None
        self._last_v2_apply_world_delta = None
        self._alignment_gate_block_count = 0
        self._alignment_window_corrections = 0
        self._readiness_eval_count = 0
        self._ready_prob_sum = 0.0
        self._readiness_close_override_count = 0
        self._readiness_open_override_count = 0
        self._readiness_hold_override_count = 0
        self._readiness_heads_missing_count = 0
        self._preclose_alignment_correction_count = 0
        self._ready_positive_count = 0
        self._last_ready_prob = 0.0
        self._last_basin_positive = 0.0
        self._raw_residual_norm_sum = 0.0
        self._raw_residual_pos_norm_sum = 0.0
        self._preclip_residual_pos_norm_sum = 0.0
        self._clipped_residual_pos_norm_sum = 0.0
        self._executed_residual_norm_sum = 0.0
        self._clip_hit_count = 0
        self._planner_close_intent_count = 0
        self._planner_close_intent_eval_count = 0
        self._scorer_pred_hist = {}
        self._scorer_group_hist = {}
        self._last_scorer_candidate_index = -1
        self._last_scorer_group_index = -1
        self._last_selected_step_scale = 1.0
        self._alignment_decision_count = 0
        self._alignment_noop_count = 0
        self._alignment_low_conf_noop_count = 0
        self._alignment_physical_mask_count = 0
        self._alignment_all_masked_fallback_count = 0
        self._handoff_yaw_priority_count = 0
        self._handoff_xy_priority_count = 0
        self._last_handoff_yaw_priority_active = False
        self._last_handoff_xy_priority_active = False
        self._last_handoff_axis_priority = "none"
        self._near_handoff_z_correction_count = 0
        self._outer_rescue_decision_count = 0
        self._outer_rescue_correction_count = 0
        self._outer_rescue_noop_count = 0
        self._outer_rescue_handoff_count = 0
        self._close_veto_block_count = 0
        self._close_veto_pass_count = 0
        self._close_latch_set_count = 0
        self._close_latch_release_count = 0
        self._close_latch_expire_count = 0
        self._phase1_post_close_hold_remaining = 0
        self._phase1_post_close_hold_count = 0
        self._phase1_post_close_hold_expire_count = 0
        self._last_phase1_post_close_hold_active = False
        self._close_veto_settle_count = 0
        self._close_latch_remaining = 0
        self._close_veto_settle_remaining = 0
        self._close_veto_runtime_geometry_fallback_eval_count = 0
        self._close_veto_runtime_geometry_fallback_ready_count = 0
        self._last_close_veto_runtime_geometry_fallback_used = False
        self._last_close_veto_runtime_geometry_fallback_ready = False
        self._bounded_auto_close_eval_count = 0
        self._bounded_auto_close_apply_count = 0
        self._bounded_auto_close_ready_streak = 0
        self._last_bounded_auto_close_ready = False
        self._last_bounded_auto_close_applied = False
        self._force_close_after_b2_eval_count = 0
        self._last_force_close_after_b2_applied = False
        self._close_intent_shadow_candidate_count = 0
        self._last_close_intent_shadow_would_auto_close = False
        self._last_close_intent_shadow_reason = "reset"
        self._last_close_intent_shadow_blocking_axis = "none"
        self._last_close_intent_shadow_confidence = 0.0
        self._alignment_takeover_count = 0
        self._last_alignment_takeover_active = False
        self._alignment_zone_state = "planner_only"
        self._alignment_active = False
        self._last_smoothed_delta_local = np.zeros(6, dtype=np.float32)
        self._candidate_hold_remaining = 0
        self._held_candidate_index = -1
        self._held_candidate_delta_local = np.zeros(6, dtype=np.float32)
        self._alignment_gate_block_reason_counts = {
            "depth": 0,
            "gripper_open": 0,
            "close_intent": 0,
            "support": 0,
            "ready_to_close": 0,
            "window": 0,
            "cooldown": 0,
        }
        self._last_alignment_gate_debug = {
            "gate_open": False,
            "blocked_reason": "none",
            "depth_proximity": None,
            "near_target": False,
            "gripper_open": None,
            "gripper_still_open": False,
            "planner_close_intent": False,
            "close_requirement_satisfied": False,
            "alignment_window_active": False,
            "support_inner_satisfied": True,
            "support_outer_satisfied": True,
            "support_soft_override": False,
            "support_soft_override_reason": "none",
            "use_outer_rescue": False,
            "short_window_available": True,
            "residual_cooldown": 0,
        }
        self._gripper_fsm_state = "block_open"
        self._fire_ready_streak = 0
        self._verify_steps_remaining = 0
        self._verified_contact_count = 0
        self._verify_fail_reopen_count = 0
        self._last_internal_readiness_fire_applied = False
        self._last_internal_readiness_fire_ready = False
        self._last_internal_readiness_band_ready = False
        self._alignment_complete_eval_count = 0
        self._alignment_complete_positive_count = 0
        self._alignment_fire_close_count = 0
        self._last_alignment_complete = False
        self._last_alignment_fire_decision = False
        self._last_alignment_fire_xy = np.nan
        self._last_alignment_fire_z = np.nan
        self._last_alignment_fire_yaw = np.nan
        self._last_outer_rescue_active = False
        self._last_close_veto_blocked = False
        self._last_close_veto_ready = False
        self._close_veto_ready_streak = 0
        self._close_veto_ready_step_idx = -1
        self._close_veto_recent_block_streak = 0
        self._last_close_state_machine = self._make_close_state_snapshot(
            state="init",
            action_decision="none",
        )
        self._near_ready_rerank_eval_count = 0
        self._near_ready_rerank_gate_pass_count = 0
        self._near_ready_rerank_apply_count = 0
        self._near_ready_rerank_change_count = 0
        self._last_near_ready_rerank_gate_open = False
        self._last_near_ready_rerank_applied = False
        self._last_near_ready_rerank_changed = False
        self._last_near_ready_rerank_prev_index = -1
        self._last_near_ready_rerank_new_index = -1
        self._last_near_ready_rerank_topk = 0
        self._last_near_ready_rerank_cur_xy = np.nan
        self._last_near_ready_rerank_cur_z = np.nan
        self._last_near_ready_rerank_cur_yaw = np.nan
        self._last_near_ready_rerank_gate_xy_max = np.nan
        self._last_near_ready_rerank_gate_z_max = np.nan
        self._last_near_ready_rerank_gate_yaw_max = np.nan
        self._near_ready_specialist_eval_count = 0
        self._near_ready_specialist_gate_pass_count = 0
        self._near_ready_specialist_use_count = 0
        self._last_near_ready_specialist_gate_open = False
        self._last_near_ready_specialist_active = False
        self._last_near_ready_specialist_cur_xy = np.nan
        self._last_near_ready_specialist_cur_z = np.nan
        self._last_near_ready_specialist_cur_yaw = np.nan
        self._last_near_ready_specialist_gate_xy_max = np.nan
        self._last_near_ready_specialist_gate_z_max = np.nan
        self._last_near_ready_specialist_gate_yaw_max = np.nan
        self._near_ready_residual_eval_count = 0
        self._near_ready_residual_gate_pass_count = 0
        self._near_ready_residual_apply_count = 0
        self._near_ready_residual_change_count = 0
        self._last_near_ready_residual_gate_open = False
        self._last_near_ready_residual_applied = False
        self._last_near_ready_residual_changed = False
        self._last_near_ready_residual_prev_index = -1
        self._last_near_ready_residual_new_index = -1
        self._near_ready_group_residual_eval_count = 0
        self._near_ready_group_residual_gate_pass_count = 0
        self._near_ready_group_residual_apply_count = 0
        self._near_ready_group_residual_change_count = 0
        self._last_near_ready_group_residual_gate_open = False
        self._last_near_ready_group_residual_applied = False
        self._last_near_ready_group_residual_changed = False
        self._last_near_ready_group_residual_prev_group = -1
        self._last_near_ready_group_residual_new_group = -1
        self._near_ready_residual_eval_count = 0
        self._near_ready_residual_gate_pass_count = 0
        self._near_ready_residual_apply_count = 0
        self._near_ready_residual_change_count = 0
        self._near_ready_group_residual_eval_count = 0
        self._near_ready_group_residual_gate_pass_count = 0
        self._near_ready_group_residual_apply_count = 0
        self._near_ready_group_residual_change_count = 0
        self._last_near_ready_group_residual_gate_open = False
        self._last_near_ready_group_residual_applied = False
        self._last_near_ready_group_residual_changed = False
        self._last_near_ready_group_residual_prev_group = -1
        self._last_near_ready_group_residual_new_group = -1
        self._b1_group_shadow_eval_count = 0
        self._b1_group_shadow_gate_pass_count = 0
        self._b1_group_shadow_change_count = 0
        self._b1_group_shadow_disagreement_count = 0
        self._b1_group_shadow_teacher_group_valid_count = 0
        self._b1_group_shadow_cost_valid_count = 0
        self._b1_group_shadow_cost_improve_count = 0
        self._b1_group_shadow_cost_worse_count = 0
        self._b1_group_shadow_regret_delta_sum = 0.0
        self._b1_group_shadow_close_count = 0
        self._b1_group_shadow_close_change_count = 0
        self._b1_group_shadow_close_cost_valid_count = 0
        self._b1_group_shadow_close_cost_improve_count = 0
        self._b1_group_shadow_close_cost_worse_count = 0
        self._b1_group_shadow_close_regret_delta_sum = 0.0
        self._b1_apply_gate_eval_count = 0
        self._b1_apply_gate_apply_count = 0
        self._b1_apply_gate_cost_valid_count = 0
        self._b1_apply_gate_cost_improve_count = 0
        self._b1_apply_gate_cost_worse_count = 0
        self._b1_apply_gate_regret_delta_sum = 0.0
        self._b1_apply_gate_close_cost_valid_count = 0
        self._b1_apply_gate_close_cost_improve_count = 0
        self._b1_apply_gate_close_cost_worse_count = 0
        self._b1_apply_gate_close_regret_delta_sum = 0.0
        self._b1_group_bounded_apply_count = 0
        self._b1_group_bounded_change_count = 0
        self._b2_candidate_shadow_eval_count = 0
        self._b2_candidate_shadow_gate_pass_count = 0
        self._b2_candidate_shadow_change_count = 0
        self._b2_candidate_shadow_cost_valid_count = 0
        self._b2_candidate_shadow_cost_improve_count = 0
        self._b2_candidate_shadow_cost_worse_count = 0
        self._b2_candidate_shadow_regret_delta_sum = 0.0
        self._b2_candidate_shadow_mode_keep_count = 0
        self._b2_candidate_shadow_mode_apply_count = 0
        self._b2_candidate_shadow_close_count = 0
        self._b2_candidate_shadow_yaw_needed_count = 0
        self._b2_candidate_shadow_yaw_keep_count = 0
        self._b2_candidate_shadow_teacher_ready_count = 0
        self._b2_candidate_shadow_xy_block_count = 0
        self._b2_candidate_shadow_nearish_count = 0
        self._b2_candidate_shadow_keep_baseline_forced_count = 0
        self._b2_candidate_bounded_eval_count = 0
        self._b2_candidate_bounded_gate_pass_count = 0
        self._b2_candidate_bounded_apply_count = 0
        self._b2_candidate_bounded_change_count = 0
        self._last_b1_group_shadow_gate_open = False
        self._last_b1_group_shadow_pred_group = -1
        self._last_b1_group_shadow_baseline_group = -1
        self._last_b1_group_shadow_teacher_group = -1
        self._last_b1_group_shadow_changed = False
        self._last_b1_group_shadow_teacher_disagree = False
        self._last_b1_group_shadow_teacher_group_valid = False
        self._last_b1_group_shadow_close_neighborhood = False
        self._last_b1_group_shadow_close_group_changed = False
        self._last_b1_group_shadow_margin = np.nan
        self._last_b1_group_shadow_teacher_best_cost = np.nan
        self._last_b1_group_shadow_baseline_group_cost = np.nan
        self._last_b1_group_shadow_pred_group_cost = np.nan
        self._last_b1_group_shadow_baseline_group_regret = np.nan
        self._last_b1_group_shadow_pred_group_regret = np.nan
        self._last_b1_group_shadow_regret_delta = np.nan
        self._last_b1_apply_gate_prob = np.nan
        self._last_b1_apply_gate_threshold = np.nan
        self._last_b1_apply_gate_apply = False
        self._last_b1_apply_gate_vetoed = False
        self._last_b1_group_bounded_applied = False
        self._last_b1_group_bounded_prev_group = -1
        self._last_b1_group_bounded_new_group = -1
        self._last_b2_candidate_shadow_gate_open = False
        self._last_b2_candidate_shadow_close_neighborhood = False
        self._last_b2_candidate_shadow_mode = -1
        self._last_b2_candidate_shadow_mode_confidence = np.nan
        self._last_b2_candidate_shadow_mode_margin = np.nan
        self._last_b2_candidate_shadow_baseline_index = -1
        self._last_b2_candidate_shadow_pred_index = -1
        self._last_b2_candidate_shadow_changed = False
        self._last_b2_candidate_shadow_best_index = -1
        self._last_b2_candidate_shadow_best_cost = np.nan
        self._last_b2_candidate_shadow_baseline_cost = np.nan
        self._last_b2_candidate_shadow_pred_cost = np.nan
        self._last_b2_candidate_shadow_baseline_regret = np.nan
        self._last_b2_candidate_shadow_pred_regret = np.nan
        self._last_b2_candidate_shadow_regret_delta = np.nan
        self._last_b2_candidate_shadow_yaw_needed = False
        self._last_b2_candidate_shadow_yaw_keep = False
        self._last_b2_candidate_shadow_teacher_ready = False
        self._last_b2_candidate_shadow_xy_block = False
        self._last_b2_candidate_shadow_runtime_scope_size = 0
        self._last_b2_candidate_shadow_small_yaw_scope_size = 0
        self._last_b2_candidate_shadow_large_yaw_scope_size = 0
        self._last_b2_candidate_shadow_probe_count = 0
        self._last_b2_candidate_shadow_nearish_runtime = False
        self._last_b2_candidate_shadow_keep_baseline_forced = False
        self._last_b2_candidate_shadow_candidate_actions_local = None
        self._last_b2_candidate_shadow_candidate_scope_mask = None
        self._last_b2_candidate_shadow_candidate_valid_mask = None
        self._last_b2_candidate_shadow_candidate_cost = None
        self._last_b2_candidate_shadow_candidate_oracle_score = None
        self._last_b2_candidate_bounded_gate_open = False
        self._last_b2_candidate_bounded_applied = False
        self._last_b2_candidate_bounded_changed = False
        self._last_b2_candidate_bounded_prev_index = -1
        self._last_b2_candidate_bounded_new_index = -1
        self._last_b2_candidate_bounded_mode_confidence = np.nan
        self._last_b2_candidate_bounded_mode_margin = np.nan
        self._close_intent_shadow_candidate_count = 0
        self._last_close_intent_shadow_would_auto_close = False
        self._last_close_intent_shadow_reason = "init"
        self._last_close_intent_shadow_blocking_axis = "none"
        self._last_close_intent_shadow_confidence = 0.0
        self._last_near_ready_residual_gate_open = False
        self._last_near_ready_residual_applied = False
        self._last_near_ready_residual_changed = False
        self._last_near_ready_residual_prev_index = -1
        self._last_near_ready_residual_new_index = -1

    def _update_handoff_xy_micro_state(
        self,
        controller,
        current_delta: np.ndarray,
    ) -> bool:
        if (
            not self.enable_handoff_xy_micro_polish
            or controller is None
            or getattr(controller, "_controller_type", "") != "pose_field_scorer"
        ):
            self._handoff_xy_micro_stall_streak = 0
            self._handoff_xy_micro_prev_xy = np.inf
            self._handoff_xy_micro_remaining = 0
            return False
        current_delta = np.asarray(current_delta, dtype=np.float32).reshape(-1)
        if current_delta.size < 6:
            self._handoff_xy_micro_stall_streak = 0
            self._handoff_xy_micro_prev_xy = np.inf
            self._handoff_xy_micro_remaining = 0
            return False

        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", -1.0))
        handoff_z = float(handoff_thresholds.get("abs_z_error", -1.0))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", -1.0))
        if not np.isfinite(handoff_xy) or handoff_xy <= 0.0:
            self._handoff_xy_micro_stall_streak = 0
            self._handoff_xy_micro_prev_xy = np.inf
            self._handoff_xy_micro_remaining = 0
            return False

        cur_xy = float(np.linalg.norm(current_delta[:2]))
        cur_z = float(abs(current_delta[2]))
        cur_yaw = float(abs(current_delta[5]))
        xy_gate = max(handoff_xy * self.final_handoff_polish_xy_multiplier, 0.010)
        z_ready = cur_z <= max((handoff_z if handoff_z > 0.0 else self.close_veto_abs_z_threshold) * 1.15, 0.0045)
        yaw_ready = handoff_yaw < 0.0 or cur_yaw <= max(handoff_yaw * 1.15, 0.14)
        xy_ready = cur_xy <= max(handoff_xy * 1.05, 0.008)
        near_ready = bool(
            cur_xy <= xy_gate
            and z_ready
            and yaw_ready
            and not xy_ready
        )

        if not near_ready:
            self._handoff_xy_micro_stall_streak = 0
            self._handoff_xy_micro_prev_xy = np.inf
            self._handoff_xy_micro_remaining = 0
            return False

        prev_xy = float(self._handoff_xy_micro_prev_xy)
        if np.isfinite(prev_xy):
            improvement = prev_xy - cur_xy
            if improvement < self.handoff_xy_micro_improve_eps:
                self._handoff_xy_micro_stall_streak += 1
            else:
                self._handoff_xy_micro_stall_streak = 0
        else:
            self._handoff_xy_micro_stall_streak = 0
        self._handoff_xy_micro_prev_xy = cur_xy

        if self._handoff_xy_micro_remaining > 0:
            self._handoff_xy_micro_remaining -= 1
            return True
        if self._handoff_xy_micro_stall_streak >= self.handoff_xy_micro_stall_frames:
            self._handoff_xy_micro_remaining = max(self.handoff_xy_micro_window_steps - 1, 0)
            self._handoff_xy_micro_stall_streak = 0
            self._handoff_xy_micro_trigger_count += 1
            return True
        return False

    def _clip_target_context_for_scorer(self, controller, current_delta: np.ndarray) -> np.ndarray:
        current_delta = np.asarray(current_delta, dtype=np.float32).reshape(6)
        clip_abs = getattr(controller, "_target_context_clip_abs", None)
        if clip_abs is None:
            return current_delta
        clip_abs = np.asarray(clip_abs, dtype=np.float32).reshape(6)
        return np.clip(current_delta, -clip_abs, clip_abs).astype(np.float32)

    def _clip_residual_no_count(self, delta_pose_6d: np.ndarray) -> np.ndarray:
        out = np.asarray(delta_pose_6d, dtype=np.float32).copy()
        pos_norm = float(np.linalg.norm(out[:3]))
        if pos_norm > float(self.safety.max_residual_pos):
            out[:3] *= float(self.safety.max_residual_pos) / max(pos_norm, 1e-8)
        rot_norm = float(np.linalg.norm(out[3:6]))
        if rot_norm > float(self.safety.max_residual_rot):
            out[3:6] *= float(self.safety.max_residual_rot) / max(rot_norm, 1e-8)
        return out

    def _make_runtime_candidate_mask(
        self,
        controller,
        candidate_actions: np.ndarray,
        base_candidate_mask: Optional[np.ndarray],
        current_quat: np.ndarray,
        gripper_pose: Optional[np.ndarray],
        exec_base_action_world: Optional[np.ndarray],
    ) -> np.ndarray:
        mask = (
            np.asarray(base_candidate_mask, dtype=np.float32).copy()
            if base_candidate_mask is not None
            else np.ones((candidate_actions.shape[0],), dtype=np.float32)
        )
        if not self.enable_alignment_physical_mask:
            return mask
        if gripper_pose is None:
            return mask
        current_xyz = np.asarray(gripper_pose[:3], dtype=np.float32)
        exec_base = (
            np.zeros(6, dtype=np.float32)
            if exec_base_action_world is None
            else np.asarray(exec_base_action_world, dtype=np.float32).reshape(6)
        )
        margin = float(getattr(controller, "_runtime_workspace_violation_margin", 0.004))
        invalid_before = int(np.sum(mask <= 0.5))
        for i, cand in enumerate(np.asarray(candidate_actions, dtype=np.float32)):
            if mask[i] <= 0.5:
                continue
            residual_local = self._clip_residual_no_count(cand * float(self.learned_residual_scale))
            residual_world = local_delta_to_world(residual_local, current_quat)
            next_xyz = current_xyz + exec_base[:3] + residual_world[:3]
            if self.safety.workspace_violation(next_xyz) > margin:
                mask[i] = 0.0
        # Keep no-op as a safe fallback so an all-masked group cannot turn into
        # argmax over arbitrary -1e9 scores.
        if mask.shape[0] > 0:
            mask[0] = 1.0
        invalid_after = int(np.sum(mask <= 0.5))
        if invalid_after > invalid_before:
            self._alignment_physical_mask_count += invalid_after - invalid_before
        return mask

    @staticmethod
    def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an < 1e-8 or bn < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    @staticmethod
    def _pose_delta_local_between(current_pose_7d: np.ndarray, target_pose_7d: np.ndarray) -> np.ndarray:
        current_pose = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
        target_pose = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
        delta_pos_world = target_pose[:3] - current_pose[:3]
        r_cur = Rotation.from_quat(current_pose[3:7])
        r_tgt = Rotation.from_quat(target_pose[3:7])
        delta_rot = (r_tgt * r_cur.inv()).as_rotvec().astype(np.float32)
        delta_pos_local = r_cur.inv().apply(delta_pos_world.astype(np.float32)).astype(np.float32)
        return np.concatenate([delta_pos_local, delta_rot], axis=0).astype(np.float32)

    @staticmethod
    def _apply_local_offset_to_pose(pose_7d: np.ndarray, delta_local_6d: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_7d, dtype=np.float32).copy().reshape(7)
        delta = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
        r_cur = Rotation.from_quat(pose[3:7])
        pose[:3] = pose[:3] + r_cur.apply(delta[:3]).astype(np.float32)
        r_delta = Rotation.from_rotvec(delta[3:6].astype(np.float32))
        pose[3:7] = (r_delta * r_cur).as_quat().astype(np.float32)
        return pose

    @staticmethod
    def _alignment_error_components(delta_pose_local: np.ndarray) -> tuple[float, float, float, float]:
        delta = np.asarray(delta_pose_local, dtype=np.float32).reshape(-1)
        xy_error = float(np.linalg.norm(delta[:2])) if delta.size >= 2 else np.inf
        z_error = float(abs(delta[2])) if delta.size >= 3 else np.inf
        tilt_error = float(np.linalg.norm(delta[3:5])) if delta.size >= 5 else np.inf
        yaw_error = float(abs(delta[5])) if delta.size >= 6 else np.inf
        return xy_error, z_error, yaw_error, tilt_error

    def _support_band_limits(self, controller, band: str = "inner") -> tuple[float, float, float, float]:
        if band == "inner":
            xy_max = float(getattr(controller, "_support_inner_xy_max", np.inf))
            z_max = float(getattr(controller, "_support_inner_abs_z_max", np.inf))
            tilt_max = (
                np.inf
                if bool(getattr(controller, "_ignore_tilt_alignment", False))
                else float(getattr(controller, "_support_inner_tilt_max", np.inf))
            )
            yaw_max = float(getattr(controller, "_support_inner_yaw_max", np.inf))
            handoff_spec_name = str(getattr(controller, "_runtime_handoff_spec_name", "none"))
            handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
            if handoff_spec_name and handoff_spec_name != "none":
                # The stage spec owns which metrics are semantically required for
                # handoff. If yaw/tilt are disabled in the spec, they must not
                # silently block the motion refiner's support gate.
                if float(handoff_thresholds.get("yaw_error", -1.0)) < 0.0:
                    yaw_max = np.inf
                if float(handoff_thresholds.get("tilt_error", -1.0)) < 0.0:
                    tilt_max = np.inf
            return xy_max, z_max, yaw_max, tilt_max
        inner_xy, inner_z, inner_yaw, inner_tilt = self._support_band_limits(controller, band="inner")
        xy_max = max(inner_xy * self.outer_rescue_xy_scale, self.outer_rescue_min_xy)
        z_max = max(inner_z * self.outer_rescue_abs_z_scale, self.outer_rescue_min_abs_z)
        yaw_max = max(inner_yaw * self.outer_rescue_yaw_scale, self.outer_rescue_min_yaw)
        tilt_max = max(inner_tilt * self.outer_rescue_yaw_scale, 0.20)
        return float(xy_max), float(z_max), float(yaw_max), float(tilt_max)

    def _motion_handoff_metric_thresholds(self, controller) -> dict:
        if controller is None:
            return {}
        if bool(getattr(controller, "_runtime_allow_handoff_motion_shaping", False)):
            return dict(getattr(controller, "_runtime_handoff_metric_thresholds", {}) or {})
        return {}

    def _delta_within_band(self, delta_pose_local: np.ndarray, controller, band: str = "inner") -> bool:
        if controller is None:
            return False
        xy_max, z_max, yaw_max, tilt_max = self._support_band_limits(controller, band=band)
        xy_error, z_error, yaw_error, tilt_error = self._alignment_error_components(delta_pose_local)
        if not np.isfinite(xy_error) or not np.isfinite(z_error) or not np.isfinite(yaw_error) or not np.isfinite(tilt_error):
            return False
        if xy_error > xy_max or z_error > z_max or tilt_error > tilt_max:
            return False
        if yaw_max >= 0.0 and yaw_error > yaw_max:
            return False
        return True

    @staticmethod
    def _runtime_has_motion_target(controller) -> bool:
        if controller is None:
            return False
        motion_target = getattr(controller, "_runtime_motion_target_pose_7d", None)
        if motion_target is None:
            motion_target = getattr(controller, "_canonical_basin_center_pose_7d", None)
        if motion_target is not None:
            try:
                arr = np.asarray(motion_target, dtype=np.float32).reshape(7)
                if np.all(np.isfinite(arr)):
                    return True
            except Exception:
                pass
        delta = getattr(controller, "_runtime_current_delta_basin_target", None)
        if delta is not None:
            try:
                arr = np.asarray(delta, dtype=np.float32).reshape(-1)
                if arr.size >= 6 and np.all(np.isfinite(arr[:6])):
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _runtime_has_handoff_geometry(controller) -> bool:
        if controller is None:
            return False
        handoff_target = getattr(controller, "_runtime_handoff_target_pose_7d", None)
        if handoff_target is not None:
            try:
                arr = np.asarray(handoff_target, dtype=np.float32).reshape(7)
                if np.all(np.isfinite(arr)):
                    return True
            except Exception:
                pass
        spec_name = str(getattr(controller, "_runtime_handoff_spec_name", "none"))
        return bool(spec_name and spec_name != "none")

    def _close_veto_runtime_geometry_ready(self, controller, gripper_open) -> bool:
        if controller is None:
            return False
        if not self._runtime_has_motion_target(controller):
            return False
        is_open = gripper_open is None or float(gripper_open) >= self.alignment_open_threshold
        if not is_open:
            return False
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        )
        xy_error, z_error, yaw_error, _tilt_error = self._alignment_error_components(current_delta)
        handoff_thresholds = dict(getattr(controller, "_runtime_handoff_metric_thresholds", {}) or {})
        xy_threshold = float(handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        z_threshold = float(handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        yaw_threshold = float(handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        # Keep fallback conservative: never relax below the active close-veto
        # thresholds configured for runtime.
        xy_threshold = min(xy_threshold, self.close_veto_xy_threshold)
        z_threshold = min(z_threshold, self.close_veto_abs_z_threshold)
        if self.close_veto_yaw_threshold >= 0.0:
            yaw_threshold = min(yaw_threshold, self.close_veto_yaw_threshold) if yaw_threshold >= 0.0 else self.close_veto_yaw_threshold
        return bool(
            self._delta_within_band(current_delta, controller, band="outer")
            and xy_error <= xy_threshold
            and z_error <= z_threshold
            and (yaw_threshold < 0.0 or yaw_error <= yaw_threshold)
        )

    def _make_close_state_snapshot(self, **updates) -> dict:
        base = {
            "state": "unknown",
            "action_decision": "none",
            "wants_close": False,
            "planner_close_intent": False,
            "gripper_open": None,
            "is_open": False,
            "has_motion_target": False,
            "has_handoff_geometry": False,
            "handoff_spec_name": "none",
            "handoff_ready_pred": False,
            "handoff_ready_applied": False,
            "handoff_shadow_only": False,
            "handoff_shadow_blocks_apply": False,
            "fallback_enabled": False,
            "fallback_used": False,
            "runtime_geometry_ready": False,
            "support_outer": False,
            "xy_error": None,
            "abs_z_error": None,
            "yaw_error": None,
            "xy_threshold": None,
            "abs_z_threshold": None,
            "yaw_threshold": None,
            "ready_streak": int(self._close_veto_ready_streak),
            "min_stable_frames": int(self.close_veto_ready_streak_frames),
            "streak_ready": False,
            "latch_remaining": int(self._close_latch_remaining),
            "settle_remaining": int(self._close_veto_settle_remaining),
            "blocked_reason": "none",
        }
        base.update(updates)
        return base

    def _runtime_geometry_ready_detail(self, controller, gripper_open, *, use_handoff_thresholds: bool) -> dict:
        if controller is None:
            return self._make_close_state_snapshot(state="no_controller", blocked_reason="no_controller")
        has_motion_target = bool(self._runtime_has_motion_target(controller))
        has_handoff_geometry = bool(self._runtime_has_handoff_geometry(controller))
        is_open = gripper_open is None or float(gripper_open) >= self.alignment_open_threshold
        if not has_motion_target:
            return self._make_close_state_snapshot(
                state="no_motion_target",
                gripper_open=None if gripper_open is None else float(gripper_open),
                is_open=bool(is_open),
                has_motion_target=False,
                has_handoff_geometry=has_handoff_geometry,
                blocked_reason="no_motion_target",
            )
        if not is_open:
            return self._make_close_state_snapshot(
                state="gripper_not_open",
                gripper_open=None if gripper_open is None else float(gripper_open),
                is_open=False,
                has_motion_target=True,
                has_handoff_geometry=has_handoff_geometry,
                blocked_reason="gripper_not_open",
            )
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        xy_error, z_error, yaw_error, _tilt_error = self._alignment_error_components(current_delta)
        if use_handoff_thresholds:
            thresholds = dict(getattr(controller, "_runtime_handoff_metric_thresholds", {}) or {})
            xy_threshold = float(thresholds.get("xy_error", self.close_veto_xy_threshold))
            z_threshold = float(thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
            yaw_threshold = float(thresholds.get("yaw_error", self.close_veto_yaw_threshold))
            xy_threshold = min(xy_threshold, self.close_veto_xy_threshold)
            z_threshold = min(z_threshold, self.close_veto_abs_z_threshold)
            if self.close_veto_yaw_threshold >= 0.0:
                yaw_threshold = (
                    min(yaw_threshold, self.close_veto_yaw_threshold)
                    if yaw_threshold >= 0.0
                    else self.close_veto_yaw_threshold
                )
        else:
            xy_threshold = float(self.close_veto_xy_threshold)
            z_threshold = float(self.close_veto_abs_z_threshold)
            yaw_threshold = float(self.close_veto_yaw_threshold)
        support_outer = bool(self._delta_within_band(current_delta, controller, band="outer"))
        ready = bool(
            support_outer
            and xy_error <= xy_threshold
            and z_error <= z_threshold
            and (yaw_threshold < 0.0 or yaw_error <= yaw_threshold)
        )
        blocked_reason = "none"
        if not ready:
            if not support_outer:
                blocked_reason = "support_outer"
            elif xy_error > xy_threshold:
                blocked_reason = "xy"
            elif z_error > z_threshold:
                blocked_reason = "z"
            elif yaw_threshold >= 0.0 and yaw_error > yaw_threshold:
                blocked_reason = "yaw"
            else:
                blocked_reason = "geometry"
        return self._make_close_state_snapshot(
            state="runtime_geometry_ready" if ready else "runtime_geometry_not_ready",
            gripper_open=None if gripper_open is None else float(gripper_open),
            is_open=True,
            has_motion_target=True,
            has_handoff_geometry=has_handoff_geometry,
            runtime_geometry_ready=ready,
            support_outer=support_outer,
            xy_error=float(xy_error),
            abs_z_error=float(z_error),
            yaw_error=float(yaw_error),
            xy_threshold=float(xy_threshold),
            abs_z_threshold=float(z_threshold),
            yaw_threshold=float(yaw_threshold),
            blocked_reason=blocked_reason,
        )

    def _bounded_auto_close_ready(self, controller, gripper_open) -> bool:
        if controller is None:
            return False
        if not self.enable_b2_candidate_bounded_v0:
            return False
        if not self.enable_bounded_auto_close_on_alignment:
            return False
        if not bool(getattr(controller, "_runtime_handoff_shadow_blocks_apply", False)):
            return False
        handoff_spec_name = str(getattr(controller, "_runtime_handoff_spec_name", "none"))
        if not (handoff_spec_name and handoff_spec_name != "none"):
            return False
        if not self._runtime_has_motion_target(controller):
            return False
        is_open = gripper_open is None or float(gripper_open) >= self.alignment_open_threshold
        if not is_open:
            return False
        controller_type = str(getattr(controller, "_controller_type", ""))
        if controller_type != "pose_field_scorer":
            # student-vNext / diffusion controllers do not necessarily populate the
            # legacy handoff-shadow fields, but they do carry a runtime grasp target.
            # Allow an explicit phase1 bridge auto-close when the bridge segment is
            # active and the basin target is already in the close-ready window.
            if not self._phase1_bridge_runtime_segment():
                return False
            current_delta = np.asarray(
                getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            if current_delta.size < 6 or not np.all(np.isfinite(current_delta[:6])):
                return False
            xy_error, z_error, yaw_error, _tilt_error = self._alignment_error_components(current_delta)
            # Keep this stricter than the generic close-veto so we only latch when
            # the bridge is genuinely in basin, not merely near-contact.
            xy_threshold = min(self.close_veto_xy_threshold, max(0.006, self.close_veto_xy_threshold * 0.85))
            z_threshold = min(self.close_veto_abs_z_threshold, max(0.008, self.close_veto_abs_z_threshold * 0.85))
            yaw_threshold = self.close_veto_yaw_threshold if self.close_veto_yaw_threshold >= 0.0 else -1.0
            if yaw_threshold >= 0.0:
                yaw_threshold = min(yaw_threshold, max(0.06, self.close_veto_yaw_threshold * 0.85))
            support_outer = bool(self._delta_within_band(current_delta, controller, band="outer"))
            ready = bool(
                support_outer
                and xy_error <= xy_threshold
                and z_error <= z_threshold
                and (yaw_threshold < 0.0 or yaw_error <= yaw_threshold)
            )
            if ready:
                self._last_close_veto_runtime_geometry_fallback_used = False
                self._last_close_veto_runtime_geometry_fallback_ready = False
                self._last_close_state_machine = self._make_close_state_snapshot(
                    state="vnext_bridge_auto_close_ready",
                    planner_close_intent=False,
                    gripper_open=None if gripper_open is None else float(gripper_open),
                    is_open=True,
                    has_motion_target=True,
                    has_handoff_geometry=True,
                    runtime_geometry_ready=True,
                    support_outer=support_outer,
                    xy_error=float(xy_error),
                    abs_z_error=float(z_error),
                    yaw_error=float(yaw_error),
                    xy_threshold=float(xy_threshold),
                    abs_z_threshold=float(z_threshold),
                    yaw_threshold=float(yaw_threshold),
                    blocked_reason="none",
                )
            return ready
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        xy_error, z_error, yaw_error, _tilt_error = self._alignment_error_components(current_delta)
        handoff_thresholds = dict(getattr(controller, "_runtime_handoff_metric_thresholds", {}) or {})
        xy_threshold = float(handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        z_threshold = float(handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        yaw_threshold = float(handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        # Conservative caps from active close-veto.
        xy_threshold = min(xy_threshold, self.close_veto_xy_threshold)
        z_threshold = min(z_threshold, self.close_veto_abs_z_threshold)
        if self.close_veto_yaw_threshold >= 0.0:
            yaw_threshold = min(yaw_threshold, self.close_veto_yaw_threshold) if yaw_threshold >= 0.0 else self.close_veto_yaw_threshold
        # Optional stricter override for bounded-auto-close experiments.
        if self.bounded_auto_close_xy_threshold >= 0.0:
            xy_threshold = min(xy_threshold, self.bounded_auto_close_xy_threshold)
        if self.bounded_auto_close_abs_z_threshold >= 0.0:
            z_threshold = min(z_threshold, self.bounded_auto_close_abs_z_threshold)
        if self.bounded_auto_close_yaw_threshold >= 0.0:
            yaw_threshold = min(yaw_threshold, self.bounded_auto_close_yaw_threshold) if yaw_threshold >= 0.0 else self.bounded_auto_close_yaw_threshold
        return bool(
            self._delta_within_band(current_delta, controller, band="outer")
            and xy_error <= xy_threshold
            and z_error <= z_threshold
            and (yaw_threshold < 0.0 or yaw_error <= yaw_threshold)
        )

    def _update_close_intent_shadow_candidate(self, gripper_open) -> None:
        state = dict(self._last_close_state_machine)
        planner_close_intent = bool(state.get("planner_close_intent", False))
        is_open = bool(state.get("is_open", gripper_open is None or float(gripper_open) >= self.alignment_open_threshold))
        runtime_geometry_ready = bool(state.get("runtime_geometry_ready", False))
        b2_gate_open = bool(self._last_b2_candidate_bounded_gate_open)
        b2_applied = bool(self._last_b2_candidate_bounded_applied)
        b2_changed = bool(self._last_b2_candidate_bounded_changed)
        conf = float(self._last_b2_candidate_bounded_mode_confidence)
        margin = float(self._last_b2_candidate_bounded_mode_margin)
        if not np.isfinite(conf):
            conf = 0.0
        if not np.isfinite(margin):
            margin = 0.0
        confidence = float(np.clip(0.5 * conf + 5.0 * max(margin, 0.0), 0.0, 1.0))
        blocking_axis = str(state.get("blocked_reason", "none") or "none")
        would_auto_close = bool(
            is_open
            and not planner_close_intent
            and runtime_geometry_ready
            and b2_gate_open
            and (b2_applied or b2_changed)
        )
        if would_auto_close:
            reason = "runtime_geometry_ready_without_planner_close"
            blocking_axis = "none"
            self._close_intent_shadow_candidate_count += 1
        elif not is_open:
            reason = "gripper_not_open"
        elif planner_close_intent:
            reason = "planner_already_has_close_intent"
        elif not runtime_geometry_ready:
            reason = "runtime_geometry_not_ready"
        elif not b2_gate_open:
            reason = "b2_gate_closed"
        else:
            reason = "b2_no_runtime_safe_change"
        self._last_close_intent_shadow_would_auto_close = would_auto_close
        self._last_close_intent_shadow_reason = reason
        self._last_close_intent_shadow_blocking_axis = blocking_axis
        self._last_close_intent_shadow_confidence = confidence

    def _close_veto_geometry_ready(self, controller, gripper_open, planner_close_intent: bool = False) -> bool:
        if controller is None:
            self._last_close_state_machine = self._make_close_state_snapshot(
                state="no_controller",
                planner_close_intent=bool(planner_close_intent),
                blocked_reason="no_controller",
            )
            return False
        has_motion_target = bool(self._runtime_has_motion_target(controller))
        has_handoff_geometry = bool(self._runtime_has_handoff_geometry(controller))
        if not has_motion_target and not has_handoff_geometry:
            self._last_close_state_machine = self._make_close_state_snapshot(
                state="no_target",
                planner_close_intent=bool(planner_close_intent),
                has_motion_target=False,
                has_handoff_geometry=False,
                blocked_reason="no_target",
            )
            return False
        is_open = gripper_open is None or float(gripper_open) >= self.alignment_open_threshold
        if not is_open:
            self._last_close_state_machine = self._make_close_state_snapshot(
                state="gripper_not_open",
                planner_close_intent=bool(planner_close_intent),
                gripper_open=None if gripper_open is None else float(gripper_open),
                is_open=False,
                has_motion_target=has_motion_target,
                has_handoff_geometry=has_handoff_geometry,
                blocked_reason="gripper_not_open",
            )
            return False
        self._last_close_veto_runtime_geometry_fallback_used = False
        self._last_close_veto_runtime_geometry_fallback_ready = False
        handoff_spec_name = str(getattr(controller, "_runtime_handoff_spec_name", "none"))
        if handoff_spec_name and handoff_spec_name != "none":
            handoff_ready_pred = bool(getattr(controller, "_runtime_handoff_ready_pred", False))
            handoff_ready = bool(getattr(controller, "_runtime_handoff_ready", False))
            shadow_only = bool(getattr(controller, "_runtime_handoff_shadow_only", False))
            shadow_blocks_apply = bool(getattr(controller, "_runtime_handoff_shadow_blocks_apply", False))
            if handoff_ready:
                self._last_close_state_machine = self._make_close_state_snapshot(
                    state="handoff_applied_ready",
                    planner_close_intent=bool(planner_close_intent),
                    gripper_open=None if gripper_open is None else float(gripper_open),
                    is_open=True,
                    has_motion_target=has_motion_target,
                    has_handoff_geometry=has_handoff_geometry,
                    handoff_spec_name=handoff_spec_name,
                    handoff_ready_pred=handoff_ready_pred,
                    handoff_ready_applied=True,
                    handoff_shadow_only=shadow_only,
                    handoff_shadow_blocks_apply=shadow_blocks_apply,
                    runtime_geometry_ready=True,
                )
                return True
            fallback_enabled = bool(
                self.enable_b2_candidate_bounded_v0
                and self.close_veto_runtime_geometry_fallback_for_bounded
                and planner_close_intent
                and shadow_blocks_apply
            )
            if not fallback_enabled:
                self._last_close_state_machine = self._make_close_state_snapshot(
                    state="handoff_not_applied",
                    planner_close_intent=bool(planner_close_intent),
                    gripper_open=None if gripper_open is None else float(gripper_open),
                    is_open=True,
                    has_motion_target=has_motion_target,
                    has_handoff_geometry=has_handoff_geometry,
                    handoff_spec_name=handoff_spec_name,
                    handoff_ready_pred=handoff_ready_pred,
                    handoff_ready_applied=False,
                    handoff_shadow_only=shadow_only,
                    handoff_shadow_blocks_apply=shadow_blocks_apply,
                    fallback_enabled=False,
                    blocked_reason="shadow_blocked" if shadow_blocks_apply else "handoff_not_ready",
                )
                return False
            self._last_close_veto_runtime_geometry_fallback_used = True
            self._close_veto_runtime_geometry_fallback_eval_count += 1
            detail = self._runtime_geometry_ready_detail(controller, gripper_open, use_handoff_thresholds=True)
            fallback_ready = bool(detail["runtime_geometry_ready"])
            self._last_close_veto_runtime_geometry_fallback_ready = fallback_ready
            if fallback_ready:
                self._close_veto_runtime_geometry_fallback_ready_count += 1
            detail.update(
                {
                    "state": "fallback_ready" if fallback_ready else "fallback_not_ready",
                    "planner_close_intent": bool(planner_close_intent),
                    "handoff_spec_name": handoff_spec_name,
                    "handoff_ready_pred": handoff_ready_pred,
                    "handoff_ready_applied": False,
                    "handoff_shadow_only": shadow_only,
                    "handoff_shadow_blocks_apply": shadow_blocks_apply,
                    "fallback_enabled": True,
                    "fallback_used": True,
                }
            )
            self._last_close_state_machine = detail
            return fallback_ready
        detail = self._runtime_geometry_ready_detail(controller, gripper_open, use_handoff_thresholds=False)
        detail.update(
            {
                "planner_close_intent": bool(planner_close_intent),
                "handoff_spec_name": handoff_spec_name,
            }
        )
        self._last_close_state_machine = detail
        return bool(detail["runtime_geometry_ready"])

    def _close_veto_ready(
        self,
        controller,
        gripper_open,
        step_idx: Optional[int] = None,
        planner_close_intent: bool = False,
        depth_proximity: Optional[float] = None,
    ) -> bool:
        if step_idx is not None and int(step_idx) == int(self._close_veto_ready_step_idx):
            return bool(self._last_close_veto_ready)
        controller = self._student_vnext_ready_gate_controller(controller)
        controller_type = str(getattr(controller, "_controller_type", ""))
        student_ready_gate = bool(
            self._student_vnext_ready_gate_supported(controller)
            and controller_type == "alignment_tc_student_vnext"
        )
        if controller is None or (controller_type != "pose_field_scorer" and not student_ready_gate):
            self._close_veto_ready_step_idx = -1 if step_idx is None else int(step_idx)
            self._close_veto_ready_streak = 0
            self._last_close_veto_ready = False
            return False
        geometry_ready = self._close_veto_geometry_ready(controller, gripper_open, planner_close_intent=planner_close_intent)
        if student_ready_gate:
            student_gate = self._student_vnext_ready_gate_state(
                controller,
                gripper_open=gripper_open,
                depth_proximity=depth_proximity,
            )
            geometry_ready = bool(
                geometry_ready
                or student_gate.get("close_ready_applied", False)
                or (
                    bool(getattr(controller, "_runtime_handoff_ready_applied", False))
                    and bool(getattr(controller, "_runtime_handoff_ready_pred", False))
                )
            )
        self._close_veto_ready_streak = self._close_veto_ready_streak + 1 if geometry_ready else 0
        min_stable_frames = int(
            getattr(controller, "_runtime_handoff_min_stable_frames", self.close_veto_ready_streak_frames)
        )
        min_stable_frames = max(min_stable_frames, 1)
        ready = bool(self._close_veto_ready_streak >= min_stable_frames)
        self._close_veto_ready_step_idx = -1 if step_idx is None else int(step_idx)
        self._last_close_veto_ready = ready
        self._last_close_state_machine.update(
            {
                "ready_streak": int(self._close_veto_ready_streak),
                "min_stable_frames": int(min_stable_frames),
                "streak_ready": ready,
                "state": "close_ready" if ready else self._last_close_state_machine.get("state", "not_ready"),
            }
        )
        return ready

    def _alignment_refine_band_ready(self, controller, gripper_open) -> bool:
        """Late alignment-entry band.

        This is intentionally tighter than the generic "near target" gate and
        looser than the final handoff-ready gate. It lets alignment step in for
        the last local correction even before planner emits close intent, while
        avoiding very-early takeover that disturbs planner transit.
        """
        if not self._phase1_bridge_controller_capable(controller):
            return False
        if not self._runtime_has_motion_target(controller):
            return False
        if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
            return False
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        xy_error, z_error, yaw_error, tilt_error = self._alignment_error_components(current_delta)
        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", -1.0))
        handoff_z = float(handoff_thresholds.get("abs_z_error", -1.0))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", -1.0))
        handoff_tilt = float(handoff_thresholds.get("tilt_error", -1.0))

        bridge_boost = bool(self._phase1_bridge_runtime_segment())
        xy_entry = max(handoff_xy * (4.5 if bridge_boost else 4.0), 0.014 if bridge_boost else 0.012) if handoff_xy >= 0.0 else max(self.close_veto_xy_threshold * (2.3 if bridge_boost else 2.0), 0.012)
        z_entry = max(handoff_z * (13.0 if bridge_boost else 12.0), 0.014 if bridge_boost else 0.012) if handoff_z >= 0.0 else max(self.close_veto_abs_z_threshold * (2.3 if bridge_boost else 2.0), 0.015)
        yaw_entry = -1.0 if handoff_yaw < 0.0 else max(handoff_yaw * (2.5 if bridge_boost else 2.0), 0.24 if bridge_boost else 0.20)
        tilt_entry = -1.0 if handoff_tilt < 0.0 else max(handoff_tilt * (2.5 if bridge_boost else 2.0), 0.24 if bridge_boost else 0.20)

        return bool(
            np.isfinite(xy_error)
            and np.isfinite(z_error)
            and xy_error <= xy_entry
            and z_error <= z_entry
            and (yaw_entry < 0.0 or yaw_error <= yaw_entry)
            and (tilt_entry < 0.0 or tilt_error <= tilt_entry)
        )

    def _alignment_takeover_band_ready(self, controller, gripper_open) -> bool:
        """Even later band for zeroing planner base motion.

        Assistive residuals may start earlier, but full takeover should only
        happen once we are much closer to the local precision zone.
        """
        if not self._phase1_bridge_controller_capable(controller):
            return False
        if not self._runtime_has_motion_target(controller):
            return False
        if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
            return False
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        xy_error, z_error, yaw_error, tilt_error = self._alignment_error_components(current_delta)
        bridge_boost = bool(self._phase1_bridge_runtime_segment())
        motion_ready = bool(
            np.isfinite(xy_error)
            and np.isfinite(z_error)
            and xy_error <= (self.alignment_takeover_motion_xy_threshold * (1.15 if bridge_boost else 1.0))
            and z_error <= (self.alignment_takeover_motion_abs_z_threshold * (1.15 if bridge_boost else 1.0))
        )
        close_block_ready = bool(
            self._close_veto_recent_block_streak > 0
            and np.isfinite(xy_error)
            and np.isfinite(z_error)
            and xy_error <= (self.alignment_takeover_close_block_xy_threshold * (1.15 if bridge_boost else 1.0))
            and z_error <= (self.alignment_takeover_close_block_abs_z_threshold * (1.15 if bridge_boost else 1.0))
        )
        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", -1.0))
        handoff_z = float(handoff_thresholds.get("abs_z_error", -1.0))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", -1.0))
        handoff_tilt = float(handoff_thresholds.get("tilt_error", -1.0))

        xy_entry = max(handoff_xy * 2.0, 0.008) if handoff_xy >= 0.0 else max(self.close_veto_xy_threshold * 1.5, 0.008)
        z_entry = max(handoff_z * 6.0, 0.008) if handoff_z >= 0.0 else max(self.close_veto_abs_z_threshold * 1.5, 0.010)
        yaw_entry = -1.0 if handoff_yaw < 0.0 else max(handoff_yaw * 1.5, 0.12)
        tilt_entry = -1.0 if handoff_tilt < 0.0 else max(handoff_tilt * 1.5, 0.12)
        handoff_scaled_ready = bool(
            np.isfinite(xy_error)
            and np.isfinite(z_error)
            and xy_error <= xy_entry
            and z_error <= z_entry
            and (yaw_entry < 0.0 or yaw_error <= yaw_entry)
            and (tilt_entry < 0.0 or tilt_error <= tilt_entry)
        )
        return bool(motion_ready or close_block_ready or handoff_scaled_ready)

    def _alignment_assist_band_ready(self, controller, gripper_open) -> bool:
        """Earlier near-field band for additive residual assistance."""
        if not self._phase1_bridge_controller_capable(controller):
            return False
        if not self._runtime_has_motion_target(controller):
            return False
        if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
            return False
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        xy_error, z_error, yaw_error, tilt_error = self._alignment_error_components(current_delta)
        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", -1.0))
        handoff_z = float(handoff_thresholds.get("abs_z_error", -1.0))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", -1.0))
        handoff_tilt = float(handoff_thresholds.get("tilt_error", -1.0))

        bridge_boost = bool(self._phase1_bridge_runtime_segment())
        xy_floor = 0.052 if bridge_boost else 0.040
        z_floor = 0.120 if bridge_boost else 0.100
        yaw_floor = 0.24 if bridge_boost else 0.30
        xy_entry = (
            max(handoff_xy * self.alignment_assist_xy_scale, xy_floor)
            if handoff_xy >= 0.0
            else max(self.close_veto_xy_threshold * self.alignment_assist_xy_scale, xy_floor)
        )
        z_entry = (
            max(handoff_z * self.alignment_assist_abs_z_scale, z_floor)
            if handoff_z >= 0.0
            else max(self.close_veto_abs_z_threshold * self.alignment_assist_abs_z_scale, z_floor)
        )
        yaw_entry = -1.0 if handoff_yaw < 0.0 else max(handoff_yaw * self.alignment_assist_yaw_scale, yaw_floor)
        tilt_entry = -1.0 if handoff_tilt < 0.0 else max(handoff_tilt * self.alignment_assist_yaw_scale, yaw_floor)
        return bool(
            np.isfinite(xy_error)
            and np.isfinite(z_error)
            and xy_error <= xy_entry
            and z_error <= z_entry
            and (yaw_entry < 0.0 or yaw_error <= yaw_entry)
            and (tilt_entry < 0.0 or tilt_error <= tilt_entry)
        )

    def _select_alignment_zone(self, controller, depth_proximity, gripper_open) -> str:
        if not self._phase1_bridge_controller_capable(controller):
            return "planner_only"
        planner_close_intent = bool(getattr(controller, "_alignment_planner_close_intent", False))
        close_requirement_satisfied = bool(
            (not self.require_close_intent_for_alignment)
            or planner_close_intent
        )
        near_target = (
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and float(depth_proximity) < self.alignment_depth_threshold
        )
        if not near_target:
            return "planner_only"
        if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
            return "planner_only"
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        )
        phase1_soft_support, _ = self._phase1_bridge_support_soft_override(
            controller,
            depth_proximity,
            gripper_open,
        )
        support_outer = self._delta_within_band(current_delta, controller, band="outer")
        support_inner = self._delta_within_band(current_delta, controller, band="inner")
        refine_ready = self._alignment_refine_band_ready(controller, gripper_open)
        takeover_ready = self._alignment_takeover_band_ready(controller, gripper_open)
        if self._alignment_zone_state == "takeover":
            if takeover_ready and close_requirement_satisfied:
                return "takeover"
            if (support_inner or phase1_soft_support) and refine_ready and close_requirement_satisfied:
                return "assist"
            return "planner_only"
        if self._alignment_zone_state == "assist":
            if takeover_ready and close_requirement_satisfied:
                return "takeover"
            if refine_ready and (support_outer or phase1_soft_support) and close_requirement_satisfied:
                return "assist"
            return "planner_only"
        if takeover_ready and support_inner and close_requirement_satisfied:
            return "takeover"
        if refine_ready and (support_outer or phase1_soft_support) and close_requirement_satisfied:
            return "assist"
        return "planner_only"

    def _apply_alignment_temporal_smoothing(self, outputs, delta_pose_local: np.ndarray) -> np.ndarray:
        scores = outputs.get("candidate_scores", None)
        pred_idx = int(outputs["pred_candidate_index"].squeeze(0).item()) if "pred_candidate_index" in outputs else -1
        axis_priority = str(outputs.get("handoff_axis_priority", "none"))
        # XY micro polish is intentionally short-lived and already constrained
        # by its candidate mask. Holding an older candidate for 4 frames can
        # turn this into a stale lateral push, so keep the low-pass but bypass
        # candidate dwell in this very-near-band mode.
        use_candidate_hold = pred_idx >= 0 and scores is not None and axis_priority != "xy_micro"
        if use_candidate_hold:
            scores_np = scores.squeeze(0).float().cpu().numpy()
            held_idx = int(self._held_candidate_index)
            if held_idx >= 0 and held_idx < scores_np.shape[0] and held_idx != pred_idx and self._candidate_hold_remaining > 0:
                if float(scores_np[pred_idx]) <= float(scores_np[held_idx]) + self.alignment_candidate_switch_margin:
                    pred_idx = held_idx
                    delta_pose_local = np.asarray(self._held_candidate_delta_local, dtype=np.float32)
                    self._candidate_hold_remaining -= 1
                else:
                    self._held_candidate_index = pred_idx
                    self._held_candidate_delta_local = np.asarray(delta_pose_local, dtype=np.float32).copy()
                    self._candidate_hold_remaining = self.alignment_candidate_hold_steps - 1
            else:
                self._held_candidate_index = pred_idx
                self._held_candidate_delta_local = np.asarray(delta_pose_local, dtype=np.float32).copy()
                self._candidate_hold_remaining = self.alignment_candidate_hold_steps - 1

        alpha = self.alignment_action_lowpass_alpha
        if axis_priority == "xy_micro":
            alpha = max(alpha, 0.88)
        smoothed = (
            alpha * np.asarray(delta_pose_local, dtype=np.float32)
            + (1.0 - alpha) * np.asarray(self._last_smoothed_delta_local, dtype=np.float32)
        ).astype(np.float32)
        self._last_smoothed_delta_local = smoothed.copy()
        return smoothed

    def _apply_near_handoff_z_correction(self, delta_pose_local: np.ndarray, controller) -> np.ndarray:
        if (
            not self.enable_near_handoff_z_correction
            or controller is None
            or getattr(controller, "_controller_type", "") != "pose_field_scorer"
        ):
            return np.asarray(delta_pose_local, dtype=np.float32)
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if current_delta.size < 6:
            return np.asarray(delta_pose_local, dtype=np.float32)
        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", -1.0))
        handoff_z = float(handoff_thresholds.get("abs_z_error", -1.0))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", -1.0))
        if not np.isfinite(handoff_z) or handoff_z <= 0.0:
            return np.asarray(delta_pose_local, dtype=np.float32)

        cur_xy = float(np.linalg.norm(current_delta[:2]))
        cur_z = float(abs(current_delta[2]))
        cur_yaw = float(abs(current_delta[5]))
        xy_gate = max(handoff_xy * max(self.near_handoff_z_xy_multiplier, 2.0), 0.012) if handoff_xy >= 0.0 else 0.012
        z_gate = max(handoff_z * max(self.near_handoff_z_gate_multiplier, 8.0), 0.020)
        yaw_gate = max(handoff_yaw * max(self.near_handoff_z_yaw_multiplier, 1.5), 0.20) if handoff_yaw >= 0.0 else -1.0

        if not (
            cur_xy <= xy_gate
            and cur_z > handoff_z
            and cur_z <= z_gate
            and (yaw_gate < 0.0 or cur_yaw <= yaw_gate)
        ):
            return np.asarray(delta_pose_local, dtype=np.float32)

        corrected = np.asarray(delta_pose_local, dtype=np.float32).copy()
        desired_z = float(np.clip(current_delta[2], -self.near_handoff_z_max_step, self.near_handoff_z_max_step))
        if abs(desired_z) < self.near_handoff_z_min_step:
            desired_z = float(np.sign(current_delta[2]) * self.near_handoff_z_min_step)
        current_z_cmd = float(corrected[2]) if corrected.size >= 3 else 0.0

        wrong_direction = abs(current_z_cmd) > 1e-8 and np.sign(current_z_cmd) != np.sign(desired_z)
        too_small = abs(current_z_cmd) < abs(desired_z)
        if wrong_direction or too_small:
            corrected[2] = float(desired_z)
            self._near_handoff_z_correction_count += 1
        return corrected

    def _apply_final_handoff_polish_mask(
        self,
        controller,
        candidate_actions_src: np.ndarray,
        runtime_mask_np: np.ndarray,
        current_delta: np.ndarray,
    ) -> tuple[np.ndarray, float | None, str]:
        if (
            not self.enable_final_handoff_polish
            or controller is None
            or getattr(controller, "_controller_type", "") != "pose_field_scorer"
        ):
            return runtime_mask_np, None, "none"
        current_delta = np.asarray(current_delta, dtype=np.float32).reshape(-1)
        if current_delta.size < 6:
            return runtime_mask_np, None, "none"
        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", -1.0))
        handoff_z = float(handoff_thresholds.get("abs_z_error", -1.0))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", -1.0))
        if not np.isfinite(handoff_xy) or handoff_xy <= 0.0:
            return runtime_mask_np, None, "none"

        cur_xy = float(np.linalg.norm(current_delta[:2]))
        cur_z = float(abs(current_delta[2]))
        cur_yaw = float(abs(current_delta[5]))
        xy_gate = max(handoff_xy * self.final_handoff_polish_xy_multiplier, 0.010)
        z_gate = max((handoff_z if handoff_z > 0.0 else self.close_veto_abs_z_threshold) * self.final_handoff_polish_z_multiplier, 0.012)
        yaw_gate = max((handoff_yaw if handoff_yaw > 0.0 else 0.12) * self.final_handoff_polish_yaw_multiplier, 0.18)

        if not (
            cur_xy <= xy_gate
            and cur_z <= z_gate
            and cur_yaw <= yaw_gate
        ):
            return runtime_mask_np, None, "none"

        cand_np = np.asarray(candidate_actions_src, dtype=np.float32)
        xy_mag = np.linalg.norm(cand_np[:, :2], axis=1)
        z_mag = np.abs(cand_np[:, 2])
        yaw_mag = np.abs(cand_np[:, 5])
        tilt_mag = np.linalg.norm(cand_np[:, 3:5], axis=1)
        xy_ready = cur_xy <= max(handoff_xy * 1.15, 0.008)
        z_ready = cur_z <= max((handoff_z if handoff_z > 0.0 else self.close_veto_abs_z_threshold) * 1.15, 0.0045)
        yaw_ready = (
            handoff_yaw < 0.0
            or cur_yaw <= max(handoff_yaw * 1.15, 0.14)
        )
        xy_micro_active = self._update_handoff_xy_micro_state(controller, current_delta)
        if xy_micro_active and z_ready and yaw_ready and not xy_ready:
            lateral_only_mask_np = (
                (xy_mag > 1e-8)
                & (xy_mag <= self.handoff_xy_micro_max_xy_step + 1e-8)
                & (z_mag <= 1e-8)
                & (yaw_mag <= 1e-8)
                & (tilt_mag <= 1e-8)
                & (runtime_mask_np > 0.5)
            )
            if np.any(lateral_only_mask_np):
                self._final_handoff_polish_count += 1
                return lateral_only_mask_np.astype(np.float32), float(self.handoff_xy_micro_scale_cap), "xy_micro"
        near_mask_np = (
            (xy_mag <= self.final_handoff_polish_max_xy_step + 1e-8)
            & (z_mag <= self.final_handoff_polish_max_z_step + 1e-8)
            & (yaw_mag <= self.final_handoff_polish_max_yaw_step + 1e-8)
            & (tilt_mag <= 1e-8)
            & (runtime_mask_np > 0.5)
        )
        if np.any(near_mask_np):
            self._final_handoff_polish_count += 1
            return near_mask_np.astype(np.float32), float(self.final_handoff_polish_scale_cap), "generic"
        return runtime_mask_np, None, "none"

    def _alignment_skip_ready(
        self,
        controller,
        gripper_open,
        depth_proximity: Optional[float] = None,
    ) -> bool:
        """Return true when planner is already close enough that alignment should stay out.

        This is intentionally separate from the close-veto readiness band: skipping
        alignment should be tight/conservative, while close veto can be a looser
        safety valve that only blocks obviously bad closes.
        """
        controller_type = str(getattr(controller, "_controller_type", ""))
        student_ready_gate = bool(
            self._student_vnext_ready_gate_supported(controller)
            and controller_type == "alignment_tc_student_vnext"
        )
        if controller is None or (controller_type != "pose_field_scorer" and not student_ready_gate):
            return False
        if not self._runtime_has_motion_target(controller) and not self._runtime_has_handoff_geometry(controller):
            return False
        if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
            return False
        handoff_spec_name = str(getattr(controller, "_runtime_handoff_spec_name", "none"))
        if student_ready_gate:
            student_gate = self._student_vnext_ready_gate_state(
                controller,
                gripper_open=gripper_open,
                depth_proximity=depth_proximity,
            )
            if bool(student_gate.get("handoff_ready_applied", False)) or bool(student_gate.get("close_ready_applied", False)):
                return True
        if handoff_spec_name and handoff_spec_name != "none":
            return bool(getattr(controller, "_runtime_handoff_ready", False))
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        xy_error, z_error, yaw_error, _tilt_error = self._alignment_error_components(current_delta)
        xy_threshold = (
            self.skip_alignment_ready_xy_threshold
            if self.skip_alignment_ready_xy_threshold >= 0.0
            else self.close_veto_xy_threshold
        )
        z_threshold = (
            self.skip_alignment_ready_abs_z_threshold
            if self.skip_alignment_ready_abs_z_threshold >= 0.0
            else self.close_veto_abs_z_threshold
        )
        yaw_threshold = (
            self.skip_alignment_ready_yaw_threshold
            if self.skip_alignment_ready_yaw_threshold >= 0.0
            else self.close_veto_yaw_threshold
        )
        return bool(
            np.isfinite(xy_error)
            and np.isfinite(z_error)
            and xy_error <= xy_threshold
            and z_error <= z_threshold
            and (yaw_threshold < 0.0 or yaw_error <= yaw_threshold)
        )

    @staticmethod
    def _controller_has_internal_readiness(controller) -> bool:
        return bool(
            controller is not None
            and getattr(controller, "_controller_type", "") == "pose_field_scorer"
            and bool(getattr(controller, "_readiness_heads_loaded", False))
        )

    def _planner_close_intent(self, a_base_7d=None, future_gripper_actions=None) -> bool:
        gripper_plan = []
        if a_base_7d is not None:
            gripper_plan.append(float(np.asarray(a_base_7d, dtype=np.float32)[6]))
        if future_gripper_actions is not None:
            gripper_plan.extend(float(x) for x in future_gripper_actions)
        if not gripper_plan:
            return False
        return bool(min(gripper_plan) <= self.alignment_close_command_threshold)

    def _alignment_close_requirement_satisfied(self, controller) -> bool:
        if not self.require_close_intent_for_alignment:
            return True
        return bool(getattr(controller, "_alignment_planner_close_intent", False))

    def _current_step_wants_close(self, a_base_7d=None) -> bool:
        if a_base_7d is None:
            return False
        return bool(float(np.asarray(a_base_7d, dtype=np.float32)[6]) <= self.alignment_close_command_threshold)

    def _alignment_gate_decision(self, depth_proximity, gripper_open, a_base_7d=None, future_gripper_actions=None, controller=None, step_idx: Optional[int] = None) -> dict:
        planner_close_intent = self._planner_close_intent(a_base_7d, future_gripper_actions)
        current_step_close = self._current_step_wants_close(a_base_7d)
        if not self.require_pregrasp_alignment_gate:
            near_target = True
            gripper_still_open = True
            close_requirement_satisfied = True
        else:
            near_target = (
                depth_proximity is not None
                and np.isfinite(depth_proximity)
                and float(depth_proximity) < self.alignment_depth_threshold
            )
            gripper_still_open = (
                gripper_open is not None
                and float(gripper_open) >= self.alignment_open_threshold
            )
            close_requirement_satisfied = planner_close_intent or (not self.require_close_intent_for_alignment)

        support_satisfied = True
        support_inner_satisfied = True
        support_outer_satisfied = True
        use_outer_rescue = False
        close_ready_for_planner = False
        refine_band_satisfied = False
        takeover_band_satisfied = False
        if self._phase1_bridge_controller_capable(controller):
            if self._runtime_has_motion_target(controller):
                cur_dist = float(getattr(controller, "_runtime_current_basin_distance", np.inf))
                support_max = float(getattr(controller, "_support_basin_distance_max", np.inf))
                current_delta = np.asarray(
                    getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                    dtype=np.float32,
                )
                support_inner_satisfied = self._delta_within_band(current_delta, controller, band="inner")
                support_outer_satisfied = self._delta_within_band(current_delta, controller, band="outer")
                support_satisfied = bool(support_inner_satisfied)
                use_outer_rescue = bool(
                    self.enable_outer_rescue
                    and support_outer_satisfied
                    and (not support_inner_satisfied)
                )
                close_ready_for_planner = bool(
                    gripper_still_open
                    and near_target
                    and (
                        (
                            current_step_close
                            and self._close_veto_ready(
                                controller,
                                gripper_open,
                                step_idx=step_idx,
                                depth_proximity=depth_proximity,
                            )
                        )
                        or (
                            self.skip_alignment_when_close_ready
                            and self._alignment_skip_ready(controller, gripper_open, depth_proximity)
                        )
                    )
                )
                refine_band_satisfied = bool(
                    gripper_still_open
                    and near_target
                    and close_requirement_satisfied
                    and self._alignment_refine_band_ready(controller, gripper_open)
                    and (not self._close_veto_ready(controller, gripper_open, step_idx=step_idx, depth_proximity=depth_proximity))
                )
                takeover_band_satisfied = bool(
                    gripper_still_open
                    and near_target
                    and close_requirement_satisfied
                    and self._alignment_takeover_band_ready(controller, gripper_open)
                    and (not self._close_veto_ready(controller, gripper_open, step_idx=step_idx, depth_proximity=depth_proximity))
                )
            else:
                cur_dist = np.inf
                support_max = np.inf
                support_satisfied = True
                support_inner_satisfied = True
                support_outer_satisfied = False
                use_outer_rescue = False
                close_ready_for_planner = False
                refine_band_satisfied = False
                takeover_band_satisfied = False

        alignment_window_active = bool(
            near_target
            and gripper_still_open
            and (
                (
                    close_requirement_satisfied
                    and (support_inner_satisfied or use_outer_rescue)
                )
                or refine_band_satisfied
            )
            and (not close_ready_for_planner)
        )
        short_window_available = bool(
            use_outer_rescue
            or (self._alignment_window_corrections < self.max_alignment_corrections_per_window)
        )
        gate_open = bool(alignment_window_active and short_window_available)
        blocked_reason = "none"
        if not gate_open:
            if not near_target:
                blocked_reason = "depth"
            elif not gripper_still_open:
                blocked_reason = "gripper_open"
            elif not close_requirement_satisfied:
                blocked_reason = "close_intent"
            elif close_ready_for_planner:
                blocked_reason = "ready_to_close"
            elif not (support_inner_satisfied or use_outer_rescue):
                blocked_reason = "support"
            elif not short_window_available:
                blocked_reason = "window"
        if not alignment_window_active:
            # Reset the learned-support budget whenever we leave the useful open-phase
            # alignment band. Outer rescue has its own path and must not consume this.
            self._alignment_window_corrections = 0
        decision = {
            "gate_open": gate_open,
            "blocked_reason": blocked_reason,
            "depth_proximity": None if depth_proximity is None else float(depth_proximity),
            "near_target": bool(near_target),
            "gripper_open": None if gripper_open is None else float(gripper_open),
            "gripper_still_open": bool(gripper_still_open),
            "planner_close_intent": bool(planner_close_intent),
            "close_requirement_satisfied": bool(close_requirement_satisfied),
            "alignment_window_active": bool(alignment_window_active),
            "support_satisfied": bool(support_satisfied),
            "support_inner_satisfied": bool(support_inner_satisfied),
            "support_outer_satisfied": bool(support_outer_satisfied),
            "use_outer_rescue": bool(use_outer_rescue),
            "close_ready_for_planner": bool(close_ready_for_planner),
            "refine_band_satisfied": bool(refine_band_satisfied),
            "takeover_band_satisfied": bool(takeover_band_satisfied),
            "support_basin_distance_satisfied": bool(cur_dist <= support_max) if getattr(controller, "_controller_type", "") == "pose_field_scorer" else True,
            "short_window_available": bool(short_window_available),
            "residual_cooldown": int(self._residual_cooldown),
        }
        self._last_alignment_gate_debug = decision
        return decision

    def _try_v2_predictor_micro_assist(
        self,
        outputs: dict,
        a_exec: np.ndarray,
        current_quat: np.ndarray,
    ) -> bool:
        """Apply a very small additive assist from the predictor-driven v2 scorer.

        This path is intentionally separate from legacy v2 shadow/takeover logic:
        it only activates in near/micro geometry, keeps planner action intact, and
        uses the target_delta_predictor source contract as the only signed delta.
        """
        self._v2_predictor_micro_assist_eval_count += 1
        self._last_v2_predictor_micro_assist_applied = False
        self._last_v2_predictor_micro_assist_block_reason = "disabled"
        self._last_v2_apply_gate_pass = False
        self._last_v2_apply_block_reason = "disabled"
        self._last_v2_apply_assist_scale = 0.0
        self._last_v2_apply_local_delta = None
        self._last_v2_apply_world_delta = None

        if not self.enable_v2_predictor_micro_assist_apply:
            self._last_v2_predictor_micro_assist_block_reason = "disabled"
            self._last_v2_apply_block_reason = "disabled"
            return False
        if self.manager.phase != StagePhase.ALIGN:
            self._last_v2_predictor_micro_assist_block_reason = "phase"
            self._last_v2_apply_block_reason = "phase"
            return False
        if self._last_nz_blocked_zone:
            self._last_v2_predictor_micro_assist_block_reason = "near_zone_block"
            self._last_v2_apply_block_reason = "near_zone_block"
            return False
        if self._alignment_zone_state == "takeover":
            self._last_v2_predictor_micro_assist_block_reason = "takeover"
            self._last_v2_apply_block_reason = "takeover"
            return False
        if str(outputs.get("v2_delta_source", "none")) != "target_delta_predictor":
            self._last_v2_predictor_micro_assist_block_reason = "source"
            self._last_v2_apply_block_reason = "source"
            return False
        v2_gate_pass = bool(outputs.get("v2_gate_pass", False))
        if self.v2_predictor_micro_assist_only_when_gate_pass and not v2_gate_pass:
            self._last_v2_predictor_micro_assist_block_reason = "v2_gate"
            self._last_v2_apply_block_reason = "v2_gate"
            return False

        bucket = str(outputs.get("v2_stage_bucket", "unknown"))
        if bucket not in ("near_alignment", "micro_contact_refine"):
            self._last_v2_predictor_micro_assist_block_reason = "bucket"
            self._last_v2_apply_block_reason = "bucket"
            return False

        v2_selected = outputs.get("v2_selected_delta", None)
        if v2_selected is None:
            v2_selected = getattr(self, "_last_v2_selected_delta", None)
        if v2_selected is None:
            self._last_v2_predictor_micro_assist_block_reason = "missing_selected"
            self._last_v2_apply_block_reason = "missing_selected"
            return False
        v2_selected = np.asarray(v2_selected, dtype=np.float32).reshape(-1)
        if v2_selected.size < 6:
            self._last_v2_predictor_micro_assist_block_reason = "bad_selected"
            self._last_v2_apply_block_reason = "bad_selected"
            return False

        cur_xy = outputs.get("v2_cur_xy", None)
        cur_z = outputs.get("v2_cur_z", None)
        cur_yaw = outputs.get("v2_cur_yaw", None)
        post_xy = outputs.get("v2_selected_post_xy", None)
        if post_xy is None:
            post_xy = getattr(self, "_last_v2_selected_post_xy", None)
        post_z = outputs.get("v2_selected_post_z", None)
        if post_z is None:
            post_z = getattr(self, "_last_v2_selected_post_z", None)
        post_yaw = outputs.get("v2_selected_post_yaw", None)
        if post_yaw is None:
            post_yaw = getattr(self, "_last_v2_selected_post_yaw", None)
        if cur_xy is None or cur_z is None or cur_yaw is None:
            self._last_v2_predictor_micro_assist_block_reason = "missing_cur"
            self._last_v2_apply_block_reason = "missing_cur"
            return False
        if post_xy is None or post_z is None:
            self._last_v2_predictor_micro_assist_block_reason = "missing_post"
            self._last_v2_apply_block_reason = "missing_post"
            return False

        cur_xy = float(cur_xy)
        cur_z = float(cur_z)
        cur_yaw = float(cur_yaw)
        post_xy = float(post_xy)
        post_z = float(post_z)
        post_yaw_f = None if post_yaw is None else float(post_yaw)
        if not (post_xy < cur_xy and post_z < cur_z):
            self._last_v2_predictor_micro_assist_block_reason = "no_improve"
            self._last_v2_apply_block_reason = "no_improve"
            return False
        if post_yaw_f is not None and post_yaw_f > cur_yaw + 0.03:
            self._last_v2_predictor_micro_assist_block_reason = "yaw_worse"
            self._last_v2_apply_block_reason = "yaw_worse"
            return False

        assist_local = np.asarray(v2_selected, dtype=np.float32).copy() * self.v2_predictor_micro_assist_scale_cap
        pos_norm = float(np.linalg.norm(assist_local[:3]))
        rot_norm = float(np.linalg.norm(assist_local[3:6]))
        if pos_norm > self.v2_predictor_micro_assist_max_pos:
            assist_local[:3] *= self.v2_predictor_micro_assist_max_pos / max(pos_norm, 1e-8)
        if rot_norm > self.v2_predictor_micro_assist_max_rot:
            assist_local[3:6] *= self.v2_predictor_micro_assist_max_rot / max(rot_norm, 1e-8)
        assist_world = local_delta_to_world(assist_local, current_quat)

        a_exec[:6] = a_exec[:6] + assist_world
        self._v2_predictor_micro_assist_apply_count += 1
        self._last_v2_predictor_micro_assist_applied = True
        self._last_v2_predictor_micro_assist_block_reason = "applied"
        self._last_v2_apply_gate_pass = True
        self._last_v2_apply_block_reason = "applied"
        self._last_v2_apply_assist_scale = float(self.v2_predictor_micro_assist_scale_cap)
        self._last_v2_apply_local_delta = assist_local.copy()
        self._last_v2_apply_world_delta = assist_world.copy()
        self._last_v2_predictor_micro_assist_pos_norm = float(np.linalg.norm(assist_local[:3]))
        self._last_v2_predictor_micro_assist_rot_norm = float(np.linalg.norm(assist_local[3:6]))
        self._last_v2_predictor_micro_assist_source = "target_delta_predictor"
        return True

    def _try_target_delta_servo(
        self,
        controller,
        a_exec: np.ndarray,
        current_quat: np.ndarray,
        *,
        apply: bool,
    ) -> bool:
        """Apply or shadow a conservative analytic servo from signed target delta."""
        self._target_delta_servo_eval_count += 1
        self._last_target_delta_servo_applied = False
        self._last_target_delta_servo_block_reason = "disabled"
        self._last_target_delta_servo_source = "none"
        self._last_target_delta_servo_local_delta = None
        self._last_target_delta_servo_world_delta = None
        self._last_target_delta_servo_pos_norm = 0.0
        self._last_target_delta_servo_rot_norm = 0.0
        self._last_target_delta_servo_cur_xy = None
        self._last_target_delta_servo_cur_z = None
        self._last_target_delta_servo_cur_yaw = None
        self._last_target_delta_servo_post_xy = None
        self._last_target_delta_servo_post_z = None
        self._last_target_delta_servo_post_yaw = None
        self._last_target_delta_servo_gate_pass = False
        if self.target_delta_servo_apply_once_per_episode and self._target_delta_servo_apply_count > 0:
            self._last_target_delta_servo_block_reason = "once_per_episode"
            return False

        if not (self.enable_target_delta_servo_shadow or self.enable_target_delta_servo_apply):
            self._last_target_delta_servo_block_reason = "disabled"
            return False
        if not self.target_delta_servo_bypass_gates and self.manager.phase != StagePhase.ALIGN:
            self._last_target_delta_servo_block_reason = "phase"
            return False
        if not self.target_delta_servo_bypass_gates and getattr(controller, "_controller_type", "") != "pose_field_scorer":
            self._last_target_delta_servo_block_reason = "controller"
            return False
        if not self.target_delta_servo_bypass_gates and self._last_nz_blocked_zone:
            self._last_target_delta_servo_block_reason = "near_zone_block"
            return False
        if not self.target_delta_servo_bypass_gates and self._alignment_zone_state == "takeover":
            self._last_target_delta_servo_block_reason = "takeover"
            return False
        if not self.target_delta_servo_bypass_gates and self._residual_cooldown > 0:
            self._last_target_delta_servo_block_reason = "cooldown"
            return False

        source_mode = str(self.target_delta_servo_source).strip().lower()
        runtime_delta = getattr(controller, "_runtime_motion_target_delta_local", None)
        runtime_delta_source = str(getattr(controller, "_runtime_motion_target_delta_source", "none"))
        if runtime_delta is None:
            self._last_target_delta_servo_block_reason = "missing_delta"
            return False
        runtime_delta = np.asarray(runtime_delta, dtype=np.float32).reshape(-1)
        if runtime_delta.size < 6 or not np.all(np.isfinite(runtime_delta[:6])):
            self._last_target_delta_servo_block_reason = "bad_delta"
            return False

        if source_mode == "privileged_replay":
            privileged_delta = getattr(controller, "_runtime_teacher_current_delta_basin_target", None)
            if privileged_delta is None:
                self._last_target_delta_servo_block_reason = "missing_privileged_delta"
                return False
            servo_source = "privileged_replay"
            servo_delta = np.asarray(privileged_delta, dtype=np.float32).reshape(-1)[:6].copy()
        elif source_mode == "predictor":
            if "predictor" not in runtime_delta_source and "learned_target" not in runtime_delta_source:
                self._last_target_delta_servo_block_reason = "source"
                return False
            servo_source = "predictor_delta"
            servo_delta = runtime_delta[:6].copy()
        elif source_mode in ("basin_diagnostic", "basin", "canonical"):
            if not any(tok in runtime_delta_source for tok in ("canonical", "basin", "fallback")):
                self._last_target_delta_servo_block_reason = "source"
                return False
            servo_source = "basin_diagnostic"
            servo_delta = runtime_delta[:6].copy()
        elif source_mode in ("zero_diagnostic", "zero"):
            servo_source = "zero_diagnostic"
            servo_delta = np.zeros(6, dtype=np.float32)
        else:
            servo_source = runtime_delta_source if runtime_delta_source != "none" else "runtime_motion_target_delta"
            servo_delta = runtime_delta[:6].copy()

        cur_xy = float(np.linalg.norm(servo_delta[:2]))
        cur_z = float(abs(servo_delta[2]))
        cur_yaw = float(abs(servo_delta[5]))
        self._last_target_delta_servo_source = str(servo_source)
        self._last_target_delta_servo_cur_xy = cur_xy
        self._last_target_delta_servo_cur_z = cur_z
        self._last_target_delta_servo_cur_yaw = cur_yaw

        if (
            not self.target_delta_servo_bypass_gates
            and (cur_xy > self.target_delta_servo_apply_xy_threshold or cur_z > self.target_delta_servo_apply_abs_z_threshold)
        ):
            self._last_target_delta_servo_block_reason = "window"
            return False
        if (
            not self.target_delta_servo_bypass_gates
            and self.target_delta_servo_apply_yaw_threshold > 0.0
            and cur_yaw > self.target_delta_servo_apply_yaw_threshold
        ):
            self._last_target_delta_servo_block_reason = "yaw_window"
            return False

        servo_local = np.zeros(6, dtype=np.float32)
        servo_local[:2] = servo_delta[:2] * float(self.target_delta_servo_k_xy)
        servo_local[2] = float(servo_delta[2]) * float(self.target_delta_servo_k_z)
        servo_local[5] = float(servo_delta[5]) * float(self.target_delta_servo_k_yaw)
        preclip_local = servo_local.copy()
        pos_norm = float(np.linalg.norm(servo_local[:3]))
        rot_norm = float(abs(servo_local[5]))
        if pos_norm > self.target_delta_servo_max_pos and pos_norm > 1e-8:
            servo_local[:3] *= float(self.target_delta_servo_max_pos / pos_norm)
        if abs(servo_local[5]) > self.target_delta_servo_max_yaw and abs(servo_local[5]) > 1e-8:
            servo_local[5] *= float(self.target_delta_servo_max_yaw / abs(servo_local[5]))

        post_delta = servo_delta[:6] - servo_local
        post_xy = float(np.linalg.norm(post_delta[:2]))
        post_z = float(abs(post_delta[2]))
        post_yaw = float(abs(post_delta[5]))
        self._last_target_delta_servo_post_xy = post_xy
        self._last_target_delta_servo_post_z = post_z
        self._last_target_delta_servo_post_yaw = post_yaw
        self._last_target_delta_servo_local_delta = np.asarray(servo_local, dtype=np.float32).copy()
        self._last_target_delta_servo_pos_norm = float(np.linalg.norm(servo_local[:3]))
        self._last_target_delta_servo_rot_norm = float(abs(servo_local[5]))
        if not self.target_delta_servo_bypass_gates and not (post_xy < cur_xy and post_z < cur_z):
            self._last_target_delta_servo_block_reason = "no_improve"
            return False
        if not self.target_delta_servo_bypass_gates and post_yaw > cur_yaw + 0.03:
            self._last_target_delta_servo_block_reason = "yaw_worse"
            return False

        self._last_target_delta_servo_gate_pass = True
        servo_world = local_delta_to_world(servo_local, current_quat)
        self._last_target_delta_servo_world_delta = np.asarray(servo_world, dtype=np.float32).copy()
        if not apply:
            self._last_target_delta_servo_block_reason = "shadow"
            return False

        a_exec[:6] = np.asarray(a_exec[:6], dtype=np.float32) + servo_world
        self._target_delta_servo_apply_count += 1
        self._last_target_delta_servo_applied = True
        self._last_target_delta_servo_block_reason = "applied"
        return True

    def _try_alignment_v3_shadow(
        self,
        controller,
        front_rgb,
        wrist_rgb,
        wrist_depth,
        ft_hist,
        proprio,
        planner_action_local,
        current_delta_local,
        step_idx: int,
        gripper_context=None,
    ) -> bool:
        """Run the v3 direct local controller in shadow-only mode.

        This path never changes the executed action; it only records whether
        the v3 controller produces a coherent near/micro direct residual on the
        live trajectory.
        """
        self._alignment_v3_shadow_eval_count += 1
        self._last_alignment_v3_shadow_active = False
        self._last_alignment_v3_shadow_source = "none"
        self._last_alignment_v3_shadow_block_reason = "disabled"
        self._last_alignment_v3_shadow_cur_xy = None
        self._last_alignment_v3_shadow_cur_z = None
        self._last_alignment_v3_shadow_cur_yaw = None
        self._last_alignment_v3_shadow_post_xy = None
        self._last_alignment_v3_shadow_post_z = None
        self._last_alignment_v3_shadow_post_yaw = None
        self._last_alignment_v3_shadow_xy_improved = False
        self._last_alignment_v3_shadow_z_improved = False
        self._last_alignment_v3_shadow_yaw_improved = False
        self._last_alignment_v3_shadow_all_improved = False
        self._last_alignment_v3_shadow_pred_residual_4d = None
        self._last_alignment_v3_shadow_pred_residual_6d = None
        self._last_alignment_v3_shadow_pred_pos_norm = 0.0
        self._last_alignment_v3_shadow_pred_yaw_abs = 0.0
        self._last_alignment_v3_shadow_risk_logit = 0.0
        self._last_alignment_v3_shadow_confidence_logit = 0.0

        shadow_controller = getattr(self, "alignment_v3_shadow_controller", None)
        if shadow_controller is None:
            self._last_alignment_v3_shadow_block_reason = "disabled"
            return False
        if self.manager.phase != StagePhase.ALIGN:
            self._last_alignment_v3_shadow_block_reason = "phase"
            return False

        current_delta = np.asarray(
            current_delta_local
            if current_delta_local is not None
            else getattr(controller, "_runtime_motion_target_delta_local", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if current_delta.size < 6 or not np.all(np.isfinite(current_delta[:6])):
            self._last_alignment_v3_shadow_block_reason = "bad_delta"
            return False

        cur_xy = float(np.linalg.norm(current_delta[:2]))
        cur_z = float(abs(current_delta[2]))
        cur_yaw = float(abs(current_delta[5]))
        self._last_alignment_v3_shadow_source = str(getattr(controller, "_runtime_motion_target_delta_source", "none"))
        self._last_alignment_v3_shadow_cur_xy = cur_xy
        self._last_alignment_v3_shadow_cur_z = cur_z
        self._last_alignment_v3_shadow_cur_yaw = cur_yaw

        shadow_gate_pass = bool(
            cur_xy <= float(self.alignment_near_zone_xy_threshold)
            and cur_z <= float(self.alignment_near_zone_z_threshold)
        )
        if shadow_gate_pass:
            self._alignment_v3_shadow_gate_pass_count += 1
        self._last_alignment_v3_shadow_active = True

        import torch

        device = next(shadow_controller.parameters()).device
        dtype = next(shadow_controller.parameters()).dtype
        wd = self._to_tensor(wrist_depth, (1, 1, 96, 96), device, dtype)
        fh = self._to_tensor(ft_hist, (1, 32, 6), device, dtype)
        pr = self._to_tensor(proprio, (1, 15), device, dtype)
        ba = self._to_tensor(planner_action_local, (1, 6), device, dtype)
        cur = torch.from_numpy(current_delta[:6]).unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            out = shadow_controller(
                wrist_depth=wd,
                force_history=fh,
                proprio=pr,
                planner_action_local=ba,
                current_to_target_delta_local=cur,
            )

        pred_residual_4d = out.get("direct_residual_4d", None)
        pred_residual_6d = out.get("direct_residual_6d", None)
        post_xyz_yaw = out.get("shadow_post_xyz_yaw", None)
        risk_logit = out.get("risk_logit", None)
        conf_logit = out.get("confidence_logit", None)

        if pred_residual_4d is not None:
            pred_residual_4d_np = pred_residual_4d.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            self._last_alignment_v3_shadow_pred_residual_4d = pred_residual_4d_np.tolist()
            self._last_alignment_v3_shadow_pred_pos_norm = float(np.linalg.norm(pred_residual_4d_np[:3]))
            self._last_alignment_v3_shadow_pred_yaw_abs = float(abs(pred_residual_4d_np[3]))
        if pred_residual_6d is not None:
            pred_residual_6d_np = pred_residual_6d.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            self._last_alignment_v3_shadow_pred_residual_6d = pred_residual_6d_np.tolist()
        if post_xyz_yaw is not None:
            post_np = post_xyz_yaw.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            # The v3 post head is trained to emit scalar errors
            # [post_xy, post_z, post_yaw], not an xyz/yaw vector.
            self._last_alignment_v3_shadow_post_xy = float(abs(post_np[0])) if post_np.size >= 1 else None
            self._last_alignment_v3_shadow_post_z = float(abs(post_np[1])) if post_np.size >= 2 else None
            self._last_alignment_v3_shadow_post_yaw = float(abs(post_np[2])) if post_np.size >= 3 else None
            self._last_alignment_v3_shadow_xy_improved = bool(self._last_alignment_v3_shadow_post_xy < cur_xy)
            self._last_alignment_v3_shadow_z_improved = bool(self._last_alignment_v3_shadow_post_z < cur_z)
            if self._last_alignment_v3_shadow_post_yaw is not None:
                self._last_alignment_v3_shadow_yaw_improved = bool(self._last_alignment_v3_shadow_post_yaw < cur_yaw)
            self._last_alignment_v3_shadow_all_improved = bool(
                self._last_alignment_v3_shadow_xy_improved
                and self._last_alignment_v3_shadow_z_improved
                and self._last_alignment_v3_shadow_yaw_improved
            )
            if self._last_alignment_v3_shadow_all_improved:
                self._alignment_v3_shadow_all_improve_count += 1
        if self._last_alignment_v3_shadow_xy_improved or self._last_alignment_v3_shadow_z_improved:
            self._alignment_v3_shadow_improve_count += 1
        if risk_logit is not None:
            self._last_alignment_v3_shadow_risk_logit = float(risk_logit.squeeze(0).detach().float().cpu().item())
        if conf_logit is not None:
            self._last_alignment_v3_shadow_confidence_logit = float(conf_logit.squeeze(0).detach().float().cpu().item())

        self._last_alignment_v3_shadow_block_reason = "gate_pass" if shadow_gate_pass else "shadow_only"
        return True

    def _try_alignment_v4_shadow(
        self,
        controller,
        front_rgb,
        wrist_depth,
        ft_hist,
        proprio,
        base_action_local,
        current_delta_local,
        step_idx,
        gripper_context=None,
    ) -> bool:
        self._alignment_v4_shadow_eval_count += 1
        self._last_alignment_v4_shadow_active = False
        self._last_alignment_v4_shadow_source = "none"
        self._last_alignment_v4_shadow_block_reason = "disabled"
        self._last_alignment_v4_shadow_cur_xy = None
        self._last_alignment_v4_shadow_cur_z = None
        self._last_alignment_v4_shadow_cur_yaw = None
        self._last_alignment_v4_shadow_post_xy = None
        self._last_alignment_v4_shadow_post_z = None
        self._last_alignment_v4_shadow_post_yaw = None
        self._last_alignment_v4_shadow_xy_improved = False
        self._last_alignment_v4_shadow_z_improved = False
        self._last_alignment_v4_shadow_yaw_improved = False
        self._last_alignment_v4_shadow_all_improved = False
        self._last_alignment_v4_shadow_pred_residual_4d = None
        self._last_alignment_v4_shadow_pred_residual_6d = None
        self._last_alignment_v4_shadow_pred_post_xyz_yaw = None
        self._last_alignment_v4_shadow_pred_reduction_xyz = None
        self._last_alignment_v4_shadow_pred_pos_norm = 0.0
        self._last_alignment_v4_shadow_pred_yaw_abs = 0.0
        self._last_alignment_v4_shadow_risk_logit = 0.0
        self._last_alignment_v4_shadow_confidence_logit = 0.0
        self._last_alignment_v4_shadow_policy_mode = "unknown"
        self._last_alignment_v4_shadow_stage_bucket = "unknown"
        self._last_alignment_v4_shadow_micro_gate = False

        shadow_controller = getattr(self, "alignment_v4_shadow_controller", None)
        if shadow_controller is None:
            self._last_alignment_v4_shadow_block_reason = "disabled"
            return False
        if self.manager.phase != StagePhase.ALIGN:
            self._last_alignment_v4_shadow_block_reason = "phase"
            return False

        current_delta = np.asarray(
            current_delta_local
            if current_delta_local is not None
            else getattr(controller, "_runtime_motion_target_delta_local", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if current_delta.size < 6 or not np.all(np.isfinite(current_delta[:6])):
            self._last_alignment_v4_shadow_block_reason = "bad_delta"
            return False

        cur_xy = float(np.linalg.norm(current_delta[:2]))
        cur_z = float(abs(current_delta[2]))
        cur_yaw = float(abs(current_delta[5]))
        self._last_alignment_v4_shadow_source = str(getattr(controller, "_runtime_motion_target_delta_source", "none"))
        self._last_alignment_v4_shadow_cur_xy = cur_xy
        self._last_alignment_v4_shadow_cur_z = cur_z
        self._last_alignment_v4_shadow_cur_yaw = cur_yaw

        if cur_xy < 0.06 and cur_z < 0.12 and cur_yaw < 0.30:
            if cur_xy < 0.020 and cur_z < 0.05 and cur_yaw < 0.16:
                stage_bucket = "micro_contact_refine"
            else:
                stage_bucket = "near_alignment"
        elif cur_xy < 0.16 and cur_z < 0.30:
            stage_bucket = "mid_approach_assist"
        else:
            stage_bucket = "far_coarse_approach"
        self._last_alignment_v4_shadow_stage_bucket = stage_bucket
        self._last_alignment_v4_shadow_micro_gate = bool(stage_bucket == "micro_contact_refine")

        if self.alignment_v4_shadow_micro_only and stage_bucket != "micro_contact_refine":
            self._last_alignment_v4_shadow_block_reason = "not_micro"
            return False

        shadow_gate_pass = bool(
            cur_xy <= float(self.alignment_near_zone_xy_threshold)
            and cur_z <= float(self.alignment_near_zone_z_threshold)
        )

        self._last_alignment_v4_shadow_active = True

        import torch

        device = next(shadow_controller.parameters()).device
        dtype = next(shadow_controller.parameters()).dtype
        fr = self._to_rgb_tensor(front_rgb, (1, 3, 128, 128), device, dtype) if front_rgb is not None else None
        wd = self._to_tensor(wrist_depth, (1, 1, 96, 96), device, dtype)
        fh = self._to_tensor(ft_hist, (1, 32, 6), device, dtype)
        pr = self._to_tensor(proprio, (1, 15), device, dtype)
        ba = self._to_tensor(base_action_local, (1, 6), device, dtype)
        cur = torch.from_numpy(current_delta[:6]).unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            out = shadow_controller(
                front_rgb=fr,
                wrist_depth=wd,
                force_history=fh,
                proprio=pr,
                planner_action_local=ba,
                current_to_target_delta_local=cur,
            )

        pred_residual_4d = out.get("direct_residual_4d", None)
        pred_residual_6d = out.get("direct_residual_6d", None)
        pred_post = out.get("shadow_post_xyz_yaw", None)
        pred_reduction = out.get("shadow_delta_reduction", None)
        risk_logit = out.get("risk_logit", None)
        conf_logit = out.get("confidence_logit", None)
        policy_mode = out.get("policy_mode_logits", None)

        if pred_residual_4d is not None:
            pred_residual_4d_np = pred_residual_4d.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            self._last_alignment_v4_shadow_pred_residual_4d = pred_residual_4d_np.tolist()
            self._last_alignment_v4_shadow_pred_pos_norm = float(np.linalg.norm(pred_residual_4d_np[:3]))
            self._last_alignment_v4_shadow_pred_yaw_abs = float(abs(pred_residual_4d_np[3]))
        if pred_residual_6d is not None:
            pred_residual_6d_np = pred_residual_6d.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            self._last_alignment_v4_shadow_pred_residual_6d = pred_residual_6d_np.tolist()
        if pred_post is not None:
            pred_post_np = pred_post.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            self._last_alignment_v4_shadow_pred_post_xyz_yaw = pred_post_np.tolist()
            self._last_alignment_v4_shadow_post_xy = float(abs(pred_post_np[0])) if pred_post_np.size >= 1 else None
            self._last_alignment_v4_shadow_post_z = float(abs(pred_post_np[1])) if pred_post_np.size >= 2 else None
            self._last_alignment_v4_shadow_post_yaw = float(abs(pred_post_np[2])) if pred_post_np.size >= 3 else None
            self._last_alignment_v4_shadow_xy_improved = bool(self._last_alignment_v4_shadow_post_xy < cur_xy)
            self._last_alignment_v4_shadow_z_improved = bool(self._last_alignment_v4_shadow_post_z < cur_z)
            if self._last_alignment_v4_shadow_post_yaw is not None:
                self._last_alignment_v4_shadow_yaw_improved = bool(self._last_alignment_v4_shadow_post_yaw < cur_yaw)
            self._last_alignment_v4_shadow_all_improved = bool(
                self._last_alignment_v4_shadow_xy_improved
                and self._last_alignment_v4_shadow_z_improved
                and self._last_alignment_v4_shadow_yaw_improved
            )
            if self._last_alignment_v4_shadow_all_improved:
                self._alignment_v4_shadow_all_improve_count += 1
        if pred_reduction is not None:
            pred_reduction_np = pred_reduction.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
            self._last_alignment_v4_shadow_pred_reduction_xyz = pred_reduction_np.tolist()
        if self._last_alignment_v4_shadow_xy_improved or self._last_alignment_v4_shadow_z_improved:
            self._alignment_v4_shadow_improve_count += 1
        if risk_logit is not None:
            self._last_alignment_v4_shadow_risk_logit = float(risk_logit.squeeze(0).detach().float().cpu().item())
        if conf_logit is not None:
            self._last_alignment_v4_shadow_confidence_logit = float(conf_logit.squeeze(0).detach().float().cpu().item())
        if policy_mode is not None:
            policy_idx = int(torch.argmax(policy_mode.squeeze(0), dim=-1).detach().cpu().item())
            self._last_alignment_v4_shadow_policy_mode = {0: "noop", 1: "assist", 2: "hold"}.get(policy_idx, "unknown")

        self._last_alignment_v4_shadow_block_reason = "gate_pass" if shadow_gate_pass else "shadow_only"
        return True

    def _try_alignment_v4_apply(self, a_exec: np.ndarray, current_quat: np.ndarray) -> bool:
        """Apply the latest v4 direct-local residual as a tiny diagnostic assist."""
        self._last_alignment_v4_apply_applied = False
        self._last_alignment_v4_apply_block_reason = "disabled"
        self._last_alignment_v4_apply_local_delta = None
        self._last_alignment_v4_apply_world_delta = None
        self._last_alignment_v4_apply_pos_norm = 0.0
        self._last_alignment_v4_apply_yaw_abs = 0.0
        self._last_alignment_v4_apply_stage_bucket = "unknown"
        self._last_alignment_v4_apply_micro_gate = False

        if not self.enable_alignment_v4_apply:
            return False
        if self.alignment_v4_shadow_controller is None:
            self._last_alignment_v4_apply_block_reason = "no_controller"
            return False
        if not self._last_alignment_v4_shadow_active:
            self._last_alignment_v4_apply_block_reason = "shadow_inactive"
            return False
        stage_bucket = str(getattr(self, "_last_alignment_v4_shadow_stage_bucket", "unknown"))
        self._last_alignment_v4_apply_stage_bucket = stage_bucket
        self._last_alignment_v4_apply_micro_gate = bool(stage_bucket == "micro_contact_refine")
        if self.alignment_v4_apply_micro_only and stage_bucket != "micro_contact_refine":
            self._last_alignment_v4_apply_block_reason = "not_micro"
            return False
        if self.alignment_v4_apply_only_when_gate_pass:
            gate_ok = bool(
                self._last_alignment_v4_shadow_cur_xy is not None
                and self._last_alignment_v4_shadow_cur_z is not None
                and self._last_alignment_v4_shadow_cur_xy <= self.alignment_near_zone_xy_threshold
                and self._last_alignment_v4_shadow_cur_z <= self.alignment_near_zone_z_threshold
            )
            if not gate_ok:
                self._last_alignment_v4_apply_block_reason = "gate"
                return False
        if self.alignment_v4_apply_require_improve and not (
            self._last_alignment_v4_shadow_xy_improved
            and self._last_alignment_v4_shadow_z_improved
            and self._last_alignment_v4_shadow_yaw_improved
        ):
            self._last_alignment_v4_apply_block_reason = "pred_no_improve"
            return False

        pred = self._last_alignment_v4_shadow_pred_residual_6d
        if pred is None:
            self._last_alignment_v4_apply_block_reason = "no_residual"
            return False
        local_delta = np.asarray(pred, dtype=np.float32).reshape(-1)
        if local_delta.size < 6 or not np.all(np.isfinite(local_delta[:6])):
            self._last_alignment_v4_apply_block_reason = "bad_residual"
            return False
        local_delta = local_delta[:6].copy() * float(self.alignment_v4_apply_scale)

        pos_norm = float(np.linalg.norm(local_delta[:3]))
        if pos_norm > self.alignment_v4_apply_max_pos and pos_norm > 1e-8:
            local_delta[:3] *= float(self.alignment_v4_apply_max_pos / pos_norm)
        yaw_abs = float(abs(local_delta[5]))
        if yaw_abs > self.alignment_v4_apply_max_yaw and yaw_abs > 1e-8:
            local_delta[5] *= float(self.alignment_v4_apply_max_yaw / yaw_abs)

        if float(np.linalg.norm(local_delta[:3])) <= 1e-8 and float(abs(local_delta[5])) <= 1e-8:
            self._last_alignment_v4_apply_block_reason = "zero"
            return False

        world_delta = local_delta_to_world(local_delta, current_quat)
        a_exec[:6] = np.asarray(a_exec[:6], dtype=np.float32) + world_delta
        self._alignment_v4_apply_count += 1
        self._last_alignment_v4_apply_applied = True
        self._last_alignment_v4_apply_block_reason = "applied"
        self._last_alignment_v4_apply_local_delta = np.asarray(local_delta, dtype=np.float32).copy()
        self._last_alignment_v4_apply_world_delta = np.asarray(world_delta, dtype=np.float32).copy()
        self._last_alignment_v4_apply_pos_norm = float(np.linalg.norm(local_delta[:3]))
        self._last_alignment_v4_apply_yaw_abs = float(abs(local_delta[5]))
        return True

    def _reset_alignment_diffusion_last(self, reason: str = "disabled") -> None:
        self._last_alignment_diffusion_enabled = bool(
            (
                self.alignment_diffusion_controller is not None
                and (self.enable_alignment_diffusion_shadow or self.enable_alignment_diffusion_apply)
            )
            or (
                self.alignment_tc_diffusion_controller is not None
                and (self.enable_alignment_tc_diffusion_shadow or self.enable_alignment_tc_diffusion_apply)
            )
            or (
                self.alignment_tc_student_vnext_controller is not None
                and (self.enable_alignment_tc_student_vnext_shadow or self.enable_alignment_tc_student_vnext_apply)
            )
        )
        self._last_alignment_diffusion_active = False
        self._last_alignment_diffusion_applied = False
        self._last_alignment_diffusion_block_reason = str(reason)
        self._last_alignment_diffusion_trigger_mode = str(self.alignment_diffusion_trigger_mode)
        self._last_alignment_diffusion_phase_name = "unknown"
        self._last_alignment_diffusion_stage_bucket = "unknown"
        self._last_alignment_diffusion_safety_reject = False
        self._last_alignment_diffusion_selected_index = -1
        self._last_alignment_diffusion_num_samples = int(self.alignment_diffusion_num_samples)
        self._last_alignment_diffusion_candidate_diversity = 0.0
        self._last_alignment_diffusion_candidate_score = 0.0
        self._last_alignment_diffusion_risk_prob = 0.0
        self._last_alignment_diffusion_stop_prob = 0.0
        self._last_alignment_diffusion_progress_logits = None
        self._last_alignment_diffusion_first_residual_4d = None
        self._last_alignment_diffusion_first_residual_6d = None
        self._last_alignment_diffusion_local_delta = None
        self._last_alignment_diffusion_world_delta = None
        self._last_alignment_diffusion_pos_norm = 0.0
        self._last_alignment_diffusion_yaw_abs = 0.0
        self._last_alignment_diffusion_workspace_violation = 0.0
        self._last_alignment_diffusion_workspace_projected = False
        self._last_alignment_diffusion_workspace_project_reason = "none"
        self._last_alignment_diffusion_phase1_bridge_blend_active = False
        self._last_alignment_diffusion_phase1_bridge_blend_reason = "none"
        self._last_alignment_diffusion_phase1_bridge_blended_base_world = None
        self._last_alignment_diffusion_phase1_bridge_blended_base_local = None
        self._last_alignment_diffusion_phase1_bridge_basin_bias_local = None
        self._last_alignment_diffusion_phase1_bridge_taskspace_yaw_error = 0.0
        self._last_alignment_diffusion_phase1_bridge_taskspace_yaw_target_source = "none"
        self._last_alignment_diffusion_phase1_bridge_direct_yaw_rescue_local = None
        self._last_alignment_diffusion_phase1_bridge_yaw_holdoff_active = False
        self._last_alignment_diffusion_phase1_bridge_yaw_holdoff_reason = "none"
        self._phase1_force_reflex_active = False
        self._phase1_force_reflex_reason = "none"
        self._phase1_force_backoff_applied = False
        self._phase1_reopen_reason = "none"
        self._phase1_close_hold_active = False
        self._last_alignment_diffusion_controller_type = "none"
        self._last_alignment_diffusion_target_confidence = 0.0
        self._last_alignment_diffusion_pred_target_delta_6d = None
        self._last_alignment_diffusion_pred_target_delta_norm = 0.0
        self._last_alignment_diffusion_pred_target_yaw_abs = 0.0
        self._last_alignment_diffusion_target_action_sign_agreement = 0.0
        self._last_alignment_diffusion_low_confidence = False
        self._last_alignment_diffusion_soft_clamp = False
        self._last_alignment_diffusion_scale_down = 1.0
        self._last_alignment_diffusion_hard_reject = False
        self._last_alignment_diffusion_phase1_bridge_soft_apply = False
        self._last_alignment_diffusion_phase1_bridge_soft_apply_reason = "none"
        self._last_alignment_diffusion_latency_total_ms = 0.0
        self._last_alignment_diffusion_top_k = int(self.alignment_tc_diffusion_top_k)

    def _bump_alignment_diffusion_hist(self, hist: dict, key) -> None:
        k = str(key)
        hist[k] = int(hist.get(k, 0)) + 1

    def _alignment_diffusion_bucket(
        self,
        depth_proximity: Optional[float],
        force_reading: Optional[np.ndarray],
        base_action_local: np.ndarray,
    ) -> str:
        if bool(
            self.alignment_tc_student_vnext_controller is not None
            and (self.enable_alignment_tc_student_vnext_shadow or self.enable_alignment_tc_student_vnext_apply)
        ):
            phase_name = "insert_commit" if bool(getattr(self.manager, "has_object_in_hand", False)) else "grasp_commit"
            corridor = (self._alignment_tc_student_vnext_corridor or {}).get(phase_name, {})
            depth_p90 = float(corridor.get("depth_p90", self.alignment_depth_threshold))
            force_p90 = float(corridor.get("force_p90", self.safety.backoff_force_threshold * 0.5))
            action_p90 = float(corridor.get("planner_action_pos_norm_p90", 0.010))
            near_depth = bool(depth_proximity is not None and np.isfinite(depth_proximity) and float(depth_proximity) <= max(depth_p90 * 1.15, 0.015))
            micro_depth = bool(depth_proximity is not None and np.isfinite(depth_proximity) and float(depth_proximity) <= max(depth_p90 * 0.70, 0.010))
            force_norm = float(np.linalg.norm(force_reading[:3])) if force_reading is not None else 0.0
            base_pos_norm = float(np.linalg.norm(np.asarray(base_action_local, dtype=np.float32)[:3]))
            if micro_depth or force_norm >= max(force_p90, self.safety.backoff_force_threshold):
                return "micro_insert" if phase_name == "insert_commit" else "micro_contact_refine"
            if near_depth or base_pos_norm <= max(action_p90 * 1.25, 0.004):
                return "insert_near_align" if phase_name == "insert_commit" else "near_contact_refine"
            return "coarse"
        near_depth = bool(
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and float(depth_proximity) < self.alignment_depth_threshold
        )
        very_near_depth = bool(
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and float(depth_proximity) < max(self.alignment_depth_threshold * 0.5, 0.015)
        )
        force_norm = float(np.linalg.norm(force_reading[:3])) if force_reading is not None else 0.0
        base_pos_norm = float(np.linalg.norm(np.asarray(base_action_local, dtype=np.float32)[:3]))
        if force_norm >= self.safety.backoff_force_threshold or very_near_depth:
            return "micro_insert"
        if near_depth:
            return "near_precontact"
        if base_pos_norm <= 0.004:
            return "coarse_late_stall"
        return "coarse"

    def _alignment_diffusion_trigger_open(
        self,
        depth_proximity: Optional[float],
        force_reading: Optional[np.ndarray],
        base_action_local: np.ndarray,
        gripper_open: Optional[float],
    ) -> tuple[bool, str]:
        if self.manager.phase != StagePhase.ALIGN:
            return False, "phase"
        vnext_enabled = bool(
            self.alignment_tc_student_vnext_controller is not None
            and (self.enable_alignment_tc_student_vnext_shadow or self.enable_alignment_tc_student_vnext_apply)
        )
        if vnext_enabled:
            if bool(getattr(self.manager, "has_object_in_hand", False)):
                if gripper_open is not None and float(gripper_open) > self.alignment_open_threshold:
                    return False, "gripper_open_phase2"
            else:
                if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
                    return False, "gripper"
            bucket = self._alignment_diffusion_bucket(depth_proximity, force_reading, base_action_local)
            return (bucket != "coarse"), bucket
        if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
            return False, "gripper"
        bucket = self._alignment_diffusion_bucket(depth_proximity, force_reading, base_action_local)
        if self.alignment_diffusion_trigger_mode == "align_phase":
            return True, bucket
        if self.alignment_diffusion_trigger_mode == "depth_only":
            return (bucket in ("near_precontact", "micro_insert")), bucket
        return (bucket in ("near_precontact", "micro_insert", "coarse_late_stall")), bucket

    def _clip_alignment_diffusion_local(self, local_delta: np.ndarray) -> np.ndarray:
        out = np.asarray(local_delta, dtype=np.float32).reshape(-1)[:6].copy()
        pos_norm = float(np.linalg.norm(out[:3]))
        if pos_norm > self.alignment_diffusion_max_pos_step and pos_norm > 1e-8:
            out[:3] *= float(self.alignment_diffusion_max_pos_step / pos_norm)
        yaw_abs = float(abs(out[5]))
        if yaw_abs > self.alignment_diffusion_max_yaw_step and yaw_abs > 1e-8:
            out[5] *= float(self.alignment_diffusion_max_yaw_step / yaw_abs)
        out[3:5] = 0.0
        return out

    def _clip_alignment_diffusion_local_caps(
        self,
        local_delta: np.ndarray,
        *,
        max_pos: float,
        max_yaw: float,
    ) -> np.ndarray:
        out = np.asarray(local_delta, dtype=np.float32).reshape(-1)[:6].copy()
        pos_norm = float(np.linalg.norm(out[:3]))
        if pos_norm > max_pos and pos_norm > 1e-8:
            out[:3] *= float(max_pos / pos_norm)
        yaw_abs = float(abs(out[5]))
        if yaw_abs > max_yaw and yaw_abs > 1e-8:
            out[5] *= float(max_yaw / yaw_abs)
        out[3:5] = 0.0
        return out

    def _phase1_bridge_runtime_segment(self, bucket: Optional[str] = None) -> bool:
        if self.manager.phase != StagePhase.ALIGN:
            return False
        if bool(getattr(self.manager, "has_object_in_hand", False)):
            return False
        if getattr(self.manager, "substage", StageSubgoal.TRANSIT) != StageSubgoal.ALIGN_PREGRASP:
            return False
        if getattr(self.manager, "stage_target_mode", StageTargetMode.NONE) != StageTargetMode.PREGRASP_OBJECT:
            return False
        if int(getattr(self.manager, "contact_state", ContactState.FREE_SPACE)) < int(ContactState.NEAR_CONTACT):
            return False
        if bucket is not None and str(bucket) not in ("near_contact_refine", "micro_contact_refine"):
            return False
        return True

    def _phase1_bridge_controller_capable(self, controller) -> bool:
        if controller is None:
            return False
        controller_type = str(getattr(controller, "_controller_type", ""))
        if controller_type == "pose_field_scorer":
            return True
        if controller_type in (
            "alignment_tc_student_vnext",
            "alignment_tc_diffusion_refiner",
            "alignment_diffusion_refiner",
        ):
            return bool(
                self._runtime_has_motion_target(controller)
                or getattr(controller, "_runtime_current_delta_basin_target", None) is not None
                or getattr(controller, "_runtime_motion_target_pose_7d", None) is not None
            )
        return bool(
            self._runtime_has_motion_target(controller)
            or getattr(controller, "_runtime_current_delta_basin_target", None) is not None
        )

    def _student_vnext_ready_gate_supported(self, controller) -> bool:
        return bool(
            self.enable_alignment_tc_student_vnext_ready_gate
            and controller is not None
            and getattr(controller, "_controller_type", "") == "alignment_tc_student_vnext"
        )

    def _student_vnext_ready_gate_controller(self, controller):
        if not self._student_vnext_ready_gate_supported(controller):
            return controller
        controller_type = str(getattr(controller, "_controller_type", ""))
        if controller_type == "alignment_tc_student_vnext":
            return controller
        student_controller = getattr(self, "alignment_tc_student_vnext_controller", None)
        return student_controller if student_controller is not None else controller

    def _student_vnext_ready_gate_state(
        self,
        controller,
        gripper_open: Optional[float] = None,
        depth_proximity: Optional[float] = None,
    ) -> dict:
        if not self._student_vnext_ready_gate_supported(controller):
            return {
                "supported": False,
                "close_ready_pred": False,
                "close_ready_applied": False,
                "handoff_ready_pred": False,
                "handoff_ready_applied": False,
                "close_ready_prob": np.nan,
                "handoff_ready_prob": np.nan,
            "close_ready_logit": np.nan,
                "handoff_ready_logit": np.nan,
            }
        if depth_proximity is None:
            depth_proximity = getattr(self, "_last_alignment_gate_debug", {}).get("depth_proximity", None)
        controller = self._student_vnext_ready_gate_controller(controller)
        close_ready_prob = float(getattr(controller, "_runtime_close_ready_prob", np.nan))
        handoff_ready_prob = float(getattr(controller, "_runtime_handoff_ready_prob", np.nan))
        close_ready_logit = float(getattr(controller, "_runtime_close_ready_logit", np.nan))
        handoff_ready_logit = float(getattr(controller, "_runtime_handoff_ready_logit", np.nan))
        close_ready_pred = bool(getattr(controller, "_runtime_close_ready_pred", False))
        close_ready_applied = bool(getattr(controller, "_runtime_close_ready_applied", False))
        handoff_ready_pred = bool(getattr(controller, "_runtime_handoff_ready_pred", False))
        handoff_ready_applied = bool(getattr(controller, "_runtime_handoff_ready_applied", False))

        # Keep the learned heads, but only allow them to release the close /
        # handoff gates once the geometry is genuinely in the near-close band.
        # This prevents the early overconfident close-ready spikes from
        # claiming the gate while xy/z/yaw are still too loose.
        current_delta = self._phase1_force_current_delta(controller)
        geometry_close_ready = False
        geometry_handoff_ready = False
        if current_delta is not None:
            xy_error, z_error, yaw_error, _tilt_error = self._alignment_error_components(current_delta)
            close_depth_ready = bool(
                depth_proximity is not None
                and np.isfinite(depth_proximity)
                and float(depth_proximity) <= min(max(self.alignment_depth_threshold * 0.17, 0.0135), 0.016)
            )
            handoff_depth_ready = bool(
                depth_proximity is not None
                and np.isfinite(depth_proximity)
                and float(depth_proximity) <= min(max(self.alignment_depth_threshold * 0.13, 0.0095), 0.0125)
            )
            if close_depth_ready or handoff_depth_ready:
                # Keep yaw out of the way until xy/z are very nearly settled.
                # This favors true bridge alignment before any close / handoff
                # release can happen.
                close_xy_threshold = min(self.close_veto_xy_threshold * 0.55, 0.0058)
                close_z_threshold = min(self.close_veto_abs_z_threshold * 0.22, 0.0042)
                close_yaw_threshold = 0.015 if self.close_veto_yaw_threshold < 0.0 else min(self.close_veto_yaw_threshold, 0.015)
                handoff_xy_threshold = min(self.close_veto_xy_threshold * 0.40, 0.0042)
                handoff_z_threshold = min(self.close_veto_abs_z_threshold * 0.16, 0.0030)
                handoff_yaw_threshold = 0.010 if self.close_veto_yaw_threshold < 0.0 else min(self.close_veto_yaw_threshold, 0.010)
                geometry_close_ready = bool(
                    close_depth_ready
                    and xy_error <= close_xy_threshold
                    and z_error <= close_z_threshold
                    and yaw_error <= close_yaw_threshold
                )
                geometry_handoff_ready = bool(
                    handoff_depth_ready
                    and xy_error <= handoff_xy_threshold
                    and z_error <= handoff_z_threshold
                    and yaw_error <= handoff_yaw_threshold
                )
                # If the gripper is already open and the geometry is strongly
                # aligned, allow the close head to be the tiebreaker. This keeps
                # the learned head in the loop while preventing early false
                # positives from firing the gate on poor geometry.
                if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
                    geometry_close_ready = False
                    geometry_handoff_ready = False

        manager_has_object_in_hand = bool(getattr(self.manager, "has_object_in_hand", False))
        if manager_has_object_in_hand and (gripper_open is None or float(gripper_open) < self.alignment_open_threshold):
            geometry_handoff_ready = True
        close_state_handoff_release = bool(
            (gripper_open is None or float(gripper_open) < self.alignment_open_threshold)
            and self._phase1_close_arbiter_state in ("CLOSE_CANDIDATE", "FORCE_CONFIRM_HOLD")
            and self._phase1_close_command_source != "none"
        )

        close_ready_applied = bool(close_ready_applied and geometry_close_ready)
        handoff_ready_applied = bool(
            (handoff_ready_applied and (geometry_handoff_ready or (geometry_close_ready and bool(getattr(controller, "_runtime_handoff_spec_name", "none") != "none"))))
            or manager_has_object_in_hand
            or close_state_handoff_release
        )

        return {
            "supported": True,
            "close_ready_pred": close_ready_pred,
            "close_ready_applied": close_ready_applied,
            "handoff_ready_pred": handoff_ready_pred,
            "handoff_ready_applied": handoff_ready_applied,
            "close_ready_prob": close_ready_prob,
            "handoff_ready_prob": handoff_ready_prob,
            "close_ready_logit": close_ready_logit,
            "handoff_ready_logit": handoff_ready_logit,
        }

    def _phase1_bridge_support_soft_override(
        self,
        controller,
        depth_proximity: Optional[float],
        gripper_open: Optional[float],
    ) -> tuple[bool, str]:
        if not self.enable_phase1_bridge_support_soft_override:
            return False, "disabled"
        if not self._phase1_bridge_controller_capable(controller):
            return False, "no_controller"
        if not self._phase1_bridge_runtime_segment():
            return False, "not_phase1_bridge"
        if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
            return False, "gripper"
        near_target = bool(
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and float(depth_proximity) < self.alignment_depth_threshold
        )
        if not near_target:
            return False, "depth"
        if not self._runtime_has_motion_target(controller):
            return False, "no_motion_target"
        assist_band_ready = bool(self._alignment_assist_band_ready(controller, gripper_open))
        refine_band_ready = bool(self._alignment_refine_band_ready(controller, gripper_open))
        if not (assist_band_ready or refine_band_ready):
            return False, "band"
        return True, "phase1_near_contact"

    def _phase1_bridge_soft_apply_delta(
        self,
        local_delta: np.ndarray,
        *,
        reason: str,
    ) -> np.ndarray:
        softened = self._clip_alignment_diffusion_local_caps(
            local_delta,
            max_pos=self.phase1_bridge_soft_apply_max_pos,
            max_yaw=self.phase1_bridge_soft_apply_max_yaw,
        )
        self._last_alignment_diffusion_phase1_bridge_soft_apply = True
        self._last_alignment_diffusion_phase1_bridge_soft_apply_reason = str(reason)
        self._last_alignment_diffusion_soft_clamp = True
        self._alignment_diffusion_soft_clamp_count += 1
        return softened

    def _phase1_bridge_project_workspace_delta(
        self,
        local_delta: np.ndarray,
        *,
        current_quat: np.ndarray,
        gripper_pose: np.ndarray,
        base_world_delta: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        pose_arr = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)
        if pose_arr.size < 3:
            return None, None, np.inf
        if base_world_delta is None:
            base_world = np.zeros(6, dtype=np.float32)
        else:
            base_world = np.asarray(base_world_delta, dtype=np.float32).reshape(-1)[:6].copy()
        raw_local = np.asarray(local_delta, dtype=np.float32).reshape(-1)[:6].copy()
        raw_world = local_delta_to_world(raw_local, current_quat)
        target_xyz = pose_arr[:3] + base_world[:3] + raw_world[:3]
        clamped_xyz = self.safety.clamp_workspace(target_xyz)
        projected_world = raw_world.copy()
        projected_world[:3] = clamped_xyz - pose_arr[:3] - base_world[:3]
        projected_local = world_delta_to_local(projected_world, current_quat)
        projected_local = self._clip_alignment_diffusion_local_caps(
            projected_local,
            max_pos=self.phase1_bridge_workspace_project_max_pos,
            max_yaw=self.phase1_bridge_workspace_project_max_yaw,
        )
        projected_world = local_delta_to_world(projected_local, current_quat)
        projected_violation = float(self.safety.workspace_violation(pose_arr[:3] + base_world[:3] + projected_world[:3]))
        if not np.all(np.isfinite(projected_local[:6])) or not np.all(np.isfinite(projected_world[:6])):
            return None, None, np.inf
        return projected_local.astype(np.float32), projected_world.astype(np.float32), projected_violation

    def _phase1_bridge_blend_planner_world(
        self,
        base_world_delta: np.ndarray,
    ) -> tuple[np.ndarray, bool, str]:
        base_world = np.asarray(base_world_delta, dtype=np.float32).reshape(-1)[:6].copy()
        if not self._phase1_bridge_runtime_segment():
            return base_world, False, "not_phase1_bridge"
        blended = base_world.copy()
        blended[0] *= self.phase1_bridge_planner_x_blend
        blended[1] *= min(self.phase1_bridge_planner_x_blend + 0.08, 0.70)
        blended[2] *= self.phase1_bridge_planner_z_blend
        blended[5] *= self.phase1_bridge_planner_yaw_blend
        return blended.astype(np.float32), True, "phase1_bridge_planner_blend"

    def _phase1_bridge_basin_bias_local(self, current_delta: Optional[np.ndarray]) -> np.ndarray:
        bias = np.zeros(6, dtype=np.float32)
        if current_delta is None:
            return bias
        cur = np.asarray(current_delta, dtype=np.float32).reshape(-1)
        if cur.size < 6 or not np.all(np.isfinite(cur[:6])):
            return bias
        bias[0] = float(cur[0]) * self.phase1_bridge_basin_bias_xy_scale
        bias[1] = float(cur[1]) * self.phase1_bridge_basin_bias_xy_scale
        bias[2] = float(cur[2]) * self.phase1_bridge_basin_bias_z_scale
        bias[5] = float(cur[5]) * self.phase1_bridge_basin_bias_yaw_scale
        bias = self._clip_alignment_diffusion_local_caps(
            bias,
            max_pos=self.phase1_bridge_basin_bias_max_pos,
            max_yaw=self.phase1_bridge_basin_bias_max_yaw,
        )
        return bias.astype(np.float32)

    def _phase1_bridge_taskspace_yaw_rescue_local(
        self,
        current_pose_7d: Optional[np.ndarray],
        target_pose_7d: Optional[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        rescue = np.zeros(6, dtype=np.float32)
        if current_pose_7d is None or target_pose_7d is None:
            return rescue, 0.0
        try:
            current_pose = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
            target_pose = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
            if not (
                np.all(np.isfinite(current_pose[:7]))
                and np.all(np.isfinite(target_pose[:7]))
            ):
                return rescue, 0.0
            current_rot = Rotation.from_quat(current_pose[3:7])
            target_rot = Rotation.from_quat(target_pose[3:7])
            current_forward = current_rot.apply(np.asarray([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
            target_forward = target_rot.apply(np.asarray([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
            current_proj = np.asarray(current_forward[:2], dtype=np.float32)
            target_proj = np.asarray(target_forward[:2], dtype=np.float32)
            current_proj_norm = float(np.linalg.norm(current_proj))
            target_proj_norm = float(np.linalg.norm(target_proj))
            if current_proj_norm < 1e-6 or target_proj_norm < 1e-6:
                current_yaw = float(current_rot.as_euler("zyx", degrees=False)[0])
                target_yaw = float(target_rot.as_euler("zyx", degrees=False)[0])
                yaw_err = float((target_yaw - current_yaw + np.pi) % (2.0 * np.pi) - np.pi)
            else:
                current_proj = current_proj / max(current_proj_norm, 1e-6)
                target_proj = target_proj / max(target_proj_norm, 1e-6)
                yaw_err = float(
                    np.arctan2(
                        current_proj[0] * target_proj[1] - current_proj[1] * target_proj[0],
                        current_proj[0] * target_proj[0] + current_proj[1] * target_proj[1],
                    )
                )
            yaw_err = float(wrap_yaw_to_symmetry(yaw_err, np.pi / 2.0))
            rescue[5] = float(yaw_err * self.phase1_bridge_basin_bias_yaw_direct_scale)
            rescue = self._clip_alignment_diffusion_local_caps(
                rescue,
                max_pos=0.0,
                max_yaw=self.phase1_bridge_basin_bias_yaw_direct_max,
            )
            return rescue.astype(np.float32), float(yaw_err)
        except Exception:
            return np.zeros(6, dtype=np.float32), 0.0

    def _phase1_force_current_delta(self, controller=None) -> Optional[np.ndarray]:
        if controller is None:
            controller = self._active_alignment_controller()
        if controller is None:
            return None
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if current_delta.size < 6 or not np.all(np.isfinite(current_delta[:6])):
            return None
        return current_delta[:6].astype(np.float32)

    def _phase1_force_close_ready(self, controller=None, depth_proximity: Optional[float] = None) -> bool:
        current_delta = self._phase1_force_current_delta(controller)
        if current_delta is None:
            return False
        planner_close_intent = bool(getattr(controller, "_alignment_planner_close_intent", False)) if controller is not None else False
        student_gate = self._student_vnext_ready_gate_state(controller, gripper_open=gripper_open)
        xy_error, z_error, yaw_error, _tilt_error = self._alignment_error_components(current_delta)
        near_depth = bool(
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and float(depth_proximity) <= max(self.alignment_depth_threshold * 0.14, 0.0125)
        )
        # Phase1 force reflex should be a little more eager than the legacy
        # close-veto path. Once the bridge is in ALIGN_PREGRASP + near_contact
        # and the planner is already hinting at close, let the arbiter claim a
        # candidate close slightly earlier. Yaw is advisory here rather than a
        # hard blocker, because the grasp basin is primarily driven by xy/z.
        xy_threshold = max(self.close_veto_xy_threshold * 1.55, 0.0115)
        z_threshold = max(self.close_veto_abs_z_threshold * 0.45, 0.0105)
        yaw_threshold = max(self.close_veto_yaw_threshold, 0.32) if self.close_veto_yaw_threshold >= 0.0 else -1.0
        if planner_close_intent:
            xy_threshold = max(xy_threshold, 0.0125)
            z_threshold = max(z_threshold, 0.0115)
            if yaw_threshold >= 0.0:
                yaw_threshold = max(yaw_threshold, 0.35)
        if student_gate.get("supported", False):
            student_close_ready = bool(student_gate.get("close_ready_applied", False))
            student_handoff_ready = bool(student_gate.get("handoff_ready_applied", False))
            if student_close_ready:
                if near_depth and (
                    planner_close_intent
                    or student_handoff_ready
                    or (xy_error <= max(xy_threshold, 0.0105) and z_error <= max(z_threshold, 0.0100))
                ):
                    return True
        if near_depth and xy_error <= xy_threshold and z_error <= z_threshold:
            if yaw_threshold < 0.0 or yaw_error <= yaw_threshold:
                return True
        return False

    def _phase1_force_metrics(
        self,
        force_reading: Optional[np.ndarray],
    ) -> dict:
        out = {
            "force_norm": 0.0,
            "fz": 0.0,
            "lateral": 0.0,
            "torque_norm": 0.0,
            "tau_z": 0.0,
            "spike": False,
            "jam": False,
        }
        if force_reading is None:
            self._phase1_force_prev_norm = None
            return out
        force_arr = np.asarray(force_reading, dtype=np.float32).reshape(-1)
        fxyz = force_arr[:3] if force_arr.size >= 3 else np.zeros(3, dtype=np.float32)
        tau = force_arr[3:6] if force_arr.size >= 6 else np.zeros(3, dtype=np.float32)
        force_norm = float(np.linalg.norm(fxyz))
        fz = float(abs(fxyz[2])) if fxyz.size >= 3 else 0.0
        lateral = float(np.linalg.norm(fxyz[:2])) if fxyz.size >= 2 else 0.0
        torque_norm = float(np.linalg.norm(tau))
        prev_force_norm = None if self._phase1_force_prev_norm is None else float(self._phase1_force_prev_norm)
        spike = bool(
            prev_force_norm is not None
            and abs(force_norm - prev_force_norm) >= self.phase1_force_spike_threshold
        )
        jam = bool(
            lateral >= self.phase1_force_lateral_threshold
            or torque_norm >= self.phase1_force_torque_threshold
        )
        self._phase1_force_prev_norm = force_norm
        out.update(
            {
                "force_norm": force_norm,
                "fz": fz,
                "lateral": lateral,
                "torque_norm": torque_norm,
                "tau_z": float(tau[2]) if tau.size >= 3 else 0.0,
                "spike": spike,
                "jam": jam,
            }
        )
        return out

    def _phase1_force_progress_insufficient(self, current_delta: Optional[np.ndarray]) -> bool:
        if current_delta is None:
            self._phase1_prev_basin_metrics = None
            return False
        cur_xy = float(np.linalg.norm(current_delta[:2]))
        cur_z = float(abs(current_delta[2]))
        cur_yaw = float(abs(current_delta[5]))
        insufficient = False
        if self._phase1_prev_basin_metrics is not None:
            prev_xy, prev_z, prev_yaw = self._phase1_prev_basin_metrics
            insufficient = bool(
                cur_xy >= prev_xy - 5e-4
                and cur_z >= prev_z - 4e-4
                and cur_yaw >= prev_yaw - 0.01
            )
        self._phase1_prev_basin_metrics = (cur_xy, cur_z, cur_yaw)
        return bool(insufficient or int(getattr(self.manager, "no_progress_steps", 0)) >= 2)

    def _phase1_force_contact_signature(
        self,
        *,
        force_metrics: dict,
        depth_proximity: Optional[float],
        current_delta: Optional[np.ndarray],
    ) -> bool:
        near_depth = bool(
            depth_proximity is not None
            and np.isfinite(depth_proximity)
            and float(depth_proximity) <= max(self.alignment_depth_threshold, 0.025)
        )
        basin_near = False
        if current_delta is not None:
            basin_near = bool(
                float(np.linalg.norm(current_delta[:2])) <= max(self.close_veto_xy_threshold * 1.5, 0.010)
                and float(abs(current_delta[2])) <= max(self.close_veto_abs_z_threshold * 1.75, 0.010)
            )
        return bool(
            force_metrics["fz"] >= self.phase1_force_contact_fz_threshold
            or force_metrics["force_norm"] >= self.phase1_force_force_norm_threshold
            or (near_depth and basin_near)
        )

    def _phase1_force_backoff_local(self, force_metrics: dict, *, reopen: bool = False) -> np.ndarray:
        adjust = np.zeros(6, dtype=np.float32)
        backoff = float(self.phase1_force_backoff_mm)
        if backoff <= 0.0:
            return adjust
        lateral = float(force_metrics.get("lateral", 0.0))
        tau_z = float(force_metrics.get("tau_z", 0.0))
        if lateral > 1e-6:
            fxy = np.asarray(
                [
                    float(force_metrics.get("fx", 0.0)),
                    float(force_metrics.get("fy", 0.0)),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(fxy) > 1e-6:
                adjust[:2] += (-fxy / max(np.linalg.norm(fxy), 1e-6)) * backoff
        adjust[2] += backoff
        if abs(tau_z) > 1e-6:
            adjust[5] += -np.sign(tau_z) * min(0.004, backoff * 2.0)
        if reopen:
            adjust[2] += backoff * 0.5
        return adjust.astype(np.float32)

    def _reset_phase1_force_runtime_state(self) -> None:
        self._phase1_close_arbiter_state = "APPROACH"
        self._phase1_close_command_source = "none"
        self._phase1_grasp_contact_confirmed = False
        self._phase1_force_reflex_active = False
        self._phase1_force_reflex_reason = "none"
        self._phase1_force_backoff_applied = False
        self._phase1_reopen_reason = "none"
        self._phase1_close_hold_active = False
        self._phase1_close_hold_remaining = 0
        self._phase1_close_confirm_streak = 0
        self._phase1_close_fail_remaining = 0
        self._phase1_reopen_cooldown_remaining = 0
        self._phase1_force_prev_norm = None
        self._phase1_prev_basin_metrics = None

    def _apply_phase1_force_contact_reflex(
        self,
        a_exec: np.ndarray,
        *,
        current_quat: np.ndarray,
        force_reading: Optional[np.ndarray],
        depth_proximity: Optional[float],
        gripper_open: Optional[float],
        controller=None,
    ) -> np.ndarray:
        if not self.enable_phase1_force_reflex:
            return a_exec
        manager_has_object_in_hand = bool(getattr(self.manager, "has_object_in_hand", False))
        if manager_has_object_in_hand:
            self._phase1_close_arbiter_state = "FORCE_CONFIRM_HOLD"
            if not self._phase1_grasp_contact_confirmed:
                self._phase1_grasp_contact_confirmed = True
                self._phase1_grasp_contact_confirmed_count += 1
            self._phase1_close_hold_active = True
            self._phase1_close_hold_remaining = max(
                int(self._phase1_close_hold_remaining),
                int(self.phase1_post_contact_hold_steps),
            )
        keep_closed_after_grasp = bool(
            manager_has_object_in_hand
            and self._phase1_close_arbiter_state in ("CLOSE_CANDIDATE", "FORCE_CONFIRM_HOLD")
        )
        if not self._phase1_bridge_runtime_segment() and not keep_closed_after_grasp:
            self._reset_phase1_force_runtime_state()
            return a_exec

        force_metrics = self._phase1_force_metrics(force_reading)
        force_metrics["fx"] = float(np.asarray(force_reading, dtype=np.float32).reshape(-1)[0]) if force_reading is not None and np.asarray(force_reading).size >= 1 else 0.0
        force_metrics["fy"] = float(np.asarray(force_reading, dtype=np.float32).reshape(-1)[1]) if force_reading is not None and np.asarray(force_reading).size >= 2 else 0.0
        if force_metrics["spike"]:
            self._phase1_force_spike_count += 1
        if force_metrics["jam"]:
            self._phase1_jam_detected_count += 1
        current_delta = self._phase1_force_current_delta(controller)
        close_ready = self._phase1_force_close_ready(controller, depth_proximity=depth_proximity)
        progress_insufficient = self._phase1_force_progress_insufficient(current_delta)
        is_open = bool(gripper_open is None or float(gripper_open) >= self.alignment_open_threshold)
        contact_signature = self._phase1_force_contact_signature(
            force_metrics=force_metrics,
            depth_proximity=depth_proximity,
            current_delta=current_delta,
        )

        def _apply_local_override(local_override: np.ndarray, *, gripper_cmd: Optional[float] = None, reason: str = "none") -> np.ndarray:
            nonlocal a_exec
            world_override = local_delta_to_world(np.asarray(local_override, dtype=np.float32), current_quat)
            out = np.asarray(a_exec, dtype=np.float32).copy()
            out[:6] = world_override[:6]
            if gripper_cmd is not None:
                out[6] = float(gripper_cmd)
            self._phase1_force_reflex_active = True
            self._phase1_force_reflex_reason = str(reason)
            self._phase1_force_reflex_activation_count += 1
            self._phase1_force_backoff_applied = bool(np.linalg.norm(local_override[:3]) > 1e-8 or abs(float(local_override[5])) > 1e-8)
            return out

        if self._phase1_close_arbiter_state == "REOPEN_REPLAN":
            local_backoff = self._phase1_force_backoff_local(force_metrics, reopen=True)
            a_exec = _apply_local_override(local_backoff, gripper_cmd=1.0, reason=f"reopen_{self._phase1_reopen_reason}")
            self._phase1_reopen_cooldown_remaining = max(self._phase1_reopen_cooldown_remaining - 1, 0)
            if self._phase1_reopen_cooldown_remaining <= 0:
                self._phase1_close_arbiter_state = "APPROACH"
                self._phase1_close_command_source = "reopen_complete"
                self._phase1_close_confirm_streak = 0
                self._phase1_close_fail_remaining = 0
            else:
                self._phase1_close_command_source = "phase1_reopen_replan"
            return a_exec

        if self._phase1_close_arbiter_state == "FORCE_CONFIRM_HOLD":
            a_exec = np.asarray(a_exec, dtype=np.float32).copy()
            a_exec[6] = 0.0
            self._phase1_force_reflex_active = True
            self._phase1_force_reflex_reason = "force_confirm_hold"
            self._phase1_force_reflex_activation_count += 1
            self._phase1_close_hold_active = True
            self._phase1_close_hold_remaining = max(self._phase1_close_hold_remaining - 1, 0)
            self._phase1_close_command_source = "phase1_force_confirm_hold"
            self._phase1_grasp_contact_confirmed = True
            return a_exec

        if self._phase1_close_arbiter_state == "CLOSE_CANDIDATE":
            if manager_has_object_in_hand:
                self._phase1_close_arbiter_state = "FORCE_CONFIRM_HOLD"
                self._phase1_grasp_contact_confirmed = True
                self._phase1_grasp_contact_confirmed_count += 1
                self._phase1_close_hold_remaining = int(self.phase1_post_contact_hold_steps)
                self._phase1_close_hold_active = True
                self._phase1_close_command_source = "phase1_force_hold_object_in_hand"
                a_exec = np.asarray(a_exec, dtype=np.float32).copy()
                a_exec[6] = 0.0
                self._phase1_force_reflex_active = True
                self._phase1_force_reflex_reason = "object_in_hand_hold"
                self._phase1_force_reflex_activation_count += 1
                return a_exec
            if force_metrics["spike"] or force_metrics["jam"]:
                self._phase1_close_arbiter_state = "REOPEN_REPLAN"
                self._phase1_reopen_reason = "spike" if force_metrics["spike"] else "jam"
                self._phase1_reopen_cooldown_remaining = int(self.phase1_reopen_cooldown_steps)
                self._phase1_reopen_count += 1
                local_backoff = self._phase1_force_backoff_local(force_metrics, reopen=True)
                self._phase1_close_command_source = "phase1_reopen_replan"
                return _apply_local_override(local_backoff, gripper_cmd=1.0, reason=self._phase1_reopen_reason)
            if contact_signature:
                self._phase1_close_confirm_streak += 1
                if self._phase1_close_confirm_streak >= self.phase1_close_confirm_steps:
                    self._phase1_close_arbiter_state = "FORCE_CONFIRM_HOLD"
                    self._phase1_grasp_contact_confirmed = True
                    self._phase1_grasp_contact_confirmed_count += 1
                    self._phase1_close_hold_remaining = int(self.phase1_post_contact_hold_steps)
                    self._phase1_close_command_source = "phase1_force_contact_confirmed"
                    return _apply_local_override(np.zeros(6, dtype=np.float32), gripper_cmd=0.0, reason="contact_confirmed")
            else:
                self._phase1_close_confirm_streak = 0
            self._phase1_close_fail_remaining = max(self._phase1_close_fail_remaining - 1, 0)
            if self._phase1_close_fail_remaining <= 0:
                self._phase1_close_arbiter_state = "REOPEN_REPLAN"
                self._phase1_reopen_reason = "no_contact"
                self._phase1_reopen_cooldown_remaining = int(self.phase1_reopen_cooldown_steps)
                self._phase1_reopen_count += 1
                local_backoff = self._phase1_force_backoff_local(force_metrics, reopen=True)
                self._phase1_close_command_source = "phase1_reopen_replan"
                return _apply_local_override(local_backoff, gripper_cmd=1.0, reason="no_contact")
            if (
                not contact_signature
                and not manager_has_object_in_hand
                and float(force_metrics["fz"]) <= self.phase1_force_contact_fz_threshold * 0.5
                and float(force_metrics["force_norm"]) <= self.phase1_force_force_norm_threshold * 0.5
            ):
                self._phase1_close_fail_remaining = min(self._phase1_close_fail_remaining, 1)
            self._phase1_close_command_source = "phase1_force_close_candidate"
            return _apply_local_override(np.zeros(6, dtype=np.float32), gripper_cmd=0.0, reason="close_candidate")

        if close_ready and is_open:
            self._phase1_close_arbiter_state = "CLOSE_CANDIDATE"
            self._phase1_close_confirm_streak = 0
            self._phase1_close_fail_remaining = int(self.phase1_close_fail_steps)
            self._phase1_close_command_source = "phase1_force_close_ready"
            return _apply_local_override(np.zeros(6, dtype=np.float32), gripper_cmd=0.0, reason="close_ready")

        if force_metrics["spike"]:
            local_backoff = self._phase1_force_backoff_local(force_metrics, reopen=False)
            self._phase1_close_command_source = "none"
            return _apply_local_override(local_backoff, reason="force_spike")
        if force_metrics["jam"]:
            local_backoff = self._phase1_force_backoff_local(force_metrics, reopen=False)
            self._phase1_close_command_source = "none"
            return _apply_local_override(local_backoff, reason="jam_backoff")
        if force_metrics["fz"] >= self.phase1_force_high_fz_threshold and progress_insufficient:
            local_exec = world_delta_to_local(np.asarray(a_exec, dtype=np.float32)[:6], current_quat)
            local_exec = np.asarray(local_exec, dtype=np.float32).copy()
            if float(local_exec[2]) < 0.0:
                local_exec[2] = 0.0
            self._phase1_close_command_source = "none"
            return _apply_local_override(local_exec, reason="high_fz_cancel_descend")

        self._phase1_close_command_source = "none"
        return a_exec

    def _active_alignment_controller(self):
        for controller in (
            getattr(self, "alignment_tc_student_vnext_controller", None),
            getattr(self, "alignment_tc_diffusion_controller", None),
            getattr(self, "alignment_diffusion_controller", None),
            getattr(self, "alignment_v4_shadow_controller", None),
            getattr(self, "alignment_controller", None),
        ):
            if controller is not None:
                return controller
        return None

    def _try_alignment_diffusion(
        self,
        a_exec: np.ndarray,
        current_quat: np.ndarray,
        *,
        front_rgb,
        wrist_rgb,
        wrist_depth,
        ft_hist,
        proprio,
        base_action_local,
        depth_proximity: Optional[float],
        force_reading: Optional[np.ndarray],
        gripper_open: Optional[float],
        gripper_pose: Optional[np.ndarray],
        gripper_context=None,
    ) -> bool:
        self._alignment_diffusion_eval_count += 1
        self._reset_alignment_diffusion_last("disabled")
        tc_enabled = bool(
            self.alignment_tc_diffusion_controller is not None
            and (self.enable_alignment_tc_diffusion_shadow or self.enable_alignment_tc_diffusion_apply)
        )
        vnext_enabled = bool(
            self.alignment_tc_student_vnext_controller is not None
            and (self.enable_alignment_tc_student_vnext_shadow or self.enable_alignment_tc_student_vnext_apply)
        )
        collector_like = bool(self.alignment_tc_student_vnext_collector_like and vnext_enabled)
        if vnext_enabled:
            controller = getattr(self, "alignment_tc_student_vnext_controller", None)
            shadow_enabled = bool(self.enable_alignment_tc_student_vnext_shadow)
            apply_enabled = bool(self.enable_alignment_tc_student_vnext_apply)
            num_samples = int(self.alignment_tc_diffusion_num_samples)
            top_k = int(self.alignment_tc_diffusion_top_k)
            risk_threshold = 1.1 if collector_like else float(self.alignment_tc_diffusion_risk_threshold)
            execute_steps = int(self.alignment_tc_diffusion_execute_steps)
            controller_type = "alignment_tc_student_vnext"
            tc_enabled = True
        elif tc_enabled:
            controller = getattr(self, "alignment_tc_diffusion_controller", None)
            shadow_enabled = bool(self.enable_alignment_tc_diffusion_shadow)
            apply_enabled = bool(self.enable_alignment_tc_diffusion_apply)
            num_samples = int(self.alignment_tc_diffusion_num_samples)
            top_k = int(self.alignment_tc_diffusion_top_k)
            risk_threshold = float(self.alignment_tc_diffusion_risk_threshold)
            execute_steps = int(self.alignment_tc_diffusion_execute_steps)
            controller_type = "alignment_tc_diffusion_refiner"
        else:
            controller = getattr(self, "alignment_diffusion_controller", None)
            shadow_enabled = bool(self.enable_alignment_diffusion_shadow)
            apply_enabled = bool(self.enable_alignment_diffusion_apply)
            num_samples = int(self.alignment_diffusion_num_samples)
            top_k = None
            risk_threshold = float(self.alignment_diffusion_risk_threshold)
            execute_steps = int(self.alignment_diffusion_execute_steps)
            controller_type = "alignment_diffusion_refiner"
        if controller is None or not (shadow_enabled or apply_enabled):
            return False
        self._last_alignment_diffusion_enabled = True
        self._last_alignment_diffusion_controller_type = controller_type
        self._bump_alignment_diffusion_hist(self._alignment_diffusion_controller_type_hist, controller_type)
        self._last_alignment_diffusion_top_k = int(top_k or 0)
        phase_name = "insert_commit" if bool(getattr(self.manager, "has_object_in_hand", False)) else "grasp_commit"
        self._last_alignment_diffusion_phase_name = str(phase_name)
        self._bump_alignment_diffusion_hist(self._alignment_diffusion_phase_hist, phase_name)
        trigger_ok, bucket = self._alignment_diffusion_trigger_open(
            depth_proximity,
            force_reading,
            base_action_local,
            gripper_open,
        )
        self._last_alignment_diffusion_stage_bucket = bucket
        self._bump_alignment_diffusion_hist(self._alignment_diffusion_bucket_hist, bucket)
        if not trigger_ok:
            self._last_alignment_diffusion_block_reason = bucket if bucket == "coarse" else "trigger"
            self._bump_alignment_diffusion_hist(
                self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
            )
            return False

        import torch
        import time

        t0 = time.perf_counter()

        device = next(controller.parameters()).device
        dtype = next(controller.parameters()).dtype
        fr = self._to_rgb_tensor(front_rgb, (1, 3, 128, 128), device, dtype) if front_rgb is not None else None
        wr = self._to_rgb_tensor(wrist_rgb, (1, 3, 128, 128), device, dtype) if wrist_rgb is not None else None
        wd = self._to_tensor(wrist_depth, (1, 1, 96, 96), device, dtype)
        fh = self._to_tensor(ft_hist, (1, 32, 6), device, dtype)
        pr = self._to_tensor(proprio, (1, 15), device, dtype)
        ba = self._to_tensor(base_action_local, (1, 6), device, dtype)
        gc = self._to_tensor(gripper_context, (1, 4), device, dtype) if gripper_context is not None else None
        if fr is not None:
            fr = torch.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)
        if wr is not None:
            wr = torch.nan_to_num(wr, nan=0.0, posinf=0.0, neginf=0.0)
        if wd is not None:
            wd = torch.nan_to_num(wd, nan=0.0, posinf=0.0, neginf=0.0)
        if fh is not None:
            fh = torch.nan_to_num(fh, nan=0.0, posinf=0.0, neginf=0.0)
        if pr is not None:
            pr = torch.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
        if ba is not None:
            ba = torch.nan_to_num(ba, nan=0.0, posinf=0.0, neginf=0.0)
        if gc is not None:
            gc = torch.nan_to_num(gc, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            if hasattr(controller, "sample_candidates"):
                kwargs = dict(
                    num_samples=num_samples,
                    front_rgb=fr,
                    wrist_rgb=wr,
                    wrist_depth=wd,
                    force_history=fh,
                    proprio=pr,
                    planner_action_local=ba,
                    gripper_context=gc,
                    phase_id=torch.tensor([1 if bool(getattr(self.manager, "has_object_in_hand", False)) else 0], device=device, dtype=torch.long),
                    stage_bucket_id=torch.tensor(
                        [
                            5
                            if bucket in ("micro_insert", "insert_precommit_micro")
                            else 4
                            if bucket == "insert_near_align"
                            else 1
                            if bucket == "micro_contact_refine"
                            else 0
                            if bucket == "near_contact_refine"
                            else 8
                        ],
                        device=device,
                        dtype=torch.long,
                    ),
                )
                if top_k is not None:
                    kwargs["top_k"] = top_k
                try:
                    out = controller.sample_candidates(**kwargs)
                except TypeError:
                    kwargs.pop("top_k", None)
                    out = controller.sample_candidates(**kwargs)
            else:
                out = controller(
                    front_rgb=fr,
                    wrist_rgb=wr,
                    wrist_depth=wd,
                    force_history=fh,
                    proprio=pr,
                    planner_action_local=ba,
                    gripper_context=gc,
                )
        self._last_alignment_diffusion_latency_total_ms = float((time.perf_counter() - t0) * 1000.0)

        self._last_alignment_diffusion_active = True
        self._alignment_diffusion_active_count += 1
        phase1_bridge_segment = bool(vnext_enabled and self._phase1_bridge_runtime_segment(bucket))

        risk_logit = out.get("risk_logit", None)
        stop_logit = out.get("stop_logit", None)
        close_ready_logit = out.get("close_ready_logit", None)
        handoff_ready_logit = out.get("handoff_ready_logit", None)
        risk_prob = float(torch.sigmoid(risk_logit.reshape(-1)[0]).detach().cpu().item()) if risk_logit is not None else 0.0
        stop_prob = float(torch.sigmoid(stop_logit.reshape(-1)[0]).detach().cpu().item()) if stop_logit is not None else 0.0
        self._last_alignment_diffusion_risk_prob = risk_prob
        self._last_alignment_diffusion_stop_prob = stop_prob
        self._alignment_diffusion_risk_prob_sum += float(risk_prob)
        self._alignment_diffusion_stop_prob_sum += float(stop_prob)
        if "apply_confidence" in out:
            self._last_alignment_diffusion_target_confidence = float(
                out["apply_confidence"].reshape(-1)[0].detach().float().cpu().item()
            )
        elif "target_confidence" in out:
            self._last_alignment_diffusion_target_confidence = float(
                out["target_confidence"].reshape(-1)[0].detach().float().cpu().item()
            )
        self._alignment_diffusion_confidence_sum += float(self._last_alignment_diffusion_target_confidence)
        if vnext_enabled and self._student_vnext_ready_gate_supported(controller):
            close_ready_prob = (
                float(torch.sigmoid(close_ready_logit.reshape(-1)[0]).detach().cpu().item())
                if close_ready_logit is not None
                else 0.0
            )
            handoff_ready_prob = (
                float(torch.sigmoid(handoff_ready_logit.reshape(-1)[0]).detach().cpu().item())
                if handoff_ready_logit is not None
                else 0.0
            )
            close_ready_threshold = float(self.alignment_tc_student_vnext_close_ready_threshold)
            handoff_ready_threshold = float(self.alignment_tc_student_vnext_handoff_ready_threshold)
            manager_has_object_in_hand = bool(getattr(self.manager, "has_object_in_hand", False))
            close_state_handoff_release = bool(
                (gripper_open is None or float(gripper_open) < self.alignment_open_threshold)
                and self._phase1_close_arbiter_state in ("CLOSE_CANDIDATE", "FORCE_CONFIRM_HOLD")
                and self._phase1_close_command_source != "none"
            )
            controller._runtime_student_ready_gate_active = True
            controller._runtime_close_ready_logit = (
                float(close_ready_logit.reshape(-1)[0].detach().cpu().item()) if close_ready_logit is not None else np.nan
            )
            controller._runtime_close_ready_prob = float(close_ready_prob)
            controller._runtime_close_ready_pred = bool(close_ready_prob >= 0.5)
            controller._runtime_close_ready_threshold = float(close_ready_threshold)
            controller._runtime_close_ready = bool(close_ready_prob >= close_ready_threshold)
            controller._runtime_close_ready_applied = bool(close_ready_prob >= close_ready_threshold)
            controller._runtime_handoff_ready_logit = (
                float(handoff_ready_logit.reshape(-1)[0].detach().cpu().item()) if handoff_ready_logit is not None else np.nan
            )
            controller._runtime_handoff_ready_prob = float(handoff_ready_prob)
            controller._runtime_handoff_ready_pred = bool(handoff_ready_prob >= 0.5)
            controller._runtime_handoff_ready_threshold = float(handoff_ready_threshold)
            controller._runtime_handoff_ready = bool(handoff_ready_prob >= handoff_ready_threshold)
            controller._runtime_handoff_ready_applied = bool(
                handoff_ready_prob >= handoff_ready_threshold
                or manager_has_object_in_hand
                or close_state_handoff_release
            )
        pred_target_delta_np = None
        if "pred_target_delta_local_6d" in out:
            pred_target_delta_np = (
                out["pred_target_delta_local_6d"].reshape(-1).detach().float().cpu().numpy().astype(np.float32)
            )
            if pred_target_delta_np.size >= 6:
                pred_target_delta_np = pred_target_delta_np[:6]
                self._last_alignment_diffusion_pred_target_delta_6d = pred_target_delta_np.tolist()
                self._last_alignment_diffusion_pred_target_delta_norm = float(np.linalg.norm(pred_target_delta_np[:3]))
                self._last_alignment_diffusion_pred_target_yaw_abs = float(abs(pred_target_delta_np[5]))
                self._alignment_diffusion_pred_target_delta_norm_sum += float(
                    self._last_alignment_diffusion_pred_target_delta_norm
                )
                self._alignment_diffusion_pred_target_yaw_abs_sum += float(
                    self._last_alignment_diffusion_pred_target_yaw_abs
                )
        low_confidence = bool(
            tc_enabled
            and not collector_like
            and self._last_alignment_diffusion_target_confidence < float(self.alignment_tc_diffusion_confidence_threshold)
        )
        self._last_alignment_diffusion_low_confidence = low_confidence
        if low_confidence:
            self._alignment_diffusion_low_confidence_count += 1
        if "best_index" in out:
            self._last_alignment_diffusion_selected_index = int(out["best_index"].reshape(-1)[0].detach().cpu().item())
        if "num_samples" in out:
            self._last_alignment_diffusion_num_samples = int(out["num_samples"].reshape(-1)[0].detach().cpu().item())
        if "candidate_diversity" in out:
            self._last_alignment_diffusion_candidate_diversity = float(
                out["candidate_diversity"].reshape(-1)[0].detach().float().cpu().item()
            )
        if "candidate_scores" in out:
            scores = out["candidate_scores"].reshape(1, -1)
            idx = max(self._last_alignment_diffusion_selected_index, 0)
            idx = min(idx, int(scores.shape[1]) - 1)
            self._last_alignment_diffusion_candidate_score = float(scores[0, idx].detach().float().cpu().item())
        if "progress_logits" in out:
            self._last_alignment_diffusion_progress_logits = (
                out["progress_logits"].reshape(-1).detach().float().cpu().numpy().astype(np.float32).tolist()
            )

        residual_4d = out.get("first_residual_4d", None)
        residual_6d = out.get("first_residual_6d", None)
        if residual_4d is not None:
            residual_4d_np = residual_4d.reshape(-1).detach().float().cpu().numpy().astype(np.float32)
            self._last_alignment_diffusion_first_residual_4d = residual_4d_np.tolist()
        if residual_6d is None:
            if residual_4d is None:
                self._last_alignment_diffusion_block_reason = "no_residual"
                return False
            residual_4d_np = residual_4d.reshape(-1).detach().float().cpu().numpy().astype(np.float32)
            residual_np = np.zeros(6, dtype=np.float32)
            residual_np[:3] = residual_4d_np[:3]
            residual_np[5] = residual_4d_np[3]
        else:
            residual_np = residual_6d.reshape(-1).detach().float().cpu().numpy().astype(np.float32)
        if residual_np.size < 6 or not np.all(np.isfinite(residual_np[:6])):
            self._last_alignment_diffusion_block_reason = "bad_residual"
            return False

        local_delta = self._clip_alignment_diffusion_local(residual_np[:6])
        if low_confidence:
            if not self.alignment_tc_diffusion_soft_clamp:
                self._last_alignment_diffusion_safety_reject = True
                self._last_alignment_diffusion_hard_reject = True
                self._last_alignment_diffusion_block_reason = "low_confidence"
                self._alignment_diffusion_safety_reject_count += 1
                self._alignment_diffusion_hard_reject_count += 1
                self._bump_alignment_diffusion_hist(
                    self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
                )
                return False
            scale_down = float(
                np.clip(
                    self._last_alignment_diffusion_target_confidence
                    / max(float(self.alignment_tc_diffusion_confidence_threshold), 1e-6),
                    0.10,
                    1.0,
                )
            )
            local_delta *= scale_down
            self._last_alignment_diffusion_scale_down = scale_down
            self._last_alignment_diffusion_soft_clamp = True
            self._alignment_diffusion_soft_clamp_count += 1
            if phase1_bridge_segment:
                local_delta = self._phase1_bridge_soft_apply_delta(local_delta, reason="low_confidence")
        self._last_alignment_diffusion_first_residual_6d = np.asarray(local_delta, dtype=np.float32).tolist()
        self._last_alignment_diffusion_pos_norm = float(np.linalg.norm(local_delta[:3]))
        self._last_alignment_diffusion_yaw_abs = float(abs(local_delta[5]))
        self._alignment_diffusion_scale_down_sum += float(self._last_alignment_diffusion_scale_down)
        self._alignment_diffusion_pos_norm_sum += float(self._last_alignment_diffusion_pos_norm)
        self._alignment_diffusion_yaw_abs_sum += float(self._last_alignment_diffusion_yaw_abs)
        if self._last_alignment_diffusion_pos_norm <= 1e-8 and self._last_alignment_diffusion_yaw_abs <= 1e-8:
            self._last_alignment_diffusion_block_reason = "zero"
            self._bump_alignment_diffusion_hist(
                self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
            )
            return False
        if pred_target_delta_np is not None and pred_target_delta_np.size >= 6:
            pred4 = np.asarray([pred_target_delta_np[0], pred_target_delta_np[1], pred_target_delta_np[2], pred_target_delta_np[5]])
            act4 = np.asarray([local_delta[0], local_delta[1], local_delta[2], local_delta[5]])
            active = np.abs(pred4) > np.asarray([1e-5, 1e-5, 1e-5, 1e-4], dtype=np.float32)
            if np.any(active):
                self._last_alignment_diffusion_target_action_sign_agreement = float(
                    np.mean(np.sign(pred4[active]) == np.sign(act4[active]))
                )
                self._alignment_diffusion_target_action_sign_agreement_sum += float(
                    self._last_alignment_diffusion_target_action_sign_agreement
                )
        if (not collector_like) and (risk_prob > risk_threshold or stop_prob > 0.70):
            can_soft_risk = bool(
                tc_enabled
                and self.alignment_tc_diffusion_soft_clamp
                and risk_prob <= 0.95
                and stop_prob <= 0.85
            )
            if phase1_bridge_segment and tc_enabled and self.alignment_tc_diffusion_soft_clamp:
                can_soft_risk = True
            if not can_soft_risk:
                self._last_alignment_diffusion_safety_reject = True
                self._last_alignment_diffusion_hard_reject = True
                self._last_alignment_diffusion_block_reason = "risk"
                self._alignment_diffusion_safety_reject_count += 1
                self._alignment_diffusion_hard_reject_count += 1
                self._bump_alignment_diffusion_hist(
                    self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
                )
                return False
            if phase1_bridge_segment and self.enable_phase1_bridge_risk_soft_apply:
                local_delta = self._phase1_bridge_soft_apply_delta(local_delta, reason="risk")
                self._last_alignment_diffusion_scale_down *= 0.10
            else:
                risk_scale = float(np.clip(risk_threshold / max(risk_prob, 1e-6), 0.25, 1.0))
                local_delta *= risk_scale
                self._last_alignment_diffusion_scale_down *= risk_scale
                self._last_alignment_diffusion_soft_clamp = True
                self._alignment_diffusion_soft_clamp_count += 1
            self._last_alignment_diffusion_first_residual_6d = np.asarray(local_delta, dtype=np.float32).tolist()
            self._last_alignment_diffusion_pos_norm = float(np.linalg.norm(local_delta[:3]))
            self._last_alignment_diffusion_yaw_abs = float(abs(local_delta[5]))

        world_delta = local_delta_to_world(local_delta, current_quat)
        raw_base_exec_world = np.asarray(a_exec[:6], dtype=np.float32).copy()
        base_exec_world = raw_base_exec_world.copy()
        if phase1_bridge_segment:
            raw_base_exec_local = world_delta_to_local(raw_base_exec_world, current_quat)
            blended_world, blend_active, blend_reason = self._phase1_bridge_blend_planner_world(raw_base_exec_world)
            if blend_active:
                base_exec_local = world_delta_to_local(blended_world, current_quat)
                current_delta_basin = None
                align_ctrl = self._active_alignment_controller()
                if align_ctrl is not None:
                    current_delta_basin = getattr(align_ctrl, "_runtime_current_delta_basin_target", None)
                    if (
                        (current_delta_basin is None or not np.any(np.isfinite(np.asarray(current_delta_basin, dtype=np.float32)[:6])))
                        and gripper_pose is not None
                    ):
                        fallback_target_pose = getattr(align_ctrl, "_runtime_motion_target_pose_7d", None)
                        if fallback_target_pose is None:
                            fallback_target_pose = getattr(align_ctrl, "_canonical_basin_center_pose_7d", None)
                        if fallback_target_pose is not None:
                            try:
                                fallback_delta = pose_delta_local_between(
                                    np.asarray(gripper_pose, dtype=np.float32).reshape(7),
                                    np.asarray(fallback_target_pose, dtype=np.float32).reshape(7),
                                )
                                current_delta_basin = np.asarray(fallback_delta, dtype=np.float32)
                                setattr(
                                    align_ctrl,
                                    "_runtime_current_delta_basin_target",
                                    np.asarray(current_delta_basin, dtype=np.float32).copy(),
                                )
                            except Exception:
                                pass
                current_pose_for_yaw = np.asarray(gripper_pose, dtype=np.float32).reshape(-1) if gripper_pose is not None else None
                target_pose_for_yaw = None
                taskspace_yaw_target_source = "none"
                target_edge_pair_index = -1
                target_edge_pair_family = -1
                if align_ctrl is not None:
                    target_pose_for_yaw = getattr(align_ctrl, "_runtime_grasp_commit_target_pose_7d", None)
                    if target_pose_for_yaw is not None:
                        taskspace_yaw_target_source = "runtime_grasp_commit_target_pose_7d"
                        target_edge_pair_index = int(getattr(align_ctrl, "_runtime_grasp_commit_edge_pair_index", -1))
                        target_edge_pair_family = int(getattr(align_ctrl, "_runtime_grasp_commit_edge_pair_family", -1))
                    else:
                        target_pose_for_yaw = getattr(align_ctrl, "_runtime_motion_target_pose_7d", None)
                        if target_pose_for_yaw is not None:
                            taskspace_yaw_target_source = "runtime_motion_target_pose_7d"
                    if target_pose_for_yaw is None:
                        target_pose_for_yaw = getattr(align_ctrl, "_canonical_basin_center_pose_7d", None)
                        if target_pose_for_yaw is not None:
                            taskspace_yaw_target_source = "canonical_basin_center_pose_7d"
                direct_yaw_rescue_local, taskspace_yaw_error = self._phase1_bridge_taskspace_yaw_rescue_local(
                    current_pose_for_yaw,
                    target_pose_for_yaw,
                )
                if target_edge_pair_index >= 0:
                    taskspace_yaw_target_source = (
                        f"{taskspace_yaw_target_source}__edge_pair_q{int(target_edge_pair_index)}"
                        f"_f{int(target_edge_pair_family)}"
                    )
                basin_bias_local = self._phase1_bridge_basin_bias_local(current_delta_basin)
                yaw_holdoff_active = False
                yaw_holdoff_reason = "none"
                if current_delta_basin is not None:
                    cur_delta = np.asarray(current_delta_basin, dtype=np.float32).reshape(-1)
                    if cur_delta.size >= 6 and np.all(np.isfinite(cur_delta[:6])):
                        xy_norm = float(np.linalg.norm(cur_delta[:2]))
                        abs_z = float(abs(cur_delta[2]))
                        yaw_holdoff_active = bool(
                            xy_norm > self.phase1_bridge_yaw_holdoff_xy_threshold
                            or abs_z > self.phase1_bridge_yaw_holdoff_abs_z_threshold
                        )
                        if yaw_holdoff_active:
                            yaw_holdoff_reason = f"xy={xy_norm:.4f}|z={abs_z:.4f}"
                            direct_yaw_rescue_local = np.zeros(6, dtype=np.float32)
                            basin_bias_local = np.asarray(basin_bias_local, dtype=np.float32).copy()
                            basin_bias_local[5] = 0.0
                if np.any(np.isfinite(direct_yaw_rescue_local[:6])):
                    basin_bias_local = np.asarray(basin_bias_local, dtype=np.float32).copy()
                    basin_bias_local[5] = float(direct_yaw_rescue_local[5])
                base_exec_local = np.asarray(base_exec_local, dtype=np.float32).copy()
                if yaw_holdoff_active:
                    base_exec_local[5] *= float(self.phase1_bridge_yaw_holdoff_blend_scale)
                base_exec_local[:6] += basin_bias_local[:6]
                base_exec_world = local_delta_to_world(base_exec_local, current_quat)
                self._last_alignment_diffusion_phase1_bridge_blend_active = True
                self._last_alignment_diffusion_phase1_bridge_blend_reason = str(blend_reason)
                self._last_alignment_diffusion_phase1_bridge_blended_base_world = (
                    np.asarray(base_exec_world, dtype=np.float32).copy()
                )
                self._last_alignment_diffusion_phase1_bridge_blended_base_local = (
                    np.asarray(base_exec_local, dtype=np.float32).copy()
                )
                self._last_alignment_diffusion_phase1_bridge_basin_bias_local = (
                    np.asarray(basin_bias_local, dtype=np.float32).copy()
                )
                self._last_alignment_diffusion_phase1_bridge_taskspace_yaw_error = float(taskspace_yaw_error)
                self._last_alignment_diffusion_phase1_bridge_taskspace_yaw_target_source = str(
                    taskspace_yaw_target_source
                )
                self._last_alignment_diffusion_phase1_bridge_direct_yaw_rescue_local = (
                    np.asarray(direct_yaw_rescue_local, dtype=np.float32).copy()
                )
                self._last_alignment_diffusion_phase1_bridge_yaw_holdoff_active = bool(yaw_holdoff_active)
                self._last_alignment_diffusion_phase1_bridge_yaw_holdoff_reason = str(yaw_holdoff_reason)
        if gripper_pose is not None:
            pose_arr = np.asarray(gripper_pose, dtype=np.float32).reshape(-1)
            if pose_arr.size >= 3:
                workspace_tolerance = (
                    float(self.alignment_tc_diffusion_workspace_tolerance) if tc_enabled else 1e-6
                )
                violation = self.safety.workspace_violation(pose_arr[:3] + base_exec_world[:3] + world_delta[:3])
                self._last_alignment_diffusion_workspace_violation = float(violation)
                if violation > workspace_tolerance:
                    can_soft_workspace = bool(tc_enabled and self.alignment_tc_diffusion_workspace_soft_clamp)
                    projected_local = None
                    projected_world = None
                    projected_violation = np.inf
                    if phase1_bridge_segment and self.enable_phase1_bridge_risk_soft_apply:
                        projected_local, projected_world, projected_violation = self._phase1_bridge_project_workspace_delta(
                            local_delta,
                            current_quat=current_quat,
                            gripper_pose=pose_arr,
                            base_world_delta=base_exec_world,
                        )
                    projected_ok = projected_local is not None and projected_violation <= workspace_tolerance
                    if projected_ok and phase1_bridge_segment and self.phase1_bridge_workspace_project_first:
                        local_delta = projected_local
                        world_delta = projected_world
                        self._last_alignment_diffusion_soft_clamp = True
                        self._alignment_diffusion_soft_clamp_count += 1
                        self._last_alignment_diffusion_phase1_bridge_soft_apply = True
                        self._last_alignment_diffusion_phase1_bridge_soft_apply_reason = "workspace_project"
                        self._last_alignment_diffusion_workspace_projected = True
                        self._last_alignment_diffusion_workspace_project_reason = "phase1_boundary_projection_first"
                        self._last_alignment_diffusion_workspace_violation = float(projected_violation)
                        self._last_alignment_diffusion_first_residual_6d = (
                            np.asarray(local_delta, dtype=np.float32).tolist()
                        )
                        self._last_alignment_diffusion_pos_norm = float(np.linalg.norm(local_delta[:3]))
                        self._last_alignment_diffusion_yaw_abs = float(abs(local_delta[5]))
                    elif can_soft_workspace:
                        chosen_scale = None
                        chosen_world_delta = None
                        chosen_violation = None
                        scale_candidates = [0.75, 0.50, 0.25, 0.10]
                        if phase1_bridge_segment and self.enable_phase1_bridge_risk_soft_apply:
                            scale_candidates.extend([0.05, 0.02])
                        for scale in scale_candidates:
                            trial_local = local_delta * float(scale)
                            trial_world = local_delta_to_world(trial_local, current_quat)
                            trial_violation = self.safety.workspace_violation(
                                pose_arr[:3] + base_exec_world[:3] + trial_world[:3]
                            )
                            if trial_violation <= workspace_tolerance:
                                chosen_scale = float(scale)
                                chosen_world_delta = trial_world
                                chosen_violation = float(trial_violation)
                                break
                        if chosen_scale is not None:
                            local_delta *= chosen_scale
                            world_delta = chosen_world_delta
                            self._last_alignment_diffusion_scale_down *= chosen_scale
                            self._last_alignment_diffusion_soft_clamp = True
                            self._alignment_diffusion_soft_clamp_count += 1
                            self._last_alignment_diffusion_workspace_violation = chosen_violation
                            self._last_alignment_diffusion_first_residual_6d = (
                                np.asarray(local_delta, dtype=np.float32).tolist()
                            )
                            self._last_alignment_diffusion_pos_norm = float(np.linalg.norm(local_delta[:3]))
                            self._last_alignment_diffusion_yaw_abs = float(abs(local_delta[5]))
                        else:
                            if projected_ok:
                                local_delta = projected_local
                                world_delta = projected_world
                                self._last_alignment_diffusion_soft_clamp = True
                                self._alignment_diffusion_soft_clamp_count += 1
                                self._last_alignment_diffusion_phase1_bridge_soft_apply = True
                                self._last_alignment_diffusion_phase1_bridge_soft_apply_reason = "workspace_project"
                                self._last_alignment_diffusion_workspace_projected = True
                                self._last_alignment_diffusion_workspace_project_reason = "phase1_boundary_projection_after_scale"
                                self._last_alignment_diffusion_workspace_violation = float(projected_violation)
                                self._last_alignment_diffusion_first_residual_6d = (
                                    np.asarray(local_delta, dtype=np.float32).tolist()
                                )
                                self._last_alignment_diffusion_pos_norm = float(np.linalg.norm(local_delta[:3]))
                                self._last_alignment_diffusion_yaw_abs = float(abs(local_delta[5]))
                            else:
                                self._last_alignment_diffusion_safety_reject = True
                                self._last_alignment_diffusion_hard_reject = True
                                self._last_alignment_diffusion_block_reason = "workspace"
                                self._alignment_diffusion_safety_reject_count += 1
                                self._alignment_diffusion_hard_reject_count += 1
                                if phase1_bridge_segment and self.enable_phase1_bridge_risk_soft_apply:
                                    fallback_local = self._phase1_bridge_soft_apply_delta(local_delta, reason="workspace")
                                    fallback_world = local_delta_to_world(fallback_local, current_quat)
                                    fallback_violation = self.safety.workspace_violation(
                                        pose_arr[:3] + base_exec_world[:3] + fallback_world[:3]
                                    )
                                    if fallback_violation <= workspace_tolerance:
                                        self._last_alignment_diffusion_safety_reject = False
                                        self._last_alignment_diffusion_hard_reject = False
                                        local_delta = fallback_local
                                        world_delta = fallback_world
                                        self._last_alignment_diffusion_scale_down *= 0.10
                                        self._last_alignment_diffusion_workspace_violation = float(fallback_violation)
                                        self._last_alignment_diffusion_first_residual_6d = (
                                            np.asarray(local_delta, dtype=np.float32).tolist()
                                        )
                                        self._last_alignment_diffusion_pos_norm = float(np.linalg.norm(local_delta[:3]))
                                        self._last_alignment_diffusion_yaw_abs = float(abs(local_delta[5]))
                                    else:
                                        self._bump_alignment_diffusion_hist(
                                            self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
                                        )
                                        return False
                                else:
                                    self._bump_alignment_diffusion_hist(
                                        self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
                                    )
                                    return False
                    else:
                        if projected_ok:
                            local_delta = projected_local
                            world_delta = projected_world
                            self._last_alignment_diffusion_phase1_bridge_soft_apply = True
                            self._last_alignment_diffusion_phase1_bridge_soft_apply_reason = "workspace_project"
                            self._last_alignment_diffusion_workspace_projected = True
                            self._last_alignment_diffusion_workspace_project_reason = "phase1_boundary_projection_no_scale"
                            self._last_alignment_diffusion_workspace_violation = float(projected_violation)
                            self._last_alignment_diffusion_first_residual_6d = (
                                np.asarray(local_delta, dtype=np.float32).tolist()
                            )
                            self._last_alignment_diffusion_pos_norm = float(np.linalg.norm(local_delta[:3]))
                            self._last_alignment_diffusion_yaw_abs = float(abs(local_delta[5]))
                        else:
                            self._last_alignment_diffusion_safety_reject = True
                            self._last_alignment_diffusion_hard_reject = True
                            self._last_alignment_diffusion_block_reason = "workspace"
                            self._alignment_diffusion_safety_reject_count += 1
                            self._alignment_diffusion_hard_reject_count += 1
                            if phase1_bridge_segment and self.enable_phase1_bridge_risk_soft_apply:
                                fallback_local = self._phase1_bridge_soft_apply_delta(local_delta, reason="workspace")
                                fallback_world = local_delta_to_world(fallback_local, current_quat)
                                fallback_violation = self.safety.workspace_violation(
                                    pose_arr[:3] + base_exec_world[:3] + fallback_world[:3]
                                )
                                if fallback_violation <= workspace_tolerance:
                                    self._last_alignment_diffusion_safety_reject = False
                                    self._last_alignment_diffusion_hard_reject = False
                                    local_delta = fallback_local
                                    world_delta = fallback_world
                                    self._last_alignment_diffusion_scale_down *= 0.10
                                    self._last_alignment_diffusion_workspace_violation = float(fallback_violation)
                                    self._last_alignment_diffusion_first_residual_6d = (
                                        np.asarray(local_delta, dtype=np.float32).tolist()
                                    )
                                    self._last_alignment_diffusion_pos_norm = float(np.linalg.norm(local_delta[:3]))
                                    self._last_alignment_diffusion_yaw_abs = float(abs(local_delta[5]))
                                else:
                                    self._bump_alignment_diffusion_hist(
                                        self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
                                    )
                                    return False
                            else:
                                self._bump_alignment_diffusion_hist(
                                    self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
                                )
                                return False

        self._last_alignment_diffusion_local_delta = np.asarray(local_delta, dtype=np.float32).copy()
        self._last_alignment_diffusion_world_delta = np.asarray(world_delta, dtype=np.float32).copy()
        if not apply_enabled:
            self._last_alignment_diffusion_block_reason = "shadow_only"
            self._bump_alignment_diffusion_hist(
                self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
            )
            return False

        # The first runtime version executes a single bounded residual per
        # re-observation.  execute_steps is traced for contract parity, but we
        # avoid unrolling multiple learned steps through a single open-loop
        # command until the shadow/audit path proves the sequence reliable.
        _ = execute_steps
        applied_world = np.asarray(world_delta, dtype=np.float32).copy()
        if self.alignment_diffusion_apply_mode == "blend":
            a_exec[:6] = np.asarray(base_exec_world, dtype=np.float32) + 0.5 * applied_world
        elif self.alignment_diffusion_apply_mode == "rewrite_micro" and bucket == "micro_insert" and not tc_enabled:
            a_exec[:6] = applied_world
        else:
            a_exec[:6] = np.asarray(base_exec_world, dtype=np.float32) + applied_world
        self._alignment_diffusion_apply_count += 1
        self._last_alignment_diffusion_applied = True
        self._last_alignment_diffusion_block_reason = "applied"
        self._bump_alignment_diffusion_hist(
            self._alignment_diffusion_block_reason_hist, self._last_alignment_diffusion_block_reason
        )
        return True

    def _try_alignment_v3_apply(self, a_exec: np.ndarray, current_quat: np.ndarray) -> bool:
        """Apply the latest v3 direct-local residual as a small additive assist."""
        self._last_alignment_v3_apply_applied = False
        self._last_alignment_v3_apply_block_reason = "disabled"
        self._last_alignment_v3_apply_local_delta = None
        self._last_alignment_v3_apply_world_delta = None
        self._last_alignment_v3_apply_pos_norm = 0.0
        self._last_alignment_v3_apply_yaw_abs = 0.0

        if not self.enable_alignment_v3_apply:
            return False
        if self.alignment_v3_shadow_controller is None:
            self._last_alignment_v3_apply_block_reason = "no_controller"
            return False
        if not self._last_alignment_v3_shadow_active:
            self._last_alignment_v3_apply_block_reason = "shadow_inactive"
            return False
        if self.alignment_v3_apply_only_when_gate_pass:
            gate_ok = bool(
                self._last_alignment_v3_shadow_cur_xy is not None
                and self._last_alignment_v3_shadow_cur_z is not None
                and self._last_alignment_v3_shadow_cur_xy <= self.alignment_near_zone_xy_threshold
                and self._last_alignment_v3_shadow_cur_z <= self.alignment_near_zone_z_threshold
            )
            if not gate_ok:
                self._last_alignment_v3_apply_block_reason = "gate"
                return False
        if self.alignment_v3_apply_require_improve and not (
            self._last_alignment_v3_shadow_xy_improved
            and self._last_alignment_v3_shadow_z_improved
            and self._last_alignment_v3_shadow_yaw_improved
        ):
            self._last_alignment_v3_apply_block_reason = "pred_no_improve"
            return False

        pred = self._last_alignment_v3_shadow_pred_residual_6d
        if pred is None:
            self._last_alignment_v3_apply_block_reason = "no_residual"
            return False
        local_delta = np.asarray(pred, dtype=np.float32).reshape(-1)
        if local_delta.size < 6 or not np.all(np.isfinite(local_delta[:6])):
            self._last_alignment_v3_apply_block_reason = "bad_residual"
            return False
        local_delta = local_delta[:6].copy() * float(self.alignment_v3_apply_scale)

        pos_norm = float(np.linalg.norm(local_delta[:3]))
        if pos_norm > self.alignment_v3_apply_max_pos and pos_norm > 1e-8:
            local_delta[:3] *= float(self.alignment_v3_apply_max_pos / pos_norm)
        yaw_abs = float(abs(local_delta[5]))
        if yaw_abs > self.alignment_v3_apply_max_yaw and yaw_abs > 1e-8:
            local_delta[5] *= float(self.alignment_v3_apply_max_yaw / yaw_abs)

        if float(np.linalg.norm(local_delta[:3])) <= 1e-8 and float(abs(local_delta[5])) <= 1e-8:
            self._last_alignment_v3_apply_block_reason = "zero"
            return False

        world_delta = local_delta_to_world(local_delta, current_quat)
        a_exec[:6] = np.asarray(a_exec[:6], dtype=np.float32) + world_delta
        self._alignment_v3_shadow_apply_like_count += 1
        self._last_alignment_v3_apply_applied = True
        self._last_alignment_v3_apply_block_reason = "applied"
        self._last_alignment_v3_apply_local_delta = np.asarray(local_delta, dtype=np.float32).copy()
        self._last_alignment_v3_apply_world_delta = np.asarray(world_delta, dtype=np.float32).copy()
        self._last_alignment_v3_apply_pos_norm = float(np.linalg.norm(local_delta[:3]))
        self._last_alignment_v3_apply_yaw_abs = float(abs(local_delta[5]))
        return True

    def _try_v2_assist(self, outputs: dict, a_exec: np.ndarray, current_quat: np.ndarray) -> bool:
        """Try to apply v2 near-zone assist. Returns True if applied."""
        self._last_v2_assist_applied = False
        if not self.enable_v2_nearzone_assist:
            return False
        # Keep the predictor as a diagnostic source unless an explicit assist
        # policy opts in. Its runtime contract is still under validation.
        if str(outputs.get("v2_delta_source", "none")) == "target_delta_predictor":
            return False
        _v2_gate_ok = bool(outputs.get("v2_gate_pass", False))
        _v2_allow = (not self.v2_assist_only_when_gate_pass) or _v2_gate_ok
        if not _v2_allow:
            return False
        _v2_delta = outputs.get("v2_selected_delta")
        if _v2_delta is None or np.asarray(_v2_delta).size < 6:
            return False
        _v2_delta = np.asarray(_v2_delta, dtype=np.float32).reshape(6)
        _v2_scaled = _v2_delta * self.v2_assist_scale_cap
        _v2_pos_norm = float(np.linalg.norm(_v2_scaled[:3]))
        _v2_rot_norm = float(np.linalg.norm(_v2_scaled[3:6]))
        if _v2_pos_norm > self.v2_assist_max_pos:
            _v2_scaled[:3] *= self.v2_assist_max_pos / max(_v2_pos_norm, 1e-8)
        if _v2_rot_norm > self.v2_assist_max_rot:
            _v2_scaled[3:6] *= self.v2_assist_max_rot / max(_v2_rot_norm, 1e-8)
        _v2_world = local_delta_to_world(_v2_scaled, current_quat)
        a_exec[:6] = a_exec[:6] + _v2_world
        self._v2_assist_apply_count += 1
        self._last_v2_assist_applied = True
        return True

    def _check_near_zone_gate(self, controller) -> bool:
        """Check whether the current pose is close enough to allow alignment takeover.

        Uses _runtime_handoff_metrics (xy_error, abs_z_error) from the handoff
        provider as the primary error source.  Falls back to
        _runtime_current_delta_basin_target if handoff metrics are unavailable.
        Returns True when the gate is disabled or the pose is within near-zone
        thresholds.
        """
        self._near_zone_gate_eval_count += 1
        self._last_near_zone_gate_pass = False
        self._last_near_zone_block_reason = "none"
        if not self.enable_alignment_near_zone_gate:
            self._last_near_zone_gate_pass = True
            self._near_zone_gate_pass_count += 1
            self._last_near_zone_block_reason = "disabled"
            return True
        if controller is None:
            self._last_near_zone_block_reason = "no_controller"
            self._near_zone_gate_block_count += 1
            return False

        # Primary source: handoff metrics (requires --handoff_provider_ckpt)
        handoff_metrics = dict(getattr(controller, "_runtime_handoff_metrics", {}) or {})
        hm_xy = handoff_metrics.get("xy_error")
        hm_z = handoff_metrics.get("abs_z_error")
        if hm_xy is not None and hm_z is not None and np.isfinite(float(hm_xy)) and np.isfinite(float(hm_z)):
            xy_error = float(hm_xy)
            z_error = float(hm_z)
        else:
            # Fallback: delta to motion target (may be zero for canonical fallback)
            delta = np.asarray(
                getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            if delta.size < 3:
                self._last_near_zone_block_reason = "no_error_source"
                self._near_zone_gate_block_count += 1
                return False
            xy_error = float(np.linalg.norm(delta[:2]))
            z_error = float(abs(delta[2]))

        self._last_near_zone_xy_error = xy_error
        self._last_near_zone_z_error = z_error
        xy_pass = xy_error <= self.alignment_near_zone_xy_threshold
        z_pass = z_error <= self.alignment_near_zone_z_threshold
        if xy_pass and z_pass:
            self._last_near_zone_gate_pass = True
            self._near_zone_gate_pass_count += 1
            return True
        if not xy_pass and not z_pass:
            self._last_near_zone_block_reason = "xy_and_z"
        elif not xy_pass:
            self._last_near_zone_block_reason = "xy"
        else:
            self._last_near_zone_block_reason = "z"
        self._near_zone_gate_block_count += 1
        return False

    @staticmethod
    def make_gripper_context(a_base_7d=None, future_gripper_actions=None) -> np.ndarray:
        if a_base_7d is None:
            return np.zeros(3, dtype=np.float32)
        values = [float(np.asarray(a_base_7d, dtype=np.float32)[6])]
        if future_gripper_actions is not None:
            values.extend(float(x) for x in future_gripper_actions)
        arr = np.asarray(values, dtype=np.float32)
        return np.asarray([arr[0], float(np.min(arr)), float(np.mean(arr))], dtype=np.float32)

    @staticmethod
    def compute_depth_proximity(wrist_depth) -> Optional[float]:
        if wrist_depth is None:
            return None
        if hasattr(wrist_depth, "detach"):
            wrist_depth = wrist_depth.detach().float().cpu().numpy()
        depth_arr = np.asarray(wrist_depth, dtype=np.float32).squeeze()
        valid = depth_arr[np.isfinite(depth_arr)]
        if valid.size == 0:
            return None
        return float(np.percentile(valid, 5.0))

    @staticmethod
    def _extract_current_quat(proprio, gripper_pose=None) -> np.ndarray:
        if gripper_pose is not None:
            return np.asarray(gripper_pose[3:7], dtype=np.float32)
        if proprio is not None:
            arr = np.asarray(proprio, dtype=np.float32)
            if arr.shape[0] >= 14:
                return arr[10:14].copy()
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    @staticmethod
    def _to_tensor(value, shape, device, dtype):
        import torch

        if value is None:
            return torch.zeros(*shape, device=device, dtype=dtype)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        return value.unsqueeze(0).to(device=device, dtype=dtype)

    @staticmethod
    def _to_rgb_tensor(value, shape, device, dtype):
        import torch

        if value is None:
            return torch.zeros(*shape, device=device, dtype=dtype)
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        ten = torch.from_numpy(arr)
        if ten.dtype != torch.float32 and ten.dtype != torch.float64:
            ten = ten.float()
        if float(ten.max().item()) > 1.5:
            ten = ten / 255.0
        return ten.unsqueeze(0).to(device=device, dtype=dtype)

    def _maybe_rerank_near_ready_topk(
        self,
        controller,
        *,
        fr,
        wr,
        wd,
        pr,
        gc,
        current_delta,
        candidate_actions_src,
        masked_scores,
        pred_idx,
        step_scale,
        valid_mask,
    ):
        import torch

        self._near_ready_rerank_eval_count += 1
        self._last_near_ready_rerank_gate_open = False
        self._last_near_ready_rerank_applied = False
        self._last_near_ready_rerank_changed = False
        self._last_near_ready_rerank_prev_index = int(pred_idx[0].item()) if pred_idx is not None else -1
        self._last_near_ready_rerank_new_index = self._last_near_ready_rerank_prev_index
        self._last_near_ready_rerank_topk = 0
        self._last_near_ready_rerank_cur_xy = np.nan
        self._last_near_ready_rerank_cur_z = np.nan
        self._last_near_ready_rerank_cur_yaw = np.nan
        self._last_near_ready_rerank_gate_xy_max = np.nan
        self._last_near_ready_rerank_gate_z_max = np.nan
        self._last_near_ready_rerank_gate_yaw_max = np.nan
        rerank_model = getattr(controller, "_near_ready_rerank_model", None)
        if rerank_model is None:
            return pred_idx
        handoff_thresholds = dict(getattr(controller, "_runtime_handoff_metric_thresholds", {}) or {})
        handoff_xy = float(handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        handoff_z = float(handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        if handoff_xy <= 0.0 or handoff_z <= 0.0 or handoff_yaw <= 0.0:
            return pred_idx
        cur_xy = float(np.linalg.norm(current_delta[:2])) if current_delta.size >= 2 else np.inf
        cur_z = float(abs(current_delta[2])) if current_delta.size >= 3 else np.inf
        cur_yaw = float(abs(current_delta[5])) if current_delta.size >= 6 else np.inf
        gate_xy_max = handoff_xy * float(getattr(controller, "_near_ready_rerank_xy_multiplier", 3.0))
        gate_z_max = handoff_z * float(getattr(controller, "_near_ready_rerank_z_multiplier", 2.0))
        gate_yaw_max = handoff_yaw * float(getattr(controller, "_near_ready_rerank_yaw_multiplier", 2.0))
        self._last_near_ready_rerank_cur_xy = cur_xy
        self._last_near_ready_rerank_cur_z = cur_z
        self._last_near_ready_rerank_cur_yaw = cur_yaw
        self._last_near_ready_rerank_gate_xy_max = gate_xy_max
        self._last_near_ready_rerank_gate_z_max = gate_z_max
        self._last_near_ready_rerank_gate_yaw_max = gate_yaw_max
        if not (
            cur_xy <= gate_xy_max
            and cur_z <= gate_z_max
            and cur_yaw <= gate_yaw_max
        ):
            return pred_idx
        self._last_near_ready_rerank_gate_open = True
        self._near_ready_rerank_gate_pass_count += 1
        topk = int(max(1, getattr(controller, "_near_ready_rerank_topk", 3)))
        topk = min(topk, int(masked_scores.shape[1]))
        self._last_near_ready_rerank_topk = topk
        topk_scores, topk_idx = torch.topk(masked_scores, k=topk, dim=-1)
        if topk <= 1:
            return pred_idx
        device = masked_scores.device
        dtype = masked_scores.dtype
        runtime_xy = max(cur_xy / max(handoff_xy, 1e-6), 0.0)
        runtime_yaw = max(cur_yaw / max(handoff_yaw, 1e-6), 0.0)
        rerank_costs = []
        score_weight = float(getattr(controller, "_near_ready_rerank_weight_score", 0.10))
        xy_weight = float(getattr(controller, "_near_ready_rerank_weight_xy", 1.0))
        yaw_weight = float(getattr(controller, "_near_ready_rerank_weight_yaw", 0.75))
        with torch.no_grad():
            for k_idx in range(topk):
                cand_idx = int(topk_idx[0, k_idx].item())
                cand_local = np.asarray(candidate_actions_src[cand_idx], dtype=np.float32)
                cand_scale = float(step_scale[0, cand_idx].detach().float().cpu().item())
                cand_scale = float(np.clip(cand_scale, 0.0, 2.0))
                next_delta = current_delta - (cand_local * cand_scale)
                next_xy = float(np.linalg.norm(next_delta[:2])) if next_delta.size >= 2 else np.inf
                next_yaw = float(abs(next_delta[5])) if next_delta.size >= 6 else np.inf
                runtime_xyyaw_norm = torch.tensor(
                    [[next_xy / max(handoff_xy, 1e-6), next_yaw / max(handoff_yaw, 1e-6)]],
                    device=device,
                    dtype=dtype,
                )
                rerank_outputs = rerank_model(
                    front_rgb=fr.float(),
                    wrist_rgb=wr.float(),
                    wrist_depth=wd.float(),
                    proprio=pr.float(),
                    gripper_context=gc.float(),
                    runtime_xyyaw_norm=runtime_xyyaw_norm.float(),
                    substage_id=torch.tensor([int(self.manager.substage)], device=device, dtype=torch.long),
                    contact_state=torch.tensor([int(self.manager.contact_state)], device=device, dtype=torch.long),
                    stage_target_mode=torch.tensor([int(self.manager.stage_target_mode)], device=device, dtype=torch.long),
                )
                pred_xyyaw = rerank_outputs["xyyaw_norm"].reshape(-1).float()
                base_score = float(topk_scores[0, k_idx].detach().float().cpu().item())
                rerank_cost = (
                    xy_weight * float(pred_xyyaw[0].item())
                    + yaw_weight * float(pred_xyyaw[1].item())
                    - score_weight * base_score
                )
                rerank_costs.append((rerank_cost, cand_idx))
        if not rerank_costs:
            return pred_idx
        rerank_costs.sort(key=lambda x: x[0])
        best_idx = int(rerank_costs[0][1])
        self._last_near_ready_rerank_applied = True
        self._near_ready_rerank_apply_count += 1
        self._last_near_ready_rerank_new_index = best_idx
        if best_idx != self._last_near_ready_rerank_prev_index:
            self._last_near_ready_rerank_changed = True
            self._near_ready_rerank_change_count += 1
        return torch.tensor([best_idx], device=device, dtype=torch.long)

    def _select_pose_field_scorer_model(self, controller, current_delta):
        self._near_ready_specialist_eval_count += 1
        self._last_near_ready_specialist_gate_open = False
        self._last_near_ready_specialist_active = False
        self._last_near_ready_specialist_cur_xy = np.nan
        self._last_near_ready_specialist_cur_z = np.nan
        self._last_near_ready_specialist_cur_yaw = np.nan
        self._last_near_ready_specialist_gate_xy_max = np.nan
        self._last_near_ready_specialist_gate_z_max = np.nan
        self._last_near_ready_specialist_gate_yaw_max = np.nan
        specialist = getattr(controller, "_near_ready_alignment_controller", None)
        if specialist is None:
            return controller, False
        current_delta = np.asarray(current_delta, dtype=np.float32).reshape(-1)
        if current_delta.size < 6:
            return controller, False
        handoff_thresholds = dict(getattr(controller, "_runtime_handoff_metric_thresholds", {}) or {})
        handoff_xy = float(handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        handoff_z = float(handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        if handoff_xy <= 0.0:
            handoff_xy = max(float(getattr(controller, "_ready_band_xy_threshold", self.close_veto_xy_threshold)), 1e-4)
        if handoff_z <= 0.0:
            handoff_z = max(float(getattr(controller, "_ready_band_abs_z_threshold", self.close_veto_abs_z_threshold)), 1e-4)
        if handoff_yaw <= 0.0:
            handoff_yaw = max(
                float(getattr(controller, "_ready_band_yaw_threshold", -1.0)),
                0.12,
            )
        cur_xy = float(np.linalg.norm(current_delta[:2]))
        cur_z = float(abs(current_delta[2]))
        cur_yaw = float(abs(current_delta[5]))
        gate_xy_max = handoff_xy * float(getattr(controller, "_near_ready_alignment_xy_multiplier", 2.5))
        gate_z_max = handoff_z * float(getattr(controller, "_near_ready_alignment_z_multiplier", 4.0))
        gate_yaw_max = handoff_yaw * float(getattr(controller, "_near_ready_alignment_yaw_multiplier", 2.0))
        self._last_near_ready_specialist_cur_xy = cur_xy
        self._last_near_ready_specialist_cur_z = cur_z
        self._last_near_ready_specialist_cur_yaw = cur_yaw
        self._last_near_ready_specialist_gate_xy_max = gate_xy_max
        self._last_near_ready_specialist_gate_z_max = gate_z_max
        self._last_near_ready_specialist_gate_yaw_max = gate_yaw_max
        if not (cur_xy <= gate_xy_max and cur_z <= gate_z_max and cur_yaw <= gate_yaw_max):
            return controller, False
        self._last_near_ready_specialist_gate_open = True
        self._near_ready_specialist_gate_pass_count += 1
        self._last_near_ready_specialist_active = True
        self._near_ready_specialist_use_count += 1
        return specialist, True

    def _apply_near_ready_residual_adapter(
        self,
        controller,
        scores,
        candidate_actions,
        candidate_mask,
        gripper_context,
        phase_age,
        steps_since_last_replan,
        current_delta,
        current_dx_sign,
        current_dy_sign,
        current_dyaw_sign,
        basin_distance_bin,
        step_idx: int,
    ):
        import torch

        self._near_ready_residual_eval_count += 1
        self._last_near_ready_residual_gate_open = False
        self._last_near_ready_residual_applied = False
        self._last_near_ready_residual_changed = False
        self._last_near_ready_residual_prev_index = -1
        self._last_near_ready_residual_new_index = -1

        adapter = getattr(controller, "_near_ready_residual_adapter", None)
        if adapter is None:
            return scores, None
        if hasattr(current_delta, "detach"):
            current_delta_np = current_delta.detach().float().cpu().numpy().reshape(-1)
        else:
            current_delta_np = np.asarray(current_delta, dtype=np.float32).reshape(-1)
        if current_delta_np.size < 6:
            return scores, None
        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        handoff_z = float(handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        if handoff_xy <= 0.0:
            handoff_xy = max(float(getattr(controller, "_ready_band_xy_threshold", self.close_veto_xy_threshold)), 1e-4)
        if handoff_z <= 0.0:
            handoff_z = max(float(getattr(controller, "_ready_band_abs_z_threshold", self.close_veto_abs_z_threshold)), 1e-4)
        if handoff_yaw <= 0.0:
            handoff_yaw = max(float(getattr(controller, "_ready_band_yaw_threshold", -1.0)), 0.12)
        cur_xy = float(np.linalg.norm(current_delta_np[:2]))
        cur_z = float(abs(current_delta_np[2]))
        cur_yaw = float(abs(current_delta_np[5]))
        gate_xy = handoff_xy * float(getattr(controller, "_near_ready_residual_xy_multiplier", 4.0))
        gate_z = handoff_z * float(getattr(controller, "_near_ready_residual_z_multiplier", 2.0))
        gate_yaw = handoff_yaw * float(getattr(controller, "_near_ready_residual_yaw_multiplier", 2.0))
        gripper_open = None
        if gripper_context is not None:
            if hasattr(gripper_context, "detach"):
                gc_np = gripper_context.detach().float().cpu().numpy().reshape(-1)
            else:
                gc_np = np.asarray(gripper_context, dtype=np.float32).reshape(-1)
            if gc_np.size > 0:
                gripper_open = float(gc_np[0])
        support_ok = bool(self._delta_within_band(current_delta_np, controller, band="outer"))
        refine_ok = bool(
            gripper_open is not None
            and self._alignment_refine_band_ready(controller, gripper_open)
        )
        close_ready = bool(
            gripper_open is not None
            and self._close_veto_ready(controller, gripper_open, step_idx=step_idx)
        )
        gate_open = bool(
            self.manager.phase == StagePhase.ALIGN
            and gripper_open is not None
            and gripper_open >= self.alignment_open_threshold
            and (support_ok or refine_ok)
            and (not close_ready)
            and cur_xy <= gate_xy
            and cur_z <= gate_z
            and cur_yaw <= gate_yaw
        )
        if not gate_open:
            return scores, None
        self._last_near_ready_residual_gate_open = True
        self._near_ready_residual_gate_pass_count += 1
        with torch.no_grad():
            residual = adapter(
                candidate_actions=candidate_actions,
                gripper_context=gripper_context,
                phase_age=phase_age,
                steps_since_last_replan=steps_since_last_replan,
                current_delta_basin_target=current_delta,
                current_dx_sign=current_dx_sign,
                current_dy_sign=current_dy_sign,
                current_dyaw_sign=current_dyaw_sign,
                basin_distance_bin=basin_distance_bin,
                candidate_mask=candidate_mask,
            )
        self._near_ready_residual_apply_count += 1
        self._last_near_ready_residual_applied = True
        return scores + residual.to(dtype=scores.dtype), residual

    def _apply_near_ready_group_residual_adapter(
        self,
        controller,
        group_logits,
        candidate_group_index,
        candidate_mask,
        gripper_context,
        phase_age,
        steps_since_last_replan,
        current_delta,
        current_dx_sign,
        current_dy_sign,
        current_dyaw_sign,
        basin_distance_bin,
        step_idx: int,
    ):
        import torch

        self._near_ready_group_residual_eval_count += 1
        self._last_near_ready_group_residual_gate_open = False
        self._last_near_ready_group_residual_applied = False
        self._last_near_ready_group_residual_changed = False
        self._last_near_ready_group_residual_prev_group = -1
        self._last_near_ready_group_residual_new_group = -1

        adapter = getattr(controller, "_near_ready_residual_adapter", None)
        if adapter is None or str(getattr(adapter, "_controller_type", "")) != "near_ready_group_residual_adapter":
            return group_logits, None
        if hasattr(current_delta, "detach"):
            current_delta_np = current_delta.detach().float().cpu().numpy().reshape(-1)
        else:
            current_delta_np = np.asarray(current_delta, dtype=np.float32).reshape(-1)
        if current_delta_np.size < 6:
            return group_logits, None

        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        handoff_z = float(handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        if handoff_xy <= 0.0:
            handoff_xy = max(float(getattr(controller, "_ready_band_xy_threshold", self.close_veto_xy_threshold)), 1e-4)
        if handoff_z <= 0.0:
            handoff_z = max(float(getattr(controller, "_ready_band_abs_z_threshold", self.close_veto_abs_z_threshold)), 1e-4)
        if handoff_yaw <= 0.0:
            handoff_yaw = max(float(getattr(controller, "_ready_band_yaw_threshold", -1.0)), 0.12)
        cur_xy = float(np.linalg.norm(current_delta_np[:2]))
        cur_z = float(abs(current_delta_np[2]))
        cur_yaw = float(abs(current_delta_np[5]))
        gate_xy = handoff_xy * float(getattr(controller, "_near_ready_residual_xy_multiplier", 4.0))
        gate_z = handoff_z * float(getattr(controller, "_near_ready_residual_z_multiplier", 2.0))
        gate_yaw = handoff_yaw * float(getattr(controller, "_near_ready_residual_yaw_multiplier", 2.0))

        gripper_open = None
        if gripper_context is not None:
            if hasattr(gripper_context, "detach"):
                gc_np = gripper_context.detach().float().cpu().numpy().reshape(-1)
            else:
                gc_np = np.asarray(gripper_context, dtype=np.float32).reshape(-1)
            if gc_np.size > 0:
                gripper_open = float(gc_np[0])
        support_ok = bool(self._delta_within_band(current_delta_np, controller, band="outer"))
        refine_ok = bool(
            gripper_open is not None
            and self._alignment_refine_band_ready(controller, gripper_open)
        )
        close_ready = bool(
            gripper_open is not None
            and self._close_veto_ready(controller, gripper_open, step_idx=step_idx)
        )
        gate_open = bool(
            self.manager.phase == StagePhase.ALIGN
            and gripper_open is not None
            and gripper_open >= self.alignment_open_threshold
            and (support_ok or refine_ok)
            and (not close_ready)
            and cur_xy <= gate_xy
            and cur_z <= gate_z
            and cur_yaw <= gate_yaw
        )
        if not gate_open:
            return group_logits, None

        self._last_near_ready_group_residual_gate_open = True
        self._near_ready_group_residual_gate_pass_count += 1

        valid_mask = candidate_mask > 0.5
        num_groups = int(group_logits.shape[1])
        group_valid = []
        for group_id in range(num_groups):
            group_valid.append(torch.any(candidate_group_index.eq(group_id) & valid_mask, dim=1))
        group_valid = torch.stack(group_valid, dim=1)

        with torch.no_grad():
            group_residual = adapter(
                gripper_context=gripper_context,
                phase_age=phase_age,
                steps_since_last_replan=steps_since_last_replan,
                current_delta_basin_target=current_delta,
                current_dx_sign=current_dx_sign,
                current_dy_sign=current_dy_sign,
                current_dyaw_sign=current_dyaw_sign,
                basin_distance_bin=basin_distance_bin,
                group_valid_mask=group_valid,
            )

        group_logits_masked = group_logits.masked_fill(~group_valid, -1e9)
        prev_group = int(group_logits_masked.argmax(dim=-1).item())
        final_group_logits = group_logits + group_residual.to(dtype=group_logits.dtype)
        final_group_logits_masked = final_group_logits.masked_fill(~group_valid, -1e9)
        new_group = int(final_group_logits_masked.argmax(dim=-1).item())

        self._near_ready_group_residual_apply_count += 1
        self._last_near_ready_group_residual_applied = True
        self._last_near_ready_group_residual_prev_group = prev_group
        self._last_near_ready_group_residual_new_group = new_group
        if prev_group != new_group:
            self._near_ready_group_residual_change_count += 1
            self._last_near_ready_group_residual_changed = True
        return final_group_logits, group_residual

    def _run_b1_group_selector_shadow(
        self,
        controller,
        *,
        front_rgb,
        wrist_rgb,
        wrist_depth,
        proprio,
        gripper_context,
        phase_age,
        steps_since_last_replan,
        current_delta,
        current_dx_sign,
        current_dy_sign,
        current_dyaw_sign,
        basin_distance_bin,
        candidate_actions,
        candidate_actions_src,
        candidate_group_index,
        group_valid,
        baseline_group,
        valid_mask,
        step_idx: int,
    ):
        import torch

        self._b1_group_shadow_eval_count += 1
        self._last_b1_group_shadow_gate_open = False
        self._last_b1_group_shadow_pred_group = -1
        self._last_b1_group_shadow_baseline_group = int(baseline_group)
        self._last_b1_group_shadow_teacher_group = -1
        self._last_b1_group_shadow_changed = False
        self._last_b1_group_shadow_teacher_disagree = False
        self._last_b1_group_shadow_teacher_group_valid = False
        self._last_b1_group_shadow_close_neighborhood = False
        self._last_b1_group_shadow_close_group_changed = False
        self._last_b1_group_shadow_margin = np.nan
        self._last_b1_apply_gate_prob = np.nan
        self._last_b1_apply_gate_threshold = np.nan
        self._last_b1_apply_gate_apply = False
        self._last_b1_apply_gate_vetoed = False
        self._last_b1_group_bounded_applied = False
        self._last_b1_group_bounded_prev_group = -1
        self._last_b1_group_bounded_new_group = -1

        shadow = getattr(controller, "_student_group_selector_shadow", None)
        if not shadow:
            return
        if hasattr(current_delta, "detach"):
            current_delta_np = current_delta.detach().float().cpu().numpy().reshape(-1)
        else:
            current_delta_np = np.asarray(current_delta, dtype=np.float32).reshape(-1)
        if current_delta_np.size < 6:
            return

        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        handoff_z = float(handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        if handoff_xy <= 0.0:
            handoff_xy = max(float(getattr(controller, "_ready_band_xy_threshold", self.close_veto_xy_threshold)), 1e-4)
        if handoff_z <= 0.0:
            handoff_z = max(float(getattr(controller, "_ready_band_abs_z_threshold", self.close_veto_abs_z_threshold)), 1e-4)
        if handoff_yaw <= 0.0:
            handoff_yaw = max(float(getattr(controller, "_ready_band_yaw_threshold", -1.0)), 0.12)
        cur_xy = float(np.linalg.norm(current_delta_np[:2]))
        cur_z = float(abs(current_delta_np[2]))
        cur_yaw = float(abs(current_delta_np[5]))

        gripper_open = None
        if gripper_context is not None:
            if hasattr(gripper_context, "detach"):
                gc_np = gripper_context.detach().float().cpu().numpy().reshape(-1)
            else:
                gc_np = np.asarray(gripper_context, dtype=np.float32).reshape(-1)
            if gc_np.size > 0:
                gripper_open = float(gc_np[0])
        support_ok = bool(self._delta_within_band(current_delta_np, controller, band="outer"))
        refine_ok = bool(
            gripper_open is not None
            and self._alignment_refine_band_ready(controller, gripper_open)
        )
        close_ready = bool(
            gripper_open is not None
            and self._close_veto_ready(controller, gripper_open, step_idx=step_idx)
        )
        close_neighborhood = bool(
            close_ready
            or (
                cur_xy <= max(handoff_xy * 2.5, 0.012)
                and cur_z <= max(handoff_z * 8.0, 0.008)
                and cur_yaw <= max(handoff_yaw * 2.5, 0.18)
            )
        )
        gate_open = bool(
            self.manager.phase == StagePhase.ALIGN
            and gripper_open is not None
            and gripper_open >= self.alignment_open_threshold
            and (support_ok or refine_ok or close_neighborhood)
        )
        if self.b1_group_shadow_gate_mode == "close_only":
            gate_open = bool(gate_open and close_neighborhood)
        self._last_b1_group_shadow_close_neighborhood = close_neighborhood
        if close_neighborhood:
            self._b1_group_shadow_close_count += 1
        if not gate_open:
            return

        self._last_b1_group_shadow_gate_open = True
        self._b1_group_shadow_gate_pass_count += 1
        handoff_model = shadow.get("handoff_model")
        group_model = shadow.get("group_model")
        if handoff_model is None or group_model is None:
            return

        device = front_rgb.device
        dtype = front_rgb.dtype
        phase_id = torch.tensor([int(self.manager.phase)], device=device, dtype=torch.long)
        substage_id = torch.tensor([int(getattr(self.manager, "substage", 0))], device=device, dtype=torch.long)
        contact_state = torch.tensor([int(getattr(self.manager, "contact_state", 0))], device=device, dtype=torch.long)
        stage_target_mode = torch.tensor([int(getattr(self.manager, "stage_target_mode", 0))], device=device, dtype=torch.long)
        with torch.no_grad():
            handoff_out = handoff_model(
                front_rgb=front_rgb.to(device=device, dtype=dtype),
                wrist_rgb=wrist_rgb.to(device=device, dtype=dtype),
                wrist_depth=wrist_depth.to(device=device, dtype=dtype),
                proprio=proprio.to(device=device, dtype=dtype),
                gripper_context=gripper_context.to(device=device, dtype=dtype),
                proxy_current_delta_basin_target=current_delta.to(device=device, dtype=dtype),
                current_dx_sign=current_dx_sign.to(device=device),
                current_dy_sign=current_dy_sign.to(device=device),
                current_dyaw_sign=current_dyaw_sign.to(device=device),
                basin_distance_bin=basin_distance_bin.to(device=device),
                substage_id=substage_id,
                contact_state=contact_state,
                stage_target_mode=stage_target_mode,
            )
            selector_logits = group_model(
                handoff_latent=handoff_out["latent"],
                proxy_current_delta_basin_target=current_delta.to(device=device, dtype=dtype),
                current_dx_sign=current_dx_sign.to(device=device),
                current_dy_sign=current_dy_sign.to(device=device),
                current_dyaw_sign=current_dyaw_sign.to(device=device),
                basin_distance_bin=basin_distance_bin.to(device=device),
            )
        common = min(int(selector_logits.shape[1]), int(group_valid.shape[1]))
        selector_logits = selector_logits[:, :common]
        group_valid = group_valid[:, :common]
        masked = selector_logits.masked_fill(~group_valid, -1e9)
        pred_group = int(masked.argmax(dim=-1).item())
        top2 = torch.topk(masked, k=min(2, masked.shape[1]), dim=-1).values
        if top2.shape[1] > 1:
            self._last_b1_group_shadow_margin = float((top2[:, 0] - top2[:, 1]).item())
        else:
            self._last_b1_group_shadow_margin = np.inf
        self._last_b1_group_shadow_pred_group = pred_group
        changed = bool(pred_group != int(baseline_group))
        self._last_b1_group_shadow_changed = changed
        if changed:
            self._b1_group_shadow_change_count += 1
            if close_neighborhood:
                self._b1_group_shadow_close_change_count += 1
                self._last_b1_group_shadow_close_group_changed = True

        apply_gate = getattr(controller, "_student_b1_apply_gate_shadow", None)
        if apply_gate:
            self._b1_apply_gate_eval_count += 1
            runtime_handoff_metrics = dict(getattr(controller, "_runtime_handoff_metrics", {}) or {})
            runtime_release_thresholds = dict(getattr(controller, "_runtime_handoff_release_metric_thresholds", {}) or {})
            runtime_handoff_aux = dict(getattr(controller, "_runtime_handoff_aux", {}) or {})
            runtime_xy_norm = float(runtime_handoff_metrics.get("xy_error", np.nan)) / max(
                float(runtime_release_thresholds.get("xy_error", handoff_xy)),
                1e-6,
            )
            runtime_z_norm = float(runtime_handoff_metrics.get("abs_z_error", np.nan)) / max(
                float(runtime_release_thresholds.get("abs_z_error", handoff_z)),
                1e-6,
            )
            runtime_yaw_norm = float(runtime_handoff_metrics.get("yaw_error", np.nan)) / max(
                float(runtime_release_thresholds.get("yaw_error", handoff_yaw)),
                1e-6,
            )
            apply_row = {
                "b1_group_shadow_margin": float(self._last_b1_group_shadow_margin),
                "alignment_runtime_basin_xy": float(cur_xy),
                "alignment_runtime_basin_z": float(cur_z),
                "alignment_runtime_basin_yaw": float(cur_yaw),
                "alignment_runtime_basin_distance": float(
                    getattr(controller, "_runtime_current_basin_distance", np.linalg.norm(current_delta_np[:2]))
                ),
                "obs_gripper_open": float(gripper_open if gripper_open is not None else 0.0),
                "refiner_alignment_planner_close_intent": float(
                    bool(getattr(controller, "_alignment_planner_close_intent", False))
                ),
                "refiner_current_close_veto_ready": float(close_ready),
                "refiner_current_close_veto_blocked": float(
                    bool(getattr(controller, "_close_veto_blocked", False))
                ),
                "handoff_ready_pred": float(bool(getattr(controller, "_runtime_handoff_ready_pred", False))),
                "refiner_substage_id": float(int(getattr(self.manager, "substage", 0))),
                "phase_before": float(int(self.manager.phase)),
                "b1_group_shadow_close_neighborhood": float(close_neighborhood),
                "refiner_current_alignment_support_satisfied": float(
                    bool(getattr(controller, "_alignment_support_satisfied", False))
                ),
                "refiner_current_alignment_support_inner_satisfied": float(
                    bool(getattr(controller, "_alignment_support_inner_satisfied", False))
                ),
                "refiner_current_alignment_support_outer_satisfied": float(
                    bool(getattr(controller, "_alignment_support_outer_satisfied", False))
                ),
                "refiner_current_alignment_refine_band_satisfied": float(
                    bool(getattr(controller, "_alignment_refine_band_satisfied", False))
                ),
                "refiner_current_alignment_takeover_band_satisfied": float(
                    bool(getattr(controller, "_alignment_takeover_band_satisfied", False))
                ),
                "refiner_alignment_close_requirement_satisfied": float(
                    bool((self._last_alignment_gate_debug or {}).get("close_requirement_satisfied", False))
                ),
                "refiner_current_close_latch_remaining": float(
                    int(getattr(controller, "_close_latch_remaining", 0))
                ),
                "refiner_last_handoff_yaw_priority_active": float(bool(self._last_handoff_yaw_priority_active)),
                "handoff_aux_pred_xy_norm": float(runtime_handoff_aux.get("pred_xy_norm", np.nan)),
                "handoff_aux_pred_abs_z_norm": float(runtime_handoff_aux.get("pred_abs_z_norm", np.nan)),
                "handoff_aux_pred_yaw_norm": float(runtime_handoff_aux.get("pred_yaw_norm", np.nan)),
                "handoff_aux_pred_band_index": float(runtime_handoff_aux.get("pred_band_index", np.nan)),
                "handoff_aux_pred_ready_prob": float(runtime_handoff_aux.get("pred_ready_prob", np.nan)),
                "handoff_aux_pred_uncertainty": float(runtime_handoff_aux.get("pred_uncertainty", np.nan)),
                "runtime_handoff_xy_norm": float(runtime_xy_norm),
                "runtime_handoff_abs_z_norm": float(runtime_z_norm),
                "runtime_handoff_yaw_norm": float(runtime_yaw_norm),
                "b1_group_shadow_pred_group": float(pred_group),
                "b1_group_shadow_baseline_group": float(int(baseline_group)),
            }
            gate_prob = float(apply_gate["predict"](apply_row))
            gate_threshold = float(apply_gate.get("threshold", 0.5))
            gate_apply = bool(gate_prob >= gate_threshold)
            gate_vetoed = False
            selection_rule = str(apply_gate.get("selection_rule", "threshold_only") or "threshold_only")
            if gate_apply and selection_rule == "yaw_uncertainty_veto":
                veto_runtime_yaw_norm_gt = float(apply_gate.get("veto_runtime_yaw_norm_gt", np.inf))
                veto_pred_uncertainty_gt = float(apply_gate.get("veto_pred_uncertainty_gt", np.inf))
                pred_uncertainty = float(runtime_handoff_aux.get("pred_uncertainty", np.nan))
                if not (np.isfinite(runtime_yaw_norm) and np.isfinite(pred_uncertainty)):
                    gate_apply = False
                    gate_vetoed = True
                elif (
                    runtime_yaw_norm > veto_runtime_yaw_norm_gt
                    and pred_uncertainty > veto_pred_uncertainty_gt
                ):
                    gate_apply = False
                    gate_vetoed = True
            if gate_apply and selection_rule == "close_yawaware_v6":
                pred_uncertainty = float(runtime_handoff_aux.get("pred_uncertainty", np.nan))
                shadow_margin = float(shadow.get("margin", self._last_b1_group_shadow_margin))
                require_yaw_ge = float(apply_gate.get("require_runtime_yaw_norm_ge", -np.inf))
                require_yaw_le = float(apply_gate.get("require_runtime_yaw_norm_le", np.inf))
                require_margin_ge = float(apply_gate.get("require_group_margin_ge", -np.inf))
                unc_or_yaw = apply_gate.get("require_uncertainty_ge_or_yaw_norm_le", None)
                pred_group_veto = apply_gate.get("veto_pred_group_if_yaw_norm_le_and_uncertainty_lt", None)
                if not (np.isfinite(runtime_yaw_norm) and np.isfinite(pred_uncertainty) and np.isfinite(shadow_margin)):
                    gate_apply = False
                    gate_vetoed = True
                elif runtime_yaw_norm < require_yaw_ge:
                    gate_apply = False
                    gate_vetoed = True
                elif runtime_yaw_norm > require_yaw_le:
                    gate_apply = False
                    gate_vetoed = True
                elif shadow_margin < require_margin_ge:
                    gate_apply = False
                    gate_vetoed = True
                elif unc_or_yaw and len(unc_or_yaw) >= 2:
                    require_unc_ge = float(unc_or_yaw[0])
                    require_yaw_le = float(unc_or_yaw[1])
                    if not (pred_uncertainty >= require_unc_ge or runtime_yaw_norm <= require_yaw_le):
                        gate_apply = False
                        gate_vetoed = True
                if gate_apply and pred_group_veto and len(pred_group_veto) >= 3:
                    veto_group = int(pred_group_veto[0])
                    veto_yaw_le = float(pred_group_veto[1])
                    veto_unc_lt = float(pred_group_veto[2])
                    if int(pred_group) == veto_group and runtime_yaw_norm <= veto_yaw_le and pred_uncertainty < veto_unc_lt:
                        gate_apply = False
                        gate_vetoed = True
            veto_group_margin_lt = float(apply_gate.get("veto_group_margin_lt", -np.inf))
            if gate_apply and np.isfinite(veto_group_margin_lt):
                shadow_margin = float(shadow.get("margin", self._last_b1_group_shadow_margin))
                if not np.isfinite(shadow_margin) or shadow_margin < veto_group_margin_lt:
                    gate_apply = False
                    gate_vetoed = True
            veto_baseline_groups = apply_gate.get("veto_baseline_groups", None)
            if gate_apply and veto_baseline_groups:
                try:
                    baseline_group_int = int(baseline_group)
                except Exception:
                    baseline_group_int = -1
                if baseline_group_int in {int(v) for v in veto_baseline_groups}:
                    gate_apply = False
                    gate_vetoed = True
            self._last_b1_apply_gate_prob = gate_prob
            self._last_b1_apply_gate_threshold = gate_threshold
            self._last_b1_apply_gate_apply = gate_apply
            self._last_b1_apply_gate_vetoed = gate_vetoed
            if gate_apply:
                self._b1_apply_gate_apply_count += 1

        teacher_delta = getattr(controller, "_runtime_teacher_current_delta_basin_target", None)
        if teacher_delta is not None:
            teacher_delta_np = np.asarray(teacher_delta, dtype=np.float32).reshape(-1)
            cand_np = np.asarray(candidate_actions_src, dtype=np.float32)
            if teacher_delta_np.size >= 6 and cand_np.ndim == 2 and cand_np.shape[0] == valid_mask.shape[1]:
                weights = np.asarray(
                    [1.0 / max(handoff_xy, 1e-4), 1.0 / max(handoff_xy, 1e-4), 1.0 / max(handoff_z, 1e-4), 0.0, 0.0, 1.0 / max(handoff_yaw, 1e-4)],
                    dtype=np.float32,
                )
                residual = teacher_delta_np[None, :6] - cand_np[:, :6]
                cost = np.linalg.norm(residual * weights[None, :], axis=1)
                valid_np = valid_mask.detach().cpu().numpy().reshape(-1).astype(bool)
                cost[~valid_np] = np.inf
                if np.any(np.isfinite(cost)):
                    best_idx = int(np.nanargmin(cost))
                    cgi_np = candidate_group_index.detach().cpu().numpy().reshape(-1)
                    teacher_group = int(cgi_np[best_idx])
                    self._last_b1_group_shadow_teacher_group = teacher_group
                    teacher_group_valid = bool(support_ok or close_neighborhood)
                    self._last_b1_group_shadow_teacher_group_valid = teacher_group_valid
                    if teacher_group_valid:
                        self._b1_group_shadow_teacher_group_valid_count += 1
                    disagree = bool(teacher_group_valid and pred_group != teacher_group)
                    self._last_b1_group_shadow_teacher_disagree = disagree
                    if disagree:
                        self._b1_group_shadow_disagreement_count += 1
                    best_cost = float(cost[best_idx])
                    pred_group_mask = (cgi_np == int(pred_group)) & valid_np
                    baseline_group_mask = (cgi_np == int(baseline_group)) & valid_np
                    pred_group_cost = float(np.min(cost[pred_group_mask])) if np.any(pred_group_mask) else np.inf
                    baseline_group_cost = (
                        float(np.min(cost[baseline_group_mask])) if np.any(baseline_group_mask) else np.inf
                    )
                    pred_regret = pred_group_cost - best_cost
                    baseline_regret = baseline_group_cost - best_cost
                    regret_delta = baseline_regret - pred_regret
                    self._last_b1_group_shadow_teacher_best_cost = best_cost
                    self._last_b1_group_shadow_baseline_group_cost = baseline_group_cost
                    self._last_b1_group_shadow_pred_group_cost = pred_group_cost
                    self._last_b1_group_shadow_baseline_group_regret = baseline_regret
                    self._last_b1_group_shadow_pred_group_regret = pred_regret
                    self._last_b1_group_shadow_regret_delta = regret_delta
                    if teacher_group_valid and np.isfinite(regret_delta):
                        self._b1_group_shadow_cost_valid_count += 1
                        self._b1_group_shadow_regret_delta_sum += float(regret_delta)
                        if regret_delta > 1e-6:
                            self._b1_group_shadow_cost_improve_count += 1
                        elif regret_delta < -1e-6:
                            self._b1_group_shadow_cost_worse_count += 1
                        if close_neighborhood:
                            self._b1_group_shadow_close_cost_valid_count += 1
                            self._b1_group_shadow_close_regret_delta_sum += float(regret_delta)
                            if regret_delta > 1e-6:
                                self._b1_group_shadow_close_cost_improve_count += 1
                            elif regret_delta < -1e-6:
                                self._b1_group_shadow_close_cost_worse_count += 1
                        if self._last_b1_apply_gate_apply:
                            self._b1_apply_gate_cost_valid_count += 1
                            self._b1_apply_gate_regret_delta_sum += float(regret_delta)
                            if regret_delta > 1e-6:
                                self._b1_apply_gate_cost_improve_count += 1
                            elif regret_delta < -1e-6:
                                self._b1_apply_gate_cost_worse_count += 1
                            if close_neighborhood:
                                self._b1_apply_gate_close_cost_valid_count += 1
                                self._b1_apply_gate_close_regret_delta_sum += float(regret_delta)
                                if regret_delta > 1e-6:
                                    self._b1_apply_gate_close_cost_improve_count += 1
                                elif regret_delta < -1e-6:
                                    self._b1_apply_gate_close_cost_worse_count += 1

    def _run_b2_candidate_evaluator_shadow(
        self,
        controller,
        *,
        front_rgb,
        wrist_rgb,
        wrist_depth,
        proprio,
        gripper_context,
        current_delta,
        current_dx_sign,
        current_dy_sign,
        current_dyaw_sign,
        basin_distance_bin,
        candidate_actions_src,
        valid_mask,
        candidate_scope_mask,
        baseline_index,
        step_idx: int,
    ):
        import torch
        import torch.nn.functional as F

        self._b2_candidate_shadow_eval_count += 1
        self._last_b2_candidate_shadow_gate_open = False
        self._last_b2_candidate_shadow_close_neighborhood = False
        self._last_b2_candidate_shadow_mode = -1
        self._last_b2_candidate_shadow_mode_confidence = np.nan
        self._last_b2_candidate_shadow_mode_margin = np.nan
        self._last_b2_candidate_shadow_baseline_index = int(baseline_index)
        self._last_b2_candidate_shadow_pred_index = -1
        self._last_b2_candidate_shadow_changed = False
        self._last_b2_candidate_shadow_best_index = -1
        self._last_b2_candidate_shadow_best_cost = np.nan
        self._last_b2_candidate_shadow_baseline_cost = np.nan
        self._last_b2_candidate_shadow_pred_cost = np.nan
        self._last_b2_candidate_shadow_baseline_regret = np.nan
        self._last_b2_candidate_shadow_pred_regret = np.nan
        self._last_b2_candidate_shadow_regret_delta = np.nan
        self._last_b2_candidate_shadow_yaw_needed = False
        self._last_b2_candidate_shadow_yaw_keep = False
        self._last_b2_candidate_shadow_teacher_ready = False
        self._last_b2_candidate_shadow_xy_block = False
        self._last_b2_candidate_shadow_runtime_scope_size = 0
        self._last_b2_candidate_shadow_small_yaw_scope_size = 0
        self._last_b2_candidate_shadow_large_yaw_scope_size = 0
        self._last_b2_candidate_shadow_probe_count = 0
        self._last_b2_candidate_shadow_candidate_actions_local = None
        self._last_b2_candidate_shadow_candidate_scope_mask = None
        self._last_b2_candidate_shadow_candidate_valid_mask = None
        self._last_b2_candidate_shadow_candidate_cost = None
        self._last_b2_candidate_shadow_candidate_oracle_score = None

        shadow = getattr(controller, "_student_candidate_evaluator_shadow", None)
        if not shadow:
            return
        if hasattr(current_delta, "detach"):
            current_delta_np = current_delta.detach().float().cpu().numpy().reshape(-1)
        else:
            current_delta_np = np.asarray(current_delta, dtype=np.float32).reshape(-1)
        if current_delta_np.size < 6:
            return

        handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        handoff_xy = float(handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        handoff_z = float(handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        handoff_yaw = float(handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        if handoff_xy <= 0.0:
            handoff_xy = max(float(getattr(controller, "_ready_band_xy_threshold", self.close_veto_xy_threshold)), 1e-4)
        if handoff_z <= 0.0:
            handoff_z = max(float(getattr(controller, "_ready_band_abs_z_threshold", self.close_veto_abs_z_threshold)), 1e-4)
        if handoff_yaw <= 0.0:
            handoff_yaw = max(float(getattr(controller, "_ready_band_yaw_threshold", -1.0)), 0.12)
        cur_xy = float(np.linalg.norm(current_delta_np[:2]))
        cur_z = float(abs(current_delta_np[2]))
        cur_yaw = float(abs(current_delta_np[5]))

        gripper_open = None
        if gripper_context is not None:
            if hasattr(gripper_context, "detach"):
                gc_np = gripper_context.detach().float().cpu().numpy().reshape(-1)
            else:
                gc_np = np.asarray(gripper_context, dtype=np.float32).reshape(-1)
            if gc_np.size > 0:
                gripper_open = float(gc_np[0])
        support_ok = bool(self._delta_within_band(current_delta_np, controller, band="outer"))
        refine_ok = bool(gripper_open is not None and self._alignment_refine_band_ready(controller, gripper_open))
        close_ready = False
        if gripper_open is not None:
            close_detail = self._runtime_geometry_ready_detail(
                controller,
                gripper_open,
                use_handoff_thresholds=bool(str(getattr(controller, "_runtime_handoff_spec_name", "none")) != "none"),
            )
            close_ready = bool(close_detail.get("runtime_geometry_ready", False))
        close_neighborhood = bool(
            close_ready
            or (
                cur_xy <= max(handoff_xy * 2.5, 0.012)
                and cur_z <= max(handoff_z * 8.0, 0.008)
                and cur_yaw <= max(handoff_yaw * 2.5, 0.18)
            )
        )
        nearish_runtime = bool(
            cur_z <= max(handoff_z * 4.0, 0.012)
            and cur_xy <= max(handoff_xy * 8.0, 0.035)
            and cur_yaw <= max(handoff_yaw * 6.0, 0.45)
        )
        self._last_b2_candidate_shadow_close_neighborhood = close_neighborhood
        self._last_b2_candidate_shadow_nearish_runtime = nearish_runtime
        if close_neighborhood:
            self._b2_candidate_shadow_close_count += 1
        if nearish_runtime:
            self._b2_candidate_shadow_nearish_count += 1
        gate_open = bool(
            self.manager.phase == StagePhase.ALIGN
            and gripper_open is not None
            and gripper_open >= self.alignment_open_threshold
            and (support_ok or refine_ok or close_neighborhood)
        )
        if self.b2_candidate_shadow_gate_mode == "close_only":
            gate_open = bool(gate_open and close_neighborhood)
        elif self.b2_candidate_shadow_gate_mode == "nearish_only":
            gate_open = bool(gate_open and nearish_runtime)
        if not gate_open:
            return

        candidate_model = shadow.get("candidate_model")
        handoff_model = shadow.get("handoff_model")
        if candidate_model is None or handoff_model is None:
            return
        cand_np = np.asarray(candidate_actions_src, dtype=np.float32)
        if cand_np.ndim != 2 or cand_np.shape[1] < 6:
            return
        probe_values = tuple(float(v) for v in getattr(controller, "_b2_candidate_shadow_yaw_probe_values", ()) or ())
        probe_actions = []
        for mag in probe_values:
            m = abs(float(mag))
            if m <= 0.0:
                continue
            pos = np.zeros((6,), dtype=np.float32)
            neg = np.zeros((6,), dtype=np.float32)
            pos[5] = m
            neg[5] = -m
            probe_actions.extend([pos, neg])
        if probe_actions:
            cand_np = np.concatenate([cand_np[:, :6], np.stack(probe_actions, axis=0)], axis=0).astype(np.float32)
            self._last_b2_candidate_shadow_probe_count = int(len(probe_actions))
        else:
            cand_np = cand_np[:, :6]
        device = front_rgb.device
        dtype = front_rgb.dtype
        n = int(cand_np.shape[0])
        valid_raw = valid_mask.detach().cpu().numpy().reshape(-1).astype(bool)
        scope_raw = candidate_scope_mask.detach().cpu().numpy().reshape(-1).astype(bool)
        valid_np = np.zeros((n,), dtype=bool)
        scope_np = np.zeros((n,), dtype=bool)
        valid_np[: min(n, valid_raw.shape[0])] = valid_raw[:n]
        scope_np[: min(n, scope_raw.shape[0])] = scope_raw[:n]
        if probe_actions:
            valid_np[-len(probe_actions):] = True
            scope_np[-len(probe_actions):] = True
        scope_np = scope_np & valid_np
        if not np.any(scope_np):
            scope_np = valid_np
        if not np.any(scope_np):
            return

        self._last_b2_candidate_shadow_gate_open = True
        self._b2_candidate_shadow_gate_pass_count += 1
        self._last_b2_candidate_shadow_runtime_scope_size = int(np.sum(scope_np))
        keep_yaw_abs = float(shadow.get("yaw_keep_abs", 0.035))
        small_scope_np = scope_np & (np.abs(cand_np[:n, 5]) <= keep_yaw_abs)
        self._last_b2_candidate_shadow_small_yaw_scope_size = int(np.sum(small_scope_np))
        self._last_b2_candidate_shadow_large_yaw_scope_size = int(np.sum(scope_np & (np.abs(cand_np[:n, 5]) > keep_yaw_abs)))
        self._last_b2_candidate_shadow_candidate_actions_local = cand_np[:n, :6].astype(np.float32).tolist()
        self._last_b2_candidate_shadow_candidate_scope_mask = scope_np.astype(np.float32).tolist()
        self._last_b2_candidate_shadow_candidate_valid_mask = valid_np.astype(np.float32).tolist()

        substage_id = torch.tensor([int(getattr(self.manager, "substage", 0))], device=device, dtype=torch.long)
        contact_state = torch.tensor([int(getattr(self.manager, "contact_state", 0))], device=device, dtype=torch.long)
        stage_target_mode = torch.tensor([int(getattr(self.manager, "stage_target_mode", 0))], device=device, dtype=torch.long)
        cand_t = torch.from_numpy(cand_np[:n, :6]).unsqueeze(0).to(device=device, dtype=dtype)
        scope_t = torch.from_numpy(scope_np).unsqueeze(0).to(device=device, dtype=torch.bool)
        valid_t = torch.from_numpy(valid_np).unsqueeze(0).to(device=device, dtype=torch.bool)
        with torch.no_grad():
            handoff_out = handoff_model(
                front_rgb=front_rgb.to(device=device, dtype=dtype),
                wrist_rgb=wrist_rgb.to(device=device, dtype=dtype),
                wrist_depth=wrist_depth.to(device=device, dtype=dtype),
                proprio=proprio.to(device=device, dtype=dtype),
                gripper_context=gripper_context.to(device=device, dtype=dtype),
                proxy_current_delta_basin_target=current_delta.to(device=device, dtype=dtype),
                current_dx_sign=current_dx_sign.to(device=device),
                current_dy_sign=current_dy_sign.to(device=device),
                current_dyaw_sign=current_dyaw_sign.to(device=device),
                basin_distance_bin=basin_distance_bin.to(device=device),
                substage_id=substage_id,
                contact_state=contact_state,
                stage_target_mode=stage_target_mode,
            )
            out = candidate_model.forward_with_mode(
                handoff_latent=handoff_out["latent"],
                proxy_current_delta_basin_target=current_delta.to(device=device, dtype=dtype),
                candidate_actions_local=cand_t,
                candidate_mask=valid_t.float(),
                yaw_aware_candidate_scope=scope_t.float(),
            )
            logits = out["yaw_mode_logits"]
            probs = F.softmax(logits, dim=-1)
            mode = int(torch.argmax(logits, dim=-1).item())
            self._last_b2_candidate_shadow_mode = mode
            self._last_b2_candidate_shadow_mode_confidence = float(torch.max(probs, dim=-1).values.item())
            top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
            if top2.shape[-1] > 1:
                self._last_b2_candidate_shadow_mode_margin = float((top2[:, 0] - top2[:, 1]).item())
            else:
                self._last_b2_candidate_shadow_mode_margin = np.inf
            apply_label = int(logits.shape[-1] - 1)
            shadow_policy = str(shadow.get("shadow_policy", "b2_mode_gated"))
            if shadow_policy == "generic_ranker_v4b":
                if mode == apply_label:
                    self._b2_candidate_shadow_mode_apply_count += 1
                else:
                    self._b2_candidate_shadow_mode_keep_count += 1
                scores = out.get("candidate_scores", out.get("candidate_scores_apply", out.get("candidate_scores_keep")))
                pred_idx = int(scores.masked_fill(~scope_t, -1e9).argmax(dim=-1).item())
            elif mode == 0 or mode != apply_label:
                self._b2_candidate_shadow_mode_keep_count += 1
                self._last_b2_candidate_shadow_keep_baseline_forced = True
                self._b2_candidate_shadow_keep_baseline_forced_count += 1
                pred_idx = int(baseline_index)
            else:
                self._b2_candidate_shadow_mode_apply_count += 1
                scores = out.get("candidate_scores_apply", out["candidate_scores"])
                pred_idx = int(scores.masked_fill(~scope_t, -1e9).argmax(dim=-1).item())
        self._last_b2_candidate_shadow_pred_index = pred_idx
        changed = bool(pred_idx != int(baseline_index))
        self._last_b2_candidate_shadow_changed = changed
        if changed:
            self._b2_candidate_shadow_change_count += 1

        teacher_delta = getattr(controller, "_runtime_teacher_current_delta_basin_target", None)
        if teacher_delta is None:
            return
        teacher_delta_np = np.asarray(teacher_delta, dtype=np.float32).reshape(-1)
        if teacher_delta_np.size < 6:
            return
        teacher_xy = float(np.linalg.norm(teacher_delta_np[:2]))
        teacher_z = float(abs(teacher_delta_np[2]))
        teacher_yaw = float(abs(teacher_delta_np[5]))
        yaw_needed = bool(teacher_yaw > handoff_yaw and teacher_xy <= max(handoff_xy * 2.5, 0.030))
        yaw_keep = bool(teacher_yaw <= max(keep_yaw_abs, handoff_yaw))
        teacher_ready = bool(teacher_xy <= handoff_xy and teacher_z <= handoff_z and teacher_yaw <= handoff_yaw)
        xy_block = bool(teacher_xy > handoff_xy and teacher_z <= max(handoff_z * 2.0, 0.010) and teacher_yaw <= max(handoff_yaw * 1.5, 0.18))
        self._last_b2_candidate_shadow_yaw_needed = yaw_needed
        self._last_b2_candidate_shadow_yaw_keep = yaw_keep
        self._last_b2_candidate_shadow_teacher_ready = teacher_ready
        self._last_b2_candidate_shadow_xy_block = xy_block
        if yaw_needed:
            self._b2_candidate_shadow_yaw_needed_count += 1
        if yaw_keep:
            self._b2_candidate_shadow_yaw_keep_count += 1
        if teacher_ready:
            self._b2_candidate_shadow_teacher_ready_count += 1
        if xy_block:
            self._b2_candidate_shadow_xy_block_count += 1

        weights = np.asarray(
            [
                1.0 / max(handoff_xy, 1e-4),
                1.0 / max(handoff_xy, 1e-4),
                1.0 / max(handoff_z, 1e-4),
                0.0,
                0.0,
                1.0 / max(handoff_yaw, 1e-4),
            ],
            dtype=np.float32,
        )
        residual = teacher_delta_np[None, :6] - cand_np[:n, :6]
        cost = np.linalg.norm(residual * weights[None, :], axis=1)
        cost[~scope_np] = np.inf
        self._last_b2_candidate_shadow_candidate_cost = cost.astype(np.float32).tolist()
        self._last_b2_candidate_shadow_candidate_oracle_score = (-cost).astype(np.float32).tolist()
        if not np.any(np.isfinite(cost)):
            return
        best_idx = int(np.nanargmin(cost))
        baseline_idx = int(baseline_index)
        self._last_b2_candidate_shadow_best_index = best_idx
        self._last_b2_candidate_shadow_best_cost = float(cost[best_idx])
        self._last_b2_candidate_shadow_pred_cost = float(cost[pred_idx]) if 0 <= pred_idx < n else np.inf
        self._last_b2_candidate_shadow_baseline_cost = float(cost[baseline_idx]) if 0 <= baseline_idx < n else np.inf
        pred_regret = self._last_b2_candidate_shadow_pred_cost - self._last_b2_candidate_shadow_best_cost
        baseline_regret = self._last_b2_candidate_shadow_baseline_cost - self._last_b2_candidate_shadow_best_cost
        regret_delta = baseline_regret - pred_regret
        self._last_b2_candidate_shadow_pred_regret = float(pred_regret)
        self._last_b2_candidate_shadow_baseline_regret = float(baseline_regret)
        self._last_b2_candidate_shadow_regret_delta = float(regret_delta)
        if np.isfinite(regret_delta):
            self._b2_candidate_shadow_cost_valid_count += 1
            self._b2_candidate_shadow_regret_delta_sum += float(regret_delta)
            if regret_delta > 1e-6:
                self._b2_candidate_shadow_cost_improve_count += 1
            elif regret_delta < -1e-6:
                self._b2_candidate_shadow_cost_worse_count += 1

    def _maybe_apply_b2_candidate_bounded_v0(self, pred_idx: int, runtime_candidate_count: int) -> tuple[int, bool]:
        self._b2_candidate_bounded_eval_count += 1
        self._last_b2_candidate_bounded_gate_open = False
        self._last_b2_candidate_bounded_applied = False
        self._last_b2_candidate_bounded_changed = False
        self._last_b2_candidate_bounded_prev_index = int(pred_idx)
        self._last_b2_candidate_bounded_new_index = int(pred_idx)
        self._last_b2_candidate_bounded_mode_confidence = float(self._last_b2_candidate_shadow_mode_confidence)
        self._last_b2_candidate_bounded_mode_margin = float(self._last_b2_candidate_shadow_mode_margin)
        if not self.enable_b2_candidate_bounded_v0:
            return int(pred_idx), False
        if not bool(self._last_b2_candidate_shadow_gate_open):
            return int(pred_idx), False
        self._last_b2_candidate_bounded_gate_open = True
        self._b2_candidate_bounded_gate_pass_count += 1
        if int(self._last_b2_candidate_shadow_mode) != 2:
            return int(pred_idx), False
        conf = float(self._last_b2_candidate_shadow_mode_confidence)
        margin = float(self._last_b2_candidate_shadow_mode_margin)
        if not np.isfinite(conf) or conf < self.b2_candidate_apply_conf_threshold:
            return int(pred_idx), False
        if not np.isfinite(margin) or margin < self.b2_candidate_apply_margin_threshold:
            return int(pred_idx), False
        bounded_idx = int(self._last_b2_candidate_shadow_pred_index)
        if bounded_idx < 0 or bounded_idx >= int(runtime_candidate_count):
            return int(pred_idx), False
        self._last_b2_candidate_bounded_applied = True
        self._b2_candidate_bounded_apply_count += 1
        self._last_b2_candidate_bounded_new_index = bounded_idx
        if bounded_idx != int(pred_idx):
            self._last_b2_candidate_bounded_changed = True
            self._b2_candidate_bounded_change_count += 1
        return bounded_idx, True

    def _run_controller(self, controller, wrist_depth, ft_hist, proprio, base_action_local, step_idx, gripper_context=None):
        import torch

        device = next(controller.parameters()).device
        dtype = next(controller.parameters()).dtype
        wd = self._to_tensor(wrist_depth, (1, 1, 96, 96), device, dtype)
        fh = self._to_tensor(ft_hist, (1, 32, 6), device, dtype)
        pr = self._to_tensor(proprio, (1, 15), device, dtype)
        ba = self._to_tensor(base_action_local, (1, 6), device, dtype)
        gc = self._to_tensor(gripper_context, (1, 3), device, dtype)
        si = torch.tensor([step_idx], device=device, dtype=torch.long)
        phase_id = torch.tensor([int(self.manager.phase)], device=device, dtype=torch.long)
        phase_age = torch.tensor([float(self.manager.phase_age)], device=device, dtype=dtype)
        since_replan = torch.tensor([float(self.manager.steps_since_last_replan)], device=device, dtype=dtype)
        with torch.no_grad():
            outputs = controller(
                wd,
                fh,
                pr,
                ba,
                si,
                phase_id=phase_id,
                phase_age=phase_age,
                steps_since_last_replan=since_replan,
                gripper_context=gc,
                return_aux=True,
            )
        return outputs

    def _select_layered_multi_head_scores(
        self,
        controller,
        multi_head_scores,
    ):
        import torch

        mode = str(getattr(controller, "_selection_mode", "layered_multi")).lower()
        weights = getattr(controller, "_selection_weights", {}) or {}
        w_safe = float(weights.get("safe", 1.0))
        w_pareto = float(weights.get("pareto", 1.0))
        w_yaw = float(weights.get("yaw", 1.0))
        w_geom = float(weights.get("geom", 0.5))
        w_risk = float(weights.get("risk", 0.5))
        if mode == "scalar":
            return multi_head_scores[..., 0]
        if mode == "best_safe":
            return multi_head_scores[..., 0]
        if mode == "pareto":
            return multi_head_scores[..., 1]
        if mode == "yaw_match":
            return multi_head_scores[..., 2]
        if mode == "risk_safe":
            return multi_head_scores[..., 3]
        if mode == "geometry_gain":
            return multi_head_scores[..., 4]
        if mode == "weighted_multi":
            return (
                w_safe * multi_head_scores[..., 0]
                + w_pareto * multi_head_scores[..., 1]
                + w_yaw * multi_head_scores[..., 2]
                + w_geom * multi_head_scores[..., 4]
                - w_risk * multi_head_scores[..., 3]
            )
        if mode in {"layered_multi", "pareto_then_best_safe", "pareto_then_geometry"}:
            pareto_prob = torch.sigmoid(multi_head_scores[..., 1])
            safe_prob = torch.sigmoid(multi_head_scores[..., 0])
            yaw_prob = torch.sigmoid(multi_head_scores[..., 2])
            risk_prob = torch.sigmoid(multi_head_scores[..., 3])
            geom_prob = torch.sigmoid(multi_head_scores[..., 4])
            if mode == "pareto_then_best_safe":
                base = safe_prob + 0.5 * geom_prob + 0.25 * yaw_prob - 0.5 * risk_prob
            elif mode == "pareto_then_geometry":
                base = geom_prob + 0.5 * safe_prob + 0.25 * yaw_prob - 0.5 * risk_prob
            else:
                base = safe_prob + pareto_prob + 0.5 * yaw_prob + 0.75 * geom_prob - 0.5 * risk_prob
            gate = pareto_prob >= 0.5
            any_gate = torch.any(gate, dim=-1, keepdim=True)
            fallback_k = min(3, base.shape[-1])
            topk_idx = torch.topk(pareto_prob, k=fallback_k, dim=-1).indices
            fallback_mask = torch.zeros_like(gate, dtype=torch.bool).scatter(-1, topk_idx, True)
            mask = torch.where(any_gate, gate, fallback_mask)
            return torch.where(mask, base, torch.full_like(base, -1e9))
        raise ValueError(f"unknown selection_mode={mode!r}")

    def _run_depth_force_local_proposal(
        self,
        controller,
        front_rgb,
        wrist_rgb,
        wrist_depth,
        ft_hist,
        proprio,
        base_action_local,
        step_idx,
        gripper_context=None,
        current_delta=None,
        current_quat=None,
        gripper_pose=None,
        exec_base_action_world=None,
    ):
        import torch

        device = next(controller.parameters()).device
        dtype = next(controller.parameters()).dtype
        fr = self._to_rgb_tensor(front_rgb, (1, 3, 128, 128), device, dtype)
        wr = self._to_rgb_tensor(wrist_rgb, (1, 3, 128, 128), device, dtype)
        wd = self._to_tensor(wrist_depth, (1, 1, 96, 96), device, dtype)
        fh = self._to_tensor(ft_hist, (1, 32, 6), device, dtype)
        pr = self._to_tensor(proprio, (1, 15), device, dtype)
        ba = self._to_tensor(base_action_local, (1, 6), device, dtype)
        gc = self._to_tensor(gripper_context, (1, 3), device, dtype)
        si = torch.tensor([step_idx], device=device, dtype=torch.long)
        phase_id = torch.tensor([int(self.manager.phase)], device=device, dtype=torch.long)
        phase_age = torch.tensor([float(self.manager.phase_age)], device=device, dtype=dtype)
        since_replan = torch.tensor([float(self.manager.steps_since_last_replan)], device=device, dtype=dtype)
        current_delta_np = np.asarray(
            current_delta if current_delta is not None else getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        cur_delta_t = torch.from_numpy(current_delta_np[:6]).unsqueeze(0).to(device=device, dtype=dtype)
        contact_phase = torch.tensor([int(self.manager.contact_state)], device=device, dtype=torch.long)
        depth_proximity = torch.tensor(
            [float(self.compute_depth_proximity(wrist_depth) or np.linalg.norm(current_delta_np[:3]))],
            device=device,
            dtype=dtype,
        )
        gripper_state = torch.tensor(
            [float(gc[0, 0].item()) if gc is not None and gc.numel() >= 1 else 0.0],
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            outputs = controller(
                fr,
                wr,
                wd,
                fh,
                pr,
                ba,
                proposal_actions_local=None,
                stage_token=phase_id,
                contact_phase=contact_phase,
                depth_proximity=depth_proximity,
                gripper_state=gripper_state,
            )
        proposals = outputs["proposal_actions_local"]
        multi = outputs["multi_head_scores"]
        scores = self._select_layered_multi_head_scores(controller, multi)
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
        selected_idx = torch.argmax(scores.masked_fill(~valid_mask, -1e9), dim=-1)
        selected_local = proposals.gather(1, selected_idx[:, None, None].expand(-1, 1, proposals.shape[-1])).squeeze(1)
        selected_score = scores.gather(1, selected_idx[:, None]).squeeze(1)

        # --- v2 shadow scoring ---
        v2_shadow = getattr(self, "_v2_shadow", None)
        v2_selected_idx = None
        v2_selected_delta = None
        v2_scores = None
        v2_gate_pass = False
        v2_bucket = "unknown"
        v2_post_delta = None
        v2_geom_imp = None
        v2_topk = None
        _v2_delta_source = "none"
        _v2_delta_norm = 0.0
        _v2_gripper_pose_present = False
        _v2_motion_target_pose_present = False
        _v2_motion_target_delta_present = False
        _v2_current_delta_basin_present = False
        if v2_shadow is not None and current_delta_np is not None and current_delta_np.size >= 6:
            # --- Step 1: compute v2_delta (priority: predictor > motion pose > basin > hardcoded) ---
            _v2_gripper_pose_present = gripper_pose is not None
            _v2_current_delta_basin_present = (
                current_delta_np is not None and current_delta_np.size >= 6
                and np.any(np.abs(current_delta_np[:6]) > 1e-10)
            )
            _pose_source = "none"
            v2_delta = None

            # Priority 1: target_delta_predictor (signed 6D, closest to v2 training distribution)
            _td_predictor = getattr(self, "_v2_target_delta_predictor", None)
            if _td_predictor is not None:
                _td_out = _td_predictor.predict(
                    front_rgb, wrist_rgb, wrist_depth, proprio,
                    contact_state=int(self.manager.contact_state),
                    substage_id=int(self.manager.substage),
                    target_mode=int(self.manager.stage_target_mode),
                    gripper_context=gc.squeeze(0).float().cpu().numpy() if gc is not None else None,
                )
                if _td_out is not None and _td_out.size >= 6:
                    v2_delta = _td_out[:6].astype(np.float32)
                    _pose_source = "target_delta_predictor"

            # Priority 2-4: motion target pose / canonical basin / hardcoded
            if v2_delta is None:
                mt_delta = getattr(controller, "_runtime_motion_target_delta_local", None)
                _v2_motion_target_delta_present = (
                    mt_delta is not None and np.asarray(mt_delta).size >= 6
                    and np.any(np.abs(np.asarray(mt_delta, dtype=np.float32).reshape(-1)[:6]) > 1e-10)
                )
                mt_pose = getattr(controller, "_runtime_motion_target_pose_7d", None)
                _v2_motion_target_pose_present = mt_pose is not None
                _pose_source = "none"
                if mt_pose is not None:
                    _pose_source = "runtime_motion_target_pose"
                else:
                    mt_pose = getattr(controller, "_canonical_basin_center_pose_7d", None)
                    if mt_pose is not None:
                        _pose_source = "canonical_basin_center_pose"
                if mt_pose is None:
                    mt_pose = np.array([0.065578, 0.040987, 0.769993,
                                        0.004968, 0.001088, 0.818756, -0.574120], dtype=np.float32)
                    _pose_source = "hardcoded_basin_center"

                if gripper_pose is not None:
                    mt_pose = np.asarray(mt_pose, dtype=np.float32).reshape(7)
                    gp = np.asarray(gripper_pose, dtype=np.float32).reshape(7)
                    v2_delta = self._pose_delta_local_between(gp, mt_pose)[:6].astype(np.float32)
                    _pose_source = _pose_source
                else:
                    v2_delta = current_delta_np[:6].copy()
                    _pose_source = "current_delta_np" if _v2_current_delta_basin_present else "fallback_zero"

            _v2_delta_source = _pose_source
            _v2_delta_norm = float(np.linalg.norm(v2_delta[:3]))

            # --- Step 2: compute ALL v2 metrics from v2_delta (single source) ---
            v2_cur_xy = float(np.linalg.norm(v2_delta[:2]))
            v2_cur_z = float(abs(v2_delta[2]))
            v2_cur_yaw = float(abs(v2_delta[5]))

            # --- Step 3: near-zone gate from v2_delta ---
            if v2_cur_xy < 0.06 and v2_cur_z < 0.12 and v2_cur_yaw < 0.30:
                v2_gate_pass = True
                if v2_cur_xy < 0.020 and v2_cur_z < 0.05 and v2_cur_yaw < 0.16:
                    v2_bucket = "micro_contact_refine"
                else:
                    v2_bucket = "near_alignment"
            elif v2_cur_xy < 0.16 and v2_cur_z < 0.30:
                v2_bucket = "mid_approach_assist"
            else:
                v2_bucket = "far_coarse_approach"

            # --- Step 4: record handoff mismatch for diagnostics ---
            hm = dict(getattr(controller, "_runtime_handoff_metrics", {}) or {})
            _hm_xy = hm.get("xy_error")
            _hm_z = hm.get("abs_z_error")
            _hm_yaw = hm.get("yaw_error")
            _v2_hm_xy_ratio = (
                float(_hm_xy) / max(v2_cur_xy, 1e-8)
                if _hm_xy is not None and np.isfinite(float(_hm_xy)) else -1.0
            )
            _v2_hm_z_ratio = (
                float(_hm_z) / max(v2_cur_z, 1e-8)
                if _hm_z is not None and np.isfinite(float(_hm_z)) else -1.0
            )

            if v2_gate_pass:
                prop_np = proposals.squeeze(0).detach().float().cpu().numpy()  # (K, 6)
                post_delta_np = v2_delta[None, :] - prop_np  # (K, 6)
                post_xy = np.linalg.norm(post_delta_np[:, :2], axis=1)
                post_z = np.abs(post_delta_np[:, 2])
                post_yaw = np.abs(post_delta_np[:, 5])
                improve_xy = v2_cur_xy - post_xy
                improve_z = v2_cur_z - post_z
                improve_yaw = v2_cur_yaw - post_yaw
                geom_imp = improve_xy + improve_z

                td_t = torch.from_numpy(v2_delta).unsqueeze(0).to(device=device, dtype=dtype)
                pa_t = proposals.to(device=device, dtype=dtype)
                pd_t = torch.from_numpy(post_delta_np).unsqueeze(0).to(device=device, dtype=dtype)
                xi_t = torch.from_numpy(improve_xy.astype(np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
                zi_t = torch.from_numpy(improve_z.astype(np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
                yi_t = torch.from_numpy(improve_yaw.astype(np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
                gi_t = torch.from_numpy(geom_imp.astype(np.float32)).unsqueeze(0).to(device=device, dtype=dtype)

                with torch.no_grad():
                    v2_out = v2_shadow(wd, fh, pr, ba, td_t, pa_t, pd_t, xi_t, zi_t, yi_t, gi_t)
                v2_scores = v2_out["candidate_scores"].squeeze(0)
                v2_selected_idx = int(v2_scores.argmax().item())
                v2_selected_delta = prop_np[v2_selected_idx].copy()
                v2_post_delta = post_delta_np
                v2_geom_imp = geom_imp
                _, v2_topk_idx = torch.topk(v2_scores, min(3, v2_scores.shape[0]))
                v2_topk = v2_topk_idx.cpu().tolist()

        return {
            "delta_pose": selected_local,
            "delta_pose_gated": selected_local,
            "alpha": torch.ones((selected_local.shape[0],), device=device, dtype=dtype),
            "ready_to_close": torch.sigmoid(multi[..., 0].gather(1, selected_idx[:, None]).squeeze(1)),
            "candidate_scores": scores,
            "group_logits": scores,
            "pred_candidate_index": selected_idx,
            "pred_candidate_group_index": selected_idx,
            "candidate_step_scale": torch.ones_like(scores),
            "selected_step_scale": torch.ones((selected_local.shape[0],), device=device, dtype=dtype),
            "selected_score": selected_score,
            "proposal_actions_local": proposals,
            "multi_head_scores": multi,
            "multi_head_score_dict": outputs.get("multi_head_score_dict", {}),
            "state_latent": outputs.get("state_latent"),
            # v2 shadow fields
            "v2_shadow_active": v2_shadow is not None,
            "v2_selected_candidate_index": v2_selected_idx,
            "v2_selected_delta": v2_selected_delta,
            "v2_candidate_scores": v2_scores.detach().float().cpu().numpy().tolist() if v2_scores is not None else None,
            "v2_gate_pass": v2_gate_pass,
            "v2_stage_bucket": v2_bucket,
            "v2_post_candidate_delta": v2_post_delta.tolist() if v2_post_delta is not None else None,
            "v2_geometry_improvement": v2_geom_imp.tolist() if v2_geom_imp is not None else None,
            "v2_topk_indices": v2_topk,
            # source trace
            "v2_delta_source": _v2_delta_source,
            "v2_delta_norm": _v2_delta_norm,
            "v2_gripper_pose_present": _v2_gripper_pose_present,
            "v2_motion_target_pose_present": _v2_motion_target_pose_present,
            "v2_motion_target_delta_present": _v2_motion_target_delta_present,
            "v2_current_delta_basin_present": _v2_current_delta_basin_present,
            # consistent error metrics from v2_delta
            "v2_cur_xy": v2_cur_xy if v2_shadow is not None else None,
            "v2_cur_z": v2_cur_z if v2_shadow is not None else None,
            "v2_cur_yaw": v2_cur_yaw if v2_shadow is not None else None,
            "v2_hm_xy_ratio": _v2_hm_xy_ratio if v2_shadow is not None else None,
            "v2_hm_z_ratio": _v2_hm_z_ratio if v2_shadow is not None else None,
        }

    def _run_pose_field_scorer(
        self,
        controller,
        front_rgb,
        wrist_rgb,
        wrist_depth,
        ft_hist,
        proprio,
        base_action_local,
        step_idx,
        gripper_context=None,
        current_quat=None,
        gripper_pose=None,
        exec_base_action_world=None,
    ):
        import torch

        if getattr(controller, "_runtime_policy_type", "") == "depth_force_local_proposal_policy":
            return self._run_depth_force_local_proposal(
                controller,
                front_rgb,
                wrist_rgb,
                wrist_depth,
                ft_hist,
                proprio,
                base_action_local,
                step_idx,
                gripper_context=gripper_context,
                current_delta=getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                current_quat=current_quat,
                gripper_pose=gripper_pose,
                exec_base_action_world=exec_base_action_world,
            )

        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        )
        active_controller, specialist_active = self._select_pose_field_scorer_model(controller, current_delta)
        device = next(active_controller.parameters()).device
        dtype = next(active_controller.parameters()).dtype
        fr = self._to_rgb_tensor(front_rgb, (1, 3, 128, 128), device, dtype)
        wr = self._to_rgb_tensor(wrist_rgb, (1, 3, 128, 128), device, dtype)
        wd = self._to_tensor(wrist_depth, (1, 1, 96, 96), device, dtype)
        pr = self._to_tensor(proprio, (1, 15), device, dtype)
        ba = self._to_tensor(base_action_local, (1, 6), device, dtype)
        gc = self._to_tensor(gripper_context, (1, 3), device, dtype)
        si = torch.tensor([step_idx], device=device, dtype=torch.long)
        phase_id = torch.tensor([int(self.manager.phase)], device=device, dtype=torch.long)
        phase_age = torch.tensor([float(self.manager.phase_age)], device=device, dtype=dtype)
        since_replan = torch.tensor([float(self.manager.steps_since_last_replan)], device=device, dtype=dtype)
        current_delta_for_model = self._clip_target_context_for_scorer(controller, current_delta)
        if bool(getattr(active_controller, "use_target_context", True)):
            dx_sign = int(np.sign(current_delta_for_model[0])) if abs(float(current_delta_for_model[0])) > 1e-4 else 0
            dy_sign = int(np.sign(current_delta_for_model[1])) if abs(float(current_delta_for_model[1])) > 1e-4 else 0
            dyaw_sign = int(np.sign(current_delta_for_model[5])) if abs(float(current_delta_for_model[5])) > 1e-3 else 0
            basin_distance = float(getattr(controller, "_runtime_current_basin_distance", 3.0))
            if basin_distance <= 0.9:
                basin_bin = 0
            elif basin_distance <= 1.05:
                basin_bin = 1
            elif basin_distance <= 1.2:
                basin_bin = 2
            else:
                basin_bin = 3

            cur_delta_t = torch.from_numpy(current_delta_for_model).unsqueeze(0).to(device=device, dtype=dtype)
            dxs = torch.tensor([dx_sign], device=device, dtype=torch.long)
            dys = torch.tensor([dy_sign], device=device, dtype=torch.long)
            dyaws = torch.tensor([dyaw_sign], device=device, dtype=torch.long)
            basin_bin_t = torch.tensor([basin_bin], device=device, dtype=torch.long)
        else:
            cur_delta_t = None
            dxs = None
            dys = None
            dyaws = None
            basin_bin_t = None
        candidate_actions_src = getattr(active_controller, "_teacher_candidate_actions_local", None)
        candidate_group_index_src = getattr(active_controller, "_teacher_candidate_group_index", None)
        if candidate_actions_src is None:
            candidate_actions_src = active_controller._candidate_actions_local.detach().cpu().numpy()
        if candidate_group_index_src is None:
            candidate_group_index_src = active_controller._candidate_group_index.detach().cpu().numpy()
        candidate_actions = torch.from_numpy(np.asarray(candidate_actions_src, dtype=np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
        candidate_group_index = torch.from_numpy(np.asarray(candidate_group_index_src, dtype=np.int64)).unsqueeze(0).to(device=device, dtype=torch.long)
        base_candidate_mask = getattr(controller, "_runtime_candidate_mask", None)
        runtime_mask_np = self._make_runtime_candidate_mask(
            controller,
            np.asarray(candidate_actions_src, dtype=np.float32),
            base_candidate_mask,
            current_quat if current_quat is not None else self._extract_current_quat(proprio),
            gripper_pose,
            exec_base_action_world,
        )
        final_handoff_polish_scale_cap = None
        final_handoff_polish_mode = "none"
        runtime_mask_np, final_handoff_polish_scale_cap, final_handoff_polish_mode = self._apply_final_handoff_polish_mask(
            controller,
            np.asarray(candidate_actions_src, dtype=np.float32),
            runtime_mask_np,
            current_delta,
        )
        candidate_mask = torch.from_numpy(runtime_mask_np).unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            outputs = active_controller(
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
        scores = outputs["candidate_scores"]
        step_scale = outputs.get("candidate_step_scale", torch.ones_like(scores))
        group_logits = outputs["group_logits"]
        valid_mask = candidate_mask > 0.5
        baseline_group_logits_before_residual = group_logits.clone()
        baseline_scores_before_residual = scores.clone()
        group_residual_logits = None
        residual_scores = None
        adapter = getattr(controller, "_near_ready_residual_adapter", None)
        adapter_type = str(getattr(adapter, "_controller_type", ""))
        if adapter is not None and adapter_type == "near_ready_group_residual_adapter":
            group_logits, group_residual_logits = self._apply_near_ready_group_residual_adapter(
                controller=controller,
                group_logits=group_logits,
                candidate_group_index=candidate_group_index,
                candidate_mask=candidate_mask,
                gripper_context=gc,
                phase_age=phase_age,
                steps_since_last_replan=since_replan,
                current_delta=cur_delta_t,
                current_dx_sign=dxs,
                current_dy_sign=dys,
                current_dyaw_sign=dyaws,
                basin_distance_bin=basin_bin_t,
                step_idx=step_idx,
            )
        else:
            scores, residual_scores = self._apply_near_ready_residual_adapter(
                controller=controller,
                scores=scores,
                candidate_actions=candidate_actions,
                candidate_mask=candidate_mask,
                gripper_context=gc,
                phase_age=phase_age,
                steps_since_last_replan=since_replan,
                current_delta=cur_delta_t,
                current_dx_sign=dxs,
                current_dy_sign=dys,
                current_dyaw_sign=dyaws,
                basin_distance_bin=basin_bin_t,
                step_idx=step_idx,
            )
        yaw_priority_active = False
        xy_priority_active = False
        yaw_priority_scale_cap = None
        xy_priority_scale_cap = None
        handoff_axis_priority = "xy_micro" if final_handoff_polish_mode == "xy_micro" else "none"
        motion_handoff_thresholds = self._motion_handoff_metric_thresholds(controller)
        yaw_threshold = float(motion_handoff_thresholds.get("yaw_error", self.close_veto_yaw_threshold))
        handoff_xy = float(motion_handoff_thresholds.get("xy_error", self.close_veto_xy_threshold))
        handoff_z = float(motion_handoff_thresholds.get("abs_z_error", self.close_veto_abs_z_threshold))
        cur_xy = float(np.linalg.norm(current_delta[:2])) if current_delta.size >= 2 else np.inf
        cur_z = float(abs(current_delta[2])) if current_delta.size >= 3 else np.inf
        cur_yaw = float(current_delta[5]) if current_delta.size >= 6 else 0.0
        if (
            self.enable_handoff_yaw_priority
            and np.isfinite(yaw_threshold)
            and yaw_threshold > 0.0
            and abs(cur_yaw) > yaw_threshold
            and cur_xy <= max(handoff_xy * 2.5, 0.010)
            and cur_z <= max(handoff_z * 8.0, 0.008)
        ):
            cand_np = np.asarray(candidate_actions_src, dtype=np.float32)
            pure_yaw_np = (
                (np.linalg.norm(cand_np[:, :3], axis=1) <= 1e-8)
                & (np.linalg.norm(cand_np[:, 3:5], axis=1) <= 1e-8)
                & (np.abs(cand_np[:, 5]) > 1e-8)
            )
            # current_delta is target-in-current local error. At execution time a
            # positive yaw primitive increases the observed yaw error when
            # current_delta[5] is positive, so the corrective primitive must have
            # the opposite sign. This is intentionally scoped to the near-handoff
            # yaw-priority path where trace evidence showed same-sign yaw actions
            # monotonically pushed the gripper away from the demo grasp yaw.
            corrective_sign_np = np.sign(cand_np[:, 5]) == -np.sign(cur_yaw)
            yaw_mask_np = pure_yaw_np & corrective_sign_np & (runtime_mask_np > 0.5)
            if not np.any(yaw_mask_np):
                yaw_mask_np = pure_yaw_np & (runtime_mask_np > 0.5)
            if np.any(yaw_mask_np):
                yaw_mask = torch.from_numpy(yaw_mask_np).unsqueeze(0).to(device=device, dtype=torch.bool)
                valid_mask = valid_mask & yaw_mask
                yaw_priority_active = True
                handoff_axis_priority = "yaw"
                yaw_step = float(np.max(np.abs(cand_np[yaw_mask_np, 5])))
                if yaw_step > 1e-8:
                    # Keep yaw correction local; do not let the step head spin
                    # beyond the remaining angular error.
                    yaw_priority_scale_cap = max(0.05, min(1.0, abs(cur_yaw) / yaw_step))
                self._handoff_yaw_priority_count += 1
        if (
            not yaw_priority_active
            and self.enable_handoff_xy_priority
            and np.isfinite(handoff_xy)
            and handoff_xy > 0.0
            and cur_xy > handoff_xy
            and cur_xy <= max(handoff_xy * self.handoff_xy_priority_gate_multiplier, 0.020)
            and cur_z <= max(handoff_z * self.handoff_xy_priority_z_multiplier, 0.006)
            and (
                yaw_threshold < 0.0
                or abs(cur_yaw) <= max(yaw_threshold * self.handoff_xy_priority_yaw_multiplier, 0.16)
            )
        ):
            cand_np = np.asarray(candidate_actions_src, dtype=np.float32)
            pure_xy_np = (
                (np.linalg.norm(cand_np[:, :2], axis=1) > 1e-8)
                & (np.abs(cand_np[:, 2]) <= 1e-8)
                & (np.linalg.norm(cand_np[:, 3:5], axis=1) <= 1e-8)
                & (np.abs(cand_np[:, 5]) <= 1e-8)
            )
            corrective_xy_np = np.einsum("nd,d->n", cand_np[:, :2], current_delta[:2]) > 1e-8
            xy_mask_np = pure_xy_np & corrective_xy_np & (runtime_mask_np > 0.5)
            if not np.any(xy_mask_np):
                xy_mask_np = pure_xy_np & (runtime_mask_np > 0.5)
            if np.any(xy_mask_np):
                xy_mask = torch.from_numpy(xy_mask_np).unsqueeze(0).to(device=device, dtype=torch.bool)
                valid_mask = valid_mask & xy_mask
                xy_priority_active = True
                handoff_axis_priority = "xy"
                xy_step = float(np.max(np.linalg.norm(cand_np[xy_mask_np, :2], axis=1)))
                if xy_step > 1e-8:
                    desired_xy_step = min(float(cur_xy), float(self.handoff_xy_priority_max_xy_step))
                    xy_priority_scale_cap = max(0.05, min(1.0, desired_xy_step / xy_step))
                self._handoff_xy_priority_count += 1
        if str(getattr(active_controller, "_selection_policy", "two_stage_group")) == "global_candidate":
            group_mask = valid_mask
            masked_scores = scores.masked_fill(~group_mask, -1e9)
            pred_idx = masked_scores.argmax(dim=-1)
            pred_group = candidate_group_index.gather(1, pred_idx[:, None]).squeeze(1)
        else:
            group_valid = []
            for group_id in range(group_logits.shape[1]):
                group_valid.append(torch.any(candidate_group_index.eq(group_id) & valid_mask, dim=1))
            group_valid = torch.stack(group_valid, dim=1)
            group_logits_masked = group_logits.masked_fill(~group_valid, -1e9)
            pred_group = group_logits_masked.argmax(dim=-1)
            baseline_pred_group = int(pred_group.item())
            self._run_b1_group_selector_shadow(
                controller,
                front_rgb=fr,
                wrist_rgb=wr,
                wrist_depth=wd,
                proprio=pr,
                gripper_context=gc,
                phase_age=phase_age,
                steps_since_last_replan=since_replan,
                current_delta=cur_delta_t,
                current_dx_sign=dxs,
                current_dy_sign=dys,
                current_dyaw_sign=dyaws,
                basin_distance_bin=basin_bin_t,
                candidate_actions=candidate_actions,
                candidate_actions_src=np.asarray(candidate_actions_src, dtype=np.float32),
                candidate_group_index=candidate_group_index,
                group_valid=group_valid,
                baseline_group=baseline_pred_group,
                valid_mask=valid_mask,
                step_idx=step_idx,
            )
            self._last_b1_group_bounded_applied = False
            self._last_b1_group_bounded_prev_group = baseline_pred_group
            self._last_b1_group_bounded_new_group = baseline_pred_group
            if bool(getattr(controller, "_student_group_selector_bounded", False)) and bool(self._last_b1_apply_gate_apply):
                bounded_group = int(self._last_b1_group_shadow_pred_group)
                if bounded_group >= 0:
                    pred_group = torch.tensor([bounded_group], device=device, dtype=torch.long)
                    self._last_b1_group_bounded_applied = True
                    self._last_b1_group_bounded_new_group = bounded_group
                    self._b1_group_bounded_apply_count += 1
                    if bounded_group != baseline_pred_group:
                        self._b1_group_bounded_change_count += 1
            group_mask = candidate_group_index.eq(pred_group.unsqueeze(1)) & valid_mask
            masked_scores = scores.masked_fill(~group_mask, -1e9)
            if not bool(torch.any(group_mask).item()):
                self._alignment_all_masked_fallback_count += 1
                group_mask = valid_mask
                masked_scores = scores.masked_fill(~group_mask, -1e9)
            pred_idx = masked_scores.argmax(dim=-1)
            self._run_b2_candidate_evaluator_shadow(
                controller,
                front_rgb=fr,
                wrist_rgb=wr,
                wrist_depth=wd,
                proprio=pr,
                gripper_context=gc,
                current_delta=cur_delta_t,
                current_dx_sign=dxs,
                current_dy_sign=dys,
                current_dyaw_sign=dyaws,
                basin_distance_bin=basin_bin_t,
                candidate_actions_src=np.asarray(candidate_actions_src, dtype=np.float32),
                valid_mask=valid_mask,
                candidate_scope_mask=valid_mask,
                baseline_index=int(pred_idx.item()),
                step_idx=step_idx,
            )
            b2_bounded_applied = False
            bounded_pred_idx, b2_bounded_applied = self._maybe_apply_b2_candidate_bounded_v0(
                int(pred_idx.item()),
                int(candidate_actions.shape[1]),
            )
            if b2_bounded_applied:
                pred_idx = torch.tensor([bounded_pred_idx], device=device, dtype=torch.long)
                group_mask = valid_mask
                masked_scores = scores.masked_fill(~group_mask, -1e9)
        if group_residual_logits is not None:
            group_valid = []
            for group_id in range(group_logits.shape[1]):
                group_valid.append(torch.any(candidate_group_index.eq(group_id) & valid_mask, dim=1))
            group_valid = torch.stack(group_valid, dim=1)
            base_group_masked = baseline_group_logits_before_residual.masked_fill(~group_valid, -1e9)
            new_group_masked = group_logits.masked_fill(~group_valid, -1e9)
            self._last_near_ready_residual_prev_index = int(base_group_masked.argmax(dim=-1).item())
            self._last_near_ready_residual_new_index = int(new_group_masked.argmax(dim=-1).item())
            if self._last_near_ready_residual_prev_index != self._last_near_ready_residual_new_index:
                self._near_ready_residual_change_count += 1
                self._last_near_ready_residual_changed = True
        elif residual_scores is not None:
            if str(getattr(active_controller, "_selection_policy", "two_stage_group")) == "global_candidate":
                base_masked_scores = baseline_scores_before_residual.masked_fill(~valid_mask, -1e9)
                base_pred_idx = base_masked_scores.argmax(dim=-1)
            else:
                base_group_mask = candidate_group_index.eq(pred_group.unsqueeze(1)) & valid_mask
                base_masked_scores = baseline_scores_before_residual.masked_fill(~base_group_mask, -1e9)
                base_pred_idx = base_masked_scores.argmax(dim=-1)
            self._last_near_ready_residual_prev_index = int(base_pred_idx.item())
            self._last_near_ready_residual_new_index = int(pred_idx.item())
            if int(base_pred_idx.item()) != int(pred_idx.item()):
                self._near_ready_residual_change_count += 1
                self._last_near_ready_residual_changed = True
        if not ('b2_bounded_applied' in locals() and b2_bounded_applied):
            pred_idx = self._maybe_rerank_near_ready_topk(
                controller,
                fr=fr,
                wr=wr,
                wd=wd,
                pr=pr,
                gc=gc,
                current_delta=current_delta,
                candidate_actions_src=np.asarray(candidate_actions_src, dtype=np.float32),
                masked_scores=masked_scores,
                pred_idx=pred_idx,
                step_scale=step_scale,
                valid_mask=group_mask if 'group_mask' in locals() else valid_mask,
            )
        if candidate_actions.shape[1] > 0:
            top2 = torch.topk(masked_scores, k=min(2, masked_scores.shape[1]), dim=-1).values
            margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.full((1,), 1e9, device=device, dtype=dtype)
            noop_score = scores[:, 0] if bool(valid_mask[:, 0].all().item()) else torch.full((1,), -1e9, device=device, dtype=dtype)
            selected_score = masked_scores.gather(1, pred_idx[:, None]).squeeze(1)
            low_conf = bool(
                self.enable_alignment_low_conf_noop
                and not ('b2_bounded_applied' in locals() and b2_bounded_applied)
                and (margin < 0.12).item()
                and (selected_score < noop_score + 0.05).item()
            )
            if low_conf:
                pred_idx = torch.zeros_like(pred_idx)
                self._alignment_low_conf_noop_count += 1
                masked_scores = scores.masked_fill(~valid_mask, -1e9)
        selected_local = candidate_actions.gather(1, pred_idx[:, None, None].expand(-1, 1, candidate_actions.shape[-1])).squeeze(1)
        selected_scale = step_scale.gather(1, pred_idx[:, None]).squeeze(1).clamp(0.0, 2.0)
        if yaw_priority_active and yaw_priority_scale_cap is not None:
            selected_scale = torch.clamp(selected_scale, max=float(yaw_priority_scale_cap))
        if xy_priority_active and xy_priority_scale_cap is not None:
            selected_scale = torch.clamp(selected_scale, max=float(xy_priority_scale_cap))
        if final_handoff_polish_scale_cap is not None:
            selected_scale = torch.clamp(selected_scale, max=float(final_handoff_polish_scale_cap))
        selected_local = selected_local * selected_scale[:, None]
        selected_group = candidate_group_index.gather(1, pred_idx[:, None]).squeeze(1)
        return {
            "delta_pose": selected_local,
            "delta_pose_gated": selected_local,
            "candidate_step_scale": step_scale,
            "selected_step_scale": selected_scale,
            "handoff_yaw_priority_active": torch.tensor([float(yaw_priority_active)], device=device, dtype=dtype),
            "handoff_xy_priority_active": torch.tensor([float(xy_priority_active)], device=device, dtype=dtype),
            "handoff_axis_priority": handoff_axis_priority,
            "handoff_xy_micro_active": torch.tensor(
                [float(final_handoff_polish_mode == "xy_micro")],
                device=device,
                dtype=dtype,
            ),
            "final_handoff_polish_active": torch.tensor(
                [float(final_handoff_polish_scale_cap is not None)],
                device=device,
                dtype=dtype,
            ),
            "alpha": torch.ones((1,), device=device, dtype=dtype),
            "ready_to_close_logits": outputs.get(
                "ready_to_close_logits",
                torch.zeros((1,), device=device, dtype=dtype),
            ),
            "ready_to_close": outputs.get(
                "ready_to_close",
                torch.zeros((1,), device=device, dtype=dtype),
            ),
            "near_ready_residual_active": torch.tensor(
                [float((residual_scores is not None) or (group_residual_logits is not None))],
                device=device,
                dtype=dtype,
            ),
            "near_ready_specialist_active": torch.tensor([float(specialist_active)], device=device, dtype=dtype),
            "candidate_scores": scores,
            "group_logits": group_logits,
            "pred_group_index": pred_group,
            "pred_candidate_index": pred_idx,
            "pred_candidate_group_index": selected_group,
        }

    def _score_outer_rescue_candidate(
        self,
        controller,
        current_delta: np.ndarray,
        next_delta: np.ndarray,
        candidate_local: np.ndarray,
        base_action_local: np.ndarray,
    ) -> float:
        current_xy, current_z, current_yaw, current_tilt = self._alignment_error_components(current_delta)
        next_xy, next_z, next_yaw, next_tilt = self._alignment_error_components(next_delta)
        inner_xy, inner_z, inner_yaw, inner_tilt = self._support_band_limits(controller, band="inner")
        outer_xy, _, outer_yaw, _outer_tilt = self._support_band_limits(controller, band="outer")
        lateral_mag = float(np.linalg.norm(np.asarray(candidate_local[:2], dtype=np.float32)))
        vertical_mag = float(abs(float(candidate_local[2])))
        yaw_mag = float(abs(float(candidate_local[5])))

        xy_improve = (current_xy - next_xy) / max(inner_xy, 1e-6)
        z_improve = (current_z - next_z) / max(inner_z, 1e-6)
        yaw_improve = (current_yaw - next_yaw) / max(max(inner_yaw, 0.08), 1e-6)
        use_tilt = not bool(getattr(controller, "_ignore_tilt_alignment", False))
        tilt_improve = ((current_tilt - next_tilt) / max(max(inner_tilt, 0.08), 1e-6)) if use_tilt else 0.0

        descend_xy_gate = min(outer_xy, max(inner_xy * 0.6, self.close_veto_xy_threshold * 1.5))
        descend_yaw_gate = max(min(inner_yaw, outer_yaw), 0.12)
        descend_ready = bool(next_xy <= descend_xy_gate and next_yaw <= descend_yaw_gate)

        score = 8.0 * xy_improve + 0.5 * yaw_improve + 0.8 * tilt_improve
        if descend_ready:
            score += 4.0 * z_improve
        elif vertical_mag > 1e-8:
            score -= 3.0 * vertical_mag / max(inner_z, 1e-6)

        if next_xy > current_xy + 1e-4:
            score -= 2.5 * lateral_mag / max(inner_xy, 1e-6)
        if current_z <= inner_z * 1.5 and next_z > current_z + 1e-4:
            score -= 1.0 * (next_z - current_z) / max(inner_z, 1e-6)
        if yaw_mag > 1e-8 and lateral_mag < 1e-8 and current_xy > descend_xy_gate:
            score -= 0.75 * yaw_mag / max(max(inner_yaw, 0.08), 1e-6)
        tilt_mag = float(np.linalg.norm(np.asarray(candidate_local[3:5], dtype=np.float32)))
        if use_tilt and tilt_mag > 1e-8 and current_tilt > next_tilt + 1e-5:
            score += 0.35
        elif use_tilt and tilt_mag > 1e-8 and next_tilt > current_tilt + 1e-5:
            score -= 0.35

        if self._delta_within_band(next_delta, controller, band="inner"):
            score += 2.0
        if next_xy <= self.close_veto_xy_threshold * 1.5:
            score += 0.4
        if next_z <= self.close_veto_abs_z_threshold * 2.0:
            score += 0.4

        score += 0.30 * self._safe_cosine(candidate_local[:3], current_delta[:3])
        if base_action_local is not None:
            score += 0.20 * self._safe_cosine(candidate_local[:3], base_action_local[:3])
        return float(score)

    def _run_pose_field_outer_rescue(self, controller, gripper_pose, base_action_local):
        import torch

        device = next(controller.parameters()).device
        dtype = next(controller.parameters()).dtype
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        )
        current_pose = None if gripper_pose is None else np.asarray(gripper_pose, dtype=np.float32).reshape(7)
        target_pose = getattr(controller, "_runtime_target_pose_7d", None)
        if target_pose is not None:
            target_pose = np.asarray(target_pose, dtype=np.float32).reshape(7)
        if base_action_local is None:
            base_action_local = np.zeros(6, dtype=np.float32)
        else:
            base_action_local = np.asarray(base_action_local, dtype=np.float32).reshape(6)

        candidate_actions = np.asarray(
            getattr(controller, "_teacher_candidate_actions_local", controller._candidate_actions_local.detach().cpu().numpy()),
            dtype=np.float32,
        )
        candidate_group_index = np.asarray(
            getattr(controller, "_teacher_candidate_group_index", controller._candidate_group_index.detach().cpu().numpy()),
            dtype=np.int64,
        )
        scores = np.full((candidate_actions.shape[0],), -1e9, dtype=np.float32)
        best_idx = 0
        best_score = -1e9
        best_next_delta = current_delta.copy()
        for idx, candidate in enumerate(candidate_actions):
            rescue_delta_local = np.asarray(candidate, dtype=np.float32) * float(self.learned_residual_scale)
            total_delta_local = base_action_local + rescue_delta_local
            if current_pose is not None and target_pose is not None:
                next_pose = self._apply_local_offset_to_pose(current_pose, total_delta_local)
                next_delta = self._pose_delta_local_between(next_pose, target_pose)
            else:
                next_delta = current_delta - total_delta_local
            score = self._score_outer_rescue_candidate(
                controller,
                current_delta=current_delta,
                next_delta=next_delta,
                candidate_local=rescue_delta_local,
                base_action_local=base_action_local,
            )
            scores[idx] = float(score)
            if score > best_score:
                best_score = float(score)
                best_idx = int(idx)
                best_next_delta = np.asarray(next_delta, dtype=np.float32).copy()

        selected_local = torch.from_numpy(candidate_actions[best_idx]).unsqueeze(0).to(device=device, dtype=dtype)
        pred_group = torch.tensor([int(candidate_group_index[best_idx])], device=device, dtype=torch.long)
        group_logits = torch.full(
            (1, int(np.max(candidate_group_index)) + 1),
            -1e9,
            device=device,
            dtype=dtype,
        )
        group_logits[0, pred_group.item()] = 0.0
        return {
            "delta_pose": selected_local,
            "delta_pose_gated": selected_local,
            "alpha": torch.ones((1,), device=device, dtype=dtype),
            "ready_to_close": torch.zeros((1,), device=device, dtype=dtype),
            "gripper_logits": torch.zeros((1, 3), device=device, dtype=dtype),
            "candidate_scores": torch.from_numpy(scores).unsqueeze(0).to(device=device, dtype=dtype),
            "group_logits": group_logits,
            "pred_group_index": pred_group,
            "pred_candidate_index": torch.tensor([best_idx], device=device, dtype=torch.long),
            "pred_candidate_group_index": pred_group,
            "next_delta_pose": torch.from_numpy(best_next_delta).unsqueeze(0).to(device=device, dtype=dtype),
            "used_outer_rescue": True,
        }

    def _apply_readiness_gripper(self, a_exec, outputs, controller, gripper_open, force_reading=None):
        if not self.enable_readiness_gripper:
            return a_exec
        if not bool(getattr(controller, "_readiness_heads_loaded", False)):
            self._readiness_heads_missing_count += 1
            return a_exec
        if "ready_to_close" not in outputs:
            self._readiness_heads_missing_count += 1
            return a_exec

        ready_prob = float(outputs["ready_to_close"].squeeze(0).float().cpu().item())
        fire_only_head = bool(getattr(controller, "fire_only_head", False))
        hold_prob = 0.0
        state = 1 if ready_prob >= self.readiness_close_threshold else 0
        state_conf = ready_prob if state == 1 else (1.0 - ready_prob)
        if (not fire_only_head) and ("gripper_logits" in outputs):
            import torch

            logits = outputs["gripper_logits"].squeeze(0).float().cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
            state = int(np.argmax(probs))  # 0=open, 1=close, 2=hold
            state_conf = float(probs[state])
            hold_prob = float(probs[2])
        is_open = gripper_open is None or float(gripper_open) >= self.alignment_open_threshold
        force_norm = float(np.linalg.norm(np.asarray(force_reading, dtype=np.float32))) if force_reading is not None else 0.0

        self._readiness_eval_count += 1
        self._ready_prob_sum += ready_prob
        self._last_ready_prob = ready_prob
        self._last_basin_positive = float(ready_prob >= self.readiness_close_threshold)
        self._last_internal_readiness_fire_applied = False
        self._last_internal_readiness_fire_ready = False
        self._last_internal_readiness_band_ready = False
        if ready_prob >= self.readiness_close_threshold:
            self._ready_positive_count += 1

        if fire_only_head:
            if self._gripper_fsm_state != "block_open":
                self._last_close_veto_ready = False
                self._close_veto_ready_streak = 0
                a_exec[6] = 0.0
                self._readiness_hold_override_count += 1
                self._last_internal_readiness_fire_applied = True
                return a_exec

                fire_band_ready = bool(self._internal_readiness_fire_band_ready(controller, gripper_open, depth_proximity))
            self._last_internal_readiness_band_ready = fire_band_ready
            fire_ready = bool(
                is_open
                and fire_band_ready
                and ready_prob >= self.readiness_close_threshold
            )
            self._last_internal_readiness_fire_ready = fire_ready
            self._fire_ready_streak = self._fire_ready_streak + 1 if fire_ready else 0
            if self._fire_ready_streak >= self.fire_hysteresis_frames:
                a_exec[6] = 0.0
                self._readiness_close_override_count += 1
                self._gripper_fsm_state = "hold_after_verified_contact"
                self._fire_ready_streak = 0
                self._last_internal_readiness_fire_applied = True
            return a_exec

        if self._gripper_fsm_state == "verify_contact":
            self._last_close_veto_ready = False
            self._close_veto_ready_streak = 0
            a_exec[6] = 0.0
            self._readiness_hold_override_count += 1
            verified_contact = bool(
                force_norm >= self.verify_force_threshold
                or (not is_open)
                or ((not is_open) and state == 2 and state_conf >= self.gripper_override_confidence)
                or ((not is_open) and hold_prob >= self.gripper_override_confidence)
            )
            if verified_contact:
                self._gripper_fsm_state = "hold_after_verified_contact"
                self._verified_contact_count += 1
                self._verify_steps_remaining = 0
            else:
                self._verify_steps_remaining = max(self._verify_steps_remaining - 1, 0)
                if self._verify_steps_remaining <= 0:
                    self._gripper_fsm_state = "block_open"
                    self._verify_fail_reopen_count += 1
                    self._fire_ready_streak = 0
                    a_exec[6] = 1.0
                    self._readiness_open_override_count += 1
            return a_exec

        if self._gripper_fsm_state == "hold_after_verified_contact":
            self._last_close_veto_ready = False
            self._close_veto_ready_streak = 0
            a_exec[6] = 0.0
            self._readiness_hold_override_count += 1
            return a_exec

        fire_ready = bool(
            is_open
            and ready_prob >= self.readiness_close_threshold
            and (
                fire_only_head
                or (state == 1 and state_conf >= self.gripper_override_confidence)
            )
        )
        self._fire_ready_streak = self._fire_ready_streak + 1 if fire_ready else 0
        if self._fire_ready_streak >= self.fire_hysteresis_frames:
            a_exec[6] = 0.0
            self._readiness_close_override_count += 1
            self._gripper_fsm_state = "verify_contact"
            self._verify_steps_remaining = self.verify_contact_steps
            self._fire_ready_streak = 0
        else:
            a_exec[6] = 1.0
            self._readiness_open_override_count += 1
        return a_exec

    def _apply_alignment_conditioned_fire(self, a_exec, controller, gripper_open):
        if not self.enable_alignment_conditioned_fire:
            return a_exec
        if self._gripper_fsm_state != "block_open":
            a_exec[6] = 0.0
            self._readiness_hold_override_count += 1
            self._last_alignment_fire_decision = True
            return a_exec

        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        xy_error = float(np.linalg.norm(current_delta[:2])) if current_delta.size >= 2 else np.inf
        z_error = float(abs(current_delta[2])) if current_delta.size >= 3 else np.inf
        yaw_error = float(abs(current_delta[5])) if current_delta.size >= 6 else np.inf
        support_ok = bool(
            float(getattr(controller, "_runtime_current_basin_distance", np.inf))
            <= float(getattr(controller, "_support_basin_distance_max", np.inf))
        )
        is_open = gripper_open is None or float(gripper_open) >= self.alignment_open_threshold
        alignment_complete = bool(
            is_open
            and support_ok
            and xy_error <= self.alignment_fire_xy_threshold
            and z_error <= self.alignment_fire_abs_z_threshold
            and (
                self.alignment_fire_yaw_threshold < 0.0
                or yaw_error <= self.alignment_fire_yaw_threshold
            )
        )

        self._alignment_complete_eval_count += 1
        self._last_alignment_complete = bool(alignment_complete)
        self._last_alignment_fire_xy = xy_error
        self._last_alignment_fire_z = z_error
        self._last_alignment_fire_yaw = yaw_error
        if alignment_complete:
            self._alignment_complete_positive_count += 1

        self._fire_ready_streak = self._fire_ready_streak + 1 if alignment_complete else 0
        if self._fire_ready_streak >= self.fire_hysteresis_frames:
            a_exec[6] = 0.0
            self._gripper_fsm_state = "hold_after_verified_contact"
            self._fire_ready_streak = 0
            self._alignment_fire_close_count += 1
            self._readiness_close_override_count += 1
            self._last_alignment_fire_decision = True
        else:
            a_exec[6] = 1.0
            self._readiness_open_override_count += 1
            self._last_alignment_fire_decision = False
        return a_exec

    def _internal_readiness_fire_band_ready(
        self,
        controller,
        gripper_open,
        depth_proximity: Optional[float] = None,
    ) -> bool:
        controller = self._student_vnext_ready_gate_controller(controller)
        controller_type = str(getattr(controller, "_controller_type", ""))
        student_ready_gate = bool(
            self._student_vnext_ready_gate_supported(controller)
            and controller_type == "alignment_tc_student_vnext"
        )
        if controller is None or (controller_type != "pose_field_scorer" and not student_ready_gate):
            return False
        if not self._runtime_has_motion_target(controller) and not self._runtime_has_handoff_geometry(controller):
            return False
        if gripper_open is not None and float(gripper_open) < self.alignment_open_threshold:
            return False
        if student_ready_gate:
            student_gate = self._student_vnext_ready_gate_state(
                controller,
                gripper_open=gripper_open,
                depth_proximity=depth_proximity,
            )
            if bool(student_gate.get("handoff_ready_applied", False)):
                return True
        handoff_spec_name = str(getattr(controller, "_runtime_handoff_spec_name", "none"))
        if handoff_spec_name and handoff_spec_name != "none":
            return bool(getattr(controller, "_runtime_handoff_ready", False))
        current_delta = np.asarray(
            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        xy_error = float(np.linalg.norm(current_delta[:2])) if current_delta.size >= 2 else np.inf
        z_error = float(abs(current_delta[2])) if current_delta.size >= 3 else np.inf
        yaw_error = float(abs(current_delta[5])) if current_delta.size >= 6 else np.inf
        support_inner_satisfied = bool(self._delta_within_band(current_delta, controller, band="inner"))
        xy_threshold = float(getattr(controller, "_ready_band_xy_threshold", self.close_veto_xy_threshold))
        z_threshold = float(getattr(controller, "_ready_band_abs_z_threshold", self.close_veto_abs_z_threshold))
        yaw_threshold = float(getattr(controller, "_ready_band_yaw_threshold", self.close_veto_yaw_threshold))
        basin_distance_threshold = float(
            getattr(controller, "_ready_band_basin_distance_threshold", 1.0)
        )
        basin_distance = float(getattr(controller, "_runtime_current_basin_distance", np.inf))
        return bool(
            support_inner_satisfied
            and (
                basin_distance_threshold < 0.0
                or basin_distance <= basin_distance_threshold
            )
            and xy_error <= xy_threshold
            and z_error <= z_threshold
            and (
                yaw_threshold < 0.0
                or yaw_error <= yaw_threshold
            )
        )

    def _apply_alignment_close_veto(
        self,
        a_exec,
        controller,
        gripper_open,
        step_idx: Optional[int] = None,
        depth_proximity: Optional[float] = None,
    ):
        self._last_close_veto_blocked = False
        self._last_close_veto_runtime_geometry_fallback_used = False
        self._last_close_veto_runtime_geometry_fallback_ready = False
        self._last_bounded_auto_close_ready = False
        self._last_bounded_auto_close_applied = False
        self._last_force_close_after_b2_applied = False
        controller = self._student_vnext_ready_gate_controller(controller)
        controller_type = str(getattr(controller, "_controller_type", "")) if controller is not None else ""
        student_ready_gate = bool(
            self._student_vnext_ready_gate_supported(controller)
            and controller_type == "alignment_tc_student_vnext"
        )
        is_open = gripper_open is None or float(gripper_open) >= self.alignment_open_threshold
        wants_close = float(a_exec[6]) <= self.alignment_close_command_threshold
        planner_close_intent_now = bool(getattr(controller, "_alignment_planner_close_intent", False))
        if student_ready_gate:
            student_gate = self._student_vnext_ready_gate_state(
                controller,
                gripper_open=gripper_open,
                depth_proximity=depth_proximity,
            )
            if bool(student_gate.get("close_ready_applied", False)) and is_open:
                a_exec = np.asarray(a_exec, dtype=np.float32).copy()
                a_exec[:6] = 0.0
                a_exec[6] = 0.0
                self._close_veto_pass_count += 1
                self._close_latch_remaining = max(self.close_latch_steps, 0)
                self._phase1_post_close_hold_remaining = max(self._phase1_post_close_hold_remaining, self.close_latch_steps)
                self._close_veto_recent_block_streak = 0
                self._phase1_close_arbiter_state = "CLOSE_CANDIDATE"
                self._phase1_close_command_source = "student_vnext_close_ready"
                self._last_close_state_machine = self._make_close_state_snapshot(
                    state="student_vnext_close_ready",
                    action_decision="student_vnext_close_ready",
                    wants_close=bool(wants_close),
                    planner_close_intent=bool(planner_close_intent_now),
                    gripper_open=None if gripper_open is None else float(gripper_open),
                    is_open=True,
                    blocked_reason="student_vnext_close_ready",
                )
                self._update_close_intent_shadow_candidate(gripper_open)
                return a_exec
        if (
            not self.enable_alignment_close_veto
            or controller is None
            or (controller_type != "pose_field_scorer" and not student_ready_gate)
            or (
                not self._runtime_has_motion_target(controller)
                and not self._runtime_has_handoff_geometry(controller)
                and not (
                    student_ready_gate
                    and bool(
                        self._student_vnext_ready_gate_state(
                            controller,
                            gripper_open=gripper_open,
                            depth_proximity=depth_proximity,
                        ).get("close_ready_applied", False)
                        or self._student_vnext_ready_gate_state(
                            controller,
                            gripper_open=gripper_open,
                            depth_proximity=depth_proximity,
                        ).get("handoff_ready_applied", False)
                    )
                )
            )
        ):
            self._last_close_veto_ready = False
            self._close_latch_remaining = 0
            delegate_state = "close_veto_disabled_or_no_target"
            delegate_reason = "disabled_or_no_target"
            if (
                self.enable_phase1_force_reflex
                and controller is not None
                and self._phase1_bridge_runtime_segment()
            ):
                delegate_state = "phase1_force_reflex_delegate"
                delegate_reason = "phase1_force_reflex_delegate"
            if student_ready_gate:
                delegate_state = "student_vnext_ready_gate_delegate"
                delegate_reason = "student_vnext_ready_gate_delegate"
            self._last_close_state_machine = self._make_close_state_snapshot(
                state=delegate_state,
                action_decision="passthrough",
                wants_close=bool(float(np.asarray(a_exec, dtype=np.float32)[6]) <= self.alignment_close_command_threshold),
                gripper_open=None if gripper_open is None else float(gripper_open),
                blocked_reason=delegate_reason,
            )
            self._update_close_intent_shadow_candidate(gripper_open)
            return a_exec
        if self.enable_readiness_gripper and self._controller_has_internal_readiness(controller) and (
            self._last_internal_readiness_fire_applied or self._gripper_fsm_state != "block_open"
        ):
            self._last_close_state_machine = self._make_close_state_snapshot(
                state="internal_readiness_gripper_active",
                action_decision="passthrough",
                wants_close=bool(float(np.asarray(a_exec, dtype=np.float32)[6]) <= self.alignment_close_command_threshold),
                planner_close_intent=bool(getattr(controller, "_alignment_planner_close_intent", False)),
                gripper_open=None if gripper_open is None else float(gripper_open),
                blocked_reason="internal_readiness_gripper_active",
            )
            self._update_close_intent_shadow_candidate(gripper_open)
            return a_exec
        if not is_open:
            self._close_latch_remaining = 0
            self._close_veto_settle_remaining = 0
            self._close_veto_ready_streak = 0
            self._close_veto_ready_step_idx = -1 if step_idx is None else int(step_idx)
            self._last_close_veto_ready = False
            self._bounded_auto_close_ready_streak = 0
            self._last_close_state_machine = self._make_close_state_snapshot(
                state="gripper_not_open",
                action_decision="passthrough",
                wants_close=bool(wants_close),
                planner_close_intent=bool(planner_close_intent_now),
                gripper_open=None if gripper_open is None else float(gripper_open),
                is_open=False,
                blocked_reason="gripper_not_open",
            )
            self._update_close_intent_shadow_candidate(gripper_open)
            return a_exec
        if not wants_close:
            if (
                self.enable_force_close_after_b2_eval
                and self.enable_b2_candidate_bounded_v0
                and bool(self._last_b2_candidate_bounded_gate_open)
                and is_open
            ):
                a_exec = np.asarray(a_exec, dtype=np.float32).copy()
                a_exec[:6] = 0.0
                a_exec[6] = 0.0
                self._force_close_after_b2_eval_count += 1
                self._last_force_close_after_b2_applied = True
                self._close_veto_pass_count += 1
                self._close_latch_remaining = 0
                self._close_veto_recent_block_streak = 0
                self._bounded_auto_close_ready_streak = 0
                self._last_close_state_machine.update(
                    {
                        "action_decision": "force_close_after_b2",
                        "wants_close": False,
                        "planner_close_intent": bool(planner_close_intent_now),
                    }
                )
                self._update_close_intent_shadow_candidate(gripper_open)
                return a_exec
            if student_ready_gate:
                close_ready = self._close_veto_ready(controller, gripper_open, step_idx=step_idx, planner_close_intent=planner_close_intent_now)
                if close_ready:
                    a_exec = np.asarray(a_exec, dtype=np.float32).copy()
                    a_exec[:6] = 0.0
                    a_exec[6] = 0.0
                    self._close_veto_pass_count += 1
                    self._close_latch_remaining = max(self.close_latch_steps, 0)
                    self._phase1_post_close_hold_remaining = max(self._phase1_post_close_hold_remaining, self.close_latch_steps)
                    self._close_veto_recent_block_streak = 0
                    self._phase1_close_arbiter_state = "CLOSE_CANDIDATE"
                    self._phase1_close_command_source = "student_vnext_close_ready"
                    self._last_close_state_machine.update(
                        {
                            "action_decision": "student_vnext_close_ready",
                            "wants_close": False,
                            "planner_close_intent": bool(planner_close_intent_now),
                            "latch_remaining": int(self._close_latch_remaining),
                        }
                    )
                    self._update_close_intent_shadow_candidate(gripper_open)
                    return a_exec
            self._bounded_auto_close_eval_count += 1
            auto_close_ready = bool(self._bounded_auto_close_ready(controller, gripper_open))
            self._last_bounded_auto_close_ready = auto_close_ready
            self._bounded_auto_close_ready_streak = (
                self._bounded_auto_close_ready_streak + 1 if auto_close_ready else 0
            )
            if self._bounded_auto_close_ready_streak >= self.bounded_auto_close_stable_frames:
                auto_close_ready_streak = int(self._bounded_auto_close_ready_streak)
                a_exec = np.asarray(a_exec, dtype=np.float32).copy()
                a_exec[:6] = 0.0
                a_exec[6] = 0.0
                self._bounded_auto_close_apply_count += 1
                self._last_bounded_auto_close_applied = True
                self._close_veto_pass_count += 1
                self._close_latch_remaining = max(self.close_latch_steps, 0)
                self._phase1_post_close_hold_remaining = max(self._phase1_post_close_hold_remaining, self.close_latch_steps)
                self._close_veto_recent_block_streak = 0
                self._bounded_auto_close_ready_streak = 0
                self._last_close_state_machine.update(
                    {
                        "state": "bounded_auto_close_ready",
                        "action_decision": "bounded_auto_close",
                        "wants_close": False,
                        "planner_close_intent": bool(planner_close_intent_now),
                        "ready_streak": auto_close_ready_streak,
                    }
                )
                self._update_close_intent_shadow_candidate(gripper_open)
                return a_exec
            if self._phase1_post_close_hold_remaining > 0 and self.manager.phase == StagePhase.ALIGN and not bool(getattr(self.manager, "has_object_in_hand", False)):
                a_exec = np.asarray(a_exec, dtype=np.float32).copy()
                a_exec[:6] = 0.0
                a_exec[6] = 0.0
                self._phase1_post_close_hold_remaining -= 1
                self._last_phase1_post_close_hold_active = True
                if self._phase1_post_close_hold_remaining <= 0:
                    self._phase1_post_close_hold_expire_count += 1
                self._last_close_state_machine.update(
                    {
                        "action_decision": "post_close_hold",
                        "wants_close": False,
                        "planner_close_intent": bool(planner_close_intent_now),
                        "hold_remaining": int(self._phase1_post_close_hold_remaining),
                    }
                )
                self._update_close_intent_shadow_candidate(gripper_open)
                return a_exec
            if self.close_latch_enabled and self._close_latch_remaining > 0:
                close_ready = self._close_veto_ready(
                    controller,
                    gripper_open,
                    step_idx=step_idx,
                    planner_close_intent=False,
                )
                if close_ready:
                    a_exec = np.asarray(a_exec, dtype=np.float32).copy()
                    a_exec[:6] = 0.0
                    a_exec[6] = 0.0
                    self._close_veto_pass_count += 1
                    self._close_latch_release_count += 1
                    self._close_latch_remaining = max(self.close_latch_steps, 0)
                    self._phase1_post_close_hold_remaining = max(self._phase1_post_close_hold_remaining, self.close_latch_steps)
                    if self.close_veto_settle_steps > 0:
                        self._close_veto_settle_count += 1
                        self._close_veto_settle_remaining = self.close_veto_settle_steps
                    self._close_veto_recent_block_streak = 0
                    self._last_close_state_machine.update(
                        {
                            "action_decision": "latch_release_close",
                            "wants_close": False,
                            "planner_close_intent": bool(planner_close_intent_now),
                            "latch_remaining": int(self._close_latch_remaining),
                        }
                    )
                    self._update_close_intent_shadow_candidate(gripper_open)
                    return a_exec
                self._close_latch_remaining -= 1
                if self._close_latch_remaining <= 0:
                    self._close_latch_expire_count += 1
                self._last_close_state_machine.update(
                    {
                        "action_decision": "latch_wait",
                        "wants_close": False,
                        "planner_close_intent": bool(planner_close_intent_now),
                        "latch_remaining": int(self._close_latch_remaining),
                    }
                )
                self._update_close_intent_shadow_candidate(gripper_open)
                return a_exec
            self._close_veto_ready_streak = 0
            self._close_veto_ready_step_idx = -1 if step_idx is None else int(step_idx)
            self._last_close_veto_ready = False
            self._close_veto_recent_block_streak = 0
            self._last_close_state_machine = self._make_close_state_snapshot(
                state="open_command",
                action_decision="passthrough_open",
                wants_close=False,
                planner_close_intent=bool(planner_close_intent_now),
                gripper_open=None if gripper_open is None else float(gripper_open),
                is_open=True,
            )
            self._update_close_intent_shadow_candidate(gripper_open)
            return a_exec
        self._bounded_auto_close_ready_streak = 0
        close_ready = self._close_veto_ready(
            controller,
            gripper_open,
            step_idx=step_idx,
            planner_close_intent=planner_close_intent_now,
        )
        if close_ready:
            if self.close_veto_settle_steps > 0:
                a_exec = np.asarray(a_exec, dtype=np.float32).copy()
                a_exec[:6] = 0.0
                a_exec[6] = 0.0
                self._close_veto_settle_count += 1
                self._close_veto_settle_remaining = self.close_veto_settle_steps
            self._close_veto_pass_count += 1
            self._close_latch_remaining = max(self.close_latch_steps, 0)
            self._phase1_post_close_hold_remaining = max(self._phase1_post_close_hold_remaining, self.close_latch_steps)
            self._close_veto_recent_block_streak = 0
            if student_ready_gate:
                self._phase1_close_arbiter_state = "CLOSE_CANDIDATE"
                self._phase1_close_command_source = "student_vnext_close_ready"
            self._last_close_state_machine.update(
                {
                    "action_decision": "pass_close",
                    "wants_close": True,
                    "planner_close_intent": bool(planner_close_intent_now),
                    "latch_remaining": int(self._close_latch_remaining),
                    "settle_remaining": int(self._close_veto_settle_remaining),
                }
            )
            self._update_close_intent_shadow_candidate(gripper_open)
            return a_exec
        a_exec = np.asarray(a_exec, dtype=np.float32).copy()
        a_exec[6] = 1.0
        self._close_veto_block_count += 1
        self._last_close_veto_blocked = True
        self._close_veto_recent_block_streak += 1
        if self.close_latch_enabled and self.close_latch_steps > 0:
            if self._close_latch_remaining <= 0:
                self._close_latch_set_count += 1
            self._close_latch_remaining = self.close_latch_steps
        self._last_close_state_machine.update(
            {
                "action_decision": "block_close",
                "wants_close": True,
                "planner_close_intent": bool(planner_close_intent_now),
                "latch_remaining": int(self._close_latch_remaining),
                "blocked_reason": self._last_close_state_machine.get("blocked_reason", "not_ready"),
            }
        )
        self._update_close_intent_shadow_candidate(gripper_open)
        return a_exec

    def step(
        self,
        a_base_7d: np.ndarray,
        step_idx: int,
        force_reading: Optional[np.ndarray] = None,
        gripper_z: Optional[float] = None,
        front_rgb=None,
        wrist_rgb=None,
        wrist_depth=None,
        ft_hist=None,
        proprio=None,
        gripper_pose: Optional[np.ndarray] = None,
        gripper_open: Optional[float] = None,
        future_gripper_actions=None,
    ) -> np.ndarray:
        if self.mode == "planner_only":
            return a_base_7d.copy()

        depth_proximity = self.compute_depth_proximity(wrist_depth)
        self._last_alignment_gate_debug = {
            "gate_open": False,
            "blocked_reason": "phase",
            "depth_proximity": None if depth_proximity is None else float(depth_proximity),
            "near_target": False,
            "gripper_open": None if gripper_open is None else float(gripper_open),
            "gripper_still_open": False,
            "planner_close_intent": False,
            "close_requirement_satisfied": False,
            "alignment_window_active": False,
            "short_window_available": self._alignment_window_corrections < self.max_alignment_corrections_per_window,
            "residual_cooldown": int(self._residual_cooldown),
        }
        self.manager.update(
            force_reading=force_reading,
            gripper_pose=gripper_pose,
            gripper_open=gripper_open,
            depth_proximity=depth_proximity,
            base_action=a_base_7d,
        )

        if self.safety.check_force_stop(force_reading):
            a_stop = np.zeros(7, dtype=np.float32)
            a_stop[6] = a_base_7d[6]
            return a_stop

        a_exec = a_base_7d.copy()
        current_quat = self._extract_current_quat(proprio, gripper_pose=gripper_pose)
        base_action_local = world_delta_to_local(a_base_7d[:6], current_quat)
        self._last_scorer_candidate_index = -1
        self._last_scorer_group_index = -1
        self._last_selected_step_scale = 1.0
        self._last_outer_rescue_active = False
        self._last_alignment_takeover_active = False
        self._alignment_active = False
        planner_close_intent_active = False

        if self._residual_cooldown > 0:
            self._residual_cooldown -= 1

        controller = None
        trigger_outputs = None
        use_outer_rescue = False
        alignment_zone_state = "planner_only"
        if self.alignment_controller is not None:
            setattr(self.alignment_controller, "_alignment_planner_close_intent", False)
        if self.manager.phase == StagePhase.ALIGN and self.mode in ("alignment", "full"):
            self._planner_close_intent_eval_count += 1
            planner_close_intent = self._planner_close_intent(a_base_7d, future_gripper_actions)
            planner_close_intent_active = bool(planner_close_intent)
            if planner_close_intent:
                self._planner_close_intent_count += 1
            if self._residual_cooldown > 0:
                self._alignment_gate_block_count += 1
                self._alignment_gate_block_reason_counts["cooldown"] += 1
                self._last_alignment_gate_debug = {
                    "gate_open": False,
                    "blocked_reason": "cooldown",
                    "depth_proximity": None if depth_proximity is None else float(depth_proximity),
                    "near_target": bool(
                        depth_proximity is not None
                        and np.isfinite(depth_proximity)
                        and float(depth_proximity) < self.alignment_depth_threshold
                    ),
                    "gripper_open": None if gripper_open is None else float(gripper_open),
                    "gripper_still_open": bool(
                        gripper_open is not None and float(gripper_open) >= self.alignment_open_threshold
                    ),
                    "planner_close_intent": bool(planner_close_intent),
                    "close_requirement_satisfied": bool(
                        planner_close_intent or (not self.require_close_intent_for_alignment)
                    ),
                    "alignment_window_active": False,
                    "support_inner_satisfied": True,
                    "support_outer_satisfied": True,
                    "support_soft_override": False,
                    "support_soft_override_reason": "cooldown",
                    "use_outer_rescue": False,
                    "short_window_available": self._alignment_window_corrections < self.max_alignment_corrections_per_window,
                    "residual_cooldown": int(self._residual_cooldown),
                }
            else:
                controller = self.alignment_controller
                if controller is not None:
                    setattr(controller, "_alignment_planner_close_intent", bool(planner_close_intent))
                alignment_zone_state = self._select_alignment_zone(controller, depth_proximity, gripper_open)
                current_delta = np.asarray(
                    getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                    dtype=np.float32,
                ) if controller is not None else np.zeros(6, dtype=np.float32)
                support_inner_satisfied = self._delta_within_band(current_delta, controller, band="inner") if controller is not None else False
                support_outer_satisfied = self._delta_within_band(current_delta, controller, band="outer") if controller is not None else False
                support_soft_override, support_soft_override_reason = self._phase1_bridge_support_soft_override(
                    controller,
                    depth_proximity,
                    gripper_open,
                ) if controller is not None else (False, "no_controller")
                refine_band_satisfied = bool(alignment_zone_state in ("assist", "takeover"))
                takeover_band_satisfied = bool(alignment_zone_state == "takeover")
                close_ready_for_planner = (
                    bool(
                        self._close_veto_ready(
                            controller,
                            gripper_open,
                            step_idx=step_idx,
                            planner_close_intent=bool(planner_close_intent),
                        )
                    )
                    if controller is not None
                    else False
                )
                use_outer_rescue = bool(
                    controller is not None
                    and self.enable_outer_rescue
                    and alignment_zone_state == "assist"
                    and support_outer_satisfied
                    and not support_inner_satisfied
                )
                self._last_outer_rescue_active = use_outer_rescue
                near_target = (
                    depth_proximity is not None
                    and np.isfinite(depth_proximity)
                    and float(depth_proximity) < self.alignment_depth_threshold
                )
                gripper_still_open = gripper_open is None or float(gripper_open) >= self.alignment_open_threshold
                self._last_alignment_gate_debug = {
                    "gate_open": bool(alignment_zone_state != "planner_only"),
                    "blocked_reason": "none" if alignment_zone_state != "planner_only" else ("cooldown" if self._residual_cooldown > 0 else "support"),
                    "depth_proximity": None if depth_proximity is None else float(depth_proximity),
                    "near_target": bool(near_target),
                    "gripper_open": None if gripper_open is None else float(gripper_open),
                    "gripper_still_open": bool(gripper_still_open),
                    "planner_close_intent": bool(planner_close_intent),
                    "close_requirement_satisfied": bool(self._alignment_close_requirement_satisfied(controller)),
                    "alignment_window_active": bool(alignment_zone_state != "planner_only"),
                    "support_satisfied": bool(support_outer_satisfied or support_soft_override),
                    "support_inner_satisfied": bool(support_inner_satisfied),
                    "support_outer_satisfied": bool(support_outer_satisfied),
                    "support_soft_override": bool(support_soft_override),
                    "support_soft_override_reason": str(support_soft_override_reason),
                    "use_outer_rescue": bool(use_outer_rescue),
                    "close_ready_for_planner": bool(close_ready_for_planner),
                    "refine_band_satisfied": bool(refine_band_satisfied),
                    "takeover_band_satisfied": bool(takeover_band_satisfied),
                    "support_basin_distance_satisfied": bool(support_outer_satisfied),
                    "short_window_available": True,
                    "residual_cooldown": int(self._residual_cooldown),
                }
                if alignment_zone_state == "planner_only":
                    self._alignment_gate_block_count += 1
                    self._alignment_gate_block_reason_counts["support"] += 1
                    # Keep controller alive so the scorer/v2-shadow still runs.
                    # Only the correction application is blocked downstream.
        elif self._residual_cooldown == 0:
            if self.manager.phase in (StagePhase.INTERACT, StagePhase.RECOVER) and self.mode in ("contact", "full"):
                controller = self.contact_controller

        if (
            self.manager.phase == StagePhase.ALIGN
            and self.enable_readiness_gripper
            and self.close_trigger_controller is not None
        ):
            gripper_context = self.make_gripper_context(a_base_7d, future_gripper_actions)
            trigger_outputs = self._run_controller(
                self.close_trigger_controller,
                wrist_depth,
                ft_hist,
                proprio,
                base_action_local,
                step_idx,
                gripper_context=gripper_context,
            )

        if controller is not None:
            gripper_context = self.make_gripper_context(a_base_7d, future_gripper_actions)
            _near_zone_ok = self._check_near_zone_gate(controller)
            use_predictor_micro_assist_apply = bool(
                self.enable_v2_predictor_micro_assist_apply
                and self.manager.phase == StagePhase.ALIGN
                and getattr(controller, "_controller_type", "") == "pose_field_scorer"
            )
            alignment_takeover_active = bool(
                self.alignment_takeover_until_close_ready
                and self.manager.phase == StagePhase.ALIGN
                and getattr(controller, "_controller_type", "") == "pose_field_scorer"
                and alignment_zone_state == "takeover"
                and _near_zone_ok
                and (gripper_open is None or float(gripper_open) >= self.alignment_open_threshold)
            )
            # When near-zone gate blocks: use planner_only (no correction at all)
            # instead of assist, to let the planner finish coarse approach undisturbed.
            _nz_blocked = bool(self.enable_alignment_near_zone_gate and not _near_zone_ok)
            self._last_nz_blocked_zone = _nz_blocked
            if alignment_takeover_active:
                self._alignment_zone_state = "takeover"
            elif self.manager.phase == StagePhase.ALIGN:
                self._alignment_zone_state = "planner_only" if _nz_blocked else "assist"
            else:
                self._alignment_zone_state = "planner_only"
            self._alignment_active = bool(self.manager.phase == StagePhase.ALIGN and self._alignment_zone_state in ("assist", "takeover"))
            if use_predictor_micro_assist_apply:
                controller_base_action_local = np.asarray(base_action_local, dtype=np.float32)
            else:
                assist_base_scale = self.alignment_assist_base_scale
                if self._close_veto_recent_block_streak > 0 or planner_close_intent_active:
                    assist_base_scale = min(assist_base_scale, self.alignment_assist_close_block_base_scale)
                controller_base_action_local = (
                    np.zeros_like(base_action_local, dtype=np.float32)
                    if alignment_takeover_active
                    else (
                        np.asarray(base_action_local, dtype=np.float32) * assist_base_scale
                        if (self.manager.phase == StagePhase.ALIGN and self._alignment_zone_state == "assist")
                        else base_action_local
                    )
                )
            exec_base_action_world = local_delta_to_world(controller_base_action_local, current_quat)
            if self.manager.phase == StagePhase.ALIGN and getattr(controller, "_controller_type", "") == "pose_field_scorer":
                # Keep the control problem seen by the scorer identical to the
                # motion actually executed. Previously assist mode fed a scaled
                # base action to the scorer but still executed the full planner
                # base action, so the residual was correcting a different system
                # than the one the robot followed.
                a_exec[:6] = exec_base_action_world
            if getattr(controller, "_controller_type", "") == "pose_field_scorer":
                if use_outer_rescue:
                    outputs = self._run_pose_field_outer_rescue(
                        controller,
                        gripper_pose=gripper_pose,
                        base_action_local=controller_base_action_local,
                    )
                else:
                    outputs = self._run_pose_field_scorer(
                        controller,
                        front_rgb,
                        wrist_rgb,
                        wrist_depth,
                        ft_hist,
                        proprio,
                        controller_base_action_local,
                        step_idx,
                        gripper_context=gripper_context,
                        current_quat=current_quat,
                        gripper_pose=gripper_pose,
                        exec_base_action_world=exec_base_action_world,
                    )
            else:
                outputs = self._run_controller(
                    controller,
                    wrist_depth,
                    ft_hist,
                    proprio,
                    controller_base_action_local,
                    step_idx,
                    gripper_context=gripper_context,
                )
            if trigger_outputs is not None:
                outputs["ready_to_close"] = trigger_outputs["ready_to_close"]
                if "gripper_logits" in trigger_outputs:
                    outputs["gripper_logits"] = trigger_outputs["gripper_logits"]
                if "hold_after_close" in trigger_outputs:
                    outputs["hold_after_close"] = trigger_outputs["hold_after_close"]
            if self.alignment_v3_shadow_controller is not None:
                current_delta_shadow = getattr(controller, "_runtime_motion_target_delta_local", None)
                if current_delta_shadow is None:
                    current_delta_shadow = getattr(controller, "_runtime_current_delta_basin_target", None)
                self._try_alignment_v3_shadow(
                    controller,
                    front_rgb,
                    wrist_rgb,
                    wrist_depth,
                    ft_hist,
                    proprio,
                    base_action_local,
                    current_delta_shadow,
                    step_idx,
                    gripper_context=gripper_context,
                )
            if self.alignment_v4_shadow_controller is not None:
                current_delta_shadow_v4 = getattr(controller, "_runtime_motion_target_delta_local", None)
                if current_delta_shadow_v4 is None:
                    current_delta_shadow_v4 = getattr(controller, "_runtime_current_delta_basin_target", None)
                self._try_alignment_v4_shadow(
                    controller,
                    front_rgb,
                    wrist_depth,
                    ft_hist,
                    proprio,
                    base_action_local,
                    current_delta_shadow_v4,
                    step_idx,
                    gripper_context=gripper_context,
                )
            pose_key = "delta_pose_gated" if self.use_pose_alpha else "delta_pose"
            raw_delta_pose_local = outputs["delta_pose"].squeeze(0).float().cpu().numpy()
            delta_pose_local = outputs[pose_key].squeeze(0).float().cpu().numpy()
            if self.manager.phase == StagePhase.ALIGN and getattr(controller, "_controller_type", "") == "pose_field_scorer":
                delta_pose_local = self._apply_alignment_temporal_smoothing(outputs, delta_pose_local)
            target_delta_servo_mode = bool(self.enable_target_delta_servo_shadow or self.enable_target_delta_servo_apply)
            self._last_raw_delta_pose_local = np.asarray(raw_delta_pose_local, dtype=np.float32).copy()
            self._last_preclip_delta_pose_local = np.asarray(delta_pose_local, dtype=np.float32).copy()
            self._last_raw_residual_pos_norm = float(np.linalg.norm(raw_delta_pose_local[:3]))
            self._last_raw_residual_yaw_abs = float(abs(raw_delta_pose_local[5])) if raw_delta_pose_local.size >= 6 else 0.0
            self._last_preclip_residual_pos_norm = float(np.linalg.norm(delta_pose_local[:3]))
            self._last_preclip_residual_yaw_abs = float(abs(delta_pose_local[5])) if delta_pose_local.size >= 6 else 0.0
            alpha = float(outputs["alpha"].squeeze(0).float().cpu().item())
            delta_pose_world = np.zeros(6, dtype=np.float32)
            budget_should_charge = False
            if "pred_candidate_index" in outputs:
                pred_idx = int(outputs["pred_candidate_index"].squeeze(0).item())
                pred_group = int(outputs["pred_candidate_group_index"].squeeze(0).item())
                if "selected_step_scale" in outputs:
                    self._last_selected_step_scale = float(outputs["selected_step_scale"].squeeze(0).float().cpu().item())
                if "handoff_yaw_priority_active" in outputs:
                    self._last_handoff_yaw_priority_active = bool(
                        float(outputs["handoff_yaw_priority_active"].squeeze(0).float().cpu().item()) > 0.5
                    )
                if "handoff_xy_priority_active" in outputs:
                    self._last_handoff_xy_priority_active = bool(
                        float(outputs["handoff_xy_priority_active"].squeeze(0).float().cpu().item()) > 0.5
                    )
                self._last_handoff_axis_priority = str(outputs.get("handoff_axis_priority", "none"))
                self._last_scorer_candidate_index = pred_idx
                self._last_scorer_group_index = pred_group
                self._scorer_pred_hist[pred_idx] = self._scorer_pred_hist.get(pred_idx, 0) + 1
                self._scorer_group_hist[pred_group] = self._scorer_group_hist.get(pred_group, 0) + 1
                # v2 shadow stats
                if outputs.get("v2_shadow_active"):
                    self._last_v2_selected_idx = outputs.get("v2_selected_candidate_index")
                    self._last_v2_gate_pass = bool(outputs.get("v2_gate_pass", False))
                    self._last_v2_bucket = str(outputs.get("v2_stage_bucket", "unknown"))
                    self._last_v2_geom_imp = outputs.get("v2_geometry_improvement")
                    self._last_v2_cur_xy = outputs.get("v2_cur_xy")
                    self._last_v2_cur_z = outputs.get("v2_cur_z")
                    self._last_v2_cur_yaw = outputs.get("v2_cur_yaw")
                    self._last_v2_post_delta = outputs.get("v2_post_candidate_delta")
                    self._last_v2_topk = outputs.get("v2_topk_indices")
                    self._last_v2_delta_source = str(outputs.get("v2_delta_source", "none"))
                    self._last_v2_delta_norm = float(outputs.get("v2_delta_norm", 0.0) or 0.0)
                    self._last_v2_gripper_pose_present = bool(outputs.get("v2_gripper_pose_present", False))
                    self._last_v2_mt_pose_present = bool(outputs.get("v2_motion_target_pose_present", False))
                    self._last_v2_mt_delta_present = bool(outputs.get("v2_motion_target_delta_present", False))
                    self._last_v2_cdb_present = bool(outputs.get("v2_current_delta_basin_present", False))
                    self._last_v2_selected_delta = outputs.get("v2_selected_delta")
                    post_delta = outputs.get("v2_post_candidate_delta")
                    sel_idx = self._last_v2_selected_idx if self._last_v2_selected_idx is not None else -1
                    if post_delta is not None and sel_idx is not None and sel_idx >= 0:
                        post_arr = np.asarray(post_delta, dtype=np.float32).reshape(-1)
                        if post_arr.size >= (sel_idx + 1) * 6:
                            post_sel = post_arr.reshape(-1, 6)[sel_idx]
                            self._last_v2_selected_post_xy = float(np.linalg.norm(post_sel[:2]))
                            self._last_v2_selected_post_z = float(abs(post_sel[2]))
                            self._last_v2_selected_post_yaw = float(abs(post_sel[5]))
                        else:
                            self._last_v2_selected_post_xy = None
                            self._last_v2_selected_post_z = None
                            self._last_v2_selected_post_yaw = None
                    else:
                        self._last_v2_selected_post_xy = None
                        self._last_v2_selected_post_z = None
                        self._last_v2_selected_post_yaw = None
                budget_should_charge = pred_idx != 0
            preclip_delta_local = None
            if self.enable_alignment_pose or self.manager.phase != StagePhase.ALIGN:
                if self.manager.phase == StagePhase.ALIGN and getattr(controller, "_controller_type", "") == "pose_field_scorer":
                    delta_pose_local = self._apply_near_handoff_z_correction(delta_pose_local, controller)
                preclip_delta_local = delta_pose_local * self.learned_residual_scale
                self._last_preclip_delta_pose_local = np.asarray(preclip_delta_local, dtype=np.float32).copy()
                clipped_delta_local = self.safety.clip_residual(preclip_delta_local)
                if np.linalg.norm(clipped_delta_local - preclip_delta_local) > 1e-8:
                    self._clip_hit_count += 1
                self._last_clipped_delta_pose_local = np.asarray(clipped_delta_local, dtype=np.float32).copy()
                delta_pose_local = clipped_delta_local
                delta_pose_world = local_delta_to_world(delta_pose_local, current_quat)
                self._last_delta_pose_world = np.asarray(delta_pose_world, dtype=np.float32).copy()
                self._last_clipped_residual_pos_norm = float(np.linalg.norm(delta_pose_local[:3]))
                self._last_clipped_residual_yaw_abs = float(abs(delta_pose_local[5])) if delta_pose_local.size >= 6 else 0.0
                self._last_preclip_residual_pos_norm = float(np.linalg.norm(preclip_delta_local[:3]))
                self._last_preclip_residual_yaw_abs = float(abs(preclip_delta_local[5])) if preclip_delta_local.size >= 6 else 0.0
                if use_predictor_micro_assist_apply:
                    _v2_applied = self._try_v2_predictor_micro_assist(outputs, a_exec, current_quat)
                    if not _v2_applied:
                        pass  # planner passthrough
                elif target_delta_servo_mode:
                    _target_delta_servo_applied = self._try_target_delta_servo(
                        controller,
                        a_exec,
                        current_quat,
                        apply=self.enable_target_delta_servo_apply,
                    )
                    if not _target_delta_servo_applied:
                        pass  # planner passthrough in shadow or blocked servo mode
                elif self.enable_alignment_v4_apply:
                    _v4_applied = self._try_alignment_v4_apply(a_exec, current_quat)
                    if not _v4_applied:
                        pass  # planner passthrough in blocked v4-apply mode
                elif self.enable_alignment_v3_apply:
                    _v3_applied = self._try_alignment_v3_apply(a_exec, current_quat)
                    if not _v3_applied:
                        pass  # planner passthrough in blocked v3-apply mode
                elif alignment_takeover_active:
                    a_exec[:6] = 0.0
                    self._alignment_takeover_count += 1
                    self._last_alignment_takeover_active = True
                    a_exec[:6] = a_exec[:6] + delta_pose_world
                elif self._alignment_zone_state == "planner_only":
                    # Hard passthrough: when the alignment zone says planner_only,
                    # do not add any learned correction. This preserves the
                    # frozen planner's coarse-approach behavior.
                    pass
                elif self._last_nz_blocked_zone:
                    # near-zone blocked: keep planner passthrough in legacy mode too.
                    pass
                else:
                    # legacy assist mode: add final_full delta or replace with legacy v2 assist.
                    _v2_applied = self._try_v2_assist(outputs, a_exec, current_quat)
                    if not _v2_applied:
                        a_exec[:6] = a_exec[:6] + delta_pose_world
            if target_delta_servo_mode:
                executed_alignment_delta = bool(self._last_target_delta_servo_applied)
            elif self.enable_alignment_v4_apply:
                executed_alignment_delta = bool(self._last_alignment_v4_apply_applied)
            elif self.enable_alignment_v3_apply:
                executed_alignment_delta = bool(self._last_alignment_v3_apply_applied)
            else:
                executed_alignment_delta = bool(
                    np.linalg.norm(delta_pose_local[:3]) > 1e-8 or np.linalg.norm(delta_pose_local[3:]) > 1e-8
                )
            if "pred_candidate_index" not in outputs:
                budget_should_charge = executed_alignment_delta
            elif target_delta_servo_mode:
                budget_should_charge = executed_alignment_delta
            elif self.enable_alignment_v4_apply:
                budget_should_charge = executed_alignment_delta
            elif self.enable_alignment_v3_apply:
                budget_should_charge = executed_alignment_delta
            if self.manager.phase == StagePhase.ALIGN:
                self._alignment_decision_count += 1
                if use_outer_rescue:
                    self._outer_rescue_decision_count += 1
                if not budget_should_charge:
                    self._alignment_noop_count += 1
                    if use_outer_rescue:
                        self._outer_rescue_noop_count += 1
                if planner_close_intent_active:
                    self._preclose_alignment_correction_count += 1
                close_gate_controller = (
                    self.alignment_tc_student_vnext_controller
                    if (
                        self.enable_alignment_tc_student_vnext_ready_gate
                        and self.alignment_tc_student_vnext_controller is not None
                    )
                    else self.alignment_controller
                )
                a_exec = self._apply_alignment_close_veto(
                    a_exec,
                    close_gate_controller,
                    gripper_open,
                    step_idx=step_idx,
                    depth_proximity=depth_proximity,
                )
            if budget_should_charge and executed_alignment_delta:
                self._correction_count += 1
                self._alpha_sum += alpha
                self._residual_norm_sum += float(np.linalg.norm(delta_pose_world[:3]))
                self._raw_residual_norm_sum += float(np.linalg.norm(raw_delta_pose_local[:3]))
                self._raw_residual_pos_norm_sum += float(np.linalg.norm(raw_delta_pose_local[:3]))
                if preclip_delta_local is not None:
                    self._preclip_residual_pos_norm_sum += float(np.linalg.norm(preclip_delta_local[:3]))
                self._clipped_residual_pos_norm_sum += float(np.linalg.norm(delta_pose_local[:3]))
                self._executed_residual_norm_sum += float(np.linalg.norm(delta_pose_local[:3]))
                if self.manager.phase == StagePhase.ALIGN:
                    self._alignment_count += 1
                    if use_outer_rescue:
                        self._outer_rescue_correction_count += 1
                        next_delta = outputs.get("next_delta_pose", None)
                        if next_delta is not None and self._delta_within_band(
                            next_delta.squeeze(0).float().cpu().numpy(),
                            controller,
                            band="inner",
                        ):
                            self._outer_rescue_handoff_count += 1
                    elif alignment_takeover_active:
                        self._alignment_window_corrections += 1
                else:
                    self._contact_count += 1
        elif trigger_outputs is not None and self.manager.phase == StagePhase.ALIGN:
            placeholder_outputs = {
                "ready_to_close": trigger_outputs["ready_to_close"],
            }
            if "gripper_logits" in trigger_outputs:
                placeholder_outputs["gripper_logits"] = trigger_outputs["gripper_logits"]
            if "hold_after_close" in trigger_outputs:
                placeholder_outputs["hold_after_close"] = trigger_outputs["hold_after_close"]
            a_exec = self._apply_readiness_gripper(a_exec, placeholder_outputs, self.close_trigger_controller, gripper_open, force_reading=force_reading)
            close_gate_controller = (
                self.alignment_tc_student_vnext_controller
                if (
                    self.enable_alignment_tc_student_vnext_ready_gate
                    and self.alignment_tc_student_vnext_controller is not None
                )
                else self.alignment_controller
            )
            a_exec = self._apply_alignment_close_veto(
                a_exec,
                close_gate_controller,
                gripper_open,
                step_idx=step_idx,
                depth_proximity=depth_proximity,
            )
        elif self.manager.phase == StagePhase.ALIGN:
            self._alignment_zone_state = "planner_only"
            self._alignment_active = False
            self._last_smoothed_delta_local = np.zeros(6, dtype=np.float32)
            self._candidate_hold_remaining = 0
            self._held_candidate_index = -1
            self._held_candidate_delta_local = np.zeros(6, dtype=np.float32)
            close_gate_controller = (
                self.alignment_tc_student_vnext_controller
                if (
                    self.enable_alignment_tc_student_vnext_ready_gate
                    and self.alignment_tc_student_vnext_controller is not None
                )
                else self.alignment_controller
            )
            a_exec = self._apply_alignment_close_veto(
                a_exec,
                close_gate_controller,
                gripper_open,
                step_idx=step_idx,
                depth_proximity=depth_proximity,
            )
        else:
            self._alignment_zone_state = "planner_only"
            self._alignment_active = False
            self._last_smoothed_delta_local = np.zeros(6, dtype=np.float32)
            self._candidate_hold_remaining = 0
            self._held_candidate_index = -1
            self._held_candidate_delta_local = np.zeros(6, dtype=np.float32)

        if (
            self.alignment_diffusion_controller is not None
            and (self.enable_alignment_diffusion_shadow or self.enable_alignment_diffusion_apply)
        ) or (
            self.alignment_tc_diffusion_controller is not None
            and (self.enable_alignment_tc_diffusion_shadow or self.enable_alignment_tc_diffusion_apply)
        ) or (
            self.alignment_tc_student_vnext_controller is not None
            and (self.enable_alignment_tc_student_vnext_shadow or self.enable_alignment_tc_student_vnext_apply)
        ):
            self._try_alignment_diffusion(
                a_exec,
                current_quat,
                front_rgb=front_rgb,
                wrist_rgb=wrist_rgb,
                wrist_depth=wrist_depth,
                ft_hist=ft_hist,
                proprio=proprio,
                base_action_local=base_action_local,
                depth_proximity=depth_proximity,
                force_reading=force_reading,
                gripper_open=gripper_open,
                gripper_pose=gripper_pose,
                gripper_context=self.make_gripper_context(a_base_7d, future_gripper_actions),
            )
        else:
            self._reset_alignment_diffusion_last("disabled")

        a_exec = self._apply_phase1_force_contact_reflex(
            a_exec,
            current_quat=current_quat,
            force_reading=force_reading,
            depth_proximity=depth_proximity,
            gripper_open=gripper_open,
            controller=self._active_alignment_controller(),
        )

        # Final pregrasp close gate. StageManager may briefly transition out of
        # ALIGN when the planner emits a close-like gripper command; do not let
        # that phase edge bypass the same handoff geometry used in ALIGN.
        active_alignment_controller = self._active_alignment_controller()
        if (
            self.manager.phase != StagePhase.ALIGN
            and active_alignment_controller is not None
            and not bool(getattr(self.manager, "has_object_in_hand", False))
            and (
                gripper_open is None
                or float(gripper_open) >= self.alignment_open_threshold
            )
            and float(np.asarray(a_exec, dtype=np.float32)[6]) <= self.alignment_close_command_threshold
            and (
                self._runtime_has_motion_target(active_alignment_controller)
                or self._runtime_has_handoff_geometry(active_alignment_controller)
            )
        ):
            a_exec = self._apply_alignment_close_veto(
                a_exec,
                active_alignment_controller,
                gripper_open,
                step_idx=step_idx,
                depth_proximity=depth_proximity,
            )

        self._last_phase1_post_close_hold_active = False
        if self._close_veto_settle_remaining > 0:
            a_exec = np.asarray(a_exec, dtype=np.float32).copy()
            a_exec[:6] = 0.0
            a_exec[6] = 0.0
            self._close_veto_settle_remaining -= 1

        if self.mode in ("safety_only", "full", "alignment", "contact"):
            reflex_adjust = self.safety.compute_reflex_override(force_reading)
            if reflex_adjust is not None:
                a_exec[:6] = a_exec[:6] + reflex_adjust

        # Do not clip the planner base action here: the planner already learned
        # task-scale deltas, and globally clipping them turns the safety layer into
        # a slow-motion controller. Residual/reflex magnitudes are bounded at their
        # source, while absolute workspace clamp is applied by the runtime before
        # env.step().
        return a_exec

    def should_replan(self) -> bool:
        if self.mode in ("planner_only", "safety_only", "alignment"):
            return False
        return self.manager.should_replan()

    def note_replan(self):
        self._replan_count += 1
        self.manager.note_replan()

    def on_invalid_action(
        self,
        base_action_7d: Optional[np.ndarray] = None,
        force_reading: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Mark invalid execution as RECOVER and return a safe one-step delta."""
        self._replan_count += 1
        self.manager.note_invalid_action()
        self.manager.note_replan()
        self._residual_cooldown = max(self._residual_cooldown, self.invalid_cooldown_steps)
        return self.safety.compute_invalid_action_recovery(base_action_7d, force_reading)

    def get_chunk_size(self) -> int:
        if self.mode in ("planner_only", "safety_only", "alignment"):
            return 8
        return self.manager.get_chunk_size()

    def reset(self):
        self.manager.reset()
        self.safety.reset_counters()
        self._correction_count = 0
        self._replan_count = 0
        self._alpha_sum = 0.0
        self._residual_norm_sum = 0.0
        self._alignment_count = 0
        self._contact_count = 0
        self._residual_cooldown = 0
        self._alignment_gate_block_count = 0
        self._alignment_window_corrections = 0
        self._readiness_eval_count = 0
        self._ready_prob_sum = 0.0
        self._readiness_close_override_count = 0
        self._readiness_open_override_count = 0
        self._readiness_hold_override_count = 0
        self._readiness_heads_missing_count = 0
        self._preclose_alignment_correction_count = 0
        self._ready_positive_count = 0
        self._last_ready_prob = 0.0
        self._last_basin_positive = 0.0
        self._raw_residual_norm_sum = 0.0
        self._executed_residual_norm_sum = 0.0
        self._clip_hit_count = 0
        self._planner_close_intent_count = 0
        self._planner_close_intent_eval_count = 0
        self._scorer_pred_hist = {}
        self._scorer_group_hist = {}
        self._last_scorer_candidate_index = -1
        self._last_scorer_group_index = -1
        self._last_raw_delta_pose_local = None
        self._last_preclip_delta_pose_local = None
        self._last_clipped_delta_pose_local = None
        self._last_delta_pose_world = None
        self._last_raw_residual_pos_norm = 0.0
        self._last_preclip_residual_pos_norm = 0.0
        self._last_clipped_residual_pos_norm = 0.0
        self._last_raw_residual_yaw_abs = 0.0
        self._last_preclip_residual_yaw_abs = 0.0
        self._last_clipped_residual_yaw_abs = 0.0
        self._last_v2_selected_post_xy = None
        self._last_v2_selected_post_z = None
        self._last_v2_selected_post_yaw = None
        self._last_v2_apply_gate_pass = False
        self._last_v2_apply_block_reason = "disabled"
        self._last_v2_apply_assist_scale = 0.0
        self._last_v2_apply_local_delta = None
        self._last_v2_apply_world_delta = None
        self._v2_predictor_micro_assist_eval_count = 0
        self._v2_predictor_micro_assist_apply_count = 0
        self._last_v2_predictor_micro_assist_applied = False
        self._last_v2_predictor_micro_assist_block_reason = "disabled"
        self._last_v2_predictor_micro_assist_pos_norm = 0.0
        self._last_v2_predictor_micro_assist_rot_norm = 0.0
        self._last_v2_predictor_micro_assist_source = "none"
        self._alignment_v3_shadow_eval_count = 0
        self._alignment_v3_shadow_gate_pass_count = 0
        self._alignment_v3_shadow_improve_count = 0
        self._alignment_v3_shadow_all_improve_count = 0
        self._alignment_v3_shadow_apply_like_count = 0
        self._last_alignment_v3_shadow_active = False
        self._last_alignment_v3_shadow_source = "none"
        self._last_alignment_v3_shadow_block_reason = "disabled"
        self._last_alignment_v3_shadow_cur_xy = None
        self._last_alignment_v3_shadow_cur_z = None
        self._last_alignment_v3_shadow_cur_yaw = None
        self._last_alignment_v3_shadow_post_xy = None
        self._last_alignment_v3_shadow_post_z = None
        self._last_alignment_v3_shadow_post_yaw = None
        self._last_alignment_v3_shadow_xy_improved = False
        self._last_alignment_v3_shadow_z_improved = False
        self._last_alignment_v3_shadow_yaw_improved = False
        self._last_alignment_v3_shadow_all_improved = False
        self._last_alignment_v3_shadow_pred_residual_4d = None
        self._last_alignment_v3_shadow_pred_residual_6d = None
        self._last_alignment_v3_shadow_pred_pos_norm = 0.0
        self._last_alignment_v3_shadow_pred_yaw_abs = 0.0
        self._last_alignment_v3_shadow_risk_logit = 0.0
        self._last_alignment_v3_shadow_confidence_logit = 0.0
        self._last_alignment_v3_apply_applied = False
        self._last_alignment_v3_apply_block_reason = "disabled"
        self._last_alignment_v3_apply_local_delta = None
        self._last_alignment_v3_apply_world_delta = None
        self._last_alignment_v3_apply_pos_norm = 0.0
        self._last_alignment_v3_apply_yaw_abs = 0.0
        self._alignment_decision_count = 0
        self._alignment_noop_count = 0
        for key in self._alignment_gate_block_reason_counts:
            self._alignment_gate_block_reason_counts[key] = 0
        self._last_alignment_gate_debug = {
            "gate_open": False,
            "blocked_reason": "none",
            "depth_proximity": None,
            "near_target": False,
            "gripper_open": None,
            "gripper_still_open": False,
            "planner_close_intent": False,
            "close_requirement_satisfied": False,
            "alignment_window_active": False,
            "support_inner_satisfied": True,
            "support_outer_satisfied": True,
            "support_soft_override": False,
            "support_soft_override_reason": "none",
            "use_outer_rescue": False,
            "short_window_available": True,
            "residual_cooldown": 0,
        }
        self._gripper_fsm_state = "block_open"
        self._fire_ready_streak = 0
        self._verify_steps_remaining = 0
        self._verified_contact_count = 0
        self._verify_fail_reopen_count = 0
        self._last_internal_readiness_fire_applied = False
        self._last_internal_readiness_fire_ready = False
        self._last_internal_readiness_band_ready = False
        self._alignment_complete_eval_count = 0
        self._alignment_complete_positive_count = 0
        self._alignment_fire_close_count = 0
        self._last_alignment_complete = False
        self._last_alignment_fire_decision = False
        self._last_alignment_fire_xy = np.nan
        self._last_alignment_fire_z = np.nan
        self._last_alignment_fire_yaw = np.nan
        self._outer_rescue_decision_count = 0
        self._outer_rescue_correction_count = 0
        self._outer_rescue_noop_count = 0
        self._outer_rescue_handoff_count = 0
        self._close_veto_block_count = 0
        self._close_veto_pass_count = 0
        self._close_latch_set_count = 0
        self._close_latch_release_count = 0
        self._close_latch_expire_count = 0
        self._phase1_post_close_hold_remaining = 0
        self._phase1_post_close_hold_count = 0
        self._phase1_post_close_hold_expire_count = 0
        self._last_phase1_post_close_hold_active = False
        self._last_outer_rescue_active = False
        self._last_close_veto_blocked = False
        self._last_close_veto_ready = False
        self._handoff_yaw_priority_count = 0
        self._handoff_xy_priority_count = 0
        self._last_handoff_yaw_priority_active = False
        self._last_handoff_xy_priority_active = False
        self._last_handoff_axis_priority = "none"
        self._final_handoff_polish_count = 0
        self._handoff_xy_micro_stall_streak = 0
        self._handoff_xy_micro_prev_xy = np.inf
        self._handoff_xy_micro_remaining = 0
        self._handoff_xy_micro_trigger_count = 0
        self._close_veto_ready_streak = 0
        self._close_veto_ready_step_idx = -1
        self._close_veto_recent_block_streak = 0
        self._close_latch_remaining = 0
        self._close_veto_settle_remaining = 0
        self._close_veto_runtime_geometry_fallback_eval_count = 0
        self._close_veto_runtime_geometry_fallback_ready_count = 0
        self._last_close_veto_runtime_geometry_fallback_used = False
        self._last_close_veto_runtime_geometry_fallback_ready = False
        self._bounded_auto_close_eval_count = 0
        self._bounded_auto_close_apply_count = 0
        self._bounded_auto_close_ready_streak = 0
        self._last_bounded_auto_close_ready = False
        self._last_bounded_auto_close_applied = False
        self._force_close_after_b2_eval_count = 0
        self._last_force_close_after_b2_applied = False
        self._alignment_takeover_count = 0
        self._last_alignment_takeover_active = False
        self._near_zone_gate_eval_count = 0
        self._near_zone_gate_pass_count = 0
        self._near_zone_gate_block_count = 0
        self._last_near_zone_gate_pass = False
        self._last_near_zone_xy_error = np.nan
        self._last_near_zone_z_error = np.nan
        self._last_near_zone_block_reason = "disabled"
        self._target_delta_servo_eval_count = 0
        self._target_delta_servo_apply_count = 0
        self._last_target_delta_servo_applied = False
        self._last_target_delta_servo_block_reason = "disabled"
        self._last_target_delta_servo_source = "none"
        self._last_target_delta_servo_local_delta = None
        self._last_target_delta_servo_world_delta = None
        self._last_target_delta_servo_pos_norm = 0.0
        self._last_target_delta_servo_rot_norm = 0.0
        self._last_target_delta_servo_cur_xy = None
        self._last_target_delta_servo_cur_z = None
        self._last_target_delta_servo_cur_yaw = None
        self._last_target_delta_servo_post_xy = None
        self._last_target_delta_servo_post_z = None
        self._last_target_delta_servo_post_yaw = None
        self._last_target_delta_servo_gate_pass = False
        self._alignment_v3_shadow_eval_count = 0
        self._alignment_v3_shadow_gate_pass_count = 0
        self._alignment_v3_shadow_improve_count = 0
        self._alignment_v3_shadow_all_improve_count = 0
        self._alignment_v3_shadow_apply_like_count = 0
        self._last_alignment_v3_shadow_active = False
        self._last_alignment_v3_shadow_source = "none"
        self._last_alignment_v3_shadow_block_reason = "disabled"
        self._last_alignment_v3_shadow_cur_xy = None
        self._last_alignment_v3_shadow_cur_z = None
        self._last_alignment_v3_shadow_cur_yaw = None
        self._last_alignment_v3_shadow_post_xy = None
        self._last_alignment_v3_shadow_post_z = None
        self._last_alignment_v3_shadow_post_yaw = None
        self._last_alignment_v3_shadow_xy_improved = False
        self._last_alignment_v3_shadow_z_improved = False
        self._last_alignment_v3_shadow_yaw_improved = False
        self._last_alignment_v3_shadow_all_improved = False
        self._last_alignment_v3_shadow_pred_residual_4d = None
        self._last_alignment_v3_shadow_pred_residual_6d = None
        self._last_alignment_v3_shadow_pred_pos_norm = 0.0
        self._last_alignment_v3_shadow_pred_yaw_abs = 0.0
        self._last_alignment_v3_shadow_risk_logit = 0.0
        self._last_alignment_v3_shadow_confidence_logit = 0.0
        self._last_alignment_v3_apply_applied = False
        self._last_alignment_v3_apply_block_reason = "disabled"
        self._last_alignment_v3_apply_local_delta = None
        self._last_alignment_v3_apply_world_delta = None
        self._last_alignment_v3_apply_pos_norm = 0.0
        self._last_alignment_v3_apply_yaw_abs = 0.0
        self._alignment_v4_apply_count = 0
        self._last_alignment_v4_apply_applied = False
        self._last_alignment_v4_apply_block_reason = "disabled"
        self._last_alignment_v4_apply_local_delta = None
        self._last_alignment_v4_apply_world_delta = None
        self._last_alignment_v4_apply_pos_norm = 0.0
        self._last_alignment_v4_apply_yaw_abs = 0.0
        self._alignment_diffusion_eval_count = 0
        self._alignment_diffusion_active_count = 0
        self._alignment_diffusion_apply_count = 0
        self._alignment_diffusion_block_reason_hist = {}
        self._alignment_diffusion_phase_hist = {}
        self._alignment_diffusion_bucket_hist = {}
        self._alignment_diffusion_controller_type_hist = {}
        self._alignment_diffusion_confidence_sum = 0.0
        self._alignment_diffusion_risk_prob_sum = 0.0
        self._alignment_diffusion_stop_prob_sum = 0.0
        self._alignment_diffusion_scale_down_sum = 0.0
        self._alignment_diffusion_pos_norm_sum = 0.0
        self._alignment_diffusion_yaw_abs_sum = 0.0
        self._alignment_diffusion_pred_target_delta_norm_sum = 0.0
        self._alignment_diffusion_pred_target_yaw_abs_sum = 0.0
        self._alignment_diffusion_target_action_sign_agreement_sum = 0.0
        self._alignment_diffusion_low_confidence_count = 0
        self._alignment_diffusion_soft_clamp_count = 0
        self._alignment_diffusion_hard_reject_count = 0
        self._alignment_diffusion_safety_reject_count = 0
        self._reset_alignment_diffusion_last("disabled")
        self._phase1_close_arbiter_state = "APPROACH"
        self._phase1_close_command_source = "none"
        self._phase1_grasp_contact_confirmed = False
        self._phase1_force_reflex_active = False
        self._phase1_force_reflex_reason = "none"
        self._phase1_force_backoff_applied = False
        self._phase1_reopen_reason = "none"
        self._phase1_close_hold_active = False
        self._phase1_close_hold_remaining = 0
        self._phase1_close_confirm_streak = 0
        self._phase1_close_fail_remaining = 0
        self._phase1_reopen_cooldown_remaining = 0
        self._phase1_force_spike_count = 0
        self._phase1_jam_detected_count = 0
        self._phase1_grasp_contact_confirmed_count = 0
        self._phase1_reopen_count = 0
        self._phase1_force_reflex_activation_count = 0
        self._phase1_force_prev_norm = None
        self._phase1_prev_basin_metrics = None
        self._near_ready_specialist_eval_count = 0
        self._near_ready_specialist_gate_pass_count = 0
        self._near_ready_specialist_use_count = 0
        self._last_near_ready_specialist_gate_open = False
        self._last_near_ready_specialist_active = False
        self._last_near_ready_specialist_cur_xy = np.nan
        self._last_near_ready_specialist_cur_z = np.nan
        self._last_near_ready_specialist_cur_yaw = np.nan
        self._last_near_ready_specialist_gate_xy_max = np.nan
        self._last_near_ready_specialist_gate_z_max = np.nan
        self._last_near_ready_specialist_gate_yaw_max = np.nan

    def get_stats(self) -> dict:
        controller = self._active_alignment_controller()
        stats = self.safety.get_stats()
        stats.update(
            {
                "phase": self.manager.phase.name,
                "phase_id": int(self.manager.phase),
                "substage": self.manager.substage.name,
                "substage_id": int(self.manager.substage),
                "has_object_in_hand": bool(self.manager.has_object_in_hand),
                "contact_state": self.manager.contact_state.name,
                "contact_state_id": int(self.manager.contact_state),
                "stage_target_mode": self.manager.stage_target_mode.name,
                "stage_target_mode_id": int(self.manager.stage_target_mode),
                "phase_age": self.manager.phase_age,
                "no_progress_steps": self.manager.no_progress_steps,
                "steps_since_last_replan": self.manager.steps_since_last_replan,
                "transition_count": self.manager.transition_count,
                "max_phase_reached": self.manager.max_phase_reached,
                "subgoal_progress": self.manager.get_subgoal_progress(),
                "failure_mode": int(self.manager.failure_mode),
                "failure_mode_name": self.manager.get_failure_mode_name(),
                "last_transitioned": self.manager.last_transitioned,
                "correction_count": self._correction_count,
                "alignment_correction_count": self._alignment_count,
                "contact_correction_count": self._contact_count,
                "replan_count": self._replan_count,
                "alpha_mean": self._alpha_sum / max(self._correction_count, 1),
                "residual_pos_norm_mean": self._residual_norm_sum / max(self._correction_count, 1),
                "raw_delta_norm_mean": self._raw_residual_norm_sum / max(self._correction_count, 1),
                "raw_delta_pos_norm_mean": self._raw_residual_pos_norm_sum / max(self._correction_count, 1),
                "preclip_delta_pos_norm_mean": self._preclip_residual_pos_norm_sum / max(self._correction_count, 1),
                "clipped_delta_pos_norm_mean": self._clipped_residual_pos_norm_sum / max(self._correction_count, 1),
                "executed_delta_norm_mean": self._executed_residual_norm_sum / max(self._correction_count, 1),
                "clip_hit_rate": self._clip_hit_count / max(self._correction_count, 1),
                "planner_close_intent_rate": self._planner_close_intent_count / max(self._planner_close_intent_eval_count, 1),
                "scorer_pred_hist": dict(self._scorer_pred_hist),
                "scorer_group_hist": dict(self._scorer_group_hist),
                "last_scorer_candidate_index": self._last_scorer_candidate_index,
                "last_scorer_group_index": self._last_scorer_group_index,
                "last_selected_step_scale": self._last_selected_step_scale,
                "alignment_decision_count": self._alignment_decision_count,
                "alignment_noop_count": self._alignment_noop_count,
                "alignment_low_conf_noop_count": self._alignment_low_conf_noop_count,
                "alignment_physical_mask_count": self._alignment_physical_mask_count,
                "alignment_all_masked_fallback_count": self._alignment_all_masked_fallback_count,
                "handoff_yaw_priority_count": self._handoff_yaw_priority_count,
                "handoff_xy_priority_count": self._handoff_xy_priority_count,
                "last_handoff_yaw_priority_active": self._last_handoff_yaw_priority_active,
                "last_handoff_xy_priority_active": self._last_handoff_xy_priority_active,
                "last_handoff_axis_priority": self._last_handoff_axis_priority,
                "near_handoff_z_correction_count": self._near_handoff_z_correction_count,
                "final_handoff_polish_count": self._final_handoff_polish_count,
                "handoff_xy_micro_trigger_count": self._handoff_xy_micro_trigger_count,
                "handoff_xy_micro_remaining": self._handoff_xy_micro_remaining,
                "handoff_xy_micro_stall_streak": self._handoff_xy_micro_stall_streak,
                "near_ready_rerank_eval_count": self._near_ready_rerank_eval_count,
                "near_ready_rerank_gate_pass_count": self._near_ready_rerank_gate_pass_count,
                "near_ready_rerank_apply_count": self._near_ready_rerank_apply_count,
                "near_ready_rerank_change_count": self._near_ready_rerank_change_count,
                "last_near_ready_rerank_gate_open": self._last_near_ready_rerank_gate_open,
                "last_near_ready_rerank_applied": self._last_near_ready_rerank_applied,
                "last_near_ready_rerank_changed": self._last_near_ready_rerank_changed,
                "last_near_ready_rerank_prev_index": self._last_near_ready_rerank_prev_index,
                "last_near_ready_rerank_new_index": self._last_near_ready_rerank_new_index,
                "last_near_ready_rerank_topk": self._last_near_ready_rerank_topk,
                "last_near_ready_rerank_cur_xy": self._last_near_ready_rerank_cur_xy,
                "last_near_ready_rerank_cur_z": self._last_near_ready_rerank_cur_z,
                "last_near_ready_rerank_cur_yaw": self._last_near_ready_rerank_cur_yaw,
                "last_near_ready_rerank_gate_xy_max": self._last_near_ready_rerank_gate_xy_max,
                "last_near_ready_rerank_gate_z_max": self._last_near_ready_rerank_gate_z_max,
                "last_near_ready_rerank_gate_yaw_max": self._last_near_ready_rerank_gate_yaw_max,
                "near_ready_specialist_eval_count": self._near_ready_specialist_eval_count,
                "near_ready_specialist_gate_pass_count": self._near_ready_specialist_gate_pass_count,
                "near_ready_specialist_use_count": self._near_ready_specialist_use_count,
                "last_near_ready_specialist_gate_open": self._last_near_ready_specialist_gate_open,
                "last_near_ready_specialist_active": self._last_near_ready_specialist_active,
                "last_near_ready_specialist_cur_xy": self._last_near_ready_specialist_cur_xy,
                "last_near_ready_specialist_cur_z": self._last_near_ready_specialist_cur_z,
                "last_near_ready_specialist_cur_yaw": self._last_near_ready_specialist_cur_yaw,
                "last_near_ready_specialist_gate_xy_max": self._last_near_ready_specialist_gate_xy_max,
                "last_near_ready_specialist_gate_z_max": self._last_near_ready_specialist_gate_z_max,
                "last_near_ready_specialist_gate_yaw_max": self._last_near_ready_specialist_gate_yaw_max,
                "near_ready_residual_eval_count": self._near_ready_residual_eval_count,
                "near_ready_residual_gate_pass_count": self._near_ready_residual_gate_pass_count,
                "near_ready_residual_apply_count": self._near_ready_residual_apply_count,
                "near_ready_residual_change_count": self._near_ready_residual_change_count,
                "near_ready_group_residual_eval_count": self._near_ready_group_residual_eval_count,
                "near_ready_group_residual_gate_pass_count": self._near_ready_group_residual_gate_pass_count,
                "near_ready_group_residual_apply_count": self._near_ready_group_residual_apply_count,
                "near_ready_group_residual_change_count": self._near_ready_group_residual_change_count,
                "last_near_ready_group_residual_gate_open": self._last_near_ready_group_residual_gate_open,
                "last_near_ready_group_residual_applied": self._last_near_ready_group_residual_applied,
                "last_near_ready_group_residual_changed": self._last_near_ready_group_residual_changed,
                "last_near_ready_group_residual_prev_group": self._last_near_ready_group_residual_prev_group,
                "last_near_ready_group_residual_new_group": self._last_near_ready_group_residual_new_group,
                "b1_group_shadow_eval_count": self._b1_group_shadow_eval_count,
                "b1_group_shadow_gate_mode": self.b1_group_shadow_gate_mode,
                "b1_group_shadow_gate_pass_count": self._b1_group_shadow_gate_pass_count,
                "b1_group_shadow_change_count": self._b1_group_shadow_change_count,
                "b1_group_shadow_disagreement_count": self._b1_group_shadow_disagreement_count,
                "b1_group_shadow_teacher_group_valid_count": self._b1_group_shadow_teacher_group_valid_count,
                "b1_group_shadow_cost_valid_count": self._b1_group_shadow_cost_valid_count,
                "b1_group_shadow_cost_improve_count": self._b1_group_shadow_cost_improve_count,
                "b1_group_shadow_cost_worse_count": self._b1_group_shadow_cost_worse_count,
                "b1_group_shadow_regret_delta_sum": self._b1_group_shadow_regret_delta_sum,
                "b1_group_shadow_close_count": self._b1_group_shadow_close_count,
                "b1_group_shadow_close_change_count": self._b1_group_shadow_close_change_count,
                "b1_group_shadow_close_cost_valid_count": self._b1_group_shadow_close_cost_valid_count,
                "b1_group_shadow_close_cost_improve_count": self._b1_group_shadow_close_cost_improve_count,
                "b1_group_shadow_close_cost_worse_count": self._b1_group_shadow_close_cost_worse_count,
                "b1_group_shadow_close_regret_delta_sum": self._b1_group_shadow_close_regret_delta_sum,
                "b1_apply_gate_eval_count": self._b1_apply_gate_eval_count,
                "b1_apply_gate_apply_count": self._b1_apply_gate_apply_count,
                "b1_apply_gate_cost_valid_count": self._b1_apply_gate_cost_valid_count,
                "b1_apply_gate_cost_improve_count": self._b1_apply_gate_cost_improve_count,
                "b1_apply_gate_cost_worse_count": self._b1_apply_gate_cost_worse_count,
                "b1_apply_gate_regret_delta_sum": self._b1_apply_gate_regret_delta_sum,
                "b1_apply_gate_close_cost_valid_count": self._b1_apply_gate_close_cost_valid_count,
                "b1_apply_gate_close_cost_improve_count": self._b1_apply_gate_close_cost_improve_count,
                "b1_apply_gate_close_cost_worse_count": self._b1_apply_gate_close_cost_worse_count,
                "b1_apply_gate_close_regret_delta_sum": self._b1_apply_gate_close_regret_delta_sum,
                "b1_group_bounded_apply_count": self._b1_group_bounded_apply_count,
                "b1_group_bounded_change_count": self._b1_group_bounded_change_count,
                "b2_candidate_shadow_eval_count": self._b2_candidate_shadow_eval_count,
                "b2_candidate_shadow_gate_mode": self.b2_candidate_shadow_gate_mode,
                "b2_candidate_shadow_gate_pass_count": self._b2_candidate_shadow_gate_pass_count,
                "b2_candidate_shadow_change_count": self._b2_candidate_shadow_change_count,
                "b2_candidate_shadow_cost_valid_count": self._b2_candidate_shadow_cost_valid_count,
                "b2_candidate_shadow_cost_improve_count": self._b2_candidate_shadow_cost_improve_count,
                "b2_candidate_shadow_cost_worse_count": self._b2_candidate_shadow_cost_worse_count,
                "b2_candidate_shadow_regret_delta_sum": self._b2_candidate_shadow_regret_delta_sum,
                "b2_candidate_shadow_mode_keep_count": self._b2_candidate_shadow_mode_keep_count,
                "b2_candidate_shadow_mode_apply_count": self._b2_candidate_shadow_mode_apply_count,
                "b2_candidate_shadow_close_count": self._b2_candidate_shadow_close_count,
                "b2_candidate_shadow_yaw_needed_count": self._b2_candidate_shadow_yaw_needed_count,
                "b2_candidate_shadow_yaw_keep_count": self._b2_candidate_shadow_yaw_keep_count,
                "b2_candidate_shadow_teacher_ready_count": self._b2_candidate_shadow_teacher_ready_count,
                "b2_candidate_shadow_xy_block_count": self._b2_candidate_shadow_xy_block_count,
                "b2_candidate_shadow_nearish_count": self._b2_candidate_shadow_nearish_count,
                "b2_candidate_shadow_keep_baseline_forced_count": self._b2_candidate_shadow_keep_baseline_forced_count,
                "b2_candidate_bounded_eval_count": self._b2_candidate_bounded_eval_count,
                "b2_candidate_bounded_gate_pass_count": self._b2_candidate_bounded_gate_pass_count,
                "b2_candidate_bounded_apply_count": self._b2_candidate_bounded_apply_count,
                "b2_candidate_bounded_change_count": self._b2_candidate_bounded_change_count,
                "last_b1_group_shadow_gate_open": self._last_b1_group_shadow_gate_open,
                "last_b1_group_shadow_pred_group": self._last_b1_group_shadow_pred_group,
                "last_b1_group_shadow_baseline_group": self._last_b1_group_shadow_baseline_group,
                "last_b1_group_shadow_teacher_group": self._last_b1_group_shadow_teacher_group,
                "last_b1_group_shadow_changed": self._last_b1_group_shadow_changed,
                "last_b1_group_shadow_teacher_disagree": self._last_b1_group_shadow_teacher_disagree,
                "last_b1_group_shadow_teacher_group_valid": self._last_b1_group_shadow_teacher_group_valid,
                "last_b1_group_shadow_close_neighborhood": self._last_b1_group_shadow_close_neighborhood,
                "last_b1_group_shadow_close_group_changed": self._last_b1_group_shadow_close_group_changed,
                "last_b1_group_shadow_margin": self._last_b1_group_shadow_margin,
                "last_b1_group_shadow_teacher_best_cost": self._last_b1_group_shadow_teacher_best_cost,
                "last_b1_group_shadow_baseline_group_cost": self._last_b1_group_shadow_baseline_group_cost,
                "last_b1_group_shadow_pred_group_cost": self._last_b1_group_shadow_pred_group_cost,
                "last_b1_group_shadow_baseline_group_regret": self._last_b1_group_shadow_baseline_group_regret,
                "last_b1_group_shadow_pred_group_regret": self._last_b1_group_shadow_pred_group_regret,
                "last_b1_group_shadow_regret_delta": self._last_b1_group_shadow_regret_delta,
                "last_b1_apply_gate_prob": self._last_b1_apply_gate_prob,
                "last_b1_apply_gate_threshold": self._last_b1_apply_gate_threshold,
                "last_b1_apply_gate_apply": self._last_b1_apply_gate_apply,
                "last_b1_apply_gate_vetoed": self._last_b1_apply_gate_vetoed,
                "last_b1_group_bounded_applied": self._last_b1_group_bounded_applied,
                "last_b1_group_bounded_prev_group": self._last_b1_group_bounded_prev_group,
                "last_b1_group_bounded_new_group": self._last_b1_group_bounded_new_group,
                "last_b2_candidate_shadow_gate_open": self._last_b2_candidate_shadow_gate_open,
                "last_b2_candidate_shadow_close_neighborhood": self._last_b2_candidate_shadow_close_neighborhood,
                "last_b2_candidate_shadow_mode": self._last_b2_candidate_shadow_mode,
                "last_b2_candidate_shadow_mode_confidence": self._last_b2_candidate_shadow_mode_confidence,
                "last_b2_candidate_shadow_mode_margin": self._last_b2_candidate_shadow_mode_margin,
                "last_b2_candidate_shadow_baseline_index": self._last_b2_candidate_shadow_baseline_index,
                "last_b2_candidate_shadow_pred_index": self._last_b2_candidate_shadow_pred_index,
                "last_b2_candidate_shadow_changed": self._last_b2_candidate_shadow_changed,
                "last_b2_candidate_shadow_best_index": self._last_b2_candidate_shadow_best_index,
                "last_b2_candidate_shadow_best_cost": self._last_b2_candidate_shadow_best_cost,
                "last_b2_candidate_shadow_baseline_cost": self._last_b2_candidate_shadow_baseline_cost,
                "last_b2_candidate_shadow_pred_cost": self._last_b2_candidate_shadow_pred_cost,
                "last_b2_candidate_shadow_baseline_regret": self._last_b2_candidate_shadow_baseline_regret,
                "last_b2_candidate_shadow_pred_regret": self._last_b2_candidate_shadow_pred_regret,
                "last_b2_candidate_shadow_regret_delta": self._last_b2_candidate_shadow_regret_delta,
                "last_b2_candidate_shadow_yaw_needed": self._last_b2_candidate_shadow_yaw_needed,
                "last_b2_candidate_shadow_yaw_keep": self._last_b2_candidate_shadow_yaw_keep,
                "last_b2_candidate_shadow_teacher_ready": self._last_b2_candidate_shadow_teacher_ready,
                "last_b2_candidate_shadow_xy_block": self._last_b2_candidate_shadow_xy_block,
                "last_b2_candidate_shadow_runtime_scope_size": self._last_b2_candidate_shadow_runtime_scope_size,
                "last_b2_candidate_shadow_small_yaw_scope_size": self._last_b2_candidate_shadow_small_yaw_scope_size,
                "last_b2_candidate_shadow_large_yaw_scope_size": self._last_b2_candidate_shadow_large_yaw_scope_size,
                "last_b2_candidate_shadow_probe_count": self._last_b2_candidate_shadow_probe_count,
                "last_b2_candidate_shadow_nearish_runtime": self._last_b2_candidate_shadow_nearish_runtime,
                "last_b2_candidate_shadow_keep_baseline_forced": self._last_b2_candidate_shadow_keep_baseline_forced,
                "last_b2_candidate_shadow_candidate_actions_local": self._last_b2_candidate_shadow_candidate_actions_local,
                "last_b2_candidate_shadow_candidate_scope_mask": self._last_b2_candidate_shadow_candidate_scope_mask,
                "last_b2_candidate_shadow_candidate_valid_mask": self._last_b2_candidate_shadow_candidate_valid_mask,
                "last_b2_candidate_shadow_candidate_cost": self._last_b2_candidate_shadow_candidate_cost,
                "last_b2_candidate_shadow_candidate_oracle_score": self._last_b2_candidate_shadow_candidate_oracle_score,
                "last_b2_candidate_bounded_gate_open": self._last_b2_candidate_bounded_gate_open,
                "last_b2_candidate_bounded_applied": self._last_b2_candidate_bounded_applied,
                "last_b2_candidate_bounded_changed": self._last_b2_candidate_bounded_changed,
                "last_b2_candidate_bounded_prev_index": self._last_b2_candidate_bounded_prev_index,
                "last_b2_candidate_bounded_new_index": self._last_b2_candidate_bounded_new_index,
                "last_b2_candidate_bounded_mode_confidence": self._last_b2_candidate_bounded_mode_confidence,
                "last_b2_candidate_bounded_mode_margin": self._last_b2_candidate_bounded_mode_margin,
                "close_intent_shadow_candidate_count": int(self._close_intent_shadow_candidate_count),
                "current_close_intent_shadow_would_auto_close": bool(
                    self._last_close_intent_shadow_would_auto_close
                ),
                "current_close_intent_shadow_reason": str(self._last_close_intent_shadow_reason),
                "current_close_intent_shadow_blocking_axis": str(
                    self._last_close_intent_shadow_blocking_axis
                ),
                "current_close_intent_shadow_confidence": float(self._last_close_intent_shadow_confidence),
                "last_near_ready_residual_gate_open": self._last_near_ready_residual_gate_open,
                "last_near_ready_residual_applied": self._last_near_ready_residual_applied,
                "last_near_ready_residual_changed": self._last_near_ready_residual_changed,
                "last_near_ready_residual_prev_index": self._last_near_ready_residual_prev_index,
                "last_near_ready_residual_new_index": self._last_near_ready_residual_new_index,
                "outer_rescue_decision_count": self._outer_rescue_decision_count,
                "outer_rescue_correction_count": self._outer_rescue_correction_count,
                "outer_rescue_noop_count": self._outer_rescue_noop_count,
                "outer_rescue_handoff_count": self._outer_rescue_handoff_count,
                "residual_cooldown": self._residual_cooldown,
                "learned_residual_scale": self.learned_residual_scale,
                "alignment_gate_block_count": self._alignment_gate_block_count,
                "alignment_gate_block_reason_counts": dict(self._alignment_gate_block_reason_counts),
                "alignment_window_corrections": self._alignment_window_corrections,
                "require_pregrasp_alignment_gate": self.require_pregrasp_alignment_gate,
                "alignment_depth_threshold": self.alignment_depth_threshold,
                "alignment_open_threshold": self.alignment_open_threshold,
                "alignment_close_command_threshold": self.alignment_close_command_threshold,
                "max_alignment_corrections_per_window": self.max_alignment_corrections_per_window,
                "require_close_intent_for_alignment": self.require_close_intent_for_alignment,
                "enable_alignment_pose": self.enable_alignment_pose,
                "use_pose_alpha": self.use_pose_alpha,
                "enable_outer_rescue": self.enable_outer_rescue,
                "outer_rescue_xy_scale": self.outer_rescue_xy_scale,
                "outer_rescue_abs_z_scale": self.outer_rescue_abs_z_scale,
                "outer_rescue_yaw_scale": self.outer_rescue_yaw_scale,
                "outer_rescue_min_xy": self.outer_rescue_min_xy,
                "outer_rescue_min_abs_z": self.outer_rescue_min_abs_z,
                "outer_rescue_min_yaw": self.outer_rescue_min_yaw,
                "enable_alignment_close_veto": self.enable_alignment_close_veto,
                "close_veto_xy_threshold": self.close_veto_xy_threshold,
                "close_veto_abs_z_threshold": self.close_veto_abs_z_threshold,
                "close_veto_yaw_threshold": self.close_veto_yaw_threshold,
                "close_veto_ready_streak_frames": self.close_veto_ready_streak_frames,
                "close_veto_settle_steps": self.close_veto_settle_steps,
                "close_veto_settle_count": self._close_veto_settle_count,
                "close_veto_settle_remaining": self._close_veto_settle_remaining,
                "close_veto_runtime_geometry_fallback_for_bounded": bool(
                    self.close_veto_runtime_geometry_fallback_for_bounded
                ),
                "enable_bounded_auto_close_on_alignment": bool(self.enable_bounded_auto_close_on_alignment),
                "bounded_auto_close_stable_frames": int(self.bounded_auto_close_stable_frames),
                "bounded_auto_close_xy_threshold": float(self.bounded_auto_close_xy_threshold),
                "bounded_auto_close_abs_z_threshold": float(self.bounded_auto_close_abs_z_threshold),
                "bounded_auto_close_yaw_threshold": float(self.bounded_auto_close_yaw_threshold),
                "enable_force_close_after_b2_eval": bool(self.enable_force_close_after_b2_eval),
                "close_latch_enabled": self.close_latch_enabled,
                "close_latch_steps": self.close_latch_steps,
                "alignment_takeover_until_close_ready": self.alignment_takeover_until_close_ready,
                "alignment_assist_xy_scale": self.alignment_assist_xy_scale,
                "alignment_assist_abs_z_scale": self.alignment_assist_abs_z_scale,
                "alignment_assist_yaw_scale": self.alignment_assist_yaw_scale,
                "alignment_takeover_xy_scale": self.alignment_takeover_xy_scale,
                "alignment_takeover_abs_z_scale": self.alignment_takeover_abs_z_scale,
                "alignment_takeover_yaw_scale": self.alignment_takeover_yaw_scale,
                "alignment_zone_hysteresis": self.alignment_zone_hysteresis,
                "alignment_candidate_hold_steps": self.alignment_candidate_hold_steps,
                "alignment_candidate_switch_margin": self.alignment_candidate_switch_margin,
                "alignment_action_lowpass_alpha": self.alignment_action_lowpass_alpha,
                "enable_alignment_physical_mask": self.enable_alignment_physical_mask,
                "enable_alignment_low_conf_noop": self.enable_alignment_low_conf_noop,
                "alignment_takeover_count": self._alignment_takeover_count,
                "close_latch_set_count": self._close_latch_set_count,
                "close_latch_release_count": self._close_latch_release_count,
                "close_latch_expire_count": self._close_latch_expire_count,
                "close_latch_remaining": self._close_latch_remaining,
                "skip_alignment_when_close_ready": self.skip_alignment_when_close_ready,
                "skip_alignment_ready_xy_threshold": self.skip_alignment_ready_xy_threshold,
                "skip_alignment_ready_abs_z_threshold": self.skip_alignment_ready_abs_z_threshold,
                "skip_alignment_ready_yaw_threshold": self.skip_alignment_ready_yaw_threshold,
                "enable_readiness_gripper": self.enable_readiness_gripper,
                "readiness_close_threshold": self.readiness_close_threshold,
                "gripper_override_confidence": self.gripper_override_confidence,
                "readiness_eval_count": self._readiness_eval_count,
                "ready_positive_rate": self._ready_positive_count / max(self._readiness_eval_count, 1),
                "ready_to_close_prob_mean": self._ready_prob_sum / max(self._readiness_eval_count, 1),
                "basin_positive_rate": self._ready_positive_count / max(self._readiness_eval_count, 1),
                "trigger_prob_mean": self._ready_prob_sum / max(self._readiness_eval_count, 1),
                "current_trigger_prob": self._last_ready_prob,
                "current_basin_positive": self._last_basin_positive,
                "current_alignment_gate_open": bool(self._last_alignment_gate_debug["gate_open"]),
                "current_alignment_blocked_reason": self._last_alignment_gate_debug["blocked_reason"],
                "current_alignment_depth_proximity": self._last_alignment_gate_debug["depth_proximity"],
                "current_alignment_near_target": bool(self._last_alignment_gate_debug["near_target"]),
                "current_alignment_gripper_open": self._last_alignment_gate_debug["gripper_open"],
                "current_alignment_gripper_still_open": bool(self._last_alignment_gate_debug["gripper_still_open"]),
                "current_alignment_planner_close_intent": bool(self._last_alignment_gate_debug["planner_close_intent"]),
                "current_alignment_close_requirement_satisfied": bool(
                    self._last_alignment_gate_debug["close_requirement_satisfied"]
                ),
                "current_alignment_window_active": bool(self._last_alignment_gate_debug["alignment_window_active"]),
                "current_alignment_support_satisfied": bool(self._last_alignment_gate_debug.get("support_satisfied", True)),
                "current_alignment_support_inner_satisfied": bool(self._last_alignment_gate_debug.get("support_inner_satisfied", True)),
                "current_alignment_support_outer_satisfied": bool(self._last_alignment_gate_debug.get("support_outer_satisfied", True)),
                "current_alignment_support_soft_override": bool(
                    self._last_alignment_gate_debug.get("support_soft_override", False)
                ),
                "current_alignment_support_soft_override_reason": str(
                    self._last_alignment_gate_debug.get("support_soft_override_reason", "none")
                ),
                "current_alignment_use_outer_rescue": bool(self._last_alignment_gate_debug.get("use_outer_rescue", False)),
                "current_alignment_refine_band_satisfied": bool(self._last_alignment_gate_debug.get("refine_band_satisfied", False)),
                "current_alignment_takeover_band_satisfied": bool(self._last_alignment_gate_debug.get("takeover_band_satisfied", False)),
                "current_alignment_short_window_available": bool(
                    self._last_alignment_gate_debug["short_window_available"]
                ),
                "close_veto_block_count": self._close_veto_block_count,
                "close_veto_pass_count": self._close_veto_pass_count,
                "current_close_veto_blocked": bool(self._last_close_veto_blocked),
                "current_close_veto_ready": bool(self._last_close_veto_ready),
                "current_close_veto_ready_streak": int(self._close_veto_ready_streak),
                "current_close_veto_recent_block_streak": int(self._close_veto_recent_block_streak),
                "current_close_latch_remaining": int(self._close_latch_remaining),
                "current_close_veto_settle_remaining": int(self._close_veto_settle_remaining),
                "current_phase1_post_close_hold_remaining": int(self._phase1_post_close_hold_remaining),
                "current_phase1_post_close_hold_active": bool(self._last_phase1_post_close_hold_active),
                "current_student_ready_gate_active": bool(
                    getattr(controller, "_runtime_student_ready_gate_active", False)
                ),
                "current_student_close_ready_logit": float(
                    getattr(controller, "_runtime_close_ready_logit", np.nan)
                ),
                "current_student_close_ready_prob": float(
                    getattr(controller, "_runtime_close_ready_prob", np.nan)
                ),
                "current_student_close_ready_pred": bool(
                    getattr(controller, "_runtime_close_ready_pred", False)
                ),
                "current_student_close_ready": bool(
                    getattr(controller, "_runtime_close_ready", False)
                ),
                "current_student_close_ready_applied": bool(
                    getattr(controller, "_runtime_close_ready_applied", False)
                ),
                "current_student_handoff_ready_logit": float(
                    getattr(controller, "_runtime_handoff_ready_logit", np.nan)
                ),
                "current_student_handoff_ready_prob": float(
                    getattr(controller, "_runtime_handoff_ready_prob", np.nan)
                ),
                "current_student_handoff_ready_pred": bool(
                    getattr(controller, "_runtime_handoff_ready_pred", False)
                ),
                "current_student_handoff_ready": bool(
                    getattr(controller, "_runtime_handoff_ready", False)
                ),
                "current_student_handoff_ready_applied": bool(
                    getattr(controller, "_runtime_handoff_ready_applied", False)
                ),
                "close_state_machine": dict(self._last_close_state_machine),
                "close_state": str(self._last_close_state_machine.get("state", "unknown")),
                "close_action_decision": str(self._last_close_state_machine.get("action_decision", "none")),
                "close_blocked_reason": str(self._last_close_state_machine.get("blocked_reason", "none")),
                "close_runtime_geometry_ready": bool(
                    self._last_close_state_machine.get("runtime_geometry_ready", False)
                ),
                "close_handoff_ready_pred": bool(self._last_close_state_machine.get("handoff_ready_pred", False)),
                "close_handoff_ready_applied": bool(
                    self._last_close_state_machine.get("handoff_ready_applied", False)
                ),
                "close_handoff_shadow_blocks_apply": bool(
                    self._last_close_state_machine.get("handoff_shadow_blocks_apply", False)
                ),
                "close_fallback_enabled": bool(self._last_close_state_machine.get("fallback_enabled", False)),
                "close_fallback_used": bool(self._last_close_state_machine.get("fallback_used", False)),
                "close_veto_runtime_geometry_fallback_eval_count": int(
                    self._close_veto_runtime_geometry_fallback_eval_count
                ),
                "close_veto_runtime_geometry_fallback_ready_count": int(
                    self._close_veto_runtime_geometry_fallback_ready_count
                ),
                "current_close_veto_runtime_geometry_fallback_used": bool(
                    self._last_close_veto_runtime_geometry_fallback_used
                ),
                "current_close_veto_runtime_geometry_fallback_ready": bool(
                    self._last_close_veto_runtime_geometry_fallback_ready
                ),
                "phase1_post_close_hold_count": int(self._phase1_post_close_hold_count),
                "phase1_post_close_hold_expire_count": int(self._phase1_post_close_hold_expire_count),
                "bounded_auto_close_eval_count": int(self._bounded_auto_close_eval_count),
                "bounded_auto_close_apply_count": int(self._bounded_auto_close_apply_count),
                "current_bounded_auto_close_ready_streak": int(self._bounded_auto_close_ready_streak),
                "current_bounded_auto_close_ready": bool(self._last_bounded_auto_close_ready),
                "current_bounded_auto_close_applied": bool(self._last_bounded_auto_close_applied),
                "force_close_after_b2_eval_count": int(self._force_close_after_b2_eval_count),
                "current_force_close_after_b2_applied": bool(self._last_force_close_after_b2_applied),
                "current_alignment_takeover_active": bool(self._last_alignment_takeover_active),
                "phase1_force_reflex_enabled": bool(self.enable_phase1_force_reflex),
                "phase1_close_arbiter_state": str(self._phase1_close_arbiter_state),
                "phase1_close_command_source": str(self._phase1_close_command_source),
                "phase1_grasp_contact_confirmed": bool(self._phase1_grasp_contact_confirmed),
                "phase1_force_reflex_active": bool(self._phase1_force_reflex_active),
                "phase1_force_reflex_reason": str(self._phase1_force_reflex_reason),
                "phase1_force_backoff_applied": bool(self._phase1_force_backoff_applied),
                "phase1_reopen_reason": str(self._phase1_reopen_reason),
                "phase1_close_hold_active": bool(self._phase1_close_hold_active),
                "phase1_close_hold_remaining": int(self._phase1_close_hold_remaining),
                "phase1_force_spike_count": int(self._phase1_force_spike_count),
                "phase1_jam_detected_count": int(self._phase1_jam_detected_count),
                "phase1_grasp_contact_confirmed_count": int(self._phase1_grasp_contact_confirmed_count),
                "phase1_reopen_count": int(self._phase1_reopen_count),
                "phase1_force_reflex_activation_count": int(self._phase1_force_reflex_activation_count),
                # target-delta servo fields
                "current_target_delta_servo_enabled": bool(self.enable_target_delta_servo_shadow or self.enable_target_delta_servo_apply),
                "current_target_delta_servo_shadow_enabled": bool(self.enable_target_delta_servo_shadow),
                "current_target_delta_servo_apply_enabled": bool(self.enable_target_delta_servo_apply),
                "current_target_delta_servo_bypass_gates": bool(self.target_delta_servo_bypass_gates),
                "current_target_delta_servo_apply_once_per_episode": bool(
                    self.target_delta_servo_apply_once_per_episode
                ),
                "current_target_delta_servo_source_mode": str(self.target_delta_servo_source),
                "current_target_delta_servo_k_xy": float(self.target_delta_servo_k_xy),
                "current_target_delta_servo_k_z": float(self.target_delta_servo_k_z),
                "current_target_delta_servo_k_yaw": float(self.target_delta_servo_k_yaw),
                "current_target_delta_servo_max_pos": float(self.target_delta_servo_max_pos),
                "current_target_delta_servo_max_yaw": float(self.target_delta_servo_max_yaw),
                "current_target_delta_servo_eval_count": int(self._target_delta_servo_eval_count),
                "current_target_delta_servo_apply_count": int(self._target_delta_servo_apply_count),
                "current_target_delta_servo_applied": bool(self._last_target_delta_servo_applied),
                "current_target_delta_servo_block_reason": str(self._last_target_delta_servo_block_reason),
                "current_target_delta_servo_source": str(self._last_target_delta_servo_source),
                "current_target_delta_servo_local_delta": getattr(self, "_last_target_delta_servo_local_delta", None),
                "current_target_delta_servo_world_delta": getattr(self, "_last_target_delta_servo_world_delta", None),
                "current_target_delta_servo_pos_norm": float(self._last_target_delta_servo_pos_norm),
                "current_target_delta_servo_rot_norm": float(self._last_target_delta_servo_rot_norm),
                "current_target_delta_servo_cur_xy": getattr(self, "_last_target_delta_servo_cur_xy", None),
                "current_target_delta_servo_cur_z": getattr(self, "_last_target_delta_servo_cur_z", None),
                "current_target_delta_servo_cur_yaw": getattr(self, "_last_target_delta_servo_cur_yaw", None),
                "current_target_delta_servo_post_xy": getattr(self, "_last_target_delta_servo_post_xy", None),
                "current_target_delta_servo_post_z": getattr(self, "_last_target_delta_servo_post_z", None),
                "current_target_delta_servo_post_yaw": getattr(self, "_last_target_delta_servo_post_yaw", None),
                "current_target_delta_servo_gate_pass": bool(self._last_target_delta_servo_gate_pass),
                # v3 direct-local shadow fields
                "current_alignment_v3_shadow_enabled": bool(self.alignment_v3_shadow_controller is not None),
                "current_alignment_v3_shadow_eval_count": int(self._alignment_v3_shadow_eval_count),
                "current_alignment_v3_shadow_gate_pass_count": int(self._alignment_v3_shadow_gate_pass_count),
                "current_alignment_v3_shadow_gate_pass": bool(
                    self._last_alignment_v3_shadow_active
                    and self._last_alignment_v3_shadow_cur_xy is not None
                    and self._last_alignment_v3_shadow_cur_z is not None
                    and self._last_alignment_v3_shadow_cur_xy <= self.alignment_near_zone_xy_threshold
                    and self._last_alignment_v3_shadow_cur_z <= self.alignment_near_zone_z_threshold
                ),
                "current_alignment_v3_shadow_improve_count": int(self._alignment_v3_shadow_improve_count),
                "current_alignment_v3_shadow_all_improve_count": int(self._alignment_v3_shadow_all_improve_count),
                "current_alignment_v3_shadow_active": bool(self._last_alignment_v3_shadow_active),
                "current_alignment_v3_shadow_source": str(self._last_alignment_v3_shadow_source),
                "current_alignment_v3_shadow_block_reason": str(self._last_alignment_v3_shadow_block_reason),
                "current_alignment_v3_shadow_cur_xy": getattr(self, "_last_alignment_v3_shadow_cur_xy", None),
                "current_alignment_v3_shadow_cur_z": getattr(self, "_last_alignment_v3_shadow_cur_z", None),
                "current_alignment_v3_shadow_cur_yaw": getattr(self, "_last_alignment_v3_shadow_cur_yaw", None),
                "current_alignment_v3_shadow_post_xy": getattr(self, "_last_alignment_v3_shadow_post_xy", None),
                "current_alignment_v3_shadow_post_z": getattr(self, "_last_alignment_v3_shadow_post_z", None),
                "current_alignment_v3_shadow_post_yaw": getattr(self, "_last_alignment_v3_shadow_post_yaw", None),
                "current_alignment_v3_shadow_xy_improved": bool(self._last_alignment_v3_shadow_xy_improved),
                "current_alignment_v3_shadow_z_improved": bool(self._last_alignment_v3_shadow_z_improved),
                "current_alignment_v3_shadow_yaw_improved": bool(self._last_alignment_v3_shadow_yaw_improved),
                "current_alignment_v3_shadow_all_improved": bool(self._last_alignment_v3_shadow_all_improved),
                "current_alignment_v3_shadow_pred_residual_4d": getattr(self, "_last_alignment_v3_shadow_pred_residual_4d", None),
                "current_alignment_v3_shadow_pred_residual_6d": getattr(self, "_last_alignment_v3_shadow_pred_residual_6d", None),
                "current_alignment_v3_shadow_pred_pos_norm": float(self._last_alignment_v3_shadow_pred_pos_norm),
                "current_alignment_v3_shadow_pred_yaw_abs": float(self._last_alignment_v3_shadow_pred_yaw_abs),
                "current_alignment_v3_shadow_risk_logit": float(self._last_alignment_v3_shadow_risk_logit),
                "current_alignment_v3_shadow_confidence_logit": float(self._last_alignment_v3_shadow_confidence_logit),
                "current_alignment_v4_shadow_enabled": bool(self.alignment_v4_shadow_controller is not None),
                "current_alignment_v4_shadow_eval_count": int(self._alignment_v4_shadow_eval_count),
                "current_alignment_v4_shadow_active": bool(self._last_alignment_v4_shadow_active),
                "current_alignment_v4_shadow_source": str(self._last_alignment_v4_shadow_source),
                "current_alignment_v4_shadow_block_reason": str(self._last_alignment_v4_shadow_block_reason),
                "current_alignment_v4_shadow_cur_xy": getattr(self, "_last_alignment_v4_shadow_cur_xy", None),
                "current_alignment_v4_shadow_cur_z": getattr(self, "_last_alignment_v4_shadow_cur_z", None),
                "current_alignment_v4_shadow_cur_yaw": getattr(self, "_last_alignment_v4_shadow_cur_yaw", None),
                "current_alignment_v4_shadow_post_xy": getattr(self, "_last_alignment_v4_shadow_post_xy", None),
                "current_alignment_v4_shadow_post_z": getattr(self, "_last_alignment_v4_shadow_post_z", None),
                "current_alignment_v4_shadow_post_yaw": getattr(self, "_last_alignment_v4_shadow_post_yaw", None),
                "current_alignment_v4_shadow_xy_improved": bool(self._last_alignment_v4_shadow_xy_improved),
                "current_alignment_v4_shadow_z_improved": bool(self._last_alignment_v4_shadow_z_improved),
                "current_alignment_v4_shadow_yaw_improved": bool(self._last_alignment_v4_shadow_yaw_improved),
                "current_alignment_v4_shadow_all_improved": bool(self._last_alignment_v4_shadow_all_improved),
                "current_alignment_v4_shadow_pred_residual_4d": getattr(self, "_last_alignment_v4_shadow_pred_residual_4d", None),
                "current_alignment_v4_shadow_pred_residual_6d": getattr(self, "_last_alignment_v4_shadow_pred_residual_6d", None),
                "current_alignment_v4_shadow_pred_post_xyz_yaw": getattr(self, "_last_alignment_v4_shadow_pred_post_xyz_yaw", None),
                "current_alignment_v4_shadow_pred_reduction_xyz": getattr(self, "_last_alignment_v4_shadow_pred_reduction_xyz", None),
                "current_alignment_v4_shadow_pred_pos_norm": float(self._last_alignment_v4_shadow_pred_pos_norm),
                "current_alignment_v4_shadow_pred_yaw_abs": float(self._last_alignment_v4_shadow_pred_yaw_abs),
                "current_alignment_v4_shadow_risk_logit": float(self._last_alignment_v4_shadow_risk_logit),
                "current_alignment_v4_shadow_confidence_logit": float(self._last_alignment_v4_shadow_confidence_logit),
                "current_alignment_v4_shadow_policy_mode": str(self._last_alignment_v4_shadow_policy_mode),
                "current_alignment_v4_shadow_stage_bucket": str(self._last_alignment_v4_shadow_stage_bucket),
                "current_alignment_v4_shadow_micro_gate": bool(self._last_alignment_v4_shadow_micro_gate),
                "current_alignment_v4_shadow_micro_only": bool(self.alignment_v4_shadow_micro_only),
                "current_alignment_v4_apply_enabled": bool(self.enable_alignment_v4_apply),
                "current_alignment_v4_apply_count": int(self._alignment_v4_apply_count),
                "current_alignment_v4_apply_applied": bool(self._last_alignment_v4_apply_applied),
                "current_alignment_v4_apply_block_reason": str(self._last_alignment_v4_apply_block_reason),
                "current_alignment_v4_apply_local_delta": getattr(self, "_last_alignment_v4_apply_local_delta", None),
                "current_alignment_v4_apply_world_delta": getattr(self, "_last_alignment_v4_apply_world_delta", None),
                "current_alignment_v4_apply_pos_norm": float(self._last_alignment_v4_apply_pos_norm),
                "current_alignment_v4_apply_yaw_abs": float(self._last_alignment_v4_apply_yaw_abs),
                "current_alignment_v4_apply_stage_bucket": str(self._last_alignment_v4_apply_stage_bucket),
                "current_alignment_v4_apply_micro_gate": bool(self._last_alignment_v4_apply_micro_gate),
                "current_alignment_v4_apply_micro_only": bool(self.alignment_v4_apply_micro_only),
                "current_alignment_diffusion_enabled": bool(self._last_alignment_diffusion_enabled),
                "current_alignment_diffusion_shadow_enabled": bool(self.enable_alignment_diffusion_shadow),
                "current_alignment_diffusion_apply_enabled": bool(self.enable_alignment_diffusion_apply),
                "current_alignment_tc_diffusion_shadow_enabled": bool(self.enable_alignment_tc_diffusion_shadow),
                "current_alignment_tc_diffusion_apply_enabled": bool(self.enable_alignment_tc_diffusion_apply),
                "current_alignment_tc_student_vnext_shadow_enabled": bool(self.enable_alignment_tc_student_vnext_shadow),
                "current_alignment_tc_student_vnext_apply_enabled": bool(self.enable_alignment_tc_student_vnext_apply),
                "current_alignment_diffusion_controller_type": str(self._last_alignment_diffusion_controller_type),
                "current_alignment_diffusion_eval_count": int(self._alignment_diffusion_eval_count),
                "current_alignment_diffusion_active_count": int(self._alignment_diffusion_active_count),
                "current_alignment_diffusion_apply_count": int(self._alignment_diffusion_apply_count),
                "alignment_diffusion_block_reason_hist": dict(self._alignment_diffusion_block_reason_hist),
                "alignment_diffusion_phase_hist": dict(self._alignment_diffusion_phase_hist),
                "alignment_diffusion_bucket_hist": dict(self._alignment_diffusion_bucket_hist),
                "alignment_diffusion_controller_type_hist": dict(self._alignment_diffusion_controller_type_hist),
                "alignment_diffusion_confidence_mean": float(
                    self._alignment_diffusion_confidence_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_risk_prob_mean": float(
                    self._alignment_diffusion_risk_prob_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_stop_prob_mean": float(
                    self._alignment_diffusion_stop_prob_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_scale_down_mean": float(
                    self._alignment_diffusion_scale_down_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_pos_norm_mean": float(
                    self._alignment_diffusion_pos_norm_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_yaw_abs_mean": float(
                    self._alignment_diffusion_yaw_abs_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_pred_target_delta_norm_mean": float(
                    self._alignment_diffusion_pred_target_delta_norm_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_pred_target_yaw_abs_mean": float(
                    self._alignment_diffusion_pred_target_yaw_abs_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_target_action_sign_agreement_mean": float(
                    self._alignment_diffusion_target_action_sign_agreement_sum / max(self._alignment_diffusion_active_count, 1)
                ),
                "alignment_diffusion_low_confidence_count": int(self._alignment_diffusion_low_confidence_count),
                "alignment_diffusion_soft_clamp_count": int(self._alignment_diffusion_soft_clamp_count),
                "alignment_diffusion_hard_reject_count": int(self._alignment_diffusion_hard_reject_count),
                "alignment_diffusion_safety_reject_count": int(self._alignment_diffusion_safety_reject_count),
                "current_alignment_diffusion_active": bool(self._last_alignment_diffusion_active),
                "current_alignment_diffusion_applied": bool(self._last_alignment_diffusion_applied),
                "current_alignment_diffusion_block_reason": str(self._last_alignment_diffusion_block_reason),
                "current_alignment_diffusion_phase_name": str(self._last_alignment_diffusion_phase_name),
                "current_alignment_diffusion_trigger_mode": str(self._last_alignment_diffusion_trigger_mode),
                "current_alignment_diffusion_stage_bucket": str(self._last_alignment_diffusion_stage_bucket),
                "current_alignment_diffusion_safety_reject": bool(self._last_alignment_diffusion_safety_reject),
                "current_alignment_diffusion_selected_index": int(self._last_alignment_diffusion_selected_index),
                "current_alignment_diffusion_num_samples": int(self._last_alignment_diffusion_num_samples),
                "current_alignment_diffusion_candidate_diversity": float(
                    self._last_alignment_diffusion_candidate_diversity
                ),
                "current_alignment_diffusion_candidate_score": float(self._last_alignment_diffusion_candidate_score),
                "current_alignment_diffusion_risk_prob": float(self._last_alignment_diffusion_risk_prob),
                "current_alignment_diffusion_stop_prob": float(self._last_alignment_diffusion_stop_prob),
                "current_alignment_diffusion_target_confidence": float(
                    self._last_alignment_diffusion_target_confidence
                ),
                "current_alignment_diffusion_pred_target_delta_6d": getattr(
                    self, "_last_alignment_diffusion_pred_target_delta_6d", None
                ),
                "current_alignment_diffusion_pred_target_delta_norm": float(
                    self._last_alignment_diffusion_pred_target_delta_norm
                ),
                "current_alignment_diffusion_pred_target_yaw_abs": float(
                    self._last_alignment_diffusion_pred_target_yaw_abs
                ),
                "current_alignment_diffusion_target_action_sign_agreement": float(
                    self._last_alignment_diffusion_target_action_sign_agreement
                ),
                "current_alignment_diffusion_low_confidence": bool(self._last_alignment_diffusion_low_confidence),
                "current_alignment_diffusion_soft_clamp": bool(self._last_alignment_diffusion_soft_clamp),
                "current_alignment_diffusion_scale_down": float(self._last_alignment_diffusion_scale_down),
                "current_alignment_diffusion_hard_reject": bool(self._last_alignment_diffusion_hard_reject),
                "current_alignment_diffusion_phase1_bridge_soft_apply": bool(
                    self._last_alignment_diffusion_phase1_bridge_soft_apply
                ),
                "current_alignment_diffusion_phase1_bridge_soft_apply_reason": str(
                    self._last_alignment_diffusion_phase1_bridge_soft_apply_reason
                ),
                "current_alignment_diffusion_latency_total_ms": float(
                    self._last_alignment_diffusion_latency_total_ms
                ),
                "current_alignment_diffusion_top_k": int(self._last_alignment_diffusion_top_k),
                "current_alignment_diffusion_progress_logits": getattr(
                    self, "_last_alignment_diffusion_progress_logits", None
                ),
                "current_alignment_diffusion_first_residual_4d": getattr(
                    self, "_last_alignment_diffusion_first_residual_4d", None
                ),
                "current_alignment_diffusion_first_residual_6d": getattr(
                    self, "_last_alignment_diffusion_first_residual_6d", None
                ),
                "current_alignment_diffusion_local_delta": getattr(
                    self, "_last_alignment_diffusion_local_delta", None
                ),
                "current_alignment_diffusion_world_delta": getattr(
                    self, "_last_alignment_diffusion_world_delta", None
                ),
                "current_alignment_diffusion_pos_norm": float(self._last_alignment_diffusion_pos_norm),
                "current_alignment_diffusion_yaw_abs": float(self._last_alignment_diffusion_yaw_abs),
                "current_alignment_diffusion_workspace_violation": float(
                    self._last_alignment_diffusion_workspace_violation
                ),
                "current_alignment_diffusion_workspace_projected": bool(
                    self._last_alignment_diffusion_workspace_projected
                ),
                "current_alignment_diffusion_workspace_project_reason": str(
                    self._last_alignment_diffusion_workspace_project_reason
                ),
                "current_alignment_diffusion_phase1_bridge_blend_active": bool(
                    self._last_alignment_diffusion_phase1_bridge_blend_active
                ),
                "current_alignment_diffusion_phase1_bridge_blend_reason": str(
                    self._last_alignment_diffusion_phase1_bridge_blend_reason
                ),
                "current_alignment_diffusion_phase1_bridge_blended_base_world_6d": getattr(
                    self, "_last_alignment_diffusion_phase1_bridge_blended_base_world", None
                ),
                "current_alignment_diffusion_phase1_bridge_blended_base_local_6d": getattr(
                    self, "_last_alignment_diffusion_phase1_bridge_blended_base_local", None
                ),
                "current_alignment_diffusion_phase1_bridge_basin_bias_local_6d": getattr(
                    self, "_last_alignment_diffusion_phase1_bridge_basin_bias_local", None
                ),
                "current_alignment_diffusion_phase1_bridge_taskspace_yaw_error": float(
                    self._last_alignment_diffusion_phase1_bridge_taskspace_yaw_error
                ),
                "current_alignment_diffusion_phase1_bridge_taskspace_yaw_target_source": str(
                    getattr(
                        self,
                        "_last_alignment_diffusion_phase1_bridge_taskspace_yaw_target_source",
                        "none",
                    )
                ),
                "current_alignment_diffusion_phase1_bridge_direct_yaw_rescue_local_6d": getattr(
                    self, "_last_alignment_diffusion_phase1_bridge_direct_yaw_rescue_local", None
                ),
                "current_alignment_diffusion_phase1_bridge_yaw_holdoff_active": bool(
                    self._last_alignment_diffusion_phase1_bridge_yaw_holdoff_active
                ),
                "current_alignment_diffusion_phase1_bridge_yaw_holdoff_reason": str(
                    self._last_alignment_diffusion_phase1_bridge_yaw_holdoff_reason
                ),
                "current_alignment_v3_apply_enabled": bool(self.enable_alignment_v3_apply),
                "current_alignment_v3_apply_scale": float(self.alignment_v3_apply_scale),
                "current_alignment_v3_apply_max_pos": float(self.alignment_v3_apply_max_pos),
                "current_alignment_v3_apply_max_yaw": float(self.alignment_v3_apply_max_yaw),
                "current_alignment_v3_apply_only_when_gate_pass": bool(
                    self.alignment_v3_apply_only_when_gate_pass
                ),
                "current_alignment_v3_apply_require_improve": bool(self.alignment_v3_apply_require_improve),
                "current_alignment_v3_apply_count": int(self._alignment_v3_shadow_apply_like_count),
                "current_alignment_v3_apply_applied": bool(self._last_alignment_v3_apply_applied),
                "current_alignment_v3_apply_block_reason": str(self._last_alignment_v3_apply_block_reason),
                "current_alignment_v3_apply_local_delta": getattr(
                    self, "_last_alignment_v3_apply_local_delta", None
                ),
                "current_alignment_v3_apply_world_delta": getattr(
                    self, "_last_alignment_v3_apply_world_delta", None
                ),
                "current_alignment_v3_apply_pos_norm": float(self._last_alignment_v3_apply_pos_norm),
                "current_alignment_v3_apply_yaw_abs": float(self._last_alignment_v3_apply_yaw_abs),
                # v2 shadow fields
                "current_v2_shadow_active": bool(getattr(self, "_v2_shadow", None) is not None),
                "current_v2_selected_candidate_index": int(getattr(self, "_last_v2_selected_idx", -1) or -1),
                "current_v2_selected_delta": getattr(self, "_last_v2_selected_delta", None),
                "current_v2_gate_pass": bool(getattr(self, "_last_v2_gate_pass", False)),
                "current_v2_stage_bucket": str(getattr(self, "_last_v2_bucket", "unknown")),
                "current_v2_geometry_improvement": getattr(self, "_last_v2_geom_imp", None),
                "current_v2_post_delta": getattr(self, "_last_v2_post_delta", None),
                "current_v2_topk_indices": getattr(self, "_last_v2_topk", None),
                "current_v2_selected_post_xy": getattr(self, "_last_v2_selected_post_xy", None),
                "current_v2_selected_post_z": getattr(self, "_last_v2_selected_post_z", None),
                "current_v2_selected_post_yaw": getattr(self, "_last_v2_selected_post_yaw", None),
                "current_v2_apply_gate_pass": bool(self._last_v2_apply_gate_pass),
                "current_v2_apply_block_reason": str(self._last_v2_apply_block_reason),
                "current_v2_apply_assist_scale": float(self._last_v2_apply_assist_scale),
                "current_v2_apply_local_delta": getattr(self, "_last_v2_apply_local_delta", None),
                "current_v2_apply_world_delta": getattr(self, "_last_v2_apply_world_delta", None),
                # v2 delta source trace
                "v2_delta_source": str(getattr(self, "_last_v2_delta_source", "none")),
                "v2_delta_norm": float(getattr(self, "_last_v2_delta_norm", 0.0) or 0.0),
                "v2_cur_xy": getattr(self, "_last_v2_cur_xy", None),
                "v2_cur_z": getattr(self, "_last_v2_cur_z", None),
                "v2_cur_yaw": getattr(self, "_last_v2_cur_yaw", None),
                "v2_gripper_pose_present": bool(getattr(self, "_last_v2_gripper_pose_present", False)),
                "v2_motion_target_pose_present": bool(getattr(self, "_last_v2_mt_pose_present", False)),
                "v2_motion_target_delta_present": bool(getattr(self, "_last_v2_mt_delta_present", False)),
                "v2_current_delta_basin_present": bool(getattr(self, "_last_v2_cdb_present", False)),
                "current_raw_delta_pose_local": getattr(self, "_last_raw_delta_pose_local", None),
                "current_preclip_delta_pose_local": getattr(self, "_last_preclip_delta_pose_local", None),
                "current_clipped_delta_pose_local": getattr(self, "_last_clipped_delta_pose_local", None),
                "current_delta_pose_world": getattr(self, "_last_delta_pose_world", None),
                "current_raw_residual_pos_norm": float(getattr(self, "_last_raw_residual_pos_norm", 0.0) or 0.0),
                "current_preclip_residual_pos_norm": float(getattr(self, "_last_preclip_residual_pos_norm", 0.0) or 0.0),
                "current_clipped_residual_pos_norm": float(getattr(self, "_last_clipped_residual_pos_norm", 0.0) or 0.0),
                "current_raw_residual_yaw_abs": float(getattr(self, "_last_raw_residual_yaw_abs", 0.0) or 0.0),
                "current_preclip_residual_yaw_abs": float(getattr(self, "_last_preclip_residual_yaw_abs", 0.0) or 0.0),
                "current_clipped_residual_yaw_abs": float(getattr(self, "_last_clipped_residual_yaw_abs", 0.0) or 0.0),
                "current_learned_residual_scale": float(self.learned_residual_scale),
                "v2_predictor_micro_assist_eval_count": int(self._v2_predictor_micro_assist_eval_count),
                "v2_predictor_micro_assist_apply_count": int(self._v2_predictor_micro_assist_apply_count),
                "current_v2_predictor_micro_assist_applied": bool(self._last_v2_predictor_micro_assist_applied),
                "current_v2_predictor_micro_assist_block_reason": str(self._last_v2_predictor_micro_assist_block_reason),
                "current_v2_predictor_micro_assist_pos_norm": float(self._last_v2_predictor_micro_assist_pos_norm),
                "current_v2_predictor_micro_assist_rot_norm": float(self._last_v2_predictor_micro_assist_rot_norm),
                "current_v2_predictor_micro_assist_source": str(self._last_v2_predictor_micro_assist_source),
                "v2_assist_apply_count": int(self._v2_assist_apply_count),
                "v2_assist_applied": bool(self._last_v2_assist_applied),
                "v2_assist_enabled": bool(self.enable_v2_nearzone_assist),
                "v2_predictor_micro_assist_enabled": bool(self.enable_v2_predictor_micro_assist_apply),
                "current_near_zone_gate_enabled": bool(self.enable_alignment_near_zone_gate),
                "current_near_zone_gate_pass": bool(self._last_near_zone_gate_pass),
                "current_near_zone_xy_error": None if not np.isfinite(self._last_near_zone_xy_error) else float(self._last_near_zone_xy_error),
                "current_near_zone_z_error": None if not np.isfinite(self._last_near_zone_z_error) else float(self._last_near_zone_z_error),
                "current_near_zone_block_reason": str(self._last_near_zone_block_reason),
                "near_zone_gate_eval_count": int(self._near_zone_gate_eval_count),
                "near_zone_gate_pass_count": int(self._near_zone_gate_pass_count),
                "near_zone_gate_block_count": int(self._near_zone_gate_block_count),
                "zone_state": str(self._alignment_zone_state),
                "alignment_active": bool(self._alignment_active),
                "alignment_takeover_active": bool(self._alignment_zone_state == "takeover"),
                "smoothed_residual_local": np.asarray(self._last_smoothed_delta_local, dtype=np.float32).tolist(),
                "readiness_close_override_count": self._readiness_close_override_count,
                "readiness_open_override_count": self._readiness_open_override_count,
                "readiness_hold_override_count": self._readiness_hold_override_count,
                "readiness_heads_missing_count": self._readiness_heads_missing_count,
                "preclose_alignment_correction_count": self._preclose_alignment_correction_count,
                "gripper_fsm_state": self._gripper_fsm_state,
                "fire_ready_streak": self._fire_ready_streak,
                "verify_steps_remaining": self._verify_steps_remaining,
                "verified_contact_count": self._verified_contact_count,
                "verify_fail_reopen_count": self._verify_fail_reopen_count,
                "current_internal_readiness_fire_applied": bool(self._last_internal_readiness_fire_applied),
                "current_internal_readiness_fire_ready": bool(self._last_internal_readiness_fire_ready),
                "current_internal_readiness_band_ready": bool(self._last_internal_readiness_band_ready),
                "alignment_complete_eval_count": self._alignment_complete_eval_count,
                "alignment_complete_rate": self._alignment_complete_positive_count / max(self._alignment_complete_eval_count, 1),
                "alignment_fire_close_count": self._alignment_fire_close_count,
                "current_alignment_complete": bool(self._last_alignment_complete),
                "current_alignment_fire_decision": bool(self._last_alignment_fire_decision),
                "current_alignment_fire_xy": None if not np.isfinite(self._last_alignment_fire_xy) else float(self._last_alignment_fire_xy),
                "current_alignment_fire_z": None if not np.isfinite(self._last_alignment_fire_z) else float(self._last_alignment_fire_z),
                "current_alignment_fire_yaw": None if not np.isfinite(self._last_alignment_fire_yaw) else float(self._last_alignment_fire_yaw),
                "current_delta_basin_target": np.asarray(
                    getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                    dtype=np.float32,
                ).tolist()
                if controller is not None
                else [0.0] * 6,
                "current_basin_distance_runtime": float(
                    getattr(controller, "_runtime_current_basin_distance", np.inf)
                )
                if controller is not None
                else None,
                "current_basin_xy_runtime": None
                if controller is None
                else float(
                    np.linalg.norm(
                        np.asarray(
                            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                            dtype=np.float32,
                        )[:2]
                    )
                ),
                "current_basin_abs_z_runtime": None
                if controller is None
                else float(
                    abs(
                        np.asarray(
                            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                            dtype=np.float32,
                        )[2]
                    )
                ),
                "current_basin_yaw_runtime": None
                if controller is None
                else float(
                    abs(
                        np.asarray(
                            getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                            dtype=np.float32,
                        )[5]
                    )
                ),
                "current_target_provider_name": None
                if controller is None
                else getattr(controller, "_runtime_target_provider_name", None),
                "current_target_provider_source": None
                if controller is None
                else getattr(controller, "_runtime_target_provider_source", None),
                "current_target_uses_privileged_runtime": bool(
                    getattr(controller, "_runtime_target_uses_privileged", False)
                )
                if controller is not None
                else False,
                "current_handoff_ready": bool(getattr(controller, "_runtime_handoff_ready", False))
                if controller is not None
                else False,
                "current_handoff_metrics": dict(getattr(controller, "_runtime_handoff_metrics", {}))
                if controller is not None
                else {},
                "current_handoff_metric_thresholds": dict(
                    getattr(controller, "_runtime_handoff_metric_thresholds", {})
                )
                if controller is not None
                else {},
                "current_handoff_spec_name": None
                if controller is None
                else getattr(controller, "_runtime_handoff_spec_name", None),
                "current_handoff_target_role": None
                if controller is None
                else getattr(controller, "_runtime_handoff_target_role", None),
                "current_handoff_uses_privileged": bool(
                    getattr(controller, "_runtime_handoff_uses_privileged", False)
                )
                if controller is not None
                else False,
                "current_handoff_min_stable_frames": int(
                    getattr(controller, "_runtime_handoff_min_stable_frames", self.close_veto_ready_streak_frames)
                )
                if controller is not None
                else int(self.close_veto_ready_streak_frames),
            }
        )
        return stats
