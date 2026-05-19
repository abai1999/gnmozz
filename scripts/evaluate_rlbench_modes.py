"""
evaluate_rlbench_modes.py

Run several RLBench eval modes while reusing a single loaded VLA planner.
This is intended for small ablations where repeated DINO/SigLIP/Qwen loading
dominates runtime.
"""

import argparse
import copy
import os
from pathlib import Path

from evaluate_rlbench import evaluate, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple NFCR modes with one planner load")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--modes", type=str, default="planner_only")
    parser.add_argument("--num_episodes", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=300)
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
    parser.add_argument("--coarse2contact_force_contact_threshold", type=float, default=1.0)
    parser.add_argument("--coarse2contact_force_delta_contact_threshold", type=float, default=0.5)
    parser.add_argument("--coarse2contact_force_jam_threshold", type=float, default=6.0)
    parser.add_argument("--coarse2contact_force_torque_threshold", type=float, default=0.25)
    parser.add_argument("--coarse2contact_force_spike_threshold", type=float, default=10.5)
    parser.add_argument("--coarse2contact_backoff_m", type=float, default=0.003)
    parser.add_argument("--coarse2contact_lateral_m", type=float, default=0.0015)
    parser.add_argument("--run_full_horizon_on_success", action="store_true", default=True)
    parser.add_argument("--stop_on_success", dest="run_full_horizon_on_success", action="store_false")
    parser.add_argument("--output_root", type=str, default="eval_logs/insert_onto_square_peg")
    parser.add_argument("--name_suffix", type=str, default="s3407")
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
    parser.add_argument("--alignment_tc_diffusion_execute_steps", type=int, default=1)
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
        choices=["active", "pregrasp", "commit"],
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
    parser.add_argument("--b1_group_shadow_gate_mode", type=str, default="broad", choices=["broad", "close_only"])
    parser.add_argument("--student_candidate_evaluator_shadow_ckpt", type=str, default=None)
    parser.add_argument("--student_candidate_evaluator_handoff_ckpt", type=str, default=None)
    parser.add_argument("--enable_b2_candidate_bounded_v0", action="store_true", default=False)
    parser.add_argument("--b2_candidate_apply_conf_threshold", type=float, default=0.431)
    parser.add_argument("--b2_candidate_apply_margin_threshold", type=float, default=0.010)
    parser.add_argument(
        "--close_veto_runtime_geometry_fallback_for_bounded",
        action="store_true",
        default=False,
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
    )
    parser.add_argument(
        "--b2_candidate_shadow_gate_mode",
        type=str,
        default="broad",
        choices=["broad", "close_only", "nearish_only"],
    )
    parser.add_argument("--b2_candidate_shadow_yaw_probe_values", type=str, default="")
    parser.add_argument("--runtime_candidate_yaw_probe_values", type=str, default="")
    parser.add_argument("--near_ready_alignment_ckpt", type=str, default=None)
    parser.add_argument("--residual_score_adapter_ckpt", type=str, default=None)
    parser.add_argument(
        "--record_teacher_truth_metrics",
        action="store_true",
        default=False,
        help="Record privileged teacher-truth xy/z/yaw metrics to traces for diagnosis only.",
    )
    parser.add_argument(
        "--enforce_no_privileged_runtime",
        action="store_true",
        default=False,
        help="Fail if a student/deployment mode receives teacher-oracle runtime targets.",
    )
    parser.add_argument(
        "--allow_privileged_runtime",
        dest="enforce_no_privileged_runtime",
        action="store_false",
        help="Allow teacher-oracle runtime targets for oracle upper-bound modes.",
    )
    parser.add_argument("--vlm_path", type=str, default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", type=str, default="pretrained_models/configs/config.json")
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--no_depth", dest="use_depth", action="store_false")
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--no_force", dest="use_force", action="store_false")
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="record_video", action="store_false")
    parser.add_argument("--write_episode_videos", action="store_true", default=True)
    parser.add_argument("--no_episode_videos", dest="write_episode_videos", action="store_false")
    parser.add_argument("--write_best_gif", action="store_true", default=False)
    parser.add_argument("--no_best_gif", dest="write_best_gif", action="store_false")
    parser.add_argument("--depth_max", type=float, default=1.0)
    parser.add_argument("--invalid_recovery_lift", type=float, default=0.008)
    parser.add_argument("--invalid_recovery_lift_after", type=int, default=1)
    parser.add_argument("--workspace_clamp_mode", type=str, default="diagnostic", choices=["off", "diagnostic", "tolerance", "hard"])
    parser.add_argument("--workspace_clamp_tolerance", type=float, default=0.01)
    parser.add_argument("--alignment_gate_lookahead", type=int, default=4)
    parser.add_argument("--require_close_intent_for_alignment", dest="require_close_intent_for_alignment", action="store_true", default=True)
    parser.add_argument("--allow_alignment_without_close_intent", dest="require_close_intent_for_alignment", action="store_false")
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
    parser.add_argument("--support_states_output_npz", type=str, default=None)
    parser.add_argument(
        "--depth_force_clean_support",
        action="store_true",
        default=False,
        help="Write planner-only RGB-D/force/action support rows for the depth-force local contact policy.",
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
    parser.add_argument("--teacher_close_xy_threshold", type=float, default=0.006)
    parser.add_argument("--teacher_close_abs_z_threshold", type=float, default=0.005)
    parser.add_argument("--teacher_close_yaw_threshold", type=float, default=0.12)
    parser.add_argument("--teacher_close_orientation_threshold_deg", type=float, default=12.0)
    parser.add_argument("--teacher_close_basin_distance_threshold", type=float, default=-1.0)
    parser.add_argument("--teacher_commit_switch_xy_threshold", type=float, default=0.012)
    parser.add_argument("--teacher_commit_switch_z_threshold", type=float, default=0.012)
    parser.add_argument("--teacher_commit_switch_yaw_threshold", type=float, default=0.20)
    parser.add_argument("--teacher_orientation_rescue_xy_threshold", type=float, default=0.015)
    parser.add_argument("--teacher_orientation_rescue_angle_threshold_deg", type=float, default=8.0)
    parser.add_argument("--teacher_rescue_pitch_small", type=float, default=0.04)
    parser.add_argument("--teacher_rescue_roll_small", type=float, default=0.04)
    parser.add_argument("--teacher_verify_hold_steps", type=int, default=8)
    parser.add_argument("--teacher_verify_lift_threshold", type=float, default=0.012)
    parser.add_argument("--teacher_verify_follow_distance", type=float, default=0.050)
    parser.add_argument("--teacher_retry_lift", type=float, default=0.010)
    parser.add_argument("--teacher_retry_steps", type=int, default=6)
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
    parser.set_defaults(planner_use_depth=False, planner_use_force=False)
    return parser.parse_args()


def build_mode_args(base_args, mode):
    args = copy.copy(base_args)
    args.stage_refiner_mode = mode
    args.use_stage_aware_refiner = mode not in ("planner_only", "gripper")
    args.use_gripper_supervisor = mode in ("gripper", "alignment_gripper")
    args.use_rule_reflex = False
    args.use_learned_residual = False
    args.residual_ckpt = None
    c2c_modes = {"depth_shadow", "depth_apply", "force_reflex", "depth_force"}
    if mode in c2c_modes:
        args.stage_refiner_mode = "planner_only"
        args.use_stage_aware_refiner = False
        args.use_gripper_supervisor = False
        args.alignment_ckpt = None
        args.contact_ckpt = None
        args.target_provider_mode = "canonical_fallback"
        args.target_provider_ckpt = None
        args.coarse2contact_mode = mode
        args.coarse2contact_shadow_only = bool(mode == "depth_shadow" or args.coarse2contact_shadow_only)
    else:
        args.coarse2contact_mode = getattr(args, "coarse2contact_mode", "off")

    if mode in c2c_modes:
        pass
    elif mode == "planner_only":
        args.alignment_ckpt = None
        args.contact_ckpt = None
        args.target_provider_mode = "canonical_fallback"
        args.target_provider_ckpt = None
        if getattr(args, "coarse2contact_mode", "off") == "off":
            args.coarse2contact_mode = "off"
    elif mode == "gripper":
        args.stage_refiner_mode = "planner_only"
        args.alignment_ckpt = None
        args.contact_ckpt = None
    elif mode == "safety_only":
        args.alignment_ckpt = None
        args.contact_ckpt = None
        args.enable_alignment_pose = False
        args.enable_readiness_gripper = False
    elif mode in ("alignment", "pose_alignment_only", "readiness_alignment_pose_only"):
        if not args.alignment_ckpt:
            raise ValueError("alignment mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_readiness_gripper = False
    elif mode in ("readiness_only", "basin_only"):
        if not args.alignment_ckpt:
            raise ValueError(f"{mode} mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = False
        args.enable_readiness_gripper = True
        args.use_pose_alpha = False
    elif mode in ("pose_alignment_only_v2", "pose_alignment_only_basin"):
        if not args.alignment_ckpt:
            raise ValueError(f"{mode} mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = False
        args.use_pose_alpha = False
        args.require_close_intent_for_alignment = False
        args.max_alignment_corrections_per_window = max(args.max_alignment_corrections_per_window, 20)
        args.target_provider_mode = "teacher_oracle"
        args.enforce_no_privileged_runtime = False
        # Keep this path as the old/simple alignment baseline:
        # motion correction only, planner owns gripper and downstream task flow.
        args.oracle_executed_align_collect = False
        args.oracle_executed_pregrasp_collect = False
    elif mode == "oracle_target_upper_bound":
        if not args.alignment_ckpt:
            raise ValueError("oracle_target_upper_bound mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = False
        args.use_pose_alpha = False
        args.require_close_intent_for_alignment = False
        args.max_alignment_corrections_per_window = max(args.max_alignment_corrections_per_window, 20)
        args.target_provider_mode = "teacher_oracle"
        args.enforce_no_privileged_runtime = False
    elif mode in ("oracle_executed_align_collect", "oracle_executed_pregrasp_collect"):
        if not args.alignment_ckpt:
            raise ValueError(f"{mode} mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = False
        args.use_pose_alpha = False
        args.require_close_intent_for_alignment = False
        args.max_alignment_corrections_per_window = max(args.max_alignment_corrections_per_window, 20)
        args.target_provider_mode = "teacher_oracle"
        args.enforce_no_privileged_runtime = False
        args.oracle_executed_align_collect = True
        args.oracle_executed_pregrasp_collect = True
        args.enable_outer_rescue = False
        args.enable_alignment_close_veto = False
    elif mode == "learned_target_mainline":
        if not args.alignment_ckpt:
            raise ValueError("learned_target_mainline mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = False
        args.use_pose_alpha = False
        args.require_close_intent_for_alignment = True
        args.require_close_intent_for_refine_band = True
        args.enable_alignment_near_zone_gate = True
        args.alignment_depth_threshold = min(float(getattr(args, "alignment_depth_threshold", 0.08)), 0.015)
        args.alignment_near_zone_xy_threshold = min(float(getattr(args, "alignment_near_zone_xy_threshold", 0.05)), 0.028)
        args.alignment_near_zone_z_threshold = min(float(getattr(args, "alignment_near_zone_z_threshold", 0.10)), 0.055)
        args.alignment_assist_xy_scale = min(args.alignment_assist_xy_scale, 3.0)
        args.alignment_assist_abs_z_scale = min(args.alignment_assist_abs_z_scale, 10.0)
        args.alignment_assist_yaw_scale = min(args.alignment_assist_yaw_scale, 1.5)
        args.alignment_assist_base_scale = min(args.alignment_assist_base_scale, 0.30)
        args.alignment_assist_close_block_base_scale = min(args.alignment_assist_close_block_base_scale, 0.18)
        args.alignment_takeover_motion_xy_threshold = min(args.alignment_takeover_motion_xy_threshold, 0.012)
        args.alignment_takeover_motion_abs_z_threshold = min(args.alignment_takeover_motion_abs_z_threshold, 0.018)
        args.alignment_takeover_close_block_xy_threshold = min(args.alignment_takeover_close_block_xy_threshold, 0.016)
        args.alignment_takeover_close_block_abs_z_threshold = min(args.alignment_takeover_close_block_abs_z_threshold, 0.022)
        args.max_alignment_corrections_per_window = max(args.max_alignment_corrections_per_window, 12)
        args.target_provider_mode = "learned"
        args.enforce_no_privileged_runtime = True
    elif mode == "learned_target_late_profile_collect":
        if not args.alignment_ckpt:
            raise ValueError("learned_target_late_profile_collect mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = False
        args.use_pose_alpha = False
        args.target_provider_mode = "learned"
        args.enforce_no_privileged_runtime = True
        # Late-intervention recollection profile:
        # keep planner in charge for longer and only allow alignment in a much
        # narrower near-close region so the collected distribution is less
        # polluted by early assist/takeover.
        args.require_close_intent_for_alignment = True
        args.max_alignment_corrections_per_window = max(args.max_alignment_corrections_per_window, 12)
        args.enable_outer_rescue = False
        args.alignment_depth_threshold = min(float(getattr(args, "alignment_depth_threshold", 0.08)), 0.06)
        args.alignment_assist_xy_scale = min(args.alignment_assist_xy_scale, 4.0)
        args.alignment_assist_abs_z_scale = min(args.alignment_assist_abs_z_scale, 12.0)
        args.alignment_assist_yaw_scale = min(args.alignment_assist_yaw_scale, 2.0)
        args.alignment_assist_base_scale = min(args.alignment_assist_base_scale, 0.35)
        args.alignment_assist_close_block_base_scale = min(args.alignment_assist_close_block_base_scale, 0.20)
        args.alignment_takeover_motion_xy_threshold = min(args.alignment_takeover_motion_xy_threshold, 0.015)
        args.alignment_takeover_motion_abs_z_threshold = min(args.alignment_takeover_motion_abs_z_threshold, 0.020)
        args.alignment_takeover_close_block_xy_threshold = min(args.alignment_takeover_close_block_xy_threshold, 0.018)
        args.alignment_takeover_close_block_abs_z_threshold = min(args.alignment_takeover_close_block_abs_z_threshold, 0.025)
    elif mode == "visual_scorer_mainline":
        if not args.alignment_ckpt:
            raise ValueError("visual_scorer_mainline mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = False
        args.use_pose_alpha = False
        args.require_close_intent_for_alignment = False
        args.max_alignment_corrections_per_window = max(args.max_alignment_corrections_per_window, 20)
        args.target_provider_mode = "canonical_fallback"
        args.enforce_no_privileged_runtime = True
        args.enable_outer_rescue = False
        args.enable_alignment_close_veto = True
    elif mode in ("readiness_pose_v2", "basin_pose_joint"):
        if not args.alignment_ckpt:
            raise ValueError(f"{mode} mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = True
        args.use_pose_alpha = False
    elif mode == "alignment_gripper":
        args.stage_refiner_mode = "alignment"
        if not args.alignment_ckpt:
            raise ValueError("alignment_gripper mode requires --alignment_ckpt")
        args.contact_ckpt = None
    elif mode == "readiness_alignment_full":
        args.stage_refiner_mode = "alignment"
        if not args.alignment_ckpt:
            raise ValueError("readiness_alignment_full mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.use_gripper_supervisor = False
        args.enable_readiness_gripper = True
    elif mode == "contact":
        if not args.contact_ckpt:
            raise ValueError("contact mode requires --contact_ckpt")
        args.alignment_ckpt = None
    elif mode == "full":
        if not args.alignment_ckpt or not args.contact_ckpt:
            raise ValueError("full mode requires --alignment_ckpt and --contact_ckpt")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if args.use_stage_aware_refiner and not args.use_depth:
        print(
            f"[multi_eval] mode={mode} requested with --no_depth; "
            "forcing eval depth on for the refiner while keeping planner depth controlled by "
            "--planner_no_depth/--planner_use_depth."
        )
        args.use_depth = True

    base_prefix = "coarse2contact" if mode in {"depth_shadow", "depth_apply", "force_reflex", "depth_force"} else "insert_vo40k"
    base_name = f"{base_prefix}_{mode}_{args.name_suffix}"
    args.output_dir = str(Path(args.output_root) / base_name)
    return args


def main():
    args = parse_args()
    os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.environ.get("COPPELIASIM_ROOT", "/home/guoning/CoppeliaSim"))
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    if args.planner_use_depth is None:
        args.planner_use_depth = args.use_depth
    if args.planner_use_force is None:
        args.planner_use_force = args.use_force

    print("[multi_eval] Loading planner once...")
    preloaded = load_checkpoint(
        args.checkpoint_dir,
        vlm_path=args.vlm_path,
        config_path=args.config_path,
        use_depth=args.planner_use_depth,
        use_force=args.planner_use_force,
    )

    summaries = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        mode_args = build_mode_args(args, mode)
        print(f"\n[multi_eval] Running mode={mode} -> {mode_args.output_dir}")
        summaries[mode] = evaluate(mode_args, preloaded_components=preloaded)

    print("\n[multi_eval] Done.")
    for mode, success_rate in summaries.items():
        print(f"  {mode}: success_rate={success_rate:.3f}")


if __name__ == "__main__":
    main()
