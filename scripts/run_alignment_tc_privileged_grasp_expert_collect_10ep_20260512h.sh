#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-0,1,2,3,4,5,6,7,8,9}"
MAX_STEPS="${MAX_STEPS:-380}"
TAG="${TAG:-20260512h_grasp_expert_10ep}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/privileged_teacher_raw_${TAG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
RAW_OUTPUT_NPZ="${RAW_OUTPUT_NPZ:-$OUTPUT_DIR/alignment_tc_privileged_teacher_raw_${TAG}.npz}"
RAW_REPORT_JSON="${RAW_REPORT_JSON:-$OUTPUT_DIR/alignment_tc_privileged_teacher_raw_report_${TAG}.json}"
AUDIT_JSON="${AUDIT_JSON:-$OUTPUT_DIR/alignment_tc_privileged_teacher_raw_audit_${TAG}.json}"

mkdir -p "$OUTPUT_DIR"

env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --episode_indices "$EPISODE_INDICES" \
  --target_provider_mode canonical_fallback \
  --enforce_no_privileged_runtime \
  --record_teacher_truth_metrics \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --output_dir "$OUTPUT_DIR" \
  --record_gripper_trace \
  --record_video \
  --write_episode_videos \
  --no_best_gif \
  --max_steps "$MAX_STEPS" \
  --alignment_tc_privileged_teacher_collect \
  --alignment_tc_privileged_teacher_close_enabled \
  --alignment_tc_teacher_close_verify \
  --alignment_tc_teacher_force_open_until_close_ready \
  --alignment_tc_teacher_broad_xy_threshold 0.060 \
  --alignment_tc_teacher_broad_abs_z_threshold 0.120 \
  --alignment_tc_teacher_broad_yaw_threshold 0.60 \
  --alignment_tc_teacher_k_xy 0.28 \
  --alignment_tc_teacher_k_z 0.10 \
  --alignment_tc_teacher_k_yaw 0.10 \
  --alignment_tc_teacher_max_pos_step 0.0015 \
  --alignment_tc_teacher_max_yaw_step 0.005 \
  --alignment_tc_teacher_broad_max_pos_step 0.0040 \
  --alignment_tc_teacher_broad_max_yaw_step 0.014 \
  --alignment_tc_teacher_near_max_pos_step 0.0025 \
  --alignment_tc_teacher_near_max_yaw_step 0.008 \
  --alignment_tc_teacher_micro_max_pos_step 0.0012 \
  --alignment_tc_teacher_micro_max_yaw_step 0.005 \
  --alignment_tc_teacher_short_horizon 3 \
  --alignment_tc_teacher_workspace_delta_tolerance 0.00001 \
  --alignment_tc_teacher_already_close_xy 0.0025 \
  --alignment_tc_teacher_already_close_abs_z 0.0030 \
  --alignment_tc_teacher_already_close_yaw 0.015 \
  --alignment_tc_teacher_broad_min_pos_step 0.0015 \
  --alignment_tc_teacher_near_min_pos_step 0.0012 \
  --alignment_tc_teacher_micro_min_pos_step 0.0008 \
  --teacher_no_planner_close_handoff \
  --teacher_close_xy_threshold 0.014 \
  --teacher_close_abs_z_threshold 0.012 \
  --teacher_close_yaw_threshold 0.06 \
  --teacher_close_contact_depth_threshold 0.022 \
  --teacher_grasp_ready_threshold 0.50 \
  --teacher_grasp_xy_threshold 0.014 \
  --teacher_grasp_abs_z_threshold 0.012 \
  --teacher_grasp_yaw_threshold 0.06 \
  --teacher_verify_hold_steps 10 \
  --teacher_verify_min_consecutive_lift_steps 2 \
  --teacher_verify_lift_threshold 0.012 \
  --teacher_verify_follow_distance 0.050 \
  --teacher_retry_lift 0.008 \
  --teacher_retry_steps 4 \
  --teacher_max_retries 2 \
  --alignment_tc_teacher_yaw_control_sign -1.0 \
  --alignment_tc_teacher_yaw_imitation_enabled \
  --alignment_diffusion_raw_output_npz "$RAW_OUTPUT_NPZ" \
  --alignment_diffusion_raw_report_json "$RAW_REPORT_JSON" \
  --alignment_diffusion_raw_horizon 8 \
  --alignment_diffusion_raw_near_depth_threshold 0.085 \
  --alignment_diffusion_raw_micro_depth_threshold 0.045 \
  --alignment_tc_raw_privileged_target_mode commit \
  --alignment_tc_raw_use_privileged_delta_gate \
  --alignment_tc_raw_near_xy_threshold 0.030 \
  --alignment_tc_raw_near_abs_z_threshold 0.070 \
  --alignment_tc_raw_near_yaw_threshold 0.35 \
  --alignment_tc_raw_micro_xy_threshold 0.010 \
  --alignment_tc_raw_micro_abs_z_threshold 0.030 \
  --alignment_tc_raw_micro_yaw_threshold 0.18 \
  --alignment_diffusion_raw_augment_copies 1 \
  --eval_seed 3407

"$PYTHON_BIN" "$ROOT/scripts/audit_alignment_tc_privileged_teacher_raw.py" \
  "$RAW_OUTPUT_NPZ" \
  --output_json "$AUDIT_JSON"
