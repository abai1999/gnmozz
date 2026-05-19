#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
TAG="${TAG:-20260515a}"
EPISODE_INDICES="${EPISODE_INDICES:-0,1,2}"
MAX_STEPS="${MAX_STEPS:-420}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/insert_phase2_compare_${TAG}}"
PLANNER_DIR="$OUTPUT_ROOT/planner_only"
TEACHER_DIR="$OUTPUT_ROOT/insert_teacher"

mkdir -p "$PLANNER_DIR" "$TEACHER_DIR"

env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name insert_onto_square_peg \
  --episode_indices "$EPISODE_INDICES" \
  --target_provider_mode canonical_fallback \
  --use_stage_aware_refiner \
  --stage_refiner_mode planner_only \
  --output_dir "$PLANNER_DIR" \
  --record_gripper_trace \
  --record_video \
  --write_episode_videos \
  --no_best_gif \
  --max_steps "$MAX_STEPS" \
  --eval_seed 3407

env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name insert_onto_square_peg \
  --episode_indices "$EPISODE_INDICES" \
  --target_provider_mode teacher_oracle \
  --record_teacher_truth_metrics \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --output_dir "$TEACHER_DIR" \
  --record_gripper_trace \
  --record_video \
  --write_episode_videos \
  --no_best_gif \
  --max_steps "$MAX_STEPS" \
  --alignment_tc_privileged_teacher_collect \
  --alignment_tc_teacher_target_mode insert_commit \
  --alignment_tc_raw_privileged_target_mode insert_commit \
  --alignment_tc_teacher_no_force_open_until_close_ready \
  --alignment_tc_teacher_yaw_imitation_enabled \
  --alignment_diffusion_raw_output_npz "$TEACHER_DIR/alignment_tc_teacher_insert_${TAG}.npz" \
  --alignment_diffusion_raw_report_json "$TEACHER_DIR/alignment_tc_teacher_insert_report_${TAG}.json" \
  --alignment_diffusion_raw_horizon 8 \
  --alignment_diffusion_raw_near_depth_threshold 0.085 \
  --alignment_diffusion_raw_micro_depth_threshold 0.045 \
  --alignment_tc_raw_use_privileged_delta_gate \
  --alignment_tc_raw_near_xy_threshold 0.030 \
  --alignment_tc_raw_near_abs_z_threshold 0.070 \
  --alignment_tc_raw_near_yaw_threshold 0.35 \
  --alignment_tc_raw_micro_xy_threshold 0.010 \
  --alignment_tc_raw_micro_abs_z_threshold 0.030 \
  --alignment_tc_raw_micro_yaw_threshold 0.18 \
  --eval_seed 3407
