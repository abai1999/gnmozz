#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${LD_PRELOAD:-/home/guoning/my_conda_envs/vla-adapter/lib/libstdc++.so.6}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
DEMO_FROM_EPISODE="${DEMO_FROM_EPISODE:-0}"
EPISODE_COUNT="${EPISODE_COUNT:-10}"
MAX_STEPS="${MAX_STEPS:-360}"
VIDEO_EPISODES="${VIDEO_EPISODES:-10}"
TAG="${TAG:-20260518a_edgepair_label_only_phase1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/privileged_teacher_raw_${TAG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
RAW_OUTPUT_NPZ="${RAW_OUTPUT_NPZ:-$OUTPUT_DIR/alignment_tc_privileged_teacher_raw_${TAG}.npz}"
RAW_REPORT_JSON="${RAW_REPORT_JSON:-$OUTPUT_DIR/alignment_tc_privileged_teacher_raw_report_${TAG}.json}"
TRACE_JSONL="${TRACE_JSONL:-$OUTPUT_DIR/takeover_trace_${TAG}.jsonl}"

mkdir -p "$OUTPUT_DIR"

env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/collect_planner_state_expert_recovery.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --demo_from_episode "$DEMO_FROM_EPISODE" \
  --demo_max_attempts 10 \
  --max_rollout_steps "$MAX_STEPS" \
  --num_episodes "$EPISODE_COUNT" \
  --video_episodes "$VIDEO_EPISODES" \
  --record_video \
  --output_dir "$OUTPUT_DIR" \
  --output_npz "$RAW_OUTPUT_NPZ" \
  --report_json "$RAW_REPORT_JSON" \
  --takeover_trace_jsonl "$TRACE_JSONL" \
  --planner_min_steps_before_takeover 8 \
  --planner_takeover_xy_threshold 0.075 \
  --planner_takeover_abs_z_threshold 0.140 \
  --planner_takeover_yaw_threshold 0.700 \
  --planner_takeover_yaw_guard_threshold 1.200 \
  --no_allow_broad_near_takeover \
  --no_fallback_to_best_broad_near \
  --force_alignment_probe_if_no_takeover \
  --planner_takeover_motion_steps 80 \
  --motion_corridor_force_descend_after_steps 12 \
  --motion_corridor_descend_xy_threshold 0.010 \
  --motion_corridor_descend_yaw_threshold 0.100 \
  --grasp_recovery_close_xy_threshold 0.0032 \
  --grasp_recovery_close_z_threshold 0.0035 \
  --grasp_recovery_close_yaw_threshold 0.025 \
  --grasp_recovery_verify_lift_threshold 0.010 \
  --grasp_recovery_verify_consecutive_steps 2 \
  --teacher_grasp_ready_threshold 0.55 \
  --teacher_close_contact_depth_threshold 0.020 \
  --record_edgepair_labels \
  --edgepair_label_target_mode commit
