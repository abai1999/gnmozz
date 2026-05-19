#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_ready.pt}"

OUT_ROOT="${OUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/current_profile_teacher_assisted_yawneeded_20260423v}"
SUPPORT_OUT_DIR="${SUPPORT_OUT_DIR:-$ROOT/runtime_artifacts/residual_data/insert_phase1_current_profile_teacher_assisted_yawneeded_20260423v}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-current_profile_teacher_assisted_yawneeded_20260423v}"

# Focus episodes with existing yaw-needed evidence from teacher-success / truthready assets.
EPISODE_INDICES="${EPISODE_INDICES:-11,18,23,26,34,36,45,46}"
GPU_IDS="${GPU_IDS:-2,3,7}"
MAX_STEPS="${MAX_STEPS:-300}"

mkdir -p "$OUT_ROOT" "$SUPPORT_OUT_DIR" "$SUPPORT_OUT_DIR/logs"
LOG="$SUPPORT_OUT_DIR/logs/driver.log"

EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:---allow_privileged_runtime \
--close_veto_ready_streak_frames 3 \
--close_veto_settle_steps 0 \
--no_best_gif \
--teacher_use_continuous_smooth_control \
--teacher_smooth_kp_xy 0.52 \
--teacher_smooth_kp_z 0.58 \
--teacher_smooth_kp_yaw 0.18 \
--teacher_smooth_yaw_deadband 0.020 \
--teacher_max_yaw_step 0.008 \
--teacher_near_max_step 0.004 \
--teacher_far_max_step 0.006 \
--teacher_candidate_hold_steps 3 \
--teacher_candidate_switch_margin 0.45}"

setsid env \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  ALIGNMENT_CKPT="$ALIGNMENT_CKPT" \
  TARGET_PROVIDER_CKPT="$TARGET_PROVIDER_CKPT" \
  HANDOFF_PROVIDER_CKPT="$HANDOFF_PROVIDER_CKPT" \
  COLLECT_MODE="oracle_target_upper_bound" \
  OUT_ROOT="$OUT_ROOT" \
  SUPPORT_OUT_DIR="$SUPPORT_OUT_DIR" \
  RUN_NAME_SUFFIX="$RUN_NAME_SUFFIX" \
  EPISODE_INDICES="$EPISODE_INDICES" \
  GPU_IDS="$GPU_IDS" \
  MAX_STEPS="$MAX_STEPS" \
  EXTRA_EVAL_ARGS="$EXTRA_EVAL_ARGS" \
  bash "$ROOT/scripts/run_collect_targeted_multi_gpu.sh" > "$LOG" 2>&1 < /dev/null &

echo $! > "$SUPPORT_OUT_DIR/logs/driver.pid"
echo "[teacher-assisted-yawneeded] launched pid=$(cat "$SUPPORT_OUT_DIR/logs/driver.pid")"
echo "[teacher-assisted-yawneeded] log=$LOG"
echo "[teacher-assisted-yawneeded] episodes=$EPISODE_INDICES gpus=$GPU_IDS"
