#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"

OUT_ROOT="${OUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/b2_yawneeded_current_profile_v13_20260424h}"
SUPPORT_OUT_DIR="${SUPPORT_OUT_DIR:-$ROOT/runtime_artifacts/residual_data/insert_phase1_b2_yawneeded_current_profile_v13_20260424h}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-b2_yawneeded_current_profile_v13_20260424h}"

# Focus episodes from the v12/v13 preflight:
# - ep018: current labels are source-confounded; recollect same-profile rows.
# - ep023/045: had oracle/source skew in v12 and need current-profile apply/keep checks.
# - ep012/014/046: existing apply-bearing episodes used to keep apply diversity.
# - ep022/026/034/36: yaw-needed/near-yaw boundary candidates.
EPISODE_INDICES="${EPISODE_INDICES:-18,23,45,12,14,46,22,26,34,36}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-340}"

mkdir -p "$OUT_ROOT" "$SUPPORT_OUT_DIR" "$SUPPORT_OUT_DIR/logs"
LOG="$SUPPORT_OUT_DIR/logs/driver.log"

# This is intentionally learned_target_mainline/no-privileged runtime.  Teacher
# truth is recorded only for labels/audits.  Do not switch this script to
# oracle_target_upper_bound when building v13 supervised labels.
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:---no_video --no_episode_videos --no_best_gif}"

setsid env \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  ALIGNMENT_CKPT="$ALIGNMENT_CKPT" \
  TARGET_PROVIDER_CKPT="$TARGET_PROVIDER_CKPT" \
  HANDOFF_PROVIDER_CKPT="$HANDOFF_PROVIDER_CKPT" \
  COLLECT_MODE="learned_target_mainline" \
  OUT_ROOT="$OUT_ROOT" \
  SUPPORT_OUT_DIR="$SUPPORT_OUT_DIR" \
  RUN_NAME_SUFFIX="$RUN_NAME_SUFFIX" \
  EPISODE_INDICES="$EPISODE_INDICES" \
  GPU_IDS="$GPU_IDS" \
  MAX_STEPS="$MAX_STEPS" \
  EXTRA_EVAL_ARGS="$EXTRA_EVAL_ARGS" \
  bash "$ROOT/scripts/run_collect_targeted_multi_gpu.sh" > "$LOG" 2>&1 < /dev/null &

echo $! > "$SUPPORT_OUT_DIR/logs/driver.pid"
echo "[b2-yawneeded-current-profile-v13] launched pid=$(cat "$SUPPORT_OUT_DIR/logs/driver.pid")"
echo "[b2-yawneeded-current-profile-v13] log=$LOG"
echo "[b2-yawneeded-current-profile-v13] episodes=$EPISODE_INDICES gpus=$GPU_IDS"
