#!/usr/bin/env bash
set -euo pipefail

# Small B2-yaw profile pilot.  This is intentionally a preflight collection,
# not a full training-data recollection.  Run a small fixed set first, then
# pass the merged support through the yaw-mode gate before launching any large
# supplement.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE="${PROFILE:-mid}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/ssd/guoning/VLA_scratch/b2_yaw_profile_pilot_v13_20260425c_${PROFILE}}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT/eval}"
SUPPORT_OUT_DIR="${SUPPORT_OUT_DIR:-$SCRATCH_ROOT/support}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-b2_yawapply_pilot_${PROFILE}_v13_20260425c}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"

# Remaining source-confounding / weak-apply blockers.
EPISODE_INDICES="${EPISODE_INDICES:-18,23,34,45}"
GPU_IDS="${GPU_IDS:-0,1}"
MAX_STEPS="${MAX_STEPS:-340}"

case "$PROFILE" in
  runtime_recollect)
    # Closest to the previous B1/B2 current-profile candidate recollection.
    # Good at same-profile keep/apply for ep14/45, but not yet for ep18.
    PROFILE_ARGS="--allow_privileged_runtime \
--close_veto_ready_streak_frames 3 \
--close_veto_settle_steps 0 \
--no_video --no_episode_videos --no_best_gif \
--teacher_use_continuous_smooth_control \
--teacher_smooth_kp_xy 0.60 \
--teacher_smooth_kp_z 0.58 \
--teacher_smooth_kp_yaw 0.16 \
--teacher_smooth_yaw_deadband 0.018 \
--teacher_max_yaw_step 0.006 \
--teacher_near_max_step 0.0035 \
--teacher_far_max_step 0.006 \
--teacher_candidate_hold_steps 3 \
--teacher_candidate_switch_margin 0.45"
    RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-b2_recollect_pilot_${PROFILE}_v13_20260425c}"
    ;;
  mid)
    # Slightly stronger than 20260425b balance, but still below the old
    # aggressive yaw top-up.  Intended to create both keep and apply windows
    # under the same targeted-yawapply source profile.
    PROFILE_ARGS="--allow_privileged_runtime \
--no_video --no_episode_videos --no_best_gif \
--teacher_use_continuous_smooth_control \
--teacher_smooth_kp_xy 0.59 \
--teacher_smooth_kp_z 0.57 \
--teacher_smooth_kp_yaw 0.13 \
--teacher_smooth_yaw_deadband 0.021 \
--teacher_max_yaw_step 0.005 \
--teacher_near_max_step 0.0035 \
--teacher_far_max_step 0.006 \
--teacher_candidate_hold_steps 3 \
--teacher_candidate_switch_margin 0.45"
    ;;
  strong)
    # Close to the earlier yaw-needed / yaw-apply teacher-assisted profile.
    # Use only as pilot unless it gives clean same-profile keep/apply.
    PROFILE_ARGS="--allow_privileged_runtime \
--no_video --no_episode_videos --no_best_gif \
--teacher_use_continuous_smooth_control \
--teacher_smooth_kp_xy 0.60 \
--teacher_smooth_kp_z 0.58 \
--teacher_smooth_kp_yaw 0.18 \
--teacher_smooth_yaw_deadband 0.018 \
--teacher_max_yaw_step 0.008 \
--teacher_near_max_step 0.0035 \
--teacher_far_max_step 0.006 \
--teacher_candidate_hold_steps 3 \
--teacher_candidate_switch_margin 0.45"
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE; expected runtime_recollect|mid|strong" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_ROOT" "$SUPPORT_OUT_DIR" "$SUPPORT_OUT_DIR/logs"
LOG="$SUPPORT_OUT_DIR/logs/driver.log"

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
  EXTRA_EVAL_ARGS="$PROFILE_ARGS" \
  bash "$ROOT/scripts/run_collect_targeted_multi_gpu.sh" > "$LOG" 2>&1 < /dev/null &

echo $! > "$SUPPORT_OUT_DIR/logs/driver.pid"
echo "[b2-yaw-profile-pilot] profile=$PROFILE pid=$(cat "$SUPPORT_OUT_DIR/logs/driver.pid")"
echo "[b2-yaw-profile-pilot] log=$LOG"
echo "[b2-yaw-profile-pilot] episodes=$EPISODE_INDICES gpus=$GPU_IDS"
