#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

DATASET_NPZ="${DATASET_NPZ:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_main_v1_20260423u/handoff_state_dataset_v1.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_main_v1_20260423u/train_minimal}"
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_overlapfix_zaware_20260423o/train_overlapfix_zaware_small/student_handoff_state_head_v2_best_ready.pt}"
CONSISTENCY_CKPT="${CONSISTENCY_CKPT:-$INIT_CKPT}"

mkdir -p "$OUTPUT_DIR"

# Mainline convergence: minimal objective only.
# Keep geometry + band + light ready calibration; disable patch-style extra losses.
python "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$DATASET_NPZ" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-4}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --lr "${LR:-3e-5}" \
  --val_ratio "${VAL_RATIO:-0.2}" \
  --val_episode_csv "${VAL_EPISODES_CSV:-}" \
  --seed "${SEED:-3407}" \
  --lambda_xy "${LAMBDA_XY:-1.6}" \
  --lambda_z "${LAMBDA_Z:-0.6}" \
  --lambda_yaw "${LAMBDA_YAW:-1.0}" \
  --lambda_band "${LAMBDA_BAND:-0.7}" \
  --lambda_ready "${LAMBDA_READY:-0.25}" \
  --lambda_uncertainty 0.0 \
  --lambda_teacher_ready_push 0.0 \
  --lambda_close_ready_crossing 0.0 \
  --lambda_z_aware_ready_crossing 0.0 \
  --lambda_late_ready_logit_lift 0.0 \
  --lambda_ready_neighborhood_consistency 0.0 \
  --lambda_current_profile_hard_negative_veto "${LAMBDA_CURRENT_PROFILE_HARD_NEGATIVE_VETO:-0.0}" \
  --hard_negative_yaw_margin_norm "${HARD_NEGATIVE_YAW_MARGIN_NORM:-1.15}" \
  --lambda_far_negative_calib 0.0 \
  --lambda_far_negative_hard 0.0 \
  --weighted_sampling \
  --sampler_weight_power 1.0 \
  --init_ckpt "$INIT_CKPT" \
  --consistency_ckpt "$CONSISTENCY_CKPT" \
  --consistency_source "${CONSISTENCY_SOURCE:-runtime_like}" \
  --lambda_consistency_band "${LAMBDA_CONS_BAND:-0.02}" \
  --lambda_consistency_ready "${LAMBDA_CONS_READY:-0.02}" \
  --deploy_false_ready_max "${DEPLOY_FALSE_READY_MAX:-1e-8}" \
  2>&1 | tee "$OUTPUT_DIR/stdout.log"

echo "[phaseA-minimal-v1] output_dir=$OUTPUT_DIR"
