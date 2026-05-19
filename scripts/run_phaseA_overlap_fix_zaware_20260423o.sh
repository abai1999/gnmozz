#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

DATASET_DIR="${DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_overlaprepair_20260423d}"
DATASET_NPZ="${DATASET_NPZ:-$DATASET_DIR/handoff_state_dataset_v2_overlaprepair.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_overlapfix_zaware_20260423o/train_overlapfix_zaware_small}"

INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_overlapfix_crossing_20260423h/train_overlapfix_crossing_small/student_handoff_state_head_v2_best_ready.pt}"
CONSISTENCY_CKPT="${CONSISTENCY_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_overlapfix_crossing_20260423h/train_overlapfix_crossing_small/student_handoff_state_head_v2_best_ready.pt}"

mkdir -p "$OUTPUT_DIR"

python "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$DATASET_NPZ" \
  --output_dir "$OUTPUT_DIR" \
  --epochs 4 \
  --batch_size 64 \
  --lr 3e-5 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_xy 1.5 \
  --lambda_z 0.5 \
  --lambda_yaw 1.0 \
  --lambda_band 0.7 \
  --lambda_ready 0.35 \
  --lambda_teacher_ready_push 0.25 \
  --lambda_close_ready_crossing 0.35 \
  --crossing_xy_max 1.05 \
  --crossing_z_max 1.00 \
  --crossing_yaw_max 1.00 \
  --lambda_z_aware_ready_crossing 0.35 \
  --z_ready_xy_near_max 1.10 \
  --z_ready_z_max 0.90 \
  --z_ready_yaw_max 1.00 \
  --ready_pos_margin_logit 1.10 \
  --lambda_far_negative_calib 0.20 \
  --lambda_far_negative_hard 0.40 \
  --far_negative_min 2.5 \
  --ready_neg_margin_logit 1.00 \
  --lambda_uncertainty 0.0 \
  --weighted_sampling \
  --sampler_weight_power 1.0 \
  --init_ckpt "$INIT_CKPT" \
  --consistency_ckpt "$CONSISTENCY_CKPT" \
  --consistency_source learned32 \
  --lambda_consistency_band 0.02 \
  --lambda_consistency_ready 0.02 \
  --deploy_false_ready_max 1e-8 \
  2>&1 | tee "$OUTPUT_DIR/stdout.log"

echo "[phaseA-overlap-fix-zaware] output_dir=$OUTPUT_DIR"
