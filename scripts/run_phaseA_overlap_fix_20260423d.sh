#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

DATASET_DIR="${DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_overlaprepair_20260423d}"
DATASET_NPZ="${DATASET_NPZ:-$DATASET_DIR/handoff_state_dataset_v2_overlaprepair.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_overlapfix_20260423d/train_overlapfix_small}"
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
CONSISTENCY_CKPT="${CONSISTENCY_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"

mkdir -p "$OUTPUT_DIR"

python "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$DATASET_NPZ" \
  --output_dir "$OUTPUT_DIR" \
  --epochs 4 \
  --batch_size 64 \
  --lr 5e-5 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_xy 1.5 \
  --lambda_z 0.5 \
  --lambda_yaw 1.0 \
  --lambda_band 0.7 \
  --lambda_ready 0.35 \
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

echo "[phaseA-overlap-fix] output_dir=$OUTPUT_DIR"
