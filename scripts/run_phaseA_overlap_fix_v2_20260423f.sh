#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

BASE_DATASET_DIR="${BASE_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run12_stage1b_merged_20260422l}"
DATASET_DIR="${DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_overlaprepair_v2_20260423f}"
DATASET_NPZ="${DATASET_NPZ:-$DATASET_DIR/handoff_state_dataset_v2_overlaprepair_v2.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_overlapfix_v2_20260423f/train_overlapfix_v2_small}"

# Start from the overlap-fix v1 best_ready ckpt because it improved ep011 ready_prob
# more than best_phaseA_deploy in runtime shadow.
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_overlapfix_20260423d/train_overlapfix_small/student_handoff_state_head_v2_best_ready.pt}"
CONSISTENCY_CKPT="${CONSISTENCY_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_overlapfix_20260423d/train_overlapfix_small/student_handoff_state_head_v2_best_ready.pt}"

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR"

python "$ROOT/scripts/build_teacher_ready_overlap_repair_dataset_v2.py" \
  --dataset_npz "$BASE_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_npz "$DATASET_NPZ" \
  --meta_json "$DATASET_DIR/handoff_state_dataset_v2_overlaprepair_v2.meta.json" \
  --focus_source learned32 \
  --teacher_ready_focus_episodes 8,28 \
  --teacher_ready_window 20 \
  --focus_window_mult 4.5 \
  --teacher_ready_exact_mult 3.5 \
  --boundary_band_mult 3.0 \
  --boundary_xy_max 1.10 \
  --boundary_z_max 1.00 \
  --boundary_yaw_max 1.00 \
  --far_negative_mult 1.75 \
  --far_negative_min 2.5 \
  --source_mult teacher_success_formal30:0.15 \
  --source_mult truthready_positive:0.75 \
  --source_mult lateprofile_ready_anchor:1.50

python "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$DATASET_NPZ" \
  --output_dir "$OUTPUT_DIR" \
  --epochs 4 \
  --batch_size 64 \
  --lr 3e-5 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_xy 1.6 \
  --lambda_z 0.5 \
  --lambda_yaw 1.0 \
  --lambda_band 0.85 \
  --lambda_ready 0.45 \
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

echo "[phaseA-overlap-fix-v2] output_dir=$OUTPUT_DIR"
