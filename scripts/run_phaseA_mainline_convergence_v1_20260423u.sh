#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

OUT_DATA_DIR="${OUT_DATA_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_main_v1_20260423u}"
OUT_TRAIN_DIR="${OUT_TRAIN_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_main_v1_20260423u/train_minimal}"
mkdir -p "$OUT_DATA_DIR" "$OUT_TRAIN_DIR"

BASE_FUSED="${BASE_FUSED:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_readyfirst_fused_20260423u/student_handoff_state_dataset_v2_fused_t2_v3c.npz}"
MAIN_V1_NPZ="$OUT_DATA_DIR/handoff_state_dataset_v1.npz"
MAIN_V1_META="$OUT_DATA_DIR/handoff_state_dataset_v1_meta.json"

python "$ROOT/scripts/build_phaseA_distill_main_dataset_v1.py" \
  --input_npz "$BASE_FUSED" \
  --output_npz "$MAIN_V1_NPZ" \
  --meta_json "$MAIN_V1_META" \
  --runtime_source_name runtime_like \
  --max_runtime_rows -1 \
  --max_teacher_ready_rows -1 \
  --max_xy_block_rows 1200 \
  --max_yaw_needed_rows 900 \
  --max_far_negative_rows 1600 \
  --min_teacher_ready_eps 3 \
  --min_yaw_needed_eps "${MIN_YAW_NEEDED_EPS:-2}"

env \
  DATASET_NPZ="$MAIN_V1_NPZ" \
  OUTPUT_DIR="$OUT_TRAIN_DIR" \
  bash "$ROOT/scripts/run_phaseA_readyfirst_minimal_v1_20260423u.sh"

echo "[phaseA-mainline-convergence-v1] dataset=$MAIN_V1_NPZ train_dir=$OUT_TRAIN_DIR"
