#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

INPUT_NPZ="${INPUT_NPZ:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_from_scratch_dataset_20260509a.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v4_direct_local_20260509a}"
STAGE_BUCKET_FILTER="${STAGE_BUCKET_FILTER:-near_alignment,micro_contact_refine}"
YAW_WEIGHT="${YAW_WEIGHT:-1.5}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" scripts/build_alignment_v4_short_horizon_teacher.py \
  --input_npz "$INPUT_NPZ" \
  --output_dir "$OUTPUT_ROOT/teacher" \
  --stage_bucket_filter "$STAGE_BUCKET_FILTER"

"$PYTHON_BIN" scripts/build_alignment_v4_contract_matched_dataset.py \
  --input_npz "$INPUT_NPZ" \
  --output_dir "$OUTPUT_ROOT/dataset" \
  --stage_bucket_filter "$STAGE_BUCKET_FILTER"

exec "$PYTHON_BIN" scripts/train_alignment_v4_direct_local_controller.py \
  --dataset_npz "$OUTPUT_ROOT/dataset/alignment_v4_contract_matched_dataset.npz" \
  --output_dir "$OUTPUT_ROOT/train" \
  --yaw_weight "$YAW_WEIGHT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR"
