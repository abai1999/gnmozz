#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

DATASET_NPZ="${DATASET_NPZ:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_from_scratch_dataset_20260507a.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_from_scratch_formal_run1_20260507a}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
YAW_WEIGHT="${YAW_WEIGHT:-1.5}"
DISABLE_PLANNER_ACTION="${DISABLE_PLANNER_ACTION:-1}"
DISABLE_FORCE="${DISABLE_FORCE:-1}"

mkdir -p "$OUTPUT_DIR"

ARGS=(
  --dataset_npz "$DATASET_NPZ"
  --output_dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --yaw_weight "$YAW_WEIGHT"
)

if [[ "$DISABLE_PLANNER_ACTION" == "1" ]]; then
  ARGS+=(--disable_planner_action)
fi
if [[ "$DISABLE_FORCE" == "1" ]]; then
  ARGS+=(--disable_force)
fi

exec "$PYTHON_BIN" scripts/train_alignment_v3_direct_local_controller.py "${ARGS[@]}"
