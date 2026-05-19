#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

DATASET="${DATASET:-$ROOT/runtime_artifacts/alignment_tc_diffusion/teacher_20260511a/alignment_tc_diffusion_teacher_20260511a.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_20260511a}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
DELTA_WEIGHT="${DELTA_WEIGHT:-1.0}"
HEATMAP_WEIGHT="${HEATMAP_WEIGHT:-0.25}"
CONFIDENCE_WEIGHT="${CONFIDENCE_WEIGHT:-0.35}"
TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT:-1.0}"
PROGRESS_WEIGHT="${PROGRESS_WEIGHT:-0.4}"
RISK_WEIGHT="${RISK_WEIGHT:-0.5}"
STOP_WEIGHT="${STOP_WEIGHT:-0.25}"
SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.05}"
DELTA_XY_WEIGHT="${DELTA_XY_WEIGHT:-1.0}"
DELTA_Z_WEIGHT="${DELTA_Z_WEIGHT:-1.0}"
DELTA_YAW_WEIGHT="${DELTA_YAW_WEIGHT:-1.0}"
DELTA_ROLLPITCH_WEIGHT="${DELTA_ROLLPITCH_WEIGHT:-0.5}"
TRAJECTORY_XY_WEIGHT="${TRAJECTORY_XY_WEIGHT:-1.0}"
TRAJECTORY_Z_WEIGHT="${TRAJECTORY_Z_WEIGHT:-1.0}"
TRAJECTORY_YAW_WEIGHT="${TRAJECTORY_YAW_WEIGHT:-1.0}"
PROGRESS_XY_WEIGHT="${PROGRESS_XY_WEIGHT:-1.0}"
PROGRESS_Z_WEIGHT="${PROGRESS_Z_WEIGHT:-1.0}"
PROGRESS_YAW_WEIGHT="${PROGRESS_YAW_WEIGHT:-1.0}"

exec "$PYTHON_BIN" "$ROOT/scripts/train_alignment_tc_diffusion_refiner.py" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --horizon 8 \
  --max_pos_step 0.0015 \
  --max_yaw_step 0.0060 \
  --delta_weight "$DELTA_WEIGHT" \
  --heatmap_weight "$HEATMAP_WEIGHT" \
  --confidence_weight "$CONFIDENCE_WEIGHT" \
  --trajectory_weight "$TRAJECTORY_WEIGHT" \
  --progress_weight "$PROGRESS_WEIGHT" \
  --risk_weight "$RISK_WEIGHT" \
  --stop_weight "$STOP_WEIGHT" \
  --smooth_weight "$SMOOTH_WEIGHT" \
  --delta_xy_weight "$DELTA_XY_WEIGHT" \
  --delta_z_weight "$DELTA_Z_WEIGHT" \
  --delta_yaw_weight "$DELTA_YAW_WEIGHT" \
  --delta_rollpitch_weight "$DELTA_ROLLPITCH_WEIGHT" \
  --trajectory_xy_weight "$TRAJECTORY_XY_WEIGHT" \
  --trajectory_z_weight "$TRAJECTORY_Z_WEIGHT" \
  --trajectory_yaw_weight "$TRAJECTORY_YAW_WEIGHT" \
  --progress_xy_weight "$PROGRESS_XY_WEIGHT" \
  --progress_z_weight "$PROGRESS_Z_WEIGHT" \
  --progress_yaw_weight "$PROGRESS_YAW_WEIGHT"
