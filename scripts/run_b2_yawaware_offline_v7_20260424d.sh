#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

DATASET_NPZ="${DATASET_NPZ:-runtime_artifacts/stage_refiner/b1b2_actioncentric_dataset_v6_yawbank_20260424c/b1b2_actioncentric_dataset_v6_yawbank.npz}"
HANDOFF_CKPT="${HANDOFF_CKPT:-runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-runtime_artifacts/stage_refiner/student_candidate_evaluator_v2_b2_yawbank_v7_20260424d}"

"$PYTHON_BIN" scripts/train_student_candidate_evaluator_v2.py \
  --dataset_npz "$DATASET_NPZ" \
  --handoff_state_ckpt "$HANDOFF_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --candidate_scope yaw_aware \
  --epochs "${EPOCHS:-8}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --lr "${LR:-2e-4}" \
  --min_val_yaw_needed_eps "${MIN_VAL_YAW_NEEDED_EPS:-3}" \
  --yaw_pairwise_weight "${YAW_PAIRWISE_WEIGHT:-0.25}" \
  --yaw_keep_pairwise_weight "${YAW_KEEP_PAIRWISE_WEIGHT:-0.35}" \
  --yaw_mode_weight "${YAW_MODE_WEIGHT:-2.0}" \
  --yaw_keep_abs "${YAW_KEEP_ABS:-0.035}"
