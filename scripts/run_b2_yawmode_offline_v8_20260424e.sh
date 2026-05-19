#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

INPUT_DATASET="${INPUT_DATASET:-runtime_artifacts/stage_refiner/b1b2_actioncentric_dataset_v6_yawbank_20260424c/b1b2_actioncentric_dataset_v6_yawbank.npz}"
DATASET_DIR="${DATASET_DIR:-runtime_artifacts/stage_refiner/b1b2_actioncentric_dataset_v7_yawmode_20260424e}"
DATASET_NPZ="${DATASET_NPZ:-$DATASET_DIR/b1b2_actioncentric_dataset_v7_yawmode.npz}"
DATASET_META="${DATASET_META:-$DATASET_DIR/b1b2_actioncentric_dataset_v7_yawmode_meta.json}"
HANDOFF_CKPT="${HANDOFF_CKPT:-runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-runtime_artifacts/stage_refiner/student_candidate_evaluator_v2_b2_yawmode_v8_20260424e}"

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR"

"$PYTHON_BIN" scripts/build_b1b2_actioncentric_dataset_v1.py \
  --input_npz "$INPUT_DATASET" \
  --output_npz "$DATASET_NPZ" \
  --meta_json "$DATASET_META" \
  --allow_insufficient

"$PYTHON_BIN" scripts/train_student_candidate_evaluator_v2.py \
  --dataset_npz "$DATASET_NPZ" \
  --handoff_state_ckpt "$HANDOFF_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --candidate_scope yaw_aware \
  --yaw_mode_stratified_split \
  --stage_m_epochs "${STAGE_M_EPOCHS:-2}" \
  --stage_r_epochs "${STAGE_R_EPOCHS:-4}" \
  --stage_j_epochs "${STAGE_J_EPOCHS:-2}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --lr "${LR:-2e-4}" \
  --min_val_yaw_apply_eps "${MIN_VAL_YAW_APPLY_EPS:-2}" \
  --min_val_yaw_keep_eps "${MIN_VAL_YAW_KEEP_EPS:-3}" \
  --min_train_yaw_apply_eps "${MIN_TRAIN_YAW_APPLY_EPS:-2}" \
  --min_train_yaw_keep_eps "${MIN_TRAIN_YAW_KEEP_EPS:-3}" \
  --yaw_pairwise_weight "${YAW_PAIRWISE_WEIGHT:-0.25}" \
  --yaw_keep_pairwise_weight "${YAW_KEEP_PAIRWISE_WEIGHT:-0.35}" \
  --yaw_mode_weight "${YAW_MODE_WEIGHT:-2.0}" \
  --yaw_keep_abs "${YAW_KEEP_ABS:-0.035}"
