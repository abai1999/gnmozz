#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
SUPPORT_NPZ="${SUPPORT_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthready_positive_support_20260420i/support_states.npz}"
NEG_SUPPORT_NPZ="${NEG_SUPPORT_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthdiag_support_20260420g/support_states.npz}"
HIST_POS_NPZ="${HIST_POS_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_teacher_support_states_demo_grasp_yaw_20260419a/support_states.npz}"
MIXED_DATASET_NPZ="${MIXED_DATASET_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthdiag_support_20260420i/near_ready_xyyaw_dataset_mixed_teacherpos_pluscollect.npz}"
MIXED_META_JSON="${MIXED_META_JSON:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthdiag_support_20260420i/near_ready_xyyaw_dataset_mixed_teacherpos_pluscollect.meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_near_ready_xyyaw_20260420i_teacherpos_pluscollect}"

mkdir -p "$(dirname "$MIXED_DATASET_NPZ")" "$OUTPUT_DIR"
cd "$REPO_ROOT"

while [[ ! -f "$SUPPORT_NPZ" ]]; do
  sleep 20
done

env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}" \
  "$PYTHON_BIN" scripts/build_near_ready_xyyaw_dataset.py \
  --support_npz "$NEG_SUPPORT_NPZ" \
  --positive_support_npz "$HIST_POS_NPZ" "$SUPPORT_NPZ" \
  --output_npz "$MIXED_DATASET_NPZ" \
  --meta_json "$MIXED_META_JSON" \
  --substage_id 1 \
  --z_near_mult 4.0 \
  --xy_window_mult 4.0 \
  --yaw_window_mult 2.0 \
  --positive_window_before 8 \
  --positive_window_after 0 \
  --positive_ready_mult_xy 1.2 \
  --positive_ready_mult_yaw 1.2 \
  --positive_sample_boost 8.0

env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}" PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" scripts/train_near_ready_xyyaw_predictor.py \
  --dataset_npz "$MIXED_DATASET_NPZ" \
  --output_dir "$OUTPUT_DIR" \
  --epochs 12 \
  --batch_size 64 \
  --lr 3e-4
