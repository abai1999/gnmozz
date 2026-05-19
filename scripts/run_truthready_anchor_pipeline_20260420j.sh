#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

NEG_SUPPORT_NPZ="${NEG_SUPPORT_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthdiag_support_20260420g/support_states.npz}"
HIST_POS_NPZ="${HIST_POS_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_teacher_support_states_demo_grasp_yaw_20260419a/support_states.npz}"
SOFT_POS_NPZ="${SOFT_POS_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthready_positive_support_20260420i/support_states.npz}"
ANCHOR_POS_NPZ="${ANCHOR_POS_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthready_anchor_support_20260420j/support_states.npz}"

OUT_DATA_DIR="${OUT_DATA_DIR:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthdiag_support_20260420j}"
OUT_MODEL_DIR="${OUT_MODEL_DIR:-/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_near_ready_xyyaw_20260420j_anchorplus}"
OUT_DATASET_NPZ="${OUT_DATASET_NPZ:-$OUT_DATA_DIR/near_ready_xyyaw_dataset_anchorplus.npz}"
OUT_META_JSON="${OUT_META_JSON:-$OUT_DATA_DIR/near_ready_xyyaw_dataset_anchorplus.meta.json}"

mkdir -p "$OUT_DATA_DIR" "$OUT_MODEL_DIR"
cd "$REPO_ROOT"

while [[ ! -f "$ANCHOR_POS_NPZ" ]]; do
  sleep 20
done

env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}" \
  "$PYTHON_BIN" scripts/build_near_ready_xyyaw_dataset.py \
  --support_npz "$NEG_SUPPORT_NPZ" \
  --positive_support_npz "$HIST_POS_NPZ" "$SOFT_POS_NPZ" "$ANCHOR_POS_NPZ" \
  --output_npz "$OUT_DATASET_NPZ" \
  --meta_json "$OUT_META_JSON" \
  --substage_id 1 \
  --z_near_mult 4.0 \
  --xy_window_mult 4.0 \
  --yaw_window_mult 2.0 \
  --positive_window_before 8 \
  --positive_window_after 0 \
  --positive_ready_mult_xy 1.2 \
  --positive_ready_mult_yaw 1.2 \
  --positive_ready_mult_z 1.5 \
  --positive_sample_boost 8.0

env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}" PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" scripts/train_near_ready_xyyaw_predictor.py \
  --dataset_npz "$OUT_DATASET_NPZ" \
  --output_dir "$OUT_MODEL_DIR" \
  --epochs 12 \
  --batch_size 64 \
  --lr 3e-4
