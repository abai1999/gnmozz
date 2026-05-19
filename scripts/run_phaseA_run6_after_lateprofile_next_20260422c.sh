#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
PYTHON_BIN="/home/guoning/my_conda_envs/vla-adapter/bin/python"

NEW_LATEPROFILE_NPZ="${NEW_LATEPROFILE_NPZ:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_next_parallel8_20260422c/support_states_merged.npz}"
DATASET_OUT_DIR="${DATASET_OUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run6_20260422c}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422c/train_run6_mixed4}"

echo "[run6] waiting for new late-profile npz: $NEW_LATEPROFILE_NPZ"
while [[ ! -s "$NEW_LATEPROFILE_NPZ" ]]; do
  sleep 10
done
echo "[run6] detected new late-profile npz"

mkdir -p "$DATASET_OUT_DIR" "$TRAIN_OUT_DIR"

"$PYTHON_BIN" "$ROOT/scripts/build_student_handoff_state_dataset_v2.py" \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_resync_learned_full32_20260421h/support_states.npz" \
  --source_name learned32 \
  --source_weight_mult 1.0 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_learned16_20260421k/support_states.npz" \
  --source_name targeted16 \
  --source_weight_mult 0.60 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_xyrecovery_parallel7_20260421k2/support_states_merged.npz" \
  --source_name xyrecovery_parallel7 \
  --source_weight_mult 0.50 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_parallel8_20260421m/support_states_merged.npz" \
  --source_name lateprofile_parallel8 \
  --source_weight_mult 1.35 \
  --support_npz "$NEW_LATEPROFILE_NPZ" \
  --source_name lateprofile_next_parallel8 \
  --source_weight_mult 1.60 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_truthready_positive_support_20260420i/support_states.npz" \
  --source_name truthready_positive \
  --source_weight_mult 1.75 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_truthready_anchor_support_20260420j/support_states.npz" \
  --source_name truthready_anchor \
  --source_weight_mult 1.25 \
  --output_npz "$DATASET_OUT_DIR/handoff_state_dataset_v2_mixed.npz" \
  --meta_json "$DATASET_OUT_DIR/handoff_state_dataset_v2_mixed.meta.json"

env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$PYTHON_BIN" "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$DATASET_OUT_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_dir "$TRAIN_OUT_DIR" \
  --epochs 16 \
  --batch_size 64 \
  --lr 2e-4 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_band 0.6 \
  --lambda_ready 0.15 \
  --lambda_uncertainty 0.0 \
  --lambda_xy 1.75 \
  --lambda_z 0.45 \
  --lambda_yaw 0.95

