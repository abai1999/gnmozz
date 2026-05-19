#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
PYTHON_BIN="/home/guoning/my_conda_envs/vla-adapter/bin/python"
GPU_ID="${GPU_ID:-1}"

STAGE0_BEST="${STAGE0_BEST:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_stage0_20260422e/train_stage0_teacher_success/student_handoff_state_head_v2_best.pt}"
ANCHOR_DATASET="${ANCHOR_DATASET:-$ROOT/runtime_artifacts/stage_refiner/teacher_success_anchor_stage0_20260422e/teacher_success_anchor_dataset.npz}"

RUN8_STAGE1A_DATASET_DIR="${RUN8_STAGE1A_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run8_stage1a_20260422g}"
RUN8_STAGE1A_OUT_DIR="${RUN8_STAGE1A_OUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422g/train_run8_stage1a_anchorbias}"

RUN8_STAGE1B_BASE_DATASET_DIR="${RUN8_STAGE1B_BASE_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run8_stage1b_base_20260422g}"
RUN8_STAGE1B_MERGED_DATASET_DIR="${RUN8_STAGE1B_MERGED_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run8_stage1b_merged_20260422g}"
RUN8_STAGE1B_OUT_DIR="${RUN8_STAGE1B_OUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422g/train_run8_stage1b_mixed}"

mkdir -p "$RUN8_STAGE1A_DATASET_DIR" "$RUN8_STAGE1A_OUT_DIR" "$RUN8_STAGE1B_BASE_DATASET_DIR" "$RUN8_STAGE1B_MERGED_DATASET_DIR" "$RUN8_STAGE1B_OUT_DIR"

echo "[run8-stage1a] build anchor-biased dataset"
"$PYTHON_BIN" "$ROOT/scripts/build_student_handoff_state_dataset_v2.py" \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_parallel8_20260421m/support_states_merged.npz" \
  --source_name lateprofile_parallel8 \
  --source_weight_mult 1.75 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_next_parallel8_20260422c/support_states_merged.npz" \
  --source_name lateprofile_next_parallel8 \
  --source_weight_mult 1.70 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_readypos_parallel8_20260422d/support_states_merged.npz" \
  --source_name lateprofile_readypos_parallel8 \
  --source_weight_mult 2.45 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_truthready_positive_support_20260420i/support_states.npz" \
  --source_name truthready_positive \
  --source_weight_mult 1.25 \
  --output_npz "$RUN8_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_supportmix.npz" \
  --meta_json "$RUN8_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_supportmix.meta.json"

echo "[run8-stage1a] merge support mix + anchor"
"$PYTHON_BIN" "$ROOT/scripts/merge_student_handoff_state_datasets_v2.py" \
  --dataset_npz "$RUN8_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_supportmix.npz" \
  --weight_mult 1.0 \
  --dataset_npz "$ANCHOR_DATASET" \
  --weight_mult 1.85 \
  --output_npz "$RUN8_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --meta_json "$RUN8_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_mixed.meta.json"

echo "[run8-stage1a] short anchor-biased training"
env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$PYTHON_BIN" "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$RUN8_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_dir "$RUN8_STAGE1A_OUT_DIR" \
  --epochs 3 \
  --batch_size 64 \
  --lr 2e-4 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_band 0.74 \
  --lambda_ready 0.24 \
  --lambda_uncertainty 0.0 \
  --lambda_xy 1.85 \
  --lambda_z 0.40 \
  --lambda_yaw 0.95 \
  --init_ckpt "$STAGE0_BEST" | tee "$RUN8_STAGE1A_OUT_DIR/stdout.log"

STAGE1A_BEST="${RUN8_STAGE1A_OUT_DIR}/student_handoff_state_head_v2_best_phaseA_deploy.pt"
if [[ ! -s "$STAGE1A_BEST" ]]; then
  STAGE1A_BEST="${RUN8_STAGE1A_OUT_DIR}/student_handoff_state_head_v2_best_ready.pt"
fi
if [[ ! -s "$STAGE1A_BEST" ]]; then
  STAGE1A_BEST="${RUN8_STAGE1A_OUT_DIR}/student_handoff_state_head_v2_best.pt"
fi
if [[ ! -s "$STAGE1A_BEST" ]]; then
  echo "[run8-stage1a] missing deploy init checkpoint" >&2
  exit 1
fi

echo "[run8-stage1b] build full mixed support dataset"
"$PYTHON_BIN" "$ROOT/scripts/build_student_handoff_state_dataset_v2.py" \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_resync_learned_full32_20260421h/support_states.npz" \
  --source_name learned32 \
  --source_weight_mult 0.75 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_learned16_20260421k/support_states.npz" \
  --source_name targeted16 \
  --source_weight_mult 0.45 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_xyrecovery_parallel7_20260421k2/support_states_merged.npz" \
  --source_name xyrecovery_parallel7 \
  --source_weight_mult 0.30 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_parallel8_20260421m/support_states_merged.npz" \
  --source_name lateprofile_parallel8 \
  --source_weight_mult 1.55 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_next_parallel8_20260422c/support_states_merged.npz" \
  --source_name lateprofile_next_parallel8 \
  --source_weight_mult 1.60 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_readypos_parallel8_20260422d/support_states_merged.npz" \
  --source_name lateprofile_readypos_parallel8 \
  --source_weight_mult 2.10 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_truthready_positive_support_20260420i/support_states.npz" \
  --source_name truthready_positive \
  --source_weight_mult 1.15 \
  --output_npz "$RUN8_STAGE1B_BASE_DATASET_DIR/handoff_state_dataset_v2_supportmix.npz" \
  --meta_json "$RUN8_STAGE1B_BASE_DATASET_DIR/handoff_state_dataset_v2_supportmix.meta.json"

echo "[run8-stage1b] merge full support mix + anchor"
"$PYTHON_BIN" "$ROOT/scripts/merge_student_handoff_state_datasets_v2.py" \
  --dataset_npz "$RUN8_STAGE1B_BASE_DATASET_DIR/handoff_state_dataset_v2_supportmix.npz" \
  --weight_mult 1.0 \
  --dataset_npz "$ANCHOR_DATASET" \
  --weight_mult 1.35 \
  --output_npz "$RUN8_STAGE1B_MERGED_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --meta_json "$RUN8_STAGE1B_MERGED_DATASET_DIR/handoff_state_dataset_v2_mixed.meta.json"

echo "[run8-stage1b] full mixed training"
env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$PYTHON_BIN" "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$RUN8_STAGE1B_MERGED_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_dir "$RUN8_STAGE1B_OUT_DIR" \
  --epochs 16 \
  --batch_size 64 \
  --lr 2e-4 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_band 0.72 \
  --lambda_ready 0.24 \
  --lambda_uncertainty 0.0 \
  --lambda_xy 1.90 \
  --lambda_z 0.45 \
  --lambda_yaw 0.95 \
  --init_ckpt "$STAGE1A_BEST" | tee "$RUN8_STAGE1B_OUT_DIR/stdout.log"
