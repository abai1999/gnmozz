#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422k/train_run11_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
ANCHOR_WINDOW_DATASET="${ANCHOR_WINDOW_DATASET:-$ROOT/runtime_artifacts/stage_refiner/teacher_success_anchor_window_run9_20260422i/teacher_success_anchor_window_dataset.npz}"

SUPPORT_LATEPROFILE="${SUPPORT_LATEPROFILE:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_parallel8_20260421m/support_states_merged.npz}"
SUPPORT_LATEPROFILE_NEXTPAR="${SUPPORT_LATEPROFILE_NEXTPAR:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_next_parallel8_20260422c/support_states_merged.npz}"
SUPPORT_LATEPROFILE_READYPOS="${SUPPORT_LATEPROFILE_READYPOS:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_readypos_parallel8_20260422d/support_states_merged.npz}"
SUPPORT_TARGETED16="${SUPPORT_TARGETED16:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_learned16_20260421k/support_states.npz}"
SUPPORT_LEARNED32="${SUPPORT_LEARNED32:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_resync_learned_full32_20260421h/support_states.npz}"
SUPPORT_XYRECOVERY="${SUPPORT_XYRECOVERY:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_xyrecovery_parallel7_20260421k2/support_states_merged.npz}"
SUPPORT_TRUTHREADY_POS="${SUPPORT_TRUTHREADY_POS:-$ROOT/runtime_artifacts/residual_data/insert_phase1_truthready_positive_support_20260420i/support_states.npz}"

DATA_STAGE1A_DIR="${DATA_STAGE1A_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run12_stage1a_20260422l}"
DATA_STAGE1B_BASE_DIR="${DATA_STAGE1B_BASE_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run12_stage1b_base_20260422l}"
DATA_STAGE1B_MERGED_DIR="${DATA_STAGE1B_MERGED_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run12_stage1b_merged_20260422l}"

RUN12_ROOT="${RUN12_ROOT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l}"
RUN12_STAGE1A_OUT_DIR="${RUN12_STAGE1A_OUT_DIR:-$RUN12_ROOT/train_run12_stage1a_calibwarm}"
RUN12_STAGE1B_OUT_DIR="${RUN12_STAGE1B_OUT_DIR:-$RUN12_ROOT/train_run12_stage1b_calibmix}"

mkdir -p "$DATA_STAGE1A_DIR" "$DATA_STAGE1B_BASE_DIR" "$DATA_STAGE1B_MERGED_DIR" "$RUN12_STAGE1A_OUT_DIR" "$RUN12_STAGE1B_OUT_DIR"

if [[ -f "$DATA_STAGE1A_DIR/handoff_state_dataset_v2_supportmix.npz" ]]; then
  echo "[run12-stage1a] reuse existing calibration-biased support mix"
else
  echo "[run12-stage1a] build calibration-biased support mix"
  "$PYTHON_BIN" "$ROOT/scripts/build_student_handoff_state_dataset_v2.py" \
    --support_npz "$SUPPORT_LATEPROFILE" --source_name lateprofile_parallel8 --source_weight_mult 1.90 \
    --support_npz "$SUPPORT_LATEPROFILE_READYPOS" --source_name lateprofile_readypos_parallel8 --source_weight_mult 2.25 \
    --support_npz "$SUPPORT_TARGETED16" --source_name targeted16 --source_weight_mult 0.95 \
    --support_npz "$SUPPORT_LEARNED32" --source_name learned32 --source_weight_mult 0.50 \
    --support_npz "$SUPPORT_XYRECOVERY" --source_name xyrecovery_parallel7 --source_weight_mult 0.30 \
    --support_npz "$SUPPORT_TRUTHREADY_POS" --source_name truthready_positive --source_weight_mult 1.20 \
    --output_npz "$DATA_STAGE1A_DIR/handoff_state_dataset_v2_supportmix.npz" \
    --meta_json "$DATA_STAGE1A_DIR/handoff_state_dataset_v2_supportmix.meta.json"
fi

if [[ -f "$DATA_STAGE1A_DIR/handoff_state_dataset_v2_mixed.npz" ]]; then
  echo "[run12-stage1a] reuse existing merged stage1a dataset"
else
  echo "[run12-stage1a] merge support mix + anchor windows"
  "$PYTHON_BIN" "$ROOT/scripts/merge_student_handoff_state_datasets_v2.py" \
    --dataset_npz "$DATA_STAGE1A_DIR/handoff_state_dataset_v2_supportmix.npz" --weight_mult 1.0 \
    --dataset_npz "$ANCHOR_WINDOW_DATASET" --weight_mult 2.60 \
    --output_npz "$DATA_STAGE1A_DIR/handoff_state_dataset_v2_mixed.npz" \
    --meta_json "$DATA_STAGE1A_DIR/handoff_state_dataset_v2_mixed.meta.json"
fi

echo "[run12-stage1a] train"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$PYTHON_BIN" "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$DATA_STAGE1A_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_dir "$RUN12_STAGE1A_OUT_DIR" \
  --epochs 2 \
  --batch_size 64 \
  --lr 1.0e-4 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_band 0.84 \
  --lambda_ready 0.14 \
  --lambda_uncertainty 0.0 \
  --lambda_xy 1.70 \
  --lambda_z 0.42 \
  --lambda_yaw 1.35 \
  --weighted_sampling \
  --sampler_weight_power 1.18 \
  --deploy_false_ready_max 0.004 \
  --init_ckpt "$INIT_CKPT" \
  | tee "$RUN12_STAGE1A_OUT_DIR/stdout.log"

STAGE1A_DEPLOY="$RUN12_STAGE1A_OUT_DIR/student_handoff_state_head_v2_best_phaseA_deploy.pt"
if [[ ! -f "$STAGE1A_DEPLOY" ]]; then
  STAGE1A_DEPLOY="$RUN12_STAGE1A_OUT_DIR/student_handoff_state_head_v2_best_anchor.pt"
fi
if [[ ! -f "$STAGE1A_DEPLOY" ]]; then
  STAGE1A_DEPLOY="$RUN12_STAGE1A_OUT_DIR/student_handoff_state_head_v2_best.pt"
fi

if [[ -f "$DATA_STAGE1B_BASE_DIR/handoff_state_dataset_v2_supportmix.npz" ]]; then
  echo "[run12-stage1b] reuse existing close-neighborhood calibration mix"
else
  echo "[run12-stage1b] build close-neighborhood calibration mix"
  "$PYTHON_BIN" "$ROOT/scripts/build_student_handoff_state_dataset_v2.py" \
    --support_npz "$SUPPORT_LEARNED32" --source_name learned32 --source_weight_mult 0.55 \
    --support_npz "$SUPPORT_TARGETED16" --source_name targeted16 --source_weight_mult 1.45 \
    --support_npz "$SUPPORT_XYRECOVERY" --source_name xyrecovery_parallel7 --source_weight_mult 0.40 \
    --support_npz "$SUPPORT_LATEPROFILE" --source_name lateprofile_parallel8 --source_weight_mult 1.85 \
    --support_npz "$SUPPORT_LATEPROFILE_NEXTPAR" --source_name lateprofile_next_parallel8 --source_weight_mult 0.95 \
    --support_npz "$SUPPORT_LATEPROFILE_READYPOS" --source_name lateprofile_readypos_parallel8 --source_weight_mult 2.10 \
    --support_npz "$SUPPORT_TRUTHREADY_POS" --source_name truthready_positive --source_weight_mult 1.05 \
    --output_npz "$DATA_STAGE1B_BASE_DIR/handoff_state_dataset_v2_supportmix.npz" \
    --meta_json "$DATA_STAGE1B_BASE_DIR/handoff_state_dataset_v2_supportmix.meta.json"
fi

if [[ -f "$DATA_STAGE1B_MERGED_DIR/handoff_state_dataset_v2_mixed.npz" ]]; then
  echo "[run12-stage1b] reuse existing merged stage1b dataset"
else
  echo "[run12-stage1b] merge close-neighborhood mix + anchor windows"
  "$PYTHON_BIN" "$ROOT/scripts/merge_student_handoff_state_datasets_v2.py" \
    --dataset_npz "$DATA_STAGE1B_BASE_DIR/handoff_state_dataset_v2_supportmix.npz" --weight_mult 1.0 \
    --dataset_npz "$ANCHOR_WINDOW_DATASET" --weight_mult 1.95 \
    --output_npz "$DATA_STAGE1B_MERGED_DIR/handoff_state_dataset_v2_mixed.npz" \
    --meta_json "$DATA_STAGE1B_MERGED_DIR/handoff_state_dataset_v2_mixed.meta.json"
fi

echo "[run12-stage1b] train"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$PYTHON_BIN" "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$DATA_STAGE1B_MERGED_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_dir "$RUN12_STAGE1B_OUT_DIR" \
  --epochs 8 \
  --batch_size 64 \
  --lr 3e-5 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_band 0.84 \
  --lambda_ready 0.14 \
  --lambda_uncertainty 0.0 \
  --lambda_xy 1.90 \
  --lambda_z 0.42 \
  --lambda_yaw 1.35 \
  --weighted_sampling \
  --sampler_weight_power 1.25 \
  --deploy_false_ready_max 0.004 \
  --consistency_ckpt "$STAGE1A_DEPLOY" \
  --consistency_source teacher_success_formal30 \
  --consistency_source lateprofile_ready_anchor \
  --consistency_source truthready_anchor \
  --lambda_consistency_band 0.20 \
  --lambda_consistency_ready 0.18 \
  --init_ckpt "$STAGE1A_DEPLOY" \
  | tee "$RUN12_STAGE1B_OUT_DIR/stdout.log"
