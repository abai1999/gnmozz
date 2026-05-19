#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
PYTHON_BIN="/home/guoning/my_conda_envs/vla-adapter/bin/python"
GPU_ID="${GPU_ID:-0}"

INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422i/train_run9_stage1b_softmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
ANCHOR_DATASET_FULL="${ANCHOR_DATASET_FULL:-$ROOT/runtime_artifacts/stage_refiner/teacher_success_anchor_stage0_20260422e/teacher_success_anchor_dataset.npz}"

ANCHOR_WINDOW_DIR="${ANCHOR_WINDOW_DIR:-$ROOT/runtime_artifacts/stage_refiner/teacher_success_anchor_window_run9_20260422i}"
ANCHOR_WINDOW_NPZ="${ANCHOR_WINDOW_NPZ:-$ANCHOR_WINDOW_DIR/teacher_success_anchor_window_dataset.npz}"

RUN10_STAGE1A_DATASET_DIR="${RUN10_STAGE1A_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run10_stage1a_20260422j}"
RUN10_STAGE1A_OUT_DIR="${RUN10_STAGE1A_OUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422j/train_run10_stage1a_anchorwarm}"

RUN10_STAGE1B_BASE_DATASET_DIR="${RUN10_STAGE1B_BASE_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run10_stage1b_base_20260422j}"
RUN10_STAGE1B_MERGED_DATASET_DIR="${RUN10_STAGE1B_MERGED_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run10_stage1b_merged_20260422j}"
RUN10_STAGE1B_OUT_DIR="${RUN10_STAGE1B_OUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422j/train_run10_stage1b_softmix}"

mkdir -p "$RUN10_STAGE1A_DATASET_DIR" "$RUN10_STAGE1A_OUT_DIR" "$RUN10_STAGE1B_BASE_DATASET_DIR" "$RUN10_STAGE1B_MERGED_DATASET_DIR" "$RUN10_STAGE1B_OUT_DIR"

if [[ ! -s "$ANCHOR_WINDOW_NPZ" ]]; then
  echo "[run10] build anchor-window dataset"
  "$PYTHON_BIN" "$ROOT/scripts/build_teacher_success_anchor_window_dataset.py" \
    --input_npz "$ANCHOR_DATASET_FULL" \
    --output_npz "$ANCHOR_WINDOW_NPZ" \
    --meta_json "$ANCHOR_WINDOW_DIR/teacher_success_anchor_window_dataset.meta.json"
fi

echo "[run10-stage1a] build anchor-biased support mix"
"$PYTHON_BIN" "$ROOT/scripts/build_student_handoff_state_dataset_v2.py" \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_parallel8_20260421m/support_states_merged.npz" \
  --source_name lateprofile_parallel8 \
  --source_weight_mult 1.60 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_readypos_parallel8_20260422d/support_states_merged.npz" \
  --source_name lateprofile_readypos_parallel8 \
  --source_weight_mult 1.90 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_learned16_20260421k/support_states.npz" \
  --source_name targeted16 \
  --source_weight_mult 1.10 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_resync_learned_full32_20260421h/support_states.npz" \
  --source_name learned32 \
  --source_weight_mult 0.75 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_xyrecovery_parallel7_20260421k2/support_states_merged.npz" \
  --source_name xyrecovery_parallel7 \
  --source_weight_mult 0.35 \
  --output_npz "$RUN10_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_supportmix.npz" \
  --meta_json "$RUN10_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_supportmix.meta.json"

echo "[run10-stage1a] merge support mix + anchor windows"
"$PYTHON_BIN" "$ROOT/scripts/merge_student_handoff_state_datasets_v2.py" \
  --dataset_npz "$RUN10_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_supportmix.npz" \
  --weight_mult 1.0 \
  --dataset_npz "$ANCHOR_WINDOW_NPZ" \
  --weight_mult 2.30 \
  --output_npz "$RUN10_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --meta_json "$RUN10_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_mixed.meta.json"

echo "[run10-stage1a] train"
env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$PYTHON_BIN" "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$RUN10_STAGE1A_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_dir "$RUN10_STAGE1A_OUT_DIR" \
  --epochs 3 \
  --batch_size 64 \
  --lr 2e-4 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_band 0.72 \
  --lambda_ready 0.18 \
  --lambda_uncertainty 0.0 \
  --lambda_xy 1.95 \
  --lambda_z 0.42 \
  --lambda_yaw 0.95 \
  --weighted_sampling \
  --sampler_weight_power 1.10 \
  --deploy_false_ready_max 0.01 \
  --init_ckpt "$INIT_CKPT" | tee "$RUN10_STAGE1A_OUT_DIR/stdout.log"

STAGE1A_DEPLOY="$RUN10_STAGE1A_OUT_DIR/student_handoff_state_head_v2_best_phaseA_deploy.pt"
if [[ ! -s "$STAGE1A_DEPLOY" ]]; then
  STAGE1A_DEPLOY="$RUN10_STAGE1A_OUT_DIR/student_handoff_state_head_v2_best_anchor.pt"
fi
if [[ ! -s "$STAGE1A_DEPLOY" ]]; then
  STAGE1A_DEPLOY="$RUN10_STAGE1A_OUT_DIR/student_handoff_state_head_v2_best.pt"
fi

echo "[run10-stage1b] build soft full mix"
"$PYTHON_BIN" "$ROOT/scripts/build_student_handoff_state_dataset_v2.py" \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_resync_learned_full32_20260421h/support_states.npz" \
  --source_name learned32 \
  --source_weight_mult 0.72 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_learned16_20260421k/support_states.npz" \
  --source_name targeted16 \
  --source_weight_mult 1.25 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_xyrecovery_parallel7_20260421k2/support_states_merged.npz" \
  --source_name xyrecovery_parallel7 \
  --source_weight_mult 0.55 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_parallel8_20260421m/support_states_merged.npz" \
  --source_name lateprofile_parallel8 \
  --source_weight_mult 1.65 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_next_parallel8_20260422c/support_states_merged.npz" \
  --source_name lateprofile_next_parallel8 \
  --source_weight_mult 1.00 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_readypos_parallel8_20260422d/support_states_merged.npz" \
  --source_name lateprofile_readypos_parallel8 \
  --source_weight_mult 1.85 \
  --support_npz "$ROOT/runtime_artifacts/residual_data/insert_phase1_truthready_positive_support_20260420i/support_states.npz" \
  --source_name truthready_positive \
  --source_weight_mult 0.90 \
  --output_npz "$RUN10_STAGE1B_BASE_DATASET_DIR/handoff_state_dataset_v2_supportmix.npz" \
  --meta_json "$RUN10_STAGE1B_BASE_DATASET_DIR/handoff_state_dataset_v2_supportmix.meta.json"

echo "[run10-stage1b] merge soft full mix + anchor windows"
"$PYTHON_BIN" "$ROOT/scripts/merge_student_handoff_state_datasets_v2.py" \
  --dataset_npz "$RUN10_STAGE1B_BASE_DATASET_DIR/handoff_state_dataset_v2_supportmix.npz" \
  --weight_mult 1.0 \
  --dataset_npz "$ANCHOR_WINDOW_NPZ" \
  --weight_mult 1.70 \
  --output_npz "$RUN10_STAGE1B_MERGED_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --meta_json "$RUN10_STAGE1B_MERGED_DATASET_DIR/handoff_state_dataset_v2_mixed.meta.json"

echo "[run10-stage1b] train"
env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "$PYTHON_BIN" "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$RUN10_STAGE1B_MERGED_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_dir "$RUN10_STAGE1B_OUT_DIR" \
  --epochs 10 \
  --batch_size 64 \
  --lr 8e-5 \
  --val_ratio 0.2 \
  --seed 3407 \
  --lambda_band 0.72 \
  --lambda_ready 0.18 \
  --lambda_uncertainty 0.0 \
  --lambda_xy 2.00 \
  --lambda_z 0.45 \
  --lambda_yaw 0.95 \
  --weighted_sampling \
  --sampler_weight_power 1.15 \
  --deploy_false_ready_max 0.01 \
  --consistency_ckpt "$STAGE1A_DEPLOY" \
  --consistency_source teacher_success_formal30 \
  --consistency_source lateprofile_ready_anchor \
  --consistency_source truthready_anchor \
  --lambda_consistency_band 0.12 \
  --lambda_consistency_ready 0.10 \
  --init_ckpt "$STAGE1A_DEPLOY" | tee "$RUN10_STAGE1B_OUT_DIR/stdout.log"
