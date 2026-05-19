#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

PHASEA_CKPT="${PHASEA_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
BASELINE_SCORER_CKPT="${BASELINE_SCORER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"

DATASET_DIR="${DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/b1b2_actioncentric_dataset_v1_20260423x}"
DATASET_NPZ="${DATASET_NPZ:-$DATASET_DIR/b1b2_actioncentric_dataset_v1.npz}"
DATASET_META="${DATASET_META:-$DATASET_DIR/b1b2_actioncentric_dataset_v1.meta.json}"
INPUT_NPZ_CSV="${INPUT_NPZ_CSV:-$ROOT/runtime_artifacts/residual_data/insert_phase1_near_ready_candidate_learned_full32_20260421d/near_ready_candidates.npz}"

mkdir -p "$DATASET_DIR"

INPUT_ARGS=()
IFS=',' read -ra INPUTS <<< "$INPUT_NPZ_CSV"
for p in "${INPUTS[@]}"; do
  [[ -n "$p" ]] && INPUT_ARGS+=(--input_npz "$p")
done

python "$ROOT/scripts/build_b1b2_actioncentric_dataset_v1.py" \
  "${INPUT_ARGS[@]}" \
  --output_npz "$DATASET_NPZ" \
  --meta_json "$DATASET_META" \
  --min_teacher_ready_eps "${MIN_TEACHER_READY_EPS:-3}" \
  --min_teacher_ready_rows "${MIN_TEACHER_READY_ROWS:-20}" \
  --min_yaw_needed_eps "${MIN_YAW_NEEDED_EPS:-2}" \
  ${ALLOW_INSUFFICIENT_DATASET:+--allow_insufficient}

B1_OUT="${B1_OUT:-$ROOT/runtime_artifacts/stage_refiner/student_group_selector_v2_b1_actioncentric_v1_20260423x}"
mkdir -p "$B1_OUT"

python "$ROOT/scripts/train_student_group_selector_v2.py" \
  --dataset_npz "$DATASET_NPZ" \
  --handoff_state_ckpt "$PHASEA_CKPT" \
  --baseline_scorer_ckpt "$BASELINE_SCORER_CKPT" \
  --output_dir "$B1_OUT" \
  --epochs "${B1_EPOCHS:-8}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --lr "${B1_LR:-2e-4}" \
  --val_ratio "${VAL_RATIO:-0.2}" \
  --seed "${SEED:-3407}" \
  2>&1 | tee "$B1_OUT/stdout.log"

if [[ "${RUN_B2:-0}" == "1" ]]; then
  B2_OUT="${B2_OUT:-$ROOT/runtime_artifacts/stage_refiner/student_candidate_evaluator_v2_b2_actioncentric_v1_20260423x}"
  mkdir -p "$B2_OUT"
  python "$ROOT/scripts/train_student_candidate_evaluator_v2.py" \
    --dataset_npz "$DATASET_NPZ" \
    --handoff_state_ckpt "$PHASEA_CKPT" \
    --output_dir "$B2_OUT" \
    --epochs "${B2_EPOCHS:-8}" \
    --batch_size "${BATCH_SIZE:-64}" \
    --lr "${B2_LR:-2e-4}" \
    --val_ratio "${VAL_RATIO:-0.2}" \
    --seed "${SEED:-3407}" \
    2>&1 | tee "$B2_OUT/stdout.log"
fi

echo "[b1b2-actioncentric-v1] dataset=$DATASET_NPZ"
echo "[b1b2-actioncentric-v1] b1_out=$B1_OUT"
if [[ "${RUN_B2:-0}" == "1" ]]; then
  echo "[b1b2-actioncentric-v1] b2_out=$B2_OUT"
fi
