#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_ranker_20260429}"
DATASET_DIR="${DATASET_DIR:-$OUTPUT_DIR/dataset}"
HANDOFF_CKPT="${HANDOFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_teacher_augmented_20260429b_dyaw_aux_full/train/stageA_pairwise_progress/student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt}"
SUPPLEMENT_NPZ="${SUPPLEMENT_NPZ:-}"

INPUT_NPZ_1="${INPUT_NPZ_1:-$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260427d_current30k_fixdebug/recollection/shard00_gpu6/support_states_shard00_gpu6.npz}"
INPUT_NPZ_2="${INPUT_NPZ_2:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_heldout_compare_20260429c/stageA_main_candidate/support_states.npz}"

mkdir -p "$OUTPUT_DIR" "$DATASET_DIR"
cd "$ROOT"

"$PYTHON_BIN" scripts/build_alignment_v4_candidate_ranking_dataset.py \
  --input_npz "$INPUT_NPZ_1" \
  --input_npz "$INPUT_NPZ_2" \
  --output_dir "$DATASET_DIR" \
  --candidate_score_std_min "${CANDIDATE_SCORE_STD_MIN:-0.5}" \
  --oracle_baseline_gap_min "${ORACLE_BASELINE_GAP_MIN:-1.0}" \
  --opportunity_policy "${OPPORTUNITY_POLICY:-close_like_or_gap}" \
  --keep_yaw_abs "${KEEP_YAW_ABS:-0.02}" \
  "$@"

TRAIN_CMD=(
  "$PYTHON_BIN" scripts/train_alignment_v4b_candidate_ranker.py
  --dataset_npz "$DATASET_DIR/alignment_v4_candidate_ranking_dataset.npz"
  --handoff_ckpt "$HANDOFF_CKPT"
  --output_dir "$OUTPUT_DIR/train"
  --epochs "${EPOCHS:-40}"
  --batch_size "${BATCH_SIZE:-16}"
  --lr "${LR:-3e-4}"
  --keep_yaw_abs "${KEEP_YAW_ABS:-0.02}"
  --pairwise_weight "${PAIRWISE_WEIGHT:-0.6}"
  --value_weight "${VALUE_WEIGHT:-0.2}"
  --best_ce_weight "${BEST_CE_WEIGHT:-0.6}"
  --mode_weight "${MODE_WEIGHT:-0.2}"
  --baseline_pair_weight "${BASELINE_PAIR_WEIGHT:-0.5}"
  --hard_negative_weight "${HARD_NEGATIVE_WEIGHT:-1.0}"
  --bad_yaw_pair_weight "${BAD_YAW_PAIR_WEIGHT:-2.0}"
  --large_yaw_negative_weight "${LARGE_YAW_NEGATIVE_WEIGHT:-2.0}"
  --yaw_positive_weight "${YAW_POSITIVE_WEIGHT:-0.5}"
  --mode_input_path "${MODE_INPUT_PATH:-summary_only}"
  --device "${DEVICE:-cuda}"
)
if [[ -n "$SUPPLEMENT_NPZ" ]]; then
  TRAIN_CMD+=(--dataset_npz "$SUPPLEMENT_NPZ")
fi
"${TRAIN_CMD[@]}"
