#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_ranker_20260430_v2_2}"
DATASET_DIR="${DATASET_DIR:-$OUTPUT_DIR/dataset}"
SUPPLEMENT_DIR="${SUPPLEMENT_DIR:-$OUTPUT_DIR/supplement}"
HANDOFF_CKPT="${HANDOFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_teacher_augmented_20260429b_dyaw_aux_full/train/stageA_pairwise_progress/student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt}"

INPUT_NPZ_1="${INPUT_NPZ_1:-$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260427d_current30k_fixdebug/recollection/shard00_gpu6/support_states_shard00_gpu6.npz}"
INPUT_NPZ_2="${INPUT_NPZ_2:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_heldout_compare_20260429c/stageA_main_candidate/support_states.npz}"
SUPPORT_NPZ_A="${SUPPORT_NPZ_A:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_shadow_20260430_batches/batchA_8_10_rerun/support_states.npz}"
SUPPORT_NPZ_B="${SUPPORT_NPZ_B:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_shadow_20260430_batches/batchB_16_17/support_states.npz}"
SUPPORT_NPZ_C="${SUPPORT_NPZ_C:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_shadow_20260430_batches/batchC_19_20/support_states.npz}"
SUPPORT_NPZ_D="${SUPPORT_NPZ_D:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_shadow_20260430_v2_1_modepatch/support_states.npz}"

mkdir -p "$OUTPUT_DIR" "$DATASET_DIR" "$SUPPLEMENT_DIR"
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

"$PYTHON_BIN" scripts/build_alignment_v4b_typed_hard_negative_supplement.py \
  --support_npz "$SUPPORT_NPZ_A" \
  --support_npz "$SUPPORT_NPZ_B" \
  --support_npz "$SUPPORT_NPZ_C" \
  --support_npz "$SUPPORT_NPZ_D" \
  --output_dir "$SUPPLEMENT_DIR" \
  --keep_yaw_abs "${KEEP_YAW_ABS:-0.02}" \
  --small_yaw_abs "${SMALL_YAW_ABS:-0.05}" \
  --large_yaw_abs "${LARGE_YAW_ABS:-0.09}" \
  --mode_keep_margin "${MODE_KEEP_MARGIN:-0.05}" \
  --worse_weight "${HARD_NEGATIVE_WEIGHT:-1.0}" \
  --better_weight "${BETTER_WEIGHT:-0.5}" \
  --baseline_preserve_weight "${BASELINE_PRESERVE_WEIGHT:-0.75}" \
  --hard_episode_indices "${HARD_EPISODE_INDICES:-17}" \
  --hard_episode_weight "${HARD_EPISODE_WEIGHT:-3.0}" \
  --candidate_score_std_min "${CANDIDATE_SCORE_STD_MIN:-0.5}" \
  --oracle_baseline_gap_min "${ORACLE_BASELINE_GAP_MIN:-1.0}" \
  --mode_keep_failure_weight "${MODE_KEEP_FAILURE_WEIGHT:-2.5}" \
  --mode_apply_failure_weight "${MODE_APPLY_FAILURE_WEIGHT:-2.0}" \
  --large_yaw_negative_weight "${LARGE_YAW_NEGATIVE_WEIGHT:-2.0}" \
  --wrong_yaw_sign_weight "${WRONG_YAW_SIGN_WEIGHT:-2.0}" \
  --xy_over_yaw_weight "${XY_OVER_YAW_WEIGHT:-2.5}" \
  --z_over_yaw_weight "${Z_OVER_YAW_WEIGHT:-2.5}" \
  --yaw_needed_but_not_selected_weight "${YAW_NEEDED_BUT_NOT_SELECTED_WEIGHT:-1.8}" \
  --include_candidate_bank_missing

"$PYTHON_BIN" scripts/train_alignment_v4b_candidate_ranker.py \
  --dataset_npz "$DATASET_DIR/alignment_v4_candidate_ranking_dataset.npz" \
  --dataset_npz "$SUPPLEMENT_DIR/alignment_v4b_typed_hard_negative_supplement.npz" \
  --handoff_ckpt "$HANDOFF_CKPT" \
  --output_dir "$OUTPUT_DIR/train" \
  --epochs "${EPOCHS:-40}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --lr "${LR:-3e-4}" \
  --keep_yaw_abs "${KEEP_YAW_ABS:-0.02}" \
  --mode_apply_margin "${MODE_APPLY_MARGIN:-0.05}" \
  --small_yaw_abs "${SMALL_YAW_ABS:-0.05}" \
  --large_yaw_abs "${LARGE_YAW_ABS:-0.09}" \
  --pairwise_weight "${PAIRWISE_WEIGHT:-0.6}" \
  --value_weight "${VALUE_WEIGHT:-0.2}" \
  --best_ce_weight "${BEST_CE_WEIGHT:-0.6}" \
  --mode_weight "${MODE_WEIGHT:-0.2}" \
  --baseline_pair_weight "${BASELINE_PAIR_WEIGHT:-0.5}" \
  --hard_negative_weight "${HARD_NEGATIVE_WEIGHT:-1.0}" \
  --bad_yaw_pair_weight "${BAD_YAW_PAIR_WEIGHT:-2.0}" \
  --large_yaw_negative_weight "${LARGE_YAW_NEGATIVE_WEIGHT:-2.0}" \
  --large_yaw_positive_weight "${LARGE_YAW_POSITIVE_WEIGHT:-1.0}" \
  --small_vs_large_yaw_weight "${SMALL_VS_LARGE_YAW_WEIGHT:-1.5}" \
  --hard_episode_negative_weight "${HARD_EPISODE_NEGATIVE_WEIGHT:-3.0}" \
  --hard_episode_indices "${HARD_EPISODE_INDICES:-17}" \
  --yaw_positive_weight "${YAW_POSITIVE_WEIGHT:-0.7}" \
  --mode_input_path "${MODE_INPUT_PATH:-summary_only}" \
  --device "${DEVICE:-cuda}"
