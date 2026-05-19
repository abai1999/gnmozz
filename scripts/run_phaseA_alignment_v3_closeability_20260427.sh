#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_closeability_20260427}"
RECOLLECT_SUPPORT_NPZ="${RECOLLECT_SUPPORT_NPZ:-$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260427d_current30k_fixdebug/recollection/shard00_gpu6/support_states_shard00_gpu6.npz}"
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
DATASET_DIR="$OUT_ROOT/dataset"
TRAIN_DIR="$OUT_ROOT/train"
STAGEA_DIR="$TRAIN_DIR/stageA_pairwise_progress"
STAGEB_DIR="$TRAIN_DIR/stageB_pairwise_counterfactual"
MAIN_CANDIDATE_CKPT="$STAGEA_DIR/student_handoff_state_head_v2_alignment_v3_main_candidate.pt"
PROGRESS_BASELINE_CKPT="$STAGEB_DIR/student_handoff_state_head_v2_alignment_v3_progress_baseline.pt"

mkdir -p "$DATASET_DIR" "$STAGEA_DIR" "$STAGEB_DIR"

echo "[alignment-v3] OUT_ROOT=$OUT_ROOT"
echo "[alignment-v3] RECOLLECT_SUPPORT_NPZ=$RECOLLECT_SUPPORT_NPZ"
echo "[alignment-v3] RECOLLECT_SUPPORT_NPZ_EXTRA=${RECOLLECT_SUPPORT_NPZ_EXTRA:-}"
echo "[alignment-v3] INIT_CKPT=$INIT_CKPT"

[[ -f "$RECOLLECT_SUPPORT_NPZ" ]] || { echo "ERROR: support npz missing: $RECOLLECT_SUPPORT_NPZ" >&2; exit 1; }
[[ -f "$INIT_CKPT" ]] || { echo "ERROR: init ckpt missing: $INIT_CKPT" >&2; exit 1; }

input_args=(--input_npz "$RECOLLECT_SUPPORT_NPZ")
if [[ -n "${RECOLLECT_SUPPORT_NPZ_EXTRA:-}" ]]; then
  IFS=':' read -r -a EXTRA_NPZS <<< "$RECOLLECT_SUPPORT_NPZ_EXTRA"
  for extra_npz in "${EXTRA_NPZS[@]}"; do
    [[ -z "$extra_npz" ]] && continue
    [[ -f "$extra_npz" ]] || { echo "ERROR: extra support npz missing: $extra_npz" >&2; exit 1; }
    input_args+=(--input_npz "$extra_npz")
  done
fi

"$PYTHON_BIN" "$ROOT/scripts/build_phaseA_alignment_v3_closeability_dataset.py" \
  "${input_args[@]}" \
  --output_dir "$DATASET_DIR" \
  --min_boundary_rows "${MIN_BOUNDARY_ROWS:-32}" \
  --strict_near_norm "${STRICT_NEAR_NORM:-1.35}" \
  --broad_near_norm "${BROAD_NEAR_NORM:-3.0}" \
  --far_negative_min_norm "${FAR_NEGATIVE_MIN_NORM:-6.0}" \
  --counterfactual_multiplier "${COUNTERFACTUAL_MULTIPLIER:-1.0}" \
  --counterfactual_weight "${COUNTERFACTUAL_WEIGHT:-0.5}" \
  --progress_k "${PROGRESS_K:-3}" \
  --progress_weighted_sum_delta_margin "${PROGRESS_WEIGHTED_SUM_DELTA_MARGIN:-0.10}" \
  --progress_max_axis_delta_margin "${PROGRESS_MAX_AXIS_DELTA_MARGIN:-0.10}" \
  --pair_weighted_sum_delta_margin "${PAIR_WEIGHTED_SUM_DELTA_MARGIN:-0.08}" \
  --pair_max_axis_delta_margin "${PAIR_MAX_AXIS_DELTA_MARGIN:-0.08}" \
  --progress_negative_weight_mult "${PROGRESS_NEGATIVE_WEIGHT_MULT:-2.0}" \
  --boundary_negative_weight_mult "${BOUNDARY_NEGATIVE_WEIGHT_MULT:-2.5}" \
  --temporal_summary_horizon "${TEMPORAL_SUMMARY_HORIZON:-3}"

STAGEA_NPZ="$DATASET_DIR/handoff_state_dataset_v2_alignment_v3_stageA.npz"
FULL_NPZ="$DATASET_DIR/handoff_state_dataset_v2_alignment_v3_full.npz"

"$PYTHON_BIN" "$ROOT/scripts/audit_phaseA_alignment_v3_dataset.py" \
  --dataset_npz "$FULL_NPZ" \
  --output_json "$OUT_ROOT/alignment_v3_dataset_report.json"

"$PYTHON_BIN" "$ROOT/scripts/train_phaseA_alignment_v3_closeability.py" \
  --dataset_npz "$STAGEA_NPZ" \
  --output_dir "$STAGEA_DIR" \
  --init_ckpt "$INIT_CKPT" \
  --epochs "${STAGEA_EPOCHS:-4}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --lr "${STAGEA_LR:-1e-5}" \
  --weighted_sampling \
  --lambda_xy "${LAMBDA_XY:-1.6}" \
  --lambda_z "${LAMBDA_Z:-0.9}" \
  --lambda_yaw "${LAMBDA_YAW:-1.1}" \
  --lambda_band "${STAGEA_LAMBDA_BAND:-0.35}" \
  --lambda_ready "${STAGEA_LAMBDA_READY:-0.02}" \
  --lambda_progress "${STAGEA_LAMBDA_PROGRESS:-0.05}" \
  --lambda_pair_rank "${STAGEA_LAMBDA_PAIR_RANK:-0.80}" \
  --pair_rank_margin "${PAIR_RANK_MARGIN:-0.25}" \
  --lambda_axis "${STAGEA_LAMBDA_AXIS:-0.25}" \
  --lambda_closeability "${STAGEA_LAMBDA_CLOSEABILITY:-0.30}" \
  --lambda_corrective "${STAGEA_LAMBDA_CORRECTIVE:-0.20}" \
  --lambda_score "${STAGEA_LAMBDA_SCORE:-0.10}" \
  --lambda_far_negative_ready "${STAGEA_LAMBDA_FAR_NEG_READY:-0.10}" \
  --closeability_pos_weight "${CLOSEABILITY_POS_WEIGHT:-1.5}" \
  --closeability_gate_min "${CLOSEABILITY_GATE_MIN:-0.58}" \
  --corrective_gate_min "${CORRECTIVE_GATE_MIN:-0.50}" \
  --use_pairwise_gate \
  --use_calibrated_pair_gate \
  --pair_gate_min "${PAIR_GATE_MIN:-0.60}" \
  --pair_pos_recall_gate_min "${PAIR_POS_RECALL_GATE_MIN:-0.45}" \
  --pair_neg_recall_gate_min "${PAIR_NEG_RECALL_GATE_MIN:-0.45}"

STAGEA_BEST="$STAGEA_DIR/student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt"

"$PYTHON_BIN" "$ROOT/scripts/train_phaseA_alignment_v3_closeability.py" \
  --dataset_npz "$FULL_NPZ" \
  --output_dir "$STAGEB_DIR" \
  --init_ckpt "$STAGEA_BEST" \
  --epochs "${STAGEB_EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --lr "${STAGEB_LR:-5e-6}" \
  --weighted_sampling \
  --lambda_xy "${LAMBDA_XY:-1.6}" \
  --lambda_z "${LAMBDA_Z:-0.9}" \
  --lambda_yaw "${LAMBDA_YAW:-1.1}" \
  --lambda_band "${STAGEB_LAMBDA_BAND:-0.30}" \
  --lambda_ready "${STAGEB_LAMBDA_READY:-0.02}" \
  --lambda_progress "${STAGEB_LAMBDA_PROGRESS:-0.05}" \
  --lambda_pair_rank "${STAGEB_LAMBDA_PAIR_RANK:-0.70}" \
  --pair_rank_margin "${PAIR_RANK_MARGIN:-0.25}" \
  --lambda_axis "${STAGEB_LAMBDA_AXIS:-0.20}" \
  --lambda_closeability "${STAGEB_LAMBDA_CLOSEABILITY:-0.35}" \
  --lambda_corrective "${STAGEB_LAMBDA_CORRECTIVE:-0.25}" \
  --lambda_score "${STAGEB_LAMBDA_SCORE:-0.10}" \
  --lambda_far_negative_ready "${STAGEB_LAMBDA_FAR_NEG_READY:-0.20}" \
  --closeability_pos_weight "${CLOSEABILITY_POS_WEIGHT:-1.5}" \
  --closeability_gate_min "${CLOSEABILITY_GATE_MIN:-0.58}" \
  --corrective_gate_min "${CORRECTIVE_GATE_MIN:-0.50}" \
  --use_pairwise_gate \
  --use_calibrated_pair_gate \
  --pair_gate_min "${PAIR_GATE_MIN:-0.60}" \
  --pair_pos_recall_gate_min "${PAIR_POS_RECALL_GATE_MIN:-0.45}" \
  --pair_neg_recall_gate_min "${PAIR_NEG_RECALL_GATE_MIN:-0.45}"

cp "$DATASET_DIR/alignment_v3_dataset_report.json" "$OUT_ROOT/alignment_v3_dataset_report.json"
cp "$STAGEB_DIR/alignment_v3_train_history.json" "$OUT_ROOT/alignment_v3_train_history.json"
cp "$STAGEB_DIR/alignment_v3_gate_report.json" "$OUT_ROOT/alignment_v3_gate_report.json"

cp "$STAGEA_BEST" "$MAIN_CANDIDATE_CKPT"
cp "$STAGEB_DIR/student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt" "$PROGRESS_BASELINE_CKPT"

"$PYTHON_BIN" "$ROOT/scripts/compare_phaseA_alignment_v3_candidates.py" \
  --stagea_gate "$STAGEA_DIR/alignment_v3_gate_report.json" \
  --stageb_gate "$STAGEB_DIR/alignment_v3_gate_report.json" \
  --stagea_ckpt "$MAIN_CANDIDATE_CKPT" \
  --stageb_ckpt "$PROGRESS_BASELINE_CKPT" \
  --output_json "$OUT_ROOT/alignment_v3_candidate_comparison.json"

"$PYTHON_BIN" "$ROOT/scripts/audit_phaseA_alignment_v3_dataset.py" \
  --dataset_npz "$FULL_NPZ" \
  --output_json "$OUT_ROOT/alignment_v3_final_dataset_audit.json"

"$PYTHON_BIN" "$ROOT/scripts/summarize_phaseA_alignment_v3_reports.py" \
  --root "$OUT_ROOT" \
  --dataset_report "$DATASET_DIR/alignment_v3_dataset_report.json" \
  --teacher_audit "$OUT_ROOT/teacher_student_distill_chain_audit.json" \
  --stagea_gate "$STAGEA_DIR/alignment_v3_gate_report.json" \
  --stageb_gate "$STAGEB_DIR/alignment_v3_gate_report.json" \
  --candidate_compare "$OUT_ROOT/alignment_v3_candidate_comparison.json"

echo "[alignment-v3] complete"
echo "[alignment-v3] gate_report=$OUT_ROOT/alignment_v3_gate_report.json"
echo "[alignment-v3] main_candidate=$MAIN_CANDIDATE_CKPT"
echo "[alignment-v3] progress_baseline=$PROGRESS_BASELINE_CKPT"
