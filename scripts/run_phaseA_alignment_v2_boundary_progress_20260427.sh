#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v2_boundary_progress_20260427}"
RECOLLECT_SUPPORT_NPZ="${RECOLLECT_SUPPORT_NPZ:-$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260427d_current30k_fixdebug/recollection/shard00_gpu6/support_states_shard00_gpu6.npz}"
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
DATASET_DIR="${DATASET_DIR:-$OUT_ROOT/dataset}"
TRAIN_DIR="${TRAIN_DIR:-$OUT_ROOT/train}"
STAGEA_DIR="$TRAIN_DIR/stageA_geometry_boundary"
STAGEB_DIR="$TRAIN_DIR/stageB_counterfactual_calibration"

mkdir -p "$DATASET_DIR" "$STAGEA_DIR" "$STAGEB_DIR"

echo "[alignment-v2] OUT_ROOT=$OUT_ROOT"
echo "[alignment-v2] RECOLLECT_SUPPORT_NPZ=$RECOLLECT_SUPPORT_NPZ"
echo "[alignment-v2] INIT_CKPT=$INIT_CKPT"

if [[ ! -f "$RECOLLECT_SUPPORT_NPZ" ]]; then
  echo "ERROR: support npz missing: $RECOLLECT_SUPPORT_NPZ" >&2
  exit 1
fi
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: init ckpt missing: $INIT_CKPT" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT/scripts/build_phaseA_alignment_v2_boundary_progress_dataset.py" \
  --input_npz "$RECOLLECT_SUPPORT_NPZ" \
  --output_dir "$DATASET_DIR" \
  --min_boundary_rows "${MIN_BOUNDARY_ROWS:-32}" \
  --strict_near_norm "${STRICT_NEAR_NORM:-1.35}" \
  --broad_near_norm "${BROAD_NEAR_NORM:-3.0}" \
  --far_negative_min_norm "${FAR_NEGATIVE_MIN_NORM:-6.0}" \
  --counterfactual_multiplier "${COUNTERFACTUAL_MULTIPLIER:-1.0}" \
  --counterfactual_weight "${COUNTERFACTUAL_WEIGHT:-0.5}"

STAGEA_NPZ="$DATASET_DIR/handoff_state_dataset_v2_alignment_v2_stageA.npz"
FULL_NPZ="$DATASET_DIR/handoff_state_dataset_v2_alignment_v2_full.npz"

"$PYTHON_BIN" "$ROOT/scripts/train_phaseA_alignment_v2_boundary_progress.py" \
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
  --lambda_progress "${STAGEA_LAMBDA_PROGRESS:-0.25}" \
  --lambda_score "${STAGEA_LAMBDA_SCORE:-0.20}" \
  --lambda_far_negative_ready "${STAGEA_LAMBDA_FAR_NEG_READY:-0.10}" \
  --progress_gate_min "${PROGRESS_GATE_MIN:-0.60}"

STAGEA_BEST="$STAGEA_DIR/student_handoff_state_head_v2_alignment_v2_best_deploy_candidate.pt"

"$PYTHON_BIN" "$ROOT/scripts/train_phaseA_alignment_v2_boundary_progress.py" \
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
  --lambda_progress "${STAGEB_LAMBDA_PROGRESS:-0.20}" \
  --lambda_score "${STAGEB_LAMBDA_SCORE:-0.20}" \
  --lambda_far_negative_ready "${STAGEB_LAMBDA_FAR_NEG_READY:-0.20}" \
  --progress_gate_min "${PROGRESS_GATE_MIN:-0.60}"

cp "$DATASET_DIR/alignment_v2_dataset_report.json" "$OUT_ROOT/alignment_v2_dataset_report.json"
cp "$STAGEB_DIR/alignment_v2_train_history.json" "$OUT_ROOT/alignment_v2_train_history.json"
cp "$STAGEB_DIR/alignment_v2_gate_report.json" "$OUT_ROOT/alignment_v2_gate_report.json"

"$PYTHON_BIN" "$ROOT/scripts/summarize_phaseA_alignment_v2_reports.py" \
  --root "$OUT_ROOT" \
  --dataset_report "$DATASET_DIR/alignment_v2_dataset_report.json" \
  --stagea_gate "$STAGEA_DIR/alignment_v2_gate_report.json" \
  --stageb_gate "$STAGEB_DIR/alignment_v2_gate_report.json"

echo "[alignment-v2] complete"
echo "[alignment-v2] dataset_report=$OUT_ROOT/alignment_v2_dataset_report.json"
echo "[alignment-v2] gate_report=$OUT_ROOT/alignment_v2_gate_report.json"
echo "[alignment-v2] deploy_candidate=$STAGEB_DIR/student_handoff_state_head_v2_alignment_v2_best_deploy_candidate.pt"

if [[ "${RUN_SHADOW:-0}" == "1" ]]; then
  echo "[alignment-v2] RUN_SHADOW=1 requested; use scripts/run_phaseA_runtime_ready_shadow_eval_parallel_20260426a.sh with CANDIDATE ckpt manually after offline gate review."
fi
