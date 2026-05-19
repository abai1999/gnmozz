#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4_residual_policy_20260429b}"
DATASET_DIR="$OUT_ROOT/dataset"
TRAIN_DIR="$OUT_ROOT/train"
mkdir -p "$DATASET_DIR" "$TRAIN_DIR"

INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_teacher_augmented_20260429b_dyaw_aux_full/train/stageA_pairwise_progress/student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt}"
V3_DATASET_NPZ="${V3_DATASET_NPZ:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_teacher_augmented_20260429b_dyaw_aux_full/dataset/handoff_state_dataset_v2_alignment_v3_full.npz}"

[[ -f "$INIT_CKPT" ]] || { echo "ERROR: init ckpt missing: $INIT_CKPT" >&2; exit 1; }
[[ -f "$V3_DATASET_NPZ" ]] || { echo "ERROR: v3 dataset missing: $V3_DATASET_NPZ" >&2; exit 1; }

input_args=(--input_npz "$V3_DATASET_NPZ")
if [[ -n "${EXTRA_SUPPORT_NPZ:-}" ]]; then
  IFS=':' read -r -a extra_npzs <<< "$EXTRA_SUPPORT_NPZ"
  for p in "${extra_npzs[@]}"; do
    [[ -z "$p" ]] && continue
    [[ -f "$p" ]] || { echo "ERROR: extra support missing: $p" >&2; exit 1; }
    input_args+=(--input_npz "$p")
  done
fi

echo "[alignment-v4] OUT_ROOT=$OUT_ROOT"
echo "[alignment-v4] INIT_CKPT=$INIT_CKPT"
echo "[alignment-v4] V3_DATASET_NPZ=$V3_DATASET_NPZ"

"$PYTHON_BIN" "$ROOT/scripts/build_alignment_v4_residual_dataset.py" \
  "${input_args[@]}" \
  --output_dir "$DATASET_DIR" \
  --max_residual_xyz "${MAX_RESIDUAL_XYZ:-0.006}" \
  --max_residual_yaw "${MAX_RESIDUAL_YAW:-0.03}" \
  --yaw_symmetry_period "${YAW_SYMMETRY_PERIOD:-1.5707963267948966}" \
  --improvement_margin "${IMPROVEMENT_MARGIN:-0.05}" \
  --temporal_summary_horizon "${TEMPORAL_SUMMARY_HORIZON:-3}" \
  --require_focus_mask \
  --require_closeability_positive \
  --max_teacher_xy_norm "${MAX_TEACHER_XY_NORM:-6.0}" \
  --max_teacher_z_norm "${MAX_TEACHER_Z_NORM:-12.0}" \
  --max_teacher_yaw_norm "${MAX_TEACHER_YAW_NORM:-2.5}" \
  --max_unclipped_residual_xyz_norm "${MAX_UNCLIPPED_RESIDUAL_XYZ_NORM:-0.03}" \
  --max_unclipped_residual_yaw_abs "${MAX_UNCLIPPED_RESIDUAL_YAW_ABS:-0.10}" \
  --allowed_phase_id "${ALLOWED_PHASE_IDS:-1}"

"$PYTHON_BIN" "$ROOT/scripts/audit_alignment_v4_residual_targets.py" \
  --dataset_npz "$DATASET_DIR/alignment_v4_residual_dataset.npz" \
  --output_json "$DATASET_DIR/alignment_v4_target_audit.json"

"$PYTHON_BIN" "$ROOT/scripts/train_alignment_v4_residual_policy.py" \
  --dataset_npz "$DATASET_DIR/alignment_v4_residual_dataset.npz" \
  --output_dir "$TRAIN_DIR" \
  --init_ckpt "$INIT_CKPT" \
  --epochs "${EPOCHS:-6}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --lr "${LR:-1e-5}" \
  --weighted_sampling \
  --lambda_xyz "${LAMBDA_XYZ:-1.0}" \
  --lambda_yaw "${LAMBDA_YAW:-0.5}" \
  --lambda_confidence "${LAMBDA_CONFIDENCE:-0.25}" \
  --lambda_closeability "${LAMBDA_CLOSEABILITY:-0.15}" \
  --lambda_progress "${LAMBDA_PROGRESS:-0.10}" \
  --gate_improvement_rate "${GATE_IMPROVEMENT_RATE:-0.60}" \
  --gate_xyz_direction_acc "${GATE_XYZ_DIRECTION_ACC:-0.55}" \
  --gate_closeability_balanced_acc "${GATE_CLOSEABILITY_BALANCED_ACC:-0.58}" \
  --gate_progress_balanced_acc "${GATE_PROGRESS_BALANCED_ACC:-0.55}"

cp "$DATASET_DIR/alignment_v4_dataset_report.json" "$OUT_ROOT/alignment_v4_dataset_report.json"
cp "$DATASET_DIR/alignment_v4_target_audit.json" "$OUT_ROOT/alignment_v4_target_audit.json"
cp "$TRAIN_DIR/alignment_v4_gate_report.json" "$OUT_ROOT/alignment_v4_gate_report.json"
cp "$TRAIN_DIR/alignment_v4_train_history.json" "$OUT_ROOT/alignment_v4_train_history.json"

echo "[alignment-v4] complete"
echo "[alignment-v4] gate_report=$OUT_ROOT/alignment_v4_gate_report.json"
echo "[alignment-v4] ckpt=$TRAIN_DIR/student_handoff_state_head_v2_alignment_v4_best_residual_policy.pt"
