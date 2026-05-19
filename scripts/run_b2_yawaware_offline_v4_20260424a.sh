#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CANDIDATE_DIR="${CANDIDATE_DIR:-$ROOT/runtime_artifacts/residual_data/b1b2_current_profile_candidate_v3_yawbank_20260424a}"
ACTIONCENTRIC_DIR="${ACTIONCENTRIC_DIR:-$ROOT/runtime_artifacts/stage_refiner/b1b2_actioncentric_dataset_v4_yawbank_20260424a}"
TRAIN_OUT="${TRAIN_OUT:-$ROOT/runtime_artifacts/stage_refiner/student_candidate_evaluator_v2_b2_yawbank_v4_20260424a}"
HANDOFF_CKPT="${HANDOFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"

mkdir -p "$ACTIONCENTRIC_DIR" "$TRAIN_OUT"

"$PYTHON_BIN" "$ROOT/scripts/build_b1b2_actioncentric_dataset_v1.py" \
  --input_npz "$CANDIDATE_DIR/learned32_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/lateprofile_readypos_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/lateprofile_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/oracleub_t2_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/teacher_assisted_v3_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/teacher_assisted_v3b_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/teacher_assisted_v3c_topup_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/teacher_assisted_yawneeded_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/lateprofile_next_candidates.npz" \
  --input_npz "$CANDIDATE_DIR/b1b2_recollect_y_candidates.npz" \
  --output_npz "$ACTIONCENTRIC_DIR/b1b2_actioncentric_dataset_v4_yawbank.npz" \
  --meta_json "$ACTIONCENTRIC_DIR/b1b2_actioncentric_dataset_v4_yawbank.meta.json" \
  --min_yaw_needed_eps 3

"$PYTHON_BIN" "$ROOT/scripts/train_student_candidate_evaluator_v2.py" \
  --dataset_npz "$ACTIONCENTRIC_DIR/b1b2_actioncentric_dataset_v4_yawbank.npz" \
  --handoff_state_ckpt "$HANDOFF_CKPT" \
  --output_dir "$TRAIN_OUT" \
  --candidate_scope yaw_aware \
  --epochs 8 \
  --batch_size 64 \
  --lr 2e-4

echo "[b2-yawaware-v4] dataset=$ACTIONCENTRIC_DIR/b1b2_actioncentric_dataset_v4_yawbank.npz"
echo "[b2-yawaware-v4] train_out=$TRAIN_OUT"
