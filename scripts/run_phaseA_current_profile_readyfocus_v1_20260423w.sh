#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

BASE_MAIN_V1="${BASE_MAIN_V1:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_main_v1_20260423u_yawseed/handoff_state_dataset_v1.npz}"
POS_JSON="${POS_JSON:-$ROOT/runtime_artifacts/stage_refiner/phaseA_current_profile_ready_scan_20260423r/current_profile_teacher_ready_positive_set_20260423r.json}"
RECOLLECTION_JSON="${RECOLLECTION_JSON:-$ROOT/runtime_artifacts/stage_refiner/phaseA_current_profile_ready_scan_20260423r/current_profile_ready_recollection_plan_20260423r.json}"

OUT_DATA_DIR="${OUT_DATA_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_main_v1_current_profile_readyfocus_20260423w}"
OUT_TRAIN_DIR="${OUT_TRAIN_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_main_v1_current_profile_readyfocus_20260423w/train_readyfocus}"
mkdir -p "$OUT_DATA_DIR" "$OUT_TRAIN_DIR"

READYFOCUS_NPZ="$OUT_DATA_DIR/handoff_state_dataset_v1_current_profile_readyfocus.npz"
READYFOCUS_META="$OUT_DATA_DIR/handoff_state_dataset_v1_current_profile_readyfocus.meta.json"
READYFOCUS_VAL_CSV="$OUT_DATA_DIR/val_episode_csv_current_profile_readyfocus.txt"

python "$ROOT/scripts/build_phaseA_current_profile_ready_focus_v1.py" \
  --input_npz "$BASE_MAIN_V1" \
  --current_profile_positive_json "$POS_JSON" \
  --current_profile_recollection_json "$RECOLLECTION_JSON" \
  --output_npz "$READYFOCUS_NPZ" \
  --meta_json "$READYFOCUS_META" \
  --val_episode_csv_out "$READYFOCUS_VAL_CSV" \
  --positive_val_episode_csv "${POSITIVE_VAL_EPISODE_CSV:-11}" \
  --hard_negative_episode_csv "${HARD_NEGATIVE_EPISODE_CSV:-14}"

VAL_EPISODES_CSV="$(tr -d '\n' < "$READYFOCUS_VAL_CSV")"

env \
  DATASET_NPZ="$READYFOCUS_NPZ" \
  OUTPUT_DIR="$OUT_TRAIN_DIR" \
  VAL_EPISODES_CSV="$VAL_EPISODES_CSV" \
  bash "$ROOT/scripts/run_phaseA_readyfirst_minimal_v1_20260423u.sh"

echo "[phaseA-current-profile-readyfocus-v1] dataset=$READYFOCUS_NPZ val_eps=$VAL_EPISODES_CSV train_dir=$OUT_TRAIN_DIR"
