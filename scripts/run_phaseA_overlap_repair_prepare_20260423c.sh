#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

BASE_DATASET_DIR="${BASE_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_run12_stage1b_merged_20260422l}"
OUT_DIR="${OUT_DIR:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_dataset_v2_phaseA_overlaprepair_20260423c}"

mkdir -p "$OUT_DIR"

python "$ROOT/scripts/build_teacher_ready_overlap_repair_dataset.py" \
  --dataset_npz "$BASE_DATASET_DIR/handoff_state_dataset_v2_mixed.npz" \
  --output_npz "$OUT_DIR/handoff_state_dataset_v2_overlaprepair.npz" \
  --meta_json "$OUT_DIR/handoff_state_dataset_v2_overlaprepair.meta.json" \
  --focus_source learned32 \
  --teacher_ready_focus_episodes 8,28 \
  --calibration_negative_episodes 14 \
  --teacher_ready_window 12 \
  --teacher_ready_focus_mult 4.0 \
  --teacher_ready_exact_mult 2.0 \
  --calibration_negative_mult 0.5

echo "[overlap-repair] dataset prepared at $OUT_DIR"
