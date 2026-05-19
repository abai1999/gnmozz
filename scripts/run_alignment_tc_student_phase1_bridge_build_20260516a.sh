#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
RAW_PHASE1="${RAW_PHASE1:-runtime_artifacts/alignment_tc_diffusion/alignment_tc_planner_state_expert_recovery_raw_20260514n_zonly80_fixed_merged_raw.npz}"
OUT_DIR="${OUT_DIR:-runtime_artifacts/alignment_tc_diffusion/student_phase1_bridge_dataset_20260516a}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" scripts/build_alignment_tc_student_phase1_bridge_dataset.py \
  --phase1_raw "$RAW_PHASE1" \
  --output_npz "$OUT_DIR/alignment_tc_student_phase1_bridge_dataset_20260516a.npz" \
  --report_json "$OUT_DIR/alignment_tc_student_phase1_bridge_dataset_report_20260516a.json" \
  --horizon 8 \
  --n_pre 8 \
  --n_align 16 \
  --n_post 8 \
  --max_negative_ratio 0.30
