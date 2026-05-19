#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TAG="${TAG:-20260515n_unified}"

PHASE1_RAW="${PHASE1_RAW:-$ROOT/runtime_artifacts/alignment_tc_diffusion/alignment_tc_planner_state_expert_recovery_raw_20260514n_zonly80_fixed_merged_raw.npz}"
PHASE2_RAW="${PHASE2_RAW:-$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_20260515n_targeted200/gpu2_ep0_n32/alignment_tc_planner_phase2_insert_teacher_gpu2_ep0_n32.npz $ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_20260515n_targeted200/gpu3_ep32_n32/alignment_tc_planner_phase2_insert_teacher_gpu3_ep32_n32.npz $ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_20260515n_targeted200/gpu4_ep64_n32/alignment_tc_planner_phase2_insert_teacher_gpu4_ep64_n32.npz $ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_20260515n_targeted200/gpu5_ep96_n32/alignment_tc_planner_phase2_insert_teacher_gpu5_ep96_n32.npz $ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_20260515n_targeted200/gpu6_ep128_n32/alignment_tc_planner_phase2_insert_teacher_gpu6_ep128_n32.npz $ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_20260515n_targeted200/gpu7_ep160_n40/alignment_tc_planner_phase2_insert_teacher_gpu7_ep160_n40.npz}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_${TAG}}"

mkdir -p "$OUT_ROOT"

exec "$PYTHON_BIN" "$ROOT/scripts/build_alignment_tc_student_vnext_dataset.py" \
  --phase1_raw "$PHASE1_RAW" \
  --phase2_raw $PHASE2_RAW \
  --output_npz "$OUT_ROOT/alignment_tc_student_vnext_dataset_${TAG}.npz" \
  --report_json "$OUT_ROOT/alignment_tc_student_vnext_dataset_report_${TAG}.json"
