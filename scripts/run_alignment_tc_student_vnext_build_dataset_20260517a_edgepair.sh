#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
TAG="${TAG:-20260517a_edgepair}"

PHASE1_RAW="${PHASE1_RAW:-$ROOT/runtime_artifacts/alignment_tc_diffusion/alignment_tc_planner_state_expert_recovery_raw_20260514n_zonly80_fixed_merged_raw.npz}"
PHASE2_RAW="${PHASE2_RAW:-$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_20260517a_edgepair_merged/alignment_tc_planner_phase2_insert_teacher_20260517a_edgepair.npz}"
PHASE1_DIAG_RAW="${PHASE1_DIAG_RAW:-$ROOT/runtime_artifacts/alignment_tc_diffusion/privileged_teacher_raw_20260517a_edgepair_grasp_10ep/alignment_tc_privileged_teacher_raw_20260517a_edgepair_grasp_10ep.npz}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_${TAG}}"

mkdir -p "$OUT_ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/guoning/my_conda_envs/vla-adapter/lib:${LD_LIBRARY_PATH:-}"

"$PYTHON_BIN" "$ROOT/scripts/build_alignment_tc_student_vnext_dataset.py" \
  --phase1_raw "$PHASE1_RAW" \
  --phase2_raw "$PHASE2_RAW" \
  --output_npz "$OUT_ROOT/alignment_tc_student_vnext_dataset_${TAG}.npz" \
  --report_json "$OUT_ROOT/alignment_tc_student_vnext_dataset_report_${TAG}.json"

"$PYTHON_BIN" "$ROOT/scripts/build_alignment_tc_student_vnext_edgepair_failure_split.py" \
  --raw "$PHASE1_DIAG_RAW" \
  --output_npz "$OUT_ROOT/diagnostic_failure/alignment_tc_student_vnext_edgepair_failure_split_${TAG}.npz" \
  --report_json "$OUT_ROOT/diagnostic_failure/alignment_tc_student_vnext_edgepair_failure_split_report_${TAG}.json"
