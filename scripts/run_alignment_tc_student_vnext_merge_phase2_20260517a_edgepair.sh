#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
TAG="${TAG:-20260517a_edgepair}"
OUT_DIR="${OUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_${TAG}_merged}"

PHASE2_SHARDS=(
  "$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_${TAG}_s0/alignment_tc_planner_phase2_insert_teacher_${TAG}_s0.npz"
  "$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_${TAG}_s1/alignment_tc_planner_phase2_insert_teacher_${TAG}_s1.npz"
  "$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_${TAG}_s2/alignment_tc_planner_phase2_insert_teacher_${TAG}_s2.npz"
  "$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_${TAG}_s3/alignment_tc_planner_phase2_insert_teacher_${TAG}_s3.npz"
)

mkdir -p "$OUT_DIR"

exec "$PYTHON_BIN" "$ROOT/scripts/merge_npz_rowwise.py" \
  --output "$OUT_DIR/alignment_tc_planner_phase2_insert_teacher_${TAG}.npz" \
  "${PHASE2_SHARDS[@]}"
