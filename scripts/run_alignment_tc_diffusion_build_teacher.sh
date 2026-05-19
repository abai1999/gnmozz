#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

INPUT_NPZ="${INPUT_NPZ:-$ROOT/runtime_artifacts/alignment_diffusion/raw_near_contact_3ep_20260511a/alignment_diffusion_raw_near_contact_3ep.npz}"
OUTPUT_NPZ="${OUTPUT_NPZ:-$ROOT/runtime_artifacts/alignment_tc_diffusion/teacher_20260511a/alignment_tc_diffusion_teacher_20260511a.npz}"
REPORT_JSON="${REPORT_JSON:-$ROOT/runtime_artifacts/alignment_tc_diffusion/teacher_20260511a/alignment_tc_diffusion_teacher_report_20260511a.json}"
HORIZON="${HORIZON:-8}"
MAX_POS_STEP="${MAX_POS_STEP:-0.0015}"
MAX_YAW_STEP="${MAX_YAW_STEP:-0.0060}"
STAGE_BUCKETS="${STAGE_BUCKETS:-near_contact_refine,micro_contact_refine,broad_near}"
PREFER_COLLECTED_TEACHER_TRAJECTORY="${PREFER_COLLECTED_TEACHER_TRAJECTORY:-1}"
MIN_ROWS="${MIN_ROWS:-32}"
if [[ "$PREFER_COLLECTED_TEACHER_TRAJECTORY" == "0" ]]; then
  PREFER_COLLECTED_TEACHER_TRAJECTORY_ARG="--no_prefer_collected_teacher_trajectory"
else
  PREFER_COLLECTED_TEACHER_TRAJECTORY_ARG="--prefer_collected_teacher_trajectory"
fi

exec "$PYTHON_BIN" "$ROOT/scripts/build_alignment_tc_diffusion_teacher.py" \
  --input_npz "$INPUT_NPZ" \
  --output_npz "$OUTPUT_NPZ" \
  --report_json "$REPORT_JSON" \
  --horizon "$HORIZON" \
  --max_pos_step "$MAX_POS_STEP" \
  --max_yaw_step "$MAX_YAW_STEP" \
  --stage_buckets "$STAGE_BUCKETS" \
  "$PREFER_COLLECTED_TEACHER_TRAJECTORY_ARG" \
  --min_rows "$MIN_ROWS"
