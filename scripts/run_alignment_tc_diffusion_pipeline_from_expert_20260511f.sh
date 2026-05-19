#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DIR="${RAW_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/privileged_expert_demo_perturb_20260511f_both_phase_expert_pilot}"
TEACHER_DIR="${TEACHER_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/teacher_20260511f_from_expert}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_20260511f_from_expert}"
SHADOW_DIR="${SHADOW_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/shadow_3ep_20260511f_from_expert}"

INPUT_NPZ="${INPUT_NPZ:-$RAW_DIR/alignment_tc_privileged_expert_demo_perturb_20260511f_both_phase_expert_pilot.npz}"
TEACHER_NPZ="${TEACHER_NPZ:-$TEACHER_DIR/alignment_tc_diffusion_teacher_20260511f_from_expert.npz}"
TEACHER_REPORT="${TEACHER_REPORT:-$TEACHER_DIR/alignment_tc_diffusion_teacher_report_20260511f_from_expert.json}"

RAW_OUTPUT_NPZ="$INPUT_NPZ" \
INPUT_NPZ="$INPUT_NPZ" \
OUTPUT_NPZ="$TEACHER_NPZ" \
REPORT_JSON="$TEACHER_REPORT" \
"$ROOT/scripts/run_alignment_tc_diffusion_build_teacher.sh"

DATASET="$TEACHER_NPZ" \
OUTPUT_DIR="$TRAIN_DIR" \
"$ROOT/scripts/run_alignment_tc_diffusion_train.sh"

ALIGNMENT_TC_DIFFUSION_CKPT="$TRAIN_DIR/alignment_tc_diffusion_refiner_best.pt" \
OUTPUT_DIR="$SHADOW_DIR" \
"$ROOT/scripts/run_alignment_tc_diffusion_shadow_3ep.sh"
