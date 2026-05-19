#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DIR="${RAW_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/raw_near_micro_20260511a}"
TEACHER_DIR="${TEACHER_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/teacher_20260511a}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_20260511a}"
SHADOW_DIR="${SHADOW_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/shadow_3ep_20260511a}"

RAW_OUTPUT_NPZ="${RAW_OUTPUT_NPZ:-$RAW_DIR/alignment_tc_diffusion_raw_near_micro_20260511a.npz}" \
OUTPUT_DIR="$RAW_DIR" \
"$ROOT/scripts/run_alignment_tc_diffusion_collect_raw_near_micro.sh"

INPUT_NPZ="$RAW_OUTPUT_NPZ" \
OUTPUT_NPZ="$TEACHER_DIR/alignment_tc_diffusion_teacher_20260511a.npz" \
REPORT_JSON="$TEACHER_DIR/alignment_tc_diffusion_teacher_report_20260511a.json" \
"$ROOT/scripts/run_alignment_tc_diffusion_build_teacher.sh"

DATASET="$TEACHER_DIR/alignment_tc_diffusion_teacher_20260511a.npz" \
OUTPUT_DIR="$TRAIN_DIR" \
"$ROOT/scripts/run_alignment_tc_diffusion_train.sh"

ALIGNMENT_TC_DIFFUSION_CKPT="$TRAIN_DIR/alignment_tc_diffusion_refiner_best.pt" \
OUTPUT_DIR="$SHADOW_DIR" \
"$ROOT/scripts/run_alignment_tc_diffusion_shadow_3ep.sh"
