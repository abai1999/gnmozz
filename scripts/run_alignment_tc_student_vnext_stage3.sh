#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_20260515n_unified/alignment_tc_student_vnext_dataset_20260515n_unified.npz}"
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage2_20260515n/alignment_tc_student_vnext_best.pt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260515n}"

mkdir -p "$OUT_ROOT"
exec "$PYTHON_BIN" "$ROOT/scripts/train_alignment_tc_student_vnext.py" \
  --dataset "$DATASET" \
  --output_dir "$OUT_ROOT" \
  --stage stage3_student_finetune \
  --init_ckpt "$INIT_CKPT"
