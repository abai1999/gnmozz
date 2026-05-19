#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260515n/alignment_tc_student_vnext_best.pt}"
CORRIDOR_JSON="${CORRIDOR_JSON:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_20260515n_unified/alignment_tc_student_vnext_dataset_report_20260515n_unified.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_shadow_3ep_20260515n}"
EVAL_SEED="${EVAL_SEED:-3407}"

mkdir -p "$OUTPUT_DIR"
exec "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --alignment_tc_student_vnext_ckpt "$CKPT" \
  --enable_alignment_tc_student_vnext_shadow \
  --alignment_tc_student_vnext_corridor_json "$CORRIDOR_JSON" \
  --alignment_tc_diffusion_confidence_threshold 0.25 \
  --alignment_tc_diffusion_risk_threshold 0.85 \
  --alignment_tc_diffusion_soft_clamp \
  --alignment_tc_diffusion_workspace_soft_clamp \
  --eval_seed "$EVAL_SEED" \
  --record_video \
  --write_episode_videos \
  --no_best_gif \
  --output_dir "$OUTPUT_DIR"
