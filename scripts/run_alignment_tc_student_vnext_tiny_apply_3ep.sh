#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
CKPT="${CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260515n/alignment_tc_student_vnext_best.pt}"
CORRIDOR_JSON="${CORRIDOR_JSON:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_20260515n_unified/alignment_tc_student_vnext_dataset_report_20260515n_unified.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_tiny_apply_3ep_20260515n}"
EVAL_SEED="${EVAL_SEED:-3407}"

mkdir -p "$OUTPUT_DIR"
exec "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --alignment_tc_student_vnext_ckpt "$CKPT" \
  --enable_alignment_tc_student_vnext_shadow \
  --enable_alignment_tc_student_vnext_apply \
  --alignment_tc_student_vnext_corridor_json "$CORRIDOR_JSON" \
  --target_provider_mode canonical_fallback \
  --enforce_no_privileged_runtime \
  --alignment_tc_diffusion_confidence_threshold 0.25 \
  --alignment_tc_diffusion_risk_threshold 0.85 \
  --alignment_tc_diffusion_soft_clamp \
  --alignment_tc_diffusion_workspace_soft_clamp \
  --alignment_tc_diffusion_execute_steps 1 \
  --eval_seed "$EVAL_SEED" \
  --episode_indices 5,8,19 \
  --record_video \
  --write_episode_videos \
  --no_best_gif \
  --output_dir "$OUTPUT_DIR"
