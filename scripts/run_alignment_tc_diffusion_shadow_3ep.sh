#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
ALIGNMENT_TC_DIFFUSION_CKPT="${ALIGNMENT_TC_DIFFUSION_CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_20260514n_verified_only/alignment_tc_diffusion_refiner_best.pt}"
TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODES="${EPISODES:-5,8,19}"
MAX_STEPS="${MAX_STEPS:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_shadow_20260515a_correct_ckpt_3ep}"

exec env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --alignment_tc_diffusion_ckpt "$ALIGNMENT_TC_DIFFUSION_CKPT" \
  --enable_alignment_tc_diffusion_shadow \
  --alignment_tc_diffusion_num_samples 8 \
  --alignment_tc_diffusion_top_k 3 \
  --alignment_tc_diffusion_confidence_threshold 0.55 \
  --alignment_tc_diffusion_risk_threshold 0.65 \
  --alignment_tc_diffusion_soft_clamp \
  --alignment_diffusion_max_pos_step 0.0008 \
  --alignment_diffusion_max_yaw_step 0.0030 \
  --alignment_diffusion_trigger_mode near_contact_stall \
  --max_steps "$MAX_STEPS" \
  --episode_indices "$EPISODES" \
  --record_video \
  --output_dir "$OUTPUT_DIR"
