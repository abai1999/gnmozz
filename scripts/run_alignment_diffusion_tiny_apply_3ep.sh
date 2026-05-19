#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-runtime_artifacts/checkpoints/final_full.pt}"
ALIGNMENT_DIFFUSION_CKPT="${ALIGNMENT_DIFFUSION_CKPT:-}"
TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODES="${EPISODES:-5,8,19}"
MAX_STEPS="${MAX_STEPS:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-runtime_artifacts/alignment_diffusion/tiny_apply_3ep}"

if [[ -z "${ALIGNMENT_DIFFUSION_CKPT}" ]]; then
  echo "Set ALIGNMENT_DIFFUSION_CKPT=/path/to/alignment_diffusion_refiner_best.pt" >&2
  exit 2
fi

python scripts/evaluate_rlbench.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --task_name "${TASK_NAME}" \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --alignment_diffusion_ckpt "${ALIGNMENT_DIFFUSION_CKPT}" \
  --enable_alignment_diffusion_shadow \
  --enable_alignment_diffusion_apply \
  --alignment_diffusion_horizon 8 \
  --alignment_diffusion_num_samples 16 \
  --alignment_diffusion_apply_mode additive \
  --alignment_diffusion_max_pos_step 0.0010 \
  --alignment_diffusion_max_yaw_step 0.0040 \
  --alignment_diffusion_risk_threshold 0.65 \
  --alignment_diffusion_trigger_mode near_contact_stall \
  --max_steps "${MAX_STEPS}" \
  --episode_indices "${EPISODES}" \
  --record_video \
  --output_dir "${OUTPUT_DIR}"
