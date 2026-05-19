#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
MAX_STEPS="${MAX_STEPS:-340}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_diffusion/raw_near_contact_3ep_20260511a}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"
RAW_OUTPUT_NPZ="${RAW_OUTPUT_NPZ:-$OUTPUT_DIR/alignment_diffusion_raw_near_contact_3ep.npz}"
RAW_REPORT_JSON="${RAW_REPORT_JSON:-$OUTPUT_DIR/alignment_diffusion_raw_near_contact_3ep_report.json}"

mkdir -p "$OUTPUT_DIR"

exec env xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --episode_indices "$EPISODE_INDICES" \
  --target_provider_mode canonical_fallback \
  --enforce_no_privileged_runtime \
  --output_dir "$OUTPUT_DIR" \
  --record_gripper_trace \
  --no_video \
  --no_episode_videos \
  --no_best_gif \
  --max_steps "$MAX_STEPS" \
  --alignment_diffusion_raw_output_npz "$RAW_OUTPUT_NPZ" \
  --alignment_diffusion_raw_report_json "$RAW_REPORT_JSON" \
  --alignment_diffusion_raw_horizon 8 \
  --alignment_diffusion_raw_augment_copies 1 \
  --eval_seed 3407
