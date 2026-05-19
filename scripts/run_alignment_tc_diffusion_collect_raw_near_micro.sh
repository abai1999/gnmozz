#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
MAX_STEPS="${MAX_STEPS:-340}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/raw_near_micro_20260511a}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"
RAW_OUTPUT_NPZ="${RAW_OUTPUT_NPZ:-$OUTPUT_DIR/alignment_tc_diffusion_raw_near_micro_20260511a.npz}"
RAW_REPORT_JSON="${RAW_REPORT_JSON:-$OUTPUT_DIR/alignment_tc_diffusion_raw_near_micro_report_20260511a.json}"

mkdir -p "$OUTPUT_DIR"

exec env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --episode_indices "$EPISODE_INDICES" \
  --target_provider_mode canonical_fallback \
  --enforce_no_privileged_runtime \
  --record_teacher_truth_metrics \
  --output_dir "$OUTPUT_DIR" \
  --record_gripper_trace \
  --no_video \
  --no_episode_videos \
  --no_best_gif \
  --max_steps "$MAX_STEPS" \
  --alignment_diffusion_raw_output_npz "$RAW_OUTPUT_NPZ" \
  --alignment_diffusion_raw_report_json "$RAW_REPORT_JSON" \
  --alignment_diffusion_raw_horizon 8 \
  --alignment_diffusion_raw_near_depth_threshold 0.085 \
  --alignment_diffusion_raw_micro_depth_threshold 0.045 \
  --alignment_tc_raw_privileged_target_mode commit \
  --alignment_tc_raw_use_privileged_delta_gate \
  --alignment_tc_raw_near_xy_threshold 0.030 \
  --alignment_tc_raw_near_abs_z_threshold 0.070 \
  --alignment_tc_raw_near_yaw_threshold 0.35 \
  --alignment_tc_raw_micro_xy_threshold 0.010 \
  --alignment_tc_raw_micro_abs_z_threshold 0.030 \
  --alignment_tc_raw_micro_yaw_threshold 0.18 \
  --alignment_diffusion_raw_augment_copies 1 \
  --eval_seed 3407
