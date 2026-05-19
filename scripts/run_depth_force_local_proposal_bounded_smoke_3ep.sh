#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/depth_force_local_proposal_final_full_bounded_smoke_3ep_20260503a}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/depth_force_contact/local_proposal_formal_depthforce_layered_k8_20260503a/checkpoints/final_full.pt}"

mkdir -p "$OUTPUT_ROOT"

exec env xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --modes learned_target_mainline \
  --num_episodes 3 \
  --max_steps 340 \
  --episode_indices "$EPISODE_INDICES" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "depth_force_local_proposal_final_full_bounded_smoke_3ep" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --student_handoff_shadow_only \
  --record_teacher_truth_metrics \
  --enforce_no_privileged_runtime \
  --enable_alignment_close_veto \
  --close_veto_xy_threshold 0.006 \
  --close_veto_abs_z_threshold 0.003 \
  --close_veto_ready_streak_frames 1 \
  --close_veto_settle_steps 0 \
  --enable_bounded_auto_close_on_alignment \
  --bounded_auto_close_stable_frames 1 \
  --bounded_auto_close_xy_threshold 0.006 \
  --bounded_auto_close_abs_z_threshold 0.003 \
  --bounded_auto_close_yaw_threshold 0.12 \
  --close_latch_enabled \
  --close_latch_steps 32 \
  --learned_residual_scale 1.0 \
  --max_residual_pos 0.006 \
  --max_residual_rot 0.030 \
  --max_alignment_corrections_per_window 120 \
  --eval_seed 3407 \
  --no_video \
  --no_episode_videos
