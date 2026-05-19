#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_from_scratch_raw_rollout_oraclecollect_3ep_20260507a}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/depth_force_contact/local_proposal_formal_depthforce_layered_k8_20260503a/checkpoints/final_full.pt}"
SUPPORT_STATES_OUTPUT_NPZ="${SUPPORT_STATES_OUTPUT_NPZ:-$OUTPUT_ROOT/raw_rollout_support_states.npz}"

mkdir -p "$OUTPUT_ROOT"

exec env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --use_stage_aware_refiner \
  --stage_refiner_mode alignment \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --target_provider_mode teacher_oracle \
  --allow_privileged_runtime \
  --depth_force_clean_support \
  --depth_force_clean_support_with_refiner \
  --depth_force_clean_privileged_labels \
  --oracle_executed_align_collect \
  --oracle_executed_pregrasp_collect \
  --allow_alignment_without_close_intent \
  --max_alignment_corrections_per_window 20 \
  --disable_alignment_close_veto \
  --no_best_gif \
  --record_teacher_truth_metrics \
  --record_video \
  --record_gripper_trace \
  --write_episode_videos \
  --no_best_gif \
  --num_episodes 3 \
  --max_steps 340 \
  --episode_indices "$EPISODE_INDICES" \
  --output_dir "$OUTPUT_ROOT" \
  --support_states_output_npz "$SUPPORT_STATES_OUTPUT_NPZ" \
  --eval_seed 3407
