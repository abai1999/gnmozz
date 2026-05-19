#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/planner_only_baseline_3ep_20260504a}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"

mkdir -p "$OUTPUT_ROOT"

ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/depth_force_contact/local_proposal_formal_depthforce_layered_k8_20260503a/checkpoints/final_full.pt}"

exec env xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --modes learned_target_mainline \
  --num_episodes 3 \
  --max_steps 340 \
  --episode_indices "$EPISODE_INDICES" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "depth_force_local_proposal_planner_only_baseline_3ep" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --disable_alignment_pose \
  --disable_alignment_close_veto \
  --record_teacher_truth_metrics \
  --enforce_no_privileged_runtime \
  --eval_seed 3407
