#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_runtime_shadow_rawdelta_3ep_20260506b}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"

ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/depth_force_contact/local_proposal_formal_depthforce_layered_k8_20260503a/checkpoints/final_full.pt}"
ALIGNMENT_V3_SHADOW_CKPT="${ALIGNMENT_V3_SHADOW_CKPT:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_direct_local_formal_run2_20260506c_privileged_teacher/alignment_v3_direct_local_best.pt}"
TD_CKPT="${TD_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"

mkdir -p "$OUTPUT_ROOT"

exec env xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --modes learned_target_mainline \
  --num_episodes 3 \
  --max_steps 340 \
  --episode_indices "$EPISODE_INDICES" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "alignment_v3_runtime_shadow_rawdelta_3ep" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --alignment_v3_shadow_ckpt "$ALIGNMENT_V3_SHADOW_CKPT" \
  --target_provider_mode learned \
  --target_provider_ckpt "$TD_CKPT" \
  --disable_learned_target_close_stage_orientation_contract \
  --record_teacher_truth_metrics \
  --enforce_no_privileged_runtime \
  --enable_alignment_near_zone_gate \
  --alignment_near_zone_xy_threshold 0.05 \
  --alignment_near_zone_z_threshold 0.10 \
  --learned_residual_scale 0.25 \
  --max_residual_pos 0.0025 \
  --max_residual_rot 0.015 \
  --max_alignment_corrections_per_window 120 \
  --eval_seed 3407
