#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/target_delta_servo_predictor_direct_3ep_20260506a}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"

ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/depth_force_contact/local_proposal_formal_depthforce_layered_k8_20260503a/checkpoints/final_full.pt}"
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
  --name_suffix "target_delta_servo_predictor_direct_3ep" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --target_provider_mode learned \
  --target_provider_ckpt "$TD_CKPT" \
  --record_teacher_truth_metrics \
  --enforce_no_privileged_runtime \
  --disable_alignment_close_veto \
  --enable_target_delta_servo_apply \
  --target_delta_servo_bypass_gates \
  --target_delta_servo_source predictor \
  --target_delta_servo_k_xy 0.12 \
  --target_delta_servo_k_z 0.09 \
  --target_delta_servo_k_yaw 0.06 \
  --target_delta_servo_max_pos 0.0015 \
  --target_delta_servo_max_yaw 0.0060 \
  --target_delta_servo_apply_xy_threshold 0.03 \
  --target_delta_servo_apply_abs_z_threshold 0.07 \
  --target_delta_servo_apply_yaw_threshold 0.25 \
  --eval_seed 3407
