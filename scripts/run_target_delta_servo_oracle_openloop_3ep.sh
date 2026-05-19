#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/target_delta_servo_oracle_openloop_3ep_20260506a}"
MODE_NAME="insert_vo40k_learned_target_mainline_target_delta_servo_oracle_openloop_3ep"
CKPT="/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt"
ALIGNMENT_CKPT="$ROOT/runtime_artifacts/depth_force_contact/local_proposal_formal_depthforce_layered_k8_20260503a/checkpoints/final_full.pt"
TARGET_PROVIDER_CKPT=""

mkdir -p "$OUT_ROOT"

python "$ROOT/scripts/evaluate_rlbench_modes.py" \
  --checkpoint_dir "$CKPT" \
  --task_name insert_onto_square_peg \
  --modes learned_target_mainline \
  --num_episodes 3 \
  --max_steps 340 \
  --run_full_horizon_on_success \
  --output_root "$OUT_ROOT" \
  --name_suffix "oracle_openloop_3ep" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --target_provider_mode teacher_oracle \
  --record_teacher_truth_metrics \
  --allow_privileged_runtime \
  --enable_alignment_near_zone_gate \
  --enable_target_delta_servo_apply \
  --target_delta_servo_bypass_gates \
  --target_delta_servo_apply_once_per_episode \
  --target_delta_servo_source privileged_replay \
  --target_delta_servo_k_xy 0.16 \
  --target_delta_servo_k_z 0.12 \
  --target_delta_servo_k_yaw 0.08 \
  --target_delta_servo_max_pos 0.0025 \
  --target_delta_servo_max_yaw 0.0100
