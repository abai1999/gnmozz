#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/v2_assist_C_v2_nearzone_3ep_20260505a}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/depth_force_contact/local_proposal_formal_depthforce_layered_k8_20260503a/checkpoints/final_full.pt}"
V2_CKPT="${V2_CKPT:-$ROOT/runtime_artifacts/depth_force_contact/target_conditioned_alignment_v2_nearmicro_run1_20260504a/target_conditioned_alignment_v2_best.pt}"
HANDOFF_CKPT="${HANDOFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
mkdir -p "$OUTPUT_ROOT"
exec env xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" --task_name "$TASK_NAME" \
  --modes learned_target_mainline --num_episodes 3 --max_steps 340 \
  --episode_indices "$EPISODE_INDICES" --output_root "$OUTPUT_ROOT" \
  --name_suffix "v2_assist_C_v2_nearzone_3ep" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --v2_alignment_shadow_ckpt "$V2_CKPT" \
  --handoff_provider_ckpt "$HANDOFF_CKPT" \
  --student_handoff_shadow_only --record_teacher_truth_metrics \
  --enforce_no_privileged_runtime \
  --enable_alignment_near_zone_gate \
  --alignment_near_zone_xy_threshold 0.05 --alignment_near_zone_z_threshold 0.10 \
  --enable_v2_nearzone_assist \
  --v2_assist_scale_cap 0.25 --v2_assist_max_pos 0.0025 --v2_assist_max_rot 0.015 \
  --learned_residual_scale 0.25 --max_residual_pos 0.0025 --max_residual_rot 0.015 \
  --max_alignment_corrections_per_window 120 --eval_seed 3407
