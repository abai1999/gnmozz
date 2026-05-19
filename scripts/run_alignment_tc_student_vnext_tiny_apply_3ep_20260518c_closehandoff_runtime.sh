#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
export LD_PRELOAD="${LD_PRELOAD:-/home/guoning/my_conda_envs/vla-adapter/lib/libstdc++.so.6}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
CKPT="${CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage2_20260518b_closehandoff/alignment_tc_student_vnext_best.pt}"
CORRIDOR_JSON="${CORRIDOR_JSON:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_20260518a_closebridge/alignment_tc_student_vnext_dataset_report_20260518a_closebridge.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_tiny_apply_3ep_20260518c_closehandoff_runtime}"
EVAL_SEED="${EVAL_SEED:-3407}"

mkdir -p "$OUTPUT_DIR"
exec env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --alignment_tc_student_vnext_ckpt "$CKPT" \
  --enable_alignment_tc_student_vnext_shadow \
  --enable_alignment_tc_student_vnext_apply \
  --alignment_tc_student_vnext_collector_like \
  --enable_alignment_tc_student_vnext_ready_gate \
  --alignment_tc_student_vnext_close_ready_threshold 0.5 \
  --alignment_tc_student_vnext_handoff_ready_threshold 0.5 \
  --alignment_tc_student_vnext_corridor_json "$CORRIDOR_JSON" \
  --target_provider_mode canonical_fallback \
  --enforce_no_privileged_runtime \
  --alignment_tc_diffusion_confidence_threshold 0.25 \
  --alignment_tc_diffusion_risk_threshold 0.85 \
  --alignment_tc_diffusion_soft_clamp \
  --alignment_tc_diffusion_workspace_soft_clamp \
  --alignment_tc_diffusion_execute_steps 1 \
  --collector_like_demo_reset \
  --eval_seed "$EVAL_SEED" \
  --record_video \
  --write_episode_videos \
  --no_best_gif \
  --record_gripper_trace \
  --episode_indices 5,8,19 \
  --output_dir "$OUTPUT_DIR"
