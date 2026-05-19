#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
CKPT="${CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260517a_edgepair/alignment_tc_student_vnext_best.pt}"
CORRIDOR_JSON="${CORRIDOR_JSON:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_20260517a_edgepair/alignment_tc_student_vnext_dataset_report_20260517a_edgepair.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_tiny_apply_3ep_20260517a_edgepair}"
COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
CONDA_LIBSTDCXX="${CONDA_LIBSTDCXX:-/home/guoning/my_conda_envs/vla-adapter/lib/libstdc++.so.6}"

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export XAUTHORITY="${XAUTHORITY:-}"
export LD_PRELOAD="${LD_PRELOAD:-$CONDA_LIBSTDCXX}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

if command -v xvfb-run >/dev/null 2>&1; then
  RUN_PREFIX=(xvfb-run -a)
else
  RUN_PREFIX=()
fi

exec env \
  PYTHONPATH="$PYTHONPATH" \
  QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
  QT_PLUGIN_PATH="$QT_PLUGIN_PATH" \
  LIBGL_ALWAYS_SOFTWARE="$LIBGL_ALWAYS_SOFTWARE" \
  XAUTHORITY="$XAUTHORITY" \
  LD_PRELOAD="$LD_PRELOAD" \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  "${RUN_PREFIX[@]}" \
  "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt" \
  --task_name insert_onto_square_peg \
  --target_provider_mode canonical_fallback \
  --enforce_no_privileged_runtime \
  --alignment_tc_student_vnext_ckpt "$CKPT" \
  --enable_alignment_tc_student_vnext_shadow \
  --enable_alignment_tc_student_vnext_apply \
  --alignment_tc_student_vnext_corridor_json "$CORRIDOR_JSON" \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --record_video \
  --write_episode_videos \
  --no_best_gif \
  --record_gripper_trace \
  --output_dir "$OUTPUT_DIR" \
  --episode_indices 5,8,19 \
  --eval_seed 3407 \
  --phase1_force_reflex_enable \
  --alignment_tc_diffusion_confidence_threshold 0.25 \
  --alignment_tc_diffusion_risk_threshold 0.85 \
  --alignment_tc_diffusion_soft_clamp \
  --alignment_tc_diffusion_workspace_soft_clamp
