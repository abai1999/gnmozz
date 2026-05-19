#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_teacher_augmented_20260429b_dyaw_aux_full/train/stageA_pairwise_progress/student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt}"
V4B_CANDIDATE_CKPT="${V4B_CANDIDATE_CKPT:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_ranker_20260429/train/student_candidate_evaluator_v4b_best.pt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_shadow_20260429}"
NAME_SUFFIX="${NAME_SUFFIX:-alignment_v4b_candidate_shadow}"
EPISODE_INDICES="${EPISODE_INDICES:-1,5,8,10,16,17,19,20,22,24,25,27}"
NUM_EPISODES="${NUM_EPISODES:-12}"
MAX_STEPS="${MAX_STEPS:-340}"
EVAL_SEED="${EVAL_SEED:-3407}"
SHADOW_GATE_MODE="${SHADOW_GATE_MODE:-nearish_only}"
SHADOW_YAW_PROBES="${SHADOW_YAW_PROBES:-0.06,0.12}"
SUPPORT_STATES_OUTPUT_NPZ="${SUPPORT_STATES_OUTPUT_NPZ:-$OUTPUT_ROOT/support_states.npz}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/ssd/guoning/hf-cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HF_CACHE_ROOT/torch}"
export TIMM_HOME="${TIMM_HOME:-$HF_CACHE_ROOT/timm}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0"
fi

cd "$ROOT"

env xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name insert_onto_square_peg \
  --modes learned_target_mainline \
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --episode_indices "$EPISODE_INDICES" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "$NAME_SUFFIX" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
  --handoff_provider_ckpt "$HANDOFF_PROVIDER_CKPT" \
  --student_handoff_shadow_only \
  --student_candidate_evaluator_shadow_ckpt "$V4B_CANDIDATE_CKPT" \
  --student_candidate_evaluator_handoff_ckpt "$HANDOFF_PROVIDER_CKPT" \
  --student_candidate_evaluator_mode_input_path "${MODE_INPUT_PATH:-summary_only}" \
  --b2_candidate_shadow_gate_mode "$SHADOW_GATE_MODE" \
  --b2_candidate_shadow_yaw_probe_values "$SHADOW_YAW_PROBES" \
  --planner_no_depth \
  --planner_no_force \
  --enable_alignment_close_veto \
  --close_veto_xy_threshold 0.006 \
  --close_veto_abs_z_threshold 0.003 \
  --close_veto_ready_streak_frames 1 \
  --close_veto_settle_steps 0 \
  --learned_residual_scale 0.50 \
  --max_residual_pos 0.006 \
  --max_alignment_corrections_per_window 120 \
  --outer_rescue_min_xy 0.10 \
  --outer_rescue_min_abs_z 0.30 \
  --eval_seed "$EVAL_SEED" \
  --close_latch_enabled \
  --close_latch_steps 32 \
  --use_legacy_teacher_candidate_bank_for_scorer \
  --disable_alignment_physical_mask \
  --record_teacher_truth_metrics \
  --enforce_no_privileged_runtime \
  --support_states_output_npz "$SUPPORT_STATES_OUTPUT_NPZ" \
  --record_video \
  --write_episode_videos \
  --no_best_gif

TRACE_DIR="$(find "$OUTPUT_ROOT" -maxdepth 1 -mindepth 1 -type d | head -n 1)"
if [[ -z "$TRACE_DIR" ]]; then
  echo "No trace subdirectory found under $OUTPUT_ROOT" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/analyze_b2_candidate_shadow_trace.py \
  --trace_dir "$TRACE_DIR" \
  --output_json "$OUTPUT_ROOT/alignment_v4b_shadow_summary.json"
