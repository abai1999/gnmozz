#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/guoning/code/VLA/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/guoning/code/VLA/eval_logs/insert_onto_square_peg/no_priv_student_runtime_20260419a}"
NAME_SUFFIX="${NAME_SUFFIX:-seed3407_nopriv_smoke_mp4}"
MODES="${MODES:-visual_scorer_mainline,learned_target_mainline}"
EPISODE_INDICES="${EPISODE_INDICES:-0,1,2}"
NUM_EPISODES="${NUM_EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_SEED="${EVAL_SEED:-3407}"

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
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if gpu="$("$REPO_ROOT/scripts/choose_idle_gpu.sh" 2>/dev/null)"; then
    export CUDA_VISIBLE_DEVICES="$gpu"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$REPO_ROOT"

CMD=(
  xvfb-run -a
  "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py
  --checkpoint_dir "$CHECKPOINT_DIR"
  --task_name insert_onto_square_peg
  --modes "$MODES"
  --num_episodes "$NUM_EPISODES"
  --max_steps "$MAX_STEPS"
  --episode_indices "$EPISODE_INDICES"
  --output_root "$OUTPUT_ROOT"
  --name_suffix "$NAME_SUFFIX"
  --alignment_ckpt "$ALIGNMENT_CKPT"
  --target_provider_ckpt "$TARGET_PROVIDER_CKPT"
  --planner_no_depth
  --planner_no_force
  --enable_alignment_close_veto
  --close_veto_xy_threshold 0.006
  --close_veto_abs_z_threshold 0.003
  --close_veto_ready_streak_frames 1
  --close_veto_settle_steps 0
  --learned_residual_scale 0.50
  --max_residual_pos 0.006
  --max_alignment_corrections_per_window 120
  --outer_rescue_min_xy 0.10
  --outer_rescue_min_abs_z 0.30
  --eval_seed "$EVAL_SEED"
  --close_latch_enabled
  --close_latch_steps 32
  --use_legacy_teacher_candidate_bank_for_scorer
  --disable_alignment_physical_mask
  --enforce_no_privileged_runtime
)

if [[ -n "$HANDOFF_PROVIDER_CKPT" ]]; then
  CMD+=(--handoff_provider_ckpt "$HANDOFF_PROVIDER_CKPT")
fi

if (( $# > 0 )); then
  CMD+=("$@")
fi

exec env "${CMD[@]}"
