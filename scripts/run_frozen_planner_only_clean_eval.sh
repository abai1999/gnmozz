#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/frozen_planner_only_clean_20260430}"
NAME_SUFFIX="${NAME_SUFFIX:-frozen_planner_only_clean}"
TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-1,5,8,10,16,17,19,20}"
NUM_EPISODES="${NUM_EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-320}"
EVAL_SEED="${EVAL_SEED:-3407}"
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

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT"

# Deliberately use planner_only and pass no alignment/contact/student ckpts.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --modes planner_only \
  --num_episodes "$NUM_EPISODES" \
  --episode_indices "$EPISODE_INDICES" \
  --max_steps "$MAX_STEPS" \
  --output_root "$OUTPUT_ROOT/eval" \
  --name_suffix "$NAME_SUFFIX" \
  --planner_no_depth \
  --planner_no_force \
  --use_depth \
  --use_force \
  --depth_force_clean_support \
  --depth_force_clean_privileged_labels \
  --eval_seed "$EVAL_SEED" \
  --support_states_output_npz "$SUPPORT_STATES_OUTPUT_NPZ" \
  --record_video \
  --write_episode_videos \
  --no_best_gif
