#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
MODE="${MODE:-planner_only}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19}"
NUM_EPISODES="${NUM_EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-320}"
EVAL_SEED="${EVAL_SEED:-3407}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/coarse2contact_v2/${MODE}_3ep}"
NAME_SUFFIX="${NAME_SUFFIX:-coarse2contact_v2_${MODE}_3ep}"
DEPTH_LOCALIZER_BACKEND="${DEPTH_LOCALIZER_BACKEND:-heuristic}"
DEPTH_LOCALIZER_CKPT="${DEPTH_LOCALIZER_CKPT:-}"
ALLOW_LEARNED_DEPTH_APPLY="${ALLOW_LEARNED_DEPTH_APPLY:-0}"
FORCE_CLASSIFIER_BACKEND="${FORCE_CLASSIFIER_BACKEND:-rule}"
FORCE_CLASSIFIER_CKPT="${FORCE_CLASSIFIER_CKPT:-}"

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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" xvfb-run -a "$PYTHON_BIN" scripts/evaluate_c2c_v2_rlbench.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --mode "$MODE" \
  --depth_localizer_backend "$DEPTH_LOCALIZER_BACKEND" \
  ${DEPTH_LOCALIZER_CKPT:+--depth_localizer_ckpt "$DEPTH_LOCALIZER_CKPT"} \
  $(if [ "$ALLOW_LEARNED_DEPTH_APPLY" = "1" ]; then printf '%s ' --allow_learned_depth_apply; fi) \
  --force_classifier_backend "$FORCE_CLASSIFIER_BACKEND" \
  ${FORCE_CLASSIFIER_CKPT:+--force_classifier_ckpt "$FORCE_CLASSIFIER_CKPT"} \
  --num_episodes "$NUM_EPISODES" \
  --episode_indices "$EPISODE_INDICES" \
  --max_steps "$MAX_STEPS" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "$NAME_SUFFIX" \
  --planner_no_depth \
  --planner_no_force \
  --use_depth \
  --use_force \
  --eval_seed "$EVAL_SEED" \
  --record_video \
  --write_episode_videos \
  --no_best_gif
