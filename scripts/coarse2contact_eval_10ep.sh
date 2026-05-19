#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-0,1,5,8,10,16,17,19,20,21}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-320}"
EVAL_SEED="${EVAL_SEED:-3407}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/coarse2contact/eval_10ep}"
NAME_SUFFIX="${NAME_SUFFIX:-coarse2contact_eval_10ep}"

MODE="${MODE:-depth_force}"
COARSE2CONTACT_SHADOW_ONLY="${COARSE2CONTACT_SHADOW_ONLY:-}"
if [[ -z "$COARSE2CONTACT_SHADOW_ONLY" ]]; then
  if [[ "$MODE" == "depth_shadow" ]]; then
    COARSE2CONTACT_SHADOW_ONLY=1
  else
    COARSE2CONTACT_SHADOW_ONLY=0
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
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

SHADOW_ONLY_ARGS=()
if [[ "$COARSE2CONTACT_SHADOW_ONLY" == "1" ]]; then
  SHADOW_ONLY_ARGS+=(--coarse2contact_shadow_only)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" xvfb-run -a "$PYTHON_BIN" scripts/evaluate_c2c_rlbench.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --mode "$MODE" \
  --num_episodes "$NUM_EPISODES" \
  --episode_indices "$EPISODE_INDICES" \
  --max_steps "$MAX_STEPS" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "$NAME_SUFFIX" \
  "${SHADOW_ONLY_ARGS[@]}" \
  --record_video \
  --write_episode_videos \
  --no_best_gif \
  --eval_seed "$EVAL_SEED"

"$PYTHON_BIN" scripts/audit_coarse2contact_current_state.py \
  --root "$ROOT" \
  --output_json "$OUTPUT_ROOT/audit_current_state.json"
