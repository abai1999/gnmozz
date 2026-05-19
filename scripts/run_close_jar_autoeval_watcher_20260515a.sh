#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HF_HOME="${HF_HOME:-/home/guoning/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
RUN_DIR="${RUN_DIR:-/home/guoning/code/VLA/outputs/close_jar_long_train}"
RUN_ID_NOTE="${RUN_ID_NOTE:-close_jar_vision_only_current_unweighted_lr1e4_40k_20260515a}"
TASK_NAME="${TASK_NAME:-close_jar}"
EVAL_ROOT_DIR="${EVAL_ROOT_DIR:-eval_logs/close_jar/${RUN_ID_NOTE}_autoeval}"
GPU_ID="${GPU_ID:-5}"
CHECKPOINT_STEP_MIN="${CHECKPOINT_STEP_MIN:-10000}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-500}"

cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook)"
conda activate vla-adapter

echo "=== Starting close_jar auto-eval watcher ==="
echo "RUN_DIR=${RUN_DIR}"
echo "RUN_ID_NOTE=${RUN_ID_NOTE}"
echo "TASK_NAME=${TASK_NAME}"
echo "EVAL_ROOT_DIR=${EVAL_ROOT_DIR}"
echo "GPU_ID=${GPU_ID}"
echo "CHECKPOINT_STEP_MIN=${CHECKPOINT_STEP_MIN}"

/home/guoning/my_conda_envs/vla-adapter/bin/python -u scripts/auto_eval_watcher.py \
  --run_dir "${RUN_DIR}" \
  --task_name "${TASK_NAME}" \
  --eval_root_dir "${EVAL_ROOT_DIR}" \
  --checkpoint_name_contains "${RUN_ID_NOTE}" \
  --checkpoint_step_min "${CHECKPOINT_STEP_MIN}" \
  --poll_interval "${POLL_INTERVAL}" \
  --gpu_id "${GPU_ID}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_steps "${MAX_EVAL_STEPS}" \
  --depth_max 1.0 \
  --no_depth \
  --no_force \
  --record_video
