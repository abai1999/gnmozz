#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORKSPACE"

SSD_ROOT="${SSD_ROOT:-/mnt/ssd/guoning/VLA_runtime}"
OUTPUTS_TARGET="${OUTPUTS_TARGET:-$SSD_ROOT/outputs}"
EVAL_LOGS_TARGET="${EVAL_LOGS_TARGET:-$SSD_ROOT/eval_logs}"
LOG_ROOT="${LOG_ROOT:-$SSD_ROOT/launch_logs}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/insert_long_train}"
RUN_ID_NOTE="${RUN_ID_NOTE:-insert_vision_only_50k_retrain_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
WATCH_NUM_EPISODES="${WATCH_NUM_EPISODES:-10}"
WATCH_MAX_STEPS="${WATCH_MAX_STEPS:-500}"
WATCH_POLL_INTERVAL="${WATCH_POLL_INTERVAL:-120}"
TRAIN_GPU="${TRAIN_GPU:-}"
EVAL_GPU="${EVAL_GPU:-}"
TRAIN_SEED="${TRAIN_SEED:-3407}"

mkdir -p "$OUTPUTS_TARGET" "$EVAL_LOGS_TARGET" "$LOG_ROOT"

if [[ -L outputs && "$(readlink outputs)" != "$OUTPUTS_TARGET" ]]; then
  rm outputs
fi
if [[ ! -e outputs ]]; then
  ln -s "$OUTPUTS_TARGET" outputs
fi

if [[ -L eval_logs && "$(readlink eval_logs)" != "$EVAL_LOGS_TARGET" ]]; then
  rm eval_logs
fi
if [[ -d eval_logs && ! -L eval_logs ]]; then
  rmdir eval_logs 2>/dev/null || true
fi
if [[ ! -e eval_logs ]]; then
  ln -s "$EVAL_LOGS_TARGET" eval_logs
fi

if [[ -z "$TRAIN_GPU" ]]; then
  TRAIN_GPU="$("$SCRIPT_DIR/choose_idle_gpu.sh")"
fi

if [[ -z "$EVAL_GPU" ]]; then
  GPU_CANDIDATES="$(seq 0 7 | tr '\n' ' ' | sed "s/\\b$TRAIN_GPU\\b//g")" \
    EVAL_GPU="$("$SCRIPT_DIR/choose_idle_gpu.sh")" || true
  if [[ -z "${EVAL_GPU:-}" ]]; then
    EVAL_GPU="$TRAIN_GPU"
  fi
fi

START_TS="$(date +%s)"
TRAIN_LOG="$LOG_ROOT/${RUN_ID_NOTE}_train.log"
WATCH_LOG="$LOG_ROOT/${RUN_ID_NOTE}_watcher.log"
RUN_DIR_ABS="$WORKSPACE/$RUN_ROOT_DIR"
EVAL_ROOT_DIR="eval_logs/insert_onto_square_peg/${RUN_ID_NOTE}_autoeval"

echo "=== Planner retrain launcher ==="
echo "WORKSPACE=$WORKSPACE"
echo "SSD_ROOT=$SSD_ROOT"
echo "RUN_ID_NOTE=$RUN_ID_NOTE"
echo "RUN_ROOT_DIR=$RUN_ROOT_DIR"
echo "EVAL_ROOT_DIR=$EVAL_ROOT_DIR"
echo "TRAIN_GPU=$TRAIN_GPU"
echo "EVAL_GPU=$EVAL_GPU"
echo "MAX_STEPS=$MAX_STEPS"
echo "SAVE_FREQ=$SAVE_FREQ"
echo "TRAIN_SEED=$TRAIN_SEED"
echo "WATCH_NUM_EPISODES=$WATCH_NUM_EPISODES"
echo "WATCH_MAX_STEPS=$WATCH_MAX_STEPS"
echo "WATCH_POLL_INTERVAL=$WATCH_POLL_INTERVAL"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "WATCH_LOG=$WATCH_LOG"

nohup bash -lc "
  cd '$WORKSPACE' &&
  export CUDA_VISIBLE_DEVICES='$TRAIN_GPU' &&
  export RUN_ID_NOTE='$RUN_ID_NOTE' &&
  export RUN_ROOT_DIR='$RUN_ROOT_DIR' &&
  export MAX_STEPS='$MAX_STEPS' &&
  export SAVE_FREQ='$SAVE_FREQ' &&
  export TRAIN_SEED='$TRAIN_SEED' &&
  bash '$SCRIPT_DIR/run_insert_no_depth_force.sh'
" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

nohup bash -lc "
  cd '$WORKSPACE' &&
  /home/guoning/my_conda_envs/vla-adapter/bin/python3.10 -u scripts/auto_eval_watcher.py \
    --run_dir '$RUN_DIR_ABS' \
    --task_name insert_onto_square_peg \
    --num_episodes '$WATCH_NUM_EPISODES' \
    --max_steps '$WATCH_MAX_STEPS' \
    --no_depth \
    --no_force \
    --record_video \
    --gpu_id '$EVAL_GPU' \
    --poll_interval '$WATCH_POLL_INTERVAL' \
    --eval_root_dir '$EVAL_ROOT_DIR' \
    --checkpoint_name_contains '$RUN_ID_NOTE' \
    --checkpoint_mtime_after '$START_TS'
" >"$WATCH_LOG" 2>&1 &
WATCH_PID=$!

echo "TRAIN_PID=$TRAIN_PID"
echo "WATCH_PID=$WATCH_PID"
echo "$TRAIN_PID" > "$LOG_ROOT/${RUN_ID_NOTE}.train.pid"
echo "$WATCH_PID" > "$LOG_ROOT/${RUN_ID_NOTE}.watcher.pid"
echo "$RUN_ID_NOTE" > "$LOG_ROOT/latest_run_id.txt"
