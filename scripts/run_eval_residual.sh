#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Evaluate with different modes: planner_only / rule_reflex / learned_residual / full
#
# Usage:
#   CUDA_VISIBLE_DEVICES=X bash scripts/run_eval_residual.sh <checkpoint_dir> <mode> [residual_ckpt] [task_name] [num_episodes]
#
# Modes:
#   planner_only      – baseline evaluation (no reflex/residual)
#   rule_reflex       – planner + rule-based force reflex
#   learned_residual  – planner + learned residual only
#   full              – planner + learned residual + rule reflex
#
# Examples:
#   bash scripts/run_eval_residual.sh outputs/insert_long_train/run--30000_chkpt planner_only
#   bash scripts/run_eval_residual.sh outputs/insert_long_train/run--30000_chkpt rule_reflex
#   bash scripts/run_eval_residual.sh outputs/insert_long_train/run--30000_chkpt learned_residual \
#       outputs/residual_train/insert_v1/residual_final.pt
#   bash scripts/run_eval_residual.sh outputs/insert_long_train/run--30000_chkpt full \
#       outputs/residual_train/insert_v1/residual_final.pt
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir> <mode> [residual_ckpt]}"
MODE="${2:?Usage: $0 <checkpoint_dir> <mode> [residual_ckpt]}"
RESIDUAL_CKPT="${3:-}"
TASK_NAME="${4:-insert_onto_square_peg}"
NUM_EPISODES="${5:-15}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$WORKSPACE"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python3.10}"
HF_CACHE_ROOT=${HF_CACHE_ROOT:-/mnt/ssd/guoning/hf-cache}
export HF_HOME=${HF_HOME:-$HF_CACHE_ROOT}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_ROOT/hub}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"

start_headless_display() {
    local display_num
    for display_num in $(seq 90 120); do
        if [ ! -e "/tmp/.X11-unix/X${display_num}" ]; then
            export DISPLAY=":${display_num}"
            Xvfb "$DISPLAY" -screen 0 1024x768x24 -ac +extension GLX +render -noreset >/tmp/run_eval_residual_xvfb_${display_num}.log 2>&1 &
            XVFB_PID=$!
            sleep 1
            if kill -0 "$XVFB_PID" 2>/dev/null; then
                export QT_X11_NO_MITSHM=1
                export XAUTHORITY=""
                trap 'kill "$XVFB_PID" 2>/dev/null || true' EXIT
                return 0
            fi
        fi
    done
    echo "ERROR: failed to start a free Xvfb display"
    exit 1
}

CKPT_NAME="$(basename "$CHECKPOINT_DIR")"
OUTPUT_DIR="eval_logs/${TASK_NAME}/${CKPT_NAME}--${MODE}"

echo "=== Evaluation ==="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Mode:       $MODE"
echo "Task:       $TASK_NAME"
echo "Episodes:   $NUM_EPISODES"
echo "Output:     $OUTPUT_DIR"

EXTRA_ARGS=""
case "$MODE" in
    planner_only)
        ;;
    rule_reflex)
        EXTRA_ARGS="--use_rule_reflex"
        ;;
    learned_residual)
        if [ -z "$RESIDUAL_CKPT" ]; then
            echo "ERROR: learned_residual mode requires residual checkpoint path as 3rd argument"
            exit 1
        fi
        EXTRA_ARGS="--use_learned_residual --residual_ckpt $RESIDUAL_CKPT"
        echo "Residual:   $RESIDUAL_CKPT"
        ;;
    full)
        if [ -z "$RESIDUAL_CKPT" ]; then
            echo "ERROR: full mode requires residual checkpoint path as 3rd argument"
            exit 1
        fi
        EXTRA_ARGS="--use_learned_residual --use_rule_reflex --residual_ckpt $RESIDUAL_CKPT"
        echo "Residual:   $RESIDUAL_CKPT"
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Use: planner_only / rule_reflex / learned_residual / full"
        exit 1
        ;;
esac

start_headless_display
"$PYTHON_BIN" scripts/evaluate_rlbench.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task_name "$TASK_NAME" \
    --num_episodes "$NUM_EPISODES" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo "=== Evaluation complete ==="
echo "Results: $OUTPUT_DIR/eval_results.json"
