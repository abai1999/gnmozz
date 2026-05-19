#!/bin/bash
set -euo pipefail

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
CONDA_LIBSTDCXX="${CONDA_LIBSTDCXX:-/home/guoning/my_conda_envs/vla-adapter/lib/libstdc++.so.6}"
BASE_LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${BASE_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${BASE_LD_LIBRARY_PATH}"
fi
if [ -n "${LD_PRELOAD:-}" ]; then
    export LD_PRELOAD="${CONDA_LIBSTDCXX} ${LD_PRELOAD}"
else
    export LD_PRELOAD="${CONDA_LIBSTDCXX}"
fi
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_X11_NO_MITSHM=1
export XAUTHORITY=""

start_headless_display() {
    local display_num
    for display_num in $(seq 90 120); do
        if [ ! -e "/tmp/.X11-unix/X${display_num}" ]; then
            export DISPLAY=":${display_num}"
            Xvfb "$DISPLAY" -screen 0 1024x768x24 -ac +extension GLX +render -noreset >/tmp/nfcr_eval_xvfb_${display_num}.log 2>&1 &
            XVFB_PID=$!
            sleep 1
            if kill -0 "$XVFB_PID" 2>/dev/null; then
                trap 'kill "$XVFB_PID" 2>/dev/null || true' EXIT
                return 0
            fi
        fi
    done
    echo "ERROR: failed to start a free Xvfb display"
    exit 1
}

MODE="${1:?Usage: $0 <planner_only|safety_only|alignment|contact|full> [controller_ckpt] [planner_ckpt] [seed_suffix]}"
CONTROLLER_CKPT="${2:-}"
PLANNER_CKPT="${3:-outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
NAME_SUFFIX="${4:-s3407}"

BASE_NAME="insert_vo40k_${MODE}_${NAME_SUFFIX}"
OUTPUT_DIR="eval_logs/insert_onto_square_peg/${BASE_NAME}"

EXTRA_ARGS=""
case "$MODE" in
    planner_only)
        ;;
    safety_only)
        EXTRA_ARGS="--use_stage_aware_refiner --stage_refiner_mode safety_only"
        ;;
    alignment)
        if [ -z "$CONTROLLER_CKPT" ]; then
            echo "ERROR: alignment requires controller checkpoint"
            exit 1
        fi
        EXTRA_ARGS="--use_stage_aware_refiner --stage_refiner_mode alignment --alignment_ckpt $CONTROLLER_CKPT"
        ;;
    contact)
        if [ -z "$CONTROLLER_CKPT" ]; then
            echo "ERROR: contact requires controller checkpoint"
            exit 1
        fi
        EXTRA_ARGS="--use_stage_aware_refiner --stage_refiner_mode contact --contact_ckpt $CONTROLLER_CKPT"
        ;;
    full)
        if [ -z "$CONTROLLER_CKPT" ]; then
            echo "ERROR: full requires controller checkpoint; pass same ckpt for alignment/contact MVP"
            exit 1
        fi
        EXTRA_ARGS="--use_stage_aware_refiner --stage_refiner_mode full --alignment_ckpt $CONTROLLER_CKPT --contact_ckpt $CONTROLLER_CKPT"
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'"
        exit 1
        ;;
esac

echo "=== NFCR insert eval ==="
echo "Planner:  $PLANNER_CKPT"
echo "Mode:     $MODE"
echo "Output:   $OUTPUT_DIR"

start_headless_display
"$PYTHON_BIN" scripts/evaluate_rlbench.py \
    --checkpoint_dir "$PLANNER_CKPT" \
    --task_name insert_onto_square_peg \
    --num_episodes 15 \
    --max_steps 500 \
    --output_dir "$OUTPUT_DIR" \
    --planner_no_depth \
    --planner_no_force \
    $EXTRA_ARGS
