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
    # Load RLBench / CoppeliaSim environment when invoked from non-login shells.
    source "$HOME/.bashrc"
fi

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"

start_headless_display() {
    local display_num
    for display_num in $(seq 90 120); do
        if [ ! -e "/tmp/.X11-unix/X${display_num}" ]; then
            export DISPLAY=":${display_num}"
            Xvfb "$DISPLAY" -screen 0 1024x768x24 -ac +extension GLX +render -noreset >/tmp/nfcr_xvfb_${display_num}.log 2>&1 &
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

PLANNER_CKPT="${1:-outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ROLLOUT_MODE="${2:-planner_only}"
RESIDUAL_CKPT="${3:-}"
OUTPUT_DIR="${4:-data/residual_data/insert_vo40k_nfcr_s3407}"

EXTRA_ARGS=""
case "$ROLLOUT_MODE" in
    planner_only)
        ;;
    rule_reflex)
        EXTRA_ARGS="--use_rule_reflex"
        ;;
    full)
        if [ -z "$RESIDUAL_CKPT" ]; then
            echo "ERROR: full rollout requires residual checkpoint as 3rd arg"
            exit 1
        fi
        EXTRA_ARGS="--use_rule_reflex --use_learned_residual --residual_ckpt $RESIDUAL_CKPT"
        ;;
    *)
        echo "ERROR: Unknown rollout mode '$ROLLOUT_MODE'. Use planner_only / rule_reflex / full"
        exit 1
        ;;
esac

echo "=== NFCR insert collector ==="
echo "Planner: $PLANNER_CKPT"
echo "Mode:    $ROLLOUT_MODE"
echo "Output:  $OUTPUT_DIR"

start_headless_display
"$PYTHON_BIN" scripts/collect_residual_planner_state.py \
    --checkpoint_dir "$PLANNER_CKPT" \
    --task_name insert_onto_square_peg \
    --data_root data/rlbench_data \
    --output_dir "$OUTPUT_DIR" \
    --planner_no_depth \
    --planner_no_force \
    --delta_clip_pos 0.01 \
    --delta_clip_rot 0.05 \
    --shard_size 10000 \
    $EXTRA_ARGS
