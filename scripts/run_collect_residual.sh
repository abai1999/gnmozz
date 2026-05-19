#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Collect residual training data using a frozen planner checkpoint.
#
# Usage: CUDA_VISIBLE_DEVICES=X bash scripts/run_collect_residual.sh <checkpoint_dir> [task_name] [collector_mode]
# Example:
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_collect_residual.sh \
#       outputs/insert_long_train/run--30000_chkpt
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir> [task_name] [collector_mode]}"
TASK_NAME="${2:-insert_onto_square_peg}"
COLLECTOR_MODE="${3:-planner_state}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$WORKSPACE"

eval "$(conda shell.bash hook)"
conda activate vla-adapter

OUTPUT_DIR="data/residual_data/${TASK_NAME}"

echo "=== Collecting residual data ==="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Task:       $TASK_NAME"
echo "Output:     $OUTPUT_DIR"
echo "Mode:       $COLLECTOR_MODE"

if [ "$COLLECTOR_MODE" = "demo_state" ]; then
    python scripts/collect_residual_data.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --task_name "$TASK_NAME" \
        --data_root data/rlbench_data \
        --output_dir "$OUTPUT_DIR" \
        --stride 4 \
        --delta_clip_pos 0.01 \
        --delta_clip_rot 0.05 \
        --shard_size 10000
else
    python scripts/collect_residual_planner_state.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --task_name "$TASK_NAME" \
        --data_root data/rlbench_data \
        --output_dir "$OUTPUT_DIR" \
        --delta_clip_pos 0.01 \
        --delta_clip_rot 0.05 \
        --shard_size 10000
fi

echo "=== Collection complete ==="
