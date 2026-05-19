#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Train the residual controller on pre-collected data.
#
# Usage: CUDA_VISIBLE_DEVICES=X bash scripts/run_train_residual.sh [task_name]
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

TASK_NAME="${1:-insert_onto_square_peg}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$WORKSPACE"

eval "$(conda shell.bash hook)"
conda activate vla-adapter

DATA_DIR="data/residual_data/${TASK_NAME}"
OUTPUT_DIR="outputs/residual_train/${TASK_NAME}_v1"

echo "=== Training residual controller ==="
echo "Data:   $DATA_DIR"
echo "Output: $OUTPUT_DIR"

python scripts/train_residual.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 50000 \
    --batch_size 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --warmup_steps 500 \
    --lambda_zero 0.1 \
    --lambda_alpha_zero 0.05 \
    --oversample_contact 5 \
    --oversample_pre_contact 3 \
    --oversample_jam 7 \
    --save_freq 5000

echo "=== Training complete ==="
