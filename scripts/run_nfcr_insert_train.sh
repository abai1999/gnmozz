#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$WORKSPACE"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python3.10}"

DATA_DIR="${1:-data/residual_data/insert_vo40k_nfcr_s3407}"
OUTPUT_DIR="${2:-outputs/residual_train/insert_vo40k_nfcr_mlp_s3407}"
SEED="${3:-3407}"

echo "=== NFCR insert residual training ==="
echo "Data:   $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Seed:   $SEED"

"$PYTHON_BIN" scripts/train_residual.py \
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
    --save_freq 5000 \
    --seed "$SEED"
