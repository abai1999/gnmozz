#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# End-to-end: train 50k steps, then auto-evaluate ALL checkpoints.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "╔══════════════════════════════════════════╗"
echo "║  Phase 1 / 2 — Training  (50k steps)     ║"
echo "╚══════════════════════════════════════════╝"
bash "${SCRIPT_DIR}/run_insert_long_train.sh"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Phase 2 / 2 — Auto-Eval All Checkpoints ║"
echo "╚══════════════════════════════════════════╝"

# Activate conda env (in case the sub-shell lost it)
eval "$(conda shell.bash hook)"
conda activate vla-adapter

export CUDA_VISIBLE_DEVICES=0

python scripts/auto_eval_watcher.py \
  --run_dir outputs/insert_long_train \
  --task_name insert_onto_square_peg \
  --num_episodes 25 \
  --max_steps 300 \
  --record_video \
  --once

echo ""
echo "Done!  Check eval_logs/insert_onto_square_peg/ for results."
echo "  - Per-checkpoint results in each *_chkpt/ subdirectory"
echo "  - Overall summary:     eval_logs/insert_onto_square_peg/summary.json"
echo "  - Best success GIF:    eval_logs/insert_onto_square_peg/overall_best.gif"
