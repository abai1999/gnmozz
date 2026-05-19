#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

DATASET="${DATASET:-$ROOT/runtime_artifacts/alignment_diffusion/raw_near_contact_3ep_20260511a/alignment_diffusion_raw_near_contact_3ep.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_diffusion/train_20260511a}"

exec "$PYTHON_BIN" scripts/train_alignment_diffusion_refiner.py \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --epochs 50 \
  --batch_size 64 \
  --lr 3e-4 \
  --horizon 8 \
  --max_pos_step 0.0015 \
  --max_yaw_step 0.0060
