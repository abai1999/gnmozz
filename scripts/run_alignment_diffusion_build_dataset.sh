#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

INPUT_NPZ_1="${INPUT_NPZ_1:-$ROOT/runtime_artifacts/alignment_diffusion/raw_collect_3ep_20260510a/raw_rollout_support_states.npz}"
INPUT_NPZ_2="${INPUT_NPZ_2:-}"
OUTPUT_NPZ="${OUTPUT_NPZ:-$ROOT/runtime_artifacts/alignment_diffusion/alignment_diffusion_dataset_20260510a.npz}"
REPORT_JSON="${REPORT_JSON:-$ROOT/runtime_artifacts/alignment_diffusion/alignment_diffusion_dataset_report_20260510a.json}"

ARGS=(--input_npz "$INPUT_NPZ_1")
if [[ -n "$INPUT_NPZ_2" ]]; then
  ARGS+=(--input_npz "$INPUT_NPZ_2")
fi

exec "$PYTHON_BIN" scripts/build_alignment_diffusion_dataset.py \
  "${ARGS[@]}" \
  --output_npz "$OUTPUT_NPZ" \
  --report_json "$REPORT_JSON"
