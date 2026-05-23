#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT="${CHECKPOINT:-$ROOT/runtime_artifacts/coarse2contact_v2/checkpoints/grasp_recovery_head_v7_targeted_aug_weighted/best.pt}"
PREFERRED_DATASET="${PREFERRED_DATASET:-$ROOT/runtime_artifacts/coarse2contact_v2/datasets_runtime_failure/grasp_recovery_runtime_failure_dataset_v1.jsonl}"
FALLBACK_DATASET="${FALLBACK_DATASET:-$ROOT/runtime_artifacts/coarse2contact_v2/datasets_recovery_targeted_aug_v2_harder/grasp_recovery_dataset_v2_targeted_aug.jsonl}"
DATASET="${DATASET:-$PREFERRED_DATASET}"
if [[ ! -f "$DATASET" && -f "$FALLBACK_DATASET" ]]; then
  DATASET="$FALLBACK_DATASET"
fi
OUTPUT="${OUTPUT:-$ROOT/runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_shadow_mainline_v7.json}"
TRACE_OUTPUT="${TRACE_OUTPUT:-$ROOT/runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_shadow_mainline_v7_trace.jsonl}"
MODEL_KIND="${MODEL_KIND:-auto}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON_BIN" "$ROOT/scripts/eval_c2c_v2_grasp_recovery_shadow.py" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --trace_output "$TRACE_OUTPUT" \
  --model_kind "$MODEL_KIND" \
  --device "$DEVICE"
