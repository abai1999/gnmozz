#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/my_conda_envs/vla-adapter/bin/python}"
DATASET="${DATASET:-$ROOT/runtime_artifacts/coarse2contact_v2/datasets_runtime_failure_30ep_hardmix/grasp_recovery_runtime_failure_dataset_v1.jsonl}"
V11_CHECKPOINT="${V11_CHECKPOINT:-$ROOT/runtime_artifacts/coarse2contact_v2/checkpoints/grasp_recovery_head_v11_runtime_failure/best.pt}"
V16_CHECKPOINT="${V16_CHECKPOINT:-$ROOT/runtime_artifacts/coarse2contact_v2/checkpoints/grasp_recovery_head_v16_tailbucket_conservative_30ep/best.pt}"
OUTPUT="${OUTPUT:-$ROOT/runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_closed_loop_30ep.json}"
TRACE_OUTPUT="${TRACE_OUTPUT:-$ROOT/runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_closed_loop_30ep_trace.jsonl}"

"$PYTHON_BIN" "$ROOT/scripts/eval_c2c_v2_grasp_recovery_closed_loop.py" \
  --dataset "$DATASET" \
  --v11_checkpoint "$V11_CHECKPOINT" \
  --v16_checkpoint "$V16_CHECKPOINT" \
  --output "$OUTPUT" \
  --trace_output "$TRACE_OUTPUT"
