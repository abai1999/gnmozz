#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BATCH_LABEL="${BATCH_LABEL:?BATCH_LABEL is required}"
EPISODE_INDICES="${EPISODE_INDICES:?EPISODE_INDICES is required}"
V4B_CANDIDATE_CKPT="${V4B_CANDIDATE_CKPT:?V4B_CANDIDATE_CKPT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_shadow_20260430_batches/$BATCH_LABEL}"
NAME_SUFFIX="${NAME_SUFFIX:-alignment_v4b_candidate_shadow_${BATCH_LABEL}}"
NUM_EPISODES="${NUM_EPISODES:-2}"
KEEP_YAW_ABS="${KEEP_YAW_ABS:-0.02}"

mkdir -p "$OUTPUT_ROOT"

OUTPUT_ROOT="$OUTPUT_ROOT" \
V4B_CANDIDATE_CKPT="$V4B_CANDIDATE_CKPT" \
EPISODE_INDICES="$EPISODE_INDICES" \
NUM_EPISODES="$NUM_EPISODES" \
NAME_SUFFIX="$NAME_SUFFIX" \
bash "$ROOT/scripts/run_alignment_v4b_candidate_shadow_20260429.sh"

TRACE_DIR="$(find "$OUTPUT_ROOT" -maxdepth 1 -mindepth 1 -type d | head -n 1)"
if [[ -z "$TRACE_DIR" ]]; then
  echo "No trace subdirectory found under $OUTPUT_ROOT" >&2
  exit 1
fi

/home/guoning/my_conda_envs/vla-adapter/bin/python "$ROOT/scripts/audit_alignment_v4b_shadow_failures.py" \
  --trace_dir "$TRACE_DIR" \
  --output_json "$OUTPUT_ROOT/alignment_v4b_shadow_failure_audit.json" \
  --keep_yaw_abs "$KEEP_YAW_ABS"
