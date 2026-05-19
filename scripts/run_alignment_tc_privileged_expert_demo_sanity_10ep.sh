#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
TAG="${TAG:-20260511a}"
EPISODES="${EPISODES:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/privileged_expert_demo_sanity_${TAG}}"

mkdir -p "$OUTPUT_DIR"

env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/collect_alignment_tc_privileged_expert_rollout.py" \
  --source demo \
  --task_name insert_onto_square_peg \
  --num_episodes "$EPISODES" \
  --output_dir "$OUTPUT_DIR" \
  --output_npz "$OUTPUT_DIR/alignment_tc_privileged_expert_demo_${TAG}.npz" \
  --report_json "$OUTPUT_DIR/alignment_tc_privileged_expert_demo_report_${TAG}.json"
