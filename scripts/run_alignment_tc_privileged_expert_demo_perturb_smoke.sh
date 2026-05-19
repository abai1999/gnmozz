#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
TAG="${TAG:-20260511d}"
EPISODES="${EPISODES:-2}"
PERTURB_FRAMES_PER_DEMO="${PERTURB_FRAMES_PER_DEMO:-4}"
PERTURB_COPIES_PER_FRAME="${PERTURB_COPIES_PER_FRAME:-2}"
PERTURB_EXPERT_STEPS="${PERTURB_EXPERT_STEPS:-30}"
MAX_PERTURB_ROLLOUTS="${MAX_PERTURB_ROLLOUTS:-12}"
VIDEO_EPISODES="${VIDEO_EPISODES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/privileged_expert_demo_perturb_smoke_${TAG}}"

mkdir -p "$OUTPUT_DIR"

env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/collect_alignment_tc_privileged_expert_rollout.py" \
  --source demo_perturb \
  --task_name insert_onto_square_peg \
  --num_episodes "$EPISODES" \
  --perturb_frames_per_demo "$PERTURB_FRAMES_PER_DEMO" \
  --perturb_copies_per_frame "$PERTURB_COPIES_PER_FRAME" \
  --perturb_expert_steps "$PERTURB_EXPERT_STEPS" \
  --max_perturb_rollouts "$MAX_PERTURB_ROLLOUTS" \
  --output_dir "$OUTPUT_DIR" \
  --output_npz "$OUTPUT_DIR/alignment_tc_privileged_expert_demo_perturb_${TAG}.npz" \
  --report_json "$OUTPUT_DIR/alignment_tc_privileged_expert_demo_perturb_report_${TAG}.json" \
  --record_video \
  --video_episodes "$VIDEO_EPISODES"
