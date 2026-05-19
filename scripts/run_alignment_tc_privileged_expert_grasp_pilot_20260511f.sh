#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
TAG="${TAG:-20260511f_grasp_expert_pilot}"
OUT="${OUT:-runtime_artifacts/alignment_tc_diffusion/privileged_expert_demo_perturb_${TAG}}"

mkdir -p "${OUT}"

xvfb-run -a "${PYTHON_BIN}" scripts/collect_alignment_tc_privileged_expert_rollout.py \
  --source demo_perturb \
  --demo_target_stage grasp_commit \
  --num_episodes "${NUM_EPISODES:-50}" \
  --perturb_frames_per_demo "${PERTURB_FRAMES_PER_DEMO:-6}" \
  --perturb_copies_per_frame "${PERTURB_COPIES_PER_FRAME:-2}" \
  --perturb_expert_steps "${PERTURB_EXPERT_STEPS:-20}" \
  --grasp_preclose_window "${GRASP_PRECLOSE_WINDOW:-80}" \
  --grasp_postclose_window "${GRASP_POSTCLOSE_WINDOW:-2}" \
  --output_dir "${OUT}" \
  --output_npz "${OUT}/alignment_tc_privileged_expert_demo_perturb_${TAG}.npz" \
  --report_json "${OUT}/alignment_tc_privileged_expert_demo_perturb_report_${TAG}.json" \
  --record_video \
  --video_episodes "${VIDEO_EPISODES:-10}"
