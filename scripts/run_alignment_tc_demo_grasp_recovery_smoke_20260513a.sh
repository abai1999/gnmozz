#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/demo_grasp_recovery_20260513a_smoke}"

mkdir -p "$OUT_DIR"

RUN_PREFIX=()
if command -v xvfb-run >/dev/null 2>&1; then
  RUN_PREFIX=(xvfb-run -a)
fi

"${RUN_PREFIX[@]}" "$PYTHON" "$ROOT/scripts/collect_alignment_tc_privileged_expert_rollout.py" \
  --source demo_grasp_recovery \
  --task_name insert_onto_square_peg \
  --demo_target_stage grasp_commit \
  --num_episodes "${NUM_EPISODES:-2}" \
  --demo_from_episode "${DEMO_FROM_EPISODE:-0}" \
  --grasp_preclose_window "${GRASP_PRECLOSE_WINDOW:-64}" \
  --perturb_frames_per_demo "${PERTURB_FRAMES_PER_DEMO:-4}" \
  --perturb_copies_per_frame "${PERTURB_COPIES_PER_FRAME:-2}" \
  --perturb_expert_steps "${PERTURB_EXPERT_STEPS:-48}" \
  --max_perturb_rollouts "${MAX_PERTURB_ROLLOUTS:-8}" \
  --perturb_micro_xy_std "${PERTURB_MICRO_XY_STD:-0.0015}" \
  --perturb_micro_z_std "${PERTURB_MICRO_Z_STD:-0.0012}" \
  --perturb_micro_yaw_std "${PERTURB_MICRO_YAW_STD:-0.025}" \
  --perturb_near_xy_std "${PERTURB_NEAR_XY_STD:-0.0030}" \
  --perturb_near_z_std "${PERTURB_NEAR_Z_STD:-0.0020}" \
  --perturb_near_yaw_std "${PERTURB_NEAR_YAW_STD:-0.045}" \
  --perturb_max_pos "${PERTURB_MAX_POS:-0.005}" \
  --perturb_max_yaw "${PERTURB_MAX_YAW:-0.08}" \
  --max_pos_step "${MAX_POS_STEP:-0.0025}" \
  --max_yaw_step "${MAX_YAW_STEP:-0.012}" \
  --grasp_recovery_close_xy_threshold "${GRASP_RECOVERY_CLOSE_XY_THRESHOLD:-0.0032}" \
  --grasp_recovery_close_z_threshold "${GRASP_RECOVERY_CLOSE_Z_THRESHOLD:-0.0035}" \
  --grasp_recovery_close_yaw_threshold "${GRASP_RECOVERY_CLOSE_YAW_THRESHOLD:-0.025}" \
  --grasp_recovery_close_steps "${GRASP_RECOVERY_CLOSE_STEPS:-18}" \
  --grasp_recovery_lift_steps "${GRASP_RECOVERY_LIFT_STEPS:-14}" \
  --record_video \
  --video_episodes "${VIDEO_EPISODES:-8}" \
  --video_fps "${VIDEO_FPS:-20}" \
  --output_dir "$OUT_DIR" \
  --output_npz "$OUT_DIR/alignment_tc_demo_grasp_recovery_raw_20260513a_smoke.npz" \
  --report_json "$OUT_DIR/report_20260513a_smoke.json"
