#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
TAG="${TAG:-20260517a_edgepair_200ep}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_${TAG}}"
NUM_EPISODES="${NUM_EPISODES:-200}"

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

RUN_PREFIX=()
if command -v xvfb-run >/dev/null 2>&1; then
  RUN_PREFIX=(xvfb-run -a)
fi

"${RUN_PREFIX[@]}" "$PYTHON_BIN" "$ROOT/scripts/collect_planner_phase2_insert_teacher.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name insert_onto_square_peg \
  --data_root "${DATA_ROOT:-data/rlbench_data}" \
  --output_dir "$OUTPUT_DIR" \
  --output_npz "$OUTPUT_DIR/alignment_tc_planner_phase2_insert_teacher_${TAG}.npz" \
  --report_json "$OUTPUT_DIR/report_${TAG}.json" \
  --takeover_trace_jsonl "$OUTPUT_DIR/takeover_trace_${TAG}.jsonl" \
  --num_episodes "$NUM_EPISODES" \
  --demo_from_episode "${DEMO_FROM_EPISODE:-0}" \
  --demo_max_attempts "${DEMO_MAX_ATTEMPTS:-10}" \
  --video_episodes "${VIDEO_EPISODES:-10}" \
  --record_video \
  --planner_use_depth \
  --planner_use_force \
  --phase1_planner_max_steps "${PHASE1_PLANNER_MAX_STEPS:-260}" \
  --phase1_force_teacher_after_steps "${PHASE1_FORCE_TEACHER_AFTER_STEPS:-80}" \
  --phase1_hard_grasp_fallback \
  --phase1_hard_pregrasp_lift "${PHASE1_HARD_PREGRASP_LIFT:-0.055}" \
  --phase1_hard_move_steps "${PHASE1_HARD_MOVE_STEPS:-18}" \
  --phase1_hard_close_steps "${PHASE1_HARD_CLOSE_STEPS:-8}" \
  --phase1_hard_lift_steps "${PHASE1_HARD_LIFT_STEPS:-10}" \
  --phase1_hard_lift_height "${PHASE1_HARD_LIFT_HEIGHT:-0.060}" \
  --phase1_demo_replay_fallback \
  --phase1_demo_replay_postclose_steps "${PHASE1_DEMO_REPLAY_POSTCLOSE_STEPS:-12}" \
  --phase2_planner_steps "${PHASE2_PLANNER_STEPS:-220}" \
  --phase2_min_planner_steps "${PHASE2_MIN_PLANNER_STEPS:-8}" \
  --phase2_takeover_xy_threshold "${PHASE2_TAKEOVER_XY_THRESHOLD:-0.060}" \
  --phase2_takeover_abs_z_threshold "${PHASE2_TAKEOVER_ABS_Z_THRESHOLD:-0.090}" \
  --phase2_takeover_yaw_threshold "${PHASE2_TAKEOVER_YAW_THRESHOLD:-0.60}" \
  --insert_teacher_steps "${INSERT_TEACHER_STEPS:-160}" \
  --insert_teacher_keep_closed \
  --planner_min_steps_before_takeover "${PLANNER_MIN_STEPS_BEFORE_TAKEOVER:-8}" \
  --planner_takeover_xy_threshold "${PLANNER_TAKEOVER_XY_THRESHOLD:-0.045}" \
  --planner_takeover_abs_z_threshold "${PLANNER_TAKEOVER_ABS_Z_THRESHOLD:-0.090}" \
  --planner_takeover_yaw_threshold "${PLANNER_TAKEOVER_YAW_THRESHOLD:-0.450}" \
  --planner_takeover_yaw_guard_threshold "${PLANNER_TAKEOVER_YAW_GUARD_THRESHOLD:-1.200}" \
  --force_spike_threshold "${FORCE_SPIKE_THRESHOLD:-3.0}" \
  --success_xy_threshold "${SUCCESS_XY_THRESHOLD:-0.004}" \
  --success_z_threshold "${SUCCESS_Z_THRESHOLD:-0.006}" \
  --success_yaw_threshold "${SUCCESS_YAW_THRESHOLD:-0.04}" \
  --expert_k_xy "${EXPERT_K_XY:-0.35}" \
  --expert_k_z "${EXPERT_K_Z:-0.28}" \
  --expert_k_yaw "${EXPERT_K_YAW:-0.18}" \
  --expert_max_pos_step "${EXPERT_MAX_POS_STEP:-0.003}" \
  --expert_max_yaw_step "${EXPERT_MAX_YAW_STEP:-0.010}" \
  --jam_force_threshold "${JAM_FORCE_THRESHOLD:-3.0}" \
  --contact_force_threshold "${CONTACT_FORCE_THRESHOLD:-0.8}" \
  --light_contact_force "${LIGHT_CONTACT_FORCE:-0.45}" \
  --unjam_lift_step "${UNJAM_LIFT_STEP:-0.006}" \
  --unjam_lateral_step "${UNJAM_LATERAL_STEP:-0.0015}" \
  --unjam_yaw_step "${UNJAM_YAW_STEP:-0.006}" \
  --align_xy_threshold "${ALIGN_XY_THRESHOLD:-0.006}" \
  --align_yaw_threshold "${ALIGN_YAW_THRESHOLD:-0.04}" \
  --align_z_step "${ALIGN_Z_STEP:-0.0015}" \
  --k_xy_align "${K_XY_ALIGN:-0.35}" \
  --k_z_hold "${K_Z_HOLD:-0.08}" \
  --k_yaw_align "${K_YAW_ALIGN:-0.18}" \
  --k_xy_descend "${K_XY_DESCEND:-0.18}" \
  --k_z_descend "${K_Z_DESCEND:-0.30}" \
  --k_yaw_descend "${K_YAW_DESCEND:-0.12}" \
  --k_xy_contact "${K_XY_CONTACT:-0.16}" \
  --k_z_contact "${K_Z_CONTACT:-0.12}" \
  --k_yaw_contact "${K_YAW_CONTACT:-0.10}" \
  --k_xy_commit "${K_XY_COMMIT:-0.10}" \
  --k_z_commit "${K_Z_COMMIT:-0.12}" \
  --k_yaw_commit "${K_YAW_COMMIT:-0.08}" \
  --spiral_step "${SPIRAL_STEP:-0.0004}" \
  --contact_z_step "${CONTACT_Z_STEP:-0.0010}" \
  --max_pos_step "${MAX_POS_STEP:-0.003}" \
  --max_yaw_step "${MAX_YAW_STEP:-0.010}" \
  --commit_xy_threshold "${COMMIT_XY_THRESHOLD:-0.006}" \
  --commit_z_threshold "${COMMIT_Z_THRESHOLD:-0.010}" \
  --commit_yaw_threshold "${COMMIT_YAW_THRESHOLD:-0.06}" \
  --grasp_recovery_close_xy_threshold "${GRASP_RECOVERY_CLOSE_XY_THRESHOLD:-0.0032}" \
  --grasp_recovery_close_z_threshold "${GRASP_RECOVERY_CLOSE_Z_THRESHOLD:-0.0035}" \
  --grasp_recovery_close_yaw_threshold "${GRASP_RECOVERY_CLOSE_YAW_THRESHOLD:-0.025}" \
  --grasp_recovery_close_steps "${GRASP_RECOVERY_CLOSE_STEPS:-18}" \
  --grasp_recovery_min_close_steps "${GRASP_RECOVERY_MIN_CLOSE_STEPS:-2}" \
  --grasp_recovery_lift_steps "${GRASP_RECOVERY_LIFT_STEPS:-12}" \
  --grasp_recovery_lift_step "${GRASP_RECOVERY_LIFT_STEP:-0.006}" \
  --grasp_recovery_verify_lift_threshold "${GRASP_RECOVERY_VERIFY_LIFT_THRESHOLD:-0.006}" \
  --grasp_recovery_verify_consecutive_steps "${GRASP_RECOVERY_VERIFY_CONSECUTIVE_STEPS:-2}" \
  --planner_takeover_motion_steps "${PLANNER_TAKEOVER_MOTION_STEPS:-80}" \
  --motion_corridor_force_descend_after_steps "${MOTION_CORRIDOR_FORCE_DESCEND_AFTER_STEPS:-4}" \
  --motion_corridor_descend_xy_threshold "${MOTION_CORRIDOR_DESCEND_XY_THRESHOLD:-0.010}" \
  --motion_corridor_descend_yaw_threshold "${MOTION_CORRIDOR_DESCEND_YAW_THRESHOLD:-0.08}" \
  --teacher_close_contact_depth_threshold "${TEACHER_CLOSE_CONTACT_DEPTH_THRESHOLD:-0.020}" \
  --teacher_grasp_ready_threshold "${TEACHER_GRASP_READY_THRESHOLD:-0.55}" \
  --phase2_positive_tail_steps "${PHASE2_POSITIVE_TAIL_STEPS:-24}" \
  --phase2_positive_weight_broad_near_aux "${PHASE2_POSITIVE_WEIGHT_BROAD_NEAR_AUX:-0.15}" \
  --phase2_positive_weight_near "${PHASE2_POSITIVE_WEIGHT_NEAR:-1.0}" \
  --phase2_positive_weight_micro "${PHASE2_POSITIVE_WEIGHT_MICRO:-1.35}" \
  --phase2_positive_weight_verified "${PHASE2_POSITIVE_WEIGHT_VERIFIED:-1.75}" \
  --phase2_positive_single_axis_scale "${PHASE2_POSITIVE_SINGLE_AXIS_SCALE:-0.5}" \
  --output_dir "$OUTPUT_DIR"
