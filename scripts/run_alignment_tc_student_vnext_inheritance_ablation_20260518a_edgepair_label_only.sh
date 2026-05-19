#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
export LD_PRELOAD="${LD_PRELOAD:-/home/guoning/my_conda_envs/vla-adapter/lib/libstdc++.so.6}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-5,8,19,20,22,24,30,31,35,38}"
MAX_STEPS="${MAX_STEPS:-340}"
EVAL_SEED="${EVAL_SEED:-3407}"
STUDENT_CKPT="${STUDENT_CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260517a_edgepair/alignment_tc_student_vnext_best.pt}"
CORRIDOR_JSON="${CORRIDOR_JSON:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_20260517a_edgepair/alignment_tc_student_vnext_dataset_report_20260517a_edgepair.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_inheritance_ablation_20260518a_edgepair_label_only}"

mkdir -p "$OUTPUT_ROOT"

run_planner_only() {
  local out_dir="$OUTPUT_ROOT/planner_only"
  mkdir -p "$out_dir"
  env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task_name "$TASK_NAME" \
    --episode_indices "$EPISODE_INDICES" \
    --use_stage_aware_refiner \
    --stage_refiner_mode planner_only \
    --collector_like_demo_reset \
    --target_provider_mode canonical_fallback \
    --enforce_no_privileged_runtime \
    --output_dir "$out_dir" \
    --record_video \
    --write_episode_videos \
    --no_best_gif \
    --record_gripper_trace \
    --max_steps "$MAX_STEPS" \
    --eval_seed "$EVAL_SEED"
}

run_teacher_residual() {
  local out_dir="$OUTPUT_ROOT/teacher_residual"
  mkdir -p "$out_dir"
  env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/collect_planner_state_expert_recovery.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task_name "$TASK_NAME" \
    --episode_indices "$EPISODE_INDICES" \
    --demo_max_attempts 10 \
    --max_rollout_steps "$MAX_STEPS" \
    --num_episodes 1 \
    --video_episodes 3 \
    --record_video \
    --output_dir "$out_dir" \
    --output_npz "$out_dir/alignment_tc_privileged_teacher_raw_teacher_residual.npz" \
    --report_json "$out_dir/alignment_tc_privileged_teacher_raw_report_teacher_residual.json" \
    --takeover_trace_jsonl "$out_dir/takeover_trace_teacher_residual.jsonl" \
    --planner_min_steps_before_takeover 8 \
    --planner_takeover_xy_threshold 0.075 \
    --planner_takeover_abs_z_threshold 0.140 \
    --planner_takeover_yaw_threshold 0.700 \
    --planner_takeover_yaw_guard_threshold 1.200 \
    --no_allow_broad_near_takeover \
    --no_fallback_to_best_broad_near \
    --force_alignment_probe_if_no_takeover \
    --planner_takeover_motion_steps 80 \
    --motion_corridor_force_descend_after_steps 12 \
    --motion_corridor_descend_xy_threshold 0.010 \
    --motion_corridor_descend_yaw_threshold 0.100 \
    --grasp_recovery_close_xy_threshold 0.0032 \
    --grasp_recovery_close_z_threshold 0.0035 \
    --grasp_recovery_close_yaw_threshold 0.025 \
    --grasp_recovery_verify_lift_threshold 0.010 \
    --grasp_recovery_verify_consecutive_steps 2 \
    --teacher_grasp_ready_threshold 0.55 \
    --teacher_close_contact_depth_threshold 0.020 \
    --record_edgepair_labels \
    --edgepair_label_target_mode commit
}

run_student_residual() {
  local out_dir="$OUTPUT_ROOT/student_residual"
  mkdir -p "$out_dir"
  env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task_name "$TASK_NAME" \
    --episode_indices "$EPISODE_INDICES" \
    --use_stage_aware_refiner \
    --stage_refiner_mode full \
    --collector_like_demo_reset \
    --target_provider_mode canonical_fallback \
    --enforce_no_privileged_runtime \
    --alignment_tc_student_vnext_ckpt "$STUDENT_CKPT" \
    --enable_alignment_tc_student_vnext_shadow \
    --enable_alignment_tc_student_vnext_apply \
    --alignment_tc_student_vnext_corridor_json "$CORRIDOR_JSON" \
    --alignment_tc_student_vnext_collector_like \
    --alignment_tc_diffusion_confidence_threshold 0.0 \
    --alignment_tc_diffusion_risk_threshold 1.1 \
    --alignment_tc_diffusion_soft_clamp \
    --alignment_tc_diffusion_workspace_soft_clamp \
    --record_video \
    --write_episode_videos \
    --no_best_gif \
    --record_gripper_trace \
    --max_steps "$MAX_STEPS" \
    --eval_seed "$EVAL_SEED" \
    --output_dir "$out_dir"
}

run_student_force_reflex() {
  local out_dir="$OUTPUT_ROOT/student_residual_force_reflex"
  mkdir -p "$out_dir"
  env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task_name "$TASK_NAME" \
    --episode_indices "$EPISODE_INDICES" \
    --use_stage_aware_refiner \
    --stage_refiner_mode full \
    --collector_like_demo_reset \
    --target_provider_mode canonical_fallback \
    --enforce_no_privileged_runtime \
    --alignment_tc_student_vnext_ckpt "$STUDENT_CKPT" \
    --enable_alignment_tc_student_vnext_shadow \
    --enable_alignment_tc_student_vnext_apply \
    --alignment_tc_student_vnext_corridor_json "$CORRIDOR_JSON" \
    --phase1_force_reflex_enable \
    --alignment_tc_student_vnext_collector_like \
    --alignment_tc_diffusion_confidence_threshold 0.0 \
    --alignment_tc_diffusion_risk_threshold 1.1 \
    --alignment_tc_diffusion_soft_clamp \
    --alignment_tc_diffusion_workspace_soft_clamp \
    --record_video \
    --write_episode_videos \
    --no_best_gif \
    --record_gripper_trace \
    --max_steps "$MAX_STEPS" \
    --eval_seed "$EVAL_SEED" \
    --output_dir "$out_dir"
}

run_planner_only
run_teacher_residual
run_student_residual
run_student_force_reflex

echo "[inheritance_ablation] done -> $OUTPUT_ROOT"
