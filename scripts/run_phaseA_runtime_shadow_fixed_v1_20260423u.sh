#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

# Runtime profile frozen for one iteration:
# - do not change provider booleanization / streak / close-veto rules
# - only swap handoff ckpt for fair model comparison

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_main_v1_20260423u/train_minimal/student_handoff_state_head_v2_best_phaseA_deploy.pt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/phaseA_runtime_shadow_fixed_v1_20260423u}"
NAME_SUFFIX="${NAME_SUFFIX:-phaseA_fixed_v1_shadow}"
EPISODE_INDICES="${EPISODE_INDICES:-11,14}"
NUM_EPISODES="${NUM_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_SEED="${EVAL_SEED:-3407}"

export PYTHONUNBUFFERED=1
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$("$ROOT/scripts/choose_idle_gpu.sh" 2>/dev/null || echo 0)"
fi

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT"

exec env xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name insert_onto_square_peg \
  --modes learned_target_mainline \
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --episode_indices "$EPISODE_INDICES" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "$NAME_SUFFIX" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
  --handoff_provider_ckpt "$HANDOFF_PROVIDER_CKPT" \
  --student_handoff_shadow_only \
  --planner_no_depth \
  --planner_no_force \
  --enable_alignment_close_veto \
  --close_veto_xy_threshold 0.006 \
  --close_veto_abs_z_threshold 0.003 \
  --close_veto_ready_streak_frames 1 \
  --close_veto_settle_steps 0 \
  --learned_residual_scale 0.50 \
  --max_residual_pos 0.006 \
  --max_alignment_corrections_per_window 120 \
  --outer_rescue_min_xy 0.10 \
  --outer_rescue_min_abs_z 0.30 \
  --close_latch_enabled \
  --close_latch_steps 32 \
  --use_legacy_teacher_candidate_bank_for_scorer \
  --disable_alignment_physical_mask \
  --record_teacher_truth_metrics \
  --enforce_no_privileged_runtime \
  --eval_seed "$EVAL_SEED" \
  "$@"

