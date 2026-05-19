#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4_shadow_heldout_20260429b}"
PLANNER_CKPT="${PLANNER_CKPT:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
V4_CKPT="${V4_CKPT:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4_residual_policy_20260429b/train/student_handoff_state_head_v2_alignment_v4_best_residual_policy.pt}"
HELDOUT_EPISODES="${HELDOUT_EPISODES:-1,5,8,10,16,17,19,20,22,24,25,27}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$OUT_ROOT"
[[ -d "$PLANNER_CKPT" ]] || { echo "ERROR: planner ckpt missing: $PLANNER_CKPT" >&2; exit 1; }
[[ -f "$ALIGNMENT_CKPT" ]] || { echo "ERROR: alignment ckpt missing: $ALIGNMENT_CKPT" >&2; exit 1; }
[[ -f "$TARGET_PROVIDER_CKPT" ]] || { echo "ERROR: target provider ckpt missing: $TARGET_PROVIDER_CKPT" >&2; exit 1; }
[[ -f "$V4_CKPT" ]] || { echo "ERROR: v4 ckpt missing: $V4_CKPT" >&2; exit 1; }

eval_root="$OUT_ROOT/eval"
support_npz="$OUT_ROOT/support_states.npz"
log="$OUT_ROOT/alignment_v4_shadow.log"
mkdir -p "$eval_root"

CUDA_VISIBLE_DEVICES="$GPU_ID" xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench_modes.py" \
  --checkpoint_dir "$PLANNER_CKPT" \
  --task_name insert_onto_square_peg \
  --modes learned_target_mainline \
  --num_episodes "$(awk -F, '{print NF}' <<< "$HELDOUT_EPISODES")" \
  --episode_indices "$HELDOUT_EPISODES" \
  --max_steps "${MAX_STEPS:-300}" \
  --output_root "$eval_root" \
  --name_suffix alignment_v4_residual_shadow \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
  --handoff_provider_ckpt "$V4_CKPT" \
  --student_handoff_shadow_only \
  --planner_no_depth --planner_no_force \
  --enable_alignment_close_veto \
  --close_veto_xy_threshold 0.006 \
  --close_veto_abs_z_threshold 0.003 \
  --close_veto_ready_streak_frames 1 \
  --close_veto_settle_steps 0 \
  --learned_residual_scale "${LEARNED_RESIDUAL_SCALE:-0.50}" \
  --max_residual_pos "${MAX_RESIDUAL_POS:-0.006}" \
  --max_alignment_corrections_per_window "${MAX_ALIGNMENT_CORRECTIONS_PER_WINDOW:-120}" \
  --outer_rescue_min_xy 0.10 \
  --outer_rescue_min_abs_z 0.30 \
  --close_latch_enabled --close_latch_steps 32 \
  --disable_alignment_physical_mask \
  --record_teacher_truth_metrics \
  --enforce_no_privileged_runtime \
  --record_video --write_episode_videos --no_best_gif \
  --support_states_output_npz "$support_npz" | tee "$log"

trace_dir="$(find "$eval_root" -maxdepth 2 -type d -name 'insert_*_learned_target_mainline_*' | sort | tail -n 1)"
if [[ -n "$trace_dir" ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/analyze_close_readiness_trace.py" \
    --trace_dir "$trace_dir" \
    --output_json "$OUT_ROOT/close_readiness_trace_report.json"
  "$PYTHON_BIN" "$ROOT/scripts/analyze_close_chain_trace.py" \
    --trace_dir "$trace_dir" \
    --output_json "$OUT_ROOT/close_chain_bucket_report.json"
  "$PYTHON_BIN" "$ROOT/scripts/audit_runtime_target_frame_trace.py" \
    --trace_dir "$trace_dir" \
    --output_json "$OUT_ROOT/runtime_target_frame_audit.json"
fi

echo "[alignment-v4-shadow] complete"
echo "[alignment-v4-shadow] out=$OUT_ROOT"
