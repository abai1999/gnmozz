#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

EPISODE_INDICES="${EPISODE_INDICES:-18,34,45}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/b2_candidate_bounded_v0_smoke_20260425e}"
MERGED_RUN_DIR="${MERGED_RUN_DIR:-$OUTPUT_ROOT/merged_b2_bounded_v0_smoke}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
B2_CANDIDATE_CKPT="${B2_CANDIDATE_CKPT:-$ROOT/runtime_artifacts/stage_refiner/b2_yawmode_dataset_v13_selected_profile_20260425c/runtime_trace_v14e_wide_no_legacy_yawaug_20260425/student_candidate_evaluator_v2_v14e_runtime_trace.pt}"
B2_APPLY_CONF_THRESHOLD="${B2_APPLY_CONF_THRESHOLD:-0.431}"
B2_APPLY_MARGIN_THRESHOLD="${B2_APPLY_MARGIN_THRESHOLD:-0.010}"
CLOSE_VETO_READY_STREAK_FRAMES="${CLOSE_VETO_READY_STREAK_FRAMES:-1}"
# Bounded-v0 default: enable conservative runtime-geometry fallback so close-veto
# can open when handoff-ready is shadow-blocked but geometry is already aligned.
CLOSE_VETO_RUNTIME_GEOMETRY_FALLBACK_FOR_BOUNDED="${CLOSE_VETO_RUNTIME_GEOMETRY_FALLBACK_FOR_BOUNDED:-1}"
ENABLE_BOUNDED_AUTO_CLOSE_ON_ALIGNMENT="${ENABLE_BOUNDED_AUTO_CLOSE_ON_ALIGNMENT:-0}"
BOUNDED_AUTO_CLOSE_STABLE_FRAMES="${BOUNDED_AUTO_CLOSE_STABLE_FRAMES:-1}"
BOUNDED_AUTO_CLOSE_XY_THRESHOLD="${BOUNDED_AUTO_CLOSE_XY_THRESHOLD:--1}"
BOUNDED_AUTO_CLOSE_ABS_Z_THRESHOLD="${BOUNDED_AUTO_CLOSE_ABS_Z_THRESHOLD:--1}"
BOUNDED_AUTO_CLOSE_YAW_THRESHOLD="${BOUNDED_AUTO_CLOSE_YAW_THRESHOLD:--1}"
ENABLE_FORCE_CLOSE_AFTER_B2_EVAL="${ENABLE_FORCE_CLOSE_AFTER_B2_EVAL:-0}"
USE_LEGACY_TEACHER_CANDIDATE_BANK_FOR_SCORER="${USE_LEGACY_TEACHER_CANDIDATE_BANK_FOR_SCORER:-0}"
WRITE_BEST_GIF="${WRITE_BEST_GIF:-0}"
RUNTIME_CANDIDATE_YAW_PROBES="${RUNTIME_CANDIDATE_YAW_PROBES:-0.06,0.12}"
B2_SHADOW_YAW_PROBES="${B2_SHADOW_YAW_PROBES:-}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-3}"
GPU_IDS="${GPU_IDS:-}"

EXTRA_ARGS=()
if [[ "$CLOSE_VETO_RUNTIME_GEOMETRY_FALLBACK_FOR_BOUNDED" == "1" ]]; then
  EXTRA_ARGS+=(--close_veto_runtime_geometry_fallback_for_bounded)
fi
if [[ "$ENABLE_BOUNDED_AUTO_CLOSE_ON_ALIGNMENT" == "1" ]]; then
  EXTRA_ARGS+=(
    --enable_bounded_auto_close_on_alignment
    --bounded_auto_close_stable_frames "$BOUNDED_AUTO_CLOSE_STABLE_FRAMES"
    --bounded_auto_close_xy_threshold "$BOUNDED_AUTO_CLOSE_XY_THRESHOLD"
    --bounded_auto_close_abs_z_threshold "$BOUNDED_AUTO_CLOSE_ABS_Z_THRESHOLD"
    --bounded_auto_close_yaw_threshold "$BOUNDED_AUTO_CLOSE_YAW_THRESHOLD"
  )
fi
if [[ "$ENABLE_FORCE_CLOSE_AFTER_B2_EVAL" == "1" ]]; then
  EXTRA_ARGS+=(--enable_force_close_after_b2_eval)
fi
if [[ "$USE_LEGACY_TEACHER_CANDIDATE_BANK_FOR_SCORER" == "1" ]]; then
  EXTRA_ARGS+=(--use_legacy_teacher_candidate_bank_for_scorer)
else
  EXTRA_ARGS+=(--no_legacy_teacher_candidate_bank_for_scorer)
fi
if [[ "$WRITE_BEST_GIF" == "1" ]]; then
  EXTRA_ARGS+=(--write_best_gif)
else
  EXTRA_ARGS+=(--no_best_gif)
fi

if [[ -z "$GPU_IDS" ]]; then
  mapfile -t GPU_ARR < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v max_jobs="$MAX_PARALLEL_JOBS" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4);
        free=$3-$2;
        if ($2 <= 1024 && $4 <= 20 && free >= 16000) {
          print $1;
          count++;
          if (count >= max_jobs) exit;
        }
      }')
else
  IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
fi

if (( ${#GPU_ARR[@]} == 0 )); then
  echo "[b2-bounded-v0-smoke] no idle GPU found; falling back to auto runner default" >&2
  GPU_ARR=("")
fi

IFS=',' read -r -a EP_ARR <<< "$EPISODE_INDICES"
mkdir -p "$OUTPUT_ROOT"
rm -rf "$MERGED_RUN_DIR"
mkdir -p "$MERGED_RUN_DIR/gripper_traces"

PIDS=()
SHARD_DIRS=()
ACTIVE_PIDS=()
ACTIVE_LIMIT="${#GPU_ARR[@]}"
fail=0
for i in "${!EP_ARR[@]}"; do
  ep="${EP_ARR[$i]}"
  gpu="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  shard_name="shard$(printf '%02d' "$i")_ep${ep}"
  shard_root="$OUTPUT_ROOT/$shard_name"
  SHARD_DIRS+=("$shard_root")
  echo "[b2-bounded-v0-smoke] launch $shard_name gpu=${gpu:-auto} episode=$ep"
  (
    if [[ -n "$gpu" ]]; then
      export CUDA_VISIBLE_DEVICES="$gpu"
    else
      unset CUDA_VISIBLE_DEVICES || true
    fi
    env xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --task_name insert_onto_square_peg \
      --modes learned_target_mainline \
      --num_episodes 1 \
      --max_steps 340 \
      --episode_indices "$ep" \
      --output_root "$shard_root" \
      --name_suffix "b2_bounded_v0_smoke_${shard_name}" \
      --alignment_ckpt "$ALIGNMENT_CKPT" \
      --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
      --handoff_provider_ckpt "$HANDOFF_PROVIDER_CKPT" \
      --student_handoff_shadow_only \
      --student_candidate_evaluator_shadow_ckpt "$B2_CANDIDATE_CKPT" \
      --student_candidate_evaluator_handoff_ckpt "$HANDOFF_PROVIDER_CKPT" \
      --student_candidate_evaluator_mode_input_path summary_only \
      --b2_candidate_shadow_gate_mode nearish_only \
      --b2_candidate_shadow_yaw_probe_values "$B2_SHADOW_YAW_PROBES" \
      --runtime_candidate_yaw_probe_values "$RUNTIME_CANDIDATE_YAW_PROBES" \
      --enable_b2_candidate_bounded_v0 \
      --b2_candidate_apply_conf_threshold "$B2_APPLY_CONF_THRESHOLD" \
      --b2_candidate_apply_margin_threshold "$B2_APPLY_MARGIN_THRESHOLD" \
      --planner_no_depth \
      --planner_no_force \
      --enable_alignment_close_veto \
      --close_veto_xy_threshold 0.006 \
      --close_veto_abs_z_threshold 0.003 \
      --close_veto_ready_streak_frames "$CLOSE_VETO_READY_STREAK_FRAMES" \
      --close_veto_settle_steps 0 \
      --learned_residual_scale 0.50 \
      --max_residual_pos 0.006 \
      --max_alignment_corrections_per_window 120 \
      --outer_rescue_min_xy 0.10 \
      --outer_rescue_min_abs_z 0.30 \
      --eval_seed 3407 \
      --close_latch_enabled \
      --close_latch_steps 32 \
      --disable_alignment_physical_mask \
      --record_teacher_truth_metrics \
      --enforce_no_privileged_runtime \
      "${EXTRA_ARGS[@]}"
  ) >"$shard_root.log" 2>&1 &
  pid="$!"
  PIDS+=("$pid")
  ACTIVE_PIDS+=("$pid")
  if (( ${#ACTIVE_PIDS[@]} >= ACTIVE_LIMIT )); then
    first_pid="${ACTIVE_PIDS[0]}"
    if ! wait "$first_pid"; then
      fail=1
    fi
    ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
  fi
done

echo "[b2-bounded-v0-smoke] launched pids: ${PIDS[*]}"
for pid in "${ACTIVE_PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if (( fail != 0 )); then
  echo "[b2-bounded-v0-smoke] one or more shards failed" >&2
  exit 1
fi

for shard_root in "${SHARD_DIRS[@]}"; do
  while IFS= read -r trace; do
    cp "$trace" "$MERGED_RUN_DIR/gripper_traces/"
  done < <(find "$shard_root" -path '*/gripper_traces/*_gripper_trace.jsonl' -type f | sort)
done

"$PYTHON_BIN" "$ROOT/scripts/analyze_b2_candidate_shadow_trace.py" \
  --trace_dir "$MERGED_RUN_DIR" \
  --output_json "$MERGED_RUN_DIR/b2_shadow_trace_analysis.json" \
  --focus_output_json "$MERGED_RUN_DIR/b2_shadow_focus_episode_diagnostics.json" \
  --gate_output_json "$MERGED_RUN_DIR/b2_shadow_gate_decision.json"

"$PYTHON_BIN" "$ROOT/scripts/analyze_close_chain_trace.py" \
  --trace_dir "$MERGED_RUN_DIR" \
  --output_json "$MERGED_RUN_DIR/close_chain_bucket_report.json"

"$PYTHON_BIN" "$ROOT/scripts/audit_runtime_target_frame_trace.py" \
  --trace_dir "$MERGED_RUN_DIR" \
  --output_json "$MERGED_RUN_DIR/runtime_target_frame_audit.json"

"$PYTHON_BIN" "$ROOT/scripts/analyze_close_readiness_trace.py" \
  --trace_dir "$MERGED_RUN_DIR" \
  --output_json "$MERGED_RUN_DIR/close_readiness_trace_report.json"

"$PYTHON_BIN" "$ROOT/scripts/visualize_b2_candidate_shadow_trace.py" \
  --trace_dir "$MERGED_RUN_DIR" \
  --output_dir "$MERGED_RUN_DIR/visualizations" \
  --focus_episodes "18,34,45"

echo "[b2-bounded-v0-smoke] reports and visualizations at $MERGED_RUN_DIR"
