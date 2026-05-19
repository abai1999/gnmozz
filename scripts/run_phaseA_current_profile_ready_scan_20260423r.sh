#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_ready_neighborhood_micro_20260423q/train_ready_neighborhood_micro/student_handoff_state_head_v2_best_ready.pt}"

OUT_ROOT="${OUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/phaseA_current_profile_ready_scan_20260423r}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-current_profile_ready_scan50}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$OUT_ROOT/insert_vo40k_learned_target_mainline_${RUN_NAME_SUFFIX}_merged}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_SEED="${EVAL_SEED:-3407}"

if [[ -z "$EPISODE_INDICES" ]]; then
  EPISODE_INDICES="$(python - <<'PY'
print(",".join(str(i) for i in range(50)))
PY
)"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/ssd/guoning/hf-cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HF_CACHE_ROOT/torch}"
export TIMM_HOME="${TIMM_HOME:-$HF_CACHE_ROOT/timm}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a EP_ARR <<< "$EPISODE_INDICES"
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
N_GPU="${#GPU_ARR[@]}"
if [[ "$N_GPU" -le 0 ]]; then
  echo "ERROR: no GPU ids" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

declare -a SHARD_EPS
for ((i=0; i<${#EP_ARR[@]}; i++)); do
  g=$((i % N_GPU))
  if [[ -z "${SHARD_EPS[$g]:-}" ]]; then
    SHARD_EPS[$g]="${EP_ARR[$i]}"
  else
    SHARD_EPS[$g]="${SHARD_EPS[$g]},${EP_ARR[$i]}"
  fi
done

declare -a PIDS
declare -a SHARD_DIRS
for ((g=0; g<N_GPU; g++)); do
  eps="${SHARD_EPS[$g]:-}"
  [[ -z "$eps" ]] && continue
  gpu="${GPU_ARR[$g]}"
  shard_name="shard${g}_gpu${gpu}"
  out_dir="$OUT_ROOT/${RUN_NAME_SUFFIX}_${shard_name}"
  log="$LOG_DIR/${RUN_NAME_SUFFIX}_${shard_name}.log"
  mkdir -p "$out_dir"
  SHARD_DIRS+=("$out_dir")
  echo "[current_profile_ready_scan] launch ${shard_name} episodes=${eps}"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" \
    xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench_modes.py" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --task_name insert_onto_square_peg \
      --modes learned_target_mainline \
      --num_episodes "$(python - <<PY
eps="${eps}"
print(len([x for x in eps.split(",") if x.strip()]))
PY
)" \
      --episode_indices "$eps" \
      --max_steps "$MAX_STEPS" \
      --output_root "$out_dir" \
      --name_suffix "$shard_name" \
      --alignment_ckpt "$ALIGNMENT_CKPT" \
      --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
      --handoff_provider_ckpt "$HANDOFF_PROVIDER_CKPT" \
      --student_handoff_shadow_only \
      --planner_no_depth \
      --planner_no_force \
      --enable_alignment_close_veto \
      --close_veto_xy_threshold 0.006 \
      --close_veto_abs_z_threshold 0.003 \
      --close_veto_ready_streak_frames 3 \
      --close_veto_settle_steps 0 \
      --learned_residual_scale 0.50 \
      --max_residual_pos 0.006 \
      --max_alignment_corrections_per_window 120 \
      --outer_rescue_min_xy 0.10 \
      --outer_rescue_min_abs_z 0.30 \
      --eval_seed "$EVAL_SEED" \
      --close_latch_enabled \
      --close_latch_steps 32 \
      --use_legacy_teacher_candidate_bank_for_scorer \
      --disable_alignment_physical_mask \
      --record_teacher_truth_metrics \
      --enforce_no_privileged_runtime \
      > "$log" 2>&1 &
  PIDS+=("$!")
done

echo "[current_profile_ready_scan] launched pids: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

args=()
for d in "${SHARD_DIRS[@]}"; do
  # evaluate_rlbench_modes nests the actual mode output one directory below.
  while IFS= read -r mode_dir; do
    args+=(--input_dir "$mode_dir")
  done < <(find "$d" -maxdepth 1 -type d -name 'insert_vo40k_learned_target_mainline_*' | sort)
done
"$PYTHON_BIN" "$ROOT/scripts/merge_gripper_trace_dirs.py" "${args[@]}" --output_dir "$MERGED_OUT_DIR"
echo "[current_profile_ready_scan] merged traces at $MERGED_OUT_DIR"
