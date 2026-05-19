#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"

BASELINE_HANDOFF_CKPT="${BASELINE_HANDOFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
CANDIDATE1_HANDOFF_CKPT="${CANDIDATE1_HANDOFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260426a/train/train_stageB_stabilize/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
CANDIDATE2_HANDOFF_CKPT="${CANDIDATE2_HANDOFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260426a/train/train_stageB_stabilize/student_handoff_state_head_v2_best_ready.pt}"

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260426a/shadow_eval}"
RECOLLECTION_EPISODES="${RECOLLECTION_EPISODES:-44,46,11,6,29,33,21,14,35,36,12,43,18,34,45,13,3,9,28,38}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_SEED="${EVAL_SEED:-3407}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/mnt/ssd/guoning/hf-cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HF_CACHE_ROOT/torch}"
export TIMM_HOME="${TIMM_HOME:-$HF_CACHE_ROOT/timm}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_eval_checkpoint_assets() {
  local ckpt="$1"
  [[ -d "$ckpt" ]] || { echo "ERROR: checkpoint_dir missing: $ckpt" >&2; exit 1; }
  compgen -G "$ckpt/action_queries--*.pt" > /dev/null || {
    echo "ERROR: missing action_queries--*.pt in $ckpt" >&2
    exit 1
  }
  compgen -G "$ckpt/proprio_projector--*checkpoint.pt" > /dev/null || {
    echo "ERROR: missing proprio_projector checkpoint in $ckpt" >&2
    exit 1
  }
  [[ -f "$ckpt/dataset_statistics.json" ]] || {
    echo "ERROR: missing dataset_statistics.json in $ckpt" >&2
    exit 1
  }
}

require_eval_checkpoint_assets "$CHECKPOINT_DIR"

HELDOUT_EPISODES="${HELDOUT_EPISODES:-$("$PYTHON_BIN" - <<PY
import random
seed = 3407
recollect = {int(x) for x in "$RECOLLECTION_EPISODES".split(",") if x.strip()}
remaining = [x for x in range(50) if x not in recollect]
rng = random.Random(seed)
picked = sorted(rng.sample(remaining, 12))
print(",".join(str(x) for x in picked))
PY
)}"
FOCUS_EPISODES="${FOCUS_EPISODES:-18,34,45}"

mkdir -p "$OUT_ROOT"

run_one_model() {
  local model_name="$1"
  local handoff_ckpt="$2"
  local episode_csv="$3"
  local subdir="${4:-merged}"
  local model_root="$OUT_ROOT/$model_name"
  local merged_dir="$model_root/$subdir"
  mkdir -p "$model_root/logs" "$merged_dir/gripper_traces"
  IFS=',' read -r -a EP_ARR <<< "$episode_csv"
  IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
  local N_GPU="${#GPU_ARR[@]}"
  declare -a SHARD_EPS=()
  for ((i=0; i<${#EP_ARR[@]}; i++)); do
    g=$((i % N_GPU))
    if [[ -z "${SHARD_EPS[$g]:-}" ]]; then
      SHARD_EPS[$g]="${EP_ARR[$i]}"
    else
      SHARD_EPS[$g]="${SHARD_EPS[$g]},${EP_ARR[$i]}"
    fi
  done
  declare -a PIDS=()
  declare -a SHARD_DIRS=()
  for ((g=0; g<N_GPU; g++)); do
    eps="${SHARD_EPS[$g]:-}"
    [[ -z "$eps" ]] && continue
    gpu="${GPU_ARR[$g]}"
    shard_name="shard$(printf '%02d' "$g")_gpu${gpu}"
    shard_eval_root="$model_root/$shard_name"
    log="$model_root/logs/${model_name}_${shard_name}.log"
    mkdir -p "$shard_eval_root"
    SHARD_DIRS+=("$shard_eval_root")
    echo "[phaseA-runtime-ready-shadow] $model_name launch $shard_name episodes=$eps"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" \
      xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench_modes.py" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --task_name insert_onto_square_peg \
        --modes learned_target_mainline \
        --num_episodes "$("$PYTHON_BIN" - <<PY
eps="$eps"
print(len([x for x in eps.split(",") if x.strip()]))
PY
)" \
      --episode_indices "$eps" \
        --max_steps "$MAX_STEPS" \
        --output_root "$shard_eval_root" \
        --name_suffix "${model_name}_${shard_name}" \
        --alignment_ckpt "$ALIGNMENT_CKPT" \
        --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
        --handoff_provider_ckpt "$handoff_ckpt" \
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
        --eval_seed "$EVAL_SEED" \
        --close_latch_enabled \
        --close_latch_steps 32 \
        --disable_alignment_physical_mask \
        --record_teacher_truth_metrics \
        --enforce_no_privileged_runtime \
        > "$log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
  merge_args=()
  for d in "${SHARD_DIRS[@]}"; do
    while IFS= read -r mode_dir; do
      merge_args+=(--input_dir "$mode_dir")
    done < <(find "$d" -maxdepth 1 -type d -name 'insert_vo40k_learned_target_mainline_*' | sort)
  done
  "$PYTHON_BIN" "$ROOT/scripts/merge_gripper_trace_dirs.py" "${merge_args[@]}" --output_dir "$merged_dir"
  "$PYTHON_BIN" "$ROOT/scripts/analyze_close_readiness_trace.py" \
    --trace_dir "$merged_dir" \
    --output_json "$merged_dir/close_readiness_trace_report.json"
  "$PYTHON_BIN" "$ROOT/scripts/analyze_close_chain_trace.py" \
    --trace_dir "$merged_dir" \
    --output_json "$merged_dir/close_chain_bucket_report.json"
  "$PYTHON_BIN" "$ROOT/scripts/audit_runtime_target_frame_trace.py" \
    --trace_dir "$merged_dir" \
    --output_json "$merged_dir/runtime_target_frame_audit.json"
  "$PYTHON_BIN" "$ROOT/scripts/audit_phasea_ready_runtime_alignment.py" \
    --dataset_npz "$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260426a/dataset/handoff_state_dataset_v2_runtime_ready_full.npz" \
    --trace_dir "$merged_dir" \
    --output_json "$merged_dir/phasea_ready_runtime_alignment_audit.json"
}

run_one_model baseline "$BASELINE_HANDOFF_CKPT" "$HELDOUT_EPISODES" "merged"
run_one_model best_phaseA_deploy "$CANDIDATE1_HANDOFF_CKPT" "$HELDOUT_EPISODES" "merged"
run_one_model best_ready "$CANDIDATE2_HANDOFF_CKPT" "$HELDOUT_EPISODES" "merged"

if ! python - <<PY
heldout = {int(x) for x in "$HELDOUT_EPISODES".split(",") if x.strip()}
focus = {int(x) for x in "$FOCUS_EPISODES".split(",") if x.strip()}
raise SystemExit(0 if focus.issubset(heldout) else 1)
PY
then
  run_one_model baseline "$BASELINE_HANDOFF_CKPT" "$FOCUS_EPISODES" "focus_diag"
  run_one_model best_phaseA_deploy "$CANDIDATE1_HANDOFF_CKPT" "$FOCUS_EPISODES" "focus_diag"
  run_one_model best_ready "$CANDIDATE2_HANDOFF_CKPT" "$FOCUS_EPISODES" "focus_diag"
fi

"$PYTHON_BIN" "$ROOT/scripts/compare_phaseA_shadow_runs.py" \
  --baseline_dir "$OUT_ROOT/baseline/merged" \
  --candidate_dir "$OUT_ROOT/best_phaseA_deploy/merged" \
  --candidate_name best_phaseA_deploy \
  --candidate_dir "$OUT_ROOT/best_ready/merged" \
  --candidate_name best_ready \
  --output_json "$OUT_ROOT/shadow_compare_summary.json"

echo "[phaseA-runtime-ready-shadow] heldout episodes=$HELDOUT_EPISODES"
echo "[phaseA-runtime-ready-shadow] focus episodes=$FOCUS_EPISODES"
