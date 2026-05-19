#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v2c_worsening_recollect_20260427}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-phaseA_alignment_v2c_worsening_recollect_20260427}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EPISODES="${EPISODES:-18,34,45,46,28,3,6,9}"
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
  compgen -G "$ckpt/action_queries--*.pt" > /dev/null || { echo "ERROR: missing action_queries--*.pt in $ckpt" >&2; exit 1; }
  compgen -G "$ckpt/proprio_projector--*checkpoint.pt" > /dev/null || { echo "ERROR: missing proprio_projector checkpoint in $ckpt" >&2; exit 1; }
  [[ -f "$ckpt/dataset_statistics.json" ]] || { echo "ERROR: missing dataset_statistics.json in $ckpt" >&2; exit 1; }
}

require_eval_checkpoint_assets "$CHECKPOINT_DIR"

mkdir -p "$OUT_ROOT/logs"
IFS=',' read -r -a EP_ARR <<< "$EPISODES"
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
N_GPU="${#GPU_ARR[@]}"
[[ "$N_GPU" -gt 0 ]] || { echo "ERROR: no GPU ids configured" >&2; exit 1; }

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
declare -a SUPPORT_NPZS
for ((g=0; g<N_GPU; g++)); do
  eps="${SHARD_EPS[$g]:-}"
  [[ -z "$eps" ]] && continue
  gpu="${GPU_ARR[$g]}"
  shard_name="shard$(printf '%02d' "$g")_gpu${gpu}"
  shard_root="$OUT_ROOT/$shard_name"
  shard_eval_root="$shard_root/eval"
  support_npz="$shard_root/support_states_${shard_name}.npz"
  log="$OUT_ROOT/logs/${RUN_NAME_SUFFIX}_${shard_name}.log"
  mkdir -p "$shard_root" "$shard_eval_root"
  SHARD_DIRS+=("$shard_eval_root")
  SUPPORT_NPZS+=("$support_npz")
  echo "[alignment-v2c-worsening-recollect] launch $shard_name episodes=$eps"
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
      --support_states_output_npz "$support_npz" \
      > "$log" 2>&1 &
  PIDS+=("$!")
done

echo "[alignment-v2c-worsening-recollect] launched pids: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

support_json="$(printf '%s\n' "${SUPPORT_NPZS[@]}" | "$PYTHON_BIN" -c 'import json, sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
cat > "$OUT_ROOT/recollection_manifest.json" <<JSON
{
  "purpose": "focused close-intent + metric-valid + worsening support rows for Alignment v2c/v2d",
  "episode_indices_csv": "$EPISODES",
  "checkpoint_dir": "$CHECKPOINT_DIR",
  "support_states_npz": $support_json
}
JSON

echo "[alignment-v2c-worsening-recollect] complete"
echo "[alignment-v2c-worsening-recollect] manifest=$OUT_ROOT/recollection_manifest.json"
echo "[alignment-v2c-worsening-recollect] logs=$OUT_ROOT/logs"
