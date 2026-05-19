#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/guoning/code/VLA"
PYTHON_BIN="/home/guoning/my_conda_envs/vla-adapter/bin/python"

CHECKPOINT_DIR="/home/guoning/code/VLA/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt"
ALIGNMENT_CKPT="/home/guoning/code/VLA/outputs/stage_refiner/insert_vo40k_posefield_scorer_onpolicy_plus_oracledistill_smoke_s3407_20260415f/pose_field_scorer_final.pt"

# Keep eval startup predictable on shared servers. Without these caps, model
# loading can spawn hundreds of BLAS/tokenizer threads and appear "hung".
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
export EVAL_USE_MERGED_STATE="${EVAL_USE_MERGED_STATE:-1}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_SEED="${EVAL_SEED:-3407}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_simple_alignment)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/guoning/code/VLA/eval_logs/insert_onto_square_peg/simple_alignment_baseline_${RUN_TAG}}"
NAME_SUFFIX="${NAME_SUFFIX:-seed${EVAL_SEED}_planner_vs_simple_alignment_mp4}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
MODES="${MODES:-planner_only,pose_alignment_only_basin}"
ALIGNMENT_CKPT_OVERRIDE="${ALIGNMENT_CKPT_OVERRIDE:-}"
TARGET_PROVIDER_MODE="${TARGET_PROVIDER_MODE:-teacher_oracle}"
CLOSE_VETO_XY_THRESHOLD="${CLOSE_VETO_XY_THRESHOLD:-0.006}"
CLOSE_VETO_ABS_Z_THRESHOLD="${CLOSE_VETO_ABS_Z_THRESHOLD:-0.003}"
CLOSE_VETO_READY_STREAK_FRAMES="${CLOSE_VETO_READY_STREAK_FRAMES:-1}"
CLOSE_VETO_SETTLE_STEPS="${CLOSE_VETO_SETTLE_STEPS:-0}"
ENABLE_CLOSE_LATCH="${ENABLE_CLOSE_LATCH:-1}"
CLOSE_LATCH_STEPS="${CLOSE_LATCH_STEPS:-32}"
LEARNED_RESIDUAL_SCALE="${LEARNED_RESIDUAL_SCALE:-0.50}"
MAX_RESIDUAL_POS="${MAX_RESIDUAL_POS:-0.006}"
MAX_ALIGNMENT_CORRECTIONS_PER_WINDOW="${MAX_ALIGNMENT_CORRECTIONS_PER_WINDOW:-120}"
OUTER_RESCUE_MIN_XY="${OUTER_RESCUE_MIN_XY:-0.10}"
OUTER_RESCUE_MIN_ABS_Z="${OUTER_RESCUE_MIN_ABS_Z:-0.30}"
USE_LEGACY_TEACHER_CANDIDATE_BANK="${USE_LEGACY_TEACHER_CANDIDATE_BANK:-1}"
ENABLE_ALIGNMENT_PHYSICAL_MASK="${ENABLE_ALIGNMENT_PHYSICAL_MASK:-0}"
GPU_CANDIDATES="${EVAL_GPU_CANDIDATES:-7,6,5,4,3,2,1,0}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-12000}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-12000}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-40}"

pick_idle_gpu() {
  IFS=',' read -r -a candidates <<< "$GPU_CANDIDATES"
  while IFS=',' read -r idx mem_used mem_total util; do
    idx="${idx// /}"
    mem_used="${mem_used// /}"
    mem_total="${mem_total// /}"
    util="${util// /}"
    free_mem=$((mem_total - mem_used))
    for cand in "${candidates[@]}"; do
      if [[ "$idx" == "$cand" ]] && (( mem_used <= MAX_USED_MEM_MB && util <= MAX_UTIL_PERCENT && free_mem >= MIN_FREE_MEM_MB )); then
        printf '%s\n' "$idx"
        return 0
      fi
    done
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
  return 1
}

GPU_ID="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_ID" ]]; then
  GPU_ID="$(pick_idle_gpu || true)"
fi
if [[ -z "$GPU_ID" ]]; then
  echo "[frozen_baseline_eval] no idle GPU found; set CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 1
fi

if [[ -n "$ALIGNMENT_CKPT_OVERRIDE" ]]; then
  ALIGNMENT_CKPT="$ALIGNMENT_CKPT_OVERRIDE"
fi

EXTRA_ARGS=()
if [[ "$ENABLE_CLOSE_LATCH" == "1" ]]; then
  EXTRA_ARGS+=(--close_latch_enabled --close_latch_steps "$CLOSE_LATCH_STEPS")
fi
if [[ -n "$EPISODE_INDICES" ]]; then
  EXTRA_ARGS+=(--episode_indices "$EPISODE_INDICES")
fi
if [[ "$USE_LEGACY_TEACHER_CANDIDATE_BANK" == "1" ]]; then
  EXTRA_ARGS+=(--use_legacy_teacher_candidate_bank_for_scorer)
fi
if [[ "$ENABLE_ALIGNMENT_PHYSICAL_MASK" == "1" ]]; then
  EXTRA_ARGS+=(--enable_alignment_physical_mask)
else
  EXTRA_ARGS+=(--disable_alignment_physical_mask)
fi
if [[ "$#" -gt 0 ]]; then
  EXTRA_ARGS+=("$@")
fi

echo "[frozen_baseline_eval] GPU=$GPU_ID"
echo "[frozen_baseline_eval] checkpoint=$CHECKPOINT_DIR"
echo "[frozen_baseline_eval] alignment_ckpt=$ALIGNMENT_CKPT"
echo "[frozen_baseline_eval] modes=$MODES"
if [[ -f "$CHECKPOINT_DIR/merged_eval_vla_state.pt" && "$EVAL_USE_MERGED_STATE" != "0" ]]; then
  echo "[frozen_baseline_eval] using merged_eval_vla_state.pt"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --modes "$MODES" \
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "$NAME_SUFFIX" \
  --alignment_ckpt "$ALIGNMENT_CKPT" \
  --target_provider_mode "$TARGET_PROVIDER_MODE" \
  --planner_no_depth \
  --planner_no_force \
  --enable_alignment_close_veto \
  --close_veto_xy_threshold "$CLOSE_VETO_XY_THRESHOLD" \
  --close_veto_abs_z_threshold "$CLOSE_VETO_ABS_Z_THRESHOLD" \
  --close_veto_ready_streak_frames "$CLOSE_VETO_READY_STREAK_FRAMES" \
  --close_veto_settle_steps "$CLOSE_VETO_SETTLE_STEPS" \
  --learned_residual_scale "$LEARNED_RESIDUAL_SCALE" \
  --max_residual_pos "$MAX_RESIDUAL_POS" \
  --max_alignment_corrections_per_window "$MAX_ALIGNMENT_CORRECTIONS_PER_WINDOW" \
  --outer_rescue_min_xy "$OUTER_RESCUE_MIN_XY" \
  --outer_rescue_min_abs_z "$OUTER_RESCUE_MIN_ABS_Z" \
  --eval_seed "$EVAL_SEED" \
  "${EXTRA_ARGS[@]}"
