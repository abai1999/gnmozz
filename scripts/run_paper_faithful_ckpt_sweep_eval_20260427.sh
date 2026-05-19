#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CKPT_BASE="${CKPT_BASE:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train}"
CKPT_PREFIX="${CKPT_PREFIX:-configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b}"
STEPS_CSV="${STEPS_CSV:-20000,30000,40000,50000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
OUT_ROOT="${OUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg}"
RUN_TAG="${RUN_TAG:-paper_faithful_phasebucket_weighted_loaderfix_20260427b}"
EPISODES="${EPISODES:-0,1,2,3,4,5,6,7,8,9}"
MAX_STEPS="${MAX_STEPS:-300}"

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

mkdir -p "$OUT_ROOT"
IFS=',' read -r -a STEP_ARR <<< "$STEPS_CSV"
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"

if [[ "${#STEP_ARR[@]}" -gt "${#GPU_ARR[@]}" ]]; then
  echo "ERROR: need at least as many GPUs as checkpoints for this parallel sweep" >&2
  exit 1
fi

declare -a PIDS
for ((i=0; i<${#STEP_ARR[@]}; i++)); do
  step="${STEP_ARR[$i]}"
  gpu="${GPU_ARR[$i]}"
  ckpt_dir="$CKPT_BASE/${CKPT_PREFIX}--${step}_chkpt"
  out_dir="$OUT_ROOT/paper_faithful_phasebucket_weighted_${step}_loaderfix_20260427b"
  log="$OUT_ROOT/paper_faithful_phasebucket_weighted_${step}_loaderfix_20260427b.log"
  require_eval_checkpoint_assets "$ckpt_dir"
  echo "[paper-faithful-sweep] launch step=$step gpu=$gpu out=$out_dir"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" \
    xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
      --checkpoint_dir "$ckpt_dir" \
      --task_name insert_onto_square_peg \
      --num_episodes "$("$PYTHON_BIN" - <<PY
eps="$EPISODES"
print(len([x for x in eps.split(",") if x.strip()]))
PY
)" \
      --episode_indices "$EPISODES" \
      --max_steps "$MAX_STEPS" \
      --output_dir "$out_dir" \
      > "$log" 2>&1 &
  PIDS+=("$!")
done

echo "[paper-faithful-sweep] launched pids: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "[paper-faithful-sweep] complete"
