#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/guoning/code/VLA/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
BOOTSTRAP_ALIGNMENT_CKPT="${BOOTSTRAP_ALIGNMENT_CKPT:-/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_unified_20260418h_all_test/pose_field_scorer_best_pose.pt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/guoning/code/VLA/eval_logs/insert_onto_square_peg/truthready_anchor_collect_20260420j}"
SUPPORT_DIR="${SUPPORT_DIR:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthready_anchor_support_20260420j}"
SUPPORT_NPZ="${SUPPORT_NPZ:-$SUPPORT_DIR/support_states.npz}"
NAME_SUFFIX="${NAME_SUFFIX:-seed3407_truthready_anchor_collect}"
NUM_EPISODES="${NUM_EPISODES:-80}"
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
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HF_CACHE_ROOT/torch}"
export TIMM_HOME="${TIMM_HOME:-$HF_CACHE_ROOT/timm}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if gpu="$("$REPO_ROOT/scripts/choose_idle_gpu.sh" 2>/dev/null)"; then
    export CUDA_VISIBLE_DEVICES="$gpu"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$SUPPORT_DIR"
cd "$REPO_ROOT"

exec xvfb-run -a "$PYTHON_BIN" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name insert_onto_square_peg \
  --modes oracle_executed_pregrasp_collect \
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --stop_on_success \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "$NAME_SUFFIX" \
  --alignment_ckpt "$BOOTSTRAP_ALIGNMENT_CKPT" \
  --planner_no_depth \
  --planner_no_force \
  --no_depth \
  --no_force \
  --no_video \
  --record_teacher_truth_metrics \
  --support_states_output_npz "$SUPPORT_NPZ" \
  --eval_seed "$EVAL_SEED"
