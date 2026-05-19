#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
RUN_TAG="${RUN_TAG:-20260419a_parallel_20260427b}"
SEED="${SEED:-3407}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_EPISODES="${NUM_EPISODES:-40}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
ALIGNMENT_TAKEOVER_UNTIL_CLOSE_READY="${ALIGNMENT_TAKEOVER_UNTIL_CLOSE_READY:-1}"
TEACHER_MOTION_ENTRY_XY_THRESHOLD="${TEACHER_MOTION_ENTRY_XY_THRESHOLD:-0.040}"
TEACHER_MOTION_ENTRY_ABS_Z_THRESHOLD="${TEACHER_MOTION_ENTRY_ABS_Z_THRESHOLD:-0.120}"
TEACHER_REQUIRE_ALIGNMENT_READY_FOR_MOTION_GATE="${TEACHER_REQUIRE_ALIGNMENT_READY_FOR_MOTION_GATE:-0}"
TEACHER_HANDOFF_REVOKE_XY_THRESHOLD="${TEACHER_HANDOFF_REVOKE_XY_THRESHOLD:-0.012}"
TEACHER_HANDOFF_REVOKE_ABS_Z_THRESHOLD="${TEACHER_HANDOFF_REVOKE_ABS_Z_THRESHOLD:-0.025}"
TEACHER_CLOSE_XY_THRESHOLD="${TEACHER_CLOSE_XY_THRESHOLD:-0.006}"
TEACHER_CLOSE_ABS_Z_THRESHOLD="${TEACHER_CLOSE_ABS_Z_THRESHOLD:-0.005}"
TEACHER_CLOSE_YAW_THRESHOLD="${TEACHER_CLOSE_YAW_THRESHOLD:-0.12}"

# Canonical teacher lineage from the original 20260419a pipeline.
# Keep the current 30k frozen planner, but use the closest existing bootstrap
# scorer lineage so we can re-collect teacher data with the tight canonical gates.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"

OUT_ROOT="${OUT_ROOT:-$ROOT/eval_logs/$TASK_NAME/phase1_demo_grasp_yaw_teacher_${RUN_TAG}}"
SUPPORT_DIR="${SUPPORT_DIR:-$ROOT/runtime_artifacts/residual_data/insert_phase1_teacher_support_states_demo_grasp_yaw_${RUN_TAG}}"
MERGED_SUPPORT_NPZ="${MERGED_SUPPORT_NPZ:-$SUPPORT_DIR/support_states.npz}"
ANCHOR_DATASET_DIR="${ANCHOR_DATASET_DIR:-$ROOT/runtime_artifacts/stage_refiner/teacher_success_anchor_stage0_${RUN_TAG}}"
ANCHOR_DATASET_NPZ="${ANCHOR_DATASET_NPZ:-$ANCHOR_DATASET_DIR/teacher_success_anchor_dataset.npz}"
ANCHOR_WINDOW_NPZ="${ANCHOR_WINDOW_NPZ:-$ANCHOR_DATASET_DIR/teacher_success_anchor_window_dataset.npz}"

mkdir -p "$OUT_ROOT" "$SUPPORT_DIR" "$ANCHOR_DATASET_DIR"

EVAL_EXTRA_ARGS=()
if [[ "$ALIGNMENT_TAKEOVER_UNTIL_CLOSE_READY" != "1" ]]; then
  EVAL_EXTRA_ARGS+=(--disable_alignment_takeover_until_close_ready)
fi
TEACHER_EXTRA_ARGS=()
if [[ "$TEACHER_REQUIRE_ALIGNMENT_READY_FOR_MOTION_GATE" == "1" ]]; then
  TEACHER_EXTRA_ARGS+=(--teacher_require_alignment_ready_for_motion_gate)
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-/home/guoning/CoppeliaSim}"
export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:-/home/guoning/CoppeliaSim}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/home/guoning/CoppeliaSim:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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

cd "$ROOT"

IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
if [[ "${#GPU_ARR[@]}" -le 0 ]]; then
  echo "ERROR: no GPU ids supplied" >&2
  exit 1
fi

episode_indices=()
for ((i=0; i<NUM_EPISODES; i++)); do
  episode_indices+=("$i")
done

declare -a shard_episode_groups
for ((i=0; i<${#episode_indices[@]}; i++)); do
  g=$((i % ${#GPU_ARR[@]}))
  if [[ -z "${shard_episode_groups[$g]:-}" ]]; then
    shard_episode_groups[$g]="${episode_indices[$i]}"
  else
    shard_episode_groups[$g]="${shard_episode_groups[$g]},${episode_indices[$i]}"
  fi
done

LOG_DIR="$SUPPORT_DIR/logs"
mkdir -p "$LOG_DIR"

declare -a shard_npzs
declare -a pids
for ((g=0; g<${#GPU_ARR[@]}; g++)); do
  eps="${shard_episode_groups[$g]:-}"
  [[ -z "$eps" ]] && continue
  gpu="${GPU_ARR[$g]}"
  shard_name="shard${g}_gpu${gpu}"
  shard_out="$OUT_ROOT/${RUN_TAG}_${shard_name}"
  shard_npz="$SUPPORT_DIR/support_states_${shard_name}.npz"
  shard_log="$LOG_DIR/${RUN_TAG}_${shard_name}.log"
  mkdir -p "$shard_out"
  shard_npzs+=("$shard_npz")
  echo "[0419a-teacher] launch ${shard_name} episodes=${eps}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS="$OMP_NUM_THREADS"
    export MKL_NUM_THREADS="$MKL_NUM_THREADS"
    export OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
    export NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS"
    export TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
    export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
    export HF_CACHE_ROOT="$HF_CACHE_ROOT"
    export HF_HOME="$HF_HOME"
    export HF_HUB_CACHE="$HF_HUB_CACHE"
    export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
    export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
    export TORCH_HOME="$TORCH_HOME"
    export TIMM_HOME="$TIMM_HOME"
    export HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
    export TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
    export HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE"
    export HF_LOCAL_FILES_ONLY="$HF_LOCAL_FILES_ONLY"
    export HF_HUB_DISABLE_TELEMETRY="$HF_HUB_DISABLE_TELEMETRY"
    export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
    IFS=',' read -r -a _eps_arr <<< "$eps"
    shard_num_episodes="${#_eps_arr[@]}"
    xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench_modes.py" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --task_name "$TASK_NAME" \
      --modes oracle_executed_pregrasp_collect \
      --alignment_ckpt "$ALIGNMENT_CKPT" \
      --teacher_close_xy_threshold "$TEACHER_CLOSE_XY_THRESHOLD" \
      --teacher_close_abs_z_threshold "$TEACHER_CLOSE_ABS_Z_THRESHOLD" \
      --teacher_close_yaw_threshold "$TEACHER_CLOSE_YAW_THRESHOLD" \
      --teacher_motion_entry_xy_threshold "$TEACHER_MOTION_ENTRY_XY_THRESHOLD" \
      --teacher_motion_entry_abs_z_threshold "$TEACHER_MOTION_ENTRY_ABS_Z_THRESHOLD" \
      --teacher_handoff_revoke_xy_threshold "$TEACHER_HANDOFF_REVOKE_XY_THRESHOLD" \
      --teacher_handoff_revoke_abs_z_threshold "$TEACHER_HANDOFF_REVOKE_ABS_Z_THRESHOLD" \
      --num_episodes "$shard_num_episodes" \
      --max_steps "$MAX_STEPS" \
      --episode_indices "$eps" \
      --output_root "$shard_out" \
      --name_suffix "$RUN_TAG" \
      --planner_no_depth --planner_no_force \
      --record_teacher_truth_metrics \
      --record_video --write_episode_videos --no_best_gif \
      --run_full_horizon_on_success \
      --support_states_output_npz "$shard_npz" \
      --eval_seed "$SEED" \
      "${TEACHER_EXTRA_ARGS[@]}" \
      "${EVAL_EXTRA_ARGS[@]}" \
      >"$shard_log" 2>&1
  ) &
  pids+=($!)
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "ERROR: no shards launched" >&2
  exit 1
fi

echo "[0419a-teacher] launched pids: ${pids[*]}"
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "[0419a-teacher] collection finished"

for p in "${shard_npzs[@]}"; do
  [[ -s "$p" ]] || { echo "ERROR: missing shard npz $p" >&2; exit 1; }
done

MERGE_ARGS=()
for p in "${shard_npzs[@]}"; do
  MERGE_ARGS+=(--input_npz "$p")
done

"$PYTHON_BIN" "$ROOT/scripts/merge_support_states_npz.py" \
  "${MERGE_ARGS[@]}" \
  --output_npz "$MERGED_SUPPORT_NPZ"

echo "[0419a-teacher] merged support npz: $MERGED_SUPPORT_NPZ"

"$PYTHON_BIN" "$ROOT/scripts/build_teacher_success_anchor_dataset.py" \
  --support_npz "$MERGED_SUPPORT_NPZ" \
  --source_name "phase1_yaw_teacher_20260419a_parallel" \
  --source_weight_mult 1.0 \
  --output_npz "$ANCHOR_DATASET_NPZ" \
  --meta_json "$ANCHOR_DATASET_DIR/teacher_success_anchor_dataset.meta.json"

"$PYTHON_BIN" "$ROOT/scripts/build_teacher_success_anchor_window_dataset.py" \
  --input_npz "$ANCHOR_DATASET_NPZ" \
  --output_npz "$ANCHOR_WINDOW_NPZ" \
  --meta_json "$ANCHOR_DATASET_DIR/teacher_success_anchor_window_dataset.meta.json"

echo "[0419a-teacher] anchor dataset: $ANCHOR_DATASET_NPZ"
echo "[0419a-teacher] anchor window: $ANCHOR_WINDOW_NPZ"
echo "[0419a-teacher] done"
