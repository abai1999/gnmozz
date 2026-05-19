#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
PYTHON_BIN="/home/guoning/my_conda_envs/vla-adapter/bin/python"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
COLLECT_MODE="${COLLECT_MODE:-learned_target_mainline}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

OUT_ROOT="${OUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/support_rows_targeted_parallel}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-seed3407_parallel}"
SUPPORT_OUT_DIR="${SUPPORT_OUT_DIR:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_parallel_collect}"
PLAN_JSON="${PLAN_JSON:-}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-300}"

mkdir -p "$OUT_ROOT" "$SUPPORT_OUT_DIR"

if [[ -z "$EPISODE_INDICES" ]]; then
  if [[ -z "$PLAN_JSON" ]]; then
    echo "ERROR: set EPISODE_INDICES or PLAN_JSON"
    exit 1
  fi
  EPISODE_INDICES="$($PYTHON_BIN - <<PY
import json
obj=json.load(open("$PLAN_JSON"))
print(obj.get("selected_episode_indices_csv",""))
PY
)"
fi

if [[ -z "$EPISODE_INDICES" ]]; then
  echo "ERROR: empty EPISODE_INDICES"
  exit 1
fi

IFS=',' read -r -a EP_ARR <<< "$EPISODE_INDICES"
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
N_GPU="${#GPU_ARR[@]}"
if [[ "$N_GPU" -le 0 ]]; then
  echo "ERROR: no GPU ids"
  exit 1
fi

echo "[parallel_collect] episodes=${#EP_ARR[@]} gpus=${GPU_IDS}"

declare -a SHARD_EPS
for ((i=0; i<${#EP_ARR[@]}; i++)); do
  g=$((i % N_GPU))
  if [[ -z "${SHARD_EPS[$g]:-}" ]]; then
    SHARD_EPS[$g]="${EP_ARR[$i]}"
  else
    SHARD_EPS[$g]="${SHARD_EPS[$g]},${EP_ARR[$i]}"
  fi
done

LOG_DIR="$SUPPORT_OUT_DIR/logs"
mkdir -p "$LOG_DIR"

declare -a PIDS
declare -a SHARD_NPZ
for ((g=0; g<N_GPU; g++)); do
  eps="${SHARD_EPS[$g]:-}"
  if [[ -z "$eps" ]]; then
    continue
  fi
  gpu="${GPU_ARR[$g]}"
  shard_name="shard${g}_gpu${gpu}"
  shard_npz="$SUPPORT_OUT_DIR/support_states_${shard_name}.npz"
  shard_out="$OUT_ROOT/${RUN_NAME_SUFFIX}_${shard_name}"
  log="$LOG_DIR/${RUN_NAME_SUFFIX}_${shard_name}.log"
  mkdir -p "$shard_out"
  SHARD_NPZ+=("$shard_npz")
  echo "[parallel_collect] launch ${shard_name} episodes=${eps}"
  nohup env \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    HF_CACHE_ROOT=/mnt/ssd/guoning/hf-cache HF_HOME=/mnt/ssd/guoning/hf-cache \
    HF_HUB_CACHE=/mnt/ssd/guoning/hf-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/ssd/guoning/hf-cache/hub \
    TRANSFORMERS_CACHE=/mnt/ssd/guoning/hf-cache/transformers TORCH_HOME=/mnt/ssd/guoning/hf-cache/torch \
    TIMM_HOME=/mnt/ssd/guoning/hf-cache/timm \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_LOCAL_FILES_ONLY=1 HF_HUB_DISABLE_TELEMETRY=1 \
    CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench_modes.py" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --task_name "$TASK_NAME" \
      --modes "$COLLECT_MODE" \
      --num_episodes "$(python - <<PY
eps="${eps}"
print(len([x for x in eps.split(",") if x.strip()]))
PY
)" \
      --max_steps "$MAX_STEPS" \
      --episode_indices "$eps" \
      --output_root "$shard_out" \
      --name_suffix "$RUN_NAME_SUFFIX" \
      --alignment_ckpt "$ALIGNMENT_CKPT" \
      --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
      --handoff_provider_ckpt "$HANDOFF_PROVIDER_CKPT" \
      --planner_no_depth \
      --planner_no_force \
      --record_teacher_truth_metrics \
      --enforce_no_privileged_runtime \
      --support_states_output_npz "$shard_npz" \
      ${EXTRA_EVAL_ARGS} \
      > "$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "ERROR: no shards launched"
  exit 1
fi

echo "[parallel_collect] launched pids: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
  wait "$pid"
done
echo "[parallel_collect] all shards finished"

for p in "${SHARD_NPZ[@]}"; do
  if [[ ! -s "$p" ]]; then
    echo "ERROR: missing shard npz $p"
    exit 1
  fi
done

MERGED_NPZ="$SUPPORT_OUT_DIR/support_states_merged.npz"
args=()
for p in "${SHARD_NPZ[@]}"; do
  args+=(--input_npz "$p")
done
"$PYTHON_BIN" "$ROOT/scripts/merge_support_states_npz.py" "${args[@]}" --output_npz "$MERGED_NPZ"
echo "[parallel_collect] merged npz: $MERGED_NPZ"
