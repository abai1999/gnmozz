#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
PYTHON_BIN="/home/guoning/my_conda_envs/vla-adapter/bin/python"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
GPU_IDS="${GPU_IDS:-1,2,3,4,5,6,7}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-teacher_oracle_diag50_parallel}"
OUT_ROOT="${OUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/phaseA_oracle_diag_readyset_20260422k_parallel}"
MERGED_OUT_DIR="${MERGED_OUT_DIR:-$OUT_ROOT/insert_vo40k_teacher_oracle_diag50_parallel_merged}"
MAX_STEPS="${MAX_STEPS:-300}"

if [[ -z "$EPISODE_INDICES" ]]; then
  EPISODE_INDICES="$(python - <<'PY'
print(",".join(str(i) for i in range(50)))
PY
)"
fi

IFS=',' read -r -a EP_ARR <<< "$EPISODE_INDICES"
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
N_GPU="${#GPU_ARR[@]}"
if [[ "$N_GPU" -le 0 ]]; then
  echo "ERROR: no GPU ids"
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
  echo "[oracle_diag_parallel] launch ${shard_name} episodes=${eps}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    VLA_PLATFORM=RLBENCH \
    QT_QPA_PLATFORM=xcb \
    QT_QPA_PLATFORM_PLUGIN_PATH=/home/guoning/CoppeliaSim \
    LIBGL_ALWAYS_SOFTWARE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    xvfb-run -a -s "-screen 0 1280x1024x24" \
    "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --task_name "$TASK_NAME" \
      --num_episodes "$(python - <<PY
eps="${eps}"
print(len([x for x in eps.split(",") if x.strip()]))
PY
)" \
      --episode_indices "$eps" \
      --max_steps "$MAX_STEPS" \
      --output_dir "$out_dir" \
      --use_stage_aware_refiner \
      --stage_refiner_mode alignment \
      --alignment_ckpt "$ALIGNMENT_CKPT" \
      --target_provider_mode teacher_oracle \
      --allow_privileged_runtime \
      --record_teacher_truth_metrics \
      --oracle_executed_align_collect \
      --oracle_executed_pregrasp_collect \
      --allow_alignment_without_close_intent \
      --max_alignment_corrections_per_window 20 \
      --disable_outer_rescue \
      --disable_alignment_close_veto \
      --no_video \
      --no_episode_videos \
      --no_best_gif \
      --eval_seed 3407 \
      > "$log" 2>&1 &
  PIDS+=("$!")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "ERROR: no shards launched"
  exit 1
fi

echo "[oracle_diag_parallel] launched pids: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

args=()
for d in "${SHARD_DIRS[@]}"; do
  args+=(--input_dir "$d")
done
"$PYTHON_BIN" "$ROOT/scripts/merge_gripper_trace_dirs.py" "${args[@]}" --output_dir "$MERGED_OUT_DIR"
echo "[oracle_diag_parallel] merged traces at $MERGED_OUT_DIR"
