#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/guoning/code/VLA"
PYTHON_BIN="/home/guoning/my_conda_envs/vla-adapter/bin/python"
EVAL_WRAPPER="${EVAL_WRAPPER:-/home/guoning/code/VLA/scripts/run_nfcr_insert_eval_modes.sh}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/guoning/code/VLA/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
ORACLE_ALIGNMENT_CKPT="${ORACLE_ALIGNMENT_CKPT:-/home/guoning/code/VLA/outputs/stage_refiner/insert_vo40k_posefield_scorer_onpolicy_plus_oracledistill_smoke_s3407_20260415f/pose_field_scorer_final.pt}"

RUN_TAG="${RUN_TAG:-20260417a}"
TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
EVAL_SEED="${EVAL_SEED:-3407}"
NUM_EPISODES_COLLECT="${NUM_EPISODES_COLLECT:-30}"
NUM_EPISODES_EVAL="${NUM_EPISODES_EVAL:-10}"
MAX_STEPS="${MAX_STEPS:-300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/guoning/code/VLA/eval_logs/${TASK_NAME}/phase1_visual_base_set_${RUN_TAG}}"
DATA_ROOT="${DATA_ROOT:-/home/guoning/code/VLA/runtime_artifacts/residual_data}"
MODEL_ROOT="${MODEL_ROOT:-/home/guoning/code/VLA/runtime_artifacts/stage_refiner}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
# Default back to the old/simple alignment path. This keeps phase-1 alignment
# as a motion-only residual module; planner still owns gripper and downstream
# task flow. Set COLLECT_MODE/ORACLE_REGRESSION_MODE explicitly if a future
# experiment really needs the oracle-executed collector.
COLLECT_MODE="${COLLECT_MODE:-pose_alignment_only_basin}"
ORACLE_REGRESSION_MODE="${ORACLE_REGRESSION_MODE:-pose_alignment_only_basin}"
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
CLOSE_LATCH_ARGS=()
if [[ "$ENABLE_CLOSE_LATCH" == "1" ]]; then
  CLOSE_LATCH_ARGS+=(--close_latch_enabled --close_latch_steps "$CLOSE_LATCH_STEPS")
fi

SUPPORT_DIR="${DATA_ROOT}/insert_phase1_teacher_support_states_base_set_${RUN_TAG}"
SUPPORT_NPZ="${SUPPORT_DIR}/support_states.npz"
DATASET_DIR="${DATA_ROOT}/insert_phase1_visual_posefield_candidates_base_set_${RUN_TAG}"
DATASET_NPZ="${DATASET_DIR}/candidates.npz"
MODEL_DIR="${MODEL_ROOT}/insert_phase1_visual_posefield_rgb_base_set_${RUN_TAG}"
MODEL_CKPT="${MODEL_DIR}/pose_field_scorer_best_pose.pt"

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
  echo "[phase1_visual_base] no idle GPU found; set CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 1
fi

mkdir -p "$SUPPORT_DIR" "$DATASET_DIR" "$MODEL_DIR" "$OUTPUT_ROOT"

if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
  if find "$SUPPORT_DIR" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "[phase1_visual_base] support dir already contains files: $SUPPORT_DIR" >&2
    echo "[phase1_visual_base] choose a new RUN_TAG (or set ALLOW_OVERWRITE=1 explicitly)." >&2
    exit 1
  fi
  if find "$OUTPUT_ROOT" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "[phase1_visual_base] output root already contains files: $OUTPUT_ROOT" >&2
    echo "[phase1_visual_base] choose a new RUN_TAG (or set ALLOW_OVERWRITE=1 explicitly)." >&2
    exit 1
  fi
fi

cd "$REPO_ROOT"

echo "[phase1_visual_base] GPU=$GPU_ID"
echo "[phase1_visual_base] step1 collect teacher support rows"
if [[ "$PIPELINE_STAGE" == "collect" || "$PIPELINE_STAGE" == "all" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$EVAL_WRAPPER" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task_name "$TASK_NAME" \
    --modes "$COLLECT_MODE" \
    --num_episodes "$NUM_EPISODES_COLLECT" \
    --max_steps "$MAX_STEPS" \
    --output_root "$OUTPUT_ROOT" \
    --name_suffix "seed${EVAL_SEED}_collect_simple_alignment_base_set_mp4" \
    --alignment_ckpt "$ORACLE_ALIGNMENT_CKPT" \
    --target_provider_mode "teacher_oracle" \
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
    "${CLOSE_LATCH_ARGS[@]}" \
    --support_states_output_npz "$SUPPORT_NPZ" \
    --eval_seed "$EVAL_SEED"
fi

if [[ "$PIPELINE_STAGE" == "collect" ]]; then
  echo "[phase1_visual_base] collect-only done"
  exit 0
fi

echo "[phase1_visual_base] step2 build no-tilt visual dataset"
"$PYTHON_BIN" scripts/build_pose_candidate_dataset.py \
  --input_dir "$SUPPORT_NPZ" \
  --output_path "$DATASET_NPZ" \
  --exclude_occluded \
  --candidate_mode primitives \
  --no_primitive_include_tilt \
  --basin_radius_tilt -1.0 \
  --oracle_mode short_horizon_funnel \
  --support_close_intent_mode all_open

echo "[phase1_visual_base] step3 train pure-visual scorer"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_pose_field_scorer.py \
  --dataset_npz "$DATASET_NPZ" \
  --output_dir "$MODEL_DIR" \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-3 \
  --lambda_ready 0.0 \
  --no_target_context

echo "[phase1_visual_base] step4 run planner/oracle/visual regression"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$EVAL_WRAPPER" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --modes "planner_only,${ORACLE_REGRESSION_MODE}" \
  --num_episodes "$NUM_EPISODES_EVAL" \
  --max_steps "$MAX_STEPS" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "seed${EVAL_SEED}_regress_base_set_mp4" \
  --alignment_ckpt "$ORACLE_ALIGNMENT_CKPT" \
  --target_provider_mode "teacher_oracle" \
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
  "${CLOSE_LATCH_ARGS[@]}" \
  --eval_seed "$EVAL_SEED"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$EVAL_WRAPPER" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --modes "visual_scorer_mainline" \
  --num_episodes "$NUM_EPISODES_EVAL" \
  --max_steps "$MAX_STEPS" \
  --output_root "$OUTPUT_ROOT" \
  --name_suffix "seed${EVAL_SEED}_regress_base_set_mp4" \
  --alignment_ckpt "$MODEL_CKPT" \
  --target_provider_mode "canonical_fallback" \
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
  "${CLOSE_LATCH_ARGS[@]}" \
  --eval_seed "$EVAL_SEED"

echo "[phase1_visual_base] done"
