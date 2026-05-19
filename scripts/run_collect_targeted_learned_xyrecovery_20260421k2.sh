#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
PYTHON_BIN="/home/guoning/my_conda_envs/vla-adapter/bin/python"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt}"
TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
HANDOFF_PROVIDER_CKPT="${HANDOFF_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_sanity_20260421f/student_handoff_state_head_v2_best.pt}"

OUT_ROOT="${OUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/support_rows_targeted_xyrecovery_20260421k2}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-seed3407_targeted16_xyrecovery_v2_shadow}"
SUPPORT_NPZ="${SUPPORT_NPZ:-$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_learned16_xyrecovery_20260421k2/support_states.npz}"
PLAN_JSON="${PLAN_JSON:-$ROOT/runtime_artifacts/residual_data/insert_phase1_targeted_recollect_xyrecovery_20260421k2/targeted_recollection_plan_v2.json}"

GPU_ID="${GPU_ID:-7}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
if [[ -z "$EPISODE_INDICES" ]]; then
  EPISODE_INDICES="$($PYTHON_BIN - <<'PY'
import json
from pathlib import Path
p = Path("/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_targeted_recollect_xyrecovery_20260421k2/targeted_recollection_plan_v2.json")
obj = json.loads(p.read_text())
print(obj["selected_episode_indices_csv"])
PY
)"
fi

mkdir -p "$(dirname "$SUPPORT_NPZ")"
mkdir -p "$OUT_ROOT"

echo "[targeted_collect_xyrecovery] episode_indices=${EPISODE_INDICES}"
echo "[targeted_collect_xyrecovery] support_npz=${SUPPORT_NPZ}"

env \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  HF_CACHE_ROOT=/mnt/ssd/guoning/hf-cache HF_HOME=/mnt/ssd/guoning/hf-cache \
  HF_HUB_CACHE=/mnt/ssd/guoning/hf-cache/hub HUGGINGFACE_HUB_CACHE=/mnt/ssd/guoning/hf-cache/hub \
  TRANSFORMERS_CACHE=/mnt/ssd/guoning/hf-cache/transformers TORCH_HOME=/mnt/ssd/guoning/hf-cache/torch \
  TIMM_HOME=/mnt/ssd/guoning/hf-cache/timm \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_LOCAL_FILES_ONLY=1 HF_HUB_DISABLE_TELEMETRY=1 \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench_modes.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task_name "$TASK_NAME" \
    --modes learned_target_mainline \
    --num_episodes 16 \
    --max_steps 300 \
    --episode_indices "$EPISODE_INDICES" \
    --output_root "$OUT_ROOT" \
    --name_suffix "$RUN_NAME_SUFFIX" \
    --alignment_ckpt "$ALIGNMENT_CKPT" \
    --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
    --handoff_provider_ckpt "$HANDOFF_PROVIDER_CKPT" \
    --planner_no_depth \
    --planner_no_force \
    --record_teacher_truth_metrics \
    --enforce_no_privileged_runtime \
    --support_states_output_npz "$SUPPORT_NPZ"

