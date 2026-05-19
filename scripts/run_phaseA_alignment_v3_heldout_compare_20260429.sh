#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"
TARGET_PROVIDER_CKPT="${TARGET_PROVIDER_CKPT:-$ROOT/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt}"
DATASET_NPZ="${DATASET_NPZ:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_teacher_augmented_20260429b_dyaw_aux_full/dataset/handoff_state_dataset_v2_alignment_v3_full.npz}"

STAGEA_HANODFF_CKPT="${STAGEA_HANODFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_teacher_augmented_20260429b_dyaw_aux_full/train/stageA_pairwise_progress/student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt}"
STAGEB_HANODFF_CKPT="${STAGEB_HANODFF_CKPT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_teacher_augmented_20260429b_dyaw_aux_full/train/stageB_pairwise_counterfactual/student_handoff_state_head_v2_alignment_v3_best_deploy_candidate.pt}"

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_alignment_v3_heldout_compare_20260429}"
GPU_ID="${GPU_ID:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_SEED="${EVAL_SEED:-3407}"
N_HELDOUT="${N_HELDOUT:-12}"
CLOSE_VETO_XY_THRESHOLD="${CLOSE_VETO_XY_THRESHOLD:-0.006}"
CLOSE_VETO_ABS_Z_THRESHOLD="${CLOSE_VETO_ABS_Z_THRESHOLD:-0.003}"
CLOSE_VETO_READY_STREAK_FRAMES="${CLOSE_VETO_READY_STREAK_FRAMES:-1}"
CLOSE_VETO_SETTLE_STEPS="${CLOSE_VETO_SETTLE_STEPS:-0}"
CLOSE_LATCH_STEPS="${CLOSE_LATCH_STEPS:-32}"
LEARNED_RESIDUAL_SCALE="${LEARNED_RESIDUAL_SCALE:-0.50}"
MAX_RESIDUAL_POS="${MAX_RESIDUAL_POS:-0.006}"
MAX_ALIGNMENT_CORRECTIONS_PER_WINDOW="${MAX_ALIGNMENT_CORRECTIONS_PER_WINDOW:-120}"
OUTER_RESCUE_MIN_XY="${OUTER_RESCUE_MIN_XY:-0.10}"
OUTER_RESCUE_MIN_ABS_Z="${OUTER_RESCUE_MIN_ABS_Z:-0.30}"

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

[[ -f "$DATASET_NPZ" ]] || { echo "ERROR: dataset npz missing: $DATASET_NPZ" >&2; exit 1; }
[[ -f "$STAGEA_HANODFF_CKPT" ]] || { echo "ERROR: stageA ckpt missing: $STAGEA_HANODFF_CKPT" >&2; exit 1; }
[[ -f "$STAGEB_HANODFF_CKPT" ]] || { echo "ERROR: stageB ckpt missing: $STAGEB_HANODFF_CKPT" >&2; exit 1; }
[[ -f "$CHECKPOINT_DIR/dataset_statistics.json" ]] || { echo "ERROR: planner checkpoint missing dataset_statistics.json: $CHECKPOINT_DIR" >&2; exit 1; }
[[ -f "$TARGET_PROVIDER_CKPT" ]] || { echo "ERROR: target provider ckpt missing: $TARGET_PROVIDER_CKPT" >&2; exit 1; }
[[ -f "$ALIGNMENT_CKPT" ]] || { echo "ERROR: alignment ckpt missing: $ALIGNMENT_CKPT" >&2; exit 1; }

HELDOUT_EPISODES="${HELDOUT_EPISODES:-$("$PYTHON_BIN" - <<PY
import numpy as np, random
raw = np.load("$DATASET_NPZ", allow_pickle=False)
used = sorted(set(int(x) for x in raw["episode_index"].tolist()))
remaining = [x for x in range(50) if x not in used]
rng = random.Random(int("$EVAL_SEED"))
take = min(int("$N_HELDOUT"), len(remaining))
picked = sorted(rng.sample(remaining, take))
print(",".join(str(x) for x in picked))
PY
)}"

mkdir -p "$OUT_ROOT"

run_candidate() {
  local label="$1"
  local handoff_ckpt="$2"
  local candidate_root="$OUT_ROOT/$label"
  local eval_root="$candidate_root/eval"
  local support_npz="$candidate_root/support_states.npz"
  local log="$candidate_root/${label}.log"
  mkdir -p "$candidate_root" "$eval_root"
  echo "[alignment-v3-heldout] running $label with ckpt=$handoff_ckpt"
  echo "[alignment-v3-heldout] heldout_episodes=$HELDOUT_EPISODES"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench_modes.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task_name insert_onto_square_peg \
    --modes learned_target_mainline \
    --num_episodes "$(awk -F, '{print NF}' <<< "$HELDOUT_EPISODES")" \
    --episode_indices "$HELDOUT_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --output_root "$eval_root" \
    --name_suffix "$label" \
    --alignment_ckpt "$ALIGNMENT_CKPT" \
    --target_provider_ckpt "$TARGET_PROVIDER_CKPT" \
    --handoff_provider_ckpt "$handoff_ckpt" \
    --student_handoff_shadow_only \
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
    --close_latch_enabled \
    --close_latch_steps "$CLOSE_LATCH_STEPS" \
    --disable_alignment_physical_mask \
    --record_teacher_truth_metrics \
    --enforce_no_privileged_runtime \
    --record_video \
    --write_episode_videos \
    --no_best_gif \
    --support_states_output_npz "$support_npz" \
    > "$log" 2>&1

  local trace_dir
  trace_dir="$(find "$eval_root" -maxdepth 2 -type d -name 'insert_*_learned_target_mainline_*' | sort | tail -n 1)"
  [[ -n "$trace_dir" ]] || { echo "ERROR: no trace dir found for $label" >&2; exit 1; }

  "$PYTHON_BIN" "$ROOT/scripts/analyze_close_readiness_trace.py" \
    --trace_dir "$trace_dir" \
    --output_json "$candidate_root/close_readiness_trace_report.json"
  "$PYTHON_BIN" "$ROOT/scripts/analyze_close_chain_trace.py" \
    --trace_dir "$trace_dir" \
    --output_json "$candidate_root/close_chain_bucket_report.json"
  "$PYTHON_BIN" "$ROOT/scripts/audit_runtime_target_frame_trace.py" \
    --trace_dir "$trace_dir" \
    --output_json "$candidate_root/runtime_target_frame_audit.json"

  local eval_results="$trace_dir/eval_results.json"
  if [[ -f "$eval_results" ]]; then
    cp "$eval_results" "$candidate_root/eval_results.json"
  fi
}

run_candidate "stageA_main_candidate" "$STAGEA_HANODFF_CKPT"
run_candidate "stageB_progress_baseline" "$STAGEB_HANODFF_CKPT"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

root = Path("$OUT_ROOT")
stagea = root / "stageA_main_candidate"
stageb = root / "stageB_progress_baseline"

def load(path):
    return json.loads(path.read_text()) if path.exists() else {}

def pick(report):
    return {
        "episode_count": report.get("summary", {}).get("episode_count"),
        "planner_close_episode_count": report.get("summary", {}).get("planner_close_episode_count"),
        "alignment_close_intent_episode_count": report.get("summary", {}).get("alignment_close_intent_episode_count"),
        "suppressed_after_alignment_episode_count": report.get("summary", {}).get("suppressed_after_alignment_episode_count"),
        "takeaways": report.get("takeaways", []),
    }

summary = {
    "decision": "stageA_main_candidate",
    "heldout_episode_indices_csv": "$HELDOUT_EPISODES",
    "main_candidate": {
        "label": "stageA_main_candidate",
        "ckpt": "$STAGEA_HANODFF_CKPT",
        "mp4_dir": str((stagea / "eval").resolve()),
        "close_readiness": pick(load(stagea / "close_readiness_trace_report.json")),
        "close_chain": load(stagea / "close_chain_bucket_report.json"),
        "runtime_target_frame": load(stagea / "runtime_target_frame_audit.json"),
        "eval_results": load(stagea / "eval_results.json"),
    },
    "comparison_baseline": {
        "label": "stageB_progress_baseline",
        "ckpt": "$STAGEB_HANODFF_CKPT",
        "mp4_dir": str((stageb / "eval").resolve()),
        "close_readiness": pick(load(stageb / "close_readiness_trace_report.json")),
        "close_chain": load(stageb / "close_chain_bucket_report.json"),
        "runtime_target_frame": load(stageb / "runtime_target_frame_audit.json"),
        "eval_results": load(stageb / "eval_results.json"),
    },
}
(root / "heldout_compare_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[alignment-v3-heldout] complete"
echo "[alignment-v3-heldout] summary=$OUT_ROOT/heldout_compare_summary.json"
