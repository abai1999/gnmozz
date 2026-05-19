#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/guoning/miniconda3/etc/profile.d/conda.sh
conda activate vla-adapter

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/stage_refiner/phaseA_runtime_ready_v20260426a}"
RECOLLECT_ROOT="${RECOLLECT_ROOT:-$OUT_ROOT/recollection}"
MERGED_TRACE_DIR="${MERGED_TRACE_DIR:-$RECOLLECT_ROOT/merged_learned_target_mainline_phaseA_runtime_ready_recollect_v20260426a}"
BUILDER_OUT_DIR="${BUILDER_OUT_DIR:-$OUT_ROOT/dataset}"
TRAIN_ROOT="${TRAIN_ROOT:-$OUT_ROOT/train}"
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/stage_refiner/student_handoff_state_v2_phaseA_20260422l/train_run12_stage1b_calibmix/student_handoff_state_head_v2_best_phaseA_deploy.pt}"
CONSISTENCY_CKPT="${CONSISTENCY_CKPT:-$INIT_CKPT}"

mkdir -p "$BUILDER_OUT_DIR" "$TRAIN_ROOT"

if [[ ! -f "$RECOLLECT_ROOT/recollection_manifest.json" ]]; then
  echo "ERROR: recollection manifest missing: $RECOLLECT_ROOT/recollection_manifest.json" >&2
  exit 1
fi

mapfile -t SUPPORT_NPZS < <(python - <<PY
import json, pathlib
p = pathlib.Path("$RECOLLECT_ROOT/recollection_manifest.json")
data = json.loads(p.read_text())
for item in data["support_states_npz"]:
    print(item)
PY
)

READY_GUARD_JSON="$OUT_ROOT/retrain_preflight_guard.json"
python - <<PY
import json
import numpy as np
from pathlib import Path

support_npzs = [Path(p) for p in """${SUPPORT_NPZS[*]}""".split() if p.strip()]
teacher_ready = 0.0
runtime_ready = 0.0
rows = 0
for p in support_npzs:
    raw = np.load(p, allow_pickle=False)
    rows += int(raw["episode_index"].shape[0]) if "episode_index" in raw.files else 0
    if "teacher_truth_handoff_ready" in raw.files:
        teacher_ready += float(np.asarray(raw["teacher_truth_handoff_ready"], dtype=np.float32).sum())
    if "runtime_handoff_ready" in raw.files:
        runtime_ready += float(np.asarray(raw["runtime_handoff_ready"], dtype=np.float32).sum())
guard = {
    "support_npz_count": len(support_npzs),
    "rows": int(rows),
    "teacher_truth_handoff_ready_sum": float(teacher_ready),
    "runtime_handoff_ready_sum": float(runtime_ready),
    "decision": "proceed" if teacher_ready > 0.0 else "blocked_no_teacher_ready_rows",
}
Path("$READY_GUARD_JSON").write_text(json.dumps(guard, indent=2))
print(json.dumps(guard, indent=2))
if teacher_ready <= 0.0:
    raise SystemExit(2)
PY

RECOLLECTION_EPISODES="$(python - <<PY
import json, pathlib
p = pathlib.Path("$RECOLLECT_ROOT/recollection_manifest.json")
data = json.loads(p.read_text())
print(data["recollection_episode_indices_csv"])
PY
)"

builder_args=()
for path in "${SUPPORT_NPZS[@]}"; do
  builder_args+=(--input_npz "$path")
done
python "$ROOT/scripts/build_phaseA_runtime_ready_dataset_v20260426a.py" \
  "${builder_args[@]}" \
  --output_dir "$BUILDER_OUT_DIR" \
  --recollection_episode_csv "$RECOLLECTION_EPISODES"

FULL_NPZ="$BUILDER_OUT_DIR/handoff_state_dataset_v2_runtime_ready_full.npz"
STAGEA_NPZ="$BUILDER_OUT_DIR/handoff_state_dataset_v2_runtime_ready_stageA.npz"
VAL_EPISODES_CSV="$(tr -d '\n' < "$BUILDER_OUT_DIR/val_episode_csv.txt")"

python "$ROOT/scripts/audit_phasea_ready_runtime_alignment.py" \
  --dataset_npz "$FULL_NPZ" \
  --trace_dir "$MERGED_TRACE_DIR" \
  --output_json "$BUILDER_OUT_DIR/phasea_ready_runtime_alignment_audit.json"

STAGEA_OUT="$TRAIN_ROOT/train_stageA_ready_lift"
mkdir -p "$STAGEA_OUT"
python "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$STAGEA_NPZ" \
  --output_dir "$STAGEA_OUT" \
  --epochs 4 \
  --batch_size 64 \
  --lr 1e-5 \
  --val_episode_csv "$VAL_EPISODES_CSV" \
  --seed 3407 \
  --lambda_xy 1.6 \
  --lambda_z 0.8 \
  --lambda_yaw 1.0 \
  --lambda_band 0.5 \
  --lambda_ready 0.45 \
  --lambda_uncertainty 0.0 \
  --lambda_teacher_ready_push 0.20 \
  --lambda_close_ready_crossing 0.35 \
  --lambda_z_aware_ready_crossing 0.20 \
  --lambda_late_ready_logit_lift 0.10 \
  --lambda_ready_neighborhood_consistency 0.0 \
  --lambda_far_negative_calib 0.0 \
  --lambda_far_negative_hard 0.0 \
  --lambda_current_profile_hard_negative_veto 0.0 \
  --weighted_sampling \
  --sampler_weight_power 1.0 \
  --init_ckpt "$INIT_CKPT" \
  --consistency_ckpt "$CONSISTENCY_CKPT" \
  --deploy_false_ready_max 0.002 \
  2>&1 | tee "$STAGEA_OUT/stdout.log"

STAGEB_OUT="$TRAIN_ROOT/train_stageB_stabilize"
mkdir -p "$STAGEB_OUT"
python "$ROOT/scripts/train_student_handoff_state_head_v2.py" \
  --dataset_npz "$FULL_NPZ" \
  --output_dir "$STAGEB_OUT" \
  --epochs 3 \
  --batch_size 64 \
  --lr 5e-6 \
  --val_episode_csv "$VAL_EPISODES_CSV" \
  --seed 3407 \
  --lambda_xy 1.6 \
  --lambda_z 0.8 \
  --lambda_yaw 1.0 \
  --lambda_band 0.5 \
  --lambda_ready 0.45 \
  --lambda_uncertainty 0.0 \
  --lambda_teacher_ready_push 0.20 \
  --lambda_close_ready_crossing 0.35 \
  --lambda_z_aware_ready_crossing 0.20 \
  --lambda_late_ready_logit_lift 0.10 \
  --lambda_ready_neighborhood_consistency 0.05 \
  --lambda_far_negative_calib 0.25 \
  --lambda_far_negative_hard 0.10 \
  --lambda_consistency_band 0.02 \
  --lambda_consistency_ready 0.05 \
  --consistency_ckpt "$CONSISTENCY_CKPT" \
  --consistency_source runtime_broad_negative_v1 \
  --lambda_current_profile_hard_negative_veto 0.0 \
  --weighted_sampling \
  --sampler_weight_power 1.0 \
  --init_ckpt "$STAGEA_OUT/student_handoff_state_head_v2_best_ready.pt" \
  --deploy_false_ready_max 0.002 \
  2>&1 | tee "$STAGEB_OUT/stdout.log"

echo "[phaseA-runtime-ready-retrain] builder_dir=$BUILDER_OUT_DIR train_root=$TRAIN_ROOT val_eps=$VAL_EPISODES_CSV"
