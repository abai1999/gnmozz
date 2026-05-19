#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

OUT_DIR="${OUT_DIR:-$ROOT/runtime_artifacts/residual_data/b1b2_current_profile_candidate_v3_yawbank_20260424a}"
YAW_PROBES="${YAW_PROBES:-0.06,0.12}"

mkdir -p "$OUT_DIR"

declare -A INPUTS=(
  [learned32]="$ROOT/runtime_artifacts/residual_data/insert_phase1_support_resync_learned_full32_20260421h/support_states.npz"
  [lateprofile_readypos]="$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_readypos_parallel8_20260422d/support_states_merged.npz"
  [lateprofile]="$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_parallel8_20260421m/support_states_merged.npz"
  [oracleub_t2]="$ROOT/runtime_artifacts/residual_data/insert_phase1_current_profile_teacher_assisted_oracleub_recollect_20260423t2/support_states_merged.npz"
  [teacher_assisted_v3]="$ROOT/runtime_artifacts/residual_data/insert_phase1_current_profile_teacher_assisted_v3_20260423u/support_states_merged.npz"
  [teacher_assisted_v3b]="$ROOT/runtime_artifacts/residual_data/insert_phase1_current_profile_teacher_assisted_v3b_20260423u/support_states_merged.npz"
  [teacher_assisted_v3c_topup]="$ROOT/runtime_artifacts/residual_data/insert_phase1_current_profile_teacher_assisted_v3c_topup_20260423u/support_states_merged.npz"
  [teacher_assisted_yawneeded]="$ROOT/runtime_artifacts/residual_data/insert_phase1_current_profile_teacher_assisted_yawneeded_20260423v/support_states_merged.npz"
  [lateprofile_next]="$ROOT/runtime_artifacts/residual_data/insert_phase1_support_targeted_lateprofile_next_parallel8_20260422c/support_states_merged.npz"
  [b1b2_recollect_y]="$ROOT/runtime_artifacts/residual_data/insert_phase1_b1b2_candidate_recollect_20260423y/support_states_merged.npz"
)

for NAME in "${!INPUTS[@]}"; do
  INPUT="${INPUTS[$NAME]}"
  OUTPUT="$OUT_DIR/${NAME}_candidates.npz"
  if [[ ! -f "$INPUT" ]]; then
    echo "[yawbank] skip missing $NAME -> $INPUT"
    continue
  fi
  if [[ -f "$OUTPUT" ]]; then
    echo "[yawbank] skip existing $NAME -> $OUTPUT"
    continue
  fi
  echo "[yawbank] rebuilding $NAME"
  "$PYTHON_BIN" "$ROOT/scripts/build_pose_candidate_dataset.py" \
    --input_dir "$INPUT" \
    --output_path "$OUTPUT" \
    --candidate_mode primitives \
    --force_rebuild_candidate_bank \
    --primitive_yaw_probe_values "$YAW_PROBES" \
    --oracle_mode stage_handoff_joint \
    --recompute_oracle_labels
done

echo "[yawbank] done -> $OUT_DIR"
