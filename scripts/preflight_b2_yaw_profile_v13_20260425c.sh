#!/usr/bin/env bash
set -euo pipefail

# Build candidate/yaw-mode diagnostics for a single pilot support collection.
# This is the gate we run before any larger B2-yaw recollection.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT to a pilot directory}"
SUPPORT_NPZ="${SUPPORT_NPZ:-$SCRATCH_ROOT/support/support_states_merged.npz}"
OUT_PREFIX="${OUT_PREFIX:-$SCRATCH_ROOT/preflight}"
FOCUS_EPISODES="${FOCUS_EPISODES:-18,23,34,45}"
SOURCE_NAME="${SOURCE_NAME:-$(basename "$SCRATCH_ROOT")_candidates}"

mkdir -p "$OUT_PREFIX"

# build_b1b2_actioncentric_dataset_v1 infers source_name from the candidate
# file parent.  Put candidates under an explicit source-named directory so a
# pilot profile is not accidentally classified as "other" and filtered out.
SOURCE_DIR="$OUT_PREFIX/$SOURCE_NAME"
mkdir -p "$SOURCE_DIR"
CAND_NPZ="$SOURCE_DIR/candidates.npz"
ACTION_NPZ="$OUT_PREFIX/actioncentric.npz"
ACTION_META="$OUT_PREFIX/actioncentric.meta.json"
YAWBANK_NPZ="$OUT_PREFIX/yawprobe_bank.npz"
YAWBANK_META="$OUT_PREFIX/yawprobe_bank.meta.json"
YAWMODE_NPZ="$OUT_PREFIX/yawmode.npz"
MANIFEST_JSON="$OUT_PREFIX/manifest.json"
GATE_JSON="$OUT_PREFIX/gate_report.json"
EP_AUDIT_JSON="$OUT_PREFIX/episode_audit.json"
FAIL_AUDIT_JSON="$OUT_PREFIX/failure_audit.json"
PROFILE_SUMMARY_JSON="$OUT_PREFIX/profile_summary.json"

"$PY" "$ROOT/scripts/build_pose_candidate_dataset.py" \
  --input_dir "$SUPPORT_NPZ" \
  --output_path "$CAND_NPZ" \
  --candidate_mode primitives \
  --force_rebuild_candidate_bank \
  --primitive_yaw_probe_values "0.06,0.12" \
  --oracle_mode stage_handoff_joint \
  --recompute_oracle_labels

"$PY" "$ROOT/scripts/build_b1b2_actioncentric_dataset_v1.py" \
  --input_npz "$CAND_NPZ" \
  --output_npz "$ACTION_NPZ" \
  --meta_json "$ACTION_META" \
  --allow_insufficient

"$PY" "$ROOT/scripts/build_b2_yaw_probe_candidate_dataset.py" \
  --input_npz "$ACTION_NPZ" \
  --output_npz "$YAWBANK_NPZ" \
  --meta_json "$YAWBANK_META" \
  --yaw_steps="-0.16,-0.12,-0.08,-0.04,0.0,0.04,0.08,0.12,0.16" \
  --compressed

"$PY" "$ROOT/scripts/build_b2_yawmode_dataset_v11.py" \
  --input_npz "$YAWBANK_NPZ" \
  --output_npz "$YAWMODE_NPZ" \
  --manifest_json "$MANIFEST_JSON" \
  --gate_report_json "$GATE_JSON" \
  --episode_audit_json "$EP_AUDIT_JSON" \
  --schema_name b2_yawmode_profile_preflight_v13_20260425c \
  --require_complete_yaw_bank \
  --supervised_profiles runtime_like,targeted_yawapply \
  --diagnostic_profiles oracle,teacher_assisted \
  --min_yaw_apply_eps 1 \
  --min_yaw_keep_eps 1 \
  --min_val_yaw_apply_eps 0 \
  --min_val_yaw_keep_eps 0 \
  --min_train_yaw_apply_eps 1 \
  --min_train_yaw_keep_eps 1 \
  --focus_episodes "$FOCUS_EPISODES" \
  --compressed

"$PY" "$ROOT/scripts/audit_b2_yaw_failure_v11.py" \
  --dataset_npz "$YAWMODE_NPZ" \
  --output_json "$FAIL_AUDIT_JSON" \
  --focus_episodes "$FOCUS_EPISODES"

"$PY" - "$YAWMODE_NPZ" "$PROFILE_SUMMARY_JSON" "$FOCUS_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

dataset = Path(sys.argv[1])
out = Path(sys.argv[2])
focus = [int(x) for x in sys.argv[3].split(",") if x.strip()]
d = np.load(dataset, allow_pickle=False)
ep = np.asarray(d["episode_index"], dtype=np.int64)
src = np.asarray(d["source_name"]).astype(str)
prof = np.asarray(d["source_profile_v12"]).astype(str)
label = np.asarray(d["yaw_mode3_label_v11"], dtype=np.int64)
valid = np.asarray(d["yaw_mode_valid_v11"], dtype=np.float32) > 0.5
supervised = np.asarray(d["yaw_mode_supervised_profile_v13"], dtype=np.float32) > 0.5
names = {0: "keep", 1: "small", 2: "apply"}

summary = {}
for e in focus:
    m0 = (ep == e) & valid & supervised
    by_profile = {}
    by_source = {}
    for arr, bucket in ((prof, by_profile), (src, by_source)):
        for key in sorted(set(arr[m0].tolist())):
            mm = m0 & (arr == key)
            counts = {names.get(int(k), str(k)): int(np.sum(label[mm] == k)) for k in sorted(set(label[mm].tolist()))}
            bucket[str(key)] = {"rows": int(np.sum(mm)), **counts}
    same_profile_ok = any(v.get("keep", 0) > 0 and v.get("apply", 0) > 0 for v in by_profile.values())
    same_source_ok = any(v.get("keep", 0) > 0 and v.get("apply", 0) > 0 for v in by_source.values())
    summary[str(e)] = {
        "valid_supervised_rows": int(np.sum(m0)),
        "same_profile_keep_apply": bool(same_profile_ok),
        "same_source_keep_apply": bool(same_source_ok),
        "by_profile": by_profile,
        "by_source": by_source,
    }

overall = {
    "same_profile_episodes": [int(k) for k, v in summary.items() if v["same_profile_keep_apply"]],
    "same_source_episodes": [int(k) for k, v in summary.items() if v["same_source_keep_apply"]],
    "focus_summary": summary,
}
json.dump(overall, out.open("w"), indent=2)
print(json.dumps(overall, indent=2))
PY

echo "[b2-yaw-profile-preflight] wrote $PROFILE_SUMMARY_JSON"
