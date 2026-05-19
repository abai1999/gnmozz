#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

SOURCE_NPZ="${SOURCE_NPZ:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_from_scratch_raw_rollout_oraclecollect_3ep_20260507b/raw_rollout_support_states.npz}"
OUTPUT_NPZ="${OUTPUT_NPZ:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_from_scratch_teacher_20260507c.npz}"
REPORT_JSON="${REPORT_JSON:-$ROOT/runtime_artifacts/depth_force_contact/alignment_v3_from_scratch_teacher_report_20260507c.json}"

exec "$PYTHON_BIN" scripts/build_alignment_v3_from_scratch_teacher.py \
  --source_npz "$SOURCE_NPZ" \
  --output_npz "$OUTPUT_NPZ" \
  --report_json "$REPORT_JSON" \
  --enforce_audit_pass
