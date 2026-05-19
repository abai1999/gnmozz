#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/depth_force_contact/depth_force_contact_v1_20260430}"
DATASET_DIR="$OUT_ROOT/dataset"
TRAIN_DIR="$OUT_ROOT/train"
AUDIT_JSON="$OUT_ROOT/source_audit.json"

INPUT_NPZ="${INPUT_NPZ:-$ROOT/runtime_artifacts/stage_refiner/alignment_v4b_candidate_shadow_20260430_batches/v2_2_yawneeded/support_states.npz}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0001}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
cd "$ROOT"

mkdir -p "$OUT_ROOT" "$DATASET_DIR" "$TRAIN_DIR"

"$PYTHON_BIN" scripts/audit_depth_force_contact_sources.py \
  --input_npz "$INPUT_NPZ" \
  --output_json "$AUDIT_JSON"

"$PYTHON_BIN" scripts/build_depth_force_contact_dataset.py \
  --input_npz "$INPUT_NPZ" \
  --output_dir "$DATASET_DIR" \
  --candidate_score_std_min "${CANDIDATE_SCORE_STD_MIN:-0.05}" \
  --switch_margin "${SWITCH_MARGIN:-0.05}"

"$PYTHON_BIN" scripts/train_depth_force_contact_policy.py \
  --dataset_npz "$DATASET_DIR/depth_force_contact_dataset.npz" \
  --output_dir "$TRAIN_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR"

echo "[depth-force-contact-v1] done"
echo "[depth-force-contact-v1] audit=$AUDIT_JSON"
echo "[depth-force-contact-v1] dataset=$DATASET_DIR/depth_force_contact_dataset.npz"
echo "[depth-force-contact-v1] ckpt=$TRAIN_DIR/depth_force_contact_policy_best.pt"
