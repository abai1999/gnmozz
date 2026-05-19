#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

SUPPORT_NPZ="${SUPPORT_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthdiag_support_20260420g/support_states.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_near_ready_xyyaw_20260420g}"
DATASET_NPZ="${DATASET_NPZ:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthdiag_support_20260420g/near_ready_xyyaw_dataset.npz}"
META_JSON="${META_JSON:-/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthdiag_support_20260420g/near_ready_xyyaw_dataset.meta.json}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUTPUT_DIR/offline_summary.json}"
POLL_SECONDS="${POLL_SECONDS:-20}"

mkdir -p "$(dirname "$OUTPUT_DIR")" "$(dirname "$DATASET_NPZ")" "$OUTPUT_DIR"

echo "[pipeline] waiting for support NPZ: $SUPPORT_NPZ"
while [[ ! -f "$SUPPORT_NPZ" ]]; do
  sleep "$POLL_SECONDS"
done

echo "[pipeline] support NPZ detected: $SUPPORT_NPZ"

cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/build_near_ready_xyyaw_dataset.py \
  --support_npz "$SUPPORT_NPZ" \
  --output_npz "$DATASET_NPZ" \
  --meta_json "$META_JSON" \
  --z_near_mult 4.0 \
  --xy_window_mult 4.0 \
  --yaw_window_mult 2.0

echo "[pipeline] built dataset: $DATASET_NPZ"

"$PYTHON_BIN" scripts/train_near_ready_xyyaw_predictor.py \
  --dataset_npz "$DATASET_NPZ" \
  --output_dir "$OUTPUT_DIR" \
  --epochs 12 \
  --batch_size 64 \
  --lr 2e-4

echo "[pipeline] training finished: $OUTPUT_DIR"

"$PYTHON_BIN" - <<'PY' "$OUTPUT_DIR" "$META_JSON" "$SUMMARY_JSON"
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
meta_json = Path(sys.argv[2])
summary_json = Path(sys.argv[3])

history = json.loads((output_dir / "train_history.json").read_text())
best = min(history, key=lambda row: float(row["loss"]))
meta = json.loads(meta_json.read_text()) if meta_json.exists() else {}

summary = {
    "output_dir": str(output_dir),
    "best_epoch": int(best["epoch"]),
    "best_val_loss": float(best["loss"]),
    "best_model_mae_xyyaw_norm": [float(x) for x in best["mae_xyyaw_norm"]],
    "runtime_baseline_mae_xyyaw_norm": [float(x) for x in best["runtime_baseline_mae_xyyaw_norm"]],
    "improvement_xy": float(best["runtime_baseline_mae_xyyaw_norm"][0] - best["mae_xyyaw_norm"][0]),
    "improvement_yaw": float(best["runtime_baseline_mae_xyyaw_norm"][1] - best["mae_xyyaw_norm"][1]),
    "ready_acc": float(best["ready_acc"]),
    "dataset_meta": meta,
}
summary_json.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[pipeline] summary written: $SUMMARY_JSON"
