#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/b1_group_selector_shadow_20260422r}"
export NAME_SUFFIX="${NAME_SUFFIX:-run12_b1_shadow_closeonly_applygate}"
export B1_APPLY_GATE_CKPT="${B1_APPLY_GATE_CKPT:-$ROOT/runtime_artifacts/stage_refiner/b1_apply_gate_20260422q/train_val_ep11_17_safethresh/b1_apply_gate_best.pt}"

exec "$ROOT/scripts/run_b1_group_selector_shadow_20260422m.sh" \
  --b1_group_shadow_gate_mode close_only \
  --student_b1_apply_gate_shadow_ckpt "$B1_APPLY_GATE_CKPT" \
  "$@"
