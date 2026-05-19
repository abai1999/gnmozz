#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/b1_group_selector_shadow_20260422p}"
export NAME_SUFFIX="${NAME_SUFFIX:-run12_b1_shadow_closeonly}"

exec "$ROOT/scripts/run_b1_group_selector_shadow_20260422m.sh" \
  --b1_group_shadow_gate_mode close_only \
  "$@"
