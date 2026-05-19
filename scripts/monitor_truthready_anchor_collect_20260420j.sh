#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
COLLECT_PID="${1:-452099}"
OUT_LOG="${2:-/home/guoning/code/VLA/runtime_artifacts/logs/truthready_anchor_collect_20260420j.monitor.log}"
TRACE_ROOT="/home/guoning/code/VLA/eval_logs/insert_onto_square_peg/truthready_anchor_collect_20260420j/insert_vo40k_oracle_executed_pregrasp_collect_seed3407_truthready_anchor_collect/gripper_traces"
TARGET_NPZ="/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthready_anchor_support_20260420j/support_states.npz"

mkdir -p "$(dirname "$OUT_LOG")"

echo "[monitor] starting pid=$COLLECT_PID trace_root=$TRACE_ROOT" >> "$OUT_LOG"

while true; do
  ts="$(date '+%F %T')"
  if ps -p "$COLLECT_PID" > /dev/null 2>&1; then
    proc_line="$(ps -p "$COLLECT_PID" -o pid=,etime=,%cpu=,%mem=,cmd= --no-headers || true)"
    trace_count="$(find "$TRACE_ROOT" -maxdepth 1 -name 'ep*_gripper_trace.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
    last_trace="$(find "$TRACE_ROOT" -maxdepth 1 -name 'ep*_gripper_trace.jsonl' 2>/dev/null | sort | tail -n 1)"
    last_step="NA"
    if [[ -n "${last_trace:-}" && -f "$last_trace" ]]; then
      last_step="$(tail -n 1 "$last_trace" | /home/guoning/my_conda_envs/vla-adapter/bin/python -c 'import sys,json; line=sys.stdin.read().strip(); print(json.loads(line).get("step","NA") if line else "NA")' 2>/dev/null || echo NA)"
    fi
    echo "[$ts] RUNNING traces=$trace_count last_trace=${last_trace:-none} last_step=$last_step proc=[$proc_line]" >> "$OUT_LOG"
  else
    npz_state="missing"
    if [[ -f "$TARGET_NPZ" ]]; then
      npz_state="present"
    fi
    echo "[$ts] EXITED npz=$npz_state pid=$COLLECT_PID" >> "$OUT_LOG"
    exit 0
  fi
  sleep 60
done
