#!/usr/bin/env bash
set -euo pipefail

LOG=/home/guoning/code/VLA/runtime_artifacts/residual_data/collect_logs/collect_oracle8_20260421c.monitor.log
OUTDIR=/home/guoning/code/VLA/eval_logs/insert_onto_square_peg/support_rows_resync_20260421c/insert_vo40k_oracle_target_upper_bound_seed3407_full_oracle8
NPZ=/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_support_resync_oracle_full8_20260421c/support_states.npz

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

while true; do
  ts=$(date "+%F %T")
  mp4_count=$(find "$OUTDIR" -type f -name "*.mp4" 2>/dev/null | wc -l)
  ep_dir_count=$(find "$OUTDIR" -maxdepth 1 -type d -name "episode_*" 2>/dev/null | wc -l)
  if [[ -f "$NPZ" ]]; then
    npz_state="present($(stat -c%s "$NPZ" 2>/dev/null || echo 0)B)"
  else
    npz_state="missing"
  fi
  echo "[$ts] oracle8 progress: episode_dirs=$ep_dir_count mp4=$mp4_count npz=$npz_state" >> "$LOG"
  sleep 30
  if ! pgrep -f "evaluate_rlbench_modes.py.*seed3407_full_oracle8" >/dev/null; then
    echo "[$(date "+%F %T")] oracle8 process finished" >> "$LOG"
    break
  fi
done
