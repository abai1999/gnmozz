#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

EPISODE_INDICES="${EPISODE_INDICES:-18,34,45,12,14,46,23,11}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/b2_candidate_shadow_v14d_parallel_20260425e}"
MERGED_RUN_DIR="${MERGED_RUN_DIR:-$OUTPUT_ROOT/merged_b2_v14d_runtimeyaw_shadow_nearish_keepbaseline}"
B2_SHADOW_YAW_PROBES="${B2_SHADOW_YAW_PROBES:-0.06,0.12}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-3}"
GPU_IDS="${GPU_IDS:-}"

if [[ -z "$GPU_IDS" ]]; then
  mapfile -t GPU_ARR < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v max_jobs="$MAX_PARALLEL_JOBS" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4);
        free=$3-$2;
        if ($2 <= 1024 && $4 <= 20 && free >= 16000) {
          print $1;
          count++;
          if (count >= max_jobs) exit;
        }
      }')
else
  IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
fi

if (( ${#GPU_ARR[@]} == 0 )); then
  echo "[b2-v14d-parallel] no idle GPU found; falling back to auto runner default" >&2
  GPU_ARR=("")
fi

IFS=',' read -r -a EP_ARR <<< "$EPISODE_INDICES"
mkdir -p "$OUTPUT_ROOT"
rm -rf "$MERGED_RUN_DIR"
mkdir -p "$MERGED_RUN_DIR/gripper_traces"

PIDS=()
SHARD_DIRS=()
ACTIVE_PIDS=()
ACTIVE_LIMIT="${#GPU_ARR[@]}"
fail=0
for i in "${!EP_ARR[@]}"; do
  ep="${EP_ARR[$i]}"
  gpu="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  shard_name="shard$(printf '%02d' "$i")_ep${ep}"
  shard_root="$OUTPUT_ROOT/$shard_name"
  SHARD_DIRS+=("$shard_root")
  echo "[b2-v14d-parallel] launch $shard_name gpu=${gpu:-auto} episode=$ep"
  (
    export EPISODE_INDICES="$ep"
    export NUM_EPISODES=1
    export OUTPUT_ROOT="$shard_root"
    export NAME_SUFFIX="b2_v14d_runtimeyaw_shadow_nearish_keepbaseline_${shard_name}"
    export B2_SHADOW_YAW_PROBES="$B2_SHADOW_YAW_PROBES"
    export B2_SHADOW_GATE_MODE="nearish_only"
    if [[ -n "$gpu" ]]; then
      export CUDA_VISIBLE_DEVICES="$gpu"
    else
      unset CUDA_VISIBLE_DEVICES || true
    fi
    "$ROOT/scripts/run_b2_candidate_shadow_v14d_20260425e.sh"
  ) >"$shard_root.log" 2>&1 &
  pid="$!"
  PIDS+=("$pid")
  ACTIVE_PIDS+=("$pid")
  if (( ${#ACTIVE_PIDS[@]} >= ACTIVE_LIMIT )); then
    first_pid="${ACTIVE_PIDS[0]}"
    if ! wait "$first_pid"; then
      fail=1
    fi
    ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
  fi
done

echo "[b2-v14d-parallel] launched pids: ${PIDS[*]}"
for pid in "${ACTIVE_PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if (( fail != 0 )); then
  echo "[b2-v14d-parallel] one or more shards failed" >&2
  exit 1
fi

for shard_root in "${SHARD_DIRS[@]}"; do
  while IFS= read -r trace; do
    cp "$trace" "$MERGED_RUN_DIR/gripper_traces/"
  done < <(find "$shard_root" -path '*/gripper_traces/*_gripper_trace.jsonl' -type f | sort)
done

"$PYTHON_BIN" "$ROOT/scripts/analyze_b2_candidate_shadow_trace.py" \
  --trace_dir "$MERGED_RUN_DIR" \
  --output_json "$MERGED_RUN_DIR/b2_shadow_trace_analysis.json" \
  --focus_output_json "$MERGED_RUN_DIR/b2_shadow_focus_episode_diagnostics.json" \
  --gate_output_json "$MERGED_RUN_DIR/b2_shadow_gate_decision.json"

"$PYTHON_BIN" "$ROOT/scripts/visualize_b2_candidate_shadow_trace.py" \
  --trace_dir "$MERGED_RUN_DIR" \
  --output_dir "$MERGED_RUN_DIR/visualizations" \
  --focus_episodes "18,34,45"

echo "[b2-v14d-parallel] merged traces, reports, and visualizations at $MERGED_RUN_DIR"
