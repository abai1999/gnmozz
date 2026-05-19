#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TOTAL_EPISODES="${TOTAL_EPISODES:-50}"
BASE_TAG="${TAG:-20260514g_50ep_multigpu}"
GPU_LIST_STR="${GPU_LIST:-2 3 4 5 6 7}"
MAX_STEPS="${MAX_STEPS:-360}"
VIDEO_EPISODES_PER_SHARD="${VIDEO_EPISODES_PER_SHARD:-2}"

read -r -a GPUS <<< "$GPU_LIST_STR"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 1
fi

BASE_OUTPUT_DIR="$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_state_expert_recovery_${BASE_TAG}"
mkdir -p "$BASE_OUTPUT_DIR/logs"

base_count=$(( TOTAL_EPISODES / ${#GPUS[@]} ))
remainder=$(( TOTAL_EPISODES % ${#GPUS[@]} ))
offset=0

echo "[multigpu] total=${TOTAL_EPISODES} gpus=${GPU_LIST_STR} base_tag=${BASE_TAG}"

for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  count="$base_count"
  if (( idx < remainder )); then
    count=$(( count + 1 ))
  fi
  if (( count <= 0 )); then
    continue
  fi

  shard_tag="${BASE_TAG}_shard${idx}_ep${offset}_n${count}"
  shard_output="$BASE_OUTPUT_DIR/$shard_tag"
  mkdir -p "$shard_output/logs"

  echo "[multigpu] gpu=${gpu} offset=${offset} count=${count} tag=${shard_tag}"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TAG="$shard_tag" \
    OUTPUT_DIR="$shard_output" \
    DEMO_FROM_EPISODE="$offset" \
    EPISODE_COUNT="$count" \
    MAX_STEPS="$MAX_STEPS" \
    VIDEO_EPISODES="$VIDEO_EPISODES_PER_SHARD" \
    bash "$ROOT/scripts/run_planner_state_expert_recovery_50ep.sh" \
    > "$shard_output/logs/stdout.log" 2>&1 &

  echo "$!" > "$shard_output/logs/pid.txt"
  offset=$(( offset + count ))
done

echo "[multigpu] launched shards under $BASE_OUTPUT_DIR"
echo "[multigpu] pids:"
find "$BASE_OUTPUT_DIR" -path '*/logs/pid.txt' -maxdepth 4 -type f -print -exec cat {} \;
