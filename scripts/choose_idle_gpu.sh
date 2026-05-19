#!/usr/bin/env bash
set -euo pipefail

MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-1024}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-20}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-16000}"
GPU_CANDIDATES="${GPU_CANDIDATES:-}"

candidates_csv=",${GPU_CANDIDATES// /,},"

while IFS=',' read -r idx mem_used mem_total util; do
  idx="${idx// /}"
  mem_used="${mem_used// /}"
  mem_total="${mem_total// /}"
  util="${util// /}"
  if [[ -n "${GPU_CANDIDATES}" && "${candidates_csv}" != *",${idx},"* ]]; then
    continue
  fi
  free_mem=$((mem_total - mem_used))
  if (( mem_used <= MAX_USED_MEM_MB && util <= MAX_UTIL_PERCENT && free_mem >= MIN_FREE_MEM_MB )); then
    printf '%s\n' "$idx"
    exit 0
  fi
done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

exit 1
