#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="$ROOT/scripts/run_phaseA_overlap_applied_smoke_20260423i.sh"

PART="${PART:-A}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/eval_logs/insert_onto_square_peg/phaseA_overlap_applied_smoke_20260423m}"

case "$PART" in
  A)
    NAME_SUFFIX="${NAME_SUFFIX:-applied_readyrich12_retrace_partA}"
    EPISODE_INDICES="${EPISODE_INDICES:-44,46,11,6}"
    NUM_EPISODES="${NUM_EPISODES:-4}"
    ;;
  B)
    NAME_SUFFIX="${NAME_SUFFIX:-applied_readyrich12_retrace_partB}"
    EPISODE_INDICES="${EPISODE_INDICES:-29,33,21,14}"
    NUM_EPISODES="${NUM_EPISODES:-4}"
    ;;
  C)
    NAME_SUFFIX="${NAME_SUFFIX:-applied_readyrich12_retrace_partC}"
    EPISODE_INDICES="${EPISODE_INDICES:-35,36,12,43}"
    NUM_EPISODES="${NUM_EPISODES:-4}"
    ;;
  *)
    echo "Unknown PART=$PART, expected A/B/C" >&2
    exit 2
    ;;
esac

exec env \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  NAME_SUFFIX="$NAME_SUFFIX" \
  EPISODE_INDICES="$EPISODE_INDICES" \
  NUM_EPISODES="$NUM_EPISODES" \
  "$BASE_SCRIPT" "$@"
