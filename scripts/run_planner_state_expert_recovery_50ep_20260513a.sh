#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_state_expert_recovery_20260513a_50ep}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/guoning/CoppeliaSim}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${LD_PRELOAD:-/home/guoning/my_conda_envs/vla-adapter/lib/libstdc++.so.6}"

mkdir -p "$OUT_DIR"

RUN_PREFIX=()
if command -v xvfb-run >/dev/null 2>&1; then
  RUN_PREFIX=(xvfb-run -a)
fi

"${RUN_PREFIX[@]}" "$PYTHON" "$ROOT/scripts/collect_planner_state_expert_recovery.py" \
  --checkpoint_dir "${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt}" \
  --task_name insert_onto_square_peg \
  --data_root "${DATA_ROOT:-data/rlbench_data}" \
  --num_episodes 50 \
  --demo_from_episode "${DEMO_FROM_EPISODE:-0}" \
  --demo_max_attempts "${DEMO_MAX_ATTEMPTS:-10}" \
  --max_rollout_steps "${MAX_ROLLOUT_STEPS:-360}" \
  --no_allow_broad_near_takeover \
  --video_episodes "${VIDEO_EPISODES:-10}" \
  --record_video \
  --output_dir "$OUT_DIR" \
  --output_npz "$OUT_DIR/alignment_tc_planner_state_expert_recovery_raw_20260513a_50ep.npz" \
  --report_json "$OUT_DIR/report_20260513a_50ep.json" \
  --takeover_trace_jsonl "$OUT_DIR/takeover_trace_20260513a_50ep.jsonl"
