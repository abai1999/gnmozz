#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"

export RUN_TAG="${RUN_TAG:-20260419a_smoke_20260428k}"
export NUM_EPISODES="${NUM_EPISODES:-2}"
export MAX_STEPS="${MAX_STEPS:-220}"
export GPU_IDS="${GPU_IDS:-0}"

# Smoke policy: keep the planner frozen, but delay hard alignment takeover so
# we can inspect whether yaw/z/close improve before the planner is handed off.
export ALIGNMENT_TAKEOVER_UNTIL_CLOSE_READY="${ALIGNMENT_TAKEOVER_UNTIL_CLOSE_READY:-0}"
export TEACHER_REQUIRE_ALIGNMENT_READY_FOR_MOTION_GATE="${TEACHER_REQUIRE_ALIGNMENT_READY_FOR_MOTION_GATE:-0}"

# Canonical close gates for smoke:
# keep the teacher frozen at the stable tight-but-working settings.
export TEACHER_CLOSE_XY_THRESHOLD="${TEACHER_CLOSE_XY_THRESHOLD:-0.006}"
export TEACHER_CLOSE_ABS_Z_THRESHOLD="${TEACHER_CLOSE_ABS_Z_THRESHOLD:-0.005}"
export TEACHER_CLOSE_YAW_THRESHOLD="${TEACHER_CLOSE_YAW_THRESHOLD:-0.12}"

# Keep canonical motion gate defaults; the smoke should test whether alignment
# itself is healthy, not silently disable it.
export TEACHER_MOTION_ENTRY_XY_THRESHOLD="${TEACHER_MOTION_ENTRY_XY_THRESHOLD:-0.040}"
export TEACHER_MOTION_ENTRY_ABS_Z_THRESHOLD="${TEACHER_MOTION_ENTRY_ABS_Z_THRESHOLD:-0.120}"
export TEACHER_HANDOFF_REVOKE_XY_THRESHOLD="${TEACHER_HANDOFF_REVOKE_XY_THRESHOLD:-0.012}"
export TEACHER_HANDOFF_REVOKE_ABS_Z_THRESHOLD="${TEACHER_HANDOFF_REVOKE_ABS_Z_THRESHOLD:-0.025}"

exec bash "$ROOT/scripts/run_phase1_yaw_teacher_pipeline_20260419a_parallel_collect.sh"
