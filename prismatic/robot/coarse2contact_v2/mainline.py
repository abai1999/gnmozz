"""Mainline pointers for Coarse2Contact v2 recovery/research artifacts."""

from __future__ import annotations

from pathlib import Path


# Stable single-step recovery candidate kept as a diagnostic baseline.
RECOVERY_MAINLINE_NAME = "grasp_recovery_head_v11_runtime_failure"
RECOVERY_MAINLINE_CHECKPOINT = Path(
    "runtime_artifacts/coarse2contact_v2/checkpoints/grasp_recovery_head_v11_runtime_failure/best.pt"
)
RECOVERY_MAINLINE_DATASET = Path(
    "runtime_artifacts/coarse2contact_v2/datasets_runtime_failure/grasp_recovery_runtime_failure_dataset_v1.jsonl"
)
RECOVERY_MAINLINE_REPORT = Path(
    "runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_shadow_v11_runtime_failure.json"
)

BASIN_RECOVERY_DATASET = Path(
    "runtime_artifacts/coarse2contact_v2/datasets_basin_recovery/basin_recovery_dataset_v1.jsonl"
)
BASIN_RECOVERY_CLOSED_LOOP_REPORT = Path(
    "runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_closed_loop_30ep_basin.json"
)

# New source of truth for future recovery work: real runtime failure traces.
RUNTIME_FAILURE_TRACE_ROOT = Path("runtime_artifacts/coarse2contact_v2/runtime_failure_traces")
RUNTIME_FAILURE_DATASET_ROOT = Path("runtime_artifacts/coarse2contact_v2/datasets_runtime_failure")
