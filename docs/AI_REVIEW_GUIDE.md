# C2C v2 External Review Guide

This file is meant for external AI reviewers that only have git access to this
repository.

## Review Goal

Review whether Coarse2Contact v2 is converging on its intended role:

- a non-privileged precision layer on top of a frozen VLA planner
- a system that can recover true failure tails in closed loop
- a design that separates residual estimation, observability gating, and
  takeover tiering

Do not treat planner-only success windows, replay-only traces, or shadow scores
as proof of success.

The current strongest intended claim is a contract-calibrated precision
takeover framework, not solved online failure-tail recovery.

## Fixed Review Anchors

- Branch: `codex/c2c-v2-status-publish`
- Latest relevant commit: current branch tip on `codex/c2c-v2-status-publish`
  (see `git log -n 1`)
- Fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- Fixed runtime environment: `conda run -n vla-adapter ...`
- Current focus stage: `RING_GRASP_ALIGN`
- Runtime apply remains disabled; evaluation probes are privileged and offline

## The Three-Layer Design To Check

### 1. Residual Estimator

Question: does the code estimate the true jaw-local residual, rather than a
heuristic visual proxy?

Look at:

- `prismatic/robot/coarse2contact_v2/frame_yaw_estimator.py`
- `prismatic/robot/coarse2contact_v2/takeover_contract.py`
- `scripts/relabel_c2c_v2_privileged_basin_frames.py`
- `scripts/audit_c2c_v2_frame_contract_relabel.py`

What to verify:

- `dx/dy/dz/dyaw` are frame-consistent and jaw-local
- privileged labels are used for relabel/audit only
- proxy PCA/image-axis yaw is not promoted to trusted control

### 2. Observability Gate

Question: does the system correctly decide when yaw or other axes are
trustworthy enough to use?

Look at:

- `prismatic/robot/coarse2contact_v2/basin_state.py`
- `scripts/classify_c2c_v2_yaw_alias_vs_drift.py`
- `scripts/run_c2c_v2_yaw_alias_drift_baseline.py`
- `scripts/run_c2c_v2_yaw_alias_drift_two_stage_baseline.py`
- `scripts/audit_c2c_v2_yaw_threshold_sweep.py`
- `scripts/diagnose_c2c_v2_yaw_frame_alignment.py`

What to verify:

- stable alias and frame drift are separated
- abstain is preserved where observability is weak
- yaw calibration is not silently widened into a control shortcut

### 3. Takeover Tier

Question: does the system expose a clean progression from coarse pullback to
near-basin, micro-entry, and close-ready?

Look at:

- `prismatic/robot/coarse2contact_v2/grasp_probe_shell.py`
- `prismatic/robot/coarse2contact_v2/supervisor.py`
- `scripts/build_c2c_v2_grasp_failure_tail_candidates.py`
- `scripts/build_c2c_v2_grasp_failure_tail_hard_bucket_gap_report.py`
- `scripts/build_c2c_v2_grasp_failure_tail_hard_bucket_focus_manifest.py`
- `scripts/build_c2c_v2_failure_tail_support_manifest.py`
- `scripts/build_c2c_v2_failure_tail_hard_manifest.py`
- `scripts/audit_c2c_v2_grasp_failure_tail_intervention.py`
- `scripts/compare_c2c_v2_queue_flush_ablation.py`

What to verify:

- tiering is explicit, not a disguised gate relaxation
- `pullback_ready`, `micro_entry_ready`, and `close_ready` are audited as
  separate readiness levels
- queue/window ablations are interpreted separately from frame semantics
- hard-bucket support is being widened without breaking the control contract

## Suggested Review Order

1. Read `README.md` and `docs/C2C_V2_PROJECT_STATUS.md`.
2. Inspect the latest commit range with:

```bash
git log --oneline --decorate -n 12
git show --stat --summary 6576053
```

3. Read the contract and runtime files first:

- `prismatic/robot/coarse2contact_v2/takeover_contract.py`
- `prismatic/robot/coarse2contact_v2/basin_state.py`
- `prismatic/robot/coarse2contact_v2/supervisor.py`

4. Then inspect the relabel and audit scripts.
5. Finally, inspect `tests/test_coarse2contact_v2.py` for the intended
   invariants.

## What Not To Count As Proof

- A single successful planner episode
- Planner-only trajectories with no hard-tail failure
- Replay-only or offline-only scores
- MP4s without matching trace interpretation
- Local `runtime_artifacts/` outputs that are not committed to git

## Current Review Question

The main question for the next reviewer is simple:

Can the current code base show a clean, auditable path from
`failure-tail candidate -> residual estimate -> observability decision ->
takeover tier -> contractive intervention`, without relying on privileged data
at runtime?

If the answer is no, the next fix should be semantic and structural, not just a
gain tweak.

## Latest Validation Snapshot

Recent hard-bucket validation tightened the current interpretation:

- `small_xy_large_yaw` is no longer treated as a sign flip. The focused
  direction diagnostic points to a `step_too_small_candidate` pattern instead.
- `large_xy_large_yaw` now shows active rows after entry-focused support was
  widened, but it still does not show near-grasp entry.
- `alias_drift_decision` now propagates through candidate, trace, and support
  plumbing, so reviewers should inspect whether any remaining `unknown` rows are
  a true observability gap or just incomplete support coverage.
