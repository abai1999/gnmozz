# C2C v2 Z/Yaw Readiness Plan

This document is the next concrete action plan after promoting
`v42_expanded_v4pilot` to the active XY baseline. The goal is to turn the
current XY-only precision pullback into a strict, non-privileged
`alignment_ready_for_handoff` path without loosening close authority.
The current close authority policy is planner-owned close plus C2C open-only
safety.

## Clear Objective

Build `v43_task_frame_z_readiness` and `v44_task_frame_yaw_readiness` so C2C can
decide, from non-privileged runtime observations, whether the gripper is ready
for planner close handoff after v42 XY alignment.

The concrete target is:

**Make `alignment_ready_for_handoff` become true only when v42 XY, task-frame Z
readiness, task-frame yaw readiness, observability, and frame consistency are
all satisfied on held-out failure tails.**

This is a readiness milestone first, not a new control milestone. C2C still
does not own close. `close_ready` remains legacy/offline diagnostic only.

## Close Ownership Contract

Runtime close authority is now centralized in
`planner_gripper_authority_decision(...)`.

- Planner close intent is the only source that can close the gripper.
- `alignment_ready_for_handoff=true` is required for the first close handoff.
- After a valid strict handoff, a latched planner close may remain closed in
  later contact/verify stages, but the source is still planner intent.
- C2C can request open for safety during reacquire, recovery, invalid action,
  unstable contact, or failed contact monitoring.
- C2C close recommendations are trace/debug signals and are ignored unless the
  planner also requests close and strict handoff policy allows it.

Required trace fields for every runtime smoke:

- `planner_gripper_close_requested`
- `planner_gripper_close_blocked`
- `planner_gripper_handoff_allowed`
- `planner_gripper_strict_handoff_ready`
- `planner_gripper_handoff_latched`
- `c2c_gripper_open_safety_requested`
- `c2c_gripper_close_recommendation_ignored`
- `gripper_authority_source`

## Implementation Status

The first implementation pass is now in place:

- dataset:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_readiness_v43_v44.jsonl`
- Z readiness checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/v43_task_frame_z_readiness.pt`
- Yaw readiness checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/v44_task_frame_yaw_readiness.pt`
- held-out validation:
  - Z precision `0.984`, recall `0.987` at threshold `0.80`
  - Yaw precision `1.000`, recall `1.000` at threshold `0.05`
- smoke:
  short `xvfb-run` sanity runs on `insert_onto_square_peg` exercised the
  readiness heads and kept the strict handoff gate in place

The readiness feature extractor was also tightened so compact dataset rows and
full runtime traces resolve local geometry, depth visibility, force, and
contact fields through the same feature names. This avoids training on zeroed
runtime-visible fields while evaluating on populated trace fields.

The remaining work is broader trace-level A/B and longer smoke coverage, not
basic wiring.

## Fixed Defaults

- Runtime environment: `conda run -n vla-adapter ...`
- Planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- Active XY baseline:
  `runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- Runtime contract:
  `alignment_ready_for_handoff` remains the only planner gripper handoff truth
  source.
- Runtime evidence boundary:
  privileged pose / mask / teacher residuals are allowed only for offline
  labels, relabeling, audit, and eval sidecars.

## Current Diagnosis

v42 moved the project forward: random held-out and hard-bucket XY worsen /
reverse improved without reopening yaw or close. The main remaining blocker is
now strict handoff readiness:

- XY is the active baseline and should be guarded, not constantly re-tuned.
- Z readiness is still not learned as a non-privileged task-frame predicate.
- Yaw readiness is still not learned as a non-privileged observability /
  ambiguity predicate.
- Strict handoff correctly blocks planner close when Z/Yaw are not ready.

The next work should therefore avoid two traps:

- Do not loosen handoff thresholds to make MP4s close.
- Do not jump directly to Z/Yaw control before readiness is validated.

## Phase v43: Task-Frame Z Readiness

### Goal

Train and validate a non-privileged `z_readiness` model for the grasp alignment
stage. It predicts whether the task-frame approach-axis relation is close,
observable, and safe enough to contribute to handoff.

### Runtime Inputs

- Wrist RGBD crop and depth-validity patch.
- Local depth profile around the gripper/ring region.
- Proprio and gripper aperture/state.
- Planner prior delta.
- v42 XY residual estimate and confidence.
- Recent K-step motion and depth history.
- Force/contact features as auxiliary evidence, not privileged labels.

### Offline Labels

Offline labels may use privileged task-frame `dz`, contact/reward/force traces,
and post-step contraction only in training/eval sidecars.

Required sidecar fields:

- `z_label_source=offline_privileged_task_frame_dz`
- `uses_privileged_runtime=false`
- `trace_path`
- `source_eval_root`
- `episode_idx`
- `step_idx`
- `alignment_xy_ready`
- `z_ready_label`
- `z_observable_label`

### Outputs

- `z_observable`
- `z_near_alignment`
- `z_contact_or_depth_ready`
- `z_confidence`
- `z_abstain_reason`

### Acceptance Gate

v43 may feed strict handoff readiness only if held-out validation shows:

- high precision on `z_ready`, because false-positive Z readiness can cause
  early close
- useful recall on v42-XY-ready rows
- no privileged runtime inputs
- clear per-slice metrics for old4, random5, random10, random holdout,
  hard-bucket, low-visibility, and ep25/26
- false-positive examples are traceable by `trace_path`

v43 must not open Z control. It only sets readiness.

## Phase v44: Task-Frame Yaw Readiness

### Goal

Train and validate a yaw observability/readiness model. The first useful yaw
model is not a dyaw controller; it is an ambiguity detector that says whether
runtime yaw evidence is trustworthy enough for handoff.

### Runtime Inputs

- Wrist RGBD crop and depth-validity patch.
- Ring/frame geometry features derived from non-privileged visual evidence.
- v42 XY estimate and confidence.
- v43 Z readiness outputs.
- Recent K-step crop/motion history.
- Planner prior and gripper proprio.

### Offline Labels

Labels may use privileged task-frame yaw residual and alias/drift annotations
only offline.

Required sidecar fields:

- `yaw_label_source=offline_privileged_task_frame_yaw`
- `uses_privileged_runtime=false`
- `alias_drift_decision`
- `yaw_ready_label`
- `yaw_observable_label`
- `yaw_ambiguous_label`
- `trace_path`

### Outputs

- `yaw_observable`
- `yaw_ambiguous`
- `yaw_unobservable`
- `yaw_confidence`
- `yaw_abstain_reason`

### Acceptance Gate

v44 may contribute to handoff only if:

- held-out yaw-ready precision is high
- ambiguous/alias-drift rows are mostly blocked, not forced ready
- frame-drift rows are negative or abstain
- PCA/image-axis yaw is not used as direct control
- all metrics are split by alias decision, observability bucket, hard bucket,
  random holdout, and old4/random5

v44 must not open yaw control.

## Phase v45: Strict Handoff Integration

### Goal

Wire v42 XY + v43 Z readiness + v44 Yaw readiness into the existing
`TaskFrameResidualEstimate` / `AlignmentTakeoverSession` path and verify that
planner close is allowed only when strict handoff is true.

This phase also keeps the task contract aligned with runtime semantics:
`RING_GRASP_CONTACT` is a contact/force monitor stage with
`gripper_mode: planner_after_handoff`, not a C2C-owned direct-close stage.

### Required Trace Fields

Every smoke and validation trace should include:

- `xy_baseline=v42_expanded_v4pilot`
- `z_readiness_source`
- `z_observable`
- `z_near_alignment`
- `z_confidence`
- `z_abstain_reason`
- `yaw_readiness_source`
- `yaw_observable`
- `yaw_ambiguous`
- `yaw_confidence`
- `yaw_abstain_reason`
- `alignment_ready_for_handoff`
- `alignment_handoff_block_reason`
- `planner_gripper_close_blocked`
- `planner_gripper_close_requested`
- `planner_gripper_handoff_allowed`
- `c2c_gripper_open_safety_requested`
- `c2c_gripper_close_recommendation_ignored`
- `gripper_authority_source`
- `uses_privileged_runtime=false`

### Acceptance Gate

The handoff path is acceptable only if:

- `planner_gripper_close_blocked=true` whenever
  `alignment_ready_for_handoff=false` and planner requests close
- handoff true rows have high offline precision on XY, Z, and Yaw
- early-close false positives decrease, not just move between labels
- no fallback path uses legacy `close_ready` as control permission
- MP4 smoke includes wrist view, but final acceptance is based on trace-level
  held-out A/B

## Future Control Phases

Only after readiness is validated:

- v45 can also carry the first guarded Z/Yaw residual-readiness candidate if
  used strictly as bounded micro-servo plus trace, never as close authority.
- v46 can test broader guarded Z micro-servo:
  small step, low speed, force guarded, only when v42 XY is ready and v43 says
  Z is observable.
- v47 can test bounded yaw servo:
  only after yaw readiness and dyaw estimator pass held-out validation.

These are deliberately out of scope for v43/v44. Readiness comes first.

## Immediate Work Items

1. Build the v43 Z readiness dataset from v42 runtime traces and runtime
   observations.
2. Train a conservative Z readiness classifier/head.
3. Run held-out Z readiness validation on random10, random holdout, old4,
   random5, hard-bucket, low-visibility, and ep25/26 slices.
4. Build the v44 yaw readiness/ambiguity dataset with alias-drift labels.
5. Train and validate yaw observability/readiness without exposing yaw control.
6. Integrate both readiness heads into trace-only strict handoff evaluation.
7. Only then run MP4 smoke for visual inspection.

## One-Line Target

The next milestone is:

**With v42 fixed as XY baseline, make strict `alignment_ready_for_handoff`
become a high-precision non-privileged predicate by adding v43 Z readiness and
v44 Yaw readiness, while keeping planner close blocked whenever either axis is
not ready.**
