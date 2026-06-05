# C2C v2 v43/v44 Review and Next Action Plan

This document reviews the new v43/v44 readiness work and defines the next
concrete milestone after promoting `v42_expanded_v4pilot` as the active XY
baseline.

## Current Verdict

The v43/v44 line is directionally correct and should remain in the main C2C v2
path, but it is a readiness milestone, not a Z/Yaw control milestone.

- `v43_task_frame_z_readiness` predicts whether task-frame Z is observable,
  near alignment, and safe enough to contribute to strict handoff.
- `v44_task_frame_yaw_readiness` predicts whether yaw evidence is observable,
  ambiguous, or unobservable enough to contribute to strict handoff.
- Neither head outputs Z motion, yaw motion, close authority, or gripper
  commands.
- Planner close must still be allowed only by
  `alignment_ready_for_handoff=true`.

The active checkpoints are:

- `runtime_artifacts/coarse2contact_v2/checkpoints/v43_task_frame_z_readiness.pt`
- `runtime_artifacts/coarse2contact_v2/checkpoints/v44_task_frame_yaw_readiness.pt`

The active dataset is:

- `runtime_artifacts/coarse2contact_v2/datasets/task_frame_readiness_v43_v44.jsonl`

The current fixed runtime assumptions remain:

- Runtime environment: `conda run -n vla-adapter ...`
- Planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- Active XY baseline:
  `runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- `uses_privileged_runtime=false`

## Review Findings

### Fixed: compact dataset/runtime feature mismatch

The first v43/v44 pass had a subtle train/runtime mismatch. The dataset builder
stores many runtime-visible values inside `runtime_features`, while the
readiness feature extractor read several of those same values only from full
trace top-level or nested trace fields.

That meant compact training rows could zero out important non-privileged inputs
such as local depth proxy, local confidence, observability, fit residual,
inlier ratio, wrist depth validity, low-visibility flags, force norm, and
contact confirmation. Runtime traces could then populate these features
differently.

This is now fixed in
`prismatic/robot/coarse2contact_v2/task_frame_readiness.py`: feature extraction
falls back from full runtime trace fields to compact `runtime_features`, so the
same feature contract works for both training rows and runtime rows.

A regression test now checks that compact readiness dataset rows expose these
fields through `task_frame_readiness_feature_vector()`.

### Correct: strict handoff remains the only close source

The evaluator loads v43 and v44, applies them to `TaskFrameResidualEstimate`,
and passes the result to `evaluate_alignment_readiness()`. The close arbiter
then uses the resulting strict handoff state rather than legacy
`EstimatedBasinError.close_ready()`.

The review did not find v43/v44 opening new Z/Yaw control authority or direct
close authority.

### Still limited: v43/v44 do not prove successful handoff

The held-out metrics are promising:

- v43 Z readiness: precision `0.984`, recall `0.987`, threshold `0.80`
- v44 Yaw readiness: precision `1.000`, recall `1.000`, threshold `0.05`

These are root-held-out readiness metrics, not closed-loop success metrics.
They show that the readiness heads can classify the offline labels, but they do
not yet prove that runtime handoff will become true at the right moments or
that the first task stage succeeds end to end.

### Risk: yaw labels may be too easy

v44 reaches perfect held-out metrics very early. That can be a good sign, but
it may also mean the current yaw label is mostly keyed by alias-drift metadata
instead of richer visual/task-frame evidence.

This is acceptable for a conservative readiness gate, but not enough to justify
bounded yaw control. A future yaw controller must require a separate dyaw
estimator and stricter held-out visual validation.

## Current Project State

The current stack is:

1. v42 owns XY bounded correction.
2. v43 predicts task-frame Z readiness.
3. v44 predicts task-frame yaw readiness/ambiguity.
4. `alignment_ready_for_handoff` combines XY, Z, Yaw, observability, and frame
   consistency.
5. Planner close remains blocked whenever strict handoff is false.

The newest 120-step MP4 smoke reached C2C later than the earlier 60-step smoke,
but it still mostly demonstrates readiness gating and close blocking, not Z/Yaw
correction.

## Next Milestone

Make v43/v44 useful as a high-precision non-privileged handoff predicate on
actual runtime traces, then decide whether to open guarded Z micro-servo.

The concrete target is:

**On held-out random and hard-bucket runtime traces with v42 XY fixed, produce
high-precision `alignment_ready_for_handoff=true` only when offline XY/Z/Yaw
are all truly inside the first-stage grasp handoff basin, while keeping all
planner close requests blocked otherwise.**

## Action Plan

### Phase 1: Trace-level handoff audit

Run a dedicated v43/v44 handoff audit on:

- old4: `ep000/003/011/018`
- random5: `ep023/024/025/026/027`
- random10 generalization
- hard-bucket active rows
- low-visibility and occlusion slices

Required report fields:

- `trace_path`
- `episode_idx`
- `step`
- `runtime_xy_entry_ready`
- `task_frame_z_ready`
- `task_frame_yaw_ready`
- `alignment_ready_for_handoff`
- `alignment_handoff_block_reason`
- `planner_gripper_close_requested`
- `planner_gripper_close_blocked`
- offline `xy_error/z_abs/yaw_abs` sidecar values
- `uses_privileged_runtime=false`

Primary metrics:

- handoff precision on offline XY/Z/Yaw basin labels
- handoff false-positive rate
- close-block precision when planner requests close
- per-axis block reason distribution
- per-slice worst-case handoff precision

### Phase 2: Longer MP4 smoke with trace overlays

Run MP4 smoke only after the trace audit is clean enough to inspect visually.
Use at least `120` steps, and extend to `150/180` when C2C activates late.

MP4s must include wrist view. The paired trace report must list the exact steps
where:

- C2C gate starts
- XY correction starts
- planner close is requested
- close is blocked
- `alignment_ready_for_handoff` becomes true, if it ever does

### Phase 3: Decide Z control scope

Only if v43 shows high precision on trace-level Z readiness should v46 open
guarded Z micro-servo.

Initial Z control constraints:

- small task-frame approach-axis step
- low speed
- force guarded
- allowed only when v42 XY is ready and v43 says Z is observable
- cannot set `alignment_ready_for_handoff` by itself
- any Z-control failure is logged as failure-tail data

Yaw control remains out of scope until yaw readiness is validated on richer
visual slices and a separate dyaw estimator is trained.

## Non-Goals

- Do not loosen `alignment_ready_for_handoff` thresholds to make MP4s close.
- Do not use legacy `close_ready` as runtime close permission.
- Do not open yaw servo from v44 readiness alone.
- Do not use privileged pose, teacher residual, or RLBench mask at runtime.
- Do not retune v42 XY while auditing v43/v44 handoff unless a trace proves XY
  is the active blocker.

## Immediate Command Targets

Recommended next commands:

```bash
conda run -n vla-adapter pytest tests/test_coarse2contact_v2.py
```

Then run the next evaluator smoke with:

- `--max_steps 150` or `--max_steps 180`
- v42 XY checkpoint
- v43 Z readiness checkpoint
- v44 Yaw readiness checkpoint
- strict close blocking enabled
- wrist/front MP4 output enabled

The next acceptance decision should be based on trace-level handoff precision,
not MP4 appearance alone.
