# Coarse2Contact v2 Project Status

This document is written for future reviewers, agents, and collaborators who
need to understand the current C2C v2 state without replaying the full
conversation. It is intentionally candid: the scaffold is useful, but the
current local skill layer does not yet solve the real failure-tail recovery
problem.

For a shorter external-review entry point, see
[`docs/AI_REVIEW_GUIDE.md`](AI_REVIEW_GUIDE.md).

## Goal

Coarse2Contact v2 is intended to become a task-general precision local skill
layer on top of a frozen VLA planner.

The desired division of labor is:

- The VLA planner performs coarse motion toward task subgoals.
- C2C activates only after the planner reaches a configured precision region.
- C2C takes ownership of high-precision subskills such as small-object grasp
  alignment, aperture-to-spoke alignment, guarded slide, and recovery.
- C2C must be non-privileged at runtime: no RLBench object handles, teacher
  targets, success poses, or mask-driven control.
- Privileged masks/poses may be used only offline for audit, relabeling,
  calibration, and evaluation.

The first proof target is `insert_onto_square_peg`, especially the ring grasp
failure tail. The real task sequence is:

1. Coarse approach to the square ring.
2. Precision align gripper jaw frame to the ring grasp frame.
3. Close and verify stable grasp.
4. Transfer toward the red spoke.
5. Precision align held ring aperture to the target spoke axis.
6. Guarded slide onto the spoke.
7. Recover from jam, misgrasp, slip, invalid actions, or loss of observability.

The research goal is not a prettier shadow score. The goal is to show that C2C
can, in closed loop and with interpretable traces, pull true planner failure
tails back into a near-grasp or near-insert basin and improve task success.

## Current Implementation Map

Core C2C v2 code:

- `prismatic/robot/coarse2contact_v2/specs.py`
  - YAML task/skill/entity/stage contract loaders.
- `configs/coarse2contact/tasks/insert_onto_square_peg.yaml`
  - Current task plugin for the square-ring-on-spoke task.
- `prismatic/robot/coarse2contact_v2/supervisor.py`
  - Owner-by-stage runtime supervisor and precision gate.
- `prismatic/robot/coarse2contact_v2/localizers.py`
  - Heuristic ring/spoke RGBD proxy localizers.
- `prismatic/robot/coarse2contact_v2/basin_state.py`
  - Estimated basin error layer and calibration-aware axis validity.
- `prismatic/robot/coarse2contact_v2/basin_recovery.py`
  - Basin recovery mode supervisor: reacquire, visual pullback, micro-servo,
    verify, abstain.
- `prismatic/robot/coarse2contact_v2/controllers.py`
  - Grasp, slide, and recovery primitives.
- `prismatic/robot/coarse2contact_v2/learned_localizer.py`
  - Learned depth/localizer model stubs and adapters.
- `prismatic/robot/coarse2contact_v2/learned_force.py`
  - Learned force classifier adapter.
- `scripts/evaluate_c2c_v2_rlbench.py`
  - Standalone v2 RLBench evaluator with video and trace output.

Audit, training, and dataset scripts:

- `scripts/build_c2c_v2_depth_localizer_dataset.py`
- `scripts/build_c2c_v2_force_contact_dataset.py`
- `scripts/train_c2c_v2_ring_frame_localizer.py`
- `scripts/train_c2c_v2_grasp_skill_head.py`
- `scripts/train_c2c_v2_depth_localizer.py`
- `scripts/train_c2c_v2_force_classifier.py`
- `scripts/build_c2c_v2_grasp_recovery_runtime_failure_dataset.py`
- `scripts/train_c2c_v2_basin_recovery.py`
- `scripts/eval_c2c_v2_grasp_recovery_closed_loop.py`
- `scripts/audit_c2c_v2_basin_state_calibration.py`
- `scripts/audit_depthgate_grasp_recovery_diagnostics.py`

Tests:

- `tests/test_coarse2contact_v2.py`

## Runtime Contract

The v2 runtime now routes control decisions through an explicit
`EstimatedBasinError` rather than directly using raw `LocalGeometryError`.

`LocalGeometryError` is treated as raw visual proxy evidence. It may contain
ring mask centroid, PCA yaw, approximate depth, and simple visibility scores.
The PCA/image-axis yaw signal is diagnostic only. It is not assumed to be the
true jaw-local basin error and is not trusted for runtime yaw control.

`EstimatedBasinError` contains:

- `dx, dy, dz, dyaw`
- per-axis validity and confidence
- frame consistency
- source and reason
- close-ready logic
- trusted control axes

The precision gate is currently tightened so that if no axis is marked trusted
for control, C2C does not take over. This is deliberate: a weak proxy signal
should not be allowed to drive a real robot action.

## Current Known Result

The current depthgate calibration says the existing proxy signals are not good
enough for runtime control.

Latest calibration summary from the depthgate 3-episode audit:

- `x`: sign match is relatively high, but contraction is too low; policy is
  `abstain`.
- `y`: sign match is high but not reliably contractive; policy is
  `diagnostic_only`.
- `z`: contraction is moderate but sign is inconsistent; policy is
  `diagnostic_only`.
- `yaw`: weak correlation with privileged yaw; policy is `abstain`.

After tightening runtime apply to only trusted axes, the smoke run
`basin_recovery_only_3ep_trusted_axes` produced:

- `c2c_gate_active = false` for all steps in episodes 5, 8, and 19.
- `phase_owner = planner` for all steps.
- The dominant gate reason after entering precision stages is
  `no_trusted_control_axis`.
- No gate-frame images were generated because the gate never opened.

This means the current code is no longer silently applying bad corrections, but
it also means C2C is not yet recovering the planner failure tail.

## Important Diagnosis

The current bottleneck is semantic, not just gain tuning.

The task contract says the grasp skill should estimate and reduce the real
jaw-local error:

- gripper jaw frame to ring grasp frame
- `dx, dy, dz, dyaw`
- with symmetry-aware yaw
- using a consistent local frame

The current heuristic localizer still produces a visual proxy:

- ring mask centroid
- PCA axis from partial crop
- approximate median depth
- image/crop-frame offsets

Those are useful diagnostics, but they are not proven to be the same coordinate
semantics as the true basin residual. As a result, using them directly can give
visually plausible but task-wrong corrections.

The current yaw path is especially sensitive here: image/PCA yaw remains a
diagnostic feature for offline analysis and focused evaluation, but it is not a
trusted runtime control axis until a frame-aware yaw estimator can predict the
jaw-local privileged `dyaw` reliably on the focused near-basin slice.

The learned residual/recovery heads explored so far also do not solve this by
themselves. They mostly learn single-step correction-like outputs. The desired
skill is a closed-loop behavior that:

- restores observability when the ring is not usable in the wrist view,
- estimates the true basin error when geometry is visible,
- applies only bounded, contractive corrections,
- re-observes after every step,
- enters a verified near-grasp basin before allowing close.

## What Is Working

- Task specs and skill contracts load.
- Unknown tasks can fall back to planner-only.
- Owner-by-stage wiring exists.
- Runtime trace fields are rich enough to audit ownership and action
  composition.
- Runtime invariants keep privileged targets and RLBench masks out of control.
- Learned checkpoints are treated as diagnostic-only unless explicitly allowed.
- Basin-state calibration can block untrusted axes.
- Unit tests cover stage ownership, action composition, recovery FSM, basin
  thresholds, calibration gating, and trace invariants.

Latest local test command:

```bash
conda run -n vla-adapter python -m unittest tests.test_coarse2contact_v2 -v
```

Latest observed result:

```text
Ran 38 tests
OK
```

## What Is Not Yet Working

- C2C does not yet prove closed-loop recovery from real planner failure tails.
- Current depth proxy does not provide trusted `x/y/z/yaw` basin axes.
- Image/PCA yaw remains diagnostic-only and should not be reinterpreted as jaw-
  local residual yaw.
- Yaw is especially unreliable because partial crop PCA and symmetry handling
  can invert or destabilize the intended grasp frame.
- Z is not yet a proper progress-to-contact / descend-to-close estimator.
- Close trigger should remain blocked until the estimated basin state is
  stable for multiple frames.
- Spoke alignment and force recovery are intentionally deferred until grasp
  basin recovery works.

## Current Smoke Outputs

These local artifacts are not tracked by git because `runtime_artifacts/` is
ignored, but they document the most recent smoke:

```text
runtime_artifacts/coarse2contact_v2/basin_recovery_only_3ep_trusted_axes/
```

Key files:

```text
eval_results.json
videos/ep005_fail.mp4
videos/ep008_fail.mp4
videos/ep019_fail.mp4
gripper_traces/ep005_gripper_trace.jsonl
gripper_traces/ep008_gripper_trace.jsonl
gripper_traces/ep019_gripper_trace.jsonl
gate_trace_split_report.md
```

The split report shows that all three episodes remain pre-gate; there is no
post-gate C2C control segment.

## Recommended Next Work

The next useful work is not another broad residual checkpoint. It should focus
on proving a true basin-state estimator and then a conservative closed-loop
controller.

Suggested next steps:

1. Build an independent privileged relabel audit for every post-depthgate
   candidate frame:
   - true jaw-local `dx/dy/dz/dyaw`
   - localizer proxy `dx/dy/dz/dyaw`
   - executed planner delta
   - next-frame true error
   - per-axis sign match and contraction

2. Replace crop proxy semantics with a frame estimator:
   - `RingFrameLocalizer`: ring visible, center heatmap, aperture/grasp frame,
     yaw observability, confidence.
   - It should answer "where is the ring frame?" rather than "what action
     should I take?"

3. Add a task-frame error estimator:
   - input: ring frame, jaw/gripper state, planner prior, local frame contract
   - output: calibrated jaw-local `EstimatedBasinError`
   - explicit axis validity
   - no control action output

4. Train or implement a conservative basin pullback controller:
   - active only when axes are trusted
   - pulls toward basin, not exact pose imitation
   - penalizes overshoot
   - re-observes every step
   - abstains or reacquires when visual evidence is weak

5. Prove grasp recovery before reopening close:
   - report true privileged error curves
   - report near-grasp and close-ready basin entry
   - report monotonicity and overshoot
   - only then enable close and lift verify

6. After grasp recovery works, reuse the same interface for:
   - held-ring aperture to spoke-axis alignment
   - guarded slide
   - force-triggered recovery

## Suggested Evaluation Commands

Smoke planner-only baseline:

```bash
bash scripts/run_c2c_v2_smoke_3ep.sh
```

Basin recovery smoke with calibration:

```bash
MODE=basin_recovery_only \
OUTPUT_ROOT=runtime_artifacts/coarse2contact_v2/basin_recovery_only_3ep_trusted_axes \
NAME_SUFFIX=coarse2contact_v2_basin_recovery_only_3ep_trusted_axes \
bash scripts/run_c2c_v2_basin_recovery_3ep.sh
```

Unit tests:

```bash
conda run -n vla-adapter python -m unittest tests.test_coarse2contact_v2 -v
```

## Ground Rules For Future Agents

- Do not use RLBench masks, object handles, success poses, or teacher targets
  in runtime control.
- Do not claim C2C is improving success until true failure-tail closed-loop
  basin entry improves.
- Do not judge recovery by pooled shadow MAE alone.
- Always split results by visual observability and failure bucket.
- Always inspect MP4 and trace together.
- Always compare `planner_action_world` to `pre_clip_action_world_6d` for
  ownership; do not compare relative planner deltas to absolute executed poses.
- Keep learned depth apply diagnostic-only until trusted axis evidence exists.
- Treat `v11`-style recovery heads as baselines, not proof of closed-loop
  recovery.
