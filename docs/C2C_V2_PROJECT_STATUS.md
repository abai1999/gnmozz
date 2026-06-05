# Coarse2Contact v2 Project Status

This document is written for future reviewers, agents, and collaborators who
need to understand the current C2C v2 state without replaying the full
conversation. It is intentionally candid: the scaffold is useful, but the
current local skill layer does not yet solve the real failure-tail recovery
problem.

For a shorter external-review entry point, see
[`docs/AI_REVIEW_GUIDE.md`](AI_REVIEW_GUIDE.md). For a route-level review of
whether the project has drifted from its research goal, see
[`docs/C2C_V2_RESEARCH_REVIEW_BRIEF.md`](C2C_V2_RESEARCH_REVIEW_BRIEF.md).
For the next concrete XY generalization push, see
[`docs/C2C_V2_XY_V42_GENERALIZATION_PLAN.md`](C2C_V2_XY_V42_GENERALIZATION_PLAN.md).
For the next Z/Yaw readiness push after v42, see
[`docs/C2C_V2_Z_YAW_READINESS_PLAN.md`](C2C_V2_Z_YAW_READINESS_PLAN.md).
For the current review of v43/v44 and the next handoff audit plan, see
[`docs/C2C_V2_V43_V44_REVIEW_AND_NEXT_PLAN.md`](C2C_V2_V43_V44_REVIEW_AND_NEXT_PLAN.md).

Operational defaults for this branch:

- Runtime environment: `conda run -n vla-adapter ...`
- Fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`

Current objective:

- `v42_expanded_v4pilot` is now the active XY baseline.
- The concrete active XY checkpoint is
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`.
- The immediate engineering target is now to keep strict
  `alignment_ready_for_handoff` as the only close predicate while feeding it
  with non-privileged task-frame Z and Yaw readiness.
- The current readiness checkpoints are:
  - `runtime_artifacts/coarse2contact_v2/checkpoints/v43_task_frame_z_readiness.pt`
  - `runtime_artifacts/coarse2contact_v2/checkpoints/v44_task_frame_yaw_readiness.pt`
- The current readiness dataset is:
  - `runtime_artifacts/coarse2contact_v2/datasets/task_frame_readiness_v43_v44.jsonl`

Success means:

- v42 XY remains non-regressed on random holdout, hard-bucket, old4/random5,
  low-visibility, and ep25/26 slices
- v43 Z readiness reaches high held-out precision on v42-XY-ready rows and is
  conservative at runtime
- v44 Yaw readiness blocks ambiguous / alias-drift rows instead of forcing them
  ready
- no reopening of `alignment_ready_for_handoff` or `close_ready`

Current status snapshot:

- `v42_expanded_v4pilot` is the active XY baseline after the
  sentinel-aware warm-start A/B, and it is the first candidate in this line to
  improve random10, random holdout, old4 reverse, and random5 reverse at the
  same time.
- This candidate is also the one favored by the latest hard-bucket A/B on
  contraction, worsen, overshoot, reverse, and ep25/26 worsen.
- The hard-bucket report also shows low-visibility worsen improving from
  `0.199` to `0.055`, with partial-worsen staying at `0.000`.
- Latest measured comparison against `v41`:
  - random10 contraction `0.825` vs `0.667`
  - random holdout contraction `0.896` vs `0.760`
  - old4 reverse `0.583` vs `0.659`
  - random5 reverse `0.514` vs `0.538`
  - hard-bucket contraction `0.945` vs `0.801`
  - hard-bucket worsen `0.055` vs `0.199`
  - hard-bucket low-visibility worsen `0.055` vs `0.199`
  - hard-bucket partial worsen `0.000` vs `0.000`
  - hard-bucket overshoot `0.059` vs `0.215`
  - hard-bucket reverse `0.025` vs `0.054`
  - hard-bucket ep25/26 worsen `0.007` vs `0.127`
- The current line is still valid, but future changes must still beat the
  active baseline on worst-slice random generalization and hard-bucket tails,
  not just on cherry-picked MP4 clips.
- The gate-aware training path now explicitly separates
  `random10_generalization` from training roots and scores checkpoints with a
  worst-case val/gate/holdout rule.
- Among the measured candidates, `v42_expanded_v4pilot` is the strongest
  measured XY model so far on the small gate/holdout/sentinel evaluation set
  and the current baseline to beat.
- The active XY baseline is now `v42_expanded_v4pilot`.
- The next milestone is now implemented in checkpoint form:
  - `v43_task_frame_z_readiness` was trained on the consolidated
    `task_frame_readiness_v43_v44.jsonl` dataset and reached held-out
    precision `0.984` and recall `0.987` at threshold `0.80` on the
    root-held-out validation split after fixing compact-dataset/runtime feature
    extraction parity.
  - `v44_task_frame_yaw_readiness` was trained on the same dataset and reached
    held-out precision `1.000` and recall `1.000` at threshold `0.05` on the
    root-held-out validation split.
  - Both checkpoints are loaded by `evaluate_c2c_v2_rlbench.py` and keep
    `close_ready` as legacy/diagnostic only.
- Two short xvfb smoke runs completed on
  `insert_onto_square_peg` with the strict handoff gate wired in. They reached
  the runtime trace path and preserved the close block semantics, but the
  short step budget did not reach a true planner close request in those runs.

Latest smoke confirmation:

- old4 smoke root:
  `runtime_artifacts/coarse2contact_v2/mp4_smoke_v42_expanded_v4pilot_old4_front_wrist`
- random5 smoke root:
  `runtime_artifacts/coarse2contact_v2/mp4_smoke_v42_expanded_v4pilot_random5_front_wrist`

Both smokes preserved the wrist camera view and kept planner close under the
strict handoff guard. They are visual evidence only; they do not replace the
offline worst-slice gate.

Hard-bucket smoke for the leading candidate also exists:

- hard-bucket smoke root:
  `runtime_artifacts/coarse2contact_v2/mp4_smoke_v42_expanded_v4pilot_hardbucket_front_wrist`

This smoke kept the same strict handoff guard and wrist-view layout. It is
useful for inspection, but it still does not override the offline promotion
gate.

Additional implementation smoke:

- `runtime_artifacts/coarse2contact_v2/eval_z_yaw_sanity_v43_v44_xvfb`
- `runtime_artifacts/coarse2contact_v2/eval_z_yaw_sanity_v43_v44_xvfb_ep3`

These runs exercised the new Z/Yaw readiness checkpoints under `xvfb-run`.
They confirmed the readiness heads load, the runtime trace fields populate, and
`alignment_ready_for_handoff` remains false when Z/Yaw are not ready. The
episodes did not request planner close within the short smoke budget, so they
are semantic sanity checks rather than a final close-block proof.

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

The gate is now tracked as three separate readiness levels:

- `pullback_ready`: a bounded x/y pullback window. This can be true with
  z/yaw still diagnostic or abstained.
- `micro_entry_ready`: a stricter near-basin entry for small local servoing.
- `close_ready`: the strict contact/close entrance; this must remain blocked
  until the basin state is stable enough to close safely.

This split matters for the current research line: C2C may gather and audit
coarse pullback evidence without claiming that close/contact is safe.

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

## Current Hard-Bucket Validation

The latest hard-bucket validation on the fixed 30k planner, restricted to
`RING_GRASP_ALIGN`, made the current diagnosis more precise.

For `small_xy_large_yaw`, the direction diagnostic now treats
`oracle_xy_step_cosine_to_residual` as the formal sign metric and points to
`step_too_small_candidate`, not a sign flip. The key `ep000` and `ep010`
small slices show `oracle_xy_step_cosine_to_residual = 1.0`, which means the
oracle step is aligned with the true residual direction; the problem is that the
step is too conservative, not reversed.

For the hard-bucket sweeps:

- `large_xy_large_yaw` now produces real active rows under the widened entry
  support, but it still has `horizon_near_grasp_after_rate = 0.000`.
- `small_xy_large_yaw` now has a narrow-window validation at
  `xy_gain=0.50`, `max_xy_step=0.0025`, `horizon=2` with alias-aware
  candidates. The rebuilt candidate manifest eliminated `unknown` entirely by
  falling back to episode-level alias support, and the flush/retain comparison
  now clearly favors flush on the narrow slice: flush keeps overshoot lower and
  near-grasp higher, while retain is noisier and more over-shoot prone. This
  should be treated as a window-protocol conclusion, not a sign-flip one.
- The hard-window support supplement now gives `small_xy_large_yaw` a slightly
  looser outer/frontier allowance than the other hard buckets, so the support
  surface can widen without changing the runtime gate.
- `alias_drift_decision` is now propagated through candidate, trace, and
  support manifests, with the rebuilt alias-aware candidate set split cleanly
  into `stable_alias_control` and `frame_drift_abstain`.

The older formal acceptance snapshot was:

| bucket | active / rows | oracle contraction | planner contraction | improvement | overshoot | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `small_xy_large_yaw` | `3496 / 14352` | `0.382` | `0.278` | `0.757` | `0.075` | older wide-window snapshot; the newer narrow `h2/ms0025` sweep now shows flush > retain |
| `large_xy_large_yaw` | `45 / 9568` | `0.914` | `0.311` | `0.908` | `0.000` | tail support now spans the remaining 16..29 episodes too |
| `large_xy_small_yaw` | `283 / 8280` | `0.804` | `0.379` | `1.000` | `0.000` | support is clearly paying off |
| `small_xy_small_yaw` | `428 / 8680` | `0.773` | `0.370` | `0.906` | `0.037` | strong but not the current bottleneck |

This table is the clearest current summary: the big-bucket entry support is
working, while the small bucket still needs step-size / bracket refinement and
cleaner alias/drift split before it can be treated as a finished support story.

The cleaner focused-validation read after the `xy_correction_ready` and
hard-bucket support updates is:

| bucket / protocol | active rows | xy contraction | overshoot | current interpretation |
| --- | ---: | ---: | ---: | --- |
| `large_xy_large_yaw` / flush | 52 | 0.935 | 0.000 | active support exists, but support remains narrow |
| `large_xy_large_yaw` / retain | 74 | 0.919 | 0.000 | support is wider than flush in this slice; protocol is not the main blocker |
| `small_xy_large_yaw` / flush | 212 | 0.791 | 0.028 | active support exists; step-size / horizon need more bracketing |
| `small_xy_large_yaw` / retain | 212 | 0.717 | 0.023 | contraction is weaker but comparable; do not call this a sign flip |

This focused read should guide the next loop: widen `large_xy_large_yaw`
support around real active windows, and keep `small_xy_large_yaw` focused on
step-size / horizon / frame-sign diagnosis. Queue flushing is a protocol choice
to measure, not the primary bottleneck by itself.

One important audit gap remains: some aggregate focused-sweep summaries still
do not expose `alias_drift_decision`. The field is present in candidate/trace
plumbing, but every future hard-bucket table should explicitly split active,
contraction, near-entry, and overshoot by `stable_alias_control`,
`frame_drift_abstain`, and `unknown`.

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

For paper framing, the strongest current claim should remain a
contract-calibrated precision takeover framework. Planner-only success windows,
success-window videos, and replay-only oracle probes are not final evidence for
closed-loop failure-tail recovery.

Suggested next steps:

1. Make focused hard-bucket summaries expose `alias_drift_decision` end to end:
   candidate manifest, trace row, failure-tail audit, and final comparison
   table.

2. Continue widening `large_xy_large_yaw` entry support, but only around
   episode/window slices that already show real active/contraction evidence.

3. Keep `small_xy_large_yaw` on narrow step-size / horizon brackets and use
   `oracle_xy_step_cosine_to_residual` as the formal sign metric. The older
   `oracle_xy_step_cosine_to_descent` field is compatibility-only.

4. Generate MP4 comparisons only from active, contractive failure-tail rows,
   and inspect them together with trace summaries.

5. Build an independent privileged relabel audit for every post-depthgate
   candidate frame:
   - true jaw-local `dx/dy/dz/dyaw`
   - localizer proxy `dx/dy/dz/dyaw`
   - executed planner delta
   - next-frame true error
   - per-axis sign match and contraction

6. Replace crop proxy semantics with a frame estimator:
   - `RingFrameLocalizer`: ring visible, center heatmap, aperture/grasp frame,
     yaw observability, confidence.
   - It should answer "where is the ring frame?" rather than "what action
     should I take?"

7. Add a task-frame error estimator:
   - input: ring frame, jaw/gripper state, planner prior, local frame contract
   - output: calibrated jaw-local `EstimatedBasinError`
   - explicit axis validity
   - no control action output

8. Train or implement a conservative basin pullback controller:
   - active only when axes are trusted
   - pulls toward basin, not exact pose imitation
   - penalizes overshoot
   - re-observes every step
   - abstains or reacquires when visual evidence is weak

9. Prove grasp recovery before reopening close:
   - report true privileged error curves
   - report near-grasp and close-ready basin entry
   - report monotonicity and overshoot
   - only then enable close and lift verify

10. After grasp recovery works, reuse the same interface for:
   - held-ring aperture to spoke-axis alignment
   - guarded slide
   - force-triggered recovery

## Latest Implementation Guardrails

- MP4 smoke is now explicitly split into two evidence types:
  `diagnostic_privileged_probe` and `runtime_style_c2c`. Diagnostic probes may
  use privileged oracle residuals for eval-only intervention analysis. Runtime
  style smoke cannot use `forced_shell`, and while `replay_oracle_xy` remains
  active it must be described as a runtime-style eval probe, not true
  non-privileged closed-loop recovery.
- The latest random failure-tail MP4 check (`ep000/003/011/018`) showed that
  broad diagnostic support was being mistaken for runtime takeover evidence.
  After enforcing mature runtime-stage windows and excluding frontier/coarse
  support tiers, early coarse-trajectory takeover is blocked, but active rows
  drop to zero because mature candidate rows mostly become
  `prior_only_abstain`.
- A representative `prior_only_abstain` row is not a missing-ring case:
  `local_geometry_error.grasp.valid=true` with usable confidence/observability,
  but `EstimatedBasinError` marks all axes invalid under
  `basin_axis_policy={x: abstain, y/z: diagnostic_only, yaw: abstain}`. The
  current blocker is therefore calibrated runtime XY residual semantics, not
  merely MP4 smoothing or action gain.
- `scripts/diagnose_c2c_v2_prior_only_abstain.py` is the official diagnostic
  for this failure mode. It classifies prior-only rows without using privileged
  residuals, separating missing/weak localizer evidence from estimator-axis
  policy abstain and trace plumbing mismatches.
- `prismatic.robot.coarse2contact_v2.runtime_xy_residual` defines the current
  non-privileged XY residual evidence boundary. It intentionally keeps yaw and
  close disabled and only reports whether visual evidence plus calibrated proxy
  axes are ready for bounded XY pullback.
- Runtime XY pullback calibration is now explicit. When a report-loaded basin
  calibration would otherwise keep `x/y` in `abstain/diagnostic_only`, the
  evaluator can enable an XY-only override that marks `x/y` as
  `trusted_control` for bounded pullback while keeping `z=diagnostic_only` and
  `yaw=abstain`. This fixes the prior-only plumbing gap without reopening yaw
  or close.
- On the v25 runtime-style trace replay for `ep000/003/011/018`, prior-only
  abstain dropped from `315/880` rows to `0/880`. Active rows appeared only
  after the stage matured (`first_active_stage_age >= 20`), with
  `z_valid_rows=0`, `yaw_valid_rows=0`, and `close_ready_rows=0` in every
  episode. This is the first clean evidence that local visual evidence plus
  XY-only calibration can open bounded pullback windows without unlocking
  yaw/close.
- Focused hard-bucket summaries must now expose `alias_drift_decision` directly:
  `stable_alias_control`, `frame_drift_abstain`, and `unknown` each get their
  own active/contraction/near-entry/overshoot read. A large `unknown` share is
  an audit plumbing problem, not yaw evidence.
- Runtime gripper traces keep privileged pose/frame metadata under
  `offline_eval_only`. The supervisor still reports
  `uses_privileged_runtime=false`; offline relabel/probe fields are not runtime
  observations.
- C2C alignment handoff is now a first-class lifecycle concept. The runtime
  smoke path emits `takeover_session_id`, `takeover_lifecycle_state`,
  `terminal_state`, `alignment_ready_for_handoff`, `safe_abstain_open`,
  `failed_retryable`, `failed_terminal`, `budget_used`, and
  `final_axis_readiness`. Sticky smoothing no longer defines semantic success
  or exit; it only smooths correction commands.
- C2C still does not own gripper closing. It only decides whether planner
  gripper handoff is allowed. In precision alignment windows, planner close
  requests must pass `alignment_ready_for_handoff`; otherwise the trace records
  `planner_gripper_close_blocked` and keeps the gripper open.
- `TaskFrameResidualEstimate` is the new target abstraction for generalized
  alignment. It records `reference_frame`, `target_frame`, `active_dofs`,
  `dx/dy/dz/dyaw`, per-axis validity/confidence, and explicit `z_semantics` /
  `yaw_semantics`. `dz` is task approach-axis residual, not a global world-z
  threshold, and image/PCA yaw remains diagnostic until a held-out yaw estimator
  passes validation.
- Supervisor traces now include a runtime proxy takeover-contract view so
  reviewers can compare `pullback_gate_ready / micro_entry_ready / close_ready`
  against the formal tier language. This is a contract-alignment diagnostic,
  not proof that the proxy residual is a true jaw-local estimator.
- Grasp runtime residual semantics remain explicitly marked as
  `calibrated_proxy`: mask centroid, median depth, and image-axis yaw are visual
  evidence, not a solved frame residual estimator.
- Failure-tail manifests are not sufficient to activate C2C by themselves. The
  eval probe now also requires the row to be inside the configured
  high-precision XY activation window, so coarse-approach rows cannot become
  takeover evidence just because they appear in a manifest.
- MP4 close-handoff smoke must distinguish close command, observed closed
  gripper state, and task reward. A close command or handoff is not counted as
  successful grasp/close evidence unless the trace also shows physical closure
  and success evidence.
- Runtime XY estimator status, as of the latest v36 check:
  - `runtime_xy_affine_calibration_v25.json` remains the default runtime smoke
    calibration. It is a small-data affine calibrator trained on 60 active rows
    from `ep000/003/011/018`, so it is useful but not final generalization
    evidence.
  - `runtime_xy_affine_calibration_hard_occlusion_v34_stable.json` used wider
    hard-bucket / occlusion data, but failed runtime A/B. On the same MP4 smoke
    set it dropped estimator direction alignment from `0.989` to `0.031` and
    raised `step_too_small_rate` to `0.816`; it must not replace v25.
  - The training script now supports a direction-first / control-aware
    objective and JSON-serialized MLP checkpoints. The v36 MLP candidate
    strongly improves offline direction metrics but has not passed runtime
    replacement criteria: MP4 contraction was `0.793` vs v25 `0.826`, and
    hard-bucket contraction was `0.826` vs v25 `0.913`, despite a modest
    near-entry gain. It remains a candidate, not the default.
  - Any future XY estimator upgrade must pass both MP4 runtime A/B and
    hard-bucket runtime A/B before replacing v25. Offline MAE, pooled cosine,
    or posthoc direction scores are insufficient by themselves.
- Current project bottleneck:
  - XY correction is now visibly useful but not robust enough to declare solved.
    The best next XY work is a control-aware estimator that preserves v25-like
    direction reliability while widening hard-bucket / occlusion coverage.
  - The larger blocker for task success is still `z/yaw` readiness. The
    alignment lifecycle correctly blocks planner gripper handoff when
    `alignment_ready_for_handoff=false`; the remaining gap is to learn
    non-privileged task-frame `z_readiness` and `yaw_readiness`, not to loosen
    the close gate.

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

Alignment takeover smoke summary:

```bash
conda run -n vla-adapter python scripts/summarize_c2c_v2_alignment_takeover_smoke.py \
  --trace_dir runtime_artifacts/coarse2contact_v2/<smoke_run>/gripper_traces \
  --output_dir runtime_artifacts/coarse2contact_v2/reports/<smoke_run>_alignment_takeover
```

XY + alignment lifecycle diagnostic:

```bash
conda run -n vla-adapter python scripts/diagnose_c2c_v2_xy_correction_hard_validation.py \
  --trace_dir runtime_artifacts/coarse2contact_v2/<smoke_run>/gripper_traces \
  --runtime_obs_dir runtime_artifacts/coarse2contact_v2/<smoke_run>/runtime_observations \
  --output_dir runtime_artifacts/coarse2contact_v2/reports/<smoke_run>_xy_alignment
```

## Ground Rules For Future Agents

- Do not use RLBench masks, object handles, success poses, or teacher targets
  in runtime control.
- Do not claim C2C is improving success until true failure-tail closed-loop
  basin entry improves.
- Do not judge recovery by pooled shadow MAE alone.
- Always split results by visual observability and failure bucket.
- Always inspect MP4 and trace together.
- Do not treat `shell_filter=off` coarse-window probes as high-precision
  takeover evidence unless the run is explicitly labeled diagnostic-only.
- Always compare `planner_action_world` to `pre_clip_action_world_6d` for
  ownership; do not compare relative planner deltas to absolute executed poses.
- Keep learned depth apply diagnostic-only until trusted axis evidence exists.
- Treat `v11`-style recovery heads as baselines, not proof of closed-loop
  recovery.
