# C2C v2 Belief + Forward Model Plan

This document converts the latest method review into the next concrete route
after the v46 typed-command ranker result.  The main correction is that C2C
should no longer treat the problem as only "estimate residual, then rank a
small command list."  The next step is an observability-aware belief model plus
an uncertainty-aware local forward model.

## Current Diagnosis

The current branch has a useful v46 scaffold:

- non-privileged wrist RGBD / depth validity / proprio / planner-prior inputs
- parallel XY/Z/Yaw residual estimates
- per-axis confidence, observability, yaw ambiguity, step scale, and risk
- command-conditioned transition and outcome heads
- source-root held-out validation
- planner-owned close and strict `alignment_ready_for_handoff`

The blocker is not close ownership and not a missing scalar rank loss.  The
latest evidence says:

- XY/Z state evidence is useful, but XY is not solved under held-out hard
  roots, partial view, and large-XY/large-Yaw windows.
- Yaw is conditionally controllable, not globally observable.  Symmetry and
  occlusion make many wrist views one-to-many in yaw.
- Candidate ranking improves when candidate type/context is explicit
  (`typed16` removes the previous worse-than-zero hard-flush fold), but the
  ranker still only ties zero on the worst roots and can sacrifice XY.
- The scarce data is not state labels.  The scarce data is same-window,
  same-source, zero-baselined executed transition supervision.

Therefore the next phase should optimize the belief/control semantics, not just
add another no-op margin.

## Clear Target

Build and validate `belief_forward_task_frame_candidate`:

**A non-privileged, observability-aware task-frame belief estimator and
uncertainty-aware local forward model that chooses bounded XY/Z/Yaw corrections
only when the predicted post-command residual beats zero/no-op under
source-held-out transition evidence.**

This is not a baseline promotion target by itself.  It is the next candidate
that must beat v46 on source-held-out/random-held-out failure tails before any
runtime MP4 or insert-success claim.

## Non-Negotiable Constraints

- Runtime environment:
  `conda run -n vla-adapter ...`
- Canonical RLBench smoke/eval path:
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py ...`
- Fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- Current XY baseline:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- v46 remains a candidate, not a promoted baseline.
- Runtime input remains non-privileged.  Privileged residuals are allowed only
  for offline labels, relabeling, sidecar eval, and audits.
- `alignment_ready_for_handoff` remains the only close/handoff predicate.
- C2C close authority remains disabled.  C2C may request open-only safety.
- Axis correction does not imply handoff.  A useful correction may happen while
  handoff stays false.

## Phase 1: Belief State, Not Only Residual Regression

The next state model should output an explicit task-frame belief:

- `mean_residual`: `dx, dy, dz, dyaw`
- `residual_uncertainty`: per-axis variance or calibrated interval
- `axis_observable`: XY, Z, Yaw
- `axis_controllable`: whether a bounded command is expected to affect the axis
- `yaw_modes`: symmetry-aware multi-hypothesis yaw distribution
- `yaw_ambiguous` / `yaw_unobservable`
- `recommended_policy`: `correct`, `tiny_step`, `hold`, `reacquire`, `probe`
- `risk_reason`: visibility, alias drift, frame conflict, force guard, low
  support, out-of-distribution root/window

Training priorities:

- reward correct abstain/reacquire on truly unobservable yaw windows
- do not force a single dyaw target in ambiguous windows
- calibrate uncertainty against held-out contraction/worsen, not only MAE
- split metrics by source root, observability bucket, and hard bucket

Minimum belief gate:

- XY/Z observable slices keep or improve v46 contraction
- yaw-observable precision improves without increasing yaw false positives
- yaw-ambiguous/unobservable windows produce abstain/reacquire/probe decisions,
  not confident yaw commands
- no close leak and no handoff relaxation

## Phase 2: Uncertainty-Aware Forward Model

Move the controller core from "candidate scalar score" to explicit
command-conditioned post-residual prediction.

Inputs:

- belief-state features
- wrist RGBD/depth validity
- spatial moment/geometry features
- proprio and planner prior
- recent history
- typed candidate command context

Outputs:

- `predicted_post_residual_mean`
- `predicted_post_residual_logvar`
- per-axis contraction probabilities
- per-axis worsen/collateral probabilities
- beat-zero probability
- support / out-of-distribution score

Selection rule:

- enumerate a small bounded candidate set, including zero/no-op
- choose a non-zero command only when expected utility beats zero after
  uncertainty and collateral penalties
- require positive worst-root evidence in LOO/source-held-out validation
- allow abstain/reacquire/probe when all candidates are unsupported or tied
  with zero

Loss terms:

- continuous post-residual NLL or robust regression
- one-step contraction / worsen / overshoot penalties
- same-window pairwise oracle/zero margins
- support calibration
- uncertainty calibration against held-out source roots
- XY collateral penalty when a yaw/Z command damages XY

## Phase 3: Data Plan

Use three data layers instead of treating all rows equally.

State/belief data:

- continue using large runtime snapshots with offline relabels
- emphasize random-held-out, hard bucket, partial view, and low visibility
- keep old4/random5/random10 as sentinels, not optimization targets

Applied-transition data:

- make existing applied-transition manifests first-class training data
- train on actual `command -> next_residual` transitions, not only candidate
  labels
- preserve source-root/session held-out splits

Command-sweep data:

- reserve expensive command sweeps for high-information windows
- every sweep must include same-window zero/no-op
- prioritize:
  - large-XY/large-Yaw flush and retain roots
  - yaw-observable near-contact windows
  - near-contact ambiguous windows that should trigger reacquire/probe
  - partial-view and occlusion hard buckets
  - windows where typed16 fixed worse-than-zero but still only tied zero
  - windows where typed16 sacrifices XY while improving Z/Yaw

Target scale for a credible forward-model gate:

- high hundreds of candidate groups
- low thousands to low tens of thousands of executed transitions
- balanced coverage of yaw-observable, yaw-ambiguous, large-XY/large-Yaw,
  partial-view, and same-window zero rows

## Phase 4: Active Reacquire / Probe

Yaw should not be treated as always observable from passive wrist history.

Add a conservative information-gathering action class:

- `reacquire_view`: move to improve wrist visibility without approaching close
- `probe_tiny_yaw`: tiny bounded yaw candidate only under force-safe and
  non-close conditions
- `hold_open`: no correction, keep gripper open, collect more evidence

These actions do not set handoff ready and do not authorize close.

Gate:

- ambiguous yaw windows should prefer abstain/reacquire/probe over wrong yaw
  servo
- probe actions must have close leak count zero
- probe/reacquire should improve later observability or reduce posterior
  uncertainty on held-out roots

## Evaluation Plan

Offline / replay:

- v42 XY baseline
- v46 typed16 outcome/pairwise ranker
- belief-only candidate
- belief-forward candidate

Required slices:

- random held-out failure tails
- large-XY/large-Yaw flush and retain
- small-XY/large-Yaw
- partial-view / low-visibility
- yaw-observable near-contact
- yaw-ambiguous near-contact
- old4/random5/random10 sentinels

Metrics:

- per-axis contraction and worsen
- combined contraction
- top1 minus zero per axis and combined
- uncertainty calibration
- yaw false-positive rate
- correct yaw abstain/reacquire rate
- XY collateral under Z/Yaw commands
- close leak count
- handoff precision
- closed-loop insert success only after offline gate passes

Promotion gate:

- worst-root top1 minus zero combined and yaw must be positive, not merely tied
- XY contraction must not regress relative to v46 typed16 and v42 baseline
- yaw-observable slice must beat zero without increasing false-positive yaw
  actions on ambiguous/unobservable slices
- close leak count must remain zero
- MP4 success-claim smoke is allowed only after the offline/replay gate is
  credible

## Immediate Execution Checklist

1. Freeze strict close/handoff authority and keep all close tests green.
2. Build a `belief_forward_manifest` from existing v46 broad/yawbalanced,
   random-heldout, hard-bucket, and applied-transition manifests.
3. Add belief labels for axis observability, controllability, uncertainty, and
   recommended policy: correct / hold / reacquire / probe.
4. Refactor the command-transition training objective to predict continuous
   post-residual mean/logvar as the primary target.
5. Keep `typed16` candidate context, but add uncertainty and XY-collateral
   utility terms.
6. Re-run the 10-root LOO gate and require:
   - worse-than-zero folds: `0`
   - worst-root combined/yaw minus zero: `> 0`
   - no XY contraction regression
7. Only then run canonical 150/180-step MP4 with front+wrist views.

## Current Status

As of the v46 typed16 result:

- v46 is a useful scaffold and candidate, not a baseline.
- typed command context fixed one hard worse-than-zero fold.
- the formal gate still fails because worst roots only tie zero and XY regresses.
- the next breakthrough should be a belief + forward-model candidate, not
  another pure rank-loss sweep.
