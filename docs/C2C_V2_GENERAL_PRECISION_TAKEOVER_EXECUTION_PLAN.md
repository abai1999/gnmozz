# C2C v2 General Precision Takeover Execution Plan

This is the execution plan for the next stage of C2C v2.  It is written for an
AI engineer to implement without relying on the prior conversation.

The goal is not to add another ranker, not to tune a single episode, and not to
make a task-specific patch for one hard bucket.  The goal is to turn C2C into a
general non-privileged local precision layer that can take over high-precision
subtasks, correct local task-frame error, abstain when the state is not
observable, and hand control back to the frozen planner only through the strict
handoff contract.

`insert_onto_square_peg` remains the first proving task, but the design must be
general enough for other precision skills: grasp alignment, aperture/frame
alignment, guarded slide, local insertion, and future tasks that can express a
reference frame, target frame, controlled axes, tolerance, and safety contract.

## Final Objective

Implement and validate `belief_forward_task_frame_candidate`:

**A non-privileged, observability-aware belief estimator plus
uncertainty-aware command-conditioned forward model for local high-precision
control.  It predicts which task-frame axes are observable/controllable,
predicts post-command residual under bounded candidate actions, and executes
only open-safe corrections/reacquire/probe actions whose conservative predicted
benefit beats same-window zero/no-op without collateral damage.**

The final success target is:

- on random/source-held-out failure tails, C2C reduces true task-frame residuals
  for the active precision skill;
- C2C improves `insert_onto_square_peg` success after the frozen planner reaches
  the local precision region;
- C2C never uses privileged runtime inputs;
- C2C never regains close authority;
- strict `alignment_ready_for_handoff` remains the only handoff/close predicate.

## Current Assets

Keep these assets and build on them.

### Safety and Contract Assets

- Planner-owned close is already the correct runtime ownership model.
- C2C has open-only safety authority, not close authority.
- `alignment_ready_for_handoff` is the only close/handoff predicate.
- The gripper authority trace records planner close request/block/allow state,
  C2C open-safety requests, ignored close recommendations, and authority
  source.
- Close leak tests and trace audits are mandatory regression checks.

### Data and Audit Assets

- Runtime observations and traces can be relabeled offline with privileged
  residuals.
- Source-root/session held-out splitting is already part of the project
  discipline.
- Existing audits can separate:
  - state residual error,
  - command-effect error,
  - yaw ambiguity/observability,
  - XY collateral,
  - worse-than-zero actions,
  - close leaks,
  - source leakage.

### Model Assets

- `v42_expanded_v4pilot` is the current XY baseline to beat, not solved XY.
- v46 provides the useful scaffold:
  - wrist RGBD/depth-valid input,
  - proprio/planner prior/history,
  - parallel task-frame residual heads,
  - observability/confidence/yaw ambiguity/risk outputs,
  - command transition/outcome heads.
- `typed16` candidate context is a useful improvement:
  - raw command,
  - candidate type bits,
  - normalized magnitudes,
  - Z/Yaw signs,
  - command norm.
  It removed worse-than-zero folds in the 10-root LOO smoke, but still tied
  zero on worst roots and regressed XY.

## Current Root Causes

Do not misdiagnose the problem as only a missing loss term.

### 1. Belief Semantics Do Not Yet Match Control Consequences

The model may estimate a plausible residual but still misunderstand what a
bounded command will do in the true task frame.  Prior audits showed strong
axis sign/correlation failures, especially in Y under some runtime slices.
This means the controller needs a command-conditioned consequence model, not
only a residual estimator.

### 2. XY Is Valuable But Not Solved

XY is not information-theoretically impossible, but it is still vulnerable to
partial view, occlusion, approach-angle changes, frame drift, and XY collateral
from Z/Yaw commands.  Any next candidate that improves Yaw/Z while regressing
XY is not promotable.

### 3. Yaw Is Partly an Observability Problem

The square ring has 90-degree symmetry.  Wrist RGBD may be one-to-many under
partial view and occlusion.  In those windows the correct behavior is not to
guess dyaw.  The correct behavior is to abstain, hold, reacquire, or execute an
open-safe information probe.

### 4. The Scarce Data Is Executed Same-Window Transition Supervision

There are many relabeled state rows, but the most valuable control data is:

```text
same window:
  zero/no-op
  candidate commands
  post-command observation
  post-command true residual
  collateral labels
```

The next phase must prioritize high-quality transition groups over generic row
count.

## Non-Negotiable Runtime Constraints

These are safety invariants, not experiment settings.

- Runtime environment:
  `conda run -n vla-adapter ...`
- Canonical RLBench path:
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py ...`
- Fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- Runtime input must be non-privileged:
  no RLBench object handles, success poses, teacher target poses, privileged
  masks, or privileged residuals in the control loop.
- Privileged residuals are allowed only offline for labels, audits, relabeling,
  and sidecar evaluation.
- Runtime close remains:

```text
planner close intent
AND strict alignment_ready_for_handoff == true
AND all required task-frame axes satisfy the skill contract
AND close leak count remains zero
```

- Diagnostic gates may be split by axis.  Runtime close/handoff gates may not.
- A stage-specific XY/Z/Yaw success milestone must never authorize close.
- Reacquire/probe/correction actions must keep the gripper open and cannot set
  `alignment_ready_for_handoff`.

## Generalization Requirement

The implementation must be skill-contract based, not insert-hardcoded.

Every precision skill should expose a contract like:

```text
reference_frame
target_frame
controlled_axes
axis_tolerances
axis_required_for_handoff
symmetry_periods
approach_axis
safe_action_bounds
observability_requirements
contact/force guards
handoff predicate
```

The belief and forward model should consume this contract and emit axis-aware
decisions.  For `insert_onto_square_peg`, the active axes are XY/Z/Yaw around
ring/aperture/spoke frames.  For future high-precision operations, the same
schema must support different frames and axis requirements.

Do not encode assumptions like "all tasks are square ring yaw" into the
generic controller.  Put task-specific assumptions in the skill contract:
symmetry, tolerances, frame definitions, and safe candidate set.

## Architecture

### Module 1: Skill Contract Resolver

Input:

- current task/stage,
- planner state,
- trace row,
- task YAML / precision skill config.

Output:

- `PrecisionSkillContract`.

Responsibilities:

- identify the active precision skill;
- define reference/target frames;
- define controlled axes and tolerances;
- define which axes are required for handoff;
- define symmetry and yaw ambiguity semantics;
- define candidate bounds and probe/reacquire limits.

### Module 2: Belief Estimator

Runtime-visible inputs:

- wrist RGBD crop and depth-valid channels;
- optional front RGBD if already available in the normal observation, but no
  privileged masks;
- proprio / gripper state;
- planner prior / local delta;
- recent motion/action/history;
- previous C2C estimate and action;
- skill contract embedding.

Outputs:

```text
mean_residual: per active axis
residual_logvar: per active axis
axis_observable
axis_controllable
axis_confidence
yaw_modes / symmetry hypotheses when applicable
yaw_ambiguous / yaw_unobservable when applicable
recommended_mode: correct / hold / reacquire / probe
risk_reason
```

Training:

- use offline privileged labels only as targets;
- do not force single yaw residual in ambiguous windows;
- train yaw permission for high precision first, then recall;
- train uncertainty to predict held-out error/worsen, not only MAE;
- train `recommended_mode` from observability, controllability, and transition
  outcomes.

### Module 3: Command-Conditioned Forward Model

Runtime-visible inputs:

- belief state;
- visual embedding;
- proprio/history/planner prior;
- skill contract embedding;
- typed candidate action context.

Candidate action types:

```text
zero
axis_correction
mixed_small_correction
hold_open
reacquire_view
tiny_probe
```

Outputs:

```text
post_residual_mean
post_residual_logvar
axis_contraction_probability
axis_worsen_probability
combined_contraction_probability
beat_zero_probability
support / OOD score
observability_after
uncertainty_after
```

Primary loss:

- continuous post-residual NLL or Huber;
- per-axis worsen/collateral BCE;
- uncertainty calibration;
- support/OOD calibration.

Auxiliary loss:

- same-window oracle/zero ranking;
- beat-zero BCE;
- contraction labels.

Do not make ranking the primary objective.

### Module 4: Conservative Bounded Search

Never select a command by simple `argmin predicted_post_residual`.

Use hard constraints first:

```text
candidate inside safe action bounds
predicted XY worsen probability < threshold
predicted Z worsen probability < threshold
predicted yaw worsen probability < threshold if yaw observable
no dyaw correction if yaw ambiguous/unobservable
predicted beat-zero margin > uncertainty margin
support score above threshold
force/contact guard safe
```

Then rank candidates lexicographically:

1. preserve safety and avoid XY collateral;
2. reduce currently observable/controllable largest residual axis;
3. reduce combined residual;
4. reduce uncertainty or improve observability;
5. prefer zero/hold if benefit is tied.

Fallback:

- if no non-zero candidate passes constraints, select `zero`, `hold_open`, or
  `reacquire_view` depending on belief state.

### Module 5: Active Reacquire / Probe

Probe/reacquire is an information-gathering action, not a contact or close
action.

Constraints:

- gripper remains open;
- no close authority;
- cannot set handoff ready;
- max step;
- max repeat count;
- max cumulative displacement;
- force/contact guard safe;
- must trigger fresh belief estimate afterward;
- must fall back if uncertainty does not decrease.

Use cases:

- yaw ambiguous but XY/Z safe;
- partial view blocks residual confidence;
- forward model predicts no safe corrective command but predicts that a
  reacquire/probe action improves observability.

## Data Plan

### Dataset A: Belief State Dataset

Build:

```text
runtime_artifacts/coarse2contact_v2/datasets/belief_forward_state_manifest.jsonl
```

Rows from:

- v46 broad/yawbalanced;
- random-heldout traces;
- hard-bucket traces;
- old4/random5/random10 sentinels;
- partial-view / low-visibility windows.

Labels:

```text
pre_residual
axis_observable
axis_controllable
axis_confidence_target
yaw_state
residual_error_bucket
recommended_mode
uses_privileged_runtime=false
```

### Dataset B: Executed Transition Dataset

Build:

```text
runtime_artifacts/coarse2contact_v2/datasets/belief_forward_transition_manifest.jsonl
```

Rows from:

- existing applied-transition manifests;
- existing command-sweep manifests;
- newly collected high-information windows.

Each group must preserve:

```text
source_eval_root
episode_idx
step_idx
same-window group id
zero/no-op candidate
candidate action type
command_local_6d
pre_residual
post_residual
delta_residual
axis worsen labels
observability_before/after
uncertainty_before/after if available
close_leak
uses_privileged_runtime=false
```

### Dataset C: High-Information Command Sweeps

Collect only where the transition is informative:

1. typed16 tied-zero worst roots;
2. typed16 yaw/Z improvement with XY regression;
3. large-XY/large-Yaw flush roots;
4. large-XY/large-Yaw retain roots;
5. small-XY/large-Yaw;
6. partial-view / occlusion;
7. yaw-observable near-contact;
8. yaw-ambiguous near-contact;
9. random/source-held-out planner failure tails.

Every sweep must include same-window zero/no-op.

Target scale:

- high hundreds of same-window groups;
- low thousands to low tens of thousands of executed transitions;
- source-root/session held-out split.

## Evaluation Gates

Keep diagnostic and runtime gates separate.

### Diagnostic Milestones

These can be split by axis and used for training feedback:

- XY beat-zero without XY collateral;
- Z contraction near contact without XY damage;
- yaw permission precision on yaw-observable windows;
- yaw abstain/reacquire precision on ambiguous windows;
- forward-model post-residual calibration;
- probe/reacquire reduces uncertainty or improves observability;
- source-held-out worst-root behavior.

### Runtime Safety Gate

This is not split and not relaxed:

```text
planner close intent
AND strict alignment_ready_for_handoff
AND all required axes ready under the active skill contract
AND no C2C close authority
AND close leak count = 0
```

### Promotion Gate Before MP4 / Insert Claim

Required:

- worse-than-zero folds: `0`;
- worst-root combined minus zero: `> 0`;
- worst-root required-axis minus zero: `> 0`;
- XY contraction not below v42/v46 evidence;
- yaw false-positive rate controlled on ambiguous/unobservable windows;
- dyaw command rate near zero when yaw ambiguous/unobservable, except for
  explicitly bounded open-only probe/reacquire candidates;
- source-held-out split integrity;
- close leak count: `0`;
- canonical front+wrist MP4 only after offline gate is credible.

## Execution Milestones

### Milestone 1: Manifest and Label Builder

Implement scripts:

```text
scripts/build_c2c_v2_belief_forward_state_manifest.py
scripts/build_c2c_v2_belief_forward_transition_manifest.py
scripts/audit_c2c_v2_belief_forward_manifest.py
```

Acceptance:

- manifests build from existing v46/v42/runtime traces;
- source-root split summary is written;
- random/sentinel/hard buckets are tagged;
- privileged labels are marked offline-only;
- same-window zero/no-op coverage is reported.

### Milestone 2: Minimal Belief Model

Implement:

```text
prismatic/robot/coarse2contact_v2/belief_forward_task_frame.py
scripts/train_c2c_v2_belief_forward_state.py
scripts/eval_c2c_v2_belief_forward_state.py
```

Acceptance:

- outputs belief fields for arbitrary active skill axes;
- yaw ambiguity handled as abstain/reacquire target when applicable;
- held-out uncertainty calibration reported;
- XY/Z observable slices do not regress versus v46/v42 diagnostics.

### Milestone 3: Forward Model

Implement:

```text
scripts/train_c2c_v2_belief_forward_transition.py
scripts/eval_c2c_v2_belief_forward_transition.py
```

Acceptance:

- predicts `post_residual_mean/logvar`;
- predicts per-axis worsen;
- same-window zero comparison available;
- ranking losses are auxiliary only;
- LOO/source-held-out report includes worst-root minus-zero metrics.

### Milestone 4: Conservative Offline Search

Implement:

```text
scripts/eval_c2c_v2_belief_forward_search.py
```

Acceptance:

- candidate set includes zero, axis corrections, hold/reacquire/probe;
- hard constraints prevent XY collateral compensation;
- ambiguous yaw blocks dyaw correction;
- typed command context retained;
- worst-root minus-zero is positive before runtime smoke.

### Milestone 5: Runtime Shadow Mode

Add evaluator flags:

```text
--belief_forward_ckpt
--enable_belief_forward_shadow
```

Runtime shadow records:

```text
belief_forward_selected_action
belief_forward_predicted_post_residual
belief_forward_uncertainty
belief_forward_zero_score
belief_forward_constraints_passed
belief_forward_close_control_allowed=false
```

Acceptance:

- no action is applied;
- trace proves no close authority;
- predicted consequences can be compared to actual planner trajectory.

### Milestone 6: Open-Only Intervention Smoke

Add evaluator flag:

```text
--enable_belief_forward_open_only_control
```

Allowed actions:

- bounded correction;
- hold_open;
- reacquire_view;
- tiny_probe.

Forbidden:

- close;
- handoff-ready override;
- contact-risk probe.

Acceptance:

- front+wrist MP4;
- close leak count `0`;
- per-axis residual contraction report;
- insert success is reported but not claimed unless promotion gate passes.

## Hidden Pitfalls To Avoid

- Do not use stage-level diagnostic success to relax close.
- Do not optimize yaw/Z while hurting XY.
- Do not train on privileged runtime input.
- Do not evaluate on only old4/random5 or one hard root.
- Do not let same episode/source appear in train and held-out.
- Do not treat yaw ambiguous windows as regression targets.
- Do not let active probe become contact/close probing.
- Do not claim insert success from MP4 before offline/source-held-out gates pass.
- Do not make `belief_forward_task_frame_candidate` insert-hardcoded; skill
  contracts must carry task-specific frame and symmetry assumptions.

## Final Goal

The final goal is a general, reusable C2C precision takeover layer:

**When the frozen planner reaches a local precision region, C2C estimates a
non-privileged task-frame belief, decides which axes are observable and
controllable, predicts the consequence of bounded open-safe candidate actions,
executes only conservative beat-zero corrections/reacquire/probe actions, and
keeps close blocked until the strict skill contract says handoff is ready.**

For `insert_onto_square_peg`, success means C2C improves random/source-held-out
failure-tail alignment and ultimately increases insert success without close
leaks or privileged runtime shortcuts.  For future high-precision operations,
success means the same belief/forward/control interface can be reused by
changing the precision skill contract rather than rewriting the controller.
