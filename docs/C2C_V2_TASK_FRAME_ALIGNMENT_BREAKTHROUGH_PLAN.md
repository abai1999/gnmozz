# C2C v2 Task-Frame Alignment Breakthrough Plan

This document turns the latest external-style review into the next concrete
project route. The important correction is simple: C2C does not just need more
diagnostics, and it does not only need Z/Yaw after a solved XY stage. The
current system needs a stronger non-privileged task-frame state estimator for
all alignment axes.

## Review Takeaways

Useful points from the review:

- The overall architecture is still right: frozen VLA planner for coarse
  semantic motion, C2C for local precision alignment.
- The current strongest contribution is the contract, trace, privilege
  boundary, and strict close safety scaffold. It is not yet proven high
  precision takeover.
- The main bottleneck is state semantics, not gain. Mask centroids, PCA yaw,
  median depth, scalar readiness features, and hard-bucket replay scores are
  evidence proxies. They are not yet a reliable task-frame residual.
- XY is not solved. `v42_expanded_v4pilot` is the current best baseline, but it
  can still fail under partial view, occlusion, approach-angle changes, proxy
  sign errors, and random held-out tails.
- Z is not a world-height threshold. It must become a task-frame approach-axis
  residual and descend-progress estimate.
- Yaw is the deepest observability problem because the ring/peg geometry is
  symmetric, often partially visible, and vulnerable to alias drift.
- The learned localizer scaffold in `learned_localizer.py` is closer to the
  right direction than continued heuristic proxy tuning, but it is not yet the
  runtime mainline.

Current-branch correction to the review:

- Close ownership has already been tightened in this branch. The active
  semantics are planner-owned close plus C2C open-only safety. C2C may monitor
  contact or emit a close recommendation for trace/debug, but final close must
  come from planner intent and strict `alignment_ready_for_handoff`.
- Therefore the next main work is not another close audit. We should keep the
  close arbiter invariant tested while moving effort to alignment capability.

## Clear Target

Build and validate `v46_unified_task_frame_alignment_candidate`:

**A non-privileged spatial-temporal task-frame estimator and bounded controller
that improves XY, Z, and Yaw alignment in parallel, produces calibrated
per-axis observability/confidence/ambiguity, and reduces true task-frame
residuals on random held-out insert trajectories without loosening strict
handoff or close authority.**

Success means C2C can show closed-loop task-frame residual contraction on
random held-out failures, not just a better audit score or a nicer MP4.

## Phase Objective

The next concrete objective is:

**Turn v46 from a runnable scaffold into a validated three-axis alignment
candidate that beats v42/v45 on held-out random failure tails without changing
the close/handoff contract.**

Minimum promotion gate:

- train on multi-source, source-root/session held-out data, not only old4,
  random5, random10, or one hardmix source root
- show positive held-out contraction for XY, Z, and Yaw individually
- reduce combined task-frame residual on random held-out failure tails
- keep worst-slice worsen and overshoot no worse than v42/v45
- demonstrate calibrated yaw ambiguity: ambiguous yaw should block yaw servo,
  not silently become a wrong yaw step
- keep planner close leak count at zero
- generate MP4 with wrist view only after the offline/replay gate is meaningful,
  and ship trace summaries next to the videos

Until this gate passes, `v46_unified_task_frame_alignment_candidate` remains a
candidate, not a baseline.

## Non-Negotiable Constraints

- Runtime environment: `conda run -n vla-adapter ...`
- Fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- Current baseline to beat:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- Runtime input must remain non-privileged:
  no RLBench object handles, GT target poses, success poses, privileged masks,
  or teacher residuals in the control loop.
- Privileged residuals are allowed only for offline labels, relabeling, audit,
  sidecar evaluation, and acceptance analysis.
- `alignment_ready_for_handoff` remains the only close/handoff predicate.
- C2C does not output close authority.
- Axis correction must not imply handoff. A useful XY/Z/Yaw step can happen
  while `alignment_ready_for_handoff=false`.

## Why The Current v42/v45 Stack Is Not Enough

`v42_expanded_v4pilot` should remain the active XY baseline because it is the
best measured candidate so far. It should not be described as solved XY.

Known XY limitations:

- weak generalization under occlusion and partial view
- proxy sign errors when mask geometry is distorted
- possible worsen tails when the local visual proxy and true task-frame
  residual disagree
- activation-window and smoke-eval plumbing that must be interpreted carefully
  when privileged probe fields are present

`v45` is a useful scaffold for Z/Yaw control plumbing, but not the final
capability:

- it mainly uses scalar readiness/proxy features
- it does not see rich wrist RGBD geometry directly
- yaw remains mostly blocked by unobservable/ambiguous states
- tiny dz/dyaw steps are safety evidence, not proof of high-precision state
  estimation

The next step must replace proxy-first correction with task-frame state
estimation.

## v46 Model Route

### Inputs

Use a short temporal window, default `K=6`, with non-privileged inputs only:

- wrist RGBD crop centered by planner/ROI prior
- depth validity and normalized depth patch
- coordinate channels
- optional non-privileged heatmaps or masks derived from runtime-visible RGBD
- gripper proprio and jaw/open state
- planner local delta and recent planner motion
- previous C2C local steps
- previous estimated residual/confidence/visibility
- force/contact safety summary for guard decisions, not privileged geometry

The model should use the existing `learned_localizer.py` scaffold where useful,
but it should be upgraded from single-frame or scalar-only prediction to a
spatial-temporal task-frame estimator.

### Outputs

The model should output state, not close:

- `dx`, `dy`, `dz`, `dyaw`
- per-axis confidence: `xy_confidence`, `z_confidence`, `yaw_confidence`
- per-axis observability: `xy_observable`, `z_observable`, `yaw_observable`
- yaw ambiguity state: `yaw_ambiguous`, `yaw_unobservable`
- multi-hypothesis yaw modes for square symmetry
- per-axis step scale: `xy_step_scale`, `z_step_scale`, `yaw_step_scale`
- risk reason logits: `normal`, `low_visibility`, `direction_conflict`,
  `insufficient_support`, `occlusion`, `force_guard`, `alias_drift`

### Yaw Representation

Yaw should not be a single PCA/image-axis regression target.

Use a symmetry-aware representation:

- predict multiple yaw hypotheses modulo square symmetry
- train with min-over-symmetry or distributional loss over equivalent modes
- estimate `yaw_observable`, `yaw_ambiguous`, and `yaw_unobservable` separately
- use temporal consistency and planner prior only to disambiguate among
  hypotheses, not as a substitute for visual evidence

Yaw control may only use a selected hypothesis when the hypothesis gap,
temporal stability, and non-ambiguity checks pass.

## Data Plan

The training set must be large enough and diverse enough to make
episode-specific patching unattractive.

Collect or consolidate:

- planner-only near-failure and failure-tail rollouts
- `v42` XY rollout traces
- `v45` axes-softgate rollout traces
- random held-out tails beyond old4/random5/random10
- hard-bucket active rows
- occlusion and partial-view rows
- C2C worsen tails where an attempted correction increased residual
- near-positive rows where the system almost reaches handoff but one axis
  blocks

Every row should record:

- `source_eval_root`
- `session_id` or source-root id
- `episode_idx`
- `step_idx`
- `stage`
- `trace_path`
- `observability_bucket`
- `failure_bucket`
- `runtime_input_schema`
- `uses_privileged_runtime=false`
- offline-only labels for true `dx/dy/dz/dyaw`, contraction, overshoot,
  yaw symmetry/ambiguity, and handoff readiness

Split policy:

- split by source root/session first, then episode
- keep `random10_generalization` and a larger random holdout pool out of
  training
- keep old4/random5/ep25/26 as sentinel slices, not optimization targets
- report worst-slice metrics before overall averages

## Loss Plan

Train for runtime control, not offline MAE.

Loss terms:

- direction/sign loss for `dx/dy/dz`
- symmetry-aware multi-hypothesis yaw loss
- bounded-step contraction loss after applying the actual controller clamp
- worsen penalty when predicted step increases residual
- overshoot penalty
- reverse/oscillation penalty over temporal windows
- confidence calibration loss
- observability and ambiguity classification losses
- multi-axis consistency loss for steps that should jointly reduce the
  task-frame residual

The model should be selected by held-out contraction and worsen behavior, not
by pooled MAE.

## Runtime Controller Plan

Replace serial readiness with parallel axis evaluation:

1. Estimate all axes each frame.
2. For each axis, decide whether evidence is strong, weak, conflicting, or
   absent.
3. Apply bounded correction per axis:
   - strong evidence: normal bounded step
   - weak evidence: lower step scale
   - conflicting evidence: tiny step or hold plus reacquire marker
   - absent evidence: no blind correction unless a recent stable history
     supports a tiny step
4. Track worsen/overshoot online.
5. If residual worsens for consecutive steps, reduce step scale or switch to a
   different hypothesis/reacquire behavior.
6. Never use any axis correction to set close/handoff directly.

Default safety shape:

- XY correction remains bounded by the v42-style clamp.
- Z correction is along the task approach axis with force guard.
- Yaw correction is tiny and requires a non-ambiguous, stable hypothesis.
- Close remains blocked unless the strict lifecycle says
  `alignment_ready_for_handoff=true`.

## Evaluation Plan

Offline A/B baselines:

- planner-only
- `v42_expanded_v4pilot`
- `v45` axes softgate
- `v46_unified_task_frame_alignment_candidate`

Slices:

- old4
- random5
- random10
- larger random holdout pool
- hard-bucket active rows
- low visibility
- partial view
- occlusion
- ep25/26 sentinel
- C2C worsen tails
- near-positive handoff windows

Primary metrics:

- XY contraction, worsen, overshoot, reverse
- Z contraction and force-guard violation count
- yaw sign/hypothesis match, ambiguity calibration, yaw worsen
- combined task-frame residual contraction
- near-entry rate
- handoff false positive and false negative
- planner close leak count
- final grasp/insert success after strict handoff

MP4 policy:

- MP4 is required for inspection but cannot promote a model alone.
- Videos must include wrist view.
- Select clips from worst slices and representative random held-out episodes,
  not only successful-looking examples.
- Every MP4 batch must ship with trace summaries of per-axis steps, block
  reasons, and gripper authority.

## Implementation Milestones

### Milestone 0: Keep Safety Fixed

- Preserve `planner_gripper_authority_decision`.
- Keep evaluator close blocker enabled by default.
- Add/keep tests proving C2C recommendations cannot close without planner
  intent plus strict handoff.

Deliverable: no close safety regression while alignment work proceeds.

Current implementation status:

- `v46_unified_task_frame_alignment_candidate` has a runnable checkpoint,
  manifest, training, and evaluator scaffold.
- `scripts/build_c2c_v2_task_frame_v46_manifest.py` consolidates
  `frame_residual_v2` rows into a v46 manifest while requiring non-privileged
  runtime observation pointers and filtering out coarse rows outside the local
  support radius.
- `scripts/train_c2c_v2_task_frame_v46_alignment.py` supports source-root
  held-out training and uses direction/sign-aware, bounded-step-aware losses.
- A multisource smoke manifest from existing relabel artifacts produced
  448 retained local-support rows across 7 source roots and 25 episodes.
- A CPU root-held-out sign-loss smoke run is directionally promising for XY/Z
  but not a promotion result:
  - validation XY sign match: `0.9297`
  - validation Z sign match: `0.8438`
  - validation Yaw sign match: `0.6406`
  - validation XY bounded-step contraction: `0.9219`
  - validation Z bounded-step contraction: `0.7813`
  - validation Yaw bounded-step contraction: `0.6094`
  - validation combined bounded-step contraction: `0.6094`
- This confirms the pipeline and loss direction, but it does not satisfy the
  full gate. The dataset is still smoke-scale, visual-observable-heavy, and
  not a random held-out insert-success proof.

### Milestone 1: Build v46 Dataset

- Build a spatial-temporal dataset from runtime observations and traces.
- Include wrist RGBD crops, depth-valid masks, proprio, planner priors,
  history, and offline labels.
- Add leakage tests for source-root/session held-out split.

Deliverable: `task_frame_alignment_v46_*.jsonl` or equivalent manifest plus
paired tensor/npz dataset.

### Milestone 2: Train v46 State Estimator

- Implement a spatial-temporal model that predicts XY/Z/Yaw residuals,
  confidence, observability, yaw ambiguity, and step scales.
- Include symmetry-aware yaw hypotheses.
- Train with bounded-step control-aware loss.

Deliverable:
`runtime_task_frame_alignment_v46_unified_candidate.pt`.

### Milestone 3: Runtime Wrapper And Controller

- Add a wrapper that returns a structured task-frame alignment estimate.
- Compose per-axis bounded steps in parallel.
- Keep `uses_privileged_runtime=false` in trace.
- Record all per-axis decisions and risk reasons.

Deliverable: v46 can run in evaluator with strict close unchanged.

### Milestone 4: Generalization Gate

- Run fixed A/B on old4, random5, random10, hard bucket, and a larger random
  holdout pool.
- Require worst-slice improvement or no-regression before any baseline change.
- Generate wrist-view MP4 only after offline gate is meaningful.

Deliverable: promotion decision: `pending`, `rejected`, or `new baseline`.

Current smoke artifacts:

- Manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_multisource_smoke_manifest.jsonl`
- Manifest summary:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_multisource_smoke_manifest.summary.json`
- Sign-loss smoke checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_multisource_signloss_smoke.pt`
- Sign-loss smoke report:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_multisource_signloss_smoke.json`
- Sign-loss smoke eval report:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_multisource_signloss_smoke_eval.json`
- Yaw-balanced checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_yawbalanced.pt`
- Yaw-balanced train report:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawbalanced_train.json`
- Yaw-balanced full-manifest eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawbalanced_eval.json`
- Random5 strict holdout manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_holdout_random5_v45_strict_manifest.jsonl`
- Random5 strict holdout eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawbalanced_holdout_random5_v45_strict_eval.json`
- Yaw-observability-gated checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_yawobs_confcal_yawweighted.pt`
- Yaw-observability-gated train report:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawobs_confcal_yawweighted_train.json`
- Yaw-observability-gated full eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawobs_confcal_yawweighted_eval.json`
- Yaw-observability-gated random5 strict holdout eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawobs_confcal_yawweighted_holdout_random5_v45_strict_eval.json`

Current stronger evidence:

- The v46 yaw-balanced manifest has 7193 rows across 56 source roots and 30
  episodes. It includes more yaw supervision than the first smoke manifest,
  but yaw-observable local-support rows are still scarce.
- Root-held-out train/val on the yaw-balanced manifest improved the held-out
  validation metrics:
  - validation XY bounded-step contraction: `0.8884`
  - validation Z bounded-step contraction: `0.8107`
  - validation Yaw bounded-step contraction: `0.8206`
  - validation combined bounded-step contraction: `0.8242`
  - validation combined worsen: `0.1758`
- On the separate random5 strict holdout, v46 shows positive three-axis
  contraction but does not pass the worst-slice gate:
  - overall XY bounded-step contraction: `0.9870`
  - overall Z bounded-step contraction: `0.8312`
  - overall Yaw bounded-step contraction: `0.6512`
  - overall combined bounded-step contraction: `0.6735`
  - overall combined worsen: `0.3265`
  - worst episode: `ep024`, combined contraction `0.2167`, yaw contraction
    `0.1750`, worsen `0.7833`
  - Interpretation: v46 is now a real candidate with random held-out positive
  signal, especially for XY/Z. It is not yet a baseline because yaw and
  worst-episode behavior remain too weak, and no closed-loop insert success
  improvement has been proven.
- A subsequent audit found the main ep023/024 failure cause: rows marked
  `yaw_observability_class=unobservable` were being treated as
  `yaw_observable=true` by the v46 label helper, so the model learned to move
  yaw in frames where yaw should be blocked.
- The yaw-observability-gated candidate fixes that semantic error:
  - random5 strict holdout combined contraction: `1.0000`
  - random5 strict holdout combined worsen: `0.0000`
  - random5 strict holdout XY contraction: `0.9722`
  - random5 strict holdout Z contraction: `0.8312`
  - random5 strict holdout Yaw contraction: `0.0000`, because this holdout has
    no yaw-observable rows and yaw is correctly blocked
  - full yaw-balanced manifest yaw-observable slice yaw contraction: `0.9726`
  - full yaw-balanced manifest yaw-observable slice yaw worsen: `0.0274`
  - full yaw-balanced manifest unobservable-yaw slice yaw worsen: `0.0000`
- Interpretation update: v46 now has a cleaner safety/capability split. It can
  contract yaw when yaw is labeled observable, and it blocks yaw without
  worsening when yaw is unobservable. The remaining proof gap is closed-loop
  runtime/MP4 and insert success, plus a larger truly random holdout pool with
  yaw-observable cases.

Reproduction commands:

```bash
conda run -n vla-adapter python scripts/build_c2c_v2_task_frame_v46_manifest.py \
  --input \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation_v17_large_support \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_focus_window_hard_ep6_12_14_16 \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_support_expanded_gap \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_small_xy_frontier \
  --output_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_multisource_smoke_manifest.jsonl \
  --summary_json runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_multisource_smoke_manifest.summary.json \
  --max_rows_per_source 64

conda run -n vla-adapter python scripts/train_c2c_v2_task_frame_v46_alignment.py \
  --dataset_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_multisource_smoke_manifest.jsonl \
  --output_checkpoint runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_multisource_signloss_smoke.pt \
  --output_json runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_multisource_signloss_smoke.json \
  --split_mode root \
  --epochs 20 \
  --batch_size 64 \
  --image_hidden_dim 48 \
  --fusion_hidden_dim 48 \
  --history_window_size 2 \
  --image_resize_size 32 \
  --device cpu

conda run -n vla-adapter python scripts/eval_c2c_v2_task_frame_v46_alignment.py \
  --dataset_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_multisource_smoke_manifest.jsonl \
  --checkpoint runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_multisource_signloss_smoke.pt \
  --output_json runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_multisource_signloss_smoke_eval.json \
  --image_resize_size 32 \
  --history_window_size 2 \
  --device cpu
```

Yaw-balanced training/eval commands:

```bash
conda run -n vla-adapter python scripts/build_c2c_v2_task_frame_v46_manifest.py \
  --input \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation_rerun \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation_v14_small_xy_large_yaw_xyready \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation_v14_large_xy_large_yaw_xyready \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation_v16_alias_clean \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation_v17_large_support \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_focus_window_hard_ep6_12_14_16 \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_frontier \
    runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_support_expanded_gap \
  --output_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_balanced_manifest.jsonl \
  --summary_json runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_balanced_manifest.summary.json \
  --max_rows_per_source_yaw_class 64

conda run -n vla-adapter python scripts/train_c2c_v2_task_frame_v46_alignment.py \
  --dataset_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_balanced_manifest.jsonl \
  --output_checkpoint runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_yawbalanced.pt \
  --output_json runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawbalanced_train.json \
  --split_mode root \
  --epochs 35 \
  --batch_size 256 \
  --image_hidden_dim 96 \
  --fusion_hidden_dim 128 \
  --history_window_size 4 \
  --image_resize_size 48 \
  --device cuda

conda run -n vla-adapter python scripts/eval_c2c_v2_task_frame_v46_alignment.py \
  --dataset_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_holdout_random5_v45_strict_manifest.jsonl \
  --checkpoint runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_yawbalanced.pt \
  --output_json runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawbalanced_holdout_random5_v45_strict_eval.json \
  --image_resize_size 48 \
  --history_window_size 4 \
  --device cuda
```

Remaining gate gaps:

- Build a full-scale manifest, not a smoke manifest capped at 64 rows/source.
- Include random held-out failure tails and low-visibility/partial/occlusion
  rows; the current smoke manifest is visual-observable-heavy.
- Train on GPU with enough capacity and a true source/session held-out split.
- Add an offline A/B report against planner-only, v42, and v45 on fixed random
  held-out slices.
- Improve yaw modeling. The current smoke eval shows XY/Z one-step contraction
  around `0.81`, and the yaw-balanced candidate improves yaw substantially.
  The subsequent yaw-observability-gated candidate fixes the ep023/024
  unobservable-yaw worsen failure. The remaining issue is no longer average
  offline yaw contraction; it is proving closed-loop runtime improvement and
  collecting a larger random holdout pool that includes yaw-observable cases.
- Only after offline/replay metrics pass, run 150/180-step MP4 with wrist view
  and strict close unchanged.

## When To Use RL Or Force/Tactile

Do not use RL to replace pre-contact task-frame state estimation.

RL becomes appropriate after v46 can reliably enter a near-contact or
micro-entry region. The right boundary is:

- contact-rich micro-entry
- jam recovery
- guarded slide
- post-handoff insertion recovery

Force/tactile should first be used as safety and contact/recovery evidence. It
can later become a critic or recovery signal, but it should not be presented as
the current source of pre-contact geometric alignment unless that is actually
implemented and validated.

## Decision Rule

Do not declare C2C solved because:

- a smoke MP4 looks better
- pooled offline MAE improves
- one old/random slice improves
- close finally happens after threshold relaxation

Declare progress only when:

- runtime inputs are non-privileged
- per-axis residuals contract on held-out random slices
- worsen and overshoot do not increase on worst slices
- yaw ambiguity is calibrated, not ignored
- strict handoff blocks close until XY/Z/Yaw/frame readiness are all valid
- closed-loop insert success improves under the fixed planner checkpoint

## 2026-06-06 v46 Implementation Update

Implemented scaffold:

- Added `v46_unified_task_frame_alignment_candidate` in
  `prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py`.
- Added manifest build, training, and offline eval scripts:
  - `scripts/build_c2c_v2_task_frame_v46_manifest.py`
  - `scripts/train_c2c_v2_task_frame_v46_alignment.py`
  - `scripts/eval_c2c_v2_task_frame_v46_alignment.py`
- Runtime evaluator now loads `--task_frame_v46_ckpt` and can run
  `--enable_v46_task_frame_micro_servo`.
- v46 outputs non-privileged `dx/dy/dz/dyaw`, per-axis confidence and
  observability, yaw ambiguity/unobservability, step scale, and risk reason.
- v46 does not output gripper close authority. Trace keeps
  `task_frame_v46_close_control_allowed=false`.
- Added correction-only activation fields:
  `task_frame_v46_activation_ready`, `task_frame_v46_activation_reason`,
  `task_frame_v46_estimated_xy_norm`, and `task_frame_v46_estimated_z_abs`.
- Added runtime XY preservation mode:
  `--v46_task_frame_xy_mode hybrid_v42_preferred`. This keeps v42 runtime-XY
  as the preferred XY controller while using v46 for Z/Yaw, unless explicitly
  running pure `unified_v46` diagnostics.
- Added risk-aware step scaling. `direction_conflict`, low visibility,
  insufficient support, occlusion, and alias drift reduce bounded step size
  rather than silently taking a full step.

Validation completed:

```bash
conda run -n vla-adapter python -m py_compile \
  prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py \
  scripts/evaluate_c2c_v2_rlbench.py \
  scripts/build_c2c_v2_task_frame_v46_manifest.py \
  scripts/train_c2c_v2_task_frame_v46_alignment.py \
  scripts/eval_c2c_v2_task_frame_v46_alignment.py

conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py
# 181 passed
```

Best offline checkpoint from this implementation round:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_yawobs_confcal_yawweighted.pt
```

Offline held-out evidence:

- random5 strict holdout eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawobs_confcal_yawweighted_holdout_random5_v45_strict_eval.json`
- result: combined contraction `1.0000`, XY contraction `0.9722`,
  Z contraction `0.8312`, yaw correctly blocked because this holdout has no
  yaw-observable rows.
- yaw-balanced eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_yawobs_confcal_yawweighted_eval.json`
- result: overall combined contraction `0.9619`, yaw-observable slice yaw
  contraction `0.9726`, unobservable yaw slice yaw worsen `0.0`.

Runtime MP4 smoke evidence:

- pure v46 runtime-style random5 smoke:
  `runtime_artifacts/coarse2contact_v2/v46_unified_runtime_smoke_random5`
- MP4s:
  - `videos/ep023_fail.mp4`
  - `videos/ep024_fail.mp4`
  - `videos/ep025_fail.mp4`
  - `videos/ep026_fail.mp4`
  - `videos/ep027_fail.mp4`
- result: v46 actually activated/applied on `384` eval-label rows, close leak
  count `0`, Z contraction `0.8333`, combined contraction `0.9010`, but XY
  contraction only `0.6510` and XY worsen `0.3490`; yaw was blocked as
  unobservable.

Risk-scaled pure v46 runtime smoke:

- path:
  `runtime_artifacts/coarse2contact_v2/v46_unified_runtime_smoke_random5_riskscaled`
- result: close leak count `0`, Z contraction `1.0000`, combined contraction
  `0.9125`, but XY contraction fell to `0.5455`. This shows the XY issue is
  not just step size; the v46 XY residual semantics are not yet robust.

Hybrid v42-XY + v46-Z/Yaw runtime smoke:

- path:
  `runtime_artifacts/coarse2contact_v2/v46_unified_runtime_smoke_random5_hybridxy`
- result: close leak count `0`, but combined contraction dropped to `0.5896`
  because many windows became Z-only while XY still drifted. This is useful
  diagnostic evidence, not a promotion result.

Current conclusion:

- v46 is implemented as a real non-privileged spatial-temporal task-frame
  estimator/controller scaffold.
- v46 can now truly activate in runtime traces; earlier false MP4 comparisons
  caused by inactive C2C windows are no longer acceptable evidence.
- strict close ownership remains intact in these smokes: no C2C direct close
  authority and no close leaks were observed.
- v46 is not a new baseline. It has not proven stable three-axis task-frame
  residual contraction on random held-out runtime failure tails, and insert
  success remains `0/5` on the random5 smoke.

Next target:

Train a stronger v46/v47 task-frame estimator that fixes XY residual semantics
instead of relying on risk scaling, and collect yaw-observable runtime windows
so yaw can be validated as an actual control axis rather than only blocked.

## 2026-06-06 Continuation: Transition Feedback Audit

Additional progress after the first v46 smoke:

- Relabeled the pure v46 random5 runtime smoke with privileged offline labels:
  `runtime_artifacts/coarse2contact_v2/relabels/v46_unified_runtime_smoke_random5_frame_residual_v2/frame_residual_v2.jsonl`.
- Built an online-feedback manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_online_feedback_random5_manifest.jsonl`
  with `698` retained rows.
- Extended `scripts/train_c2c_v2_task_frame_v46_alignment.py` to carry
  optional `next_residual` / `has_next_residual` arrays and add a
  transition-aware loss when `next_privileged_dx/dy/dz/dyaw` is present.
- Trained a diagnostic transition-feedback checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_transition_feedback_candidate.pt`.
- Added a closed-loop control-effect audit:
  `scripts/audit_c2c_v2_task_frame_control_effect.py`.

Diagnostic result:

- The transition-feedback candidate did not fix the online-feedback XY failure.
  On `task_frame_alignment_v46_online_feedback_random5_manifest.jsonl`, it had
  `xy_sign_match=0.4828` and `xy_bounded_step_contraction=0.1791`.
- The control-effect audit shows the deeper issue: runtime local command to
  true task-frame residual delta is not the idealized `post = residual - step`
  used by the offline bounded-step loss.
- Example from pure v46 ep024:
  - `delta_command_corr` for y: `-0.9485`
  - y same-sign rate between command and residual delta: `0.0`
  - xy contraction rate: `0.1020`
- Across random5, y-axis command/true-delta correlation is often strongly
  negative, and the empirical XY response matrix has cross-axis terms. This
  means the controller needs a learned or calibrated task-frame control-effect
  model/Jacobian, not just a better residual regressor or smaller step scale.

Updated next target:

Build `v47_control_effect_task_frame_candidate`: keep v46's non-privileged
state estimator, but add a runtime-visible control-effect/Jacobian head trained
from offline-labeled transitions. The controller should choose bounded local
steps by minimizing predicted post-step task-frame residual under this learned
effect model, while preserving strict handoff and planner-owned close.

## 2026-06-06 Continuation: Control-Effect Runtime Check

Implemented the first control-effect extension on top of v46:

- `TaskFrameV46AlignmentNet` now has an `xy_control_effect_head` that predicts a
  local-XY-command to task-frame-XY-residual-delta matrix.
- `TaskFrameV46AlignmentEstimate` records
  `task_frame_v46_xy_control_effect` in trace output.
- Runtime evaluator supports `--v46_task_frame_xy_mode effect_aware`.
- `task_frame_v46_effect_aware_xy_correction(...)` computes a bounded XY
  correction from the predicted control-effect matrix.
- The effect-aware path still has no close authority. It only writes bounded
  local XY correction; planner close remains strict-handoff owned.
- The effect-aware helper now obeys the same evidence/risk softgate as direct
  v46 XY. Direction-conflict or insufficient-support frames cannot bypass risk
  scaling and become full-step XY control.

Verification:

```bash
conda run -n vla-adapter python -m py_compile \
  prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py \
  scripts/evaluate_c2c_v2_rlbench.py

conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py
# 183 passed
```

Offline control-effect checkpoint:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v47_control_effect_bounded_candidate.pt
```

Offline diagnostic evidence was promising but not sufficient:

- On online-feedback data, direct bounded XY contraction remained poor
  (`0.1805`), but effect-aware offline contraction reached `0.8840`.
- On the strict holdout eval, effect-aware XY contraction reached `1.0000`.
- This justified a focused runtime smoke, not a baseline promotion.

Focused runtime smoke on ep024:

- Unguarded effect-aware run:
  `runtime_artifacts/coarse2contact_v2/v47_control_effect_runtime_smoke_ep024`
- MP4:
  `runtime_artifacts/coarse2contact_v2/v47_control_effect_runtime_smoke_ep024/videos/ep024_fail.mp4`
- Trace:
  `runtime_artifacts/coarse2contact_v2/v47_control_effect_runtime_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl`
- Result: `40` applied eval-label rows, close leak count `0`, Z contraction
  `1.0000`, combined contraction `0.7750`, but XY contraction only `0.2000`
  with XY worsen `0.8000`.
- Diagnosis: online v47 residual semantics failed on this window. The model
  estimated `x` with the wrong sign on most applied rows and saturated `dy` at
  `+0.04`; effect-aware control then optimized against the wrong residual.

Guarded effect-aware rerun:

- Path:
  `runtime_artifacts/coarse2contact_v2/v47_control_effect_runtime_smoke_ep024_guarded`
- MP4:
  `runtime_artifacts/coarse2contact_v2/v47_control_effect_runtime_smoke_ep024_guarded/videos/ep024_fail.mp4`
- Result: `125` applied eval-label rows, close leak count `0`, XY contraction
  `0.4400`, XY worsen `0.5600`, mean XY norm delta `-0.000036`, Z contraction
  `0.6480`, yaw allowed rows `0`.
- Interpretation: risk softgating removed the full-step conflict behavior and
  made XY roughly neutral instead of strongly harmful, but it did not create a
  reliable three-axis aligner. Yaw still lacks observable runtime rows in this
  slice.

Current promotion decision:

- Do not promote v47/effect-aware to baseline.
- Do not run a broad random5 promotion smoke as a success claim from this
  checkpoint; ep024 already fails the focused gate.
- Keep the code as a diagnostic candidate and safety-improved scaffold.
- The next model step must improve non-privileged task-frame residual semantics
  before the control-effect head can help. In particular, the model needs
  better object/task-frame representation, yaw-observable data, and transition
  labels from diverse held-out failure tails rather than relying on scalar
  proxy residuals or idealized bounded-step assumptions.

## 2026-06-06 Continuation: Residual Range And Runtime-Tail Check

Implementation fix retained:

- The v46/v47 checkpoint schema now saves and reloads residual output support:
  `max_abs_xy`, `max_abs_z`, and `max_abs_yaw`.
- Training now constructs `TaskFrameV46AlignmentNet` with the same residual
  support used by the label filter. This prevents a silent mismatch where
  training admits `8cm` XY labels but the model can only output `4cm`.
- Offline metrics now report both:
  - `xy_effect_aware_contraction`, which uses oracle target residual for the
    effect-aware solver and can be over-optimistic.
  - `xy_predicted_effect_aware_contraction`, which uses the model-predicted
    residual and is closer to runtime.
- Unit coverage now includes residual-output-support checkpoint roundtrip.

Verification:

```bash
conda run -n vla-adapter python -m py_compile \
  prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py \
  scripts/train_c2c_v2_task_frame_v46_alignment.py \
  scripts/eval_c2c_v2_task_frame_v46_alignment.py \
  scripts/evaluate_c2c_v2_rlbench.py

conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py
# 184 passed
```

Range-matched local-support candidate:

- Checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v48_range_matched_candidate.pt`
- Training support: `max_abs_xy=0.080`, `max_abs_z=0.080`,
  `max_abs_yaw=0.350`.
- Important diagnostic: the online-feedback random5 manifest has no rows
  inside this local support because its failure-tail residuals are much larger
  (`xy` median about `0.113m`, `z` p90 about `0.485m`).
- v48 therefore trained effectively on the yaw-balanced/local-support data,
  not on online-feedback runtime tails.
- Strict random5 holdout:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v48_range_matched_candidate_holdout_random5_strict_eval.json`
- Result: `xy_predicted_effect_aware_contraction=0.8942`,
  `xy_sign_match=0.8905`, `z_bounded_step_contraction=0.8998`, but this is
  still local-support evidence, not random tail recovery.

Runtime-tail-range candidate:

- Checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v49_runtime_tail_range_candidate.pt`
- Training support: `max_abs_xy=0.400`, `max_abs_z=0.700`,
  `max_abs_yaw=0.450`, so online-feedback rows are included.
- Online-feedback eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v49_runtime_tail_range_candidate_online_feedback_eval.json`
- Result: `698` eval rows,
  `xy_predicted_effect_aware_contraction=1.0000`,
  `xy_sign_match=0.9993`, `z_bounded_step_contraction=0.9799`.
- Strict random5 holdout:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v49_runtime_tail_range_candidate_holdout_random5_strict_eval.json`
- Result: `xy_predicted_effect_aware_contraction=0.9054`,
  `xy_sign_match=0.9072`, `z_bounded_step_contraction=0.8905`, but yaw
  degraded (`yaw_sign_match=0.3729`).

Focused runtime check:

- Path:
  `runtime_artifacts/coarse2contact_v2/v49_runtime_tail_range_smoke_ep024`
- MP4:
  `runtime_artifacts/coarse2contact_v2/v49_runtime_tail_range_smoke_ep024/videos/ep024_fail.mp4`
- Trace:
  `runtime_artifacts/coarse2contact_v2/v49_runtime_tail_range_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl`
- Result: close leak count `0`; planner requested close `28` times and all
  were blocked by strict handoff.
- v49 only applied on `9` eval-label rows. XY contraction was `0.5556`, XY
  worsen `0.4444`, but mean XY norm worsened by `0.00374m`; combined
  contraction was `0.4444`.
- Yaw remained blocked (`yaw_allowed=0`).

Current interpretation:

- The residual-output-support fix is necessary and should stay.
- v48/v49 are not baselines. v49 improves offline online-feedback metrics, but
  the focused ep024 runtime check still fails the residual-contraction gate.
- The gap is no longer just output saturation. The estimator still lacks a
  robust object/task-frame representation and source-held-out runtime-tail
  generalization.
- The next cycle should build a genuinely held-out random-tail dataset with
  multiple source roots, include yaw-observable slices, and train/evaluate with
  predicted-runtime effect metrics as a gate before any MP4 promotion run.

## 2026-06-06 Continuation: Multisource Tail And Near-Field Gate

Built a larger balanced runtime-tail manifest:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v50_runtime_tail_multisource_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v50_runtime_tail_multisource_manifest.summary.json
```

Manifest summary:

- `25,454` retained rows
- `191` source eval roots
- `30` episodes
- failure buckets:
  - `large_xy_large_yaw`: `5,254`
  - `large_xy_small_yaw`: `7,196`
  - `small_xy_large_yaw`: `4,476`
  - `small_xy_small_yaw`: `8,528`
- yaw observability:
  - observable: `3,790`
  - ambiguous: `11,401`
  - unobservable: `10,263`

Trained root-held-out multisource candidate:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v50_multisource_tail_candidate.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v50_multisource_tail_candidate_train.json
```

Root-held-out split:

- train: `20,294` rows from `153` roots
- validation: `5,160` rows from `38` held-out roots

v50 held-out validation:

- `xy_sign_match=0.9338`
- `z_sign_match=0.9618`
- `yaw_sign_match=0.8147`
- `xy_predicted_effect_aware_contraction=0.8905`
- `z_bounded_step_contraction=0.9498`
- `bounded_step_contraction=0.9548`
- yaw remains weak: `yaw_bounded_step_contraction=0.0787`

v50 sentinel eval:

- strict random5 holdout:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v50_multisource_tail_candidate_holdout_random5_strict_eval.json`
  - `xy_predicted_effect_aware_contraction=0.8998`
  - `xy_sign_match=0.9026`
  - `z_bounded_step_contraction=0.9462`
  - `yaw_sign_match=0.8738`
- online-feedback random5:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v50_multisource_tail_candidate_online_feedback_eval.json`
  - `xy_predicted_effect_aware_contraction=0.8037`
  - `xy_sign_match=0.7679`
  - `z_bounded_step_contraction=0.8582`

v50 focused runtime ep024:

```text
runtime_artifacts/coarse2contact_v2/v50_multisource_tail_smoke_ep024/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v50_multisource_tail_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl
```

- close leak count: `0`
- planner close requested `27` times, all blocked by strict handoff
- v50 applied on `0` eval-label rows
- reason: model estimated `xy_norm` far outside the correction activation
  radius even when privileged eval labels showed the true XY residual was
  already small.

Interpretation:

- v50 is the first meaningful multisource, root-held-out runtime-tail
  candidate, but it still fails runtime activation on ep024.
- The failure mode is now sharper: large-range residual heads can improve
  source-held-out offline direction metrics while losing near-field magnitude
  calibration needed to open bounded correction.

Implemented a learned near-field gate:

- Added `near_field_head` to `TaskFrameV46AlignmentNet`.
- Trace now records:
  - `task_frame_v46_near_field_confidence`
  - `task_frame_v46_near_field_head_available`
  - `task_frame_v46_learned_near_field_ready`
  - `task_frame_v46_radius_ready`
- Runtime activation can use learned near-field readiness for correction
  only. It does not affect strict handoff or close.
- Near-field label radii are now train/eval parameters:
  `--near_field_xy_radius` and `--near_field_z_radius`.

Verification:

```bash
conda run -n vla-adapter python -m py_compile \
  prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py \
  scripts/train_c2c_v2_task_frame_v46_alignment.py \
  scripts/eval_c2c_v2_task_frame_v46_alignment.py \
  scripts/evaluate_c2c_v2_rlbench.py

conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py
# 184 passed
```

Trained near-field candidate:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v51_nearfield_multisource_tail_candidate.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v51_nearfield_multisource_tail_candidate_train.json
```

v51 root-held-out validation:

- near-field accuracy `0.9670`
- near-field recall `0.9476`
- near-field predicted positive rate `0.2325` vs true positive rate `0.2228`
- `xy_predicted_effect_aware_contraction=0.8779`
- `z_bounded_step_contraction=0.9337`

v51 focused runtime ep024:

```text
runtime_artifacts/coarse2contact_v2/v51_nearfield_multisource_tail_smoke_ep024/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v51_nearfield_multisource_tail_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl
```

- close leak count: `0`
- applied rows: `0`
- learned near-field ready rows: `0`
- max near-field confidence: `0.2013`

Current conclusion:

- The near-field mechanism is useful and should stay, but the current v51
  checkpoint is not sufficient.
- ep024 remains a hard OOD case for the current representation: it is true-near
  by offline labels but predicted far / not-near by the model.
- Do not promote v50 or v51.
- Next work should focus on representation and data: hard-negative/positive
  near-field rows from ep024-like windows, stronger wrist object-centric
  features, and a split that explicitly holds out the random5 roots while still
  including other near-field positives.

## 2026-06-07 Continuation: Spatial Moments v52

Implemented a lightweight non-privileged spatial geometry branch:

- Added `task_frame_v46_spatial_moment_features(...)`, computed directly from
  runtime-visible RGBD/depth-valid/coordinate channels.
- Features include depth-valid moments, near-depth/far-depth weighted
  centroids, second moments, depth statistics, and RGB means.
- Added `use_spatial_moments` to `TaskFrameV46AlignmentNet`.
- The option is checkpoint-serialized. Old checkpoints default to
  `use_spatial_moments=false`, so v46-v51 remain loadable.
- v52 enables this branch; it is intended to restore object-location cues that
  global pooled CNN features can wash out.

Verification:

```bash
conda run -n vla-adapter python -m py_compile \
  prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py \
  scripts/train_c2c_v2_task_frame_v46_alignment.py \
  scripts/eval_c2c_v2_task_frame_v46_alignment.py \
  scripts/evaluate_c2c_v2_rlbench.py

conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py
# 184 passed
```

Trained v52:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v52_spatial_moments_tail_candidate.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v52_spatial_moments_tail_candidate_train.json
```

v52 root-held-out validation:

- `use_spatial_moments=true`
- `xy_sign_match=0.9399`
- `z_sign_match=0.9557`
- `yaw_sign_match=0.8000`
- `xy_predicted_effect_aware_contraction=0.8955`
- `z_bounded_step_contraction=0.9420`
- `yaw_bounded_step_contraction=0.1574`
- near-field accuracy `0.9679`
- near-field recall `0.8745`

Focused ep024 runtime:

```text
runtime_artifacts/coarse2contact_v2/v52_spatial_moments_tail_smoke_ep024/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v52_spatial_moments_tail_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl
```

- v52 applied on `22` eval-label rows.
- learned-near ready rows: `22`; radius-ready rows: `12`.
- close leak count: `0`.
- planner close requested `37` times; all blocked by strict handoff.
- XY contraction `0.5909`, XY worsen `0.4091`, mean XY norm delta
  `-0.000043m`.
- combined contraction `0.5455`.
- Z contraction `0.6364`.
- Yaw action remained blocked (`yaw_allowed=0`), but yaw eval residual
  contraction was `0.4545` from planner/via coupled motion.

Random5 runtime smoke:

```text
runtime_artifacts/coarse2contact_v2/v52_spatial_moments_tail_smoke_random5
```

MP4s:

- `videos/ep023_fail.mp4`
- `videos/ep024_fail.mp4`
- `videos/ep025_fail.mp4`
- `videos/ep026_fail.mp4`
- `videos/ep027_fail.mp4`

Random5 result:

- insert success: `0/5`
- total applied eval-label rows: `89`
- close leak count: `0`
- total XY contraction `0.3933`, XY worsen `0.6067`
- mean XY norm delta `-0.000043m` (tiny average improvement despite poor
  contraction rate)
- combined contraction `0.6067`
- Z contraction `0.5506`
- yaw eval residual contraction `0.5506`, but yaw servo itself was still not
  allowed.

Current interpretation:

- v52 is the first candidate that reliably opens the correction window on the
  ep024-style near-field failure; this is a real activation/observability
  improvement over v50/v51.
- v52 is not a baseline. It does not prove stable XY correction on random5,
  and insert success remains `0/5`.
- The next bottleneck is online XY control quality, especially y-axis worsen
  and local-command-to-task-frame coupling. The next cycle should target
  control-effect calibration and axis-coupling losses on applied runtime rows,
  not merely near-field activation.

## 2026-06-07 Continuation: Applied-Transition Control-Effect v53/v54

The next implementation step added an explicit applied-transition learning
path. This does not change the close contract: planner-owned close and strict
handoff remain unchanged, and C2C still has no close authority.

New code:

- `scripts/build_c2c_v2_task_frame_applied_transition_manifest.py`
  builds pre/action/post rows from runtime gripper traces plus
  `runtime_observations/*.npz`.
- The manifest stores true pre residuals in `offline_labels` and true post
  residuals in `next_privileged_d*` fields. These are offline labels only;
  runtime input remains non-privileged.
- `scripts/train_c2c_v2_task_frame_v46_alignment.py` now prefers
  `applied_control_command_xy` for effect-head training and reports x/y split
  control-effect metrics.
- `scripts/evaluate_c2c_v2_rlbench.py` now prevents learned near-field
  confidence from bypassing the estimated XYZ activation radius by default.
  The bypass can only be enabled explicitly with
  `--v46_task_frame_allow_learned_near_field_radius_bypass`. This gates
  correction only; it never grants close/handoff.

Generated manifests:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v53_applied_transition_v52_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v53_applied_transition_v52_manifest.summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v54_applied_transition_v53_radius_guarded_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v54_applied_transition_v53_radius_guarded_manifest.summary.json
```

The v53 manifest has `111` applied-transition rows from v52 random5/focused
ep024, with close leak count `0`. The v54 on-policy manifest has `36` rows
from the radius-guarded v53 ep024 run, also with close leak count `0`.

New checkpoints:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v53_applied_transition_control_effect_candidate.pt
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v54_onpolicy_xy_effect_candidate.pt
```

Offline applied-transition eval:

- On the 111-row v52 applied-transition manifest, v53 improves over v52:
  - `xy_predicted_effect_aware_contraction`: `0.8018 -> 0.9189`
  - `y_predicted_effect_aware_worsen`: `0.1982 -> 0.1081`
  - `xy_effect_delta_mae`: `0.001338 -> 0.000880`
  - near-field accuracy: `0.0090 -> 0.9910`
- This was useful but not sufficient. The first v53 runtime smoke opened
  correction much too early because the learned near-field head was
  overconfident far outside the local support.

v53 focused ep024, before radius guard:

```text
runtime_artifacts/coarse2contact_v2/v53_applied_transition_control_effect_smoke_ep024/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v53_applied_transition_control_effect_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl
```

- Applied rows: `145`
- First applied row was step `20`, with offline true z about `0.598m`.
- close leak count: `0`
- XY contraction: `0.1241`
- Z contraction: `0.6138`
- combined contraction: `0.6759`

This run is a negative result for activation: learned near-field confidence
cannot be allowed to override the radius/near-field correction gate.

v53 focused ep024, radius guarded:

```text
runtime_artifacts/coarse2contact_v2/v53_applied_transition_control_effect_smoke_ep024_radius_guarded/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v53_applied_transition_control_effect_smoke_ep024_radius_guarded/gripper_traces/ep024_gripper_trace.jsonl
```

- Applied rows: `36`
- First applied row moved to step `66`, with offline true z about `0.0345m`.
- close leak count: `0`
- planner close requested `30` times; handoff allowed `0`
- XY contraction: `0.0`
- Z contraction: `0.6944`
- yaw eval residual contraction: `0.4722`
- combined contraction: `0.6667`

The radius guard fixed early correction, but the online XY control remained
wrong. The guarded run showed y-axis worsen on every applied transition.

v54 on-policy feedback:

- v54 was trained with v50 multi-source data, the v52 applied-transition
  manifest, and the radius-guarded v53 ep024 applied-transition manifest.
- The v53 guarded source was in the train split, not validation.
- On the guarded 36-row manifest, v54 still did not solve the issue:
  - observed transition XY contraction remained `0.0`
  - observed y-axis contraction remained `0.0`
  - model-predicted effect-aware contraction remained `1.0`

Current interpretation:

- v53/v54 are not promotion candidates.
- The activation/close side is now safer: strict close remains intact and
  learned near-field no longer bypasses radius by default.
- The core alignment bottleneck is now sharper: the current 2x2
  `xy_control_effect` head is not a trustworthy online local-command to
  task-frame residual transition model. It can predict contraction while the
  actual closed-loop y residual worsens.
- The next method should replace the 2x2 XY effect head with a
  command-conditioned transition model/Jacobian that takes the full local 6D
  command, current task-frame estimate, wrist RGBD, proprio, and temporal
  motion, then predicts `delta(dx,dy,dz,dyaw)` and uncertainty for the actual
  bounded command. The controller should choose a bounded step by minimizing
  predicted post-residual with uncertainty penalties, and should abstain or
  fall back to v42 XY when the transition model is not calibrated.

## 2026-06-07 Continuation: v55/v56 Command-Conditioned Transition

Implemented the first command-conditioned transition head inside the v46 model
family:

- `TaskFrameV46AlignmentNet.forward(..., command_6d=...)` predicts
  `command_delta`, `command_logvar`, and `command_support` for a proposed full
  local 6D command.
- `TaskFrameV46AlignmentCalibration.predict_command_transition_from_trace(...)`
  exposes the transition prediction to runtime using only the same
  non-privileged RGBD/proprio/history inputs plus the candidate local command.
- Training now stores `command_6d`, `has_command_6d`, and `command_support`
  arrays and applies transition delta/sign/support losses.
- Runtime added `--v46_task_frame_xy_mode transition_guarded_effect_aware`.
  This mode first proposes the effect-aware XY correction, predicts the
  post-command task-frame XY residual, and suppresses XY if the transition head
  predicts XY worsen. It never grants close/handoff.

New multi-run applied-transition manifest:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v56_applied_transition_multirun_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v56_applied_transition_multirun_manifest.summary.json
```

It has `1772` applied-transition rows from `9` source roots, covers episodes
`023-027`, and has close leak count `0`.

Checkpoints:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v55_command_transition_candidate.pt
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v56_applied_heavy_command_transition_candidate.pt
```

Offline evidence:

- v55 learned a transition head but still predicted the guarded ep024 y-axis
  with the wrong sign. It predicted y delta negative while all observed
  guarded ep024 transitions had positive y delta and y worsen.
- v56, trained with the larger multi-run applied-transition manifest, changed
  the behavior: on the guarded ep024 manifest it predicted
  `command_xy_predicted_contraction=0.0` and
  `command_xy_predicted_worsen=1.0`, matching the observed fact that the
  proposed XY command was harmful.
- v56 command support is still poorly calibrated. It should not be used as a
  success predicate.

Runtime focused ep024 with transition guard:

```text
runtime_artifacts/coarse2contact_v2/v56_transition_guarded_smoke_ep024/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v56_transition_guarded_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl
```

Result:

- Applied rows with eval labels: `95`
- transition-valid rows: `95`
- transition predicted XY contraction rows: `0`
- XY suppressed by transition guard: `95`
- close leak count: `0`
- planner close requested `39` times; handoff allowed `0`
- XY contraction: `0.2000`
- Z contraction: `0.6211`
- yaw eval residual contraction: `0.4737`
- combined contraction: `0.6526`

Negative finding:

- v56 transition guard successfully prevents the known harmful effect-aware XY
  correction from being applied.
- However, v56 still opens the correction window too early in ep024. The first
  applied row is step `36`, with offline true z about `0.364m`, while the model
  estimates z at only about `0.033m`.
- Therefore the remaining blocker is not just XY transition modeling. The
  near-field/task-frame residual estimator itself still has false-positive
  local-window errors, especially for Z/progress-to-contact.

Current next target:

Build a stricter non-privileged near-field/progress estimator before trusting
any bounded servo:

- train near-field/progress from positive and negative runtime windows, with
  far-z negatives like v56 step36 explicitly represented
- require agreement between residual radius, learned near-field, and temporal
  depth/proprio progress before activating correction
- keep transition guard for XY, but do not use it to compensate for a bad
  near-field predicate
- only after activation false positives are reduced should random5/random
  held-out MP4 be repeated

## 2026-06-07 Continuation: v57/v58 Near-Field And Axis-Strict Command Search

v57/v58 keep the same `v46_unified_task_frame_alignment_candidate` runtime
contract: runtime inputs are wrist RGBD/depth validity, proprio, planner prior,
history, and candidate local command only; privileged residuals remain
offline/eval labels. Neither checkpoint provides close authority, and strict
handoff was not relaxed.

New near-field/progress artifacts:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v57_nearfield_progress_guard_candidate.pt
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v58_nearfield_onpolicy_manifest.jsonl
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v58_nearfield_onpolicy_candidate.pt
```

Evidence:

- v57 fixed the known far-Z false positive offline: ep024 step36 predicted
  z moved back near the true far value instead of the earlier false near-field
  estimate.
- v57 was too conservative online: focused ep024 produced `0` applied rows.
- v58 restored on-policy near-field recall on the v58 manifest:
  near-field accuracy about `0.966`, recall about `0.967`, and z MAE about
  `0.0076m`.
- v58 focused ep024 without Z transition guarding applied `7` rows but pushed
  Z in the wrong direction. This confirmed that near-field activation alone is
  insufficient; the bounded command itself needs a calibrated transition model.

Runtime focused ep024 with XYZ transition guard:

```text
runtime_artifacts/coarse2contact_v2/v58_transition_xyz_guard_smoke_ep024/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v58_transition_xyz_guard_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl
```

Result:

- Applied rows: `6`
- XY suppressed by transition guard: `6`
- Z suppressed by transition guard: `6`
- close leak count: `0`
- XY contraction: `0.1667`
- Z contraction: `0.5000`
- yaw eval residual contraction: `0.6667`
- combined contraction: `0.6667`

This was a useful safety step, but still not a capability win: the guard mostly
suppressed bad commands instead of choosing good commands.

Implemented next controller step:

- Added `TaskFrameV46CommandSearchResult`.
- Added `task_frame_v46_transition_command_search(...)`.
- Runtime `transition_guarded_effect_aware` now evaluates a small set of
  bounded local candidate commands and chooses a command only if the
  command-conditioned transition head predicts improvement over no-correction.
- The selector additionally requires every moved axis to predict contraction;
  a combined-score improvement is not allowed to hide XY/Z/Yaw worsen.
- Trace now records selected local step, selected command, transition delta,
  support, score, per-axis contraction flags, and suppressed-axis flags.
- Unit tests cover supported bounded axis selection and rejection of candidates
  that improve combined score while worsening a moved axis.

Axis-strict focused ep024 smoke:

```text
runtime_artifacts/coarse2contact_v2/v58_transition_command_search_axisstrict_smoke_ep024/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v58_transition_command_search_axisstrict_smoke_ep024/gripper_traces/ep024_gripper_trace.jsonl
runtime_artifacts/coarse2contact_v2/reports/v58_transition_command_search_axisstrict_smoke_ep024_audit.json
```

Result:

- `190` unit tests pass.
- MP4 with wrist view was preserved.
- search-valid rows: `6`
- search-applied rows: `0`
- active-axis contraction violations: `0`
- true gripper close leak count using gripper index 7: `0`
- all six search rows reported `no_candidate_improves_no_correction`
- eval-label all-row contraction: XY `0.2105`, Z `0.6947`, yaw `0.4211`,
  combined `0.7368`

Interpretation:

- The axis-strict selector is now safer and auditable: it no longer applies a
  command when the learned transition predicts that any active axis worsens.
- The current v58 transition head is still not strong enough to choose useful
  three-axis commands in this focused runtime window. It becomes a suppressor,
  not a high-precision controller.
- Therefore v46/v58 is still a candidate, not a baseline. The project has not
  yet satisfied the random held-out three-axis contraction gate or the insert
  success-improvement gate.

Next target:

- Collect broader on-policy command-transition data from random held-out
  failure tails, with candidate command sweeps rather than only commands that
  the current controller happened to apply.
- Train the transition head to rank/choose among bounded candidate commands,
  not only predict deltas for a narrow on-policy action distribution.
- Keep the axis-strict per-moved-axis contraction rule in runtime.
- Promote only after random held-out failure tails show positive XY/Z/Yaw
  contraction and insert success improves without close leaks.

## 2026-06-07 Continuation: v59 Candidate Command Sweep Spec

The next data path is now explicit. Instead of assigning fake post-residual
labels to unexecuted candidate commands, the project builds a command-sweep
execution spec. A later evaluator pass must execute these candidate commands;
only then can `build_c2c_v2_task_frame_applied_transition_manifest.py` attach
offline pre/post labels.

Implemented:

- `scripts/build_c2c_v2_task_frame_command_sweep_spec.py`
- The script reads runtime traces plus runtime observations and selects
  runtime-near `RING_GRASP_ALIGN` rows.
- It emits bounded local candidate commands: zero, +/-x, +/-y, +/-z, +/-yaw,
  and combined XY corners.
- Rows contain runtime-visible pointers and candidate commands only:
  `runtime_obs_path`, `trace_path`, `base_command_local_6d`,
  `candidate_step_local_6d`, `candidate_command_local_6d`, and `command_6d`.
- Rows explicitly set `has_next_residual=false`,
  `uses_privileged_label_for_training=false`, and
  `privileged_label_boundary=no_transition_label_until_candidate_command_executed`.
- Privileged trace fields such as `grasp_probe_pre_true_error_t` and
  `grasp_probe_post_true_error_t` are stripped from the spec.

Smoke spec built from the axis-strict ep024 run:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.summary.json
```

Summary:

- selected runtime rows: `6`
- candidate commands per runtime row: `13`
- retained spec rows: `78`
- `uses_privileged_runtime=false`
- `uses_privileged_label_for_training=false`
- privileged substring audit on a sample row: none
- unit tests: `191 passed`

Command:

```bash
conda run -n vla-adapter python scripts/build_c2c_v2_task_frame_command_sweep_spec.py \
  --input runtime_artifacts/coarse2contact_v2/v58_transition_command_search_axisstrict_smoke_ep024 \
  --output_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.jsonl \
  --summary_json runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.summary.json \
  --max_source_rows 20 \
  --near_field_threshold 0.50 \
  --xy_step 0.003 \
  --z_step 0.003 \
  --yaw_step 0.010
```

Implemented first execution hook:

- `scripts/evaluate_c2c_v2_rlbench.py` now accepts
  `--task_frame_v46_command_sweep_spec_jsonl` and
  `--task_frame_v46_command_sweep_row_index`.
- The evaluator executes exactly one selected candidate command at its matching
  `episode_idx + step_idx`.
- The command-sweep hook overrides only bounded local 6D motion. It does not
  grant handoff and does not grant close authority.
- Trace fields include `task_frame_v46_command_sweep_active`,
  `task_frame_v46_command_sweep_executed`, candidate name, candidate step, and
  candidate command.

First executed command-sweep smoke:

```text
runtime_artifacts/coarse2contact_v2/v59_command_sweep_exec_ep024_row000/videos/ep024_fail.mp4
runtime_artifacts/coarse2contact_v2/v59_command_sweep_exec_ep024_row000/gripper_traces/ep024_gripper_trace.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_exec_ep024_row000_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_exec_ep024_row000_manifest.summary.json
```

Result:

- selected spec row: `0`, candidate `x_neg`
- runtime rows: `115`
- command-sweep active rows: `1`
- command-sweep executed rows: `1`
- true close leak count using gripper index 7: `0`
- MP4 with front+wrist view preserved
- applied-transition manifest retained rows: `1`
- manifest `close_leak_rows=0`
- observed transition for the executed command:
  - XY contracted: `true`
  - Z contracted: `false`
  - yaw eval residual contracted: `true`
  - combined contracted: `true`

Execution command:

```bash
conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py \
  --checkpoint_dir /home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt \
  --mode basin_recovery_shadow \
  --c2c_grasp_probe_policy runtime_estimator_xy \
  --c2c_grasp_probe_smoke_type runtime_style_c2c \
  --runtime_xy_calibration_json runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt \
  --task_frame_v46_ckpt runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v58_nearfield_onpolicy_candidate.pt \
  --enable_v46_task_frame_micro_servo \
  --v46_task_frame_xy_mode transition_guarded_effect_aware \
  --task_frame_v46_command_sweep_spec_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.jsonl \
  --task_frame_v46_command_sweep_row_index 0 \
  --episode_indices 24 \
  --max_steps 115 \
  --eval_seed 3407 \
  --output_root runtime_artifacts/coarse2contact_v2/v59_command_sweep_exec_ep024_row000 \
  --name_suffix v59_command_sweep_exec_ep024_row000 \
  --record_video \
  --video_layout front_wrist \
  --write_episode_videos \
  --record_gripper_trace \
  --dump_runtime_obs \
  --dump_runtime_obs_all_episodes \
  --capture_failure_target_pose
```

Next required work:

- Batch this command-sweep execution over enough spec rows and random held-out
  failure-tail episodes, preferably multi-GPU.
- Build a combined executed-sweep applied-transition manifest.
- Train the next command-conditioned transition model on executed sweep
  transitions.
- Rerun the axis-strict selector on random held-out failure tails and require
  positive XY/Z/Yaw contraction before any baseline promotion.

## 2026-06-07 Continuation: v59 Batch Sweep Runner

The command-sweep data-collection path now has a batch runner:

```text
scripts/run_c2c_v2_task_frame_command_sweep_batch.py
```

It reads a v59 command-sweep spec and launches one independent evaluator run per
selected spec row. Each run executes exactly one candidate command, preserves
front+wrist MP4, gripper trace, runtime observations, and offline eval sidecar,
then optionally builds a combined applied-transition manifest from successful
run roots.

Implemented behavior:

- row selection by explicit row indices, start/max rows, episode filter, and
  candidate-name filter
- optional multi-GPU scheduling with `--gpus` and `--max_parallel`
- `--dry_run` planning mode for checking commands without launching RLBench
- default fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- default v42 XY baseline:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- default v58 task-frame checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v58_nearfield_onpolicy_candidate.pt`
- no close/handoff changes; every child evaluator still uses strict handoff

Verification:

- Unit tests: `192 passed`
- Dry run:

```text
runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_dryrun_ep024_rows000_001.json
```

Dry-run summary:

- selected rows: `2`
- candidates: `x_neg`, `x_pos`
- planned GPUs: `0`, `1`
- generated evaluator commands include the fixed planner checkpoint, v42 XY
  checkpoint, v58 task-frame checkpoint, command-sweep row index, front+wrist
  MP4, gripper trace, runtime observations, and offline eval sidecar.

Dry-run command:

```bash
conda run -n vla-adapter python scripts/run_c2c_v2_task_frame_command_sweep_batch.py \
  --spec_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.jsonl \
  --output_root runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_dryrun_ep024_rows000_001 \
  --summary_json runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_dryrun_ep024_rows000_001.json \
  --row_indices 0,1 \
  --max_parallel 2 \
  --gpus 0,1 \
  --dry_run
```

Next execution command template:

```bash
conda run -n vla-adapter python scripts/run_c2c_v2_task_frame_command_sweep_batch.py \
  --spec_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.jsonl \
  --output_root runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_ep024_rows000_025 \
  --summary_json runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_ep024_rows000_025.json \
  --start_index 0 \
  --max_rows 26 \
  --max_parallel 2 \
  --gpus 0,1 \
  --build_manifest \
  --manifest_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows000_025_manifest.jsonl \
  --manifest_summary_json runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows000_025_manifest.summary.json
```

This is still data collection, not success evidence. Promotion still requires
training on executed sweep transitions and then proving random held-out
XY/Z/Yaw contraction plus insert success improvement.

## 2026-06-07 Continuation: v59 Real Small Batch And Training Smoke

The first real batch command-sweep execution beyond a single row has completed.
This validates the data path, not the final model.

Executed batch:

```text
runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_ep024_rows001_002
runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_ep024_rows001_002.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows001_002_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows001_002_manifest.summary.json
```

Batch command:

```bash
conda run -n vla-adapter python scripts/run_c2c_v2_task_frame_command_sweep_batch.py \
  --spec_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.jsonl \
  --output_root runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_ep024_rows001_002 \
  --summary_json runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_ep024_rows001_002.json \
  --row_indices 1,2 \
  --max_parallel 1 \
  --gpus 0 \
  --build_manifest \
  --manifest_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows001_002_manifest.jsonl \
  --manifest_summary_json runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows001_002_manifest.summary.json
```

Result:

- selected rows: `2`
- candidates executed: `x_pos`, `xy_nn`
- child evaluator return codes: `0`, `0`
- MP4s with front+wrist view preserved for both child runs
- applied-transition retained rows: `2`
- close leak rows: `0`
- observed XY contraction: `0.0`
- observed Z contraction: `0.0`
- observed yaw eval residual contraction: `1.0`

Combined seed manifest:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_seed_ep024_rows000_002_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_seed_ep024_rows000_002_manifest.summary.json
```

Summary:

- retained rows: `3`
- source eval roots: `3`
- close leak rows: `0`

Training smoke:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_command_sweep_seed_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_command_sweep_seed_smoke_train.json
```

Training command:

```bash
conda run -n vla-adapter python scripts/train_c2c_v2_task_frame_v46_alignment.py \
  --dataset_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_seed_ep024_rows000_002_manifest.jsonl \
  --output_checkpoint runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_command_sweep_seed_smoke.pt \
  --output_json runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_command_sweep_seed_smoke_train.json \
  --epochs 1 \
  --batch_size 2 \
  --image_hidden_dim 32 \
  --fusion_hidden_dim 32 \
  --device cpu \
  --use_spatial_moments \
  --max_abs_xy_label 0.08 \
  --max_abs_z_label 0.08 \
  --max_abs_yaw_label 0.35
```

Result:

- train rows: `2`
- val rows: `1`
- command transition rows: train `2`, val `1`
- `uses_privileged_runtime=false`
- checkpoint saved

Interpretation:

- The real data loop now exists:
  command-sweep spec -> independent candidate execution -> applied-transition
  manifest -> command-transition training.
- The tiny seed checkpoint is not a candidate and must not be promoted.
- The observed batch is useful because it contains action outcomes, including
  commands that worsen XY/Z while improving yaw/combined. This is the kind of
  counterfactual action evidence the transition selector was missing.
- Next step is scale, not claim: run the batch over many spec rows and random
  held-out failure-tail episodes, then train a real v59/v60 transition model.

Follow-up executed-command batch:

```text
runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_ep024_rows001_005
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows001_005_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows001_005_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_ep024_rows001_005.json
```

Result:

- selected rows: `5`
- candidates executed: `x_pos`, `xy_nn`, `xy_np`, `xy_pn`, `xy_pp`
- child evaluator return codes: all `0`
- MP4s with front+wrist view preserved for each child run
- applied-transition retained rows: `5`
- `uses_privileged_runtime=false`
- close leak rows: `0`
- observed XY contraction: `0.4`
- observed Z contraction: `0.0`
- observed yaw eval residual contraction: `1.0`
- combined residual contraction: `5/5`, but this is not enough for promotion
  because the per-axis Z result is still negative and the batch is one episode,
  one step, and five candidate commands.

Per-command outcome summary:

- `x_pos`: XY worsened, Z worsened, yaw contracted, combined contracted.
- `xy_nn`: XY worsened, Z worsened, yaw contracted, combined contracted.
- `xy_np`: XY contracted, Z worsened, yaw contracted, combined contracted.
- `xy_pn`: XY worsened, Z worsened, yaw contracted, combined contracted.
- `xy_pp`: XY contracted, Z worsened, yaw contracted, combined contracted.

Interpretation update:

- The close/handoff safety boundary survived direct command execution.
- The command-sweep execution path is now useful for training a transition
  selector because it contains both good and bad XY command outcomes under the
  same runtime context.
- The current evidence still says the v58/v59 transition head is not a working
  controller. Z did not contract in this batch, XY only contracted on two of
  five candidate commands, and the sample is far too narrow to support random
  held-out failure-tail claims.
- The next required action is broader command-outcome collection across many
  rows and episodes, then training a real command-aware v59/v60 selector before
  returning to closed-loop insert success MP4s.

Additional axis-command batch:

```text
runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_ep024_rows006_012
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows006_012_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows006_012_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_ep024_rows006_012.json
```

Result:

- selected rows: `7`
- child evaluator return codes: all `0`
- applied-transition retained rows: `6`; the `zero` row did not become a
  `task_frame_v46_applied` transition label
- MP4s with front+wrist view preserved for all child runs
- `uses_privileged_runtime=false`
- close leak rows: `0`
- observed XY contraction: `0.1667`
- observed Z contraction: `0.1667`
- observed yaw eval residual contraction: `0.8333`

Combined executed-command seed:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_ep024_rows000_012_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_ep024_rows000_012_manifest.summary.json
```

Summary:

- retained rows: `12`
- source eval roots: `12`
- episode coverage: `ep024` only
- `uses_privileged_runtime=false`
- close leak rows: `0`
- observed XY contraction: `0.3333`
- observed Z contraction: `0.0833`
- observed yaw eval residual contraction: `0.9167`

Interpretation update:

- The command-outcome pipeline is now stronger than a smoke: it can run
  bounded candidate commands, preserve videos, retain only applied transitions,
  and build trainable manifests without privileged runtime input.
- It is still far from the v46 success gate. The combined seed has only one
  episode and one intervention step, and only `1/12` retained commands reduced
  Z residual. The next model should treat this as evidence that Z action
  semantics and sampling need more work, not as a controller promotion.
- Before any insert-success claim, the same transition collection must cover
  random held-out failure tails with enough XY, Z, and yaw-positive/negative
  candidates to train and validate a selector by per-axis contraction.

Combined seed training smoke:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_command_sweep_ep024_rows000_012_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_command_sweep_ep024_rows000_012_smoke_train.json
```

Training command:

```bash
conda run -n vla-adapter python scripts/train_c2c_v2_task_frame_v46_alignment.py \
  --dataset_jsonl runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_ep024_rows000_012_manifest.jsonl \
  --output_checkpoint runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_command_sweep_ep024_rows000_012_smoke.pt \
  --output_json runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_command_sweep_ep024_rows000_012_smoke_train.json \
  --epochs 3 \
  --batch_size 4 \
  --image_hidden_dim 32 \
  --fusion_hidden_dim 32 \
  --device cpu \
  --use_spatial_moments \
  --max_abs_xy_label 0.08 \
  --max_abs_z_label 0.08 \
  --max_abs_yaw_label 0.35
```

Result:

- train rows: `10`
- val rows: `2`
- command transition rows: train `10`, val `2`
- `uses_privileged_runtime=false`
- `uses_privileged_label_for_training=true`
- `upgrade_gate=pending_random_holdout_closed_loop_validation`
- validation observed transition contraction in the held-out split:
  - XY: `0.0`
  - Z: `0.5`
  - Yaw: `1.0`
- predicted command contraction in validation:
  - XY: `0.0`
  - Z: `0.0`
  - Yaw: `1.0`

Interpretation:

- The 12-row checkpoint is only a pipeline smoke. It proves that the new
  applied-transition manifest can train through the v46 script with non-
  privileged runtime inputs.
- It also confirms the data imbalance problem: the current seed gives the
  learner enough yaw signal to predict yaw contraction, but not enough reliable
  Z/XY command-support signal to select a three-axis controller.

Random-heldout correction after the ep024-only concern:

- The first command-sweep spec was ep024-only because it came from
  `v58_transition_command_search_axisstrict_smoke_ep024`, the only trace that
  had `task_frame_v46_near_field_activation_ready` rows at the time.
- Random5 strictclose traces (`ep023`-`ep027`) initially produced
  `retained_rows=0` under the old builder because those traces did not contain
  the v46 near-field ready fields.
- `scripts/build_c2c_v2_task_frame_command_sweep_spec.py` now supports explicit
  random-heldout selection modes:
  - `probe_actionable`
  - `offline_residual_band`
  - `probe_or_offline_residual_band`
- Offline residual band selection is used only to choose replay windows. The
  emitted command-sweep spec still strips privileged residuals and records
  `uses_privileged_runtime=false`, `uses_privileged_label_for_training=false`,
  and `privileged_label_boundary=no_transition_label_until_candidate_command_executed`.

Random5 command-sweep spec:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_random5_strictclose_probe_seed.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_random5_strictclose_probe_seed.summary.json
```

Summary:

- selected runtime rows: `5`
- episode coverage: `ep023`, `ep024`, `ep025`, `ep026`, `ep027`
- retained command rows: `65`
- candidate commands per row: `13`
- selected steps: `ep023/024/025/027 step40`, `ep026 step41`
- `uses_privileged_runtime=false`
- `uses_privileged_label_for_training=false`

Random5 executed Z seed:

```text
runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_random5_z_seed
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_z_seed_executed_only_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_z_seed_executed_only_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_random5_z_seed.json
```

Result:

- child evaluator runs: `10/10` succeeded
- MP4s with front+wrist view preserved for every child run
- strict executed-only retained rows: `10`
- episode coverage: `ep023`-`ep027`, two Z candidates per episode
- `require_command_sweep_executed=true`
- `uses_privileged_runtime=false`
- close leak rows: `0`
- observed XY contraction: `0.8`
- observed Z contraction: `1.0`
- observed yaw contraction: `1.0`

Important interpretation:

- The previous random5 Z manifest had `16` rows because it kept later natural
  `task_frame_v46_applied` rows from the same child runs. That was too broad
  for command-sweep training.
- `scripts/build_c2c_v2_task_frame_applied_transition_manifest.py` and
  `scripts/run_c2c_v2_task_frame_command_sweep_batch.py` now support and use
  `require_command_sweep_executed=true`, so command-sweep manifests keep only
  the row where the requested candidate command actually executed.
- The random5 Z seed is real random-heldout transition evidence, but it still
  does not prove v46 success. Both `z_neg` and `z_pos` are relative offsets on
  top of a positive base command, and both contracted in these windows. This
  proves useful Z action-outcome signal exists, not that the model has learned
  a general Z selector.

Combined executed-only command seed:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_ep024_random5_executed_only_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_ep024_random5_executed_only_manifest.summary.json
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_command_sweep_ep024_random5_executed_only_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_command_sweep_ep024_random5_executed_only_smoke_train.json
```

Summary:

- retained rows: `22`
- episode coverage: `ep023`-`ep027`
- ep024 remains overrepresented (`14/22` rows)
- close leak rows: `0`
- observed XY contraction: `0.5455`
- observed Z contraction: `0.5000`
- observed yaw contraction: `0.9545`
- CPU training smoke completed with `uses_privileged_runtime=false`
- upgrade gate remains `pending_random_holdout_closed_loop_validation`

Interpretation:

- The project has now moved past ep024-only pipeline smoke.
- The current data is still too small and imbalanced to train or promote a
  command-aware v46/v59 controller.
- Next collection should run XY, Z, and yaw candidates over multiple selected
  windows per random held-out episode, then train on source-root held-out
  splits and evaluate per-axis contraction before any closed-loop insert
  success claim.

Random5 X/Yaw command seed:

```text
runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_random5_x_yaw_seed
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_x_yaw_seed_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_x_yaw_seed_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_random5_x_yaw_seed.json
```

Result:

- child evaluator runs: `20/20` succeeded
- episode coverage: `ep023`-`ep027`
- candidates per episode: `x_neg`, `x_pos`, `yaw_neg`, `yaw_pos`
- MP4s with front+wrist view preserved for every child run
- strict executed-only retained rows: `20`
- close leak rows: `0`
- observed XY contraction: `0.85`
- observed Z contraction: `1.0`
- observed yaw contraction: `1.0`

Important interpretation:

- This batch is useful because it adds random-heldout X/Yaw command-outcome
  rows beyond the previous Z-only seed.
- It is still not a clean causal proof that x/yaw candidate signs are solved.
  In these windows, the base planner/C2C local command already contains a
  strong positive Z component and the residual often contracts across all
  candidate variants. The data is useful for training and audit, but promotion
  still requires selector evaluation on held-out windows where candidate choice
  matters.

Random5 balanced XYZ/Yaw executed-only seed:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_random5_xyz_yaw_executed_only_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_random5_xyz_yaw_executed_only_manifest.summary.json
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_command_sweep_random5_xyz_yaw_executed_only_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_command_sweep_random5_xyz_yaw_executed_only_smoke_train.json
```

Summary:

- retained rows: `30`
- episode coverage: `ep023`-`ep027`, `6` rows per episode
- close leak rows: `0`
- observed XY contraction: `0.8333`
- observed Z contraction: `1.0000`
- observed yaw contraction: `1.0000`
- CPU training smoke completed with `uses_privileged_runtime=false`
- train rows: `24`
- val rows: `6`
- val observed transition contraction:
  - XY: `0.6667`
  - Z: `1.0000`
  - Yaw: `1.0000`
- val predicted command contraction:
  - XY: `0.3333`
  - Z: `1.0000`
  - Yaw: `0.6667`
- upgrade gate remains `pending_random_holdout_closed_loop_validation`

Important caveat:

- The first training smoke with `max_abs_z_label=0.08` retained no rows because
  these random5 windows are around large Z residuals. The successful smoke used
  `max_abs_z_label=0.75`, so this seed is a failure-tail transition dataset,
  not yet the near-contact high-precision Z dataset needed for final insert
  success.
- Next data collection should deliberately sample closer windows and include
  zero/no-op plus y/xy candidates, so the selector learns when to move, when to
  hold, and which axis sign truly matters near the handoff basin.

## 2026-06-07 Continuation: Random5 Late-Near Command Sweep

The previous random5 balanced seed was useful but too far from the handoff
basin: large Z residuals and natural planner/C2C drift made many variants look
contractive. To test whether the v46/v59 line can learn high-precision
near-contact action choice, a late-near spec was built from the same random5
strictclose traces using step `95`-`105` and tighter residual bands:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_random5_late_near_seed.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_random5_late_near_seed.summary.json
```

Spec summary:

- selected runtime rows: `5`
- retained candidate commands: `65`
- selected windows: `ep023`/`ep024`/`ep025`/`ep026` step `95`,
  `ep027` step `98`
- command variants per source row: `13`
- runtime boundary: `uses_privileged_runtime=false`
- spec boundary: no privileged training labels are written into the replay
  command spec; privileged pre/post residuals are attached only after the
  candidate command is actually executed.

The first late-near execution batch intentionally covered only `y_neg`,
`y_pos`, and `zero` so the dataset begins to include "do not move" and
candidate-choice rows near the grasp basin:

```text
runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_random5_late_near_y_zero_seed
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_late_near_y_zero_seed_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_late_near_y_zero_seed_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_random5_late_near_y_zero_seed.json
```

Result:

- child evaluator runs: `15/15` succeeded
- front+wrist MP4s preserved: `15`
- strict executed-only retained rows: `14`
- episode coverage: `ep023`-`ep027`; `ep027` retained `2` rows because the
  zero child did not produce a retained applied transition row
- close leak rows: `0`
- observed contraction:
  - XY: `0.4286`
  - Z: `0.1429`
  - Yaw: `0.2143`
- candidate split:
  - `y_pos`: XY contracted `4/5`, Z `1/5`, Yaw `1/5`
  - `y_neg`: XY contracted `1/5`, Z `1/5`, Yaw `1/5`
  - `zero`: XY contracted `1/4`, Z `0/4`, Yaw `1/4`

Interpretation:

- This is not a promotion result. It is a useful negative/diagnostic result:
  near the handoff basin, current v46/v59 commands do not yet provide stable
  three-axis residual contraction.
- `y_pos` has a real local XY signal in this small sample, but X still often
  worsens and the Z/Yaw residuals remain weak. The current model/controller
  should not be described as a high-precision task-frame alignment candidate.
- The `zero` rows are important because they expose natural drift and make it
  possible to train "hold/reacquire" decisions instead of treating every
  command as action evidence.
- The immediate next step is broader near-contact command-outcome collection,
  not another threshold tweak. Required additions are: X/Y/XY combined
  variants, Z variants with smaller base-step confounds, yaw variants separated
  by yaw-observable vs yaw-ambiguous rows, multiple windows per episode, more
  random held-out episodes, and source-root held-out train/validation splits.

Next target:

- Build a near-contact command-outcome dataset with enough rows where candidate
  choice changes the outcome, including `zero/no-op`, single-axis, and
  multi-axis candidates.
- Train the next selector on executed transitions only, with privileged labels
  confined to offline training/eval sidecars and no privileged runtime input.
- Gate promotion on held-out near-contact XY/Z/Yaw contraction and close leak
  count `0`; only then run closed-loop insert MP4 and success evaluation.

## 2026-06-08 Continuation: Random5 Late-Near Full-Candidate Sweep

The remaining late-near candidates were executed so each random5 source window
has X, XY-combined, Z, Yaw, y, and zero/no-op evidence:

```text
runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_random5_late_near_remaining_seed
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_late_near_remaining_seed_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_late_near_remaining_seed_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_random5_late_near_remaining_seed.json
```

Result:

- child evaluator runs: `50/50` succeeded
- front+wrist MP4s preserved: `50`
- strict executed-only retained rows: `50`
- episode coverage: `ep023`-`ep027`, `10` rows per episode
- close leak rows: `0`
- observed contraction:
  - XY: `0.3200`
  - Z: `0.2200`
  - Yaw: `0.1800`

The full late-near candidate dataset merges those `50` rows with the previous
`y_neg`/`y_pos`/`zero` batch:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_random5_late_near_full_candidates_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_random5_late_near_full_candidates_manifest.summary.json
```

Summary:

- retained rows: `64` (`ep027` zero did not retain an applied transition)
- episode coverage: `ep023`-`ep026` have `13` rows each, `ep027` has `12`
- close leak rows: `0`
- observed contraction:
  - XY: `0.3438`
  - Z: `0.2031`
  - Yaw: `0.1875`
  - combined: `0.2969`
- best visible XY candidate signals:
  - `xy_pp`: XY contraction `1.0000`
  - `xy_np`: XY contraction `0.8000`
  - `y_pos`: XY contraction `0.8000`
- Z/Yaw remain weak:
  - `z_pos`: Z contraction `0.4000`
  - `z_neg`: Z contraction `0.2000`
  - `yaw_pos`: Yaw contraction `0.2000`
  - `yaw_neg`: Yaw contraction `0.0000`

Training smoke:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_random5_late_near_full_candidates_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_random5_late_near_full_candidates_smoke_train.json
```

- split: source-root held-out
- train rows: `51`
- val rows: `13`
- `uses_privileged_runtime=false`
- val observed transition contraction:
  - XY: `0.5385`
  - Z: `0.0000`
  - Yaw: `0.1538`
- val predicted/effect metrics:
  - effect-aware XY predicted contraction: `0.8462`
  - direct command-delta XY predicted contraction: `0.0000`
  - direct command-delta XY predicted worsen: `1.0000`
  - command Z predicted contraction: `0.0000`
  - command Yaw predicted contraction: `0.4615`

Interpretation:

- The v46/v59 command-outcome infrastructure is now strong enough to expose
  which action families matter near contact.
- The current checkpoint is still not a candidate baseline. It is a smoke
  artifact proving that near-contact executed transitions can be trained
  without privileged runtime input.
- XY has a usable effect-model signal, but the direct command-delta objective is
  not yet aligned with selector promotion. The next selector should rank
  candidate commands by predicted post-residual / contraction instead of
  treating command-delta regression as sufficient.
- Z/Yaw are still the bottleneck. Current Z/Yaw command variants do not produce
  robust near-contact contraction. The next data/model iteration must improve
  task-frame observability for Z/Yaw rather than only retuning soft-gate
  thresholds.

Immediate next target:

- Collect a larger near-contact set over more random held-out episodes, with
  multiple windows per episode and explicit yaw-observable/yaw-ambiguous
  slices.
- Add smaller, cleaner Z command brackets that separate approach-axis residual
  from planner descent drift.
- Train a candidate-ranking head that predicts per-candidate post-residual,
  contraction probability, worsen risk, and axis-wise uncertainty; promote only
  if held-out full-candidate windows show improved XY/Z/Yaw contraction with
  close leak count `0`.

## 2026-06-08 Continuation: v60 Candidate Ranking Smoke

The direct command-delta head in the v59 smoke did not learn a usable selector:
it could model an XY effect-aware correction path, but direct command-delta
predicted XY contraction was still `0.0` on the held-out slice. A new training
entry point was added:

```text
scripts/train_c2c_v2_task_frame_v46_candidate_ranker.py
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v60_random5_late_near_candidate_ranker_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v60_random5_late_near_candidate_ranker_smoke_train.json
```

This ranker keeps the same `TaskFrameV46AlignmentNet` and the same
non-privileged runtime inputs, but changes the training target:

- group rows by `episode_idx + step_idx`, so candidate commands from the same
  runtime state are compared together;
- predict post-command residual score from the command-transition head;
- optimize a listwise cross-entropy so the lowest predicted post-residual
  candidate matches the observed best candidate;
- keep privileged residuals as offline labels only.

Smoke result on the 64-row random5 late-near full-candidate manifest:

- train groups: `4`
- held-out groups: `1`
- held-out group: `ep025:step0095`
- selected candidate: `xy_np`
- oracle candidate: `xy_np`
- top-1 best-score match: `1.0`
- held-out top-1 XY contraction: `1.0`
- held-out zero/no-op XY contraction: `0.0`
- held-out top-1 Z contraction: `0.0`
- held-out top-1 Yaw contraction: `0.0`

Interpretation:

- Candidate ranking is the correct direction for near-contact XY command
  choice. It directly fixes the failure mode where per-row command-delta
  regression can be numerically reasonable but still useless for choosing an
  action.
- This is not a promotion result. The held-out split has only one group, and
  Z/Yaw remain unsolved. The checkpoint is a smoke artifact, not a runtime
  baseline.
- The next real candidate must scale this ranking objective to many random
  held-out groups and add stronger Z/Yaw observability labels/commands before
  it can claim three-axis task-frame contraction.

Next target:

- Collect at least tens of near-contact candidate groups across random
  held-out episodes, with complete `zero`, X/Y/XY, Z, and Yaw candidates for
  each group.
- Split by episode/session group, not candidate row, so no same-window leakage
  occurs.
- Promote only if the ranking top-1 beats zero/no-op and current v59/v60 smoke
  on XY, Z, Yaw, combined contraction, and worsen risk across held-out random
  groups, with close leak count `0`.

## 2026-06-08 Continuation: v61 Hard-Bucket Full Command Sweep

The v61 hard-bucket command sweep was completed as a larger executed-transition
dataset for the v46 candidate ranker. It is still a training/evidence artifact,
not a promoted runtime baseline.

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v61_hardbucket_near_contact_ranker_spec.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v61_command_sweep_hardbucket_full_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v61_command_sweep_hardbucket_full_manifest.summary.json
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v61_hardbucket_candidate_ranker_z016_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v61_hardbucket_candidate_ranker_z016_smoke_train.json
```

Collection summary:

- spec rows: `208`
- complete candidate groups: `16`
- candidates per group: `13`
- executed candidates: `208 / 208`
- preserved MP4s: `208`
- close leak rows: `0`
- privileged runtime input: `false`
- privileged labels: offline pre/post transition labels only

Full-manifest observed contraction:

- XY: `0.2452`
- Z: `1.0000`
- Yaw: `0.0048`
- combined: `0.7788`

Candidate-level signal:

- `xy_np`: XY contraction `1.0000`
- `xy_pp`: XY contraction `0.8125`
- `y_pos`: XY contraction `1.0000`
- `zero`: XY contraction `0.0000`
- `z_pos`: Z contraction `1.0000`
- `zero`: Z contraction `1.0000`
- `yaw_pos`: Yaw contraction `0.0625`
- `yaw_neg`: Yaw contraction `0.0000`

Interpretation:

- The command-sweep pipeline is now large enough to reveal stable hard-bucket
  XY candidate families. Candidate ranking remains a useful direction for XY.
- Z contraction in this slice is not strong controller evidence because
  `zero` also has Z contraction `1.0000`; it is likely dominated by natural
  planner descent / residual drift in these near-contact windows.
- Yaw remains the central unsolved axis. The current yaw command bracket almost
  never creates observed yaw contraction, so the ranker has almost no positive
  yaw intervention signal to learn from.
- `max_abs_z_label=0.08` filtered many hard-bucket rows. The `z016` smoke used
  `max_abs_z_label=0.16` to keep all 16 groups for analysis; this is a data
  support diagnostic, not a runtime threshold change.

Ranker smoke on v61 hard-bucket with `max_abs_z_label=0.16`:

- train groups: `12`
- held-out groups: `4`
- held-out top-1 best-score match: `0.5`
- held-out top-1 XY contraction: `0.25`
- held-out zero/no-op XY contraction: `0.0`
- held-out top-1 Z contraction: `1.0`
- held-out zero/no-op Z contraction: `1.0`
- held-out top-1 Yaw contraction: `0.0`
- held-out top-1 combined contraction: `0.75`
- held-out zero/no-op combined contraction: `0.75`

Combined random5 + v61 hard-bucket smoke:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v61_random5_hardbucket_candidate_ranker_z016_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v61_random5_hardbucket_candidate_ranker_z016_smoke_train.json
```

- train groups: `16`
- held-out groups: `5`
- held-out top-1 best-score match: `0.4`
- held-out top-1 XY contraction: `0.2`
- held-out zero/no-op XY contraction: `0.0`
- held-out top-1 Z contraction: `1.0`
- held-out zero/no-op Z contraction: `1.0`
- held-out top-1 Yaw contraction: `0.0`
- held-out top-1 combined contraction: `0.6`
- held-out zero/no-op combined contraction: `0.8`

Decision:

- Do not promote v61/v46 as a baseline.
- Keep v42 as the fixed XY runtime baseline.
- Keep v61 ranker checkpoints as smoke artifacts only.
- Next work should change the data/modeling problem, not just train longer:
  build random held-out command sweeps with stronger source-held-out splits,
  add yaw-observable positive intervention windows, use symmetry-aware yaw
  hypotheses, and separate Z command effect from natural planner descent.

## 2026-06-08 Continuation: v62 Z/Yaw Diagnostic Command Grid

v61 showed that raw Z contraction was contaminated by zero/no-op drift and that
Yaw almost never produced raw contraction. Two changes were added to make the
next evidence more meaningful:

- `scripts/build_c2c_v2_task_frame_command_sweep_spec.py` now supports
  `candidate_profile=z_yaw_diagnostic`, with configurable `--z_steps` and
  `--yaw_steps`.
- `scripts/audit_c2c_v2_task_frame_zero_adjusted_effect.py` audits each
  candidate against the same-window `zero` command, so natural drift/descent is
  subtracted before deciding whether a command helped.

v62 diagnostic spec:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_spec.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_spec.summary.json
```

- selected runtime windows: `8`
- candidates per window: `35`
- candidate rows: `280`
- profile: `z_yaw_diagnostic`
- Z magnitudes: `0.0015`, `0.0030`
- Yaw magnitudes: `0.006`, `0.012`, `0.024`
- XY candidates disabled for this diagnostic
- privileged runtime input: `false`
- fake transition labels: not written

v62 group000 execution:

```text
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group000
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_group000_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_group000_zero_adjusted_effect_audit.json
```

- executed candidates: `35 / 35`
- preserved MP4s: `35`
- close leak rows: `0`
- privileged runtime input: `false`
- raw contraction: XY `1.0`, Z `1.0`, Yaw `0.0`
- zero-adjusted overall:
  - beats zero XY: `0.5000`
  - beats zero Z: `0.4706`
  - beats zero Yaw: `0.5882`
  - beats zero combined: `0.4706`

Best zero-adjusted candidates in this one window:

- `zyaw_pn_z0030_y0240`: beats zero Z/Yaw/combined, mean adjusted Z
  `-0.002577`, mean adjusted Yaw `-0.000804`
- `zyaw_pn_z0030_y0120`: beats zero Z/Yaw/combined, mean adjusted Z
  `-0.002574`, mean adjusted Yaw `-0.000499`
- `z_pos_0030`: beats zero Z/Yaw/combined, mean adjusted Z `-0.002573`
- `yaw_neg_0240`: beats zero Yaw/combined, mean adjusted Yaw `-0.000640`
- `yaw_pos_*` worsened yaw relative to zero in this window

Interpretation:

- The zero-adjusted audit is necessary. Raw Yaw contraction was `0.0`, but
  yaw-negative commands did create a small positive effect relative to zero.
- The Yaw effect is still tiny and based on one window, so it is evidence for
  data-design direction, not a learned control result.
- The sign asymmetry (`yaw_neg` helps, `yaw_pos` hurts) should be mined over
  more windows and expressed as a symmetry-aware yaw hypothesis target.
- Z positive commands show a real zero-adjusted effect in this window; future Z
  training/eval should use zero-adjusted command effect, not raw Z contraction.

Next target:

- Execute the remaining v62 diagnostic groups over random held-out and
  hard-bucket windows.
- Train a ranker with zero-adjusted axis-effect targets and source/session
  held-out splits.
- Promote only if held-out top-1 beats zero on XY, Z, Yaw, and combined effect,
  with close leak count `0` and MP4 trace evidence.

## 2026-06-08 Continuation: v62 Groups 001-003

The diagnostic sweep has now moved beyond the original ep024/group000 window.
This matters because a single window made yaw-negative commands look promising,
but the project goal is random held-out generalization, not an episode-specific
fix.

Executed artifacts:

```text
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group001
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_group001_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_group001_zero_adjusted_effect_audit.json

runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group002
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_group002_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_group002_zero_adjusted_effect_audit.json

runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group003
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_group003_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_group003_zero_adjusted_effect_audit.json
```

Execution summary:

- group001 window: `ep008`, step `63`
- group002 window: `ep016`, step `63`
- group003 window: `ep024`, step `60`
- executed candidates: `35 / 35` for each group
- front+wrist MP4s preserved: `35` per group
- close leak rows: `0` in all groups
- privileged runtime input: `false`

Zero-adjusted beats-zero rates:

```text
group000: combined 0.4706, XY 0.5000, Z 0.4706, Yaw 0.5882
group001: combined 0.5429, XY 0.6000, Z 0.4857, Yaw 0.2857
group002: combined 0.4857, XY 0.4857, Z 0.4857, Yaw 0.6000
group003: combined 0.4857, XY 0.3143, Z 0.4571, Yaw 0.4857
mean:     combined 0.4962, XY 0.4750, Z 0.4748, Yaw 0.4899
worst:    combined 0.4706, XY 0.3143, Z 0.4571, Yaw 0.2857
```

Interpretation update:

- The v62 grid confirms that Z/Yaw command-effect analysis must be
  zero-adjusted. Raw contraction is too polluted by natural planner descent and
  same-window drift.
- Z positive commands continue to show useful signal in some windows, but the
  overall beats-zero rate is still only near chance across the first three
  groups.
- Yaw is strongly window-dependent. Group000 favored yaw-negative commands;
  group001 penalized them and showed poor yaw beats-zero; group002 recovered
  yaw signal again; group003 returned to near-chance yaw effect. This is not
  ready for a fixed-sign or scalar-gain yaw controller.
- Group003 also shows that XY side effects from the Z/Yaw grid can become
  strongly negative (`0.3143` beats-zero), so future ranking must be
  multi-axis and worsen-aware, not yaw-only or Z-only.
- The correct next modeling move is symmetry-aware, observability-conditioned
  yaw ranking or multi-hypothesis residual prediction, not a direct promotion
  of v46/v61/v62.

Current stage:

- `v46_unified_task_frame_alignment_candidate` remains implemented but not
  validated as a baseline.
- `v42_expanded_v4pilot` remains the fixed XY baseline.
- v62 is an evidence-collection stage for building better zero-adjusted,
  source-held-out ranker targets.
- Continue executing groups004-007 before training another ranker, unless a
  stronger yaw-observable positive-intervention data design replaces this grid.

## 2026-06-08 Full v62 Diagnostic Result

The full v62 diagnostic grid has now been executed.

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group000
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group001
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group002
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group003
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group004
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group005
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group006
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group007
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_all_groups_summary.json
```

Execution summary:

- planned candidates: `280`
- executed candidates: `280 / 280`
- front+wrist MP4s preserved: `280`
- close leak rows: `0`
- privileged runtime input: `false`
- fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`

Zero-adjusted beats-zero rates:

```text
group000: combined 0.4706, XY 0.5000, Z 0.4706, Yaw 0.5882
group001: combined 0.5429, XY 0.6000, Z 0.4857, Yaw 0.2857
group002: combined 0.4857, XY 0.4857, Z 0.4857, Yaw 0.6000
group003: combined 0.4857, XY 0.3143, Z 0.4571, Yaw 0.4857
group004: combined 0.4857, XY 0.4571, Z 0.4857, Yaw 0.6571
group005: combined 0.5143, XY 0.6000, Z 0.4857, Yaw 0.4571
group006: combined 0.4857, XY 0.4857, Z 0.4857, Yaw 0.6000
group007: combined 0.4857, XY 0.3714, Z 0.4571, Yaw 0.4857
mean:     combined 0.4945, XY 0.4768, Z 0.4767, Yaw 0.5200
worst:    combined 0.4706, XY 0.3143, Z 0.4571, Yaw 0.2857
best:     combined 0.5429, XY 0.6000, Z 0.4857, Yaw 0.6571
```

Decision:

- Do not train or promote a production candidate directly from the v62 fixed
  Z/Yaw diagnostic grid.
- The grid is useful because it proves the current fixed-sign Z/Yaw command
  family is not a reliable task-frame controller: average combined and Z
  effects are near chance, and worst-window yaw is poor.
- Yaw has observable positive windows but not a stable fixed sign. The next
  data design must be yaw-observable and symmetry-conditioned.
- Z/Yaw commands can create bad XY side effects, so the next ranker must use
  multi-axis zero-adjusted labels and explicit worsen penalties.

Next target:

Build a v63-style positive-intervention dataset/spec that selects windows by
non-privileged yaw observability and symmetry consistency, samples yaw
hypotheses rather than fixed global signs, keeps same-window zero controls, and
labels candidate commands by zero-adjusted combined/axis contraction before any
new ranker is trained.

## 2026-06-08 v63 Spec Builder Status

The v63 data-collection entry point has been added to
`scripts/build_c2c_v2_task_frame_command_sweep_spec.py`.

New spec options:

- `selection_mode=yaw_observable_symmetry`
- `candidate_profile=yaw_observable_symmetry`

The selection requires runtime-visible yaw evidence:

- `task_frame_v46_yaw_observable=true`
- `task_frame_v46_yaw_ambiguous=false`
- `task_frame_v46_yaw_unobservable=false`
- yaw confidence and hypothesis gap above configured thresholds
- stable alias/symmetry decision by default

The emitted rows still keep the same non-privileged boundary as v61/v62:

- `uses_privileged_runtime=false`
- `uses_privileged_label_for_training=false`
- no pre/post privileged residual labels in the spec rows
- `close_control_allowed=false`
- same-window `zero` candidate preserved

Current artifact scan result:

- existing v46/v58/v62 runtime traces contain many `task_frame_v46_yaw_*`
  fields, but all scanned rows report
  `(yaw_observable=false, yaw_ambiguous=true, yaw_unobservable=true)`.
- the maximum observed yaw confidence is still low, so no current artifact row
  satisfies the strict v63 yaw-observable selection.

Interpretation:

- The lack of v63 rows is not an ep24-only issue. The current runtime estimator
  is globally treating yaw as unobservable/ambiguous.
- Therefore the next collection cannot be a direct training run from existing
  traces. It must first produce yaw-observable windows, either by improving the
  non-privileged yaw observability estimator or by running targeted collection
  where wrist view/support makes the square-symmetry yaw hypotheses separable.
- v63 should then execute symmetry-conditioned yaw/Z candidates with zero
  controls and multi-axis zero-adjusted labels before any new v46/v47 ranker is
  trained or promoted.

## 2026-06-08 v63 Bootstrap Progress

Two implementation fixes were made after the first v63 scan:

- `task_frame_v46_alignment._as_bool` now parses string booleans explicitly.
  This prevents `"False"` labels from being interpreted as true when manifests
  come from mixed JSON/CSV-style sources.
- v46 offline/train/eval metrics now expose observability calibration rates:
  `xy/z/yaw_observable_target_rate`, `*_predicted_rate`,
  `yaw_ambiguous_*_rate`, and `yaw_control_*_rate`.

Yaw observability audits:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yawbalanced_yaw_observability_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v58_nearfield_yaw_observability_audit.json
```

On `task_frame_alignment_v46_yaw_balanced_manifest.jsonl`:

- `v46_yawbalanced` predicts yaw observable/control for all rows, while the
  target yaw-control rate is only about `0.0203`. This is an over-open yaw gate.
- `v58_nearfield_onpolicy` predicts yaw observable at about `0.0412` and yaw
  control at about `0.0407`, closer to the `0.0203` target, but still not
  validated as a runtime yaw controller.

This explains the confusing MP4/runtime behavior: different v46-era
checkpoints have incompatible yaw gate calibration. Some are effectively
over-open in offline replay; the current runtime traces in the scanned smoke
roots are effectively yaw-closed because their actual windows are mostly
ambiguous/unobservable.

New bootstrap spec:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_spec.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_spec.summary.json
```

Spec summary:

- source: `task_frame_alignment_v46_yaw_balanced_manifest.jsonl`
- selected yaw-positive source rows: `114`
- candidate commands: `3990`
- candidates per source row: `35`
- episodes covered: `9`
- source roots covered: `25`
- same-window zero controls: `114`
- close-control rows: `0`
- privileged-runtime rows: `0`
- privileged labels are stripped from spec rows

This is still a bootstrap data-collection spec, not a training result. The next
required step is to execute a held-out subset of these v63 rows, preserve
front+wrist MP4, then run zero-adjusted XY/Z/Yaw/combined audits before using
the results to train a candidate ranker or promote any v46/v47 checkpoint.

## 2026-06-08 v63 Group000 Runtime Result

Executed the first complete v63 yaw-observable candidate group:

```text
runtime_artifacts/coarse2contact_v2/v63_command_sweep_yaw_observable_bootstrap_group000
runtime_artifacts/coarse2contact_v2/reports/v63_command_sweep_yaw_observable_bootstrap_group000.json
runtime_artifacts/coarse2contact_v2/reports/v63_command_sweep_yaw_observable_bootstrap_group000_retry_failed.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group000_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group000_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_yaw_observable_bootstrap_group000_zero_adjusted_effect_audit.json
```

Execution details:

- selected spec rows: `0-34`
- episode/window: `ep006`, step `121`
- candidate commands: `35 / 35`
- front+wrist MP4s preserved: `35`
- gripper traces preserved: `35`
- planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- XY baseline: `runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- v46 task-frame checkpoint:
  `runtime_task_frame_alignment_v58_nearfield_onpolicy_candidate.pt`
- close leak rows: `0`
- privileged runtime rows: `0`

Operational note:

- The first parallel run used `max_parallel=4` and hit CUDA OOM on 6 rows.
- The failed rows were rerun serially with `max_parallel=1`, and the complete
  group was rebuilt into a 35-row manifest.

Zero-adjusted result:

```text
beats_zero_combined: 0.4706
beats_zero_xy:       0.4118
beats_zero_z:        0.4412
beats_zero_yaw:      0.4706
```

Interpretation:

- v63 group000 is safe with respect to close ownership and runtime privilege.
- It is not a promotion result: overall beats-zero rates are below chance.
- It does contain local positive candidates. The best combined candidates were
  `zyaw_sym_p_hyp_neg_z0030_y0240`,
  `zyaw_sym_p_hyp_neg_z0030_y0120`, and
  `zyaw_sym_p_hyp_neg_z0015_y0240`, which suggests the symmetry-conditioned
  candidate family can expose useful yaw/Z interventions.
- Because this is one window only, the next gate is to execute multiple
  source-held-out v63 groups, then train/evaluate a ranker only if top-1
  selection beats same-window zero on XY, Z, Yaw, and combined metrics without
  XY side-effect regressions.

## 2026-06-08 v63 Group003 Runtime Result

Executed a second complete v63 yaw-observable bootstrap group:

```text
runtime_artifacts/coarse2contact_v2/v63_command_sweep_yaw_observable_bootstrap_group003
runtime_artifacts/coarse2contact_v2/reports/v63_command_sweep_yaw_observable_bootstrap_group003.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group003_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group003_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_yaw_observable_bootstrap_group003_zero_adjusted_effect_audit.json
```

Execution details:

- selected spec rows: `105-139`
- episode/window: `ep010`, step `119`
- candidate commands: `35 / 35`
- front+wrist MP4s preserved: `35`
- gripper traces preserved: `35`
- planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- XY baseline: `runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- v46 task-frame checkpoint:
  `runtime_task_frame_alignment_v58_nearfield_onpolicy_candidate.pt`
- close leak rows: `0`
- privileged runtime rows: `0`
- manifest retained rows: `35`
- manifest built with `require_command_sweep_executed=true`

Zero-adjusted result:

```text
beats_zero_combined: 0.4857
beats_zero_xy:       0.3429
beats_zero_z:        0.5143
beats_zero_yaw:      0.4571
```

Manifest observed contraction, before zero adjustment:

```text
observed_xy_contraction:  0.0571
observed_z_contraction:   0.6571
observed_yaw_contraction: 0.3714
```

Interpretation:

- v63 group003 confirms that the command-sweep and MP4 preservation path is
  now reliable across another held-out window.
- Close ownership remains clean: planner close requests were blocked by
  strict handoff, and close leak count stayed at zero.
- The result is not promotable. Z has weak positive evidence in this window,
  but XY/Yaw and combined same-window beats-zero remain below the required
  gate.
- The main next step is not to hand-pick a command from this grid. The right
  next step is to execute more source-held-out v63 groups and train/evaluate a
  symmetry-conditioned candidate ranker/top-1 selector on executed transition
  outcomes. Promotion requires held-out top-1 improvement over zero on
  combined, XY, Z, and Yaw without reopening close authority.

## 2026-06-08 v63 Group030 Runtime Result

Executed a third complete v63 yaw-observable bootstrap group on a different
episode/window:

```text
runtime_artifacts/coarse2contact_v2/v63_command_sweep_yaw_observable_bootstrap_group030
runtime_artifacts/coarse2contact_v2/reports/v63_command_sweep_yaw_observable_bootstrap_group030.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group030_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group030_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_yaw_observable_bootstrap_group030_zero_adjusted_effect_audit.json
```

Execution details:

- selected spec rows: `1050-1084`
- episode/window: `ep013`, step `134`
- candidate commands: `35 / 35`
- front+wrist MP4s preserved: `35`
- gripper traces preserved: `35`
- close leak rows: `0`
- privileged runtime rows: `0`
- manifest retained rows: `35`
- manifest built with `require_command_sweep_executed=true`

Zero-adjusted result:

```text
beats_zero_combined: 0.6000
beats_zero_xy:       0.3429
beats_zero_z:        0.4857
beats_zero_yaw:      0.4857
```

Manifest observed contraction, before zero adjustment:

```text
observed_xy_contraction:  0.4857
observed_z_contraction:   0.4000
observed_yaw_contraction: 0.4000
```

Interpretation:

- v63 group030 is safe with respect to close ownership and runtime privilege.
- It is the first v63 group in this bootstrap set with combined beats-zero
  clearly above chance, so the candidate family does contain useful held-out
  interventions.
- It still does not solve the v46 objective. XY beats-zero is weak and Z/Yaw
  are approximately chance, so this is evidence for training a selector, not
  evidence for direct runtime promotion.
- The next gate should require a source-held-out ranker/top-1 selector to beat
  zero on combined and each axis across multiple episode-diverse groups. A
  single strong combined group is not enough for insert success claims.

## 2026-06-08 v63 Three-Group Ranker Smoke

The three complete v63 executed candidate groups were merged into a small
leave-one-group-out ranker smoke dataset:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_groups000_003_030_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_groups000_003_030_candidate_ranker_loo_summary.json
```

Dataset:

- rows: `105`
- groups: `ep006:step0121`, `ep010:step0119`, `ep013:step0134`
- candidates per group: `35`
- close leak rows: `0`
- privileged runtime rows: `0`

Ranker update:

- `scripts/train_c2c_v2_task_frame_v46_candidate_ranker.py` now supports
  `--rank_score_mode axis_balanced`.
- The original `residual` mode ranks candidates by predicted post-residual
  norm.
- The new `axis_balanced` mode adds control-aware penalties for axis worsen
  and missing per-axis contraction. This better matches the v46 objective:
  high-precision alignment requires all relevant axes to improve, not just a
  smaller combined scalar.

Initial smoke result:

- The residual ranker is not reliable on held-out groups. It often selects a
  candidate that helps one part of the residual while losing yaw or XY.
- The first axis-balanced ranker improves the failure shape in some splits, but
  it is still not stable.

Updated axis-balanced v2 result:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_groups000_003_030_candidate_ranker_axis_v2_loo_summary.json
```

- A unit test exposed that the initial axis-balanced penalty still allowed a
  large XY improvement to hide a yaw worsen. The axis-worsen penalty was raised
  to `5.0`.
- held-out `ep006`: top-1 matches the axis-balanced oracle candidate but loses
  combined/yaw relative to zero, which already contracts all axes
- held-out `ep013`: top-1 improves combined/yaw over zero, but loses XY and Z
- held-out `ep010`: top-1 still fails combined, XY, and Yaw
- Therefore no v63 ranker checkpoint should be promoted.

Decision:

- Keep `axis_balanced` as the preferred offline ranker objective for the next
  data round, but treat it as a training objective, not as evidence of solved
  task-frame control.
- Do not upgrade v46/v63 from this smoke. Three windows are insufficient, and
  the selector still fails a held-out yaw/axis-coupling case.
- Continue with episode/source-diverse executed candidate collection, then
  require source-held-out top-1 to beat zero on combined, XY, Z, and Yaw before
  any runtime MP4 promotion or insert-success claim.
