# C2C v2 XY v42 Generalization Plan

This document is the working plan for the next C2C v2 XY phase. It fixes the
direction, the data policy, the evaluation gate, and the promotion rule:
train with enough diverse data, but accept candidates only when they
generalize across random episodes, observability regimes, failure buckets,
and worsen tails.

## Clear Objective

Train `v42_xy_spatial_temporal_generalization` into a new XY baseline that can
generalize beyond the small sentinel set and still remain safe under the strict
handoff contract.

In concrete terms, the target is to make `v42_expanded_v4pilot` or its next
spatial-temporal successor the active XY baseline only if it:

- improves random held-out failure-tail alignment, including worst-slice
  random generalization
- reduces XY worsen/reverse on hard buckets and occlusion / partial-view tails
- does not regress old4/random5 sentinels beyond the allowed tolerance
- keeps strict handoff semantics unchanged
- wins by offline A/B before any MP4 claim is made

In plain terms: improve XY alignment on random held-out failure tails and hard
buckets without reopening close/handoff authority, overfitting to a small
episode set, or confusing visual smoke success with true generalization.

Current promotion status:

- `v42_expanded_v4pilot` is the active XY baseline.
- Future successors must beat it on worst-slice random generalization and
  hard-bucket validation, while keeping strict handoff semantics unchanged.

## Fixed Defaults

- Runtime environment: `conda run -n vla-adapter ...`
- Planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- Current stage: `RING_GRASP_ALIGN`
- Baselines to preserve: `v37`, `v38`, and `v41`
- Runtime contract: strict handoff stays enabled; this phase only upgrades XY
  alignment, not z/yaw/close authority.

## Goal

The v42 target is:

**Build a non-privileged XY spatial-temporal estimator that improves random
failure-tail alignment without sacrificing old4/random5 sentinels, hard-bucket
performance, or low-observability safety.**

Operationally, this means v42 only becomes the active XY baseline if it beats
the current best baseline on worst-slice random generalization and hard-bucket
validation while preserving the strict handoff contract and the sentinel
episodes.

Success means v42 reduces XY worsen/reverse behavior on random held-out traces
and hard buckets while keeping visible-scene alignment stable. It is not enough
to look good on selected MP4s, old4, random5, or ep25/26 alone.

## Data Strategy

- Training data is not limited to 10 episodes. Prefer enough balanced data,
  ideally `50-100+` episodes when runtime observations and labels are
  available.
- Use multiple sources: old4/random5, random planner failure tails, hard-bucket
  active rows, occlusion/partial-view rows, and C2C worsen tails.
- Parallel collection is allowed. Multi-GPU or multi-process collection should
  write separate eval roots that are later merged by dataset builders.
- Each sample must preserve audit fields: `source_eval_root`, `sequence_id`,
  `episode_idx`, `step_idx`, `bucket`, `observability_bucket`, `trace_path`,
  and `uses_privileged_runtime=false`.
- Privileged residual labels may be used only for offline training/eval
  sidecars. Runtime inputs must remain non-privileged.

## Generalization Split

- Split train/val/test by `source_eval_root` or session, not only by
  `episode_idx`.
- `random10_generalization` is a fixed quick acceptance gate, not a training
  size limit.
- Keep a larger `random_holdout_pool` for periodic checks after a candidate
  passes the quick gate.
- old4, random5, ep25/26, and hard-bucket rows are sentinel slices for
  regression detection and diagnosis. They are not the whole objective.

## v42 Candidate

- Generic trainer output name:
  `runtime_xy_spatial_temporal_v42_generalization_candidate.pt`
- Current concrete promotion candidate:
  `runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- Inputs: wrist RGBD/depth validity, proprio, planner prior, motion/history,
  and non-privileged geometry or heatmap features.
- Outputs: XY residual, direction confidence, visible confidence, step scale,
  and risk reason.
- Runtime control remains bounded XY only: `xy_gain=0.35`,
  `max_xy_step=0.003`, multiplied by model `xy_step_scale`.
- Loss remains control-aware: direction/sign, bounded-step contraction,
  worsen, overshoot, and reverse penalties.
- Sampling must balance random, hard-bucket, occlusion, partial-view, and
  worsen-tail rows so visual-observable majority rows do not dominate training.

## Upgrade Gate

A/B must compare `v37`, `v38`, `v41`, and `v42`.

Report slices:

- random10 overall and per episode
- failure bucket
- observability bucket
- ep25/26 worsen tails
- old4
- random5
- hard-bucket active rows

Metrics per slice:

- `entry_rate`
- `contraction`
- `worsen`
- `overshoot`
- `reverse`
- `near_entry`
- `mean_step`
- dominant failure reason
- trace paths

v42 can only become the active XY baseline if:

- random10 overall contraction beats the stronger of v38/v41
- random10 worst-slice worsen does not regress against the current baseline
- ep25/26 worsen is clearly below v38 and targets `< 0.50`
- old4/random5 contraction do not regress by more than `0.05` absolute
- overshoot is not worse than v41
- reverse is not worse than v41

If v42 improves only one slice while degrading random or low-observability
behavior, its status remains `pending`.

## MP4 Gate

Run MP4 only after offline gate passes.

MP4 comparison must include planner-only, v37, v38, v41, and v42. Videos must
include wrist camera view and cover:

- random10 worst 3 episodes
- random10 best 2 episodes
- old4
- ep25/26
- hard-bucket samples

This phase evaluates XY alignment stability and reduced push-worse behavior.
Close/handoff is intentionally not the v42 success metric.

## Current Candidate Read

- `v42_expanded_v4pilot` is currently the strongest measured model and the
  active XY baseline in this line.
- It is the concrete checkpoint to beat for all future XY candidates.
- It beats `v41` and `v42_expanded_v3pilot` on the latest runtime A/B for the
  small gate/holdout/sentinel evaluation set, including random10 contraction,
  random holdout contraction, old4 reverse, and random5 reverse.
- The latest hard-bucket A/B also favors `v42_expanded_v4pilot` over `v41`
  across contraction, worsen, overshoot, reverse, and ep25/26 worsen.
- The hard-bucket report also shows low-visibility worsen improving from
  `0.199` to `0.055`, with partial-worsen remaining at `0.000`.
- Hard-bucket MP4 smoke now exists for the leading candidate with wrist camera
  preserved, but that smoke is still only evidence for qualitative inspection.
- It is now the unconditional active baseline for this XY line, because the
  promotion rule was satisfied by the offline worst-slice gate and hard-bucket
  validation.

## Latest Smoke Evidence

The current branch now has matching MP4 smoke evidence for the leading
candidate, with wrist camera views preserved:

- old4 smoke:
  `runtime_artifacts/coarse2contact_v2/mp4_smoke_v42_expanded_v4pilot_old4_front_wrist`
- random5 smoke:
  `runtime_artifacts/coarse2contact_v2/mp4_smoke_v42_expanded_v4pilot_random5_front_wrist`

Both smokes ran with:

- `c2c_stage_shadow`
- `runtime_estimator_xy`
- `runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- strict close blocking kept on
- `front_wrist` video layout

These are smoke-level visual confirmations, not the final promotion gate. The
promotion rule is still the offline worst-slice A/B plus hard-bucket gate.

## One-Line Phase Rule

Do not replace `v42_expanded_v4pilot` unless a future candidate is better than
the current baseline on the worst random holdout slices and the hard-bucket
tails, without regressing old4/random5 or reopening strict handoff semantics.
