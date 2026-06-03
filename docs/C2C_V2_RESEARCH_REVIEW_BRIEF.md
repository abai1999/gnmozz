# Coarse2Contact v2 Research Review Brief

This brief is the main entry point for external reviewers who need to judge
whether the current C2C v2 code is still aligned with the research goal.

The short answer is: the project is still on the intended research path, but it
has not yet proven the final claim. The current code is best understood as a
contract-calibrated precision takeover framework plus offline failure-tail
diagnostics. It should not yet be reviewed as a finished closed-loop recovery
controller.

## Research Goal

C2C v2 is meant to be the precision layer on top of a frozen VLA planner.

The planner should handle coarse approach and transfer. C2C should take over
all high-precision stages:

- align gripper jaw frame to the ring grasp frame,
- enter a verified near-grasp basin,
- close only when close-ready is truly satisfied,
- later align the held ring aperture to the target spoke axis,
- then slide and recover from jams or loss of observability.

The current proof target is narrower: `RING_GRASP_ALIGN` on
`insert_onto_square_peg`, especially planner failure tails. Success-window
episodes such as diagnostic `ep006`-style clips are sanity checks only. They
are not main evidence.

## Fixed Review Assumptions

- Runtime environment: `conda run -n vla-adapter ...`
- Fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- Current stage under study: `RING_GRASP_ALIGN`
- Runtime supervisor must remain non-privileged.
- Privileged pose/mask data is allowed only for offline relabel, audit, oracle
  probe evaluation, and acceptance analysis.
- PCA/image-axis yaw is diagnostic only. It must not become a trusted yaw
  control signal unless a jaw-local dyaw estimator passes held-out validation.

## Code Map

Core runtime and contract:

- `configs/coarse2contact/tasks/insert_onto_square_peg.yaml`
- `prismatic/robot/coarse2contact_v2/specs.py`
- `prismatic/robot/coarse2contact_v2/supervisor.py`
- `prismatic/robot/coarse2contact_v2/basin_state.py`
- `prismatic/robot/coarse2contact_v2/takeover_contract.py`
- `prismatic/robot/coarse2contact_v2/grasp_probe_execution.py`
- `prismatic/robot/coarse2contact_v2/grasp_probe_shell.py`

Relabel, candidate building, and audit:

- `scripts/relabel_c2c_v2_privileged_basin_frames.py`
- `scripts/audit_c2c_v2_frame_contract_relabel.py`
- `scripts/build_c2c_v2_grasp_failure_tail_candidates.py`
- `scripts/build_c2c_v2_failure_tail_hard_manifest.py`
- `scripts/build_c2c_v2_hard_window_support_supplement.py`
- `scripts/build_c2c_v2_grasp_failure_tail_hard_bucket_gap_report.py`
- `scripts/audit_c2c_v2_grasp_failure_tail_intervention.py`
- `scripts/diagnose_c2c_v2_grasp_failure_tail_direction.py`
- `scripts/compare_c2c_v2_queue_flush_ablation.py`

Runtime/eval harness and MP4 support:

- `scripts/evaluate_c2c_v2_rlbench.py`
- `scripts/run_c2c_v2_grasp_shell_episode_sweep.py`
- `scripts/make_c2c_side_by_side_mp4.py`

Regression tests:

- `tests/test_coarse2contact_v2.py`

## Intended Three-Layer Design

### 1. Residual Estimator

The estimator should represent true frame-to-frame residual:
`gripper_jaw_frame -> ring_grasp_frame`, expressed as jaw-local
`dx/dy/dz/dyaw`.

Important boundary: raw visual features such as mask centroid, depth support,
and PCA axis are evidence. They are not automatically the control residual.

### 2. Observability Gate

The system must decide when each axis is trustworthy enough to control.

Current yaw policy is intentionally conservative:

- `stable_alias_control` can be used as yaw audit evidence.
- `frame_drift_abstain` should keep yaw control blocked.
- PCA yaw remains diagnostic-only.

### 3. Takeover Tier

The code should distinguish these states rather than use one hard gate:

- `frontier_pullback_candidate`
- `outer_pullback_candidate`
- `coarse_pullback_candidate`
- `near_basin_shell`
- `micro_entry_ready`
- `close_ready`

This is central to the project. C2C should be allowed to show bounded xy
pullback evidence without falsely claiming that close is safe.

## Current Evidence Snapshot

The latest local hard-bucket validation used the fixed 30k planner and only
`RING_GRASP_ALIGN`. Runtime artifacts are not committed, so reviewers should
treat the numbers below as a documented status snapshot rather than
independently reproducible git evidence.

Focused hard-bucket validation summary:

| bucket / protocol | active rows | xy contraction | overshoot | current read |
| --- | ---: | ---: | ---: | --- |
| `large_xy_large_yaw` / flush | 52 | 0.935 | 0.000 | active support exists; support still narrow |
| `large_xy_large_yaw` / retain | 74 | 0.919 | 0.000 | more active rows; not obviously worse than flush |
| `small_xy_large_yaw` / flush | 212 | 0.791 | 0.028 | active support exists; contraction still not clean |
| `small_xy_large_yaw` / retain | 212 | 0.717 | 0.023 | comparable active rows; protocol alone is not the full issue |

Interpretation:

- `large_xy_large_yaw` is no longer zero-active. The main question is whether
  support can be widened beyond a few favorable windows.
- `small_xy_large_yaw` is no longer best explained as a sign flip. The formal
  sign diagnostic is `oracle_xy_step_cosine_to_residual`; the current pattern
  points more toward step-size / horizon / frame-alignment refinement.
- Queue flushing helps in some slices, but the current bottleneck is not simply
  queue protocol. The next audit must keep support/window/step-size separated.
- Focused aggregate reports now expose `alias_drift_decision` as a first-class
  split. Remaining `unknown` rows are reported explicitly and should be treated
  as an audit coverage gap, not silently mixed into yaw/frame conclusions.

## What Reviewers Should Not Count As Final Proof

- Planner-only successful episodes.
- MP4s without matching trace and residual audit.
- Replay-only oracle probes as runtime controller evidence.
- Shadow MAE improvements without true privileged residual movement.
- Success-window or near-success windows as evidence for failure-tail recovery.
- Runtime artifacts that are not supplied with the git checkout.

## Current Bottlenecks

The main bottleneck is still semantic coverage and runtime readiness, not raw
model size.

The system has a better takeover contract now, but it still needs to prove that
non-privileged observations can reliably produce the jaw-local residual needed
for control across the hard failure-tail support surface.

Current hard points:

- runtime XY: the v25 affine calibrator is still the best runtime default, but
  it is a small-data smoke calibrator and should not be treated as final
  generalization evidence. The v34 wider affine and v36 direction-first MLP did
  not pass replacement A/B. Future estimator work must improve runtime
  contraction and overshoot, not only offline direction or MAE.
- `large_xy_large_yaw`: entry support has started to exist, but the active
  surface is not broad enough yet.
- `small_xy_large_yaw`: active support exists, but contraction and near-entry
  remain sensitive to step-size, horizon, and frame/sign conventions.
- yaw/frame: stable alias and frame drift are conceptually separated, but the
  split must appear consistently in candidate, trace, and aggregate audit.
- z/yaw readiness: current alignment lifecycle blocks planner close correctly,
  but it cannot hand off because task-frame `z_readiness` and `yaw_readiness`
  are not yet learned with non-privileged runtime inputs.
- close: must remain planner-owned and blocked until
  `alignment_ready_for_handoff` is true.

## Direction-Drift Review Questions

A useful external review should answer these questions:

1. Does runtime control remain non-privileged?
2. Are offline privileged probes clearly labeled as eval-only?
3. Is the project still focused on planner failure tails rather than already
   successful planner windows?
4. Does the code separate residual estimation, observability, and takeover
   tiering?
5. Does widening support for hard buckets preserve the control contract, or is
   it silently becoming a gate relaxation?
6. Does `xy_correction_ready` correctly expose bounded pullback opportunity
   without implying yaw control or close readiness?
7. Are `stable_alias_control` and `frame_drift_abstain` propagated through all
   relevant audits?
8. Is `small_xy_large_yaw` being diagnosed with the residual-aligned sign
   metric, not the older descent-cosine compatibility metric?
9. Are MP4 smoke tests being selected from active, contractive failure-tail
   rows rather than arbitrary failed episodes?
10. Are runtime XY estimator upgrades being accepted only after MP4 and
    hard-bucket runtime A/B, rather than offline MAE/cosine alone?
11. Is the next work aimed at solving task-frame z/yaw readiness and robust XY
    generalization, rather than loosening handoff/close gates?

## Recommended Next Work

The next useful implementation work is:

1. Use the alias-aware focused aggregates to reduce remaining `unknown` rows
   before drawing yaw/frame drift conclusions.
2. Continue widening `large_xy_large_yaw` entry support around episodes that
   already show active evidence.
3. Keep `small_xy_large_yaw` on a narrow step-size / horizon bracket and use
   `oracle_xy_step_cosine_to_residual` to separate step-too-small from true
   frame/sign errors.
4. Generate MP4 comparisons only from rows that are active and contractive in
   the audit, with trace summaries attached.
5. Align runtime gate traces with the formal takeover contract and keep
   privileged eval metadata under `offline_eval_only`.
6. Do not reopen runtime close or yaw control until the non-privileged
   estimator and audit evidence support it.
7. Keep v25 as the runtime XY default until a candidate improves both MP4
   runtime A/B and hard-bucket runtime A/B on contraction, near-entry, and
   overshoot.
8. Prioritize a non-privileged task-frame readiness dataset/model for z/yaw:
   C2C should hand off to planner gripper only when alignment is ready, not
   because sticky takeover expired.
