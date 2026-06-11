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
For the next method-level push beyond proxy correction, see
[`docs/C2C_V2_TASK_FRAME_ALIGNMENT_BREAKTHROUGH_PLAN.md`](C2C_V2_TASK_FRAME_ALIGNMENT_BREAKTHROUGH_PLAN.md).
For the post-v46 belief/forward-model route, see
[`docs/C2C_V2_BELIEF_FORWARD_MODEL_PLAN.md`](C2C_V2_BELIEF_FORWARD_MODEL_PLAN.md).

Operational defaults for this branch:

- Runtime environment: `conda run -n vla-adapter ...`
- Canonical RLBench/C2C runtime smoke path:
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py ...`
  Do not replace this default with `conda run -p`, direct env python, or
  `QT_QPA_PLATFORM=offscreen` when comparing against previous v42/v45/v46/v63
  smoke results; those are different launch paths and should only be used as
  explicit diagnostics.
- Fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`

Current objective:

- `v42_expanded_v4pilot` is now the active XY baseline.
- The concrete active XY checkpoint is
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`.
- `v42_expanded_v4pilot` is the baseline to beat, not solved XY. The next
  capability target is a unified non-privileged spatial-temporal task-frame
  estimator/controller that improves XY, Z, and Yaw alignment in parallel.
- The immediate engineering target is to keep strict
  `alignment_ready_for_handoff` as the only close predicate while replacing
  proxy-first correction with task-frame state estimation.
- The next named capability target should use semantic names, not another
  version-number label: `belief_forward_task_frame_candidate`.
  Its purpose is observability-aware belief estimation plus uncertainty-aware
  command-conditioned forward modeling, not a pure candidate-ranker loss sweep.
- The current readiness checkpoints are:
  - `runtime_artifacts/coarse2contact_v2/checkpoints/v43_task_frame_z_readiness.pt`
  - `runtime_artifacts/coarse2contact_v2/checkpoints/v44_task_frame_yaw_readiness.pt`
- The current readiness dataset is:
  - `runtime_artifacts/coarse2contact_v2/datasets/task_frame_readiness_v43_v44.jsonl`
- Close ownership has been tightened to planner-owned close plus C2C open-only
  safety. C2C contact/force stages may monitor contact, request open for
  safety, or emit a close recommendation for trace/debug, but they may not
  directly write a close command to `final_action[6]`.
- The unified gripper authority trace now records
  `planner_gripper_close_requested`, `planner_gripper_close_blocked`,
  `planner_gripper_handoff_allowed`,
  `planner_gripper_strict_handoff_ready`, `planner_gripper_handoff_latched`,
  `c2c_gripper_open_safety_requested`,
  `c2c_gripper_close_recommendation_ignored`, and
  `gripper_authority_source`.
- The latest review and smoke verification show the strict handoff gate still
  blocks planner close correctly under v43/v44 and the v46/v59 transition
  plumbing; `conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py`
  currently passes with `228` tests.
- Current v46 phase as of 2026-06-09: `v46_unified_task_frame_alignment_candidate`
  is implemented and has moved into wide source-held-out / random-held-out
  validation and retraining. It is not a promoted baseline. The latest broad
  source-held-out training artifact,
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_broad_plus_yawbalanced_holdout.pt`,
  used `6458` train rows across `176` train roots and `1240` held-out rows
  across `41` held-out roots. XY and Z show useful contraction evidence, but
  Yaw is still the weak axis and does not yet have enough reliable control
  evidence for promotion.
- The current v46 offline gate remains failed/pending, not promotable:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_summary.json`
  reports `offline_gate_status=fail` and
  `promotion_status=fail_offline_gate`, mainly because yaw-control evidence is
  insufficient on random5/yawbalanced slices.
- A wider yaw-collateral command-ranker LOO gate was generated after adding
  six selector-permitted hard roots (`root420`, `root490`, `root595`,
  `root700`, `root805`, `root910`) to the original z08 four-root pool:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_490_595_700_805_910_yaw_collateral_ranker_loo_smoke_summary.json`
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_ranker_loo_plus_root420_490_595_700_805_910_summary.json`.
  This is still a failed/pending candidate gate, not promotion evidence:
  `folds=10` now satisfies the minimum fold-count requirement, but the gate
  still fails with `worse_than_zero_folds=2`, worst-root
  `top1_minus_zero_combined_contraction=-1.0`, and worst-root
  `top1_minus_zero_yaw_contraction=-1.0`. The useful interpretation is that
  the blocker has moved from "not enough LOO roots" to the real algorithmic
  issue: the ranker still sometimes picks a yaw/Z command that is worse than
  doing nothing, and its average top-1 benefit over zero is too small. The next
  training objective must explicitly penalize worse-than-zero decisions and
  XY/Z collateral damage instead of treating raw top-1 contraction as enough.
  The new small-XY/large-Yaw roots are informative yaw-positive slices with
  clean close safety: `root700` retained `9` strict executed rows with
  observed XY/Z/Yaw contraction `0.0/0.0/1.0`; `root805` retained `9` rows
  with `0.0/0.111/0.889`; `root910` retained `9` rows with `0.0/0.0/1.0`.
  All three used the canonical `conda run -n vla-adapter xvfb-run -a python
  scripts/evaluate_c2c_v2_rlbench.py` path, had `close_leak_rows=0`, and
  recorded `uses_privileged_runtime=false`. This reinforces that v46 needs a
  joint multi-axis residual/controller objective, not a yaw-only selector.
- The v46 candidate ranker now includes a training-time zero/no-op guard loss:
  within a same-window command sweep, candidates that do not beat the observed
  zero command are margin-penalized if the model ranks them ahead of zero, and
  candidates that truly beat zero are allowed to outrank it. This uses only
  offline pre/post transition labels and does not change runtime inputs,
  strict handoff, or close authority. Focused verification:
  `conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q -k "candidate_ranker"`
  -> `9 passed`.
- Zero-guard LOO did not make v46 promotable. With the same 10-root pool,
  `zero_guard_weight=1.0` improved top-1 oracle match from `0.60` to `0.70`
  and top-1 Z contraction from `0.50` to `0.60`, but still had
  `worse_than_zero_folds=2` and the same worst-root
  `top1_minus_zero_combined/yaw=-1.0/-1.0`.
  `zero_guard_weight=5.0` did not fix the worst root and regressed the oracle
  match back to `0.60`. Reports:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_490_595_700_805_910_zero_guard_yaw_collateral_ranker_loo_smoke_summary.json`,
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_zero_guard_ranker_loo_plus_root420_490_595_700_805_910_summary.json`,
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_490_595_700_805_910_zero_guard_w5_yaw_collateral_ranker_loo_smoke_summary.json`, and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_zero_guard_w5_ranker_loo_plus_root420_490_595_700_805_910_summary.json`.
  Interpretation: no-op margin is necessary but not sufficient. The next model
  step should use a stronger multi-axis transition/outcome head or pairwise
  candidate-ranking objective that explicitly predicts beat-zero, per-axis
  collateral, and source-root uncertainty, rather than relying on a scalar
  residual score alone.
- The ranker support semantics were tightened to match runtime better:
  `command_support` is now trained against same-window beat-zero labels when a
  zero/no-op candidate is present, and LOO ranking can add a support penalty to
  low-support candidates before top-1 selection. Focused verification:
  `conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q -k "candidate_ranker"`
  -> `11 passed`. Full verification:
  `conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q`
  -> `221 passed`.
- Support-aware 10-root LOO still does not promote v46:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_490_595_700_805_910_support_aware_yaw_collateral_ranker_loo_smoke_summary.json`
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_support_aware_ranker_loo_plus_root420_490_595_700_805_910_summary.json`
  remain `offline_gate_status=fail`. The worst-root failure is unchanged
  (`top1_minus_zero_combined/yaw=-1.0/-1.0`). This rules out a simple
  support-target mismatch as the main cause. The next concrete model change
  should add explicit candidate outcome heads for `beats_zero`,
  per-axis contraction, per-axis collateral/worsen, and root uncertainty, then
  rank by a calibrated multi-term utility instead of a single residual score
  plus support.
- The candidate ranker now has an explicit command-outcome head:
  `TaskFrameV46AlignmentNet` emits `command_outcome_logits` for beat-zero,
  XY/Z/Yaw/combined contraction, and XY/Z/Yaw/combined worsen/collateral
  outcomes.  The ranker can train this head with `--outcome_loss_weight` and
  select candidates with `--rank_score_mode outcome_utility`.  This is still
  non-privileged at runtime: the outcome targets come only from offline
  pre/post transition labels, strict handoff is unchanged, and C2C close
  authority remains disabled.  Focused ranker verification now passes with
  `17` tests, and full verification passes with `228` tests:
  `conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q`.
- Outcome-utility 10-root LOO is the best ranker evidence so far, but it still
  fails the promotion gate:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_490_595_700_805_910_outcome_utility_ranker_loo_smoke_summary.json`
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_outcome_utility_ranker_loo_plus_root420_490_595_700_805_910_summary.json`.
  LOO metrics: `folds=10`, top-1 oracle match `0.70`, combined contraction
  `0.60`, Z contraction `0.60`, Yaw contraction `0.80`, versus zero/no-op
  combined/Z/Yaw `0.50/0.40/0.60`.  The formal gate remains
  `offline_gate_status=fail` and `promotion_status=fail_offline_gate` because
  one held-out fold is still worse than zero: the source root
  `large_xy_large_yaw_focus_flush_30k/chunk_000_000_007/eval` selects
  `z_guard_pos_0100` while the oracle is `yaw_hyp_neg_0100`, giving
  `top1_minus_zero_combined_contraction=-1.0` and
  `top1_minus_zero_yaw_contraction=-1.0`.
- Interpretation: v46 has moved from implementation into source-held-out /
  random-held-out validation and retraining.  The outcome head improves the
  average behavior and reduces bad held-out folds, but v46 is still a
  candidate, not a baseline.  The next model step should target the remaining
  z-positive versus yaw-negative confusion with source-root/domain uncertainty,
  pairwise oracle/zero losses, and more large-XY/large-Yaw flush roots before
  any MP4 promotion smoke.
- Current stage summary: v46 is in wide held-out validation and retraining,
  not framework bring-up and not baseline promotion. The next execution loop is
  short and strict:
  1. Expand source-held-out/random-held-out failure-tail coverage, with special
     attention to yaw-observable, partial-view, hard-bucket, and low-visibility
     windows.
  2. Run selector-permitted command sweeps from
     `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_selector_permitted_command_sweep_spec.jsonl`
     using the canonical `conda run -n vla-adapter xvfb-run -a python
     scripts/evaluate_c2c_v2_rlbench.py ...` path and the fixed planner
     checkpoint above.
  3. Audit true pre/post XY/Z/Yaw contraction, worsen/overshoot, yaw sign
     match, XY/Z collateral worsen, source-held-out split integrity, and close
     leak count.
  4. Retrain the dyaw/collateral ranker or unified v46 head only on expanded
     source-held-out evidence; do not tune only for random5/ep24 or any single
     window.
  5. Keep `alignment_ready_for_handoff` strict and C2C close authority disabled;
     v46 axis correction is allowed to move while handoff remains false, but it
     may not authorize close.
  6. Run 150/180-step MP4 with wrist view only after the offline/replay evidence
     shows credible three-axis contraction without close leaks.
- A new non-execution yaw-head audit was added at
  `scripts/audit_c2c_v2_task_frame_yaw_model_observability.py`. It reuses saved
  runtime observations and the v46 checkpoint to compare label-side yaw-visible
  seed rows against the model's yaw observability/confidence/ambiguity outputs;
  it does not run RLBench and does not add privileged runtime inputs. On the
  z08 yaw-visible seed pool, all `146` rows are label-side yaw-control rows,
  but the current broad+ yawbalanced v46 model only permits `43/146`
  (`29.45%`) as model-side yaw-control and blocks `103/146` (`70.55%`) as
  `model_yaw_not_observable`. Report:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_model_observability_audit.json`.
  Interpretation: before expanding command-sweep execution, fix yaw
  observability calibration/consistency so seed-visible windows are not
  systematically collapsed into runtime/model ambiguous windows.
- A first yaw-calibrated retraining smoke was run with weighted yaw
  observability/ambiguity losses:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_broad_plus_yawcal_holdout.pt`.
  It is not a promotable candidate. It improved the z08 yaw-visible seed-pool
  model-side yaw-control coverage from `43/146` to `74/146`, but it badly
  over-predicted yaw-control on the broader yawbalanced manifest:
  old broad+ yawbalanced model had `300/7193` model-yaw-control rows with
  `184` seed-noncontrol false positives; the yawcal smoke had `1819/7193`
  model-yaw-control rows with `1674` seed-noncontrol false positives. Raising
  the yaw-observable threshold to `0.8` still left `1625` false positives.
  Interpretation: the next yaw work should be precision-aware calibration or a
  separate yaw-control selector/ranker with explicit false-positive penalty,
  not a raw recall-weighted observability loss.
- The yaw model-observability audit now includes a threshold sweep over yaw
  observable/confidence/ambiguity thresholds. On the yawbalanced manifest, the
  old broad+ yawbalanced model can reach zero false positives only at unusably
  low recall (`2/146` true yaw-control rows). Its best-F1 point is the default
  region: `tp=116`, `fp=184`, `fn=30`, precision `0.3867`, recall `0.7945`.
  The recall-weighted yawcal smoke cannot be rescued by thresholding: its
  best-F1 sweep point still has precision only `0.1283` with `938` false
  positives. Reports:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yawbalanced_model_observability_old_sweep_audit.json`
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yawbalanced_model_observability_yawcal_sweep_audit.json`.
  This confirms the next implementation should be a precision-aware
  yaw-control selector/ranker trained with explicit false-positive cost, not
  another scalar threshold sweep over the existing yaw head.
- A first precision-aware yaw-control permission selector was added:
  `scripts/train_c2c_v2_task_frame_v46_yaw_control_selector.py` and
  `scripts/eval_c2c_v2_task_frame_v46_yaw_control_selector.py`. It is trained
  on frozen v46 runtime-visible outputs plus the same scalar
  geometry/readiness features already consumed by v46; privileged yaw-control
  labels are only offline targets. It does not output actions and records
  `close_control_allowed=false`.
  The head-only selector failed source-held-out precision (`precision=0.1837`,
  `fp=40`, `tp=9` on yawbalanced val), confirming that v46 yaw scores alone
  are not separable enough. Adding scalar geometry/readiness features produced
  a safer but low-recall selector:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_yaw_control_selector_scalar_precision_smoke.pt`.
  On yawbalanced source-held-out val it selected `tp=3`, `fp=0`, precision
  `1.0`, recall `0.2143`; on the z08 yaw-visible seed pool it selected
  `37/146` positives with precision `1.0`. Reports:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_control_selector_scalar_precision_smoke.json`
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_control_selector_scalar_precision_z08_eval.json`.
  Interpretation: this is a useful high-precision yaw permission seed, not a
  promotable v46 baseline. The next improvement must raise recall without
  losing the zero/near-zero false-positive behavior, likely by adding more
  source-held-out yaw-positive data and stronger spatial-temporal visual
  features rather than only v46 scalar/head outputs.
- A follow-up scalar+spatial selector added RGBD spatial moment features from
  the same non-privileged v46 wrist RGBD/depth-valid input. This substantially
  improved the yaw permission tradeoff:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_yaw_control_selector_scalar_spatial_precision_smoke.pt`.
  On yawbalanced source-held-out val it selected all `14/14` yaw-control
  positives with `3` false positives, precision `0.8235`, recall `1.0`, and
  FPR `0.00214`, satisfying the current precision/FPR selector gate. On the
  z08 yaw-visible seed pool it selected `117/146` rows, up from the scalar-only
  selector's `37/146`. Report:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_control_selector_scalar_spatial_precision_smoke.json`;
  z08 eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_control_selector_scalar_spatial_precision_z08_eval.json`.
  Interpretation: spatial RGBD evidence is materially useful for yaw
  observability/permission. This still does not promote v46.
- The scalar+spatial yaw-control selector is now wired into the canonical
  v46 runtime evaluator as an optional yaw micro-servo permission filter via
  `--task_frame_v46_yaw_selector_ckpt`. It only suppresses or permits the yaw
  step produced by the existing v46 micro-servo; it does not output actions,
  does not set `alignment_ready_for_handoff`, and records
  `task_frame_v46_yaw_selector_close_control_allowed=false`. Trace rows now
  include selector loaded/allowed/score/threshold/block-reason fields. The
  safety tests cover selector-denied yaw suppression, selector-allowed-but-
  history-unstable yaw suppression, and planner-owned close invariants. Current
  verification:
  `conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q`
  -> `214 passed`. Next evidence still required: rerun random held-out runtime
  traces with
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_yaw_control_selector_scalar_spatial_precision_smoke.pt`
  and prove improved yaw false-positive control plus three-axis residual
  contraction without close leaks.
- Selector-aware offline v46 evaluation was added to
  `scripts/eval_c2c_v2_task_frame_v46_alignment.py` through
  `--yaw_selector_checkpoint`, and the v46 gate summary now prefers
  selector-gated runtime yaw permission when those metrics exist. New reports:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_selector_aware_eval.json`,
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yawbalanced_selector_aware_eval.json`,
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_selector_aware_gate_summary.json`.
  Results:
  - z08 yaw-visible seed pool: selector permits `117/146` rows
    (`recall=0.8014`) with `precision=1.0` and `FPR=0.0`; raw v46 yaw control
    permission was only `0.2945`. However yaw bounded-step contraction remains
    `0.2671`, so the gate still fails on `yaw_observable_contraction_below_gate`.
  - yawbalanced manifest: selector permission rate is `0.0228` against target
    `0.0203`, with `precision=0.8537`, `recall=0.9589`, `FPR=0.0034`, and
    `24` false positives. This is a real permission improvement over the raw
    yaw head, but yaw bounded-step contraction is only `0.0170`; the slice also
    remains below the current yaw evidence-rate gate. Interpretation: the
  selector improves "should yaw move?" but does not solve "which dyaw step
  contracts the task-frame residual?". The next v46 work must collect more
  source-held-out yaw-positive transition data and train/calibrate the dyaw
  control/transition head itself, not just tune yaw observability thresholds.
- Added a yaw-transition effect audit:
  `scripts/audit_c2c_v2_task_frame_yaw_transition_effect.py`. It consumes only
  executed applied-transition/command-sweep manifests and uses true pre/post
  residuals strictly as offline labels. Current diagnostic pool:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_transition_effect_audit_current_pool.json`
  from the available v63 yaw-observable bootstrap, v63 additional roots, v62
  Z/Yaw diagnostic group000, and v59 random5 executed manifests. Key result:
  all executed yaw commands have only `0.4436` yaw contraction and `0.5564`
  yaw worsen, but oracle-best-per-window yaw choices reach `0.875` yaw
  contraction and `0.625` beat-zero-yaw. This means the environment/task frame
  does contain useful dyaw actions, but the current v46 policy/head is not yet
  selecting the right sign/magnitude. Sign diagnostics are also asymmetric:
  command-yaw same-sign-as-residual rows show `0.6391` yaw contraction versus
  `0.2606` for opposite-sign rows; `neg_0060`, `neg_0120`, and especially
  `neg_0240` are materially better than the corresponding positive buckets in
  this pool. However XY/Z collateral remains high (`best_per_group`
  `xy_collateral_worsen_rate=0.5`, `z_collateral_worsen_rate=0.375`), so the
  next dyaw learner must optimize yaw contraction jointly with XY/Z collateral,
  not just copy a sign rule.
- The existing v46 command-candidate ranker now has a `yaw_collateral` score
  mode and source/sequence-aware grouping, so candidate groups from different
  roots cannot be accidentally merged by identical `episode_idx + step_idx`.
  A tiny dyaw-collateral smoke was trained on the current executed yaw
  transition pool:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_dyaw_collateral_ranker_smoke.pt`
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_dyaw_collateral_ranker_smoke_train.json`.
  With source-root split, the available complete candidate supervision is still
  very small (`4` train groups / `1` val group). The val group selected the
  oracle candidate `zyaw_sym_p_hyp_neg_z0030_y0240`, improving over zero on
  yaw and combined contraction (`top1_yaw_contraction=1.0`,
  `top1_z_contraction=1.0`, `top1_combined_contraction=1.0`, while zero had
  `0.0/1.0/0.0`). This is a useful target-function smoke, not a generalization
  result. The next data action is to collect many more random-held-out,
  selector-permitted yaw-positive command-sweep groups with zero, yaw sign/
  magnitude, guarded Z+Yaw, and collateral-safe candidates before treating the
  dyaw ranker as runtime-ready.
- The selector-aware offline eval now writes row-level prediction dumps with
  `task_frame_v46_yaw_selector_score/allowed`, trace/runtime-observation
  pointers, and no privileged runtime inputs. The command-sweep builder now
  supports `selection_mode=yaw_selector_permitted`. Generated collection spec:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_selector_permitted_command_sweep_spec.jsonl`
  with summary
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_selector_permitted_command_sweep_spec.summary.json`.
  It contains `77` selector-permitted runtime windows across `30` source roots,
  `35` candidates per window, and `2695` executable candidate rows:
  zero, yaw `0.006/0.012/0.024` in both signs, Z guard
  `0.0015/0.003` in both signs, and Z+Yaw combinations. Dry-run command
  validation:
  `runtime_artifacts/coarse2contact_v2/v46_yaw_selector_permitted_command_sweep_dryrun/summary.json`.
  The dry-run uses the fixed canonical path
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py`
  and the fixed planner checkpoint
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`.
  This is the next executable data-collection target for widening dyaw
  transition supervision; it is not yet closed-loop success evidence.
- A canonical oldflow dry-run was regenerated for rows
  `2380,2450,2975,3920` at
  `runtime_artifacts/coarse2contact_v2/retest_oldflow_canonical_dryrun/summary.json`.
  The generated child commands use the fixed old path
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py ...`
  with the fixed planner checkpoint and front+wrist MP4 recording. In the
  current managed sandbox, actually launching the same RLBench smoke stops
  before evaluation with `NoWritableEnvsDirError`; do not treat that as v46
  runtime evidence. A real MP4/runtime rerun still needs the same command path
  on the normal RLBench-capable shell.
- A new offline gate summary lives at
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_summary.json`.
  It compares the older `yawobs_confcal_yawweighted` estimator with the newer
  v63 axis-v2 ranker checkpoint on random5 and oldflow slices. Result:
  `yawobs_confcal_yawweighted` passes the current offline state-estimator gate
  on both checked slices but remains `pending_runtime_insert_success`; the
  v63 `root_recovered_v40val` ranker checkpoint fails offline as a runtime
  state head because it predicts `xy_observable=false` when XY labels are
  observable and predicts yaw control on yaw-ambiguous/unobservable slices.
  Interpretation: keep state estimation and candidate ranking separate until
  the ranker is trained with explicit state-head calibration/regularization.
- The first v43/v44 handoff audit found 5 offline-ready rows in the selected
  random/generalization slices, all on the same random10 trace tail
  (`ep009`, steps `156-160`). Those source traces do not contain the current
  v43/v44 runtime handoff/readiness fields, so they should be treated as replay
  targets for current-runtime recall, not as proven current-runtime false
  negatives. The strict-smoke traces did show no planner-close leak: every
  close request was blocked.
- The project should not keep expanding diagnostics as the main work. With v42
  fixed as the current baseline and strict close safety intact, the
  highest-value work is now a unified task-frame alignment estimator that
  improves XY/Z/Yaw residual quality together.
- The v46/v58/v59 line currently has real command-outcome data plumbing, but
  it has not yet proven the final goal. The latest command-sweep work fixed an
  ep024-only data collection issue by adding random-heldout selection modes to
  the sweep spec builder. A random5 strictclose spec now covers
  `ep023`-`ep027` and emits `65` candidate command rows with
  `uses_privileged_runtime=false` and no privileged training labels. A
  random5 Z executed seed ran `10/10` child evaluator jobs, preserved
  front+wrist MP4s, retained `10` strict command-sweep-executed transition
  rows, and had `0` close leaks. In that small Z seed, observed Z contraction
  was `1.0`, XY contraction `0.8`, and yaw contraction `1.0`. This is real
  random-heldout transition evidence, not a promotion result: both Z candidates
  are relative offsets on top of a positive base command, and the model still
  has not learned a validated general three-axis selector.
- Command-sweep applied-transition manifests now support
  `require_command_sweep_executed=true`; the batch runner uses this when
  building command-sweep manifests so later natural `task_frame_v46_applied`
  rows from the same child run cannot contaminate candidate-command labels.
- The current combined executed-only command seed covers `ep023`-`ep027` with
  `22` retained rows, `0` close leaks, observed XY contraction `0.5455`, Z
  contraction `0.5000`, and yaw contraction `0.9545`. It remains too small and
  ep024-heavy for baseline promotion.
- A follow-up random5 X/Yaw seed ran `20/20` child evaluator jobs over
  `ep023`-`ep027`, preserving front+wrist MP4s and strict executed-only
  labels. It retained `20` rows with `0` close leaks, observed XY contraction
  `0.85`, Z contraction `1.0`, and yaw contraction `1.0`.
- The current balanced random5 executed-only seed combines Z plus X/Yaw
  candidates into `30` rows, exactly `6` per episode for `ep023`-`ep027`, with
  `0` close leaks. A CPU training smoke completed with
  `uses_privileged_runtime=false`, train rows `24`, val rows `6`, and val
  observed transition contraction of XY `0.6667`, Z `1.0`, and Yaw `1.0`.
  This is still a smoke, not a baseline: these windows have large Z residuals
  and strong base-command contraction, so they do not yet prove a selector that
  chooses correct near-contact high-precision actions.
- A near-contact random5 command sweep was then built from later windows
  (`ep023`-`ep026` step `95`, `ep027` step `98`) with residuals closer to the
  handoff basin and with `y_neg`, `y_pos`, and `zero` variants. It produced
  `15/15` child evaluator runs and preserved `15` front+wrist MP4s under
  `runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_random5_late_near_y_zero_seed`.
  The strict executed-only manifest retained `14` rows, had `0` close leaks,
  and reported observed contraction of XY `0.4286`, Z `0.1429`, and Yaw
  `0.2143`. Candidate split: `y_pos` had useful XY signal (`4/5`), while
  `y_neg` and `zero` mostly worsened XY; neither Z nor Yaw is solved in this
  near-contact sample. This is the strongest current evidence that the v46/v59
  line has working data plumbing but still lacks a validated high-precision
  task-frame selector/controller.
- The remaining late-near candidates were then executed for X, XY combined, Z,
  and Yaw (`50/50` child evaluator runs succeeded; `50` MP4s preserved). The
  manifest
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_random5_late_near_remaining_seed_manifest.jsonl`
  retained `50` strict command-sweep rows with `0` close leaks, exactly `10`
  rows per episode for `ep023`-`ep027`. Merging that with the earlier y/zero
  batch produced
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_random5_late_near_full_candidates_manifest.jsonl`
  with `64` retained rows (`ep027` zero did not retain an applied transition),
  `0` close leaks, XY contraction `0.3438`, Z contraction `0.2031`, Yaw
  contraction `0.1875`, and combined contraction `0.2969`. Candidate-level XY
  signal is visible (`xy_pp` XY contraction `1.0`, `xy_np` and `y_pos` `0.8`),
  but Z/Yaw remain weak and highly episode-dependent.
- A CPU training smoke on the 64-row late-near full-candidate manifest produced
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_random5_late_near_full_candidates_smoke.pt`
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_random5_late_near_full_candidates_smoke_train.json`.
  The root-held-out split kept train `51` / val `13` rows with
  `uses_privileged_runtime=false`. Val observed transition contraction was XY
  `0.5385`, Z `0.0`, Yaw `0.1538`. The learned effect-aware XY model predicted
  XY contraction `0.8462` on val, but the direct command-delta head still
  predicted XY contraction `0.0` / worsen `1.0`, Z predicted contraction `0.0`,
  and Yaw predicted contraction `0.4615`. Treat this checkpoint as a smoke and
  diagnostic artifact only, not a candidate baseline.
- A follow-up v60 candidate-ranker smoke added
  `scripts/train_c2c_v2_task_frame_v46_candidate_ranker.py`. It keeps the same
  non-privileged v46 runtime inputs and same `TaskFrameV46AlignmentNet`, but
  trains the command-transition output with same-window candidate ranking:
  for each episode+step group, predicted post-residual scores are compared
  across candidate commands and optimized to select the observed best
  post-residual. The smoke checkpoint/report are
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v60_random5_late_near_candidate_ranker_smoke.pt`
  and
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v60_random5_late_near_candidate_ranker_smoke_train.json`.
  With four train groups and one held-out group, the held-out group
  (`ep025:step0095`) selected `xy_np`, matching the oracle best candidate and
  improving XY contraction from zero/no-op `0.0` to top-1 `1.0`. Z and Yaw
  stayed at `0.0` on that held-out group. This shows ranking is the right
  direction for XY candidate choice, but it does not yet solve three-axis
  alignment or justify runtime promotion.

Success means:

- v42 XY remains non-regressed on random holdout, hard-bucket, old4/random5,
  low-visibility, and ep25/26 slices
- v43 Z readiness reaches high held-out precision on v42-XY-ready rows and is
  conservative at runtime
- v44 Yaw readiness blocks ambiguous / alias-drift rows instead of forcing them
  ready
- no reopening of `alignment_ready_for_handoff` or `close_ready`
- no direct C2C close ownership; every close command must be planner intent
  allowed by strict handoff or by an already latched prior strict handoff
- the runtime gate eventually recovers some of the offline-ready rows without
  reopening false positives

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
  alignment, aperture-to-spoke alignment, guarded slide, recovery, contact
  monitoring, and open-only gripper safety.
- The planner owns close intent. C2C can allow that intent only through
  `alignment_ready_for_handoff`; after a valid handoff, later closed-gripper
  hold still comes from planner intent and can still be overridden open by C2C
  safety.
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

## 2026-06-06 v46 Task-Frame Alignment Status

The current branch now contains a v46 unified task-frame alignment candidate,
but it is not promoted to baseline.

What is implemented:

- `TaskFrameV46AlignmentNet` and runtime calibration/wrapper live in
  `prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py`.
- v46 consumes runtime-visible wrist/front RGBD, depth validity through RGBD
  preprocessing, proprio/planner prior, trace scalar features, and temporal
  history. It does not consume privileged runtime pose/mask/object handles.
- v46 outputs parallel `dx/dy/dz/dyaw`, per-axis confidence/observability,
  yaw ambiguity/unobservability, step scale, and risk reason.
- v46 runtime evaluator wiring is behind `--task_frame_v46_ckpt` and
  `--enable_v46_task_frame_micro_servo`.
- v46 has no close authority. The evaluator trace keeps
  `task_frame_v46_close_control_allowed=false`, and planner close still goes
  through the strict handoff arbiter.
- Runtime correction activation is explicitly traceable through
  `task_frame_v46_activation_ready` and
  `task_frame_v46_applied_local_6d`.

Current evidence:

- Unit/smoke tests: `181 passed` under `conda run -n vla-adapter pytest -q
  tests/test_coarse2contact_v2.py`.
- Offline random5 strict holdout for
  `runtime_task_frame_alignment_v46_unified_candidate_yawobs_confcal_yawweighted.pt`
  is strong: combined contraction `1.0000`, XY contraction `0.9722`, Z
  contraction `0.8312`; yaw is correctly blocked because the slice has no
  yaw-observable rows.
- Offline yaw-balanced eval shows yaw can contract when observable:
  yaw-observable yaw contraction `0.9726`, unobservable yaw worsen `0.0`.
- Runtime random5 MP4 smoke confirms v46 can actually activate and write
  bounded steps:
  `runtime_artifacts/coarse2contact_v2/v46_unified_runtime_smoke_random5`.
  This run had close leak count `0`, Z contraction `0.8333`, and combined
  contraction `0.9010`, but XY contraction was only `0.6510` with XY worsen
  `0.3490`; yaw remained blocked as unobservable.
- Risk-scaled and hybrid XY follow-up smokes did not solve XY. Risk scaling
  improved Z safety but did not fix XY semantics. Hybrid v42-XY + v46-Z/Yaw
  preserved close safety but produced weaker combined contraction because many
  windows became Z-only while XY drift persisted.

Current interpretation:

- v46 is a real non-privileged estimator/controller scaffold and a useful
  diagnostic candidate.
- It has not proven high-precision three-axis takeover or insert success.
  Runtime insert success on the random5 v46 smokes remains `0/5`.
- The next model breakthrough should focus on task-frame state semantics,
  especially XY under partial view/approach-angle changes and yaw-observable
  data collection. Do not spend the next cycle on close threshold relaxation.

Continuation evidence:

- A pure v46 random5 relabel was generated at
  `runtime_artifacts/coarse2contact_v2/relabels/v46_unified_runtime_smoke_random5_frame_residual_v2/frame_residual_v2.jsonl`.
  It produced `827` valid labels from `900` rows and kept runtime invariants:
  `uses_privileged_runtime=false`.
- An online-feedback manifest was built:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_online_feedback_random5_manifest.jsonl`
  with `698` retained rows.
- The v46 training script now supports optional transition labels
  (`next_residual`, `has_next_residual`) and adds transition-aware loss when
  offline next residuals are available.
- Diagnostic checkpoint
  `runtime_task_frame_alignment_v46_transition_feedback_candidate.pt` did not
  solve XY. On the online-feedback manifest, `xy_sign_match=0.4828` and
  `xy_bounded_step_contraction=0.1791`.
- New control-effect audit
  `scripts/audit_c2c_v2_task_frame_control_effect.py` shows why: the closed-loop
  local command to task-frame residual-delta mapping is not the simplified
  offline model. In pure v46 ep024, y-axis command/true-delta correlation was
  `-0.9485` and same-sign rate was `0.0`. The empirical XY response also has
  cross-axis terms.

Updated interpretation:

- The next breakthrough should not be another residual-only v46 checkpoint.
  The system needs a control-effect/Jacobian-aware task-frame controller,
  trained from non-runtime-privileged transition sidecars, so bounded local
  steps are selected by predicted post-step residual rather than by assuming
  `post = residual - step`.

## 2026-06-06 v47 Control-Effect Candidate Status

The branch now contains the first control-effect/Jacobian-aware extension of
the v46 task-frame estimator. It is useful, but it is not promoted.

Implemented:

- `TaskFrameV46AlignmentNet` predicts `task_frame_v46_xy_control_effect`, a
  bounded 2x2 local-XY-command to task-frame-XY-residual-delta matrix.
- `--v46_task_frame_xy_mode effect_aware` in
  `scripts/evaluate_c2c_v2_rlbench.py` uses that matrix to choose a bounded XY
  correction.
- The effect-aware XY helper now obeys v46 evidence/risk softgating, so
  `direction_conflict`, low support, or low confidence cannot bypass into a
  full XY step.
- No close authority was added. Trace keeps
  `task_frame_v46_close_control_allowed=false`, and planner close remains owned
  by strict handoff.

Verification:

- `conda run -n vla-adapter python -m py_compile
  prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py
  scripts/evaluate_c2c_v2_rlbench.py`
- `conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py`
  now reports `183 passed`.

Evidence:

- Offline checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v47_control_effect_bounded_candidate.pt`.
- Offline metrics looked promising: online-feedback effect-aware XY contraction
  `0.8840`, strict-holdout effect-aware XY contraction `1.0000`.
- Focused runtime ep024 smoke without effect-risk guard:
  `runtime_artifacts/coarse2contact_v2/v47_control_effect_runtime_smoke_ep024`.
  It produced a wrist/front MP4 at `videos/ep024_fail.mp4`, applied v46/v47 on
  `40` eval-label rows, and kept close leak count `0`; however XY contraction
  was only `0.2000` with XY worsen `0.8000`.
- The failure diagnosis is semantic, not close-related: online v47 estimated
  task-frame `x` with the wrong sign on most applied rows and saturated `dy` at
  `+0.04`, so the control-effect solver optimized against the wrong residual.
- Guarded runtime ep024 smoke:
  `runtime_artifacts/coarse2contact_v2/v47_control_effect_runtime_smoke_ep024_guarded`.
  It produced `videos/ep024_fail.mp4`, applied on `125` eval-label rows, kept
  close leak count `0`, improved XY from harmful to roughly neutral
  (`0.4400` contraction, `0.5600` worsen, mean XY norm delta `-0.000036`), kept
  some Z contraction (`0.6480`), and still had `0` yaw-allowed rows.

Current interpretation:

- v47 confirms that the runtime controller needs a control-effect model, but
  also shows that a control-effect head cannot compensate for wrong
  non-privileged task-frame residual semantics.
- The immediate blocker is still state estimation: XY is not solved, Z is only
  partially useful, and yaw is mostly unobservable in the current random
  failure-tail runtime slice.
- Do not upgrade v47/effect-aware to baseline and do not claim random-held-out
  three-axis contraction or insert success from this checkpoint.
- The next dataset/model cycle should prioritize stronger object/task-frame
  representation, yaw-observable slice collection, symmetry-aware yaw labels,
  and transition labels from diverse random failure tails.

## 2026-06-06 v48/v49 Residual Range Status

The branch now fixes a real model-capacity bug, but the resulting candidates
still do not pass runtime promotion.

Implemented:

- v46/v47 checkpoints now serialize residual output support:
  `max_abs_xy`, `max_abs_z`, and `max_abs_yaw`.
- Training now instantiates `TaskFrameV46AlignmentNet` with label-support
  ranges, avoiding the previous mismatch where labels could be admitted up to
  `8cm` while the model output saturated at `4cm`.
- Offline metrics now include `xy_predicted_effect_aware_contraction`, which
  uses model-predicted residuals and is closer to runtime than the oracle
  `xy_effect_aware_contraction` metric.
- Unit tests now report `184 passed`.

Evidence:

- `runtime_task_frame_alignment_v48_range_matched_candidate.pt` uses local
  support (`xy=0.080`, `z=0.080`, `yaw=0.350`). It improves strict local
  holdout metrics, but the online-feedback random5 rows are outside this
  support and are filtered out. Therefore v48 is local-support evidence only.
- `runtime_task_frame_alignment_v49_runtime_tail_range_candidate.pt` expands
  support (`xy=0.400`, `z=0.700`, `yaw=0.450`) and includes the online-feedback
  runtime tail rows.
- v49 online-feedback eval:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v49_runtime_tail_range_candidate_online_feedback_eval.json`
  shows `xy_predicted_effect_aware_contraction=1.0000`,
  `xy_sign_match=0.9993`, and `z_bounded_step_contraction=0.9799`.
- v49 strict random5 holdout:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v49_runtime_tail_range_candidate_holdout_random5_strict_eval.json`
  shows `xy_predicted_effect_aware_contraction=0.9054`, but yaw remains weak
  (`yaw_sign_match=0.3729`).
- v49 focused runtime ep024:
  `runtime_artifacts/coarse2contact_v2/v49_runtime_tail_range_smoke_ep024`.
  MP4 is at `videos/ep024_fail.mp4`; trace is at
  `gripper_traces/ep024_gripper_trace.jsonl`.
  Close leak count remained `0`; planner close was requested `28` times and
  all were blocked by strict handoff. However v49 applied on only `9`
  eval-label rows, mean XY norm worsened by `0.00374m`, combined contraction
  was `0.4444`, and yaw had `0` allowed rows.

Current interpretation:

- The output-range serialization fix should stay.
- v48/v49 are candidates only, not baselines.
- Expanding residual support is not enough to prove random-tail recovery.
  Current v49 offline gains do not transfer cleanly to ep024 runtime.
- The next decisive step is a source-held-out runtime-tail dataset and a
  stronger object/task-frame representation, not another threshold or close
  change.

## 2026-06-06 v50/v51 Multisource Tail Status

The branch now has a real multisource runtime-tail manifest and two follow-up
candidates. They improve the evidence chain, but they still do not pass runtime
promotion.

Dataset:

- Manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v50_runtime_tail_multisource_manifest.jsonl`
- Summary:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v50_runtime_tail_multisource_manifest.summary.json`
- Retained rows: `25,454`
- Source roots: `191`
- Episodes: `30`
- Yaw observable rows: `3,790`

v50:

- Checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v50_multisource_tail_candidate.pt`
- Root-held-out validation:
  - train `20,294` rows from `153` roots
  - val `5,160` rows from `38` roots
  - `xy_sign_match=0.9338`
  - `z_sign_match=0.9618`
  - `yaw_sign_match=0.8147`
  - `xy_predicted_effect_aware_contraction=0.8905`
  - `z_bounded_step_contraction=0.9498`
- v50 focused ep024 runtime:
  `runtime_artifacts/coarse2contact_v2/v50_multisource_tail_smoke_ep024`.
  The MP4 is `videos/ep024_fail.mp4`.
  Close leak count stayed `0`, but v50 applied on `0` eval-label rows because
  estimated XY remained outside activation radius even when offline labels
  showed true XY was near.

v51:

- Added a learned near-field activation head. This head gates bounded
  correction only; it does not affect strict handoff or close.
- Runtime trace fields include
  `task_frame_v46_near_field_confidence`,
  `task_frame_v46_learned_near_field_ready`, and
  `task_frame_v46_radius_ready`.
- Train/eval near-field label radii are configurable via
  `--near_field_xy_radius` and `--near_field_z_radius`.
- Checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v51_nearfield_multisource_tail_candidate.pt`
- Root-held-out validation:
  - near-field accuracy `0.9670`
  - near-field recall `0.9476`
  - predicted positive rate `0.2325` vs true positive rate `0.2228`
  - `xy_predicted_effect_aware_contraction=0.8779`
- v51 focused ep024 runtime:
  `runtime_artifacts/coarse2contact_v2/v51_nearfield_multisource_tail_smoke_ep024`.
  The MP4 is `videos/ep024_fail.mp4`.
  Close leak count stayed `0`, but applied rows remained `0`; learned
  near-field ready rows were `0` and max near-field confidence was `0.2013`.

Current interpretation:

- v50/v51 are not baselines.
- The source-held-out offline metrics are now meaningful, but ep024 runtime
  still exposes an object/task-frame representation gap.
- Near-field gating is a useful mechanism to keep, but it needs ep024-like
  positive windows and stronger visual features to generalize.
- Continue to preserve strict handoff and planner-owned close; the bottleneck
  remains pre-close task-frame state estimation.

## 2026-06-07 v52 Spatial Moments Status

v52 adds a lightweight non-privileged spatial geometry branch to the v46/v51
model family.

Implemented:

- `task_frame_v46_spatial_moment_features(...)` computes depth-valid,
  near-depth/far-depth centroid, second-moment, depth-statistic, and RGB-mean
  features from runtime-visible RGBD/coord channels.
- `TaskFrameV46AlignmentNet(use_spatial_moments=True)` concatenates these
  features into the fusion trunk.
- The setting is checkpoint-serialized; old checkpoints keep
  `use_spatial_moments=false`.
- Unit tests still pass: `184 passed`.

Checkpoint:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v52_spatial_moments_tail_candidate.pt
```

Root-held-out validation:

- `xy_sign_match=0.9399`
- `z_sign_match=0.9557`
- `yaw_sign_match=0.8000`
- `xy_predicted_effect_aware_contraction=0.8955`
- `z_bounded_step_contraction=0.9420`
- near-field accuracy `0.9679`
- near-field recall `0.8745`

Runtime evidence:

- Focused ep024:
  `runtime_artifacts/coarse2contact_v2/v52_spatial_moments_tail_smoke_ep024`.
  MP4: `videos/ep024_fail.mp4`.
  v52 applied on `22` eval-label rows, close leak count was `0`, XY
  contraction was `0.5909`, combined contraction was `0.5455`.
- Random5:
  `runtime_artifacts/coarse2contact_v2/v52_spatial_moments_tail_smoke_random5`.
  MP4s are available under `videos/ep023_fail.mp4` through
  `videos/ep027_fail.mp4`.
  v52 applied on `89` eval-label rows, close leak count was `0`, insert
  success was `0/5`, total XY contraction was `0.3933`, total combined
  contraction was `0.6067`, and Z contraction was `0.5506`.

Current interpretation:

- v52 fixes an important activation failure: C2C now opens on ep024-style
  near-field windows where v50/v51 stayed silent.
- v52 does not solve high-precision alignment. Random5 XY contraction is still
  below acceptable promotion criteria, with y-axis worsen still prominent.
- The next priority is online XY control-effect / axis-coupling accuracy on
  applied runtime rows. Do not relax close/handoff to compensate.

## 2026-06-07 v53/v54 Applied-Transition Status

This branch now has an explicit applied-transition learning path for the v46
unified task-frame line. The close contract did not change: C2C still has no
close authority, and planner close is only allowed through strict
`alignment_ready_for_handoff`.

New implementation:

- `scripts/build_c2c_v2_task_frame_applied_transition_manifest.py`
  converts runtime smoke traces into pre/action/post rows while keeping true
  residuals as offline labels only.
- v46/v53 training now prefers `applied_control_command_xy` for control-effect
  supervision and reports x/y split effect metrics.
- Runtime v46 activation no longer lets learned near-field confidence bypass
  the estimated XYZ radius gate by default. This prevents correction from
  opening far outside the local task-frame region.

Artifacts:

- v53 applied manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v53_applied_transition_v52_manifest.jsonl`
- v54 on-policy manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v54_applied_transition_v53_radius_guarded_manifest.jsonl`
- v53 checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v53_applied_transition_control_effect_candidate.pt`
- v54 checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v54_onpolicy_xy_effect_candidate.pt`

Key results:

- v53 improves offline metrics on the 111-row v52 applied-transition manifest
  (`xy_predicted_effect_aware_contraction` from `0.8018` to `0.9189`), but the
  first runtime smoke opened correction too early because learned near-field
  was overconfident far outside local support.
- After radius guarding, focused ep024 no longer opens early: the first
  applied row moves from step `20` with true z about `0.598m` to step `66` with
  true z about `0.0345m`.
- close leak count remains `0` in both v53 smokes.
- Radius-guarded v53 focused ep024:
  `runtime_artifacts/coarse2contact_v2/v53_applied_transition_control_effect_smoke_ep024_radius_guarded/videos/ep024_fail.mp4`
  has `36` applied rows, XY contraction `0.0`, Z contraction `0.6944`, yaw eval
  contraction `0.4722`, combined contraction `0.6667`.
- v54 on-policy feedback did not fix the core issue: on the guarded 36-row
  manifest, observed y-axis contraction remains `0.0` even though the model
  still predicts effect-aware contraction.

Current interpretation:

- v53/v54 are not baselines and should not be promoted.
- The latest evidence narrows the bottleneck: activation and close safety are
  now better controlled, but the current 2x2 XY control-effect head cannot
  model online local-command-to-task-frame residual dynamics.
- The next candidate should be a command-conditioned transition/Jacobian model
  over the full local 6D command, current task-frame estimate, wrist RGBD,
  proprio, and temporal motion. It must predict actual
  `delta(dx,dy,dz,dyaw)` plus uncertainty for the bounded command before the
  controller trusts a step.

## 2026-06-07 v55/v56 Command-Transition Status

The v46 model family now has a command-conditioned transition head. This is a
candidate mechanism only; it is not a baseline.

Implemented:

- `TaskFrameV46AlignmentNet` accepts an optional `command_6d` and predicts
  `command_delta`, `command_logvar`, and `command_support`.
- `TaskFrameV46AlignmentCalibration.predict_command_transition_from_trace(...)`
  exposes transition prediction using runtime-visible RGBD/proprio/history and
  the candidate full local 6D command. It does not provide close authority.
- Runtime mode `--v46_task_frame_xy_mode transition_guarded_effect_aware`
  suppresses effect-aware XY when the transition model predicts post-command
  XY worsen.

Artifacts:

- Multi-run applied-transition manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v56_applied_transition_multirun_manifest.jsonl`
- v55 checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v55_command_transition_candidate.pt`
- v56 checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v56_applied_heavy_command_transition_candidate.pt`
- v56 focused MP4:
  `runtime_artifacts/coarse2contact_v2/v56_transition_guarded_smoke_ep024/videos/ep024_fail.mp4`

Key evidence:

- The multi-run applied-transition manifest has `1772` rows from `9` source
  roots and close leak count `0`.
- v55 did not fix the guarded ep024 y-axis sign problem.
- v56 does identify the known harmful XY command on guarded ep024:
  `command_xy_predicted_contraction=0.0` and
  `command_xy_predicted_worsen=1.0`.
- Runtime v56 transition guard suppresses XY on all `95` transition-valid
  applied rows in focused ep024. close leak remains `0`.
- Runtime v56 focused ep024 still fails high-precision alignment:
  XY contraction `0.2000`, Z contraction `0.6211`, combined contraction
  `0.6526`, insert success still absent.

Updated blocker:

- The transition guard can prevent a known bad XY action, but it cannot rescue
  a bad activation predicate.
- v56 opens too early: first applied row is step `36`, with offline true z
  about `0.364m`, while the model estimates z at about `0.033m`.
- The next bottleneck is therefore the non-privileged near-field/progress
  estimator, especially Z/progress-to-contact false positives. Do not run a
  random5 promotion MP4 until activation false positives are reduced.

## 2026-06-07 v57/v58 Near-Field And Command Search Status

The v46 family now has a safer near-field/transition-controller path, but it is
still not a baseline and has not proven random held-out three-axis contraction.

Implemented:

- `near_field_head` and near-field/progress manifest support.
- `runtime_task_frame_alignment_v57_nearfield_progress_guard_candidate.pt`
  reduced the known far-Z false positive offline, but became too conservative
  online.
- `runtime_task_frame_alignment_v58_nearfield_onpolicy_candidate.pt` restored
  on-policy near-field recall, but without transition guarding it could still
  push Z in the wrong direction.
- `TaskFrameV46CommandSearchResult` and
  `task_frame_v46_transition_command_search(...)` now let runtime search a
  small bounded local command set with the command-conditioned transition head.
- Runtime `transition_guarded_effect_aware` now uses axis-strict command search:
  every moved axis must predict contraction. Combined-score improvements cannot
  hide XY/Z/Yaw worsen.

Key artifacts:

- v58 near-field on-policy manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v58_nearfield_onpolicy_manifest.jsonl`
- v58 near-field checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v58_nearfield_onpolicy_candidate.pt`
- v58 XYZ guard MP4:
  `runtime_artifacts/coarse2contact_v2/v58_transition_xyz_guard_smoke_ep024/videos/ep024_fail.mp4`
- v58 axis-strict command-search MP4:
  `runtime_artifacts/coarse2contact_v2/v58_transition_command_search_axisstrict_smoke_ep024/videos/ep024_fail.mp4`
- axis-strict audit:
  `runtime_artifacts/coarse2contact_v2/reports/v58_transition_command_search_axisstrict_smoke_ep024_audit.json`

Latest focused ep024 evidence:

- Unit tests: `190 passed`.
- Axis-strict command search: `6` search-valid rows, `0` search-applied rows.
- Active-axis contraction violations: `0`.
- True close leak count using gripper index `7`: `0`.
- All search rows returned `no_candidate_improves_no_correction`.
- Eval-label all-row contraction in the focused smoke: XY `0.2105`,
  Z `0.6947`, yaw `0.4211`, combined `0.7368`.

Interpretation:

- Close ownership remains closed: C2C does not regain close authority, and strict
  handoff remains the only close/handoff path.
- The axis-strict selector is doing the right safety thing: it suppresses
  commands when the transition model cannot certify per-moved-axis contraction.
- The current transition head is still too weak/narrow to choose useful
  commands. It is mostly a suppressor, not yet a high-precision three-axis
  controller.
- Next work must collect and train on broader on-policy/off-policy candidate
  command sweeps over random held-out failure tails, so the model learns to rank
  bounded candidate commands rather than only detect known bad on-policy moves.

## 2026-06-07 v59 Command-Sweep Spec Status

The branch now has a safe command-sweep data-collection spec builder. This is
not a trained model and not a promotion result; it is the next data path needed
to train a transition head that can choose among candidate actions.

Implemented:

- `scripts/build_c2c_v2_task_frame_command_sweep_spec.py`
- The builder selects runtime-near `RING_GRASP_ALIGN` rows from existing
  runtime traces and expands each selected row into bounded local candidate
  commands.
- It does not create fake transition labels for unexecuted commands.
- It strips privileged pre/post residual trace fields and marks
  `uses_privileged_label_for_training=false`.
- Unit tests now report `191 passed`.

Artifact:

- `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.jsonl`
- `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_spec_ep024.summary.json`

Summary:

- selected runtime rows: `6`
- candidate commands per runtime row: `13`
- retained spec rows: `78`
- `uses_privileged_runtime=false`
- `uses_privileged_label_for_training=false`

Execution hook:

- `scripts/evaluate_c2c_v2_rlbench.py` now accepts
  `--task_frame_v46_command_sweep_spec_jsonl` and
  `--task_frame_v46_command_sweep_row_index`.
- It executes exactly one selected candidate command at the matching
  `episode_idx + step_idx`, overriding only bounded local 6D motion.
- It does not grant close authority or handoff.

First execution smoke:

- Output root:
  `runtime_artifacts/coarse2contact_v2/v59_command_sweep_exec_ep024_row000`
- MP4:
  `runtime_artifacts/coarse2contact_v2/v59_command_sweep_exec_ep024_row000/videos/ep024_fail.mp4`
- Applied-transition manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_exec_ep024_row000_manifest.jsonl`
- Selected spec row `0`, candidate `x_neg`.
- Command-sweep active rows: `1`.
- Command-sweep executed rows: `1`.
- True close leak count using gripper index `7`: `0`.
- Applied-transition manifest retained rows: `1`, close leak rows `0`.
- The executed command contracted XY and yaw eval residual, worsened Z, and
  contracted the combined score.

Next implementation step:

- Batch command-sweep execution over enough spec rows/random held-out
  failure-tail episodes, then train the next transition model on those executed
  sweep transitions.

Batch runner:

- Added `scripts/run_c2c_v2_task_frame_command_sweep_batch.py`.
- It launches one independent evaluator run per selected sweep-spec row.
- It supports row/candidate/episode filters, `--dry_run`, `--max_parallel`,
  and GPU cycling via `--gpus`.
- It can optionally build a combined applied-transition manifest from
  successful run roots.
- Defaults preserve the fixed planner checkpoint, v42 XY baseline, v58
  task-frame checkpoint, front+wrist MP4, runtime observations, and strict
  close/handoff.

Verification:

- Unit tests: `192 passed`.
- Dry-run summary:
  `runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_dryrun_ep024_rows000_001.json`
- The dry run selected rows `0,1`, assigned GPUs `0,1`, and produced valid
  child evaluator commands for candidates `x_neg` and `x_pos`.

Real small-batch execution:

- Output root:
  `runtime_artifacts/coarse2contact_v2/v59_command_sweep_batch_ep024_rows001_002`
- Batch summary:
  `runtime_artifacts/coarse2contact_v2/reports/v59_command_sweep_batch_ep024_rows001_002.json`
- Applied-transition manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_batch_ep024_rows001_002_manifest.jsonl`
- Executed candidates: `x_pos`, `xy_nn`
- Successful child runs: `2/2`
- Applied-transition retained rows: `2`
- close leak rows: `0`
- observed XY contraction `0.0`, Z contraction `0.0`, yaw eval residual
  contraction `1.0`
- MP4s with front+wrist view were preserved for both child runs.

Seed training smoke:

- Combined seed manifest:
  `runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v59_command_sweep_seed_ep024_rows000_002_manifest.jsonl`
- Smoke checkpoint:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v59_command_sweep_seed_smoke.pt`
- Smoke report:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v59_command_sweep_seed_smoke_train.json`
- Train rows `2`, val rows `1`, command-transition rows recognized in both
  splits, `uses_privileged_runtime=false`.

Status:

- The executed command-sweep data loop is now functional end to end.
- The seed checkpoint is a pipeline smoke only and is not a candidate.
- The next useful step is to scale command-sweep execution over many rows and
  random held-out failure tails, then train a real transition model.

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

## 2026-06-08 Status: v61 Hard-Bucket Command Sweep Completed

Current default constraints remain unchanged:

- planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- runtime environment: `conda run -n vla-adapter ...`
- fixed XY baseline:
  `runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt`
- strict handoff remains enforced
- C2C close authority remains disabled; C2C may not directly close
- runtime privileged input remains forbidden

v61 collection artifacts:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v61_hardbucket_near_contact_ranker_spec.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v61_command_sweep_hardbucket_full_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v61_command_sweep_hardbucket_full_manifest.summary.json
```

Execution summary:

- candidate groups: `16`
- candidate rows: `208`
- executed rows: `208 / 208`
- front+wrist MP4s preserved: `208`
- close leak rows: `0`
- `uses_privileged_runtime_any=false`
- privileged residuals are offline pre/post labels only

Observed full-manifest contraction:

- XY: `0.2452`
- Z: `1.0000`
- Yaw: `0.0048`
- combined: `0.7788`

Candidate signals:

- XY has repeated usable candidates:
  - `xy_np` XY contraction `1.0000`
  - `xy_pp` XY contraction `0.8125`
  - `y_pos` XY contraction `1.0000`
  - `zero` XY contraction `0.0000`
- Z is contaminated by natural drift/descent in this slice:
  - `z_pos` Z contraction `1.0000`
  - `zero` Z contraction `1.0000`
- Yaw remains effectively unsolved:
  - `yaw_pos` Yaw contraction `0.0625`
  - `yaw_neg` Yaw contraction `0.0000`

Ranker smoke artifacts:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v61_hardbucket_candidate_ranker_z016_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v61_hardbucket_candidate_ranker_z016_smoke_train.json
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v61_random5_hardbucket_candidate_ranker_z016_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v61_random5_hardbucket_candidate_ranker_z016_smoke_train.json
```

Hard-bucket ranker smoke with expanded Z support:

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

Combined random5 + hard-bucket ranker smoke:

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

- v61/v46 is not a new baseline.
- v42 remains the XY baseline.
- v61 checkpoints are smoke artifacts only.
- The current evidence supports candidate ranking for XY, but does not prove
  three-axis task-frame contraction or insert success improvement.

Next required work:

- collect broader random held-out command sweeps, not only hard-bucket windows;
- make group splitting source/session held-out, not just `episode_idx+step_idx`;
- redesign yaw data with symmetry-aware, yaw-observable positive intervention
  windows;
- redesign Z labels/evaluation to separate true command effect from planner
  descent / natural residual drift;
- only after held-out top-1 beats zero/no-op on XY, Z, Yaw, combined contraction
  and worsen risk should closed-loop MP4/success promotion be attempted.

## 2026-06-08 Status: v62 Z/Yaw Diagnostic Setup

New tools:

```text
scripts/audit_c2c_v2_task_frame_zero_adjusted_effect.py
```

`build_c2c_v2_task_frame_command_sweep_spec.py` now also supports:

```text
--candidate_profile z_yaw_diagnostic
--z_steps <comma-separated magnitudes>
--yaw_steps <comma-separated magnitudes>
```

Purpose:

- separate real Z/Yaw command effect from same-window natural drift;
- mine yaw-positive / yaw-negative intervention windows;
- avoid training v46/v61 on raw contraction signals where `zero` already
  contracts.

v61 zero-adjusted audit:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v61_hardbucket_zero_adjusted_effect_audit.json
```

- audited groups: `16`
- close leak rows: `0`
- uses privileged runtime: `false`
- overall beats-zero rates:
  - XY: `0.4471`
  - Z: `0.4567`
  - Yaw: `0.4615`
  - combined: `0.4567`
- useful candidate signs:
  - `xy_np`, `xy_pp`, `y_pos` still beat zero on XY
  - `z_pos` beats zero on Z
  - `yaw_neg` beats zero on Yaw more often than `yaw_pos`, but mean effect is
    very small

v62 diagnostic spec:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_spec.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_spec.summary.json
```

- selected windows: `8`
- candidate rows: `280`
- candidates per window: `35`
- XY candidates disabled
- Z magnitudes: `0.0015`, `0.0030`
- Yaw magnitudes: `0.006`, `0.012`, `0.024`

v62 group000 execution:

```text
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group000
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_group000_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_group000_zero_adjusted_effect_audit.json
```

- executed candidates: `35 / 35`
- front+wrist MP4s preserved: `35`
- close leak rows: `0`
- raw Yaw contraction: `0.0`
- zero-adjusted beats-zero rates:
  - XY: `0.5000`
  - Z: `0.4706`
  - Yaw: `0.5882`
  - combined: `0.4706`

Best diagnostic signals in group000:

- `zyaw_pn_z0030_y0240`: best combined zero-adjusted effect
- `zyaw_pn_z0030_y0120`: second-best combined zero-adjusted effect
- `z_pos_0030`: clean Z effect relative to zero
- `yaw_neg_0240`: yaw-only command with clear yaw effect relative to zero
- `yaw_pos_*`: worsened yaw relative to zero in this window

Decision:

- This is still diagnostic-only.
- v46/v61/v62 are not promoted.
- The next evidence step is to execute more v62 groups and train a
  zero-adjusted, source-held-out ranker. Raw contraction should no longer be
  used as the main Z/Yaw training target.

## 2026-06-08 Status Update: v62 Expanded Beyond ep024

The v62 diagnostic command sweep is no longer only using the original ep024
window. Additional random held-out windows have been executed:

```text
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group001
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group002
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group003
```

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_group001_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v62_zyaw_diagnostic_group002_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_group001_zero_adjusted_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_group002_zero_adjusted_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_group003_zero_adjusted_effect_audit.json
```

Execution:

- group001: `ep008`, step `63`, `35 / 35` candidates executed, `35` MP4s
  preserved, close leaks `0`
- group002: `ep016`, step `63`, `35 / 35` candidates executed, `35` MP4s
  preserved, close leaks `0`
- group003: `ep024`, step `60`, `35 / 35` candidates executed, `35` MP4s
  preserved, close leaks `0`
- both used the fixed planner checkpoint and runtime-visible inputs only

Zero-adjusted beats-zero rates across the first four v62 groups:

```text
group000: combined 0.4706, XY 0.5000, Z 0.4706, Yaw 0.5882
group001: combined 0.5429, XY 0.6000, Z 0.4857, Yaw 0.2857
group002: combined 0.4857, XY 0.4857, Z 0.4857, Yaw 0.6000
group003: combined 0.4857, XY 0.3143, Z 0.4571, Yaw 0.4857
mean:     combined 0.4962, XY 0.4750, Z 0.4748, Yaw 0.4899
worst:    combined 0.4706, XY 0.3143, Z 0.4571, Yaw 0.2857
```

Current interpretation:

- Close ownership remains safe in these sweeps: close leak count is still `0`.
- The command-sweep and MP4 preservation path is working.
- The first four groups do not justify promoting v46/v61/v62. Average
  zero-adjusted effects are near chance, and yaw has a severe worst-window
  failure (`0.2857` beats-zero in group001).
- Yaw is not a fixed-sign problem. It needs symmetry-aware and
  observability-conditioned modeling before it can become a reliable
  micro-servo axis.
- Group003 shows that Z/Yaw diagnostic commands can also create bad XY side
  effects, so the next selector/ranker must optimize multi-axis combined
  effect and worsen risk, not a single-axis yaw or Z label.
- Continue groups003-007 or replace the grid with a stronger yaw-observable
  positive-intervention collection design. Do not train on raw Z/Yaw
  contraction or promote a ranker before held-out top-1 beats zero on
  individual XY, Z, Yaw, and combined residual effect.

## 2026-06-08 Status Update: Full v62 Diagnostic Result

The full v62 Z/Yaw diagnostic grid has now run to completion.

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v62_zyaw_diagnostic_all_groups_summary.json
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group000
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group001
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group002
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group003
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group004
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group005
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group006
runtime_artifacts/coarse2contact_v2/v62_command_sweep_zyaw_diagnostic_group007
```

Execution:

- candidate rows: `280`
- executed candidates: `280 / 280`
- front+wrist MP4s preserved: `280`
- close leak rows: `0`
- privileged runtime input: `false`

Full zero-adjusted summary:

```text
mean:  combined 0.4945, XY 0.4768, Z 0.4767, Yaw 0.5200
worst: combined 0.4706, XY 0.3143, Z 0.4571, Yaw 0.2857
best:  combined 0.5429, XY 0.6000, Z 0.4857, Yaw 0.6571
```

Decision:

- v62 is diagnostic-only and must not be promoted.
- Do not train a production ranker directly from this fixed-sign Z/Yaw grid.
- The result supports a change in data design, not another longer training run:
  collect yaw-observable, symmetry-conditioned positive-intervention windows;
  preserve same-window zero controls; train with multi-axis zero-adjusted
  contraction and explicit XY side-effect penalties.

## 2026-06-08 Status Update: v63 Yaw-Observable Spec Entry

Implemented the next data-design hook in
`scripts/build_c2c_v2_task_frame_command_sweep_spec.py`:

- `selection_mode=yaw_observable_symmetry`
- `candidate_profile=yaw_observable_symmetry`
- same-window `zero` controls are retained
- yaw hypotheses are sampled explicitly rather than treated as a single fixed
  global sign
- spec rows remain non-privileged and do not carry offline transition labels
- close authority remains disabled in every command-sweep row

Validation:

```text
conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py
200 passed in 3.54s
```

Current artifact scan:

- Scanned existing runtime traces under
  `runtime_artifacts/coarse2contact_v2`.
- Rows with v46 yaw fields exist, but the observed state is consistently
  `yaw_observable=false`, `yaw_ambiguous=true`, and
  `yaw_unobservable=true`.
- Strict v63 selection therefore yields no usable current rows.

Meaning:

- The current bottleneck is still yaw state semantics/observability, not the
  command-sweep runner.
- Do not train v63/v46 rankers from the current fixed-grid artifacts alone.
- Next work should create yaw-observable support windows with runtime-visible
  evidence, then execute the v63 symmetry-conditioned intervention grid and
  audit it with zero-adjusted XY/Z/Yaw/combined metrics.

## 2026-06-08 Status Update: v63 Bootstrap Spec Generated

Additional implementation work:

- Fixed boolean parsing in `task_frame_v46_alignment._as_bool`; string values
  such as `"False"` are no longer treated as true.
- Added observability calibration metrics to v46 train/eval reports:
  target/predicted rates for XY/Z/Yaw observability, yaw ambiguity, and yaw
  control eligibility.
- Made the command-sweep spec builder manifest-aware: it now honors per-row
  `source_eval_root`, `runtime_obs_path`, `trace_path`, and `stage_name`.
- Added `selection_mode=yaw_observable_symmetry_or_offline_label` for
  bootstrap collection. This mode may use offline sidecar labels to locate
  yaw-positive windows, but the emitted spec rows still strip labels and remain
  non-privileged runtime specs.

New artifacts:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yawbalanced_yaw_observability_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v58_nearfield_yaw_observability_audit.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_spec.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_spec.summary.json
```

Key audit result on `task_frame_alignment_v46_yaw_balanced_manifest.jsonl`:

```text
target yaw_control_rate:          0.0203
v46_yawbalanced predicted rate:   1.0000
v58_nearfield predicted rate:     0.0407
```

Interpretation:

- The yaw problem is calibration/semantics, not simply lack of labels.
- Existing manifests contain yaw-positive labels, but runtime smoke traces
  often land in ambiguous/unobservable windows.
- `v46_yawbalanced` is unsafe as a yaw gate because it over-opens yaw.
- `v58_nearfield` is closer but still requires closed-loop intervention proof.

v63 bootstrap spec summary:

```text
selected yaw-positive source rows: 114
candidate rows:                   3990
candidates per source row:        35
episodes covered:                 9
source roots covered:             25
same-window zero controls:        114
close-control rows:               0
privileged-runtime rows:          0
```

Validation:

```text
conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py
201 passed in 4.58s
```

Next required action:

Execute a held-out subset of the v63 bootstrap spec with front+wrist MP4
preserved, then audit zero-adjusted XY/Z/Yaw/combined contraction. Do not train
or promote a new ranker until the executed v63 candidates beat same-window zero
without close leaks and without XY side-effect regressions.

## 2026-06-08 Status Update: v63 Group000 Executed

Executed the first full v63 yaw-observable bootstrap command-sweep group.

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/v63_command_sweep_yaw_observable_bootstrap_group000
runtime_artifacts/coarse2contact_v2/reports/v63_command_sweep_yaw_observable_bootstrap_group000.json
runtime_artifacts/coarse2contact_v2/reports/v63_command_sweep_yaw_observable_bootstrap_group000_retry_failed.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group000_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group000_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_yaw_observable_bootstrap_group000_zero_adjusted_effect_audit.json
```

Execution:

- spec rows: `0-34`
- episode/window: `ep006`, step `121`
- candidates executed: `35 / 35`
- front+wrist MP4s: `35`
- gripper traces: `35`
- close leak rows: `0`
- privileged runtime rows: `0`
- planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`

The first `max_parallel=4` run completed `29 / 35` rows and hit CUDA OOM on 6
rows. Those failed rows were rerun serially with `max_parallel=1`, then the
full 35-row applied-transition manifest was rebuilt from the output root.

Zero-adjusted audit:

```text
audited_groups:      1
rows:                35
beats_zero_combined: 0.4706
beats_zero_xy:       0.4118
beats_zero_z:        0.4412
beats_zero_yaw:      0.4706
```

Candidate-level signal:

- Best combined candidates in this window were negative-yaw, positive-Z
  combinations such as `zyaw_sym_p_hyp_neg_z0030_y0240` and
  `zyaw_sym_p_hyp_neg_z0030_y0120`.
- Opposite-sign candidates worsened combined residual, confirming that yaw is
  not a fixed-sign axis and needs a learned symmetry-conditioned selector.

Decision:

- Do not promote v63 group000 or train from it alone.
- The result is useful as a positive-intervention probe: it proves the v63
  candidate family can expose useful local actions, while also showing that
  unranked candidate sampling is below chance overall.
- Next: execute additional source-held-out v63 groups across different
  episodes/source roots, then train a ranker only if held-out top-1 selection
  beats same-window zero on combined, XY, Z, and Yaw without close leaks.

## 2026-06-08 Status Update: v63 Group003 Executed

Executed a second complete v63 yaw-observable bootstrap command-sweep group.
This moves the evidence beyond the first `ep006` window and keeps the same
strict close and non-privileged runtime boundary.

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/v63_command_sweep_yaw_observable_bootstrap_group003
runtime_artifacts/coarse2contact_v2/reports/v63_command_sweep_yaw_observable_bootstrap_group003.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group003_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group003_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_yaw_observable_bootstrap_group003_zero_adjusted_effect_audit.json
```

Execution:

- spec rows: `105-139`
- episode/window: `ep010`, step `119`
- candidates executed: `35 / 35`
- front+wrist MP4s: `35`
- gripper traces: `35`
- close leak rows: `0`
- privileged runtime rows: `0`
- manifest retained rows: `35`
- manifest mode: `require_command_sweep_executed=true`
- planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`

Zero-adjusted audit:

```text
audited_groups:      1
rows:                35
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

- v63 group003 is also safe: no close leak, no privileged runtime input, and
  MP4/trace preservation is complete.
- It is still not a promotion result. Z has weak positive same-window signal,
  but XY and Yaw are below the required level and combined beats-zero remains
  below chance.
- The result reinforces the current diagnosis: the candidate family can
  expose useful local interventions, but unranked candidate sampling is not a
  controller. The next model step should be a source-held-out,
  symmetry-conditioned candidate ranker/top-1 selector trained from executed
  candidates, with additional held-out groups before any baseline upgrade.

## 2026-06-08 Status Update: v63 Group030 Executed

Executed a third complete v63 yaw-observable bootstrap command-sweep group on a
new episode/window, adding evidence beyond `ep006` and `ep010`.

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/v63_command_sweep_yaw_observable_bootstrap_group030
runtime_artifacts/coarse2contact_v2/reports/v63_command_sweep_yaw_observable_bootstrap_group030.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group030_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_group030_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_yaw_observable_bootstrap_group030_zero_adjusted_effect_audit.json
```

Execution:

- spec rows: `1050-1084`
- episode/window: `ep013`, step `134`
- candidates executed: `35 / 35`
- front+wrist MP4s: `35`
- gripper traces: `35`
- close leak rows: `0`
- privileged runtime rows: `0`
- manifest retained rows: `35`
- manifest mode: `require_command_sweep_executed=true`

Zero-adjusted audit:

```text
audited_groups:      1
rows:                35
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

- group030 is the strongest v63 group so far on combined beats-zero, and it
  confirms that the candidate family can expose useful held-out interventions.
- It is still not a baseline candidate: XY beats-zero remains weak, Z/Yaw are
  roughly chance, and unranked candidate sampling is not a deployable
  controller.
- The next useful training step is a source-held-out candidate ranker/top-1
  selector over executed v63 groups, but only after enough episode-diverse
  groups exist to prevent fitting a single window's symmetry/gain pattern.

## 2026-06-08 Status Update: v63 Three-Group Ranker Smoke

Combined the three complete v63 executed candidate groups into one
episode-diverse smoke dataset:

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

Two ranker scoring modes were tested with leave-one-group-out CPU smokes:

- `residual`: original post-residual norm ranking
- `axis_balanced`: new control-aware ranking score that penalizes axis worsen
  and missing per-axis contraction

Initial result with residual scoring and the first axis-balanced scoring pass:

- The original residual ranker overfits train groups and fails held-out top-1
  selection. It can pick candidates that reduce one part of the residual while
  sacrificing yaw or XY.
- The first `axis_balanced` pass reduced some combined-only failure shape, but
  it still did not pass the held-out gate.

Updated axis-balanced v2 result:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_groups000_003_030_candidate_ranker_axis_v2_loo_summary.json
```

- The axis-worsen penalty weight was increased to `5.0` after a unit test
  showed that the first penalty still allowed a large XY improvement to hide a
  yaw worsen.
- held-out `ep006`: selected the axis-balanced oracle candidate, but it lost
  combined/yaw relative to zero; zero already contracted all axes
- held-out `ep013`: selected a useful combined/yaw candidate, but lost XY and
  Z relative to the full task-frame objective
- held-out `ep010`: still failed combined, XY, and Yaw; only Z matched zero
- No tested ranker is promotable. The smoke confirms that candidate ranking is
  necessary, but the current three-window dataset and small ranker do not yet
  learn stable yaw-symmetry/axis-coupling rules.

Implementation note:

- `scripts/train_c2c_v2_task_frame_v46_candidate_ranker.py` now supports
  `--rank_score_mode axis_balanced`.
- The mode only changes offline training/evaluation scoring. It does not
  loosen strict handoff, restore C2C close authority, or add privileged runtime
  inputs.
- `conda run -n vla-adapter pytest -q tests/test_coarse2contact_v2.py`
  passes with `202` tests after this change.

Next:

- Execute additional episode/source-diverse v63 groups, especially windows
  where zero does not already contract all axes.
- Train a ranker only after enough groups exist for a real source-held-out
  split.
- Add an upgrade gate that requires top-1 to beat zero on combined and on each
  axis; combined-only wins are not enough for v46 promotion.

## 2026-06-08 Status Update: v63 Five-Group Axis-Balanced Smoke

The v63 line has now expanded beyond the initial three-group smoke into a
five-group episode-diverse held-out dataset:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_groups000_003_030_032_070_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_groups000_003_030_032_070_candidate_ranker_axis_v2_loo_summary.json
```

Dataset:

- rows: `175`
- groups: `ep006:step0121`, `ep010:step0119`, `ep013:step0134`,
  `ep014:step0096`, `ep029:step0107`
- candidates per group: `35`
- close leak rows: `0`
- privileged runtime rows: `0`

Axis-balanced LOO summary:

```text
top1_combined_mean: 0.6000
top1_xy_mean:       0.6000
top1_z_mean:        0.8000
top1_yaw_mean:      0.6000
zero_combined_mean:  0.4000
zero_xy_mean:        0.6000
zero_z_mean:         0.6000
zero_yaw_mean:       0.4000
```

Interpretation:

- The axis-balanced score is better than the original residual score because it
  avoids some obvious axis-coupling mistakes.
- The five-group result is still smoke-only. It does not yet prove a
  promotable three-axis controller.
- XY remains flat against zero on this small set, so the current signal is not
  strong enough to justify a baseline switch.
- Z is the strongest axis in this smoke, while yaw remains the most fragile
  cross-window axis.
- The candidate-ranker smoke now uses source-root/session held-out splitting
  by default, not just episode-step grouping. That keeps the validation closer
  to the intended generalization target.
- The applied-transition manifest now recovers the original source root from
  the command-sweep spec path plus row index, instead of inheriting the command
  execution output directory. That keeps source-root/session held-out split
  semantics honest for executed command sweeps.
- A follow-up root-held-out smoke on the same five-group v63 dataset preserved
  the signal but did not change the promotion verdict:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_groups000_003_030_032_070_candidate_ranker_axis_v2_root_smoke_train.json`
  reports train top1 combined `0.4000` vs zero `0.4000`, val top1 combined
  `0.6000` vs zero `1.0000`, and the held-out split still shows XY/Yaw
  coupling mistakes. This is useful as a stronger split sanity check, but it
  remains smoke-only.
- A wider root-held-out smoke with `val_fraction=0.4` on the recovered
  manifest produced a more realistic two-root validation split:
  `runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_groups000_003_030_032_070_candidate_ranker_axis_v2_root_recovered_v40val_smoke_train.json`
  reports train top1 combined `1.0000` vs zero `0.5000`, val top1 combined
  `0.5000` vs zero `0.0000`. The honest source-root split keeps the signal,
  but it is still smoke-only and still not promotable.
- The next useful step is to keep collecting more source-diverse held-out
  groups and only then rerun the ranker gate. Do not promote on this five-group
  result alone.

## 2026-06-08 Status Update: v63 Additional-Roots Full Sweep

The v63 source-diverse command-sweep batch over the additional roots has now
completed and the recovered applied-transition manifest is available:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_additional_roots_batch_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_additional_roots_batch_manifest.summary.json
```

Full batch summary:

- rows: `70`
- source roots: `2`
- episode coverage: `ep006`
- close leak rows: `0`
- observed XY contraction: `0.9286`
- observed Z contraction: `0.6000`
- observed Yaw contraction: `0.5714`
- observed combined contraction: `0.6143`

Interpretation:

- This batch confirms the source-diverse command-sweep plumbing is still
  healthy at full size and still respects the strict handoff boundary.
- The data are not a promotion result by themselves. The row mix is still
  concentrated on one episode tail, and the contraction numbers remain a
  candidate-side diagnostic rather than a closed-loop insert success proof.
- A root-held-out candidate-ranker smoke on the full 70-row manifest preserved
  the same honest split behavior, but it still did not beat zero/no-op on the
  held-out split. That means the current v63/v46 candidate family can produce
  useful intervention data, but it still does not yet yield a promotable
  source-held-out selector.
- Next step: expand beyond the single episode tail, collect more random
  held-out episodes/source roots, and rerun the source-held-out gate before any
  baseline upgrade.

## 2026-06-08 Status Update: v63 Combined Source-Held-Out Ranker Smoke

The v63 source-held-out candidate-ranker smoke was rerun on the union of the
five-group recovered manifest and the 70-row additional-roots full manifest:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_groups000_003_030_032_070_manifest_recovered_source_root.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_yaw_observable_bootstrap_additional_roots_batch_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v63_groups000_003_030_032_070_plus_additional_roots_candidate_ranker_smoke_train.json
```

Combined smoke summary:

- dataset rows: `245`
- candidate groups: `6`
- held-out source roots: `2`
- train rows: `140`
- val rows: `70`
- close leak rows: `0`

Held-out val metrics:

- top1 best-score match: `0.0`
- top1 combined contraction: `1.0`
- zero combined contraction: `0.5`
- top1 XY contraction: `0.0`
- top1 Z contraction: `1.0`
- top1 Yaw contraction: `1.0`
- zero XY contraction: `0.5`
- zero Z contraction: `1.0`
- zero Yaw contraction: `0.5`

Interpretation:

- This is the first combined smoke where the held-out selector beats zero on
  combined contraction while keeping strict close safety intact.
- The selector still does not match the oracle best-score candidate, and XY is
  still the fragile axis on the held-out split. So this is promising evidence,
  not promotion evidence.
- The result is useful because it shows the ranker can start to generalize
  once source-root diversity is widened beyond a single episode tail.
- The next useful step is to keep widening the random/source-held-out pool and
  to look for a selector that also improves held-out XY, not just combined/Z/Yaw.

## 2026-06-08 Status Update: v46 Broad Source-Held-Out Smoke

The current v46 candidate-ranker smoke was rerun on a broader union of the
existing v62/v63 diagnostic manifests:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_ranker_v62_v63_source_holdout_smoke.json
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_candidate_ranker_v62_v63_source_holdout_smoke.pt
```

This run used the eight v62 diagnostic manifests plus the v63 recovered
source-root manifest and the v63 additional-roots batch manifest.

Smoke summary:

- train rows: `224`
- val rows: `10`
- train groups: `6`
- held-out split: `root`
- close leak rows: `0`
- upgrade gate: `pending_large_random_holdout_and_closed_loop_insert_success`

Held-out val metrics:

- top1 best-score match: `0.16666666666666666`
- top1 combined contraction: `0.8333333333333334`
- zero combined contraction: `0.5`
- top1 XY contraction: `0.5`
- zero XY contraction: `0.0`
- top1 Z contraction: `0.8333333333333334`
- zero Z contraction: `0.6666666666666666`
- top1 Yaw contraction: `0.6666666666666666`
- zero Yaw contraction: `0.3333333333333333`

Interpretation:

- This smoke is broader than the earlier ep024-heavy tail and keeps the strict
  close boundary intact.
- It is the first broader held-out smoke where combined, Z, and Yaw all beat
  the zero/no-op baseline on the held-out split.
- XY is no longer empty, but it is still not clearly the axis that moves the
  result from promising to promotable. The next step is to widen the held-out
  pool again and look for a selector that also improves held-out XY, not just
  combined/Z/Yaw.
- The branch is still not promotable. The remaining gate is larger random held-
  out validation plus closed-loop insert success.

## 2026-06-08 Status Update: v46 Stricter Held-Out Split Smoke

The same broader v62/v63 source pool was rerun with a stricter held-out split:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_ranker_v62_v63_source_holdout_smoke_v2.json
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_candidate_ranker_v62_v63_source_holdout_smoke_v2.pt
```

This run kept the same manifest union but increased `val_fraction` to `0.4`.

Smoke summary:

- train rows: `207`
- val rows: `98`
- train groups: `5`
- held-out split: `root`
- close leak rows: `0`
- upgrade gate: `pending_large_random_holdout_and_closed_loop_insert_success`

Held-out val metrics:

- top1 best-score match: `0.3333333333333333`
- top1 combined contraction: `0.3333333333333333`
- zero combined contraction: `0.6666666666666666`
- top1 XY contraction: `0.6666666666666666`
- zero XY contraction: `0.6666666666666666`
- top1 Z contraction: `1.0`
- zero Z contraction: `0.6666666666666666`
- top1 Yaw contraction: `0.0`
- zero Yaw contraction: `0.3333333333333333`

Interpretation:

- This stricter held-out gate is more informative than the lighter smoke.
- It shows the current selector family is still not robust enough to be called
  promotable: combined contraction fell below zero on the held-out split, and
  yaw is still the weakest axis under a harder root holdout.
- The broader held-out pool is still useful because it makes the failure
  visible instead of hiding behind a single tail.
- The next useful step is to keep widening the random/source-held-out pool and
  improve the selector so it can beat zero consistently on combined and XY, not
  just on easier slices.

## 2026-06-08 Status Update: v46 Wider Pool Smoke with Partial Manifests

The source-held-out smoke was expanded again by adding the two partial v63
additional-roots manifests to the same v62/v63 union:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_ranker_v62_v63_source_holdout_smoke_v3.json
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_candidate_ranker_v62_v63_source_holdout_smoke_v3.pt
```

Smoke summary:

- train rows: `284`
- val rows: `116`
- train groups: `6`
- held-out split: `root`
- close leak rows: `0`
- upgrade gate: `pending_large_random_holdout_and_closed_loop_insert_success`

Held-out val metrics:

- top1 best-score match: `0.6666666666666666`
- top1 combined contraction: `0.3333333333333333`
- zero combined contraction: `1.0`
- top1 XY contraction: `0.6666666666666666`
- zero XY contraction: `0.6666666666666666`
- top1 Z contraction: `1.0`
- zero Z contraction: `1.0`
- top1 Yaw contraction: `0.0`
- zero Yaw contraction: `0.6666666666666666`

Interpretation:

- This run is the first time the broader pool clearly showed that the selector
  family still does not consistently beat zero once the held-out split gets
  wider.
- XY is not yet the main differentiator here; the loss is dominated by yaw on
  the held-out split.
- Combined contraction is still not promotable, so this remains a candidate
  result rather than a baseline promotion path.
- The right next move is still more random/source-held-out coverage with
  better yaw observability, not relaxing strict handoff.

## 2026-06-08 Status Update: v46 Broad Plus Yaw-Balanced Unified Training

The unified v46 estimator/controller was retrained on the combination of the
yaw-balanced manifest plus the broader v61/v62/v63 source-held-out pool:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_unified_candidate_broad_plus_yawbalanced_holdout.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_unified_candidate_broad_plus_yawbalanced_holdout_train.json
```

This run used `6458` train rows and `1240` held-out rows across `176` train
source roots and `41` held-out source roots.

Train metrics:

- bounded_step_contraction: `0.9459584951400757`
- xy_bounded_step_contraction: `0.9001238942146301`
- z_bounded_step_contraction: `0.8624961376190186`
- yaw_bounded_step_contraction: `0.013471662998199463`
- yaw_control_predicted_rate: `0.036543820053339005`
- yaw_control_target_rate: `0.016878290101885796`

Validation metrics:

- bounded_step_contraction: `0.9112903475761414`
- xy_bounded_step_contraction: `0.8580645322799683`
- z_bounded_step_contraction: `0.85161292552948`
- yaw_bounded_step_contraction: `0.02661290392279625`
- yaw_control_predicted_rate: `0.05000000074505806`
- yaw_control_target_rate: `0.02983870916068554`

Random-heldout style checks on the same checkpoint:

- `task_frame_alignment_v46_holdout_random5_v45_strict_manifest.jsonl`
  - bounded_step_contraction: `0.9499072432518005`
  - xy_bounded_step_contraction: `0.8849721550941467`
  - z_bounded_step_contraction: `0.8311688303947449`
  - yaw_observable_target_rate: `0.0`
  - yaw_control_target_rate: `0.0`
- `task_frame_alignment_v46_yaw_balanced_manifest.jsonl`
  - bounded_step_contraction: `0.9361879825592041`
  - xy_bounded_step_contraction: `0.8858612775802612`
  - z_bounded_step_contraction: `0.855832040309906`
  - yaw_control_predicted_rate: `0.04170721396803856`
  - yaw_control_target_rate: `0.02029751054942608`
  - yaw_bounded_step_contraction: `0.016960933804512024`

Interpretation:

- This is the first unified v46 retrain that nudged yaw back into a non-zero
  control regime while still holding the broader XY/Z improvements from the
  source-held-out pool.
- The checkpoint is still not promotable: yaw remains weak, and the random
  held-out strict eval still does not give a clean three-axis promotion signal.
- Still, this is a meaningful step toward the objective because it shows the
  unified estimator can absorb both broad source-held-out data and yaw-balanced
  data without collapsing the strict close boundary.
- The next step is to keep widening the held-out source pool and to target yaw
  observability more explicitly, rather than relaxing the handoff contract.

## 2026-06-08 Status Update: v63 Additional Old-Flow Held-Out Audits

Extended the held-out evidence pool with two additional old-flow smoke batches
and audited their task-frame response:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_oldflow_4row_audit_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_oldflow_4row_audit_manifest.summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_oldflow_4row_audit_manifest_v2.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v63_oldflow_4row_audit_manifest_v2.summary.json
runtime_artifacts/coarse2contact_v2/reports/v63_oldflow_4row_control_effect_audit.json
```

Observed summary:

- batch v1: 189 rows across 3 source roots, `close_leak_rows=0`
- batch v2: 248 rows across 4 source roots, `close_leak_rows=0`
- control-effect audit on the newer four traces showed strong XY command-to-
  residual coupling, with `delta_vs_command_xy_regression` R^2 ranging from
  `0.66` to `0.97` on the selected windows
- the same audit also showed the learned `task_frame_v46_applied_local_6d`
  branch is still sparse on these traces, which is consistent with a narrow
  activation window rather than an unconditional controller

Interpretation:

- These windows strengthen the random/source-held-out evidence pool without
  introducing close leaks or privileged runtime input.
- The new traces are still not promotion evidence, but they are useful
  because they expose where v46 applies, where it abstains, and how strongly
  the executed local command maps to the true residual response.
- The next step remains widening the held-out pool and validating whether
  broader activation coverage improves XY/Z/Yaw without letting close
  authority leak back in.

## 2026-06-09 Status Update: Artifact Hygiene And Conservative Slimming

The repository was measured before cleanup at about `92G`, with almost all
space under `runtime_artifacts/coarse2contact_v2`.  The first conservative
slimming pass reduced the project to about `88G`.

Removed as disposable artifacts:

- Python caches and bytecode: `__pycache__`, `.pytest_cache`, `.mypy_cache`,
  `.ruff_cache`, and `*.pyc`
- failed or non-canonical environment probes: `directpy`, `offscreen`, dry-run,
  old failed `retest_oldflow_*`, and legacy batch test directories
- unreferenced historical smoke videos and superseded visual comparison dumps
  from older v20-v35 style experiments
- unreferenced temporary/intermediate dirs such as `tmp_direction_inputs` and
  old hard-bucket A/B scratch outputs
- unreferenced superseded masked-geometry dataset variants `v3` and `v5`

Intentionally retained:

- `runtime_artifacts/coarse2contact_v2/checkpoints`
- current and baseline datasets, including v42/v46/v59/v61/v62/v63 manifests
- `runtime_artifacts/coarse2contact_v2/reports`
- current canonical dry-run evidence and old-flow summaries
- v42/v45/v46 MP4s and any MP4 referenced by docs/reports
- large hard-bucket sweep roots that are still referenced by training reports,
  gate summaries, or the task-frame alignment breakthrough plan

Policy going forward:

- Do not delete checkpoint, dataset, report, current MP4, or canonical test
  evidence directories during routine cleanup.
- Large raw sweep roots can only be removed after exporting a compact evidence
  bundle containing the summary JSON/Markdown, resolved source roots, split
  manifests, and the exact command provenance needed to interpret the result.
- The canonical RLBench evaluation path remains:
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py ...`
- The planner checkpoint remains:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`

## 2026-06-09 Status Update: Stricter v46 Three-Axis Gate

The v46 promotion gate was tightened so the project cannot treat strong XY/Z
held-out contraction as proof of full task-frame alignment when yaw-control
evidence is missing.  `scripts/summarize_c2c_v2_task_frame_v46_gate.py` now
requires enough yaw-control target coverage before a report can pass the
offline gate; otherwise it records `insufficient_yaw_control_evidence`.

Updated gate output:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_summary.json
offline_gate_status: fail
promotion_status: fail_offline_gate
```

Checked reports:

- `random5` held-out strict eval:
  - bounded-step contraction: `0.9499072432518005`
  - XY contraction: `0.8849721550941467`
  - Z contraction: `0.8311688303947449`
  - yaw-control target rate: `0.0`
  - violation: `insufficient_yaw_control_evidence`
- `yawbalanced` eval:
  - bounded-step contraction: `0.9361879825592041`
  - XY contraction: `0.8858612775802612`
  - Z contraction: `0.855832040309906`
  - yaw-control target rate: `0.02029751054942608`
  - yaw bounded-step contraction: `0.016960933804512024`
  - violation: `insufficient_yaw_control_evidence`

Interpretation:

- Current v46 evidence is still useful for XY/Z and close-safety preservation,
  but it does not yet prove the requested three-axis random held-out task-frame
  contraction.
- The next data collection/training step must deliberately include random
  held-out yaw-observable near-contact windows.  A holdout with no yaw-control
  target rows is a safety check, not a yaw capability proof.
- Do not promote v46 or run success-claim MP4s from this gate result.  Continue
  using the canonical RLBench path and preserve strict handoff/planner-owned
  close while widening yaw-observable random failure-tail coverage.

## 2026-06-09 Status Update: Yaw-Observable Holdout Coverage Audit

Added a dedicated offline coverage audit:

```text
scripts/audit_c2c_v2_task_frame_yaw_holdout_coverage.py
```

The audit uses `task_frame_v46_labels_from_row(...)` rather than raw manifest
field names, because current manifests mix `yaw_observable`,
`yaw_control_observable`, `yaw_observability_class`, and `offline_labels`.
This avoids falsely concluding that a pool has or lacks yaw-control evidence.

Strict near-contact coverage check:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_holdout_coverage_audit.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_control_near_contact_candidates.jsonl
```

Inputs:

- `task_frame_alignment_v46_yaw_balanced_manifest.jsonl`
- `task_frame_alignment_v46_holdout_random5_v45_strict_manifest.jsonl`
- `task_frame_alignment_v63_yaw_observable_bootstrap_groups000_003_030_032_070_manifest_recovered_source_root.jsonl`
- `task_frame_alignment_v63_yaw_observable_bootstrap_additional_roots_batch_manifest.jsonl`

Result with `near_xy_radius=0.060`, `near_z_radius=0.040`,
`max_abs_yaw=0.350`:

- input rows: `7977`
- label rows: `7977`
- near-contact rows: `3408`
- yaw-observable / yaw-control rows: `146`
- selected yaw-control near-contact rows: `0`
- status: `insufficient_coverage`
- violations: `insufficient_yaw_control_rows`,
  `insufficient_yaw_control_roots`

Breakdown:

- all `3408` strict near-contact rows are yaw ambiguous/unobservable
- all `146` yaw-control rows are outside the strict near-contact Z radius
- the blocker is therefore not just model quality; the current validation pool
  lacks random/source-held-out rows where yaw is both controllable and close
  enough in task-frame Z.

A relaxed candidate pool was exported for the next data/replay step:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_candidate_pool_audit.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_candidate_pool.jsonl
```

With `near_z_radius=0.080`, the pool contains:

- selected rows: `146`
- selected source roots: `25`
- status: `pass` for coverage only

Interpretation:

- The `z08` pool is not promotion evidence and should not be used to claim
  near-contact yaw success.
- It is the correct seed pool for the next run: collect or replay these
  yaw-visible windows toward stricter `z<=0.040` near-contact coverage, then
  rerun the v46 held-out gate.

Follow-up command-sweep spec:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_spec.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_spec.summary.json
runtime_artifacts/coarse2contact_v2/v46_yaw_visible_z08_command_sweep_dryrun/summary.json
```

The spec was built from the `z08` yaw-visible pool with:

- `selection_mode=yaw_observable_symmetry_or_offline_label`
- `candidate_profile=yaw_observable_symmetry`
- `max_source_rows=1`
- no combined XY candidates
- Z candidate magnitudes `0.003`, `0.006`, `0.010`
- Yaw candidate magnitudes `0.004`, `0.006`, `0.010`

Spec summary:

- selected runtime rows: `25`
- source roots: `25`
- candidate commands per runtime row: `49`
- retained candidate rows: `1225`
- `uses_privileged_runtime=false`
- `uses_privileged_label_for_training=false`
- no privileged residual/teacher/success-pose keys are present in the emitted
  command-sweep rows

Dry-run summary:

- first `3` candidate rows dry-ran successfully
- generated commands use the canonical path
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py ...`
- fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- front+wrist videos, gripper traces, runtime observations, and failure target
  capture are enabled for the real run

Next action:

- Execute this spec in batches on the normal RLBench-capable shell, then build
  an executed-only applied-transition manifest.  The resulting rows should show
  whether z/yaw candidate commands can move the `z08` yaw-visible pool into the
  strict `z<=0.040` near-contact band without close leaks.

## 2026-06-09 Status Update: z08 Yaw-Visible Command-Sweep Smoke Rows 000-009

Executed the first 10 rows of the `z08` yaw-visible command-sweep spec through
the canonical RLBench path:

```text
conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py ...
```

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/v46_yaw_visible_z08_command_sweep_smoke_rows000_009/summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_rows000_009_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_rows000_009_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_rows000_009_coverage_audit.json
```

Execution summary:

- selected candidate rows: `10`
- success count: `10`
- failure count: `0`
- front+wrist MP4s saved: `10`
- planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- launcher: canonical `conda run -n vla-adapter xvfb-run -a python`

Executed-only manifest summary:

- retained rows: `10`
- source roots: `1`
- episode: `ep006`
- close leak rows: `0`
- observed XY contraction: `1.0`
- observed Z contraction: `0.8`
- observed Yaw contraction: `0.7`
- `uses_privileged_runtime=false`
- privileged labels remain offline pre/post transition labels only

Candidate observations on this one source/window:

- yaw-negative candidates slightly reduced yaw residual.
- yaw-positive candidates worsened yaw in some cases.
- `z_guard_pos_0060` reduced Z from about `0.01086` to `0.00667`.
- `z_guard_pos_0100` reduced Z further to about `0.00422` but worsened yaw
  substantially on this smoke.
- Z-negative guard candidates worsened Z, as expected from the sign.

Important caveat:

- The original `z08` seed rows were selected as yaw-visible offline candidates,
  but the executed transition manifest relabeled this runtime window as
  `yaw_ambiguous=True`, `yaw_observable=False`.
- A follow-up coverage audit on the executed manifest selected `0` yaw-control
  rows, despite all rows being near-contact in Z.
- Therefore this smoke proves the command-sweep execution path, MP4 retention,
  close safety, and useful Z/Yaw transition signal on one source root.  It does
  not yet provide yaw-control held-out promotion evidence.

Next action:

- Do not train/promote from this smoke as if it were yaw-control held-out
  proof.
- Diagnose why replayed/executed yaw observability changed from the `z08`
  source label to ambiguous/unobservable in the runtime transition rows.
- Expand execution across more source roots only after either preserving
  yaw-observable windows through replay or explicitly treating this as
  near-contact ambiguous-yaw transition data.

## 2026-06-09 Status Update: z08 Yaw-Visible Command-Sweep Smoke ep010 Basic + Zero

Executed the next basic candidate block from the same `z08` command-sweep spec,
covering `ep010`, step `119`, rows `049-058`, then added row `061` as the
zero/no-op baseline for a proper zero-adjusted comparison.

Artifacts:

```text
runtime_artifacts/coarse2contact_v2/v46_yaw_visible_z08_command_sweep_smoke_rows049_058/summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_rows049_058_manifest.jsonl
runtime_artifacts/coarse2contact_v2/v46_yaw_visible_z08_command_sweep_smoke_row061_zero/summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_row061_zero_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_ep010_basic_plus_zero_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_ep010_basic_plus_zero_zero_adjusted_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_ep010_basic_plus_zero_transition_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_ep010_basic_plus_zero_coverage_audit.json
```

Execution summary:

- rows `049-058`: `10/10` evaluator runs succeeded
- row `061` zero baseline: `1/1` evaluator run succeeded
- launcher: canonical `conda run -n vla-adapter xvfb-run -a python`
- planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- videos use front+wrist layout
- close leak rows: `0`
- `uses_privileged_runtime=false`

Zero-adjusted audit on ep010 basic+zero:

- audited groups: `1`
- beats zero combined: `0.7272727272727273`
- beats zero XY: `0.36363636363636365`
- beats zero Z: `0.45454545454545453`
- beats zero Yaw: `0.7272727272727273`

Transition-effect audit on the yaw candidates:

- yaw contraction rate: `1.0`
- yaw worsen rate: `0.0`
- combined contraction rate: `1.0`
- best per group beats zero yaw: `1.0`
- best candidate: `yaw_hyp_pos_0040`
- mean zero-adjusted yaw delta: `-0.0030780136585235596`
- XY collateral worsen rate: `1.0`
- Z collateral worsen rate: `0.0`

Coverage audit:

- near-contact rows: `11`
- yaw-observable rows: `11`
- yaw-ambiguous rows: `11`
- selected yaw-control rows: `0`
- status: `insufficient_coverage`

Interpretation:

- This ep010 smoke is useful transition supervision: some yaw/Z candidates
  clearly beat zero, with no close leak and no privileged runtime input.
- It is still not promotion evidence because the same executed rows are marked
  yaw ambiguous, so the strict yaw-control coverage gate remains empty.
- The best yaw candidate also worsens XY collateral in this tiny window. The
  next ranker/controller update must explicitly trade yaw contraction against
  XY collateral, not optimize yaw alone.
- The immediate next data action is to keep expanding source roots from the
  `z08` spec, but each window must include a zero baseline row so the
  zero-adjusted audit remains meaningful.

## 2026-06-09 Status Update: z08 Three-Window Basic + Zero Sweep

Expanded the `z08` yaw-visible command-sweep evidence to three independent
source/window groups, each with the same basic yaw/Z candidates and a zero
baseline:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_three_basic_plus_zero_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_three_basic_plus_zero_zero_adjusted_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_three_basic_plus_zero_transition_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_three_basic_plus_zero_coverage_audit.json
```

The third source/window came from:

```text
runtime_artifacts/coarse2contact_v2/v46_yaw_visible_z08_command_sweep_smoke_rows098_107_110/summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_rows098_107_110_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_rows098_107_110_zero_adjusted_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_rows098_107_110_transition_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_rows098_107_110_coverage_audit.json
```

Execution summary:

- new rows `098-107,110`: `11/11` evaluator runs succeeded
- launcher: canonical `conda run -n vla-adapter xvfb-run -a python`
- front+wrist videos saved
- close leak rows: `0`
- `uses_privileged_runtime=false`

Third-window zero-adjusted audit:

- beats zero combined: `0.36363636363636365`
- beats zero XY: `0.5454545454545454`
- beats zero Z: `0.36363636363636365`
- beats zero Yaw: `0.45454545454545453`

Third-window transition-effect audit:

- overall yaw contraction: `0.6666666666666666`
- overall yaw worsen: `0.3333333333333333`
- overall beats zero yaw: `0.5`
- best per group beats zero yaw: `1.0`
- best per group yaw contraction: `1.0`
- best per group XY collateral worsen: `0.0`
- best per group Z collateral worsen: `0.0`

Three-window combined audit:

- zero-adjusted rows: `33`
- audited groups: `3`
- close leak rows: `0`
- beats zero combined: `0.48484848484848486`
- beats zero XY: `0.48484848484848486`
- beats zero Z: `0.3939393939393939`
- beats zero Yaw: `0.5454545454545454`
- transition-effect yaw contraction: `0.7777777777777778`
- transition-effect yaw worsen: `0.2222222222222222`
- transition-effect beats zero yaw: `0.6666666666666666`
- best per group beats zero yaw: `1.0`
- best per group yaw contraction: `1.0`
- best per group XY collateral worsen: `0.3333333333333333`
- best per group Z collateral worsen: `0.0`

Coverage caveat:

- strict coverage still selects `0` yaw-control rows
- violation remains `insufficient_yaw_control_rows` and
  `insufficient_yaw_control_roots`
- the executed rows remain useful transition/ranker supervision, but not v46
  promotion evidence

Interpretation:

- The environment contains useful yaw/Z actions: every window has at least one
  candidate that beats zero on yaw, and the best candidate never worsened yaw.
- The current policy/ranker is not yet reliable enough: overall candidate rows
  only beat zero yaw `54.5%` of the time and beat zero combined `48.5%` of the
  time.
- The next model update should train a collateral-aware selector/ranker on
  these executed windows, with explicit penalties for XY collateral and
  zero-adjusted non-improvement.  This is more aligned than tuning yaw
  observability thresholds alone.
- Because strict yaw-control coverage remains empty, this dataset should be
  treated as near-contact ambiguous-yaw transition supervision until the yaw
  ambiguity semantics are corrected or a truly yaw-control held-out pool is
  collected.

Follow-up ranker smoke:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_z08_three_basic_yaw_collateral_ranker_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_three_basic_yaw_collateral_ranker_smoke_train.json
```

Training setup:

- dataset:
  `task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_three_basic_plus_zero_manifest.jsonl`
- split mode: `root`
- rank score mode: `yaw_collateral`
- train groups: `2`
- held-out val groups: `1`
- upgrade gate: `pending_large_random_holdout_and_closed_loop_insert_success`

Smoke result:

- train top1 best-score match: `1.0`
- train top1 XY/Z/Yaw/combined contraction: `1.0`
- val top1 best-score match: `1.0`
- val top1 XY/Z/Yaw/combined contraction: `1.0`

Interpretation:

- This proves the new executed transition data and collateral-aware ranker
  objective connect end-to-end.
- It is not held-out promotion evidence because the validation set has only one
  group and comes from the same tiny z08 smoke pool.
- The next real step is to execute more source-root windows with zero baselines
  and retrain/evaluate the ranker on a materially larger source-held-out split.

## 2026-06-09 Status Update: z08 Four-Window Basic + Zero Sweep

Expanded the same basic yaw/Z + zero command-sweep protocol to a fourth
source/window:

```text
runtime_artifacts/coarse2contact_v2/v46_yaw_visible_z08_command_sweep_smoke_rows147_156_159/summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_rows147_156_159_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_rows147_156_159_zero_adjusted_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_rows147_156_159_transition_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_rows147_156_159_coverage_audit.json
```

Four-window merged artifacts:

```text
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_four_basic_plus_zero_manifest.jsonl
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_four_basic_plus_zero_zero_adjusted_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_four_basic_plus_zero_transition_effect_audit.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_visible_z08_smoke_four_basic_plus_zero_coverage_audit.json
```

Execution summary:

- new rows `147-156,159`: `11/11` evaluator runs succeeded
- launcher: canonical `conda run -n vla-adapter xvfb-run -a python`
- front+wrist videos saved
- close leak rows: `0`
- `uses_privileged_runtime=false`

Fourth-window audit:

- zero-adjusted beats zero combined: `0.45454545454545453`
- zero-adjusted beats zero yaw: `0.36363636363636365`
- transition overall yaw contraction: `0.0`
- transition overall yaw worsen: `1.0`
- transition overall beats zero yaw: `0.5`
- best per group beats zero yaw: `1.0`
- best per group yaw contraction: `0.0`
- best per group yaw worsen: `1.0`
- best per group XY collateral worsen: `1.0`
- strict yaw-control coverage: `0` selected rows

Four-window combined audit:

- zero-adjusted rows: `44`
- audited groups: `4`
- close leak rows: `0`
- beats zero combined: `0.4772727272727273`
- beats zero XY: `0.4772727272727273`
- beats zero Z: `0.4090909090909091`
- beats zero Yaw: `0.5`
- transition-effect yaw contraction: `0.5833333333333334`
- transition-effect yaw worsen: `0.4166666666666667`
- transition-effect beats zero yaw: `0.625`
- best per group beats zero yaw: `1.0`
- best per group yaw contraction: `0.75`
- best per group yaw worsen: `0.25`
- best per group XY collateral worsen: `0.5`
- best per group Z collateral worsen: `0.0`
- strict yaw-control coverage still fails with `insufficient_yaw_control_rows`
  and `insufficient_yaw_control_roots`

Interpretation:

- The fourth window is an important hard negative: commands can beat the zero
  drift baseline while still increasing absolute yaw residual and worsening XY.
- Therefore the next ranker target must not rely on `beats_zero_yaw` alone. It
  must jointly require absolute yaw contraction, combined contraction, and low
  XY/Z collateral.
- The four-window pool is still tiny and non-promotable, but it is now more
  useful for training/debugging because it contains both positive and negative
  source-held-out transition behavior.

Follow-up four-window ranker smoke:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_z08_four_basic_yaw_collateral_ranker_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_four_basic_yaw_collateral_ranker_smoke_train.json
```

Training setup:

- dataset:
  `task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_four_basic_plus_zero_manifest.jsonl`
- split mode: `root`
- rank score mode: `yaw_collateral`
- train groups: `2`
- held-out val groups: `1`
- upgrade gate: `pending_large_random_holdout_and_closed_loop_insert_success`

Smoke result:

- train top1 best-score match: `1.0`
- train top1 XY/Z/Yaw/combined contraction: `1.0`
- val top1 best-score match: `1.0`
- val top1 XY/Z/Yaw/combined contraction: `1.0`

Important caveat:

- This split did not explicitly hold out the fourth hard-negative window where
  absolute yaw worsened and XY collateral worsened.
- The result only confirms the four-window data still connects to the
  collateral-aware ranker objective.  It is not enough to show the ranker can
  generalize to the newly observed hard-negative behavior.
- A follow-up hard-negative-held-out smoke should force that fourth source root
  into validation, or run leave-one-root validation across all four groups.

Hard-negative-held-out follow-up:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_z08_four_basic_yaw_collateral_ranker_hardneg_val_seed3_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_four_basic_yaw_collateral_ranker_hardneg_val_seed3_smoke_train.json
```

This run used `seed=3`, which places the fourth hard-negative source root in
validation:

```text
runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation/large_xy_large_yaw_focus_flush_30k/chunk_001_008_015/eval
```

Result:

- train groups: `2`
- val groups: `1`
- validation oracle candidate: `z_guard_pos_0100`
- validation selected candidate: `yaw_hyp_neg_0100`
- validation top1 best-score match: `0.0`
- validation top1 combined contraction: `0.0`
- validation top1 XY contraction: `0.0`
- validation top1 Yaw contraction: `0.0`
- validation top1 Z contraction: `1.0`
- zero combined/Yaw/XY contraction on the same val group: `0.0`

Interpretation:

- This is the most useful ranker result so far because it is a targeted
  negative: when the hard-negative source root is truly held out, the current
  four-window ranker fails to choose the collateral-safe oracle.
- The failure mode is concrete: training is still biased toward yaw-step
  candidates from the two easier train groups, so the ranker does not yet learn
  when a guarded Z command is the safer local action.
- Do not use the four-window ranker as a runtime candidate.  The next data step
  should deliberately add more hard-negative windows where yaw commands worsen
  absolute yaw or XY collateral, plus windows where Z guard is the oracle, then
  rerun leave-one-root or targeted hard-negative validation.

Explicit held-out-root ranker rerun:

```text
runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v46_z08_four_basic_yaw_collateral_ranker_hardneg_explicit_val_smoke.pt
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_four_basic_yaw_collateral_ranker_hardneg_explicit_val_smoke_train.json
```

The candidate-ranker training script now accepts explicit split roots:

```text
--train_source_eval_root
--val_source_eval_root
--test_source_eval_root
```

This removes the previous seed-guessing problem when auditing specific
hard-negative source roots.  A focused unit test covers the rule that an
explicit validation root remains in validation and is not silently reassigned
to train/test.

The explicit rerun forced this validation root:

```text
/home/guoning/code/VLA2/runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation/large_xy_large_yaw_focus_flush_30k/chunk_001_008_015/eval
```

Result:

- train groups: `2`
- val groups: `1`
- requested validation root was recorded in the report metadata
- validation oracle candidate: `z_guard_pos_0100`
- validation selected candidate: `z_guard_pos_0100`
- validation top1 best-score match: `1.0`
- validation top1 XY/Z/Yaw/combined contraction: `1.0`
- zero XY/Yaw/combined contraction on the same val group: `0.0`

Interpretation:

- This rerun supersedes seed-dependent split bookkeeping for this specific
  hard-negative audit: the validation root is now explicit and reproducible.
- It does not promote the ranker.  The validation set is still only one source
  root/group, and the earlier four-window evidence remains too small for a
  runtime candidate.
- Next step remains the same: collect more executed source-root windows,
  especially yaw-worsen/XY-collateral hard negatives and Z-guard-oracle cases,
  then run leave-one-root or explicitly targeted held-out validation at a scale
  where top-1 success cannot be explained by a single group.

Verification:

```text
conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q
218 passed
```

Leave-one-source-root ranker smoke:

```text
scripts/eval_c2c_v2_task_frame_v46_candidate_ranker_loo.py
runtime_artifacts/coarse2contact_v2/v46_yaw_visible_z08_four_basic_ranker_loo_smoke
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_four_basic_yaw_collateral_ranker_loo_smoke_summary.json
```

Implementation note:

- `scripts/train_c2c_v2_task_frame_v46_candidate_ranker.py` now defaults its
  unused `test_fraction` to `0.0` for ranker training, so a source-held-out
  validation run does not silently drop an extra random source root from train.
- A `--test_fraction` override remains available for explicit three-way
  train/val/test experiments.
- The LOO summary now reports `top1_minus_zero_*`,
  `top1_beats_zero_*_rate`, `top1_worse_than_zero_*_rate`, and `worst_folds`.
  This makes same-window zero drift an explicit baseline instead of relying on
  raw contraction alone.

LOO smoke setup:

- dataset:
  `task_frame_alignment_v46_yaw_visible_z08_command_sweep_smoke_four_basic_plus_zero_manifest.jsonl`
- folds: `4`
- held out unit: one `source_eval_root` per fold
- rank score mode: `yaw_collateral`
- device: `cpu`
- epochs per fold: `30`

LOO smoke result:

- mean/min/max validation top1 best-score match: `1.0 / 1.0 / 1.0`
- mean/min/max validation top1 XY contraction: `1.0 / 1.0 / 1.0`
- mean/min/max validation top1 Z contraction: `1.0 / 1.0 / 1.0`
- mean/min/max validation top1 Yaw contraction: `1.0 / 1.0 / 1.0`
- mean/min/max validation top1 combined contraction: `1.0 / 1.0 / 1.0`
- zero-policy mean combined contraction on the same folds: `0.5`
- fold-level `top1_minus_zero_combined_contraction`: `[0.0, 1.0, 0.0, 1.0]`
- fold-level `top1_minus_zero_yaw_contraction`: `[0.0, 1.0, 0.0, 1.0]`
- no fold was worse than zero on XY/Z/Yaw/combined in this four-root smoke

Interpretation:

- This is a stronger bookkeeping result than the earlier single explicit-val
  smoke: every one of the four current source roots can now be held out and
  evaluated reproducibly.
- It still does not promote the ranker or v46.  Four roots with one validation
  group each is a debugging pool, not a random/source-held-out success proof.
- The more precise reading is that two folds genuinely beat same-window zero
  on combined/Yaw contraction, while two folds only matched zero.  Future
  expanded pools should require positive worst-root `top1_minus_zero_*`, not
  only raw top1 contraction.
- The next evidence step is to add many more executed command-sweep roots,
  especially yaw-worsen / XY-collateral hard negatives and Z-guard oracle
  cases, then rerun this LOO utility and require worst-root metrics to stay
  positive before any closed-loop insert MP4/success claim.

Verification:

```text
conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q
220 passed
```

Ranker LOO gate integration:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_ranker_loo_summary.json
offline_gate_status: fail
promotion_status: fail_offline_gate
```

`scripts/summarize_c2c_v2_task_frame_v46_gate.py` now accepts optional
`--ranker_loo_json` inputs.  When provided, the gate checks:

- enough leave-one-source-root folds are present;
- worst-root top1 combined contraction beats same-window zero;
- worst-root top1 yaw contraction beats same-window zero;
- top1 is not worse than zero on any axis/combined metric.

Current four-root LOO gate result:

- ranker LOO folds: `4`
- minimum required folds: `10`
- worst-root top1 minus zero combined contraction: `0.0`
- worst-root top1 minus zero yaw contraction: `0.0`
- worse-than-zero folds: `0`
- ranker LOO violations:
  `insufficient_ranker_loo_folds`,
  `ranker_worst_root_combined_not_beating_zero`,
  `ranker_worst_root_yaw_not_beating_zero`

Interpretation:

- The current LOO ranker smoke is useful diagnostic evidence, but the formal
  gate now prevents it from being treated as promotion evidence.
- Next data collection should target at least `10+` executed source roots with
  yaw-worsen / XY-collateral hard negatives and Z-guard oracle cases, then
  rerun the same `--ranker_loo_json` gate and require worst-root
  `top1_minus_zero_combined/yaw` to become positive.

Five-root expansion smoke:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_selector_permitted_expand_root420_summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_selector_permitted_expand_root420_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_selector_permitted_expand_root420_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_yaw_collateral_ranker_loo_smoke_summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_ranker_loo_plus_root420_summary.json
```

Executed batch:

- source root:
  `large_xy_large_yaw_focus_retain_30k/chunk_000_000_007/eval`
- episode/step: `ep006 step117`
- spec rows:
  `420,421,423,424,426,427,428,429,430`
- candidates:
  `yaw_hyp_neg_0060`, `yaw_hyp_neg_0120`,
  `yaw_hyp_pos_0060`, `yaw_hyp_pos_0120`,
  `z_guard_neg_0015`, `z_guard_neg_0030`,
  `z_guard_pos_0015`, `z_guard_pos_0030`, `zero`
- canonical launcher:
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py`
- checkpoint:
  `pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`
- v46 checkpoint:
  `runtime_task_frame_alignment_v46_unified_candidate_broad_plus_yawbalanced_holdout.pt`
- MP4: front+wrist videos were written for all 9 rows
- success count: `9/9`
- close leak rows: `0`
- `uses_privileged_runtime=false`

Applied-transition summary for the new root:

- retained rows: `9`
- observed XY contraction: `0.0`
- observed Z contraction: `0.0`
- observed Yaw contraction: `0.2222222222222222`

Five-root LOO result after merging with the previous four-root pool:

- folds: `5`
- top1 best-score match mean/min/max: `0.2 / 0.0 / 1.0`
- top1 combined contraction mean/min/max: `0.2 / 0.0 / 1.0`
- top1 XY contraction mean/min/max: `0.6 / 0.0 / 1.0`
- top1 Z contraction mean/min/max: `0.8 / 0.0 / 1.0`
- top1 Yaw contraction mean/min/max: `0.4 / 0.0 / 1.0`
- worse-than-zero folds in gate: `2`
- worst-root top1 minus zero combined contraction: `-1.0`
- worst-root top1 minus zero yaw contraction: `-1.0`

Gate result with the five-root LOO report:

- `offline_gate_status=fail`
- `promotion_status=fail_offline_gate`
- ranker LOO violations:
  `insufficient_ranker_loo_folds`,
  `ranker_top1_match_below_gate`,
  `ranker_combined_contraction_below_gate`,
  `ranker_worst_root_combined_not_beating_zero`,
  `ranker_worst_root_yaw_not_beating_zero`,
  `ranker_top1_worse_than_zero`

Interpretation:

- The newly executed root is a useful hard negative: it shows the current
  yaw-collateral ranker can select commands that are worse than zero on held-out
  source roots.
- This moves the project forward by making the failure mode measurable under
  the formal gate. It does not move v46 toward promotion yet.
- The next data step is to execute at least five more source roots from
  `task_frame_alignment_v46_yaw_selector_permitted_command_sweep_spec.jsonl`,
  then retrain/evaluate a ranker objective that explicitly penalizes
  worse-than-zero held-out roots and not only oracle score matching on tiny
  pools.

Six-root expansion smoke:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_yaw_selector_permitted_expand_root490_summary.json
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_selector_permitted_expand_root490_manifest.jsonl
runtime_artifacts/coarse2contact_v2/datasets/task_frame_alignment_v46_yaw_selector_permitted_expand_root490_manifest.summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_490_yaw_collateral_ranker_loo_smoke_summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_ranker_loo_plus_root420_490_summary.json
```

Executed batch:

- source root:
  `large_xy_large_yaw_focus_retain_30k/chunk_001_008_015/eval`
- episode/step: `ep010 step096`
- spec rows:
  `490,491,493,494,496,497,498,499,500`
- candidates:
  `yaw_hyp_neg_0060`, `yaw_hyp_neg_0120`,
  `yaw_hyp_pos_0060`, `yaw_hyp_pos_0120`,
  `z_guard_neg_0015`, `z_guard_neg_0030`,
  `z_guard_pos_0015`, `z_guard_pos_0030`, `zero`
- canonical launcher:
  `conda run -n vla-adapter xvfb-run -a python scripts/evaluate_c2c_v2_rlbench.py`
- MP4: front+wrist videos were written for all 9 rows
- success count: `9/9`
- close leak rows: `0`
- `uses_privileged_runtime=false`

Applied-transition summary for root490:

- retained rows: `9`
- observed XY contraction: `0.0`
- observed Z contraction: `0.1111111111111111`
- observed Yaw contraction: `0.0`

Six-root LOO result after merging root420 and root490 with the previous
four-root pool:

- folds: `6`
- top1 best-score match mean/min/max: `0.3333333432674408 / 0.0 / 1.0`
- top1 combined contraction mean/min/max: `0.3333333432674408 / 0.0 / 1.0`
- top1 XY contraction mean/min/max: `0.6666666865348816 / 0.0 / 1.0`
- top1 Z contraction mean/min/max: `0.6666666865348816 / 0.0 / 1.0`
- top1 Yaw contraction mean/min/max: `0.5 / 0.0 / 1.0`
- worse-than-zero folds in gate: `2`
- worst-root top1 minus zero combined contraction: `-1.0`
- worst-root top1 minus zero yaw contraction: `-1.0`

Gate result with the six-root LOO report:

- `offline_gate_status=fail`
- `promotion_status=fail_offline_gate`
- ranker LOO violations:
  `insufficient_ranker_loo_folds`,
  `ranker_top1_match_below_gate`,
  `ranker_combined_contraction_below_gate`,
  `ranker_worst_root_combined_not_beating_zero`,
  `ranker_worst_root_yaw_not_beating_zero`,
  `ranker_top1_worse_than_zero`

Interpretation:

- root490 is another useful hard negative. It keeps close safety intact but
  shows no XY/Yaw contraction and only one Z contraction out of nine candidates.
- The six-root LOO pool confirms the current yaw-collateral ranker objective is
  not robust to held-out hard roots.  The next ranker objective should penalize
  worse-than-zero choices directly and should not be evaluated on fewer than
  `10` source roots.

## 2026-06-09 Status Update: Outcome + Pairwise Ranker Negative Result

The v46 candidate ranker now has an explicit same-window pairwise margin loss:

- oracle command should outrank non-oracle candidates by a configurable margin
- when zero/no-op exists, candidates that do not beat zero should not outrank
  zero
- the loss uses only offline pre/post transition labels; runtime inputs,
  strict handoff, and planner-owned close semantics are unchanged

Implementation:

```text
scripts/train_c2c_v2_task_frame_v46_candidate_ranker.py
scripts/eval_c2c_v2_task_frame_v46_candidate_ranker_loo.py
tests/test_coarse2contact_v2.py
```

New CLI knobs:

```text
--pairwise_margin
--pairwise_weight
--pairwise_zero_margin
```

Verification:

```text
conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q -k "candidate_ranker"
14 passed
```

The pairwise outcome-utility LOO was run on the same 10-source-root pool:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_490_595_700_805_910_outcome_pairwise_ranker_loo_smoke_summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_outcome_pairwise_ranker_loo_plus_root420_490_595_700_805_910_summary.json
```

Setup:

- rank score mode: `outcome_utility`
- zero guard margin/weight: `0.050 / 1.000`
- support score penalty: `0.250`
- outcome loss weight: `1.000`
- pairwise margin/weight: `0.050 / 1.000`
- device: `cpu`

Result:

- folds: `10`
- top1 oracle match: `0.70`
- top1 combined contraction: `0.60`
- top1 XY/Z/Yaw contraction: `0.40 / 0.60 / 0.80`
- zero combined/XY/Z/Yaw contraction: `0.50 / 0.20 / 0.40 / 0.60`
- gate: `offline_gate_status=fail`, `promotion_status=fail_offline_gate`
- worse-than-zero folds: `1`
- worst-root top1 minus zero combined/yaw:
  `-1.0 / -1.0`

The remaining worst fold is unchanged:

```text
source root:
runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep_hard_bucket_30k_validation/large_xy_large_yaw_focus_flush_30k/chunk_000_000_007/eval

oracle:   yaw_hyp_neg_0100
selected: z_guard_pos_0100
```

Interpretation:

- The explicit pairwise margin is useful infrastructure, but it did not fix
  the held-out z-positive versus yaw-negative confusion on the hard flush root.
- This rules out "missing a simple oracle/zero margin" as the primary blocker
  for the current 10-root pool.
- The next meaningful step is to add more source-held-out roots that resemble
  this failure mode, add source-root/domain uncertainty or pairwise hard-negative
  weighting, and improve candidate/context representation so the ranker can
  distinguish yaw-negative support from guarded-z support under large XY/Yaw
  flush shifts.
- Do not promote v46 and do not run promotion MP4s from this result.

## 2026-06-09 Status Update: Typed Command Context Ranker

The v46 command-transition model and ranker now support an optional extended
candidate-command feature vector while preserving old checkpoint/runtime
compatibility:

- default `command_feature_dim=6` keeps existing checkpoints and runtime calls
  unchanged
- `--command_feature_mode typed16` adds non-privileged candidate context:
  raw local command, candidate type bits (`zero`, `xy`, `z`, `yaw`), normalized
  XY/Z/Yaw magnitudes, Z/Yaw signs, and command norm
- checkpoints now save/load `command_feature_dim`; a runtime 6D command is
  padded when loading a wider command-feature checkpoint

Implementation:

```text
prismatic/robot/coarse2contact_v2/task_frame_v46_alignment.py
scripts/train_c2c_v2_task_frame_v46_candidate_ranker.py
scripts/eval_c2c_v2_task_frame_v46_candidate_ranker_loo.py
tests/test_coarse2contact_v2.py
```

Focused verification:

```text
conda run -n vla-adapter python -m pytest tests/test_coarse2contact_v2.py -q -k "candidate_ranker or extended_command_feature or command_transition"
17 passed
```

Typed16 + outcome/pairwise 10-root LOO:

```text
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_z08_plus_root420_490_595_700_805_910_typed16_outcome_pairwise_ranker_loo_smoke_summary.json
runtime_artifacts/coarse2contact_v2/reports/runtime_task_frame_alignment_v46_candidate_gate_with_typed16_outcome_pairwise_ranker_loo_plus_root420_490_595_700_805_910_summary.json
```

Setup:

- rank score mode: `outcome_utility`
- zero guard margin/weight: `0.050 / 1.000`
- support score penalty: `0.250`
- outcome loss weight: `1.000`
- pairwise margin/weight: `0.050 / 1.000`
- command feature mode: `typed16`
- device: `cpu`

Result:

- folds: `10`
- top1 oracle match: `0.80`
- top1 combined contraction: `0.70`
- top1 XY/Z/Yaw contraction: `0.30 / 0.60 / 0.80`
- zero combined/XY/Z/Yaw contraction: `0.50 / 0.20 / 0.40 / 0.60`
- worse-than-zero folds: `0`
- worst-root top1 minus zero combined/yaw: `0.0 / 0.0`
- gate: `offline_gate_status=fail`, `promotion_status=fail_offline_gate`

Interpretation:

- Typed candidate context is a real improvement over raw 6D command features:
  it fixes the previous hard-flush worse-than-zero failure on
  `large_xy_large_yaw_focus_flush_30k/chunk_000_000_007/eval`, where the model
  now selects the oracle `yaw_hyp_neg_0100` instead of `z_guard_pos_0100`.
- It is still not promotable.  The formal gate fails because worst-root
  combined/Yaw only ties zero instead of beating it, top1 match remains below
  the gate, and held-out XY contraction regresses from `0.40` to `0.30`.
- A new exposed weak slice is
  `large_xy_large_yaw_focus_retain_30k/chunk_000_000_007/eval`, where the
  oracle is `yaw_hyp_neg_0120` but typed16 selects `z_guard_pos_0030`, producing
  no XY/Z/Yaw/combined contraction.
- The next useful step is not another pure loss tweak.  It should collect more
  retain/flush large-XY/large-Yaw source roots and add a hard-negative or
  uncertainty-aware selector that can distinguish "safe Z guard" from "needed
  yaw-negative" without sacrificing XY contraction.

## 2026-06-11 Status Update: Belief + Forward-Model Route

The latest external-style method review is adopted as the next route, with one
naming correction: do not introduce another `v47`-style label for the next
candidate.  The next semantic target is:

```text
belief_forward_task_frame_candidate
```

New plan document:

```text
docs/C2C_V2_BELIEF_FORWARD_MODEL_PLAN.md
```

Core takeaways:

- The main blocker is not a missing scalar/ranker loss.  It is the gap between
  task-frame belief semantics and true closed-loop command consequences.
- Diagnostic/evaluation gates may be split by axis, but runtime safety gates
  may not.  XY-only/Z-only/yaw-observable/ambiguous-yaw-abstain milestones are
  training signals and promotion diagnostics only; they do not relax close.
- Yaw is conditionally controllable but often ambiguous or unobservable because
  of square symmetry, occlusion, and partial wrist views.
- The scarce supervision is same-window, zero-baselined executed transition
  data, not generic relabeled state snapshots.
- Typed candidate context was useful because it removed worse-than-zero folds,
  but it still does not prove positive worst-root beat-zero behavior and it
  regressed held-out XY.

Next target:

- build an observability-aware belief model that can say `correct`, `hold`,
  `reacquire`, or `probe`, rather than forcing a confident residual on every
  axis
- train an uncertainty-aware command-conditioned forward model that predicts
  post-command residual mean/logvar and selects commands only when conservative
  lower-confidence beat-zero margins pass hard collateral constraints
- treat `hold`, `reacquire`, and `probe` as open-only candidate action types,
  not close/handoff actions
- collect more retain/flush large-XY/large-Yaw, partial-view, yaw-observable,
  and yaw-ambiguous transition groups with zero/no-op baselines
- keep strict handoff and planner-owned close frozen

Gate before MP4/promotion:

- worst-root combined and yaw minus zero must be positive, not tied
- XY contraction must not regress relative to v42/v46 evidence
- yaw false positives on ambiguous/unobservable windows must stay controlled
- dyaw must be blocked when yaw is ambiguous/unobservable unless the action is
  an explicitly bounded open-only probe/reacquire candidate
- close leak count remains zero
