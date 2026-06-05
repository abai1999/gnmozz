# VLA2 Planner Baseline + Coarse2Contact v2

VLA2 is a clean planner-oriented fork of the original VLA project, with a
parallel Coarse2Contact v2 research line for high-precision local skills.

Project defaults:

- Runtime environment: `conda run -n vla-adapter ...`
- Fixed planner checkpoint:
  `/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt`

Mainline scope:

- frozen VLA planner training
- RLBench planner evaluation
- smoke tests with MP4 traces
- Coarse2Contact v2 task contracts, owner-by-stage runtime, basin recovery
  audits, and diagnostic learned modules
- future C2C recovery work should prove real runtime failure-tail closed-loop
  basin entry, not just improve shadow scores

Not part of the VLA2 mainline:

- alignment / student / residual chains
- teacher-oracle online runtime targets
- legacy diagnostic artifacts as positive training data
- replay-only recovery variants that do not prove closed-loop basin recovery

## Layout

- `data/` points to the shared dataset tree from the original project
- `pretrained_models/planner_checkpoints/` holds planner checkpoint links
- `outputs/` is reserved for VLA2 training outputs
- `eval_logs/` is reserved for VLA2 evaluation logs
- `runtime_artifacts/` is reserved for VLA2 eval artifacts
- `prismatic/robot/coarse2contact_v2/` contains the C2C v2 precision skill
  layer scaffold
- `configs/coarse2contact/tasks/` contains task-level precision contracts
- `docs/C2C_V2_PROJECT_STATUS.md` summarizes the current C2C v2 design,
  status, and known blockers for future analysis
- `docs/C2C_V2_RESEARCH_REVIEW_BRIEF.md` is the route-level review brief for
  checking whether C2C v2 has drifted from the research goal
- `docs/AI_REVIEW_GUIDE.md` is the short entry point for external AI reviewers
  that only have git access

## Quick start

Train the planner baseline:

```bash
bash scripts/run_planner_train_baseline.sh
```

Run planner evaluation:

```bash
bash scripts/run_planner_eval_baseline.sh
```

Run a 3-episode smoke test with MP4 output:

```bash
bash scripts/run_planner_smoke_3ep.sh
```

Run C2C v2 unit tests:

```bash
conda run -n vla-adapter python -m unittest tests.test_coarse2contact_v2 -v
```

Run the current C2C v2 basin recovery smoke:

```bash
MODE=basin_recovery_only bash scripts/run_c2c_v2_basin_recovery_3ep.sh
```

## Default task

The baseline is wired for `insert_onto_square_peg` by default.
Override `TASK_NAME`, `CHECKPOINT_DIR`, `RUN_ROOT_DIR`, or `OUTPUT_ROOT`
through the environment variables exposed in the wrapper scripts.

## Current C2C v2 status

C2C v2 is not yet a solved high-precision controller. The current scaffold is
designed to make the failure explicit and analyzable: learned depth apply is
diagnostic-only by default, runtime remains non-privileged, and the tightened
basin-state gate blocks control unless an axis is calibrated as trusted. The
latest smoke showed that all grasp axes are currently blocked by calibration,
so C2C does not yet take over real planner failure tails. See
`docs/C2C_V2_PROJECT_STATUS.md` before extending the system.
