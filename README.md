# VLA2 Planner Baseline

VLA2 is a clean planner-only fork of the original VLA project.

Mainline scope:

- frozen VLA planner training
- RLBench planner evaluation
- smoke tests with MP4 traces

Not part of the VLA2 mainline:

- alignment / student / residual chains
- teacher-oracle online runtime targets
- legacy diagnostic artifacts as positive training data

## Layout

- `data/` points to the shared dataset tree from the original project
- `pretrained_models/planner_checkpoints/` holds planner checkpoint links
- `outputs/` is reserved for VLA2 training outputs
- `eval_logs/` is reserved for VLA2 evaluation logs
- `runtime_artifacts/` is reserved for VLA2 eval artifacts

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

## Default task

The baseline is wired for `insert_onto_square_peg` by default.
Override `TASK_NAME`, `CHECKPOINT_DIR`, `RUN_ROOT_DIR`, or `OUTPUT_ROOT`
through the environment variables exposed in the wrapper scripts.
