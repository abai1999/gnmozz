#!/usr/bin/env python3
"""Build a failure-replay augmentation dataset for Coarse2Contact v2 recovery training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.recovery_augmentation import build_failure_replay_augmented_records, load_bias_template


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dataset", type=Path, required=True)
    ap.add_argument("--shadow_report", type=Path, required=True)
    ap.add_argument("--bias_template", type=Path, default=None)
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets_recovery_replay_aug"))
    ap.add_argument("--hard_fraction", type=float, default=0.35)
    ap.add_argument("--min_trajectories", type=int, default=6)
    ap.add_argument("--tail_rows", type=int, default=6)
    ap.add_argument("--replay_strengths", type=str, default="1.25,1.5,1.75")
    ap.add_argument("--drift_strengths", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--replay_modes", type=str, default="overshoot,oscillate,cross_couple")
    args = ap.parse_args()

    base_records = _load_jsonl(args.base_dataset)
    shadow_report = json.loads(args.shadow_report.read_text(encoding="utf-8"))
    bias_template = load_bias_template(args.bias_template) if args.bias_template else shadow_report
    replay_strengths = [float(x.strip()) for x in str(args.replay_strengths).split(",") if x.strip()]
    drift_strengths = [float(x.strip()) for x in str(args.drift_strengths).split(",") if x.strip()]
    replay_modes = [x.strip() for x in str(args.replay_modes).split(",") if x.strip()]
    combined, report = build_failure_replay_augmented_records(
        base_records,
        shadow_report=shadow_report,
        bias_template=bias_template,
        hard_fraction=args.hard_fraction,
        min_trajectories=args.min_trajectories,
        tail_rows=args.tail_rows,
        replay_strengths=replay_strengths,
        drift_strengths=drift_strengths,
        replay_modes=replay_modes,
    )

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "grasp_recovery_dataset_v3_replay_aug.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "base_dataset": str(args.base_dataset),
        "shadow_report": str(args.shadow_report),
        "bias_template": str(args.bias_template) if args.bias_template else "",
        "output_dataset": str(out_path),
        "num_base_rows": len(base_records),
        "num_total_rows": len(combined),
        "num_augmented_rows": len(combined) - len(base_records),
        **report,
    }
    (out_root / "grasp_recovery_dataset_v3_replay_aug_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(out_path)
    print(out_root / "grasp_recovery_dataset_v3_replay_aug_summary.json")


if __name__ == "__main__":
    main()
