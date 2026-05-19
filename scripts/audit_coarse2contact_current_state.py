#!/usr/bin/env python3
"""Write a compact audit JSON for the current Coarse2Contact asset state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _count_episodes(root: Path) -> int:
    episodes_root = root / "train" / "episodes"
    if not episodes_root.exists():
        return 0
    return sum(1 for p in episodes_root.iterdir() if p.is_dir() and p.name.startswith("episode"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument(
        "--output_json",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact/audit_current_state.json"),
    )
    args = ap.parse_args()

    root = args.root.resolve()
    data_root = root / "data" / "rlbench_data"
    task_root = data_root / "insert_onto_square_peg"
    spoke_root = data_root / "insert_onto_square_spoke"
    ckpt_root = root / "pretrained_models" / "planner_checkpoints"
    c2c_eval = root / "scripts" / "evaluate_rlbench.py"
    task_map_has_spoke = "insert_onto_square_spoke" in c2c_eval.read_text(encoding="utf-8")

    report = {
        "root": str(root),
        "task": "insert_onto_square_peg",
        "insert_onto_square_peg_exists": task_root.exists(),
        "insert_onto_square_peg_episode_count": _count_episodes(task_root),
        "insert_onto_square_spoke_exists": spoke_root.exists(),
        "insert_onto_square_spoke_episode_count": _count_episodes(spoke_root),
        "planner_checkpoint_dir": str(ckpt_root / "insert_onto_square_peg_30000_chkpt"),
        "planner_checkpoint_exists": (ckpt_root / "insert_onto_square_peg_30000_chkpt").exists(),
        "insert_onto_square_spoke_checkpoint_exists": any(
            p.name.startswith("insert_onto_square_spoke") for p in ckpt_root.glob("*")
        ),
        "evaluate_rlbench_has_square_spoke_task": task_map_has_spoke,
        "uses_privileged_target": False,
        "notes": [
            "square_spoke is treated as a missing asset in the current workspace",
            "planner baseline remains wired to square_peg",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
