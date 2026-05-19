#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate check before B2: teacher_ready_eps/rows thresholds.")
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--min_teacher_ready_eps", type=int, default=3)
    ap.add_argument("--min_teacher_ready_rows", type=int, default=20)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    arr = np.load(args.dataset_npz, allow_pickle=False)
    data = {k: np.asarray(arr[k]) for k in arr.files}
    teacher_ready = np.asarray(data["teacher_truth_handoff_ready"], dtype=np.float32) > 0.5
    episode_index = np.asarray(data["episode_index"], dtype=np.int64)

    teacher_ready_rows = int(np.sum(teacher_ready))
    teacher_ready_eps = int(np.unique(episode_index[teacher_ready]).size) if teacher_ready_rows > 0 else 0
    passed = (teacher_ready_eps >= int(args.min_teacher_ready_eps)) and (
        teacher_ready_rows >= int(args.min_teacher_ready_rows)
    )

    result = {
        "dataset_npz": str(args.dataset_npz),
        "teacher_ready_rows": teacher_ready_rows,
        "teacher_ready_eps": teacher_ready_eps,
        "min_teacher_ready_eps": int(args.min_teacher_ready_eps),
        "min_teacher_ready_rows": int(args.min_teacher_ready_rows),
        "passed": bool(passed),
        "decision": "proceed_phaseA_readyfirst" if passed else "collect_more_do_not_enter_B2",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
