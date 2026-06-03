#!/usr/bin/env python3
"""Find which validation chunks contain active rows for selected episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_filter(text: str) -> set[int]:
    return {int(part.strip()) for part in str(text).split(",") if part.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Find small-bucket episode sources.")
    ap.add_argument("--root", type=Path, default=Path("/home/guoning/code/VLA2/runtime_artifacts/coarse2contact_v2"))
    ap.add_argument("--run_glob", type=str, required=True)
    ap.add_argument("--episodes", type=str, default="4,16,18,27,29")
    args = ap.parse_args()

    eps = _episode_filter(args.episodes)
    for run_dir in sorted(args.root.glob(args.run_glob)):
        if not run_dir.is_dir():
            continue
        counts = {ep: 0 for ep in eps}
        chunk_files = sorted(run_dir.glob("chunk_*/audit/grasp_probe_intervention_audit.json"))
        for chunk_file in chunk_files:
            data = _read_json(chunk_file)
            for item in data.get("by_episode", []):
                ep = int(item.get("episode_idx", -1))
                if ep in counts:
                    counts[ep] += int(item.get("active_count", 0))
        if any(counts.values()):
            print(json.dumps({"run_dir": str(run_dir), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
