#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge gripper_trace shard directories into one trace dir.")
    ap.add_argument("--input_dir", action="append", required=True, help="repeatable shard eval output dir")
    ap.add_argument("--output_dir", required=True, help="merged eval output dir")
    args = ap.parse_args()

    inputs = [Path(p).resolve() for p in args.input_dir]
    output_dir = Path(args.output_dir).resolve()
    trace_out = output_dir / "gripper_traces"
    trace_out.mkdir(parents=True, exist_ok=True)

    merged_manifest = {
        "source_dirs": [str(p) for p in inputs],
        "merged_trace_paths": [],
        "eval_results_paths": [],
    }

    seen = set()
    for shard_dir in inputs:
        if not shard_dir.exists():
            raise FileNotFoundError(str(shard_dir))
        eval_results = shard_dir / "eval_results.json"
        if eval_results.exists():
            merged_manifest["eval_results_paths"].append(str(eval_results))
        trace_dir = shard_dir / "gripper_traces"
        if not trace_dir.exists():
            continue
        for trace_path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
            if trace_path.name in seen:
                raise RuntimeError(f"duplicate trace file name while merging: {trace_path.name}")
            seen.add(trace_path.name)
            dest = trace_out / trace_path.name
            shutil.copy2(trace_path, dest)
            merged_manifest["merged_trace_paths"].append(str(dest))

    manifest_path = output_dir / "merged_trace_manifest.json"
    manifest_path.write_text(json.dumps(merged_manifest, indent=2, ensure_ascii=False))
    print(f"[merge_gripper_trace_dirs] wrote {manifest_path} traces={len(merged_manifest['merged_trace_paths'])}")


if __name__ == "__main__":
    main()
