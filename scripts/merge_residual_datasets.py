"""
merge_residual_datasets.py

Merge multiple residual dataset directories into one directory by copying and
renumbering residual shards, and write a lightweight aggregate meta summary.
"""

import argparse
import json
import shutil
from pathlib import Path


def load_meta(path: Path) -> dict:
    meta_path = path / "residual_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def sum_counter_fields(metas, keys):
    out = {}
    for key in keys:
        merged = {}
        for meta in metas:
            counts = meta.get(key, {}) or {}
            for k, v in counts.items():
                sk = str(k)
                merged[sk] = merged.get(sk, 0) + int(v)
        if merged:
            out[key] = merged
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.inputs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    metas = []
    input_summaries = []
    for input_dir in input_dirs:
        metas.append(load_meta(input_dir))
        shard_files = sorted(input_dir.glob("residual_shard_*.npz"))
        input_summaries.append(
            {
                "input_dir": str(input_dir),
                "num_shards": len(shard_files),
            }
        )
        for shard_path in shard_files:
            dst = output_dir / f"residual_shard_{shard_idx:04d}.npz"
            shutil.copy2(shard_path, dst)
            shard_idx += 1

    merged_meta = {
        "source": "merged_residual_datasets",
        "input_dirs": [str(p) for p in input_dirs],
        "num_input_dirs": len(input_dirs),
        "num_shards": shard_idx,
        "input_summaries": input_summaries,
        "num_samples": int(sum(int(meta.get("num_samples", 0)) for meta in metas)),
        "allow_close_count": int(sum(int(meta.get("allow_close_count", 0)) for meta in metas)),
        "hold_positive_count": int(sum(int(meta.get("hold_positive_count", 0)) for meta in metas)),
        "block_open_count": int(sum(int(meta.get("block_open_count", 0)) for meta in metas)),
    }
    merged_meta.update(
        sum_counter_fields(
            metas,
            [
                "phase_counts",
                "stage_role_counts",
                "failure_counts",
                "readiness_counts",
                "basin_positive_counts",
                "hold_counts",
                "negative_reason_counts",
                "planner_close_intent_counts",
                "gripper_state_counts",
                "support_source_counts",
            ],
        )
    )

    (output_dir / "residual_meta.json").write_text(json.dumps(merged_meta, indent=2))
    print(f"[merge_residual_datasets] saved {output_dir}")


if __name__ == "__main__":
    main()
