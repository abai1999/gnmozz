#!/usr/bin/env python3
"""Merge row-wise npz shards into a single npz.

This utility is intentionally conservative:
- arrays with a leading row dimension are concatenated on axis 0 when possible
- scalar / zero-d arrays are taken from the first shard
- missing keys are ignored if a shard doesn't have them

It is meant for collector outputs where each shard stores the same schema and
the first axis corresponds to rows/examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _merge_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("No arrays to merge")
    if len(arrays) == 1:
        return arrays[0]
    first = arrays[0]
    if first.shape == ():
        return first
    try:
        return np.concatenate(arrays, axis=0)
    except Exception:
        # Fall back to object concatenation when row concatenation is not legal.
        return np.asarray(list(arrays), dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    shard_dicts = [np.load(p, allow_pickle=True) for p in args.inputs]
    keys = sorted(set().union(*(d.files for d in shard_dicts)))
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        vals = [d[key] for d in shard_dicts if key in d.files]
        if not vals:
            continue
        merged[key] = _merge_arrays(vals)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    print(f"Wrote {output} with keys={list(merged.keys())}")


if __name__ == "__main__":
    main()
