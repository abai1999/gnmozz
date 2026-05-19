#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge shard support_states npz files by row concat.")
    ap.add_argument("--input_npz", action="append", required=True, help="repeatable input npz path")
    ap.add_argument("--output_npz", required=True)
    args = ap.parse_args()

    inputs = [Path(p).resolve() for p in args.input_npz]
    if not inputs:
        raise RuntimeError("no inputs")
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(str(p))

    first = np.load(str(inputs[0]), allow_pickle=False)
    keys = list(first.files)
    merged: dict[str, list[np.ndarray]] = {k: [np.asarray(first[k])] for k in keys}
    first.close()

    for p in inputs[1:]:
        d = np.load(str(p), allow_pickle=False)
        cur_keys = list(d.files)
        if cur_keys != keys:
            d.close()
            raise RuntimeError(
                f"schema mismatch for {p}\nexpected keys={keys}\nactual keys={cur_keys}"
            )
        for k in keys:
            merged[k].append(np.asarray(d[k]))
        d.close()

    out = {}
    row_count = None
    for k in keys:
        arr = np.concatenate(merged[k], axis=0)
        out[k] = arr
        if row_count is None:
            row_count = int(arr.shape[0])

    output = Path(args.output_npz).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **out)
    print(f"[merge_support_states_npz] wrote {output} rows={row_count} keys={len(keys)}")


if __name__ == "__main__":
    main()

