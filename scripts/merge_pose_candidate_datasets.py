"""
merge_pose_candidate_datasets.py

Merge multiple pose-field candidate datasets into one npz.
"""

import argparse
from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--repeat_last", type=int, default=1)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args()

    inputs = [Path(x) for x in args.inputs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded = [load_npz(p) for p in inputs]
    all_keys = sorted(set().union(*(item.keys() for item in loaded)))

    ref_specs = {}
    incompatible_keys = []
    for key in all_keys:
        shapes = []
        for item in loaded:
            if key in item:
                shapes.append(tuple(item[key].shape[1:]))
        if len(set(shapes)) > 1:
            incompatible_keys.append((key, shapes))
            continue
        for item in loaded:
            if key in item:
                ref_specs[key] = (item[key].shape[1:], item[key].dtype)
                break

    if incompatible_keys and args.strict:
        raise ValueError(f"Incompatible keys: {incompatible_keys}")
    if incompatible_keys:
        print("[merge_pose_candidate_datasets] skipping incompatible keys:")
        for key, shapes in incompatible_keys:
            print(f"  - {key}: {shapes}")
        all_keys = [k for k in all_keys if k in ref_specs]

    merged = {}
    for key in all_keys:
        arrays = []
        for idx, item in enumerate(loaded):
            if key in item:
                arr = item[key]
            else:
                tail_shape, dtype = ref_specs[key]
                length = next(iter(item.values())).shape[0]
                arr = np.zeros((length, *tail_shape), dtype=dtype)
            repeat = int(args.repeat_last) if (idx == len(loaded) - 1 and len(loaded) > 1) else 1
            if repeat > 1:
                arr = np.repeat(arr, repeat, axis=0)
            arrays.append(arr)
        merged[key] = np.concatenate(arrays, axis=0)

    np.savez_compressed(output, **merged)
    print(f"[merge_pose_candidate_datasets] saved {output}")


if __name__ == "__main__":
    main()
