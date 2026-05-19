"""
Merge multiple student handoff-state v2 datasets with per-dataset sample-weight scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_npz(path: str) -> dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    return {k: np.asarray(arr[k]) for k in arr.files}


def _concat_fields(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise RuntimeError("need at least one dataset to merge")
    fields = list(chunks[0].keys())
    merged: dict[str, np.ndarray] = {}
    for key in fields:
        merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return merged


def _source_summary(data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    names = data.get("source_name")
    if names is None:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for src in np.unique(names).tolist():
        mask = names == src
        summary[str(src)] = {
            "rows": int(np.sum(mask)),
            "episodes": int(np.unique(data["episode_index"][mask]).size) if "episode_index" in data else 0,
            "teacher_ready_pos": int(np.nansum(data["teacher_truth_handoff_ready"][mask]))
            if "teacher_truth_handoff_ready" in data
            else 0,
            "ready_support_pos": int(np.nansum(data["ready_support"][mask])) if "ready_support" in data else 0,
            "near_xy_hard_pos": int(np.nansum(data["near_xy_hard"][mask])) if "near_xy_hard" in data else 0,
            "broad_xy_recovery_pos": int(np.nansum(data["broad_xy_recovery"][mask]))
            if "broad_xy_recovery" in data
            else 0,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", action="append", required=True)
    ap.add_argument("--weight_mult", action="append", type=float, required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", required=True)
    args = ap.parse_args()

    if len(args.dataset_npz) != len(args.weight_mult):
        raise RuntimeError("`--dataset_npz` and `--weight_mult` must have the same length")

    chunks: list[dict[str, np.ndarray]] = []
    input_summaries = []
    for path, mult in zip(args.dataset_npz, args.weight_mult):
        data = _load_npz(path)
        if "sample_weight" not in data:
            raise RuntimeError(f"{path} is missing `sample_weight`")
        data["sample_weight"] = data["sample_weight"].astype(np.float32) * float(mult)
        chunks.append(data)
        input_summaries.append(
            {
                "path": path,
                "weight_mult": float(mult),
                "rows": int(next(iter(data.values())).shape[0]),
            }
        )

    merged = _concat_fields(chunks)
    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **merged)

    meta = {
        "num_rows": int(next(iter(merged.values())).shape[0]),
        "input_datasets": input_summaries,
        "teacher_ready_pos": int(np.nansum(merged["teacher_truth_handoff_ready"]))
        if "teacher_truth_handoff_ready" in merged
        else 0,
        "ready_support_pos": int(np.nansum(merged["ready_support"])) if "ready_support" in merged else 0,
        "near_xy_hard_pos": int(np.nansum(merged["near_xy_hard"])) if "near_xy_hard" in merged else 0,
        "broad_xy_recovery_pos": int(np.nansum(merged["broad_xy_recovery"]))
        if "broad_xy_recovery" in merged
        else 0,
        "source_summaries": _source_summary(merged),
    }
    Path(args.meta_json).write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
