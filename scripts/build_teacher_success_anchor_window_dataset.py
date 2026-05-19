"""
Build a window-focused teacher-success anchor dataset.

This trims the broad teacher-success anchor set down to the rows that are most
useful for Phase-A:
  - ready-support / teacher-ready rows
  - very-near-to-ready rows (teacher_band_label >= 1)

The goal is to preserve success-band semantics without letting the broad anchor
distribution dominate training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", required=True)
    args = ap.parse_args()

    data = np.load(args.input_npz, allow_pickle=False)
    fields = {k: np.asarray(data[k]) for k in data.files}

    teacher_ready = fields.get("teacher_truth_handoff_ready", np.zeros(0, dtype=np.float32)) > 0.5
    ready_support = fields.get("ready_support", np.zeros_like(teacher_ready, dtype=np.float32)) > 0.5
    teacher_band = fields.get("teacher_band_label", np.zeros_like(teacher_ready, dtype=np.int64))
    very_near_or_ready = teacher_band >= 1

    mask = teacher_ready | ready_support | very_near_or_ready
    if not np.any(mask):
        raise RuntimeError("anchor window mask is empty")

    out: dict[str, np.ndarray] = {}
    for key, arr in fields.items():
        out[key] = arr[mask]

    sample_weight = out.get("sample_weight", np.ones(int(np.sum(mask)), dtype=np.float32)).astype(np.float32)
    teacher_ready_out = out.get("teacher_truth_handoff_ready", np.zeros_like(sample_weight)) > 0.5
    ready_support_out = out.get("ready_support", np.zeros_like(sample_weight)) > 0.5
    teacher_band_out = out.get("teacher_band_label", np.zeros_like(sample_weight, dtype=np.int64))

    # Keep the windows focused: ready rows get the strongest per-row emphasis,
    # very-near rows get a lighter boost.
    sample_weight = sample_weight * np.where(teacher_band_out >= 1, 1.35, 1.0).astype(np.float32)
    sample_weight = sample_weight * np.where(ready_support_out | teacher_ready_out, 1.60, 1.0).astype(np.float32)
    out["sample_weight"] = sample_weight.astype(np.float32)

    window_kind = np.full(sample_weight.shape[0], "very_near", dtype="<U24")
    window_kind[teacher_band_out >= 2] = "release_ready"
    window_kind[ready_support_out] = "ready_support"
    window_kind[teacher_ready_out] = "teacher_ready"
    out["anchor_window_kind"] = window_kind

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    meta = {
        "input_npz": args.input_npz,
        "num_rows": int(sample_weight.shape[0]),
        "teacher_ready_pos": int(np.sum(teacher_ready_out)),
        "ready_support_pos": int(np.sum(ready_support_out)),
        "very_near_or_ready_pos": int(np.sum(teacher_band_out >= 1)),
        "source_summaries": {},
    }
    if "source_name" in out:
        for src in np.unique(out["source_name"]).tolist():
            src_mask = out["source_name"] == src
            meta["source_summaries"][str(src)] = {
                "rows": int(np.sum(src_mask)),
                "teacher_ready_pos": int(np.sum(teacher_ready_out[src_mask])),
                "ready_support_pos": int(np.sum(ready_support_out[src_mask])),
                "very_near_or_ready_pos": int(np.sum((teacher_band_out >= 1)[src_mask])),
            }
    Path(args.meta_json).write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

