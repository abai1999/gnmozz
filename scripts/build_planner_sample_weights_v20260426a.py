#!/usr/bin/env python3
"""
Build optional per-frame sample weights for RLBench planner training.

Goal:
- modestly emphasize close-adjacent planning windows without pushing the planner
  into static/template behavior
- emphasize Grasp / Refine slightly
- do not hard-filter frames or apply fake "rotation saturation" filters when the
  reconstructed 7D action distribution does not support that hypothesis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHASE_NAME = {0: "Reach", 1: "Grasp", 2: "Transfer", 3: "Refine"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_root", type=Path, default=Path("data/rlbench_data/insert_onto_square_peg/train/episodes"))
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/stage_refiner/planner_sample_weights_v20260426a"),
    )
    ap.add_argument("--transition_radius", type=int, default=5)
    ap.add_argument("--preclose_radius", type=int, default=8)
    ap.add_argument("--reach_weight", type=float, default=1.0)
    ap.add_argument("--grasp_weight", type=float, default=1.4)
    ap.add_argument("--transfer_weight", type=float, default=1.0)
    ap.add_argument("--refine_weight", type=float, default=1.2)
    ap.add_argument("--preclose_multiplier", type=float, default=1.8)
    ap.add_argument("--close_transition_multiplier", type=float, default=1.15)
    ap.add_argument("--normalize_mean_to_one", action="store_true", default=True)
    return ap.parse_args()


def episode_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("episode*"), key=lambda p: int(p.name.replace("episode", "")))


def transition_masks(gripper_open: np.ndarray, transition_radius: int, preclose_radius: int) -> tuple[np.ndarray, np.ndarray]:
    g = gripper_open.reshape(-1)
    transition_mask = np.zeros(len(g), dtype=bool)
    preclose_mask = np.zeros(len(g), dtype=bool)
    close_idxs = np.flatnonzero(np.abs(np.diff(g, prepend=g[:1])) > 0.5)
    for t in close_idxs:
        transition_mask[max(0, t - transition_radius) : min(len(g), t + transition_radius + 1)] = True
        preclose_mask[max(0, t - preclose_radius) : t] = True
    preclose_mask &= ~transition_mask
    return transition_mask, preclose_mask


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase_weight = {
        0: args.reach_weight,
        1: args.grasp_weight,
        2: args.transfer_weight,
        3: args.refine_weight,
    }

    weights_all = []
    manifest = []
    bucket_counts = {"close_transition": 0, "preclose_only": 0, "plain": 0}
    phase_counts = {PHASE_NAME[k]: 0 for k in PHASE_NAME}

    for ep_dir in episode_dirs(args.episodes_root):
        npz_path = ep_dir / "model_inputs.npz"
        phase_path = ep_dir / "phase_ids.npy"
        if not npz_path.exists() or not phase_path.exists():
            continue

        npz = np.load(npz_path)
        action_targets = npz["action_targets"]
        n_frames = int(action_targets.shape[0])
        if n_frames < 1:
            continue

        phase_ids = np.load(phase_path).astype(np.int64)
        gripper_open = npz["gripper_open"].astype(np.float32)
        if len(phase_ids) != n_frames:
            raise ValueError(f"{ep_dir}: phase_ids length {len(phase_ids)} != action_targets length {n_frames}")

        transition_mask, preclose_mask = transition_masks(
            gripper_open, transition_radius=args.transition_radius, preclose_radius=args.preclose_radius
        )

        ep_weights = np.ones(n_frames, dtype=np.float64)
        for pid in PHASE_NAME:
            m = phase_ids == pid
            ep_weights[m] *= phase_weight[pid]
            phase_counts[PHASE_NAME[pid]] += int(m.sum())

        ep_weights[transition_mask] *= args.close_transition_multiplier
        ep_weights[preclose_mask] *= args.preclose_multiplier

        bucket_counts["close_transition"] += int(transition_mask.sum())
        bucket_counts["preclose_only"] += int(preclose_mask.sum())
        bucket_counts["plain"] += int((~transition_mask & ~preclose_mask).sum())

        start = len(weights_all)
        weights_all.extend(ep_weights.tolist())
        manifest.append(
            {
                "episode": ep_dir.name,
                "num_frames": n_frames,
                "global_start": start,
                "global_end": start + n_frames,
                "mean_weight": float(ep_weights.mean()),
                "min_weight": float(ep_weights.min()),
                "max_weight": float(ep_weights.max()),
                "transition_frames": int(transition_mask.sum()),
                "preclose_frames": int(preclose_mask.sum()),
            }
        )

    sample_weights = np.asarray(weights_all, dtype=np.float64)
    if args.normalize_mean_to_one and sample_weights.size > 0:
        sample_weights /= float(sample_weights.mean())

    np.save(args.output_dir / "sample_weights.npy", sample_weights.astype(np.float32))

    meta = {
        "policy_name": "planner_phase_bucket_reweight_v20260426a",
        "summary": (
            "Optional planner-training sample weights. Uses modest phase and close-adjacent reweighting; "
            "no hard filtering and no rotation-saturation drop, because reconstructed 7D actions do not show "
            "true saturation near 1.0."
        ),
        "episodes_root": str(args.episodes_root),
        "num_frames": int(sample_weights.size),
        "weight_stats": {
            "min": float(sample_weights.min()) if sample_weights.size else 0.0,
            "mean": float(sample_weights.mean()) if sample_weights.size else 0.0,
            "max": float(sample_weights.max()) if sample_weights.size else 0.0,
            "p50": float(np.percentile(sample_weights, 50)) if sample_weights.size else 0.0,
            "p90": float(np.percentile(sample_weights, 90)) if sample_weights.size else 0.0,
            "p99": float(np.percentile(sample_weights, 99)) if sample_weights.size else 0.0,
        },
        "phase_weights": {
            PHASE_NAME[0]: args.reach_weight,
            PHASE_NAME[1]: args.grasp_weight,
            PHASE_NAME[2]: args.transfer_weight,
            PHASE_NAME[3]: args.refine_weight,
        },
        "window_weights": {
            "close_transition_multiplier": args.close_transition_multiplier,
            "preclose_multiplier": args.preclose_multiplier,
        },
        "bucket_counts": bucket_counts,
        "phase_counts": phase_counts,
        "notes": {
            "rotation_saturation_filter_applied": False,
            "near_zero_drop_applied": False,
            "close_transition_drop_applied": False,
        },
        "episodes": manifest,
    }

    with open(args.output_dir / "sample_weights_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta["weight_stats"], indent=2))
    print(f"[ok] wrote {args.output_dir / 'sample_weights.npy'}")
    print(f"[ok] wrote {args.output_dir / 'sample_weights_meta.json'}")


if __name__ == "__main__":
    main()
