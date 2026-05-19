#!/usr/bin/env python3
"""
Deep-dive audit for planner data by phase bucket and close-transition windows.

Uses the same reconstructed 7D action semantics as the RLBench training dataset.
Outputs:
- phase-bucket motion statistics
- close-transition vs non-transition motion stats
- pre-close window stats
- a conservative reweight/filter suggestion for planner training data
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PHASE_NAME = {0: "Reach", 1: "Grasp", 2: "Transfer", 3: "Refine"}


def load_ep(ep_dir: Path) -> dict[str, np.ndarray]:
    z = np.load(ep_dir / "model_inputs.npz")
    return {k: z[k] for k in z.files}


def action7(ep: dict[str, np.ndarray]) -> np.ndarray:
    at = ep["action_targets"]
    if at.shape[1] == 7:
        return at.astype(np.float32)

    gp = ep["gripper_pose"]
    go = ep["gripper_open"]
    T = gp.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        if t < T - 1:
            delta_pos = gp[t + 1, :3] - gp[t, :3]
            r0 = Rotation.from_quat(gp[t, 3:7])
            r1 = Rotation.from_quat(gp[t + 1, 3:7])
            delta_rv = (r1 * r0.inv()).as_rotvec().astype(np.float32)
            gripper = go[t + 1, 0]
        else:
            delta_pos = np.zeros(3, dtype=np.float32)
            delta_rv = np.zeros(3, dtype=np.float32)
            gripper = go[t, 0]
        out[t] = np.concatenate([delta_pos, delta_rv, [gripper]])
    return out


def pstats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_root", type=Path, default=Path("data/rlbench_data/insert_onto_square_peg/train/episodes"))
    ap.add_argument("--transition_radius", type=int, default=5)
    ap.add_argument("--preclose_radius", type=int, default=8)
    ap.add_argument("--output_json", type=Path, default=Path("runtime_artifacts/stage_refiner/planner_phase_rotation_audit_20260426a.json"))
    args = ap.parse_args()

    phase_pos = defaultdict(list)
    phase_rot = defaultdict(list)
    phase_grip = defaultdict(list)
    bucket_pos = defaultdict(list)
    bucket_rot = defaultdict(list)
    bucket_grip = defaultdict(list)
    phase_counts = defaultdict(int)
    transition_phase_counts = defaultdict(int)
    preclose_phase_counts = defaultdict(int)

    ep_dirs = sorted(args.episodes_root.glob("episode*"))
    for ep_dir in ep_dirs:
        ep = load_ep(ep_dir)
        actions = action7(ep)
        phase_ids = np.load(ep_dir / "phase_ids.npy")
        gripper_open = ep["gripper_open"].reshape(-1)

        pos_norm = np.linalg.norm(actions[:, :3], axis=1)
        rot_norm = np.linalg.norm(actions[:, 3:6], axis=1)
        grip_delta = np.abs(np.diff(actions[:, 6], prepend=actions[:1, 6]))

        close_idxs = np.flatnonzero(np.abs(np.diff(gripper_open, prepend=gripper_open[:1])) > 0.5)
        transition_mask = np.zeros(len(actions), dtype=bool)
        preclose_mask = np.zeros(len(actions), dtype=bool)
        for t in close_idxs:
            transition_mask[max(0, t - args.transition_radius) : min(len(actions), t + args.transition_radius + 1)] = True
            preclose_mask[max(0, t - args.preclose_radius) : t] = True

        non_transition_mask = ~transition_mask

        for pid in sorted(PHASE_NAME):
            m = phase_ids == pid
            phase_counts[pid] += int(m.sum())
            transition_phase_counts[pid] += int((m & transition_mask).sum())
            preclose_phase_counts[pid] += int((m & preclose_mask).sum())
            phase_pos[pid].append(pos_norm[m])
            phase_rot[pid].append(rot_norm[m])
            phase_grip[pid].append(grip_delta[m])

        buckets = {
            "close_transition": transition_mask,
            "preclose_only": preclose_mask,
            "non_transition": non_transition_mask,
        }
        for name, mask in buckets.items():
            bucket_pos[name].append(pos_norm[mask])
            bucket_rot[name].append(rot_norm[mask])
            bucket_grip[name].append(grip_delta[mask])

    report = {
        "phase_counts": {PHASE_NAME[k]: int(v) for k, v in phase_counts.items()},
        "transition_phase_counts": {PHASE_NAME[k]: int(v) for k, v in transition_phase_counts.items()},
        "preclose_phase_counts": {PHASE_NAME[k]: int(v) for k, v in preclose_phase_counts.items()},
        "phase_stats": {},
        "bucket_stats": {},
        "reweight_filtering_recommendation": {},
    }

    for pid in sorted(PHASE_NAME):
        name = PHASE_NAME[pid]
        report["phase_stats"][name] = {
            "pos_norm": pstats(np.concatenate(phase_pos[pid])),
            "rot_norm": pstats(np.concatenate(phase_rot[pid])),
            "grip_delta": pstats(np.concatenate(phase_grip[pid])),
            "transition_frame_ratio_within_phase": float(transition_phase_counts[pid] / max(1, phase_counts[pid])),
            "preclose_frame_ratio_within_phase": float(preclose_phase_counts[pid] / max(1, phase_counts[pid])),
        }

    for name in ("close_transition", "preclose_only", "non_transition"):
        report["bucket_stats"][name] = {
            "pos_norm": pstats(np.concatenate(bucket_pos[name])),
            "rot_norm": pstats(np.concatenate(bucket_rot[name])),
            "grip_delta": pstats(np.concatenate(bucket_grip[name])),
        }

    # Conservative suggestion:
    # - don't upweight close-transition frames blindly if they are mostly static
    # - upweight preclose frames modestly
    # - no rotation saturation filtering at 7D action level unless a real tail exists
    preclose_rot_p95 = report["bucket_stats"]["preclose_only"]["rot_norm"].get("p95", 0.0)
    nontrans_rot_p95 = report["bucket_stats"]["non_transition"]["rot_norm"].get("p95", 0.0)
    close_rot_p95 = report["bucket_stats"]["close_transition"]["rot_norm"].get("p95", 0.0)

    recommendation = {
        "summary": (
            "Use phase/bucket reweighting rather than hard rotation-saturation filtering. "
            "Close-transition frames are mostly low-motion; blindly boosting them risks making the planner more static."
        ),
        "sample_weight_policy": {
            "Reach": 1.0,
            "Grasp": 1.4,
            "Transfer": 1.0,
            "Refine": 1.2,
            "preclose_only_multiplier": 1.8,
            "close_transition_multiplier": 1.15,
            "non_transition_multiplier": 1.0,
        },
        "filter_policy": {
            "drop_near_zero_frames_globally": False,
            "drop_close_transition_static_frames": False,
            "rotation_saturation_filter_needed": False,
            "reason": "Reconstructed 7D planner actions do not show true rotation saturation tails near 1.0.",
        },
        "evidence": {
            "preclose_rot_p95": preclose_rot_p95,
            "close_transition_rot_p95": close_rot_p95,
            "non_transition_rot_p95": nontrans_rot_p95,
            "close_transition_pos_p95": report["bucket_stats"]["close_transition"]["pos_norm"].get("p95"),
            "preclose_pos_p95": report["bucket_stats"]["preclose_only"]["pos_norm"].get("p95"),
        },
    }
    report["reweight_filtering_recommendation"] = recommendation

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\n[ok] wrote {args.output_json}")


if __name__ == "__main__":
    main()
