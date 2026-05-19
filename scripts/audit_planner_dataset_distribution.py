#!/usr/bin/env python3
"""
Audit planner-training data distribution for RLBench and compare with LIBERO demos when available.

Focus areas:
1. near-zero / no-op / homing-frame prevalence
2. action chunk magnitude distribution
3. whether close-transition windows are swamped by static frames
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Thresholds:
    pos_eps: float = 1e-3
    rot_eps: float = 2e-2
    grip_eps: float = 5e-2
    transition_window: int = 5


def _episode_npzs(episodes_root: Path) -> list[Path]:
    return sorted(episodes_root.glob("episode*/model_inputs.npz"))


def _load_rlbench_episode(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(npz_path)
    return {k: z[k] for k in z.files}


def _rlbench_action7(ep: dict[str, np.ndarray]) -> np.ndarray:
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


def _step_metrics(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_norm = np.linalg.norm(actions[:, :3], axis=1)
    rot_norm = np.linalg.norm(actions[:, 3:6], axis=1)
    grip = actions[:, 6]
    grip_delta = np.abs(np.diff(grip, prepend=grip[:1]))
    return pos_norm, rot_norm, grip, grip_delta


def _is_near_zero(pos_norm: np.ndarray, rot_norm: np.ndarray, grip_delta: np.ndarray, th: Thresholds) -> np.ndarray:
    return (pos_norm <= th.pos_eps) & (rot_norm <= th.rot_eps) & (grip_delta <= th.grip_eps)


def _first_active_index(near_zero_mask: np.ndarray) -> int:
    idx = np.flatnonzero(~near_zero_mask)
    return int(idx[0]) if idx.size else int(len(near_zero_mask))


def _close_transition_window(gripper_open: np.ndarray, radius: int) -> np.ndarray:
    g = gripper_open.reshape(-1)
    transitions = np.flatnonzero(np.abs(np.diff(g, prepend=g[:1])) > 0.5)
    mask = np.zeros_like(g, dtype=bool)
    for t in transitions:
        lo = max(0, int(t) - radius)
        hi = min(len(g), int(t) + radius + 1)
        mask[lo:hi] = True
    return mask


def _percentiles(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def audit_rlbench(episodes_root: Path, thresholds: Thresholds) -> dict:
    npzs = _episode_npzs(episodes_root)
    assert npzs, f"No model_inputs.npz found under {episodes_root}"

    total_frames = 0
    total_near_zero = 0
    total_transition_frames = 0
    total_transition_near_zero = 0
    total_non_transition_frames = 0
    total_non_transition_near_zero = 0
    first_active_indices = []
    episode_lengths = []
    episodes_with_close = 0
    episodes_all_near_zero = 0

    pos_all, rot_all, grip_delta_all = [], [], []
    pos_transition, rot_transition = [], []
    pos_non_transition, rot_non_transition = [], []

    per_episode = []
    for npz_path in npzs:
        ep = _load_rlbench_episode(npz_path)
        actions = _rlbench_action7(ep)
        gripper_open = ep["gripper_open"].astype(np.float32)
        pos_norm, rot_norm, _, grip_delta = _step_metrics(actions)
        near_zero = _is_near_zero(pos_norm, rot_norm, grip_delta, thresholds)
        transition_mask = _close_transition_window(gripper_open, thresholds.transition_window)

        first_active = _first_active_index(near_zero)
        first_active_indices.append(first_active)
        episode_lengths.append(len(actions))
        total_frames += len(actions)
        total_near_zero += int(near_zero.sum())

        if first_active >= len(actions):
            episodes_all_near_zero += 1

        if transition_mask.any():
            episodes_with_close += 1

        total_transition_frames += int(transition_mask.sum())
        total_transition_near_zero += int((near_zero & transition_mask).sum())
        total_non_transition_frames += int((~transition_mask).sum())
        total_non_transition_near_zero += int((near_zero & ~transition_mask).sum())

        pos_all.append(pos_norm)
        rot_all.append(rot_norm)
        grip_delta_all.append(grip_delta)
        pos_transition.append(pos_norm[transition_mask])
        rot_transition.append(rot_norm[transition_mask])
        pos_non_transition.append(pos_norm[~transition_mask])
        rot_non_transition.append(rot_norm[~transition_mask])

        per_episode.append(
            {
                "episode": npz_path.parent.name,
                "length": int(len(actions)),
                "first_active_index": int(first_active),
                "homing_ratio": float(first_active / max(1, len(actions))),
                "near_zero_ratio": float(near_zero.mean()),
                "has_close_transition": bool(transition_mask.any()),
                "transition_frame_ratio": float(transition_mask.mean()),
            }
        )

    pos_all = np.concatenate(pos_all)
    rot_all = np.concatenate(rot_all)
    grip_delta_all = np.concatenate(grip_delta_all)
    pos_transition = np.concatenate([x for x in pos_transition if x.size > 0]) if any(x.size > 0 for x in pos_transition) else np.array([])
    rot_transition = np.concatenate([x for x in rot_transition if x.size > 0]) if any(x.size > 0 for x in rot_transition) else np.array([])
    pos_non_transition = np.concatenate([x for x in pos_non_transition if x.size > 0]) if any(x.size > 0 for x in pos_non_transition) else np.array([])
    rot_non_transition = np.concatenate([x for x in rot_non_transition if x.size > 0]) if any(x.size > 0 for x in rot_non_transition) else np.array([])

    per_episode_sorted = sorted(per_episode, key=lambda x: x["homing_ratio"], reverse=True)

    return {
        "dataset": "rlbench_insert_onto_square_peg",
        "num_episodes": len(npzs),
        "total_frames": int(total_frames),
        "thresholds": asdict(thresholds),
        "near_zero_ratio": float(total_near_zero / max(1, total_frames)),
        "episodes_all_near_zero": int(episodes_all_near_zero),
        "episodes_with_close_transition": int(episodes_with_close),
        "mean_first_active_index": float(np.mean(first_active_indices)),
        "mean_homing_ratio": float(np.mean(np.array(first_active_indices) / np.maximum(1, np.array(episode_lengths)))),
        "close_transition_frame_ratio": float(total_transition_frames / max(1, total_frames)),
        "transition_near_zero_ratio": float(total_transition_near_zero / max(1, total_transition_frames)),
        "non_transition_near_zero_ratio": float(total_non_transition_near_zero / max(1, total_non_transition_frames)),
        "action_magnitude": {
            "pos_norm": _percentiles(pos_all),
            "rot_norm": _percentiles(rot_all),
            "grip_delta": _percentiles(grip_delta_all),
            "pos_norm_transition": _percentiles(pos_transition),
            "rot_norm_transition": _percentiles(rot_transition),
            "pos_norm_non_transition": _percentiles(pos_non_transition),
            "rot_norm_non_transition": _percentiles(rot_non_transition),
        },
        "worst_homing_episodes": per_episode_sorted[:20],
        "best_homing_episodes": list(reversed(per_episode_sorted[-20:])),
    }


def audit_libero_actions(libero_root: Path, sample_limit: int | None = None) -> dict:
    files = sorted(libero_root.glob("*.hdf5"))
    if sample_limit is not None:
        files = files[:sample_limit]
    if not files:
        return {"available": False, "reason": f"no hdf5 files under {libero_root}"}

    actions_all = []
    for path in files:
        with h5py.File(path, "r") as f:
            demos = f["data"]
            for demo_key in demos.keys():
                actions = demos[demo_key]["actions"][()]
                actions_all.append(np.asarray(actions, dtype=np.float32))

    actions = np.concatenate(actions_all, axis=0)
    pos_norm, rot_norm, _, grip_delta = _step_metrics(actions)
    return {
        "available": True,
        "root": str(libero_root),
        "num_files": len(files),
        "total_frames": int(len(actions)),
        "action_magnitude": {
            "pos_norm": _percentiles(pos_norm),
            "rot_norm": _percentiles(rot_norm),
            "grip_delta": _percentiles(grip_delta),
        },
    }


def compare_distributions(rlbench_report: dict, libero_report: dict) -> dict:
    if not libero_report.get("available"):
        return {"available": False, "reason": libero_report.get("reason", "libero unavailable")}

    out = {}
    for key in ("pos_norm", "rot_norm", "grip_delta"):
        r = rlbench_report["action_magnitude"][key]
        l = libero_report["action_magnitude"][key]
        out[key] = {
            "rlbench_p50": r.get("p50"),
            "libero_p50": l.get("p50"),
            "rlbench_p90": r.get("p90"),
            "libero_p90": l.get("p90"),
            "rlbench_mean": r.get("mean"),
            "libero_mean": l.get("mean"),
            "mean_ratio_rlbench_over_libero": (r.get("mean") / l.get("mean")) if l.get("mean", 0) not in (0, None) else None,
        }
    return {"available": True, "metrics": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlbench_episodes_root", type=Path, default=Path("data/rlbench_data/insert_onto_square_peg/train/episodes"))
    ap.add_argument("--libero_root", type=Path, default=Path("/mnt/ssd/guoning/LIBERO-master/datasets/libero_object"))
    ap.add_argument("--libero_sample_limit", type=int, default=4)
    ap.add_argument("--output_json", type=Path, default=Path("runtime_artifacts/stage_refiner/planner_dataset_distribution_audit_20260426a.json"))
    ap.add_argument("--pos_eps", type=float, default=1e-3)
    ap.add_argument("--rot_eps", type=float, default=2e-2)
    ap.add_argument("--grip_eps", type=float, default=5e-2)
    ap.add_argument("--transition_window", type=int, default=5)
    args = ap.parse_args()

    th = Thresholds(
        pos_eps=args.pos_eps,
        rot_eps=args.rot_eps,
        grip_eps=args.grip_eps,
        transition_window=args.transition_window,
    )
    rlbench = audit_rlbench(args.rlbench_episodes_root, th)
    libero = audit_libero_actions(args.libero_root, sample_limit=args.libero_sample_limit)
    compare = compare_distributions(rlbench, libero)

    report = {
        "rlbench": rlbench,
        "libero_reference": libero,
        "rlbench_vs_libero": compare,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\n[ok] wrote {args.output_json}")


if __name__ == "__main__":
    main()
