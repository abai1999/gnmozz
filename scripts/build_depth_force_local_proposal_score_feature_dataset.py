#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTION_SCALE = np.asarray([0.008, 0.008, 0.006, 0.06, 0.06, 0.12], dtype=np.float32)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: np.asarray(raw[k]) for k in raw.files}


def _pick(arrs: dict[str, np.ndarray], *keys: str, default=None) -> np.ndarray:
    for key in keys:
        if key in arrs:
            return np.asarray(arrs[key])
    if default is not None:
        return np.asarray(default)
    raise KeyError(f"none of the keys exist: {keys}")


def _resize_nearest(depth: np.ndarray, size: int = 96) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape == (size, size):
        return arr
    ys = np.linspace(0, arr.shape[0] - 1, size).round().astype(np.int64)
    xs = np.linspace(0, arr.shape[1] - 1, size).round().astype(np.int64)
    return arr[ys][:, xs]


def _candidate_depth_stats(depth: np.ndarray, actions: np.ndarray) -> np.ndarray:
    d = _resize_nearest(depth)
    h, w = d.shape
    center_y = (h - 1) * 0.5
    center_x = (w - 1) * 0.5
    global_mean = float(np.mean(d))
    global_std = float(np.std(d))
    out = []
    for action in np.asarray(actions, dtype=np.float32):
        dx = float(np.clip(action[0] / max(ACTION_SCALE[0], 1e-6), -1.5, 1.5) * 8.0)
        dy = float(np.clip(action[1] / max(ACTION_SCALE[1], 1e-6), -1.5, 1.5) * 8.0)
        cx = int(np.clip(round(center_x + dx), 2, w - 3))
        cy = int(np.clip(round(center_y + dy), 2, h - 3))
        patch = d[cy - 2 : cy + 3, cx - 2 : cx + 3]
        mean = float(np.mean(patch))
        std = float(np.std(patch))
        mn = float(np.min(patch))
        mx = float(np.max(patch))
        center = float(patch[2, 2])
        left = float(np.mean(patch[:, :2]))
        right = float(np.mean(patch[:, 3:]))
        top = float(np.mean(patch[:2, :]))
        bottom = float(np.mean(patch[3:, :]))
        grad_x = right - left
        grad_y = bottom - top
        edge = float(np.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8))
        out.append([mean, std, mn, mx, center, center - global_mean, std - global_std, grad_x, grad_y, edge])
    return np.asarray(out, dtype=np.float32)


def _force_action_interactions(force_history: np.ndarray, actions: np.ndarray) -> np.ndarray:
    force = np.asarray(force_history, dtype=np.float32)
    if force.ndim != 2 or force.shape[-1] < 6:
        force = np.zeros((1, 6), dtype=np.float32)
    last = force[-1, :6]
    prev = force[-2, :6] if force.shape[0] > 1 else np.zeros((6,), dtype=np.float32)
    delta = last - prev
    f_xyz = last[:3]
    torque = last[3:6]
    f_norm = float(np.linalg.norm(f_xyz))
    torque_norm = float(np.linalg.norm(torque))
    spike = float(np.linalg.norm(delta[:3]))
    out = []
    for action in np.asarray(actions, dtype=np.float32):
        cand_xyz = action[:3]
        cand_rot = action[3:6]
        action_norm = float(np.linalg.norm(action))
        out.append(
            [
                f_norm * float(cand_xyz[2]),
                float(np.dot(f_xyz[:2], cand_xyz[:2])),
                float(torque[2] * cand_rot[2]),
                spike * action_norm,
                float(np.dot(torque[:2], cand_rot[:2])),
                float(f_xyz[2] * cand_xyz[2]),
                torque_norm * float(abs(cand_rot[2])),
                f_norm * float(np.linalg.norm(cand_xyz[:2])),
            ]
        )
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--proposal_cache_npz", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--output_meta_json", default="")
    args = ap.parse_args()

    data = _load_npz(Path(args.dataset_npz))
    cache = _load_npz(Path(args.proposal_cache_npz))
    proposals = np.asarray(cache["proposal_actions_local"], dtype=np.float32)
    n, k, _ = proposals.shape
    depth = _pick(data, "wrist_depth")
    force = _pick(data, "force_history_normalized", "force_history", "ft_hist", default=np.zeros((n, 32, 6), dtype=np.float32))

    depth_stats = np.zeros((n, k, 10), dtype=np.float32)
    force_interactions = np.zeros((n, k, 8), dtype=np.float32)
    for i in range(n):
        depth_stats[i] = _candidate_depth_stats(depth[i], proposals[i])
        force_interactions[i] = _force_action_interactions(force[i], proposals[i])

    out = {
        "proposal_actions_local": proposals,
        "candidate_depth_stats": depth_stats,
        "force_action_interactions": force_interactions,
    }
    for key in (
        "row_index",
        "episode_index",
        "step_index",
        "proposal_geometry_gain",
        "proposal_risk_delta",
        "proposal_pareto_mask",
        "proposal_budget_mask",
        "proposal_baseline_index",
        "proposal_geom_top1_index",
        "proposal_best_safe_index",
        "proposal_best_soft_index",
        "proposal_target_delta_local",
    ):
        if key in cache:
            out[key] = cache[key]
        elif key in data:
            out[key] = data[key]

    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    meta = {
        "dataset_npz": str(args.dataset_npz),
        "proposal_cache_npz": str(args.proposal_cache_npz),
        "output_npz": str(output),
        "rows": int(n),
        "proposal_count": int(k),
        "candidate_depth_stats_dim": 10,
        "force_action_interactions_dim": 8,
    }
    meta_path = Path(args.output_meta_json) if args.output_meta_json else output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
