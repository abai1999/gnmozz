#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def finite_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def percentile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def load_trace_invalid_map(trace_dir: Path | None) -> dict[tuple[int, int], bool]:
    if trace_dir is None or not trace_dir.exists():
        return {}
    out: dict[tuple[int, int], bool] = {}
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        stem = path.name.split("_", 1)[0]
        try:
            ep = int(stem.replace("ep", ""))
        except ValueError:
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = int(item.get("step", -1))
                if step >= 0:
                    out[(ep, step)] = bool(item.get("invalid_action", False))
    return out


def labels_near_event(episodes: np.ndarray, steps: np.ndarray, event_map: dict[tuple[int, int], bool], window: int) -> np.ndarray:
    labels = np.zeros((episodes.shape[0],), dtype=bool)
    if not event_map:
        return labels
    events_by_ep: dict[int, list[int]] = {}
    for (ep, step), is_event in event_map.items():
        if is_event:
            events_by_ep.setdefault(int(ep), []).append(int(step))
    for ep in events_by_ep:
        events_by_ep[ep].sort()
    for i, (ep, step) in enumerate(zip(episodes, steps)):
        for event_step in events_by_ep.get(int(ep), []):
            if 0 <= int(event_step) - int(step) <= int(window):
                labels[i] = True
                break
    return labels


def rowwise_delta(values: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(values, dtype=np.float32)
    for ep in np.unique(episodes):
        idx = np.where(episodes == ep)[0]
        if idx.size <= 1:
            continue
        delta[idx[1:]] = values[idx[1:]] - values[idx[:-1]]
    return delta


def pose_motion_deltas(poses: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.shape[-1] < 7:
        zeros = np.zeros((poses.shape[0],), dtype=np.float32)
        return zeros, zeros
    pos = poses[:, :3]
    quat = poses[:, 3:7]
    pos_delta = np.zeros((poses.shape[0],), dtype=np.float32)
    rot_delta_deg = np.zeros((poses.shape[0],), dtype=np.float32)
    for ep in np.unique(episodes):
        idx = np.where(episodes == ep)[0]
        if idx.size <= 1:
            continue
        pos_delta[idx[1:]] = np.linalg.norm(pos[idx[1:]] - pos[idx[:-1]], axis=1)
        q_prev = R.from_quat(quat[idx[:-1]])
        q_next = R.from_quat(quat[idx[1:]])
        rel = q_prev.inv() * q_next
        rot_delta_deg[idx[1:]] = rel.magnitude() * (180.0 / np.pi)
    return pos_delta, rot_delta_deg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--trace_dir", default=None)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--contact_force_threshold", type=float, default=0.05)
    ap.add_argument("--high_force_threshold", type=float, default=0.5)
    ap.add_argument("--force_spike_threshold", type=float, default=0.05)
    ap.add_argument("--near_depth_threshold", type=float, default=0.08)
    ap.add_argument("--invalid_near_window", type=int, default=8)
    args = ap.parse_args()

    data = np.load(args.input_npz, allow_pickle=False)
    episodes = np.asarray(data["episode_index"], dtype=np.int64)
    steps = np.asarray(data.get("step_index", data.get("rollout_step")), dtype=np.int64)
    force_hist = np.asarray(data.get("force_history", data.get("ft_hist")), dtype=np.float32)
    if force_hist.ndim == 2:
        force_hist = force_hist[:, None, :]
    raw_force = np.asarray(data.get("gripper_touch_forces", force_hist[:, -1, :]), dtype=np.float32)
    force_xyz = raw_force[:, :3]
    torque_xyz = raw_force[:, 3:6] if raw_force.shape[1] >= 6 else np.zeros_like(force_xyz)
    force_norm = np.linalg.norm(force_xyz, axis=1).astype(np.float32)
    torque_norm = np.linalg.norm(torque_xyz, axis=1).astype(np.float32)
    force_delta_vec = rowwise_delta(raw_force[:, :6], episodes)
    force_delta_norm = np.linalg.norm(force_delta_vec[:, :3], axis=1).astype(np.float32)
    torque_delta_norm = np.linalg.norm(force_delta_vec[:, 3:6], axis=1).astype(np.float32)
    depth = np.asarray(data.get("depth_proximity", np.full_like(force_norm, np.nan)), dtype=np.float32)
    close = np.asarray(data.get("abs_gripper_cmd", np.ones_like(force_norm)), dtype=np.float32) <= 0.5
    invalid_map = load_trace_invalid_map(Path(args.trace_dir) if args.trace_dir else None)
    invalid_near = labels_near_event(episodes, steps, invalid_map, args.invalid_near_window)
    invalid_now = np.asarray([invalid_map.get((int(ep), int(step)), False) for ep, step in zip(episodes, steps)], dtype=bool)

    contact_event = force_norm > float(args.contact_force_threshold)
    high_force_event = force_norm > float(args.high_force_threshold)
    force_spike = force_delta_norm > float(args.force_spike_threshold)
    near_depth = np.isfinite(depth) & (depth < float(args.near_depth_threshold))

    per_episode = []
    for ep in np.unique(episodes):
        idx = np.where(episodes == ep)[0]
        close_idx = idx[close[idx]]
        first_close = int(steps[close_idx[0]]) if close_idx.size else -1
        event_idx = idx[contact_event[idx] | force_spike[idx] | high_force_event[idx]]
        first_contact = int(steps[event_idx[0]]) if event_idx.size else -1
        invalid_count = int(np.sum(invalid_now[idx]))
        invalid_near_count = int(np.sum(invalid_near[idx]))
        close_ep = close[idx]
        force_ep = force_norm[idx]
        ep_report = {
            "episode": int(ep),
            "rows": int(idx.size),
            "force_norm_mean": float(np.mean(force_norm[idx])),
            "force_norm_p95": percentile(force_norm[idx], 95),
            "force_norm_max": float(np.max(force_norm[idx])),
            "torque_norm_mean": float(np.mean(torque_norm[idx])),
            "torque_norm_p95": percentile(torque_norm[idx], 95),
            "torque_norm_max": float(np.max(torque_norm[idx])),
            "force_delta_norm_mean": float(np.mean(force_delta_norm[idx])),
            "force_delta_norm_p95": percentile(force_delta_norm[idx], 95),
            "force_delta_norm_max": float(np.max(force_delta_norm[idx])),
            "contact_event_count": int(np.sum(contact_event[idx])),
            "high_force_event_count": int(np.sum(high_force_event[idx])),
            "force_spike_count": int(np.sum(force_spike[idx])),
            "near_depth_count": int(np.sum(near_depth[idx])),
            "invalid_action_count": invalid_count,
            "invalid_action_nearby_count": invalid_near_count,
            "first_contact_or_spike_step": first_contact,
            "first_close_step": first_close,
            "force_norm_before_close_mean": float(np.mean(force_ep[~close_ep])) if np.any(~close_ep) else 0.0,
            "force_norm_after_close_mean": float(np.mean(force_ep[close_ep])) if np.any(close_ep) else 0.0,
        }
        per_episode.append(ep_report)

    invalid_rows = invalid_near
    non_invalid_rows = ~invalid_near
    summary = {
        "input_npz": str(args.input_npz),
        "rows": int(force_norm.shape[0]),
        "episodes": [int(x) for x in np.unique(episodes)],
        "force_history_exists": bool("force_history" in data.files or "ft_hist" in data.files),
        "force_history_shape": list(force_hist.shape),
        "force_finite_ratio": finite_ratio(force_hist),
        "force_norm_mean": float(np.mean(force_norm)),
        "force_norm_p95": percentile(force_norm, 95),
        "force_norm_max": float(np.max(force_norm)),
        "torque_norm_mean": float(np.mean(torque_norm)),
        "torque_norm_p95": percentile(torque_norm, 95),
        "torque_norm_max": float(np.max(torque_norm)),
        "force_delta_norm_mean": float(np.mean(force_delta_norm)),
        "force_delta_norm_p95": percentile(force_delta_norm, 95),
        "force_delta_norm_max": float(np.max(force_delta_norm)),
        "contact_event_count": int(np.sum(contact_event)),
        "high_force_event_count": int(np.sum(high_force_event)),
        "force_spike_count": int(np.sum(force_spike)),
        "near_depth_count": int(np.sum(near_depth)),
        "invalid_action_count_from_trace": int(np.sum(invalid_now)),
        "invalid_action_nearby_count": int(np.sum(invalid_near)),
        "force_norm_invalid_near_mean": float(np.mean(force_norm[invalid_rows])) if np.any(invalid_rows) else 0.0,
        "force_norm_non_invalid_near_mean": float(np.mean(force_norm[non_invalid_rows])) if np.any(non_invalid_rows) else 0.0,
        "force_spike_invalid_near_rate": float(np.mean(force_spike[invalid_rows])) if np.any(invalid_rows) else 0.0,
        "force_spike_non_invalid_near_rate": float(np.mean(force_spike[non_invalid_rows])) if np.any(non_invalid_rows) else 0.0,
        "contact_invalid_near_rate": float(np.mean(contact_event[invalid_rows])) if np.any(invalid_rows) else 0.0,
        "contact_non_invalid_near_rate": float(np.mean(contact_event[non_invalid_rows])) if np.any(non_invalid_rows) else 0.0,
        "per_episode": per_episode,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
