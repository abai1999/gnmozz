#!/usr/bin/env python3
"""Contract audit for close-enabled target-conditioned teacher rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.residual_transforms import local_delta_to_world


def pose_delta_local_between(current_pose_7d: np.ndarray, target_pose_7d: np.ndarray) -> np.ndarray:
    current_pose_7d = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
    target_pose_7d = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
    delta_pos_world = target_pose_7d[:3] - current_pose_7d[:3]
    r_cur = Rotation.from_quat(current_pose_7d[3:7])
    r_tgt = Rotation.from_quat(target_pose_7d[3:7])
    delta_rot = (r_tgt * r_cur.inv()).as_rotvec().astype(np.float32)
    delta_pos_local = r_cur.inv().apply(delta_pos_world.astype(np.float32)).astype(np.float32)
    return np.concatenate([delta_pos_local, delta_rot], axis=0).astype(np.float32)


def apply_executed_local_delta_to_pose(pose_7d: np.ndarray, delta_local_6d: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_7d, dtype=np.float32).copy().reshape(7)
    delta = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    delta_world = local_delta_to_world(delta, pose[3:7]).astype(np.float32)
    pose[:3] = pose[:3] + delta_world[:3]
    r_cur = Rotation.from_quat(pose[3:7])
    r_delta = Rotation.from_rotvec(delta_world[3:6])
    pose[3:7] = (r_delta * r_cur).as_quat().astype(np.float32)
    return pose.astype(np.float32)


def _rate(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    return float(mask.mean()) if mask.size else float("nan")


def _stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def audit(path: Path, *, z_step: float, yaw_step: float) -> dict:
    data = _load_npz(path)
    rows = int(next(iter(data.values())).shape[0]) if data else 0
    current_pose = np.asarray(data["current_pose_7d"], dtype=np.float32).reshape(rows, 7)
    target_pose = np.asarray(data["privileged_motion_target_pose_7d"], dtype=np.float32).reshape(rows, 7)
    current_delta = np.asarray(data["privileged_current_to_target_delta_local"], dtype=np.float32).reshape(rows, 6)
    finite = np.all(np.isfinite(current_pose[:, :7]), axis=1) & np.all(np.isfinite(target_pose[:, :7]), axis=1) & np.all(
        np.isfinite(current_delta[:, :6]), axis=1
    )

    summaries = []
    for axis_name, axis_idx, step in [("z", 2, z_step), ("yaw", 5, yaw_step)]:
        pre_err = np.abs(current_delta[:, axis_idx])
        target_sign = np.sign(current_delta[:, axis_idx])
        nonzero_target = np.isfinite(target_sign) & (target_sign != 0.0)
        valid_target_mask = finite & nonzero_target
        post_pos = []
        post_neg = []
        delta_pos = []
        delta_neg = []
        pos_better = []
        neg_better = []
        for sign, accum_post, accum_delta in [(1.0, post_pos, delta_pos), (-1.0, post_neg, delta_neg)]:
            action = np.zeros((rows, 6), dtype=np.float32)
            action[:, axis_idx] = sign * float(step)
            for i in range(rows):
                if not finite[i]:
                    accum_post.append(np.nan)
                    accum_delta.append(np.nan)
                    continue
                next_pose = apply_executed_local_delta_to_pose(current_pose[i], action[i])
                next_delta = pose_delta_local_between(next_pose, target_pose[i])
                next_delta = np.asarray(next_delta, dtype=np.float32)
                post_err = float(abs(next_delta[axis_idx]))
                accum_post.append(post_err)
                accum_delta.append(post_err - float(pre_err[i]))
        post_pos = np.asarray(post_pos, dtype=np.float32)
        post_neg = np.asarray(post_neg, dtype=np.float32)
        delta_pos = np.asarray(delta_pos, dtype=np.float32)
        delta_neg = np.asarray(delta_neg, dtype=np.float32)
        finite_mask = finite & np.isfinite(post_pos) & np.isfinite(post_neg)
        pos_sign_match = valid_target_mask & (target_sign > 0.0)
        neg_sign_match = valid_target_mask & (target_sign < 0.0)
        summaries.append(
            {
                "axis": axis_name,
                "step": float(step),
                "n": int(finite_mask.sum()),
                "injected_action": f"+{axis_name}",
                "post_error_delta": _stats(delta_pos[finite_mask]),
                "improve_rate": _rate(delta_pos[finite_mask] < 0.0),
                "correct_sign_rate": _rate(pos_sign_match[valid_target_mask]),
                "target_nonzero_rate": _rate(nonzero_target[finite_mask]),
            }
        )
        summaries.append(
            {
                "axis": axis_name,
                "step": float(step),
                "n": int(finite_mask.sum()),
                "injected_action": f"-{axis_name}",
                "post_error_delta": _stats(delta_neg[finite_mask]),
                "improve_rate": _rate(delta_neg[finite_mask] < 0.0),
                "correct_sign_rate": _rate(neg_sign_match[valid_target_mask]),
                "target_nonzero_rate": _rate(nonzero_target[finite_mask]),
            }
        )

    return {
        "path": str(path),
        "rows": rows,
        "finite_rate": _rate(finite),
        "current_xy_stats": _stats(np.linalg.norm(current_delta[:, :2], axis=1)[finite]),
        "current_abs_z_stats": _stats(np.abs(current_delta[:, 2])[finite]),
        "current_yaw_stats": _stats(np.abs(current_delta[:, 5])[finite]),
        "sign_audit": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--z_step", type=float, default=0.002)
    parser.add_argument("--yaw_step", type=float, default=0.006)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.npz, z_step=args.z_step, yaw_step=args.yaw_step)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
