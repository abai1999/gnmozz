#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def concat_npz(paths: list[Path]) -> dict[str, np.ndarray]:
    chunks = [{k: np.asarray(v) for k, v in np.load(p, allow_pickle=False).items()} for p in paths]
    keys = sorted(set().union(*(c.keys() for c in chunks)))
    out: dict[str, np.ndarray] = {}

    def chunk_rows(c: dict[str, np.ndarray]) -> int:
        for preferred in ("candidate_actions_local", "front_rgb", "wrist_depth", "proprio", "episode_index"):
            if preferred in c and np.asarray(c[preferred]).ndim > 0:
                return int(np.asarray(c[preferred]).shape[0])
        return int(next(iter(c.values())).shape[0])

    for key in keys:
        exemplar = next((c[key] for c in chunks if key in c), None)
        if exemplar is None:
            continue
        arrs = []
        for c in chunks:
            n = chunk_rows(c)
            if key in c and tuple(c[key].shape[1:]) == tuple(exemplar.shape[1:]):
                arrs.append(c[key])
            else:
                shape = (n,) + tuple(exemplar.shape[1:])
                if exemplar.dtype.kind in ("U", "S", "O"):
                    arrs.append(np.full(shape, "", dtype=exemplar.dtype))
                else:
                    arrs.append(np.zeros(shape, dtype=exemplar.dtype))
        out[key] = np.concatenate(arrs, axis=0)
    return out


def ensure_force_history(data: dict[str, np.ndarray], n: int, history_len: int = 32) -> np.ndarray:
    if "force_history" in data:
        force = np.asarray(data["force_history"], dtype=np.float32)
    elif "ft_hist" in data:
        force = np.asarray(data["ft_hist"], dtype=np.float32)
    elif "gripper_touch_forces" in data:
        raw = np.asarray(data["gripper_touch_forces"], dtype=np.float32)
        if raw.ndim == 2:
            force = np.repeat(raw[:, None, :], history_len, axis=1)
        else:
            force = raw
    else:
        force = np.zeros((n, history_len, 6), dtype=np.float32)
    if force.ndim == 2:
        force = np.repeat(force[:, None, :], history_len, axis=1)
    if force.shape[1] != history_len:
        out = np.zeros((n, history_len, force.shape[-1]), dtype=np.float32)
        take = min(history_len, force.shape[1])
        out[:, -take:, :] = force[:, -take:, :]
        force = out
    if force.shape[-1] < 6:
        out = np.zeros((n, force.shape[1], 6), dtype=np.float32)
        out[:, :, : force.shape[-1]] = force
        force = out
    return force[:, :, :6].astype(np.float32)


def get_index(data: dict[str, np.ndarray], key_options: tuple[str, ...], n: int, default: int) -> np.ndarray:
    for key in key_options:
        if key in data:
            return np.asarray(data[key], dtype=np.int64)
    return np.full((n,), default, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--candidate_score_std_min", type=float, default=0.05)
    ap.add_argument("--switch_margin", type=float, default=0.05)
    ap.add_argument("--force_contact_threshold", type=float, default=0.5)
    ap.add_argument("--force_jam_threshold", type=float, default=3.0)
    ap.add_argument("--depth_near_threshold", type=float, default=0.08)
    args = ap.parse_args()

    data = concat_npz([Path(p) for p in args.input_npz])
    if "candidate_actions_local" not in data or "candidate_oracle_score" not in data:
        raise RuntimeError("input must contain candidate_actions_local and candidate_oracle_score")
    if "front_rgb" not in data or "wrist_depth" not in data or "proprio" not in data:
        raise RuntimeError("input must contain front_rgb, wrist_depth, and proprio")

    actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    n = int(actions.shape[0])
    scores = np.nan_to_num(np.asarray(data["candidate_oracle_score"], dtype=np.float32), nan=-1e9)
    mask = np.asarray(data.get("candidate_mask", np.ones(actions.shape[:2], dtype=np.float32)), dtype=np.float32) > 0.5
    scores_masked = np.where(mask, scores, -1e9)
    best_idx = get_index(data, ("oracle_candidate_index", "best_candidate_index", "candidate_best_index"), n, -1)
    fallback_best = np.argmax(scores_masked, axis=1).astype(np.int64)
    best_idx = np.where((best_idx >= 0) & (best_idx < actions.shape[1]), best_idx, fallback_best)
    baseline_idx = get_index(
        data,
        ("candidate_baseline_index", "runtime_selected_candidate_index", "pred_candidate_index"),
        n,
        0,
    )
    baseline_idx = np.where((baseline_idx >= 0) & (baseline_idx < actions.shape[1]), baseline_idx, 0)

    rows = np.arange(n)
    best_score = scores_masked[rows, best_idx]
    baseline_score = scores_masked[rows, baseline_idx]
    score_std = np.nanstd(np.where(mask, scores, np.nan), axis=1)
    keep = np.isfinite(best_score) & np.isfinite(baseline_score) & (score_std >= float(args.candidate_score_std_min))
    keep &= np.all(np.isfinite(actions[:, :, :6]), axis=(1, 2))
    idx = np.where(keep)[0]
    if idx.size == 0:
        raise RuntimeError("no depth-force contact rows survived filtering")

    force_hist = ensure_force_history(data, n)
    force_last = force_hist[:, -1, :6]
    force_norm = np.linalg.norm(force_last[:, :3], axis=1)
    torque_norm = np.linalg.norm(force_last[:, 3:6], axis=1)
    depth = np.asarray(data["wrist_depth"], dtype=np.float32)
    depth_flat = depth.reshape(n, -1)
    depth_prox = np.nanpercentile(np.where(np.isfinite(depth_flat), depth_flat, np.nan), 5, axis=1).astype(np.float32)
    depth_near = np.isfinite(depth_prox) & (depth_prox < float(args.depth_near_threshold))
    contact_risk = np.zeros((n,), dtype=np.int64)
    contact_risk[depth_near] = 1
    contact_risk[force_norm > float(args.force_contact_threshold)] = 2
    contact_risk[(force_norm > float(args.force_jam_threshold)) | (torque_norm > float(args.force_jam_threshold))] = 3

    switch_target = ((best_score - baseline_score) > float(args.switch_margin)).astype(np.float32)
    progress_target = (best_score - baseline_score).astype(np.float32)
    residual_aux = actions[rows, best_idx, :6].astype(np.float32)

    wrist_rgb = np.asarray(data.get("wrist_rgb", data["front_rgb"]), dtype=np.uint8)
    planner_base = np.asarray(
        data.get("planner_base_action_local_raw", data.get("executed_action_local", np.zeros((n, 6), dtype=np.float32))),
        dtype=np.float32,
    )
    if planner_base.shape[-1] >= 7:
        gripper_state = planner_base[:, 6].astype(np.float32)
    else:
        gripper_state = np.asarray(data.get("gripper_state", np.ones((n,), dtype=np.float32)), dtype=np.float32)
    planner_base = planner_base[:, :6] if planner_base.shape[-1] >= 6 else np.zeros((n, 6), dtype=np.float32)

    out = {
        "front_rgb": np.asarray(data["front_rgb"], dtype=np.uint8)[idx],
        "wrist_rgb": wrist_rgb[idx],
        "wrist_depth": depth[idx].astype(np.float32),
        "force_history": force_hist[idx],
        "proprio": np.asarray(data["proprio"], dtype=np.float32)[idx],
        "planner_base_action_local": planner_base[idx],
        "candidate_actions_local": actions[idx],
        "candidate_mask": mask[idx].astype(np.float32),
        "candidate_value": scores_masked[idx].astype(np.float32),
        "best_candidate_index": best_idx[idx].astype(np.int64),
        "baseline_candidate_index": baseline_idx[idx].astype(np.int64),
        "switch_target": switch_target[idx],
        "contact_risk": contact_risk[idx],
        "progress_target": progress_target[idx].astype(np.float32),
        "residual_aux": residual_aux[idx],
        "depth_proximity": depth_prox[idx].astype(np.float32),
        "gripper_state": gripper_state[idx].astype(np.float32),
        "stage_token": np.asarray(data.get("stage_token", data.get("substage_id", np.zeros((n,), dtype=np.int64))), dtype=np.int64)[idx],
        "episode_index": np.asarray(data.get("episode_index", np.zeros((n,), dtype=np.int64)), dtype=np.int64)[idx],
        "step_index": np.asarray(data.get("step_index", np.arange(n, dtype=np.int64)), dtype=np.int64)[idx],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "depth_force_contact_dataset.npz"
    np.savez_compressed(out_path, **out)

    yaw_abs = np.abs(out["candidate_actions_local"][:, :, 5])
    report = {
        "rows": int(idx.size),
        "input_rows": int(n),
        "episodes": int(np.unique(out["episode_index"]).size),
        "score_std_mean": float(np.nanmean(score_std[idx])),
        "switch_positive_rate": float(np.mean(out["switch_target"])),
        "contact_risk_counts": {str(i): int(np.sum(out["contact_risk"] == i)) for i in range(4)},
        "candidate_yaw_nonzero_ratio": float(np.mean(yaw_abs > 1e-4)),
        "best_is_yaw_ratio": float(np.mean(np.abs(out["candidate_actions_local"][np.arange(idx.size), out["best_candidate_index"], 5]) > 1e-4)),
        "baseline_diff_rate": float(np.mean(out["best_candidate_index"] != out["baseline_candidate_index"])),
        "output_npz": str(out_path),
    }
    (out_dir / "depth_force_contact_dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
