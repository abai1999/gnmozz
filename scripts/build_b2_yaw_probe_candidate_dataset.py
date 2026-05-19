#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cost(xy: np.ndarray, z: np.ndarray, yaw: np.ndarray, w_xy: float, w_yaw: float, w_z_guard: float) -> np.ndarray:
    return w_xy * xy**2 + w_yaw * yaw**2 + w_z_guard * np.maximum(z - 1.0, 0.0) ** 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Append explicit yaw probe candidates for B2 offline diagnostics.")
    ap.add_argument("--input_npz", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", default=None)
    ap.add_argument("--yaw_steps", default="-0.12,0.0,0.12", help="Comma-separated local yaw offsets in radians.")
    ap.add_argument("--w_xy", type=float, default=1.0)
    ap.add_argument("--w_yaw", type=float, default=0.75)
    ap.add_argument("--w_z_guard", type=float, default=0.35)
    ap.add_argument("--ready_bonus", type=float, default=0.50)
    ap.add_argument("--compressed", action="store_true", help="Use compressed npz output; slower for image-heavy candidate banks.")
    args = ap.parse_args()

    src = np.load(args.input_npz, allow_pickle=False)
    data = {k: np.asarray(src[k]) for k in src.files}
    yaw_steps = np.asarray([float(v.strip()) for v in args.yaw_steps.split(",") if v.strip()], dtype=np.float32)
    if yaw_steps.size == 0:
        raise SystemExit("No yaw steps provided")
    if "teacher_current_delta_basin_target" not in data:
        raise SystemExit("Input dataset must contain teacher_current_delta_basin_target")

    actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    n, c, action_dim = actions.shape
    extra = yaw_steps.size
    new_c = c + extra
    yaw_needed = np.asarray(data.get("yaw_needed_v1", np.zeros((n,), dtype=np.float32)), dtype=np.float32) > 0.5
    delta = np.asarray(data["teacher_current_delta_basin_target"], dtype=np.float32)[:, :6]
    cur_xy = np.asarray(data["teacher_xy_norm_v1"], dtype=np.float32)
    cur_z = np.asarray(data["teacher_abs_z_norm_v1"], dtype=np.float32)
    cur_yaw = np.asarray(data["teacher_yaw_norm_v1"], dtype=np.float32)
    current_cost = _cost(cur_xy, cur_z, cur_yaw, args.w_xy, args.w_yaw, args.w_z_guard)

    out = {}
    for k, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim >= 2 and arr.shape[0] == n and arr.shape[1] == c and k in {
            "candidate_actions_local",
            "candidate_group_index",
            "candidate_mask",
            "candidate_oracle_score",
            "candidate_next_basin_distance",
            "candidate_improvement",
            "candidate_basin_positive",
            "teacher_next_xy_norm_v1",
            "teacher_next_abs_z_norm_v1",
            "teacher_next_yaw_norm_v1",
            "teacher_next_ready_v1",
            "b2_best_group_candidate_scope_v1",
            "b2_yaw_aware_candidate_scope_v3",
        }:
            shape = (n, new_c) + arr.shape[2:]
            if arr.dtype.kind in ("U", "S", "O"):
                filled = np.full(shape, "", dtype=arr.dtype)
            else:
                filled = np.zeros(shape, dtype=arr.dtype)
                if k in {"candidate_oracle_score", "candidate_next_basin_distance", "candidate_improvement"}:
                    filled[:, c:] = -1e9
            filled[:, :c] = arr
            out[k] = filled
        else:
            out[k] = arr

    if "candidate_group_index" in out:
        out["candidate_group_index"][:, c:] = np.arange(c, new_c, dtype=np.int64)[None, :]
    out["candidate_mask"][:, c:] = yaw_needed[:, None].astype(np.float32)
    out["candidate_actions_local"][:, c:, :] = 0.0
    out["candidate_actions_local"][:, c:, 5] = yaw_steps[None, :]

    probe_next = delta[:, None, :] - out["candidate_actions_local"][:, c:, :6]
    probe_xy = np.linalg.norm(probe_next[:, :, :2], axis=2).astype(np.float32)
    probe_z = np.abs(probe_next[:, :, 2]).astype(np.float32)
    probe_yaw = np.abs(probe_next[:, :, 5]).astype(np.float32)
    probe_cost = _cost(probe_xy, probe_z, probe_yaw, args.w_xy, args.w_yaw, args.w_z_guard).astype(np.float32)
    probe_improve = (current_cost[:, None] - probe_cost).astype(np.float32)
    probe_ready = ((probe_xy <= 1.0) & (probe_z <= 1.0) & (probe_yaw <= 1.0)).astype(np.float32)
    probe_score = probe_improve + probe_ready * float(args.ready_bonus)
    invalid = np.broadcast_to(~yaw_needed[:, None], probe_score.shape)
    probe_score[invalid] = -1e9
    probe_cost[invalid] = -1e9
    probe_improve[invalid] = -1e9
    probe_ready[invalid] = 0.0

    out["teacher_next_xy_norm_v1"][:, c:] = probe_xy
    out["teacher_next_abs_z_norm_v1"][:, c:] = probe_z
    out["teacher_next_yaw_norm_v1"][:, c:] = probe_yaw
    out["teacher_next_ready_v1"][:, c:] = probe_ready
    if "candidate_next_basin_distance" in out:
        out["candidate_next_basin_distance"][:, c:] = probe_cost
    if "candidate_improvement" in out:
        out["candidate_improvement"][:, c:] = probe_improve
    if "candidate_basin_positive" in out:
        out["candidate_basin_positive"][:, c:] = probe_ready
    out["candidate_oracle_score"][:, c:] = probe_score

    scope = np.asarray(out.get("b2_yaw_aware_candidate_scope_v3", out["candidate_mask"]), dtype=np.float32) > 0.5
    scope[:, c:] = yaw_needed[:, None]
    out["b2_yaw_aware_candidate_scope_v3"] = scope.astype(np.float32)
    scope_scores = np.asarray(out["candidate_oracle_score"], dtype=np.float32).copy()
    scope_scores[~scope] = -1e9
    out["b2_yaw_aware_best_candidate_index_v3"] = np.argmax(scope_scores, axis=1).astype(np.int64)
    out["b2_yaw_aware_scope_size_v3"] = np.sum(scope, axis=1).astype(np.float32)
    out["b2_yaw_aware_scope_yaw_range_v3"] = (
        np.nanmax(np.where(scope, out["teacher_next_yaw_norm_v1"], np.nan), axis=1)
        - np.nanmin(np.where(scope, out["teacher_next_yaw_norm_v1"], np.nan), axis=1)
    ).astype(np.float32)

    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.compressed:
        np.savez_compressed(output, **out)
    else:
        np.savez(output, **out)
    yaw_scope = out["b2_yaw_aware_scope_yaw_range_v3"]
    meta = {
        "input_npz": args.input_npz,
        "output_npz": str(output),
        "yaw_steps": yaw_steps.tolist(),
        "rows": int(n),
        "original_candidates": int(c),
        "new_candidates": int(new_c),
        "yaw_needed_rows": int(np.sum(yaw_needed)),
        "yaw_needed_nonzero_yaw_scope_rows": int(np.sum((yaw_scope > 1e-6) & yaw_needed)),
        "yaw_needed_nonzero_yaw_scope_rate": float(np.mean((yaw_scope > 1e-6)[yaw_needed])) if np.any(yaw_needed) else 0.0,
        "yaw_needed_scope_yaw_range_p95": float(np.percentile(yaw_scope[yaw_needed], 95)) if np.any(yaw_needed) else 0.0,
        "yaw_needed_complete_no_cw_ccw_rows": int(
            np.sum(
                yaw_needed
                & (np.any(scope & (np.abs(out["candidate_actions_local"][:, :, 5]) <= 0.035), axis=1))
                & (np.any(scope & (out["candidate_actions_local"][:, :, 5] < -1e-6), axis=1))
                & (np.any(scope & (out["candidate_actions_local"][:, :, 5] > 1e-6), axis=1))
            )
        ),
    }
    meta_path = Path(args.meta_json) if args.meta_json else output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
