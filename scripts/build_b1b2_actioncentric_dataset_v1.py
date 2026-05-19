#!/usr/bin/env python3
"""Build the B1/B2 action-centric distillation dataset v1.

This builder is intentionally *not* a Phase-A ready repair dataset.  It keeps
Phase-A frozen and prepares candidate/group supervision for action-centric
teacher-student learning:

- B1: teacher-cost group correction.
- B2: within-group candidate next-step evaluation.

Rows are selected at window level.  Runtime-like rows are kept as the student
distribution base; oracle/teacher-assisted rows are only retained when they are
useful near-ready windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = (
    "episode_index",
    "candidate_actions_local",
    "candidate_group_index",
    "candidate_mask",
    "candidate_oracle_score",
    "best_candidate_index",
    "best_group_index",
    "proxy_current_delta_basin_target",
)


def _source_from_path(path: str) -> str:
    p = Path(path)
    parent = p.parent.name
    if parent.startswith("b1b2_current_profile_candidate"):
        return p.stem
    return parent if parent else p.stem


def _as_source_array(data: dict[str, np.ndarray], n: int, fallback: str) -> np.ndarray:
    if "source_name" in data:
        return np.asarray(data["source_name"]).astype("U64")
    if "source_domain" in data:
        dom = np.asarray(data["source_domain"])
        return np.asarray([f"source_domain_{int(x)}" for x in dom], dtype="U64")
    return np.full((n,), fallback, dtype="U64")


def _safe_get(data: dict[str, np.ndarray], key: str, n: int, default, dtype=None) -> np.ndarray:
    if key in data:
        arr = np.asarray(data[key])
    else:
        arr = np.full((n,), default, dtype=dtype or np.asarray(default).dtype)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _episode_inverse_frequency_weights(ep: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.ones((ep.shape[0],), dtype=np.float32)
    active_ep = ep[mask]
    if active_ep.size == 0:
        return weights
    uniq, counts = np.unique(active_ep, return_counts=True)
    inv = {int(u): 1.0 / max(int(c), 1) for u, c in zip(uniq, counts)}
    mean_inv = float(np.mean(list(inv.values()))) if inv else 1.0
    for k in inv:
        inv[k] /= max(mean_inv, 1e-6)
    for i, e in enumerate(ep.tolist()):
        if mask[i]:
            weights[i] = float(inv.get(int(e), 1.0))
    return weights


def _source_inverse_frequency_weights(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.ones((source.shape[0],), dtype=np.float32)
    active_source = source[mask]
    if active_source.size == 0:
        return weights
    uniq, counts = np.unique(active_source, return_counts=True)
    inv = {str(u): 1.0 / max(int(c), 1) for u, c in zip(uniq, counts)}
    mean_inv = float(np.mean(list(inv.values()))) if inv else 1.0
    for k in inv:
        inv[k] /= max(mean_inv, 1e-6)
    for i, s in enumerate(source.tolist()):
        if mask[i]:
            weights[i] = float(inv.get(str(s), 1.0))
    return weights


def _concat_union(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys: set[str] = set()
    n_by_chunk = []
    for c in chunks:
        keys.update(c.keys())
        n_by_chunk.append(int(c["episode_index"].shape[0]))
    out: dict[str, np.ndarray] = {}
    for k in sorted(keys):
        template = next((c[k] for c in chunks if k in c), None)
        if template is None:
            continue
        if np.asarray(template).ndim == 0:
            template = np.asarray([np.asarray(template).item()])
        vals = []
        for c, n in zip(chunks, n_by_chunk):
            if k in c:
                arr = np.asarray(c[k])
                if arr.ndim == 0:
                    arr = np.full((n,), arr.item(), dtype=arr.dtype)
                vals.append(arr)
                continue
            shape = (n,) + tuple(template.shape[1:])
            if template.dtype.kind in ("U", "S", "O"):
                vals.append(np.full(shape, "missing", dtype=template.dtype))
            else:
                vals.append(np.zeros(shape, dtype=template.dtype))
        out[k] = np.concatenate(vals, axis=0)
    return out


def _metrics_from_teacher_delta(data: dict[str, np.ndarray], rel_xy: float, rel_z: float, rel_yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "teacher_metrics_norm" in data:
        m = np.asarray(data["teacher_metrics_norm"], dtype=np.float32)
        return m[:, 0], m[:, 1], m[:, 2]
    key = "teacher_current_delta_basin_target" if "teacher_current_delta_basin_target" in data else "current_delta_basin_target"
    delta = np.asarray(data[key], dtype=np.float32)
    xy = np.linalg.norm(delta[:, :2], axis=1) / max(float(rel_xy), 1e-6)
    z = np.abs(delta[:, 2]) / max(float(rel_z), 1e-6)
    yaw = np.abs(delta[:, 5]) / max(float(rel_yaw), 1e-6)
    return xy.astype(np.float32), z.astype(np.float32), yaw.astype(np.float32)


def _candidate_next_norms(
    data: dict[str, np.ndarray],
    rel_xy: float,
    rel_z: float,
    rel_yaw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate next-step teacher geometry from local candidate deltas.

    Candidate labels are still governed by candidate_oracle_score.  These arrays
    are diagnostics for B2 and bucket reporting when exact next component fields
    are not stored by older builders.
    """
    if all(k in data for k in ("candidate_next_xy_norm", "candidate_next_abs_z_norm", "candidate_next_yaw_norm")):
        return (
            np.asarray(data["candidate_next_xy_norm"], dtype=np.float32),
            np.asarray(data["candidate_next_abs_z_norm"], dtype=np.float32),
            np.asarray(data["candidate_next_yaw_norm"], dtype=np.float32),
        )
    delta_key = "teacher_current_delta_basin_target" if "teacher_current_delta_basin_target" in data else "current_delta_basin_target"
    delta = np.asarray(data[delta_key], dtype=np.float32)
    actions = np.asarray(data["candidate_actions_local"], dtype=np.float32)
    next_delta = delta[:, None, :6] - actions[:, :, :6]
    next_xy = np.linalg.norm(next_delta[:, :, :2], axis=2) / max(float(rel_xy), 1e-6)
    next_z = np.abs(next_delta[:, :, 2]) / max(float(rel_z), 1e-6)
    next_yaw = np.abs(next_delta[:, :, 5]) / max(float(rel_yaw), 1e-6)
    return next_xy.astype(np.float32), next_z.astype(np.float32), next_yaw.astype(np.float32)


def _summary(idx: np.ndarray, ep: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    out = {"rows": int(idx.size), "episodes": int(np.unique(ep[idx]).size) if idx.size else 0}
    for name, mask in masks.items():
        m = mask[idx]
        out[f"{name}_rows"] = int(np.sum(m))
        out[f"{name}_eps"] = int(np.unique(ep[idx][m]).size) if np.any(m) else 0
        out[f"{name}_low_confidence"] = bool(int(np.sum(m)) < 25)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", action="append", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--release_xy", type=float, default=0.007)
    ap.add_argument("--release_abs_z", type=float, default=0.0035)
    ap.add_argument("--release_yaw", type=float, default=0.12434)
    ap.add_argument("--very_near_max_norm", type=float, default=1.8)
    ap.add_argument("--xy_block_z_max", type=float, default=1.2)
    ap.add_argument("--xy_block_yaw_max", type=float, default=1.2)
    ap.add_argument("--yaw_needed_xy_max", type=float, default=1.3)
    ap.add_argument("--yaw_needed_z_max", type=float, default=1.2)
    ap.add_argument("--far_negative_min_norm", type=float, default=2.5)
    ap.add_argument("--oracle_source_substrings", default="oracle,oracleub,teacher_assisted")
    ap.add_argument("--oracle_yaw_drop_norm", type=float, default=2.0)
    ap.add_argument("--min_teacher_ready_eps", type=int, default=3)
    ap.add_argument("--min_teacher_ready_rows", type=int, default=20)
    ap.add_argument("--min_yaw_needed_eps", type=int, default=2)
    ap.add_argument("--allow_insufficient", action="store_true")
    ap.add_argument("--yaw_needed_episode_balance_weight", type=float, default=1.6)
    ap.add_argument("--yaw_needed_source_balance_weight", type=float, default=1.3)
    ap.add_argument("--yaw_apply_min_abs_yaw", type=float, default=0.055)
    ap.add_argument("--yaw_apply_no_yaw_margin", type=float, default=0.75)
    ap.add_argument("--yaw_keep_max_abs_yaw", type=float, default=0.035)
    args = ap.parse_args()

    chunks = []
    for path in args.input_npz:
        arr = np.load(path, allow_pickle=False)
        data = {k: np.asarray(arr[k]) for k in arr.files}
        missing = [k for k in REQUIRED if k not in data]
        if missing:
            raise RuntimeError(f"{path} missing required fields: {missing}")
        n = int(data["episode_index"].shape[0])
        data["source_name"] = _as_source_array(data, n, _source_from_path(path))
        chunks.append(data)

    data = _concat_union(chunks)
    n = int(data["episode_index"].shape[0])
    ep = np.asarray(data["episode_index"], dtype=np.int64)
    source = np.asarray(data["source_name"]).astype("U64")
    xy, z, yaw = _metrics_from_teacher_delta(data, args.release_xy, args.release_abs_z, args.release_yaw)
    max_norm = np.maximum(np.maximum(xy, z), yaw)
    teacher_ready = _safe_get(data, "teacher_truth_handoff_ready", n, 0.0, np.float32) > 0.5
    release = (xy <= 1.0) & (z <= 1.0) & (yaw <= 1.0)
    very_near = max_norm <= float(args.very_near_max_norm)
    xy_block = very_near & (xy > 1.0) & (z <= float(args.xy_block_z_max)) & (yaw <= float(args.xy_block_yaw_max))
    yaw_needed = very_near & (yaw > 1.0) & (xy <= float(args.yaw_needed_xy_max)) & (z <= float(args.yaw_needed_z_max))
    far_negative = (~teacher_ready) & (np.maximum(np.maximum(xy, z), yaw) >= float(args.far_negative_min_norm))

    oracle_tokens = [s.strip().lower() for s in args.oracle_source_substrings.split(",") if s.strip()]
    source_lower = np.char.lower(source.astype(str))
    oracle_like = np.zeros((n,), dtype=bool)
    for token in oracle_tokens:
        oracle_like |= np.char.find(source_lower, token) >= 0
    oracle_keep = teacher_ready | release | (very_near & (max_norm <= 1.8))
    oracle_keep &= yaw <= float(args.oracle_yaw_drop_norm)
    keep = (~oracle_like) | oracle_keep
    keep_idx = np.flatnonzero(keep)

    out = {}
    for k, v in data.items():
        arr = np.asarray(v)
        out[k] = arr[keep_idx] if arr.ndim >= 1 and arr.shape[0] == n else arr
    out["teacher_xy_norm_v1"] = xy[keep_idx].astype(np.float32)
    out["teacher_abs_z_norm_v1"] = z[keep_idx].astype(np.float32)
    out["teacher_yaw_norm_v1"] = yaw[keep_idx].astype(np.float32)
    out["teacher_ready_v1"] = teacher_ready[keep_idx].astype(np.float32)
    out["teacher_release_v1"] = release[keep_idx].astype(np.float32)
    out["very_near_v1"] = very_near[keep_idx].astype(np.float32)
    out["xy_block_v1"] = xy_block[keep_idx].astype(np.float32)
    out["yaw_needed_v1"] = yaw_needed[keep_idx].astype(np.float32)
    out["far_negative_v1"] = far_negative[keep_idx].astype(np.float32)
    out["oracle_like_source_v1"] = oracle_like[keep_idx].astype(np.float32)
    next_xy, next_z, next_yaw = _candidate_next_norms(data, args.release_xy, args.release_abs_z, args.release_yaw)
    out["teacher_next_xy_norm_v1"] = next_xy[keep_idx].astype(np.float32)
    out["teacher_next_abs_z_norm_v1"] = next_z[keep_idx].astype(np.float32)
    out["teacher_next_yaw_norm_v1"] = next_yaw[keep_idx].astype(np.float32)
    out["teacher_next_ready_v1"] = (
        (out["teacher_next_xy_norm_v1"] <= 1.0)
        & (out["teacher_next_abs_z_norm_v1"] <= 1.0)
        & (out["teacher_next_yaw_norm_v1"] <= 1.0)
    ).astype(np.float32)
    out["teacher_cost_drop_v1"] = np.asarray(out["candidate_oracle_score"], dtype=np.float32)

    # B2-v3 yaw-aware scope:
    # The original B2 objective ranks candidates only inside teacher-best group.
    # In yaw-needed rows that scope often has zero yaw variation, so B2 cannot
    # learn yaw even when the full candidate bank contains useful yaw actions.
    # Keep the old best-group scope for normal rows, but for yaw-needed windows
    # add candidates that reduce teacher yaw without wildly increasing xy/z.
    valid_candidates = np.asarray(out["candidate_mask"], dtype=np.float32) > 0.5
    candidate_group = np.asarray(out["candidate_group_index"], dtype=np.int64)
    best_group = np.asarray(out["best_group_index"], dtype=np.int64)
    best_group_scope = valid_candidates & (candidate_group == best_group[:, None])
    cur_xy = out["teacher_xy_norm_v1"][:, None]
    cur_z = out["teacher_abs_z_norm_v1"][:, None]
    cur_yaw = out["teacher_yaw_norm_v1"][:, None]
    yaw_improves = out["teacher_next_yaw_norm_v1"] < (cur_yaw - 0.02)
    xy_not_much_worse = out["teacher_next_xy_norm_v1"] <= np.maximum(cur_xy + 0.5, 1.8)
    z_not_much_worse = out["teacher_next_abs_z_norm_v1"] <= np.maximum(cur_z + 0.5, 1.8)
    yaw_scope_extra = (
        (out["yaw_needed_v1"][:, None] > 0.5)
        & valid_candidates
        & yaw_improves
        & xy_not_much_worse
        & z_not_much_worse
    )
    yaw_aware_scope = best_group_scope | yaw_scope_extra
    empty_scope = ~np.any(yaw_aware_scope, axis=1)
    yaw_aware_scope[empty_scope] = best_group_scope[empty_scope]
    out["b2_best_group_candidate_scope_v1"] = best_group_scope.astype(np.float32)
    out["b2_yaw_aware_candidate_scope_v3"] = yaw_aware_scope.astype(np.float32)
    scope_scores = np.asarray(out["candidate_oracle_score"], dtype=np.float32).copy()
    scope_scores[~yaw_aware_scope] = -1e9
    out["b2_yaw_aware_best_candidate_index_v3"] = np.argmax(scope_scores, axis=1).astype(np.int64)
    out["b2_yaw_aware_scope_size_v3"] = np.sum(yaw_aware_scope, axis=1).astype(np.float32)
    out["b2_yaw_aware_scope_yaw_range_v3"] = (
        np.nanmax(np.where(yaw_aware_scope, out["teacher_next_yaw_norm_v1"], np.nan), axis=1)
        - np.nanmin(np.where(yaw_aware_scope, out["teacher_next_yaw_norm_v1"], np.nan), axis=1)
    ).astype(np.float32)

    candidate_actions = np.asarray(out["candidate_actions_local"], dtype=np.float32)
    action_xyz = candidate_actions[:, :, :3]
    action_yaw = candidate_actions[:, :, 5]
    best_idx = out["b2_yaw_aware_best_candidate_index_v3"]
    row_idx = np.arange(best_idx.shape[0])
    best_abs_yaw_action = np.abs(action_yaw[row_idx, best_idx])
    no_yaw_candidates = yaw_aware_scope & (np.abs(action_yaw) <= float(args.yaw_keep_max_abs_yaw))
    no_yaw_scores = np.where(no_yaw_candidates, scope_scores, -1e9)
    no_yaw_best_score = np.max(no_yaw_scores, axis=1)
    no_yaw_valid = np.any(no_yaw_candidates, axis=1)
    best_score = scope_scores[row_idx, best_idx]
    yaw_apply = (
        (out["yaw_needed_v1"] > 0.5)
        & (best_abs_yaw_action >= float(args.yaw_apply_min_abs_yaw))
        & no_yaw_valid
        & ((best_score - no_yaw_best_score) >= float(args.yaw_apply_no_yaw_margin))
    )
    yaw_keep = (
        (out["yaw_needed_v1"] > 0.5)
        & no_yaw_valid
        & (
            (best_abs_yaw_action <= float(args.yaw_keep_max_abs_yaw))
            | ((best_score - no_yaw_best_score) < float(args.yaw_apply_no_yaw_margin))
        )
    )
    out["yaw_apply_v6"] = yaw_apply.astype(np.float32)
    out["yaw_keep_v6"] = yaw_keep.astype(np.float32)
    out["b2_best_abs_yaw_action_v6"] = best_abs_yaw_action.astype(np.float32)
    yaw_mode_margin = np.where(
        no_yaw_valid,
        best_score - no_yaw_best_score,
        0.0,
    ).astype(np.float32)
    out["b2_best_vs_no_yaw_score_margin_v6"] = yaw_mode_margin
    yaw_mode_label = np.full((keep_idx.size,), -1, dtype=np.int64)
    yaw_mode_label[yaw_keep] = 0
    yaw_mode_label[yaw_apply] = 1
    out["yaw_mode_label_v7"] = yaw_mode_label.astype(np.int64)
    out["yaw_mode_margin_v7"] = yaw_mode_margin.astype(np.float32)
    out["yaw_mode_valid_v7"] = ((yaw_apply | yaw_keep) & no_yaw_valid).astype(np.float32)

    # v5 yaw ladder support:
    # build pairwise-ready supervision inside yaw-aware scope for candidates that
    # share the same translational family and differ mainly by yaw magnitude.
    xyz_anchor = np.round(action_xyz / 1e-6).astype(np.int64)
    family_equal = np.all(xyz_anchor[:, :, None, :] == xyz_anchor[:, None, :, :], axis=-1)
    yaw_diff = np.abs(action_yaw[:, :, None] - action_yaw[:, None, :])
    oracle = np.asarray(out["candidate_oracle_score"], dtype=np.float32)
    oracle_diff = oracle[:, :, None] - oracle[:, None, :]
    yaw_ladder_pair_mask = (
        yaw_aware_scope[:, :, None]
        & yaw_aware_scope[:, None, :]
        & family_equal
        & (yaw_diff > 1e-5)
        & (oracle_diff > 0.05)
        & (out["yaw_apply_v6"][:, None, None] > 0.5)
    )
    yaw_keep_pair_mask = (
        yaw_aware_scope[:, :, None]
        & yaw_aware_scope[:, None, :]
        & family_equal
        & (np.abs(action_yaw[:, :, None]) <= float(args.yaw_keep_max_abs_yaw))
        & (np.abs(action_yaw[:, None, :]) > float(args.yaw_keep_max_abs_yaw))
        & (oracle_diff > 0.05)
        & (out["yaw_keep_v6"][:, None, None] > 0.5)
    )
    out["b2_yaw_ladder_pair_mask_v5"] = yaw_ladder_pair_mask.astype(np.float32)
    out["b2_yaw_ladder_pair_target_v5"] = (oracle_diff > 0.0).astype(np.float32)
    out["b2_yaw_ladder_pair_oracle_gap_v5"] = np.where(yaw_ladder_pair_mask, oracle_diff, 0.0).astype(np.float32)
    out["b2_yaw_ladder_family_count_v5"] = np.sum(
        np.any(yaw_ladder_pair_mask, axis=2),
        axis=1,
    ).astype(np.float32)
    out["b2_yaw_keep_pair_mask_v6"] = yaw_keep_pair_mask.astype(np.float32)
    out["b2_yaw_keep_pair_oracle_gap_v6"] = np.where(yaw_keep_pair_mask, oracle_diff, 0.0).astype(np.float32)

    sw = np.asarray(out.get("sample_weight", np.ones((keep_idx.size,), dtype=np.float32)), dtype=np.float32).copy()
    sw *= np.where(out["teacher_ready_v1"] > 0.5, 1.8, 1.0).astype(np.float32)
    sw *= np.where(out["xy_block_v1"] > 0.5, 1.5, 1.0).astype(np.float32)
    sw *= np.where(out["yaw_needed_v1"] > 0.5, 1.6, 1.0).astype(np.float32)
    sw *= np.where(out["yaw_apply_v6"] > 0.5, 1.35, 1.0).astype(np.float32)
    sw *= np.where(out["yaw_keep_v6"] > 0.5, 1.25, 1.0).astype(np.float32)
    sw *= np.where(out["far_negative_v1"] > 0.5, 1.2, 1.0).astype(np.float32)
    yaw_ep_bal = _episode_inverse_frequency_weights(ep[keep_idx], out["yaw_needed_v1"] > 0.5)
    yaw_src_bal = _source_inverse_frequency_weights(source[keep_idx], out["yaw_needed_v1"] > 0.5)
    yaw_balance = (
        1.0
        + (yaw_ep_bal - 1.0) * float(args.yaw_needed_episode_balance_weight)
        + (yaw_src_bal - 1.0) * float(args.yaw_needed_source_balance_weight)
    )
    yaw_balance = np.clip(yaw_balance, 0.25, 4.0).astype(np.float32)
    sw *= np.where(
        out["yaw_needed_v1"] > 0.5,
        yaw_balance,
        1.0,
    ).astype(np.float32)
    out["sample_weight"] = sw.astype(np.float32)

    masks = {
        "teacher_ready": teacher_ready,
        "xy_block": xy_block,
        "yaw_needed": yaw_needed,
        "far_negative": far_negative,
    }
    kept_masks = {k: v[keep_idx] for k, v in masks.items()}
    kept_masks["yaw_apply"] = out["yaw_apply_v6"] > 0.5
    kept_masks["yaw_keep"] = out["yaw_keep_v6"] > 0.5
    kept_masks["yaw_mode_valid"] = out["yaw_mode_valid_v7"] > 0.5
    full_summary = _summary(np.arange(n, dtype=np.int64), ep, masks)
    kept_summary = _summary(np.arange(keep_idx.size, dtype=np.int64), ep[keep_idx], kept_masks)
    insufficient = []
    if kept_summary["teacher_ready_eps"] < int(args.min_teacher_ready_eps):
        insufficient.append(f"teacher_ready_eps {kept_summary['teacher_ready_eps']} < {args.min_teacher_ready_eps}")
    if kept_summary["teacher_ready_rows"] < int(args.min_teacher_ready_rows):
        insufficient.append(f"teacher_ready_rows {kept_summary['teacher_ready_rows']} < {args.min_teacher_ready_rows}")
    if kept_summary["yaw_needed_eps"] < int(args.min_yaw_needed_eps):
        insufficient.append(f"yaw_needed_eps {kept_summary['yaw_needed_eps']} < {args.min_yaw_needed_eps}")

    meta = {
        "input_npz": [str(x) for x in args.input_npz],
        "output_npz": str(args.output_npz),
        "thresholds": vars(args),
        "full_summary": full_summary,
        "kept_summary": kept_summary,
        "selected_teacher_ready_eps": [int(x) for x in np.unique(ep[keep_idx][out["teacher_ready_v1"] > 0.5]).tolist()],
        "selected_yaw_needed_eps": [int(x) for x in np.unique(ep[keep_idx][out["yaw_needed_v1"] > 0.5]).tolist()],
        "selected_yaw_apply_eps": [int(x) for x in np.unique(ep[keep_idx][out["yaw_apply_v6"] > 0.5]).tolist()],
        "selected_yaw_keep_eps": [int(x) for x in np.unique(ep[keep_idx][out["yaw_keep_v6"] > 0.5]).tolist()],
        "source_counts": {str(k): int(v) for k, v in zip(*np.unique(source[keep_idx], return_counts=True))},
        "yaw_apply_source_counts": {
            str(k): int(v)
            for k, v in zip(*np.unique(source[keep_idx][out["yaw_apply_v6"] > 0.5], return_counts=True))
        },
        "yaw_keep_source_counts": {
            str(k): int(v)
            for k, v in zip(*np.unique(source[keep_idx][out["yaw_keep_v6"] > 0.5], return_counts=True))
        },
        "insufficient_reasons": insufficient,
        "passes_hard_gate": not insufficient,
    }

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)
    meta_path = Path(args.meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if insufficient and not args.allow_insufficient:
        raise SystemExit("Dataset hard gate failed: " + "; ".join(insufficient))


if __name__ == "__main__":
    main()
