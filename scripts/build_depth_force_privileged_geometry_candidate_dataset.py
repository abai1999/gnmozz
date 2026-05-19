#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from build_depth_force_candidate_cost_dataset import (
    _candidate_kind,
    _load_npz,
    _risk_cost,
    _risk_join,
    _row_weights,
    _safe_best_index,
)
from build_pose_candidate_dataset import apply_local_offset_to_pose, candidate_offsets, pose_delta_local_between


def _angle_abs_sym(yaw: np.ndarray, period: float) -> np.ndarray:
    y = np.asarray(yaw, dtype=np.float32)
    if not np.isfinite(float(period)) or float(period) <= 0.0:
        return np.abs((y + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    p = float(period)
    return np.abs((y + 0.5 * p) % p - 0.5 * p).astype(np.float32)


def _select_target_pose(data: dict[str, np.ndarray], key: str) -> tuple[np.ndarray | None, str]:
    candidates = [
        "privileged_motion_target_pose_7d",
        "privileged_basin_center_pose_7d",
        "privileged_pregrasp_target_pose_7d",
        "privileged_grasp_commit_target_pose_7d",
        "motion_target_pose_7d",
        "basin_center_pose_7d",
        "pregrasp_target_pose_7d",
        "grasp_commit_target_pose_7d",
        "reference_anchor_pose_7d",
    ]
    if key and key != "auto":
        candidates = [key]
    for k in candidates:
        if k in data:
            arr = np.asarray(data[k], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 7:
                return arr, k
    return None, "missing"


def _select_delta(data: dict[str, np.ndarray], key: str) -> tuple[np.ndarray | None, str]:
    candidates = [
        "privileged_current_delta_basin_target",
        "teacher_current_delta_basin_target",
        "current_delta_basin_target",
        "proxy_current_delta_basin_target",
        "target_delta_teacher",
        "motion_target_delta_local",
    ]
    if key and key != "auto":
        candidates = [key]
    for k in candidates:
        if k in data:
            arr = np.asarray(data[k], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                return arr[:, :6], k
    return None, "missing"


def _select_candidate_bank(
    data: dict[str, np.ndarray],
    key: str,
) -> tuple[np.ndarray | None, str]:
    candidates = [
        "candidate_actions_local",
        "candidate_bank",
        "candidate_offsets",
    ]
    if key and key != "auto":
        candidates = [key]
    for k in candidates:
        if k not in data:
            continue
        arr = np.asarray(data[k], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 6:
            return arr, k
        if arr.ndim == 3 and arr.shape[-1] == 6 and arr.shape[0] == 1:
            return arr[0], k
    return None, "missing"


def _geometry_cost_from_delta(
    delta: np.ndarray,
    *,
    r_xy: float,
    r_z: float,
    r_yaw: float,
    r_tilt: float,
    yaw_symmetry_period: float,
    w_xy: float,
    w_z: float,
    w_yaw: float,
    w_tilt: float,
) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float32)
    xy = np.linalg.norm(d[..., :2], axis=-1) / max(float(r_xy), 1e-6)
    z = np.abs(d[..., 2]) / max(float(r_z), 1e-6)
    yaw = _angle_abs_sym(d[..., 5], float(yaw_symmetry_period)) / max(float(r_yaw), 1e-6)
    tilt = np.linalg.norm(d[..., 3:5], axis=-1) / max(float(r_tilt), 1e-6)
    return (
        float(w_xy) * (xy ** 2)
        + float(w_z) * (z ** 2)
        + float(w_yaw) * (yaw ** 2)
        + float(w_tilt) * (tilt ** 2)
    ).astype(np.float32)


def _rollout_geometry_cost(
    current_pose: np.ndarray,
    target_pose: np.ndarray | None,
    current_delta: np.ndarray | None,
    candidates: np.ndarray,
    valid_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(valid_idx.shape[0])
    c = int(candidates.shape[0])
    cost = np.full((n, c), np.inf, dtype=np.float32)
    post_delta = np.zeros((n, c, 6), dtype=np.float32)
    source_code = np.zeros((n,), dtype=np.int64)
    valid = np.zeros((n,), dtype=bool)

    for out_i, src_i in enumerate(valid_idx.tolist()):
        if target_pose is not None and current_pose is not None:
            cur = np.asarray(current_pose[src_i], dtype=np.float32)
            tgt = np.asarray(target_pose[src_i], dtype=np.float32)
            if np.all(np.isfinite(cur)) and np.all(np.isfinite(tgt)):
                for j, cand in enumerate(candidates):
                    nxt = apply_local_offset_to_pose(cur, cand)
                    d = pose_delta_local_between(nxt, tgt)
                    d[5] = float(_angle_abs_sym(np.asarray([d[5]], dtype=np.float32), args.yaw_symmetry_period)[0])
                    post_delta[out_i, j] = d.astype(np.float32)
                source_code[out_i] = 1
                valid[out_i] = True
                continue
        if current_delta is not None:
            d0 = np.asarray(current_delta[src_i], dtype=np.float32).reshape(6)
            if np.all(np.isfinite(d0)):
                approx = d0[None, :] - np.asarray(candidates, dtype=np.float32)
                approx[:, 5] = _angle_abs_sym(approx[:, 5], args.yaw_symmetry_period)
                post_delta[out_i] = approx.astype(np.float32)
                source_code[out_i] = 2
                valid[out_i] = True
                continue
    finite = valid[:, None]
    if np.any(valid):
        cost[finite.repeat(c, axis=1)] = _geometry_cost_from_delta(
            post_delta[finite.repeat(c, axis=1)].reshape(-1, 6),
            r_xy=args.r_xy,
            r_z=args.r_z,
            r_yaw=args.r_yaw,
            r_tilt=args.r_tilt,
            yaw_symmetry_period=args.yaw_symmetry_period,
            w_xy=args.lambda_xy,
            w_z=args.lambda_z,
            w_yaw=args.lambda_yaw,
            w_tilt=args.lambda_tilt,
        )
    return cost, post_delta, source_code, valid


def _target_modes(
    labels: dict[str, np.ndarray],
    *,
    best_geometry_index: np.ndarray,
    baseline_index: np.ndarray,
    geometry_improvement: np.ndarray,
    align_refine_margin: float,
) -> np.ndarray:
    n = int(best_geometry_index.shape[0])
    mode = np.full((n,), "planner", dtype="U32")
    kin = (labels["kinematic_invalid_label"] > 0.5) | (labels["action_range_invalid_label"] > 0.5)
    contact = (labels["contact_label"] > 0.5) | (labels["force_spike_label"] > 0.5) | (labels["jam_label"] > 0.5)
    near = labels["near_depth_label"] > 0.5
    mode[near] = "near_hold"
    mode[contact] = "contact_backoff"
    mode[kin] = "kinematic_hold"
    no_hard_risk = (~kin) & (~contact)
    refine = (
        no_hard_risk
        & (geometry_improvement > float(align_refine_margin))
        & (best_geometry_index != baseline_index)
    )
    mode[refine] = "align_refine"
    return mode


def _pose_yaw_perturb(current_pose_7d: np.ndarray, yaw_delta_rad: float) -> np.ndarray:
    pose = np.asarray(current_pose_7d, dtype=np.float32).copy()
    if pose.shape != (7,):
        raise ValueError(f"expected pose shape (7,), got {pose.shape}")
    r_cur = Rotation.from_quat(pose[3:7])
    r_delta = Rotation.from_euler("z", float(yaw_delta_rad))
    pose[3:7] = (r_cur * r_delta).as_quat().astype(np.float32)
    return pose


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _robust_scale(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    scale = float(np.percentile(np.abs(arr), 90))
    return max(scale, 1e-6)


def _second_best_gap(cost: np.ndarray, best_index: np.ndarray) -> np.ndarray:
    arr = np.asarray(cost, dtype=np.float32).copy()
    row = np.arange(arr.shape[0], dtype=np.int64)
    best = arr[row, best_index]
    arr[row, best_index] = np.inf
    second = np.min(arr, axis=1)
    return (second - best).astype(np.float32)


def _hist(x: np.ndarray) -> dict[str, int]:
    uniq, cnt = np.unique(x, return_counts=True)
    return {str(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--risk_npz", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_pose_key", default="auto")
    ap.add_argument("--delta_key", default="auto")
    ap.add_argument("--candidate_bank_npz", default="")
    ap.add_argument("--candidate_bank_key", default="auto")
    ap.add_argument("--drop_missing_privileged", action="store_true", default=True)
    ap.add_argument("--keep_missing_privileged", dest="drop_missing_privileged", action="store_false")
    ap.add_argument("--candidate_mode", type=str, default="primitives", choices=("grid", "primitives"))
    ap.add_argument("--candidate_xy_values", type=str, default="0.004,0.008")
    ap.add_argument("--candidate_z_values", type=str, default="0.0")
    ap.add_argument("--candidate_yaw_values", type=str, default="0.03,0.05,0.12")
    ap.add_argument("--candidate_include_diagonals", action="store_true", default=True)
    ap.add_argument("--no_candidate_include_diagonals", dest="candidate_include_diagonals", action="store_false")
    ap.add_argument("--primitive_xy_small", type=float, default=0.004)
    ap.add_argument("--primitive_xy_large", type=float, default=0.008)
    ap.add_argument("--primitive_xy_micro_values", type=str, default="0.001,0.0015,0.002,0.003")
    ap.add_argument("--primitive_z_small", type=float, default=0.004)
    ap.add_argument("--primitive_yaw_small", type=float, default=0.03)
    ap.add_argument("--primitive_yaw_probe_values", type=str, default="0.0174533,0.0349066,0.0698132")
    ap.add_argument("--primitive_pitch_small", type=float, default=0.06)
    ap.add_argument("--primitive_roll_small", type=float, default=0.06)
    ap.add_argument("--primitive_include_descend", action="store_true", default=True)
    ap.add_argument("--no_primitive_include_descend", dest="primitive_include_descend", action="store_false")
    ap.add_argument("--primitive_include_combos", action="store_true", default=True)
    ap.add_argument("--no_primitive_include_combos", dest="primitive_include_combos", action="store_false")
    ap.add_argument("--primitive_include_tilt", action="store_true", default=False)
    ap.add_argument("--no_primitive_include_tilt", dest="primitive_include_tilt", action="store_false")
    ap.add_argument("--r_xy", type=float, default=0.008)
    ap.add_argument("--r_z", type=float, default=0.005)
    ap.add_argument("--r_yaw", type=float, default=np.deg2rad(5.0))
    ap.add_argument("--r_tilt", type=float, default=0.12)
    ap.add_argument("--yaw_symmetry_period", type=float, default=np.pi / 2.0)
    ap.add_argument("--lambda_xy", type=float, default=1.0)
    ap.add_argument("--lambda_z", type=float, default=1.0)
    ap.add_argument("--lambda_yaw", type=float, default=1.0)
    ap.add_argument("--lambda_tilt", type=float, default=0.25)
    ap.add_argument("--lambda_geo", type=float, default=1.0)
    ap.add_argument("--lambda_risk", type=float, default=0.25)
    ap.add_argument("--lambda_smooth", type=float, default=0.05)
    ap.add_argument("--align_refine_margin", type=float, default=0.02)
    ap.add_argument("--yaw_opportunity_margin", type=float, default=0.02)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    ap.add_argument("--yaw_augment_degrees", type=str, default="")
    ap.add_argument("--yaw_augment_min_abs_yaw_error", type=float, default=np.deg2rad(8.0))
    ap.add_argument("--yaw_augment_max_abs_xy_error", type=float, default=0.05)
    ap.add_argument("--yaw_augment_max_abs_z_error", type=float, default=0.08)
    ap.add_argument("--yaw_augment_only_pose_rollout", action="store_true", default=True)
    ap.add_argument("--no_yaw_augment_only_pose_rollout", dest="yaw_augment_only_pose_rollout", action="store_false")
    args = ap.parse_args()

    support_all = _load_npz(Path(args.support_npz))
    risk_all = _load_npz(Path(args.risk_npz)) if args.risk_npz else None
    labels_all = _risk_join(support_all, risk_all)
    n_all = int(next(iter(support_all.values())).shape[0])
    candidate_override = None
    candidate_override_key = "built_in"
    if args.candidate_bank_npz:
        candidate_override = _load_npz(Path(args.candidate_bank_npz))
        candidates, candidate_override_key = _select_candidate_bank(candidate_override, args.candidate_bank_key)
        if candidates is None:
            raise SystemExit(
                f"candidate bank override {args.candidate_bank_npz} does not contain a valid (C, 6) array; "
                "expected candidate_actions_local or candidate_bank"
            )
    else:
        candidates = candidate_offsets(args).astype(np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != 6:
        raise ValueError(f"candidate bank must have shape (C, 6), got {candidates.shape}")

    target_pose, target_key = _select_target_pose(support_all, args.target_pose_key)
    current_delta, delta_key = _select_delta(support_all, args.delta_key)
    if "privileged_current_pose_7d" in support_all:
        current_pose = np.asarray(support_all["privileged_current_pose_7d"], dtype=np.float32)
    elif "current_pose_7d" in support_all:
        current_pose = np.asarray(support_all["current_pose_7d"], dtype=np.float32)
    else:
        current_pose = None
    if target_pose is None and current_delta is None:
        raise SystemExit(
            "No privileged target pose or delta fields found. Re-run clean support with label-only target fields "
            "or pass a support npz containing motion_target_pose_7d/basin_center_pose_7d/current_delta_basin_target."
        )

    valid_idx = np.arange(n_all, dtype=np.int64)
    geom_cost, post_delta, source_code, valid = _rollout_geometry_cost(
        current_pose,
        target_pose,
        current_delta,
        candidates,
        valid_idx,
        args,
    )
    if args.drop_missing_privileged:
        base_mask = valid
    else:
        base_mask = np.ones_like(valid, dtype=bool)
    base_idx = valid_idx[base_mask]
    base_geom = geom_cost[base_mask]
    base_post_delta = post_delta[base_mask]
    base_source_code = source_code[base_mask]
    if base_idx.size == 0:
        raise SystemExit("No rows with usable privileged geometry labels.")

    support = {}
    for key, arr in support_all.items():
        a = np.asarray(arr)
        if a.shape[0] == n_all:
            support[key] = a[base_idx]
    labels = {k: np.asarray(v)[base_idx] for k, v in labels_all.items()}
    augment_source_index = np.full((base_idx.shape[0],), -1, dtype=np.int64)
    augment_yaw_deg = np.zeros((base_idx.shape[0],), dtype=np.float32)
    augment_flag = np.zeros((base_idx.shape[0],), dtype=np.float32)
    current_pose_base = None if current_pose is None else np.asarray(current_pose, dtype=np.float32)[base_idx]
    current_delta_base = None if current_delta is None else np.asarray(current_delta, dtype=np.float32)[base_idx]
    current_pose_use = current_pose_base
    geom_cost_use = base_geom
    post_delta_use = base_post_delta
    source_code_use = base_source_code

    yaw_aug_raw = [x.strip() for x in str(args.yaw_augment_degrees).split(",") if x.strip()]
    yaw_aug_degs = [float(x) for x in yaw_aug_raw]
    if current_pose_base is not None and target_pose is not None and yaw_aug_degs:
        target_delta = np.array([pose_delta_local_between(current_pose[i], target_pose[i]) for i in base_idx], dtype=np.float32)
        yaw_abs = _angle_abs_sym(target_delta[:, 5], float(args.yaw_symmetry_period))
        xy_norm = np.linalg.norm(target_delta[:, :2], axis=-1).astype(np.float32)
        z_abs = np.abs(target_delta[:, 2]).astype(np.float32)
        augmentable = (
            (yaw_abs >= float(args.yaw_augment_min_abs_yaw_error))
            & (xy_norm <= float(args.yaw_augment_max_abs_xy_error))
            & (z_abs <= float(args.yaw_augment_max_abs_z_error))
        )
        if args.yaw_augment_only_pose_rollout:
            augmentable &= (base_source_code == 1)
        aug_source_rows = []
        aug_degs = []
        for src_i in np.where(augmentable)[0].tolist():
            for deg in yaw_aug_degs:
                if abs(float(deg)) <= 1e-8:
                    continue
                aug_source_rows.append(src_i)
                aug_degs.append(float(deg))
        if aug_source_rows:
            aug_source_rows = np.asarray(aug_source_rows, dtype=np.int64)
            aug_degs = np.asarray(aug_degs, dtype=np.float32)
            aug_current_pose = current_pose_base[aug_source_rows].copy()
            for i, deg in enumerate(aug_degs.tolist()):
                aug_current_pose[i] = _pose_yaw_perturb(aug_current_pose[i], np.deg2rad(float(deg)))
            aug_target_pose = target_pose[base_idx[aug_source_rows]]
            aug_current_delta = None if current_delta_base is None else current_delta_base[aug_source_rows]
            aug_geom_cost, aug_post_delta, aug_source_code, aug_valid = _rollout_geometry_cost(
                aug_current_pose,
                aug_target_pose,
                aug_current_delta,
                candidates,
                np.arange(aug_current_pose.shape[0], dtype=np.int64),
                args,
            )
            if np.any(aug_valid):
                aug_valid_rows = np.where(aug_valid)[0]
                base_from_aug = aug_source_rows[aug_valid_rows]
                geom_cost_use = np.concatenate([geom_cost_use, aug_geom_cost[aug_valid_rows]], axis=0)
                post_delta_use = np.concatenate([post_delta_use, aug_post_delta[aug_valid_rows]], axis=0)
                source_code_use = np.concatenate([source_code_use, aug_source_code[aug_valid_rows]], axis=0)
                current_pose_use = np.concatenate([current_pose_use, aug_current_pose[aug_valid_rows]], axis=0)
                if current_delta_base is not None:
                    current_delta_base = np.concatenate([current_delta_base, aug_current_delta[aug_valid_rows]], axis=0)
                for key in list(support.keys()):
                    support[key] = np.concatenate([support[key], np.asarray(support_all[key])[base_idx[base_from_aug]]], axis=0)
                for key in list(labels.keys()):
                    labels[key] = np.concatenate([labels[key], np.asarray(labels_all[key])[base_idx[base_from_aug]]], axis=0)
                augment_source_index = np.concatenate([augment_source_index, base_idx[base_from_aug]], axis=0)
                augment_yaw_deg = np.concatenate([augment_yaw_deg, aug_degs[aug_valid_rows]], axis=0)
                augment_flag = np.concatenate([augment_flag, np.ones((aug_valid_rows.shape[0],), dtype=np.float32)], axis=0)
    n = int(geom_cost_use.shape[0])
    geom_cost = geom_cost_use
    post_delta = post_delta_use
    source_code = source_code_use
    valid_idx = base_idx
    if n == 0:
        raise SystemExit("No rows with usable privileged geometry labels.")
    c = int(candidates.shape[0])
    candidate_bank = np.broadcast_to(candidates[None, :, :], (n, c, 6)).copy().astype(np.float32)
    candidate_mask = np.isfinite(geom_cost).astype(np.float32)

    planner = np.asarray(
        support.get("planner_base_action_local_raw", support.get("planner_base_action_local", np.zeros((n, 6), dtype=np.float32))),
        dtype=np.float32,
    )
    if planner.ndim == 1:
        planner = np.repeat(planner[None, :], n, axis=0)
    gripper = np.asarray(support.get("gripper_state", np.zeros((n,), dtype=np.float32)), dtype=np.float32)

    risk_cost = _risk_cost(
        candidate_bank.reshape(-1, 6),
        contact=np.repeat(labels["contact_label"][:, None], c, axis=1).reshape(-1),
        force_spike=np.repeat(labels["force_spike_label"][:, None], c, axis=1).reshape(-1),
        jam=np.repeat(labels["jam_label"][:, None], c, axis=1).reshape(-1),
        motion_stall=np.repeat(labels["motion_stall_label"][:, None], c, axis=1).reshape(-1),
        near_depth=np.repeat(labels["near_depth_label"][:, None], c, axis=1).reshape(-1),
        kin_invalid=np.repeat(labels["kinematic_invalid_label"][:, None], c, axis=1).reshape(-1),
        action_range_invalid=np.repeat(labels["action_range_invalid_label"][:, None], c, axis=1).reshape(-1),
        gripper_state=np.repeat(gripper[:, None], c, axis=1).reshape(-1),
        w_contact_xy=1.0,
        w_contact_z=1.4,
        w_contact_yaw=1.1,
        w_contact_hold=0.8,
        w_contact_backoff=0.6,
        w_spike_xy=0.8,
        w_spike_z=1.0,
        w_spike_yaw=1.2,
        w_jam_xy=1.2,
        w_jam_z=1.6,
        w_jam_yaw=1.4,
        w_jam_hold=1.0,
        w_jam_backoff=0.9,
        w_stall_hold=1.2,
        w_stall_small=0.6,
        w_near_xy=0.6,
        w_near_z=0.8,
        w_near_yaw=0.6,
        w_kin_mag=1.4,
        w_kin_yaw=0.8,
        w_range_mag=1.6,
    ).reshape(n, c)
    smooth_cost = np.linalg.norm(candidate_bank, axis=-1).astype(np.float32)

    geometry_scale = _robust_scale(geom_cost[candidate_mask > 0.5])
    risk_scale = _robust_scale(risk_cost[candidate_mask > 0.5])
    smooth_scale = _robust_scale(smooth_cost[candidate_mask > 0.5])
    geom_cost_norm = (geom_cost / geometry_scale).astype(np.float32)
    risk_cost_norm = (risk_cost / risk_scale).astype(np.float32)
    smooth_cost_norm = (smooth_cost / smooth_scale).astype(np.float32)
    total_cost_norm = (
        float(args.lambda_geo) * geom_cost_norm
        + float(args.lambda_risk) * risk_cost_norm
        + float(args.lambda_smooth) * smooth_cost_norm
    ).astype(np.float32)

    total_cost = (
        float(args.lambda_geo) * geom_cost
        + float(args.lambda_risk) * risk_cost
        + float(args.lambda_smooth) * smooth_cost
    ).astype(np.float32)

    baseline_index = np.argmin(np.linalg.norm(candidates, axis=1)).astype(np.int64)
    baseline_index = np.full((n,), int(baseline_index), dtype=np.int64)
    best_geometry_index = _safe_best_index(geom_cost, candidate_mask)
    best_risk_aware_index = _safe_best_index(total_cost, candidate_mask)
    best_risk_aware_index_norm = _safe_best_index(total_cost_norm, candidate_mask)
    rows = np.arange(n, dtype=np.int64)
    baseline_geometry = geom_cost[rows, baseline_index]
    best_geometry = geom_cost[rows, best_geometry_index]
    baseline_total = total_cost[rows, baseline_index]
    best_total = total_cost[rows, best_risk_aware_index]
    baseline_total_norm = total_cost_norm[rows, baseline_index]
    best_total_norm = total_cost_norm[rows, best_risk_aware_index_norm]
    geometry_improvement = (baseline_geometry - best_geometry).astype(np.float32)
    total_improvement = (baseline_total - best_total).astype(np.float32)
    total_improvement_norm = (baseline_total_norm - best_total_norm).astype(np.float32)
    target_mode = _target_modes(
        labels,
        best_geometry_index=best_geometry_index,
        baseline_index=baseline_index,
        geometry_improvement=geometry_improvement,
        align_refine_margin=args.align_refine_margin,
    )

    yaw_abs = np.abs(candidate_bank[:, :, 5])
    best_geom_yaw = yaw_abs[rows, best_geometry_index] > float(args.keep_yaw_abs)
    yaw_candidates = yaw_abs > float(args.keep_yaw_abs)
    yaw_best_cost = np.min(np.where(yaw_candidates, geom_cost, np.inf), axis=1)
    no_yaw_best_cost = np.min(np.where(~yaw_candidates, geom_cost, np.inf), axis=1)
    yaw_opportunity = (no_yaw_best_cost - yaw_best_cost) > float(args.yaw_opportunity_margin)

    out: dict[str, np.ndarray] = {}
    runtime_safe_keys = [
        "front_rgb",
        "wrist_rgb",
        "wrist_depth",
        "proprio",
        "gripper_context",
        "force_history",
        "force_history_raw",
        "force_history_normalized",
        "ft_hist",
        "gripper_touch_forces",
        "force_norm",
        "torque_norm",
        "force_delta_norm",
        "torque_delta_norm",
        "depth_proximity",
        "gripper_state",
        "planner_base_action_local_raw",
        "planner_base_action_7d_raw",
        "planner_base_action_local",
        "executed_action_local",
        "executed_action_7d",
        "stage_token",
        "phase_id",
        "substage_id",
        "contact_state",
        "planner_close_intent",
        "stage_target_mode",
        "episode_index",
        "step_index",
    ]
    for key in runtime_safe_keys:
        if key in support:
            out[key] = np.asarray(support[key])
    if "force_history" not in out and "ft_hist" not in out:
        out["force_history"] = np.zeros((n, 32, 6), dtype=np.float32)
    if "planner_base_action_local_raw" not in out:
        out["planner_base_action_local_raw"] = planner.astype(np.float32)

    out["candidate_actions_local"] = candidate_bank
    out["candidate_mask"] = candidate_mask.astype(np.float32)
    out["candidate_privileged_geometry_cost"] = geom_cost.astype(np.float32)
    out["candidate_geometry_cost"] = geom_cost.astype(np.float32)
    out["candidate_risk_cost"] = risk_cost.astype(np.float32)
    out["candidate_smooth_cost"] = smooth_cost.astype(np.float32)
    out["candidate_total_cost"] = total_cost.astype(np.float32)
    out["candidate_privileged_geometry_cost_norm"] = geom_cost_norm.astype(np.float32)
    out["candidate_geometry_cost_norm"] = geom_cost_norm.astype(np.float32)
    out["candidate_risk_cost_norm"] = risk_cost_norm.astype(np.float32)
    out["candidate_smooth_cost_norm"] = smooth_cost_norm.astype(np.float32)
    out["candidate_total_cost_norm"] = total_cost_norm.astype(np.float32)
    out["candidate_geometry_score"] = (-geom_cost).astype(np.float32)
    out["candidate_risk_score"] = (-risk_cost).astype(np.float32)
    out["candidate_oracle_score"] = (-total_cost).astype(np.float32)
    out["best_geometry_candidate_index"] = best_geometry_index.astype(np.int64)
    out["candidate_best_geometry_index"] = best_geometry_index.astype(np.int64)
    out["best_risk_aware_candidate_index"] = best_risk_aware_index.astype(np.int64)
    out["candidate_best_index"] = best_risk_aware_index.astype(np.int64)
    out["best_candidate_index"] = best_risk_aware_index.astype(np.int64)
    out["best_risk_aware_candidate_index_norm"] = best_risk_aware_index_norm.astype(np.int64)
    out["candidate_best_index_norm"] = best_risk_aware_index_norm.astype(np.int64)
    out["best_candidate_index_norm"] = best_risk_aware_index_norm.astype(np.int64)
    out["candidate_best_geometry_index_norm"] = best_geometry_index.astype(np.int64)
    out["best_geometry_candidate_index_norm"] = best_geometry_index.astype(np.int64)
    out["baseline_candidate_index"] = baseline_index.astype(np.int64)
    out["candidate_baseline_index"] = baseline_index.astype(np.int64)
    out["candidate_kind"] = np.broadcast_to(np.asarray([_candidate_kind(a) for a in candidates], dtype="U16")[None, :], (n, c)).copy()
    out["candidate_post_delta_privileged"] = post_delta.astype(np.float32)
    out["candidate_geometry_improvement"] = geometry_improvement
    out["candidate_total_improvement"] = total_improvement
    out["candidate_best_geometry_cost"] = best_geometry.astype(np.float32)
    out["candidate_best_total_cost"] = best_total.astype(np.float32)
    out["candidate_best_total_cost_norm"] = best_total_norm.astype(np.float32)
    out["candidate_baseline_geometry_cost"] = baseline_geometry.astype(np.float32)
    out["candidate_baseline_total_cost"] = baseline_total.astype(np.float32)
    out["candidate_baseline_total_cost_norm"] = baseline_total_norm.astype(np.float32)
    out["candidate_target_mode"] = target_mode
    out["target_mode"] = target_mode
    out["yaw_opportunity_label"] = yaw_opportunity.astype(np.float32)
    out["best_geometry_is_yaw"] = best_geom_yaw.astype(np.float32)
    out["yaw_augmentation_applied"] = augment_flag.astype(np.float32)
    out["yaw_augmentation_source_index"] = augment_source_index.astype(np.int64)
    out["yaw_augmentation_deg"] = augment_yaw_deg.astype(np.float32)
    out["label_privileged_source_code"] = source_code.astype(np.int64)
    out["label_privileged_source"] = np.asarray(
        ["pose_rollout" if x == 1 else "delta_approx" if x == 2 else "missing" for x in source_code],
        dtype="U32",
    )
    out["label_target_pose_key"] = np.full((n,), str(target_key), dtype="U64")
    out["label_delta_key"] = np.full((n,), str(delta_key), dtype="U64")
    out["label_yaw_symmetry_period"] = np.full((n,), float(args.yaw_symmetry_period), dtype=np.float32)
    for key, value in labels.items():
        out[key] = np.asarray(value).astype(np.float32)
    out["sample_weight"] = _row_weights(labels).astype(np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "depth_force_privileged_geometry_candidate_dataset.npz"
    np.savez_compressed(out_path, **out)

    mode_hist = _hist(target_mode)
    source_hist = _hist(out["label_privileged_source"])
    geom_second_gap = _second_best_gap(geom_cost, best_geometry_index)
    total_second_gap = _second_best_gap(total_cost, best_risk_aware_index)
    report = {
        "support_npz": str(args.support_npz),
        "risk_npz": str(args.risk_npz),
        "output_npz": str(out_path),
        "input_rows": int(n_all),
        "rows": int(n),
        "candidate_count": int(c),
        "target_pose_key": str(target_key),
        "delta_key": str(delta_key),
        "candidate_bank_npz": str(args.candidate_bank_npz) if args.candidate_bank_npz else "",
        "candidate_bank_key": str(candidate_override_key),
        "privileged_source_hist": source_hist,
        "target_mode_hist": mode_hist,
        "best_geometry_is_yaw_rate": float(np.mean(best_geom_yaw.astype(np.float32))),
        "yaw_opportunity_count": int(np.sum(yaw_opportunity)),
        "yaw_opportunity_rate": float(np.mean(yaw_opportunity.astype(np.float32))),
        "yaw_augmentation_count": int(np.sum(augment_flag > 0.5)),
        "yaw_augmentation_rate": float(np.mean(augment_flag.astype(np.float32))),
        "align_refine_count": int(np.sum(target_mode == "align_refine")),
        "best_vs_baseline_geometry_gap": _summary_stats(geometry_improvement),
        "best_vs_second_geometry_gap": _summary_stats(geom_second_gap),
        "risk_aware_best_vs_baseline_gap": _summary_stats(total_improvement),
        "risk_aware_best_vs_second_gap": _summary_stats(total_second_gap),
        "baseline_geometry_cost": _summary_stats(baseline_geometry),
        "best_geometry_cost": _summary_stats(best_geometry),
        "baseline_total_cost": _summary_stats(baseline_total),
        "best_total_cost": _summary_stats(best_total),
        "runtime_input_keys": [k for k in runtime_safe_keys if k in out],
        "label_privileged_keys": [
            "candidate_privileged_geometry_cost",
            "candidate_post_delta_privileged",
            "label_privileged_source",
            "label_privileged_source_code",
            "label_target_pose_key",
            "label_delta_key",
            "label_yaw_symmetry_period",
        ],
        "runtime_input_privileged_leak_keys": [
            k for k in out.keys()
            if k in runtime_safe_keys and (k.startswith("privileged_") or "privileged" in k)
        ],
        "output_privileged_label_keys": [k for k in out.keys() if k.startswith("privileged_") or "privileged" in k],
        "weights": {
            "lambda_geo": float(args.lambda_geo),
            "lambda_risk": float(args.lambda_risk),
            "lambda_smooth": float(args.lambda_smooth),
            "r_xy": float(args.r_xy),
            "r_z": float(args.r_z),
            "r_yaw": float(args.r_yaw),
            "r_tilt": float(args.r_tilt),
            "yaw_symmetry_period": float(args.yaw_symmetry_period),
        },
        "cost_scales": {
            "geometry_scale_p90": float(geometry_scale),
            "risk_scale_p90": float(risk_scale),
            "smooth_scale_p90": float(smooth_scale),
        },
        "normalized_best_vs_baseline_gap": _summary_stats(total_improvement_norm),
        "per_mode": {},
        "per_episode": {},
    }
    for mode in sorted(set(target_mode.tolist())):
        m = target_mode == mode
        report["per_mode"][str(mode)] = {
            "rows": int(np.sum(m)),
            "best_geometry_is_yaw_rate": float(np.mean(best_geom_yaw[m].astype(np.float32))) if np.any(m) else 0.0,
            "yaw_opportunity_count": int(np.sum(yaw_opportunity[m])) if np.any(m) else 0,
            "yaw_augmentation_count": int(np.sum(augment_flag[m] > 0.5)) if np.any(m) else 0,
            "best_vs_baseline_geometry_gap": _summary_stats(geometry_improvement[m]),
            "risk_aware_best_vs_baseline_gap": _summary_stats(total_improvement[m]),
            "privileged_source_hist": _hist(out["label_privileged_source"][m]) if np.any(m) else {},
        }
    if "episode_index" in out:
        ep_arr = np.asarray(out["episode_index"], dtype=np.int64)
        for ep in sorted(int(x) for x in np.unique(ep_arr)):
            m = ep_arr == ep
            report["per_episode"][str(ep)] = {
                "rows": int(np.sum(m)),
                "target_mode_hist": _hist(target_mode[m]),
                "best_geometry_is_yaw_rate": float(np.mean(best_geom_yaw[m].astype(np.float32))) if np.any(m) else 0.0,
                "yaw_opportunity_count": int(np.sum(yaw_opportunity[m])) if np.any(m) else 0,
                "yaw_augmentation_count": int(np.sum(augment_flag[m] > 0.5)) if np.any(m) else 0,
                "best_vs_baseline_geometry_gap": _summary_stats(geometry_improvement[m]),
                "risk_aware_best_vs_baseline_gap": _summary_stats(total_improvement[m]),
            }
    (out_dir / "privileged_geometry_candidate_dataset_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
