#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_depth_force_candidate_cost_dataset import _angle_abs_diff, _candidate_kind, _load_npz
from build_pose_candidate_dataset import parse_float_list_arg, pose_delta_local_between


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _group_stats(arr: np.ndarray, label: np.ndarray) -> dict[str, object]:
    mask = np.asarray(label, dtype=np.float32) > 0.5
    return {
        "rows": int(np.sum(mask)),
        "rate": float(np.mean(mask)),
        "mean": _summary_stats(np.asarray(arr)[mask]) if np.any(mask) else _summary_stats(np.array([])),
    }


def _default_future_risk(support: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    n = int(next(iter(support.values())).shape[0])
    contact = np.asarray(support.get("contact_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
    force_spike = np.asarray(support.get("force_spike_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
    jam = np.asarray(support.get("jam_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
    motion_stall = np.asarray(support.get("motion_stall_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
    kin_invalid = np.asarray(support.get("kinematic_invalid_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
    action_invalid = np.asarray(support.get("action_range_invalid_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
    episode = np.asarray(support["episode_index"], dtype=np.int64)
    step = np.asarray(support["step_index"], dtype=np.int64)
    row_future = np.zeros((n,), dtype=np.float32)
    row_future_contact = np.zeros((n,), dtype=np.float32)
    row_future_spike = np.zeros((n,), dtype=np.float32)
    row_future_jam = np.zeros((n,), dtype=np.float32)
    row_future_stall = np.zeros((n,), dtype=np.float32)
    row_future_kin = np.zeros((n,), dtype=np.float32)
    row_future_invalid = np.zeros((n,), dtype=np.float32)
    row_future_mask = np.zeros((n,), dtype=np.float32)
    by_ep = {}
    for i, ep in enumerate(episode.tolist()):
        by_ep.setdefault(int(ep), []).append(i)
    for ep, ids in by_ep.items():
        ids = sorted(ids, key=lambda i: int(step[i]))
        for pos, idx in enumerate(ids):
            future = ids[pos + 1 : pos + 1 + horizon]
            if not future:
                future = ids[pos : pos + 1]
            row_future_mask[idx] = 1.0
            row_future_contact[idx] = float(np.max(contact[future]))
            row_future_spike[idx] = float(np.max(force_spike[future]))
            row_future_jam[idx] = float(np.max(jam[future]))
            row_future_stall[idx] = float(np.max(motion_stall[future]))
            row_future_kin[idx] = float(np.max(kin_invalid[future]))
            row_future_invalid[idx] = float(np.max(action_invalid[future]))
            row_future[idx] = float(
                np.max(
                    np.stack(
                        [
                            row_future_contact[idx],
                            row_future_spike[idx],
                            row_future_jam[idx],
                            row_future_stall[idx],
                            row_future_kin[idx],
                            row_future_invalid[idx],
                        ],
                        axis=0,
                    )
                )
            )
    return {
        "future_risk_score": row_future,
        "future_contact_label": row_future_contact,
        "future_force_spike_label": row_future_spike,
        "future_jam_label": row_future_jam,
        "future_motion_stall_label": row_future_stall,
        "future_kinematic_invalid_label": row_future_kin,
        "future_action_range_invalid_label": row_future_invalid,
        "future_risk_mask": row_future_mask,
    }


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


def _candidate_matrix(
    data: dict[str, np.ndarray],
    keys: tuple[str, ...],
    n: int,
    c: int,
    *,
    default: float = 0.0,
) -> np.ndarray:
    for key in keys:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.shape == (n, c):
            return arr.astype(np.float32)
        if arr.ndim == 1 and arr.size == n * c:
            return arr.reshape(n, c).astype(np.float32)
    return np.full((n, c), float(default), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--risk_npz", default="")
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--candidate_bank_npz", default="")
    ap.add_argument("--candidate_bank_key", default="auto")
    ap.add_argument("--candidate_yaw_abs", type=float, default=0.02)
    ap.add_argument("--contact_weight", type=float, default=1.0)
    ap.add_argument("--spike_weight", type=float, default=1.0)
    ap.add_argument("--jam_weight", type=float, default=1.5)
    ap.add_argument("--stall_weight", type=float, default=0.75)
    ap.add_argument("--kin_invalid_weight", type=float, default=1.25)
    ap.add_argument("--action_invalid_weight", type=float, default=1.0)
    ap.add_argument("--future_risk_margin", type=float, default=0.0)
    ap.add_argument("--candidate_match_xy", type=float, default=0.0)
    ap.add_argument("--candidate_match_z", type=float, default=0.0)
    ap.add_argument("--candidate_match_yaw", type=float, default=0.0)
    ap.add_argument("--augment_interp_between_baseline_and_geometry", action="store_true", default=False)
    ap.add_argument("--interp_alphas", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--augment_toward_target_delta", action="store_true", default=False)
    ap.add_argument("--target_interp_alphas", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--augment_componentwise_target_grid", action="store_true", default=False)
    ap.add_argument("--componentwise_xy_scales", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--componentwise_z_scales", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--componentwise_yaw_scales", type=str, default="0.25,0.5,0.75")
    args = ap.parse_args()

    support = _load_npz(Path(args.support_npz))
    risk = _load_npz(Path(args.risk_npz)) if args.risk_npz else None
    labels = _default_future_risk(support, horizon=int(args.horizon))
    if risk is not None:
        for key in [
            "contact_label",
            "force_spike_label",
            "high_force_label",
            "jam_label",
            "motion_stall_label",
            "near_depth_label",
            "invalid_action_label",
            "invalid_action_nearby_label",
            "kinematic_invalid_label",
            "action_range_invalid_label",
            "safe_motion_label",
            "contact_risk",
        ]:
            if key in risk and key not in labels:
                labels[key] = np.asarray(risk[key], dtype=np.float32)

    n = int(next(iter(support.values())).shape[0])
    candidate_override_key = "built_in"
    if args.candidate_bank_npz:
        candidate_override = _load_npz(Path(args.candidate_bank_npz))
        candidate_actions, candidate_override_key = _select_candidate_bank(candidate_override, args.candidate_bank_key)
        if candidate_actions is None:
            raise SystemExit(
                f"candidate bank override {args.candidate_bank_npz} does not contain a valid (C, 6) array; "
                "expected candidate_actions_local or candidate_bank"
            )
    else:
        candidate_actions = np.asarray(support["candidate_actions_local"], dtype=np.float32)
    candidate_actions = np.asarray(candidate_actions, dtype=np.float32)
    episode_index = np.asarray(support["episode_index"], dtype=np.int64)
    step_index = np.asarray(support["step_index"], dtype=np.int64)
    current_pose = np.asarray(
        support.get("privileged_current_pose_7d", support.get("current_pose_7d", np.zeros((n, 7), dtype=np.float32))),
        dtype=np.float32,
    )
    target_pose = np.asarray(
        support.get("privileged_motion_target_pose_7d", support.get("privileged_basin_center_pose_7d", np.zeros((n, 7), dtype=np.float32))),
        dtype=np.float32,
    )
    if target_pose.shape[-1] != 7:
        target_pose = np.zeros((n, 7), dtype=np.float32)
    if current_pose.shape[-1] != 7:
        current_pose = np.zeros((n, 7), dtype=np.float32)
    if candidate_actions.ndim == 2:
        candidate_actions = np.broadcast_to(candidate_actions[None, :, :], (n, candidate_actions.shape[0], 6)).copy()
    if candidate_actions.ndim != 3 or candidate_actions.shape[0] != n or candidate_actions.shape[-1] != 6:
        raise ValueError(
            f"candidate bank must have shape (C,6) or (N,C,6) with N={n}; got {candidate_actions.shape}"
        )
    c = int(candidate_actions.shape[1])
    baseline_idx = np.argmin(np.linalg.norm(candidate_actions[0], axis=1)).astype(np.int64)
    baseline_idx = np.full((n,), int(baseline_idx), dtype=np.int64)

    cand_kind = np.asarray([[_candidate_kind(a) for a in row] for row in candidate_actions], dtype=object)
    baseline_actions = candidate_actions[np.arange(n), baseline_idx]
    target_delta_local = np.stack([pose_delta_local_between(current_pose[i], target_pose[i]) for i in range(n)], axis=0).astype(np.float32)

    # One-step kinematic proxy: apply local offsets directly to current pose.
    candidate_post_pose = np.broadcast_to(current_pose[:, None, :], (n, c, 7)).copy()
    candidate_post_pose[..., :3] = candidate_post_pose[..., :3] + candidate_actions[..., :3]
    candidate_post_pose[..., 5] = candidate_post_pose[..., 5] + candidate_actions[..., 5]
    candidate_xy = np.linalg.norm(candidate_post_pose[..., :2] - target_pose[:, None, :2], axis=-1)
    candidate_z = np.abs(candidate_post_pose[..., 2] - target_pose[:, None, 2])
    candidate_yaw = _angle_abs_diff(candidate_post_pose[..., 5], target_pose[:, None, 5])
    candidate_geo_cost_seed = candidate_xy + candidate_z + candidate_yaw
    best_geom_idx_seed = np.argmin(candidate_geo_cost_seed, axis=1).astype(np.int64)
    geom_actions_seed = candidate_actions[np.arange(n), best_geom_idx_seed]

    interp_alphas = [float(x) for x in parse_float_list_arg(args.interp_alphas)]
    if bool(args.augment_interp_between_baseline_and_geometry) and interp_alphas:
        aug_actions = []
        aug_kind = []
        for alpha in interp_alphas:
            alpha = float(alpha)
            blended = (1.0 - alpha) * baseline_actions + alpha * geom_actions_seed
            aug_actions.append(blended[:, None, :].astype(np.float32))
            aug_kind.append(
                np.asarray(
                    [f"interp_bg_{alpha:.3f}"] * n,
                    dtype="U32",
                )[:, None]
            )
        if aug_actions:
            candidate_actions = np.concatenate([candidate_actions] + aug_actions, axis=1).astype(np.float32)
            cand_kind = np.concatenate([cand_kind] + aug_kind, axis=1).astype(object)

    target_interp_alphas = [float(x) for x in parse_float_list_arg(args.target_interp_alphas)]
    if bool(args.augment_toward_target_delta) and target_interp_alphas:
        aug_actions = []
        aug_kind = []
        for alpha in target_interp_alphas:
            alpha = float(alpha)
            blended = target_delta_local * alpha
            aug_actions.append(blended[:, None, :].astype(np.float32))
            aug_kind.append(
                np.asarray(
                    [f"interp_tgt_{alpha:.3f}"] * n,
                    dtype="U32",
                )[:, None]
            )
        if aug_actions:
            candidate_actions = np.concatenate([candidate_actions] + aug_actions, axis=1).astype(np.float32)
            cand_kind = np.concatenate([cand_kind] + aug_kind, axis=1).astype(object)

    if bool(args.augment_componentwise_target_grid):
        xy_scales = [float(x) for x in parse_float_list_arg(args.componentwise_xy_scales)]
        z_scales = [float(x) for x in parse_float_list_arg(args.componentwise_z_scales)]
        yaw_scales = [float(x) for x in parse_float_list_arg(args.componentwise_yaw_scales)]
        aug_actions = []
        aug_kind = []
        for sx in xy_scales:
            for sz in z_scales:
                for syaw in yaw_scales:
                    proposal = np.zeros((n, 6), dtype=np.float32)
                    proposal[:, 0] = target_delta_local[:, 0] * float(sx)
                    proposal[:, 1] = target_delta_local[:, 1] * float(sx)
                    proposal[:, 2] = target_delta_local[:, 2] * float(sz)
                    proposal[:, 5] = target_delta_local[:, 5] * float(syaw)
                    aug_actions.append(proposal[:, None, :].astype(np.float32))
                    aug_kind.append(
                        np.asarray(
                            [f"comp_tgt_xy{sx:.2f}_z{sz:.2f}_yaw{syaw:.2f}"] * n,
                            dtype="U48",
                        )[:, None]
                    )
        if aug_actions:
            candidate_actions = np.concatenate([candidate_actions] + aug_actions, axis=1).astype(np.float32)
            cand_kind = np.concatenate([cand_kind] + aug_kind, axis=1).astype(object)
    c = int(candidate_actions.shape[1])

    candidate_post_pose = np.broadcast_to(current_pose[:, None, :], (n, c, 7)).copy()
    candidate_post_pose[..., :3] = candidate_post_pose[..., :3] + candidate_actions[..., :3]
    candidate_post_pose[..., 5] = candidate_post_pose[..., 5] + candidate_actions[..., 5]
    candidate_xy = np.linalg.norm(candidate_post_pose[..., :2] - target_pose[:, None, :2], axis=-1)
    candidate_z = np.abs(candidate_post_pose[..., 2] - target_pose[:, None, 2])
    candidate_yaw = _angle_abs_diff(candidate_post_pose[..., 5], target_pose[:, None, 5])
    candidate_geo_cost = candidate_xy + candidate_z + candidate_yaw
    best_geom_idx = np.argmin(candidate_geo_cost, axis=1).astype(np.int64)
    geom_actions = candidate_actions[np.arange(n), best_geom_idx]

    base_post_pose = current_pose.copy()
    base_post_pose[:, :3] += baseline_actions[:, :3]
    base_post_pose[:, 5] += baseline_actions[:, 5]
    base_xy = np.linalg.norm(base_post_pose[:, :2] - target_pose[:, :2], axis=-1)
    base_z = np.abs(base_post_pose[:, 2] - target_pose[:, 2])
    base_yaw = _angle_abs_diff(base_post_pose[:, 5], target_pose[:, 5])
    baseline_geo = base_xy + base_z + base_yaw

    geom_post_pose = current_pose.copy()
    geom_post_pose[:, :3] += geom_actions[:, :3]
    geom_post_pose[:, 5] += geom_actions[:, 5]
    geom_xy = np.linalg.norm(geom_post_pose[:, :2] - target_pose[:, :2], axis=-1)
    geom_z = np.abs(geom_post_pose[:, 2] - target_pose[:, 2])
    geom_yaw = _angle_abs_diff(geom_post_pose[:, 5], target_pose[:, 5])
    geom_geo = geom_xy + geom_z + geom_yaw

    future_risk = labels["future_risk_score"].astype(np.float32)
    future_contact = labels["future_contact_label"].astype(np.float32)
    future_spike = labels["future_force_spike_label"].astype(np.float32)
    future_jam = labels["future_jam_label"].astype(np.float32)
    future_stall = labels["future_motion_stall_label"].astype(np.float32)
    future_kin = labels["future_kinematic_invalid_label"].astype(np.float32)
    future_invalid = labels["future_action_range_invalid_label"].astype(np.float32)
    future_mask = labels["future_risk_mask"].astype(np.float32)

    candidate_risk_proxy = _candidate_matrix(
        support,
        ("candidate_risk_cost_norm", "candidate_risk_cost"),
        n,
        c,
        default=0.0,
    )
    candidate_action_norm = np.linalg.norm(candidate_actions, axis=-1)
    candidate_xy_norm = np.linalg.norm(candidate_actions[..., :2], axis=-1)
    candidate_z_abs = np.abs(candidate_actions[..., 2])
    candidate_yaw_abs = np.abs(candidate_actions[..., 5])
    kind_bias = np.zeros_like(candidate_risk_proxy, dtype=np.float32)
    kind_bias[np.asarray(cand_kind == "hold")] -= 0.15
    kind_bias[np.asarray(cand_kind == "descend")] += 0.05
    kind_bias[np.asarray(cand_kind == "yaw")] += 0.08
    kind_bias[np.asarray(cand_kind == "xy")] += 0.05
    kind_bias[np.asarray(cand_kind == "xy_yaw")] += 0.12
    kind_bias[np.asarray(cand_kind == "xy_z")] += 0.10
    kind_bias[np.asarray(cand_kind == "tilt")] += 0.10
    action_proxy = (
        0.30 * candidate_action_norm
        + 0.40 * candidate_xy_norm
        + 0.20 * candidate_z_abs
        + 0.35 * candidate_yaw_abs
        + kind_bias
    ).astype(np.float32)
    candidate_future_risk = future_risk[:, None] + candidate_risk_proxy + action_proxy
    candidate_future_risk_delta = candidate_future_risk - future_risk[:, None]
    candidate_future_contact = np.repeat(future_contact[:, None], candidate_actions.shape[1], axis=1).astype(np.float32)
    candidate_future_spike = np.repeat(future_spike[:, None], candidate_actions.shape[1], axis=1).astype(np.float32)
    candidate_future_jam = np.repeat(future_jam[:, None], candidate_actions.shape[1], axis=1).astype(np.float32)
    candidate_future_stall = np.repeat(future_stall[:, None], candidate_actions.shape[1], axis=1).astype(np.float32)
    candidate_future_kin = np.repeat(future_kin[:, None], candidate_actions.shape[1], axis=1).astype(np.float32)
    candidate_future_invalid = np.repeat(future_invalid[:, None], candidate_actions.shape[1], axis=1).astype(np.float32)

    candidate_risk_mask = np.isfinite(candidate_future_risk)
    candidate_total_future_risk = (
        candidate_future_risk
        + float(args.contact_weight) * candidate_future_contact
        + float(args.spike_weight) * candidate_future_spike
        + float(args.jam_weight) * candidate_future_jam
        + float(args.stall_weight) * candidate_future_stall
        + float(args.kin_invalid_weight) * candidate_future_kin
        + float(args.action_invalid_weight) * candidate_future_invalid
    ).astype(np.float32)

    baseline_future_risk = candidate_total_future_risk[np.arange(n), baseline_idx]
    best_geom_idx = np.argmin(candidate_geo_cost, axis=1).astype(np.int64)
    geometry_future_risk = candidate_total_future_risk[np.arange(n), best_geom_idx]
    baseline_geometry_cost = candidate_geo_cost[np.arange(n), baseline_idx]
    best_geometry_cost = candidate_geo_cost[np.arange(n), best_geom_idx]
    best_future_risk_idx = np.argmin(candidate_total_future_risk + (~candidate_risk_mask) * 1e9, axis=1)

    future_risk_label = (candidate_total_future_risk > (baseline_future_risk[:, None] + float(args.future_risk_margin))).astype(np.float32)
    risk_nonincrease_label = (candidate_total_future_risk <= baseline_future_risk[:, None] + 1e-6).astype(np.float32)
    candidate_kind_flat = cand_kind.reshape(-1).astype("U32")

    out = dict(support)
    out.update(labels)
    out["candidate_actions_local"] = candidate_actions.astype(np.float32)
    out["candidate_mask"] = candidate_risk_mask.astype(np.float32)
    out["candidate_geometry_cost"] = candidate_geo_cost.astype(np.float32)
    out["candidate_geometry_score"] = (-candidate_geo_cost).astype(np.float32)
    out["candidate_future_risk_score"] = candidate_total_future_risk.astype(np.float32)
    out["candidate_future_risk_delta"] = candidate_future_risk_delta.astype(np.float32)
    out["candidate_future_contact_risk"] = candidate_future_contact.astype(np.float32)
    out["candidate_future_force_spike_risk"] = candidate_future_spike.astype(np.float32)
    out["candidate_future_jam_risk"] = candidate_future_jam.astype(np.float32)
    out["candidate_future_motion_stall_risk"] = candidate_future_stall.astype(np.float32)
    out["candidate_future_kinematic_invalid_risk"] = candidate_future_kin.astype(np.float32)
    out["candidate_future_action_range_invalid_risk"] = candidate_future_invalid.astype(np.float32)
    out["candidate_future_risk_label"] = future_risk_label.astype(np.float32)
    out["candidate_future_risk_nonincrease_label"] = risk_nonincrease_label.astype(np.float32)
    out["candidate_future_risk_mask"] = candidate_risk_mask.astype(np.float32)
    out["candidate_future_risk_best_index"] = best_future_risk_idx.astype(np.int64)
    out["candidate_future_risk_baseline_index"] = baseline_idx.astype(np.int64)
    out["candidate_future_risk_geom_index"] = best_geom_idx.astype(np.int64)
    out["candidate_best_geometry_index"] = best_geom_idx.astype(np.int64)
    out["candidate_baseline_index"] = baseline_idx.astype(np.int64)
    out["candidate_baseline_geometry_cost"] = baseline_geometry_cost.astype(np.float32)
    out["candidate_best_geometry_cost"] = best_geometry_cost.astype(np.float32)
    out["candidate_future_risk_best_vs_baseline"] = (baseline_future_risk - candidate_total_future_risk[np.arange(n), best_future_risk_idx]).astype(np.float32)
    out["candidate_future_risk_best_vs_geom"] = (geometry_future_risk - candidate_total_future_risk[np.arange(n), best_future_risk_idx]).astype(np.float32)
    out["candidate_kind"] = candidate_kind_flat.reshape(cand_kind.shape)
    out["candidate_future_risk_scale"] = np.asarray(
        {
            "contact_weight": float(args.contact_weight),
            "spike_weight": float(args.spike_weight),
            "jam_weight": float(args.jam_weight),
            "stall_weight": float(args.stall_weight),
            "kin_invalid_weight": float(args.kin_invalid_weight),
            "action_invalid_weight": float(args.action_invalid_weight),
            "future_risk_margin": float(args.future_risk_margin),
            "horizon": int(args.horizon),
        },
        dtype=object,
    )
    out["risk_future_report"] = np.asarray(
        json.dumps(
            {
                "rows": int(n),
                "episodes": int(np.unique(episode_index).size),
                "candidate_count": int(candidate_actions.shape[1]),
                "augment_interp_between_baseline_and_geometry": bool(args.augment_interp_between_baseline_and_geometry),
                "interp_alphas": interp_alphas,
                "augment_toward_target_delta": bool(args.augment_toward_target_delta),
                "target_interp_alphas": target_interp_alphas,
                "augment_componentwise_target_grid": bool(args.augment_componentwise_target_grid),
                "componentwise_xy_scales": [float(x) for x in parse_float_list_arg(args.componentwise_xy_scales)],
                "componentwise_z_scales": [float(x) for x in parse_float_list_arg(args.componentwise_z_scales)],
                "componentwise_yaw_scales": [float(x) for x in parse_float_list_arg(args.componentwise_yaw_scales)],
                "future_risk_score": _summary_stats(candidate_total_future_risk.reshape(-1)),
                "future_risk_delta": _summary_stats(candidate_future_risk_delta.reshape(-1)),
                "baseline_future_risk": _summary_stats(baseline_future_risk),
                "geometry_future_risk": _summary_stats(geometry_future_risk),
                "candidate_future_risk_dependent_rate": float(np.mean(np.std(candidate_total_future_risk, axis=1) > 1e-6)),
                "best_is_baseline_rate": float(np.mean(best_future_risk_idx == baseline_idx)),
                "best_is_geometry_rate": float(np.mean(best_future_risk_idx == best_geom_idx)),
                "geometry_selected_risk_increase_rate": float(np.mean(geometry_future_risk > baseline_future_risk + 1e-6)),
                "geometry_selected_risk_nonincrease_rate": float(np.mean(geometry_future_risk <= baseline_future_risk + 1e-6)),
                "future_risk_contact_rate": float(np.mean(candidate_future_contact.reshape(-1) > 0.5)),
                "future_risk_spike_rate": float(np.mean(candidate_future_spike.reshape(-1) > 0.5)),
                "future_risk_jam_rate": float(np.mean(candidate_future_jam.reshape(-1) > 0.5)),
                "future_risk_stall_rate": float(np.mean(candidate_future_stall.reshape(-1) > 0.5)),
                "future_risk_kin_invalid_rate": float(np.mean(candidate_future_kin.reshape(-1) > 0.5)),
                "future_risk_action_invalid_rate": float(np.mean(candidate_future_invalid.reshape(-1) > 0.5)),
                "per_episode": {
                    str(ep): {
                        "rows": int(np.sum(episode_index == ep)),
                        "geometry_future_risk": _summary_stats(geometry_future_risk[episode_index == ep]),
                        "baseline_future_risk": _summary_stats(baseline_future_risk[episode_index == ep]),
                        "geometry_selected_risk_increase_rate": float(np.mean((geometry_future_risk > baseline_future_risk + 1e-6)[episode_index == ep])),
                        "candidate_future_risk_dependent_rate": float(np.mean(np.std(candidate_total_future_risk[episode_index == ep], axis=1) > 1e-6)),
                    }
                    for ep in sorted(int(x) for x in np.unique(episode_index))
                },
            },
            indent=2,
        ),
        dtype=object,
    )

    np.savez_compressed(args.output_npz, **out)
    Path(args.output_json).write_text(
        json.dumps(json.loads(out["risk_future_report"].item()), indent=2),
        encoding="utf-8",
    )
    print(out["risk_future_report"].item())


if __name__ == "__main__":
    main()
