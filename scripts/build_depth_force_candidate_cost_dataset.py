#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_pose_candidate_dataset import candidate_offsets


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: np.asarray(raw[k]) for k in raw.files}


def _select_candidate_bank(data: dict[str, np.ndarray], key: str) -> tuple[np.ndarray | None, str]:
    candidates = ["candidate_actions_local", "candidate_bank", "candidate_offsets"]
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


def _concat_fields(fields: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = sorted(set().union(*(f.keys() for f in fields)))
    out: dict[str, np.ndarray] = {}
    for key in keys:
        exemplar = next((f[key] for f in fields if key in f), None)
        if exemplar is None:
            continue
        arrs = []
        for f in fields:
            n = int(next(iter(f.values())).shape[0])
            if key in f and tuple(np.asarray(f[key]).shape[1:]) == tuple(exemplar.shape[1:]):
                arrs.append(np.asarray(f[key]))
            else:
                shape = (n,) + tuple(exemplar.shape[1:])
                if exemplar.dtype.kind in ("U", "S", "O"):
                    arrs.append(np.full(shape, "", dtype=exemplar.dtype))
                else:
                    arrs.append(np.zeros(shape, dtype=exemplar.dtype))
        out[key] = np.concatenate(arrs, axis=0)
    return out


def _angle_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return np.abs((diff + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _candidate_kind(action: np.ndarray, *, eps: float = 1e-6) -> str:
    a = np.asarray(action, dtype=np.float32).reshape(6)
    x, y, z, p, r, yaw = map(float, a)
    if np.linalg.norm(a) <= eps:
        return "hold"
    if abs(yaw) > eps and abs(x) <= eps and abs(y) <= eps and abs(z) <= eps and abs(p) <= eps and abs(r) <= eps:
        return "yaw"
    if abs(z) > eps and abs(x) <= eps and abs(y) <= eps and abs(yaw) <= eps and abs(p) <= eps and abs(r) <= eps:
        return "descend"
    if abs(x) > eps or abs(y) > eps:
        if abs(yaw) > eps and abs(z) <= eps:
            return "xy_yaw"
        if abs(z) > eps:
            return "xy_z"
        if abs(p) > eps or abs(r) > eps:
            return "xy_tilt"
        return "xy"
    if abs(p) > eps or abs(r) > eps:
        return "tilt"
    return "other"


def _risk_join(
    support: dict[str, np.ndarray],
    risk: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    n = int(next(iter(support.values())).shape[0])
    out: dict[str, np.ndarray] = {}

    if risk is None:
        # Fallback to weak labels from the clean support if a risk file was not provided.
        force_norm = np.asarray(support.get("force_norm", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
        force_delta_norm = np.asarray(support.get("force_delta_norm", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
        depth_proximity = np.asarray(support.get("depth_proximity", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
        contact_state = np.asarray(support.get("contact_state", np.zeros((n,), dtype=np.int64)), dtype=np.int64)
        gripper_state = np.asarray(support.get("gripper_state", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
        contact_label = (contact_state > 0).astype(np.float32)
        force_spike_label = ((force_norm > 0.045) | (force_delta_norm > 0.012)).astype(np.float32)
        near_depth_label = (depth_proximity < 0.085).astype(np.float32)
        motion_stall_label = ((force_norm > 0.03) & (depth_proximity < 0.09) & (gripper_state > 0.5)).astype(np.float32)
        jam_label = np.maximum(force_spike_label, motion_stall_label).astype(np.float32)
        invalid_action_label = np.zeros((n,), dtype=np.float32)
        invalid_action_nearby_label = np.zeros((n,), dtype=np.float32)
        kinematic_invalid_label = np.zeros((n,), dtype=np.float32)
        action_range_invalid_label = np.zeros((n,), dtype=np.float32)
        safe_motion_label = (1.0 - np.maximum.reduce([contact_label, force_spike_label, jam_label])).astype(np.float32)
        risk = {
            "contact_label": contact_label,
            "force_spike_label": force_spike_label,
            "high_force_label": (force_norm > 0.08).astype(np.float32),
            "jam_label": jam_label,
            "motion_stall_label": motion_stall_label,
            "near_depth_label": near_depth_label,
            "invalid_action_label": invalid_action_label,
            "invalid_action_nearby_label": invalid_action_nearby_label,
            "kinematic_invalid_label": kinematic_invalid_label,
            "action_range_invalid_label": action_range_invalid_label,
            "safe_motion_label": safe_motion_label,
        }
        return risk

    support_index = {
        (int(e), int(s)): i
        for i, (e, s) in enumerate(zip(np.asarray(support["episode_index"], dtype=np.int64), np.asarray(support["step_index"], dtype=np.int64)))
    }
    risk_index = {
        (int(e), int(s)): i
        for i, (e, s) in enumerate(zip(np.asarray(risk["episode_index"], dtype=np.int64), np.asarray(risk["step_index"], dtype=np.int64)))
    }
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
        if key not in risk:
            continue
        exemplar = np.asarray(risk[key])
        if exemplar.shape[0] != len(risk_index):
            raise ValueError(f"risk field {key} has incompatible row count")
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
        if key not in risk:
            out[key] = np.zeros((n,), dtype=np.float32)
            continue
        filled = np.zeros((n,), dtype=np.asarray(risk[key]).dtype)
        for row, key_ in enumerate(zip(np.asarray(support["episode_index"], dtype=np.int64), np.asarray(support["step_index"], dtype=np.int64))):
            idx = risk_index.get((int(key_[0]), int(key_[1])))
            if idx is None:
                continue
            filled[row] = np.asarray(risk[key])[idx]
        out[key] = filled.astype(np.float32)
    return out


def _row_weights(labels: dict[str, np.ndarray]) -> np.ndarray:
    contact = np.asarray(labels["contact_label"], dtype=np.float32)
    force_spike = np.asarray(labels["force_spike_label"], dtype=np.float32)
    jam = np.asarray(labels["jam_label"], dtype=np.float32)
    motion_stall = np.asarray(labels["motion_stall_label"], dtype=np.float32)
    near_depth = np.asarray(labels["near_depth_label"], dtype=np.float32)
    kin_invalid = np.asarray(labels["kinematic_invalid_label"], dtype=np.float32)
    weight = np.ones_like(contact, dtype=np.float32)
    weight += 0.5 * contact
    weight += 0.75 * force_spike
    weight += 1.0 * jam
    weight += 0.5 * motion_stall
    weight += 0.25 * near_depth
    weight += 0.5 * kin_invalid
    return weight.astype(np.float32)


def _safe_target_actions(
    planner_action: np.ndarray,
    *,
    contact: np.ndarray,
    force_spike: np.ndarray,
    jam: np.ndarray,
    motion_stall: np.ndarray,
    near_depth: np.ndarray,
    kin_invalid: np.ndarray,
    action_range_invalid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    plan = np.asarray(planner_action, dtype=np.float32).copy()
    target = plan.copy()
    mode = np.full((plan.shape[0],), "planner", dtype="U32")

    kin_mask = kin_invalid > 0.5
    range_mask = action_range_invalid > 0.5
    hold_mask = kin_mask | range_mask
    if np.any(hold_mask):
        target[hold_mask] = 0.25 * plan[hold_mask]
        mode[hold_mask] = "kinematic_hold"

    contact_mask = ((contact > 0.5) | (force_spike > 0.5) | (jam > 0.5)) & (~hold_mask)
    if np.any(contact_mask):
        target[contact_mask, :2] *= 0.25
        target[contact_mask, 5] *= 0.25
        target[contact_mask, 2] = -np.maximum(np.abs(plan[contact_mask, 2]), 0.004)
        mode[contact_mask] = "contact_backoff"

    stall_mask = (motion_stall > 0.5) & (~hold_mask) & (~contact_mask)
    if np.any(stall_mask):
        target[stall_mask, :2] *= 0.60
        target[stall_mask, 5] *= 0.50
        mode[stall_mask] = "stall_nudge"

    near_mask = (near_depth > 0.5) & (~hold_mask) & (~contact_mask) & (~stall_mask)
    if np.any(near_mask):
        target[near_mask, :2] *= 0.80
        target[near_mask, 5] *= 0.80
        mode[near_mask] = "near_hold"

    return target.astype(np.float32), mode


def _geom_cost(candidate: np.ndarray, target: np.ndarray, *, w_xy: float, w_z: float, w_yaw: float, w_tilt: float) -> np.ndarray:
    cand = np.asarray(candidate, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    xy = np.linalg.norm(cand[:, :2] - tgt[:, :2], axis=-1)
    z = np.abs(cand[:, 2] - tgt[:, 2])
    yaw = _angle_abs_diff(cand[:, 5], tgt[:, 5])
    tilt = np.linalg.norm(cand[:, 3:5] - tgt[:, 3:5], axis=-1)
    return (
        float(w_xy) * (xy ** 2)
        + float(w_z) * z
        + float(w_yaw) * (yaw ** 2)
        + float(w_tilt) * (tilt ** 2)
    ).astype(np.float32)


def _risk_cost(
    candidate: np.ndarray,
    *,
    contact: np.ndarray,
    force_spike: np.ndarray,
    jam: np.ndarray,
    motion_stall: np.ndarray,
    near_depth: np.ndarray,
    kin_invalid: np.ndarray,
    action_range_invalid: np.ndarray,
    gripper_state: np.ndarray,
    w_contact_xy: float,
    w_contact_z: float,
    w_contact_yaw: float,
    w_contact_hold: float,
    w_contact_backoff: float,
    w_spike_xy: float,
    w_spike_z: float,
    w_spike_yaw: float,
    w_jam_xy: float,
    w_jam_z: float,
    w_jam_yaw: float,
    w_jam_hold: float,
    w_jam_backoff: float,
    w_stall_hold: float,
    w_stall_small: float,
    w_near_xy: float,
    w_near_z: float,
    w_near_yaw: float,
    w_kin_mag: float,
    w_kin_yaw: float,
    w_range_mag: float,
) -> np.ndarray:
    cand = np.asarray(candidate, dtype=np.float32)
    abs_xy = np.linalg.norm(cand[:, :2], axis=-1)
    abs_z = np.abs(cand[:, 2])
    pos_z = np.maximum(cand[:, 2], 0.0)
    neg_z = np.maximum(-cand[:, 2], 0.0)
    abs_yaw = np.abs(cand[:, 5])
    abs_tilt = np.linalg.norm(cand[:, 3:5], axis=-1)
    abs_total = np.linalg.norm(cand, axis=-1)
    is_hold = abs_total <= 1e-8
    is_open = gripper_state > 0.5

    risk = np.zeros((cand.shape[0],), dtype=np.float32)
    risk += contact * (
        float(w_contact_xy) * abs_xy
        + float(w_contact_z) * pos_z
        + float(w_contact_yaw) * abs_yaw
        + float(w_contact_hold) * is_hold.astype(np.float32)
        - float(w_contact_backoff) * neg_z
    )
    risk += force_spike * (
        float(w_spike_xy) * abs_xy
        + float(w_spike_z) * pos_z
        + float(w_spike_yaw) * abs_yaw
        + 0.15 * abs_tilt
    )
    risk += jam * (
        float(w_jam_xy) * abs_xy
        + float(w_jam_z) * pos_z
        + float(w_jam_yaw) * abs_yaw
        + float(w_jam_hold) * is_hold.astype(np.float32)
        - float(w_jam_backoff) * neg_z
    )
    risk += motion_stall * (
        float(w_stall_hold) * is_hold.astype(np.float32)
        + float(w_stall_small) * np.minimum(abs_total, 0.01) / 0.01
    )
    risk += near_depth * (
        float(w_near_xy) * abs_xy
        + float(w_near_z) * pos_z
        + float(w_near_yaw) * abs_yaw
    )
    risk += kin_invalid * (
        float(w_kin_mag) * abs_total
        + float(w_kin_yaw) * abs_yaw
        + 0.15 * abs_tilt
    )
    risk += action_range_invalid * (
        float(w_range_mag) * abs_total
        + 0.20 * abs_yaw
    )
    # A tiny stabilization term: if the gripper is open and the candidate is
    # a pure hold, keep the cost slightly above 0 so that the ranker can still
    # learn to move when motion is needed.
    risk += ((is_open.astype(np.float32)) * 0.0)
    return risk.astype(np.float32)


def _safe_best_index(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(cost, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    masked = np.where(mask, arr, np.inf)
    return np.argmin(masked, axis=1).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_npz", required=True)
    ap.add_argument("--risk_npz", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--candidate_bank_npz", default="")
    ap.add_argument("--candidate_bank_key", default="auto")
    ap.add_argument("--candidate_mode", type=str, default="primitives", choices=("grid", "primitives"))
    ap.add_argument("--force_rebuild_candidate_bank", action="store_true", default=False)
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
    ap.add_argument("--primitive_yaw_probe_values", type=str, default="0.06,0.12")
    ap.add_argument("--primitive_pitch_small", type=float, default=0.06)
    ap.add_argument("--primitive_roll_small", type=float, default=0.06)
    ap.add_argument("--primitive_include_descend", action="store_true", default=True)
    ap.add_argument("--no_primitive_include_descend", dest="primitive_include_descend", action="store_false")
    ap.add_argument("--primitive_include_combos", action="store_true", default=True)
    ap.add_argument("--no_primitive_include_combos", dest="primitive_include_combos", action="store_false")
    ap.add_argument("--primitive_include_tilt", action="store_true", default=False)
    ap.add_argument("--no_primitive_include_tilt", dest="primitive_include_tilt", action="store_false")
    ap.add_argument("--risk_scale", type=float, default=1.0)
    ap.add_argument("--w_geom_xy", type=float, default=8.0)
    ap.add_argument("--w_geom_z", type=float, default=6.0)
    ap.add_argument("--w_geom_yaw", type=float, default=4.0)
    ap.add_argument("--w_geom_tilt", type=float, default=2.0)
    ap.add_argument("--w_contact_xy", type=float, default=1.0)
    ap.add_argument("--w_contact_z", type=float, default=1.4)
    ap.add_argument("--w_contact_yaw", type=float, default=1.1)
    ap.add_argument("--w_contact_hold", type=float, default=0.8)
    ap.add_argument("--w_contact_backoff", type=float, default=0.6)
    ap.add_argument("--w_spike_xy", type=float, default=0.8)
    ap.add_argument("--w_spike_z", type=float, default=1.0)
    ap.add_argument("--w_spike_yaw", type=float, default=1.2)
    ap.add_argument("--w_jam_xy", type=float, default=1.2)
    ap.add_argument("--w_jam_z", type=float, default=1.6)
    ap.add_argument("--w_jam_yaw", type=float, default=1.4)
    ap.add_argument("--w_jam_hold", type=float, default=1.0)
    ap.add_argument("--w_jam_backoff", type=float, default=0.9)
    ap.add_argument("--w_stall_hold", type=float, default=1.2)
    ap.add_argument("--w_stall_small", type=float, default=0.6)
    ap.add_argument("--w_near_xy", type=float, default=0.6)
    ap.add_argument("--w_near_z", type=float, default=0.8)
    ap.add_argument("--w_near_yaw", type=float, default=0.6)
    ap.add_argument("--w_kin_mag", type=float, default=1.4)
    ap.add_argument("--w_kin_yaw", type=float, default=0.8)
    ap.add_argument("--w_range_mag", type=float, default=1.6)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    args = ap.parse_args()

    support = _load_npz(Path(args.support_npz))
    risk = _load_npz(Path(args.risk_npz)) if args.risk_npz else None
    labels = _risk_join(support, risk)

    n = int(next(iter(support.values())).shape[0])
    if risk is not None:
        if int(next(iter(risk.values())).shape[0]) != n:
            raise ValueError("support and risk npz must have the same row count")

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
        candidate_actions = None
    if args.force_rebuild_candidate_bank or candidate_actions is None:
        candidate_actions = candidate_offsets(args).astype(np.float32)
    else:
        candidate_actions = candidate_actions.astype(np.float32)
    if candidate_actions.ndim != 2 or candidate_actions.shape[1] != 6:
        raise ValueError(f"candidate bank must have shape (C, 6); got {candidate_actions.shape}")

    candidate_mask = np.ones((n, candidate_actions.shape[0]), dtype=np.float32)
    candidate_bank = np.broadcast_to(candidate_actions[None, :, :], (n, candidate_actions.shape[0], 6)).copy().astype(np.float32)

    planner_action = np.asarray(
        support.get("planner_base_action_local_raw", support.get("planner_base_action_local", support.get("planner_base_action", np.zeros((n, 6), dtype=np.float32)))),
        dtype=np.float32,
    )
    if planner_action.ndim == 1:
        planner_action = np.repeat(planner_action[None, :], n, axis=0)
    executed_action = np.asarray(support.get("executed_action_local", planner_action), dtype=np.float32)
    if executed_action.ndim == 1:
        executed_action = np.repeat(executed_action[None, :], n, axis=0)

    # Geometry oracle is anchored to a risk-conditioned safe target derived from
    # the clean planner action and the split risk labels. This keeps the dataset
    # aligned with the frozen planner while still exposing non-trivial candidate
    # preference shifts in contact / jam / invalid contexts.
    safe_target, target_mode = _safe_target_actions(
        planner_action,
        contact=np.asarray(labels["contact_label"], dtype=np.float32),
        force_spike=np.asarray(labels["force_spike_label"], dtype=np.float32),
        jam=np.asarray(labels["jam_label"], dtype=np.float32),
        motion_stall=np.asarray(labels["motion_stall_label"], dtype=np.float32),
        near_depth=np.asarray(labels["near_depth_label"], dtype=np.float32),
        kin_invalid=np.asarray(labels["kinematic_invalid_label"], dtype=np.float32),
        action_range_invalid=np.asarray(labels["action_range_invalid_label"], dtype=np.float32),
    )
    geom_cost = _geom_cost(
        candidate_bank.reshape(-1, 6),
        np.repeat(safe_target[:, None, :], candidate_actions.shape[0], axis=1).reshape(-1, 6),
        w_xy=args.w_geom_xy,
        w_z=args.w_geom_z,
        w_yaw=args.w_geom_yaw,
        w_tilt=args.w_geom_tilt,
    ).reshape(n, candidate_actions.shape[0])
    risk_cost = _risk_cost(
        candidate_bank.reshape(-1, 6),
        contact=np.repeat(labels["contact_label"][:, None], candidate_actions.shape[0], axis=1).reshape(-1),
        force_spike=np.repeat(labels["force_spike_label"][:, None], candidate_actions.shape[0], axis=1).reshape(-1),
        jam=np.repeat(labels["jam_label"][:, None], candidate_actions.shape[0], axis=1).reshape(-1),
        motion_stall=np.repeat(labels["motion_stall_label"][:, None], candidate_actions.shape[0], axis=1).reshape(-1),
        near_depth=np.repeat(labels["near_depth_label"][:, None], candidate_actions.shape[0], axis=1).reshape(-1),
        kin_invalid=np.repeat(labels["kinematic_invalid_label"][:, None], candidate_actions.shape[0], axis=1).reshape(-1),
        action_range_invalid=np.repeat(labels["action_range_invalid_label"][:, None], candidate_actions.shape[0], axis=1).reshape(-1),
        gripper_state=np.repeat(np.asarray(support.get("gripper_state", np.zeros((n,), dtype=np.float32)), dtype=np.float32)[:, None], candidate_actions.shape[0], axis=1).reshape(-1),
        w_contact_xy=args.w_contact_xy,
        w_contact_z=args.w_contact_z,
        w_contact_yaw=args.w_contact_yaw,
        w_contact_hold=args.w_contact_hold,
        w_contact_backoff=args.w_contact_backoff,
        w_spike_xy=args.w_spike_xy,
        w_spike_z=args.w_spike_z,
        w_spike_yaw=args.w_spike_yaw,
        w_jam_xy=args.w_jam_xy,
        w_jam_z=args.w_jam_z,
        w_jam_yaw=args.w_jam_yaw,
        w_jam_hold=args.w_jam_hold,
        w_jam_backoff=args.w_jam_backoff,
        w_stall_hold=args.w_stall_hold,
        w_stall_small=args.w_stall_small,
        w_near_xy=args.w_near_xy,
        w_near_z=args.w_near_z,
        w_near_yaw=args.w_near_yaw,
        w_kin_mag=args.w_kin_mag,
        w_kin_yaw=args.w_kin_yaw,
        w_range_mag=args.w_range_mag,
    ).reshape(n, candidate_actions.shape[0])

    total_cost = geom_cost + float(args.risk_scale) * risk_cost
    candidate_geometry_score = (-geom_cost).astype(np.float32)
    candidate_risk_score = (-risk_cost).astype(np.float32)
    candidate_total_score = (-total_cost).astype(np.float32)

    baseline_index = _safe_best_index(
        _geom_cost(
            candidate_bank.reshape(-1, 6),
            np.repeat(planner_action[:, None, :], candidate_actions.shape[0], axis=1).reshape(-1, 6),
            w_xy=args.w_geom_xy,
            w_z=args.w_geom_z,
            w_yaw=args.w_geom_yaw,
            w_tilt=args.w_geom_tilt,
        ).reshape(n, candidate_actions.shape[0]),
        candidate_mask,
    )
    executed_index = _safe_best_index(geom_cost, candidate_mask)
    best_geometry_index = _safe_best_index(geom_cost, candidate_mask)
    best_risk_index = _safe_best_index(total_cost, candidate_mask)

    row_ids = np.arange(n, dtype=np.int64)
    best_geom_cost = geom_cost[row_ids, best_geometry_index]
    best_risk_cost = total_cost[row_ids, best_risk_index]
    baseline_geom_cost = geom_cost[row_ids, baseline_index]
    baseline_risk_cost = total_cost[row_ids, baseline_index]
    executed_geom_cost = geom_cost[row_ids, executed_index]
    executed_risk_cost = total_cost[row_ids, executed_index]

    candidate_kind = np.asarray([_candidate_kind(a) for a in candidate_actions], dtype="U16")
    candidate_kind_per_row = np.broadcast_to(candidate_kind[None, :], (n, candidate_kind.shape[0])).copy()
    target_mode_per_row = np.asarray(target_mode, dtype="U32")

    out: dict[str, np.ndarray] = {}
    for key in [
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
    ]:
        if key in support:
            out[key] = np.asarray(support[key])

    out["candidate_actions_local"] = candidate_bank
    out["candidate_mask"] = candidate_mask.astype(np.float32)
    out["candidate_geometry_cost"] = geom_cost.astype(np.float32)
    out["candidate_risk_cost"] = risk_cost.astype(np.float32)
    out["candidate_total_cost"] = total_cost.astype(np.float32)
    out["candidate_geometry_score"] = candidate_geometry_score
    out["candidate_risk_score"] = candidate_risk_score
    out["candidate_oracle_score"] = candidate_total_score
    out["candidate_best_index"] = best_risk_index.astype(np.int64)
    out["best_candidate_index"] = best_risk_index.astype(np.int64)
    out["candidate_best_geometry_index"] = best_geometry_index.astype(np.int64)
    out["candidate_best_risk_index"] = best_risk_index.astype(np.int64)
    out["candidate_baseline_index"] = baseline_index.astype(np.int64)
    out["baseline_candidate_index"] = baseline_index.astype(np.int64)
    out["candidate_executed_index"] = executed_index.astype(np.int64)
    out["candidate_kind"] = candidate_kind_per_row
    out["candidate_geometry_improvement"] = (baseline_geom_cost - best_geom_cost).astype(np.float32)
    out["candidate_risk_improvement"] = (baseline_risk_cost - best_risk_cost).astype(np.float32)
    out["candidate_total_improvement"] = (baseline_risk_cost - best_risk_cost).astype(np.float32)
    out["candidate_best_geometry_cost"] = best_geom_cost.astype(np.float32)
    out["candidate_best_risk_cost"] = best_risk_cost.astype(np.float32)
    out["candidate_baseline_geom_cost"] = baseline_geom_cost.astype(np.float32)
    out["candidate_baseline_risk_cost"] = baseline_risk_cost.astype(np.float32)
    out["candidate_executed_geom_cost"] = executed_geom_cost.astype(np.float32)
    out["candidate_executed_risk_cost"] = executed_risk_cost.astype(np.float32)
    out["candidate_target_mode"] = target_mode_per_row
    out["safe_target_action_local"] = safe_target.astype(np.float32)

    for key, value in labels.items():
        out[key] = np.asarray(value).astype(np.float32)

    out["sample_weight"] = _row_weights(labels).astype(np.float32)
    out["candidate_scope_size"] = candidate_mask.sum(axis=1).astype(np.float32)
    out["risk_scale"] = np.full((n,), float(args.risk_scale), dtype=np.float32)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "depth_force_candidate_cost_dataset.npz"
    np.savez_compressed(out_path, **out)

    best_is_baseline = best_risk_index == baseline_index
    best_is_executed = best_risk_index == executed_index
    best_is_yaw = np.abs(candidate_bank[row_ids, best_risk_index, 5]) > float(args.keep_yaw_abs)
    executed_is_yaw = np.abs(candidate_bank[row_ids, executed_index, 5]) > float(args.keep_yaw_abs)
    baseline_is_yaw = np.abs(candidate_bank[row_ids, baseline_index, 5]) > float(args.keep_yaw_abs)
    total_improve = baseline_risk_cost - best_risk_cost
    geom_improve = baseline_geom_cost - best_geom_cost
    risk_improve = baseline_risk_cost - best_risk_cost

    report = {
        "rows": int(n),
        "episodes": int(np.unique(out["episode_index"]).size) if "episode_index" in out else 0,
        "candidate_count": int(candidate_actions.shape[0]),
        "mean_scope_size": float(np.mean(out["candidate_scope_size"])),
        "candidate_kind_hist": {k: int(v) for k, v in zip(*np.unique(candidate_kind, return_counts=True))},
        "target_mode_hist": {k: int(v) for k, v in zip(*np.unique(target_mode_per_row, return_counts=True))},
        "contact_positive_rate": float(np.mean(labels["contact_label"])),
        "force_spike_positive_rate": float(np.mean(labels["force_spike_label"])),
        "jam_positive_rate": float(np.mean(labels["jam_label"])),
        "motion_stall_positive_rate": float(np.mean(labels["motion_stall_label"])),
        "kinematic_invalid_positive_rate": float(np.mean(labels["kinematic_invalid_label"])),
        "action_range_invalid_positive_rate": float(np.mean(labels["action_range_invalid_label"])),
        "near_depth_positive_rate": float(np.mean(labels["near_depth_label"])),
        "best_total_is_baseline_rate": float(np.mean(best_is_baseline.astype(np.float32))),
        "best_total_is_executed_rate": float(np.mean(best_is_executed.astype(np.float32))),
        "baseline_is_yaw_rate": float(np.mean(baseline_is_yaw.astype(np.float32))),
        "executed_is_yaw_rate": float(np.mean(executed_is_yaw.astype(np.float32))),
        "best_is_yaw_rate": float(np.mean(best_is_yaw.astype(np.float32))),
        "baseline_total_cost_mean": float(np.mean(baseline_risk_cost)),
        "best_total_cost_mean": float(np.mean(best_risk_cost)),
        "baseline_geom_cost_mean": float(np.mean(baseline_geom_cost)),
        "best_geom_cost_mean": float(np.mean(best_geom_cost)),
        "baseline_risk_cost_mean": float(np.mean(baseline_risk_cost)),
        "best_risk_cost_mean": float(np.mean(best_risk_cost)),
        "total_improvement_mean": float(np.mean(total_improve)),
        "geometry_improvement_mean": float(np.mean(geom_improve)),
        "risk_improvement_mean": float(np.mean(risk_improve)),
        "per_label_positive_summary": {
            key: {
                "rows": int(np.sum(labels[key] > 0.5)),
                "best_is_base_rate": float(np.mean((best_risk_index == baseline_index)[labels[key] > 0.5]))
                if np.any(labels[key] > 0.5)
                else 0.0,
                "total_improvement_mean": float(np.mean(total_improve[labels[key] > 0.5]))
                if np.any(labels[key] > 0.5)
                else 0.0,
                "geometry_improvement_mean": float(np.mean(geom_improve[labels[key] > 0.5]))
                if np.any(labels[key] > 0.5)
                else 0.0,
            }
            for key in [
                "contact_label",
                "force_spike_label",
                "jam_label",
                "motion_stall_label",
                "kinematic_invalid_label",
                "action_range_invalid_label",
                "near_depth_label",
            ]
        },
        "risk_scale": float(args.risk_scale),
        "keep_yaw_abs": float(args.keep_yaw_abs),
        "candidate_mode": str(args.candidate_mode),
        "candidate_bank_npz": str(args.candidate_bank_npz) if args.candidate_bank_npz else "",
        "candidate_bank_key": str(candidate_override_key),
        "output_npz": str(out_path),
        "per_episode_rows": {
            str(int(ep)): int(np.sum(out["episode_index"] == ep)) for ep in np.unique(out["episode_index"])
        }
        if "episode_index" in out
        else {},
    }
    (out_dir / "depth_force_candidate_cost_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
