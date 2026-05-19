from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from build_depth_force_candidate_cost_dataset import (
    _angle_abs_diff,
    _candidate_kind,
    _geom_cost,
    _load_npz,
    _risk_cost,
    _risk_join,
    _safe_target_actions,
)
from build_pose_candidate_dataset import apply_local_offset_to_pose, pose_delta_local_between, parse_float_list_arg


@dataclass(frozen=True)
class LocalProposalConfig:
    scale_values: tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 0.70, 1.00, 1.25)
    blend_alphas: tuple[float, ...] = (0.25, 0.50, 0.75)
    componentwise_xy_scales: tuple[float, ...] = (0.25, 0.50, 0.75)
    componentwise_z_scales: tuple[float, ...] = (0.25, 0.50, 0.75)
    componentwise_yaw_scales: tuple[float, ...] = (0.25, 0.50, 0.75)
    safe_jitter_count: int = 8
    target_jitter_count: int = 8
    jitter_xy_sigma: float = 0.0015
    jitter_z_sigma: float = 0.0015
    jitter_yaw_sigma: float = 0.02
    include_yaw_mirror: bool = True


def summary_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
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


def frontier_size(gain: np.ndarray, risk_delta: np.ndarray, mask: np.ndarray) -> int:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0
    g = np.asarray(gain[idx], dtype=np.float32)
    r = np.asarray(risk_delta[idx], dtype=np.float32)
    frontier = np.ones((idx.size,), dtype=bool)
    eps = 1e-6
    for i in range(idx.size):
        if not frontier[i]:
            continue
        dominated = (g >= g[i] - eps) & (r <= r[i] + eps) & ((g > g[i] + eps) | (r < r[i] - eps))
        dominated[i] = False
        if np.any(dominated):
            frontier[i] = False
    return int(np.sum(frontier))


def select_pose_fields(support: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, str]:
    cur = None
    for key in ("privileged_current_pose_7d", "current_pose_7d"):
        if key in support:
            cur = np.asarray(support[key], dtype=np.float32)
            break
    if cur is None:
        raise KeyError("support is missing current pose field")

    target = None
    target_source = ""
    for key in (
        "privileged_motion_target_pose_7d",
        "privileged_basin_center_pose_7d",
        "privileged_grasp_commit_target_pose_7d",
        "privileged_pregrasp_target_pose_7d",
        "basin_center_pose_7d",
        "motion_target_pose_7d",
        "target_pose_7d",
    ):
        if key in support:
            candidate = np.asarray(support[key], dtype=np.float32)
            if candidate.ndim == 2 and candidate.shape[1] == 7:
                target = candidate
                target_source = key
                break
    if target is None:
        raise KeyError("support is missing privileged target pose field")
    return cur, target, target_source


def select_planner_action(support: dict[str, np.ndarray]) -> np.ndarray:
    planner = np.asarray(
        support.get(
            "planner_base_action_local_raw",
            support.get("planner_base_action_local", support.get("planner_base_action", np.zeros((1, 6), dtype=np.float32))),
        ),
        dtype=np.float32,
    )
    if planner.ndim == 1:
        planner = planner[None, :]
    return planner.astype(np.float32)


def make_state_conditioned_proposals(
    planner_action: np.ndarray,
    target_delta_local: np.ndarray,
    safe_target_action: np.ndarray,
    *,
    cfg: LocalProposalConfig,
    row_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planner_action = np.asarray(planner_action, dtype=np.float32).reshape(6)
    target_delta_local = np.asarray(target_delta_local, dtype=np.float32).reshape(6)
    safe_target_action = np.asarray(safe_target_action, dtype=np.float32).reshape(6)
    rng = np.random.default_rng(int(row_seed))

    actions: list[np.ndarray] = []
    kinds: list[str] = []
    families: list[str] = []

    def add(action: np.ndarray, *, kind: str, family: str) -> None:
        arr = np.asarray(action, dtype=np.float32).reshape(6)
        actions.append(arr)
        kinds.append(kind)
        families.append(family)

    add(planner_action, kind="baseline", family="baseline")
    add(np.zeros(6, dtype=np.float32), kind="hold", family="baseline")
    add(safe_target_action, kind=_candidate_kind(safe_target_action), family="safe_target")
    add(target_delta_local, kind=_candidate_kind(target_delta_local), family="target_delta")

    for scale in cfg.scale_values:
        add(scale * target_delta_local, kind="target_scale", family=f"target_scale_{scale:.3f}")

    for alpha in cfg.blend_alphas:
        blended = (1.0 - float(alpha)) * planner_action + float(alpha) * target_delta_local
        add(blended, kind="blend", family=f"blend_{float(alpha):.3f}")

    for sx in cfg.componentwise_xy_scales:
        for sz in cfg.componentwise_z_scales:
            for syaw in cfg.componentwise_yaw_scales:
                comp = np.zeros(6, dtype=np.float32)
                comp[0] = target_delta_local[0] * float(sx)
                comp[1] = target_delta_local[1] * float(sx)
                comp[2] = target_delta_local[2] * float(sz)
                comp[5] = target_delta_local[5] * float(syaw)
                add(comp, kind="component_grid", family=f"comp_{sx:.2f}_{sz:.2f}_{syaw:.2f}")

    for idx in range(int(cfg.target_jitter_count)):
        noise = np.array(
            [
                rng.normal(0.0, float(cfg.jitter_xy_sigma)),
                rng.normal(0.0, float(cfg.jitter_xy_sigma)),
                rng.normal(0.0, float(cfg.jitter_z_sigma)),
                0.0,
                0.0,
                rng.normal(0.0, float(cfg.jitter_yaw_sigma)),
            ],
            dtype=np.float32,
        )
        add(target_delta_local + noise, kind="target_jitter", family=f"target_jitter_{idx}")

    for idx in range(int(cfg.safe_jitter_count)):
        noise = np.array(
            [
                rng.normal(0.0, float(cfg.jitter_xy_sigma)),
                rng.normal(0.0, float(cfg.jitter_xy_sigma)),
                rng.normal(0.0, float(cfg.jitter_z_sigma)),
                0.0,
                0.0,
                rng.normal(0.0, float(cfg.jitter_yaw_sigma)),
            ],
            dtype=np.float32,
        )
        add(safe_target_action + noise, kind="safe_jitter", family=f"safe_jitter_{idx}")

    if bool(cfg.include_yaw_mirror):
        mirror = safe_target_action.copy()
        mirror[5] = -float(target_delta_local[5])
        add(mirror, kind="yaw_mirror", family="yaw_mirror")
        mirror2 = target_delta_local.copy()
        mirror2[5] = -float(target_delta_local[5])
        add(mirror2, kind="yaw_mirror", family="yaw_mirror_target")

    return np.stack(actions, axis=0).astype(np.float32), np.asarray(kinds, dtype="U32"), np.asarray(families, dtype="U32")


def evaluate_state_conditioned_proposals(
    *,
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    candidate_actions: np.ndarray,
    contact: np.ndarray,
    force_spike: np.ndarray,
    jam: np.ndarray,
    motion_stall: np.ndarray,
    near_depth: np.ndarray,
    kin_invalid: np.ndarray,
    action_range_invalid: np.ndarray,
    gripper_state: np.ndarray,
    w_geom_xy: float = 8.0,
    w_geom_z: float = 6.0,
    w_geom_yaw: float = 4.0,
    w_geom_tilt: float = 2.0,
    w_contact_xy: float = 1.0,
    w_contact_z: float = 1.4,
    w_contact_yaw: float = 1.1,
    w_contact_hold: float = 0.8,
    w_contact_backoff: float = 0.6,
    w_spike_xy: float = 0.8,
    w_spike_z: float = 1.0,
    w_spike_yaw: float = 1.2,
    w_jam_xy: float = 1.2,
    w_jam_z: float = 1.6,
    w_jam_yaw: float = 1.4,
    w_jam_hold: float = 1.0,
    w_jam_backoff: float = 0.9,
    w_stall_hold: float = 1.2,
    w_stall_small: float = 0.6,
    w_near_xy: float = 0.6,
    w_near_z: float = 0.8,
    w_near_yaw: float = 0.6,
    w_kin_mag: float = 1.4,
    w_kin_yaw: float = 0.8,
    w_range_mag: float = 1.6,
) -> dict[str, np.ndarray]:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    candidate_actions = np.asarray(candidate_actions, dtype=np.float32)
    n, c, _ = candidate_actions.shape

    geom_cost = np.zeros((n, c), dtype=np.float32)
    for i in range(n):
        for j in range(c):
            post_pose = apply_local_offset_to_pose(current_pose[i], candidate_actions[i, j])
            delta_local = pose_delta_local_between(post_pose, target_pose[i])
            xy = float(np.linalg.norm(delta_local[:2]))
            z = float(np.abs(delta_local[2]))
            yaw = float(np.abs(delta_local[5]))
            tilt = float(np.linalg.norm(delta_local[3:5]))
            geom_cost[i, j] = (
                float(w_geom_xy) * (xy ** 2)
                + float(w_geom_z) * z
                + float(w_geom_yaw) * (yaw ** 2)
                + float(w_geom_tilt) * (tilt ** 2)
            )

    abs_xy = np.linalg.norm(candidate_actions[..., :2], axis=-1)
    abs_z = np.abs(candidate_actions[..., 2])
    pos_z = np.maximum(candidate_actions[..., 2], 0.0)
    neg_z = np.maximum(-candidate_actions[..., 2], 0.0)
    abs_yaw = np.abs(candidate_actions[..., 5])
    abs_tilt = np.linalg.norm(candidate_actions[..., 3:5], axis=-1)
    abs_total = np.linalg.norm(candidate_actions, axis=-1)
    is_hold = abs_total <= 1e-8
    is_open = np.asarray(gripper_state, dtype=np.float32) > 0.5

    risk = np.zeros((n, c), dtype=np.float32)
    risk += contact[:, None] * (
        float(w_contact_xy) * abs_xy
        + float(w_contact_z) * pos_z
        + float(w_contact_yaw) * abs_yaw
        + float(w_contact_hold) * is_hold.astype(np.float32)
        - float(w_contact_backoff) * neg_z
    )
    risk += force_spike[:, None] * (
        float(w_spike_xy) * abs_xy
        + float(w_spike_z) * pos_z
        + float(w_spike_yaw) * abs_yaw
        + 0.15 * abs_tilt
    )
    risk += jam[:, None] * (
        float(w_jam_xy) * abs_xy
        + float(w_jam_z) * pos_z
        + float(w_jam_yaw) * abs_yaw
        + float(w_jam_hold) * is_hold.astype(np.float32)
        - float(w_jam_backoff) * neg_z
    )
    risk += motion_stall[:, None] * (
        float(w_stall_hold) * is_hold.astype(np.float32)
        + float(w_stall_small) * np.minimum(abs_total, 0.01) / 0.01
    )
    risk += near_depth[:, None] * (
        float(w_near_xy) * abs_xy
        + float(w_near_z) * pos_z
        + float(w_near_yaw) * abs_yaw
    )
    risk += kin_invalid[:, None] * (
        float(w_kin_mag) * abs_total
        + float(w_kin_yaw) * abs_yaw
        + 0.15 * abs_tilt
    )
    risk += action_range_invalid[:, None] * (
        float(w_range_mag) * abs_total
        + 0.20 * abs_yaw
    )
    risk += ((is_open.astype(np.float32))[:, None] * 0.0)
    return {
        "candidate_geometry_cost": geom_cost.astype(np.float32),
        "candidate_risk_cost": risk.astype(np.float32),
        "candidate_total_cost": (geom_cost + risk).astype(np.float32),
    }


def select_best_indices(
    geom_cost: np.ndarray,
    risk_cost: np.ndarray,
    *,
    baseline_index: int | np.ndarray = 0,
    geo_margin: float = 0.0,
    risk_budget: float = 0.05,
    soft_alpha: float = 0.3,
) -> dict[str, np.ndarray]:
    geom = np.asarray(geom_cost, dtype=np.float32)
    risk = np.asarray(risk_cost, dtype=np.float32)
    if np.isscalar(baseline_index):
        baseline = np.full((geom.shape[0],), int(baseline_index), dtype=np.int64)
    else:
        baseline = np.asarray(baseline_index, dtype=np.int64).reshape(-1)
        if baseline.shape[0] != geom.shape[0]:
            raise ValueError("baseline_index must be scalar or have one entry per row")
    geom_top1 = np.argmin(geom, axis=1).astype(np.int64)
    base_geom = geom[np.arange(geom.shape[0]), baseline]
    base_risk = risk[np.arange(risk.shape[0]), baseline]
    geom_gain = base_geom[:, None] - geom
    risk_delta = risk - base_risk[:, None]
    safe_mask = (geom_gain > float(geo_margin)) & (risk_delta <= float(risk_budget))
    utility = geom_gain - float(soft_alpha) * np.maximum(risk_delta, 0.0)
    utility = np.where(np.isfinite(utility), utility, -1e9)
    safe_utility = np.where(safe_mask, utility, -1e9)
    soft_utility = utility.copy()
    budget_mask = risk_delta <= float(risk_budget)
    budget_utility = np.where(budget_mask, utility, -1e9)

    def _argbest(arr: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        best = np.argmax(arr, axis=1).astype(np.int64)
        invalid = ~np.isfinite(np.max(arr, axis=1))
        if np.any(invalid):
            best[invalid] = fallback[invalid]
        return best

    best_safe = _argbest(safe_utility, baseline)
    best_soft = _argbest(soft_utility, baseline)
    best_budget = _argbest(budget_utility, baseline)
    return {
        "baseline_index": baseline,
        "geom_top1_index": geom_top1,
        "best_safe_index": best_safe,
        "best_soft_index": best_soft,
        "best_budget_index": best_budget,
        "geometry_gain": geom_gain.astype(np.float32),
        "risk_delta": risk_delta.astype(np.float32),
        "safe_mask": safe_mask.astype(np.float32),
        "budget_mask": budget_mask.astype(np.float32),
        "pareto_feasible": np.any(safe_mask, axis=1).astype(np.float32),
        "missing_compromise": (~np.any(safe_mask, axis=1)).astype(np.float32),
        "safe_count": np.sum(safe_mask, axis=1).astype(np.float32),
        "frontier_size": np.asarray(
            [frontier_size(geom_gain[i], risk_delta[i], np.isfinite(geom[i]) & np.isfinite(risk[i])) for i in range(geom.shape[0])],
            dtype=np.float32,
        ),
        "geom_top1_risk_increase": (risk_delta[np.arange(geom.shape[0]), geom_top1] > 1e-6).astype(np.float32),
        "best_safe_risk_increase": (risk_delta[np.arange(geom.shape[0]), best_safe] > 1e-6).astype(np.float32),
        "best_soft_risk_increase": (risk_delta[np.arange(geom.shape[0]), best_soft] > 1e-6).astype(np.float32),
        "best_budget_risk_increase": (risk_delta[np.arange(geom.shape[0]), best_budget] > 1e-6).astype(np.float32),
        "geom_top1_safe": safe_mask[np.arange(geom.shape[0]), geom_top1].astype(np.float32),
        "best_safe_is_baseline": (best_safe == baseline).astype(np.float32),
        "best_safe_is_geom_top1": (best_safe == geom_top1).astype(np.float32),
    }
