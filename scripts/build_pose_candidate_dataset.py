"""
build_pose_candidate_dataset.py

Convert support-aligned stage-refiner shards into candidate-action scorer training data.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
from scipy.spatial.transform import Rotation


def parse_float_list_arg(raw: str) -> list[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def load_ready_thresholds_from_spec(
    spec_json: str | None,
    *,
    substage_id: int = 1,
    threshold_kind: str = "release",
) -> dict[str, float] | None:
    if not spec_json:
        return None
    spec_path = Path(spec_json)
    if not spec_path.exists():
        raise FileNotFoundError(f"ready spec json not found: {spec_json}")
    obj = json.loads(spec_path.read_text())
    for stage in obj.get("stages", []):
        if int(stage.get("substage_id", -1)) != int(substage_id):
            continue
        threshold_kind = str(threshold_kind).lower()
        if threshold_kind == "optimization":
            metrics = dict(stage.get("optimization_thresholds", stage.get("metric_thresholds", {})) or {})
        else:
            metrics = dict(stage.get("release_thresholds", stage.get("metric_thresholds", {})) or {})
        return {
            "xy_error": float(metrics.get("xy_error", -1.0)),
            "abs_z_error": float(metrics.get("abs_z_error", -1.0)),
            "yaw_error": float(metrics.get("yaw_error", -1.0)),
        }
    raise ValueError(f"no stage spec found in {spec_json} for substage_id={substage_id}")


def build_local_perturb_offsets(xy_values, z_values, yaw_values, include_diagonals=True):
    offsets = []
    seen = set()

    def _add(offset):
        arr = np.asarray(offset, dtype=np.float32).reshape(6)
        key = tuple(np.round(arr, 6).tolist())
        if key in seen:
            return
        seen.add(key)
        offsets.append(arr)

    for mag in xy_values:
        m = abs(float(mag))
        if m <= 0:
            continue
        _add([m, 0, 0, 0, 0, 0])
        _add([-m, 0, 0, 0, 0, 0])
        _add([0, m, 0, 0, 0, 0])
        _add([0, -m, 0, 0, 0, 0])
        if include_diagonals:
            _add([m, m, 0, 0, 0, 0])
            _add([m, -m, 0, 0, 0, 0])
            _add([-m, m, 0, 0, 0, 0])
            _add([-m, -m, 0, 0, 0, 0])
    for mag in z_values:
        m = abs(float(mag))
        if m <= 0:
            continue
        _add([0, 0, m, 0, 0, 0])
        _add([0, 0, -m, 0, 0, 0])
    for mag in yaw_values:
        m = abs(float(mag))
        if m <= 0:
            continue
        _add([0, 0, 0, 0, 0, m])
        _add([0, 0, 0, 0, 0, -m])
    return offsets


def build_action_primitives(
    xy_small: float = 0.004,
    xy_large: float = 0.008,
    xy_micro_values: list[float] | tuple[float, ...] = (),
    z_small: float = 0.004,
    yaw_small: float = 0.03,
    yaw_probe_values: list[float] | tuple[float, ...] = (),
    pitch_small: float = 0.06,
    roll_small: float = 0.06,
    include_descend: bool = True,
    include_combos: bool = True,
    include_tilt: bool = True,
):
    """Semantically structured local action primitives for near-field alignment."""
    offsets = []
    seen = set()

    def _add(offset):
        arr = np.asarray(offset, dtype=np.float32).reshape(6)
        key = tuple(np.round(arr, 6).tolist())
        if key in seen:
            return
        seen.add(key)
        offsets.append(arr)

    _add([0, 0, 0, 0, 0, 0])  # hold

    # Search in XY while holding height.
    for mag in [xy_small, xy_large]:
        _add([mag, 0, 0, 0, 0, 0])
        _add([-mag, 0, 0, 0, 0, 0])
        _add([0, mag, 0, 0, 0, 0])
        _add([0, -mag, 0, 0, 0, 0])
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            _add([sx * xy_small, sy * xy_small, 0, 0, 0, 0])

    for mag in xy_micro_values:
        m = abs(float(mag))
        if m <= 0.0:
            continue
        _add([m, 0, 0, 0, 0, 0])
        _add([-m, 0, 0, 0, 0, 0])
        _add([0, m, 0, 0, 0, 0])
        _add([0, -m, 0, 0, 0, 0])
        diag = float(m / np.sqrt(2.0))
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                _add([sx * diag, sy * diag, 0, 0, 0, 0])

    # Pure yaw alignment while holding position. Extra probe magnitudes let the
    # real candidate bank expose clockwise / counter-clockwise step-size
    # alternatives, which B2 can then learn to rank offline.
    yaw_values = [abs(float(yaw_small))]
    for mag in yaw_probe_values:
        m = abs(float(mag))
        if m > 0.0:
            yaw_values.append(m)
    for m in sorted(set(yaw_values)):
        _add([0, 0, 0, 0, 0, m])
        _add([0, 0, 0, 0, 0, -m])

    if include_tilt:
        _add([0, 0, 0, pitch_small, 0, 0])
        _add([0, 0, 0, -pitch_small, 0, 0])
        _add([0, 0, 0, 0, roll_small, 0])
        _add([0, 0, 0, 0, -roll_small, 0])

    if include_descend and z_small > 0:
        # Guarded vertical approach / relief primitives.
        _add([0, 0, z_small, 0, 0, 0])
        _add([0, 0, -z_small, 0, 0, 0])

    if include_combos:
        # Couple lateral search with yaw alignment to preserve approach structure.
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for syaw in (-1.0, 1.0):
                    _add([sx * xy_small, sy * xy_small, 0, 0, 0, syaw * yaw_small])
        # Slight descend only while already nudging in-plane.
        if include_descend and z_small > 0:
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    _add([sx * xy_small, sy * xy_small, z_small, 0, 0, 0])
        if include_tilt:
            for sx in (-1.0, 1.0):
                _add([sx * xy_small, 0, 0, pitch_small, 0, 0])
                _add([sx * xy_small, 0, 0, -pitch_small, 0, 0])
            for sy in (-1.0, 1.0):
                _add([0, sy * xy_small, 0, 0, roll_small, 0])
                _add([0, sy * xy_small, 0, 0, -roll_small, 0])

    return offsets


def build_orientation_rescue_primitives(
    pitch_small: float = 0.04,
    roll_small: float = 0.04,
    xy_small: float = 0.004,
    coupled_xy_tilt: bool = True,
):
    offsets = []
    seen = set()

    def _add(offset):
        arr = np.asarray(offset, dtype=np.float32).reshape(6)
        key = tuple(np.round(arr, 6).tolist())
        if key in seen:
            return
        seen.add(key)
        offsets.append(arr)

    _add([0, 0, 0, pitch_small, 0, 0])
    _add([0, 0, 0, -pitch_small, 0, 0])
    _add([0, 0, 0, 0, roll_small, 0])
    _add([0, 0, 0, 0, -roll_small, 0])
    if coupled_xy_tilt:
        for sx in (-1.0, 1.0):
            _add([sx * xy_small, 0, 0, pitch_small, 0, 0])
            _add([sx * xy_small, 0, 0, -pitch_small, 0, 0])
        for sy in (-1.0, 1.0):
            _add([0, sy * xy_small, 0, 0, roll_small, 0])
            _add([0, sy * xy_small, 0, 0, -roll_small, 0])
    return offsets


def world_delta_to_local(delta_world_6d: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    r_cur = Rotation.from_quat(np.asarray(current_quat, dtype=np.float32))
    out = np.asarray(delta_world_6d, dtype=np.float32).copy()
    out[:3] = r_cur.inv().apply(out[:3]).astype(np.float32)
    return out


def pose_delta_local_between(current_pose_7d: np.ndarray, target_pose_7d: np.ndarray) -> np.ndarray:
    current_pose_7d = np.asarray(current_pose_7d, dtype=np.float32)
    target_pose_7d = np.asarray(target_pose_7d, dtype=np.float32)
    delta_pos_world = target_pose_7d[:3] - current_pose_7d[:3]
    r_cur = Rotation.from_quat(current_pose_7d[3:7])
    r_tgt = Rotation.from_quat(target_pose_7d[3:7])
    delta_rot = (r_tgt * r_cur.inv()).as_rotvec().astype(np.float32)
    delta_pos_local = world_delta_to_local(
        np.concatenate([delta_pos_world.astype(np.float32), np.zeros(3, dtype=np.float32)], axis=0),
        current_pose_7d[3:7],
    )[:3]
    return np.concatenate([delta_pos_local.astype(np.float32), delta_rot.astype(np.float32)], axis=0)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-8 or bn < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def apply_local_offset_to_pose(pose_7d: np.ndarray, delta_local_6d: np.ndarray) -> np.ndarray:
    pose_7d = np.asarray(pose_7d, dtype=np.float32).copy()
    delta_local_6d = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    r_cur = Rotation.from_quat(pose_7d[3:7])
    pose_7d[:3] = pose_7d[:3] + r_cur.apply(delta_local_6d[:3]).astype(np.float32)
    r_delta = Rotation.from_rotvec(delta_local_6d[3:6].astype(np.float32))
    pose_7d[3:7] = (r_delta * r_cur).as_quat().astype(np.float32)
    return pose_7d


def compute_basin_metrics(delta_basin_target: np.ndarray, r_xy: float, r_z: float, r_yaw: float, r_tilt: float = 0.12):
    delta_arr = np.asarray(delta_basin_target, dtype=np.float32).reshape(-1)
    e_xy = float(np.linalg.norm(delta_arr[:2])) if delta_arr.size >= 2 else 0.0
    e_z = float(abs(delta_arr[2])) if delta_arr.size >= 3 else 0.0
    e_tilt = float(np.linalg.norm(delta_arr[3:5])) if delta_arr.size >= 5 else 0.0
    e_yaw = float(abs(delta_arr[5])) if delta_arr.size >= 6 else 0.0
    use_tilt = bool(np.isfinite(float(r_tilt)) and float(r_tilt) > 0.0)
    basin_distance = max(
        e_xy / max(float(r_xy), 1e-6),
        e_z / max(float(r_z), 1e-6),
        (e_tilt / max(float(r_tilt), 1e-6)) if use_tilt else 0.0,
        e_yaw / max(float(r_yaw), 1e-6),
    )
    return basin_distance, e_xy, e_z, e_yaw, e_tilt


def compute_funnel_cost(
    delta_basin_target: np.ndarray,
    reference_axis_local: np.ndarray | None,
    r_xy: float,
    r_z: float,
    r_yaw: float,
    r_tilt: float = 0.12,
    far_z_scale: float = 3.0,
    xy_near_gain: float = 4.0,
    yaw_near_gain: float = 3.0,
    tilt_near_gain: float = 2.5,
    axis_lateral_weight: float = 0.75,
    axis_reverse_weight: float = 0.35,
) -> tuple[float, dict]:
    delta_arr = np.asarray(delta_basin_target, dtype=np.float32).reshape(-1)
    axis_arr = np.asarray(reference_axis_local, dtype=np.float32).reshape(-1) if reference_axis_local is not None else None
    basin_distance, e_xy, e_z, e_yaw, e_tilt = compute_basin_metrics(delta_arr, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw, r_tilt=r_tilt)
    use_tilt = bool(np.isfinite(float(r_tilt)) and float(r_tilt) > 0.0)

    z_ratio = float(np.clip(e_z / max(float(far_z_scale) * float(r_z), 1e-6), 0.0, 1.0))
    z_closeness = float(1.0 - z_ratio)
    xy_weight = float(1.0 + float(xy_near_gain) * (z_closeness ** 2))
    yaw_weight = float(1.0 + float(yaw_near_gain) * (z_closeness ** 2))
    tilt_weight = float(1.0 + float(tilt_near_gain) * (z_closeness ** 2)) if use_tilt else 0.0
    z_weight = float(0.35 + 0.65 * z_ratio)

    lateral_penalty = 0.0
    reverse_penalty = 0.0
    along_axis = 0.0
    axis_norm = 0.0
    if axis_arr is not None and axis_arr.shape[0] >= 2:
        axis_xy = axis_arr[:2]
        axis_norm = float(np.linalg.norm(axis_xy))
        if axis_norm > 1e-6:
            unit = axis_xy / axis_norm
            perp = np.asarray([-unit[1], unit[0]], dtype=np.float32)
            along_axis = float(np.dot(delta_arr[:2], unit))
            lateral_error = float(np.dot(delta_arr[:2], perp))
            lateral_penalty = ((0.25 + axis_lateral_weight * z_closeness) * (lateral_error / max(float(r_xy), 1e-6)) ** 2)
            reverse_penalty = (axis_reverse_weight * (max(0.0, -along_axis) / max(float(r_xy), 1e-6)) ** 2)

    cost = (
        xy_weight * (e_xy / max(float(r_xy), 1e-6)) ** 2
        + yaw_weight * (e_yaw / max(float(r_yaw), 1e-6)) ** 2
        + (tilt_weight * (e_tilt / max(float(r_tilt), 1e-6)) ** 2 if use_tilt else 0.0)
        + z_weight * (e_z / max(float(r_z), 1e-6))
        + lateral_penalty
        + reverse_penalty
    )
    details = {
        "basin_distance": float(basin_distance),
        "e_xy": float(e_xy),
        "e_z": float(e_z),
        "e_yaw": float(e_yaw),
        "e_tilt": float(e_tilt),
        "z_ratio": float(z_ratio),
        "z_closeness": float(z_closeness),
        "xy_weight": float(xy_weight),
        "yaw_weight": float(yaw_weight),
        "tilt_weight": float(tilt_weight),
        "z_weight": float(z_weight),
        "axis_norm": float(axis_norm),
        "along_axis": float(along_axis),
        "lateral_penalty": float(lateral_penalty),
        "reverse_penalty": float(reverse_penalty),
        "cost": float(cost),
    }
    return float(cost), details


def score_candidate_stage_handoff_joint(
    *,
    current_delta: np.ndarray,
    next_delta: np.ndarray,
    candidate_local: np.ndarray,
    base_action_local: np.ndarray | None,
    reference_anchor_local: np.ndarray | None,
    xy_threshold: float,
    abs_z_threshold: float,
    yaw_threshold: float,
) -> tuple[float, dict]:
    """Near-ready local objective that teaches XY and yaw to converge together.

    The legacy funnel labels are good for coarse attraction, but in the last few
    millimeters they treat XY, Z, and yaw too independently. This objective
    explicitly stages the correction:
    1. center XY
    2. descend while preserving XY
    3. polish yaw only once XY/Z are already close
    """
    delta_cur = np.asarray(current_delta, dtype=np.float32).reshape(-1)
    delta_next = np.asarray(next_delta, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate_local, dtype=np.float32).reshape(-1)
    base = None if base_action_local is None else np.asarray(base_action_local, dtype=np.float32).reshape(-1)
    ref = None if reference_anchor_local is None else np.asarray(reference_anchor_local, dtype=np.float32).reshape(-1)

    cur_xy = float(np.linalg.norm(delta_cur[:2])) if delta_cur.size >= 2 else 0.0
    cur_z = float(abs(delta_cur[2])) if delta_cur.size >= 3 else 0.0
    cur_yaw = float(abs(delta_cur[5])) if delta_cur.size >= 6 else 0.0
    next_xy = float(np.linalg.norm(delta_next[:2])) if delta_next.size >= 2 else 0.0
    next_z = float(abs(delta_next[2])) if delta_next.size >= 3 else 0.0
    next_yaw = float(abs(delta_next[5])) if delta_next.size >= 6 else 0.0

    xy_thr = max(float(xy_threshold), 1e-6)
    z_thr = max(float(abs_z_threshold), 1e-6)
    use_yaw = bool(np.isfinite(float(yaw_threshold)) and float(yaw_threshold) > 0.0)
    yaw_thr = max(float(yaw_threshold), 1e-6) if use_yaw else 0.12

    lateral_mag = float(np.linalg.norm(cand[:2])) if cand.size >= 2 else 0.0
    vertical_mag = float(abs(cand[2])) if cand.size >= 3 else 0.0
    yaw_mag = float(abs(cand[5])) if cand.size >= 6 else 0.0
    cand_norm = float(np.linalg.norm(cand))

    xy_improve = (cur_xy - next_xy) / xy_thr
    z_improve = (cur_z - next_z) / z_thr
    yaw_improve = ((cur_yaw - next_yaw) / yaw_thr) if use_yaw else 0.0
    current_joint = max(cur_xy / xy_thr, cur_z / z_thr, (cur_yaw / yaw_thr) if use_yaw else 0.0)
    next_joint = max(next_xy / xy_thr, next_z / z_thr, (next_yaw / yaw_thr) if use_yaw else 0.0)

    if cur_xy > max(1.5 * xy_thr, 0.010):
        phase = "search_xy"
    elif cur_z > max(2.0 * z_thr, 0.006):
        phase = "guarded_descend"
    elif use_yaw and cur_yaw > max(1.10 * yaw_thr, 0.08) and cur_xy <= max(2.0 * xy_thr, 0.010) and cur_z <= max(2.5 * z_thr, 0.006):
        phase = "yaw_refine"
    else:
        phase = "preclose_commit"

    score = 3.0 * (current_joint - next_joint)

    if phase == "search_xy":
        score += 12.0 * xy_improve
        score += 0.35 * yaw_improve
        if vertical_mag > 1e-8:
            score -= 8.0 * vertical_mag / z_thr
        if next_xy <= cur_xy:
            score += 1.25 * lateral_mag / xy_thr
        else:
            score -= 4.0 * lateral_mag / xy_thr
        if yaw_mag > 1e-8 and lateral_mag < 1e-8:
            score -= 1.5 * yaw_mag / max(yaw_thr, 0.08)
    elif phase == "guarded_descend":
        score += 8.0 * xy_improve
        score += 5.5 * z_improve
        score += 0.45 * yaw_improve
        if next_xy > max(cur_xy + 1e-4, 1.25 * xy_thr):
            score -= 5.0 * (next_xy - min(cur_xy, 1.25 * xy_thr)) / xy_thr
        if vertical_mag > 1e-8 and next_xy <= max(1.10 * xy_thr, cur_xy):
            score += 0.6
        if yaw_mag > 1e-8 and lateral_mag < 1e-8 and vertical_mag < 1e-8:
            score -= 1.0 * yaw_mag / max(yaw_thr, 0.08)
    elif phase == "yaw_refine":
        score += 5.0 * xy_improve
        score += 1.0 * z_improve
        score += 6.5 * yaw_improve
        if next_xy > cur_xy + 1e-4:
            score -= 4.5 * lateral_mag / xy_thr
        if vertical_mag > 1e-8:
            score -= 4.0 * vertical_mag / z_thr
        if yaw_mag > 1e-8 and lateral_mag < 1e-8:
            if next_yaw < cur_yaw and next_xy <= max(1.25 * xy_thr, cur_xy + 1e-4):
                score += 0.5
            else:
                score -= 1.25 * yaw_mag / max(yaw_thr, 0.08)
    else:
        score += 4.0 * xy_improve
        score += 2.5 * z_improve
        score += 2.5 * yaw_improve
        if next_xy > 1.15 * xy_thr:
            score -= 6.0 * (next_xy - 1.15 * xy_thr) / xy_thr
        if next_z > cur_z + 1e-4:
            score -= 2.5 * (next_z - cur_z) / z_thr
        if yaw_mag > 1e-8 and lateral_mag < 1e-8 and vertical_mag < 1e-8:
            score -= 1.0 * yaw_mag / max(yaw_thr, 0.08)

    if lateral_mag > 1e-8 and yaw_mag > 1e-8 and next_xy <= cur_xy and next_yaw <= cur_yaw:
        score += 0.8

    if next_xy <= 1.25 * xy_thr and next_z <= 1.25 * z_thr and (not use_yaw or next_yaw <= 1.5 * yaw_thr):
        score += 2.0
    if next_xy <= xy_thr and next_z <= z_thr and (not use_yaw or next_yaw <= yaw_thr):
        score += 4.0

    if cand_norm < 1e-8:
        if cur_xy <= 1.1 * xy_thr and cur_z <= 1.1 * z_thr and (not use_yaw or cur_yaw <= 1.1 * yaw_thr):
            score += 0.75
        else:
            score -= 0.5

    if base is not None and base.size >= 3:
        score += 0.15 * safe_cosine(cand[:3], base[:3])
    if ref is not None and ref.size >= 2:
        score += 0.10 * safe_cosine(cand[:2], ref[:2])

    return float(score), {
        "phase": phase,
        "current_xy": float(cur_xy),
        "current_abs_z": float(cur_z),
        "current_yaw": float(cur_yaw),
        "next_xy": float(next_xy),
        "next_abs_z": float(next_z),
        "next_yaw": float(next_yaw),
    }


def score_candidate_approach_funnel(
    current_pose_7d: np.ndarray,
    next_pose_7d: np.ndarray,
    current_delta: np.ndarray,
    next_delta: np.ndarray,
    candidate_local: np.ndarray,
    reference_anchor_pose_7d: np.ndarray | None = None,
    base_action_local: np.ndarray | None = None,
    depth_proximity: float | None = None,
    r_xy: float = 0.008,
    r_z: float = 0.01,
    r_yaw: float = 0.05,
    r_tilt: float = 0.12,
    far_z_scale: float = 3.0,
    descend_guard_xy_scale: float = 1.25,
    descend_guard_yaw_scale: float = 1.25,
) -> tuple[float, dict]:
    """Score a candidate by an approach-funnel objective instead of point-distance greed."""
    cand = np.asarray(candidate_local, dtype=np.float32).reshape(6)
    current_pose = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
    next_pose = np.asarray(next_pose_7d, dtype=np.float32).reshape(7)
    anchor_pose = np.asarray(reference_anchor_pose_7d, dtype=np.float32).reshape(7) if reference_anchor_pose_7d is not None else None

    axis_current = pose_delta_local_between(current_pose, anchor_pose) if anchor_pose is not None else current_delta
    axis_next = pose_delta_local_between(next_pose, anchor_pose) if anchor_pose is not None else next_delta
    cur_cost, cur_cost_details = compute_funnel_cost(
        current_delta,
        axis_current,
        r_xy=r_xy,
        r_z=r_z,
        r_yaw=r_yaw,
        r_tilt=r_tilt,
        far_z_scale=far_z_scale,
    )
    next_cost, next_cost_details = compute_funnel_cost(
        next_delta,
        axis_next,
        r_xy=r_xy,
        r_z=r_z,
        r_yaw=r_yaw,
        r_tilt=r_tilt,
        far_z_scale=far_z_scale,
    )
    cur_dist, cur_xy, cur_z, cur_yaw, cur_tilt = compute_basin_metrics(current_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw, r_tilt=r_tilt)
    next_dist, next_xy, next_z, next_yaw, next_tilt = compute_basin_metrics(next_delta, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw, r_tilt=r_tilt)

    xy_gain = (cur_xy - next_xy) / max(r_xy, 1e-6)
    yaw_gain = (cur_yaw - next_yaw) / max(r_yaw, 1e-6)
    use_tilt = bool(np.isfinite(float(r_tilt)) and float(r_tilt) > 0.0)
    tilt_gain = ((cur_tilt - next_tilt) / max(r_tilt, 1e-6)) if use_tilt else 0.0
    z_gain = (cur_z - next_z) / max(r_z, 1e-6)
    score = float(cur_cost - next_cost)
    details = {
        "cur_cost": float(cur_cost),
        "next_cost": float(next_cost),
        "cur_dist": float(cur_dist),
        "next_dist": float(next_dist),
        "xy_gain": float(xy_gain),
        "yaw_gain": float(yaw_gain),
        "tilt_gain": float(tilt_gain),
        "z_gain": float(z_gain),
        "cur_cost_terms": cur_cost_details,
        "next_cost_terms": next_cost_details,
    }

    # Prefer lateral corrections that point toward the reference approach axis, not just the point target.
    score += 0.30 * safe_cosine(cand[:2], axis_current[:2])
    score += 0.20 * safe_cosine(cand[:2], current_delta[:2])

    if abs(float(cand[5])) > 1e-6 and abs(float(current_delta[5])) > 1e-4:
        score += 0.20 * float(np.sign(float(cand[5]) * float(current_delta[5])))
    if use_tilt and np.linalg.norm(cand[3:5]) > 1e-6 and np.linalg.norm(current_delta[3:5]) > 1e-4:
        score += 0.25 * safe_cosine(cand[3:5], current_delta[3:5]) + 0.40 * tilt_gain

    # Guard vertical motion: only reward it after XY/yaw are sufficiently aligned.
    has_vertical = abs(float(cand[2])) > 1e-6
    z_ratio = float(cur_cost_details["z_ratio"])
    descend_guard_xy = float(r_xy * (descend_guard_xy_scale + 0.50 * z_ratio))
    descend_guard_yaw = float(r_yaw * (descend_guard_yaw_scale + 0.50 * z_ratio))
    descend_allowed = bool(next_xy <= descend_guard_xy and next_yaw <= descend_guard_yaw)
    z_dir_match = bool(abs(float(current_delta[2])) < 1e-6 or np.sign(float(cand[2])) == np.sign(float(current_delta[2])))
    details["descend_allowed"] = descend_allowed
    details["z_dir_match"] = z_dir_match
    details["descend_guard_xy"] = float(descend_guard_xy)
    details["descend_guard_yaw"] = float(descend_guard_yaw)
    if has_vertical:
        if descend_allowed and z_dir_match:
            score += 0.75 * z_gain
        else:
            score -= 1.50 * abs(float(cand[2])) / max(r_z, 1e-6)

    # Keep low-level motion roughly aligned with the planner's coarse local intent.
    if base_action_local is not None:
        base = np.asarray(base_action_local, dtype=np.float32).reshape(6)
        score += 0.30 * safe_cosine(cand[:3], base[:3])
        if abs(float(cand[5])) > 1e-6 and abs(float(base[5])) > 1e-6:
            score += 0.10 * float(np.sign(float(cand[5]) * float(base[5])))

    # Depth-derived collision / premature descent guard.
    if depth_proximity is not None and np.isfinite(depth_proximity) and has_vertical:
        if float(depth_proximity) < 0.02 and not descend_allowed:
            score -= 0.75

    if next_xy <= r_xy and next_yaw <= r_yaw:
        score += 0.25
    if next_dist <= 1.0:
        score += 0.5

    details["score"] = float(score)
    return float(score), details


def load_concat_npz(data_dir: Path) -> dict:
    if data_dir.is_file() and data_dir.suffix == ".npz":
        shard = np.load(data_dir)
        return {k: shard[k] for k in shard.files}
    shards = sorted(data_dir.glob("residual_shard_*.npz"))
    if not shards:
        support_npz = data_dir / "support_states.npz"
        if support_npz.exists():
            shard = np.load(support_npz)
            return {k: shard[k] for k in shard.files}
    if not shards:
        raise FileNotFoundError(f"No residual shards found in {data_dir}")
    raw = {}
    for shard_path in shards:
        shard = np.load(shard_path)
        for key in shard.files:
            raw.setdefault(key, []).append(shard[key])
    return {k: np.concatenate(v, axis=0) for k, v in raw.items()}


def infer_episode_ids(step_idx: np.ndarray) -> np.ndarray:
    step_idx = np.asarray(step_idx, dtype=np.int64).reshape(-1)
    if step_idx.size == 0:
        return np.zeros((0,), dtype=np.int64)
    episode_ids = np.zeros((step_idx.size,), dtype=np.int64)
    cur_ep = 0
    for i in range(1, step_idx.size):
        if int(step_idx[i]) <= int(step_idx[i - 1]):
            cur_ep += 1
        episode_ids[i] = cur_ep
    return episode_ids


def build_ready_targets_from_events(
    raw: dict,
    *,
    open_threshold: float,
    positive_window: int,
    hard_negative_gap: int,
    xy_threshold: float,
    abs_z_threshold: float,
    yaw_threshold: float,
    basin_distance_threshold: float,
) -> tuple[np.ndarray, dict]:
    step_idx = np.asarray(raw.get("step_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    phase_id = np.asarray(raw.get("phase_id", np.zeros_like(step_idx)), dtype=np.int64)
    rollout_gripper_open = np.asarray(
        raw.get("rollout_gripper_open", np.ones((step_idx.shape[0],), dtype=np.float32)),
        dtype=np.float32,
    )
    ready_field = np.asarray(raw.get("ready_to_close", np.zeros((step_idx.shape[0],), dtype=np.float32)), dtype=np.float32)
    stable_field = np.asarray(
        raw.get("post_close_stability_proxy", np.zeros((step_idx.shape[0],), dtype=np.float32)),
        dtype=np.float32,
    )
    lift_field = np.asarray(raw.get("grasp_lift_proxy", np.zeros((step_idx.shape[0],), dtype=np.float32)), dtype=np.float32)
    reopen_field = np.asarray(
        raw.get("reopen_after_trigger", raw.get("reopen_within_horizon", np.zeros((step_idx.shape[0],), dtype=np.float32))),
        dtype=np.float32,
    )
    invalid_field = np.asarray(raw.get("invalid_after_trigger", np.zeros((step_idx.shape[0],), dtype=np.float32)), dtype=np.float32)
    early_close_field = np.asarray(raw.get("planner_close_too_early", np.zeros((step_idx.shape[0],), dtype=np.float32)), dtype=np.float32)
    if "current_pose_7d" in raw and "basin_center_pose_7d" in raw:
        current_delta = np.stack(
            [
                pose_delta_local_between(raw["current_pose_7d"][i], raw["basin_center_pose_7d"][i])
                for i in range(step_idx.shape[0])
            ],
            axis=0,
        ).astype(np.float32)
    elif "delta_basin_target" in raw:
        current_delta = np.asarray(raw["delta_basin_target"], dtype=np.float32)
    else:
        current_delta = np.zeros((step_idx.shape[0], 6), dtype=np.float32)
    current_xy = np.linalg.norm(current_delta[:, :2], axis=1).astype(np.float32)
    current_abs_z = np.abs(current_delta[:, 2]).astype(np.float32)
    current_yaw = np.abs(current_delta[:, 5]).astype(np.float32)
    if "basin_distance" in raw:
        current_basin_distance = np.asarray(raw["basin_distance"], dtype=np.float32).reshape(-1)
    else:
        current_basin_distance = np.maximum.reduce(
            [
                current_xy / max(float(xy_threshold), 1e-6),
                current_abs_z / max(float(abs_z_threshold), 1e-6),
                current_yaw / max(float(yaw_threshold if yaw_threshold >= 0.0 else 0.05), 1e-6),
            ]
        ).astype(np.float32)
    episode_ids = infer_episode_ids(step_idx)
    ready_targets = np.zeros((step_idx.shape[0],), dtype=np.float32)
    anchor_kind_hist = {"stable": 0, "ready_last": 0, "episode_end": 0}
    hard_negative_count = 0
    anchor_deltas = []
    band_eligible_count = 0

    for ep_id in np.unique(episode_ids):
        ep_idx = np.where(episode_ids == ep_id)[0]
        if ep_idx.size == 0:
            continue
        align_open_mask = (
            (phase_id[ep_idx] == 1)
            & (rollout_gripper_open[ep_idx] >= float(open_threshold))
        )
        if not np.any(align_open_mask):
            continue
        stable_candidates = np.where(align_open_mask & ((stable_field[ep_idx] > 0.5) | (lift_field[ep_idx] > 0.5)))[0]
        if stable_candidates.size > 0:
            anchor_local = int(stable_candidates[0])
            anchor_kind = "stable"
        else:
            ready_candidates = np.where(align_open_mask & (ready_field[ep_idx] > 0.5))[0]
            if ready_candidates.size > 0:
                anchor_local = int(ready_candidates[-1])
                anchor_kind = "ready_last"
            else:
                anchor_local = int(np.where(align_open_mask)[0][-1])
                anchor_kind = "episode_end"
        anchor_kind_hist[anchor_kind] += 1
        bad_mask = (
            (early_close_field[ep_idx] > 0.5)
            | (reopen_field[ep_idx] > 0.5)
            | (invalid_field[ep_idx] > 0.5)
        )
        for local_pos, src_i in enumerate(ep_idx.tolist()):
            if not bool(align_open_mask[local_pos]):
                continue
            delta = int(anchor_local - local_pos)
            anchor_deltas.append(delta)
            band_ready = bool(
                (
                    float(basin_distance_threshold) < 0.0
                    or current_basin_distance[src_i] <= float(basin_distance_threshold)
                )
                and current_xy[src_i] <= float(xy_threshold)
                and current_abs_z[src_i] <= float(abs_z_threshold)
                and (
                    float(yaw_threshold) < 0.0
                    or current_yaw[src_i] <= float(yaw_threshold)
                )
            )
            band_eligible_count += int(band_ready)
            is_hard_negative = bool(bad_mask[local_pos] or delta > int(hard_negative_gap))
            hard_negative_count += int(is_hard_negative)
            ready_targets[src_i] = float(
                band_ready
                and (0 <= delta <= int(positive_window))
                and (not is_hard_negative)
            )

    meta = {
        "mode": "event_mined",
        "positive_window": int(positive_window),
        "hard_negative_gap": int(hard_negative_gap),
        "positive_count": int(np.sum(ready_targets > 0.5)),
        "positive_rate": float(np.mean(ready_targets > 0.5)) if ready_targets.size > 0 else 0.0,
        "hard_negative_count": int(hard_negative_count),
        "band_eligible_count": int(band_eligible_count),
        "xy_threshold": float(xy_threshold),
        "abs_z_threshold": float(abs_z_threshold),
        "yaw_threshold": float(yaw_threshold),
        "basin_distance_threshold": float(basin_distance_threshold),
        "anchor_kind_hist": anchor_kind_hist,
        "anchor_delta_stats": {
            "min": int(np.min(anchor_deltas)) if anchor_deltas else None,
            "max": int(np.max(anchor_deltas)) if anchor_deltas else None,
            "mean": float(np.mean(anchor_deltas)) if anchor_deltas else None,
        },
    }
    return ready_targets, meta


def candidate_offsets(args) -> np.ndarray:
    if getattr(args, "candidate_mode", "grid") == "primitives":
        return np.stack(
            build_action_primitives(
                xy_small=float(getattr(args, "primitive_xy_small", 0.004)),
                xy_large=float(getattr(args, "primitive_xy_large", 0.008)),
                xy_micro_values=parse_float_list_arg(getattr(args, "primitive_xy_micro_values", "")),
                z_small=float(getattr(args, "primitive_z_small", 0.004)),
                yaw_small=float(getattr(args, "primitive_yaw_small", 0.03)),
                yaw_probe_values=parse_float_list_arg(getattr(args, "primitive_yaw_probe_values", "")),
                pitch_small=float(getattr(args, "primitive_pitch_small", 0.06)),
                roll_small=float(getattr(args, "primitive_roll_small", 0.06)),
                include_descend=bool(getattr(args, "primitive_include_descend", True)),
                include_combos=bool(getattr(args, "primitive_include_combos", True)),
                include_tilt=bool(getattr(args, "primitive_include_tilt", True)),
            ),
            axis=0,
        ).astype(np.float32)
    offsets = [np.zeros(6, dtype=np.float32)]
    offsets.extend(
        build_local_perturb_offsets(
            parse_float_list_arg(args.candidate_xy_values),
            parse_float_list_arg(args.candidate_z_values),
            parse_float_list_arg(args.candidate_yaw_values),
            include_diagonals=bool(args.candidate_include_diagonals),
        )
    )
    return np.stack(offsets, axis=0).astype(np.float32)


def candidate_kind_from_actions(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    kinds = []
    for action in arr:
        is_rescue = bool(np.linalg.norm(action[3:5]) > 1e-8)
        kinds.append("rescue" if is_rescue else "base")
    return np.asarray(kinds)


def improvement_tiers(improve_row: np.ndarray, basin_positive_row: np.ndarray) -> np.ndarray:
    arr = np.asarray(improve_row, dtype=np.float32)
    tiers = np.zeros_like(arr, dtype=np.int64)
    if arr.size == 0:
        return tiers
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    best_val = float(np.max(arr))
    tiers[arr >= p50] = 1
    tiers[arr >= p75] = 2
    tiers[(arr >= best_val - 1e-6) | (np.asarray(basin_positive_row, dtype=np.float32) > 0.5)] = 3
    return tiers


def basin_distance_bin(dist: float) -> int:
    if dist <= 0.9:
        return 0
    if dist <= 1.05:
        return 1
    if dist <= 1.2:
        return 2
    return 3


def sign_bucket(val: float, eps: float) -> int:
    if val > eps:
        return 1
    if val < -eps:
        return -1
    return 0


def candidate_group_key(action_local: np.ndarray) -> tuple[int, int, int, int, int, int]:
    arr = np.asarray(action_local, dtype=np.float32).reshape(6)
    return (
        sign_bucket(float(arr[0]), 1e-5),
        sign_bucket(float(arr[1]), 1e-5),
        sign_bucket(float(arr[2]), 1e-5),
        sign_bucket(float(arr[3]), 1e-5),
        sign_bucket(float(arr[4]), 1e-5),
        sign_bucket(float(arr[5]), 1e-5),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--open_threshold", type=float, default=0.5)
    parser.add_argument(
        "--support_close_intent_mode",
        type=str,
        default="all_open",
        choices=["required", "all_open", "no_close"],
        help="Support filter for planner close intent. all_open is the phase-1 default: keep ALIGN+open states, then let the oracle decide whether to hold or correct.",
    )
    parser.add_argument("--exclude_occluded", action="store_true", default=False)
    parser.add_argument("--early_depth_median_threshold", type=float, default=0.04)
    parser.add_argument("--candidate_xy_values", type=str, default="0.004,0.008")
    parser.add_argument("--candidate_z_values", type=str, default="0.0")
    parser.add_argument("--candidate_yaw_values", type=str, default="0.03,0.05")
    parser.add_argument("--candidate_include_diagonals", action="store_true", default=True)
    parser.add_argument("--no_candidate_include_diagonals", dest="candidate_include_diagonals", action="store_false")
    parser.add_argument("--candidate_mode", type=str, default="primitives", choices=["grid", "primitives"])
    parser.add_argument(
        "--force_rebuild_candidate_bank",
        action="store_true",
        default=False,
        help="Ignore candidate_actions_local stored in support rows and rebuild the bank from CLI primitive/grid args.",
    )
    parser.add_argument("--primitive_xy_small", type=float, default=0.004)
    parser.add_argument("--primitive_xy_large", type=float, default=0.008)
    parser.add_argument("--primitive_xy_micro_values", type=str, default="0.001,0.0015,0.002,0.003")
    parser.add_argument("--primitive_z_small", type=float, default=0.004)
    parser.add_argument("--primitive_yaw_small", type=float, default=0.03)
    parser.add_argument(
        "--primitive_yaw_probe_values",
        type=str,
        default="",
        help="Optional extra pure-yaw probe magnitudes, e.g. '0.06,0.12'.",
    )
    parser.add_argument("--primitive_pitch_small", type=float, default=0.06)
    parser.add_argument("--primitive_roll_small", type=float, default=0.06)
    parser.add_argument("--primitive_include_descend", action="store_true", default=True)
    parser.add_argument("--no_primitive_include_descend", dest="primitive_include_descend", action="store_false")
    parser.add_argument("--primitive_include_combos", action="store_true", default=True)
    parser.add_argument("--no_primitive_include_combos", dest="primitive_include_combos", action="store_false")
    parser.add_argument("--primitive_include_tilt", action="store_true", default=False)
    parser.add_argument("--no_primitive_include_tilt", dest="primitive_include_tilt", action="store_false")
    parser.add_argument("--oracle_mode", type=str, default="short_horizon_funnel", choices=["basin_distance", "approach_funnel", "short_horizon_funnel", "stage_handoff_joint"])
    parser.add_argument(
        "--recompute_oracle_labels",
        action="store_true",
        default=False,
        help="Ignore candidate_oracle_score stored in support rows and rebuild labels from the current objective.",
    )
    parser.add_argument("--basin_radius_xy", type=float, default=0.008)
    parser.add_argument("--basin_radius_z", type=float, default=0.01)
    parser.add_argument("--basin_radius_yaw", type=float, default=0.05)
    parser.add_argument(
        "--basin_radius_tilt",
        type=float,
        default=-1.0,
        help="If <=0, ignore pitch/roll error in the oracle basin metric. This is the phase-1 stable base-set default.",
    )
    parser.add_argument("--horizon_k", type=int, default=4)
    parser.add_argument("--discount_gamma", type=float, default=0.9)
    parser.add_argument("--funnel_axis", type=str, default="demo_anchor_axis", choices=["demo_anchor_axis"])
    parser.add_argument(
        "--no_intent_hold_basin_distance",
        type=float,
        default=1.2,
        help="When gripper is open and planner has no close intent, prefer follow-planner/no-op if already in or near the basin.",
    )
    parser.add_argument(
        "--no_intent_hold_abs_z_threshold",
        type=float,
        default=0.025,
        help="When planner has no close intent and the pre-contact z gap is still large, prefer following the planner instead of early lateral takeover.",
    )
    parser.add_argument("--no_intent_noop_bonus", type=float, default=8.0)
    parser.add_argument("--no_intent_motion_penalty", type=float, default=4.0)
    parser.add_argument("--support_min_abs_z", type=float, default=None)
    parser.add_argument("--support_max_abs_z", type=float, default=None)
    parser.add_argument("--support_min_depth_median", type=float, default=None)
    parser.add_argument("--support_max_depth_median", type=float, default=None)
    parser.add_argument("--mid_abs_z_threshold", type=float, default=0.01)
    parser.add_argument(
        "--ready_label_mode",
        type=str,
        default="event_mined",
        choices=["geometry", "event_mined", "provider_handoff", "teacher_ready_or_handoff"],
    )
    parser.add_argument("--ready_xy_threshold", type=float, default=0.009)
    parser.add_argument("--ready_abs_z_threshold", type=float, default=0.040)
    parser.add_argument("--ready_yaw_threshold", type=float, default=-1.0)
    parser.add_argument("--ready_basin_distance_threshold", type=float, default=-1.0)
    parser.add_argument("--ready_row_xy_threshold", type=float, default=None)
    parser.add_argument("--ready_row_abs_z_threshold", type=float, default=None)
    parser.add_argument("--ready_row_yaw_threshold", type=float, default=None)
    parser.add_argument("--ready_row_basin_distance_threshold", type=float, default=None)
    parser.add_argument("--ready_spec_json", type=str, default=None)
    parser.add_argument("--ready_spec_substage_id", type=int, default=1)
    parser.add_argument("--oracle_spec_json", type=str, default=None)
    parser.add_argument("--oracle_spec_substage_id", type=int, default=1)
    parser.add_argument("--ready_positive_window", type=int, default=1)
    parser.add_argument("--ready_hard_negative_gap", type=int, default=4)
    parser.add_argument(
        "--phase1_truncate_to_first_success",
        action="store_true",
        default=False,
        help="Keep only rows up to the first qualified phase-1 grasp for each episode.",
    )
    parser.add_argument(
        "--phase1_drop_weak_success_episodes",
        action="store_true",
        default=False,
        help="Drop episodes whose first grasp/lift proxy is geometrically weak.",
    )
    parser.add_argument("--phase1_success_xy_threshold", type=float, default=0.008)
    parser.add_argument("--phase1_success_abs_z_threshold", type=float, default=0.008)
    parser.add_argument("--phase1_success_yaw_threshold", type=float, default=0.20)
    parser.add_argument("--phase1_success_tilt_threshold", type=float, default=-1.0)
    parser.add_argument("--yaw_focus_xy_multiplier", type=float, default=2.0)
    parser.add_argument("--yaw_focus_abs_z_multiplier", type=float, default=2.5)
    parser.add_argument("--yaw_hard_negative_weight", type=float, default=3.0)
    parser.add_argument("--yaw_hard_positive_weight", type=float, default=2.0)
    parser.add_argument("--near_ready_sample_weight", type=float, default=1.5)
    parser.add_argument("--xy_focus_sample_weight", type=float, default=3.0)
    parser.add_argument("--xy_focus_xy_min_multiplier", type=float, default=1.0)
    parser.add_argument("--xy_focus_xy_max_multiplier", type=float, default=3.0)
    parser.add_argument("--xy_focus_abs_z_multiplier", type=float, default=1.5)
    parser.add_argument("--xy_focus_yaw_multiplier", type=float, default=1.5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_concat_npz(input_dir)
    prefilter_num_states = int(raw["current_pose_7d"].shape[0])
    occluded_prefilter_count = int(np.sum(raw["is_occluded"] > 0.5)) if "is_occluded" in raw else 0
    if args.exclude_occluded and "is_occluded" in raw:
        keep = np.asarray(raw["is_occluded"], dtype=np.float32).reshape(-1) <= 0.5
        num_rows = int(keep.shape[0])
        filtered = {}
        for k, v in raw.items():
            arr = np.asarray(v)
            if arr.ndim > 0 and int(arr.shape[0]) == num_rows:
                filtered[k] = arr[keep]
            else:
                filtered[k] = arr
        raw = filtered
    cur_pose = raw["current_pose_7d"].astype(np.float32)
    basin_center = raw["basin_center_pose_7d"].astype(np.float32)
    rollout_gripper_open = raw.get("rollout_gripper_open", np.ones((cur_pose.shape[0],), dtype=np.float32)).astype(np.float32)
    phase_id = raw.get("phase_id", np.zeros((cur_pose.shape[0],), dtype=np.int64)).astype(np.int64)
    planner_close_intent = raw.get("planner_close_intent", np.ones((cur_pose.shape[0],), dtype=np.float32)).astype(np.float32)
    is_augmented = raw.get("is_augmented", np.zeros((cur_pose.shape[0],), dtype=np.int64)).astype(np.int64)
    phase1_keep_mask = np.ones((cur_pose.shape[0],), dtype=bool)
    phase1_good_episode = {}
    phase1_first_success_step = {}
    phase1_no_success_episodes = []
    if bool(args.phase1_truncate_to_first_success):
        ep_ids = raw.get("episode_index", np.zeros((cur_pose.shape[0],), dtype=np.int64)).astype(np.int64)
        rollout_steps = raw.get("rollout_step", np.arange(cur_pose.shape[0], dtype=np.int64)).astype(np.int64)
        ready_raw = raw.get("ready_to_close_target", np.zeros((cur_pose.shape[0],), dtype=np.float32)).astype(np.float32)
        verified_raw = raw.get("grasp_verified_target", np.zeros((cur_pose.shape[0],), dtype=np.float32)).astype(np.float32)
        xy_raw = raw.get("handoff_metric_xy_error", np.full((cur_pose.shape[0],), np.nan, dtype=np.float32)).astype(np.float32)
        z_raw = raw.get("handoff_metric_abs_z_error", np.full((cur_pose.shape[0],), np.nan, dtype=np.float32)).astype(np.float32)
        yaw_raw = raw.get("handoff_metric_yaw_error", np.full((cur_pose.shape[0],), np.nan, dtype=np.float32)).astype(np.float32)
        tilt_raw = raw.get("handoff_metric_tilt_error", np.full((cur_pose.shape[0],), np.nan, dtype=np.float32)).astype(np.float32)
        for ep in np.unique(ep_ids):
            idx = np.where(ep_ids == ep)[0]
            verified_candidates = idx[verified_raw[idx] > 0.5]
            ready_candidates = idx[ready_raw[idx] > 0.5]
            success_candidates = verified_candidates if verified_candidates.size > 0 else ready_candidates
            if success_candidates.size == 0:
                phase1_good_episode[int(ep)] = False
                phase1_no_success_episodes.append(int(ep))
                # A teacher rollout with no qualified ready/verified grasp is not
                # a phase-1 demonstration. Do not train the student on partial
                # correction attempts from workspace-boundary or otherwise
                # unrecoverable states.
                phase1_keep_mask[idx] = False
                continue
            success_step = int(np.min(rollout_steps[success_candidates]))
            if verified_candidates.size == 0:
                success_step += int(args.ready_positive_window)
            phase1_first_success_step[int(ep)] = success_step
            pre_success = idx[rollout_steps[idx] <= success_step]
            ready_idx = pre_success[ready_raw[pre_success] > 0.5]
            quality_idx = ready_idx if ready_idx.size > 0 else success_candidates
            xy_ok = np.nanmin(xy_raw[quality_idx]) <= float(args.phase1_success_xy_threshold)
            z_ok = np.nanmin(z_raw[quality_idx]) <= float(args.phase1_success_abs_z_threshold)
            yaw_vals = yaw_raw[quality_idx]
            yaw_ok = True if float(args.phase1_success_yaw_threshold) < 0.0 or np.all(np.isnan(yaw_vals)) else (
                np.nanmin(yaw_vals) <= float(args.phase1_success_yaw_threshold)
            )
            tilt_vals = tilt_raw[quality_idx]
            tilt_ok = True if float(args.phase1_success_tilt_threshold) < 0.0 or np.all(np.isnan(tilt_vals)) else (
                np.nanmin(tilt_vals) <= float(args.phase1_success_tilt_threshold)
            )
            good = bool(xy_ok and z_ok and yaw_ok and tilt_ok)
            phase1_good_episode[int(ep)] = good
            phase1_keep_mask[idx] &= rollout_steps[idx] <= success_step
            if bool(args.phase1_drop_weak_success_episodes) and not good:
                phase1_keep_mask[idx] = False

    support_mask = (
        (is_augmented == 0)
        & (rollout_gripper_open >= float(args.open_threshold))
        & (phase_id == 1)
        & phase1_keep_mask
    )
    if args.support_close_intent_mode == "required":
        support_mask = support_mask & (planner_close_intent > 0.5)
    elif args.support_close_intent_mode == "no_close":
        support_mask = support_mask & (planner_close_intent <= 0.5)
    initial_support_idx = np.where(support_mask)[0]
    if initial_support_idx.size == 0:
        raise RuntimeError("No support-aligned base states found for candidate dataset.")
    initial_support_delta = np.stack(
        [pose_delta_local_between(cur_pose[i], basin_center[i]) for i in initial_support_idx], axis=0
    ).astype(np.float32)
    initial_support_abs_z = np.abs(initial_support_delta[:, 2]).astype(np.float32)
    initial_support_depth_median = np.median(
        raw["wrist_depth"][initial_support_idx].reshape(initial_support_idx.size, -1), axis=1
    ).astype(np.float32)
    refined_mask = np.ones((initial_support_idx.size,), dtype=bool)
    if args.support_min_abs_z is not None:
        refined_mask &= initial_support_abs_z >= float(args.support_min_abs_z)
    if args.support_max_abs_z is not None:
        refined_mask &= initial_support_abs_z <= float(args.support_max_abs_z)
    if args.support_min_depth_median is not None:
        refined_mask &= initial_support_depth_median >= float(args.support_min_depth_median)
    if args.support_max_depth_median is not None:
        refined_mask &= initial_support_depth_median <= float(args.support_max_depth_median)
    support_idx = initial_support_idx[refined_mask]
    support_delta = initial_support_delta[refined_mask]
    if support_idx.size == 0:
        raise RuntimeError("Support filters removed all candidate states.")
    if str(args.ready_label_mode) == "event_mined":
        event_ready_targets, event_ready_meta = build_ready_targets_from_events(
            raw,
            open_threshold=float(args.open_threshold),
            positive_window=int(args.ready_positive_window),
            hard_negative_gap=int(args.ready_hard_negative_gap),
            xy_threshold=float(args.ready_xy_threshold),
            abs_z_threshold=float(args.ready_abs_z_threshold),
            yaw_threshold=float(args.ready_yaw_threshold),
            basin_distance_threshold=float(args.ready_basin_distance_threshold),
        )
    else:
        event_ready_targets = None
        event_ready_meta = None

    spec_ready_thresholds = load_ready_thresholds_from_spec(
        args.ready_spec_json,
        substage_id=int(args.ready_spec_substage_id),
        threshold_kind="release",
    )
    spec_oracle_thresholds = load_ready_thresholds_from_spec(
        args.oracle_spec_json or args.ready_spec_json,
        substage_id=int(args.oracle_spec_substage_id),
        threshold_kind="optimization",
    )
    ready_row_xy_threshold = float(
        (spec_ready_thresholds["xy_error"] if spec_ready_thresholds is not None else args.ready_xy_threshold)
        if args.ready_row_xy_threshold is None else args.ready_row_xy_threshold
    )
    ready_row_abs_z_threshold = float(
        (spec_ready_thresholds["abs_z_error"] if spec_ready_thresholds is not None else args.ready_abs_z_threshold)
        if args.ready_row_abs_z_threshold is None else args.ready_row_abs_z_threshold
    )
    ready_row_yaw_threshold = float(
        (spec_ready_thresholds["yaw_error"] if spec_ready_thresholds is not None else args.ready_yaw_threshold)
        if args.ready_row_yaw_threshold is None else args.ready_row_yaw_threshold
    )
    ready_row_basin_distance_threshold = float(
        args.ready_basin_distance_threshold
        if args.ready_row_basin_distance_threshold is None
        else args.ready_row_basin_distance_threshold
    )
    oracle_row_xy_threshold = float(
        spec_oracle_thresholds["xy_error"] if spec_oracle_thresholds is not None else ready_row_xy_threshold
    )
    oracle_row_abs_z_threshold = float(
        spec_oracle_thresholds["abs_z_error"] if spec_oracle_thresholds is not None else ready_row_abs_z_threshold
    )
    oracle_row_yaw_threshold = float(
        spec_oracle_thresholds["yaw_error"] if spec_oracle_thresholds is not None else ready_row_yaw_threshold
    )

    if "candidate_actions_local" in raw and not bool(args.force_rebuild_candidate_bank):
        cand_actions = np.asarray(raw["candidate_actions_local"][support_idx[0]], dtype=np.float32)
    else:
        cand_actions = candidate_offsets(args)
    num_cands = cand_actions.shape[0]
    if "candidate_group_index" in raw and not bool(args.force_rebuild_candidate_bank):
        candidate_group_index = np.asarray(raw["candidate_group_index"][support_idx[0]], dtype=np.int64)
        candidate_group_keys = [candidate_group_key(c) for c in cand_actions]
        unique_group_keys = sorted(set(candidate_group_keys))
    else:
        candidate_group_keys = [candidate_group_key(c) for c in cand_actions]
        unique_group_keys = sorted(set(candidate_group_keys))
        group_key_to_idx = {key: idx for idx, key in enumerate(unique_group_keys)}
        candidate_group_index = np.asarray([group_key_to_idx[key] for key in candidate_group_keys], dtype=np.int64)
    candidate_kind = candidate_kind_from_actions(cand_actions)
    num_groups = int(np.max(candidate_group_index)) + 1 if candidate_group_index.size > 0 else 0
    num_states = support_idx.size
    wrist_depth_median = np.median(raw["wrist_depth"][support_idx].reshape(num_states, -1), axis=1).astype(np.float32)
    no_op_indices = np.where(np.linalg.norm(cand_actions, axis=1) < 1e-8)[0].astype(np.int64)

    out = {
        "front_rgb": raw["front_rgb"][support_idx].astype(np.uint8)
        if "front_rgb" in raw
        else np.zeros((num_states, 128, 128, 3), dtype=np.uint8),
        "wrist_rgb": raw["wrist_rgb"][support_idx].astype(np.uint8)
        if "wrist_rgb" in raw
        else np.zeros((num_states, 128, 128, 3), dtype=np.uint8),
        "wrist_depth": raw["wrist_depth"][support_idx].astype(np.float32),
        "proprio": raw["proprio"][support_idx].astype(np.float32),
        "base_action": raw["base_action"][support_idx].astype(np.float32),
        "gripper_context": raw["gripper_context"][support_idx].astype(np.float32),
        "rollout_gripper_open": rollout_gripper_open[support_idx].astype(np.float32),
        "planner_close_intent": planner_close_intent[support_idx].astype(np.float32),
        "depth_proximity": raw["depth_proximity"][support_idx].astype(np.float32) if "depth_proximity" in raw else np.full((num_states,), np.nan, dtype=np.float32),
        "wrist_depth_median": wrist_depth_median,
        "step_idx": raw["step_idx"][support_idx].astype(np.int64),
        "phase_id": raw["phase_id"][support_idx].astype(np.int64),
        "substage_id": raw["substage_id"][support_idx].astype(np.int64)
        if "substage_id" in raw
        else np.zeros((num_states,), dtype=np.int64),
        "has_object_in_hand": raw["has_object_in_hand"][support_idx].astype(np.float32)
        if "has_object_in_hand" in raw
        else np.zeros((num_states,), dtype=np.float32),
        "contact_state": raw["contact_state"][support_idx].astype(np.int64)
        if "contact_state" in raw
        else np.zeros((num_states,), dtype=np.int64),
        "stage_target_mode": raw["stage_target_mode"][support_idx].astype(np.int64)
        if "stage_target_mode" in raw
        else np.zeros((num_states,), dtype=np.int64),
        "phase_age": raw["phase_age"][support_idx].astype(np.float32),
        "steps_since_last_replan": raw["steps_since_last_replan"][support_idx].astype(np.float32),
        "current_pose_7d": cur_pose[support_idx],
        "basin_center_pose_7d": basin_center[support_idx],
        "reference_anchor_pose_7d": raw["reference_anchor_pose_7d"][support_idx].astype(np.float32),
        "target_delta_teacher": raw["target_delta_teacher"][support_idx].astype(np.float32)
        if "target_delta_teacher" in raw
        else np.zeros((num_states, 6), dtype=np.float32),
        "teacher_source": raw["teacher_source"][support_idx].astype(np.float32)
        if "teacher_source" in raw
        else np.zeros((num_states,), dtype=np.float32),
        "contact_onset": raw["contact_onset"][support_idx].astype(np.float32)
        if "contact_onset" in raw
        else np.zeros((num_states,), dtype=np.float32),
        "post_contact_outcome": raw["post_contact_outcome"][support_idx].astype(np.float32)
        if "post_contact_outcome" in raw
        else np.zeros((num_states,), dtype=np.float32),
        "current_delta_basin_target": np.zeros((num_states, 6), dtype=np.float32),
        "current_basin_distance": np.zeros((num_states,), dtype=np.float32),
        "current_dx_sign": np.zeros((num_states,), dtype=np.int64),
        "current_dy_sign": np.zeros((num_states,), dtype=np.int64),
        "current_dyaw_sign": np.zeros((num_states,), dtype=np.int64),
        "basin_distance_bin": np.zeros((num_states,), dtype=np.int64),
        "candidate_actions_local": np.repeat(cand_actions[None, :, :], num_states, axis=0).astype(np.float32),
        "candidate_group_index": np.repeat(candidate_group_index[None, :], num_states, axis=0).astype(np.int64),
        "candidate_mask": np.repeat(np.ones((1, num_cands), dtype=np.float32), num_states, axis=0),
        "candidate_kind": np.repeat(candidate_kind[None, :], num_states, axis=0),
        "candidate_next_basin_distance": np.zeros((num_states, num_cands), dtype=np.float32),
        "candidate_improvement": np.zeros((num_states, num_cands), dtype=np.float32),
        "candidate_oracle_score": np.zeros((num_states, num_cands), dtype=np.float32),
        "candidate_basin_positive": np.zeros((num_states, num_cands), dtype=np.float32),
        "candidate_tier": np.zeros((num_states, num_cands), dtype=np.int64),
        "best_candidate_index": np.zeros((num_states,), dtype=np.int64),
        "best_group_index": np.zeros((num_states,), dtype=np.int64),
        "ready_to_close_target": np.zeros((num_states,), dtype=np.float32),
        "teacher_truth_handoff_ready": raw["teacher_truth_handoff_ready"][support_idx].astype(np.float32)
        if "teacher_truth_handoff_ready" in raw
        else np.zeros((num_states,), dtype=np.float32),
        "grasp_verified_target": raw["grasp_verified_target"][support_idx].astype(np.float32)
        if "grasp_verified_target" in raw
        else np.zeros((num_states,), dtype=np.float32),
        "retry_required_target": raw["retry_required_target"][support_idx].astype(np.float32)
        if "retry_required_target" in raw
        else np.zeros((num_states,), dtype=np.float32),
        "planner_base_action_local_raw": raw["planner_base_action_local_raw"][support_idx].astype(np.float32)
        if "planner_base_action_local_raw" in raw
        else np.zeros((num_states, 6), dtype=np.float32),
        "oracle_action_local": raw["oracle_action_local"][support_idx].astype(np.float32)
        if "oracle_action_local" in raw
        else np.zeros((num_states, 6), dtype=np.float32),
        "residual_label_local": raw["residual_label_local"][support_idx].astype(np.float32)
        if "residual_label_local" in raw
        else np.zeros((num_states, 6), dtype=np.float32),
        "support_source_index": support_idx.astype(np.int64),
        "episode_index": raw["episode_index"][support_idx].astype(np.int64)
        if "episode_index" in raw
        else np.zeros((num_states,), dtype=np.int64),
        "geometry_conditioned_pose_support": raw["geometry_conditioned_pose_support"][support_idx].astype(np.int64)
        if "geometry_conditioned_pose_support" in raw
        else np.zeros((num_states,), dtype=np.int64),
        "planner_conditioned_support": raw["planner_conditioned_support"][support_idx].astype(np.int64)
        if "planner_conditioned_support" in raw
        else np.zeros((num_states,), dtype=np.int64),
        "background_align_support": raw["background_align_support"][support_idx].astype(np.int64)
        if "background_align_support" in raw
        else np.zeros((num_states,), dtype=np.int64),
        "orientation_rescue_active": raw["orientation_rescue_active"][support_idx].astype(np.float32)
        if "orientation_rescue_active" in raw
        else np.zeros((num_states,), dtype=np.float32),
        "sample_weight": np.ones((num_states,), dtype=np.float32),
        "near_ready_xy_z_band": np.zeros((num_states,), dtype=np.float32),
        "yaw_hard_negative": np.zeros((num_states,), dtype=np.float32),
        "yaw_hard_positive": np.zeros((num_states,), dtype=np.float32),
        "xy_focus": np.zeros((num_states,), dtype=np.float32),
        "proxy_current_delta_basin_target": raw["proxy_current_delta_basin_target"][support_idx].astype(np.float32)
        if "proxy_current_delta_basin_target" in raw
        else raw["current_delta_basin_target"][support_idx].astype(np.float32)
        if "current_delta_basin_target" in raw
        else np.zeros((num_states, 6), dtype=np.float32),
        "teacher_current_delta_basin_target": raw["teacher_current_delta_basin_target"][support_idx].astype(np.float32)
        if "teacher_current_delta_basin_target" in raw
        else raw["target_delta_teacher"][support_idx].astype(np.float32)
        if "target_delta_teacher" in raw
        else np.zeros((num_states, 6), dtype=np.float32),
    }

    if "candidate_mask" in raw and not bool(args.force_rebuild_candidate_bank):
        out["candidate_mask"] = raw["candidate_mask"][support_idx].astype(np.float32)
    if "candidate_kind" in raw and not bool(args.force_rebuild_candidate_bank):
        out["candidate_kind"] = raw["candidate_kind"][support_idx]

    for row, src_i in enumerate(support_idx):
        current = cur_pose[src_i]
        center = basin_center[src_i]
        reference_anchor = raw["reference_anchor_pose_7d"][src_i].astype(np.float32)
        current_delta = support_delta[row]
        current_dist, _, _, _, _ = compute_basin_metrics(
            current_delta,
            r_xy=args.basin_radius_xy,
            r_z=args.basin_radius_z,
            r_yaw=args.basin_radius_yaw,
            r_tilt=args.basin_radius_tilt,
        )
        current_xy = float(np.linalg.norm(current_delta[:2]))
        current_abs_z = float(abs(current_delta[2]))
        current_yaw = float(abs(current_delta[5]))
        out["current_delta_basin_target"][row] = current_delta.astype(np.float32)
        out["current_basin_distance"][row] = float(current_dist)
        out["current_dx_sign"][row] = sign_bucket(float(current_delta[0]), 1e-4)
        out["current_dy_sign"][row] = sign_bucket(float(current_delta[1]), 1e-4)
        out["current_dyaw_sign"][row] = sign_bucket(float(current_delta[5]), 1e-3)
        out["basin_distance_bin"][row] = basin_distance_bin(float(current_dist))
        row_ready_band = bool(
            (
                ready_row_basin_distance_threshold < 0.0
                or current_dist <= ready_row_basin_distance_threshold
            )
            and current_xy <= ready_row_xy_threshold
            and current_abs_z <= ready_row_abs_z_threshold
            and (
                ready_row_yaw_threshold < 0.0
                or current_yaw <= ready_row_yaw_threshold
            )
        )
        if str(args.ready_label_mode) == "event_mined":
            out["ready_to_close_target"][row] = float(
                bool(event_ready_targets[src_i] > 0.5) and row_ready_band
            )
        elif str(args.ready_label_mode) == "provider_handoff":
            out["ready_to_close_target"][row] = float(
                bool(raw.get("handoff_ready_target", np.zeros((cur_pose.shape[0],), dtype=np.float32))[src_i] > 0.5)
            )
        elif str(args.ready_label_mode) == "teacher_ready_or_handoff":
            teacher_ready = bool(
                raw.get("ready_to_close_target", np.zeros((cur_pose.shape[0],), dtype=np.float32))[src_i] > 0.5
            )
            provider_ready = bool(
                raw.get("handoff_ready_target", np.zeros((cur_pose.shape[0],), dtype=np.float32))[src_i] > 0.5
            )
            ep = int(raw.get("episode_index", np.zeros((cur_pose.shape[0],), dtype=np.int64))[src_i])
            good_ep = bool(phase1_good_episode.get(ep, True))
            out["ready_to_close_target"][row] = float(good_ep and (teacher_ready or provider_ready))
        else:
            out["ready_to_close_target"][row] = float(row_ready_band)

        yaw_focus_enabled = float(ready_row_yaw_threshold) >= 0.0
        near_ready_xy_z_band = bool(
            current_xy <= float(max(ready_row_xy_threshold * float(args.yaw_focus_xy_multiplier), ready_row_xy_threshold))
            and current_abs_z <= float(max(ready_row_abs_z_threshold * float(args.yaw_focus_abs_z_multiplier), ready_row_abs_z_threshold))
        )
        yaw_hard_negative = bool(
            yaw_focus_enabled
            and near_ready_xy_z_band
            and current_yaw > float(ready_row_yaw_threshold)
        )
        yaw_hard_positive = bool(
            yaw_focus_enabled
            and near_ready_xy_z_band
            and current_yaw <= float(ready_row_yaw_threshold)
            and out["ready_to_close_target"][row] > 0.5
        )
        sample_weight = 1.0
        if near_ready_xy_z_band:
            sample_weight *= float(args.near_ready_sample_weight)
        if yaw_hard_negative:
            sample_weight *= float(args.yaw_hard_negative_weight)
        if yaw_hard_positive:
            sample_weight *= float(args.yaw_hard_positive_weight)
        out["near_ready_xy_z_band"][row] = float(near_ready_xy_z_band)
        out["yaw_hard_negative"][row] = float(yaw_hard_negative)
        out["yaw_hard_positive"][row] = float(yaw_hard_positive)
        release_xy = float(ready_row_xy_threshold)
        release_z = float(ready_row_abs_z_threshold)
        release_yaw = float(ready_row_yaw_threshold)
        xy_focus = bool(
            current_xy >= release_xy * float(args.xy_focus_xy_min_multiplier)
            and current_xy <= release_xy * float(args.xy_focus_xy_max_multiplier)
            and current_abs_z <= release_z * float(args.xy_focus_abs_z_multiplier)
            and (
                release_yaw < 0.0
                or current_yaw <= release_yaw * float(args.xy_focus_yaw_multiplier)
            )
            and out["ready_to_close_target"][row] <= 0.5
        )
        if xy_focus:
            sample_weight *= float(args.xy_focus_sample_weight)
        out["xy_focus"][row] = float(xy_focus)
        out["sample_weight"][row] = float(sample_weight)

        if (
            not bool(args.recompute_oracle_labels)
            and "candidate_oracle_score" in raw
            and raw["candidate_oracle_score"].shape[-1] == num_cands
            and "best_candidate_index" in raw
        ):
            out["candidate_next_basin_distance"][row] = raw["candidate_next_basin_distance"][src_i].astype(np.float32)
            out["candidate_improvement"][row] = raw["candidate_improvement"][src_i].astype(np.float32)
            out["candidate_oracle_score"][row] = raw["candidate_oracle_score"][src_i].astype(np.float32)
            out["candidate_basin_positive"][row] = (
                raw["candidate_basin_positive"][src_i].astype(np.float32)
                if "candidate_basin_positive" in raw
                else (out["candidate_next_basin_distance"][row] <= 1.0).astype(np.float32)
            )
            out["candidate_tier"][row] = (
                raw["candidate_tier"][src_i].astype(np.int64)
                if "candidate_tier" in raw
                else improvement_tiers(out["candidate_oracle_score"][row], out["candidate_basin_positive"][row]).astype(np.int64)
            )
            out["best_candidate_index"][row] = int(raw["best_candidate_index"][src_i])
            out["best_group_index"][row] = int(raw["best_group_index"][src_i]) if "best_group_index" in raw else int(candidate_group_index[int(raw["best_candidate_index"][src_i])])
            continue

        row_xy_threshold = float(raw["handoff_optimization_threshold_xy_error"][src_i]) if "handoff_optimization_threshold_xy_error" in raw else float("nan")
        row_z_threshold = float(raw["handoff_optimization_threshold_abs_z_error"][src_i]) if "handoff_optimization_threshold_abs_z_error" in raw else float("nan")
        row_yaw_threshold = float(raw["handoff_optimization_threshold_yaw_error"][src_i]) if "handoff_optimization_threshold_yaw_error" in raw else float("nan")
        if not np.isfinite(row_xy_threshold) and "handoff_threshold_xy_error" in raw:
            row_xy_threshold = float(raw["handoff_threshold_xy_error"][src_i])
        if not np.isfinite(row_z_threshold) and "handoff_threshold_abs_z_error" in raw:
            row_z_threshold = float(raw["handoff_threshold_abs_z_error"][src_i])
        if not np.isfinite(row_yaw_threshold) and "handoff_threshold_yaw_error" in raw:
            row_yaw_threshold = float(raw["handoff_threshold_yaw_error"][src_i])
        if not np.isfinite(row_xy_threshold) or row_xy_threshold <= 0.0:
            row_xy_threshold = float(oracle_row_xy_threshold)
        if not np.isfinite(row_z_threshold) or row_z_threshold <= 0.0:
            row_z_threshold = float(oracle_row_abs_z_threshold)
        if not np.isfinite(row_yaw_threshold):
            row_yaw_threshold = float(oracle_row_yaw_threshold)
        best_idx = 0
        best_score = -1e9
        for j, cand in enumerate(cand_actions):
            next_pose = apply_local_offset_to_pose(current, cand)
            delta_next = pose_delta_local_between(next_pose, center)
            next_dist, _, _, _, _ = compute_basin_metrics(
                delta_next,
                r_xy=args.basin_radius_xy,
                r_z=args.basin_radius_z,
                r_yaw=args.basin_radius_yaw,
                r_tilt=args.basin_radius_tilt,
            )
            improve = float(current_dist - next_dist)
            in_basin = float(next_dist <= 1.0)
            if args.oracle_mode == "short_horizon_funnel":
                total_score = 0.0
                pose_t = current.copy()
                delta_t = current_delta.copy()
                cand_t = np.asarray(cand, dtype=np.float32)
                for t in range(max(int(args.horizon_k), 1)):
                    pose_next_t = apply_local_offset_to_pose(pose_t, cand_t)
                    delta_next_t = pose_delta_local_between(pose_next_t, center)
                    step_score, _ = score_candidate_approach_funnel(
                        current_pose_7d=pose_t,
                        next_pose_7d=pose_next_t,
                        current_delta=delta_t,
                        next_delta=delta_next_t,
                        candidate_local=cand_t,
                        reference_anchor_pose_7d=reference_anchor,
                        base_action_local=raw["base_action"][src_i][:6].astype(np.float32),
                        depth_proximity=float(raw["depth_proximity"][src_i]) if "depth_proximity" in raw else None,
                        r_xy=args.basin_radius_xy,
                        r_z=args.basin_radius_z,
                        r_yaw=args.basin_radius_yaw,
                        r_tilt=args.basin_radius_tilt,
                    )
                    total_score += (float(args.discount_gamma) ** t) * float(step_score)
                    pose_t = pose_next_t
                    delta_t = delta_next_t
                    if t + 1 < int(args.horizon_k):
                        best_future_score = -1e9
                        best_future_cand = cand_actions[0]
                        for cand2 in cand_actions:
                            pose_future = apply_local_offset_to_pose(pose_t, cand2)
                            delta_future = pose_delta_local_between(pose_future, center)
                            step_future_score, _ = score_candidate_approach_funnel(
                                current_pose_7d=pose_t,
                                next_pose_7d=pose_future,
                                current_delta=delta_t,
                                next_delta=delta_future,
                                candidate_local=cand2,
                                reference_anchor_pose_7d=reference_anchor,
                                base_action_local=raw["base_action"][src_i][:6].astype(np.float32),
                                depth_proximity=float(raw["depth_proximity"][src_i]) if "depth_proximity" in raw else None,
                                r_xy=args.basin_radius_xy,
                                r_z=args.basin_radius_z,
                                r_yaw=args.basin_radius_yaw,
                                r_tilt=args.basin_radius_tilt,
                            )
                            if step_future_score > best_future_score:
                                best_future_score = float(step_future_score)
                                best_future_cand = np.asarray(cand2, dtype=np.float32)
                        cand_t = best_future_cand
                score = float(total_score)
            elif args.oracle_mode == "approach_funnel":
                score, _ = score_candidate_approach_funnel(
                    current_pose_7d=current,
                    next_pose_7d=next_pose,
                    current_delta=current_delta,
                    next_delta=delta_next,
                    candidate_local=cand,
                    reference_anchor_pose_7d=reference_anchor,
                    base_action_local=raw["base_action"][src_i][:6].astype(np.float32),
                    depth_proximity=float(raw["depth_proximity"][src_i]) if "depth_proximity" in raw else None,
                    r_xy=args.basin_radius_xy,
                    r_z=args.basin_radius_z,
                    r_yaw=args.basin_radius_yaw,
                    r_tilt=args.basin_radius_tilt,
                )
            elif args.oracle_mode == "stage_handoff_joint":
                reference_axis_local = pose_delta_local_between(current, reference_anchor)
                score, _ = score_candidate_stage_handoff_joint(
                    current_delta=current_delta,
                    next_delta=delta_next,
                    candidate_local=cand,
                    base_action_local=raw["base_action"][src_i][:6].astype(np.float32),
                    reference_anchor_local=reference_axis_local,
                    xy_threshold=row_xy_threshold,
                    abs_z_threshold=row_z_threshold,
                    yaw_threshold=row_yaw_threshold,
                )
            else:
                score = improve + 0.5 * in_basin
            if (
                planner_close_intent[src_i] <= 0.5
                and (
                    current_dist <= float(args.no_intent_hold_basin_distance)
                    or abs(float(current_delta[2])) >= float(args.no_intent_hold_abs_z_threshold)
                )
            ):
                cand_norm = float(np.linalg.norm(np.asarray(cand, dtype=np.float32).reshape(6)))
                if cand_norm < 1e-8:
                    score += float(args.no_intent_noop_bonus)
                else:
                    score -= float(args.no_intent_motion_penalty) * cand_norm / max(float(args.primitive_xy_small), 1e-6)
            out["candidate_next_basin_distance"][row, j] = float(next_dist)
            out["candidate_improvement"][row, j] = improve
            out["candidate_oracle_score"][row, j] = float(score)
            out["candidate_basin_positive"][row, j] = in_basin
            if score > best_score:
                best_score = score
                best_idx = j
        out["best_candidate_index"][row] = int(best_idx)
        out["best_group_index"][row] = int(candidate_group_index[best_idx])
        out["candidate_tier"][row] = improvement_tiers(out["candidate_oracle_score"][row], out["candidate_basin_positive"][row])

    np.savez_compressed(output_path, **out)

    best_hist = {
        int(k): int(v)
        for k, v in zip(*np.unique(out["best_candidate_index"], return_counts=True))
    }
    best_group_hist = {
        int(k): int(v)
        for k, v in zip(*np.unique(out["best_group_index"], return_counts=True))
    }
    tier_hist = {
        int(k): int(v)
        for k, v in zip(*np.unique(out["candidate_tier"], return_counts=True))
    }
    close_intent_hist = {
        int(k): int(v)
        for k, v in zip(*np.unique((out["planner_close_intent"] > 0.5).astype(np.int64), return_counts=True))
    }
    support_source_summary = {
        "geometry_conditioned_pose_support": int(np.sum(out["geometry_conditioned_pose_support"] > 0)),
        "planner_conditioned_support": int(np.sum(out["planner_conditioned_support"] > 0)),
        "background_align_support": int(np.sum(out["background_align_support"] > 0)),
    }
    no_close_mask = out["planner_close_intent"] <= 0.5
    high_depth_mask = out["wrist_depth_median"] >= float(args.early_depth_median_threshold)
    early_high_depth_mask = no_close_mask & high_depth_mask
    abs_z = np.abs(out["current_delta_basin_target"][:, 2])
    high_z_mask = no_close_mask & (abs_z >= float(args.no_intent_hold_abs_z_threshold))
    low_z_or_close_mask = (out["planner_close_intent"] > 0.5) | (abs_z < float(args.mid_abs_z_threshold))
    mid_z_no_close_mask = (
        no_close_mask
        & (abs_z >= float(args.mid_abs_z_threshold))
        & (abs_z < float(args.no_intent_hold_abs_z_threshold))
    )

    def _masked_best_summary(mask):
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return {"count": 0, "best_candidate_hist": {}, "best_group_hist": {}, "no_op_best_rate": 0.0, "lateral_yaw_best_rate": 0.0}
        best_vals = out["best_candidate_index"][mask]
        best_actions = cand_actions[best_vals]
        lateral_yaw = (np.linalg.norm(best_actions[:, :2], axis=1) > 1e-6) | (np.abs(best_actions[:, 5]) > 1e-6)
        no_op = np.linalg.norm(best_actions, axis=1) < 1e-8
        uniq, cnt = np.unique(best_vals, return_counts=True)
        guniq, gcnt = np.unique(out["best_group_index"][mask], return_counts=True)
        return {
            "count": int(mask.sum()),
            "best_candidate_hist": {int(k): int(v) for k, v in zip(uniq, cnt)},
            "best_group_hist": {int(k): int(v) for k, v in zip(guniq, gcnt)},
            "no_op_best_rate": float(np.mean(no_op)),
            "lateral_yaw_best_rate": float(np.mean(lateral_yaw)),
        }

    selected_improve = out["candidate_improvement"][np.arange(num_states), out["best_candidate_index"]]
    selected_oracle_score = out["candidate_oracle_score"][np.arange(num_states), out["best_candidate_index"]]
    group_summary = {}
    for group_name, values in {
        "basin_distance_bin": out["basin_distance_bin"],
        "dx_sign": out["current_dx_sign"],
        "dy_sign": out["current_dy_sign"],
        "dyaw_sign": out["current_dyaw_sign"],
    }.items():
        summary = {}
        for val in np.unique(values):
            mask = values == val
            uniq, cnt = np.unique(out["best_candidate_index"][mask], return_counts=True)
            summary[int(val)] = {
                "count": int(mask.sum()),
                "best_candidate_hist": {int(k): int(v) for k, v in zip(uniq, cnt)},
                "best_group_hist": {
                    int(k): int(v)
                    for k, v in zip(*np.unique(out["best_group_index"][mask], return_counts=True))
                },
            }
        group_summary[group_name] = summary

    meta = {
        "input_dir": str(input_dir),
        "prefilter_num_states": int(prefilter_num_states),
        "prefilter_occluded_count": int(occluded_prefilter_count),
        "exclude_occluded": bool(args.exclude_occluded),
        "output_path": str(output_path),
        "num_support_states": int(num_states),
        "num_candidates": int(num_cands),
        "num_candidate_groups": int(num_groups),
        "candidate_actions_local": cand_actions.tolist(),
        "no_op_candidate_indices": no_op_indices.tolist(),
        "follow_planner_residual_candidate_indices": no_op_indices.tolist(),
        "candidate_mode": args.candidate_mode,
        "force_rebuild_candidate_bank": bool(args.force_rebuild_candidate_bank),
        "support_close_intent_mode": args.support_close_intent_mode,
        "phase1_truncate_to_first_success": bool(args.phase1_truncate_to_first_success),
        "phase1_drop_weak_success_episodes": bool(args.phase1_drop_weak_success_episodes),
        "phase1_good_episode_count": int(sum(1 for v in phase1_good_episode.values() if v)),
        "phase1_weak_episode_count": int(sum(1 for v in phase1_good_episode.values() if not v)),
        "phase1_no_success_episode_count": int(len(phase1_no_success_episodes)),
        "phase1_no_success_episodes": [int(ep) for ep in phase1_no_success_episodes],
        "phase1_first_success_step": {str(k): int(v) for k, v in phase1_first_success_step.items()},
        "phase1_success_quality_thresholds": {
            "xy": float(args.phase1_success_xy_threshold),
            "abs_z": float(args.phase1_success_abs_z_threshold),
            "yaw": float(args.phase1_success_yaw_threshold),
            "tilt": float(args.phase1_success_tilt_threshold),
        },
        "support_source_summary": support_source_summary,
        "support_planner_close_intent_hist": close_intent_hist,
        "support_no_close_intent_summary": _masked_best_summary(no_close_mask),
        "support_close_intent_summary": _masked_best_summary(~no_close_mask),
        "support_early_high_depth_summary": _masked_best_summary(early_high_depth_mask),
        "support_high_z_no_close_summary": _masked_best_summary(high_z_mask),
        "support_mid_z_no_close_summary": _masked_best_summary(mid_z_no_close_mask),
        "support_low_z_or_close_summary": _masked_best_summary(low_z_or_close_mask),
        "early_depth_median_threshold": float(args.early_depth_median_threshold),
        "support_filter_abs_z": {
            "min": None if args.support_min_abs_z is None else float(args.support_min_abs_z),
            "max": None if args.support_max_abs_z is None else float(args.support_max_abs_z),
        },
        "support_filter_depth_median": {
            "min": None if args.support_min_depth_median is None else float(args.support_min_depth_median),
            "max": None if args.support_max_depth_median is None else float(args.support_max_depth_median),
        },
        "mid_abs_z_threshold": float(args.mid_abs_z_threshold),
        "ready_label": {
            "mode": str(args.ready_label_mode),
            "xy_threshold": float(ready_row_xy_threshold),
            "abs_z_threshold": float(ready_row_abs_z_threshold),
            "yaw_threshold": float(ready_row_yaw_threshold),
            "basin_distance_threshold": float(ready_row_basin_distance_threshold),
            "spec_json": None if args.ready_spec_json is None else str(args.ready_spec_json),
            "spec_substage_id": int(args.ready_spec_substage_id),
            "positive_window": int(args.ready_positive_window),
            "hard_negative_gap": int(args.ready_hard_negative_gap),
            "positive_rate": float(np.mean(out["ready_to_close_target"] > 0.5)),
            "positive_count": int(np.sum(out["ready_to_close_target"] > 0.5)),
            "pre_row_filter_positive_count": int(
                np.sum(event_ready_targets[support_idx] > 0.5)
            )
            if event_ready_targets is not None
            else int(np.sum(out["ready_to_close_target"] > 0.5)),
            "source_xy_threshold": float(args.ready_xy_threshold),
            "source_abs_z_threshold": float(args.ready_abs_z_threshold),
            "source_yaw_threshold": float(args.ready_yaw_threshold),
            "source_basin_distance_threshold": float(args.ready_basin_distance_threshold),
            "threshold_kind": "release",
        },
        "oracle_thresholds": {
            "xy_threshold": float(oracle_row_xy_threshold),
            "abs_z_threshold": float(oracle_row_abs_z_threshold),
            "yaw_threshold": float(oracle_row_yaw_threshold),
            "spec_json": None if (args.oracle_spec_json or args.ready_spec_json) is None else str(args.oracle_spec_json or args.ready_spec_json),
            "spec_substage_id": int(args.oracle_spec_substage_id),
            "threshold_kind": "optimization",
        },
        "yaw_focus": {
            "xy_multiplier": float(args.yaw_focus_xy_multiplier),
            "abs_z_multiplier": float(args.yaw_focus_abs_z_multiplier),
            "hard_negative_weight": float(args.yaw_hard_negative_weight),
            "hard_positive_weight": float(args.yaw_hard_positive_weight),
            "near_ready_sample_weight": float(args.near_ready_sample_weight),
            "near_ready_xy_z_count": int(np.sum(out["near_ready_xy_z_band"] > 0.5)),
            "yaw_hard_negative_count": int(np.sum(out["yaw_hard_negative"] > 0.5)),
            "yaw_hard_positive_count": int(np.sum(out["yaw_hard_positive"] > 0.5)),
            "sample_weight_mean": float(np.mean(out["sample_weight"])),
            "sample_weight_max": float(np.max(out["sample_weight"])),
        },
        "xy_focus": {
            "xy_min_multiplier": float(args.xy_focus_xy_min_multiplier),
            "xy_max_multiplier": float(args.xy_focus_xy_max_multiplier),
            "abs_z_multiplier": float(args.xy_focus_abs_z_multiplier),
            "yaw_multiplier": float(args.xy_focus_yaw_multiplier),
            "sample_weight": float(args.xy_focus_sample_weight),
            "count": int(np.sum(out["xy_focus"] > 0.5)),
        },
        "oracle_mode": args.oracle_mode,
        "oracle_label_mode": args.oracle_mode,
        "recompute_oracle_labels": bool(args.recompute_oracle_labels),
        "visual_label_mode": "oracle_candidate_scoring",
        "horizon_k": int(args.horizon_k),
        "gamma": float(args.discount_gamma),
        "funnel_axis": str(args.funnel_axis),
        "funnel_cost_terms": {
            "far_z_scale": 3.0,
            "xy_near_gain": 4.0,
            "yaw_near_gain": 3.0,
            "tilt_near_gain": 2.5,
            "planner_nominal_regularization": 0.30,
            "guarded_descend": True,
            "no_intent_hold_basin_distance": float(args.no_intent_hold_basin_distance),
            "no_intent_hold_abs_z_threshold": float(args.no_intent_hold_abs_z_threshold),
            "no_intent_noop_bonus": float(args.no_intent_noop_bonus),
            "no_intent_motion_penalty": float(args.no_intent_motion_penalty),
        },
        "candidate_group_index": candidate_group_index.tolist(),
        "candidate_group_keys": [list(key) for key in unique_group_keys],
        "basin_radius_tilt": float(args.basin_radius_tilt),
        "best_candidate_hist": best_hist,
        "best_group_hist": best_group_hist,
        "candidate_tier_hist": tier_hist,
        "selected_improvement_stats": {
            "mean": float(np.mean(selected_improve)),
            "p50": float(np.percentile(selected_improve, 50)),
            "p95": float(np.percentile(selected_improve, 95)),
        },
        "selected_oracle_score_stats": {
            "mean": float(np.mean(selected_oracle_score)),
            "p50": float(np.percentile(selected_oracle_score, 50)),
            "p95": float(np.percentile(selected_oracle_score, 95)),
        },
        "grouped_oracle_best_hist": group_summary,
    }
    if event_ready_meta is not None:
        meta["ready_label"]["event_mined"] = event_ready_meta
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
