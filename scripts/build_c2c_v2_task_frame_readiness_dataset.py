#!/usr/bin/env python3
"""Build task-frame z/yaw readiness rows from C2C v2 traces.

Labels may use privileged relabel/probe residuals offline.  Features are kept
to runtime-visible evidence and task-frame contract fields so the resulting
dataset can train non-privileged readiness estimators.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_from_path(path: Path) -> int:
    for token in path.stem.split("_"):
        if token.startswith("ep") and token[2:].isdigit():
            return int(token[2:])
    return -1


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in paths:
        files = sorted(item.glob("ep*_gripper_trace.jsonl")) if item.is_dir() else [item]
        for path in files:
            ep = _episode_from_path(path)
            for row in _read_jsonl(path):
                item_row = dict(row)
                item_row.setdefault("episode_idx", ep)
                item_row.setdefault("source_trace_path", str(path))
                rows.append(item_row)
    rows.sort(key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step", r.get("step_idx", -1)))))
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _nested(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, Mapping):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, Mapping) else {}


def _vec(value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1) if value is not None else np.zeros((0,), dtype=np.float32)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), constant_values=np.nan)
    return arr[:length].astype(np.float32)


def _compact_true_error(row: Mapping[str, Any]) -> np.ndarray:
    for key in ("grasp_probe_pre_true_error_t", "true_basin_error_t", "task_frame_true_residual"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return np.asarray(
                [
                    _safe_float(value.get("dx")),
                    _safe_float(value.get("dy")),
                    _safe_float(value.get("dz")),
                    _safe_float(value.get("dyaw")),
                ],
                dtype=np.float32,
            )
        arr = _vec(value, 6)
        if np.any(np.isfinite(arr)):
            yaw_idx = 5 if np.isfinite(arr[5]) else 3
            return np.asarray([arr[0], arr[1], arr[2], arr[yaw_idx]], dtype=np.float32)
    return np.full((4,), np.nan, dtype=np.float32)


def _skill_contract(row: Mapping[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    skill = str(row.get("c2c_v2_skill_type", row.get("skill_type", "precision_grasp")))
    stage = str(row.get("c2c_v2_stage", row.get("stage_name", "")))
    if skill == "precision_align" or stage.startswith("RING_SPOKE"):
        return skill, "ring_aperture_frame", "target_spoke_axis_frame", ("x", "y", "z", "yaw")
    return skill if skill else "precision_grasp", "gripper_jaw_frame", "ring_grasp_frame", ("x", "y", "z", "yaw")


def build_readiness_row(
    row: Mapping[str, Any],
    *,
    xy_threshold: float,
    z_threshold: float,
    yaw_threshold: float,
) -> dict[str, Any] | None:
    truth = _compact_true_error(row)
    if not np.all(np.isfinite(truth[:4])):
        return None
    skill, reference_frame, target_frame, active_dofs = _skill_contract(row)
    grasp = _nested(row, "local_geometry_error", "grasp")
    est = _nested(row, "estimated_basin_error")
    runtime_xy = _nested(row, "runtime_xy_estimator")
    force_norm = _safe_float(row.get("grasp_contact_rule_force_norm"), _safe_float(row.get("force_model_contact"), 0.0))
    contact_confirmed = bool(row.get("grasp_contact_rule_contact_confirmed", False))
    xy_error = float(np.linalg.norm(truth[:2]))
    z_abs = float(abs(float(truth[2])))
    yaw_abs = float(abs(float(truth[3])))
    xy_ready = bool(xy_error <= float(xy_threshold))
    z_near = bool(z_abs <= float(z_threshold))
    yaw_near = bool(yaw_abs <= float(yaw_threshold))
    local_conf = _safe_float(grasp.get("confidence"), _safe_float(row.get("depth_conf"), 0.0))
    local_obs = _safe_float(grasp.get("observability"), _safe_float(row.get("depth_obs_quality"), 0.0))
    wrist_occluded = bool(row.get("wrist_is_occluded", False))
    z_observable = bool(local_conf >= 0.20 and local_obs >= 5.0e-4 and not wrist_occluded)
    yaw_decision = str(row.get("alias_drift_decision", row.get("yaw_alias_drift_decision", "unknown")))
    yaw_observable = bool(yaw_decision == "stable_alias_control" or row.get("alignment_yaw_ready", False))
    z_contact_or_depth_ready = bool(z_near or contact_confirmed or force_norm > 0.10)
    handoff_ready = bool(xy_ready and z_near and yaw_near and z_observable and yaw_observable)
    return {
        "schema_version": "c2c_v2_task_frame_readiness_row_v1",
        "episode_idx": int(row.get("episode_idx", -1)),
        "step": int(row.get("step", row.get("step_idx", -1))),
        "skill_id": skill,
        "stage_name": str(row.get("c2c_v2_stage", row.get("stage_name", ""))),
        "reference_frame": reference_frame,
        "target_frame": target_frame,
        "active_dofs": list(active_dofs),
        "failure_bucket": str(row.get("failure_bucket", "")),
        "alias_drift_decision": yaw_decision,
        "visual_observability_class": str(row.get("visual_observability_class", row.get("grasp_probe_visibility_bucket", ""))),
        "source_trace_path": str(row.get("source_trace_path", "")),
        "runtime_features": {
            "local_dx": _safe_float(grasp.get("dx"), 0.0),
            "local_dy": _safe_float(grasp.get("dy"), 0.0),
            "local_dz_proxy": _safe_float(grasp.get("dz"), 0.0),
            "image_axis_yaw": _safe_float(grasp.get("image_axis_yaw"), 0.0),
            "local_confidence": float(local_conf),
            "local_observability": float(local_obs),
            "local_fit_residual": _safe_float(grasp.get("fit_residual"), 0.0),
            "local_inlier_ratio": _safe_float(grasp.get("inlier_ratio"), 0.0),
            "estimated_proxy_dx": _safe_float(est.get("estimated_basin_error_proxy_dx", est.get("proxy_dx")), 0.0),
            "estimated_proxy_dy": _safe_float(est.get("estimated_basin_error_proxy_dy", est.get("proxy_dy")), 0.0),
            "estimated_proxy_dz": _safe_float(est.get("estimated_basin_error_proxy_dz", est.get("proxy_dz")), 0.0),
            "estimated_proxy_dyaw": _safe_float(est.get("estimated_basin_error_proxy_dyaw", est.get("proxy_dyaw")), 0.0),
            "runtime_xy_dx": _safe_float(runtime_xy.get("dx"), 0.0),
            "runtime_xy_dy": _safe_float(runtime_xy.get("dy"), 0.0),
            "runtime_xy_entry_ready": bool(runtime_xy.get("entry_ready", False)),
            "wrist_valid_depth_ratio": _safe_float(row.get("wrist_valid_depth_ratio"), 0.0),
            "wrist_depth_near_fraction": _safe_float(row.get("wrist_depth_near_fraction"), 0.0),
            "wrist_is_occluded": wrist_occluded,
            "wrist_is_low_visibility": bool(row.get("wrist_is_low_visibility", False)),
            "force_norm": float(force_norm),
            "contact_confirmed": contact_confirmed,
            "planner_local_dx": float(_vec(row.get("planner_chunk_local_6d"), 6)[0]),
            "planner_local_dy": float(_vec(row.get("planner_chunk_local_6d"), 6)[1]),
            "planner_local_dz": float(_vec(row.get("planner_chunk_local_6d"), 6)[2]),
            "planner_local_dyaw": float(_vec(row.get("planner_chunk_local_6d"), 6)[5]),
        },
        "offline_labels": {
            "dx": float(truth[0]),
            "dy": float(truth[1]),
            "dz": float(truth[2]),
            "dyaw": float(truth[3]),
            "xy_error": float(xy_error),
            "z_abs": float(z_abs),
            "yaw_abs": float(yaw_abs),
            "xy_ready": bool(xy_ready),
            "z_observable": bool(z_observable),
            "z_near_alignment": bool(z_near),
            "z_contact_or_depth_ready": bool(z_contact_or_depth_ready),
            "z_abstain_reason": "" if z_observable else "z_not_observable",
            "yaw_observable": bool(yaw_observable),
            "yaw_ambiguous": bool(yaw_decision == "unknown"),
            "yaw_unobservable": bool(yaw_decision == "frame_drift_abstain"),
            "yaw_near_alignment": bool(yaw_near),
            "yaw_abstain_reason": "" if yaw_observable else str(yaw_decision or "yaw_unobservable"),
            "alignment_ready_for_handoff": bool(handoff_ready),
        },
        "z_semantics": "task_approach_axis_residual",
        "yaw_semantics": "task_frame_yaw_residual",
        "uses_privileged_label_for_training": True,
        "uses_privileged_runtime": False,
    }


def build_dataset(rows: list[Mapping[str, Any]], *, xy_threshold: float, z_threshold: float, yaw_threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        built = build_readiness_row(row, xy_threshold=xy_threshold, z_threshold=z_threshold, yaw_threshold=yaw_threshold)
        if built is not None:
            out.append(built)
    return out


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [r["offline_labels"] for r in rows]
    buckets = Counter(str(r.get("failure_bucket", "")) for r in rows)
    yaw = Counter(str(r.get("alias_drift_decision", "")) for r in rows)
    return {
        "schema_version": "c2c_v2_task_frame_readiness_dataset_summary_v1",
        "rows": int(len(rows)),
        "episodes": sorted({int(r.get("episode_idx", -1)) for r in rows}),
        "failure_bucket_counts": dict(buckets),
        "alias_drift_decision_counts": dict(yaw),
        "xy_ready_rate": float(np.mean([bool(l["xy_ready"]) for l in labels])) if labels else 0.0,
        "z_observable_rate": float(np.mean([bool(l["z_observable"]) for l in labels])) if labels else 0.0,
        "z_near_alignment_rate": float(np.mean([bool(l["z_near_alignment"]) for l in labels])) if labels else 0.0,
        "z_contact_or_depth_ready_rate": float(np.mean([bool(l["z_contact_or_depth_ready"]) for l in labels])) if labels else 0.0,
        "yaw_observable_rate": float(np.mean([bool(l["yaw_observable"]) for l in labels])) if labels else 0.0,
        "yaw_near_alignment_rate": float(np.mean([bool(l["yaw_near_alignment"]) for l in labels])) if labels else 0.0,
        "alignment_ready_for_handoff_rate": float(np.mean([bool(l["alignment_ready_for_handoff"]) for l in labels])) if labels else 0.0,
        "uses_privileged_label_for_training": True,
        "uses_privileged_runtime": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_paths", nargs="+", required=True)
    ap.add_argument("--output_jsonl", type=Path, required=True)
    ap.add_argument("--output_summary", type=Path, required=True)
    ap.add_argument("--xy_threshold", type=float, default=0.005)
    ap.add_argument("--z_threshold", type=float, default=0.020)
    ap.add_argument("--yaw_threshold", type=float, default=0.030)
    args = ap.parse_args()
    rows = build_dataset(
        load_rows([Path(p) for p in args.trace_paths]),
        xy_threshold=float(args.xy_threshold),
        z_threshold=float(args.z_threshold),
        yaw_threshold=float(args.yaw_threshold),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(rows)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
