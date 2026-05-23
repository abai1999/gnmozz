#!/usr/bin/env python3
"""Offline privileged grasp-error relabel evaluator for C2C v2 rollouts.

This script does not trust runtime localizer errors.  Instead it recomputes
per-step local dx/dy/dz/dyaw from:
  - runtime-observation gripper pose
  - captured privileged ring pose (`episode_target_pose_7d`)
  - task-specific phase-1 grasp contract (`configs/grasp_specs/*.json`)

The goal is to answer whether rollout variants differ under an independent
labeler, not whether the online localizer self-reports improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.stage_target_provider import (
    apply_yaw_symmetry_to_delta,
    build_phase1_teacher_targets,
    load_phase1_grasp_spec,
    pose_delta_local_between,
    select_phase1_teacher_target,
)
from prismatic.robot.coarse2contact_v2.recovery_audit import in_close_ready_basin, in_near_grasp_basin


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("step", -1)))
    return rows


def _episode_target_pose(npz: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "episode_target_pose_7d" not in npz.files:
        return None
    arr = np.asarray(npz["episode_target_pose_7d"], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.size < 7 or not np.all(np.isfinite(arr[:7])):
        return None
    return arr[:7].astype(np.float32)


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _variant_name(eval_root: Path, results: dict[str, Any]) -> str:
    stats = list(results.get("stage_stats", []))
    if stats:
        item = stats[0]
        variant = str(item.get("basin_pullback_variant", "") or "")
        gain = item.get("basin_visual_gain", None)
        step = item.get("basin_max_pullback_xy_step", None)
        budget = item.get("basin_max_recovery_steps", None)
        bits = [variant or str(results.get("mode", "unknown"))]
        if gain is not None:
            bits.append(f"g{_float(gain):.2f}")
        if step is not None:
            bits.append(f"s{_float(step):.4f}")
        if budget is not None:
            bits.append(f"b{int(budget)}")
        return "_".join(bits)
    return eval_root.name


def _relabeled_delta(current_gripper_pose: np.ndarray, ring_pose_7d: np.ndarray, task_name: str) -> tuple[np.ndarray, bool]:
    spec = load_phase1_grasp_spec(task_name)
    pregrasp_target, grasp_commit_target = build_phase1_teacher_targets(ring_pose_7d, spec)
    active_target, use_commit = select_phase1_teacher_target(
        current_gripper_pose=current_gripper_pose,
        pregrasp_target_pose_7d=pregrasp_target,
        grasp_commit_target_pose_7d=grasp_commit_target,
        grasp_spec=spec,
    )
    delta = pose_delta_local_between(current_gripper_pose, active_target)
    delta = apply_yaw_symmetry_to_delta(delta, float(spec.yaw_symmetry_period))
    return delta.astype(np.float32), bool(use_commit)


def _row_summary(trace_row: dict[str, Any], delta_local: np.ndarray) -> dict[str, Any]:
    dx, dy, dz, dyaw = [float(v) for v in delta_local[:4]]
    xy = float(np.hypot(dx, dy))
    return {
        "step": int(trace_row.get("step", -1)),
        "c2c_gate_active": bool(trace_row.get("c2c_gate_active", False)),
        "basin_recovery_mode": str(trace_row.get("basin_recovery_mode", "")),
        "phase_owner": str(trace_row.get("phase_owner", "")),
        "privileged_dx": dx,
        "privileged_dy": dy,
        "privileged_dz": dz,
        "privileged_dyaw": dyaw,
        "privileged_xy_error": xy,
        "privileged_near_grasp": bool(in_near_grasp_basin(dx, dy, dyaw)),
        "privileged_close_ready": bool(in_close_ready_basin(dx, dy, dyaw)),
        "runtime_dx": _float((((trace_row.get("local_geometry_error") or {}).get("grasp") or {}).get("dx")), 0.0),
        "runtime_dy": _float((((trace_row.get("local_geometry_error") or {}).get("grasp") or {}).get("dy")), 0.0),
        "runtime_dyaw": _float((((trace_row.get("local_geometry_error") or {}).get("grasp") or {}).get("dyaw")), 0.0),
        "local_correction_xy": float(
            np.linalg.norm(np.asarray((trace_row.get("local_correction_local_6d") or [0.0] * 6)[:2], dtype=np.float32))
        ),
        "local_correction_yaw": abs(float((trace_row.get("local_correction_local_6d") or [0.0] * 6)[5])),
    }


def evaluate_root(eval_root: Path, task_name: str) -> dict[str, Any]:
    eval_root = eval_root.resolve()
    results_path = eval_root / "eval_results.json"
    trace_dir = eval_root / "gripper_traces"
    obs_dir = eval_root / "runtime_observations"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing eval_results.json under {eval_root}")
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing gripper_traces under {eval_root}")
    if not obs_dir.exists():
        raise FileNotFoundError(f"Missing runtime_observations under {eval_root}; rerun with --dump_runtime_obs")

    results = _read_json(results_path)
    out: dict[str, Any] = {
        "eval_root": str(eval_root),
        "variant_name": _variant_name(eval_root, results),
        "mode": str(results.get("mode", "")),
        "episodes": [],
        "counts": Counter(),
    }

    for trace_path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep_tag = trace_path.stem.split("_")[0]
        obs_path = obs_dir / f"{ep_tag}_runtime_obs.npz"
        if not obs_path.exists():
            continue
        trace_rows = _trace_rows(trace_path)
        if not trace_rows:
            continue
        with np.load(obs_path, allow_pickle=True) as obs_npz:
            ring_pose_7d = _episode_target_pose(obs_npz)
            if ring_pose_7d is None:
                raise RuntimeError(f"{obs_path} missing valid episode_target_pose_7d")
            rows = []
            for row in trace_rows:
                step = int(row.get("step", -1))
                if step < 0 or step >= int(obs_npz["gripper_pose"].shape[0]):
                    continue
                gripper_pose = np.asarray(obs_npz["gripper_pose"][step], dtype=np.float32)
                delta_local, use_commit = _relabeled_delta(gripper_pose, ring_pose_7d, task_name)
                item = _row_summary(row, delta_local)
                item["privileged_use_commit_target"] = bool(use_commit)
                rows.append(item)

        first_gate = next((i for i, r in enumerate(rows) if r["c2c_gate_active"]), None)
        post = rows[first_gate:] if first_gate is not None else []
        pull = [r for r in post if r["basin_recovery_mode"] == "VISUAL_PULLBACK"]
        micro = [r for r in post if r["basin_recovery_mode"] == "MICRO_SERVO_TO_BASIN"]
        pull_plus_micro = [r for r in post if r["basin_recovery_mode"] in {"VISUAL_PULLBACK", "MICRO_SERVO_TO_BASIN"}]
        privileged_xy = [float(r["privileged_xy_error"]) for r in pull]
        privileged_xy_pm = [float(r["privileged_xy_error"]) for r in pull_plus_micro]
        runtime_xy = [math.hypot(float(r["runtime_dx"]), float(r["runtime_dy"])) for r in pull]
        runtime_xy_pm = [math.hypot(float(r["runtime_dx"]), float(r["runtime_dy"])) for r in pull_plus_micro]

        episode_report = {
            "episode": ep_tag,
            "first_gate_idx": first_gate,
            "post_mode_counts": dict(Counter(r["basin_recovery_mode"] for r in post)),
            "privileged_near_grasp_post_gate": bool(any(r["privileged_near_grasp"] for r in post)),
            "privileged_close_ready_post_gate": bool(any(r["privileged_close_ready"] for r in post)),
            "pull_count": len(pull),
            "micro_count": len(micro),
            "pull_plus_micro_count": len(pull_plus_micro),
            "privileged_xy_start": float(privileged_xy[0]) if privileged_xy else None,
            "privileged_xy_end": float(privileged_xy[-1]) if privileged_xy else None,
            "privileged_xy_gain": float(privileged_xy[0] - privileged_xy[-1]) if len(privileged_xy) >= 2 else None,
            "privileged_xy_monotonic_rate": (
                float(sum(1 for i in range(1, len(privileged_xy)) if privileged_xy[i] <= privileged_xy[i - 1] + 1e-9) / (len(privileged_xy) - 1))
                if len(privileged_xy) > 1
                else None
            ),
            "privileged_xy_pm_start": float(privileged_xy_pm[0]) if privileged_xy_pm else None,
            "privileged_xy_pm_end": float(privileged_xy_pm[-1]) if privileged_xy_pm else None,
            "privileged_xy_pm_gain": float(privileged_xy_pm[0] - privileged_xy_pm[-1]) if len(privileged_xy_pm) >= 2 else None,
            "privileged_xy_pm_monotonic_rate": (
                float(sum(1 for i in range(1, len(privileged_xy_pm)) if privileged_xy_pm[i] <= privileged_xy_pm[i - 1] + 1e-9) / (len(privileged_xy_pm) - 1))
                if len(privileged_xy_pm) > 1
                else None
            ),
            "runtime_xy_start": float(runtime_xy[0]) if runtime_xy else None,
            "runtime_xy_end": float(runtime_xy[-1]) if runtime_xy else None,
            "runtime_xy_pm_start": float(runtime_xy_pm[0]) if runtime_xy_pm else None,
            "runtime_xy_pm_end": float(runtime_xy_pm[-1]) if runtime_xy_pm else None,
            "gate_frame": next((row.get("c2c_gate_frame_path") for row in trace_rows if row.get("c2c_gate_frame_path")), None),
        }
        out["episodes"].append(episode_report)
        out["counts"]["episodes"] += 1
        out["counts"]["privileged_near_grasp_hits"] += int(bool(episode_report["privileged_near_grasp_post_gate"]))
        out["counts"]["privileged_close_ready_hits"] += int(bool(episode_report["privileged_close_ready_post_gate"]))

    out["counts"] = dict(out["counts"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_roots", type=Path, nargs="+", required=True)
    ap.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    ap.add_argument("--output", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_error_relabel_eval.json"))
    args = ap.parse_args()

    report = {"task_name": args.task_name, "groups": []}
    for root in args.eval_roots:
        report["groups"].append(evaluate_root(root, args.task_name))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
