#!/usr/bin/env python3
"""Offline calibration audit for the C2C v2 basin-state layer.

This script compares:
  - runtime proxy geometry (from trace `local_geometry_error`)
  - privileged relabeled basin error (from runtime observations)
  - local correction actually emitted by the supervisor
  - next-step privileged error reduction

The goal is to decide, per axis, whether the runtime proxy is good enough to
trust for control, should remain diagnostic only, or should be abstained.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.stage_target_provider import (  # noqa: E402
    apply_yaw_symmetry_to_delta,
    build_phase1_teacher_targets,
    load_phase1_grasp_spec,
    pose_delta_local_between,
    select_phase1_teacher_target,
)


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


def _safe_sign(value: float, eps: float = 1.0e-9) -> int:
    if not np.isfinite(value) or abs(float(value)) <= eps:
        return 0
    return 1 if float(value) > 0.0 else -1


def _axis_stats(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        return {
            "num_samples": 0,
            "sign_match_rate": 0.0,
            "action_sign_match_rate": 0.0,
            "contraction_rate": 0.0,
            "proxy_priv_corr": 0.0,
            "action_priv_corr": 0.0,
            "scale_ratio_median": None,
            "action_scale_ratio_median": None,
            "recommended_policy": "abstain",
            "reason": "no_samples",
        }

    proxy = np.asarray([s["proxy"] for s in samples], dtype=np.float32)
    priv = np.asarray([s["priv"] for s in samples], dtype=np.float32)
    action = np.asarray([s["action"] for s in samples], dtype=np.float32)
    next_priv = np.asarray([s["next_priv"] for s in samples], dtype=np.float32)

    finite = np.isfinite(proxy) & np.isfinite(priv)
    finite_action = np.isfinite(action) & np.isfinite(priv)
    finite_next = np.isfinite(priv) & np.isfinite(next_priv)
    sign_mask = finite & (np.abs(priv) > 1.0e-6) & (np.abs(proxy) > 1.0e-6)
    action_sign_mask = finite_action & (np.abs(action) > 1.0e-6) & (np.abs(priv) > 1.0e-6)
    sign_match = float(np.mean([_safe_sign(vp) == _safe_sign(vr) for vp, vr in zip(proxy[sign_mask], priv[sign_mask])])) if np.any(sign_mask) else 0.0
    action_sign_match = float(np.mean([_safe_sign(va) == _safe_sign(vr) for va, vr in zip(action[action_sign_mask], priv[action_sign_mask])])) if np.any(action_sign_mask) else 0.0
    contraction = float(np.mean(np.abs(next_priv[finite_next]) <= np.abs(priv[finite_next]) + 1.0e-9)) if np.any(finite_next) else 0.0

    def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        if np.count_nonzero(mask) < 2:
            return 0.0
        aa = a[mask]
        bb = b[mask]
        if np.std(aa) <= 1.0e-9 or np.std(bb) <= 1.0e-9:
            return 0.0
        return float(np.corrcoef(aa, bb)[0, 1])

    proxy_priv_corr = _corr(proxy, priv, finite)
    action_priv_corr = _corr(action, priv, finite_action)

    ratio_mask = finite & (np.abs(priv) > 1.0e-6)
    action_ratio_mask = finite_action & (np.abs(priv) > 1.0e-6)
    scale_ratio_median = float(np.median(np.abs(proxy[ratio_mask]) / np.maximum(np.abs(priv[ratio_mask]), 1.0e-9))) if np.any(ratio_mask) else None
    action_scale_ratio_median = float(np.median(np.abs(action[action_ratio_mask]) / np.maximum(np.abs(priv[action_ratio_mask]), 1.0e-9))) if np.any(action_ratio_mask) else None

    if sign_match >= 0.70 and contraction >= 0.70 and abs(proxy_priv_corr) >= 0.25:
        policy = "trusted_control"
        reason = "sign_match_and_contraction_strong"
    elif sign_match >= 0.40 and contraction >= 0.40 and abs(proxy_priv_corr) >= 0.10:
        policy = "diagnostic_only"
        reason = "some_signal_but_not_closed_loop_stable"
    else:
        policy = "abstain"
        reason = "weak_or_inconsistent_signal"

    return {
        "num_samples": int(len(samples)),
        "sign_match_rate": float(sign_match),
        "action_sign_match_rate": float(action_sign_match),
        "contraction_rate": float(contraction),
        "proxy_priv_corr": float(proxy_priv_corr),
        "action_priv_corr": float(action_priv_corr),
        "scale_ratio_median": scale_ratio_median,
        "action_scale_ratio_median": action_scale_ratio_median,
        "recommended_policy": policy,
        "reason": reason,
    }


def _parse_proxy_axis(trace_row: dict[str, Any], axis: str) -> float:
    geom = (trace_row.get("local_geometry_error") or {}).get("grasp") or {}
    if axis == "x":
        return _float(geom.get("dx"))
    if axis == "y":
        return _float(geom.get("dy"))
    if axis == "z":
        return _float(geom.get("dz"))
    if axis == "yaw":
        return _float(geom.get("dyaw"))
    return 0.0


def _parse_action_axis(trace_row: dict[str, Any], axis: str) -> float:
    corr = np.asarray(trace_row.get("local_correction_local_6d", [0.0] * 6), dtype=np.float32).reshape(-1)
    corr = np.pad(corr, (0, max(0, 6 - corr.size)))[:6]
    if axis == "x":
        return float(corr[0])
    if axis == "y":
        return float(corr[1])
    if axis == "z":
        return float(corr[2])
    if axis == "yaw":
        return float(corr[5])
    return 0.0


def _privileged_delta(current_gripper_pose: np.ndarray, ring_pose_7d: np.ndarray, task_name: str) -> np.ndarray:
    spec = load_phase1_grasp_spec(task_name)
    pregrasp_target, grasp_commit_target = build_phase1_teacher_targets(ring_pose_7d, spec)
    active_target, _ = select_phase1_teacher_target(
        current_gripper_pose=current_gripper_pose,
        pregrasp_target_pose_7d=pregrasp_target,
        grasp_commit_target_pose_7d=grasp_commit_target,
        grasp_spec=spec,
    )
    delta = pose_delta_local_between(current_gripper_pose, active_target)
    delta = apply_yaw_symmetry_to_delta(delta, float(spec.yaw_symmetry_period))
    return np.asarray(delta, dtype=np.float32)


def _plot_episode(ep_tag: str, rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not rows:
        return
    steps = [int(r["step"]) for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax, axis_name in zip(axes, ["x", "y", "z", "yaw"]):
        proxy = [float(r[f"proxy_{axis_name}"]) for r in rows]
        priv = [float(r[f"priv_{axis_name}"]) for r in rows]
        action = [float(r[f"action_{axis_name}"]) for r in rows]
        ax.plot(steps, proxy, label="proxy", linewidth=1.6)
        ax.plot(steps, priv, label="privileged", linewidth=1.6)
        ax.plot(steps, action, label="action", linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(axis_name)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.suptitle(f"{ep_tag} basin-state calibration")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ep_tag}_basin_state_calibration.png", dpi=160)
    plt.close(fig)


def evaluate_root(eval_root: Path, task_name: str, output_dir: Path) -> dict[str, Any]:
    results_path = eval_root / "eval_results.json"
    trace_dir = eval_root / "gripper_traces"
    obs_dir = eval_root / "runtime_observations"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing eval_results.json under {eval_root}")
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing gripper_traces under {eval_root}")
    if not obs_dir.exists():
        raise FileNotFoundError(f"Missing runtime_observations under {eval_root}; rerun with --dump_runtime_obs")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = _read_json(results_path)
    episodes: list[dict[str, Any]] = []
    axis_samples: dict[str, list[dict[str, float]]] = defaultdict(list)
    axis_episode_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()

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
                continue
            rows: list[dict[str, Any]] = []
            gate_idx = next((i for i, r in enumerate(trace_rows) if bool(r.get("c2c_gate_active", False))), None)
            if gate_idx is None:
                continue
            post_trace_rows = trace_rows[gate_idx:]
            for row in post_trace_rows:
                step = int(row.get("step", -1))
                if step < 0 or step >= int(obs_npz["gripper_pose"].shape[0]):
                    continue
                gripper_pose = np.asarray(obs_npz["gripper_pose"][step], dtype=np.float32)
                privileged = _privileged_delta(gripper_pose, ring_pose_7d, task_name)
                next_privileged = privileged
                if step + 1 < int(obs_npz["gripper_pose"].shape[0]):
                    next_gripper_pose = np.asarray(obs_npz["gripper_pose"][step + 1], dtype=np.float32)
                    next_privileged = _privileged_delta(next_gripper_pose, ring_pose_7d, task_name)
                item = {
                    "episode": ep_tag,
                    "step": step,
                    "mode": str(row.get("basin_recovery_mode", "")),
                    "phase_owner": str(row.get("phase_owner", "")),
                    "proxy_x": _parse_proxy_axis(row, "x"),
                    "proxy_y": _parse_proxy_axis(row, "y"),
                    "proxy_z": _parse_proxy_axis(row, "z"),
                    "proxy_yaw": _parse_proxy_axis(row, "yaw"),
                    "priv_x": float(privileged[0]),
                    "priv_y": float(privileged[1]),
                    "priv_z": float(privileged[2]),
                    "priv_yaw": float(privileged[3]) if privileged.size >= 4 else 0.0,
                    "next_priv_x": float(next_privileged[0]),
                    "next_priv_y": float(next_privileged[1]),
                    "next_priv_z": float(next_privileged[2]),
                    "next_priv_yaw": float(next_privileged[3]) if next_privileged.size >= 4 else 0.0,
                    "action_x": _parse_action_axis(row, "x"),
                    "action_y": _parse_action_axis(row, "y"),
                    "action_z": _parse_action_axis(row, "z"),
                    "action_yaw": _parse_action_axis(row, "yaw"),
                    "estimated_basin_error": row.get("estimated_basin_error", {}),
                    "basin_axis_validity": row.get("basin_axis_validity", {}),
                    "basin_axis_source": row.get("basin_axis_source", "none"),
                }
                rows.append(item)
                for axis in ["x", "y", "z", "yaw"]:
                    axis_samples[axis].append(
                        {
                            "proxy": float(item[f"proxy_{axis}"]),
                            "priv": float(item[f"priv_{axis}"]),
                            "action": float(item[f"action_{axis}"]),
                            "next_priv": float(item[f"next_priv_{axis}"]),
                        }
                    )
            _plot_episode(ep_tag, rows, plot_dir)
            epi_summary = {
                "episode": ep_tag,
                "gate_idx": int(gate_idx),
                "gate_step": int(trace_rows[gate_idx].get("step", gate_idx)),
                "post_rows": int(len(rows)),
                "mode_counts": dict(Counter(r["mode"] for r in rows)),
            }
            for axis in ["x", "y", "z", "yaw"]:
                axis_stats = _axis_stats(axis_samples[axis][-len(rows):]) if rows else _axis_stats([])
                epi_summary[f"{axis}_policy"] = axis_stats["recommended_policy"]
                epi_summary[f"{axis}_sign_match_rate"] = axis_stats["sign_match_rate"]
                epi_summary[f"{axis}_contraction_rate"] = axis_stats["contraction_rate"]
            episodes.append(epi_summary)
            counts["episodes"] += 1
            counts["rows"] += len(rows)

    overall_axes = {axis: _axis_stats(samples) for axis, samples in axis_samples.items()}
    report = {
        "eval_root": str(eval_root),
        "task_name": task_name,
        "variant_name": results.get("mode", eval_root.name),
        "episodes": episodes,
        "axis_summary": overall_axes,
        "counts": dict(counts),
        "recommendation": {
            axis: overall_axes[axis]["recommended_policy"] for axis in ["x", "y", "z", "yaw"]
        },
    }

    out_json = output_dir / "basin_state_calibration.json"
    out_md = output_dir / "basin_state_calibration.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Basin-state calibration",
        "",
        f"- eval_root: `{eval_root}`",
        f"- task_name: `{task_name}`",
        f"- variant: `{report['variant_name']}`",
        f"- episodes: `{counts['episodes']}`",
        f"- rows: `{counts['rows']}`",
        "",
        "## Axis summary",
        "",
        "| axis | samples | sign_match | action_sign | contraction | proxy_corr | action_corr | scale_ratio | action_scale | policy | reason |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for axis in ["x", "y", "z", "yaw"]:
        s = overall_axes[axis]
        lines.append(
            "| {axis} | {samples} | {sign:.3f} | {act_sign:.3f} | {contr:.3f} | {pcorr:.3f} | {acorr:.3f} | {scale} | {ascale} | {policy} | {reason} |".format(
                axis=axis,
                samples=int(s["num_samples"]),
                sign=float(s["sign_match_rate"]),
                act_sign=float(s["action_sign_match_rate"]),
                contr=float(s["contraction_rate"]),
                pcorr=float(s["proxy_priv_corr"]),
                acorr=float(s["action_priv_corr"]),
                scale="-" if s["scale_ratio_median"] is None else f"{float(s['scale_ratio_median']):.3f}",
                ascale="-" if s["action_scale_ratio_median"] is None else f"{float(s['action_scale_ratio_median']):.3f}",
                policy=str(s["recommended_policy"]),
                reason=str(s["reason"]),
            )
        )
    lines.extend(["", "## Episodes", "", "| episode | gate_step | rows | x_policy | y_policy | z_policy | yaw_policy |", "|---|---:|---:|---|---|---|---|"])
    for epi in episodes:
        lines.append(
            "| {episode} | {gate_step} | {post_rows} | {x_policy} | {y_policy} | {z_policy} | {yaw_policy} |".format(
                episode=epi["episode"],
                gate_step=int(epi["gate_step"]),
                post_rows=int(epi["post_rows"]),
                x_policy=str(epi["x_policy"]),
                y_policy=str(epi["y_policy"]),
                z_policy=str(epi["z_policy"]),
                yaw_policy=str(epi["yaw_policy"]),
            )
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[c2c-v2] Saved calibration report to {out_md}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit basin-state calibration for C2C v2")
    parser.add_argument("--eval_root", type=Path, required=True, help="Evaluation root containing eval_results.json, gripper_traces/, runtime_observations/")
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg", help="Task name for privileged relabel")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (defaults to <eval_root>/../reports/basin_state_calibration)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = eval_root.parent / "reports" / "basin_state_calibration"
    evaluate_root(eval_root, args.task_name, output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
