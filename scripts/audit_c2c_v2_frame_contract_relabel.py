#!/usr/bin/env python3
"""Audit privileged basin relabels produced by the C2C v2 frame contract path."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _corr(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(list(a), dtype=np.float32)
    bb = np.asarray(list(b), dtype=np.float32)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if np.count_nonzero(mask) < 2:
        return 0.0
    aa = aa[mask]
    bb = bb[mask]
    if np.std(aa) <= 1e-9 or np.std(bb) <= 1e-9:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _axis_key(axis: str) -> str:
    return "dyaw" if axis == "yaw" else axis


def _axis_from_mapping(mapping: Mapping[str, Any], axis: str) -> float:
    key = _axis_key(axis)
    alt_key = axis if axis != "yaw" else "yaw"
    if key in mapping:
        return _safe_float(mapping.get(key, 0.0))
    if alt_key in mapping:
        return _safe_float(mapping.get(alt_key, 0.0))
    if axis == "x":
        return _safe_float(mapping.get("dx", 0.0))
    if axis == "y":
        return _safe_float(mapping.get("dy", 0.0))
    if axis == "z":
        return _safe_float(mapping.get("dz", 0.0))
    if axis == "yaw":
        return _safe_float(mapping.get("dyaw", mapping.get("yaw", 0.0)))
    return 0.0


def _value(row: Mapping[str, Any], axis: str, *, source: str = "privileged") -> float:
    key = _axis_key(axis)
    if source == "privileged":
        return _safe_float(row.get(f"privileged_{key}", row.get(f"next_privileged_{key}", 0.0)))
    if source == "next_privileged":
        return _safe_float(row.get(f"next_privileged_{key}", float("nan")))
    if source == "action":
        vec = np.asarray(row.get("local_residual_vs_planner_local_6d", [0.0] * 6), dtype=np.float32).reshape(-1)
        vec = np.pad(vec, (0, max(0, 6 - vec.size)))[:6]
        idx = 0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 5
        return float(vec[idx])
    if source == "proxy":
        proxy = row.get("proxy_local_geometry_error", {}) or {}
        return _axis_from_mapping(proxy, axis)
    if source == "estimated":
        est = row.get("estimated_basin_error", {}) or {}
        return _axis_from_mapping(est, axis)
    if source == "planner":
        vec = np.asarray(row.get("planner_local_delta_6d", [0.0] * 6), dtype=np.float32).reshape(-1)
        vec = np.pad(vec, (0, max(0, 6 - vec.size)))[:6]
        idx = 0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 5
        return float(vec[idx])
    raise KeyError(source)


def _sequence_monotonic_rate(rows: list[dict[str, Any]], axis: str) -> float:
    if len(rows) < 2:
        return 0.0
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_idx", -1))].append(row)
    episode_rates: list[float] = []
    for ep_rows in by_episode.values():
        if len(ep_rows) < 2:
            continue
        vals = [abs(_value(r, axis, source="privileged")) for r in sorted(ep_rows, key=lambda r: int(r.get("step_idx", -1)))]
        if len(vals) < 2:
            continue
        episode_rates.append(float(np.mean([1.0 if vals[i] <= vals[i - 1] + 1e-9 else 0.0 for i in range(1, len(vals))])))
    return float(np.mean(episode_rates)) if episode_rates else 0.0


def _axis_stats(rows: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    proxy = np.asarray([_value(r, axis, source="proxy") for r in rows], dtype=np.float32)
    est = np.asarray([_value(r, axis, source="estimated") for r in rows], dtype=np.float32)
    priv = np.asarray([_value(r, axis, source="privileged") for r in rows], dtype=np.float32)
    action = np.asarray([_value(r, axis, source="action") for r in rows], dtype=np.float32)
    next_priv = np.asarray([_value(r, axis, source="next_privileged") for r in rows], dtype=np.float32)
    planner = np.asarray([_value(r, axis, source="planner") for r in rows], dtype=np.float32)

    finite = np.isfinite(proxy) & np.isfinite(priv)
    finite_est = np.isfinite(est) & np.isfinite(priv)
    finite_action = np.isfinite(action) & np.isfinite(priv)
    finite_next = np.isfinite(next_priv) & np.isfinite(priv)

    sign_mask = finite & (np.abs(proxy) > 1e-6) & (np.abs(priv) > 1e-6)
    action_sign_mask = finite_action & (np.abs(action) > 1e-6) & (np.abs(priv) > 1e-6)
    sign_match = float(np.mean([np.sign(proxy[i]) == np.sign(priv[i]) for i in np.where(sign_mask)[0]])) if np.any(sign_mask) else 0.0
    action_sign_match = float(np.mean([np.sign(action[i]) == np.sign(priv[i]) for i in np.where(action_sign_mask)[0]])) if np.any(action_sign_mask) else 0.0
    contraction = float(np.mean(np.abs(next_priv[finite_next]) <= np.abs(priv[finite_next]) + 1e-9)) if np.any(finite_next) else 0.0
    overshoot = float(np.mean((np.sign(next_priv[finite_next]) != np.sign(priv[finite_next])) & (np.abs(next_priv[finite_next]) >= np.abs(priv[finite_next])))) if np.any(finite_next) else 0.0

    trusted_policy = "trusted_control"
    if not (sign_match >= 0.70 and contraction >= 0.70 and abs(_corr(proxy, priv)) >= 0.25):
        trusted_policy = "diagnostic_only" if sign_match >= 0.40 and contraction >= 0.40 else "abstain"

    return {
        "num_rows": int(len(rows)),
        "sign_match_rate": sign_match,
        "action_sign_match_rate": action_sign_match,
        "contraction_rate": contraction,
        "overshoot_rate": overshoot,
        "monotonic_prefix_rate": _sequence_monotonic_rate(rows, axis),
        "proxy_priv_corr": _corr(proxy, priv),
        "estimated_priv_corr": _corr(est, priv),
        "action_priv_corr": _corr(action, priv),
        "planner_priv_corr": _corr(planner, priv),
        "recommended_policy": trusted_policy,
    }


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return groups


def _plot_overview(report: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    axes = ["x", "y", "z", "yaw"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.reshape(-1)
    for ax, axis in zip(axs, axes):
        stats = report["axis_summary"][axis]
        bars = [stats["sign_match_rate"], stats["contraction_rate"], abs(stats["proxy_priv_corr"]), abs(stats["action_priv_corr"])]
        labels = ["sign", "contract", "proxy", "action"]
        ax.bar(labels, bars, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{axis}: {stats['recommended_policy']}")
    fig.tight_layout()
    fig.savefig(output_dir / "frame_contract_relabel_overview.png", dpi=160)
    plt.close(fig)


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    axis_summary = {axis: _axis_stats(rows, axis) for axis in ["x", "y", "z", "yaw"]}

    by_stage = []
    for key, subset in sorted(_group_rows(rows, ("stage_name",)).items()):
        by_stage.append(
            {
                "stage_name": key[0],
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
            }
        )

    by_skill = []
    for key, subset in sorted(_group_rows(rows, ("skill_name",)).items()):
        by_skill.append(
            {
                "skill_name": key[0],
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
            }
        )

    by_visual = []
    for key, subset in sorted(_group_rows(rows, ("visual_observability_class",)).items()):
        by_visual.append(
            {
                "visual_observability_class": key[0],
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
            }
        )

    by_bucket = []
    for key, subset in sorted(_group_rows(rows, ("failure_bucket",)).items()):
        by_bucket.append(
            {
                "failure_bucket": key[0],
                "axis_summary": {axis: _axis_stats(subset, axis) for axis in ["x", "y", "z", "yaw"]},
            }
        )

    overall = {
        "num_rows": len(rows),
        "near_grasp_rate": float(np.mean([bool(r.get("near_grasp_basin", False)) for r in rows])) if rows else 0.0,
        "close_ready_rate": float(np.mean([bool(r.get("close_ready_basin", False)) for r in rows])) if rows else 0.0,
        "visual_observable_rate": float(np.mean([str(r.get("visual_observability_class", "")) == "visual_observable" for r in rows])) if rows else 0.0,
        "prior_only_rate": float(np.mean([str(r.get("visual_observability_class", "")) == "prior_only" for r in rows])) if rows else 0.0,
    }

    return {
        "overall": overall,
        "axis_summary": axis_summary,
        "by_stage": by_stage,
        "by_skill": by_skill,
        "by_visual_observability": by_visual,
        "by_failure_bucket": by_bucket,
        "runtime_invariants": {
            "uses_privileged_target": False,
            "uses_privileged_runtime": False,
            "uses_privileged_label": True,
            "uses_rlbench_mask_runtime": False,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/frame_contract_relabel"),
    )
    args = ap.parse_args()

    rows = _read_jsonl(args.relabel_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {args.relabel_jsonl}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = audit(rows)
    report["source_jsonl"] = str(args.relabel_jsonl.resolve())

    out_json = output_dir / "frame_contract_audit.json"
    out_md = output_dir / "frame_contract_audit.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _plot_overview(report, output_dir)

    md_lines = [
        "# Frame Contract Audit",
        "",
        f"- source: `{args.relabel_jsonl}`",
        f"- rows: `{len(rows)}`",
        "",
        "## Overall",
        f"- near_grasp_rate: `{report['overall']['near_grasp_rate']:.3f}`",
        f"- close_ready_rate: `{report['overall']['close_ready_rate']:.3f}`",
        f"- visual_observable_rate: `{report['overall']['visual_observable_rate']:.3f}`",
        f"- prior_only_rate: `{report['overall']['prior_only_rate']:.3f}`",
        "",
        "## Axis Summary",
    ]
    for axis, stats in report["axis_summary"].items():
        md_lines.append(
            f"- `{axis}`: policy={stats['recommended_policy']}, sign={stats['sign_match_rate']:.3f}, "
            f"contract={stats['contraction_rate']:.3f}, proxy_corr={stats['proxy_priv_corr']:.3f}, "
            f"action_corr={stats['action_priv_corr']:.3f}, monotonic={stats['monotonic_prefix_rate']:.3f}"
        )
    md_lines.append("")
    md_lines.append("## Failure Buckets")
    for item in report["by_failure_bucket"]:
        bucket = item["failure_bucket"]
        md_lines.append(f"- `{bucket}`")
        for axis, stats in item["axis_summary"].items():
            md_lines.append(
                f"  - `{axis}`: policy={stats['recommended_policy']}, sign={stats['sign_match_rate']:.3f}, "
                f"contract={stats['contraction_rate']:.3f}, proxy_corr={stats['proxy_priv_corr']:.3f}"
            )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(out_json)
    print(out_md)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
