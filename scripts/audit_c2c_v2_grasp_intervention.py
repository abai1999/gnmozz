#!/usr/bin/env python3
"""Audit grasp-only intervention probes for C2C v2 basin pullback.

This script is intentionally narrow:
it consumes the probe trace produced by `evaluate_c2c_v2_rlbench.py` when
`--c2c_grasp_probe_policy replay_oracle_xy` is enabled, and reports whether
the inserted bounded xy step actually contracts the privileged grasp basin
error in live dynamics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def _safe_vec(value: Any, *, length: int, default: float = float("nan")) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), constant_values=default)
    return arr[:length].astype(np.float32)


def _wilson_lower_bound(successes: int, n: int, *, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = float(successes) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = phat + (z * z) / (2.0 * float(n))
    margin = z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * float(n))) / float(n))
    return float(max((center - margin) / denom, 0.0))


def _episode_from_trace_path(path: Path) -> int:
    stem = path.stem
    for token in stem.split("_"):
        if token.startswith("ep") and token[2:].isdigit():
            return int(token[2:])
    return -1


def _load_trace_rows(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        episode_idx = _episode_from_trace_path(trace_path)
        for row in _read_jsonl(trace_path):
            row = dict(row)
            row.setdefault("episode_idx", int(episode_idx))
            row.setdefault("source_trace_path", str(trace_path))
            rows.append(row)
    rows.sort(key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step", r.get("step_idx", -1)))))
    return rows


def _trace_row_error(row: Mapping[str, Any], prefix: str) -> np.ndarray:
    return _safe_vec(row.get(prefix), length=4, default=float("nan"))


def _xy_norm(vec: Iterable[float]) -> float:
    arr = np.asarray(list(vec), dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return float("nan")
    return float(np.hypot(float(arr[0]), float(arr[1])))


def _axis_abs_contraction_rate(rows: list[dict[str, Any]], axis: str) -> float:
    if not rows:
        return 0.0
    idx = 0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 3
    good = []
    for row in rows:
        pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
        post = _trace_row_error(row, "grasp_probe_post_true_error_t")
        if not np.all(np.isfinite(pre[:4])) or not np.all(np.isfinite(post[:4])):
            continue
        good.append(bool(abs(float(post[idx])) <= abs(float(pre[idx])) + 1.0e-9))
    return float(np.mean(good)) if good else 0.0


def _xy_contraction_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    good = []
    for row in rows:
        pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
        post = _trace_row_error(row, "grasp_probe_post_true_error_t")
        if not np.all(np.isfinite(pre[:4])) or not np.all(np.isfinite(post[:4])):
            continue
        good.append(bool(_xy_norm(pre[:2]) > _xy_norm(post[:2]) + 1.0e-9))
    return float(np.mean(good)) if good else 0.0


def _mean_step_delta(rows: list[dict[str, Any]], axis: str) -> float:
    if not rows:
        return 0.0
    idx = 0 if axis == "x" else 1 if axis == "y" else 2 if axis == "z" else 3
    deltas = []
    for row in rows:
        pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
        post = _trace_row_error(row, "grasp_probe_post_true_error_t")
        if not np.all(np.isfinite(pre[:4])) or not np.all(np.isfinite(post[:4])):
            continue
        deltas.append(float(post[idx] - pre[idx]))
    return float(np.mean(deltas)) if deltas else 0.0


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(r.get(key, False)) for r in rows]))


def _row_group_value(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    return str(value)


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(_row_group_value(row, key) for key in keys)].append(row)
    return groups


def _probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("grasp_probe_policy", "off")) != "off"]


def _active_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("grasp_probe_active", False))]


def _plot_episode(ep_tag: str, rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    steps = [int(r.get("step", r.get("step_idx", -1))) for r in rows]
    pre = np.asarray([_trace_row_error(r, "grasp_probe_pre_true_error_t") for r in rows], dtype=np.float32)
    post = np.asarray([_trace_row_error(r, "grasp_probe_post_true_error_t") for r in rows], dtype=np.float32)
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    labels = ["dx", "dy", "dz", "dyaw"]
    for idx, ax in enumerate(axes[:4]):
        ax.plot(steps, pre[:, idx], label=f"pre_{labels[idx]}", linewidth=1.5)
        ax.plot(steps, post[:, idx], label=f"post_{labels[idx]}", linewidth=1.5)
        active_steps = [s for s, r in zip(steps, rows) if bool(r.get("grasp_probe_active", False))]
        active_post = [float(_trace_row_error(r, "grasp_probe_post_true_error_t")[idx]) for r in rows if bool(r.get("grasp_probe_active", False))]
        if active_steps:
            ax.scatter(active_steps, active_post, s=18, alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(labels[idx])
        ax.legend(loc="upper right")
    xy_pre = np.asarray([_xy_norm(r.get("grasp_probe_pre_true_error_t", [np.nan, np.nan])) for r in rows], dtype=np.float32)
    xy_post = np.asarray([_xy_norm(r.get("grasp_probe_post_true_error_t", [np.nan, np.nan])) for r in rows], dtype=np.float32)
    axes[4].plot(steps, xy_pre, label="pre_xy", linewidth=1.5)
    axes[4].plot(steps, xy_post, label="post_xy", linewidth=1.5)
    axes[4].set_ylabel("xy_norm")
    axes[4].set_xlabel("step")
    axes[4].legend(loc="upper right")
    fig.suptitle(f"{ep_tag} grasp probe intervention")
    fig.tight_layout()
    fig.savefig(output_dir / f"{ep_tag}_grasp_probe_intervention.png", dpi=160)
    plt.close(fig)


def _bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = _active_probe_rows(rows)
    prior_only = [r for r in rows if _row_group_value(r, "grasp_probe_visibility_bucket", "prior_only") == "prior_only"]
    visual = [r for r in rows if _row_group_value(r, "grasp_probe_visibility_bucket", "") == "visual_observable"]
    partial = [r for r in rows if _row_group_value(r, "grasp_probe_visibility_bucket", "") == "partial_observable"]
    active_xy = [r for r in active if np.all(np.isfinite(_trace_row_error(r, "grasp_probe_pre_true_error_t")[:2])) and np.all(np.isfinite(_trace_row_error(r, "grasp_probe_post_true_error_t")[:2]))]
    xy_contracted = [bool(_xy_norm(_trace_row_error(r, "grasp_probe_post_true_error_t")[:2]) < _xy_norm(_trace_row_error(r, "grasp_probe_pre_true_error_t")[:2]) - 1.0e-9) for r in active_xy]
    active_count = len(active)
    contracted_count = int(np.count_nonzero(xy_contracted))
    active_xy_rate = float(np.mean(xy_contracted)) if xy_contracted else 0.0
    micro_after = float(np.mean([bool(r.get("grasp_probe_micro_entry_ready_after", False)) for r in active])) if active else 0.0
    near_after = float(np.mean([bool(r.get("grasp_probe_near_grasp_after", False)) for r in active])) if active else 0.0
    close_after = float(np.mean([bool(r.get("grasp_probe_close_ready_after", False)) for r in active])) if active else 0.0
    overshoot = float(np.mean([bool(r.get("grasp_probe_overshoot", False)) for r in active])) if active else 0.0
    prior_abstain = float(np.mean([_row_group_value(r, "grasp_probe_reason", "") == "prior_only_abstain" for r in prior_only])) if prior_only else 0.0

    def _hist_key(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "+".join(str(x) for x in value)
        return str(value)

    return {
        "count": int(len(rows)),
        "active_count": int(active_count),
        "xy_contraction_rate": float(active_xy_rate),
        "xy_contraction_lower_ci": float(_wilson_lower_bound(contracted_count, len(active_xy))),
        "micro_entry_ready_after_rate": micro_after,
        "near_grasp_after_rate": near_after,
        "close_ready_after_rate": close_after,
        "overshoot_rate": overshoot,
        "prior_only_abstain_rate": prior_abstain,
        "mean_delta_dx": _mean_step_delta(active, "x"),
        "mean_delta_dy": _mean_step_delta(active, "y"),
        "mean_delta_dz": _mean_step_delta(active, "z"),
        "mean_delta_dyaw": _mean_step_delta(active, "yaw"),
        "axis_abs_contraction_rate": {
            "x": _axis_abs_contraction_rate(active, "x"),
            "y": _axis_abs_contraction_rate(active, "y"),
            "z": _axis_abs_contraction_rate(active, "z"),
            "yaw": _axis_abs_contraction_rate(active, "yaw"),
        },
        "visual_observable_count": int(len(visual)),
        "partial_observable_count": int(len(partial)),
        "prior_only_count": int(len(prior_only)),
        "active_xy_rows": int(len(active_xy)),
        "active_gate_axes_hist": dict(Counter(_hist_key(r.get("grasp_probe_control_gate_axes", [])) for r in active)),
        "active_pullback_axes_hist": dict(Counter(_hist_key(r.get("grasp_probe_pullback_ready_axes", [])) for r in active)),
        "reason_counts": dict(Counter(_row_group_value(r, "grasp_probe_reason", "") for r in rows)),
    }


def _active_xy_contracted(row: Mapping[str, Any]) -> bool:
    pre = _trace_row_error(row, "grasp_probe_pre_true_error_t")
    post = _trace_row_error(row, "grasp_probe_post_true_error_t")
    if not np.all(np.isfinite(pre[:2])) or not np.all(np.isfinite(post[:2])):
        return False
    return bool(_xy_norm(post[:2]) < _xy_norm(pre[:2]) - 1.0e-9)


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probe_rows = _probe_rows(rows)
    active_rows = _active_probe_rows(rows)
    by_bucket: list[dict[str, Any]] = []
    for key, subset in sorted(_group_rows(probe_rows, ("failure_bucket",)).items()):
        by_bucket.append({"failure_bucket": key[0], **_bucket_summary(subset)})

    by_visual: list[dict[str, Any]] = []
    for key, subset in sorted(_group_rows(probe_rows, ("grasp_probe_visibility_bucket",)).items()):
        by_visual.append({"grasp_probe_visibility_bucket": key[0], **_bucket_summary(subset)})

    by_episode: list[dict[str, Any]] = []
    episode_groups = list(_group_rows(probe_rows, ("episode_idx",)).items())
    episode_groups.sort(key=lambda item: int(item[0][0]) if str(item[0][0]).lstrip("-").isdigit() else -1)
    for key, subset in episode_groups:
        by_episode.append({"episode_idx": int(key[0]) if str(key[0]).lstrip("-").isdigit() else -1, **_bucket_summary(subset)})

    overall = {
        "num_rows": int(len(rows)),
        "probe_rows": int(len(probe_rows)),
        "active_rows": int(len(active_rows)),
        "xy_contraction_rate": float(np.mean([_active_xy_contracted(r) for r in active_rows])) if active_rows else 0.0,
        "xy_contraction_lower_ci": float(_wilson_lower_bound(
            int(np.count_nonzero([_active_xy_contracted(r) for r in active_rows])),
            len(active_rows),
        )),
        "micro_entry_ready_after_rate": float(np.mean([bool(r.get("grasp_probe_micro_entry_ready_after", False)) for r in active_rows])) if active_rows else 0.0,
        "near_grasp_after_rate": float(np.mean([bool(r.get("grasp_probe_near_grasp_after", False)) for r in active_rows])) if active_rows else 0.0,
        "close_ready_after_rate": float(np.mean([bool(r.get("grasp_probe_close_ready_after", False)) for r in active_rows])) if active_rows else 0.0,
        "overshoot_rate": float(np.mean([bool(r.get("grasp_probe_overshoot", False)) for r in active_rows])) if active_rows else 0.0,
        "prior_only_abstain_rate": float(np.mean([_row_group_value(r, "grasp_probe_reason", "") == "prior_only_abstain" for r in probe_rows if _row_group_value(r, "grasp_probe_visibility_bucket", "prior_only") == "prior_only"])) if probe_rows else 0.0,
        "reacquire_rate": float(np.mean([_row_group_value(r, "grasp_probe_reason", "") == "prior_only_abstain" or _row_group_value(r, "grasp_probe_reason", "") == "inactive" for r in probe_rows])) if probe_rows else 0.0,
        "axis_abs_contraction_rate": {
            "x": _axis_abs_contraction_rate(active_rows, "x"),
            "y": _axis_abs_contraction_rate(active_rows, "y"),
            "z": _axis_abs_contraction_rate(active_rows, "z"),
            "yaw": _axis_abs_contraction_rate(active_rows, "yaw"),
        },
        "mean_delta": {
            "dx": _mean_step_delta(active_rows, "x"),
            "dy": _mean_step_delta(active_rows, "y"),
            "dz": _mean_step_delta(active_rows, "z"),
            "dyaw": _mean_step_delta(active_rows, "yaw"),
        },
        "reason_counts": dict(Counter(_row_group_value(r, "grasp_probe_reason", "") for r in probe_rows)),
    }

    return {
        "overall": overall,
        "by_failure_bucket": by_bucket,
        "by_visibility_bucket": by_visual,
        "by_episode": by_episode,
        "runtime_invariants": {
            "uses_privileged_target": False,
            "uses_privileged_runtime": False,
            "uses_privileged_label_for_eval": True,
            "uses_rlbench_mask_runtime": False,
        },
    }


def _plot_overview(report: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["xy_contract", "micro_ready", "near_grasp", "close_ready", "overshoot"]
    bars = [
        report["overall"]["xy_contraction_rate"],
        report["overall"]["micro_entry_ready_after_rate"],
        report["overall"]["near_grasp_after_rate"],
        report["overall"]["close_ready_after_rate"],
        report["overall"]["overshoot_rate"],
    ]
    ax.bar(labels, bars, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("C2C grasp probe intervention summary")
    fig.tight_layout()
    fig.savefig(output_dir / "grasp_probe_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_intervention"),
    )
    args = ap.parse_args()

    trace_dir = args.trace_dir.resolve()
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing trace_dir: {trace_dir}")

    rows = _load_trace_rows(trace_dir)
    if not rows:
        raise RuntimeError(f"No trace rows found in {trace_dir}")
    probe_rows = _probe_rows(rows)
    if not probe_rows:
        raise RuntimeError(
            f"No grasp probe rows found in {trace_dir}. "
            "Re-run evaluate_c2c_v2_rlbench.py with --c2c_grasp_probe_policy replay_oracle_xy."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    report = audit(rows)
    report["source_trace_dir"] = str(trace_dir)
    report["num_trace_files"] = int(len(list(trace_dir.glob("ep*_gripper_trace.jsonl"))))

    out_json = output_dir / "grasp_probe_intervention_audit.json"
    out_md = output_dir / "grasp_probe_intervention_audit.md"
    out_rows = output_dir / "grasp_probe_intervention_rows.jsonl"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _plot_overview(report, output_dir)

    with open(out_rows, "w", encoding="utf-8") as handle:
        for row in probe_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            ep_tag = f"ep{int(row.get('episode_idx', -1)):03d}"
    for key, subset in sorted(_group_rows(probe_rows, ("episode_idx",)).items()):
        ep_tag = f"ep{int(key[0]):03d}" if str(key[0]).lstrip("-").isdigit() else f"episode_{key[0]}"
        _plot_episode(ep_tag, subset, plot_dir)

    md_lines = [
        "# Grasp Intervention Audit",
        "",
        f"- source_trace_dir: `{trace_dir}`",
        f"- trace_files: `{report['num_trace_files']}`",
        f"- rows: `{report['overall']['num_rows']}`",
        f"- probe_rows: `{report['overall']['probe_rows']}`",
        f"- active_rows: `{report['overall']['active_rows']}`",
        "",
        "## Overall",
        f"- xy_contraction_rate: `{report['overall']['xy_contraction_rate']:.3f}`",
        f"- xy_contraction_lower_ci: `{report['overall']['xy_contraction_lower_ci']:.3f}`",
        f"- micro_entry_ready_after_rate: `{report['overall']['micro_entry_ready_after_rate']:.3f}`",
        f"- near_grasp_after_rate: `{report['overall']['near_grasp_after_rate']:.3f}`",
        f"- close_ready_after_rate: `{report['overall']['close_ready_after_rate']:.3f}`",
        f"- overshoot_rate: `{report['overall']['overshoot_rate']:.3f}`",
        f"- prior_only_abstain_rate: `{report['overall']['prior_only_abstain_rate']:.3f}`",
        "",
        "## Axis Contraction",
    ]
    for axis, value in report["overall"]["axis_abs_contraction_rate"].items():
        md_lines.append(f"- `{axis}`: `{float(value):.3f}`")
    md_lines.append("")
    md_lines.append("## Failure Buckets")
    for item in report["by_failure_bucket"]:
        md_lines.append(f"- `{item['failure_bucket']}`")
        md_lines.append(f"  - count: `{item['count']}`")
        md_lines.append(f"  - active_count: `{item['active_count']}`")
        md_lines.append(f"  - xy_contraction_rate: `{item['xy_contraction_rate']:.3f}`")
        md_lines.append(f"  - xy_contraction_lower_ci: `{item['xy_contraction_lower_ci']:.3f}`")
        md_lines.append(f"  - micro_entry_ready_after_rate: `{item['micro_entry_ready_after_rate']:.3f}`")
        md_lines.append(f"  - near_grasp_after_rate: `{item['near_grasp_after_rate']:.3f}`")
        md_lines.append(f"  - close_ready_after_rate: `{item['close_ready_after_rate']:.3f}`")
        md_lines.append(f"  - overshoot_rate: `{item['overshoot_rate']:.3f}`")
        md_lines.append(f"  - prior_only_abstain_rate: `{item['prior_only_abstain_rate']:.3f}`")
        md_lines.append(
            "  - axis_abs_contraction_rate: "
            + ", ".join(f"{axis}={float(val):.3f}" for axis, val in item["axis_abs_contraction_rate"].items())
        )
    md_lines.append("")
    md_lines.append("## Visibility Buckets")
    for item in report["by_visibility_bucket"]:
        md_lines.append(f"- `{item['grasp_probe_visibility_bucket']}`")
        md_lines.append(f"  - xy_contraction_rate: `{item['xy_contraction_rate']:.3f}`")
        md_lines.append(f"  - micro_entry_ready_after_rate: `{item['micro_entry_ready_after_rate']:.3f}`")
        md_lines.append(f"  - near_grasp_after_rate: `{item['near_grasp_after_rate']:.3f}`")
        md_lines.append(f"  - close_ready_after_rate: `{item['close_ready_after_rate']:.3f}`")
        md_lines.append(f"  - overshoot_rate: `{item['overshoot_rate']:.3f}`")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(out_json)
    print(out_md)
    print(out_rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
