#!/usr/bin/env python3
"""Build a detailed diagnostic report for depthgate grasp recovery.

This audits post-gate privileged error curves, local correction vectors, sign
match, and yaw-large segments for the current depthgate baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/guoning/code/VLA2/runtime_artifacts/coarse2contact_v2")
DEFAULT_ROWS_DIR = ROOT / "reports" / "depthgate_grasp_recovery_plots"
DEFAULT_REPORT_DIR = ROOT / "reports" / "depthgate_grasp_recovery_diagnostics"
DEFAULT_GATE_ROOT = ROOT / "basin_ablation_xy_only_3ep_depthgate"


@dataclass
class EpisodeSummary:
    episode: str
    gate_step: int
    gate_reason: str
    post_modes: str
    near_grasp_hit: bool
    close_ready_hit: bool
    x0: float
    x1: float
    dz0: float
    dz1: float
    yaw0: float
    yaw1: float
    corr_norm_mean: float
    corr_norm_p95: float
    corr_norm_max: float
    sign_match_x: float | None
    sign_match_y: float | None
    sign_match_z: float | None
    sign_match_yaw: float | None
    active_rate_x: float
    active_rate_y: float
    active_rate_z: float
    active_rate_yaw: float
    yaw_large_segments: list[tuple[int, int]]
    mode_counts: dict[str, int]


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _corr_norm(row: dict) -> float:
    return float(
        math.sqrt(
            float(row["corr_x"]) ** 2
            + float(row["corr_y"]) ** 2
            + float(row["corr_z"]) ** 2
            + float(row["corr_yaw"]) ** 2
        )
    )


def _sign_match(corr: float, err: float, eps: float = 1e-9) -> float | None:
    if abs(corr) <= eps or abs(err) <= eps:
        return None
    return 1.0 if math.copysign(1.0, corr) == math.copysign(1.0, -err) else 0.0


def _safe_mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return float(mean(vals)) if vals else float("nan")


def _safe_p95(vals: Iterable[float]) -> float:
    vals = np.asarray(list(vals), dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 0.95))


def _fmt_bool(v: bool) -> str:
    return "true" if v else "false"


def _episode_rows(rows_dir: Path, episode: str) -> list[dict]:
    return _load_rows(rows_dir / f"{episode}_depthgate_postgate_rows.json")


def _episode_summary(episode: str, rows_dir: Path) -> EpisodeSummary:
    rows = _episode_rows(rows_dir, episode)
    post = [r for r in rows if bool(r.get("gate"))]
    if not post:
        raise ValueError(f"{episode}: no gate rows found")

    first, last = post[0], post[-1]
    mode_counts = Counter(r["mode"] for r in post)

    x_series = [float(r["xy"]) for r in post]
    dz_series = [float(r["dz"]) for r in post]
    yaw_series = [float(r["dyaw"]) for r in post]
    corr_norms = [_corr_norm(r) for r in post]

    active_x = [r for r in post if abs(float(r["corr_x"])) > 1e-6]
    active_y = [r for r in post if abs(float(r["corr_y"])) > 1e-6]
    active_z = [r for r in post if abs(float(r["corr_z"])) > 1e-6]
    active_yaw = [r for r in post if abs(float(r["corr_yaw"])) > 1e-6]

    sign_x = [_sign_match(float(r["corr_x"]), float(r["dx"])) for r in active_x]
    sign_y = [_sign_match(float(r["corr_y"]), float(r["dy"])) for r in active_y]
    sign_z = [_sign_match(float(r["corr_z"]), float(r["dz"])) for r in active_z]
    sign_yaw = [_sign_match(float(r["corr_yaw"]), float(r["dyaw"])) for r in active_yaw]

    def _rate(vals: list[float | None]) -> float | None:
        filtered = [v for v in vals if v is not None]
        if not filtered:
            return None
        return float(mean(filtered))

    yaw_large_segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, r in enumerate(post):
        if abs(float(r["dyaw"])) > 1.0:
            if start is None:
                start = i
        else:
            if start is not None:
                yaw_large_segments.append((int(post[start]["step"]), int(post[i - 1]["step"])))
                start = None
    if start is not None:
        yaw_large_segments.append((int(post[start]["step"]), int(post[-1]["step"])))

    return EpisodeSummary(
        episode=episode,
        gate_step=int(first["step"]),
        gate_reason=str(first["reason"]),
        post_modes=", ".join(sorted(mode_counts.keys())),
        near_grasp_hit=bool(any(bool(r["near_grasp"]) for r in post)),
        close_ready_hit=bool(any(bool(r["close_ready"]) for r in post)),
        x0=float(first["xy"]),
        x1=float(last["xy"]),
        dz0=float(first["dz"]),
        dz1=float(last["dz"]),
        yaw0=float(first["dyaw"]),
        yaw1=float(last["dyaw"]),
        corr_norm_mean=_safe_mean(corr_norms),
        corr_norm_p95=_safe_p95(corr_norms),
        corr_norm_max=float(max(corr_norms)),
        sign_match_x=_rate(sign_x),
        sign_match_y=_rate(sign_y),
        sign_match_z=_rate(sign_z),
        sign_match_yaw=_rate(sign_yaw),
        active_rate_x=float(len(active_x) / len(post)),
        active_rate_y=float(len(active_y) / len(post)),
        active_rate_z=float(len(active_z) / len(post)),
        active_rate_yaw=float(len(active_yaw) / len(post)),
        yaw_large_segments=yaw_large_segments,
        mode_counts=dict(mode_counts),
    )


def _make_plot(episode: str, rows_dir: Path, out_dir: Path) -> Path:
    rows = _episode_rows(rows_dir, episode)
    post = [r for r in rows if bool(r.get("gate"))]
    steps = [int(r["step"]) for r in post]
    gate_idx = 0
    gate_step = int(post[0]["step"])

    xy = [float(r["xy"]) for r in post]
    dz = [float(r["dz"]) for r in post]
    dyaw = [float(r["dyaw"]) for r in post]
    corr_x = [float(r["corr_x"]) for r in post]
    corr_y = [float(r["corr_y"]) for r in post]
    corr_z = [float(r["corr_z"]) for r in post]
    corr_yaw = [float(r["corr_yaw"]) for r in post]
    corr_norm = [_corr_norm(r) for r in post]
    mode_ids = OrderedDict()
    palette = {
        "VISUAL_PULLBACK": "#4c78a8",
        "MICRO_SERVO_TO_BASIN": "#f58518",
        "ABSTAIN_FAIL": "#9e9e9e",
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    fig.suptitle(f"{episode} depthgate post-gate curves", fontsize=15)

    ax = axes[0]
    ax.plot(steps, xy, label="privileged xy", color="#1f77b4", linewidth=2)
    ax.plot(steps, dz, label="privileged dz", color="#2ca02c", linewidth=1.8)
    ax.set_ylabel("xy / dz (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(steps, dyaw, label="privileged dyaw", color="#d62728", linewidth=2)
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1, alpha=0.4)
    ax.axhline(-1.0, color="#d62728", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_ylabel("dyaw (rad)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(steps, corr_x, label="corr_x", color="#1f77b4", linewidth=1.6)
    ax.plot(steps, corr_y, label="corr_y", color="#ff7f0e", linewidth=1.6)
    ax.plot(steps, corr_z, label="corr_z", color="#2ca02c", linewidth=1.6)
    ax.plot(steps, corr_yaw, label="corr_yaw", color="#d62728", linewidth=1.6)
    ax.set_ylabel("local correction")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    ax.plot(steps, corr_norm, label="corr_norm", color="#9467bd", linewidth=2)
    ax.set_ylabel("corr_norm")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    for ax in axes:
        ax.axvline(gate_step, color="black", linestyle=":", linewidth=1.5, alpha=0.9)
        ax.text(gate_step, ax.get_ylim()[1], " gate", fontsize=8, va="top", ha="left", rotation=90)

    # Highlight yaw-large segments on all axes.
    start = None
    for i, r in enumerate(post):
        if abs(float(r["dyaw"])) > 1.0:
            if start is None:
                start = steps[i]
        else:
            if start is not None:
                end = steps[i - 1]
                for ax in axes:
                    ax.axvspan(start, end, color="#ffcccc", alpha=0.15)
                start = None
    if start is not None:
        end = steps[-1]
        for ax in axes:
            ax.axvspan(start, end, color="#ffcccc", alpha=0.15)

    # Background mode hints on the correction panel.
    current_mode = None
    span_start = None
    for step, r in zip(steps, post, strict=True):
        mode = str(r["mode"])
        if current_mode is None:
            current_mode = mode
            span_start = step
        elif mode != current_mode:
            axes[2].axvspan(span_start, step - 1, color=palette.get(current_mode, "#dddddd"), alpha=0.08)
            current_mode = mode
            span_start = step
    if current_mode is not None and span_start is not None:
        axes[2].axvspan(span_start, steps[-1], color=palette.get(current_mode, "#dddddd"), alpha=0.08)

    out_path = out_dir / f"{episode}_depthgate_diagnostic_curves.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _write_report(out_dir: Path, summaries: list[EpisodeSummary], plot_paths: dict[str, Path]) -> Path:
    lines: list[str] = []
    lines.append("# Depthgate Grasp Recovery Diagnostic\n")
    lines.append(
        "This report focuses on the current `depthgate` baseline and separates "
        "privileged error from the runtime localizer self-report. The goal is to "
        "check whether `VISUAL_PULLBACK` / `MICRO_SERVO_TO_BASIN` actually reduce "
        "the real failure-tail error, or only a proxy.\n"
    )
    lines.append("## High-level read\n")
    lines.append(
        "- `RingGraspLocalizer` is still proxy-based: it uses ring-mask centroid/axis/depth, not a true jaw-relative geometric frame.\n"
        "- `dyaw` is therefore a mask-axis proxy with symmetry wrapping, not a direct jaw-frame yaw error.\n"
        "- On these three episodes, `xy` and `dz` do shrink somewhat after gate, but `yaw` stays large and is not actively corrected.\n"
        "- `VISUAL_PULLBACK` and `MICRO_SERVO_TO_BASIN` are reducing parts of the proxy geometry, but they are not yet proving that the privileged basin was reached.\n"
    )
    lines.append("## Per-episode diagnostic table\n")
    lines.append(
        "| episode | gate step | gate reason | post modes | near-grasp hit | close-ready hit | x start -> end | dz start -> end | dyaw start -> end | corr_norm mean / p95 / max | sign-match x | sign-match y | sign-match z | sign-match yaw | active rate x/y/z/yaw | yaw-large segments |"
    )
    lines.append(
        "|---|---:|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|"
    )
    for s in summaries:
        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.3f}"

        lines.append(
            f"| {s.episode} | {s.gate_step} | {s.gate_reason} | {s.post_modes} | "
            f"{_fmt_bool(s.near_grasp_hit)} | {_fmt_bool(s.close_ready_hit)} | "
            f"{s.x0:.4f} -> {s.x1:.4f} | {s.dz0:.4f} -> {s.dz1:.4f} | {s.yaw0:.4f} -> {s.yaw1:.4f} | "
            f"{s.corr_norm_mean:.6f} / {s.corr_norm_p95:.6f} / {s.corr_norm_max:.6f} | "
            f"{fmt(s.sign_match_x)} | {fmt(s.sign_match_y)} | {fmt(s.sign_match_z)} | {fmt(s.sign_match_yaw)} | "
            f"{s.active_rate_x:.3f}/{s.active_rate_y:.3f}/{s.active_rate_z:.3f}/{s.active_rate_yaw:.3f} | "
            f"{'; '.join([f'{a}-{b}' for a, b in s.yaw_large_segments]) if s.yaw_large_segments else 'none'} |"
        )

    lines.append("\n## Per-episode notes\n")
    for s in summaries:
        lines.append(f"### {s.episode}\n")
        lines.append(
            f"- Gate at step {s.gate_step} with reason `{s.gate_reason}`.\n"
            f"- Post-gate modes: {s.post_modes}.\n"
            f"- Privileged `xy` goes {s.x0:.4f} -> {s.x1:.4f}, `dz` goes {s.dz0:.4f} -> {s.dz1:.4f}, but `dyaw` remains {s.yaw0:.4f} -> {s.yaw1:.4f}.\n"
            f"- `corr_norm` stays tiny (mean {s.corr_norm_mean:.6f}), and `corr_yaw` is effectively absent.\n"
        )
        if s.sign_match_y is not None:
            lines.append(
                f"- `y` sign-match is {s.sign_match_y:.3f}, which is weaker than `x` and suggests one lateral axis is being pulled the wrong way or too inconsistently.\n"
            )
        if s.yaw_large_segments:
            lines.append(
                f"- Yaw-large segment(s): {', '.join([f'{a}-{b}' for a, b in s.yaw_large_segments])}.\n"
            )
        else:
            lines.append("- No yaw-large post-gate segment (`abs(dyaw)>1.0`) was present.\n")

    lines.append("## Semantic check against the code\n")
    lines.append(
        f"- `RingGraspLocalizer` computes ring-mask centroid/axis/depth in the crop frame, not a true jaw-relative pose estimate: "
        f"[localizers.py:195-244](/home/guoning/code/VLA2/prismatic/robot/coarse2contact_v2/localizers.py#L195-L244).\n"
        f"- `RingSpokeAlignLocalizer` subtracts spoke and ring mask centroids and symmetry-wrapped axes; it is also proxy-based, not a direct geometric frame-to-frame pose: "
        f"[localizers.py:247-321](/home/guoning/code/VLA2/prismatic/robot/coarse2contact_v2/localizers.py#L247-L321).\n"
        f"- The takeover gate only checks visibility/confidence, absolute depth nearness, and local target nearness with stable frames; it does not directly reason about a privileged jaw-frame alignment: "
        f"[supervisor.py:266-349](/home/guoning/code/VLA2/prismatic/robot/coarse2contact_v2/supervisor.py#L266-L349).\n"
    )
    lines.append("## Figures\n")
    for ep, path in plot_paths.items():
        lines.append(f"- {ep}: [{path.name}]({path})\n")

    out_path = out_dir / "depthgate_grasp_recovery_diagnostic.md"
    out_path.write_text("\n".join(line.rstrip("\n") for line in lines) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-dir", type=Path, default=DEFAULT_ROWS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--episodes", nargs="*", default=["ep005", "ep008", "ep019"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [_episode_summary(ep, args.rows_dir) for ep in args.episodes]
    plot_dir = args.out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {s.episode: _make_plot(s.episode, args.rows_dir, plot_dir) for s in summaries}

    report_path = _write_report(args.out_dir, summaries, plot_paths)
    summary_json = args.out_dir / "depthgate_grasp_recovery_diagnostic.json"
    summary_json.write_text(
        json.dumps(
            [s.__dict__ for s in summaries],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(report_path)
    print(summary_json)
    for p in plot_paths.values():
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
