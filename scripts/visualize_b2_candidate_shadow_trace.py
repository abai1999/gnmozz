#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "gripper_traces").is_dir():
        path = path / "gripper_traces"
    return sorted(path.glob("*_gripper_trace.jsonl"))


def parse_focus_episodes(text: str) -> set[str]:
    return {x.strip().zfill(3) for x in text.split(",") if x.strip()}


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def as_bool(row: dict, key: str) -> bool:
    return bool(row.get(key, False))


def episode_id_from_name(name: str) -> str:
    stem = Path(name).stem
    for token in stem.split("_"):
        if token.startswith("ep") and token[2:].isdigit():
            return token[2:].zfill(3)
    return stem


def plot_episode(rows: list[dict], output_path: Path, title: str) -> None:
    steps = [int(r.get("step", i)) for i, r in enumerate(rows)]
    gate = [1.0 if as_bool(r, "b2_candidate_shadow_gate_open") else 0.0 for r in rows]
    nearish = [1.0 if as_bool(r, "b2_candidate_shadow_nearish_runtime") else 0.0 for r in rows]
    keep_forced = [1.0 if as_bool(r, "b2_candidate_shadow_keep_baseline_forced") else 0.0 for r in rows]
    mode = [int(r.get("b2_candidate_shadow_mode", -1)) for r in rows]
    delta = [as_float(r, "b2_candidate_shadow_regret_delta") for r in rows]
    pred_regret = [as_float(r, "b2_candidate_shadow_pred_regret") for r in rows]
    base_regret = [as_float(r, "b2_candidate_shadow_baseline_regret") for r in rows]
    truth_xy = [abs(as_float(r, "teacher_truth_basin_xy", math.nan)) for r in rows]
    truth_z = [abs(as_float(r, "teacher_truth_basin_z", math.nan)) for r in rows]
    truth_yaw = [abs(as_float(r, "teacher_truth_basin_yaw", math.nan)) for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(steps, truth_xy, label="teacher_xy", color="#1f77b4")
    ax.plot(steps, truth_z, label="teacher_z", color="#2ca02c")
    ax.plot(steps, truth_yaw, label="teacher_yaw", color="#d62728")
    ax.set_ylabel("Teacher Basin")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, base_regret, label="baseline_regret", color="#7f7f7f")
    ax.plot(steps, pred_regret, label="pred_regret", color="#9467bd")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("Regret")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    colors = []
    for m in mode:
        if m == 2:
            colors.append("#d62728")
        elif m == 0:
            colors.append("#1f77b4")
        else:
            colors.append("#7f7f7f")
    ax.bar(steps, [0.8 if m >= 0 else 0.0 for m in mode], color=colors, width=1.0, alpha=0.7, label="mode")
    ax.plot(steps, gate, color="#17becf", linewidth=1.0, label="gate_open")
    ax.plot(steps, nearish, color="#ff7f0e", linewidth=1.0, label="nearish_runtime")
    ax.plot(steps, keep_forced, color="#2ca02c", linewidth=1.0, label="keep_baseline_forced")
    ax.set_ylabel("Mode/Gate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    bar_colors = ["#2ca02c" if math.isfinite(v) and v >= 0.0 else "#d62728" for v in delta]
    ax.bar(steps, [0.0 if not math.isfinite(v) else v for v in delta], color=bar_colors, width=1.0, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Delta")
    ax.set_xlabel("Step")
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--focus_episodes", type=str, default="18,34,45")
    args = parser.parse_args()

    focus = parse_focus_episodes(args.focus_episodes)
    for path in find_trace_files(args.trace_dir):
        ep = episode_id_from_name(path.name)
        if focus and ep not in focus:
            continue
        rows = load_jsonl(path)
        plot_episode(rows, args.output_dir / f"ep{ep}_b2_shadow_diagnostics.png", f"B2 Shadow Diagnostics ep{ep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
