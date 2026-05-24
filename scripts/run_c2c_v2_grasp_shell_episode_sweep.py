#!/usr/bin/env python3
"""Evaluate a candidate episode sweep for C2C v2 grasp shell collection.

This driver keeps the current runtime contract unchanged and uses the existing
grasp probe + audit pipeline to answer a narrower question:

    Which episode / failure-bucket pairs actually produce near-basin shell rows?

It runs `scripts/evaluate_c2c_v2_rlbench.py` on a candidate episode sweep,
audits the resulting trace dir with `scripts/audit_c2c_v2_grasp_intervention.py`,
and writes a compact ranked summary of shell-rich episodes and buckets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/home/guoning/my_conda_envs/vla-adapter/bin/python")


def _python_bin() -> str:
    env_python = os.environ.get("PYTHON_BIN")
    if env_python:
        return env_python
    if DEFAULT_PYTHON.is_file():
        return str(DEFAULT_PYTHON)
    return sys.executable


def _parse_csv_ints(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0 or len(values) <= chunk_size:
        return [list(values)]
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def _resolve_episode_indices(args: argparse.Namespace) -> list[int]:
    if args.episode_indices:
        return _parse_csv_ints(args.episode_indices)
    start = int(args.episode_start)
    end = int(args.episode_end)
    step = int(args.episode_step)
    if step <= 0:
        raise ValueError("--episode_step must be positive")
    if end < start:
        raise ValueError("--episode_end must be >= --episode_start")
    return list(range(start, end + 1, step))


def _build_env(*, gpu_id: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    env["HF_HOME"] = env.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    env["HUGGINGFACE_HUB_CACHE"] = env.get("HUGGINGFACE_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
    project_root = str(ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing}" if existing else project_root
    coppeliasim_root = os.path.expanduser("~/CoppeliaSim")
    env["COPPELIASIM_ROOT"] = coppeliasim_root
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{coppeliasim_root}:{existing_ld}" if coppeliasim_root not in existing_ld else existing_ld
    env["QT_QPA_PLATFORM"] = "xcb"
    env["QT_PLUGIN_PATH"] = coppeliasim_root
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    return env


def _run_with_optional_xvfb(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    display_num = random.randint(50, 199)
    xvfb_proc = None
    if shutil.which("Xvfb"):
        while os.path.exists(f"/tmp/.X{display_num}-lock"):
            display_num = random.randint(50, 199)
        xvfb_bin = shutil.which("Xvfb") or "Xvfb"
        xvfb_proc = subprocess.Popen(
            [
                xvfb_bin,
                f":{display_num}",
                "-screen",
                "0",
                "1280x1024x24",
                "+extension",
                "GLX",
                "+extension",
                "RENDER",
                "-ac",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        env = dict(env)
        env["DISPLAY"] = f":{display_num}"

    with open(log_path, "w", encoding="utf-8") as log_handle:
        try:
            result = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        finally:
            if xvfb_proc is not None:
                xvfb_proc.terminate()
                xvfb_proc.wait()
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _top_hits(items: Iterable[dict[str, Any]], *, limit: int, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    def _sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for key in keys:
            value = item.get(key, 0)
            if isinstance(value, (int, float)):
                values.append(-float(value))
            else:
                values.append(str(value))
        values.append(int(item.get("episode_idx", -1)))
        values.append(str(item.get("failure_bucket", "")))
        return tuple(values)

    ordered = sorted(list(items), key=_sort_key)
    return ordered[: max(0, int(limit))]


def _episode_focus_windows(episode_indices: Iterable[int], *, radius: int) -> list[dict[str, Any]]:
    radius = max(0, int(radius))
    unique_episode_indices = sorted({int(ep) for ep in episode_indices})
    windows: list[dict[str, Any]] = []
    for center in unique_episode_indices:
        start = max(0, center - radius)
        end = center + radius
        window_episode_indices = list(range(start, end + 1))
        windows.append(
            {
                "center_episode_idx": center,
                "start_episode_idx": start,
                "end_episode_idx": end,
                "episode_indices": window_episode_indices,
            }
        )
    return windows


def _summarize_sweep_reports(
    chunk_reports: list[dict[str, Any]],
    *,
    top_k: int,
    focus_radius: int,
) -> dict[str, Any]:
    episode_rows: list[dict[str, Any]] = []
    episode_bucket_rows: list[dict[str, Any]] = []
    shell_episode_counter: Counter[int] = Counter()
    shell_bucket_counter: Counter[str] = Counter()
    selected_episode_indices: set[int] = set()
    selected_failure_buckets: set[str] = set()

    for chunk_report in chunk_reports:
        for item in chunk_report.get("by_episode", []):
            row = dict(item)
            if "chunk_tag" in chunk_report:
                row["chunk_tag"] = chunk_report["chunk_tag"]
            episode_rows.append(row)
            if int(row.get("near_basin_shell_rows", 0)) > 0:
                selected_episode_indices.add(int(row.get("episode_idx", -1)))

        for item in chunk_report.get("by_episode_failure_bucket", []):
            row = dict(item)
            if "chunk_tag" in chunk_report:
                row["chunk_tag"] = chunk_report["chunk_tag"]
            episode_bucket_rows.append(row)
            if int(row.get("near_basin_shell_rows", 0)) > 0:
                shell_episode_counter[int(row.get("episode_idx", -1))] += int(row.get("near_basin_shell_rows", 0))
                shell_bucket_counter[str(row.get("failure_bucket", ""))] += int(row.get("near_basin_shell_rows", 0))
                selected_failure_buckets.add(str(row.get("failure_bucket", "")))

    ranked_episodes = _top_hits(
        episode_rows,
        limit=top_k,
        keys=("near_basin_shell_rows", "coarse_pullback_candidate_rows", "horizon_xy_feasible_rows", "yaw_observable_rows", "yaw_feasible_rows", "active_count"),
    )
    ranked_episode_buckets = _top_hits(
        episode_bucket_rows,
        limit=top_k,
        keys=("near_basin_shell_rows", "coarse_pullback_candidate_rows", "horizon_xy_feasible_rows", "yaw_observable_rows", "yaw_feasible_rows", "active_count"),
    )

    if selected_episode_indices:
        recommended_episode_indices = sorted(
            selected_episode_indices,
            key=lambda ep: (
                -int(shell_episode_counter.get(ep, 0)),
                -max((int(item.get("coarse_pullback_candidate_rows", 0)) for item in episode_rows if int(item.get("episode_idx", -1)) == ep), default=0),
                -max((int(item.get("horizon_xy_feasible_rows", 0)) for item in episode_rows if int(item.get("episode_idx", -1)) == ep), default=0),
                -max((int(item.get("yaw_observable_rows", item.get("yaw_feasible_rows", 0))) for item in episode_rows if int(item.get("episode_idx", -1)) == ep), default=0),
                ep,
            ),
        )
    else:
        recommended_episode_indices = [int(item["episode_idx"]) for item in ranked_episodes]

    return {
        "chunk_reports": chunk_reports,
        "episode_rows": episode_rows,
        "episode_bucket_rows": episode_bucket_rows,
        "ranked_episodes": ranked_episodes,
        "ranked_episode_buckets": ranked_episode_buckets,
        "shell_hit_episode_indices": sorted(selected_episode_indices),
        "shell_hit_failure_buckets": sorted(selected_failure_buckets),
        "shell_hit_episode_counts": dict(sorted(shell_episode_counter.items())),
        "shell_hit_bucket_counts": dict(sorted(shell_bucket_counter.items())),
        "shell_hit_episode_focus_windows": _episode_focus_windows(selected_episode_indices, radius=focus_radius),
        "recommended_next_episode_indices": recommended_episode_indices[: max(1, int(top_k))],
        "recommended_focus_episode_indices": sorted(
            {
                int(ep)
                for window in _episode_focus_windows(selected_episode_indices, radius=focus_radius)
                for ep in window["episode_indices"]
            }
        ),
    }


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# C2C v2 Grasp Shell Episode Sweep",
        "",
        f"- candidate_episodes: `{summary['candidate_episode_indices_csv']}`",
        f"- chunk_count: `{summary['chunk_count']}`",
        f"- shell_hit_episode_count: `{len(summary['shell_hit_episode_indices'])}`",
        f"- shell_hit_failure_bucket_count: `{len(summary['shell_hit_failure_buckets'])}`",
        "",
        "## Shell Hit Episodes",
    ]
    if summary["shell_hit_episode_indices"]:
        for ep in summary["shell_hit_episode_indices"]:
            lines.append(f"- `ep{int(ep):03d}`")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Shell Hit Buckets",
    ])
    if summary["shell_hit_failure_buckets"]:
        for bucket in summary["shell_hit_failure_buckets"]:
            lines.append(f"- `{bucket}`")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Top Episodes",
    ])
    for item in summary["ranked_episodes"]:
        lines.append(
            f"- `ep{int(item['episode_idx']):03d}`: shell={int(item['near_basin_shell_rows'])}, "
            f"coarse={int(item.get('coarse_pullback_candidate_rows', 0))}, active={int(item['active_count'])}, "
            f"horizon_xy={int(item['horizon_xy_feasible_rows'])}, yaw={int(item.get('yaw_observable_rows', item.get('yaw_feasible_rows', 0)))}"
        )
    lines.extend([
        "",
        "## Top Episode Buckets",
    ])
    for item in summary["ranked_episode_buckets"]:
        lines.append(
            f"- `ep{int(item['episode_idx']):03d}` / `{item['failure_bucket']}`: shell={int(item['near_basin_shell_rows'])}, "
            f"coarse={int(item.get('coarse_pullback_candidate_rows', 0))}, active={int(item['active_count'])}, "
            f"horizon_xy={int(item['horizon_xy_feasible_rows'])}, yaw={int(item.get('yaw_observable_rows', item.get('yaw_feasible_rows', 0)))}"
        )
    lines.extend([
        "",
        "## Recommended Next Episodes",
    ])
    for ep in summary["recommended_next_episode_indices"]:
        lines.append(f"- `ep{int(ep):03d}`")
    lines.extend([
        "",
        "## Shell Hit Focus Windows",
    ])
    if summary["shell_hit_episode_focus_windows"]:
        for window in summary["shell_hit_episode_focus_windows"]:
            episodes = ", ".join(f"`ep{int(ep):03d}`" for ep in window["episode_indices"])
            lines.append(
                f"- center `ep{int(window['center_episode_idx']):03d}`: "
                f"[{episodes}]"
            )
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Recommended Confirmation Sweep",
    ])
    if summary["recommended_focus_episode_indices"]:
        lines.append(
            "- " + ", ".join(f"`ep{int(ep):03d}`" for ep in summary["recommended_focus_episode_indices"])
        )
    else:
        lines.append("- none")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep candidate episodes to find near-basin shell hits for C2C v2 grasp collection.")
    ap.add_argument("--checkpoint_dir", type=Path, required=True)
    ap.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    ap.add_argument("--mode", type=str, default="basin_recovery_shadow")
    ap.add_argument("--episode_indices", type=str, default="")
    ap.add_argument("--episode_start", type=int, default=5)
    ap.add_argument("--episode_end", type=int, default=19)
    ap.add_argument("--episode_step", type=int, default=1)
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=320)
    ap.add_argument("--eval_seed", type=int, default=3407)
    ap.add_argument("--depth_max", type=float, default=1.0)
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/grasp_shell_episode_sweep"))
    ap.add_argument("--name_suffix", type=str, default="c2c_v2_grasp_shell_sweep")
    ap.add_argument("--gpu_id", type=int, default=None)
    ap.add_argument("--near_grasp_xy_threshold", type=float, default=0.015)
    ap.add_argument("--near_grasp_yaw_threshold", type=float, default=0.08)
    ap.add_argument("--close_ready_xy_threshold", type=float, default=0.005)
    ap.add_argument("--close_ready_yaw_threshold", type=float, default=0.03)
    ap.add_argument("--c2c_grasp_probe_xy_gain", type=float, default=0.35)
    ap.add_argument("--c2c_grasp_probe_max_xy_step", type=float, default=0.003)
    ap.add_argument("--c2c_grasp_probe_horizon", type=int, default=3)
    ap.add_argument("--c2c_grasp_probe_window_mode", type=str, default="forced_shell", choices=["stage", "forced_shell"])
    ap.add_argument("--c2c_grasp_probe_shell_filter", type=str, default="near_yaw_feasible", choices=["off", "near_yaw_feasible"])
    ap.add_argument("--basin_state_calibration_report", type=str, default="runtime_artifacts/coarse2contact_v2/reports/basin_state_calibration/basin_state_calibration.json")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--focus_radius", type=int, default=1)
    ap.add_argument("--stop_after_first_hit", action="store_true", default=False)
    ap.add_argument("--record_video", action="store_true", default=False)
    args = ap.parse_args()

    candidates = _resolve_episode_indices(args)
    if not candidates:
        raise RuntimeError("No candidate episodes provided")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_dir = output_root / args.name_suffix
    sweep_dir.mkdir(parents=True, exist_ok=True)

    python_bin = _python_bin()
    env = _build_env(gpu_id=args.gpu_id)

    chunk_reports: list[dict[str, Any]] = []
    chunk_dirs: list[Path] = []

    for chunk_idx, chunk in enumerate(_chunked(candidates, int(max(1, args.chunk_size)))):
        chunk_tag = f"chunk_{chunk_idx:03d}_{chunk[0]:03d}_{chunk[-1]:03d}"
        chunk_dir = sweep_dir / chunk_tag
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_dirs.append(chunk_dir)

        eval_output_root = chunk_dir / "eval"
        audit_output_root = chunk_dir / "audit"
        eval_log = chunk_dir / "evaluate.log"
        audit_log = chunk_dir / "audit.log"
        episode_csv = ",".join(str(ep) for ep in chunk)

        eval_cmd = [
            python_bin,
            "-u",
            "scripts/evaluate_c2c_v2_rlbench.py",
            "--checkpoint_dir",
            str(args.checkpoint_dir),
            "--task_name",
            args.task_name,
            "--mode",
            args.mode,
            "--num_episodes",
            str(len(chunk)),
            "--episode_indices",
            episode_csv,
            "--max_steps",
            str(args.max_steps),
            "--output_root",
            str(eval_output_root),
            "--name_suffix",
            f"{args.name_suffix}_{chunk_tag}",
            "--eval_seed",
            str(args.eval_seed),
            "--depth_max",
            str(args.depth_max),
            "--dump_runtime_obs",
            "--dump_runtime_obs_all_episodes",
            "--capture_failure_target_pose",
            "--c2c_grasp_probe_policy",
            "replay_oracle_xy",
            "--c2c_grasp_probe_xy_gain",
            str(args.c2c_grasp_probe_xy_gain),
            "--c2c_grasp_probe_max_xy_step",
            str(args.c2c_grasp_probe_max_xy_step),
            "--c2c_grasp_probe_horizon",
            str(args.c2c_grasp_probe_horizon),
            "--c2c_grasp_probe_window_mode",
            args.c2c_grasp_probe_window_mode,
            "--c2c_grasp_probe_shell_filter",
            args.c2c_grasp_probe_shell_filter,
            "--near_grasp_xy_threshold",
            str(args.near_grasp_xy_threshold),
            "--near_grasp_yaw_threshold",
            str(args.near_grasp_yaw_threshold),
            "--close_ready_xy_threshold",
            str(args.close_ready_xy_threshold),
            "--close_ready_yaw_threshold",
            str(args.close_ready_yaw_threshold),
            "--basin_state_calibration_report",
            str(args.basin_state_calibration_report),
            "--no_episode_videos",
            "--no_best_gif",
        ]
        if args.record_video:
            eval_cmd.append("--record_video")
        else:
            eval_cmd.append("--no_video")

        print(f"[sweep] chunk={chunk_tag} episodes={episode_csv}", flush=True)
        _run_with_optional_xvfb(eval_cmd, cwd=ROOT, env=env, log_path=eval_log)

        trace_dir = eval_output_root / "gripper_traces"
        audit_cmd = [
            python_bin,
            "-u",
            "scripts/audit_c2c_v2_grasp_intervention.py",
            "--trace_dir",
            str(trace_dir),
            "--output_dir",
            str(audit_output_root),
            "--near_grasp_xy_threshold",
            str(args.near_grasp_xy_threshold),
            "--near_grasp_yaw_threshold",
            str(args.near_grasp_yaw_threshold),
            "--max_xy_step",
            str(args.c2c_grasp_probe_max_xy_step),
            "--horizon_steps",
            str(args.c2c_grasp_probe_horizon),
        ]
        _run_with_optional_xvfb(audit_cmd, cwd=ROOT, env=env, log_path=audit_log)

        report_path = audit_output_root / "grasp_probe_intervention_audit.json"
        report = _read_json(report_path)
        report["chunk_tag"] = chunk_tag
        report["chunk_episode_indices"] = list(chunk)
        report["eval_output_root"] = str(eval_output_root)
        report["audit_output_root"] = str(audit_output_root)
        report["trace_dir"] = str(trace_dir)
        report["evaluate_log"] = str(eval_log)
        report["audit_log"] = str(audit_log)
        chunk_reports.append(report)

        if args.stop_after_first_hit:
            if any(int(item.get("near_basin_shell_rows", 0)) > 0 for item in report.get("by_episode_failure_bucket", [])):
                break

    summary = _summarize_sweep_reports(
        chunk_reports,
        top_k=int(args.top_k),
        focus_radius=int(args.focus_radius),
    )
    summary.update(
        {
            "candidate_episode_indices": candidates,
            "candidate_episode_indices_csv": ",".join(str(ep) for ep in candidates),
            "chunk_count": int(len(chunk_reports)),
            "chunk_dirs": [str(path) for path in chunk_dirs],
            "sweep_output_root": str(sweep_dir),
            "task_name": args.task_name,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "mode": args.mode,
            "episode_range": {
                "start": int(args.episode_start),
                "end": int(args.episode_end),
                "step": int(args.episode_step),
            },
            "sweep_config": {
                "c2c_grasp_probe_window_mode": args.c2c_grasp_probe_window_mode,
                "c2c_grasp_probe_shell_filter": args.c2c_grasp_probe_shell_filter,
                "c2c_grasp_probe_horizon": int(args.c2c_grasp_probe_horizon),
                "c2c_grasp_probe_xy_gain": float(args.c2c_grasp_probe_xy_gain),
                "c2c_grasp_probe_max_xy_step": float(args.c2c_grasp_probe_max_xy_step),
                "focus_radius": int(args.focus_radius),
                "near_grasp_xy_threshold": float(args.near_grasp_xy_threshold),
                "near_grasp_yaw_threshold": float(args.near_grasp_yaw_threshold),
                "close_ready_xy_threshold": float(args.close_ready_xy_threshold),
                "close_ready_yaw_threshold": float(args.close_ready_yaw_threshold),
            },
        }
    )

    out_json = sweep_dir / "grasp_shell_episode_sweep_summary.json"
    out_md = sweep_dir / "grasp_shell_episode_sweep_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(summary, out_md)

    print(out_json)
    print(out_md)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
