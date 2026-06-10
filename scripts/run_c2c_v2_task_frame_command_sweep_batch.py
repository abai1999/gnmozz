#!/usr/bin/env python3
"""Run v59 command-sweep specs through the C2C v2 evaluator.

Each selected spec row is executed as an independent evaluator run. This keeps
candidate commands from contaminating each other and creates real pre/action/post
transition traces that can later be converted into an applied-transition
manifest. The script can also dry-run the commands for scheduling/debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_c2c_v2_task_frame_applied_transition_manifest import build_manifest as build_applied_manifest  # noqa: E402


DEFAULT_PLANNER_CHECKPOINT = "/home/guoning/code/VLA2/pretrained_models/planner_checkpoints/insert_onto_square_peg_30000_chkpt"
DEFAULT_RUNTIME_XY_CHECKPOINT = "runtime_artifacts/coarse2contact_v2/checkpoints/runtime_xy_spatial_temporal_v42_expanded_v4pilot_candidate.pt"
DEFAULT_TASK_FRAME_V46_CHECKPOINT = "runtime_artifacts/coarse2contact_v2/checkpoints/runtime_task_frame_alignment_v58_nearfield_onpolicy_candidate.pt"
DEFAULT_PYTHON_BIN = "/home/guoning/my_conda_envs/vla-adapter/bin/python"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_int_set(text: str) -> set[int]:
    out: set[int] = set()
    if not text:
        return out
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _parse_str_set(text: str) -> set[str]:
    return {part.strip() for part in str(text or "").split(",") if part.strip()}


def select_sweep_rows(
    rows: list[Mapping[str, Any]],
    *,
    row_indices: str = "",
    start_index: int = 0,
    max_rows: int = 0,
    episode_indices: str = "",
    candidate_names: str = "",
) -> list[tuple[int, Mapping[str, Any]]]:
    allowed_rows = _parse_int_set(row_indices)
    allowed_eps = _parse_int_set(episode_indices)
    allowed_candidates = _parse_str_set(candidate_names)
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for idx, row in enumerate(rows):
        if allowed_rows and idx not in allowed_rows:
            continue
        if not allowed_rows and idx < int(start_index):
            continue
        if allowed_eps and int(row.get("episode_idx", -1)) not in allowed_eps:
            continue
        if allowed_candidates and str(row.get("candidate_name", "")) not in allowed_candidates:
            continue
        selected.append((idx, row))
        if not allowed_rows and int(max_rows) > 0 and len(selected) >= int(max_rows):
            break
    return selected


def output_root_for_row(base_root: Path, row_index: int, row: Mapping[str, Any]) -> Path:
    ep = int(row.get("episode_idx", -1))
    step = int(row.get("step_idx", row.get("step", -1)))
    candidate = str(row.get("candidate_name", "candidate")).replace("/", "_").replace(" ", "_")
    return base_root / f"row{row_index:05d}_{candidate}_ep{ep:03d}_step{step:03d}"


def build_eval_command(
    *,
    spec_jsonl: Path,
    row_index: int,
    row: Mapping[str, Any],
    output_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    ep = int(row.get("episode_idx", -1))
    step = int(row.get("step_idx", row.get("step", 0)) or 0)
    max_steps = max(int(args.min_max_steps), step + int(args.post_steps))
    python_bin = str(getattr(args, "python_bin", "") or "").strip() or DEFAULT_PYTHON_BIN
    cmd = [
        python_bin,
        "scripts/evaluate_c2c_v2_rlbench.py",
        "--checkpoint_dir",
        str(args.checkpoint_dir),
        "--mode",
        "basin_recovery_shadow",
        "--c2c_grasp_probe_policy",
        "runtime_estimator_xy",
        "--c2c_grasp_probe_smoke_type",
        "runtime_style_c2c",
        "--runtime_xy_calibration_json",
        str(args.runtime_xy_calibration_json),
        "--task_frame_v46_ckpt",
        str(args.task_frame_v46_ckpt),
        "--enable_v46_task_frame_micro_servo",
        "--v46_task_frame_xy_mode",
        "transition_guarded_effect_aware",
        "--task_frame_v46_command_sweep_spec_jsonl",
        str(spec_jsonl),
        "--task_frame_v46_command_sweep_row_index",
        str(int(row_index)),
        "--episode_indices",
        str(ep),
        "--max_steps",
        str(max_steps),
        "--eval_seed",
        str(int(args.eval_seed)),
        "--output_root",
        str(output_root),
        "--name_suffix",
        str(output_root.name),
        "--record_video",
        "--video_layout",
        "front_wrist",
        "--write_episode_videos",
        "--record_gripper_trace",
        "--dump_runtime_obs",
        "--dump_runtime_obs_all_episodes",
        "--capture_failure_target_pose",
    ]
    return cmd


def _run_one(
    item: tuple[int, Mapping[str, Any]],
    *,
    spec_jsonl: Path,
    output_base: Path,
    args: argparse.Namespace,
    gpu: str,
) -> dict[str, Any]:
    row_index, row = item
    root = output_root_for_row(output_base, row_index, row)
    root.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = build_eval_command(spec_jsonl=spec_jsonl, row_index=row_index, row=row, output_root=root, args=args)
    if str(getattr(args, "conda_prefix", "") or "").strip():
        conda_prefix = str(getattr(args, "conda_prefix", "")).strip()
        launcher = [str(Path(conda_prefix).expanduser() / "bin" / "conda"), "run", "--prefix", conda_prefix, "xvfb-run", "-a", "python"]
    else:
        launcher = ["conda", "run", "-n", str(args.conda_env), "xvfb-run", "-a", "python"]
    cmd = launcher + cmd[1:]
    result: dict[str, Any] = {
        "row_index": int(row_index),
        "episode_idx": int(row.get("episode_idx", -1)),
        "step_idx": int(row.get("step_idx", row.get("step", -1))),
        "candidate_name": str(row.get("candidate_name", "")),
        "output_root": str(root),
        "gpu": str(gpu),
        "command": cmd,
        "returncode": None,
        "launcher": "conda_run_xvfb_run",
    }
    eval_results = root / "eval_results.json"
    trace_dir = root / "gripper_traces"
    has_trace = bool(trace_dir.is_dir() and any(trace_dir.glob("*_gripper_trace.jsonl")))
    if bool(getattr(args, "skip_existing", False)) and eval_results.exists() and has_trace:
        result["returncode"] = 0
        result["dry_run"] = False
        result["skipped_existing"] = True
        return result
    if bool(args.dry_run):
        result["returncode"] = 0
        result["dry_run"] = True
        result["skipped_existing"] = False
        return result
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    result["returncode"] = int(completed.returncode)
    result["dry_run"] = False
    result["skipped_existing"] = False
    return result


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    spec_jsonl = Path(args.spec_jsonl)
    rows = read_jsonl(spec_jsonl)
    selected = select_sweep_rows(
        rows,
        row_indices=str(args.row_indices),
        start_index=int(args.start_index),
        max_rows=int(args.max_rows),
        episode_indices=str(args.episode_indices),
        candidate_names=str(args.candidate_names),
    )
    output_base = Path(args.output_root)
    output_base.mkdir(parents=True, exist_ok=True)
    gpus = [part.strip() for part in str(args.gpus or "").split(",") if part.strip()]
    if not gpus:
        gpus = [""]
    results: list[dict[str, Any]] = []
    max_parallel = max(1, int(args.max_parallel))
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = []
        for order, item in enumerate(selected):
            gpu = gpus[order % len(gpus)]
            futures.append(pool.submit(_run_one, item, spec_jsonl=spec_jsonl, output_base=output_base, args=args, gpu=gpu))
        for fut in as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda row: int(row["row_index"]))
    successful_roots = [Path(row["output_root"]) for row in results if int(row.get("returncode", 1)) == 0 and not bool(row.get("dry_run", False))]
    manifest_summary: dict[str, Any] | None = None
    if bool(args.build_manifest) and successful_roots:
        manifest_jsonl = Path(args.manifest_jsonl) if str(args.manifest_jsonl) else output_base / "command_sweep_applied_transition_manifest.jsonl"
        manifest_summary_json = Path(args.manifest_summary_json) if str(args.manifest_summary_json) else output_base / "command_sweep_applied_transition_manifest.summary.json"
        manifest_summary = build_applied_manifest(
            successful_roots,
            output_jsonl=manifest_jsonl,
            summary_json=manifest_summary_json,
            require_command_sweep_executed=True,
        )
    summary = {
        "schema_version": "c2c_v2_task_frame_command_sweep_batch_summary_v1",
        "spec_jsonl": str(spec_jsonl),
        "output_root": str(output_base),
        "selected_rows": len(selected),
        "dry_run": bool(args.dry_run),
        "max_parallel": int(max_parallel),
        "gpus": gpus,
        "results": results,
        "success_count": int(sum(int(row.get("returncode", 1)) == 0 for row in results)),
        "failure_count": int(sum(int(row.get("returncode", 1)) != 0 for row in results)),
        "skipped_existing_count": int(sum(bool(row.get("skipped_existing", False)) for row in results)),
        "applied_manifest_summary": manifest_summary,
    }
    if str(args.summary_json):
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_jsonl", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--summary_json", default="", type=Path)
    parser.add_argument("--row_indices", default="", help="Comma/range row filter, e.g. 0,3,8-12.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--episode_indices", default="")
    parser.add_argument("--candidate_names", default="")
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--build_manifest", action="store_true", default=False)
    parser.add_argument("--manifest_jsonl", default="")
    parser.add_argument("--manifest_summary_json", default="")
    parser.add_argument("--conda_env", default="vla-adapter")
    parser.add_argument("--conda_prefix", default="", help="Optional conda env prefix; overrides --conda_env when set.")
    parser.add_argument("--python_bin", default=DEFAULT_PYTHON_BIN, help="Python executable used to launch the evaluator.")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_PLANNER_CHECKPOINT)
    parser.add_argument("--runtime_xy_calibration_json", default=DEFAULT_RUNTIME_XY_CHECKPOINT)
    parser.add_argument("--task_frame_v46_ckpt", default=DEFAULT_TASK_FRAME_V46_CHECKPOINT)
    parser.add_argument("--eval_seed", type=int, default=3407)
    parser.add_argument("--post_steps", type=int, default=15)
    parser.add_argument("--min_max_steps", type=int, default=115)
    args = parser.parse_args()
    summary = run_batch(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
