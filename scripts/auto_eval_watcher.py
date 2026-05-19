"""
auto_eval_watcher.py

Watches a training run directory for new checkpoints and evaluates each one
on a RLBench task.  Tracks the best success rate across all checkpoints and
saves the overall-best GIF.

Can operate in two modes:
  • Polling mode (default):  runs continuously, checking for new checkpoints.
  • One-shot mode (--once):  evaluates all existing checkpoints and exits.

Usage (polling – start alongside or after training):
    python scripts/auto_eval_watcher.py \
        --run_dir outputs/insert_long_train \
        --task_name insert_onto_square_peg \
        --poll_interval 120

Usage (one-shot – after training finishes):
    python scripts/auto_eval_watcher.py \
        --run_dir outputs/insert_long_train \
        --task_name insert_onto_square_peg \
        --once

Note: With a single GPU, run this AFTER training completes (or on a
      separate GPU) to avoid OOM.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_checkpoint_dirs(run_dir):
    """Find all ``*_chkpt`` directories that are siblings of any run inside *run_dir*.

    Checkpoint naming convention from finetune.py:
        {run_dir}/{run_id}--{step}_chkpt
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []

    chkpt_dirs = []
    for item in sorted(run_dir.iterdir()):
        if item.is_dir() and "_chkpt" in item.name:
            chkpt_dirs.append(item)

    # Also look one level deeper (if run_dir points to the root, checkpoints
    # are siblings of the run subdirectory inside it).
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir() and "_chkpt" not in sub.name:
            for item in sorted(sub.parent.iterdir()):
                if item.is_dir() and "_chkpt" in item.name and item not in chkpt_dirs:
                    chkpt_dirs.append(item)

    # Sort by step number extracted from the directory name
    def _step(p):
        m = re.search(r"--(\d+)_chkpt", p.name)
        return int(m.group(1)) if m else 0

    return sorted(chkpt_dirs, key=_step)


def checkpoint_is_ready(chkpt_dir):
    """Return True if the checkpoint looks complete (has action_head file)."""
    return bool(list(Path(chkpt_dir).glob("action_head--*checkpoint.pt")))


def read_eval_result(eval_dir):
    """Read ``eval_results.json`` from an eval output directory."""
    path = Path(eval_dir) / "eval_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# Path to the vla-adapter conda environment Python
_CONDA_PYTHON = os.path.expanduser(
    "~/my_conda_envs/vla-adapter/bin/python"
)


def run_evaluation(chkpt_dir, eval_dir, args):
    """Launch ``evaluate_rlbench.py`` as a subprocess."""
    python_exe = _CONDA_PYTHON if os.path.isfile(_CONDA_PYTHON) else sys.executable
    cmd = [
        python_exe, "-u",
        "scripts/evaluate_rlbench.py",
        "--checkpoint_dir", str(chkpt_dir),
        "--task_name", args.task_name,
        "--num_episodes", str(args.num_episodes),
        "--max_steps", str(args.max_steps),
        "--output_dir", str(eval_dir),
        "--depth_max", str(args.depth_max),
    ]
    if args.use_depth:
        cmd.append("--use_depth")
    else:
        cmd.append("--no_depth")
    if args.use_force:
        cmd.append("--use_force")
    else:
        cmd.append("--no_force")
    if args.record_video:
        cmd.append("--record_video")
    else:
        cmd.append("--no_video")

    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    env["HF_HOME"] = env.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    env["HUGGINGFACE_HUB_CACHE"] = env.get("HUGGINGFACE_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
    # Ensure project root is on PYTHONPATH so prismatic/experiments etc. are importable
    project_root = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing}" if existing else project_root

    # CoppeliaSim / PyRep environment (required for headless RLBench)
    coppeliasim_root = os.path.expanduser("~/CoppeliaSim")
    env["COPPELIASIM_ROOT"] = coppeliasim_root
    # Extend LD_LIBRARY_PATH to include CoppeliaSim libs
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{coppeliasim_root}:{existing_ld}" if coppeliasim_root not in existing_ld else existing_ld
    # Qt xcb + CoppeliaSim plugins.
    # QT_PLUGIN_PATH must point at the CoppeliaSim ROOT so Qt can find both:
    #   <root>/platforms/libqxcb.so
    #   <root>/xcbglintegrations/libqxcb-glx-integration.so  ← this is what was missing
    env["QT_QPA_PLATFORM"] = "xcb"
    env["QT_PLUGIN_PATH"] = coppeliasim_root
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    # NVIDIA hardware GLX does not work on virtual Xvfb displays;
    # force Mesa software renderer so CoppeliaSim's OpenGL3 renderer can create GL contexts.
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"

    # Start a dedicated Xvfb virtual display with GLX support for CoppeliaSim's
    # OpenGL3 renderer.  We manage the process directly (instead of xvfb-run)
    # so we can pass +extension GLX as proper separate arguments.
    import random
    display_num = random.randint(50, 199)
    # Make sure the lock file doesn't already exist
    while os.path.exists(f"/tmp/.X{display_num}-lock"):
        display_num = random.randint(50, 199)

    xvfb_bin = shutil.which("Xvfb") or "Xvfb"
    xvfb_proc = subprocess.Popen(
        [xvfb_bin, f":{display_num}",
         "-screen", "0", "1280x1024x24",
         "+extension", "GLX",
         "+extension", "RENDER",
         "-ac"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # Let Xvfb initialise

    env["DISPLAY"] = f":{display_num}"

    print(f"  DISPLAY=:{display_num}  CMD: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, cwd=project_root)
    finally:
        xvfb_proc.terminate()
        xvfb_proc.wait()
    return result


def main():
    parser = argparse.ArgumentParser(description="Auto-evaluate RLBench checkpoints")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Training run root directory to watch for checkpoints")
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--num_episodes", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--no_depth", dest="use_depth", action="store_false")
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--no_force", dest="use_force", action="store_false")
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="record_video", action="store_false")
    parser.add_argument("--depth_max", type=float, default=1.0)
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="CUDA_VISIBLE_DEVICES override (default: inherit from env)")
    parser.add_argument("--poll_interval", type=int, default=120,
                        help="Seconds between polls (polling mode)")
    parser.add_argument("--once", action="store_true",
                        help="Evaluate all existing checkpoints then exit")
    # Isolation filters
    parser.add_argument("--eval_root_dir", type=str, default=None,
                        help="Custom evaluation output root directory (default: eval_logs/<task>)")
    parser.add_argument("--checkpoint_name_contains", type=str, default=None,
                        help="Only evaluate checkpoints whose name contains this substring")
    parser.add_argument("--checkpoint_step_min", type=int, default=0,
                        help="Only evaluate checkpoints with step >= this value")
    parser.add_argument("--checkpoint_mtime_after", type=float, default=0,
                        help="Only evaluate checkpoints modified after this Unix timestamp")
    args = parser.parse_args()

    eval_root = Path(args.eval_root_dir) if args.eval_root_dir else Path("eval_logs") / args.task_name
    eval_root.mkdir(parents=True, exist_ok=True)
    summary_path = eval_root / "summary.json"

    evaluated = set()
    best_sr = -1.0
    best_ckpt = None
    all_results = {}

    # Load previous summary if any
    if summary_path.exists():
        with open(summary_path) as f:
            prev = json.load(f)
        all_results = prev.get("checkpoints", {})
        evaluated = set(all_results.keys())
        best_sr = prev.get("best_success_rate", -1.0)
        best_ckpt = prev.get("best_checkpoint", None)
        print(f"[watcher] Resuming — {len(evaluated)} checkpoints already evaluated, best SR={best_sr:.1%}")

    print(f"[watcher] Watching: {args.run_dir}")
    print(f"[watcher] Task: {args.task_name}")
    print(f"[watcher] Eval root: {eval_root}")
    if args.checkpoint_name_contains:
        print(f"[watcher] Filter name contains: {args.checkpoint_name_contains}")
    if args.checkpoint_mtime_after:
        print(f"[watcher] Filter mtime after: {args.checkpoint_mtime_after}")
    if args.once:
        print("[watcher] Mode: one-shot")
    else:
        print(f"[watcher] Mode: polling (interval={args.poll_interval}s)")

    def checkpoint_matches_filters(chkpt_dir):
        name = chkpt_dir.name
        if args.checkpoint_name_contains and args.checkpoint_name_contains not in name:
            return False
        if args.checkpoint_step_min > 0:
            m = re.search(r"--(\d+)_chkpt", name)
            if m and int(m.group(1)) < args.checkpoint_step_min:
                return False
        if args.checkpoint_mtime_after > 0:
            mtime = chkpt_dir.stat().st_mtime
            if mtime < args.checkpoint_mtime_after:
                return False
        return True

    while True:
        chkpt_dirs = find_checkpoint_dirs(args.run_dir)

        for chkpt_dir in chkpt_dirs:
            ckpt_name = chkpt_dir.name
            if not checkpoint_matches_filters(chkpt_dir):
                continue
            if ckpt_name in evaluated:
                continue
            if not checkpoint_is_ready(chkpt_dir):
                continue

            print(f"\n{'=' * 60}")
            print(f"[watcher] New checkpoint: {ckpt_name}")
            print(f"{'=' * 60}")

            eval_dir = eval_root / ckpt_name
            result = run_evaluation(chkpt_dir, eval_dir, args)

            evaluated.add(ckpt_name)

            if result.returncode != 0:
                print(f"[watcher] Evaluation FAILED for {ckpt_name} (exit code {result.returncode})")
                all_results[ckpt_name] = {"status": "failed"}
            else:
                eval_result = read_eval_result(eval_dir)
                if eval_result is not None:
                    sr = eval_result.get("success_rate", 0.0)
                    print(f"[watcher] {ckpt_name}: SR = {sr:.1%}")
                    all_results[ckpt_name] = eval_result

                    if sr > best_sr:
                        best_sr = sr
                        best_ckpt = ckpt_name
                        # Copy best GIF (success preferred, else best_fail fallback)
                        src_gif = eval_dir / "best_success.gif"
                        if not src_gif.exists():
                            src_gif = eval_dir / "best_fail.gif"
                        if src_gif.exists():
                            dst_gif = eval_root / "overall_best.gif"
                            shutil.copy2(src_gif, dst_gif)
                            print(f"[watcher] ★ New best! SR={sr:.1%} ({ckpt_name}) → {dst_gif}")
                else:
                    all_results[ckpt_name] = {"status": "no_result"}

            # Persist summary
            summary = {
                "task_name": args.task_name,
                "best_success_rate": best_sr,
                "best_checkpoint": best_ckpt,
                "checkpoints": all_results,
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

        # Print status
        if best_ckpt:
            print(f"\n[watcher] Best so far: {best_ckpt} (SR={best_sr:.1%})")

        if args.once:
            if not chkpt_dirs:
                print("[watcher] No checkpoints found.")
            print("[watcher] One-shot mode — done.")
            break

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
