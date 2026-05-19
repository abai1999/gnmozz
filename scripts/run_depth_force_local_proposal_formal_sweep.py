#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TRAIN_SCRIPT = Path("scripts/train_depth_force_local_proposal_policy.py")
EVAL_SCRIPT = Path("scripts/evaluate_depth_force_local_proposal_policy.py")


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--proposal_counts", default="8,16")
    ap.add_argument(
        "--modes",
        default="proprio_planner,force_proprio_planner,depth_proprio_planner,depth_force_proprio_planner",
        help="comma-separated input modes to sweep",
    )
    ap.add_argument("--state_dim", type=int, default=384)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--frontier_cover_weight", type=float, default=1.0)
    ap.add_argument("--best_safe_weight", type=float, default=1.0)
    ap.add_argument("--best_geom_weight", type=float, default=0.5)
    ap.add_argument("--yaw_weight", type=float, default=1.0)
    ap.add_argument("--score_weight", type=float, default=0.25)
    ap.add_argument("--mode_weight", type=float, default=0.25)
    ap.add_argument("--diversity_weight", type=float, default=0.05)
    ap.add_argument("--scale_reg_weight", type=float, default=0.01)
    ap.add_argument("--selection_mode", default="layered_multi")
    ap.add_argument("--multi_head_selection_mode", default="layered_multi")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--distance_tol", type=float, default=0.75)
    args = ap.parse_args()

    proposal_counts = [int(x) for x in args.proposal_counts.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for proposal_count in proposal_counts:
        for mode in modes:
            run_dir = root / f"{mode}_k{proposal_count}"
            run_dir.mkdir(parents=True, exist_ok=True)
            train_cmd = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--dataset_npz",
                args.dataset_npz,
                "--output_dir",
                str(run_dir),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight_decay",
                str(args.weight_decay),
                "--proposal_count",
                str(proposal_count),
                "--state_dim",
                str(args.state_dim),
                "--hidden_dim",
                str(args.hidden_dim),
                "--input_mode",
                mode,
                "--frontier_cover_weight",
                str(args.frontier_cover_weight),
                "--best_safe_weight",
                str(args.best_safe_weight),
                "--best_geom_weight",
                str(args.best_geom_weight),
                "--yaw_weight",
                str(args.yaw_weight),
                "--score_weight",
                str(args.score_weight),
                "--mode_weight",
                str(args.mode_weight),
                "--diversity_weight",
                str(args.diversity_weight),
                "--scale_reg_weight",
                str(args.scale_reg_weight),
                "--multi_head_selection_mode",
                str(args.multi_head_selection_mode),
                "--device",
                str(args.device),
            ]
            _run(train_cmd)

            ckpt = run_dir / "checkpoints" / "final_full.pt"
            eval_json = run_dir / "local_proposal_eval_report.json"
            eval_cmd = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--dataset_npz",
                args.dataset_npz,
                "--checkpoint",
                str(ckpt),
                "--output_json",
                str(eval_json),
                "--device",
                str(args.device),
                "--distance_tol",
                str(args.distance_tol),
                "--selection_mode",
                str(args.selection_mode),
            ]
            _run(eval_cmd)
            report = json.loads(eval_json.read_text(encoding="utf-8"))
            summary.append(
                {
                    "mode": mode,
                    "proposal_count": proposal_count,
                    "run_dir": str(run_dir),
                    "selected_geom_gain_mean": report["all_rows"]["selected_geom_gain_mean"],
                    "selected_risk_delta_mean": report["all_rows"]["selected_risk_delta_mean"],
                    "selected_yaw_rate": report["all_rows"]["selected_yaw_rate"],
                    "selected_correct_yaw_sign_rate": report["all_rows"]["selected_correct_yaw_sign_rate"],
                    "selected_best_safe_hit_rate": report["all_rows"]["selected_best_safe_hit_rate"],
                    "selected_pareto_hit_rate": report["all_rows"]["selected_pareto_hit_rate"],
                    "score_selected_mode_acc": report["all_rows"]["score_selected_mode_acc"],
                }
            )

    summary_path = root / "formal_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
