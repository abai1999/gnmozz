#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_CONFIGS = [
    {
        "name": "geometry_only_yawpair",
        "dataset": "runtime_artifacts/depth_force_contact/privgeo_candidate_geometry_only_8ep_yawcf_20260501h/depth_force_privileged_geometry_candidate_dataset.npz",
        "output_dir": "runtime_artifacts/depth_force_contact/privgeo_sweep_geometry_only_yawpair_20260501j",
        "epochs": 5,
        "extra_args": ["--use_normalized_costs", "--yaw_pair_weight", "0.15", "--yaw_sign_weight", "0.15", "--small_over_large_yaw_weight", "0.15"],
    },
    {
        "name": "geomrisk_lam03_yawpair",
        "dataset": "runtime_artifacts/depth_force_contact/privgeo_candidate_geomrisk_8ep_yawcf_norm_lam03_20260501h/depth_force_privileged_geometry_candidate_dataset.npz",
        "output_dir": "runtime_artifacts/depth_force_contact/privgeo_sweep_geomrisk_lam03_yawpair_20260501j",
        "epochs": 5,
        "extra_args": ["--use_normalized_costs", "--yaw_pair_weight", "0.15", "--yaw_sign_weight", "0.15", "--small_over_large_yaw_weight", "0.15"],
    },
    {
        "name": "geomrisk_lam10_yawpair",
        "dataset": "runtime_artifacts/depth_force_contact/privgeo_candidate_geomrisk_8ep_yawcf_norm_lam10_20260501h/depth_force_privileged_geometry_candidate_dataset.npz",
        "output_dir": "runtime_artifacts/depth_force_contact/privgeo_sweep_geomrisk_lam10_yawpair_20260501j",
        "epochs": 5,
        "extra_args": ["--use_normalized_costs", "--yaw_pair_weight", "0.15", "--yaw_sign_weight", "0.15", "--small_over_large_yaw_weight", "0.15"],
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="/home/guoning/my_conda_envs/vla-adapter/bin/python")
    ap.add_argument("--train_script", default="scripts/train_depth_force_mode_first_geometry_risk_policy.py")
    ap.add_argument("--summary_json", default="runtime_artifacts/depth_force_contact/privgeo_sweep_summary_20260501j.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--configs", nargs="*", default=[])
    args = ap.parse_args()

    if args.configs:
        selected = [c for c in DEFAULT_CONFIGS if c["name"] in args.configs]
        if not selected:
            raise SystemExit(f"no matching configs in {args.configs}")
    else:
        selected = DEFAULT_CONFIGS

    results = []
    for cfg in selected:
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            args.train_script,
            "--dataset_npz",
            cfg["dataset"],
            "--output_dir",
            cfg["output_dir"],
            "--epochs",
            str(cfg.get("epochs", args.epochs)),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
        ] + list(cfg.get("extra_args", []))
        print("RUN", cfg["name"])
        print("CMD", " ".join(cmd))
        subprocess.run(cmd, check=True)
        report_path = out_dir / "mode_first_geometry_risk_policy_report.json"
        report = json.loads(report_path.read_text())
        results.append({
            "name": cfg["name"],
            "dataset": cfg["dataset"],
            "output_dir": cfg["output_dir"],
            "report_path": str(report_path),
            "final_train_metrics": report.get("final_train_metrics", {}),
            "split_reports": report.get("split_reports", []),
        })

    summary = {
        "configs": [cfg["name"] for cfg in selected],
        "results": results,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
