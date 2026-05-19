#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_case(
    eval_script: Path,
    dataset_npz: Path,
    geometry_checkpoint: Path,
    future_risk_checkpoint: Path,
    output_json: Path,
    batch_size: int,
    device: str,
    use_geometry_score_feature: bool,
    topk: int,
    soft_alpha: float | None = None,
    risk_budget: float | None = None,
) -> dict:
    cmd = [
        sys.executable,
        str(eval_script),
        "--dataset_npz",
        str(dataset_npz),
        "--geometry_checkpoint",
        str(geometry_checkpoint),
        "--future_risk_checkpoint",
        str(future_risk_checkpoint),
        "--output_json",
        str(output_json),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--topk",
        str(topk),
    ]
    if use_geometry_score_feature:
        cmd.append("--use_geometry_score_feature")
    if soft_alpha is not None:
        cmd.extend(["--soft_alpha", str(soft_alpha)])
    if risk_budget is not None:
        cmd.extend(["--risk_budget", str(risk_budget)])
    subprocess.run(cmd, check=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--geometry_checkpoint", required=True)
    ap.add_argument("--future_risk_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_geometry_score_feature", action="store_true", default=False)
    args = ap.parse_args()

    dataset_npz = Path(args.dataset_npz)
    geometry_checkpoint = Path(args.geometry_checkpoint)
    future_risk_checkpoint = Path(args.future_risk_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_script = Path(__file__).with_name("evaluate_depth_force_geometry_future_risk_selector.py")

    configs = [
        {"name": "geom_top1", "mode": "geom_top1", "topk": 1, "soft_alpha": None, "risk_budget": None},
        {"name": "soft_topk_k5_a0.1", "mode": "soft", "topk": 5, "soft_alpha": 0.1, "risk_budget": None},
        {"name": "soft_topk_k5_a0.3", "mode": "soft", "topk": 5, "soft_alpha": 0.3, "risk_budget": None},
        {"name": "soft_topk_k5_a0.5", "mode": "soft", "topk": 5, "soft_alpha": 0.5, "risk_budget": None},
        {"name": "soft_topk_k10_a0.1", "mode": "soft", "topk": 10, "soft_alpha": 0.1, "risk_budget": None},
        {"name": "soft_topk_k10_a0.3", "mode": "soft", "topk": 10, "soft_alpha": 0.3, "risk_budget": None},
        {"name": "soft_topk_k10_a0.5", "mode": "soft", "topk": 10, "soft_alpha": 0.5, "risk_budget": None},
        {"name": "budget_topk_k5_b0.05", "mode": "budget", "topk": 5, "soft_alpha": None, "risk_budget": 0.05},
        {"name": "budget_topk_k10_b0.05", "mode": "budget", "topk": 10, "soft_alpha": None, "risk_budget": 0.05},
    ]

    reports: list[dict[str, object]] = []
    for idx, cfg in enumerate(configs):
        out_json = output_dir / f"{idx:03d}_{cfg['name']}.json"
        report = _run_case(
            eval_script=eval_script,
            dataset_npz=dataset_npz,
            geometry_checkpoint=geometry_checkpoint,
            future_risk_checkpoint=future_risk_checkpoint,
            output_json=out_json,
            batch_size=int(args.batch_size),
            device=str(args.device),
            use_geometry_score_feature=bool(args.use_geometry_score_feature),
            topk=int(cfg["topk"]),
            soft_alpha=cfg["soft_alpha"],
            risk_budget=cfg["risk_budget"],
        )
        groups = report.get("groups", {})
        all_rows = groups.get("all_rows", {})
        yaw_rows = groups.get("yaw_opportunity_rows", {})
        summary = {
            "name": cfg["name"],
            "mode": cfg["mode"],
            "topk": int(cfg["topk"]),
            "soft_alpha": cfg["soft_alpha"],
            "risk_budget": cfg["risk_budget"],
            "all_rows.geometry_improve_rate": float(all_rows.get("geometry_improve_rate", 0.0)),
            "all_rows.yaw_success_rate": float(all_rows.get("yaw_success_rate", 0.0)),
            "all_rows.future_risk_nonincrease_rate": float(all_rows.get("future_risk_nonincrease_rate", 0.0)),
            "all_rows.fallback_rate": float(all_rows.get("fallback_rate", 0.0)),
            "all_rows.future_risk_auc": float(all_rows.get("future_risk_auc", 0.0)),
            "all_rows.risk_increase_ece": float(all_rows.get("risk_increase_ece", 0.0)),
            "all_rows.risk_increase_brier": float(all_rows.get("risk_increase_brier", 0.0)),
            "all_rows.geometry_retention_vs_geom_top1": float(all_rows.get("geometry_retention_vs_geom_top1", 0.0)),
            "all_rows.yaw_retention_vs_geom_top1": float(all_rows.get("yaw_retention_vs_geom_top1", 0.0)),
            "yaw_opportunity_rows.geometry_improve_rate": float(yaw_rows.get("geometry_improve_rate", 0.0)),
            "yaw_opportunity_rows.yaw_success_rate": float(yaw_rows.get("yaw_success_rate", 0.0)),
            "yaw_opportunity_rows.future_risk_nonincrease_rate": float(yaw_rows.get("future_risk_nonincrease_rate", 0.0)),
            "yaw_opportunity_rows.fallback_rate": float(yaw_rows.get("fallback_rate", 0.0)),
            "yaw_opportunity_rows.future_risk_auc": float(yaw_rows.get("future_risk_auc", 0.0)),
            "yaw_opportunity_rows.risk_increase_ece": float(yaw_rows.get("risk_increase_ece", 0.0)),
            "yaw_opportunity_rows.risk_increase_brier": float(yaw_rows.get("risk_increase_brier", 0.0)),
            "yaw_opportunity_rows.geometry_retention_vs_geom_top1": float(yaw_rows.get("geometry_retention_vs_geom_top1", 0.0)),
            "yaw_opportunity_rows.yaw_retention_vs_geom_top1": float(yaw_rows.get("yaw_retention_vs_geom_top1", 0.0)),
            "report_path": str(out_json),
        }
        reports.append(summary)
        print(json.dumps(summary, indent=2))

    def score(row: dict[str, object]) -> float:
        return (
            0.40 * float(row["all_rows.geometry_retention_vs_geom_top1"])
            + 0.25 * float(row["all_rows.yaw_retention_vs_geom_top1"])
            + 0.20 * float(row["all_rows.future_risk_nonincrease_rate"])
            - 0.10 * float(row["all_rows.fallback_rate"])
            - 0.05 * float(row["all_rows.risk_increase_ece"])
        )

    reports.sort(key=score, reverse=True)
    summary_path = output_dir / "soft_selector_sweep_summary.json"
    summary_path.write_text(json.dumps({"configs": configs, "reports": reports}, indent=2), encoding="utf-8")
    print(json.dumps({"best": reports[:5], "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
