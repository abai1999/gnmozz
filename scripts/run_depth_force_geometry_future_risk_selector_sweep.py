#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _run_selector(
    eval_script: Path,
    dataset_npz: Path,
    geometry_checkpoint: Path,
    future_risk_checkpoint: Path,
    output_json: Path,
    batch_size: int,
    device: str,
    use_geometry_score_feature: bool,
    topk: int,
    risk_margin: float,
    geo_margin: float,
) -> dict:
    output_json.parent.mkdir(parents=True, exist_ok=True)
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
        "--risk_margin",
        str(risk_margin),
        "--geo_margin",
        str(geo_margin),
    ]
    if use_geometry_score_feature:
        cmd.append("--use_geometry_score_feature")
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
    ap.add_argument("--topks", default="3,5,10")
    ap.add_argument("--risk_margins", default="0.0,0.05,0.1")
    ap.add_argument("--geo_margins", default="0.0,0.05")
    args = ap.parse_args()

    dataset_npz = Path(args.dataset_npz)
    geometry_checkpoint = Path(args.geometry_checkpoint)
    future_risk_checkpoint = Path(args.future_risk_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_script = Path(__file__).with_name("evaluate_depth_force_geometry_future_risk_selector.py")

    topks = _parse_csv_ints(args.topks)
    risk_margins = _parse_csv_floats(args.risk_margins)
    geo_margins = _parse_csv_floats(args.geo_margins)

    configs: list[dict[str, object]] = []

    configs.append({
        "name": "geom_top1",
        "topk": 1,
        "risk_margin": 0.0,
        "geo_margin": 0.0,
    })

    for topk, risk_margin, geo_margin in itertools.product(topks, risk_margins, geo_margins):
        configs.append({
            "name": f"geom_topk_safe_k{topk}_r{risk_margin:g}_g{geo_margin:g}",
            "topk": int(topk),
            "risk_margin": float(risk_margin),
            "geo_margin": float(geo_margin),
        })
        configs.append({
            "name": f"risk_veto_k{topk}_r{risk_margin:g}_g{geo_margin:g}",
            "topk": int(topk),
            "risk_margin": float(risk_margin),
            "geo_margin": float(geo_margin),
        })

    for risk_margin, geo_margin in itertools.product(risk_margins, geo_margins):
        configs.append({
            "name": f"pareto_r{risk_margin:g}_g{geo_margin:g}",
            "topk": 1,
            "risk_margin": float(risk_margin),
            "geo_margin": float(geo_margin),
        })

    reports: list[dict[str, object]] = []
    for idx, cfg in enumerate(configs):
        out_json = output_dir / f"{idx:03d}_{cfg['name']}.json"
        report = _run_selector(
            eval_script=eval_script,
            dataset_npz=dataset_npz,
            geometry_checkpoint=geometry_checkpoint,
            future_risk_checkpoint=future_risk_checkpoint,
            output_json=out_json,
            batch_size=int(args.batch_size),
            device=str(args.device),
            use_geometry_score_feature=bool(args.use_geometry_score_feature),
            topk=int(cfg["topk"]),
            risk_margin=float(cfg["risk_margin"]),
            geo_margin=float(cfg["geo_margin"]),
        )
        groups = report.get("groups", {})
        all_rows = groups.get("all_rows", {})
        yaw_rows = groups.get("yaw_opportunity_rows", {})
        summary = {
            "name": cfg["name"],
            "topk": int(cfg["topk"]),
            "risk_margin": float(cfg["risk_margin"]),
            "geo_margin": float(cfg["geo_margin"]),
            "all_rows.geometry_improve_rate": float(all_rows.get("geometry_improve_rate", 0.0)),
            "all_rows.yaw_success_rate": float(all_rows.get("yaw_success_rate", 0.0)),
            "all_rows.future_risk_nonincrease_rate": float(all_rows.get("future_risk_nonincrease_rate", 0.0)),
            "all_rows.fallback_rate": float(all_rows.get("fallback_rate", 0.0)),
            "all_rows.future_risk_auc": float(all_rows.get("future_risk_auc", 0.0)),
            "all_rows.risk_increase_ece": float(all_rows.get("risk_increase_ece", 0.0)),
            "all_rows.risk_increase_brier": float(all_rows.get("risk_increase_brier", 0.0)),
            "yaw_opportunity_rows.geometry_improve_rate": float(yaw_rows.get("geometry_improve_rate", 0.0)),
            "yaw_opportunity_rows.yaw_success_rate": float(yaw_rows.get("yaw_success_rate", 0.0)),
            "yaw_opportunity_rows.future_risk_nonincrease_rate": float(yaw_rows.get("future_risk_nonincrease_rate", 0.0)),
            "yaw_opportunity_rows.fallback_rate": float(yaw_rows.get("fallback_rate", 0.0)),
            "yaw_opportunity_rows.future_risk_auc": float(yaw_rows.get("future_risk_auc", 0.0)),
            "yaw_opportunity_rows.risk_increase_ece": float(yaw_rows.get("risk_increase_ece", 0.0)),
            "yaw_opportunity_rows.risk_increase_brier": float(yaw_rows.get("risk_increase_brier", 0.0)),
            "per_episode_min_geometry_improve": min(
                (float(v.get("geometry_improve_rate", 0.0)) for v in report.get("episodes", {}).values()),
                default=0.0,
            ),
            "per_episode_min_yaw_success": min(
                (float(v.get("yaw_success_rate", 0.0)) for v in report.get("episodes", {}).values()),
                default=0.0,
            ),
            "per_episode_max_fallback": max(
                (float(v.get("fallback_rate", 0.0)) for v in report.get("episodes", {}).values()),
                default=0.0,
            ),
            "report_path": str(out_json),
        }
        reports.append(summary)
        print(json.dumps(summary, indent=2))

    # Rank by preserving geometry/yaw while reducing future risk and fallback.
    def score(row: dict[str, object]) -> float:
        return (
            0.45 * float(row["all_rows.geometry_improve_rate"])
            + 0.30 * float(row["all_rows.yaw_success_rate"])
            + 0.15 * float(row["all_rows.future_risk_nonincrease_rate"])
            - 0.05 * float(row["all_rows.fallback_rate"])
            - 0.05 * float(row["all_rows.risk_increase_ece"])
        )

    reports.sort(key=score, reverse=True)
    summary_path = output_dir / "selector_sweep_summary.json"
    summary_path.write_text(json.dumps({"configs": configs, "reports": reports}, indent=2), encoding="utf-8")
    print(json.dumps({"best": reports[:10], "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
