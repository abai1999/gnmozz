#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_CHECKPOINTS = [
    (
        "v5_k8_proprio",
        "/home/guoning/code/VLA/runtime_artifacts/depth_force_contact/"
        "local_proposal_k8_proprio_from_v5_20260502b/checkpoints/final_full.pt",
    ),
    (
        "formal_proprio",
        "/home/guoning/code/VLA/runtime_artifacts/depth_force_contact/"
        "local_proposal_formal_proprio_k8_20260501e/checkpoints/final_full.pt",
    ),
    (
        "formal_force",
        "/home/guoning/code/VLA/runtime_artifacts/depth_force_contact/"
        "local_proposal_formal_force_k8_20260501e/checkpoints/final_full.pt",
    ),
    (
        "formal_depthforce",
        "/home/guoning/code/VLA/runtime_artifacts/depth_force_contact/"
        "local_proposal_formal_depthforce_k8_20260501e/checkpoints/final_full.pt",
    ),
]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _parse_checkpoint(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"checkpoint spec must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"checkpoint spec must be NAME=PATH, got {spec!r}")
    return name, path


def _run_eval(
    eval_script: Path,
    dataset_npz: str,
    proposal_cache_npz: str,
    checkpoint: str,
    *,
    device: str,
    batch_size: int,
    distance_tol: float,
    yaw_presence_threshold: float,
    yaw_match_tol: float,
    selection_mode: str,
    multi_utility_w_safe: float,
    multi_utility_w_pareto: float,
    multi_utility_w_yaw: float,
    multi_utility_w_geom: float,
    multi_utility_w_risk: float,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="local_proposal_obs_") as tmpdir:
        out_json = Path(tmpdir) / "eval.json"
        cmd = [
            sys.executable,
            str(eval_script),
            "--dataset_npz",
            dataset_npz,
            "--proposal_cache_npz",
            proposal_cache_npz,
            "--checkpoint",
            checkpoint,
            "--output_json",
            str(out_json),
            "--batch_size",
            str(batch_size),
            "--distance_tol",
            str(distance_tol),
            "--yaw_presence_threshold",
            str(yaw_presence_threshold),
            "--yaw_match_tol",
            str(yaw_match_tol),
            "--selection_mode",
            selection_mode,
            "--multi_utility_w_safe",
            str(multi_utility_w_safe),
            "--multi_utility_w_pareto",
            str(multi_utility_w_pareto),
            "--multi_utility_w_yaw",
            str(multi_utility_w_yaw),
            "--multi_utility_w_geom",
            str(multi_utility_w_geom),
            "--multi_utility_w_risk",
            str(multi_utility_w_risk),
            "--device",
            device,
        ]
        subprocess.run(cmd, check=True)
        return json.loads(out_json.read_text(encoding="utf-8"))


def _select(report: dict[str, object], *keys: str, default: float = 0.0) -> float:
    cur: object = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return float(default)
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    return float(default)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--proposal_cache_npz", required=True)
    ap.add_argument("--checkpoint", action="append", default=[])
    ap.add_argument(
        "--eval_script",
        default=str(Path(__file__).resolve().with_name("evaluate_depth_force_local_proposal_policy.py")),
    )
    ap.add_argument("--output_json", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--distance_tol", type=float, default=0.75)
    ap.add_argument("--yaw_presence_threshold", type=float, default=0.0025)
    ap.add_argument("--yaw_match_tol", type=float, default=0.0015)
    ap.add_argument(
        "--selection_mode",
        default="layered_multi",
        choices=(
            "scalar",
            "best_safe",
            "pareto",
            "yaw_match",
            "risk_safe",
            "geometry_gain",
            "weighted_multi",
            "layered_multi",
            "pareto_then_best_safe",
            "pareto_then_geometry",
        ),
    )
    ap.add_argument("--multi_utility_w_safe", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_pareto", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_yaw", type=float, default=1.0)
    ap.add_argument("--multi_utility_w_geom", type=float, default=0.5)
    ap.add_argument("--multi_utility_w_risk", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if False else "cpu")
    args = ap.parse_args()

    ckpts = [_parse_checkpoint(s) for s in args.checkpoint] if args.checkpoint else list(DEFAULT_CHECKPOINTS)
    eval_script = Path(args.eval_script)
    if not eval_script.exists():
        raise FileNotFoundError(f"evaluation script not found: {eval_script}")

    reports: dict[str, dict[str, object]] = {}
    for name, path in ckpts:
        reports[name] = _run_eval(
            eval_script,
            args.dataset_npz,
            args.proposal_cache_npz,
            path,
            device=args.device,
            batch_size=args.batch_size,
            distance_tol=args.distance_tol,
            yaw_presence_threshold=args.yaw_presence_threshold,
            yaw_match_tol=args.yaw_match_tol,
            selection_mode=str(args.selection_mode),
            multi_utility_w_safe=float(args.multi_utility_w_safe),
            multi_utility_w_pareto=float(args.multi_utility_w_pareto),
            multi_utility_w_yaw=float(args.multi_utility_w_yaw),
            multi_utility_w_geom=float(args.multi_utility_w_geom),
            multi_utility_w_risk=float(args.multi_utility_w_risk),
        )

    rows = []
    for name, report in reports.items():
        all_rows = report.get("all_rows", {})
        rows.append(
            {
                "name": name,
                "checkpoint": report.get("checkpoint", ""),
                "selected_best_safe_hit_rate": _select(report, "all_rows", "selected_best_safe_hit_rate"),
                "selected_pareto_hit_rate": _select(report, "all_rows", "selected_pareto_hit_rate"),
                "selected_yaw_presence_rate": _select(report, "all_rows", "selected_yaw_presence_rate"),
                "selected_yaw_match_rate": _select(report, "all_rows", "selected_yaw_match_rate"),
                "selected_correct_yaw_sign_rate": _select(report, "all_rows", "selected_correct_yaw_sign_rate"),
                "best_safe_score_rank_mean": _select(report, "all_rows", "best_safe_score_rank_mean"),
                "pareto_best_score_rank_mean": _select(report, "all_rows", "pareto_best_score_rank_mean"),
                "score_utility_spearman_mean": _select(report, "all_rows", "score_utility_spearman_mean"),
                "score_geom_gain_spearman_mean": _select(report, "all_rows", "score_geom_gain_spearman_mean"),
                "score_risk_delta_spearman_mean": _select(report, "all_rows", "score_risk_delta_spearman_mean"),
                "set_best_safe_recall_at_k": _select(report, "all_rows", "set_best_safe_recall_at_k"),
                "set_pareto_hit_rate_at_k": _select(report, "all_rows", "set_pareto_hit_rate_at_k"),
                "set_yaw_opportunity_recall_at_k": _select(report, "all_rows", "set_yaw_opportunity_recall_at_k"),
                "set_yaw_match_recall_at_k": _select(report, "all_rows", "set_yaw_match_recall_at_k"),
                "topk_min_dist_to_best_safe_mean": _select(report, "all_rows", "topk_min_dist_to_best_safe_mean"),
                "topk_best_safe_geometry_gain_mean": _select(report, "all_rows", "topk_best_safe_geometry_gain_mean"),
                "topk_best_safe_risk_delta_mean": _select(report, "all_rows", "topk_best_safe_risk_delta_mean"),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: (r["selected_pareto_hit_rate"], r["selected_best_safe_hit_rate"]), reverse=True)
    best_set = max(rows, key=lambda r: (r["set_best_safe_recall_at_k"], r["set_pareto_hit_rate_at_k"]))
    best_selected = max(rows, key=lambda r: (r["selected_pareto_hit_rate"], r["selected_best_safe_hit_rate"]))
    proprio = reports.get("formal_proprio", {})
    depthforce = reports.get("formal_depthforce", {})
    v5_anchor = reports.get("v5_k8_proprio", {})

    recommendation = "ranking/observability bottleneck"
    if best_set["set_best_safe_recall_at_k"] < 0.15 and best_set["set_pareto_hit_rate_at_k"] < 0.25:
        recommendation = "proposal upper bound likely insufficient"
    elif (
        _select(depthforce, "all_rows", "selected_pareto_hit_rate")
        > _select(proprio, "all_rows", "selected_pareto_hit_rate") + 0.02
        or _select(depthforce, "all_rows", "score_utility_spearman_mean")
        > _select(proprio, "all_rows", "score_utility_spearman_mean") + 0.02
    ):
        recommendation = "observability improves with extra cues; depth/visual cues are worth testing"
    elif _select(v5_anchor, "all_rows", "set_best_safe_recall_at_k") >= 0.20 and _select(v5_anchor, "all_rows", "set_pareto_hit_rate_at_k") >= 0.30:
        recommendation = "proposal set is already sufficient; ranking supervision is the bottleneck"

    report = {
        "dataset_npz": args.dataset_npz,
        "yaw_presence_threshold": float(args.yaw_presence_threshold),
        "yaw_match_tol": float(args.yaw_match_tol),
        "distance_tol": float(args.distance_tol),
        "checkpoints": rows_sorted,
        "best_set_upper_bound": best_set,
        "best_selected": best_selected,
        "recommendation": recommendation,
        "raw_reports": _jsonable(reports),
    }

    out_json = Path(args.output_json) if args.output_json else Path("score_observability_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
