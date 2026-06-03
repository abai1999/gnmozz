#!/usr/bin/env python3
"""Audit micro-stability for small_xy_large_yaw active rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.c2c_v2_grasp_probe_metrics import grasp_probe_xy_metric_fields, safe_float, safe_int, trace_vec, xy_norm


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_filter(text: str | None) -> set[int]:
    if not text:
        return set()
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.add(safe_int(part))
    return out


def _row_step_idx(row: Mapping[str, Any]) -> int:
    return safe_int(row.get("step_idx", row.get("step", -1)))


def _trace_pre_vec(row: Mapping[str, Any]) -> np.ndarray:
    if row.get("grasp_probe_pre_true_error_t") is not None:
        return trace_vec(row, "grasp_probe_pre_true_error_t")
    if row.get("true_basin_error_t") is not None:
        return trace_vec(row, "true_basin_error_t")
    return np.full((4,), np.nan, dtype=np.float64)


def _trace_post_vec(row: Mapping[str, Any]) -> np.ndarray:
    if row.get("grasp_probe_horizon_final_true_error_t") is not None:
        return trace_vec(row, "grasp_probe_horizon_final_true_error_t")
    if row.get("grasp_probe_post_true_error_t") is not None:
        return trace_vec(row, "grasp_probe_post_true_error_t")
    if row.get("true_basin_error_t_plus_1") is not None:
        return trace_vec(row, "true_basin_error_t_plus_1")
    return np.full((4,), np.nan, dtype=np.float64)


def _residual_norm(row: Mapping[str, Any]) -> float:
    metrics = grasp_probe_xy_metric_fields(row)
    value = safe_float(metrics["scalar_xy_before"])
    if np.isfinite(value):
        return value
    return xy_norm(_trace_pre_vec(row)[:2])


def _final_xy_norm(row: Mapping[str, Any]) -> float:
    metrics = grasp_probe_xy_metric_fields(row)
    value = safe_float(metrics["scalar_xy_after"])
    if np.isfinite(value):
        return value
    return xy_norm(_trace_post_vec(row)[:2])


def _axis_abs_contracted(row: Mapping[str, Any], axis: str) -> bool:
    idx = 0 if axis == "x" else 1
    pre = _trace_pre_vec(row)
    post = _trace_post_vec(row)
    if not np.all(np.isfinite(pre[:2])) or not np.all(np.isfinite(post[:2])):
        return False
    return bool(abs(float(post[idx])) <= abs(float(pre[idx])) + 1.0e-9)


def _near_entry(row: Mapping[str, Any]) -> bool:
    if "grasp_probe_horizon_near_grasp_after" in row:
        return bool(row.get("grasp_probe_horizon_near_grasp_after", False))
    if "grasp_probe_near_grasp_after" in row:
        return bool(row.get("grasp_probe_near_grasp_after", False))
    return bool(safe_float(_final_xy_norm(row)) <= 0.015)


def _micro_entry_ready_after(row: Mapping[str, Any]) -> bool:
    if "grasp_probe_horizon_micro_entry_ready_after" in row:
        return bool(row.get("grasp_probe_horizon_micro_entry_ready_after", False))
    if "grasp_probe_micro_entry_ready_after" in row:
        return bool(row.get("grasp_probe_micro_entry_ready_after", False))
    return _near_entry(row)


def _overshoot(row: Mapping[str, Any]) -> bool:
    if "grasp_probe_horizon_overshoot" in row:
        return bool(row.get("grasp_probe_horizon_overshoot", False))
    if "grasp_probe_overshoot" in row:
        return bool(row.get("grasp_probe_overshoot", False))
    return False


def _direction_hint(row: Mapping[str, Any]) -> str:
    hint = str(row.get("direction_hint", "") or "").strip()
    if hint:
        return hint
    cosine = safe_float(row.get("oracle_xy_step_cosine_to_residual", float("nan")))
    if np.isfinite(cosine):
        if cosine <= -0.35:
            return "direction_flip_candidate"
        if cosine >= 0.65:
            return "step_too_small_candidate"
        return "mixed_or_unclear"
    return "unknown"


def _bucket_name(norm: float) -> str:
    if not np.isfinite(norm):
        return "unknown"
    if norm < 0.001:
        return "<0.001"
    if norm < 0.005:
        return "0.001-0.005"
    if norm < 0.015:
        return "0.005-0.015"
    return ">0.015"


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0


def _percentile(values: Iterable[float], p: float) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.percentile(arr, p)) if arr.size else 0.0


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    failure_bucket: str,
    episodes: set[int],
    active_only: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        stage_name = str(row.get("stage_name", row.get("c2c_v2_stage", "")) or "")
        if stage_name != "RING_GRASP_ALIGN":
            continue
        skill_type = str(row.get("skill_type", row.get("c2c_v2_skill_type", "")) or "")
        if skill_type != "precision_grasp":
            continue
        if str(row.get("failure_bucket", "")) != failure_bucket:
            continue
        if episodes and safe_int(row.get("episode_idx", -1)) not in episodes:
            continue
        active = bool(row.get("intervention_active", False)) or bool(row.get("grasp_probe_active", False))
        if active_only and not active:
            continue
        selected.append(dict(row))
    selected.sort(key=lambda row: (safe_int(row.get("episode_idx", -1)), _row_step_idx(row)))
    return selected


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "scalar_xy_contraction_rate": 0.0,
            "vector_norm_contraction_rate": 0.0,
            "near_entry_rate": 0.0,
            "micro_entry_ready_after_rate": 0.0,
            "overshoot_rate": 0.0,
            "mean_final_xy_norm": 0.0,
            "p50_final_xy_norm": 0.0,
            "p90_final_xy_norm": 0.0,
            "axis_abs_contraction_rate": {"x": 0.0, "y": 0.0},
            "direction_hint_counts": {},
            "mean_oracle_xy_step_cosine_to_residual": 0.0,
        }
    metrics = [grasp_probe_xy_metric_fields(r) for r in rows]
    final_xy = [_final_xy_norm(r) for r in rows]
    return {
        "rows": int(len(rows)),
        "scalar_xy_contraction_rate": float(np.mean([bool(m["scalar_xy_contracted"]) for m in metrics])),
        "vector_norm_contraction_rate": float(np.mean([bool(m["vector_norm_contracted"]) for m in metrics])),
        "near_entry_rate": float(np.mean([_near_entry(r) for r in rows])),
        "micro_entry_ready_after_rate": float(np.mean([_micro_entry_ready_after(r) for r in rows])),
        "overshoot_rate": float(np.mean([_overshoot(r) for r in rows])),
        "mean_final_xy_norm": _mean(final_xy),
        "p50_final_xy_norm": _percentile(final_xy, 50.0),
        "p90_final_xy_norm": _percentile(final_xy, 90.0),
        "axis_abs_contraction_rate": {
            "x": float(np.mean([_axis_abs_contracted(r, "x") for r in rows])),
            "y": float(np.mean([_axis_abs_contracted(r, "y") for r in rows])),
        },
        "direction_hint_counts": dict(Counter(_direction_hint(r) for r in rows)),
        "mean_oracle_xy_step_cosine_to_residual": _mean(
            safe_float(r.get("oracle_xy_step_cosine_to_residual", float("nan"))) for r in rows
        ),
        "mean_scalar_xy_delta": _mean(float(m["scalar_xy_delta"]) for m in metrics),
        "mean_vector_xy_delta": _mean(float(m["vector_xy_delta"]) for m in metrics),
    }


def audit(
    rows: list[dict[str, Any]],
    *,
    failure_bucket: str,
    episodes: set[int],
    active_only: bool,
) -> dict[str, Any]:
    selected = _select_rows(rows, failure_bucket=failure_bucket, episodes=episodes, active_only=active_only)
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bin_alias: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        alias = str(row.get("alias_drift_decision", row.get("yaw_alias_drift_decision", "unknown")) or "unknown")
        bin_name = _bucket_name(_residual_norm(row))
        by_bin[bin_name].append(row)
        by_alias[alias].append(row)
        by_bin_alias[(bin_name, alias)].append(row)
    return {
        "schema_version": "small_xy_micro_stability_audit_v1",
        "failure_bucket": failure_bucket,
        "episodes": sorted(int(ep) for ep in episodes),
        "active_only": bool(active_only),
        "overall": _summarize(selected),
        "by_residual_norm_bin": [
            {"residual_norm_bin": bin_name, **_summarize(subset)}
            for bin_name, subset in sorted(by_bin.items(), key=lambda item: item[0])
        ],
        "by_alias_drift_decision": [
            {"alias_drift_decision": alias, **_summarize(subset)}
            for alias, subset in sorted(by_alias.items(), key=lambda item: item[0])
        ],
        "by_residual_norm_bin_and_alias": [
            {"residual_norm_bin": bin_name, "alias_drift_decision": alias, **_summarize(subset)}
            for (bin_name, alias), subset in sorted(by_bin_alias.items(), key=lambda item: (item[0][0], item[0][1]))
        ],
        "rows": selected,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit micro-stability for small_xy_large_yaw active rows.")
    ap.add_argument("--rows_jsonl", type=Path, action="append", required=True)
    ap.add_argument("--failure_bucket", type=str, default="small_xy_large_yaw")
    ap.add_argument("--episodes", type=str, default="4,16,18,27,29")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--active_only", action="store_true", default=True)
    args = ap.parse_args()

    rows = [row for path in args.rows_jsonl for row in _read_jsonl(path)]
    report = audit(
        rows,
        failure_bucket=str(args.failure_bucket),
        episodes=_episode_filter(args.episodes),
        active_only=bool(args.active_only),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "small_xy_micro_stability_audit.json"
    out_md = args.output_dir / "small_xy_micro_stability_audit.md"
    out_json.write_text(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [
        "# Small XY Micro Stability Audit",
        "",
        f"- rows: `{report['overall']['rows']}`",
        f"- scalar_xy_contraction_rate: `{report['overall']['scalar_xy_contraction_rate']:.3f}`",
        f"- vector_norm_contraction_rate: `{report['overall']['vector_norm_contraction_rate']:.3f}`",
        f"- near_entry_rate: `{report['overall']['near_entry_rate']:.3f}`",
        f"- micro_entry_ready_after_rate: `{report['overall']['micro_entry_ready_after_rate']:.3f}`",
        f"- overshoot_rate: `{report['overall']['overshoot_rate']:.3f}`",
        f"- mean_final_xy_norm: `{report['overall']['mean_final_xy_norm']:.6f}`",
        f"- p50_final_xy_norm: `{report['overall']['p50_final_xy_norm']:.6f}`",
        f"- p90_final_xy_norm: `{report['overall']['p90_final_xy_norm']:.6f}`",
        f"- mean_oracle_xy_step_cosine_to_residual: `{report['overall']['mean_oracle_xy_step_cosine_to_residual']:.3f}`",
        "",
        "## By Residual Bin",
    ]
    for item in report["by_residual_norm_bin"]:
        md_lines.append(
            f"- `{item['residual_norm_bin']}`: rows={int(item['rows'])}, "
            f"scalar_contract={float(item['scalar_xy_contraction_rate']):.3f}, "
            f"vector_contract={float(item['vector_norm_contraction_rate']):.3f}, "
            f"near={float(item['near_entry_rate']):.3f}, "
            f"overshoot={float(item['overshoot_rate']):.3f}"
        )
    md_lines.extend(["", "## By Alias"])
    for item in report["by_alias_drift_decision"]:
        md_lines.append(
            f"- `{item['alias_drift_decision']}`: rows={int(item['rows'])}, "
            f"scalar_contract={float(item['scalar_xy_contraction_rate']):.3f}, "
            f"vector_contract={float(item['vector_norm_contraction_rate']):.3f}, "
            f"near={float(item['near_entry_rate']):.3f}, "
            f"overshoot={float(item['overshoot_rate']):.3f}"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
