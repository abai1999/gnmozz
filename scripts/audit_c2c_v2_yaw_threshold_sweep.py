#!/usr/bin/env python3
"""Sweep non-privileged yaw-control gates against privileged frame residuals.

The sweep deliberately separates two questions:

* yaw_entry_feasible: privileged residual says yaw does not block near-grasp
  entry.
* yaw_control_observable: non-privileged visual evidence says runtime may trust
  yaw as a controllable axis.

Privileged yaw is used only for offline audit metrics here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.takeover_contract import NEAR_GRASP_YAW_THRESHOLD


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _corr(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray(list(a), dtype=np.float64)
    bb = np.asarray(list(b), dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) <= 1.0e-12 or float(np.std(bb)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _proxy_yaw(row: Mapping[str, Any]) -> float:
    proxy = row.get("proxy_local_geometry_error") if isinstance(row.get("proxy_local_geometry_error"), Mapping) else {}
    est = row.get("estimated_basin_error") if isinstance(row.get("estimated_basin_error"), Mapping) else {}
    return _safe_float(proxy.get("dyaw", est.get("dyaw", 0.0)), 0.0)


def _privileged_yaw(row: Mapping[str, Any]) -> float:
    nested = row.get("true_basin_error_t") if isinstance(row.get("true_basin_error_t"), Mapping) else {}
    return _safe_float(nested.get("dyaw", row.get("privileged_dyaw", float("nan"))), float("nan"))


def _xy_error(row: Mapping[str, Any]) -> float:
    if "xy_error" in row:
        return _safe_float(row.get("xy_error"), float("nan"))
    nested = row.get("true_basin_error_t") if isinstance(row.get("true_basin_error_t"), Mapping) else {}
    dx = _safe_float(nested.get("dx", row.get("privileged_dx", float("nan"))), float("nan"))
    dy = _safe_float(nested.get("dy", row.get("privileged_dy", float("nan"))), float("nan"))
    return float(math.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else float("nan")


def _yaw_entry_feasible(row: Mapping[str, Any], *, near_yaw: float) -> bool:
    if "yaw_entry_feasible" in row and abs(near_yaw - NEAR_GRASP_YAW_THRESHOLD) <= 1.0e-12:
        return bool(row.get("yaw_entry_feasible", False))
    yaw = abs(_privileged_yaw(row))
    return bool(np.isfinite(yaw) and yaw <= float(near_yaw) + 1.0e-9)


def _gate_values(row: Mapping[str, Any]) -> tuple[float, float, float, bool, bool, str]:
    obs_t = row.get("obs_t") if isinstance(row.get("obs_t"), Mapping) else {}
    conf = _safe_float(row.get("yaw_observability_frame_confidence", row.get("source_frame_confidence", obs_t.get("frame_confidence", 0.0))), 0.0)
    obs = _safe_float(row.get("yaw_observability_frame_observability", row.get("source_frame_observability", obs_t.get("frame_observability", 0.0))), 0.0)
    axis = _safe_float(row.get("yaw_observability_frame_axis_strength", row.get("source_frame_axis_strength", obs_t.get("frame_axis_strength", 0.0))), 0.0)
    wide = bool(row.get("yaw_observability_wide_ring_visible", row.get("wide_ring_visible", False)))
    occluded = bool(row.get("yaw_observability_wrist_occluded", row.get("wrist_is_occluded", False)))
    visual = str(row.get("visual_observability_class", obs_t.get("visual_observability_class", "prior_only")))
    return conf, obs, axis, wide, occluded, visual


def _control_observable(
    row: Mapping[str, Any],
    *,
    min_confidence: float,
    min_frame_observability: float,
    min_axis_strength: float,
    allow_wide_visible: bool,
) -> bool:
    conf, obs, axis, wide, occluded, visual = _gate_values(row)
    if visual == "prior_only" or occluded:
        return False
    if allow_wide_visible and wide and obs >= min_frame_observability:
        return bool(conf >= min_confidence and axis >= min_axis_strength)
    return bool(conf >= min_confidence and obs >= min_frame_observability and axis >= min_axis_strength)


def _sign_match_rate(proxy: list[float], priv: list[float]) -> float:
    pairs = [
        (p, q)
        for p, q in zip(proxy, priv)
        if np.isfinite(p) and np.isfinite(q) and abs(float(p)) > 1.0e-6 and abs(float(q)) > 1.0e-6
    ]
    if not pairs:
        return 0.0
    return float(np.mean([np.sign(p) == np.sign(q) for p, q in pairs]))


def _summarize_subset(rows: list[dict[str, Any]], *, near_yaw: float) -> dict[str, Any]:
    proxy = [_proxy_yaw(r) for r in rows]
    priv = [_privileged_yaw(r) for r in rows]
    abs_err = [
        abs(float(p) - float(q))
        for p, q in zip(proxy, priv)
        if np.isfinite(float(p)) and np.isfinite(float(q))
    ]
    return {
        "rows": int(len(rows)),
        "yaw_entry_feasible_rows": int(sum(_yaw_entry_feasible(r, near_yaw=near_yaw) for r in rows)),
        "near_basin_shell_rows": int(sum(bool(r.get("near_basin_shell", False)) for r in rows)),
        "micro_entry_ready_rows": int(sum(bool(r.get("micro_entry_ready", False)) for r in rows)),
        "proxy_privileged_yaw_mae": float(np.mean(abs_err)) if abs_err else 0.0,
        "proxy_privileged_yaw_corr": _corr(proxy, priv),
        "proxy_privileged_yaw_sign_match_rate": _sign_match_rate(proxy, priv),
        "mean_xy_error": float(np.nanmean([_xy_error(r) for r in rows])) if rows else 0.0,
        "mean_yaw_abs": float(np.nanmean([abs(_privileged_yaw(r)) for r in rows])) if rows else 0.0,
    }


def sweep(
    rows: list[dict[str, Any]],
    *,
    frame_observability_thresholds: list[float],
    confidence_thresholds: list[float],
    axis_strength_thresholds: list[float],
    near_yaw: float,
    allow_wide_visible: bool = False,
) -> dict[str, Any]:
    baseline_control = [r for r in rows if bool(r.get("yaw_control_observable", r.get("yaw_observable", False)))]
    entry_feasible = [r for r in rows if _yaw_entry_feasible(r, near_yaw=near_yaw)]
    items: list[dict[str, Any]] = []
    for min_obs in frame_observability_thresholds:
        for min_conf in confidence_thresholds:
            for min_axis in axis_strength_thresholds:
                selected = [
                    r
                    for r in rows
                    if _control_observable(
                        r,
                        min_confidence=float(min_conf),
                        min_frame_observability=float(min_obs),
                        min_axis_strength=float(min_axis),
                        allow_wide_visible=allow_wide_visible,
                    )
                ]
                selected_entry = [r for r in selected if _yaw_entry_feasible(r, near_yaw=near_yaw)]
                selected_entry_ids = {(int(r.get("episode_idx", -1)), int(r.get("step_idx", -1))) for r in selected_entry}
                baseline_ids = {(int(r.get("episode_idx", -1)), int(r.get("step_idx", -1))) for r in baseline_control}
                newly_released = [
                    r
                    for r in selected_entry
                    if (int(r.get("episode_idx", -1)), int(r.get("step_idx", -1))) not in baseline_ids
                ]
                item = {
                    "min_frame_observability": float(min_obs),
                    "min_confidence": float(min_conf),
                    "min_axis_strength": float(min_axis),
                    "control_observable_rows": int(len(selected)),
                    "entry_feasible_control_observable_rows": int(len(selected_entry)),
                    "entry_feasible_control_observable_rate": float(len(selected_entry) / len(entry_feasible)) if entry_feasible else 0.0,
                    "newly_released_entry_feasible_rows": int(len(newly_released)),
                    "selected": _summarize_subset(selected, near_yaw=near_yaw),
                    "entry_feasible_selected": _summarize_subset(selected_entry, near_yaw=near_yaw),
                    "newly_released_entry_feasible": _summarize_subset(newly_released, near_yaw=near_yaw),
                }
                items.append(item)
    items.sort(
        key=lambda item: (
            -int(item["entry_feasible_control_observable_rows"]),
            -float(item["entry_feasible_selected"]["proxy_privileged_yaw_sign_match_rate"]),
            float(item["entry_feasible_selected"]["proxy_privileged_yaw_mae"]),
        )
    )
    return {
        "schema_version": "yaw_threshold_sweep_v1",
        "near_yaw_threshold": float(near_yaw),
        "allow_wide_visible": bool(allow_wide_visible),
        "overall": {
            "rows": int(len(rows)),
            "baseline_control_observable_rows": int(len(baseline_control)),
            "yaw_entry_feasible_rows": int(len(entry_feasible)),
            "entry_feasible_control_overlap_rows": int(
                sum(bool(r.get("yaw_control_observable", r.get("yaw_observable", False))) for r in entry_feasible)
            ),
            "entry_feasible_control_blocked_rows": int(
                sum(not bool(r.get("yaw_control_observable", r.get("yaw_observable", False))) for r in entry_feasible)
            ),
            "baseline_control": _summarize_subset(baseline_control, near_yaw=near_yaw),
            "entry_feasible": _summarize_subset(entry_feasible, near_yaw=near_yaw),
        },
        "sweep": items,
        "best_by_entry_feasible_rows": items[:10],
    }


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep C2C v2 yaw observability gate thresholds on frame_residual_v2 labels.")
    ap.add_argument("--relabel_jsonl", type=Path, required=True)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/yaw_threshold_sweep"),
    )
    ap.add_argument("--frame_observability_thresholds", type=str, default="0.005,0.01,0.02,0.05,0.10")
    ap.add_argument("--confidence_thresholds", type=str, default="0.20,0.30,0.50")
    ap.add_argument("--axis_strength_thresholds", type=str, default="0.60,0.80")
    ap.add_argument("--near_yaw_threshold", type=float, default=NEAR_GRASP_YAW_THRESHOLD)
    ap.add_argument("--allow_wide_visible", action="store_true", default=False)
    args = ap.parse_args()

    rows = _read_jsonl(args.relabel_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {args.relabel_jsonl}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = sweep(
        rows,
        frame_observability_thresholds=_parse_float_list(args.frame_observability_thresholds),
        confidence_thresholds=_parse_float_list(args.confidence_thresholds),
        axis_strength_thresholds=_parse_float_list(args.axis_strength_thresholds),
        near_yaw=float(args.near_yaw_threshold),
        allow_wide_visible=bool(args.allow_wide_visible),
    )
    report["source_jsonl"] = str(args.relabel_jsonl.resolve())
    out_json = output_dir / "yaw_threshold_sweep.json"
    out_md = output_dir / "yaw_threshold_sweep.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Yaw Threshold Sweep",
        "",
        f"- source: `{args.relabel_jsonl}`",
        f"- rows: `{report['overall']['rows']}`",
        f"- yaw_entry_feasible_rows: `{report['overall']['yaw_entry_feasible_rows']}`",
        f"- baseline_control_observable_rows: `{report['overall']['baseline_control_observable_rows']}`",
        f"- entry_feasible_control_blocked_rows: `{report['overall']['entry_feasible_control_blocked_rows']}`",
        "",
        "## Top Thresholds",
    ]
    for item in report["best_by_entry_feasible_rows"][:10]:
        selected = item["entry_feasible_selected"]
        released = item["newly_released_entry_feasible"]
        lines.append(
            f"- obs>={item['min_frame_observability']:.3f}, conf>={item['min_confidence']:.2f}, "
            f"axis>={item['min_axis_strength']:.2f}: entry_control={item['entry_feasible_control_observable_rows']}, "
            f"new={item['newly_released_entry_feasible_rows']}, sign={selected['proxy_privileged_yaw_sign_match_rate']:.3f}, "
            f"corr={selected['proxy_privileged_yaw_corr']:.3f}, mae={selected['proxy_privileged_yaw_mae']:.3f}, "
            f"new_sign={released['proxy_privileged_yaw_sign_match_rate']:.3f}, new_mae={released['proxy_privileged_yaw_mae']:.3f}"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
