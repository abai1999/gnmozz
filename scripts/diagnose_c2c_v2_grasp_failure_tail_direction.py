#!/usr/bin/env python3
"""Per-step direction diagnostic for grasp failure-tail intervention rows.

This is an offline report generator for the specific question:

    when a hard-bucket row is active, is the oracle xy correction
    actually pointing at the privileged residual, or is it mostly
    biased, flipped, or simply too small?

The formal sign metric is `oracle_xy_step_cosine_to_residual`.  The older
`oracle_xy_step_cosine_to_descent` is still reported for compatibility, but
it is no longer the primary sign decision field.

The script is intentionally narrow.  It consumes already-joined failure-tail
intervention rows and summarizes one episode / bucket slice at a time so the
largest disagreement patterns are easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


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
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _xy_norm(dx: Any, dy: Any) -> float:
    dx_f = _safe_float(dx)
    dy_f = _safe_float(dy)
    return float(np.hypot(dx_f, dy_f)) if np.isfinite(dx_f) and np.isfinite(dy_f) else float("nan")


def _vec2(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 2:
            return _safe_float(arr[0]), _safe_float(arr[1])
    return float("nan"), float("nan")


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    if not all(np.isfinite(v) for v in (*a, *b)):
        return float("nan")
    return float(a[0] * b[0] + a[1] * b[1])


def _cosine(a: tuple[float, float], b: tuple[float, float]) -> float:
    denom = float(np.hypot(*a) * np.hypot(*b))
    if not np.isfinite(denom) or denom <= 1.0e-12:
        return float("nan")
    return float(_dot(a, b) / denom)


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else 0.0


def _text_or_default(value: Any, default: str = "unknown") -> str:
    if value is None:
        return str(default)
    text = str(value).strip()
    if not text or text == "None":
        return str(default)
    return text


def _episode_filter(text: str | None) -> set[int]:
    if not text:
        return set()
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _selected_rows(
    rows: list[dict[str, Any]],
    *,
    failure_bucket: str | None,
    episodes: set[int],
    active_only: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("stage_name", "")) != "RING_GRASP_ALIGN":
            continue
        if str(row.get("skill_type", "")) != "precision_grasp":
            continue
        if failure_bucket and str(row.get("failure_bucket", "")) != failure_bucket:
            continue
        if episodes and int(row.get("episode_idx", -1)) not in episodes:
            continue
        active = bool(row.get("intervention_active", row.get("grasp_probe_active", False)))
        if active_only and not active:
            continue
        selected.append(dict(row))
    selected.sort(key=lambda r: (_safe_int(r.get("episode_idx", -1)), _safe_int(r.get("step_idx", r.get("step", -1)))))
    return selected


def _merge_with_failure_tail_rows(
    rows: list[dict[str, Any]],
    failure_tail_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not failure_tail_rows:
        return rows
    merged: list[dict[str, Any]] = []
    tail_by_key: dict[tuple[int, int], dict[str, Any]] = {
        (_safe_int(row.get("episode_idx", -1)), _safe_int(row.get("step_idx", row.get("step", -1)))): dict(row)
        for row in failure_tail_rows
    }
    for row in rows:
        key = (_safe_int(row.get("episode_idx", -1)), _safe_int(row.get("step_idx", row.get("step", -1))))
        tail_row = tail_by_key.get(key)
        if tail_row is None:
            merged.append(dict(row))
            continue
        merged.append({**row, **tail_row})
    return merged


def _direction_hint(row: Mapping[str, Any]) -> str:
    if not bool(row.get("intervention_active", False)):
        return "inactive"
    step = (_safe_float(row.get("oracle_xy_step_dx")), _safe_float(row.get("oracle_xy_step_dy")))
    pre = (_safe_float(row.get("privileged_dx")), _safe_float(row.get("privileged_dy")))
    pre_norm = float(row.get("privileged_xy_norm", float("nan")))
    step_norm = float(row.get("oracle_xy_step_norm", float("nan")))
    cosine_to_residual = _safe_float(row.get("oracle_xy_step_cosine_to_residual", float("nan")))
    cosine_to_descent = _safe_float(row.get("oracle_xy_step_cosine_to_descent", float("nan")))
    if not np.isfinite(step_norm) or step_norm <= 1.0e-12:
        return "zero_step"
    # Keep the primary decision tied to the residual sign.  Only fall back to
    # the descent cosine when the residual cosine is unavailable.
    if np.isfinite(cosine_to_residual) and cosine_to_residual <= -0.35:
        return "direction_flip_candidate"
    if np.isfinite(cosine_to_residual) and cosine_to_residual >= 0.65 and np.isfinite(pre_norm) and pre_norm > 1.0e-9 and step_norm < 0.55 * pre_norm:
        return "step_too_small_candidate"
    if np.isfinite(cosine_to_residual) and cosine_to_residual >= 0.65 and np.isfinite(pre_norm) and pre_norm > 1.0e-9 and step_norm >= 0.55 * pre_norm:
        if np.isfinite(float(row.get("oracle_after_xy", float("nan")))) and np.isfinite(float(row.get("oracle_before_xy", float("nan")))):
            if float(row["oracle_after_xy"]) > float(row["oracle_before_xy"]) - 1.0e-9:
                return "fixed_bias_or_frame_offset_candidate"
        return "fixed_bias_or_frame_offset_candidate"
    if np.isfinite(cosine_to_residual):
        if cosine_to_residual >= 0.65:
            return "step_too_small_candidate" if np.isfinite(pre_norm) and pre_norm > 1.0e-9 and step_norm < 0.55 * pre_norm else "mixed_or_unclear"
        if cosine_to_residual <= -0.35:
            return "direction_flip_candidate"
    if not np.isfinite(cosine_to_residual) and np.isfinite(cosine_to_descent) and cosine_to_descent <= -0.35:
        return "direction_flip_candidate"
    if all(np.isfinite(v) for v in (*step, *pre)) and abs(_dot(step, pre)) <= 1.0e-12:
        return "orthogonal_or_unclear"
    return "mixed_or_unclear"


def build_direction_diagnostic(
    rows: list[dict[str, Any]],
    *,
    failure_tail_rows: list[dict[str, Any]] | None = None,
    failure_bucket: str | None = None,
    episodes: set[int] | None = None,
    active_only: bool = False,
) -> dict[str, Any]:
    joined_rows = _merge_with_failure_tail_rows(rows, failure_tail_rows)
    selected = _selected_rows(joined_rows, failure_bucket=failure_bucket, episodes=episodes or set(), active_only=active_only)
    per_row: list[dict[str, Any]] = []
    for row in selected:
        pre_source = row.get("grasp_probe_pre_true_error_t", row.get("true_residual", row.get("true_basin_error_t")))
        after_source = row.get(
            "grasp_probe_horizon_final_true_error_t",
            row.get("grasp_probe_post_true_error_t", row.get("true_basin_error_t_plus_1")),
        )
        privileged_dx, privileged_dy = _vec2(pre_source)
        oracle_before_xy = _safe_float(row.get("grasp_probe_pre_xy_error", row.get("oracle_xy_before", float("nan"))))
        oracle_after_xy = _safe_float(row.get("grasp_probe_horizon_final_xy_error", row.get("oracle_xy_after", float("nan"))))
        planner_before_xy = _safe_float(row.get("xy_error", row.get("planner_xy_before", oracle_before_xy)))
        planner_after_xy = _safe_float(row.get("next_xy_error", row.get("planner_xy_after", oracle_after_xy)))
        if after_source is None:
            after_dx = _safe_float(row.get("next_privileged_dx", float("nan")))
            after_dy = _safe_float(row.get("next_privileged_dy", float("nan")))
        else:
            after_dx, after_dy = _vec2(after_source)
        pre_dx, pre_dy = privileged_dx, privileged_dy
        step_dx, step_dy = _vec2(row.get("grasp_probe_applied_xy_step_local_6d", []))
        privileged_xy_norm = _xy_norm(privileged_dx, privileged_dy)
        oracle_xy_step_norm = _xy_norm(step_dx, step_dy)
        oracle_step_cosine_to_residual = _cosine((step_dx, step_dy), (pre_dx, pre_dy))
        oracle_step_cosine_to_descent = _cosine((step_dx, step_dy), (-pre_dx, -pre_dy))
        after_xy_norm = _xy_norm(after_dx, after_dy)
        after_minus_before = after_xy_norm - privileged_xy_norm if np.isfinite(after_xy_norm) and np.isfinite(privileged_xy_norm) else float("nan")
        step_ratio = oracle_xy_step_norm / privileged_xy_norm if np.isfinite(oracle_xy_step_norm) and np.isfinite(privileged_xy_norm) and privileged_xy_norm > 1.0e-12 else float("nan")
        planner_delta = planner_after_xy - planner_before_xy if np.isfinite(planner_before_xy) and np.isfinite(planner_after_xy) else float("nan")
        oracle_delta = oracle_after_xy - oracle_before_xy if np.isfinite(oracle_before_xy) and np.isfinite(oracle_after_xy) else float("nan")
        per_row.append(
            {
                "episode_idx": _safe_int(row.get("episode_idx", -1)),
                "step_idx": _safe_int(row.get("step_idx", row.get("step", -1))),
                "failure_bucket": str(row.get("failure_bucket", "")),
                "takeover_tier": str(row.get("takeover_tier", "")),
                "alias_drift_decision": _text_or_default(row.get("alias_drift_decision", None), _text_or_default(row.get("yaw_alias_drift_decision", None))),
                "window_protocol": str(row.get("window_protocol", row.get("grasp_probe_window_protocol", ""))),
                "intervention_active": bool(row.get("intervention_active", row.get("grasp_probe_active", False))),
                "intervention_reason": str(row.get("intervention_reason", row.get("grasp_probe_reason", ""))),
                "planner_xy_before": planner_before_xy,
                "planner_xy_after": planner_after_xy,
                "planner_xy_delta": planner_delta,
                "oracle_xy_before": oracle_before_xy,
                "oracle_xy_after": oracle_after_xy,
                "oracle_xy_delta": oracle_delta,
                "privileged_dx": privileged_dx,
                "privileged_dy": privileged_dy,
                "privileged_xy_norm": privileged_xy_norm,
                "oracle_xy_step_dx": step_dx,
                "oracle_xy_step_dy": step_dy,
                "oracle_xy_step_norm": oracle_xy_step_norm,
                "oracle_xy_step_cosine_to_residual": oracle_step_cosine_to_residual,
                "oracle_xy_step_cosine_to_descent": oracle_step_cosine_to_descent,
                "after_privileged_dx": after_dx,
                "after_privileged_dy": after_dy,
                "after_privileged_xy_norm": after_xy_norm,
                "after_minus_before_xy_norm": after_minus_before,
                "step_ratio_to_residual": step_ratio,
                "overshoot": bool(row.get("oracle_overshoot", row.get("grasp_probe_horizon_overshoot", row.get("overshoot", False)))),
                "direction_hint": _direction_hint(
                    {
                        **row,
                        "privileged_xy_norm": privileged_xy_norm,
                        "oracle_xy_step_norm": oracle_xy_step_norm,
                        "oracle_xy_step_cosine_to_residual": oracle_step_cosine_to_residual,
                        "oracle_xy_step_cosine_to_descent": oracle_step_cosine_to_descent,
                        "oracle_before_xy": oracle_before_xy,
                        "oracle_after_xy": oracle_after_xy,
                    }
                ),
            }
        )

    active = [row for row in per_row if bool(row.get("intervention_active", False))]
    by_episode_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_row:
        by_episode_rows[int(row["episode_idx"])].append(row)
    return {
        "schema_version": "grasp_failure_tail_direction_diagnostic_v1",
        "filters": {
            "failure_bucket": failure_bucket,
            "episodes": sorted(episodes or set()),
            "active_only": bool(active_only),
        },
        "overall": {
            "num_rows": int(len(per_row)),
            "active_rows": int(len(active)),
            "mean_privileged_xy_norm": _mean(row["privileged_xy_norm"] for row in active),
            "mean_oracle_step_norm": _mean(row["oracle_xy_step_norm"] for row in active),
            "mean_step_ratio_to_residual": _mean(row["step_ratio_to_residual"] for row in active),
            "mean_oracle_step_cosine_to_residual": _mean(row["oracle_xy_step_cosine_to_residual"] for row in active),
            "mean_oracle_step_cosine_to_descent": _mean(row["oracle_xy_step_cosine_to_descent"] for row in active),
            "primary_sign_metric": "oracle_xy_step_cosine_to_residual",
            "compat_sign_metric": "oracle_xy_step_cosine_to_descent",
            "mean_after_privileged_xy_norm": _mean(row["after_privileged_xy_norm"] for row in active),
            "mean_after_minus_before_xy_norm": _mean(row["after_minus_before_xy_norm"] for row in active),
            "mean_planner_xy_delta": _mean(row["planner_xy_delta"] for row in per_row),
            "mean_oracle_xy_delta": _mean(row["oracle_xy_delta"] for row in active),
            "overshoot_rate": float(np.mean([bool(row.get("overshoot", False)) for row in active])) if active else 0.0,
            "direction_hint_counts": dict(Counter(str(row.get("direction_hint", "")) for row in active)),
            "by_alias_drift_decision": dict(Counter(_text_or_default(row.get("alias_drift_decision", None), _text_or_default(row.get("yaw_alias_drift_decision", None))) for row in per_row)),
        },
        "by_episode": {
            f"ep{ep:03d}": {
                "num_rows": int(len(subset)),
                "active_rows": int(sum(bool(row.get("intervention_active", False)) for row in subset)),
                "mean_oracle_step_cosine_to_residual": _mean(row["oracle_xy_step_cosine_to_residual"] for row in subset if bool(row.get("intervention_active", False))),
                "mean_oracle_step_cosine_to_descent": _mean(row["oracle_xy_step_cosine_to_descent"] for row in subset if bool(row.get("intervention_active", False))),
                "mean_step_ratio_to_residual": _mean(row["step_ratio_to_residual"] for row in subset if bool(row.get("intervention_active", False))),
                "mean_after_privileged_xy_norm": _mean(row["after_privileged_xy_norm"] for row in subset if bool(row.get("intervention_active", False))),
                "direction_hint_counts": dict(Counter(str(row.get("direction_hint", "")) for row in subset if bool(row.get("intervention_active", False)))),
            }
            for ep, subset in sorted(by_episode_rows.items())
        },
        "rows": per_row,
    }


def _write_markdown(report: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Grasp Failure-Tail Direction Diagnostic",
        "",
        f"- rows: `{report['overall']['num_rows']}`",
        f"- active_rows: `{report['overall']['active_rows']}`",
        f"- mean_privileged_xy_norm: `{report['overall']['mean_privileged_xy_norm']:.6f}`",
        f"- mean_oracle_step_norm: `{report['overall']['mean_oracle_step_norm']:.6f}`",
        f"- mean_step_ratio_to_residual: `{report['overall']['mean_step_ratio_to_residual']:.3f}`",
        f"- mean_oracle_step_cosine_to_residual: `{report['overall']['mean_oracle_step_cosine_to_residual']:.3f}`",
        f"- mean_oracle_step_cosine_to_descent: `{report['overall']['mean_oracle_step_cosine_to_descent']:.3f}`",
        f"- primary_sign_metric: `{report['overall']['primary_sign_metric']}`",
        f"- compat_sign_metric: `{report['overall']['compat_sign_metric']}`",
        f"- mean_after_privileged_xy_norm: `{report['overall']['mean_after_privileged_xy_norm']:.6f}`",
        f"- mean_after_minus_before_xy_norm: `{report['overall']['mean_after_minus_before_xy_norm']:.6f}`",
        f"- overshoot_rate: `{report['overall']['overshoot_rate']:.3f}`",
        "",
        "## Direction Hints",
    ]
    for name, count in report["overall"]["direction_hint_counts"].items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Per-Step Rows", "", "| ep | step | bucket | tier | alias | pre_xy | step_xy | step_cos | after_xy | overshoot | hint |", "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in report["rows"]:
        lines.append(
            "| "
            f"{int(row['episode_idx']):03d} | "
            f"{int(row['step_idx'])} | "
            f"{row['failure_bucket']} | "
            f"{row['takeover_tier']} | "
            f"{row['alias_drift_decision']} | "
            f"{row['privileged_xy_norm']:.5f} | "
            f"{row['oracle_xy_step_norm']:.5f} | "
            f"{row['oracle_xy_step_cosine_to_residual']:.3f} | "
            f"{row['after_privileged_xy_norm']:.5f} | "
            f"{row['overshoot']} | "
            f"{row['direction_hint']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a per-step grasp failure-tail direction diagnostic.")
    ap.add_argument("--audit_rows_jsonl", type=Path, required=True)
    ap.add_argument(
        "--failure_tail_rows_jsonl",
        type=Path,
        default=None,
        help="Optional failure-tail audit rows JSONL joined by episode/step to recover active/after-residual fields.",
    )
    ap.add_argument("--failure_bucket", type=str, default="small_xy_large_yaw")
    ap.add_argument("--episode_indices", type=str, default="0")
    ap.add_argument("--active_only", action="store_true", default=False)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_failure_tail_direction_diagnostic"),
    )
    args = ap.parse_args()

    rows = _read_jsonl(args.audit_rows_jsonl)
    failure_tail_rows = _read_jsonl(args.failure_tail_rows_jsonl) if args.failure_tail_rows_jsonl is not None else None
    report = build_direction_diagnostic(
        rows,
        failure_tail_rows=failure_tail_rows,
        failure_bucket=args.failure_bucket or None,
        episodes=_episode_filter(args.episode_indices),
        active_only=bool(args.active_only),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "grasp_failure_tail_direction_diagnostic.json"
    out_md = output_dir / "grasp_failure_tail_direction_diagnostic.md"
    out_rows = output_dir / "grasp_failure_tail_direction_diagnostic_rows.jsonl"
    out_json.write_text(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True), encoding="utf-8")
    with open(out_rows, "w", encoding="utf-8") as handle:
        for row in report["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_markdown(report, out_md)
    print(out_json)
    print(out_md)
    print(out_rows)


if __name__ == "__main__":
    main()
