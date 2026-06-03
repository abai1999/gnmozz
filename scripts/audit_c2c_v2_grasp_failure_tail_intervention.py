#!/usr/bin/env python3
"""Audit C2C v2 grasp intervention evidence on planner failure-tail rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scripts.c2c_v2_grasp_probe_metrics import grasp_probe_xy_metric_fields, safe_float, safe_int, trace_vec, xy_norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_c2c_v2_grasp_intervention import _load_trace_rows  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_trace_rows_from_dirs(trace_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for trace_dir in trace_dirs:
        for row in _load_trace_rows(trace_dir):
            key = (
                int(row.get("episode_idx", -1)),
                int(row.get("step", row.get("step_idx", -1))),
                str(trace_dir.resolve()),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _xy_from_mapping(mapping: Mapping[str, Any]) -> float:
    dx = safe_float(mapping.get("dx", float("nan")))
    dy = safe_float(mapping.get("dy", float("nan")))
    if not np.isfinite(dx) or not np.isfinite(dy):
        return float("nan")
    return float(np.hypot(dx, dy))


def _trace_horizon_post(row: Mapping[str, Any]) -> np.ndarray:
    if row.get("grasp_probe_horizon_final_true_error_t") is not None:
        return trace_vec(row, "grasp_probe_horizon_final_true_error_t")
    return trace_vec(row, "grasp_probe_post_true_error_t")


def _candidate_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (safe_int(row.get("episode_idx", -1)), safe_int(row.get("step_idx", row.get("step", -1))))


def _trace_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (safe_int(row.get("episode_idx", -1)), safe_int(row.get("step", row.get("step_idx", -1))))


def _planner_xy(row: Mapping[str, Any]) -> tuple[float, float]:
    return (safe_float(row.get("xy_error", float("nan"))), safe_float(row.get("next_xy_error", float("nan"))))


def _oracle_xy(row: Mapping[str, Any]) -> tuple[float, float]:
    pre = trace_vec(row, "grasp_probe_pre_true_error_t")
    post = _trace_horizon_post(row)
    if not np.all(np.isfinite(pre[:2])) or not np.all(np.isfinite(post[:2])):
        return (float("nan"), float("nan"))
    return (float(np.hypot(float(pre[0]), float(pre[1]))), float(np.hypot(float(post[0]), float(post[1]))))


def _oracle_near(row: Mapping[str, Any]) -> bool:
    if "grasp_probe_horizon_near_grasp_after" in row:
        return bool(row.get("grasp_probe_horizon_near_grasp_after", False))
    return bool(row.get("grasp_probe_near_grasp_after", False))


def _planner_near_next(row: Mapping[str, Any], *, near_xy: float = 0.015, near_yaw: float = 0.08) -> bool:
    xy = safe_float(row.get("next_xy_error", float("nan")))
    yaw = safe_float(row.get("next_yaw_abs", float("nan")))
    return bool(np.isfinite(xy) and np.isfinite(yaw) and xy <= near_xy and yaw <= near_yaw)


def _blocked_reason_layer(row: Mapping[str, Any]) -> str:
    if bool(row.get("intervention_active", False)):
        return "active"
    if bool(row.get("xy_correction_ready", row.get("grasp_probe_xy_correction_ready", False))):
        return "xy_ready"
    reason = str(row.get("intervention_reason", ""))
    tier = str(row.get("takeover_tier", ""))
    yaw_observable = bool(row.get("yaw_control_observable", row.get("yaw_observable", False)))
    if reason in {"missing_trace", "failure_tail_candidate_abstain"}:
        return "candidate_actionable"
    if tier in {"abstain_prior_only", "invalid"}:
        return "candidate_actionable"
    if str(row.get("abstain_reason", "")):
        return "candidate_actionable"
    if tier == "yaw_entry_blocked" or not yaw_observable or str(row.get("yaw_observability_class", "")) in {"ambiguous", "unobservable"}:
        return "yaw"
    if tier in {"too_far", "outside_takeover"}:
        return "xy"
    return "candidate_actionable"


def _joined_rows(candidates: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_by_key = {_trace_key(row): row for row in trace_rows}
    joined: list[dict[str, Any]] = []
    for cand in candidates:
        trace = trace_by_key.get(_candidate_key(cand), {})
        planner_pre, planner_next = _planner_xy(cand)
        oracle_pre, oracle_next = _oracle_xy(trace) if trace else (float("nan"), float("nan"))
        xy_metrics = grasp_probe_xy_metric_fields(trace or cand)
        planner_delta = planner_next - planner_pre if np.isfinite(planner_pre) and np.isfinite(planner_next) else float("nan")
        oracle_delta = oracle_next - oracle_pre if np.isfinite(oracle_pre) and np.isfinite(oracle_next) else float("nan")
        joined.append(
            {
                **cand,
                "alias_drift_decision": str(
                    cand.get(
                        "alias_drift_decision",
                        trace.get("alias_drift_decision", trace.get("yaw_alias_drift_decision", "unknown")) if trace else "unknown",
                    )
                ),
                "alias_label": str(
                    cand.get("alias_label", trace.get("alias_label", "unknown") if trace else "unknown")
                ),
                "acceptance_role": str(
                    cand.get("acceptance_role", trace.get("acceptance_role", "unknown") if trace else "unknown")
                ),
                "intervention_trace_found": bool(trace),
                "intervention_active": bool(trace.get("grasp_probe_active", False)) if trace else False,
                "intervention_reason": str(trace.get("grasp_probe_reason", "missing_trace")) if trace else "missing_trace",
                "planner_xy_before": planner_pre,
                "planner_xy_after": planner_next,
                "planner_xy_delta": planner_delta,
                "planner_natural_contracted": bool(np.isfinite(planner_delta) and planner_delta < -1.0e-9),
                "planner_natural_near_grasp_next": bool(_planner_near_next(cand)),
                "oracle_xy_before": oracle_pre,
                "oracle_xy_after": oracle_next,
                "oracle_xy_delta": oracle_delta,
                "oracle_contracted": bool(np.isfinite(oracle_delta) and oracle_delta < -1.0e-9),
                "oracle_near_grasp_after": bool(_oracle_near(trace)) if trace else False,
                "oracle_overshoot": bool(trace.get("grasp_probe_horizon_overshoot", trace.get("grasp_probe_overshoot", False))) if trace else False,
                "intervention_vs_planner_improvement": bool(
                    np.isfinite(planner_next)
                    and np.isfinite(oracle_next)
                    and oracle_next < planner_next - 1.0e-9
                ),
                "scalar_xy_before": xy_metrics["scalar_xy_before"],
                "scalar_xy_after": xy_metrics["scalar_xy_after"],
                "scalar_xy_delta": xy_metrics["scalar_xy_delta"],
                "scalar_xy_contracted": xy_metrics["scalar_xy_contracted"],
                "vector_xy_before": xy_metrics["vector_xy_before"],
                "vector_xy_after": xy_metrics["vector_xy_after"],
                "vector_xy_delta": xy_metrics["vector_xy_delta"],
                "vector_norm_contracted": xy_metrics["vector_norm_contracted"],
                "scalar_vector_xy_agree": xy_metrics["scalar_vector_xy_agree"],
                "scalar_xy_before_source": xy_metrics["scalar_xy_before_source"],
                "scalar_xy_after_source": xy_metrics["scalar_xy_after_source"],
            }
        )
    return joined


def _rate(rows: list[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([bool(row.get(key, False)) for row in rows])) if rows else 0.0


def _wilson_lower_bound(successes: int, n: int, *, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = float(successes) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = phat + (z * z) / (2.0 * float(n))
    margin = z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * float(n))) / float(n))
    return float(max((center - margin) / denom, 0.0))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [r for r in rows if bool(r.get("intervention_active", False))]
    planner_contract_count = int(sum(bool(r.get("planner_natural_contracted", False)) for r in rows))
    oracle_contract_count = int(sum(bool(r.get("oracle_contracted", False)) for r in active))
    blocked_reason_counts = dict(Counter(_blocked_reason_layer(r) for r in rows))
    return {
        "num_rows": int(len(rows)),
        "intervention_trace_found_rows": int(sum(bool(r.get("intervention_trace_found", False)) for r in rows)),
        "active_failure_tail_rows": int(len(active)),
        "planner_natural_contraction_rate": _rate(rows, "planner_natural_contracted"),
        "planner_natural_contraction_lower_ci": _wilson_lower_bound(planner_contract_count, len(rows)),
        "oracle_intervention_contraction_rate": _rate(active, "oracle_contracted"),
        "oracle_intervention_contraction_lower_ci": _wilson_lower_bound(oracle_contract_count, len(active)),
        "intervention_vs_planner_improvement_rate": _rate(active, "intervention_vs_planner_improvement"),
        "scalar_xy_contraction_rate": _rate(active, "scalar_xy_contracted"),
        "vector_norm_contraction_rate": _rate(active, "vector_norm_contracted"),
        "scalar_vector_xy_agreement_rate": _rate(active, "scalar_vector_xy_agree"),
        "mean_scalar_xy_delta": float(np.mean([float(r.get("scalar_xy_delta", float("nan"))) for r in active if np.isfinite(float(r.get("scalar_xy_delta", float("nan"))))])) if active else 0.0,
        "mean_vector_xy_delta": float(np.mean([float(r.get("vector_xy_delta", float("nan"))) for r in active if np.isfinite(float(r.get("vector_xy_delta", float("nan"))))])) if active else 0.0,
        "planner_natural_near_grasp_next_rate": _rate(rows, "planner_natural_near_grasp_next"),
        "oracle_near_grasp_after_rate": _rate(active, "oracle_near_grasp_after"),
        "near_grasp_entry_gain": _rate(active, "oracle_near_grasp_after") - _rate(rows, "planner_natural_near_grasp_next"),
        "overshoot_rate": _rate(active, "oracle_overshoot"),
        "abstain_correct_rate": float(
            np.mean([not bool(r.get("intervention_active", False)) for r in rows if str(r.get("abstain_reason", ""))])
        ) if any(str(r.get("abstain_reason", "")) for r in rows) else 1.0,
        "takeover_tier_counts": dict(Counter(str(r.get("takeover_tier", "")) for r in rows)),
        "intervention_reason_counts": dict(Counter(str(r.get("intervention_reason", "")) for r in rows)),
        "blocked_reason_counts": blocked_reason_counts,
        "xy_correction_ready_rows": int(sum(bool(r.get("xy_correction_ready", r.get("grasp_probe_xy_correction_ready", False))) for r in rows)),
    }


def _group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    for value, subset in sorted(groups.items()):
        out.append({key: value, **_summary(subset)})
    return out


def _group_blocked_reason(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_blocked_reason_layer(row)].append(row)
    out: list[dict[str, Any]] = []
    for value, subset in sorted(groups.items()):
        out.append({"blocked_reason": value, **_summary(subset)})
    return out


def audit(candidates: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    joined = _joined_rows(candidates, trace_rows)
    return {
        "overall": _summary(joined),
        "by_failure_bucket": _group(joined, "failure_bucket"),
        "by_blocked_reason": _group_blocked_reason(joined),
        "by_takeover_tier": _group(joined, "takeover_tier"),
        "by_yaw_observability": _group(joined, "yaw_observability_class"),
        "by_visual_bucket": _group(joined, "visual_observability_class"),
        "active_by_alias_drift_decision": _group(joined, "alias_drift_decision"),
        "contraction_by_alias_drift_decision": _group(joined, "alias_drift_decision"),
        "near_entry_by_alias_drift_decision": _group(joined, "alias_drift_decision"),
        "by_episode": _group(joined, "episode_idx"),
        "by_stage": _group(joined, "stage_name"),
        "joined_rows": joined,
        "runtime_invariants": {
            "uses_privileged_runtime": False,
            "uses_privileged_target": False,
            "uses_privileged_label_for_eval": True,
            "uses_rlbench_mask_runtime": False,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit grasp failure-tail intervention against planner natural outcome.")
    ap.add_argument("--candidate_jsonl", type=Path, required=True)
    ap.add_argument("--trace_dir", type=Path, action="append", default=None)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_failure_tail_intervention"),
    )
    args = ap.parse_args()

    candidates = _read_jsonl(args.candidate_jsonl)
    trace_dirs = list(args.trace_dir or [])
    trace_rows = _load_trace_rows_from_dirs(trace_dirs) if trace_dirs else []
    report = audit(candidates, trace_rows)
    report["source_candidate_jsonl"] = str(args.candidate_jsonl.resolve())
    report["source_trace_dir"] = [str(path.resolve()) for path in trace_dirs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "grasp_failure_tail_intervention_audit.json"
    out_rows = args.output_dir / "grasp_failure_tail_intervention_rows.jsonl"
    out_md = args.output_dir / "grasp_failure_tail_intervention_audit.md"
    out_json.write_text(json.dumps({k: v for k, v in report.items() if k != "joined_rows"}, indent=2, sort_keys=True), encoding="utf-8")
    with open(out_rows, "w", encoding="utf-8") as handle:
        for row in report["joined_rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    lines = [
        "# C2C v2 Grasp Failure-Tail Intervention Audit",
        "",
        f"- candidates: `{len(candidates)}`",
        f"- trace_rows: `{len(trace_rows)}`",
        "",
        "## Overall",
    ]
    for key, value in report["overall"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blocked Reasons"])
    for item in report["by_blocked_reason"]:
        lines.append(f"- `{item['blocked_reason']}`: `{item['num_rows']}`")
    lines.extend(["", "## Alias/Drift Split"])
    alias_unknown = next((item for item in report["active_by_alias_drift_decision"] if item["alias_drift_decision"] == "unknown"), None)
    total_alias_rows = sum(int(item["num_rows"]) for item in report["active_by_alias_drift_decision"])
    unknown_rows = int(alias_unknown["num_rows"]) if alias_unknown is not None else 0
    lines.append(f"- unknown_rows: `{unknown_rows}`")
    lines.append(f"- unknown_rate: `{(unknown_rows / total_alias_rows) if total_alias_rows else 0.0:.3f}`")
    for item in report["active_by_alias_drift_decision"]:
        lines.append(
            f"- `{item['alias_drift_decision']}`: active={int(item['active_failure_tail_rows'])}, "
            f"xy_contract={float(item['oracle_intervention_contraction_rate']):.3f}, "
            f"near={float(item['oracle_near_grasp_after_rate']):.3f}, "
            f"overshoot={float(item['overshoot_rate']):.3f}"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)
    print(out_rows)


if __name__ == "__main__":
    main()
