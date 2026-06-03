#!/usr/bin/env python3
"""Diagnose why runtime-style C2C rows become `prior_only_abstain`.

The report is intentionally runtime-facing: privileged residuals may be present
in the trace under `offline_eval_only`, but this script does not use them to
classify the abstain source.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.runtime_xy_residual import estimate_runtime_xy_residual_from_trace


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_from_path(path: Path) -> int:
    for token in path.stem.split("_"):
        if token.startswith("ep") and token[2:].isdigit():
            return int(token[2:])
    return -1


def load_trace_rows(trace_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep = _episode_from_path(path)
        for row in _read_jsonl(path):
            item = dict(row)
            item.setdefault("episode_idx", ep)
            item.setdefault("source_trace_path", str(path))
            rows.append(item)
    rows.sort(key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step", r.get("step_idx", -1)))))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _nested(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, Mapping):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, Mapping) else {}


def classify_prior_only_source(row: Mapping[str, Any]) -> str:
    """Return the dominant runtime reason for a prior-only abstain row."""

    if str(row.get("grasp_probe_reason", "")) != "prior_only_abstain":
        return "not_prior_only_abstain"
    grasp = _nested(row, "local_geometry_error", "grasp")
    if not grasp:
        return "missing_local_geometry_trace"
    if not bool(grasp.get("valid", False)):
        return f"localizer_invalid:{str(grasp.get('reason', 'unknown'))}"
    if _safe_float(grasp.get("confidence"), 0.0) <= 0.0 or _safe_float(grasp.get("observability"), 0.0) <= 0.0:
        return "weak_visual_evidence"
    est = _nested(row, "estimated_basin_error")
    if not est:
        return "missing_estimated_basin_error_trace"
    axis_validity = row.get("basin_axis_validity", {})
    if isinstance(axis_validity, Mapping) and not any(bool(v) for v in axis_validity.values()):
        policy = row.get("basin_axis_policy", {})
        if isinstance(policy, Mapping):
            return "estimator_axis_policy_abstain:" + ",".join(f"{k}={v}" for k, v in sorted(policy.items()))
        return "estimator_axis_policy_abstain"
    if not bool(est.get("estimated_basin_error_valid", est.get("valid", False))):
        return f"estimator_invalid:{str(est.get('estimated_basin_error_reason', est.get('reason', 'unknown')))}"
    residual = estimate_runtime_xy_residual_from_trace(row)
    if not residual.entry_ready:
        return f"runtime_xy_entry_not_ready:{residual.reason}"
    return "visibility_field_mismatch"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probe_rows = [r for r in rows if str(r.get("grasp_probe_policy", "")) == "replay_oracle_xy"]
    prior_rows = [r for r in probe_rows if str(r.get("grasp_probe_reason", "")) == "prior_only_abstain"]
    source_counts = Counter(classify_prior_only_source(r) for r in prior_rows)
    by_episode: dict[str, Counter[str]] = defaultdict(Counter)
    confidence: dict[str, list[float]] = defaultdict(list)
    observability: dict[str, list[float]] = defaultdict(list)
    for row in prior_rows:
        source = classify_prior_only_source(row)
        ep = str(row.get("episode_idx", -1))
        by_episode[ep][source] += 1
        grasp = _nested(row, "local_geometry_error", "grasp")
        confidence[source].append(_safe_float(grasp.get("confidence"), float("nan")))
        observability[source].append(_safe_float(grasp.get("observability"), float("nan")))

    def _mean(vals: list[float]) -> float:
        arr = np.asarray(vals, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else 0.0

    source_summary = []
    for source, count in source_counts.most_common():
        source_summary.append(
            {
                "source": source,
                "rows": int(count),
                "rate_among_prior_only": float(count / max(len(prior_rows), 1)),
                "mean_localizer_confidence": _mean(confidence[source]),
                "mean_localizer_observability": _mean(observability[source]),
            }
        )
    return {
        "schema_version": "c2c_v2_prior_only_abstain_diagnostic_v1",
        "total_rows": int(len(rows)),
        "probe_rows": int(len(probe_rows)),
        "prior_only_abstain_rows": int(len(prior_rows)),
        "prior_only_abstain_rate": float(len(prior_rows) / max(len(probe_rows), 1)),
        "source_summary": source_summary,
        "by_episode": [
            {"episode_idx": int(ep), "sources": dict(counter)}
            for ep, counter in sorted(by_episode.items(), key=lambda kv: int(kv[0]))
        ],
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_classification": False,
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "prior_only_abstain_diagnostic.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    lines = [
        "# Prior-only Abstain Diagnostic",
        "",
        f"- probe_rows: `{report['probe_rows']}`",
        f"- prior_only_abstain_rows: `{report['prior_only_abstain_rows']}`",
        f"- prior_only_abstain_rate: `{report['prior_only_abstain_rate']:.3f}`",
        f"- uses_privileged_runtime: `{report['uses_privileged_runtime']}`",
        f"- uses_privileged_label_for_classification: `{report['uses_privileged_label_for_classification']}`",
        "",
        "## Sources",
    ]
    for item in report["source_summary"]:
        lines.append(
            f"- `{item['source']}` rows=`{item['rows']}` rate=`{item['rate_among_prior_only']:.3f}` "
            f"mean_conf=`{item['mean_localizer_confidence']:.3f}` mean_obs=`{item['mean_localizer_observability']:.6f}`"
        )
    lines.append("")
    lines.append("## Episodes")
    for item in report["by_episode"]:
        lines.append(f"- ep`{item['episode_idx']:03d}`: `{item['sources']}`")
    (output_dir / "prior_only_abstain_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    report = summarize(load_trace_rows(args.trace_dir))
    write_report(report, args.output_dir)


if __name__ == "__main__":
    main()
