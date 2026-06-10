#!/usr/bin/env python3
"""Audit yaw-observability shifts between command-sweep seed rows and execution.

The z08 yaw-visible pool can be selected from offline/source labels, but a real
runtime replay may still produce v46 yaw heads that mark the same window as
ambiguous or unobservable.  This audit makes that shift explicit so command
sweep data is not mistaken for yaw-control promotion evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=16)
def _read_jsonl_cached(path_str: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_str)
    if not path.exists():
        return tuple()
    return tuple(_read_jsonl(path))


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(float(value) > 0.5)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if out == out else float(default)
    except Exception:
        return float(default)


def _spec_row(row: Mapping[str, Any], explicit_spec: Path | None) -> dict[str, Any]:
    try:
        idx = int(row.get("task_frame_v46_command_sweep_row_index", -1))
    except Exception:
        idx = -1
    spec_path = explicit_spec
    if spec_path is None:
        value = str(row.get("task_frame_v46_command_sweep_spec_path", "") or "").strip()
        if value:
            spec_path = Path(value)
    if spec_path is None or idx < 0:
        return {}
    rows = _read_jsonl_cached(str(spec_path))
    if idx >= len(rows):
        return {}
    return dict(rows[idx])


def _seed_yaw_info(spec: Mapping[str, Any]) -> dict[str, Any]:
    runtime_fields = spec.get("runtime_trace_fields", {})
    runtime_fields = runtime_fields if isinstance(runtime_fields, Mapping) else {}
    yaw_class = str(
        runtime_fields.get(
            "task_frame_v46_yaw_observability_class",
            runtime_fields.get("yaw_observability_class", spec.get("yaw_observability_class", "")),
        )
        or ""
    )
    yaw_observable = _as_bool(
        runtime_fields.get("task_frame_v46_label_yaw_observable", runtime_fields.get("yaw_observable", False)),
        yaw_class == "observable",
    )
    yaw_ambiguous = _as_bool(runtime_fields.get("task_frame_v46_label_yaw_ambiguous", False), yaw_class in {"ambiguous", "unobservable"})
    return {
        "seed_yaw_observable": bool(yaw_observable),
        "seed_yaw_ambiguous": bool(yaw_ambiguous),
        "seed_yaw_observability_class": yaw_class or "unknown",
        "seed_yaw_control": bool(yaw_observable and not yaw_ambiguous),
        "seed_yaw_frame_observability": _safe_float(runtime_fields.get("yaw_observability_frame_observability"), 0.0),
        "seed_yaw_frame_confidence": _safe_float(runtime_fields.get("yaw_observability_frame_confidence"), 0.0),
        "seed_yaw_control_block_reason": str(runtime_fields.get("yaw_control_block_reason", "")),
    }


def _exec_yaw_info(row: Mapping[str, Any]) -> dict[str, Any]:
    labels = row.get("offline_labels", {})
    labels = labels if isinstance(labels, Mapping) else {}
    yaw_observable = _as_bool(labels.get("yaw_observable", row.get("task_frame_v46_yaw_observable", False)), False)
    yaw_ambiguous = _as_bool(labels.get("yaw_ambiguous", row.get("task_frame_v46_yaw_ambiguous", True)), True)
    return {
        "exec_yaw_observable": bool(yaw_observable),
        "exec_yaw_ambiguous": bool(yaw_ambiguous),
        "exec_yaw_control": bool(yaw_observable and not yaw_ambiguous),
        "exec_yaw_observability_class": str(row.get("yaw_observability_class", "unknown") or "unknown"),
        "exec_yaw_contracted": bool(row.get("yaw_contracted_observed", False)),
        "exec_z_contracted": bool(row.get("z_contracted_observed", False)),
        "exec_xy_contracted": bool(row.get("xy_contracted_observed", False)),
        "exec_close_leak": bool(row.get("close_leak", False)),
        "pre_abs_yaw": abs(_safe_float(labels.get("dyaw"), 0.0)),
        "post_abs_yaw": abs(_safe_float(row.get("next_privileged_dyaw"), 0.0)),
        "pre_abs_z": abs(_safe_float(labels.get("dz"), 0.0)),
        "post_abs_z": abs(_safe_float(row.get("next_privileged_dz"), 0.0)),
    }


def audit_shift(
    manifest_jsonl: Path,
    *,
    output_json: Path,
    output_jsonl: Path | None = None,
    spec_jsonl: Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(manifest_jsonl)
    counters: Counter[str] = Counter()
    by_candidate: dict[str, Counter[str]] = defaultdict(Counter)
    by_episode: dict[str, Counter[str]] = defaultdict(Counter)
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        counters["input_rows"] += 1
        spec = _spec_row(row, spec_jsonl)
        if not spec:
            counters["missing_spec_row"] += 1
        seed = _seed_yaw_info(spec)
        executed = _exec_yaw_info(row)
        shifted = bool(seed["seed_yaw_control"] and not executed["exec_yaw_control"])
        if shifted:
            counters["seed_control_to_exec_noncontrol"] += 1
        if seed["seed_yaw_control"]:
            counters["seed_yaw_control_rows"] += 1
        if executed["exec_yaw_control"]:
            counters["exec_yaw_control_rows"] += 1
        if executed["exec_close_leak"]:
            counters["close_leak_rows"] += 1
        if executed["exec_yaw_contracted"]:
            counters["yaw_contracted_rows"] += 1
        if executed["exec_z_contracted"]:
            counters["z_contracted_rows"] += 1
        if executed["exec_xy_contracted"]:
            counters["xy_contracted_rows"] += 1
        candidate = str(row.get("task_frame_v46_command_sweep_candidate_name", spec.get("candidate_name", "unknown")) or "unknown")
        episode = f"ep{int(row.get('episode_idx', -1)):03d}"
        for group in (by_candidate[candidate], by_episode[episode]):
            group["rows"] += 1
            if shifted:
                group["seed_control_to_exec_noncontrol"] += 1
            if executed["exec_yaw_contracted"]:
                group["yaw_contracted_rows"] += 1
            if executed["exec_z_contracted"]:
                group["z_contracted_rows"] += 1
        enriched = {
            "episode_idx": int(row.get("episode_idx", -1)),
            "step_idx": int(row.get("step_idx", -1)),
            "candidate_name": candidate,
            "source_eval_root": str(row.get("source_eval_root", "")),
            "command": row.get("applied_control_command_local_6d", []),
            **seed,
            **executed,
            "yaw_observability_shift": shifted,
        }
        enriched_rows.append(enriched)

    for key in (
        "missing_spec_row",
        "seed_yaw_control_rows",
        "exec_yaw_control_rows",
        "seed_control_to_exec_noncontrol",
        "close_leak_rows",
        "yaw_contracted_rows",
        "z_contracted_rows",
        "xy_contracted_rows",
    ):
        counters.setdefault(key, 0)
    total = max(1, counters["input_rows"])
    summary = {
        "schema_version": "c2c_v2_task_frame_yaw_observability_shift_audit_v1",
        "manifest_jsonl": str(manifest_jsonl),
        "spec_jsonl": str(spec_jsonl) if spec_jsonl is not None else "",
        "rows": int(counters["input_rows"]),
        "counters": dict(counters),
        "rates": {
            "seed_control_to_exec_noncontrol": float(counters["seed_control_to_exec_noncontrol"] / total),
            "exec_yaw_control": float(counters["exec_yaw_control_rows"] / total),
            "yaw_contraction": float(counters["yaw_contracted_rows"] / total),
            "z_contraction": float(counters["z_contracted_rows"] / total),
            "xy_contraction": float(counters["xy_contracted_rows"] / total),
        },
        "by_candidate": {key: dict(value) for key, value in sorted(by_candidate.items())},
        "by_episode": {key: dict(value) for key, value in sorted(by_episode.items())},
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_audit": True,
        "privileged_label_boundary": "offline_seed_vs_executed_observability_audit_only",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for row in enriched_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary["output_jsonl"] = str(output_jsonl)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit yaw observability shift from seed/spec to executed manifest.")
    parser.add_argument("--manifest_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--spec_jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_shift(
        args.manifest_jsonl,
        output_json=args.output_json,
        output_jsonl=args.output_jsonl,
        spec_jsonl=args.spec_jsonl,
    )
    print(json.dumps({k: summary[k] for k in ("rows", "counters", "rates")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
