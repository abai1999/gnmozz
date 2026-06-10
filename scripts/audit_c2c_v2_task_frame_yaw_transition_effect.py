#!/usr/bin/env python3
"""Audit executed yaw command effects for v46 task-frame control.

This is an offline-only audit over applied-transition manifests. It keeps true
pre/post residuals as labels and never creates runtime inputs from privileged
state. The question it answers is narrower than promotion: when a yaw command
was actually executed, did it reduce jaw-local yaw residual, and what XY/Z
collateral did it create relative to the zero command from the same window?
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _vec(value: Any, length: int) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < int(length) or not np.all(np.isfinite(arr[: int(length)])):
        return None
    return arr[: int(length)].astype(np.float32)


def _pre_post(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    labels = row.get("offline_labels", {})
    labels = labels if isinstance(labels, Mapping) else {}
    try:
        pre = np.asarray([float(labels[k]) for k in ("dx", "dy", "dz", "dyaw")], dtype=np.float32)
        post = np.asarray(
            [
                float(row["next_privileged_dx"]),
                float(row["next_privileged_dy"]),
                float(row["next_privileged_dz"]),
                float(row["next_privileged_dyaw"]),
            ],
            dtype=np.float32,
        )
    except Exception:
        return None
    if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post)):
        return None
    return pre, post


def _candidate_name(row: Mapping[str, Any]) -> str:
    return str(row.get("task_frame_v46_command_sweep_candidate_name", row.get("candidate_name", "unknown")) or "unknown")


def _row_index(row: Mapping[str, Any]) -> int | None:
    try:
        value = int(row.get("task_frame_v46_command_sweep_row_index"))
        return value if value >= 0 else None
    except Exception:
        return None


def _group_key(row: Mapping[str, Any], *, candidates_per_group: int) -> str:
    idx = _row_index(row)
    if idx is not None and candidates_per_group > 0:
        return f"row_group:{idx // int(candidates_per_group):06d}"
    return (
        f"episode_step:{row.get('sequence_id', '')}:"
        f"ep{int(row.get('episode_idx', -1)):03d}:step{int(row.get('step_idx', -1)):04d}"
    )


def _command(row: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("applied_control_command_local_6d", "candidate_command_local_6d", "command_6d"):
        command = _vec(row.get(key), 6)
        if command is not None:
            return command
    return None


def _delta(pre: np.ndarray, post: np.ndarray) -> dict[str, float]:
    return {
        "xy": float(np.linalg.norm(post[:2]) - np.linalg.norm(pre[:2])),
        "z": float(abs(float(post[2])) - abs(float(pre[2]))),
        "yaw": float(abs(float(post[3])) - abs(float(pre[3]))),
        "combined": float(
            (np.linalg.norm(post[:2]) + abs(float(post[2])) + abs(float(post[3])))
            - (np.linalg.norm(pre[:2]) + abs(float(pre[2])) + abs(float(pre[3])))
        ),
    }


def _rate(records: list[dict[str, Any]], field: str) -> float:
    return float(sum(bool(row.get(field, False)) for row in records) / len(records)) if records else 0.0


def _mean(records: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in records if np.isfinite(float(row.get(field, float("nan"))))]
    return float(np.mean(values)) if values else 0.0


def _yaw_bucket(command_yaw: float) -> str:
    mag = abs(float(command_yaw))
    if mag <= 1.0e-9:
        return "zero"
    return f"{'pos' if command_yaw > 0.0 else 'neg'}_{int(round(mag * 10000)):04d}"


def audit(
    input_jsonl: list[Path],
    *,
    output_json: Path,
    candidates_per_group: int = 0,
    improvement_margin: float = 1.0e-7,
    collateral_margin: float = 1.0e-7,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in input_jsonl:
        rows.extend(_read_jsonl(path))
    if candidates_per_group <= 0:
        names = {_candidate_name(row) for row in rows}
        candidates_per_group = max(1, len(names))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row, candidates_per_group=candidates_per_group)].append(row)

    counters: Counter[str] = Counter(input_rows=len(rows), candidate_groups=len(grouped))
    records: list[dict[str, Any]] = []
    group_best: dict[str, Any] = {}
    for key, group in sorted(grouped.items()):
        zero_rows = [row for row in group if _candidate_name(row) == "zero"]
        zero_delta = None
        if len(zero_rows) == 1:
            pair = _pre_post(zero_rows[0])
            if pair is not None:
                zero_delta = _delta(*pair)
        if zero_delta is None:
            counters["groups_without_valid_zero"] += 1
        yaw_records: list[dict[str, Any]] = []
        for row in group:
            name = _candidate_name(row)
            command = _command(row)
            pair = _pre_post(row)
            if command is None or pair is None:
                counters["rows_without_command_or_prepost"] += 1
                continue
            command_yaw = float(command[5])
            if abs(command_yaw) <= 1.0e-9 and "yaw" not in name:
                continue
            pre, post = pair
            d = _delta(pre, post)
            adjusted = {axis: float(d[axis] - zero_delta[axis]) for axis in ("xy", "z", "yaw", "combined")} if zero_delta else {}
            pre_yaw = float(pre[3])
            post_yaw = float(post[3])
            rec = {
                "group_key": key,
                "candidate": name,
                "episode_idx": int(row.get("episode_idx", -1)),
                "step_idx": int(row.get("step_idx", -1)),
                "source_eval_root": str(row.get("source_eval_root", "")),
                "pre_yaw": pre_yaw,
                "post_yaw": post_yaw,
                "abs_pre_yaw": abs(pre_yaw),
                "abs_post_yaw": abs(post_yaw),
                "command_yaw": command_yaw,
                "command_yaw_bucket": _yaw_bucket(command_yaw),
                "command_same_sign_as_residual": bool(np.sign(command_yaw) == np.sign(pre_yaw) and abs(command_yaw) > 1.0e-9 and abs(pre_yaw) > 1.0e-9),
                "command_opposes_residual": bool(np.sign(command_yaw) == -np.sign(pre_yaw) and abs(command_yaw) > 1.0e-9 and abs(pre_yaw) > 1.0e-9),
                "yaw_delta": d["yaw"],
                "xy_delta": d["xy"],
                "z_delta": d["z"],
                "combined_delta": d["combined"],
                "zero_adjusted_yaw_delta": float(adjusted.get("yaw", float("nan"))),
                "zero_adjusted_xy_delta": float(adjusted.get("xy", float("nan"))),
                "zero_adjusted_z_delta": float(adjusted.get("z", float("nan"))),
                "zero_adjusted_combined_delta": float(adjusted.get("combined", float("nan"))),
                "yaw_contracts": bool(d["yaw"] < -float(improvement_margin)),
                "yaw_worsens": bool(d["yaw"] > float(improvement_margin)),
                "beats_zero_yaw": bool(adjusted and adjusted["yaw"] < -float(improvement_margin)),
                "worse_than_zero_yaw": bool(adjusted and adjusted["yaw"] > float(improvement_margin)),
                "xy_collateral_worsen": bool(d["xy"] > float(collateral_margin)),
                "z_collateral_worsen": bool(d["z"] > float(collateral_margin)),
                "combined_contracts": bool(d["combined"] < -float(improvement_margin)),
                "yaw_observable_label": bool((row.get("offline_labels") or {}).get("yaw_observable", False)),
                "yaw_ambiguous_label": bool((row.get("offline_labels") or {}).get("yaw_ambiguous", True)),
                "close_leak": bool(row.get("close_leak", False)),
                "uses_privileged_runtime": bool(row.get("uses_privileged_runtime", False)),
            }
            records.append(rec)
            yaw_records.append(rec)
        if yaw_records:
            group_best[key] = min(yaw_records, key=lambda rec: (float(rec["zero_adjusted_yaw_delta"]) if np.isfinite(float(rec["zero_adjusted_yaw_delta"])) else float(rec["yaw_delta"])))
            counters["audited_groups"] += 1

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sign_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_bucket[str(rec["command_yaw_bucket"])].append(rec)
        relation = "same_sign" if rec["command_same_sign_as_residual"] else "opposes_residual" if rec["command_opposes_residual"] else "zero_or_no_residual"
        by_sign_relation[relation].append(rec)

    def summarize(records_in: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(records_in),
            "yaw_contract_rate": _rate(records_in, "yaw_contracts"),
            "yaw_worsen_rate": _rate(records_in, "yaw_worsens"),
            "beats_zero_yaw_rate": _rate(records_in, "beats_zero_yaw"),
            "worse_than_zero_yaw_rate": _rate(records_in, "worse_than_zero_yaw"),
            "xy_collateral_worsen_rate": _rate(records_in, "xy_collateral_worsen"),
            "z_collateral_worsen_rate": _rate(records_in, "z_collateral_worsen"),
            "combined_contract_rate": _rate(records_in, "combined_contracts"),
            "mean_yaw_delta": _mean(records_in, "yaw_delta"),
            "mean_zero_adjusted_yaw_delta": _mean(records_in, "zero_adjusted_yaw_delta"),
            "mean_xy_delta": _mean(records_in, "xy_delta"),
            "mean_z_delta": _mean(records_in, "z_delta"),
        }

    best_records = list(group_best.values())
    summary = {
        "schema_version": "c2c_v2_task_frame_yaw_transition_effect_audit_v1",
        "input_jsonl": [str(path) for path in input_jsonl],
        "output_json": str(output_json),
        "candidates_per_group": int(candidates_per_group),
        "improvement_margin": float(improvement_margin),
        "collateral_margin": float(collateral_margin),
        "counters": dict(counters),
        "rows": len(records),
        "close_leak_rows": int(sum(bool(row.get("close_leak", False)) for row in records)),
        "uses_privileged_runtime_any": bool(any(bool(row.get("uses_privileged_runtime", False)) for row in records)),
        "overall": summarize(records),
        "best_per_group": summarize(best_records),
        "by_command_yaw_bucket": {key: summarize(value) for key, value in sorted(by_bucket.items())},
        "by_command_residual_sign_relation": {key: summarize(value) for key, value in sorted(by_sign_relation.items())},
        "best_group_examples": list(best_records[:20]),
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_eval": True,
        "privileged_label_boundary": "offline_pre_post_transition_labels_only",
        "upgrade_gate": "diagnostic_only_collect_more_yaw_positive_transition_data",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit executed yaw command effects for v46 task-frame control.")
    parser.add_argument("--input_jsonl", nargs="+", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--candidates_per_group", type=int, default=0)
    parser.add_argument("--improvement_margin", type=float, default=1.0e-7)
    parser.add_argument("--collateral_margin", type=float, default=1.0e-7)
    args = parser.parse_args()
    summary = audit(
        list(args.input_jsonl),
        output_json=args.output_json,
        candidates_per_group=int(args.candidates_per_group),
        improvement_margin=float(args.improvement_margin),
        collateral_margin=float(args.collateral_margin),
    )
    print(json.dumps({"rows": summary["rows"], "audited_groups": summary["counters"].get("audited_groups", 0), "overall": summary["overall"], "best_per_group": summary["best_per_group"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
