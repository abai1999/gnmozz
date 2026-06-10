#!/usr/bin/env python3
"""Audit command-sweep effects after subtracting same-window zero drift.

The applied-transition manifest records true pre/post residuals only as offline
labels. This audit keeps that boundary intact and answers a narrower question:
did a candidate command improve the residual more than the zero/no-op candidate
from the same runtime window?
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


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


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
        return f"row_group:{idx // int(candidates_per_group):05d}"
    return f"episode_step:{row.get('sequence_id', '')}:ep{int(row.get('episode_idx', -1)):03d}:step{int(row.get('step_idx', -1)):04d}"


def _pre_post(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    pre_labels = row.get("offline_labels") or {}
    keys = ("dx", "dy", "dz", "dyaw")
    try:
        pre = np.asarray([float(pre_labels[k]) for k in keys], dtype=np.float32)
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


def _axis_delta(row: Mapping[str, Any]) -> dict[str, float] | None:
    pair = _pre_post(row)
    if pair is None:
        return None
    pre, post = pair
    return {
        "xy": float(np.linalg.norm(post[:2]) - np.linalg.norm(pre[:2])),
        "z": float(abs(float(post[2])) - abs(float(pre[2]))),
        "yaw": float(abs(float(post[3])) - abs(float(pre[3]))),
        "combined": float(
            (np.linalg.norm(post[:2]) + abs(float(post[2])) + abs(float(post[3])))
            - (np.linalg.norm(pre[:2]) + abs(float(pre[2])) + abs(float(pre[3])))
        ),
    }


def audit(
    input_jsonl: Path,
    *,
    output_json: Path,
    candidates_per_group: int = 0,
    improvement_margin: float = 1.0e-7,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    if candidates_per_group <= 0:
        names = {_candidate_name(row) for row in rows}
        candidates_per_group = max(1, len(names))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row, candidates_per_group=candidates_per_group)].append(row)

    candidate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_details: dict[str, Any] = {}
    counters: Counter[str] = Counter()
    for key, group in sorted(grouped.items()):
        zero_rows = [row for row in group if _candidate_name(row) == "zero"]
        if len(zero_rows) != 1:
            counters["groups_without_single_zero"] += 1
            continue
        zero_delta = _axis_delta(zero_rows[0])
        if zero_delta is None:
            counters["groups_without_valid_zero_delta"] += 1
            continue
        counters["audited_groups"] += 1
        details: list[dict[str, Any]] = []
        for row in group:
            delta = _axis_delta(row)
            if delta is None:
                counters["rows_without_valid_delta"] += 1
                continue
            name = _candidate_name(row)
            adjusted = {axis: float(delta[axis] - zero_delta[axis]) for axis in ("xy", "z", "yaw", "combined")}
            record = {
                "candidate": name,
                "row_index": _row_index(row),
                "episode_idx": int(row.get("episode_idx", -1)),
                "step_idx": int(row.get("step_idx", -1)),
                "raw_delta": delta,
                "zero_delta": zero_delta,
                "zero_adjusted_delta": adjusted,
                "beats_zero_xy": bool(adjusted["xy"] < -float(improvement_margin)),
                "beats_zero_z": bool(adjusted["z"] < -float(improvement_margin)),
                "beats_zero_yaw": bool(adjusted["yaw"] < -float(improvement_margin)),
                "beats_zero_combined": bool(adjusted["combined"] < -float(improvement_margin)),
                "close_leak": bool(row.get("close_leak", False)),
                "uses_privileged_runtime": bool(row.get("uses_privileged_runtime", False)),
            }
            details.append(record)
            candidate_rows[name].append(record)
        group_details[key] = {
            "episode_idx": int(group[0].get("episode_idx", -1)),
            "step_idx": int(group[0].get("step_idx", -1)),
            "rows": details,
        }

    def _rate(records: list[dict[str, Any]], field: str) -> float:
        return float(sum(bool(row[field]) for row in records) / len(records)) if records else 0.0

    candidate_metrics = {
        name: {
            "rows": len(records),
            "beats_zero_xy": _rate(records, "beats_zero_xy"),
            "beats_zero_z": _rate(records, "beats_zero_z"),
            "beats_zero_yaw": _rate(records, "beats_zero_yaw"),
            "beats_zero_combined": _rate(records, "beats_zero_combined"),
            "mean_zero_adjusted_xy": float(np.mean([row["zero_adjusted_delta"]["xy"] for row in records])) if records else 0.0,
            "mean_zero_adjusted_z": float(np.mean([row["zero_adjusted_delta"]["z"] for row in records])) if records else 0.0,
            "mean_zero_adjusted_yaw": float(np.mean([row["zero_adjusted_delta"]["yaw"] for row in records])) if records else 0.0,
            "mean_zero_adjusted_combined": float(np.mean([row["zero_adjusted_delta"]["combined"] for row in records])) if records else 0.0,
        }
        for name, records in sorted(candidate_rows.items())
    }
    all_records = [record for records in candidate_rows.values() for record in records]
    summary = {
        "schema_version": "c2c_v2_task_frame_zero_adjusted_effect_audit_v1",
        "input_jsonl": str(input_jsonl),
        "output_json": str(output_json),
        "rows": len(rows),
        "candidate_groups": len(grouped),
        "audited_groups": int(counters["audited_groups"]),
        "counters": dict(counters),
        "candidates_per_group": int(candidates_per_group),
        "close_leak_rows": int(sum(bool(row.get("close_leak", False)) for row in rows)),
        "uses_privileged_runtime_any": bool(any(bool(row.get("uses_privileged_runtime", False)) for row in rows)),
        "overall": {
            "beats_zero_xy": _rate(all_records, "beats_zero_xy"),
            "beats_zero_z": _rate(all_records, "beats_zero_z"),
            "beats_zero_yaw": _rate(all_records, "beats_zero_yaw"),
            "beats_zero_combined": _rate(all_records, "beats_zero_combined"),
        },
        "candidate_metrics": candidate_metrics,
        "group_details": group_details,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--candidates_per_group", type=int, default=0)
    parser.add_argument("--improvement_margin", type=float, default=1.0e-7)
    args = parser.parse_args()
    summary = audit(
        args.input_jsonl,
        output_json=args.output_json,
        candidates_per_group=int(args.candidates_per_group),
        improvement_margin=float(args.improvement_margin),
    )
    print(json.dumps({k: summary[k] for k in ("rows", "audited_groups", "close_leak_rows", "overall")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
