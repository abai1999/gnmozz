#!/usr/bin/env python3
"""Mine yaw-positive near-basin windows from C2C v2 frame residual relabels.

The target window for proving a yaw/frame estimator is:

    yaw_control_observable && near_basin_shell && visual_observable

This script scans one or more `frame_residual_v2.jsonl` files, writes matching
rows to a stable JSONL, and reports adjacent failure modes so the next
collection pass can distinguish "no near-basin window" from "yaw gate blocks
an otherwise useful near-basin window".
"""

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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                row.setdefault("source_relabel_jsonl", str(path))
                rows.append(row)
    return rows


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value) > 0.5)
    return bool(value)


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    return value if isinstance(value, Mapping) else {}


def _visual_class(row: Mapping[str, Any]) -> str:
    obs = _mapping(row, "obs_t")
    return str(obs.get("visual_observability_class", row.get("visual_observability_class", "")))


def _yaw_control_observable(row: Mapping[str, Any]) -> bool:
    return _as_bool(row.get("yaw_control_observable", row.get("yaw_observable", False)))


def _yaw_entry_feasible(row: Mapping[str, Any]) -> bool:
    if "yaw_entry_feasible" in row:
        return _as_bool(row.get("yaw_entry_feasible", False))
    yaw_abs = _safe_float(row.get("yaw_abs", abs(_safe_float(_mapping(row, "true_basin_error_t").get("dyaw"), float("nan")))), float("nan"))
    return bool(np.isfinite(yaw_abs) and yaw_abs <= 0.08 + 1.0e-9)


def _frame_observability(row: Mapping[str, Any]) -> float:
    obs = _mapping(row, "obs_t")
    return _safe_float(
        row.get("yaw_observability_frame_observability", row.get("source_frame_observability", obs.get("frame_observability", 0.0))),
        0.0,
    )


def _wrist_occluded(row: Mapping[str, Any]) -> bool:
    return _as_bool(row.get("yaw_observability_wrist_occluded", row.get("wrist_is_occluded", False)))


def _near_basin(row: Mapping[str, Any]) -> bool:
    return _as_bool(row.get("near_basin_shell", False))


def _visual_observable(row: Mapping[str, Any]) -> bool:
    return _visual_class(row) == "visual_observable"


def _label_valid(row: Mapping[str, Any]) -> bool:
    return _as_bool(row.get("label_valid", True), True)


def _precision_grasp(row: Mapping[str, Any], stage_name: str, skill_type: str) -> bool:
    if stage_name and str(row.get("stage_name", "")) != stage_name:
        return False
    if skill_type and str(row.get("skill_type", "")) != skill_type:
        return False
    return True


def _target_window(row: Mapping[str, Any]) -> bool:
    return bool(_label_valid(row) and _visual_observable(row) and _near_basin(row) and _yaw_control_observable(row))


def _relaxed_yaw_diagnostic_candidate(
    row: Mapping[str, Any],
    *,
    min_frame_observability: float,
    include_wrist_occluded: bool,
) -> bool:
    return bool(
        _label_valid(row)
        and _visual_observable(row)
        and _near_basin(row)
        and _yaw_entry_feasible(row)
        and not _yaw_control_observable(row)
        and _frame_observability(row) >= float(min_frame_observability) - 1.0e-12
        and (include_wrist_occluded or not _wrist_occluded(row))
    )


def _row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    residual = _mapping(row, "true_basin_error_t")
    return {
        "schema_version": "yaw_positive_window_v1",
        "task_name": str(row.get("task_name", "")),
        "episode_idx": int(row.get("episode_idx", -1)),
        "step_idx": int(row.get("step_idx", row.get("step", -1))),
        "stage_name": str(row.get("stage_name", "")),
        "skill_name": str(row.get("skill_name", "")),
        "skill_type": str(row.get("skill_type", "")),
        "failure_bucket": str(row.get("failure_bucket", "")),
        "takeover_tier": str(row.get("takeover_tier", "")),
        "visual_observability_class": _visual_class(row),
        "yaw_observability_class": str(row.get("yaw_observability_class", "")),
        "yaw_control_observable": _yaw_control_observable(row),
        "yaw_entry_feasible": _yaw_entry_feasible(row),
        "near_basin_shell": _near_basin(row),
        "micro_entry_ready": _as_bool(row.get("micro_entry_ready", False)),
        "close_ready_ready": _as_bool(row.get("close_ready_ready", False)),
        "xy_error": _safe_float(row.get("xy_error"), float("nan")),
        "yaw_abs": _safe_float(row.get("yaw_abs", abs(_safe_float(residual.get("dyaw"), float("nan")))), float("nan")),
        "privileged_dyaw": _safe_float(row.get("privileged_dyaw", residual.get("dyaw", float("nan"))), float("nan")),
        "yaw_observability_primary_blocker": str(row.get("yaw_observability_primary_blocker", "")),
        "yaw_observability_blocker_combo": str(row.get("yaw_observability_blocker_combo", "")),
        "yaw_observability_frame_observability": _frame_observability(row),
        "yaw_observability_frame_confidence": _safe_float(row.get("yaw_observability_frame_confidence", row.get("source_frame_confidence", float("nan"))), float("nan")),
        "yaw_observability_frame_axis_strength": _safe_float(row.get("yaw_observability_frame_axis_strength", row.get("source_frame_axis_strength", float("nan"))), float("nan")),
        "yaw_observability_wrist_occluded": _wrist_occluded(row),
        "source_relabel_jsonl": str(row.get("source_relabel_jsonl", "")),
        "source_runtime_obs_path": str(row.get("source_runtime_obs_path", "")),
        "source_trace_path": str(row.get("source_trace_path", "")),
    }


def _counter(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def _episode_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("episode_idx", -1))].append(row)
    out: list[dict[str, Any]] = []
    for ep, subset in grouped.items():
        out.append(
            {
                "episode_idx": int(ep),
                "rows": int(len(subset)),
                "target_rows": int(sum(_target_window(r) for r in subset)),
                "near_visual_yaw_blocked_rows": int(sum(_visual_observable(r) and _near_basin(r) and not _yaw_control_observable(r) for r in subset)),
                "yaw_observable_not_near_rows": int(sum(_visual_observable(r) and _yaw_control_observable(r) and not _near_basin(r) for r in subset)),
                "entry_visual_not_near_rows": int(sum(_visual_observable(r) and _yaw_entry_feasible(r) and not _near_basin(r) for r in subset)),
                "failure_bucket_counts": _counter(subset, "failure_bucket"),
                "source_count": int(len({str(r.get("source_relabel_jsonl", "")) for r in subset})),
            }
        )
    return sorted(out, key=lambda item: (-item["target_rows"], -item["near_visual_yaw_blocked_rows"], -item["yaw_observable_not_near_rows"], item["episode_idx"]))


def mine(
    rows: list[dict[str, Any]],
    *,
    stage_name: str = "RING_GRASP_ALIGN",
    skill_type: str = "precision_grasp",
    min_target_rows: int = 30,
    min_target_episodes: int = 3,
    relaxed_min_frame_observability: float = 0.02,
    relaxed_include_wrist_occluded: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered = [r for r in rows if _precision_grasp(r, stage_name, skill_type) and _label_valid(r)]
    target = [_row_summary(r) for r in filtered if _target_window(r)]
    relaxed_candidates = [
        _row_summary(r)
        for r in filtered
        if _relaxed_yaw_diagnostic_candidate(
            r,
            min_frame_observability=float(relaxed_min_frame_observability),
            include_wrist_occluded=bool(relaxed_include_wrist_occluded),
        )
    ]
    near_visual = [r for r in filtered if _visual_observable(r) and _near_basin(r)]
    near_visual_blocked = [r for r in near_visual if not _yaw_control_observable(r)]
    yaw_observable_visual = [r for r in filtered if _visual_observable(r) and _yaw_control_observable(r)]
    yaw_observable_visual_not_near = [r for r in yaw_observable_visual if not _near_basin(r)]
    entry_visual = [r for r in filtered if _visual_observable(r) and _yaw_entry_feasible(r)]
    entry_visual_not_near = [r for r in entry_visual if not _near_basin(r)]

    target_episode_count = len({int(r.get("episode_idx", -1)) for r in target})
    if len(target) >= int(min_target_rows) and target_episode_count >= int(min_target_episodes):
        recommendation = "use_target_windows_for_frame_yaw_eval"
    elif target:
        recommendation = "target_windows_exist_but_insufficient_collect_focus_windows"
    elif near_visual_blocked:
        recommendation = "yaw_gate_blocks_near_basin_visual_rows"
    elif yaw_observable_visual_not_near:
        recommendation = "collect_or_force_nearer_window_for_yaw_observable_rows"
    elif entry_visual_not_near:
        recommendation = "window_protocol_or_planner_handoff_not_reaching_near_basin"
    else:
        recommendation = "collect_more_visual_entry_feasible_precision_rows"

    by_episode = _episode_table(filtered)
    collection_focus = [
        item
        for item in by_episode
        if int(item["target_rows"]) > 0
        or int(item["near_visual_yaw_blocked_rows"]) > 0
        or int(item["yaw_observable_not_near_rows"]) > 0
    ][:20]

    report = {
        "schema_version": "yaw_positive_window_mining_v1",
        "selection": {
            "stage_name": stage_name,
            "skill_type": skill_type,
            "target_predicate": "label_valid && visual_observable && near_basin_shell && yaw_control_observable",
            "min_target_rows": int(min_target_rows),
            "min_target_episodes": int(min_target_episodes),
        },
        "overall": {
            "rows": int(len(filtered)),
            "episodes": int(len({int(r.get("episode_idx", -1)) for r in filtered})),
            "target_rows": int(len(target)),
            "target_episodes": int(len({int(r.get("episode_idx", -1)) for r in target})),
            "near_visual_rows": int(len(near_visual)),
            "near_visual_yaw_blocked_rows": int(len(near_visual_blocked)),
            "yaw_observable_visual_rows": int(len(yaw_observable_visual)),
            "yaw_observable_visual_not_near_rows": int(len(yaw_observable_visual_not_near)),
            "entry_visual_rows": int(len(entry_visual)),
            "entry_visual_not_near_rows": int(len(entry_visual_not_near)),
            "recommendation": recommendation,
        },
        "counts": {
            "target_by_episode": _counter(target, "episode_idx"),
            "target_by_failure_bucket": _counter(target, "failure_bucket"),
            "target_by_source": _counter(target, "source_relabel_jsonl"),
            "relaxed_yaw_diagnostic_by_episode": _counter(relaxed_candidates, "episode_idx"),
            "relaxed_yaw_diagnostic_by_primary_blocker": _counter(relaxed_candidates, "yaw_observability_primary_blocker"),
            "near_visual_blocked_by_primary_blocker": _counter([_row_summary(r) for r in near_visual_blocked], "yaw_observability_primary_blocker"),
            "near_visual_blocked_by_blocker_combo": _counter([_row_summary(r) for r in near_visual_blocked], "yaw_observability_blocker_combo"),
            "yaw_observable_not_near_by_episode": _counter([_row_summary(r) for r in yaw_observable_visual_not_near], "episode_idx"),
        },
        "by_episode": by_episode,
        "collection_focus_episodes": collection_focus,
        "top_target_examples": target[:100],
        "relaxed_yaw_diagnostic_selection": {
            "predicate": "label_valid && visual_observable && near_basin_shell && yaw_entry_feasible && !yaw_control_observable && frame_observability >= min && !wrist_occluded",
            "min_frame_observability": float(relaxed_min_frame_observability),
            "include_wrist_occluded": bool(relaxed_include_wrist_occluded),
            "rows": int(len(relaxed_candidates)),
            "episodes": int(len({int(r.get("episode_idx", -1)) for r in relaxed_candidates})),
        },
        "_relaxed_yaw_diagnostic_candidates": relaxed_candidates,
        "top_relaxed_yaw_diagnostic_examples": relaxed_candidates[:100],
        "top_near_visual_yaw_blocked_examples": [_row_summary(r) for r in near_visual_blocked[:100]],
        "top_yaw_observable_visual_not_near_examples": [_row_summary(r) for r in yaw_observable_visual_not_near[:100]],
    }
    report["overall"]["relaxed_yaw_diagnostic_candidate_rows"] = int(len(relaxed_candidates))
    report["overall"]["relaxed_yaw_diagnostic_candidate_episodes"] = int(
        len({int(r.get("episode_idx", -1)) for r in relaxed_candidates})
    )
    return target, report


def _expand_inputs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.rglob("frame_residual_v2.jsonl")))
        elif path.is_file():
            out.append(path)
    dedup: list[Path] = []
    seen: set[str] = set()
    for path in out:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            dedup.append(path)
    return dedup


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine yaw-positive near-basin frame residual windows.")
    ap.add_argument("--input", type=Path, nargs="+", required=True, help="frame_residual_v2.jsonl files or directories to scan")
    ap.add_argument("--output_dir", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/reports/yaw_positive_window_mining"))
    ap.add_argument("--stage_name", type=str, default="RING_GRASP_ALIGN")
    ap.add_argument("--skill_type", type=str, default="precision_grasp")
    ap.add_argument("--min_target_rows", type=int, default=30)
    ap.add_argument("--min_target_episodes", type=int, default=3)
    ap.add_argument("--relaxed_min_frame_observability", type=float, default=0.02)
    ap.add_argument("--relaxed_include_wrist_occluded", action="store_true", default=False)
    args = ap.parse_args()

    paths = _expand_inputs(list(args.input))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    target, report = mine(
        rows,
        stage_name=str(args.stage_name),
        skill_type=str(args.skill_type),
        min_target_rows=int(args.min_target_rows),
        min_target_episodes=int(args.min_target_episodes),
        relaxed_min_frame_observability=float(args.relaxed_min_frame_observability),
        relaxed_include_wrist_occluded=bool(args.relaxed_include_wrist_occluded),
    )
    report["source_jsonl_files"] = [str(p.resolve()) for p in paths]
    relaxed_candidates = list(report.pop("_relaxed_yaw_diagnostic_candidates", []))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "yaw_positive_windows.jsonl"
    with open(target_path, "w", encoding="utf-8") as handle:
        for row in target:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report["target_windows_jsonl"] = str(target_path)
    relaxed_path = output_dir / "relaxed_yaw_diagnostic_candidates.jsonl"
    with open(relaxed_path, "w", encoding="utf-8") as handle:
        for row in relaxed_candidates:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report["relaxed_yaw_diagnostic_candidates_jsonl"] = str(relaxed_path)
    report_path = output_dir / "yaw_positive_window_mining.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Yaw Positive Window Mining",
        "",
        f"- source_files: `{len(paths)}`",
        f"- rows: `{report['overall']['rows']}`",
        f"- target_rows: `{report['overall']['target_rows']}`",
        f"- target_episodes: `{report['overall']['target_episodes']}`",
        f"- near_visual_rows: `{report['overall']['near_visual_rows']}`",
        f"- near_visual_yaw_blocked_rows: `{report['overall']['near_visual_yaw_blocked_rows']}`",
        f"- yaw_observable_visual_not_near_rows: `{report['overall']['yaw_observable_visual_not_near_rows']}`",
        f"- relaxed_yaw_diagnostic_candidate_rows: `{report['overall']['relaxed_yaw_diagnostic_candidate_rows']}`",
        f"- relaxed_yaw_diagnostic_candidate_episodes: `{report['overall']['relaxed_yaw_diagnostic_candidate_episodes']}`",
        f"- recommendation: `{report['overall']['recommendation']}`",
        "",
        "## Top Episodes",
    ]
    for item in report["collection_focus_episodes"][:20]:
        md.append(
            f"- ep{int(item['episode_idx']):03d}: target={item['target_rows']}, "
            f"near_blocked={item['near_visual_yaw_blocked_rows']}, "
            f"yaw_obs_not_near={item['yaw_observable_not_near_rows']}, "
            f"entry_not_near={item['entry_visual_not_near_rows']}"
        )
    md_path = output_dir / "yaw_positive_window_mining.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    print(report_path)
    print(target_path)
    print(relaxed_path)
    print(md_path)


if __name__ == "__main__":
    main()
