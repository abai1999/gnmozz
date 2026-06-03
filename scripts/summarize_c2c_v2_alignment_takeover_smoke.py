#!/usr/bin/env python3
"""Summarize C2C v2 alignment takeover smoke traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


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
            rows.append(item)
    rows.sort(key=lambda r: (int(r.get("episode_idx", -1)), int(r.get("step", -1))))
    return rows


def _rate(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


def _session_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row.get("episode_idx", -1)), int(row.get("takeover_session_id", 0) or 0))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active_rows = [r for r in rows if bool(r.get("grasp_probe_active", False))]
    session_rows = [r for r in rows if int(r.get("takeover_session_id", 0) or 0) > 0]
    by_session: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        by_session[_session_key(row)].append(row)
    terminal_rows = [
        r for r in session_rows
        if str(r.get("terminal_state", ""))
        or bool(r.get("alignment_ready_for_handoff", False))
        or bool(r.get("safe_abstain_open", False))
        or bool(r.get("failed_terminal", False))
    ]
    terminal_by_session = {_session_key(r): r for r in terminal_rows}
    close_requested = [r for r in rows if bool(r.get("planner_gripper_close_requested", False))]
    close_blocked = [r for r in rows if bool(r.get("planner_gripper_close_blocked", False))]
    handoff_rows = [r for r in rows if bool(r.get("planner_gripper_handoff_allowed", False))]
    reward_after_handoff = [
        r for r in handoff_rows
        if float(r.get("reward", 0.0) or 0.0) > 0.0
    ]
    block_reasons = Counter(
        str(r.get("alignment_handoff_block_reason", ""))
        for r in rows
        if str(r.get("alignment_handoff_block_reason", ""))
    )
    terminal_reasons = Counter(
        str(r.get("exit_reason", ""))
        for r in terminal_rows
        if str(r.get("exit_reason", ""))
    )
    session_summaries = []
    for key, srows in sorted(by_session.items()):
        terminal = terminal_by_session.get(key, {})
        active = [r for r in srows if bool(r.get("grasp_probe_active", False))]
        session_summaries.append(
            {
                "episode_idx": int(key[0]),
                "takeover_session_id": int(key[1]),
                "rows": int(len(srows)),
                "active_rows": int(len(active)),
                "terminal_state": str(terminal.get("terminal_state", "")),
                "exit_reason": str(terminal.get("exit_reason", "")),
                "alignment_ready_for_handoff": bool(any(r.get("alignment_ready_for_handoff", False) for r in srows)),
                "safe_abstain_open": bool(any(r.get("safe_abstain_open", False) for r in srows)),
                "failed_terminal": bool(any(r.get("failed_terminal", False) for r in srows)),
                "planner_close_blocked_rows": int(sum(1 for r in srows if r.get("planner_gripper_close_blocked", False))),
                "xy_ready_rate": _rate(srows, "alignment_xy_ready"),
                "z_ready_rate": _rate(srows, "alignment_z_ready"),
                "yaw_ready_rate": _rate(srows, "alignment_yaw_ready"),
                "max_reward": float(max([float(r.get("reward", 0.0) or 0.0) for r in srows] or [0.0])),
            }
        )
    return {
        "schema_version": "c2c_v2_alignment_takeover_smoke_summary_v1",
        "rows": int(len(rows)),
        "active_rows": int(len(active_rows)),
        "takeover_sessions": int(len(by_session)),
        "terminal_takeover_sessions": int(len(terminal_by_session)),
        "alignment_success_sessions": int(sum(1 for item in session_summaries if item["alignment_ready_for_handoff"])),
        "safe_abstain_sessions": int(sum(1 for item in session_summaries if item["safe_abstain_open"])),
        "failed_terminal_sessions": int(sum(1 for item in session_summaries if item["failed_terminal"])),
        "alignment_success_rate": float(sum(1 for item in session_summaries if item["alignment_ready_for_handoff"]) / max(1, len(session_summaries))),
        "safe_abstain_rate": float(sum(1 for item in session_summaries if item["safe_abstain_open"]) / max(1, len(session_summaries))),
        "failed_terminal_rate": float(sum(1 for item in session_summaries if item["failed_terminal"]) / max(1, len(session_summaries))),
        "alignment_xy_ready_rate": _rate(active_rows, "alignment_xy_ready"),
        "alignment_z_ready_rate": _rate(active_rows, "alignment_z_ready"),
        "alignment_yaw_ready_rate": _rate(active_rows, "alignment_yaw_ready"),
        "handoff_allowed_rows": int(len(handoff_rows)),
        "planner_close_requested_rows": int(len(close_requested)),
        "planner_close_blocked_rows": int(len(close_blocked)),
        "planner_close_blocked_rate": float(len(close_blocked) / max(1, len(close_requested))),
        "reward_after_handoff_rows": int(len(reward_after_handoff)),
        "handoff_block_reason_counts": dict(block_reasons),
        "terminal_exit_reason_counts": dict(terminal_reasons),
        "by_session": session_summaries,
    }


def write_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment_takeover_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# C2C v2 Alignment Takeover Smoke Summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- active_rows: `{summary['active_rows']}`",
        f"- takeover_sessions: `{summary['takeover_sessions']}`",
        f"- terminal_takeover_sessions: `{summary['terminal_takeover_sessions']}`",
        f"- alignment_success_rate: `{summary['alignment_success_rate']:.3f}`",
        f"- safe_abstain_rate: `{summary['safe_abstain_rate']:.3f}`",
        f"- failed_terminal_rate: `{summary['failed_terminal_rate']:.3f}`",
        f"- alignment_xy_ready_rate: `{summary['alignment_xy_ready_rate']:.3f}`",
        f"- alignment_z_ready_rate: `{summary['alignment_z_ready_rate']:.3f}`",
        f"- alignment_yaw_ready_rate: `{summary['alignment_yaw_ready_rate']:.3f}`",
        f"- handoff_allowed_rows: `{summary['handoff_allowed_rows']}`",
        f"- planner_close_requested_rows: `{summary['planner_close_requested_rows']}`",
        f"- planner_close_blocked_rows: `{summary['planner_close_blocked_rows']}`",
        f"- planner_close_blocked_rate: `{summary['planner_close_blocked_rate']:.3f}`",
        f"- reward_after_handoff_rows: `{summary['reward_after_handoff_rows']}`",
        f"- handoff_block_reason_counts: `{summary['handoff_block_reason_counts']}`",
        f"- terminal_exit_reason_counts: `{summary['terminal_exit_reason_counts']}`",
        "",
        "## By Session",
    ]
    for item in summary["by_session"]:
        lines.append(
            f"- ep`{item['episode_idx']:03d}` session=`{item['takeover_session_id']}` "
            f"active=`{item['active_rows']}` terminal=`{item['terminal_state']}` "
            f"exit=`{item['exit_reason']}` handoff=`{item['alignment_ready_for_handoff']}` "
            f"safe=`{item['safe_abstain_open']}` failed=`{item['failed_terminal']}` "
            f"xy=`{item['xy_ready_rate']:.3f}` z=`{item['z_ready_rate']:.3f}` "
            f"yaw=`{item['yaw_ready_rate']:.3f}` close_blocked=`{item['planner_close_blocked_rows']}` "
            f"reward=`{item['max_reward']:.3f}`"
        )
    (output_dir / "alignment_takeover_smoke_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    summary = summarize(load_trace_rows(args.trace_dir))
    write_summary(summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
