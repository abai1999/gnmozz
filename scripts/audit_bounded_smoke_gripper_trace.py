#!/usr/bin/env python3
"""Audit bounded smoke gripper traces -> per-episode + aggregate JSON report.

Reads gripper_traces/ from a bounded smoke run dir and produces a structured
report.  Does not need privileged runtime data.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _episode_from_filename(path: Path) -> int:
    m = re.search(r"ep(\d+)", path.name)
    if m is None:
        raise ValueError(f"cannot extract episode from {path.name}")
    return int(m.group(1))


def _load_traces(trace_dir: Path) -> dict[int, list[dict]]:
    """Return {episode_index: [step_rows]} sorted by step."""
    episodes: dict[int, list[dict]] = {}
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep = _episode_from_filename(path)
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        episodes[ep] = sorted(rows, key=lambda r: int(r.get("step", 0)))
    return episodes


def _per_episode(ep: int, rows: list[dict]) -> dict:
    total = len(rows)

    # first_close_step
    first_close = -1
    for r in rows:
        fcs = r.get("first_close_step_so_far", -1)
        if fcs is not None and fcs >= 0:
            first_close = int(fcs)
            break

    # close_latch_set_count: count of non-zero latch remaining transitions
    close_latch_set = 0
    prev_latch = 0
    for r in rows:
        cur = int(r.get("refiner_current_close_latch_remaining", 0) or 0)
        if cur > 0 and prev_latch == 0:
            close_latch_set += 1
        prev_latch = cur

    # bounded_auto_close_applied_count
    bounded_auto_close_applied = sum(
        1 for r in rows if r.get("refiner_bounded_auto_close_applied") is True
    )

    # alignment_takeover_count
    alignment_takeover = sum(
        1 for r in rows if r.get("refiner_alignment_takeover_active") is True
    )

    # scorer_candidate_hist (from steps with valid index)
    candidate_hist: dict[str, int] = {}
    for r in rows:
        idx = r.get("refiner_last_scorer_candidate_index", -1)
        if idx is not None and idx >= 0:
            key = str(int(idx))
            candidate_hist[key] = candidate_hist.get(key, 0) + 1

    # alignment_gate_open_rate
    n_with_gate = 0
    gate_open_count = 0
    for r in rows:
        v = r.get("refiner_alignment_gate_open")
        if v is not None:
            n_with_gate += 1
            if v is True:
                gate_open_count += 1
    alignment_gate_open_rate = gate_open_count / max(n_with_gate, 1)

    # close_contact_rate: fraction of rows where contact_state is exactly "CLOSE_CONTACT"
    close_contact_count = sum(
        1 for r in rows if r.get("refiner_contact_state") == "CLOSE_CONTACT"
    )
    close_contact_rate = close_contact_count / max(total, 1)

    # workspace_violation_count is not in traces; set to -1 (derive from eval_results)
    workspace_violation_count = -1

    # clip_hit_rate: fraction of steps where alpha < 1 (i.e. clipped)
    # We don't have per-step alpha; derive from eval_results
    clip_hit_rate = -1.0

    # final_substage
    final_substage = "unknown"
    for r in reversed(rows):
        s = r.get("refiner_substage")
        if s:
            final_substage = str(s)
            break

    # final_close_state
    final_close_state = "unknown"
    for r in reversed(rows):
        s = r.get("refiner_close_state")
        if s:
            final_close_state = str(s)
            break

    # planner_close_intent_rate
    intent_count = sum(
        1 for r in rows if r.get("refiner_alignment_planner_close_intent") is True
    )
    planner_close_intent_rate = intent_count / max(total, 1)

    # ready_to_close_prob_mean
    prob_vals = []
    for r in rows:
        v = r.get("refiner_ready_to_close_prob_mean")
        if v is not None:
            try:
                prob_vals.append(float(v))
            except (TypeError, ValueError):
                pass
    ready_to_close_prob_mean = float(np.mean(prob_vals)) if prob_vals else 0.0

    # trigger_prob_mean
    trigger_vals = []
    for r in rows:
        v = r.get("refiner_current_trigger_prob")
        if v is not None:
            try:
                trigger_vals.append(float(v))
            except (TypeError, ValueError):
                pass
    trigger_prob_mean = float(np.mean(trigger_vals)) if trigger_vals else 0.0

    # invalid_action_count per episode
    invalid_action_count = sum(
        1 for r in rows if r.get("invalid_action") is True
    )

    # close_requirement_satisfied rate
    close_req_satisfied_count = sum(
        1 for r in rows if r.get("refiner_alignment_close_requirement_satisfied") is True
    )

    # close state machine summary
    close_state_hist: dict[str, int] = {}
    for r in rows:
        sm = r.get("refiner_close_state_machine")
        if isinstance(sm, dict):
            state = str(sm.get("state", "unknown"))
            close_state_hist[state] = close_state_hist.get(state, 0) + 1
        else:
            state = str(r.get("refiner_close_state", "unknown"))
            close_state_hist[state] = close_state_hist.get(state, 0) + 1

    # close_action_decision hist
    action_hist: dict[str, int] = {}
    for r in rows:
        a = str(r.get("refiner_close_action_decision", "unknown"))
        action_hist[a] = action_hist.get(a, 0) + 1

    # alignment_blocked_reason hist
    block_hist: dict[str, int] = {}
    for r in rows:
        reason = str(r.get("refiner_alignment_blocked_reason", "unknown"))
        block_hist[reason] = block_hist.get(reason, 0) + 1

    return {
        "episode": ep,
        "steps": total,
        "first_close_step": first_close,
        "close_latch_set_count": close_latch_set,
        "bounded_auto_close_applied_count": bounded_auto_close_applied,
        "alignment_takeover_count": alignment_takeover,
        "scorer_candidate_hist": candidate_hist,
        "alignment_gate_open_rate": round(alignment_gate_open_rate, 6),
        "close_contact_rate": round(close_contact_rate, 6),
        "workspace_violation_count": workspace_violation_count,
        "clip_hit_rate": clip_hit_rate,
        "final_substage": final_substage,
        "final_close_state": final_close_state,
        "planner_close_intent_rate": round(planner_close_intent_rate, 6),
        "ready_to_close_prob_mean": round(ready_to_close_prob_mean, 6),
        "trigger_prob_mean": round(trigger_prob_mean, 6),
        "invalid_action_count": invalid_action_count,
        "close_requirement_satisfied_rate": (
            round(close_req_satisfied_count / max(total, 1), 6)
        ),
        "close_state_hist": close_state_hist,
        "close_action_decision_hist": action_hist,
        "alignment_blocked_reason_hist": block_hist,
    }


def _enrich_with_eval_results(per_ep: list[dict], eval_path: Path) -> list[dict]:
    """Pull per-episode workspace_violation_count and clip_hit_rate from eval_results."""
    if not eval_path.exists():
        return per_ep

    with eval_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    stage_stats = eval_data.get("stage_stats", [])
    refiner_stats = eval_data.get("refiner_stats", [])

    # Build lookup by episode_index
    stage_by_ep: dict[int, dict] = {}
    for s in stage_stats:
        stage_by_ep[int(s.get("episode_index", -1))] = s

    refiner_by_ep: dict[int, dict] = {}
    for r in refiner_stats:
        ep_idx = None
        # refiner_stats may or may not have episode_index
        # We match by position order to the episode_indices list
        if "episode_index" in r:
            refiner_by_ep[int(r["episode_index"])] = r

    for entry in per_ep:
        ep = entry["episode"]
        ss = stage_by_ep.get(ep, {})
        entry["workspace_violation_count"] = int(ss.get("workspace_violation_count", -1))

        # Try refiner by episode
        rs = refiner_by_ep.get(ep)
        if rs is not None:
            entry["clip_hit_rate"] = float(rs.get("clip_hit_rate", -1.0))
        else:
            # Fall back to positional matching
            ep_indices = eval_data.get("episode_indices", [])
            try:
                pos = ep_indices.index(ep)
                if pos < len(refiner_stats):
                    rs = refiner_stats[pos]
                    entry["clip_hit_rate"] = float(rs.get("clip_hit_rate", -1.0))
            except (ValueError, IndexError):
                pass

    return per_ep


def _aggregate(per_ep: list[dict], eval_path: Path | None) -> dict:
    n = len(per_ep)

    success_rate = -1.0
    total_invalid = 0
    total_bounded_auto_close = 0
    total_steps = 0
    total_alignment_takeover = 0
    planner_close_intent_weighted = 0.0
    ready_to_close_sum = 0.0
    trigger_sum = 0.0

    for entry in per_ep:
        total_invalid += entry["invalid_action_count"]
        total_bounded_auto_close += entry["bounded_auto_close_applied_count"]
        total_steps += entry["steps"]
        total_alignment_takeover += entry["alignment_takeover_count"]
        planner_close_intent_weighted += entry["planner_close_intent_rate"] * entry["steps"]
        ready_to_close_sum += entry["ready_to_close_prob_mean"] * entry["steps"]
        trigger_sum += entry["trigger_prob_mean"] * entry["steps"]

    # Enrich from eval_results.json
    if eval_path is not None and eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)
        success_rate = float(eval_data.get("success_rate", -1.0))
        total_invalid = int(eval_data.get("invalid_action_count", total_invalid))
        total_workspace_violations = int(eval_data.get("workspace_violation_count", -1))

        # These may come from eval_results aggregate
        planner_close_intent_rate = float(eval_data.get("planner_close_intent_rate", -1.0))
        ready_to_close_prob_mean_agg = float(eval_data.get("ready_to_close_prob_mean", -1.0))
        trigger_prob_mean_agg = float(eval_data.get("trigger_prob_mean", -1.0))
    else:
        total_workspace_violations = sum(
            e.get("workspace_violation_count", 0) for e in per_ep
        )
        planner_close_intent_rate = (
            planner_close_intent_weighted / max(total_steps, 1)
        )
        ready_to_close_prob_mean_agg = ready_to_close_sum / max(total_steps, 1)
        trigger_prob_mean_agg = trigger_sum / max(total_steps, 1)

    bounded_auto_close_applied_rate = total_bounded_auto_close / max(total_steps, 1)

    return {
        "num_episodes": n,
        "total_steps": total_steps,
        "success_rate": success_rate,
        "invalid_action_count": total_invalid,
        "planner_close_intent_rate": round(planner_close_intent_rate, 6),
        "ready_to_close_prob_mean": round(ready_to_close_prob_mean_agg, 6),
        "trigger_prob_mean": round(trigger_prob_mean_agg, 6),
        "bounded_auto_close_applied_rate": round(bounded_auto_close_applied_rate, 6),
        "bounded_auto_close_applied_count": total_bounded_auto_close,
        "alignment_takeover_count": total_alignment_takeover,
        "workspace_violation_count": total_workspace_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit bounded smoke gripper traces -> per-episode + aggregate JSON"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Bounded smoke run directory (contains .../gripper_traces/ and eval_results.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for output JSON report. Default: <run-dir>/trace_audit_report.json",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    # Find the task subdir (contains gripper_traces/ and eval_results.json)
    eval_candidates = sorted(run_dir.glob("*/eval_results.json"))
    if not eval_candidates:
        print(f"ERROR: no eval_results.json found under {run_dir}", flush=True)
        raise SystemExit(1)

    task_dir = eval_candidates[0].parent
    trace_dir = task_dir / "gripper_traces"
    if not trace_dir.is_dir():
        print(f"ERROR: no gripper_traces/ in {task_dir}", flush=True)
        raise SystemExit(1)

    eval_path = task_dir / "eval_results.json"
    print(f"[audit] run_dir    = {run_dir}")
    print(f"[audit] task_dir   = {task_dir}")
    print(f"[audit] trace_dir  = {trace_dir}")
    print(f"[audit] eval_path  = {eval_path}")

    episodes = _load_traces(trace_dir)
    print(f"[audit] loaded {len(episodes)} episodes: {sorted(episodes.keys())}")

    per_ep = [_per_episode(ep, rows) for ep, rows in sorted(episodes.items())]
    per_ep = _enrich_with_eval_results(per_ep, eval_path)
    agg = _aggregate(per_ep, eval_path)

    report = {
        "audit": "bounded_smoke_gripper_trace",
        "run_dir": str(run_dir),
        "per_episode": per_ep,
        "aggregate": agg,
    }

    output_path = args.output or (task_dir / "trace_audit_report.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    print(f"\n[audit] report written to {output_path}")
    print(f"\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"\n=== Per-Episode ===")
    for entry in per_ep:
        print(f"  ep{entry['episode']:03d}: steps={entry['steps']} "
              f"first_close_step={entry['first_close_step']} "
              f"close_latch_set={entry['close_latch_set_count']} "
              f"bounded_auto_close_applied={entry['bounded_auto_close_applied_count']} "
              f"alignment_takeover={entry['alignment_takeover_count']} "
              f"alignment_gate_open_rate={entry['alignment_gate_open_rate']} "
              f"close_contact_rate={entry['close_contact_rate']} "
              f"final_substage={entry['final_substage']} "
              f"final_close_state={entry['final_close_state']} "
              f"candidate_hist={entry['scorer_candidate_hist']}")


if __name__ == "__main__":
    main()
