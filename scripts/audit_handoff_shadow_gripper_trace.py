#!/usr/bin/env python3
"""Audit handoff-shadow gripper traces: handoff readiness + close plumbing signals.

Reads gripper_traces/ from a bounded run dir and produces per-episode + aggregate
handoff-readiness diagnostics. Does not need privileged runtime data.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def _episode_from_filename(path: Path) -> int:
    m = re.search(r"ep(\d+)", path.name)
    if m is None:
        raise ValueError(f"cannot extract episode from {path.name}")
    return int(m.group(1))


def _load_traces(trace_dir: Path) -> dict[int, list[dict]]:
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


def _handoff_diagnostics(rows: list[dict]) -> dict:
    """Extract handoff-specific diagnostics from trace rows."""
    total = len(rows)

    # --- handoff_spec_name histogram ---
    spec_hist: dict[str, int] = {}
    for r in rows:
        sn = str(r.get("refiner_current_handoff_spec_name", "unknown"))
        spec_hist[sn] = spec_hist.get(sn, 0) + 1

    # --- handoff_target_role histogram ---
    role_hist: dict[str, int] = {}
    for r in rows:
        rn = str(r.get("refiner_current_handoff_target_role", "unknown"))
        role_hist[rn] = role_hist.get(rn, 0) + 1

    # --- handoff_ready (from refiner stats) ---
    handoff_ready_count = sum(
        1 for r in rows if r.get("refiner_current_handoff_ready") is True
    )

    # --- handoff_ready_pred (from trace: handoff_ready_provider / handoff_ready_pred) ---
    pred_vals = []
    for r in rows:
        # Use trace-level handoff_ready_pred field
        v = r.get("handoff_ready_pred")
        if v is not None:
            try:
                pred_vals.append(float(v))
            except (TypeError, ValueError):
                pass
    # Also try refiner_close_handoff_ready_pred
    if not pred_vals:
        for r in rows:
            v = r.get("refiner_close_handoff_ready_pred")
            if v is not None:
                try:
                    pred_vals.append(float(v))
                except (TypeError, ValueError):
                    pass

    # handoff_ready_provider boolean
    provider_ready_count = sum(
        1 for r in rows if r.get("handoff_ready_provider") is True
    )

    # --- handoff_metrics from close_state_machine ---
    handoff_metrics_xy: list[float] = []
    handoff_metrics_z: list[float] = []
    handoff_metrics_yaw: list[float] = []
    handoff_ready_streak_vals: list[int] = []
    handoff_runtime_geometry_ready_count = 0
    handoff_shadow_blocks_apply_count = 0
    handoff_fallback_enabled_count = 0

    for r in rows:
        sm = r.get("refiner_close_state_machine")
        if isinstance(sm, dict):
            if sm.get("runtime_geometry_ready") is True:
                handoff_runtime_geometry_ready_count += 1
            if sm.get("handoff_shadow_blocks_apply") is True:
                handoff_shadow_blocks_apply_count += 1
            if sm.get("fallback_enabled") is True:
                handoff_fallback_enabled_count += 1
            rs = sm.get("ready_streak")
            if rs is not None:
                try:
                    handoff_ready_streak_vals.append(int(rs))
                except (TypeError, ValueError):
                    pass

    # --- Handoff aux fields from trace ---
    handoff_aux_pred_ready_prob: list[float] = []
    handoff_aux_pred_uncertainty: list[float] = []
    handoff_aux_pred_band: list[int] = []
    handoff_aux_metric_valid_count = 0

    # These are in the handoff_aux embedded in the trace's handoff_metrics_provider
    # or handoff_aux_provider fields
    for r in rows:
        ha = r.get("handoff_aux_provider")
        if isinstance(ha, dict):
            pr = ha.get("pred_ready_prob")
            if pr is not None:
                try:
                    handoff_aux_pred_ready_prob.append(float(pr))
                except (TypeError, ValueError):
                    pass
            pu = ha.get("pred_uncertainty")
            if pu is not None:
                try:
                    handoff_aux_pred_uncertainty.append(float(pu))
                except (TypeError, ValueError):
                    pass
            pb = ha.get("pred_band_index")
            if pb is not None:
                try:
                    handoff_aux_pred_band.append(int(pb))
                except (TypeError, ValueError):
                    pass
            if ha.get("metric_valid") is True:
                handoff_aux_metric_valid_count += 1

    # Also check handoff_metrics_provider
    if not handoff_aux_pred_ready_prob:
        for r in rows:
            hm = r.get("handoff_metrics_provider")
            if isinstance(hm, dict):
                # It may have ready-related fields
                pr = hm.get("pred_ready_prob") or hm.get("ready_prob")
                if pr is not None:
                    try:
                        handoff_aux_pred_ready_prob.append(float(pr))
                    except (TypeError, ValueError):
                        pass

    # --- Handoff threshold fields ---
    threshold_xy_vals: list[float] = []
    threshold_z_vals: list[float] = []
    threshold_yaw_vals: list[float] = []
    for r in rows:
        ht = r.get("handoff_metric_thresholds_provider")
        if isinstance(ht, dict):
            for key, store in [("xy_error", threshold_xy_vals), ("abs_z_error", threshold_z_vals), ("yaw_error", threshold_yaw_vals)]:
                v = ht.get(key)
                if v is not None:
                    try:
                        store.append(float(v))
                    except (TypeError, ValueError):
                        pass

    def _percentiles(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"n": len(vals), "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
        }

    return {
        "handoff_spec_name_hist": spec_hist,
        "handoff_target_role_hist": role_hist,
        "handoff_ready_rate": round(handoff_ready_count / max(total, 1), 6),
        "handoff_ready_provider_rate": round(provider_ready_count / max(total, 1), 6),
        "handoff_ready_pred": _percentiles(pred_vals),
        "handoff_aux_pred_ready_prob": _percentiles(handoff_aux_pred_ready_prob),
        "handoff_aux_pred_uncertainty": _percentiles(handoff_aux_pred_uncertainty),
        "handoff_aux_pred_band_hist": dict(Counter(str(b) for b in handoff_aux_pred_band)),
        "handoff_aux_metric_valid_count": handoff_aux_metric_valid_count,
        "handoff_runtime_geometry_ready_rate": round(handoff_runtime_geometry_ready_count / max(total, 1), 6),
        "handoff_shadow_blocks_apply_rate": round(handoff_shadow_blocks_apply_count / max(total, 1), 6),
        "handoff_fallback_enabled_rate": round(handoff_fallback_enabled_count / max(total, 1), 6),
        "handoff_ready_streak_max": int(max(handoff_ready_streak_vals)) if handoff_ready_streak_vals else 0,
        "handoff_metric_thresholds_xy": _percentiles(threshold_xy_vals),
        "handoff_metric_thresholds_z": _percentiles(threshold_z_vals),
        "handoff_metric_thresholds_yaw": _percentiles(threshold_yaw_vals),
    }


def _per_episode(ep: int, rows: list[dict]) -> dict:
    total = len(rows)

    # --- standard fields ---
    first_close = -1
    for r in rows:
        fcs = r.get("first_close_step_so_far", -1)
        if fcs is not None and fcs >= 0:
            first_close = int(fcs)
            break

    alignment_takeover = sum(
        1 for r in rows if r.get("refiner_alignment_takeover_active") is True
    )

    candidate_hist: dict[str, int] = {}
    for r in rows:
        idx = r.get("refiner_last_scorer_candidate_index", -1)
        if idx is not None and idx >= 0:
            candidate_hist[str(int(idx))] = candidate_hist.get(str(int(idx)), 0) + 1

    n_with_gate = 0
    gate_open_count = 0
    for r in rows:
        v = r.get("refiner_alignment_gate_open")
        if v is not None:
            n_with_gate += 1
            if v is True:
                gate_open_count += 1
    alignment_gate_open_rate = gate_open_count / max(n_with_gate, 1)

    close_contact_count = sum(
        1 for r in rows if r.get("refiner_contact_state") == "CLOSE_CONTACT"
    )

    final_substage = "unknown"
    for r in reversed(rows):
        s = r.get("refiner_substage")
        if s:
            final_substage = str(s)
            break

    final_close_state = "unknown"
    for r in reversed(rows):
        s = r.get("refiner_close_state")
        if s:
            final_close_state = str(s)
            break

    intent_count = sum(
        1 for r in rows if r.get("refiner_alignment_planner_close_intent") is True
    )
    planner_close_intent_rate = intent_count / max(total, 1)

    prob_vals = []
    for r in rows:
        v = r.get("refiner_ready_to_close_prob_mean")
        if v is not None:
            try:
                prob_vals.append(float(v))
            except (TypeError, ValueError):
                pass
    ready_to_close_prob_mean = float(np.mean(prob_vals)) if prob_vals else 0.0

    trigger_vals = []
    for r in rows:
        v = r.get("refiner_current_trigger_prob")
        if v is not None:
            try:
                trigger_vals.append(float(v))
            except (TypeError, ValueError):
                pass
    trigger_prob_mean = float(np.mean(trigger_vals)) if trigger_vals else 0.0

    bounded_auto_close_applied = sum(
        1 for r in rows if r.get("refiner_bounded_auto_close_applied") is True
    )

    action_hist: dict[str, int] = {}
    for r in rows:
        a = str(r.get("refiner_close_action_decision", "unknown"))
        action_hist[a] = action_hist.get(a, 0) + 1

    close_state_hist: dict[str, int] = {}
    for r in rows:
        sm = r.get("refiner_close_state_machine")
        if isinstance(sm, dict):
            state = str(sm.get("state", "unknown"))
        else:
            state = str(r.get("refiner_close_state", "unknown"))
        close_state_hist[state] = close_state_hist.get(state, 0) + 1

    block_hist: dict[str, int] = {}
    for r in rows:
        reason = str(r.get("refiner_alignment_blocked_reason", "unknown"))
        block_hist[reason] = block_hist.get(reason, 0) + 1

    close_state_machine_sm = None
    for r in reversed(rows):
        sm = r.get("refiner_close_state_machine")
        if isinstance(sm, dict):
            close_state_machine_sm = {
                "state": sm.get("state"),
                "handoff_spec_name": sm.get("handoff_spec_name"),
                "handoff_ready_pred": sm.get("handoff_ready_pred"),
                "handoff_ready_applied": sm.get("handoff_ready_applied"),
                "handoff_shadow_only": sm.get("handoff_shadow_only"),
                "handoff_shadow_blocks_apply": sm.get("handoff_shadow_blocks_apply"),
                "runtime_geometry_ready": sm.get("runtime_geometry_ready"),
                "has_motion_target": sm.get("has_motion_target"),
                "has_handoff_geometry": sm.get("has_handoff_geometry"),
                "fallback_enabled": sm.get("fallback_enabled"),
                "fallback_used": sm.get("fallback_used"),
                "wants_close": sm.get("wants_close"),
                "planner_close_intent": sm.get("planner_close_intent"),
                "blocked_reason": sm.get("blocked_reason"),
                "ready_streak": sm.get("ready_streak"),
                "streak_ready": sm.get("streak_ready"),
                "xy_error": sm.get("xy_error"),
                "abs_z_error": sm.get("abs_z_error"),
                "yaw_error": sm.get("yaw_error"),
                "xy_threshold": sm.get("xy_threshold"),
                "abs_z_threshold": sm.get("abs_z_threshold"),
                "yaw_threshold": sm.get("yaw_threshold"),
            }
            break

    base = {
        "episode": ep,
        "steps": total,
        "first_close_step": first_close,
        "alignment_takeover_count": alignment_takeover,
        "scorer_candidate_hist": candidate_hist,
        "alignment_gate_open_rate": round(alignment_gate_open_rate, 6),
        "close_contact_rate": round(close_contact_count / max(total, 1), 6),
        "final_substage": final_substage,
        "final_close_state": final_close_state,
        "planner_close_intent_rate": round(planner_close_intent_rate, 6),
        "ready_to_close_prob_mean": round(ready_to_close_prob_mean, 6),
        "trigger_prob_mean": round(trigger_prob_mean, 6),
        "bounded_auto_close_applied_count": bounded_auto_close_applied,
        "bounded_auto_close_applied_rate": round(bounded_auto_close_applied / max(total, 1), 6),
        "close_action_decision_hist": action_hist,
        "close_state_hist": close_state_hist,
        "alignment_blocked_reason_hist": block_hist,
        "last_close_state_machine_snapshot": close_state_machine_sm,
    }
    base.update(_handoff_diagnostics(rows))
    return base


def _enrich_with_eval_results(per_ep: list[dict], eval_path: Path) -> list[dict]:
    if not eval_path.exists():
        return per_ep
    with eval_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)
    stage_stats = eval_data.get("stage_stats", [])
    refiner_stats = eval_data.get("refiner_stats", [])
    stage_by_ep: dict[int, dict] = {int(s.get("episode_index", -1)): s for s in stage_stats}
    refiner_by_ep: dict[int, dict] = {}
    for r in refiner_stats:
        if "episode_index" in r:
            refiner_by_ep[int(r["episode_index"])] = r
    for entry in per_ep:
        ep = entry["episode"]
        ss = stage_by_ep.get(ep, {})
        entry["workspace_violation_count"] = int(ss.get("workspace_violation_count", -1))
        rs = refiner_by_ep.get(ep)
        if rs is None:
            ep_indices = eval_data.get("episode_indices", [])
            try:
                pos = ep_indices.index(ep)
                if pos < len(refiner_stats):
                    rs = refiner_stats[pos]
            except (ValueError, IndexError):
                pass
        if rs is not None:
            entry["clip_hit_rate"] = float(rs.get("clip_hit_rate", -1.0))
        else:
            entry["clip_hit_rate"] = -1.0
    return per_ep


def _aggregate(per_ep: list[dict], eval_path: Path | None) -> dict:
    n = len(per_ep)
    total_steps = sum(e["steps"] for e in per_ep)
    total_takeover = sum(e["alignment_takeover_count"] for e in per_ep)
    total_bounded_auto_close = sum(e["bounded_auto_close_applied_count"] for e in per_ep)
    total_invalid = 0

    success_rate = -1.0
    planner_close_intent_rate = -1.0
    ready_to_close_prob_mean_agg = -1.0
    trigger_prob_mean_agg = -1.0
    total_ws_violations = -1
    target_provider_source_hist = {}
    handoff_provider_ckpt_used = "unknown"

    if eval_path is not None and eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            ed = json.load(f)
        success_rate = float(ed.get("success_rate", -1.0))
        total_invalid = int(ed.get("invalid_action_count", 0))
        planner_close_intent_rate = float(ed.get("planner_close_intent_rate", -1.0))
        ready_to_close_prob_mean_agg = float(ed.get("ready_to_close_prob_mean", -1.0))
        trigger_prob_mean_agg = float(ed.get("trigger_prob_mean", -1.0))
        total_ws_violations = int(ed.get("workspace_violation_count", -1))
        target_provider_source_hist = ed.get("target_provider_source_hist", {})
        handoff_provider_ckpt_used = str(ed.get("handoff_provider_ckpt", "unknown"))

    # Aggregate handoff_ready_pred across all rows
    all_pred_ready_probs = []
    all_pred_uncertainties = []
    all_pred_bands = []
    for ep_data in per_ep:
        h = ep_data.get("handoff_aux_pred_ready_prob", {})
        if isinstance(h, dict) and h.get("n", 0) > 0:
            all_pred_ready_probs.append(h)
        h2 = ep_data.get("handoff_aux_pred_uncertainty", {})
        if isinstance(h2, dict) and h2.get("n", 0) > 0:
            all_pred_uncertainties.append(h2)

    return {
        "num_episodes": n,
        "total_steps": total_steps,
        "success_rate": success_rate,
        "invalid_action_count": total_invalid,
        "planner_close_intent_rate": planner_close_intent_rate,
        "ready_to_close_prob_mean": ready_to_close_prob_mean_agg,
        "trigger_prob_mean": trigger_prob_mean_agg,
        "bounded_auto_close_applied_rate": round(total_bounded_auto_close / max(total_steps, 1), 6),
        "bounded_auto_close_applied_count": total_bounded_auto_close,
        "alignment_takeover_count": total_takeover,
        "workspace_violation_count": total_ws_violations,
        "target_provider_source_hist": target_provider_source_hist,
        "handoff_provider_ckpt": handoff_provider_ckpt_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit handoff-shadow gripper traces"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
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
    print(f"[audit] run_dir   = {run_dir}")
    print(f"[audit] task_dir  = {task_dir}")
    print(f"[audit] trace_dir = {trace_dir}")

    episodes = _load_traces(trace_dir)
    print(f"[audit] loaded {len(episodes)} episodes: {sorted(episodes.keys())}")

    per_ep = [_per_episode(ep, rows) for ep, rows in sorted(episodes.items())]
    per_ep = _enrich_with_eval_results(per_ep, eval_path)
    agg = _aggregate(per_ep, eval_path)

    report = {
        "audit": "handoff_shadow_gripper_trace",
        "run_dir": str(run_dir),
        "per_episode": per_ep,
        "aggregate": agg,
    }

    output_path = args.output or (task_dir / "handoff_shadow_audit_report.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[audit] report -> {output_path}")
    print(f"\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    for entry in per_ep:
        ep = entry["episode"]
        print(f"\n=== ep{ep:03d} ===")
        print(f"  steps={entry['steps']}  first_close_step={entry['first_close_step']}")
        print(f"  final_substage={entry['final_substage']}  final_close_state={entry['final_close_state']}")
        print(f"  alignment_takeover={entry['alignment_takeover_count']}  gate_open_rate={entry['alignment_gate_open_rate']}")
        print(f"  close_contact_rate={entry['close_contact_rate']}")
        print(f"  planner_close_intent_rate={entry['planner_close_intent_rate']}")
        print(f"  ready_to_close_prob_mean={entry['ready_to_close_prob_mean']}")
        print(f"  trigger_prob_mean={entry['trigger_prob_mean']}")
        print(f"  bounded_auto_close_applied={entry['bounded_auto_close_applied_count']}")
        print(f"  handoff_spec_name_hist={entry['handoff_spec_name_hist']}")
        print(f"  handoff_target_role_hist={entry['handoff_target_role_hist']}")
        print(f"  handoff_ready_rate={entry['handoff_ready_rate']}")
        print(f"  handoff_ready_provider_rate={entry['handoff_ready_provider_rate']}")
        print(f"  handoff_ready_pred={entry['handoff_ready_pred']}")
        print(f"  handoff_aux_pred_ready_prob={entry['handoff_aux_pred_ready_prob']}")
        print(f"  handoff_aux_pred_uncertainty={entry['handoff_aux_pred_uncertainty']}")
        print(f"  handoff_aux_pred_band_hist={entry['handoff_aux_pred_band_hist']}")
        print(f"  handoff_runtime_geometry_ready_rate={entry['handoff_runtime_geometry_ready_rate']}")
        print(f"  handoff_shadow_blocks_apply_rate={entry['handoff_shadow_blocks_apply_rate']}")
        print(f"  handoff_fallback_enabled_rate={entry['handoff_fallback_enabled_rate']}")
        print(f"  handoff_ready_streak_max={entry['handoff_ready_streak_max']}")
        print(f"  scorer_candidate_hist={entry['scorer_candidate_hist']}")
        print(f"  close_action_decision_hist={entry['close_action_decision_hist']}")
        print(f"  close_state_hist={entry['close_state_hist']}")
        if entry.get("last_close_state_machine_snapshot"):
            sm = entry["last_close_state_machine_snapshot"]
            print(f"  last_close_sm: state={sm['state']} wants_close={sm['wants_close']} "
                  f"planner_close_intent={sm['planner_close_intent']} "
                  f"handoff_spec_name={sm['handoff_spec_name']} "
                  f"handoff_ready_pred={sm['handoff_ready_pred']} "
                  f"runtime_geometry_ready={sm['runtime_geometry_ready']} "
                  f"blocked_reason={sm['blocked_reason']}")


if __name__ == "__main__":
    main()
