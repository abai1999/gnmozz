#!/usr/bin/env python3
"""Offline close-readiness replay audit.

Post-hoc analysis of handoff-shadow traces: simulates "what if planner emitted
close at each step" and evaluates every gate in the close-readiness chain.

Does NOT run new episodes. Does NOT modify checkpoints. Read-only.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ep_from_filename(path: Path) -> int:
    m = re.search(r"ep(\d+)", path.name)
    if m is None:
        raise ValueError(f"cannot extract episode from {path.name}")
    return int(m.group(1))


def _load_traces(trace_dir: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = {}
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep = _ep_from_filename(path)
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        episodes[ep] = sorted(rows, key=lambda r: int(r.get("step", 0)))
    return episodes


def _load_shadow_traces(trace_dir: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = {}
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep = _ep_from_filename(path)
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        episodes[ep] = sorted(rows, key=lambda r: int(r.get("step", 0)))
    return episodes


def _p(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


# ---------------------------------------------------------------------------
# per-step replay
# ---------------------------------------------------------------------------

def _replay_step(r: dict) -> dict:
    """Compute what would happen if planner emitted close at this step."""
    ha = r.get("handoff_aux_provider") or {}
    hm = r.get("handoff_metrics_provider") or {}
    ht = r.get("handoff_metric_thresholds_provider") or {}
    sm = r.get("refiner_close_state_machine") or {}

    # --- handoff model raw signals ---
    ready_prob = float(ha.get("pred_ready_prob", np.nan))
    uncertainty = float(ha.get("pred_uncertainty", np.nan))
    band = ha.get("pred_band_index")

    # --- metric errors ---
    xy_error = float(hm.get("xy_error", np.nan))
    z_error = float(hm.get("abs_z_error", np.nan))
    yaw_error = float(hm.get("yaw_error", np.nan))

    # --- metric thresholds ---
    xy_th = float(ht.get("xy_error", 0.0085))
    z_th = float(ht.get("abs_z_error", 0.0035))
    yaw_th = float(ht.get("yaw_error", 0.1243404))

    # --- metric pass ---
    xy_pass = bool(np.isfinite(xy_error) and xy_error <= xy_th)
    z_pass = bool(np.isfinite(z_error) and z_error <= z_th)
    yaw_pass = bool(yaw_th < 0.0 or (np.isfinite(yaw_error) and yaw_error <= yaw_th))
    metric_pass = bool(xy_pass and z_pass and yaw_pass)

    # --- handoff_ready_bool (standard thresholds) ---
    handoff_ready_bool = bool(
        np.isfinite(ready_prob) and ready_prob >= 0.5
        and np.isfinite(uncertainty) and uncertainty <= 0.75
        and metric_pass
    )

    # --- runtime state ---
    takeover = bool(r.get("refiner_alignment_takeover_active", False))
    gate_open = bool(r.get("refiner_alignment_gate_open", False))
    blocked_reason = str(r.get("refiner_alignment_blocked_reason", "unknown"))
    support_blocked = blocked_reason == "support"
    close_intent = bool(r.get("refiner_alignment_planner_close_intent", False))

    # --- handoff spec ---
    handoff_spec_name = str(r.get("refiner_current_handoff_spec_name", "none"))
    has_handoff = handoff_spec_name != "none" and handoff_spec_name != "unknown"

    return {
        "step": int(r.get("step", -1)),
        "takeover": takeover,
        "gate_open": gate_open,
        "blocked_reason": blocked_reason,
        "support_blocked": support_blocked,
        "close_intent_actual": close_intent,
        "handoff_spec_available": has_handoff,
        "ready_prob": None if not np.isfinite(ready_prob) else round(float(ready_prob), 8),
        "uncertainty": None if not np.isfinite(uncertainty) else round(float(uncertainty), 8),
        "band": band,
        "xy_error": None if not np.isfinite(xy_error) else round(float(xy_error), 8),
        "z_error": None if not np.isfinite(z_error) else round(float(z_error), 8),
        "yaw_error": None if not np.isfinite(yaw_error) else round(float(yaw_error), 8),
        "xy_threshold": xy_th,
        "z_threshold": z_th,
        "yaw_threshold": yaw_th,
        "xy_pass": xy_pass,
        "z_pass": z_pass,
        "yaw_pass": yaw_pass,
        "metric_pass": metric_pass,
        "handoff_ready_bool_standard": handoff_ready_bool,
    }


# ---------------------------------------------------------------------------
# uncertainty sweep
# ---------------------------------------------------------------------------

SWEEP_UNC = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
SWEEP_PROB = [0.5, 0.7, 0.9]


def _sweep(replay_rows: list[dict]) -> list[dict]:
    """Sweep uncertainty and ready_prob thresholds."""
    results = []
    for unc_th in SWEEP_UNC:
        for prob_th in SWEEP_PROB:
            all_pass = 0
            takeover_pass = 0
            support_pass = 0
            late_takeover_pass = 0
            total_all = len(replay_rows)
            total_takeover = 0
            total_support = 0
            total_late = 0
            first_pass_eps: dict[int, int] = {}
            streak_lengths = []
            cur_streak = 0

            for r in replay_rows:
                rp = r["ready_prob"]
                pu = r["uncertainty"]
                mp = r["metric_pass"]
                ready = bool(
                    rp is not None and rp >= prob_th
                    and pu is not None and pu <= unc_th
                    and mp
                )
                step = r["step"]

                if ready:
                    all_pass += 1
                    cur_streak += 1
                else:
                    if cur_streak > 0:
                        streak_lengths.append(cur_streak)
                    cur_streak = 0

                if r["takeover"]:
                    total_takeover += 1
                    if ready:
                        takeover_pass += 1

                if r["support_blocked"]:
                    total_support += 1
                    if ready:
                        support_pass += 1

                if r["takeover"] and step >= 100:
                    total_late += 1
                    if ready:
                        late_takeover_pass += 1

            if cur_streak > 0:
                streak_lengths.append(cur_streak)

            results.append({
                "unc_threshold": unc_th,
                "prob_threshold": prob_th,
                "ready_pass_rate_all": round(all_pass / max(total_all, 1), 6),
                "ready_pass_count_all": all_pass,
                "ready_pass_rate_takeover": round(takeover_pass / max(total_takeover, 1), 6),
                "ready_pass_count_takeover": takeover_pass,
                "ready_pass_rate_support_blocked": round(support_pass / max(total_support, 1), 6),
                "ready_pass_count_support_blocked": support_pass,
                "ready_pass_rate_late_takeover": round(late_takeover_pass / max(total_late, 1), 6),
                "ready_pass_count_late_takeover": late_takeover_pass,
                "streak_max": int(max(streak_lengths)) if streak_lengths else 0,
                "streak_mean": round(float(np.mean(streak_lengths)), 2) if streak_lengths else 0.0,
            })
    return results


# ---------------------------------------------------------------------------
# metric distance audit
# ---------------------------------------------------------------------------

def _metric_audit(replay_rows: list[dict]) -> dict:
    takeover = [r for r in replay_rows if r["takeover"]]
    non_takeover = [r for r in replay_rows if not r["takeover"]]
    support = [r for r in replay_rows if r["support_blocked"]]

    def _stats(rows, label):
        xy = np.array([r["xy_error"] for r in rows if r["xy_error"] is not None], dtype=np.float64)
        z = np.array([r["z_error"] for r in rows if r["z_error"] is not None], dtype=np.float64)
        yaw = np.array([r["yaw_error"] for r in rows if r["yaw_error"] is not None], dtype=np.float64)

        # Which axis blocks most?
        xy_block = int((xy > 0.0085).sum())
        z_block = int((z > 0.0035).sum())
        yaw_block = int((yaw > 0.12434).sum())

        return {
            "n": len(rows),
            "xy_error": {
                "mean": round(float(xy.mean()), 6), "p50": round(_p(xy, 50), 6),
                "p90": round(_p(xy, 90), 6), "max": round(float(xy.max()), 6),
                "threshold": 0.0085, "pass_count": int(len(xy)) - xy_block, "block_count": xy_block,
            },
            "z_error": {
                "mean": round(float(z.mean()), 6), "p50": round(_p(z, 50), 6),
                "p90": round(_p(z, 90), 6), "max": round(float(z.max()), 6),
                "threshold": 0.0035, "pass_count": int(len(z)) - z_block, "block_count": z_block,
            },
            "yaw_error": {
                "mean": round(float(yaw.mean()), 6), "p50": round(_p(yaw, 50), 6),
                "p90": round(_p(yaw, 90), 6), "max": round(float(yaw.max()), 6),
                "threshold": 0.12434, "pass_count": int(len(yaw)) - yaw_block, "block_count": yaw_block,
            },
            "primary_blocking_axis": (
                "z" if z_block >= xy_block and z_block >= yaw_block
                else "xy" if xy_block >= yaw_block
                else "yaw"
            ),
        }

    # Trend by step range
    step_ranges = [(0, 40), (40, 100), (100, 200), (200, 340)]
    trend = {}
    for lo, hi in step_ranges:
        rows = [r for r in takeover if lo <= r["step"] < hi]
        if rows:
            trend[f"steps_{lo}_{hi}"] = _stats(rows, f"takeover_{lo}_{hi}")

    return {
        "takeover": _stats(takeover, "takeover"),
        "non_takeover": _stats(non_takeover, "non_takeover"),
        "support_blocked": _stats(support, "support_blocked"),
        "takeover_trend": trend,
    }


# ---------------------------------------------------------------------------
# trace consistency
# ---------------------------------------------------------------------------

def _trace_consistency(bounded_dir: Path, shadow_dir: Path) -> dict:
    """Compare bounded vs shadow traces for close_contact and other signals."""
    bounded = _load_traces(bounded_dir)
    shadow = _load_traces(shadow_dir)

    result: dict = {}
    for ep in sorted(bounded.keys()):
        b_rows = bounded.get(ep, [])
        s_rows = shadow.get(ep, [])

        b_contact = Counter(r.get("refiner_contact_state") for r in b_rows)
        s_close_contact = Counter(r.get("shadow_close_contact") for r in s_rows)

        # Also check shadow for refiner_contact_state (if present)
        s_contact = Counter(r.get("refiner_contact_state") for r in s_rows)

        b_has_shadow = "shadow_close_contact" in (b_rows[0] if b_rows else {})
        s_has_refiner = "refiner_contact_state" in (s_rows[0] if s_rows else {})

        result[f"ep{ep:03d}"] = {
            "bounded_steps": len(b_rows),
            "shadow_steps": len(s_rows),
            "bounded_refiner_contact_state": {str(k): v for k, v in b_contact.items()},
            "shadow_close_contact": {str(k): v for k, v in s_close_contact.items()},
            "shadow_refiner_contact_state": {str(k): v for k, v in s_contact.items()},
            "bounded_has_shadow_close_contact": b_has_shadow,
            "shadow_has_refiner_contact_state": s_has_refiner,
        }

    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline close-readiness replay audit")
    parser.add_argument("--handoff-shadow-run-dir", type=Path, required=True,
                        help="Bounded handoff-shadow run dir (with gripper_traces/)")
    parser.add_argument("--shadow-3ep-dir", type=Path, default=None,
                        help="Optional shadow-only 3ep trace dir for consistency comparison")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.handoff_shadow_run_dir
    eval_candidates = sorted(run_dir.glob("*/eval_results.json"))
    if not eval_candidates:
        print("ERROR: no eval_results.json found", flush=True)
        raise SystemExit(1)
    task_dir = eval_candidates[0].parent
    trace_dir = task_dir / "gripper_traces"

    print(f"[replay] trace_dir = {trace_dir}")
    episodes = _load_traces(trace_dir)
    print(f"[replay] loaded {len(episodes)} episodes: {sorted(episodes.keys())}")

    # --- per-step replay ---
    all_replay: list[dict] = []
    per_ep_replay: dict[int, list[dict]] = {}
    for ep, rows in sorted(episodes.items()):
        ep_replay = [_replay_step(r) for r in rows]
        per_ep_replay[ep] = ep_replay
        all_replay.extend(ep_replay)

    # --- per-episode summaries ---
    per_ep_summary = {}
    for ep, rows in per_ep_replay.items():
        takeover_rows = [r for r in rows if r["takeover"]]
        support_rows = [r for r in rows if r["support_blocked"]]
        per_ep_summary[f"ep{ep:03d}"] = {
            "total_steps": len(rows),
            "takeover_steps": len(takeover_rows),
            "support_blocked_steps": len(support_rows),
            "standard_ready_pass_all": sum(1 for r in rows if r["handoff_ready_bool_standard"]),
            "standard_ready_pass_takeover": sum(1 for r in takeover_rows if r["handoff_ready_bool_standard"]),
            "standard_ready_pass_support": sum(1 for r in support_rows if r["handoff_ready_bool_standard"]),
            "metric_pass_all": sum(1 for r in rows if r["metric_pass"]),
            "metric_pass_takeover": sum(1 for r in takeover_rows if r["metric_pass"]),
            "metric_pass_support": sum(1 for r in support_rows if r["metric_pass"]),
            "first_takeover_step": min(r["step"] for r in takeover_rows) if takeover_rows else -1,
        }

    # --- uncertainty sweep ---
    sweep = _sweep(all_replay)

    # --- metric distance audit ---
    metric_audit = _metric_audit(all_replay)

    # --- trace consistency ---
    consistency = None
    if args.shadow_3ep_dir:
        consistency = _trace_consistency(trace_dir, args.shadow_3ep_dir)

    # --- assemble report ---
    report = {
        "audit": "close_readiness_replay",
        "run_dir": str(run_dir),
        "per_episode_summary": per_ep_summary,
        "uncertainty_sweep": sweep,
        "metric_distance_audit": metric_audit,
    }
    if consistency:
        report["trace_consistency"] = consistency

    output_path = args.output or (task_dir / "close_readiness_replay_audit.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # --- print summary ---
    print(f"\n[replay] report -> {output_path}")
    print("\n=== PER-EPISODE SUMMARY ===")
    for ep_label, s in per_ep_summary.items():
        print(f"  {ep_label}: total={s['total_steps']} takeover={s['takeover_steps']} "
              f"support_blocked={s['support_blocked_steps']} "
              f"std_ready_all={s['standard_ready_pass_all']} "
              f"std_ready_takeover={s['standard_ready_pass_takeover']} "
              f"std_ready_support={s['standard_ready_pass_support']} "
              f"metric_pass_takeover={s['metric_pass_takeover']}")

    print("\n=== UNCERTAINTY SWEEP (key rows) ===")
    print(f"{'unc_th':>8} {'prob_th':>8} {'all_rate':>10} {'takeover_rate':>14} {'support_rate':>13} {'late_tk_rate':>12} {'streak_max':>10} {'streak_mean':>11}")
    for s in sweep:
        if s["unc_threshold"] in [0.75, 1.0, 1.5, 2.0, 3.0] and s["prob_threshold"] in [0.5, 0.7]:
            print(f"{s['unc_threshold']:>8} {s['prob_threshold']:>8} {s['ready_pass_rate_all']:>10} {s['ready_pass_rate_takeover']:>14} {s['ready_pass_rate_support_blocked']:>13} {s['ready_pass_rate_late_takeover']:>12} {s['streak_max']:>10} {s['streak_mean']:>11}")

    print("\n=== METRIC DISTANCE AUDIT (takeover) ===")
    ma = metric_audit["takeover"]
    for axis in ["xy_error", "z_error", "yaw_error"]:
        ax = ma[axis]
        print(f"  {axis}: mean={ax['mean']:.6f} p50={ax['p50']:.6f} p90={ax['p90']:.6f} max={ax['max']:.6f} "
              f"thr={ax['threshold']:.6f} pass={ax['pass_count']} block={ax['block_count']}")
    print(f"  primary_blocking_axis: {ma['primary_blocking_axis']}")

    print("\n=== METRIC TREND (takeover by step range) ===")
    for label, stats in metric_audit.get("takeover_trend", {}).items():
        print(f"  {label}: n={stats['n']} xy_mean={stats['xy_error']['mean']:.6f} z_mean={stats['z_error']['mean']:.6f} yaw_mean={stats['yaw_error']['mean']:.6f} primary={stats['primary_blocking_axis']}")

    if consistency:
        print("\n=== TRACE CONSISTENCY ===")
        for ep_label, c in consistency.items():
            print(f"  {ep_label}: bounded_contact={c['bounded_refiner_contact_state']} shadow_close_contact={c['shadow_close_contact']}")


if __name__ == "__main__":
    main()
