#!/usr/bin/env python3
"""Compare ablation runs A/B/C vs strong alignment baseline."""
from __future__ import annotations

import argparse, json, re
from collections import Counter
from pathlib import Path
import numpy as np


def _task_dir(run_dir: Path) -> Path:
    for p in sorted(run_dir.glob("*/eval_results.json")):
        return p.parent
    raise FileNotFoundError(f"no eval_results.json under {run_dir}")


def _load_traces(trace_dir: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = {}
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        m = re.search(r"ep(\d+)", path.name)
        if not m: continue
        ep = int(m.group(1))
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        episodes[ep] = sorted(rows, key=lambda r: int(r.get("step", 0)))
    return episodes


def _extract_metrics(run_dir: Path, label: str) -> dict:
    td = _task_dir(run_dir)
    with (td / "eval_results.json").open() as f:
        ed = json.load(f)
    traces = _load_traces(td / "gripper_traces")

    all_rows = []
    for ep, rows in sorted(traces.items()):
        for r in rows:
            r["_ep"] = ep
        all_rows.extend(rows)

    # --- basic metrics ---
    metrics = {
        "label": label,
        "success_rate": float(ed.get("success_rate", -1)),
        "avg_episode_length": float(ed.get("avg_episode_length", -1)),
        "invalid_action_count": int(ed.get("invalid_action_count", -1)),
        "workspace_violation_count": int(ed.get("workspace_violation_count", -1)),
        "workspace_violation_mean": float(ed.get("workspace_violation_mean", -1)),
    }

    # --- per-episode stage stats ---
    for ss in ed.get("stage_stats", []):
        ep_idx = ss.get("episode_index", -1)
        if ep_idx == 5:
            metrics["subgoal_progress"] = float(ss.get("subgoal_progress", -1))
            metrics["max_phase_reached"] = int(ss.get("max_phase_reached", -1))
            metrics["first_close_step"] = int(ss.get("first_close_step", -1))
            metrics["final_substage"] = str(ss.get("final_substage", ss.get("substage", "?")))

    # --- alignment metrics ---
    metrics["planner_close_intent_rate"] = float(ed.get("planner_close_intent_rate", -1))
    metrics["alignment_takeover_count"] = int(ed.get("alignment_takeover_count", -1))
    metrics["clip_hit_rate"] = float(ed.get("clip_hit_rate", -1))
    metrics["alpha_mean"] = float(ed.get("alpha_mean", -1))
    metrics["correction_count"] = int(ed.get("correction_count", -1))
    metrics["alignment_correction_count"] = int(ed.get("alignment_correction_count", -1))
    metrics["alignment_gate_block_count"] = int(ed.get("alignment_gate_block_count", -1))

    # --- near-zone gate ---
    nz_gate_pass = sum(1 for r in all_rows if r.get("refiner_alignment_near_zone_gate_pass"))
    nz_gate_eval = sum(1 for r in all_rows if r.get("refiner_alignment_near_zone_gate_enabled"))
    nz_enabled = any(r.get("refiner_alignment_near_zone_gate_enabled") for r in all_rows)
    metrics["near_zone_gate_enabled"] = nz_enabled
    metrics["near_zone_gate_pass_count"] = nz_gate_pass
    metrics["near_zone_gate_eval_count"] = nz_gate_eval
    metrics["near_zone_gate_pass_rate"] = round(nz_gate_pass / max(nz_gate_eval, 1), 4)

    # Block reason histogram
    block_reasons = Counter()
    for r in all_rows:
        br = r.get("refiner_alignment_near_zone_block_reason", "disabled")
        block_reasons[str(br)] += 1
    metrics["near_zone_block_reason_hist"] = dict(block_reasons)

    # --- correction direction (from handoff_aux) ---
    tk_rows = [r for r in all_rows if r.get("refiner_alignment_takeover_active")]
    dx_vals = []
    dy_vals = []
    dz_vals = []
    dyaw_vals = []
    for r in tk_rows:
        ha = r.get("handoff_aux_provider") or {}
        for key, store in [("pred_residual_dx", dx_vals), ("pred_residual_dy", dy_vals),
                           ("pred_residual_dz", dz_vals), ("pred_residual_dyaw", dyaw_vals)]:
            v = ha.get(key)
            if v is not None and np.isfinite(float(v)):
                store.append(float(v))

    for label_axis, vals in [("dx", dx_vals), ("dy", dy_vals), ("dz", dz_vals), ("dyaw", dyaw_vals)]:
        if vals:
            a = np.array(vals)
            metrics[f"{label_axis}_mean"] = round(float(a.mean()), 6)
            metrics[f"{label_axis}_positive_rate"] = round(float((a > 1e-8).sum() / len(a)), 4)
        else:
            metrics[f"{label_axis}_mean"] = 0.0
            metrics[f"{label_axis}_positive_rate"] = 0.0

    # --- candidate histogram ---
    cand_hist = Counter()
    for r in tk_rows:
        idx = r.get("refiner_last_scorer_candidate_index", -1)
        if idx >= 0:
            cand_hist[str(int(idx))] += 1
    metrics["selected_candidate_histogram"] = dict(cand_hist.most_common(5))
    metrics["dominant_candidate"] = cand_hist.most_common(1)[0][0] if cand_hist else "none"

    # --- handoff metrics for z overshoot ---
    z_vals = []
    for r in all_rows:
        hm = r.get("handoff_metrics_provider") or {}
        z = hm.get("abs_z_error")
        if z is not None and np.isfinite(float(z)):
            z_vals.append(float(z))
    if z_vals:
        z_arr = np.array(z_vals)
        n = len(z_arr)
        metrics["z_error_min"] = round(float(z_arr.min()), 4)
        metrics["z_error_mean"] = round(float(z_arr.mean()), 4)
        # z overshoot: z at end vs z at best (min)
        last_quarter = z_arr[max(0, n - n//4):]
        metrics["z_error_last_quarter_mean"] = round(float(last_quarter.mean()), 4)
        metrics["z_overshoot"] = round(float(last_quarter.mean() - z_arr.min()), 4)
    else:
        metrics["z_error_min"] = -1
        metrics["z_error_mean"] = -1
        metrics["z_overshoot"] = -1

    # --- yaw improvement ---
    yaw_vals = []
    for r in all_rows:
        hm = r.get("handoff_metrics_provider") or {}
        y = hm.get("yaw_error")
        if y is not None and np.isfinite(float(y)):
            yaw_vals.append(float(y))
    if yaw_vals:
        yaw_arr = np.array(yaw_vals)
        n = len(yaw_arr)
        first_q = yaw_arr[:max(1, n//4)]
        last_q = yaw_arr[max(0, n - n//4):]
        metrics["yaw_error_first_quarter_mean"] = round(float(first_q.mean()), 4)
        metrics["yaw_error_last_quarter_mean"] = round(float(last_q.mean()), 4)
        metrics["yaw_delta"] = round(float(last_q.mean() - first_q.mean()), 4)
        metrics["yaw_improved"] = metrics["yaw_delta"] < 0

    # --- alignment gate open rate ---
    gate_open = sum(1 for r in all_rows if r.get("refiner_alignment_gate_open"))
    gate_total = sum(1 for r in all_rows if r.get("refiner_alignment_gate_open") is not None)
    metrics["alignment_gate_open_rate"] = round(gate_open / max(gate_total, 1), 4)

    # --- per-episode breakdown ---
    per_ep = {}
    for ep in sorted(traces.keys()):
        rows = traces[ep]
        tk = sum(1 for r in rows if r.get("refiner_alignment_takeover_active"))
        nz_pass = sum(1 for r in rows if r.get("refiner_alignment_near_zone_gate_pass"))
        close_intent = sum(1 for r in rows if r.get("refiner_alignment_planner_close_intent"))
        for ss in ed.get("stage_stats", []):
            if ss.get("episode_index") == ep:
                per_ep[f"ep{ep:03d}"] = {
                    "takeover_count": tk,
                    "near_zone_pass_count": nz_pass,
                    "planner_close_intent_count": close_intent,
                    "subgoal_progress": float(ss.get("subgoal_progress", -1)),
                    "max_phase_reached": int(ss.get("max_phase_reached", -1)),
                    "first_close_step": int(ss.get("first_close_step", -1)),
                    "workspace_violation_count": int(ss.get("workspace_violation_count", -1)),
                }
                break

    metrics["per_episode"] = per_ep
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong-alignment-dir", type=Path, required=True)
    parser.add_argument("--ablation-a-dir", type=Path, required=True)
    parser.add_argument("--ablation-b-dir", type=Path, required=True)
    parser.add_argument("--ablation-c-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    strong = _extract_metrics(args.strong_alignment_dir, "strong_alignment_1.0")
    a = _extract_metrics(args.ablation_a_dir, "A_weak_baseline_0.25")
    b = _extract_metrics(args.ablation_b_dir, "B_nearzone_weak_0.25")
    c = _extract_metrics(args.ablation_c_dir, "C_nearzone_medium_0.5")

    all_metrics = [strong, a, b, c]

    # Comparison table
    keys = [
        "success_rate", "workspace_violation_count", "subgoal_progress",
        "max_phase_reached", "first_close_step", "planner_close_intent_rate",
        "alignment_takeover_count", "alignment_gate_open_rate",
        "near_zone_gate_enabled", "near_zone_gate_pass_rate",
        "clip_hit_rate", "alpha_mean", "correction_count",
        "dx_positive_rate", "dy_positive_rate", "dx_mean", "dy_mean",
        "z_error_min", "z_overshoot", "yaw_delta",
    ]

    print("=" * 100)
    print("ABLATION COMPARISON TABLE")
    print("=" * 100)

    header = f"{'Metric':<35}"
    for m in all_metrics:
        header += f" {m['label']:<28}"
    print(header)
    print("-" * 100)

    for key in keys:
        row = f"{key:<35}"
        for m in all_metrics:
            v = m.get(key, "N/A")
            if isinstance(v, float):
                row += f" {v:<28.4f}"
            else:
                row += f" {str(v):<28}"
        print(row)

    print("\n=== CANDIDATE HISTOGRAMS ===")
    for m in all_metrics:
        print(f"  {m['label']}: dominant={m['dominant_candidate']} hist={m['selected_candidate_histogram']}")

    print("\n=== NEAR-ZONE GATE ===")
    for m in all_metrics:
        if m['near_zone_gate_enabled']:
            print(f"  {m['label']}: pass_rate={m['near_zone_gate_pass_rate']} block_reasons={m['near_zone_block_reason_hist']}")

    print("\n=== PER-EPISODE BREAKDOWN ===")
    for m in all_metrics:
        print(f"  {m['label']}:")
        for ep_label, ep_data in m.get("per_episode", {}).items():
            print(f"    {ep_label}: tk={ep_data['takeover_count']} nz_pass={ep_data['near_zone_pass_count']} "
                  f"close_intent={ep_data['planner_close_intent_count']} "
                  f"subgoal_progress={ep_data['subgoal_progress']:.3f} "
                  f"phase={ep_data['max_phase_reached']} "
                  f"first_close={ep_data['first_close_step']} "
                  f"ws_viol={ep_data['workspace_violation_count']}")

    # Write report
    report = {"audit": "ablation_comparison", "runs": all_metrics}
    output_path = args.output or (_task_dir(args.ablation_a_dir) / "ablation_comparison_report.json")
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[compare] report -> {output_path}")


if __name__ == "__main__":
    main()
