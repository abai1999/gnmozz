#!/usr/bin/env python3
"""Comprehensive v2 vs final_full runtime shadow comparison.

Reads the 3ep source-trace traces and produces per-bucket metrics,
delta source analysis, and a verdict on v2 runtime readiness.
"""
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def _load_traces(trace_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_traces(args.trace_dir)
    print(f"[audit] loaded {len(rows)} rows")

    # --- 1. Basic health checks ---
    scorer_ran = sum(1 for r in rows if r.get("refiner_last_scorer_candidate_index", -1) >= 0)
    v2_preds = sum(1 for r in rows if r.get("refiner_v2_selected_candidate_index", -1) >= 0)
    v2_gate = sum(1 for r in rows if r.get("refiner_v2_gate_pass"))

    print(f"  scorer_ran: {scorer_ran}/{len(rows)}")
    print(f"  v2_predictions: {v2_preds}")
    print(f"  v2_gate_pass: {v2_gate}")

    # --- 2. Delta source analysis ---
    delta_sources = Counter()
    delta_norms = defaultdict(list)
    for r in rows:
        src = str(r.get("refiner_v2_delta_source", "missing"))
        dn = r.get("refiner_v2_delta_norm", 0) or 0
        delta_sources[src] += 1
        delta_norms[src].append(float(dn))

    delta_stats = {}
    for src, vals in delta_norms.items():
        a = np.array(vals)
        delta_stats[src] = {
            "n": len(a), "mean": round(float(a.mean()), 6), "p50": round(float(np.percentile(a, 50)), 6),
            "p90": round(float(np.percentile(a, 90)), 6), "max": round(float(a.max()), 6),
        }

    # At gate-pass steps
    gate_rows = [r for r in rows if r.get("refiner_v2_gate_pass")]
    gate_delta_src = Counter(str(r.get("refiner_v2_delta_source", "?")) for r in gate_rows)

    # --- 3. Per-bucket comparison ---
    buckets_order = ["micro_contact_refine", "near_alignment", "mid_approach_assist", "far_coarse_approach"]

    def _hm(r, key, default=np.nan):
        hm = r.get("handoff_metrics_provider") or {}
        v = hm.get(key)
        return float(v) if v is not None and np.isfinite(float(v)) else default

    # Classify each step by actual handoff error
    bucket_rows = defaultdict(list)
    for r in rows:
        z = _hm(r, "abs_z_error", 999)
        xy = _hm(r, "xy_error", 999)
        yaw = _hm(r, "yaw_error", 999)
        if z > 900:  # no handoff metrics
            bucket_rows["unknown"].append(r)
        elif xy < 0.015 and z < 0.03 and yaw < 0.12:
            bucket_rows["micro_contact_refine"].append(r)
        elif xy < 0.05 and z < 0.10 and yaw < 0.25:
            bucket_rows["near_alignment"].append(r)
        elif xy < 0.12 and z < 0.25:
            bucket_rows["mid_approach_assist"].append(r)
        else:
            bucket_rows["far_coarse_approach"].append(r)

    per_bucket = {}
    for b in buckets_order + ["unknown"]:
        br = bucket_rows.get(b, [])
        if not br:
            continue
        n = len(br)

        # Scorer stats
        ff_idx = Counter()
        v2_idx = Counter()
        v2_agree = 0
        v2_present = 0
        for r in br:
            fi = r.get("refiner_last_scorer_candidate_index", -1)
            vi = r.get("refiner_v2_selected_candidate_index", -1)
            if fi >= 0: ff_idx[str(fi)] += 1
            if vi >= 0:
                v2_idx[str(vi)] += 1
                v2_present += 1
                if fi >= 0 and vi == fi:
                    v2_agree += 1

        per_bucket[b] = {
            "n": n,
            "scorer_ran": sum(1 for r in br if r.get("refiner_last_scorer_candidate_index", -1) >= 0),
            "v2_predictions": v2_present,
            "v2_gate_pass": sum(1 for r in br if r.get("refiner_v2_gate_pass")),
            "v2_ff_agree": round(v2_agree / max(v2_present, 1), 4) if v2_present > 0 else 0,
            "ff_candidate_hist": dict(ff_idx.most_common(5)),
            "v2_candidate_hist": dict(v2_idx.most_common(5)),
        }

    # --- 4. Overall comparison ---
    ff_all = Counter()
    v2_all = Counter()
    for r in rows:
        fi = r.get("refiner_last_scorer_candidate_index", -1)
        vi = r.get("refiner_v2_selected_candidate_index", -1)
        if fi >= 0: ff_all[str(fi)] += 1
        if vi >= 0: v2_all[str(vi)] += 1

    # --- 5. Zone state analysis ---
    zone_states = Counter(str(r.get("refiner_zone_state", "?")) for r in rows)
    tk_active = sum(1 for r in rows if r.get("refiner_alignment_takeover_active"))
    nz_blocked = sum(1 for r in rows if r.get("refiner_alignment_near_zone_gate_pass") is False
                     and r.get("refiner_alignment_near_zone_gate_enabled"))

    # --- 6. Delta source at v2 predictions ---
    v2_pred_rows = [r for r in rows if r.get("refiner_v2_selected_candidate_index", -1) >= 0]
    v2_pred_delta_src = Counter(str(r.get("refiner_v2_delta_source", "?")) for r in v2_pred_rows)

    # --- Assemble report ---
    report = {
        "audit": "v2_runtime_shadow_final",
        "trace_dir": str(args.trace_dir),
        "health": {
            "total_rows": len(rows),
            "scorer_ran": scorer_ran,
            "v2_predictions": v2_preds,
            "v2_gate_pass": v2_gate,
            "takeover_active": tk_active,
            "near_zone_blocked": nz_blocked,
            "zone_states": dict(zone_states),
        },
        "delta_source": {
            "overall_distribution": dict(delta_sources),
            "at_gate_pass": dict(gate_delta_src),
            "at_v2_predictions": dict(v2_pred_delta_src),
            "stats_by_source": delta_stats,
        },
        "overall": {
            "ff_candidate_histogram": dict(ff_all.most_common()),
            "v2_candidate_histogram": dict(v2_all.most_common()),
            "v2_ff_agree_count": sum(1 for r in rows
                if r.get("refiner_v2_selected_candidate_index", -1) >= 0
                and r.get("refiner_last_scorer_candidate_index", -1) >= 0
                and r.get("refiner_v2_selected_candidate_index") == r.get("refiner_last_scorer_candidate_index")),
        },
        "per_bucket": per_bucket,
        "verdict": {
            "scorer_restored": scorer_ran > 500,
            "v2_gets_real_delta": v2_gate > 0 and gate_delta_src.get("runtime_motion_target_pose", 0) > 0,
            "v2_not_template_collapsed": len(v2_all) > 1,
            "near_zone_gate_blocks_apply_only": tk_active == 0 and scorer_ran > 500,
            "ready_for_near_zone_shadow_assist": bool(
                scorer_ran > 500 and v2_gate > 0 and len(v2_all) > 1 and tk_active == 0
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n=== HEALTH ===")
    for k, v in report["health"].items():
        print(f"  {k}: {v}")

    print(f"\n=== DELTA SOURCE ===")
    print(f"  overall: {dict(delta_sources)}")
    print(f"  at gate_pass: {dict(gate_delta_src)}")
    for src, s in delta_stats.items():
        print(f"  {src}: mean={s['mean']:.4f} p90={s['p90']:.4f}")

    print(f"\n=== OVERALL ===")
    print(f"  ff candidates: {dict(ff_all.most_common(5))}")
    print(f"  v2 candidates: {dict(v2_all.most_common(5))}")

    print(f"\n=== PER BUCKET ===")
    for b in buckets_order:
        s = per_bucket.get(b, {})
        if not s: continue
        print(f"  {b}: n={s['n']} scorer_ran={s['scorer_ran']} v2_pred={s['v2_predictions']} "
              f"v2_gate={s['v2_gate_pass']} v2_ff_agree={s['v2_ff_agree']} "
              f"ff_hist={s['ff_candidate_hist']} v2_hist={s['v2_candidate_hist']}")

    print(f"\n=== VERDICT ===")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")

    print(f"\n[audit] report -> {args.output}")


if __name__ == "__main__":
    main()
