#!/usr/bin/env python3
"""Audit v2 runtime delta source: which source is v2 actually using at runtime."""
from __future__ import annotations

import argparse, json
from collections import Counter
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _load_traces(args.trace_dir)
    if not rows:
        print("ERROR: no trace rows found"); return

    # --- Delta source distribution ---
    delta_sources = Counter()
    gate_pass_sources = Counter()
    v2_pred_sources = Counter()
    gripper_missing = 0
    delta_zero = 0
    total_v2_gate = 0
    total_v2_pred = 0

    for r in rows:
        src = str(r.get("refiner_v2_delta_source", "missing"))
        delta_sources[src] += 1

        gate = r.get("refiner_v2_gate_pass", False)
        if gate:
            total_v2_gate += 1
            gate_pass_sources[src] += 1

        vi = r.get("refiner_v2_selected_candidate_index", -1)
        if vi is not None and vi >= 0:
            total_v2_pred += 1
            v2_pred_sources[src] += 1

        if not r.get("refiner_v2_gripper_pose_present", False):
            gripper_missing += 1

        dn = r.get("refiner_v2_delta_norm", -1)
        if dn is not None and float(dn) < 1e-10:
            delta_zero += 1

    print(f"Total rows: {len(rows)}")
    print(f"v2 gate pass: {total_v2_gate}")
    print(f"v2 predictions: {total_v2_pred}")
    print(f"\n=== Delta source distribution (all steps) ===")
    for src, cnt in delta_sources.most_common():
        print(f"  {src}: {cnt} ({100*cnt/len(rows):.1f}%)")
    print(f"\n=== Delta source at gate_pass steps ===")
    for src, cnt in gate_pass_sources.most_common():
        print(f"  {src}: {cnt} ({100*cnt/max(total_v2_gate,1):.1f}%)")
    print(f"\n=== Delta source at v2 prediction steps ===")
    for src, cnt in v2_pred_sources.most_common():
        print(f"  {src}: {cnt} ({100*cnt/max(total_v2_pred,1):.1f}%)")

    print(f"\n=== Source trace flags ===")
    print(f"  gripper_pose missing: {gripper_missing}/{len(rows)}")
    print(f"  delta_norm zero: {delta_zero}/{len(rows)}")
    mtp = sum(1 for r in rows if r.get("refiner_v2_motion_target_pose_present"))
    mtd = sum(1 for r in rows if r.get("refiner_v2_motion_target_delta_present"))
    cdb = sum(1 for r in rows if r.get("refiner_v2_current_delta_basin_present"))
    print(f"  motion_target_pose_present: {mtp}/{len(rows)}")
    print(f"  motion_target_delta_present: {mtd}/{len(rows)}")
    print(f"  current_delta_basin_present: {cdb}/{len(rows)}")

    # Delta norm distribution by source
    norms_by_source = {}
    for r in rows:
        src = str(r.get("refiner_v2_delta_source", "missing"))
        dn = r.get("refiner_v2_delta_norm", 0) or 0
        norms_by_source.setdefault(src, []).append(float(dn))

    print(f"\n=== Delta norm stats by source ===")
    for src, vals in sorted(norms_by_source.items()):
        a = np.array(vals)
        print(f"  {src}: n={len(a)} mean={a.mean():.4f} p50={np.percentile(a,50):.4f} max={a.max():.4f}")

    # V2 candidate distribution by delta source
    v2_idx_by_source = {}
    for r in rows:
        src = str(r.get("refiner_v2_delta_source", "missing"))
        vi = r.get("refiner_v2_selected_candidate_index", -1)
        if vi is not None and vi >= 0:
            v2_idx_by_source.setdefault(src, Counter())[str(vi)] += 1

    print(f"\n=== v2 candidate by delta source ===")
    for src, c in sorted(v2_idx_by_source.items()):
        print(f"  {src}: {dict(sorted(c.items()))}")

    # Write report
    report = {
        "audit": "v2_runtime_delta_source_trace",
        "total_rows": len(rows),
        "v2_gate_pass": total_v2_gate,
        "v2_predictions": total_v2_pred,
        "delta_source_distribution": dict(delta_sources),
        "delta_source_at_gate_pass": dict(gate_pass_sources),
        "delta_source_at_prediction": dict(v2_pred_sources),
        "gripper_pose_missing_count": gripper_missing,
        "delta_zero_count": delta_zero,
        "motion_target_pose_present": mtp,
        "motion_target_delta_present": mtd,
        "current_delta_basin_present": cdb,
        "v2_candidate_by_delta_source": {k: dict(v) for k, v in v2_idx_by_source.items()},
    }

    output_path = args.output or (args.trace_dir.parent / "v2_delta_source_trace_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[audit] report -> {output_path}")


if __name__ == "__main__":
    main()
