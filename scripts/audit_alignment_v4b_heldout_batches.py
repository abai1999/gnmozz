#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _yaw_bucket(act) -> str:
    if not isinstance(act, list) or len(act) < 6:
        return "invalid"
    yaw = abs(float(act[5]))
    if yaw < 0.01:
        return "no_yaw"
    if yaw < 0.05:
        return "small_yaw"
    if yaw < 0.08:
        return "mid_yaw"
    return "large_yaw"


def _safe_rate(num: int, den: int) -> float:
    return float(num / den) if den > 0 else 0.0


def summarize_audit(audit: dict) -> dict:
    worse = audit.get("hard_worse_examples", []) or []
    better = audit.get("best_better_examples", []) or []

    def bucket_counts(rows: list[dict], key: str) -> dict[str, int]:
        out = {"no_yaw": 0, "small_yaw": 0, "mid_yaw": 0, "large_yaw": 0, "invalid": 0}
        for row in rows:
            out[_yaw_bucket(row.get(key))] += 1
        return out

    worse_pred = bucket_counts(worse, "pred_action_local")
    better_pred = bucket_counts(better, "pred_action_local")
    return {
        "changed_frames": int(audit.get("changed_frames", 0)),
        "better_frames": int(audit.get("better_frames", 0)),
        "worse_frames": int(audit.get("worse_frames", 0)),
        "better_rate": float(audit.get("better_rate", 0.0)),
        "worse_rate": float(audit.get("worse_rate", 0.0)),
        "pred_has_yaw_rate_worse_group": float(audit.get("groups", {}).get("worse", {}).get("pred_has_yaw_rate", 0.0)),
        "oracle_has_yaw_rate_worse_group": float(audit.get("groups", {}).get("worse", {}).get("oracle_has_yaw_rate", 0.0)),
        "pred_has_yaw_rate_better_group": float(audit.get("groups", {}).get("better", {}).get("pred_has_yaw_rate", 0.0)),
        "oracle_has_yaw_rate_better_group": float(audit.get("groups", {}).get("better", {}).get("oracle_has_yaw_rate", 0.0)),
        "worse_pred_yaw_buckets": worse_pred,
        "better_pred_yaw_buckets": better_pred,
        "large_yaw_worse_count": int(worse_pred["large_yaw"]),
        "large_yaw_better_count": int(better_pred["large_yaw"]),
        "large_yaw_better_rate": _safe_rate(int(better_pred["large_yaw"]), int(audit.get("better_frames", 0))),
        "large_yaw_worse_rate": _safe_rate(int(worse_pred["large_yaw"]), int(audit.get("worse_frames", 0))),
        "episode_summaries": audit.get("episode_summaries", []),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", action="append", required=True, help="Batch output root(s)")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    batches = []
    all_episode_rows = []
    totals = {
        "changed_frames": 0,
        "better_frames": 0,
        "worse_frames": 0,
        "large_yaw_worse_count": 0,
        "large_yaw_better_count": 0,
        "weighted_regret_delta_sum": 0.0,
    }

    for batch_dir_str in args.batch_dir:
        batch_dir = Path(batch_dir_str)
        summary_path = batch_dir / "alignment_v4b_shadow_summary.json"
        audit_path = batch_dir / "alignment_v4b_shadow_failure_audit.json"
        if not summary_path.exists() or not audit_path.exists():
            raise FileNotFoundError(f"missing summary or audit under {batch_dir}")
        summary = _load_json(summary_path)
        audit = _load_json(audit_path)
        batch = {
            "batch_dir": str(batch_dir),
            "batch_label": batch_dir.name,
            "summary": summary.get("summary", {}),
            "audit": summarize_audit(audit),
        }
        batches.append(batch)
        s = batch["summary"]
        a = batch["audit"]
        totals["changed_frames"] += int(s.get("changed_frames", 0))
        totals["better_frames"] += int(s.get("pred_better_than_baseline_count", 0))
        totals["worse_frames"] += int(s.get("pred_worse_than_baseline_count", 0))
        totals["large_yaw_worse_count"] += int(a.get("large_yaw_worse_count", 0))
        totals["large_yaw_better_count"] += int(a.get("large_yaw_better_count", 0))
        totals["weighted_regret_delta_sum"] += float(s.get("regret_delta_mean_baseline_minus_pred", 0.0)) * int(s.get("changed_frames", 0))
        summary_eps = {
            ep.get("episode_trace"): ep for ep in summary.get("episodes", [])
        }
        for ep in a.get("episode_summaries", []):
            row = dict(ep)
            details = summary_eps.get(ep.get("trace_file"), {})
            row["better_rate"] = float(details.get("pred_better_than_baseline_rate", _safe_rate(int(ep.get("better_frames", 0)), int(ep.get("changed_frames", 0)))))
            row["worse_rate"] = float(details.get("pred_worse_than_baseline_rate", _safe_rate(int(ep.get("worse_frames", 0)), int(ep.get("changed_frames", 0)))))
            row["regret_delta_mean_baseline_minus_pred"] = float(details.get("regret_delta_mean_baseline_minus_pred", 0.0))
            row["batch_label"] = batch_dir.name
            all_episode_rows.append(row)

    overall = {
        "batches": len(batches),
        "changed_frames": totals["changed_frames"],
        "better_frames": totals["better_frames"],
        "worse_frames": totals["worse_frames"],
        "better_rate": _safe_rate(totals["better_frames"], totals["changed_frames"]),
        "worse_rate": _safe_rate(totals["worse_frames"], totals["changed_frames"]),
        "regret_delta_mean_baseline_minus_pred": (
            totals["weighted_regret_delta_sum"] / totals["changed_frames"]
            if totals["changed_frames"] > 0
            else 0.0
        ),
        "large_yaw_better_count": totals["large_yaw_better_count"],
        "large_yaw_worse_count": totals["large_yaw_worse_count"],
    }

    out = {
        "overall": overall,
        "batches": batches,
        "episodes": all_episode_rows,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
