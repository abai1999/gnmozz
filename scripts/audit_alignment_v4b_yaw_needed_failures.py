#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "gripper_traces").is_dir():
        path = path / "gripper_traces"
    files = sorted(path.glob("*_gripper_trace.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    return files


def _safe_float(v, default=math.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=-1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_list(v):
    return v if isinstance(v, list) else None


def _candidate_action(candidate_actions, idx: int):
    if candidate_actions is None or idx < 0 or idx >= len(candidate_actions):
        return None
    act = candidate_actions[idx]
    if not isinstance(act, list):
        return None
    return [float(x) for x in act[:6]]


def _yaw_bucket(act, keep_abs: float, small_abs: float, large_abs: float) -> str:
    if act is None or len(act) < 6:
        return "invalid"
    yaw = abs(float(act[5]))
    if yaw < float(keep_abs):
        return "no_yaw"
    if yaw < float(small_abs):
        return "small_yaw"
    if yaw < float(large_abs):
        return "medium_yaw"
    return "large_yaw"


def _sign(v: float) -> int:
    if not math.isfinite(v) or abs(v) <= 1e-9:
        return 0
    return 1 if v > 0 else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.02)
    ap.add_argument("--small_yaw_abs", type=float, default=0.05)
    ap.add_argument("--large_yaw_abs", type=float, default=0.09)
    ap.add_argument("--mode_keep_margin", type=float, default=0.05)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    files = _find_trace_files(trace_dir)
    if not files:
        raise RuntimeError(f"no trace files found under {trace_dir}")

    rows: list[dict] = []
    episode_rows: dict[int, list[dict]] = defaultdict(list)
    type_counts: Counter[str] = Counter()

    for trace_path in files:
        with trace_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not bool(row.get("b2_candidate_shadow_gate_open", False)):
                    continue
                if not bool(row.get("b2_candidate_shadow_changed", False)):
                    continue
                cand_actions = _safe_list(row.get("b2_candidate_shadow_candidate_actions_local"))
                pred_idx = _safe_int(row.get("b2_candidate_shadow_pred_index"))
                baseline_idx = _safe_int(row.get("b2_candidate_shadow_baseline_index"))
                best_idx = _safe_int(row.get("b2_candidate_shadow_best_index"))
                pred_cost = _safe_float(row.get("b2_candidate_shadow_pred_cost"))
                baseline_cost = _safe_float(row.get("b2_candidate_shadow_baseline_cost"))
                best_cost = _safe_float(row.get("b2_candidate_shadow_best_cost"))
                pred_act = _candidate_action(cand_actions, pred_idx)
                baseline_act = _candidate_action(cand_actions, baseline_idx)
                best_act = _candidate_action(cand_actions, best_idx)
                pred_yaw = abs(float(pred_act[5])) if pred_act is not None else math.nan
                baseline_yaw = abs(float(baseline_act[5])) if baseline_act is not None else math.nan
                best_yaw = abs(float(best_act[5])) if best_act is not None else math.nan
                pred_bucket = _yaw_bucket(pred_act, args.keep_yaw_abs, args.small_yaw_abs, args.large_yaw_abs)
                baseline_bucket = _yaw_bucket(baseline_act, args.keep_yaw_abs, args.small_yaw_abs, args.large_yaw_abs)
                best_bucket = _yaw_bucket(best_act, args.keep_yaw_abs, args.small_yaw_abs, args.large_yaw_abs)
                yaw_needed = bool(row.get("b2_candidate_shadow_yaw_needed", False))
                yaw_keep = bool(row.get("b2_candidate_shadow_yaw_keep", False))
                teacher_ready = bool(row.get("b2_candidate_shadow_teacher_ready", False))
                xy_block = bool(row.get("b2_candidate_shadow_xy_block", False))
                close_neighborhood = bool(row.get("b2_candidate_shadow_close_neighborhood", False))
                nearish_runtime = bool(row.get("b2_candidate_shadow_nearish_runtime", False))
                oracle_gap = baseline_cost - best_cost if math.isfinite(baseline_cost) and math.isfinite(best_cost) else math.nan
                pred_better = math.isfinite(pred_cost) and math.isfinite(baseline_cost) and pred_cost < baseline_cost - 1e-6
                pred_worse = math.isfinite(pred_cost) and math.isfinite(baseline_cost) and pred_cost > baseline_cost + 1e-6

                failure_type = "none"
                if math.isfinite(oracle_gap) and oracle_gap <= float(args.mode_keep_margin):
                    failure_type = "baseline_preserve_opportunity"
                if yaw_needed and pred_bucket in {"no_yaw", "small_yaw"} and best_bucket in {"small_yaw", "medium_yaw", "large_yaw"}:
                    failure_type = "yaw_needed_missing"
                elif yaw_needed and pred_bucket == "large_yaw" and best_bucket in {"no_yaw", "small_yaw"}:
                    failure_type = "yaw_needed_large_overuse"
                elif yaw_needed and pred_bucket in {"medium_yaw", "large_yaw"} and best_bucket in {"medium_yaw", "large_yaw"} and pred_act is not None and best_act is not None and _sign(pred_act[5]) != _sign(best_act[5]) and abs(best_act[5]) >= float(args.small_yaw_abs):
                    failure_type = "yaw_needed_wrong_sign"
                elif (not yaw_needed) and pred_bucket in {"medium_yaw", "large_yaw"}:
                    failure_type = "yaw_not_needed_but_selected"
                elif pred_bucket == "large_yaw" and best_bucket in {"no_yaw", "small_yaw"}:
                    failure_type = "large_yaw_overuse"
                elif pred_bucket in {"medium_yaw", "large_yaw"} and best_bucket in {"no_yaw", "small_yaw"}:
                    failure_type = "small_vs_large_yaw"
                elif pred_bucket in {"medium_yaw", "large_yaw"} and best_bucket in {"medium_yaw", "large_yaw"} and pred_act is not None and best_act is not None and _sign(pred_act[5]) != _sign(best_act[5]) and abs(best_act[5]) >= float(args.small_yaw_abs):
                    failure_type = "wrong_yaw_sign"
                elif pred_bucket in {"medium_yaw", "large_yaw"} and best_bucket in {"no_yaw", "small_yaw"} and xy_block:
                    failure_type = "xy_over_yaw"
                elif pred_bucket in {"medium_yaw", "large_yaw"} and best_bucket in {"no_yaw", "small_yaw"} and not xy_block:
                    failure_type = "z_over_yaw"

                rec = {
                    "trace_file": trace_path.name,
                    "episode_index": _safe_int(row.get("episode_index")),
                    "step_index": _safe_int(row.get("step_index")),
                    "baseline_index": baseline_idx,
                    "pred_index": pred_idx,
                    "oracle_index": best_idx,
                    "baseline_cost": baseline_cost,
                    "pred_cost": pred_cost,
                    "oracle_cost": best_cost,
                    "baseline_score": -baseline_cost if math.isfinite(baseline_cost) else math.nan,
                    "pred_score": -pred_cost if math.isfinite(pred_cost) else math.nan,
                    "oracle_score": -best_cost if math.isfinite(best_cost) else math.nan,
                    "oracle_gap": oracle_gap,
                    "pred_action_local": pred_act,
                    "baseline_action_local": baseline_act,
                    "oracle_action_local": best_act,
                    "pred_yaw_bucket": pred_bucket,
                    "baseline_yaw_bucket": baseline_bucket,
                    "oracle_yaw_bucket": best_bucket,
                    "pred_has_yaw": bool(pred_bucket != "no_yaw"),
                    "baseline_has_yaw": bool(baseline_bucket != "no_yaw"),
                    "oracle_has_yaw": bool(best_bucket != "no_yaw"),
                    "yaw_needed": yaw_needed,
                    "yaw_keep": yaw_keep,
                    "teacher_ready": teacher_ready,
                    "xy_block": xy_block,
                    "close_neighborhood": close_neighborhood,
                    "nearish_runtime": nearish_runtime,
                    "failure_type": failure_type,
                    "pred_better_than_baseline": pred_better,
                    "pred_worse_than_baseline": pred_worse,
                }
                rows.append(rec)
                episode_rows[int(rec["episode_index"])].append(rec)
                type_counts[failure_type] += 1

    def summarize_rows(rs: list[dict]) -> dict:
        if not rs:
            return {
                "frames": 0,
                "better_rate": 0.0,
                "worse_rate": 0.0,
                "mean_oracle_gap": math.nan,
                "yaw_needed_rate": 0.0,
                "pred_large_yaw_rate": 0.0,
                "oracle_large_yaw_rate": 0.0,
            }
        return {
            "frames": len(rs),
            "better_rate": float(np.mean([bool(r["pred_better_than_baseline"]) for r in rs])),
            "worse_rate": float(np.mean([bool(r["pred_worse_than_baseline"]) for r in rs])),
            "mean_oracle_gap": float(np.mean([float(r["oracle_gap"]) for r in rs])),
            "yaw_needed_rate": float(np.mean([bool(r["yaw_needed"]) for r in rs])),
            "pred_large_yaw_rate": float(np.mean([r["pred_yaw_bucket"] == "large_yaw" for r in rs])),
            "oracle_large_yaw_rate": float(np.mean([r["oracle_yaw_bucket"] == "large_yaw" for r in rs])),
        }

    episode_summary = [
        {
            "episode_index": int(ep),
            "frames": len(rs),
            "better_rate": summarize_rows(rs)["better_rate"],
            "worse_rate": summarize_rows(rs)["worse_rate"],
            "mean_oracle_gap": summarize_rows(rs)["mean_oracle_gap"],
            "yaw_needed_rate": summarize_rows(rs)["yaw_needed_rate"],
            "pred_large_yaw_rate": summarize_rows(rs)["pred_large_yaw_rate"],
            "oracle_large_yaw_rate": summarize_rows(rs)["oracle_large_yaw_rate"],
            "failure_types": dict(Counter(r["failure_type"] for r in rs)),
        }
        for ep, rs in sorted(episode_rows.items())
    ]

    summary = {
        "trace_dir": str(trace_dir),
        "num_trace_files": len(files),
        "changed_frames": len(rows),
        "better_frames": int(sum(bool(r["pred_better_than_baseline"]) for r in rows)),
        "worse_frames": int(sum(bool(r["pred_worse_than_baseline"]) for r in rows)),
        "better_rate": float(np.mean([bool(r["pred_better_than_baseline"]) for r in rows])) if rows else 0.0,
        "worse_rate": float(np.mean([bool(r["pred_worse_than_baseline"]) for r in rows])) if rows else 0.0,
        "oracle_gap_mean": float(np.mean([float(r["oracle_gap"]) for r in rows])) if rows else math.nan,
        "failure_type_counts": dict(type_counts),
        "episode_summary": episode_summary,
        "yaw_needed_rows": int(sum(bool(r["yaw_needed"]) for r in rows)),
        "yaw_needed_better_rate": float(np.mean([bool(r["pred_better_than_baseline"]) for r in rows if r["yaw_needed"]])) if any(bool(r["yaw_needed"]) for r in rows) else 0.0,
        "yaw_needed_worse_rate": float(np.mean([bool(r["pred_worse_than_baseline"]) for r in rows if r["yaw_needed"]])) if any(bool(r["yaw_needed"]) for r in rows) else 0.0,
        "large_yaw_pred_rate": float(np.mean([r["pred_yaw_bucket"] == "large_yaw" for r in rows])) if rows else 0.0,
        "large_yaw_oracle_rate": float(np.mean([r["oracle_yaw_bucket"] == "large_yaw" for r in rows])) if rows else 0.0,
        "baseline_preserve_opportunity_rate": float(np.mean([float(r["oracle_gap"]) <= float(args.mode_keep_margin) for r in rows])) if rows else 0.0,
        "hard_examples": sorted(rows, key=lambda r: float(r["oracle_gap"]))[:20],
        "hard_worse_examples": [r for r in sorted(rows, key=lambda r: float(r["oracle_gap"])) if bool(r["pred_worse_than_baseline"])][:20],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
