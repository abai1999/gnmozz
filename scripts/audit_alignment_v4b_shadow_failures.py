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
    if isinstance(v, list):
        return v
    return None


def _candidate_action(candidate_actions, idx: int):
    if candidate_actions is None or idx < 0 or idx >= len(candidate_actions):
        return None
    act = candidate_actions[idx]
    if not isinstance(act, list):
        return None
    return [float(x) for x in act[:6]]


def _has_yaw(act, thresh: float) -> bool:
    return bool(act is not None and len(act) >= 6 and abs(float(act[5])) > float(thresh))


def _score_from_cost(cost):
    return -float(cost) if math.isfinite(float(cost)) else math.nan


def categorize(delta: float, eps: float = 1e-6) -> str:
    if not math.isfinite(delta):
        return "invalid"
    if delta > eps:
        return "better"
    if delta < -eps:
        return "worse"
    return "tie"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--keep_yaw_abs", type=float, default=0.035)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    files = _find_trace_files(trace_dir)
    if not files:
        raise RuntimeError(f"no trace files found under {trace_dir}")

    changed_rows = []
    episode_summaries = []
    bucket_rows: dict[str, list[dict]] = defaultdict(list)

    for trace_path in files:
        rows = []
        with trace_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        ep_counter = Counter()
        for row in rows:
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
            regret_delta = _safe_float(row.get("b2_candidate_shadow_regret_delta"))
            cls = categorize(regret_delta)
            pred_act = _candidate_action(cand_actions, pred_idx)
            baseline_act = _candidate_action(cand_actions, baseline_idx)
            best_act = _candidate_action(cand_actions, best_idx)
            has_yaw_candidate = False
            if cand_actions:
                has_yaw_candidate = any(_has_yaw(act, args.keep_yaw_abs) for act in cand_actions if isinstance(act, list))
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
                "baseline_score": _score_from_cost(baseline_cost),
                "pred_score": _score_from_cost(pred_cost),
                "oracle_score": _score_from_cost(best_cost),
                "regret_delta_mean_baseline_minus_pred": regret_delta,
                "class": cls,
                "pred_action_local": pred_act,
                "baseline_action_local": baseline_act,
                "oracle_action_local": best_act,
                "pred_has_yaw": _has_yaw(pred_act, args.keep_yaw_abs),
                "baseline_has_yaw": _has_yaw(baseline_act, args.keep_yaw_abs),
                "oracle_has_yaw": _has_yaw(best_act, args.keep_yaw_abs),
                "has_yaw_candidate": has_yaw_candidate,
                "pred_mode": _safe_int(row.get("b2_candidate_shadow_mode")),
                "pred_mode_confidence": _safe_float(row.get("b2_candidate_shadow_mode_confidence")),
                "pred_mode_margin": _safe_float(row.get("b2_candidate_shadow_mode_margin")),
                "yaw_needed": bool(row.get("b2_candidate_shadow_yaw_needed", False)),
                "yaw_keep": bool(row.get("b2_candidate_shadow_yaw_keep", False)),
                "teacher_ready": bool(row.get("b2_candidate_shadow_teacher_ready", False)),
                "xy_block": bool(row.get("b2_candidate_shadow_xy_block", False)),
                "close_neighborhood": bool(row.get("b2_candidate_shadow_close_neighborhood", False)),
                "nearish_runtime": bool(row.get("b2_candidate_shadow_nearish_runtime", False)),
                "runtime_scope_size": _safe_int(row.get("b2_candidate_shadow_runtime_scope_size"), 0),
                "small_yaw_scope_size": _safe_int(row.get("b2_candidate_shadow_small_yaw_scope_size"), 0),
                "large_yaw_scope_size": _safe_int(row.get("b2_candidate_shadow_large_yaw_scope_size"), 0),
            }
            changed_rows.append(rec)
            bucket_rows[cls].append(rec)
            ep_counter[cls] += 1

        episode_summaries.append(
            {
                "trace_file": trace_path.name,
                "changed_frames": int(sum(ep_counter.values())),
                "better_frames": int(ep_counter["better"]),
                "worse_frames": int(ep_counter["worse"]),
                "tie_frames": int(ep_counter["tie"]),
            }
        )

    def summarize_group(rows: list[dict]) -> dict:
        if not rows:
            return {
                "frames": 0,
                "pred_has_yaw_rate": 0.0,
                "oracle_has_yaw_rate": 0.0,
                "baseline_has_yaw_rate": 0.0,
                "yaw_needed_rate": 0.0,
                "nearish_runtime_rate": 0.0,
                "close_neighborhood_rate": 0.0,
                "mean_mode_confidence": math.nan,
                "mean_mode_margin": math.nan,
                "mean_regret_delta": math.nan,
            }
        return {
            "frames": len(rows),
            "pred_has_yaw_rate": float(np.mean([float(r["pred_has_yaw"]) for r in rows])),
            "oracle_has_yaw_rate": float(np.mean([float(r["oracle_has_yaw"]) for r in rows])),
            "baseline_has_yaw_rate": float(np.mean([float(r["baseline_has_yaw"]) for r in rows])),
            "yaw_needed_rate": float(np.mean([float(r["yaw_needed"]) for r in rows])),
            "nearish_runtime_rate": float(np.mean([float(r["nearish_runtime"]) for r in rows])),
            "close_neighborhood_rate": float(np.mean([float(r["close_neighborhood"]) for r in rows])),
            "mean_mode_confidence": float(np.mean([r["pred_mode_confidence"] for r in rows])),
            "mean_mode_margin": float(np.mean([r["pred_mode_margin"] for r in rows])),
            "mean_regret_delta": float(np.mean([r["regret_delta_mean_baseline_minus_pred"] for r in rows])),
        }

    summary = {
        "trace_dir": str(trace_dir),
        "num_trace_files": len(files),
        "changed_frames": len(changed_rows),
        "better_frames": len(bucket_rows["better"]),
        "worse_frames": len(bucket_rows["worse"]),
        "tie_frames": len(bucket_rows["tie"]),
        "better_rate": float(len(bucket_rows["better"]) / len(changed_rows)) if changed_rows else 0.0,
        "worse_rate": float(len(bucket_rows["worse"]) / len(changed_rows)) if changed_rows else 0.0,
        "tie_rate": float(len(bucket_rows["tie"]) / len(changed_rows)) if changed_rows else 0.0,
        "groups": {
            "better": summarize_group(bucket_rows["better"]),
            "worse": summarize_group(bucket_rows["worse"]),
            "tie": summarize_group(bucket_rows["tie"]),
        },
        "episode_summaries": episode_summaries,
        "hard_worse_examples": sorted(
            bucket_rows["worse"],
            key=lambda r: float(r["regret_delta_mean_baseline_minus_pred"]),
        )[:20],
        "best_better_examples": sorted(
            bucket_rows["better"],
            key=lambda r: -float(r["regret_delta_mean_baseline_minus_pred"]),
        )[:20],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
