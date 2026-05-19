#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from statistics import median


def as_bool(row: dict, key: str) -> bool:
    return bool(row.get(key, False))


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return default


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value


def find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "gripper_traces").is_dir():
        path = path / "gripper_traces"
    files = sorted(path.glob("*_gripper_trace.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    return files


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def finite_values(rows: list[dict], key: str) -> list[float]:
    vals = []
    for row in rows:
        value = as_float(row, key)
        if math.isfinite(value):
            vals.append(value)
    return vals


def first_finite_values(rows: list[dict], keys: tuple[str, ...]) -> list[float]:
    vals = []
    for row in rows:
        for key in keys:
            value = as_float(row, key)
            if math.isfinite(value):
                vals.append(value)
                break
    return vals


def top_counts(rows: list[dict], key: str, mask_key: str | None = None, limit: int = 8) -> list[dict]:
    counter: collections.Counter[int] = collections.Counter()
    for row in rows:
        if mask_key is not None and not as_bool(row, mask_key):
            continue
        value = as_int(row, key, -1)
        if value >= 0:
            counter[value] += 1
    return [{"value": int(v), "count": int(c)} for v, c in counter.most_common(limit)]


def consecutive_switches(rows: list[dict], key: str, mask_key: str | None = None) -> int:
    switches = 0
    prev = None
    for row in rows:
        if mask_key is not None and not as_bool(row, mask_key):
            prev = None
            continue
        value = as_int(row, key, -1)
        if value < 0:
            prev = None
            continue
        if prev is not None and value != prev:
            switches += 1
        prev = value
    return switches


def summarize_episode(path: Path, rows: list[dict]) -> dict:
    gate = [r for r in rows if as_bool(r, "b1_group_shadow_gate_open")]
    close = [r for r in rows if as_bool(r, "b1_group_shadow_close_neighborhood")]
    apply = [r for r in gate if as_bool(r, "b1_apply_gate_apply")]
    apply_close = [r for r in apply if as_bool(r, "b1_group_shadow_close_neighborhood")]
    valid_teacher = [r for r in rows if as_bool(r, "b1_group_shadow_teacher_group_valid")]
    valid_close = [r for r in close if as_bool(r, "b1_group_shadow_teacher_group_valid")]
    changed = [r for r in gate if as_bool(r, "b1_group_shadow_changed")]
    close_changed = [r for r in close if as_bool(r, "b1_group_shadow_close_group_changed")]
    teacher_disagree = [r for r in valid_teacher if as_bool(r, "b1_group_shadow_teacher_disagree")]
    close_teacher_disagree = [r for r in valid_close if as_bool(r, "b1_group_shadow_teacher_disagree")]
    margins = finite_values(gate, "b1_group_shadow_margin")
    cost_valid = [r for r in valid_teacher if math.isfinite(as_float(r, "b1_group_shadow_regret_delta"))]
    cost_valid_close = [r for r in valid_close if math.isfinite(as_float(r, "b1_group_shadow_regret_delta"))]
    cost_valid_apply = [r for r in apply if math.isfinite(as_float(r, "b1_group_shadow_regret_delta"))]
    cost_valid_apply_close = [r for r in apply_close if math.isfinite(as_float(r, "b1_group_shadow_regret_delta"))]
    cost_improve = [r for r in cost_valid if as_float(r, "b1_group_shadow_regret_delta") > 1e-6]
    cost_worse = [r for r in cost_valid if as_float(r, "b1_group_shadow_regret_delta") < -1e-6]
    cost_tie = [r for r in cost_valid if abs(as_float(r, "b1_group_shadow_regret_delta")) <= 1e-6]
    cost_improve_close = [r for r in cost_valid_close if as_float(r, "b1_group_shadow_regret_delta") > 1e-6]
    cost_worse_close = [r for r in cost_valid_close if as_float(r, "b1_group_shadow_regret_delta") < -1e-6]
    cost_improve_apply = [r for r in cost_valid_apply if as_float(r, "b1_group_shadow_regret_delta") > 1e-6]
    cost_worse_apply = [r for r in cost_valid_apply if as_float(r, "b1_group_shadow_regret_delta") < -1e-6]
    cost_improve_apply_close = [r for r in cost_valid_apply_close if as_float(r, "b1_group_shadow_regret_delta") > 1e-6]
    cost_worse_apply_close = [r for r in cost_valid_apply_close if as_float(r, "b1_group_shadow_regret_delta") < -1e-6]
    regret_delta_vals = [as_float(r, "b1_group_shadow_regret_delta") for r in cost_valid]
    regret_delta_close_vals = [as_float(r, "b1_group_shadow_regret_delta") for r in cost_valid_close]
    regret_delta_apply_vals = [as_float(r, "b1_group_shadow_regret_delta") for r in cost_valid_apply]
    regret_delta_apply_close_vals = [as_float(r, "b1_group_shadow_regret_delta") for r in cost_valid_apply_close]
    pred_regret_vals = [as_float(r, "b1_group_shadow_pred_group_regret") for r in cost_valid]
    baseline_regret_vals = [as_float(r, "b1_group_shadow_baseline_group_regret") for r in cost_valid]

    yaw_fields = {
        "teacher_yaw": first_finite_values(rows, ("teacher_truth_handoff_metric_yaw_error", "teacher_truth_basin_yaw")),
        "runtime_yaw": first_finite_values(
            rows,
            (
                "runtime_handoff_metric_yaw_error",
                "handoff_metric_yaw_error",
                "alignment_runtime_basin_yaw",
                "refiner_current_basin_yaw_runtime",
            ),
        ),
    }
    yaw_thresholds = first_finite_values(
        rows,
        (
            "teacher_truth_handoff_release_threshold_yaw_error",
            "runtime_handoff_release_threshold_yaw_error",
            "handoff_release_threshold_yaw_error",
        ),
    )
    yaw_threshold = median(yaw_thresholds) if yaw_thresholds else 0.12434
    yaw_threshold_source = "trace" if yaw_thresholds else "default_0.12434"

    return {
        "episode_trace": path.name,
        "frames": len(rows),
        "gate_open_frames": len(gate),
        "close_neighborhood_frames": len(close),
        "apply_gate_frames": len(apply),
        "apply_gate_close_frames": len(apply_close),
        "changed_frames": len(changed),
        "close_changed_frames": len(close_changed),
        "teacher_group_valid_frames": len(valid_teacher),
        "teacher_group_valid_close_frames": len(valid_close),
        "teacher_disagree_valid_frames": len(teacher_disagree),
        "teacher_disagree_close_valid_frames": len(close_teacher_disagree),
        "pred_group_switches_gate": consecutive_switches(rows, "b1_group_shadow_pred_group", "b1_group_shadow_gate_open"),
        "pred_group_switches_close": consecutive_switches(rows, "b1_group_shadow_pred_group", "b1_group_shadow_close_neighborhood"),
        "change_rate_gate": rate(len(changed), len(gate)),
        "change_rate_close": rate(len(close_changed), len(close)),
        "teacher_valid_rate_gate": rate(len(valid_teacher), len(gate)),
        "teacher_valid_rate_close": rate(len(valid_close), len(close)),
        "teacher_disagree_rate_valid": rate(len(teacher_disagree), len(valid_teacher)),
        "teacher_disagree_rate_close_valid": rate(len(close_teacher_disagree), len(valid_close)),
        "teacher_cost_valid_frames": len(cost_valid),
        "teacher_cost_valid_close_frames": len(cost_valid_close),
        "teacher_cost_valid_apply_frames": len(cost_valid_apply),
        "teacher_cost_valid_apply_close_frames": len(cost_valid_apply_close),
        "pred_better_than_baseline_count": len(cost_improve),
        "pred_worse_than_baseline_count": len(cost_worse),
        "pred_tie_baseline_count": len(cost_tie),
        "pred_better_than_baseline_close_count": len(cost_improve_close),
        "pred_worse_than_baseline_close_count": len(cost_worse_close),
        "apply_gate_better_than_baseline_count": len(cost_improve_apply),
        "apply_gate_worse_than_baseline_count": len(cost_worse_apply),
        "apply_gate_better_than_baseline_close_count": len(cost_improve_apply_close),
        "apply_gate_worse_than_baseline_close_count": len(cost_worse_apply_close),
        "pred_better_than_baseline_rate": rate(len(cost_improve), len(cost_valid)),
        "pred_worse_than_baseline_rate": rate(len(cost_worse), len(cost_valid)),
        "pred_better_than_baseline_close_rate": rate(len(cost_improve_close), len(cost_valid_close)),
        "pred_worse_than_baseline_close_rate": rate(len(cost_worse_close), len(cost_valid_close)),
        "apply_gate_better_than_baseline_rate": rate(len(cost_improve_apply), len(cost_valid_apply)),
        "apply_gate_worse_than_baseline_rate": rate(len(cost_worse_apply), len(cost_valid_apply)),
        "apply_gate_better_than_baseline_close_rate": rate(len(cost_improve_apply_close), len(cost_valid_apply_close)),
        "apply_gate_worse_than_baseline_close_rate": rate(len(cost_worse_apply_close), len(cost_valid_apply_close)),
        "regret_delta_mean_baseline_minus_pred": float(sum(regret_delta_vals) / len(regret_delta_vals))
        if regret_delta_vals
        else math.nan,
        "regret_delta_p50_baseline_minus_pred": float(median(regret_delta_vals)) if regret_delta_vals else math.nan,
        "regret_delta_close_mean_baseline_minus_pred": float(
            sum(regret_delta_close_vals) / len(regret_delta_close_vals)
        )
        if regret_delta_close_vals
        else math.nan,
        "apply_gate_regret_delta_mean_baseline_minus_pred": float(
            sum(regret_delta_apply_vals) / len(regret_delta_apply_vals)
        )
        if regret_delta_apply_vals
        else math.nan,
        "apply_gate_regret_delta_close_mean_baseline_minus_pred": float(
            sum(regret_delta_apply_close_vals) / len(regret_delta_apply_close_vals)
        )
        if regret_delta_apply_close_vals
        else math.nan,
        "pred_group_regret_mean": float(sum(pred_regret_vals) / len(pred_regret_vals)) if pred_regret_vals else math.nan,
        "baseline_group_regret_mean": float(sum(baseline_regret_vals) / len(baseline_regret_vals))
        if baseline_regret_vals
        else math.nan,
        "pred_group_top_counts": top_counts(rows, "b1_group_shadow_pred_group", "b1_group_shadow_gate_open"),
        "baseline_group_top_counts": top_counts(rows, "b1_group_shadow_baseline_group", "b1_group_shadow_gate_open"),
        "teacher_group_top_counts": top_counts(rows, "b1_group_shadow_teacher_group", "b1_group_shadow_teacher_group_valid"),
        "margin_mean": float(sum(margins) / len(margins)) if margins else math.nan,
        "margin_p50": float(median(margins)) if margins else math.nan,
        "yaw_threshold": float(yaw_threshold),
        "yaw_threshold_source": yaw_threshold_source,
        "min_teacher_yaw": float(min(yaw_fields["teacher_yaw"])) if yaw_fields["teacher_yaw"] else math.nan,
        "min_runtime_yaw": float(min(yaw_fields["runtime_yaw"])) if yaw_fields["runtime_yaw"] else math.nan,
        "teacher_yaw_in_band_rate": rate(sum(v <= yaw_threshold for v in yaw_fields["teacher_yaw"]), len(yaw_fields["teacher_yaw"]))
        if math.isfinite(yaw_threshold)
        else math.nan,
        "runtime_yaw_in_band_rate": rate(sum(v <= yaw_threshold for v in yaw_fields["runtime_yaw"]), len(yaw_fields["runtime_yaw"]))
        if math.isfinite(yaw_threshold)
        else math.nan,
    }


def combine(episodes: list[dict]) -> dict:
    totals = {
        "episodes": len(episodes),
        "frames": sum(e["frames"] for e in episodes),
        "gate_open_frames": sum(e["gate_open_frames"] for e in episodes),
        "close_neighborhood_frames": sum(e["close_neighborhood_frames"] for e in episodes),
        "apply_gate_frames": sum(e["apply_gate_frames"] for e in episodes),
        "apply_gate_close_frames": sum(e["apply_gate_close_frames"] for e in episodes),
        "changed_frames": sum(e["changed_frames"] for e in episodes),
        "close_changed_frames": sum(e["close_changed_frames"] for e in episodes),
        "teacher_group_valid_frames": sum(e["teacher_group_valid_frames"] for e in episodes),
        "teacher_group_valid_close_frames": sum(e["teacher_group_valid_close_frames"] for e in episodes),
        "teacher_disagree_valid_frames": sum(e["teacher_disagree_valid_frames"] for e in episodes),
        "teacher_disagree_close_valid_frames": sum(e["teacher_disagree_close_valid_frames"] for e in episodes),
        "teacher_cost_valid_frames": sum(e["teacher_cost_valid_frames"] for e in episodes),
        "teacher_cost_valid_close_frames": sum(e["teacher_cost_valid_close_frames"] for e in episodes),
        "teacher_cost_valid_apply_frames": sum(e["teacher_cost_valid_apply_frames"] for e in episodes),
        "teacher_cost_valid_apply_close_frames": sum(e["teacher_cost_valid_apply_close_frames"] for e in episodes),
        "pred_better_than_baseline_count": sum(e["pred_better_than_baseline_count"] for e in episodes),
        "pred_worse_than_baseline_count": sum(e["pred_worse_than_baseline_count"] for e in episodes),
        "pred_tie_baseline_count": sum(e["pred_tie_baseline_count"] for e in episodes),
        "pred_better_than_baseline_close_count": sum(e["pred_better_than_baseline_close_count"] for e in episodes),
        "pred_worse_than_baseline_close_count": sum(e["pred_worse_than_baseline_close_count"] for e in episodes),
        "apply_gate_better_than_baseline_count": sum(e["apply_gate_better_than_baseline_count"] for e in episodes),
        "apply_gate_worse_than_baseline_count": sum(e["apply_gate_worse_than_baseline_count"] for e in episodes),
        "apply_gate_better_than_baseline_close_count": sum(e["apply_gate_better_than_baseline_close_count"] for e in episodes),
        "apply_gate_worse_than_baseline_close_count": sum(e["apply_gate_worse_than_baseline_close_count"] for e in episodes),
        "pred_group_switches_gate": sum(e["pred_group_switches_gate"] for e in episodes),
        "pred_group_switches_close": sum(e["pred_group_switches_close"] for e in episodes),
    }
    margins = [e["margin_p50"] for e in episodes if math.isfinite(e["margin_p50"])]
    pred_counts: collections.Counter[int] = collections.Counter()
    base_counts: collections.Counter[int] = collections.Counter()
    teacher_counts: collections.Counter[int] = collections.Counter()
    for episode in episodes:
        for item in episode.get("pred_group_top_counts", []):
            pred_counts[int(item["value"])] += int(item["count"])
        for item in episode.get("baseline_group_top_counts", []):
            base_counts[int(item["value"])] += int(item["count"])
        for item in episode.get("teacher_group_top_counts", []):
            teacher_counts[int(item["value"])] += int(item["count"])
    totals.update(
        {
            "change_rate_gate": rate(totals["changed_frames"], totals["gate_open_frames"]),
            "change_rate_close": rate(totals["close_changed_frames"], totals["close_neighborhood_frames"]),
            "teacher_valid_rate_gate": rate(totals["teacher_group_valid_frames"], totals["gate_open_frames"]),
            "teacher_valid_rate_close": rate(totals["teacher_group_valid_close_frames"], totals["close_neighborhood_frames"]),
            "teacher_disagree_rate_valid": rate(totals["teacher_disagree_valid_frames"], totals["teacher_group_valid_frames"]),
            "teacher_disagree_rate_close_valid": rate(
                totals["teacher_disagree_close_valid_frames"], totals["teacher_group_valid_close_frames"]
            ),
            "pred_better_than_baseline_rate": rate(
                totals["pred_better_than_baseline_count"], totals["teacher_cost_valid_frames"]
            ),
            "pred_worse_than_baseline_rate": rate(
                totals["pred_worse_than_baseline_count"], totals["teacher_cost_valid_frames"]
            ),
            "pred_tie_baseline_rate": rate(totals["pred_tie_baseline_count"], totals["teacher_cost_valid_frames"]),
            "pred_better_than_baseline_close_rate": rate(
                totals["pred_better_than_baseline_close_count"],
                totals["teacher_cost_valid_close_frames"],
            ),
            "pred_worse_than_baseline_close_rate": rate(
                totals["pred_worse_than_baseline_close_count"],
                totals["teacher_cost_valid_close_frames"],
            ),
            "apply_gate_better_than_baseline_rate": rate(
                totals["apply_gate_better_than_baseline_count"],
                totals["teacher_cost_valid_apply_frames"],
            ),
            "apply_gate_worse_than_baseline_rate": rate(
                totals["apply_gate_worse_than_baseline_count"],
                totals["teacher_cost_valid_apply_frames"],
            ),
            "apply_gate_better_than_baseline_close_rate": rate(
                totals["apply_gate_better_than_baseline_close_count"],
                totals["teacher_cost_valid_apply_close_frames"],
            ),
            "apply_gate_worse_than_baseline_close_rate": rate(
                totals["apply_gate_worse_than_baseline_close_count"],
                totals["teacher_cost_valid_apply_close_frames"],
            ),
            "regret_delta_mean_baseline_minus_pred": rate(
                sum(
                    e["regret_delta_mean_baseline_minus_pred"] * e["teacher_cost_valid_frames"]
                    for e in episodes
                    if math.isfinite(e["regret_delta_mean_baseline_minus_pred"])
                ),
                totals["teacher_cost_valid_frames"],
            ),
            "regret_delta_close_mean_baseline_minus_pred": rate(
                sum(
                    e["regret_delta_close_mean_baseline_minus_pred"] * e["teacher_cost_valid_close_frames"]
                    for e in episodes
                    if math.isfinite(e["regret_delta_close_mean_baseline_minus_pred"])
                ),
                totals["teacher_cost_valid_close_frames"],
            ),
            "apply_gate_regret_delta_mean_baseline_minus_pred": rate(
                sum(
                    e["apply_gate_regret_delta_mean_baseline_minus_pred"] * e["teacher_cost_valid_apply_frames"]
                    for e in episodes
                    if math.isfinite(e["apply_gate_regret_delta_mean_baseline_minus_pred"])
                ),
                totals["teacher_cost_valid_apply_frames"],
            ),
            "apply_gate_regret_delta_close_mean_baseline_minus_pred": rate(
                sum(
                    e["apply_gate_regret_delta_close_mean_baseline_minus_pred"] * e["teacher_cost_valid_apply_close_frames"]
                    for e in episodes
                    if math.isfinite(e["apply_gate_regret_delta_close_mean_baseline_minus_pred"])
                ),
                totals["teacher_cost_valid_apply_close_frames"],
            ),
            "pred_switch_rate_gate": rate(totals["pred_group_switches_gate"], totals["gate_open_frames"]),
            "pred_switch_rate_close": rate(totals["pred_group_switches_close"], totals["close_neighborhood_frames"]),
            "episode_margin_p50_median": float(median(margins)) if margins else math.nan,
            "pred_group_top_counts": [{"value": int(v), "count": int(c)} for v, c in pred_counts.most_common(8)],
            "baseline_group_top_counts": [{"value": int(v), "count": int(c)} for v, c in base_counts.most_common(8)],
            "teacher_group_top_counts": [{"value": int(v), "count": int(c)} for v, c in teacher_counts.most_common(8)],
        }
    )
    return totals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True, help="Evaluation run dir, gripper_traces dir, or one jsonl trace file.")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    trace_root = Path(args.trace_dir)
    files = find_trace_files(trace_root)
    if not files:
        raise SystemExit(f"No jsonl trace files found under {trace_root}")

    episodes = [summarize_episode(path, load_jsonl(path)) for path in files]
    report = {
        "trace_dir": str(trace_root),
        "trace_files": [str(p) for p in files],
        "summary": combine(episodes),
        "episodes": episodes,
        "notes": {
            "teacher_disagreement_only_counts_valid": True,
            "teacher_group_is_diagnostic_approximation": True,
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
