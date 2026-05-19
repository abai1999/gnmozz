#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


NUM_GROUPS = 37


BASE_FEATURES = [
    "b1_group_shadow_margin",
    "alignment_runtime_basin_xy",
    "alignment_runtime_basin_z",
    "alignment_runtime_basin_yaw",
    "alignment_runtime_basin_distance",
    "obs_gripper_open",
    "refiner_alignment_planner_close_intent",
    "refiner_current_close_veto_ready",
    "refiner_current_close_veto_blocked",
    "handoff_ready_pred",
    "refiner_substage_id",
    "phase_before",
]

V2_EXTRA_FEATURES = [
    "b1_group_shadow_close_neighborhood",
    "refiner_current_alignment_support_satisfied",
    "refiner_current_alignment_support_inner_satisfied",
    "refiner_current_alignment_support_outer_satisfied",
    "refiner_current_alignment_refine_band_satisfied",
    "refiner_current_alignment_takeover_band_satisfied",
    "refiner_alignment_close_requirement_satisfied",
    "refiner_current_close_latch_remaining",
    "refiner_last_handoff_yaw_priority_active",
    "handoff_aux_pred_xy_norm",
    "handoff_aux_pred_abs_z_norm",
    "handoff_aux_pred_yaw_norm",
    "handoff_aux_pred_band_index",
    "handoff_aux_pred_ready_prob",
    "handoff_aux_pred_uncertainty",
    "runtime_handoff_xy_norm",
    "runtime_handoff_abs_z_norm",
    "runtime_handoff_yaw_norm",
]


def find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "gripper_traces").is_dir():
        path = path / "gripper_traces"
    files = sorted(path.glob("*_gripper_trace.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    return files


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if isinstance(value, bool):
        return float(value)
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def as_int(row: dict, key: str, default: int = -1) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return default


def one_hot(index: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def nested_float(row: dict, key: str, subkey: str, default: float = 0.0) -> float:
    value = row.get(key)
    if not isinstance(value, dict):
        return default
    return as_float(value, subkey, default)


def feature_dict(row: dict) -> dict[str, float]:
    pred_group = as_int(row, "b1_group_shadow_pred_group")
    baseline_group = as_int(row, "b1_group_shadow_baseline_group")
    out: dict[str, float] = {key: as_float(row, key) for key in BASE_FEATURES}
    out["b1_group_shadow_close_neighborhood"] = as_float(row, "b1_group_shadow_close_neighborhood")
    out["refiner_current_alignment_support_satisfied"] = as_float(row, "refiner_current_alignment_support_satisfied")
    out["refiner_current_alignment_support_inner_satisfied"] = as_float(row, "refiner_current_alignment_support_inner_satisfied")
    out["refiner_current_alignment_support_outer_satisfied"] = as_float(row, "refiner_current_alignment_support_outer_satisfied")
    out["refiner_current_alignment_refine_band_satisfied"] = as_float(row, "refiner_current_alignment_refine_band_satisfied")
    out["refiner_current_alignment_takeover_band_satisfied"] = as_float(row, "refiner_current_alignment_takeover_band_satisfied")
    out["refiner_alignment_close_requirement_satisfied"] = as_float(row, "refiner_alignment_close_requirement_satisfied")
    out["refiner_current_close_latch_remaining"] = as_float(row, "refiner_current_close_latch_remaining")
    out["refiner_last_handoff_yaw_priority_active"] = as_float(row, "refiner_last_handoff_yaw_priority_active")
    out["handoff_aux_pred_xy_norm"] = nested_float(row, "handoff_aux_provider", "pred_xy_norm")
    out["handoff_aux_pred_abs_z_norm"] = nested_float(row, "handoff_aux_provider", "pred_abs_z_norm")
    out["handoff_aux_pred_yaw_norm"] = nested_float(row, "handoff_aux_provider", "pred_yaw_norm")
    out["handoff_aux_pred_band_index"] = nested_float(row, "handoff_aux_provider", "pred_band_index")
    out["handoff_aux_pred_ready_prob"] = nested_float(row, "handoff_aux_provider", "pred_ready_prob")
    out["handoff_aux_pred_uncertainty"] = nested_float(row, "handoff_aux_provider", "pred_uncertainty")

    runtime_xy = nested_float(row, "handoff_metrics_provider", "xy_error")
    runtime_z = nested_float(row, "handoff_metrics_provider", "abs_z_error")
    runtime_yaw = nested_float(row, "handoff_metrics_provider", "yaw_error")
    rel_xy = nested_float(row, "handoff_release_metric_thresholds_provider", "xy_error", 0.0085)
    rel_z = nested_float(row, "handoff_release_metric_thresholds_provider", "abs_z_error", 0.0035)
    rel_yaw = nested_float(row, "handoff_release_metric_thresholds_provider", "yaw_error", 0.12434040009975433)
    out["runtime_handoff_xy_norm"] = float(runtime_xy / max(rel_xy, 1e-6))
    out["runtime_handoff_abs_z_norm"] = float(runtime_z / max(rel_z, 1e-6))
    out["runtime_handoff_yaw_norm"] = float(runtime_yaw / max(rel_yaw, 1e-6))

    out["pred_equals_baseline"] = float(pred_group == baseline_group)
    for i, v in enumerate(one_hot(pred_group, NUM_GROUPS)):
        out[f"pred_group_{i}"] = v
    for i, v in enumerate(one_hot(baseline_group, NUM_GROUPS)):
        out[f"baseline_group_{i}"] = v
    return out


def default_feature_names(version: str = "v1") -> list[str]:
    feature_names = list(BASE_FEATURES)
    if version == "v2":
        feature_names += list(V2_EXTRA_FEATURES)
    feature_names += ["pred_equals_baseline"]
    feature_names += [f"pred_group_{i}" for i in range(NUM_GROUPS)]
    feature_names += [f"baseline_group_{i}" for i in range(NUM_GROUPS)]
    return feature_names


def row_features(row: dict, feature_names: list[str] | None = None) -> list[float]:
    fmap = feature_dict(row)
    names = feature_names if feature_names is not None else default_feature_names("v2")
    return [float(fmap.get(name, 0.0)) for name in names]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--output_meta", default=None)
    ap.add_argument("--gate_mode", choices=["close_only", "all_gate"], default="close_only")
    ap.add_argument("--positive_eps", type=float, default=1e-6)
    ap.add_argument("--feature_version", choices=["v1", "v2"], default="v2")
    args = ap.parse_args()

    trace_files = find_trace_files(Path(args.trace_dir))
    if not trace_files:
        raise SystemExit(f"No trace files found under {args.trace_dir}")

    xs, ys, deltas, episodes, steps = [], [], [], [], []
    pred_groups, baseline_groups = [], []
    for trace_file in trace_files:
        ep_name = trace_file.stem.replace("_gripper_trace", "")
        try:
            ep_idx = int(ep_name.replace("ep", ""))
        except Exception:
            ep_idx = len(episodes)
        with trace_file.open() as f:
            for step, line in enumerate(f):
                row = json.loads(line)
                if not row.get("b1_group_shadow_gate_open", False):
                    continue
                if args.gate_mode == "close_only" and not row.get("b1_group_shadow_close_neighborhood", False):
                    continue
                regret_delta = as_float(row, "b1_group_shadow_regret_delta", math.nan)
                if not math.isfinite(regret_delta):
                    continue
                feature_names = default_feature_names(args.feature_version)
                xs.append(row_features(row, feature_names))
                ys.append(1.0 if regret_delta > args.positive_eps else 0.0)
                deltas.append(regret_delta)
                episodes.append(ep_idx)
                steps.append(step)
                pred_groups.append(as_int(row, "b1_group_shadow_pred_group"))
                baseline_groups.append(as_int(row, "b1_group_shadow_baseline_group"))

    if not xs:
        raise SystemExit("No valid B1 apply-gate rows found")

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    regret_delta = np.asarray(deltas, dtype=np.float32)
    episode_index = np.asarray(episodes, dtype=np.int64)
    step_index = np.asarray(steps, dtype=np.int64)
    pred_group = np.asarray(pred_groups, dtype=np.int64)
    baseline_group = np.asarray(baseline_groups, dtype=np.int64)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=x,
        labels=y,
        regret_delta=regret_delta,
        episode_index=episode_index,
        step_index=step_index,
        pred_group=pred_group,
        baseline_group=baseline_group,
        feature_names=np.asarray(feature_names),
    )

    meta = {
        "trace_dir": str(args.trace_dir),
        "output_npz": str(out),
        "gate_mode": args.gate_mode,
        "feature_version": args.feature_version,
        "rows": int(x.shape[0]),
        "positive": int(np.sum(y > 0.5)),
        "negative": int(np.sum(y <= 0.5)),
        "episodes": sorted(int(v) for v in np.unique(episode_index)),
        "feature_dim": int(x.shape[1]),
        "feature_names": feature_names,
    }
    meta_path = Path(args.output_meta) if args.output_meta else out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
