#!/usr/bin/env python3
"""Compare v3 runtime shadow contract against offline teacher distribution.

This audit is intended to answer three questions:
1. What delta/source does v3 actually consume in runtime?
2. How far are those runtime deltas from the offline teacher distribution?
3. Why do predicted post-errors look worse than the current runtime errors?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _load_trace_rows(trace_dir: Path) -> list[dict]:
    rows: list[dict] = []
    paths = sorted(trace_dir.rglob("*_gripper_trace.jsonl"))
    if not paths:
        paths = sorted(trace_dir.rglob("*.jsonl"))
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_trace_path"] = str(path)
                rows.append(row)
    return rows


def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _vec6(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 6 or not np.all(np.isfinite(arr[:6])):
        return None
    return arr[:6].astype(np.float32)


def _delta_norm_summary(deltas: list[np.ndarray]) -> dict:
    if not deltas:
        return {
            "rows": 0,
            "pos_norm": _stats(np.array([], dtype=np.float32)),
            "yaw_abs": _stats(np.array([], dtype=np.float32)),
        }
    mat = np.stack(deltas, axis=0).astype(np.float32)
    return {
        "rows": int(mat.shape[0]),
        "pos_norm": _stats(np.linalg.norm(mat[:, :3], axis=-1)),
        "xy_norm": _stats(np.linalg.norm(mat[:, :2], axis=-1)),
        "z_abs": _stats(np.abs(mat[:, 2])),
        "yaw_abs": _stats(np.abs(mat[:, 5])),
    }


def _safe_arr(d: np.ndarray, key: str, fallback=None):
    if key not in d.files:
        if fallback is None:
            raise KeyError(key)
        return fallback
    return np.asarray(d[key])


def _source_summary(rows: list[dict]) -> dict:
    active = [r for r in rows if bool(r.get("refiner_alignment_v3_shadow_active", False))]
    source_hist = Counter(str(r.get("refiner_alignment_v3_shadow_source", "none")) for r in rows)
    active_source_hist = Counter(str(r.get("refiner_alignment_v3_shadow_source", "none")) for r in active)
    runtime_delta_source_hist = Counter(str(r.get("runtime_motion_target_delta_source", "none")) for r in rows)
    active_runtime_delta_source_hist = Counter(str(r.get("runtime_motion_target_delta_source", "none")) for r in active)
    runtime_deltas = [_vec6(r.get("refiner_motion_target_delta_local")) for r in active]
    runtime_deltas = [d for d in runtime_deltas if d is not None]

    def _axis_stats(subset: list[dict]) -> dict:
        cur_xy = [r.get("refiner_alignment_v3_shadow_cur_xy") for r in subset]
        cur_z = [r.get("refiner_alignment_v3_shadow_cur_z") for r in subset]
        cur_yaw = [r.get("refiner_alignment_v3_shadow_cur_yaw") for r in subset]
        post_xy = [r.get("refiner_alignment_v3_shadow_post_xy") for r in subset]
        post_z = [r.get("refiner_alignment_v3_shadow_post_z") for r in subset]
        post_yaw = [r.get("refiner_alignment_v3_shadow_post_yaw") for r in subset]
        pred_pos = [r.get("refiner_alignment_v3_shadow_pred_pos_norm") for r in subset]
        pred_yaw = [r.get("refiner_alignment_v3_shadow_pred_yaw_abs") for r in subset]
        return {
            "n": len(subset),
            "cur_xy_mean": _mean(cur_xy),
            "cur_z_mean": _mean(cur_z),
            "cur_yaw_mean": _mean(cur_yaw),
            "post_xy_mean": _mean(post_xy),
            "post_z_mean": _mean(post_z),
            "post_yaw_mean": _mean(post_yaw),
            "xy_improved_rate": _mean([bool(r.get("refiner_alignment_v3_shadow_xy_improved", False)) for r in subset]),
            "z_improved_rate": _mean([bool(r.get("refiner_alignment_v3_shadow_z_improved", False)) for r in subset]),
            "yaw_improved_rate": _mean([bool(r.get("refiner_alignment_v3_shadow_yaw_improved", False)) for r in subset]),
            "all_improved_rate": _mean([bool(r.get("refiner_alignment_v3_shadow_all_improved", False)) for r in subset]),
            "pred_pos_norm_mean": _mean(pred_pos),
            "pred_yaw_abs_mean": _mean(pred_yaw),
            "worse_xy_rate": _mean([
                bool(r.get("refiner_alignment_v3_shadow_post_xy", 0.0) is not None and r.get("refiner_alignment_v3_shadow_cur_xy", 0.0) is not None and float(r.get("refiner_alignment_v3_shadow_post_xy", 0.0)) > float(r.get("refiner_alignment_v3_shadow_cur_xy", 0.0)))
                for r in subset
            ]),
            "worse_z_rate": _mean([
                bool(r.get("refiner_alignment_v3_shadow_post_z", 0.0) is not None and r.get("refiner_alignment_v3_shadow_cur_z", 0.0) is not None and float(r.get("refiner_alignment_v3_shadow_post_z", 0.0)) > float(r.get("refiner_alignment_v3_shadow_cur_z", 0.0)))
                for r in subset
            ]),
            "worse_yaw_rate": _mean([
                bool(r.get("refiner_alignment_v3_shadow_post_yaw", 0.0) is not None and r.get("refiner_alignment_v3_shadow_cur_yaw", 0.0) is not None and float(r.get("refiner_alignment_v3_shadow_post_yaw", 0.0)) > float(r.get("refiner_alignment_v3_shadow_cur_yaw", 0.0)))
                for r in subset
            ]),
        }

    per_source: dict[str, dict] = {}
    for source in sorted(source_hist.keys()):
        subset = [r for r in active if str(r.get("refiner_alignment_v3_shadow_source", "none")) == source]
        per_source[source] = _axis_stats(subset)
        per_source[source]["gate_pass_rate"] = _mean([bool(r.get("refiner_alignment_v3_shadow_gate_pass", False)) for r in subset])

    return {
        "rows": len(rows),
        "active_rows": len(active),
        "active_rate": float(len(active) / max(len(rows), 1)),
        "source_hist": dict(source_hist),
        "active_source_hist": dict(active_source_hist),
        "runtime_delta_source_hist": dict(runtime_delta_source_hist),
        "active_runtime_delta_source_hist": dict(active_runtime_delta_source_hist),
        "runtime_motion_target_delta_local": _delta_norm_summary(runtime_deltas),
        "canonical_source_rate": _mean([
            any(tok in str(r.get("runtime_motion_target_delta_source", "")).lower() for tok in ("canonical", "basin", "fallback"))
            for r in active
        ]),
        "overall": _axis_stats(active),
        "per_source": per_source,
    }


def _dataset_summary(ds: np.lib.npyio.NpzFile, prefix: str) -> dict:
    if "current_xy_error" in ds.files:
        current_xy = _safe_arr(ds, "current_xy_error")
        current_z = _safe_arr(ds, "current_z_error")
        current_yaw = _safe_arr(ds, "current_yaw_error")
    else:
        cur = np.asarray(_safe_arr(ds, "current_to_target_delta_local"), dtype=np.float32)[:, :6]
        current_xy = np.linalg.norm(cur[:, :2], axis=-1)
        current_z = np.abs(cur[:, 2])
        current_yaw = np.abs(cur[:, 5])
    out = {
        "rows": int(np.asarray(current_xy).shape[0]),
        "current_xy": _stats(current_xy),
        "current_z": _stats(current_z),
        "current_yaw": _stats(current_yaw),
    }
    if f"{prefix}post_xy_error" in ds.files:
        out["post_xy"] = _stats(_safe_arr(ds, f"{prefix}post_xy_error"))
        out["post_z"] = _stats(_safe_arr(ds, f"{prefix}post_z_error"))
        out["post_yaw"] = _stats(_safe_arr(ds, f"{prefix}post_yaw_error"))
    if f"{prefix}residual_local_4d" in ds.files:
        resid = np.asarray(_safe_arr(ds, f"{prefix}residual_local_4d"), dtype=np.float32)
        out["residual_pos_norm"] = _stats(np.linalg.norm(resid[:, :3], axis=-1))
        if resid.shape[1] >= 4:
            out["residual_yaw_abs"] = _stats(np.abs(resid[:, 3]))
    if "teacher_residual_local_4d" in ds.files:
        resid = np.asarray(_safe_arr(ds, "teacher_residual_local_4d"), dtype=np.float32)
        out["teacher_residual_pos_norm"] = _stats(np.linalg.norm(resid[:, :3], axis=-1))
        out["teacher_residual_yaw_abs"] = _stats(np.abs(resid[:, 3]))
    return out


def _dataset_contract_summary(ds: np.lib.npyio.NpzFile) -> dict:
    current = np.asarray(_safe_arr(ds, "current_to_target_delta_local"), dtype=np.float32)[:, :6]
    out = {
        "rows": int(current.shape[0]),
        "current_to_target_delta_local": _delta_norm_summary([row for row in current]),
        "yaw_unit_rad_check": {
            "max_abs_yaw": float(np.max(np.abs(current[:, 5]))) if current.size else None,
            "p99_abs_yaw": float(np.percentile(np.abs(current[:, 5]), 99)) if current.size else None,
            "looks_like_degrees": bool(np.max(np.abs(current[:, 5])) > np.pi * 2.0) if current.size else False,
        },
    }
    if "raw_learned_predictor_delta_local" in ds.files:
        raw = np.asarray(ds["raw_learned_predictor_delta_local"], dtype=np.float32)[:, :6]
        out["raw_predictor_delta_norm"] = _delta_norm_summary([row for row in raw])
        yaw_prod = raw[:, 5] * current[:, 5]
        nonzero = (np.abs(raw[:, 5]) > 1e-8) & (np.abs(current[:, 5]) > 1e-8)
        out["yaw_sign_agreement"] = (
            float(np.mean(yaw_prod[nonzero] > 0.0)) if np.any(nonzero) else None
        )
    if "runtime_target_delta_source" in ds.files:
        out["dataset_delta_source_histogram"] = dict(Counter(np.asarray(ds["runtime_target_delta_source"], dtype=str).tolist()))
    return out


def _matched_runtime_vs_dataset(rows: list[dict], ds: np.lib.npyio.NpzFile) -> dict:
    if "episode_index" not in ds.files or "step_index" not in ds.files or "current_to_target_delta_local" not in ds.files:
        return {"matched_rows": 0, "reason": "missing_dataset_keys"}
    lookup = {}
    ds_delta = np.asarray(ds["current_to_target_delta_local"], dtype=np.float32)[:, :6]
    for i, (ep, step) in enumerate(zip(np.asarray(ds["episode_index"]).reshape(-1), np.asarray(ds["step_index"]).reshape(-1))):
        lookup[(int(ep), int(step))] = ds_delta[i]
    diffs = []
    yaw_sign = []
    for row in rows:
        delta = _vec6(row.get("refiner_motion_target_delta_local"))
        if delta is None:
            continue
        ep = row.get("episode_index", None)
        step = row.get("step_index", None)
        if ep is None or step is None:
            continue
        key = (int(ep), int(step))
        if key not in lookup:
            continue
        ref = lookup[key]
        diffs.append(delta - ref)
        if abs(float(delta[5])) > 1e-8 and abs(float(ref[5])) > 1e-8:
            yaw_sign.append(float(delta[5]) * float(ref[5]) > 0.0)
    if not diffs:
        return {"matched_rows": 0, "reason": "no_episode_step_matches"}
    mat = np.stack(diffs, axis=0).astype(np.float32)
    return {
        "matched_rows": int(mat.shape[0]),
        "runtime_vs_dataset_delta_diff": {
            "pos_norm": _stats(np.linalg.norm(mat[:, :3], axis=-1)),
            "xy_norm": _stats(np.linalg.norm(mat[:, :2], axis=-1)),
            "z_abs": _stats(np.abs(mat[:, 2])),
            "yaw_abs": _stats(np.abs(mat[:, 5])),
        },
        "yaw_sign_agreement": float(np.mean(yaw_sign)) if yaw_sign else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--teacher_npz", type=Path, required=True)
    ap.add_argument("--dataset_npz", type=Path, default=None)
    ap.add_argument("--output_json", type=Path, required=True)
    args = ap.parse_args()

    rows = _load_trace_rows(args.trace_dir)
    runtime = _source_summary(rows)

    teacher = np.load(args.teacher_npz, allow_pickle=True)
    teacher_summary = _dataset_summary(teacher, prefix="teacher_")
    teacher_summary["selected_rows"] = int(teacher_summary["rows"])
    teacher_summary["bucket_hist"] = dict(Counter(np.asarray(_safe_arr(teacher, "stage_bucket"), dtype=str).tolist()))
    dataset_contract = None
    matched_contract = None
    if args.dataset_npz is not None:
        dataset = np.load(args.dataset_npz, allow_pickle=True)
        dataset_contract = _dataset_contract_summary(dataset)
        matched_contract = _matched_runtime_vs_dataset(rows, dataset)

    runtime_active = runtime["overall"]
    # Build a compact contract gap summary.
    contract_gap = {
        "runtime_vs_teacher_current_mean_ratio": {
            "xy": None if teacher_summary["current_xy"]["mean"] in (None, 0) else float(runtime_active["cur_xy_mean"] / teacher_summary["current_xy"]["mean"]),
            "z": None if teacher_summary["current_z"]["mean"] in (None, 0) else float(runtime_active["cur_z_mean"] / teacher_summary["current_z"]["mean"]),
            "yaw": None if teacher_summary["current_yaw"]["mean"] in (None, 0) else float(runtime_active["cur_yaw_mean"] / teacher_summary["current_yaw"]["mean"]),
        },
        "runtime_vs_teacher_current_p90_ratio": {
            "xy": None if teacher_summary["current_xy"]["p90"] in (None, 0) else float(runtime_active["cur_xy_mean"] / teacher_summary["current_xy"]["p90"]),
            "z": None if teacher_summary["current_z"]["p90"] in (None, 0) else float(runtime_active["cur_z_mean"] / teacher_summary["current_z"]["p90"]),
            "yaw": None if teacher_summary["current_yaw"]["p90"] in (None, 0) else float(runtime_active["cur_yaw_mean"] / teacher_summary["current_yaw"]["p90"]),
        },
        "runtime_pred_vs_teacher_residual_mean_ratio": {
            "pos": None,
            "yaw": None,
        },
    }
    if "teacher_residual_pos_norm" in teacher_summary and teacher_summary["teacher_residual_pos_norm"]["mean"] not in (None, 0):
        contract_gap["runtime_pred_vs_teacher_residual_mean_ratio"]["pos"] = float(runtime_active["pred_pos_norm_mean"] / teacher_summary["teacher_residual_pos_norm"]["mean"])
    if "teacher_residual_yaw_abs" in teacher_summary and teacher_summary["teacher_residual_yaw_abs"]["mean"] not in (None, 0):
        contract_gap["runtime_pred_vs_teacher_residual_mean_ratio"]["yaw"] = float(runtime_active["pred_yaw_abs_mean"] / teacher_summary["teacher_residual_yaw_abs"]["mean"])

    report = {
        "audit": "alignment_v3_runtime_contract",
        "runtime": runtime,
        "teacher_dataset": teacher_summary,
        "runtime_dataset_contract": dataset_contract,
        "matched_runtime_vs_dataset": matched_contract,
        "contract_gap": contract_gap,
        "interpretation": {
            "runtime_delta_source_dominant": max(runtime["source_hist"].items(), key=lambda kv: kv[1])[0] if runtime["source_hist"] else "none",
            "canonical_fallback_present_in_active_v3": bool((runtime.get("canonical_source_rate") or 0.0) > 0.0),
            "runtime_post_is_worse_xy": bool(runtime_active["post_xy_mean"] is not None and runtime_active["cur_xy_mean"] is not None and runtime_active["post_xy_mean"] > runtime_active["cur_xy_mean"]),
            "runtime_post_is_worse_z": bool(runtime_active["post_z_mean"] is not None and runtime_active["cur_z_mean"] is not None and runtime_active["post_z_mean"] > runtime_active["cur_z_mean"]),
            "runtime_post_is_worse_yaw": bool(runtime_active["post_yaw_mean"] is not None and runtime_active["cur_yaw_mean"] is not None and runtime_active["post_yaw_mean"] > runtime_active["cur_yaw_mean"]),
        },
        "source_paths": sorted({row["_trace_path"] for row in rows}),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
