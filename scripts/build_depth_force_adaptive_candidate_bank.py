#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_pose_candidate_dataset import build_action_primitives, build_orientation_rescue_primitives, parse_float_list_arg
from build_depth_force_candidate_cost_dataset import _candidate_kind


def _unique_rows(actions: list[np.ndarray], meta: list[dict[str, object]], *, decimals: int = 6) -> tuple[np.ndarray, list[dict[str, object]]]:
    seen: set[tuple[float, ...]] = set()
    kept_actions: list[np.ndarray] = []
    kept_meta: list[dict[str, object]] = []
    for action, info in zip(actions, meta):
        arr = np.asarray(action, dtype=np.float32).reshape(6)
        key = tuple(np.round(arr, decimals=decimals).tolist())
        if key in seen:
            continue
        seen.add(key)
        kept_actions.append(arr)
        kept_meta.append(info)
    if not kept_actions:
        return np.zeros((0, 6), dtype=np.float32), []
    return np.stack(kept_actions, axis=0).astype(np.float32), kept_meta


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--primitive_xy_small", type=float, default=0.004)
    ap.add_argument("--primitive_xy_large", type=float, default=0.008)
    ap.add_argument("--primitive_xy_micro_values", type=str, default="0.001,0.0015,0.002,0.003")
    ap.add_argument("--primitive_z_small", type=float, default=0.004)
    ap.add_argument("--primitive_yaw_small", type=float, default=0.03)
    ap.add_argument("--primitive_yaw_probe_values", type=str, default="0.0174533,0.0349066,0.0698132")
    ap.add_argument("--primitive_pitch_small", type=float, default=0.06)
    ap.add_argument("--primitive_roll_small", type=float, default=0.06)
    ap.add_argument("--primitive_include_descend", action="store_true", default=True)
    ap.add_argument("--no_primitive_include_descend", dest="primitive_include_descend", action="store_false")
    ap.add_argument("--primitive_include_combos", action="store_true", default=True)
    ap.add_argument("--no_primitive_include_combos", dest="primitive_include_combos", action="store_false")
    ap.add_argument("--primitive_include_tilt", action="store_true", default=True)
    ap.add_argument("--no_primitive_include_tilt", dest="primitive_include_tilt", action="store_false")
    ap.add_argument("--scale_factors", type=str, default="0.125,0.25,0.375,0.5,0.625,0.75")
    ap.add_argument("--include_rescue_primitives", action="store_true", default=True)
    ap.add_argument("--no_include_rescue_primitives", dest="include_rescue_primitives", action="store_false")
    ap.add_argument("--rescue_pitch_small", type=float, default=0.04)
    ap.add_argument("--rescue_roll_small", type=float, default=0.04)
    ap.add_argument("--rescue_xy_small", type=float, default=0.004)
    args = ap.parse_args()

    scales = [float(x) for x in parse_float_list_arg(args.scale_factors)]
    base_actions = build_action_primitives(
        xy_small=float(args.primitive_xy_small),
        xy_large=float(args.primitive_xy_large),
        xy_micro_values=parse_float_list_arg(args.primitive_xy_micro_values),
        z_small=float(args.primitive_z_small),
        yaw_small=float(args.primitive_yaw_small),
        yaw_probe_values=parse_float_list_arg(args.primitive_yaw_probe_values),
        pitch_small=float(args.primitive_pitch_small),
        roll_small=float(args.primitive_roll_small),
        include_descend=bool(args.primitive_include_descend),
        include_combos=bool(args.primitive_include_combos),
        include_tilt=bool(args.primitive_include_tilt),
    )
    rescue_actions = (
        build_orientation_rescue_primitives(
            pitch_small=float(args.rescue_pitch_small),
            roll_small=float(args.rescue_roll_small),
            xy_small=float(args.rescue_xy_small),
            coupled_xy_tilt=True,
        )
        if bool(args.include_rescue_primitives)
        else []
    )

    actions: list[np.ndarray] = []
    meta: list[dict[str, object]] = []

    def add(action: np.ndarray, *, source: str, scale: float, origin: str) -> None:
        arr = np.asarray(action, dtype=np.float32).reshape(6)
        actions.append(arr)
        meta.append(
            {
                "source": source,
                "origin": origin,
                "scale": float(scale),
                "kind": _candidate_kind(arr),
            }
        )

    for arr in base_actions:
        add(arr, source="base", scale=1.0, origin="base")
    for arr in rescue_actions:
        add(arr, source="rescue", scale=1.0, origin="base")

    for source_name, bank in [("base", base_actions), ("rescue", rescue_actions)]:
        for arr in bank:
            a = np.asarray(arr, dtype=np.float32).reshape(6)
            if np.linalg.norm(a) <= 1e-8:
                continue
            for scale in scales:
                s = float(scale)
                if s <= 0.0 or abs(s - 1.0) <= 1e-8:
                    continue
                add(s * a, source=source_name, scale=s, origin="scaled")

    candidate_actions, meta = _unique_rows(actions, meta)
    candidate_kind = np.asarray([m["kind"] for m in meta], dtype="U16")
    candidate_source = np.asarray([m["source"] for m in meta], dtype="U16")
    candidate_origin = np.asarray([m["origin"] for m in meta], dtype="U16")
    candidate_scale = np.asarray([float(m["scale"]) for m in meta], dtype=np.float32)
    candidate_norm = np.linalg.norm(candidate_actions, axis=1).astype(np.float32)

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        candidate_actions_local=candidate_actions.astype(np.float32),
        candidate_kind=candidate_kind,
        candidate_source=candidate_source,
        candidate_origin=candidate_origin,
        candidate_scale=candidate_scale,
        candidate_norm=candidate_norm,
    )
    report = {
        "output_npz": str(out_path),
        "candidate_count": int(candidate_actions.shape[0]),
        "base_count": int(len(base_actions)),
        "rescue_count": int(len(rescue_actions)),
        "scale_factors": scales,
        "candidate_kind_hist": {
            k: int(v) for k, v in zip(*np.unique(candidate_kind, return_counts=True))
        },
        "candidate_source_hist": {
            k: int(v) for k, v in zip(*np.unique(candidate_source, return_counts=True))
        },
        "candidate_origin_hist": {
            k: int(v) for k, v in zip(*np.unique(candidate_origin, return_counts=True))
        },
        "candidate_scale_hist": {
            str(float(k)): int(v) for k, v in zip(*np.unique(candidate_scale, return_counts=True))
        },
        "candidate_norm": _summary_stats(candidate_norm),
        "xy_norm": _summary_stats(np.linalg.norm(candidate_actions[:, :2], axis=1)),
        "z_abs": _summary_stats(np.abs(candidate_actions[:, 2])),
        "yaw_abs": _summary_stats(np.abs(candidate_actions[:, 5])),
    }
    out_json = Path(args.output_json) if args.output_json else out_path.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
