"""
Extract provider-owned stage handoff specs from successful support rows.

This is the rollout-side counterpart to extract_stage_handoff_specs_from_demos.py:
instead of using official demo close windows, it mines the handoff-ready geometry
from successful privileged rollouts. The intended use is to refresh runtime
handoff gates without changing the motion target or scorer weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support_npz", type=str, required=True)
    parser.add_argument("--meta_json", type=str, required=True)
    parser.add_argument("--base_spec_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--substage_id", type=int, default=1)
    parser.add_argument("--window_before_success", type=int, default=3)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--threshold_scale", type=float, default=1.0)
    parser.add_argument("--max_xy_threshold", type=float, default=0.0045)
    parser.add_argument("--max_z_threshold", type=float, default=0.0060)
    parser.add_argument("--max_yaw_threshold", type=float, default=0.13)
    parser.add_argument("--min_xy_threshold", type=float, default=0.0030)
    parser.add_argument("--min_z_threshold", type=float, default=0.0015)
    parser.add_argument("--min_yaw_threshold", type=float, default=0.10)
    args = parser.parse_args()

    raw = np.load(Path(args.support_npz))
    meta = json.loads(Path(args.meta_json).read_text())
    spec_doc = json.loads(Path(args.base_spec_json).read_text())

    first_success = {
        int(k): int(v)
        for k, v in dict(meta.get("phase1_first_success_step", {})).items()
    }
    ep_ids = np.asarray(raw["episode_index"], dtype=np.int64)
    rollout_step = np.asarray(raw["rollout_step"], dtype=np.int64)
    ready = np.asarray(raw["ready_to_close_target"], dtype=np.float32) > 0.5
    xy = np.asarray(raw["handoff_metric_xy_error"], dtype=np.float32)
    z = np.asarray(raw["handoff_metric_abs_z_error"], dtype=np.float32)
    yaw = np.asarray(raw["handoff_metric_yaw_error"], dtype=np.float32)
    tilt = np.asarray(raw["handoff_metric_tilt_error"], dtype=np.float32)

    selected_rows = []
    selected_eps = []
    for ep, success_step in sorted(first_success.items()):
        idx = np.where(ep_ids == ep)[0]
        if idx.size == 0:
            continue
        pre = idx[
            (rollout_step[idx] >= max(0, success_step - int(args.window_before_success)))
            & (rollout_step[idx] <= success_step)
        ]
        chosen = pre[ready[pre]]
        if chosen.size == 0:
            continue
        selected_eps.append(int(ep))
        for i in chosen.tolist():
            selected_rows.append(
                [
                    float(xy[i]),
                    float(z[i]),
                    float(yaw[i]),
                    float(tilt[i]),
                ]
            )
    if not selected_rows:
        raise RuntimeError("No successful ready rows found in support NPZ.")

    arr = np.asarray(selected_rows, dtype=np.float32)

    def q(col: int, qv: float) -> float:
        return float(np.quantile(arr[:, col], float(qv)) * float(args.threshold_scale))

    xy_thr = float(np.clip(q(0, args.quantile), float(args.min_xy_threshold), float(args.max_xy_threshold)))
    z_thr = float(np.clip(q(1, args.quantile), float(args.min_z_threshold), float(args.max_z_threshold)))
    yaw_thr = float(np.clip(q(2, args.quantile), float(args.min_yaw_threshold), float(args.max_yaw_threshold)))

    stages = list(spec_doc.get("stages", []))
    replaced = False
    for row in stages:
        if int(row.get("substage_id", -1)) == int(args.substage_id):
            thresholds = {
                "xy_error": xy_thr,
                "abs_z_error": z_thr,
                "yaw_error": yaw_thr,
                "tilt_error": -1.0,
            }
            row["release_thresholds"] = dict(thresholds)
            row["metric_thresholds"] = dict(thresholds)
            row.setdefault("optimization_thresholds", dict(thresholds))
            row["source"] = "privileged_success_rollout"
            row["quantile"] = float(args.quantile)
            row["threshold_scale"] = float(args.threshold_scale)
            row["num_metric_rows"] = int(arr.shape[0])
            row["num_episodes_used"] = int(len(selected_eps))
            row["episode_names"] = [f"episode{ep}" for ep in selected_eps[:100]]
            row["support_npz"] = str(args.support_npz)
            row["meta_json"] = str(args.meta_json)
            replaced = True
            break
    if not replaced:
        raise RuntimeError(f"Could not find substage_id={args.substage_id} in {args.base_spec_json}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec_doc, indent=2))
    q50 = np.quantile(arr, 0.5, axis=0).tolist()
    q90 = np.quantile(arr, 0.9, axis=0).tolist()
    q95 = np.quantile(arr, 0.95, axis=0).tolist()
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "num_rows": int(arr.shape[0]),
                "num_episodes_used": int(len(selected_eps)),
                "thresholds": {
                    "xy_error": xy_thr,
                    "abs_z_error": z_thr,
                    "yaw_error": yaw_thr,
                    "tilt_error": -1.0,
                },
                "quantiles": {
                    "q50": q50,
                    "q90": q90,
                    "q95": q95,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
