#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str):
    return json.loads(Path(path).read_text())


def _pick_best_epoch(history: list[dict], false_ready_max: float) -> dict:
    candidates = [h for h in history if float(h.get("false_ready_rate", 1.0)) <= false_ready_max]
    if not candidates:
        candidates = history
    # Ready-first deploy proxy: prioritize overlap-supportive metrics while keeping safety.
    def score(h: dict) -> float:
        return (
            0.40 * float(h.get("teacher_ready_ready_prob_mean", 0.0))
            + 0.25 * float(h.get("ready_support_release_band_rate", 0.0))
            + 0.20 * float(h.get("teacher_ready_xy_in_band_rate", 0.0))
            + 0.15 * float(h.get("ready_support_xy_in_band_rate", 0.0))
        )

    return max(candidates, key=score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_history_json", required=True)
    ap.add_argument("--shadow_analysis_json", action="append", default=[])
    ap.add_argument("--baseline_history_json", default=None)
    ap.add_argument("--output_json", required=True)
    ap.add_argument(
        "--gate_mode",
        choices=("shadow", "applied"),
        default="applied",
        help="shadow uses teacher_pred_ready_overlap; applied uses teacher_runtime_handoff_ready_overlap.",
    )
    ap.add_argument("--max_far_negative_false_ready", type=float, default=0.01)
    ap.add_argument("--min_overlap_frames_total", type=int, default=2)
    ap.add_argument("--min_overlap_episodes", type=int, default=2)
    ap.add_argument("--min_yaw_needed_mae_gain", type=float, default=0.0)
    args = ap.parse_args()

    hist = _load_json(args.train_history_json)
    if not isinstance(hist, list) or not hist:
        raise RuntimeError("train_history_json must be a non-empty list")
    best = _pick_best_epoch(hist, false_ready_max=float(args.max_far_negative_false_ready))

    far_neg_ok = float(best.get("subset_far_negative_false_ready_rate", best.get("false_ready_rate", 1.0))) <= float(
        args.max_far_negative_false_ready
    )

    overlap_field = (
        "teacher_pred_ready_overlap_frames"
        if args.gate_mode == "shadow"
        else "teacher_runtime_handoff_ready_overlap_frames"
    )
    overlap_total = 0
    overlap_eps = 0
    for p in args.shadow_analysis_json:
        d = _load_json(p)
        summary = d.get("summary", {})
        overlap_total += int(summary.get(overlap_field, 0))
        for ep in d.get("episodes", []):
            if int(ep.get(overlap_field, 0)) > 0:
                overlap_eps += 1
    overlap_ok = overlap_total >= int(args.min_overlap_frames_total) and overlap_eps >= int(args.min_overlap_episodes)

    yaw_gain = None
    yaw_ok = True
    if args.baseline_history_json:
        base_hist = _load_json(args.baseline_history_json)
        if not isinstance(base_hist, list) or not base_hist:
            raise RuntimeError("baseline_history_json must be a non-empty list")
        base_best = _pick_best_epoch(base_hist, false_ready_max=1.0)
        cur_mae = float(best.get("subset_yaw_needed_mae_yaw_norm", best.get("mae_yaw_norm", 1e9)))
        base_mae = float(base_best.get("subset_yaw_needed_mae_yaw_norm", base_best.get("mae_yaw_norm", 1e9)))
        yaw_gain = base_mae - cur_mae
        yaw_ok = yaw_gain >= float(args.min_yaw_needed_mae_gain)

    decision = bool(overlap_ok and far_neg_ok and yaw_ok)
    out = {
        "decision": "pass_to_B1_B2" if decision else "stay_phaseA",
        "passed": decision,
        "criteria": {
            "overlap_ok": overlap_ok,
            "far_negative_ok": far_neg_ok,
            "yaw_ok": yaw_ok,
        },
        "values": {
            "gate_mode": str(args.gate_mode),
            "overlap_field": str(overlap_field),
            "overlap_total_frames": overlap_total,
            "overlap_episode_count": overlap_eps,
            "best_epoch": int(best.get("epoch", -1)),
            "best_teacher_ready_ready_prob_mean": float(best.get("teacher_ready_ready_prob_mean", 0.0)),
            "best_ready_support_release_band_rate": float(best.get("ready_support_release_band_rate", 0.0)),
            "best_far_negative_false_ready_rate": float(
                best.get("subset_far_negative_false_ready_rate", best.get("false_ready_rate", 1.0))
            ),
            "best_subset_yaw_needed_mae_yaw_norm": float(
                best.get("subset_yaw_needed_mae_yaw_norm", best.get("mae_yaw_norm", 1e9))
            ),
            "yaw_gain_vs_baseline": yaw_gain,
        },
        "inputs": {
            "train_history_json": args.train_history_json,
            "shadow_analysis_json": list(args.shadow_analysis_json),
            "baseline_history_json": args.baseline_history_json,
        },
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
