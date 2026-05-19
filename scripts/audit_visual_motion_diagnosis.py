#!/usr/bin/env python3
"""Visual motion diagnosis: per-episode kinematics + alignment + handoff summary.

Companion to visual eval MP4s. Reads eval_results.json + gripper_traces/
and produces a structured report for human video review cross-referencing.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def _ep_from_filename(path: Path) -> int:
    m = re.search(r"ep(\d+)", path.name)
    if m is None:
        raise ValueError(f"cannot extract episode from {path.name}")
    return int(m.group(1))


def _load_traces(trace_dir: Path) -> dict[int, list[dict]]:
    episodes: dict[int, list[dict]] = {}
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep = _ep_from_filename(path)
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        episodes[ep] = sorted(rows, key=lambda r: int(r.get("step", 0)))
    return episodes


def _motion_diagnosis(rows: list[dict], eval_data: dict | None = None) -> dict:
    total = len(rows)

    # --- delta statistics from exec_gripper_raw / base_gripper_raw ---
    # The trace has base_gripper_raw and exec_gripper_raw, but for xyz we need
    # to derive from handoff_metrics or the delta_basin fields.
    # Use alignment_runtime_basin_{xy,z,yaw} as position error estimates.

    basin_xy = []
    basin_z = []
    basin_yaw = []
    for r in rows:
        v = r.get("alignment_runtime_basin_xy")
        if v is not None and np.isfinite(float(v)):
            basin_xy.append(float(v))
        v = r.get("alignment_runtime_basin_z")
        if v is not None and np.isfinite(float(v)):
            basin_z.append(float(v))
        v = r.get("alignment_runtime_basin_yaw")
        if v is not None and np.isfinite(float(v)):
            basin_yaw.append(float(v))

    # Handoff metrics for richer error data
    hm_xy = []
    hm_z = []
    hm_yaw = []
    for r in rows:
        hm = r.get("handoff_metrics_provider") or {}
        for key, store in [("xy_error", hm_xy), ("abs_z_error", hm_z), ("yaw_error", hm_yaw)]:
            v = hm.get(key)
            if v is not None and np.isfinite(float(v)):
                store.append(float(v))

    # --- state histograms ---
    contact_hist = Counter(r.get("refiner_contact_state", "?") for r in rows)
    substage_hist = Counter(r.get("refiner_substage", "?") for r in rows)
    close_state_hist = Counter(r.get("refiner_close_state", "?") for r in rows)

    # --- alignment signals ---
    takeover_count = sum(1 for r in rows if r.get("refiner_alignment_takeover_active"))
    gate_open_count = sum(1 for r in rows if r.get("refiner_alignment_gate_open"))
    total_with_gate = sum(1 for r in rows if r.get("refiner_alignment_gate_open") is not None)

    # --- planner close intent ---
    close_intent_count = sum(1 for r in rows if r.get("refiner_alignment_planner_close_intent"))

    # --- candidate histogram ---
    candidate_hist: dict[str, int] = {}
    for r in rows:
        idx = r.get("refiner_last_scorer_candidate_index", -1)
        if idx is not None and idx >= 0:
            candidate_hist[str(int(idx))] = candidate_hist.get(str(int(idx)), 0) + 1

    # --- block reasons ---
    block_hist = Counter(str(r.get("refiner_alignment_blocked_reason", "?")) for r in rows)

    # --- action decision ---
    action_hist = Counter(str(r.get("refiner_close_action_decision", "?")) for r in rows)

    # --- workspace violations ---
    ws_viol_count = -1
    if eval_data:
        stage_stats = eval_data.get("stage_stats", [])
        for s in stage_stats:
            if int(s.get("episode_index", -1)) == _ep_from_trace(rows):
                ws_viol_count = int(s.get("workspace_violation_count", -1))
                break

    # --- clip hit rate ---
    clip_rate = -1.0
    if eval_data:
        refiner_stats = eval_data.get("refiner_stats", [])
        for rs in refiner_stats:
            if int(rs.get("episode_index", -1)) == _ep_from_trace(rows) or (
                "episode_index" not in rs and len(refiner_stats) > 0
            ):
                clip_rate = float(rs.get("clip_hit_rate", -1.0))
                break

    # --- handoff readiness (raw) ---
    ready_probs = []
    uncertainties = []
    for r in rows:
        ha = r.get("handoff_aux_provider") or {}
        for key, store in [("pred_ready_prob", ready_probs), ("pred_uncertainty", uncertainties)]:
            v = ha.get(key)
            if v is not None and np.isfinite(float(v)):
                store.append(float(v))

    # --- first_close_step ---
    first_close = -1
    for r in rows:
        fcs = r.get("first_close_step_so_far", -1)
        if fcs is not None and fcs >= 0:
            first_close = int(fcs)
            break

    # --- end-effector / target info from last step ---
    last = rows[-1]
    last_sm = last.get("refiner_close_state_machine") or {}
    final_sm_summary = {
        "state": last_sm.get("state"),
        "has_motion_target": last_sm.get("has_motion_target"),
        "has_handoff_geometry": last_sm.get("has_handoff_geometry"),
        "handoff_spec_name": last_sm.get("handoff_spec_name"),
        "runtime_geometry_ready": last_sm.get("runtime_geometry_ready"),
        "xy_error": last_sm.get("xy_error"),
        "abs_z_error": last_sm.get("abs_z_error"),
        "yaw_error": last_sm.get("yaw_error"),
        "wants_close": last_sm.get("wants_close"),
    }

    def _p(arr, q):
        if arr is None or (hasattr(arr, '__len__') and len(arr) == 0):
            return float("nan")
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, q)) if a.size > 0 else float("nan")

    def _stats(arr, label):
        a = np.array(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "mean": round(float(a.mean()), 6),
            "p50": round(_p(a, 50), 6),
            "p90": round(_p(a, 90), 6),
            "min": round(float(a.min()), 6),
            "max": round(float(a.max()), 6),
        }

    return {
        "steps": total,
        "first_close_step": first_close,
        "final_substage": str(last.get("refiner_substage", "?")),
        "final_close_state": str(last.get("refiner_close_state", "?")),
        "close_state_hist": {str(k): v for k, v in close_state_hist.items()},
        "contact_state_hist": {str(k): v for k, v in contact_hist.items()},
        "substage_hist": {str(k): v for k, v in substage_hist.items()},
        "planner_close_intent_count": close_intent_count,
        "planner_close_intent_rate": round(close_intent_count / max(total, 1), 6),
        "alignment_takeover_count": takeover_count,
        "alignment_gate_open_count": gate_open_count,
        "alignment_gate_open_rate": round(gate_open_count / max(total_with_gate, 1), 6),
        "alignment_blocked_reason_hist": {str(k): v for k, v in block_hist.items()},
        "close_action_decision_hist": {str(k): v for k, v in action_hist.items()},
        "scorer_candidate_hist": candidate_hist,
        "workspace_violation_count": ws_viol_count,
        "clip_hit_rate": clip_rate,
        "basin_distance": {
            "xy": _stats(basin_xy, "xy"),
            "z": _stats(basin_z, "z"),
            "yaw": _stats(basin_yaw, "yaw"),
        },
        "handoff_metrics": {
            "xy": _stats(hm_xy, "xy"),
            "z": _stats(hm_z, "z"),
            "yaw": _stats(hm_yaw, "yaw"),
        },
        "handoff_model": {
            "ready_prob": _stats(ready_probs, "ready_prob"),
            "uncertainty": _stats(uncertainties, "uncertainty"),
        },
        "final_close_state_machine_snapshot": final_sm_summary,
    }


def _ep_from_trace(rows: list[dict]) -> int:
    for r in rows:
        if "_ep" in r:
            return int(r["_ep"])
    return -1


def _find_videos(task_dir: Path) -> dict[int, str]:
    video_dir = task_dir / "videos"
    videos: dict[int, str] = {}
    if video_dir.is_dir():
        for path in sorted(video_dir.glob("*.mp4")):
            m = re.search(r"ep(\d+)", path.name)
            if m:
                videos[int(m.group(1))] = str(path)
    return videos


def _find_best_gifs(task_dir: Path) -> list[str]:
    gif_dir = task_dir / "best_gifs" if (task_dir / "best_gifs").is_dir() else task_dir
    gifs = list(gif_dir.glob("*.gif"))
    return [str(p) for p in sorted(gifs)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual motion diagnosis report")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    eval_candidates = sorted(run_dir.glob("*/eval_results.json"))
    if not eval_candidates:
        print("ERROR: no eval_results.json", flush=True)
        raise SystemExit(1)

    task_dir = eval_candidates[0].parent
    trace_dir = task_dir / "gripper_traces"

    with eval_candidates[0].open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    episodes = _load_traces(trace_dir)
    videos = _find_videos(task_dir)
    gifs = _find_best_gifs(task_dir)

    per_ep = {}
    for ep, rows in sorted(episodes.items()):
        # tag rows with episode
        for r in rows:
            r["_ep"] = ep
        per_ep[f"ep{ep:03d}"] = _motion_diagnosis(rows, eval_data)

    # Summary from eval_data
    summary = {
        "success_rate": float(eval_data.get("success_rate", -1)),
        "avg_episode_length": float(eval_data.get("avg_episode_length", -1)),
        "invalid_action_count": int(eval_data.get("invalid_action_count", -1)),
        "planner_close_intent_rate": float(eval_data.get("planner_close_intent_rate", -1)),
        "ready_to_close_prob_mean": float(eval_data.get("ready_to_close_prob_mean", -1)),
        "trigger_prob_mean": float(eval_data.get("trigger_prob_mean", -1)),
        "alignment_takeover_count": int(eval_data.get("alignment_takeover_count", -1)),
        "close_latch_set_count": int(eval_data.get("close_latch_set_count", -1)),
        "workspace_violation_count": int(eval_data.get("workspace_violation_count", -1)),
        "clip_hit_rate": float(eval_data.get("clip_hit_rate", -1)),
        "alpha_mean": float(eval_data.get("alpha_mean", -1)),
        "task_name": str(eval_data.get("task_name", "")),
        "mode": str(eval_data.get("mode", "")),
        "checkpoint": str(eval_data.get("checkpoint", "")),
        "target_provider_source_hist": eval_data.get("target_provider_source_hist", {}),
        "handoff_provider_ckpt": str(eval_data.get("handoff_provider_ckpt", "none")),
    }

    report = {
        "audit": "visual_motion_diagnosis",
        "run_dir": str(run_dir),
        "task_dir": str(task_dir),
        "videos": {f"ep{ep:03d}": path for ep, path in sorted(videos.items())},
        "best_gifs": gifs,
        "summary": summary,
        "per_episode": per_ep,
    }

    output_path = args.output or (task_dir / "visual_motion_diagnosis_report.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Print
    print(f"[diagnosis] report -> {output_path}")
    print(f"\n=== VIDEOS ===")
    for ep, path in sorted(videos.items()):
        print(f"  ep{ep:03d}: {path}")
    if not videos:
        print("  NO MP4 VIDEOS FOUND!")
    print(f"\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n=== PER-EPISODE ===")
    for ep_label, d in per_ep.items():
        print(f"\n  {ep_label}:")
        print(f"    steps={d['steps']}  first_close_step={d['first_close_step']}")
        print(f"    final_substage={d['final_substage']}  final_close_state={d['final_close_state']}")
        print(f"    contact_state_hist={d['contact_state_hist']}")
        print(f"    planner_close_intent_rate={d['planner_close_intent_rate']}")
        print(f"    alignment_takeover={d['alignment_takeover_count']}  gate_open_rate={d['alignment_gate_open_rate']}")
        print(f"    block_reasons={d['alignment_blocked_reason_hist']}")
        print(f"    basin_xy: {d['basin_distance']['xy']}")
        print(f"    basin_z:  {d['basin_distance']['z']}")
        print(f"    basin_yaw: {d['basin_distance']['yaw']}")
        print(f"    handoff_metrics_z: {d['handoff_metrics']['z']}")
        print(f"    handoff_model_ready_prob: {d['handoff_model']['ready_prob']}")
        print(f"    candidate_hist={d['scorer_candidate_hist']}")
        print(f"    clip_hit_rate={d['clip_hit_rate']}  ws_viol={d['workspace_violation_count']}")

    print(f"\n=== HUMAN REVIEW CHECKLIST ===")
    checklist = [
        "1. Is the end-effector approaching the peg/hole/target region?",
        "2. Is z consistently too high (above target) or too low (below)?",
        "3. Is there persistent lateral xy offset (which direction)?",
        "4. Is the motion smooth micro-adjustment or oscillating jitter?",
        "5. Does the gripper stay open throughout?",
        "6. Are there visible workspace clipping / boundary violations?",
        "7. Does alignment correction pull the trajectory off-course or help?",
        "8. Does the baseline planner (pre-alignment) reach near-close positions?",
        "9. At the final frame, is the peg visually aligned with the hole?",
        "10. Does z_error in metrics match the visual gap to target?",
    ]
    for c in checklist:
        print(f"  {c}")


if __name__ == "__main__":
    main()
