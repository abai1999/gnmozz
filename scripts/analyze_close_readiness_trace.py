#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def find_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "gripper_traces").is_dir():
        path = path / "gripper_traces"
    files = sorted(path.glob("*_gripper_trace.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    return files


def as_bool(row: dict, key: str) -> bool:
    return bool(row.get(key, False))


def as_float(value, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def summarize_trace(path: Path) -> dict:
    planner_close_frames = 0
    close_req_frames = 0
    close_block_frames = 0
    teacher_ready_frames = 0
    predicted_ready_frames = 0
    runtime_handoff_ready_frames = 0
    close_ready_diag_frames = 0
    teacher_pred_overlap_frames = 0
    pred_false_ready_frames = 0
    teacher_runtime_handoff_overlap_frames = 0
    false_close_apply_frames = 0
    false_hold_close_frames = 0
    missing_gripper_fsm_state_frames = 0
    ready_prob_peak = 0.0
    pred_yaw_norm_min = math.inf
    pred_xy_norm_min = math.inf
    pred_z_norm_min = math.inf
    first_planner_close_step = -1
    first_teacher_ready_step = -1
    first_pred_ready_step = -1
    first_runtime_handoff_ready_step = -1

    with path.open() as f:
        for step, line in enumerate(f):
            row = json.loads(line)
            planner_close = as_bool(row, "refiner_alignment_planner_close_intent")
            close_req = as_bool(row, "refiner_alignment_close_requirement_satisfied")
            close_block = as_bool(row, "refiner_current_close_veto_blocked")
            # Prefer explicit runtime prediction fields (new schema), then fall back to legacy aliases.
            pred_ready = (
                as_bool(row, "runtime_handoff_ready_pred")
                or as_bool(row, "handoff_ready_pred")
                or as_bool(row, "handoff_ready_provider")
            )
            runtime_handoff_ready = as_bool(row, "refiner_current_handoff_ready") or as_bool(
                row, "runtime_handoff_ready_applied"
            )
            close_ready_diag = as_bool(row, "refiner_current_close_veto_ready")
            teacher_ready = as_bool(row, "teacher_truth_handoff_ready")
            gripper_open = as_float(row.get("obs_gripper_open"), math.nan)
            gripper_open_bool = bool(gripper_open >= 0.5) if math.isfinite(gripper_open) else True
            gripper_fsm_state_raw = row.get("refiner_current_gripper_fsm_state") or row.get("gripper_fsm_state")
            gripper_fsm_state = str(gripper_fsm_state_raw or "unknown")
            if not gripper_fsm_state_raw:
                missing_gripper_fsm_state_frames += 1
            hold_close_active = gripper_fsm_state in {"verify_contact", "hold_after_verified_contact"}

            planner_close_frames += int(planner_close)
            close_req_frames += int(close_req)
            close_block_frames += int(close_block)
            teacher_ready_frames += int(teacher_ready)
            predicted_ready_frames += int(pred_ready)
            runtime_handoff_ready_frames += int(runtime_handoff_ready)
            close_ready_diag_frames += int(close_ready_diag)
            teacher_pred_overlap_frames += int(teacher_ready and pred_ready)
            pred_false_ready_frames += int((not teacher_ready) and pred_ready)
            teacher_runtime_handoff_overlap_frames += int(teacher_ready and runtime_handoff_ready)
            false_close_apply_frames += int((not teacher_ready) and runtime_handoff_ready)
            false_hold_close_frames += int(
                (not teacher_ready)
                and (not runtime_handoff_ready)
                and (not gripper_open_bool)
                and hold_close_active
            )

            if planner_close and first_planner_close_step < 0:
                first_planner_close_step = step
            if teacher_ready and first_teacher_ready_step < 0:
                first_teacher_ready_step = step
            if pred_ready and first_pred_ready_step < 0:
                first_pred_ready_step = step
            if runtime_handoff_ready and first_runtime_handoff_ready_step < 0:
                first_runtime_handoff_ready_step = step

            aux = row.get("handoff_aux_provider") or {}
            ready_prob_peak = max(ready_prob_peak, as_float(aux.get("pred_ready_prob"), 0.0))
            pred_yaw = as_float(aux.get("pred_yaw_norm"))
            pred_xy = as_float(aux.get("pred_xy_norm"))
            pred_z = as_float(aux.get("pred_abs_z_norm"))
            if math.isfinite(pred_yaw):
                pred_yaw_norm_min = min(pred_yaw_norm_min, pred_yaw)
            if math.isfinite(pred_xy):
                pred_xy_norm_min = min(pred_xy_norm_min, pred_xy)
            if math.isfinite(pred_z):
                pred_z_norm_min = min(pred_z_norm_min, pred_z)

    return {
        "episode_trace": path.name,
        "planner_close_frames": planner_close_frames,
        "close_requirement_frames": close_req_frames,
        "close_veto_block_frames": close_block_frames,
        "predicted_ready_frames": predicted_ready_frames,
        "runtime_handoff_ready_frames": runtime_handoff_ready_frames,
        "teacher_ready_frames": teacher_ready_frames,
        "teacher_pred_ready_overlap_frames": teacher_pred_overlap_frames,
        "pred_false_ready_frames": pred_false_ready_frames,
        "teacher_runtime_handoff_ready_overlap_frames": teacher_runtime_handoff_overlap_frames,
        "false_close_apply_frames": false_close_apply_frames,
        "false_hold_close_frames": false_hold_close_frames,
        "missing_gripper_fsm_state_frames": missing_gripper_fsm_state_frames,
        "internal_close_veto_ready_diag_frames": close_ready_diag_frames,
        "teacher_pred_ready_overlap_rate": (
            float(teacher_pred_overlap_frames / teacher_ready_frames) if teacher_ready_frames else 0.0
        ),
        "pred_false_ready_rate": (
            float(pred_false_ready_frames / max(1, predicted_ready_frames)) if predicted_ready_frames else 0.0
        ),
        "teacher_runtime_handoff_ready_overlap_rate": (
            float(teacher_runtime_handoff_overlap_frames / teacher_ready_frames) if teacher_ready_frames else 0.0
        ),
        "false_close_apply_rate": (
            float(false_close_apply_frames / max(1, runtime_handoff_ready_frames))
            if runtime_handoff_ready_frames
            else 0.0
        ),
        "false_hold_close_rate": (
            float(false_hold_close_frames / max(1, planner_close_frames)) if planner_close_frames else 0.0
        ),
        "ready_prob_peak": ready_prob_peak,
        "pred_yaw_norm_min": None if pred_yaw_norm_min is math.inf else pred_yaw_norm_min,
        "pred_xy_norm_min": None if pred_xy_norm_min is math.inf else pred_xy_norm_min,
        "pred_z_norm_min": None if pred_z_norm_min is math.inf else pred_z_norm_min,
        "first_planner_close_step": first_planner_close_step,
        "first_teacher_ready_step": first_teacher_ready_step,
        "first_pred_ready_step": first_pred_ready_step,
        "first_runtime_handoff_ready_step": first_runtime_handoff_ready_step,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    files = find_trace_files(Path(args.trace_dir))
    episodes = [summarize_trace(p) for p in files]
    total = {
        "episode_count": len(episodes),
        "planner_close_frames": sum(ep["planner_close_frames"] for ep in episodes),
        "close_requirement_frames": sum(ep["close_requirement_frames"] for ep in episodes),
        "close_veto_block_frames": sum(ep["close_veto_block_frames"] for ep in episodes),
        "predicted_ready_frames": sum(ep["predicted_ready_frames"] for ep in episodes),
        "runtime_handoff_ready_frames": sum(ep["runtime_handoff_ready_frames"] for ep in episodes),
        "teacher_ready_frames": sum(ep["teacher_ready_frames"] for ep in episodes),
        "teacher_pred_ready_overlap_frames": sum(ep["teacher_pred_ready_overlap_frames"] for ep in episodes),
        "pred_false_ready_frames": sum(ep["pred_false_ready_frames"] for ep in episodes),
        "teacher_runtime_handoff_ready_overlap_frames": sum(
            ep["teacher_runtime_handoff_ready_overlap_frames"] for ep in episodes
        ),
        "false_close_apply_frames": sum(ep["false_close_apply_frames"] for ep in episodes),
        "false_hold_close_frames": sum(ep["false_hold_close_frames"] for ep in episodes),
        "missing_gripper_fsm_state_frames": sum(ep["missing_gripper_fsm_state_frames"] for ep in episodes),
        "internal_close_veto_ready_diag_frames": sum(
            ep["internal_close_veto_ready_diag_frames"] for ep in episodes
        ),
        "ready_prob_peak_max": max((ep["ready_prob_peak"] for ep in episodes), default=0.0),
    }
    if total["teacher_ready_frames"]:
        total["teacher_pred_ready_overlap_rate"] = (
            total["teacher_pred_ready_overlap_frames"] / total["teacher_ready_frames"]
        )
        total["teacher_runtime_handoff_ready_overlap_rate"] = (
            total["teacher_runtime_handoff_ready_overlap_frames"] / total["teacher_ready_frames"]
        )
    else:
        total["teacher_pred_ready_overlap_rate"] = 0.0
        total["teacher_runtime_handoff_ready_overlap_rate"] = 0.0
    if total["predicted_ready_frames"]:
        total["pred_false_ready_rate"] = total["pred_false_ready_frames"] / total["predicted_ready_frames"]
    else:
        total["pred_false_ready_rate"] = 0.0
    if total["runtime_handoff_ready_frames"]:
        total["false_close_apply_rate"] = (
            total["false_close_apply_frames"] / total["runtime_handoff_ready_frames"]
        )
    else:
        total["false_close_apply_rate"] = 0.0
    if total["planner_close_frames"]:
        total["false_hold_close_rate"] = total["false_hold_close_frames"] / total["planner_close_frames"]
    else:
        total["false_hold_close_rate"] = 0.0
    report = {"trace_dir": args.trace_dir, "summary": total, "episodes": episodes}
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
