"""
build_alignment_complete_fire_dataset.py

Build a minimal fire-only residual dataset from:

- teacher final aligned band (positive)
- student false-fire band (negative)

This keeps the trigger semantics intentionally narrow:
the classifier only learns whether the current state has entered the
final alignment-complete firing band.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_trace(trace_json: Path, mode_name: str):
    with trace_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if mode_name not in data:
        raise KeyError(f"Mode `{mode_name}` not found in {trace_json}")
    return {int(r["step"]): r for r in data[mode_name]["curves"]}


def load_npz(npz_path: Path):
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def first_successful_close(rows_by_step, close_threshold: float, lift_threshold: float, lift_horizon: int):
    steps = sorted(rows_by_step.keys())
    prev_open = True
    for step in steps:
        row = rows_by_step[step]
        is_open = float(row.get("exec_gripper_raw", 1.0)) >= close_threshold
        if prev_open and not is_open:
            obj = row.get("live_object_pose")
            if obj is None:
                prev_open = is_open
                continue
            base_z = float(obj[2])
            max_future_z = base_z
            reopen = False
            for fut_step in range(step + 1, min(step + int(lift_horizon), steps[-1]) + 1):
                fut = rows_by_step.get(fut_step)
                if fut is None:
                    continue
                fut_open = float(fut.get("exec_gripper_raw", 1.0)) >= close_threshold
                if fut_open:
                    reopen = True
                    break
                fut_obj = fut.get("live_object_pose")
                if fut_obj is not None:
                    max_future_z = max(max_future_z, float(fut_obj[2]))
            if (not reopen) and (max_future_z - base_z >= float(lift_threshold)):
                return int(step)
        prev_open = is_open
    raise RuntimeError("No successful teacher close-with-lift event found.")


def load_false_fire_steps(audit_json: Path):
    with audit_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mode = data["trigger_only_basin"]
    events = mode.get("events", [])
    steps = []
    for event in events:
        if bool(event.get("air_close", False)) and bool(event.get("close_before_basin", False)):
            steps.append(int(event["close_step"]))
    if not steps:
        raise RuntimeError("No student false-fire close steps found in audit.")
    return sorted(set(steps))


def build_step_index(npz_rows: dict):
    step_arr = np.asarray(npz_rows["step_idx"], dtype=np.int64)
    idx = {}
    for i, step in enumerate(step_arr.tolist()):
        idx.setdefault(int(step), []).append(i)
    return idx


def build_fire_row(npz_rows: dict, row_idx: int, label: int):
    chunk_steps = np.asarray(npz_rows.get("chunk_step_idx", npz_rows["step_idx"]), dtype=np.int64)
    phase_ids = np.asarray(npz_rows.get("phase_id", np.zeros_like(npz_rows["step_idx"])), dtype=np.int64)
    phase_age = np.asarray(npz_rows.get("phase_age", np.zeros_like(npz_rows["step_idx"], dtype=np.float32)), dtype=np.float32)
    since_replan = np.asarray(
        npz_rows.get("steps_since_last_replan", np.zeros_like(npz_rows["step_idx"], dtype=np.float32)),
        dtype=np.float32,
    )
    planner_close_intent = np.asarray(
        npz_rows.get("planner_close_intent", np.zeros_like(npz_rows["step_idx"], dtype=np.float32)),
        dtype=np.float32,
    )
    current_xy = np.asarray(npz_rows.get("current_basin_xy", np.zeros_like(npz_rows["step_idx"], dtype=np.float32)), dtype=np.float32)
    current_z = np.asarray(npz_rows.get("current_basin_z", np.zeros_like(npz_rows["step_idx"], dtype=np.float32)), dtype=np.float32)
    current_yaw = np.asarray(npz_rows.get("current_basin_yaw", np.zeros_like(npz_rows["step_idx"], dtype=np.float32)), dtype=np.float32)
    current_dist = np.asarray(npz_rows.get("current_basin_distance", np.zeros_like(npz_rows["step_idx"], dtype=np.float32)), dtype=np.float32)

    return {
        "wrist_depth": np.asarray(npz_rows["wrist_depth"][row_idx], dtype=np.float32),
        "ft_hist": np.zeros((32, 6), dtype=np.float32),
        "proprio": np.asarray(npz_rows["proprio"][row_idx], dtype=np.float32),
        "base_action": np.asarray(npz_rows["base_action"][row_idx], dtype=np.float32),
        "gripper_context": np.asarray(npz_rows["gripper_context"][row_idx], dtype=np.float32),
        "interaction_role": np.asarray(0, dtype=np.int64),
        "step_idx": np.asarray(int(chunk_steps[row_idx]) % 8, dtype=np.int64),
        "delta_target": np.zeros(6, dtype=np.float32),
        "delta_align_target": np.zeros(6, dtype=np.float32),
        "delta_basin_target": np.zeros(6, dtype=np.float32),
        "contact_mask": np.asarray(1, dtype=np.int64),
        "phase_label": np.asarray(1, dtype=np.int64),
        "phase_id": np.asarray(int(phase_ids[row_idx]), dtype=np.int64),
        "phase_age": np.asarray(float(phase_age[row_idx]), dtype=np.float32),
        "steps_since_last_replan": np.asarray(float(since_replan[row_idx]), dtype=np.float32),
        "stage_role": np.asarray(0, dtype=np.int64),
        "failure_mode": np.asarray(0, dtype=np.int64),
        "transition_flag": np.asarray(0, dtype=np.int64),
        "subgoal_progress": np.asarray(0.0, dtype=np.float32),
        "rollout_gripper_open": np.asarray(float(npz_rows["rollout_gripper_open"][row_idx]), dtype=np.float32),
        "depth_proximity": np.asarray(float(npz_rows["depth_proximity"][row_idx]), dtype=np.float32),
        "planner_close_intent": np.asarray(float(planner_close_intent[row_idx]), dtype=np.float32),
        "planner_close_intent_strength": np.asarray(
            float(np.clip(1.0 - float(npz_rows["gripper_context"][row_idx][1]), 0.0, 1.0)),
            dtype=np.float32,
        ),
        "readiness_label": np.asarray(float(label), dtype=np.float32),
        "basin_positive": np.asarray(float(label), dtype=np.float32),
        "basin_distance": np.asarray(float(current_dist[row_idx]), dtype=np.float32),
        "hold_label": np.asarray(-1.0, dtype=np.float32),
        "negative_reason": np.asarray(-1 if label == 1 else 0, dtype=np.int64),
        "frames_to_expert_close": np.asarray(-1, dtype=np.int64),
        "frames_to_reference_trigger": np.asarray(-1, dtype=np.int64),
        "post_close_stability_proxy": np.asarray(0.0, dtype=np.float32),
        "grasp_lift_proxy": np.asarray(0.0, dtype=np.float32),
        "reopen_within_horizon": np.asarray(0.0, dtype=np.float32),
        "reopen_after_trigger": np.asarray(0.0, dtype=np.float32),
        "no_progress_after_trigger": np.asarray(0.0, dtype=np.float32),
        "invalid_after_trigger": np.asarray(0.0, dtype=np.float32),
        "gripper_state_target": np.asarray(1 if label == 1 else 0, dtype=np.int64),
        "ready_to_close": np.asarray(float(label), dtype=np.float32),
        "planner_close_too_early": np.asarray(float(label == 0), dtype=np.float32),
        "expert_hold_after_close": np.asarray(0.0, dtype=np.float32),
        "current_basin_xy": np.asarray(float(current_xy[row_idx]), dtype=np.float32),
        "current_basin_z": np.asarray(float(current_z[row_idx]), dtype=np.float32),
        "current_basin_yaw": np.asarray(float(current_yaw[row_idx]), dtype=np.float32),
        "rollout_step": np.asarray(int(npz_rows["step_idx"][row_idx]), dtype=np.int64),
    }


def count_values(values):
    if not values:
        return {}
    arr = np.asarray(values)
    uniq, counts = np.unique(arr, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def main():
    parser = argparse.ArgumentParser(description="Build a fire-only alignment-complete classifier dataset.")
    parser.add_argument("--teacher_trace_json", type=str, required=True)
    parser.add_argument("--teacher_trigger_states_npz", type=str, required=True)
    parser.add_argument("--student_trace_json", type=str, required=True)
    parser.add_argument("--student_trigger_states_npz", type=str, required=True)
    parser.add_argument("--student_audit_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--close_threshold", type=float, default=0.5)
    parser.add_argument("--lift_threshold", type=float, default=0.03)
    parser.add_argument("--lift_horizon", type=int, default=24)
    parser.add_argument("--teacher_suffix_steps", type=int, default=2)
    parser.add_argument("--student_preclose_offsets", type=str, default="-1,0")
    parser.add_argument("--teacher_repeat", type=int, default=32)
    parser.add_argument("--student_repeat", type=int, default=8)
    args = parser.parse_args()

    teacher_rows = load_trace(Path(args.teacher_trace_json), "oracle_live_target_alignment")
    teacher_npz = load_npz(Path(args.teacher_trigger_states_npz))
    teacher_close_step = first_successful_close(
        teacher_rows,
        close_threshold=float(args.close_threshold),
        lift_threshold=float(args.lift_threshold),
        lift_horizon=int(args.lift_horizon),
    )
    teacher_positive_steps = list(range(int(teacher_close_step) - int(args.teacher_suffix_steps), int(teacher_close_step)))
    teacher_step_index = build_step_index(teacher_npz)

    student_rows = load_trace(Path(args.student_trace_json), "trigger_only_basin")
    _ = student_rows  # symmetry, and future metadata
    student_npz = load_npz(Path(args.student_trigger_states_npz))
    student_false_fire_steps = load_false_fire_steps(Path(args.student_audit_json))
    offsets = [int(x.strip()) for x in str(args.student_preclose_offsets).split(",") if x.strip()]
    student_negative_steps = sorted(
        {
            int(step + offset)
            for step in student_false_fire_steps
            for offset in offsets
            if int(step + offset) >= 0
        }
    )
    student_step_index = build_step_index(student_npz)

    teacher_selected = []
    for step in teacher_positive_steps:
        if step not in teacher_step_index:
            raise RuntimeError(f"Teacher step {step} not found in trigger states npz.")
        teacher_selected.append(teacher_step_index[step][0])

    student_selected = []
    for step in student_negative_steps:
        if step not in student_step_index:
            continue
        student_selected.append(student_step_index[step][0])
    if not student_selected:
        raise RuntimeError("No student false-fire rows selected.")

    rows = []
    for idx in teacher_selected:
        row = build_fire_row(teacher_npz, idx, label=1)
        rows.extend([row] * int(args.teacher_repeat))
    for idx in student_selected:
        row = build_fire_row(student_npz, idx, label=0)
        rows.extend([row] * int(args.student_repeat))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack([np.asarray(r[k]) for r in rows], axis=0) for k in rows[0].keys()}
    np.savez_compressed(output_dir / "residual_shard_0000.npz", **arrays)

    teacher_xy = [float(teacher_npz["current_basin_xy"][idx]) for idx in teacher_selected]
    teacher_z = [float(teacher_npz["current_basin_z"][idx]) for idx in teacher_selected]
    teacher_depth = [float(teacher_npz["depth_proximity"][idx]) for idx in teacher_selected]
    teacher_phase_age = [float(teacher_npz["phase_age"][idx]) for idx in teacher_selected]

    student_xy = [float(student_npz["current_basin_xy"][idx]) for idx in student_selected]
    student_z = [float(student_npz["current_basin_z"][idx]) for idx in student_selected]
    student_depth = [float(student_npz["depth_proximity"][idx]) for idx in student_selected]
    student_phase_age = [float(student_npz["phase_age"][idx]) for idx in student_selected]

    meta = {
        "dataset_view": "fire_trigger_teacher_aligned_vs_student_false_fire",
        "split_role": "fire",
        "num_samples": int(len(rows)),
        "allow_close_count": int(sum(int(float(r["readiness_label"]) > 0.5) for r in rows)),
        "hold_positive_count": 0,
        "gripper_state_counts": count_values([int(r["gripper_state_target"]) for r in rows]),
        "teacher_close_step": int(teacher_close_step),
        "teacher_positive_steps": [int(x) for x in teacher_positive_steps],
        "student_false_fire_close_steps": [int(x) for x in student_false_fire_steps],
        "student_negative_steps": [int(x) for x in student_negative_steps],
        "teacher_repeat": int(args.teacher_repeat),
        "student_repeat": int(args.student_repeat),
        "teacher_feature_summary": {
            "xy_mean": float(np.mean(teacher_xy)),
            "z_mean": float(np.mean(teacher_z)),
            "depth_mean": float(np.mean(teacher_depth)),
            "phase_age_mean": float(np.mean(teacher_phase_age)),
        },
        "student_feature_summary": {
            "xy_mean": float(np.mean(student_xy)),
            "z_mean": float(np.mean(student_z)),
            "depth_mean": float(np.mean(student_depth)),
            "phase_age_mean": float(np.mean(student_phase_age)),
        },
    }
    (output_dir / "residual_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
