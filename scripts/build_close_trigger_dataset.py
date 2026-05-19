"""
build_close_trigger_dataset.py

Build a geometry-conditioned close-trigger dataset from fixed-scene trigger
support states and rollout traces.

V2 semantics:
- open states supervise `block open` vs `allow close`
- closed states supervise `hold closed` vs `do not hold`

This avoids collapsing every failure mode into a single ready/not-ready label,
which previously encouraged an always-open controller.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_trace(trace_json: Path, mode_name: str):
    with trace_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(r["step"]): r for r in data[mode_name]["curves"]}


def next_close_step(curves_by_step, step_idx: int, close_threshold: float, horizon: int):
    max_step = max(curves_by_step.keys())
    end_step = min(max_step, int(step_idx) + int(horizon))
    for step in range(int(step_idx) + 1, end_step + 1):
        row = curves_by_step.get(step)
        if row is None:
            continue
        if float(row.get("exec_gripper_raw", 1.0)) < close_threshold:
            return step
    return -1


def next_reopen_step(curves_by_step, close_step: int, close_threshold: float, horizon: int):
    if close_step < 0:
        return -1
    max_step = max(curves_by_step.keys())
    end_step = min(max_step, int(close_step) + int(horizon))
    for step in range(int(close_step) + 1, end_step + 1):
        row = curves_by_step.get(step)
        if row is None:
            continue
        if float(row.get("exec_gripper_raw", 1.0)) >= close_threshold:
            return step
    return -1


def min_future_basin(curves_by_step, step_idx: int, horizon: int):
    max_step = max(curves_by_step.keys())
    end_step = min(max_step, int(step_idx) + int(horizon))
    best_step = int(step_idx)
    best_val = float(curves_by_step[int(step_idx)]["basin_distance"])
    for step in range(int(step_idx), end_step + 1):
        row = curves_by_step.get(step)
        if row is None:
            continue
        val = float(row["basin_distance"])
        if val < best_val:
            best_val = val
            best_step = step
    return best_step, best_val


def count_values(values):
    if len(values) == 0:
        return {}
    arr = np.asarray(values)
    return {int(v): int((arr == v).sum()) for v in np.unique(arr)}


def write_dataset(output_dir: Path, rows, meta: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return False
    keys = list(rows[0].keys())
    stacked = {k: np.stack([row[k] for row in rows], axis=0) for k in keys}
    np.savez_compressed(output_dir / "residual_shard_0000.npz", **stacked)
    (output_dir / "residual_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True


def geom_ok(value: float, threshold: float) -> bool:
    if threshold < 0:
        return True
    if np.isnan(value):
        return False
    return float(value) <= float(threshold)


def main():
    parser = argparse.ArgumentParser(description="Build close-trigger residual dataset from trace support states.")
    parser.add_argument("--trace_json", type=str, required=True)
    parser.add_argument("--trigger_states_npz", type=str, required=True)
    parser.add_argument("--mode_name", type=str, default="pose_alignment_only_basin")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--close_threshold", type=float, default=0.5)
    parser.add_argument("--trigger_horizon", type=int, default=8)
    parser.add_argument("--reopen_horizon", type=int, default=8)
    parser.add_argument("--improve_horizon", type=int, default=20)
    parser.add_argument("--improve_eps", type=float, default=0.25)
    parser.add_argument("--allow_close_basin_threshold", type=float, default=2.4)
    parser.add_argument("--allow_close_z_threshold", type=float, default=0.03)
    parser.add_argument("--allow_close_xy_threshold", type=float, default=0.010)
    parser.add_argument("--allow_close_yaw_threshold", type=float, default=-1.0)
    parser.add_argument("--hold_basin_threshold", type=float, default=3.0)
    parser.add_argument("--hold_z_threshold", type=float, default=0.05)
    parser.add_argument("--hold_xy_threshold", type=float, default=0.014)
    parser.add_argument("--hold_yaw_threshold", type=float, default=-1.0)
    parser.add_argument("--write_split_datasets", action="store_true", default=True)
    parser.add_argument("--no_split_datasets", dest="write_split_datasets", action="store_false")
    args = parser.parse_args()

    mode_index = 0 if args.mode_name == "planner_only" else 1
    curves_by_step = load_trace(Path(args.trace_json), args.mode_name)
    states = np.load(args.trigger_states_npz)
    if "mode_index" in states.files:
        mask_mode = states["mode_index"].astype(np.int64) == mode_index
    else:
        mask_mode = np.ones(states["step_idx"].shape[0], dtype=bool)

    ft_hist_shape = (32, 6)
    rows = []
    negative_reason_counts = {}
    close_too_early_count = 0
    open_state_count = 0
    closed_state_count = 0
    allow_close_count = 0
    hold_positive_count = 0

    for idx in np.where(mask_mode)[0]:
        step = int(states["step_idx"][idx])
        row = curves_by_step.get(step)
        if row is None:
            continue

        current_basin = float(states["current_basin_distance"][idx])
        current_xy = float(states["current_basin_xy"][idx]) if "current_basin_xy" in states.files else float("nan")
        current_z = float(states["current_basin_z"][idx]) if "current_basin_z" in states.files else float("nan")
        current_yaw = float(states["current_basin_yaw"][idx]) if "current_basin_yaw" in states.files else float("nan")
        rollout_gripper_open = float(states["rollout_gripper_open"][idx])
        is_open = rollout_gripper_open >= float(args.close_threshold)

        close_step = next_close_step(curves_by_step, step, args.close_threshold, args.trigger_horizon)
        reopen_step = next_reopen_step(curves_by_step, close_step, args.close_threshold, args.reopen_horizon)
        future_best_step, future_best_basin = min_future_basin(curves_by_step, step, args.improve_horizon)

        close_before_allow = False
        if close_step >= 0:
            close_before_allow = float(curves_by_step[close_step]["basin_distance"]) > float(args.allow_close_basin_threshold)
        planner_can_still_approach = future_best_basin < (current_basin - float(args.improve_eps))
        reopen_after_close = reopen_step >= 0

        allow_close_geom = bool(
            geom_ok(current_basin, args.allow_close_basin_threshold)
            and geom_ok(current_z, args.allow_close_z_threshold)
            and geom_ok(current_xy, args.allow_close_xy_threshold)
            and geom_ok(current_yaw, args.allow_close_yaw_threshold)
        )
        hold_geom = bool(
            geom_ok(current_basin, args.hold_basin_threshold)
            and geom_ok(current_z, args.hold_z_threshold)
            and geom_ok(current_xy, args.hold_xy_threshold)
            and geom_ok(current_yaw, args.hold_yaw_threshold)
        )
        hold_positive = bool((not is_open) and hold_geom and (not reopen_after_close) and (not planner_can_still_approach))

        if is_open:
            open_state_count += 1
        else:
            closed_state_count += 1

        if is_open and allow_close_geom:
            readiness_label = 1.0
            basin_positive = 1.0
            hold_label = -1.0
            gripper_state_target = 1  # allow close
            negative_reason = -1
            allow_close_count += 1
        elif (not is_open) and hold_positive:
            readiness_label = -1.0
            basin_positive = -1.0
            hold_label = 1.0
            gripper_state_target = 2  # hold closed
            negative_reason = -1
            hold_positive_count += 1
        elif close_before_allow and planner_can_still_approach:
            readiness_label = 0.0 if is_open else -1.0
            basin_positive = 0.0 if is_open else -1.0
            hold_label = 0.0 if not is_open else -1.0
            gripper_state_target = 0
            negative_reason = 0  # early_close_can_improve_later
        elif reopen_after_close:
            readiness_label = 0.0 if is_open else -1.0
            basin_positive = 0.0 if is_open else -1.0
            hold_label = 0.0 if not is_open else -1.0
            gripper_state_target = 0
            negative_reason = 1  # reopen_after_close
        elif not is_open:
            readiness_label = -1.0
            basin_positive = -1.0
            hold_label = 0.0
            gripper_state_target = 0
            negative_reason = 3  # closed_but_not_holdable
        else:
            readiness_label = 0.0
            basin_positive = 0.0
            hold_label = -1.0
            gripper_state_target = 0
            negative_reason = 2  # not_ready_far_or_unstable

        close_too_early = float(close_before_allow and planner_can_still_approach)
        close_too_early_count += int(close_too_early > 0.5)
        negative_reason_counts[int(negative_reason)] = negative_reason_counts.get(int(negative_reason), 0) + 1

        frames_to_trigger = int(close_step - step) if close_step >= 0 else -1
        post_close_stability = float(close_step >= 0 and not reopen_after_close)
        rows.append(
            {
                "wrist_depth": np.asarray(states["wrist_depth"][idx], dtype=np.float32),
                "ft_hist": np.zeros(ft_hist_shape, dtype=np.float32),
                "proprio": np.asarray(states["proprio"][idx], dtype=np.float32),
                "base_action": np.asarray(states["base_action"][idx], dtype=np.float32),
                "gripper_context": np.asarray(states["gripper_context"][idx], dtype=np.float32),
                "interaction_role": np.asarray(0, dtype=np.int64),
                "step_idx": np.asarray(step % 8, dtype=np.int64),
                "delta_target": np.zeros(6, dtype=np.float32),
                "delta_align_target": np.zeros(6, dtype=np.float32),
                "delta_basin_target": np.zeros(6, dtype=np.float32),
                "contact_mask": np.asarray(1, dtype=np.int64),
                "phase_label": np.asarray(1, dtype=np.int64),
                "phase_id": np.asarray(int(states["phase_id"][idx]), dtype=np.int64),
                "phase_age": np.asarray(float(states["phase_age"][idx]), dtype=np.float32),
                "steps_since_last_replan": np.asarray(float(states["steps_since_last_replan"][idx]), dtype=np.float32),
                "stage_role": np.asarray(0, dtype=np.int64),
                "failure_mode": np.asarray(0, dtype=np.int64),
                "transition_flag": np.asarray(0, dtype=np.int64),
                "subgoal_progress": np.asarray(0.0, dtype=np.float32),
                "rollout_gripper_open": np.asarray(rollout_gripper_open, dtype=np.float32),
                "depth_proximity": np.asarray(float(states["depth_proximity"][idx]), dtype=np.float32),
                "planner_close_intent": np.asarray(float(states["planner_close_intent"][idx]), dtype=np.float32),
                "planner_close_intent_strength": np.asarray(float(np.clip(1.0 - states["gripper_context"][idx][1], 0.0, 1.0)), dtype=np.float32),
                "readiness_label": np.asarray(float(readiness_label), dtype=np.float32),
                "basin_positive": np.asarray(float(basin_positive), dtype=np.float32),
                "basin_distance": np.asarray(current_basin, dtype=np.float32),
                "hold_label": np.asarray(float(hold_label), dtype=np.float32),
                "negative_reason": np.asarray(int(negative_reason), dtype=np.int64),
                "frames_to_expert_close": np.asarray(frames_to_trigger, dtype=np.int64),
                "frames_to_reference_trigger": np.asarray(frames_to_trigger, dtype=np.int64),
                "post_close_stability_proxy": np.asarray(post_close_stability, dtype=np.float32),
                "grasp_lift_proxy": np.asarray(0.0, dtype=np.float32),
                "reopen_within_horizon": np.asarray(float(reopen_after_close), dtype=np.float32),
                "reopen_after_trigger": np.asarray(float(reopen_after_close), dtype=np.float32),
                "no_progress_after_trigger": np.asarray(float(not planner_can_still_approach), dtype=np.float32),
                "invalid_after_trigger": np.asarray(0.0, dtype=np.float32),
                "gripper_state_target": np.asarray(int(gripper_state_target), dtype=np.int64),
                "ready_to_close": np.asarray(float(readiness_label), dtype=np.float32),
                "planner_close_too_early": np.asarray(close_too_early, dtype=np.float32),
                "expert_hold_after_close": np.asarray(float(hold_positive), dtype=np.float32),
            }
        )

    if not rows:
        raise RuntimeError("No rows matched the requested mode / trigger-state inputs.")

    keys = list(rows[0].keys())
    stacked = {k: np.stack([row[k] for row in rows], axis=0) for k in keys}

    readiness = stacked["readiness_label"][stacked["readiness_label"] >= 0].astype(np.int64)
    hold = stacked["hold_label"][stacked["hold_label"] >= 0].astype(np.int64)
    meta = {
        "dataset_view": "close_trigger_trace_view_v2",
        "mode_name": args.mode_name,
        "num_samples": int(len(rows)),
        "open_state_count": int(open_state_count),
        "closed_state_count": int(closed_state_count),
        "allow_close_count": int(allow_close_count),
        "hold_positive_count": int(hold_positive_count),
        "readiness_counts": count_values(readiness),
        "hold_counts": count_values(hold),
        "gripper_state_counts": count_values(stacked["gripper_state_target"][stacked["gripper_state_target"] >= 0].astype(np.int64)),
        "negative_reason_counts": negative_reason_counts,
        "planner_close_too_early_fraction": float(close_too_early_count / max(len(rows), 1)),
        "allow_close_basin_threshold": float(args.allow_close_basin_threshold),
        "allow_close_z_threshold": float(args.allow_close_z_threshold),
        "allow_close_xy_threshold": float(args.allow_close_xy_threshold),
        "allow_close_yaw_threshold": float(args.allow_close_yaw_threshold),
        "hold_basin_threshold": float(args.hold_basin_threshold),
        "hold_z_threshold": float(args.hold_z_threshold),
        "hold_xy_threshold": float(args.hold_xy_threshold),
        "hold_yaw_threshold": float(args.hold_yaw_threshold),
        "trigger_horizon": int(args.trigger_horizon),
        "reopen_horizon": int(args.reopen_horizon),
        "improve_horizon": int(args.improve_horizon),
        "improve_eps": float(args.improve_eps),
    }
    output_dir = Path(args.output_dir)
    write_dataset(output_dir, rows, meta)

    if args.write_split_datasets:
        fire_rows = [row for row in rows if float(row["readiness_label"]) >= 0.0]
        hold_rows = [row for row in rows if float(row["hold_label"]) >= 0.0]
        fire_dir = output_dir / "fire_trigger"
        hold_dir = output_dir / "hold_trigger"
        fire_meta = {
            **meta,
            "dataset_view": f"{meta['dataset_view']}_fire",
            "num_samples": int(len(fire_rows)),
            "readiness_counts": count_values([int(float(row["readiness_label"])) for row in fire_rows]),
            "hold_counts": {},
            "gripper_state_counts": count_values([int(row["gripper_state_target"]) for row in fire_rows]),
            "split_role": "fire",
        }
        hold_meta = {
            **meta,
            "dataset_view": f"{meta['dataset_view']}_hold",
            "num_samples": int(len(hold_rows)),
            "readiness_counts": {},
            "hold_counts": count_values([int(float(row["hold_label"])) for row in hold_rows]),
            "gripper_state_counts": count_values([int(row["gripper_state_target"]) for row in hold_rows]),
            "split_role": "hold",
        }
        write_dataset(fire_dir, fire_rows, fire_meta)
        write_dataset(hold_dir, hold_rows, hold_meta)
        meta["split_outputs"] = {
            "fire_dir": str(fire_dir),
            "hold_dir": str(hold_dir),
            "fire_num_samples": int(len(fire_rows)),
            "hold_num_samples": int(len(hold_rows)),
        }
        (output_dir / "residual_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
