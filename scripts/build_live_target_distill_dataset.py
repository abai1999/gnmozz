"""
build_live_target_distill_dataset.py

Turn a successful `oracle_live_target_alignment` rollout into train-ready
distillation datasets for:

- pose candidate scorer distillation
- geometry-conditioned close trigger distillation

This script keeps the representation task-agnostic:
- it never uses object names in labels
- it relies on live target geometry + successful close/lift outcome
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_trace(trace_json: Path, mode_name: str):
    with trace_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if mode_name not in data:
        raise KeyError(f"Mode `{mode_name}` not found in {trace_json}.")
    rows = data[mode_name]["curves"]
    return rows, {int(r["step"]): r for r in rows}


def load_npz_rows(npz_path: Path):
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def filter_npz_rows(data: dict, mask: np.ndarray) -> dict:
    out = {}
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.shape[:1] == mask.shape:
            out[k] = arr[mask]
        else:
            out[k] = arr
    return out


def save_npz_dict(data: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)


def first_successful_close(rows, close_threshold: float, lift_threshold: float, lift_horizon: int):
    prev_open = True
    for idx, row in enumerate(rows):
        is_open = float(row.get("exec_gripper_raw", 1.0)) > close_threshold
        if prev_open and not is_open:
            base_obj = row.get("live_object_pose")
            if base_obj is None:
                prev_open = is_open
                continue
            base_z = float(base_obj[2])
            max_future_z = base_z
            reopen = False
            end = min(len(rows), idx + int(lift_horizon) + 1)
            for fut in rows[idx + 1:end]:
                fut_open = float(fut.get("exec_gripper_raw", 1.0)) > close_threshold
                if fut_open:
                    reopen = True
                    break
                fut_obj = fut.get("live_object_pose")
                if fut_obj is not None:
                    max_future_z = max(max_future_z, float(fut_obj[2]))
            if (not reopen) and (max_future_z - base_z >= float(lift_threshold)):
                return int(row["step"]), idx, float(base_z), float(max_future_z)
        prev_open = is_open
    return -1, -1, float("nan"), float("nan")


def select_strong_fire_positive_steps(
    rows,
    *,
    close_step: int,
    close_threshold: float,
    positive_window: int,
    alignment_band_max_steps: int,
    alignment_xy_threshold: float,
    alignment_z_threshold: float,
    alignment_distance_threshold: float,
):
    if close_step < 0:
        return set(), {}

    open_candidates = []
    start_step = max(0, int(close_step) - int(positive_window))
    for row in rows[start_step : int(close_step) + 1]:
        step = int(row["step"])
        exec_open = float(row.get("exec_gripper_raw", 1.0)) >= float(close_threshold)
        if not exec_open:
            continue
        live_dist = float(row.get("live_target_distance", np.inf))
        live_xy = float(row.get("live_target_xy", np.inf))
        live_z = float(row.get("live_target_z", np.inf))
        open_candidates.append((step, live_dist, live_xy, live_z))

    if not open_candidates:
        return set(), {
            "near_best_step": -1,
            "num_open_candidates": 0,
            "num_fire_positives": 0,
        }

    near_best_step, near_best_dist, near_best_xy, near_best_z = min(open_candidates, key=lambda x: x[1])
    candidate_by_step = {
        int(step): (float(live_dist), float(live_xy), float(live_z))
        for step, live_dist, live_xy, live_z in open_candidates
    }
    positive_steps = []
    last_open_step = int(close_step) - 1
    for step in range(last_open_step, max(last_open_step - int(alignment_band_max_steps), -1), -1):
        stats = candidate_by_step.get(int(step))
        if stats is None:
            break
        live_dist, live_xy, live_z = stats
        if float(alignment_distance_threshold) >= 0.0 and float(live_dist) > float(alignment_distance_threshold):
            break
        if float(alignment_xy_threshold) >= 0.0 and float(live_xy) > float(alignment_xy_threshold):
            break
        if float(alignment_z_threshold) >= 0.0 and float(live_z) > float(alignment_z_threshold):
            break
        positive_steps.append(int(step))
    positive_steps = set(sorted(positive_steps))

    return positive_steps, {
        "near_best_step": int(near_best_step),
        "near_best_live_target_distance": float(near_best_dist),
        "near_best_live_target_xy": float(near_best_xy),
        "near_best_live_target_z": float(near_best_z),
        "num_open_candidates": int(len(open_candidates)),
        "num_fire_positives": int(len(positive_steps)),
        "fire_positive_band_steps": [int(x) for x in sorted(positive_steps)],
    }


def build_trigger_rows(
    trigger_states: dict,
    rows_by_step: dict,
    *,
    close_step: int,
    close_threshold: float,
    positive_steps=None,
):
    out_rows = []
    allow_close_count = 0
    hold_count = 0
    block_count = 0
    if positive_steps is None:
        positive_steps = set()

    step_arr = trigger_states["step_idx"].astype(np.int64)
    mode_arr = trigger_states["mode_index"].astype(np.int64)
    mode_mask = mode_arr == 2

    for idx in np.where(mode_mask)[0]:
        step = int(step_arr[idx])
        row = rows_by_step.get(step)
        if row is None:
            continue
        gripper_open = float(trigger_states["rollout_gripper_open"][idx])
        is_open = gripper_open >= float(close_threshold)
        current_basin = float(trigger_states["current_basin_distance"][idx])
        current_xy = float(trigger_states["current_basin_xy"][idx])
        current_z = float(trigger_states["current_basin_z"][idx])
        current_yaw = float(trigger_states["current_basin_yaw"][idx])

        if is_open:
            if int(step) in positive_steps:
                readiness_label = 1.0
                basin_positive = 1.0
                hold_label = -1.0
                gripper_state_target = 1  # allow_close
                negative_reason = -1
                allow_close_count += 1
            else:
                readiness_label = 0.0
                basin_positive = 0.0
                hold_label = -1.0
                gripper_state_target = 0  # block_open
                negative_reason = 0
                block_count += 1
        else:
            if close_step >= 0 and int(step) >= int(close_step):
                readiness_label = -1.0
                basin_positive = -1.0
                hold_label = 1.0
                gripper_state_target = 2  # hold_closed
                negative_reason = -1
                hold_count += 1
            else:
                readiness_label = -1.0
                basin_positive = -1.0
                hold_label = 0.0
                gripper_state_target = 0
                negative_reason = 3
                block_count += 1

        out_rows.append(
            {
                "wrist_depth": np.asarray(trigger_states["wrist_depth"][idx], dtype=np.float32),
                "ft_hist": np.zeros((32, 6), dtype=np.float32),
                "proprio": np.asarray(trigger_states["proprio"][idx], dtype=np.float32),
                "base_action": np.asarray(trigger_states["base_action"][idx], dtype=np.float32),
                "gripper_context": np.asarray(trigger_states["gripper_context"][idx], dtype=np.float32),
                "interaction_role": np.asarray(0, dtype=np.int64),
                "step_idx": np.asarray(int(trigger_states.get("chunk_step_idx", trigger_states["step_idx"])[idx]) % 8, dtype=np.int64),
                "delta_target": np.zeros(6, dtype=np.float32),
                "delta_align_target": np.zeros(6, dtype=np.float32),
                "delta_basin_target": np.zeros(6, dtype=np.float32),
                "contact_mask": np.asarray(1, dtype=np.int64),
                "phase_label": np.asarray(1, dtype=np.int64),
                "phase_id": np.asarray(int(trigger_states["phase_id"][idx]), dtype=np.int64),
                "phase_age": np.asarray(float(trigger_states["phase_age"][idx]), dtype=np.float32),
                "steps_since_last_replan": np.asarray(float(trigger_states["steps_since_last_replan"][idx]), dtype=np.float32),
                "stage_role": np.asarray(0, dtype=np.int64),
                "failure_mode": np.asarray(0, dtype=np.int64),
                "transition_flag": np.asarray(0, dtype=np.int64),
                "subgoal_progress": np.asarray(0.0, dtype=np.float32),
                "rollout_gripper_open": np.asarray(gripper_open, dtype=np.float32),
                "depth_proximity": np.asarray(float(trigger_states["depth_proximity"][idx]), dtype=np.float32),
                "planner_close_intent": np.asarray(float(trigger_states["planner_close_intent"][idx]), dtype=np.float32),
                "planner_close_intent_strength": np.asarray(float(np.clip(1.0 - trigger_states["gripper_context"][idx][1], 0.0, 1.0)), dtype=np.float32),
                "readiness_label": np.asarray(float(readiness_label), dtype=np.float32),
                "basin_positive": np.asarray(float(basin_positive), dtype=np.float32),
                "basin_distance": np.asarray(current_basin, dtype=np.float32),
                "hold_label": np.asarray(float(hold_label), dtype=np.float32),
                "negative_reason": np.asarray(int(negative_reason), dtype=np.int64),
                "frames_to_expert_close": np.asarray(int(close_step - step) if close_step >= 0 else -1, dtype=np.int64),
                "frames_to_reference_trigger": np.asarray(int(close_step - step) if close_step >= 0 else -1, dtype=np.int64),
                "post_close_stability_proxy": np.asarray(float((close_step >= 0) and (step >= close_step)), dtype=np.float32),
                "grasp_lift_proxy": np.asarray(float((close_step >= 0) and (step >= close_step)), dtype=np.float32),
                "reopen_within_horizon": np.asarray(0.0, dtype=np.float32),
                "reopen_after_trigger": np.asarray(0.0, dtype=np.float32),
                "no_progress_after_trigger": np.asarray(0.0, dtype=np.float32),
                "invalid_after_trigger": np.asarray(0.0, dtype=np.float32),
                "gripper_state_target": np.asarray(int(gripper_state_target), dtype=np.int64),
                "ready_to_close": np.asarray(float(max(readiness_label, 0.0)), dtype=np.float32),
                "planner_close_too_early": np.asarray(0.0, dtype=np.float32),
                "expert_hold_after_close": np.asarray(float(gripper_state_target == 2), dtype=np.float32),
                "current_basin_xy": np.asarray(current_xy, dtype=np.float32),
                "current_basin_z": np.asarray(current_z, dtype=np.float32),
                "current_basin_yaw": np.asarray(current_yaw, dtype=np.float32),
                "rollout_step": np.asarray(step, dtype=np.int64),
            }
        )

    summary = {
        "num_samples": int(len(out_rows)),
        "allow_close_count": int(allow_close_count),
        "hold_positive_count": int(hold_count),
        "block_open_count": int(block_count),
        "gripper_state_counts": {
            int(k): int(v)
            for k, v in zip(
                *np.unique(np.asarray([int(r["gripper_state_target"]) for r in out_rows], dtype=np.int64), return_counts=True)
            )
        } if out_rows else {},
    }
    return out_rows, summary


def write_rows(rows, output_dir: Path, meta: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    arrays = {}
    for key in keys:
        arrays[key] = np.stack([np.asarray(r[key]) for r in rows], axis=0)
    np.savez_compressed(output_dir / "residual_shard_0000.npz", **arrays)
    (output_dir / "residual_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_split_rows(rows, output_dir: Path, meta: dict):
    fire_rows = [row for row in rows if float(row["readiness_label"]) >= 0.0]
    hold_rows = [row for row in rows if float(row["hold_label"]) >= 0.0]
    if fire_rows:
        fire_meta = {
            **meta,
            "dataset_view": "oracle_live_target_distill_fire",
            "split_role": "fire",
            "num_samples": int(len(fire_rows)),
            "allow_close_count": int(sum(int(float(r["readiness_label"]) > 0.5) for r in fire_rows)),
            "hold_positive_count": 0,
            "gripper_state_counts": {
                int(k): int(v)
                for k, v in zip(
                    *np.unique(np.asarray([int(r["gripper_state_target"]) for r in fire_rows], dtype=np.int64), return_counts=True)
                )
            },
        }
        write_rows(fire_rows, output_dir / "fire_trigger", fire_meta)
    if hold_rows:
        hold_meta = {
            **meta,
            "dataset_view": "oracle_live_target_distill_hold",
            "split_role": "hold",
            "num_samples": int(len(hold_rows)),
            "allow_close_count": 0,
            "hold_positive_count": int(sum(int(float(r["hold_label"]) > 0.5) for r in hold_rows)),
            "gripper_state_counts": {
                int(k): int(v)
                for k, v in zip(
                    *np.unique(np.asarray([int(r["gripper_state_target"]) for r in hold_rows], dtype=np.int64), return_counts=True)
                )
            },
        }
        write_rows(hold_rows, output_dir / "hold_trigger", hold_meta)


def main():
    parser = argparse.ArgumentParser(description="Build oracle-live-target distillation datasets from a successful rollout.")
    parser.add_argument("--trace_json", type=str, required=True)
    parser.add_argument("--support_states_npz", type=str, required=True)
    parser.add_argument("--trigger_states_npz", type=str, required=True)
    parser.add_argument("--mode_name", type=str, default="oracle_live_target_alignment")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--close_threshold", type=float, default=0.5)
    parser.add_argument("--lift_threshold", type=float, default=0.03)
    parser.add_argument("--lift_horizon", type=int, default=24)
    parser.add_argument("--allow_positive_window", type=int, default=6)
    parser.add_argument("--fire_alignment_band_max_steps", type=int, default=2)
    parser.add_argument("--fire_alignment_xy_threshold", type=float, default=0.0025)
    parser.add_argument("--fire_alignment_z_threshold", type=float, default=0.055)
    parser.add_argument("--fire_alignment_distance_threshold", type=float, default=-1.0)
    parser.add_argument("--pose_ready_positive_window", type=int, default=1)
    parser.add_argument("--pose_ready_xy_threshold", type=float, default=0.009)
    parser.add_argument("--pose_ready_abs_z_threshold", type=float, default=0.040)
    parser.add_argument("--pose_ready_yaw_threshold", type=float, default=-1.0)
    parser.add_argument("--pose_ready_basin_distance_threshold", type=float, default=-1.0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows, rows_by_step = load_trace(Path(args.trace_json), args.mode_name)
    support_states = load_npz_rows(Path(args.support_states_npz))
    trigger_states = load_npz_rows(Path(args.trigger_states_npz))

    close_step, close_idx, close_z, max_future_z = first_successful_close(
        rows,
        close_threshold=float(args.close_threshold),
        lift_threshold=float(args.lift_threshold),
        lift_horizon=int(args.lift_horizon),
    )
    if close_step < 0:
        raise RuntimeError("No successful close-with-lift event found in the live-target trace.")

    positive_steps, fire_meta = select_strong_fire_positive_steps(
        rows,
        close_step=int(close_step),
        close_threshold=float(args.close_threshold),
        positive_window=int(args.allow_positive_window),
        alignment_band_max_steps=int(args.fire_alignment_band_max_steps),
        alignment_xy_threshold=float(args.fire_alignment_xy_threshold),
        alignment_z_threshold=float(args.fire_alignment_z_threshold),
        alignment_distance_threshold=float(args.fire_alignment_distance_threshold),
    )
    if not positive_steps:
        raise RuntimeError("No strong fire-positive steps found in the successful live-target trace.")

    support_mode = support_states["mode_index"].astype(np.int64) == 2
    support_open = support_states["rollout_gripper_open"].astype(np.float32) >= float(args.close_threshold)
    rollout_steps = support_states["rollout_step"].astype(np.int64)
    pose_mask = support_mode & support_open & (rollout_steps <= int(close_step))
    pose_data = filter_npz_rows(support_states, pose_mask)
    pose_delta = pose_data["current_delta_basin_target"].astype(np.float32)
    pose_xy = np.linalg.norm(pose_delta[:, :2], axis=1).astype(np.float32)
    pose_abs_z = np.abs(pose_delta[:, 2]).astype(np.float32)
    pose_yaw = np.abs(pose_delta[:, 5]).astype(np.float32)
    pose_basin_distance = pose_data["current_basin_distance"].astype(np.float32)
    pose_band_ready = (
        (
            float(args.pose_ready_basin_distance_threshold) < 0.0
            or pose_basin_distance <= float(args.pose_ready_basin_distance_threshold)
        )
        & (pose_xy <= float(args.pose_ready_xy_threshold))
        & (pose_abs_z <= float(args.pose_ready_abs_z_threshold))
        & (
            float(args.pose_ready_yaw_threshold) < 0.0
            or pose_yaw <= float(args.pose_ready_yaw_threshold)
        )
    )
    pose_ready_target = (
        ((int(close_step) - pose_data["rollout_step"].astype(np.int64)) >= 0)
        & ((int(close_step) - pose_data["rollout_step"].astype(np.int64)) <= int(args.pose_ready_positive_window))
        & pose_band_ready
    ).astype(np.float32)
    pose_data["ready_to_close_target"] = pose_ready_target
    pose_meta = {
        "source": "oracle_live_target_alignment_distill",
        "mode_name": args.mode_name,
        "success_close_step": int(close_step),
        "success_close_row_index": int(close_idx),
        "object_z_at_close": float(close_z),
        "max_object_z_after_close": float(max_future_z),
        "num_pose_rows": int(np.sum(pose_mask)),
        "ready_label": {
            "mode": "distill_close_step_window",
            "positive_window": int(args.pose_ready_positive_window),
            "xy_threshold": float(args.pose_ready_xy_threshold),
            "abs_z_threshold": float(args.pose_ready_abs_z_threshold),
            "yaw_threshold": float(args.pose_ready_yaw_threshold),
            "basin_distance_threshold": float(args.pose_ready_basin_distance_threshold),
            "band_eligible_count": int(np.sum(pose_band_ready)),
            "positive_count": int(np.sum(pose_ready_target > 0.5)),
            "positive_rate": float(np.mean(pose_ready_target > 0.5)) if pose_ready_target.size > 0 else 0.0,
        },
    }
    pose_out = output_root / "pose_distill_candidates.npz"
    save_npz_dict(pose_data, pose_out)
    pose_meta_path = pose_out.with_suffix(".meta.json")
    pose_meta_path.write_text(json.dumps(pose_meta, indent=2), encoding="utf-8")

    trigger_rows, trigger_summary = build_trigger_rows(
        trigger_states,
        rows_by_step,
        close_step=int(close_step),
        close_threshold=float(args.close_threshold),
        positive_steps=positive_steps,
    )
    if not trigger_rows:
        raise RuntimeError("No trigger rows produced for oracle live-target distillation.")
    trigger_meta = {
        "source": "oracle_live_target_alignment_distill",
        "mode_name": args.mode_name,
        "success_close_step": int(close_step),
        "success_close_row_index": int(close_idx),
        "object_z_at_close": float(close_z),
        "max_object_z_after_close": float(max_future_z),
        "fire_positive_mode": "alignment_complete_suffix",
        "fire_positive_steps": [int(x) for x in sorted(positive_steps)],
        **fire_meta,
        **trigger_summary,
    }
    trigger_output_dir = output_root / "close_trigger_distill"
    write_rows(trigger_rows, trigger_output_dir, trigger_meta)
    write_split_rows(trigger_rows, trigger_output_dir, trigger_meta)

    print(json.dumps(
        {
            "pose_output": str(pose_out),
            "trigger_output_dir": str(trigger_output_dir),
            "trigger_fire_split_dir": str(trigger_output_dir / "fire_trigger"),
            "trigger_hold_split_dir": str(trigger_output_dir / "hold_trigger"),
            "pose_meta": pose_meta,
            "trigger_meta": trigger_meta,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
