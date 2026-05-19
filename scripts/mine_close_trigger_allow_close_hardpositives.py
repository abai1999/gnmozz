"""
mine_close_trigger_allow_close_hardpositives.py

Extract near-best on-policy open states from a fixed-scene rollout and package
them as hard-mined allow-close positives for the close-trigger controller.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_trace(trace_json: Path, mode_name: str):
    with trace_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data[mode_name]["curves"]


def main():
    parser = argparse.ArgumentParser(description="Mine near-best on-policy allow-close hard positives.")
    parser.add_argument("--trace_json", type=str, required=True)
    parser.add_argument("--trigger_states_npz", type=str, required=True)
    parser.add_argument("--mode_name", type=str, default="pose_alignment_only_basin")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--close_threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--basin_distance_max", type=float, default=2.2)
    parser.add_argument("--xy_max", type=float, default=0.014)
    parser.add_argument("--z_max", type=float, default=0.022)
    parser.add_argument("--require_planner_open", action="store_true", default=True)
    args = parser.parse_args()

    mode_index = 0 if args.mode_name == "planner_only" else 1
    curves = load_trace(Path(args.trace_json), args.mode_name)
    states = np.load(args.trigger_states_npz)

    if "mode_index" in states.files:
        mask_mode = states["mode_index"].astype(np.int64) == mode_index
    else:
        mask_mode = np.ones(states["step_idx"].shape[0], dtype=bool)

    state_by_step = {}
    for idx in np.where(mask_mode)[0]:
        step = int(states["step_idx"][idx])
        state_by_step[step] = {k: states[k][idx] for k in states.files}

    candidates = []
    for row in curves:
        step = int(row["step"])
        state = state_by_step.get(step)
        if state is None:
            continue
        gripper_open = float(row.get("gripper_open", 1.0))
        planner_close_intent = bool(row.get("planner_close_intent", False))
        basin_distance = float(row["basin_distance"])
        basin_xy = float(row["basin_xy"])
        basin_z = float(row["basin_z"])
        if gripper_open < float(args.close_threshold):
            continue
        if basin_distance > float(args.basin_distance_max):
            continue
        if basin_xy > float(args.xy_max):
            continue
        if basin_z > float(args.z_max):
            continue
        if args.require_planner_open and planner_close_intent:
            continue
        candidates.append(
            {
                "step": step,
                "basin_distance": basin_distance,
                "basin_xy": basin_xy,
                "basin_z": basin_z,
                "planner_close_intent": planner_close_intent,
                "trigger_prob": float(row.get("trigger_prob", 0.0)),
                "state": state,
            }
        )

    candidates.sort(key=lambda r: (r["basin_distance"], r["basin_xy"], r["basin_z"], r["step"]))
    selected = candidates[: int(args.top_k)]
    if not selected:
        raise RuntimeError("No near-best allow-close candidates matched the requested thresholds.")

    ft_hist_shape = (32, 6)
    rows = []
    selected_steps = []
    for item in selected:
        state = item["state"]
        selected_steps.append(
            {
                "step": int(item["step"]),
                "basin_distance": float(item["basin_distance"]),
                "basin_xy": float(item["basin_xy"]),
                "basin_z": float(item["basin_z"]),
                "planner_close_intent": bool(item["planner_close_intent"]),
                "trigger_prob": float(item["trigger_prob"]),
            }
        )
        rows.append(
            {
                "wrist_depth": np.asarray(state["wrist_depth"], dtype=np.float32),
                "ft_hist": np.zeros(ft_hist_shape, dtype=np.float32),
                "proprio": np.asarray(state["proprio"], dtype=np.float32),
                "base_action": np.asarray(state["base_action"], dtype=np.float32),
                "gripper_context": np.asarray(state["gripper_context"], dtype=np.float32),
                "interaction_role": np.asarray(0, dtype=np.int64),
                "step_idx": np.asarray(int(item["step"]) % 8, dtype=np.int64),
                "delta_target": np.zeros(6, dtype=np.float32),
                "delta_align_target": np.zeros(6, dtype=np.float32),
                "delta_basin_target": np.zeros(6, dtype=np.float32),
                "contact_mask": np.asarray(1, dtype=np.int64),
                "phase_label": np.asarray(1, dtype=np.int64),
                "phase_id": np.asarray(int(state["phase_id"]), dtype=np.int64),
                "phase_age": np.asarray(float(state["phase_age"]), dtype=np.float32),
                "steps_since_last_replan": np.asarray(float(state["steps_since_last_replan"]), dtype=np.float32),
                "stage_role": np.asarray(0, dtype=np.int64),
                "failure_mode": np.asarray(0, dtype=np.int64),
                "transition_flag": np.asarray(0, dtype=np.int64),
                "subgoal_progress": np.asarray(0.0, dtype=np.float32),
                "rollout_gripper_open": np.asarray(float(state["rollout_gripper_open"]), dtype=np.float32),
                "depth_proximity": np.asarray(float(state["depth_proximity"]), dtype=np.float32),
                "planner_close_intent": np.asarray(float(state["planner_close_intent"]), dtype=np.float32),
                "planner_close_intent_strength": np.asarray(float(np.clip(1.0 - state["gripper_context"][1], 0.0, 1.0)), dtype=np.float32),
                "readiness_label": np.asarray(1.0, dtype=np.float32),
                "basin_positive": np.asarray(1.0, dtype=np.float32),
                "basin_distance": np.asarray(float(item["basin_distance"]), dtype=np.float32),
                "hold_label": np.asarray(-1.0, dtype=np.float32),
                "negative_reason": np.asarray(-1, dtype=np.int64),
                "frames_to_expert_close": np.asarray(0, dtype=np.int64),
                "frames_to_reference_trigger": np.asarray(0, dtype=np.int64),
                "post_close_stability_proxy": np.asarray(1.0, dtype=np.float32),
                "grasp_lift_proxy": np.asarray(0.0, dtype=np.float32),
                "reopen_within_horizon": np.asarray(0.0, dtype=np.float32),
                "reopen_after_trigger": np.asarray(0.0, dtype=np.float32),
                "no_progress_after_trigger": np.asarray(0.0, dtype=np.float32),
                "invalid_after_trigger": np.asarray(0.0, dtype=np.float32),
                "gripper_state_target": np.asarray(1, dtype=np.int64),
                "ready_to_close": np.asarray(1.0, dtype=np.float32),
                "planner_close_too_early": np.asarray(0.0, dtype=np.float32),
                "expert_hold_after_close": np.asarray(0.0, dtype=np.float32),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    stacked = {k: np.stack([row[k] for row in rows], axis=0) for k in keys}
    np.savez_compressed(output_dir / "residual_shard_0000.npz", **stacked)

    meta = {
        "dataset_view": "close_trigger_allow_close_hardpositives",
        "mode_name": args.mode_name,
        "num_samples": int(len(rows)),
        "top_k": int(args.top_k),
        "basin_distance_max": float(args.basin_distance_max),
        "xy_max": float(args.xy_max),
        "z_max": float(args.z_max),
        "require_planner_open": bool(args.require_planner_open),
        "selected_steps": selected_steps,
    }
    (output_dir / "residual_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
