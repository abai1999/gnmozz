#!/usr/bin/env python3
"""Build a planner-bias-mimic near-grasp recovery dataset for Coarse2Contact v2."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.recovery_audit import (
    color_mask,
    binary_mask,
    first_close_index,
    load_trace_rows,
    largest_component,
    planner_bias_xyyaw,
    frame_keypoints_from_crop,
    roi_box_from_action_prior,
    trace_episode_index,
)
from scripts.build_c2c_v2_depth_localizer_dataset import (
    _episode_dirs,
    _estimate_wrist_grasp_reference,
    _label_from_mask,
    _phase_name,
)


def _bias_score_threshold(scores: list[float], quantile: float) -> float:
    if not scores:
        return 0.0
    q = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(np.asarray(scores, dtype=np.float32), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task_root",
        type=Path,
        default=Path("data/rlbench_data/insert_onto_square_peg"),
        help="RLBench episode root used to recover the actual RGBD frames.",
    )
    ap.add_argument(
        "--source_trace_root",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/eval_depth_v2_shadow_3ep/gripper_traces"),
        help="Trace root used to extract recovery windows and planner bias context.",
    )
    ap.add_argument(
        "--bias_template_trace_root",
        type=Path,
        nargs="*",
        default=[
            Path("runtime_artifacts/coarse2contact/formal_3ep_planner_only/gripper_traces"),
            Path("runtime_artifacts/coarse2contact_v2/smoke_probe_planner_only/gripper_traces"),
        ],
        help="Trace roots used to summarize the planner tail bias template.",
    )
    ap.add_argument(
        "--output_root",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/datasets_recovery"),
    )
    ap.add_argument("--window_before_close", type=int, default=12)
    ap.add_argument("--min_trace_confidence", type=float, default=0.12)
    ap.add_argument("--bias_quantile", type=float, default=0.70)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--max_samples_per_episode", type=int, default=0)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Build a planner tail bias template from planner-only traces or fallback to the source traces.
    bias_scores: list[float] = []
    bias_components = defaultdict(list)
    for root in [args.source_trace_root, *(args.bias_template_trace_root or [])]:
        root = Path(root)
        if not root.exists():
            continue
        for trace_path in sorted(root.glob("*.jsonl")):
            rows = load_trace_rows(trace_path)
            for row in rows:
                planner_local = np.asarray(row.get("planner_chunk_local_6d", []), dtype=np.float32).reshape(-1)
                if planner_local.size < 6:
                    continue
                xy, yaw_abs, dyaw, score = planner_bias_xyyaw(planner_local[:6])
                bias_scores.append(score)
                bias_components["dx"].append(float(planner_local[0]))
                bias_components["dy"].append(float(planner_local[1]))
                bias_components["dyaw"].append(float(planner_local[5]))
                bias_components["xy"].append(xy)
                bias_components["yaw_abs"].append(yaw_abs)
    planner_bias_score_threshold = _bias_score_threshold(bias_scores, args.bias_quantile)

    records: list[dict] = []
    summary = {
        "task_name": "insert_onto_square_peg",
        "task_root": str(args.task_root.resolve()),
        "source_trace_root": str(args.source_trace_root.resolve()),
        "output_root": str(out_root),
        "bias_template_trace_root": [str(Path(p).resolve()) for p in (args.bias_template_trace_root or [])],
        "window_before_close": int(args.window_before_close),
        "min_trace_confidence": float(args.min_trace_confidence),
        "bias_quantile": float(args.bias_quantile),
        "planner_bias_score_threshold": float(planner_bias_score_threshold),
        "num_episodes": 0,
        "num_windows": 0,
        "num_records": 0,
        "view_counts": Counter(),
        "phase_counts": Counter(),
        "trajectory_counts": Counter(),
        "trajectory_recovery_phase_counts": Counter(),
        "error_score": [],
        "bias_score": [],
        "error_xy": [],
        "error_yaw": [],
        "planner_bias_xy": [],
        "planner_bias_yaw": [],
    }

    episode_dirs = _episode_dirs(args.task_root.resolve())
    if args.max_episodes:
        episode_dirs = episode_dirs[: int(args.max_episodes)]
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {args.task_root}")
    grasp_reference = _estimate_wrist_grasp_reference(episode_dirs)
    summary["grasp_reference"] = dict(grasp_reference)

    for ep_dir in episode_dirs:
        ep_idx = int(ep_dir.name.replace("episode", ""))
        model_inputs_path = ep_dir / "model_inputs.npz"
        phase_path = ep_dir / "phase_annotation.json"
        phase_ids_path = ep_dir / "phase_ids.npy"
        if not model_inputs_path.exists() or not phase_path.exists() or not phase_ids_path.exists():
            continue
        model_inputs = np.load(model_inputs_path, allow_pickle=True)
        phase_ids = np.load(phase_ids_path)
        gripper_open = np.asarray(model_inputs["gripper_open"], dtype=np.float32).reshape(-1)
        close_idx = first_close_index(gripper_open)
        if close_idx is None:
            continue
        phase_annotation = json.loads(phase_path.read_text(encoding="utf-8"))
        start = max(0, int(close_idx) - int(args.window_before_close))
        end = int(close_idx)
        trajectory_id = f"ep{ep_idx:03d}_close{int(close_idx):03d}"
        summary["num_episodes"] += 1
        summary["num_windows"] += 1
        kept_in_episode = 0

        for step_idx in range(start, end):
            rgb_path = ep_dir / "wrist_rgb" / f"{step_idx}.png"
            depth_path = ep_dir / "wrist_depth" / f"{step_idx}.png"
            mask_path = ep_dir / "wrist_mask" / f"{step_idx}.png"
            if not (rgb_path.exists() and depth_path.exists() and mask_path.exists()):
                continue
            rgb = np.asarray(Image.open(rgb_path))
            depth = np.asarray(Image.open(depth_path), dtype=np.float32)
            if float(depth.max()) > 1.5:
                depth = depth / 255.0
            mask = binary_mask(np.asarray(Image.open(mask_path)))
            obj_mask = mask & color_mask(rgb, "blue")
            obj_mask = largest_component(obj_mask)
            h, w = rgb.shape[:2]
            action_target = np.asarray(model_inputs["action_targets"][step_idx], dtype=np.float32)
            gripper_pose = np.asarray(model_inputs["gripper_pose"][step_idx], dtype=np.float32)
            roi_box = roi_box_from_action_prior(rgb.shape, action_target, gripper_pose, 96)
            frame_label = frame_keypoints_from_crop(rgb, depth, obj_mask, roi_box, symmetry=np.pi / 2.0)
            full_frame_label = frame_keypoints_from_crop(rgb, depth, obj_mask, (0, 0, w, h), symmetry=np.pi / 2.0)
            grasp_geom_label = _label_from_mask(
                rgb,
                depth,
                obj_mask,
                symmetry=np.pi / 2.0,
                center_xy=(
                    float(grasp_reference["jaw_center_u"]) * float(max(w - 1, 1)),
                    float(grasp_reference["jaw_center_v"]) * float(max(h - 1, 1)),
                ),
                yaw_reference=float(grasp_reference["jaw_axis_angle"]),
            )
            planner_local = np.asarray(action_target[:6], dtype=np.float32)
            planner_xy, planner_yaw_abs, planner_dyaw, planner_score = planner_bias_xyyaw(planner_local)
            current_error = np.array([grasp_geom_label["dx"], grasp_geom_label["dy"], grasp_geom_label["dyaw"]], dtype=np.float32)
            current_error_xy = float(np.linalg.norm(current_error[:2]))
            current_error_yaw = float(abs(current_error[2]))
            current_error_score = float(np.linalg.norm(np.array([current_error[0], current_error[1], 0.04 * current_error[2]], dtype=np.float32)))
            if current_error_score < planner_bias_score_threshold and float(grasp_geom_label["confidence"]) < max(args.min_trace_confidence * 1.5, 0.18):
                continue
            phase_name = _phase_name(phase_annotation, int(phase_ids[step_idx]))
            recovery_phase = "BIAS" if step_idx < end - 4 else ("REFINE" if step_idx < end - 1 else "RECOVER")
            record = {
                "task_name": "insert_onto_square_peg",
                "episode_idx": int(ep_idx),
                "step_idx": int(step_idx),
                "trajectory_id": trajectory_id,
                "trajectory_step": int(step_idx - start),
                "trajectory_len": int(end - start),
                "trajectory_window_start": int(start),
                "trajectory_window_end": int(end - 1),
                "trajectory_phase": recovery_phase,
                "phase_name": phase_name,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "grasp_recovery_audit",
                "skill_type": "precision_grasp",
                "target_entity": "held_square_ring",
                "reference_entity": "gripper_jaw_frame",
                "controlled_dofs": ["x", "y", "z", "yaw"],
                "view_name": "wrist",
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "mask_path": str(mask_path),
                "roi_box": list(map(int, roi_box)),
                "roi_size_px": 96,
                "roi_resize_px": 128,
                "gripper_pose": gripper_pose.tolist(),
                "proprio": np.asarray(model_inputs["proprio"][step_idx], dtype=np.float32).tolist(),
                "planner_prior_delta": planner_local.tolist(),
                "planner_action_world": planner_local.tolist(),
                "planner_chunk_local_6d": planner_local.tolist(),
                "planner_bias_xy": planner_xy,
                "planner_bias_yaw": planner_yaw_abs,
                "planner_bias_dyaw": planner_dyaw,
                "planner_bias_score": planner_score,
                "planner_bias_rank": float(planner_score / max(planner_bias_score_threshold, 1e-8)),
                "trace_error_valid": bool(grasp_geom_label["confidence"] >= 0.12),
                "trace_error_confidence": float(grasp_geom_label["confidence"]),
                "trace_error_observability": float(grasp_geom_label["observability"]),
                "trace_error_fit_residual": float(grasp_geom_label["fit_residual"]),
                "trace_error_inlier_ratio": float(grasp_geom_label["inlier_ratio"]),
                "trace_error_dx": float(grasp_geom_label["dx"]),
                "trace_error_dy": float(grasp_geom_label["dy"]),
                "trace_error_dyaw": float(grasp_geom_label["dyaw"]),
                "trace_error_norm_xy": current_error_xy,
                "trace_error_norm_yaw": current_error_yaw,
                "trace_error_norm": current_error_score,
                "recovery_target_dx": float(grasp_geom_label["dx"]),
                "recovery_target_dy": float(grasp_geom_label["dy"]),
                "recovery_target_dyaw": float(grasp_geom_label["dyaw"]),
                "recovery_target_norm_xy": current_error_xy,
                "recovery_target_norm_yaw": current_error_yaw,
                "recovery_target_norm": current_error_score,
                "recovery_target_kind": "local_geometry_error",
                "recovery_needed": bool(current_error_score >= planner_bias_score_threshold),
                "recovery_bias_source": "planner_tail_bias_template",
                "recovery_phase": recovery_phase,
                "frame_center_u": float(frame_label["center_u"]),
                "frame_center_v": float(frame_label["center_v"]),
                "frame_axis_pos_u": float(frame_label["axis_pos_u"]),
                "frame_axis_pos_v": float(frame_label["axis_pos_v"]),
                "frame_axis_neg_u": float(frame_label["axis_neg_u"]),
                "frame_axis_neg_v": float(frame_label["axis_neg_v"]),
                "frame_axis_dir_x": float(frame_label["axis_dir_x"]),
                "frame_axis_dir_y": float(frame_label["axis_dir_y"]),
                "frame_confidence": float(frame_label["frame_confidence"]),
                "frame_observability": float(frame_label["frame_observability"]),
                "frame_axis_strength": float(frame_label["frame_axis_strength"]),
                "frame_completeness": float(frame_label["frame_completeness"]),
                "frame_border_touch": float(frame_label["frame_border_touch"]),
                "priv_frame_center_u": float(full_frame_label["center_u"]),
                "priv_frame_center_v": float(full_frame_label["center_v"]),
                "priv_frame_axis_pos_u": float(full_frame_label["axis_pos_u"]),
                "priv_frame_axis_pos_v": float(full_frame_label["axis_pos_v"]),
                "priv_frame_axis_neg_u": float(full_frame_label["axis_neg_u"]),
                "priv_frame_axis_neg_v": float(full_frame_label["axis_neg_v"]),
                "priv_frame_axis_dir_x": float(full_frame_label["axis_dir_x"]),
                "priv_frame_axis_dir_y": float(full_frame_label["axis_dir_y"]),
                "priv_frame_confidence": float(full_frame_label["frame_confidence"]),
                "priv_frame_observability": float(full_frame_label["frame_observability"]),
                "priv_frame_axis_strength": float(full_frame_label["frame_axis_strength"]),
                "priv_frame_completeness": float(full_frame_label["frame_completeness"]),
                "label_source": "local_geometry_error_from_image",
                "uses_privileged_label": False,
                "uses_privileged_runtime": False,
                "uses_privileged_target": False,
                "uses_rlbench_mask_runtime": False,
                "mp4_path": "",
                "source_trace_path": "",
                "source_phase_owner": "planner",
                "source_phase_reason": "planner_tail_bias_mimic",
                "source_c2c_stage": "RING_GRASP_ALIGN",
                "source_localizer_abstained": bool(grasp_geom_label["confidence"] < 0.12),
                "source_localizer_confidence": float(grasp_geom_label["confidence"]),
                "source_recovery_cycle_id": 0,
                "source_retry_id": 0,
                "source_invalid_action_flag": False,
                "source_force_state": "planner",
                "source_planner_reaches_precontact": bool(current_error_score < 0.02),
                "source_planner_reaches_preinsert": bool(current_error_score < 0.02),
            }
            records.append(record)
            summary["view_counts"][record["view_name"]] += 1
            summary["phase_counts"][record["trajectory_phase"]] += 1
            summary["trajectory_counts"][trajectory_id] += 1
            summary["trajectory_recovery_phase_counts"][recovery_phase] += 1
            summary["error_score"].append(current_error_score)
            summary["bias_score"].append(planner_score)
            summary["error_xy"].append(current_error_xy)
            summary["error_yaw"].append(current_error_yaw)
            summary["planner_bias_xy"].append(planner_xy)
            summary["planner_bias_yaw"].append(planner_yaw_abs)
            kept_in_episode += 1
            if args.max_samples_per_episode and kept_in_episode >= int(args.max_samples_per_episode):
                break

    out_path = out_root / "grasp_recovery_dataset_v1.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary["num_records"] = len(records)
    summary["view_counts"] = dict(summary["view_counts"])
    summary["phase_counts"] = dict(summary["phase_counts"])
    summary["trajectory_counts"] = dict(summary["trajectory_counts"])
    summary["trajectory_recovery_phase_counts"] = dict(summary["trajectory_recovery_phase_counts"])
    for key in ["error_score", "bias_score", "error_xy", "error_yaw", "planner_bias_xy", "planner_bias_yaw"]:
        arr = np.asarray(summary[key], dtype=np.float32)
        summary[key] = {
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
            "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
            "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
        }

    template_path = out_root / "planner_bias_template.json"
    template_path.write_text(
        json.dumps(
            {
                "bias_quantile": float(args.bias_quantile),
                "bias_score_threshold": float(planner_bias_score_threshold),
                "planner_bias_components": {
                    key: {
                        "mean": float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0,
                        "median": float(np.median(np.asarray(values, dtype=np.float32))) if values else 0.0,
                        "p90": float(np.percentile(np.asarray(values, dtype=np.float32), 90)) if values else 0.0,
                    }
                    for key, values in bias_components.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary_path = out_root / "grasp_recovery_dataset_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    print(template_path)
    print(summary_path)


if __name__ == "__main__":
    main()
