#!/usr/bin/env python3
"""Build a recovery dataset from real runtime failure traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.learned_localizer import load_ring_frame_localizer_checkpoint, _softargmax_2d
from prismatic.robot.coarse2contact_v2.recovery_audit import load_trace_rows, planner_bias_xyyaw, roi_box_from_action_prior
from prismatic.robot.stage_target_provider import pose_delta_local_between


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_episode_target_pose(npz: np.lib.npyio.NpzFile) -> np.ndarray | None:
    for key in ("episode_target_pose_7d", "target_pose_7d", "privileged_target_pose_7d"):
        if key in npz.files:
            arr = np.asarray(npz[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                arr = arr[0]
            arr = arr.reshape(-1)
            if arr.size >= 7 and np.all(np.isfinite(arr[:7])):
                return arr[:7].astype(np.float32)
    return None


def _rgbd_tensor_from_arrays(rgb: np.ndarray, depth: np.ndarray, crop_box: tuple[int, int, int, int], resize_to: int) -> torch.Tensor:
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    rgb_img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").crop((x0, y0, x1, y1))
    depth_arr = np.asarray(depth, dtype=np.float32)
    if float(np.nanmax(depth_arr)) > 1.5:
        depth_arr = depth_arr / 255.0
    depth_arr = np.clip(depth_arr, 0.0, 1.0)
    depth_img = Image.fromarray(depth_arr.astype(np.float32), mode="F").crop((x0, y0, x1, y1))
    if resize_to > 0:
        rgb_img = rgb_img.resize((resize_to, resize_to), resample=Image.BILINEAR)
        depth_img = depth_img.resize((resize_to, resize_to), resample=Image.BILINEAR)
    rgb_arr = np.asarray(rgb_img, dtype=np.float32) / 255.0
    depth_arr = np.asarray(depth_img, dtype=np.float32)
    h, w = depth_arr.shape[:2]
    xs = np.linspace(-1.0, 1.0, num=w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, num=h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    rgbd = np.concatenate([rgb_arr, depth_arr[..., None], grid_x[..., None], grid_y[..., None]], axis=-1)
    return torch.from_numpy(np.transpose(rgbd, (2, 0, 1))).float()


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)


def _save_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(depth, dtype=np.float32)
    if float(np.nanmax(arr)) <= 1.5:
        arr = np.clip(arr * 255.0, 0.0, 255.0)
    else:
        arr = np.clip(arr, 0.0, 255.0)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _safe_np(value: object, dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=Path, required=True, help="Root directory from evaluate_c2c_v2_rlbench.py")
    ap.add_argument("--output_root", type=Path, default=Path("runtime_artifacts/coarse2contact_v2/datasets_runtime_failure"))
    ap.add_argument(
        "--ring_frame_checkpoint",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/ring_frame_localizer_v2_refined/best.pt"),
    )
    ap.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    ap.add_argument("--tail_steps", type=int, default=12)
    ap.add_argument("--crop_size", type=int, default=96)
    ap.add_argument("--resize_to", type=int, default=128)
    ap.add_argument(
        "--ring_frame_conf_threshold",
        type=float,
        default=0.33,
        help="Metadata threshold for marking ring-frame localizer abstention in the runtime-failure dataset.",
    )
    ap.add_argument("--all_episodes", action="store_true", default=False)
    ap.add_argument("--allow_missing_target_pose", action="store_true", default=False)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    eval_root = args.eval_root.resolve()
    trace_dir = eval_root / "gripper_traces"
    obs_dir = eval_root / "runtime_observations"
    eval_results_path = eval_root / "eval_results.json"
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing trace dir: {trace_dir}")
    if not obs_dir.exists():
        raise FileNotFoundError(f"Missing runtime observation dir: {obs_dir}")

    eval_report = _load_json(eval_results_path) if eval_results_path.exists() else {}
    stage_stats = list(eval_report.get("stage_stats", []))
    if stage_stats and not args.all_episodes:
        selected_eps = [int(s.get("episode_index", -1)) for s in stage_stats if not bool(s.get("success", False)) or int(s.get("invalid_action_count", 0)) > 0]
    else:
        selected_eps = [int(p.name.replace("ep", "").split("_")[0]) for p in trace_dir.glob("ep*_gripper_trace.jsonl")]
    selected_eps = sorted({ep for ep in selected_eps if ep >= 0})
    if not selected_eps:
        raise RuntimeError(f"No selected episodes found under {eval_root}")

    ring_model, ring_ckpt = load_ring_frame_localizer_checkpoint(args.ring_frame_checkpoint, map_location=args.device)
    ring_model = ring_model.to(args.device)
    ring_model.eval()

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    image_root = out_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    summary = {
        "eval_root": str(eval_root),
        "trace_dir": str(trace_dir),
        "obs_dir": str(obs_dir),
        "output_root": str(out_root),
        "ring_frame_checkpoint": str(args.ring_frame_checkpoint),
        "ring_frame_conf_threshold": float(args.ring_frame_conf_threshold),
        "selected_episodes": selected_eps,
        "tail_steps": int(args.tail_steps),
        "crop_size": int(args.crop_size),
        "resize_to": int(args.resize_to),
        "failure_only": bool(not args.all_episodes),
        "all_episodes": bool(args.all_episodes),
        "rows": 0,
        "episodes": len(selected_eps),
        "failure_episodes": 0,
        "invalid_episode_count": 0,
        "target_pose_available_rate": 0.0,
        "selected_step_counts": [],
        "planner_bias_scores": [],
        "recovery_target_norm": [],
    }

    target_pose_available = 0
    for ep_idx in selected_eps:
        trace_path = trace_dir / f"ep{ep_idx:03d}_gripper_trace.jsonl"
        obs_path = obs_dir / f"ep{ep_idx:03d}_runtime_obs.npz"
        if not trace_path.exists() or not obs_path.exists():
            continue
        trace_rows = load_trace_rows(trace_path)
        if not trace_rows:
            continue
        trace_rows = sorted(trace_rows, key=lambda r: int(r.get("step", -1)))
        ep_record = next((s for s in stage_stats if int(s.get("episode_index", -1)) == ep_idx), None)
        ep_success = bool(ep_record.get("success", False)) if ep_record is not None else False
        ep_invalid_count = int(ep_record.get("invalid_action_count", 0)) if ep_record is not None else int(sum(bool(r.get("invalid_action", False)) for r in trace_rows))
        if not ep_success:
            summary["failure_episodes"] += 1
        if ep_invalid_count > 0:
            summary["invalid_episode_count"] += 1

        with np.load(obs_path, allow_pickle=True) as obs_npz:
            target_pose_7d = _read_episode_target_pose(obs_npz)
            if target_pose_7d is not None:
                target_pose_available += 1
            if target_pose_7d is None and not args.allow_missing_target_pose:
                continue

            if target_pose_7d is None:
                target_pose_7d = np.full((7,), np.nan, dtype=np.float32)

            tail_start = max(0, len(trace_rows) - int(args.tail_steps))
            if ep_invalid_count > 0:
                invalid_steps = [int(r.get("step", -1)) for r in trace_rows if bool(r.get("invalid_action", False))]
                if invalid_steps:
                    tail_start = max(0, min(invalid_steps) - max(0, int(args.tail_steps) // 2))
            selected_rows = [r for r in trace_rows if int(r.get("step", -1)) >= tail_start]
            summary["selected_step_counts"].append(len(selected_rows))

            for row in selected_rows:
                step_idx = int(row.get("step", -1))
                if step_idx < 0 or step_idx >= int(obs_npz["gripper_pose"].shape[0]):
                    continue
                wrist_rgb = np.asarray(obs_npz["wrist_rgb"][step_idx], dtype=np.uint8)
                wrist_depth = np.asarray(obs_npz["wrist_depth"][step_idx], dtype=np.float32)
                gripper_pose = np.asarray(obs_npz["gripper_pose"][step_idx], dtype=np.float32)
                gripper_open = float(np.asarray(obs_npz["gripper_open"][step_idx], dtype=np.float32))
                proprio = np.asarray(obs_npz["proprio"][step_idx], dtype=np.float32)
                planner_prior = np.asarray(obs_npz["planner_action_world_6d"][step_idx], dtype=np.float32)
                planner_post = np.asarray(obs_npz["executed_action_world_6d"][step_idx], dtype=np.float32)
                invalid_flag = bool(float(np.asarray(obs_npz["invalid_action"][step_idx], dtype=np.float32)) > 0.5)
                reward = float(np.asarray(obs_npz["reward"][step_idx], dtype=np.float32))
                terminate = bool(float(np.asarray(obs_npz["terminate"][step_idx], dtype=np.float32)) > 0.5)
                roi_box = roi_box_from_action_prior(wrist_rgb.shape, planner_prior, gripper_pose, int(args.crop_size))
                crop_rgb = Image.fromarray(wrist_rgb, mode="RGB").crop(roi_box).resize((args.resize_to, args.resize_to), resample=Image.BILINEAR)
                depth_arr = np.asarray(wrist_depth, dtype=np.float32)
                if float(np.nanmax(depth_arr)) > 1.5:
                    depth_arr = depth_arr / 255.0
                crop_depth = Image.fromarray(np.clip(depth_arr, 0.0, 1.0).astype(np.float32), mode="F").crop(roi_box).resize((args.resize_to, args.resize_to), resample=Image.BILINEAR)
                ep_dir = image_root / f"ep{ep_idx:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                rgb_path = ep_dir / f"step{step_idx:04d}_rgb.png"
                depth_path = ep_dir / f"step{step_idx:04d}_depth.png"
                crop_rgb.save(rgb_path)
                _save_depth(depth_path, np.asarray(crop_depth, dtype=np.float32))

                rgbd_tensor = _rgbd_tensor_from_arrays(wrist_rgb, wrist_depth, roi_box, int(args.resize_to)).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    pred = ring_model(rgbd_tensor)
                    cx, cy, _ = _softargmax_2d(pred["center_heatmap_logits"])
                    px, py, _ = _softargmax_2d(pred["axis_pos_heatmap_logits"])
                    nx, ny, _ = _softargmax_2d(pred["axis_neg_heatmap_logits"])
                    visible = torch.sigmoid(pred["visible_logit"]).item()
                    conf = torch.sigmoid(pred["confidence_logit"]).item()
                center_u = float(cx[0, 0].item())
                center_v = float(cy[0, 0].item())
                axis_pos_u = float(px[0, 0].item())
                axis_pos_v = float(py[0, 0].item())
                axis_neg_u = float(nx[0, 0].item())
                axis_neg_v = float(ny[0, 0].item())
                axis_dir = np.asarray([axis_pos_u - axis_neg_u, axis_pos_v - axis_neg_v], dtype=np.float32)
                axis_dir_norm = float(np.linalg.norm(axis_dir))
                if axis_dir_norm > 1e-6:
                    axis_dir = axis_dir / axis_dir_norm
                else:
                    axis_dir = np.asarray([1.0, 0.0], dtype=np.float32)
                frame_confidence = float(np.clip(conf, 0.0, 1.0))
                frame_observability = float(np.clip(visible, 0.0, 1.0))
                frame_axis_strength = float(np.clip(frame_confidence * frame_observability, 0.0, 1.0))
                border_touch = float(
                    center_u <= 0.05
                    or center_v <= 0.05
                    or center_u >= 0.95
                    or center_v >= 0.95
                )
                frame_completeness = float(np.clip(frame_observability, 0.0, 1.0))
                local_error = pose_delta_local_between(gripper_pose, target_pose_7d)
                planner_xy, planner_yaw_abs, planner_dyaw, planner_score = planner_bias_xyyaw(planner_prior[:6])
                runtime_failure = True
                record = {
                    "task_name": str(args.task_name),
                    "episode_idx": int(ep_idx),
                    "step_idx": int(step_idx),
                    "trajectory_id": f"ep{ep_idx:03d}_runtime_failure_tail",
                    "trajectory_step": int(step_idx - tail_start),
                    "trajectory_len": int(max(1, len(selected_rows))),
                    "trajectory_phase": "runtime_failure_tail",
                    "phase_name": str(row.get("phase_name", row.get("c2c_v2_stage", "planner_only"))),
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_name": "grasp_recovery_runtime_failure",
                    "skill_type": "precision_grasp",
                    "target_entity": "held_square_ring",
                    "reference_entity": "gripper_jaw_frame",
                    "controlled_dofs": ["x", "y", "z", "yaw"],
                    "view_name": "wrist",
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "mask_path": "",
                    "roi_box": list(map(int, roi_box)),
                    "roi_size_px": int(args.crop_size),
                    "roi_resize_px": int(args.resize_to),
                    "gripper_pose": gripper_pose.tolist(),
                    "proprio": proprio.tolist(),
                    "planner_prior_delta": planner_prior[:6].tolist(),
                    "planner_action_world": planner_prior[:6].tolist(),
                    "planner_chunk_local_6d": planner_prior[:6].tolist(),
                    "planner_bias_xy": planner_xy,
                    "planner_bias_yaw": planner_yaw_abs,
                    "planner_bias_dyaw": planner_dyaw,
                    "planner_bias_score": planner_score,
                    "planner_bias_rank": float(planner_score),
                    "trace_error_valid": bool(np.all(np.isfinite(local_error[:3]))),
                    "trace_error_confidence": frame_confidence,
                    "trace_error_observability": frame_observability,
                    "trace_error_fit_residual": 0.0,
                    "trace_error_inlier_ratio": frame_axis_strength,
                    "trace_error_dx": float(local_error[0]),
                    "trace_error_dy": float(local_error[1]),
                    "trace_error_dyaw": float(local_error[5]),
                    "trace_error_norm_xy": float(np.linalg.norm(local_error[:2])),
                    "trace_error_norm_yaw": float(abs(local_error[5])),
                    "trace_error_norm": float(np.linalg.norm(np.array([local_error[0], local_error[1], 0.04 * local_error[5]], dtype=np.float32))),
                    "recovery_target_dx": float(local_error[0]),
                    "recovery_target_dy": float(local_error[1]),
                    "recovery_target_dyaw": float(local_error[5]),
                    "recovery_target_norm_xy": float(np.linalg.norm(local_error[:2])),
                    "recovery_target_norm_yaw": float(abs(local_error[5])),
                    "recovery_target_norm": float(np.linalg.norm(np.array([local_error[0], local_error[1], 0.04 * local_error[5]], dtype=np.float32))),
                    "recovery_target_kind": "runtime_failure_local_geometry_error",
                    "recovery_needed": True,
                    "recovery_bias_source": "real_runtime_failure_trace",
                    "recovery_phase": "FAILURE_TAIL",
                    "frame_center_u": center_u,
                    "frame_center_v": center_v,
                    "frame_axis_pos_u": axis_pos_u,
                    "frame_axis_pos_v": axis_pos_v,
                    "frame_axis_neg_u": axis_neg_u,
                    "frame_axis_neg_v": axis_neg_v,
                    "frame_axis_dir_x": float(axis_dir[0]),
                    "frame_axis_dir_y": float(axis_dir[1]),
                    "frame_confidence": frame_confidence,
                    "frame_observability": frame_observability,
                    "frame_axis_strength": frame_axis_strength,
                    "frame_completeness": frame_completeness,
                    "frame_border_touch": border_touch,
                    "priv_frame_center_u": center_u,
                    "priv_frame_center_v": center_v,
                    "priv_frame_axis_pos_u": axis_pos_u,
                    "priv_frame_axis_pos_v": axis_pos_v,
                    "priv_frame_axis_neg_u": axis_neg_u,
                    "priv_frame_axis_neg_v": axis_neg_v,
                    "priv_frame_axis_dir_x": float(axis_dir[0]),
                    "priv_frame_axis_dir_y": float(axis_dir[1]),
                    "priv_frame_confidence": frame_confidence,
                    "priv_frame_observability": frame_observability,
                    "priv_frame_axis_strength": frame_axis_strength,
                    "priv_frame_completeness": frame_completeness,
                    "label_source": "real_runtime_failure_trace+captured_target_pose+ring_frame_shadow",
                    "uses_privileged_label": True,
                    "uses_privileged_runtime": False,
                    "uses_privileged_target": False,
                    "uses_rlbench_mask_runtime": False,
                    "mp4_path": str(row.get("mp4_path", "")),
                    "source_trace_path": str(trace_path),
                    "source_phase_owner": str(row.get("phase_owner", row.get("c2c_v2_owner", "planner"))),
                    "source_phase_reason": str(row.get("phase_reason", row.get("c2c_v2_stage", "planner"))),
                    "source_c2c_stage": str(row.get("c2c_v2_stage", "planner_only")),
                    "source_localizer_abstained": bool(frame_confidence < float(args.ring_frame_conf_threshold)),
                    "source_localizer_confidence": frame_confidence,
                    "source_recovery_cycle_id": int(row.get("recovery_cycle_id", 0)),
                    "source_retry_id": int(row.get("retry_id", 0)),
                    "source_invalid_action_flag": invalid_flag,
                    "source_force_state": str(row.get("force_skill_state", "planner_only")),
                    "source_planner_reaches_precontact": bool(row.get("planner_reaches_precontact", False)),
                    "source_planner_reaches_preinsert": bool(row.get("planner_reaches_preinsert", False)),
                    "source_success": bool(ep_success),
                    "source_invalid_action_count": int(ep_invalid_count),
                    "source_reward": float(reward),
                    "source_terminate": bool(terminate),
                }
                records.append(record)
                summary["planner_bias_scores"].append(float(planner_score))
                summary["recovery_target_norm"].append(float(record["recovery_target_norm"]))

    out_path = out_root / "grasp_recovery_runtime_failure_dataset_v1.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary["rows"] = len(records)
    summary["target_pose_available_rate"] = float(target_pose_available / max(len(selected_eps), 1))
    summary["failure_episodes_rate"] = float(summary["failure_episodes"] / max(len(selected_eps), 1))
    summary["invalid_episode_rate"] = float(summary["invalid_episode_count"] / max(len(selected_eps), 1))
    summary["selected_step_counts"] = [int(v) for v in summary["selected_step_counts"]]
    summary["planner_bias_scores"] = {
        "mean": float(np.mean(summary["planner_bias_scores"])) if summary["planner_bias_scores"] else 0.0,
        "median": float(np.median(summary["planner_bias_scores"])) if summary["planner_bias_scores"] else 0.0,
        "p90": float(np.percentile(summary["planner_bias_scores"], 90)) if summary["planner_bias_scores"] else 0.0,
        "p95": float(np.percentile(summary["planner_bias_scores"], 95)) if summary["planner_bias_scores"] else 0.0,
    }
    summary["recovery_target_norm"] = {
        "mean": float(np.mean(summary["recovery_target_norm"])) if summary["recovery_target_norm"] else 0.0,
        "median": float(np.median(summary["recovery_target_norm"])) if summary["recovery_target_norm"] else 0.0,
        "p90": float(np.percentile(summary["recovery_target_norm"], 90)) if summary["recovery_target_norm"] else 0.0,
        "p95": float(np.percentile(summary["recovery_target_norm"], 95)) if summary["recovery_target_norm"] else 0.0,
    }

    summary_path = out_root / "grasp_recovery_runtime_failure_dataset_v1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
