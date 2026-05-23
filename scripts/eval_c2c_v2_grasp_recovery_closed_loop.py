#!/usr/bin/env python3
"""Closed-loop offline recovery auditor for Coarse2Contact v2 grasp recovery."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.datasets import DepthLocalizerJsonlDataset
from prismatic.robot.coarse2contact_v2.learned_localizer import load_grasp_recovery_checkpoint
from prismatic.robot.coarse2contact_v2.basin_recovery import (
    BasinRecoveryConfig,
    BasinRecoverySupervisor,
    classify_visual_evidence_for_basin,
    visual_error_from_record,
)
from prismatic.robot.coarse2contact_v2.recovery_augmentation import failure_morphology_bucket
from prismatic.robot.coarse2contact_v2.recovery_audit import (
    apply_closed_loop_recovery_step,
    choose_gated_hybrid_candidate,
    classify_visual_evidence,
    in_close_ready_basin,
    in_near_grasp_basin,
    monotonic_decay_prefix,
    recovery_error_norm,
    recovery_overshoot_flag,
)


def _filter_records(records: list[dict]) -> list[dict]:
    filtered = []
    for record in records:
        if str(record.get("view_name", "")) != "wrist":
            continue
        if float(record.get("trace_error_confidence", 0.0)) <= 0.0:
            continue
        if not record.get("rgb_path") or not record.get("depth_path"):
            continue
        filtered.append(record)
    return filtered


def _frame_tensor(record: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [
                float(record.get("frame_center_u", 0.5)),
                float(record.get("frame_center_v", 0.5)),
                float(record.get("frame_axis_pos_u", 0.5)),
                float(record.get("frame_axis_pos_v", 0.5)),
                float(record.get("frame_axis_neg_u", 0.5)),
                float(record.get("frame_axis_neg_v", 0.5)),
                float(record.get("frame_confidence", 0.0)),
                float(record.get("frame_observability", 0.0)),
                float(record.get("frame_completeness", 0.0)),
                float(record.get("frame_border_touch", 1.0)),
            ]
        ],
        dtype=torch.float32,
        device=device,
    )


def _proprio_tensor(record: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [((list(record.get("proprio", [])) + [0.0] * 15)[:15])],
        dtype=torch.float32,
        device=device,
    )


def _prior_tensor(prior_state: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(prior_state[None, :6], dtype=torch.float32, device=device)


def _predict_model_step(
    model: torch.nn.Module,
    image_rgbd: torch.Tensor,
    frame: torch.Tensor,
    proprio: torch.Tensor,
    planner_prior_state: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        out = model(
            image_rgbd.to(device),
            frame,
            proprio,
            _prior_tensor(planner_prior_state, device),
        )
    return {
        "dx": float(out["dx"][0].detach().cpu().item()),
        "dy": float(out["dy"][0].detach().cpu().item()),
        "dyaw": float(out["dyaw"][0].detach().cpu().item()),
        "confidence": float(torch.sigmoid(out["confidence_logit"])[0].detach().cpu().item()),
    }


def _planner_prior_only_step(prior_state: np.ndarray) -> dict[str, float]:
    return {
        "dx": float(prior_state[0]),
        "dy": float(prior_state[1]),
        "dyaw": float(prior_state[5]),
        "confidence": 1.0,
    }


def _zero_step() -> dict[str, float]:
    return {"dx": 0.0, "dy": 0.0, "dyaw": 0.0, "confidence": 0.0}


def _summarize_rollouts(rollouts: list[dict], *, bucket_names: list[str]) -> dict[str, Any]:
    if not rollouts:
        return {
            "count": 0,
            "closed_loop_gain_mean": 0.0,
            "closed_loop_gain_median": 0.0,
            "basin_entry_rate": 0.0,
            "close_ready_entry_rate": 0.0,
            "monotonic_decay_rate": 0.0,
            "overshoot_rate": 0.0,
            "micro_entry_ready_rate": 0.0,
            "micro_yaw_active_rate": 0.0,
            "reacquire_view_rate": 0.0,
            "visual_pullback_rate": 0.0,
            "micro_servo_rate": 0.0,
            "verify_basin_rate": 0.0,
            "steps_to_basin_median": None,
            "steps_to_close_ready_median": None,
            "bucketwise_gain": {},
        }
    gains = np.asarray([float(r["closed_loop_gain"]) for r in rollouts], dtype=np.float32)
    basin_entries = np.asarray([bool(r["entered_near_grasp_basin"]) for r in rollouts], dtype=np.float32)
    close_entries = np.asarray([bool(r["entered_close_ready_basin"]) for r in rollouts], dtype=np.float32)
    monotonic = np.asarray([bool(r["monotonic_decay"]) for r in rollouts], dtype=np.float32)
    overshoot = np.asarray([bool(r["overshoot_any"]) for r in rollouts], dtype=np.float32)
    total_steps = int(sum(int(r.get("step_count", 0)) for r in rollouts))
    micro_entry_ready_steps = int(sum(int(r.get("micro_entry_ready_steps", 0)) for r in rollouts))
    micro_yaw_active_steps = int(sum(int(r.get("micro_yaw_active_steps", 0)) for r in rollouts))
    mode_counts: dict[str, int] = {}
    total_modes = 0
    for rollout in rollouts:
        for mode in rollout.get("basin_recovery_mode_sequence", []):
            key = str(mode)
            mode_counts[key] = mode_counts.get(key, 0) + 1
            total_modes += 1
    steps_to_basin = [int(r["steps_to_near_grasp_basin"]) for r in rollouts if r["steps_to_near_grasp_basin"] is not None]
    steps_to_close = [int(r["steps_to_close_ready_basin"]) for r in rollouts if r["steps_to_close_ready_basin"] is not None]
    bucketwise_gain: dict[str, float] = {}
    for bucket in bucket_names:
        subset = [r for r in rollouts if str(r["failure_bucket"]) == str(bucket)]
        if subset:
            bucketwise_gain[bucket] = float(np.mean(np.asarray([float(r["closed_loop_gain"]) for r in subset], dtype=np.float32)))
    return {
        "count": int(len(rollouts)),
        "closed_loop_gain_mean": float(np.mean(gains)),
        "closed_loop_gain_median": float(np.median(gains)),
        "basin_entry_rate": float(np.mean(basin_entries)),
        "close_ready_entry_rate": float(np.mean(close_entries)),
        "monotonic_decay_rate": float(np.mean(monotonic)),
        "overshoot_rate": float(np.mean(overshoot)),
        "micro_entry_ready_rate": float(micro_entry_ready_steps / max(total_steps, 1)),
        "micro_yaw_active_rate": float(micro_yaw_active_steps / max(total_steps, 1)),
        "reacquire_view_rate": float(mode_counts.get("REACQUIRE_VIEW", 0) / max(total_modes, 1)),
        "visual_pullback_rate": float(mode_counts.get("VISUAL_PULLBACK", 0) / max(total_modes, 1)),
        "micro_servo_rate": float(mode_counts.get("MICRO_SERVO_TO_BASIN", 0) / max(total_modes, 1)),
        "verify_basin_rate": float(mode_counts.get("VERIFY_BASIN", 0) / max(total_modes, 1)),
        "steps_to_basin_median": float(np.median(np.asarray(steps_to_basin, dtype=np.float32))) if steps_to_basin else None,
        "steps_to_close_ready_median": float(np.median(np.asarray(steps_to_close, dtype=np.float32))) if steps_to_close else None,
        "bucketwise_gain": bucketwise_gain,
    }


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--v11_checkpoint",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/grasp_recovery_head_v11_runtime_failure/best.pt"),
    )
    ap.add_argument(
        "--v16_checkpoint",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/checkpoints/grasp_recovery_head_v16_tailbucket_conservative_30ep/best.pt"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_closed_loop_30ep.json"),
    )
    ap.add_argument(
        "--trace_output",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_closed_loop_30ep_trace.jsonl"),
    )
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--large_bias_quantile", type=float, default=0.70)
    ap.add_argument("--visual_conf_threshold", type=float, default=1.0e-3)
    ap.add_argument("--visual_observability_threshold", type=float, default=1.0e-3)
    ap.add_argument("--visual_axis_strength_threshold", type=float, default=1.0e-6)
    ap.add_argument("--basin_visual_conf_threshold", type=float, default=0.01)
    ap.add_argument("--basin_visual_observability_threshold", type=float, default=0.002)
    ap.add_argument("--basin_visual_axis_strength_threshold", type=float, default=1.0e-5)
    ap.add_argument("--basin_visual_gain", type=float, default=0.35)
    ap.add_argument("--basin_micro_gain", type=float, default=0.20)
    ap.add_argument("--basin_eval_line_search", action="store_true", default=True)
    ap.add_argument("--no_basin_eval_line_search", dest="basin_eval_line_search", action="store_false")
    ap.add_argument("--near_grasp_xy_threshold", type=float, default=0.015)
    ap.add_argument("--near_grasp_yaw_threshold", type=float, default=0.08)
    ap.add_argument("--close_ready_xy_threshold", type=float, default=0.005)
    ap.add_argument("--close_ready_yaw_threshold", type=float, default=0.03)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--v11_single_step_bucket_report",
        type=Path,
        default=Path("runtime_artifacts/coarse2contact_v2/reports/grasp_recovery_failure_buckets_30ep_v11.json"),
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    full = DepthLocalizerJsonlDataset(args.dataset)
    filtered_records = _filter_records(full.records)
    dataset = DepthLocalizerJsonlDataset(args.dataset, records=filtered_records)
    if len(dataset) == 0:
        raise RuntimeError("No valid wrist recovery rows found for closed-loop evaluation")

    v11_model, _ = load_grasp_recovery_checkpoint(args.v11_checkpoint, map_location=device)
    v16_model, _ = load_grasp_recovery_checkpoint(args.v16_checkpoint, map_location=device)
    v11_model = v11_model.to(device).eval()
    v16_model = v16_model.to(device).eval()

    hard_buckets = ["small_xy_small_yaw", "large_xy_small_yaw"]
    candidate_names = ["zero_action", "planner_prior_only", "v11_general", "v16_specialist", "gated_hybrid", "basin_recovery_supervisor"]
    basin_config = BasinRecoveryConfig(
        near_grasp_xy_threshold=float(args.near_grasp_xy_threshold),
        near_grasp_yaw_threshold=float(args.near_grasp_yaw_threshold),
        close_ready_xy_threshold=float(args.close_ready_xy_threshold),
        close_ready_yaw_threshold=float(args.close_ready_yaw_threshold),
        visual_conf_threshold=float(args.basin_visual_conf_threshold),
        visual_observability_threshold=float(args.basin_visual_observability_threshold),
        visual_axis_strength_threshold=float(args.basin_visual_axis_strength_threshold),
        visual_gain=float(args.basin_visual_gain),
        micro_gain=float(args.basin_micro_gain),
        max_recovery_steps=int(args.steps) + 1,
    )

    planner_bias_scores = np.asarray([float(r.get("planner_bias_score", 0.0)) for r in filtered_records], dtype=np.float32)
    large_bias_threshold = float(np.quantile(planner_bias_scores, float(np.clip(args.large_bias_quantile, 0.0, 1.0)))) if planner_bias_scores.size else 0.0

    traces: list[dict[str, Any]] = []
    summaries_by_candidate: dict[str, list[dict[str, Any]]] = {name: [] for name in candidate_names}
    bucket_names = sorted({failure_morphology_bucket(record) for record in filtered_records})

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image_rgbd"].unsqueeze(0).to(device)
        frame = _frame_tensor(sample, device)
        proprio = _proprio_tensor(sample, device)
        init_error = np.asarray(
            [
                float(sample.get("recovery_target_dx", sample.get("trace_error_dx", 0.0))),
                float(sample.get("recovery_target_dy", sample.get("trace_error_dy", 0.0))),
                float(sample.get("recovery_target_dyaw", sample.get("trace_error_dyaw", 0.0))),
            ],
            dtype=np.float32,
        )
        init_prior = np.asarray((list(sample.get("planner_prior_delta", [])) + [0.0] * 6)[:6], dtype=np.float32)
        failure_bucket = failure_morphology_bucket(sample)
        evidence_class = classify_visual_evidence(
            sample,
            conf_threshold=float(args.visual_conf_threshold),
            observability_threshold=float(args.visual_observability_threshold),
            axis_strength_threshold=float(args.visual_axis_strength_threshold),
        )
        basin_evidence_class = classify_visual_evidence_for_basin(sample, config=basin_config).value
        initial_norm = recovery_error_norm(float(init_error[0]), float(init_error[1]), float(init_error[2]))

        for candidate_name in candidate_names:
            current_error = init_error.copy()
            current_prior = init_prior.copy()
            current_visual_error = visual_error_from_record(sample)
            basin_supervisor = BasinRecoverySupervisor(config=basin_config)
            error_curve = [float(initial_norm)]
            entered_near_grasp = in_near_grasp_basin(
                float(current_error[0]),
                float(current_error[1]),
                float(current_error[2]),
                xy_threshold=float(args.near_grasp_xy_threshold),
                yaw_threshold=float(args.near_grasp_yaw_threshold),
            )
            entered_close_ready = in_close_ready_basin(
                float(current_error[0]),
                float(current_error[1]),
                float(current_error[2]),
                xy_threshold=float(args.close_ready_xy_threshold),
                yaw_threshold=float(args.close_ready_yaw_threshold),
            )
            steps_to_basin = 0 if entered_near_grasp else None
            steps_to_close_ready = 0 if entered_close_ready else None
            overshoot_any = False
            selected_names: list[str] = []
            basin_mode_sequence: list[str] = []
            micro_entry_ready_steps = 0
            micro_yaw_active_steps = 0

            for step in range(int(args.steps)):
                pre_error = current_error.copy()
                pre_prior = current_prior.copy()
                pre_norm = recovery_error_norm(float(pre_error[0]), float(pre_error[1]), float(pre_error[2]))

                v11_pred = _predict_model_step(v11_model, image, frame, proprio, current_prior, device)
                v16_pred = _predict_model_step(v16_model, image, frame, proprio, current_prior, device)

                if candidate_name == "zero_action":
                    chosen_name = "zero_action"
                    pred = _zero_step()
                elif candidate_name == "planner_prior_only":
                    chosen_name = "planner_prior_only"
                    pred = _planner_prior_only_step(current_prior)
                elif candidate_name == "v11_general":
                    chosen_name = "v11_general"
                    pred = v11_pred
                elif candidate_name == "v16_specialist":
                    chosen_name = "v16_specialist"
                    pred = v16_pred
                elif candidate_name == "gated_hybrid":
                    v11_post_error, _ = apply_closed_loop_recovery_step(pre_error, pre_prior, [v11_pred["dx"], v11_pred["dy"], v11_pred["dyaw"]])
                    v16_post_error, _ = apply_closed_loop_recovery_step(pre_error, pre_prior, [v16_pred["dx"], v16_pred["dy"], v16_pred["dyaw"]])
                    chosen_name = choose_gated_hybrid_candidate(
                        failure_bucket,
                        v11_post_error_norm=recovery_error_norm(float(v11_post_error[0]), float(v11_post_error[1]), float(v11_post_error[2])),
                        v16_post_error_norm=recovery_error_norm(float(v16_post_error[0]), float(v16_post_error[1]), float(v16_post_error[2])),
                        hard_buckets=hard_buckets,
                    )
                    pred = v16_pred if chosen_name == "v16_specialist" else v11_pred
                else:
                    chosen_name = "basin_recovery_supervisor"
                    decision = basin_supervisor.step(
                        record=sample,
                        planner_prior_state=current_prior,
                        visual_error_state=current_visual_error,
                        model_prediction=v11_pred,
                        target_error_state_for_eval=pre_error,
                        allow_eval_line_search=bool(args.basin_eval_line_search),
                    )
                    basin_mode_sequence.append(decision.mode.value)
                    pred = {
                        "dx": float(decision.correction_xyyaw[0]),
                        "dy": float(decision.correction_xyyaw[1]),
                        "dyaw": float(decision.correction_xyyaw[2]),
                        "confidence": float(decision.confidence),
                    }
                    pred.update(decision.to_trace())
                    micro_entry_ready_steps += int(bool(pred.get("basin_recovery_micro_entry_ready", False)))
                    micro_yaw_active_steps += int(bool(pred.get("basin_recovery_micro_yaw_active", False)))

                selected_names.append(chosen_name)
                correction = np.asarray([float(pred["dx"]), float(pred["dy"]), float(pred["dyaw"])], dtype=np.float32)
                post_error, post_prior = apply_closed_loop_recovery_step(pre_error, pre_prior, correction)
                if candidate_name == "basin_recovery_supervisor":
                    current_visual_error, _ = apply_closed_loop_recovery_step(current_visual_error, pre_prior, correction)
                post_norm = recovery_error_norm(float(post_error[0]), float(post_error[1]), float(post_error[2]))
                overshoot = recovery_overshoot_flag(pre_error, post_error)
                overshoot_any = bool(overshoot_any or overshoot)
                error_curve.append(float(post_norm))
                monotonic_prefix = monotonic_decay_prefix(error_curve)

                step_entered_near = in_near_grasp_basin(
                    float(post_error[0]),
                    float(post_error[1]),
                    float(post_error[2]),
                    xy_threshold=float(args.near_grasp_xy_threshold),
                    yaw_threshold=float(args.near_grasp_yaw_threshold),
                )
                step_entered_close = in_close_ready_basin(
                    float(post_error[0]),
                    float(post_error[1]),
                    float(post_error[2]),
                    xy_threshold=float(args.close_ready_xy_threshold),
                    yaw_threshold=float(args.close_ready_yaw_threshold),
                )
                if step_entered_near and steps_to_basin is None:
                    steps_to_basin = int(step + 1)
                if step_entered_close and steps_to_close_ready is None:
                    steps_to_close_ready = int(step + 1)
                entered_near_grasp = bool(entered_near_grasp or step_entered_near)
                entered_close_ready = bool(entered_close_ready or step_entered_close)

                traces.append(
                    {
                        "episode_idx": int(sample.get("episode_idx", -1)),
                        "step_idx": int(sample.get("step_idx", -1)),
                        "trajectory_id": str(sample.get("trajectory_id", "")),
                        "trajectory_step": int(sample.get("trajectory_step", 0)),
                        "candidate_name": candidate_name,
                        "selected_model_name": chosen_name,
                        "closed_loop_step": int(step),
                        "failure_bucket": failure_bucket,
                        "visual_evidence_class": evidence_class,
                        "basin_visual_evidence_class": basin_evidence_class,
                        "pre_error_dx": float(pre_error[0]),
                        "pre_error_dy": float(pre_error[1]),
                        "pre_error_dyaw": float(pre_error[2]),
                        "pred_correction_dx": float(correction[0]),
                        "pred_correction_dy": float(correction[1]),
                        "pred_correction_dyaw": float(correction[2]),
                        "post_error_dx": float(post_error[0]),
                        "post_error_dy": float(post_error[1]),
                        "post_error_dyaw": float(post_error[2]),
                        "error_norm": float(pre_norm),
                        "post_error_norm": float(post_norm),
                        "entered_near_grasp_basin": bool(entered_near_grasp),
                        "entered_close_ready_basin": bool(entered_close_ready),
                        "entered_near_insert_basin": False,
                        "overshoot_flag": bool(overshoot),
                        "monotonic_error_decay": bool(monotonic_prefix),
                        "planner_bias_score": float(sample.get("planner_bias_score", 0.0)),
                        "basin_recovery_mode": str(pred.get("basin_recovery_mode", "")),
                        "basin_recovery_reason": str(pred.get("basin_recovery_reason", "")),
                        "basin_recovery_basin_label": str(pred.get("basin_recovery_basin_label", "")),
                        "basin_recovery_line_search_scale": float(pred.get("basin_recovery_line_search_scale", 1.0)),
                        "basin_recovery_used_visual_geometry": bool(pred.get("basin_recovery_used_visual_geometry", False)),
                        "basin_recovery_used_learned_proposal": bool(pred.get("basin_recovery_used_learned_proposal", False)),
                        "basin_recovery_dry_run_scaled_for_eval": bool(pred.get("basin_recovery_dry_run_scaled_for_eval", False)),
                        "visual_error_dx": float(current_visual_error[0]) if candidate_name == "basin_recovery_supervisor" else float(sample.get("trace_error_dx", 0.0)),
                        "visual_error_dy": float(current_visual_error[1]) if candidate_name == "basin_recovery_supervisor" else float(sample.get("trace_error_dy", 0.0)),
                        "visual_error_dyaw": float(current_visual_error[2]) if candidate_name == "basin_recovery_supervisor" else float(sample.get("trace_error_dyaw", 0.0)),
                        "uses_privileged_target": False,
                        "uses_privileged_label_for_eval": True,
                        "uses_rlbench_mask_runtime": False,
                    }
                )
                current_error = post_error
                current_prior = post_prior

            final_norm = float(error_curve[-1])
            summaries_by_candidate[candidate_name].append(
                {
                    "episode_idx": int(sample.get("episode_idx", -1)),
                    "step_idx": int(sample.get("step_idx", -1)),
                    "trajectory_id": str(sample.get("trajectory_id", "")),
                    "trajectory_step": int(sample.get("trajectory_step", 0)),
                    "failure_bucket": failure_bucket,
                    "visual_evidence_class": evidence_class,
                    "basin_visual_evidence_class": basin_evidence_class,
                    "planner_bias_score": float(sample.get("planner_bias_score", 0.0)),
                    "initial_error_norm": float(initial_norm),
                    "final_error_norm": final_norm,
                    "closed_loop_gain": float(initial_norm - final_norm),
                    "entered_near_grasp_basin": bool(entered_near_grasp),
                    "entered_close_ready_basin": bool(entered_close_ready),
                    "steps_to_near_grasp_basin": steps_to_basin,
                    "steps_to_close_ready_basin": steps_to_close_ready,
                    "overshoot_any": bool(overshoot_any),
                    "monotonic_decay": bool(monotonic_decay_prefix(error_curve)),
                    "error_curve": [float(x) for x in error_curve],
                    "selected_model_sequence": selected_names,
                    "basin_recovery_mode_sequence": basin_mode_sequence,
                    "step_count": int(len(error_curve) - 1),
                    "micro_entry_ready_steps": int(micro_entry_ready_steps),
                    "micro_yaw_active_steps": int(micro_yaw_active_steps),
                }
            )

    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "v11_checkpoint": str(args.v11_checkpoint),
        "v16_checkpoint": str(args.v16_checkpoint),
        "steps": int(args.steps),
        "large_bias_quantile": float(args.large_bias_quantile),
        "large_bias_threshold": float(large_bias_threshold),
        "near_grasp_basin": {
            "xy_threshold": float(args.near_grasp_xy_threshold),
            "yaw_threshold": float(args.near_grasp_yaw_threshold),
        },
        "close_ready_basin": {
            "xy_threshold": float(args.close_ready_xy_threshold),
            "yaw_threshold": float(args.close_ready_yaw_threshold),
        },
        "visual_observable_thresholds": {
            "frame_confidence": float(args.visual_conf_threshold),
            "frame_observability": float(args.visual_observability_threshold),
            "frame_axis_strength": float(args.visual_axis_strength_threshold),
        },
        "basin_recovery": {
            "candidate_name": "basin_recovery_supervisor",
            "visual_observable_thresholds": {
                "frame_confidence": float(args.basin_visual_conf_threshold),
                "frame_observability": float(args.basin_visual_observability_threshold),
                "frame_axis_strength": float(args.basin_visual_axis_strength_threshold),
            },
            "visual_gain": float(args.basin_visual_gain),
            "micro_gain": float(args.basin_micro_gain),
            "eval_line_search_enabled": bool(args.basin_eval_line_search),
            "runtime_policy": "prior_only_reacquire_view__visual_observable_geometry_pullback__near_basin_micro_servo",
        },
        "hard_buckets": hard_buckets,
        "uses_privileged_target": False,
        "uses_privileged_label_for_eval": True,
        "candidates": {},
    }

    for candidate_name, rollouts in summaries_by_candidate.items():
        large_bias_rollouts = [r for r in rollouts if float(r["planner_bias_score"]) >= float(large_bias_threshold)]
        visual_rollouts = [r for r in rollouts if str(r["visual_evidence_class"]) == "visual_observable"]
        prior_only_rollouts = [r for r in rollouts if str(r["visual_evidence_class"]) == "prior_only"]
        hard_rollouts = [r for r in rollouts if str(r["failure_bucket"]) in set(hard_buckets)]
        bucketwise_summary = {}
        for bucket_name in bucket_names:
            bucketwise_summary[bucket_name] = _summarize_rollouts([r for r in rollouts if str(r["failure_bucket"]) == bucket_name], bucket_names=bucket_names)
        report["candidates"][candidate_name] = {
            "summary_all": _summarize_rollouts(rollouts, bucket_names=bucket_names),
            "summary_large_bias": _summarize_rollouts(large_bias_rollouts, bucket_names=bucket_names),
            "summary_visual_observable": _summarize_rollouts(visual_rollouts, bucket_names=bucket_names),
            "summary_prior_only": _summarize_rollouts(prior_only_rollouts, bucket_names=bucket_names),
            "summary_hard_buckets": _summarize_rollouts(hard_rollouts, bucket_names=bucket_names),
            "bucketwise_summary": bucketwise_summary,
        }

    single_step_bucket_report = _load_json(args.v11_single_step_bucket_report)
    single_step_v11_hard_gain: dict[str, float] = {}
    if single_step_bucket_report is not None:
        for item in single_step_bucket_report.get("buckets", []):
            bucket_name = str(item.get("bucket", ""))
            if bucket_name in set(hard_buckets) and item.get("recovery_gain_mean") is not None:
                single_step_v11_hard_gain[bucket_name] = float(item["recovery_gain_mean"])

    zero_basin_entry = float(report["candidates"]["zero_action"]["summary_all"]["basin_entry_rate"])
    prior_basin_entry = float(report["candidates"]["planner_prior_only"]["summary_all"]["basin_entry_rate"])
    acceptance: dict[str, Any] = {
        "single_step_v11_hard_bucket_gain_baseline": single_step_v11_hard_gain,
        "per_candidate": {},
    }
    for candidate_name, candidate_report in report["candidates"].items():
        summary_all = candidate_report["summary_all"]
        bucketwise_summary = candidate_report["bucketwise_summary"]
        hard_bucket_non_regression = True
        hard_bucket_gain_delta: dict[str, float | None] = {}
        for bucket_name in hard_buckets:
            baseline_gain = single_step_v11_hard_gain.get(bucket_name)
            closed_loop_gain = bucketwise_summary.get(bucket_name, {}).get("closed_loop_gain_mean")
            if baseline_gain is None or closed_loop_gain is None:
                hard_bucket_gain_delta[bucket_name] = None
                continue
            delta = float(closed_loop_gain) - float(baseline_gain)
            hard_bucket_gain_delta[bucket_name] = delta
            if float(closed_loop_gain) + 1.0e-9 < float(baseline_gain):
                hard_bucket_non_regression = False
        basin_beats_baselines = bool(
            float(summary_all["basin_entry_rate"]) > float(zero_basin_entry)
            and float(summary_all["basin_entry_rate"]) > float(prior_basin_entry)
        )
        overshoot_ok = bool(float(summary_all["overshoot_rate"]) <= 0.20)
        monotonic_ok = bool(float(summary_all["monotonic_decay_rate"]) >= 0.60)
        acceptance["per_candidate"][candidate_name] = {
            "basin_entry_beats_zero_and_prior": basin_beats_baselines,
            "overshoot_ok": overshoot_ok,
            "monotonic_decay_ok": monotonic_ok,
            "hard_bucket_non_regression_vs_v11_single_step": hard_bucket_non_regression,
            "hard_bucket_gain_delta_vs_v11_single_step": hard_bucket_gain_delta,
            "passes_for_runtime_apply": bool(
                basin_beats_baselines and overshoot_ok and monotonic_ok and hard_bucket_non_regression
            ),
        }
    report["acceptance"] = acceptance

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.trace_output, "w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(args.output)
    print(args.trace_output)


if __name__ == "__main__":
    main()
