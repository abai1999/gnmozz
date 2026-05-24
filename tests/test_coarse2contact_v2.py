from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from prismatic.robot.coarse2contact_v2 import (
    PrecisionObservationBundle,
    PrecisionSkillSupervisor,
    DepthGeometryLocalizerNet,
    RECOVERY_MAINLINE_CHECKPOINT,
    PrecisionTaskRegistry,
    RecoveryFSM,
    RecoveryPhase,
    GraspRecoveryHeadNet,
    augment_recovery_record,
    build_bucket_tail_replay_augmented_records,
    augment_recovery_record_replay,
    build_failure_replay_augmented_records,
    build_failure_replay_library,
    build_targeted_augmented_records,
    RingGraspLocalizer,
    RingSpokeAlignLocalizer,
    BasinRecoverySupervisor,
    BasinRecoveryConfig,
    BasinRecoveryMode,
    VisualEvidenceClass,
    classify_basin_label,
    classify_visual_evidence_for_basin,
    load_precision_task_spec,
)
from prismatic.robot.coarse2contact_v2.recovery_audit import planner_bias_xyyaw, recovery_phase_label, trace_episode_index
from prismatic.robot.coarse2contact_v2.recovery_audit import (
    apply_closed_loop_recovery_step,
    choose_gated_hybrid_candidate,
    in_close_ready_basin,
    in_near_grasp_basin,
    monotonic_decay_prefix,
    recovery_error_norm,
    recovery_overshoot_flag,
)
from prismatic.robot.coarse2contact_v2.recovery_augmentation import failure_morphology_bucket
from prismatic.robot.coarse2contact_v2.learned_localizer import GraspSkillHeadNet
from prismatic.robot.coarse2contact_v2.basin_recovery import BasinStateEstimatorNet, BasinPullbackPolicyNet, GraspOnlyBasinPullbackPolicy
from prismatic.robot.coarse2contact_v2.basin_state import BasinAxisCalibration, BasinStateCalibration, CalibratedGraspBasinEstimator, EstimatedBasinError, FrameRelabelBasinEstimator, ReplayBasinEstimator, load_basin_state_calibration_report
from prismatic.robot.residual_safety import ResidualSafety
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from scripts.audit_c2c_v2_frame_contract_relabel import audit as audit_frame_contract_relabel
from scripts.audit_c2c_v2_grasp_intervention import audit as audit_grasp_intervention
from scripts.run_c2c_v2_grasp_shell_episode_sweep import _summarize_sweep_reports
from scripts.relabel_c2c_v2_privileged_basin_frames import _frame_label_fields
from prismatic.robot.coarse2contact_v2.grasp_probe_shell import grasp_probe_shell_fields


ROOT = Path(__file__).resolve().parents[1]


def _make_observation(
    *,
    blue_center: tuple[int, int] = (28, 48),
    red_center: tuple[int, int] = (76, 48),
    img_size: int = 96,
) -> dict:
    rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    x0, y0 = blue_center
    x1, y1 = red_center
    rgb[max(0, y0 - 8) : min(img_size, y0 + 8), max(0, x0 - 10) : min(img_size, x0 + 10), 2] = 220
    rgb[max(0, y0 - 8) : min(img_size, y0 + 8), max(0, x0 - 10) : min(img_size, x0 + 10), 1] = 60
    rgb[max(0, y0 - 8) : min(img_size, y0 + 8), max(0, x0 - 10) : min(img_size, x0 + 10), 0] = 20
    rgb[:, max(0, x1 - 2) : min(img_size, x1 + 2), 0] = 220
    rgb[:, max(0, x1 - 2) : min(img_size, x1 + 2), 1] = 30
    rgb[:, max(0, x1 - 2) : min(img_size, x1 + 2), 2] = 20
    depth = np.full((img_size, img_size), 0.22, dtype=np.float32)
    return {
        "front_rgb": rgb,
        "wrist_rgb": rgb,
        "front_depth": depth,
        "wrist_depth": depth,
        "gripper_pose": np.array([0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_open": 1.0,
        "gripper_touch_forces": np.zeros(6, dtype=np.float32),
        "joint_positions": np.zeros(7, dtype=np.float32),
    }


class Coarse2ContactV2Tests(unittest.TestCase):
    def test_task_spec_loads(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.task_name, "insert_onto_square_peg")
        self.assertIn("held_square_ring", spec.entities)
        self.assertIn("target_red_spoke", spec.entities)
        self.assertIn("ring_grasp_frame", spec.entities)
        self.assertIn("target_spoke_axis_frame", spec.entities)
        self.assertEqual(spec.get_skill("precision_grasp_ring").target_entity, "ring_grasp_frame")
        self.assertEqual(spec.get_skill("grasp_contact_ring").target_entity, "ring_grasp_frame")
        self.assertEqual(spec.get_skill("precision_align_ring_to_spoke").target_entity, "target_spoke_axis_frame")
        self.assertEqual(spec.default_stage, "COARSE_TO_RING")
        self.assertGreaterEqual(len(spec.stage_graph), 6)

    def test_precision_skill_contract_rejects_object_targets(self) -> None:
        raw = yaml.safe_load((ROOT / "configs" / "coarse2contact" / "tasks" / "insert_onto_square_peg.yaml").read_text(encoding="utf-8"))
        raw["skills"]["precision_grasp_ring"]["target_entity"] = "held_square_ring"
        with self.assertRaises(ValueError):
            from prismatic.robot.coarse2contact_v2.specs import PrecisionTaskSpec

            PrecisionTaskSpec.from_dict(raw)

    def test_frame_to_frame_relabel_synthetic_geometry(self) -> None:
        gripper_pose = np.array([0.0, 0.0, 0.50, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ring_pose = np.array([0.02, -0.01, 0.42, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        spoke_pose = np.array([0.05, 0.03, 0.40, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        trace_row = {
            "c2c_v2_skill_type": "precision_grasp",
            "c2c_v2_stage": "RING_GRASP_ALIGN",
            "planner_local_delta_6d": [0.0] * 6,
            "local_residual_vs_planner_local_6d": [0.0] * 6,
        }
        row = _frame_label_fields(
            task_name="insert_onto_square_peg",
            trace_row=trace_row,
            gripper_pose=gripper_pose,
            ring_pose=ring_pose,
            spoke_pose=spoke_pose,
        )
        self.assertEqual(row["target_frame"], "ring_grasp_frame")
        self.assertEqual(row["reference_frame"], "gripper_jaw_frame")
        self.assertAlmostEqual(float(row["privileged_dx"]), 0.02, places=6)
        self.assertAlmostEqual(float(row["privileged_dy"]), 0.01, places=6)
        self.assertGreater(float(row["privileged_dz"]), 0.0)
        self.assertTrue(row["near_grasp_basin"] or not row["close_ready_basin"])
        self.assertEqual(row["z_semantics"], "descend_progress_to_grasp_frame")

        align_row = _frame_label_fields(
            task_name="insert_onto_square_peg",
            trace_row={"c2c_v2_skill_type": "precision_align", "c2c_v2_stage": "RING_SPOKE_ALIGN"},
            gripper_pose=gripper_pose,
            ring_pose=ring_pose,
            spoke_pose=spoke_pose,
        )
        self.assertEqual(align_row["target_frame"], "target_spoke_axis_frame")
        self.assertEqual(align_row["reference_frame"], "held_ring_aperture_frame")
        self.assertEqual(align_row["z_semantics"], "axis_alignment_depth")
        self.assertTrue(np.isfinite(float(align_row["privileged_dyaw"])))

    def test_frame_relabel_outputs_unified_sample_schema(self) -> None:
        gripper_pose = np.array([0.0, 0.0, 0.50, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ring_pose = np.array([0.02, -0.01, 0.42, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        spoke_pose = np.array([0.05, 0.03, 0.40, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        row = _frame_label_fields(
            task_name="insert_onto_square_peg",
            trace_row={"c2c_v2_skill_type": "precision_grasp", "c2c_v2_stage": "RING_GRASP_ALIGN"},
            gripper_pose=gripper_pose,
            ring_pose=ring_pose,
            spoke_pose=spoke_pose,
        )
        self.assertIn("obs_t", row)
        self.assertIn("planner_prior", row)
        self.assertIn("frame_contract", row)
        self.assertIn("true_basin_error_t", row)
        self.assertIn("action_t", row)
        self.assertIn("true_basin_error_t_plus_1", row)
        self.assertIn("yaw_observable", row)
        self.assertIn("micro_entry_ready", row)
        self.assertEqual(row["frame_contract"]["target_frame"], "ring_grasp_frame")
        self.assertEqual(row["frame_contract"]["reference_frame"], "gripper_jaw_frame")
        self.assertIn("local_delta_6d", row["planner_prior"])
        self.assertIn("local_correction_local_6d", row["action_t"])
        self.assertEqual(row["obs_t"]["visual_observability_class"], row["visual_observability_class"])
        self.assertEqual(row["schema_version"], "frame_residual_v2")
        self.assertIn(row["yaw_observability_class"], {"observable", "ambiguous", "unobservable"})
        self.assertIn(row["takeover_tier"], {"near_basin_shell", "micro_entry_ready", "close_ready", "outside_takeover", "abstain_prior_only", "invalid"})
        self.assertTrue(row["label_valid"])

    def test_frame_relabel_yaw_observability_and_tiers_are_separate(self) -> None:
        gripper_pose = np.array([0.0, 0.0, 0.50, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ring_pose = np.array([0.020, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        spoke_pose = np.array([0.05, 0.03, 0.40, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        observable_row = _frame_label_fields(
            task_name="insert_onto_square_peg",
            trace_row={
                "c2c_v2_skill_type": "precision_grasp",
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "local_geometry_error": {
                    "grasp": {"confidence": 0.9, "observability": 0.2, "fit_residual": 0.01}
                },
            },
            gripper_pose=gripper_pose,
            ring_pose=ring_pose,
            spoke_pose=spoke_pose,
        )
        self.assertEqual(observable_row["yaw_observability_class"], "observable")
        self.assertTrue(observable_row["yaw_observable"])
        self.assertIn(observable_row["takeover_tier"], {"near_basin_shell", "micro_entry_ready", "close_ready"})

        far_row = _frame_label_fields(
            task_name="insert_onto_square_peg",
            trace_row={
                "c2c_v2_skill_type": "precision_grasp",
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "local_geometry_error": {
                    "grasp": {"confidence": 0.9, "observability": 0.2, "fit_residual": 0.01}
                },
            },
            gripper_pose=gripper_pose,
            ring_pose=np.array([0.070, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            spoke_pose=spoke_pose,
        )
        self.assertFalse(far_row["near_basin_shell"])
        self.assertNotEqual(far_row["takeover_tier"], "near_basin_shell")

    def test_frame_contract_audit_splits_yaw_and_takeover_tiers(self) -> None:
        base = {
            "episode_idx": 1,
            "step_idx": 1,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_name": "precision_grasp_ring",
            "skill_type": "precision_grasp",
            "schema_version": "frame_residual_v2",
            "visual_observability_class": "visual_observable",
            "yaw_observability_class": "observable",
            "yaw_observable": True,
            "label_valid": True,
            "near_grasp_basin": False,
            "close_ready_basin": False,
            "micro_entry_ready": False,
            "axis_gate_policy": {"x": "trusted_control", "y": "trusted_control", "z": "diagnostic_only", "yaw": "abstain"},
            "proxy_local_geometry_error": {"dx": 0.03, "dy": 0.0, "dz": 0.02, "dyaw": 0.0},
            "estimated_basin_error": {"dx": 0.03, "dy": 0.0, "dz": 0.02, "dyaw": 0.0},
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "action_t": {"local_correction_local_6d": [0.0] * 6},
            "true_basin_error_t": {"dx": 0.030, "dy": 0.0, "dz": 0.020, "dyaw": 0.0},
            "true_basin_error_t_plus_1": {"dx": 0.026, "dy": 0.0, "dz": 0.019, "dyaw": 0.0},
            "privileged_dx": 0.030,
            "privileged_dy": 0.0,
            "privileged_dz": 0.020,
            "privileged_dyaw": 0.0,
            "next_privileged_dx": 0.026,
            "next_privileged_dy": 0.0,
            "next_privileged_dz": 0.019,
            "next_privileged_dyaw": 0.0,
            "xy_error": 0.030,
            "next_xy_error": 0.026,
            "yaw_abs": 0.0,
            "takeover_tier": "coarse_pullback_candidate",
            "coarse_pullback_candidate": True,
            "overshoot": False,
        }
        blocked = dict(base)
        blocked.update({
            "episode_idx": 2,
            "yaw_observability_class": "unobservable",
            "yaw_observable": False,
            "visual_observability_class": "prior_only",
            "takeover_tier": "abstain_prior_only",
            "coarse_pullback_candidate": False,
            "axis_gate_policy": {"x": "abstain", "y": "abstain", "z": "abstain", "yaw": "abstain"},
        })
        report = audit_frame_contract_relabel([base, blocked])
        self.assertEqual(report["overall"]["schema_version_counts"]["frame_residual_v2"], 2)
        self.assertEqual(report["overall"]["coarse_pullback_candidate_rows"], 1)
        self.assertEqual(report["overall"]["yaw_observability_counts"]["observable"], 1)
        self.assertEqual(report["overall"]["yaw_observability_counts"]["unobservable"], 1)
        tiers = {item["takeover_tier"]: item for item in report["by_takeover_tier"]}
        self.assertIn("coarse_pullback_candidate", tiers)
        self.assertIn("abstain_prior_only", tiers)

    def test_replay_basin_estimator_and_grasp_only_pullback_policy(self) -> None:
        gripper_pose = np.array([0.0, 0.0, 0.50, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ring_pose = np.array([0.02, -0.01, 0.42, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        spoke_pose = np.array([0.05, 0.03, 0.40, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        row = _frame_label_fields(
            task_name="insert_onto_square_peg",
            trace_row={
                "c2c_v2_skill_type": "precision_grasp",
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "local_geometry_error": {
                    "grasp": {
                        "confidence": 0.92,
                        "observability": 0.18,
                        "fit_residual": 0.01,
                    }
                },
            },
            gripper_pose=gripper_pose,
            ring_pose=ring_pose,
            spoke_pose=spoke_pose,
        )
        estimator = FrameRelabelBasinEstimator()
        est = estimator.estimate(row, stage_name="RING_GRASP_ALIGN")
        policy = GraspOnlyBasinPullbackPolicy()
        decision = policy.step(estimated_basin_error=est, visual_evidence_class=row["visual_observability_class"])
        self.assertIn(decision.mode, {BasinRecoveryMode.VISUAL_PULLBACK, BasinRecoveryMode.MICRO_SERVO_TO_BASIN})
        post, _ = apply_closed_loop_recovery_step(
            [float(row["true_basin_error_t"]["dx"]), float(row["true_basin_error_t"]["dy"]), float(row["true_basin_error_t"]["dyaw"])],
            row["planner_prior"]["local_delta_6d"],
            decision.correction_xyyaw,
        )
        self.assertLessEqual(
            recovery_error_norm(float(post[0]), float(post[1]), float(post[2])),
            recovery_error_norm(float(row["true_basin_error_t"]["dx"]), float(row["true_basin_error_t"]["dy"]), float(row["true_basin_error_t"]["dyaw"])),
        )
        prior_only = policy.step(estimated_basin_error=est, visual_evidence_class=VisualEvidenceClass.PRIOR_ONLY)
        self.assertEqual(prior_only.mode, BasinRecoveryMode.REACQUIRE_VIEW)
        self.assertLess(float(prior_only.local_action_6d[2]), 0.0)

    def test_pullback_ready_axes_do_not_imply_close_ready(self) -> None:
        est = EstimatedBasinError(
            valid=True,
            confidence=0.92,
            dx=0.010,
            dy=0.008,
            dz=0.030,
            dyaw=0.120,
            x_valid=True,
            y_valid=True,
            z_valid=False,
            yaw_valid=False,
            x_confidence=0.9,
            y_confidence=0.9,
            z_confidence=0.1,
            yaw_confidence=0.0,
            frame_consistency=0.85,
            source="unit_test",
            reason="xy_pullback_only",
        )
        self.assertIn("x", est.pullback_ready_axes)
        self.assertIn("y", est.pullback_ready_axes)
        self.assertFalse(est.close_ready(xy_threshold=0.005, z_threshold=0.010, yaw_threshold=0.03, yaw_required=True))

    def test_xy_pullback_can_apply_while_micro_entry_false(self) -> None:
        row = {
            "task_name": "insert_onto_square_peg",
            "stage_name": "RING_GRASP_ALIGN",
            "skill_name": "precision_grasp_ring",
            "skill_type": "precision_grasp",
            "visual_observability_class": "visual_observable",
            "reacquire_needed": False,
            "yaw_observable": False,
            "true_basin_error_t": {"dx": 0.020, "dy": -0.016, "dz": 0.028, "dyaw": 0.110},
            "planner_prior": {"local_delta_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "proxy_local_geometry_error": {"dx": 0.030, "dy": -0.020, "dz": 0.030, "dyaw": 0.110},
            "estimated_basin_error": {
                "dx": 0.020,
                "dy": -0.016,
                "dz": 0.028,
                "dyaw": 0.110,
                "axis_validity": {"x": True, "y": True, "z": False, "yaw": False},
                "axis_confidence": {"x": 0.8, "y": 0.8, "z": 0.1, "yaw": 0.0},
            },
        }
        replay = ReplayBasinEstimator(xy_gain=0.35, max_xy_step=0.0030, yaw_enabled=False, z_enabled=False)
        result = replay.replay(row, stage_name="RING_GRASP_ALIGN")
        self.assertEqual(result.estimated_basin_error.pullback_ready_axes[:2], ("x", "y"))
        self.assertEqual(result.estimated_basin_error.source, "privileged_relabel")
        self.assertEqual(result.estimated_basin_error.reason, "replay_xy_pullback")
        self.assertEqual(result.reason, "privileged_relabel_replay")
        self.assertEqual(result.micro_entry_block_reason, "privileged_relabel_replay")
        self.assertFalse(result.micro_entry_ready)
        self.assertFalse(result.close_ready_ready)
        self.assertEqual(result.reason, "privileged_relabel_replay")
        self.assertEqual(result.correction_local_6d.shape[0], 6)
        self.assertGreater(np.linalg.norm(result.correction_local_6d[:2]), 0.0)

    def test_close_stays_locked_until_entry_gate(self) -> None:
        row = {
            "task_name": "insert_onto_square_peg",
            "stage_name": "RING_GRASP_ALIGN",
            "skill_name": "precision_grasp_ring",
            "skill_type": "precision_grasp",
            "visual_observability_class": "visual_observable",
            "reacquire_needed": False,
            "yaw_observable": True,
            "true_basin_error_t": {"dx": 0.022, "dy": 0.017, "dz": 0.032, "dyaw": 0.090},
            "planner_prior": {"local_delta_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            "proxy_local_geometry_error": {"dx": 0.025, "dy": 0.019, "dz": 0.032, "dyaw": 0.090},
            "estimated_basin_error": {
                "dx": 0.022,
                "dy": 0.017,
                "dz": 0.032,
                "dyaw": 0.090,
                "axis_validity": {"x": True, "y": True, "z": False, "yaw": True},
                "axis_confidence": {"x": 0.8, "y": 0.8, "z": 0.1, "yaw": 0.7},
            },
        }
        replay = ReplayBasinEstimator(xy_gain=0.35, max_xy_step=0.0030, yaw_enabled=False, z_enabled=False)
        proposal = replay.propose(row, stage_name="RING_GRASP_ALIGN")
        self.assertEqual(proposal["mode"], "VISUAL_PULLBACK")
        result = replay.replay(row, stage_name="RING_GRASP_ALIGN")
        self.assertFalse(result.micro_entry_ready)
        self.assertFalse(result.close_ready_ready)
        self.assertGreater(np.linalg.norm(result.correction_local_6d[:2]), 0.0)

    def test_grasp_pullback_policy_uses_explicit_z_threshold(self) -> None:
        est = EstimatedBasinError(
            valid=True,
            confidence=0.9,
            dx=0.004,
            dy=0.0,
            dz=0.020,
            dyaw=0.0,
            x_valid=True,
            y_valid=True,
            z_valid=True,
            yaw_valid=False,
            x_confidence=0.9,
            y_confidence=0.9,
            z_confidence=0.9,
            yaw_confidence=0.0,
            frame_consistency=0.9,
            source="unit_test",
            reason="z_threshold_regression",
        )
        config = BasinRecoveryConfig(close_ready_z_threshold=0.010, close_ready_yaw_threshold=0.030)
        policy = GraspOnlyBasinPullbackPolicy(config=config)
        decision = policy.step(estimated_basin_error=est, visual_evidence_class=VisualEvidenceClass.VISUAL_OBSERVABLE)
        self.assertEqual(decision.mode, BasinRecoveryMode.VISUAL_PULLBACK)
        self.assertGreater(abs(float(decision.local_action_6d[0])), config.max_micro_xy_step + 1.0e-6)

    def test_grasp_intervention_audit_feasible_horizon_shell(self) -> None:
        rows = [
            {
                "episode_idx": 8,
                "step": 58,
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": True,
                "grasp_probe_reason": "replay_oracle_xy",
                "grasp_probe_visibility_bucket": "visual_observable",
                "grasp_probe_requested_horizon": 3,
                "grasp_probe_horizon_steps_executed": 3,
                "grasp_probe_queue_len_before": 2,
                "grasp_probe_queue_len_after": 0,
                "grasp_probe_queue_flushed": True,
                "grasp_probe_pre_true_error_t": [0.023, 0.0, 0.0, 0.040],
                "grasp_probe_post_true_error_t": [0.020, 0.0, 0.0, 0.040],
                "grasp_probe_horizon_final_true_error_t": [0.014, 0.0, 0.0, 0.040],
                "grasp_probe_near_grasp_after": False,
                "grasp_probe_horizon_near_grasp_after": True,
                "grasp_probe_horizon_micro_entry_ready_after": True,
                "grasp_probe_horizon_overshoot": False,
            }
        ]
        report = audit_grasp_intervention(
            rows,
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        shells = report["overall"]["feasible_shells"]
        self.assertEqual(shells["one_step_xy_feasible"]["count"], 0)
        self.assertEqual(shells["horizon_xy_and_yaw_feasible"]["count"], 1)
        self.assertEqual(shells["horizon_xy_and_yaw_feasible"]["horizon_near_grasp_after_rate"], 1.0)
        self.assertEqual(report["overall"]["horizon_xy_contraction_rate"], 1.0)
        self.assertEqual(report["overall"]["yaw_blocked_rate_within_horizon_xy_feasible"], 0.0)
        self.assertEqual(report["overall"]["ring_grasp_align_dwell_steps"]["total"], 1)
        self.assertEqual(report["overall"]["queue_protocol"]["queue_flushed_rate"], 1.0)
        self.assertEqual(report["overall"]["yaw_observable_rows"], 1)
        self.assertEqual(report["overall"]["micro_entry_ready_rows"], 1)

    def test_grasp_intervention_audit_counts_coarse_pullback_candidates(self) -> None:
        rows = [
            {
                "episode_idx": 9,
                "step": 44,
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": False,
                "grasp_probe_reason": "inactive",
                "grasp_probe_visibility_bucket": "visual_observable",
                "grasp_probe_near_basin_shell": False,
                "grasp_probe_horizon_xy_feasible": False,
                "grasp_probe_yaw_feasible": True,
                "grasp_probe_pre_true_error_t": [0.040, 0.0, 0.0, 0.040],
                "grasp_probe_post_true_error_t": [0.038, 0.0, 0.0, 0.040],
            }
        ]
        report = audit_grasp_intervention(
            rows,
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        self.assertEqual(report["overall"]["near_basin_shell_rows"], 0)
        self.assertEqual(report["overall"]["coarse_pullback_candidate_rows"], 1)
        self.assertEqual(report["overall"]["yaw_observable_rows"], 1)

    def test_grasp_probe_shell_fields_separate_coarse_and_near_windows(self) -> None:
        shell_fields = grasp_probe_shell_fields(
            np.array([0.028, 0.0, 0.0, 0.040], dtype=np.float32),
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        self.assertFalse(shell_fields["grasp_probe_near_basin_shell"])
        self.assertTrue(shell_fields["grasp_probe_yaw_feasible"])
        self.assertTrue(shell_fields["grasp_probe_coarse_pullback_candidate"])

    def test_grasp_intervention_audit_splits_yaw_window_and_queue(self) -> None:
        rows = [
            {
                "episode_idx": 3,
                "step": 1,
                "c2c_v2_stage": "RECOVER",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": False,
                "grasp_probe_reason": "inactive",
                "grasp_probe_visibility_bucket": "visual_observable",
            },
            {
                "episode_idx": 3,
                "step": 2,
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": True,
                "grasp_probe_reason": "replay_oracle_xy",
                "grasp_probe_visibility_bucket": "visual_observable",
                "grasp_probe_requested_horizon": 3,
                "grasp_probe_horizon_steps_executed": 3,
                "grasp_probe_queue_len_before": 4,
                "grasp_probe_queue_len_after": 0,
                "grasp_probe_queue_flushed": True,
                "grasp_probe_pre_true_error_t": [0.020, 0.0, 0.0, 0.12],
                "grasp_probe_post_true_error_t": [0.018, 0.0, 0.0, 0.12],
                "grasp_probe_horizon_final_true_error_t": [0.014, 0.0, 0.0, 0.12],
                "grasp_probe_horizon_overshoot": False,
            },
            {
                "episode_idx": 4,
                "step": 2,
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": True,
                "grasp_probe_reason": "replay_oracle_xy",
                "grasp_probe_visibility_bucket": "visual_observable",
                "grasp_probe_requested_horizon": 3,
                "grasp_probe_horizon_steps_executed": 3,
                "grasp_probe_queue_len_before": 4,
                "grasp_probe_queue_len_after": 4,
                "grasp_probe_queue_flushed": False,
                "grasp_probe_pre_true_error_t": [0.020, 0.0, 0.0, 0.04],
                "grasp_probe_post_true_error_t": [0.021, 0.0, 0.0, 0.04],
                "grasp_probe_horizon_final_true_error_t": [0.021, 0.0, 0.0, 0.04],
                "grasp_probe_horizon_overshoot": True,
            },
        ]
        report = audit_grasp_intervention(
            rows,
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        self.assertEqual(report["overall"]["yaw_blocked_rate_within_horizon_xy_feasible"], 0.5)
        self.assertEqual(report["overall"]["recover_preempt_rate_before_first_probe_step"], 0.5)
        self.assertEqual(report["overall"]["ring_grasp_align_dwell_steps"]["total"], 2)
        self.assertEqual(report["overall"]["queue_protocol"]["queue_flushed_rate"], 0.5)
        self.assertGreater(report["overall"]["queue_protocol"]["queue_flush_ablation_delta"], 0.0)

    def test_forced_shell_candidates_are_counted_separately_from_runtime_stage(self) -> None:
        rows = [
            {
                "episode_idx": 7,
                "step": 12,
                "c2c_v2_stage": "RECOVER",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": True,
                "grasp_probe_reason": "replay_oracle_xy",
                "grasp_probe_stage_source": "forced_shell",
                "grasp_probe_visibility_bucket": "visual_observable",
                "grasp_probe_near_basin_shell": True,
                "grasp_probe_horizon_xy_feasible": True,
                "grasp_probe_yaw_feasible": True,
                "grasp_probe_requested_horizon": 3,
                "grasp_probe_horizon_steps_executed": 3,
                "grasp_probe_pre_true_error_t": [0.023, 0.0, 0.0, 0.04],
                "grasp_probe_post_true_error_t": [0.020, 0.0, 0.0, 0.04],
                "grasp_probe_horizon_final_true_error_t": [0.014, 0.0, 0.0, 0.04],
                "grasp_probe_horizon_near_grasp_after": True,
            }
        ]
        report = audit_grasp_intervention(rows)
        self.assertEqual(report["overall"]["active_rows"], 1)
        self.assertEqual(report["overall"]["near_basin_shell_rows"], 1)
        self.assertEqual(report["overall"]["stage_source_counts"]["forced_shell"], 1)

    def test_grasp_intervention_audit_includes_episode_bucket_shell_hits(self) -> None:
        rows = [
            {
                "episode_idx": 2,
                "step": 10,
                "failure_bucket": "small_xy_small_yaw",
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": True,
                "grasp_probe_reason": "replay_oracle_xy",
                "grasp_probe_stage_source": "forced_shell",
                "grasp_probe_visibility_bucket": "visual_observable",
                "grasp_probe_near_basin_shell": True,
                "grasp_probe_yaw_feasible": True,
                "grasp_probe_horizon_xy_feasible": True,
                "grasp_probe_requested_horizon": 3,
                "grasp_probe_horizon_steps_executed": 3,
                "grasp_probe_pre_true_error_t": [0.021, 0.0, 0.0, 0.04],
                "grasp_probe_post_true_error_t": [0.018, 0.0, 0.0, 0.04],
                "grasp_probe_horizon_final_true_error_t": [0.014, 0.0, 0.0, 0.04],
                "grasp_probe_horizon_near_grasp_after": True,
            },
            {
                "episode_idx": 2,
                "step": 11,
                "failure_bucket": "large_xy_small_yaw",
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": True,
                "grasp_probe_reason": "replay_oracle_xy",
                "grasp_probe_stage_source": "forced_shell",
                "grasp_probe_visibility_bucket": "visual_observable",
                "grasp_probe_near_basin_shell": False,
                "grasp_probe_yaw_feasible": False,
                "grasp_probe_horizon_xy_feasible": False,
                "grasp_probe_requested_horizon": 3,
                "grasp_probe_horizon_steps_executed": 3,
                "grasp_probe_pre_true_error_t": [0.030, 0.0, 0.0, 0.12],
                "grasp_probe_post_true_error_t": [0.028, 0.0, 0.0, 0.12],
                "grasp_probe_horizon_final_true_error_t": [0.025, 0.0, 0.0, 0.12],
                "grasp_probe_horizon_near_grasp_after": False,
            },
        ]
        report = audit_grasp_intervention(rows)
        hits = [
            item for item in report["by_episode_failure_bucket"]
            if item["near_basin_shell_rows"] > 0
        ]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["episode_idx"], 2)
        self.assertEqual(hits[0]["failure_bucket"], "small_xy_small_yaw")

    def test_prior_only_probe_rows_remain_abstain_in_audit(self) -> None:
        rows = [
            {
                "episode_idx": 5,
                "step": 10,
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": False,
                "grasp_probe_reason": "prior_only_abstain",
                "grasp_probe_visibility_bucket": "prior_only",
                "grasp_probe_pre_true_error_t": [0.030, 0.0, 0.0, 0.0],
                "grasp_probe_post_true_error_t": [0.030, 0.0, 0.0, 0.0],
            }
        ]
        report = audit_grasp_intervention(rows)
        self.assertEqual(report["overall"]["active_rows"], 0)
        self.assertEqual(report["overall"]["prior_only_abstain_rate"], 1.0)

    def test_grasp_shell_episode_sweep_ranks_shell_hits(self) -> None:
        chunk_reports = [
            {
                "chunk_tag": "chunk_000",
                "by_episode": [
                    {
                        "episode_idx": 2,
                        "near_basin_shell_rows": 0,
                        "horizon_xy_feasible_rows": 1,
                        "yaw_feasible_rows": 0,
                        "active_count": 2,
                    },
                    {
                        "episode_idx": 7,
                        "near_basin_shell_rows": 3,
                        "horizon_xy_feasible_rows": 4,
                        "yaw_feasible_rows": 2,
                        "active_count": 5,
                    },
                ],
                "by_episode_failure_bucket": [
                    {
                        "episode_idx": 2,
                        "failure_bucket": "large_xy_small_yaw",
                        "near_basin_shell_rows": 0,
                        "horizon_xy_feasible_rows": 1,
                        "yaw_feasible_rows": 0,
                        "active_count": 2,
                    },
                    {
                        "episode_idx": 7,
                        "failure_bucket": "small_xy_small_yaw",
                        "near_basin_shell_rows": 3,
                        "horizon_xy_feasible_rows": 4,
                        "yaw_feasible_rows": 2,
                        "active_count": 5,
                    },
                ],
            }
        ]
        summary = _summarize_sweep_reports(chunk_reports, top_k=4, focus_radius=1)
        self.assertEqual(summary["shell_hit_episode_indices"], [7])
        self.assertEqual(summary["shell_hit_failure_buckets"], ["small_xy_small_yaw"])
        self.assertEqual(summary["recommended_next_episode_indices"], [7])
        self.assertEqual(summary["recommended_focus_episode_indices"], [6, 7, 8])
        self.assertEqual(summary["shell_hit_episode_focus_windows"][0]["center_episode_idx"], 7)
        self.assertEqual(summary["shell_hit_episode_focus_windows"][0]["episode_indices"], [6, 7, 8])
        self.assertEqual(summary["ranked_episodes"][0]["episode_idx"], 7)
        self.assertEqual(summary["ranked_episode_buckets"][0]["failure_bucket"], "small_xy_small_yaw")

    def test_grasp_shell_episode_sweep_reports_collection_target(self) -> None:
        chunk_reports = [
            {
                "chunk_tag": "chunk_000",
                "overall": {
                    "active_rows": 120,
                    "yaw_feasible_rows": 35,
                    "horizon_xy_contracted_count": 112,
                    "horizon_near_grasp_after_count": 4,
                },
                "by_episode": [],
                "by_episode_failure_bucket": [],
            }
        ]
        summary = _summarize_sweep_reports(chunk_reports, top_k=4, focus_radius=1)
        self.assertTrue(summary["collection_target"]["meets_target"])
        self.assertGreater(summary["collection_target"]["horizon_xy_contraction_lower_ci"], 0.8)
        self.assertEqual(summary["collection_target"]["active_rows"], 120)
        self.assertEqual(summary["collection_target"]["yaw_feasible_rows"], 35)
        self.assertGreater(summary["collection_target"]["horizon_near_grasp_after_rate"], 0.0)

    def test_recovery_mainline_points_to_v11(self) -> None:
        self.assertIn("grasp_recovery_head_v11_runtime_failure", str(RECOVERY_MAINLINE_CHECKPOINT))

    def test_registry_unknown_task_falls_back(self) -> None:
        registry = PrecisionTaskRegistry(config_root=ROOT / "configs" / "coarse2contact" / "tasks")
        self.assertIsNone(registry.load("does_not_exist"))

    def test_world_local_round_trip(self) -> None:
        delta = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02], dtype=np.float32)
        quat = np.array([0.1, 0.2, -0.3, 0.9], dtype=np.float32)
        quat = quat / np.linalg.norm(quat)
        local = world_delta_to_local(delta, quat)
        world = local_delta_to_world(local, quat)
        np.testing.assert_allclose(world, delta, atol=1e-6)

    def test_clip_attenuation(self) -> None:
        safety = ResidualSafety(max_residual_pos=0.005, max_residual_rot=0.03, max_delta_pos=0.02, max_delta_rot=0.1)
        clipped = safety.clip_residual(np.array([0.02, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float32))
        self.assertLessEqual(np.linalg.norm(clipped[:3]), 0.005 + 1e-6)
        self.assertLessEqual(np.linalg.norm(clipped[3:6]), 0.03 + 1e-6)
        final = safety.clip_final_action(np.array([0.1, -0.1, 0.1, 0.2, -0.2, 0.2, 1.0], dtype=np.float32))
        self.assertLessEqual(np.max(np.abs(final[:3])), 0.02 + 1e-6)
        self.assertLessEqual(np.max(np.abs(final[3:6])), 0.1 + 1e-6)

    def test_recovery_fsm_progression(self) -> None:
        safety = ResidualSafety()
        fsm = RecoveryFSM(safety=safety)
        phase_names = []
        for idx in range(8):
            result = fsm.step(force_reading=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), local_base=np.zeros(6, dtype=np.float32), invalid_action=(idx == 0), jam=False)
            phase_names.append(result.state_name)
        for wanted in ["BACKOFF", "UNLOAD", "MICRO_SEARCH", "REAPPROACH"]:
            self.assertIn(wanted, phase_names)
        self.assertLess(phase_names.index("BACKOFF"), phase_names.index("UNLOAD"))
        self.assertLess(phase_names.index("UNLOAD"), phase_names.index("MICRO_SEARCH"))
        self.assertLess(phase_names.index("MICRO_SEARCH"), phase_names.index("REAPPROACH"))

    def test_localizers_shadow_signal(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        grasp_skill = spec.get_skill("precision_grasp_ring")
        align_skill = spec.get_skill("precision_align_ring_to_spoke")
        obs = _make_observation()
        bundle = PrecisionObservationBundle.from_observation(obs)
        grasp_error = RingGraspLocalizer().localize(bundle, {}, spec, grasp_skill, stage_name="RING_GRASP_ALIGN")
        spoke_error = RingSpokeAlignLocalizer().localize(bundle, {}, spec, align_skill, stage_name="RING_SPOKE_ALIGN")
        self.assertTrue(grasp_error.confidence >= 0.0)
        self.assertTrue(spoke_error.confidence >= 0.0)
        self.assertNotEqual(grasp_error.reason, "missing_rgbd")
        self.assertNotEqual(spoke_error.reason, "missing_rgbd_or_ring")

    def test_owner_by_stage_applies_only_allowed_dofs(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        spec.runtime_flags["precision_takeover_stable_frames"] = 1
        spec.runtime_flags["precision_grasp_takeover_error_xy_max"] = 0.040
        spec.runtime_flags["precision_grasp_takeover_depth_max"] = 0.20
        spec.runtime_flags["precision_grasp_takeover_confidence_min"] = 0.0
        spec.runtime_flags["precision_grasp_takeover_observability_min"] = 0.0
        sup = PrecisionSkillSupervisor(spec, mode="grasp_depth_apply", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        sup._basin_state_estimator = CalibratedGraspBasinEstimator(
            BasinStateCalibration(
                x=BasinAxisCalibration(valid=True, sign=1.0, scale=1.0, confidence=1.0, source="unit_test", reason="boost_x"),
                y=BasinAxisCalibration(valid=True, sign=1.0, scale=1.0, confidence=1.0, source="unit_test", reason="boost_y"),
                z=BasinAxisCalibration(valid=True, sign=1.0, scale=1.0, confidence=1.0, source="unit_test", reason="boost_z"),
                yaw=BasinAxisCalibration(valid=False, sign=1.0, scale=1.0, confidence=0.0, source="unit_test", reason="yaw_abstain"),
            )
        )
        sup._precision_gate_active["precision_grasp"] = True
        sup._precision_gate_stable_frames["precision_grasp"] = 1
        obs = _make_observation(blue_center=(48, 48))
        obs["wrist_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        obs["front_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        planner_delta = np.zeros(7, dtype=np.float32)
        action = sup.step(planner_delta, obs, robot_state={}, task_spec=spec, current_instruction="put the ring on the red spoke")
        trace = sup.get_last_trace()
        self.assertEqual(trace["c2c_v2_owner"], "c2c_depth")
        self.assertEqual(trace["c2c_v2_skill_type"], "precision_grasp")
        self.assertFalse(trace["localizer_abstained"])
        self.assertTrue(trace["c2c_gate_active"])
        self.assertAlmostEqual(float(action[2]), 0.0, places=6)
        self.assertAlmostEqual(float(action[3]), 0.0, places=6)
        self.assertAlmostEqual(float(action[4]), 0.0, places=6)

    def test_precision_gate_blocks_early_takeover_until_coarse_approach_finishes(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        sup = PrecisionSkillSupervisor(spec, mode="basin_recovery_only", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        obs = _make_observation(blue_center=(10, 12))
        planner_delta = np.array([0.012, -0.010, 0.006, 0.0, 0.0, 0.020, 1.0], dtype=np.float32)
        action = sup.step(
            planner_delta,
            obs,
            robot_state={
                "invalid_action_flag": False,
                "wrist_valid_depth_ratio": 0.8,
                "wrist_depth_near_fraction": 0.15,
                "wrist_is_occluded": False,
                "wrist_is_low_visibility": False,
            },
            task_spec=spec,
            current_instruction="put the ring on the red spoke",
        )
        trace = sup.get_last_trace()
        np.testing.assert_allclose(action, planner_delta, atol=1e-6)
        self.assertFalse(trace["c2c_gate_active"])
        self.assertEqual(trace["phase_owner"], "planner")
        self.assertIn(trace["c2c_gate_reason"], {"waiting_absolute_nearfield_depth", "waiting_target_near_precision_basin", "waiting_visible_confident_localizer"})

    def test_precision_gate_requires_absolute_nearfield_depth(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        spec.runtime_flags["precision_takeover_stable_frames"] = 2
        spec.runtime_flags["precision_grasp_takeover_error_xy_max"] = 0.040
        spec.runtime_flags["precision_grasp_takeover_depth_max"] = 0.20
        spec.runtime_flags["precision_grasp_takeover_confidence_min"] = 0.0
        spec.runtime_flags["precision_grasp_takeover_observability_min"] = 0.0
        sup = PrecisionSkillSupervisor(spec, mode="basin_recovery_only", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        planner_delta = np.array([0.002, -0.002, 0.001, 0.0, 0.0, 0.01, 1.0], dtype=np.float32)
        obs_hi = _make_observation(blue_center=(48, 48))
        obs_hi["wrist_depth"] = np.full((96, 96), 0.22, dtype=np.float32)
        obs_hi["front_depth"] = np.full((96, 96), 0.22, dtype=np.float32)
        obs_lo = _make_observation(blue_center=(48, 48))
        obs_lo["wrist_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        obs_lo["front_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        _ = sup.step(planner_delta, obs_hi, robot_state={}, task_spec=spec, current_instruction="put the ring on the red spoke")
        self.assertFalse(bool(sup.get_last_trace().get("c2c_gate_active", False)))
        _ = sup.step(planner_delta, obs_lo, robot_state={}, task_spec=spec, current_instruction="put the ring on the red spoke")
        trace = sup.get_last_trace()
        self.assertFalse(bool(trace["c2c_gate_active"]))
        self.assertIn(trace["c2c_gate_reason"], {"stable_near_precision_basin", "waiting_absolute_nearfield_depth", "waiting_target_near_precision_basin"})
        self.assertLessEqual(float(trace["c2c_gate_nearfield_depth_m"]), float(trace["c2c_gate_nearfield_depth_max"]) + 1e-6)

    def test_basin_recovery_budget_does_not_advance_before_gate(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        sup = PrecisionSkillSupervisor(spec, mode="basin_recovery_only", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        obs = _make_observation(blue_center=(4, 4))
        planner_delta = np.array([0.012, -0.010, 0.006, 0.0, 0.0, 0.020, 1.0], dtype=np.float32)
        steps = []
        for _ in range(3):
            _ = sup.step(
                planner_delta,
                obs,
                robot_state={
                    "invalid_action_flag": False,
                    "wrist_valid_depth_ratio": 0.1,
                    "wrist_depth_near_fraction": 0.0,
                    "wrist_is_occluded": True,
                    "wrist_is_low_visibility": True,
                },
                task_spec=spec,
                current_instruction="put the ring on the red spoke",
            )
            steps.append(int(sup.get_last_trace().get("basin_recovery_recovery_step", -1)))
            self.assertFalse(bool(sup.get_last_trace().get("c2c_gate_active", False)))
        self.assertEqual(steps, [0, 0, 0])

    def test_basin_recovery_switches_from_pullback_to_micro_on_xy_basin(self) -> None:
        cfg = BasinRecoveryConfig(micro_entry_xy_threshold=0.006, micro_entry_required_frames=2, yaw_control_enabled=True)
        sup = BasinRecoverySupervisor(config=cfg)
        record = {
            "frame_confidence": 0.6,
            "frame_observability": 0.01,
            "frame_axis_strength": 1.0e-4,
            "wide_ring_visible": True,
            "wide_ring_dx": 0.0,
            "wide_ring_dy": 0.0,
            "wide_ring_observability": 0.01,
        }
        vis_error = np.array([0.003, 0.004, 2.5], dtype=np.float32)
        dec1 = sup.step(record=record, planner_prior_state=np.zeros(6, dtype=np.float32), visual_error_state=vis_error)
        dec2 = sup.step(record=record, planner_prior_state=np.zeros(6, dtype=np.float32), visual_error_state=vis_error)
        self.assertEqual(dec1.mode, BasinRecoveryMode.VISUAL_PULLBACK)
        self.assertEqual(dec2.mode, BasinRecoveryMode.MICRO_SERVO_TO_BASIN)
        self.assertFalse(dec2.micro_yaw_active)
        self.assertGreaterEqual(dec2.micro_entry_xy_error, 0.0)

    def test_shadow_mode_does_not_double_add_planner_delta(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        sup = PrecisionSkillSupervisor(spec, mode="c2c_stage_shadow", shadow_only=True)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        obs = _make_observation()
        planner_delta = np.array([0.01, -0.02, 0.02, 0.04, -0.05, 0.06, 1.0], dtype=np.float32)
        action = sup.step(planner_delta, obs, robot_state={}, task_spec=spec, current_instruction="put the ring on the red spoke")
        trace = sup.get_last_trace()
        np.testing.assert_allclose(action, planner_delta, atol=1e-6)
        np.testing.assert_allclose(np.asarray(trace["executed_action_world_6d"], dtype=np.float32), planner_delta[:6], atol=1e-6)
        np.testing.assert_allclose(np.asarray(trace["pre_clip_action_world_6d"], dtype=np.float32), planner_delta[:6], atol=1e-6)
        np.testing.assert_allclose(np.asarray(trace["local_command_local_6d"], dtype=np.float32), np.asarray(trace["planner_chunk_local_6d"], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(np.asarray(trace["local_residual_vs_planner_local_6d"], dtype=np.float32), np.zeros(6, dtype=np.float32), atol=1e-6)

    def test_heatmap_localizer_exposes_axis_endpoints(self) -> None:
        model = DepthGeometryLocalizerNet.from_vocab(
            {
                "skill_type": {"<unk>": 0, "precision_grasp": 1},
                "stage_name": {"<unk>": 0, "RING_GRASP_ALIGN": 1},
                "entity": {"<unk>": 0, "held_square_ring": 1, "target_red_spoke": 2},
                "controlled_dofs": {"<unk>": 0, "x": 1, "y": 2, "yaw": 3},
            },
            prediction_mode="heatmap",
            heatmap_size=32,
            heatmap_sigma=0.9,
            heatmap_xy_range_m=0.04,
            heatmap_channels=3,
            heatmap_pos_weight=8.0,
        )
        image = torch.zeros((1, 6, 32, 32), dtype=torch.float32)
        skill_type_id = torch.tensor([1], dtype=torch.long)
        stage_id = torch.tensor([1], dtype=torch.long)
        target_id = torch.tensor([1], dtype=torch.long)
        reference_id = torch.tensor([2], dtype=torch.long)
        dof_vec = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
        out = model.predict(image, skill_type_id, stage_id, target_id, reference_id, dof_vec)
        self.assertIn("axis_pos_u", out)
        self.assertIn("axis_neg_u", out)
        self.assertIn("confidence", out)

    def test_grasp_skill_head_exposes_yaw_observable_logit(self) -> None:
        model = GraspSkillHeadNet()
        image = torch.zeros((2, 6, 128, 128), dtype=torch.float32)
        frame = torch.zeros((2, 10), dtype=torch.float32)
        proprio = torch.zeros((2, 15), dtype=torch.float32)
        out = model(image, frame, proprio)
        self.assertIn("yaw_observable_logit", out)
        self.assertEqual(tuple(out["yaw_observable_logit"].shape), (2,))

    def test_grasp_recovery_head_exposes_prior_conditioning(self) -> None:
        model = GraspRecoveryHeadNet()
        image = torch.zeros((2, 6, 128, 128), dtype=torch.float32)
        frame = torch.zeros((2, 10), dtype=torch.float32)
        proprio = torch.zeros((2, 15), dtype=torch.float32)
        prior = torch.zeros((2, 6), dtype=torch.float32)
        out = model(image, frame, proprio, prior)
        self.assertIn("dx", out)
        self.assertIn("dyaw", out)
        self.assertIn("confidence_logit", out)
        self.assertEqual(tuple(out["dx"].shape), (2,))

    def test_recovery_targeted_augmentation_scales_prior(self) -> None:
        report = {
            "large_bias_threshold": 1.0e-4,
            "trajectory_summaries": [
                {"trajectory_id": "hard", "gain_mean": -0.01, "bias_score_max": 2.0e-4, "num_steps": 12},
                {"trajectory_id": "easy", "gain_mean": 0.01, "bias_score_max": 5.0e-5, "num_steps": 12},
            ],
        }
        base = {
            "trajectory_id": "hard",
            "planner_prior_delta": [0.01, -0.02, 0.0, 0.0, 0.0, 0.03],
            "planner_bias_score": 2.0e-4,
            "recovery_needed": True,
        }
        aug = augment_recovery_record(base, scale=1.5, template={"planner_bias_components": {"dx": {"median": 0.001}, "dy": {"median": -0.002}, "dyaw": {"median": 0.01}}, "bias_score_threshold": 1.0e-4})
        self.assertEqual(aug["is_augmented"], 1.0)
        self.assertGreater(float(aug["planner_bias_score"]), float(base["planner_bias_score"]))

    def test_failure_morphology_bucket(self) -> None:
        self.assertEqual(
            failure_morphology_bucket({"recovery_target_dx": 0.08, "recovery_target_dy": 0.02, "recovery_target_dyaw": 0.3}),
            "large_xy_large_yaw",
        )
        self.assertEqual(
            failure_morphology_bucket({"recovery_target_dx": 0.08, "recovery_target_dy": 0.01, "recovery_target_dyaw": 0.02}),
            "large_xy_small_yaw",
        )
        self.assertEqual(
            failure_morphology_bucket({"recovery_target_dx": 0.02, "recovery_target_dy": -0.01, "recovery_target_dyaw": 0.3}),
            "small_xy_large_yaw",
        )
        self.assertEqual(
            failure_morphology_bucket({"recovery_target_dx": 0.02, "recovery_target_dy": -0.01, "recovery_target_dyaw": 0.03}),
            "small_xy_small_yaw",
        )

    def test_recovery_replay_augmentation_uses_tail_residuals(self) -> None:
        base = {
            "trajectory_id": "hard",
            "trajectory_step": 5,
            "trajectory_len": 12,
            "trajectory_phase": "BIAS",
            "phase_name": "Reach",
            "planner_prior_delta": [0.01, -0.02, 0.0, 0.0, 0.0, 0.03],
            "planner_bias_score": 2.0e-4,
            "recovery_target_dx": -0.002,
            "recovery_target_dy": -0.018,
            "recovery_target_dyaw": 0.02,
            "recovery_needed": True,
        }
        replay_template = {
            "trajectory_id": "hard_replay",
            "trajectory_step": 10,
            "trajectory_phase": "REFINE",
            "planner_prior_delta": [0.02, -0.03, 0.0, 0.0, 0.0, 0.04],
            "recovery_target": [-0.002, -0.018, 0.02],
            "failure_residual": [0.02, -0.01, 0.03],
            "failure_drift": [0.005, -0.003, 0.002],
            "large_bias_threshold": 1.0e-4,
        }
        aug = augment_recovery_record_replay(base, replay_template=replay_template, replay_strength=1.5, replay_drift_strength=0.5)
        self.assertEqual(aug["augment_strategy"], "tail_failure_replay")
        self.assertGreater(float(aug["planner_bias_score"]), float(base["planner_bias_score"]))
        self.assertEqual(aug["replay_source_trajectory"], "hard_replay")
        self.assertIn("replay_source_residual", aug)
        lib = build_failure_replay_library([base, replay_template], selected_trajectories={"hard"}, tail_rows=4, large_bias_threshold=1e-4)
        self.assertTrue(lib)
        combined, report = build_failure_replay_augmented_records([base], shadow_report={"large_bias_threshold": 1e-4, "trajectory_summaries": [{"trajectory_id": "hard", "gain_mean": -0.01, "bias_score_max": 2e-4, "num_steps": 12}]}, hard_fraction=1.0, min_trajectories=1, tail_rows=4, replay_strengths=(1.25,), drift_strengths=(0.25,))
        self.assertGreater(len(combined), 1)
        self.assertEqual(report["augmentation_mode"], "tail_failure_replay")

    def test_tailbucket_replay_focuses_hard_buckets(self) -> None:
        records = [
            {
                "trajectory_id": "ep0_runtime_failure_tail",
                "trajectory_step": 12,
                "planner_prior_delta": [0.01, -0.01, 0.0, 0.0, 0.0, 0.02],
                "planner_bias_score": 0.004,
                "recovery_target_dx": -0.02,
                "recovery_target_dy": -0.01,
                "recovery_target_dyaw": 0.03,
                "episode_idx": 0,
                "step_idx": 12,
                "trajectory_phase": "runtime_failure_tail",
                "phase_name": "RECOVER",
            },
            {
                "trajectory_id": "ep1_runtime_failure_tail",
                "trajectory_step": 18,
                "planner_prior_delta": [0.08, -0.05, 0.0, 0.0, 0.0, 0.01],
                "planner_bias_score": 0.005,
                "recovery_target_dx": 0.07,
                "recovery_target_dy": -0.09,
                "recovery_target_dyaw": 0.02,
                "episode_idx": 1,
                "step_idx": 18,
                "trajectory_phase": "runtime_failure_tail",
                "phase_name": "RECOVER",
            },
        ]
        combined, report = build_bucket_tail_replay_augmented_records(
            records,
            focus_buckets=["small_xy_small_yaw", "large_xy_small_yaw"],
            trajectory_fraction=1.0,
            min_trajectories_per_bucket=1,
            tail_rows=2,
            min_tail_rows_per_bucket=1,
            replay_strengths_by_bucket={
                "small_xy_small_yaw": [1.05],
                "large_xy_small_yaw": [1.10],
            },
            drift_strengths_by_bucket={
                "small_xy_small_yaw": [0.05],
                "large_xy_small_yaw": [0.10],
            },
            replay_modes_by_bucket={
                "small_xy_small_yaw": ["oscillate"],
                "large_xy_small_yaw": ["overshoot"],
            },
            bucket_weight_by_bucket={
                "small_xy_small_yaw": 1.8,
                "large_xy_small_yaw": 1.4,
            },
        )
        self.assertGreater(len(combined), len(records))
        self.assertIn("selected_trajectories_by_bucket", report)
        self.assertTrue(report["selected_trajectories_by_bucket"])

    def test_planner_only_fallback_no_task_spec(self) -> None:
        sup = PrecisionSkillSupervisor(None, mode="planner_only", shadow_only=True)
        sup.reset()
        obs = _make_observation()
        obs["gripper_pose"] = np.array([0.0, 0.0, 0.35, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
        planner_delta = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        action = sup.step(planner_delta, obs, robot_state={}, task_spec=None, current_instruction="test")
        trace = sup.get_last_trace()
        np.testing.assert_allclose(action, planner_delta, atol=1e-6)
        self.assertEqual(trace["c2c_v2_stage"], "planner_only")
        self.assertFalse(trace["uses_privileged_target"])
        self.assertFalse(trace["uses_rlbench_mask_runtime"])
        expected_local = world_delta_to_local(planner_delta[:6], obs["gripper_pose"][3:7])
        np.testing.assert_allclose(np.asarray(trace["planner_chunk_local_6d"], dtype=np.float32), expected_local, atol=1e-6)
        np.testing.assert_allclose(np.asarray(trace["local_command_local_6d"], dtype=np.float32), expected_local, atol=1e-6)
        np.testing.assert_allclose(np.asarray(trace["local_residual_vs_planner_local_6d"], dtype=np.float32), np.zeros(6, dtype=np.float32), atol=1e-6)

    def test_precision_gate_prefers_active_skill_over_first_skill_type_match(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        sup = PrecisionSkillSupervisor(spec, mode="basin_recovery_only", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_CONTACT")
        obs = _make_observation(blue_center=(48, 48))
        obs["wrist_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        obs["front_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        planner_delta = np.array([0.001, -0.001, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        _ = sup.step(
            planner_delta,
            obs,
            robot_state={
                "wrist_valid_depth_ratio": 1.0,
                "wrist_depth_near_fraction": 1.0,
                "wrist_is_occluded": False,
                "wrist_is_low_visibility": False,
            },
            task_spec=spec,
            current_instruction="put the ring on the red spoke",
        )
        trace = sup.get_last_trace()
        self.assertEqual(trace["c2c_v2_stage"], "RING_GRASP_CONTACT")
        self.assertEqual(trace["c2c_gate_skill"], "precision_grasp")
        self.assertEqual(trace["c2c_gate_skill_name"], "grasp_contact_ring")

    def test_recovery_audit_helpers(self) -> None:
        xy, yaw_abs, dyaw, score = planner_bias_xyyaw([0.003, -0.004, 0.0, 0.0, 0.0, 0.02])
        self.assertGreater(xy, 0.0)
        self.assertGreater(yaw_abs, 0.0)
        self.assertAlmostEqual(dyaw, 0.02, places=6)
        self.assertGreater(score, 0.0)
        self.assertEqual(trace_episode_index(Path("ep019_gripper_trace.jsonl")), 19)
        self.assertEqual(recovery_phase_label(0, 12, 12), "BIAS")
        self.assertEqual(recovery_phase_label(6, 12, 12), "REFINE")
        self.assertEqual(recovery_phase_label(11, 12, 12), "RECOVER")

    def test_closed_loop_step_reduces_known_error(self) -> None:
        post_error, post_prior = apply_closed_loop_recovery_step(
            [0.03, -0.02, 0.08],
            [0.03, -0.02, 0.0, 0.0, 0.0, 0.08],
            [0.01, -0.01, 0.03],
        )
        self.assertLess(
            recovery_error_norm(float(post_error[0]), float(post_error[1]), float(post_error[2])),
            recovery_error_norm(0.03, -0.02, 0.08),
        )
        self.assertAlmostEqual(float(post_prior[0]), 0.02, places=6)
        self.assertAlmostEqual(float(post_prior[1]), -0.01, places=6)
        self.assertAlmostEqual(float(post_prior[5]), 0.05, places=6)

    def test_recovery_overshoot_flag_fires_on_crossing_with_larger_error(self) -> None:
        self.assertTrue(recovery_overshoot_flag([0.01, 0.0, 0.0], [-0.02, 0.0, 0.0]))
        self.assertFalse(recovery_overshoot_flag([0.02, 0.0, 0.0], [0.01, 0.0, 0.0]))

    def test_basin_thresholds_classify_synthetic_examples(self) -> None:
        self.assertTrue(in_near_grasp_basin(0.01, 0.002, 0.05))
        self.assertFalse(in_near_grasp_basin(0.02, 0.002, 0.05))
        self.assertTrue(in_close_ready_basin(0.003, 0.002, 0.02))
        self.assertFalse(in_close_ready_basin(0.006, 0.0, 0.02))
        self.assertEqual(classify_basin_label([0.003, 0.001, 0.02]).value, "close_ready")
        self.assertEqual(classify_basin_label([0.010, 0.001, 0.05]).value, "near_grasp")
        self.assertEqual(classify_basin_label([0.030, 0.001, 0.05]).value, "outside")

    def test_estimated_basin_error_close_ready_and_abstain(self) -> None:
        est = EstimatedBasinError(
            valid=True,
            confidence=0.8,
            dx=0.002,
            dy=-0.001,
            dz=0.004,
            dyaw=0.22,
            x_valid=True,
            y_valid=True,
            z_valid=True,
            yaw_valid=False,
            x_confidence=0.8,
            y_confidence=0.8,
            z_confidence=0.8,
            yaw_confidence=0.05,
            frame_consistency=0.9,
            source="unit_test",
            reason="synthetic",
            target_entity="ring_grasp_frame",
            reference_entity="gripper_jaw_frame",
            stage_name="RING_GRASP_ALIGN",
        )
        self.assertTrue(est.close_ready(xy_threshold=0.005, z_threshold=0.010, yaw_threshold=0.03, yaw_required=None))
        self.assertFalse(est.close_ready(xy_threshold=0.005, z_threshold=0.010, yaw_threshold=0.03, yaw_required=True))
        trace = est.to_trace()
        self.assertIn("estimated_basin_error_close_ready", trace)
        self.assertFalse(trace["estimated_basin_error_yaw_valid"])

    def test_calibrated_basin_estimator_abstains_yaw_by_default(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        skill = spec.get_skill("precision_grasp_ring")
        estimator = CalibratedGraspBasinEstimator(BasinStateCalibration())
        local_error = type(
            "LocalError",
            (),
            {
                "valid": True,
                "confidence": 0.8,
                "dx": 0.002,
                "dy": -0.001,
                "dz": 0.004,
                "dyaw": 0.22,
                "observability": 0.02,
                "fit_residual": 0.05,
                "inlier_ratio": 0.95,
                "reason": "ok",
                "target_entity": "ring_grasp_frame",
                "reference_entity": "gripper_jaw_frame",
                "stage_name": "RING_GRASP_ALIGN",
            },
        )()
        est = estimator.estimate(
            local_error,
            robot_state={"wrist_valid_depth_ratio": 0.2, "wrist_depth_near_fraction": 0.2, "wrist_is_occluded": False, "wrist_is_low_visibility": False},
            task_spec=spec,
            skill_spec=skill,
            stage_name="RING_GRASP_ALIGN",
        )
        self.assertTrue(est.x_valid)
        self.assertTrue(est.y_valid)
        self.assertTrue(est.z_valid)
        self.assertFalse(est.yaw_valid)
        self.assertIn(est.reason, {"calibrated_xy_z_supported", "calibrated_xy_supported", "calibrated_z_supported", "abstain_low_axis_validity"})

    def test_basin_state_calibration_report_loads_policies(self) -> None:
        report = load_basin_state_calibration_report(
            ROOT / "runtime_artifacts" / "coarse2contact_v2" / "reports" / "basin_state_calibration" / "basin_state_calibration.json"
        )
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.x.policy, "abstain")
        self.assertEqual(report.y.policy, "diagnostic_only")
        self.assertEqual(report.z.policy, "diagnostic_only")
        self.assertEqual(report.yaw.policy, "abstain")

    def test_gated_hybrid_prefers_v11_outside_hard_bucket(self) -> None:
        self.assertEqual(
            choose_gated_hybrid_candidate("large_xy_large_yaw", v11_post_error_norm=0.04, v16_post_error_norm=0.02),
            "v11_general",
        )
        self.assertEqual(
            choose_gated_hybrid_candidate("small_xy_small_yaw", v11_post_error_norm=0.04, v16_post_error_norm=0.02),
            "v16_specialist",
        )
        self.assertTrue(monotonic_decay_prefix([0.3, 0.2, 0.2, 0.1]))
        self.assertFalse(monotonic_decay_prefix([0.3, 0.2, 0.25]))

    def test_basin_recovery_prior_only_reacquires_without_xyyaw_correction(self) -> None:
        supervisor = BasinRecoverySupervisor()
        decision = supervisor.step(
            record={"frame_confidence": 0.0, "frame_observability": 0.0, "frame_axis_strength": 0.0},
            planner_prior_state=np.zeros(6, dtype=np.float32),
            visual_error_state=[0.02, -0.01, 0.1],
            target_error_state_for_eval=[0.02, -0.01, 0.1],
        )
        self.assertEqual(decision.mode, BasinRecoveryMode.REACQUIRE_VIEW)
        np.testing.assert_allclose(decision.correction_xyyaw, np.zeros(3, dtype=np.float32), atol=1e-8)
        self.assertLess(float(decision.local_action_6d[2]), 0.0)
        self.assertFalse(decision.uses_privileged_target)
        trace = decision.to_trace()
        self.assertIn("basin_recovery_micro_entry_ready", trace)
        self.assertIn("basin_recovery_micro_yaw_active", trace)
        self.assertFalse(trace["basin_recovery_micro_entry_ready"])

    def test_basin_recovery_visual_pullback_can_reduce_aligned_error(self) -> None:
        config = BasinRecoveryConfig(visual_conf_threshold=0.01, visual_observability_threshold=0.001, visual_axis_strength_threshold=1e-6)
        supervisor = BasinRecoverySupervisor(config=config)
        record = {"frame_confidence": 0.5, "frame_observability": 0.02, "frame_axis_strength": 0.1}
        self.assertEqual(classify_visual_evidence_for_basin(record, config=config), VisualEvidenceClass.VISUAL_OBSERVABLE)
        pre = np.array([0.02, -0.01, 0.1], dtype=np.float32)
        decision = supervisor.step(
            record=record,
            planner_prior_state=np.zeros(6, dtype=np.float32),
            visual_error_state=pre,
            target_error_state_for_eval=pre,
            allow_eval_line_search=True,
        )
        self.assertIn(decision.mode, {BasinRecoveryMode.VISUAL_PULLBACK, BasinRecoveryMode.MICRO_SERVO_TO_BASIN})
        post, _ = apply_closed_loop_recovery_step(pre, np.zeros(6, dtype=np.float32), decision.correction_xyyaw)
        self.assertLessEqual(
            recovery_error_norm(float(post[0]), float(post[1]), float(post[2])),
            recovery_error_norm(float(pre[0]), float(pre[1]), float(pre[2])),
        )
        self.assertTrue(decision.used_visual_geometry)
        self.assertFalse(decision.uses_privileged_target)

    def test_basin_recovery_xy_only_variant_zeros_yaw_correction(self) -> None:
        config = BasinRecoveryConfig(
            variant_name="xy_only",
            yaw_control_enabled=False,
            visual_conf_threshold=0.01,
            visual_observability_threshold=0.001,
            visual_axis_strength_threshold=1e-6,
        )
        supervisor = BasinRecoverySupervisor(config=config)
        decision = supervisor.step(
            record={"frame_confidence": 0.5, "frame_observability": 0.02, "frame_axis_strength": 0.1},
            planner_prior_state=np.zeros(6, dtype=np.float32),
            visual_error_state=np.array([0.02, -0.01, 0.2], dtype=np.float32),
            target_error_state_for_eval=np.array([0.02, -0.01, 0.2], dtype=np.float32),
            allow_eval_line_search=False,
        )
        self.assertEqual(decision.variant_name, "xy_only")
        self.assertAlmostEqual(float(decision.correction_xyyaw[2]), 0.0, places=7)
        self.assertAlmostEqual(float(decision.local_action_6d[5]), 0.0, places=7)

    def test_basin_recovery_config_overrides_gain_step_budget(self) -> None:
        config = BasinRecoveryConfig(
            variant_name="xy_only",
            yaw_control_enabled=False,
            visual_gain=0.6,
            max_pullback_xy_step=0.005,
            max_recovery_steps=36,
            visual_conf_threshold=0.01,
            visual_observability_threshold=0.001,
            visual_axis_strength_threshold=1e-6,
        )
        supervisor = BasinRecoverySupervisor(config=config)
        decision = supervisor.step(
            record={"frame_confidence": 0.5, "frame_observability": 0.02, "frame_axis_strength": 0.1},
            planner_prior_state=np.zeros(6, dtype=np.float32),
            visual_error_state=np.array([0.02, -0.01, 0.2], dtype=np.float32),
            target_error_state_for_eval=np.array([0.02, -0.01, 0.2], dtype=np.float32),
            allow_eval_line_search=False,
        )
        self.assertEqual(decision.variant_name, "xy_only")
        self.assertLessEqual(float(np.linalg.norm(decision.correction_xyyaw[:2])), 0.005 + 1e-7)
        self.assertEqual(supervisor.config.max_recovery_steps, 36)

    def test_basin_recovery_networks_expose_state_and_policy_outputs(self) -> None:
        features = torch.zeros((3, 24), dtype=torch.float32)
        state = BasinStateEstimatorNet(feature_dim=24)
        policy = BasinPullbackPolicyNet(feature_dim=24)
        state_out = state(features)
        policy_out = policy(features)
        self.assertEqual(tuple(state_out["visual_evidence_logits"].shape), (3, 3))
        self.assertEqual(tuple(state_out["basin_logits"].shape), (3, 3))
        self.assertEqual(tuple(policy_out["dx"].shape), (3,))
        self.assertIn("confidence", policy_out)

    def test_supervisor_basin_recovery_mode_traces_reacquire(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        sup = PrecisionSkillSupervisor(spec, mode="basin_recovery_only", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        obs = _make_observation(blue_center=(4, 4))
        planner_delta = np.zeros(7, dtype=np.float32)
        _ = sup.step(
            planner_delta,
            obs,
            robot_state={
                "invalid_action_flag": False,
                "wrist_valid_depth_ratio": 0.1,
                "wrist_depth_near_fraction": 0.0,
                "wrist_is_occluded": True,
                "wrist_is_low_visibility": True,
            },
            task_spec=spec,
            current_instruction="put the ring on the red spoke",
        )
        trace = sup.get_last_trace()
        self.assertIn(trace.get("basin_recovery_mode"), {"REACQUIRE_VIEW", "VISUAL_PULLBACK", "MICRO_SERVO_TO_BASIN"})
        self.assertFalse(trace["uses_privileged_target"])
        self.assertFalse(trace["c2c_gate_active"])

    def test_supervisor_emits_estimated_basin_trace_fields(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        sup = PrecisionSkillSupervisor(spec, mode="grasp_depth_apply", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        obs = _make_observation(blue_center=(51, 48))
        obs["wrist_depth"] = np.full((96, 96), 0.09, dtype=np.float32)
        obs["front_depth"] = np.full((96, 96), 0.09, dtype=np.float32)
        planner_delta = np.array([0.001, -0.001, 0.001, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        _ = sup.step(planner_delta, obs, robot_state={"wrist_valid_depth_ratio": 0.2, "wrist_depth_near_fraction": 0.2}, task_spec=spec, current_instruction="put the ring on the red spoke")
        trace = sup.get_last_trace()
        self.assertIn("estimated_basin_error", trace)
        self.assertIn("basin_axis_validity", trace)
        self.assertIn("basin_close_ready", trace)
        self.assertFalse(trace["uses_privileged_target"])
        self.assertFalse(trace["uses_rlbench_mask_runtime"])

    def test_calibration_report_blocks_untrusted_axis_apply(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        spec.runtime_flags["basin_state_calibration"] = {
            "x": {"policy": "abstain", "valid": False, "confidence": 0.0, "source": "unit_test", "reason": "block_x"},
            "y": {"policy": "abstain", "valid": False, "confidence": 0.0, "source": "unit_test", "reason": "block_y"},
            "z": {"policy": "abstain", "valid": False, "confidence": 0.0, "source": "unit_test", "reason": "block_z"},
            "yaw": {"policy": "abstain", "valid": False, "confidence": 0.0, "source": "unit_test", "reason": "block_yaw"},
            "xy_confidence_floor": 0.25,
            "z_confidence_floor": 0.25,
            "yaw_confidence_floor": 0.30,
            "frame_consistency_floor": 0.20,
        }
        sup = PrecisionSkillSupervisor(spec, mode="grasp_depth_apply", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        sup._precision_gate_active["precision_grasp"] = True
        sup._precision_gate_stable_frames["precision_grasp"] = 1
        obs = _make_observation(blue_center=(48, 48))
        obs["wrist_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        obs["front_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        planner_delta = np.array([0.01, -0.02, 0.01, 0.0, 0.0, 0.03, 1.0], dtype=np.float32)
        action = sup.step(planner_delta, obs, robot_state={}, task_spec=spec, current_instruction="put the ring on the red spoke")
        trace = sup.get_last_trace()
        np.testing.assert_allclose(action, planner_delta, atol=1e-6)
        self.assertEqual(trace["basin_axis_policy"]["x"], "abstain")
        self.assertEqual(trace["basin_axis_policy"]["yaw"], "abstain")

    def test_calibration_blocks_gate_without_trusted_axis(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        spec.runtime_flags["basin_state_calibration"] = {
            "x": {"policy": "abstain", "valid": False, "confidence": 0.0, "source": "unit_test", "reason": "block_x"},
            "y": {"policy": "diagnostic_only", "valid": False, "confidence": 0.0, "source": "unit_test", "reason": "diag_y"},
            "z": {"policy": "diagnostic_only", "valid": False, "confidence": 0.0, "source": "unit_test", "reason": "diag_z"},
            "yaw": {"policy": "abstain", "valid": False, "confidence": 0.0, "source": "unit_test", "reason": "block_yaw"},
            "xy_confidence_floor": 0.0,
            "z_confidence_floor": 0.0,
            "yaw_confidence_floor": 0.0,
            "frame_consistency_floor": 0.0,
        }
        sup = PrecisionSkillSupervisor(spec, mode="basin_recovery_only", shadow_only=False)
        sup.reset()
        sup._set_stage("RING_GRASP_ALIGN")
        obs = _make_observation(blue_center=(48, 48))
        obs["wrist_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        obs["front_depth"] = np.full((96, 96), 0.08, dtype=np.float32)
        planner_delta = np.array([0.01, -0.02, 0.01, 0.0, 0.0, 0.03, 1.0], dtype=np.float32)
        _ = sup.step(
            planner_delta,
            obs,
            robot_state={
                "wrist_valid_depth_ratio": 1.0,
                "wrist_depth_near_fraction": 1.0,
                "wrist_is_occluded": False,
                "wrist_is_low_visibility": False,
            },
            task_spec=spec,
            current_instruction="put the ring on the red spoke",
        )
        trace = sup.get_last_trace()
        self.assertFalse(bool(trace["c2c_gate_active"]))
        self.assertEqual(trace["c2c_gate_reason"], "no_trusted_control_axis")
        self.assertFalse(bool(trace["basin_close_ready"]))
        self.assertEqual(trace["phase_owner"], "planner")


if __name__ == "__main__":
    unittest.main()
