from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import yaml

from prismatic.robot.coarse2contact_v2 import (
    PrecisionObservationBundle,
    PrecisionSkillSupervisor,
    DepthGeometryLocalizerNet,
    LocalGeometryError,
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
from prismatic.robot.coarse2contact_v2.frame_yaw_estimator import FRAME_YAW_FEATURE_NAMES, FrameYawEstimatorNet, frame_yaw_feature_vector, frame_yaw_label_from_row, resolve_yaw_observable_threshold, save_frame_yaw_checkpoint
from prismatic.robot.coarse2contact_v2.takeover_contract import FrameResidual, TakeoverThresholds, classify_yaw_observability, decide_takeover_tier
from prismatic.robot.residual_safety import ResidualSafety
from prismatic.robot.residual_transforms import local_delta_to_world, world_delta_to_local
from scripts.audit_c2c_v2_frame_contract_relabel import audit as audit_frame_contract_relabel
from scripts.audit_c2c_v2_frame_contract_relabel import _apply_calibrated_yaw_observability
from scripts.audit_c2c_v2_grasp_failure_tail_intervention import audit as audit_grasp_failure_tail_intervention
from scripts.audit_c2c_v2_grasp_intervention import audit as audit_grasp_intervention
from scripts.audit_c2c_v2_yaw_threshold_sweep import sweep as audit_yaw_threshold_sweep
from scripts.build_c2c_v2_grasp_failure_tail_candidates import build_candidates
from scripts.build_c2c_v2_grasp_failure_tail_hard_bucket_gap_report import build_gap_report
from scripts.build_c2c_v2_failure_tail_balanced_manifest import build_balanced_manifest
from scripts.build_c2c_v2_failure_tail_hard_manifest import build_hard_manifest
from scripts.build_c2c_v2_failure_tail_hard_observability_supplement import build_hard_observability_supplement
from scripts.build_c2c_v2_failure_tail_support_manifest import merge_manifests
from scripts.build_c2c_v2_yaw_alias_drift_manifest import build_alias_drift_manifest
from scripts.build_c2c_v2_yaw_alias_drift_support_manifest import build_support_manifest
from scripts.build_c2c_v2_yaw_alias_drift_row_support_manifest import build_row_support_manifest
from scripts.build_c2c_v2_hard_window_support_supplement import build_hard_window_support_supplement
from scripts.build_c2c_v2_frame_yaw_dataset import build_dataset as build_frame_yaw_dataset
from scripts.diagnose_c2c_v2_grasp_failure_tail_direction import build_direction_diagnostic
from scripts.eval_c2c_v2_frame_yaw_estimator import evaluate as evaluate_frame_yaw_estimator
from scripts.diagnose_c2c_v2_yaw_frame_alignment import diagnose as diagnose_yaw_frame_alignment
from scripts.mine_c2c_v2_yaw_positive_windows import mine as mine_yaw_positive_windows
from scripts.run_c2c_v2_yaw_alias_drift_baseline import run_baseline as run_yaw_alias_drift_baseline
from scripts.run_c2c_v2_yaw_alias_drift_two_stage_baseline import _calibrate_threshold, run_two_stage_baseline
from scripts.run_c2c_v2_grasp_shell_episode_sweep import _summarize_sweep_reports
from scripts.relabel_c2c_v2_privileged_basin_frames import _frame_label_fields
from prismatic.robot.coarse2contact_v2.grasp_probe_shell import grasp_probe_shell_fields
from prismatic.robot.coarse2contact_v2.grasp_probe_shell import grasp_probe_inactive_reason


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
        self.assertIn("yaw_entry_feasible", row)
        self.assertIn("yaw_control_observable", row)
        self.assertIn("yaw_entry_block_reason", row)
        self.assertIn("yaw_control_block_reason", row)
        self.assertIn("micro_entry_ready", row)
        self.assertEqual(row["frame_contract"]["target_frame"], "ring_grasp_frame")
        self.assertEqual(row["frame_contract"]["reference_frame"], "gripper_jaw_frame")
        self.assertIn("local_delta_6d", row["planner_prior"])
        self.assertIn("local_correction_local_6d", row["action_t"])
        self.assertEqual(row["obs_t"]["visual_observability_class"], row["visual_observability_class"])
        self.assertEqual(row["schema_version"], "frame_residual_v2")
        self.assertIn(row["yaw_observability_class"], {"observable", "ambiguous", "unobservable"})
        self.assertIn("yaw_observability_blocker_combo", row)
        self.assertIn("yaw_observability_primary_blocker", row)
        self.assertIn("yaw_observability_gate_passes", row)
        self.assertIn(row["takeover_tier"], {"frontier_pullback_candidate", "outer_pullback_candidate", "coarse_pullback_candidate", "near_basin_shell", "micro_entry_ready", "close_ready", "outside_takeover", "abstain_prior_only", "invalid"})
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
            "window_protocol": {
                "window_mode": "stage",
                "shell_filter": "tight_near_yaw_feasible",
                "queue_flushed": True,
                "requested_horizon": 5,
            },
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
            "next_privileged_dx": 0.033,
            "next_privileged_dy": 0.0,
            "next_privileged_dz": 0.021,
            "next_privileged_dyaw": 0.0,
            "next_xy_error": 0.033,
            "true_basin_error_t_plus_1": {"dx": 0.033, "dy": 0.0, "dz": 0.021, "dyaw": 0.0},
        })
        report = audit_frame_contract_relabel([base, blocked])
        self.assertEqual(report["overall"]["schema_version_counts"]["frame_residual_v2"], 2)
        self.assertEqual(report["overall"]["xy_contracted_count"], 1)
        self.assertIn("by_episode", report)
        self.assertIn("by_window_protocol", report)
        self.assertEqual(report["overall"]["coarse_pullback_candidate_rows"], 1)
        self.assertEqual(report["overall"]["yaw_observability_counts"]["observable"], 1)
        self.assertEqual(report["overall"]["yaw_observability_counts"]["unobservable"], 1)
        self.assertIn("near_basin_shell_yaw_observable_rate", report["overall"])
        tiers = {item["takeover_tier"]: item for item in report["by_takeover_tier"]}
        self.assertIn("coarse_pullback_candidate", tiers)
        self.assertIn("abstain_prior_only", tiers)
        self.assertGreaterEqual(tiers["coarse_pullback_candidate"]["xy_contraction_lower_ci"], 0.0)

    def test_frame_contract_audit_exposes_yaw_alignment_blockers(self) -> None:
        rows = [
            {
                "episode_idx": 4,
                "step_idx": 10,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "yaw_observable": False,
                "xy_error": 0.024,
                "next_xy_error": 0.020,
                "xy_contracted": True,
                "yaw_observability_blocker_combo": "frame_observability_lt_010",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
            },
            {
                "episode_idx": 4,
                "step_idx": 11,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "unobservable",
                "yaw_observable": False,
                "xy_error": 0.031,
                "next_xy_error": 0.032,
                "xy_contracted": False,
                "yaw_observability_blocker_combo": "frame_confidence_lt_050+frame_observability_lt_010",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
            },
            {
                "episode_idx": 4,
                "step_idx": 12,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "visual_observability_class": "partial_observable",
                "yaw_observability_class": "observable",
                "yaw_observable": True,
                "xy_error": 0.014,
                "next_xy_error": 0.013,
                "xy_contracted": True,
            },
        ]
        report = audit_frame_contract_relabel(rows)
        self.assertEqual(report["yaw_alignment"]["visual_observable_yaw_blocked_rows"], 2)
        self.assertEqual(report["yaw_alignment"]["visual_observable_xy_contracted_yaw_blocked_rows"], 1)
        self.assertEqual(report["overall"]["yaw_blocker_combo_counts"]["frame_observability_lt_010"], 1)
        self.assertEqual(report["yaw_alignment"]["blocker_term_counts"]["frame_observability_lt_010"], 2)
        self.assertEqual(report["overall"]["yaw_primary_blocker_counts"]["frame_observability_lt_010"], 2)
        self.assertEqual(report["yaw_alignment"]["visual_observable_xy_contracted_blocker_combo_counts"]["frame_observability_lt_010"], 1)
        crosstab = {item["visual_observability_class"]: item for item in report["yaw_alignment"]["visual_yaw_crosstab"]}
        self.assertEqual(crosstab["visual_observable"]["yaw_blocked_rows"], 2)
        self.assertEqual(crosstab["visual_observable"]["xy_contracted_yaw_blocked_rows"], 1)
        self.assertIn("frame_confidence_lt_050+frame_observability_lt_010", crosstab["visual_observable"]["blocker_combo_counts"])

    def test_takeover_contract_prior_only_abstains(self) -> None:
        residual = FrameResidual(dx=0.004, dy=0.0, dz=0.002, dyaw=0.01)
        obs = classify_yaw_observability({}, {}, visual_observability_class="prior_only")
        decision = decide_takeover_tier(
            residual,
            obs,
            precision_row=True,
            requires_yaw_observability=True,
            xy_contracted=True,
            thresholds=TakeoverThresholds(),
        )
        self.assertEqual(decision.takeover_tier, "abstain_prior_only")
        self.assertFalse(decision.pullback_allowed)
        self.assertEqual(decision.axis_gate_policy["x"], "abstain")
        self.assertEqual(decision.axis_gate_policy["yaw"], "abstain")

    def test_takeover_contract_yaw_observable_controls_yaw_axis_only(self) -> None:
        residual = FrameResidual(dx=0.010, dy=0.0, dz=0.004, dyaw=0.03)
        obs = classify_yaw_observability(
            {},
            {"frame_confidence": 0.9, "frame_observability": 0.2, "frame_axis_strength": 0.95},
            visual_observability_class="visual_observable",
        )
        decision = decide_takeover_tier(
            residual,
            obs,
            precision_row=True,
            requires_yaw_observability=True,
            xy_contracted=True,
            thresholds=TakeoverThresholds(),
        )
        self.assertEqual(obs.yaw_observability_class, "observable")
        self.assertEqual(decision.axis_gate_policy["x"], "trusted_control")
        self.assertEqual(decision.axis_gate_policy["y"], "trusted_control")
        self.assertEqual(decision.axis_gate_policy["z"], "diagnostic_only")
        self.assertEqual(decision.axis_gate_policy["yaw"], "trusted_control")
        self.assertIn(decision.takeover_tier, {"near_basin_shell", "micro_entry_ready"})

    def test_takeover_contract_splits_yaw_entry_from_control_observability(self) -> None:
        residual = FrameResidual(dx=0.020, dy=0.0, dz=0.020, dyaw=0.03)
        blocked_obs = classify_yaw_observability(
            {},
            {"frame_confidence": 0.30, "frame_observability": 0.02, "frame_axis_strength": 0.50},
            visual_observability_class="visual_observable",
        )
        blocked = decide_takeover_tier(
            residual,
            blocked_obs,
            precision_row=True,
            requires_yaw_observability=True,
            xy_contracted=True,
            thresholds=TakeoverThresholds(),
        )
        self.assertFalse(blocked_obs.yaw_observable)
        self.assertTrue(blocked.yaw_entry_feasible)
        self.assertFalse(blocked.yaw_control_observable)
        self.assertEqual(blocked.takeover_tier, "near_basin_shell")
        self.assertTrue(blocked.near_basin_shell)
        self.assertEqual(blocked.axis_gate_policy["yaw"], "abstain")

        ready_obs = classify_yaw_observability(
            {},
            {"frame_confidence": 0.95, "frame_observability": 0.25, "frame_axis_strength": 0.97},
            visual_observability_class="visual_observable",
        )
        ready = decide_takeover_tier(
            residual,
            ready_obs,
            precision_row=True,
            requires_yaw_observability=True,
            xy_contracted=True,
            thresholds=TakeoverThresholds(),
        )
        self.assertTrue(ready_obs.yaw_observable)
        self.assertTrue(ready.yaw_entry_feasible)
        self.assertTrue(ready.yaw_control_observable)
        self.assertEqual(ready.takeover_tier, "near_basin_shell")
        self.assertTrue(ready.near_basin_shell)

    def test_takeover_contract_coarse_candidate_requires_contraction(self) -> None:
        residual = FrameResidual(dx=0.040, dy=0.0, dz=0.020, dyaw=0.04)
        obs = classify_yaw_observability(
            {},
            {"frame_confidence": 0.9, "frame_observability": 0.2, "frame_axis_strength": 0.95},
            visual_observability_class="visual_observable",
        )
        no_contraction = decide_takeover_tier(
            residual,
            obs,
            precision_row=True,
            requires_yaw_observability=True,
            xy_contracted=False,
            thresholds=TakeoverThresholds(),
        )
        contraction = decide_takeover_tier(
            residual,
            obs,
            precision_row=True,
            requires_yaw_observability=True,
            xy_contracted=True,
            thresholds=TakeoverThresholds(),
        )
        self.assertEqual(no_contraction.takeover_tier, "outside_takeover")
        self.assertEqual(contraction.takeover_tier, "coarse_pullback_candidate")

    def test_takeover_contract_outer_frontier_expands_support_surface(self) -> None:
        residual = FrameResidual(dx=0.100, dy=0.000, dz=0.020, dyaw=0.16)
        obs = classify_yaw_observability(
            {},
            {"frame_confidence": 0.6, "frame_observability": 0.02, "frame_axis_strength": 0.9},
            visual_observability_class="visual_observable",
        )
        decision = decide_takeover_tier(
            residual,
            obs,
            precision_row=True,
            requires_yaw_observability=True,
            xy_contracted=True,
            thresholds=TakeoverThresholds(),
        )
        self.assertEqual(decision.takeover_tier, "outer_pullback_candidate")
        self.assertTrue(decision.outer_pullback_candidate)
        self.assertTrue(decision.pullback_allowed)
        self.assertEqual(decision.axis_gate_policy["x"], "trusted_control")
        self.assertEqual(decision.axis_gate_policy["yaw"], "abstain")

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
        self.assertTrue(est.pullback_ready(xy_threshold=0.020, min_frame_consistency=0.20))
        self.assertEqual(est.pullback_block_reason(xy_threshold=0.020, min_frame_consistency=0.20), "ready")
        self.assertFalse(est.close_ready(xy_threshold=0.005, z_threshold=0.010, yaw_threshold=0.03, yaw_required=True))

    def test_prior_only_estimate_never_becomes_pullback_ready(self) -> None:
        est = EstimatedBasinError(
            valid=False,
            confidence=0.0,
            dx=0.002,
            dy=0.001,
            dz=0.004,
            dyaw=0.0,
            x_valid=False,
            y_valid=False,
            z_valid=False,
            yaw_valid=False,
            x_confidence=0.0,
            y_confidence=0.0,
            z_confidence=0.0,
            yaw_confidence=0.0,
            frame_consistency=0.0,
            source="unit_test",
            reason="prior_only_reacquire",
        )
        self.assertFalse(est.pullback_ready(xy_threshold=0.020, min_frame_consistency=0.20))
        self.assertIn("prior_only", est.pullback_block_reason(xy_threshold=0.020, min_frame_consistency=0.20))

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

    def test_supervisor_pullback_gate_allows_xy_without_z_close_ready(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        spec.runtime_flags["precision_takeover_stable_frames"] = 1
        spec.runtime_flags["precision_grasp_takeover_error_xy_max"] = 0.030
        spec.runtime_flags["precision_grasp_takeover_confidence_min"] = 0.0
        stage = spec.get_stage("RING_GRASP_ALIGN")
        skill = spec.skills["precision_grasp_ring"]
        sup = PrecisionSkillSupervisor(spec, mode="basin_recovery_only", shadow_only=False)
        est = EstimatedBasinError(
            valid=True,
            confidence=0.90,
            dx=0.012,
            dy=-0.010,
            dz=0.040,
            dyaw=0.12,
            x_valid=True,
            y_valid=True,
            z_valid=False,
            yaw_valid=False,
            x_confidence=0.9,
            y_confidence=0.9,
            z_confidence=0.0,
            yaw_confidence=0.0,
            frame_consistency=0.90,
            source="unit_test",
            reason="xy_pullback_only",
        )
        zero_error = LocalGeometryError(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, "unused")
        active, info = sup._precision_gate_status(
            stage=stage,
            active_skill=skill,
            local_base=np.zeros(6, dtype=np.float32),
            grasp_error=zero_error,
            spoke_error=zero_error,
            estimated_basin_error=est,
        )
        self.assertTrue(active)
        self.assertTrue(info["pullback_gate_ready"])
        self.assertFalse(info["micro_entry_ready"])
        self.assertFalse(info["close_ready"])
        self.assertEqual(info["pullback_block_reason"], "ready")
        self.assertIn("z", info["close_block_reason"])

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

    def test_grasp_intervention_audit_counts_outer_pullback_candidates(self) -> None:
        rows = [
            {
                "episode_idx": 10,
                "step": 45,
                "c2c_v2_stage": "RING_GRASP_ALIGN",
                "grasp_probe_policy": "replay_oracle_xy",
                "grasp_probe_active": False,
                "grasp_probe_reason": "inactive",
                "grasp_probe_visibility_bucket": "visual_observable",
                "grasp_probe_near_basin_shell": False,
                "grasp_probe_coarse_pullback_candidate": False,
                "grasp_probe_outer_pullback_candidate": True,
                "grasp_probe_horizon_xy_feasible": False,
                "grasp_probe_yaw_feasible": False,
                "grasp_probe_pre_true_error_t": [0.100, 0.0, 0.0, 0.150],
                "grasp_probe_post_true_error_t": [0.098, 0.0, 0.0, 0.150],
            }
        ]
        report = audit_grasp_intervention(
            rows,
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        self.assertEqual(report["overall"]["outer_pullback_candidate_rows"], 1)

    def test_failure_tail_candidate_builder_excludes_success_window(self) -> None:
        base = {
            "task_name": "insert_onto_square_peg",
            "episode_idx": 12,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_name": "precision_grasp_ring",
            "skill_type": "precision_grasp",
            "label_valid": True,
            "visual_observability_class": "visual_observable",
            "yaw_observability_class": "observable",
            "yaw_observable": True,
            "failure_bucket": "xy_bias",
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "true_basin_error_t": {"dx": 0.04, "dy": 0.0, "dz": 0.02, "dyaw": 0.03},
            "true_basin_error_t_plus_1": {"dx": 0.039, "dy": 0.0, "dz": 0.02, "dyaw": 0.03},
            "xy_error": 0.04,
            "yaw_abs": 0.03,
            "next_xy_error": 0.039,
            "next_yaw_abs": 0.03,
            "xy_contracted": True,
            "overshoot": False,
            "near_basin_shell": False,
            "takeover_tier": "outside_takeover",
        }
        success = dict(base, step_idx=1, xy_error=0.004, yaw_abs=0.01, next_xy_error=0.003, next_yaw_abs=0.01, takeover_tier="close_ready")
        candidate = dict(base, step_idx=2)
        rows = build_candidates([success, candidate], include_success_controls=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step_idx"], 2)
        self.assertEqual(rows[0]["sample_role"], "failure_tail_candidate")
        self.assertEqual(rows[0]["takeover_tier"], "coarse_pullback_candidate")
        self.assertEqual(rows[0]["recommended_intervention_axes"], ["x", "y"])

    def test_failure_tail_candidate_builder_abstains_prior_and_splits_yaw_entry_control(self) -> None:
        base = {
            "task_name": "insert_onto_square_peg",
            "episode_idx": 13,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_name": "precision_grasp_ring",
            "skill_type": "precision_grasp",
            "label_valid": True,
            "failure_bucket": "yaw_bias",
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "true_basin_error_t": {"dx": 0.035, "dy": 0.0, "dz": 0.02, "dyaw": 0.03},
            "true_basin_error_t_plus_1": {"dx": 0.036, "dy": 0.0, "dz": 0.02, "dyaw": 0.03},
            "xy_error": 0.035,
            "yaw_abs": 0.03,
            "next_xy_error": 0.036,
            "next_yaw_abs": 0.03,
            "xy_contracted": False,
            "near_basin_shell": False,
        }
        prior = dict(base, step_idx=3, visual_observability_class="prior_only", yaw_observability_class="unobservable", yaw_observable=False)
        control_blocked_entry_ok = dict(
            base,
            step_idx=4,
            visual_observability_class="visual_observable",
            yaw_observability_class="ambiguous",
            yaw_observable=False,
            yaw_control_observable=False,
            yaw_entry_feasible=True,
        )
        entry_blocked = dict(
            base,
            step_idx=5,
            visual_observability_class="visual_observable",
            yaw_observability_class="ambiguous",
            yaw_observable=False,
            yaw_control_observable=False,
            yaw_entry_feasible=False,
            yaw_abs=0.12,
        )
        rows = build_candidates([prior, control_blocked_entry_ok, entry_blocked], include_success_controls=False)
        by_step = {row["step_idx"]: row for row in rows}
        self.assertEqual(by_step[3]["takeover_tier"], "abstain_prior_only")
        self.assertEqual(by_step[3]["abstain_reason"], "prior_only")
        self.assertEqual(by_step[4]["takeover_tier"], "coarse_pullback_candidate")
        self.assertEqual(by_step[4]["recommended_intervention_axes"], ["x", "y"])
        self.assertFalse(by_step[4]["yaw_control_observable"])
        self.assertTrue(by_step[4]["yaw_entry_feasible"])
        self.assertEqual(by_step[5]["takeover_tier"], "yaw_entry_blocked")
        self.assertEqual(by_step[5]["recommended_intervention_axes"], [])

    def test_failure_tail_candidate_builder_promotes_small_xy_large_yaw_yaw_blocked_into_outer_support(self) -> None:
        row = {
            "task_name": "insert_onto_square_peg",
            "episode_idx": 14,
            "step_idx": 6,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_name": "precision_grasp_ring",
            "skill_type": "precision_grasp",
            "label_valid": True,
            "failure_bucket": "small_xy_large_yaw",
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "true_basin_error_t": {"dx": 0.045, "dy": 0.0, "dz": 0.02, "dyaw": 0.17},
            "true_basin_error_t_plus_1": {"dx": 0.044, "dy": 0.0, "dz": 0.02, "dyaw": 0.17},
            "xy_error": 0.045,
            "yaw_abs": 0.17,
            "next_xy_error": 0.044,
            "next_yaw_abs": 0.17,
            "xy_contracted": False,
            "near_basin_shell": False,
            "visual_observability_class": "visual_observable",
            "yaw_observability_class": "unobservable",
            "yaw_observable": False,
        }
        rows = build_candidates([row], include_success_controls=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["takeover_tier"], "outer_pullback_candidate")
        self.assertEqual(rows[0]["recommended_intervention_axes"], ["x", "y"])
        self.assertEqual(rows[0]["abstain_reason"], "")
        self.assertFalse(rows[0]["yaw_entry_feasible"])

    def test_failure_tail_candidate_builder_annotates_alias_drift_decision_from_support_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            support_path = tmpdir / "yaw_alias_support.jsonl"
            support_rows = [
                {
                    "episode_idx": 14,
                    "step_idx": 6,
                    "acceptance_role": "calibration_positive",
                    "alias_label": "stable_alias",
                    "alias_drift_decision": "stable_alias_control",
                    "selected_step_idxs": [6],
                }
            ]
            with open(support_path, "w", encoding="utf-8") as handle:
                for row in support_rows:
                    handle.write(json.dumps(row) + "\n")

            rows = build_candidates(
                [
                    {
                        "task_name": "insert_onto_square_peg",
                        "episode_idx": 14,
                        "step_idx": 6,
                        "stage_name": "RING_GRASP_ALIGN",
                        "skill_name": "precision_grasp_ring",
                        "skill_type": "precision_grasp",
                        "label_valid": True,
                        "failure_bucket": "small_xy_large_yaw",
                        "planner_prior": {"local_delta_6d": [0.0] * 6},
                        "true_basin_error_t": {"dx": 0.045, "dy": 0.0, "dz": 0.02, "dyaw": 0.17},
                        "true_basin_error_t_plus_1": {"dx": 0.044, "dy": 0.0, "dz": 0.02, "dyaw": 0.17},
                        "xy_error": 0.045,
                        "yaw_abs": 0.17,
                        "next_xy_error": 0.044,
                        "next_yaw_abs": 0.17,
                        "xy_contracted": False,
                        "near_basin_shell": False,
                        "visual_observability_class": "visual_observable",
                        "yaw_observability_class": "unobservable",
                        "yaw_observable": False,
                    }
                ],
                include_success_controls=False,
                alias_drift_support_jsonl=[support_path],
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["alias_drift_decision"], "stable_alias_control")
            self.assertEqual(rows[0]["yaw_alias_drift_decision"], "stable_alias_control")
            self.assertEqual(rows[0]["alias_drift_support_source"], str(support_path.resolve()))

    def test_failure_tail_candidate_builder_falls_back_to_episode_level_alias_support(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            support_path = tmpdir / "yaw_alias_support.jsonl"
            support_rows = [
                {
                    "episode_idx": 14,
                    "step_idx": 6,
                    "acceptance_role": "calibration_positive",
                    "alias_label": "stable_alias",
                    "alias_drift_decision": "stable_alias_control",
                    "selected_step_idxs": [6],
                }
            ]
            with open(support_path, "w", encoding="utf-8") as handle:
                for row in support_rows:
                    handle.write(json.dumps(row) + "\n")

            rows = build_candidates(
                [
                    {
                        "task_name": "insert_onto_square_peg",
                        "episode_idx": 14,
                        "step_idx": 11,
                        "stage_name": "RING_GRASP_ALIGN",
                        "skill_name": "precision_grasp_ring",
                        "skill_type": "precision_grasp",
                        "label_valid": True,
                        "failure_bucket": "small_xy_large_yaw",
                        "planner_prior": {"local_delta_6d": [0.0] * 6},
                        "true_basin_error_t": {"dx": 0.045, "dy": 0.0, "dz": 0.02, "dyaw": 0.17},
                        "true_basin_error_t_plus_1": {"dx": 0.044, "dy": 0.0, "dz": 0.02, "dyaw": 0.17},
                        "xy_error": 0.045,
                        "yaw_abs": 0.17,
                        "next_xy_error": 0.044,
                        "next_yaw_abs": 0.17,
                        "xy_contracted": False,
                        "near_basin_shell": False,
                        "visual_observability_class": "visual_observable",
                        "yaw_observability_class": "unobservable",
                        "yaw_observable": False,
                    }
                ],
                include_success_controls=False,
                alias_drift_support_jsonl=[support_path],
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["alias_drift_decision"], "stable_alias_control")
            self.assertEqual(rows[0]["yaw_alias_drift_decision"], "stable_alias_control")
            self.assertEqual(rows[0]["alias_drift_support_source"], str(support_path.resolve()))

    def test_failure_tail_candidate_builder_emits_outer_pullback_frontier(self) -> None:
        base = {
            "task_name": "insert_onto_square_peg",
            "episode_idx": 16,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_name": "precision_grasp_ring",
            "skill_type": "precision_grasp",
            "label_valid": True,
            "failure_bucket": "large_xy_large_yaw",
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "true_basin_error_t": {"dx": 0.10, "dy": 0.0, "dz": 0.02, "dyaw": 0.15},
            "true_basin_error_t_plus_1": {"dx": 0.099, "dy": 0.0, "dz": 0.02, "dyaw": 0.15},
            "xy_error": 0.10,
            "yaw_abs": 0.15,
            "next_xy_error": 0.099,
            "next_yaw_abs": 0.15,
            "xy_contracted": True,
            "near_basin_shell": False,
        }
        row = dict(
            base,
            step_idx=7,
            visual_observability_class="visual_observable",
            yaw_observability_class="unobservable",
            yaw_observable=False,
        )
        rows = build_candidates([row], include_success_controls=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["takeover_tier"], "outer_pullback_candidate")
        self.assertEqual(rows[0]["recommended_intervention_axes"], ["x", "y"])
        self.assertFalse(rows[0]["yaw_entry_feasible"])
        self.assertFalse(rows[0]["coarse_pullback_candidate"])

    def test_hard_window_support_supplement_promotes_large_xy_large_yaw_frontier_rows(self) -> None:
        rows = [
            {
                "episode_idx": 7,
                "step_idx": 1,
                "failure_bucket": "large_xy_large_yaw",
                "takeover_tier": "yaw_entry_blocked",
                "xy_error": 0.16,
            },
        ]
        selected, summary = build_hard_window_support_supplement(
            rows,
            episode_windows=[(5, 7)],
            outer_xy_threshold=0.12,
            frontier_xy_threshold=0.18,
            max_rows_per_episode=10,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["frontier_support_rows"], 1)
        self.assertEqual(selected[0]["takeover_tier"], "frontier_pullback_candidate")
        self.assertEqual(selected[0]["selection_reason"], "hard_window_frontier_support")
        self.assertEqual(selected[0]["support_mode"], "frontier")

    def test_direction_diagnostic_tracks_step_size_vs_flip(self) -> None:
        rows = [
            {
                "episode_idx": 0,
                "step_idx": 10,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_large_yaw",
                "takeover_tier": "outer_pullback_candidate",
                "alias_drift_decision": "frame_drift_abstain",
                "intervention_active": True,
                "privileged_dx": 0.04,
                "privileged_dy": 0.03,
                "grasp_probe_applied_xy_step_local_6d": [0.005, 0.004, 0.0, 0.0, 0.0, 0.0],
                "grasp_probe_pre_true_error_t": [0.04, 0.03, 0.0, 0.12],
                "grasp_probe_horizon_final_true_error_t": [0.038, 0.028, 0.0, 0.12],
                "grasp_probe_pre_xy_error": 0.05,
                "grasp_probe_horizon_final_xy_error": 0.047,
                "grasp_probe_horizon_overshoot": False,
                "xy_error": 0.05,
                "next_xy_error": 0.047,
            },
            {
                "episode_idx": 0,
                "step_idx": 11,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_large_yaw",
                "takeover_tier": "outer_pullback_candidate",
                "alias_drift_decision": "frame_drift_abstain",
                "intervention_active": True,
                "privileged_dx": 0.04,
                "privileged_dy": 0.03,
                "grasp_probe_applied_xy_step_local_6d": [-0.04, -0.03, 0.0, 0.0, 0.0, 0.0],
                "grasp_probe_pre_true_error_t": [0.04, 0.03, 0.0, 0.12],
                "grasp_probe_horizon_final_true_error_t": [0.045, 0.034, 0.0, 0.12],
                "grasp_probe_pre_xy_error": 0.05,
                "grasp_probe_horizon_final_xy_error": 0.056,
                "grasp_probe_horizon_overshoot": True,
                "xy_error": 0.05,
                "next_xy_error": 0.056,
            },
        ]
        report = build_direction_diagnostic(rows, failure_bucket="small_xy_large_yaw", episodes={0}, active_only=True)
        self.assertEqual(report["overall"]["active_rows"], 2)
        self.assertEqual(report["rows"][0]["direction_hint"], "step_too_small_candidate")
        self.assertEqual(report["rows"][1]["direction_hint"], "direction_flip_candidate")
        self.assertIn("frame_drift_abstain", report["overall"]["by_alias_drift_decision"])

    def test_failure_tail_balanced_manifest_spreads_coverage_beyond_high_recover_episode(self) -> None:
        rows = []
        for step in range(10):
            rows.append(
                {
                    "sample_role": "failure_tail_candidate",
                    "task_name": "insert_onto_square_peg",
                    "episode_idx": 23,
                    "step_idx": step,
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_type": "precision_grasp",
                    "failure_bucket": "large_xy_large_yaw",
                    "visual_observability_class": "visual_observable",
                    "yaw_observability_class": "ambiguous",
                    "takeover_tier": "coarse_pullback_candidate",
                }
            )
        rows.extend(
            [
                {
                    "sample_role": "failure_tail_candidate",
                    "task_name": "insert_onto_square_peg",
                    "episode_idx": 5,
                    "step_idx": 0,
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_type": "precision_grasp",
                    "failure_bucket": "small_xy_small_yaw",
                    "visual_observability_class": "partial_observable",
                    "yaw_observability_class": "unobservable",
                    "takeover_tier": "near_basin_shell",
                },
                {
                    "sample_role": "failure_tail_candidate",
                    "task_name": "insert_onto_square_peg",
                    "episode_idx": 26,
                    "step_idx": 1,
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_type": "precision_grasp",
                    "failure_bucket": "large_xy_small_yaw",
                    "visual_observability_class": "visual_observable",
                    "yaw_observability_class": "ambiguous",
                    "takeover_tier": "micro_entry_ready",
                },
            ]
        )
        selected, summary = build_balanced_manifest(rows, max_rows_per_episode=3, max_rows_per_coverage_key=2)
        self.assertLess(len(selected), len(rows))
        self.assertIn("ep023", summary["by_episode"])
        self.assertIn("ep005", summary["by_episode"])
        self.assertIn("ep026", summary["by_episode"])
        self.assertGreaterEqual(len(summary["by_failure_bucket"]), 3)
        self.assertGreaterEqual(len(summary["by_visual_observability"]), 2)
        self.assertGreaterEqual(len(summary["by_yaw_observability"]), 2)
        self.assertGreaterEqual(len(summary["by_takeover_tier"]), 3)
        self.assertTrue(all("coverage_key" in row for row in selected))
        self.assertTrue(all(row["selection_reason"] == "balanced_failure_tail_coverage" for row in selected))

    def test_failure_tail_hard_manifest_prioritizes_hard_buckets_and_observability(self) -> None:
        hard_rows = []
        easy_rows = []
        for step in range(4):
            hard_rows.append(
                {
                    "sample_role": "failure_tail_candidate",
                    "task_name": "insert_onto_square_peg",
                    "episode_idx": 14,
                    "step_idx": step,
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_type": "precision_grasp",
                    "failure_bucket": "large_xy_large_yaw",
                    "visual_observability_class": "partial_observable",
                    "yaw_observability_class": "unobservable",
                    "takeover_tier": "yaw_entry_blocked",
                    "planner_natural_outcome": "natural_diverges_or_stalls",
                }
            )
            easy_rows.append(
                {
                    "sample_role": "failure_tail_candidate",
                    "task_name": "insert_onto_square_peg",
                    "episode_idx": 5,
                    "step_idx": step,
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_type": "precision_grasp",
                    "failure_bucket": "small_xy_small_yaw",
                    "visual_observability_class": "visual_observable",
                    "yaw_observability_class": "observable",
                    "takeover_tier": "coarse_pullback_candidate",
                    "planner_natural_outcome": "natural_contracts",
                }
            )
        selected, summary = build_hard_manifest(
            hard_rows + easy_rows,
            max_rows_per_episode=10,
            easy_rows_per_coverage_key=1,
            hard_rows_per_coverage_key=3,
            hard_coverage_threshold=4,
            max_rows_per_coverage_key=3,
        )
        hard_selected = [row for row in selected if row["failure_bucket"] == "large_xy_large_yaw"]
        easy_selected = [row for row in selected if row["failure_bucket"] == "small_xy_small_yaw"]
        self.assertEqual(len(hard_selected), 3)
        self.assertEqual(len(easy_selected), 1)
        self.assertTrue(all(row["selection_reason"] == "hard_coverage_prioritized" for row in selected))
        self.assertGreater(int(summary["selected_hard_rows"]), int(summary["selected_easy_rows"]))
        self.assertEqual(summary["by_failure_bucket"]["large_xy_large_yaw"], 3)
        self.assertEqual(summary["by_failure_bucket"]["small_xy_small_yaw"], 1)

    def test_failure_tail_hard_manifest_reports_hard_support_rows(self) -> None:
        rows = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 14,
                "step_idx": 0,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "takeover_tier": "coarse_pullback_candidate",
            },
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 14,
                "step_idx": 1,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "unobservable",
                "takeover_tier": "yaw_entry_blocked",
            },
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 15,
                "step_idx": 2,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_large_yaw",
                "visual_observability_class": "partial_observable",
                "yaw_observability_class": "observable",
                "takeover_tier": "near_basin_shell",
            },
        ]
        selected, summary = build_hard_manifest(
            rows,
            max_rows_per_episode=10,
            easy_rows_per_coverage_key=1,
            hard_rows_per_coverage_key=2,
            hard_coverage_threshold=1,
            max_rows_per_coverage_key=2,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(summary["selected_hard_support_rows"], 2)
        self.assertEqual(summary["by_hard_support_failure_bucket"]["large_xy_large_yaw"], 1)
        self.assertEqual(summary["by_hard_support_failure_bucket"]["small_xy_large_yaw"], 1)
        self.assertEqual(summary["by_hard_support_takeover_tier"]["coarse_pullback_candidate"], 1)
        self.assertEqual(summary["by_hard_support_takeover_tier"]["near_basin_shell"], 1)

    def test_failure_tail_hard_manifest_includes_outer_pullback_support_rows(self) -> None:
        rows = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 18,
                "step_idx": 0,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_large_yaw",
                "visual_observability_class": "partial_observable",
                "yaw_observability_class": "unobservable",
                "takeover_tier": "outer_pullback_candidate",
            }
        ]
        selected, summary = build_hard_manifest(
            rows,
            max_rows_per_episode=10,
            easy_rows_per_coverage_key=1,
            hard_rows_per_coverage_key=2,
            hard_coverage_threshold=1,
            max_rows_per_coverage_key=2,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["selected_hard_support_rows"], 1)
        self.assertEqual(summary["by_hard_support_failure_bucket"]["small_xy_large_yaw"], 1)
        self.assertEqual(summary["by_hard_support_takeover_tier"]["outer_pullback_candidate"], 1)

    def test_failure_tail_support_manifest_merges_core_and_hard_rows_without_duplicates(self) -> None:
        core = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 14,
                "step_idx": 3,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "takeover_tier": "near_basin_shell",
            }
        ]
        supplement = [
            dict(core[0]),
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 23,
                "step_idx": 9,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "partial_observable",
                "yaw_observability_class": "unobservable",
                "takeover_tier": "yaw_entry_blocked",
            },
        ]
        merged, summary = merge_manifests(core, supplement)
        self.assertEqual(len(merged), 2)
        self.assertEqual(summary["selected_core_rows"], 0)
        self.assertEqual(summary["selected_supplement_rows"], 2)
        self.assertEqual(merged[0]["manifest_source"], "supplement")
        self.assertEqual(merged[1]["manifest_source"], "supplement")

    def test_failure_tail_support_manifest_tracks_hard_support_rows(self) -> None:
        core = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 14,
                "step_idx": 3,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "takeover_tier": "coarse_pullback_candidate",
            }
        ]
        supplement = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 15,
                "step_idx": 4,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "observable",
                "takeover_tier": "near_basin_shell",
            }
        ]
        merged, summary = merge_manifests(core, supplement)
        self.assertEqual(summary["hard_support_rows"], 1)
        self.assertEqual(summary["hard_support_by_failure_bucket"]["large_xy_large_yaw"], 1)
        self.assertEqual(summary["hard_support_by_takeover_tier"]["coarse_pullback_candidate"], 1)
        self.assertEqual(summary["hard_support_by_yaw_observability"]["ambiguous"], 1)

    def test_failure_tail_support_manifest_includes_hard_window_outer_frontier_rows(self) -> None:
        core = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 7,
                "step_idx": 1,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_small_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "takeover_tier": "coarse_pullback_candidate",
            }
        ]
        hard_window = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 10,
                "step_idx": 83,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_large_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "unobservable",
                "takeover_tier": "outer_pullback_candidate",
                "selection_reason": "hard_window_pre_takeover_frontier",
            }
        ]
        merged, summary = merge_manifests(core, hard_window)
        self.assertEqual(summary["hard_support_rows"], 2)
        self.assertEqual(summary["hard_support_by_failure_bucket"]["small_xy_large_yaw"], 1)
        self.assertEqual(summary["hard_support_by_takeover_tier"]["outer_pullback_candidate"], 1)
        self.assertEqual(summary["hard_support_by_yaw_observability"]["unobservable"], 1)
        self.assertEqual(merged[-1]["failure_bucket"], "small_xy_large_yaw")
        self.assertEqual(merged[-1]["manifest_source"], "supplement")

    def test_failure_tail_support_manifest_prefers_harder_supplement_rows_for_same_key(self) -> None:
        core = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 14,
                "step_idx": 3,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "unobservable",
                "takeover_tier": "outside_takeover",
            }
        ]
        supplement = [
            {
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 14,
                "step_idx": 3,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "partial_observable",
                "yaw_observability_class": "ambiguous",
                "takeover_tier": "coarse_pullback_candidate",
            }
        ]
        merged, summary = merge_manifests(core, supplement)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["manifest_source"], "supplement")
        self.assertEqual(merged[0]["takeover_tier"], "coarse_pullback_candidate")
        self.assertEqual(summary["selected_core_rows"], 0)
        self.assertEqual(summary["selected_supplement_rows"], 1)
        self.assertEqual(summary["replaced_rows"], 1)
        self.assertEqual(summary["hard_support_rows"], 1)

    def test_hard_observability_supplement_prioritizes_recoverable_hard_rows(self) -> None:
        rows = [
            {
                "task_name": "insert_onto_square_peg",
                "episode_idx": 1,
                "step_idx": 1,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "takeover_tier": "outside_takeover",
                "source_runtime_obs_path": "obs1",
                "source_trace_path": "trace1",
            },
            {
                "task_name": "insert_onto_square_peg",
                "episode_idx": 2,
                "step_idx": 2,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "takeover_tier": "near_basin_shell",
                "source_runtime_obs_path": "obs2",
                "source_trace_path": "trace2",
            },
            {
                "task_name": "insert_onto_square_peg",
                "episode_idx": 3,
                "step_idx": 3,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_small_yaw",
                "visual_observability_class": "prior_only",
                "yaw_observability_class": "ambiguous",
                "takeover_tier": "outside_takeover",
                "source_runtime_obs_path": "obs3",
                "source_trace_path": "trace3",
            },
        ]
        fake_report = {
            "candidate_rows": 3,
            "recoverable_rows": 2,
            "recoverable_rate": 2 / 3,
            "rows": [
                {
                    "episode_idx": 1,
                    "step_idx": 1,
                    "recoverable_by_estimator": True,
                    "estimator_yaw_observable_probability": 0.91,
                    "estimator_yaw_observable_threshold": 0.075,
                    "estimator_yaw_margin": 0.835,
                },
                {
                    "episode_idx": 2,
                    "step_idx": 2,
                    "recoverable_by_estimator": True,
                    "estimator_yaw_observable_probability": 0.88,
                    "estimator_yaw_observable_threshold": 0.075,
                    "estimator_yaw_margin": 0.805,
                },
                {
                    "episode_idx": 3,
                    "step_idx": 3,
                    "recoverable_by_estimator": False,
                    "estimator_yaw_observable_probability": 0.01,
                    "estimator_yaw_observable_threshold": 0.075,
                    "estimator_yaw_margin": -0.065,
                },
            ],
        }
        with patch("scripts.build_c2c_v2_failure_tail_hard_observability_supplement.rank_candidates", return_value=fake_report):
            selected, summary = build_hard_observability_supplement(
                rows,
                checkpoint=ROOT / "runtime_artifacts" / "coarse2contact_v2" / "checkpoints" / "frame_yaw_estimator_observability_balanced.pt",
                threshold=0.075,
                max_rows_per_episode=10,
                easy_rows_per_coverage_key=1,
                hard_rows_per_coverage_key=2,
                hard_coverage_threshold=1,
                max_rows_per_coverage_key=2,
            )
        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["selected_rows"], 2)
        self.assertEqual(summary["selected_hard_rows"], 1)
        self.assertEqual(summary["selected_hard_support_rows"], 1)
        self.assertEqual(summary["by_recoverable"]["True"], 2)
        self.assertEqual(summary["by_selection_reason"]["hard_observability_recoverable_support"], 2)
        self.assertEqual(selected[0]["selection_reason"], "hard_observability_recoverable_support")
        self.assertEqual(summary["by_failure_bucket"]["small_xy_small_yaw"], 1)
        self.assertEqual(summary["by_failure_bucket"]["large_xy_large_yaw"], 1)

    def test_yaw_threshold_sweep_reports_entry_control_split(self) -> None:
        rows = [
            {
                "episode_idx": 1,
                "step_idx": 1,
                "visual_observability_class": "visual_observable",
                "source_frame_confidence": 0.30,
                "source_frame_observability": 0.02,
                "source_frame_axis_strength": 0.90,
                "yaw_control_observable": False,
                "yaw_entry_feasible": True,
                "proxy_local_geometry_error": {"dyaw": 0.04},
                "true_basin_error_t": {"dx": 0.020, "dy": 0.0, "dz": 0.0, "dyaw": 0.04},
                "xy_error": 0.020,
                "near_basin_shell": True,
            },
            {
                "episode_idx": 1,
                "step_idx": 2,
                "visual_observability_class": "visual_observable",
                "source_frame_confidence": 0.30,
                "source_frame_observability": 0.02,
                "source_frame_axis_strength": 0.90,
                "yaw_control_observable": False,
                "yaw_entry_feasible": True,
                "proxy_local_geometry_error": {"dyaw": -0.03},
                "true_basin_error_t": {"dx": 0.021, "dy": 0.0, "dz": 0.0, "dyaw": -0.03},
                "xy_error": 0.021,
                "near_basin_shell": True,
            },
        ]
        report = audit_yaw_threshold_sweep(
            rows,
            frame_observability_thresholds=[0.01, 0.10],
            confidence_thresholds=[0.20],
            axis_strength_thresholds=[0.80],
            near_yaw=0.08,
        )
        self.assertEqual(report["overall"]["yaw_entry_feasible_rows"], 2)
        self.assertEqual(report["overall"]["entry_feasible_control_blocked_rows"], 2)
        best = report["best_by_entry_feasible_rows"][0]
        self.assertEqual(best["min_frame_observability"], 0.01)
        self.assertEqual(best["entry_feasible_control_observable_rows"], 2)
        self.assertEqual(best["entry_feasible_selected"]["proxy_privileged_yaw_sign_match_rate"], 1.0)

    def test_yaw_frame_diagnostic_selects_entry_feasible_control_blocked_rows(self) -> None:
        rows = [
            {
                "episode_idx": 1,
                "step_idx": 1,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_small_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "yaw_entry_feasible": True,
                "yaw_control_observable": False,
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "yaw_observability_blocker_combo": "frame_observability_lt_010",
                "yaw_observability_frame_confidence": 0.9,
                "yaw_observability_frame_observability": 0.02,
                "yaw_observability_frame_axis_strength": 0.9,
                "proxy_local_geometry_error": {"valid": True, "dyaw": -0.04},
                "true_basin_error_t": {"dx": 0.02, "dy": 0.0, "dz": 0.0, "dyaw": 0.04},
                "privileged_dyaw": 0.04,
                "xy_error": 0.02,
                "near_basin_shell": True,
                "reference_frame_pose_7d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "target_frame_pose_7d": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0199987, 0.9998000],
                "frame_contract": {"yaw_mode": "square_symmetry_axis"},
            },
            {
                "episode_idx": 1,
                "step_idx": 2,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "yaw_entry_feasible": True,
                "yaw_control_observable": True,
                "proxy_local_geometry_error": {"valid": True, "dyaw": 0.04},
                "true_basin_error_t": {"dx": 0.02, "dy": 0.0, "dz": 0.0, "dyaw": 0.04},
            },
        ]
        selected, report = diagnose_yaw_frame_alignment(
            rows,
            stage_name="RING_GRASP_ALIGN",
            skill_type="precision_grasp",
            visual_only=True,
            default_symmetry_period=float(np.pi / 2.0),
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(report["overall"]["num_rows"], 1)
        self.assertEqual(report["overall"]["near_basin_shell_rows"], 1)
        self.assertEqual(report["counts"]["by_diagnosis_label"]["sign_flip_candidate"], 1)
        self.assertLess(report["overall"]["raw_pose_wrapped_privileged_mae"], 1.0e-5)

    def test_yaw_frame_diagnostic_uses_image_axis_yaw_when_present(self) -> None:
        rows = [
            {
                "episode_idx": 1,
                "step_idx": 1,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_small_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "yaw_entry_feasible": True,
                "yaw_control_observable": False,
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "yaw_observability_blocker_combo": "frame_observability_lt_010",
                "yaw_observability_frame_confidence": 0.9,
                "yaw_observability_frame_observability": 0.02,
                "yaw_observability_frame_axis_strength": 0.9,
                "proxy_local_geometry_error": {
                    "valid": True,
                    "dyaw": 0.0,
                    "yaw_valid": False,
                    "yaw_reason": "image_pca_axis_not_jaw_local_residual",
                    "image_axis_yaw": -0.04,
                },
                "true_basin_error_t": {"dx": 0.02, "dy": 0.0, "dz": 0.0, "dyaw": 0.04},
                "privileged_dyaw": 0.04,
                "xy_error": 0.02,
                "near_basin_shell": True,
                "reference_frame_pose_7d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "target_frame_pose_7d": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0199987, 0.9998000],
                "frame_contract": {"yaw_mode": "square_symmetry_axis"},
            }
        ]
        selected, report = diagnose_yaw_frame_alignment(
            rows,
            stage_name="RING_GRASP_ALIGN",
            skill_type="precision_grasp",
            visual_only=True,
            default_symmetry_period=float(np.pi / 2.0),
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["proxy_yaw_semantics"], "image_pca_axis_yaw")
        self.assertAlmostEqual(float(selected[0]["proxy_yaw"]), -0.04, places=6)
        self.assertAlmostEqual(float(selected[0]["symmetry_aware_proxy_yaw"]), 0.04, places=6)
        self.assertAlmostEqual(float(selected[0]["proxy_residual_yaw"]), 0.0, places=6)
        self.assertFalse(selected[0]["proxy_yaw_valid"])
        self.assertEqual(report["counts"]["by_diagnosis_label"]["sign_flip_candidate"], 1)
        self.assertAlmostEqual(float(report["overall"]["symmetry_aware_proxy_mae"]), 0.0, places=6)
        self.assertAlmostEqual(float(report["overall"]["symmetry_aware_proxy_bias"]), 0.0, places=6)
        self.assertAlmostEqual(float(report["overall"]["symmetry_aware_proxy_bias_corrected_mae"]), 0.0, places=6)

    def test_yaw_frame_diagnostic_symmetry_aware_baseline_matches_real_slice_scale(self) -> None:
        rows = [
            {
                "episode_idx": 10,
                "step_idx": 300,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_large_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "unobservable",
                "yaw_entry_feasible": True,
                "yaw_control_observable": False,
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "yaw_observability_blocker_combo": "frame_observability_lt_010",
                "yaw_observability_frame_confidence": 0.7,
                "yaw_observability_frame_observability": 0.08,
                "yaw_observability_frame_axis_strength": 0.95,
                "proxy_local_geometry_error": {"valid": True, "dyaw": 2.9327126811590665, "image_axis_yaw": 2.9327126811590665},
                "true_basin_error_t": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.15161998569965363},
                "privileged_dyaw": 0.15161998569965363,
                "xy_error": 0.001,
                "near_basin_shell": False,
                "reference_frame_pose_7d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "target_frame_pose_7d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "frame_contract": {"yaw_mode": "square_symmetry_axis"},
            }
        ]
        selected, report = diagnose_yaw_frame_alignment(
            rows,
            stage_name="RING_GRASP_ALIGN",
            skill_type="precision_grasp",
            visual_only=True,
            default_symmetry_period=float(np.pi / 2.0),
        )
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(float(selected[0]["symmetry_aware_proxy_yaw"]), 0.20887997243072665, places=6)
        self.assertAlmostEqual(float(report["overall"]["symmetry_aware_proxy_bias"]), 0.05725998673107302, places=6)
        self.assertAlmostEqual(float(report["overall"]["symmetry_aware_proxy_mae"]), 0.05725998673107302, places=6)
        self.assertAlmostEqual(float(report["overall"]["symmetry_aware_proxy_bias_corrected_mae"]), 0.0, places=6)

    def test_failure_tail_intervention_audit_distinguishes_planner_and_oracle(self) -> None:
        candidates = [
            {
                "schema_version": "grasp_failure_tail_candidate_v1",
                "episode_idx": 2,
                "step_idx": 10,
                "stage_name": "RING_GRASP_ALIGN",
                "failure_bucket": "xy_bias",
                "takeover_tier": "coarse_pullback_candidate",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "observable",
                "yaw_observable": True,
                "alias_drift_decision": "stable_alias_control",
                "alias_label": "stable_alias",
                "abstain_reason": "",
                "xy_error": 0.040,
                "yaw_abs": 0.030,
                "next_xy_error": 0.043,
                "next_yaw_abs": 0.030,
                "next_planner_residual": {"dx": 0.043, "dy": 0.0, "dz": 0.02, "dyaw": 0.03},
            }
        ]
        trace_rows = [
            {
                "episode_idx": 2,
                "step": 10,
                "grasp_probe_active": True,
                "grasp_probe_reason": "replay_oracle_xy",
                "grasp_probe_pre_true_error_t": [0.040, 0.0, 0.02, 0.03],
                "grasp_probe_horizon_final_true_error_t": [0.020, 0.0, 0.02, 0.03],
                "grasp_probe_horizon_near_grasp_after": False,
                "grasp_probe_horizon_overshoot": False,
            }
        ]
        report = audit_grasp_failure_tail_intervention(candidates, trace_rows)
        self.assertEqual(report["overall"]["active_failure_tail_rows"], 1)
        self.assertEqual(report["overall"]["planner_natural_contraction_rate"], 0.0)
        self.assertEqual(report["overall"]["oracle_intervention_contraction_rate"], 1.0)
        self.assertEqual(report["overall"]["intervention_vs_planner_improvement_rate"], 1.0)
        by_alias = {item["alias_drift_decision"]: item for item in report["active_by_alias_drift_decision"]}
        self.assertEqual(by_alias["stable_alias_control"]["active_failure_tail_rows"], 1)

    def test_failure_tail_intervention_audit_splits_blocked_reasons(self) -> None:
        candidates = [
            {
                "schema_version": "grasp_failure_tail_candidate_v1",
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 1,
                "step_idx": 1,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "prior_only",
                "yaw_observability_class": "unobservable",
                "takeover_tier": "abstain_prior_only",
                "abstain_reason": "prior_only",
                "yaw_control_observable": False,
            },
            {
                "schema_version": "grasp_failure_tail_candidate_v1",
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 1,
                "step_idx": 2,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "unobservable",
                "takeover_tier": "yaw_entry_blocked",
                "yaw_control_observable": False,
            },
            {
                "schema_version": "grasp_failure_tail_candidate_v1",
                "sample_role": "failure_tail_candidate",
                "task_name": "insert_onto_square_peg",
                "episode_idx": 1,
                "step_idx": 3,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_small_yaw",
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "observable",
                "takeover_tier": "too_far",
                "yaw_control_observable": True,
            },
        ]
        trace_rows = [
            {
                "episode_idx": 1,
                "step": 2,
                "grasp_probe_active": False,
                "grasp_probe_reason": "yaw_entry_blocked",
                "grasp_probe_pre_true_error_t": [0.02, 0.0, 0.0, 0.1],
                "grasp_probe_post_true_error_t": [0.02, 0.0, 0.0, 0.1],
            },
            {
                "episode_idx": 1,
                "step": 3,
                "grasp_probe_active": False,
                "grasp_probe_reason": "too_far",
                "grasp_probe_pre_true_error_t": [0.05, 0.0, 0.0, 0.01],
                "grasp_probe_post_true_error_t": [0.05, 0.0, 0.0, 0.01],
            },
        ]
        report = audit_grasp_failure_tail_intervention(candidates, trace_rows)
        self.assertEqual(report["overall"]["blocked_reason_counts"]["candidate_actionable"], 1)
        self.assertEqual(report["overall"]["blocked_reason_counts"]["yaw"], 1)
        self.assertEqual(report["overall"]["blocked_reason_counts"]["xy"], 1)
        by_bucket = {item["failure_bucket"]: item for item in report["by_failure_bucket"]}
        self.assertEqual(by_bucket["large_xy_large_yaw"]["blocked_reason_counts"]["candidate_actionable"], 1)
        self.assertEqual(by_bucket["large_xy_large_yaw"]["blocked_reason_counts"]["yaw"], 1)
        self.assertEqual(by_bucket["large_xy_small_yaw"]["blocked_reason_counts"]["xy"], 1)

    def test_hard_bucket_gap_report_groups_window_and_alias_drift(self) -> None:
        candidates = [
            {
                "episode_idx": 4,
                "step_idx": 7,
                "failure_bucket": "small_xy_large_yaw",
                "takeover_tier": "frontier_pullback_candidate",
                "yaw_observability_class": "ambiguous",
                "visual_observability_class": "visual_observable",
                "alias_drift_decision": "frame_drift_abstain",
            }
        ]
        traces = [
            {
                "episode_idx": 4,
                "step": 7,
                "grasp_probe_active": False,
                "grasp_probe_reason": "shell_yaw_blocked",
                "grasp_probe_window_protocol": "retain_h5",
                "grasp_probe_candidate_actionable": True,
            }
        ]
        report = build_gap_report(candidates, traces)
        self.assertEqual(report["overall"]["shell_yaw_blocked_rows"], 1)
        by_window = {item["window_protocol"]: item for item in report["by_window_protocol"]}
        by_alias = {item["alias_drift_decision"]: item for item in report["by_alias_drift_decision"]}
        self.assertEqual(by_window["retain_h5"]["candidate_rows"], 1)
        self.assertEqual(by_alias["frame_drift_abstain"]["shell_yaw_blocked_rows"], 1)

    def test_yaw_alias_drift_manifest_separates_calibration_positive_from_frame_drift(self) -> None:
        reports = [
            {
                "episode_idx": 6,
                "failure_bucket": "small_xy_small_yaw",
                "primary_blocker": "frame_observability_lt_010",
                "num_rows": 65,
                "raw_proxy_mae": 2.3452248130886866,
                "symmetry_aware_mae": 0.3043517796275638,
                "bias_corrected_mae": 0.3235587906100927,
                "num_jump_points": 16,
                "gif_path": "/tmp/ep6.gif",
                "jump_sheet_path": "/tmp/ep6_jump.png",
                "report_path": "/tmp/ep6.json",
            },
            {
                "episode_idx": 10,
                "failure_bucket": "large_xy_large_yaw",
                "primary_blocker": "frame_observability_lt_010",
                "num_rows": 12,
                "raw_proxy_mae": 3.636191816125864,
                "symmetry_aware_mae": 0.912514131270641,
                "bias_corrected_mae": 0.002537650165050994,
                "num_jump_points": 0,
                "gif_path": "/tmp/ep10.gif",
                "jump_sheet_path": "/tmp/ep10_jump.png",
                "report_path": "/tmp/ep10.json",
            },
            {
                "episode_idx": 14,
                "failure_bucket": "large_xy_small_yaw",
                "primary_blocker": "frame_observability_lt_010",
                "num_rows": 12,
                "raw_proxy_mae": 3.15390289411141,
                "symmetry_aware_mae": 0.10843765663148223,
                "bias_corrected_mae": 0.0029218889700358985,
                "num_jump_points": 0,
                "gif_path": "/tmp/ep14.gif",
                "jump_sheet_path": "/tmp/ep14_jump.png",
                "report_path": "/tmp/ep14.json",
            },
        ]
        rows, summary = build_alias_drift_manifest(reports)
        self.assertEqual(summary["by_acceptance_role"]["calibration_positive"], 2)
        self.assertEqual(summary["by_acceptance_role"]["frame_drift_hard_case"], 1)
        self.assertEqual(rows[0]["acceptance_role"], "calibration_positive")
        self.assertEqual(rows[0]["alias_label"], "stable_alias")
        self.assertEqual(rows[-1]["acceptance_role"], "frame_drift_hard_case")
        self.assertEqual(rows[-1]["alias_label"], "frame_drift")

    def test_yaw_alias_drift_baseline_fits_stable_alias_and_preserves_holdout_drift(self) -> None:
        relabel_rows = [
            {
                "episode_idx": 10,
                "step_idx": 308,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "proxy_local_geometry_error": {
                    "image_axis_yaw": 2.9327126811590665,
                    "confidence": 0.716,
                    "observability": 0.0833,
                    "fit_residual": 0.001,
                    "inlier_ratio": 0.999,
                },
                "yaw_observability_frame_confidence": 0.716,
                "yaw_observability_frame_observability": 0.0833,
                "yaw_observability_frame_axis_strength": 0.91,
                "wide_ring_visible": True,
                "yaw_observability_wrist_occluded": False,
                "visual_observability_class": "visual_observable",
                "planner_prior": {"local_delta_6d": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]},
                "true_basin_error_t": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.15161998569965363},
                "xy_error": 0.0,
            },
            {
                "episode_idx": 10,
                "step_idx": 314,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_large_yaw",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "proxy_local_geometry_error": {
                    "image_axis_yaw": 2.9101326811590664,
                    "confidence": 0.717,
                    "observability": 0.0837,
                    "fit_residual": 0.001,
                    "inlier_ratio": 0.999,
                },
                "yaw_observability_frame_confidence": 0.717,
                "yaw_observability_frame_observability": 0.0837,
                "yaw_observability_frame_axis_strength": 0.92,
                "wide_ring_visible": True,
                "yaw_observability_wrist_occluded": False,
                "visual_observability_class": "visual_observable",
                "planner_prior": {"local_delta_6d": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]},
                "true_basin_error_t": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.16161998569965363},
                "xy_error": 0.0,
            },
            {
                "episode_idx": 14,
                "step_idx": 308,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_small_yaw",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "proxy_local_geometry_error": {
                    "image_axis_yaw": 3.094,
                    "confidence": 0.735,
                    "observability": 0.089,
                    "fit_residual": 0.001,
                    "inlier_ratio": 0.999,
                },
                "yaw_observability_frame_confidence": 0.735,
                "yaw_observability_frame_observability": 0.089,
                "yaw_observability_frame_axis_strength": 0.93,
                "wide_ring_visible": True,
                "yaw_observability_wrist_occluded": False,
                "visual_observability_class": "visual_observable",
                "planner_prior": {"local_delta_6d": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]},
                "true_basin_error_t": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.019},
                "xy_error": 0.0,
            },
            {
                "episode_idx": 14,
                "step_idx": 314,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "large_xy_small_yaw",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "proxy_local_geometry_error": {
                    "image_axis_yaw": 3.102,
                    "confidence": 0.736,
                    "observability": 0.0892,
                    "fit_residual": 0.001,
                    "inlier_ratio": 0.999,
                },
                "yaw_observability_frame_confidence": 0.736,
                "yaw_observability_frame_observability": 0.0892,
                "yaw_observability_frame_axis_strength": 0.94,
                "wide_ring_visible": True,
                "yaw_observability_wrist_occluded": False,
                "visual_observability_class": "visual_observable",
                "planner_prior": {"local_delta_6d": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]},
                "true_basin_error_t": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.029},
                "xy_error": 0.0,
            },
            {
                "episode_idx": 6,
                "step_idx": 45,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "proxy_local_geometry_error": {
                    "image_axis_yaw": 2.2724691817002505,
                    "confidence": 0.4734681162238121,
                    "observability": 0.0074462890625,
                    "fit_residual": 0.0020000487565994263,
                    "inlier_ratio": 0.9979999512434006,
                },
                "yaw_observability_frame_confidence": 0.4734681162238121,
                "yaw_observability_frame_observability": 0.0074462890625,
                "yaw_observability_frame_axis_strength": 0.55,
                "wide_ring_visible": True,
                "yaw_observability_wrist_occluded": False,
                "visual_observability_class": "visual_observable",
                "planner_prior": {"local_delta_6d": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]},
                "true_basin_error_t": {"dx": 0.006, "dy": 0.014, "dz": 0.119, "dyaw": 0.015},
                "xy_error": 0.015,
            },
            {
                "episode_idx": 6,
                "step_idx": 46,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "proxy_local_geometry_error": {
                    "image_axis_yaw": 1.3926353536684106,
                    "confidence": 0.4725515164583921,
                    "observability": 0.0072021484375,
                    "fit_residual": 0.0027519918978214236,
                    "inlier_ratio": 0.9972480081021786,
                },
                "yaw_observability_frame_confidence": 0.4725515164583921,
                "yaw_observability_frame_observability": 0.0072021484375,
                "yaw_observability_frame_axis_strength": 0.55,
                "wide_ring_visible": True,
                "yaw_observability_wrist_occluded": False,
                "visual_observability_class": "visual_observable",
                "planner_prior": {"local_delta_6d": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]},
                "true_basin_error_t": {"dx": 0.006, "dy": 0.013, "dz": 0.115, "dyaw": 0.015},
                "xy_error": 0.014,
            },
            {
                "episode_idx": 6,
                "step_idx": 47,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "proxy_local_geometry_error": {
                    "image_axis_yaw": 2.179441759378892,
                    "confidence": 0.47625029053539036,
                    "observability": 0.00830078125,
                    "fit_residual": 0.0017344970256090164,
                    "inlier_ratio": 0.998265502974391,
                },
                "yaw_observability_frame_confidence": 0.47625029053539036,
                "yaw_observability_frame_observability": 0.00830078125,
                "yaw_observability_frame_axis_strength": 0.55,
                "wide_ring_visible": True,
                "yaw_observability_wrist_occluded": False,
                "visual_observability_class": "visual_observable",
                "planner_prior": {"local_delta_6d": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]},
                "true_basin_error_t": {"dx": 0.005, "dy": 0.012, "dz": 0.111, "dyaw": 0.016},
                "xy_error": 0.013,
            },
        ]
        train_reports = [
            {
                "episode_idx": 10,
                "failure_bucket": "large_xy_large_yaw",
                "primary_blocker": "frame_observability_lt_010",
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
            },
            {
                "episode_idx": 14,
                "failure_bucket": "large_xy_small_yaw",
                "primary_blocker": "frame_observability_lt_010",
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
            },
        ]
        holdout_reports = [
            {
                "episode_idx": 6,
                "failure_bucket": "small_xy_small_yaw",
                "primary_blocker": "frame_observability_lt_010",
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
            }
        ]
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            relabel_path = tmpdir / "relabel.jsonl"
            with open(relabel_path, "w", encoding="utf-8") as handle:
                for row in relabel_rows:
                    handle.write(json.dumps(row) + "\n")
            train_report_paths = []
            holdout_report_paths = []
            for idx, rep in enumerate(train_reports):
                p = tmpdir / f"train_report_{idx}.json"
                p.write_text(json.dumps(rep), encoding="utf-8")
                train_report_paths.append(p)
            for idx, rep in enumerate(holdout_reports):
                p = tmpdir / f"holdout_report_{idx}.json"
                p.write_text(json.dumps(rep), encoding="utf-8")
                holdout_report_paths.append(p)
            report = run_yaw_alias_drift_baseline(
                relabel_jsonl=relabel_path,
                train_reports=train_report_paths,
                holdout_reports=holdout_report_paths,
                output_dir=tmpdir / "out",
                ridge=1.0e-2,
            )
            self.assertLess(report["train"]["learned_mae"], report["train"]["raw_proxy_mae"])
            self.assertLess(report["train"]["learned_mae"], report["train"]["symmetry_aware_mae"])
            self.assertGreater(report["holdout"]["raw_proxy_mae"], report["train"]["learned_mae"])
        self.assertGreater(report["holdout"]["learned_mae"], report["train"]["learned_mae"])
        self.assertGreater(report["holdout_jump_points_raw_proxy"], 0)
        self.assertGreaterEqual(report["holdout_jump_points_predicted"], 0)

    def test_yaw_alias_drift_support_manifest_widens_with_diagnostic_rows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            seq_report = {
                "episode_idx": 10,
                "failure_bucket": "large_xy_large_yaw",
                "primary_blocker": "frame_observability_lt_010",
                "num_rows": 12,
                "raw_proxy_mae": 3.636191816125864,
                "symmetry_aware_mae": 0.912514131270641,
                "bias_corrected_mae": 0.002537650165050994,
                "num_jump_points": 0,
                "selected_step_idxs": [308, 314],
                "gif_path": "/tmp/ep10.gif",
                "jump_sheet_path": "/tmp/ep10_jump.png",
                "report_path": "/tmp/ep10.json",
            }
            seq_path = tmpdir / "ep10.json"
            seq_path.write_text(json.dumps(seq_report), encoding="utf-8")
            diag_path = tmpdir / "diag6.jsonl"
            diag_rows = [
                {
                    "episode_idx": 6,
                    "step_idx": 0,
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_name": "precision_grasp_ring",
                    "failure_bucket": "small_xy_small_yaw",
                    "visual_observability_class": "visual_observable",
                    "yaw_observability_class": "ambiguous",
                    "yaw_observability_primary_blocker": "frame_observability_lt_010",
                    "proxy_yaw": 1.50,
                    "privileged_yaw": 0.20,
                },
                {
                    "episode_idx": 6,
                    "step_idx": 1,
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_name": "precision_grasp_ring",
                    "failure_bucket": "small_xy_small_yaw",
                    "visual_observability_class": "visual_observable",
                    "yaw_observability_class": "ambiguous",
                    "yaw_observability_primary_blocker": "frame_observability_lt_010",
                    "proxy_yaw": -1.50,
                    "privileged_yaw": 0.21,
                },
                {
                    "episode_idx": 6,
                    "step_idx": 2,
                    "stage_name": "RING_GRASP_ALIGN",
                    "skill_name": "precision_grasp_ring",
                    "failure_bucket": "small_xy_small_yaw",
                    "visual_observability_class": "visual_observable",
                    "yaw_observability_class": "ambiguous",
                    "yaw_observability_primary_blocker": "frame_observability_lt_010",
                    "proxy_yaw": 1.60,
                    "privileged_yaw": 0.22,
                },
            ]
            with open(diag_path, "w", encoding="utf-8") as handle:
                for row in diag_rows:
                    row = dict(row)
                    row["source_relabel_jsonl"] = str(tmpdir / "relabel.jsonl")
                    handle.write(json.dumps(row) + "\n")
            rows, summary = build_support_manifest(sequence_reports=[seq_path], diagnostic_jsonl=[diag_path])
            self.assertEqual(summary["by_acceptance_role"]["calibration_positive"], 1)
            self.assertEqual(summary["by_acceptance_role"]["frame_drift_hard_case"], 1)
            self.assertEqual(summary["by_alias_label"]["stable_alias"], 1)
            self.assertEqual(summary["by_alias_label"]["frame_drift"], 1)
            self.assertEqual(len(rows), 2)
            by_ep = {row["episode_idx"]: row for row in rows}
            self.assertEqual(by_ep[10]["acceptance_role"], "calibration_positive")
            self.assertEqual(by_ep[10]["alias_label"], "stable_alias")
            self.assertEqual(by_ep[6]["acceptance_role"], "frame_drift_hard_case")
            self.assertEqual(by_ep[6]["alias_label"], "frame_drift")

    def test_yaw_alias_drift_row_support_manifest_keeps_window_level_positive_rows(self) -> None:
        diagnostic_rows = [
            {
                "episode_idx": 6,
                "step_idx": 10,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "diagnosis_label": "symmetry_alias_candidate",
                "best_symmetry_alias_abs_error": 0.008,
                "proxy_privileged_abs_error": 0.122,
                "best_symmetry_alias_k": 1,
                "best_symmetry_alias_yaw": 0.018,
                "proxy_yaw": 1.592,
                "privileged_yaw": 0.020,
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "frame_confidence": 0.72,
                "frame_observability": 0.11,
                "frame_axis_strength": 0.91,
                "xy_error": 0.016,
                "near_basin_shell": True,
                "micro_entry_ready": False,
                "source_relabel_jsonl": "/tmp/diag.jsonl",
            },
            {
                "episode_idx": 6,
                "step_idx": 11,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "diagnosis_label": "sign_flip_candidate",
                "best_symmetry_alias_abs_error": 0.012,
                "proxy_privileged_abs_error": 0.158,
                "best_symmetry_alias_k": -1,
                "best_symmetry_alias_yaw": 0.019,
                "proxy_yaw": -1.580,
                "privileged_yaw": 0.021,
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "frame_confidence": 0.70,
                "frame_observability": 0.10,
                "frame_axis_strength": 0.90,
                "xy_error": 0.015,
                "near_basin_shell": True,
                "micro_entry_ready": False,
                "source_relabel_jsonl": "/tmp/diag.jsonl",
            },
            {
                "episode_idx": 6,
                "step_idx": 12,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_name": "precision_grasp_ring",
                "skill_type": "precision_grasp",
                "failure_bucket": "small_xy_small_yaw",
                "diagnosis_label": "frame_definition_drift_candidate",
                "best_symmetry_alias_abs_error": 0.091,
                "proxy_privileged_abs_error": 0.104,
                "best_symmetry_alias_k": 0,
                "best_symmetry_alias_yaw": 0.122,
                "proxy_yaw": 1.910,
                "privileged_yaw": 0.106,
                "visual_observability_class": "visual_observable",
                "yaw_observability_class": "ambiguous",
                "frame_confidence": 0.61,
                "frame_observability": 0.08,
                "frame_axis_strength": 0.58,
                "xy_error": 0.020,
                "near_basin_shell": True,
                "micro_entry_ready": False,
                "source_relabel_jsonl": "/tmp/diag.jsonl",
            },
        ]
        rows, summary = build_row_support_manifest(diagnostic_rows, stable_alias_max_abs_error=0.02)
        self.assertEqual(summary["positive_rows"], 2)
        self.assertEqual(summary["hard_case_rows"], 1)
        self.assertEqual(summary["by_episode"]["ep006"], 3)
        self.assertEqual(rows[0]["acceptance_role"], "calibration_positive")
        self.assertEqual(rows[1]["acceptance_role"], "calibration_positive")
        self.assertEqual(rows[2]["acceptance_role"], "frame_drift_hard_case")
        self.assertEqual(rows[0]["selected_step_idxs"], [10])
        self.assertEqual(rows[1]["selected_step_idxs"], [11])
        self.assertEqual(rows[2]["selected_step_idxs"], [12])

    def test_yaw_alias_drift_support_manifest_and_two_stage_baseline_form_a_real_split(self) -> None:
        def make_row(
            episode_idx: int,
            step_idx: int,
            *,
            proxy_yaw: float,
            privileged_yaw: float,
            failure_bucket: str,
            visual_class: str = "visual_observable",
        ) -> dict[str, object]:
            return {
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "stage_name": "RING_GRASP_ALIGN",
                "skill_type": "precision_grasp",
                "failure_bucket": failure_bucket,
                "visual_observability_class": visual_class,
                "yaw_observability_class": "ambiguous" if visual_class == "visual_observable" else "unobservable",
                "yaw_observability_primary_blocker": "frame_observability_lt_010",
                "yaw_observability_blocker_combo": "frame_observability_lt_010",
                "yaw_observability_frame_confidence": 0.72,
                "yaw_observability_frame_observability": 0.083,
                "yaw_observability_frame_axis_strength": 0.91,
                "yaw_observability_wrist_occluded": False,
                "wide_ring_visible": True,
                "planner_prior": {"local_delta_6d": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]},
                "proxy_local_geometry_error": {
                    "image_axis_yaw": proxy_yaw,
                    "dyaw": 0.0,
                    "confidence": 0.72,
                    "observability": 0.083,
                    "fit_residual": 0.001,
                    "inlier_ratio": 0.999,
                },
                "estimated_basin_error": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0},
                "true_basin_error_t": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": privileged_yaw},
                "xy_error": 0.0,
                "yaw_abs": abs(privileged_yaw),
                "yaw_control_observable": visual_class == "visual_observable",
                "requires_yaw_observability": True,
                "frame_contract": {"requires_yaw_observability": True},
            }

        def make_support_row(
            episode_idx: int,
            *,
            alias_label: str,
            acceptance_role: str,
            failure_bucket: str,
            rows: int,
            selected_step_idxs: list[int],
            source_kind: str,
            source_path: str,
            report_path: str,
            raw_mae: float,
            symm_mae: float,
            bc_mae: float,
            jump_points: int,
        ) -> dict[str, object]:
            return {
                "schema_version": "yaw_alias_drift_support_manifest_v1",
                "episode_idx": episode_idx,
                "failure_bucket": failure_bucket,
                "primary_blocker": "frame_observability_lt_010",
                "rows": rows,
                "num_rows": rows,
                "selected_step_idxs": selected_step_idxs,
                "selected_step_count": len(selected_step_idxs),
                "acceptance_role": acceptance_role,
                "alias_label": alias_label,
                "support_role": acceptance_role,
                "support_label": alias_label,
                "acceptance_reason": "synthetic",
                "classification_source": source_kind,
                "source_kind": source_kind,
                "source_path": source_path,
                "report_path": report_path,
                "source_paths": [source_path],
                "source_relabel_jsonl": source_path,
                "source_row_count": rows,
                "raw_mae": raw_mae,
                "symmetry_aware_mae": symm_mae,
                "bias_corrected_mae": bc_mae,
                "jump_points": jump_points,
                "support_priority": [1, 1, rows, 1 if source_kind == "sequence_report" else 0],
            }

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            relabel_path = tmpdir / "relabel.jsonl"
            relabel_rows = []
            relabel_rows.extend(
                [
                    make_row(10, 0, proxy_yaw=2.93, privileged_yaw=0.1516, failure_bucket="large_xy_large_yaw"),
                    make_row(10, 1, proxy_yaw=2.91, privileged_yaw=0.1616, failure_bucket="large_xy_large_yaw"),
                    make_row(14, 0, proxy_yaw=3.09, privileged_yaw=0.019, failure_bucket="large_xy_small_yaw"),
                    make_row(14, 1, proxy_yaw=3.10, privileged_yaw=0.029, failure_bucket="large_xy_small_yaw"),
                    make_row(6, 0, proxy_yaw=1.50, privileged_yaw=0.20, failure_bucket="small_xy_small_yaw"),
                    make_row(6, 1, proxy_yaw=-1.50, privileged_yaw=0.21, failure_bucket="small_xy_small_yaw"),
                    make_row(6, 2, proxy_yaw=1.60, privileged_yaw=0.22, failure_bucket="small_xy_small_yaw"),
                    make_row(7, 0, proxy_yaw=1.45, privileged_yaw=0.18, failure_bucket="small_xy_small_yaw"),
                    make_row(7, 1, proxy_yaw=-1.55, privileged_yaw=0.19, failure_bucket="small_xy_small_yaw"),
                    make_row(7, 2, proxy_yaw=1.58, privileged_yaw=0.20, failure_bucket="small_xy_small_yaw"),
                ]
            )
            with open(relabel_path, "w", encoding="utf-8") as handle:
                for row in relabel_rows:
                    handle.write(json.dumps(row) + "\n")

            support_rows = [
                make_support_row(
                    10,
                    alias_label="stable_alias",
                    acceptance_role="calibration_positive",
                    failure_bucket="large_xy_large_yaw",
                    rows=2,
                    selected_step_idxs=[0, 1],
                    source_kind="sequence_report",
                    source_path=str(tmpdir / "ep10.json"),
                    report_path=str(tmpdir / "ep10.json"),
                    raw_mae=3.6,
                    symm_mae=0.91,
                    bc_mae=0.0025,
                    jump_points=0,
                ),
                make_support_row(
                    14,
                    alias_label="stable_alias",
                    acceptance_role="calibration_positive",
                    failure_bucket="large_xy_small_yaw",
                    rows=2,
                    selected_step_idxs=[0, 1],
                    source_kind="sequence_report",
                    source_path=str(tmpdir / "ep14.json"),
                    report_path=str(tmpdir / "ep14.json"),
                    raw_mae=3.1,
                    symm_mae=0.11,
                    bc_mae=0.0029,
                    jump_points=0,
                ),
                make_support_row(
                    6,
                    alias_label="frame_drift",
                    acceptance_role="frame_drift_hard_case",
                    failure_bucket="small_xy_small_yaw",
                    rows=3,
                    selected_step_idxs=[0, 1, 2],
                    source_kind="diagnostic_rows",
                    source_path=str(tmpdir / "diag6.jsonl"),
                    report_path=str(tmpdir / "diag6.jsonl"),
                    raw_mae=2.3,
                    symm_mae=0.32,
                    bc_mae=0.31,
                    jump_points=2,
                ),
                make_support_row(
                    7,
                    alias_label="frame_drift",
                    acceptance_role="frame_drift_hard_case",
                    failure_bucket="small_xy_small_yaw",
                    rows=3,
                    selected_step_idxs=[0, 1, 2],
                    source_kind="diagnostic_rows",
                    source_path=str(tmpdir / "diag7.jsonl"),
                    report_path=str(tmpdir / "diag7.jsonl"),
                    raw_mae=2.1,
                    symm_mae=0.41,
                    bc_mae=0.39,
                    jump_points=3,
                ),
            ]
            support_path = tmpdir / "support.jsonl"
            with open(support_path, "w", encoding="utf-8") as handle:
                for row in support_rows:
                    handle.write(json.dumps(row) + "\n")

            report = run_two_stage_baseline(
                relabel_jsonl=relabel_path,
                support_manifest_jsonl=support_path,
                output_dir=tmpdir / "out",
                holdout_ratio=0.25,
                seed=7,
            )
            self.assertEqual(report["support_summary"]["train_positive_rows"], 2)
            self.assertGreater(report["support_summary"]["holdout_frame_drift_rows"], 0)
            self.assertLessEqual(report["classifier"]["holdout_drift_false_accept_rate"], 1.0 / 3.0)
            self.assertGreaterEqual(report["classifier"]["holdout_drift_abstain_rate"], 2.0 / 3.0)
            self.assertGreaterEqual(report["classifier"]["holdout_positive_accept_rate"], 1.0)
            self.assertLess(report["regression"]["train"]["learned_mae"], 1.0e-3)
            self.assertLessEqual(report["end_to_end"]["drift_false_accept_rate"], 1.0 / 3.0)

    def test_yaw_alias_drift_threshold_calibration_falls_back_to_fit_anchor_on_degenerate_calibration(self) -> None:
        calibration_prob = np.array([0.02, 0.05, 0.10, 0.12], dtype=np.float64)
        calibration_target = np.zeros((4,), dtype=np.float64)
        report = _calibrate_threshold(
            calibration_prob,
            calibration_target,
            min_specificity=0.95,
            anchor_threshold=0.37,
        )
        self.assertEqual(report["selection_policy"], "anchor_fallback_degenerate_calibration")
        self.assertEqual(report["selected_threshold"], 0.37)
        self.assertEqual(report["anchor_threshold"], 0.37)
        self.assertEqual(report["best_threshold"], 1.0)
        self.assertEqual(report["best_specificity"], 1.0)
        self.assertEqual(report["best_recall"], 0.0)

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

    def test_grasp_probe_shell_fields_track_outer_frontier(self) -> None:
        shell_fields = grasp_probe_shell_fields(
            np.array([0.100, 0.0, 0.0, 0.150], dtype=np.float32),
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        self.assertFalse(shell_fields["grasp_probe_near_basin_shell"])
        self.assertFalse(shell_fields["grasp_probe_coarse_pullback_candidate"])
        self.assertTrue(shell_fields["grasp_probe_outer_pullback_candidate"])

    def test_grasp_probe_shell_fields_track_frontier_frontier(self) -> None:
        shell_fields = grasp_probe_shell_fields(
            np.array([0.150, 0.0, 0.0, 0.150], dtype=np.float32),
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        self.assertFalse(shell_fields["grasp_probe_near_basin_shell"])
        self.assertFalse(shell_fields["grasp_probe_outer_pullback_candidate"])
        self.assertTrue(shell_fields["grasp_probe_frontier_pullback_candidate"])

    def test_grasp_probe_inactive_reason_handles_outer_frontier_filter(self) -> None:
        shell_fields = grasp_probe_shell_fields(
            np.array([0.100, 0.0, 0.0, 0.150], dtype=np.float32),
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        reason = grasp_probe_inactive_reason(
            policy="replay_oracle_xy",
            stage_ok=True,
            visibility_bucket="visual_observable",
            has_error=True,
            finite_xy=True,
            shell_filter="frontier_pullback_feasible",
            shell_fields=shell_fields,
        )
        self.assertEqual(reason, "inactive")

    def test_grasp_probe_inactive_reason_handles_small_xy_large_yaw_frontier_filter(self) -> None:
        shell_fields = grasp_probe_shell_fields(
            np.array([0.070, 0.0, 0.0, 0.150], dtype=np.float32),
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        reason = grasp_probe_inactive_reason(
            policy="replay_oracle_xy",
            stage_ok=True,
            visibility_bucket="visual_observable",
            has_error=True,
            finite_xy=True,
            shell_filter="small_xy_large_yaw_frontier_feasible",
            shell_fields=shell_fields,
        )
        self.assertEqual(reason, "shell_outside_small_xy_large_yaw_frontier")

    def test_grasp_probe_shell_fields_use_yaw_from_6d_residual(self) -> None:
        shell_fields = grasp_probe_shell_fields(
            np.array([0.028, 0.0, 0.0, 2.4, 0.0, 0.040], dtype=np.float32),
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=3,
        )
        self.assertTrue(shell_fields["grasp_probe_yaw_feasible"])
        self.assertTrue(shell_fields["grasp_probe_coarse_pullback_candidate"])

    def test_grasp_probe_shell_fields_track_tight_near_shell(self) -> None:
        shell_fields = grasp_probe_shell_fields(
            np.array([0.017, 0.0, 0.0, 0.040], dtype=np.float32),
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=5,
        )
        self.assertTrue(shell_fields["grasp_probe_one_step_xy_feasible"])
        self.assertTrue(shell_fields["grasp_probe_tight_near_basin_shell"])
        self.assertTrue(shell_fields["grasp_probe_near_basin_shell"])

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

    def test_grasp_probe_inactive_reason_handles_tight_near_shell(self) -> None:
        shell_fields = grasp_probe_shell_fields(
            np.array([0.017, 0.0, 0.0, 0.120], dtype=np.float32),
            near_grasp_xy_threshold=0.015,
            near_grasp_yaw_threshold=0.08,
            max_xy_step=0.003,
            horizon_steps=5,
        )
        reason = grasp_probe_inactive_reason(
            policy="replay_oracle_xy",
            stage_ok=True,
            visibility_bucket="visual_observable",
            has_error=True,
            finite_xy=True,
            shell_filter="tight_near_yaw_feasible",
            shell_fields=shell_fields,
        )
        self.assertEqual(reason, "shell_yaw_blocked")

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

    def test_grasp_shell_episode_sweep_tracks_hard_support_episodes(self) -> None:
        chunk_reports = [
            {
                "chunk_tag": "chunk_000",
                "by_episode": [
                    {
                        "episode_idx": 2,
                        "near_basin_shell_rows": 0,
                        "coarse_pullback_candidate_rows": 0,
                        "horizon_xy_feasible_rows": 1,
                        "yaw_feasible_rows": 0,
                        "active_count": 2,
                    },
                ],
                "by_episode_failure_bucket": [
                    {
                        "episode_idx": 2,
                        "failure_bucket": "large_xy_large_yaw",
                        "near_basin_shell_rows": 0,
                        "coarse_pullback_candidate_rows": 2,
                        "horizon_xy_feasible_rows": 1,
                        "yaw_feasible_rows": 0,
                        "active_count": 2,
                    },
                    {
                        "episode_idx": 7,
                        "failure_bucket": "small_xy_small_yaw",
                        "near_basin_shell_rows": 3,
                        "coarse_pullback_candidate_rows": 1,
                        "horizon_xy_feasible_rows": 4,
                        "yaw_feasible_rows": 2,
                        "active_count": 5,
                    },
                ],
            }
        ]
        summary = _summarize_sweep_reports(chunk_reports, top_k=4, focus_radius=1)
        self.assertEqual(summary["hard_support_episode_indices"], [2])
        self.assertEqual(summary["hard_support_failure_buckets"], ["large_xy_large_yaw"])
        self.assertEqual(summary["hard_support_bucket_counts"]["large_xy_large_yaw"], 2)
        self.assertEqual(summary["hard_support_episode_focus_windows"][0]["center_episode_idx"], 2)
        self.assertEqual(summary["hard_support_episode_focus_windows"][0]["episode_indices"], [1, 2, 3])
        self.assertEqual(summary["hard_support_episode_bucket_rows"][0]["failure_bucket"], "large_xy_large_yaw")

    def test_grasp_shell_episode_sweep_counts_outer_pullback_rows(self) -> None:
        chunk_reports = [
            {
                "chunk_tag": "chunk_000",
                "by_episode": [
                    {
                        "episode_idx": 11,
                        "near_basin_shell_rows": 0,
                        "coarse_pullback_candidate_rows": 0,
                        "outer_pullback_candidate_rows": 2,
                        "horizon_xy_feasible_rows": 0,
                        "yaw_feasible_rows": 0,
                        "active_count": 2,
                    },
                ],
                "by_episode_failure_bucket": [
                    {
                        "episode_idx": 11,
                        "failure_bucket": "small_xy_large_yaw",
                        "near_basin_shell_rows": 0,
                        "coarse_pullback_candidate_rows": 0,
                        "outer_pullback_candidate_rows": 2,
                        "horizon_xy_feasible_rows": 0,
                        "yaw_feasible_rows": 0,
                        "active_count": 2,
                    },
                ],
            }
        ]
        summary = _summarize_sweep_reports(chunk_reports, top_k=4, focus_radius=1)
        self.assertEqual(summary["hard_support_episode_indices"], [11])
        self.assertEqual(summary["hard_support_bucket_counts"]["small_xy_large_yaw"], 2)

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

    def test_hard_window_support_supplement_promotes_relaxed_frontier_rows(self) -> None:
        rows = [
            {
                "episode_idx": 6,
                "step_idx": 3,
                "failure_bucket": "large_xy_small_yaw",
                "takeover_tier": "outside_takeover",
                "xy_error": 0.11,
                "abstain_reason": "candidate_too_far",
            },
            {
                "episode_idx": 6,
                "step_idx": 4,
                "failure_bucket": "small_xy_large_yaw",
                "takeover_tier": "yaw_entry_blocked",
                "xy_error": 0.16,
            },
            {
                "episode_idx": 9,
                "step_idx": 1,
                "failure_bucket": "large_xy_large_yaw",
                "takeover_tier": "coarse_pullback_candidate",
                "xy_error": 0.02,
            },
            {
                "episode_idx": 20,
                "step_idx": 0,
                "failure_bucket": "large_xy_small_yaw",
                "takeover_tier": "coarse_pullback_candidate",
                "xy_error": 0.02,
            },
        ]
        selected, summary = build_hard_window_support_supplement(
            rows,
            episode_windows=[(5, 7), (8, 10)],
            outer_xy_threshold=0.12,
            frontier_xy_threshold=0.18,
            max_rows_per_episode=10,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(summary["selected_rows"], 3)
        self.assertEqual(summary["strict_support_rows"], 1)
        self.assertEqual(summary["frontier_support_rows"], 2)
        self.assertEqual(summary["relaxed_frontier_rows"], 0)
        self.assertEqual(summary["selected_by_episode"]["ep006"], 2)
        self.assertEqual(summary["selected_by_episode"]["ep009"], 1)

        relaxed = next(row for row in selected if row["episode_idx"] == 6)
        self.assertEqual(relaxed["takeover_tier"], "frontier_pullback_candidate")
        self.assertEqual(relaxed["selection_reason"], "hard_window_frontier_support")
        self.assertEqual(relaxed["support_mode"], "frontier")
        self.assertEqual(relaxed["support_window_tag"], "5-7")
        self.assertTrue(relaxed["support_window_match"])
        self.assertEqual(relaxed["recommended_intervention_axes"], ["x", "y"])

        frontier = next(row for row in selected if row["episode_idx"] == 6 and row["step_idx"] == 4)
        self.assertEqual(frontier["takeover_tier"], "frontier_pullback_candidate")
        self.assertEqual(frontier["selection_reason"], "hard_window_frontier_support")
        self.assertEqual(frontier["support_mode"], "frontier")
        self.assertEqual(frontier["recommended_intervention_axes"], ["x", "y"])
        self.assertEqual(frontier["support_frontier"], "pre_takeover")

        strict = next(row for row in selected if row["episode_idx"] == 9)
        self.assertEqual(strict["takeover_tier"], "coarse_pullback_candidate")
        self.assertEqual(strict["selection_reason"], "hard_window_strict_support")
        self.assertEqual(strict["support_mode"], "strict")
        self.assertEqual(strict["support_window_tag"], "8-10")

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
        self.assertFalse(grasp_error.yaw_valid)
        self.assertEqual(grasp_error.yaw_reason, "image_pca_axis_not_jaw_local_residual")
        self.assertAlmostEqual(float(grasp_error.dyaw), 0.0, places=6)
        self.assertTrue(np.isfinite(float(grasp_error.image_axis_yaw)))

    def test_ring_grasp_pca_yaw_is_diagnostic_not_residual(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        skill = spec.get_skill("precision_grasp_ring")
        obs = _make_observation(blue_center=(48, 48))
        bundle = PrecisionObservationBundle.from_observation(obs)
        grasp_error = RingGraspLocalizer().localize(bundle, {}, spec, skill, stage_name="RING_GRASP_ALIGN")
        self.assertTrue(grasp_error.valid)
        self.assertFalse(grasp_error.yaw_valid)
        self.assertEqual(grasp_error.yaw_reason, "image_pca_axis_not_jaw_local_residual")
        self.assertAlmostEqual(float(grasp_error.dyaw), 0.0, places=6)
        self.assertTrue(np.isfinite(float(grasp_error.image_axis_yaw)))

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

    def test_precision_gate_splits_pullback_from_micro_depth(self) -> None:
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
        self.assertTrue(bool(trace["c2c_gate_active"]))
        self.assertTrue(bool(trace["c2c_gate_pullback_ready"]))
        self.assertFalse(bool(trace["c2c_gate_micro_entry_ready"]))
        self.assertFalse(bool(trace["c2c_gate_close_ready"]))
        self.assertEqual(trace["c2c_gate_reason"], "takeover_gate_armed")
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

    def test_frame_yaw_features_do_not_leak_privileged_yaw(self) -> None:
        row = {
            "episode_idx": 1,
            "step_idx": 4,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_type": "precision_grasp",
            "planner_prior": {"local_delta_6d": [0.001, -0.002, 0.0, 0.0, 0.0, 0.01]},
            "proxy_local_geometry_error": {
                "valid": True,
                "dx": 0.010,
                "dy": -0.004,
                "dz": 0.020,
                "dyaw": 0.0,
                "yaw_valid": False,
                "image_axis_yaw": -0.31,
                "confidence": 0.8,
                "observability": 0.04,
                "fit_residual": 0.02,
                "inlier_ratio": 0.98,
            },
            "estimated_basin_error": {
                "dx": 0.010,
                "dy": -0.004,
                "dz": 0.020,
                "dyaw": 0.0,
                "x_valid": True,
                "y_valid": True,
                "z_valid": False,
                "yaw_valid": False,
                "confidence": 0.7,
                "yaw_confidence": 0.0,
            },
            "obs_t": {
                "visual_observability_class": "visual_observable",
                "frame_confidence": 0.8,
                "frame_observability": 0.03,
                "frame_axis_strength": 0.9,
            },
            "frame_contract": {"requires_yaw_observability": True},
            "true_basin_error_t": {"dx": 0.02, "dy": 0.0, "dz": 0.0, "dyaw": 0.04},
            "privileged_dyaw": 0.04,
            "yaw_control_observable": False,
            "label_valid": True,
        }
        feat = frame_yaw_feature_vector(row)
        changed = dict(row)
        changed["privileged_dyaw"] = -0.42
        changed["true_basin_error_t"] = dict(row["true_basin_error_t"], dyaw=-0.42)
        np.testing.assert_allclose(feat, frame_yaw_feature_vector(changed))
        dyaw, yaw_obs, valid = frame_yaw_label_from_row(row)
        self.assertAlmostEqual(dyaw, 0.04, places=6)
        self.assertEqual(yaw_obs, 0.0)
        self.assertEqual(valid, 1.0)
        self.assertEqual(feat.shape[0], len(FRAME_YAW_FEATURE_NAMES))
        self.assertAlmostEqual(float(feat[FRAME_YAW_FEATURE_NAMES.index("proxy_image_axis_yaw")]), -0.31, places=6)

    def test_frame_yaw_estimator_outputs_residual_and_observability(self) -> None:
        model = FrameYawEstimatorNet()
        features = torch.zeros((3, len(FRAME_YAW_FEATURE_NAMES)), dtype=torch.float32)
        out = model(features)
        self.assertIn("dyaw", out)
        self.assertIn("yaw_observable_logit", out)
        self.assertIn("yaw_observable_probability", out)
        self.assertEqual(tuple(out["dyaw"].shape), (3,))
        self.assertTrue(torch.all(torch.abs(out["dyaw"]) <= model.max_abs_yaw + 1.0e-6))

    def test_frame_yaw_dataset_builder_filters_and_labels_rows(self) -> None:
        base = {
            "episode_idx": 1,
            "step_idx": 1,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_type": "precision_grasp",
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "proxy_local_geometry_error": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.0, "image_axis_yaw": 0.4},
            "estimated_basin_error": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.0},
            "visual_observability_class": "visual_observable",
            "true_basin_error_t": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.05},
            "yaw_control_observable": True,
            "label_valid": True,
        }
        rows = [
            base,
            dict(base, step_idx=2, stage_name="RING_SPOKE_ALIGN"),
            dict(base, step_idx=3, label_valid=False, true_basin_error_t={"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": float("nan")}),
        ]
        dataset = build_frame_yaw_dataset(rows, stage_name="RING_GRASP_ALIGN", skill_type="precision_grasp")
        self.assertEqual(dataset["features"].shape, (1, len(FRAME_YAW_FEATURE_NAMES)))
        self.assertAlmostEqual(float(dataset["dyaw"][0]), 0.05, places=6)
        self.assertEqual(float(dataset["yaw_observable"][0]), 1.0)

    def test_frame_yaw_dataset_builder_creates_balanced_positive_split(self) -> None:
        base = {
            "stage_name": "RING_GRASP_ALIGN",
            "skill_type": "precision_grasp",
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "proxy_local_geometry_error": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.0, "image_axis_yaw": 0.4},
            "estimated_basin_error": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.0},
            "visual_observability_class": "visual_observable",
            "true_basin_error_t": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.05},
            "label_valid": True,
            "yaw_entry_feasible": True,
            "near_basin_shell": True,
        }
        rows = []
        for ep in (1, 2):
            for step in range(3):
                rows.append(dict(base, episode_idx=ep, step_idx=step, yaw_control_observable=True))
        for ep in (3, 4):
            for step in range(4):
                rows.append(dict(base, episode_idx=ep, step_idx=step, yaw_control_observable=False))
        dataset = build_frame_yaw_dataset(
            rows,
            stage_name="RING_GRASP_ALIGN",
            skill_type="precision_grasp",
            balanced_split=True,
            val_ratio=0.5,
            seed=3,
            min_val_yaw_positive=1,
            min_val_focus_positive=1,
        )
        split = np.asarray(dataset["split"]).astype(str)
        self.assertGreater(int(np.count_nonzero(split == "train")), 0)
        self.assertGreater(int(np.count_nonzero(split == "val")), 0)
        self.assertGreater(int(np.count_nonzero((split == "val") & (dataset["yaw_observable"] > 0.5))), 0)
        self.assertGreater(int(np.count_nonzero((split == "train") & (dataset["yaw_observable"] > 0.5))), 0)
        self.assertGreater(int(np.count_nonzero((split == "val") & (dataset["yaw_positive_focus"] > 0.5))), 0)
        self.assertIn("pos_visual_entry_near", set(np.asarray(dataset["yaw_stratum"]).astype(str).tolist()))
        self.assertGreater(float(np.max(dataset["sample_weight"])), float(np.min(dataset["sample_weight"])))

    def test_frame_yaw_dataset_builder_balances_observability_pool(self) -> None:
        base = {
            "stage_name": "RING_GRASP_ALIGN",
            "skill_type": "precision_grasp",
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "proxy_local_geometry_error": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.0, "image_axis_yaw": 0.4},
            "estimated_basin_error": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.0},
            "visual_observability_class": "visual_observable",
            "true_basin_error_t": {"dx": 0.01, "dy": 0.0, "dz": 0.02, "dyaw": 0.05},
            "label_valid": True,
            "yaw_entry_feasible": True,
            "near_basin_shell": True,
        }
        rows = []
        for ep in range(1, 7):
            rows.append(dict(base, episode_idx=ep, step_idx=0, yaw_control_observable=True))
            rows.append(dict(base, episode_idx=ep, step_idx=1, yaw_control_observable=False))
            rows.append(dict(base, episode_idx=ep, step_idx=2, yaw_control_observable=False, visual_observability_class="prior_only"))
        dataset = build_frame_yaw_dataset(
            rows,
            stage_name="RING_GRASP_ALIGN",
            skill_type="precision_grasp",
            balanced_split=True,
            balance_observability_pool=True,
            observability_negative_to_positive_ratio=1.0,
            val_ratio=0.5,
            seed=11,
            min_val_yaw_positive=2,
            min_val_focus_positive=2,
        )
        self.assertLess(dataset["features"].shape[0], len(rows))
        split = np.asarray(dataset["split"]).astype(str)
        self.assertGreater(int(np.count_nonzero(split == "train")), 0)
        self.assertGreater(int(np.count_nonzero(split == "val")), 0)
        self.assertGreater(int(np.count_nonzero((split == "train") & (dataset["yaw_observable"] > 0.5))), 0)
        self.assertGreater(int(np.count_nonzero((split == "val") & (dataset["yaw_observable"] > 0.5))), 0)
        self.assertGreater(float(np.mean(dataset["yaw_observable"] > 0.5)), 0.2)

    def test_yaw_positive_window_miner_reports_target_and_gaps(self) -> None:
        base = {
            "task_name": "insert_onto_square_peg",
            "episode_idx": 6,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_name": "precision_grasp_ring",
            "skill_type": "precision_grasp",
            "label_valid": True,
            "visual_observability_class": "visual_observable",
            "true_basin_error_t": {"dx": 0.010, "dy": 0.0, "dz": 0.02, "dyaw": 0.04},
            "xy_error": 0.010,
            "yaw_abs": 0.04,
            "yaw_entry_feasible": True,
            "near_basin_shell": True,
            "failure_bucket": "small_xy_small_yaw",
        }
        rows = [
            dict(base, step_idx=1, yaw_control_observable=True, yaw_observable=True),
            dict(base, step_idx=2, yaw_control_observable=False, yaw_observable=False, yaw_observability_primary_blocker="frame_axis_ambiguous"),
            dict(base, episode_idx=7, step_idx=3, yaw_control_observable=True, yaw_observable=True, near_basin_shell=False),
        ]
        target, report = mine_yaw_positive_windows(rows, min_target_rows=1, min_target_episodes=1)
        self.assertEqual(len(target), 1)
        self.assertEqual(report["overall"]["target_rows"], 1)
        self.assertEqual(report["overall"]["near_visual_yaw_blocked_rows"], 1)
        self.assertEqual(report["overall"]["yaw_observable_visual_not_near_rows"], 1)
        self.assertEqual(report["overall"]["recommendation"], "use_target_windows_for_frame_yaw_eval")
        self.assertEqual(report["counts"]["target_by_episode"]["6"], 1)

    def test_frame_yaw_focused_eval_selects_near_basin_visual_windows(self) -> None:
        base = {
            "episode_idx": 6,
            "step_idx": 58,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_type": "precision_grasp",
            "visual_observability_class": "visual_observable",
            "near_basin_shell": True,
            "yaw_entry_feasible": True,
            "label_valid": True,
            "yaw_control_observable": False,
            "true_basin_error_t": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.04},
            "proxy_local_geometry_error": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0, "image_axis_yaw": 1.2},
            "estimated_basin_error": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0},
            "obs_t": {"visual_observability_class": "visual_observable", "frame_observability": 0.03},
        }
        rows = [base, dict(base, step_idx=59, visual_observability_class="prior_only", obs_t={"visual_observability_class": "prior_only", "frame_observability": 0.0})]
        selected, report = evaluate_frame_yaw_estimator(
            rows,
            checkpoint=None,
            visual_only=True,
            near_basin_only=True,
            min_frame_observability=0.02,
            require_not_wrist_occluded=False,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(report["overall"]["rows"], 1)
        self.assertAlmostEqual(report["overall"]["proxy_image_axis_yaw_mae"], 1.16, places=2)
        self.assertEqual(report["selection"]["pca_yaw_is_diagnostic_only"], True)
        self.assertEqual(report["overall"]["near_basin_rows"], 1)
        self.assertEqual(report["overall"]["yaw_entry_feasible_rows"], 1)

    def test_calibrated_yaw_observability_threshold_is_resolved_from_checkpoint_metadata(self) -> None:
        row = {
            "episode_idx": 6,
            "step_idx": 58,
            "stage_name": "RING_GRASP_ALIGN",
            "skill_type": "precision_grasp",
            "visual_observability_class": "visual_observable",
            "obs_t": {
                "visual_observability_class": "visual_observable",
                "frame_confidence": 0.9,
                "frame_observability": 0.12,
                "frame_axis_strength": 0.88,
                "wide_ring_visible": True,
            },
            "planner_prior": {"local_delta_6d": [0.0] * 6},
            "proxy_local_geometry_error": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0, "image_axis_yaw": 0.0},
            "estimated_basin_error": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0},
            "requires_yaw_observability": True,
            "yaw_control_observable": False,
            "true_basin_error_t": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.04},
            "xy_error": 0.0,
            "yaw_abs": 0.04,
            "yaw_entry_feasible": True,
            "near_basin_shell": True,
            "label_valid": True,
            "frame_contract": {"requires_yaw_observability": True},
        }
        with TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "frame_yaw.pt"
            model = FrameYawEstimatorNet(feature_dim=len(FRAME_YAW_FEATURE_NAMES))
            save_frame_yaw_checkpoint(
                ckpt_path,
                model,
                metadata={
                    "val": {"threshold_best_threshold": 0.075, "threshold_best_balanced_accuracy": 0.97},
                    "calibrated_yaw_observable_threshold": 0.075,
                },
            )
            resolved = resolve_yaw_observable_threshold({"val": {"threshold_best_threshold": 0.075}}, default=0.5)
            self.assertAlmostEqual(resolved, 0.075, places=6)
            rows = [dict(row)]
            calibrated = _apply_calibrated_yaw_observability(rows, checkpoint=ckpt_path)
            self.assertAlmostEqual(calibrated["threshold"], 0.075, places=6)
            self.assertEqual(calibrated["threshold_source"], "checkpoint_metadata")
            self.assertEqual(calibrated["rows"], 1)
            self.assertEqual(rows[0]["yaw_control_observable_threshold"], 0.075)
            self.assertEqual(rows[0]["yaw_control_observable_source"], "calibrated_checkpoint")

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

    def test_calibrated_basin_estimator_rejects_image_axis_yaw_even_if_calibrated(self) -> None:
        spec = load_precision_task_spec("insert_onto_square_peg")
        assert spec is not None
        skill = spec.get_skill("precision_grasp_ring")
        estimator = CalibratedGraspBasinEstimator(
            BasinStateCalibration(
                x=BasinAxisCalibration(valid=True, policy="trusted_control", confidence=1.0),
                y=BasinAxisCalibration(valid=True, policy="trusted_control", confidence=1.0),
                z=BasinAxisCalibration(valid=True, policy="trusted_control", confidence=1.0),
                yaw=BasinAxisCalibration(valid=True, policy="trusted_control", confidence=1.0),
                yaw_confidence_floor=0.0,
            )
        )
        local_error = LocalGeometryError(
            valid=True,
            confidence=0.9,
            dx=0.004,
            dy=-0.003,
            dz=0.020,
            dyaw=0.31,
            observability=0.04,
            fit_residual=0.02,
            inlier_ratio=0.98,
            reason="ok",
            target_entity="ring_grasp_frame",
            reference_entity="gripper_jaw_frame",
            stage_name="RING_GRASP_ALIGN",
            yaw_valid=False,
            yaw_reason="image_pca_axis_not_jaw_local_residual",
            image_axis_yaw=0.31,
        )
        est = estimator.estimate(
            local_error,
            robot_state={"wrist_valid_depth_ratio": 0.2, "wrist_depth_near_fraction": 0.2},
            task_spec=spec,
            skill_spec=skill,
            stage_name="RING_GRASP_ALIGN",
        )
        self.assertTrue(est.x_valid)
        self.assertTrue(est.y_valid)
        self.assertFalse(est.yaw_valid)
        self.assertAlmostEqual(float(est.dyaw), 0.0, places=6)
        self.assertAlmostEqual(float(est.proxy_dyaw), 0.0, places=6)
        self.assertAlmostEqual(float(est.proxy_image_axis_yaw), 0.31, places=6)
        trace = est.to_trace()
        self.assertFalse(trace["estimated_basin_error_yaw_valid"])
        self.assertAlmostEqual(float(trace["estimated_basin_error_proxy_image_axis_yaw"]), 0.31, places=6)

    def test_basin_visual_error_does_not_expose_invalid_image_axis_yaw(self) -> None:
        sup = PrecisionSkillSupervisor(load_precision_task_spec("insert_onto_square_peg"), mode="basin_recovery_only")
        visual = sup._basin_visual_error(
            LocalGeometryError(
                valid=True,
                confidence=0.8,
                dx=0.010,
                dy=-0.006,
                dz=0.020,
                dyaw=0.45,
                observability=0.03,
                fit_residual=0.02,
                inlier_ratio=0.98,
                reason="ok",
                yaw_valid=False,
                yaw_reason="image_pca_axis_not_jaw_local_residual",
                image_axis_yaw=0.45,
            )
        )
        self.assertIsNotNone(visual)
        assert visual is not None
        self.assertAlmostEqual(float(visual[0]), 0.010, places=6)
        self.assertAlmostEqual(float(visual[1]), -0.006, places=6)
        self.assertAlmostEqual(float(visual[2]), 0.0, places=6)

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
        self.assertIn("basin_pullback_gate_ready", trace)
        self.assertIn("basin_micro_entry_ready", trace)
        self.assertIn("basin_close_ready", trace)
        self.assertIn("basin_pullback_block_reason", trace)
        self.assertIn("basin_close_block_reason", trace)
        self.assertIn("c2c_gate_pullback_ready", trace)
        self.assertIn("c2c_gate_micro_entry_ready", trace)
        self.assertIn("c2c_gate_close_ready", trace)
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
