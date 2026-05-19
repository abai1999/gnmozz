"""
stage_target_provider.py

Target-provider abstraction that separates teacher-time privileged geometry from
deployment-time target estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

from prismatic.models.handoff_predictor import HandoffPredictor
from prismatic.models.near_ready_xyyaw_predictor import NearReadyXYYawPredictor
from prismatic.models.student_handoff_state_head_v2 import StudentHandoffStateHeadV2
from prismatic.models.target_delta_predictor import TargetDeltaPredictor
from prismatic.robot.residual_transforms import local_delta_to_world
from prismatic.robot.stage_manager import StageTargetMode


def pose_delta_local_between(current_pose_7d, target_pose_7d):
    current_pose_7d = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
    target_pose_7d = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
    delta_pos_world = target_pose_7d[:3] - current_pose_7d[:3]
    r_cur = Rotation.from_quat(current_pose_7d[3:7])
    r_tgt = Rotation.from_quat(target_pose_7d[3:7])
    delta_rot = (r_tgt * r_cur.inv()).as_rotvec().astype(np.float32)
    delta_pos_local = r_cur.inv().apply(delta_pos_world.astype(np.float32)).astype(np.float32)
    return np.concatenate([delta_pos_local, delta_rot], axis=0).astype(np.float32)


def wrap_yaw_to_symmetry(yaw: float, period: float) -> float:
    period = float(period)
    if not np.isfinite(period) or period <= 0.0:
        return float(yaw)
    return float(((float(yaw) + 0.5 * period) % period) - 0.5 * period)


def apply_yaw_symmetry_to_delta(delta_local, yaw_symmetry_period: float = -1.0):
    delta = np.asarray(delta_local, dtype=np.float32).copy()
    if delta.size >= 6 and float(yaw_symmetry_period) > 0.0:
        delta[5] = wrap_yaw_to_symmetry(float(delta[5]), float(yaw_symmetry_period))
    return delta.astype(np.float32)


def build_object_center_target_pose(current_object_pose_7d, anchor_pose_7d):
    current_object_pose = np.asarray(current_object_pose_7d, dtype=np.float32).reshape(7)
    anchor_pose = np.asarray(anchor_pose_7d, dtype=np.float32).reshape(7)
    target_pose = anchor_pose.copy()
    target_pose[:2] = current_object_pose[:2]
    target_pose[2] = float(anchor_pose[2])
    return target_pose.astype(np.float32)


def apply_object_frame_offset(object_pose_7d, offset_pose_7d):
    object_pose = np.asarray(object_pose_7d, dtype=np.float32).reshape(7)
    offset_pose = np.asarray(offset_pose_7d, dtype=np.float32).reshape(7)
    r_obj = Rotation.from_quat(object_pose[3:7])
    r_off = Rotation.from_quat(offset_pose[3:7])
    target_pose = np.zeros((7,), dtype=np.float32)
    target_pose[:3] = object_pose[:3] + r_obj.apply(offset_pose[:3]).astype(np.float32)
    target_pose[3:7] = (r_obj * r_off).as_quat().astype(np.float32)
    return target_pose.astype(np.float32)


def apply_local_offset_to_pose(pose_7d, delta_local_6d):
    pose = np.asarray(pose_7d, dtype=np.float32).copy().reshape(7)
    delta = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    pose[:3] = pose[:3] + local_delta_to_world(delta, pose[3:7])[:3].astype(np.float32)
    r_cur = Rotation.from_quat(pose[3:7])
    r_delta = Rotation.from_rotvec(delta[3:6].astype(np.float32))
    pose[3:7] = (r_delta * r_cur).as_quat().astype(np.float32)
    return pose.astype(np.float32)


def safe_task_low_dim_pose_7d_from_task(task) -> Optional[np.ndarray]:
    if task is None:
        return None
    try:
        low = np.asarray(task.get_low_dim_state(), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if low.size >= 7 and np.all(np.isfinite(low[:7])):
        return low[:7].astype(np.float32)
    return None


def resolve_task_success_centre_pose(task, obs=None) -> Optional[np.ndarray]:
    pose = safe_task_low_dim_pose_7d_from_task(task)
    if pose is not None:
        return pose
    if obs is not None:
        low = np.asarray(getattr(obs, "task_low_dim_state", []), dtype=np.float32).reshape(-1)
        if low.size >= 7 and np.all(np.isfinite(low[:7])):
            return low[:7].astype(np.float32)
    return None


def align_square_edge_pair_target_pose(
    current_gripper_pose_7d,
    grasp_commit_target_pose_7d,
    yaw_symmetry_period: float,
):
    current_gripper_pose = np.asarray(current_gripper_pose_7d, dtype=np.float32).reshape(7)
    grasp_commit_target_pose = np.asarray(grasp_commit_target_pose_7d, dtype=np.float32).reshape(7)
    if not np.all(np.isfinite(current_gripper_pose[:7])) or not np.all(np.isfinite(grasp_commit_target_pose[:7])):
        return grasp_commit_target_pose.astype(np.float32), -1, -1, 0.0
    period = float(yaw_symmetry_period)
    if not np.isfinite(period) or period <= 0.0:
        period = float(np.pi / 2.0)
    current_yaw = float(Rotation.from_quat(current_gripper_pose[3:7]).as_euler("zyx", degrees=False)[0])
    target_rot = Rotation.from_quat(grasp_commit_target_pose[3:7])
    best_pose = grasp_commit_target_pose.copy()
    best_err = float("inf")
    best_quadrant = -1
    for quadrant in range(4):
        cand_rot = target_rot * Rotation.from_euler("z", float(quadrant) * period, degrees=False)
        cand_pose = grasp_commit_target_pose.copy()
        cand_pose[3:7] = cand_rot.as_quat().astype(np.float32)
        cand_yaw = float(cand_rot.as_euler("zyx", degrees=False)[0])
        yaw_err = float(((cand_yaw - current_yaw + np.pi) % (2.0 * np.pi)) - np.pi)
        yaw_err_abs = abs(yaw_err)
        if yaw_err_abs < best_err:
            best_err = yaw_err_abs
            best_pose = cand_pose.astype(np.float32)
            best_quadrant = quadrant
    return (
        best_pose.astype(np.float32),
        int(best_quadrant),
        int(best_quadrant % 2 if best_quadrant >= 0 else -1),
        float(best_err if np.isfinite(best_err) else 0.0),
    )


def compose_pose_with_orientation_contract(current_pose_7d, delta_local_6d, orientation_contract_pose_7d):
    current_pose = np.asarray(current_pose_7d, dtype=np.float32).reshape(7)
    delta = np.asarray(delta_local_6d, dtype=np.float32).reshape(6)
    contract_pose = np.asarray(orientation_contract_pose_7d, dtype=np.float32).reshape(7)
    target_pose = current_pose.copy()
    target_pose[:3] = target_pose[:3] + local_delta_to_world(delta, current_pose[3:7])[:3].astype(np.float32)
    target_pose[3:7] = contract_pose[3:7].astype(np.float32)
    return target_pose.astype(np.float32)


def basin_distance_bin(dist: float) -> int:
    if dist <= 0.9:
        return 0
    if dist <= 1.05:
        return 1
    if dist <= 1.2:
        return 2
    return 3


@dataclass
class TargetProviderResult:
    motion_target_pose_7d: Optional[np.ndarray]
    motion_target_delta_local: Optional[np.ndarray]
    target_pose_7d: Optional[np.ndarray]
    target_delta_local: Optional[np.ndarray]
    provider_name: str
    source: str
    uses_privileged_target: bool
    stage_target_mode: int
    handoff_ready: bool = False
    handoff_metrics: dict[str, float] = field(default_factory=dict)
    handoff_metric_thresholds: dict[str, float] = field(default_factory=dict)
    handoff_release_metric_thresholds: dict[str, float] = field(default_factory=dict)
    handoff_optimization_metric_thresholds: dict[str, float] = field(default_factory=dict)
    handoff_target_pose_7d: Optional[np.ndarray] = None
    handoff_spec_name: str = "none"
    handoff_target_role: str = "none"
    handoff_uses_privileged: bool = False
    handoff_min_stable_frames: int = 1
    yaw_symmetry_period: float = -1.0


@dataclass
class StageHandoffSpec:
    task_name: str
    substage_id: int
    motion_target_mode: str = "active_target"
    handoff_target_mode: str = "spec_target"
    target_frame: str = "active_target"
    metric_thresholds: dict[str, float] = field(default_factory=dict)
    release_metric_thresholds: dict[str, float] = field(default_factory=dict)
    optimization_metric_thresholds: dict[str, float] = field(default_factory=dict)
    target_offset_local_7d: Optional[np.ndarray] = None
    yaw_symmetry_period: float = -1.0
    min_stable_frames: int = 1
    required_gripper_state: str = "open"
    required_contact_state: str = "any"
    source: str = "manual_fallback"
    name: str = "default_handoff_spec"
    target_role: str = "pregrasp"
    uses_privileged: bool = True


@dataclass
class Phase1GraspFrameSpec:
    grasp_offset_local_7d: np.ndarray
    pregrasp_hover_offset_local_6d: np.ndarray
    close_xy_threshold: float
    close_abs_z_threshold: float
    close_yaw_threshold: float
    yaw_symmetry_period: float
    orientation_rescue_enabled: bool
    orientation_rescue_xy_threshold: float
    orientation_rescue_angle_threshold_deg: float
    commit_switch_xy_threshold: float = 0.010
    commit_switch_z_threshold: float = 0.020
    commit_switch_yaw_threshold: float = 0.12
    name: str = "default_phase1_grasp_spec"


@dataclass
class TeacherPhase1TargetResult(TargetProviderResult):
    pregrasp_target_pose_7d: Optional[np.ndarray] = None
    grasp_commit_target_pose_7d: Optional[np.ndarray] = None
    grasp_spec_name: str = "none"
    grasp_commit_edge_pair_index: int = -1
    grasp_commit_edge_pair_family: int = -1
    grasp_commit_edge_pair_yaw_error: float = np.nan
    use_commit_target: bool = False
    orientation_rescue_enabled: bool = False
    close_xy_threshold: float = 0.010
    close_abs_z_threshold: float = 0.010
    close_yaw_threshold: float = -1.0
    commit_switch_xy_threshold: float = 0.012
    commit_switch_z_threshold: float = 0.012
    commit_switch_yaw_threshold: float = -1.0


def build_phase1_teacher_targets(current_object_pose_7d, grasp_spec: Phase1GraspFrameSpec):
    grasp_commit_target = apply_object_frame_offset(
        current_object_pose_7d,
        np.asarray(grasp_spec.grasp_offset_local_7d, dtype=np.float32),
    )
    pregrasp_target = apply_local_offset_to_pose(
        grasp_commit_target,
        np.asarray(grasp_spec.pregrasp_hover_offset_local_6d, dtype=np.float32),
    )
    return pregrasp_target.astype(np.float32), grasp_commit_target.astype(np.float32)


def select_phase1_teacher_target(
    *,
    current_gripper_pose,
    pregrasp_target_pose_7d,
    grasp_commit_target_pose_7d,
    grasp_spec: Phase1GraspFrameSpec,
):
    cur_to_pregrasp = pose_delta_local_between(current_gripper_pose, pregrasp_target_pose_7d)
    xy_pre = float(np.linalg.norm(cur_to_pregrasp[:2]))
    z_pre = float(abs(cur_to_pregrasp[2]))
    yaw_pre = float(abs(cur_to_pregrasp[5]))
    yaw_ok = True
    if float(grasp_spec.commit_switch_yaw_threshold) >= 0.0:
        yaw_ok = yaw_pre <= float(grasp_spec.commit_switch_yaw_threshold)
    use_commit_target = bool(
        xy_pre <= float(grasp_spec.commit_switch_xy_threshold)
        and z_pre <= float(grasp_spec.commit_switch_z_threshold)
        and yaw_ok
    )
    active_target = grasp_commit_target_pose_7d if use_commit_target else pregrasp_target_pose_7d
    return np.asarray(active_target, dtype=np.float32), use_commit_target


def load_phase1_grasp_spec(task_name: Optional[str] = None) -> Phase1GraspFrameSpec:
    task_key = str(task_name or "insert_onto_square_peg")
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "grasp_specs" / f"{task_key}_phase1.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text())
            return Phase1GraspFrameSpec(
                grasp_offset_local_7d=np.asarray(raw["grasp_offset_local_7d"], dtype=np.float32).reshape(7),
                pregrasp_hover_offset_local_6d=np.asarray(raw["pregrasp_hover_offset_local_6d"], dtype=np.float32).reshape(6),
                close_xy_threshold=float(raw.get("close_xy_threshold", 0.010)),
                close_abs_z_threshold=float(raw.get("close_abs_z_threshold", 0.010)),
                close_yaw_threshold=float(raw.get("close_yaw_threshold", -1.0)),
                yaw_symmetry_period=float(raw.get("yaw_symmetry_period", -1.0)),
                orientation_rescue_enabled=bool(raw.get("orientation_rescue_enabled", False)),
                orientation_rescue_xy_threshold=float(raw.get("orientation_rescue_xy_threshold", 0.015)),
                orientation_rescue_angle_threshold_deg=float(raw.get("orientation_rescue_angle_threshold_deg", 8.0)),
                commit_switch_xy_threshold=float(raw.get("commit_switch_xy_threshold", 0.012)),
                commit_switch_z_threshold=float(raw.get("commit_switch_z_threshold", 0.012)),
                commit_switch_yaw_threshold=float(raw.get("commit_switch_yaw_threshold", -1.0)),
                name=str(raw.get("name", config_path.stem)),
            )
        except Exception:
            pass
    return Phase1GraspFrameSpec(
        grasp_offset_local_7d=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        pregrasp_hover_offset_local_6d=np.asarray([0.0, 0.0, 0.015, 0.0, 0.0, 0.0], dtype=np.float32),
        close_xy_threshold=0.010,
        close_abs_z_threshold=0.010,
        close_yaw_threshold=-1.0,
        yaw_symmetry_period=-1.0,
        orientation_rescue_enabled=False,
        orientation_rescue_xy_threshold=0.015,
        orientation_rescue_angle_threshold_deg=8.0,
        commit_switch_xy_threshold=0.012,
        commit_switch_z_threshold=0.012,
        commit_switch_yaw_threshold=-1.0,
        name=f"{task_key}_default",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def infer_task_name(task=None) -> str:
    task_name = "insert_onto_square_peg"
    if task is not None and hasattr(task, "_task"):
        task_cls = getattr(task._task, "__class__", type(task._task)).__name__.lower()
        if "insert" in task_cls:
            task_name = "insert_onto_square_peg"
        elif "stack" in task_cls:
            task_name = "stack_blocks"
        elif "close" in task_cls and "jar" in task_cls:
            task_name = "close_jar"
    return task_name


def _coerce_handoff_spec(task_name: str, raw: dict) -> StageHandoffSpec:
    release_thresholds = raw.get("release_thresholds", raw.get("metric_thresholds", {}))
    if release_thresholds is None:
        release_thresholds = {}
    optimization_thresholds = raw.get("optimization_thresholds", release_thresholds)
    if optimization_thresholds is None:
        optimization_thresholds = dict(release_thresholds)
    return StageHandoffSpec(
        task_name=str(raw.get("task_name", task_name)),
        substage_id=int(raw.get("substage_id", 0)),
        motion_target_mode=str(raw.get("motion_target_mode", "active_target")),
        handoff_target_mode=str(raw.get("handoff_target_mode", "spec_target")),
        target_frame=str(raw.get("target_frame", "active_target")),
        metric_thresholds={str(k): float(v) for k, v in dict(release_thresholds).items()},
        release_metric_thresholds={str(k): float(v) for k, v in dict(release_thresholds).items()},
        optimization_metric_thresholds={str(k): float(v) for k, v in dict(optimization_thresholds).items()},
        target_offset_local_7d=None
        if raw.get("target_offset_local_7d", None) is None
        else np.asarray(raw.get("target_offset_local_7d"), dtype=np.float32).reshape(7),
        yaw_symmetry_period=float(raw.get("yaw_symmetry_period", -1.0)),
        min_stable_frames=max(int(raw.get("min_stable_frames", 1)), 1),
        required_gripper_state=str(raw.get("required_gripper_state", "open")),
        required_contact_state=str(raw.get("required_contact_state", "any")),
        source=str(raw.get("source", "manual_fallback")),
        name=str(raw.get("name", f"{task_name}_substage{raw.get('substage_id', 0)}_handoff")),
        target_role=str(raw.get("target_role", raw.get("handoff_target_role", "pregrasp"))),
        uses_privileged=bool(raw.get("uses_privileged", True)),
    )


def load_stage_handoff_specs(task_name: Optional[str] = None) -> dict[int, StageHandoffSpec]:
    """Load task-stage handoff specs.

    The spec owns task-specific "ready to hand control back to planner" logic.
    Alignment controllers should not hard-code task-stage close gates.
    """

    task_key = str(task_name or "insert_onto_square_peg")
    config_path = _repo_root() / "configs" / "stage_handoff_specs" / f"{task_key}.json"
    specs: dict[int, StageHandoffSpec] = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text())
            stage_rows = raw.get("stages", raw if isinstance(raw, list) else [])
            for row in stage_rows:
                spec = _coerce_handoff_spec(task_key, row)
                specs[int(spec.substage_id)] = spec
        except Exception:
            specs = {}

    # Conservative temporary fallback for current insert phase-1. This preserves
    # the working v6 target口径 while making the source explicit and replaceable
    # by demo-derived specs.
    if not specs and task_key == "insert_onto_square_peg":
        specs[int(1)] = StageHandoffSpec(
            task_name=task_key,
            substage_id=1,
            motion_target_mode="basin_v6",
            handoff_target_mode="active_target",
            target_frame="active_target",
            metric_thresholds={
                "xy_error": 0.006,
                "abs_z_error": 0.003,
                "yaw_error": -1.0,
                "tilt_error": -1.0,
            },
            release_metric_thresholds={
                "xy_error": 0.006,
                "abs_z_error": 0.003,
                "yaw_error": -1.0,
                "tilt_error": -1.0,
            },
            optimization_metric_thresholds={
                "xy_error": 0.006,
                "abs_z_error": 0.003,
                "yaw_error": -1.0,
                "tilt_error": -1.0,
            },
            target_offset_local_7d=None,
            min_stable_frames=1,
            required_gripper_state="open",
            required_contact_state="any",
            source="manual_fallback_insert_phase1",
            name=f"{task_key}_phase1_manual_fallback",
            target_role="pregrasp_close",
            uses_privileged=True,
        )
    return specs


def handoff_metrics_from_delta(delta_local, yaw_symmetry_period: float = -1.0) -> dict[str, float]:
    if delta_local is None:
        return {
            "xy_error": float("inf"),
            "abs_z_error": float("inf"),
            "yaw_error": float("inf"),
            "tilt_error": float("inf"),
        }
    delta = apply_yaw_symmetry_to_delta(delta_local, yaw_symmetry_period).reshape(-1)
    return {
        "xy_error": float(np.linalg.norm(delta[:2])) if delta.size >= 2 else float("inf"),
        "abs_z_error": float(abs(delta[2])) if delta.size >= 3 else float("inf"),
        "yaw_error": float(abs(delta[5])) if delta.size >= 6 else float("inf"),
        "tilt_error": float(np.linalg.norm(delta[3:5])) if delta.size >= 5 else float("inf"),
    }


def evaluate_stage_handoff_ready(
    spec: Optional[StageHandoffSpec],
    delta_local,
    *,
    gripper_open: Optional[float] = None,
    contact_state: Optional[int] = None,
) -> tuple[bool, dict[str, float]]:
    del contact_state
    metrics = handoff_metrics_from_delta(
        delta_local,
        getattr(spec, "yaw_symmetry_period", -1.0) if spec is not None else -1.0,
    )
    if spec is None:
        return False, metrics
    required_gripper = str(spec.required_gripper_state or "any").lower()
    if required_gripper == "open" and gripper_open is not None and float(gripper_open) < 0.5:
        return False, metrics
    if required_gripper == "closed" and gripper_open is not None and float(gripper_open) >= 0.5:
        return False, metrics

    release_thresholds = dict(
        getattr(spec, "release_metric_thresholds", None)
        or getattr(spec, "metric_thresholds", {})
        or {}
    )
    for key, threshold in release_thresholds.items():
        threshold = float(threshold)
        if threshold < 0.0:
            continue
        value = float(metrics.get(str(key), float("inf")))
        if not np.isfinite(value) or value > threshold:
            return False, metrics
    return True, metrics


class StageTargetProvider:
    provider_name = "base"

    def resolve(
        self,
        *,
        current_gripper_pose,
        manager,
        controller=None,
        live_target_handle=None,
        obs=None,
        task=None,
        gripper_context=None,
    ) -> TargetProviderResult:
        raise NotImplementedError

    def resolve_handoff_only(
        self,
        *,
        current_gripper_pose,
        manager,
        controller=None,
        live_target_handle=None,
        obs=None,
        task=None,
        gripper_context=None,
    ) -> Optional[dict]:
        del current_gripper_pose, manager, controller, live_target_handle, obs, task, gripper_context
        return None

    @staticmethod
    def _build_result(
        *,
        provider_name: str,
        source: str,
        uses_privileged_target: bool,
        stage_target_mode: int,
        current_gripper_pose,
        target_pose_7d,
        yaw_symmetry_period: float = -1.0,
    ) -> TargetProviderResult:
        if target_pose_7d is None or current_gripper_pose is None:
            return TargetProviderResult(
                motion_target_pose_7d=None,
                motion_target_delta_local=None,
                target_pose_7d=None,
                target_delta_local=None,
                provider_name=provider_name,
                source=source,
                uses_privileged_target=uses_privileged_target,
                stage_target_mode=int(stage_target_mode),
            )
        target_pose = np.asarray(target_pose_7d, dtype=np.float32).reshape(7)
        current_pose = np.asarray(current_gripper_pose, dtype=np.float32).reshape(7)
        target_delta = apply_yaw_symmetry_to_delta(
            pose_delta_local_between(current_pose, target_pose),
            yaw_symmetry_period,
        )
        return TargetProviderResult(
            motion_target_pose_7d=target_pose,
            motion_target_delta_local=target_delta,
            target_pose_7d=target_pose,
            target_delta_local=target_delta,
            provider_name=provider_name,
            source=source,
            uses_privileged_target=uses_privileged_target,
            stage_target_mode=int(stage_target_mode),
        )


class CanonicalFallbackTargetProvider(StageTargetProvider):
    provider_name = "canonical_fallback_provider"

    def __init__(self):
        self._handoff_specs_by_task: dict[str, dict[int, StageHandoffSpec]] = {}

    def _task_name(self, task=None) -> str:
        return infer_task_name(task)

    def _handoff_spec_for(self, task_name: str, manager) -> Optional[StageHandoffSpec]:
        if task_name not in self._handoff_specs_by_task:
            self._handoff_specs_by_task[task_name] = load_stage_handoff_specs(task_name)
        specs = self._handoff_specs_by_task.get(task_name, {})
        try:
            substage_id = int(getattr(manager, "substage", 0))
        except Exception:
            substage_id = 0
        return specs.get(substage_id) or specs.get(0)

    def resolve(
        self,
        *,
        current_gripper_pose,
        manager,
        controller=None,
        live_target_handle=None,
        obs=None,
        task=None,
        gripper_context=None,
    ) -> TargetProviderResult:
        del live_target_handle, gripper_context
        target_pose = getattr(controller, "_canonical_basin_center_pose_7d", None) if controller is not None else None
        base_result = self._build_result(
            provider_name=self.provider_name,
            source="canonical_fallback",
            uses_privileged_target=False,
            stage_target_mode=int(getattr(manager, "stage_target_mode", StageTargetMode.NONE)),
            current_gripper_pose=current_gripper_pose,
            target_pose_7d=target_pose,
        )
        task_name = self._task_name(task)
        handoff_spec = self._handoff_spec_for(task_name, manager)
        gripper_open = None
        if obs is not None and hasattr(obs, "gripper_open"):
            gripper_open = float(obs.gripper_open)
        handoff_delta_local = (
            None
            if base_result.target_delta_local is None
            else apply_yaw_symmetry_to_delta(
                base_result.target_delta_local,
                -1.0 if handoff_spec is None else float(handoff_spec.yaw_symmetry_period),
            )
        )
        handoff_ready, handoff_metrics = evaluate_stage_handoff_ready(
            handoff_spec,
            handoff_delta_local,
            gripper_open=gripper_open,
            contact_state=int(getattr(manager, "contact_state", 0)),
        )
        result_kwargs = dict(base_result.__dict__)
        result_kwargs.update(
            handoff_ready=bool(handoff_ready),
            handoff_metrics=dict(handoff_metrics),
            handoff_metric_thresholds={}
            if handoff_spec is None
            else dict(getattr(handoff_spec, "release_metric_thresholds", {}) or getattr(handoff_spec, "metric_thresholds", {})),
            handoff_release_metric_thresholds={}
            if handoff_spec is None
            else dict(getattr(handoff_spec, "release_metric_thresholds", {}) or getattr(handoff_spec, "metric_thresholds", {})),
            handoff_optimization_metric_thresholds={}
            if handoff_spec is None
            else dict(getattr(handoff_spec, "optimization_metric_thresholds", {}) or getattr(handoff_spec, "metric_thresholds", {})),
            handoff_target_pose_7d=None if base_result.target_pose_7d is None else np.asarray(base_result.target_pose_7d, dtype=np.float32),
            handoff_spec_name="none" if handoff_spec is None else str(handoff_spec.name),
            handoff_target_role="none" if handoff_spec is None else str(handoff_spec.target_role),
            handoff_uses_privileged=False,
            handoff_min_stable_frames=int(1 if handoff_spec is None else handoff_spec.min_stable_frames),
            yaw_symmetry_period=float(-1.0 if handoff_spec is None else handoff_spec.yaw_symmetry_period),
        )
        return TargetProviderResult(**result_kwargs)


class TeacherOracleTargetProvider(StageTargetProvider):
    provider_name = "teacher_target_provider"

    def __init__(self, fallback: Optional[StageTargetProvider] = None):
        self.fallback = fallback if fallback is not None else CanonicalFallbackTargetProvider()
        self._handoff_specs_by_task: dict[str, dict[int, StageHandoffSpec]] = {}

    def _task_name(self, task=None) -> str:
        return infer_task_name(task)

    def _handoff_spec_for(self, task_name: str, manager) -> Optional[StageHandoffSpec]:
        if task_name not in self._handoff_specs_by_task:
            self._handoff_specs_by_task[task_name] = load_stage_handoff_specs(task_name)
        specs = self._handoff_specs_by_task.get(task_name, {})
        try:
            substage_id = int(getattr(manager, "substage", 0))
        except Exception:
            substage_id = 0
        return specs.get(substage_id) or specs.get(0)

    def resolve(
        self,
        *,
        current_gripper_pose,
        manager,
        controller=None,
        live_target_handle=None,
        obs=None,
        task=None,
        gripper_context=None,
    ) -> TargetProviderResult:
        del gripper_context
        anchor_pose = getattr(controller, "_reference_anchor_pose_7d", None) if controller is not None else None
        if live_target_handle is not None:
            try:
                object_pose = np.asarray(live_target_handle.get_pose(), dtype=np.float32)
                task_name = self._task_name(task)
                grasp_spec = load_phase1_grasp_spec(task_name)
                pregrasp_target, grasp_commit_target = build_phase1_teacher_targets(object_pose, grasp_spec)
                grasp_commit_edge_pair_index = -1
                grasp_commit_edge_pair_family = -1
                grasp_commit_edge_pair_yaw_error = 0.0
                if current_gripper_pose is not None:
                    (
                        grasp_commit_target,
                        grasp_commit_edge_pair_index,
                        grasp_commit_edge_pair_family,
                        grasp_commit_edge_pair_yaw_error,
                    ) = align_square_edge_pair_target_pose(
                        current_gripper_pose,
                        grasp_commit_target,
                        float(getattr(grasp_spec, "yaw_symmetry_period", np.pi / 2.0)),
                    )
                    pregrasp_target = apply_local_offset_to_pose(
                        grasp_commit_target,
                        np.asarray(grasp_spec.pregrasp_hover_offset_local_6d, dtype=np.float32),
                    )
                handoff_spec = self._handoff_spec_for(task_name, manager)
                # Keep the deploy/runtime teacher target on the v6-compatible basin
                # geometry. The object-frame pregrasp/commit targets are still
                # exported for debug/future demo-derived specs, but the current
                # fallback spec is not trustworthy enough to drive close gates.
                handoff_target_pose = None
                if handoff_spec is not None:
                    target_frame = str(handoff_spec.target_frame)
                    if (
                        target_frame in ("object_grasp_target", "object_frame_grasp")
                        and handoff_spec.target_offset_local_7d is not None
                    ):
                        handoff_target_pose = apply_object_frame_offset(object_pose, handoff_spec.target_offset_local_7d)
                    elif target_frame in ("task_success_centre", "success_centre"):
                        handoff_target_pose = resolve_task_success_centre_pose(task, obs=obs)
                # Stable legacy baseline contract:
                # - motion target stays on the v6 basin/object-center target,
                #   matching the scorer that produced the best observed behavior.
                # - handoff/close readiness may still use the stage spec target.
                #
                # Do not silently unify these targets for the legacy scorer; doing
                # so changes the closed-loop objective without retraining.
                if anchor_pose is not None:
                    active_target = build_object_center_target_pose(object_pose, anchor_pose)
                    motion_target_source = "teacher_motion_basin_v6__handoff_stage_spec"
                else:
                    # Student-vNext / bridge repair paths do not always carry the
                    # legacy basin anchor seed. Fall back to the demo-derived
                    # phase1 grasp target so runtime still gets a nonzero basin
                    # delta in ALIGN_PREGRASP + near-contact.
                    active_target = grasp_commit_target
                    motion_target_source = "teacher_motion_demo_grasp_commit_fallback__handoff_stage_spec"
                if grasp_commit_edge_pair_index >= 0:
                    motion_target_source = (
                        f"{motion_target_source}__edge_pair_q{int(grasp_commit_edge_pair_index)}"
                        f"_f{int(grasp_commit_edge_pair_family)}"
                    )
                use_commit_target = False
                base_result = self._build_result(
                    provider_name=self.provider_name,
                    source=str(motion_target_source),
                    uses_privileged_target=True,
                    stage_target_mode=int(getattr(manager, "stage_target_mode", StageTargetMode.NONE)),
                    current_gripper_pose=current_gripper_pose,
                    target_pose_7d=active_target,
                    yaw_symmetry_period=-1.0 if handoff_spec is None else float(handoff_spec.yaw_symmetry_period),
                )
                gripper_open = None
                if obs is not None and hasattr(obs, "gripper_open"):
                    gripper_open = float(obs.gripper_open)
                handoff_delta_local = (
                    base_result.target_delta_local
                    if handoff_target_pose is None
                    else apply_yaw_symmetry_to_delta(
                        pose_delta_local_between(current_gripper_pose, handoff_target_pose),
                        -1.0 if handoff_spec is None else float(handoff_spec.yaw_symmetry_period),
                    )
                )
                handoff_ready, handoff_metrics = evaluate_stage_handoff_ready(
                    handoff_spec,
                    handoff_delta_local,
                    gripper_open=gripper_open,
                    contact_state=int(getattr(manager, "contact_state", 0)),
                )
                motion_target_pose = active_target
                motion_target_source = "teacher_motion_basin_v6__handoff_stage_spec"
                if handoff_ready and handoff_target_pose is not None:
                    motion_target_pose = np.asarray(handoff_target_pose, dtype=np.float32)
                    motion_target_source = "teacher_motion_privileged_handoff_target__close_ready"
                edge_suffix = ""
                if grasp_commit_edge_pair_index >= 0:
                    edge_suffix = f"__edge_pair_q{grasp_commit_edge_pair_index}_f{grasp_commit_edge_pair_family}"
                result_kwargs = dict(base_result.__dict__)
                result_kwargs.update(
                    motion_target_pose_7d=np.asarray(motion_target_pose, dtype=np.float32),
                    motion_target_delta_local=np.asarray(
                        apply_yaw_symmetry_to_delta(
                            pose_delta_local_between(current_gripper_pose, motion_target_pose),
                            -1.0 if handoff_spec is None else float(handoff_spec.yaw_symmetry_period),
                        ),
                        dtype=np.float32,
                    ),
                    target_pose_7d=np.asarray(motion_target_pose, dtype=np.float32),
                    target_delta_local=np.asarray(
                        apply_yaw_symmetry_to_delta(
                            pose_delta_local_between(current_gripper_pose, motion_target_pose),
                            -1.0 if handoff_spec is None else float(handoff_spec.yaw_symmetry_period),
                        ),
                        dtype=np.float32,
                    ),
                    source=str(motion_target_source) + edge_suffix,
                    handoff_ready=bool(handoff_ready),
                    handoff_metrics=dict(handoff_metrics),
                    handoff_metric_thresholds={}
                    if handoff_spec is None
                    else dict(getattr(handoff_spec, "release_metric_thresholds", {}) or getattr(handoff_spec, "metric_thresholds", {})),
                    handoff_release_metric_thresholds={}
                    if handoff_spec is None
                    else dict(getattr(handoff_spec, "release_metric_thresholds", {}) or getattr(handoff_spec, "metric_thresholds", {})),
                    handoff_optimization_metric_thresholds={}
                    if handoff_spec is None
                    else dict(getattr(handoff_spec, "optimization_metric_thresholds", {}) or getattr(handoff_spec, "metric_thresholds", {})),
                    handoff_target_pose_7d=None
                    if handoff_target_pose is None
                    else np.asarray(handoff_target_pose, dtype=np.float32),
                    handoff_spec_name="none" if handoff_spec is None else str(handoff_spec.name),
                    handoff_target_role="none" if handoff_spec is None else str(handoff_spec.target_role),
                    handoff_uses_privileged=bool(False if handoff_spec is None else handoff_spec.uses_privileged),
                    handoff_min_stable_frames=int(1 if handoff_spec is None else handoff_spec.min_stable_frames),
                    yaw_symmetry_period=float(-1.0 if handoff_spec is None else handoff_spec.yaw_symmetry_period),
                )
                return TeacherPhase1TargetResult(
                    **result_kwargs,
                    pregrasp_target_pose_7d=pregrasp_target,
                    grasp_commit_target_pose_7d=grasp_commit_target,
                    grasp_spec_name=f"{grasp_spec.name}:basin_v6",
                    grasp_commit_edge_pair_index=int(grasp_commit_edge_pair_index),
                    grasp_commit_edge_pair_family=int(grasp_commit_edge_pair_family),
                    grasp_commit_edge_pair_yaw_error=float(grasp_commit_edge_pair_yaw_error),
                    use_commit_target=bool(use_commit_target),
                    orientation_rescue_enabled=bool(getattr(grasp_spec, "orientation_rescue_enabled", False)),
                    close_xy_threshold=float(grasp_spec.close_xy_threshold),
                    close_abs_z_threshold=float(grasp_spec.close_abs_z_threshold),
                    close_yaw_threshold=float(grasp_spec.close_yaw_threshold),
                    commit_switch_xy_threshold=float(grasp_spec.commit_switch_xy_threshold),
                    commit_switch_z_threshold=float(grasp_spec.commit_switch_z_threshold),
                    commit_switch_yaw_threshold=float(grasp_spec.commit_switch_yaw_threshold),
                )
            except Exception:
                pass
        return self.fallback.resolve(
            current_gripper_pose=current_gripper_pose,
            manager=manager,
            controller=controller,
            live_target_handle=live_target_handle,
            obs=obs,
            task=task,
        )

    def resolve_handoff_only(
        self,
        *,
        current_gripper_pose,
        manager,
        controller=None,
        live_target_handle=None,
        obs=None,
        task=None,
        gripper_context=None,
    ) -> Optional[dict]:
        del gripper_context
        anchor_pose = getattr(controller, "_reference_anchor_pose_7d", None) if controller is not None else None
        if live_target_handle is None or anchor_pose is None or current_gripper_pose is None:
            return None
        try:
            object_pose = np.asarray(live_target_handle.get_pose(), dtype=np.float32)
            task_name = self._task_name(task)
            handoff_spec = self._handoff_spec_for(task_name, manager)
            if handoff_spec is None:
                return None
            if (
                str(handoff_spec.target_frame) in ("object_grasp_target", "object_frame_grasp")
                and handoff_spec.target_offset_local_7d is not None
            ):
                handoff_target_pose = apply_object_frame_offset(object_pose, handoff_spec.target_offset_local_7d)
                handoff_delta_local = apply_yaw_symmetry_to_delta(
                    pose_delta_local_between(current_gripper_pose, handoff_target_pose),
                    float(handoff_spec.yaw_symmetry_period),
                )
            else:
                basin_target = build_object_center_target_pose(object_pose, anchor_pose)
                handoff_delta_local = apply_yaw_symmetry_to_delta(
                    pose_delta_local_between(current_gripper_pose, basin_target),
                    float(handoff_spec.yaw_symmetry_period),
                )
            gripper_open = None
            if obs is not None and hasattr(obs, "gripper_open"):
                gripper_open = float(obs.gripper_open)
            handoff_ready, handoff_metrics = evaluate_stage_handoff_ready(
                handoff_spec,
                handoff_delta_local,
                gripper_open=gripper_open,
                contact_state=int(getattr(manager, "contact_state", 0)),
            )
            release_thresholds = dict(
                getattr(handoff_spec, "release_metric_thresholds", None)
                or getattr(handoff_spec, "metric_thresholds", {})
                or {}
            )
            optimization_thresholds = dict(
                getattr(handoff_spec, "optimization_metric_thresholds", {})
                or getattr(handoff_spec, "metric_thresholds", {})
                or {}
            )
            return {
                "handoff_ready": bool(handoff_ready),
                "handoff_metrics": dict(handoff_metrics),
                "handoff_metric_thresholds": dict(release_thresholds),
                "handoff_release_metric_thresholds": dict(release_thresholds),
                "handoff_optimization_metric_thresholds": dict(optimization_thresholds),
                "handoff_spec_name": str(handoff_spec.name),
                "handoff_target_role": str(handoff_spec.target_role),
                "handoff_uses_privileged": bool(handoff_spec.uses_privileged),
                "handoff_min_stable_frames": int(handoff_spec.min_stable_frames),
                "yaw_symmetry_period": float(handoff_spec.yaw_symmetry_period),
            }
        except Exception:
            return None


class LearnedTargetProvider(StageTargetProvider):
    provider_name = "learned_target_provider"

    def __init__(
        self,
        fallback: Optional[StageTargetProvider] = None,
        ckpt_path: Optional[str] = None,
        handoff_ckpt_path: Optional[str] = None,
    ):
        self.fallback = fallback if fallback is not None else CanonicalFallbackTargetProvider()
        self.ckpt_path = ckpt_path
        self.handoff_ckpt_path = handoff_ckpt_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.handoff_model = None
        self.handoff_model_kind: Optional[str] = None
        self._handoff_specs_by_task: dict[str, dict[int, StageHandoffSpec]] = {}
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model_state = ckpt["model_state_dict"]
            is_legacy_motion_ckpt = "delta_head.weight" not in model_state and "head.4.weight" in model_state
            self.model = TargetDeltaPredictor(legacy_output_head=is_legacy_motion_ckpt)
            self.model.load_state_dict(model_state, strict=False)
            self.model.to(self.device).eval()
        if handoff_ckpt_path:
            handoff_ckpt = torch.load(handoff_ckpt_path, map_location="cpu")
            handoff_state = handoff_ckpt["model_state_dict"]
            if any(k.startswith("uncertainty_head.") or k.startswith("band_head.") for k in handoff_state.keys()):
                self.handoff_model = StudentHandoffStateHeadV2()
                self.handoff_model_kind = "student_handoff_state_v2"
            else:
                rgb_weight = handoff_state.get("front_rgb_encoder.fc.weight", None)
                rgb_out_dim = int(rgb_weight.shape[0]) if rgb_weight is not None else None
                if rgb_out_dim is not None and rgb_out_dim <= 96:
                    self.handoff_model = NearReadyXYYawPredictor()
                    self.handoff_model_kind = "near_ready_xyyaw"
                else:
                    self.handoff_model = HandoffPredictor()
                    self.handoff_model_kind = "handoff_predictor"
            self.handoff_model.load_state_dict(handoff_ckpt["model_state_dict"], strict=False)
            self.handoff_model.to(self.device).eval()

    @staticmethod
    def _normalize_gripper_context(gripper_context) -> np.ndarray:
        if gripper_context is None:
            return np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
        arr = np.asarray(gripper_context, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
        if arr.size == 1:
            return np.asarray([arr[0], arr[0], 0.0], dtype=np.float32)
        if arr.size == 2:
            return np.asarray([arr[0], arr[1], 0.0], dtype=np.float32)
        return np.asarray(arr[:3], dtype=np.float32)

    def _task_name(self, task=None) -> str:
        return infer_task_name(task)

    def _handoff_spec_for(self, task_name: str, manager) -> Optional[StageHandoffSpec]:
        if task_name not in self._handoff_specs_by_task:
            self._handoff_specs_by_task[task_name] = load_stage_handoff_specs(task_name)
        specs = self._handoff_specs_by_task.get(task_name, {})
        try:
            substage_id = int(getattr(manager, "substage", 0))
        except Exception:
            substage_id = 0
        return specs.get(substage_id) or specs.get(0)

    @staticmethod
    def _rgb_tensor(rgb: np.ndarray) -> torch.Tensor:
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)
        ten = torch.from_numpy(arr).float().unsqueeze(0)
        if ten.max().item() > 1.5:
            ten = ten / 255.0
        return ten

    @staticmethod
    def _depth_tensor(depth: np.ndarray, depth_max: float = 1.0) -> torch.Tensor:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.clip(arr / max(depth_max, 1e-6), 0.0, 1.0)
        ten = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        return ten

    def resolve(
        self,
        *,
        current_gripper_pose,
        manager,
        controller=None,
        live_target_handle=None,
        obs=None,
        task=None,
        gripper_context=None,
    ) -> TargetProviderResult:
        del live_target_handle
        fallback_result = self.fallback.resolve(
            current_gripper_pose=current_gripper_pose,
            manager=manager,
            controller=controller,
            live_target_handle=None,
            obs=obs,
            task=task,
            gripper_context=gripper_context,
        )
        if self.model is not None and obs is not None:
            with torch.no_grad():
                front_rgb = self._rgb_tensor(obs.front_rgb).to(self.device)
                wrist_rgb = self._rgb_tensor(obs.wrist_rgb).to(self.device)
                wrist_depth = self._depth_tensor(obs.wrist_depth).to(self.device)
                proprio = torch.from_numpy(
                    np.concatenate([obs.joint_positions, obs.gripper_pose, [float(obs.gripper_open)]], axis=0).astype(np.float32)
                ).unsqueeze(0).to(self.device)
                depth_valid = np.asarray(obs.wrist_depth, dtype=np.float32)
                depth_valid = depth_valid[np.isfinite(depth_valid)]
                depth_prox = float(np.percentile(depth_valid, 5.0)) if depth_valid.size > 0 else np.nan
                if gripper_context is None:
                    grip_ctx = np.asarray([float(obs.gripper_open), float(obs.gripper_open), depth_prox], dtype=np.float32)
                else:
                    grip_ctx = self._normalize_gripper_context(gripper_context)
                gripper_context_t = torch.from_numpy(grip_ctx).unsqueeze(0).to(self.device, dtype=torch.float32)
                pred_delta = self.model(
                    front_rgb=front_rgb,
                    wrist_rgb=wrist_rgb,
                    wrist_depth=wrist_depth,
                    proprio=proprio,
                    gripper_context=gripper_context_t,
                    has_object_in_hand=torch.tensor([float(manager.has_object_in_hand)], dtype=torch.float32, device=self.device),
                    substage_id=torch.tensor([int(manager.substage)], dtype=torch.long, device=self.device),
                    contact_state=torch.tensor([int(manager.contact_state)], dtype=torch.long, device=self.device),
                    stage_target_mode=torch.tensor([int(manager.stage_target_mode)], dtype=torch.long, device=self.device),
                )
            pred_delta = np.asarray(pred_delta.squeeze(0).float().cpu().numpy(), dtype=np.float32)
            target_pose = None
            motion_delta = np.asarray(pred_delta, dtype=np.float32)
            handoff_target_pose = getattr(fallback_result, "handoff_target_pose_7d", None)
            fallback_motion_pose = (
                getattr(fallback_result, "motion_target_pose_7d", None)
                if getattr(fallback_result, "motion_target_pose_7d", None) is not None
                else getattr(fallback_result, "target_pose_7d", None)
            )
            handoff_role = str(getattr(fallback_result, "handoff_target_role", "none"))
            disable_close_stage_orientation_contract = bool(
                getattr(controller, "_disable_learned_target_close_stage_orientation_contract", False)
            )
            use_close_stage_orientation_contract = bool(
                current_gripper_pose is not None
                and fallback_motion_pose is not None
                and handoff_role in ("pregrasp_close", "pregrasp", "close", "commit_close")
                and not disable_close_stage_orientation_contract
            )
            if current_gripper_pose is not None:
                try:
                    if use_close_stage_orientation_contract:
                        target_pose = compose_pose_with_orientation_contract(
                            current_gripper_pose,
                            pred_delta,
                            fallback_motion_pose,
                        )
                        motion_delta = apply_yaw_symmetry_to_delta(
                            pose_delta_local_between(current_gripper_pose, target_pose),
                            float(getattr(fallback_result, "yaw_symmetry_period", -1.0)),
                        )
                    else:
                        target_pose = apply_local_offset_to_pose(current_gripper_pose, pred_delta)
                except Exception:
                    target_pose = None
            return TargetProviderResult(
                motion_target_pose_7d=target_pose,
                motion_target_delta_local=np.asarray(motion_delta, dtype=np.float32),
                target_pose_7d=target_pose,
                target_delta_local=np.asarray(motion_delta, dtype=np.float32),
                provider_name=self.provider_name,
                source=(
                    "learned_target_predictor__canonical_close_orientation_contract"
                    if use_close_stage_orientation_contract
                    else "learned_target_predictor"
                ),
                uses_privileged_target=False,
                stage_target_mode=int(getattr(manager, "stage_target_mode", StageTargetMode.NONE)),
                handoff_ready=bool(getattr(fallback_result, "handoff_ready", False)),
                handoff_metrics=dict(getattr(fallback_result, "handoff_metrics", {}) or {}),
                handoff_metric_thresholds=dict(getattr(fallback_result, "handoff_metric_thresholds", {}) or {}),
                handoff_release_metric_thresholds=dict(getattr(fallback_result, "handoff_release_metric_thresholds", {}) or {}),
                handoff_optimization_metric_thresholds=dict(getattr(fallback_result, "handoff_optimization_metric_thresholds", {}) or {}),
                handoff_target_pose_7d=None if handoff_target_pose is None else np.asarray(handoff_target_pose, dtype=np.float32),
                handoff_spec_name=str(getattr(fallback_result, "handoff_spec_name", "none")),
                handoff_target_role=handoff_role,
                handoff_uses_privileged=False,
                handoff_min_stable_frames=int(getattr(fallback_result, "handoff_min_stable_frames", 1)),
                yaw_symmetry_period=float(getattr(fallback_result, "yaw_symmetry_period", -1.0)),
            )
        result = fallback_result
        return TargetProviderResult(
            motion_target_pose_7d=result.motion_target_pose_7d if hasattr(result, "motion_target_pose_7d") else result.target_pose_7d,
            motion_target_delta_local=result.motion_target_delta_local if hasattr(result, "motion_target_delta_local") else result.target_delta_local,
            target_pose_7d=result.target_pose_7d,
            target_delta_local=result.target_delta_local,
            provider_name=self.provider_name,
            source="learned_target_unavailable__canonical_fallback",
            uses_privileged_target=False,
            stage_target_mode=int(getattr(manager, "stage_target_mode", StageTargetMode.NONE)),
            handoff_ready=bool(getattr(result, "handoff_ready", False)),
            handoff_metrics=dict(getattr(result, "handoff_metrics", {}) or {}),
            handoff_metric_thresholds=dict(getattr(result, "handoff_metric_thresholds", {}) or {}),
            handoff_release_metric_thresholds=dict(getattr(result, "handoff_release_metric_thresholds", {}) or {}),
            handoff_optimization_metric_thresholds=dict(getattr(result, "handoff_optimization_metric_thresholds", {}) or {}),
            handoff_target_pose_7d=None
            if getattr(result, "handoff_target_pose_7d", None) is None
            else np.asarray(getattr(result, "handoff_target_pose_7d"), dtype=np.float32),
            handoff_spec_name=str(getattr(result, "handoff_spec_name", "none")),
            handoff_target_role=str(getattr(result, "handoff_target_role", "none")),
            handoff_uses_privileged=False,
            handoff_min_stable_frames=int(getattr(result, "handoff_min_stable_frames", 1)),
            yaw_symmetry_period=float(getattr(result, "yaw_symmetry_period", -1.0)),
        )

    def resolve_handoff_only(
        self,
        *,
        current_gripper_pose,
        manager,
        controller=None,
        live_target_handle=None,
        obs=None,
        task=None,
        gripper_context=None,
    ) -> Optional[dict]:
        del current_gripper_pose, live_target_handle
        task_name = self._task_name(task)
        handoff_spec = self._handoff_spec_for(task_name, manager)
        if self.handoff_model is None or obs is None or handoff_spec is None:
            return None
        release_thresholds = dict(
            getattr(handoff_spec, "release_metric_thresholds", None)
            or getattr(handoff_spec, "metric_thresholds", {})
            or {}
        )
        with torch.no_grad():
            front_rgb = self._rgb_tensor(obs.front_rgb).to(self.device)
            wrist_rgb = self._rgb_tensor(obs.wrist_rgb).to(self.device)
            wrist_depth = self._depth_tensor(obs.wrist_depth).to(self.device)
            proprio = torch.from_numpy(
                np.concatenate([obs.joint_positions, obs.gripper_pose, [float(obs.gripper_open)]], axis=0).astype(np.float32)
            ).unsqueeze(0).to(self.device)
            depth_valid = np.asarray(obs.wrist_depth, dtype=np.float32)
            depth_valid = depth_valid[np.isfinite(depth_valid)]
            depth_prox = float(np.percentile(depth_valid, 5.0)) if depth_valid.size > 0 else np.nan
            if gripper_context is None:
                grip_ctx = np.asarray([float(obs.gripper_open), float(obs.gripper_open), depth_prox], dtype=np.float32)
            else:
                grip_ctx = self._normalize_gripper_context(gripper_context)
            gripper_context_t = torch.from_numpy(grip_ctx).unsqueeze(0).to(self.device, dtype=torch.float32)
            runtime_delta = np.asarray(
                getattr(controller, "_runtime_current_delta_basin_target", np.zeros(6, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            dx_sign = int(np.sign(runtime_delta[0])) if runtime_delta.size >= 1 and np.isfinite(runtime_delta[0]) else 0
            dy_sign = int(np.sign(runtime_delta[1])) if runtime_delta.size >= 2 and np.isfinite(runtime_delta[1]) else 0
            dyaw_sign = int(np.sign(runtime_delta[5])) if runtime_delta.size >= 6 and np.isfinite(runtime_delta[5]) else 0
            runtime_basin_distance = float(getattr(controller, "_runtime_current_basin_distance", np.inf))
            if not np.isfinite(runtime_basin_distance):
                runtime_basin_distance = float(np.linalg.norm(runtime_delta[:2])) if runtime_delta.size >= 2 else np.inf
            basin_bin = int(basin_distance_bin(float(runtime_basin_distance)))
            if self.handoff_model_kind == "student_handoff_state_v2":
                outputs = self.handoff_model(
                    front_rgb=front_rgb,
                    wrist_rgb=wrist_rgb,
                    wrist_depth=wrist_depth,
                    proprio=proprio,
                    gripper_context=gripper_context_t,
                    proxy_current_delta_basin_target=torch.from_numpy(runtime_delta.astype(np.float32)).unsqueeze(0).to(self.device),
                    current_dx_sign=torch.tensor([dx_sign], dtype=torch.long, device=self.device),
                    current_dy_sign=torch.tensor([dy_sign], dtype=torch.long, device=self.device),
                    current_dyaw_sign=torch.tensor([dyaw_sign], dtype=torch.long, device=self.device),
                    basin_distance_bin=torch.tensor([basin_bin], dtype=torch.long, device=self.device),
                    substage_id=torch.tensor([int(manager.substage)], dtype=torch.long, device=self.device),
                    contact_state=torch.tensor([int(manager.contact_state)], dtype=torch.long, device=self.device),
                    stage_target_mode=torch.tensor([int(manager.stage_target_mode)], dtype=torch.long, device=self.device),
                )
                pred_metrics_norm = np.asarray(outputs["pred_metrics_norm"].squeeze(0).float().cpu().numpy(), dtype=np.float32)
                pred_band = int(outputs["band_logits"].argmax(dim=-1).reshape(-1)[0].detach().cpu().item())
                pred_ready_prob = float(torch.sigmoid(outputs["ready_logit"].reshape(-1)[0]).detach().cpu().item())
                pred_uncertainty = float(outputs["uncertainty"].reshape(-1)[0].detach().cpu().item())
                residual_delta_local = np.asarray(
                    outputs.get("residual_delta_local", torch.zeros((1, 4), device=self.device))
                    .reshape(1, -1)[0, :4]
                    .float()
                    .cpu()
                    .numpy(),
                    dtype=np.float32,
                )
                residual_confidence = float(
                    torch.sigmoid(
                        outputs.get("residual_confidence_logit", torch.zeros((1,), device=self.device)).reshape(-1)[0]
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                handoff_metrics = {
                    "xy_error": float(pred_metrics_norm[0] * max(float(release_thresholds.get("xy_error", 0.0085)), 1e-6)),
                    "abs_z_error": float(pred_metrics_norm[1] * max(float(release_thresholds.get("abs_z_error", 0.0035)), 1e-6)),
                    "yaw_error": float(pred_metrics_norm[2] * max(float(release_thresholds.get("yaw_error", 0.1243404)), 1e-6)),
                    "tilt_error": float(np.linalg.norm(runtime_delta[3:5])) if runtime_delta.size >= 5 else float("nan"),
                }
                metric_valid = bool(
                    np.all(np.isfinite(pred_metrics_norm[:3]))
                    and np.all(np.isfinite([
                        handoff_metrics["xy_error"],
                        handoff_metrics["abs_z_error"],
                        handoff_metrics["yaw_error"],
                    ]))
                    and np.isfinite(pred_ready_prob)
                    and np.isfinite(pred_uncertainty)
                )
                handoff_ready = True
                required_gripper = str(handoff_spec.required_gripper_state or "any").lower()
                if required_gripper == "open" and hasattr(obs, "gripper_open") and float(obs.gripper_open) < 0.5:
                    handoff_ready = False
                if required_gripper == "closed" and hasattr(obs, "gripper_open") and float(obs.gripper_open) >= 0.5:
                    handoff_ready = False
                handoff_ready = bool(
                    handoff_ready
                    and pred_ready_prob >= 0.5
                    and pred_uncertainty <= 0.75
                    and pred_metrics_norm[0] <= 1.0
                    and pred_metrics_norm[1] <= 1.0
                    and pred_metrics_norm[2] <= 1.0
                )
                return {
                    "handoff_ready": bool(handoff_ready),
                    "handoff_metrics": dict(handoff_metrics),
                    "handoff_metric_thresholds": dict(release_thresholds),
                    "handoff_release_metric_thresholds": dict(release_thresholds),
                    "handoff_optimization_metric_thresholds": dict(
                        getattr(handoff_spec, "optimization_metric_thresholds", {}) or getattr(handoff_spec, "metric_thresholds", {})
                    ),
                    "handoff_spec_name": str(handoff_spec.name),
                    "handoff_target_role": str(handoff_spec.target_role),
                    "handoff_uses_privileged": False,
                    "handoff_min_stable_frames": int(handoff_spec.min_stable_frames),
                    "yaw_symmetry_period": float(handoff_spec.yaw_symmetry_period),
                    "handoff_aux": {
                        "pred_xy_norm": float(pred_metrics_norm[0]),
                        "pred_abs_z_norm": float(pred_metrics_norm[1]),
                        "pred_yaw_norm": float(pred_metrics_norm[2]),
                        "pred_band_index": int(pred_band),
                        "pred_ready_prob": float(pred_ready_prob),
                        "pred_uncertainty": float(pred_uncertainty),
                        "pred_residual_dx": float(residual_delta_local[0]),
                        "pred_residual_dy": float(residual_delta_local[1]),
                        "pred_residual_dz": float(residual_delta_local[2]),
                        "pred_residual_dyaw": float(residual_delta_local[3]),
                        "pred_residual_confidence": float(residual_confidence),
                        "metric_valid": bool(metric_valid),
                    },
                    "handoff_metric_valid": bool(metric_valid),
                }
            if self.handoff_model_kind == "near_ready_xyyaw":
                xy_thr = max(float(release_thresholds.get("xy_error", 0.0085)), 1e-6)
                yaw_thr = max(float(release_thresholds.get("yaw_error", 0.1243404)), 1e-6)
                runtime_xyyaw_norm = torch.tensor(
                    [[float(np.linalg.norm(runtime_delta[:2])) / xy_thr, float(abs(runtime_delta[5])) / yaw_thr]],
                    dtype=torch.float32,
                    device=self.device,
                )
                near_ready_outputs = self.handoff_model(
                    front_rgb=front_rgb,
                    wrist_rgb=wrist_rgb,
                    wrist_depth=wrist_depth,
                    proprio=proprio,
                    gripper_context=gripper_context_t,
                    runtime_xyyaw_norm=runtime_xyyaw_norm,
                    substage_id=torch.tensor([int(manager.substage)], dtype=torch.long, device=self.device),
                    contact_state=torch.tensor([int(manager.contact_state)], dtype=torch.long, device=self.device),
                    stage_target_mode=torch.tensor([int(manager.stage_target_mode)], dtype=torch.long, device=self.device),
                )
                pred_xyyaw_norm = np.asarray(
                    near_ready_outputs["xyyaw_norm"].squeeze(0).float().cpu().numpy(),
                    dtype=np.float32,
                )
                pred_handoff_ready_prob = float(
                    torch.sigmoid(near_ready_outputs["ready_logit"].reshape(-1)[0]).detach().cpu().item()
                )
                handoff_metrics = {
                    "xy_error": float(pred_xyyaw_norm[0] * xy_thr),
                    "abs_z_error": float(abs(runtime_delta[2])) if runtime_delta.size >= 3 else float("inf"),
                    "yaw_error": float(pred_xyyaw_norm[1] * yaw_thr),
                    "tilt_error": float(np.linalg.norm(runtime_delta[3:5])) if runtime_delta.size >= 5 else float("nan"),
                }
            else:
                handoff_outputs = self.handoff_model(
                    front_rgb=front_rgb,
                    wrist_rgb=wrist_rgb,
                    wrist_depth=wrist_depth,
                    proprio=proprio,
                    gripper_context=gripper_context_t,
                    has_object_in_hand=torch.tensor([float(manager.has_object_in_hand)], dtype=torch.float32, device=self.device),
                    substage_id=torch.tensor([int(manager.substage)], dtype=torch.long, device=self.device),
                    contact_state=torch.tensor([int(manager.contact_state)], dtype=torch.long, device=self.device),
                    stage_target_mode=torch.tensor([int(manager.stage_target_mode)], dtype=torch.long, device=self.device),
                )
                pred_handoff_metrics = np.asarray(
                    handoff_outputs["handoff_metrics"].squeeze(0).float().cpu().numpy(),
                    dtype=np.float32,
                )
                pred_handoff_ready_prob = float(
                    torch.sigmoid(handoff_outputs["handoff_ready_logit"].reshape(-1)[0]).detach().cpu().item()
                )
                handoff_metrics = {
                    "xy_error": float(pred_handoff_metrics[0]),
                    "abs_z_error": float(pred_handoff_metrics[1]),
                    "yaw_error": float(pred_handoff_metrics[2]),
                    "tilt_error": float(pred_handoff_metrics[3]) if pred_handoff_metrics.size >= 4 else float("nan"),
                }
        handoff_ready = True
        required_gripper = str(handoff_spec.required_gripper_state or "any").lower()
        if required_gripper == "open" and hasattr(obs, "gripper_open") and float(obs.gripper_open) < 0.5:
            handoff_ready = False
        if required_gripper == "closed" and hasattr(obs, "gripper_open") and float(obs.gripper_open) >= 0.5:
            handoff_ready = False
        for key, threshold in release_thresholds.items():
            threshold = float(threshold)
            if threshold < 0.0:
                continue
            value = float(handoff_metrics.get(str(key), float("inf")))
            if not np.isfinite(value) or value > threshold:
                handoff_ready = False
                break
        handoff_ready = bool(handoff_ready and (pred_handoff_ready_prob >= 0.5))
        return {
            "handoff_ready": bool(handoff_ready),
            "handoff_metrics": dict(handoff_metrics),
            "handoff_metric_thresholds": dict(release_thresholds),
            "handoff_release_metric_thresholds": dict(release_thresholds),
            "handoff_optimization_metric_thresholds": dict(
                getattr(handoff_spec, "optimization_metric_thresholds", {}) or getattr(handoff_spec, "metric_thresholds", {})
            ),
            "handoff_spec_name": str(handoff_spec.name),
            "handoff_target_role": str(handoff_spec.target_role),
            "handoff_uses_privileged": False,
            "handoff_min_stable_frames": int(handoff_spec.min_stable_frames),
            "yaw_symmetry_period": float(handoff_spec.yaw_symmetry_period),
            "handoff_aux": {},
        }


def build_stage_target_provider(
    mode: str,
    ckpt_path: Optional[str] = None,
    handoff_ckpt_path: Optional[str] = None,
) -> StageTargetProvider:
    mode = str(mode or "legacy_auto").strip().lower()
    if mode in ("teacher_oracle", "oracle_target_upper_bound", "legacy_auto"):
        return TeacherOracleTargetProvider()
    if mode in ("learned", "learned_target_mainline"):
        return LearnedTargetProvider(ckpt_path=ckpt_path, handoff_ckpt_path=handoff_ckpt_path)
    if mode in ("canonical", "canonical_fallback"):
        return CanonicalFallbackTargetProvider()
    raise ValueError(f"Unknown target provider mode: {mode}")
