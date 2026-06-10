#!/usr/bin/env python3
"""Build candidate command-sweep specs for v46 task-frame data collection.

This script intentionally does not create transition labels for unexecuted
commands. It emits runtime-visible rows that describe which bounded local
commands should be executed by a later evaluator run. True pre/post residuals
must be attached only after those commands are actually executed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PRIVILEGED_TRACE_KEYS = {
    "grasp_probe_pre_true_error",
    "grasp_probe_pre_true_error_t",
    "grasp_probe_post_true_error",
    "grasp_probe_post_true_error_t",
    "privileged_frame_pack",
    "teacher_target_pose",
    "target_pose",
    "success_pose",
    "rlbench_object_handle",
    "rlbench_mask",
    "gt_mask",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_trace_files(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        if item.is_file() and item.name.endswith(".jsonl"):
            out.append(item)
        elif item.is_dir():
            out.extend(sorted(item.rglob("*_gripper_trace.jsonl")))
    return sorted(set(out))


def _source_root_from_trace(path: Path) -> Path:
    if path.parent.name == "gripper_traces":
        return path.parent.parent
    for parent in path.parents:
        if (parent / "runtime_observations").is_dir():
            return parent
    return path.parent


def _episode_from_trace(path: Path, row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("episode_idx"))
    except Exception:
        pass
    match = re.search(r"ep(\d+)_gripper_trace", path.name)
    if match:
        return int(match.group(1))
    return -1


def _runtime_obs_path(source_root: Path, episode_idx: int) -> Path | None:
    candidates = [
        source_root / "runtime_observations" / f"ep{episode_idx:03d}_runtime_obs.npz",
        source_root / "runtime_observations" / f"ep{episode_idx}_runtime_obs.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    obs_dir = source_root / "runtime_observations"
    if not obs_dir.is_dir():
        return None
    matches = sorted(obs_dir.glob(f"*{episode_idx:03d}*runtime_obs*.npz"))
    return matches[0] if matches else None


def _runtime_obs_path_from_row(row: Mapping[str, Any]) -> Path | None:
    obs_pointer = row.get("obs_pointer", {})
    obs_pointer = obs_pointer if isinstance(obs_pointer, Mapping) else {}
    for key in ("runtime_obs_path", "npz_path", "runtime_observation_npz", "source_runtime_obs_path"):
        value = row.get(key, obs_pointer.get(key, ""))
        if value:
            path = Path(str(value))
            if path.exists():
                return path
    value = obs_pointer.get("runtime_obs_path", "")
    if value:
        path = Path(str(value))
        if path.exists():
            return path
    return None


def _vec(value: Any, length: int) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < int(length) or not np.all(np.isfinite(arr[: int(length)])):
        return None
    return arr[: int(length)].astype(np.float32)


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _runtime_near_ready(row: Mapping[str, Any], *, near_field_threshold: float) -> bool:
    if bool(row.get("task_frame_v46_near_field_activation_ready", False)):
        return True
    if bool(row.get("task_frame_v46_activation_ready", False)):
        return True
    near_conf = _as_float(row.get("task_frame_v46_near_field_confidence"), float("nan"))
    radius_ready = bool(row.get("task_frame_v46_radius_ready", False))
    if np.isfinite(near_conf) and radius_ready and near_conf >= float(near_field_threshold):
        return True
    return False


def _offline_residual_band_ready(
    row: Mapping[str, Any],
    *,
    min_xy_error: float,
    max_xy_error: float,
    min_abs_z: float,
    max_abs_z: float,
    max_abs_yaw: float,
) -> bool:
    residual = _vec(row.get("grasp_probe_pre_true_error_t"), 4)
    if residual is None:
        return False
    xy = float(np.linalg.norm(residual[:2]))
    z = abs(float(residual[2]))
    yaw = abs(float(residual[3]))
    if xy < float(min_xy_error) or xy > float(max_xy_error):
        return False
    if z < float(min_abs_z) or z > float(max_abs_z):
        return False
    if yaw > float(max_abs_yaw):
        return False
    return True


def _yaw_observable_symmetry_ready(
    row: Mapping[str, Any],
    *,
    min_yaw_confidence: float,
    min_yaw_hypothesis_gap: float,
    require_yaw_alias_stable: bool,
) -> bool:
    if not _as_bool(row.get("task_frame_v46_yaw_observable"), False):
        return False
    if _as_bool(row.get("task_frame_v46_yaw_ambiguous"), False):
        return False
    if _as_bool(row.get("task_frame_v46_yaw_unobservable"), False):
        return False
    yaw_conf = _as_float(row.get("task_frame_v46_yaw_confidence"), float("nan"))
    if not np.isfinite(yaw_conf) or yaw_conf < float(min_yaw_confidence):
        return False
    hypothesis_gap = _as_float(row.get("task_frame_v46_yaw_hypothesis_gap"), float("nan"))
    if np.isfinite(hypothesis_gap) and hypothesis_gap < float(min_yaw_hypothesis_gap):
        return False
    if require_yaw_alias_stable:
        decision = str(row.get("yaw_alias_drift_decision", row.get("alias_drift_decision", "")) or "").strip()
        if decision not in {"stable_alias_control", "stable_alias", "yaw_stable_alias_control"}:
            return False
    return True


def _offline_yaw_observable_symmetry_ready(
    row: Mapping[str, Any],
    *,
    require_yaw_alias_stable: bool,
) -> bool:
    labels = row.get("offline_labels", {})
    labels = labels if isinstance(labels, Mapping) else {}
    yaw_label = row.get("yaw_label", {})
    yaw_label = yaw_label if isinstance(yaw_label, Mapping) else {}
    yaw_class = str(row.get("yaw_observability_class", yaw_label.get("yaw_observability_class", "")) or "").strip().lower()
    yaw_observable = _as_bool(
        labels.get(
            "yaw_observable",
            row.get("yaw_control_observable", row.get("yaw_observable", yaw_label.get("yaw_control_observable", yaw_class == "observable"))),
        ),
        yaw_class == "observable",
    )
    yaw_ambiguous = _as_bool(labels.get("yaw_ambiguous", row.get("yaw_ambiguous", yaw_class in {"ambiguous", "unobservable"})), yaw_class in {"ambiguous", "unobservable"})
    yaw_unobservable = _as_bool(labels.get("yaw_unobservable", row.get("yaw_unobservable", yaw_class == "unobservable")), yaw_class == "unobservable")
    if yaw_class == "unobservable":
        yaw_observable = False
        yaw_unobservable = True
    if yaw_class == "ambiguous":
        yaw_ambiguous = True
    if not yaw_observable or yaw_ambiguous or yaw_unobservable:
        return False
    if require_yaw_alias_stable:
        decision = str(row.get("yaw_alias_drift_decision", row.get("alias_drift_decision", "")) or "").strip()
        if decision not in {"stable_alias_control", "stable_alias", "yaw_stable_alias_control"}:
            return False
    return True


def _selection_ready(
    row: Mapping[str, Any],
    *,
    selection_mode: str,
    near_field_threshold: float,
    min_xy_error: float,
    max_xy_error: float,
    min_abs_z: float,
    max_abs_z: float,
    max_abs_yaw: float,
    min_yaw_confidence: float = 0.20,
    min_yaw_hypothesis_gap: float = 0.10,
    require_yaw_alias_stable: bool = True,
) -> bool:
    mode = str(selection_mode)
    if mode == "v46_near":
        return _runtime_near_ready(row, near_field_threshold=near_field_threshold)
    if mode == "probe_actionable":
        return bool(row.get("grasp_probe_candidate_actionable", False)) and (
            bool(row.get("grasp_probe_frontier_pullback_candidate", False))
            or bool(row.get("grasp_probe_outer_pullback_candidate", False))
            or bool(row.get("grasp_probe_near_basin_shell", False))
            or bool(row.get("grasp_probe_tight_near_basin_shell", False))
            or bool(row.get("z_near_alignment", False))
            or bool(row.get("c2c_gate_target_nearfield", False))
        )
    if mode == "offline_residual_band":
        return _offline_residual_band_ready(
            row,
            min_xy_error=min_xy_error,
            max_xy_error=max_xy_error,
            min_abs_z=min_abs_z,
            max_abs_z=max_abs_z,
            max_abs_yaw=max_abs_yaw,
        )
    if mode == "probe_or_offline_residual_band":
        return _selection_ready(
            row,
            selection_mode="probe_actionable",
            near_field_threshold=near_field_threshold,
            min_xy_error=min_xy_error,
            max_xy_error=max_xy_error,
            min_abs_z=min_abs_z,
            max_abs_z=max_abs_z,
            max_abs_yaw=max_abs_yaw,
            min_yaw_confidence=min_yaw_confidence,
            min_yaw_hypothesis_gap=min_yaw_hypothesis_gap,
            require_yaw_alias_stable=require_yaw_alias_stable,
        ) or _selection_ready(
            row,
            selection_mode="offline_residual_band",
            near_field_threshold=near_field_threshold,
            min_xy_error=min_xy_error,
            max_xy_error=max_xy_error,
            min_abs_z=min_abs_z,
            max_abs_z=max_abs_z,
            max_abs_yaw=max_abs_yaw,
            min_yaw_confidence=min_yaw_confidence,
            min_yaw_hypothesis_gap=min_yaw_hypothesis_gap,
            require_yaw_alias_stable=require_yaw_alias_stable,
        )
    if mode == "yaw_observable_symmetry":
        return _yaw_observable_symmetry_ready(
            row,
            min_yaw_confidence=min_yaw_confidence,
            min_yaw_hypothesis_gap=min_yaw_hypothesis_gap,
            require_yaw_alias_stable=require_yaw_alias_stable,
        )
    if mode == "yaw_observable_symmetry_or_offline_label":
        return _yaw_observable_symmetry_ready(
            row,
            min_yaw_confidence=min_yaw_confidence,
            min_yaw_hypothesis_gap=min_yaw_hypothesis_gap,
            require_yaw_alias_stable=require_yaw_alias_stable,
        ) or _offline_yaw_observable_symmetry_ready(row, require_yaw_alias_stable=require_yaw_alias_stable)
    if mode == "yaw_selector_permitted":
        return _as_bool(row.get("task_frame_v46_yaw_selector_allowed"), False)
    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


def _candidate_steps(
    *,
    xy_step: float,
    z_step: float,
    yaw_step: float,
    include_combined: bool,
    candidate_profile: str = "bounded_axis_sweep",
    z_steps: tuple[float, ...] = (),
    yaw_steps: tuple[float, ...] = (),
) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []

    def add(name: str, values: tuple[float, float, float, float, float, float]) -> None:
        arr = np.asarray(values, dtype=np.float32)
        if not any(np.allclose(arr, existing, atol=1.0e-9, rtol=0.0) for _, existing in candidates):
            candidates.append((name, arr))

    add("zero", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    profile = str(candidate_profile)
    if profile not in {"bounded_axis_sweep", "z_yaw_diagnostic", "yaw_observable_symmetry"}:
        raise ValueError(f"Unsupported candidate_profile: {candidate_profile}")
    for axis, label, step in ((0, "x", xy_step), (1, "y", xy_step), (2, "z", z_step), (5, "yaw", yaw_step)):
        if step <= 0.0:
            continue
        for sign in (1.0, -1.0):
            arr = np.zeros((6,), dtype=np.float32)
            arr[axis] = float(sign * step)
            add(f"{label}_{'pos' if sign > 0 else 'neg'}", tuple(float(v) for v in arr.tolist()))
    if profile == "z_yaw_diagnostic":
        for step in z_steps:
            if step <= 0.0:
                continue
            for sign in (1.0, -1.0):
                arr = np.zeros((6,), dtype=np.float32)
                arr[2] = float(sign * step)
                add(f"z_{'pos' if sign > 0 else 'neg'}_{int(round(step * 10000)):04d}", tuple(float(v) for v in arr.tolist()))
        for step in yaw_steps:
            if step <= 0.0:
                continue
            for sign in (1.0, -1.0):
                arr = np.zeros((6,), dtype=np.float32)
                arr[5] = float(sign * step)
                add(f"yaw_{'pos' if sign > 0 else 'neg'}_{int(round(step * 10000)):04d}", tuple(float(v) for v in arr.tolist()))
        for z_mag in z_steps:
            if z_mag <= 0.0:
                continue
            for yaw_mag in yaw_steps:
                if yaw_mag <= 0.0:
                    continue
                for sz in (1.0, -1.0):
                    for syaw in (1.0, -1.0):
                        arr = np.zeros((6,), dtype=np.float32)
                        arr[2] = float(sz * z_mag)
                        arr[5] = float(syaw * yaw_mag)
                        add(
                            f"zyaw_{'p' if sz > 0 else 'n'}{'p' if syaw > 0 else 'n'}_z{int(round(z_mag * 10000)):04d}_y{int(round(yaw_mag * 10000)):04d}",
                            tuple(float(v) for v in arr.tolist()),
                        )
    if profile == "yaw_observable_symmetry":
        yaw_magnitudes = tuple(float(v) for v in yaw_steps if float(v) > 0.0) or (
            float(yaw_step),
            float(yaw_step) * 2.0,
        )
        z_magnitudes = tuple(float(v) for v in z_steps if float(v) > 0.0) or ((float(z_step),) if z_step > 0.0 else ())
        for step in yaw_magnitudes:
            if step <= 0.0:
                continue
            for sign, label in ((1.0, "hyp_pos"), (-1.0, "hyp_neg")):
                arr = np.zeros((6,), dtype=np.float32)
                arr[5] = float(sign * step)
                add(f"yaw_{label}_{int(round(step * 10000)):04d}", tuple(float(v) for v in arr.tolist()))
        for z_mag in z_magnitudes:
            if z_mag <= 0.0:
                continue
            for sz in (1.0, -1.0):
                arr = np.zeros((6,), dtype=np.float32)
                arr[2] = float(sz * z_mag)
                add(f"z_guard_{'pos' if sz > 0 else 'neg'}_{int(round(z_mag * 10000)):04d}", tuple(float(v) for v in arr.tolist()))
            for yaw_mag in yaw_magnitudes:
                if yaw_mag <= 0.0:
                    continue
                for sz in (1.0, -1.0):
                    for syaw, ylabel in ((1.0, "hyp_pos"), (-1.0, "hyp_neg")):
                        arr = np.zeros((6,), dtype=np.float32)
                        arr[2] = float(sz * z_mag)
                        arr[5] = float(syaw * yaw_mag)
                        add(
                            f"zyaw_sym_{'p' if sz > 0 else 'n'}_{ylabel}_z{int(round(z_mag * 10000)):04d}_y{int(round(yaw_mag * 10000)):04d}",
                            tuple(float(v) for v in arr.tolist()),
                        )
    if include_combined and xy_step > 0.0:
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                arr = np.zeros((6,), dtype=np.float32)
                arr[0] = float(sx * xy_step)
                arr[1] = float(sy * xy_step)
                add(f"xy_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}", tuple(float(v) for v in arr.tolist()))
    return candidates


def _public_runtime_trace_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    keep_prefixes = (
        "task_frame_v46_",
        "grasp_probe_visibility",
        "visual_observability",
        "failure_bucket",
        "c2c_v2_stage",
        "stage",
        "runtime_stage",
        "planner_gripper_",
        "alignment_",
    )
    keep_exact = {
        "alias_drift_decision",
        "yaw_alias_drift_decision",
        "alias_drift_support_source",
        "observability_bucket",
        "visual_observability_class",
        "yaw_observability_class",
        "yaw_control_block_reason",
    }
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in PRIVILEGED_TRACE_KEYS:
            continue
        if key in keep_exact or key.startswith(keep_prefixes):
            out[key] = value
    return out


def _build_sweep_rows_for_trace_row(
    raw: Mapping[str, Any],
    *,
    trace_path: Path,
    source_root: Path,
    runtime_obs: Path,
    episode_idx: int,
    sweep_steps: list[tuple[str, np.ndarray]],
    candidate_policy: str,
) -> list[dict[str, Any]]:
    base = _vec(raw.get("grasp_probe_local_command_local_6d"), 6)
    if base is None:
        base = _vec(raw.get("pre_clip_action_local_6d"), 6)
    if base is None:
        base = np.zeros((6,), dtype=np.float32)
    step_idx = int(raw.get("step_idx", raw.get("step", 0)) or 0)
    public_trace = _public_runtime_trace_fields(raw)
    rows: list[dict[str, Any]] = []
    for name, step in sweep_steps:
        command = (base + step).astype(np.float32)
        rows.append(
            {
                "schema_version": "c2c_v2_task_frame_command_sweep_spec_v1",
                "source_eval_root": str(source_root),
                "source_eval_root_kind": "runtime_command_sweep_spec",
                "session_id": str(source_root.name),
                "sequence_id": f"{source_root}::ep{episode_idx:03d}",
                "episode_idx": int(episode_idx),
                "step_idx": int(step_idx),
                "trace_path": str(trace_path),
                "runtime_obs_path": str(runtime_obs),
                "obs_pointer": {"trace_path": str(trace_path), "runtime_obs_path": str(runtime_obs)},
                "uses_privileged_runtime": False,
                "uses_privileged_label_for_training": False,
                "privileged_label_boundary": "no_transition_label_until_candidate_command_executed",
                "runtime_input_schema": "wrist_rgbd_depth_validity_proprio_planner_prior_history_candidate_command",
                "candidate_policy": str(candidate_policy),
                "candidate_name": str(name),
                "base_command_local_6d": [float(v) for v in base.tolist()],
                "candidate_step_local_6d": [float(v) for v in step.tolist()],
                "candidate_command_local_6d": [float(v) for v in command.tolist()],
                "has_next_residual": False,
                "has_command_6d": True,
                "command_6d": [float(v) for v in command.tolist()],
                "planner_gripper_close_requested": bool(raw.get("planner_gripper_close_requested", False)),
                "planner_gripper_handoff_allowed": bool(raw.get("planner_gripper_handoff_allowed", False)),
                "alignment_ready_for_handoff": bool(raw.get("alignment_ready_for_handoff", False)),
                "close_control_allowed": False,
                "runtime_trace_fields": public_trace,
            }
        )
    return rows


def _source_root_for_row_or_trace(trace_path: Path, row: Mapping[str, Any]) -> Path:
    value = row.get("source_eval_root", "")
    if value:
        return Path(str(value))
    obs = _runtime_obs_path_from_row(row)
    if obs is not None:
        return obs.parent.parent if obs.parent.name == "runtime_observations" else obs.parent
    return _source_root_from_trace(trace_path)


def _trace_path_for_row_or_input(input_path: Path, row: Mapping[str, Any]) -> Path:
    obs_pointer = row.get("obs_pointer", {})
    obs_pointer = obs_pointer if isinstance(obs_pointer, Mapping) else {}
    value = row.get("trace_path", row.get("source_trace_path", obs_pointer.get("trace_path", "")))
    if value:
        return Path(str(value))
    return input_path


def build_manifest(
    input_paths: list[Path],
    *,
    output_jsonl: Path,
    summary_json: Path,
    max_source_rows: int = 0,
    near_field_threshold: float = 0.50,
    xy_step: float = 0.003,
    z_step: float = 0.003,
    yaw_step: float = 0.010,
    candidate_profile: str = "bounded_axis_sweep",
    z_steps: tuple[float, ...] = (),
    yaw_steps: tuple[float, ...] = (),
    include_combined_xy: bool = True,
    require_stage: str = "RING_GRASP_ALIGN",
    selection_mode: str = "v46_near",
    min_step: int = 0,
    max_step: int = -1,
    min_xy_error: float = 0.0,
    max_xy_error: float = 0.18,
    min_abs_z: float = 0.0,
    max_abs_z: float = 0.75,
    max_abs_yaw: float = 0.80,
    min_yaw_confidence: float = 0.20,
    min_yaw_hypothesis_gap: float = 0.10,
    require_yaw_alias_stable: bool = True,
) -> dict[str, Any]:
    trace_files = _discover_trace_files(input_paths)
    sweep_steps = _candidate_steps(
        xy_step=xy_step,
        z_step=z_step,
        yaw_step=yaw_step,
        include_combined=include_combined_xy,
        candidate_profile=candidate_profile,
        z_steps=tuple(float(v) for v in z_steps),
        yaw_steps=tuple(float(v) for v in yaw_steps),
    )
    counters: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    per_source_rows: Counter[str] = Counter()
    for trace_path in trace_files:
        trace_rows = _read_jsonl(trace_path)
        counters["input_rows"] += len(trace_rows)
        for raw in trace_rows:
            effective_trace_path = _trace_path_for_row_or_input(trace_path, raw)
            source_root = _source_root_for_row_or_trace(effective_trace_path, raw)
            step_idx = int(raw.get("step_idx", raw.get("step", 0)) or 0)
            if step_idx < int(min_step):
                counters["dropped_before_min_step"] += 1
                continue
            if int(max_step) >= 0 and step_idx > int(max_step):
                counters["dropped_after_max_step"] += 1
                continue
            stage = str(raw.get("c2c_v2_stage", raw.get("stage", raw.get("runtime_stage", raw.get("stage_name", "")))) or "")
            if require_stage and stage != str(require_stage):
                counters["dropped_stage"] += 1
                continue
            if not _selection_ready(
                raw,
                selection_mode=selection_mode,
                near_field_threshold=near_field_threshold,
                min_xy_error=min_xy_error,
                max_xy_error=max_xy_error,
                min_abs_z=min_abs_z,
                max_abs_z=max_abs_z,
                max_abs_yaw=max_abs_yaw,
                min_yaw_confidence=min_yaw_confidence,
                min_yaw_hypothesis_gap=min_yaw_hypothesis_gap,
                require_yaw_alias_stable=require_yaw_alias_stable,
            ):
                counters[f"dropped_not_selected_{selection_mode}"] += 1
                continue
            episode_idx = _episode_from_trace(trace_path, raw)
            if episode_idx < 0:
                counters["dropped_missing_episode"] += 1
                continue
            runtime_obs = _runtime_obs_path_from_row(raw)
            if runtime_obs is None:
                runtime_obs = _runtime_obs_path(source_root, episode_idx)
            if runtime_obs is None:
                counters["dropped_missing_runtime_obs"] += 1
                continue
            source_key = str(source_root)
            if max_source_rows > 0 and per_source_rows[source_key] >= int(max_source_rows):
                counters["dropped_source_cap"] += 1
                continue
            built = _build_sweep_rows_for_trace_row(
                raw,
                trace_path=effective_trace_path,
                source_root=source_root,
                runtime_obs=runtime_obs,
                episode_idx=episode_idx,
                sweep_steps=sweep_steps,
                candidate_policy=str(candidate_profile),
            )
            rows.extend(built)
            per_source_rows[source_key] += 1
            counters["selected_runtime_rows"] += 1
            counters["candidate_command_rows"] += len(built)
    rows.sort(key=lambda row: (str(row["source_eval_root"]), int(row["episode_idx"]), int(row["step_idx"]), str(row["candidate_name"])))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_counts = Counter(str(row["source_eval_root"]) for row in rows)
    candidate_counts = Counter(str(row["candidate_name"]) for row in rows)
    summary = {
        "schema_version": "c2c_v2_task_frame_command_sweep_spec_summary_v1",
        "output_jsonl": str(output_jsonl),
        "input_paths": [str(path) for path in input_paths],
        "trace_files": [str(path) for path in trace_files],
        "retained_rows": len(rows),
        "selected_runtime_rows": int(counters["selected_runtime_rows"]),
        "candidate_commands_per_runtime_row": len(sweep_steps),
        "source_eval_roots": len(source_counts),
        "source_eval_root_counts": dict(source_counts),
        "candidate_counts": dict(candidate_counts),
        "counters": dict(counters),
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_training": False,
        "privileged_label_boundary": "no_transition_label_until_candidate_command_executed",
        "require_stage": str(require_stage),
        "near_field_threshold": float(near_field_threshold),
        "xy_step": float(xy_step),
        "z_step": float(z_step),
        "yaw_step": float(yaw_step),
        "candidate_profile": str(candidate_profile),
        "z_steps": [float(v) for v in z_steps],
        "yaw_steps": [float(v) for v in yaw_steps],
        "include_combined_xy": bool(include_combined_xy),
        "selection_mode": str(selection_mode),
        "selection_label_boundary": "offline_selection_allowed_but_spec_rows_strip_privileged_labels",
        "min_step": int(min_step),
        "max_step": int(max_step),
        "min_xy_error": float(min_xy_error),
        "max_xy_error": float(max_xy_error),
        "min_abs_z": float(min_abs_z),
        "max_abs_z": float(max_abs_z),
        "max_abs_yaw": float(max_abs_yaw),
        "min_yaw_confidence": float(min_yaw_confidence),
        "min_yaw_hypothesis_gap": float(min_yaw_hypothesis_gap),
        "require_yaw_alias_stable": bool(require_yaw_alias_stable),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", required=True, type=Path)
    parser.add_argument("--max_source_rows", type=int, default=0)
    parser.add_argument("--near_field_threshold", type=float, default=0.50)
    parser.add_argument("--xy_step", type=float, default=0.003)
    parser.add_argument("--z_step", type=float, default=0.003)
    parser.add_argument("--yaw_step", type=float, default=0.010)
    parser.add_argument(
        "--candidate_profile",
        default="bounded_axis_sweep",
        choices=("bounded_axis_sweep", "z_yaw_diagnostic", "yaw_observable_symmetry"),
    )
    parser.add_argument("--z_steps", default="", help="Comma-separated additional Z step magnitudes for z_yaw_diagnostic.")
    parser.add_argument("--yaw_steps", default="", help="Comma-separated additional yaw step magnitudes for z_yaw_diagnostic.")
    parser.add_argument("--no_combined_xy", action="store_true")
    parser.add_argument("--require_stage", default="RING_GRASP_ALIGN")
    parser.add_argument(
        "--selection_mode",
        default="v46_near",
        choices=(
            "v46_near",
            "probe_actionable",
            "offline_residual_band",
            "probe_or_offline_residual_band",
            "yaw_observable_symmetry",
            "yaw_observable_symmetry_or_offline_label",
            "yaw_selector_permitted",
        ),
    )
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=-1)
    parser.add_argument("--min_xy_error", type=float, default=0.0)
    parser.add_argument("--max_xy_error", type=float, default=0.18)
    parser.add_argument("--min_abs_z", type=float, default=0.0)
    parser.add_argument("--max_abs_z", type=float, default=0.75)
    parser.add_argument("--max_abs_yaw", type=float, default=0.80)
    parser.add_argument("--min_yaw_confidence", type=float, default=0.20)
    parser.add_argument("--min_yaw_hypothesis_gap", type=float, default=0.10)
    parser.add_argument("--allow_yaw_alias_unknown", action="store_true")
    args = parser.parse_args()
    summary = build_manifest(
        list(args.input),
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        max_source_rows=int(args.max_source_rows),
        near_field_threshold=float(args.near_field_threshold),
        xy_step=float(args.xy_step),
        z_step=float(args.z_step),
        yaw_step=float(args.yaw_step),
        candidate_profile=str(args.candidate_profile),
        z_steps=tuple(float(v) for v in str(args.z_steps).split(",") if v.strip()),
        yaw_steps=tuple(float(v) for v in str(args.yaw_steps).split(",") if v.strip()),
        include_combined_xy=not bool(args.no_combined_xy),
        require_stage=str(args.require_stage),
        selection_mode=str(args.selection_mode),
        min_step=int(args.min_step),
        max_step=int(args.max_step),
        min_xy_error=float(args.min_xy_error),
        max_xy_error=float(args.max_xy_error),
        min_abs_z=float(args.min_abs_z),
        max_abs_z=float(args.max_abs_z),
        max_abs_yaw=float(args.max_abs_yaw),
        min_yaw_confidence=float(args.min_yaw_confidence),
        min_yaw_hypothesis_gap=float(args.min_yaw_hypothesis_gap),
        require_yaw_alias_stable=not bool(args.allow_yaw_alias_unknown),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
