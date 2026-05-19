#!/usr/bin/env python3
"""Exercise StageAwareRefiner alignment gate logic with a fixed scenario matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from prismatic.robot.stage_aware_refiner import StageAwareRefiner


@dataclass(frozen=True)
class GateCase:
    name: str
    depth: float | None
    gripper_open: float | None
    action_gripper: float
    future_gripper: tuple[float, ...]
    support_inner: bool
    support_outer: bool
    outer_rescue: bool
    close_veto_ready: bool
    refine_ready: bool
    takeover_ready: bool
    window_count: int
    expect_gate_open: bool
    expect_window_active: bool
    expect_block: str


def _make_refiner(case: GateCase) -> StageAwareRefiner:
    refiner = StageAwareRefiner.__new__(StageAwareRefiner)
    refiner.require_pregrasp_alignment_gate = True
    refiner.alignment_depth_threshold = 0.10
    refiner.alignment_open_threshold = 0.50
    refiner.alignment_close_command_threshold = 0.20
    refiner.require_close_intent_for_alignment = True
    refiner.enable_outer_rescue = bool(case.outer_rescue)
    refiner.skip_alignment_when_close_ready = False
    refiner.max_alignment_corrections_per_window = 2
    refiner._alignment_window_corrections = int(case.window_count)
    refiner._residual_cooldown = 0
    refiner._last_alignment_gate_debug = {}

    refiner._runtime_has_motion_target = lambda controller: True
    refiner._runtime_has_handoff_geometry = lambda controller: False
    refiner._delta_within_band = lambda current_delta, controller, band="inner": (
        bool(case.support_inner) if band == "inner" else bool(case.support_outer)
    )
    refiner._close_veto_ready = lambda controller, gripper_open, step_idx=None: bool(case.close_veto_ready)
    refiner._alignment_refine_band_ready = lambda controller, gripper_open: bool(case.refine_ready)
    refiner._alignment_takeover_band_ready = lambda controller, gripper_open: bool(case.takeover_ready)
    refiner._alignment_skip_ready = lambda controller, gripper_open: False
    return refiner


def _controller() -> SimpleNamespace:
    return SimpleNamespace(
        _controller_type="pose_field_scorer",
        _runtime_current_basin_distance=0.01,
        _support_basin_distance_max=0.05,
        _runtime_current_delta_basin_target=np.array([0.01, 0.0, 0.02, 0.0, 0.0, 0.05], dtype=np.float32),
    )


def _cases() -> list[GateCase]:
    return [
        GateCase("far_depth_blocks", 0.20, 1.0, 0.0, (), True, True, False, False, False, False, 0, False, False, "depth"),
        GateCase("missing_depth_blocks", None, 1.0, 0.0, (), True, True, False, False, False, False, 0, False, False, "depth"),
        GateCase("closed_gripper_blocks", 0.03, 0.1, 0.0, (), True, True, False, False, False, False, 0, False, False, "gripper_open"),
        GateCase("no_close_intent_blocks", 0.03, 1.0, 1.0, (), True, True, False, False, False, False, 0, False, False, "close_intent"),
        GateCase("near_support_opens_current_close", 0.03, 1.0, 0.0, (), True, True, False, False, False, False, 0, True, True, "none"),
        GateCase("near_support_opens_future_close", 0.03, 1.0, 1.0, (0.0,), True, True, False, False, False, False, 0, True, True, "none"),
        GateCase("support_inner_missing_blocks", 0.03, 1.0, 0.0, (), False, False, False, False, False, False, 0, False, False, "support"),
        GateCase("outer_rescue_opens_without_inner", 0.03, 1.0, 0.0, (), False, True, True, False, False, False, 0, True, True, "none"),
        GateCase("ready_to_close_blocks_alignment", 0.03, 1.0, 0.0, (), True, True, False, True, False, False, 0, False, False, "ready_to_close"),
        GateCase("refine_ready_opens_without_close_intent", 0.03, 1.0, 1.0, (), False, False, False, False, True, False, 0, True, True, "none"),
        GateCase("budget_exhausted_blocks", 0.03, 1.0, 0.0, (), True, True, False, False, False, False, 2, False, True, "window"),
        GateCase("outer_rescue_ignores_budget", 0.03, 1.0, 0.0, (), False, True, True, False, False, False, 2, True, True, "none"),
    ]


def main() -> None:
    failures: list[dict] = []
    rows: list[dict] = []
    for case in _cases():
        refiner = _make_refiner(case)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, case.action_gripper], dtype=np.float32)
        decision = refiner._alignment_gate_decision(
            depth_proximity=case.depth,
            gripper_open=case.gripper_open,
            a_base_7d=action,
            future_gripper_actions=case.future_gripper,
            controller=_controller(),
            step_idx=0,
        )
        row = {
            "case": case.name,
            "gate_open": bool(decision["gate_open"]),
            "alignment_window_active": bool(decision["alignment_window_active"]),
            "blocked_reason": str(decision["blocked_reason"]),
            "planner_close_intent": bool(decision["planner_close_intent"]),
            "near_target": bool(decision["near_target"]),
            "support_inner_satisfied": bool(decision["support_inner_satisfied"]),
            "use_outer_rescue": bool(decision["use_outer_rescue"]),
        }
        rows.append(row)
        if (
            row["gate_open"] != case.expect_gate_open
            or row["alignment_window_active"] != case.expect_window_active
            or row["blocked_reason"] != case.expect_block
        ):
            failures.append({"expected": case.__dict__, "actual": row})

    report = {"audit": "alignment_gate_decision_matrix", "rows": rows, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
