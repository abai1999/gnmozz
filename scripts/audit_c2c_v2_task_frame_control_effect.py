#!/usr/bin/env python3
"""Audit closed-loop task-frame residual response to local correction commands.

This is an offline diagnostic only. It reads gripper traces containing
privileged pre/post residual sidecars and estimates how local controller steps
map to one-step task-frame residual changes. It does not affect runtime policy
and must not be treated as runtime privileged input.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _vec(row: dict[str, Any], key: str, length: int) -> np.ndarray | None:
    value = row.get(key, None)
    if not isinstance(value, (list, tuple)):
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < int(length):
        return None
    arr = arr[: int(length)]
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _regress(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if x.shape[0] < 3:
        return {"rows": int(x.shape[0]), "coef": [], "r2": 0.0}
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    pred = x_aug @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-12 else 0.0
    return {
        "rows": int(x.shape[0]),
        "coef": coef[:-1].tolist(),
        "bias": coef[-1].tolist(),
        "r2": float(r2),
    }


def audit_trace(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    samples: list[dict[str, Any]] = []
    for row in rows:
        pre = _vec(row, "grasp_probe_pre_true_error_t", 4)
        post = _vec(row, "grasp_probe_post_true_error_t", 4)
        if pre is None or post is None:
            continue
        command = _vec(row, "grasp_probe_local_command_local_6d", 6)
        v46_step = _vec(row, "task_frame_v46_applied_local_6d", 6)
        legacy_xy = _vec(row, "grasp_probe_applied_xy_step_local_6d", 6)
        if command is None:
            command = np.zeros((6,), dtype=np.float64)
        if v46_step is None:
            v46_step = np.zeros((6,), dtype=np.float64)
        if legacy_xy is None:
            legacy_xy = np.zeros((6,), dtype=np.float64)
        samples.append(
            {
                "pre": pre,
                "post": post,
                "delta": post - pre,
                "command": command,
                "v46_step": v46_step,
                "legacy_xy": legacy_xy,
                "stage": str(row.get("c2c_v2_stage", row.get("stage_name", ""))),
                "v46_applied": bool(row.get("task_frame_v46_applied", False)),
                "xy_source": str(row.get("task_frame_v46_xy_control_source", "")),
                "risk": str(row.get("task_frame_v46_risk_reason", "")),
            }
        )
    if not samples:
        return {"trace_path": str(path), "rows": int(len(rows)), "usable_rows": 0}
    pre = np.stack([s["pre"] for s in samples])
    delta = np.stack([s["delta"] for s in samples])
    command = np.stack([s["command"] for s in samples])
    v46_step = np.stack([s["v46_step"] for s in samples])
    legacy_xy = np.stack([s["legacy_xy"] for s in samples])
    xy_pre = np.linalg.norm(pre[:, :2], axis=1)
    xy_post = np.linalg.norm(pre[:, :2] + delta[:, :2], axis=1)
    out: dict[str, Any] = {
        "trace_path": str(path),
        "rows": int(len(rows)),
        "usable_rows": int(len(samples)),
        "stage_counts": dict(Counter(str(s["stage"]) for s in samples)),
        "v46_applied_rows": int(sum(bool(s["v46_applied"]) for s in samples)),
        "xy_source_counts": dict(Counter(str(s["xy_source"]) for s in samples)),
        "risk_counts": dict(Counter(str(s["risk"]) for s in samples)),
        "xy_contraction_rate": float(np.mean(xy_post < xy_pre - 1.0e-9)),
        "delta_vs_command_xy_regression": _regress(command[:, :2], delta[:, :2]),
        "delta_vs_v46_xy_regression": _regress(v46_step[:, :2], delta[:, :2]),
        "delta_vs_legacy_xy_regression": _regress(legacy_xy[:, :2], delta[:, :2]),
    }
    axis = {}
    for idx, name in enumerate(("x", "y")):
        nz = np.abs(command[:, idx]) > 1.0e-7
        if bool(np.any(nz)):
            axis[name] = {
                "rows": int(np.sum(nz)),
                "delta_command_same_sign_rate": float(np.mean(np.sign(delta[nz, idx]) == np.sign(command[nz, idx]))),
                "delta_command_corr": float(np.corrcoef(delta[nz, idx], command[nz, idx])[0, 1]) if int(np.sum(nz)) > 2 else 0.0,
            }
        else:
            axis[name] = {"rows": 0, "delta_command_same_sign_rate": 0.0, "delta_command_corr": 0.0}
    out["axis_response"] = axis
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit task-frame residual response to local C2C commands.")
    ap.add_argument("--trace", nargs="+", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    args = ap.parse_args()
    reports = [audit_trace(path) for path in args.trace]
    output = {
        "schema_version": "c2c_v2_task_frame_control_effect_audit_v1",
        "uses_privileged_runtime": False,
        "uses_privileged_label_for_audit": True,
        "reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
