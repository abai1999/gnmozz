#!/usr/bin/env python3
"""Audit the analytic target-delta servo baseline and replay upper bound."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _load_traces(trace_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        episode_label = path.name.replace("_gripper_trace.jsonl", "")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row.setdefault("_trace_episode", episode_label)
                    rows.append(row)
    return rows


def _finite_vec(row: dict, key: str, dim: int) -> np.ndarray | None:
    value = row.get(key, None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < dim or not np.all(np.isfinite(arr[:dim])):
        return None
    return arr[:dim].copy()


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(r.get(key, False)) for r in rows]))


def _servo_projection(delta: np.ndarray, *, k_xy: float, k_z: float, k_yaw: float, max_pos: float, max_yaw: float):
    delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    if delta.size < 6:
        return None
    cur_xy = float(np.linalg.norm(delta[:2]))
    cur_z = float(abs(delta[2]))
    cur_yaw = float(abs(delta[5]))
    servo = np.zeros(6, dtype=np.float32)
    servo[:2] = delta[:2] * float(k_xy)
    servo[2] = float(delta[2]) * float(k_z)
    servo[5] = float(delta[5]) * float(k_yaw)
    pos_norm = float(np.linalg.norm(servo[:3]))
    if pos_norm > max_pos and pos_norm > 1e-8:
        servo[:3] *= float(max_pos / pos_norm)
    if abs(servo[5]) > max_yaw and abs(servo[5]) > 1e-8:
        servo[5] *= float(max_yaw / abs(servo[5]))
    post = delta[:6] - servo
    return {
        "cur_xy": cur_xy,
        "cur_z": cur_z,
        "cur_yaw": cur_yaw,
        "servo_local": servo,
        "servo_pos_norm": float(np.linalg.norm(servo[:3])),
        "servo_yaw_abs": float(abs(servo[5])),
        "post_xy": float(np.linalg.norm(post[:2])),
        "post_z": float(abs(post[2])),
        "post_yaw": float(abs(post[5])),
        "xy_improved": bool(np.linalg.norm(post[:2]) < cur_xy),
        "z_improved": bool(abs(post[2]) < cur_z),
        "yaw_improved": bool(abs(post[5]) < cur_yaw),
    }


def _summarize_projection(
    rows: list[dict],
    *,
    delta_key: str | None,
    source_key: str | None,
    block_key: str | None,
    k_xy: float,
    k_z: float,
    k_yaw: float,
    max_pos: float,
    max_yaw: float,
    source_filter: tuple[str, ...] | None = None,
    zero_delta: bool = False,
):
    gate_rows: list[dict] = []
    applied_rows: list[dict] = []
    delta_missing = 0
    cur_xy_vals: list[float] = []
    cur_z_vals: list[float] = []
    cur_yaw_vals: list[float] = []
    post_xy_vals: list[float] = []
    post_z_vals: list[float] = []
    post_yaw_vals: list[float] = []
    servo_pos_vals: list[float] = []
    servo_yaw_vals: list[float] = []
    xy_improved = 0
    z_improved = 0
    yaw_improved = 0
    source_hist = Counter()
    block_hist = Counter()
    for row in rows:
        if source_filter is not None and not any(tok in str(row.get("refiner_target_delta_servo_source", "")) for tok in source_filter):
            continue
        delta = np.zeros(6, dtype=np.float32) if zero_delta else _finite_vec(row, delta_key or "", 6)
        if delta is None:
            delta_missing += 1
            continue
        proj = _servo_projection(delta, k_xy=k_xy, k_z=k_z, k_yaw=k_yaw, max_pos=max_pos, max_yaw=max_yaw)
        if proj is None:
            delta_missing += 1
            continue
        source_hist[
            str(
                row.get(
                    source_key or "refiner_target_delta_servo_source",
                    row.get("target_provider_source", "unknown"),
                )
            )
        ] += 1
        block_hist[str(row.get(block_key or "refiner_target_delta_servo_block_reason", "unknown"))] += 1
        cur_xy_vals.append(proj["cur_xy"])
        cur_z_vals.append(proj["cur_z"])
        cur_yaw_vals.append(proj["cur_yaw"])
        post_xy_vals.append(proj["post_xy"])
        post_z_vals.append(proj["post_z"])
        post_yaw_vals.append(proj["post_yaw"])
        servo_pos_vals.append(proj["servo_pos_norm"])
        servo_yaw_vals.append(proj["servo_yaw_abs"])
        gate_rows.append(row)
        if bool(row.get("refiner_target_delta_servo_applied", False)):
            applied_rows.append(row)
        xy_improved += int(proj["xy_improved"])
        z_improved += int(proj["z_improved"])
        yaw_improved += int(proj["yaw_improved"])
    total = len(gate_rows)
    return {
        "rows": total,
        "delta_missing_rows": delta_missing,
        "apply_rows": len(applied_rows),
        "apply_rate": _rate(rows, "refiner_target_delta_servo_applied"),
        "gate_pass_rate": _rate(rows, "refiner_target_delta_servo_gate_pass"),
        "enabled_rate": _rate(rows, "refiner_target_delta_servo_enabled"),
        "shadow_enabled_rate": _rate(rows, "refiner_target_delta_servo_shadow_enabled"),
        "apply_enabled_rate": _rate(rows, "refiner_target_delta_servo_apply_enabled"),
        "source_histogram": dict(source_hist),
        "block_reason_histogram": dict(block_hist),
        "cur_xy": _stats(cur_xy_vals),
        "cur_z": _stats(cur_z_vals),
        "cur_yaw": _stats(cur_yaw_vals),
        "post_xy": _stats(post_xy_vals),
        "post_z": _stats(post_z_vals),
        "post_yaw": _stats(post_yaw_vals),
        "servo_pos_norm": _stats(servo_pos_vals),
        "servo_yaw_abs": _stats(servo_yaw_vals),
        "xy_improved_rate": float(xy_improved / max(total, 1)),
        "z_improved_rate": float(z_improved / max(total, 1)),
        "yaw_improved_rate": float(yaw_improved / max(total, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k_xy", type=float, default=0.08)
    parser.add_argument("--k_z", type=float, default=0.06)
    parser.add_argument("--k_yaw", type=float, default=0.04)
    parser.add_argument("--max_pos", type=float, default=0.0010)
    parser.add_argument("--max_yaw", type=float, default=0.0040)
    args = parser.parse_args()

    rows = _load_traces(args.trace_dir)
    if not rows:
        raise SystemExit(f"no trace rows found under {args.trace_dir}")

    runtime = _summarize_projection(
        rows,
        delta_key="refiner_target_delta_servo_local_delta",
        source_key="refiner_target_delta_servo_source",
        block_key="refiner_target_delta_servo_block_reason",
        k_xy=args.k_xy,
        k_z=args.k_z,
        k_yaw=args.k_yaw,
        max_pos=args.max_pos,
        max_yaw=args.max_yaw,
    )

    privileged = _summarize_projection(
        rows,
        delta_key="privileged_current_delta_basin_target",
        source_key="privileged_target_provider_source",
        block_key="refiner_target_delta_servo_block_reason",
        k_xy=args.k_xy,
        k_z=args.k_z,
        k_yaw=args.k_yaw,
        max_pos=args.max_pos,
        max_yaw=args.max_yaw,
    )

    zero = _summarize_projection(
        rows,
        delta_key=None,
        source_key="refiner_target_delta_servo_source",
        block_key="refiner_target_delta_servo_block_reason",
        k_xy=args.k_xy,
        k_z=args.k_z,
        k_yaw=args.k_yaw,
        max_pos=args.max_pos,
        max_yaw=args.max_yaw,
        zero_delta=True,
    )

    basin = _summarize_projection(
        rows,
        delta_key="refiner_target_delta_servo_local_delta",
        source_key="refiner_target_delta_servo_source",
        block_key="refiner_target_delta_servo_block_reason",
        k_xy=args.k_xy,
        k_z=args.k_z,
        k_yaw=args.k_yaw,
        max_pos=args.max_pos,
        max_yaw=args.max_yaw,
        source_filter=("basin", "canonical", "fallback"),
    )

    per_episode: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("_trace_episode", row.get("episode_index", row.get("episode_id", "unknown"))))].append(row)
    for ep, ep_rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        per_episode[ep] = {
            "runtime": _summarize_projection(
                ep_rows,
                delta_key="refiner_target_delta_servo_local_delta",
                source_key="refiner_target_delta_servo_source",
                block_key="refiner_target_delta_servo_block_reason",
                k_xy=args.k_xy,
                k_z=args.k_z,
                k_yaw=args.k_yaw,
                max_pos=args.max_pos,
                max_yaw=args.max_yaw,
            ),
            "privileged_replay": _summarize_projection(
                ep_rows,
                delta_key="privileged_current_delta_basin_target",
                source_key="privileged_target_provider_source",
                block_key="refiner_target_delta_servo_block_reason",
                k_xy=args.k_xy,
                k_z=args.k_z,
                k_yaw=args.k_yaw,
                max_pos=args.max_pos,
                max_yaw=args.max_yaw,
            ),
        }

    report = {
        "rows": len(rows),
        "overall": {
            "runtime": runtime,
            "privileged_replay": privileged,
            "zero_diagnostic": zero,
            "basin_diagnostic": basin,
        },
        "per_episode": per_episode,
        "zone_histogram": dict(Counter(str(r.get("zone_state", "unknown")) for r in rows)),
        "target_delta_servo_source_histogram": dict(
            Counter(str(r.get("refiner_target_delta_servo_source", "unknown")) for r in rows)
        ),
        "target_delta_servo_block_reason_histogram": dict(
            Counter(str(r.get("refiner_target_delta_servo_block_reason", "unknown")) for r in rows)
        ),
        "workspace_violation_count": int(sum(int(r.get("workspace_violation_count", 0) or 0) for r in rows)),
        "invalid_action_count": int(sum(int(r.get("invalid_action_count", 0) or 0) for r in rows)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
