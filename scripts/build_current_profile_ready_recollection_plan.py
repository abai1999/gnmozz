#!/usr/bin/env python3
"""Build a targeted recollection plan from current-profile gripper traces.

This script is intentionally trace-native.  The current ready bottleneck is
visible in online gripper traces before a support NPZ exists, so we select
episodes/windows directly from teacher-truth handoff metrics recorded during
runtime diagnosis.

The plan is designed for targeted recollection, not for validation scoring:
it prioritizes episodes that are close to release but miss one axis, especially
xy-edge and yaw-edge windows that can become teacher-ready neighborhoods with a
more focused recollection profile.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_REL_XY = 0.0085
DEFAULT_REL_Z = 0.0035
DEFAULT_REL_YAW = 0.12434040009975433


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _metric(row: dict[str, Any], name: str) -> float:
    metrics = row.get("teacher_truth_handoff_metrics") or {}
    aliases = {
        "xy": ("xy_error", "xy"),
        "z": ("abs_z_error", "z_error", "abs_z"),
        "yaw": ("yaw_error", "yaw"),
    }
    for key in aliases[name]:
        if key in metrics:
            return _as_float(metrics[key])
    # Compatibility with a few older flattened trace variants.
    flat = {
        "xy": "teacher_truth_handoff_metric_xy_error",
        "z": "teacher_truth_handoff_metric_abs_z_error",
        "yaw": "teacher_truth_handoff_metric_yaw_error",
    }
    return _as_float(row.get(flat[name]))


def _threshold(row: dict[str, Any], name: str) -> float:
    thresholds = (
        row.get("teacher_truth_handoff_release_thresholds")
        or row.get("teacher_truth_handoff_release_metric_thresholds")
        or row.get("handoff_release_metric_thresholds_provider")
        or {}
    )
    aliases = {
        "xy": ("xy_error", "xy"),
        "z": ("abs_z_error", "z_error", "abs_z"),
        "yaw": ("yaw_error", "yaw"),
    }
    for key in aliases[name]:
        if key in thresholds and _finite(thresholds[key]):
            return float(thresholds[key])
    defaults = {"xy": DEFAULT_REL_XY, "z": DEFAULT_REL_Z, "yaw": DEFAULT_REL_YAW}
    return defaults[name]


def _phase1_open(row: dict[str, Any], open_threshold: float) -> bool:
    phase = int(row.get("phase_after", row.get("refiner_phase_after", row.get("phase_before", -1))))
    gripper_open = _as_float(row.get("obs_gripper_open"), default=1.0)
    return phase == 1 and gripper_open >= open_threshold


def _append_window(
    windows: list[dict[str, Any]],
    ep: int,
    center_step: int,
    radius: int,
    tag: str,
    row: dict[str, Any],
    norms: tuple[float, float, float],
) -> None:
    xy_n, z_n, yaw_n = norms
    windows.append(
        {
            "episode_index": int(ep),
            "tag": tag,
            "center_step": int(center_step),
            "window_start": int(max(0, center_step - radius)),
            "window_end": int(center_step + radius),
            "xy_norm": xy_n,
            "z_norm": z_n,
            "yaw_norm": yaw_n,
            "teacher_ready": bool(row.get("teacher_truth_handoff_ready", False)),
            "planner_close_intent": bool(row.get("base_gripper_raw", 1.0) < 0.5)
            or bool(row.get("refiner_alignment_planner_close_intent", False)),
            "handoff_ready_pred": bool(row.get("handoff_ready_pred", False)),
        }
    )


def _pick_unique(rows: list[dict[str, Any]], selected: set[int], quota: int) -> list[int]:
    out: list[int] = []
    if quota <= 0:
        return out
    for row in rows:
        ep = int(row["episode_index"])
        if ep in selected:
            continue
        selected.add(ep)
        out.append(ep)
        if len(out) >= quota:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_windows_jsonl", required=True)
    ap.add_argument("--open_threshold", type=float, default=0.5)
    ap.add_argument("--window_radius", type=int, default=12)
    ap.add_argument("--max_episodes", type=int, default=16)
    ap.add_argument("--quota_ready_anchor", type=int, default=2)
    ap.add_argument("--quota_very_near", type=int, default=4)
    ap.add_argument("--quota_xy_edge", type=int, default=6)
    ap.add_argument("--quota_yaw_edge", type=int, default=4)
    ap.add_argument("--quota_broad_block", type=int, default=4)
    ap.add_argument("--quota_close_near", type=int, default=4)
    ap.add_argument("--very_near_factor", type=float, default=1.5)
    ap.add_argument("--edge_in_factor", type=float, default=1.2)
    ap.add_argument("--edge_out_factor", type=float, default=2.0)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    files = sorted(trace_dir.glob("ep*_gripper_trace.jsonl"))
    if not files:
        raise RuntimeError(f"no gripper trace files found in {trace_dir}")

    per_episode: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []

    for path in files:
        stem = path.name.split("_", 1)[0]
        ep = int(stem.replace("ep", ""))
        rows: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if not rows:
            continue

        valid_rows: list[tuple[dict[str, Any], tuple[float, float, float]]] = []
        for row in rows:
            if not _phase1_open(row, float(args.open_threshold)):
                continue
            xy = _metric(row, "xy")
            z = _metric(row, "z")
            yaw = _metric(row, "yaw")
            if not (_finite(xy) and _finite(z) and _finite(yaw)):
                continue
            rel_xy = max(_threshold(row, "xy"), 1e-9)
            rel_z = max(_threshold(row, "z"), 1e-9)
            rel_yaw = max(_threshold(row, "yaw"), 1e-9)
            valid_rows.append((row, (xy / rel_xy, z / rel_z, yaw / rel_yaw)))

        if not valid_rows:
            continue

        ready = [(r, n) for r, n in valid_rows if bool(r.get("teacher_truth_handoff_ready", False))]
        release = [(r, n) for r, n in valid_rows if max(n) <= 1.0]
        very_near = [(r, n) for r, n in valid_rows if max(n) <= float(args.very_near_factor)]
        xy_edge = [
            (r, n)
            for r, n in valid_rows
            if n[1] <= float(args.edge_in_factor)
            and n[2] <= float(args.edge_in_factor)
            and 1.0 < n[0] <= float(args.edge_out_factor)
        ]
        yaw_edge = [
            (r, n)
            for r, n in valid_rows
            if n[0] <= float(args.edge_in_factor)
            and n[1] <= float(args.edge_in_factor)
            and 1.0 < n[2] <= float(args.edge_out_factor)
        ]
        broad_xy_block = [
            (r, n)
            for r, n in valid_rows
            if n[0] == max(n) and n[0] > 1.0 and n[0] <= 3.0 and n[1] <= 1.8 and n[2] <= 1.8
        ]
        broad_yaw_block = [
            (r, n)
            for r, n in valid_rows
            if n[2] == max(n) and n[2] > 1.0 and n[2] <= 3.0 and n[0] <= 1.8 and n[1] <= 1.8
        ]
        close_near = [
            (r, n)
            for r, n in valid_rows
            if max(n) <= float(args.edge_out_factor)
            and (
                bool(r.get("refiner_alignment_planner_close_intent", False))
                or _as_float(r.get("base_gripper_raw"), default=1.0) < 0.5
            )
        ]

        def best_by_max_norm(items: list[tuple[dict[str, Any], tuple[float, float, float]]]):
            if not items:
                return None
            return min(items, key=lambda rn: max(rn[1]))

        tags: list[str] = []
        for tag, items in (
            ("A_teacher_ready", ready),
            ("R_release", release),
            ("V_very_near", very_near),
            ("B_xy_edge", xy_edge),
            ("Y_yaw_edge", yaw_edge),
            ("B_broad_xy_block", broad_xy_block),
            ("Y_broad_yaw_block", broad_yaw_block),
            ("C_close_near", close_near),
        ):
            if items:
                tags.append(tag)
                picked = best_by_max_norm(items)
                if picked is not None:
                    row, norms = picked
                    _append_window(windows, ep, int(row.get("step", 0)), int(args.window_radius), tag, row, norms)

        best = best_by_max_norm(valid_rows)
        assert best is not None
        best_row, best_norms = best
        per_episode.append(
            {
                "episode_index": ep,
                "tags": tags,
                "valid_rows": len(valid_rows),
                "teacher_ready_rows": len(ready),
                "release_rows": len(release),
                "very_near_rows": len(very_near),
                "xy_edge_rows": len(xy_edge),
                "yaw_edge_rows": len(yaw_edge),
                "broad_xy_block_rows": len(broad_xy_block),
                "broad_yaw_block_rows": len(broad_yaw_block),
                "close_near_rows": len(close_near),
                "min_xy_norm": min(n[0] for _, n in valid_rows),
                "min_z_norm": min(n[1] for _, n in valid_rows),
                "min_yaw_norm": min(n[2] for _, n in valid_rows),
                "best_max_norm": max(best_norms),
                "best_step": int(best_row.get("step", 0)),
                "score": (
                    20.0 * len(ready)
                    + 8.0 * len(release)
                    + 3.0 * len(very_near)
                    + 2.0 * len(xy_edge)
                    + 1.5 * len(yaw_edge)
                    + 1.2 * len(broad_xy_block)
                    + 1.0 * len(broad_yaw_block)
                    + 1.0 * len(close_near)
                    - 5.0 * max(best_norms)
                ),
            }
        )

    ready_rows = sorted(
        [r for r in per_episode if r["teacher_ready_rows"] > 0],
        key=lambda r: (-r["teacher_ready_rows"], r["episode_index"]),
    )
    very_near_rows = sorted(
        [r for r in per_episode if r["very_near_rows"] > 0 and r["teacher_ready_rows"] == 0],
        key=lambda r: (-r["very_near_rows"], r["best_max_norm"], r["episode_index"]),
    )
    xy_rows = sorted(
        [r for r in per_episode if r["xy_edge_rows"] > 0],
        key=lambda r: (-r["xy_edge_rows"], r["min_xy_norm"], r["episode_index"]),
    )
    yaw_rows = sorted(
        [r for r in per_episode if r["yaw_edge_rows"] > 0],
        key=lambda r: (-r["yaw_edge_rows"], r["min_yaw_norm"], r["episode_index"]),
    )
    close_rows = sorted(
        [r for r in per_episode if r["close_near_rows"] > 0],
        key=lambda r: (-r["close_near_rows"], r["best_max_norm"], r["episode_index"]),
    )
    broad_block_rows = sorted(
        [r for r in per_episode if r["broad_xy_block_rows"] > 0 or r["broad_yaw_block_rows"] > 0],
        key=lambda r: (
            -(r["broad_xy_block_rows"] + r["broad_yaw_block_rows"]),
            r["best_max_norm"],
            r["episode_index"],
        ),
    )

    selected: set[int] = set()
    selected_order: list[int] = []
    for bucket, quota in (
        (ready_rows, args.quota_ready_anchor),
        (very_near_rows, args.quota_very_near),
        (xy_rows, args.quota_xy_edge),
        (yaw_rows, args.quota_yaw_edge),
        (broad_block_rows, args.quota_broad_block),
        (close_rows, args.quota_close_near),
    ):
        selected_order.extend(_pick_unique(bucket, selected, int(quota)))
    if len(selected_order) < int(args.max_episodes):
        for row in sorted(per_episode, key=lambda r: (-r["score"], r["episode_index"])):
            ep = int(row["episode_index"])
            if ep in selected:
                continue
            selected.add(ep)
            selected_order.append(ep)
            if len(selected_order) >= int(args.max_episodes):
                break
    selected_order = selected_order[: int(args.max_episodes)]

    selected_rows = [r for r in per_episode if int(r["episode_index"]) in set(selected_order)]
    selected_windows = [w for w in windows if int(w["episode_index"]) in set(selected_order)]

    out = {
        "trace_dir": str(trace_dir),
        "selected_episode_indices": selected_order,
        "selected_episode_indices_csv": ",".join(str(e) for e in selected_order),
        "selected_episode_count": len(selected_order),
        "selected_bucket_counts": {
            "teacher_ready_episodes": sum(1 for r in selected_rows if r["teacher_ready_rows"] > 0),
            "very_near_episodes": sum(1 for r in selected_rows if r["very_near_rows"] > 0),
            "xy_edge_episodes": sum(1 for r in selected_rows if r["xy_edge_rows"] > 0),
            "yaw_edge_episodes": sum(1 for r in selected_rows if r["yaw_edge_rows"] > 0),
            "broad_xy_block_episodes": sum(1 for r in selected_rows if r["broad_xy_block_rows"] > 0),
            "broad_yaw_block_episodes": sum(1 for r in selected_rows if r["broad_yaw_block_rows"] > 0),
            "close_near_episodes": sum(1 for r in selected_rows if r["close_near_rows"] > 0),
        },
        "selected_rows": sorted(selected_rows, key=lambda r: selected_order.index(int(r["episode_index"]))),
        "all_rows": sorted(per_episode, key=lambda r: (-r["score"], r["episode_index"])),
        "notes": [
            "Use this plan to recollect targeted current-profile near-ready windows.",
            "A_teacher_ready is a true anchor; V/B/Y buckets are recovery candidates, not proof of ready.",
            "Teacher metrics are labels only; do not feed teacher deltas into runtime student inputs.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2, sort_keys=True))

    output_windows = Path(args.output_windows_jsonl)
    output_windows.parent.mkdir(parents=True, exist_ok=True)
    with output_windows.open("w") as f:
        for row in selected_windows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"wrote plan: {output_json}")
    print(f"wrote windows: {output_windows}")
    print(f"selected episodes: {out['selected_episode_indices_csv']}")
    print(json.dumps(out["selected_bucket_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
