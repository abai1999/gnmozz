"""
Build a runtime-ready shadow episode set from an online teacher-oracle diagnosis
rollout collected under the *current* eval/runtime configuration.

This avoids reusing historical episode indices that may no longer reproduce
teacher-ready / very-near / close-neighborhood windows online.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_float(v):
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:  # NaN
        return None
    return x


def _first_true_step(rows, key: str):
    for r in rows:
        if bool(r.get(key, False)):
            return int(r.get("step", 0))
    return None


def _count_true(rows, key: str) -> int:
    return sum(bool(r.get(key, False)) for r in rows)


def _teacher_metrics(rows):
    xy = []
    z = []
    yaw = []
    for r in rows:
        metrics = (r.get("teacher_truth_handoff_metrics") or {})
        xv = _safe_float(metrics.get("xy_error"))
        zv = _safe_float(metrics.get("abs_z_error"))
        yv = _safe_float(metrics.get("yaw_error"))
        if xv is not None:
            xy.append(xv)
        if zv is not None:
            z.append(zv)
        if yv is not None:
            yaw.append(yv)
    return xy, z, yaw


def _window(rows, center_step: int | None, radius: int):
    if center_step is None:
        return []
    lo = center_step - max(radius, 0)
    hi = center_step + max(radius, 0)
    return [r for r in rows if lo <= int(r.get("step", 0)) <= hi]


def _score_episode(
    rows,
    rel_xy: float,
    rel_z: float,
    rel_yaw: float,
    very_near_factor: float,
    yaw_boundary_lo: float,
    yaw_boundary_hi: float,
):
    teacher_ready_rows = _count_true(rows, "teacher_truth_handoff_ready")
    teacher_metrics_rows = sum(
        1
        for r in rows
        if any(
            _safe_float((r.get("teacher_truth_handoff_metrics") or {}).get(k)) is not None
            for k in ("xy_error", "abs_z_error", "yaw_error")
        )
    )
    first_close_step = _first_true_step(rows, "refiner_alignment_planner_close_intent")
    close_window = _window(rows, first_close_step, 20)
    close_teacher_xy, close_teacher_z, close_teacher_yaw = _teacher_metrics(close_window)
    all_teacher_xy, all_teacher_z, all_teacher_yaw = _teacher_metrics(rows)

    very_near_rows = 0
    ready_support_rows = 0
    yaw_boundary_rows = 0
    for r in rows:
        metrics = (r.get("teacher_truth_handoff_metrics") or {})
        xv = _safe_float(metrics.get("xy_error"))
        zv = _safe_float(metrics.get("abs_z_error"))
        yv = _safe_float(metrics.get("yaw_error"))
        if xv is None or zv is None or yv is None:
            continue
        if xv <= very_near_factor * rel_xy and zv <= very_near_factor * rel_z and yv <= very_near_factor * rel_yaw:
            very_near_rows += 1
        if xv <= rel_xy and zv <= rel_z and yv <= rel_yaw:
            ready_support_rows += 1
        if yaw_boundary_lo * rel_yaw <= yv <= yaw_boundary_hi * rel_yaw:
            yaw_boundary_rows += 1

    close_near_rows = 0
    for xv, zv, yv in zip(close_teacher_xy, close_teacher_z, close_teacher_yaw):
        if xv <= very_near_factor * rel_xy and zv <= very_near_factor * rel_z and yv <= very_near_factor * rel_yaw:
            close_near_rows += 1

    return {
        "teacher_ready_rows": int(teacher_ready_rows),
        "teacher_metric_rows": int(teacher_metrics_rows),
        "very_near_rows": int(very_near_rows),
        "ready_support_rows": int(ready_support_rows),
        "yaw_boundary_rows": int(yaw_boundary_rows),
        "first_close_step": first_close_step,
        "close_near_rows": int(close_near_rows),
        "min_teacher_xy": min(all_teacher_xy) if all_teacher_xy else None,
        "min_teacher_z": min(all_teacher_z) if all_teacher_z else None,
        "min_teacher_yaw": min(all_teacher_yaw) if all_teacher_yaw else None,
        "close_min_teacher_xy": min(close_teacher_xy) if close_teacher_xy else None,
        "close_min_teacher_z": min(close_teacher_z) if close_teacher_z else None,
        "close_min_teacher_yaw": min(close_teacher_yaw) if close_teacher_yaw else None,
        "score": float(
            10.0 * teacher_ready_rows
            + 4.0 * ready_support_rows
            + 2.0 * very_near_rows
            + 1.5 * close_near_rows
            + 0.5 * yaw_boundary_rows
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True, help="Directory containing epXXX_gripper_trace.jsonl")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--max_episodes", type=int, default=12)
    ap.add_argument("--rel_xy", type=float, default=0.007)
    ap.add_argument("--rel_z", type=float, default=0.005)
    ap.add_argument("--rel_yaw", type=float, default=0.12434040009975433)
    ap.add_argument("--very_near_factor", type=float, default=1.5)
    ap.add_argument("--yaw_boundary_lo", type=float, default=0.8)
    ap.add_argument("--yaw_boundary_hi", type=float, default=1.8)
    ap.add_argument("--quota_teacher_ready", type=int, default=6)
    ap.add_argument("--quota_close_near", type=int, default=4)
    ap.add_argument("--quota_yaw_boundary", type=int, default=4)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    episode_rows = []
    for trace_path in sorted(trace_dir.glob("ep*_gripper_trace.jsonl")):
        ep = int(trace_path.name.split("_")[0][2:])
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        score = _score_episode(
            rows,
            rel_xy=float(args.rel_xy),
            rel_z=float(args.rel_z),
            rel_yaw=float(args.rel_yaw),
            very_near_factor=float(args.very_near_factor),
            yaw_boundary_lo=float(args.yaw_boundary_lo),
            yaw_boundary_hi=float(args.yaw_boundary_hi),
        )
        score["episode_index"] = ep
        episode_rows.append(score)

    teacher_ready = sorted(
        [r for r in episode_rows if r["teacher_ready_rows"] > 0],
        key=lambda r: (-r["teacher_ready_rows"], -r["ready_support_rows"], -r["score"], r["episode_index"]),
    )
    close_near = sorted(
        [r for r in episode_rows if r["close_near_rows"] > 0],
        key=lambda r: (-r["close_near_rows"], -r["very_near_rows"], -r["score"], r["episode_index"]),
    )
    yaw_boundary = sorted(
        [r for r in episode_rows if r["yaw_boundary_rows"] > 0],
        key=lambda r: (-r["yaw_boundary_rows"], -r["score"], r["episode_index"]),
    )

    selected = []
    selected_set = set()

    def _pick(rows, quota):
        for r in rows:
            if len(selected) >= int(args.max_episodes):
                return
            if len([x for x in selected if x in selected_set]) > 10**9:
                return
            ep = int(r["episode_index"])
            if ep in selected_set:
                continue
            selected.append(ep)
            selected_set.add(ep)
            if len([x for x in selected if x == ep]) >= quota:
                pass
            if sum(1 for x in selected if x in selected_set) >= int(args.max_episodes):
                return

    _pick(teacher_ready[: int(args.quota_teacher_ready)], int(args.quota_teacher_ready))
    _pick(close_near[: int(args.quota_close_near)], int(args.quota_close_near))
    _pick(yaw_boundary[: int(args.quota_yaw_boundary)], int(args.quota_yaw_boundary))
    for r in sorted(episode_rows, key=lambda x: (-x["score"], x["episode_index"])):
        if len(selected) >= int(args.max_episodes):
            break
        ep = int(r["episode_index"])
        if ep in selected_set:
            continue
        selected.append(ep)
        selected_set.add(ep)

    out = {
        "description": "Online-ready runtime shadow set rebuilt from current teacher-oracle diagnosis traces.",
        "source_trace_dir": str(trace_dir),
        "episode_count": len(episode_rows),
        "selected_episode_indices": selected,
        "selected_episode_indices_csv": ",".join(str(x) for x in selected),
        "teacher_ready_candidates": teacher_ready,
        "close_near_candidates": close_near,
        "yaw_boundary_candidates": yaw_boundary,
        "per_episode": sorted(episode_rows, key=lambda x: x["episode_index"]),
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(args.output_json)


if __name__ == "__main__":
    main()
