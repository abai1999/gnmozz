#!/usr/bin/env python3
"""Audit runtime shadow traces for alignment_v3_direct_local_controller."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _load_rows(trace_dir: Path) -> list[dict]:
    rows: list[dict] = []
    paths = sorted(trace_dir.rglob("*_gripper_trace.jsonl"))
    if not paths:
        paths = sorted(trace_dir.rglob("*.jsonl"))
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_trace_path"] = str(path)
                rows.append(row)
    return rows


def _stats(rows: list[dict], noop_pos_epsilon: float, noop_yaw_epsilon: float) -> dict:
    if not rows:
        return {
            "rows": 0,
            "active_rows": 0,
            "gate_pass_rows": 0,
            "source_hist": {},
        }

    def _mean(key: str, subset: list[dict] | None = None) -> float | None:
        vals = []
        for row in subset or rows:
            v = row.get(key, None)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        return float(np.mean(vals)) if vals else None

    active = [r for r in rows if bool(r.get("refiner_alignment_v3_shadow_active", False))]
    gate_pass = [r for r in rows if bool(r.get("refiner_alignment_v3_shadow_gate_pass", False))]
    applied = [r for r in rows if bool(r.get("refiner_alignment_v3_apply_applied", False))]
    source_hist = Counter(str(r.get("refiner_alignment_v3_shadow_source", "none")) for r in rows)
    block_hist = Counter(str(r.get("refiner_alignment_v3_shadow_block_reason", "none")) for r in rows)
    apply_block_hist = Counter(str(r.get("refiner_alignment_v3_apply_block_reason", "none")) for r in rows)
    gate_block_hist = Counter(str(r.get("refiner_alignment_blocked_reason", "none")) for r in rows)
    per_episode = defaultdict(list)
    for row in rows:
        ep = row.get("episode_index", None)
        if ep is None:
            stem = Path(str(row.get("_trace_path", "episode_unknown.jsonl"))).stem
            ep = stem
        per_episode[str(ep)].append(row)

    def _improve_rate(subset: list[dict], key: str) -> float | None:
        vals = [bool(r.get(key, False)) for r in subset if key in r]
        return float(np.mean(vals)) if vals else None

    def _near_noop_rate(subset: list[dict]) -> float | None:
        vals = []
        for row in subset:
            pos = row.get("refiner_alignment_v3_shadow_pred_pos_norm", None)
            yaw = row.get("refiner_alignment_v3_shadow_pred_yaw_abs", None)
            if pos is None or yaw is None:
                continue
            vals.append(float(pos) <= float(noop_pos_epsilon) and float(yaw) <= float(noop_yaw_epsilon))
        return float(np.mean(vals)) if vals else None

    def _episode_summary(ep_rows: list[dict]) -> dict:
        return {
            "rows": len(ep_rows),
            "active_rows": int(sum(bool(r.get("refiner_alignment_v3_shadow_active", False)) for r in ep_rows)),
            "gate_pass_rows": int(sum(bool(r.get("refiner_alignment_v3_shadow_gate_pass", False)) for r in ep_rows)),
            "apply_rows": int(sum(bool(r.get("refiner_alignment_v3_apply_applied", False)) for r in ep_rows)),
            "alignment_gate_open_rows": int(sum(bool(r.get("refiner_alignment_gate_open", False)) for r in ep_rows)),
            "alignment_window_active_rows": int(sum(bool(r.get("refiner_alignment_window_active", False)) for r in ep_rows)),
            "source_hist": dict(Counter(str(r.get("refiner_alignment_v3_shadow_source", "none")) for r in ep_rows)),
            "block_hist": dict(Counter(str(r.get("refiner_alignment_v3_shadow_block_reason", "none")) for r in ep_rows)),
            "apply_block_hist": dict(
                Counter(str(r.get("refiner_alignment_v3_apply_block_reason", "none")) for r in ep_rows)
            ),
            "alignment_gate_block_hist": dict(Counter(str(r.get("refiner_alignment_blocked_reason", "none")) for r in ep_rows)),
            "planner_close_intent_rate": _mean("refiner_alignment_planner_close_intent", ep_rows),
            "near_target_rate": _mean("refiner_alignment_near_target", ep_rows),
            "support_inner_rate": _mean("refiner_alignment_support_inner_satisfied", ep_rows),
            "outer_rescue_rate": _mean("refiner_alignment_use_outer_rescue", ep_rows),
            "cur_xy_mean": _mean("refiner_alignment_v3_shadow_cur_xy", ep_rows),
            "cur_z_mean": _mean("refiner_alignment_v3_shadow_cur_z", ep_rows),
            "cur_yaw_mean": _mean("refiner_alignment_v3_shadow_cur_yaw", ep_rows),
            "post_xy_mean": _mean("refiner_alignment_v3_shadow_post_xy", ep_rows),
            "post_z_mean": _mean("refiner_alignment_v3_shadow_post_z", ep_rows),
            "post_yaw_mean": _mean("refiner_alignment_v3_shadow_post_yaw", ep_rows),
            "xy_improved_rate": _improve_rate(ep_rows, "refiner_alignment_v3_shadow_xy_improved"),
            "z_improved_rate": _improve_rate(ep_rows, "refiner_alignment_v3_shadow_z_improved"),
            "yaw_improved_rate": _improve_rate(ep_rows, "refiner_alignment_v3_shadow_yaw_improved"),
            "all_improved_rate": _improve_rate(ep_rows, "refiner_alignment_v3_shadow_all_improved"),
            "pred_pos_norm_mean": _mean("refiner_alignment_v3_shadow_pred_pos_norm", ep_rows),
            "pred_yaw_abs_mean": _mean("refiner_alignment_v3_shadow_pred_yaw_abs", ep_rows),
            "apply_pos_norm_mean": _mean("refiner_alignment_v3_apply_pos_norm", ep_rows),
            "apply_yaw_abs_mean": _mean("refiner_alignment_v3_apply_yaw_abs", ep_rows),
            "student_near_noop_rate": _near_noop_rate(ep_rows),
            "risk_logit_mean": _mean("refiner_alignment_v3_shadow_risk_logit", ep_rows),
            "confidence_logit_mean": _mean("refiner_alignment_v3_shadow_confidence_logit", ep_rows),
            "video_paths": sorted({str(Path(r["_trace_path"]).parent.parent / "videos") for r in ep_rows}),
        }

    report = {
        "audit": "alignment_v3_runtime_shadow",
        "rows": len(rows),
        "active_rows": len(active),
        "gate_pass_rows": len(gate_pass),
        "apply_rows": len(applied),
        "active_rate": float(len(active) / max(len(rows), 1)),
        "gate_pass_rate": float(len(gate_pass) / max(len(rows), 1)),
        "apply_rate": float(len(applied) / max(len(rows), 1)),
        "source_hist": dict(source_hist),
        "block_hist": dict(block_hist),
        "apply_block_hist": dict(apply_block_hist),
        "alignment_gate_block_hist": dict(gate_block_hist),
        "alignment_gate_open_rate": _mean("refiner_alignment_gate_open"),
        "alignment_window_active_rate": _mean("refiner_alignment_window_active"),
        "planner_close_intent_rate": _mean("refiner_alignment_planner_close_intent"),
        "near_target_rate": _mean("refiner_alignment_near_target"),
        "support_inner_rate": _mean("refiner_alignment_support_inner_satisfied"),
        "outer_rescue_rate": _mean("refiner_alignment_use_outer_rescue"),
        "cur_xy_mean": _mean("refiner_alignment_v3_shadow_cur_xy", active),
        "cur_z_mean": _mean("refiner_alignment_v3_shadow_cur_z", active),
        "cur_yaw_mean": _mean("refiner_alignment_v3_shadow_cur_yaw", active),
        "post_xy_mean": _mean("refiner_alignment_v3_shadow_post_xy", active),
        "post_z_mean": _mean("refiner_alignment_v3_shadow_post_z", active),
        "post_yaw_mean": _mean("refiner_alignment_v3_shadow_post_yaw", active),
        "xy_improved_rate": _improve_rate(active, "refiner_alignment_v3_shadow_xy_improved"),
        "z_improved_rate": _improve_rate(active, "refiner_alignment_v3_shadow_z_improved"),
        "yaw_improved_rate": _improve_rate(active, "refiner_alignment_v3_shadow_yaw_improved"),
        "all_improved_rate": _improve_rate(active, "refiner_alignment_v3_shadow_all_improved"),
        "pred_pos_norm_mean": _mean("refiner_alignment_v3_shadow_pred_pos_norm", active),
        "pred_yaw_abs_mean": _mean("refiner_alignment_v3_shadow_pred_yaw_abs", active),
        "apply_pos_norm_mean": _mean("refiner_alignment_v3_apply_pos_norm", applied),
        "apply_yaw_abs_mean": _mean("refiner_alignment_v3_apply_yaw_abs", applied),
        "student_near_noop_rate": _near_noop_rate(active),
        "noop_pos_epsilon": float(noop_pos_epsilon),
        "noop_yaw_epsilon": float(noop_yaw_epsilon),
        "risk_logit_mean": _mean("refiner_alignment_v3_shadow_risk_logit", active),
        "confidence_logit_mean": _mean("refiner_alignment_v3_shadow_confidence_logit", active),
        "per_episode": {ep: _episode_summary(ep_rows) for ep, ep_rows in sorted(per_episode.items())},
        "source_paths": sorted({row["_trace_path"] for row in rows}),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--noop_pos_epsilon", type=float, default=1e-4)
    ap.add_argument("--noop_yaw_epsilon", type=float, default=1e-4)
    args = ap.parse_args()

    rows = _load_rows(args.trace_dir)
    report = _stats(rows, noop_pos_epsilon=args.noop_pos_epsilon, noop_yaw_epsilon=args.noop_yaw_epsilon)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
