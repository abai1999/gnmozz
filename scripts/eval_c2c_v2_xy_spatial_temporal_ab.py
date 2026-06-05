#!/usr/bin/env python3
"""Offline runtime-XY A/B for scalar and spatial-temporal calibrators."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismatic.robot.coarse2contact_v2.runtime_xy_residual import (
    RuntimeXYAffineCalibration,
    RuntimeXYSpatialTemporalCalibration,
    calibrated_runtime_xy_residual_from_trace,
)
from prismatic.robot.coarse2contact_v2.xy_spatial_temporal_generalization import episode_identity, source_eval_root_key


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sequence_key(row: Mapping[str, Any]) -> str:
    sequence = str(row.get("sequence_id", "") or "")
    if sequence:
        return sequence
    trace = str(row.get("trace_path", "") or "")
    if trace:
        return trace
    return f"ep{int(row.get('episode_idx', -1)):03d}"


def _split_name(row: Mapping[str, Any], group_lookup: Mapping[tuple[str, int], str] | None = None) -> str:
    if group_lookup is not None:
        key = episode_identity(row)
        if key in group_lookup:
            return str(group_lookup[key])
    ep = int(row.get("episode_idx", -1))
    source = str(row.get("source_eval_root", row.get("trace_path", "")))
    if ep in {0, 3, 11, 18} and "mp4_smoke_v38_alignment_lifecycle_runtime_xy_mlp_temporal_old4" in source:
        return "old4"
    if ep in {23, 24, 25, 26, 27} and "mp4_smoke_v38_alignment_lifecycle_runtime_xy_mlp_temporal_random5" in source:
        return "random5"
    return "hard_bucket"


def _xy_label(row: Mapping[str, Any]) -> np.ndarray | None:
    label = np.asarray(row.get("label_pre_true_error_t", row.get("grasp_probe_pre_true_error_t", [])), dtype=np.float32).reshape(-1)
    if label.size < 2 or not np.all(np.isfinite(label[:2])):
        return None
    return label[:2].astype(np.float32)


def _bounded_step(xy: np.ndarray, *, xy_gain: float, max_xy_step: float) -> np.ndarray:
    step = float(xy_gain) * np.asarray(xy, dtype=np.float32).reshape(-1)[:2]
    if step.size < 2 or not np.all(np.isfinite(step[:2])):
        return np.zeros((2,), dtype=np.float32)
    norm = float(np.linalg.norm(step[:2]))
    if norm > float(max_xy_step) > 0.0:
        step = step * (float(max_xy_step) / max(norm, 1.0e-9))
    return step[:2].astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1.0e-9 or bn <= 1.0e-9:
        return 0.0
    return float(np.dot(a, b) / max(an * bn, 1.0e-9))


def _load_group_manifest(path: Path | None) -> tuple[dict[tuple[str, int], str], dict[str, list[dict[str, Any]]], list[str]]:
    if path is None or not str(path):
        return {}, {}, []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = payload.get("groups", payload)
    if not isinstance(groups, Mapping):
        raise ValueError(f"invalid group manifest: {path}")
    lookup: dict[tuple[str, int], str] = {}
    ordered_groups: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for group_name, rows in groups.items():
        if not isinstance(rows, list):
            continue
        group_name = str(group_name)
        group_order.append(group_name)
        ordered_groups[group_name] = []
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            root = source_eval_root_key(item)
            ep = int(item.get("episode_idx", -1))
            key = (root, ep)
            if key not in lookup:
                lookup[key] = group_name
            ordered_groups[group_name].append(dict(item))
    return lookup, ordered_groups, group_order


class RuntimeObsCache:
    def __init__(self) -> None:
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def observation(self, row: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
        path = str(row.get("runtime_obs_path", row.get("npz_path", "")) or "")
        if not path:
            return None
        if path not in self._cache:
            with np.load(path, allow_pickle=False) as npz:
                self._cache[path] = {
                    "wrist_rgb": np.asarray(npz["wrist_rgb"], dtype=np.uint8),
                    "wrist_depth": np.asarray(npz["wrist_depth"], dtype=np.float32),
                    "front_rgb": np.asarray(npz["front_rgb"], dtype=np.uint8) if "front_rgb" in npz.files else np.asarray(npz["wrist_rgb"], dtype=np.uint8),
                    "gripper_pose": np.asarray(npz["gripper_pose"], dtype=np.float32),
                    "planner_action_world_6d": np.asarray(npz["planner_action_world_6d"], dtype=np.float32),
                    "proprio": np.asarray(npz["proprio"], dtype=np.float32),
                }
        arrays = self._cache[path]
        step = int(row.get("step_idx", row.get("step", -1)))
        if step < 0 or step >= int(arrays["gripper_pose"].shape[0]):
            return None
        return {
            "wrist_rgb": arrays["wrist_rgb"][step],
            "wrist_depth": arrays["wrist_depth"][step],
            "front_rgb": arrays["front_rgb"][step],
            "gripper_pose": arrays["gripper_pose"][step],
        }

    def robot_state(self, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        path = str(row.get("runtime_obs_path", row.get("npz_path", "")) or "")
        step = int(row.get("step_idx", row.get("step", -1)))
        if path and path in self._cache and step >= 0:
            arrays = self._cache[path]
            if step < int(arrays["planner_action_world_6d"].shape[0]):
                return {
                    "planner_delta_7d": arrays["planner_action_world_6d"][step][:6],
                    "proprio": arrays["proprio"][step],
                }
        return {
            "planner_delta_7d": np.asarray(row.get("planner_prior_world_6d", [0.0] * 6), dtype=np.float32),
            "proprio": np.asarray(row.get("proprio", [0.0] * 15), dtype=np.float32),
        }


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "rows": 0,
            "entry_rows": 0,
            "entry_rate": 0.0,
            "contraction": 0.0,
            "worsen": 0.0,
            "overshoot": 0.0,
            "reverse": 0.0,
            "near_entry": 0.0,
            "mean_step": 0.0,
        }
    rows = len(items)
    entry = [x for x in items if bool(x["entry_ready"])]
    denom = max(len(entry), 1)
    return {
        "rows": int(rows),
        "entry_rows": int(len(entry)),
        "entry_rate": float(len(entry) / max(rows, 1)),
        "contraction": float(np.mean([x["contraction"] for x in entry])) if entry else 0.0,
        "worsen": float(np.mean([x["worsen"] for x in entry])) if entry else 0.0,
        "overshoot": float(np.mean([x["overshoot"] for x in entry])) if entry else 0.0,
        "reverse": float(np.mean([x["reverse"] for x in entry])) if entry else 0.0,
        "near_entry": float(np.mean([x["near_entry"] for x in entry])) if entry else 0.0,
        "mean_step": float(np.mean([x["step_norm"] for x in entry])) if entry else 0.0,
        "ep25_26_worsen": float(np.mean([x["worsen"] for x in entry if x["episode_idx"] in {25, 26}])) if any(x["episode_idx"] in {25, 26} for x in entry) else 0.0,
        "low_visibility_worsen": float(np.mean([x["worsen"] for x in entry if x["low_visibility"]])) if any(x["low_visibility"] for x in entry) else 0.0,
        "partial_worsen": float(np.mean([x["worsen"] for x in entry if x["partial"]])) if any(x["partial"] for x in entry) else 0.0,
        "entry_denominator": int(denom),
    }


def evaluate_model(
    rows: list[dict[str, Any]],
    calibration: Any,
    *,
    xy_gain: float,
    max_xy_step: float,
    group_lookup: Mapping[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    obs_cache = RuntimeObsCache()
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[_sequence_key(row)].append(row)
    records: list[dict[str, Any]] = []
    for _sequence, seq_rows in sorted(by_sequence.items()):
        seq_rows.sort(key=lambda r: int(r.get("step_idx", r.get("step", -1))))
        history: list[dict[str, Any]] = []
        for row in seq_rows:
            if not bool(row.get("grasp_probe_active", False)):
                history.insert(0, row)
                history = history[:5]
                continue
            label = _xy_label(row)
            if label is None:
                history.insert(0, row)
                history = history[:5]
                continue
            observation = None
            robot_state = None
            if isinstance(calibration, RuntimeXYSpatialTemporalCalibration):
                observation = obs_cache.observation(row)
                robot_state = obs_cache.robot_state(row)
            est = calibrated_runtime_xy_residual_from_trace(
                row,
                calibration,
                history_rows=history,
                observation=observation,
                robot_state=robot_state,
            )
            entry = bool(est.entry_ready)
            step = _bounded_step(np.asarray([est.dx, est.dy], dtype=np.float32), xy_gain=xy_gain, max_xy_step=max_xy_step) if entry else np.zeros((2,), dtype=np.float32)
            pre_norm = float(np.linalg.norm(label))
            post_norm = float(np.linalg.norm(label - step))
            step_norm = float(np.linalg.norm(step))
            vis = str(row.get("observability_bucket", "") or "").lower()
            records.append(
                {
                    "split": _split_name(row, group_lookup),
                    "episode_idx": int(row.get("episode_idx", -1)),
                    "entry_ready": entry,
                    "contraction": bool(post_norm < pre_norm),
                    "worsen": bool(post_norm > pre_norm),
                    "overshoot": bool(step_norm > pre_norm),
                    "reverse": bool(_cos(step, label) < 0.0),
                    "near_entry": bool(post_norm <= 0.005),
                    "step_norm": step_norm,
                    "low_visibility": bool(row.get("wrist_is_occluded", False) or row.get("wrist_is_low_visibility", False) or "low" in vis or "occl" in vis),
                    "partial": bool("partial" in vis),
                    "reason": str(est.reason),
                    "source": str(est.source),
                }
            )
            row_with_est = dict(row)
            row_with_est["runtime_xy_estimator"] = est.to_dict()
            history.insert(0, row_with_est)
            history = history[:5]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_split[str(rec["split"])].append(rec)
        by_episode[f"ep{int(rec['episode_idx']):03d}"].append(rec)
        by_group[str(rec["split"])].append(rec)
    payload = {
        "overall": _summarize(records),
        "splits": {k: _summarize(v) for k, v in sorted(by_split.items())},
        "episodes": {k: _summarize(v) for k, v in sorted(by_episode.items())},
    }
    if group_lookup:
        payload["groups"] = {k: _summarize(v) for k, v in sorted(by_group.items())}
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint path, or name=none")
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--output_md", type=Path, required=True)
    ap.add_argument("--xy_gain", type=float, default=0.35)
    ap.add_argument("--max_xy_step", type=float, default=0.003)
    ap.add_argument("--group_manifest_json", type=Path, default=None)
    args = ap.parse_args()

    rows = _read_jsonl(args.dataset_jsonl)
    group_lookup, _, group_order = _load_group_manifest(args.group_manifest_json)
    results: dict[str, Any] = {
        "schema_version": "c2c_v2_runtime_xy_spatial_temporal_ab_v1",
        "dataset_jsonl": str(args.dataset_jsonl),
        "rows": int(len(rows)),
        "xy_gain": float(args.xy_gain),
        "max_xy_step": float(args.max_xy_step),
        "group_manifest_json": str(args.group_manifest_json) if args.group_manifest_json else "",
        "group_manifest_groups": group_order,
        "models": {},
    }
    for spec in args.model:
        if "=" not in str(spec):
            raise ValueError(f"--model must be name=path: {spec}")
        name, path = str(spec).split("=", 1)
        calibration = None if path == "none" else RuntimeXYAffineCalibration.load(path)
        results["models"][name] = evaluate_model(
            rows,
            calibration,
            xy_gain=float(args.xy_gain),
            max_xy_step=float(args.max_xy_step),
            group_lookup=group_lookup if group_lookup else None,
        )
        results["models"][name]["checkpoint"] = str(path)
        if group_order:
            results["models"][name]["groups"] = {
                group_name: results["models"][name]["groups"][group_name]
                for group_name in group_order
                if "groups" in results["models"][name] and group_name in results["models"][name]["groups"]
            }
            group_metrics = results["models"][name]["groups"]
            if group_metrics:
                worst_group = max(group_metrics.items(), key=lambda item: float(item[1]["worsen"]))
                results["models"][name]["worst_group"] = worst_group[0]
                results["models"][name]["worst_group_worsen"] = float(worst_group[1]["worsen"])
                results["models"][name]["worst_group_contraction"] = float(worst_group[1]["contraction"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Runtime XY Spatial-Temporal A/B", ""]
    for name, payload in results["models"].items():
        overall = payload["overall"]
        lines.append(f"## {name}")
        lines.append(
            f"- overall rows `{overall['rows']}`, entry `{overall['entry_rows']}` ({overall['entry_rate']:.3f}), "
            f"contraction `{overall['contraction']:.3f}`, worsen `{overall['worsen']:.3f}`, "
            f"overshoot `{overall['overshoot']:.3f}`, reverse `{overall['reverse']:.3f}`, near_entry `{overall['near_entry']:.3f}`"
        )
        for split, summary in payload["splits"].items():
            lines.append(
                f"- {split}: rows `{summary['rows']}`, entry `{summary['entry_rows']}` ({summary['entry_rate']:.3f}), "
                f"contraction `{summary['contraction']:.3f}`, worsen `{summary['worsen']:.3f}`, "
                f"overshoot `{summary['overshoot']:.3f}`, reverse `{summary['reverse']:.3f}`, "
                f"ep25/26_worsen `{summary['ep25_26_worsen']:.3f}`"
            )
        if "groups" in payload:
            lines.append(f"- worst_group: `{payload.get('worst_group', '')}`")
            for group_name, summary in payload["groups"].items():
                lines.append(
                    f"- group {group_name}: rows `{summary['rows']}`, entry `{summary['entry_rows']}` ({summary['entry_rate']:.3f}), "
                    f"contraction `{summary['contraction']:.3f}`, worsen `{summary['worsen']:.3f}`, "
                    f"overshoot `{summary['overshoot']:.3f}`, reverse `{summary['reverse']:.3f}`"
                )
        lines.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
